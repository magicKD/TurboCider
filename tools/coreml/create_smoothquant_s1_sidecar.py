"""Create an unqualified S1 candidate from bounded input and weight statistics.

Pure standard library: no Core ML/NumPy imports, checkpoint reads, model copies,
GPU execution or runtime registration. Weight statistics contain only H-channel
maxima. CPU replay uses deterministic synthetic rows bounded by those maxima:
it checks S1 algebra, not real-weight quality, INT8, FP16 lowering, LoRA or speed.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct

MIB = 1024 * 1024
CAPTURE_RECIPE = "ffn-input-fp16-regions-outliers-v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value, minimum, maximum, name):
    require(type(value) is int and minimum <= value <= maximum, f"invalid {name}")
    return value


def text(value, name):
    require(isinstance(value, str) and 0 < len(value) <= 256 and
            all(ord(c) >= 32 for c in value), f"missing/invalid {name}")
    return value


def number(value, name, nonnegative=False):
    require(type(value) in (int, float) and -1e100 <= value <= 1e100 and math.isfinite(value) and
            (not nonnegative or value >= 0), f"nonfinite/invalid {name}")
    return float(value)


def read_small(path, limit, exact=None):
    # Reject final-component symlinks and nonregular files; all sample paths
    # must be local simple filenames in the same resolved capture directory.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= limit and
                (exact is None or info.st_size == exact), "input file exceeds budget or geometry")
        with os.fdopen(fd, "rb", closefd=False) as source:
            data = source.read(limit + 1)
        require(len(data) == info.st_size, "input changed or exceeded budget while reading")
        return data
    finally:
        os.close(fd)


def load_json(path, limit):
    data = read_small(path, limit)
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(data, object_pairs_hook=unique_object,
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"invalid {token}")))
    require(type(value) is dict, "expected JSON object")
    return value, data


def binding(value):
    require(type(value) is dict, "missing calibration binding")
    require(set(value) == {"request_id", "model_id", "model_fingerprint", "identity_kind", "recipe", "loras",
                           "seed", "width", "height", "total_steps", "reference_size", "reference_count", "execution_route",
                           "runtime_recipe", "configured_backend", "actual_execution"},
            "binding must contain only complete calibration provenance")
    for key in ("request_id", "model_id", "model_fingerprint", "recipe"):
        text(value.get(key), key)
    require(value.get("identity_kind") in ("validated-loader-identity", "canonical-stat-identity", "checkpoint-sha256"),
            "missing explicit model identity kind")
    integer(value.get("seed"), 0, (1 << 64) - 1, "seed")
    integer(value.get("width"), 1, 4096, "width")
    integer(value.get("height"), 1, 4096, "height")
    integer(value.get("total_steps"), 1, 4096, "total steps")
    references = integer(value.get("reference_count"), 0, 8, "reference count")
    integer(value.get("reference_size"), 1 if references else 0, 4096, "reference size")
    text(value.get("execution_route"), "execution route")
    for key in ("runtime_recipe", "configured_backend", "actual_execution"):
        text(value.get(key), key)
    require(type(value.get("loras")) is list and len(value["loras"]) <= 16, "missing LoRA binding (use [] for base)")
    for lora in value["loras"]:
        require(type(lora) is dict and set(lora) == {"id", "fingerprint", "strength"}, "invalid LoRA binding")
        text(lora.get("id"), "LoRA id")
        text(lora.get("fingerprint"), "LoRA fingerprint")
        number(lora.get("strength"), "LoRA strength")
    return value


def vector(values, hidden, name):
    require(type(values) is list and len(values) == hidden, f"invalid {name} channel geometry")
    result = [number(v, name, nonnegative=True) for v in values]
    require(all(v <= 3.4028234663852886e38 for v in result), f"{name} exceeds FP32 statistics range")
    return result


def scales(a, w, alpha):
    number(alpha, "alpha")
    require(0 <= alpha <= 1 and len(a) == len(w), "invalid S1 parameters")
    result = []
    for activation, weight in zip(a, w):
        number(activation, "activation maximum", nonnegative=True)
        number(weight, "weight maximum", nonnegative=True)
        value = 1.0 if activation == 0 or weight == 0 else min(16.0, max(1 / 16,
            max(activation, 1e-8) ** alpha / max(weight, 1e-8) ** (1 - alpha)))
        result.append(struct.unpack("<f", struct.pack("<f", value))[0])
    require(all(math.isfinite(v) and 1 / 16 <= v <= 16 for v in result), "invalid S1 scale")
    return result


def build_sidecar(capture_path, weight_stats_path, alpha=0.5):
    capture_path, weight_stats_path = Path(capture_path), Path(weight_stats_path)
    capture, capture_data = load_json(capture_path, 8 * MIB)
    require(capture.get("schema_version") == 1 and type(capture.get("schema_version")) is int and
            capture.get("artifact_type") == "bounded_ffn_input_calibration" and
            capture.get("capture_recipe") == CAPTURE_RECIPE and capture.get("complete") is True and
            capture.get("performance_sample") is False and capture.get("quantization_qualified") is False,
            "capture must be complete, bounded and explicitly unqualified")
    identity = binding(capture.get("binding"))
    selection = capture.get("selection")
    require(type(selection) is dict, "missing calibration selection")
    selected = []
    for name in ("layers", "steps"):
        values = selection.get(name)
        require(type(values) is list and 1 <= len(values) <= (32 if name == "layers" else 3), "invalid selection")
        parsed = [integer(v, 0, 31 if name == "layers" else 4096, name) for v in values]
        require(len(parsed) == len(set(parsed)), "duplicate selection")
        selected.append(parsed)
    require(all(step < identity["total_steps"] for step in selected[1]), "selected step exceeds request provenance")
    sample_limit = integer(selection.get("rows_per_point"), 1, 64, "sample count")
    budgets = capture.get("budgets")
    require(type(budgets) is dict, "missing capture budgets")
    input_budget = integer(budgets.get("input_bytes"), 1, 8 * MIB, "input budget")
    stats_budget = integer(budgets.get("statistics_bytes"), 1, 2 * MIB, "statistics budget")
    total_budget = integer(budgets.get("total_bytes"), 8 * MIB + 1, 24 * MIB, "total budget")
    points = capture.get("points")
    require(type(points) is list and len(points) == len(selected[0]) * len(selected[1]), "incomplete capture point coverage")
    by_layer, seen = {}, set()
    input_bytes = stats_bytes = 0
    for index, point in enumerate(points):
        require(type(point) is dict, "invalid point")
        layer = integer(point.get("layer"), 0, 31, "layer")
        step = integer(point.get("step"), 0, 4096, "step")
        require(layer in selected[0] and step in selected[1] and (layer, step) not in seen, "unexpected/duplicate point")
        seen.add((layer, step))
        require(point.get("phase") in ("prefill", "denoise") and point.get("dtype") == "float16_le", "invalid phase/dtype")
        rows = integer(point.get("rows"), 1, 32768, "rows")
        hidden = integer(point.get("hidden"), 1, 8192, "hidden")
        require(point.get("rows_observed") == rows and type(point.get("rows_observed")) is int, "partial channel statistics")
        regions = point.get("regions")
        require(type(regions) is list and 1 <= len(regions) <= 8, "missing row regions")
        end, names = 0, set()
        for region in regions:
            require(type(region) is dict, "invalid region")
            name = text(region.get("name"), "region name")
            require(name not in names and region.get("begin") == end and type(region.get("begin")) is int, "overlapping/duplicate regions")
            names.add(name)
            end = integer(region.get("end"), end + 1, rows, "region end")
        require(end == rows, "regions omit source rows")
        indexes = point.get("sample_rows")
        require(type(indexes) is list and len(indexes) == min(rows, sample_limit) and len(set(indexes)) == len(indexes), "invalid sample rows")
        for row in indexes:
            integer(row, 0, rows - 1, "sample row")
        require(all(any(r["begin"] <= row < r["end"] for row in indexes) for r in regions), "sample misses row region")
        require(point.get("sample_file") == f"input-{index}.f16", "sample filename must be capture-owned")
        maximum = vector(point.get("channel_max"), hidden, "activation maximum")
        require(all(v <= 65504 for v in maximum), "activation statistics exceed finite FP16")
        input_bytes += len(indexes) * hidden * 2
        stats_bytes += hidden * 4
        require(input_bytes <= input_budget and stats_bytes <= stats_budget and
                input_bytes + len(capture_data) <= total_budget, "capture exceeds budget")
        if layer not in by_layer:
            by_layer[layer] = {"hidden": hidden, "activation_max": [0.] * hidden, "points": []}
        require(by_layer[layer]["hidden"] == hidden, "layer hidden geometry changes")
        by_layer[layer]["activation_max"] = [max(a, b) for a, b in zip(by_layer[layer]["activation_max"], maximum)]
        by_layer[layer]["points"].append(point)
    weights, weight_data = load_json(weight_stats_path, 2 * MIB)
    require(type(weights.get("schema_version")) is int and weights["schema_version"] == 1 and
            weights.get("artifact_type") == "ffn_weight_channel_max" and
            weights.get("statistics_scope") == "base_gate_up_all_rows", "invalid weight statistics scope")
    weight_binding = binding(weights.get("binding"))
    require(weight_binding == identity, "weight request/model/recipe binding mismatch")
    weight_layers = weights.get("layers")
    require(type(weight_layers) is list and len(weight_layers) == len(by_layer), "incomplete weight statistics")
    weight_map = {}
    for item in weight_layers:
        require(type(item) is dict, "invalid weight layer")
        layer = integer(item.get("layer"), 0, 31, "weight layer")
        require(layer in by_layer and layer not in weight_map and item.get("hidden") == by_layer[layer]["hidden"] and
                type(item.get("hidden")) is int, "weight layer geometry/binding mismatch")
        hidden = item["hidden"]
        maximum = vector(item.get("gate_up_channel_max"), hidden, "weight maximum")
        text(item.get("source_fingerprint"), "weight source fingerprint")
        stats_bytes += hidden * 4
        require(stats_bytes <= stats_budget, "weight statistics exceed combined budget")
        # No model rows are exported. These artificial rows test the algebraic
        # transform with actual input samples, and are labelled accordingly.
        probes = [[peak * (1 if (column + pattern) % 3 else -1) * (1 if pattern == 0 else .25)
                   for column, peak in enumerate(maximum)] for pattern in range(2)]
        weight_map[layer] = (item, maximum, probes)
    require(capture.get("used") == {"input_bytes": input_bytes, "statistics_bytes": stats_bytes}, "capture accounting mismatch")
    require(input_bytes + len(capture_data) + len(weight_data) <= total_budget, "capture output exceeds total budget")

    digest = hashlib.sha256(capture_data + b"\0" + weight_data)
    candidates = []
    max_replay_error, replay_dots = 0., 0
    root = capture_path.parent.resolve()
    for layer in selected[0]:
        info = by_layer[layer]
        item, weight_max, probes = weight_map[layer]
        s1 = scales(info["activation_max"], weight_max, alpha)
        contexts = []
        for point in info["points"]:
            count, hidden = len(point["sample_rows"]), info["hidden"]
            data = read_small(root / point["sample_file"], 8 * MIB, count * hidden * 2)
            digest.update(point["sample_file"].encode() + b"\0" + data)
            values = [v[0] for v in struct.iter_unpack("<e", data)]
            require(all(math.isfinite(v) and abs(v) <= point["channel_max"][i % hidden] * (1 + 1e-7) + 1e-8
                        for i, v in enumerate(values)), "sample nonfinite or exceeds complete channel statistics")
            for row_index in sorted(set((0, count // 3, 2 * count // 3, count - 1))):
                x = values[row_index * hidden:(row_index + 1) * hidden]
                for probe in probes:
                    original = math.fsum(a * w for a, w in zip(x, probe))
                    transformed = math.fsum((a / s) * (w * s) for a, w, s in zip(x, probe, s1))
                    normalizer = max(1., math.fsum(abs(a * w) for a, w in zip(x, probe)))
                    error = abs(original - transformed) / normalizer
                    require(math.isfinite(error) and error <= 1e-12, "CPU S1 reparameterization replay failed")
                    max_replay_error = max(max_replay_error, error)
                    replay_dots += 1
            contexts.append({k: point[k] for k in ("layer", "step", "phase", "rows", "regions")})
        candidates.append({"layer": layer, "hidden": info["hidden"], "s1": s1,
                           "weight_source_fingerprint": item["source_fingerprint"], "contexts": contexts})
    return {"schema_version": 2, "artifact_type": "smoothquant_s1_candidate",
            "experimental_only": True, "complete": True,
            "scope": "full-model-s1" if (sorted(selected[0]) == list(range(32)) and len(selected[1]) == 3 and
                                          sample_limit == 8) else "partial-diagnostic",
            "capture_recipe": CAPTURE_RECIPE, "selection": selection,
            "binding": identity, "calibration_sha256": digest.hexdigest(),
            "recipe": "smoothquant-s1-fp32-v1", "alpha": alpha, "scale_bounds": [1 / 16, 16.],
            "runtime_applied": False, "quantization_qualified": False, "performance_qualified": False,
            "pending": ["runtime S1 integration", "FP16/rotation/INT8 and LoRA verification", "held-out 512 end-to-end validation"],
            "cpu_replay": {"scope": "synthetic weight rows bounded by supplied maxima; unquantized S1 algebra only", "passed": True,
                           "dot_products": replay_dots, "max_normalized_error": max_replay_error},
            "layers": candidates}


def write_sidecar(path, sidecar):
    require(type(sidecar.get("schema_version")) is int and sidecar["schema_version"] == 2 and
            sidecar.get("artifact_type") == "smoothquant_s1_candidate" and sidecar.get("recipe") == "smoothquant-s1-fp32-v1" and
            sidecar.get("experimental_only") is True and sidecar.get("complete") is True and
            sidecar.get("scope") in ("full-model-s1", "partial-diagnostic") and sidecar.get("capture_recipe") == CAPTURE_RECIPE and
            sidecar.get("runtime_applied") is False and
            sidecar.get("quantization_qualified") is False and sidecar.get("performance_qualified") is False,
            "sidecar must be explicitly unqualified and unapplied")
    binding(sidecar.get("binding"))
    number(sidecar.get("alpha"), "alpha")
    require(0 <= sidecar["alpha"] <= 1 and sidecar.get("scale_bounds") == [1 / 16, 16.] and
            isinstance(sidecar.get("calibration_sha256"), str) and len(sidecar["calibration_sha256"]) == 64 and
            all(c in "0123456789abcdef" for c in sidecar["calibration_sha256"]), "invalid sidecar calibration identity")
    selection = sidecar.get("selection")
    require(type(selection) is dict and set(selection) == {"layers", "steps", "rows_per_point"}, "missing sidecar selection")
    for key, count, maximum in (("layers",32,31),("steps",3,4096)):
        require(type(selection[key]) is list and 1 <= len(selection[key]) <= count, "invalid sidecar selection")
        for value in selection[key]:
            integer(value, 0, maximum, key)
        require(len(selection[key]) == len(set(selection[key])), "duplicate sidecar selection")
    integer(selection["rows_per_point"], 1, 64, "sample count")
    require(type(sidecar.get("layers")) is list and len(sidecar["layers"]) == len(selection["layers"]), "missing sidecar layers")
    require(sidecar["scope"] == ("full-model-s1" if (sorted(selection["layers"]) == list(range(32)) and
            len(selection["steps"]) == 3 and selection["rows_per_point"] == 8) else "partial-diagnostic"),
            "sidecar scope does not match layer coverage")
    seen = set()
    for layer in sidecar["layers"]:
        require(type(layer) is dict, "invalid sidecar layer")
        number_layer = integer(layer.get("layer"), 0, 31, "sidecar layer")
        require(number_layer in selection["layers"] and number_layer not in seen, "duplicate/missing sidecar layer")
        seen.add(number_layer)
        hidden = integer(layer.get("hidden"), 1, 8192, "sidecar hidden")
        text(layer.get("weight_source_fingerprint"), "weight source fingerprint")
        require(type(layer.get("s1")) is list and len(layer["s1"]) == hidden and
                all(type(v) in (float, int) and math.isfinite(v) and 1 / 16 <= v <= 16 for v in layer["s1"]), "invalid sidecar scale")
    data = (json.dumps(sidecar, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    require(len(data) <= 4 * MIB, "sidecar exceeds small metadata budget")
    with Path(path).open("xb") as destination:
        destination.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path, help="bounded capture.json")
    parser.add_argument("--weight-stats", required=True, type=Path, help="small selected-layer gate/up channel maxima JSON")
    parser.add_argument("--output", required=True, type=Path, help="new sidecar JSON; existing paths are preserved")
    parser.add_argument("--alpha", type=float, default=.5)
    args = parser.parse_args()
    try:
        sidecar = build_sidecar(args.capture, args.weight_stats, args.alpha)
        write_sidecar(args.output, sidecar)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(str(args.output))


if __name__ == "__main__":
    main()
