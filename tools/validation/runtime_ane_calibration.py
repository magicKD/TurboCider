"""Verify native channel-auto evidence; never promote FFN evidence to E2E."""
import argparse
import math
import statistics


SCOPE = "complete FFN host spans; not E2E qualification or physical engine trace"


def channel_policy(value):
    if value == "auto":
        return value
    if not value.isascii() or not value.isdigit() or len(value) > 5:
        raise argparse.ArgumentTypeError("channels require auto or a nonnegative integer")
    return int(value)


def _integer(mapping, name, *, minimum=0):
    value = mapping.get(name)
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid channel calibration {name}")
    return value


def _number(value, name, *, positive=True):
    if type(value) not in (int, float) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"invalid channel calibration {name}")
    return value


def _vector(values, count, name):
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"missing channel calibration raw {name}")
    return [_number(value, name) for value in values]


def _layer(raw, repeats, name):
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"missing channel calibration one/four {name}")
    one, four = (_vector(v, repeats, name) for v in raw)
    layer = (statistics.median(four) - statistics.median(one)) / 3
    return _number(layer, name)


def _equal(actual, expected, name):
    _number(actual, name)
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"channel calibration {name} disagrees with raw samples")


def validate_channel_calibration(report, full_width, hidden):
    if not isinstance(report, dict) or report.get("schema_version") != 1 or type(report.get("schema_version")) is not int:
        raise ValueError("missing/version-invalid native channel calibration receipt")
    if report.get("enabled") is not True or report.get("scope") != SCOPE:
        raise ValueError("channel calibration activation/timing scope missing")
    for name in ("complete", "cache_hit", "trial_passed"):
        if type(report.get(name)) is not bool:
            raise ValueError(f"invalid channel calibration {name}")
    lora=report.get("lora",False)
    if type(lora) is not bool:
        raise ValueError("invalid channel calibration adapter marker")
    if report.get("status") not in ("accepted", "gpu_only", "rejected", "unsupported"):
        raise ValueError("invalid channel calibration final status")
    if not isinstance(report.get("reason"), str) or not report["reason"]:
        raise ValueError("channel calibration decision reason missing")
    rows = _integer(report, "actual_rows", minimum=1)
    bucket = _integer(report, "bucket_rows", minimum=1)
    if rows > 4224 or bucket > 4224 or _integer(report,"hidden",minimum=1) != hidden or _integer(report,"width",minimum=1) != full_width:
        raise ValueError("channel calibration/model geometry mismatch")
    layers = _integer(report, "layer_count", minimum=5)
    repeats = _integer(report, "repeats", minimum=3)
    warmups = _integer(report, "warmups", minimum=1)
    if repeats > 31 or not repeats % 2 or warmups > 8:
        raise ValueError("invalid bounded channel calibration sampling")
    selected = _integer(report, "selected_channels")
    proposed = _integer(report, "proposed_channels")
    if any(c and (c % 512 or c >= full_width) for c in (selected, proposed)):
        raise ValueError("invalid channel calibration width/alignment")
    accepted = report["status"] == "accepted"
    if accepted != report["trial_passed"] or bool(selected) != accepted or (not accepted and report["cache_hit"]):
        raise ValueError("channel calibration trial/adoption/cache flags disagree")

    identity = report.get("identity")
    if identity is not None:
        if not isinstance(identity, dict):
            raise ValueError("invalid channel calibration identity")
        sha = identity.get("model_sha256")
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("invalid channel calibration source digest")
        for name in ("encoding", "precision", "backend", "recipe", "soc", "os_build", "runtime_build",
                     "metal_abi", "graph_abi", "source_generation"):
            if not isinstance(identity.get(name), str) or not identity[name]:
                raise ValueError(f"missing channel calibration identity {name}")
        if (identity.get("backend") != "private_ane" or identity.get("precision") not in ("bf16", "fp16") or
                identity.get("recipe") != "sylvester-dh-b128-b512-rne-norm-f16-v2-"+("lora" if lora else "base") or
                _integer(identity,"rows",minimum=1) != rows or _integer(identity,"hidden",minimum=1) != hidden or
                _integer(identity,"width",minimum=1) != full_width or
                type(identity.get("prefetch")) is not bool or type(identity.get("adapter")) is not str):
            raise ValueError("channel calibration identity/geometry/recipe mismatch")
        if bool(identity["adapter"]) != lora:
            raise ValueError("channel calibration bound adapter/recipe mismatch")
        for name in ("tile_k", "tile_n"):
            _integer(identity, name, minimum=1)
    if accepted and (identity is None or rows > bucket or proposed != selected or not report["complete"]):
        raise ValueError("adopted channel calibration lacks complete matching identity/evidence")

    baseline = report.get("baseline")
    if baseline is not None:
        if not isinstance(baseline, dict):
            raise ValueError("invalid channel calibration GPU baseline")
        _equal(baseline.get("layer_seconds"), _layer(baseline.get("raw_seconds"), repeats, "baseline"), "baseline")
    points = report.get("points")
    if not isinstance(points, list) or len(points) > 2:
        raise ValueError("invalid channel calibration point count")
    if points and identity is None:
        raise ValueError("channel calibration samples lack their identity")
    if report["complete"] or accepted:
        if baseline is None or len(points) != 2 or identity is None:
            raise ValueError("complete channel calibration lacks both independent points/baseline")
        expected_depths = [0, (layers-1)//4, (layers-1)//2, 3*(layers-1)//4, layers-1]
        if report.get("sampled_depths") != expected_depths:
            raise ValueError("channel calibration did not sample the declared model depths")
    shares = []
    for point in points:
        if not isinstance(point, dict):
            raise ValueError("invalid channel calibration point")
        share = _number(point.get("share"), "share")
        if share >= 1 or type(point.get("prefetch")) is not bool or point["prefetch"] != identity["prefetch"]:
            raise ValueError("channel calibration share/prefetch mismatch")
        if _integer(point, "ane_calls") != 2 * (warmups + repeats) * 5:
            raise ValueError("channel calibration independent ANE call count mismatch")
        if lora and (_integer(point,"correction_computations") != 2*(warmups+repeats)*5 or
                     _integer(point,"correction_uploads") != 4*(warmups+repeats)*5):
            raise ValueError("LoRA calibration omitted actual correction compute/upload traffic")
        raw = point.get("raw_seconds")
        if not isinstance(raw, list) or len(raw) != 3:
            raise ValueError("channel calibration lacks GPU/ANE/Both raw samples")
        for part, name in enumerate(("gpu", "ane", "both")):
            _equal(point.get(name), _layer(raw[part], repeats, name), name)
        shares.append(share)
    if len(shares) == 2 and not shares[0] < shares[1]:
        raise ValueError("channel calibration shares are not independent ordered points")
    if accepted and not shares[0]-1e-12 <= selected/full_width <= shares[1]+1e-12:
        raise ValueError("channel calibration extrapolated outside measured shares")

    trial = report.get("trial")
    if trial is not None:
        if not isinstance(trial, dict) or type(trial.get("completed")) is not bool or type(trial.get("accepted")) is not bool:
            raise ValueError("invalid channel calibration runtime trial")
        for name in ("calls", "fallbacks", "retries"):
            _integer(trial, name)
    if accepted:
        if trial is None or trial.get("completed") is not True or trial.get("accepted") is not True:
            raise ValueError("channel calibration adopted proposal without actual runtime trial")
        gpu = _vector(trial.get("gpu_seconds"), repeats, "trial GPU")
        candidate = _vector(trial.get("candidate_seconds"), repeats, "trial candidate")
        if (trial["calls"] != 4 * (warmups + repeats) or trial["fallbacks"] or trial["retries"] or
                statistics.median(candidate) > .95 * statistics.median(gpu)):
            raise ValueError("channel calibration actual runtime call/gain gate failed")
        error = _number(trial.get("relative_l2"), "trial relative L2", positive=False)
        cosine = _number(trial.get("cosine"), "trial cosine")
        if error > .03 or not .999 <= cosine <= 1.000001:
            raise ValueError("channel calibration independent FFN numerical gate failed")
        predicted = _number(report.get("predicted_layer_seconds"), "predicted layer")
        # Selection gates the best fitted point, then allows a 1% near-best
        # smaller share. The actual selected runtime window still earns 5%.
        if predicted > 1.01 * .95 * baseline["layer_seconds"]:
            raise ValueError("channel calibration predicted gain gate failed")
    return selected
