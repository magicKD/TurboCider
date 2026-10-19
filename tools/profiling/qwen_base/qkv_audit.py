#!/usr/bin/env python3
"""CPU-only raw-array/source/actual-selector audit of the base QKV screen."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import tarfile
import warnings

import numpy as np

import qkv_collect as screen
from collect import ROOT, model_snapshot, sha256_file, write_json
from vpipe_audit import trace


def arrays(path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= size <= 16 << 20:
            raise ValueError("bounded tensor header required")
        header = json.loads(stream.read(size))
    result = {}
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if value["dtype"] != "BF16":
            raise ValueError("original BF16 source/control arrays required")
        begin, end = value["data_offsets"]
        if end - begin != math.prod(value["shape"]) * 2 or 8 + size + end > path.stat().st_size:
            raise ValueError("complete tensor extents required")
        bits = np.memmap(path, mode="r", dtype="<u2", offset=8 + size + begin, shape=tuple(value["shape"]))
        result[name] = (bits.astype(np.uint32) << 16).view(np.float32)
    return result


def quality(reference, value):
    if reference.shape != value.shape or not np.isfinite(reference).all() or not np.isfinite(value).all():
        raise ValueError("same-shape finite full arrays required")
    x, y = reference.astype(np.float64).reshape(-1), value.astype(np.float64).reshape(-1)
    delta = x - y
    energy, other = float(np.dot(x, x)), float(np.dot(y, y))
    return dict(relative_l2=math.sqrt(float(np.dot(delta, delta)) / max(energy, 1e-30)),
                max_abs=float(np.max(np.abs(delta))), changed=int(np.count_nonzero(x != y)), elements=x.size,
                cosine=float(np.dot(x, y)) / math.sqrt(max(energy * other, 1e-60)))


def actual_weight_hashes(model):
    path = model / "diffusion_models/qwen_image_2.1_bf16.safetensors"
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= size <= 16 << 20:
            raise ValueError("bounded original checkpoint header required")
        header = json.loads(stream.read(size))
        result = {}
        for block in (0, 1):
            for part, leaf in enumerate(("q", "k", "v")):
                value = header[f"transformer_blocks.{block}.attn.to_{leaf}.weight"]
                if value["dtype"] != "BF16" or value["shape"] != [4096, 4096]:
                    raise ValueError("actual base dense BF16 QKV source required")
                begin, end = value["data_offsets"]
                stream.seek(8 + size + begin)
                payload = stream.read(end - begin)
                if len(payload) != 4096 * 4096 * 2:
                    raise ValueError("original QKV source truncated")
                result[f"block{block}-{part}"] = hashlib.sha256(payload).hexdigest()
        return result


def cpu_projection_oracle(model, normalized, raw_reference):
    """All1024 rows, original layer0 bytes, independent CPU F32 dot + BF16 RNE."""
    path = model / "diffusion_models/qwen_image_2.1_bf16.safetensors"
    results = []
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
        for part, leaf in enumerate(("q", "k", "v")):
            info = header[f"transformer_blocks.0.attn.to_{leaf}.weight"]
            begin, end = info["data_offsets"]
            stream.seek(8 + size + begin)
            weight = (np.frombuffer(stream.read(end - begin), dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(4096, 4096)
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always", RuntimeWarning)
                product = normalized.reshape(1024, 4096) @ weight.T
            if not np.isfinite(product).all():
                raise ValueError("nonfinite independent CPU projection")
            bits = product.view(np.uint32)
            rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
            expected = (rounded.astype(np.uint32) << 16).view(np.float32)
            q = quality(expected, raw_reference[:, :, part * 4096:(part + 1) * 4096].reshape(1024, 4096))
            if q["relative_l2"] >= .002:
                raise ValueError("complete GPU projection reference disagrees with original-weight CPU oracle")
            results.append(dict(projection=leaf, **q, cpu_product_max_abs=float(np.max(np.abs(product))),
                                numpy_blas_warnings=[str(w.message) for w in observed]))
    return dict(scope="all1024 rows, CPU F32 reduction then BF16 RNE; GPU reduction can differ. NumPy/BLAS warnings retained; finite full outputs/tolerance checked, warning cause not established", projections=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", action="append", type=Path, required=True)
    parser.add_argument("--failed", action="append", type=Path, default=[])
    parser.add_argument("--model", type=Path, default=ROOT / "models/Comfy-Org-Qwen-Image-2.1")
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--probe-library", type=Path, required=True)
    parser.add_argument("--native-source-archive", type=Path, required=True)
    parser.add_argument("--tool-source-archive", action="append", type=Path, default=[])
    parser.add_argument("--cpu-oracle", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh audit/derived-trace directory required")
    manifest = json.loads((args.native_library.parent / "runtime-build/runtime-build-manifest.json").read_text())
    native_sha = sha256_file(args.native_library)
    probe_sha = sha256_file(args.probe_library)
    with tarfile.open(args.native_source_archive, "r:gz") as archive:
        members = {m.name: m for m in archive.getmembers()}
        if set(members) != {e["path"] for e in manifest["inputs"]["sources"]}:
            raise ValueError("full native archive required")
        for e in manifest["inputs"]["sources"]:
            if hashlib.sha256(archive.extractfile(members[e["path"]]).read()).hexdigest() != e["sha256"] or sha256_file(ROOT / e["path"]) != e["sha256"]:
                raise ValueError("native current/archive identity differs")
    historical = {}
    for path in args.tool_source_archive:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():
                    historical.setdefault(member.name, set()).add(archive.extractfile(member).read())
    source_snapshot, weights = model_snapshot(args.model), actual_weight_hashes(args.model)
    observations, derived, cpu_cache = [], [], {}
    for root in args.cohort:
        summary = json.loads((root / "summary.json").read_text())
        report = json.loads((root / "trial/report.json").read_text())
        if (summary["status"] != "complete" or summary["assets"]["native_library"] != native_sha or summary["assets"]["library"] != probe_sha or
                summary["model_snapshot"] != source_snapshot or report["loaded_runtime_id"] != manifest["runtime_build_id"]):
            raise ValueError("complete unchanged native/model component required")
        snapshot = json.loads((root / "source.json").read_text())
        if set(snapshot) != set(summary["source_sha256"]):
            raise ValueError("complete profiler source snapshot required")
        if (report["cpp_sha256"] != summary["source_sha256"]["tools/profiling/qwen_base/qkv_probe.mm"] or
                report["header_sha256"] != summary["source_sha256"]["tools/profiling/qwen_base/qkv_ane.hpp"]):
            raise ValueError("compiled probe source differs from bound snapshot")
        for name, digest in summary["source_sha256"].items():
            raw = snapshot[name].encode()
            if hashlib.sha256(raw).hexdigest() != digest or not (raw == (ROOT / name).read_bytes() or raw in historical.get(name, set())):
                raise ValueError("profiler current/archive identity differs")
        if report["actual_qkv_weights_sha256"] != weights:
            raise ValueError("captured QKV bytes are not the original two base layers")
        if list((root / "trial").glob("*.png")) or report["actual_source_request"]["operation"] != "image." + summary["case"]:
            raise ValueError("bounded source operation/image-export contract differs")
        if report["mil_sha256"] != sha256_file(root / "trial/program.mil"):
            raise ValueError("compiled MIL source differs")
        if summary["arrays_sha256"] != sha256_file(root / "trial/arrays.safetensors"):
            raise ValueError("raw full-array artifact changed")
        if screen.validate(report, summary["repeats"]) != summary["analysis"]:
            raise ValueError("rotated complete-block denominators differ")
        for key in ("report", "evidence"):
            if sha256_file(root / summary["memory"][key]) != summary["memory"][key + "_sha256"]:
                raise ValueError("memory evidence digest differs")
        if not summary["memory"]["verified"]["complete"] or summary["memory"]["verified"]["command_exit_code"] != 0:
            raise ValueError("complete successful process-tree window required")
        values = arrays(root / "trial/arrays.safetensors")
        pairs = (("raw_quality", "raw_reference", "raw_private"), ("complete_block_quality", "block_reference", "block_private"),
                 ("staged_gpu_quality", "block_reference", "block_staged"),
                 ("alternate_activation_quality", "alternate_raw_reference", "alternate_raw_private"),
                 ("alternate_weight_quality", "other_weight_reference", "other_weight_private"))
        qualities = {}
        for name, ref, candidate in pairs:
            q = quality(values[ref], values[candidate])
            for key in q:
                if not math.isclose(q[key], report[name][key], rel_tol=1e-8, abs_tol=1e-10):
                    raise ValueError("independent full-array quality differs")
            qualities[name] = q
        qualities["last_private_prestaged"] = quality(values["last_gpu_original"], values["last_private_prestaged"])
        qualities["last_private_restaged"] = quality(values["last_gpu_original"], values["last_private_restaged"])
        observation = dict(cohort=root.name, case=summary["case"], chunk=summary["chunk"], layout=summary["layout"], mode=summary["mode"],
                           analysis=summary["analysis"], quality=qualities, slot_bytes=report["slot_bytes"],
                           source_qkv_weights_verified=True, summary_sha256=sha256_file(root / "summary.json"),
                           memory=summary["memory"]["verified"], load_error=summary.get("load_error"))
        if args.cpu_oracle:
            key = hashlib.sha256(values["normalized"].tobytes()).hexdigest()
            if key not in cpu_cache:
                cpu_cache[key] = cpu_projection_oracle(args.model, values["normalized"], values["raw_reference"])
            observation["cpu_projection_oracle"] = cpu_cache[key]
        if summary["mode"] == "timeline":
            observer = json.loads((root / "trial/observer.json").read_text())
            if observer["source_sha256"] != summary["source_sha256"]["tools/profiling/qwen_base/observer.mm"]:
                raise ValueError("actual observer source differs")
            observation["observer"] = screen.observer_contract(observer, report["predictions"])
            exported = trace(observer)
            for event in exported["traceEvents"]:
                index = event["args"]["phase_index"] - 2
                if 0 <= index < len(report["samples"]):
                    sample = report["samples"][index]
                    event["args"].update(arm=sample["arm"], rep=sample["rep"], warmup=sample["warmup"],
                                         scope="component sample; not complete image request")
            derived.append((root.name, exported))
        observations.append(observation)
    failures = []
    for root in args.failed:
        summary = json.loads((root / "summary.json").read_text())
        report = json.loads((root / "trial/report.json").read_text())
        if summary["status"] != "failed":
            raise ValueError("preserved failure required")
        failures.append(dict(cohort=root.name, report_status=report["status"], error=report.get("error", summary.get("error")),
                             accepted_performance_samples=0, summary_sha256=sha256_file(root / "summary.json")))
    args.output.mkdir(parents=True)
    for name, value in derived:
        write_json(args.output / (name + "-trace.json"), value)
    result = dict(schema="tc-base-private-qkv-audit-v1", status="complete", qualification_passed=False,
                  native_sources_verified=len(manifest["inputs"]["sources"]), native_library_sha256=native_sha,
                  probe_library_sha256=probe_sha,
                  native_source_archive_sha256=sha256_file(args.native_source_archive), observations=observations, preserved_failures=failures,
                  script_sha256=sha256_file(Path(__file__)),
                  scope="CPU original six QKV payloads/full controls/actual selectors recheck; block0 component, not full-image or joint-FFN qualification")
    write_json(args.output / "audit.json", result)
    print(json.dumps(dict(status="complete", cohorts=len(observations), failures=len(failures), output=str(args.output))))


if __name__ == "__main__":
    main()
