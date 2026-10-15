#!/usr/bin/env python3
"""Read owned private-ANE diagnostics; never infer a PMU from successful math.

Only explicit directories are read. No native/API imports, benchmark execution,
global logs, system cache inspection, security changes or upstream fetches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import plistlib
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs"))
from pmu import classify_stats, witness_hashes

UPSTREAM = "20080a945096eb5932490b024120b513ae5427e2"
CAPACITY = "2d3809ba2a3654db71d6790574c9b0247d5b4493"
ZERO_SURFACE_SHA = hashlib.sha256(bytes(4096)).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local(root, relative):
    item = Path(relative)
    path = (root / item).resolve()
    if item.is_absolute() or ".." in item.parts or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("explicit existing non-escaping artifact required")
    return path


def pin(root, commit):
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True)
    if actual != commit or dirty:
        raise ValueError("pinned clean source required; never fetch/reset an existing checkout")


def verify(document):
    if (document.get("schema") != "tc-private-ane-pmu-probe-v1" or document.get("production_qualified") is not False
            or document.get("counter_semantics_qualified") is not False or document.get("formal_performance_score") is not False
            or document.get("source_inputs_unchanged") is not True or document.get("model_source_unchanged") is not True):
        raise ValueError("source-stable unqualified diagnostic required")
    native = document["native"]
    if (native.get("status") != "executed-and-numerically-validated" or native.get("unload_succeeded") is not True
            or native.get("requested_mask") != document.get("requested_mask")
            or native.get("counter_semantics_qualified") is not False):
        raise ValueError("successful numerics and explicit unload required")
    shape = document["shape_nchw"]
    if (not isinstance(shape, list) or len(shape) != 4 or any(type(v) is not int for v in shape)
            or shape[0] != 1 or shape[2] != 1 or shape[1] not in (32, 64, 128)
            or shape[3] not in (32, 64, 128, 256, 512) or type(document["requested_mask"]) is not int
            or type(native["requested_mask"]) is not int or document["requested_mask"] not in (0, 2, 15)):
        raise ValueError("bounded exact NCHW workload required")
    for descriptor in (native["input_descriptor"], native["output_descriptor"]):
        if [descriptor.get(key) for key in ("Batches", "Channels", "Height", "Width")] != shape:
            raise ValueError("physical shape differs")
        if (descriptor.get("Depth") not in (None, 1) or descriptor.get("Offset") not in (None, 0)
                or descriptor.get("PlaneCount") not in (None, shape[1])):
            raise ValueError("unsupported physical offset/depth/planes")
        if (descriptor.get("Type") != "Float16" or any(type(descriptor.get(key)) is not int for key in
                ("RowStride", "PlaneStride", "BatchStride")) or descriptor["RowStride"] < 2 * shape[3]
                or descriptor["PlaneStride"] < descriptor["RowStride"]
                or not shape[1] * descriptor["PlaneStride"] <= descriptor["BatchStride"] <= 8 * 1024**2):
            raise ValueError("explicit FP16 physical pitch required")
    samples = native["samples"]
    if len(samples) != 9 or [row["trial"] for row in samples] != list(range(9)):
        raise ValueError("all nine ordered calls required")
    states, witnessed = [], 0
    kind = document.get("recipe_kind", "affine")
    for i, row in enumerate(samples):
        if (type(row["trial"]) is not int or type(row.get("input_generation")) is not int
                or row.get("input_generation") != i or type(row["warmup"]) is not bool
                or row["warmup"] != (i < 3) or type(row["api_wall_ns"]) is not int or row["api_wall_ns"] <= 0
                or type(row["host_prepare_call_readback_ns"]) is not int or row["host_prepare_call_readback_ns"] < row["api_wall_ns"]):
            raise ValueError("typed warmup/generation and complete positive host spans required")
        expected = witness_hashes(kind, shape[1], shape[3], i)
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("full logical tensor hash differs from independent Python math.fsum/FP16 witness")
        error = row.get("maximum_abs_error")
        if (row.get("all_outputs_finite") is not True or row.get("numeric_pass") is not True
                or type(error) not in (int, float) or not 0 <= error <= (0 if kind == "affine" else .001)):
            raise ValueError("full independent numeric gate failed")
        states.append(classify_stats(row["stats"])["state"])
        if "stats_surface" in row:
            surface = row["stats_surface"]
            if (surface.get("total_bytes") != 4096 or type(surface.get("nonzero_bytes")) is not int
                    or not 0 <= surface["nonzero_bytes"] <= 4096 or surface.get("opaque_bytes_exported") is not False):
                raise ValueError("bounded opaque surface witness required")
            if surface["nonzero_bytes"] == 0 and surface.get("sha256") != ZERO_SURFACE_SHA:
                raise ValueError("zero-surface hash differs")
            witnessed += 1
    expected_capture = ("not-requested" if document["requested_mask"] == 0 else
                        "raw-only-unqualified" if all(s == "raw-available-unqualified" for s in states) else "unavailable-for-this-run")
    if "counter_capture_status" in document and document["counter_capture_status"] != expected_capture:
        raise ValueError("capture promotion differs from actual payloads")
    return {"calls_independently_witnessed": 9, "capture_state": expected_capture,
            "payload_state_counts": {state: states.count(state) for state in sorted(set(states))},
            "stats_surface_observed_calls": witnessed, "physical_duration_seconds": None,
            "counter_rates": None, "production_qualified": False, "counter_semantics_qualified": False}


def curate(root, upstream, capacity, selected):
    pin(upstream, UPSTREAM); pin(capacity, CAPACITY)
    result = {"schema": "tc-ane-pmu-campaign-evidence-v1", "curator_sha256": sha(Path(__file__)),
        "contract_helper_sha256": sha(ROOT / "labs/pmu.py"), "production_qualified": False,
        "counter_semantics_qualified": False, "upstream_commit": UPSTREAM, "capacity_commit": CAPACITY,
        "raw_opaque_buffers_exported": False, "runs": [], "upstream_cli_controls": []}
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate selected run")
    for name in selected:
        path = local(root, name + "/report.json")
        value = json.loads(path.read_text())
        supervisor = local(root, name + "-launch/receipt.json")
        receipt = json.loads(supervisor.read_text())
        if receipt.get("owned_process_exit_confirmed") is not True or receipt.get("returncode") not in (0, 3):
            raise ValueError("owned child terminal receipt required")
        validation = verify(value)
        for filename, expected in value["source_sha256"].items():
            if sha(local(path.parent / "source", filename)) != expected:
                raise ValueError("probe source snapshot differs")
        if sha(local(path.parent, "model/model.mil")) != value["model_mil_sha256"]:
            raise ValueError("model changed")
        if "weights_blob_sha256" in value and sha(local(path.parent, "model/weights.bin")) != value["weights_blob_sha256"]:
            raise ValueError("synthetic weights changed")
        if sha(local(path.parent, "pmu-probe")) != value["binary_sha256"]:
            raise ValueError("binary changed")
        result["runs"].append({"id": name, "raw_report_sha256": sha(path), "supervisor_sha256": sha(supervisor),
            "child_returncode": receipt["returncode"], "child_exit_confirmed": True,
            "independent_validation": validation, "report": value})
    # Upstream status text and exit codes are recorded, never accepted as telemetry.
    for name, binary in (("upstream-help", "upstream-profiler"), ("upstream-missing", "upstream-profiler"),
                         ("upstream-owned-conv", "upstream-profiler"), ("upstream-debug-conv", "upstream-profiler-debug"),
                         ("capacity-owned", "capacity-profiler")):
        folder = root / (name + "-launch")
        receipt = json.loads(local(folder, "receipt.json").read_text())
        stdout, stderr = local(folder, "stdout.log"), local(folder, "stderr.log")
        out, err = stdout.read_text(), stderr.read_text()
        result["upstream_cli_controls"].append({"id": name, "returncode": receipt["returncode"],
            "child_exit_confirmed": receipt.get("owned_process_exit_confirmed"), "binary_sha256": sha(local(root, binary)),
            "stdout_sha256": sha(stdout), "stderr_sha256": sha(stderr), "supervisor_sha256": sha(local(folder, "receipt.json")),
            "claimed_pmu_active": "SILICON PMU STATUS: ACTIVE" in out,
            "model_missing_reported": "Model file not found" in out,
            "no_stats_object_reported": "No live performance stats object" in err,
            "pmu_branch_entered_marker": "[Silicon PMU Profiler]" in out,
            "numeric_qualified": False, "counter_qualified": False})
    signature = root / "debug-signature-xml-launch"
    combined = local(signature, "stdout.log").read_bytes() + local(signature, "stderr.log").read_bytes()
    start, end = combined.index(b"<?xml"), combined.index(b"</plist>") + len(b"</plist>")
    result["debug_signature_entitlements_readonly"] = plistlib.loads(combined[start:end])
    package = root / "capacity-owned/packages/conv_fp16.mpsgraphpackage"
    result["capacity_export"] = {"ane_bundle_present": (package / "ane_bundle").exists(),
        "files": [{"path": str(p.relative_to(package)), "sha256": sha(p)} for p in sorted(package.rglob("*")) if p.is_file()]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--capacity-root", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("fresh curated output required")
    value = curate(args.root.resolve(), args.upstream_root.resolve(), args.capacity_root.resolve(), args.run)
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if "/Users/" in encoded or "/Volumes/" in encoded:
        parser.error("machine-local paths must stay private")
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    print(json.dumps({"device_runs": len(value["runs"]), "counter_semantics_qualified": False}))


if __name__ == "__main__":
    main()
