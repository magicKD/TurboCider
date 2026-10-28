#!/usr/bin/env python3
"""Deferred, isolated SeedVR2 schedule screen after a terminal speed cohort.

This is component-only evidence. It never enables public capabilities, builds
the stable runtime, downloads weights, touches a user's server, or promotes
experimental source. One GPU job runs at a time. Failed raw runs are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

# The native restoration probe accepts explicit CSV indices, not the
# exporter's range syntax. Keep the two command-line ABIs separate.
ALL_BLOCKS_CSV = ",".join(map(str, range(32)))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def terminal_cohort(path, library_sha):
    if not path.is_file():
        return False
    report = json.loads(path.read_text())
    if report.get("status") in ("failed", "cancelled"):
        raise ValueError("preceding speed cohort failed; do not start a hardware screen")
    if report.get("status") != "complete":
        return False
    order = report.get("order")
    cohorts = report.get("cohorts", [])
    if (not report.get("speed_gate_passed") or report.get("library_sha256") != library_sha or
            order != ["native-gpu", "comfy-sdpa-gpu", "comfy-sdpa-gpu", "native-gpu"] or
            len(cohorts) != 4 or [row.get("route") for row in cohorts] != order or
            [row.get("index") for row in cohorts] != list(range(4))):
        raise ValueError("preceding cohort is not an identity-matched complete ABBA gate")
    for row in cohorts:
        child = json.loads(Path(row["report"]).read_text())
        if child.get("status") != "complete" or len(child.get("runs", [])) < 4:
            raise ValueError("preceding cohort has incomplete child requests")
        if row["route"].startswith("comfy-") and child.get("server_terminal") is not True:
            raise ValueError("preceding task-owned reference server is not terminal")
        if row["route"] == "native-gpu" and child.get("native_artifact_identity", {}).get("library_sha256") != library_sha:
            raise ValueError("preceding native child has another library identity")
    return True


def run(command, log, environment, cwd):
    print("screen:", log.stem, flush=True)
    with log.open("w") as stream:
        subprocess.run(list(map(str, command)), cwd=cwd, env=environment,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stable-root", "after-cohort", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workload", nargs=3, action="append", metavar=("NAME", "INPUT", "FP32_REFERENCE"), required=True)
    parser.add_argument("--timeout", type=float, default=5400)
    args = parser.parse_args()
    lab = Path(__file__).resolve().parents[2]
    args.stable_root = args.stable_root.resolve(strict=True)
    args.after_cohort = args.after_cohort.absolute()
    args.manifest = args.manifest.resolve(strict=True)
    args.output = args.output.absolute()
    if args.output.exists():
        parser.error("a fresh schedule-screen output is required")
    if not 1 <= args.timeout <= 7200:
        parser.error("deferred wait must be in 1...7200 seconds")
    # Host tensor reader is preflighted without importing MLX/using hardware.
    subprocess.run([sys.executable, "-c", "import numpy; import safetensors.torch"], check=True)
    workload_names = set()
    workloads = []
    for name, raw_input, raw_reference in args.workload:
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in name) or name in workload_names:
            parser.error("unique simple workload names are required")
        workload_names.add(name)
        input_path, reference = Path(raw_input).resolve(strict=True), Path(raw_reference).resolve(strict=True)
        reference_report = json.loads((reference / "report.json").read_text())
        input_sha = sha256(input_path)
        if reference_report.get("dtype") != "fp32" or reference_report.get("input_sha256") != input_sha:
            parser.error("every independent FP32 reference must bind the exact shared input/noise")
        workloads.append(dict(name=name, input=str(input_path), input_sha256=input_sha,
            reference=str(reference), reference_sha256=sha256(reference / "restoration.safetensors")))
    library = args.stable_root / "build/native/libturbocider.dylib"
    library_sha = sha256(library)
    manifest_sha = sha256(args.manifest)
    args.output.mkdir(parents=True)
    report = dict(schema="seedvr2-buffered-schedule-screen-v1", status="waiting",
        preceding_cohort=str(args.after_cohort), stable_library_sha256=library_sha,
        manifest_sha256=manifest_sha, workloads=workloads, micro=[],
        numerical_gates_passed=False, full_request_speed_qualified=False,
        public_capability_enabled=False,
        scope="isolated component screens only; CPU+NE prediction does not prove physical ANE occupancy")
    try:
        start = time.monotonic()
        print("waiting for the preceding terminal ABBA cohort; no hardware screen overlaps it", flush=True)
        while not terminal_cohort(args.after_cohort, library_sha):
            if time.monotonic() - start > args.timeout:
                raise TimeoutError("preceding cohort did not reach an accepted terminal state")
            write_json(args.output / "report.json", report)
            time.sleep(5)
        report["status"] = "building-isolated-probes"
        write_json(args.output / "report.json", report)
        if sha256(library) != library_sha or sha256(args.manifest) != manifest_sha:
            raise ValueError("stable library or artifact bank changed while waiting")
        model = args.stable_root / "models/Comfy-Org-SeedVR2"
        provenance = json.loads((model / "source.json").read_text())
        for source in provenance["files"]:
            path = model / source["path"]
            if path.stat().st_size != source["bytes"] or sha256(path) != source["sha256"]:
                raise ValueError("original model provenance failed before the screen")
        report["model_provenance"] = provenance
        environment = dict(os.environ)
        # Do not leak unrelated recipe experiments into the new screen.
        for key in list(environment):
            if key.startswith("TURBOCIDER_SEEDVR2_"):
                del environment[key]
        environment.update(MLX_ROOT=str(args.stable_root / ".venv/lib/python3.11/site-packages/mlx"),
            TURBOCIDER_SEEDVR2_NATIVE_LIB_DIR=str(library.parent),
            TURBOCIDER_SEEDVR2_HYBRID_SCHEDULE="buffered_suffix")
        run(["bash", "tools/native/build_seedvr2_hybrid_probes.sh"], args.output / "build.log", environment, lab)
        probe = lab / "build/seedvr2/seedvr2-hybrid-probe"
        restore = lab / "build/seedvr2/seedvr2-hybrid-restore-probe"
        report["probe_sha256"] = sha256(probe)
        report["restore_probe_sha256"] = sha256(restore)
        report["status"] = "running"
        for tokens in (512, 1088, 4096):
            output = args.output / f"micro-{tokens}"
            run([probe, model, args.manifest, output, tokens],
                args.output / f"micro-{tokens}.log", environment, lab)
            result = json.loads((output / "report.json").read_text())
            if not result.get("parity_passed") or result.get("prefix_snapshot_copied_bytes") != 0:
                raise ValueError("buffered FFN parity/copy guard failed")
            report["micro"].append(result)
            write_json(args.output / "report.json", report)
        oracle = args.stable_root / "tools/validation/seedvr2_oracle.py"
        for workload in workloads:
            name = workload["name"]
            # Four image repeats give cold + warm screens. Whole t25 videos
            # initially run once per route to gate every frame before tuning.
            repeats = 1 if "video" in name else 4
            if sha256(Path(workload["input"])) != workload["input_sha256"]:
                raise ValueError("shared-input fixture changed")
            gpu, candidate = args.output / (name + "-gpu"), args.output / (name + "-buffered")
            run([restore, model, args.manifest, workload["input"], gpu, "gpu", repeats],
                args.output / (name + "-gpu.log"), environment, lab)
            if sha256(Path(workload["input"])) != workload["input_sha256"]:
                raise ValueError("shared input changed between matched routes")
            run([restore, model, args.manifest, workload["input"], candidate, ALL_BLOCKS_CSV, repeats],
                args.output / (name + "-buffered.log"), environment, lab)
            for label, reference in (("gpu", gpu), ("fp32", Path(workload["reference"]))):
                parity = args.output / (name + "-vs-" + label + ".json")
                run([sys.executable, oracle, "compare", "--reference", reference,
                     "--candidate", candidate, "--files", "restoration.safetensors", "--output", parity],
                    args.output / (name + "-vs-" + label + ".log"), environment, args.stable_root)
                if json.loads(parity.read_text()).get("passed") is not True:
                    raise ValueError("full restoration tensor gate failed")
            gpu_result = json.loads((gpu / "report.json").read_text())
            actual = json.loads((candidate / "report.json").read_text())
            if (actual.get("source_operand_recipe") != "original-fp16" or
                    actual.get("schedule") != "buffered_suffix" or
                    actual.get("selected_blocks") != list(range(32)) or
                    actual.get("coreml_calls", 0) <= 0 or
                    actual.get("prefix_snapshot_copied_bytes") != 0):
                raise ValueError("buffered schedule/all-block execution was not observed")
            workload.update(gpu_report=str(gpu / "report.json"), candidate_report=str(candidate / "report.json"), passed=True)
            if repeats > 1:
                gpu_time = statistics.median(row["denoise_seconds"] for row in gpu_result["iterations"][1:])
                candidate_time = statistics.median(row["denoise_seconds"] for row in actual["iterations"][1:])
                workload.update(component_gpu_warm_median=gpu_time,
                    component_candidate_warm_median=candidate_time,
                    component_screen_ratio=gpu_time / candidate_time)
            if sha256(Path(workload["reference"]) / "restoration.safetensors") != workload["reference_sha256"]:
                raise ValueError("independent reference changed")
            if sha256(Path(workload["input"])) != workload["input_sha256"]:
                raise ValueError("shared input changed during the matched routes")
            write_json(args.output / "report.json", report)
        if sha256(library) != library_sha or sha256(args.manifest) != manifest_sha:
            raise ValueError("stable runtime or artifact bank changed during the isolated screen")
        for source in provenance["files"]:
            if sha256(model / source["path"]) != source["sha256"]:
                raise ValueError("original weights changed during the screen")
        report.update(status="complete", numerical_gates_passed=True)
        print("isolated schedule screens complete; public integration remains disabled", flush=True)
    except BaseException as error:
        report.update(status="failed", failure=str(error))
        raise
    finally:
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
