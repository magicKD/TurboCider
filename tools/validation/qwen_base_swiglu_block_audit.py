#!/usr/bin/env python3
"""CPU-only raw/source/kernel/strong-denominator recheck for both tile recipes."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

from flux9_bf16_framework_screen import write_json
from qwen_lora_swiglu_block_audit import arrays, same
import qwen_base_swiglu_block_screen as screen
from runtime_ane_common import sha256_file

ROOT = Path(__file__).resolve().parents[2]
V1 = "original-base-paired-b-swiglu-v1"
V2 = "original-base-paired-b-swiglu-v2-b64"


def kernel_contract(path, rows):
    value = json.loads(path.read_text())
    if (value.get("status") != "complete" or value.get("all_fixtures_bitwise_exact") is not True
            or value.get("production_geometry_bitwise_exact") is not True or value.get("rejected") != 30):
        raise ValueError("complete exact per-stage kernel controls required")
    tiles = (16, 32) if rows == 84 else (16, 32, 64)
    expected = {(fixture, bm, bn, stage) for fixture in range(3) for bm in tiles for bn in (64, 128) for stage in range(7)}
    found = set()
    for row in value["cases"]:
        key = row["fixture"], row["bm"], row["bn"], row["stage"]
        if key in found or key not in expected:
            raise ValueError("missing/duplicate/unexpected kernel control")
        found.add(key)
        m, k, n = ((33, 128, 257), (65, 256, 513), (1024, 4096, 12288))[row["fixture"]]
        q = row["quality"]
        if ((row["m"], row["k"], row["n"]) != (m, k, n) or q.get("bits_exact") is not True
                or q.get("changed") != 0 or q.get("relative_l2") != 0 or q.get("max_abs") != 0 or q.get("elements") != m * n):
            raise ValueError("full per-stage bitwise evidence differs")
    if found != expected or len(value["cases"]) != rows:
        raise ValueError("all kernel fixture/tile/stage controls required")
    return dict(stages=rows, rejected=value["rejected"], all_bitwise_exact=True, sha256=sha256_file(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("library", "native-source-archive", "v1-source-archive", "binary-v1", "binary-v2", "kernel-v1", "kernel-v2", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--cohort", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh audit output required")
    manifest = json.loads((args.library.parent / "runtime-build/runtime-build-manifest.json").read_text())
    with tarfile.open(args.native_source_archive, "r:gz") as archive:
        entries = {entry.name: entry for entry in archive.getmembers()}
        if set(entries) != {entry["path"] for entry in manifest["inputs"]["sources"]}:
            raise ValueError("complete sealed native archive required")
        for entry in manifest["inputs"]["sources"]:
            stream = archive.extractfile(entries[entry["path"]])
            if stream is None or hashlib.sha256(stream.read()).hexdigest() != entry["sha256"] or sha256_file(ROOT / entry["path"]) != entry["sha256"]:
                raise ValueError("native current/archive source differs from library")
    with tarfile.open(args.v1_source_archive, "r:gz") as archive:
        members = [entry for entry in archive.getmembers() if entry.isfile()]
        if any(entry.name not in screen.SOURCES and not Path(entry.name).name.startswith("._") for entry in members):
            raise ValueError("unexpected historical source archive member")
        old_sources = {entry.name: archive.extractfile(entry).read().decode() for entry in members if entry.name in screen.SOURCES}
    if set(old_sources) != set(screen.SOURCES):
        raise ValueError("complete old tool source archive required")
    library_sha = sha256_file(args.library)
    binaries = {V1: sha256_file(args.binary_v1), V2: sha256_file(args.binary_v2)}
    observations, by_task, seen, raw_hashes = [], {}, set(), []
    exact = 0
    for root in args.cohort:
        summary = json.loads((root / "summary.json").read_text())
        native = json.loads((root / "trial/native.json").read_text())
        recipe = summary["descriptor"]["recipe"]
        key = recipe, summary["edit"], summary["order"]
        if (recipe not in binaries or key in seen or summary["status"] != "complete"
                or summary["asset_sha256"]["library"] != library_sha or summary["asset_sha256"]["binary"] != binaries[recipe]
                or native["loaded_runtime_id"] != manifest["runtime_build_id"]):
            raise ValueError("distinct complete same-library/bound-recipe windows required")
        seen.add(key)
        snapshots = json.loads((root / "source.json").read_text())
        if set(snapshots) != set(screen.SOURCES) or set(summary["source_sha256"]) != set(screen.SOURCES):
            raise ValueError("all bound tool sources required")
        for name, digest in summary["source_sha256"].items():
            expected = old_sources[name] if recipe == V1 else (ROOT / name).read_text()
            if snapshots[name] != expected or hashlib.sha256(snapshots[name].encode()).hexdigest() != digest:
                raise ValueError("historical archive/current tool source differs")
        analysis = screen.analyze(native, summary["order"], legacy=recipe == V1)
        if analysis != summary["analysis"]:
            raise ValueError("full strong-denominator reanalysis differs")
        arms = screen.V1_ARMS if recipe == V1 else screen.ARMS
        medians = {arm: statistics.median(row["seconds"] for row in native["samples"] if row["iteration"] >= 0 and row["arm"] == arm) for arm in arms}
        if medians != analysis["median_seconds"]:
            raise ValueError("independent raw sample medians differ")
        values = arrays(root / "trial/inputs-and-hidden.safetensors")
        reference, hidden = values["block-reference"], values["hidden-reference"]
        if reference[:2] != ("BF16", (1, 1024, 4096)) or hidden[:2] != ("BF16", (1, 1024, 12288)):
            raise ValueError("complete original block and hidden required")
        if (hashlib.sha256(reference[2]).hexdigest() != native["ordinary_block_sha256"]
                or hashlib.sha256(values["argument-0"][2]).hexdigest() != native["input_sha256"]
                or values["changed-reference"] == reference):
            raise ValueError("actual original/perturbed input or reference differs")
        for arm in arms[2:]:
            same(hidden, values[arm])
            exact += 1
        outputs = arrays(root / "trial/complete-block-outputs.safetensors")
        if set(outputs) != set(arms):
            raise ValueError("all saved final output arms required")
        for value in outputs.values():
            same(reference, value)
            exact += 1
        for arm in arms[1:]:
            same(reference, arrays(root / f"trial/{arm}-initial.safetensors")["block"])
            exact += 1
        common = {name: value for name, value in values.items() if name not in arms[2:]}
        prior = by_task.setdefault(summary["edit"], (native["actual_layer_source_sha256"], common))
        if prior != (native["actual_layer_source_sha256"], common):
            raise ValueError("cross-recipe/order input/ordinary reference/hidden differs")
        for name in ("evidence", "report"):
            if sha256_file(root / summary["memory"][name]) != summary["memory"][name + "_sha256"]:
                raise ValueError("process memory evidence changed")
        memory = summary["memory"]["verified"]
        if memory["complete"] is not True or memory["command_exit_code"] != 0:
            raise ValueError("complete owned process window required")
        loads = [json.loads(line) for line in (root / "load.jsonl").read_text().splitlines()]
        if "load" in summary and sha256_file(root / summary["load"]["evidence"]) != summary["load"]["evidence_sha256"]:
            raise ValueError("load evidence digest changed")
        observations.append(dict(cohort=root.name, recipe=recipe, edit=summary["edit"], order=summary["order"], analysis=analysis,
                                 memory=memory, load_error=summary.get("load_error"), competing_load_samples=sum(bool(row["competing_processes"]) for row in loads),
                                 summary_sha256=sha256_file(root / "summary.json")))
        raw_hashes.extend(dict(path=str(path), sha256=sha256_file(path)) for path in sorted((root / "trial").glob("*.safetensors")))
    if set(by_task) != {False, True} or by_task[False][0] != by_task[True][0]:
        raise ValueError("both tasks with the same complete original layer sources required")
    if not {(V1, False, "forward"), (V1, True, "reverse"), (V2, False, "reverse")} <= seen:
        raise ValueError("both v1 tasks and the stopped v2 generation experiment required")
    result = dict(schema="tc-qwen-base-swiglu-block-audit-v1", status="complete", qualification_passed=False,
                  library_sha256=library_sha, runtime_build_id=manifest["runtime_build_id"], native_source_hashes_verified=len(manifest["inputs"]["sources"]),
                  native_source_archive_sha256=sha256_file(args.native_source_archive), v1_source_archive_sha256=sha256_file(args.v1_source_archive), binary_sha256=binaries,
                  kernel_v1=kernel_contract(args.kernel_v1, 84), kernel_v2=kernel_contract(args.kernel_v2, 126),
                  v2_edit_measured=any(key[0] == V2 and key[1] for key in seen),
                  saved_bitwise_exact_array_comparisons=exact, measured_full_blocks=64 * len(observations), warm_full_blocks=12 * len(observations),
                  observations=observations, raw_file_sha256=raw_hashes, script_sha256=sha256_file(Path(__file__)),
                  scope="CPU first/final full block and hidden/raw/source/kernel/denominator recheck; not every timed output, full images, physical overlap or formal performance qualification")
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
