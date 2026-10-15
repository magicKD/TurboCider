#!/usr/bin/env python3
"""Read explicitly selected ANEForge receipts; verify lineage and all-call gates.

Does not import upstream, candidate runner, NumPy/MLX or a native shim. This is a
data verifier, not numerical re-execution or physical-device attestation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs"))
from search_contracts import paired_result

COMMIT = "896c4943c5ebf811c0e815f8d1381fd1e157fa7d"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local(root, name):
    relative = Path(name)
    value = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not value.is_relative_to(root.resolve()) or not value.is_file():
        raise ValueError("missing or escaping selected artifact")
    return value


def verify_device(document):
    if (document.get("schema") != "tc-aneforge-research-v1" or document.get("status") != "measured"
            or document.get("production_qualified") is not False or document.get("source_inputs_unchanged") is not True
            or document.get("native_compute_device_mask") != 4):
        raise ValueError("successful source-stable explicit private-runtime receipt required")
    samples = document["samples"]
    if len(samples) != 27 or set(row["arm"] for row in samples) != {"gpu", "ane-baseline", "ane-candidate"}:
        raise ValueError("all three arms and all 27 first/warm/measured calls required")
    arms = ("gpu", "ane-baseline", "ane-candidate")
    orders = (arms, arms[::-1], (arms[1], arms[2], arms[0]),
              (arms[0], arms[2], arms[1]), (arms[2], arms[0], arms[1]), (arms[1], arms[0], arms[2]))
    expected = [(phase, trial, arm) for phase, count in (("first", 1), ("warmup", 2), ("measured", 6))
                for trial in range(count) for arm in orders[trial % 6]]
    if [(row["phase"], row["trial"], row["arm"]) for row in samples] != expected:
        raise ValueError("complete balanced three-arm execution order required")
    comparisons = [paired_result(samples, a, b, .01) for a, b in
                   (("ane-baseline", "ane-candidate"), ("gpu", "ane-baseline"), ("gpu", "ane-candidate"))]
    if comparisons != document.get("comparisons"):
        raise ValueError("declared ratio or numerical rejection differs")
    return comparisons


def curate(root, upstream, ids):
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True)
    if commit != COMMIT or dirty:
        raise ValueError("fixed clean upstream source required; never fetch/reset it")
    tree = {}
    for path in sorted((upstream / "aneforge").rglob("*")):
        if not path.is_file() or path.suffix in (".pyc", ".dylib") or path.name == ".DS_Store":
            continue
        relative = path.relative_to(upstream / "aneforge").as_posix()
        expected_hash = sha(path)
        if sha(local(root / "source/aneforge", relative)) != expected_hash:
            raise ValueError("complete isolated source copy differs")
        tree[relative] = expected_hash
    result = {"schema": "tc-aneforge-campaign-evidence-v1", "curator_sha256": sha(Path(__file__)),
              "upstream": {"repository": "sbryngelson/ANEForge", "commit": commit, "worktree_dirty": False},
              "production_qualified": False, "scope": "bounded synthetic graphs; not Qwen/request/handoff qualification",
              "physical_device_attestation": False,
              "isolated_copy_post_run": {"files_compared": len(tree),
                  "relative_file_hash_map_sha256": hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest(),
                  "scope": "complete source copy compared after runs; selected per-run source hashes checked before and after execution"},
              "runs": []}
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate selected run")
    for name in ids:
        path = local(root, name + "/report.json")
        value = json.loads(path.read_text())
        supervisor = local(root, name + "-launch/receipt.json")
        launched = json.loads(supervisor.read_text())
        if launched.get("owned_process_exit_confirmed") is not True:
            raise ValueError("owned child exit unconfirmed")
        for relative, expected in value["source_sha256"].items():
            if sha(local(upstream / "aneforge", relative)) != expected:
                raise ValueError("upstream file identity differs")
        snapshots = []
        for filename, key in (("aneforge_probe.py", "runner_sha256"),
                              ("search_contracts.py", "source_contract_helper_sha256"),
                              ("precision_oracle.py", "independent_oracle_sha256")):
            if key not in value:
                continue
            candidate = path.parent / "source" / filename
            available = candidate.is_file()
            if available and sha(candidate) != value[key]:
                raise ValueError("probe snapshot differs")
            snapshots.append({"name": filename, "sha256": value[key], "local_snapshot_available": available})
        if value["status"] == "measured":
            if launched.get("returncode") != 0:
                raise ValueError("successful report and supervisor differ")
            verify_device(value)
            if sha(local(root, "source/aneforge/_lib/libane_e5rt_dispatch.dylib")) != value["shim_sha256"]:
                raise ValueError("explicit shim differs")
            for arm in ("ane-baseline", "ane-candidate"):
                for filename, key in (("model.mil", "mil_sha256"), ("weights.bin", "weights_sha256")):
                    if sha(local(path.parent, arm + "/" + filename)) != value["setup"][arm][key]:
                        raise ValueError("emitted program changed")
        elif value["status"] == "cpu-contracts-observed":
            if launched.get("returncode") != 0 or value["contracts"]["dylib_loaded"] is not False:
                raise ValueError("CPU contract must not load native runtime")
            for item in value["contracts"]["emitted_programs"]:
                directory = path.parent / "cache" / item["upstream_content_key"]
                mil, weights = local(directory, "model.mil"), local(directory, "weights.bin")
                if (sha(mil) != item["mil_sha256"] or sha(weights) != item["weights_sha256"]
                        or hashlib.sha256(mil.read_bytes() + weights.read_bytes()).hexdigest()[:24] != directory.name):
                    raise ValueError("CPU emitted content identity differs")
        elif value["status"] != "failed" or not value["errors"] or launched.get("returncode") == 0:
            raise ValueError("unknown or unproven failure status")
        result["runs"].append({"id": name, "raw_report_sha256": sha(path),
            "supervisor_receipt_sha256": sha(supervisor), "child_exit_confirmed": True,
            "source_snapshots": snapshots, "report": value})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("use a fresh output path")
    value = curate(args.root.resolve(), args.upstream_root.resolve(), args.run)
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if "/Users/" in encoded or "/Volumes/" in encoded:
        parser.error("machine-local paths must remain private")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    print(json.dumps({"runs": len(value["runs"]), "production_qualified": False}))


if __name__ == "__main__":
    main()
