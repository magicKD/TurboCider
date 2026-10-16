#!/usr/bin/env python3
"""Read explicit MLX adoption runs/failures, pinned source and installed assets.

No candidate/MLX import, device execution, downloads or source checkout mutation.
Independent full logical witnesses are data verification, not attestation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs"))
from mlx_adoption import verify, verify_failure


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            result.update(data)
    return result.hexdigest()


def local(root, name):
    value = Path(name); path = (root / value).resolve()
    if value.is_absolute() or ".." in value.parts or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("explicit bounded local file required")
    return path


def curate(root, mlx, upstream, runs, failures):
    selected = runs + failures
    if not runs or len(selected) != len(set(selected)):
        raise ValueError("explicit nonempty unique selected runs required")
    node = next(value for value in json.loads((ROOT / "references/sources.json").read_text())["sources"] if value["id"] == "upstream-mlx-adoption")
    pin = node["upstream"]
    environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    command = ["git", "-c", "core.fsmonitor=false", "-C", str(upstream)]
    if (subprocess.check_output(command + ["rev-parse", "HEAD"], text=True, env=environment).strip() != pin["commit"]
            or subprocess.check_output(command + ["status", "--porcelain", "--untracked-files=no"], text=True, env=environment)):
        raise ValueError("pinned clean reading source required; never fetch/reset")
    for item in pin["files"]:
        if sha(local(upstream, item["path"])) != item["sha256"]:
            raise ValueError("reading source content differs")
    result = {"schema": "tc-mlx-adoption-evidence-v1", "curator_sha256": sha(Path(__file__)),
              "contract_sha256": sha(ROOT / "labs/mlx_adoption.py"), "upstream_reading_identity": pin,
              "production_qualified": False, "real_qwen": False, "formal_performance_score": False,
              "runtime_internal_copy_behavior": "unknown", "raw_addresses_or_device_ids_exported": False, "runs": [], "failures": []}
    checked_assets = set()
    for name in selected:
        path = local(root, name + "/report.json"); document = json.loads(path.read_text())
        failed = name in failures
        validation = verify_failure(document) if failed else verify(document)
        assets = document["mlx_assets"]
        identity = hashlib.sha256(json.dumps(assets, sort_keys=True).encode()).hexdigest()
        if identity not in checked_assets:
            actual_headers = [str(item.relative_to(mlx)) for item in sorted((mlx / "include").rglob("*")) if item.is_file()]
            if [item["path"] for item in assets["headers"]] != actual_headers:
                raise ValueError("complete installed-header inventory differs")
            for item in assets["headers"]:
                if sha(local(mlx, item["path"])) != item["sha256"]:
                    raise ValueError("installed MLX header differs")
            for item, digest in assets["libraries"].items():
                if sha(local(mlx, "lib/" + item)) != digest:
                    raise ValueError("installed MLX library asset differs")
            checked_assets.add(identity)
        if set(document["source_sha256"]) != {"mlx_adoption_probe.py", "mlx_adoption_probe.mm"}:
            raise ValueError("complete own-source snapshots required")
        for item, digest in document["source_sha256"].items():
            if sha(local(path.parent, "source/" + item)) != digest:
                raise ValueError("own source snapshot differs")
        for item, key in (("mlx-adoption", "binary_sha256"), ("native.json", "raw_native_sha256")):
            if sha(local(path.parent, item)) != document[key]:
                raise ValueError("binary/native content differs")
        if json.loads(local(path.parent, "native.json").read_text()) != document["native"]:
            raise ValueError("wrapper/native capsule differs")
        receipt_path = local(root, name + "-launch/receipt.json"); receipt = json.loads(receipt_path.read_text())
        if receipt.get("owned_process_exit_confirmed") is not True or receipt.get("returncode") != (1 if failed else 0):
            raise ValueError("actual owned child terminal receipt required")
        result["failures" if failed else "runs"].append({"id": name, "raw_report_sha256": sha(path),
            "supervisor_sha256": sha(receipt_path), "child_exit_confirmed": True, "independent_validation": validation, "report": document})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mlx-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--failure", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = curate(args.root.resolve(), args.mlx_root.resolve(), args.upstream_root.resolve(), args.run, args.failure)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"runs": len(value["runs"]), "failures": len(value["failures"]),
                      "complete_positive_consumer_hashes": sum(item["independent_validation"]["full_positive_consumer_hashes"] for item in value["runs"])}))


if __name__ == "__main__":
    main()
