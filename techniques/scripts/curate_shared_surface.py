#!/usr/bin/env python3
"""Verify explicitly selected shared-backing experiments; never execute devices.

Full logical hashes are reconstructed independently with Python struct; source,
MIL, binary and owned-child receipts are checked. Not hardware attestation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs"))
from shared_backing import verify


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local(root, relative):
    name = Path(relative)
    result = (root / name).resolve()
    if name.is_absolute() or ".." in name.parts or not result.is_relative_to(root.resolve()) or not result.is_file():
        raise ValueError("explicit bounded local file required")
    return result


def curate(root, selected):
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("nonempty unique selected runs required")
    result = {"schema": "tc-private-shared-evidence-v1", "curator_sha256": sha(Path(__file__)),
              "contract_sha256": sha(ROOT / "labs/shared_backing.py"), "production_qualified": False,
              "runtime_internal_copy_behavior": "unknown", "physical_overlap_qualified": False,
              "raw_addresses_or_surface_ids_exported": False, "runs": []}
    for name in selected:
        path = local(root, name + "/report.json")
        document = json.loads(path.read_text())
        validation = verify(document)
        receipt_path = local(root, name + "-launch/receipt.json")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("status") != "succeeded" or receipt.get("returncode") != 0 or receipt.get("owned_process_exit_confirmed") is not True:
            raise ValueError("terminal successful owned child required")
        if set(document["source_sha256"]) != {"shared_surface_probe.py", "shared_surface_probe.mm"}:
            raise ValueError("complete source identities required")
        for filename, digest in document["source_sha256"].items():
            if sha(local(path.parent, "source/" + filename)) != digest:
                raise ValueError("source snapshot differs")
        for filename, field in (("model/model.mil", "model_mil_sha256"), ("shared-probe", "binary_sha256"), ("native.json", "raw_native_sha256")):
            if sha(local(path.parent, filename)) != document[field]:
                raise ValueError("native/model/binary content differs")
        if json.loads(local(path.parent, "native.json").read_text()) != document["native"]:
            raise ValueError("wrapper/native capsule differs")
        result["runs"].append({"id": name, "raw_report_sha256": sha(path), "supervisor_sha256": sha(receipt_path),
                               "child_exit_confirmed": True, "independent_validation": validation, "report": document})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = curate(args.root.resolve(), args.run)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"runs": len(value["runs"]), "calls_independently_witnessed": sum(row["independent_validation"]["calls_independently_witnessed"] for row in value["runs"])}))


if __name__ == "__main__":
    main()
