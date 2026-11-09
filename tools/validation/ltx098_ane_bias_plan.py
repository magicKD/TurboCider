#!/usr/bin/env python3
"""Record Core ML's anticipated device plan for the biased LTX FFN graph.

The plan is a preference, not a measurement of hardware occupancy or proof
that any particular prediction executed on the Neural Engine.
"""
import argparse
import collections
import json
from pathlib import Path

from ltx098_comfy_reference import sha256
from qwen21_ane_placement import inspect_plan, load_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        parser.error("output must not already exist")
    before = sha256(manifest)
    spec = json.loads(manifest.read_text())
    if spec.get("backend") != "ltx098_experimental_bias_fp16" or spec.get("compiled_model") != "graph.mlmodelc":
        raise ValueError("expected the experimental bias-inclusive graph")
    root = manifest.parent
    files = {name: value for name, value in spec["files"].items() if name.startswith("graph.mlmodelc/")}
    if not files:
        raise ValueError("compiled graph receipt missing")
    compiled = root / "graph.mlmodelc"
    if compiled.is_symlink() or not compiled.is_dir():
        raise ValueError("compiled graph must be a real directory")
    observed = set()
    for path in compiled.rglob("*"):
        if path.is_symlink():
            raise ValueError("compiled graph symlinks are not supported")
        if path.is_file():
            observed.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError("compiled graph contains a nonregular entry")
    if observed != set(files):
        raise ValueError("compiled graph receipt does not cover every file")
    for name, expected in files.items():
        path = root / name
        if ".." in Path(name).parts or path.is_symlink() or sha256(path) != expected:
            raise ValueError(f"compiled graph identity mismatch: {name}")
    placement = inspect_plan(load_plan(compiled))
    if sha256(manifest) != before or any(sha256(root / name) != digest for name, digest in files.items()):
        raise ValueError("graph changed during plan inspection")
    counts = collections.Counter(row["category"] for row in placement)
    by_operator = {}
    for row in placement:
        bucket = by_operator.setdefault(row["operator"], collections.Counter())
        bucket[row["category"]] += 1
    report = {"schema": "ltx098-ane-bias-plan-v1", "status": "complete", "manifest_sha256": before,
              "scope": "Core ML anticipated device preference under CPU_AND_NE; not measured hardware occupancy",
              "physical_ane_placement": "unknown", "counts": dict(counts),
              "operators": {key: dict(value) for key, value in by_operator.items()}, "operations": placement}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "operations"}, indent=2))


if __name__ == "__main__":
    main()
