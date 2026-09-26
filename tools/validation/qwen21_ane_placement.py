"""Offline Core ML plan inspection; planned placement is not a runtime trace."""
import argparse
import json
from pathlib import Path

import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    report = {"scope": "Core ML planned placement under CPU_AND_NE, not runtime hardware utilization", "blocks": {}}
    for block, artifacts in manifest["artifacts"].items():
        path = args.manifest.parent / artifacts["int8_pc"]
        plan = MLComputePlan.load_from_path(str(path.resolve()), compute_units=ct.ComputeUnit.CPU_AND_NE)
        placement = []
        for function in plan.model_structure.program.functions.values():
            for op in function.block.operations:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                placement.append({"operator": op.operator_name,
                                  "preferred": type(usage.preferred_compute_device).__name__ if usage else None})
        report["blocks"][block] = placement
        print(json.dumps({"block": block, "placement": placement}), flush=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
