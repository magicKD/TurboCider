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
    parser.add_argument("--blocks", type=int, nargs="*",
                        help="inspect selected blocks; default: every block")
    parser.add_argument("--quiet", action="store_true", help="print only a summary")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    report = {"scope": "Core ML planned placement under CPU_AND_NE, not runtime hardware utilization", "blocks": {}}
    blocks = args.blocks if args.blocks is not None else sorted(map(int, manifest["artifacts"]))
    if not blocks or len(set(blocks)) != len(blocks):
        raise ValueError("select one or more unique blocks")
    preferred = {"neural_engine": 0, "other": 0, "unassigned": 0, "constant": 0}
    for block_number in blocks:
        block = str(block_number)
        if block not in manifest["artifacts"]:
            raise ValueError(f"manifest has no block {block}")
        artifacts = manifest["artifacts"][block]
        path = args.manifest.parent / artifacts["int8_pc"]
        if path.suffix != ".mlmodelc" or not path.is_dir():
            raise ValueError("placement needs a compiled-cache manifest containing .mlmodelc artifacts")
        plan = MLComputePlan.load_from_path(str(path.resolve()), compute_units=ct.ComputeUnit.CPU_AND_NE)
        placement = []
        for function in plan.model_structure.program.functions.values():
            for op in function.block.operations:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                device = type(usage.preferred_compute_device).__name__ if usage else None
                category = ("constant" if op.operator_name == "const" or
                            "constexpr" in op.operator_name else
                            "neural_engine" if device == "MLNeuralEngineComputeDevice" else
                            "unassigned" if device is None else "other")
                preferred[category] += 1
                placement.append({"operator": op.operator_name, "preferred": device})
        report["blocks"][block] = placement
        if not args.quiet:
            print(json.dumps({"block": block, "placement": placement}), flush=True)
    report["summary"] = {"blocks": len(blocks), "operator_preferences": preferred}
    print(json.dumps(report["summary"]), flush=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
