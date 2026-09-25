"""Screen all blocks and bind training-only channel routes to a source manifest.

The held-out metrics are observations, never used to pick groups. This script
does not export Core ML or qualify image quality. Run the resulting route
through the exporter and native GPU gather before drawing runtime conclusions.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--heldout-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--group-width", type=int, default=256)
    parser.add_argument("--ane-width", type=int,
                        help="candidate ANE width; the source manifest supplies calibration provenance")
    parser.add_argument("--calibration-rows", type=int, default=32)
    parser.add_argument("--heldout-rows", type=int, default=32)
    parser.add_argument("--caption-route-weight", type=float, default=0.,
                        help="training caption NMSE relative to image NMSE (default: image only)")
    parser.add_argument("--caption-route-blocks", default="",
                        help="comma-separated block indexes for caption-aware selection; "
                             "empty means all non-refiner blocks when weight is positive")
    parser.add_argument("--caption-image-slack", type=float,
                        help="optional training image-SSE budget for caption-improving swaps")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    identity, shape = manifest["export_identity"], manifest["shape"]
    target_width = args.ane_width or shape["ane_mlp_end"]
    caption_blocks = (set(map(int, args.caption_route_blocks.split(",")))
                      if args.caption_route_blocks else set(range(2, 32))
                      if args.caption_route_weight else set())
    if (sorted(map(int, manifest["artifacts"])) != list(range(32)) or
            args.group_width < 32 or args.group_width % 32 or
            10240 % args.group_width or shape["ane_mlp_end"] % args.group_width or
            not 0 < target_width <= 10240 - args.group_width or
            target_width % args.group_width or
            args.output.exists() or args.output.is_symlink() or
            any(block < 2 or block >= 32 for block in caption_blocks) or
            (args.caption_route_blocks and not args.caption_route_weight) or
            (args.caption_image_slack is not None and
             (not 0 <= args.caption_image_slack < float("inf") or
              not args.caption_route_weight))):
        raise ValueError("requires all 32 blocks, aligned routing, and a new output file")
    routes, assessments = {}, {}
    for block in range(32):
        command = [sys.executable,
                   str(Path(__file__).with_name("z_image_w8a8_routing.py")),
                   "--manifest", str(args.manifest), "--model", str(args.model),
                   "--calibration-dir", str(args.calibration_dir),
                   "--heldout-dir", str(args.heldout_dir), "--block", str(block),
                   "--group-width", str(args.group_width),
                   "--ane-width", str(target_width),
                   "--calibration-rows", str(args.calibration_rows),
                   "--heldout-rows", str(args.heldout_rows),
                   "--caption-route-weight", str(args.caption_route_weight
                                                   if block in caption_blocks else 0.)]
        if block in caption_blocks and args.caption_image_slack is not None:
            command.extend(("--caption-image-slack", str(args.caption_image_slack)))
        result = json.loads(subprocess.check_output(command, text=True))
        if result["block"] != block or len(result["selected_groups"]) != (
                target_width // args.group_width):
            raise ValueError(f"invalid route for block {block}")
        routes[str(block)] = result["selected_groups"]
        assessments[str(block)] = result
        print(json.dumps({"block": block, "contiguous_error_vs_full":
                          result["contiguous_error_vs_full_relative_l2"],
                          "routed_error_vs_full": result["routed_error_vs_full_relative_l2"]}),
              flush=True)
    data = {"owner": "turbocider.z_image.w8a8.channel_routing.v1",
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "calibration_sha256": {block: identity["calibration"][block]["sha256"]
                                   for block in routes},
            "ane_mlp_width": target_width, "mlp_width": 10240,
            "calibration_source_ane_width": shape["ane_mlp_end"],
            "group_width": args.group_width, "ane_group_indexes": routes,
            "caption_route_weight": args.caption_route_weight,
            "caption_route_blocks": sorted(caption_blocks),
            **({"caption_image_slack": args.caption_image_slack}
               if args.caption_image_slack is not None else {}),
            "assessment": assessments,
            "scope": "FP32 A8-only routes selected on calibration; heldout not used for selection"}
    # The source route is immutable input to the exporter; refuse overwrite.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"route": str(args.output), "blocks": len(routes)}))


if __name__ == "__main__":
    main()
