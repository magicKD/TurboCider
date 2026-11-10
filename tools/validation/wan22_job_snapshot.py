#!/usr/bin/env python3
"""Copy one Wan2.2 job request into a new isolated experiment fixture."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--replacement-seed", type=int, help="use a different seed in the copied request")
    args = parser.parse_args()
    history = json.loads(args.history.read_text())
    matches = [job["request"] for job in history if job.get("request", {}).get("model") ==
               "wan2.2-ti2v-5b-fastwan" and job["request"].get("seed") == args.seed]
    if len(matches) != 1:
        parser.error("seed must identify exactly one Wan2.2 request")
    request = matches[0]
    if args.replacement_seed is not None:
        if args.replacement_seed < 0 or args.replacement_seed > (1 << 64) - 1:
            parser.error("replacement seed must be an unsigned 64-bit integer")
        request = dict(request, seed=args.replacement_seed)
    fields = ("model", "prompt", "seed", "inputs", "width", "height", "frames", "fps", "steps")
    with args.output.open("x") as output:
        json.dump([{"request": {key: request[key] for key in fields}}], output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(f"Saved one request for seed {request['seed']}; no other job history was copied")


if __name__ == "__main__":
    main()
