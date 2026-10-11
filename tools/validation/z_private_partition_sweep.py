#!/usr/bin/env python3
"""Serial complete Z GPU/frozen/Private partition candidates; retain failures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from flux9_bf16_framework_screen import generation, check_generations, write_json
from runtime_ane_common import sha256_file
from z_mflux_framework_screen import PRIVATE_SHARES

ROOT = Path(__file__).resolve().parents[2]
ORDER = "native-gpu,native-frozen,native-private,native-private,native-frozen,native-gpu"


def candidates(shares, joins):
    if not shares or not joins or len(set(shares)) != len(shares) or len(set(joins)) != len(joins) or any(
            share not in PRIVATE_SHARES for share in shares) or any(join not in ("f16", "f32") for join in joins):
        raise ValueError("unique declared partial channel shares and join recipes required")
    return [dict(channels=share, join=join) for join in joins for share in shares]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("library", "reference-model", "mflux", "python", "extra", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--shares", default=",".join(map(str, PRIVATE_SHARES)))
    parser.add_argument("--joins", default="f32,f16")
    args = parser.parse_args()
    selected = candidates([int(value) for value in args.shares.split(",")], args.joins.split(","))
    if args.output.exists():
        parser.error("fresh sweep destination required")
    for name in ("library", "reference_model", "mflux", "extra"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.python = args.python.absolute()
    args.output = args.output.absolute()
    args.output.mkdir(parents=True)
    screen = ROOT / "tools/validation/z_mflux_framework_screen.py"
    identities = {str(path): generation(path) for path in (args.library, screen, Path(__file__).resolve())}
    result = dict(schema="tc-z-private-partition-sweep-v1", status="running", qualification_passed=False,
        goal_complete=False, candidates=selected, trials=[], library_sha256=sha256_file(args.library),
        timing_scope="each candidate owns full8-step six-process GPU/frozen/Private palindrome; never pool different candidates",
        source_policy="underlying same-source generation-bound registration; no per-inference model SHA")
    destination = args.output / "summary.json"
    write_json(destination, result)
    try:
        for index, candidate in enumerate(selected):
            check_generations(identities)
            output = args.output / f'{index}-a{candidate["channels"]}-{candidate["join"]}'
            command = [str(args.python), "-B", str(screen), "--library", str(args.library),
                "--reference-model", str(args.reference_model), "--mflux", str(args.mflux), "--python", str(args.python),
                "--extra", str(args.extra), "--output", str(output), "--order", ORDER,
                "--private-channels", str(candidate["channels"]), "--private-join", candidate["join"]]
            print(json.dumps(dict(starting=index, **candidate)), flush=True)
            # One owned model campaign at a time. A failed candidate is kept
            # and does not erase the rest of this finite parameter grid.
            with (args.output / f"candidate-{index}-stdout.txt").open("x") as out, (args.output / f"candidate-{index}-stderr.txt").open("x") as err:
                completed = subprocess.run(command, stdout=out, stderr=err)
            check_generations(identities)
            summary = output / "summary.json"
            data = json.loads(summary.read_text()) if summary.is_file() else None
            accepted = completed.returncode == 0 and data is not None and data.get("status") == "complete_diagnostic"
            result["trials"].append(dict(index=index, **candidate, exit_code=completed.returncode,
                accepted=accepted, evidence=str(summary.relative_to(ROOT)),
                evidence_sha256=sha256_file(summary) if summary.is_file() else None,
                medians=data.get("medians") if accepted else None, failure=data.get("error") if data else "no terminal candidate summary"))
            write_json(destination, result)
        result["status"] = "complete_diagnostic" if all(trial["accepted"] for trial in result["trials"]) else "complete_grid_with_retained_failures"
    except BaseException as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        write_json(destination, result)
    print(json.dumps(dict(status=result["status"], accepted=sum(row["accepted"] for row in result["trials"]))))


if __name__ == "__main__":
    main()
