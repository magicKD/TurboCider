#!/usr/bin/env python3
"""Dispatch TurboCider LoRA preparation to its bundled merge tools.

The audited H3 safetensors and LTX ConvRot implementations live beside this
entrypoint in the development repository, not the App/CLI release package.
Preparation leaves a provenance
manifest beside the merged checkpoint. Native inference itself does not
require Python once the optional prepared artifact exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from lora_runtime_cache import ensure_cache


MODEL_ALIASES = {
    "h3": "h3",
    "minimax-h3-turbo": "h3",
    "ltx": "ltx",
    "ltx-2.5-distilled": "ltx",
}


def upstream_script(model: str) -> Path:
    canonical = MODEL_ALIASES[model]
    name = "merge_h3_lora.py" if canonical == "h3" else "merge_ltx_refiner.py"
    bundled = Path(__file__).resolve().parent / name
    if bundled.is_file():
        return bundled
    raise FileNotFoundError(
        f"bundled {name} is unavailable beside {Path(__file__).name}"
    )


def build_command(args: argparse.Namespace) -> list[str]:
    canonical = MODEL_ALIASES[args.model]
    command = [
        sys.executable,
        str(upstream_script(args.model)),
        str(Path(args.base).expanduser()),
        str(Path(args.lora).expanduser()),
        str(Path(args.output).expanduser()),
    ]
    if args.check_only:
        command.append("--check-only")
    if args.overwrite:
        command.append("--overwrite")
    if args.device:
        command.extend(["--device", args.device])
    if canonical == "h3":
        if args.profile != "auto":
            command.extend(["--profile", args.profile])
        if args.strength is not None:
            command.extend(["--strength", str(args.strength)])
        if args.source_revision:
            command.extend(["--source-revision", args.source_revision])
    elif args.profile != "auto" or args.strength is not None or args.source_revision:
        raise ValueError(
            "--profile, --strength and --source-revision are only valid for H3"
        )
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] not in {"prepare", "runtime-cache"}:
        arguments.insert(0, "prepare")
    parser = argparse.ArgumentParser(
        description="Prepare a provenance-bound H3 or LTX LoRA checkpoint"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("prepare", "runtime-cache"),
        default="prepare",
        help="prepare an explicit output, or build an identity-bound runtime cache",
    )
    parser.add_argument("model", choices=sorted(MODEL_ALIASES))
    parser.add_argument("base", help="unmerged base checkpoint or transformer directory")
    parser.add_argument("lora", help="LoRA safetensors file")
    parser.add_argument("output", nargs="?", help="separate merged checkpoint/directory")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"))
    parser.add_argument(
        "--profile",
        choices=("auto", "native", "lightx2v-4step"),
        default="auto",
    )
    parser.add_argument("--strength", type=float)
    parser.add_argument("--source-revision")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument("--hardware", default="apple-silicon")
    return parser.parse_args(arguments)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "runtime-cache":
        if args.output is not None:
            print("runtime-cache does not accept an output positional", file=sys.stderr)
            return 2
        try:
            value = ensure_cache(
                MODEL_ALIASES[args.model],
                args.base,
                args.lora,
                args.strength if args.strength is not None else 1.0,
                role="transformer",
                cache_dir=args.cache_dir,
                profile=args.profile,
                hardware=args.hardware,
            )
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
            print(f"turbocider runtime-cache: {error}", file=sys.stderr)
            return 2
        print(json.dumps(value, indent=2, sort_keys=True))
        return 0
    if args.output is None:
        print("prepare requires OUTPUT", file=sys.stderr)
        return 2
    try:
        command = build_command(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"turbocider prepare-lora: {error}", file=sys.stderr)
        return 2
    if args.print_command:
        print(
            json.dumps(
                {
                    "model": MODEL_ALIASES[args.model],
                    "upstream": command[1],
                    "command": command,
                },
                sort_keys=True,
            )
        )
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
