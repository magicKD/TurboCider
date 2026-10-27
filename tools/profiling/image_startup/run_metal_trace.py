"""Launch the public-ABI image benchmark with an isolated Metal observer.

macOS framework Python launchers can exec a second binary, dropping injected
hooks. Select the actual framework application binary before injecting, while
leaving the caller's interpreter and all non-benchmark processes untouched.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def interpreter():
    application = Path(sys.base_prefix) / "Resources/Python.app/Contents/MacOS/Python"
    return str(application if application.is_file() else Path(sys.executable).resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--observer", type=Path, default=ROOT / "build/profiling/image-startup/metal_trace.dylib")
    parser.add_argument("benchmark_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("the observer requires macOS Metal")
    if args.trace.exists() or args.trace.is_symlink():
        parser.error("choose a fresh trace path; existing files are never overwritten")
    if not args.observer.is_file():
        parser.error("build the observer with build_metal_trace.sh first")
    arguments = args.benchmark_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    if not arguments:
        parser.error("pass benchmark arguments after --")
    if os.environ.get("DYLD_INSERT_LIBRARIES"):
        parser.error("use an uninstrumented parent; refusing to replace existing DYLD_INSERT_LIBRARIES")
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, DYLD_INSERT_LIBRARIES=str(args.observer.resolve()),
               TURBOCIDER_METAL_TRACE_FILE=str(args.trace.resolve()))
    command = [interpreter(), str(ROOT / "tools/validation/image_first_step.py"), *arguments,
               "--metal-trace", str(args.trace.resolve())]
    raise SystemExit(subprocess.run(command, env=env).returncode)


if __name__ == "__main__":
    main()
