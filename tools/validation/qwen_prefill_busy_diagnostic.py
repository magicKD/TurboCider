#!/usr/bin/env python3
"""Record a complete Qwen prefill screen and its potentially busy load window.

Strict --observe-load screens still abort unchanged. This explicit diagnostic
never certifies performance, even if its enclosing CPU-load check passes. Keep
the raw observation and any failure instead of stopping unrelated processes.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from runtime_ane_common import sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_owned

ROOT=Path(__file__).resolve().parents[2]
SCREEN=ROOT/"tools/validation/qwen_encoder_residency_screen.py"
PHASE_LAYER_SCREEN=ROOT/"tools/validation/qwen_ffn_phase_screen.py"


def screen_arguments(arguments,phase_layers=False):
    args=list(arguments)
    if args[:1]==["--"]:args.pop(0)
    if not args or any(flag in args for flag in
            ("--observe-load","--defer-prefill-screen","--prefill-layer-screen","--sample-memory")):
        raise ValueError("provide plain screen arguments; wrapper owns diagnostic/phase/memory policy")
    if phase_layers:return ["--prefill-layer-screen","--sample-memory",*args]
    phase=[] if any(flag in args for flag in ("--gpu-first-prefill-screen","--compiled-encoder-screen")) else ["--defer-prefill-screen"]
    return [*phase,"--sample-memory",*args]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-evidence",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--phase-layers",action="store_true",help="use the same-library complete GPU/prefill layer screen instead of encoder/deferred screen")
    parser.add_argument("arguments",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    try:flags=screen_arguments(args.arguments,args.phase_layers)
    except ValueError as error:parser.error(str(error))
    if args.load_evidence==args.report or any(p.exists() or p.is_symlink() for p in (args.load_evidence,args.report)):
        parser.error("choose distinct fresh evidence/report files")
    observer=LoadObservation(args.load_evidence)
    result=run_owned([sys.executable,str(PHASE_LAYER_SCREEN if args.phase_layers else SCREEN),*flags],cwd=ROOT,env=os.environ.copy(),timeout=3600,observer=observer)
    load=None;load_error=None
    try:load=observer.verify()
    except ValueError as error:load_error=str(error)
    report=dict(schema="tc-qwen-busy-prefill-diagnostic-v1",qualification_passed=False,
        screen="prefill_layers" if args.phase_layers else "encoder_deferred",
        screen_exit_code=result.returncode,load_check_passed=load is not None,load=load,load_error=load_error,
        load_evidence=args.load_evidence.name,load_evidence_sha256=sha256_file(args.load_evidence),
        scope="explicit busy-window host diagnostic; no competing process signals; not qualified performance, physical overlap or visual acceptance")
    with args.report.open("x") as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps(report,indent=2))
    if result.returncode:raise SystemExit(result.returncode)


if __name__=="__main__":main()
