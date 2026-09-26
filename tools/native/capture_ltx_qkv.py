#!/usr/bin/env python3
"""Capture one real Stage-2 attention input; intentionally not a speed benchmark."""
import argparse
import ctypes as c
import json
import os
from pathlib import Path
from benchmark_ltx_resident import load_library, create_engine, consume
from benchmark_ltx_env_abba import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step", type=int, choices=range(3), default=0)
    parser.add_argument("--block", type=int, choices=range(1, 47), default=1)
    parser.add_argument("--prompt", default="A cinematic red fox running through a snowy forest")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("output exists; use a fresh capture directory")
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    binary = root / "build/native/libturbocider.dylib"
    fixture = output / "qkv"
    request = dict(schema_version=1, model="ltx-2.5-distilled", operation="video.generate",
                   prompt=args.prompt, width=args.width, height=args.height, frames=args.frames,
                   fps=24, steps=11, seed=args.seed, execution="gpu", residency="component_staged",
                   audio=True, ltx_backend="c_metal", ltx_fast_av=True,
                   allow_approximation=True, ltx_sol_stage2=True,
                   ltx_sol_dense_edge_blocks=1, ltx_sol_dense_edge_steps=0,
                   ltx_sparse_mode=1, ltx_sparse_radius=256, ltx_sparse_anchor_stride=0,
                   ltx_sparse_tokens_per_frame=0,
                   output=str(output / "source.mp4"), dump_tensors=str(output / "tensors"))
    report = dict(schema="ltx-qkv-capture-request-v1", request=request,
                  library_sha256=file_sha256(binary),
                  shader_sha256=file_sha256(root / "build/native/ltx_shaders.metal"),
                  selected_step=args.step, selected_block=args.block,
                  warning="capture includes synchronous I/O; timings are not benchmark results",
                  complete=False)
    report_path = output / "capture.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    environment = {"TURBOCIDER_LTX_CAPTURE_QKV_DIR": str(fixture),
                   "TURBOCIDER_LTX_CAPTURE_QKV_STEP": str(args.step),
                   "TURBOCIDER_LTX_CAPTURE_QKV_BLOCK": str(args.block)}
    originals = {key: os.environ.get(key) for key in environment}
    library = load_library(binary)
    engine = create_engine(library, args.model.resolve())
    try:
        os.environ.update(environment)
        value, error = c.c_void_p(), c.c_void_p()
        status = library.tc_engine_generate(engine, json.dumps(request).encode(), None, None,
                                            c.byref(value), c.byref(error))
        result, failure = consume(library, value), consume(library, error)
        if status:
            raise RuntimeError(failure)
        metadata = json.loads((fixture / "metadata.json").read_text())
        if metadata["step"] != args.step or metadata["block"] != args.block:
            raise RuntimeError("wrong capture step/block")
        report.update(result=json.loads(result), metadata=metadata,
                      fixture_sha256={p.name: file_sha256(p) for p in fixture.iterdir()}, complete=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(metadata), flush=True)
    finally:
        library.tc_engine_free(engine)
        for key, value in originals.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    main()
