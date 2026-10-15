#!/usr/bin/env python3
"""Run two licensed GGML Metal dequantizer excerpts, not a whole-engine bench."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import struct
import sys

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))
from quantization import ggml_q4_0_block, ggml_q8_0_block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import mlx.core as mx
    asset = LABS.parent / "assets/source-excerpts/ggml-dequant-64e9bce"
    excerpt = (asset / "dequantizers.metal").read_text()
    header = ("#include <metal_stdlib>\nusing namespace metal;\n"
              "struct block_q4_0 { half d; uint8_t qs[16]; };\n"
              "struct block_q8_0 { half d; int8_t qs[32]; };\n"
              "static_assert(sizeof(block_q4_0)==18); static_assert(sizeof(block_q8_0)==34);\n" + excerpt)
    cases = []
    for bits, scale, payload, label in ((4, .125, bytes(i | ((15-i)<<4) for i in range(16)), "q4-nibble-order"),
                                       (4, 2**-24, bytes([255]*16), "q4-subnormal-scale"),
                                       (8, .25, struct.pack("<32b", *range(-16,16)), "q8-signed-codes")):
        raw = struct.pack("<e", scale) + payload
        reference = ggml_q4_0_block(raw) if bits == 4 else ggml_q8_0_block(raw)
        source = (f"uint il=thread_position_in_grid.x; float4x4 value; "
                  f"dequantize_q{bits}_0((device const block_q{bits}_0*)bytes, short(il), value); "
                  "for(uint i=0;i<16;i++) out[il*16+i]=value[i/4][i%4];")
        kernel = mx.fast.metal_kernel(name=f"tc_ggml_q{bits}_excerpt", input_names=["bytes"], output_names=["out"],
                                     source=source, header=header, compile_options={"math_mode":"safe"})
        actual = kernel(inputs=[mx.array(np.frombuffer(raw,dtype=np.uint8).copy())], grid=(2,1,1), threadgroup=(2,1,1),
                        output_shapes=[(32,)], output_dtypes=[mx.float32])[0]
        mx.eval(actual)
        values = np.array(actual).astype(np.float64)
        expected = np.array(reference)
        cases.append({"id":label,"bits":bits,"bytes_sha256":hashlib.sha256(raw).hexdigest(),
                      "reference":reference,"actual":values.tolist(),"max_abs":float(np.max(np.abs(values-expected))),
                      "equal":bool(np.array_equal(values,expected)),"source_kind":"measured-output/derived-error"})
    report = {"schema":"tc-ggml-dequant-excerpt-probe-v1","mlx":version("mlx"),"production_qualified":False,
              "scope":"selected upstream functions only, float4x4 instantiation, safe math; not GGML matmul or engine timing",
              "provenance":json.loads((asset/"PROVENANCE.json").read_text()),
              "excerpt_sha256":hashlib.sha256((asset/"dequantizers.metal").read_bytes()).hexdigest(),
              "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"cases":cases,
              "interpretation":"Mismatch witnesses need per-expression analysis; do not generalize to all GGUF types or devices."}
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
