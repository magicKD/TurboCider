#!/usr/bin/env python3
"""Small actual-device affine decode precision diagnostic, not model timing.

Compares public installed primitives to independent canonical FP16 storage and
a clearly labelled simulation of v0.32.0's half scale/16 Q4 expression. Does not
infer an installed binary's commit from its version string or one observation.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quantization import affine_slice, pack_codes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import mlx.core as mx
    from affine_gemm import actual_strides, factory
    sources = [Path(__file__), Path(__file__).with_name("affine_gemm.py"), Path(__file__).with_name("affine_gemm.metal"),
               Path(__file__).resolve().parents[1] / "quantization.py"]
    (output / "source").mkdir()
    snapshots = []
    for path in sources:
        shutil.copy2(path, output / "source" / path.name)
        snapshots.append({"path": "source/" + path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    report = {"schema": "tc-affine-decode-probe-v1", "source_kind": "measured-and-derived-diagnostic",
              "recorded_at_singapore": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() + 8 * 3600)) + "+08:00",
              "mlx": version("mlx"), "loaded_core_sha256": hashlib.sha256(Path(mx.__file__).read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_snapshots": snapshots,
              "production_qualified": False, "physical_instruction_attribution": "unverified", "cases": []}
    for bits in (4, 8):
        for scale, bias, label in ((2 ** -24, 2 ** -14, "tiny-scale-normal-output"), (.007, -.123, "normal-scale-cancellation")):
            codes = [((1 << bits) - 1) if label.startswith("tiny") else index % (1 << bits) for index in range(32)]
            scale, bias = np.float16(scale), np.float16(bias)
            q = mx.array(np.frombuffer(pack_codes(codes, bits), dtype="<u4").copy().reshape(1, -1))
            s, b = mx.array([[float(scale)]], dtype=mx.float16), mx.array([[float(bias)]], dtype=mx.float16)
            canonical = (np.array(codes, dtype=np.float64) * float(scale) + float(bias)).astype(np.float16)
            # Simulation only: division is explicitly rounded to FP16 first.
            simulated = np.array([np.float16((float(np.float16(scale / np.float16(16))) * (code * 16)
                                              if bits == 4 and index % 2 else float(scale) * code) + float(bias))
                                  for index, code in enumerate(codes)], dtype=np.float16)
            decoded = mx.dequantize(q, s, b, group_size=32, bits=bits)
            mx.eval(decoded)
            decoded_numpy = np.array(decoded).reshape(-1)
            measurements = []
            for rows in (1, 33):
                x = mx.ones((rows, 32), dtype=mx.float16)
                direct = mx.quantized_matmul(x, q, s, b, group_size=32, bits=bits)
                custom, guard = factory(mx, bits, 32, 32)(x, q, s, b)
                mx.eval(direct, custom, guard)
                measurements.append({"rows": rows, "packed_result": float(np.array(direct)[0, 0]),
                                     "custom_result": float(np.array(custom)[0, 0]),
                                     "guard_pass": bool(mx.all(guard == 17).item()),
                                     "all_rows_equal": bool(mx.all(direct == direct[0]).item())})
            report["cases"].append({"bits": bits, "label": label, "scale": float(scale), "bias": float(bias),
                                    "canonical_fp16": canonical.astype(np.float64).tolist(),
                                    "installed_dequantize_fp16": decoded_numpy.astype(np.float64).tolist(),
                                    "simulated_half_division_fp16": simulated.astype(np.float64).tolist(),
                                    "simulated_half_division_sum_rounded_fp16": float(np.float16(np.sum(simulated.astype(np.float64)))),
                                    "canonical_equal_installed": bool(np.array_equal(canonical.view(np.uint16), decoded_numpy.view(np.uint16))),
                                    "canonical_sum_rounded_fp16": float(np.float16(np.sum(canonical.astype(np.float64)))),
                                    "matmuls": measurements})
    rng = np.random.default_rng(29)
    x_numpy = rng.normal(0,.3,(33,96)).astype(np.float16)
    q,s,b = mx.quantize(mx.array(rng.normal(0,.1,(41,96)).astype(np.float16)),group_size=32,bits=4)
    mx.eval(q,s,b)
    values = [x_numpy,np.array(q),np.array(s),np.array(b)]
    views = []
    for value in values:
        padded = np.zeros((value.shape[0],value.shape[1]*2),dtype=value.dtype)
        padded[:,::2] = value
        views.append(mx.as_strided(mx.array(padded),shape=value.shape,strides=(value.shape[1]*2,2)))
    mx.eval(views)
    decoded = affine_slice(values[1].tobytes(),rows=41,physical_columns=96,bits=4,first=0,count=96,
                           scales=values[2].tolist(),offsets=values[3].tolist(),group=32)
    reference = np.einsum("mk,nk->mn",x_numpy.astype(np.float64),np.array(decoded,dtype=np.float16).astype(np.float64),optimize=False)
    custom,guard = factory(mx,4,32,64,coalesced=True,pad=8)(*views)
    public = mx.quantized_matmul(*views,group_size=32,bits=4)
    mx.eval(custom,guard,public)
    report["strided_operands"] = {"actual_strides_elements": {name:actual_strides(mx,value) for name,value in zip(("X","Q","S","B"),views)},
                                   "shape_mnk":[33,41,96],"custom_guard_pass":bool(mx.all(guard==17).item()),
                                   "custom_relative_l2":float(np.sqrt(np.sum((np.array(custom).astype(np.float64)-reference)**2)/np.sum(reference**2))),
                                   "public_relative_l2":float(np.sqrt(np.sum((np.array(public).astype(np.float64)-reference)**2)/np.sum(reference**2))),
                                   "source_kind":"measured-output/derived-error"}
    report["interpretation_limit"] = "A formula/dispatch precision witness, not all-format accuracy, physical kernel trace or evidence that the installed wheel equals a particular source commit."
    report["source_stable_at_end"] = all(hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"] for path,item in zip(sources,snapshots))
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return int(not report["source_stable_at_end"])


if __name__ == "__main__":
    raise SystemExit(main())
