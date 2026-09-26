# Vector QKV access candidate: rejected after measurement

## Final disposition

ABBA session71775 finished before compilation/GPU testing began. One GPU
unittest method passed all36 combinations (four row counts, three group sizes,
three dtypes), zero absolute error against scalar fusion. Independent screening
then used20 alternating measured pairs per geometry after four warmup pairs.

BF16 synchronized wall milliseconds (scalar / candidate):

| Rows | Threads | Scalar | Vector |
| --- | ---: | ---: | ---: |
| 1056 | 64 | 1.99556 | 2.03369 |
| 1056 | 128 | 2.01135 | 2.01562 |
| 1056 | 256 | 1.99908 | 2.00944 |
| 4128 | 64 | 6.97396 | 7.01162 |
| 4128 | 128 | 7.02440 | 7.01483 |
| 4128 | 256 | 7.01733 | 9.05894 |

No useful BF16 gain; large-shape256 regresses. FP16/FP32 also passed exact
parity but their isolated timings do not qualify the requested BF16 workload.
The candidate code, probe switches, and candidate-only test were removed;
the original scalar production header/probe were restored using targeted edits.
No production library rebuild or default change was made. Historical commands
below document the rejected experiment and are no longer supported switches.

## Original design and pending-test notes (superseded)

Source-only experiment prepared while virtual256 ABBA session 71775 runs.
Do not build or run it concurrently with that GPU experiment.

`metal/qkv.hpp` accepts optional `vector_access` (default false) and physical
threadgroup size (default128, candidates64/128/256). Production callers still
use the defaults. The loaded native library has not been rebuilt.

The candidate replaces Q/K loads and Q/K/V stores with four-element accesses,
retaining the same four consecutive channels per SIMD lane, scalar sum order,
SIMD reduction, precise inverse RMS, normalization-to-BF16 rounding and RoPE
equations. It does not quantize weights or change attention arithmetic.

Read-only local Splash references inspected this turn:

- `runtime/metal/kernels/prefill/attention_qkv.metal`: combined normalization,
  RoPE and final consumer layout specialization.
- `runtime/metal/kernels/shared/normalization.metal` and
  `runtime/metal/kernels/common/rms_inverse.h`: physical-thread multiplexing
  and fused normalization consumers. Their arithmetic is not copied blindly;
  Z-Image retains its own epsilon and reduction contract.

After the ABBA completes, build the independent probe and run:

```sh
bash tools/native/build_z_image_metal_probe.sh
TURBOCIDER_TEST_GPU=1 .venv/bin/python3 -m unittest discover \
  -s tests/native -p test_z_image_metal.py -k vector_qkv
build/native/z-image-metal-probe 1056 20 inline scalar 0 vector-qkv 128
```

New GPU test covers rows1/33/1056/4128, 64/128/256 threads, BF16/FP16/FP32,
and requires zero max absolute error against scalar fused QKV. It is written
but **not yet run**. The probe omits unrelated norm benchmarks in vector-QKV
mode. No speedup or parity claim is justified yet. If the microbenchmark is
promising, whole-model opt-in integration and parity are still required before
any default change. Seven CPU benchmark tests pass; these do not validate
Metal compilation or this QKV candidate.
