# Cancellable image imports and GPU INT8 comparison — 2026-10-01

This continues the M4 Pro laptop goal after `67563e8`. No checkpoint or
dependency is downloaded. GPU/Core ML experiments are serialized and bounded;
the numerical runtime remains unchanged unless explicitly stated below.

## Input-provider cancellation

Both Creation and Playground previously awaited an `NSItemProvider` callback
through a checked continuation and discarded the returned `Progress`. Checking
task cancellation after the callback was insufficient: a provider that never
replied kept `importing` set and the editing controls disabled indefinitely.
The previous cancellation regression explicitly finished the provider after
cancelling, so it did not cover this failure.

The shared `ItemProviderDataLoader` now propagates task cancellation to the
provider's `Progress`, resumes its waiter independently of the callback and
accepts only one completion. Creation owns its import task and exposes
“取消导入”; Playground owns its task across page changes and exposes
“取消图片处理”. Both buttons remain outside the disabled editing controls.
Pending tasks reserve the input state before their async body begins; cleanup
finishes before another import is admitted. Synchronous file/image processing
can still take time to finish: the immediate cancellation guarantee applies to
the pending provider callback, not arbitrary synchronous I/O.

Both complete Swift state regression targets built and passed:
`PlaygroundStateTests` and `StudioBehaviorTests` + `StudioStreamingQueryTests`.
They check cancellation of a provider that never calls back, input/file rollback,
pre-start cancellation, immediate retry, and a late callback while the new
provider is pending. The new import remains locked and retains the correct
image order/role data. The full Studio target also exercises Turbo parameters,
editing results, masks, model/LoRA state and worker discovery/query lifecycle.
Source hashes were identical before and after the runs. Evidence:
`outputs/cancellable-import-20261001/regression-report.json` and its build/run
logs. No model was loaded by these tests.

## CPU preparation experiment

A component profile at rows/hidden/width `128/512/768`, Hadamard group 64 and
LoRA rank 8 identified about 9 ms in three weight rotations and another 3 ms
in weight normalization. Low-rank gate/up corrections together took about
0.64 ms after their first call. These are synthetic CPU preparation spans,
not Qwen request timings or a hardware ANE trace.

An attempted FWHT change replaced two operand snapshots and two temporary
results with one snapshot and explicit NumPy ufunc outputs. All prepared FP16
bits, including their signs, matched for A/changed-B/A/base/zero cases. However,
four warmed ABBA samples per version measured median preparation at **13.269 ms
before and 13.779 ms after**. The slower candidate was withdrawn; the production
and research graph arithmetic were not changed. Read-only, strided inputs and
groups 64–256 were added to the independent CPU oracle coverage.

Evidence: `outputs/runtime-ane-preparation-profile-20261001/`, including the
original and rejected source files, component spans, comparison and test log.
This is a useful rejection of that allocation strategy, not a speedup claim.

## GPU comparison

The isolated MLX/Metal probe compares three complete synthetic SwiGLU paths:
FP16 normalized GEMM; INT8 storage dequantized to FP16 GEMM; and signed INT8 loads
with scalar int32 integer multiply/accumulate. The integer route is ordinary
Metal ALU code, **not native INT8 matrix/TensorOps acceleration**. All three
include CPU-to-MLX entry, all three dynamic weight FWHT/normalization/quantization,
original-basis gate/up/down LoRA, activation processing, synchronization and
NumPy readback. No prepared-weight cache hides setup costs.

The first tiny run exposed a report bug: the FP16 reference norm rounded its
`1e-12` denominator floor to zero, producing NaN for zero/zero comparison.
Its initial `success` field is superseded and must not be used for acceptance.
The common metric now widens both norms to FP64, validates finite metrics and
writes strict JSON without NaN. A CPU regression checks zero, large finite FP16,
nonzero-versus-zero and invalid input. The corrected tiny run and the medium
run both pass with empty stderr. Five GPU-oracle host tests pass; the separate
Core ML candidate's six host tests also passed during preparation profiling.

| Shape: rows/hidden/width | FP16 warm total | INT8 storage + FP16 warm total | Integer ALU warm total |
| --- | ---: | ---: | ---: |
| 32/64/96, group 32, rank 4 | 1.409 ms | 1.207 ms | 1.200 ms |
| 128/512/768, group 64, rank 8 | 2.064 ms | 2.864 ms | 6.405 ms |

Each median contains two warm samples in forward/reverse variant order, a
screening sample size, not production qualification. At the medium shape the
floating dequant route is about 1.39× slower and the scalar integer route 3.10×
slower than the matched FP16 control. This implementation does not justify a
product acceleration switch. The complete tiny and medium worker invocations
lasted about 0.26 s and 1.92 s respectively, excluding process startup by the shell.

First complete medium requests were 136.896 / 12.128 / 60.180 ms respectively.
These are process-first samples, not independent cold-cache comparisons: the
variants share MLX/driver caches, and the tiny rerun also reused system caches.
They show setup sensitivity but do not isolate compilation from allocation and
other first-use work, nor explain the full Qwen model's early denoising steps.

All variants passed A → changed B → A exact restoration, base and zero cases,
FP16 output boundaries and identical warm-request repeatability. Exact integer
smoke tests include signed tails, random [-128,127] data and ± extreme values;
results match an independent int64 CPU oracle. The Hadamard basis order also
matches. At the medium shape the largest relative L2 discrepancy from each
variant's own CPU rounding contract was 0.0124% / 0.0349% / 0.0190%. Against
original unquantized synthetic math, maxima were 0.0767% / 1.871% / 1.869%.
These activation errors are not image-quality metrics.

Evidence: `outputs/runtime-gpu-swiglu-tiny-20261001-r2/report.json` and
`outputs/runtime-gpu-swiglu-medium-20261001/report.json`. Each records matching
before/after source hashes and successful removal of its owned temporary
directory. No checkpoint, Core ML model or graph is exported by this GPU probe;
MLX/Metal system driver caches are outside its cleanup scope. GPU row scales,
factors and deltas are FP32, whereas the earlier Core ML graph has FP16 input
boundaries. Those earlier timings are not a paired ANE/GPU speed comparison.

Apple's [MPP programming guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
describes kernels targeting the M5 GPU Neural Accelerators. Its existence is
not evidence that this M4 Pro has the same integer matrix throughput. Likewise,
the repository's H3 TensorOps INT8 route defaults to M5; that route-selection
policy does not prove hardware inability on another chip. This comparison
reports its actual arithmetic implementation and measured local timings.


## App build and handoff boundary

The final optimized App and model-library helper compile successfully with the
Command Line Tools / macOS 15.2 SDK. The final View-only review aligns busy
guards across Creation, Playground, reference sizing, annotation, upscaling
and model pages; all state and test sources still match the passing regression
binaries. Final hashes and the exact View files changed after those suites
are recorded in `outputs/cancellable-import-20261001/final-validation.json`.

The deliverable uses `dist/release-<commit>/TurboCider.app` and
`dist/TurboCider-macOS-arm64-<commit>.zip`; its separate delivery receipt records
source identity, executable-section matching, local ad-hoc signature validation,
archive byte comparison and SHA-256. This is a local build, not a notarization.
Numerical native code is unchanged. This stage does not run another whole-model
generation; the prior actual 512² Turbo GPU generation passed at `67563e8` in
27.08 s and is recorded in the preceding App-refresh report.

The Mac was locked during this stage. Real button/layout interaction, graceful
quit/relaunch and replacement of the active App remain pending; the last
actually installed and opened version is `67563e8`. Automated state tests and
compile success must not be described as fresh real-UI acceptance. No current
draft, history or installed model was changed during these checks.
