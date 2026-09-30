# Dynamic-weight INT8 / Hadamard feasibility

> Follow-up: testing was reauthorized and focused checks were executed. See
> [resumed acceptance](2026-10-01-resumed-validation.md). The source-only
> notes below describe the earlier stopped-test milestone.

Testing remains stopped. This stage reads source and official interfaces, and
adds an offline candidate exporter; it does not export/compile a Core ML graph,
load a checkpoint or run a prediction. The delivered `c4caaa2` App is unchanged.
The preceding source milestone `a301d64` handles failed-runtime resource release.

## What the inspected interfaces establish

The installed coremltools package metadata reports **8.3.0**. Its iOS 17 MIL
`quantize`/`dequantize` definitions require constant scales and zero points;
`constexpr_blockwise_shift_scale` additionally requires constant weight data.
The iOS 17 `matmul` definition accepts FP16/FP32/INT32 operands. These are MIL
representation constraints, not proof that a compiler cannot fuse Q/DQ into
integer kernels. Apple's published [Q/DQ definitions](https://apple.github.io/coremltools/_modules/coremltools/converters/mil/mil/ops/defs/iOS17/quantization_ops.html)
and [blockwise compression definition](https://apple.github.io/coremltools/_modules/coremltools/converters/mil/mil/ops/defs/iOS18/compression.html)
describe those constant parameters.

The local macOS SDK's `MLMultiArray.h` exposes FP16, FP32, FP64 and INT32
interfaces, with no INT8 multi-array entry in that header. This observation is
scoped to the SDK used here, not all future Core ML versions. RuntimeGraph
currently requires FP16 features and fills FP16 IOSurface slots.

Apple documents possible ANE latency improvements from quantizing both weights
and activations on M4-class hardware. Its weight-only compression API preserves
floating-point compute semantics. Neither statement establishes support or a
speedup for this particular dynamic-weight graph.
[Activation quantization](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html),
[weight-only API](https://apple.github.io/coremltools/source/coremltools.optimize.coreml.quantization.html).

Consequently, applying the existing constant-weight compression tool to the
runtime graph is not a complete implementation: its changing layer weights
are inputs, and each layer needs different normalization. Adding an INT8 label
to the current FP16 slots would also provide no evidence of integer execution.

## Candidate now in source

[`runtime_ane_int8_candidate.py`](../../tools/coreml/runtime_ane_int8_candidate.py)
provides a single-projection FP16 control and a two-operand INT8 Q/DQ candidate.
Both use dynamic matrices and scales; no checkpoint weight is embedded. The
normalization resolves the constant-scale restriction at the mathematical
interface, while device support remains unknown:

1. Optionally rotate each contiguous group of input channels in both matrices
   by the same normalized Sylvester Hadamard matrix, `H`. Group one is identity.
2. Divide each rotated row by its own FP16-representable absolute maximum.
   A zero row uses scale one. Reject nonfinite data, scale overflow and scale
   underflow instead of silently clipping these inputs.
3. Apply constant-scale `fp16(1/127)` INT8 quantize/dequantize to each normalized
   operand, then MatMul. The control omits only those Q/DQ pairs.
4. Restore the row scales in FP32, then cast output to FP16. FP32 restoration
   avoids a large first scale overflowing FP16 before a small second scale
   brings the final value back into range. Final output finiteness still needs
   a check by any future caller.

In exact arithmetic, `(X H) (W H)^T = X W^T` because `H H^T = I`. The candidate
adds rotation, normalization, FP16 boundaries and quantization errors, so this
identity is not an accuracy result. The CPU preparation function is a reference
implementation, and its cost must be counted. It is not a fast production
stager. The existing ConvRot checkpoint ordering differs from Sylvester;
already-rotated ConvRot codes must not be passed as dense weights here.

The manifest uses `research_dynamic_weight_projection`, marks arithmetic
unverified and runtime compatibility false. The current native RuntimeGraph
requires `runtime_weight_fp16`, so this candidate cannot enter the App route
through an otherwise valid-looking manifest. Export is an explicit offline
operation which uses the existing artifact writer with a program-factory
argument. It disables the converter's automatic FP16 rewrite so the explicit
FP32 restoration is retained in MIL. The normal exporter retains its existing
program factory and FP16 conversion setting.

## Required next evidence

There is no exported graph or timing result yet. Before expanding to Qwen FFN:

- Export only the small matching FP16/QDQ projections with existing tooling.
  Inspect MIL to establish that both matrix operands remain runtime inputs and
  both Q/DQ pairs survive; do not treat conversion success as ANE eligibility.
- Inspect compute-device plans for projection **and** restoration. Apple's
  [MLComputePlan](https://developer.apple.com/documentation/coreml/mlcomputeplan-85vdw?language=objc)
  exposes preferred/supported devices and estimated costs. These are planning
  evidence, not a hardware trace or proof of INT8 arithmetic.
- On small synthetic matrices, compare original unrotated FP32, prepared FP16
  control and Q/DQ output; use zero rows, signed values, changed scales and
  A/B/A weight switching. Reject nonfinite output. Inspect device execution
  and time the complete preparation/prediction/restoration path, including the
  FP32 restoration placement, before considering production dimensions.
- If that gate passes, build the complete SwiGLU candidate. Gate/up may share
  input rotation. LoRA corrections must be added to their projections **before**
  SiLU; down-projection rotation applies after the nonlinear hidden activation.
  Hadamard cannot simply commute through SiLU. Preserve down-LoRA correction
  in the original hidden basis and full GPU fallback after any failed chunk.

These are future checks, not commands run in this stage. Small component tests
require lifting the current testing stop; full-model tests need separate scope.
No new model download is needed. Production integration, GPU A8W8 kernels,
encoder coverage and a matched sparse laptop benchmark remain open.

## Scheduling and disk boundary

The current runtime FFN partitions token rows, stages one layer's weights ahead
of GPU attention, and overlaps ANE tail with GPU head. Frozen FFN artifacts
instead partition intermediate channels. Neither choice is universally faster:
compare whole-block time including staging, rotation, bridge/restore and memory
contention. Do not import previous M4 Max timings as M4 Pro evidence.

A one-slot runtime avoids a graph per checkpoint layer. Double buffering would
increase retained memory and needs a demonstrated exposed staging bottleneck
before implementation. Dynamic INT8 eligibility should be established before
adding more graph slots or an encoder route.

The candidate exporter still writes a source package and compiled graph to its
explicit destination. Native runtime snapshots and source artifacts are also
disk-backed; model-dependent weights are runtime inputs. Private-lease cleanup
does not control Core ML's system caches. Zero disk activity or zero remnants
after every failure is not established by this design.

## Static validation only

Both Python sources were parsed/compiled as source without importing coremltools
or executing numerical preparation, conversion or prediction. No functional,
numerical, placement, performance or image-quality acceptance is claimed.
