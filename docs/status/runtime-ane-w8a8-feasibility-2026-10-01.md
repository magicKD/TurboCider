# Dynamic-weight INT8 / Hadamard feasibility

The user reauthorized testing. The isolated projection candidate has now been
exported, compiled and exercised on this M4 Pro / 48 GiB laptop, with existing
tooling and no downloaded checkpoint. The current App remains GPU-first for
Qwen Turbo. This candidate is **not** a production W8A8 backend: neither its
physical INT8 execution nor an end-to-end speedup has been demonstrated.
See [measured acceptance](2026-10-01-resumed-validation.md) for raw-data locations.

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

## Measured candidate decision

The installed tools successfully exported matching FP16 and dynamic-operand
Q/DQ controls. Converted MIL retains both runtime matrices, both INT8 Q/DQ pairs,
and FP32 scale restoration. Synthetic zero/signed inputs, A/B/A weight switching,
group-64 rotation and a finite result with an overflowing FP16 intermediate
were checked. Explicit temporary model packages/compiled graphs were removed.

| Geometry (rows / hidden / width) | FP16 total median | Q/DQ total median | MatMul preferred device |
| --- | ---: | ---: | --- |
| 32 / 64 / 96, butterfly preparation | 0.140 ms | 0.144 ms | CPU for both |
| 128 / 512 / 768, earlier dense-H preparation | 2.409 ms | 2.374 ms | CPU for both |

These small diagnostic timings include group-one preparation, prediction and
restoration; they exclude Hadamard rotation cost. Four warmed samples per
variant do not establish a production speedup. Some MatMul/QDQ nodes list NE
support, but the plans prefer CPU; scale restoration is CPU-only. A plan is not
a hardware trace. Maximum relative L2 versus original FP64 projection was
0.129% / 0.836% (small FP16/QDQ) and 0.335% / 1.107% (medium). The earlier failed
arbitrary 0.3% medium-control gate is retained in the raw evidence; it is not
silently reclassified as image-quality acceptance.

The Sylvester CPU preparation now uses butterflies, with host tests of basis
ordering, projection invariance, input ownership, zero rows and invalid scales.
It remains a reference transform, not the GPU/ANE production stager. Current
receipts explicitly set `hardware_int8_verified=false` and
`production_speedup_verified=false`; the candidate is rejected by the App
runtime manifest contract rather than silently exposed as an acceleration mode.

A complete SwiGLU/LoRA backend would still need all three projections and their
conversion/bridge costs. Gate/up may share input rotation; LoRA corrections
must be added **before** SiLU. The down-projection rotation comes after the
nonlinear hidden activation; Hadamard cannot commute through SiLU. Down-LoRA
must use the corrected hidden in the original basis, and any failed chunk must
trigger complete GPU-tail recomputation. No such complete dynamic INT8 backend,
GPU A8W8 kernel, encoder route or matched whole-image speed comparison has been
implemented or qualified by this single-projection experiment.

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

## Reproduction

```sh
.venv/bin/python -B tests/native/test_runtime_ane_int8_candidate.py
.venv/bin/python -B tools/validation/runtime_ane_int8_probe.py --output /tmp/tc-int8-small-new
# Optional bounded second geometry, not a production-size experiment:
.venv/bin/python -B tools/validation/runtime_ane_int8_probe.py --medium --output /tmp/tc-int8-medium-new
```

Use a new output directory. Run serially with other GPU/ANE work. The four host
tests passed; the final small report is
`outputs/resumed-validation-20261001/int8-fwht-final/report.json`, and the medium
report is `outputs/resumed-validation-20261001/int8-medium-complete/report.json`.
Current numerical/placement evidence supersedes the earlier source-only stage;
it does not supersede the stated production and quality limits.
