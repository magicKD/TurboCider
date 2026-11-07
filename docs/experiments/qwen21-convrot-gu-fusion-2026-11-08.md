# Qwen 2.1 ConvRot paired GU fusion: numerical qualification

The shipping pure-GPU ConvRot path remains the qualified original W8A8
TensorOps implementation. This experiment targets the DiT's original I8
`gate_up` matrix (N24576/K4096, paired width12288), without bias or LoRA.
It does not establish encoder fusion, whole-model speed, or an App change.

## Paired fragment coordinates

The prototype maps the existing TN2 operand fragments to matching gate16 and
up16 rows while retaining the original I32 accumulators, K traversal, A8
quantizer, scale order, threadgroup count and barriers. Stage1 still publishes
the complete original GU tensor. It passed84 full three-arm byte comparisons,
32 full-A8 CPU controls and768 sampled integer dots,20 dynamic W/scale replays
and6 source-owner retirement controls. These are component results, not an
FF-only or whole-image qualification.

Local evidence is under
`outputs/qwen-convrot-gpu-fusion-20261107/gateup-swiglu-fusion/attempt-1794085148793610000/stage1/`.
The independently preserved first stage2 failure was a missing BF16 header
helper at Metal compilation; no numerical verdict was obtained by that run.

## Direct typed epilogue rejected

The next attempt changed only the missing complete BF16 header context. It
compiled, then failed the first synthetic M33/seed20261107 input with the real
GU weights. The actual FF-only output differed in133687 of405504 BF16
elements (32.9681%;134222 bytes):132382 differed by1 ordered BF16 ULP and1305
by2. The first mismatch was original1.3125 versus candidate1.3046875.

The diagnostic five-plane kernel's gate/up/SiLU/final outputs were identical
to their references; its independently published sigmoid differed in139
bytes. The original complete compiled FF and independently tapped FF were
also identical. Consequently the successful diagnostic final plane does
not qualify the actual compact final-only kernel.

CPU inspection of the saved original SiLU and up payloads, followed by one
F32 multiply and BF16 RNE, reproduced every original final byte. Multiplying
the independently published reference sigmoid by gate instead differed
from the actual compound SiLU at139 elements. A replacement must preserve
the compound function, rather than infer it from a standalone sigmoid tap.
Source and metadata inspection found no N/grid/stride parameter mismatch;
it was not an actual GPU argument capture or proof of a compiler cause.

The rejected attempt and its read-only counterexample analysis are preserved
under `gateup-swiglu-fusion/header-context-attempt-1794087327743238000/`.
No speed experiment or deployment was performed for this rejected design.

## Independent conversion/multiply control

A small probe used seven independent Metal entries under the historical
default MathMode, observed as Fast, and Metal4.0. Each candidate final-only
entry had its own output and PSO; it did not publish diagnostic intermediate
values. The baseline first completed a separate global BF16 store, then a
separate multiply. Inputs covered65536 high16 patterns with eight low16
patterns and independently permuted second operands,524288 pairs total.

Ordinary local BF16 casts and dynamic device identity-LUT final-only routes
both matched all baseline bytes. Plain integer RNE produced16 final
differences; its separate cast diagnostic differed at4076 of1048576 inputs.
All4076 were original F32 subnormals. Across all4092 original F32 subnormal
inputs, the native store produced signed zero; normal and nonfinite cast
domains had no difference. No NaN payload masking was applied.

This proves a scope limit: the small pointwise native-cast control passes,
while the preceding GEMM-context compact kernel fails. It neither identifies
the failed GEMM's internal values nor qualifies a new full SiLU recipe.
Integer RNE must also retain the observed native conversion's subnormal
behavior before it can serve as a replacement boundary.

The initial host compilation failed on a const-only logging API call; it was
preserved and no GPU ran. The new, minimal host-only fix completed successfully
with7 dispatches,14.25MiB logical owners, sampled peak17678936B and cache0.
Existing system swap allocation was1445.19MiB before and after; the sampled
interval's new swap-in/out were both0. This is not a zero-total-swap claim.
There is no performance conclusion.

Evidence:
`outputs/qwen-convrot-gpu-fusion-20261107/bf16-rounding-barrier-probe/host-only-fix-attempt-1794089792510534000/`
contains the actual outputs, comparison, source, loaded-library binding,
process/exit/memory receipts and `READOUT.json`.

## Next candidates and acceptance

Two source-only hypotheses remain: preserve the observed native conversion
with explicit integer rounding at the dequant boundary; or index a runtime
table made from the actual compiled BF16 SiLU. Neither is qualified here.
The LUT adds96MiB of logical reads at M1024 while removing96MiB of GU
write/read, so there is no net logical-byte saving. Preparation, ownership,
cache effects and complete consumer latency must all be counted.

New component qualification must compare the actual FF-only output, include
M1024 decode and current short-prompt M1048 prefill, nonfinite/scale edge
cases, dynamic input owners and replay. M1056 is a historical32-text
fixture; M33/65/257 are synthetic tails. Component correctness must precede
complete image/trajectory equivalence and whole-request speed measurements.
The reusable mechanism and failure conditions belong in
[fusion and layout](../../techniques/metal/fusion-and-layout.md).
