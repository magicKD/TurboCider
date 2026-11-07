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

## Integer dequant boundary: component qualified

A new candidate changes only G/U's dequant conversion: F32 exponent0 becomes
signed zero, NaN becomes the observed canonical0x7fc0, and remaining values
use integer BF16 RNE with real low-bit discard. The original Sigmoid, SiLU
and final multiply expressions, A8 Safe profile, I32/K traversal and historical
GEMM MathMode remain unchanged. This is an experimentally qualified recipe,
not proof of a compiler-internal cause for the preceding failure.

The new FF-only PSO passed63 complete byte comparisons,907714560 bytes per
arm:39 core cases and24 dynamic W/scale/restoration cases. The39 necessary
G/U/actual-compound-SiLU/final diagnostic comparisons and4 eager/compiled
source-owner retirement controls passed. The core matrix covers M33/65/257,
M1024 decode, current short-prompt M1048 prefill and historical M1056, two
normal seeds, zero/ties/extreme/nonfinite inputs and three scale-edge cases.
Projection/A8/row/padding counters match the matrix-derived expectations.

The independent standalone sigmoid tap remains a different scope:21 cases
differed in65698 bytes from the compound diagnostic. These differences and
their complete payloads are preserved. Neither the sigmoid tap nor a
diagnostic final plane replaces the actual FF-only output gate; no NaN or
signed-zero masking was used.

The successful attempt used101 process samples, max gap110.035292ms,
peak sampled physical footprint1925974008B, cache0, no guard denial and0
new swap-in/out. Existing swap allocation remained1445.19MiB. Prior host
logging compilation failures were preserved separately; their GPU stages
never ran. The successful shader SHA is
`bd6f7d0cad515fee65b870f7c968029688927c57018bcc3070bb47be793fadab`.
Evidence: `gateup-swiglu-fusion/integer-dequant-barrier/host-only-braces-fix-attempt-1794091126902446000/execution/`.

A separate CPU ablation of the old saved M33 failure also rejected an overly
simple explanation. Removing only the SiLU rounding and multiplying the
saved gate, published compound sigmoid and up still differed from the old
compact output in102816 elements. These published taps do not reveal the
compact kernel's internal values. The analysis is under
`gateup-swiglu-fusion/cpu-precision-ablation-1794091091545351000/`.

## Actual SiLU table: domain and component qualified

The independent LUT proposal generated its256KiB runtime table from the
actual compiled compound SiLU. It passed12 layouts covering all65536 BF16
patterns,64 full-byte checks,36 binary consumer controls,3 dynamic table
mutation/restoration checks and1 consumer-owner check. The domain executable
used an unusable fixture sentinel and no model/weight payload reader. Its
physical fused-GEMM call count was0. This qualifies the unary/table consumer
scope alone, independently of the integer candidate.

The table SHA is
`1149d706cc547a4c155a2f5f38f5b1a0e1f3b6e7a62080006f2dbf002d38f8cc`.
Evidence: `gateup-swiglu-fusion/lut-rounding-probe/host-only-type-fix-attempt-1794091064390937000/domain/`.
Its preceding host signed-comparison compilation failure is retained.

The same frozen host subsequently passed39 actual compact-GEMM FF cases,
including M1024/M1048. All complete outputs matched the original whole FF;
the independent native-GU/LUT-consumer and2D/3D consumer-layout controls also matched.
The physical counters were117 native projections/A8 and39 fused calls,
23003 fused rows and25664 padded rows. A separately authorized replay
passed18 dynamic four-input cases and2 source-owner retirement controls.
Six changed-table cases compare an explicit LUT consumer counterfactual;
they do not assert equality to the unmodified original nonlinear function.
All eight fresh input Data owners expired while retained outputs remained
exact, without replay. Boundary/replay physical peaks were913212592B and
1071973576B; both had0 new swap-in/out and unchanged existing swap use.
Their `boundary/` and `replay/` receipts remain component evidence only.

The LUT adds96MiB of logical reads at M1024 while removing96MiB of GU write/read,
so there is no net logical-byte saving. Cache behavior, preparation and the
complete consumer latency still determine its value.

## Remaining acceptance

Neither component result changes the shipping App. Compare complete
GU-to-FF consumers before integrating a new native primitive, then recheck
the actual H256/Weights/streamed dynamic arguments, all real model steps,
final image and whole request. A component speed result will not qualify
the full FFN, encoder, full model, cold request or another device.
M1056 remains a historical32-text fixture; M33/65/257 are synthetic tails.
The reusable mechanism and failure conditions belong in
[fusion and layout](../../techniques/metal/fusion-and-layout.md).

## Three-arm component performance passed

A separately frozen executable compared the native typed GU plus original
compiled SiLU×up with the qualified integer and LUT FF-only kernels in the
same process. Both original per-call A8 quantization and required copies
were inside the monotonic wall-clock call/eval/synchronize window. H256,
down projection, source I/O and complete FFN/model/request were excluded.
The two fixed alternate BF16 inputs and original W/scales were shared.

The native/RNE and native/LUT pairs each used ABBA then BAAB, three batches
per window and20 calls per block. Candidate pair order reversed in BAAB.
All1920 timed calls had an actual projection/A8/row receipt. Twelve
prevalidated output records and32 retained actual last-timed owners passed
full-byte checks; no diagnostic or LUT-consumer dispatch ran in the windows.

| M | RNE ABBA | RNE BAAB | LUT ABBA | LUT BAAB |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 1.126619× | 1.120728× | 1.159365× | 1.156379× |
| 1048 | 1.109773× | 1.108525× | 1.151529× | 1.145113× |

Ratios are paired native median divided by candidate median, not whole-image
speedups. Every candidate/shape combination improved in all six batches;
all samples and their ranges are retained. For M1024, native medians were
6.335–6.356ms, RNE5.639–5.653ms and LUT5.482–5.495ms. For M1048 they were
7.041–7.078ms, RNE6.345–6.362ms and LUT6.146–6.181ms.

Actual table preparation was separately measured once: generation99.216709ms,
pack0.016750ms, GPU materialization0.110750ms and verification0.163166ms,
total99.507375ms. It regenerated the qualified table SHA from the actual
compound SiLU; it did not run the64-check domain suite. Operator windows
start with the table prepared. This one fixed cost and any additional new
shader compilation must be included in a future complete-request comparison.
Steady-state LUT superiority alone therefore does not select a cold-request
winner or prove how the App's reusable runtime will behave.

The process had150 samples, max gap110.033334ms, physical peak697320312B,
no guard denial, no new compression and0 new swap-in/out. Existing swap
allocation remained1445.19MiB. Native/RNE/LUT actual projection/A8 counts
were972/488/488, all exact. Both compile and GPU execution returned0.
Evidence:
`gateup-swiglu-fusion/integer-dequant-barrier/performance/attempt-1794092095226455000/execution/`.

These results justify preparing native opt-in integration. The no-table
integer route is being wired in an isolated worktree first, preserving
default behavior and requiring actual native API, streamed full trajectory,
image and complete-request qualification before promotion. LUT preparation
layout/reuse remains a separate optimization hypothesis. Neither route has
been enabled in the packaged App by this component campaign.
