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
layout/reuse has since received the scoped preparation control below.
Neither route has been enabled in the packaged App by this component campaign.

## Table preparation after a common M2 SiLU

Four fresh processes ran in A1/B1/B2/A2 order. Each first evaluated the
actual exported `tc::silu` on the same contiguous BF16 `[2,4096]` input.
It then evaluated the complete BF16 bit-pattern domain as `[6,12288]`,
using either a gate view with row stride24576 (A) or contiguous row
stride12288 (B). Both contain the same73728 ordered values. Packing,
GPU table materialization and complete table read/SHA were timed as well.

| Run | Common M2 (ms) | Generation (ms) | Complete table preparation (ms) | M2 + table (ms) |
| --- | ---: | ---: | ---: | ---: |
| A1, strided | 7.960125 | 0.477500 | 0.805000 | 8.765125 |
| B1, contiguous | 5.694875 | 0.376542 | 0.619625 | 6.314500 |
| B2, contiguous | 5.408792 | 0.355416 | 0.627708 | 6.036500 |
| A2, strided | 5.380208 | 0.498500 | 0.760125 | 6.140333 |

All four256KiB tables matched the qualified SHA above. Each process
executed exactly two actual `tc::silu` calls and zero ConvRot projections
or A8 operations; no model or weight payload was read. Both layouts had
sub-millisecond table preparation in this context. The earlier99.507375ms
observation did not include the same common M2 preparation and is not a
matched denominator for a speedup or a JIT-cache causal claim. Common M2
is reported separately and in the combined total; it is not a free warmup.
The production Time-condition route must still be measured with any table
integration inside the complete request.

All four commands and samplers completed, with normal pressure and no
new interval swap-in/out; existing swap use remained1445.19MiB. The100ms
sampler captured only2–6 samples per short process, so its observed
footprints cannot bound brief GPU allocation peaks. The original reserve,
growth and swap guards were unchanged. The initial host logging-string
compile failure and its one-character syntax fix are retained separately.
Evidence:
`gateup-swiglu-fusion/lut-rounding-probe/prep-layout-reuse/host-only-log-syntax-fix-attempt-1794095692362029000/ABBA-readback-summary.json`.

## Actual native API and streamed trajectory passed

The isolated RNE implementation now builds as native runtime
`9af4581d…`, with the qualified FF-only shader unchanged. The build's684
source inputs match the fresh library; the actual ID getter and loaded
library path were checked. The new library/PSO cache is separate from
the original kernels, and the request-frozen fusion flag remains off by
default. Shipping artifacts still use the previously qualified runtime.

The component host now calls the actual exported native typed-GU and
`tc::silu` reference, rather than a local SiLU copy. Its corrected include
path compiled successfully. Default unset/0 completed the real original
Weights/H256 path without compiling or dispatching the new GU kernel.
The20 policy rejections,7 Weights rejections (including an actually bound
LoRA) and3 primitive input rejections passed.

The first broad numerical case stopped at its unchanged1536MiB admission:
free100MiB plus non-speculative file5485MiB minus the4GiB system reserve
provided about1489MiB. No candidate numerical comparison ran; neither
owner nor candidate H256 checks are qualified by this attempt. The sampler
completed11 samples, max gap110.03ms, observed footprint267617024B and
zero new swap-in/out. The earlier host include error is also preserved.

A follow-up delays the96.09375MiB original W/scale CPU copies
until the owner checks, which are their only consumers. It preserves the
39-case matrix,24 dynamic cases,4 owner checks, counters and all original
1536/1280MiB admissions. This removes an unnecessary logical lifetime;
it does not establish a measured RSS reduction of that amount.

The complete32-layer,8-step streamed trajectory subsequently passed under
the production4GiB reserve and512MiB per-step growth guard. All16 retained
velocity/Euler BF16 payloads,131072B each, matched the independent295
runtime reference byte for byte, including the old final latent. Actual
counts were1536 logical projections,1536 A8 operations and256 fused GU
dispatches over262912 fused rows. Only the first step compiled the single
new GU library and Large PSO. All256 source reader/fill fences, compact KV
owners and the194MiB Time-source release/640KiB compact frames passed.
This driver used retained text/noise and excludes encoder/VAE/PNG and
performance qualification.

The sampler completed123 samples with max gap110.028ms and observed
footprint916243416B. New interval swap-in/out were0, while system-wide
compression was7984840704B and decompression1044201472B; existing swap use
remained1445.19MiB. This is not a no-compression observation or an
attribution of all system activity to the driver. Native verification
actually read7256787829B and matched the original DiT SHA, with0 receipt
cache hits. Evidence is in the isolated worktree:
`outputs/convrot-fused-gu-draft/trajectory/attempt-1794096370638669000/`.

After that real-model run, a fresh system snapshot gave about4169.7MiB of
system-side headroom under the same reserve/credit formula. The deferred
copy component host then compiled and completed its entire matrix once:
63 actual FF comparisons,907714560B per arm, were byte-exact; all4 source
owner checks and3 actual compiled Weights/H256 checks (M33/1024/1048,
51732480B per arm) passed. Matrix counters were130 logical/A8 operations
and65 fused calls over37001 fused rows, with one new library/two PSOs.
The earlier defaults and all rejection checks also passed. Its sampler
completed40 samples, max gap110.037ms, observed footprint757810304B,
compression0B, decompression6160384B and new swap-in/out0. These are
actual-API numerical/ownership results, not whole-image speed or memory
improvement claims. The previous resource refusal remains retained.
Evidence:
`outputs/convrot-fused-gu-components/defer-owner-copies-attempt-1794096053248110000/execution/`.

The first complete CLI correctness campaign finished control0 with exact
text/noise/latents/pixels and the original full PNG. Afterward the system
pressure flag stayed elevated beyond its60s prelaunch readiness deadline,
so control1 was never launched and the campaign remains failed/unqualified.
Its observed footprint was4174890592B, system compression5717835776B,
decompression2102181888B and new swap-in/out0. A separate controller changes
only that readiness deadline to180s; the failed record is preserved, and
normal pressure, native reserves/growth, sampled process cap, zero new
swap and all numerical gates remain unchanged. Readiness waiting is
recorded separately from the native Popen-through-wait timer.

That separate controller completed a fresh control0/control1 correctness
campaign. Both complete PNGs and all four full tensor payloads (text,
initial noise, final latent and F32 pixels) match the original references.
Control1 reports all256 fused calls,1536 DiT A8 operations and exactly
one GU library/Large PSO compilation. Both arms retain the same prefix0,
271589376 resident-source bytes,55880953856 streamed-source bytes and
256 fills. Their complete native-command times were10.397327s and
7.866168s, but this two-run correctness campaign is not speed evidence:
system compression was7264747520B versus1805828096B and file/driver
caches were not flushed. Both samplers completed with0 new swap-in/out.
The separate dump-free ABBA/BAAB complete-request comparison remains the
performance gate. Correctness evidence:
`fused-gu-whole-image-long-wait/correctness-attempt-1794096847852555000/`.

## Complete-request performance did not qualify promotion

The separate eight-process, dump-free ABBA/BAAB campaign completed. Each
candidate actually executed256 fused calls and compiled one GU library
and one Large PSO. All original W8A8 counts, complete PNGs, source receipts,
prefix0/fill/read-byte signatures and memory guards passed. The timer covers
native process launch through exit, including request, PNG/result output
and cleanup; it excludes sampler/timer startup and recorded readiness waits.
No OS/driver cache flush was performed.

| Index | GU | Complete CLI (s) | Denoise (s) |
| --- | ---: | ---: | ---: |
| 0 | 0 | 9.913343333 | 7.786798250 |
| 1 | 1 | 7.551347791 | 5.432296583 |
| 2 | 1 | 7.664601750 | 5.561640916 |
| 3 | 0 | 7.498239375 | 5.392233209 |
| 4 | 1 | 8.302063875 | 6.159119833 |
| 5 | 0 | 7.840152917 | 5.680542375 |
| 6 | 0 | 7.465453334 | 5.342293625 |
| 7 | 1 | 7.737048208 | 5.607164000 |

ABBA complete-command medians were8.705791354s native and7.607974771s
fused (1.144298×), but BAAB was7.652803126s native and8.019556042s
fused (0.954268×). The reverse window regressed about4.8%. This fails
consistent whole-request speed qualification, despite exact numerics and
the earlier precompiled component gain. The first native run is retained;
removing an inconvenient sample would not change that acceptance decision.

Observed process-tree footprints ranged3663808264–4275832536B. Every
sampler completed with0 new swap-in/out; system compression ranged
560447488–3280748544B and decompression144490496–856670208B. These
varying system observations do not establish process-specific causality.
Text/VAE timing stayed much narrower than denoise. GU library/PSO timing
was not instrumented, so cold compilation is a hypothesis rather than a
measured explanation. The local CLT installation has no `xcrun metal`,
which also prevents assuming an immediately usable offline-metallib path.
The candidate remains opt-in/default-off, and shipping App artifacts are
unchanged. Next work must measure preparation and full-request costs,
including any overlap inside the request rather than moving it outside
the timer. Evidence:
`fused-gu-whole-image-long-wait/performance-attempt-1794096984908139000/`.
