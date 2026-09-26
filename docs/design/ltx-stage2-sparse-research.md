# LTX‑2.5 Stage‑2 sparse attention: research ledger

For a concise Chinese handoff with the current results, reproduction commands
and unverified requirements, see [交付摘要](ltx-stage2-sparse-results.md).
Sections below are chronological; later evidence supersedes earlier checkpoints.

## Scope and acceptance

Compare the existing dense implementation with Sol pooled correction,
CiderSol exact safety regions, STA-like local tiles, LVSA-like windows and
anchors, and direct block-routed VSA. Measure kernel, full attention, Stage-2,
and complete generation separately. A Stage-2 speedup of at least 1.5x with
near-dense final-video quality is the target, not an established result.

Use matched checkpoint, prompt, seed, precision, steps, frames, decoder and
residency. Report exact dimensions: existing production buckets are 768x448
and 1280x704, not literal 480- and 720-pixel heights. Include 121-frame runs;
9-frame smoke tests cannot establish temporal quality. Keep Stage-1, text,
audio self-attention and audio/video cross-attention dense in the initial
Stage-2 video-self-attention experiment.

## Decision checkpoint (study still incomplete)

Keep production defaults dense. The six routing modes are available through
explicit approximation opt-in, but none of the completed approximate
full-video configurations meets both 1.5x Stage-2 speedup and the aligned-RGB
gate. This is a result for the measured configurations, not a proof that all
sparse attention is incapable of meeting the target.

The completed 121-frame fox/seed-42 trials are summarized below. Each speedup
uses its own paired dense baseline; cross-trial differences are not direct
head-to-head measurements. A value below 1 is slower than dense. All rows
except the explicitly labeled 480p ABBA row are single measured AB pilots
after warmups.

| output bucket | candidate | Stage-2 speedup | request-wall speedup | aligned RGB gate |
|---|---|---:|---:|---|
| 768x448 | all-exact split-path control | 0.950x | 0.975x | pass |
| 768x448 | structured pooled, frame radius 1 | 1.061x | 1.012x | fail |
| 768x448 | CiderSol-like, frame radius 4 | 0.995x | 0.989x | fail |
| 768x448 | direct top-k32 | 1.042x | 0.990x | fail |
| 768x448 | pooled top-k32 (ABBA) | 1.038x | 1.017x | fail |
| 768x448 | Sol tau=0, middle step only | 1.008x | 0.998x | fail |
| 1280x704 | structured pooled, frame radius 1 | 1.203x | 1.099x | fail |
| 1280x704 | Sol tau=0, all Stage-2 steps | 1.096x | 1.048x | fail |
| 1280x704 | pooled top-k64 | 1.173x | 1.079x | fail |
| 1280x704 | pooled top-k32 + same-frame safety | 1.227x | 1.102x | fail |
| 1280x704 | pooled top-k32 + same-frame safety, middle step only | 1.066x | 1.038x | fail |
| 1280x704 | direct top-k32 + same-frame safety | 1.238x | 1.106x | fail |
| 1280x704 | all-exact split-path control | 0.922x | 0.965x | pass |

The RGB gate is reference fidelity, not a calibrated human-preference score:
mean correlation >=0.99, minimum frame correlation >=0.95, mean cosine
>=0.995, mean MAE <=5/255 and maximum motion-energy relative error <=0.15.
Semantic similarity may survive even when this strict gate fails. The
three-frame qualitative inspection below does not replace temporal, detail,
identity or decoded-audio evaluation. Full-video cross-resolution coverage,
multi-prompt/seed qualification and repeated timing remain incomplete.

## Inspected baseline (2026-09-13)

Repository revision: `2785947c10c49c7bf2086fd66b3d5f80e622bafa`.
`native/models/ltx_mlx/block.cpp` currently sends all attention through
`tc::attend`; its fused-SDPA override is not a sparse backend or Stage-2-only
switch. **Correction to the initial audit:** this is the diagnostic MLX
backend, not the default production route. `native/models/ltx_runtime` already
contains a C/Metal tiled Sol backend and Stage-2 admission via
`ltx_sol_stage2`. Sparse work belongs in that production route. Its existing
admission distinguishes Stage-1 and Stage-2 via their workload row counts.

The existing `validation/ltx-ane-and-720p-tiled-2026-09-13.json` reports a
768x448, 121-frame GPU Stage-2 baseline of 40.345626584 seconds. This is a
historical result, not a fresh matched sparse comparison. The 1.5x target for
that baseline would require Stage-2 <=26.8971 seconds. Fresh baselines must
be paired with candidates in one experiment.

## Existing native candidate rerun

Command: `./build/native/h3-mlx-vdn-attention-probe` (GPU access requires
execution outside the sandbox). Existing binary output:

```json
{"sequence":4416,"heads":4,"dim":128,"max_abs":3.05175781e-05,"relative_rmse":7.99616391e-05,"cosine":1,"reference_seconds":0.003401625,"span_seconds":0.045693042,"speedup":0.074445142,"mma_max_abs":9.47713852e-05,"mma_relative_rmse":0.0089808302,"mma_cosine":0.999959648,"mma_seconds":0.0080105,"mma_speedup":0.424645777}
```

Important limitations:

- The reference is VDN window softmax, **not LTX dense attention**.
- This is an existing binary rerun, not a source-verified fresh build.
- The source declares a larger packed input geometry; do not silently assume
  the binary represents the currently inspected source. Rebuild before reuse.
- Neither span nor MMA wins this run: approximately 13.43x and 2.35x slower
  than its window reference. Low numerical error does not imply video quality.
- The MMA implementation stages score/output in threadgroup memory and
  synchronizes repeatedly; profiling and tile sweeps are needed before using
  it as the LTX sparse foundation.

## Locally available references

| Reference | Inspected checkout revision | Relevant mechanism |
|---|---|---|
| ComfyUI-SolAttn-MPS | 45071126b0c1ee30b0e6b7103fa9d70924828ba5 | SIMD-group matrix exact tiles, pooled K and summed V, weighted denominator |
| attn-bench | b68b79dd1e04bd137e3a23d2e69c1c52d4c43e5d | Gather/masked/block-sparse comparisons |
| LongVideoSparseAttention | 100666e06026b98dfdab39036d9013e02319b479 | Structured local windows and anchors |

These checkouts already exist; additional upstream acquisition and source
comparison remain to be done. `mac-kernel-opt.md` supplies the design agenda,
not proof that its proposed kernels meet the target.

## Next experimental gates

1. Fresh-build a parameterized benchmark with LTX head dimensions and actual
   Stage-2 sequence lengths. Compare optimized dense, masked reference and
   direct sparse traversal; include routing/layout overhead.
2. Establish masked-oracle parity for every pattern, tail tiles and BF16/FP16.
   Validate pooled block multiplicity in both numerator and denominator.
3. Sweep BQ/BK, preserve contiguous KV accesses, avoid per-query KV gathers.
   Benchmark Sol/CiderSol and structured variants at matched retained work.
4. Extend the existing Stage-2-only opt-in dispatch and add trace counters proving which
   blocks/steps ran sparse; leave defaults dense until quality evidence exists.
5. Run alternating dense/candidate paired generations at both buckets. Capture
   Stage-2 latents plus decoded audio/video. Evaluate perceptual differences,
   detail, motion/flicker, identity and lip-sync over a stress-prompt set.

Status: research in progress. The sections below supersede the initial audit;
Stage-2 speedup and final-video quality remain separate acceptance gates.

## Production-kernel probe (September 13, 2026)

The repository already contains the production C/Metal Sol implementation in
`native/models/ltx_runtime/ltx_gpu.m` and `ltx_shaders.metal`. I added
`tools/native/ltx_attention_probe.c` plus its build script to measure the
complete self-attention core rather than an isolated shader. It includes split
RoPE, head packing, summary/routing passes, tiled online softmax, exact routed
blocks, output unpack and gate handling. Build with
`bash tools/native/build_ltx_attention_probe.sh`.

Fresh M4 GPU probe results (BF16 random tensors, 16 heads, head dimension 128):

| rows | tau | dense ms | Sol ms | kernel speedup | exact block fraction | relative L2 |
|---:|---:|---:|---:|---:|---:|---:|
| 4,004 | 0.00 | 11.30 | 9.18 | 1.23x | 52.28% | 0.569 |
| 4,004 | 0.25 | 11.23 | 7.95 | 1.41x | 42.76% | 0.624 |
| 4,004 | 0.50 | 11.30 | 6.91 | 1.63x | 33.99% | 0.670 |
| 4,004 | 1.00 | 11.24 | 5.08 | 2.21x | 19.75% | 0.731 |
| 16,016 | 0.50 | 161.09 | 76.60 | 2.10x | 31.55% | 0.687 |

The 4,004-row case corresponds to 704x448 with 97 frames, not the 121-frame
768x448 workload. Both initial probes use 16 heads, while the actual LTX model
config specifies 32 video heads. The 16,016-row case is a generic stress shape,
not the requested 720p workload. These results show attention-core performance
potential but cannot establish model quality. The probe intentionally uses
random tensors and therefore does not replace paired LTX generation. It also
exposes the current design's key trade-off: higher `tau` removes more exact
blocks and improves kernel time, while increasing approximation error.

The current implementation hard-limits the tiled Sol backend to 256 blocks
(16,384 rows), which covers the 16,016-row stress shape but must be checked
against every 720p bucket and conditioning layout before integration.

## Structured pattern API and parity

The production API now exposes an experimental `ltx_sparse_pattern` descriptor
and `ltx_gpu_self_attention_core_sparse_bf16`. The legacy Sol entry point is a
compatibility wrapper, so existing request behavior is unchanged. Modes are:

* `0`: routed Sol (pooled remote blocks plus exact routed blocks);
* `1`: structured-drop (local/anchor exact blocks only, no pooled remote);
* `2`: structured pooled (local/anchor exact plus pooled remote);
* `3`: CiderSol-style local/anchor safety region plus pooled remote routing.

The probe checks materialized structured routes against a scalar route oracle,
and for small non-multiple-of-64 shapes checks the complete BF16 output against
an independent scalar pooled/exact oracle. A 257-row, 2-head structured-drop
run passed with scalar-oracle relative L2 0.00231. Its dense-vs-candidate error
was intentionally large on random tensors, reinforcing that route parity is
not model-quality parity.

The completed 32-head/121-frame-shape matrix is saved to
`outputs/ltx-sparse-research/native-core-matrix.json`, including probe binary,
shader, host source hashes and every command. It contains 20 passing oracle
cases (single token, tail tiles, frame-crossing tiles, all-exact limit) and
12 performance cases. Production output buckets correspond to 5,376 and
14,080 Stage-2 tokens. The current route is a union-of-frames block pattern;
it deliberately retains whole boundary tiles and is not an exact tokenwise
temporal mask or a reproduction of the full STA/LVSA algorithms.

### Request integration

Experimental request fields:

```json
{
  "allow_approximation": true,
  "ltx_sol_stage2": true,
  "ltx_sol_dense_edge_blocks": 1,
  "ltx_sol_dense_edge_steps": 0,
  "ltx_sparse_mode": 2,
  "ltx_sparse_radius": 1,
  "ltx_sparse_anchor_stride": 16,
  "ltx_sparse_tokens_per_frame": 336
}
```

For 768x448 the frame geometry is 336 tokens; for 1280x704 it is 880. Zero
selects contiguous-block radius rather than temporal radius. Nonzero modes
are restricted to Stage-2-only admission. Configuration is included in the
native denoiser cache key and result plan metadata. The old mode-0 Sol
defaults remain unchanged, including for zero-initialized native options.

`tools/native/benchmark_ltx_sparse_stage2.py` runs matched dense/candidate
requests and asserts byte-identical Stage-1 output and Stage-2 input. It saves
full request/result metadata and separate Stage-2/request-wall speedups.
`--pilot` is a two-warmup, one-AB-pair preliminary experiment, not an ABBA
qualification. `--skip-rgb` explicitly defers decoded-video metrics.

Fresh production native library and application builds succeeded with the
workspace Python MLX distribution supplied through `MLX_ROOT`. The new
request-contract test and the existing approximate-path contract test pass.

## First production paired video result

The pilot `outputs/ltx-sparse-research/480p-pooled-pilot/report.json` used the
real 768x448, 121-frame, 11-step, audio-enabled pipeline and mode 2
(structured pooled, radius 1, anchor stride 16). Dense and candidate had
byte-identical Stage-1 and Stage-2 input tensors, proving the comparison was
Stage-2-only. Measured results:

| metric | dense | candidate | speedup |
|---|---:|---:|---:|
| Stage-2 | 40.8775 s | 38.5164 s | 1.0613× |
| request wall | — | — | 1.0123× |

The candidate Stage-2 latent had relative L2 0.4853 and cosine 0.8820. Full
121-frame decoded comparison gave mean frame correlation 0.8783, minimum
0.8330, mean cosine 0.9760, mean MAE 21.03/255 and maximum motion-energy
relative error 0.1436; the standard RGB quality gate failed. The contact sheet
shows the fox and motion survive, but pose/detail/background differences are
visible. This is a measured negative result, not a release candidate.

FastVideo checkout `bfc9c017977d1f428d43b6e580a0ff20503d442b` was inspected.
Its current documentation says the STA pipeline is archived and that VSA is
the maintained path; its MLX VSA SIMD backend uses 64-token tiles, head
dimension 128 and one threadgroup per (head, query tile). This supports the
current Metal direction (regular tiles and no gathered K/V), but is a CUDA/MLX
reference and cannot be treated as an Apple Silicon speed claim.

Verification after integration: the full request-contract suite passed
68 tests with one unavailable Wan fixture skipped (69 total). An earlier
concurrent run hit the GPU-process lock in an unrelated Core ML test; rerunning
after generation completed passed. Shader provenance hashes now explicitly
record the local sparse-pattern adaptation.

Remaining work includes actual-QKV capture and profiling of the complete
video-self-attention path (the dense path fuses projections while the Sol path
uses separate projections/layout), direct top-k/VSA and spatial STA-like
variants, routing/tile optimization, 720p paired videos, and multi-prompt
perceptual/temporal/audio quality. Neither this pilot nor the random-core
matrix establishes the requested 1.5x Stage-2 + near-dense-quality target.

## Follow-up: 720p pilot and direct top-k

The 720p bucket pilot (`1280x704`, 121 frames) completed with Stage-2
125.2518 s dense versus 104.1559 s structured-pooled, or **1.2025x**;
request-wall speedup was **1.0986x**. Stage-1 and Stage-2 input were
byte-identical. Its Stage-2 latent relative L2 was 0.5047 and cosine 0.8745.
Full 121-frame RGB comparison gave mean correlation 0.8204, minimum 0.7642,
mean cosine 0.9723, mean MAE 23.64/255 and maximum motion-energy relative
error 0.1377; the quality gate failed. Artifacts are in
`outputs/ltx-sparse-research/720p-pooled-pilot/`.

Mode 4 now implements VSA-like pooled-QK top-k blocks plus the exact safety
region, dropping unselected remote blocks. Routing uses 1 KiB threadgroup
scores and stable rank (ties resolved by key-block index), without sorting or
KV gather. This is not the trained-gate FastH3 VSA model. Top-k and safety
regions are unioned, so retained degree can exceed the requested top-k.

The API/request parameter `ltx_sparse_keep_blocks` is required in modes 4 and 5
(1–256); other modes require zero. The new mode remains opt-in and Stage-2-only.
`native-core-topk-matrix.json` contains 29 scalar/route parity cases and
18 production-shape microbenchmarks. Small top-k cases compare rank decisions
independently in C as well as checking the attention output against the scalar
oracle. On the 720p shape, 16/32/64 kept blocks measured 6.07x/3.91x/2.30x,
but random-tensor error is very large; no full-video top-k quality claim is
established.

Next: run an all-exact sparse-path control to isolate graph/BF16 boundary
changes, profile actual projected QKV rather than random tensors, and then
evaluate top-k/per-head adaptive retention. Formal ABBA and multi-prompt
quality remain outstanding at both output buckets.

## All-exact control

`outputs/ltx-sparse-research/480p-all-exact-smoke/report.json` compares a
9-frame mode-1 request with radius 256 (every block exact) against dense.
Stage-1 and Stage-2 input are byte-identical. Stage-2 is 6.69795 s dense vs
7.00108 s candidate (**0.9567x**, 4.53% slower); Stage-2 latent relative L2 is
0.05449 and cosine 0.99852. This shows modest split-path overhead and
numerical divergence for this short shape. It suggests approximation is an
important source of the larger pooled errors, but a matched 121-frame
all-exact control is required before attributing most of that divergence to
routing. This is not an end-to-end release gate.

## 121-frame all-exact profile and real-QKV replay

`480p-all-exact-profile-121/report.json` now supplies a matched 768x448,
121-frame control. Dense Stage-2 is 40.89523 s versus 43.06990 s all-exact
(0.94951x). Stage-1 and Stage-2 input are byte-identical. The final latent has
relative L2 0.08971 and cosine 0.99598; full RGB comparison **passes** the
standard gate (mean frame correlation 0.99437). This is a much stronger
control for the failed pooled pilot than the earlier nine-frame test.

Native timing for the measured dense Stage-2:

| region | time |
|---|---:|
| video-self branch | 13.507 s |
| video text branch | 6.219 s |
| video FFN | 15.793 s |
| AV cross-attention parallel wall | 4.433 s |
| full Stage-2 | 40.895 s |

`video_self_ms` includes AdaLN, QKV, attention, output projection and residual,
not just SDPA. Parallel branch times overlap and must not be summed. Holding
all other costs fixed, even eliminating the **entire** video-self branch gives
the idealized serial estimate 40.895/(40.895-13.507)≈1.493x. This is a useful
warning about the 480p target, not a hardware-independent impossibility proof:
contention/overlap can change, and more workloads are required. The all-exact
candidate's video-self branch was 15.657 s, approximately 2.150 s slower.
This is a diagnostic pilot; development compilation overlapped part of its
warmup period, and it is not an isolated ABBA qualification run.

The diagnostic capture `qkv-480p-121-step0-block1/` contains real Stage-2
step-0/block-1 QKV (5376 rows, 32 heads, dimension 128) after projection/norm,
plus split-RoPE frequencies and actual gates. Capture uses synchronous I/O;
its generation timings must not be used as benchmarks. The capture metadata,
source request, library/shader hashes and tensor hashes are saved.

The new `benchmark_ltx_qkv_replay.py` completed 18 configurations on this
same input, with five alternating dense/candidate measurements each:

| configuration | core speedup | exact block fraction | relative L2 |
|---|---:|---:|---:|
| all-exact tiled | 0.717x | 100% | 0.000680 |
| Sol tau=-0.5 | 0.958x | 68.0% | 0.0159 |
| Sol tau=0 | 1.255x | 48.7% | 0.0266 |
| Sol tau=0.5 | 1.783x | 30.0% | 0.0446 |
| Sol tau=1 | 2.315x | 19.2% | 0.0773 |
| CiderSol temporal radius 4 | 1.007x | 64.5% | 0.0186 |
| top-k 16 | 2.466x | 19.5% | 0.374 |
| top-k 32 | 1.562x | 38.4% | 0.234 |
| top-k 64 | 0.901x | 76.3% | 0.0965 |

On this input Sol tau=0.5 is both faster and more accurate than top-k 32.
The retained pooled contribution appears important; fixed-degree top-k with
pooled remote correction is a logical next comparison. These are single
step/block attention errors, **not** full diffusion/video quality. Capture
coverage across steps, layers and prompts remains incomplete.

## Pooled top-k replay follow-up

Mode 5 retains mode 4's exact top-k/safety union but uses pooled correction
for unselected remote blocks. The same captured input above was replayed in
`qkv-480p-121-step0-block1/replay-pooled-topk.json`:

| kept blocks | core speedup | relative L2 | cosine |
|---|---:|---:|---:|
| 16 | 2.322x | 0.10729 | 0.994804 |
| 32 | 1.501x | 0.04824 | 0.998876 |
| 64 | 0.882x | 0.01186 | 0.999930 |
| 128 (all exact at this shape) | 0.699x | 0.000680 | 1.000000 |

Pooled correction substantially reduces the error of direct top-k at the
same degree, but 64 kept blocks already loses to dense on this input. The
32-block result is a candidate for full-video validation, not evidence of
1.5x Stage-2 speedup or near-dense generated quality. Nine-frame tests have
far fewer blocks and cannot validate the 121-frame sparsity regime.

## 480p 121-frame pooled top-k end-to-end result

Formal retained-engine ABBA at 768x448, 121 frames, seed 42, identical
prompt/checkpoint and component-staged residency is recorded in
`outputs/ltx-sparse-research/480p-pooled-topk32-abba-121/report.json`.
Mode 5 used 32 pooled top-k blocks, radius zero and no anchors. Radius zero
still unions the diagonal query block with the selected top-k blocks.
Dense Stage-2 was 40.8998 s median and candidate Stage-2 was 39.3949 s
median, only **1.0382x**. Request wall speedup was **1.0167x**; Stage-1 was
0.9991x. Stage-1 and Stage-2 input tensors were byte-identical.

The candidate Stage-2 video latent had relative L2 **0.2941** and cosine
**0.9564**. Decoded RGB over all 121 frames had mean correlation **0.9590**,
minimum correlation **0.9448**, mean MAE **11.41/255**, and motion-energy
relative error **7.07%**. This does not meet the near-dense quality gate.
Therefore pooled top-k 32 is rejected as a production Stage-2 policy despite
its attractive single-attention-core replay point; its speedup is diluted by
projection/FFN/parallel branches and its diffusion error accumulates.

## 720p 121-frame Sol tau=0 pilot

The matched 1280x704, 121-frame pilot is in
`outputs/ltx-sparse-research/720p-sol-tau0-pilot-121/`. Dense Stage-2 was
125.278 s and Sol tau=0 was 114.275 s, for **1.0963x** Stage-2 speedup and
**1.0475x** request-wall speedup. Stage-1 remained 0.9999x. Stage-1 and
Stage-2 input tensors were byte-identical.

The final Stage-2 latent relative L2 was **0.2927** (cosine **0.9571**).
Decoded RGB over all 121 frames had mean correlation **0.9377**, minimum
correlation **0.9101**, mean MAE **12.38/255**, and motion-energy relative
error **4.74%**. The RGB gate therefore fails. This routing threshold gives
a modest speedup at the 720p bucket without meeting near-dense quality.
This is one measured AB pair after warmups, not an ABBA qualification or
a matched cross-resolution comparison of Sol tau=0.

## 480p direct top-k 32 pilot

The direct-drop variant (mode 4, 32 selected blocks, radius/anchors zero)
was measured at 768x448/121 frames with the same prompt and seed. Dense
Stage-2 was **40.8974 s**, direct top-k was **39.2462 s**, giving **1.0421x**
Stage-2 speedup and **0.9904x** request-wall speedup. Stage-1 remained
1.0033x. Stage-1 and Stage-2 input tensors were byte-identical.

Final video latent relative L2 was **0.3557** (cosine **0.9360**). RGB over
all 121 frames had mean correlation **0.9362**, minimum **0.9199**, mean MAE
**14.56/255**, and motion-energy error **7.41%**; the RGB gate failed.
Compared with pooled top-k 32 (1.0382x, mean RGB correlation 0.9590), direct
drop has a similar measured Stage-2 time but greater RGB error. The small
timing difference is not established statistically across these separate
pilot/ABBA runs. A first/middle/last-frame contact sheet is saved as
`paired-frames.png` in the pilot directory. The sampled frames preserve the
fox and forest scene, with changed leg poses, tail and background detail;
three stills cannot establish temporal quality. Audio latent relative L2 is
**0.0482**. No decoded-audio or lip-sync quality claim follows from that metric.

## 480p CiderSol-like radius-4 pilot

Mode 3 with radius 4 and anchor stride 16 was measured at 768x448/121
frames. Dense Stage-2 was **40.9226 s** and candidate Stage-2 **41.1399 s**
(**0.9947x**); request wall was **0.9887x**. Stage-1 was **0.9937x**.
Stage-1 and Stage-2 inputs remained byte-identical. Final video latent
relative L2 was **0.2373** (cosine **0.9718**). RGB mean correlation was
**0.9720**, minimum **0.9550**, mean MAE **9.04/255**, and motion-energy error
**3.82%**; the RGB gate failed. The larger exact safety region improves over
direct top-k but still misses the quality gate and is slightly slower than
dense. Audio was not independently decoded; latent-only audio differences
must not be interpreted as lip-sync quality.

## Late-step/layer real-QKV replay

Stage-2 step 2/block 36 was captured in
`outputs/ltx-sparse-research/qkv-480p-121-step2-block36/` and replayed with
five alternating measurements per configuration. Pooled top-k 32 achieved
**1.506x** attention-core speedup, but relative L2 rose to **0.0842** and
cosine fell to **0.99667**. Pooled top-k 64 was only **0.881x** with relative
L2 **0.0243**. Sol tau=0 reached **1.298x** with relative L2 **0.0636**.
The late block therefore does not remove the quality/speed trade-off; the
single-layer Pareto point is location-sensitive and cannot by itself justify
a full-video policy.

Both replays used identical probe/shader hashes. The capture trajectory uses
the all-exact split sparse path, not the original dense path, and capture I/O
timings are excluded. Step and block both changed, so these two samples do
not separately identify step sensitivity versus layer sensitivity. The
benchmark now exposes `--dense-edge-steps` and `--dense-edge-blocks` using the
existing runtime admission guards; the defaults remain unchanged. A
middle-step-only policy can be tested with `--dense-edge-steps 1`, but has
not yet been qualified.

## 480p middle-step-only Sol pilot

Using `--dense-edge-steps 1`, only the middle Stage-2 denoising step was
allowed to use Sol tau=0; the first and last steps remained dense. At
768x448/121 frames, dense Stage-2 was **40.9073 s** and the candidate was
**40.5970 s**, only **1.0076x**. Request wall was **0.9983x**. Stage-1 and
Stage-2 inputs remained byte-identical. Final latent relative L2 improved to
**0.1542** (cosine **0.9881**), and audio latent relative L2 was **0.00560**.
Decoded RGB over all frames had mean correlation **0.9863**, minimum
correlation **0.9809**, mean MAE **6.15/255**, and motion-energy error
**2.83%**. The RGB gate still fails (mean correlation and MAE thresholds),
while the speed benefit is negligible. This pilot illustrates the trade-off:
restricting sparse execution to one middle step reduces error, but this
configuration does not deliver the requested Stage-2 acceleration. It is
not a bound on all sparse algorithms or all possible step schedules.

## QKV projection overhead check

The sparse path already uses one MPSGraph containing shared ConvRot and three
INT8 matrix projections; it does not issue three independent command-buffer
round trips. `ltx_qkv_projection_probe.c` compares this graph with the
existing packed-QKV graph using synthetic BF16 input and identical random
weights. At 5376 rows (480p Stage-2 shape), separate and packed projections
were **0.039988 s** and **0.039825 s** (1.004x), with relative L2 **0.00265**.
At 14080 rows (720p shape), they were **0.103541 s** and **0.102996 s**
(1.005x), with the same relative L2. These projection-only measurements
exclude weight packing/copying and use synthetic input. Replacing the current
sparse QKV call with the packed graph is therefore unlikely to recover the
missing Stage-2 speedup; routing, output projection, FFN and branch overlap
remain the dominant optimization targets.

## 720p pooled top-k64 pilot

At 1280x704/121 frames, mode 5 with 64 kept blocks, zero radius and no
anchors achieved **1.1730x** Stage-2 speedup and **1.0785x** request-wall
speedup (dense Stage-2 **125.2732 s**, candidate **106.7957 s**). Stage-1
was 1.0004x; Stage-1 and Stage-2 video inputs were byte-identical. Final
Stage-2 latent relative L2 was **0.3589** (cosine **0.9356**). RGB over all
121 frames had mean correlation **0.9076**, minimum **0.8350**, mean MAE
**15.59/255**, and motion-energy error **7.53%**. The RGB gate failed.

The 64-block result is not a same-resolution comparison against pooled
top-k32: that full-video trial used 768x448. At 720p there are 220 blocks,
versus 84 at 480p, so the requested retained fractions differ. Diagonal
safety blocks may also increase exact work beyond the requested top-k.
This measured AB pilot does not meet either acceptance target.

## 720p all-exact control and profile

`720p-all-exact-profile-121/` supplies the missing matched 1280x704/121-frame
control. Dense Stage-2 was **125.3051 s**, all-exact **135.9254 s**
(**0.9219x**, 8.48% more time); request wall was **0.9645x**. Stage-1 and
Stage-2 video inputs were byte-identical. Final video latent relative L2 was
**0.09976**, cosine **0.99503**. RGB mean correlation **0.99046**, minimum
**0.98190**, mean MAE **4.463/255**, and motion-energy error **1.29%** pass
the existing gate. Mean correlation is only narrowly above its 0.99
threshold; this is one prompt/seed, not general numerical qualification.

Measured Stage-2 branch profile:

| region | dense seconds | all-exact seconds |
|---|---:|---:|
| video self (includes QKV/output processing) | 56.531 | 67.148 |
| video text | 14.664 | 14.665 |
| video FFN | 40.663 | 40.661 |
| AV cross parallel wall | 11.389 | 11.389 |

Nearly all added time is in the video-self branch, but the profile cannot
separate its kernels. The serial estimate from removing that *entire*
branch is approximately 1.822x. Holding other costs fixed, reaching 1.5x
would require approximately 3.83x acceleration of the whole branch, not
merely of SDPA. These are diagnostic estimates, not hardware-independent
bounds: overlap and contention can change. This profiled AB pilot is not
an isolated ABBA qualification. Both resolutions now have passing all-exact
controls, whereas the measured approximate policies fail the same RGB gate.

## Native correctness recheck

`native-core-parity-final.json` records a fresh 38-case BF16 correctness
matrix using the per-head-enabled probe: all six modes, single-token and
non-64-aligned tails, frame-crossing windows/anchors, and top-k budgets
1/2/256. All cases passed their applicable route/scalar checks; maximum
scalar-oracle relative L2 was **0.002366** and all explicit all-exact controls
passed. The oracle tests the intended exact/pooled/drop mathematical policy,
not fidelity to dense output or final-video quality. Small-case timing is
not a production performance result. The runner now rejects existing report
paths and out-of-range run counts before execution.

## Per-head routing diagnostic

The step-0/block-1 replay was rerun with per-head metrics. High-error heads
were not stable across sampled locations: early Sol tau=0's largest squared
error contribution was head 21 (relative L2 **0.221**), while late
step-2/block-36's largest contributions were heads 14 (**0.233**) and 26
(**0.153**). Pooled top-k32's three largest late contributions were heads
17, 19 and 14. These samples do not justify a fixed protected-head list.
Per-head adaptive retention remains
unimplemented and would require bounded routing metadata plus multi-prompt,
multi-step qualification.

`head-error-cross-location.json` records the cross-location comparison.
Sol's top-six squared-error heads overlap by only 2/6, pooled top-k32 by
3/6. The early top-six lists account for 64.8%/70.3% of early error but only
27.5%/34.8% of late error, respectively. This comparison changes both layer
and diffusion step; head indices at different layers need not have the same
function. It is not a prediction of speedup or full-video quality from
selectively replacing heads with dense attention.

## Threshold-dispatch elimination and compatibility audit

Modes 1/2/4/5 do not read Sol thresholds in the route shader. The host now
skips their statistics and threshold encoders (two dispatches at production
sizes; one at small sizes). Modes 0/3 retain those computations. A subsequent
audit caught and fixed a NULL-pattern dereference introduced by the first
version of this optimization: the legacy Sol wrapper intentionally passes
NULL for its default pattern. The short-form native probe now calls that
legacy wrapper rather than substituting an explicit mode-0 descriptor.

`outputs/ltx-sparse-research/threshold-skip-parity.json` completes 41 GPU
oracle cases, including three legacy-wrapper cases. Maximum scalar-oracle
relative L2 is 0.002366. The late real-QKV replay completed all 22
configurations with nine measured iterations per configuration in
`qkv-480p-121-step2-block36/replay-threshold-skip.json`. Every reported global
error metric, exact-block fraction and per-head metric equals the preceding
three-iteration `replay-v2.json` report. This is agreement of the reported
metrics, not a byte-for-byte comparison of saved output tensors.

Removing these encoders has **no demonstrated stable speedup**. For example,
pooled top-k32 takes 23.960 ms before and 23.972 ms after; structured pooled
radius 1 takes 17.877 ms before and 17.927 ms after. Other configurations
improve slightly. These sequential runs are not an interleaved old/new ABBA
test, so differences of this size must not be attributed to the optimization.
The new top-k32 replay is 1.512x versus dense core with relative L2 0.084205;
it does not supersede its failing full-video quality result. The earlier
synthetic 1.478x/1.481x comparison likewise establishes no stable gain.

Metal access is available outside the sandbox. An in-sandbox “no Metal
device” result is not evidence that this machine lacks a GPU. Aggregate
all-exact slowdown also does not establish how much time is fixed dispatch
overhead versus the attention kernel itself; that requires isolated profiling.

## 720p late real-QKV replay

`qkv-720p-121-step2-block36/replay.json` completes 28 configurations at
14080 rows, 32 heads and head dimension 128, with three measured iterations
after two warmup pairs. This captures the same prompt/seed and step/block
indices as the 480p late fixture. The capture uses the all-exact split path,
not a native-dense activation trajectory. Generation was inadvertently
interrupted after QKV and Stage-2 tensors were written but before successful
completion; `capture.json` correctly remains `complete: false`. There is no
completed video from this capture. The independently completed replay records
fixture hashes, verifies file sizes and validates head-wise error accounting;
it is usable as attention-core evidence, not as a successful generation or
video-quality result.

| routing | requested blocks | actual exact fraction | dense/core speedup | relative L2 |
|---|---:|---:|---:|---:|
| direct top-k | 32 | 14.6% | 4.045x | 0.15666 |
| pooled top-k | 32 | 14.6% | 3.715x | 0.11886 |
| direct top-k | 64 | 29.1% | 2.358x | 0.09119 |
| pooled top-k | 64 | 29.1% | 2.230x | 0.07072 |
| direct top-k | 110 | 50.0% | 1.459x | 0.04923 |
| pooled top-k | 110 | 50.0% | 1.414x | 0.03971 |
| direct top-k | 165 | 75.0% | 0.995x | 0.02312 |
| pooled top-k | 165 | 75.0% | 0.975x | 0.01999 |

The replay runner now includes 25%/50%/75% requested block budgets alongside
the historical absolute budgets. These are fractions of blocks, not retained
attention probability mass; safety regions can increase the actual work.
For example, top-k32 requests 38.1% of the 84 blocks at 480p but only 14.5%
of the 220 blocks at 720p. Its late pooled replay error increases from
0.08421 at 480p to 0.11886 at 720p while core speedup increases from 1.512x
to 3.715x. These are different input tensors and sequential runs, not an
isolated resolution-only causal comparison. The prior 720p top-k64 full-video
quality failure still applies; no new Stage-2 or video-quality success is
established here.

## 720p pooled top-k32 with same-frame safety: full-video pilot

`720p-pooled-topk32-pilot-121/report.json` completes two warmup generations
and one measured AB pair at 1280x704, 121 frames, fox prompt/seed 42, audio
enabled. Mode 5 uses top-k32, radius 0, no anchors, and 880 tokens per latent
frame. **This retains the same-frame safety region**, unlike the block-radius-0
top-k32 core replay and the prior top-k64 full-video pilot. The 3.715x core
result cannot directly predict this configuration's Stage-2 speedup.

Dense/candidate Stage-2 times are 125.2483/102.0871 seconds (**1.2269x**).
Request-wall times are 251.1930/227.8921 seconds (**1.1022x**). Stage-1
and Stage-2 video inputs are byte-identical. Stage-2 video latent relative
L2 is **0.39394**, cosine **0.92210**. All 121 decoded frames were compared:
mean correlation **0.88943**, minimum correlation **0.81329**, mean cosine
**0.98330**, mean MAE **17.4548/255**. The RGB gate **fails**.

This candidate therefore meets neither the 1.5x Stage-2 target nor the
near-dense reference-fidelity gate. It has the largest measured Stage-2 ratio
among the current approximate pilots, but timings come from distinct paired
experiments and do not establish a statistically significant ranking. The
same-frame safety region did not suffice to meet quality acceptance here;
this is not a controlled test of its benefit versus pure top-k32.

## Sampled attention-mass routing diagnostic

`tools/native/analyze_ltx_route_recall.py` now provides a bounded CPU
diagnostic for captured, RoPE-rotated Q/K. It samples eight query blocks and
four queries per block across all 32 heads, then compares centroid ranking and
sample log-mean-exp ranking with an in-sample exact attention-mass oracle.
This is deliberately not a deployable router: it materializes exact scores,
uses sampled queries, and does not evaluate V accumulation or pooled
correction. Its purpose is to distinguish poor block selection from kernel
numerical error.

On the 720p late fixture, centroid top-k retained mean attention mass of
0.7648/0.8656/0.9310 for k=32/64/110; the sampled log-mean-exp variant was
0.7651/0.8662/0.9327. The in-sample oracle upper bound was
0.7991/0.8904/0.9466. P10 retained mass for centroid was only
0.4351/0.6484/0.8059. Thus the current centroid score leaves substantial
probability mass outside the selected blocks, especially at k=32. The small
gain from sampled log-mean-exp does not justify its additional query sampling
and routing work yet. These mass figures are diagnostic only; they are not a
quality gate and cannot be translated directly into latent or RGB error.

The diagnostic has four CPU unit tests covering RoPE identity/rotation,
BF16 rounding, tail blocks and oracle upper-bound ordering.

### Adaptive-mass budget simulation (diagnostic only)

Version 2 of the diagnostic also simulates choosing the smallest centroid
prefix whose pooled distribution estimates 90%, 95% or 99% mass. It does not
change the Metal API: a deployable version would need bounded metadata and a
regular execution schedule. On the 480p late fixture, centroid-estimated
budgets use mean exact fractions **0.399/0.512/0.717** for targets
0.90/0.95/0.99; their sampled true retained masses are **0.883/0.933/0.979**,
with P10 **0.757/0.864/0.962**. On 720p, mean exact fractions are
**0.375/0.492/0.710**, sampled retained masses **0.913/0.952/0.988**, and
P10 **0.836/0.906/0.974**. The 90% target therefore still retains roughly
40% of 480p blocks and 38% of 720p blocks on average, while its tail cases
need substantially more. This does not yet offer a quality guarantee or a
stable fixed work bound suitable for the current tiled kernel.

## Diagonal key-variance scoring experiment

The route diagnostic also evaluates `diagonal_key_variance`: centroid
scaled centroid dot-product plus one half of
`sum_d(mean_query[d]^2 * key_variance[d]) / head_dim`. This heuristic
omits cross-dimension covariance and does not bound the true logits. It uses only
per-block first and second moments and remains diagnostic-only; no request API
or production Metal route has been changed.

On 480p late, mean retained mass for k=16/32/64 changes from
0.7355/0.8725/0.9693 (centroid) to 0.7368/0.8744/0.9715; P10 changes from
0.3719/0.6474/0.9068 to 0.3671/0.6440/0.9145. On early 480p, means are
0.4688/0.6283/0.8706 versus 0.4729/0.6287/0.8710. On 720p late, means are
0.7648/0.8656/0.9310 versus 0.7680/0.8689/0.9336, with P10 changing from
0.4351/0.6484/0.8059 to 0.4502/0.6549/0.8125. The gain is small and
inconsistent at low budgets, so this candidate is deprioritized for now.
At the version-3 checkpoint, neither GPU cost nor attention-output error had
been measured; mass recall alone could not establish improvement. Version 4
below adds sampled CPU output-error measurements, but no GPU timing.

In particular, early 480p centroid top-k32 has lower sampled retained mass
than late 480p (0.6283 versus 0.8725), whereas previously measured pooled
top-k32 core relative L2 is lower early (approximately 0.048 versus 0.084).
These comparisons differ in query coverage and omit safety regions in the
mass diagnostic. They reinforce that a mass threshold is not an output-error
or video-quality guarantee. Version 4 now includes actual V accumulation,
BF16 block summaries, tail multiplicity and gate application; it remains a
sampled CPU diagnostic, not Metal parity, timing or video-quality evidence.

With output error included, late 480p centroid relative L2 is
**0.2114/0.0835/0.0265** at k=16/32/64; diagonal variance is
**0.2115/0.0917/0.0235**. Late 720p centroid is **0.1222/0.0729/0.0388**
at k=32/64/110, versus diagonal variance **0.1182/0.0672/0.0343**.
The variance term is therefore inconsistent and no new Metal mode or
acceptance result is claimed. Early 480p centroid k=32 has output error
**0.0473** despite retained mass near **0.628**, demonstrating that mass
recall alone cannot predict output error.

Each report samples eight query blocks, four queries per block and all 32
heads (1024 query/head observations). Reported output relative L2 is computed
from total squared error divided by total squared reference norm, not an
average of per-query ratios. Gate-zero observations are counted explicitly.
Both reference and candidate use FP64 attention math followed by BF16 output
rounding, gate multiplication and BF16 rounding. The mass oracle optimizes
sampled retained probability mass, not output error, and is not a deployable
router. In early 480p at k=32, variance correction increases output relative
L2 from **0.0473 to 0.0644**, reinforcing the cross-location regression.

The version-4 reports are saved beside each QKV fixture as
`route-recall-v4.json`; ten CPU tests cover the output and validation paths.

## Query-fragment cache experiment

`tools/native/benchmark_ltx_query_cache.py` generated an isolated shader
variant that loads the 16 query MMA fragments once per query tile and reuses
them for summary and exact-block passes. The production shader was not
modified. Source-shape checks, native compilation and boundary oracle runs all
passed, and reported numerical metrics matched the current shader.

On 480p late, current/query-cache median times were 50.322/50.524 ms
(all-exact), 23.8685/23.9415 ms (pooled top-k32), and 40.944/41.1065 ms
(pooled top-k64): the variant was 0.40%, 0.31% and 0.40% slower. On 720p late,
times were 322.239/324.5335 ms, 66.432/66.749 ms and 110.9265/111.6635 ms,
or 0.71%, 0.48% and 0.66% slower. These are attention-core ABBA results, not
full Stage-2 timings. The variant is rejected for integration; no stable win
was measured. Resource-pressure effects are not independently profiled.

## Acceptance audit after evidence summary v9

### Conditional summary-read experiment (September 14, 2026)

An isolated Metal shader transform was evaluated without changing production
code.  The summary kernel returns immediately for structured-drop (mode 1),
omits query-centroid work for structured-pooled (mode 2), and omits value-sum
work for direct top-k (mode 4).  All six mode parity probes and reported
numerical metrics matched the current shader.  ABBA real-QKV timings showed
about 1.037x at 720p and 1.040x at 480p for structured-drop, about 1.009x for
structured-pooled, and roughly 1.00--1.004x for Sol/Cider/top-k variants.  The
completed pooled-top-k32 rows were 1.0004x (480p) and 0.9995x (720p).
Since the best full-video candidate is pooled top-k32,
these gains do not justify production integration or alter the conclusion
that 1.5x Stage-2 with near-dense quality has not been demonstrated.

Evidence: `outputs/ltx-sparse-research/selective-summaries-v1/report.json`
is complete with 18 scalar-parity probes and 12 real-QKV ABBA comparisons
(five timed iterations per process). All compared global/per-head metrics
match exactly; this does not establish byte-identical output tensors.
The experiment retains all host allocations and dispatches, so it measures
selective reads rather than eliminating buffers or encoder overhead. No
boundary sweep or production smoke was performed for this unintegrated
variant. Targeted CPU tests: **99 passed, 1 skipped, 71 subtests**.

`outputs/ltx-sparse-research/evidence-summary-v9.json` contains 14 completed
paired-generation reports: 11 are 121-frame experiments and three are
nine-frame smoke runs without verified RGB quality. Among the 121-frame
experiments, two all-exact controls pass the RGB gate and all nine approximate
candidates fail it. Only the 480p pooled top-k32 report uses measured ABBA;
the other ten are AB pilots. Five `report.json` files are omitted: three
incomplete earlier paired attempts and two completed query-cache experiments
whose schema is intentionally not a paired-generation schema. The incomplete
720p QKV capture is also not a successful generation and is not in this table.

| requirement | authoritative evidence | audit result |
|---|---|---|
| Compare Sol/CiderSol/structured/top-k Metal strategies | mode 0–5 dispatch, native oracle matrices and real-QKV reports | implemented as documented heuristics; not full STA/LVSA/VSA paper reproductions |
| Integrate into TurboCider without changing dense defaults | request-contract tests and production native build | verified for the tested contract; explicit approximation opt-in retained |
| Compare 480p and 720p against matched dense | paired 768x448 and 1280x704 reports, 121 frames | both buckets covered; output sizes are not literal 480/720 heights |
| Distinguish core, Stage-2 and end-to-end times | probe timing, Stage-2 and request-wall fields | covered; no core-to-video speedup substitution |
| At least 1.5x Stage-2 and near-dense output together | paired Stage-2 timing and decoded RGB gates | not achieved in measured configurations; highest pilot ratio 1.2269x also fails quality |
| Broad spatial/temporal/identity/audio quality from design section 29 | all-frame RGB metrics, motion-energy comparison and audio latent checks | incomplete: no LPIPS/DISTS/VBench, flow-warped temporal residual, stress-set identity or decoded lip-sync qualification |
| Microkernel bandwidth/occupancy and internal bottleneck evidence | aggregate core and branch timings | incomplete: no bandwidth/occupancy counters or isolated route/attention timing |
| Avoid full routing tables per design section 39 | host route allocation and route-mask shader | at v9 a float block table remains; the packed layout below removes floats but still materializes bounded block connectivity, not a token attention matrix or gathered K/V |

At the v9 audit the table was `heads * blocks * blocks * sizeof(float)`: 903168 bytes
at 5376 rows and 6195200 bytes (about 5.91 MiB) at 14080 rows with 32 heads.
This is a concrete remaining implementation opportunity, not proof that the
table dominates execution time. A compact-bit route representation should be
tested independently with route/oracle parity and full-core timings before
claiming any improvement; existing public buffer contracts and scalar probes
must remain coherent if that representation changes.

The four local references were rechecked at their documented revisions:
SolAttn `45071126`, attn-bench `b68b79dd`, LVSA `100666e0`, FastVideo
`bfc9c017`. The production native-only rebuild succeeded. Targeted suites
(`test_contract`, `test_ltx_sparse_benchmark`, `test_ltx_route_recall`,
`test_ltx_query_cache`) report **93 passed, 1 skipped, 71 subtests**; the
skip is the unavailable Wan fixture, not a sparse test success. All 13
entries actually listed by `SOURCE_MANIFEST.json` match their source hashes;
this does not imply that every runtime file is covered by that manifest.
`git diff --check` passes. Broad research/quality acceptance remains open.

## Packed-u32 route integration

The production host, workspace allocator, scalar/tiled attention shaders and
native probe now share a packed route scratch format:
`uint32[head][query_block][ceil(key_blocks/32)]`, LSB first, with zero tail
bits. The route kernel writes one word per SIMD group using a uniform
reduction, without atomics, an initialization pass, gathered K/V or changed
selection rules. Dispatch requires a 32-lane SIMD width. Scratch capacity is
the larger of the packed layout and the preceding Sol key-statistics scratch
(`heads * 128 * 2` words). The shared C allocation helper checks the supported
row range. Function signatures and request options are unchanged, but callers
that inspect/allocate route scratch must rebuild for the new format. Old
float-route shaders and probes must not be mixed with packed-route binaries.

| 121-frame shape | previous float bytes | packed scratch bytes including stats reserve |
|---|---:|---:|
| 5376 rows, 32 heads | 903168 | 32768 |
| 14080 rows, 32 heads | 6195200 | 197120 |

This reduces that allocation by approximately 96.4%/96.8%, not total model
memory. Very small shapes can allocate more than the old float table because
the statistics reserve is maintained. Block connectivity is still
materialized, now as bits; the strict design proposal of no full route table
at all is not satisfied merely by this representation change.

Evidence:

- `packed-route-parity.json`: 41 scalar/route checks, including legacy Sol
  and poisoned small-route buffers; maximum scalar-oracle relative L2
  0.002366. The probe checks packed tail bits explicitly.
- `packed-route-abba/report.json`: 30 float/packed boundary pairs spanning
  32/33/128/129/256 blocks across all six modes, plus 16 real-QKV ABBA
  configurations across 480p and 720p late fixtures. All compared global and
  per-head numerical metrics match exactly as reported. This is not a
  cryptographic equality check on saved output tensors.
- C allocation-helper tests cover every supported row count and invalid
  inputs; the production native rebuild and shader manifest check pass.

Timing changes are small and mixed. In 720p, pooled top-k32 is
66.512 -> 66.269 ms (1.0037x), top-k64 110.965 -> 110.003 ms (1.0087x),
and all-exact 322.469 -> 318.125 ms (1.0137x). Structured modes 1/2 with
block radius 1 instead take 19.880 -> 20.308 ms and 25.343 -> 25.862 ms
(approximately 2.2%/2.0% more time). Those block-window settings are not the
temporal-window full-video pilots. CPU native rebuilding overlapped part of
this core experiment; no other GPU benchmark was run in parallel. These
small ratios do not establish a stable general speedup. The integration is
an allocation reduction with a measured trade-off, not evidence for 1.5x
Stage-2 acceleration or new video-quality qualification. Dense defaults and
all six mathematical policies remain unchanged.

`packed-route-production-smoke/capture.json` also records a completed
768x448, nine-frame all-exact opt-in generation through the production
workspace, with audio, video export and QKV capture. This verifies integration
on a small shape, not temporal quality or production-resolution performance.
Targeted contract and diagnostic suites report **95 passed, 1 skipped,
71 subtests** after the production rebuild; `git diff --check` passes.

## Perceptual metric attempt

The LPIPS author repository was downloaded at commit `082bb24f` into
`references/mac-kernel/PerceptualSimilarity`. `tools/native/evaluate_ltx_lpips.py`
implements all-frame, full-resolution LPIPS AlexNet v0.1 with explicit RGB
normalization, frame-count/fps validation, model-state hashing and an identity
check. SciPy and tqdm were installed into the research-only directory
`outputs/ltx-sparse-research/lpips-deps`; no base environment was modified.
The first sandboxed run could not obtain torchvision's ImageNet AlexNet
checkpoint, but the approved external retry downloaded
`alexnet-owt-7be5be79.pth` to the research cache. The completed
`lpips-comparison-v1.json` compares five video entries (the 480p pooled
report appears twice because its report contains two candidate rows): 480p
all-exact mean **0.04671**, pooled top-k32 mean **0.18270**; 720p all-exact
mean **0.05746**, pooled top-k32 mean **0.27153**. The metric is descriptive:
there is no calibrated pass threshold, and it does not qualify temporal
flicker, identity, audio alignment or lip-sync. The LPIPS input-contract tests
pass. VNFeaturePrint remains available through the existing native helper.

## Reference-flow temporal diagnostic

`tools/native/evaluate_ltx_temporal.py` adds a CPU-only, all-transition
diagnostic using OpenCV Farneback flow. The dense reference defines both the
forward motion and forward/backward-consistency mask. The next RGB frame is
sampled along that same reference motion for both videos, and the temporal
residual difference is measured on the same valid pixels. Candidate flow is
also estimated for a descriptive endpoint difference, but never chooses the
mask. Out-of-bounds samples and inconsistent reference flow are excluded.
The report records their coverage so masking cannot silently hide low
confidence. Defaults resize with area interpolation to width 640, preserving
aspect ratio: 640x373 for the 480p bucket and 640x352 for 720p. These are not
full-resolution temporal metrics.

`outputs/ltx-sparse-research/temporal-comparison-v1.json` is complete with
five comparisons, including both candidate repetitions of the 480p ABBA
report. Every comparison covers all 120 transitions of a 121-frame video.

| Workload/policy | residual-difference MAE /255 | flow endpoint difference (evaluation pixels) | valid pixel coverage |
|---|---:|---:|---:|
| 480p all-exact control | 3.2025 | 0.4000 | 83.15% |
| 480p pooled top-k32 (both repetitions) | 6.5554 | 1.3424 | 83.15% |
| 720p all-exact control | 4.0898 | 0.4695 | 78.47% |
| 720p pooled top-k32 | 10.8363 | 1.8293 | 78.47% |

Numbers are valid-pixel weighted. The 720p sparse candidate has 2.65x the
residual discrepancy and 3.90x the flow endpoint discrepancy of the
all-exact control. These are comparisons of diagnostics, not a calibrated
perceptual scale or an independent proof of flicker. Flow estimation can fail
on fast motion, occlusion and textureless content; a static brightness offset
can have zero temporal discrepancy. Spatial RGB/LPIPS evidence remains
necessary. No pass threshold, identity qualification or lip-sync claim is
introduced. Eight unit tests cover warp direction, border and consistency
masking, candidate-independent coverage, identity/flicker/static-offset cases,
invalid flow and decoded frame-count validation.

The single-threaded CPU evaluation overlapped the middle-only experiment's
candidate warmup, not its measured timing pairs. It used no GPU, and was
finished before measured generation began.

## Dense-edge schedule experiment (720p, 121 frames)

`720p-pooled-topk32-middle-only-pilot-121/report.json` retains dense video
self-attention on the first and last Stage-2 diffusion steps and uses pooled
top-k32 only in the middle step (with same-frame safety and one dense
transformer block at each edge). The matched dense Stage-2 time was
**125.2812 s** and the candidate was **117.5404 s**, a **1.06586x** Stage-2
speedup; request wall speedup was **1.03804x**. Stage-1 and Stage-2 input
video tensors were byte-identical. The candidate latent relative L2 was
**0.21918** and cosine **0.97598**. Decoded RGB mean correlation was
**0.96332**, minimum correlation **0.93957**, mean MAE **9.2373/255**, and
maximum motion-energy relative error **2.49%**; the existing RGB gate failed.

The same candidate's descriptive metrics were mean LPIPS **0.14428** and
reference-flow residual-difference MAE **7.0523/255** at 640-pixel evaluation
width. For comparison, the all-exact 720p control has LPIPS **0.05746** and
flow residual-difference MAE **4.0898/255**, while the all-step pooled top-k32
candidate has LPIPS **0.27153** and flow residual-difference MAE
**10.8363/255**. Dense edge steps therefore show a better measured
quality/speed tradeoff than all-step top-k32, but still fail the reference
fidelity gate and reduce speedup far below 1.5x. These are cross-experiment
comparisons: the binaries/shaders differ (this experiment uses packed routes),
and the three baseline MP4 file hashes differ. This is not a controlled
same-binary comparison of diffusion-step schedules. LPIPS and flow values
remain descriptive; they do not create a new acceptance threshold.

Native profile records Stage-2 video-self branch times of **56.516 s** dense
versus **48.782 s** for the middle-only candidate. Video FFN was
**40.660/40.659 s** and parallel cross-attention wall was
**11.372/11.366 s**, respectively. Branch intervals can overlap and must not
be summed as independent wall time. This AB pilot used one warmup per
variant and a retained engine; it is not ABBA confidence or multi-seed
qualification. There were no other GPU benchmarks during measurement.

Evidence: `lpips-middle-only-v1.json`, `temporal-middle-only-v1.json`, and the
paired generation report under `outputs/ltx-sparse-research/`. All 121 frames
and 120 transitions were evaluated. `evidence-summary-v10.json` now contains
15 completed paired reports, 12 with 121 frames: only the two all-exact
controls pass RGB; all ten approximate 121-frame candidates fail. Seven
incomplete or unsupported-schema reports are omitted. Dense defaults and
production kernels were not changed in this experiment.

Validation: **107 passed, 1 skipped, 71 subtests** after generation finished.
An earlier concurrent test run had one contract test intercepted by the
generation process's GPU lock; the serial rerun passed. No new native build
was needed because this iteration changes only offline Python diagnostics,
tests and research documentation.

### 720p direct top-k32 full-video comparison

`720p-direct-topk32-pilot-121/report.json` completes the missing 1280x704
direct-routing video trial. It uses the same 32-block budget, same-frame
safety and all Stage-2 steps as pooled top-k32, but drops unselected remote
blocks without pooled V correction. Dense Stage-2 was **125.2884 s** and
direct top-k32 was **101.2114 s** (**1.2379x**); request-wall speedup was
**1.1063x**. Stage-1 and Stage-2 input video tensors were byte-identical.
The candidate Stage-2 latent relative L2 was **0.42663**, cosine **0.90793**.

Decoded RGB over all 121 frames had mean correlation **0.87390**, minimum
correlation **0.83317**, mean cosine **0.98158**, mean MAE **18.9279/255** and
maximum motion-energy relative error **12.00%**; the RGB gate failed. LPIPS
mean was **0.29907**. Reference-flow residual-difference MAE was **11.3375/255**
with flow endpoint difference **1.9675** evaluation pixels. These are worse
than pooled top-k32 (RGB mean correlation 0.88943, LPIPS 0.27153, flow
residual MAE 10.8363/255), with only a slightly larger observed speedup.
These are separate AB pilots with different binary/shader hashes; the small
speed difference is not a statistically established policy advantage. Native profiles
show the video-self branch at **32.438 s** versus **56.518 s** dense (1.74x),
while fixed FFN and cross-attention work remains approximately unchanged.
Direct top-k32 therefore is not quality-preserving and still does not reach
1.5x Stage-2.

This closes the missing 720p full-video direct-versus-pooled comparison.
Evidence: `lpips-direct-topk32-v1.json`, `temporal-direct-topk32-v1.json`,
and the paired report above. Stage-1 audio is byte-identical; Stage-2 audio
relative L2 is 0.25622 and cosine is 0.97133. Joint AV computation allows
video-path changes to affect audio; these latent metrics are not a lip-sync
qualification.
Multi-prompt/seed, identity, lip-sync and human-preference qualification
remain outside this single-prompt experiment.

The refreshed `evidence-summary-v11.json` contains 16 completed paired
experiments and omits 7 incomplete or unsupported reports. It includes the
request-wall medians and speedups; reports without a valid request-wall
duration remain explicitly unverified rather than being assigned a default.

Despite different MP4 container hashes, the full decoded RGB24 streams of
the measured dense baselines in all-step pooled top-k32, middle-only pooled
top-k32 and direct top-k32 have the same SHA-256:
`334e662c7f7974fbdfe7473db4b93a47fc2d106b752703f696ba2fa759073086`.
This verifies identical reference video pixels for these quality comparisons,
but does not remove the timing confound from separate runs/builds.
Final targeted tests: **107 passed, 1 skipped, 83 subtests**;
`git diff --check` passes. Production kernels and dense defaults are unchanged
in this iteration.

## Full video-self branch attribution (synthetic Metal probe)

`tools/native/ltx_attention_branch_probe.c` and
`tools/native/benchmark_ltx_attention_branch.py` measure the same QKV, gate,
attention and output APIs used by the production video-self path. Inputs and
weights are synthetic, so this is a kernel/branch attribution experiment, not
generated-video evidence. The probe uses two warmups, nine rotating-order
iterations, and two independent repeats at 5,376 rows (480p bucket) and
14,080 rows (720p bucket). `outputs/ltx-sparse-research/branch-attribution-v1.json`
records source and shader hashes.

| rows | variant | total branch ms | core ms | core speedup vs split dense | branch speedup vs fused dense |
|---:|---|---:|---:|---:|---:|
| 5,376 | split dense | 90.73 | 36.07 | 1.00x | 0.99x |
| 5,376 | direct top-k32 | 77.81 | 23.05 | 1.56x | 1.16x |
| 5,376 | pooled top-k32 | 78.88 | 24.11 | 1.50x | 1.14x |
| 14,080 | split dense | 387.23 | 247.06 | 1.00x | 1.00x |
| 14,080 | direct top-k32 | 202.80 | 62.45 | 3.96x | 1.90x |
| 14,080 | pooled top-k32 | 208.49 | 68.17 | 3.62x | 1.85x |

At 720p direct top-k32, QKV is **103.53 ms**, gate **2.23 ms**, and output
post-processing **34.57 ms**; these components total about 140 ms even after
the attention core drops from 247.06 to 62.45 ms. A roughly 4x core result
therefore becomes only 1.90x for this complete synthetic branch. This is
consistent with smaller full-video gains, but not a quantitative prediction
of the real Stage-2 result. In particular, this probe has no same-frame safety
region, no real model activations, no AV concurrency and no command batching.
At 480p, the non-core work dominates more strongly and the synthetic complete
branch speedup is only 1.16x. These measurements favor investigating QKV and
output fusion over another small routing optimization; they do not prove a
hardware-independent ceiling or impossibility of 1.5x Stage-2.

Table values are medians of two per-process medians, and component medians
need not sum exactly to total medians. A 65-row tail shape is included for
execution coverage, not performance qualification. On the two production
shapes, split dense relative L2 versus fused dense is about 0.0015--0.0016;
all-exact is about 0.0024--0.0025. Those numerical diagnostics do not replace
the existing scalar/oracle parity suite. Synthetic top-k errors are much
larger and must not be substituted for real-QKV or video quality. Production
defaults and kernels remain unchanged.

## Output ConvRot/INT8-linear fusion experiment

The branch probe tests the existing `ltx_gpu_linear_int8_convrot_mps_bf16`
graph in place of separate Metal ConvRot and MPSGraph INT8 output projection.
`branch-output-fusion-v1.json` is complete with two repeated processes per
shape, nine measured iterations, rotating variant order and both pooled
top-k32 and all-exact controls. At 5,376 rows pooled branch time changes
from **78.88 to 79.40 ms**, about **0.66% slower**; at 14,080 rows it changes
from **208.45 to 210.24 ms**, about **0.86% slower**. All-exact is also slower.
Only the 65-row execution-smoke shape improves, by roughly 2--3%.

The production-shape paired pooled outputs are not bit-identical: 519 BF16
elements change at 480p and 1,605 at 720p; relative L2 is **1.272e-5** and
**1.378e-5**, respectively, with maximum absolute difference **0.00024414**.
`branch-output-fusion-same-core-v1.json` repeats numerical comparison using
the exact same core buffer for both output paths, outside timed sections,
and reproduces those counts/errors. This follow-up has only three measured
iterations and overlapped CPU tests; use the first report for timing.

The combined graph call therefore is **not integrated**: it gives no speed
benefit at the target shapes and introduces small numerical differences.
This is an API-level graph fusion experiment, not proof that all possible
custom Metal output-fusion kernels are slower. No new full-video run or
quality qualification is claimed for it. Both reports are under
`outputs/ltx-sparse-research/`; production kernels/defaults are unchanged.
Native probe build passes; targeted tests report **112 passed, 1 skipped,
96 subtests** and `git diff --check` passes.

## Fine remote-summary diagnostic

To test whether pooled remote error is caused mainly by the 64-token summary
granularity, `tools/native/analyze_ltx_fine_pooling.py` keeps the exact
64-token top-k route fixed and subdivides only unselected remote blocks into
32/16/8/4/2/1-token summaries. This is a CPU sampled diagnostic: it does not
change the Metal API, route selection, dispatch schedule or production
buffers. The 64-token rows exactly reproduce `route-recall-v4.json` for all
three fixtures, including output errors.

| fixture / location | keep | summary width | sampled output relative L2 | represented-key fraction |
|---|---:|---:|---:|---:|
| 480p early | 32 | 64 | 0.04727 | 0.3906 |
| 480p early | 32 | 8 | 0.04002 | 0.4583 |
| 480p late | 32 | 64 | 0.08347 | 0.3906 |
| 480p late | 32 | 8 | 0.07589 | 0.4583 |
| 720p late | 32 | 64 | 0.12215 | 0.1588 |
| 720p late | 32 | 8 | 0.11322 | 0.2523 |
| 720p late | 32 | 4 | 0.10752 | 0.3591 |
| 720p late | 32 | 2 | 0.09597 | 0.5727 |
| 720p late | 32 | 1 | 0.00000 | 1.0000 |

For 720p keep-32, moving from 64-token to 8-token remote summaries reduces
sampled error only about 7.3% while increasing represented entries by about
59%; 2-token summaries use 8,064 entries (57.3% of the dense key count) and
remain at relative L2 0.09597. Width-1 has no compression and reproduces the
CPU dense diagnostic. Keep-64 shows the same diminishing-return curve
(0.07292 at width 64,
0.06863 at width 8, 0.06168 at width 2). The 480p early/late fixtures are
consistent with the same qualitative tradeoff.

Uniform fine pooling is therefore deprioritized for Metal implementation,
not proven incapable of improvement. In the 720p sample, coarse keep-64 has
lower error (0.07292) with fewer represented entries (4,252) than keep-32
with width-4 summaries (0.10752 and 5,056). Represented entries count exact
tokens plus remote summaries; they do not measure GPU time, SIMD tile
padding, memory traffic or summary-generation cost. Each fixture samples
eight query blocks, four queries per block and all 32 heads (1,024 query/head
observations), without safety regions. Attention uses FP64 arithmetic and
BF16-rounded summaries, outputs and gates; no Metal timing or final-video
quality conclusion is inferred. The
reports are stored beside each QKV fixture as `fine-pooling-v1.json`; the
720p width-1/2/4 extension is `fine-pooling-v2.json`. Sixteen fine-pooling
and route-recall unit tests pass. No production shader or default was changed.
The full targeted suite reports **118 passed, 1 skipped, 96 subtests**;
`git diff --check` passes.

## Per-head keep-budget allocation diagnostic

The current Metal API uses one `keep_blocks` value for every head. To test
whether this uniform budget is the main quality bottleneck,
`tools/native/analyze_ltx_head_budget.py` fits a discrete dynamic program to
the per-head errors already measured by real-QKV replay. It minimizes summed
squared output error under the same total selected query/key block-cell
budget as a uniform policy. This is an optimistic offline simulation: it uses
errors measured against dense output, assumes head-specific routing metadata
is free, and does not model route generation, divergent SIMD work, padding,
or Metal dispatch cost. It is not a production policy or video-quality gate.

| fixture | uniform keep | uniform core relative L2 | optimized same-cell-budget L2 | improvement |
|---|---:|---:|---:|---:|
| 480p late | 32 | 0.08420 | 0.06026 | 28.4% |
| 720p late | 32 | 0.11886 | 0.09808 | 17.5% |
| 720p late | 64 | 0.07072 | 0.04934 | 30.2% |

The optimizer assigns larger budgets to heads with steep error curves and
smaller budgets to heads already insensitive to remote blocks. The 480p late
32-budget allocation has mean keep exactly 32; the 720p allocation also uses
mean keep 31.94 after integer route-cell matching. The reports are
`head-budget-480p-late-v1.json` and `head-budget-720p-v1.json`.

This initially motivates a bounded head-budget diagnostic, not integration.
The current route kernel takes a common keep count; a per-head budget needs
additional metadata and validation. The tiled attention kernel already
consumes per-head route masks, so different exact masks do not themselves
require a new attention layout. Early 480p `replay.json` was rejected for
missing mode-5 data; late 480p `replay.json` was rejected for missing per-head
metrics. Valid `replay-per-head.json` and `replay-threshold-skip.json` supply
the cross-position test below.

### Frozen cross-position allocation check

Evaluate each optimized 480p allocation at the other captured step/block,
without re-optimizing:

| source -> evaluation | uniform keep | evaluation uniform L2 | frozen allocation L2 | selected-cell ratio vs uniform |
|---|---:|---:|---:|---:|
| early -> late | 32 | 0.08420 | 0.12557 | 0.98510 |
| early -> late | 64 | 0.02434 | 0.09663 | 0.99464 |
| late -> early | 32 | 0.04824 | 0.07709 | 1.00014 |
| late -> early | 64 | 0.01186 | 0.05408 | 0.99745 |

All four frozen allocations perform worse than their evaluation-position
uniform baseline. Actual selected-cell cost changes by up to about 1.5%,
because diagonal safety membership depends on QKV; the late-to-early keep-32
case slightly exceeds the uniform cost. The evaluation reports expose these
differences instead of claiming perfectly matched holdout budgets. Budget
128 at 480p is clamped by the 84 available parent blocks; mean *requested*
keep can therefore exceed the uniform reference despite matching selected
cells. There is no timed mixed-budget Metal dispatch in this experiment.

Reports: `head-budget-480p-early-to-late-v1.json` and
`head-budget-480p-late-to-early-v1.json`. This rejects reusing one global
head-budget vector across these two positions, not per-layer calibrated or
dynamic routing in general. Since step and block both differ, it does not
isolate which change causes the failure. Head indices in different layers
need not represent the same function. Optimizing against dense output on the
same snapshot is an in-sample oracle over the measured discrete choices, not
a production router or universal bound.

Dynamic-program tests match exhaustive search with nonmonotonic errors and
head-specific costs; frozen evaluation is tested not to re-optimize.
Targeted suite: **123 passed, 1 skipped, 96 subtests**;
`git diff --check` passes. No production kernel or default was changed.

## Second prompt/seed: portrait turn (480p, 121 frames)

`480p-portrait-seed7-pooled-topk32-pilot-121/report.json` adds a second
prompt and seed, keeping the 480p pooled top-k32 routing configuration:
mode 5, keep 32, radius/anchors/frame tokens zero, dense edge blocks 1,
all Stage-2 steps. The prompt is:

> A close-up cinematic portrait of an adult woman with curly brown hair,
> slowly turning her head toward the camera and smiling, soft window light,
> stable camera.

Seed is 7; other generation settings are 768x448, 121 frames, 24 fps, audio
enabled and component-staged. One warmup per variant precedes one measured
AB pair. Dense Stage-2 is **40.8791 s**, candidate **39.3454 s** (**1.0390x**);
request wall is **116.3427 -> 113.2156 s** (**1.0276x**). Stage-1 video/audio
and Stage-2 input video are byte-identical. Final video latent relative L2
is **0.30000**, cosine **0.95458**; audio latent relative L2 is **0.03540**,
cosine **0.99940**. Latent audio metrics do not establish decoded sync.

All 121 RGB frames: mean correlation **0.97588**, minimum **0.96694**, mean
cosine **0.99348**, mean MAE **6.0768/255**. The RGB gate fails its mean
correlation, cosine and MAE criteria. Mean LPIPS is **0.08413**. At 640-pixel
evaluation width, all 120 reference-flow transitions give residual-difference
MAE **1.7997/255**, endpoint difference **0.2307** evaluation pixels, and
**98.14%** valid coverage. These descriptive scores are scene-dependent;
there is no calibrated LPIPS/flow acceptance threshold.

`paired-frames.png` shows frames 0/60/120 with dense above candidate. The
sampled composition, pose and broad facial/hair structure are similar, with
local shading/detail differences. Three resized frames cannot establish
identity stability or temporal quality over the whole clip. There is no
all-exact split-path control for this portrait, so numerical-path drift and
sparse approximation are not separately attributed here.

Independent reports are `lpips-portrait-seed7-v1.json`,
`temporal-portrait-seed7-v1.json`, and the generation directory's
`decoded-quality.json`. `evidence-summary-v13.json` now has 17 completed
paired reports, 14 with 121 frames: two all-exact controls pass and all
12 approximate experiments fail RGB. There are two prompt/seed combinations,
not a crossed multi-prompt/multi-seed study: prompt and seed changed together.
This improves workload coverage but still establishes neither 1.5x Stage-2
nor general near-dense quality. Production kernels/defaults are unchanged;
targeted tests remain **123 passed, 1 skipped, 96 subtests**.

### Incomplete portrait all-exact control

The follow-up `480p-portrait-seed7-all-exact-pilot-121/` run used mode 1,
block radius 256, seed 7 and the same portrait workload. It was accidentally
interrupted by the assistant after both warmups, during measured inference.
Its report remains incomplete: warmup timings are not benchmark results and
this run provides no paired speed or quality evidence. Existing completed
portrait pooled-top-k32 measurements are unaffected. A fresh completed
all-exact pair is still needed to attribute portrait numerical-path drift.
The follow-up tool checks passed **23 tests and 12 subtests**;
`git diff --check` also passed before this documentation update.
