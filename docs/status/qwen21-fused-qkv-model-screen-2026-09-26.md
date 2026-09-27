# Qwen Image 2.1: experimental model-level fused QKV GPU screen

The opt-in `TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC=1` uses the
shape-screened BM32 Metal TensorOps kernel in
`native/models/qwen21/metal/qkv_projection.hpp`. One kernel performs the
three BF16 projections and Q/K RMSNorm + RoPE, writing Q/K/V directly in
head-major SDPA layout. At resident-session initialization it concatenates
each block's Q/K/V matrices **once** and replaces the original three
matrices in the session's weight table; the bank holds shallow references
to those replacement weights. A first version kept both source and packed
weights, increasing MLX active allocation by ~3.22 GB. Replacing sources
returned the three-reference active allocation from 18.13 GB to 14.91 GB,
the same as the pure-GPU control. If the flag is disabled within that
Session, the original checkpoint is reloaded before the next GPU request;
cancellation during a partial pack also invalidates the bank.

This is **not** mathematically exact: BF16 rounding in the projection and
Q/K normalization changes slightly. It requires explicit approximation
opt-in, 512² pure GPU, no LoRA, 1–3 resized-512 edit references, and does not
combine with other Q/K fusion or reference-local attention. The initial
screen did not combine it with resident prefix-KV; a separate guarded
combination is evaluated below. It is never enabled by default. On the M4 Max, independently
prepared resident Sessions each performed a full five-step warmup, then a
cached five-step request; both arms used the same checkpoint, prompt, seed,
requested image/reference geometry and dumped inputs. The comparison
verifier checked exact text/noise/ordered-reference latent payloads.
Walls include VAE decode, PNG export and input dumps, not preparation:

| 512² workload | BF16 GPU wall | Fused QKV wall | Ratio | RGB correlation |
| --- | ---: | ---: | ---: | ---: |
| Text-to-image fox, seed 42 | 6.002 s | 5.815 s | 1.032× | 0.99814 |
| One resized-512 reference, seed 42 | 7.281 s | 7.060 s | 1.031× | 0.99997 |
| Two resized-512 references, seed 17 | 8.643 s | 8.441 s | 1.024× | 0.99948 |
| Three resized-512 references, seed 17, repeat 1 | 10.116 s | 9.816 s | 1.031× | 0.99876 |
| Three resized-512 references, seed 17, repeat 2 | 10.149 s | 9.823 s | 1.033× | 0.99876 |

The three-reference pair repeats the same output in each arm and shows a
median **1.032×** end-to-end gain after the weight-bank replacement. Its
candidate PNGs are byte-identical to the older candidate that duplicated
weights. Manual inspection retains the red fox in snow, the single teapot,
both teapots in the two-reference edit, and both teapots plus central
dragon sticker in the three-reference edit; colors and local edges shift
slightly. The single-reference matte-finish instruction still fails in
both arms. Pixel correlation is **not** a semantic quality score; one
device, prompt set and two repeats of only one fixture do not establish
general quality or a reliable 1.1× pure-GPU gain.

The first cold CLI `generate` with three resized-512 references exited 0,
reported GPU execution and the diagnostic approximation label, and wrote
a PNG byte-identical to the resident candidate; cold request wall was
13.177 s and is not comparable with warmed walls. A resident probe then
disabled the fused flag, prepared the ordinary GPU route (reloading the
source checkpoint matrices), and generated a PNG **byte-identical** to
the saved pure-GPU control. This verifies the route switch, not thermal
robustness. No model weights or PNG are tracked; raw receipts are ignored
under `results/qwen21/session-*-fusedqkv-*`.

## W8A8/6144-channel hybrid interaction

The same diagnostic kernel is also explicitly permitted on 512² five-step
edits with 1–3 resized-512 references, checkpoint-matched 32-layer W8A8
Core ML and a **BF16 GPU** FFN complement; W8A16 complement and full-size
references are not admitted for this combination. The first step's
attention geometry and the ordinary 32-layer decode FFN coverage remain
unchanged. Hybrid model prediction calls and failure counts are verified
equal in the paired Sessions; calls do not prove physical ANE execution.

| Three-reference seed 17, two matched full-warm requests | Baseline | + fused QKV | Median speedup |
| --- | ---: | ---: | ---: |
| Decode-only W8A8 | 8.907 / 8.882 s | 8.584 / 8.617 s | 1.034× |
| Last-16 first-step tiles + final-step FFN reuse | 8.078 / 8.075 s | 7.768 / 7.754 s | 1.041× |

The last-16 combo also measured **5.551→5.375 s** (one reference) and
**6.774→6.531 s** (two references), with exactly matched input dumps and
the same W8A8 per-request call counts. A second three-reference seed 29
measured **8.071→7.733 s**, input dumps equal; all subjects remain
recognizable, although seed-17's blue handle shows a more conspicuous
double edge and the sticker outline changes slightly. Seed-29 preserves
the two pots and sticker face with smaller local shifts. The three-ref
last-16 + final-step reuse + fused QKV is **~1.306×** faster than the
matched plain BF16 GPU baseline (10.116/10.149 s), but combines several
lossy approximations and is not a quality-qualified 1.3× default.

Manual comparison of decode-only hybrid with/without the fusion shows both
pots and the sticker face with less detail drift than the last-16 + step
reuse candidate. The combined route's `cpuAndNeuralEngine` policy still
does not establish hardware-level ANE residency. In the same resident
process, turning off fusion and switching back to ordinary GPU caused a
checkpoint reload and produced a PNG byte-identical to the saved exact-GPU
baseline, also when the preceding requests had been hybrid.

## Additional GPU-only approximate combination

On the same three-reference seed-17 fixture, explicit fused QKV plus
final-step and penultimate-even-layer GPU FFN reuse measured
**8.806/8.805 s** against the matched **10.116/10.149 s** BF16 GPU
control (median 1.151×). Both teapots and sticker are visible, but the
sticker edge and blue-pot handle differ. This is neither an exact GPU
kernel-only 1.15× nor a general editing-quality claim.

Further work: qualify on more prompts and repeated 40-step edits;
measure cold preparation and device-memory pressure before
deciding whether the 2–4% kernel gain justifies an automatic route. The
W8A16 GPU FFN complement remains slower than BF16 in the separate Qwen21
screen; this QKV change does not alter FFN quantization.

A follow-up single three-reference **40-step** pair used the same
resized-512 inputs, two-step warmup, exact dumped text/noise/reference
latents and the same BF16 checkpoint. Ordinary GPU wall **52.320 s** versus
fused QKV **50.856 s** (1.029×), including VAE/PNG/dumps. Both PNGs keep
two recognizable teapots and a clear sticker face; the left teapot's
surface/position and some lighting differ. RGB correlation 0.98862 and
RMSE 0.04673 are not semantic quality metrics. This one pair does not
qualify long-step quality or consistent speed on other prompts/devices.

The explicit fast research combination can be invoked with a request JSON
containing `execution=gpu_ane`, `allow_approximation=true`,
`qwen21_w8a8=true`, `qwen21_reference_size=512`, a matching 32-layer
6144-channel W8A8 manifest and 1–3 ordered image references, with five
steps and 512² output:

```sh
TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC=1 \
TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC=16 \
TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_FFN_DIAGNOSTIC=1 \
build/native/turbocider generate MODEL_DIRECTORY EDIT_REQUEST.json
```

This is **not** an automatic preset. The Core ML compute-unit policy and
prediction count cannot establish actual ANE device residence. The default
GPU and GPU/ANE routes remain unchanged with all flags unset. The native
Session lifecycle checks included cancellation, disabled-flag GPU switch,
and unload. The final rebuilt library passed `make test-qwen21` and
`git diff --check`; those checks alone do not verify image quality.
The actual CLI `plan` reported all five relevant approximation labels and
`generate` completed in a cold process. Its three-reference five-step PNG
matched the resident fast hybrid PNG byte-for-byte; cold wall was **19.146 s**
including model/manifest load and prompt/image encoding. That cold number
must not be compared with warm-session 7.75–7.77 s.

One extra **rejected** three-full-size-reference GPU diagnostic initially
passed the experimental plan gate. Its first step had 13,442 rows. Under
the same five-step warmup and exact text/noise/ordered-reference-latent
dumps it measured 25.291→24.545 s (one run, 1.030×). The baseline already
ghosted the central sticker; the fused-QKV PNG made its eyes and outline
conspicuously worse, despite retaining both teapots. This is not an
acceptable reference edit for a ~3% improvement. The plan gate now rejects
full-size-reference fused QKV editing; the local receipt is preserved in
`results/qwen21/session-fullref-edit3-gpu-fusedqkv-fullwarm-{control-,}run/`.

## Guarded combination with repeated-edit prefix KV

The experimental resident prefix-KV bank can now coexist with the fused
QKV bank on the same opt-in 512², 1–3 resized-reference edit routes, without
LoRA. Its cache identity includes the fused/ordinary QKV mode; before
unpacking or clearing fused weights, the resident cached Transformer is
destroyed so it cannot retain a pointer to an obsolete weight bank. This
does **not** enable sequence tiling together with cross-request prefix KV:
the former requires a full-sequence first step whereas a cache hit bypasses
it. The two-step warmup below deliberately cannot seed the five-step cache.
All arms use three identical ordered 512px references and seed 17.

| Five-step warm request | First miss | Repeated hit | MLX active on hit |
| --- | ---: | ---: | ---: |
| Ordinary BF16 GPU + prefix KV, earlier control | 10.156 s | 6.612 s | 17.13 GB |
| Fused QKV BF16 GPU + prefix KV | 9.827 s | 6.489 s | 17.13 GB |
| Ordinary W8A8/6144 + BF16 GPU + prefix KV, earlier control | 8.916 s | 5.481 s | 21.97 GB |
| Fused QKV W8A8/6144 + BF16 GPU + prefix KV | 8.586 s | 5.281 s | 21.97 GB |

The new fused-GPU and fused-hybrid experiments each passed miss/hit, equal
same-seed PNG, changed-seed hit versus uncached same-route PNG, cancellation
and retry, GPU route switch and unload in a resident Session. Each new
same-route PNG is byte-identical to its earlier fused-QKV *no-prefix* PNG;
the prefix bank therefore adds no extra image loss. Dumped text, initial
noise and all three ordered reference latents were exactly equal between
the new GPU and W8A8 routes. The W8A8 arm made 128 Core ML predictions
on **each** measured request with no runtime failure. Its blue teapot,
beige teapot and central sticker face remained visible; its QKV rounding
and W8A8 precision still differ from exact GPU.

Fused-QKV prefix-hit hybrid versus fused-QKV prefix-hit GPU is **1.229×**
on this one fixture (5.281 versus 6.489 s). Versus the older unfused
prefix-hit hybrid, fused QKV saved **0.200 s**; that 3.6% is the incremental
single-run gain, not a general 1.3× kernel improvement. Without a cached
prefix, the fused hybrid's 8.586 s is only about **1.18×** faster than
the earlier 10.116–10.149 s plain GPU control. The extra retained bank
adds about **2.22 GB** active MLX allocation against a no-prefix fused
hybrid request (19.75 to 21.97 GB); Core ML and system allocations are
excluded. First-miss and repeated-hit conditions must never be mixed in
a speedup ratio. A matched-full-step warmup can itself seed this bank, so
the native probe now recognizes an initial hit under that setting.

For *fresh* references, the separate last-16 tiled-first-step W8A8 +
final-step FFN reuse + fused QKV candidate still reached ~1.306× versus
plain five-step GPU (7.754–7.768 s), but it has visible blue-handle and
sticker-edge differences and remains diagnostic-only. The prefix-hit
combination avoids that additional first-step/FFN-reuse loss but only speeds
**subsequent requests with identical conditioning**. These are small
single-seed experiments on one M4 Max, not production/default presets or
proof that predictions physically ran on ANE.

## New-reference sequence loop without final-step reuse

The same fixed 1024-row W8A8 layer can process a first-step edit with
2091/3157/4226 rows by **repeatedly** consuming 3/4/5 tiles. Padding is
confined to the final tile; it is trimmed afterward. Only the *last 16*
of 32 first-step FFNs use this diagnostic loop, while all 32 FFNs on each
of the following four denoise steps use the ordinary hybrid partition.
The first-step attention remains GPU and retains the full ordered sequence.
Combining this existing loop with fused GPU QKV is supported without
turning on final-step FFN reuse or cross-request prefix KV.

On the M4 Max, independently prepared resident Sessions used matched
five-step warmups, then two cached five-step 512² requests per 1/2/3
ordered 512px-reference fixture; all comparisons use exactly equal dumped
text, noise and ordered reference latents. Wall includes VAE, PNG export
and dumps, excludes initial model/manifest load and conditioning. Baseline
GPU timings for one/two refs are the prior same-input warmup-matched fused
QKV measurements; the three-reference plain GPU baseline is two repeats.

| 512px references | Plain BF16 GPU | Fused QKV GPU | Fused QKV + last-16 tiled W8A8 | Core ML calls/request |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 7.281 s | 7.060 s | 5.675 / 5.677 s | 176 = 128 + 16×3 |
| 2 | 8.643 s | 8.441 s | 6.864 / 6.864 s | 192 = 128 + 16×4 |
| 3 | 10.116 / 10.149 s | 9.816 / 9.823 s | 8.103 / 8.066 s | 208 = 128 + 16×5 |

The three-reference new-condition result is **1.253×** versus the matched
plain GPU median and about **1.214×** versus the fused-QKV GPU median.
Against the earlier last-16-only hybrid matched five-step warmup
(8.413/8.402 s), adding QKV fusion yields about **1.040×** improvement.
No last-step denoising reuse is counted in this result, unlike the faster
but more visibly changed 7.754–7.768 s research combination above. The
three-reference PNG has two pots and a discernible sticker face and eyes;
the blue handle and sticker contour differ slightly from exact GPU. The
one-reference output retains its teapot but, like the GPU control, misses
the requested matte finish. The two-reference output retains both pots.
An additional three-reference seed-29 run took **8.115 s**, versus **8.459 s**
for the older same-input last-16 no-fusion experiment; the latter used a
two-step rather than matched five-step warmup, so this pair supports input
and visual comparison, not a controlled timing ratio. Dumped tensors matched;
the two pots and sticker face remained visible. These are a few
seed/prompt/image examples, not broad subjective-quality
qualification. The 1/2/3-ref checks each passed the Session cancellation,
GPU switch and unload tests, including the expected call counts. The
Core ML `cpuAndNeuralEngine` preference and prediction counts do not prove
physical ANE scheduling.

To reproduce on a 512² explicit W8A8 6144-channel, 32-block manifest and
a resized-512 edit request with `allow_approximation=true`:

```sh
TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC=1 \
TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC=16 \
build/native/turbocider generate MODEL_DIRECTORY EDIT_REQUEST.json
```

This uses a *different input* from editing with the original full-size
1024px references; the resize itself is lossy. Neither the loop nor its
composition is enabled by default. On a repeated edit with the same
conditioning, the separate prefix-KV route above avoids prefill entirely;
it is intentionally incompatible with tiled first-step inference.
The CLI `generate` path independently completed this three-reference edit
with a staged Session, reported both diagnostic labels, and wrote a PNG
byte-identical to the resident mild-candidate output. Its **18.563 s**
cold wall includes model, text/image and Core ML setup and must not be
compared to a prepared Session's ~8.1 s warm request. The manifest remains
explicit and the CLI does not auto-select this experiment.

A further three-reference seed-17 screen increased first-step tiled
coverage from the last 16 to the last **20** blocks, with fused QKV in
both arms and no final-step reuse. Identical five-step warmups and dumped
inputs gave **8.103/8.066 → 7.993/7.980 s** (1.012× median incremental
gain), with 20 additional Core ML predictions per request. The blue pot
handle shows a conspicuous doubled/smeared edge, so the small saving does
not justify selecting last-20 as the preferred visual-quality tradeoff.
Keep it diagnostic-only; last-16 remains the milder new-condition candidate.

Two further schedule-only experiments on the last-16 fused-QKV candidate
were screened and removed. First, replacing `mx::eval(input, packed)` with
`mx::eval(packed)` at each GPU/Core ML bridge produced an identical PNG but
**8.093/8.104 s** versus **8.103/8.066 s**; this does not establish a
speedup. Second, preparing the FP16 inputs and enqueuing all GPU suffix
tiles *before* calling the Core ML tiles produced the same PNG and passed
the Session lifecycle, but slowed to **8.746/8.721 s**. The latter changes
the GPU/ANE scheduling order, not the number of predictions; its roughly
0.64 s penalty may reflect lost per-tile overlap or memory contention, but
these measurements do not isolate a single cause. Both temporary code
paths and the prequeue experiment flag were removed. Retain the original
per-tile `async_eval(GPU suffix) → Core ML predict → merge` schedule.
After rebuilding the reverted library, one more three-reference request
took **8.149 s**, made the expected **208** Core ML calls with zero runtime
failures, passed cancellation/GPU switch/unload, and wrote a PNG
byte-identical to the original 8.103/8.066 s mild-candidate output. This
checks the actual loaded binary rather than just the source rollback.

## FFN width and placement checks before a larger export

The existing compiled, block-0-only 1024-row W8A8 models permit a cheap
screen of 6144 versus 8192 ANE FFN hidden channels. On the same M4 Max,
same BF16 checkpoint and deterministically generated random BF16 activation
(**not** a real-model edit activation), the native FFN component probe ran
12 alternating GPU/hybrid iterations, dropping the first two. Two separate
runs gave the following summary; medians include host API and GPU completion,
not model loading or a whole DiT block:

| ANE channels | Pure GPU FFN | Hybrid FFN, two repeats | GPU/hybrid | rRMSE to GPU |
| ---: | ---: | ---: | ---: | ---: |
| 6144 | 21.37/21.46 ms | 11.01/10.94 ms | 1.94×/1.96× | 0.163 |
| 8192 | 21.36/21.47 ms | 13.82/13.78 ms | 1.55×/1.56× | 0.200 |

The wider Core ML partition is **slower** despite requiring fewer GPU
complement channels, and differs more from the BF16 oracle on this random
activation. This does not predict real-image quality, but gives no basis
for paying the disk/memory cost of a 32-layer 8192-channel export. Keep the
checkpoint-matched 6144-channel route. With ~12 GiB disk free during this
screen, no new model or compiled cache was generated.

The offline `qwen21_ane_placement.py` tool now accepts `--blocks` and
`--quiet`, rejects source `.mlpackage` manifests (it needs compiled
`.mlmodelc` artifacts), and summarizes each inspected plan. For the
**same** 6144-channel 32-layer manifest, one all-layer plan query reported
384/384 runtime operators preferred Neural Engine, while a second reported
348 preferred and 36 **unassigned** across blocks 1/24/31; those three
blocks were still unassigned in an isolated recheck. Neither query
reported a CPU/GPU preference. Constants/constexpr operators (736) were
not treated as device-executed operators. This discrepancy means an all-
layer ANE preference claim is not reproducible here, and *neither* plan
is proof of where predictions physically ran. Per-request Core ML call
counts still prove the model was invoked, not hardware utilization.

## Per-tile bridge timing and rejected batch pack

The opt-in `TURBOCIDER_QWEN21_PROFILE_TILED_FFN=1` emits four host-side
stage timings for each tiled first-step FFN block: waiting for BF16 input
and FP16 packed input to be ready, enqueuing the GPU suffix, synchronous
Core ML prediction API time, and waiting for the merged output. It does
not force any new GPU barrier beyond the existing per-tile materialization.
It is a diagnostic, not a device trace or production benchmark. On the
three-reference last-16 fused-QKV route, a five-step warmup and measured
five-step request each used five tiles/block. The first offloaded block's
input-ready stage took ~2.32 s because it also had to evaluate earlier
deferred GPU work; **this is not a 2.32 s FP16 conversion**. In the
following 15 blocks, input-ready totaled roughly 55–64 ms/block, Core ML
prediction API 47–51 ms/block, and merge wait 5–9 ms/block. GPU suffix
submission itself was under 0.5 ms/block, but its *execution* overlaps
prediction and input barriers, so it cannot be inferred from that figure.
The profiled warm request took **8.153 s**; profiling changes scheduling
and the actual non-profiled baseline remains ~8.1 s.

A temporary schedule experiment built all five FP16 input tiles and
submitted a **single** `mx::eval(packed_tiles)` before the existing
GPU-suffix/ANE/merge per-tile loop. It produced a byte-identical PNG and
passed the Session lifecycle but measured **8.148/8.101 s**, versus the
non-profiled original **8.103/8.066 s** after matched warmup. No stable
speed improvement was established; the batch-pack path and its temporary
flag were removed. Do not interpret the input-ready stage above as purely
pack time: it also consumes upstream GPU dependencies, which cannot be
removed by batching identical casts.

The separate existing reference-local attention approximation was also
screened on the 512px three-reference / last-16 tiled W8A8 / fused-QKV
route, **last reference only**. It left target-image attention over the
full prefix unchanged, but stopped the last reference's queries from
scanning the previous reference images. Matched five-step warmups and
equal dumped text/noise/three reference latents yielded **8.103/8.066 s**
for full attention versus **8.038/8.034 s** for last-reference-local
attention, only ~0.6% median improvement. Both arms still made 208 Core ML
predictions per request; the candidate passed cancellation/GPU switch and
unload. Its blue teapot handle gained a noticeable doubled/distorted edge,
and the sticker's bottom outline changed. This quality cost is not worth
the tiny gain. The temporary 512px hybrid gate and test were rolled back;
the older, separately documented full-size GPU-only local-attention
diagnostic remains available under its original guard.

## Optional final-step reuse of only the last 16 FFNs

The additional explicit diagnostic
`TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC=1`
retains the above first-step sequence-tiling loop and fused QKV, captures
independent FFN outputs on step four, and reuses only blocks 16–31 on step
five. Blocks 0–15 and all attention still execute normally on the last
step. This is an *extra lossy approximation*, not a faster INT8 kernel or
evidence of actual ANE residency. It requires five steps, 512² output,
1–3 resized-512 references, the 32-block W8A8/6144 manifest, BF16 GPU
complement, `allow_approximation=true`, no LoRA and
`TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC=16`. The ordinary path
remains unchanged when the switch is unset. It is incompatible with the
full-final-step FFN-reuse switch.

On the same M4 Max, separately prepared Sessions used matching five-step
warmups, cached identical conditioning and dumped identical text, initial
noise and ordered reference latents. Each seed-17 arm completed two
requests including VAE, PNG and dump I/O; seed 29 has one previously
recorded baseline and two new candidate requests. The verifier confirmed
**16 fewer Core ML calls per request**, zero prediction failures and
identical input tensor payloads. The per-request call counts with reuse
were 160/176/192 for 1/2/3 references, versus 176/192/208 without it.

| Resized-512 references | No last-step reuse | Reuse last 16 FFNs | Matched median ratio | RGB correlation |
| ---: | ---: | ---: | ---: | ---: |
| 1, seed 42 | 5.675 / 5.677 s | 5.559 / 5.590 s | 1.018× | 0.99983 |
| 2, seed 17 | 6.864 / 6.864 s | 6.735 / 6.759 s | 1.017× | 0.99983 |
| 3, seed 17 | 8.103 / 8.066 s | 7.991 / 8.001 s | 1.011× | 0.99980 |
| 3, seed 29 (one matched comparison) | 8.115 s | 8.000 s | 1.014× | 0.99982 |

In the inspected PNG pairs, the one- and two-reference subjects and both
three-reference compositions remain recognizable, with small edge and
texture differences. The one-reference matte-finish instruction continues
to fail in both arms. Pixel similarity is **not** a semantic quality
guarantee; this narrow sample does not justify auto-selecting a lossy
mode. The verified same-input seed-17 BF16 GPU control
(10.114/10.137 s) versus the candidate yields **1.266×** median warm
request speedup and RGB correlation 0.99117. Most of that gain comes from the already
tested W8A8 partition, sequence tiling and fused QKV; the extra cache
saves only ~1.1%. The native probe passed cancellation, flag-disabled GPU
switch and unload for each reference count and seed. CLI `plan` reported
all three diagnostic approximation labels. The production CLI `generate`
completed the same three-reference request with a new staged Session:
**20.987 s cold request wall**, including encoding, model/manifest loading
and setup. Its PNG SHA-256 was byte-identical to the resident candidate's
seed-17 output. That cold time must not be compared with the resident
warm timings above. Ignored JSON/PNG/tensor-dump receipts live under
`results/qwen21/session-ref512-edit{1,2,3}-w6144-fusedqkv-last16-finalreuse16-*`.

To opt in only for an eligible explicit request:

```sh
TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC=1 \
TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC=16 \
TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC=1 \
build/native/turbocider generate MODEL_DIRECTORY EDIT_REQUEST.json
```

With `TURBOCIDER_QWEN21_PROFILE_STEPS=1`, a further prepared three-reference
seed-17 request in this last-16-reuse mode spent 4.070 s on the first
long-sequence step, then 0.929/0.846/0.853/0.672 s on four decode steps;
VAE/export/other request time accounts for the remainder of its 8.001 s
wall. The first step is still about **55% of denoising**. The profiler's
Core ML prediction API took 0.779 s during that first step and 0.153–0.360 s
per decode step, *overlapping* GPU work: these values must not be subtracted
from step walls. The lifecycle probe's later GPU-switch request took
4.697 s first step and ~1.20 s per decode step, but reloaded the original
checkpoint and is **not** a matched warm pure-GPU timing control. These
figures prioritize an exact GPU long-sequence/attention improvement or
better GPU/ANE overlap over reducing already-short final-step FFN work.

One more **rejected** scheduling variant kept the first four 1024-row
W8A8/GPU tiles but ran each layer's last 130-row tile as a complete BF16
GPU FFN, rather than padding it to 1024 and invoking Core ML. With the
same seed-17 three-reference input dumps and matching five-step warmups,
the last-16-final-reuse route changed **7.991/8.001 → 7.923/7.955 s**,
only ~0.7% median improvement. It saved 16 first-step Core ML calls, but
shifted the dragon sticker's eyes and outline (RGB correlation 0.99717,
RMSE 0.02470). The probe passed cancellation, GPU switch and unload; an
initial pilot aborted at the expected-call-count assertion before that
assertion was updated to account for fewer predictions. The tiny benefit
and visual drift do not justify another runtime path. The switch, call
count adjustment and candidate-only contract/verifier branches were
removed. Raw diagnostic outputs remain ignored under
`results/qwen21/session-ref512-edit3-w6144-fusedqkv-last16-finalreuse16-gputail-*`.
The rebuilt ordinary library and one last-16-reuse request then passed the
probe's lifecycle checks, made the expected **192** Core ML calls and
produced a PNG byte-identical to the pre-experiment candidate; request
wall was 8.054 s. `make test-qwen21` and `git diff --check` also passed.

An additional *pure GPU*, synchronized block-0 operator profile on the
matching three resized-512 references (4226 rows) measured first-step
gate/up plus down/residual at 62.64 + 30.91 ms, QKV at 30.22 ms,
Q/K norm/RoPE at 12.96 ms and all attention at 14.34 ms. It wrote a PNG
byte-identical to the saved uninstrumented GPU oracle. This is one
instrumented block, not an end-to-end contribution breakdown or a figure
to multiply by 32. A separate *random-input* 4226-row QKV microprobe
screened Metal BM16/32/48/64: 28.93/28.71/28.96/31.14 ms respectively;
BM32 is already the best of these samples, so BM tuning does not offer a
credible ~0.2-second request saving. An exact long-sequence GPU FFN or
memory-traffic improvement remains the higher-value next experiment;
it needs checkpoint-matched PNG and warm end-to-end validation.
