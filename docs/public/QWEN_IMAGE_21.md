# Qwen-Image-2.1 in TurboCider

TurboCider runs Comfy-Org/Qwen-Image-2.1 natively through C++/MLX/Metal. Python
is used only for offline validators and development diagnostics; inference is
performed by the native runtime.

## Supported App/CLI paths

- Text-to-image and image editing with the BF16 Comfy checkpoint.
- RGBA generation, transparent subject extraction, and transparent-layer edits.
- One to ten ordered image references. References are encoded as visual
  conditions; a separate black/white mask is an additional visual reference,
  not hard pixel-preserving inpainting.
- Ellipse/brush annotation references, prefix KV caching, staged or resident
  GPU execution, and native App job persistence/reuse.
- Experimental PE-T2I and explicitly opt-in PE-I2I FP32 visual conditioning.

The App exposes Qwen-specific examples, ten-reference import limits, RGBA
handling, mask/annotation authoring, and the PE-I2I warning. Use the GPU route
for normal operation. The experimental GPU+ANE route requires an explicit
manifest and approximation opt-in; it is not hardware-placement or image-quality
proof.

## Base-schedule LoRA and optional DiT cache

The App's right settings column exposes **DiT cache** for 512×512, 20–40-step
GPU generation and 1–3-reference editing with normal 1024px reference processing
and prompt enhancement off. It defaults to **Off**. The named choices are
`off`, `conservative`, `balanced` and `fast`; the selected value is recorded in
the draft, submitted request and result. Unsupported combinations show a reason
instead of silently changing the sampling steps. Viggle's six-step student is
outside this cache's supported range.

For schema v1, set `qwen21_dit_cache` at the top level; for schema v2 put it
under `execution`. Any enabled mode also requires `allow_approximation: true`.
This reuses the aggregate residual of the middle 24 transformer blocks within
one sampling request. The first eight blocks always run, the first eight steps
warm the cache, and the last step always computes all blocks. It can change
texture, color and local details. Results report the actual cached-step and
saved-block counts in `qwen21_dbcache`; selecting a mode alone does not prove
that it saved work.

Compatible ordinary transformer LoRAs use the **base schedule**, with 20–40
steps on the qualified 512×512 GPU route. They remain separate runtime low-rank
matrices, with adjustable strength, and may opt into DiT cache. A known Viggle
v0.2.1 filename still requires its pinned contents and six-step settings; it
cannot silently become an ordinary adapter. Unknown adapter quality and training
settings cannot be established from tensor compatibility alone.

DiT cache is separate from cross-request editing prefix reuse. The latter retains
a bounded completed K/V bank for identical instructions and ordered reference
contents, including new seeds, on qualified GPU edits. It is memory-only and
cleared by unload. In this release DiT cache bypasses that prefix snapshot, so
benchmark gains must not be multiplied together. See the
[editing startup investigation](../status/2026-09-30-qwen21-edit-startup.md).

Start with **Balanced** for a speed/detail tradeoff, **Conservative** when local
detail matters more, and **Fast** for previews. Keep **Off** for full-step sampling.
The local qualification covers 512×512 generation and single-reference editing:
Base at 25/40 steps, and the supplied rank-16 ordinary LoRA at 25 steps and
strength 1.0. This is a tested strength, not an adapter-author recommendation.
Other prompts, adapters and reference counts can behave differently. See the
[DiT-cache measurements and correctness report](../status/2026-09-30-qwen21-dit-cache.md)
for timing controls, image metrics and the repeated-edit caveat.

## Viggle v0.2.1 r128 / r256: six-step GPU student (research/evaluation only)

The public GPU route also accepts the official **v0.2.1 six-step r128**
adapter, with the same schedule, strength 1 and 227 projection bindings.
Its SHA-256 is
`bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b`.
The runtime checks the filename and its corresponding digest; renaming a
different adapter does not qualify it. The two ranks remain different
adapters, so the historical r256 timings below do not establish r128 speed
or quality. See the [upstream adapter card](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo).
The [2026-09-30 GPU and App verification](../status/dev-verify-qwen21-gpu-uiux-2026-09-30.md)
records the r128 generation/editing matrix, lifecycle checks, quality limits
and the LoRA-only workflow for subsequent iterations.

For an existing Comfy installation, keep its three BF16 files in
`diffusion_models/`, `text_encoders/` and `vae/`. The native encoder also
needs `processor/tokenizer.json` from Qwen/Qwen-Image-2.1; this small
tokenizer configuration is separate from the weights. A directory containing
only the three checkpoints is incomplete. No second set of model weights is
needed. In the App, add the supported adapter and apply **512×512 / 6-step
GPU preset**. It preserves the prompt and references, selects strength 1 and
inference-time LoRA, and explicitly marks the distilled sampling approximate.
Disabling or removing the adapter excludes it from the next request; reset
the base model to 40 steps for a meaningful quality comparison.

The optional `Viggle/Qwen-Image-2.1-viggle-turbo` **v0.2.1 6-step r256**
adapter can run on the existing BF16 Qwen-Image-2.1 checkpoint for text-to-image
and 1–3-reference editing. Download the specifically named LoRA into
`models/Viggle-Qwen-Image-2.1-viggle-turbo/` (for example through the Hugging
Face CLI or a trusted mirror); the runtime verifies SHA-256
`2a0148f5c73abbed5f97da5ea356e439318aadb281d01fce4af39cdf43728803`
before binding all 227 projections. The locally tested file came from revision
`bb26a0f38e5fe6c124aaccc9187a87eed5d9ed13`. Download and read the model's
`LICENSE` and `NOTICE`: it is **non-commercial research/evaluation only**.

The adapter remains separate at inference time (`W x + B A x`); do **not**
premerge into the BF16 checkpoint, because BF16 rounding loses parts of the
distilled update. The native sampler uses the shipped raw six nodes
`[1, .9375, .875, .75, .5, .25]`, Qwen21's resolution-dependent shift, a zero
endpoint, and no base-model terminal stretch. Use scale 1, six steps, explicit
`allow_approximation: true`, 512×512, `execution: "gpu"`, and the default 1024px
reference encoding for edits. GPU+ANE W8A8 is rejected by default. The explicit
`lora_fused` route now computes the complete runtime adapter using a reusable,
base-only Core ML graph; it does not merge LoRA weights and has not established
a stable whole-request advantage over GPU. See the
[current routes and portable CLI examples](../status/runtime-lora-acceleration-2026-09-28.md).
The older `lora_suffix` experiment omits ANE-prefix LoRA; its faster results
are not complete-LoRA speedups.
FFN step reuse remains unsupported with this adapter. The previous BF16 GPU path
remains available without `loras`.

For faster **r128 GPU editing**, the request may explicitly select
`qwen21_reference_size: 512` without setting a diagnostic environment variable.
This requires a 512×512 output, 1–3 ordered references, six steps, strength 1,
`hybrid_mlp_mode: "auto"`, inference-time LoRA, prompt enhancement and DiT cache
off, and `allow_approximation: true`. The r128 filename is recognized during
planning; its pinned SHA-256 is still checked before binding. This request
opt-in does not widen ordinary-LoRA, r256 or GPU+ANE routes. Their existing
full-size or explicitly gated diagnostic rules remain in force.

The default is still **1024**. Reference size is an approximate squared-pixel
area: aspect ratio is preserved and dimensions are aligned to 32 pixels, so a
non-square reference is not forced into a 512×512 square. Reducing it can lose
small details and change the edited image. It changes reference encoding,
not output resolution. Base-model editing already supports explicit 256/512
reference resizing with approximation enabled; ordinary LoRA continues to
require 1024. The App's 512 shortcut uses the qualified base 20–40-step or
Viggle r128 six-step GPU settings.

In schema v1, add the field at the top level. In schema v2 use
`"parameters": {"qwen21_reference_size": 512}`; keep `allow_approximation`
under `execution`. `models` discovery reports the field locations and limits
in `reference_encoding`. Run `plan` to validate the complete request. Plans
and results report `qwen21_reference_size` and the
`qwen21_reference_resize_512` approximation; qualified r128 requests also
report `qwen21_viggle_r128_reference_resize_512`. Cross-request prefix snapshot
reuse currently requires 1024-reference processing and is bypassed at 512;
the conditioning cache still distinguishes the two sizes.

```json
{
  "model": "qwen-image-2.1", "operation": "image.generate",
  "prompt": "A red fox in falling snow beside pine trees",
  "output": "results/fox.png", "width": 512, "height": 512,
  "steps": 6, "seed": 42, "frames": 1, "audio": false,
  "execution": "gpu", "allow_approximation": true,
  "lora_strategy": "inference_time",
  "loras": [{"path": "models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors",
             "role": "transformer", "strength": 1.0}]
}
```

For editing, change `operation` to `image.edit` and add 1–3 ordered
`inputs` with `{"kind":"image","role":"reference","path":"..."}`. Run
`build/native/turbocider plan request.json` and then
`build/native/turbocider generate models/Comfy-Org-Qwen-Image-2.1 request.json`.
On one 512² fox prompt/seed, two prepared, prompt-cached resident requests
took **8.187/8.229 s** with the adapter versus **43.180/43.157 s** with the
40-step base GPU model (**5.26×** median request-wall speedup); matched
six-step base GPU was **7.073/7.049 s**, so the runtime LoRA makes each step
costlier, while reducing the number of steps. This is a *distilled model with
a different schedule and result*, not lossless acceleration. The fox stays
recognizable; its pose, face and snow differ. Single-, two- and three-reference
examples all exported; the one-reference matte-red request stayed glossy,
and the three-reference dragon sticker was redrawn as a head close-up. These
are qualitative samples, not a general edit-fidelity pass. Full timing and
quality notes: [Viggle validation](../status/qwen21-viggle-v021-r256-2026-09-26.md).

For a separate, explicitly approximate GPU kernel option *on this adapter*,
set `TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE=1` when running the CLI. On two
matched prepared resident repetitions, fused Q/K was about **1.024×** faster
for both text-to-image (8.033/7.996 s) and two-full-size-reference editing
(21.471/21.385 s) relative to the same adapter without the kernel; all input
tensors matched. Output RGB correlation was 0.99908 (fox) and 0.99643 (two
references), but pixels changed. This is a limited opt-in, not a new default
or a substitute for user review of material/detail preservation.

For the fastest **tested** Viggle GPU diagnostic, combine that switch with
`TURBOCIDER_QWEN21_VIGGLE_LORA_FP16=1`. It computes only the runtime LoRA's
rank-sized matmuls in FP16 and still adds their result in FP32; the BF16 base
weights remain unchanged. The two switches together measured **7.882/7.858
s** on text-to-image and **20.988/21.044 s** on two-reference editing, about
**1.043×/1.044×** against the plain six-step adapter. Exact dumped inputs
matched; RGB correlations were 0.99965/0.99885. The one-/three-reference
samples had one prepared run each: **14.147 s** (1.039×) and **28.695 s**
(1.049×), with the same visible matte/sticker limitations. These are not
general quality qualifications. Leave both switches off when detail fidelity
matters more than a few percent speed.

For a separate **GPU-only** short-step experiment, set
`TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN=1` and explicitly request
`execution: "gpu"`, `allow_approximation: true`, 512×512 and at least 3 steps.
The final denoise step reuses the previous step's 32 FFN outputs while still
running current-step attention and modulation; 1–3-reference edits additionally
require `qwen21_reference_size: 256`. It is disabled by default and cannot be
combined with the GPU+ANE path. On one matched 5-step text-to-image and
1/2/3-reference workload, prepared warm-request speedups were approximately
**1.126× / 1.125× / 1.112× / 1.107×**; two matched 40-step cases were only
about **1.01–1.02×**. Images were close in these samples, not broadly
quality-qualified. See the [GPU FFN cache report](../status/qwen21-gpu-final-ffn-cache-2026-09-26.md).
For the fastest **tested 512², 5-step GPU-only diagnostic combination** on
this M4 Max, also set `TURBOCIDER_QWEN21_METAL_QK_ROPE=1` when invoking the
CLI. Both environment flags must be explicitly set; the request still needs
`execution: "gpu"` and `allow_approximation: true` (and 256px references for
editing). Prepared warm-request speedups against matching default GPU were
**1.145×** for text-to-image and **1.139× / 1.136× / 1.126×** for 1/2/3
reference edits. The same combined flags yielded only **1.033×** on tested
40-step text-to-image and two-reference edits. These are single-machine,
limited-prompt diagnostics, not a quality-qualified default or cold-start
speed guarantee; both flags work through the normal CLI `generate` command.

There is also an opt-in **GPU-only fused Q/K RMSNorm+RoPE Metal kernel**:
`TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE=1`. It requires a 512×512 GPU request
with `allow_approximation: true` and cannot be combined with
`TURBOCIDER_QWEN21_METAL_QK_ROPE=1`. Against plain GPU, two-run warm-Session
request speedups were about **1.02×** on the tested 5-step text-to-image and
1/2/3-reference edits, and **1.019×** on the tested 40-step two-reference edit.
The final image is visually similar but not bit-identical. When combined with
final-step FFN reuse, it was only about **0.3–0.4%** faster than the previous
paired-RoPE plus FFN-reuse combination on matched samples; this is too small
to recommend it as the fastest option. Defaults and the existing W8A8
GPU/ANE route remain unchanged. See the
[fused GPU kernel ablation](../status/qwen21-gpu-fused-qk-norm-rope-2026-09-26.md)
for paired inputs, image differences and the mixed-route comparison.

For a separate **speed-first, approximate 5-step GPU-only experiment** on
512×512, keep the paired-RoPE and final-FFN-reuse flags above and additionally
set `TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN=1`. The request must
still opt into `allow_approximation: true`; 1–3-reference edits require 256px
references. This reuses even-numbered FFN layers on the penultimate step and
all layers on the final step; it is not a kernel-equivalent computation. One
matched two-run resident test measured approximately **1.226×** text-to-image
and **1.219× / 1.204× / 1.193×** for 1/2/3-reference edits against plain
GPU. The three-reference sticker detail visibly changed. At 40 steps, the
tested two-reference improvement was only about **1.041×** against plain GPU;
it is disabled by default, not broadly quality-qualified, and is not an ANE
speedup. See the [half-layer FFN experiment](../status/qwen21-gpu-penultimate-half-ffn-2026-09-26.md).

## Explicit 512² GPU/Core ML experiments

The default remains BF16 GPU, with references resized to approximately 1024
pixels. There are two opt-in mixed routes: a 32-layer FP16 Core ML FFN prefix
for 512² text-to-image, and a W8A8 Core ML prefix for 512² text-to-image or
1–3-reference editing. The W8A8 editing candidates keep a BF16 GPU FFN
suffix and explicitly choose 256px or 512px reference resize. A separate
original-size 1024px-reference diagnostic leaves its long first-step FFNs
on GPU. The speed-first 6144-channel candidate uses all 32/32 decode FFN layers in parallel;
the alternative runs layers 3, 5 and 7 entirely on GPU and covers 29/32
decode FFNs (90.625%). These reference dimensions change the input and can lose
details; this mode is **not** the default or a production image-quality gate.
The resident Session now caches the text/visual conditioning and reference VAE
latents for an unchanged prompt, reference resize and ordered reference file
contents (checked by SHA-256 on every request). The historical mixed-route
measurements below used request-owned denoising prefix KV; the newer ordinary
GPU bounded snapshot described above is a separate route. Replacing a reference
in place invalidates conditioning. Under the matched 512²/40-step
two-reference, seed-17 workload, two cached GPU requests took 44.733/44.777 s
and W8A8 mixed requests took 37.159/37.176 s, about **1.204×** request-wall
speedup. At 5 steps the corresponding cached figures were 6.688/6.727 s and
6.012/5.999 s, about **1.117×**. These numbers differ from the earlier
re-encode-on-every-request result (40-step 1.130×, 5-step 0.835×), and apply
only to repeated identical editing conditions after Session preparation. They
do not establish a benefit for a new prompt/reference, cold process, other
seeds, or 1024² requests; GPU stays the default.
Separate 40-step cached resident samples with one and three 256px references
measured about **1.209×** and **1.198×** request-wall speedup respectively;
all three reference subjects remained visible in the tested three-image output.
Each reference count still has only one prompt/seed and requires broader visual
validation. Earlier three-reference tests with the default 1024px reference
resize were slower on the mixed route; these are different inputs.

An explicitly selected **6144-channel W8A8 with GPU fallback 3/5/7** was
measured on the same 512²/40-step prepared, condition-cached 1/2/3-reference
workloads. Its request-wall speedups over matched GPU were **1.357× / 1.348× /
1.338×**, compared with **1.209× / 1.204× / 1.198×** for the established
4096 partition. For the same two-reference input at 5 steps, a new
condition-cached comparison measured GPU 6.706/6.707 s, 4096 W8A8
6.003/5.996 s, and 6144 W8A8 5.560/5.543 s (6144: **1.208×** request-wall
vs GPU). The earlier 5-step 0.835× test had `prompt_cache_hit=false`;
it does not describe this repeated-input cache-hit scenario.

The faster explicit 6144-channel **full 32-layer edit** achieved about
**1.410× / 1.400× / 1.387×** request-wall speedup for 1/2/3 references at
40 steps; two-reference 5-step cached requests achieved **1.234×**. To use
it, point `ane_manifest` at a checkpoint-matched 1024-row, 32-block,
6144-channel compiled manifest and leave `qwen21_gpu_full_ffn_blocks` empty.
No global manifest path is bundled: a local model must first be exported and
compiled from the same checkpoint. Retain `[3, 5, 7]` with either the 4096
or 6144 manifest when reference details matter: on this three-reference
sample the full-coverage route altered the left teapot's spout/handle more
than the fallback. On the two-reference sample, however, full coverage was
closer to GPU (RGB correlation 0.944 versus 0.928 for fallback). Neither
route establishes physical ANE occupancy or general image fidelity; GPU
remains the default. See the
[full-coverage edit comparison](../status/qwen21-w8a8-full32-edit.md)
for timings and visual caveats.

A separate **diagnostic-only full-size-reference editing** option is now
available at 512²: set `TURBOCIDER_QWEN21_FULL_REF_W8A8_DIAGNOSTIC=1`, use
`qwen21_w8a8: true`, `qwen21_reference_size: 1024`, `execution: "gpu_ane"`,
`allow_approximation: true`, a checkpoint-matched 1024-row/6144-channel
compiled manifest, and empty `qwen21_gpu_full_ffn_blocks`. Only 1–3 ordered
editing references are permitted. The 32/32 W8A8 ANE FFN coverage uses the
same output latent rows while the full reference tokens remain on the GPU;
there is no 256px-reference information loss. On one prompt/seed per reference
count, prepared resident 5-step request-wall speedups over **same-input** GPU
were approximately **1.121× / 1.065× / 1.042×** for 1/2/3 references; at 40
steps the corresponding samples were **1.299× / 1.160× / 1.149×**. The
three-reference 40-step sticker placement/size visibly changed, and both
5-step outputs had ghosting. The tested W8A16 GPU suffix was again slower
than BF16 (BF16/W8A16 median **0.986×** on the full-size two-reference 5-step
sample). This is not a quality-qualified default or cold-start speed claim;
see the [full-reference diagnostic report](../status/qwen21-w8a8-fullref-512-diagnostics-2026-09-26.md).

For **20–40-step base-model** requests where the user accepts visible
texture/shadow differences, `TURBOCIDER_QWEN21_DBCACHE_DIAGNOSTIC=1` enables
an independent, request-scoped decode residual cache on either BF16 GPU or
the validated W8A8 GPU/ANE route. For the tested speed-first policy, set
`TURBOCIDER_QWEN21_DBCACHE_THRESHOLD=0.25` and
`TURBOCIDER_QWEN21_DBCACHE_MAX_CONSECUTIVE=8`, with
`allow_approximation: true` in the request. The default policy remains much
more conservative and may not skip any steps; no cache is enabled by default.
On one 512², 40-step, **two original-size-reference** edit, fully warmed
resident GPU went from 68.414 to 38.224 s, and a checkpoint-matched
6144-channel W8A8 hybrid went from 57.308 to 32.797 s; the cached hybrid
was 1.165× faster than the cached GPU. The two teapots, positions and
recognizable details were retained on visual inspection, but material and
shadow pixels changed. This is not a cold-CLI timing or a general speed or
quality guarantee. See the [DBCache status](../status/qwen21-dbcache-2026-09-27.md)
for the 20/40-step, 0–3-reference matrix and the matched input checks.

Looping the same 1024-row W8A8 graph over the **first eight** FFNs of a
three-full-reference edit shortened a five-step warm request by about 4.4%,
but visibly doubled/glowed the central sticker; the temporary route was
removed. See the [rejected first-eight screen](../status/qwen21-fullref-first8-tiled-prefill-screen-2026-09-27.md).

A more conservative **opt-in full-reference research path** uses
`TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION=3` to keep the first 16 blocks'
cross-reference attention exact and restrict only later reference queries
in the last 16 blocks. The 512² three-reference, five-step pure-GPU request
was ~1.044× faster on two seeds with recognizable subjects; at 40 steps
the single-run saving was only ~1.1%. On the full-reference W8A8/BF16
hybrid, add `TURBOCIDER_QWEN21_FULL_REF_W8A8_DIAGNOSTIC=1`; combining mode
3 with the existing final-block target-only diagnostic measured **22.80 s**
against **24.23 s** for the same-input decode-only hybrid (about 1.063×).
The target-only addition produced the same PNG as mode 3 without it in
this fixture. Relative to the matched **25.22 s** exact GPU baseline, this
mixed route is about 1.106× faster, but it changes reference attention and
FFN precision, so it is neither a GPU-kernel-only improvement nor a default
quality guarantee. The stronger Q/K-fused GPU combination was faster but
blurred the sticker; avoid it when editing fidelity matters. See the
[late-block edit screen](../status/qwen21-fullref-last16-local-2026-09-27.md)
for flags, visual caveats, full warm-request comparisons and lifecycle checks.

With the same **original-size references**, a long-lived resident Session
can additionally enable `TURBOCIDER_QWEN21_RESIDENT_PREFIX_KV=1` for repeated
identical edits. In a three-reference, five-step matched run, the *first*
request was GPU 23.857 s versus W8A8 hybrid 22.825 s; three later **prefix
hits** had GPU/hybrid median wall 8.732/7.890 s, about **1.107×**. These
numbers do not compare a cache hit against a miss. Inputs were tensor-equal;
both teapots and the sticker remain recognizable, with small visible detail
changes. Memory use is higher and only one prompt/seed was checked. This
route does not tile full-reference FFNs, and a one-shot CLI has no repeat
hit. See the [steady-prefix comparison](../status/qwen21-fullref-prefix-steady-2026-09-27.md).
Offloading the 1,024 target FFN rows on the *first step of a prefix hit* to
the same W8A8 graphs saved about 4–6% in one three-reference screen but
visibly distorted the sticker face; that additional candidate was removed.
See the [rejected hit-FFN screen](../status/qwen21-fullref-prefix-hit-w8a8-rejected-2026-09-27.md).

For an explicit **five-step, resized-512-reference** speed-first edit, the
last 16 first-step FFNs can loop over 1,024-token W8A8 tiles in the *same*
Core ML models, while GPU executes the complementary channels. With the
screened final/penultimate FFN reuse and fused-QKV diagnostics, a three-image
first request measured **7.706 s** against **10.114 s** for the same-input
pure-GPU control. Optional resident prefix-KV reuse reduced a subsequent
identical-condition request to **4.887 s**; this hit must not be compared with
the uncached GPU control. These are visibly lossy, explicitly gated research
options: a one-shot CLI call receives no cache hit; LoRA, full-size references
and other steps are excluded. GPU and the lower-loss hybrid remain available.
See the [tiled-prefix experiment](../status/qwen21-tiled-prefix-kv-edit-2026-09-27.md)
for flags, matched checks, memory and image-quality caveats.

For the *same repeated-edit workload*, an additional explicit
`TURBOCIDER_QWEN21_TILED_PREFIX_TARGET_ONLY_DIAGNOSTIC=1` computes the
1,024 target FFN rows with one Core ML call per eligible first-step layer
on a prefix-KV hit. Three-reference hit calls fall from 127 to 112; tested
one/two/three-reference request walls were **4.185 / 4.637 / 4.673 s**.
Matched strict-tail PNGs were byte-identical in these fixtures, but the
option remains opt-in pending wider prompt and device checks. It offers
**no first-request or one-shot CLI hit**. See the
[target-only follow-up](../status/qwen21-prefix-hit-target-only-2026-09-27.md).

For explicit **512² text-to-image** with a checkpoint-matched 6144-channel
W8A8 manifest, use `operation: "image.generate"`, no `inputs`,
`qwen21_w8a8: true`, `execution: "gpu_ane"`, and
`allow_approximation: true`. Leave `qwen21_gpu_full_ffn_blocks` empty for the
32/32-layer route. A single fox prompt/seed in a prepared, prompt-cached
resident Session measured GPU **5.999/5.997 s** versus W8A8
**5.042/4.820 s** at 5 steps (about **1.216×** request-wall), and
**43.184/43.223 s** versus **30.468/30.489 s** at 40 steps
(about **1.418×**). The 40-step fox remains recognizable, with differences
around its legs and tail; an independent glass-bottle-and-cactus prompt/seed
also gave visually similar results in a diagnostic generator. These are
limited samples, not a default quality or cold-start speed guarantee; the
[text-to-image evidence](../status/qwen21-w8a8-6144-t2i.md) distinguishes
per-step timing from complete resident and first requests.

For the speed-first explicit W8A8 edit candidate, use schema 1 fields like
these with a compiled 1024-row, 32-layer, checkpoint-matched, per-tensor
**6144-channel** W8A8 manifest:

```json
{
  "model": "qwen-image-2.1",
  "operation": "image.edit",
  "prompt": "Compose these two referenced objects in one scene.",
  "output": "results/edit.png",
  "width": 512, "height": 512, "steps": 40, "seed": 17,
  "frames": 1, "audio": false,
  "execution": "gpu_ane", "allow_approximation": true,
  "ane_manifest": "path/to/compiled/manifest-HASH.json",
  "qwen21_w8a8": true,
  "qwen21_reference_size": 256,
  "qwen21_gpu_full_ffn_blocks": [],
  "inputs": [
    {"kind": "image", "role": "reference", "path": "path/to/first.png"},
    {"kind": "image", "role": "reference", "path": "path/to/second.png"}
  ]
}
```

Run `build/native/turbocider plan request.json` to check the policy and
`build/native/turbocider generate path/to/model request.json` to verify the
manifest and generate an image. Schema 2 puts the W8A8 and fallback flags
under `execution`, and `qwen21_reference_size` under `parameters`. W8A16 GPU
suffix is an additional explicit research option (`qwen21_gpu_w8a16: true`),
but tested slower than the BF16 suffix; do not enable it for speed by default.
On the matched 6144-channel W8A8, 512²/two-reference/5-step resident input,
the BF16 suffix took **5.420/5.437 s** and the W8A16 suffix took
**5.768/6.022 s** (BF16/W8A16 median **0.921×**). Exact text, initial-noise
and both reference tensors matched; this measures the GPU suffix, not faster
or slower ANE int8 compute.
For the conservative W8A8 editing option, set `qwen21_gpu_full_ffn_blocks`
to `[3, 5, 7]`; a 4096-channel edit manifest **requires** this fallback at
runtime even though `plan` does not load or inspect the manifest.
The 1024² W8A8 text-to-image results remain diagnostic-only. A local
`TURBOCIDER_QWEN21_1024_W8A8_DIAGNOSTIC=1` opt-in now permits explicit
1024² W8A8 text-to-image with a checkpoint-matched 4096-row/4096-channel
compiled manifest for resident Session benchmarking, but does not qualify
1024² editing or change the default GPU policy. The measured warm-request
speedups are 1.114× at 5 steps and 1.159× at 40 steps on one prompt/seed;
physical ANE placement and broad image quality remain unverified. See the
[512²](../status/qwen21-w8a8-ane-diagnostics-2026-09-25.md) and
[1024²](../status/qwen21-w8a8-ane-1024-diagnostics.md) reports for timing
scope, visual differences, and remaining validation work.

## 512² validation snapshot

The maintained regression scope is 512². Evidence includes native GPU text
generation. App model selection and prompt examples default to 512×512;
reference encoding geometry is independent of that output canvas. Coverage includes
RGBA export, prefix-cache/session lifecycle, one-step progress
events (`0/steps` through `steps/steps`), and App transport/workflow tests.
The matched 512² GPU comparison against mflux is recorded in
`results/qwen21/matched-refresh-mflux-512-report.json`; native cached denoise
was measured at approximately 1.068× the mflux denoise speed in that matched
workload, with latent relative RMSE 0.007073 and RGB correlation 0.999890.

Additional completed samples exercise 1024² edits, transparent extraction,
annotation/mask edits, three-reference composition, and the ten-reference
runtime boundary. They are representative evidence, not a blanket perceptual
quality gate.

## Correctness and limitations

Run `make test-qwen21` after building the native/App targets. The target runs
focused report, mask-analysis, native contract, and App workflow checks without
starting inference. Full reports and experiment history are in
`docs/status/qwen21-convergence-2026-09-22.md` and the local
`notes/2026-09-21-qwen-image-21.md`. NumPy is an offline-test dependency;
Pillow is only needed for the image-report CLIs, not these unit tests.

Known limitations are intentional and explicit: BF16 Qwen3.5 visual parity
does not pass the strict reference gate, PE-I2I remains experimental FP32,
edit preservation is not pixel-locked, JPEG decoder parity has small byte
differences, and actual ANE occupancy cannot be proven without privileged
hardware tracing. The 2K path is exposed by the model contract but is outside
the maintained test scope; do not use its incomplete experiment as evidence
for production quality or performance.

## Phase-scoped Runtime FFN diagnostic

The explicit runtime FFN route can be tested at 512×512 with
`TURBOCIDER_QWEN21_RUNTIME_STAGED_DIAGNOSTIC=1` and
`residency: "component_staged"`. It requires standard 1024 reference encoding,
0–3 references, no prompt enhancement, and the existing runtime approximation
and cache restrictions. Only base or pinned Viggle v0.2.1 r128 at six steps and
strength 1 are admitted by this diagnostic. It does not enable encoder ANE,
QKV or an App performance preset.

The runtime owner is destroyed after denoising, before VAE decode, and after
preparation/cancellation. Graph load and self-test therefore recur on each
request. `hybrid.runtime_weight.session_released` reports owner destruction;
`counter_scope` distinguishes its counters from the longer-lived model engine.
Prepared GPU weights may still be retained. Existing memory guards remain in
force. See [lifecycle and validation](../status/2026-10-02-runtime-staged-lifecycle.md)
for the tested scope and performance evidence.
