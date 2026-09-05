# h3-metal

Native MiniMax-H3 inference for Apple Silicon. The project is being built as a
sequence of working vertical slices: deterministic host/model metadata first,
then portable Metal block parity, prompt encoding, prompt-to-video/audio, and
first/last-frame conditioning and then ordered references.

Prompt-to-video/audio, first/last-frame conditioning, and ordered Ref2VA
image/video/audio references work end to end. The current work is incremental
H3-specific Metal performance and memory optimization on M3 Max and M5 Max.

## Tutorial

### 1. Build and inspect the model

The examples assume that the Hugging Face snapshot is in `./MiniMax-H3` and
that FFmpeg and FFprobe are available on `PATH`.

```sh
make -j8
mkdir -p outputs
./h3 --info -d ./MiniMax-H3
```

`--info` checks the model layout and prints the selected Metal device without
mapping all weights or generating media. Run `./h3 --help` for the complete CLI
reference.

For the current local quality-first Super checkpoint, download the pinned
`lightx2v/Minimax-h3-Turbo` file
`minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors` at revision
`2f8ea0dc0a7e2b26c9a43124eb89673787189b4e`, then merge it into the FL2VA
transformer once:

```sh
python3 tools/merge_h3_lora.py \
  ./MiniMax-H3/FL2VA/transformer \
  ./models/LightX2V-Minimax-H3-Turbo/minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors \
  ./models/MiniMax-H3-LightX2V-v1.1-768p-Turbo/FL2VA/transformer \
  --profile lightx2v-4step --strength 1.0 --device mps
```

The v1.1 adapter declares alpha/rank `128/128`, so the required merge strength
is `1.0`; its published video/audio flow shifts are `6/3`. The merger records
those values in `h3-turbo-merge-manifest.json`, and Super reads the manifest to
select the matching native schedule. Mixing the v1.1 weights with the older
`0.0625` strength or video shift `12` is rejected. The merger also validates the
canonical file SHA-256, all 624 tensors, 312 A/B pairs, rank-128 shapes, the 50
main plus two token-refiner blocks, and the 312-to-208 native mapping. Q/K/V
deltas are applied to the corresponding slices of the fused native QKV weight.
Each output shard is written atomically and recorded with its base/output SHA;
an interrupted merge may resume only when the identity-bound state and shard
hashes match. Use `--check-only --device cpu` before writing the roughly 62 GiB
output.

Sana Super v2 itself pins the older 544p v0.1 artifact. For exact Sana Stage-1
provenance, keep using
`minimax_h3_fl2v_turbo_4step_v0.1.safetensors` at revision
`050494d5fe05bd1b1140b8565ea51dc33a5085a5` with strength `0.0625` and flow
shifts `12/3`. The local v1.1 recommendation deliberately trades strict Sana
Stage-1 identity for the newer 768p training contract and measurably better
person geometry.

The older `minimax_h3_turbo_v4_step600_ema.safetensors` is a different native
259-pair/rank-64-or-16 adapter, not the Sana LightX2V artifact. It remains
supported with `--profile native --strength 1.0`, but its output must not be
described as Stage-1 weight parity with Sana Super. In either case, reuse or
link the base `model_index.json`, tokenizer, processor, text encoder, and
audio/video VAEs in the new model root. A base checkpoint run with `--steps 4`
is not equivalent to either Turbo LoRA model.

Without `-p`, the same binary starts an Iris-style interactive session:

```sh
./h3 -d ./MiniMax-H3 --width 512 --height 512 --steps 6
```

Type a prompt to generate a numbered video. The session keeps the exact BF16
prompt conditioning for an identical prompt. For a different prompt with a
compatible model/schedule/backend configuration, it now releases the old
request-sized RoPE/maps/activation arena before Qwen runs, retains the loaded
DiT weights, Metal runtime, AdaLN schedule, and active-block policy, then
rebuilds only request-dependent state. A 320-square, 40-block real-model gate
measured 9.092 seconds for the original load versus 0.738 seconds for resident
reprepare, with byte-identical video/audio output across different text content
and token counts. The TAEH3 decoder is also retained for a compatible latent
canvas. Repeating an identical prompt still avoids both model loading and text
encoding. Useful commands are `!status`, `!seed random`, `!seconds 2`, `!show`,
`!save output.mp4`, and `!cache`. Use `!help` for the full, short list.

First/last-frame conditioning is persistent in the session:

```text
h3> !first opening.png
h3> !last ending.png
h3> The camera moves slowly around the subject.
```

Use `!first clear` or `!last clear` to remove an anchor. Generated videos are
written to the session directory printed at startup.

For a general Ref2VA conditioning image, use `!ref-image PATH` instead. Images
are appended in order and exposed to the model as `<Picture 1>`, `<Picture 2>`,
and so on; filenames have no meaning to the model.

```text
h3> !ref-image person.png
h3> Make the person shown in Picture 1 wave to the camera.
```

`!refs` lists the current order, `!ref-remove N` removes one entry, and
`!refs clear` removes them all. Ref2VA references cannot be mixed with
`!first`/`!last` anchors.

### 2. Make a first fast video

Start with the validated balanced preset. It generates 22 frames at 24 fps
(about 0.92 seconds), displays the evolving middle-video frame after every
denoising transition in a supported graphical terminal, and prints phase
timings:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 \
  --frames 22 --steps 20 \
  --layers 45 --reuse 2 \
  --show \
  -o outputs/fox-fast.mp4
```

This is deliberately not the most aggressive configuration:

- `--steps 20` performs the default 20 denoising passes.
- `--reuse 2` computes 11 fresh denoiser velocities instead of all 20 and
  extrapolates the skipped transitions.
- `--layers 45` runs 45 of the 50 transformer blocks, reducing both time and
  unified-memory use.
- `--show` is optional. It supports Kitty/Ghostty and
  iTerm2/WezTerm/Konsole graphical protocols. It loads a resident preview VAE,
  displays one representative middle-video frame after every Euler transition,
  and then displays all final frames. Display dimensions default to 2x so the
  image has its intended logical size on macOS Retina screens; use `--zoom 1`
  on a non-HiDPI display. This adds preview decode time and roughly 10 GiB of
  temporary model residency; runs without `--show` are unchanged.
- `--profile` is optional and does not select a different generation path.

The first process invocation also pays model loading and filesystem-cache
costs. Compare performance using repeated runs, and alternate variants when
the machines are warming up because this workload is sensitive to thermal
throttling.

For a very short iteration, request four denoising passes directly:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur." \
  --width 512 --height 512 --frames 22 \
  --steps 4 --layers 50 --reuse 1 \
  --show \
  -o outputs/fox-four-step.mp4
```

`--steps N` always means exactly N denoising passes. Four through seven passes
use the same schedule that won the existing low-budget comparison; increasing
from 4 to 7 progressively improved detail and motion in that test. Eight passes
use the same released-grid schedule builder, but are not quality-approved until
the matched 4/6/8 sweep below is complete. Keep `--reuse 1` at such small
budgets so every requested pass runs the model. `--show` displays one preview
after each pass.

Several tail-heavy schedules were evaluated because most visible cleanup
happens late in a long run. They preserved too few early composition updates
and produced woven texture, weak motion, or clipped colors. The retained mode
uses the released linear base grid with one terminal point. On the 512-square,
22-frame fox test, the selected four-pass result had 0.556 full-video SSIM
against a 29-pass reference; an independent surfer test measured 0.547. The
four-pass denoise took about 3.5 seconds on M5 Max, versus 26.4 seconds for the
reference.

For a matched full-video comparison of the merged Turbo checkpoint at 4, 6,
and 8 passes, use the guarded local sweep:

```sh
tools/run_turbo_step_sweep.sh --check-only
tools/run_turbo_step_sweep.sh
```

It fixes the workload at 512x512, 124 frames, 50 blocks, `--reuse 1`, the
official Video VAE, and tile batch 2. The only inference variable is the pass
count, run in the time-symmetric order `4,6,8,8,6,4`. Each arm records the
profile, process/system memory state, final video latent, MP4/WAV/PCM, a six
frame contact sheet, ffprobe metadata, and SHA-256 hashes. After all six arms,
it computes latent and PCM max-abs/MAE/RMSE/relative-L2/cosine/SNR plus encoded
video SSIM, PSNR, and VMAF when available. Same-step repeat pairs establish the
cross-process noise floor before 4/6/8 differences are interpreted. The
cross-step reports treat the higher-step arm as the reference for asymmetric
relative-L2, SNR, and VMAF calculations. The preflight fails
closed when memory pressure, throttling, Apple GPU utilization above 5%, a
second H3 process, or Chrome/VM/model-runtime CPU load would invalidate
wall-time comparisons. Set
`BENCH_ALLOW_BUSY=1` only to test the capture path; results from that override
are not valid performance measurements. Every completed arm also compares the
system Swapouts counter before/after, `Pages throttled`, and `/usr/bin/time`
process swaps. A nonzero delta writes an `.invalid.txt` marker and prevents the
run from entering postprocessing or the matched timing set.

For a low-memory run, add `--ssd-streaming`:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 50 --reuse 1 --ssd-streaming \
  -o outputs/fox-ssd.mp4
```

This uses the original BF16 checkpoint without conversion or quantization. It
keeps two DiT blocks in memory and reads the next block from SSD while the GPU
runs the current one. On M5 Max, tracked DiT storage fell from about 36.5 GiB to
2.0 GiB at 512 square and 2.1 GiB at 864x480. A warm 50-block forward measured
1.35 versus 2.49 seconds at 512 square (84% slower), and 2.14 versus 2.68
seconds at 864x480 (26% slower). These are comparisons against the same
full-residency BF16 path, and the results were byte-identical in both checks.

The 2.0--2.1 GiB figure is the DiT's tracked tensor storage, not total system
RAM. Prompt encoding and the two VAEs run in separate phases rather than adding
their full peaks to it; the OS, media buffers, and output resolution still need
headroom. `--show` keeps a preview VAE resident and adds roughly 10 GiB, so omit
it for the lowest-memory run.

SSD streaming is an explicit memory/speed tradeoff and is not the default. It
cannot be combined with `--use-int8-row-fc2`. In an interactive session, use
`!ssd-streaming on`.

### 3. Move toward reference quality

Change one control at a time when evaluating quality. First restore all layers,
then all denoiser evaluations, and finally raise the default 20-pass schedule
to the slower 50-pass reference:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 \
  --frames 22 --steps 50 \
  --layers 50 --reuse 1 \
  -o outputs/fox-close.mp4
```

The defaults are `--steps 20 --layers 50 --reuse 1`; keep `--steps 50`
explicit for this close path. It performs 50 complete 50-block denoiser
forwards and is much more expensive than the default, but is the right oracle
when a fast mode changes the subject, anatomy, motion, or composition.
Numerical pixel identity with MLX is not expected because the random-number and
execution engines differ; the depicted content and motion should agree.

### 4. Choose a speed/quality preset

These controls are independent unless noted otherwise:

| Control | Slow reference | Default | Aggressive | Main impact |
|---|---:|---:|---:|---|
| Denoising passes | `--steps 50` | `--steps 20` | `--steps 4..7` | The number always names actual denoising passes. |
| Whole denoiser reuse | `--reuse 1` | `--reuse 2` | `--reuse 3` | At 20 steps: 20, 11, or 8 fresh DiT evaluations. |
| Active DiT blocks | `--layers 50` | `--layers 45` | `--layers 40` | Fewer blocks reduce compute and resident transformer weights. |
| Core residual reuse | `--core-reuse 1` | `--core-reuse 4` | `--core-reuse 6` | Refreshes patch/head work every step but runs the expensive core less often. |
| Token reduction | off | optional | `--token-reduction` | Pairs horizontal video tokens inside middle blocks; faster but may change composition. |
| Internal canvas | output size | `384x384` for 512 square output | `320x320` | Runs DiT/VAE smaller, then upscales with vImage. |

On M5, `--use-int8-row-fc2` uses one activation scale per FC2 row and a single
full-width TensorOps product. It is optional because it is less numerically
conservative than grouped int8. It reduced complete denoiser forwards by about
2.6% in reciprocal tests. Matched four-step fox and surfer videos kept the same
subjects, setting, and motion (full-video SSIM 0.919 and 0.828). In the
interactive session, use `!int8-row-fc2 on`.

`--reuse` and `--core-reuse` are mutually exclusive. Layer thinning can be
combined with either one.

To make the first command faster while keeping its output resolution, add
token reduction:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A surfer riding inside a sharp blue ocean wave, one rider and one white board, realistic spray." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --reuse 2 --token-reduction \
  -o outputs/surfer-fast.mp4
```

At the validated 512 square shape, token reduction cut the `45 layers + reuse
2` denoise profile from 16.69 to 12.60 seconds on the IT M5 Max. Independent
fox and surfer renders stayed coherent, but composition can diverge more from
the close path.

For an aggressive preview, render internally at 320 square and upscale to the
requested 512 square output:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walking through snow, realistic, tracking shot." \
  --width 512 --height 512 \
  --render-width 320 --render-height 320 \
  --frames 22 --steps 20 --layers 40 --reuse 3 \
  -o outputs/fox-aggressive.mp4
```

This combination produced a clean, recognizable 22-frame fox in validation,
but loses fine detail and can change framing. Do **not** add `--token-reduction`
to both `--layers 40` and `--reuse 3`: that tested combination produced color
ringing, outlines, and ghosted limbs.

As an alternative to whole-velocity reuse, this keeps the timestep-dependent
patch and output heads fresh at every transition:

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A surfer riding a blue ocean wave." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --core-reuse 4 \
  -o outputs/surfer-core-reuse.mp4
```

Use `--core-reuse 6` only as an aggressive preview. Values above 6 are not
exposed because validation lost subject fidelity.

### 5. Pick resolution and duration

Width and height must each be multiples of 32, at least 32, and their product
must not exceed `768 * 1344` pixels. Those are mechanical limits, not a promise
that every tiny canvas has good model quality. H3-Base is a 768p model.

| Canvas | Current guidance |
|---|---|
| `512x512` | Safest development size; repeatedly validated with multiple prompts. |
| `768x768` | Validated close-quality square output; substantially more expensive. |
| `1344x768`, `768x1344` | Released 768p-class landscape/portrait limit. |
| `1024x768`, `768x1024` | Valid 4:3 and 3:4 768p-class canvases. |
| `384x384` internal to `512x512` | Validated fast-quality scaling point. |
| `320x320` internal to `512x512` | Validated aggressive scaling point. |
| `256x256` | Native fast-preview canvas with automatic low-resolution RoPE adaptation. |

For a fast native 256-square preview:

```sh
./h3 -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest." \
  --width 256 --height 256 \
  --frames 22 --steps 20 \
  --layers 50 --reuse 1 \
  -o outputs/fox-256.mp4
```

At 256 square, H3 has only an `8x8` effective spatial-token grid, so it has less
room for fine detail and complex composition. H3 automatically halves spatial
RoPE coordinates at exactly 256 square. This removed repeating lattice
artifacts in long fox renders and stayed coherent on an independent portrait,
without adding tokens or runtime. Use `--use-reference-rope` to restore the
released/MLX coordinates for parity checks. Keep token reduction off at this
size. Native 128 square remains unsupported: its `4x4` token grid did not
recover a recognizable subject even with adjusted RoPE.

`--render-width` and `--render-height` must be set together, must have the same
aspect ratio as the output, and cannot exceed the output dimensions. The model
and VAE use the internal size; terminal frames and the encoded video retain the
requested output size.

H3 emits 24 fps and aligns frame requests upward to `5 + 17*n`:

Use `--seconds N` for a duration-oriented request, or `--frames N` for direct
frame control; the two options are mutually exclusive. Fractional seconds are
accepted. Seconds are converted at 24 fps and then rounded upward to the next
legal H3 temporal shape, so `--seconds 10` produces 243 frames (10.125 seconds).

| Frames | Approximate video duration |
|---:|---:|
| 22 | 0.917 seconds |
| 39 | 1.625 seconds |
| 56 | 2.333 seconds |
| 107 | 4.458 seconds |
| 243 | 10.125 seconds |
| 362 | 15.083 seconds |

Short clips are useful for development. The released workflow is intended for
roughly 4–15 second videos. A request such as `--frames 23` is rounded up to 39
frames rather than producing an arbitrary temporal shape.

### 6. Improve the prompt

A short prompt works, but the released system expects a Context-IR-like
description. State the subject, action, setting, camera, lighting/style, and
desired sound. For example:

```text
Scene: a single red fox in a snow-covered pine forest at dawn.
Action: the fox walks steadily left to right and looks toward the camera once.
Camera: medium-height lateral tracking shot, 50 mm lens, stable framing.
Look: photorealistic fur, cold blue ambient light, warm sunrise rim light.
Audio: soft footsteps in snow, light wind through pine branches, no music.
```

Keep identity and object counts explicit when they matter. `--seed N` controls
the native random stream; the default is 42. Compare options with the same
prompt, seed, resolution, frame count, and step count.

### 7. Preview frames and diagnose performance

- `--show` displays a representative frame after every denoising transition,
  followed by all frames from the completed video. Like Iris, it advertises 2x
  display dimensions by default for Retina terminals; `--zoom N` changes that
  factor without resizing the generated video or the encoded terminal image.
- `--frames-dir DIR` writes final callback frames as PPM files. Intermediate
  `--show` previews are not written there.
- `-o ''` disables MP4 encoding; combine it with `--frames-dir` when FFmpeg is
  unavailable.
- `--profile` reports phase wall time, Metal encoding/wait time, peak live
  tensor storage, cumulative allocation, and dispatch counts.

Set `H3_PROFILE_STEPS=1` together with `--profile` to submit and synchronize at
every Euler transition and print per-step wall, encode, wait, root-command GPU
timestamp, memory, and dispatch deltas. This mode deliberately adds a step
boundary for measurement, so omit it in production. The root GPU timestamp can
exclude child command buffers scheduled by MPSGraph; use wall/wait time and
matched A/B runs for conclusions.

For example:

```sh
./h3 --profile -d ./MiniMax-H3 -p "A hummingbird hovering over red flowers." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --reuse 2 --frames-dir outputs/hummingbird-frames \
  -o ''
```

### 8. Add image, video, and audio references

First/last-frame anchors select the FL2VA path:

```sh
./h3 -d ./MiniMax-H3 -p "The fox keeps walking through the snow." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --reuse 2 \
  --first-frame fox.png --last-frame fox-later.png \
  -o outputs/fox-anchored.mp4
```

Ordered references select the distinct Ref2VA checkpoint. Use the flag matching
the media semantics:

```sh
# One image reference.
./h3 -d ./MiniMax-H3 -p "Use the animal and setting in the reference." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --ref-image fox.png -o outputs/fox-reference.mp4

# Continue a clip but ignore its soundtrack.
./h3 -d ./MiniMax-H3 -p "Continue the motion in this clip." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --ref-silent-video fox.mp4 -o outputs/fox-video-reference.mp4

# Preserve the clip's embedded audio.
./h3 -d ./MiniMax-H3 -p "Continue this audiovisual scene." \
  --width 512 --height 512 --frames 56 --steps 20 \
  --ref-video fox-with-audio.mp4 -o outputs/fox-video-audio.mp4

# Replace a video's soundtrack explicitly.
./h3 -d ./MiniMax-H3 -p "Continue the scene with the supplied music." \
  --width 512 --height 512 --frames 56 --steps 20 \
  --ref-video-audio silent-fox.mp4 replacement.wav \
  -o outputs/fox-replaced-audio.mp4

# An ordered image plus standalone audio reference.
./h3 -d ./MiniMax-H3 -p "Use the animal and music from the references." \
  --width 512 --height 512 --frames 56 --steps 20 \
  --ref-image fox.png --ref-audio music.wav \
  -o outputs/fox-image-audio.mp4
```

Reference flags may be repeated and their command-line order is preserved.
Standalone audio must accompany an image or video reference. Audio references
must be 2–15 seconds; at most three audio inputs are accepted and their total
decoded duration is capped at 15 seconds.

### 9. Run the local 480p Super pipeline

`--super` runs native H3 Turbo, hands the retained TAEH3 frames and original PCM
directly to the configured local ComfyUI service, then applies the LTX-2.5
official input VAE, learned x2 latent upsampler, LTX refiner, Metal Sol-Attn,
and TAEHV wide decoder. The strict/default LTX schedule has three updates; the
current speed/quality recommendation uses the measured one-update
`tail-0p42` schedule. Generate one 1344x768 master and derive the matched
864x480 delivery from it rather than running a second model trajectory.
The production default keeps the official dev base INT8 and applies the
official BF16 refiner LoRA as a bypass low-rank forward branch. Each affected
linear computes the original INT8 base output and adds the BF16 LoRA correction;
the merged weight is never written back through the INT8 quantizer. Runtime
verification fails closed unless the pinned official base, LoRA, revision, SHA,
all 1,660 bypass mappings, and strength 0.8 contract match. The audit also
requires all 1,344 TensorWise INT8 linears to remain on the quantized fast path.
The derived merged-INT8 and selective-BF16 checkpoints remain explicit
low-memory or numerical-diagnostic alternatives.

The distinction between the old and current LoRA paths matters. Merging the
BF16 LoRA into the dequantized base and then requantizing the whole matrix kept
the base close but measured 97.29% relative error against the small LoRA delta.
The current bypass branch instead evaluates the INT8 base and BF16 LoRA
correction separately and adds their outputs, so the correction is never
requantized. The remaining INT8 error is in the base model, not a second
quantization of the LoRA update.

```sh
./h3 -d models/MiniMax-H3-LightX2V-v1.1-768p-Turbo \
  -p "A single fox walks through a snowy pine forest." \
  --super --super-profile v2 --super-stage2-schedule tail-0p42 \
  --super-sol-attn --super-decoder taehv \
  --super-taeh3 ./models \
  -o outputs/fox-super-480p.mp4
```

Use `--super-keep-stage1` to retain the direct F32/PCM `.h3sh` artifact.  A
matched Stage-2-only replay can then use `--super-refine-handoff PATH`; unlike
the older `--super-refine-input PATH` diagnostic, this keeps the direct mmap
handoff contract instead of converting to the MP4 loader.  Native-H3 prefetch
is disabled for replay, while the profile, prompt, seed, first-frame guide,
refiner, Sol policy, and decoder remain configurable.  This is intended for
decoder and Stage-2 A/B work:

```sh
./h3 -d models/MiniMax-H3-LightX2V-v1.1-768p-Turbo \
  --super-refine-handoff /path/to/h3-super-run.h3sh \
  --super-profile v2 --super-stage2-schedule tail-0p42 \
  --super-decoder taehv --super-sol-attn \
  --first-frame /path/to/opening.jpg -p "..." --seed 42 \
  -o outputs/replayed-super.mp4
```

For a strict attention A/B, add `--super-dense-attn` to force ComfyUI's
original dense LTX self-attention. This explicit override is also honored after
`--super-person-quality-preset`, independent of command-line option order.
`--super-sol-attn` and `--super-dense-attn` are mutually exclusive.

Add `--super-output-quality-gate` to a person handoff replay when only the
refined MP4 needs delivery admission. Unlike the native
`--super-person-quality-gate`, this diagnostic does not have Stage-1 metrics;
it checks only the final MP4. A close or three-quarter opening with fewer than
three complete limb chains is treated as an intentional partial-body
composition: the face gate remains active while the full-body gate is skipped.
Add `--super-min-face-area 0.006` to reject outputs whose sampled face occupies
less than 0.6% of the frame.

`--super-bf16-handoff` enables the experimental Sana-style boundary: native H3
center-crops and bilinearly resizes the retained frames to the LTX input canvas,
rounds them to BF16, and writes an `h3-super-handoff-v2` artifact.  The custom
node consumes v2 directly while retaining backward-compatible v1 replay.  On
the measured 576x320 quality workload this reduced the handoff from 268,927,400
to 84,558,248 bytes and cut matched Stage-2 input preparation by about 1.16
seconds, but two replay averages improved complete Stage 2 by only 0.37 seconds
and sustained full E2E runs did not beat the existing v10 Pareto.  It therefore
remains opt-in for low-memory experiments; direct F32 v1 remains the default.

Use `--super-profile 480p-quality` when motion/detail retention matters more
than the lowest latency. It raises only the H3 Stage-1 canvas from 512x288 to
576x320; the LTX input, learned-x2 canvas, TAEHV decoder, and final 864x480
delivery contract stay unchanged. On the matched 64 GiB workload, staged
quality completed in 263.41 seconds with zero swap, versus 329.50 seconds for
direct Turbo and 237.00 seconds for staged balanced. Its measured temporal
information increased from 1.696 to 2.696 relative to balanced, while direct
Turbo remained higher at 6.458. It is therefore the current quality-first
recommendation, not a claim of frame-by-frame Turbo motion parity.

Use `--super-profile 480p-motion` only when retaining more of the Stage-1
trajectory is worth additional latency. It keeps H3 at 576x320 and also raises
the LTX input from 448x256 to 512x288, so learned x2 runs at 1024x576 before the
same 864x480 delivery resize. On the matched staged/no-prefetch run it completed
in 288.03 seconds with zero swap, versus 263.41 seconds for `480p-quality` and
329.50 seconds for direct Turbo. Temporal information increased only from 2.696
to 2.836 relative to quality, while Turbo remained at 6.458. The larger LTX
canvas adds 28.57% more spatial tokens and made three-step LTX denoise about
22.28 seconds slower; this is an opt-in motion/trajectory diagnostic, not the
quality-first default.

Create the derived checkpoint outside the repository and keep both files in
the ComfyUI diffusion-model directory:

```sh
python3 tools/merge_ltx_refiner.py \
  /path/to/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors \
  /path/to/ltx-2.5-22b-distilled-lora-450-bf16.safetensors \
  /path/to/ltx-2.5-22b-dev-refiner-lora-0.8-comfy-int8-convrot.safetensors
```

The merge dequantizes only the patched tensor, applies the LoRA in the same
FP16/FP32 arithmetic contract as the validated Comfy path, and requantizes with
the same ConvRot schema. This avoids retaining the 8.9 GB LoRA and patch
intermediates during every request.

The final requantization is the largest remaining precision compromise.  An
offline audit of all 1,660 LoRA targets found only 1.16% relative error against
the complete merged weight, but 97.29% error relative to the much smaller LoRA
delta for targets stored back to INT8.  In other words, the base model remains
close while a substantial part of the refiner correction can be rotated or
rescaled by the INT8 grid. This is why the fully merged INT8 checkpoint remains
an explicit low-memory option rather than the production default.

For an opt-in quality mode, build the selective-BF16 checkpoint.  It raw-copies
the verified merged checkpoint except for the 48 video self-attention blocks'
Q/K/V/output projections.  Those 192 formerly-INT8 matrices are reconstructed
as `BF16(dequant(INT8 base) + FP16(0.8 * FP32(B @ A)))`; the other 1,152 INT8
and 316 original-BF16 LoRA targets remain unchanged.  The builder streams one
matrix at a time, so it does not load a complete BF16 LTX Transformer:

```sh
python3 tools/build_ltx_refiner_hybrid.py \
  /path/to/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors \
  /path/to/ltx-2.5-22b-distilled-lora-450-bf16.safetensors \
  /path/to/ltx-2.5-22b-dev-refiner-lora-0.8-comfy-int8-convrot.safetensors \
  /path/to/ltx-2.5-22b-dev-refiner-lora-0.8-video-self-bf16-int8-convrot.safetensors
```

The output must retain that filename and its generated `.manifest.json` in the
ComfyUI diffusion-model directory.  The measured artifact is 24,722,047,456
bytes (about 23.02 GiB), 3.00 GiB larger than merged INT8.  Across all video
self-attention LoRA targets it improves LoRA-effect cosine from 0.73694 to
0.99232 and reduces LoRA-effect relative L2 error from 0.91713 to 0.12375.
Verify another build or selector with:

```sh
python3 tools/analyze_ltx_refiner_quant.py \
  /path/to/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors \
  /path/to/ltx-2.5-22b-distilled-lora-450-bf16.safetensors \
  /path/to/ltx-2.5-22b-dev-refiner-lora-0.8-video-self-bf16-int8-convrot.safetensors \
  --category video_self_attention --output outputs/lora-hybrid-audit.json
```

Select the hybrid checkpoint explicitly at runtime.  This 480p quality recipe
keeps the complete official three-update refiner schedule, Metal Sol-Attn,
TAEHV, BF16 handoff, and the measured step-2 conditioning overlap:

```sh
./h3 -d models/MiniMax-H3-LightX2V-v1.1-768p-Turbo \
  -p "A detailed five-second audiovisual scene." \
  --first-frame /path/to/opening.png --seconds 5 --seed 50803 \
  --super --super-profile 480p-quality \
  --super-transformer \
    ltx-2.5-22b-dev-refiner-lora-0.8-video-self-bf16-int8-convrot.safetensors \
  --super-stage2-schedule default --super-sol-attn --super-decoder taehv \
  --super-bf16-handoff --super-prefetch-conditioning \
  --super-prefetch-start-step 2 --super-allow-swap --super-taeh3 ./models \
  -o outputs/hybrid-super-480p.mp4
```

On a 48 GiB machine, start with that command and leave input-model and whole-
Transformer prefetch disabled.  On the measured 64 GiB host,
`--super-prefetch-input-models` reduced input preparation but raised sampled
used memory to 59.76 GB and produced about 8.62 GiB of swap during the complete
five-second run.  Hybrid itself raised a matched Stage-2 peak from 40.60 GB to
49.08 GB; whole-Transformer prefetch is still not recommended.  The hybrid is
a numerical-fidelity option rather than a guaranteed sharpness filter: matched
videos were visually very close, and its benefit may appear more in motion,
identity, or fine refiner corrections than in a simple edge metric.

For person-centric delivery, the current quality-first entry point is
`--super-person-quality-preset`. It locks the measured `person-strict-v1`
contract instead of relying on a long list of independently mutable flags:

- LightX2V v1.1 768p Stage 1 at 896x512, four H3 passes, and its manifest-
  validated strength/flow-shift contract;
- the official INT8 LTX base plus the official BF16 refiner LoRA at strength
  0.8, applied as a runtime low-rank bypass rather than requantized into INT8;
- the complete three-update LTX schedule, Metal Sol-Attn, prepared BF16 direct
  handoff, sequential BF16 TAEHV, and post-decode model eviction;
- stride-1 Apple Vision admission of both Stage 1 and the final MP4, with up to
  three consecutive seeds; and
- explicit swap acceptance, but no cross-stage prefetch that deliberately
  overlaps the resident H3 and LTX model families.

The preset requires a pinned opening image and the LightX2V v1.1 model root.
The following is the recommended five-second person-quality command:

```sh
PROMPT="$(cat outputs/benchmark/super-person-strict-v1-20260830/prompt.txt)"

./h3 -d models/MiniMax-H3-LightX2V-v1.1-768p-Turbo \
  -p "$PROMPT" \
  --first-frame \
    outputs/benchmark/super-lightx-v11-20260830/dancer-opening-three-quarter-1152x648.png \
  --seconds 5 --seed 60020 \
  --super-person-quality-preset --super-taeh3 ./models \
  -o outputs/dancer-person-strict-1344x768.mp4
```

`person-strict-v1` always generates one 1344x768 master. Derive the matched
480p delivery from that master instead of running a second lower-resolution
trajectory. The measured clarity candidate uses a deliberately mild luma
unsharp after Lanczos scaling:

```sh
ffmpeg -i outputs/dancer-person-strict-1344x768.mp4 \
  -vf "scale=864:-2:flags=lanczos,crop=864:480,unsharp=5:5:0.6:5:5:0" \
  -c:v libx264 -preset slow -crf 12 -pix_fmt yuv420p \
  -c:a copy -movflags +faststart \
  outputs/dancer-person-strict-864x480.mp4
```

The accepted seed-60020 cold-cache run completed in 913.33 seconds: 382.72
seconds in H3 Stage 1 and 528.75 seconds in LTX Stage 2. Sampled system use
peaked at 51.97 GB decimal (48.40 GiB), with no new swapout. Its final gate
detected the face in all 121 frames; maximum normalized center jump was
0.00297, maximum absolute log-area jump was 0.03813, and maximum face-contour
distance was 0.422. Historical warm-cache runs of the same complete
three-update Stage 2 were about 268--280 seconds, so a normal warm first-seed
admission remains approximately 10.9 minutes. Candidate retries are a quality
budget, not a latency guarantee.

The local machine has the BF16 refiner LoRA and BF16 Video VAE, but not the
42,018,190,584-byte BF16 LTX Transformer. Only about 30 GiB of disk was free
when this mode was finalized. More importantly, matched BF16-base output was
already very close to INT8 base plus BF16 bypass LoRA (SSIM 0.985789, PSNR
40.361 dB), while two BF16-base runs exited during the following 1344x768
decode lifetime. Downloading the full BF16 base is therefore not recommended
for a 48--60 GiB target until Stage 2 has an isolated worker lifetime or true
BF16 block streaming.

The latency-oriented precision path uses the same official INT8 dev base with
the official BF16 refiner LoRA in Comfy's bypass mode. The faster recipe below
pairs that precision path with `tail-0p42`: sigma `0.421875 -> 0.0`, one LTX update,
learned x2, the original high-resolution first-frame guide, Metal Sol-Attn, and
TAEHV all remain active. `default` retains the strict three-update LTX schedule
when closer algorithmic parity with Sana is more important than latency. The
filenames are shown explicitly below so a production command records the exact
model contract. The current
Stage-1 quality recommendation is the LightX2V v1.1 768p merge above; it is not
bitwise Sana-v2 Stage 1. For people, two full quality candidates are the normal
workstation budget; raise this to three only when fail-closed delivery is worth
the extra worst-case latency. Consecutive seeds are first
evaluated by the Stage-1 person gate. A Stage-1 rejection reuses the loaded H3
and never loads LTX. A Stage-1 admission enters Stage 2; if the refined MP4 is
then rejected, ComfyUI is already released, H3 is reloaded, and the next seed
uses the remaining candidate budget.

```sh
./h3 -d models/MiniMax-H3-LightX2V-v1.1-768p-Turbo \
  -p "A controlled five-second audiovisual scene with stable anatomy." \
  --first-frame /path/to/opening.png --seconds 5 --seed 50803 \
  --super --super-profile v2 \
  --super-transformer \
    ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors \
  --super-refiner-lora \
    ltx-2.5-22b-distilled-lora-450-bf16.safetensors \
  --super-refiner-lora-strength 0.8 \
  --super-stage2-schedule tail-0p42 --super-sol-attn --super-decoder taehv \
  --super-taehv-mode sequential --super-taehv-evict \
  --super-bf16-handoff --super-person-quality-gate \
  --super-min-face-area 0.006 \
  --super-quality-attempts 2 \
  --super-allow-swap --super-taeh3 ./models \
  -o outputs/quality-super-1344x768.mp4
```

Super delivery now forces H.264 re-encoding at CRF 18 while the decoded frames
are still present in the Comfy graph.  The earlier automatic codec selection
produced only about 1.13 Mb/s for a 1344x768 five-second master and removed
additional high-frequency detail from the already-soft TAEHV result.  A matched
face-detail replay produced 2.22 Mb/s at CRF 18; SI improved from 28.119 to
28.223 and the FFmpeg blur metric improved from 10.688 to 10.521, with only a
1.9-second Stage-2 difference inside normal MPS run variance.  This is a small
but effectively free preservation improvement, not a substitute for the much
slower official Video VAE.

Conditioning, refiner, and latent state files are also retained until SaveVideo,
the output copy, and the final person gate have all succeeded.  Native h3.c then
removes them.  An encoder configuration error can therefore be recovered from
the completed latent instead of discarding a roughly 220-second denoise result.
The command above needs no additional option for either behavior.

On 2026-08-30, the accepted dancer handoff replay with `tail-0p42` completed
Stage 2 in 141.115 seconds: 17.354 seconds conditioning, 24.613 seconds input
preparation, 93.304 seconds denoise, 5.055 seconds decode/save, and 0.682 seconds
for the final Vision gate. Sampled used memory peaked at 52,744,503,296 bytes
(49.12 GiB), minimum sampled free memory was 14.88 GiB, and swapout delta was
zero. Adding the measured approximately 383-second accepted H3 candidate gives
a normal first-admission E2E estimate of about 524 seconds / 8.7 minutes. The
same matched three-update replay took about 270 seconds in Stage 2, so the
one-update schedule reduced Stage-2 time by about 47.7% and estimated E2E by
about 19.7%. It is about 1.59x slower than the historical 329.86-second Turbo
run, but about 3.14x faster than historical Base20 at 1,644.01 seconds.

For comparison, a matched strict/default three-update Stage-2-only replay with true bypass LoRA took
267.89 seconds: 18.20 seconds conditioning, 23.59 seconds input preparation,
221.28 seconds denoise, and 4.04 seconds decode. Sampled system use peaked at
49,682,726,912 bytes (46.27 GiB) with zero new swapout. A complete three-candidate
quality run took 1,695.60 seconds because one candidate failed before LTX and a
second narrowly failed the refined-output body gate; its accepted candidate used
1,148.06 seconds of cumulative H3 and 273.12 seconds for the final LTX pass.
Therefore the strict/default first-admission expectation remains roughly one H3
pass plus one 268--280 second Stage 2; the recommended `tail-0p42` expectation is
the approximately 524-second E2E estimate above. `--super-quality-attempts N` is
a safety budget, not a latency promise. Generate the 1344x768 master
once and downscale/crop it to 864x480 for the sharpest matched 480p delivery;
running a second smaller profile produces a different trajectory and usually
loses Stage-1 detail.

Older v14 metadata called this path `runtime_lora_int8_base`, but Comfy's full-load
quantized `set_weight()` actually requantized the merged weight. Telemetry v15
uses `runtime_bypass_lora_int8_base` only after all official adapters have been
installed as bypass hooks. It also
changes `timing.total_seconds` to real CLI E2E from Super validation through
the final gate and records `stage1.model_load.attempts/seconds` separately.
This makes the initial load and every H3 reload after a post-refine rejection
use the same accounting. Compare older v13 totals only after adding their
unreported initial model-load time.

Telemetry v16 added sampled Stage-1 and refined-output face-area, capture-quality,
and landmark-coverage metrics. The general person gate samples every third retained
frame; the legacy face-only compatibility gate remains every sixth frame. For
a requested minimum face area at or above `0.006`, both stages require face
detection at least `0.90`, landmark coverage at least `0.90`, and minimum Apple
Vision capture quality at least `0.12`. The close-composition contour ceilings
are `0.70` for Stage 1 and `0.80` for the final MP4; the larger final envelope
accepts the measured clean one-update result without admitting low-capture or
missing-face motion. General person compositions require capture quality at
least `0.10`. In the accepted final master, all 41 sampled frames had faces,
capture quality was `0.254/0.171` mean/minimum, and final contour maximum was
`0.755`.

Telemetry v17 adds `quality_preset`, LightX2V v1.1 provenance, face-center and
log-area temporal jumps, and the strict `person-strict-v1` gate. Strict mode
checks every retained frame (`stride=1`): face detection must be at least 0.98,
capture quality at least 0.14, center jump at most 0.02, absolute log-area jump
at most 0.10, and contour distance at most 0.70 before LTX / 0.80 after LTX.
The general, tail-oriented gate retains its earlier stride-3 behavior.

```sh
ffmpeg -i outputs/quality-super-1344x768.mp4 \
  -vf "scale=864:-2:flags=lanczos,crop=864:480" \
  -c:v libx264 -preset slow -crf 12 -pix_fmt yuv420p \
  -c:a copy -movflags +faststart outputs/quality-super-864x480-hq.mp4
```

The official 42,018,190,584-byte BF16 dev Transformer is also accepted with
the same verified BF16 LoRA.  On this 64 GiB machine its denoise completed, but
the single ComfyUI process twice exited when the cached BF16 model/patch state
was followed by 1344x768 decode.  Recovering the saved latent in a clean process
decoded with sequential TAEHV in 2.05 seconds, showing that the problem is the
whole-model lifetime peak rather than TAEHV.  Matched output was nearly
identical to the INT8-base runtime-LoRA arm (SSIM 0.985789, PSNR 40.36 dB) and
did not repair fast-spin face deformation already present in the H3 handoff.
Do not use the full BF16 base as the local production default until there is a
separate Stage-2 worker lifetime or real BF16 block streaming.

For people, prompt choreography is currently more important than increasing
LTX precision.  Keep the face front or three-quarter, use controlled side
steps and arm arcs, avoid a complete fast spin inside five seconds, and avoid
hands crossing the face.  In the measured dancer run this changed final SI/TI
from 26.90/11.42 to 28.40/9.37 and visibly reduced face/body deformation before
LTX refinement.

`--super-person-quality-gate` is the recommended fail-fast guard for a native
person Stage 1.  It samples every third retained RGB frame with Apple Vision
after H3 completes but before the handoff is written or LTX is loaded.  The
face side combines detection, duplicate-face rejection, a coarse opening-face
feature print, and eye-aligned normalized landmark geometry for the eyes, nose,
mouth, and face contour.  The body side verifies all 14 core/limb joints, four
complete shoulder-elbow-wrist / hip-knee-ankle chains, core coverage, duplicate
bodies, and bounded left/right limb-length asymmetry. A close or partial-body
opening with joint coverage below 0.85 or fewer than three complete limb chains
keeps the face gate but skips the body gate; a non-person
opening skips both.  `--super-face-quality-gate` remains available as the
legacy face-only form.

`--super-quality-attempts N` makes the complete admission automatic for
`N=1..16`. Attempt one uses the requested seed and later attempts use
consecutive seeds. H3 stays loaded across Stage-1 rejections. Once a candidate
enters LTX, H3 is released as usual; a post-refine rejection frees ComfyUI and
reloads H3 before generating the next seed. Cross-stage Transformer,
conditioning, and input-model prefetch are rejected for a multi-attempt run so
H3 and LTX are never deliberately kept resident together. Intermediate
rejected handoffs are removed unless `--super-keep-stage1` is set; the final
failed handoff is retained for diagnosis. Telemetry v16 separately records
Stage-1 attempts/rejections and refined attempts/output rejections, their
cumulative times, the requested/final seed, and the last rejection reasons.
This improves delivery reliability but does not make rejected attempts free:
on the measured v2 workload a Stage-1 rejection costs about 382--383 seconds, while
a post-refine rejection pays roughly 383 seconds of H3 plus 273 seconds of LTX
before the next H3 reload.

The production person gate also rechecks the copied refined MP4. It samples
every third output frame and requires at least 0.90 landmark-frame coverage,
minimum capture quality `0.10` (`0.12` for `--super-min-face-area 0.006`), no
duplicate faces, and the same full-body coverage envelope used before LTX.
General compositions require face detection at least `0.85` and
landmark/mouth/contour maxima at most `0.60/0.60/0.70`; the close-up profile
raises detection to `0.90` and only the contour ceiling to `0.80`. This catches a Stage-1
candidate whose face geometry is amplified or softened by refinement. A failed
final gate deletes the delivery path and never silently publishes the rejected
video. If the full-candidate budget remains, the intermediate handoff is
removed by default and the next seed restarts at H3; otherwise the final
diagnostic handoff is retained, the CLI fails, and telemetry status is
`output_rejected`. The stride-3 Vision pass analyzes 41 frames and is small compared
with either model stage. Stage-2-only replay can request the same final check
with `--super-output-quality-gate` but cannot synthesize another H3 candidate.

The production thresholds are calibrated from native and decoded positive and
negative dancer samples. General Stage-1 face detection must be at least 80%;
the `min_face_area >= 0.006` close-up profile requires 90%. The legacy
face-only Stage-1 feature-print mean/max remain 0.65/0.95; the production
person Stage-1 mean is 0.75 because repeated front-safe candidates measured
0.676--0.690 while their landmark/body geometry was clean. The final refined
MP4 keeps the stricter 0.65/0.95 identity guard. The legacy face-only gate
keeps normalized landmark/mouth/contour maxima at 0.80/0.75/0.90. The
production person gate tightens them to 0.65/0.60/0.55 for general composition;
the close-up Stage-1 contour ceiling is 0.70. Every detected person-gate face
must also expose capture quality, with minimum 0.10 generally or 0.12 for the
close-up profile, and at least 90% of detected frames must have landmarks. When
a full body is present, detection must be at least 95%, mean/min joint coverage
0.95/0.85, core minimum 0.80, complete-chain mean/min 3.5/2, duplicate bodies
zero, and limb asymmetry at most 2.25.  Feature print is intentionally not used
alone: repeated native runs showed a known-good seed moving from 0.531 to 0.557,
while some visibly bad faces were around 0.55.  Landmark geometry and body pose
provide the missing structural signal.

The accepted front-safe five-second v2 run passed 21/21 faces and bodies.  Its
native landmark/mouth/contour maxima were 0.336/0.407/0.391; every sampled frame
contained all 14 joints and all four limb chains.  It completed in 702.03
seconds: 383.43 seconds in Stage 1 and 318.32 seconds in Stage 2.  Peak sampled
used memory was about 55.12 GiB and system-wide swapout increased by about
49.69 GiB.  The official three-update LTX path issued 141 Metal Sol-Attn calls
with no fallback; sequential BF16 TAEHV decode itself took 1.64 seconds.

A rejected native candidate had landmark/mouth/contour maxima
0.789/0.848/1.064 and visible frame-72/96 facial smearing.  It stopped before
LTX and, with `--super-keep-stage1`, retained its `.h3sh` for diagnosis.  This
is the gate's main latency value: a rejection pays only the roughly 383-second
H3 Stage 1 and avoids the measured 318-second LTX Stage 2.  The gate is an
admission check, not a restoration filter; it materially reduces the chance of
shipping a deformed person but cannot mathematically guarantee fingers or every
possible anatomy.  It requires native Stage 1 and cannot be combined with
`--super-refine-input` or `--super-refine-handoff` replay.

Inspect a decoded video with the same face/body metrics, or convert a retained
handoff to a diagnostic MP4:

```sh
make h3_face_stability
./h3_face_stability stage1.mp4 --stride 3
python3 tools/h3sh_to_mp4.py rejected.h3sh rejected-stage1.mp4
```

Super keeps the Sana-compatible four-pass LightX2V Stage-1 schedule by default.
`--super-stage1-steps 6` and `--super-stage1-steps 8` are explicit diagnostics,
not production quality presets.  On the matched difficult full-spin dancer,
six passes raised Stage-1 SI/TI only from 46.278/10.405 to 46.746/10.926 while
changing the motion trajectory: the subject was fully back-facing at frame 72
and the frame-24/48 face and arm blur remained.  Six-pass H3 denoise took
526.91 seconds and the H3 DiT phase took 540.78 seconds; its profiled Stage-1
phases totaled about 556.57 seconds, versus 380.63 seconds for the accepted
four-pass Super Stage 1.  Because it was slower without a consistent anatomy
or identity improvement, neither six nor eight passes is recommended for the
current LightX2V checkpoint.

An official LightX2V v1.0 544p eight-step adapter was also tested at its
manifest-recommended NFE 8, strength 0.0625, and video/audio shifts 12/3.  On
the controlled seed-50803 dancer it raised stride-1 face detection from 81.82%
to 95.16% and reduced the maximum face-center jump from 0.1011 to 0.0304, but
the maximum landmark/mouth/contour distances regressed to
0.809/0.977/1.224 and limb asymmetry reached 3.195.  Its H3 Euler denoise took
705.09 seconds and H3 DiT total took 718.86 seconds, versus about 380.63 seconds
for the matched Sana-v0.1 four-step Stage 1.  A subsequent official INT8-base plus BF16
bypass-LoRA three-update LTX replay took 271.81 seconds and was rejected by the
refined-output gate (capture minimum 0.094, contour maximum 0.846).  The
eight-step artifact is therefore supported and provenance-audited, but it is
not the current person-quality default.  Detailed evidence is in
`../notes/super-quality-root-cause-20260830.md`.

A strict matched original-BF16 Base20 control also did not make that full-spin
prompt production-safe.  Its H3 DiT phase took 1770.22 seconds (about 4.65x the
accepted four-pass Stage-1 wall time), but frames 24/48 still showed face and
fast-arm deformation and frame 72 was back-facing.  Base20 and LightX2V 4-step
diverged strongly (SSIM 0.6933 / PSNR 19.35 dB) because they selected different
motion trajectories; that is not evidence that either trajectory has better
anatomy.  Do not automatically fall back to Base20 for a difficult five-second
spin.  Admit identity-sensitive prompts by choreography first: reduce angular
velocity, keep the face front or three-quarter, avoid hair/hands crossing the
face, or split a required full turn across a longer clip or shot boundary.

When `--first-frame PATH` is present, that exact source file conditions both
H3 FL2VA and the LTX high-resolution frame-0 guide. Native code atomically
imports it into the ComfyUI input root and verifies source/copy SHA-256; the
custom node verifies the digest again before decode. Any identity mismatch
fails closed. Without `--first-frame`, Stage 2 falls back to the decoded H3
opening frame for text-only compatibility.

Telemetry v15 additionally matches the official LTX-2.5 image-conditioning
distribution.  The verified opening asset is round-tripped in memory through
libx264 CRF 18 (`veryfast`, YUV420P), then `LTXVAddGuide` performs its native
bilinear aspect-fill/center-crop before the Video VAE encode.  CRF 18 is resolved
from the local Video VAE's `model_version=2.5.0`; it is not the final MP4's
encoding CRF.  The custom node reports the round-trip PSNR and fails instead of
silently using an uncompressed or differently compressed guide.

Stage 2 now uses Comfy's positive-only `BasicGuider`, matching Sana's
`SimpleDenoiser`, rather than constructing an LTX dual-CFG guider at video/audio
scale 1.  A fixed-handoff A/B with the legacy raw guide was pixel-identical to
the previous dual-CFG output (121-frame SSIM 1.0, PSNR infinite), so this change
does not alter the numerical result.  CRF18 versus the legacy raw guide measured
SSIM 0.982944 / PSNR 39.133 dB at 1344x768 and produced only a slight texture
softening in the inspected dancer frames.  It is retained because it matches the
model's training and Sol-Engine contract, not because it is a sharpening filter.

On the measured 64 GiB host, the current true-bypass CRF18 Stage-2 replay took
267.89 seconds with zero new swapout. Its matched old-requantized output differed
by SSIM 0.973125 / PSNR 35.596 dB, so bypass is a real numerical correction, not
only a telemetry rename. Do not attribute the entire timing change to bypass:
hash caches, the later sequential TAEHV lifecycle, system load, and unified-memory
state also changed. The recommended v2 command keeps the fail-fast person gate,
official three-update schedule, Metal Sol-Attn, and TAEHV decoder unchanged.

The default endpoint is `http://127.0.0.1:8188`; `H3_SUPER_INPUT_DIR` and
`H3_SUPER_OUTPUT_DIR` select the shared ComfyUI directories. The default staged
mode unloads each large component between phases. Whole-Transformer prefetch is
available only as `--super-prefetch-transformer`: on the measured 64 GiB host it
was slower and caused substantial swap, so it is not the production default.
Starting it only after final-pass block 15 while progressively releasing all 50
H3 blocks improved the old eager-prefetch result, but still took 262.05 seconds
versus 250.30 seconds for conditioning step-2 and produced about 7.02 GiB of
swapout.  The hidden 22.32-second Transformer load saved only about one second
from LTX denoise while forfeiting the more valuable conditioning overlap.
`--super-prefetch-conditioning` is the narrower aggressive option: it overlaps
Gemma encoding with H3 denoise, retains only the compact conditioning artifact,
and releases Gemma before Stage 2 input. On the matched 64 GiB run it reduced
wall time from 237.00 to 228.40 seconds, but produced about 8.63 GiB of system
swapout. Keep staged mode for 48 GiB or no-swap operation. Use conditioning
prefetch only when this measured memory trade is acceptable.
`--super-allow-swap` remains as an explicit telemetry acknowledgement, but is
not required for a successful run: swapout is warning-only and telemetry records
both the overlap/wait split and swap delta. Release admission is based on stable
E2E and output quality, not a zero-swap hard gate.

`--super-prefetch-input-models` is a narrower 64 GiB speed-first extension to
conditioning prefetch.  It does not submit a separate prefetch prompt.  Instead,
the conditioning output node saves the compact Gemma artifact, then fully loads
the Video VAE, Audio VAE, and latent upscaler in the same prompt tail and retains
those three patchers until the immediately following input phase.  This avoids
about 3 seconds of loader/API work without keeping the 20 GiB Transformer alive.
On the matched quality workload, two warm runs completed in 248.68 and 247.35
seconds versus a 250.01-second warm control; input preparation fell from 11.11
to 9.07/8.09 seconds.  Swapout increased from 4.84 GiB to 8.32/8.40 GiB.  Keep
this flag explicit rather than automatic on 48 GiB hosts; omit it when sustained
swap makes E2E slower.

`--super-fuse-denoise-decode` is an experimental TAEHV-only Stage-2 topology.
It connects the final cropped LTX latent directly to TAEHV in one Comfy prompt,
then unloads the Transformer in-graph before decoder residency.  This removes
the latent `torch.save/load`, one queue/history round trip, and the intermediate
`/free` boundary without changing the model, sigmas, Sol schedule, or output.
Two strict matched replays were pixel-identical to staged execution (SSIM 1.0,
PSNR infinite), but averaged 139.67 seconds versus 139.02 seconds for staged and
showed higher variance/peak memory.  The flag therefore remains opt-in and is
not part of the recommended 48 or 64 GiB recipes.  A follow-up resident CPU
state and synchronous-release experiment was also rejected and removed because
the states were under 1 MiB while matched E2E regressed by 2.11 seconds.

For `480p-quality`, the historical eager step-0 prefetch reduced 263.41 to
252.78 seconds but produced about 17.11 GiB of swapout. The deadline-controlled
step-2 result below supersedes it, while staged mode remains the no-swap default
on 48 GiB systems.

`--super-prefetch-start-step N` selects the completed H3 denoise step that
launches prefetch (`N=0` through `3`). The default three-update schedule uses the
measured `2/4` conditioning deadline. The opt-in two-update `skip-0p9` schedule
instead uses final-pass `block15/evict15`, which is its measured E2E Pareto point;
whole-Transformer prefetch retains `0/4` for historical diagnostics. On the
matched quality workload, three-update conditioning at `2/4` completed in
249.97 seconds with about 3.35 GiB swapout, versus 252.78 seconds and 17.11 GiB
at `0/4`; `1/4` completed in 250.63 seconds with about 3.71 GiB swapout. Later
launch reduces H3/Gemma contention but may add a post-H3 wait if the artifact
misses its deadline. Telemetry records both configured and actual deadlines.
Do not combine a nonzero start step with no prefetch mode.

For stylized high-motion inputs, the explicit three-update action recipe is
`--super-profile 480p --super-prefetch-conditioning
--super-prefetch-input-models --super-prefetch-start-step 1`.  Two warm runs
completed in 223.55 and 224.15 seconds versus 226.72 seconds without input-model
residency and 329.86 seconds for direct Turbo, while retaining all three official
refiner updates.  Input preparation fell from 11.26 to 8.45/8.07 seconds;
swapout rose from 3.50 GiB to 11.41/11.72 GiB.  Its TI slightly exceeded Turbo,
but it achieved that motion partly through a wider camera pull-back and lower
spatial detail.  It is therefore the fastest measured complete-three-update
action recipe on the 64 GiB host, not the global quality default.

TAEHV wide remains the production 480p decoder.  In a matched replay from one
direct handoff, TAEHV decoded/muxed in 4.03 seconds while the official tiled
Video VAE required 155.40 seconds.  Complete Stage 2 was 126.95 versus 278.19
seconds.  The two decoded videos retained the same structure and measured SSIM
0.9463 / PSNR 33.62 dB; the official VAE was sharper, while TAEHV preserved the
opening-frame reference slightly better.  The official decoder therefore stays
as a correctness diagnostic rather than a latency profile.

## Tests and runtime requirements

```sh
make test
make parity
```

`make test` runs the deterministic host suite and, when the ignored MLX fixture
is installed under `misc/fixtures/`, compiles the Metal source at runtime and
checks a complete toy H3 block against named MLX outputs. Runtime compilation is
intentional: it follows Iris and does not require Xcode's optional offline Metal
toolchain. The test covers both an F32 diagnosis path and the production BF16
storage path; wide BF16 matrix products and SDPA use cached MPSGraph graphs, with
direct Metal correctness fallbacks. `make parity` runs only those Metal/MLX
checks.

FFmpeg and FFprobe must be available on `PATH` for media inputs and MP4 output
(`H3_FFMPEG` and `H3_FFPROBE` may select explicit executables). Generated RGB24 and
32 kHz stereo F32 PCM are fed through concurrent pipes; no intermediate
uncompressed media file is created.

## Implementation and performance notes

The remainder documents the implementation behind the tutorial presets and the
environment variables retained for exact A/B diagnosis.

### Sampler and DiT controls

The default sampler uses the released shifted video/audio schedule. `--steps`
always names the number of denoising passes, with terminal zero added after the
last pass. Whole-denoiser reuse evaluates the first and last pass plus every
requested interval, then extrapolates skipped video and audio velocities on
their independent schedules. With very small step counts, keep `--reuse 1`.

For the low-budget path, the released linear base grid won against
actual-video-sigma linear spacing,
quadratic and cubic warps, exact 30-point tail subsets, mild power warps,
zero-order held full-grid velocities, linear velocity extrapolation, and RES.
The more tail-heavy candidates often sharpened the subject but damaged motion
or left a repetitive woven background; sparse RES and long extrapolation
intervals failed much more visibly.

Layer thinning ranks the checkpoint's actual AdaLN gates while protecting
structurally important first and final blocks. Unused weights and schedule
tensors are not retained, so `--layers 45` and `--layers 40` reduce both
transformer time and unified-memory use. Core reuse holds the previous full
transformer residual while refreshing the patch projection and timestep-aware
head; it remains mutually exclusive with whole-velocity reuse.

Dynamic gate ranking must first materialize all 50 AdaLN schedules in order to
measure their gate magnitudes. `H3_DIT_GATE_SKIP=LIST` is an explicit
checkpoint/schedule-specific fast path: it accepts exactly `50 - --layers`
unique comma-separated block IDs in `[2,48]`, precomputes AdaLN only for the
remaining blocks, and rejects malformed lists or the `uniform` layer policy.
The prepared-model session key includes both the layer policy and skip list.
Do not copy a list between checkpoints, step schedules, or FL2VA/Ref2VA modes
without rerunning dynamic ranking and a quality gate.

For the tested merged Turbo checkpoint, four-step FL2VA schedule, and 38-layer
profile, dynamic ranking selected:

```sh
H3_DIT_GATE_SKIP=4,14,16,17,13,12,11,15,10,8,9,7
```

On the 64-GiB M4 Max 384-square internal canvas this reduced DiT load/AdaLN
time from 10.550 to 9.238 seconds and complete generation from 81.85 to 80.96
seconds, without changing the 38-block denoising topology. Dynamic ranking
remains the safe default; the explicit list is only a pinned local profile.

`H3_DIT_GATE_CACHE=PATH` removes the need to copy that list by hand. On a cache
miss, H3 still computes the full dynamic gate-ranking oracle and atomically
writes a small manifest. A later compatible process validates all transformer
shard identities, the exact video/audio sigma arrays, conditioning modalities,
the ranking-policy version, and the requested active-block count before using
the stored mask to skip inactive AdaLN projections. Any malformed or mismatched
manifest is ignored and replaced only after a successful dynamic ranking.
Explicit `H3_DIT_GATE_SKIP` takes precedence; uniform and 50-block profiles do
not use the cache. A cache file stores one profile, so use distinct paths when
alternating active-block counts:

```sh
H3_DIT_GATE_CACHE=models/turbo-gates-l38-v1.txt ./h3 \
  -d models/MiniMax-H3-Turbo -p "..." \
  --width 512 --height 512 --render-width 320 --render-height 320 \
  --seconds 5 --steps 4 --layers 38 --reuse 1 -o output.mp4
```

The first run is intentionally the full oracle. At 320 square on the tested M4
Max, a 38-block second cold load fell from 11.092 to 9.094 seconds (18.0%), and
a 40-block load fell from 10.720 to 9.423 seconds (12.1%). Same-process cached
and dynamic forwards produced byte-identical video and audio outputs. Canvas and
prompt are intentionally absent from the manifest identity because they do not
participate in AdaLN gate scoring.

`H3_PARALLEL_PREPARE=1` overlaps a fresh prompt-only FL2VA DiT load with Qwen
encoding. The worker uses `h3_dit_load_t2va_core()` to build only the schedule,
backend, resident/quantized weights, Metal runtime, and model core; it does not
run dummy text refinement or allocate dummy RoPE, maps, or activation arenas.
After joining, the main thread calls the same resident-request rebuild used by
interactive sessions with the exact Qwen embedding and layout, allocating the
real request state once. It is deliberately limited to conditioning-cache
misses with no first/last frame, no Ref2VA references, no compatible resident
DiT, no SSD streaming, and at least 48 GiB of physical memory. The worker never
invokes application progress callbacks. Thread start, load, or request-rebuild
failures discard the speculative object and fall back to the original
sequential cold load. A core-only object is fail-closed and cannot forward or
denoise before a successful request rebuild.

On the tested 64-GiB M4 Max, the original full-dummy parallel path measured
59.27/59.31 seconds. The core-only path measured 58.69/59.53 seconds (59.11
second mean), while reducing cumulative DiT allocation by about 1.97 GiB and
removing two submissions, 15 direct dispatches, nine linear dispatches, and two
attention dispatches from prepare. Final peak footprint remains about 32.19 GB
because the real request state is still required, and swaps remain zero. The
optimization does not change block/step math: same-process core-only and cold
forwards were byte-identical after the resident rebuild. It remains opt-in
because it trades memory and temporary Qwen/DiT contention for cold-start
latency; resident-session hits already avoid the load and do not use it.

`H3_DIT_STEP_GATE_SKIP='2:5,6;3:5,6'` is a one-based research/preview control
that skips selected resident blocks only on selected denoising steps. It is
fail-closed with SSD streaming, token reduction, adaptive cache, Sol-Attn, and
core reuse, protects blocks 0, 1, and 49, and retains at least 25 resident
blocks. On the tested 320-square four-step/38-block preview it reduced denoise
by about 1.07 seconds in a same-process AB and complete generation to 58.51
seconds, but video changed materially and decoded audio drift was large. It is
therefore not a balanced/default optimization. `H3_STEP_GATE_RUNTIME_DISABLE=1`
provides a same-loaded-model diagnostic baseline.

`H3_FBC_THRESHOLD=VALUE` enables the experimental native FirstBlockCache for
20/50-step quality schedules. Every step still evaluates the first active DiT
block. A Metal relative-L1 reduction compares that block's residual with the
last fresh residual; a hit skips the remaining active blocks and adds their
cached BF16 tail residual. Only the small F32 reduction result is read by the
CPU. The final denoising step is always fresh, and the cache is reset for every
request. `0.06` is the current conservative threshold and `0.08` the faster
balanced candidate; neither is enabled by default until a broader prompt/seed
quality suite has passed.

`H3_FBC_MAX_HITS` limits consecutive cache hits and defaults to `1`; `0`
removes the cap. A cap-forced fresh step stays in the existing command stream
and avoids the otherwise unnecessary decision synchronization. FirstBlockCache
is intentionally incompatible with core reuse, whole-denoiser reuse, token
reduction, and SSD streaming while their combined state and quality remain
unvalidated. It is not recommended for the four-step Turbo model: the measured
Turbo probe produced no hits at threshold `0.08`, so it only added controller
overhead. `H3_FBC_FORCE_FRESH=1` exercises the cache data path without reuse;
`H3_FBC_RUNTIME_DISABLE=1` supplies a same-loaded-model exact A/B arm.

`H3_TEACACHE_THRESHOLD=VALUE` enables the native TeaCache path for Base
20/50-step quality schedules. It evaluates the first active block's attention
AdaLN, compares that timestep-modulated BF16 input with the preceding step via
a Metal relative-L1 reduction, and accumulates the score. A cache hit skips all
active transformer blocks and adds the last fresh full-core residual. The
final step is always fresh and all trajectory state is reset per request.

The validated visual-balanced M4 setting is threshold `0.06`, retain `5`,
cooldown `1`, and at most one consecutive hit. For the recommended
media-balanced preset, add the independently calibrated target-audio gate:

```sh
H3_TEACACHE_THRESHOLD=0.06 \
H3_TEACACHE_AUDIO_THRESHOLD=0.05 \
H3_TEACACHE_RETAIN_STEPS=5 \
H3_TEACACHE_COOLDOWN_STEPS=1 \
H3_TEACACHE_MAX_HITS=1 \
./h3 -d /path/to/MiniMax-H3 -p "..." \
  --width 512 --height 512 --seconds 5 \
  --steps 20 --layers 50 --reuse 1 -o output.mp4
```

On the tested 64-GiB M4 Max, the matched 512x512x124 M4-int8 run reduced
denoise from 820.348 to 575.953 seconds and end-to-end time from 878.55 to
633.86 seconds: 14 fresh / 6 reuse, 1.424x denoise and 1.386x end-to-end.
Full-video SSIM was 0.9078 against the matched oracle and the six-frame visual
gate retained one coherent fox and continuous motion. This remains opt-in
pending a broader prompt/seed suite; audio is approximate rather than
sample-identical. Threshold `0.08` is only a research-fast tier: both one-hit
and two-hit 22-frame runs showed substantially more pose/timing/audio drift,
so it is not a balanced recommendation.

For dialogue, music, or tight event synchronization,
`H3_TEACACHE_AUDIO_THRESHOLD=VALUE` adds a second exact relative-L1 gate over
the packed target-audio rows. The global and audio accumulators are evaluated
independently; either reaching its threshold forces a fresh step. The audio
reduction is encoded into the same command buffer and readback as the global
probe, so it does not add a second synchronization. On a strict same-loaded
20-step A/B, global `0.06` plus audio `0.05` achieved 1.405x denoise speedup
with video/audio latent relative-L2 `0.03084/0.06876`; plain global `0.06`
measured `0.03384/0.12353`. On the complete five-second ambient-audio test it
kept all six cache hits and added only about 0.20 seconds of decision wait, so
normalized end-to-end speed remained about 1.389x. Omitting this variable
preserves the faster visual-balanced whole-sequence policy; use cache-off for
an exact quality oracle.

`H3_TEACACHE_FORCE_FRESH=1` tests the controller without reuse and is
byte-identical to the disabled path in the same loaded model.
`H3_TEACACHE_RUNTIME_DISABLE=1` provides that matched disabled arm. TeaCache is
intentionally incompatible with FirstBlockCache, fixed core reuse,
whole-denoiser reuse, token reduction, SSD streaming, Sol-Attn, and per-step
gate skipping until each combination has an independent quality calibration.
EasyCache was also prototyped but removed: its extra full-sequence reductions,
copies, and synchronization made the tested 20-step M4 path slower even when it
reused two steps.

### Rejected attention scheduling experiments

`H3_FUSED_SDPA_OUT=1` evaluates dense BF16 SDPA and the following bias-free
output projection in one fixed-shape MPSGraph. It is numerically exact in the
Metal unit test and in a same-loaded-model real-weight A/B, but only improved a
50-block 512x512x22 forward from 6.1445 to 6.1319 seconds (0.21%). It remains a
diagnostic switch and is not selected over the faster default int8/head-major
attention-output path.

The repository also retains a standalone public-Metal Sol-Attn research
harness: BF16 Q/K/V summaries, diagonal routing thresholds, exact prefix and
neighbor blocks, and FP32 online softmax without PyTorch private headers. Its
129-token full-sink unit gate reached relative L2 `3.85e-4` against dense SDPA,
but the scalar/simdgroup forward is not a production kernel. At 25 blocks it
was 1.30x slower than dense for 22 frames and 1.92x slower for the real
512x512/124-frame layout; approximation drift was also too large. Therefore
`H3_SOL_ATTN=1` additionally requires `H3_SOL_EXPERIMENTAL=1`, and its shaders
are omitted from normal library compilation. The next viable Sol-Attn step is
a BQ32/BQ64 tiled TensorOps implementation with real-QKV all-exact gates, not
further tuning of this scalar prototype.

`H3_SOL_BQ64_EXACT=1` switches that experimental path to a public-Metal
TensorOps BQ64 all-exact kernel. It requires the head-major int8 QKV and
attention-output path (`H3_NAX_FORCE=1` on the tested M4) and preserves dense
attention connectivity. Full 64-key tiles use TensorOps; the final partial key
tile uses exact scalar accumulation to avoid undefined out-of-range TensorOps
reads. This remains a bring-up/performance gate for a later tiled sparse
backend, not a released acceleration mode.

`H3_SOL_BQ64_SPARSE=1` enables the corresponding tiled sparse research path.
It reuses the public-Metal centroid/threshold/route preprocessing, keeps the
packed prefix and prefix queries exact, evaluates routed blocks with BQ64
TensorOps, and represents unrouted blocks with centroid QK plus multiplicity-
aware V sums. `H3_SOL_ROUTE_PROFILE=1` reports exact-block density, while
`H3_SOL_BQ64_VERIFY=1` compares the result with a dense attention oracle. A
deterministic 961-token gate containing both exact and summary blocks reaches
relative L2 `1.69e-3` against the scalar sparse implementation. It is still
rejected for normal inference: at the real 512x512/124-frame shape the
all-exact tiled kernel is about 1.50x slower than MPSGraph dense SDPA, and
route densities that preserve useful video accuracy did not produce a
reliable end-to-end gain. The switch therefore still requires both
`H3_SOL_ATTN=1` and `H3_SOL_EXPERIMENTAL=1` and is never selected by default.

`H3_MPS_EXECUTABLE_LINEAR=1` tests explicit precompiled fixed-shape MPSGraph
linear executables. A 25-block all-BF16 A/B was byte-identical but measured
2.8171 seconds through the graph API and 2.8196 seconds through the executable,
so the normal graph path remains selected.

### Exact DiT fusions

Every active DiT block fuses its attention residual gate with the following MLP
AdaLN. The rounded BF16 residual is still written exactly, but the same row is
kept in threadgroup memory for normalization, eliminating one dispatch and one
global reread. Away from token-reduction boundaries, the MLP residual gate also
produces the next block's attention AdaLN and carries that normalized state
across the loop. `H3_DISABLE_FUSED_GATE_ADALN=1` and
`H3_DISABLE_FUSED_CROSS_BLOCK_ADALN=1` restore the two-kernel oracles.
The final audio/video AdaLN kernels bind directly to offsets in the residual
stream, avoiding two slice blits and 18.8 MiB of scratch at 512x512 (29.4 MiB
at the 864-class benchmark shape).
`H3_DISABLE_FUSED_FINAL_SLICE=1` restores the copy-plus-AdaLN oracle at load.
The BF16 final heads then apply AdaLN while loading their 16x16 projection
tiles, preserving the standalone rounding and accumulation order while
removing another equally sized normalized activation. The two optimizations
together save 37.5/58.9 MiB. `H3_DISABLE_FUSED_FINAL_HEAD=1` restores the
offset-AdaLN-plus-linear oracle at load.

### Token-reduction internals

`--token-reduction` is an independent aggressive DiT mode. After block 3 it
pairs adjacent horizontal target-video tokens while leaving text, audio,
conditions, and reference tokens exact. The complete full-resolution state is
kept as a bypass. During the first ten noisy evaluations it restores before
block 40; subsequent detail-forming evaluations restore before block 30. Each
token returns as its original value plus the update learned by its pair, so
within-pair detail is not discarded.
The pooling kernel writes only true-pair baselines into a dense tail of the
already allocated attention scratch buffer; odd-width singleton tokens need no
baseline. The full bypass uses the oversized QKV tail when it fits, with a
guarded dedicated fallback only for reference-heavy layouts. Common text-only
canvases therefore add no activation arena at any token-grid width. Pooling
also snapshots both source tokens while their BF16 values are already in
registers, avoiding a separate full-hidden blit and redundant source read. The
same entry kernel keeps each pooled row in threadgroup memory and emits the
first reduced block's attention AdaLN, eliminating another global residual read.
At the restore boundary, the first full-resolution attention AdaLN is fused
into expansion: a 10.5 KiB threadgroup row avoids a global residual reread while
still writing the exact bypass needed by the following residual branch.
On a thermal-balanced 512x512x22, 19-forward IT M5 Max A/B this reduced denoise
time from 39.13 to 28.06 seconds (28.3%). Final video/audio latent relative L2
was 5.56%/15.14%. First/middle/last fox frames retained one clean muzzle,
coherent legs, and sharp fur; an independent surfer remained consistent with
one rider and board through the wave spray. It changes composition and is
therefore opt-in rather than the close-reference default.
`H3_TOKEN_REDUCTION_BLOCKS` can override the later `4:30` interval;
`H3_TOKEN_REDUCTION_EARLY=STEPS:END` overrides the early schedule and `0`
disables it. `H3_DISABLE_TOKEN_REDUCTION=1` provides an in-context exact oracle.
`H3_DISABLE_FUSED_TOKEN_POOL_ADALN=1` and
`H3_DISABLE_FUSED_TOKEN_ADALN=1` independently restore the two-kernel entry and
exit boundaries for diagnosis.
Token reduction composes cleanly with the validated `--layers 45 --reuse 2`
settings: on the same 512 benchmark it reduced that profile from 16.69 to
12.60 seconds (24.5% marginal), and independent fox and surfer renders stayed
coherent. Do not combine it with both `--layers 40` and `--reuse 3`; that
6.47-second experiment produced chromatic ringing and ghosted limbs despite
acceptable latent norms.

### Internal canvas and video VAE

`--render-width` and `--render-height` run the model and VAE on a lower
same-aspect internal canvas, then high-quality vImage-scale RGB frames to the
requested output size before callbacks, terminal display, and encoding. This is an
explicit quality/speed tradeoff: a measured 384-to-512 prompt render reduced
M5 DiT time by 33% and video-VAE time by 18% while retaining a clean,
recognizable photorealistic result. Both values must be multiples of 32; the
exact output canvas remains the default.
For square 512 output, 384 is the fast-quality point and 320 is the validated
aggressive point. The latter produced a coherent walking fox and repeated at
8.02 seconds of DiT versus about 15.82 seconds natively. Native 256 uses the
same-cost spatial-RoPE adaptation described above; it remains a fast composition
preview rather than a substitute for a 512- or 768-class final render.
The video VAE automatically chooses a 256-320 pixel spatial tile from the
requested canvas geometry, minimizing repeated overlap work while keeping peak
storage bounded. `H3_VAE_TILE_PIXELS=256` restores the original conservative
tile plan for close-reference diagnosis.
When the chunk uses four spatial tiles, the final non-resident decoder batches
two tiles per transformer pass. On a 512x512, 124-frame M4 Max decode this
reduced warm Video VAE time from 34.82 to 33.24 seconds (4.5%), halved MPSGraph
linear/attention dispatches, and raised tracked peak storage from 9.45 to
9.88 GiB. A captured real Turbo latent measured F32 RGB max-absolute error
`7.39e-6` and relative L2 `3.25e-7` against batch one. Set
`H3_VAE_TILE_BATCH=1` for the exact previous scheduling path; batch four is
available for experiments but saves only another roughly 0.05 seconds while
raising peak storage to 10.73 GiB on this workload.

### TAEH3 fast decoder

The repository includes a native public-Metal/MPSGraph implementation of the
MIT-licensed madebyollin TAEH3 decoder. It keeps the normalized H3 latent in its
native `[24,T,H,W]` contract, uses BF16 NHWC Conv2D, causal temporal-memory
blocks, nearest spatial upsampling, TGrow channel/time relayout, and a fused
pixel-shuffle/time-trim output kernel. It is an explicit fast-quality decoder;
the official Video VAE remains the default.

The upstream checkpoint is not committed. Fetch the source-pinned artifact,
verify/convert it without installing PyTorch, then opt in with its containing
directory:

```sh
curl -L \
  https://raw.githubusercontent.com/madebyollin/taehv/e589fddc076e77f5ba8cd6baabe4ba3260b261cd/taeh3.pth \
  -o models/taeh3.pth
python3 tools/convert_taeh3.py \
  models/taeh3.pth models/taeh3.safetensors

H3_TAEH3_WEIGHTS=models H3_NAX_FORCE=1 ./h3 \
  -d models/MiniMax-H3-Turbo -p "..." \
  --seed 42 --width 512 --height 512 --seconds 5 \
  --steps 4 --layers 50 --reuse 1 -o output.mp4
```

The converter accepts only the pinned SHA-256 and whitelists the three pickle
globals used by this state dict before writing BF16 safetensors. On the tested
M4 Max, a captured 512x512x124 Turbo latent decoded in 0.650 seconds with a
2.548 GiB tracked Metal peak, versus roughly 34 seconds for the official tiled
decoder. A full Turbo/int8/TAEH3 CLI run completed in 193.54 seconds with zero
swaps. Six-point visual review preserved subject, motion, framing, and temporal
continuity; encoded-video SSIM against the official decoder was 0.832, with the
expected loss of some fur/background high-frequency texture. TAEH3 is therefore
recommended only through the explicit environment switch for FL2VA final
decode; Ref2VA is rejected until separately validated. Denoising previews keep
using the official resident preview decoder.

TAEH3 also composes with the validated lower internal canvas and gate-ranked
layer profiles. Single-prompt 512-output measurements on the same M4 Max were:

| Internal canvas / active blocks | End-to-end | Positioning |
| --- | ---: | --- |
| 384 / 45 | 94.32 s | fast-quality; 22.1% faster than the 121.11 s official-decoder arm |
| 384 / 38 + pinned gate skip | 80.96 s | more aggressive fast candidate |
| 352 / 38 + pinned gate skip | 69.52 s | fast/preview boundary; needs broader prompt validation |
| 320 / 40 + pinned gate skip | 62.80 s | preview only |
| 320 / 38 + pinned gate skip | 59.92 s | fastest coherent preview in the current gate |
| 288 / 40 + pinned gate skip | 54.50 s | ultra-preview only; composition and detail change visibly |

At 384, a captured latent decoded with the official VAE in 26.262 seconds and
TAEH3 in 0.379 seconds; encoded-video SSIM was 0.9031 and PSNR 33.07 dB. The
320/38 clip preserved one coherent fox and continuous gait but simplified fur,
legs, and branches (SSIM 0.7164, PSNR 19.50 dB against the 320/40 reference).
The 288/40 clip also remained temporally coherent, but saturation, composition,
legs, and tail changed more substantially (SSIM 0.6387, PSNR 15.68 dB); it is
deliberately labeled ultra-preview. Lower canvas/layer approximations can change
framing and fine detail, so none of these aggressive profiles are promoted to a
general quality default from one prompt.

In an interactive two-prompt 320/40 session, the first five-second request took
62.30 seconds and a different second prompt took 55.41 seconds. Resident request
rebuild was 0.356 seconds, saving 6.89 seconds end to end (11.1%) after warm-up.
Peak process footprint was 36.755 GB, maximum RSS was 21.791 GB, and the run
reported zero swaps on the 64-GiB M4 Max. The resident core is reused only when
its independent identity key matches checkpoint shards, FL2VA/Ref2VA mode,
sigma schedule, active-layer policy, and all allocation/backend switches;
otherwise the code fails closed to a complete reload.

### Weight residency and streamed prompt encoding

On M5-class GPUs, persistent transformer weights are mapped directly from their
safetensor shards instead of copied into anonymous shared buffers. This keeps
the 37 GiB model file-backed/reclaimable and slightly improves total transformer
time; M3 uses the faster copied-buffer path. `H3_ZERO_COPY_WEIGHTS=0` disables
the M5 selection for diagnostics.
The streamed Qwen text encoder preallocates a small ring of future layer
buffers and fills them on eight I/O workers while Metal executes the current
layer. The default ring depth is two layers on M3/older hardware and three on
M5, where the target machine has 128 GiB. `H3_QWEN_PREFETCH=0` restores the
single-layer synchronous reference path; values 1-8 select the worker count,
and `H3_QWEN_PREFETCH_DEPTH=1` through `6` overrides the ring depth.

`--ssd-streaming` is a separate, more aggressive residency mode for the DiT.
Only its small per-block normalization weights remain resident. Two complete
BF16 matrix slots alternate while a background reader fills the next slot in
checkpoint-offset order; the current Metal command buffer runs concurrently.
Darwin uncached reads avoid retaining a second copy in the filesystem cache.
The first active block is prefetched again during the final block, so a cached
interactive DiT is ready for its next denoiser evaluation. Measurements reached
about 13--14.6 GiB/s from the internal SSD. `H3_PROFILE=1` reports total bytes,
read throughput, and the part of the read wait that was not hidden by GPU work.

### Metal 4 and TensorOps paths

M5 GPUs automatically use native BF16 Metal 4/TensorOps for the DiT QKV and
attention-output projections at sequence lengths up to 2,048. The compact
Morton schedule routes Q/K/V directly into head-major attention inputs, avoids
three MPSGraph input transposes, and is byte-identical to the portable path. It
improves a complete 512x512 50-block forward by about 2% across repeated IT/US
M5 Max runs. For 2,049-3,072 rows, including 864x480, two row-offset Morton
dispatches preserve the efficient tile geometry and improve the complete
forward by about 2% in balanced runs. Still larger sequences stay on MPSGraph.
`H3_NAX=0` disables TensorOps for exact A/B diagnosis. The selection is guarded
at runtime and falls back to the unchanged portable library if compilation is
unavailable.

`H3_NAX_FORCE=1` explicitly allows the same Metal 4 TensorOps/int8 library on a
non-M5 device for controlled testing. On one 64-GiB M4 Max, a 512-square,
124-frame, four-step Turbo run improved denoise by about 3.1% and end-to-end
time by about 1.6%, while tracked DiT peak storage fell from 37.1 to 19.5 GiB.
A 22-frame probe was about 8% slower, so this remains a long-sequence or
low-memory opt-in rather than an M4 default. Compilation failure retains the
portable fallback; compare output quality for each model and OS/Metal version.

`H3_NAX=1` forces the broader native BF16 linear path. It passes the complete
50-block MLX fixture, but remains opt-in: exact-shape microbenchmarks favor its
128-row tile while full DiT runs currently favor MPSGraph scheduling. This
keeps a working NAX integration available for later quantized/fused kernels
without making a benchmark regression the default.
`H3_NAX=mlp` selects a more specialized Metal 4 path: paired FC1 gate/up
TensorOps tiles apply SwiGLU in threadgroup memory and write only the
14,336-wide activated intermediate, then FC2 also stays on TensorOps.
`H3_DISABLE_NAX_MLP=1` keeps the MPSGraph MLP in a context created this way for
same-process A/B testing. The path is deliberately opt-in because scheduling
depends on the OS GPU stack: the primary macOS 26.5.2 M5 Max gained 1.3-2.0%
in isolated real-weight MLP runs but lost about 1-3% in a complete 50-block forward,
while an otherwise identical macOS 26.5 M5 Max gained 1.4% in a same-context
forward A/B. The resulting 50-block velocities were close (1.9% video and 2.4%
audio relative L2), but not byte-identical.

### Specialized projection kernels

The narrow DiT audio/video output heads convert their small released F32
weights to BF16 once and use the Iris-derived 16x16 tiled linear directly on
BF16 activations. At the production 320-render geometry, isolated paired-head
measurements are 2.30x faster on M3 Max and 1.83x faster on M5 Max, with
relative L2 `8.64e-4`; the absolute M5 saving is about 0.6 ms per evaluated
step. Full fox and surfer sequences remained clean and measured 29.9/38.4 dB
against the F32-head renders. `H3_DIT_F32_FINAL=1` restores the close-reference
head and its extra activation buffers.
The F32 `96->5376` video and `32->5376` audio patch projections use a dedicated
16x16 cooperative tile, retaining F32 weights, inputs and accumulation while
rounding the tile result directly to BF16.
Paired production-shape measurements are 1.77x faster on M3 and 1.62-1.78x
on M5; the complete generated RGB stream is byte-identical to the scalar path.
Fusing the final cast improves the 2835-row tile itself from 2.499 to 1.734 ms
on M3 and 1.555 to 1.186 ms on M5, and removes 38.27/59.66 MiB of F32 scratch
at 512/864-class geometry. `H3_DISABLE_FUSED_PATCH_CAST=1` restores the tiled
F32 output plus standalone cast; `H3_SCALAR_PATCH=1` selects the scalar
diagnostic path.
The same tile binds its output directly into the packed hidden stream, removing
the BF16 media staging buffers and their blits. This saves another 19.13/29.83
MiB and improves the 2835-row boundary from 1.847 to 1.730 ms on M3 and 1.282
to 1.184 ms on M5. Contiguous T2VA uses byte offsets; FL2VA/Ref2VA use compact
destination-row maps so each modality remains one large dispatch. A complete
six-segment Ref2VA M5 ABBA remained byte-identical and improved 5.067 to 5.033
seconds per measured forward pair. `H3_DISABLE_FUSED_PATCH_PACK=1` restores the
staging buffers and packing blits.

### Scheduling and activation memory

The DiT core is split into two ordered Metal command buffers so GPU execution
of the first part overlaps CPU encoding of the second. Thermal-balanced ABBA
measurements select a 60%-depth split on M5 (30/50, 27/45, and 24/40), with
roughly 0.5-1.8% wins; M3 automatically splits only the validated 30/50 case,
which measured 1.2% faster, because 24/40 regressed there. The operation order
and generated bytes are unchanged. `H3_DIT_COMMAND_BLOCKS=0` restores one
command buffer; values 1-50 override the split for further tuning.
DiT activation buffers also follow their actual intra-block lifetimes: the QKV
projection arena is reused first for attention heads and then for the normalized
MLP input, while the current attention-output arena becomes the MLP output after
its branch has been consumed. This removes 61.25 MiB at 512-class geometry and
99.63 MiB at 864-class geometry without changing dispatches or arithmetic.
`H3_DISABLE_DIT_ACTIVATION_ALIAS=1` restores separate diagnostic buffers.
MPSGraph tensor-data wrappers for immutable DiT weights and biases are retained
with their resident buffers. This avoids rebuilding the same binding metadata
for every block and denoiser evaluation without copying tensor storage; measured
ABBA gains were 1.6% on M3 Max and 0.4-1.1% on M5 Max. Activation wrappers stay
transient because retaining them regressed the M5. The outputs remain
byte-identical, and `H3_DISABLE_GRAPH_DATA_CACHE=1` restores transient wrappers
for all tensors.
On M3/older hardware, the four MPSGraph segments in each DiT block also reuse
one `MPSCommandBuffer` wrapper for their shared underlying Metal command buffer.
Repeated thermal-balanced runs measured 1.0-1.6% faster on M3 Max; M5 measured
neutral, so it retains fresh wrappers. `H3_REUSE_MPS_COMMAND=0` or `1` overrides
the automatic selection. Results are byte-identical.
On M5, the serving Euler sampler keeps its patch-packed F32 latents and cached
BF16 velocities in Metal buffers. Each selected denoiser refresh is completed
before the next is encoded, avoiding MPSGraph back-pressure while removing all
intermediate latent/velocity readbacks and repacking. Two warm eight-run A/B
sequences measured small 0.1% and 0.3% gains with byte-identical final latents;
the path also saves roughly 16 bytes of transient host state per video-latent
element (about 136 MB at the 768p shape). M3 and older GPUs retain the CPU
sampler by default. `H3_CPU_SAMPLER=1` restores it on M5;
`H3_GPU_SAMPLER=1` selects the GPU-state path explicitly, and
`H3_GPU_SAMPLER_WINDOW=0` enables the slower unbounded encode-ahead diagnostic.

### Checkpoint layout and media pipeline

The released checkpoint stores DiT QKV rows interleaved per attention head.
Native Metal consumes that layout directly in the fused QK-normalization/RoPE
kernel, avoiding a checkpoint transpose and extra RAM. The earlier identity
interpretation was the cause of the noisy diagnostic outputs.

The public generation path decodes the joint audio latent with a streamed native
BigVGAN/AudioVAE and writes synchronized H.264 plus 32 kHz stereo AAC. The native
waveform agrees with the corrected MLX oracle to relative L2 `6.94e-5`.
`--first-frame`, `--last-frame`, and their combination use the released visual
VAE encoder, Qwen3-VL vision tower and three-deepstack multimodal presentation,
0.999 condition augmentation, and fixed condition rows in the native DiT. The
first image is stretched to the target canvas; the last image is aspect-cover
scaled and center cropped, matching the reference implementation. `--ref-image`
selects the distinct Ref2VA transformer, preserves ordered `<Picture N>`
presentation, and uses the released down-only aspect-preserving reference canvas.
`--ref-silent-video` additionally performs bounded 24 fps decoding, the visual
VAE's causal `ceil(T/4)` compression, two-frame Qwen sampling, and timestamped
`<Video N>` presentation. `--ref-video` preserves an embedded soundtrack,
`--ref-video-audio VIDEO AUDIO` supplies an explicit replacement, and
`--ref-audio` appends an ordered standalone clip. Reference audio is decoded as
32 kHz stereo F32, encoded by the native AudioVAE posterior-mean path, mixed as
0.999 clean latent plus 0.001 seeded noise, pinned to the audio condition
timestep 1.0, and packed as width-32 rows on the same rotary timeline as visual
references. Audio inputs are 2-15 seconds, at most three are
accepted, their total decoded duration is capped at 15 seconds, and a standalone
audio reference must be combined with an image or video reference.

The native audio encoder matches the corrected MLX oracle at relative L2
`3.59e-6` on a real two-second stereo fixture. The correction is important: the
original MLX reshape interleaved left/right samples, whereas the official
PyTorch/SGLang path folds intact stereo channels into the batch dimension. On
the 128 GB M5 Max, clean end-to-end image+audio and embedded-video+audio renders
completed in 74.58 and 76.99 seconds respectively, each with about a 40.1 GB
peak physical footprint and zero swaps.

### Profiling and diagnostic paths

`--profile` reports each Metal-backed phase separately: wall time, CPU-side
command encoding, complete commit-to-fence wait, root-command GPU timestamps,
peak live tensor storage, cumulative allocation, and dispatch counts. The wait
measurement is the complete command turnaround; the root GPU timestamp alone
can omit child buffers scheduled internally by MPSGraph and is labeled
accordingly.

The DiT fast path evaluates each BF16 `fc1 -> SwiGLU -> fc2` block as one cached
graph, avoiding separate graph boundaries and persistent intermediate tensors.
Set `H3_DISABLE_FUSED_MLP=1` to retain the close-reference operation boundaries
for numerical diagnosis.

On supported M5 Metal 4 TensorOps hardware, the native int8 MLP engine is the
default. It dynamically quantizes activations, uses per-output-channel weight
scales, and gives the sensitive FC2 input one scale per 1,024 channels.
The selected FC2 kernel keeps scaled partial products in private cooperative
fragments instead of repeatedly spilling a 32 KiB threadgroup tile. A fixed
50-layer, 19-transition 512x512 render measured 36.30 seconds with BF16 MPS and
25.80 seconds with int8 on M5 Max. Beginning, middle, and final decoded frames
retained the same subject, composition, and motion; small edge and fur details
can differ. The current diagnostic implementation retains both BF16 and int8
MLP weights only when an A/B diagnostic requests them. Normal int8 loading
releases each block's BF16 FC1/FC2 buffers after their submitted quantization
finishes, reducing measured peak tensor storage to 25.9 GiB from the BF16
path's 36.4 GiB. Runtime weight quantization still adds startup time.

The fastest M5 path also quantizes each DiT QKV projection and writes its
Q/K/V tiles directly in head-major attention layout before the existing Q/K
normalization and RoPE kernel. In a fixed 50-layer, 19-transition 512x512
render this reduced denoising again, from 25.80 to 19.32 seconds. Sampled
beginning, middle, and final frames remained a coherent detailed fox walking
through snow; quantized attention can change framing and fine detail. Use
`--use-slower-bf16-qkv` for the close-reference BF16 projection. Normal int8
loading releases the redundant BF16 QKV weights after quantization.

The following attention-output projection is int8 as well on the default M5
path. Crossed same-model tests improve a complete forward by another 4.5-5.5%
at 512 and 864. A decoded fox render remained clean and closely matched the
int8-QKV-only composition; its thermally hot denoise measured 19.18 seconds.
Use `--use-slower-bf16-attention-output` to retain that projection in BF16.

On that int8 path, SDPA now leaves its result in native
`[head,row,dimension]` order. A specialized 256-thread kernel gathers and
quantizes each H3 row directly into the projection's row-major int8 buffer,
eliminating the intervening full-width BF16 transpose without changing any
output byte. Thermally controlled crossed runs improve complete 512 and 864
forwards by roughly 0.2-1.2%. Use
`--use-slower-row-major-attention-output` to restore the explicit BF16
row-major SDPA output and ordinary quantizer.

The M5 path also folds QKV and MLP activation quantization into the preceding
gated AdaLN kernel. This removes 99 standalone quantizer dispatches per
50-layer forward while preserving the previous output bytes, improving crossed
512/864 measurements by about 0.3-0.6%. Use
`--use-slower-unfused-int8-inputs` to restore the standalone quantizers.

The fused gated-AdaLN path loads its full 5,376-wide H3 rows as BF16x4 vectors
and writes int8x4. It stages the rounded values locally before computing the
original per-thread RMS sequence, so the reduction tree and every output byte
remain unchanged. Crossed measurements save roughly another 0.1-0.5%. The
existing `--use-slower-unfused-int8-inputs` option retains the portable scalar
and standalone-quantizer fallback.

Q/K RMS normalization and RoPE are performed inside the int8 QKV projection
tile as well. The fused epilogue is byte-identical and improves complete
forwards by 2.1-3.2% at 512 and 1.0-1.8% at 864 in crossed M5 measurements.
Use `--use-slower-unfused-qkv-rope` to restore the separate Q/K kernel.

That epilogue processes four adjacent Q/K dimensions per work item with
BF16x4 loads and stores. The per-element arithmetic and BF16 rounding order are
unchanged, while crossed cool-state measurements improve complete forwards by
about 0.4-1.0% at both 512 and 864. The same
`--use-slower-unfused-qkv-rope` option restores the scalar standalone path.

At up to 2,048 rows, the exact RMS loop uses BF16x4 loads followed by four
explicit ordered FMAs. This preserves every output bit and improves 512-class
forwards by another 0.5-0.6%; larger shapes retain scalar loads because the two
forms tie there. Use `--use-slower-scalar-qkv-rms` to force scalar loads.

The int8 attention-output projection caches its 128 row and column scales in
1 KiB of threadgroup memory instead of rereading them for every cooperative
fragment element. Above 2,048 rows the fused QKV kernel uses the same idea and
then recycles that storage for inverse RMS values; smaller QKV shapes retain
direct loads because the two forms tie there. Both are byte-identical and
improve complete forwards by about 0.2-0.7% where selected. Use
`--use-slower-uncached-int8-scales` to restore direct device-scale loads.

For sequences of at most 2,048 rows, the H3 attention-output projection also
compiles its 7,168-by-5,376 shape into the TensorOps kernel. The result remains
byte-identical while saving about 0.2-0.8% in crossed complete 512-forward
measurements. Larger sequences retain the dynamic-shape kernel because the
specialization regresses there. `--use-slower-uncached-int8-scales` restores
the general dynamic, direct-scale-load implementation.

FC1 also uses an H3-specialized, compile-time 5,376-wide TensorOps loop. It is
byte-identical to the generic loop and saves about 0.1-0.4% in crossed complete
forwards. Use `--use-slower-dynamic-fc1-k` to restore the runtime-bound loop.

```sh
./h3 --profile -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 50 --reuse 1 -o outputs/fox-int8.mp4
```

Use `--use-slower-bf16-mlp` to force the portable close-reference MPS/BF16 MLP
path for numerical comparison. Older Metal hardware selects that path
automatically when the required native TensorOps kernels are unavailable.
For FC2 activation quantization, sequences of at most 2,048 rows use an exact
128-thread reduction. Each thread retains its eight BF16 input values while
computing the group maximum, avoiding a second device-memory read when it emits
the int8 values; crossed M5 measurements improved complete 512 forwards by
about 0.2-0.8% without changing any output byte. Larger sequences retain the
measured 256-thread kernel. `--use-slower-grouped-quantizer` forces the latter
at every size for A/B comparison.

The native baseline targets the original `FL2VA/` and `Ref2VA/` checkpoint
trees. Model phases are loaded and released separately so the 33B transformer,
Qwen encoder, and decoders never have to coexist in unified memory.
