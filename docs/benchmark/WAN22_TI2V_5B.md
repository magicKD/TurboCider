# Wan2.2 TI2V-5B Turbo: development evidence

This document records an experimental TurboCider Wan2.2 executor and its
measured examples. The results do not by themselves establish broad numerical
parity, visual quality or a universal speed guarantee. HunyuanVideo 1.5 is
outside the current implementation focus.

## Components and reference schedule

The reference checkpoint is the full three-step distilled Transformer from
`FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers`, not a LoRA adapter. Its
model card describes DMD training at 121×704×1280 and warns that quality may
degrade at other resolutions; the 512×512 tests here are outside that training
geometry. The
tested source checkpoint SHA-256 is
`a1cf4acd5d0242ca0cad68e115064f4ac7e748686c9cf0da050ce87decc6b9bf`.
The UMT5-XXL FP16 text encoder SHA-256 is
`7b8850f1961e1cf8a77cca4c964a358d303f490833c6c087d0cff4b2f99db2af`;
the Wan2.2 VAE SHA-256 is
`e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156`.
Weights are local artifacts, not committed. A header-only APFS clone maps
Diffusers tensor names to ComfyUI names; tensor bytes and offsets are unchanged.

The ComfyUI comparison uses stock Wan22 model, first-frame conditioning,
UMT5 and full VAE nodes. A diagnostic-only custom sampler supplies DMD's
fresh-noise update, with FastVideo's MLX reference training indices
`1000,757,522`, flow shift 5 and `warp_denoising_step=true`: sigmas approximately
`1.0,0.9396723,0.8452073`. This is not stock three-step Euler. FastVideo's
other pipeline versions and four-step community checkpoints can use different
schedules; do not treat these results as universal.

### Upstream I2V qualification

In [FastVideo issue #711](https://github.com/hao-ai-lab/FastVideo/issues/711),
a project collaborator stated on August 5, 2025 that FastWan-5B had been
trained on T2V data and might only support T2V. This is historical context,
not proof that every current FastVideo pipeline rejects image conditioning.
The current [FastVideo support matrix](https://github.com/hao-ai-lab/FastVideo/blob/main/docs/inference/support_matrix.md)
lists both T2V and I2V for FastWan2.2 TI2V-5B and separately notes I2V
quality issues for Wan2.2 TI2V-5B. The 512 and 768 portrait experiments
below demonstrate that this integration does follow the initial image, but
they do not establish robust identity preservation or the cause of its drift.
The author of [quanhaol/Wan2.2-TI2V-5B-Turbo](https://huggingface.co/quanhaol/Wan2.2-TI2V-5B-Turbo)
describes that separate four-step checkpoint as an I2V-distilled version;
that description does not prove it performs better on any particular image.

## First ComfyUI MPS baseline

On November 9, 2026, an M4 Max with 64 GiB unified memory ran an isolated
ComfyUI 0.39.0 instance with PyTorch 2.12.1, MPS, BF16 UNet/text/VAE and
`--gpu-only`; MPS CPU fallback was not enabled. A 512×512, 97-frame, 24 fps
image-to-video request, seed 42, completed in **152.28 s**, including output
encoding and retrieval but excluding server startup and component hashing.
The run saved a verified 97-frame MP4 and independent evidence of all three
sampler calls. ComfyUI reported partially unloading the text model while
loading the transformer; this is a fresh request, not a warm-session median.
Matched TurboCider job measurements appear later in this document. This single
ComfyUI request does not establish a stable cross-framework median.

The input was a generated red fox image. Sampled output frames show forward
motion, but the face/feet soften and the color darkens across time. This is
one image, prompt and seed, not a model-wide visual-quality conclusion.
The video and run-specific reports remain in the local `outputs/` directory.

## Native port status

The existing Wan 2.1 1.3B executor uses 16 latent channels, only supports
text-to-video and limits frames to 81. Wan2.2 TI2V requires a separate
48-channel VAE and first-frame conditioning with per-position timesteps.
The new `native/models/wan22` Transformer block and transformer are an
experimental native candidate. A tiny BF16 two-timestep block comparison
against PyTorch passed.

Additionally, an isolated ComfyUI 128×128, 9-frame run captured the actual
input and output of the **first transformer block at the first DMD step**.
The native block using the original full checkpoint and the same captured
BF16 inputs had relative L2 error **0.0026631**, mean absolute error
**0.0011433**, and maximum absolute error **0.25** against ComfyUI. This is
component evidence, not a full 30-block or video-level parity result. The
optional fixture-backed test is `tests/native/test_wan22_real_block_parity.py`.

The new full-weight native VAE **first-frame encoder** also has a fixture
captured at the actual input/output of ComfyUI's stock Wan2.2 VAE. At 128×128
its BF16 latent relative L2 error was **0.0091814**; at 512×512 it was
**0.0119813**, with mean absolute error **0.0040677**. These are raw,
unnormalized 48-channel VAE means, not decoded pixels. The fixture-backed test
is `tests/native/test_wan22_vae_encoder_parity.py`.

The chunked native decoder was also checked with the same real ComfyUI VAE
latents. For 512×512 and 97 frames it returned all 97 frames with no
nonfinite values; raw pixel relative L2 error was **0.004112** and mean
absolute error was **0.001644** (on the decoder's unbounded `[-1, 1]`-style
output, before clipping). Relative L2 was **0.00306** on frame 0 and
**0.00343** on frame 96. The peak absolute deviation was **0.2003**, so this
is not bit-for-bit equality. A second 128×128, 9-frame fixture gave relative
L2 **0.006535**. The optional test is
`tests/native/test_wan22_vae_decoder_parity.py`. This validates the decoder
on these examples, not the complete generation or any performance target.
Real-weight full-trajectory parity and broader performance/quality
qualification remain open. Packaged desktop-job validation is recorded below;
it does not by itself exercise a graphical window.

## First native 512×512 trajectory

One native GPU prototype was run with the same 512×512 fox image, seed 42,
97 frames and FastVideo three-step schedule. It reused captured ComfyUI text
conditioning instead of running native UMT5, and wrote uncompressed tensors
before a separate native MP4 export. The prototype took **55.85 s** including
model loads, first-frame VAE, denoising, video VAE and tensor write, but
excluding text encoding and MP4 export. This number is **not directly
comparable** with the 152.28 s full ComfyUI request above. Its process peak
footprint was approximately 47.37 GB, as reported by `/usr/bin/time`.

The native decoder returned 97 finite frames. Against a separately captured
ComfyUI run's raw decoded pixels, relative L2 was **0.08684** and mean
absolute error **0.02171**; error increased toward the final frame. Sampled
images show a similar fox, background and forward motion, with local
differences in fur, face and legs. These are results for one prompt/image/seed,
not proof of video-level equivalence or a performance win. The raw output,
native H.264 video and five-frame comparison are in local `outputs/`.

A real-weight 128×128 first-step Transformer comparison initially had about
23.1% output relative L2 error. Using a fused BF16 matrix-multiply plus bias
reduced it to about 17.3%; intermediate block errors grow across layers, so
the exact numerical discrepancy remains under investigation. At 128×128 the
reference and native full trajectories both exhibit strong distortions;
that smoke-test size should not be used to judge 512×512 quality.

## TAEW2.2-Super decoder comparison

The upstream TAEHV repository's `9055eee0391548d1787938d040a161cbe396bf98`
commit was checked on November 9, 2026. Its README recommends
`taew2_2_super` for a higher-quality Wan2.2 5B tiny decoder, while noting it
does not exactly reproduce the full VAE. The safetensors file has SHA-256
`06503c1f388993e5314e2d1c7ab1276af57b4adab2f467196ceff100d3a216c4`
and Git blob `2021b7ee3c9996cb2248fe36b044f18c6fefcb97`. It is not checked
into this repository. TAE expects normalized diffusion latents and emits
`[0,1]` pixels; the full VAE uses raw VAE-space latents and emits
`[-1,1]`-style pixels. The comparison explicitly converts between these
conventions and uses the full encoder for the first-frame condition.

Against the pinned upstream PyTorch MPS decoder, the native Super decoder
using real weights and the same 128×128, 9-frame latent produced mean
absolute error **0.0002208**, maximum absolute error **0.0078125**, and
relative L2 **0.0007319**, on `[0,1]` RGB pixels. A separate synthetic-weight
architecture parity test also passed.

On the same M4 Max/64 GiB, the same saved 512×512, 97-frame FastVideo
latent was decoded in separate native processes, each with one warmup and
three measured repetitions. Timing excludes model load and MP4 export.

| Decoder | Decode seconds | Median | MLX peak bytes |
| --- | --- | --- | --- |
| Full VAE, BF16 | 37.3735, 37.4142, 37.3989 | 37.3989 | 13,341,486,216 |
| TAEW2.2-Super, FP16 | 0.290991, 0.290791, 0.290467 | 0.290791 | 4,348,450,844 |

The approximately **128.6×** ratio describes only this warm native decoder
microbenchmark, not end-to-end generation. The MLX counters are allocator
peaks in separate processes, not total system memory or a guarantee for other
sizes. The Super implementation decodes the frames in parallel; the full VAE
uses the native causal chunked implementation. This is an implementation- and
device-specific comparison, not a claim about all full-VAEs.

On this single fox trajectory, comparing every clipped RGB frame before MP4
encoding gave mean absolute error **0.00950**, PSNR **33.62 dB**, and mean
SSIM **0.9466** relative to the native full VAE. These are full-VAE agreement
metrics, not direct scores of fidelity to the real scene. Visual inspection of
frames 0, 24, 48, 72 and 96 showed similar motion and structure, with small
changes around the eyes, fur and shading. Both outputs became softer near the
end, so changing the decoder alone does not remove that degradation. This is
one image, prompt and seed; it does not establish general visual quality or
temporal stability for humans, text or other scenes.

Local reproducibility artifacts are under
`outputs/wan22-vae-compare-512x97-20261109-v1/`: `full.mp4`, `super.mp4`,
`side-by-side.mp4` (full on the left), contact sheets and `metrics.json`.
`tools/validation/wan22_vae_compare.py` calculates the metrics from saved
pixel tensors; the measurement probes and upstream reference tool reside under
`tools/native/` and `tools/validation/` respectively.

## Paired desktop job measurements

Using the packaged native library through `StudioState` and `NativeJobStore`,
three separate full-VAE jobs were interleaved with three Super jobs on the
same M4 Max. The same 512×512 input, prompt, seed 42, three-step FastVideo DMD
schedule, 97 frames and 24 fps were used. These job-wall measurements include
text encoding, first-frame encoding, transformer and VAE loads/inference,
MP4 encoding and history bookkeeping; each is a fresh process, with OS caches
potentially warm. They are not GUI-window automation.

| Final video decoder | Job wall seconds | Median |
| --- | --- | --- |
| Full VAE | 57.209, 55.634, 55.634 | 55.634 |
| TAEW2.2-Super | 19.252, 19.302, 19.256 | 19.256 |

Super reduced the median end-to-end job time by about **65.4%** (approximately
**2.89×** throughput for this test), not 128.6×. The median measured
transformer-load-plus-three-step phase was roughly 16.6 s for both decoders;
the Super video-VAE-load-plus-decode phase was roughly 0.36 s versus roughly
37.56 s for the full VAE. These timings are not isolated kernel timings.
The 97-frame output is **4.04 seconds** at 24 fps, not a five-second video.
The 97-frame limit is currently an experimental App constraint.

The desktop setting defaults to Super as requested; the user can explicitly
select the full VAE. The packaged default is an approximate final decoder,
while the full VAE still encodes the first frame. The request receipt
identifies the approximation and decoder. The desktop package can bundle the
pinned Super safetensors file; missing or invalid Super weights must fail
rather than silently falling back. Direct native requests without a Tiny VAE
configuration continue to use the full VAE. The tests and sample videos
remain local; broader visual and GUI qualification are still needed.

The default-Super package smoke test loaded the pinned checkpoint from an
isolated copy of the App's own `Contents/Resources/tiny-vae/` directory and
completed a 512×512, 97-frame job in **20.95 s**. The signed shipping package
was separately verified to have the same contents and checkpoint hash. This
checks the desktop draft/job-store path and bundle resource resolution, not a
GUI click-through or the presence of the much larger external Wan2.2 model.

For a repeat reference, three independent ComfyUI MPS requests using its
stock full-VAE nodes and the diagnostic DMD sampler completed in **152.283**,
**145.581** and **144.555** s (median **145.581** s). These include the
request and MP4 export, but exclude server startup and model-file hashing.
ComfyUI's BF16 text/transformer/VAE implementation differs from the native
MLX implementation, and models may be staged differently, so this is an
end-to-end implementation comparison rather than operator parity. On the
paired compressed MP4s, FFmpeg's YUV SSIM All was **0.96064** between native
full-VAE and ComfyUI, and **0.97243** between native full and Super. Those
numbers include encoder and color-format effects and are not directly
comparable to the raw-RGB SSIM above. The previously observed larger
128×128 transformer numerical discrepancy remains unresolved.

## Portrait and higher-resolution investigation

On one 512×512 first-frame portrait, the original App request contained 525
SentencePiece content tokens. The old native path silently retained only the
first 511 plus EOS; ComfyUI encoded all 525 plus EOS. The truncated suffix was
part of the requested ending, but this does not by itself explain earlier
identity, hand or sword changes. Wan2.2 now supports a bounded extended context
of up to 1023 content tokens plus EOS and fails explicitly above that limit;
up to 511 content tokens still use the original 512-token padded shape.
FastVideo's MLX example uses a 512-token maximum, so longer prompts are a
deliberate Comfy-style extension rather than identical FastVideo behavior.

The same image, seed and 97-frame request were also run through an isolated
ComfyUI MPS graph with the FastVideo three-step sampler. It showed related
identity, hand, clothing and sword changes. This comparison is not fully
controlled for text length (526 versus the old native path's 512 tokens) or
decoder (full versus Super). Decoding the exact Comfy latent with both VAEs
preserved the same principal structural changes, while altering some facial
and clothing detail. On this one sample, raw-RGB mean SSIM between the two
decoders was 0.9455; it is an agreement statistic, not a quality score.

An additional 512×512 run called the unmodified FastVideo MLX Wan2.2 DiT
blocks with the native captured first-frame/text conditions, the same CPU noise
draws and three-step schedule, then used Super for decode. The small wrapper
implements image conditioning because FastVideo's inspected MLX example is
text-to-video; it is not the official end-to-end I2V CLI. The output again
showed similar motion and identity changes. This is evidence for this sample,
not proof that all native and FastVideo trajectories are equivalent.

The same saved request and seed ran at 640×640 in 33.19 s and 768×768 in
49.10 s using the old 512-token prompt path and Super decoding. The original
source image was only 512×512. Higher resolution changed detail and motion,
but visual inspection still found identity drift and inconsistent hand/sword
geometry; changing dimensions changes noise shape, so a single seed does not
make the trajectories pixel-paired. A separate 768×768 full-VAE decode took
87.95 s and reported a 53.10 GB peak process footprint on this device.
The 768×768 Super and full VAE decodes of that same latent had raw-RGB mean
SSIM 0.9758. These are single-run results on one 64-GiB M4 Max, not broad
quality or memory guarantees. App canvases over 512 are therefore experimental
and require at least 64 GiB unified memory.

The corrected desktop job path was then run with the entire 526-token prompt,
the same input and seed, and the bundled Super decoder. All measurements are
single fresh-process jobs, including model loads, generation and MP4 export;
they are not GUI-window automation or a timing distribution:

| Canvas | Frames / fps | Job wall time |
| --- | --- | --- |
| 512×512 | 97 / 24 | 18.89 s |
| 768×768 | 97 / 24 | 49.21 s |
| 512×768 | 97 / 24 | 29.36 s |

The square 768 result had clearer small features on this input, but the face
still drifted away from the first frame, and hand, sleeve and sword motion was
not consistently stable. A separate 768×768 run with the same input and seed
and a shorter, simpler motion prompt took 49.01 s; identity and hand/sword
consistency were still imperfect. Prompt changes and different noise shapes
alter the trajectory, so these examples cannot isolate a general causal
effect or establish a reliable quality improvement. The 512×768 run validates
the independent width/height path, but changes the composition because the
square source is cropped, so it is not a controlled resolution comparison.

The 768-pixel ceiling is a conservative experimental App constraint, not the
FastWan training geometry. The official Wan2.2 TI2V-5B example uses 1280×704
or 704×1280, and the FastWan model card specifies 121×704×1280 training;
neither geometry has been validated on this Mac in this implementation.

## Matched FastVideo MLX DiT follow-up

A further comparison used the same portrait, seed 201748011, 97 frames,
24 fps, FastWan checkpoint, three warped DMD steps and Super decoder at
512×512 and 768×768. The FastVideo side executes the unchanged MLX DiT at
commit `a943220c115228ade5d57b3bab9a6a87fd600a10`, but supplies the native
captured text and first-frame latents and a custom I2V/DMD wrapper. Both
sides used the same CPU random-number draws at each resolution; all three
draws were checked bit-identical against PyTorch for these two tensor shapes
and this seed. The 526-row
text conditioning exceeds the FastVideo MLX example's usual 512-token
limit; this is an explicit diagnostic extension, not an official end-to-end
FastVideo I2V run or an independent test of image/text encoding.

At 512, FastVideo's three DiT steps took **18.054 s** in total; at 768,
they took **49.942 s**. These timings exclude text and image encoding,
weight loading, video decoding and export, so they must not be compared as
complete App job times. The checked native-versus-FastVideo raw latents had
relative L2 errors **0.1706 / 0.2152** at 512 / 768 (using the native raw
latents as the denominator); the first latent frame was equal and the other
frames differed. After Super decode and MP4 export, FFmpeg's full-frame YUV
SSIM All was **0.939537 / 0.942964**, respectively. This is numerical and
compressed-video agreement, not an identity or perceptual quality score,
and does not establish exact parity or rule out native implementation errors.
To isolate the first call, the 512×512 FastVideo input and BF16 prediction
were captured and fed unchanged to the native Transformer at timestep 1000.
Its prediction had relative L2 **0.02563**, mean absolute error **0.02005**
and cosine similarity **0.999553** against FastVideo. With and without
cross-attention reuse and fused RoPE, the native prediction was bit-identical
on this input. This isolates a same-input Transformer difference before the
later three-step trajectory divergence; it does not identify the particular
operation responsible, establish which result is correct, or prove that the
differences do or do not affect perceived faces.

In the earlier 128×128/9-frame ComfyUI capture, the native prepared text
context differed from the captured ComfyUI context by relative L2
**0.002457**. Re-running the two BF16 text projections and tanh GELU on
PyTorch **2.14.0 MPS**, using the saved text and original checkpoint, produced
a context bit-identical to the native capture; the captured ComfyUI process
reported PyTorch **2.12.1**. This rules out a difference between those two
tested projection results, but it does not establish why the older ComfyUI
capture differed or which result is more faithful. Differences in version,
kernel behavior or capture path remain hypotheses, not diagnoses.

Face crops sampled every six frames showed that the 512 result had rapid
changes in the eyes, mouth and facial proportions from about frame 6; both
the native and FastVideo DiT outputs showed related changes. The 768 result
had clearer features and a steadier-looking face, but the face still moved
away from the first-frame identity, and hands, clothing and the sword were
not consistently preserved. A second native seed, 201748012, produced the
same qualitative tradeoff: improved definition at 768 but noticeable
identity drift at both sizes. It took **18.116 s** at 512 and **47.657 s**
at 768 in isolated probes including text, model loads, inference, decoding
and diagnostic output; neither number is a GUI job time or a median. The
512 source image was enlarged before the 768 run, and the change in canvas
also changes the shape and realization of the noise. Two seeds of one image
are not enough to establish a general quality advantage or a sole cause.

The local comparison videos and face sheets are under
`outputs/wan22-fastvideo-extended-{512,768}x97-20261110-v1/` and
`outputs/wan22-portrait-second-seed-{512,768}x97-20261110-v1/`. Face
detection is only a crop aid and may choose an incorrect region; inspect the
corresponding full video when interpreting a crop.

## FastVideo MLX versus dense-CUDA timestep conventions

The inspected FastVideo commit's MLX example maps indices 1000, 757, 522
through a shift-5 table, producing model timesteps approximately
**1000, 939.672, 845.207**. Its non-causal dense CUDA DMD stage instead
passes **1000, 757, 522** directly to the model and looks up the nearest
sigma in a separate shift-8 training table: **1, 0.7567567, 0.5217391**.
The repository's published data-free 5B training script omits the optional
`warp_denoising_step` flag, whose inspected training-argument default is
false. These observations concern the pinned source snapshot; they do not
prove which exact code or configuration produced a particular checkpoint.

The diagnostic FastVideo MLX I2V wrapper was extended to test the CUDA
*timestep and sigma values* while holding its input, checkpoint, CPU FP32
noise, FP32 update and Super decoder fixed. It is still not the official
end-to-end CUDA I2V path: CUDA BF16 noise and intermediate rounding, image
preprocessing and other pipeline components were not reproduced. Two seeds
of the same portrait were paired at 512×512, and one seed at 768×768, each
with 97 frames. Both schedules produced identity and geometry changes. The
raw-timestep samples had stronger contrast and different motion, with some
overbright areas; subjective inspection did not establish a consistent
identity improvement. The current App default was therefore not changed.

For seed 201748011, the three FastVideo DiT steps took **18.054 / 17.882 s**
at 512 and **49.942 / 49.974 s** at 768 for MLX-warped / CUDA-timestep
respectively. These are single GPU DiT-only runs, not end-to-end App timings.
Artifacts are under `outputs/wan22-fastvideo-cudaraw-portrait-*`; each
side-by-side MP4 has the MLX-warped result on the left. More inputs and an
independent supported I2V reference are needed before treating either
schedule as a general quality improvement.

## Quanhaol four-step candidate

`quanhaol/Wan2.2-TI2V-5B-Turbo` is a separate full-model checkpoint, not
the FastWan three-step model or a LoRA. The model card links to Kijai's
ComfyUI-compatible FP16 conversion. The evaluated file, pinned at Kijai
revision `0965236c19506478103b9709085db130c2742dbe`, has SHA-256
`fa70f4f760bd97ed435e2a8206f0b459192d9955cd60414e2815ed0c5d147f16`.
Its tensors are a mixture of FP16 and FP32 and are converted to BF16 for the
native graph; it is not the author's original 19,999,421,147-byte PyTorch
checkpoint. The model card does not specify a weight license. The upstream
code repository declares CC BY-NC-SA 4.0; that does not establish separate
permission for the weights. Confirm the applicable terms before use.

The I2V reference uses four model calls with shift 8 and table indices
1000, 750, 500 and 250: sigmas 1, 0.96, 8/9 and 8/11. The published I2V
path narrows the model timesteps through BF16 before converting to integers,
giving 1000, 960, 888 and 728. Its re-noising is also BF16. The native
candidate follows those choices, but uses the existing seeded CPU FP32 noise
generator followed by BF16 conversion rather than CUDA BF16 random draws,
and uses FP32 rather than upstream FP64 for the clean-latent calculation.
This portrait's 526-token Comfy-style context also exceeds the upstream
512-token text limit. Consequently it is not bit-identical to the original
CUDA implementation.

On the same portrait, input size, 526-token prompt and seed as above, the
isolated native probe generated 512×512, 97 frames at 24 fps in **24.17 s**
with Super, or **61.99 s** with the full VAE. The latter includes text,
model loads, inference, video export and diagnostic latent/conditioning
writes. A fresh isolated ComfyUI 0.39.0 MPS request using the same FP16
conversion, full VAE, BF16 model computation, and a diagnostic four-step
sampler completed in **151.35 s**, including export but excluding startup
and hashes. ComfyUI used PyTorch 2.10.0 in an isolated dependency overlay;
the prior FastWan run used PyTorch 2.12.1. Neither path is the original
upstream CUDA pipeline, and the single probe and request are not a speed
distribution or a universal performance guarantee.

Visual inspection showed similar main sword movement between the native and
ComfyUI four-step outputs, with differences in face, clothing and fine
geometry. Their compressed-video YUV SSIM All was **0.9213**; this is an
agreement measure, not a quality score. The four-step and FastWan samples
also differed in expression and sword motion; neither kept the first-frame
identity and hand/sword geometry perfectly. One image and one seed cannot
establish which model is generally better. Local artifacts are under
`outputs/wan22-quanhaol-portrait-*`; no private image or prompt is checked in.

The same four-step portrait was generated at 768×768 with Super in **64.81 s**
in the uncached isolated probe. Its face was more detailed but still changed
relative to the input, and the sword and hands were not consistently stable.
The source image was 512×512, and changing resolution also changes the noise
shape; this is not a controlled estimate of identity improvement.

The packaged desktop `StudioState`/`NativeJobStore` path subsequently ran
the four-step 512×512 request with bundled Super and cross-attention reuse in
**24.21 s**. It verified all 526 text tokens, four actual denoising calls,
97 frames, and automatic reuse of a complete sibling FastWan model directory.
The same packaged path took **64.68 s** at 768×768; a FastWan 512×512
regression job in the same package took **18.36 s**. These are real embedded
desktop jobs, not graphical-window click-throughs or repeated-run medians.

## Cross-attention reuse experiment

The text context is constant across denoising steps, so the native executor
can compute each block's cross-attention key and value once per request.
For the 526-token portrait at 512×512, two four-step paired probe timings
were **22.185 / 22.186 s** for transformer load and denoising without reuse
and **22.046 / 22.044 s** with reuse. One FastWan three-step pair was
**16.634 / 16.556 s**. At 768×768 one four-step pair was **62.365 / 62.245 s**.
The saved final latent tensors were bit-identical for the checked three-step
512 and four-step 512/768 pairs. This small saving is device- and
workload-specific; these are not complete App job medians.

The probe's sampled process footprint increased by approximately 0.18 GB
at 512 and 0.36 GB at 768 in the four-step tests. The cache is consequently
enabled only on devices with at least 64 GiB unified memory; smaller systems
retain the previous execution path. The plan adds 384 MiB of conservative
headroom, and the result receipt identifies whether reuse ran. A separate
two-block evaluation grouping experiment produced no material speedup on
the tested 512 sample and was not enabled.

### Rejected padded cross-attention compression

Short prompts contain many identical zero-padding rows before text projection.
An isolated experiment kept all 53 valid rows but replaced 459 identical
projected padding K/V rows with 16 copies and BF16 additive biases chosen to
approximate their total softmax mass. Float32 additive masks were rejected by
the MLX BF16 SDPA implementation. On captured block-0 K/V and repeated
queries at 6,400 tokens, the 16-copy masked attention had relative L2 error
**0.000201**, with **33,629** BF16 output values changed; warm isolated
attention was roughly **0.75 ms** rather than **3.26 ms** on the sampled
steady iterations. The pre-rounding theoretical mass error was only
**0.00000236**, illustrating that softmax reduction and output rounding still
matter. These are synthetic-query operator results, not production timings.

A complete 512×512, 97-frame FastVideo MLX DiT experiment then used the same
portrait, seed and 50-valid-token conditioning, with 462 padding rows reduced
to 16. Three denoising calls took **17.866 s** without compression and
**17.635 s** with it in single fresh processes, excluding model load, text
encoding, VAE and MP4 export. Despite a theoretical mass error of only
**0.00000137**, the final raw latents had relative L2 difference **0.08554**;
their Super-decoded MP4s had FFmpeg YUV SSIM All **0.97173**, with local
facial differences on inspection. SSIM is not an identity or quality score.
This approximate route was **not enabled** in the native executor or App.
The reference is an MLX DiT/custom-I2V wrapper, not an official end-to-end
FastVideo job; one image and seed do not establish general behavior.

## Short-prompt text alignment and rejected experiments

Wan's UMT5 output masks padded rows to zero. On the tested M4 Max, macOS
26.6.2 and MLX 0.32.0, encoding a short padded prompt using 256 rows and
restoring the remaining zero rows reduced the median text-encoding time from
**0.6925 s to 0.5544 s** across 36 fresh-process cases using the real UMT5
weights. Cases used captured and random vocabulary IDs with 1–256 valid
tokens; all 36 produced bit-identical 512-row BF16 outputs. Four additional
cases with 257–526 valid tokens retained their original shape and output.

For one 50-token portrait and seed, full 512×512/97-frame FastWan and
quanhaol trajectories produced bit-identical final latent tensors with and
without 256-row text encoding; the FastWan MP4s also had compressed YUV
SSIM 1.0. These results do not prove equivalence for every prompt, weight
file, driver, or device. Automatic use is therefore limited to the tested
M4 Max/64-GiB-or-larger, macOS 26.6.2, MLX 0.32.0 combination and only when
the 512-row context has at most 256 valid tokens. Other configurations use
the previous full-length path. Receipts report the actual encoder length and
alignment; a prompt-cache hit performs no text encoding. Long prompts such
as the 526-token portrait above do not benefit.

For the same 50-token portrait, two paired fresh-process FastWan 512×512
probe runs took **18.456 / 18.274 s** without alignment and **18.147 /
18.156 s** with it, including text, model loads, denoising, decoding and
diagnostic exports. Their corresponding text-encoder phases were **0.704 /
0.706 s** versus **0.571 / 0.568 s**. These are single-device probes with
normal run-to-run variation, not guaranteed App or repeated-session speedups.
The packaged desktop job path then completed the same short FastWan request
in **18.168 s** and a repeated-prompt job in **17.121 s**; the second job
reported a prompt-cache hit and zero text-encoder tokens. A packaged
quanhaol job with the same prompt took **23.684 s**. The two FastWan MP4s
and the quanhaol App/probe comparison each had compressed YUV SSIM 1.0.
These are individual embedded desktop jobs, not GUI click-throughs.
A separate 526-token FastWan desktop job completed in **18.347 s** and
reported alignment 0 and all 526 encoder rows, confirming the long-prompt
fallback on this example.

More aggressive 64- and 128-row encodings were rejected: one 53-token text
fixture differed by relative L2 **0.000346**, and a 50-token FastWan portrait
then differed by **0.09375** relative L2 in final latents (compressed-video
SSIM **0.9700**). These are agreement statistics, not perceptual quality
scores, but the amplification makes that route inappropriate as a default.

At 512×512/97 frames, a synchronized first-block profile with real FastWan
weights and tiled small-fixture activations measured an untraced block median
of **0.181 s**. Individual traced phases included about **0.0375 s** for the
FFN up projection, **0.0380 s** for FFN down, and **0.0442 s** for self
attention including RoPE. Trace points synchronize the GPU and disable
cross-KV reuse, so those phase values are diagnostic rather than an exact
end-to-end breakdown. Fusing Q/K/V weights gave identical tested latents but
did not improve the four-step load-plus-denoise time (about **22.11 s**
versus **22.05 s**); it was not enabled.

In separate warm isolated GPU matrix tests at 6,400 rows, the real BF16 FFN
up projection took a median **37.5 ms**, versus **49.0 ms** for MLX affine
W8 and **48.7 ms** for W4. W8 and W4 output relative L2 were approximately
**0.0059** and **0.0952**, respectively. At 14,400 rows the same BF16/W8/W4
projection took **84.2/110.0/109.3 ms**. Quantization was excluded from
timing, input activations were synthetic, and none of these is an end-to-end
video-quality test. These W8/W4 routes were not enabled.
An isolated FP16 version of the same 6,400-row projection was also about
**37.6 ms**, versus **37.5 ms** BF16, with relative L2 **0.00168** against
the BF16 result; a precision switch was not enabled.

## Dense attention / fused interleaved RoPE

On a 512×512/97-frame portrait probe, the three-step Transformer phase broke
down into **0.233 s** weight loading, **0.074 s** text/KV preparation, and
**16.293 s** sampling; the four-step path took **0.336 / 0.057 / 21.661 s**.
The combined phase also includes object cleanup. These are single warm-file
cache observations, not cold-start guarantees. Retaining roughly 10 GB of
Transformer weights between jobs was not enabled for this small loading cost.
Isolated RoPE-frequency construction took about **0.32 ms** at 512 and
**0.38 ms** at 768, so caching those frequency tables was not pursued.

The native GPU candidate combines interleaved Q/K RoPE and Q/K/V head-major
packing into one Metal kernel, retaining dense attention and FP32 arithmetic
boundaries. In synthetic warm tests, RoPE plus SDPA took approximately
**44.2 ms → 38.2 ms** at 6,400 tokens and **207.7 ms → 191.3 ms** at
14,400 tokens, with identical BF16 outputs on those fixtures. These are
operator timings, not end-to-end speedups.

Two four-step 768×768/97-frame portrait probe pairs measured sampling at
**61.861 / 61.853 s** without the kernel and **59.902 / 59.915 s** with it.
The corresponding final latent tensors were bit-identical in both pairs. A
four-step 512×768 pair measured **35.984 → 34.848 s**, also with identical
latents; a four-step 512×512 pair measured **21.661 → 20.983 s**, with
identical latents. One three-step 512×512 pair measured **16.239 → 15.730 s**, with
identical latents. One three-step 768×768 pair measured **46.380 → 44.939 s**,
also with identical latents. A 544×544 pair (7,225 tokens, including an SDPA
tail) measured **18.998 → 18.444 s**, with identical latents. One small
160×224/9-frame pair was slower (**0.457 →
0.502 s** sampling), even though its latents matched. These measurements
use a single device, portrait/fox inputs, and limited seeds; they do not prove
identity across all inputs, drivers or hardware.

Automatic use is restricted to the tested M4 Max with at least 64 GiB,
macOS 26.6.2 and MLX 0.32.0, for 97-frame canvases with both dimensions
between 512 and 768 and divisible by 32. All other routes use the prior RoPE
path. The session receipt identifies the selected path and breaks down
Transformer loading, context preparation and sampling time.

The packaged desktop task path completed the long-prompt three-step
512×512 job in **17.911 s** and a repeated-prompt job in **16.629 s**; the
four-step 512×512 job took **23.215 s** with Super, and the four-step 768×768
job took **62.680 s**. A packaged 160×224/9-frame job
reported the original RoPE path and completed successfully. These were
individual embedded desktop jobs, not graphical-window click-throughs.
Against their previous non-fused App outputs, the three-step 512 and
four-step 768 compressed videos both had YUV SSIM 1.0 on these samples.
The four-step short-prompt 512 App output also had YUV SSIM 1.0 against
the non-fused App output for the checked seed and input.
The packaged four-step 512×512 job with the full VAE took **60.328 s**;
the earlier isolated ComfyUI full-VAE request took **151.35 s**. Their
compressed-video YUV SSIM All was **0.9213**. The implementations, sampler
wrappers, MP4 encoders and measurement scopes differ, so these figures are
not a same-kernel benchmark or a general quality score.
Both model variants also passed cancellation during the large Transformer
route and existing-output protection checks without publishing a partial MP4.
