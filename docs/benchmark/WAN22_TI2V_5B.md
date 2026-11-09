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
