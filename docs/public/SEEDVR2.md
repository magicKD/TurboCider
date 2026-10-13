# SeedVR2 image and video super-resolution

SeedVR2 3B is available through the native App, CLI and existing C / Swift
SDK request interfaces. This integration is still being qualified. Pure GPU
image/video inference works; GPU+ANE is not yet available in the public route.
The quality and speed results below are workload/profile-specific. The stable
FP16/FP32 profile passes App image/video regression; its full-request speed
cohorts are being re-run. GPU+ANE qualification remains pending.

## Prepare weights

```sh
.venv/bin/python tools/models/download_seedvr2.py
```

This downloads the requested original ModelScope
`Comfy-Org/SeedVR2` FP16 DiT and its matching VAE. Byte counts and SHA-256s are
checked before final filenames are published. The directory is:

```text
Comfy-Org-SeedVR2/
  diffusion_models/seedvr2_3b_fp16.safetensors
  vae/seedvr2_ema_vae_fp16.safetensors
  source.json
```

No text encoder is required: the checkpoint includes its conditioning.
Original FP16 checkpoint coefficients and DiT projections are retained without
BF16 coefficient rounding or weight quantization. DiT residuals, normalization,
time embedding, nonlinear arithmetic and final velocity use FP32. Attention
and the causal VAE use FP16; the one-step Euler update uses FP32. This stable
profile prevents the late text/residual overflow of all-FP16 state while
remaining close to original-parameter FP32 computation. All model inference is
native C++/MLX on Metal; Python is only used by
offline preparation and validation tools.

Results identify `numerical_recipe=seedvr2-stable-fp16-fp32-v1` and each effective
precision. Public sessions reject unqualified DiT recipe overrides; older BF16
diagnostics remain offline and are not claimed equivalent or qualified.

## Run

After building, edit the source/output paths in the example JSON and run:

```sh
build/native/turbocider generate models/Comfy-Org-SeedVR2 \
  examples/requests/seedvr2-image-upscale.json
build/native/turbocider generate models/Comfy-Org-SeedVR2 \
  examples/requests/seedvr2-video-upscale.json
```

Both use model ID `seedvr2-3b`, `steps=1`, and GPU execution.
`image.upscale` requires one `image` input with role `source`; `video.upscale`
requires one `video` input with that role. No prompt is accepted. Schema 2
output kind follows the operation (`image` or `video`).

Width and height specify the output canvas; they must be multiples of sixteen.
Use 512×512 to restore a smaller square source to 512, or 1024×1024 to enlarge
a 512×512 square source 2×. The input is resized with GPU bicubic interpolation
before the model. Non-square sources are resized to the explicit canvas, not
automatically center-cropped. Image alpha is resized and reapplied to PNG.

For videos, set `frames` to the **complete source frame count** and `fps` to
its integer constant frame rate. Current support is 2–121 frames, 1–120 FPS.
The importer rejects mismatches, variable frame rates and implicit truncation.
The VAE's internal 4n+1 padding is removed before export, preserving the source
frame count. Current video output is silent H.264 MP4; soundtracks are not
preserved. Image output is PNG.

The App exposes “图像超分” and “视频超分”. Choose the SeedVR2 model, register
its directory, then add a source image or video. Video import fills the source
frame count/FPS. GPU is the available execution route; unqualified ANE
manifests, LoRA, graph-compilation and streaming options fail explicitly.

The default VAE uses safe singleton-frame 2D convolutions, fused Metal
per-frame GroupNorm, FP16 activations, bounded full spatial attention, and
owned compact causal histories for videos. Replicated initial histories share
one owned frame, and 1x1x1 shortcuts use flattened FP16 matrix projections with
FP32 accumulation. Single images do not retain unused histories. The native
allocator keeps up to 4 GiB of
temporary buffers during a request and clears unused buffers on exit; loaded
model weights remain resident. Reported `mlx_peak_bytes` is the current
request's active MLX peak, not the whole process's or Mac's memory usage.
Results also report process physical-footprint and Metal-allocation sampled
peaks at a nominal 20 ms interval. Those figures overlap and must not be added;
sampling may miss short-lived allocations and is not a hard memory bound.

For diagnostics, `TURBOCIDER_SEEDVR2_VAE_CONV_RECIPE=conv3d` selects the
lower-active-peak/slower 3D image route. `singleton2d` is the tested default;
`taps2d` remains an experimental temporal decomposition. Set
`TURBOCIDER_SEEDVR2_ALLOCATOR_CACHE_BYTES` to an integer byte count in
0..17179869184 to change temporary cache retention. Smaller limits may slow
inference; these settings are not a hard memory budget.

`TURBOCIDER_SEEDVR2_VAE_NORM_RECIPE=mlx` selects the pre-fusion reference
normalization; `metal` is the validated default. Experimental `tiled3d`
convolution reads causal histories virtually and bounds column workspaces to
two tiles of at most 256 MiB each. It preserves full-frame normalization and
attention, but currently trades speed for lower active memory, so is not the
default. Its `TURBOCIDER_SEEDVR2_VAE_CONV_TILE_ROWS` limit accepts 128-aligned
integers from 128 to 65536 (default 16384).

`mixed3d` is an optional middle ground for 1024-or-larger videos: only the
largest noninitial temporal convolution uses the bounded stencil. Smaller
operators and initial singleton convolutions stay on the faster route. The
1024/t9 codec screen observes about 13.31 GB active peak and 26.5 s, versus
about 31.4 s for full `tiled3d`. This is a component observation, not a complete
media-request speed gate; the default remains `singleton2d`.
The optional mode also passes the App's complete image/video checks. Its
1024/t25 video observes **20.70 GB active / 86.55 s**, versus the default's
**21.59 GB / 83.32 s**. This trades speed for active-memory reduction; sampled
whole-process footprint is not demonstrably lower.

For a pre-change control, set
`TURBOCIDER_SEEDVR2_VAE_HISTORY_RECIPE=concat` and
`TURBOCIDER_SEEDVR2_VAE_POINTWISE_RECIPE=conv`; their new defaults are `compact`
and `matmul`. Actual recipes are included in result telemetry. The history
selector controls the MLX halo route; bounded-stencil operators always retain
their causal tails directly. A reusable VAE-only ABBA observer is
`tools/validation/seedvr2_vae_cohort.py`; run it with
a validation Python environment containing NumPy and safetensors.
Its 1024/t9 default-route cohort records **25.50 -> 25.20 s** warm medians and
**14.33 -> 14.20 GB** active peaks, six warm observations per route, with
bitwise-identical latents and decoded tensors. This is a roughly 1.2% codec
speed improvement, not an equally sized whole-model speed guarantee.

## Validation boundary

The stable FP16/FP32 candidate passes the unchanged cosine >=0.999 /
relative-L2 <=0.05 gates against the same original checkpoint's independently
configured FP32 reference: all saved embeddings, blocks, velocity, sampling,
VAE condition and decoded pixels pass for 512/1024 real images and complete
512/1024, 25-frame LTX-derived videos. Pixel relative-L2 is 0.00101–0.00176
across those four witnesses. This is original-parameter accuracy, not bitwise
equivalence to Comfy's BF16 recipe; the old BF16 equivalence failures remain
recorded. Full-request speed ratios from the previous BF16 profile below are
historical and must not be relabeled as new-profile speed evidence.

The current App build passes all four real generations and validates executed
precision against the plan. Its 1024/t25 video observes 84.69 s, 21.60 GB active
MLX peak and 24.87 GB sampled process footprint, with every frame/FPS retained.
Those are single App observations, not paired speed or hard peak guarantees.

Fresh stable-profile image ABBA cohorts pass against the explicitly recorded
Comfy default-attention/ordinary-residency observer (FP16 VAE): **0.827 vs
3.985 s** at 256->512 and **3.506 vs 15.535 s** at 512->1024, six full warm
requests per route. These are about **4.82x / 4.43x** for those workloads;
new-profile video and SDPA/GPU-only paired checks remain separate requirements.

On the verified M4 Max / 64 GiB host, the native CLI generated a complete
512→1024 image and a complete 512→1024, 25-frame / 24-FPS LTX-derived video.
The MP4 frame count, FPS and dimensions were independently checked. The App's
source staging, request kinds, prompt suppression, frame/FPS binding and draft
persistence have a dedicated integration test. It now also runs real App
generation and validates 512/1024 image and full 25-frame video output,
strict PNG publication, history persistence and unload. The graphical App's
isolated-state startup was separately checked; this is not an automated
manual-click UI test.

```sh
TURBOCIDER_APP_TARGETS=seedvr2 bash tools/native/build_app.sh
build/native/turbocider-seedvr2-app-tests SOURCE_512_T25_FPS24.mp4 \
  outputs/seedvr2-app-new models/Comfy-Org-SeedVR2 SOURCE_512.png
```

The first image-only full-request precheck records two warm observations per
route: original ComfyUI v0.39 MPS/SDPA **15.639 s**, native GPU **5.913 s**,
about **2.65×**. This was not an ABBA paired cohort; both reports intentionally
retain `qualification_passed=false`. Later workload-specific speed cohorts are
reported below; full quality and GPU+ANE qualification remain pending.

After VAE optimization, the updated native 512->1024 image precheck is
**3.913 s** warm median, with **17.8 GB** active MLX peak versus the earlier
**29.9 GB**. Real App video 512->1024 t25 / 24 FPS passes all frame/geometry
checks at **24.8 GB** active peak, versus an earlier **37.4 GB** observation.
These before/after records are not a paired complete ComfyUI qualification.
The BF16 DiT's strict generic latent/intermediate gate remains pending; VAE
parity at 128/512/1024 passes independently.

The subsequent fused-normalization precheck records **3.359 s** warm median
for 512->1024 images and **16.5 GB** active MLX peak. The real App's complete
25-frame / 24-FPS 1024 video observes **84.163 s**, **21.7 GB** active peak,
and **25.3 GB** sampled process physical footprint. These are observations,
not a final balanced ComfyUI image/video qualification. A separate ABBA
observer, `tools/validation/seedvr2_paired_cohort.py`, now records two cohorts
per route with at least three warm requests each and unchanged source/build/
reference identities; its speed gate does not imply the overall quality/ANE
goal is complete.

The fresh balanced image ABBA cohorts now pass their workload-specific speed
gate against pinned stock ComfyUI MPS SDPA/GPU-only: 256->512 is **0.820 vs
3.974 s**, and 512->1024 is **3.369 vs 15.632 s**, six warm requests per route
per workload. These are complete import/encode/sample/decode/export timings.
They do not imply completed default-Comfy, shared-noise quality or
GPU+ANE qualification. The public route remains GPU-only.

The complete 256->512, 25-frame / 24-FPS video ABBA cohort also passes its
speed gate: **20.44 s native vs 71.33 s MPS-compatible Comfy SDPA/GPU-only**,
about **3.49x**, six warm requests per route. All frames/FPS/dimensions are
verified, with the CUDA-event compatibility patch explicitly disclosed.
Default-Comfy controls and full quality/ANE qualification remain separate
requirements.

The complete 512->1024 / t25 / 24-FPS video ABBA cohort now also passes:
**84.08 s native vs 298.96 s MPS-compatible Comfy SDPA/GPU-only**, about
**3.56x**, six warm requests per route. Every frame, FPS and output dimension
is verified. All four stated SDPA/GPU-only speed workloads pass, but full
quality/default-reference controls and GPU/ANE qualification remain pending.

The subsequent compact-history/pointwise version passes the real App's four
image/video generation checks again. Its PNGs are byte-identical to the
preceding version; the 1024/t25 video observes 83.32 s and 21.59 GB active MLX
peak (previously 21.73 GB). This is an App observation, not a new paired Comfy
measurement; image active peaks remain unchanged. Selective 16-layer CPU+NE
experiments pass shared-noise image-component gates at 512 and 1024 against
GPU, but are not enabled in the public App. Physical ANE occupancy, complete
media-request performance and real-video qualification remain pending.

The pinned stock ComfyUI video's VAE currently reaches a CUDA-event-only cache
offload path on MPS and fails. An independently owned benchmark copy disables
only that CUDA transfer policy on non-CUDA devices, preserving arithmetic and
the stock workflow. The original failure must be reported separately from
the compatible reference; a crash is not a speed baseline. The user's ComfyUI
installation and running server are not modified.

See [implementation and acceptance evidence](../status/seedvr2-integration.md)
for intermediate parity limits, measurements and remaining requirements.
