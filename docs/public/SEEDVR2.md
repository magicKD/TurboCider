# SeedVR2 image and video super-resolution

SeedVR2 3B is available through the native App, CLI and existing C / Swift
SDK request interfaces. This integration is still being qualified. Pure GPU
image/video inference works; GPU+ANE is not yet available. Do not interpret the
image speed precheck as a completed paired image/video benchmark.

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
Original FP16 checkpoint weights are used without quantization. DiT projection
math uses BF16, the causal VAE uses FP16, and the one-step Euler update uses
FP32. All model inference is native C++/MLX on Metal; Python is only used by
offline preparation and validation tools.

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
causal histories for videos. Single images
do not retain unused histories. The native allocator keeps up to 4 GiB of
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

## Validation boundary

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
about **2.65×**. This is not an ABBA paired cohort; both reports intentionally
retain `qualification_passed=false`. Final image/video speed and GPU+ANE
qualification are still pending.

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
They do not imply completed video, default-Comfy, shared-noise quality or
GPU+ANE qualification. The public route remains GPU-only.

The pinned stock ComfyUI video's VAE currently reaches a CUDA-event-only cache
offload path on MPS and fails. An independently owned benchmark copy disables
only that CUDA transfer policy on non-CUDA devices, preserving arithmetic and
the stock workflow. The original failure must be reported separately from
the compatible reference; a crash is not a speed baseline. The user's ComfyUI
installation and running server are not modified.

See [implementation and acceptance evidence](../status/seedvr2-integration.md)
for intermediate parity limits, measurements and remaining requirements.
