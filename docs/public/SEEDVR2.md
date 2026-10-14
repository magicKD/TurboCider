# SeedVR2 image and video super-resolution

SeedVR2 3B is available through the native App, CLI and existing C / Swift
SDK request interfaces. This integration is still being qualified. Pure GPU
image/video inference works; an optional manual GPU/Core ML CPU+NE prefix
route is available for the exact qualified original-FP16 compiled bank.
The quality and speed results below are workload/profile-specific. The stable
FP16/FP32 profile passes App image/video regression and the matched full-request
SDPA/GPU-only image/video speed cohorts at 512 and 1024 outputs. The manual
hybrid passes four-media App/component checks and complete-request GPU/hybrid
ABBA speed cohorts. The input-lifetime library also passes the complete
1024/t25 compatible Comfy default-attention control. Physical ANE occupancy
and broader/new-candidate workload qualification remain unmeasured/pending.

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
frame count/FPS. GPU is the default execution route; unqualified ANE
manifests, LoRA, graph-compilation and streaming options fail explicitly.

To package an already tested isolated native build without replacing a running
App, set both output directories explicitly. The targeted App build includes
the model-library helper as well as the complete UI; it only narrows the set of
integration-test executables built, not the App's model support:

```sh
TURBOCIDER_BUILD_OUTPUT_DIR=build/seedvr2-owned-input-native-v1 \
  TURBOCIDER_APP_TARGETS=seedvr2 bash tools/native/build_app.sh
TURBOCIDER_BUILD_OUTPUT_DIR=build/seedvr2-owned-input-native-v1 \
  TURBOCIDER_PACKAGE_OUTPUT_DIR=dist/seedvr2-20261014 bash tools/native/package.sh
open dist/seedvr2-20261014/TurboCider.app
```

Close the old App before using the new bundle with the same default state
directory. Packaging alone does not migrate or overwrite user drafts, models
or output media. Omit `TURBOCIDER_PACKAGE_OUTPUT_DIR` to use the usual `dist/`
delivery location only when it is safe to replace that App and CLI.

For manual hybrid execution, enable “额外启用 ANE” and select the already
compiled SeedVR2 prefix manifest. CLI/SDK schema 1 uses `execution="gpu_ane"`,
`allow_approximation=true` and `ane_manifest`; schema 2 places `policy`,
`allow_approximation` and `ane_manifest` inside `execution`. Approximation
consent is required because prefix activation/partial-down arithmetic differs
from the stable GPU recipe. `auto` remains GPU-only, and GPU/auto requests
carrying a prefix manifest are rejected, not silently rerouted.

The only public qualified manifest SHA-256 is
`109dde47f82b7acb3a6d8cbba16b9d42e9f15fddae539d64141a4ec0ac8a8f76`.
It references all 32 compiled video FFNs, fixed 512 rows, 2560 input/output
channels, and 2048 of 6912 intermediate channels. The runtime checks every
compiled file/tree hash before constructing Core ML models and checks source
stamps around requests. Recompiling on another host/toolchain can produce a
different identity; such banks require separate qualification and are not
automatically accepted. `tools/coreml/export_seedvr2_ffn.py` is an offline
experiment/export tool, not an automatic App bank installer. Missing-bank
errors leave the default GPU path available.

Core ML uses CPU+NE with GPU excluded; GPU simultaneously executes the other
4864 FFN channels. Attention, text FFN and VAE stay GPU. The FP16 prefix/down
result and GPU suffix are joined in FP32. Results explicitly identify
`seedvr2-original-fp16-coreml-prefix-v1`, actual prefix calls, all-block coverage,
bank identity and `physical_ane_occupancy_measured=false`. The independent
caller-owned output pool has at most 64 slots / 256 MiB and is reused only
after a completed separately owned join. Larger videos use bounded row groups;
this preserves all tokens but does not qualify 121-frame real-media quality.
Hybrid staged mode retains the Core ML bank in addition to VAE parameters;
`resident_parameter_bytes` covers MLX parameters only, not the entire bank or
process.

The tested optional low-active-memory policy is `component_staged`. The App's
“模型驻留” selector offers “常驻（速度优先）” (the default) and
“分阶段（降低活跃内存）”. In schema 1 set `residency`; in schema 2 set
`execution.residency`. This is stage-wise parameter release/reload, not spatial
VAE tiling or transformer weight streaming. DiT is absent during VAE encode
and decode, then reloaded from the same original checkpoint for sampling;
the VAE remains available between requests. It adds loading/synchronization
cost and preserves the stable numerical recipe.

Balanced component ABBA checks at 1024 output record **16.47 -> 9.69 GB**
active peak for images and **21.28 -> 14.49 GB** for complete 25-frame videos,
with identical saved tensors. Including weight management, image warm median
is **3.398 -> 3.787 s** (about 11.45% slower), video **84.499 -> 84.971 s**
(about 0.56% slower). These are component, not full-request/App timings.
Real App image/video and resident/staged/resident transitions also pass; PNG
bytes and every decoded video frame match. The App's 1024/t25 observation is
**14.80 GB active / 18.47 GB sampled footprint**, versus resident
**21.60 / 24.73 GB**. Sampled peaks are lower bounds, not enforced budgets.
Conservative planning estimates are deliberately unchanged; selecting staged
does not grant a lower hard memory-budget guarantee.

The default VAE uses safe singleton-frame 2D convolutions, fused Metal
per-frame GroupNorm, FP16 activations, bounded full spatial attention, and
owned compact causal histories for videos. Replicated initial histories share
one owned frame, and 1x1x1 shortcuts use flattened FP16 matrix projections with
FP32 accumulation. Single images do not retain unused histories. The native
allocator keeps up to 4 GiB of
temporary buffers during a request and clears unused buffers on exit; loaded
model weights remain resident under the default policy. Staged mode retains
only VAE parameters after restoration. Reported `mlx_peak_bytes` is the current
request's active MLX peak, not the whole process's or Mac's memory usage.
Results also report process physical-footprint and Metal-allocation sampled
peaks at a nominal 20 ms interval. Those figures overlap and must not be added;
sampling may miss short-lived allocations and is not a hard memory bound.
Actual results declare `weight_residency_recipe`, `resident_components`,
`resident_parameter_bytes`, DiT/VAE reuse, and `timings_seconds.weight_management`
plus `restoration_total`. Reload cost is included in the overall request time.

After media normalization video releases the resized NHWC source before VAE
inference and retains only normalized RGB. Staged images keep an independently
owned alpha plane with unchanged export rounding; resident images preserve
the original faster RGBA-backed alpha view. Telemetry identifies
`media_input_retention_recipe`, `media_retained_alpha_bytes` and
`media_retained_source_rgba_bytes`; alpha-view logical bytes must not be
mistaken for the larger retained RGBA backing.
This shortens source-buffer residency without changing the VAE, DiT, resize
or sampling numerical recipe; it does not imply a smaller enforced memory
budget.
The revised App's hybrid 1024/t25 active peaks are about **21.43 GB resident /
14.63 GB staged**, with all exported pixels/frames identical to the preceding
build. This is about 314.6 MB less active memory per video policy, not a
demonstrated universal physical-footprint improvement. Resident images keep
their original implementation; a six-warm-sample before/after 1024 GPU image
check records 3.500405/3.500845 s. Its strict non-regression Boolean is false;
this input-lifetime feature is not presented as an image speed win.

Videos and staged images now also transfer normalized RGB ownership into
restoration and release it after completed, independently owned FP16 VAE
conditioning. Noise, alpha and restored outputs remain available; resident
images keep the original non-owning path. Results declare
`normalized_input_release_recipe`, `normalized_input_released_logical_bytes`
and `timings_seconds.input_release`. Logical released bytes are not a physical
memory guarantee. Four-media components and all real App export comparisons
remain bitwise exact. The subsequent App 1024/t25 hybrid observations are
about **21.11 GB resident / 14.32 GB staged** active peaks, another approximately
314.6 MB reduction. These are App observations, not new paired speed evidence;
the preceding Comfy speed cohort remains bound to its own library SHA.

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
new-profile video paired checks and GPU+ANE qualification remain separate
requirements for this historical image-only cohort.

The new-profile SDPA/GPU-only image cohorts also pass: **0.828 vs 3.978 s**
for 256->512 and **3.504 vs 15.636 s** for 512->1024, six complete warm requests
per route. Default controls remain recorded alongside these results.
The stable 256->512 / full t25 video SDPA/GPU-only ABBA cohort also passes:
**20.60 vs 71.34 s**, about **3.46x**, with every frame/FPS verified and the
non-CUDA cache-transfer compatibility guard disclosed. The stable full
512->1024 / t25 / 24-FPS video SDPA/GPU-only ABBA cohort also passes:
**84.879 vs 298.901 s**, about **3.52x**, six complete warm requests per route.
All 25 frames, dimensions and FPS are verified; the same MPS cache-transfer
guard is disclosed. The new staged-capable resident GPU library also passes
the 512/t25 compatible default-attention video control: **20.586 vs 71.190 s**,
about **3.46x**. The later input-lifetime library's complete 1024/t25 compatible
default-attention ABBA passes too: **85.010773 vs 295.438371 s**, about
**3.47531x**, six genuine warm requests per route and all frames/FPS retained.
This binds library SHA `596c160aabcc7ec248c6e842e86da1b2b27a0338bab41749014dff866a35d7e4`,
not subsequent experimental source or a physical ANE-placement claim. The
same disclosed non-CUDA cache-transfer compatibility guard remains in use.

The original-FP16 GPU/CPU+NE prefix route passes the unchanged
512/1024 image and complete 25-frame video tensor gates against both stable
GPU and independent original-parameter FP32 controls, using all 32 video FFN
prefixes. A new independent-output-backing schedule avoids the prior mutable
output/copy problem and the row-chunk image regression: 1024 warm component
denoise is about **1.727 vs 1.838 s**. These component observations are not
whole-request speed evidence. Separate public-candidate ABBA cohorts use six
genuine warm requests per route and include import, resize, VAE, sampling and
media export:

| Workload | GPU / CPU+NE hybrid warm median | Ratio GPU/hybrid |
| --- | ---: | ---: |
| Image 256->512 | 0.828580 / 0.772848 s | 1.07211x |
| Image 512->1024 | 3.500113 / 3.385148 s | 1.03396x |
| Video 256->512, all t25 | 20.576623 / 20.282359 s | 1.01451x |
| Video 512->1024, all t25 | 84.951658 / 83.104532 s | 1.02223x |

These records bind candidate library SHA `2fdc400244fa4163465c2bdb38ad04ee8d495d4fd30062e0d4c501db58e2a4d7`.
The subsequent metadata-only build uses a separate identity; these timings
are not relabeled as new-binary paired results. The modest gains are
hardware/workload-specific. A synthetic CPU-only prefix control is also
faster than GPU-only, so the entire improvement cannot be credited to NE.
Physical occupancy requires separate hardware evidence; local `powermetrics`
sampling needs administrator permission and was unavailable. No whole-process
memory improvement is claimed for hybrid resident: its backing pool adds
active memory. Failed/slow earlier schedules remain documented.

Actual App GPU, hybrid resident and hybrid staged workflows each pass seven
requests: all four media/size cases, resident/staged/resident transitions,
history persistence and unload. Hybrid resident/staged PNG files and all 25
decoded frames at both sizes match bitwise; MP4 container hashes differ.
GPU-vs-hybrid exports are not identical. On the App witnesses, image RGBA
PSNR is about 52.42/53.81 dB and decoded-video RGB PSNR about 33.43/38.68 dB at
512/1024. These are diagnostic export statistics (video includes codec
effects), not substitutes for the unchanged shared-noise tensor gates.

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

## Historical observations (older builds and numerical recipes)

The following chronology is retained for reproducibility; it is not the
current capability/acceptance summary above.

The first image-only full-request precheck records two warm observations per
route: original ComfyUI v0.39 MPS/SDPA **15.639 s**, native GPU **5.913 s**,
about **2.65×**. This was not an ABBA paired cohort; both reports intentionally
retain `qualification_passed=false`. Later workload-specific speed cohorts are
reported above; physical ANE occupancy and broader workloads remain separate
qualification boundaries.

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
GPU+ANE qualification. At that stage the public route remained GPU-only.

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
