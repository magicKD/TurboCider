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

### App: 超分 workspace

1. Open **模型 → 超分 → SeedVR2**, and register the existing model directory
   containing both DiT and VAE. SeedVR2 also appears in the image/video filters.
   Registration retains the original files; no download starts automatically.
2. Click **超分** in the sidebar. Use **图片 / 视频** to select the source
   kind, then choose the model at the top. SeedVR2 supports both kinds;
   Real-ESRGAN x2plus/x4plus are image-only and are not offered for videos.
   Select SeedVR2 and add one source image or video to use this native engine.
3. Choose **2× / 4× / 长边 512 / 长边 1024**. These source-based shortcuts
   preserve aspect ratio up to the native 16-pixel alignment and reject sizes
   beyond the current 2048-pixel per-axis bound. Explicit width/height remain
   available in the right inspector. Start with 512 or 1024 for testing.
4. Click **开始图像超分 / 开始视频超分**. Every request is one restoration
   step, without a prompt. The source is preserved; PNG/silent MP4 outputs
   appear in the shared history with progress and cancellation controls.

Video import fills and locks the complete source frame count/FPS; output-size
shortcuts do not truncate frames or change the frame rate. The existing 2–121
frame, integer-CFR and silent-output limits still apply. **速度优先** explicitly
selects resident weights, full spatial computation, buffered I/O, frame history
and a 4-GiB allocator cache. **省内存** selects staged weights, streaming video
I/O, frame history and a 1-GiB cache, retaining explicit spatial-row choices.
The current effective settings are visible. Neither shortcut lowers output
resolution or drops frames; the memory option can be slower and is not a hard
memory guarantee. No new speed/approximation qualification is implied.
Detailed VAE controls are under **模型 → SeedVR2 → 加速与编译缓存**; residency
can also be selected in the creation inspector's advanced settings.

Running SeedVR2 requests show five stages: preparation/import, VAE encode,
model restoration, VAE decode and save. Encode/decode count completed source
frames across the whole clip, not a repeatedly resetting 0–5 slice counter.
Restoration counts all 32 transformer blocks. Elapsed time updates once per
second; encode/decode remaining time is a current-stage estimate after at least
five completed frames, not a guaranteed whole-request ETA. Completed request
details report VAE encode, restoration and VAE decode seconds separately.

GPU is the default execution route; unqualified ANE manifests, LoRA,
graph-compilation and streaming options fail explicitly. Ordinary creation,
Playground, Real-ESRGAN and existing model paths remain available.

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

## Request-local VAE controls (active optimization candidate)

The streaming/configuration candidate adds a `seedvr2_vae` object at the root
of schema 1 requests or inside schema 2 `execution`. Buffered input and output
remain default. Example schema 2 execution:

```json
{
  "policy": "gpu",
  "residency": "component_staged",
  "seedvr2_vae": {
    "encoder": "streaming",
    "decoder": "streaming",
    "convolution": "singleton2d",
    "spatial": "full",
    "encoder_spatial": "full",
    "spatial_rows": 64,
    "spatial_disk_budget_bytes": 8589934592,
    "tile_rows": 16384,
    "allocator_cache_bytes": 1073741824
  }
}
```

The optional `history="frames"` policy retains independent single-frame
causal owners and shares known materialized singleton normalization outputs;
`history="compact"` remains the default, including older App drafts. All
temporal operands and phases are preserved. The App exposes this separately
from allocator-cache retention. A 1024/t9 VAE-only ABBA observes about 1.77%
faster encode+decode and 939.5 MB lower active peak, with byte-identical latents
and decoded tensors. The subsequently completed 1024/t25 complete-request ABBA
records 85.119 -> 83.793 s (about 1.58% faster) and 939.5 MB lower active maximum;
all eight export pairs retain every pixel. These are not hard RAM budgets.

An additional opt-in `share_upscale_history=true` requires `history=frames`,
`convolution=singleton2d` and `spatial=full`. It retains the completed independent
spatial-upscale frame as history instead of copying it again. Omission/older
drafts keep it **false**; images create no unused history. Both request schemas,
the Swift SDK and App expose this field. Unsupported combinations fail explicitly.

Complete1024/t25 decoder words and each audited-FP32 frame pass unchanged.
The isolated upsampling-region peak falls512 MiB, but the maximum moves to the
following conv: complete decoder reduction is only133.7 MB. Matched same-build
single staged App requests observe13.148 ->13.015 GB active maximum and
90.375/90.053 s, not a paired warm-speed or physical hard-budget qualification.
All512/1024 image/video exports match the disabled route exactly. Result fields
`vae_shared_upscale_frames` and `vae_shared_upscale_logical_bytes` describe
actual padded frames/avoided logical copies, **not** physical or peak savings.
The App's user-applied memory advice includes streaming video input/output,
staged residency,1-GiB cache and supported sharing without changing explicit
row/workspace choices. Conservative admission and the121-frame cap are unchanged.
See `docs/design/seedvr2-vae-phase-and-upscale-history.md` for scopes and receipts.
The independent signed App is
`dist/seedvr2-upscale-history-20261015/TurboCider.app` (macOS26.2+). It does not
replace the currently running App; quit an older instance before opening the
new bundle against the same default state directory.

`encoder` accepts `buffered` or `streaming`. Streaming input is video-only and
rejects complete input tensor dumps. It reads one RGB24 frame at a time and
retains the original causal initial-five/subsequent-four encoder partition;
4n+1 padding still repeats only the final source frame. EOF, full frame count,
FPS/PTS and source-file stamps must verify before sampling, so an undeclared
tail cannot be silently dropped. Omission, including older saved drafts, keeps
buffered input. Image requests in the App adapt both source/sink controls to
buffered. The native API rejects explicitly requested image streaming modes.

Actual reader/FP32-normalization/latent and two seven-request App chains pass
bitwise export checks. A same-library 512/t25 staged/cache1 full-request ABBA
records 21.254 / 21.252 s, effectively equal, with a negligible 37-KB active
maximum difference. This is a bounded-source-residency option, not a proven
whole-request speed or maximum-memory improvement. In a 512-source movie, the
observed caller-owned reused RGB24 buffer is 786,432 bytes; codec allocations,
encoder chunks, causal state and model storage are not included in that number.

`decoder` accepts `buffered` or `streaming`; streaming is video-only and
cannot also request complete raw-pixel tensor dumps. `convolution` accepts
`singleton2d`, `conv3d`, `mixed3d` or `tiled3d`. Tile rows must be 128-aligned
in 128...65536, and allocator-cache retention accepts 0...16 GiB. Explicit
request controls select the convolution/tile/cache policy without changing
process environment or reloading VAE weights. Omitting the object preserves
the existing default/diagnostic environment behavior.

The separately built spatial candidate adds `spatial="full"|"file_rows"`.
`full` remains the default, including older saved drafts. `file_rows` schedules
eligible high-resolution decoder norm/convolution stages in row bands, using
complete-frame statistics, original causal taps/halos and all spatial attention
keys. It is not independent complete-VAE crops, encoder spatial scheduling or
a hard memory budget. Tile-shaped convolution changes rounding: outputs are
numerically qualified, **not bitwise equal** to the full route.

`spatial_rows` must be 8-aligned in 8...128 (default 64), independently of the
convolution-workspace `tile_rows`. `spatial_disk_budget_bytes` is a positive
live-scratch-file limit at most 8 GiB (default 8 GiB). Public file-row requests
require both canvas axes <=1024 and convolution `singleton2d`/`conv3d`; incompatible
`mixed3d`/`tiled3d` combinations fail rather than silently falling back. Images
do not write causal-history files. Video file histories are lossless FP16, but
cumulative SSD IO can be substantial: the complete1024/t25 decoder observation
writes 75 GiB and reads about 142 GiB, with a 6.5-GiB live-file peak. Its
decoder-only active peak is 13.197 -> 5.648 GB while observed time is
56.074 -> 189.350 s, including host quality/export. These are not warmed
full-request/App/low-memory-machine performance results. The full25 same-latent
all-FP32 VAE pixel comparison passes independently. See
`docs/design/seedvr2-vae-spatial-scheduler.md` for scopes, failure records and
App/public-engine gates. Two seven-request real App workflows now pass, together
with every exported frame's numerical gate and exact PNG alpha. Actual engine
cancellation, quota failure and same-engine recovery also pass. In the single
1024/t25 staged/cache1 App observation, full/file-row request latency is
89.902/221.645 s and active peak is13.148/12.236 GB, not the much smaller
decoder-only peak. These observations are not warmed ABBA or a16-GiB claim.
No automatic file-row recommendation or VAE
ANE routing follows from these observations.

The separately packaged candidate is
`dist/seedvr2-spatial-rows-20261014/TurboCider.app` (minimum macOS 26.2). It passes
local signing/release guards, packaged-library contracts and a real 1024
file-row image smoke whose PNG matches the tested App workflow byte for byte.
The previously running App/package remains untouched.

The encoder-head candidate independently accepts
`encoder_spatial="full"|"file_rows"`; older drafts/defaults keep `full`.
It shares row/live-file-budget settings with the decoder, retains complete
global statistics/causal history and only schedules the singleton high-resolution
head, not temporal downsamplers/later stages/attention. Streaming source import
can be combined with it. The App has a separate manual selector and does not
automatically recommend spilling. The same <=1024 canvas and singleton2d/conv3d
restrictions apply. Rounded convolution outputs are numerically qualified,
not bitwise equal to full encoding; original coefficients are unchanged.

Image encoder-only ABBA observes .659063 -> .561925 s (about14.7% lower latency)
and active maxima5.426 ->2.932 GB. This is not whole-request speed/RAM admission.
Complete1024/t25 encoder observations trade20.665 ->69.627 s for8.002 ->5.854 GB
active peak; video writes25 GiB/reads49.40625 GiB, peaking at2.25 GiB live files.
Independent same-input audited FP32 encoder image/full25 gates pass, and source
streaming/padding matches the same row-file buffered route byte for byte.

Two real seven-request App workflows pass; the default exports remain byte
exact, while combined encoder+decoder rows pass every exported frame's unchanged
numerical gate and exact PNG alpha. Public-engine encoder cancellation, quota
failure and same-engine recovery pass. Single staged1024/t25 full/combined
requests observe90.043/272.910 s and13.215/12.236 GB active peaks; adding encoder
rows does not materially reduce the earlier decoder-only row request peak.
Remaining phases still matter. Hard budgets/16-GiB/>121-frame support are not
claimed; the admission guard/defaults remain unchanged.

New independent package:
`dist/seedvr2-encoder-spatial-20261015/TurboCider.app` (minimum macOS26.2).
Packaged-library contracts and a real1024 combined-row image smoke pass; its PNG
matches the tested App workflow byte for byte. The previously running App remains
untouched. Detailed scopes/records are in the spatial-scheduler design document.

The App exposes input encoder, output decoder, convolution-workspace and cache selectors and
heuristic staged/streaming advice for larger frame counts or low physical
memory. This is not automatic admission to a hard RAM budget. The streaming
source option avoids complete imported/resized RGB, while conditioning and DiT
still retain complete-clip state; the 121-frame model limit remains in force.
Reader/writer 1000-frame tests are not long-video model qualification. The
existing conservative admission estimate can still reject some 16-GiB tasks;
the App advice is not proof that those tasks are admitted or runnable.
The resource-advice candidate also checks the actual imported source count,
unsupported row-file canvases and the current temporary-volume free-space
snapshot. It displays the configured scratch quota instead of assuming8 GiB.
For measured1024-square/64-row video cases, decoder/encoder reference live-file
peaks are6.5/2.25 GiB; their separate phase lifetimes mean they are not added.
A lower quota or insufficient free-space snapshot produces a warning, not an
automatic disk reservation or permission to bypass native admission. Other
dimensions/row heights show an unmeasured-reference notice. Unknown free space
stays unknown, and single images do not inherit video-history requirements.
The App never deletes files, changes output size, truncates frames or enables
unqualified VAE ANE in response to these notices. Refresh the snapshot before a
large spilling job; it can become stale. See
`docs/design/seedvr2-vae-norm-pack-and-resource-advice.md` for scopes and records.
The independent ordinary App with these notices is
`dist/seedvr2-resource-advice-20261015/TurboCider.app` (macOS26.2+).
It passes seven real App requests with unchanged ordinary exports, package
signing/release/contracts and a complete512/t25 embedded-CLI smoke. Existing
running Apps are not replaced; quit an older instance before using the new
bundle against the same default state directory. The experimental fused-norm
and VAE ANE paths are not enabled in this package.
Full spatial VAE tiling, reusable history/workspace pools
and VAE Core ML/ANE convolution remain active work; see
`docs/design/seedvr2-vae-optimization.md`.

The later fixed Core ML projection experiment runs through a complete1024/t25
decoder and passes same-input audited-FP32/per-frame quality, but single
55.045/55.363-s GPU/CPU+NE observations have the same13.130-GB active maximum.
It is excluded from ordinary/App builds, not an enabled VAE ANE setting or a
low-memory recommendation. Optional read-window pooling also remains offline
and disabled: fewer allocations did not establish a speed/peak benefit. See
`docs/design/seedvr2-vae-ane-feasibility.md` for boundaries and preserved receipts.

An additional isolated direct-frame Core ML causal-conv experiment now avoids
the full temporal halo at one subsequent-frame hot layer. Complete25/FP32 gates
and decoder ABBA observe403.2 MB lower active maximum versus shared-history GPU,
with55.459/55.189-s warm medians. The small component difference is not a
full-request speed claim; no public VAE ANE selector/bank/default is enabled.
See `docs/design/seedvr2-vae-virtual-causal-conv.md` for qualification boundaries.

For streaming input, `vae_encode_wall` includes per-frame import and EOF work;
`vae_encode` subtracts the separately reported `frame_import` portion. That
portion is also included in `media_import`. Do not add overlapping timing
fields such as `restoration_total`, `frame_import` and encoder wall together.

The source-streaming App candidate is separately packaged at
`dist/seedvr2-stream-input-20261014/TurboCider.app` (macOS 26.2+), without replacing
the older delivery. Its two real App workflows and complete exported-media
checks are recorded in `docs/status/seedvr2-integration.md`. Do not run old/new
instances concurrently against the same default state directory.

The first same-library 512/t25 full-request ABBA observes buffered/streaming
warm medians **20.715874 / 20.768963 s**, strict speed gate **false**. Aggregate
warm active maxima also do not improve. Streaming is kept as an optional
decoded-output residency strategy, not advertised as a complete-request
speed/peak win. Decoder-only and App export checks cover every frame.

Slow producers exposed cadence-dependent H.264 keyframe selection in the
first writer. The corrected writer declares source FPS, uses a two-second
**frame-count** GOP and disables the duration limit. This changes the video
encoder recipe, not restoration math; its output must not be relabeled as
the preceding encoder's bitwise export qualification. Raw RGB diagnostics,
slow/fast producer checks and current-candidate App/export results are tracked
in `docs/status/seedvr2-integration.md`. The stable App delivery above remains
separate until the current optimization candidate is promoted.

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
