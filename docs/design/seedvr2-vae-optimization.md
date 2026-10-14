# SeedVR2 VAE acceleration and bounded-memory roadmap

## Scope and acceptance

The active objective remains VAE speed and peak-memory optimization, spatial
tiling, streaming decode/buffer reuse, public Core ML/ANE convolution (including
GPU/ANE cooperation and full-VAE feasibility), configurable speed/memory
tradeoffs, and usable long-video/low-memory App workflows. A successful small
codec screen is not completion of that objective.

Promote defaults only after original-parameter FP32 quality gates, finite
intermediates, full source frame/FPS preservation, matched full-request speed
and memory observations, actual App requests and cancellation/cleanup checks.
Keep lower-memory/slower schedules selectable. Do not credit CPU+NE gains
entirely to ANE or report a compute-plan preference as physical occupancy.

| Work item | Current evidence / remaining requirement |
| --- | --- |
| Singleton Conv3D -> Conv2D | Existing default folds replicated initial frames in FP32 and handles temporal-kernel-one layers in 2D. Real temporal convolutions retain causal operands. |
| End-to-end streaming decoded output | Frame sink/backpressured writer and explicit frame-based GOP pass codec and 14 v4 App requests; new memory policies match exported frames. Initial paired full-request speed/active-peak gate fails, so buffered stays default. New encoder differs from legacy encoding. |
| Streaming source / VAE encode | Optional `encoder=streaming` imports one RGB frame and keeps the original initial-five/subsequent-four encode schedule and padding. Reader/display/FP32 normalization and encoded latents pass bitwise checks; 14 real App requests and all exported frames match. 512/t25 staged/cache1 full-request ABBA is effectively flat, with no substantial request peak win. Buffered remains default. |
| Buffer reuse | Independent frame owners/normalization sharing retain the prior exact App and full-request ABBA qualification. New opt-in spatial-upscale history sharing passes complete25 words/FP32, cancellation/recovery and14 App requests; same-build single requests lower active peak133.7 MB, with no new warm-speed claim. Optional row-window pooling lowers allocation count but remains default-off/no proven speed gain. Fixed GPU workspace pools remain open. See [phase/ownership evidence](seedvr2-vae-phase-and-upscale-history.md). |
| Request/App configuration | Encoder/decoder/convolution/tile-row/cache/history and default-off upscale-sharing controls; typed native guards and real App persistence/switches pass. User-applied heuristic advice selects compatible staged/streamed settings without lowering admission. Real lower-RAM hardware and longer-model clips remain unqualified. |
| Full VAE spatial tiling | Qualified manual public/App row-file decoder and independent encoder-head schedules preserve whole-frame statistics/attention/causal context and complete25 quality. Combined App requests trade90.043 ->272.910 s for13.215 ->12.236 GB active maximum. This is still not tiling of every operator/downsampling stage or proof of16-GiB/>121 operation. See [stage scheduler](seedvr2-vae-spatial-scheduler.md). `tiled3d` remains workspace-only. |
| GPU/ANE convolution | Earlier prepared512 hotconv split is slower/no memory win. New direct separate-history packing executes6144 predictions inside a complete1024/t25 decoder; every frame passes audited FP32 and ABBA observes403.2 MB lower active maximum, warm55.459/55.189 s. It is still experimental: full-request/App/public-bank/physical occupancy gates remain. See [virtual causal boundary](seedvr2-vae-virtual-causal-conv.md); production hybrid bank remains DiT FFN only. |
| Full VAE on ANE | Feasibility experiment required; no support/performance claim from conversion alone. |
| Long complete videos | Reader/writer both test complete 1000-frame codec sequences. Streaming import and causal encode are integrated, but model/request still cap complete source clips at 121 frames. Complete conditioning/DiT residency and real longer-model-clip qualification remain required. |

The latest complete25 phase trace localizes the original13.130-GB maximum to
spatial upsampling plus its causal convolution, with9.061 GB of deduplicated
cached history already resident. Upscale sharing removes512 MiB from that
region but moves the maximum to final-stage norm1/conv1, reducing the complete
decoder by only133.7 MB. Further work must address that new maximum and persistent
history/whole-request residency, not credit local savings as a completed goal.

## Streaming decoded output

Keep the exact existing temporal partition and causal histories. At the
high-resolution tail, consume each FP16 BCTHW RGB frame synchronously instead
of saving it in a vector and concatenating all output frames. Convert with the
same FP32 clamp/round-to-uint8 recipe, then append to AVFoundation with an
exact frame-index/FPS timebase. Backpressure must not queue unlimited buffers.
Trim only declared 4n+1 padding, never source frames. Cancellation or failure
must remove only private staging; complete geometry validation precedes atomic
publication. Buffered raw-pixel dumps remain available through the old route.

This removes growing decoded-output residency, not the convolution peak or
the full source/conditioning/DiT state. Decode wall time includes sink work;
report frame-export time separately without counting it twice. A consumer
that retains frames defeats this bounded-output contract.

## Streaming source / encode boundary (2026-10-14)

The buffered path remains unchanged. The explicit video-only
`seedvr2_vae.encoder=streaming` path decodes into one reused RGB24 frame buffer,
applies the same display transform and evaluated FP32 uint8 conversion, bicubic
resize and normalization, then supplies singleton FP32 BCTHW to the VAE. A
completed FP16 cast owns each encoder frame before the reader buffer is reused.
The VAE concatenates the same initial five / subsequent four frames, keeps the
same global spatial statistics and causal caches, and repeats only the last real
frame for existing 4n+1 model padding. This is temporal input residency, not full
spatial VAE tiling or independent temporal restoration windows.

The pipeline must verify complete EOF after encoding and before any sampling or
DiT staged reload; declaring the frame count alone is not enough. The reader
rejects short/extra sequences, FPS/PTS mismatches, unsupported transforms,
changing dimensions and source-file stamp changes. Its cancellation retains the
native typed status. Reader codec/internal allocations are not claimed bounded
by the single caller-owned RGB capacity.

Evidence: eight transforms and all 72 decoded/normalized frames byte exact;
source counts 1/2/4/5/6/9 encode to identical FP16 latents including padding;
two seven-request App chains retain identical PNG alpha and every t25/24-FPS
video pixel. Public engine tests cancel on source frame one (status 2), reject
an undeclared tail before sampling (status 1), then restore successfully on the
same engine (status 0); the existing destination/source and staging cleanup pass.

Matched 512/t25 staged/cache1 full-request ABBA has six warm requests per mode:
**21.254022 / 21.251528 s**, effectively flat, and active maxima
**9,035,267,926 / 9,035,230,774 bytes**. The 37,152-byte difference is not a
substantial peak-memory result. Every frame in all eight export pairs is exact.
Keep the route selectable for bounded source RGB residency, without claiming
whole-request acceleration, whole-VAE tiling, a hard RAM budget, 16-GiB-device
support or >121-frame model support. The existing conservative admission guard
can still reject a 16-GiB host even at 512 video output; it needs a separately
justified model/workload estimate or qualification, not a lowered assertion.

Records: `seedvr2-rgb-reader-guards-v3-20261014/report.json`,
`seedvr2-vae-encode-source-128-v1-20261014/report.json`,
`seedvr2-app-stream-input-export-screen-v1-20261014.json`,
`seedvr2-stream-input-video512-t25-staged-abba-v1-20261014/report.json`,
`seedvr2-stream-input-video512-t25-staged-exports-v1-20261014.json`,
`seedvr2-stream-input-lifecycle-v1-20261014/report.json`.

## Exact-context spatial tiling

The VAE normalizes over the whole spatial frame and attends to all spatial
keys. Running independent complete VAE crops changes both. Two distinct
experiments must therefore have distinct names/consent:

1. Exact-context stage tiling: preserve global frame statistics, complete
   attention, causal histories, convolution halos/padding/strides and temporal
   upsampling phases. Stream layer tiles into bounded output storage; calculate
   normalization statistics in a separate full-frame pass. Use bounded query
   attention with all keys, not local tile attention. Compare reduction-rounding
   changes against original FP32 and stable GPU witnesses.
2. Approximate independent VAE tiles: only an explicit approximation option,
   with overlap/blending and image/video seam/temporal-flicker gates. It cannot
   inherit the exact-context numerical recipe or the old quality qualification.

A layer scheduler needs storage for global intermediates and causal tails.
Moving them to CPU is not proof of lower physical memory on unified-memory
hardware. Evaluate bounded host staging or owned temporary spill files,
including their I/O time and sampled process footprint. Do not retain views
that pin complete GPU activations. The existing bounded convolution is a
useful reference, not completion of full-stage tiling.

## Public Core ML/ANE convolution experiment

Start with the measured hot decoder convolutions, not the entire VAE at once.
Use only original checkpoint weights and fixed, bounded spatial geometries.
For a causal Kt x Kh x Kw filter, arrange the required temporal taps as input
channels of a Conv2D bank. Preserve tap order, bias, spatial halos and strides;
do not pretend a noninitial frame has no history. Compare this arithmetic to
the safe FP32 singleton-fold and original-parameter controls.

Evaluate three controls with identical operands/shapes: GPU-only, Core ML
CPU-only, and Core ML CPU+NE. Include cold loading/compilation separately and
include host/layout transfer, synchronization and merge costs in warm results.
Inspect supported/preferred operation devices through MLComputePlan, but label
that as anticipated placement. A runtime occupancy claim requires independent
hardware evidence.

The initial cooperative route splits **output channels**: GPU computes one
range, Core ML the other from the same normalized input/causal history. This
avoids a partial-convolution sum across temporal taps; assemble output channels
in original order before bias/residual/normalization. Schedule nonaliasing,
bounded output slots, and never overwrite a Core ML output still consumed by
MLX. Sweep split ratios and tile sizes; prediction overhead may outweigh the
compute gain, so neither parallelism nor ANE is presumed faster.

The alternative assigns whole convolutions or residual stages to Core ML,
reducing per-operation transfer. Full-VAE conversion must expose causal state
inputs/outputs and test normalization, attention, upsampling, initial-frame
phases, all finite activations and buffer residency. CPU fallbacks are not an
ANE-only VAE. Unsupported or slower stages remain on GPU; only qualified
manual routes may enter the App before a machine/workload-specific auto policy.

Official API/architecture references:

- Apple, Deploying Transformers on the Apple Neural Engine:
  https://machinelearning.apple.com/research/neural-engine-transformers
- Apple, MLComputePlan device usage:
  https://developer.apple.com/documentation/coreml/mlcomputeplan/deviceusage

These references motivate candidate layouts and device inspection; they are
not evidence that this VAE convolution or complete network runs on ANE.

## Long-video and App policy

Expose speed-first, staged residency, streaming output, convolution workspace
and cache retention independently; smaller cache/workspaces are not hard RAM
budgets. Advice considers actual imported frame count, output dimensions and
physical memory, is explicitly heuristic, and requires user confirmation.

Do not lift the 121-frame limit based on codec tests alone. Bounded import/resize
and causal encoding plus decode/export are now optional routes, but genuinely
longer restoration still needs a bounded complete-context DiT schedule and real
longer-clip validation. Explore layer/window scheduling with
global state stored outside active GPU memory for exact context. If independent
temporal restoration windows are needed, make their changed context explicit,
align latent phases, control noise and verify overlap/seams/temporal consistency.
Every source frame and the exact source CFR must survive; no automatic
truncation, frame-rate substitution or hidden cropping is permitted.

Low-memory-machine qualification requires actual hardware/process observations
or a clearly bounded simulation scope. A 64-GiB host and a heuristic App
recommendation alone do not prove successful operation on a 16-GiB machine.
