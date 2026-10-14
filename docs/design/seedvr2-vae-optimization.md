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
| Buffer reuse | Independent frame owners and known materialized normalization-output sharing pass exact codec/App, 1024/t9 VAE ABBA (about 1.77% faster) and completed 1024/t25 full-request ABBA (about 1.58% faster, 939.5 MB lower active peak). All eight full-request export pairs match every frame. Fixed reusable workspace pools remain open. |
| Request/App configuration | Decoder/convolution/tile-row/cache controls and heuristic memory advice; native plan guards and real resident/cache4, staged/cache1 workflow/switching pass. Actual lower-RAM hardware, additional tile settings and long-model clips remain unqualified. |
| Full VAE spatial tiling | Not implemented by `tiled3d`: that option only bounds each convolution's workspace. Prototype a global-statistics, halo-correct stage scheduler and measure real process peak. |
| GPU/ANE convolution | Public shared-IO, GPU pack/consume, double buffering and disjoint output-channel cooperation pass full-spatial-convolution numerical/layout screens. Final 512 hotconv boundary is 32.58 ms GPU / 35.39 ms CPU+NE pipeline / 42.75 ms hybrid, with no memory win over GPU; not promoted. Other layers/real-activation/full-VAE qualification remain open. See [feasibility](seedvr2-vae-ane-feasibility.md); production hybrid bank is DiT FFN only. |
| Full VAE on ANE | Feasibility experiment required; no support/performance claim from conversion alone. |
| Long complete videos | Writer supports a tested 1000-frame producer, but model/import/request still cap complete source clips at 121 frames. Import/encoder/DiT bounded residency and real longer-clip validation remain required. |

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

## Spatial tiling without silently changing the model

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

Do not lift the 121-frame limit based on writer tests alone. A genuinely longer
pipeline needs bounded import/resize and causal encoding, then a bounded DiT
schedule, then streaming decode/export. Explore layer/window scheduling with
global state stored outside active GPU memory for exact context. If independent
temporal restoration windows are needed, make their changed context explicit,
align latent phases, control noise and verify overlap/seams/temporal consistency.
Every source frame and the exact source CFR must survive; no automatic
truncation, frame-rate substitution or hidden cropping is permitted.

Low-memory-machine qualification requires actual hardware/process observations
or a clearly bounded simulation scope. A 64-GiB host and a heuristic App
recommendation alone do not prove successful operation on a 16-GiB machine.
