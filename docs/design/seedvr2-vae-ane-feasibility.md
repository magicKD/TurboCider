# SeedVR2 VAE: fixed-tile Core ML / ANE feasibility

## Conclusion (2026-10-14)

There is a **conditional component opportunity**, not a qualified VAE backend.
Native reusable FP16 IOSurface IO substantially changes the earlier negative
Python-predict result. Small fixed tiles and one measured upsampling projection
are candidates; larger tiles and indiscriminate whole-VAE migration are not.
Compare against an optimized GPU route, not only the existing Conv3D operator.
No production VAE routing, App default, or checkpoint was changed by this screen.
The subsequent **complete spatial convolution boundary** tests below include
GPU pack/synchronization/assembly and cooperative output channels. At the tested
512-square hot layer they do **not** establish a speed or memory win over GPU.

Machine: Apple M4 Max, 64 GiB, macOS 26.6.2. Original VAE SHA-256:
`20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1`.
Weights remain FP16 without BF16 rounding/quantization. Three temporal taps are
stacked into Conv2D input channels **in tap/channel order**, preserving spatial
halos. Bias stays outside the graph, preserving the native separate-round/add
recipe. Inputs are synthetic normalized activations, not captured real media.

## Measured component timings

Warm medians below are milliseconds. GPU is native MLX dispatch plus completion;
the pack column includes an NDHWC-to-stacked-Conv2D GPU packing operation. CPU+NE
is native synchronous Core ML prediction with reusable FP16 IOSurface input and
requested output backing. **Its input packing into that surface, GPU handoff,
output layout conversion and whole-frame assembly are not included.** These are
feasibility screens, not balanced full-VAE ABBA speed qualifications.

| Original operator / core tile | GPU Conv3D | GPU prepacked Conv2D | GPU pack + Conv2D | Core ML CPU+NE, surface |
| --- | ---: | ---: | ---: | ---: |
| Hot 3x3x3, 768 stacked input / 64 output, 64x64 | 1.009 | 0.947 | 0.552 | 0.313 |
| Same, **all 128 outputs**, 64x64 | 0.818 | 0.618 | 0.688 | 0.526 |
| Hot 3x3x3, 64/128 outputs, 128x128 | 1.497 | 1.004 | 1.182 | 1.130 |
| Hot 3x3x3, all 128 outputs, 128x128 | 2.506 | 1.314 | 1.588 | 1.996 |
| Hot 3x3x3, 64/128 outputs, 256x256 | 4.380 | 2.706 | 3.123 | 26.235 |

Hot layer: `decoder.up_blocks.3.resnets.0.conv1`, original OCTHW
`[128,256,3,3,3]`. The full-output 64-square row is a separate 15-iteration
screen; the others use nine GPU warm calls and three signed-input blocks with
nine warm native calls each. Reversing tile/route order in the second screen
again finds small-tile opportunity and a pronounced 256-square CPU+NE cliff.
Independent short GPU observations vary (including pack timing below prepacked
timing); do not interpret those differences as negative packing cost or a
portable optimum. A matched warmed-cadence/ABBA scheduler is still required.

Also screened the real high-density 1x1 upsampling projections, **all outputs**:

| Original operator / tile | Native GPU matmul | GPU pack + Conv2D | CPU+NE surface |
| --- | ---: | ---: | ---: |
| `up_blocks.2.upsamplers.0.upscale_conv`, 256 -> 1024, 64x64 | 0.833 | 0.491 | 0.337 |
| `up_blocks.0.upsamplers.0.upscale_conv`, 512 -> 4096, 32x32 | 0.842 | 0.897 | 0.878 |

The production upsampling baseline already uses matmul, not Conv3D. The first
projection is worth a fixed projection/shuffle graph experiment; the second
does not show a standalone advantage. More arithmetic or more output channels
does not automatically make ANE faster. The 256-square cliff's internal cause
(compiler schedule, tiling, IO or hardware behavior) is **not identified**.

### IO matters, but a shared pointer is not a zero-copy proof

For the 128-square/64-output hot convolution:

- Earlier Python prediction: GPU 3.250 ms / CPU+NE 13.937 ms, with Python copies
  and GPU output readback included. Do not compare that scope as a speedup ratio
  to the new resident/native-only scope.
- New native CPU+NE: regular MLMultiArray 1.593 ms; wrapped pointer/output backing
  1.603 ms; IOSurface-backed input/output 1.130 ms in the reverse-order screen.
- All requested native output backings in the screens returned the expected
  pointer/pixel-buffer identity. Every prediction was accessed stride-correctly
  and compared with the FP32 witness; signed input **+/-/+** catches stale reuse.
  This does **not** exclude internal Core ML layout conversions/copies.

For the complete 64-square convolution, CPU+NE prediction saves about **0.162 ms**
against GPU pack+Conv2D. The separately observed warm **CPU** input refill alone
costs about **0.182 ms**, already consuming that margin before output handoff.
This arithmetic is illustrative, not a measured combined scheduler: it shows
why naive serial CPU packing is not a proven acceleration. Packing directly on
GPU into an IOSurface-backed MTLBuffer and overlapping the next tile must be
measured rather than assumed free.

CPU-only controls, no-output-backing controls, regular allocated MLMultiArray,
wrapped host pointers and public IOSurface allocations all executed. CPU+NE
must not be described as an ANE-only graph or isolated ANE arithmetic timing.
MLComputePlan anticipates NeuralEngine for these convolution operators and CPU
for CPU-only controls. Physical ANE occupancy and runtime overlap are unmeasured.

## Numerical and memory evidence

The unchanged component gate is finite outputs, cosine >= .999 and relative
L2 <= .05 against an original-order FP32 Conv3D reference. All measured routes
pass; no threshold was relaxed. Three-tap GPU Conv3D error is about .000208,
CPU+NE about .000710, and GPU FP16 stacked Conv2D about .0082-.0083. Passing a
synthetic convolution gate does **not** qualify compounded VAE image/video
quality, temporal stability, initial-frame folding, or real activation extremes.

NumPy/BLAS floating-point warnings were retained. Each reference is checked for
finite values and independently sampled at 27 points with direct FP64 products;
observed maximum reference errors are below 2.8e-6. They are not silently treated
as harmless without those checks.

The complete 64-square hot-convolution surface slots occupy **10,780,672 bytes**
(padding included); two such pairs would require 21,561,344 bytes before any
extra consumer/output pools. Its isolated CPU+NE process sampled about 26.4 MB
physical footprint. The 128-square/64-output pair occupies 34,045,952 bytes,
while the 256-square pair occupies 122,519,552 bytes. These are **probe IO/process
observations**, not complete-VAE memory or a hard peak/budget. GPU probe active
peaks include prepared control operands and use a different memory metric.

ANE placement alone does not remove full-frame activations, all layer causal
histories, normalization/attention state or allocator cache. Tile-size/residency
scheduling is the memory lever. Additional input/output pools can increase the
request peak. Do not claim a whole-process memory win by comparing GPU active
allocation to a separate Core ML process footprint.

## Exact fixed-pattern design

### Implemented shared-IO boundary follow-up

The isolated native probe now implements, and actually executes:

- GPU temporal/halo packing into public FP16 IOSurface-backed MLMultiArray using
  page-aligned Metal no-copy bindings, plus stride-correct GPU output assembly.
- Serial CPU-only/CPU+NE controls, a bounded two-slot producer/consumer pipeline,
  and CPU+NE prefix plus a disjoint native GPU Conv3D/stacked-Conv2D suffix.
  The suffix runs via MLX async submission and is completed before channel merge.
- Scalar and 16x16 threadgroup transpose pack/consume kernels, with independent
  bitwise indexing witnesses for every packed/assembled element across + / - / +
  inputs. Output is NaN-poisoned before each observed frame, outside timing, to
  detect missing tiles/channels instead of accepting a previous frame's storage.
- Completed Metal submissions retain the MLX allocation owners and multiarray
  surface owners; exceptional unwinding joins before releasing external memory.
  A ring slot is consumed/joined before prediction reuses its output.

CoreVideo reports 9,732,160 logical allocation bytes for one packed-input
surface, 64 bytes beyond its pixel-row extent, despite a page-aligned address.
The initial strict page-size guard rejected this; the failed screen is preserved.
The final implementation rounds only after `mach_vm_region` verifies the entire
mapped readable/writable VM extent. Kernels access only validated logical strides;
no unverified trailing page is wrapped. Each complete-output slot pair then has
10,813,440 mapped bytes; two pairs have 21,626,880 bytes. Output backing identity,
shape and all strides must match exactly or the route fails closed.

#### Full boundary observations

Final 512x512 results use 64x64 tiles, **all 128 output channels**, original
three-tap coefficients, nine warm requests per process and two processes per
route in forward/reverse order. Every request computes all 64 tiles anew.
The half-channel routes compute the other 64 channels on GPU and merge them;
they are not incomplete-prefix timings. All scopes include actual packing,
producer completion, prediction, consumer completion, output layout assembly
and any suffix/join. Compilation/loading, layout preflight, output poisoning,
CPU numerical scans and fixture IO are outside the timing.

| Complete 512-square convolution boundary | Warm median (ms) | Maximum sampled process footprint (bytes) | MLX active peak (bytes) |
| --- | ---: | ---: | ---: |
| GPU Conv3D | 32.580 | 1,208,304,984 | 480,002,072 |
| GPU pack + stacked Conv2D | 44.833 | 2,888,828,272 | 2,158,788,634 |
| CPU+NE serial shared tiles | 61.325 | 1,222,526,272 | 480,002,072 |
| CPU+NE two-slot pipeline | 35.386 | 1,233,372,504 | 480,002,072 |
| CPU+NE prefix + GPU Conv3D suffix | 42.753 | 1,268,893,016 | 516,210,692 |
| CPU+NE prefix + GPU stacked-Conv2D suffix | 42.913 | 2,881,455,520 | 2,128,117,766 |
| CPU-only serial shared tiles | 92.702 | 1,222,559,088 | 480,002,072 |

These are isolated **synthetic full-spatial-convolution** process observations,
not complete VAE/App/system peaks or enforced budgets. Child processes retain
host fixture/reference data and prepared control filters. The final GPU-only
route does **not** allocate the unused Core ML assembly target: an earlier v4
probe did, biasing GPU peak upward by one 64-MiB activation. The v5 rerun corrects
that baseline. No memory reduction over GPU Conv3D is demonstrated: active peaks
are identical, while two Core ML surface pairs raise process footprint by about
25.1 MB. Allocator-active accounting excludes those external surface allocations.

At this geometry the pipeline is about 8.6% slower than GPU Conv3D; half-channel
cooperation is slower still. The pipeline hides much producer/consumer waiting
relative to serial prediction, but that is not a whole-VAE/over-GPU speed win.
Also, full-frame stacked Conv2D here is **slower and substantially larger** than
Conv3D despite being faster in some small-tile screens. Do not generalize the
small-tile results or choose a route based on MLX active bytes alone.

At 128-square, the transposed-kernel screen observes GPU Conv2D about 1.774 ms,
pipeline 2.580 ms and mixed Conv2D 2.108 ms. Independent GPU short-screen medians
vary substantially by process/order; these are exploratory component cohorts,
not a stable default-promotion/full-request ABBA gate. Scalar shared-IO controls
also complete and pass the numerical gates. No production route is promoted.

Each successful 512-square non-GPU process checks 642,318,336 input pack values
bitwise. Full-output assembly checks 100,663,296 values (half-prefix: 50,331,648).
There are 768 actual Core ML predictions per process, with 768 exact backing
identity/shape/stride hits. Each full output passes the unchanged FP32 finite,
.999-cosine/.05-relative-L2 gate. The FP32 reference's 27 independently sampled
FP64 points have maximum error 3.11e-6. This still excludes real activations,
actual VAE global normalization, bias, causal state transitions and whole-VAE
image/video qualification. Physical ANE occupancy/overlap is not measured.

Records:

- `seedvr2-vae-shared-tile-128-v1-20261014/report.json`: completed attempts with
  failed page-size binding guard; **not** a passed shared-IO screen.
- `seedvr2-vae-shared-tile-128-v2-20261014/report.json`: successful scalar IO.
- `seedvr2-vae-shared-tile-128-transpose-v1-20261014/report.json`: successful
  transposed pack/consume and hybrid routes.
- `seedvr2-vae-shared-tile-512-transpose-v1-20261014/report.json`: v4 success,
  but GPU-only memory includes the unused assembly allocation.
- `seedvr2-vae-shared-tile-512-transpose-v2-20261014/report.json`: final v5
  corrected-memory comparison; use this for the 512 table above.

Native implementation: `tools/native/seedvr2_vae_shared_tile_probe.mm`; compile
with `bash tools/native/build_seedvr2_vae_shared_tile_probe.sh` into a fresh
`TURBOCIDER_VAE_SHARED_PROBE_OUT` directory. Driver:
`tools/validation/seedvr2_vae_shared_tile_screen.py` (`--kernel transpose`,
`--size`, `--rounds`, optional `--hybrid-manifest`). CPU fixture guards plus the
existing VAE numerical/cohort guards pass 20 tests with the NumPy/safetensors
validation interpreter. The managed minimal environment passes 12 and skips
the eight reader-dependent cohort tests; those eight were separately run and
passed, not relabelled from skips.

Next work should target captured real activations and alternative graph/layer
boundaries, global-statistics stage tiling and bounded VAE residency, rather than
enabling the tested slow hotconv split. Full-VAE offload, real-media quality,
low-memory-machine behavior and long-video/App integration remain open.

The safest next graph boundary is a single convolution, or a local projection
plus layout/shuffle/activation segment with verified explicit cast boundaries.
Use original coefficient/tap order and the exact outer padding/strides. Initial
replicated-frame folding currently uses FP32 coefficients/accumulation; do not
replace it with a three-tap FP16 graph without separate quality qualification.

Two barriers prevent simply putting an entire residual block into independent
tiles:

1. `VAE::resnet` is global GroupNorm/SiLU -> causal conv1 -> global GroupNorm/SiLU
   -> causal conv2 -> shortcut/add. GroupNorm reduces **all spatial positions**
   within each group and frame. Tile-local statistics change the model. To keep
   context, calculate whole-frame statistics, then pass fixed mean/inverse-std
   inputs into a tile affine/SiLU/conv graph. Conv1 must finish its full-frame
   statistics pass before norm2/conv2; both convolutions cannot be fused into an
   independent one-pass crop using unknown future statistics.
2. Mid-block attention uses all spatial keys, and upsampling has explicit causal
   temporal phases. Keep global attention on GPU first. Preserve current/history
   ownership and phase-zero initial-frame handling; distinct fixed state/phase
   signatures are preferable to pretending time is spatial batch.

Thus fixed **affine + SiLU + conv** with externally supplied global statistics,
or **1x1 projection + pixel/temporal shuffle**, is more defensible than an
unchecked full-residual/full-VAE tile graph. Explicit FP32/FP16 rounding may
prevent some Core ML fusions; verify the compiled numerical recipe.

## GPU/ANE cooperation plan and acceptance

The full-output 64-square hot convolution now has the negative full-boundary
result above. Prioritize other layers/boundaries and the 256->1024 projection
before considering any production convolution route.
Use public APIs; neither private ANE APIs nor automatic App routing are needed.

1. Allocate a bounded ring of nonaliasing IOSurface input/output slots. Prepare
   halo plus temporal-channel packing on GPU in the actual backing layout.
   Synchronize the producer before Core ML reads it; do not reuse the slot until
   both prediction and its GPU consumer complete. Identity hits are only one
   part of verifying this ownership contract.
2. Compare complete-convolution offload with **spatial-tile distribution**:
   GPU processes one disjoint tile range while CPU+NE processes another.
   Maintain global normalization/attention outside this local routing. Measure
   real overlap and shared-bandwidth contention, including pack/merge.
3. Also compare **output-channel split** using a common normalized source.
   Assemble channels in original order before bias/residual. Avoid splitting
   temporal taps and summing partial FP16 convolutions: that adds reduction and
   extra rounding. Prefix timing alone cannot prove the other half or a join.
4. For streaming throughput, GPU packs tile n+1 / runs its assigned tiles while
   Core ML predicts tile n and GPU consumes completed tile n-1. This is an
   intended schedule, **not observed overlap**. Double buffering is useful only
   if reduced critical-path time outweighs extra memory/bandwidth.
5. Measure the complete boundary:
   `pack + producer fence + max(GPU branch, Core ML branch) + consumer fence +
   output-layout/merge + bias/residual`. A pipeline may overlap some terms, but
   only an observed full-frame schedule can establish that gain.

Next gates: captured real decoder operands, whole-frame border/interior tiles,
original FP32 VAE and stable GPU witnesses, finite intermediates, initial and
subsequent causal phases, image and multi-frame export/temporal checks, warmed
ABBA latency plus sampled process peak, actual App cancellation/release tests.
Keep routes experimental/manual until machine/workload-specific gains pass.
The cooperative **convolution prototype** has now been implemented and tested;
the full VAE has **not** been converted or tested by this work. Full-VAE ANE
support remains an open feasibility question.

## Records and reproduction

Artifacts are under `outputs/` and are not checked-in generated media/models:

- `seedvr2-vae-native-io-tiles-v1-20261014/report.json`: four tiles, five warm
  predictions per signed-input block; Python GPU control and native IO routes.
- `seedvr2-vae-native-io-tiles-v2-20261014/report.json`: six operators/geometries,
  reverse route order, nine warm predictions/block and native GPU controls.
- `seedvr2-vae-native-io-full64-v1-20261014/report.json`: all 128 outputs at
  64-square, fifteen warm predictions/block, same v2 probe binaries.

The reports bind the checkpoint, compiled tree/manifest, input/reference and
probe executable hashes. Native source: `tools/native/seedvr2_vae_coreml_io_probe.mm`
and `seedvr2_vae_gpu_tile_probe.mm`; driver/guards:
`tools/validation/seedvr2_vae_native_io_screen.py` and
`tests/native/test_seedvr2_vae_native_io_screen.py`. Compile with
`bash tools/native/build_seedvr2_vae_io_probe.sh`; run its driver's `--help`.
The default build is isolated from canonical App/native binaries. Export fixed
original-weight banks using `tools/coreml/export_seedvr2_vae_conv.py`; models,
fixtures and reports must use fresh output paths. Do not overwrite a live probe.

Primary API/architecture references (not this VAE's performance evidence):

- Apple, Deploying Transformers on the Apple Neural Engine:
  https://machinelearning.apple.com/research/neural-engine-transformers
- Apple, MLComputePlan: https://developer.apple.com/documentation/coreml/mlcomputeplan
- Apple, MLPredictionOptions outputBackings:
  https://developer.apple.com/documentation/coreml/mlpredictionoptions/outputbackings
- Apple SDK CoreML/CoreVideo public interfaces used in the compiled probes.
