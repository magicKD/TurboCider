# SeedVR2 VAE: fixed-tile Core ML / ANE feasibility

## Conclusion (2026-10-15)

There is a **conditional component opportunity**, not a qualified VAE backend.
Native reusable FP16 IOSurface IO substantially changes the earlier negative
Python-predict result. Small fixed tiles and one measured upsampling projection
are candidates; larger tiles and indiscriminate whole-VAE migration are not.
Compare against an optimized GPU route, not only the existing Conv3D operator.
No production VAE routing, App default, or checkpoint was changed by this screen.
The subsequent **complete spatial convolution boundary** tests below include
GPU pack/synchronization/assembly and cooperative output channels. At the tested
512-square hot layer they do **not** establish a speed or memory win over GPU.
The later captured-real upsampling screen below finds a conditional lower-
allocation projection boundary, but no meaningful speed win over its corrected
GPU baseline. A whole-VAE ANE backend remains unqualified.

Subsequent [direct-frame causal convolution](seedvr2-vae-virtual-causal-conv.md)
changes the actual VAE boundary: it packs two separate histories/current input
without a full temporal halo. Complete1024/t25 audited-FP32/per-frame gates and
full-decoder ABBA pass, with403.2 MB lower active maximum versus the already-
shared GPU baseline. Warm medians55.459/55.189 s are only about0.49% apart; the
stronger result is component workspace reduction, not substantial full-request
acceleration. The original fixed bank remains unqualified/experimental and no
VAE ANE App default is enabled. Subsequent isolated complete requests preserve
the403.2-MB MLX active reduction but are speed-flat (warm90.231/90.196 s);
seven experimental App requests pass. Actual preencoder RGB differs by at most1
on the diagnosed1024/t25 fixture, versus138 after H264; predecode hashes are exact.
See the linked direct-frame document for scopes/receipts. Balanced full-request
speed, broader media/hardware, physical occupancy and a public bank remain open.

The subsequent [256-output upscale causal convolution](seedvr2-vae-upscale-causal-conv.md)
adds a second independently validated signature. Dual-versus-hot-only complete
decoder ABBA now measures55.682/54.192 s (about2.68% lower latency), with another
402.4 MB lower active maximum. Complete raw request and fourteen App regressions
pass; single whole-request active maxima are13.015/12.236 GB. This still is not
balanced full-request speed, full-VAE ANE or low-memory-hardware qualification.

The later [halo-aligned rectangular tile comparison](seedvr2-vae-halo-aligned-tiles.md)
also executes64x62/64x94 banks. Against the already optimized dual-norm route,
complete1024/t25 ABBA finds0.84% slower/0.44% lower median latency, respectively,
with unchanged active maxima. The smaller62-wide external pool and fewer94-wide
predictions do not establish a new overall App memory/speed profile. Neither
geometry is promoted to default or a qualified public VAE bank.

Machine: Apple M4 Max, 64 GiB, macOS 26.6.2. Original VAE SHA-256:
`20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1`.
Weights remain FP16 without BF16 rounding/quantization. Three temporal taps are
stacked into Conv2D input channels **in tap/channel order**, preserving spatial
halos. Bias stays outside the graph, preserving the native separate-round/add
recipe. The initial screens use synthetic normalized activations; the later
upsampling follow-up explicitly uses captured real complete-context operands.

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

## Captured-real spatial upsampling follow-up (2026-10-15)

An offline observation hook captures the actual inputs to
`decoder.up_blocks.2.upsamplers.0.upscale_conv` at frames0 and24 while decoding
the **complete25-frame** restored native latent. Both inputs are original FP16
NDHWC `[1,1,512,512,256]`. No source/model request is truncated; selected input
capture is not a whole-video ANE test. Frame0 ranges from -224.5 to115.25,
substantially different from the earlier synthetic normalized inputs.

The new original-weight public Core ML banks have fixed64-square inputs:
projection-only and projection plus exact spatial shuffle. The channel order
is `[spatial_row,spatial_col,c]`, **not** ordinary channel-first pixel shuffle.
Bias stays outside both graphs and is added after a materialized FP16 projection
by the GPU consumer, preserving the original separate-round/add recipe. Both
graphs remain probe-only/`qualified=false`; full-VAE/App routing is unchanged.

Complete512-square layer requests compute all64 tiles, all1024 projection
channels, bias and the1024-square/256-channel output. CPU-only, CPU+NE serial,
two-slot pipeline, whole GPU matmul/Conv2D and bounded GPU-tile controls execute.
Every pack/assembly value has an independent scalar-index witness, every output
passes unchanged finite/.05-relative-L2/.999-cosine gates against a blocked FP32
reference, and15 independent FP64 products check that reference. Inputs alternate
`+/-/+` with fixed original bias; output poisoning detects incomplete writes.
Core ML backing identity/shape/strides must match every actual prediction.

### Corrected complete-layer observations

The final frame0 comparison uses two independent processes per route,
forward/reverse order and six warm complete requests each. Timing includes pack,
producer completion, Core ML prediction, consumer completion, bias/shuffle and
assembly; loading, reference scans, preflight witnesses and poisoning are outside.

| Complete captured frame0 boundary | Warm median (ms) | Active MLX peak (bytes) |
| --- | ---: | ---: |
| Native GPU matmul + separate bias/shuffle | 22.773 | 1,209,010,180 |
| CPU+NE projection, serial shared tiles | 37.412 | 672,139,268 |
| CPU+NE projection, two-slot pipeline + GPU bias/shuffle | 22.888 | 672,139,268 |
| GPU1x1 Conv2D + native bias/shuffle | 23.803 | 1,209,010,180 |
| GPU matmul + standalone fused bias/shuffle handoff | 24.658 | 1,209,010,180 |
| Bounded GPU tile matmul + fused bias/shuffle | 43.386 | 682,625,028 |
| CPU-only projection, serial | 55.666 | 672,139,268 |
| CPU+NE projection+shuffle graph, two-slot + GPU bias | 36.047 | 672,139,268 |
| CPU-only projection+shuffle graph, serial | 617.350 | 672,139,268 |

CPU+NE projection is effectively speed-flat versus the corrected whole GPU
route, not a qualified acceleration. It saves **512 MiB active allocation** at
this prepared layer, before the separate **21,037,056-byte** two-slot surface pool;
bounded GPU tiles have similar active allocation but are slower here. This
motivates a selectable lower-workspace boundary, not a whole-VAE/RAM budget.
Sampled process maxima for corrected GPU/NE projection are
1,397,655,544/884,901,832 bytes. They include probe operands/reference/code and
are not whole-VAE/device peaks. The fused standalone control's process footprint
is affected by its extra preflight resources and is not a fair physical-memory
comparison; its timed active peak is reported separately, with no process-memory
win inferred. Internal Core ML copy/placement and hardware overlap are unmeasured.

Adding shuffle to Core ML is slower here even though the compute plan anticipates
NeuralEngine for conv, reshape and transpose. Anticipated placement is not actual
occupancy or proof that more graph fusion is faster. Relative L2 for native/NE
projection is about .00026367/.00026394; both pass. CPU-only also passes at about
.00140731. This is one selected real layer, not compounded VAE quality.

### Preserved preliminary records and baseline correction

Earlier frame0/frame24 complete-layer medians observed NE pipeline23.873/23.277
ms versus GPU24.618/25.358 ms. Those GPU controls unnecessarily retained named
projection/biased intermediates, raising active peak to1,745,881,092 bytes.
They are preserved, **not** used as the final speed/memory baseline. The corrected
control matches the original expression's ownership and eliminates that extra
512-MiB retention. Earlier apparent gains of3–8% and1-GiB active reduction must
not be treated as final qualification. Corrected later-frame replication remains
an independent next gate.

The standalone fused GPU control checks805,306,368 FP16 output words across signed
ABA bitwise against the original operator sequence. Its explicit GPU queue
handoff is not the in-graph native implementation below. Both original/fused
control records remain separate; they are not relabelled after source changes.

### Native in-graph fused bias/shuffle candidate

`metal/upscale.hpp` integrates a separate FP16 bias add and spatial/temporal
shuffle into one native MLX Metal kernel. The GEMM still outputs FP16 first;
bias is never folded into FP32 accumulation. Initial temporal phase1 is skipped
only on the first slice, with all later phases and batches preserved. Thirty-three
bitwise unit cases cover ragged channels, batch/time phases, strided operands
and subnormals. The diagnostic environment
`TURBOCIDER_SEEDVR2_VAE_UPSCALE_RECIPE=metal_fused` selects it **only in an
experimental probe build** (`TURBOCIDER_VAE_UPSCALE_EXPERIMENT=1` when building
codec/spatial probes). Ordinary native/App builds reject this selector rather
than silently falling back. Default remains `native`, with explicit executed-
recipe reporting. No App picker/default or public request field is promoted.

The first native prototype retained a named GEMM tensor through the following
causal convolution, unnecessarily increasing whole-decoder active peak by about
0.5GB. It is not promoted. The corrected expression removes that owner before
convolution. Its complete t25 full/file-row outputs have the same raw export
hashes as their respective original routes; all78,643,200 decoded values are
checked. It passes the independently audited same-latent FP32 VAE reference with
unchanged errors .000522324/.000522439. Single decoder observations are54.920 /
189.959 s, active peaks13,196,954,446/5,647,731,538 bytes; these are **not** balanced
warm speed gates or full requests. The completed1024-image encode+decode ABBA
uses four independent processes, one cold plus six freshly computed warm
iterations each, identical frame history/pointwise/norm/cache settings. Native/
first fused medians are **1.409834/1.463264 s**, identical active peaks
9,683,323,726 bytes; every latent/pixel word is exact. The fusion is **slower**
and fails the speed gate. Replacing its NCHW-style transpose thread organization
with channel-contiguous linear access gives **1.415152/1.456195 s**, still slower
with identical peaks and words. Neither implementation is promoted. The earlier
complete25 quality checks apply to the transpose implementation; the linear
revision has33 bitwise unit cases plus image ABBA; subsequent complete25 raw
full/file-row hashes also match their original routes and pass the same audited
FP32 reference. Those decoder-only observations are55.479/187.734 s with
unchanged active peaks, not speed gates. Actual App/full-request and video warm qualification remain
required. This negative gate is useful evidence, not a completed acceleration.

Records (all generated under `outputs/`, not committed media/models):

- `seedvr2-vae-upscale-real-inputs-t25-v1-20261014/capture.json`: complete25 decode,
  selected actual initial/later inputs; declared latent identity is in the drivers.
- `seedvr2-upscale-real-128-integration-v1-20261014/report.json`: small integration
  crop, **not** a complete512 layer.
- `seedvr2-upscale-real-full512-frame{0,24}-v1-20261014/report.json`: preliminary
  complete-layer records with the GPU intermediate-retention caveat.
- `seedvr2-upscale-real-full512-fused-control-frame0-v2-20261015/report.json`:
  added fused control, before correcting GPU baseline ownership.
- `seedvr2-upscale-real-full512-corrected-frame0-v3-20261015/report.json`:
  use this corrected baseline for the table above.
- `seedvr2-fused-upscale-full1024-t25-v1-20261015/report.json`: first in-graph
  implementation, byte-correct but with extra GEMM retention.
- `seedvr2-fused-upscale-full1024-t25-v2-20261015/report.json`: corrected lifetime,
  explicit execution recipe, complete raw exports and component observations.
- `seedvr2-fused-upscale-full1024-t25-fp32-quality-v2-20261015.json`: independent
  original-FP32 whole-pixel gate; no reference rerun or precision relabelling.
- `seedvr2-fused-upscale-image1024-abba-v1-20261015/report.json`: first fused
  candidate is exact but slower, with no active-peak reduction.
- `seedvr2-fused-upscale-linear-image1024-abba-v2-20261015/report.json`: linear
  revision is also exact/slower; no default promotion or peak claim.
- `seedvr2-fused-upscale-linear-full1024-t25-v3-20261015/report.json`: all25
  raw outputs/full-context phases and row-file mode are byte equal to their own
  original route; no full-request or warm-video speed qualification.
- `seedvr2-fused-upscale-linear-full1024-t25-fp32-quality-v3-20261015.json`:
  separate original-FP32 whole-pixel comparison; both routes pass.

Export tool: `tools/coreml/export_seedvr2_vae_upscale.py`; build native tooling
with `tools/native/build_seedvr2_vae_upscale_probe.sh` into a fresh output.
`seedvr2_vae_upscale_screen.py --help` exposes the bounded complete-layer driver.
Keep the Core ML and native fused choices experimental until their individual
whole-VAE/App/latency/ownership gates pass; the overall goal remains open.

## GPU/ANE cooperation plan and acceptance

### Complete-context VAE projection experiment (2026-10-15)

The projection-only two-slot executor now runs **inside the real VAE decoder**,
not just on a captured layer. Only `up_blocks.2.upsamplers.0.upscale_conv` is
offloaded; its GPU consumer reproduces the original separate bias rounding and
spatial phase ordering. The following causal convolution, global norms,
attention and histories retain the original implementation. The
executor is compiled only into isolated experimental tooling. Ordinary builds
have no hook or selector; stable packaging rejects the experimental build flag
and actual projection shader/recipe bytes. No public request/App option is added.

Native implementation: `tools/native/seedvr2_vae_ane_projection.mm`, using the
public IOSurface ownership helper. Each complete output frame uses a fresh owned
MLX activation and the two reusable nonaliasing Core ML slots. Producer completion
precedes prediction; slot reuse waits for its consumer. GPU submissions retain
both MLX and surface owners and drain on exceptional unwinding. A nonconcurrent
executor guard, typed cancellation before/after actual predictions and explicit
frame/call/backing counters fail closed. Output poisoning and completed pixel
checks detect incomplete frame assembly. The boundary includes input materialization,
producer completion, poisoning, prediction and consumer completion; it is not
isolated ANE arithmetic timing or exclusive whole-decoder phase attribution.

The first integration attempt is preserved as failed: checking a surface-backed
`MLMultiArray.dataPointer` locks its pixel buffer and prevents the next output
backing reuse. The corrected executor never accesses that property. It verifies
pixel-buffer identity and exact dtype/shape/strides, and addresses the shared
Metal allocation through its separately validated IOSurface VM mapping. This
still does not prove that Core ML performs no internal copies.

The complete128/t9 CPU+NE and CPU-only controls pass the unchanged saved-pixel
gate (relative-L2 .000177968/.000456990), with9 actual predictions/backing hits
each. They are single diagnostic observations, not speed comparisons; different
process warmup/cadence must not be interpreted as a small-image acceleration.
The native unit test separately checks original-coefficient FP32 projection
quality under signed ABA, strided/offset inputs, retained-output independence,
invalid operands, pre-cancellation, cancellation after a real prediction,
same-runtime recovery, an actual VAE sink failure, complete9-frame same-VAE
recovery and zero MLX active allocations after all test tensors leave scope.

Complete restored1024/t25 observations use the same probe binary, original
weights/latent and history/cache settings, sequential independent processes:

| Complete decoder observation | GPU native | CPU+NE spatial projection |
| --- | ---: | ---: |
| Seconds, including host pixel scan/export | 55.045347458 | 55.362970208 |
| Maximum MLX active bytes | 13,129,845,582 | 13,129,845,582 |
| Maximum sampled process footprint bytes | 14,654,178,528 | 14,705,100,096 |
| Actual projection frames / Core ML calls / backing hits | 0 / 0 / 0 | 25 / 1600 / 1600 |

All78,643,200 output values are finite and checked. The CPU+NE accumulated
projection boundary is .657838873 s, with21,037,056 external surface bytes.
Against the independently audited **same-latent all-FP32 VAE**, native/CPU+NE
relative-L2 is .000522323646/.000522323592. Both also pass the unchanged gate on
**each of25 frames**; worst frame relative-L2 is .000609564/.000609726. Native
export retains its qualified SHA
`6c71226c6ca815422e3a881e0757a9cdd42799f72ca20b408d89a0790018ddec`;
the CPU+NE export SHA is
`4a489c65df318bc8e483c71cfd221c19d604a8069b81de1919383a25712283f7`.
CPU+NE is numerically qualified for this decoder fixture, **not byte identical**.

This establishes a complete-decoder execution/quality path, not a useful
production acceleration: single observations are effectively speed-flat,
maximum active allocation is unchanged and sampled footprint increases.
The earlier512-MiB **isolated-layer** reduction does not reduce this decoder's
maximum. Do not promote this route, recommend it for low-memory machines, lower
admission estimates or claim full-request/App/physical ANE qualification. Phase
attribution and broader media/hardware/speed qualification remain open. Next
work should address the actual request/decoder peak and critical path rather
than optimizing this projection in isolation indefinitely.

Records under `outputs/`:

- `seedvr2-vae-ane-full-128-t9-ne-v1-20261015/report.json`: preserved failed
  backing-reuse attempt, not a passed gate.
- `seedvr2-vae-ane-full-128-t9-{ne,cpu}-v3-20261015/report.json`: corrected
  complete9 controls, actual prediction/backing coverage and raw exports.
- `seedvr2-vae-ane-full-real1024-t25-{native,ne}-v1-20261015/report.json`:
  complete25 decoder observations, source/bank/probe identities and exports.
- `seedvr2-vae-ane-full-real1024-t25-{native,ne}-fp32-quality-v1-20261015.json`:
  same-input audited-FP32 complete and per-frame gates.

Build into a fresh `TURBOCIDER_VAE_ANE_DECODE_OUT` using
`bash tools/native/build_seedvr2_vae_ane_decode_probe.sh`. The spatial screen's
`--projection-manifest`, `--projection-control ne|cpu` and `--routes baseline`
select this explicit offline experiment. Its driver validates original bank
metadata and the actual compiled tree before/after execution and requires all
frames/tiles/backing hits. It is not an App setting or a full-VAE ANE conversion.

### Remaining cooperation gates

The full-output 64-square hot convolution now has the negative full-boundary
result above. Prioritize other layers/boundaries and the 256->1024 projection
before considering any production convolution route. The projection follow-up
above now narrows that opportunity to allocation reduction at a speed-flat
boundary; it does not support default ANE migration or adding shuffle indiscriminately.
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
