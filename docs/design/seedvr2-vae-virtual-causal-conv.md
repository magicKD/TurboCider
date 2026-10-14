# SeedVR2 VAE: direct-frame Core ML causal convolution

## Implemented experimental boundary (2026-10-15)

The isolated executor replaces only subsequent-frame
`decoder.up_blocks.3.resnets.0.conv1`: original256 input /128 output channels,
three temporal taps and3x3 spatial coefficients. The first frame remains the
original GPU FP32 folded convolution. Global normalization, every other layer,
temporal partitions and the following residual path remain native. This is not
a full-VAE ANE backend or a public production/App setting. An isolated
full-request/App integration is described below; ordinary builds reject it.

Two prior independently owned normalized frames and the current frame remain
separate. The GPU packs their exact half words and spatial zero-padding directly
into a bounded Core ML input surface in tap/channel order. No full3-frame
concatenation is created. A two-slot ring overlaps the next producer and previous
consumer with synchronous CPU+NE prediction; slots cannot be overwritten before
their consumers finish. Original separate FP16 output/bias rounding is applied
by GPU assembly. Borders and incomplete last tiles use exact zero padding and
bounded writes, not overlapping crops/blending. Causal history advances only
after all tiles have consumed the old owners.

Public Core ML/Metal/IOSurface APIs only. Predictions must return the exact
requested pixel-buffer identity, dtype, shape and strides. MLMultiArray CPU
pointer access is deliberately avoided because it locks reusable output backing.
GPU submissions retain all source/output/surface owners and drain during
exceptions. A nonconcurrent executor guard and typed cancellation bracket every
actual prediction. NaN-poisoned fresh outputs expose incomplete assembly.

## Completed correctness and isolated observations

The original full-output64-square compiled bank is reused unchanged and remains
`qualified=false`. Its original checkpoint/compiled-tree identities are checked
before/after each screen. Unit QA checks120,434,688 packed half words and
11,857,664 assembled/bias words independently, plus original-coefficient FP32
convolution under signed ABA. Ragged17x19/65x71 and64-square cases, strided and
contiguous-offset inputs, replicated history aliases, retained-output safety,
invalid operands, pre/active cancellation after a genuine prediction,
same-runtime recovery, actual VAE sink failure/recovery and zero-allocation
teardown pass. CPU-only complete9 control also passes the unchanged pixel gate.

Same-binary sequential complete restored1024/t25 screens (sharing enabled):

| Complete decoder observation | GPU native | CPU+NE direct-frame hotconv |
| --- | ---: | ---: |
| Seconds including host pixel checks/export | 55.505209875 | 54.920405542 |
| MLX active maximum bytes | 12,996,152,418 | 12,592,974,678 |
| Maximum sampled process footprint bytes | 14,520,190,056 | 14,058,341,600 |
| Actual subsequent frames / predictions / backing hits | 0 /0 /0 | 24 /6144 /6144 |

There are21,626,880 external surface bytes, excluded from MLX active accounting.
The CPU+NE measured boundary totals4.015486458 s. Active maximum falls
403,177,740 bytes versus the already-shared GPU route. These are component,
single observations, not paired warm/full-request/App/hard-budget qualification.
Physical ANE occupancy and actual hardware overlap remain unmeasured.

All78,643,200 values pass the saved-native finite/.05-relative-L2/.999-cosine
gate; the new output is not byte identical. The independent same-latent audited
all-FP32 VAE gate passes on the whole tensor and each of25 frames:
relative-L2 .000525150398, cosine .9999998670. The first frame's raw FP16 SHA
matches native exactly:
`c82dd778672454b6001e58eb0a597e80b7fdac5d24513f15653bcf1ad8a27617`.
Repeated benchmarks additionally require zero Core ML calls during that first
frame and exact original first-frame pixels in every fresh iteration.

## Completed balanced component gate

A dedicated decoder benchmark recomputes the entire clip on every iteration;
only weights, compiled model/kernels and allocator cache remain warm. Every
pixel is checked per iteration, all frame/tile counts must match, and original
latent bytes must remain unchanged. An ABBA driver binds input/checkpoint/
compiled/executable identities and separates cold from warm observations.

The first complete25 cohort failed in host pre-processing before decoder
timing: a latent snapshot used a half-word-sized vector although the restored
latent is FP32. Its native crash/failed cohort remain preserved. The corrected
benchmark allocates the original byte extent and keeps the FP32 input unchanged;
no lower-precision input or replacement reference is substituted. The successful
small9 cohort is a harness smoke, not1024 speed acceptance. Complete25 balanced
acceptance must be read from the final cohort receipt before claiming a speed
gain or promoting the backend.

The corrected complete25 ABBA cohort has now completed four independent
processes, one cold and two fresh warm decoder iterations per process. All
12 complete decodes check every output value, preserve the original input bytes
and verify the original GPU initial frame with zero new Core ML calls.
There are four warm observations per route:

| Warm complete decoder component | GPU shared-history baseline | CPU+NE virtual halo |
| --- | ---: | ---: |
| Warm median seconds including host pixel gate | 55.459255625 | 55.189463709 |
| Maximum active bytes | 12,996,152,418 | 12,592,974,678 |

The component speed gate passes by about0.49%, which is a small cadence/order-
sensitive difference, not a claim of substantial acceleration. The stable
403,177,740-byte active reduction is the stronger result. Balanced full-request
performance, broader media/machines, a qualified public bank and physical ANE
occupancy remain separate gates. No default/public VAE routing is promoted.

## Isolated complete-request and App integration

An explicit experimental native build now accepts
`TURBOCIDER_SEEDVR2_VAE_CAUSAL_MANIFEST` and the optional
`TURBOCIDER_SEEDVR2_VAE_CAUSAL_COMPUTE=ne|cpu`. It requires the explicit GPU
backbone, <=1024 canvas, singleton2d/full spatial/frame history settings. The
original GPU plan and DiT execution are not silently changed. The exact original
manifest SHA and every compiled-tree file SHA/size are verified before/after
requests; symlinks, unregistered files and a changed bank fail closed. The
executor/slots can be reused across requests and reset on selector clearing,
unload or residency changes. This is not a public VAE configuration field.

Actual video results say `gpu_vae_cpu_ne_experimental`, report the numerical
recipe suffix, real frames/calls/backing hits, external slot bytes and boundary
time. Images retain `gpu`, zero predictions and the native recipe; their resident
component list still truthfully includes the loaded unused Core ML bank.
Ordinary libraries reject both the selector and raw-capture diagnostic. Build
identity includes the linked experimental helper sources; library-only builds
now also verify source freshness before returning.

Complete import/resize/encode/DiT/decode/export observations, same experimental
library, source, seeds42/43, staged/cache1, source/sink streaming and existing
upscale sharing enabled:

| Whole-request observation | GPU shared-history | CPU+NE virtual halo |
| --- | ---: | ---: |
| Cold request seconds | 90.802605041 | 90.905491833 |
| Subsequent warm request seconds | 90.231037458 | 90.195729042 |
| MLX active maximum bytes | 13,014,502,498 | 12,611,324,758 |
| Warm sampled process footprint maximum bytes | 14,350,271,712 | 14,274,250,024 |
| Per-request actual calls / backing hits / subsequent frames | 0 /0 /0 | 6144 /6144 /24 |

The403,177,740-byte active reduction survives the complete request. Whole-request
speed is effectively flat, not an ABBA speed qualification: there is one cold
and one warm observation per route, not a balanced multi-process cohort. The
21,626,880-byte external surface pool and internal Core ML allocations are not
included in MLX active. Sampled process footprint varies by request/caching and
does not establish a403-MB physical-RAM saving or a hard device budget. At512,
other phases dominate the request maximum; no substantial peak win is claimed.

The first experimental Swift App workflow completes seven real requests:
512/1024 images and complete25-frame/24-FPS videos, job persistence and
resident/staged/resident switches. Videos execute1536/6144 predictions; images
execute none and match the ordinary App pixels/PNG alpha exactly. Every exported
video frame passes the unchanged .05/.999 gate. This is an App-state/job-store
workflow, not GUI clicking or a distributable/public ANE feature.

After adding capture diagnostics, fresh experimental/ordinary libraries pass
another fourteen actual App requests with the **same SHA-bound Swift frontend**
and no capture enabled. Every experimental export matches the earlier
experimental App on all pixels; every ordinary export matches the earlier
ordinary App. The intended GPU-versus-experimental video differences pass every
frame's existing gate; images and alpha are exact. The1024 App active maxima
again are13,014,502,498/12,611,324,758 bytes. These are functional regressions,
not an additional warm-performance cohort. The ordinary library passes14 public
contracts, actual release byte/flag/link checks, and actual rejection of each
nonempty experiment/capture selector followed by same-engine prepare recovery
and zero-active unload. The complete SeedVR2 CPU/contract suite passes108 tests,
with separate build-identity and release-guard tests also passing.

The full-request lifecycle driver separately rejects an altered **owned copy**
of a compiled bank, returns typed cancellation after output frame2 (therefore
after real subsequent-frame predictions), recovers on the same engine, turns
the selector off for a resident GPU video, and back on for a staged experimental
video. All three successful requests preserve25 frames/24 FPS and verify actual
route/call counts. Source, original bank and a private existing destination are
unchanged; private media staging drains and explicit unload reports zero MLX
active bytes. This is cancellation after real convolution work, not during a
synchronous prediction; the latter remains the separate component test.

### Raw-versus-codec diagnosis

The first complete-request export comparison had maximum8-bit errors138/101
despite passing aggregate and per-frame gates. Those outliers were not dismissed
as codec noise. A new compile-isolated streaming diagnostic captures each actual
FP16 THWC frame and the exact RGB24 bytes handed to the existing encoder. It
retains no frame owners after the sink returns, binds every file/hash/extent,
records the actual condition/velocity/latent hashes and checks fresh output plus
available scratch space. `--capture-raw-video` requires one explicit streaming
video request; its timings/peaks include diagnostic IO and are NOT benchmarks.

Complete1024/t25 seed42 matched requests now establish:

- Condition (FP16), velocity (FP32) and restored latent (FP32), including shape
  and byte extent, are byte-identical between routes.
- Every raw/quantized frame is checked; the first frame remains bitwise equal.
  Whole raw FP16 relative-L2 is .0002050380613, cosine .99999997898 and maximum
  absolute error .0048828125, versus the stable native output.
- Actual preencoder RGB24 maximum difference is **1**, MAE .005832914 and
  PSNR70.471948 dB. Concatenated captured bytes exactly reproduce each request's
  independently recorded preencoder hash. No CPU reimplementation of rounding
  is substituted for the real native quantized bytes.
- H264 decoded maximum difference is **138**, relative-L2 .014485507 and
  PSNR39.501286 dB. Both captured exports match every decoded RGB frame of their
  respective earlier noncaptured requests, despite different container hashes.

Thus this fixture's large exported differences are introduced/amplified after
the tiny preencoder difference, in the lossy encoding path; they are not a
138-level raw VAE error. This does not establish general perceptual/temporal
quality, diagnose a specific encoder implementation bug, or qualify other seeds,
media or devices. The independent original-FP32 component gate remains required;
the raw native comparison complements it rather than replacing it.

Build isolated request libraries with
`TURBOCIDER_BUILD_VAE_CAUSAL_EXPERIMENT=1` and a fresh
`TURBOCIDER_BUILD_OUTPUT_DIR` using `tools/native/build.sh`. The framework observer
requires explicit `--experimental-causal-manifest` to record that route and
rejects unrecorded inherited selectors/captures. Compare captures with
`seedvr2_causal_request_media.py --require-raw-capture` and the two full-request
reports. Release packaging must reject this library, including the actual
capture/shader bytes even if someone supplies a misleading policy manifest.

### Next useful boundary, not yet implemented

The original checkpoint header confirms
`decoder.up_blocks.2.upsamplers.0.conv.weight` is original FP16
`[256,256,3,3,3]`, following the spatial shuffle at final resolution. The original
phase trace identifies its projection/shuffle/conv bundle as an8.073-s region,
whereas projection-only offload covers only about .658 s and does not improve
the decoder maximum. This makes **the following causal convolution**, with
separate histories/direct halo packing, a better next candidate than retesting
projection-only migration. It requires a new full256-output bank, numerical and
ownership tests and per-layer accounting: the current128-output bank/hook
cannot be relabelled or reused as if it computed that layer.

In parallel with that operator choice, an affine/SiLU/conv tile graph could use
externally computed whole-frame FP32 statistics. Preserve all original casts,
initial-frame folding and history ownership; independent per-tile GroupNorm or
fusing across the next unknown global-statistics pass changes the network.
Neither candidate is a measured gain or an implemented full-VAE ANE route.
Before extending Core ML fusion, also screen applying the existing FP32
statistics/affine and explicit FP16 SiLU rounding **inside GPU tile packing**.
That could avoid materializing the current full normalized activation while
leaving convolution on the unchanged Core ML bank. It needs raw-frame plus
per-frame-statistics history ownership, not substitution of tile statistics or
an assumption that old normalized history can simply be discarded. This remains
a design candidate, not an implemented or timed optimization.

## Records and next gates

All generated records are under ignored `outputs/`:

- `seedvr2-vae-virtual-causal-128-t9-{ne,cpu}-v1-20261015/report.json`
- `seedvr2-vae-virtual-causal-real1024-t25-{native,ne}-v1-20261015/report.json`
- `seedvr2-vae-virtual-causal-real1024-t25-ne-fp32-quality-v1-20261015.json`
- `seedvr2-virtual-causal-decode-real1024-t25-abba-v1-20261015/`: preserved
  pre-processing crash, not a passed cohort.
- `seedvr2-virtual-causal-decode-real1024-t25-abba-v2-20261015/report.json`:
  corrected complete25 cohort; only terminal `complete` is acceptance evidence.
- `seedvr2-virtual-causal-full-request-1024-t25-{gpu,ne}-v1-20261015/report.json`:
  noncaptured cold/warm complete-request observations, not a balanced cohort.
- `seedvr2-virtual-causal-full-request-1024-media-v1-20261015.json`:
  preserved initial exported-media comparison, including138/101 maximum errors.
- `seedvr2-virtual-causal-experimental-app-e2e-v1-20261015/generation-report.json`:
  seven experimental App requests; not stable packaging qualification.
- `seedvr2-virtual-causal-experimental-app-media-v1-20261015.json`.
- `seedvr2-virtual-causal-raw-request-1024-t25-{gpu,ne}-v1-20261015/`:
  complete diagnostic request/capture receipts and original raw frames.
- `seedvr2-virtual-causal-raw-request-1024-quality-v1-20261015.json`:
  every actual FP16/preencoder/exported frame and predecode hashes.
- `seedvr2-virtual-causal-raw-capture-export-stability-v1-20261015.json`:
  every decoded frame versus the respective prior noncaptured route.
- `seedvr2-virtual-causal-full-request-lifecycle-v1-20261015/report.json`:
  five actual public-engine cases, including typed cancellation and unload.
- `seedvr2-virtual-causal-experimental-app-e2e-v2-20261015/generation-report.json`
  and `seedvr2-virtual-causal-ordinary-app-regression-v2-20261015/generation-report.json`:
  final fourteen App requests on fresh libraries, capture disabled.
- `seedvr2-virtual-causal-experimental-app-export-stability-v2-20261015.json`
  and `seedvr2-virtual-causal-final-app-export-gates-v1-20261015.json`.
- `seedvr2-virtual-causal-ordinary-selector-rejection-v1-20261015/report.json`:
  actual ordinary-library prepare rejection/recovery, no inference.
- `seedvr2-virtual-causal-request-final-gates-v2-20261015.json`:
  final eight-receipt index, current compiled-source snapshots, actual library
  hashes/build IDs and the identical Swift frontend hash. Identity queries use
  separate processes and have empty stderr. The preserved v1 index emitted
  duplicate Objective-C class warnings by loading both libraries together;
  no inference ran in that index process, and v2 supersedes it. All actual
  request/App/lifecycle runs already used independent processes.

Ordinary-build regression also passes a newly built native library's14 public
contracts and release flag/byte/link checks. The unchanged, hash-bound Swift
frontend loads that library for seven real App requests;512/1024 image/video
exports match the prior sharing package on every frame and exact PNG alpha.
Receipts: `seedvr2-virtual-causal-ordinary-app-regression-v1-20261015/generation-report.json`
and `seedvr2-virtual-causal-ordinary-app-export-equality-v1-20261015.json` under
`outputs/`. This is ordinary-build regression, **not ANE App qualification**.

Build isolated tools with `build_seedvr2_vae_causal_conv_probe.sh` and a fresh
`TURBOCIDER_VAE_CAUSAL_PROBE_OUT`. Ordinary builds contain no experimental executor;
release packaging rejects the experiment flag and actual shader/guard marker.
The current public App remains the earlier history-sharing package. Useful
balanced whole-request performance, broader App/media/lifecycle qualification
and a public bank still precede any VAE ANE promotion. Broader encoder/operator scheduling, real low-
memory hardware and bounded complete-context DiT beyond121 remain open; this
boundary does not complete or narrow the overall optimization objective.
