# SeedVR2 VAE: direct-frame Core ML causal convolution

## Implemented experimental boundary (2026-10-15)

The isolated executor replaces only subsequent-frame
`decoder.up_blocks.3.resnets.0.conv1`: original256 input /128 output channels,
three temporal taps and3x3 spatial coefficients. The first frame remains the
original GPU FP32 folded convolution. Global normalization, every other layer,
temporal partitions and the following residual path remain native. This is not
a full-VAE ANE backend or a production/App configuration.

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
403,177,740-byte active reduction is the stronger result. Full-request/App
performance, broader media/machines, a qualified public bank and physical ANE
occupancy are still separate gates. No default/public VAE routing is promoted.

## Records and next gates

All generated records are under ignored `outputs/`:

- `seedvr2-vae-virtual-causal-128-t9-{ne,cpu}-v1-20261015/report.json`
- `seedvr2-vae-virtual-causal-real1024-t25-{native,ne}-v1-20261015/report.json`
- `seedvr2-vae-virtual-causal-real1024-t25-ne-fp32-quality-v1-20261015.json`
- `seedvr2-virtual-causal-decode-real1024-t25-abba-v1-20261015/`: preserved
  pre-processing crash, not a passed cohort.
- `seedvr2-virtual-causal-decode-real1024-t25-abba-v2-20261015/report.json`:
  corrected complete25 cohort; only terminal `complete` is acceptance evidence.

Ordinary-build regression also passes a newly built native library's14 public
contracts and release flag/byte/link checks. The unchanged, hash-bound Swift
frontend loads that library for seven real App requests;512/1024 image/video
exports match the prior sharing package on every frame and exact PNG alpha.
Receipts: `seedvr2-virtual-causal-ordinary-app-regression-v1-20261015/generation-report.json`
and `seedvr2-virtual-causal-ordinary-app-export-equality-v1-20261015.json` under
`outputs/`. This is ordinary-build regression, **not ANE App qualification**.

Build isolated tools with `build_seedvr2_vae_causal_conv_probe.sh` and a fresh
`TURBOCIDER_VAE_CAUSAL_PROBE_OUT`. Ordinary builds contain no callback/selector;
release packaging rejects the experiment flag and actual shader/guard marker.
The current public App remains the earlier history-sharing package. Useful
whole-request timing/peak and complete App/media/lifecycle qualification still
precede any VAE ANE promotion. Broader encoder/operator scheduling, real low-
memory hardware and bounded complete-context DiT beyond121 remain open; this
boundary does not complete or narrow the overall optimization objective.
