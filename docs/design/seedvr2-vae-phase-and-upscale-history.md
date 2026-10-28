# SeedVR2 VAE phase residency and shared spatial-upscale history

## Evidence and scope (2026-10-15)

Complete restored1024/t25 decoder diagnostics retain the existing eval/fence
boundaries, add synchronization, and never retain extra Tensor owners. All
78,643,200 output words remain byte exact. The observed maximum is the same
13,129,845,582-byte MLX active maximum as the noninstrumented control. The
471-row trace is checked for complete stage/slice/frame coverage, monotonic
high waters and deduplicated history accounting. These diagnostic timings are
not matched warmed full-request speed measurements. Inclusive mid-block rows
overlap their children and are excluded from time sums.

At output frame2, `up_blocks.2`'s complete spatial projection/shuffle/causal-conv
boundary reaches that maximum. Entering it, cached history buffers have
**9,060,745,216 deduplicated allocator bytes**, and active allocation is
10,445,491,022 bytes. Histories remain the dominant persistent decoder storage.
Their buffer extents can alias current inputs/outputs; they are not disjoint
physical-memory categories and cannot be added to active/process peaks.

Largest accumulated diagnostic regions:

| Original-boundary bundle | Calls | Completed diagnostic seconds |
| --- | ---: | ---: |
| Spatial256->1024-channel projection + shuffle + following causal conv | 25 | 8.073 |
| Last temporal projection + shuffle + following causal conv | 13 | 6.781 |
| Final stage first residual norm1 +256->128 causal conv1 | 25 | 4.514 |
| Previous stage first residual norm1 +512->256 causal conv1 | 25 | 3.921 |

The previous Core ML projection-only boundary totals roughly .658 s in a
complete25 decoder. It is not the dominant combined boundary above. Migrating
only that projection did not reduce the complete decoder maximum. These
measurements redirect work toward history/halo/workspace residency and the
following causal convolutions; they do not prove exclusive arithmetic attribution.

## Exact ownership change

The original spatial-only upsampler produces one complete FP16 frame by the
existing matmul, separate bias add and shuffle. Unlike temporal upsampling, it
has no phase crop that could pin a larger temporal parent. The new option
materializes that known producer before temporal concatenation and lets the
`frames` history store retain its owner instead of making another full-frame
copy. Contiguity alone is not considered proof: the actual allocation extent
must also fit the complete logical frame plus page padding. No coefficients,
casts, bias-rounding boundaries, normalization, phases or causal operands change.

Three same-binary diagnostic controls show:

| Recipe | Complete decoder active maximum (bytes) | Single seconds including host scan/export |
| --- | ---: | ---: |
| Original | 13,129,845,582 | 55.488 |
| Materialization only | 13,129,845,582 | 55.518 |
| Materialization + known-owner sharing | 12,996,152,418 | 55.279 |

Materialization alone does not establish a peak benefit. Sharing lowers the
upsampling region by exactly512 MiB, but the highest region then becomes the
following residual norm1/conv1. **Complete decoder reduction is133,693,164 bytes
(about127.5 MiB), not512 MiB.** All25 outputs retain the original raw SHA
`6c71226c6ca815422e3a881e0757a9cdd42799f72ca20b408d89a0790018ddec`.
The independently audited same-latent all-FP32 VAE comparison passes every
frame and the complete tensor, unchanged relative-L2 .000522323646.

## Public opt-in and App evidence

`seedvr2_vae.share_upscale_history` is a strict Boolean, **false by default**.
It requires `history=frames`, `convolution=singleton2d`, `spatial=full` and the
stable Metal norm path. Unsupported combinations fail rather than silently
select another schedule. Encoder-head rows remain independently configurable.
Images create no unused history; the sharing count is zero for them.

Executed result fields are `vae_share_upscale_history`,
`vae_shared_upscale_frames` and `vae_shared_upscale_logical_bytes`. Frame counts
cover the actual padded decoder frames; a complete25 video counts25. At1024,
13,421,772,800 cumulative logical frame bytes avoid the extra history copy.
**That number is not a physical or peak-memory saving.**

The ordinary, noninstrumented decoder independently passes the complete25
word gate with the same12,996,152,418-byte active maximum. Native tests cover
actual sharing/counters, typed cancellation, sink failure, same-VAE recovery,
source preservation, opt-in/off switching, unused image histories and zero
active allocations after all test objects leave scope. Both public request
schemas expose the typed field and preserve the conservative admission estimate.

Two real seven-request App workflows pass512/1024 images, complete25-frame
24-FPS videos, streaming input/output, staged/resident switches and persistence.
Enabled versus disabled exports match every decoded frame and exact PNG alpha.
The new disabled exports also match the prior qualified App. Same-build single
staged1024/t25 requests observe **90.375/90.053 s** and active maxima
**13,148,195,662/13,014,502,498 bytes** (disabled/enabled). They are not warmed
speed qualification, physical hard bounds or proof of16-GiB operation. The
option remains manually selected/default-off; no speed gain is advertised.

The App has an explicit sharing control. Its user-applied memory advice selects
staged residency, streaming video input/output,1-GiB cache and frame history;
sharing is selected only for supported full/singleton video settings. Existing
row/workspace choices are preserved, and images select buffered input/output
without unused sharing. Advice is heuristic, not automatic RAM admission.

Independent ad-hoc signed package:
`dist/seedvr2-upscale-history-20261015/TurboCider.app` (minimum macOS26.2).
It passes signature/release guards and14 contracts against its actual packaged
library. A packaged-library complete512/t25 video smoke executes25 shared
frames and preserves25 frames/24 FPS; its decoded RGB matches the tested App.
The first observer attempt failed before model execution because the relocated
bundle has no adjacent build manifest. The separate successful receipt supplies
the exact manifest and verifies its ID against the actual loaded library; no
identity check is waived and the original preflight failure is preserved.
Existing running/default Apps are not replaced. The packaged smoke is not a
new1024/low-memory/warm-performance or GUI-click qualification.

## Reproduction and remaining work

Build the diagnostic observer into a fresh directory with
`bash tools/native/build_seedvr2_vae_phase_probe.sh`. Phase profiling is guarded
by an isolated compile flag and stable packaging rejects its flag/actual marker.
`seedvr2_vae_phase_summary.py --screen REPORT --output FRESH_SUMMARY` validates
and summarizes complete traces. Ordinary spatial probes accept the screen's
`--share-upscale-history --routes baseline` option without diagnostic compilation.

Records, generated under ignored `outputs/`:

- `seedvr2-vae-phase-full-real1024-t25-v1-20261015/{report,summary}.json`
- `seedvr2-upscale-lifetime-{native,materialized,owned}-real1024-t25-v1-20261015/`
- `seedvr2-upscale-history-public-real1024-t25-v1-20261015/report.json`
- `seedvr2-upscale-history-public-real1024-t25-fp32-quality-v1-20261015.json`
- `seedvr2-upscale-history-{shared,default}-app-e2e-v1-20261015/generation-report.json`
- `seedvr2-upscale-history-app-export-equality-v1-20261015.json`
- `seedvr2-upscale-history-packaged-video512-smoke-v2-20261015/report.json`

The next useful ANE test should consume separate old/current frame owners and
pack temporal/spatial halos directly into bounded tile surfaces, not retain a
full pre-concatenated3-frame operand as the earlier prepared-layer probe did.
The new limiting norm1/conv1 is a candidate; keep global statistics and initial
FP32 folded frames on GPU first. Actual complete-VAE quality, memory, timing,
ownership and App qualification are still required before an ANE route is
promoted. Broader operator/encoder scheduling, real low-memory hardware and
bounded full-context DiT for source clips beyond121 remain incomplete. This
milestone does not complete or narrow the overall optimization objective.
