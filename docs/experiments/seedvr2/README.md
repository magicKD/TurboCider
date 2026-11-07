# SeedVR2 experiment archive

These records preserve eight 2026-10-14–15 implementation and experiment records, including failed attempts,
source/artifact identities and limits on the conclusions. Archiving them does
not rerun their tests or qualify the current checkout. Product instructions
start at the [SeedVR2 user guide](../../public/SEEDVR2.md); implementation history
is indexed in the [SeedVR2 integration record](../../status/seedvr2-integration.md).

## Records and evidence boundaries

| Record | Measured scope | Conclusion to preserve |
| --- | --- | --- |
| [Verified serial block-weight residency](2026-10-15-block-weight-residency.md) | Original-weight complete DiT component and source/cancellation lifecycle | Lower component residency with a slower single warm observation; compile-isolated, without whole-request/App/low-memory qualification |
| [VAE phases and shared upscale history](2026-10-15-vae-phase-and-upscale-history.md) | Complete decoder, recorded App workflows and packaged smoke | Local sharing savings move the highest allocation phase; the complete decoder benefit is smaller. The recorded public option stays opt-in, with no warmed speed or physical hard-budget claim |
| [Halo-aligned rectangular tiles](2026-10-15-vae-halo-aligned-tiles.md) | Two independent complete-decoder ABBA comparisons | Smaller IO pools or fewer predictions do not reduce the MLX active maximum; neither candidate is promoted to App defaults |
| [Row/file VAE scheduling](2026-10-14-vae-spatial-scheduler.md) | Decoder/encoder residency, source lifetime, App/package observations and read-window reuse | Component memory reductions and exact-context numerical comparisons retain their precision corrections; no balanced whole-request speed or low-memory-device qualification |
| [Fixed-tile Core ML / ANE feasibility](2026-10-15-vae-ane-feasibility.md) | Reusable surface IO, complete convolution/projection boundaries and captured-real corrected GPU baselines | Conditional component opportunities; failed surface binding and preliminary baselines remain visible, with no qualified whole-VAE ANE backend |
| [Direct-frame causal convolution](2026-10-15-vae-virtual-causal-conv.md) | One subsequent-frame residual convolution, balanced decoder cohort and isolated request/App integration | Measured component gains do not qualify ordinary-build ANE routing, public banks or whole-request speed |
| [Spatial-upsample causal convolution](2026-10-15-vae-upscale-causal-conv.md) | Independent upsampling convolution and dual-boundary complete decoder/request observations | Keep operator, full-decoder, raw-media and App gates separate; no default promotion or complete low-memory/long-video qualification |
| [Norm packing and resource advice](2026-10-15-vae-norm-pack-and-resource-advice.md) | Cumulative request high-water observations, full-statistics packing, App advice and residency planning | Phase counters are not additive RAM savings; packing remains isolated and logical state sizes do not establish device admission |

The newly archived VAE records also preserve failed reference/cohort attempts,
precision and GPU-baseline corrections, and outstanding whole-request, App/media
and low-memory qualification gates. Later receipts apply only to their stated scopes.

The block-weight record retains the failed IO-future/cancellation gate before
the corrected source result. The shared-history record retains the initial
packaged-smoke identity failure, and the tile record retains the failed final
aggregation before its corrected audit. Those failures must not be relabelled
as passing gates or combined with the later successful receipts.

Component active allocations, history storage, external IO pools and sampled
process footprints have different scopes and can overlap. Do not add their
savings or turn a single warm observation into a balanced speed qualification.
The original workload, device, precision and reference-output boundaries remain
part of every result. These records do not establish real <=16-GiB operation or
complete model clips beyond the recorded source-frame limit.

## Design context

The [VAE optimization roadmap](../../design/seedvr2-vae-optimization.md) remains
in the design directory. The archived scheduler, IO, convolution and norm-packing
records above preserve their original design rationale alongside the historical
measurements and uncompleted gates; they are not a current support matrix.

## Storage and reproduction

Each narrative has one full canonical copy here. Its former `docs/design/`
page forwards to this archive and preserves the original section headings.
Existing validation JSON, source code, build inputs and model/artifact paths
stay in place; commands should still run from the repository root.

The records describe local `build/`, `outputs/`, `models/` and `dist/` artifacts
that are not guaranteed to accompany a checkout. They are historical identities
and reproduction references, not files created by this migration. Reproduce into
fresh output directories and verify the source, checkpoint and compiled artifact
identities required by the original record. New measurements belong in a new
[experiment record](../README.md), without overwriting these observations.
