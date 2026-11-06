# Transformer heterogeneous execution experiments

These records describe separate devices, workloads and campaigns. Their
measurements must not be pooled into one performance claim.

| Campaign | Record | Runnable sources |
| --- | --- | --- |
| M4 Max: LTX replay, VAE lifecycle and GPU/ANE experiments | [Reproduction ledger](M4_MAX_REPRODUCTION.md) | [Transformer laboratory](../../../experimental/transformer/README.md) |
| M4 Pro: synthetic 1–3 layer Transformer splitting | [Historical report](m4-pro/REPORT.md) · [Protocol](m4-pro/PROTOCOL.md) | [Transformer laboratory](../../../experimental/transformer/README.md) |
| Cross-model adoption background | [2026-09-08 technical report](2026-09-08-adoption-report.md) | See each experiment's recorded source revision |
| Cross-model stage decisions | [2026-09-15 ANE decisions](2026-09-15-ane-decisions.md) | Historical encoder/DiT/VAE, startup and complete-request boundaries |
| Encoder-prefill qualification | [2026-09-15 optional paths](2026-09-15-encoder-prefill.md) | Recorded build/test commands; H3 reproducibility limits and LTX performance threshold retained |

The M4 Max ledger was moved here without changing its observations. It includes
negative results, limited sample counts and unfinished follow-up work. Its
references to local output files identify historical evidence; those files are
not all included in the repository.

The M4 Pro report, protocol, related work, third-party attribution, CSV and SVG
figures are archived together in `m4-pro/`. Its report template is stored beside
the evidence; the generator reads that template and writes ignored local output
under `experimental/transformer/notes/REPORT.generated.*`. Reviewed historical
reports are not overwritten by a rerun. Raw JSON was never committed and must
be reproduced locally. The redundant generated HTML is no longer tracked.

Research executables remain under `experimental/transformer/`. New campaigns
should use the ignored root `outputs/` directory; the existing historical
drivers retain their local `notes/raw/` paths.

## Unavailable historical report

The separately referenced `coreml-ane-startup.md` report is absent from the
current source tree, as recorded in the [2026-09-08 checkpoint](../../status/current-status-2026-09-08.md#本轮架构与发行边界收口).
Its link in the archived ANE decisions points to this availability note.
No report text or original evidence was recreated. Current artifact setup is
covered by the [CLI guide](../../public/USAGE.md#core-ml-artifacts).
