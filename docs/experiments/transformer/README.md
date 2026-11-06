# Transformer heterogeneous execution experiments

These records describe separate devices, workloads and campaigns. Their
measurements must not be pooled into one performance claim.

| Campaign | Record | Runnable sources |
| --- | --- | --- |
| M4 Max: LTX replay, VAE lifecycle and GPU/ANE experiments | [Reproduction ledger](M4_MAX_REPRODUCTION.md) | [Transformer laboratory](../../../experimental/transformer/README.md) |
| M4 Pro: synthetic 1–3 layer Transformer splitting | [Historical report](../../../experimental/transformer/notes/REPORT.md) · [Protocol](../../../experimental/transformer/notes/PROTOCOL.md) | [Transformer laboratory](../../../experimental/transformer/README.md) |
| Cross-model adoption background | [2026-09-08 technical report](../../design/transformer-heterogeneous-report.md) | See each experiment's recorded source revision |

The M4 Max ledger was moved here without changing its observations. It includes
negative results, limited sample counts and unfinished follow-up work. Its
references to local output files identify historical evidence; those files are
not all included in the repository.

The M4 Pro report, template, figures and report-generation scripts remain
together at their original location pending a separate migration. Do not copy
them here and maintain a second version. New narrative records belong in this
archive; research executables remain under `experimental/transformer/` and new
raw output belongs under the ignored root `outputs/` directory.
