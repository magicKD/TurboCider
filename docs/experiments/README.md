# Experiment archive

This is the entry point and canonical destination for TurboCider experiment
records. Experiments explain what was measured, including failed or inconclusive
attempts. They do not by themselves establish supported product behavior.

## Find an existing record

| Topic | Record | Scope |
| --- | --- | --- |
| LTX Stage-2 sparse attention | [Results](ltx-stage2-sparse/results.md) · [Research ledger](ltx-stage2-sparse/research.md) | M4 Max; core, stage and full-request timing; quality failures retained |
| M1 Pro streaming | [Implementation and experiments](2026-09-20-m1-streaming.md) · [2026-09-21 checkpoint](2026-09-21-streaming-current-status.md) | FLUX.2 Klein 4B and Z-Image; adjacent dated evidence belongs to this historical campaign |
| M4 Pro streaming integration | [Integration history](2026-09-27-streaming-integration-history.md) | App/worker lifecycle, memory qualification and rejected campaigns |
| Transformer CPU/GPU/ANE splitting | [M4 Pro report](transformer/m4-pro/REPORT.md) · [M4 Max reproduction](transformer/M4_MAX_REPRODUCTION.md) | Separate devices and workloads; archived reports, protocol and provenance |
| GPU/ANE adoption across models | [Historical analysis](../design/transformer-heterogeneous-report.md) · [Current routing entry](../status/acceleration.md) | Historical component evidence and maintained routing guidance |
| Quantization and streaming design | [Quantized execution](../design/quantized-execution/README.md) · [Streaming](../design/streaming/README.md) | Proposals, contracts and links to their validation records |
| Native Metal / Qwen command graph | [Design and evidence index](../design/qwen-image-gpu-ane-command-graph/README.md) | Experimental backends and qualification boundaries |
| Published model comparisons | [Benchmark index](../benchmark/README.md) | Curated performance reports with workload and quality limits |
| Older validation summaries | [Validation index](../design/validation/README.md) | Existing JSON evidence remains at its original paths until migrated with its consumers |
| Other implementation and experiment history | [Status index](../status/README.md) · [Design index](../design/README.md) | Legacy records; a dated status is not a current support guarantee |

This is a curated topic index, not an exhaustive list of every evidence file.
Existing status reports and validation JSON are preserved: some are consumed by
tools and tests. Their relocation requires a separate, focused PR that updates
those consumers. Do not copy them here and maintain two versions.

## Add a record

1. Choose `docs/experiments/<topic>/`. For repeated campaigns, add a dated
   `<YYYY-MM-DD>-<device-or-question>/` subdirectory within that topic.
2. Start from [TEMPLATE.md](TEMPLATE.md). Record the hypothesis, exact source
   revision, device, workload, precision, baseline, commands, quality and timing
   boundaries. Use repository-root commands or state the working directory.
3. Keep the narrative in `README.md` or a descriptive Markdown file. Store only
   small, reviewed, portable evidence alongside it in `evidence/`. Put fresh raw
   output in `outputs/<topic>/<run>/`; do not overwrite a historical campaign.
4. Report negative and inconclusive results with the same care as successful
   runs. Distinguish kernel, stage and full-request results, cold and warm runs,
   and observed speed from release qualification.
5. Add an entry above and link the record from the relevant design or status
   page. Publish selected conclusions in `../benchmark/` only when their scope
   and evidence are clear.

Do not commit weights, compiled models, generated media, private prompts,
credentials, raw traces or absolute local model paths in new records. Refer to
local artifacts using a placeholder or an ignored repository-relative path;
record enough identity and methodology to explain the result. Existing curated
historical evidence is not a blanket permission to add new raw artifacts.

## Migration rules

Move one topic per PR; keep its conclusions, provenance, licenses and evidence
intact. Update relative links and any executable consumers before removing an
old path. A forwarding page is appropriate for published Markdown links, but
not for JSON loaded by a script. Historical records may describe older runtime
behavior; add a dated correction instead of silently changing their claims.
