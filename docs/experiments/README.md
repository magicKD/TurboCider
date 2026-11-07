# Experiment archive

This is the entry point and canonical destination for TurboCider experiment
records. Experiments explain what was measured, including failed or inconclusive
attempts. They do not by themselves establish supported product behavior.

## Find an existing record

| Topic | Record | Scope |
| --- | --- | --- |
| LTX Stage-2 sparse attention | [Results](ltx-stage2-sparse/results.md) · [Research ledger](ltx-stage2-sparse/research.md) | M4 Max; core, stage and full-request timing; quality failures retained |
| Z-Image Metal 历史实验 | [记录与证据边界](z-image-metal/README.md) | 2026-09 M4 Max 重启前无效计时、健康状态对照与独立 M4 Pro 四图 screen |
| SeedVR2 residency and VAE scheduling | [Records and evidence boundaries](seedvr2/README.md) | 2026-10-14–15 residency, row/file scheduling, Core ML/ANE feasibility, causal convolutions and norm packing; component, request and App scopes kept separate |
| M1 Pro streaming | [Implementation and experiments](2026-09-20-m1-streaming.md) · [2026-09-21 checkpoint](2026-09-21-streaming-current-status.md) | FLUX.2 Klein 4B and Z-Image; adjacent dated evidence belongs to this historical campaign |
| M4 Pro streaming integration | [Integration history](2026-09-27-streaming-integration-history.md) | App/worker lifecycle, memory qualification and rejected campaigns |
| Transformer CPU/GPU/ANE splitting | [M4 Pro report](transformer/m4-pro/REPORT.md) · [M4 Max reproduction](transformer/M4_MAX_REPRODUCTION.md) | Separate devices and workloads; archived reports, protocol and provenance |
| GPU/ANE adoption across models | [Historical analysis](transformer/2026-09-08-adoption-report.md) · [2026-09-15 decisions](transformer/2026-09-15-ane-decisions.md) · [Encoder prefill](transformer/2026-09-15-encoder-prefill.md) · [Current routing entry](../status/acceleration.md) | Historical component, stage and full-request evidence; current routing guidance remains separate |
| GGUF / ConvRot quantized execution | [Implementation and experiment ledger](quantized-execution/README.md) | Historical CPU/GPU/ANE component and full-request observations; quality, speed and memory limits retained |
| Quantization and streaming design | [Quantized execution](../design/quantized-execution/README.md) · [Streaming](../design/streaming/README.md) | Proposals, contracts and links to their validation records |
| Qwen-Image-2.1 frozen W8A8 | [Records and evidence boundaries](qwen21-w8a8/README.md) | M4 Max component, generator and resident Session comparisons; 512²/1024² and reference-size limits kept separate |
| Native LoRA implementation and experiments | [Six 2026-10-08 records](native-lora/README.md) | M4 Pro component, real-FFN and complete-request scopes; negative speed results and budget stop retained |
| Qwen edit encoder and prefill | [Six 2026-10-08 records](qwen-encoder/README.md) | M4 Pro GPU/ANE and Native Metal experiments; component, full encoder, edit step and cold/warm request boundaries retained |
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
