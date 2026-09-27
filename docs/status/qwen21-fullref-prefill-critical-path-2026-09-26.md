# Qwen Image 2.1: three full-size references, short-step critical path

The explicit diagnostic `TURBOCIDER_QWEN21_PROFILE_STEPS=1` prints one JSON
line per denoise step to stderr. The timer surrounds the same latent eval and
finiteness check as the regular Session; profiling adds only clock reads,
Core ML metric aggregation and stderr writes **after** step completion.
It is off by default. `coreml_prediction_api_seconds` is accumulated within
Core ML's prediction API and overlaps concurrent GPU work: **never subtract
it directly from step wall**. The flag does not trace hardware placement.

M4 Max, native BF16 checkpoint, 512² output, three ordered 1024px references
(12,288 reference tokens), seed 17, five steps. Each independently prepared
resident Session performed a two-step warmup, then one matched prompt-cached
request. The W8A8 hybrid used the 6,144-channel / 1,024-row, full 32-block
checkpoint-matched manifest; this full-reference experiment requires
`TURBOCIDER_QWEN21_FULL_REF_W8A8_DIAGNOSTIC=1`. No cross-request KV reuse,
GPU final-FFN reuse or experimental Q/K fusion was enabled.

| Phase | BF16 GPU step wall | W8A8 Core ML + BF16 GPU step wall | Hybrid Core ML prediction API |
| --- | ---: | ---: | ---: |
| First full-prefix step | 18.044 s | 18.051 s | 0 s |
| Decode step 2 | 1.615 s | 1.407 s | 0.425 s |
| Decode step 3 | 1.617 s | 1.363 s | 0.349 s |
| Decode step 4 | 1.617 s | 1.397 s | 0.355 s |
| Decode step 5 | 1.614 s | 1.346 s | 0.352 s |
| Total denoise / request wall | 24.507 / 25.132 s | 23.563 / 24.258 s | — |

In this single matched run, the first step consumes ~74% of the pure GPU
denoise wall and is unchanged by the decode-only FFN offload. The four
later steps each save ~0.21–0.27 s, but their total saving is only ~0.94 s
against an 18 s prefill. Compared with the 25.132 s GPU wall, 1.3× would
require a hybrid wall at or below 19.33 s: at the current 18.05 s prefill,
only ~1.28 s would remain for **all** four decode steps, VAE and export.
Reducing ANE prediction time alone, without shortening this GPU prefill,
cannot plausibly reach that short-step target.

The resident probe checked 128 W8A8 Core ML calls, zero output copies and
normal cancellation/switch lifecycle. These profile requests did not dump
all input tensors, but used the exact same request metadata, ordered paths,
seed and checkpoint as the byte-matched tensor-dump experiments in
`qwen21-w8a8-fullref-512-diagnostics-2026-09-26.md`; do not call this a
new byte-for-byte input verification. The raw PNG/JSON receipts are ignored
under `results/qwen21/session-fullref-edit3-steps-{gpu,hybrid}-20260926/`.
The old five-step three-reference images show ghosting in both arms and do
not establish final edit quality.

Priority: profile **production, uninstrumented** first-step prefill by
segment and block, then compare exact GPU improvements to long-prefix
attention and prefix FFN. First-step ANE offload would require new
checkpoint-matched row geometries and hundreds of model calls or a different
tiling/streaming scheme; the current 1,024-row decode artifact does not
accelerate those 12,288 reference rows. Existing block-0 synchronized op
timers identify candidates but cannot be multiplied by 32 to predict wall.
