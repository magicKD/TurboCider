# Qwen Image 2.1: first-step target-token-only W8A8 screen (rejected)

The normal 512² W8A8 hybrid runs BF16 GPU for the whole first-step prefill
and W8A8 Core ML/BF16 GPU channel-parallel FFNs for each subsequent decode
step. A temporary explicit candidate split the **first-step FFN along token
sequence** in all 32 blocks: 12,418 reference/text prefix rows still ran
the complete BF16 GPU FFN; only the 1,024 *output image* rows used the
existing 6,144-channel/1,024-row W8A8+BF16 GPU FFN. The attention before
the split and every cross-reference dependency remained unchanged. This
added 32 Core ML calls per request, without prefix tiling or quantizing the
reference rows. Later decode steps used the ordinary hybrid route.

The candidate was gated to explicit W8A8, 512², 1–3 full-size editing
references, BF16 GPU suffix, 32/32-layer manifest, approximation opt-in and
no LoRA or cross-request prefix KV. It was never a default. The matched
M4 Max, three-1024px-reference, seed-17, five-step resident Session request
used a separate prepare and two-step warmup per arm; the cached request
included VAE/PNG and equally enabled tensor dumping. The verifier checked
identical text/noise/each reference latent payload, checkpoint identity,
Core ML geometry and 128 baseline versus **160 candidate predictions**.

| Route | Request wall | RGB correlation / RMSE ([0,1]) |
| --- | ---: | ---: |
| Decode-only W8A8 baseline | 24.1557 s | baseline |
| Target-token-first-step W8A8 | 24.9480 s | 0.98183 / 0.06601 |

The candidate was **about 3.3% slower** in this single matched run. Both
teapots are still visible, but the central sticker's face, outline and lower
body have additional obvious ghosting; it is not an acceptable quality
improvement over an already imperfect five-step baseline. The probe completed
the cancellation, GPU switch, image export and unload lifecycle (exit 0).
This does not establish actual physical ANE occupancy: Core ML was configured
for `cpuAndNeuralEngine`, and its prediction API timings overlap GPU work.
The prefix FFN remains almost all of the first-step rows, while the split
adds per-block GPU/ANE synchronization and target/prefix concatenation; that
explains why this layout has limited theoretical upside, but the precise
cause of the measured slowdown was not separately traced.

The candidate code, CLI admission and report label were removed after the
failed speed/visual screen; the original faster decode-only W8A8 route
remains. No 40-step run or 1–2-reference quality claim follows. The PNG,
JSON, tensor dumps and verified report are ignored under
`results/qwen21/session-fullref-edit3-target-prefill-5-20260926/` and
`results/qwen21/session-fullref-edit3-target-prefill-5-comparison-20260926.json`.

After rollback, the native library and matching resident probe rebuilt,
`make test-qwen21` (27 + 7 + 14 checks) and `git diff --check` passed. A
fresh ordinary five-step hybrid request took 24.205 s, used the expected
128 decode-only Core ML predictions, passed cancellation/GPU-switch/unload
with exit 0, and exported a PNG byte-identical to the pre-experiment
decode-only control (SHA-256
`fecbea96e8d5285d6a266c33468bf796c220cec16755ae0023bed48ed4d4f248`).
The ignored post-rollback receipt is in
`results/qwen21/session-fullref-edit3-post-target-rollback-5-20260926/`.
