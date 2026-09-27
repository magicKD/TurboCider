# Qwen Image 2.1: GPU FFN on the short first-step tail (rejected)

For a new three-reference 512² edit with each reference explicitly resized
to 512px, the first-step sequence has 4,226 rows: four full 1,024-row tiles
and one 130-row tail. The existing last-16 first-step W8A8 diagnostic loops
over the same calibrated 1,024-row Core ML models, padding that tail and
trimming the output. A temporary explicit switch instead ran a **full BF16
GPU FFN** on the 130-row tail of the first 15 offloaded layers; the final
block already computes target rows only. The GPU suffix and ANE prefix of
the full tiles, GPU attention, decode FFNs, fused QKV, and two short-step
FFN-reuse switches stayed unchanged. The candidate saved 15 Core ML calls.

M4 Max, independently prepared resident Sessions, the same prompt and
three ordered resized-512 references, seed 17, matching five-step warmup,
two cached requests per arm, equal tensor dumps and PNG export. The
comparison verifier checked byte-equal text, initial noise and all three
reference latents, the manifest identity, prediction deltas, and zero Core
ML runtime failures. The 512² request wall includes VAE/PNG and dump I/O,
but excludes preparation:

| Five-step three-reference request | Padded W8A8 tail | BF16 GPU short tail |
| --- | ---: | ---: |
| Request 1 | 7.706 s | 7.665 s |
| Request 2 | 7.707 s | 7.624 s |
| Median ratio | — | **1.008×** |
| Core ML predictions per request | 172 | 157 |

RGB correlation was 0.99714 and normalized RMSE 0.02482. Inspection found
both teapots and the central sticker recognizable in both outputs, with
visible changes to the sticker eyes/edges and the blue pot; this small
single-fixture gain is insufficient to claim a general quality-qualified
acceleration. A previous distinct short-tail **GPU suffix** attempt was
slower and changed the one-reference output, so these three-reference data
cannot justify enabling the new full-FFN tail for 1–2 references.

The switch, special weights, plan label, call-count branch and temporary
verifier logic were removed. The usual serial sequence-tile loop and its
tested fast combination remain available. Ignored receipts are
`results/qwen21/session-ref512-edit3-gpu-tail-{control,candidate}-20260927/`
and `results/qwen21/session-ref512-edit3-gpu-tail-vs-control-20260927.json`.
Both probes passed cancellation, GPU route switch and unload. The policy
`cpuAndNeuralEngine` and call counts alone do not establish physical ANE
placement; the 0.8% saving is too small to assign to one hardware cause.

After removing the switch, the native library and matching Session probe
rebuilt; `make test-qwen21` (27 + 7 + 22 contracts) and `git diff --check`
passed. A fresh three-reference, five-step warm request made the expected
172 predictions, reported zero Core ML runtime failures, took **7.699 s**,
passed the cancellation/GPU-switch/unload lifecycle and exported a PNG
byte-identical to the padded-W8A8 control (SHA-256
`d2f4ce564ebb059a1858c250a3593afcc4880385b65eed3279c1f6e94850a759`).
Its ignored receipt is
`results/qwen21/session-ref512-edit3-post-gpu-tail-rollback-20260927/`.
