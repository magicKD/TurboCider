# Qwen Image 2.1: full-reference first-eight FFN tiling (rejected)

This was a temporary **first-request** experiment, not an enabled mode. A
512² edit with three ordered, original-size 1024px references has 13,442
first-step rows. We left the first-step attention and 24 later FFNs on the
BF16 GPU, and fed the **first eight** post-attention FFNs through the same
checkpoint-matched 1024-row / 6144-channel W8A8 Core ML layer graphs in
14 sequential tiles per layer. The four subsequent decode steps continued
to use the existing 32-layer hybrid. The candidate added 112 first-step
Core ML predictions (128 to 240 predictions per five-step request). Its
merged tile outputs had to be materialized before the next prediction reused
the session's output backing. This is a sequence-tile loop, **not** the
parallel, full-FFN token-row partition explored in `../splash`.

On the M4 Max, independent resident Sessions used the same ordered
reference files, seed 17, **full five-step warmup**, BF16 checkpoint, cached
conditioning, and 512² PNG export. The original 24.156 s control had only
a two-step warmup, so it is retained as historical evidence, **not** the
matched-warmup denominator. After removing the candidate we rebuilt the
library and Session probe and ran a fresh flag-off, five-step-warmed control.
Text, initial noise and all three reference-latent safetensors are byte-equal
to the candidate; both used the same calibrated artifact. Warm request wall
includes VAE, export and identical dumps but excludes prepare/load:

| 3 full-size references, five steps | Decode-only hybrid | First-eight tiled hybrid |
| --- | ---: | ---: |
| Wall | 24.226 s | **23.209 s** |
| Ratio versus decode-only | — | 1.044× |
| Core ML predictions per request | 128 | 240 |

The PNG is **not acceptable** for the requested edit. Although both teapots
and the dragon sticker remain recognizable, the candidate gives the sticker
a prominent glowing/doubled edge and face and worsens its object identity.
Normalized RGB correlation/RMSE against decode-only were 0.92589/0.13656;
these metrics are diagnostics, not a semantic quality gate. Only one seed
and one five-step request per arm were screened. The earlier last-eight and
last-sixteen *late* layer full-reference trials also found weak short-step
gains, a 40-step regression or visible fidelity changes; we cannot infer
this first-eight trial's 40-step speed or other image quality from them.

This is why isolated INT8 FFN throughput does not deliver a Splash-like
whole-edit ratio here: 14 predictions per offloaded layer, MLX/Core ML
handoff and output-lifetime barriers, a still-GPU long-prefix attention,
and a small fraction of total wall offloaded. `cpuAndNeuralEngine` records
a policy, **not measured physical ANE occupancy**. Quantizing the earliest
reference representations also affects every later block, unlike keeping
the early blocks exact and quantizing a late suffix. That is a plausible
mechanism for the observed degradation, not a proved attribution.

The temporary implementation, plan label and probe-specific prediction
branch were removed; the existing resized-512 sequence-tiled diagnostic
and conservative full-reference decode-only route are unchanged. Ignored
local evidence is in
`results/qwen21/session-fullref-edit3-prefill-first8-5-20260927-b/`
and `results/qwen21/session-fullref-edit3-prefill-first8-5-20260927-vs-control.json`,
with the control in
`results/qwen21/session-fullref-edit3-tiledprefill-control-5-20260926/`.
The first pilot without matched full-step warmup failed admission before
measurement; it is not a timing result. The measured probe passed
cancellation, GPU route switch and unload. After rollback, the native
library and matching probe rebuilt, `make test-qwen21` (27 + 7 + 22
contracts) and `git diff --check` passed. The new full-warmup control
performed 128 predictions, had no Core ML runtime failures and passed
cancellation, route switch and unload. Its PNG SHA-256 matched the saved
decode-only hybrid control exactly:
`fecbea96e8d5285d6a266c33468bf796c220cec16755ae0023bed48ed4d4f248`.
Its ignored receipt is
`results/qwen21/session-fullref-edit3-first8-rollback-20260927/`.
Further full-reference work
should preserve early reference conditioning and target the exact GPU
attention or reduce bridge cost; retain the known fastest quality-checked
resized-reference path for users explicitly accepting resize and FFN loss.
