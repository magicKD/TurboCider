# Qwen Image 2.1: batched first-step image-input projection (rejected)

In the first step of multi-reference edits, the GPU projects each ordered
reference image and the output image independently through `img_in`. A
temporary opt-in experiment concatenated their latent tokens, ran a single
BF16 `img_in` projection, then sliced the output into the original reference
and target segments. Text and segment order, subsequent attention, FFN and
KV cache semantics were unchanged. This is **not** a new attention kernel or
a faster diffusion schedule. The diagnostic admitted only explicit GPU,
512², 2–3-reference editing without LoRA. It was never enabled by default.

M4 Max, native BF16 checkpoint, three ordered references, seed 17, five
steps, separately prepared resident Sessions with full five-step warmup.
Each measured request reused encoded conditioning, dumped text/noise/ordered
reference latents, decoded VAE and exported PNG. The verifier found the
request tensor payloads equal. PNGs were byte-identical within each pair.
Individual wall times below exclude preparation but include dumps and VAE:

| Reference encoding | Ordinary GPU | Batched projection | Ratio |
| --- | ---: | ---: | ---: |
| Explicit 512px references | 10.116 / 10.149 s | 10.163 / 10.170 s | ~0.997× |
| Original 1024px references | 25.240 s | 25.193 s | ~1.002× |

For resized references the candidate was slightly slower twice. The
full-size single-run ~0.2% difference is too small to claim a speedup.
Both Session probes passed request export, cancellation, route switch and
unload. Thus fewer projection dispatches did not yield a useful whole-edit
improvement on either tested sequence geometry. The candidate code, CLI
gate, label and temporary contract case were removed; default and fastest
existing GPU/ANE routes remain as before. The ignored evidence remains
under `results/qwen21/session-ref512-edit3-gpu-batch-imgin-20260927/`
and `results/qwen21/session-fullref-edit3-gpu-batch-imgin-20260927/`, with
paired comparison receipts alongside. These one- or two-run screens do not
rule out different image projection implementations or other hardware.

After rollback the matching library and Session probe rebuilt; `make
test-qwen21` (27 + 7 + 22 contracts) and `git diff --check` passed. A fresh
five-step GPU request measured **10.116 s**, passed the full Session
lifecycle, and exported a PNG byte-identical to the pre-experiment GPU
control. Its ignored receipt is
`results/qwen21/session-ref512-edit3-gpu-postbatch-rollback-20260927/`.
Next GPU work should target the expensive **32-layer** long-sequence
attention/FFN work, not another one-time prefill input projection.
