# Qwen Image 2.1: hybrid penultimate even-FFN reuse (opt-in)

The existing 512² edit speed-first route sends 1,024-row FFN tiles from the
last 16 first-step blocks to the same W8A8 Core ML layer graph in a loop,
then runs all 32 hybrid FFNs on the next three denoising steps and reuses the
last 16 outputs on the final step. The additional **explicit**
`TURBOCIDER_QWEN21_HYBRID_REUSE_PENULTIMATE_EVEN_FFN_DIAGNOSTIC=1` caches
step-three FFN outputs, reuses those of the 16 even blocks in step four, and
captures step-four outputs for the existing last-16 step-five reuse. This
removes 16 Core ML predictions per five-step request. The 16 odd FFNs on
step four still execute, as do all 32 FFNs on step three. For three resized
references the combined route uses **172** W8A8 predictions per request
(76 first-step + 32 + 32 + 16 + 16), versus 188 without the new flag.

The new gate requires W8A8, BF16 GPU complement, the calibrated full-32-layer
6,144-channel/1,024-row manifest, 512² edit with 1–3 ordered references
resized to 512px, exactly five steps, explicit approximation opt-in, the
already screened last-16 prefill tiling and last-16 final-step reuse, and no
LoRA or GPU penultimate reuse. It changes no unflagged route, full-size
reference edit, text-to-image, 40-step inference, or model artifacts. The
preexisting `TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC=1` and final
prefill target-only switch were enabled in the measurements below. These
are **lossy** FFN-step approximations, not an exact GPU kernel speedup; a
Core ML `cpuAndNeuralEngine` policy is not hardware ANE residency proof.

M4 Max, independently prepared resident Sessions, matching five-step warmup,
same prompt and each ordered reference per comparison, cached conditioning,
matched text/initial-noise/reference-latent tensor payloads. Request wall
includes VAE, PNG export and equivalent tensor dumping, excludes cold
preparation. Two requests per seed-17 arm; one request per seed-29 arm.

| 512px references / seed | Previously fastest hybrid | Additional even FFN reuse | RGB correlation / RMSE vs previous |
| --- | ---: | ---: | ---: |
| 1 / 42 | 5.550 / 5.775 s | **5.381 / 5.360 s** | 0.99956 / 0.00918 |
| 2 / 17 | 6.725 / 6.746 s | **6.519 / 6.519 s** | 0.99863 / 0.01590 |
| 3 / 17 | 7.928 / 7.919 s | **7.792 / 7.806 s** | 0.99761 / 0.02265 |
| 3 / 29 | 7.950 s | **7.773 s** | 0.99775 / 0.02226 |

Each normalized RMSE is on `[0,1]`; pixel metrics are not semantic quality
scores. On the same seed-17, same resized-input three-reference pure GPU
control (10.114/10.137 s), the new 7.792/7.806 s hybrid is **~1.298×**
faster by the median warm request wall, not a qualified general 1.3× claim.
The new flag alone adds about 1–3% to the previous fast hybrid for two/three
references. The previous one-reference control had an outlier; do not use
the resulting ~5% median ratio as a stable one-reference speedup.

Visual inspection of both three-reference seeds: two teapots and the orange
dragon sticker remain in the requested arrangement, with subtle changes to
eyes/sticker edge and pot textures, not a missing subject. The two-reference
pots retain their shapes; the one-reference reddish pot still fails the
requested matte finish in both arms. This is acceptable as a **research
speed-first** tradeoff on these inspected fixtures, not a blanket reference
identity, editing-instruction or LoRA-quality guarantee. More prompts,
reference styles, seeds and longer denoising schedules require their own
visual review. Full-size 1024px references are **not** silently resized by
this flag; `qwen21_reference_size=512` is an independent, explicit input
fidelity tradeoff.

The Session probe verified exactly 16 fewer predictions per request, zero
prediction failures and zero Core ML output-copy bytes, and completed
cancellation, GPU-route switch, and unload. A formal `turbocider plan`
reported the new approximation label; formal `generate` produced a PNG
byte-identical to the resident seed-17 candidate. Its **cold** request took
20.222 s, including loading and encoding, not comparable to the warm walls.
Afterward a fresh flag-off fast-hybrid probe made its original 188 calls,
passed lifecycle and produced the saved flag-off PNG byte-for-byte.
`make test-qwen21` (27 + 7 + 20 contracts) and `git diff --check` passed.

The local ignored evidence is under
`results/qwen21/session-ref512-edit{1,2,3}-hybrid-half-even-20260927-a/`,
`results/qwen21/session-ref512-edit3-hybrid-half-even-seed29-20260927-a/`,
and their `*-vs-fast.json` comparison receipts. The candidate requires the
four preexisting research switches **in addition** to the new switch:

```sh
TURBOCIDER_QWEN21_METAL_FUSED_QKV_DIAGNOSTIC=1 \
TURBOCIDER_QWEN21_TILED_PREFILL_W8A8_DIAGNOSTIC=16 \
TURBOCIDER_QWEN21_HYBRID_REUSE_FINAL_LAST16_FFN_DIAGNOSTIC=1 \
TURBOCIDER_QWEN21_PREFILL_LAST_TARGET_ONLY_DIAGNOSTIC=1 \
TURBOCIDER_QWEN21_HYBRID_REUSE_PENULTIMATE_EVEN_FFN_DIAGNOSTIC=1 \
build/native/turbocider generate models/Comfy-Org-Qwen-Image-2.1 EDIT_REQUEST.json
```

Keep the lower-loss flag-off fast hybrid available; do not enable this mode
automatically for untested inputs. Larger speedups on fresh full-size edits
still require reducing first-step GPU attention/FFN time without losing
cross-reference fidelity.
