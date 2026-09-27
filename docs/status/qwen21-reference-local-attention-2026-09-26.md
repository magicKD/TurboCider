# Qwen Image 2.1: reference-local prefill attention experiment

The default prefill lets each later reference image attend to all earlier
reference images. A new **explicit GPU-only approximation** keeps earlier
text and the current reference image in that reference's K/V, but excludes
earlier reference image rows. The final output image still attends to the
full, unpruned prefix. This changes the prefix representations and hence
later image content; it is not an exact fused-kernel or KV-cache optimization.

`TURBOCIDER_QWEN21_REF_LOCAL_ATTENTION=1` applies the change to all later
references; `=2` applies it to the last reference only. Both are restricted
to explicit 512² BF16 GPU, 2–3 **full-size** edit references, no LoRA,
`allow_approximation=true`. Other requests and the default `=0` remain
unchanged. The plan and generated result mark the approximation; the
resident prefix-KV experiment includes this flag in its cache identity.
The existing `TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE=1` can be combined,
independently marked and remains opt-in. No W8A8 model or GPU/ANE schedule
is changed.

M4 Max, three ordered 1024px references / 512² output, same prompt and seed
17. Each arm prepared a separate resident Session, ran a two-step warmup,
then a prompt/reference-cached request. Two five-step runs/arm; one 40-step
run/arm. Request walls include VAE, export, and equal input tensor dumping,
but not initial preparation. The verifier checked exact text, noise and all
three reference latent tensor bytes; same-route PNGs matched across the two
five-step runs. The 40-step singletons and noninterleaved 5-step runs are
not a qualified ABBA distribution. The flag-off GPU PNG SHA-256 at both
five and 40 steps matched the previously saved original GPU PNG, verifying
the unchanged default edit on these two fixtures.

| Steps | Baseline GPU wall | Later references local (mode 1) | Last reference only (mode 2) | Mode 1 + fused Q/K |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 25.166 / 25.250 s | 23.225 / 23.177 s, **1.086×** | 23.786 / 23.760 s, **1.060×** | 22.610 / 22.631 s, **1.114×** |
| 40 | 81.855 s | 79.848 s, **1.025×** | 80.905 s, **1.012×** | not measured |

Image differences from exact GPU (normalized `[0,1]` RGB):

| Steps | Mode | RGB correlation / RMSE | Visual check |
| ---: | --- | ---: | --- |
| 5 | 1 | 0.96366 / 0.09143 | Two teapots and sticker remain, but outlines/ghosting worsen; baseline is also ghosted. |
| 5 | 2 | 0.98382 / 0.06124 | Subjects remain, sticker and teapot outlines still ghosted. |
| 5 | 1 + fused Q/K | 0.96431 / 0.09062 | Similar visible double contours to mode 1. |
| 40 | 1 | 0.78639 / 0.20387 | Three recognizable subjects, but sticker size/position and blue teapot transparency differ substantially. |
| 40 | 2 | 0.86884 / 0.15912 | Sticker visibly larger; the teapots retain more of their positions than mode 1. |

Under a loose *recognizable subject* criterion the 40-step candidates may
be usable as speed-first edits, but these tests do **not** establish faithful
reference editing or a good five-step image-quality result. In particular
the 1.114× five-step wall ratio does not satisfy a 1.1× **quality-qualified**
target: its output has clearly worse ghosting. At 40 steps the maximum
observed uncombined speed gain is only ~2.5% and visibly changes the edit.
Keep all variants explicit, with the accurate approximation label. Do not
promote to the default or attribute their speed gain to a faster ANE.

Raw single-request reports, input dumps and PNGs are ignored in
`results/qwen21/session-fullref-edit3-{local,last-local,local-fusedqk}-*20260926/`;
comparison JSONs have matching `*-comparison-20260926.json` names. The
normal CLI `plan` showed both approximation labels for the combined mode;
one cold `generate` completed, exported PNG and reported ~21.99 s denoise,
~33.10 s total including encoders/loading. That cold point is **not** the
warm speedup above. Its PNG SHA matched the resident candidate byte-for-byte.
Native build and `make test-qwen21` passed.

An additional **single-run** two-full-reference / five-step pair with the
same combined flags measured 18.080 s GPU versus 16.984 s candidate wall
(1.065×); text, noise and both reference latents matched byte-for-byte.
Both teapots remain, while the blue teapot's base and transparency shift;
RGB correlation/RMSE are 0.94821/0.09944. This is not a qualification for
two-reference or 40-step editing. Paired report:
`results/qwen21/session-fullref-edit2-local-fusedqk-comparison-20260926.json`.

Next: exact first-step acceleration that preserves cross-reference attention,
or quality-gated selective block/segment coverage; test 40-step two-reference
edits and more prompts/seeds before considering an automatic speed/quality preset.
