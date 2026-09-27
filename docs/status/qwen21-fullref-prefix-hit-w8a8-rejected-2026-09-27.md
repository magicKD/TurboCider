# Qwen Image 2.1: full-reference prefix-hit target W8A8 (rejected)

This was an opt-in **development screen**, not a retained CLI accelerator.
For a five-step 512² edit with three original-size 1024px references, the
resident prefix-KV bank makes a later identical-condition request's first
denoising step target-only (1024 rows). The experiment routed that step's
32 FFNs through the existing 6144-channel W8A8 Core ML/BF16 GPU channel
split instead of the GPU full FFN; the full 13,442-row first-step miss,
reference attention, cached K/V, and subsequent four hybrid steps remained
unchanged. This is different from looping the long reference sequence
through ANE or GPU/ANE splitting whole token rows.

The candidate was gated to explicit 32-layer W8A8, five steps, resident
prefix-KV, three original-size references, BF16 GPU suffix, no LoRA and
approximation opt-in. M4 Max, seed 17, same checkpoint/prompt/ordered
reference files and five-step warmup with the prefix bank disabled during
warmup. Both runs cached text and reference encodings; first measured
request missed, the next two hit. Request wall includes VAE/PNG/tensor
dumps but excludes preparation. The previously saved same-input hybrid
control also warmed five steps and dumped tensors; the screen was serial
and **not** a freshly interleaved ABBA control.

| Request | Previously saved hybrid control | Target-W8A8 candidate | Core ML calls per request |
| --- | ---: | ---: | ---: |
| First prefix miss | 22.825 s | 22.829 s | 128 / 128 |
| Prefix hit 1 | 7.890 s | 7.433 s | 128 / 160 |
| Prefix hit 2 | 7.791 s | 7.444 s | 128 / 160 |

On hit the cached-first-step time fell from roughly 1.62 to 1.37 seconds;
the candidate's two hit walls improved about 4–6% relative to the saved
control. The first-miss PNGs were byte-identical. Hit PNGs had normalized
RGB correlation ~0.98173 and RMSE ~0.06597 versus the hybrid control, but
**visual inspection found conspicuous sticker-face/eye distortion and
ghosting** despite recognizable teapots. The performance saving is not
worth this reference-identity loss on the inspected edit. The candidate
also changes hit output relative to its own miss, a counterintuitive
property for repeated identical editing. This is a quality rejection,
not a claim that W8A8 hardware arithmetic cannot run faster.

The probe finished with exit 0: miss/hit sequence, expected call counts,
changed-seed oracle handling, cancellation/retry, GPU switch and unload.
The source-level runtime switch, plan admission, report label and probe
branches were removed after the screen. No code path selects it in the
rebuilt library; the lower-loss full-reference hybrid and the faster
explicit resized-reference tile-loop route remain separate and available.
After removal, the native library and Session probe rebuilt. Two
full-size three-reference prefix hits exported SHA-256
`7e675aaf25b942755f27c0ec0326cba4c11a277a352d5734504768fe9875af0e`,
byte-identical to the saved unmodified hybrid route; both made 128
predictions with zero failures. The probe exited 0 after cancellation,
GPU switch and unload. `make test-qwen21` and the exact hybrid-merge test
also passed. This confirms restoration of the measured fixture, not wider
edit quality or a new speedup.
Raw output and tensor dumps remain ignored under
`results/qwen21/session-fullref-edit3-prefix-hit-w8a8-screen-20260927/`.
Rollback evidence is under
`results/qwen21/session-fullref-edit3-code-cleanup-rollback-20260927/`.
One prompt/seed and sequential timing cannot establish performance or
quality for other subjects; no 1–2-reference or 40-step inference follows.
