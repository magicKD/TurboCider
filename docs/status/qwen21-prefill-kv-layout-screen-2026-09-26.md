# Qwen Image 2.1: first-step K/V layout screen (rejected)

In the 512² BF16 GPU edit with three ordered 1024px references, the first
step has 13,442 rows and each transformer block's attention reads several
overlapping K/V prefixes. A temporary explicit diagnostic materialized
head-major `k` and `v` once with `mx::contiguous` immediately after Q/K
norm and RoPE, before the eight attention segments and cached prefix slices.
It changed no keys, values, mask, prompt, reference ordering or precision.
The option was restricted to the full-reference GPU edit and never enabled
on the default or Core ML hybrid routes.

On M4 Max, independently prepared resident Sessions each ran two-step
warmup and the same cached five-step request. The 1024px references, prompt,
seed, model and output were matched. Request wall includes VAE/PNG, excludes
preparation. Baseline in the same rebuilt native library: **25.1964 and
25.1875 s**; contiguous K/V: **25.2432 s** (one run). The candidate PNG
SHA-256 `fb9a24d697f2156ecef092d5e7b49638a2c8e7c0b4bd8fe310d1792c4c5f8560`
matched the saved exact-GPU reference byte-for-byte. Cancellation, GPU
switch and unload passed, exit 0. No tensor dumps were made in this screen;
the paired request metadata, fixture paths and PNG identity are corroborating
checks, not a new byte-verified input claim.

This provides no evidence of a meaningful wall improvement. Its small
~0.2% deficit cannot distinguish kernel cost from ordinary run noise. The
temporary option and layout conversion were removed; no 40-step or LoRA
claim follows. Ignored receipts are under
`results/qwen21/session-fullref-edit3-prefill-contiguous-{kv,control}-5-20260926/`.
Further exact optimization needs GPU execution traces of attention/KV
packing and long-sequence projections, not an assumption that explicitly
contiguous arrays accelerate the existing fused SDPA implementation.
