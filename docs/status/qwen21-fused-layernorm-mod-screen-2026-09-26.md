# Qwen Image 2.1: fused LayerNorm/AdaLN screen (rejected)

This GPU-only experiment fused each DiT block's 4096-channel unweighted
LayerNorm and following AdaLN multiplication into one MLX Metal kernel.
The first candidate handled only the full-sequence prefill; a second also
handled broadcast modulation for all four decode steps. It was explicitly
gated to 512² BF16 GPU edits with 1–3 full-size references. It never changed
the production default or the hybrid W8A8 route.

M4 Max, three ordered 1024px references, 512²/5 steps, same prompt and seed,
separately prepared resident Session, 2-step warmup, one measured cached
request per candidate and two runs for the baseline. Wall includes VAE/PNG,
not model loading; the earlier same-binary unmodified baseline was
**25.1964/25.1875 s**, first-step-only fusion was **25.1487 s**, and
all-step fusion was **25.1466 s**. The approximately 0.2% gap is too small
to establish an improvement, especially with only one sample per candidate.
The all-step output RGB correlation to the base was **0.99805**, RMSE
**0.02118** (normalized [0,1]). Visual inspection retained both teapots and
the central dragon sticker, but sticker details shifted. No tensor dumps were
made in this particular screen, so the same request fixture and paths do not
constitute a new byte-for-byte input check.

The first smoke hit a Metal scalar argument syntax error. After correction,
the first version reached the decode step but rejected its broadcast
modulation shape; the completed first-step and all-step versions passed
the resident-probe cancellation/GPU-switch/unload lifecycle with exit 0.
The rejected intermediate failures were **not** treated as successful runs.
The kernel and experimental gate were removed because no meaningful speed
gain was established. Local ignored receipts are under
`results/qwen21/session-fullref-edit3-fused-layernorm-{v3,v4}-512-5-20260926/`;
the failed pilots are retained as `fused-layernorm-512` and
`fused-layernorm-v2-512` under the same prefix.

Future work should prioritize full-sequence QKV projections, gate/up FFN and
exact attention kernels, which dominate this large first step, rather than
fusing only the relatively small normalization/modulation boundary.

After rollback, the native library and matching probe rebuilt successfully,
`make test-qwen21` (27 + 7 + 14 checks) and `git diff --check` passed. A new
default GPU warm request completed in 25.303 s, passed the probe lifecycle
with exit 0, and exported the same PNG SHA-256
`fb9a24d697f2156ecef092d5e7c0b4bd8fe310d1792c4c5f8560` as the
pre-experiment GPU baseline. Its ignored receipt is
`results/qwen21/session-fullref-edit3-post-fusednorm-rollback-512-5-20260926/`.
