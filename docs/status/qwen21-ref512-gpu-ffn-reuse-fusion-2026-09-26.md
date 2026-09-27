# Qwen Image 2.1: 512px-reference pure-GPU FFN reuse + Metal Q/K fusion

This extends the existing **explicit approximate** 512² GPU final-step FFN
reuse from 1–3 resized-256 references to 1–3 resized-512 references. It
requires `execution=gpu`, `allow_approximation=true`, `qwen21_reference_size=512`,
at least three denoise steps and no Viggle LoRA. The native first-step BF16
attention, FFN and per-request prefix KV remain unchanged. Only the last
step reuses the preceding step's 32 FFN outputs. The already existing
`TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN=1` can additionally reuse
16 even-layer outputs in the penultimate step. All other operations still
run; this is **not** mathematically exact or a new faster GEMM kernel. The
existing `TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE=1` fuses GPU Q/K RMSNorm,
RoPE and output layout in a Metal kernel. GPU/ANE and default requests keep
their previous behavior. None of these flags enables itself.

M4 Max, Comfy BF16 model, resident Session: independently prepared routes,
two-step warmup, then two cached 5-step requests with VAE, PNG and matched
tensor dumps. One reference uses seed 42/red-teapot edit; two/three use
seed 17/ordered-product edit. Verifier matched the actual text, initial
noise and each ordered reference-latent payload exactly. Entries below
are **request wall medians** in seconds; load/prepare is excluded. The
plain-GPU column uses the same resized inputs, not the original
1024px-reference default input.

| Resized-512 references | Plain GPU | Final FFN reuse | Final + fused Q/K | Final + half penultimate + fused Q/K | Best / plain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1, 5 steps | 7.268 | 6.582 | **6.434** | **6.068** | **1.198×** |
| 2, 5 steps | 8.673 | 7.992 | **7.796** | **7.441** | **1.166×** |
| 3, 5 steps | 10.126 | 9.459 | **9.234** | **8.881** | **1.140×** |
| 3, 40 steps (one run) | 52.327 | not measured | 50.510 | 50.216 | **1.042×** |

The speedup is largely saved FFN compute in one or two of five steps; as
expected it shrinks markedly over 40 steps. The 512px reference resize
itself is an **additional input-fidelity tradeoff**, not GPU kernel
acceleration. Earlier full-size three-reference 5-step GPU timing was
25.107 s versus this resized-input 10.126 s; those times must not be used
to claim a same-input GPU optimization.

Visually inspected 1-/2-/3-reference five-step pairs and the three-reference
40-step pair. Single-pot shape/lid/spout/handle remain recognizable, but
**neither baseline nor fast arm obeys the matte-finish instruction well**.
Both ordered teapots remain in the two-reference image; the three-reference
5-step image still has the two pots and central dragon sticker, but the
sticker eyes/edge and blue-pot surface visibly change in the half-penultimate
route. The three-reference 40-step subjects remain recognizable. Relative
to plain GPU, the three-reference aggressive five-/40-step RGB correlations
are 0.9967/0.9894 and RMSE 0.0272/0.0451 on [0,1]; these numbers are not
semantic quality metrics. A second five-step seed 29 on the same three
references measured **10.179 → 8.886 s** (1.145×); both images keep the
two teapots and dragon, but there are local texture/sticker changes. Two
seeds and one prompt do not qualify broad image-editing fidelity.

The less aggressive final-step-only + Q/K fusion 5-step wall medians are
6.434/7.796/9.234 s and remain visually nearer the GPU baselines. When
semantic detail matters, use exact GPU or that lower-loss explicit option.
At 40 steps use exact GPU if reference details are more important than the
small ~4% one-run speed improvement; the GPU/ANE W8A8 route has a separate
speed/quantization/quality profile and no proven physical ANE placement.

The warm Session probes passed normal PNG export, cancellation, GPU-switch
and unload checks. Formal `turbocider plan` for the three-reference combined
mode reports all four approximation labels: `qwen21_reference_resize_512`,
`qwen21_gpu_reuse_final_ffn`, `qwen21_gpu_reuse_penultimate_even_ffn`, and
`qwen21_metal_qk_norm_rope`. `make test-qwen21` and `git diff --check`
passed. The 5-/40-step PNGs, payload dumps and JSON comparisons are under
ignored `results/qwen21/session-ref512-edit*-gpu-{reusefinal,half-reuse}*`.
The formal CLI `generate` command also completed the three-reference 5-step
mode, wrote a PNG, and reported GPU execution, five actual steps and all four
approximation labels. Its **cold-process** request wall was 11.994 s and is
not interchangeable with the resident benchmark. Normal default requests
still do not enable these approximation flags: a fresh same-input default
GPU request after the gate change took 10.144 s, passed the Session lifecycle,
and exported a PNG **byte-identical** to the earlier plain-GPU baseline.

Example explicit 5-step research mode, with a 512px-reference GPU edit JSON
already containing `allow_approximation=true`:

```sh
TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN=1 \
TURBOCIDER_QWEN21_GPU_REUSE_PENULTIMATE_EVEN_FFN=1 \
TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE=1 \
build/native/turbocider generate MODEL_DIRECTORY EDIT_REQUEST.json
```

This reaches the short-step 1.1–1.2× goal for these three **resized** edit
fixtures, but does not constitute a general exact GPU kernel acceleration:
full-size inputs, other prompts, LoRA, cold starts, other hardware and
40-step 1.1× remain unverified or below target. Next work should continue
with exact first-step long-sequence attention/FFN GPU kernels and W8A8
GPU/ANE scheduling without weakening conditioning.
