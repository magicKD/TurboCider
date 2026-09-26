# Qwen Image 2.1: 512² W8A8 editing with 1024px references (2026-09-26)

The established 32-block, 6144-channel SmoothQuant W8A8 Core ML FFN
partition operates on the **output** image's 1024 decode rows. Reference
latent tokens stay in the GPU attention prefix KV. This diagnostic therefore
uses the **same full-size 1024px reference conditioning** on both the BF16 GPU
and hybrid routes; no 256px resize or new Core ML compilation is involved.
The GPU branch remains BF16 by default. `gpu_ane` is still an explicit
experiment, and Core ML CPU+NeuralEngine configuration and runtime-call counts
do not independently prove physical ANE occupancy.

`TURBOCIDER_QWEN21_FULL_REF_W8A8_DIAGNOSTIC=1` permits 512², 1–3-reference
editing with `qwen21_reference_size: 1024`, `qwen21_w8a8: true`, an explicit
compiled manifest and `allow_approximation: true`. Runtime checks require
the checkpoint-matched 6144-channel, 1024-row, 32-block manifest and **full
32/32 hybrid FFN coverage**, without the 3/5/7 GPU-only fallback. With the
flag absent, the established production policy still requires 256px edit
references. No model weights, private machine path, or compiled Core ML
artifact is committed.

## Prepared resident Session comparison (M4 Max, one prompt/seed per count)

Each route prepared independently, ran two warmup steps, then reused the
same ordered reference files and prompt. These are *prompt-cache-hit* request
wall times including decode/export and optional diagnostic tensor writes,
not cold-start times. The **5-step** comparisons use two runs per route and
byte-identical dumped text, initial noise and all reference latent tensors.
The **40-step** 1- and 3-reference comparisons have one run per route and
matching dumped inputs. The 40-step **two-reference** BF16 base and hybrid
each have one matched-prompt/seed/reference run but *neither* dumped inputs;
do not claim byte-verified equality for that pair.

| Ordered references | Steps | GPU wall (s) | W8A8+BF16 GPU wall (s) | Speedup | RGB correlation |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 11.632 / 11.600 | 10.364 / 10.370 | 1.121× | 0.99764 |
| 2 | 5 | 18.104 / 18.017 | 16.975 / 16.925 | 1.065× | 0.99877 |
| 3 | 5 | 25.107 / 25.146 | 24.130 / 24.103 | 1.042× | 0.99717 |
| 1 | 40 | 55.194 | 42.477 | 1.299× | 0.99430 |
| 2 | 40 | 68.480 | 59.026 | 1.160× | 0.98746 |
| 3 | 40 | 82.303 | 71.634 | 1.149× | 0.94371 |

The 5-step comparison verifier checked the input tensors, and the resident
probe checked 128 hybrid calls per request (4 decode steps × 32 layers). The
40-step probe checked 1248 calls (39 × 32); no runtime failures occurred.
The speedup falls with more references at fixed steps because longer GPU
attention/prefix work is not transferred to Core ML; this is an inference
from matched timings and the execution graph, not a per-kernel occupancy
measurement. The 256px-reference route can be much faster, but it changes
the input and is **not** the same-input comparison here. At 5 steps the base
model itself often shows ghosting or underdeveloped shapes; a high GPU/hybrid
pixel correlation does not mean either image meets a quality gate.

The normal CLI also ran the exact one-reference 5-step diagnostic request
once. It exported the same PNG SHA-256
`09bff43656498bd069684f9a119321a120eb613ec11f97c810e4e96e72394e66`
as the resident probe, with `prompt_cache_hit=false`, a **21.730 s** cold
request wall time and **7.065 s** hybrid setup. These are *not* the
10.36-second resident/cache-hit figures in the table, nor evidence of a
cold-start speedup over a matched one-shot GPU request.

On the *same* 6144-channel ANE model, full-size two-reference, 5-step
workload, explicitly quantizing the GPU suffix to W8A16 gave **17.186 /
17.202 s**, versus the BF16 suffix's **16.975 / 16.925 s**. The BF16/W8A16
median ratio is **0.986×**, so W8A16 is about 1.4% slower end-to-end here.
Both runs matched their text/noise/two reference tensors and 32-layer Core ML
call counts; W8A16-versus-BF16 RGB correlation was 0.99979. The separately
tested 256px-reference workload likewise favored BF16 more clearly (ratio
0.921×). Int8 ANE arithmetic does not imply that MLX GPU weight-only W8A16
matmuls accelerate the complementary branch; keep BF16 as the fast suffix.

## Visual check (not an automatic acceptance gate)

- One reference, 40 steps: red-brown teapot, lid, spout and handle are close
  between GPU and hybrid, but both are **glossy** despite the prompt's matte
  requirement. Normalized RGB RMSE 0.03409; cannot call the edit successful.
- Two references, 40 steps: beige left and blue right teapots retain the warm
  wooden-table composition. Hybrid changes some glaze/handle detail; RGB
  RMSE 0.04782. The 5-step images both have a distorted/ghosted blue base.
- Three references, 40 steps: two teapots and central orange dragon sticker
  all remain recognizable, but the sticker position/size and teapot contours
  shift visibly; RGB RMSE 0.10525. The 5-step images show conspicuous sticker
  ghosting in **both** routes, so their visual quality is inadequate even
  though the hybrid is faster.

Source requests, PNGs, JSONs, and tensor dumps are in ignored
`results/qwen21/session-fullref-edit{1,2,3}-*` directories. Use the report
files `session-fullref-edit{1,2,3}-gpu-vs-w6144-512-{5,40}-comparison.json`
where present. The invalid first three-reference attempt ended before Core
ML loading because its *command-line* manifest filename was mistyped; the
valid run was made in the `-retry` directory without changing the original
reference files.

## Implications

Full-reference W8A8 is a useful **40-step** speed/quality tradeoff on these
samples, and covers 100% of the decode FFNs, exceeding the 90% target.
It does not meet a blanket 1.1× short-step speed target for two and three
full-size references. Do not make this diagnostic the CLI default until more
prompts, seeds, editing instructions, cold requests, and the true hardware
placement are checked. Retain BF16 GPU for fidelity-sensitive requests and
the established 256px-reference hybrid option for speed-first edits.
