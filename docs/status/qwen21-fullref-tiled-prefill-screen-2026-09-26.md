# Qwen Image 2.1: full-reference first-step tiled W8A8 screen

This was a **rejected experiment**, not a shipped accelerator. The ordinary
512² W8A8/6144-channel hybrid still offloads only decode FFNs; its first
full-sequence step stays BF16 GPU. On M4 Max with three ordered full-size
1024px references and 130 text tokens, the first step has 13,442 rows and
previously cost ~18 seconds. The screen temporarily split each block's
post-attention FFN input into 1,024-row tiles, padded the last tile, and
ran the existing 32-block W8A8 Core ML / BF16 GPU channel partition on
each tile. Attention and prefix-KV geometry were left intact. The result
of each call was explicitly `mx::copy`-materialized before the next Core ML
prediction could reuse its session-wide output backing. This requires
`ceil(13442/1024)*32 = 448` extra predictions and about 3.28 GiB of
first-step BF16 tile-result copying; the Core ML `output_copy_bytes` counter
does **not** count those MLX copies.

Each arm used the same BF16 checkpoint and ordered reference files, a
separate resident Session, two-step warmup, cached conditioning, seed 17,
and tensor dumps. Measurements include VAE decode, PNG export and equal
dumping, but exclude prepare/model load. The comparison verifier checked
byte-identical text, initial noise and all three reference latents, the
checkpoint-matched 1,024-row/6,144-channel manifest, zero Core ML runtime
failures and the expected per-request Core ML call counts. One run per arm,
sequential rather than ABBA; neither speedup is a multi-seed qualification.

| Same-input steps | Original decode-only W8A8 wall | Tiled-prefill wall | Wall ratio | Calls, baseline / candidate | RGB correlation / RMSE `[0,1]` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 24.156 s | 20.661 s | **1.169×** | 128 / 576 | 0.92436 / 0.13743 |
| 40 | 71.600 s | 75.127 s | **0.953×** | 1248 / 1696 | 0.75441 / 0.22152 |

The step profiler observed first-step **18.03 → 14.43–14.45 seconds**, so
offload did accelerate that isolated phase. In the 40-step candidate some
subsequent decode steps took ~1.62–1.65 seconds rather than the control's
~1.34–1.39 seconds, and the complete run was slower despite the faster
prefill. Core ML prediction API time and GPU work overlap and cannot simply
be subtracted from denoise wall. The experiment needs resource/concurrency
traces before assigning the decode regression to memory pressure, scheduling
or thermal behavior. `CPU_AND_NE` is a compute-unit policy, not a hardware
ANE residency measurement.

Visual inspection matters more here than pixel parity. At five steps the
baseline already ghosts the sticker; the tiled route changes it into a
glowing/translucent dragon and loses the recognizable opaque sticker edge.
At 40 steps both retain two teapots and a dragon sticker, but sticker size
and placement, pot shape and lighting change noticeably. This is not an
acceptable improvement to a full-reference edit, even under a relaxed
image-generation numerical threshold. It was **not** tested on 1–2
references, another seed, or a different checkpoint/LoRA; no inference
about those workloads follows.

The probe completed cancellation-after-prefill, GPU route switch, PNG export
and session teardown with exit 0 after the experimental flag was disabled
for the GPU switch. The first pilot finished a valid PNG but exited 1 at
that switch because the flag was still set; it is not a passing lifecycle
run. The verified five-step pilot produced the same PNG SHA as the initial
pilot. The `make test-qwen21` contracts passed with the candidate in place;
those contracts alone never established output quality or wall speed.

To preserve the known faster long-step hybrid and avoid an opt-in that
degrades reference fidelity, the tiled runtime, CLI gate, counter changes
and candidate-only verifier branch were removed after this screen. The
ignored receipts are `results/qwen21/session-fullref-edit3-tiledprefill-{control-5,candidate-5-verified,control-40,candidate-40}-20260926/` and the
matching `*-{5,40}-comparison-20260926.json` files; the earlier pilot is
`session-fullref-edit3-tiledprefill-candidate-20260926/`.

After rollback, the rebuilt native library, rebuilt matching Session probe,
`make test-qwen21` and `git diff --check` passed. A fresh five-step
three-reference hybrid request took 24.205 seconds, executed 128 decode
FFN predictions, reported zero Core ML output copies, passed the probe's
cancellation/GPU-switch/unload checks, and produced a PNG byte-identical
to the pre-rollback original hybrid control. The ignored receipt is
`results/qwen21/session-fullref-edit3-postrollback-5-20260926/`.

Next candidates should target the long-prefix GPU attention and memory
traffic, or investigate selectively retaining precise prefix FFNs rather
than quantizing all reference/text representations. Any new option needs
same-input 5/40-step checks, multiple seeds and human evaluation of subject
count, placement and reference details before entering CLI selection.
