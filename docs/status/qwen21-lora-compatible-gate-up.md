# Qwen-Image-2.1: LoRA-compatible base Core ML gate/up diagnostic

Status: experimental, explicitly selected, **not the default or fastest LoRA route**.
The existing base fused-FFN and full runtime-LoRA GPU paths remain unchanged.

## Why the existing fused graph cannot reproduce the student FFN

The fused W8A8 Core ML graph returns a completed base FFN prefix. Runtime
gate/up LoRA must be injected **before** SiLU and the elementwise product;
runtime down LoRA must act on the resulting hidden activation. Once the base
graph has discarded gate/up intermediates, an output-only correction cannot
recover the student FFN. The old `LORA_BASE_ANE_DIAGNOSTIC` therefore only
applies runtime LoRA in the BF16 GPU suffix and is knowingly lossy.

The new graph exports the first 6144 gate rows and first 6144 up rows as
12,288 FP16 output channels from a fixed 1024-token W8A8 input-only Core ML
projection. There is no SiLU, hidden quantization or down projection in that
graph. The GPU starts its BF16 suffix gate/up and down while Core ML predicts;
after prediction, it adds the prefix runtime gate/up LoRA before SiLU and
applies the **base plus runtime LoRA** prefix down matrix. The prefix and
suffix outputs are added. Attention and the first-step prefill still use the
ordinary complete runtime LoRA GPU implementation. This fixes the missing
adapter terms, but W8A8 base projection and the different arithmetic order
remain approximate relative to full BF16 GPU. No adapter is baked into the
compiled Core ML artifact: the checkpoint hash still identifies the base.

## Selection and safety

Export with `tools/coreml/export_qwen3.py --qwen21-gate-up-only` and the
existing Qwen21 32-layer W8A8 export inputs (6144 channels, fixed 1024 rows,
input-only A8 calibration). Compile the resulting manifest with the normal
Core ML compilation tool. An example **ignored local artifact**, not a
redistributable model, is
`results/qwen21/lora-gate-up-compiled-32/manifest-91f6f6de438ee026bc8ec90d0ac4a78066b41dab21d7325f7abda12fd9ec108b.json`.
This manifest is tied to its source checkpoint; use your own export for a
different checkpoint. Neither weights nor compiled models are committed.

Set `TURBOCIDER_QWEN21_LORA_GATE_UP_DIAGNOSTIC=1` and explicitly request
`gpu_ane`, `qwen21_w8a8: true`, `allow_approximation: true`, a compatible
manifest and 512×512. For six-step Viggle v0.2.1 r256 runtime LoRA, also set
`TURBOCIDER_QWEN21_LORA_BASE_ANE_DIAGNOSTIC=1`. Base-only five-, 20- and
40-step 512² requests do not require that second flag. A 1–3-image edit using
512px resized references also needs the existing
`TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC=1` opt-in. The gate/up path rejects
GPU W8A16 suffix, full-GPU FFN blocks, DBCache, tiled prefill, final-step FFN
reuse and incompatible manifest identities; the generic C FFN prediction ABI
rejects its larger intermediate tensor to avoid a buffer overflow. The
resident probe disables/restores the gate/up flag during GPU/base/LoRA route
switches. This is a diagnostic, not a CLI default or recommendation.

## Local measurements and visual check

On this machine (M4 Max), matched resident Sessions after their own complete
warmups. Times include denoising, VAE/PNG and tensor-dump I/O; they exclude
model/Core ML first load. Same request, seed and text/noise/reference tensors
were checked byte-for-byte by `qwen21_compare_sessions.py`. They do **not**
prove the compiled operations actually ran on the ANE: Core ML was asked to
use its CPU/Neural Engine compute policy. One seed and 1–2 warm repetitions
are not a cross-seed quality or ABBA speed qualification.

| 512² request | Full GPU | Old fused base ANE + GPU suffix | New gate/up base ANE + GPU SiLU/down | New vs full GPU | New vs old hybrid |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base, fox text-to-image, 5 steps (one warm run) | 6.012 s | 4.797 s | 5.391 s | 1.115× faster | 12.4% slower |
| Viggle runtime LoRA, fox text-to-image, 6 steps (two warm runs) | 7.969 / 7.867 s | 6.395 / 6.655 s | 8.642 / 8.603 s | 8.9% slower (median) | 32.1% slower (median) |
| Viggle runtime LoRA, 3-reference sticker-on-blue-teapot edit, 6 steps (one warm run) | 12.575 s | 10.857 s | 13.147 s | 4.5% slower | 21.1% slower |

The optimized LoRA comparisons enabled `VIGGLE_LORA_FP16` and
`METAL_QK_NORM_ROPE` in **all three routes**. Base five-step tests used neither
option. Candidate runs used the expected 128 (base) or 160 (six-step LoRA)
Core ML predictions per request, with zero runtime failures and all 227
Viggle runtime LoRA projections bound. The Core ML output is 12,288 channels
versus 4096 for fused FFN, or roughly 24 versus 8 MiB per 1024-row output.
For base five-step runs, reported Core ML prediction API accumulated across
prepare/warm/run was 2.548 s (new) versus 2.892 s (old): faster Core ML calls
alone do not repay extra GPU SiLU/down work and synchronization. First model
load is separate from the warm-request timings: the base five-step runs
reported ~6.60 s (new) and ~6.82 s (old) including ~5.88 s each of checkpoint
manifest validation; model-loading numbers varied much more across different
LoRA runs and should not be read as cold-start superiority.

At one 1024-row single-layer synthetic-input probe, GPU BF16 FFN took
~21.3 ms, old fused hybrid ~11.1 ms, new gate/up hybrid ~14.7 ms. These
figures omit 32-layer model costs and runtime LoRA; synthetic-input relative
error is not an image-quality metric. A separate held-out real base FFN input
showed ~0.0064 relative RMSE after GPU SiLU/down, but full-image evaluation
is essential. Parallel submission of gate/up LoRA GPU deltas before the Core
ML call was also tried: warm fox timings stayed 8.634 / 8.611 s, with no
measurable benefit, so that scheduling change was removed.

Visual inspection of the fox shows the new result preserves its posture,
tail, fur, snowy ground and pine background, close to full LoRA GPU; RGB
RMSE is 4.36/255 and correlation ~0.99796 for that single seed. In the edit,
the new result preserves both teapots, spouts, lids and handles and the small
white-bordered orange decal **on the blue teapot**, also matching the GPU
intent; RGB RMSE is 3.95/255 and correlation ~0.99863. These numerical
statistics are diagnostic only. The old faster suffix-only edit also appears
acceptable for this specific prompt, but its missing prefix adapter terms
can cause larger pose/texture differences on other prompts. Other seeds,
strict materials/text, and one-/two-reference edits have not passed a visual
gate for the new route.

Conclusion: the linear-output graph is the right boundary for **runtime LoRA
correctness**, not a faster default. It removes fused SiLU/down from Core ML,
triples intermediate output traffic, and moves the 6144-wide BF16 prefix
down/LoRA work onto the GPU after prediction. Keep full GPU as the accurate
default for LoRA and retain the faster fused graph for base; suffix-only hybrid
remains a separately labelled, explicitly lossy experiment.
