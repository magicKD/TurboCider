# GPU/ANE request examples

Run these examples from the repository root. Model and adapter files are not
included; replace the relative `models/...` paths with your local locations.
Choose a new `output` for each result you want to retain. These are generation
templates, not benchmark drivers or evidence that every combination is faster.
The commands below use the source-built `build/native/turbocider`; after
packaging, `dist/cli/turbocider` is the corresponding distribution entry point.
Do not compare an older packaged binary against a newly built native library.

## Choose a route

| Request | Starting route | Optional acceleration |
| --- | --- | --- |
| [Qwen base, 512², 40 steps](qwen21-base-512.json) | GPU | Matching frozen base graph for the measured fastest base route; `runtime` for the reusable runtime-weight graph |
| [Qwen base, 1024², 40 steps](qwen21-base-1024.json) | GPU | Base-only runtime v1 c1792/K1024/N512 plus optional Q/K fusion is fastest in the latest matched 1024² screen; v2 remains for LoRA |
| [Qwen Viggle v0.2.1 r256, 512², 6 steps](qwen21-viggle-runtime-lora-512.json) | GPU, inference-time LoRA | `runtime` v2 with LoRA inputs; no established general Qwen editing speedup |
| [Qwen shared frozen base graph](qwen21-viggle-runtime-lora-hybrid-512.json) | Explicit `lora_fused` | Requires its own base-only `lora_fused` manifest, not an ordinary frozen base graph |
| [Z-Image base, 512², 8 steps](z-image-turbo-512.json) | GPU | Matching frozen base graph; keep its checkpoint/shape/channel requirements |
| [Z-Image base, 1024², 8 steps](z-image-turbo-1024.json) | GPU | Base-only runtime v1 c352/K1024/N512 optional; the 512² image-only frozen graph is not a 1024² baseline |
| [Z-Image runtime LoRA, 512², 8 steps](z-image-runtime-lora-512.json) | GPU, inference-time LoRA | `lora_fused` or `runtime` v2 with the corresponding manifest |
| [Z-Image GGUF base, 512², 8 steps](z-image-gguf-base-512.json) | GPU | Explicit runtime-weight base only; do not infer GGUF runtime-LoRA support |

The two LoRA-capable graph types are different; runtime v1 is base-only:

- `lora_fused` keeps **base weights** frozen in per-layer Core ML graphs. The
  name means fused base FFN operators, not merged LoRA weights.
- `runtime` v2 uses a fixed-shape graph with runtime base-weight slots and
  activation corrections. It can reuse the graph across layers and adapters.
- `runtime` v1 keeps the reusable runtime base-weight slots but has no LoRA
  activation inputs or corrected hidden output. Use it only without an adapter.

Both preserve the full activation: gate/up LoRA enters before SiLU and
down-LoRA consumes the corrected hidden state. No adapter is merged into the
base checkpoint, Core ML artifact or runtime base-weight slots. Use the same
compatible manifest for base and different adapters; do not select `lora_merged`.

## Minimal commands

For the measured 1024² base configuration (not editing or LoRA):

```sh
TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE=1 \
  build/native/turbocider generate models/Comfy-Org-Qwen-Image-2.1 \
  examples/requests/qwen21-base-1024.json \
  --hybrid-mode runtime --ane-manifest path/to/runtime-v1-c1792/manifest.json
```

This needs the matching c1792/K1024/N512 v1 base-only graph. Keep
`TURBOCIDER_RUNTIME_ANE_CHUNKS` unset or `auto`, and profiling off. The
[same-build report](../../docs/status/runtime-ane-v1-base-2026-09-30.md)
records 1.266× versus GPU for this fox/seed42 workload, with memory and
single-prompt visual limits. For runtime LoRA use the separate v2 graph below.

For Z-Image 1024² base (not 512² or LoRA), the matched two-prompt screen used
the base-only c352/K1024/N512 v1 graph:

```sh
build/native/turbocider generate models/Comfy-Org-z_image_turbo \
  examples/requests/z-image-turbo-1024.json \
  --hybrid-mode runtime --ane-manifest path/to/z-runtime-v1-c352/manifest.json
```

Keep `TURBOCIDER_RUNTIME_ANE_CHUNKS` unset or `auto`, with profiling off.
The [Z 1024² report](../../docs/status/runtime-ane-z-1024-v1-2026-09-30.md)
records about 1.106× versus same-build GPU on lighthouse/portrait prompts.
The existing 512² image-only W8A8 frozen artifact cannot serve as this
workload's matched frozen baseline. Z runtime LoRA still requires v2.

For runtime LoRA, start with GPU:

```sh
build/native/turbocider generate models/Comfy-Org-Qwen-Image-2.1 \
  examples/requests/qwen21-viggle-runtime-lora-512.json
```

To opt into the reusable runtime-weight graph, use the **GPU template** and
override the route with a matching v2 manifest:

```sh
build/native/turbocider generate models/Comfy-Org-Qwen-Image-2.1 \
  examples/requests/qwen21-viggle-runtime-lora-512.json \
  --hybrid-mode runtime --ane-manifest path/to/runtime-v2/manifest.json
```

For the separate frozen, LoRA-capable graph:

```sh
build/native/turbocider generate models/Comfy-Org-Qwen-Image-2.1 \
  examples/requests/qwen21-viggle-runtime-lora-hybrid-512.json \
  --ane-manifest path/to/lora-fused/manifest.json
```

Do not pass `--hybrid-mode runtime` to the hybrid template: it explicitly
requests `lora_fused`, and conflicting overrides are rejected. A successful
`plan` checks the request contract, not actual model/artifact compatibility,
speed or visual quality. `--ane-manifest` consents to approximation;
`--hybrid-mode` alone does not. Neither bypasses the model's compatibility gates.

Q/K fusion, FP16 low-rank math, fixed chunks and caches are separate opt-ins;
do not enable them all as an unmeasured "fastest" preset. For current decisions
and matched measurements see [acceleration status](../../docs/status/acceleration.md).
For graph preparation and precise commands see the
[runtime-LoRA guide](../../docs/status/runtime-lora-acceleration-2026-09-28.md).
For 1–3-reference editing and repeatable timing use the
[validation guide](../../docs/status/runtime-ane-validation.md), not a shortened
base-generation request. Visual acceptance and reference fidelity still need
to be checked for each workload.
