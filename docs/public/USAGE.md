# Usage reference

[Documentation](README.md) · [Getting started](GETTING_STARTED.md)

TurboCider's App, CLI, C / Swift SDK and Unix socket API share a native runtime.
Production inference does not start Python model workers. Offline conversion,
Core ML export and premerged LoRA preparation are separate tasks.

## Requests and CLI

```sh
dist/cli/turbocider doctor
dist/cli/turbocider models
dist/cli/turbocider plan request.json
dist/cli/turbocider generate /absolute/model request.json
dist/cli/turbocider batch /absolute/model first.json second.json
```

Use the model catalog's executable operations, not the upstream model's entire
feature list. A plan checks request compatibility; loading checks actual weights.
Registered installations can be selected as `@model-id`; see
[the model library](MODEL_LIBRARY.md).

Schema 2 separates inputs, outputs, sampling and execution. Schema 1 remains
supported, as shown in [Getting started](GETTING_STARTED.md#5-use-the-cli).
Save the following as a request for FLUX reference editing:

```json
{
  "schema_version": 2,
  "model": "flux2-klein-4b",
  "operation": "image.edit",
  "inputs": [
    {"kind": "text", "role": "prompt", "text": "Place this fox in a snowy forest, preserving its appearance."},
    {"kind": "image", "role": "reference", "path": "/absolute/reference.png"}
  ],
  "outputs": [
    {"kind": "image", "path": "/absolute/result.png", "width": 512, "height": 512}
  ],
  "sampling": {"steps": 4, "seed": 42},
  "execution": {"policy": "gpu", "residency": "resident"},
  "parameters": {"dynamic_text": true}
}
```

- `image.generate`: text input only.
- `image.transform`: one `init_image`; optional image-input `strength`.
- `image.edit`: 1–8 ordered `reference` images for supported FLUX models.
- Video inputs, frames, audio and supported operations are model-specific.

For FLUX image transformation, strength describes preservation of the original:
a positive value starts at `max(1, floor(steps * strength))`; 1 skips DiT and
0 runs all steps. Reference-editing strength is not a noise-control parameter.
Reference order affects positional encoding.

General image limits are 64–2048 pixels per dimension, multiples of 16,
seed 0–2147483647 and 1–50 steps. Model-specific limits may be narrower.
Dynamic text is capped at 512 encoded tokens, including template tokens.

stdout contains final JSON; stderr contains progress events. Ctrl-C cancels at
a safe boundary, with exit code 2 for cancellation. Once an output has been
atomically committed, the operation succeeds. Choose unique output paths.
Batch requests must use the same model.

## Model capabilities

| Model ID | Executable scope | Important limits |
|---|---|---|
| `flux2-klein-4b`, `flux2-klein-9b` | Text-to-image, transformation and reference editing | 4B has qualified hybrid profiles; 9B hybrid coverage is incomplete |
| `z-image-turbo` | Text-to-image | Resident; default 9 steps; 1–50 configurable |
| `z-image-turbo-gguf` | Native MLX text-to-image | Resident; supported tensor types only; default 9 steps |
| `llada-image-turbo` | Native text-to-image | Editing and independent LoRA are not qualified; hybrid remains explicit |
| `minimax-h3-turbo` | Native video generation, keyframes and references | Correct installation and provenance required for the selected operation |
| `ltx-2.5-distilled` | Text-to-video, without audio | Image input and audio remain gated |
| `wan2.1-1.3b-qad` | Native text-to-video | 832×480, 3 steps, 16 fps, 5–81 frames in the form 4n+1; no audio |

Run `turbocider models` to check exact IDs and operations in your build.

### Z-Image layouts and steps

A complete Diffusers directory works directly. A compatible Comfy directory
contains `models/diffusion_models/z_image_turbo_bf16.safetensors` and
`models/vae/ae.safetensors`. Missing text components can be linked from a
compatible Qwen3 installation using the model center.

Both Z-Image formats default to 9 steps when omitted. Set `steps` in schema 1
or `sampling.steps` in schema 2. The App exposes the same 1–50 range and reset
control. Preparing or reselecting the current model preserves custom steps.
The benchmark numbers describe their recorded step counts, not arbitrary ones.

Native GGUF supports Q8_0, Q4_0, Q4_1, F16, BF16 and F32 tensor types.
Mixed K-quant and GGUF streaming are not supported. Selection is based on tensor
metadata, not a filename's quantization label. No external GGUF server is used.

A GGUF installation needs a transformer file, `tokenizer/` and compatible
text encoder / VAE components. Supported component locations include
`split_files/text_encoders/qwen_3_4b.safetensors` and
`split_files/vae/ae.safetensors`, or Diffusers `text_encoder/` and `vae/`
directories. A directly selected GGUF file uses its parent as the component
root. For several GGUF files, select one explicitly or specify `model_variant`.
Components are not discovered from unrelated neighboring workspaces.

### Wan

The native pipeline uses MLX UMT5, DiT, a prepared TAEHV decoder and AVFoundation
output. A prepared installation contains:

- `mlx_dit.json` and `mlx_dit.safetensors`
- `tokenizer/tokenizer.json`
- `text_encoder/` configuration and safetensors shards
- `vae/taew2_1.safetensors` with native conversion metadata

End users need a compatible prepared bundle. Conversion is offline, not a
fallback Python worker. GPU supports optional `compile_gpu=true`; this and
hybrid execution are mutually exclusive.

Explicit hybrid execution requires a complete 30-block manifest with
rows 32760, hidden width 1536, intermediate width 8960 and ANE/GPU split
4096/4864. It supports 832×480×81 only. The artifact schema is
`turbocider-wan-ane-mlp-v1`. Hybrid is not a promised speedup for Wan.

### LTX video

The public executor accepts `video.generate` with `audio=false` and defaults
to `component_staged`. The native finalizer decodes in a clean process to avoid
inheriting the denoiser's allocator state. Conditioning can be reused on disk
when model, tokenizer and prompt identity match.

Example request, with a unique output path:

```json
{
  "model": "ltx-2.5-distilled",
  "operation": "video.generate",
  "prompt": "A slow camera move over a sunlit mountain lake.",
  "width": 704,
  "height": 448,
  "frames": 97,
  "fps": 24,
  "steps": 11,
  "audio": false,
  "execution": "gpu",
  "residency": "component_staged",
  "seed": 42,
  "output": "/absolute/outputs/lake-42.mp4"
}
```

Audio-asset inspection through `tc_ltx_audio_preflight_json` does not enable
audio generation. Verified assets and a qualified end-to-end executor are
different conditions. Do not interpret individual decoder tests as full
audio/video readiness.

### H3 residency

H3 streamed execution can use `memory_budget_bytes` to retain a prefix of
DiT blocks while streaming the remainder. This is a working-set target, not a
hard process-memory cap. Invalid budgets are rejected. Without a budget, the
existing two-slot streaming path is used.

Results expose `ssd_pinned_blocks`, `ssd_streamed_blocks`,
`ssd_request_bytes_read`, `ssd_request_read_seconds`,
`ssd_request_wait_seconds`, `cache_prepared_dit` and `cache_video_decoder`.
The session may reuse DiT and conditioning without retaining the video decoder.
No cross-framework video speed claim follows from those counters alone.

## LoRA

Specify adapters with their path, role and strength. At the top level of either
request schema, `lora_strategy` can select a supported strategy:

| Model | Strategy boundary |
|---|---|
| FLUX | Native in-memory merge |
| Z-Image safetensors | In-memory merge by default; explicit low-rank inference branch supported |
| Z-Image GGUF | Native packed-base low-rank branch by default; explicit in-memory merge supported |
| H3 / LTX / Wan | Offline premerged checkpoints with verified provenance; no runtime disk merge |

`auto` resolves through the model descriptor's default. An unsupported explicit
combination fails. A strategy is not meaningful without an active adapter.
In-memory merging of quantized weights can materialize dense tensors and increase
memory use; packed low-rank branches trade memory, performance and rounding.

Z-Image accepts transformer adapters with strength from -8 to 8. An adapter
disabled in the App remains configured but is not sent to inference.

Premerged artifacts must match the base, adapter and merged-output hashes and
the requested role/strength. Merely renaming a checkpoint is not preparation.
Wan accepts at most one verified premerged transformer adapter and currently
restricts LoRA to GPU.

GPU + ANE with LoRA requires an in-memory-merge-compatible route and artifacts
bound to the same adapter identity and strength. Base-only artifacts are not
interchangeable with LoRA-bound artifacts.

## Core ML artifacts

GPU is the product default. Explicit `gpu_ane` requires
`allow_approximation=true` and compatible compiled partitions. Model, checkpoint,
shape, token capacity, precision and LoRA identity must all match.
Device-specific automatic routing remains limited to qualified workloads.

Exporting partitions from weights is offline preparation with a separate
toolchain. The production resource API compiles existing sources; it does not
convert arbitrary networks or run Python exporters.

Save a compilation request as `coreml-request.json`:

```json
{
  "action": "compile",
  "model": "z-image-turbo",
  "source_manifest": "/absolute/partitions/source/manifest.json",
  "cache": "/absolute/coreml-cache"
}
```

```sh
dist/cli/turbocider coreml coreml-request.json
dist/cli/turbocider compile-coreml /absolute/block.mlpackage /absolute/cache
```

Select the returned compiled manifest in the App before enabling ANE.
Compilation is cached by source content and system/device identity. First-use
device specialization can still take time. Do not alter metadata to claim
capacity absent from the actual graph.

For Z-Image at 512×512, 1024 image tokens plus up to 512 encoded text tokens
require up to 1536 total rows. Enumerated 1056–1536-row partitions cover this
range in 32-row increments. A fixed 1056-row cache allows only 32 padded text
tokens. Larger images require larger compatible partitions. Continuous-range
shapes are not the qualified default.

The resource interface also supports `inventory`, `delete_artifacts`,
`clear_compiled` and `clear_runtime`. Destructive actions default to preview;
review the returned plan and use the required confirmation fields before
applying it. See [cache boundaries](CACHES.md).

Core ML invocation counts and compute-unit preferences do not establish actual
per-operator ANE residency. INT8 partitions are approximate, not bit-exact.
See [performance and fidelity](PERFORMANCE.md).

## Local task service

Use the [Local API guide](LOCAL_API.md) for RPC fields, a complete Python client,
session reuse, cancellation and service ownership. The protocol uses a user-only
Unix socket, one newline-terminated JSON request per connection, and JSON
success/error responses. It is not an HTTP API.

## SDK ownership

The C ABI is exposed by `bindings/c/include/turbocider/turbocider.h`.
`tc_engine_create_model` selects a module; `tc_engine_generate` is synchronous.
Free returned strings using `tc_string_free`. Callbacks run synchronously on
the generation thread: do not block them or reenter generation. Free an engine
only after all calls finish. Cancellation may be requested from another thread.

The Swift wrapper at `bindings/swift/TurboCiderNative.swift` provides async
generation, planning and catalog access. Use `engine.cancel()` explicitly;
Swift task cancellation is not automatically equivalent to native cancellation.

## Studio history and memory

The App saves drafts and job history. Generated images in the App-owned output
directory can be moved to Trash while retaining their job records. Removing a
job record is separate from removing its output. Cancel active work first.

Loading prepares the selected session without exporting an image. Warmup is a
separate operation. Unloading releases the embedded session, not original model
files. An App-owned API session is released by stopping the service.
[Cache and memory](CACHES.md) explains counters, retention and deletion scope.
