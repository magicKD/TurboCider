# Usage reference

[Documentation](README.md) · [Getting started](GETTING_STARTED.md)

TurboCider's App, CLI, C / Swift SDK and Unix socket API share a native runtime.
Production inference does not start Python model workers. Offline conversion,
Core ML export and premerged LoRA preparation are separate tasks.

## Playground settings

Playground has an independently collapsible settings panel on the right, with
model directory, canvas dimensions, steps, seed, memory management, LoRA
strength/strategy, DiT cache and reference encoding. Each template saves its
own settings; switching templates does not change Creation. Synchronizing from
Creation is an explicit action and retains the template's images and instruction.

New image canvases and templates start at 512×512. Existing saved dimensions
remain unchanged; use the ratio presets or edit width/height for another size.
Qwen defaults to `component_staged`: encoder weights are released before DiT
execution, and DiT is released before final decoding. Some bounded conditioning
caches may remain; this does not mean every allocation is immediately zero.

Fast 512 reference encoding can be combined with DiT cache for GPU base/ordinary
LoRA editing at 512×512 and 20–40 steps with 1–3 references. These are explicit
approximations, separate from resizing the imported file or output canvas.
Six-step Viggle Turbo keeps DiT cache off; its preset is applied only when
requested. Incompatible parameters display a reason and recovery controls.

## Requests and CLI

```sh
dist/cli/turbocider --help
dist/cli/turbocider capabilities
dist/cli/turbocider doctor
dist/cli/turbocider models
dist/cli/turbocider plan request.json
dist/cli/turbocider generate /absolute/model request.json
dist/cli/turbocider batch /absolute/model first.json second.json
dist/cli/turbocider generate path/to/model request.json --ane-manifest path/to/compiled/manifest-HASH.json
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
For `plan`, `generate` and `batch`, the optional trailing
`--ane-manifest path/to/compiled/manifest-HASH.json` explicitly selects a
GPU/ANE manifest and consents to approximation for that invocation. It works
with schema 1 and 2, preserves the request JSON, and rejects a conflicting
`ane_manifest` already in the request. It does not select a quantization
profile by filename or override the runtime's checkpoint, shape, token,
adapter or artifact validation. Existing JSON-specified ANE execution and
other model-specific ANE profiles still work without this flag.

For the fastest **measured** Z-Image 512²/8-step W8A8 route (M4 Max 64 GB),
use `examples/requests/z-image-turbo-512.json` and
the compiled *1,024-image-row, 5,120-intermediate-channel* manifest. See the
[Z-Image GPU/ANE status and exact timing scope](Z_IMAGE_ANE.md). `plan` checks
request syntax and execution policy; a real generation is still required to
verify that the model and compiled ANE artifact are compatible.

For an adapter-independent Core ML FFN, select an explicitly exported
`lora_fused` **base-only** manifest and `--hybrid-mode lora_fused` for both
base and inference-time LoRA requests. Reuse the same manifest and resident
CLI session across adapters; do not export/compile the adapter into Core ML
or select `lora_merged`. The mode name is historical: **base FFN operators**
are fused, but no LoRA weights, rank, strength or adapter identity are baked
into the Core ML artifact. In the no-adapter request, the runtime supplies zero
gate/up corrections. With an adapter, GPU LoRA gate/up deltas enter before the
Core ML SiLU, and GPU computes down-LoRA from the Core ML hidden output;
non-FFN LoRA also stays on GPU. This is a separate opt-in graph, **not** the
default fastest base graph. Qwen-Image-2.1 runtime LoRA has not established
a whole-request speedup over GPU; Z-Image Turbo's explicit 4096-channel
runtime-LoRA graph has. See the [Qwen runtime LoRA experiments](../status/qwen21-runtime-lora-fused-2026-09-28.md)
and [Z-Image runtime LoRA experiments](../status/z-image-runtime-lora-fused-2026-09-28.md)
for exact geometry, accuracy limits and measured timings. A different LoRA
does not need a new Core ML compilation, but still needs independent schedule
and image-quality validation. The [route comparison and code map](../status/runtime-lora-acceleration-2026-09-28.md)
distinguish the fastest base path, complete runtime LoRA, and deliberately
incomplete diagnostics.

The explicit [runtime-weight Core ML route](../status/runtime-ane-integration-2026-09-28.md)
uses `--hybrid-mode runtime --ane-manifest MANIFEST` for resident
requests with approximation consent: Qwen21 BF16, Z-Image BF16, and the explicit
native Z-Image GGUF route. These checkpoint-independent manifests are **not**
interchangeable with frozen-base manifests. An optional graph-v2 exported with
`--lora-inputs` supports **inference-time LoRA on BF16 Z/Qwen** through activation
corrections; weight slots remain base-only. A v1 graph cannot accept LoRA.
GGUF remains base-only and streaming is unsupported. See the
[v2 computation and switch checks](../status/runtime-ane-lora-2026-09-28.md)
and the [maintained performance decision](../status/acceleration.md): Qwen
six-step LoRA has not established a meaningful advantage over optimized GPU,
so runtime is not its recommended fast path.
Z Q8_0 has paired whole-model measurements;
Q4_0 now has a [real-checkpoint smoke screen and padding fix](../status/runtime-ane-q4.md),
not broad qualification. Q4_1 still has only component coverage.
Affine staging converts packed weights to FP16; it is not INT8 ANE compute.
See the [GGUF integration and limits](../status/runtime-ane-integration-2026-09-28.md#z-原生-gguf-q8_0整模型接入与-chunk-筛选).
Existing LoRA paths and automatic defaults are unchanged. The component probe
remains a separate `make build-runtime-ane-probe` developer tool.
Use `make test-acceleration-contract` for no-inference route/report/host checks;
`make test-runtime-ane` explicitly runs small Core ML/MLX integration tests.
Within explicit `runtime`, `TURBOCIDER_RUNTIME_ANE_CHUNKS=auto` (the default)
adapts the row split; it does not select runtime mode for ordinary requests.
Qwen and Z-Image's unprofitable layers return to full GPU blocks, with periodic whole-block
timing probes. Stable profitable hybrid blocks avoid extra whole-block timing
fences; warmup, partition changes and periodic rechecks retain complete samples.
With profiling off, stable hybrid blocks also submit the GPU head asynchronously
and wait at the final output boundary. Sampling/profile paths retain synchronous
head timings. Async host waits overlap GPU work and must not be read as exposed
ANE latency; see the [timing semantics and regression evidence](../status/runtime-ane-async-join.md).
The maintained [acceleration decision table](../status/acceleration.md) separates
default/fastest base routes, optional complete LoRA paths and diagnostics, with
build-specific timing and quality evidence. Qwen six-step runtime LoRA/editing
still prefers GPU; the measured Z distill-patch workload benefits from runtime.
Do not combine independent approximation flags into an untested fastest preset.
Qwen's existing `TURBOCIDER_QWEN21_VIGGLE_LORA_FP16=1` changes only LoRA
rank matmuls and stays opt-in; it does not merge adapters or change the base graph.
The [matched three-reference experiment](../status/runtime-ane-qwen-lora-rank.md)
found a small GPU benefit but no auto-runtime win. The developer screen accepts
`--qwen-lora-fp16` to apply it equally to all compared routes and verify native
precision receipts; this is not a native CLI argument or a new default preset.
For explicit BF16 base experiments at 512², the [tile/chunk recipes](../status/runtime-ane-tiles.md)
use K=1024/N=512 with Qwen chunk320 and Z chunk352. Export a separate artifact
and supply its manifest; these are optional base recipes, not evidence of faster
LoRA/editing, and do not change exporter or product defaults.
For Qwen 1024² base, the [matched c1792/c320 comparison](../status/runtime-ane-qwen-chunks.md)
measured 153.050/165.000 s warm requests (1.078× versus c320, not a new matched
GPU/frozen comparison). Export a separate v2 graph with `--rows 1792`, K1024/N512
and `--lora-inputs`, then select it with the existing `--hybrid-mode runtime`
and `--ane-manifest MANIFEST` options; keep chunks `auto` and profiling off.
This geometry is optional, not a universal preset: short 512² sequences return
to GPU. A [three-reference runtime-LoRA screen and shared-graph switch](../status/runtime-ane-qwen-edit-chunks.md)
exercise real predictions. The short-batch screen was 0.977×; a newer long-resident
ABBA measured only 1.005× over all warm requests (1.018× in its predeclared late
window). The subsequent [matched Q/K-fusion campaign](../status/runtime-ane-qwen-qk.md)
measured 0.998× versus equally optimized GPU over all warm requests (1.016× in
the predeclared late window). This is not a universal faster preset; broader editing/adapter
qualification remains incomplete and the default route is unchanged.
Graph-v2 validates unused hidden outputs without copying them for base requests;
LoRA retains the full hidden and supplies gate/up corrections before SiLU.
Readiness/copy optimizations are internal to the explicit route, not new flags.
The [benchmark guide](../status/runtime-ane-validation.md) documents the shared
GPU/runtime/frozen screen, ordered 1–3 Qwen editing references, runtime LoRA,
ABBA trials and incomplete-result handling. Tool support is not performance
qualification; ordinary runtime-LoRA requests are not silently promoted to ANE.
Setting chunks to `0` instead preserves the split GPU boundary
for diagnostics and is **not** the ordinary GPU baseline. Keep
`TURBOCIDER_RUNTIME_ANE_PROFILE` unset for timings. Check for competing local
inference before benchmarking; wait rather than terminating unrelated processes.

The existing optional GPU Q/K norm-RoPE fusion can also be combined with explicit
Qwen runtime ANE at **512²**, resident, BF16 GPU and approximation consent:

```sh
TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE=1 \
  dist/cli/turbocider generate models/Comfy-Org-Qwen-Image-2.1 \
  examples/requests/qwen21-base-512.json \
  --hybrid-mode runtime --ane-manifest path/to/runtime-v2/manifest.json
```

Use a matching runtime graph (the measured base recipe is c320/K1024/N512),
not an ordinary frozen manifest. Omit the environment variable to disable this
GPU experiment; it does not enable LoRA FP16 or change base weights/SiLU.
It cannot be combined with paired RoPE, streaming or a W8A16 GPU suffix.
The newer build also admits **1024² resident base text-to-image only**;
1024² LoRA/editing and rectangular canvases remain rejected. Its long-sequence
kernel/contracts/full regression pass. The [matched 1024² comparison](../status/runtime-ane-qwen-qk-1024.md)
completed 12 trials/36 requests: with fusion enabled equally, GPU/runtime/frozen
took 182.961/147.349/160.718 s. Runtime was 1.242× GPU and 1.091× frozen for this
workload; the six reviewed images were visually close for this one prompt/seed.
This is not broader editing/LoRA, hardware-residency or memory-pressure qualification.
Use a matching runtime graph (c1792/K1024/N512 v2); frozen
1024² still needs its separate 4096-row graph and existing diagnostic gate.
The [1024² request and explicit CLI command](../../examples/requests/README.md#minimal-commands)
keep this measured fastest configuration optional, with GPU as the template default.
The validation tools expose `--qwen-qk-norm-rope` to apply the same
setting to **all** compared routes and check planned/actual kernel receipts;
that flag is not a native CLI option. Only the screen extends to 1024² base;
the runtime-LoRA shared-graph switch retains its 512² scope. The completed 512²/40-step base comparison
measured GPU/runtime/frozen at 41.742/35.675/29.068 s with fusion enabled equally
(1.170× runtime and 1.436× frozen speedup). See the matched report for workload,
memory and visual limits; this is not an automatic fastest preset.

The [request example index](../../examples/requests/README.md) separates GPU,
frozen base, shared-base LoRA and runtime-weight graph choices. Portable GPU requests are provided in
`examples/requests/z-image-runtime-lora-512.json` and
`examples/requests/qwen21-viggle-runtime-lora-512.json`. For the explicit Qwen
shared-base experiment use `qwen21-viggle-runtime-lora-hybrid-512.json` and
pass `--ane-manifest` at invocation; it intentionally has no machine-specific
manifest path. See the [commands and artifact requirements](../status/runtime-lora-acceleration-2026-09-28.md#便携-cli-示例).

For the small, visually checked Z-Image-only direct-FP16 delta experiment,
set `TURBOCIDER_Z_RUNTIME_LORA_DIRECT_FP16=1` on the CLI process while using
the explicit `lora_fused` manifest; omit the variable for the prior numerical
boundary. Qwen ignores this flag. It does not change the base model or merge
an adapter, and it has not been promoted to automatic routing.

To check adapter-file switching without compiling another Core ML graph,
`tools/validation/runtime_lora_shared_graph_switch.py` uses a temporary synthetic
second adapter and verifies base → adapter A → adapter B → base isolation;
this is a correctness test, not quality validation for a trained adapter B.
The switch runner requires matched requests (including prompt/seed), 512px
output and Qwen 6 / Z 8 steps; only output and adapter binding may differ.
Generation supports both modes; Qwen `runtime` also accepts 1–3 ordered ref512
editing inputs. The tool checks reference byte identity, operation/token counts,
graph/library/adapter identity and actual prediction increments on every request.
It sets the required ref512 diagnostic for the mixed base/adapter batch and pins
one chunk, so this is not an auto-scheduler benchmark. Its `--help` and host
contracts do not load MLX. Use the model screen for matched timings; see the
[switch-runner contract](../status/runtime-ane-validation.md#共图切换工具的边界).

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

GPU + ANE LoRA depends on the selected route. Explicit `lora_fused` uses a
frozen **base-only** graph with runtime activation corrections; `runtime` can
use the checkpoint-independent v2 interface described above. Neither merges
the adapter into base weights. Older merged routes instead require artifacts
bound to the same adapter identity and strength. These artifact interfaces are
not interchangeable.

## Core ML artifacts

GPU is the product default. Explicit `gpu_ane` requires
`allow_approximation=true` and compatible compiled partitions. Model, checkpoint,
shape, token capacity, precision and LoRA identity must all match.
Device-specific automatic routing remains limited to qualified workloads.

Exporting partitions from weights is offline preparation with a separate
toolchain. The production resource API compiles existing sources; it does not
convert arbitrary networks or run Python exporters.

See [local environment setup](ENVIRONMENT_SETUP.md) for the pinned export
environment and [FLUX preparation](FLUX_PREPARATION.md) for an end-to-end example
of exporting, compiling and registering FLUX.2 Klein 4B partitions.

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
The separate image-only W8A8 profile fixes **ANE input at 1,024 image rows**
and executes caption rows on the GPU; its fixed bucket is not the older
1,056-row all-token profile. The FFN GPU/ANE split still occurs over
intermediate channels for each image token. Use the exact marked compiled
manifest, not an uncompiled source `.mlpackage`.

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
