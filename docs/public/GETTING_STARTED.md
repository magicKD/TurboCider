# Getting started

[Documentation](README.md) · [Usage reference](USAGE.md)

## 1. Build or use a local package

Use an Apple silicon Mac with a working macOS SDK, Swift / Clang toolchain and
compatible MLX 0.32.x dependencies. Python 3.11 is used for setup and offline
preparation, not production model inference. From the repository root:

```sh
make setup
make package
dist/cli/turbocider doctor
```

Setup installs development dependencies, not model weights. Packaging creates
`dist/TurboCider.app` and `dist/cli/`. Keep the CLI directory intact: it includes
required dynamic libraries, Metal resources and native helpers.

The minimum macOS version follows the linked MLX library's deployment target.
A package built with newer dependencies may require a newer OS. Local ad-hoc
signing is not Developer ID notarization. If the default Xcode selection is
unusable but Command Line Tools is installed:

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools \
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk make package
```

An existing compatible MLX installation can be selected with `MLX_ROOT`.
Generation needs a macOS session with access to Metal; a sandbox device-access
failure does not establish that a model is invalid.

## 2. Choose what to create

Open `dist/TurboCider.app`. The Studio shows **Image generation** and
**Video generation** before you choose a model. Select a creation type, then
an operation such as text-to-image, reference editing or text-to-video.

TurboCider selects a compatible executor when necessary, preferring a model
with a configured path. The model picker is filtered by creation type.
An automatically selected path still needs an installation check. Switching
models resets model-specific parameters, LoRA and acceleration settings;
review them before generating. Automatic operation-driven switches retain
your prompt and imported assets.

## 3. Register or download weights

Open the model center. Search by name, model ID or registered path, choose a
folder, inspect its installation and select it for creation. Browsing model
details alone does not change the draft.

Existing weights stay where they are. For Z-Image, supported choices include a
complete Diffusers layout and a compatible Comfy layout. Missing Qwen3 text
components can be linked from a compatible local FLUX.2 Klein 4B installation.
Do not move linked source directories afterward.

If you need weights, preview the download file list and required space first.
ModelScope is the default provider; Hugging Face is also supported. Downloads
are explicit and cancellable. See [model management](MODEL_LIBRARY.md) for
registration, sharing, integrity checks and model-specific preparation.

A saved registration is not proof that a model can generate. The library's
inspection checks files and metadata; the native loader validates tensors.

## 4. Generate with GPU first

Enter a prompt, choose dimensions and a seed, and use the generation button.
Add images only for an operation that accepts them. Video models also expose
frame count, frame rate and supported audio options.

Z-Image Turbo, including GGUF, defaults to **9 steps** and accepts **1–50**.
The primary parameter panel exposes the step field and a reset button.
Custom steps persist in the draft. Non-default step counts have not all been
qualified for image quality or acceleration.

LoRA support depends on the executor. Disabling an adapter keeps its settings
but excludes it from the request. Keep GPU selected for the first successful
output; configure ANE separately afterward.

Progress reports real phases and sampling steps. Decode and export follow
denoising. A compatible resident session can reuse text conditioning when only
the seed changes. Releasing memory closes that session without deleting weights.
See [cache behavior and cleanup](CACHES.md).

## 5. Use the CLI

Save this as `request.json`, replacing the output path:

```json
{
  "model": "z-image-turbo",
  "prompt": "A red fox in a snowy forest, soft morning light.",
  "width": 512,
  "height": 512,
  "steps": 9,
  "seed": 42,
  "execution": "gpu",
  "output": "/absolute/outputs/fox-42.png"
}
```

```sh
dist/cli/turbocider models
dist/cli/turbocider plan request.json
dist/cli/turbocider generate /absolute/Z-Image-Turbo request.json
```

A successful plan validates a request, not the presence of every model file.
Use unique output paths. Relative paths resolve from the working directory.

For several requests using the same model:

```sh
dist/cli/turbocider batch /absolute/Z-Image-Turbo first.json second.json
```

Batch generation reuses the session. Separate one-shot CLI processes do not.
stdout contains result JSON; stderr contains progress events. Ctrl-C requests
cancellation at a safe execution boundary. See [requests and models](USAGE.md)
for image editing, video and GGUF layouts.

## 6. Add ANE when artifacts are ready

ANE requires partitions exported for the actual checkpoint, geometry and
LoRA identity. Export is offline preparation; the production App does not run
a Python exporter. Obtain compatible source partitions, then compile them
through the App's acceleration/cache panel or the
[Core ML resource interface](USAGE.md#core-ml-artifacts).

Select the compiled manifest and explicitly enable ANE. Editing manifest
dimensions cannot enlarge a compiled graph. First-use specialization can
still be slow even when compilation is cached. See
[measured workloads](PERFORMANCE.md) before choosing a hardware policy.

## 7. Automate with the local API

The App's local API page starts and stops an owned service and displays its
Unix socket path. Starting it releases the embedded Studio session; stop the
service before returning to embedded generation.

For an independently managed foreground service:

```sh
dist/cli/turbocider serve /tmp/turbocider.sock /absolute/service-state
```

See [Local API](LOCAL_API.md) for a complete client, RPC fields and cancellation.
This is a Unix socket API, not an OpenAI-compatible HTTP endpoint.

## Troubleshooting

| Symptom | Next step |
|---|---|
| Missing weights or broken links | Re-register the source and inspect its components |
| ANE geometry mismatch | Obtain partitions with enough image and text capacity |
| ANE fails after a LoRA change | Use artifacts bound to that adapter and strength |
| First request is slow | Separate loading, compilation and warm request timing |
| GPU is busy | Finish or stop the other process holding the inference lease |
| Memory stays allocated | Check residency and release the idle session |
| Video input or audio is unavailable | Check executable operations, not upstream model features |
