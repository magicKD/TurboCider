<p align="center">
  <img src="assets/branding/logo.svg" width="760" alt="TurboCider — More from your Mac." />
</p>

<p align="center"><strong>Native inference. More of your Apple silicon.</strong></p>
<p align="center">SwiftUI Studio · CLI · C / Swift SDK · Local HTTP API · ComfyUI</p>
<p align="center"><a href="README.zh-CN.md">中文</a> · <strong>English</strong> · <a href="docs/public/GETTING_STARTED.md">Getting started</a> · <a href="docs/benchmark/PERFORMANCE_REPORT.md">Benchmarks</a></p>

TurboCider is a local multimodal inference engine for Apple silicon. Create images and videos in a native macOS App, automate jobs through the API, or connect it to ComfyUI workflows. Models and generated media stay on your Mac.

**3.40× speedup over the measured Unsloth Q4_K_M GGUF image-generation route, using 70.6% less time; 1.76× for LTX video generation versus ComfyUI.** Native Metal / MLX GPU inference and optional GPU + ANE execution put more of the same Mac to work.

## Performance highlights

On the same **Apple M4 Max · 64 GiB**, comparing complete generation time, including encoding, sampling, decoding and output.

| Model and workload | Reference framework / format | Reference time | TurboCider | Speedup / time saved |
|---|---|---:|---:|---:|
| Qwen-Image-2.1 · 512×512 · 40 steps | **Unsloth Q4_K_M GGUF**, sd-server Metal | 125.365 s | **36.870 s** · GPU + ANE | **3.40× / 70.6%** |
| Qwen-Image-2.1 · 512×512 · 40 steps | **ComfyUI v0.39**, BF16 source | 48.350 s | **31.059 s** · GPU + ANE | **1.56× / 35.8%** |
| Z-Image Turbo · 512×512 · 8 steps | **ComfyUI v0.39**, BF16 source | 8.535 s | **5.780 s** · GPU + ANE | **1.48× / 32.3%** |
| LTX 2.5 · 704×448 · 97 frames · 11 steps, with audio | **ComfyUI v0.39**, same model sources | 112.230 s | **63.848 s** · GPU quality configuration | **1.76× / 43.1%** |

**The GGUF comparison uses the same Unsloth Q4_K_M encoder and DiT weights with a BF16 VAE**: Unsloth runs through a pinned sd-server Metal backend, while TurboCider uses native GPU + ANE execution. ComfyUI comparisons use the faster measured MPS configuration for each task. LTX uses GPU alone, saving approximately **48 seconds per video**.

Image results are warm medians. The ComfyUI table reuses TurboCider's best recorded configurations across measurement windows; LTX's ComfyUI result is a single warm observation. A separate paired Z-Image experiment measured **1.46×**. GPU + ANE configurations require preparation and explicit opt-in and use numerical approximation (Private ANE for Qwen GGUF). GPU is the default; warm gains do not imply cold-start gains or pixel equality.

[Unsloth GGUF and full performance report](docs/benchmark/PERFORMANCE_REPORT.md) · [ComfyUI model comparisons](docs/benchmark/COMFYUI_MODEL_COMPARISON.md) · [Paired ComfyUI validation](docs/benchmark/COMFYUI_MAC.md)

## Features

- **Image and video creation:** text-to-image, image transforms, reference editing, text-to-video and image-to-video, according to each model's capabilities.
- **LoRA and generation controls:** manage adapters, strengths, seeds, dimensions and steps, with supported runtime or in-memory merge strategies.
- **Local API:** start an HTTP service from the App; discover models, validate requests, submit jobs, track progress, cancel and download results. A Unix socket service, Python HTTP client and C / Swift SDK are also available.
- **ComfyUI / Comfy Desktop:** model, LoRA, generation, editing and native video request nodes. Feed generated images into standard preview, save and compositing nodes; TurboCider manages the weights without a second model installation in ComfyUI.
- **Shared model library:** register existing local models, preview Hugging Face / ModelScope downloads and reuse compatible text components across App and CLI.
- **Repeated jobs and memory management:** resident sessions, batches, prompt caches, compiled GPU graphs, and model-specific staged or streamed loading.

## App preview

![TurboCider macOS App](assets/app/app.png)

## API and ComfyUI

Open **Local API** in the App and click **Start**. The default endpoint is `http://127.0.0.1:8765`; the page includes curl and Python examples.

```sh
curl http://127.0.0.1:8765/v1/health
curl http://127.0.0.1:8765/v1/catalog
```

Install the nodes into an existing ComfyUI installation, then restart ComfyUI:

```sh
python3 tools/comfyui/install.py /path/to/ComfyUI
```

Connect **Model (API) → Generate Image → Save Image**, using the App's HTTP URL in the Model node. Add **LoRA (API)** when needed. The integration composes at the image boundary; TurboCider owns sampling.

[ComfyUI and HTTP guide](docs/public/COMFYUI.md) · [Chinese tutorial](docs/public/COMFYUI_TUTORIAL.zh-CN.md) · [Local job API](docs/public/LOCAL_API.md)

## Build and run

Use an Apple silicon Mac, a working macOS SDK / Swift / Clang toolchain and MLX 0.32.x. A working Command Line Tools installation is sufficient for the validated build; select it with `DEVELOPER_DIR` and `SDKROOT` if necessary. Install arm64 CPython 3.11 **before** `make setup`; Python is used for setup and offline tools, while all production inference paths, including Wan, are native.

On a fresh Mac, follow [local environment setup](docs/public/ENVIRONMENT_SETUP.md) for Python bootstrapping, pinned dependency environments, SDK selection, optional FFmpeg tools and troubleshooting. The minimum runtime macOS version follows the linked MLX library, not the SDK version.

```sh
make setup
make package
make test
open dist/TurboCider.app
```

`make setup` installs the pinned packages into `.venv` and a separate offline Core ML environment. Model weights and ANE artifacts require additional preparation; see the [FLUX.2 Klein 4B walkthrough](docs/public/FLUX_PREPARATION.md).

Choose image or video creation in the App; a compatible executor is selected automatically. Register its model folder and generate with GPU first. Compatible Z-Image text components can be linked from a local FLUX.2 Klein 4B installation. Configure ANE only after preparing artifacts for the actual model, shape and LoRA identity. Current packages are locally ad-hoc signed, not notarized releases.

For automation, use `dist/cli/turbocider generate MODEL_DIRECTORY REQUEST.json`. `batch` reuses a session; `serve` exposes the local job API. See [getting started](docs/public/GETTING_STARTED.md) and the [request / SDK / API reference](docs/public/USAGE.md).

## Model capabilities

| Model | Main capabilities |
|---|---|
| Qwen-Image-2.1 | Text-to-image, multiple-reference editing and LoRA; selected six-step Viggle adapters |
| FLUX.2 Klein 4B / 9B | Text-to-image, transforms, reference editing and LoRA; qualified hybrid profiles for 4B |
| Z-Image Turbo | Text-to-image, LoRA and optional GPU / ANE execution |
| MiniMax H3 Turbo | Video generation, keyframes and references; media tools required |
| LTX 2.5 Distilled | Text-to-video, image-to-video and optional audio output; backend/configuration limits apply |
| Wan 2.1 1.3B QAD | Native three-step video generation; offline-converted assets and verified premerged LoRA required |

`turbocider models` and request `plan` report the executable capabilities. LoRA, input and acceleration options vary by model. See the [model and request reference](docs/public/USAGE.md#model-capabilities), [Qwen guide](docs/public/QWEN_IMAGE_21.md) and [GPU/ANE acceleration status](docs/status/acceleration.md).

## Project map

| Directory | Responsibility |
|---|---|
| `native/` | Inference core, model executors, Metal / MLX / Core ML backends |
| `apps/`, `services/` | SwiftUI App, CLI, local job service and model library |
| `bindings/`, `integrations/comfyui/` | C / Swift SDK, Python HTTP client and ComfyUI nodes |
| `tools/`, `tests/` | Builds, conversion, regression tests and benchmarks |
| `docs/public/`, `docs/benchmark/` | User guides and verifiable performance records |
| [`techniques/`](techniques/README.md) | Reusable Apple Silicon optimization wiki: Metal, ANE, GPU/ANE, memory, profiling and model onboarding |

[Documentation](docs/public/README.md) · [Model library](docs/public/MODEL_LIBRARY.md) · [Cache and memory](docs/public/CACHES.md)

Original TurboCider code uses the MIT license. Third-party code and model weights retain their respective terms. Large model files and generated outputs are not included in the repository.
