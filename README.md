<p align="center">
  <img src="assets/branding/logo.svg" width="760" alt="TurboCider — More from your Mac." />
</p>

<p align="center"><strong>Native inference. More of your Apple silicon.</strong></p>
<p align="center">SwiftUI Studio · Native CLI · C / Swift SDK · Local API</p>
<p align="center"><a href="README.zh-CN.md">中文</a> · <strong>English</strong> · <a href="docs/GETTING_STARTED.md">Getting started</a> · <a href="docs/PERFORMANCE.md">Benchmarks</a></p>

TurboCider is a local multimodal inference engine for Apple silicon. It turns unified memory into an opportunity for **heterogeneous execution: the CPU orchestrates native work while GPU and ANE process selected model partitions in parallel**. Shared output buffers, compiled GPU graphs and reusable sessions reduce copies and repeated preparation in compute-intensive image inference.

Create images and videos in a native SwiftUI studio, automate jobs with the CLI, or embed the same runtime through C / Swift bindings and a local task API. Models and generated media stay on your Mac.

## What makes it different

- **Hardware-aware parallelism:** Metal / MLX on the GPU and Core ML FFN partitions eligible for the Neural Engine, coordinated by the CPU.
- **Measured performance and fidelity:** FLUX.2 Klein 4B at 512×512, 4 steps on M4 Max 64 GB achieved **1.39× throughput and 28% lower generation time**, with GPU-to-hybrid pixel cosine similarity **0.999840**.
- **Reuse across requests:** resident sessions, prompt encoding caches, compiled graphs and Core ML artifact caches. Z-Image's flexible ANE partitions support up to 512 encoded text tokens at 512×512.
- **One runtime:** native App, CLI, C / Swift SDK and a local Unix socket job service share execution contracts.
- **Explicit capabilities:** image generation, image editing and video generation are offered only where the executor supports them. LoRA and GPU / ANE controls remain explicit.
- **Shared model management:** App / CLI registrations, ModelScope / Hugging Face download previews and compatible text-component reuse. Start the local API from the App and manage regenerable text-tensor retention.

The ANE routes use validated INT8 partitions: they are high-fidelity approximations, not bit-exact inference. Gains depend on hardware, model and shape. GPU is the default. Core ML compute-unit selection is not proof of ANE occupancy. These are reproducible workload results, not a universal SOTA claim. See the measurements below and [benchmark methodology](docs/PERFORMANCE.md).

## Measured performance

Median warm request wall times for the named workloads below, excluding first model loading and compilation. Throughput multiplier = GPU baseline time / hybrid time.

| Model / device / workload | TurboCider GPU | GPU + ANE | Throughput |
|---|---:|---:|---:|
| FLUX.2 Klein 4B · M4 Max 64 GB · 512² · 4 steps | 2.266 s | **1.628 s** | **1.39×** |
| Z-Image Turbo · M4 Max 64 GB · 1024² · 9 steps | 36.369 s | **30.001 s** | **1.21×** |
| Z-Image Turbo · M4 Pro 48 GB · 512² · 512 tokens · 9 steps | 23.964 s | **20.425 s** | **1.17×** |

[Methodology, quality metrics and sources](docs/PERFORMANCE.md). Support for 512 text tokens does not imply support for arbitrary image resolutions.

## Build and run

Use an Apple silicon Mac, a working macOS SDK / Swift / Clang toolchain and MLX 0.32.x. Full Xcode is recommended; an existing Command Line Tools installation can be selected with `DEVELOPER_DIR` and `SDKROOT`. Python 3.11 is used for setup and offline tools; all production inference paths, including Wan, are native. Python export and merge tools are development-only.

```sh
make setup
make package
make test
open dist/TurboCider.app
```

Register an existing model folder in the App, choose an operation, and generate with GPU first. Compatible Z-Image text components can be linked from a local FLUX.2 Klein 4B installation. Configure ANE only after preparing artifacts for the actual model, shape and LoRA identity. Current packages are locally ad-hoc signed, not notarized releases.

For automation, use `dist/cli/turbocider generate MODEL_DIRECTORY REQUEST.json`. `batch` reuses a session; `serve` exposes the local job API. See [getting started](docs/GETTING_STARTED.md) and the [request / SDK / API reference](docs/USAGE.md).

## Model capabilities

`turbocider models` reports the authoritative executable operations. Availability below describes the current runtime, not every upstream model feature.

| Model | Output | Input | Notes |
|---|---|---|---|
| FLUX.2 Klein 4B / 9B | Image | Text; image transform; reference editing | 4B has qualified hybrid profiles; 9B performance coverage is incomplete |
| Z-Image Turbo | Image | Text | Comfy / Diffusers layouts; in-memory LoRA; flexible ANE inputs |
| MiniMax H3 Turbo | Video | Text; keyframes; references | Native Metal / MPS; artifact-gated ANE; media tools required |
| LTX 2.5 Distilled | Video | Text, video-only | Image input and audio remain gated; staged residency |
| Wan 2.1 1.3B QAD | Video | Native MLX / Core ML, 832×480, 3 steps | Offline-converted TAEHV; verified premerged LoRA only |

Weights retain their upstream licenses. Large model files and generated outputs are not included in the repository.

## Project map

| Directory | Responsibility |
|---|---|
| `native/` | Inference core, model executors, Metal / MLX / Core ML backends |
| `apps/` | SwiftUI App and native CLI |
| `services/` | Persistent local task service and model/cache management helper |
| `bindings/` | C ABI and Swift SDK |
| `profiles/`, `examples/` | Hardware policies and runnable request examples |
| `tools/`, `tests/` | Setup, packaging, conversion, regression and benchmark tools |
| `assets/branding/` | Vector identity and reproducible App icon |
| `docs/` | User guides, design and retained development records |
| `experimental/video/` | Frozen migration snapshots, excluded from shipping builds |

[Contributing](CONTRIBUTING.md) · [Third-party notices](native/THIRD_PARTY_NOTICES.md) · [Current validation records](docs/status/README.md)

[Model library](docs/MODEL_LIBRARY.md) · [Local API](docs/LOCAL_API.md) · [Cache and memory](docs/CACHES.md)

Original TurboCider code is provided under [MIT](LICENSE). Third-party code
retains its [original notices](native/THIRD_PARTY_NOTICES.md); model weights
remain subject to their upstream terms.

Native-only boundaries and GGUF limitations: [refactor summary](docs/status/native-refactor-summary-2026-09-09.md).
