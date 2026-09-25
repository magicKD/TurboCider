# TurboCider documentation

Run image and video models locally on Apple silicon through the native Studio,
CLI, C / Swift SDK or local job API.

| Guide | Start here when you want to… |
|---|---|
| [Local environment setup](ENVIRONMENT_SETUP.md) | Prepare Python, Apple tools, pinned dependencies and media tests on a fresh Mac |
| [Getting started](GETTING_STARTED.md) | Build, register weights and create your first output |
| [FLUX.2 Klein 4B preparation](FLUX_PREPARATION.md) | Download and inspect weights, export/compile ANE partitions and plan a first comparison |
| [Z-Image 512² GPU/ANE](Z_IMAGE_ANE.md) | Use the fastest measured local W8A8 hybrid manifest from the CLI, with its exact qualification limits |
| [Usage reference](USAGE.md) | Configure requests, model capabilities, LoRA, ANE and SDKs |
| [Model library](MODEL_LIBRARY.md) | Register, inspect, download and share model components |
| [Local API](LOCAL_API.md) | Submit jobs and reuse a persistent session |
| [Caches and memory](CACHES.md) | Understand residency, prompt reuse and safe cleanup |
| [Performance](PERFORMANCE.md) | Read selected benchmark highlights with their measurement limits |
| [Benchmark evidence](BENCHMARKS.md) | Inspect dates, samples, baselines and quality evidence |

These English guides are self-contained. All relative documentation links
remain within this public documentation set; no development notes are required.
Commands assume the repository root unless stated otherwise. English UI names
in these guides describe the controls; the current App may display localized
labels.

GPU is the default. Optional GPU + ANE execution requires compatible artifacts
and allows quantization differences. Model weights are obtained separately and
retain their upstream terms. Locally packaged Apps use ad-hoc signatures, not
notarized release signatures.
