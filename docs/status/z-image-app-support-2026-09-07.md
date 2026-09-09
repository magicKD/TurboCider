# Z-Image-Turbo App support — 2026-09-07

Branch: `dev-verify`. Host: Apple M4 Pro, 48 GB, macOS 26.6.

## App behavior

- Model center accepts the existing `comfy_z_image_turbo` directory. When text components are missing, it asks for the existing `FLUX.2-klein-4B` directory and binds its tokenizer and Qwen3 text encoder.
- Installation creates directory symlinks in the App state directory; the original checkpoint and shared text files are unchanged. No model downloads or weight copies are required.
- Z-Image defaults to 512×512, 9 steps, resident mode. Transformer LoRAs remain separate files with configurable strengths.
- Loading the current model retains LoRA, dimensions and acceleration. Native load-only preparation no longer denoises or writes an image; warmup retains the full computation path.
- Model center can import a saved App generation draft, restoring the model, prompt, seed, LoRA and acceleration while preserving other model registrations.
- GPU/ANE remains explicit on this host. The imported base configuration uses a8192; the imported LoRA configuration uses the independently exported a6144 artifact bound to the exact adapter and strength 1.0. A base manifest with an active LoRA falls back to GPU in App request construction.
- The acceleration panel reads the manifest asynchronously and reports its text capacity. File access waiting on macOS permissions cannot block SwiftUI rendering. Bucket 1056 at 512×512 allows 32 encoded text tokens. Core ML INT8 FFN approximations can change the image, and hardware ANE residency is not measured by the public API.

## App-only crash fixed

Real generation through `NativeJobStore` exposed a SIGBUS in `tc::sha256_file`. The shared provenance helper allocated a 1 MiB automatic buffer on a libdispatch worker whose stack was approximately 544 KiB. CLI benchmarks used the main thread and did not reveal this failure. The buffer now lives on the heap. A native regression test hashes a known input on a deliberately restricted 256 KiB pthread stack.

## Local artifacts and reproduction

Packaged App: `dist/TurboCider.app` (local ad-hoc signature; minimum macOS 26.2 inherited from the installed MLX library).

The real-model integration executable is `build/native/turbocider-z-image-app-tests` and takes six arguments: Comfy model directory, shared text model directory, LoRA file, compiled base manifest, compiled LoRA manifest and a fresh output directory. It runs the actual App state, request builder and asynchronous native job store.

Final run artifacts are under `outputs/z-image-app-20260907/final/`:

- `report.json`, `integration.log`, persisted `jobs.json` and `studio-draft.json`.
- Four 512×512 images: `base-gpu.png`, `base-gpu-ane.png`, `lora-gpu.png`, `lora-gpu-ane.png`.
- Four App-importable JSONs in `configurations/`, numbered `01-z-image-gpu` through `04-z-image-lora-gpu-ane`.

The configurations contain absolute paths for this machine, including the small model binding directory inside the run output. Keep those bindings and the previously compiled Core ML artifacts available when importing them. Alternatively register the original model directories in the App's model center.

The earlier CLI warm-run performance and quality comparison remains in [the M4 Pro benchmark](z-image-m4pro-verification-2026-09-07.md). First-use App request timing includes checkpoint identity hashing, LoRA fusion and Core ML loading, so it must not be compared directly with warm medians.

## Validation results

The final real-model integration passed all ten checks, including load without output, all four generation routes, adapter identity, rejection of the base artifact with LoRA, persistence and unload. All four resulting PNGs are byte-for-byte identical to their corresponding earlier CLI references; see `final/parity.json`. Each LoRA run applies all 180 projections; the mixed LoRA run reports `lora_identity_verified=true` and 288 Core ML predictions.

| App job-store route | Request wall (s) | Denoise (s) |
| --- | ---: | ---: |
| GPU | 23.553 | 20.056 |
| GPU/ANE a8192 | 25.049 | 15.864 |
| LoRA GPU, after load | 20.653 | 19.729 |
| LoRA GPU/ANE a6144 | 25.697 | 15.498 |

These are single integration requests, with different load/cache states, not a new acceleration benchmark. The App behavior suite additionally covers installation without source mutation, missing shared text, invalid Z-Image steps/LoRA roles, load retaining the draft, configuration import and preservation of existing model registrations when imported paths are empty. The 256 KiB worker-stack SHA-256 regression passes.

The repository-wide `make test` previously encountered two unrelated pre-existing LTX source-sync assertions (`ltx_transformer_io.c`, `ltx_shaders.metal` versus the sibling `ltx-mac` checkout); this change does not alter those vendored sources.
