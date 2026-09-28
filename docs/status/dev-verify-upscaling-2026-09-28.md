# dev-verify integration and image upscaling — September 28, 2026

`dev-verify` was fast-forwarded from `a39f833` to `dev` at `8bdc89f`. `main` is an ancestor; a merge-tree preflight found no conflicts. The native engine was rebuilt from this tree with Z-Image Turbo's default changed to 8 steps. The App, CLI default and reset action obtain this default from the model descriptor; existing saved steps are preserved.

## Implementation

- `ImageUpscaler.swift`: native Swift/Core ML loading, FP16/FP32 NCHW adapters, reflected tile padding, overlap blending, orientation normalization, alpha preservation and PNG export. There is no Python inference subprocess or runtime dependency.
- `StudioState.swift`: persistent local model path, automatic post-generation toggle and independent GPU / CPU+ANE choice. New drafts retain the installed model and device choice while resetting automatic processing.
- `JobStore.swift`: upscaling shares the single-job admission guard, cancellation, history and transactional image publication. The original generation survives any later upscaling failure.
- `App.swift`: a dedicated super-resolution page with x2/x4 selection, download/processing links, preload/release controls, plus result and automatic post-generation upscaling. Model selection reads an external model in place.

Only the user-authorized x2plus and x4plus Core ML packages were downloaded (about 30 MiB per archive). Other models were not downloaded. URL and SHA-256 are recorded in `outputs/dev-verify-20260928/upscaler-model-provenance.json`.

## GPU versus CPU/ANE

Apple M4 Pro / 48 GiB; same x4plus FP16 model, same 512×512 RGB input, 2048×2048 PNG output. Six independent processes ran in GPU, ANE, ANE, GPU, GPU, ANE order. Each loads the model and renders one image; OS compilation caches were not cleared. These are process-first-request measurements, not claims about a new Mac with empty system caches.

| Configuration | Model loading median | Image processing median | Total median |
| --- | ---: | ---: | ---: |
| CPU/GPU | 0.574 s | 1.997 s | 2.583 s |
| CPU/ANE | 3.951 s | 0.887 s | 4.837 s |

**GPU remains the default** because its first-request total is lower in this comparison. CPU/ANE is selectable; its processing phase is faster but its startup costs more. This phase includes image processing and PNG export, not just a neural-network kernel. This initial measurement predates the resident-model cache described below. Larger multi-tile images may change the balance.

Both configurations succeeded. Their 8-bit RGB outputs differ slightly: MAE 0.1103 levels, maximum difference 5 levels, PSNR 57.65 dB, cosine 0.9999980. CPU/ANE excludes GPU, but unsupported operators can fall back to CPU; actual ANE occupancy was not measured.

Evidence: `outputs/dev-verify-20260928/upscale-compute/comparison.json`. The native Swift harness is `tests/integration/UpscaleComputeBenchmark.swift`; arguments are model path, input image, output PNG, and `gpu` or `ane`.

## Resident cache and x2plus comparison

The final implementation retains one Core ML model in an actor, serializes inference, and invalidates it when model file metadata or compute choice changes. Explicit/automatic preload includes one dummy prediction. Starting an image without preload skips that dummy prediction; the real image warms the model. Model release is available from the dedicated page and global memory controls.

Same 512×512 fox input; x2 outputs 1024×1024 and x4 outputs 2048×2048. Each model/backend has a separate fresh-process measurement and another process that preloads once then renders three images. OS caches were not cleared. These figures include decoding, tiling and PNG export and do not establish hardware ANE residency.

| Model | Compute | First request total, without preload | Preload + warmup | Resident processing median (3 images) |
| --- | --- | ---: | ---: | ---: |
| x2plus | CPU/GPU | 1.205 s | 1.127 s | 0.490 s |
| x2plus | CPU/ANE | 3.019 s | 2.930 s | 0.223 s |
| x4plus | CPU/GPU | 2.717 s | 2.441 s | 1.978 s |
| x4plus | CPU/ANE | 4.959 s | 4.629 s | 0.876 s |

Preloaded CPU/ANE takes about 45% of CPU/GPU's processing time for both models. x2plus takes about one quarter of x4plus's processing time in this workload, with one quarter as many output pixels; it is not an equal-resolution comparison. Preload does not remove startup cost, but hides it before the user requests an image and amortizes it over subsequent requests. GPU remains the default for lower startup overhead; ANE is user-selectable with automatic preload enabled for a configured model.

Raw reports and PNGs: `outputs/dev-verify-20260928/upscale-resident/`. Reproduce with the native `turbocider-upscale-benchmark MODEL INPUT OUTPUT gpu|ane [resident]` harness. Without the optional argument it measures first load+render; with `resident` it measures preload and three cache-reusing images.

## Verification

- Native contract suite: 110 tests, 107 passed / 3 conditional skips.
- Studio configuration and worker query regressions passed, including the new Z-Image default.
- Image/history transaction and fake-worker lifecycle regressions passed.
- Synthetic 2×/4× upscaler tests passed for RGB ordering, multi-tile seams, reflection, one-pixel transparent inputs, cancellation, invalid tensors, missing model rejection and persistence.
- Real x4plus: 512²→2048² and 768×384→3072×1536 exports passed; the integrated job store passed history restoration and cancellation without publishing a cancelled output.
- Real x2plus: 768×384→1536×768, explicit preload, cache reuse, GPU→ANE invalidation, release, history restore and cancellation passed.
- UI inspection confirmed the dedicated sidebar entry, Z-Image default 8 steps and GPU default. Further interaction was temporarily blocked by the computer-use transport (`native pipe closed before response`); final evidence is recorded below when available.

These checks do not qualify the existing experimental streaming catalog or all optional models. Production streaming records remain empty. Models absent locally receive contract/static coverage only. Main integration must retain those existing experimental labels and capability gates.

## Final delivery and main readiness

The final native Swift App and test targets compile successfully. `dist/TurboCider.app` was repackaged with the rebuilt native engine and passed `codesign --verify --deep --strict`. The upscaling path links Core ML and contains no Python process invocation or runtime dependency.

A real controller-level flow using the existing local Z-Image installation passed: default 8 steps, 512×512 GPU generation, then preloaded CPU/ANE x2plus super-resolution to 1024×1024. The receipt reports `actual_denoise_steps: 8`, `model_cache_hit: true`, and validated image artifacts. Both original and upscaled records survive store reinitialization. Native generation request wall time was 19.78 s. Evidence: `outputs/dev-verify-20260928/generation-upscale-final.log`; final model/cache tests: `upscale-cache-tests-final.log`.

Visual inspection of the x2 output found correct orientation and colors. GPU/ANE output comparison has mean absolute 8-bit differences 0.079 (x2) and 0.110 (x4), with maxima 2 and 5 respectively; this confirms close numerical agreement for this image, not a general quality certification.

Computer use launched the packaged test copy and confirmed the sidebar entry, GPU default and the 8-step control/help text. Subsequent controls could not be exercised because the computer-use native transport repeatedly returned `Sky Computer Use native pipe closed before response`, including after session reset. A complete dedicated-page click-through remains a manual QA item; backend and controller tests above are separate evidence, not a substitute claim that all UI actions passed.

`dev` is integrated into `dev-verify`; `main` is an ancestor and the merge preflight is conflict-free. No main merge or remote push was performed. Build and covered native/controller regressions support integration, with a final manual UI smoke recommended before main. This change does not qualify production streaming, optional absent models, or a public distribution release; existing experimental labels and gates remain in place.

## Right sidebar refinement

Creation now keeps its prompt and action controls on the left. Upscaling model, device, auto-after-generation and preload/release settings share one `UpscaleSettingsView` in the right sidebar, also used by the standalone upscaling page. Qwen prompt-enhancement configuration moved to the creation sidebar. Text-only generation hides irrelevant image-import controls; the toolbar toggles the standalone settings panel too.

The revised App compiles. Interactive verification is pending: AppleScript fallback was authorized and macOS reports accessibility enabled, but the console subsequently became locked (`IOConsoleLocked` and `CGSSessionScreenIsLocked` both true), producing a black capture and inaccessible windows. The user has been asked to unlock the desktop. These checks are not recorded as passed.
