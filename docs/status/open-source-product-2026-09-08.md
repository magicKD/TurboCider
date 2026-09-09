# Open-source product pass — 2026-09-08

Branch: `dev-verify`. Existing historical documents, models and user media are
retained. This record distinguishes product integration from per-model runtime
qualification. The final package and UI checks passed. Actual video generation
on this Mac remains unverified because no compatible local video weights were
found; no weights were downloaded for testing.

| Requirement | Implemented behavior | Evidence / remaining check |
|---|---|---|
| Bilingual README and reference-inspired identity | Chinese/English introduction, reproducible vector logo/App icon, measured CPU/GPU/ANE explanation and named-workload performance | `README.md` (English), `README.zh-CN.md` (中文), `assets/branding/`, `tools/branding/generate.swift`, `docs/PERFORMANCE.md`; actual App logo and model-page layout inspected |
| Newcomer and contributor guidance | Build, App/CLI, model library, LoRA, flexible ANE export/compile, API and cache guides; MIT top-level license prepared for review, separate third-party notices | `docs/GETTING_STARTED.md`, `docs/MODEL_LIBRARY.md`, `docs/LOCAL_API.md`, `docs/CACHES.md`, `CONTRIBUTING.md`, `LICENSE`; package includes project and dependency licenses |
| Scalable model page | Search by model/name/path/capability; filters; browsing separate from active draft; explicit executable operations and current session | `ModelLibraryTests`, actual FLUX/Z-Image/LTX page inspection |
| Unified model storage and sharing | Shared App/CLI root, external registrations, configuration import, directory links, verified shared Qwen components and content-addressed blobs | `LibraryStoreTests`, `LibraryToolTests`; App shared-component preview; existing user paths remain unchanged |
| ModelScope/Hugging Face downloads | ModelScope default, metadata preview, exact file selections, byte progress, cancellation, immutable source revisions, size/hash verification and atomic publication | Tiny loopback provider tests; actual Z-Image and video-model metadata plans; no model weights downloaded |
| Multi-repository installation | LTX combines four exact weight files with one supplementary tokenizer, without a second encoder download | App and CLI show5 files/39.36GB; pinned source records in `outputs/open-source-installation-20260908/`; tiny transfer/reuse/conflict tests |
| Availability and repair | Required-file/shard/header inspection, incompatible FLUX variant detection, explicit preparation requirements, path re-selection and registration removal retaining files | `InstallationInspectionTests`; real Z/FLUX CLI checks; actual FLUX App check4 files/15.96GB. Z App check hit an OS `open` wait/timeout; no false success reported |
| CLI model selection | `generate`/`batch` accept `@model-id`, resolved from the shared library with request-model matching | Actual `@flux2-klein-4b` GPU generation produced512×512 PNG, seed733; `outputs/open-source-alias-20260908/` |
| ANE preparation and reuse | Active model/LoRA-bound export, explicit compile step, source/compiled manifests, existing-cache discovery, exact tokenizer capacity routing | `ZImageShapeTests`, native flexible-shape tests and earlier actual App/CLI runs; `profiles/z-image-flex-512.example.json`, `examples/coreml/` provide editable requests |
| Image/video creation and restrictions | Image generation/editing and video controls follow executor operations; LTX offers text-only, video-only creation | Actual Z-Image GPU+ANE image generation and App step display verified. Final packaged LTX page offers text-to-video, disables image input and displays “当前执行器输出无音轨视频。” without an audio toggle. Unsupported audio requests fail draft validation. No usable local video weights were found; actual App video generation remains pending an existing model path |
| Progress and device/memory presentation | Sampling step count/speed/ETA; optional timing/cache metrics; current session/model, MLX snapshots and cumulative Core ML calls; ANE occupancy explicitly unknown | Actual App showed2/9 steps with ETA; completed runs26.065/16.624s, calls288→576, prompt miss→hit and no output copies. `RunInsightsTests` and Studio behavior tests |
| Local API | App start/status/stop, standalone CLI serve/rpc, child ownership, session exclusion and shutdown on parent loss | `LocalAPITests`, service/lifecycle tests; actual App start→external RPC→512PNG→stop; `outputs/open-source-api-app-20260908/` |
| Seed-independent prompt cache | Same compatible session reuses text conditioning across seed changes; LTX disk conditioning includes model/tokenizer/prompt identity | Actual App random-seed runs and native API tests show cache hits; session release clears memory caches |
| Tensor cleanup and retention | LTX conditioning plus explicitly registered, nonrecursive FLUX/Z diagnostic dumps; manual deletion, off/7/30/90 days, idle hourly App sweep | `TensorCacheTests` and synthetic CLI enrollment/prune/forget tests passed. Final App registered a temporary directory, removed its one 77-byte diagnostic tensor, preserved the other file and removed the registration. No real research output was enrolled or deleted; retention remains disabled |
| Build and delivery | Native baseline tests, complete Swift build, scoped subsequent rebuild, packaged App/CLI and signatures | Full Swift build and baseline `make test` passed; final audio-gate Studio/model tests, package and strict signature checks passed. App and CLI contain the project license. Optional NumPy/real-weight fixture skips are not counted as runtime passes |

## Remaining verification

1. Use an existing local video model, if supplied, for an actual App generation.
   No model download is authorized for this smoke test. The current machine's
   model-folder inspection and targeted file index search found no compatible
   LTX/H3/FastMetal artifact. Prior cross-machine video qualification records
   remain in `docs/USAGE.md`; they are not a substitute for this local App check.
2. Recheck the optional Z-Image installation inspection after OS file access is
   available. Its CLI check and actual inference already passed; the helper's
   system-open wait is recorded separately from model validity.

The user's original Z-Image draft was restored through App configuration import
after video-page inspection, including empty prompt, random seed,512×512,
GPU+ANE and the unchecked LoRA at1.0. A read-only comparison confirmed exact
draft equality after the final packaged UI check. User history and original
files were preserved. The App was left on Z-Image creation with no loaded model
session; no local API or test inference was left running.

## Model-card and README follow-up

The model-row button now defines its hit area after padding and full-width
layout, so the entire card is interactive. The rebuilt packaged App was
verified by coordinate clicks in the right-hand whitespace (FLUX4B), left
bottom padding (FLUX9B), and icon (back to FLUX4B); each changed the detail pane.
The active creation model and generation settings were not changed.

README languages now have separate files: `README.md` for English and
`README.zh-CN.md` for Chinese, with reciprocal language links and complete
feature, performance, getting-started and model-capability sections in each.
Both files' local links passed validation. App compilation, packaging and
strict signature verification passed for this UI change.
