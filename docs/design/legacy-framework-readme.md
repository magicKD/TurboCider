> 历史兼容框架说明；其中 Production/Validated 标签不代表本次原生重构已经验收。

# TurboCider

TurboCider is a unified local multimodal-generation framework for Apple
Silicon. It exposes one request model, execution-policy layer, job system,
CLI, HTTP API, Swift SDK, and SwiftUI app while retaining the optimized native
engines for MiniMax H3, LTX-2.5, FLUX.2, and FastMetal-1.3B-QAD.

The engine boundary is deliberately thin: adapters translate a stable
TurboCider request into each engine's native flags and environment. The H3
Metal/MPSGraph, LTX MLX/Core ML, FLUX.2 MLX/Core ML, and FastMetal MLX/Core ML
graphs continue to own their memory and compute lifetimes, preserving
direct-engine speed and output behavior.

## Included capabilities

| Model pack | Generation | Pure GPU | GPU+ANE | Model-specific features |
|---|---|---:|---:|---|
| `minimax-h3-turbo` | Video | Production | Explicit experimental route | Text, first/last frame, reference image/video/audio, embedded audio, preview profiles, native engine passthrough |
| `ltx-2.5-distilled` | Text-to-video + synchronized audio, single first-frame image-to-video | Production | Validated 704×480 route | Fixed distilled 8+3 schedule, parallel A/V finalization, dense and explicit preview plans, per-token first-frame conditioning for both GPU and GPU+ANE |
| `flux2-klein-4b` | Text-to-image, img2img, multi-image edit | Production | Validated standard/img2img route | Persistent model/ANE sessions, dynamic prompt length, image strength, ordered references, precision and block controls |
| `fastmetal-1.3b-qad` | Video | Production | Validated 832×480×81 route | Fixed 3-step DMD schedule, INT8 MLX DiT, 16 fps video without audio |

Execution placement (`auto`, `gpu`, `gpu_ane`), generation profile (`quality`,
`balanced`, `preview`), and approximation permission (`exact`, `validated`,
`experimental`) are independent. Explicit `gpu_ane` requests fail closed when
the matching artifacts or shape are unavailable. `auto` only selects a
production plan whose requirements match and otherwise falls back to GPU.

## Quick start

Run from this directory:

```sh
PYTHONPATH=src python3 -m turbocider.cli doctor
PYTHONPATH=src python3 -m turbocider.cli models
PYTHONPATH=src python3 -m turbocider.cli generate \
  --model minimax-h3-turbo \
  --prompt "A cinematic red fox walking through fresh snow" \
  --width 512 --height 512 --frames 22 --steps 4 \
  --execution gpu --profile quality --approximation exact \
  --output outputs/h3.mp4
```

An editable install provides the `turbocider` and `turbociderd` commands:

```sh
python3 -m pip install -e .
turbocider doctor
```

Native engines are resolved from `${TURBOCIDER_ENGINES}` and default to
`engines/` inside this checkout. Point `TURBOCIDER_ENGINES_DIR` at an existing
installation, or materialize the expected layout from engine source checkouts:

```sh
turbocider bootstrap --source /path/to/native-engines
turbocider bootstrap --source /path/to/native-engines --copy
```

The bootstrap links (or copies) `h3`, `ltx-mac`, `flux2`, and `fastmetal`
directories into `engines/` without recording any machine-specific monorepo
paths in the repository. Set `TURBOCIDER_MODELS_DIR`,
`TURBOCIDER_STATE_DIR`, and `TURBOCIDER_OUTPUT_DIR` to relocate downloaded
models, receipts, and generated media.

Common real-generation examples:

```sh
# LTX pure GPU
PYTHONPATH=src python3 -m turbocider.cli generate \
  --model ltx-2.5-distilled --prompt "A runner beside the ocean" \
  --width 704 --height 480 --frames 97 --steps 11 \
  --execution gpu --approximation exact --output outputs/ltx-gpu.mp4

# LTX validated GPU+ANE
PYTHONPATH=src python3 -m turbocider.cli generate \
  --model ltx-2.5-distilled --prompt "A runner beside the ocean" \
  --width 704 --height 480 --frames 97 --steps 11 \
  --execution gpu_ane --approximation validated --output outputs/ltx-hybrid.mp4

# FLUX.2 pure GPU with a reusable worker
PYTHONPATH=src python3 -m turbocider.cli generate \
  --model flux2-klein-4b --task image --prompt "A cider press in an orchard" \
  --width 512 --height 512 --frames 1 --steps 4 \
  --execution gpu --approximation exact --persistent \
  --output outputs/flux2-gpu.png

# FLUX.2 validated GPU+ANE with persistent weights and Core ML sessions
PYTHONPATH=src python3 -m turbocider.cli generate \
  --model flux2-klein-4b --task image --prompt "A cider press in an orchard" \
  --width 512 --height 512 --frames 1 --steps 4 \
  --execution gpu_ane --approximation validated --persistent \
  --output outputs/flux2-hybrid.png

# FLUX.2 single-image img2img; standard token geometry remains GPU+ANE capable
PYTHONPATH=src python3 -m turbocider.cli generate \
  --model flux2-klein-4b --task image --mode image_to_image \
  --prompt "Restyle this as a watercolor poster" \
  --init-image input.png --image-strength 0.55 \
  --width 512 --height 512 --steps 4 \
  --execution gpu_ane --approximation validated --persistent \
  --output outputs/flux2-img2img.png

# FLUX.2 ordered multi-reference editing uses pure GPU/MLX because the
# reference-token sequence is variable length.
PYTHONPATH=src python3 -m turbocider.cli generate \
  --model flux2-klein-4b --task image --mode image_edit \
  --prompt "Combine the subject from the first image with the style of the second" \
  --ref-image subject.png --ref-image style.png \
  --width 512 --height 512 --steps 4 \
  --execution gpu --approximation exact --persistent \
  --output outputs/flux2-edit.png

# FastMetal pure GPU
PYTHONPATH=src python3 -m turbocider.cli generate \
  --model fastmetal-1.3b-qad --prompt "A bird flying above a forest valley" \
  --width 832 --height 480 --frames 81 --fps 16 --steps 3 \
  --execution gpu --approximation exact \
  --output outputs/fastmetal-gpu.mp4

# FastMetal validated GPU+ANE on its fixed M4 Max profile and shape
PYTHONPATH=src python3 -m turbocider.cli generate \
  --model fastmetal-1.3b-qad --prompt "A bird flying above a forest valley" \
  --width 832 --height 480 --frames 81 --fps 16 --steps 3 \
  --execution gpu_ane --approximation validated \
  --output outputs/fastmetal-hybrid.mp4
```

## Model preparation and Core ML cache

Each bundled model pack includes immutable upstream repository/revision data,
an estimated disk budget, the exact runtime-compatible files to download, and
the engine-owned conversion tools. Inspect or plan work before writing files:

```sh
PYTHONPATH=src python3 -m turbocider.cli prepare-model flux2-klein-4b
PYTHONPATH=src python3 -m turbocider.cli prepare-model flux2-klein-4b \
  --download --ane --cache --dry-run
```

Prepare the four current model families:

```sh
# FLUX.2: download, export fixed-bucket INT8 ANE slices, then create stable,
# content-addressed .mlmodelc paths for Core ML specialization reuse.
turbocider prepare-model flux2-klein-4b --download --ane --cache --workers 4

# The same download step can use ModelScope when that hub hosts the pinned
# revision. TurboCider keeps engine conversion identical for both hubs.
turbocider prepare-model flux2-klein-4b --download --hub modelscope

# LTX: download only the five Comfy/convrot files consumed by ltx-mac, then
# export and compile the 1001/4004-row MLP and 1024-row text K/V artifacts.
turbocider prepare-model ltx-2.5-distilled --download --ane --cache

# FastMetal: download the pinned MLX/runtime files, then compile all 30 INT8
# FFN prefixes for the fixed 832×480×81 route (32760 rows, 4096 ANE channels).
turbocider prepare-model fastmetal-1.3b-qad --download --ane

# H3: its Core ML graph is fixed-row. The exact target row count is mandatory;
# TurboCider never guesses this shape.
turbocider prepare-model minimax-h3-turbo --ane --cache \
  --source-model engines/h3/models/MiniMax-H3-LightX2V-Turbo \
  --h3-rows 15405 --blocks 0-49
```

H3's full base download plus merged Turbo checkpoint requires substantially
more than 200 GiB of temporary/new storage. `prepare-model` checks available
space before downloading or converting and fails before writing when the
configured reserve cannot be retained. Existing checkpoints can be supplied
with `--source-model` and `--lora`. `--model-dir`, `--ane-output`, and
`--cache-dir` support external volumes. Set `TURBOCIDER_MODELS_DIR` to make a
custom managed-model root the default.

Successful non-dry runs write an auditable receipt under
`$TURBOCIDER_STATE_DIR/model-management/`. Receipts include pinned revisions,
commands, artifact sizes, timings, and the privacy-safe device profile. Model
weights and generated Core ML artifacts remain ignored by Git.

List downloaded models and their ANE/cache artifacts without starting a job:

```sh
turbocider assets
turbocider assets flux2-klein-4b
```

`--first-frame`, `--last-frame`, `--ref-image`, `--ref-video`, and
`--ref-silent-video`, `--ref-video-audio`, and `--ref-audio` expose H3's
multimodal inputs. `--engine-arg` and `--engine-env NAME=VALUE` retain access
to native engine features that have not yet become typed common fields,
including the complete H3 Super/refiner flag surface. More structured options
can be supplied inline or from a file:

```sh
turbocider generate --model flux2-klein-4b --task image \
  --prompt "An orchard at sunrise" --persistent \
  --engine-options '{"flux2":{"dynamic_text_length":false,"ane_blocks":[0,1]}}'
turbocider generate --model flux2-klein-4b --task image \
  --prompt "An orchard at sunrise" --persistent \
  --engine-options @engine-options.json
```

The JSON API, Python request model, Swift SDK, and App all preserve recursive
namespaced `engine_options.h3`, `engine_options.ltx`,
`engine_options.flux2`, and `engine_options.fastmetal` values. Typed common
fields remain the preferred stable interface; adapter-specific options are the
escape hatch for engine experiments and advanced deployment controls.

### LTX-2.5 image-input status

The LTX native adapter supports single first-frame image-to-video (`image_to_video`)
on both pure GPU and GPU+ANE. Input images are CRF-preprocessed, encoded with the
bundled VAE encoder helper, and carried through Stage-1 and Stage-2 as clean
first-frame prefixes with split per-token timesteps. `strength` controls how far
the first frame may move during denoising.

```sh
turbocider generate --model ltx-2.5-distilled \
  --task video --mode image_to_video \
  --first-frame input.png --image-strength 1.0 \
  --width 704 --height 480 --frames 97 --fps 24 \
  --execution gpu_ane --approximation validated \
  --output outputs/ltx-i2v.mp4
```

The downloaded native pack is not the complete upstream LTX API surface.
Multi-keyframe interpolation, retake, extend, audio-to-video, and IC-LoRA are
not currently exposed through the native TurboCider adapter.

## FLUX.2 runtime bootstrap

The reproducible bootstrap creates the Python environment, installs
`mflux==0.19.1`, downloads the pinned official model revision, builds the
native bridge, and exports the 20-block INT8 Core ML package:

```sh
python3 scripts/bootstrap_flux2.py \
  --source /path/to/native-engines \
  --target engines
```

The `--source` tree must contain `gpu_ane/flux2-engine` and
`gpu_ane/mac_local_ai`. Use `--skip-model` or `--skip-ane` to reuse existing
artifacts. Model weights, runtime environments, generated media, and Core ML
packages are intentionally excluded from source control.

## Local HTTP API

Start the service on loopback:

```sh
PYTHONPATH=src python3 -m turbocider.cli serve \
  --host 127.0.0.1 --port 11435
```

Submit and monitor a job:

```sh
curl -sS http://127.0.0.1:11435/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"flux2-klein-4b",
    "task":"image",
    "mode":"image_to_image",
    "prompt":"A glass of cider on a wooden table",
    "inputs":[{"type":"image","role":"init_image","path":"/Users/me/Pictures/cider.png","strength":0.55}],
    "output":{"type":"image","width":512,"height":512,"frames":1},
    "sampling":{"seed":42,"steps":4},
    "policy":{"execution":"gpu_ane","approximation":"validated","persistent":true}
  }'

curl -N http://127.0.0.1:11435/v1/jobs/JOB_ID/events
curl -X DELETE http://127.0.0.1:11435/v1/jobs/JOB_ID
```

Other endpoints are `GET /health`, `GET /v1/models`, `GET /v1/system`,
`GET /v1/jobs`, `GET /v1/jobs/{id}`, and `POST /v1/plans`.

The CLI can use that same long-running service. Detached generation returns as
soon as the daemon has durably accepted the job; it does not create a temporary
in-process worker that disappears with the CLI:

```sh
turbocider generate --detach --model flux2-klein-4b --task image \
  --prompt "A cider press" --persistent
turbocider jobs
turbocider jobs --id JOB_ID
turbocider jobs --id JOB_ID --cancel
```

Attached CLI runs print state, phase, progress, and ETA to stderr. Job records
are persisted under `TURBOCIDER_STATE_DIR`: completed jobs remain queryable
after daemon restart, while jobs interrupted by a stopped daemon are restored
as failed with phase `interrupted` instead of being left permanently running.

API output paths are restricted to the configured TurboCider output directory
unless `--allow-external-output` is explicitly set. Input files must be below
an `--allow-input-root`; loopback service mode defaults to the user's home
directory. Binding to a non-loopback address requires a bearer token:

```sh
TURBOCIDER_API_TOKEN='replace-me' PYTHONPATH=src \
  python3 -m turbocider.cli serve --host 0.0.0.0 --port 11435 \
  --allow-input-root /path/to/media
```

Clients then send `Authorization: Bearer replace-me`. Public model and system
responses omit model paths and other machine-local configuration.

## Swift SDK and macOS app

Build and test the Swift products:

```sh
swift build
.build/debug/turbocider-swift-selftest
.build/debug/TurboCiderApp --smoke
.build/debug/turbocider-swift --self-start
```

`TurboCiderKit` provides `TurboCiderClient`, request/response types, and
`TurboCiderDaemon`. The daemon helper starts the local service when needed and
does not stop a service it does not own. A minimal SDK request is:

```swift
import TurboCiderKit

let client = TurboCiderClient()
let request = TCGenerationRequest(
    model: "flux2-klein-4b",
    task: "image",
    prompt: "A glass of cider on a wooden table",
    mode: "image_to_image",
    inputs: [
        TCInputAsset(
            type: "image",
            role: "init_image",
            path: "/Users/me/Pictures/cider.png",
            strength: 0.55
        )
    ],
    output: TCOutputSpec(type: "image", width: 512, height: 512, frames: 1),
    policy: TCPolicySpec(execution: .gpuANE, persistent: true),
    engineOptions: [
        "flux2": [
            "dynamic_text_length": true,
            "reuse_ane_outputs": true,
        ]
    ]
)
let job = try await client.submit(request)
for try await update in client.events(id: job.id) {
    print(update.phase, update.progress, update.estimatedRemainingSeconds as Any)
}
```

The SDK also exposes `models()`, `plans(for:)`, `jobs()`, `job(id:)`,
`cancel(id:)`, and the SSE-backed `events(id:)` stream. Set
`TURBOCIDER_API_URL` to point both the client and daemon helper at another
loopback port.

Create a self-contained control-plane app bundle:

```sh
sh scripts/build_app_bundle.sh
open dist/TurboCider.app
```

The app starts/stops its owned daemon, displays model availability, live phase,
progress, elapsed time, and ETA, offers model-declared generation modes,
accepts supported reference media with thumbnails and img2img strength, previews
generated images/video, and opens or reveals outputs. It initializes generation
settings from each model pack's recommended shape/FPS/step values, disables
unsupported audio, locks required native audio, filters task/input/profile/
execution controls from model capabilities, stages selected input files under
`~/Library/Application Support/TurboCider/inputs`, exposes advanced namespaced engine
options as JSON, and uses SSE with polling fallback. Bundled state, logs, and
outputs are written to
`~/Library/Application Support/TurboCider/`, never into the signed app bundle.
The bundle contains the common control plane but references the native engines
and model data through its generated `workspace.path`; deployments in another
layout should set `TURBOCIDER_WORKSPACE`.

## Configuration

Core paths:

- `TURBOCIDER_WORKSPACE`: directory containing `h3.c`, `ltx-mac`, and
  `gpu_ane`.
- `TURBOCIDER_MODEL_PACKS`: additional model-pack directories separated by
  `:`.
- `TURBOCIDER_STATE_DIR` and `TURBOCIDER_OUTPUT_DIR`: durable job metadata and
  generated output locations.
- `TURBOCIDER_MODELS_DIR`: managed downloaded weights and Core ML artifacts;
  defaults to `TurboCider/models` for source checkouts.
- `TURBOCIDER_API_TOKEN`: default service bearer token.
- `TURBOCIDER_API_URL`: Swift SDK/App and detached-CLI service URL; defaults
  to `http://127.0.0.1:11435`.
- `TURBOCIDER_PYTHON`: Python executable used when the Swift App/SDK helper
  starts the daemon; useful when external adapter entry points live in a venv.

Engine overrides include `LTX_TRANSFORMER`, `LTX_UPSAMPLER`, `LTX_VIDEO_VAE`,
`LTX_TEXT_ENCODER`, `LTX_AUDIO_VAE`, `LTX_CONDITIONING_DIR`, `COMFYUI_ROOT`,
`COMFYUI_PYTHON`, `LTX_MLX_PYTHON`, `MFLUX_ROOT`, `FLUX2_MODEL_PATH`,
`MFLUX_PY`, colon-separated `FLUX2_ANE_MANIFESTS`,
`FLUX2_ANE_BRIDGE_DIR`, `FASTMETAL_PYTHON`,
`FASTMETAL_MODEL_PATH`, `FASTMETAL_ANE_MANIFEST`, and
`FASTMETAL_ANE_BRIDGE_DIR`.

## Verification and design notes

```sh
python3 scripts/verify.py
python3 scripts/verify.py --real --bundle
```

The first command runs Python control-plane tests, FLUX and FastMetal engine
tests when their runtimes are installed, the Swift build/SDK self-test, and an
App smoke that submits a fake native-engine job through Swift SDK, HTTP,
persistence, subprocess execution, and SSE and requires intermediate progress
plus ETA before success.
`--real` also probes all locally installed engines, weights, and ANE artifacts;
`--bundle` builds the signed app and starts its embedded daemon from outside
the source tree.
See `docs/ARCHITECTURE.md`, `docs/CONFORMANCE.md`, and `docs/PERFORMANCE.md` for
the integration contract and measured parity results. Developers adding a new
checkpoint or runtime should follow `docs/EXTENDING.md`; external runtime
packages can register an `EngineAdapter` through the `turbocider.adapters`
Python entry-point group without modifying the App, SDK, or control plane.

Re-run direct/integrated performance and decoded-output comparisons from JSON
specs:

```sh
turbocider benchmark --spec benchmarks/flux2-klein-4b-gpu.json \
  --output outputs/benchmark-flux2-klein-4b-gpu.json
turbocider benchmark --spec benchmarks/flux2-klein-4b-gpu-ane-warm.json \
  --output outputs/benchmark-flux2-klein-4b-gpu-ane-warm.json
turbocider benchmark --spec benchmarks/minimax-h3-turbo-gpu.json \
  --output outputs/benchmark-minimax-h3-turbo-gpu.json
turbocider benchmark --spec benchmarks/ltx-2.5-native-gpu-vs-gpu-ane.json \
  --output outputs/benchmark-ltx-2.5-native-gpu-vs-gpu-ane.json
turbocider benchmark --spec benchmarks/ltx-2.5-gpu-e2e.json \
  --output outputs/benchmark-ltx-2.5-gpu-e2e.json
turbocider benchmark --spec benchmarks/ltx-2.5-gpu-ane-e2e.json \
  --output outputs/benchmark-ltx-2.5-gpu-ane-e2e.json
turbocider benchmark --spec benchmarks/fastmetal-1.3b-qad-gpu.json \
  --output outputs/benchmark-fastmetal-1.3b-qad-gpu.json
turbocider benchmark --spec benchmarks/fastmetal-1.3b-qad-gpu-ane.json \
  --output outputs/benchmark-fastmetal-1.3b-qad-gpu-ane.json
```

The report separates process/job/engine timings and compares decoded visual
and audio samples. LTX additionally compares directory-valued native BF16
artifacts and worker phase timings, so cold setup is not mistaken for the
decoded-pixel hot path and container metadata differences are not mistaken for
quality differences. Optional `artifact_pairs` additionally verify latent or
intermediate files by size, SHA-256, and an engine-owned numerical comparator.
