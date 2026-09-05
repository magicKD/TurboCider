# FLUX.2 Mac Engine

`flux2-engine` is a persistent FLUX.2 inference engine for Apple Silicon. It
uses the local mflux implementation as its MLX model-kernel layer and adds a
stable Python API, an easy CLI, runtime policy, fixed-shape artifact discovery,
and a zero-copy MLX GPU + Core ML ANE backend.

The production hybrid split is deliberately coarse:

- MLX/Metal computes QKV, fused attention, residuals, text encoding, and VAE.
- Core ML restricted to CPU+Neural Engine computes the fused MLP branch in all
  selected single-transformer blocks.
- Both branches consume the same normalized hidden state and execute in
  parallel. Core ML writes its partial directly into reusable MLX FP16 buffers.

This split is already faster end to end and keeps image quality close. ANE
attention variants exist as gated research plans, not unsafe default switches.

## Quick start

Use the existing mflux virtual environment; no model download is required.

```bash
cd /Users/kd/Documents/project/mac_image_generation/flux2-engine
MFLUX_PY=/Users/kd/Documents/project/mac_image_generation/mflux/.venv/bin/python

PYTHONPATH=src "$MFLUX_PY" -m flux2_engine.cli doctor
"$MFLUX_PY" scripts/build_native.py

# Portable pure MLX.
PYTHONPATH=src "$MFLUX_PY" -m flux2_engine.cli generate \
  "A tiny red panda programmer using a silver MacBook" \
  --mode mlx --profile quality \
  --output ../outputs/flux2-engine/mlx.png

# Explicit persistent-throughput backend using the 512 fast package.
PYTHONPATH=src "$MFLUX_PY" -m flux2_engine.cli generate \
  "A tiny red panda programmer using a silver MacBook" \
  --mode mlx-ane \
  --ane-manifest ../../mac_local_ai/models/coreml/flux2_stack_runtime_m1088/manifest.json \
  --output ../outputs/flux2-engine/mlx-ane.png

# Auto uses MLX for one-shot work. With --persistent and a fitting discovered
# manifest, it selects MLX+ANE and falls back to MLX on unsupported shapes.
PYTHONPATH=src "$MFLUX_PY" -m flux2_engine.cli generate \
  "A cinematic alpine lake" --mode auto --persistent \
  --output ../outputs/flux2-engine/auto.png
```

Every CLI image gets a sibling `.engine.json` with backend choice, wall time,
peak MLX memory, sequence length, Core ML bucket, ANE p50, and session state.

## Python API

The API is designed for a long-running app or service. Keeping one engine alive
is important: model weights, compiled Core ML models, prediction bindings, and
caller-owned output buffers are reused across requests.

```python
from pathlib import Path

from flux2_engine import EngineConfig, ExecutionMode, Flux2Engine, GenerationRequest

project = Path("/Users/kd/Documents/project")
config = EngineConfig(
    model_path=project / "mac_image_generation/models/FLUX.2-klein-4B",
    mflux_root=project / "mac_image_generation/mflux",
    mode=ExecutionMode.MLX_ANE,
    persistent=True,
    ane_manifests=(
        project / "mac_local_ai/models/coreml/flux2_stack_runtime_m1088/manifest.json",
    ),
)

with Flux2Engine(config) as engine:
    result = engine.generate(
        GenerationRequest(
            prompt="A tiny red panda programmer using a silver MacBook",
            output=project / "mac_image_generation/outputs/flux2-engine/api.png",
            width=512,
            height=512,
        )
    )
    print(result.metrics.as_dict())
```

## Profiles and modes

| Option | Behavior |
|---|---|
| `--profile quality` | Four steps; default close-quality mode |
| `--profile balanced` | Three steps; fast for scenes without important small text |
| `--profile preview` | Two steps; preview only |
| `--mode mlx` | Pure mflux/MLX, portable and general |
| `--mode mlx-ane` | Explicit fixed-bucket GPU+ANE backend; fails closed |
| `--mode auto` | One-shot uses MLX; persistent mode uses a fitting ANE artifact with MLX fallback |

The current real 512 smoke produced a pixel-identical image to the research
hybrid reference. In one persistent engine instance, the first generation took
4.03 seconds and the second took 2.80 seconds; second-request Core ML output
binding hit rate was 100%. These numbers are machine-state dependent and the
session's first Core ML model load is reported separately.

At 1024x1024 with the same prompt, seed, BF16 weights, and four steps, pure MLX
took 22.65 seconds and the MLX+ANE generation path took 17.19 seconds: 1.32x
speedup (24.1% less latency) after backend/session initialization. The first
4160-bucket Core ML session load added 8.68 seconds and is amortized only by a
persistent process. MLX peak allocation was 24.17 GB versus 24.68 GB.
The INT8-ANE result kept the same composition and had pixel MAE 4.39/255 and
PSNR 25.14 dB against pure MLX. Back-to-back 1024 runs can throttle or page on
this machine, so the engine clears freed Metal buffers between requests while
keeping weights, compiled models, output backing, and sessions resident.

## ANE artifacts

The runtime accepts existing stack manifests without copying 1.7 GB packages.
To export another bucket through the validated real-weight exporter:

```bash
PYTHONPATH=src "$MFLUX_PY" scripts/export_ane_artifacts.py \
  --checkpoint ../models/FLUX.2-klein-4B \
  --out-dir artifacts/coreml/m1088 \
  --buckets 1088
```

The wrapper intentionally reuses `mac_local_ai`'s exporter and compression
tooling. Runtime inference and the native bridge are self-contained here.

## Attention policy

Inspect placement candidates and activation traffic:

```bash
PYTHONPATH=src "$MFLUX_PY" -m flux2_engine.cli attention-plan \
  --width 1024 --height 1024 --text-tokens 21
```

`attention=auto` and `attention=mlx` use MLX fused SDPA. Full-ANE and GPU/ANE
head-split modes fail closed until a fixed-shape Core ML attention artifact
beats the current overlapping GPU-attention/ANE-MLP block end to end.

The included operator probes currently reject full ANE attention: versus MLX
fused SDPA, the public Core ML path was 20.7x slower at sequence 1088 and 17.8x
slower at sequence 4160; the long-sequence result also failed the FP32 numerical
gate. ANE therefore remains assigned to the independent MLP branch, where its
work overlaps Metal attention.

See [architecture](docs/ARCHITECTURE.md) and the detailed
[attention roadmap](docs/ATTENTION.md).

## Development checks

```bash
PYTHONPATH=src "$MFLUX_PY" -m pytest -q
PYTHONPATH=src "$MFLUX_PY" -m compileall -q src tests scripts
PYTHONPATH=src "$MFLUX_PY" scripts/benchmark_engine.py \
  "A cinematic alpine lake" --mode mlx-ane --width 1024 --height 1024
```
