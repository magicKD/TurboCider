# ltx-mac

`ltx-mac` is a native Apple-Silicon inference engine for the real
LTX-2.5 22B distilled checkpoint. The hot path is C11, Objective-C, Metal,
MPS and MPSGraph; MLX is used for the spatial upsampler and isolated media
decode. Core ML is optional and is used only for measured GPU+ANE splits.

The engine runs the official distilled recipe without reducing the schedule:

```text
Gemma conditioning
  -> Stage 1 DiT (8 ancestral steps)
  -> BF16 latent spatial upsample x2
  -> Stage 2 DiT (3 deterministic steps)
  -> Video VAE / Audio VAE / vocoder / mux
```

The project is intentionally organised around evidence rather than a single
"fast" setting:

- **default GPU** is dense attention and full 1,024-row text context;
- **GPU+ANE dense** is the quality-first heterogeneous mode;
- **fast/approx** additionally enables Stage-2 Sol attention and `text256`
  context pruning. It is useful for research, but is not the default.

See [notes/technical-report-2026-09-04.md](notes/technical-report-2026-09-04.md)
for architecture, quality evidence, rejected approaches and the current
optimisation backlog.

## Status and measured performance

All figures below use the local LTX-2.5 distilled INT8 ConvRot checkpoint,
the original 8+3 schedule, 97 frames and an M4 Max with 64 GB unified memory.
`Decoded-pixel E2E` starts after text conditioning and ends after the native
RGB BF16 pixels are produced; it excludes MP4 encoding.

| Mode | Actual output | Decoded-pixel E2E | Relative to GPU | Quality status |
| --- | ---: | ---: | ---: | --- |
| GPU dense baseline | 704x448, 24 FPS | 59.483 s | 1.000x | Reference |
| GPU+ANE, dense attention/full context | 704x448, 24 FPS | 49.467 s | 1.202x | Accepted |
| GPU+ANE + Stage-2 Sol | 704x448, 24 FPS | 48.837 s | 1.218x | One prompt/seed accepted |
| GPU+ANE + Sol + `text256` + ANE K/V | 704x448, 24 FPS | **46.824 s** mean | **1.270x** | Current fast candidate; broader suite pending |
| GPU dense 720p-class | 1280x704, 24 FPS | 169.609 s | — | Valid output; no ANE path yet |

The 480p-class request is `704x480`, but the two-stage latent grid is aligned
to 64 pixels, so the decoded result is `704x448`. The 720p-class request
`1280x720` similarly becomes `1280x704`.

The 480p fast candidate peaks at about 49.56 GiB without swap. The 720p GPU
run reaches about 59.5 GiB, so it has little headroom on a 64 GB machine.

## Requirements

- Apple Silicon Mac, macOS 15 or newer for the Core ML/ANE path;
- Xcode Command Line Tools with Metal, MPS and Core ML frameworks;
- the LTX-2.5 distilled Transformer, spatial upsampler, video VAE, Gemma text
  encoder and optional combined Audio VAE/vocoder checkpoint;
- a Python environment containing MLX for the media pipeline;
- a Python environment containing `coremltools` only when exporting ANE
  artifacts.

The checkpoint, generated videos, Core ML packages and virtual environments
are ignored by Git. Keep them outside source control.

## Build and self-test

From `ltx-mac/`:

```bash
make -j8
make test
make tools
```

Build the MLX media and ANE-capable generation binary. `MLX_ROOT` is the
root of the installed MLX Python package, not the virtual-environment root.

```bash
export MLX_ROOT="$PWD/.venv-mlx/lib/python3.13/site-packages/mlx"
make -j8 MLX_ROOT="$MLX_ROOT" mlx-ane-pipeline
```

Useful smoke checks:

```bash
./build/ltx --device
./build/ltx --self-test
./build/ltx --info -d "$LTX_TRANSFORMER"
./build/ltx --workload --width 704 --height 480 --frames 97 --fps 24
```

## Model paths and conditioning

Set model paths once. Names are illustrative; the engine works with the
locally available ComfyUI LTX-2.5 distilled files.

```bash
export COMFYUI_ROOT=/path/to/ComfyUI
export COMFY_PY=/path/to/ComfyUI/python
export LTX_TRANSFORMER=/path/to/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors
export LTX_TEXT_ENCODER=/path/to/gemma4-with-ltx-projection.safetensors
export LTX_UPSAMPLER=/path/to/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
export LTX_VIDEO_VAE=/path/to/ltx-2.5-video-vae-conv-bf16.safetensors
export LTX_AUDIO_VAE=/path/to/ltx-2.5-audio-vae-bf16.safetensors
```

Text encoding is deliberately isolated from DiT generation so the large text
encoder does not compete for unified memory. Create conditioning once per
prompt:

```bash
mkdir -p outputs/conditioning/demo/raw outputs/conditioning/demo/connected

"$COMFY_PY" tools/comfy_bridge.py encode \
  "$LTX_TEXT_ENCODER" "$LTX_TRANSFORMER" \
  outputs/conditioning/demo/raw \
  --prompt "A cinematic red fox running through a snowy forest" --raw-only

./build/run_connector "$LTX_TRANSFORMER" \
  outputs/conditioning/demo/raw outputs/conditioning/demo/connected 1
```

The connector places valid prompt tokens first and then fills to at least
1,024 rows with learned registers. The full context is the default quality
path.

## Run a quality-first GPU generation

This is the recommended starting point. It keeps dense attention and the
full context, uses the native Transformer, MLX upsampler and an isolated MLX
video VAE process.

```bash
mkdir -p outputs/videos/demo_480p

LTX_MEDIA_BACKEND=mlx \
LTX_MLX_MEDIA_WARMUP=1 \
./build/bench_block_mlx "$LTX_TRANSFORMER" all --generate \
  "$LTX_UPSAMPLER" "$LTX_VIDEO_VAE" \
  outputs/conditioning/demo/connected outputs/videos/demo_480p 42
```

The command writes native `video_pixels.bf16`, `audio_latent.bf16` and
`generation.json`. Encode the RGB pixels, then decode/mux the audio:

```bash
"$COMFY_PY" tools/comfy_bridge.py encode-native-video \
  outputs/videos/demo_480p/video_pixels.bf16 outputs/demo_480p.mp4 \
  --frames 97 --height 448 --width 704 --fps 24

.venv-mlx/bin/python tools/finalize_audio.py \
  outputs/videos/demo_480p --video-root outputs --sync-to-video \
  --target-lufs=-18
```

The final muxed file is `outputs/demo_480p_with_audio.mp4`.

### Higher resolution

The native geometry derives all token grids from the requested size. The
following is a real latent-grid 720p-class run, not post-decode resizing:

```bash
LTX_OUTPUT_WIDTH=1280 LTX_OUTPUT_HEIGHT=720 \
LTX_MEDIA_BACKEND=mlx LTX_MLX_MEDIA_WARMUP=1 \
./build/bench_block_mlx "$LTX_TRANSFORMER" all --generate \
  "$LTX_UPSAMPLER" "$LTX_VIDEO_VAE" \
  outputs/conditioning/demo/connected outputs/videos/demo_720p 42
```

It produces `1280x704`. The current high-resolution path is GPU-only: ANE
artifacts and Sol routing are shape-specific and have not yet been extended to
the 2,860/11,440 Stage-1/Stage-2 token shapes.

## Export Core ML artifacts for ANE

The Core ML path is not a monolithic Transformer. It uses per-block,
fixed-shape artifacts because Transformer blocks have different weights and
are separated by GPU attention/residual dependencies:

- Video FFN: a 6,912-channel ANE prefix plus a 9,472-channel GPU suffix;
- Video-text K/V: complete K/V projection per block;
- optional experimental V-to-A and QKV artifacts are exported separately and
  are not part of the recommended command.

Use a Python environment with `numpy` and `coremltools`. In this workspace
the existing environment is normally:

```bash
export COREML_PY="$PWD/../gpu_ane/mac_transformer/.venv/bin/python"
export BLOCKS="$(jot -s, 48 0)"
```

Export all 48 Video FFN artifacts for the two 480p shapes. The exporter folds
ConvRot into the ANE weights, quantizes Core ML weights per output channel to
INT8, compiles the package and retains only `.mlmodelc` plus the GPU shard.

```bash
"$COREML_PY" tools/export_ane_mlp_blocks.py \
  "$LTX_TRANSFORMER" models/coreml/ltx_mlp_a6912_int8 \
  --blocks "$BLOCKS" --ane-intermediate 6912 --variant int8_pc
```

For the full 1,024-row context, export matching Video-text K/V artifacts:

```bash
"$COREML_PY" tools/export_ane_kv_blocks.py \
  "$LTX_TRANSFORMER" models/coreml/ltx_text_kv_1024_int8 \
  --blocks "$BLOCKS" --text-rows 1024 --variant int8_pc
```

`text256` is an approximate fast mode. It selects the first 256 conditioning
rows, so it needs a separate 256-row K/V export. It normally retains all
short-prompt tokens and removes later register rows; a long prompt can be
truncated and should use the 1,024-row path.

```bash
"$COREML_PY" tools/export_ane_kv_blocks.py \
  "$LTX_TRANSFORMER" models/coreml/ltx_text_kv_256_int8 \
  --blocks "$BLOCKS" --text-rows 256 --variant int8_pc
```

Artifacts are specific to checkpoint weights, Stage shape and text-row count.
Do not reuse the 480p artifacts for 720p.

## Run GPU+ANE

### Dense, quality-first heterogeneous mode

This keeps dense attention and the full context. The MLP split and Video-text
K/V run through Core ML with `CPU_AND_NE`; Video-heavy attention remains on
Metal. ANE is warmed by model loading and sessions are stage-scoped to avoid
unnecessary 64 GB unified-memory pressure.

```bash
LTX_MEDIA_BACKEND=mlx \
LTX_MLX_MEDIA_WARMUP=1 \
LTX_AV_PARALLEL_GPU=1 \
LTX_ANE_MLP_STAGE1_DIR=models/coreml/ltx_mlp_a6912_int8/stage1 \
LTX_ANE_MLP_STAGE2_DIR=models/coreml/ltx_mlp_a6912_int8/stage2 \
LTX_ANE_MLP_VARIANT=int8_pc \
LTX_ANE_MLP_RELEASE_FULL_GPU=1 \
LTX_ANE_KV_DIR=models/coreml/ltx_text_kv_1024_int8 \
LTX_ANE_KV_VARIANT=int8_pc \
LTX_ANE_KV_STAGE1=1 LTX_ANE_KV_STAGE2=0 \
./build/bench_block_mlx_ane "$LTX_TRANSFORMER" all --generate \
  "$LTX_UPSAMPLER" "$LTX_VIDEO_VAE" \
  outputs/conditioning/demo/connected outputs/videos/demo_480p_ane 42
```

Measured dense result: **49.467 s**, or **1.202x** versus the 59.483 s GPU
baseline. Video MLP's replaced full GPU weights are released after coverage
validation, saving about 6.0 GiB.

### Fast approximate candidate

The following is the current 46.824 s candidate. It is intentionally
explicit because it changes the computation:

- trims context to the first 256 rows in both stages;
- uses 256-row ANE K/V in both stages;
- uses tiled Sol only for Stage-2 blocks 1–46; blocks 0 and 47 stay dense;
- retains the original 8+3 diffusion updates.

```bash
LTX_MEDIA_BACKEND=mlx \
LTX_MLX_MEDIA_WARMUP=1 \
LTX_AV_PARALLEL_GPU=1 \
LTX_ANE_MLP_STAGE1_DIR=models/coreml/ltx_mlp_a6912_int8/stage1 \
LTX_ANE_MLP_STAGE2_DIR=models/coreml/ltx_mlp_a6912_int8/stage2 \
LTX_ANE_MLP_VARIANT=int8_pc \
LTX_ANE_MLP_RELEASE_FULL_GPU=1 \
LTX_ANE_KV_DIR=models/coreml/ltx_text_kv_256_int8 \
LTX_ANE_KV_VARIANT=int8_pc \
LTX_ANE_KV_STAGE1=1 LTX_ANE_KV_STAGE2=1 \
LTX_TEXT_ROWS_LIMIT=256 \
LTX_SOL_VIDEO_SELF_STAGE2=1 \
LTX_SOL_DENSE_EDGE_BLOCKS=1 \
LTX_SOL_DENSE_EDGE_STEPS=0 \
LTX_SOL_TAU=1.0 \
./build/bench_block_mlx_ane "$LTX_TRANSFORMER" all --generate \
  "$LTX_UPSAMPLER" "$LTX_VIDEO_VAE" \
  outputs/conditioning/demo/connected outputs/videos/demo_480p_fast 42
```

Use this mode only after checking the final video for the target prompt/seed
family. It passed the current fox visual check, but the broad human/complex
motion quality suite remains work in progress.

## What is on by default?

Without ANE artifact environment variables or explicit fast flags, generation
uses the quality-first GPU path: dense attention, full 1,024-row context and
the original 8+3 schedule. Sol, context pruning, GPU dual-queue AV overlap,
ANE MLP/K/V, QKV sequence split and experimental MLP boundary fusions are all
off by default.

Memory-lifetime safeguards are on by default for full generation:

- close the checkpoint mapping after upload when it is no longer needed;
- release blocks during the final Stage-2 step;
- release Transformer state before VAE decode.

These preserve the math and prevent the VAE from competing with DiT state.

## Benchmarks, tests and reports

```bash
make test
./build/bench_block "$LTX_TRANSFORMER" 24 1001 256 2 5
./build/bench_block "$LTX_TRANSFORMER" 24 4004 256 2 5
./build/bench_attention "$LTX_TRANSFORMER" 24 4004 5
```

Important reports:

- [technical-report-2026-09-04.md](notes/technical-report-2026-09-04.md):
  consolidated architecture, performance, quality and decisions;
- [plan.md](notes/plan.md): staged project plan and remaining acceptance gates;
- [gpu_ane.md](notes/gpu_ane.md): GPU+ANE design and scheduling rationale;
- [sol-attention-e2e-2026-09-03.md](notes/sol-attention-e2e-2026-09-03.md):
  Sol micro/E2E/quality evidence;
- [720p-gpu-2026-09-03.md](notes/720p-gpu-2026-09-03.md): high-resolution
  GPU baseline and memory constraints.

`notes/` retains negative results as well as wins. Do not re-enable a
rejected path solely because it improves a single operator benchmark; all
production choices require an original-8+3 E2E result and a final-video check.
