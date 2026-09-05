# Model Conformance

| Model | Tasks and inputs | GPU | GPU+ANE | Important constraints |
|---|---|---:|---:|---|
| MiniMax H3 LightX2V Turbo | text/video, first and last frame, reference image/video/audio | Yes | Explicit experimental plans only | Native output always includes audio; Ref2VA requires a compatible Ref2VA checkpoint |
| LTX-2.5 22B Distilled | text-to-video with synchronized audio | Yes | Yes, validated 704×480 request | Native output is 704×448, 97 frames at 24 fps, fixed 8+3 schedule |
| FLUX.2 Klein 4B | text-to-image | Yes | Yes, validated persistent 512×512 path | 1088 sequence bucket; other shapes fail closed unless another manifest is installed |
| FastMetal 1.3B QAD | text-to-video | Yes | Yes, validated fixed-shape path | 832×480, 81 frames, 16 fps, no audio, fixed 3-step DMD schedule; hybrid requires the measured M4 Max profile |

For LTX, conformance is checked at both boundaries: the standalone native GPU
versus GPU+ANE decoded-pixel path records BF16 cosine/rel-L2 metrics, while
direct worker versus TurboCider runs require byte-identical latent, pixel,
encoded video, and decoded audio results for the same backend. The validated
hybrid plan must also record a matching device profile rather than relying on a
forced ANE override.

FastMetal conformance has two levels. Direct engine versus TurboCider must be
byte-identical for the same GPU or GPU+ANE plan, including the saved denoised
latent and encoded MP4. GPU+ANE versus pure GPU is an approved approximate
comparison and must retain the recorded latent cosine/relative-L2 and decoded
SSIM/PSNR thresholds. The hybrid manifest must cover all 30 blocks and declare
the exact 32,760-row, 1,536-hidden, 8,960-intermediate partition; partial block
coverage and mismatched shapes are rejected before generation.

The common request keeps execution placement, generation profile, and approximation level independent. Engine-specific options remain namespaced under `engine_options`.

## Interface conformance

All four front doors use the same serialized request and job record:

| Surface | Submit/plan | Progress + ETA | Cancel | Advanced engine options |
|---|---:|---:|---:|---:|
| Python CLI | Yes | Attached stderr or daemon jobs | Yes | Inline JSON, `@file`, native args/env |
| HTTP API | Yes | Polling and SSE | Yes | Recursive namespaced JSON |
| Swift SDK | Yes | Polling and `AsyncThrowingStream` SSE | Yes | Recursive `TCJSONValue` |
| SwiftUI App | Yes | SSE with polling fallback | Yes | Advanced JSON panel |

The conformance suite exercises a complete App submission against a temporary
fake H3 executable and requires an intermediate progress update, non-null ETA,
and successful terminal event. This is a control-plane/UI integration test;
the expensive real-model quality and timing evidence remains the benchmark
matrix in `PERFORMANCE.md`.

Request validation rejects task/output-type mismatches, image requests with
more than one frame, audio on non-video tasks, audio for models without audio
output, disabling audio for a native path that always emits it, an LTX audio
request without the configured Audio VAE, and H3 requests at an FPS other than
24. Daemon restart tests require
completed records to remain queryable and previously active records to become
terminal `interrupted` failures.

## Pinned preparation sources

| Model pack | Repository | Revision | Managed preparation |
|---|---|---|---|
| `minimax-h3-turbo` | `MiniMaxAI/MiniMax-H3` + `lightx2v/Minimax-h3-Turbo` | `42ed227e…` + `2f8ea0dc…` | FL2VA base, v1.1 Turbo LoRA merge, explicit fixed-row Core ML export |
| `ltx-2.5-distilled` | `Lightricks/LTX-2.5` | `5e6e7101…` | Five runtime-compatible Comfy/convrot files, 48-block MLP and text K/V export |
| `flux2-klein-4b` | `black-forest-labs/FLUX.2-klein-4B` | `e7b7dc27…` | MLX snapshot, 20-block INT8 ANE export, content-addressed compiled cache |
| `fastmetal-1.3b-qad` | `FastVideo/FastMetal-1.3B-QAD` | `2dac0154…` | Runtime-compatible MLX snapshot and complete 30-block fixed-shape INT8 Core ML export |

The full revisions are stored in the JSON model packs and emitted by
`turbocider prepare-model ... --dry-run`. Gated repositories use the normal
Hugging Face token environment/keychain; TurboCider never places tokens in
commands or receipts.

## FLUX.2 Klein 9B readiness audit

Klein 9B is not a TurboCider model pack yet. The September 4, 2026 audit used
repository revision `92196c8e11f7b6cf2b7493e037d8c5345c559216` and did not
download weights. Hugging Face metadata reports 52,888,736,795 bytes for the
whole repository. Excluding the duplicate 18,157,185,168-byte single-file
checkpoint and sample media, the Diffusers/MLX runtime subset is
34,722,802,792 bytes (32.34 GiB).

The host had 76.85 GiB free, leaving 44.51 GiB after that subset. Scaling the
measured 4B source-plus-compiled Core ML footprint by hidden width, MLP width,
and block count estimates about 6.75 GiB for 9B ANE artifacts, before temporary
conversion files. Capacity is therefore sufficient only when the duplicate
single-file checkpoint is excluded and TurboCider's free-space reserve is
retained.

The remaining blockers are functional and contractual:

- The repository is gated under the FLUX non-commercial license. On September
  4, 2026, the current credential still received HTTP 403 for both
  `transformer/diffusion_pytorch_model.safetensors.index.json` and
  `transformer/diffusion_pytorch_model-00001-of-00002.safetensors`. Access and
  intended product use must be approved before any download.
- No 64 GiB GPU memory result, direct/TurboCider parity result, GPU+ANE speedup,
  or cross-backend image-quality result exists for 9B.

The code-side architecture blockers found by the audit are resolved, but this
is readiness work rather than production admission. `flux2-engine` now selects
the matching 4B, 9B, or 9B-KV mflux configuration and rejects a declared
variant that conflicts with `transformer/config.json`. The ANE exporter reads
indexed sharded safetensors, derives hidden/MLP dimensions and the complete
single-block range from checkpoint headers, and records those dimensions in
the manifest. Runtime artifact selection matches sequence bucket, hidden
dimensions, block coverage, and model block range, so installed 4B and 9B
artifacts cannot be selected interchangeably.

Production admission requires a real four-step 9B GPU run without OOM or
material swap pressure, byte-identical direct/TurboCider output for each
backend, no more than 2% engine overhead through TurboCider, and at least a 5%
warm engine speedup from GPU+ANE. A 9B-specific decoded-image quality gate must
be established from real GPU and GPU+ANE outputs and then passed. Until access,
license, memory, parity, performance, and quality checks all pass, TurboCider
must not add, advertise, or auto-select a production Klein 9B model pack.
