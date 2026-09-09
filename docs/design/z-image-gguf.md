# Z-Image Turbo GGUF backend

更新时间：2026-09-09

TurboCider now exposes a separate `z-image-turbo-gguf` model module. It accepts a local GGUF transformer plus the independent Z-Image VAE and Qwen3-4B text encoder:

```text
<model-root>/z-image-turbo-Q4_K_M.gguf
<model-root>/split_files/vae/ae.safetensors
<model-root>/split_files/text_encoders/qwen_3_4b.safetensors
```

The module deliberately reuses the same pinned `stable-diffusion.cpp` backend used by Unsloth's native GGUF path. This avoids implementing a second Q4_K/Q5_K/Q6_K/Q8_0 dequantizer and keeps the comparison meaningful. All GGUF files are selected by filename or `model_variant`; the backend does not convert them into a merged safetensors checkpoint.

## Runtime path

`native/platform/apple/sd_cpp_session.mm` starts one resident `sd-server` per selected GGUF checkpoint, waits for `/v1/models`, and submits `/sdcpp/v1/img_gen` jobs. Server setup is therefore paid once per session; the readiness measurement is not presented as proof that the operating system has faulted every lazily mapped weight page into memory. TurboCider owns the process lifetime, loopback port, cancellation, six-hour generation timeout, temporary LoRA files, and GPU lease. The sd.cpp process remains outside the parent MLX allocator, so its result reports `runtime_backend=stable-diffusion-cpp-metal` and `mlx_active_bytes=0`.

The fixed Apple Silicon binary is installed by:

```bash
make setup-sd-cpp
```

`tools/native/install_sd_cpp.py` verifies the pinned archive SHA-256 before placing it under ignored `.deps/stable-diffusion-cpp/`. The current pin is `master-813-bfbef5b-u13b9d92`, source commit `13b9d92b5e9a1563536c9c980e700470f9ab6702`.

### Low-memory streaming/offload

Set `residency` to `streaming` (or set `streaming_offload: true`) and provide an
explicit `memory_budget_bytes`. TurboCider then launches the pinned server with
`--params-backend diffusion=cpu,te=disk,vae=disk --mmap --stream-layers
--max-vram <GiB> --vae-tiling`. The diffusion parameters stay in host memory so
sd.cpp's layer prefetch/eviction path is actually enabled; the text encoder and
VAE remain disk-backed.
The request is routed to sd.cpp rather than native MLX/ANE, because the latter
requires resident packed weights. `in_memory_merge` LoRA is rejected in this
mode; separate LoRA files remain available through the request-time
`inference_time` path.

The requested budget is an sd.cpp `--max-vram` working-set hint, not a hard cap
on the whole child process or a promise that an 8 GiB Mac can run the model.
TurboCider reports the child process lifetime physical footprint separately;
sampled RSS can be higher because memory-mapped file pages are counted.

The corrected CPU-staged M4 Max ABBA×2 matrix at 256²/9 steps used one warmup
and four measured requests per route. All twelve paired decoded outputs were
pixel exact, while streaming reduced lifetime physical footprint by
33.3–45.9%:

| Variant | Resident warm median | Streaming warm median | Streaming overhead | Resident → streaming physical footprint |
|---|---:|---:|---:|---:|
| Q3_K_S | 9.551 s | 10.031 s | +5.02% | 15.015 → 10.013 GB (−33.3%) |
| Q4_K_M | 9.497 s | 9.994 s | +5.23% | 16.063 → 10.010 GB (−37.7%) |
| Q8_0 | 9.341 s | 10.679 s | +14.32% | 18.484 → 10.004 GB (−45.9%) |

The default material-regression gate is streaming/resident ≤ 1.02, so none
of these low-memory routes is currently performance-qualified as a free
optimization. They are valid explicit low-memory fallbacks with exact decoded
RGB results at 256².

Q4_K_M at 1024² with the 8 GiB hint measured 94.586 s resident and 107.956 s
streaming (`1.1414×`), while physical footprint fell from 15.454 GB to
10.015 GB (`35.2%`). The decoded images were not pixel exact, but passed the
explicit image-quality gate: correlation `0.998117`, cosine `0.999823`, and
MAE `2.147/255`. A direction-only 16 GiB probe remained `1.1143×` slower, so
raising the hint did not make this a performance optimization. The official
independent LoRA at 256² also remained a separate request-time file and produced
4/4 pixel-exact resident/streaming pairs, with `37.6%` footprint reduction and
`1.0386×` runtime ratio.

The current sanitized evidence is
[`z-image-gguf-streaming-2026-09-09.json`](validation/z-image-gguf-streaming-2026-09-09.json).
The 2026-09-08 files are retained as historical disk-backend measurements; that
configuration did reduce footprint, but sd.cpp ignored `--stream-layers` when
the diffusion parameter backend was `disk`.

## Quantization and validated files

The selection layer and pinned sd.cpp runtime recognize Q2_K, Q3_K_S/M/L,
Q4_0/Q4_1/Q4_K_S/Q4_K_M, Q5_0/Q5_1/Q5_K_S/Q5_K_M, Q6_K, Q8_0,
IQ4_NL/IQ4_XS, F16, BF16 and F32 labels. Three quantized transformers have
actually been downloaded and executed locally:

```text
Q3_K_S  3,951,806,016 bytes  sha256 918f233b1697b36f2906b86187c8514d60676eb3b7bcf29dd24b93ba33a19283
Q4_K_M  5,017,613,376 bytes  sha256 e6494f87de6abaf6a561924f50317a5f271fc34bb4222aabbd801197df8f7daa
Q8_0     7,224,707,136 bytes  sha256 f163d60b0eb427469510b8226243d196574a18139a2e40c017409cfbda95ecfe
```

Q3_K_S and Q8_0 have representative 256×256 load/generation/parity samples;
Q4_K_M additionally has an interleaved 1024×1024 acceptance run. The other
recognized labels are selection/runtime capabilities, not claimed validation.
`runtime_precision` reports the selected filename quantization, for example
`gguf:q4_k_m`.

## Independent LoRA

LoRA remains a separate file. For each request TurboCider stages a symlink (copy fallback) with a unique filename directly in the server's LoRA directory and sends:

```json
{"lora":[{"path":"gen_1_style.safetensors","multiplier":1.0}]}
```

The staged file is removed when the request ends. No merged checkpoint is written. This is request-time application inside sd.cpp, which is the correct mode for a quantized GGUF base. The native MLX Q8 path applies the same official file as an in-memory delta and reports all 238 matched projections. The direct comparison server and TurboCider-managed server must both start with `--cfg-scale 1.0`, because Z-Image Turbo is distilled and the sd.cpp default is otherwise 7.0. The managed launch now passes this flag explicitly. With symmetric 100 ms result polling, the matched Q4_K_M request-time LoRA ABBA×2 run measured 13.1163 s for TurboCider versus 13.1240 s for direct and all four paired decoded images were pixel exact. Separately, at 256×256/9 steps/seed 42, native Q8+LoRA warm was 3.654 s versus 13.512 s for the pinned sd.cpp reference. The native Q8 and sd.cpp runtimes use different RNG/latent conventions, so their PNG correlation is diagnostic rather than a strict numerical parity gate; against the same TurboCider BF16+LoRA pipeline, Q8 correlation was 0.9775.

The Q8 LoRA-bound Core ML export and compiled 32-block manifest were exercised at 1024×1024. Checkpoint and LoRA SHA/strength provenance were verified, with 288 Core ML calls per request and zero output-copy bytes. The warm end-to-end wall median was 65.02 s for GPU+ANE versus 55.76 s for native GPU; denoise alone improved from 48.59 s to 46.68 s. This does not pass the required end-to-end not-slower gate, so LoRA-bound ANE remains explicit and `auto` continues to select GPU.

## GPU and ANE status

Mixed K-quants use sd.cpp Metal acceleration (`--diffusion-fa --diffusion-conv-direct`) with the Qwen3 text encoder on CPU (`--clip-on-cpu`); the reference server must also use `--cfg-scale 1.0`. The external sd.cpp HTTP API cannot skip a quantized FFN prefix and join an ANE result, so those variants remain GPU-only.

Native MLX GGUF now supports Q8_0/Q4_0/Q4_1 affine weights. For the validated Q8_0 path, Core ML owns `[0,4096)` of each 10240-wide gated FFN, while MLX Metal owns attention and `[4096,10240)`. The GPU suffix is submitted with `mx::async_eval` before the Core ML prediction and both branches join in the residual block. A checkpoint-bound 32-block Q8_0 manifest was compiled and run at 1024²/9 steps with 288 calls/request and zero output copies. Resident warm wall was 37.5848 s versus 45.9634 s for native Q8 GPU, a `1.2229×` speedup. GPU↔GPU+ANE PNG correlation was 0.9991566. This remains an explicit Q8_0 candidate; it is not automatic for all GGUF variants or LoRA requests. Public Core ML reports `CPUAndNeuralEngine`, but OS-level ANE residency remains unobservable and is reported as unknown.

## Matched Unsloth/runtime measurement

Use `tools/native/benchmark_z_image_gguf.py` after the model and runtime are installed to compare TurboCider with the direct pinned sd.cpp server. Use `tools/native/benchmark_z_image_gguf_streaming.py` for a repeated resident/streaming ABBA comparison; it records per-request wall time, decoded-pixel metrics, and child lifetime physical footprint. Both tools keep sessions resident and use the same files, sampler, prompt and seed. The streaming tool's required gate defaults to a material ratio of 1.02, a 25% footprint reduction, correlation ≥ 0.99, cosine ≥ 0.995 and MAE ≤ 5/255; strict no-slowdown and decoded-pixel equality are reported separately. `--require-pixel-exact` is available for workloads that genuinely require it.

Current matched results are:

| Variant/workload | Direct median | TurboCider median | Ratio | Pixel parity |
|---|---:|---:|---:|---|
| Q4_K_M, 256², ABBA×2 | 16.3675 s | 16.3467 s | 0.99872 | exact, 4/4 |
| Q4_K_M, 1024², ABBA | 179.7119 s | 179.6199 s | 0.99949 | exact, 2/2 |
| Q3_K_S, 256², AB | 16.3700 s | 16.3493 s | 0.99874 | exact |
| Q8_0, 256², AB | 15.9540 s | 15.9074 s | 0.99708 | exact |
| Q4_K_M + official LoRA, 256², ABBA×2 | 13.1240 s | 13.1163 s | 0.99942 | exact, 4/4 |

The base Q3/Q4/Q8 samples and the current Q4_K_M LoRA sample pass the literal not-slower requirement. An earlier LoRA result of 23.5064 s versus 23.3321 s is superseded: it compared a TurboCider server missing `--cfg-scale 1.0` with a direct server that already used it. After fixing the managed launch and making both polling loops 100 ms, the outputs and workload match. These figures prove that the TurboCider bridge does not regress the same pinned Unsloth sd.cpp runtime for the validated requests; they do not prove that GGUF is faster than TurboCider's native BF16 Z-Image path. At 1024² the Q4 runtime is about 179.6 seconds versus 36.37 seconds for the separately validated native BF16 GPU path.

The aggregate sanitized evidence is stored in `docs/design/validation/z-image-gguf-2026-09-07.json`; the corrected LoRA gate is stored in `docs/design/validation/z-image-gguf-q4km-lora-cfg1-2026-09-08.json`. Raw images, logs and machine paths remain under ignored `outputs/`.
