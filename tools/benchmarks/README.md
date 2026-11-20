# Qwen ConvRot benchmark tools

Run these scripts from the repository root with the project Python environment.
Use `--help` to see each script's build, model and output arguments. Prior work
and its validation limits are recorded in the [ConvRot experiment archive](../../docs/experiments/qwen21-convrot/README.md).

| Entry point | Measurement |
| --- | --- |
| [Projection benchmark](qwen21_convrot_projection_benchmark.py) | Selected real encoder and DiT weights with synthetic activation inputs. `--prepare-only` extracts bytes and compiles the harness without running the GPU. |
| [Private ANE FFN benchmark](qwen21_convrot_ane_ffn_benchmark.py) | One real FFN, comparing GPU W8A8 with a fixed diagnostic channel split. |
| [Encoder/DiT ablation](qwen21_convrot_w8a8_benchmark.py) | Separate encoder and DiT activation routes with per-process memory sampling. `--plan-only` validates requests without loading model weights. |
| [Streamed GPU/ANE comparison](qwen21_convrot_stream_ane_benchmark.py) | Complete streamed requests in ABBA or BAAB order. Requires an explicitly private diagnostic build. |
| [Weight-cache budget sweep](qwen21_stream_cache_budget.py) | Forward and reverse budget order, output agreement and sampled peak memory. The budget measures managed weight capacity. |

Use a fresh directory under ignored `outputs/` for each run. The
[performance guide](../../docs/public/PERFORMANCE.md) explains how to keep
component timings, full requests and publication claims separate.
