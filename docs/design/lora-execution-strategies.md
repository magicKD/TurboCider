# TurboCider LoRA 执行策略

更新时间：2026-09-09

TurboCider 的 schema 1、schema 2、C++ runtime、Swift binding 和 Studio App 现在使用同一个 `lora_strategy` 契约。LoRA 文件仍以独立资产传入；策略只决定基础权重和低秩增量在什么阶段组合，不会修改用户的基础 checkpoint 或 LoRA 文件。

## 公共策略

| `lora_strategy` | 含义 | 磁盘行为 |
|---|---|---|
| `auto` | 根据模型 descriptor 的 `default_lora_strategy` 解析 | 由解析后的策略决定 |
| `disk_premerge` | 使用离线生成、provenance-verified 的预融合 checkpoint | 运行时不创建融合产物 |
| `in_memory_merge` | 加载权重时在 unified memory 中应用 delta | 不写 merged checkpoint |
| `inference_time` | 请求期间由执行后端加载独立 adapter | 不长期保存融合后的 checkpoint |

没有 LoRA 文件时只能使用 `auto`，计划中报告 `lora_strategy=none`。带 LoRA 时，`auto` 会解析为模型默认值；未知策略以及模型未声明的策略均 fail closed。旧的 `lora_fusion` 字段继续保留，用于兼容已有 telemetry，例如 `load_time_baked`、`in_memory_delta`、`inference_time_low_rank` 和 `premerged_manifest_verified`。

## 当前模型能力

| 模型 | 支持策略 | 默认值 | 当前实现边界 |
|---|---|---|---|
| FLUX.2 Klein 4B/9B | `in_memory_merge` | `in_memory_merge` | Transformer/Text Encoder 在加载时融合；不写 merged checkpoint |
| Z-Image Turbo safetensors | `in_memory_merge`、`inference_time` | `in_memory_merge` | BF16 内存融合已完成 ComfyUI parity；packed ConvRot 可用低秩运行时分支保持量化基础权重 |
| Z-Image Turbo GGUF | `inference_time`、`in_memory_merge` | `inference_time` | Q8_0/Q4_0/Q4_1/F16/BF16/F32 使用 native MLX 内存融合或 packed 低秩分支；mixed K-quant 不支持 |
| MiniMax H3 Turbo | `disk_premerge` | `disk_premerge` | 仅使用已安装的预融合 checkpoint 与验证通过的 sidecar；不启动 Python |
| LTX 2.5 Distilled | `disk_premerge` | `disk_premerge` | 仅使用已安装的预融合 checkpoint 与验证通过的 sidecar；不启动 Python |
| Wan 2.1 1.3B QAD | `disk_premerge` | `disk_premerge` | 要求预融合 checkpoint 和完整 provenance manifest；ANE 仍需单独匹配 artifact |
| LLaDA-Image-Turbo | 无 | 无 | 当前原生 executor 拒绝 LoRA；旧 worker 仅是显式诊断入口 |

Z-Image GGUF 只使用 native MLX，默认 inference_time，也支持显式 in_memory_merge。不再使用后端环境变量或 sd.cpp fallback；不支持的 tensor 类型在加载时拒绝。带 LoRA 的 gpu_ane 只允许 in_memory_merge，且 Core ML 分区必须绑定相同 adapter provenance。

## 请求示例

Schema 1：

```json
{
  "schema_version": 1,
  "model": "z-image-turbo-gguf",
  "model_variant": "Q8_0",
  "lora_strategy": "inference_time",
  "loras": [
    {"path": "models/loras/style.safetensors", "role": "transformer", "strength": 0.8}
  ]
}
```

Schema 2 的 `lora_strategy` 同样位于请求顶层。模型目录、基础 GGUF 和 LoRA 保持独立；结果的顶层和 `plan` 都会记录解析后的策略。

## ANE 与量化限制

FLUX/Z-Image 的 ANE artifact 不是通用 LoRA runtime。带 LoRA 时必须使用绑定相同 path、bytes、SHA-256、role 和 strength 的 artifact；基础 artifact 不能复用。

Z-Image ConvRot/native GGUF 的 `in_memory_merge` 会把 LoRA 命中的量化 projection 解量化为 dense BF16。`inference_time` 现在保留 packed base projection，并在投影输出处追加 FP32 累积的 `x @ A.T @ B.T * scale`；普通 projection 和 Z-Image fused QKV 输出切片均已接通，命中 runtime LoRA 时会避开只接受 dense 权重的 compiled block。

当前 256×256 ConvRot 官方 LoRA 对比说明该路径仍是低内存候选，而不是默认质量路径：

| 指标 | `in_memory_merge` | `inference_time` |
|---|---:|---:|
| warm request wall | 2.6349 s | 2.9408 s |
| warm denoise | 2.5033 s | 2.8090 s |
| active MLX | 12.52 GB | 6.67 GB |
| 相对内存融合 PNG correlation / cosine | baseline | 0.978807 / 0.997737 |

`inference_time` 将 active MLX 降低约 46.7%，但 warm wall 增加约 11.6%，且最终 latent correlation 仅 0.986464，尚未达到把它设为默认值的严格 parity 门槛。LoRA strength=0 的输出与 packed base PNG 逐像素一致，证明低秩分支没有非零残留。完整数据见 [z-image-convrot-inference-time-lora-2026-09-08.json](validation/z-image-convrot-inference-time-lora-2026-09-08.json)。

验证摘要见 [lora-strategy-contract-2026-09-08.json](validation/lora-strategy-contract-2026-09-08.json)。
