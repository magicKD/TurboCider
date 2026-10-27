# 图像首步耗时：测量工具与本机观察

开发工具 `tools/validation/image_first_step.py` 通过公开 C ABI 测量 Qwen Image 2.1
文生图／编辑，以及 Z-Image Turbo 文生图。Z-Image 当前原生执行器没有图片编辑入口，
工具拒绝为它构造编辑耗时。App 不依赖此 Python 工具。

## 口径

- 构造 engine 单独计时；请求总耗时包含编码、采样、解码、导出。
- `time_to_denoise_seconds`：进入第零个 denoise 边界之前的时间。
- `first_step_seconds`：denoise 0→1；后续步中位数来自 1→2 及以后。
- 同一 engine 默认四次请求：首次、相同提示词、改提示词、重复新提示词。
- `--prepare none|load|warmup` 独立记录准备成本。预热成本不计入后续请求，但保留在报告中。
- 首次指新 engine 的首次请求，不代表清空 OS 页缓存或 Metal 驱动缓存。
  报告保存构建身份、系统、完整请求、进度边界、原生指标和 PNG SHA256。
- 事件边界内同时可能包含惰性求值、编译、真实前缀计算等，不能直接称为编译耗时。

```sh
python tools/validation/image_first_step.py --model MODEL_ROOT --model-id z-image-turbo --residency resident --output outputs/startup-z
python tools/validation/image_first_step.py --model MODEL_ROOT --model-id qwen-image-2.1 --residency component_staged --steps 20 --output outputs/startup-qwen
python tools/validation/image_first_step.py --model MODEL_ROOT --model-id qwen-image-2.1 --residency component_staged --steps 20 --reference INPUT.png --output outputs/startup-edit
python tools/validation/image_first_step.py --model MODEL_ROOT --model-id qwen-image-2.1 --residency component_staged --steps 2 --lora ADAPTER.safetensors --sampler dmd2 --output outputs/startup-dmd2
```

为避免系统负载干扰，请串行运行推理，并在独立进程对照中重复采样。
这是启动诊断工具，不是外部框架配对性能评测。

## 2026-10-27 本机观察

Apple M4 Max、64 GiB、macOS 26.6.2、MLX 0.32.0；512×512、seed 42、纯 GPU、
BF16 基础组件、原始 VAE。Z-Image 常驻、8 步；Qwen 分组件驻留、基础 20 步
或 DMD2 2 步；编辑使用一张标准 1024 编码的参考图。
未使用 PE、ANE、DiT 近似缓存或 Tiny VAE。
以下每行是一次观察，不是置信区间或中位性能保证；单位为秒。

| 路线／请求 | 采样前 | 首步 | 后续步中位数 | 完整请求 |
|---|---:|---:|---:|---:|
| Z-Image 首次 | 3.151 | 0.873 | 0.844 | 10.309 |
| Z-Image 重复 | 0.001 | 0.878 | 0.844 | 7.053 |
| Z-Image 改提示词 | 4.326 | 3.485 | 0.844 | 14.023 |
| Z-Image 重复新提示词 | 0.001 | 0.850 | 0.843 | 6.996 |
| Qwen 文生图首次 | 4.808 | 1.138 | 1.048 | 26.525 |
| Qwen 文生图重复 | 0.367 | 1.129 | 1.048 | 22.007 |
| Qwen 编辑首次 | 6.246 | 5.953 | 1.228 | 36.419 |
| Qwen 编辑重复 | 0.674 | 1.241 | 1.227 | 25.999 |
| Qwen 编辑改提示词 | 4.709 | 5.953 | 1.228 | 34.856 |
| Qwen 编辑重复新提示词 | 0.741 | 1.245 | 1.227 | 25.936 |
| DMD2 文生图首次 | 4.839 | 1.163 | 1.130 | 7.816 |
| DMD2 文生图重复 | 0.437 | 1.155 | 1.081 | 3.288 |
| DMD2 编辑首次 | 6.343 | 6.096 | 1.270 | 14.509 |
| DMD2 编辑重复 | 0.980 | 1.274 | 1.254 | 4.125 |

DMD2 两路各做了首次与重复两次请求，均报告实际 2 步、128 个 LoRA 投影。
查看结果图确认文生图生成红色茶壶、编辑改为蓝色；这只是有限冒烟验证，
不是与 20 步基础模型等画质的证明，不能把减少步数包装成无损算子加速。

Qwen 文生图首步只略慢；编辑首次和改提示词时明显更慢，重复条件后恢复至常规步时。
Z-Image 改提示词出现一次首步尖峰，重复新提示词后消失。缓存／形状专用编译都可能
影响这些边界；本轮没有独立编译事件追踪，不能量化其中多少是 compile overhead。
已有条件／前缀复用确实值得保持，但尚未新增或验证持久化 Metal 编译缓存。
预热可以移动首次成本，不能将该成本从端到端体验中抹去。

本机原始报告位于忽略的 `outputs/image-startup-20261027/`：
`baseline-z`、`baseline-qwen-t2i`、`baseline-qwen-edit`、`dmd2-t2i`、`dmd2-edit`。
模型、媒体和包含本机路径的原始报告不提交 Git。
