# Qwen Image 2.1 DMD2 两步 LoRA

这是独立的实验性两步采样路线，使用原始 Qwen Image 2.1 基础模型，
不能与原生 Turbo 八步 checkpoint、Viggle 六步或普通基础采样互换。
App 推理仍为 C++/MLX/Metal，不需要 Python。

## 权重与采样契约

- 上游：`linoyts/Qwen-Image-2.1-DMD2-2steps`，固定版本
  `95c7e9c2b32607d5464d6b74a86d66c49ce87e03`。
- 上游文件名：`pytorch_lora_weights.safetensors`；可将本地文件命名为
  `Qwen-Image-2.1-DMD2-2steps.safetensors` 以便自动识别。
- 大小：67,144,520 字节；SHA256：
  `3a50c28d2f70170a06011e7b5d93a12418edbc4a0fc4de4acee8d9d227a38c21`。
- Rank 32、alpha 32、32 层各四个注意力投影，共 128 对 A/B。
  原生加载支持其 `.lora.down.weight` / `.lora.up.weight` 键名。
- 固定 **2 步、CFG 1、sigma `[1, 0.5, 0]`**；无动态 shift、无末端拉伸。
  LoRA 强度必须为 1，需明确接受近似模式。

仅接受这份固定摘要的官方适配器，不因文件名相同就信任权重。
生成时检查摘要与投影绑定，错误文件会被拒绝。请遵守上游模型卡的许可证和使用限制。

## App

在 Qwen 创作页或 Playground 添加并单独启用上述 LoRA。规范文件名会自动应用
512×512／2 步／纯 GPU／强度 1 预设；保留上游通用文件名时，点击
「应用官方 DMD2 两步设置」或 Playground 的对应按钮。其他适配器不能使用此按钮
冒充官方 DMD2，运行时仍会校验摘要。

支持文生图及 1–3 张参考图编辑、标准 1024 或快速 512 参考编码。
原生画布需 32 对齐、边长 64–2048、总面积不超过 1,048,576 像素；
App 快速 512 参考编码沿用 512×512 画布限制。
可选择 `resident` 或 `component_staged`；不与 ANE、提示词增强、DiT 近似缓存、
流式内存档位或 memory-constrained 配方组合。关闭／移除适配器会恢复基础采样设置。
仅验证了 512×512 的真实推理冒烟；其他尺寸／多参考图是契约准入，不代表画质资格。

## API

Schema 1 使用 `qwen21_sampler="dmd2"`，Schema 2 放在
`parameters.qwen21_sampler`；默认 `auto` 保持普通 LoRA／Viggle 行为。
`auto` 仅对上述规范本地文件名识别 DMD2，通用文件名必须显式指定 sampler。

```json
{
  "model": "qwen-image-2.1",
  "operation": "image.generate",
  "prompt": "A red ceramic teapot on a wooden table",
  "width": 512,
  "height": 512,
  "frames": 1,
  "audio": false,
  "steps": 2,
  "seed": 42,
  "execution": "gpu",
  "residency": "component_staged",
  "allow_approximation": true,
  "qwen21_sampler": "dmd2",
  "lora_strategy": "inference_time",
  "loras": [{"path": "/absolute/path/pytorch_lora_weights.safetensors", "strength": 1, "role": "transformer"}],
  "output": "/absolute/path/result.png"
}
```

Plan 的 `algorithm_approximations` 标记
`qwen21_dmd2_fixed_two_step_distillation`，结果报告实际步数与绑定投影数。
Plan 不读取完整权重；计划通过不能代替生成时的摘要校验。

## 开发验证

```sh
python -m unittest discover -s tests/native -p 'test_qwen21_dmd2.py'
TURBOCIDER_TEST_QWEN21_DMD2=/path/to/adapter.safetensors python -m unittest discover -s tests/native -p 'test_qwen21_dmd2.py'
TURBOCIDER_APP_TARGETS=qwen21 tools/native/build_app.sh
build/native/turbocider-qwen21-workflow-tests
build/native/turbocider-playground-tests
```

测试覆盖 v1/v2 准入、固定调度节点、128 个注意力绑定、PEFT 默认元数据、
危险配置拒绝、App 预设、持久化与模型／LoRA 切换。部分 App 测试需要剪贴板权限，
MLX 调度测试需要 Metal 设备。性能口径见[首步耗时记录](../benchmark/IMAGE_FIRST_STEP.md)。
