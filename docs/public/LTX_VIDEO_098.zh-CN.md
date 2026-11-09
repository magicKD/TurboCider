# LTX-Video 0.9.8 图生视频（实验性）

本路径直接使用原生 MLX / Metal 在 GPU 上运行 `ltxv-2b-0.9.8-distilled.safetensors`，不需要运行 ComfyUI 或 Python 推理服务。该检查点包含 Transformer 和视频 VAE；还需要配套的 T5-XXL 文本编码器和 tokenizer。双阶段流程另需官方的 0.9.8 潜空间放大器。T5 使用 BF16 权重、FP32 中间计算，Transformer 和 VAE 使用 BF16，调度使用 FP32。模型权重受其自身许可约束，使用前请核对发布方的条款，尤其是商业使用限制。

## 模型目录

可以在 App 模型中心选择 **LTX-Video 0.9.8 2B Distilled**，使用 Hugging Face 来源安装，或注册已有目录。目录应包含：

```text
模型目录/
├── ltxv-2b-0.9.8-distilled.safetensors
├── ltxv-spatial-upscaler-0.9.8.safetensors  # 双阶段需要
├── text_encoders/
│   └── t5xxl_fp16.safetensors
└── tokenizer/
    └── tokenizer.json
```

模型库配方从 Lightricks/LTX-Video 获取检查点和潜空间放大器，从 comfyanonymous/flux_text_encoders 获取 T5-XXL，并从 google-t5/t5-base 获取 tokenizer；它们会映射到上述目录。放大器约 505 MB；没有它时仍可选择 8 步单阶段。模型下载示例见 `examples/requests/ltx098-model-library.json`。

## 生成

在 App 中选择此模型和「图生视频」，导入首帧图片，填写提示词并选择纯 GPU、常驻加载。默认双阶段是 7+3 次 Transformer 推理：第一阶段使用官方时间表，潜空间放大 2 倍后进行逐通道 AdaIN 统计匹配，再使用官方第二阶段时间表进行 3 步精修。121 帧、24 fps 约为 5.04 秒，无音轨。图片按基准画布比例居中裁剪后缩放；输出路径不能已经存在。

双阶段宽高是**基准画布**，而非始终等于最终输出。按照官方 0.6666666 缩小并向下对齐 32，512×512 的第一阶段为 320×320，放大和精修为 640×640。App 默认保留 640×640；打开「最后缩回基准尺寸」则输出 512×512，接近官方示例最后的双线性缩放。两者共享相同的生成阶段，缩回只改变最终分辨率，不会修复人脸、毛发或运动本身。也可选旧的 8 步单阶段，其输出与基准尺寸相同，不使用潜空间放大器。首帧强度 1 锁定首帧条件，后续帧仍然会生成。

CLI 请求示例位于 `examples/requests/ltx098-image-to-video.json`。将其中的图片和输出路径替换为实际绝对路径后运行：

```sh
dist/cli/turbocider generate /absolute/path/to/model examples/requests/ltx098-image-to-video.json
```

请求使用模型 ID `ltx-video-0.9.8-distilled`、操作 `video.image`，且必须有一个 `first_frame` 图片输入。首帧强度范围为 0–1。`steps: 10` 选择双阶段；可选 `ltx098_resize_to_base: true` 缩回基准尺寸（默认 false）。`steps: 8` 保留旧单阶段。基准画布宽高是 64–1024 的 32 倍数，帧数为 9–121 的 `8n+1`、帧率固定 24；基准总像素帧数不超过 512×512×121，第二阶段不超过 6400 个 latent token 和 640×640×121 像素帧。输出回执中的 `width` 和 `height` 为实际尺寸，`requested_base_width` 与 `requested_base_height` 为用户输入的基准尺寸。

这是实验性路径：目前不支持音频、LoRA、流式加载、自动提示词增强或 ANE；不同的 LTX 2.5 优化开关不能直接用于该模型。完整速度和画质随硬件、图片、提示词、种子及视频编码而变化。双阶段不保证优于 8 步单阶段，也无法保证人脸、肢体和运动一致性；模型与不同后端的数值及随机噪声实现并非逐位一致。

如果只在源码树的独立 `build/` 目录构建了新版本，已有的 `dist/TurboCider.app` 不会自动更新；请按项目构建流程生成需要使用的版本。

测量条件、失败的画质门槛和 ANE 组件实验见 [LTX 0.9.8 基准记录](../benchmark/LTX_VIDEO_098.md)。
