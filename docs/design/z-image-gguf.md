# Z-Image Turbo GGUF（native MLX）

更新时间：2026-09-09

正式 TurboCider App 只支持 native MLX GGUF，不包含 stable-diffusion.cpp。
运行时由 ZImage 加载 GGUF、tokenizer、Qwen3 和 VAE，使用 MLX C++/Metal；
不启动子进程、不使用 HTTP、不依赖 Python。

支持 tensor 类型：F32、F16、BF16、Q4_0、Q4_1、Q8_0。加载前读取 GGUF tensor
目录校验，不按文件名判断。mixed K-quant、GGUF streaming/offload 目前明确拒绝。

推荐布局：

```text
<model-root>/tokenizer/
<model-root>/<checkpoint>.gguf
<model-root>/split_files/text_encoders/qwen_3_4b.safetensors
<model-root>/split_files/vae/ae.safetensors
```

也支持 text_encoder/ 和 vae/ 的 diffusers 组件布局。多个 GGUF 时需明确
model_variant 或选择单文件；选择单文件时其父目录就是组件根目录。
不再扫描兄弟项目或通过环境变量选择 native 后端。

LoRA 支持 native inference_time / in_memory_merge。GPU+ANE 仍需显式的
checkpoint-bound Core ML manifest；带 LoRA 时要求 in_memory_merge 和匹配的
adapter identity。Core ML 已有分区的编译和执行无需 Python；从原始权重导出
Core ML 源分区仍需要开发准备工具链。

开发对照工具位于 tools/validation/sd_cpp/，不进入构建或发行包。
历史性能和 streaming 数据保留在 docs/design/validation/，不代表当前 native
能力。最新边界见 [整理说明](../status/native-gguf-boundary-2026-09-09.md)。
