# Native-only GGUF production boundary

2026-09-09：本说明取代旧状态文档中的 sd.cpp 发行、mixed K-quant 和 GGUF streaming 能力描述。

- 正式模型 ID 保持 z-image-turbo-gguf，内部只使用 MLX C++ ZImage executor。
- 不启动 sd-server，不包含 HTTP bridge，不扫描 PATH 或兄弟项目。
- App/CLI 不携带 sd-cli、sd-server，也不要求安装 sd.cpp 才能构建或打包。
- 格式范围：F32、F16、BF16、Q4_0、Q4_1、Q8_0，实际 tensor 类型在加载前校验。
- mixed K-quants 和 GGUF streaming 明确拒绝；H3 streaming 不受本次调整影响。
- 保留 native inference_time / in_memory_merge LoRA 和显式 GPU+ANE。
- GGUF 及已准备 Core ML artifact 的执行无需 Python；Core ML 导出、H3/LTX 磁盘 LoRA merge、FastMetal 的原有依赖不在本次移除范围内。

模型组件应明确放在所选根目录内，布局见 ../USAGE.md。多个 GGUF 不再按文件名优先级猜测；请选择明确的 variant 或单个文件。

sd.cpp 仅保留在 tools/validation/sd_cpp，供开发者做质量/速度对照。
旧验证 JSON 不作改写，避免把历史 sd.cpp 测量误标为 native MLX 测量。

## 本轮验证

- Native/Swift 构建、打包及签名校验通过，App/CLI 包不含 sd-server 或 sd-cli。
- make test 通过；系统 Python 缺 NumPy 而跳过的两项，已用仓库开发 Python 单独补跑通过。
- 新增 CPU-only GGUF tensor 元数据校验测试：支持类型、混合不支持类型、改名、截断和越界。
- 将 CLI 包复制到 /private/tmp，在源码目录外以 `env -i PATH=/usr/bin:/bin` 执行真实 Q8_0 256×256/9-step 生成成功，后端为 mlx_cpp_metal_gguf。
- 同样的清空环境测试下，已有 GGUF Core ML 分区执行成功，后端为 mlx_cpp_metal_gguf+coreml，记录 288 次 Core ML runtime prediction，checkpoint SHA 校验通过。
- 两次真实 GPU 测试因沙箱内 Metal 不可用，经批准在沙箱外执行；模型资源由明确的临时目录链接提供。
- 上述为单机功能/独立运行验证，不是全格式、多机器质量或性能矩阵。
