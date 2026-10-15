# 可选产品接入：与自包含核心分开

[总目录](../README.md) · [自包含 ANE 样本](../labs/ane.md) · [混合 FFN 设计](../designs/hybrid-ffn.md)

本目录保留原来的完整 TurboCider 原生 Public W8A8 集成能力，没有用较小教学边界冒充或替代它。它需要产品`native/`、`tools/`、`tests/native/`，**不是学习核心的运行依赖**，也不依赖 Splash/vpipe/Sana checkout。只复制`techniques/`时，使用本地 CPU/Metal/Core ML 样本，不运行这个可选接入脚本。

[public_w8_product.py](public_w8_product.py)导出完整 normalized SwiGLU/W8A8 图，调用产品两个原生 fixture，测试真实 GPU stage→CoreML→restore、修正输入、bank/source、late failure 等合同。仍不授予实际模型 LoRA、端到端性能或硬件 ANE 驻留资格。

```sh
python -B techniques/integrations/public_w8_product.py --coreml-overlay "$TC_COREML9_OVERLAY" --output outputs/new-product-w8-integration.json
```

使用已准备好NumPy/CoreML等依赖的Python环境；脚本不安装包或下载模型。原始报告写新的ignored路径，源码输入前后校验SHA。历史结果见[上一轮验收](../review/labs-validation.md)，当前教学源码结果另见[自包含验收](../review/self-contained-validation.md)。
