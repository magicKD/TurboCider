# 可选产品与上游接入：与自包含核心分开

[总目录](../README.md) · [自包含 ANE 样本](../labs/ane.md) · [混合 FFN 设计](../designs/hybrid-ffn.md)

本目录的 Public W8A8 脚本保留原来的完整 TurboCider 原生集成能力，没有用较小教学边界冒充或替代它。它需要产品`native/`、`tools/`、`tests/native/`，**不是学习核心的运行依赖**，也不依赖 Splash/vpipe/Sana checkout。只复制`techniques/`时，使用本地 CPU/Metal/Core ML 样本，不运行这些可选接入。

[public_w8_product.py](public_w8_product.py)导出完整 normalized SwiGLU/W8A8 图，调用产品两个原生 fixture，测试真实 GPU stage→CoreML→restore、修正输入、bank/source、late failure 等合同。仍不授予实际模型 LoRA、端到端性能或硬件 ANE 驻留资格。

```sh
python -B techniques/integrations/public_w8_product.py --coreml-overlay "$TC_COREML9_OVERLAY" --output outputs/new-product-w8-integration.json
```

使用已准备好NumPy/CoreML等依赖的Python环境；脚本不安装包或下载模型。原始报告写新的ignored路径，源码输入前后校验SHA。历史结果见[上一轮验收](../review/labs-validation.md)，当前教学源码结果另见[自包含验收](../review/self-contained-validation.md)。

## MFA / MPS GEMM研究客户端

[mfa_gemm_bench.swift](mfa_gemm_bench.swift)是可选的原始Swift客户端，需要另外固定版本的`philipturner/metal-flash-attention`之`FlashAttention`模块；目前参考版本为`8671cddc38f19a6eadb804dee6a3ca2954b8bf32`。在上游的隔离副本中添加executable target使用，不修改参考checkout，也不将它加入产品依赖。

它比较合成FP16输入的MPS与MFA单个驻留GEMM，含ABBA顺序、每对输入generation、完整command完成、padding初始化、逐轮NaN输出poison、尾部guard、有限值检查和每arm独立的抽样double oracle。它不是完整FFN、真实权重、模型质量或端到端性能验证；MPS也不等于产品优化后的compiled MLX基线。

2026-10-15已有尝试在上游动态Metal shader编译时报`illegal string literal in 'asm'`；另一次上游测试缺少XCTest。尚无有效的数值/性能报告。本次整理仅通过Swift语法解析，未重新构建上游或运行GPU；不能据此宣称兼容性或加速。研究状态继续在[工作台](../review/research-campaign-20261015.md)保留。

## Qwen真实FFN捕获与可移植fixture

[capture_qwen_ffn.py](capture_qwen_ffn.py)调用现有native诊断hook：仅base、512px、20steps、seed42、cache off，强制GPU FFN；保存3层×3步的9个原始BF16张量，最多约72 MiB。校验CLI构建的native source hashes，保留二进制hash和checkpoint stat identity；不拿任意旧binary冒充当前源码。

产品request仍要求显式公共runtime manifest，即使chunks=0；它必须是完整H4096/F12288的SwiGLU，而不是6144-channel子图。本轮先后发现这两项请求/geometry拒绝，未绕过验证器。正确入口是：

```sh
.venv/bin/python -B techniques/integrations/capture_qwen_ffn.py \
  --native-dir "$TC_NATIVE_DIR" --model-root "$TC_QWEN_ROOT" \
  --ane-manifest "$TC_PUBLIC_RUNTIME_MANIFEST" --output-dir outputs/my-new-qwen-capture
.venv/bin/python -B techniques/integrations/qwen_ffn_fixture.py \
  --checkpoint "$TC_QWEN_CHECKPOINT" --capture-root outputs/my-new-qwen-capture \
  --layer 0 --output-dir outputs/my-new-qwen-ffn-fixture
```

两个输出目录必须是新的。实际模型、原始prompt/图片/张量只留本地ignored目录。捕获是带instrumentation的请求，不作普通wall分数；不会安装/替换App、下载模型、修改默认路由或清理系统缓存。

[qwen_ffn_fixture.py](qwen_ffn_fixture.py)严格读取原checkpoint选定BF16 tensor与其capture，导出FP16数组、原BF16 compiled MLX replica输出及hash。BF16→FP16是新的近似表示，不自动继承原模型质量；Replica也不替代完整请求oracle。生成fixture后核心[tiling lab](../labs/ane.md)无需读取产品或兄弟源码。

本轮结果：[真实Qwen FFN证据](../review/qwen-ffn-tiling-20261015.md)。GPU捕获成功不等于新候选通过；数值通过也不等于消费者更快或请求画质合格。

## 有界attention原配方replay

[capture_qwen_attention.py](capture_qwen_attention.py)只编译[独立小客户端](capture_qwen_attention.cpp)，显式复用与当前source/MLX assets匹配的已有public native library。不会自动重建全项目或替换App；不匹配时拒绝，不能用旧ABI勉强运行。

```sh
.venv/bin/python -B techniques/integrations/capture_qwen_attention.py \
  --native-dir "$TC_NATIVE_DIR" --model-root "$TC_QWEN_ROOT" --seed 42 \
  --output-dir outputs/my-new-attention-capture
```

ordinary GPU observer在actual完整step0、step1 layer0提供借用参数、真实prefix cache和原callable输出。客户端按原公式重放、暴露额外QKV/O，核对完整block/prefix/输入存储后才写BF16副本；它不是未改变compiled kernel内部的直接capture。两phase合计128MiB cap，停在VAE decode/export之前；普通路线可能预加载已有VAE权重。source、loaded runtime、文件hash与统计生成前后复核，私有/实验环境不继承到子进程。

核心benchmark只需要这些显式raw文件，不需要产品或兄弟源码；[全部结果](../review/qwen-attention-20261016.md)与[步骤](../labs/gpu.md)包含独立seed确认、noop和empty-row拒绝。原始QKV与prompt/session本地数据不入库。
