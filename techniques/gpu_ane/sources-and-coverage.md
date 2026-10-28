# 来源与证据：可追溯，但不让审计记录取代方法

<!-- knowledge
id: technique-gpu-ane-sources-and-coverage
type: navigation
-->

[方法导航](README.md) · [固定源码索引](sources.md) · [案例](case-studies/README.md)

本页帮助寻找可迁移实现，不提供审计或实验包。源码事实、作者报告、历史观察和推导分别阅读；固定读本不表示最新版本或本机已运行。

## 核心源码学习入口

| 方向 | 固定源码或本库读本 |
| --- | --- |
| GEMM/SDPA与GPU调度 | [MLX固定dispatch](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/quantized.cpp) · [GGML few-row](case-studies/ggml-fewrow-mma-and-reuse.md) |
| ANE图表达与封闭FFN | [Apple固定FFN](https://github.com/apple-aiml-research/ml-ane-transformers/blob/590eb35913622cdef217ebc5329047aab389ac98/ane_transformers/reference/ffn.py) · [Split-Einsum](case-studies/coreml-split-einsum-export-and-runtime.md) |
| 异构切分/控制 | [vpipe](case-studies/vpipe-controller-and-failure.md) · [ccv](case-studies/ccv-hybrid-cost-and-failure-boundaries.md) · [Splash](case-studies/splash-w8a8-prefill-and-layout.md) |
| 融合、replay与cache | [tinygrad](case-studies/tinygrad-metal-hcq-and-replay.md) · [MLX compile](case-studies/mlx-compile-and-cross-runtime-ffn.md) |
| 权重bank与阶段交接 | [MoE](case-studies/anemll-original-bank-gpu-consumer.md) · [Whisper](case-studies/whisper-generation-and-complete-encoder-handoff.md) · [FastVLM](case-studies/fastvlm-coreml-vision-and-mlx-handoff.md) |

## 阅读顺序

先读 selector/图表达，再读实际 kernel/执行层，最后追到消费者、owner 和失败分支。比较时固定任务与语义，不从项目名字、低位宽、统一内存或 plan 推导实际收益。

更多版本和关键文件见[源码索引](sources.md)。历史案例保留取舍与反例，但不附原始数组、hash 清单、采集程序或验证环境。材料规则见[维护说明](../MAINTENANCE.md)。
