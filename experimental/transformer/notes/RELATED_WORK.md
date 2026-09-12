# 调研笔记与设计依据

核查日期：2026-09-10。以下为原始论文或官方实现/文档；历史性能未并入本次统计。

1. [Megatron-LM (2019)](https://arxiv.org/abs/1909.08053)：以完整 attention head 和 MLP 中间维为张量切分单位，后续投影归约部分输出。本实验借用其代数分解，在单机 UMA 异构设备上重建同步与成本模型；训练集群的吞吐数字不能移植到本机。
2. [Reducing Activation Recomputation (2022/MLSys 2023)](https://arxiv.org/abs/2205.05198)：sequence parallel 结合 tensor parallel 减少训练激活内存。本文 FFN token-row split 只是无跨 token 依赖子图的行切分，不声称实现了论文的完整训练 SP。
3. [NVIDIA context parallel documentation](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/context_parallel.md)：跨上下文的 attention 仍需其他分片的 KV。因而本实验 head 分片保留每个 head 的完整 causal sequence；不能在每个 token shard 内单独做 attention 后直接拼接。
4. [Apple: Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)：ANE 友好布局为四维 channels-first，线性层可改写为 1×1 convolution；布局、复制和中间张量大小会影响执行效率。本文 private FFN 使用这一布局，同时实际计入 GPU/CPU 转置成本。
5. [MLPredictionOptions.outputBackings](https://developer.apple.com/documentation/coreml/mlpredictionoptions/outputbackings?language=objc) 与 [Optimize your Core ML usage](https://developer-mdn.apple.com/videos/play/wwdc2022/10027/)：支持调用者提供输出缓冲。本文 public runner 检验 MLMultiArray 对象和 dataPointer identity，仍保留 prediction 完成的同步屏障。
6. [Deploy machine learning and AI models on-device with Core ML (WWDC24)](https://developer-mdn.apple.com/videos/play/wwdc2024/10161/)：MLComputePlan 提供算子支持/首选设备及估计成本。本实验保存其原始返回值，把 compiler preference 与硬件 trace 区分。
7. [maderix/ANE](https://github.com/maderix/ANE)：private MIL / IOSurface 机制的来源。具体适配沿用 mac_local_ai 记录的 commit d91c9845c0784dec7753048954fc6d0e8411fe29 和动态权重 blob 修正；参见 THIRD_PARTY.md。这里不将 private ABI 当成发行接口。

## 与 mac_local_ai / TurboCider 现有工作的关系

`src/baseline.mm` 来自 mac_local_ai 的完整 TransformerBlock runner。复用 Metal/MPS、CPU BNNS 常驻工作线程、Core ML output backing、分段 FFN 和 fused attention 接口。新增内容包括独立 synthetic layer 数据集、切分间固定模型权重、全 head CPU/ANE 分支、完整 private FFN、GPU tiled layout bridge、串行消融、分层 bootstrap 与独立 FP32/FP64 校验。

TurboCider 现有 `docs/design/transformer-heterogeneous-report.md` 中 M4 Max 和真实模型数字只作为历史背景。本文的硬件、权重、序列与计时边界不同，不能把其真实模型加速比和这里的 synthetic stack 加速比直接排序。
