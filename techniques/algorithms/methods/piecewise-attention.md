# PISA：关键块精算，其余块分段近似

<!-- knowledge
id: technique-algorithms-methods-piecewise-attention
type: method
stage: dit
operator: attention, sampling
backend: mlx, metal
device: gpu
symptom: precision-loss, launch-bound
semantics: approximate
representation: bf16
boundary: request, producer-consumer
api: reference-only
-->

配套：[机制 witness 与证据合同实验](../../examples/README.md)。这是支持性小实验，不是本方法的完整模型复现。

[算法入口](../README.md) · [Sol-Attn](sol-attention.md) · [SVG-EAR](svg.md)

来源：作者 `piecewise_sparse_attn_hyd.py` 关键前向与 README 的0阶变体说明 [S12](../sources.md#s12)；独立0阶kernel未逐行审计。分类：`approximate-train-free`。原 kernel 主要面向 NVIDIA Hopper/Triton，不是 Apple 库。

## 精确分支与近似分支如何合并

块均值的 QK proxy 排序，选择固定 density/top-k 的 critical blocks；exact 分支对这些 token 做完整 online-softmax，remainder 用块 centroid / value sum 提供零阶 contribution。论文版还计算与块内 `(K-mean(K))` 和 V 相关的摘要，提供一阶修正。

本次 `hyd` 源码分为 exact、zero-order remainder、first-order correction 三段。`fused_chunk_reduce` 同时生成 qc/kc/vc 和 K×V 统计，并在 host wrapper 汇总 h；最终校正在分子施加并与 remainder 的分母质量协调。这不等于“对每个未选块都独立跑完整 Taylor 展开”，也不能把简化0阶版本当作论文高阶版本。

summary 路由本身在该实现中物化 block×block score 和 top-k indices；与 Sol-Attn on-the-fly threshold 在系统成本上不同。`sink_idx` 示例只保护首/末**一个块**，并不自动保护全部有效文本 span。

## 复杂度的正确解读

若 block size B、exact density ρ 固定，exact token 工作仍约 `O(ρ N² D)`；block-summary 路由也有 `O((N/B)²D)` 项。只有预算随 N 降低或结构假设不同，才可讨论严格的次二次增长。不要把 README 的 sub-quadratic 描述当作任意固定 density 设置的数学结论。

一阶 K×V summary 额外占 O((N/B)D²) 临时量及 reduction；对较大 head dim、短序列，它可能比0阶更贵。实现压缩/汇总方式影响真实峰值，必须绑定版本。

## Apple 迁移建议

先实现0阶 CPU oracle与 FP32/FP64 witness：验证 all-exact 退化、constant-key block 精确性、部分尾块质量 n、exact/approx 不重叠。再在 Metal 以连续 tiles、simdgroup matrix 和 FP32 online softmax 实现；把 threshold/top-k route 与 summary 的成本独立计时。

PISA0 对应 MLX 参考容易写，但若通过多次 gather+SDPA 加 approximate tensor 再平均，往往破坏 softmax 归一化，不能称正确实现。必须组合**未归一化的分子/分母或正确 log-sum-exp**。

比较组：同 QKV 的 dense、all-exact、keep/drop、0阶补偿、论文版补偿、Sol-like threshold。匹配 actual exact density 之外，还要固定 sink/neighbor 保护范围；同一个 density 参数不等于相同算量。

高 logit 方差、远端小物体、长距离动作与文字是压力场景。质量不过时不要只扩大“近似可接受”的含义；先检查路由是否漏关键块及补偿误差来源。

论文：[PISA, arXiv:2602.01077](https://arxiv.org/abs/2602.01077)。
