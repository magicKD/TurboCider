# SageAttention / SpargeAttention：数值量化与连接稀疏要拆开测

配套：[机制 witness 与证据合同实验](../labs/algorithm-research.md)。这是支持性小实验，不是本方法的完整模型复现。

[算法入口](README.md) · [权重量化](quantization.md) · [Sol-Attn](sol-attention.md)

来源：SageAttention `quant.py`，SpargeAttention `spas_sage_attn/core.py` 的 selector/quant/kernel 调用链及 README [S14](sources.md#s14)。分类：`approximate-train-free`；二者不是同一个优化。

## SageAttention 省的是表示与矩阵运算成本

Q/K 按 block、warp 或更细粒度量化，scale 与 attention scale 协调；部分变体对 V/PV 使用低精度，并保留适当精度的 softmax、累加或修正。不同版本的 INT8/INT4/FP8 路径及硬件要求不同，不能把一个版本配方泛称所有 Sage。

源码的 key smoothing 先减去所有 keys 共享的均值 μ：`q·(k_j-μ)=q·k_j-q·μ`，每行减去相同常数，实数下 softmax 不变。**共享均值重参数化可代数等价，后续量化仍有误差**；按不同 block 随意减不同均值却不补回 row score 会改变分布。

## SpargeAttention 还省低价值块

先用 block 相似性/proxy 预测可省连接，输出 block map 或 LUT，稀疏 QK/PV 内核只执行保留块；进一步的 online softmax/PV 贡献判据还能跳过价值低的工作。高方差块不适合纯均值代理，需要保守 full 或相似性门槛。

读到的 recommended `spas_sage2_attn_meansim_topk_cuda` 把 map 构建、QK 量化、架构分派、V 转置/量化与 sparse kernel 串起来。sm90 的 BLKQ/BLKK 与其他架构不同；API 的 `return_sparsity` 也是特定 tile 分母，不是 E2E FLOPs。

## 为什么不能在 Mac 上直接复用 CUDA 倍率

原实现依赖 CUDA 自定义扩展、特定低比特 tensor-core 指令和各代不同的 accumulate 配方。Apple 的 BF16 simdgroup matrix、MLX QMM 或 ANE W8A8 能力，不证明已有相同低比特 QK/PV ISA、shape 路径或吞吐。

数学仍可借用：K smoothing、per-head/block scale、相似性保护、稀疏 LUT、敏感层 dense。第一版 Metal 可以保留 BF16 QKV 只验证 sparse 选择；第二版再单独测量化，不为保留名字强行模拟 CUDA INT4 后端。

## 四路消融

| 变体 | 连接 | 数值 | 回答的问题 |
| --- | --- | --- | --- |
| A | dense | 原 BF16/FP16 | 优化过的 SDPA 分母 |
| B | dense | quantized | 量化到底省多少、损失多少 |
| C | sparse | 原精度 | 连接减少本身的影响 |
| D | sparse | quantized | 组合的净收益与误差 |

全算但 quantized 不能用来证明 sparse；sparse BF16 也不是完成 Sage 低比特实现。包括 scale/quantize/transposes/LUT 在内测完整 attention，检查 softmax overflow、全 mask 行、causal/noncausal、head dim、GQA、有效文本与尾块。

论文：[SageAttention 2410.02367](https://arxiv.org/abs/2410.02367)、[SpargeAttention 2502.18137](https://arxiv.org/abs/2502.18137)。
