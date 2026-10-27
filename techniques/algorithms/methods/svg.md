# Sparse VideoGen / SVG2 / SVG-EAR：从时空 head 到语义重排与误差补偿

<!-- knowledge
id: technique-algorithms-methods-svg
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

[算法入口](../README.md) · [稀疏分类](model-sparsity.md) · [PISA](piecewise-attention.md)

来源：同一个作者仓库 `svg-project/Sparse-VideoGen` [S13](../sources.md#s13)，Sana 的 SAP 纯算法入口 [S01](../sources.md#s01)。独立 `Sparse-VideoGen2` 仓库地址本次访问失败，后查作者 README 确认 SVG2/EAR 已在 SVG 仓库，不保留错误地址作实现证据。

## SVG1：在线识别时空模式

作者观察到部分 heads 偏空间/邻帧、部分偏跨帧固定空间区域。采样少量 query rows 计算 dense reference，再比较候选 mask 的输出误差，为 head 选模式；之后通过时空布局重排与专门 sparse kernel 执行。

本次读 `svg/models/wan/attention.py::sample_mse`，其 sampled dense QK/softmax/PV 本身有成本；self-attention、cross-attention 与 I2V image 分支分开处理。不能只拿 sparse attention core 耗时而忽略 profiling，也不能把一次 head 分类永远复用到其他分辨率/prompt。

## SVG2：Semantic-Aware Permutation（SAP）

几何块混有不同物体时，top-p token 集合很难成为高效 block。SVG2 对 Q/K 聚类，以 centroid 路由和 cluster-size 加权 softmax 决定要计算哪些簇，随后按 label 排序重排 tokens、运行变长 block-sparse attention，再 inverse permutation。

Sana `sparse_attention_policies.py` 提供清楚的五步链：k-means → dynamic map → permutation → sparse attention → inverse permutation。纯 Torch selector 是算法证据，不是其 Apple 高效 kernel 已经实现的证明。

cluster 的 token 数不同，路由的 softmax 质量必须按 key cluster size 加权；只对 centroid 做普通 softmax 会把小簇/大簇当成同等质量。inverse 排序必须恢复每个 query 的原始 slot；padding、empty cluster、GQA 与 CFG batch 不能乱。

## SVG-EAR：被跳过的块也有信息

作者仓库还纳入 Error-aware Routing 与参数无关 centroid compensation。它把“哪块容易近似出错”纳入路由，而不只看重要性分数，同时补偿没有精算的块。与 PISA/Sol 同属 exact-or-approximate 思路，但具体 error proxy、summary、kernel 不同，不当作同一算法的别名。

本轮对 EAR 只核对作者 README 的机制与论文入口，没有逐行审计独立 EAR kernel；不能据此声称已完整理解/复现其实现。SVG2 则读到 Wan attention、k-means/dynamic-map 与 permutation 接缝。

## Apple 路径的优劣

空间/时间固定模式最易实现，静态 permutation 可 amortize；语义 SAP 更灵活，但 GPU k-means、top-p/sort、gather/scatter 和变长稀疏 block 都要重做 Metal/MLX 路径。把 sklearn k-means 放 CPU、每层同步回来通常不是公平高性能实现。

可先试少数 centroid、重用上一步中心作初始化、少量 Lloyd 更新，再做 stale routing 上限与定期重新聚类。**这些是本库迁移建议**，不是已测 Apple 最优配方。预算需按真实 N/head dim 和 summary 误差测，不能机械复制 Sana 的400/1000中心默认值。

必须对比：几何 local、均值 block proxy、SAP full routing、重用 cluster/route；总时间包含初始聚类与每次 update、permutation、sparse、inverse。高 motion、遮挡、物体出入与镜头切换是必要 holdout。

论文：[SVG1 2502.01776](https://arxiv.org/abs/2502.01776)、[SVG2 2505.18875](https://arxiv.org/abs/2505.18875)、[SVG-EAR 2603.08982](https://arxiv.org/abs/2603.08982)。
