# ToMeSD / token pruning：把 token 数真正减下来

配套：[机制 witness 与证据合同实验](../labs/algorithm-research.md)。这是支持性小实验，不是本方法的完整模型复现。

[算法入口](README.md) · [模型 sparsity](model-sparsity.md) · [Sana 技术组合](sana-sol-engine.md)

来源：ToMeSD `merge.py/patch.py` [S16](sources.md#s16)，Sana `token_prune.py` [S01](sources.md#s01)。分类：`approximate-train-free`；merge、prune、stale-token update 不是同一方法。

## ToMeSD 的匹配与恢复

随机2D网格从每个区域选 destination，其余为 source；按 normalized feature 的 cosine similarity 找 source→destination 匹配，挑最高相似的 r 对做 `scatter_reduce(mean)`，在缩短序列上执行 attention，再把代表输出复制回源位置，接原始 residual。

unmerge 恢复的是输出 shape / token slot，**不恢复 merge 前的独立信息**。相似 feature 不代表位置一样；对高频文字、小物体和人物边界，两个 token 合并也可能损失细节。

原 patch 默认只 merge self-attention，cross-attention/MLP 默认关，并警告扩展到 MLP 风险较高。因此不能用 token 从 N 到 rN 推导原 ToMeSD 的所有 FFN 自动省 r 倍。

## pruning 与时间重用

Sana 的接口可 score token、top-k gather，只在较短序列上执行 blocks，再 scatter 回 full N；未更新的 token 可用前一步 hidden 或零补偿。另一类 convergence/staleness selector 依据变化挑更新簇，并轮换避免长期饿死。shape-stable mask 仅维持 N，不保证底层 dense GEMM 少算。

ToMe 保留代表并合并，pruning 丢更新，stale update 延后更新：它们保留的物理量不同，不能共用同一质量解释。Sana 的 CAT controller 也标明不是公共 CAT 的 graph pooling / SD3 joint-attention 完整复现。

## Apple 特点

ToMeSD 已有 MPS gather workaround，说明作者考虑过 MPS 算子兼容；它不构成当前 torch/macOS 的全模型测速证明。`scatter_reduce`、排序、pairwise matching、临时 indices 的成本要计。大比例 merge 在短序列上可能被选择成本吃掉。

MLX 可以用 gather/scatter、sort 和量化 GEMM；动态 token 数会改变编译 specialization/ANE shape bucket。使用有限 buckets、padding 后仍执行的算量必须报告，不能以逻辑 K 替代实际 padded K。

## positional mass / RoPE 的注意点

原 ToMeSD 并不是通用的“所有合并都 mass-aware attention”。推广到不同 cluster size 的 KV summary 时，重复 token mass 可能要用 log(count) 或加权分母处理；这是新算法变体，需要单独验证。`merge(mean)` 后的普通 attention 不天然与未合并的结果一致。

对3D RoPE/joint多模态，不随意平均位置或重用 compressed slot 的位置编号。先限定只减视觉 tokens，保护文字、参考、音频和 mask；gather 的 K/V 与位置信息一起保持身份，恢复后再接原 residual。

## 最小实验

ratio=0 byte/numeric no-op，保留索引唯一性、inverse mapping、non-divisible image edge、CFG 两路相同匹配规则和随机 generator 独立 seed。然后 attention-only vs attention+MLP、随机2D vs 固定网格、merge vs prune 做消融。

对视频跨帧匹配容易混淆运动对应与空间相似；先限制同帧/局部时窗，再用身份、遮挡和运动 holdout 评估。zero-fill 是强近似负控，不是推荐的高质量默认补偿。

论文：[Token Merging for Fast Stable Diffusion, arXiv:2303.17604](https://arxiv.org/abs/2303.17604)。
