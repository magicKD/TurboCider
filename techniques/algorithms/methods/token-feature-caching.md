# ToCa / DuCa：按 token 更新特征，与“剪掉 token”不同

<!-- knowledge
id: technique-algorithms-methods-token-feature-caching
type: method
stage: dit
operator: sampling, attention, gemm
backend: cpu-reference, mlx, metal
device: cpu, gpu
symptom: stale-cache, memory-bound
semantics: approximate
representation: bf16, fp32
boundary: producer-consumer, request
api: reference-only
-->

[算法入口](../README.md) · [ToMe与pruning](token-reduction.md) · [配套实验](../../examples/README.md)

第二轮源码：ToCa [S30](../sources.md#s30)、DuCa [S31](../sources.md#s31)。本页重点核对各自 DiT 示例的实际 forward，不宣称审计所有 FLUX/OpenSora adapter；未运行其模型。属于 `approximate-train-free`。

## ToCa 的真正计算接缝

token-wise feature caching 不必把整个模型序列永久从N缩到K。选中的 fresh tokens 做新计算，旧 tokens 留在 full-size 输出缓存；随后按原位置接回完整 residual stream。它与ToMe的代表合并、全栈token pruning是三种不同函数近似。

ToCa 的 `DiTBlock.forward` 中，full 步算全部 attention/MLP，缓存的是 **gate之前的分支输出**；ToCa步重新计算当前AdaLN/gates，直接重用整个attention输出，只对fresh MLP tokens做norm/modulation/MLP，再scatter更新MLP缓存。当前gate乘旧分支输出，不等于缓存最终hidden或gate之后的值。

这里的主要局部节省是MLP行数，以及跨步attention重用；**并非每个ToCa步只给K个tokens执行全局attention**。只看到 `cache_cutfresh` 支持 `module='attn'`，不能证明该分支在最终forward里启用。

## 选择器为什么可能很贵

`attn_score` 汇总历史attention map中每个key收到的注意力，再归一化；staleness计数为长时间未更新的tokens加分；局部网格最大值bonus减少更新只集中在一个区域的问题。fresh ratio还可随layer/step/module改变。

但所读DiT attention在 `cache_type='attention'` 时显式计算softmax矩阵并按head平均，绕开只返回输出的fused SDPA。N=8192的一份FP32 N×N矩阵就约256MiB，尚未计heads/batch/其他层。为了指导少量FFN更新而失去fused SDPA，可能让优化分母变弱，Apple统一内存下尤其要计。

`local_selection_with_bonus` 还假设方形2D网格：仅断言token数可被grid面积整除并不足以支持任意矩形/视频/带文字前缀的N。可迁移的是“覆盖约束与stale预算”，不是这个reshape可原封不动套到Wan或Qwen。

## DuCa：两种更新强度，源码版本要看具体分支

论文/README提出dual feature caching并强调兼容FlashAttention；然而本次DiT示例仍保留attention-map选择器：`cal_type` 产生 `full`、`ToCa`、`aggressive`。ToCa分支局部更新MLP；该快照中 `aggressive` 未命中专用forward分支，而进入末尾fallback，在最后一层恢复保存的`noise` hidden。不能只按名称解释成“两个独立完整cache”。

同一文件还保留FORA、Residual等实验分支，部分开关在forward内设为False；存在代码不等于正常配方启用。`cache_type`、具体模型adapter和最终配置一起决定是否真的使用FlashAttention。不能将README的全项目支持声明，当作所有DiT配置都无attention-map开销的证据。

## Apple 移植建议

第一版仅在逐token独立的FFN上做fresh-row gather →原QMM/MLP→scatter，保留attention dense，使用便宜的norm/change/staleness proxy。之后再独立探索attention-output重用，避免同时改选择器、precision、branch边界。

MLX/Metal可实现这些操作，但K变小会改变GEMM利用率和量化kernel选择；ANE需有限K buckets，padding后的实际行数必须计。读取/写回完整缓存、索引排序、LoRA以及当前step modulation都不能从计时中排除。

保护文字/参考/音频，维护每个token的age，允许周期性全刷新。双路CFG必须明确采用共同还是各自indices；原示例某些随机选择直接复制半个batch，不能推广到任意batch或多分支。

## 最小对照

fresh_ratio=1必须与同路full比较；ratio=0仍可能执行modulation/residual，不是零成本；检查untouched token确实取旧值、fresh callback实际只处理K行、scatter恢复原位置、换条件/shape/step清状态。将full SDPA基线与“为得到map而禁用fused”的基线分开，才能解释净收益。

论文：[ToCa 2410.05317](https://arxiv.org/abs/2410.05317)、[DuCa 2412.18911](https://arxiv.org/abs/2412.18911)。本页不把论文中的近无损任务指标改称逐元素无损。
