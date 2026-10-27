# VMoBA：路由到块、变长 attention 与正确的 LSE 合并

<!-- knowledge
id: technique-algorithms-methods-vmoba
type: method
stage: dit
operator: attention, layout
backend: cpu-reference, mlx, metal
device: cpu, gpu
symptom: launch-bound, wrong-layout
semantics: approximate, requires-training
representation: bf16, fp32
boundary: producer-consumer, request
api: reference-only
-->

[算法入口](../README.md) · [结构化稀疏](structured-sparsity.md) · [模型地图](model-sparsity.md) · [实验](../../examples/README.md)

来源：作者单文件 `src/vmoba.py` [S35](../sources.md#s35)。作者定位于视频diffusion的稀疏attention训练；本轮不证明任意dense预训练模型可train-free替换且保真。下面拆解可迁移的数据流，不称为现成Apple后端。

## 分块和路由

作者按层轮换1D/2D/3D block partition，`process_moba_input/output`成对重排与恢复。K均值作chunk gate representation，与每个query/head打分，选择own chunk和其他块；batch边界与部分尾块独立处理。

除了query/head独立top-k，还可按head或全局分配预算；同一个平均budget不代表每个query都拥有相同计算量。threshold选择器在此快照采用**min-max归一化后的gate分数**，尽管注释写LSE-style，它不是softmax概率，不能解释为“保留90%真实attention质量”。

## 为什么分两条 attention 路径

own chunk使用noncausal varlen attention；其他块将选中的query/head重排到各chunk expert，调用另一组varlen attention。最后把多条输出scatter回原query位置。

每个分支输出已经归一化，不能普通平均或直接相加。源码保留各自log-sum-exp，按总归一化质量加权：

`L=logsumexp(L_1,...,L_m)`，`O=sum(exp(L_i-L)*O_i)`。

前提是各分支KV集合互不重复；own chunk在第二条路由中被排除。重复own chunk会让其softmax质量被计两次。部分尾块的valid length同样属于合同。

## 一处反直觉 fallback

源码先从route中去掉own chunk；若其余gate全部为空，则直接调用**full-sequence dense varlen attention**。因此把top-k降到仅own chunk，未必得到极稀疏的局部执行，反而可能全部dense fallback。必须记录实际执行臂，不能用请求的k估算速度。

`calc_chunks`的LRU按tensor对象参数缓存，也不等于按cu_seqlens内容缓存；如调用者原地改对象内容，需要失效合同。这里是迁移审查项，未演示作者pipeline真的如此修改。

## Apple 上借什么，不借什么

优先借非因果时空partition、local/global预算、partial-length与LSE合并；不要照搬依赖CUDA FlashAttention私有forward/backward签名的执行器。MLX若所用SDPA不返回LSE，不能仅拿各块O拼起来；需要合并未归一化统计的自写Metal，或对选中KV取一次联合SDPA。

query gather、按expert重排、variable lengths、多个SDPA与index_add可能很贵。作者README在其CUDA实现上指出约33K tokens之后才明显加速，这个阈值不是Mac阈值，但提醒长序列成本收益才可能盖过调度。

最低验证：all-blocks对dense、own/remote无重叠、空remote实际fallback、多个batch/tail、重排逆映射和LSE-merge负控。训练过的路由/结构与未训练的插入实验分开做生成质量验收。

论文：[VMoBA 2506.23858](https://arxiv.org/abs/2506.23858)。
