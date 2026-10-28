# SpargeAttention2与训练型稀疏：混合mask、蒸馏目标和代码可得性

<!-- knowledge
id: technique-algorithms-methods-spargeattention2
type: method
stage: dit
operator: attention, quantization
backend: cpu-reference, mlx, metal
device: cpu, gpu
symptom: precision-loss, measurement
semantics: requires-training, approximate, contract-only
representation: fp32
boundary: weight-runtime, producer-consumer
api: reference-only
-->

[算法入口](../README.md) · [Sparge旧入口](../case-studies/sparge-call-contracts.md) · [SLA实际源码](sparse-linear-attention.md) · [前沿记录](research-frontier.md) · [实验](../../examples/README.md)

2026-10-16第七轮。本页分开**选读论文**、可取得的固定源码与本库原创数学分析；没有执行SpargeAttention2或SLA2模型微调，没有Mac速度/画质复现。版本/公开refs及未验证字段见机器记录。

## 论文所提策略，不是已有Sage2入口的别名

[SpargeAttention2 v1，2602.13515](https://arxiv.org/html/2602.13515v1)是训练型方案：从高概率块的Top-k与Top-p取并集，配合attention和velocity两层蒸馏；不能只按固定块数或只按CDF阈值理解。它与固定Sparge仓库函数名里的`spas_sage2`（SageAttention2基础）不是自动等价实现。

本轮公开Git refs返回的SpargeAttn main仍为[S14的固定实现](../sources.md#s14)，另有add_sage2++/parallel_compile/return_lse。未定位并核验论文2的训练代码，不声称作者在其它地址、私有分支、supplement或未来也没有代码；未确认的同名仓库不纳入实现账本。

本轮没有重新发布论文在其它GPU上的最大倍率，也不从“训练后稀疏”推导普通base checkpoint直接调到相同density就有同样质量。

## 并集为什么不是一个固定计算预算

若两种选择基于同一排序，并集的块数相当于两种prefix长度的较大者。Top-p保证至少覆盖指定的**proxy**质量，Top-k给最小块数；合并后某些行会多选，而不是强行每行都维持k个。

本库原创标准库reference明确采用“包含跨过阈值的最小prefix”，并非作者CUDA代码：k=2、p=.8时，均匀四块选4；[.91,.03,.03,.03]选2。它只说明同样超参可以有不同成本，不能把此reference叫作论文算法复现。

以前[Sparge v1 CDF接缝](../case-studies/sparge-call-contracts.md)有threshold=.9却保留.8的例子，涉及index/count解释；不能未经源码验证就将该行为或本库的prefix修正套到论文2。每个实现都要重新核对等号、最少块、valid tail与sink。

## 更集中不意味着每行输出误差必然更小

这是本库独立分析。对真实概率p、保留集合S、mass τ>0，记保留和遗漏V的归一化均值为μS、μD，则

`dense − normalized_keep = (1−τ)(μD−μS)`。

因此有基于V范数的上界；但τ之外还依赖V及遗漏部分的分布。固定V=[0,1,−1]且S={0}：p=[.5,.25,.25]保留.5而误差0；p=[.8,.19,.01]保留.8，绝对误差却为.18。更小的上界和更小的实际误差不是同一命题。

这不否定学习attention concentration，而是解释为什么还要检查value-sensitive输出与velocity。它也不是作者论文理论或训练损失的逐式复现。`trainable-sparse-contracts`同时验证精确误差分解、范数界和这个反例。

## 蒸馏端必须与部署端对齐

以下是迁移建议，不是声称已有Apple训练实现：teacher/student应使用相同noisy latent、时间/sigma、文本/参考、CFG分支、mask与输出参数化。先验证关闭压缩的对照，再区分attention局部误差和完整velocity漂移。

只对原扩散target继续训练、只对局部attention拟合、或对完整teacher velocity蒸馏，是不同目标；训练损失下降不授予holdout媒体资格。checkpoint、routing、归一化、feature map、低精度/QAT规则、激活/scale dtype都属于部署recipe，不能把同样bit数当格式兼容证明。

Apple上可用普通GEMM构建reference和autodiff；要有真正稀疏/低精度收益仍需相应Metal后端。量化cast/整数packing本身不可微时，需要已验证的梯度策略，不能把快forward自然当成QAT；训练激活/optimizer/teacher内存也不能按推理weight bytes估算。

## SLA2：确认论文存在，不夸大所查公开类

[SLA2 v1，2602.12675](https://arxiv.org/abs/2602.12675v1)作者摘要强调learnable routing与QAT。本轮仅核对摘要并对照README citation，不发布其完整训练recipe或模型结果。

实际取得的[SLA main/SLA-1](sparse-linear-attention.md)却足以证明另一个迁移风险：两份同名类、同shape参数，linear scope分别为全局和补集。不能因main README加了SLA2 citation，便将固定φ、pooled top-k的公开类标成已验证的论文2 learned-router/QAT。

因此本轮采用S40/S41登记真实源码、CPU接缝与明确替身，同时把论文2的未验证路径留在可得性记录。代码存在、参数能加载、native backward正确、原论文复现、Apple性能及生成质量是不同证据轴。

后续优先取得作者训练实现/确切checkpoint元数据，再验证hard routing的gradient、QAT前后语义和完整consumer。没有这些证据，不按作者最大sparsity或速度给Mac默认配置。
