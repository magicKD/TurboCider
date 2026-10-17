# 前沿候选与代码可得性：论文、同名仓库和真实实现分开

<!-- knowledge
id: technique-algorithms-methods-research-frontier
type: method
stage: dit
operator: attention, sampling
backend: cpu-reference, metal, mlx
device: cpu, gpu
symptom: stale-cache, measurement
semantics: approximate, contract-only
boundary: producer-consumer
api: reference-only
-->

[算法入口](../README.md) · [路由复用实测](../case-studies/sparse-route-reuse.md) · [DiCache实现](dicache.md) · [来源账本](../sources.md) · [实验](../../examples/README.md)

2026-10-16第四轮重新检索作者论文入口，并核查新仓库。以下是新增研究线索，不是“都开源/都跑通”的支持列表。机器记录保留所读版本、范围及候选tree身份。

## HEART / 检索别名HASTE

2026-05-14的v1题为HASTE；2026-08-09的v2题为HEART: Exploiting Head Heterogeneity in Sparse Attention for Video Diffusion。本次选读的是同一arXiv身份[2605.14513v2](https://arxiv.org/html/2605.14513v2)，并核对[v1摘要](https://arxiv.org/abs/2605.14513v1)。按同一研究版本链追踪，不当作两个独立复现结果。

两项互补机制：TMR在线比较每head的mean(Q/K)与最近refresh anchor的L1；reuse时anchor不更新，并用层级gate摊薄零碎refresh的launch。EBC离线校准head阈值；孤立head误差求和只是局部surrogate，不是联合误差分解。本文未取得并执行作者实现，论文倍率不作Apple结果。

可迁移研究问题：给每head/layer选择refresh周期，约束质量预算时同时计当前attention与搜索/计划开销。周期化方案便于Apple静态调度，但分辨率、prompt、guidance、权重或sigma日程改变都需要重新校准。上述是本库设计推导，不表示已实现原论文优化器；[本轮路由实验](../case-studies/sparse-route-reuse.md)只是原创单行top-k迁移，不能冒充HEART。

## SVD-Cache：与SVDQuant、DMD不是同一对象

[Forecast the Principal, Stabilize the Residual，2601.07396v1](https://arxiv.org/abs/2601.07396v1)的作者摘要介绍feature主子空间EMA预测与残差子空间复用。这里不是SVDQuant权重低比特outlier，也不是DMD对演化算子的eig拟合。本轮只核对摘要，未完整审读方法，不发布其质量或速度复现。

该版本摘要称代码在supplementary material，并计划发布GitHub；本轮没有取得该补充材料，不能将下面的同名候选替代作者代码。

本轮取得同名候选[BlackMaple1203/SVDCache固定快照](https://github.com/BlackMaple1203/SVDCache/tree/ad22fcebd2501b2cc5c2a140c33f0d974fa6cdf0)，递归tree仅有LICENSE，没有实现/README；**未确认候选与论文作者的关联**。这只能否定“该快照提供了代码”的说法，不能证明作者在别处/之后也没开源，更不能拿同名license授权复制论文或另一仓库。

Apple后续研究重点应是子空间建立/刷新成本、history与basis驻留、投影/重建及缓存残差如何共用buffer。若只比较低秩乘法而忽略SVD/传输，不能判断完整收益。优先获得作者实现与确切状态合同；若自行研究，应明确标作独立原型，并保留无预测、仅残差复用和full-refresh对照。

## 第24轮补充：Delta-DiT仍按论文级候选处理

2026-10-17选读[Delta-DiT，2406.01125v1](https://arxiv.org/html/2406.01125v1)的方法/阶段示意与可得性声明：缓存block输入输出之差，让当前输入走identity路径，以减轻直接重用旧输出的失配；采样阶段再切换前/后block的缓存位置。它与TeaCache的差值阈值不是同一控制器。

所读论文版本仍写代码将公开，本轮搜索未取得可核验的作者实现。这里是阅读/检索范围的限制，不是证明作者在所有位置或之后均未开源；不能用同名仓库或自写模型代称复现。下一步先取得作者固定实现，再核对block边界/残差坐标/阶段切换与强对照。本轮转向确有两份作者脚本的[VDE](vde-input-adaptive-velocity.md)，其真实FLUX CPU证据和Qwen接口拒绝单独记录，不为Delta-DiT转授执行资格。

## 早期轮次结论及重开条件

DiCache有可读实现，已运行的是其FLUX原forward配微型blocks的CPU接缝；HEART/SVD-Cache是选读论文线索，没有因此授予源码执行资格。外部论文数据、所读源码、本库原创分析和实际设备结果始终分层；不同项目的“无损”名称、少量示例或同名仓库不构成媒体质量门槛。

重开条件：作者仓库/代码发布、论文版本与命名变化、适用模型/数据分布、Apple可用数值路径或完整生成质量预算改变。继续检索不是用新增候选数量代替方法审计。
