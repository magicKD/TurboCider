# 稀疏路由复用：缓存plan，不等于缓存attention输出

<!-- knowledge
id: technique-algorithms-case-studies-sparse-route-reuse
type: case-study
stage: prefill, dit
operator: attention, layout, sampling
backend: cpu-reference, metal, mlx
device: cpu, gpu
symptom: stale-cache, precision-loss, measurement
semantics: approximate, contract-only
representation: bf16, fp32
boundary: producer-consumer
api: reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [真实稀疏性](real-qwen-sparsity.md) · [Sol组件](sol-mps-component-study.md) · [研究前沿](../methods/research-frontier.md) · [实验](../../examples/README.md)

2026-10-16第四轮。稀疏attention除了减少QK/PV，还可能花时间在聚类、路由、排序、indices构造和重排。复用旧plan旨在减少这些准备工作；**仍用本步Q/K/V重新计算所选attention**，与跨步复用旧输出不是一回事。

## 四种容易混淆的缓存对象

| 复用对象 | 当前仍须执行 | 首要失效条件 |
| --- | --- | --- |
| 上步attention output | residual/tail等消费者 | Q/K/V变动导致输出过时 |
| block mask / indices | 当前所选QK/PV、可选remainder | 坐标或重要连接变化 |
| 聚类labels/permutation | 当前centroid/路由/attention | token聚类身份变化或重排不一致 |
| 旧K/V summaries | 当前score/近似贡献 | 值本身过时；不只是mask staleness |

不能把“省了selector”报告成“整个routing都消失”，也不能用普通top-k plan的重叠率替代cluster label/permutation一致性。

## Sana通用policy的六个实际CPU观察

[S01](../sources.md#s01)的`build_sparse_route_mask`在进入reuse分支之前计算`_score_blocks`。命中后返回旧mask转bool/device，同时重建固定宽indices并计算density；`budgets`按本次density新建。这是通用纯算法helper，所检查`techniques/`与`models/`的Python文本中未找到其他调用点；不能因此声称某模型已经接入，也不穷尽动态/生成的消费者。

sparse_policy_probe.py按完整源文件hash提取原函数与真实miss selector，CPU Torch执行；仅score/selector外包计数，不改函数body。

| 输入前提 | reused | score / selector调用 | 观察 |
| --- | --- | --- | --- |
| previous mask同shape，drift=None | true | 1 / 0 | 缺失drift按0；不是实际测得变化为0 |
| drift=threshold=0.05 | true | 1 / 0 | 等号命中 |
| drift=0.051 | false | 1 / 1 | 重新选当前高分块，控制通过 |
| drift=−0.1 | true | 1 / 0 | helper未拒绝负signal；caller应定义合法域 |
| 请求density从0.25改成1 | true | 1 / 0 | 旧mask仍0.25，budget却报4；shape不足以定义plan身份 |
| 同dtype/device旧mask，caller显式zero | true | 1 / 0 | returned mask共享storage并随之变零，已构造indices不随之变 |

最后一行是故意的caller mutation，不证明默认模型会修改旧mask。它要求明确immutable/owned plan合同，而不是机械地宣称`.to()`就是snapshot copy。全部六个观察和原始mask/indices保留在。

这些是指定helper的接缝事实，不是AdaSpa、HEART或其它论文的完整实现复现。特别是此helper的`online_mask_search_reuse`名称不能自动继承任何论文速度/质量数字。

## 真实Qwen step0→step1：只隔离路由过时

qwen_route_transfer.py只读既有公开样例seed42/43、同layer0的step0 prefill与step1 steady。用视觉token ordinal对齐33个query、全部32 heads，共2112个目标行；text-query不入样本，全部1048 valid keys保留。独立FP64 dense与原BF16对照，另有math.fsum的补偿见证。

JSON中的head输入drift对Q只统计这33行，对K/V统计全部1048 keys，不是完整Q的统一范数。相对输出L2还受目标向量能量影响；要判断绝对幅度或消费者放大需回到原capsule，不能由它直接设定媒体阈值。

原创selector按单行q与64-key块mean-K的score排top-k，block0强制sink，保护24个text和40个visual keys。并非Sana的pooled query路由、Sol tau、聚类路由或HEART实现。

对照fresh mask与上一步mask，两边都重新计算**当前Q/K/V和当前K/V summaries**，分别做keep/drop与0阶remainder；不混入旧value误差。另列复用上步dense输出的误差。k=17为all-exact控制；没有GPU调用或任何速度测量。

以下均为当前dense归一化的行向量相对L2，不是RGB损失：

| seed / k=4 | fresh补偿误差 median / p95 / max | 旧mask+当前补偿 median / p95 / max | mask Jaccard median / min |
| --- | --- | --- | --- |
| 42 | 0.000910 / 0.6805 / 1.7623 | 0.000937 / 0.6805 / 1.7623 | 0.600 / 0.333 |
| 43 | 0.001587 / 0.6874 / 1.0205 | 0.001578 / 0.6892 / 1.0205 | 1.000 / 0.143 |

许多行近乎全部mass位于sink附近，所以中位数很小；seed42/k4 fresh保留mass最低0.171，旧mask最低0.099，不能把中位数约0.9998当所有heads/rows的资格。fresh重选也不是dense-mass oracle，更不是输出误差最优选择，故旧mask偶尔略好并不矛盾。

复用上一步dense输出的p95误差为0.6145/0.6722，最差3.8950/5.0539。与上表并非逐行策略优劣证明，但说明“plan可复用”不授权“输出也可复用”。k=2/4/8/17全范围摘要、每head k4全部33行scalar指标和source身份见。

跨step mask重叠高、输入drift小、保留mass高都不是独立质量保证；需要看具体行、head、模态和消费者放大。此两步早层实验不能推广到中晚层、长视频、多prompt或产品默认。

## Apple实现应真正省掉哪一段

若将命中分支提前到score/聚类/indices转换之前，才可能省掉相应工作；这属于新实现提议，不是本轮已修改或计时的版本。mask可只存压缩indices，但仍要记录有效counts、重复padding语义、block size、token order、valid tail和exact sinks；否则不同kernel会把同一plan解释成不同连接。

plan身份至少包括request/CFG/stage/layer、token spans与排列、shape/dtype、block几何、route family、density/预算、校准版本。复用时冻结generation，禁止writer修改活跃读者的mask；完整构造mask+indices后再一次提交。ANE若使用固定bucket图，也需把plan构造/交接成本纳入消费者路径。

原创标准库case `route-reuse-contracts`另有两个负控：交换两行Q保持mean(Q)完全不变，但重要连接翻转，旧plan行相对误差超过2；将重复padding indices当有效keys会改变softmax分母。这不是HEART的实机回归测试，而是“proxy相同/shape相同不足以证明语义相同”的数学见证。见route_reuse_contracts.py。

CPU可做小型refresh schedule，GPU保留summary与plan；在MLX/MPS中为每head每step回读scalar可能引入同步。必须比较强dense、每次fresh route、固定周期reuse和自适应reuse，分别统计selector/score/indices/permutation/sparse calls，而不是只统计cache hits。

完整tensor不随Wiki分发；默认离线tests只验证标量证据、hash、两seed和独立统计，不读上游/activation或提交GPU工作。
