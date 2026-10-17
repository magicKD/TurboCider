# Sol-Attn诊断反例：调试route mask不能天然充当oracle

<!-- knowledge
id: technique-algorithms-case-studies-sol-debug-bounds
type: case-study
stage: dit
operator: attention, layout
backend: metal, cpu-reference
device: gpu, cpu
symptom: wrong-layout, measurement
semantics: contract-only, numerically-equivalent
representation: bf16, fp32
boundary: producer-consumer
api: reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [组件实测](sol-mps-component-study.md) · [Sol机制](../methods/sol-attention.md) · [实验合同](../../examples/README.md)

2026-10-16第三轮。问题位于固定Sana `d0c0a4685ab5dc2336d18b7213d85f13def92418`的**非生产调试shader**，不是据此发现生产forward错误。没有修改上游文件、发issue或PR。

## 从拒绝开始，不调整误差门槛

真实Qwen seed42/prefill的早期v2/v3在tau0.5 source-matched witness处拒绝，均未发布timing。v3保存：生产近似输出相对dense L2=0.341091，而按原debug mask计算的“同语义”最大行误差=0.596426。v2异常前没有保存失败臂的行误差，不能给它补上v3的值。两份原report和SHA保存在整理数据的`historical_refusals`。

旧v2/v3 harness原文件未归档；其script hash保留但不满足当前harness重放资格。它们仅证明当时记录的拒绝，不进入最终七点速度矩阵。

## 诊断分配与loader维度不一致

固定文件的`sol_route_mask_debug`设置BK64、LDK=BK+8=72，并使用`BlockLoaderT<T, BK, BD, 1, LDK, 0, 128>`，其转置K布局需要BD×LDK。

| 路径 | 分配/需要元素 | 判读 |
| --- | --- | --- |
| 原debug，D128 | BK×LDK=4608 / BD×LDK=9216 | 空间不足，不能作为可信route参考 |
| 原debug，D64 | 4608 / 4608 | 没有同样的维度失配；不证明其所有其他行为 |
| 生产tiled forward | `max(LDK*BD, BK*LDV)`共享区 | 不使用上述不足分配；不能由debug失败推断其同错 |

`sol_route_tile`沿`dim_tile*8*LDK`读取，D128会超出debug的BK×LDK区域。all-exact的保护逻辑会把route设为true，因此all-exact结果正确也不能证明原debug的稀疏route可信。

原始源码定位：[metal.py固定行](https://github.com/NVlabs/Sana/blob/d0c0a4685ab5dc2336d18b7213d85f13def92418/techniques/sparse_backends/sol_attn/mps/metal.py#L217-L261)；来源文件digest见[S01](../sources.md#s01)。

## 只修正诊断参考库

probe的显式`--repair-debug-bounds`仅将一次出现的`threadgroup T keys[BK * LDK];`替换为`threadgroup T keys[BD * LDK];`，单独编译非计时debug library。不修改磁盘、production body或原计时library；report保留修正前后shader字符串SHA和声明。该库仍是来源派生参考，不冒充完全独立实现。

为避免单参考自证，另做三项检查：

1. 只记录真实forward消费的Q/K summary与threshold，要求debug重算结果逐tensor一致；记录wrapper不能改变输出。
2. 用CPU FP64独立计算selector、sink与邻居，要求debug所有route位一致。
3. 使用真实forward消费的舍入K centroid/V sum，按相同exact/approx分区计算每head8行FP64输出；原0.02 gate保持不变。

修正后七点均通过上述检查，但这仍是有限行witness，未直接捕获原forward内部每一分支。约0.34的dense近似误差没有消失：修正的是诊断参考，不是把近似算法变成dense。

## 可以迁移的设计技巧

调试器也需要shape、dtype、共享内存和索引审计。发生“大误差”时分清三个对象：dense语义、近似语义、实现的诊断路径。只有参考自身可信之后，才能对实现下结论。拒绝记录、参考修正、计时库身份和画质验收应各自保留，不能混成一个“全部通过”。
