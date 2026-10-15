# 技术实现卡模板

[总目录](../README.md) · [SOP](../guides/sop.md) · [可执行目录](../labs/catalog.json)

## 摘要

名称 / 优化对象 / 适用拓扑 / 预期节省的工作：待填写。

状态分别填写：文献启发、CPUreference、模拟、actualdevice、realblock、realrequest、产品资格；不要只写一个“已验证”。来源URL/revision/具体文件、作者/许可证、当前实现是否独立重写：待填写。

## 数学与数据合同

shape、logical→physical mapping、code/group/scales/offsets、dtype/accumulator/casts、bias/residual、mask/position/temporal、LoRA顺序。列一个最小公式和一个带stride/尾形状的数值例子。

## 实现

- 代码入口与函数：待填写。
- 在模型哪里接：capabilities、phase、reads/writes、step/shape/device policy。
- producer/consumer与owner：谁保活、何时ready、何时允许reuse、失败/timeout如何drain。
- 有界资源：banks/slots/pipelines、临时文件、cache key与失效。
- 冷开销与steadytraffic：compile/verify/load/stage/predict/restore/join、实际bytes/calls。

## 实验矩阵

| arm | 实际做什么 | 和谁比 |
| --- | --- | --- |
| reference | 原始数学/precision/source | 数值与质量基准 |
| optimized GPU | 当前最强同任务GPU | ANE或新kernel性能分母 |
| boundary ablation | 保留边界但不启候选 | 找新增materialize/fence税 |
| candidate | 单一新技巧 | 完整operator/block/request |
| negative controls | wrongpitch/lateerror/stalegeneration等 | 确认检验能抓错 |

写可复制CLI、新输出路径、固定输入/seed/noise与sourcehash、ABBA/warmup/samples、shape/dtype/tail、数值/媒体/内存/生命周期门；硬件工具不满足时的unsupported也要记录。

## 结果与解释

原始记录/实际binary、执行计数、样本/波动、质量和内存分别列。写清模型近似 vs storage优化、CPU模拟 vs actualdevice、hostspan vs kerneltime、组件 vs E2E。

## 失败条件与重开条件

历史失败出现在哪个硬件/shape/recipe/boundary/consumer？哪些证据支持原因，哪些只是猜测？

`revisit_when`：当布局、SDK/芯片、sourceencoding、weight residency、bucket/分段、共享IO、LoRA成本、qualitybudget或下游consumer变化时，怎样最小化重测？不要把局部负结果写成永久禁止。

## 新模型迁移练习

指出应替换的H/F/heads、packing、adapter、capability和质量数据；列哪些代码可直接复用、哪些必须重写；提出一个有明确否决条件的下一次实验。
