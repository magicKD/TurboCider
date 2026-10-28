# ANE 优化实验设计：从候选方法到可信结论

<!-- knowledge
id: technique-ane-experiments-20261016
type: method
operator: quantization
backend: coreml, private-ane, cpu-reference
device: cpu, ane
symptom: measurement, precision-loss
semantics: contract-only
boundary: producer-consumer
api: reference-only
-->

[ANE 学习目录](README.md) · [首轮完整实验记录](../MAINTENANCE.md) · [证据归档](../MAINTENANCE.md) · [负结果与回归](negative-results.md)

本页保留原入口，内容改为可复用的实验方法。2026-10-16 的环境、逐项数字、失败和运行过程已移到研究归档；历史结果不作为当前机器或新模型的性能承诺。

## 先选择一个可以证伪的优化假设

| 候选方法 | 最小对照 | 必须计入的成本 | 继续采用的门槛 |
| --- | --- | --- | --- |
| [Linear→Conv / BC1S](layout-and-lowering.md) | 原表达、仅换布局、换算子表达 | pack、预测、恢复、真实消费者 | 参数/索引等价，完整输出正确，边界总成本下降 |
| [LUT / W8 / W8A8](weight-compression.md) | 同一权重的未压缩图和各压缩图 | 文件、加载、解压后驻留、完整预测 | 实际图符合意图，质量预算不变，目标指标改善 |
| [GPU+ANE 分担](hybrid-runtime-comparison.md) | 强 GPU、同图串行、合法并发 | 两支准备、同步、join、下游 | 先证明切分等价，再证明并发临界路径更短 |
| [session / state 复用](lifecycle.md) | 新建与复用；独立请求与连续请求 | first、warm、失败重试、内存峰值 | 身份与代际正确，失败不会重复提交状态 |
| [动态图与训练驻留](training.md) | 同等 K 步、相同优化器与状态完成点 | 更新、同步、checkpoint、参数搬运 | 完整梯度及 W/m/v/t 正确，不仅是 loss finite |

## 按这个顺序验证

1. 固定输入、权重、源码 commit、SoC/OS/SDK、shape、dtype 和实际消费者；先写下质量门槛，再测候选。
2. 在 CPU 上用独立数学或数据布局参考核对完整输出；保留原实现、有限值错误、尾块和异常范围对照。
3. 把 compile、load、first、warm 分开；正式计时关闭诊断 observer。按相同请求边界对比，不只摘取 predict 或某个 kernel。
4. 保留同轮配对、顺序、warmup、失败与缺失值；先筛数值资格，再计算速度比。比率中位数与中位数之比需分别标注。
5. 将组件结果迁移到真实模型/任务，重新验收质量、端到端时延及峰值内存；计划首选 ANE 不代替物理执行证据。

具体计时口径见 [profiling](profiling.md)，可复用失败见 [negative-results](negative-results.md)。更强 GPU 分母、GPU-resident 输入或消费者变化，都需要重新评估旧结果。

## 结果怎样保存才便于下一次学习

- 专题页写“机制、适用条件、步骤、反例、来源链接”；不按进程启动顺序记流水账。
- [来源阅读](source-audit.md)与[源码索引](sources.md)保留固定版本入口和阅读重点，不附机器清单。
- 逐批过程、原始数据、模型、trace 和编译产物不放进知识库；材料范围见[维护规则](../MAINTENANCE.md)。
- 历史校验只是当时的快照；目录整理、链接检查或 CPU 回放通过，不会新增 ANE 或产品加速资格。

## 小型阅读样本

[量化参考](../examples/reference/quantization.py)帮助理解格式与数值配方，[所有权模型](../examples/reference/contracts.py)帮助理解 owner 和完成边界。它们是教学代码，不是 ANE 启动器或原实验复算入口。
