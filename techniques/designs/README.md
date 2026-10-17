# 吸收后的设计样本：不用打开原仓库

<!-- knowledge
id: technique-designs-readme
type: navigation
-->

[总目录](../README.md) · [模型阶段](../models/stage-map.md) · [小型代码](../examples/README.md) · [固定来源](../sources/README.md)

每篇解释问题、数据布局、算法、所有权、失败与迁移接缝。保留思想和小型原创样本，不镜像原项目运行时，也不附实验数据。

| 设计 | 学什么 | 小型参考 |
| --- | --- | --- |
| [GPU 数据流](gpu-dataflow.md) | 算术强度、pack/consumer、融合 | [MSL](../examples/metal/kernels.metal) · [算子](../examples/reference/operators.py) |
| [owned command graph](owned-graphs.md) | 参数快照、owner 与完成 | [OwnedGraph / CpuTicket](../examples/reference/contracts.py) |
| [GPU/ANE FFN](hybrid-ffn.md) | paired 切分、非线性、双 bank 与失败回退 | [切分与 LoRA](../examples/split_ffn_reference.py) · [DualBank](../examples/reference/contracts.py) |
| [不可变权重与 typed refill](packed-streaming.md) | 对齐、read-only view、有界预读 | [PackedStore](../examples/reference/contracts.py) · [IO 模型](../examples/reference/systems.py) |
| [驻留与阶段协作](residency-and-phases.md) | 固定子集、反馈与 hysteresis | [FixedResidency](../examples/reference/contracts.py) |
| [技术组合与 schedule](technique-composition.md) | capability/effect、phase 与日程 | [组合](../examples/reference/policies.py) · [cache 状态](../examples/reference/caches.py) |
| [并行拓扑](parallel-topologies.md) | 合法独立分支、heads/rows/F 切分 | [拓扑例子](../examples/reference/heterogeneous.py) |
| [消费者 ABI](consumer-abi.md) | stride、segments、allocation identity | [布局例子](../examples/reference/heterogeneous.py) |
| [异构 planner](heterogeneous-planner.md) | deadline、资源与精度岛 | [规划模型](../examples/reference/heterogeneous.py) |

这些样本用于理解，不声称恢复任何历史实验或真实设备行为。项目思想来源见[阅读记录](../references/gpu-ane-projects.md)，署名和实现边界见 [NOTICE](../NOTICE.md)。
