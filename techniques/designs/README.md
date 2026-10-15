# 吸收后的设计样本：不用打开原仓库

[总目录](../README.md) · [按模型阶段选方案](../models/stage-map.md) · [来源账本](../references/absorbed-designs.json)

这里不是第三方源码文件名索引：每篇给出问题、数据布局、算法、所有权、失败路径、移植接缝和本目录内的实现。外部名称说明思想来源，不是运行依赖；实现是重新编写的教学样本，不是原项目运行时的替代品。原项目的特殊数值、布局和机器成绩不自动继承。

| 设计 | 吸收的精髓 | 本地可读实现 / 对照 |
| --- | --- | --- |
| [GPU 数据流](gpu-dataflow.md) | 按算术强度选 kernel；pack/consumer 共同设计；融合减少中间流量 | [实际 MSL](../labs/metal/kernels.metal)、[CPU 算子](../labs/operators.py) |
| [owned command graph](owned-graphs.md) | 参数按值保存、buffer 保活、少提交、完成后回收 | [OwnedGraph / CpuTicket](../labs/contracts.py)、[真实 Metal graph](../labs/metal/bench.mm) |
| [GPU/ANE FFN](hybrid-ffn.md) | intermediate 切分、完整非线性、双 W bank、校准及失败整块回退 | [切分与 LoRA](../examples/split_ffn_reference.py)、[DualBank](../labs/contracts.py) |
| [不可变权重与 typed refill](packed-streaming.md) | 对齐 pack 与 read-only view、完整来源、F16/BF16 区别、有界预读 | [PackedStore / typed_refill](../labs/contracts.py)、[实际小文件 pread](../labs/systems.py) |
| [驻留与阶段协作](residency-and-phases.md) | 固定子集、真实驻留反馈、hysteresis/ratchet、两遍预算 | [FixedResidency](../labs/contracts.py)、[interval planner](../scripts/interval_peak.py) |
| [技术组合与 schedule](technique-composition.md) | capability 与 effect 接缝、build/load 与 runtime 分离、逐 step/stage 启用 | [compose / select_share](../labs/policies.py)、[cache 状态机](../labs/caches.py) |
| [并行拓扑分类](parallel-topologies.md) | sequential vs parallel-branch、F/heads/rows不同合法边界 | [parallel/head demos](../labs/heterogeneous.py) |
| [消费者ABI](consumer-abi.md) | SEOP、paired segments、actual backing、resource identity/alias | [stride/segment/hazard demos](../labs/heterogeneous.py) |
| [异构planner与精度岛](heterogeneous-planner.md) | 可选GPU/ANE/CPU、p95 deadline、状态/参数精度、service生命周期 | [planner/precision demos](../labs/heterogeneous.py) |

gpu_ane专项来源、版本与工作树状态见[项目阅读记录](../references/gpu-ane-projects.md)，模型迁移选择看[hybrid selection](../models/hybrid-selection.md)。

运行：

```sh
python3 -B techniques/labs/run.py owned-command-graph
python3 -B techniques/labs/run.py dual-bank-handoff
python3 -B techniques/labs/run.py packed-weight-store
python3 -B techniques/labs/run.py residency-policy
python3 -B techniques/labs/run.py technique-composition
```

CPU 样本回答“设计是否自洽、错误能否被拒绝”，真实 Metal/Core ML 样本回答其小型设备边界是否执行；两者都不自动回答“完整模型能快多少”。方法与边界见 [实验室](../labs/README.md)。第三方署名和独立实现边界见 [NOTICE](../NOTICE.md)。
