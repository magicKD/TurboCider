# 主题索引：优化方法、实现案例与学习路径

<!-- knowledge
id: technique-gpu-ane-report-index
type: navigation
-->

[方法导航](README.md) · [14种技巧总览](optimization-playbook.md) · [全部案例](case-studies/README.md)

## 优化方法

| 主题 | 从哪里开始 | 迁移时重点检查 |
| --- | --- | --- |
| GEMM/GEMV与量化 | [shape选核与解码复用](gpu-gemm-quantization.md) | M/K/N、packing、group、stride/tail、完整FFN |
| Attention数据流 | [分块、融合与KV](gpu-attention-runtime.md) | mask/GQA、page、全空行、完整输出 |
| 融合与提交复用 | [融合、编译和replay](fusion-and-replay.md) | 中间精度、强GPU分母、cache与owner |
| ANE封闭图 | [图表达、shape与量化](ane-graphs-and-quantization.md) | 完整子图、图转换/交接、实际回退 |
| 合法异构切分 | [row/paired-F/branch/stage](partition-and-scheduling.md) | 完整代数、bias/residual/LoRA和依赖 |
| 负载平衡 | [收益开关与份额控制](load-balancing.md) | 并发成本、接缝、probe和share=0 |
| 共享内存 | [布局、完成与所有权](memory-and-handoff.md) | actual backing/stride、generation和最后读者 |
| 驻留与预取 | [专家bank与阶段工作集](residency-and-prefetch.md) | packed格式、slot有效性、I/O与峰值 |
| 阶段流水 | [encoder/decoder与跨请求流水](stage-pipelines.md) | 实际tensor/token、TTFT/完整响应与背压 |
| 验收与回退 | [数值、计时和状态事务](validation-and-fallback.md) | 独立参考、负控、完整请求和恢复 |

## 选型与实施

- [硬件与TensorOps](hardware-and-tensorops.md)：设备、API、NAX与ANE分层。
- [推理引擎地图](inference-engines.md)：按可借鉴机制选择读本，不是速度排名。
- [协同项目与模型阶段](hybrid-projects-and-stages.md)：结合真实模型图判断并行机会。
- [私有ANE与图搜索](ane-private-and-search.md)：研究方法与公共部署边界。
- [优化PR读本](optimization-prs.md)：固定改动、治理状态与性能资格分开。
- [实验与反例](experiments.md)、[离线学习](examples/README.md)：先验证方法，再上真实设备。
- [来源与覆盖](sources-and-coverage.md)、[证据保留策略](../MAINTENANCE.md)：需要核对数据时深入。

历史批次只在[归档](../MAINTENANCE.md)查找；旧105项定位表保存在[历史索引](../MAINTENANCE.md)。新文档按问题和机制命名，不新增round作为教学标题。
