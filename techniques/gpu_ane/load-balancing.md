# GPU/ANE负载平衡：先决定是否分担，再决定分多少

<!-- knowledge
id: technique-gpu-ane-load-balancing
type: method
stage: ffn
operator: scheduling
backend: cpu-reference, coreml, mlx
device: gpu, ane
symptom: straggler, memory-bound, measurement
semantics: contract-only
boundary: gpu-ane, request
api: public, reference-only
-->

[方法导航](README.md) · [切分代数](partition-and-scheduling.md) · [共享交接](memory-and-handoff.md)

## 解决什么问题

GPU与ANE的独立吞吐不能直接给出最优份额。分片改变shape、tile利用率、权重流量和交接开销；两支还可能争用内存带宽。目标是缩短完整消费者关键路径，不是让两设备都忙。

先读[切分代数](partition-and-scheduling.md)：row split、paired-F、独立branch与多请求流水不是同一种方案。未证明依赖合法时，不使用并行成本模型。

## 机制：收益开关与份额控制分开

在真正独立、已提交的两支上，可用下式提出候选：

```text
T_split = 暴露的准备成本
        + max(并发条件下的GPU补集时间, 并发条件下的ANE时间)
        + 暴露的恢复/归约/同步成本
```

它不是物理overlap证明。必须与优化后的原始GPU完整调用比较，而非与人为拆碎、增加等待的GPU分母比较。

1. 建立原GPU、串行split、并发split三个控制，先确认完整数值。
2. 加一个保留切分接缝、但不实际offload的probe，识别额外drain/commit的固定损失。
3. 外层判断split是否真正盈利；不盈利时允许`share=0`。
4. 内层只搜索合法chunk/bucket，用平滑、迟滞和有界探测减少振荡。
5. 固定候选后，用反序窗口和未用于调参的输入确认；最后检查完整请求。

## 可借鉴实现

| 实现 | 值得学习 | 不应直接复制 |
| --- | --- | --- |
| [vpipe固定plan/finish](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/generative-models/shared/ane-ffn.cc#L581) · [源码走查](case-studies/vpipe-controller-and-failure.md) | split/probe/GPU三种模式，收益门与份额迟滞 | 3%/5%等源版本阈值 |
| [ccv固定chooser](https://github.com/liuliu/ccv/blob/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b/lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_gemm.cpp#L995) · [成本分析](case-studies/ccv-hybrid-cost-and-failure-boundaries.md) | 实际两种GPU分支、chunk约束、校准成本 | 256/4096行粒度及其他机器拟合值 |
| [Splash校准/重跑](case-studies/splash-handoff-calibration-and-replay.md) | 邻域稳定性、资格检查、失败整块回退 | 单点最小耗时作为永久份额 |

这些是固定源码设计和各案例所列实验，不代表本库已运行所有控制器或证明物理ANE并发。

## 何时适用与何时否决

适用于有足够计算量、合法独立分片、权重可驻留且交接可控的消费者。小shape、频繁变化的shape、额外量化质量损失、完整请求无收益时，应保留原GPU。

性能缓存至少区分SoC/OS/工具链、shape/layout、dtype/量化、驻留状态与并发负载；质量资格还要绑定checkpoint/adapter。首次chooser可能触发设备probe，不能把它当纯CPU查询。

## 最小学习与验收

运行[离线方法实验](examples/README.md)中的`parallel-branches`与`segmented-ffn`，学习合法性和归约，而不是测硬件速度。真实迁移记录prepare、两支完成、join、first/warm和完整请求；缺失物理counter保持unknown。采用门见[验收与回退](validation-and-fallback.md)。
