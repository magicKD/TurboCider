# Splash：模型专用内核与显式执行图

[总目录](../README.md) · [其他参考](related-projects.md)

## 核查来源与版本

本地只读checkout：`../splash`，HEAD `134807b80bd6f64b1533cb1306358c042bb750d5`。另一个带ANE扩展的`../references/splash-ffn`，HEAD `80827e1c3593f7d10c266394e4d4381321b73ad0`。两棵树不是同一状态；这里是2026-10-15本地检查，不声称上游最新。上游：[incoai/splash](https://github.com/incoai/splash)，Apache-2.0。

## 读代码时最值得看什么

| 文件（相对Splash根目录） | 读到的机制 | TurboCider迁移点 |
| --- | --- | --- |
| `runtime/metal/CommandGraph.hpp` | ordered dispatch list；参数复制到graph-owned deque；禁copy避免指针指回旧payload | operator私有组装、owned参数、减少submit |
| `runtime/metal/MetalBackend.mm` | validate bindings/geometry、async ticket、watchdog、completion | 有界segments、actual完成与资源保活 |
| `runtime/model/WeightStore.cpp` | read-only MAP_SHARED、对齐sections、magic/layer/type/size验证、shared mapping owner | 自定义pack在消费前验证；view保住完整backing |
| `runtime/engine/MemoryPlan.cpp` | fixed权重/vision/draft/scratch/reserve先扣，再定KV/batch容量 | 不让dynamic cache抢固定工作集；unknown/overflow拒绝 |
| `runtime/ops/ExecutionPlans.*` / `Linear.cpp` | shape/format专用dispatch与原打包格式配套 | 让pack/consumer共同设计，别先随意转格式 |

Splash profiling mode可把多dispatch graph串行为一dispatch一command；这种profile会改变正常并发。不能把profile时序当生产端到端收益。TurboCider同样应把定位和正式计时分开。

## 带ANE分支的额外经验

在`splash-ffn`查看 `runtime/ops/AneFfn.hpp`、`AneFfnCalibration.hpp/.cpp`、`AneFfnMeasurement.*`、`runtime/engine/AneFfnStartup.*`：

- prefill SwiGLU按intermediate channels切GPU/ANE，GPU直接读Q4/GGUF planes；GPU提前一层requantize到双W surfaces。
- 最小可容纳chunk的fixed-row function，避免小chunk总付最大bucket成本。
- ready/done event把ANE工作夹在GPU input pack与partial join之间；failure使当前结果不可用，整chunkGPU重做并退役split。
- calibration选择share和minimum rows；运行时breaker观察一段commands后停用不盈利split。
- idle释放program与恢复，保留自己的surface账本；timeout未完成资源不假装可回收。

这些是LLM prefill实现，不直接推成扩散模型E2E资格；其channel/rows/格式限制要读实际版本。成本拟合也不是硬件带宽真值。TurboCider用 [保守耦合模型](../gpu-ane/calibration.md)与完整候选试验，不照搬系数。

## 哪些不应机械迁移

Speculative decoding/draft acceptance是自回归机制，不是diffusion直接可用的少步替代。LLM pagedKV/前缀cache对编辑encoder可启发layout/identity，但扩散timestep和参考prefix数学不同。特定模型exactshape kernel不适合未经验证的任意H/F/heads；必须重新建立numeric/performance gates。

这不排除探索diffusion draft/verify、可证明的trajectory复用或新prefix策略；先定义接受/修正与误差合同。更细的 [W8A8实现导读](splash-w8a8-walkthrough.md)记录segment、sign recipe、Measurement与Handoff的具体接缝。

## 当前TurboCider对应实现

[metal_graph/backend](../../native/backends/metal_graph/backend.hpp)标明设计审计与implementation-time参考revision；没有链接Splash runtime。Qwen independent native graph、SourceLease、memory ledger和actual receipts仍归TurboCider原框架管理。不把Splash当构建依赖，也不复制其模型包/权重。

继续：[command graph合同](../metal/command-graphs.md)、[shared IO](../gpu-ane/shared-memory.md)、[源码清单](evidence-map.md)。
