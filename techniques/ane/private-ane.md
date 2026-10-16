# Public Core ML 与 Private ANE：复用边界

[总目录](../README.md) · [当前后端合同](../../native/backends/README.md)

跨项目学习：[私有runtime层次对照](private-runtime-stack.md)、[动态参数训练](training-dynamic-weights.md)、[delta/LoRA](training-delta-and-lora.md)与[autograd/驻留优化器](training-autograd.md)。这些研究路线不改变下文发行隔离合同。

## 明确三个状态

1. Public Core ML 是可分发默认后端。
2. 当前 `native/backends/private/` 已有显式 research build 的 private executor、MIL emitter、W8A8/banks/events；不是只有早期 `experimental/` probe。
3. 实现存在不等于正式产品资格。稳定 package 应检查实际 private class strings / links / experiment markers，而不是只相信一个 build flag。

早期 [异构报告](../../docs/design/transformer-heterogeneous-report.md)写 private 仅在 `experimental/`，那是其历史阶段的发布边界。不要把它覆盖到 10 月之后的源码，也不要反过来把后续 research 能力写成稳定包已启用。

## 能复用的公共合同

[Executor](../../native/backends/ane_runtime.hpp)、[factory](../../native/backends/ane_backend.hpp)、HybridFfn、logical weights、LoRA ABI、source lifetime、memory admission、整 operation GPU 重算。Metal-only surface/staging 位于 [ane_gpu](../../native/backends/ane_gpu.hpp)，不为 Public 再实现第二套 decode/rotation。

Private client、MIL program、动态 framework/selectors、driver completion 和系统兼容性应留在 conditional private source。模型代码不直接 import 私有 API，不把研究 binary 替换正式 App。

## Private 能减少什么，不能自动解决什么

直接 surface 与 event 可能降低 wrapper / handoff；不能改变 attention→FFN 的依赖，也不能让 UMA 竞争消失。W/A staging 仍占 GPU/带宽；两套 W bank 仍占内存；quantization 仍有误差。

Public 同步 prediction worker 和 Private 真实 device event 的时序语义不同。共同接口不代表共同调度能力。任何 fallback 都要报告实际 backend，private graph load 失败后的纯 GPU 不能继续挂 Private 加速标签。

## 四类资格必须独立

- ABI/数学：code/scales/rotation/headroom/LoRA、tail 和 failed producer。
- 完整性能：优化 GPU 分母、相同 workload、cold/warm/反序与竞争负载。
- 完整质量/内存：多样本、全过程峰值、压力/取消/重试。
- 硬件证据：physical placement/overlap/INT8 MAC 需独立 trace；worker 数和 call counters 不替代。

## 研究环境的安全默认

新增[ANEForge源码与图搜索](graph-rewrite-search.md)已有[本机同步e5rt探针与失败缩减](../review/aneforge-20261016.md)。它保持为显式可选integration：没有修改产品后端，mask0x4不自动当成物理trace；结构winner cache也不代替checkpoint/LoRA精度资格。

显式 opt-in、独立新 build/output 目录、有界 banks/slots/pipelines、超时 quarantine、严格 source recipe。保留失败记录；不改用户权重、系统 wired/swap、电源策略、用户服务；不因实验需要清理原模型腾空间。

[Public W8 实现](../../docs/status/public-w8-shared-ffn-2026-10-07.md)证明公共 API 也可复用 GPU IO/normalized FFN；不要把 Private 热态成绩借给 Public，也不要把 Public 能编译借成 Private 系统兼容保证。

## 实现参考与实验

可读实现：[参考代码](../labs/systems.py)；[实验步骤与边界](../labs/ane.md)；[本项全部 case / controls](../labs/catalog.json)。

```sh
python3 -B techniques/labs/run.py slot-ownership
```

本项控制：public binary private-class isolation；timeout/quarantine lifetime。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：explicitly authorized isolated research with new ABI/device evidence。历史负结果限定于其来源/shape/边界，不关闭该方向。
