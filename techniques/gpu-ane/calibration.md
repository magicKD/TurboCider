# 带宽耦合、share 搜索与性能校准

<!-- knowledge
id: technique-gpu-ane-calibration
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

[总目录](../README.md) · [测量口径](../profiling/measurement.md)

## 基本成本模型

```text
T_hybrid(s) ≈ T_exposed_stage(s)
            + max(T_gpu_concurrent(s), T_ane_concurrent(s))
            + T_exposed_restore_join(s)
```

alone 时间只能作起点：共享内存、GPU W/A staging、CPU 工作会让 concurrent 两支都变慢。`finish()` 等待很短可能只是 ANE 已被别的工作隐藏，并不能拿来当 `T_ane_alone`。

## 正确的三臂采样

1. 准备相同、已完成 producer 的输入和独立输出。
2. GPU alone：执行实际 GPU family shard 与其属于关键路径的工作，不提交 ANE。
3. ANE alone：实际提交对应图，不偷偷依赖同场 GPU callback。
4. Concurrent：两支没有互相结果依赖，同时按真实 pipeline运行，等待两侧完成。
5. 计入真实模型层深度、stage/restore/join/LoRA 与 future prefetch；fixture 只有 FFN head 时明确标为不完整。

固定绑定/setup 很大时，可测1层和4层，假设同质线性部分，估计 `T_layer=(T4-T1)/3`。必须满足有限、正值、`T4>T1`，且说明此假设；不能跨不同 shape/热状态相减。

## 当前可复用实现与边界

[ChannelCostModel](../../native/backends/ane_cost_model.hpp)是 CPU policy：对两份 share 的 GPU/ANE/both 观测拟合，保留带宽耦合不确定性的保守 envelope，拒绝超出已测 share 区间的外推；只在 memory-admitted candidates 中选达到 minimum gain 的 near-optimal 最小 share，并明确要求独立 candidate trial。

它是一个可测试的决策模型，不是自动完成的全模型调优器。早期 [prepared calibration](../../docs/status/private-ane-prepared-calibration-2026-10-05.md) 的真实设备 fixture只含合成权重/GPU FFN head，不能据此声称模型带宽校准、staging overlap 或低内存资格完成。

后续实际接入在 [ane_channel_calibration.cpp](../../native/backends/ane_channel_calibration.cpp)：模型提供五个深度的真实sources与family callbacks，用确定性合成activation，独立完整traffic one/four测量，再建立新的真实HybridFfn候选、四个完整FFN数值对照和交错热窗口；只将试验通过的正选择存入有界weak-source cache。graph admission在重建时重查。它不是只有纯公式，也不是完整diffusion trajectory/真实activation holdout资格。历史 [auto接线记录](../../docs/status/native-channel-auto-integration-2026-10-05.md) 与当前源码的LoRA范围有演进，支持与限制以当前guard为准，不能拿早期base-only结论覆盖后来接口。

## Share / bucket / chunk 不同层次

- share：哪些 intermediate channels 或 token rows 分到 ANE。
- bucket：单次图容纳多少 rows，padding 比例与调用成本。
- chunks：本次调用多少固定 row chunks。
- backend：GPU / Public frozen/runtime / Private research。
- prefetch：额外 producer 是否值得并行。

逐轴少量候选筛选，拒绝无边界大网格。先 memory gate，后 numeric gate，再完整 block，最后 E2E；成本模型建议不能跳过独立测量。

## Cache key

模型/权重与 logical source generation、adapter/strength/有效精度、rows/geometry/phase、encoding/rotation/S1、backend、SoC/RAM/OS、runtime build、kernel/graph ABI、bucket/chunk/share、prefetch policy。热/负载状态有 TTL 或失效机制。pointer-only 或 geometry-only key 不安全。

## 非线性与回退

不同 tile、编译/shape 跳变、内存 compression、stage queues 会让成本不连续。观察到排序反转就保留多窗口数据，不强行解释成固定线性带宽模型。内存不合、setup 太重、gain margin 不足，回普通 GPU；校准本身的时间/内存也属于冷请求成本。

四个结果必须分别报告：建议 share、实际执行 share、完整 FFN/block 收益、完整请求收益。再额外标记是否有物理 device trace；预测图不是硬件并发证据。

## 实现参考与实验

可读实现：[参考代码](../examples/reference/policies.py)；[实验步骤与边界](../examples/README.md)；本项全部 case / controls。

本项控制：memory-admitted measured-range proposal；new actual candidate execution vs optimized GPU。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：new complete traffic samples or nonlinear model fit。历史负结果限定于其来源/shape/边界，不关闭该方向。
