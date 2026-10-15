# 权重 streaming、typed slots 与 pinned subset

[总目录](../README.md) · [vpipe](../references/vpipe.md) · [存储](storage.md)

## 把不可控换页变成显式生命周期

一个典型路径：读下一 block → 填 reusable typed slot → 绑定当前 generation → 提交计算 → 等所有 last readers → slot refill。预取隐藏 IO，但不是让 IO 消失。

近似下界为 `T_block ≥ max(T_compute,T_read+decode)`，外加不能隐藏的绑定和等待；真实 pipeline 有有限 slots 和共享带宽，不能只拿 SSD 顺序带宽预测整模型。

## 容量估计

```text
working_set ≈ nonblock_trunk + live_activations + reserve
            + pinned_blocks * block_bytes
            + Σ_class(slot_count[class] * slot_capacity[class])
            + conversion/staging scratch + ANE banks/IO
```

同一个“K2”如果有 dual/single 两类 pool，可能是四个 bundle，不是总共两槽。block group size 改加载/驻留粒度，不等于 kernel fusion。应使用最宽真实 block/class，不用平均值或 block0 猜容量。

当前 C policy 见 [block_residency.h](../../native/runtime/block_residency.h)：明确 working-set target，不冒充 RSS cap；固定/自适应 refill slots、是否允许全驻留由调用方合同决定。通用实现分布在 [slot_pool](../../native/runtime/streaming/slot_pool.cpp)、[StageExecutor/context](../../native/runtime/streaming/context.cpp)、[io_executor](../../native/runtime/streaming/io_executor.cpp)，模型 adapter 负责数学与绑定。

## 循环扫描下的LRU风险，以及可重新比较的条件

顺序循环 `0…N-1`，capacity C<N 的纯 LRU 可能每轮零命中：下一轮需要的正是刚被逐出的 block。固定 pinned prefix/subset 能保住明确份额，其余继续 streaming。vpipe 进一步采用 admission slack、eviction ratchet 与实际 residency 观察，避免 growth/evict 反复抖动；见 [外部源码对照](../references/vpipe.md)。

free+file-cache 不等于新 headroom：自己的顺序读会增大 file cache，若把它当“可再放权重”，可能自我放大直到 compression/swap。reserve 中已经实际分配的 activation 也不能再扣一次。

若新模型访问不是顺序循环，例如非均匀专家/热层/稀疏路由，LRU或其他replacement也可能合理；先收集访问trace，对比fixedsubset、LRU和新policy，不机械套用这个负结果。

## Typed refill：三种结果不能合成 bool

1. filled：全部 bytes、dtype、shape 与 consumer 合同正确。
2. unservable：这个 tensor 不能 raw-copy，走它自己的转换慢路径。
3. failed：short read / source mismatch / partial write，整份目标无效，不能继续消费或补几项就发布。

BF16/F16 虽同宽但位语义不同；U32 packed words 不可当浮点 cast。F32 modulation 可能只占极小字节，不应因为它不 raw-copy 就关闭整个 block 的快路径。slot generation、pass/step、source identity、physical last-reader fence 都需验收。

## Streaming 与量化应一起规划

离线 provenance-bound I8/F32 shards 或直接 raw GGUF packed import 可避免每步完整BF16→INT8重做。packed GPU QMM、bounded ahead decode、ANE stage 是不同路线；都要算 staging banks/解码 scratch，不把全模型浮点展开叫低内存。

[ahead decode 进度](../../docs/status/performance-progress-2026-10-10.md)的四层真实依赖链中，Q4/Q8 ahead 仍比 packed 慢；不能因单次 decode kernel 快就加更多预取槽。

## 一个容易漏的异步 IO 异常

[SeedVR2 block-weight 实验](../../docs/design/seedvr2-block-weight-residency.md)发现所用 MLX Load 路径等待 IO future 但未重新抛出 reader 异常。修复不是只检查取消 flag，而是 reader 锁存原 `exception_ptr`，owner 在 raw array eval 后、cast/GPU消费前检查，再在原子发布前检查。

新 loader 必测真实 partial-read failure、非取消 IO error、source替换、同模型恢复、完整系数/读取范围与全 forward 比较。`future.wait()` 完成不能代替 `get()` / 显式错误传播。

## 验证顺序

先 resident/default 无回退 → 同一 frozen streaming layout 的旧/新 executor → streaming vs resident 速度/内存 → 实际低 RAM/压力环境 → App/API完整请求。四种比较回答不同问题。

[streaming 设计目录](../../docs/design/streaming/README.md)包含大量实施/发布方案，不是所有卡片都已通过；[GPU/ANE integration plan](../../docs/design/streaming/57-gpu-ane-integration-and-delivery-plan.md)明确 hybrid 是新增资格，不能继承 GPU-only slot 的内存证据。

## 实现参考与实验

可读实现：[参考代码](../labs/systems.py)；[实验步骤与边界](../labs/memory-io.md)；[本项全部 case / controls](../labs/catalog.json)。

```sh
python3 -B techniques/labs/run.py bounded-io
```

本项控制：serial vs bounded one-future checksum；short read and late generation errors。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：different block/pool/slot layout, source encoding or access trace。历史负结果限定于其来源/shape/边界，不关闭该方向。
