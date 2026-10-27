# 驻留策略与真实阶段峰值

<!-- knowledge
id: technique-memory-residency
type: method
stage: encoder, dit, vae, io, pipeline
operator: scheduling
backend: cpu-reference
device: cpu, gpu, ane
symptom: memory-bound, lifetime
semantics: contract-only
boundary: weight-runtime, request
api: reference-only
-->

[总目录](../README.md) · 阶段预算演示

## 预算先按 lifetime，而不是 checkpoint 总大小

```text
condition: encoder + encoder scratch + conditioning
denoise:   conditioning + DiT trunk/blocks + slots + activation + latent + ANE IO
decode:    latent + VAE weights + global state/history + convolution scratch + writer
export:    尚未消费的帧/音频 + codec / mux / staging
```

串行请求峰值是各阶段 live allocations 之和的最大值。conditioning/latent 跨阶段存活，不能只放在生产阶段；同 backing 的 views 只计一次，但要计完整 backing。跨请求 pipeline 并发会破坏串行 phase max 假设，必须画联合时间线。

## 四类策略

| 策略 | 适合 | 代价 / 风险 |
| --- | --- | --- |
| resident | 重复请求、实际 RAM 足够 | idle encoder/DiT 可挤压 VAE；冷加载不免费 |
| component-staged | encoder→DiT→VAE 不同时使用 | 重新 load；需保留 conditioning/latent 而非权重 |
| block-streamed + pinned subset | 模型超过目标工作集 | 反复 IO、slot/decoder scratch，见 streaming |
| isolated helper / exec finalizer | allocator/cache 或生命周期干扰下一阶段 | 新进程、IPC/文件、startup；计入完整边界 |

[native README](../../native/README.md)保留 LTX resident candidate 导致 VAE decode 变慢的历史经验；模型“常驻”不保证完整请求更快。保住小 conditioning，释放大 encoder，通常比保住所有 weights 更值得先测。

## 计数器不能相加

- MLX active：allocator 已用张量；cache/peak 有各自含义。
- Core ML / IOSurface / native pools：可能不在 MLX active 内。
- RSS 与 physical footprint：系统不同口径，包含/排除项不相同。
- process-tree sampler：覆盖选中进程和可追踪后代，仍可能漏外部 daemon/driver。
- compression/swap：系统观测，不全部归因于某一模型或 ANE。

`MLX active + physical footprint` 是重叠计数；“GPU tensor 放 CPU”在统一内存上也不证明物理 RAM 下降。应记录每个指标和时间范围，而非拼一个假总数。

## 高水位移动

SeedVR2 释放/共享 upscale causal history 后，decoder 的高水位会移动到另一个 norm/conv；再优化 VAE 后，全请求高水位可能仍位于 DiT 输出前。依据 [phase/history](../../docs/design/seedvr2-vae-phase-and-upscale-history.md)、[norm-pack](../../docs/design/seedvr2-vae-norm-pack-and-resource-advice.md)，每轮重新定位最大值，不把局部节省直接当全请求节省。

[serial block weights](../../docs/design/seedvr2-block-weight-residency.md)在真实1024/t25 DiT component 将 active maximum 从约11.73降到5.23 GB；这是 compile-isolated 组件观察，未证明完整请求或16-GiB机器可用，也未推广App。

## 设计 admission 的规则

实际 tensor table、最宽 block/class、activation geometry、所有 pool slots、conversion workspace、Core ML/surface、framework allowance、媒体缓冲和磁盘 scratch 都有项。未知项标 unknown，不写0。budget hint 与已认证 hard bound 分开；no swap 要用持续系统采样证明该窗口，而不是只看结束时 swap used 没变。

压力不足时先拒绝/释放可选资源，不能减少 shape/steps/参考图来蒙混过关。释放只有在 actual last consumer 完成后成立；不能因 C++ 变量离开作用域就相信 GPU 不读了。

## 辅助工具

`interval_peak.py`用声明的连续 phase 区间计算 backing 去重峰值与 target slack。它不读系统、不自动改内存策略、不推断未声明 driver bytes、不提供硬上限。实际 runtime 应复用 [memory accounting](../../native/runtime/memory_accounting.cpp) 与 [memory scheduler](../../native/runtime/memory_scheduler.cpp)，不要把教学 planner 接成第二套 admission。

## 实现参考与实验

可读实现：参考代码；[实验步骤与边界](../examples/README.md)；本项全部 case / controls。

本项控制：alias-backed phase intervals；live readers vs apparent lexical release。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：new phase overlap, lower RAM, external pools or peak movement。历史负结果限定于其来源/shape/边界，不关闭该方向。
