# 缓存：先定义“同一个输入”

<!-- knowledge
id: technique-algorithms-methods-caches
type: method
stage: encoder, prefill, decode, dit
operator: attention, scheduling
backend: mlx, metal, coreml, cpu-reference
device: cpu, gpu, ane
symptom: stale-cache, memory-bound, compile-bound
semantics: bitwise, approximate
representation: packed
boundary: weight-runtime, producer-consumer
api: public, reference-only
-->

[总目录](../../README.md) · [测量](../../profiling/measurement.md)

算法扩展：[总览](../README.md) → [TeaCache](teacache.md) / [EasyCache](easycache.md) / [FBCache与DBCache](block-caches.md) / [TaylorSeer](taylorseer.md) / [PAB](pab.md) / [DeepCache](deepcache.md) / [FasterCache](fastercache.md) / [Sea与Mag](spectral-magnitude-cache.md)。本页继续负责身份、失效和生命周期，不将近似缓存变成精确缓存。

## 六种 cache 不可混称

| 缓存 | 保存什么 | 典型失效 / 代价 |
| --- | --- | --- |
| artifact compile | `.mlmodelc` / kernel pipeline | graph/SDK/recipe/shape；磁盘与首次验证 |
| compiled closure | 固定GPU图 | shape/dtype/source/adapter/有效选项；捕获旧owner |
| conditioning | text、visual条件、reference VAE latent | prompt/tokenizer、ordered reference content/resize、模型身份 |
| prefix KV | 每层只读prefix | timestep/position/route/adapter/precision、内存很大 |
| W-code / scale metadata | staging重复量化或小scales | immutable generation、offset/pitch/region/recipe；内存换时间 |
| DBCache residual | 省中间blocks的近似残差 | step/history/条件变化、质量与warmup；不是exact缓存 |

最先应复用的是已计算且语义相同的conditioning；不能只按文件路径缓存参考图。当前 resident Qwen condition比较prompt、resize和有序reference SHA，参见 [公开缓存说明](../../../docs/public/CACHES.md)。Session unload释放，不自动变成disk KV cache。

## Prefix KV 的正确范围

每层独立bank，实际文本/参考latent/position/sigma、steps、输出geometry、LoRA identity、precision、route与manifest均参与。cancel/failure不能保留半成品，换输入要miss。跨seed是否可复用要通过uncached same-route oracle，而不是cached结果自比。

[resident prefix-KV 实验](../../../docs/status/qwen21-resident-prefix-kv-2026-09-26.md)发现LoRA同seed自比一致但跨seed未匹配uncached oracle，因此该实验加了LoRA guard；后来默认prefix policy的边界与实际request以当前代码为准，不把早期实验限制泛称整个Qwen永远不支持LoRA缓存。

缓存命中改变首步shape，ANE callback应按真实denoise step/phase安装，不能仅看“当前已经是decode-shaped”就让首步走未经许可的低精度FFN。actual calls应符合被选中的步骤。

## 近似 residual reuse

DBCache风格：前部blocks计算残差变化，满足阈值时重用上一次完整middle residual；有full warmup、最大连续reuse与最后一步full。它减少真实计算，不是编译缓存，也不是输出图缓存。

step重启/跳跃、trace、条件变更、callback失败和cancel/retry清状态；stored residual必须已evaluated且不保留旧输入计算图。MLX `copy`可能仍alias，不能只从API名字判断独立存储。

[Qwen DiT cache](../../../docs/status/2026-09-30-qwen21-dit-cache.md)在M4 Pro48GiB的特定25/40步场景有速度/质量权衡；六步student不应直接套相同warmup。该报告重复编辑的already-warm prefix基线下收益小于full-prefill基线，两个倍率不能相乘。

## W-code / metadata cache

小scale cache可降低反复统计/量化代价，但必须有weak generation和bounded bytes/entries。不要以hit数宣称收益；[Qwen W-code cache](../../../docs/status/local512-qwen-prefill-weight-cache-2026-10-10.md)多留约1.0–1.3GB且反序收益翻转，继续关闭。

模型需要保存的用cache，消费后就丢的用read；循环权重扫描用fixed resident subset，不用LRU，详见 [streaming](../../memory/streaming.md)。

## 必测序列

同输入repeat → 换seed → private reference原路径换bytes → reference顺序/resize变更 → adapter A/B/base → route/精度/steps变更 → cancel/retry → unload/reload。核对每步hit/miss、actual compute counters与same-route oracle。只加计数不验证内容，或只看“第二次更快”，都不足。

## 实现参考与实验

可读实现：[参考代码](../../examples/reference/caches.py)；[实验步骤与边界](../../examples/README.md)；本项全部 case / controls。

本项控制：content and effective recipe invalidations；off/full vs bounded residual replay。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：new cache object, invalidation proof or calibrated timestep policy。历史负结果限定于其来源/shape/边界，不关闭该方向。
