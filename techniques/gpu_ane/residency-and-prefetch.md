# 权重驻留与预取：减少搬运，同时守住bank和代际

<!-- knowledge
id: technique-gpu-ane-residency-and-prefetch
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

[方法导航](README.md) · [共享交接](memory-and-handoff.md) · [阶段划分](stage-pipelines.md)

## 解决什么问题

权重文件能放进内存，不表示完整请求没有装载、解码、staging或GPU工作集切换。先区分磁盘文件、CPU暂存、packed GPU bank、解码后权重和运行workspace，再决定缓存哪一层。

## 两类可复用设计

**稳定专家bank。** 用expert ID映射稳定slot，GPU直接消费正确packed格式；CPU暂存只负责装载，不拥有GPU bank的整个生命周期。slot应同时绑定expert ID、格式、scale、有效性、generation及最后读者完成。

**按阶段/分段驻留。** 将权重storage与图执行分开；按阶段预测工作集、提前装载、在安全完成点释放pin。预算包括workspace和prefetch双buffer，不能只相加模型文件体积。

## 实现参考

| 入口 | 值得借鉴 | 迁移边界 |
| --- | --- | --- |
| [固定MoE完整block](https://github.com/Anemll/anemll-flash-mlx/blob/f86c0682a5dcd44eee1f1c090c588e4f209b62c9/flash_moe_mlx/model.py#L647) · [I/O wrapper](https://github.com/Anemll/anemll-flash-mlx/blob/f86c0682a5dcd44eee1f1c090c588e4f209b62c9/flash_moe_mlx/expert_io.py) | 稳定slot、按ID路由、packed驻留 | short-read不得把无效slot交给GPU |
| [原MoE完整GPU消费者](case-studies/anemll-original-bank-gpu-consumer.md) | packed/Python-bank/native-bank同值对照 | 组件数值通过不证明SSD、整模型速度或容量收益 |
| [SD.cpp驻留与分段](case-studies/sdcpp-metal-residency-and-segments.md) | pin、swap、prefetch与workspace分账 | 独立backend context不自动等于独立queue或物理overlap |
| [Apple pipeline资源管理](case-studies/apple-coreml-pipeline-lifecycle.md) | lazy load、阶段unload、preview与取消 | 对象引用清空不证明框架cache或设备已drain |

## 常见误判

- cache hit率不等于实际SSD节省；文件缓存热度、装载字节和消费速度分别测量。
- 内存映射不等于GPU零拷贝；实际可能经shared temporary再blit。
- `segment_end`解除pin不等于立即evict；预取加深也可能扩大峰值或争用带宽。
- 编译资产缓存、性能路由缓存、权重/质量身份不是同一个key。

## 怎样验证

先固定packed数据和独立参考，覆盖hit/miss、跨层换专家、重复ID、无效ID、short-read、A/B/A及旧输出快照；完整consumer输出通过后，再测I/O与请求时间。

记录逻辑字节和实际resident/峰值时，不混入未测的驱动内存推断。显式预算预取buffer、dense展开、workspace和并发请求；保留不预取、低驻留及纯GPU回退。通用方法见[驻留](../memory/residency.md)和[streaming](../memory/streaming.md)，离线练习见[方法实验](examples/README.md)。
