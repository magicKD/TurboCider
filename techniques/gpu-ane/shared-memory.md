# 共享 IO：零拷贝、同步与所有权是三件事

<!-- knowledge
id: technique-gpu-ane-shared-memory
type: method
stage: pipeline
operator: layout, scheduling
backend: coreml, metal
device: gpu, ane
symptom: wrong-layout, lifetime
semantics: contract-only
representation: iosurface, pixelbuffer, fp16
boundary: gpu-ane, producer-consumer
api: public
-->

[总目录](../README.md) · [生命周期](../ane/lifecycle.md)

## “零拷贝”的准确边界

caller-owned Core ML `outputBackings` 可避免默认输出 bridge copy；IOSurface / CVPixelBuffer / MLMultiArray / Metal no-copy 可以共享 storage。它们都不证明：无 pack、无 dtype conversion、无 cache traffic、无 host barrier、无额外 pool 或物理设备 overlap。

[coreml.mm](../../native/backends/coreml.mm)会核对实际 output pointer；Core ML 不使用请求 backing 时需要 stride-correct bounded copy 并计数。`copied_bytes=0` 只证明被计数的输出 copy 为零，不等于完整请求没有任何 copy。

## 每个 view 必须带的合同

`buffer + byte offset + logical shape + dtype + physical row/plane pitch + mapped extent + owner + source generation + completion`。

对齐要分清：逻辑 tensor 大小、surface row extent、VM 可映射页范围和实际 allocation bytes。不能简单向上 page-round 然后假设多出来的页可读写。SeedVR2 probe 用实际 VM region 验证再包 no-copy，见 [shared tile 记录](../../docs/design/seedvr2-vae-ane-feasibility.md)。

## 合法 producer / consumer 时间线

```text
GPU pack/input producer → ready
                           ↓
                  Core ML / ANE execute → done
                                             ↓
                                   GPU restore / join → consumer done
```

Surface 只在最后 consumer 真正完成后复用。prediction API 返回不必然意味着外部 GPU consumer 已消费；host `finish()` 返回更不是下一个请求重写原 scratch 的许可。

Public W8 使用 CPU listener 等 ready，持久 worker 同步 prediction 后 signal done，GPU epilogue消费；这是公共 API 边界。若同一 GPU queue 先排 done-wait、后排产生 ready 的工作，就有依赖环。Private 的 shared event 必须另做 driver completion 验证。

## 两槽 / 两 bank 的规则

- 当前 W bank 只读，下一 W bank 由 producer 写；激活下一 bank 前既要 producer done，也要旧 consumer done。
- A8 slot 和 W bank 是不同资源/时间线，不把“两槽”泛称为“两完整 FFN copies”。
- future identity 必须包含 allocation/generation、encoding、metadata、full pitch、logical region、rotation/quantization recipe。
- source mismatch 丢弃 future 并 restage；failed producer 不发布 ready，不拿上次内容填空。
- 取消/失败先 join/drain，再 discard/release；timeout 时 quarantine 尚未证明完成的资源。

## 从 ANE scratch 发布给 MLX

默认 worker scratch 属于下一次调用可复用存储。发布结果必须独立拥有，或有严格不可复用的 lease。Deferred join 也只能在 ANE/restore 完成且独立 owner 已建立后，把剩下的 lazy GPU add/down-LoRA交给正常 consumer。

[deferred join](../../docs/status/private-ane-deferred-join-2026-10-05.md)的关键不是少一个 `eval`，而是“先复用旧 surface / 销毁 executor，再消费已返回 tensor”仍正确。host post-join 变短的时间不包含未来 GPU 消费，不可当作算术节省。

## 必测故障

poison output 检查漏写；非连续 stride；first/last tail；source 同地址新 generation；input/output alias；future A/B switch；late chunk NaN；callback 抛错；timeout/late completion；取消后恢复；base→A→base；返回结果后 stage 下一任务或销毁 executor。

真实接口参考：[DeviceMatrixView / DeviceWeightView](../../native/backends/ane_runtime.hpp)、[shared GPU stager](../../native/backends/ane_gpu.hpp)、[Public W8 完整桥接](../../docs/status/public-w8-shared-ffn-2026-10-07.md)。

## 实现参考与实验

可读实现：[参考代码](../examples/reference/systems.py)；[实验步骤与边界](../examples/README.md)；本项全部 case / controls。

本项控制：wrong generation and live-reader overwrite；actual IOSurface/MLMultiArray shared IO。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：new backing, pitch, event or output ownership boundary。历史负结果限定于其来源/shape/边界，不关闭该方向。
