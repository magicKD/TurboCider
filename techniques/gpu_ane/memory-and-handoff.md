# 共享内存与消费者 ABI：把“零拷贝”拆成可验证的事实

<!-- knowledge
id: technique-gpu-ane-memory-and-handoff
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

[专题导航](README.md) · [ccv 源码](case-studies/ccv-rowwise-iosurface.md) · [实验记录](../MAINTENANCE.md)

官方入口：[Metal shared resource](https://developer.apple.com/documentation/metal/mtlstoragemode/shared)、[IOSurface](https://developer.apple.com/documentation/iosurface)、[MLPredictionOptions](https://developer.apple.com/documentation/coreml/mlpredictionoptions)。本库既有 [公共共享内存取证](../sources/official/shared-memory.md)、[实际 request surface](../gpu-ane/shared-surface.md) 提供更详细版本/源文件记录。

## 方法卡：让消费者直接使用合法 backing

- **机制**：在 producer 和 consumer 间保留可消费的存储/布局，减少重复 materialize 或复制；ready、可见性与所有权仍是必需工作。
- **何时考虑**：交接复制/转换在关键路径上，且框架能采用符合 ABI 的实际 backing。
- **代价/反例**：共享 DRAM 不证明无复制；wrong stride、过期 generation、未完成写入和提前复用都可能产生 finite 但错误的输出。
- **实现参考**：[实际 surface](../gpu-ane/shared-surface.md) → [MLX adoption](../metal/case-studies/external-memory-adoption-and-fresh-views.md) → [完整共享 FFN 消费者](case-studies/qwen-shared-ffn-consumer-cost.md)，各自范围不能拼接扩大。
- **最小验证**：先用 stride/旧输出/提前复用负控验证检测能力，再对照 copy 与合法共享路径，覆盖最后消费者的输出、时间和 lifetime。

## 一条完整交接链

```text
GPU 原始输入
  → GPU pack/cast/quantize 到实际 ANE input backing
  → producer 完成 + 跨框架可见
  → ANE graph 完成并写 actual output backing
  → GPU view/restore + complement partial
  → runtime join → 完整输出 ready
```

统一物理 DRAM 只消除某类设备间独立显存复制的必要性，不能自动消除 pack、padding、layout conversion、框架内部 staging、cache flush、ownership 和 completion。每个箭头分别计时，外部 copy=0 不等于 runtime 内部 copy=0。

## Tensor ABI 至少包含什么

| 类别 | 必须绑定的内容 | 典型错误 |
| --- | --- | --- |
| 数学 | checkpoint、LoRA、bias/residual、precision recipe | 两支计算了不同模型 |
| 几何 | shape、dtype、element stride、byte pitch、offset、padding | 逻辑 shape 对，物理读错仍 finite |
| 分片 | row/F 区间、gate/up 配对、packed group | gate/up 对不齐、down 漏片 |
| 存储 | actual surface/buffer、owner、allocation generation | requested backing 未采用、地址复用 |
| 时序 | producer ready、ANE ready、last-reader done | 未提交/未完成、旧输出、提前释放 |

例如 `[1,C,1,S]` 的 channel-first surface 与 GPU `[S,C]` row-major 不因元素数量相同就兼容；CCV 读本、Core ML 图与真实 private matmul port 还可能不同。本库 Qwen 的实际端口为 `[1,1,1024,4096]`、row pitch 8192 bytes，不能套用之前 toy 的 BC1S stride。

## 如何证明 backing 被真正采用

请求 wrapper、返回 multiarray、底层 surface、Metal mapping 与 GPU view 分层核对。若 returned pointer/surface 与期望不一致，按实际 stride copy 到 owned buffer，或拒绝当前优化。只核对 shape/count、调用 `setOutputBackings` 成功或 callback 返回，都不充分。

直接映射也要满足 API 的 page/alignment/length/resource 合同；owner 要覆盖 GPU/ANE 最后一次访问。noop deallocator 只表示借用，不提供外部对象的 lifetime。tracked/untracked Metal resource 的 hazard 规则不同，不能为了去等待盲目改为 untracked。

## MLX 的特殊接缝

先确认 actual external-memory adoption，失败回退是否发生 copy；view 是否共享 owner，非连续 stride 是否触发 contiguous temporary。见 [MLX 外部存储导读](../sources/repos/mlx-external-memory.md) 和 [实测](../metal/case-studies/external-memory-adoption-and-fresh-views.md)。

输出 buffer 内容会改变时，把每次输入 view 作为 compiled join 的**运行时参数**，而不是只在闭包里捕获一个已 eval 的 array。same-dtype copy 在某路径可 alias；“我调用过 copy”不证明 snapshot。generation、poison、先前 point 的真实输出是很有效的 stale 负控。

## 所有权与失败

单 slot 串行复用，或者用有限 ring/bank 明确每个 slot 的 last-reader completion。取消不能立即 free/recycle；先停止后续提交、等本请求已提交工作完成，再丢弃输出并释放 owned resources。只释放自己创建的对象，不清全局系统缓存或重启服务。

普通 command completion 是保守边界；fast fence/atomic flag 只有在具体 runtime 的执行顺序与可见性都经验证时才可替代一部分等待。CPU 观察窗口相交不是物理设备并行证明。

[Splash的完成门与watchdog](case-studies/splash-command-capacity-and-watchdog.md)进一步区分GPU终态、ticket发布和host消费：callback停计时但不放开下一次提交；未完成命令超时后backend unhealthy，completion handler继续保活其state/allocations。timeout返回、丢弃ticket或释放wait都不授予free/recycle权限；该结论来自固定源码和CPU策略观察，本轮未实测driver timeout或drain。

[SD.cpp的参数streaming链](case-studies/sdcpp-metal-residency-and-segments.md)则展示另一种接缝：统一内存下CPU source仍可能经复制式Shared temporary再blit到Metal backing；第二backend context沿所读固定GGML仍共用device queue。prefetch activation先等event再swap buffer/data/extra；segment_end减pin不等于evict。实际只执行原始CPU预算planner，copy/owner/drain与物理重叠仍未设备验证。

[Apple公共pipeline](case-studies/apple-coreml-pipeline-lifecycle.md)的ManagedMLModel又是单模型serial gate：perform时lazy-load，unload将引用置nil，不证明框架cache/设备内存已drain。分块预测保留先前已load模型；reduceMemory正常按阶段释放，preview可能把decoder带回denoiser驻留窗口，早停和异常还需独立cleanup。原始资源协议仅用假CPU资源运行，未取得真实CoreML模型的释放/placement证明。

## 负控与测试

以上是 CPU 合同演示。设备实验需额外故意错 stride/padding、改 generation、拿前一输入输出、alias 写冲突，并核对完整输出而不是几个抽样点。本库 [shared FFN](case-studies/qwen-shared-ffn-consumer-cost.md) 的 exact runtime join 与 [shared surface](case-studies/surface-strides-and-stale-output.md) 的 finite-but-wrong 负控是独立证据，不应拼成所有引擎都具备的性质。
