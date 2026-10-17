# PAB：按 attention 类型设不同的跨步广播周期

<!-- knowledge
id: technique-algorithms-methods-pab
type: method
stage: dit
operator: sampling, attention
backend: mlx, metal
device: gpu
symptom: stale-cache, precision-loss
semantics: approximate
representation: bf16
boundary: request, producer-consumer
api: reference-only
-->

配套：[机制 witness 与证据合同实验](../../examples/README.md)。这是支持性小实验，不是本方法的完整模型复现。

[算法入口](../README.md) · [FasterCache](fastercache.md) · [模型结构](model-sparsity.md)

来源：VideoSys `pab_mgr.py` [S09](../sources.md#s09)，Diffusers 的 PAB 文档/hook [S25](../sources.md#s25)。分类：`approximate-train-free`；不是缓存 softmax 权重的统一算法，也不是跨机器广播通信的同义词。

## 为什么分 spatial / temporal / cross

Pyramid Attention Broadcast 观察到不同类型 attention output 的跨步变化程度不同，因此用不同 refresh 周期、timestep 窗口复用输出。典型经验是 cross 可复用更久，其次 temporal，再 spatial；这是论文/文档经验，不是所有 joint-attention 模型都有这三个可分分支。

VideoSys 控制器用 `count % range != 0` 和 timestep 区间决定广播，另提供按 block/timestep 的 MLP skip 表与输出池。在 cache hit 上，调用者必须真正绕过 attention 并使用缓存 tensor；只实现 manager 不足以证明节省计算。

## Apple 实现与成本

固定周期不需要每一步读回误差 scalar，适合 eager MLX，或把周期展开为 full/reuse 固定图。代价是每个缓存模块一份 attention/FFN 输出，长视频可占明显内存；如果 memory pressure 导致权重换入换出，减少算子可能仍变慢。

同 prompt 的 text embedding 不变，只能证明 embedding cache 精确；cross-attention 的 Q 来自不断变化的视觉 hidden，**cross output 并不因此精确不变**。很多模型还有 timestep-conditioned text 状态，连 K/V 是否可精确复用都要按层核实。

Diffusers 文档给出 MPS 设备的使用方向，但 hook 的 tensor 操作可表达不等于整模型在该 torch/macOS 组合中已获速度/质量资格。不要把文档中的 `to("mps")` 当成作者 Apple 基准。

## 比较与验证

PAB 的优势是控制逻辑简单、周期可预测；劣势是不能随 prompt/step 难度自适应。相对于 TeaCache/DBCache，应比较同等媒体质量约束下的 full count 与 E2E，而不是相同数字阈值。

[AdaCache/MoReg第十六轮](adacache-motion-and-period.md)提供动态period对照：在选中full residual上测变化、调整后续间隔，原consumer将rate共享给全部spatial/temporal attention、cross和MLP；不是仅在选中的metric模块省算。八个完整tiny CPU sampler请求说明NFE仍100、MoReg会增加full次数并改变合成端点误差，没有新GPU速度或真实视频资格。固定周期与动态周期的收益要在相同quality/cost预算下比较。

先 range=1 的 no-hit oracle；验证每层计数与 CFG 分支，不要把 `timesteps` 原始值、循环 index 与模型 forward 次数混为一个轴。对长短视频、镜头切换、动作峰值分别做 holdout，窗口外/最后步强制 full，换 scheduler 后重新构建周期。缓存失效仍遵守 [身份合同](caches.md)。

论文：[Real-Time Video Generation with Pyramid Attention Broadcast, arXiv:2408.12588](https://arxiv.org/abs/2408.12588)。
