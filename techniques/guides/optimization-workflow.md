# 从瓶颈到可交付优化

<!-- knowledge
id: technique-guides-optimization-workflow
type: guide
-->

[总目录](../README.md) · [实验模板](../templates/experiment.md)

## 一条完整的证据链

```text
固定语义与输入 → 正确性基线 → 完整请求分阶段 → 定位关键路径
    → 单变量组件实验 → 完整 block → 真实完整请求 → 压力 / 生命周期 / 产品验收
```

任何环节变了（模型来源、LoRA、精度、尺寸、缓存、编译策略、媒体输出），先重新建立匹配基线。不要从“已实现一个 kernel”直接跳到“新模型已加速”。

## 1. 固定 workload

保存 checkpoint / adapter 内容身份、tokenizer、调度器、seed 或明确的初始 noise、尺寸、帧数、实际 token rows、参考图顺序 / SHA / resize、steps / sigma、输出格式和缓存状态。跨框架相同 seed 不一定产生同一 noise，比较张量时尽量注入同一 fixture。

先用普通 GPU 路线完成输入 → conditioning → denoise → VAE → PNG/MP4。编译、`plan`、正确的输出 shape 都不是实际生成证明。对音视频还要验证真实帧数、FPS、音频长度和封装。

## 2. 做阶段预算，不先选技术

| 主导成本 | 先尝试 | 常见误判 |
| --- | --- | --- |
| 首次准备 / 哈希 / 编译 | 内容寻址 artifact、验证缓存、load-only、会话复用 | 把准备移到 UI 后台就声称总成本消失 |
| 很多小 dispatch / reshape | 图编译、布局传播、norm/RoPE/epilogue fusion | kernel body 快但 graph 构建与同步更慢 |
| 大 FFN | 优化 GEMM；再测 channel / row split | 只用单独 GPU / ANE 延迟推并行效果 |
| 长 attention | fused / tiled exact、prefix KV、规则块路由 | gather/top-k 搬运吞掉稀疏收益 |
| 权重 IO / 压缩 / swap | 阶段释放、typed slots、固定 pinned subset | LRU 循环扫描抖动；file cache 当新增余量 |
| VAE / 媒体峰值 | 原始精度配方、bounded workspace、frame sink、history ownership | 只压 DiT 内存，VAE 仍决定峰值 |

## 3. 逐层扩大验证

1. CPU 数学 / 索引 oracle：尾块、stride、切片、bias 次数、有限性、rounding。
2. Metal / Core ML 组件：真实 shape、真实权重，包含 pack / stage / consume；合成输入只证明该输入。
3. 完整 block：包含依赖、adapter、残差、调度；先 drain 上游 lazy 工作再启动诊断计时。
4. 完整请求：固定 cold/warm 定义，交错 ABBA / 反序，关掉强同步 profile；记录所有失败和重试。
5. 质量 / 生命周期：多 prompt/seed、参考编辑与时间一致性、取消后恢复、晚期失败完整重算、卸载后 tracked owners 释放。
6. 产品：真实 App/CLI/API 路径、实际 loaded library 身份、保存/导入/切换配置；package 检查实验字节和 private links。

## 4. 选择能减少关键路径的优化

假设原请求中 FFN 占比为 `p`，局部加速 `s`，理想请求加速上限约为 `1 / ((1-p)+p/s)`；实际还要减去新增 stage/handoff 和竞争。若 FFN 占 30%，局部快两倍，整请求理论上也只有约 1.18×。这只是诊断模型，不是预测承诺。

同时优化两个阶段时不能把各自倍率相乘；缓存改变工作集后更必须重测 share / bucket / 是否启用 ANE。首步长 prefill 和后续 KV-hit 的短序列应分开。

## 5. 何时保留、何时撤回

- 有机制正确性但没有净收益：保留测试和简短负结果，可把实现留在明确 research 入口，不给产品堆开关。
- 降内存但稍慢：作为独立低内存策略，报告延迟代价；不称无代价加速。
- 某一机器有效：device/内存/workload allowlist，未测机器 GPU fallback。
- 只有一次观察：标 pilot；排序反转或 competing-load 检查失败，不升默认。
- 单层画质好、最终图差：做逐层 / 逐 step 误差分解，不能不断调松阈值直到绿色。

## 仓库证据

- [性能进度](../../docs/status/performance-progress-2026-10-10.md)：组件、整图和默认选择分离。
- [Native SGMMA](../../docs/status/2026-10-08-native-lora-sgmma-e2e.md)：完整 FFN 局部收益，但同组 native 整请求仍慢于 MLX。
- [SeedVR2 驻留](../../docs/design/seedvr2-block-weight-residency.md)：DiT active 下降不是完整请求容量资格。
- 下一步：[新模型接入](new-model.md)、[测量](../profiling/measurement.md)。
