# 少步数、蒸馏、latent 压缩与 linear attention：改变产品配方的加速

<!-- knowledge
id: technique-algorithms-methods-sampling-and-distillation
type: method
stage: dit
operator: attention, sampling
backend: mlx, metal
device: gpu
symptom: precision-loss, launch-bound
semantics: approximate, requires-training
representation: bf16
boundary: request, producer-consumer
api: reference-only
-->

配套：[机制 witness 与证据合同实验](../../examples/README.md)。这是支持性小实验，不是本方法的完整模型复现。

[算法入口](../README.md) · [缓存](teacache.md) · [Tiny decoder](tiny-autoencoders.md)

来源：DPM-Solver/Diffusers [S25](../sources.md#s25)，LCM [S21](../sources.md#s21)，LTX-2-MLX [S22](../sources.md#s22)，Sana [S01](../sources.md#s01)。必须与“相同步数、相同权重的实现加速”分开报告。

## 三种不同改动

| 方法 | 是否更换学习参数 | 省什么 | 不能说成什么 |
| --- | --- | --- | --- |
| 更好的 ODE/SDE solver / schedule | 可不换 | 达到目标质量所需 NFE | 只换 kernel 的等语义加速 |
| LCM / consistency / guidance / trajectory distillation | 使用训练的 student/LoRA | 更少主体调用、可省 CFG 双路 | 任意 base checkpoint 设4步也相同 |
| 更强 latent 压缩、linear attention 等结构 | 需要训练/匹配 checkpoint | 减少 N 或消除 quadratic attention | 对现有 dense 模型的直接精确替换 |

solver_order 不是 NFE：多步法复用历史通常每步一次新前向，某些多阶段 solver 一步多次；CFG 的两路也可能 batched 或 sequential。应报告 actual model evaluations、branch evaluations、最终帧数/分辨率与端到端，不只采样器 progress bar。

Diffusers 的 DPM-Solver++ 文档建议 guided 情况较低 order，且某些 dynamic thresholding 不适合 latent-space 模型。Flow-matching 的 velocity/x0/epsilon 参数化、sigma shift 与原 VP diffusion 不同，不能只换 scheduler 类名。

## LCM 的“无需训练”如何理解

部署者使用已经训练好的 LCM/LCM-LoRA 可以不自己训练，但方法本身通过训练/蒸馏获得少步能力，属于 `requires-training` 来源。LoRA 权重与对应 scheduler、guidance 使用方法共同定义模型；不能把普通风格 LoRA 当作 acceleration LoRA。

LCM 作者仓库有 MPS 使用入口，证明方法不绑定 CUDA ISA；其具体 Apple 效果仍需同一模型条件的实测，而不是引用 A800 延迟。

## Apple 的实际工程边界

LTX-2-MLX 有 dev 半分辨率 CFG → latent upsample → distilled refine 的两阶段路径，分别使用不同模型绑定与步表。其 stage-1 TeaCache 校准只针对特定版本/日程，2.5 pack 显式拒绝旧 cache 参数，值得学习这种 fail-closed 策略。[S22](../sources.md#s22)

SANA 的高压缩 latent 和已训练 efficient/linear attention 降低网络成本；这解释为何 SANA-Video 优化配方偏 cache/kernel，而 dense Wan/LTX 还考虑 sparse attention。不能用线性注意力公式直接替换已有 softmax 权重而宣称质量保持。

[SpargeAttention2](spargeattention2.md)把Top-k/Top-p混合mask与attention/velocity蒸馏结合，是训练型稀疏路线；[SLA的公开实现对照](sparse-linear-attention.md)则展示sparse＋learned linear correction及global/complement分支的checkpoint差异。论文2的learned router/QAT不能仅凭README citation确认；本库CPU projection-only一步下降不等于完整模型微调或媒体验收。

## “少步 + cache”常有收益递减

蒸馏把重要变化集中到少数步，相邻特征未必接近；少到4/6/8步时 warmup 和最后 full 几乎吃掉可 cache 区间。`mlx-teacache` 的短 Klein 零命中负结果是具体例子。继续放宽 threshold 可能省掉关键语义，而非找到免费优化。

实验分别列 base 默认、base+新 solver、student 默认、student+算法；不给它们混用权重/步数的同语义速度排名。质量测试关注提示词遵循、分布多样性、视频运动、参考保真，并把任意换 checkpoint 导致的画风变化单列。

论文：[DPM-Solver 2206.00927](https://arxiv.org/abs/2206.00927)、[DPM-Solver++ 2211.01095](https://arxiv.org/abs/2211.01095)、[LCM 2310.04378](https://arxiv.org/abs/2310.04378)、[LCM-LoRA 2311.05556](https://arxiv.org/abs/2311.05556)。
