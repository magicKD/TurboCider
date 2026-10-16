# EasyCache：在线估计变换变化，不预先拟合 TeaCache 多项式

配套：[机制 witness 与证据合同实验](../labs/algorithm-research.md)。这是支持性小实验，不是本方法的完整模型复现。

[算法入口](README.md) · [TeaCache](teacache.md) · [FBCache / DBCache](block-caches.md)

来源：作者 Wan 2.1 代码 [S04](sources.md#s04)、stable-diffusion.cpp 的 C++ 移植 [S08](sources.md#s08)、Sana 文档与模型接入 [S01](sources.md#s01)。分类：`approximate-train-free`。

## 怎样省计算

用已完整计算过的输入/输出差异，在线估计输入变化会造成多大输出变化。随后 cheap 地观察当前 latent / transform 输入的变化，累计预测变化；低于阈值时在当前输入上复用旧 transformation/residual，否则全算并更新变化率。相比 TeaCache 的模型拟合多项式，它把增益估计放在运行时；相比 FBCache，它不要求每次先跑完整首个 transformer block。

这是机制概括，不同 adapter 的归一化与信号定义不完全相同，阈值不能互换。C++ `easycache.hpp` 中 `relative_transformation_rate`、`last_input_change`、`output_prev_norm` 形成累计指标；`before_condition` 决策，`after_condition` 更新完整结果。缓存仍按条件分支保存，不是让正负 prompt 共用一个向量。

作者 Wan 脚本按 cond/uncond 调用对管理计数：`num_steps = sample_steps*2`，warmup/cutoff 也与分支调用数关联。两路 residual 不同；不能把计数解释为采样器真正走了两倍步数，也不能跳过其中一路后让 pair 错位。

## 为什么 Apple 可以迁移

核心是 reduction、少量状态和绕过大段计算，MLX/MPS 都可表达；不依赖其 CUDA 前向内核。但原始脚本还包含 CUDA AMP、同步与分布式 setup，**数学可移植不等于原脚本可在 MPS 直接运行**。

C++ 移植提供另一个边界：controller 与 tensor cache 数据在 host `std::vector<float>`；它证明控制逻辑可移植，但不证明目标 Metal 端完全没有回读。需要沿 denoiser 输出到 `sd::Tensor<float>` 的路径检查同步/转移，或在 MLX 中做 device-resident residual 加轻量 indicator。统一内存不消除 CPU/GPU fence。

## 与其他 cache 怎么选

| 条件 | 候选 | 代价与风险 |
| --- | --- | --- |
| 缺校准数据、较长 base 日程 | EasyCache | 在线增益 warmup，变化突增时可能低估 |
| 调制信号便宜且已校准 | TeaCache | 多项式迁移风险、同步开销 |
| 首段能预测后段且愿付首段成本 | DBCache | 每个 hit 仍执行前/后段 |

Sana 对 SANA-Video 的 EasyCache 与 Cosmos 的 TeaCache 选择是特定模型方案，不是证明 EasyCache 通常优于 TeaCache。SANA 已是 linear-attention 网络，继续做 quadratic sparse attention 的空间与 dense Wan 不同。

## 质量与复现边界

作者 README 的 HunyuanVideo A800 示例是 1124.30→507.97 秒、约 2.21×，不是 Mac 基准，且不能同其他硬件/模型数值直接排序。应先跑同模型 full、controller-on/no-hit、正常 threshold 三路，分别统计 warmup、最后步、actual cond/uncond 完整调用及 residual 占用。[S04](sources.md#s04)

否决条件：branch cache 混用、非连续 step 重用、变化率异常后继续 hit、zero/near-zero denominator 导致错误判断、完整请求收益被 staging 抵消、视频快速运动/切镜头失败。在线阈值是经验控制，不提供像素误差的硬保证。

论文：[Less is Enough, arXiv:2507.02860](https://arxiv.org/abs/2507.02860)；作者站点：[EasyCache](https://H-EmbodVis.github.io/EasyCache/)。
