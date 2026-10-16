# TeaCache：以便宜信号控制昂贵主体，及其 MLX 移植

配套：[机制 witness 与证据合同实验](../labs/algorithm-research.md)。这是支持性小实验，不是本方法的完整模型复现。

[算法入口](README.md) · [缓存身份](caches.md) · [评测](evaluation.md)

证据：作者 FLUX 实现 [S02](sources.md#s02)，纯 MLX 控制器与 mflux adapter [S03](sources.md#s03)，Sana 的通用/模型边界 [S01](sources.md#s01)。分类：`approximate-train-free`；本轮读源码，不声称复现作者完整图像基准。

## 机制与执行顺序

每一步仍计算 embedding、首块调制输入 `m_t` 等 cheap prelude。控制器求

`d_t = mean(abs(m_t-m_(t-1))) / max(mean(abs(m_(t-1))), eps)`，
`a_t = a_(t-1) + p(d_t)`。

`p` 是拟合信号变化与主体变化关系的多项式，不是理论误差上界。累计指标低于阈值且不在强制 full 区间时，跳过主体并做 `h_out = h_in_current + R_last_full`；否则执行主体、保存 `R = h_out-h_in` 并清累计量。当前 timestep 的 tail / norm / projection 仍执行。

作者 FLUX 示例读取 `transformer_blocks[0].norm1` 输出，用一次主体残差覆盖 double/single block 主体。**直接返回旧 denoiser 输出**与**在当前输入上加旧 residual**不是同一算法。Sana 同时存在 `TeaCache` whole-step 控制器和 `TeaCacheResidual` 的 PRE_BLOCKS 接缝，必须检查实际模型 adapter 用了哪一种。

## MLX 实现值得吸收什么

`mlx-teacache/_kernel/gate.py::gate_step` 把结果拆成 `should_compute` 和 `should_update_cache`。threshold≤0 时不建立无用 residual，避免改变 MLX graph/buffer donation；leading/trailing 强制窗口与 cache 更新是独立语义。逐次 gate 的 signal anchor 更新，但 residual 只在完整计算时更新。

该实现还做了 FP32 reduction、一次 `mx.eval` 取两标量、非 finite miss 后清 residual、超校准范围强制重算、非负累计指标与连续跳过上限。最后几项是相对原始 TeaCache 的工程差异，不能只说“完全同款”。原多项式可能离开拟合范围后变负；仅 `max(0,p)` 并不够，仍可能令指标长期停在零。

`variants/flux1_dev/integration.py` 将 prelude/body/tail 分开，代理原 transformer 并管理 request 生命周期；CFG 的正负分支需要各自 residual。不能机械复制旧 mflux 私有方法签名到新版本。

## Apple 收益与限制

跳过的是实际大型算子和权重读取，所以不要求 CUDA 特有指令。代价是信号 reduction、host scalar 同步和完整 hidden residual 的驻留。MLX lazy graph 必须 materialize cache，避免保留整条前向图；MPS 的 `.item()` 也会产生同步。Core ML 可导出 full-body 与 replay/tail 分图，但图间 materialization 与调用成本必须测。

作者在 M1 Max 的 FLUX.1-dev 25 步示例报告跳过 6 步、约 1.57×（三次冷进程中位数口径）；这是上游报告，不是本机结果。其短蒸馏 Klein 日程为零跳过，另一个旧的 2.68× Klein-base 说法被上游更正为冷进程约 1.37×，原因是 allocator 冷暖不对称。**wrapped-no-gate / wrapped-gated / 原生 vanilla 三路比较**比只跑“前后两次”更有信息。[S03](sources.md#s03)

Qwen-Image、Qwen-Image-2512 与 Qwen-Image-2.1 不能共享校准资格；该扩展 README 明确不支持后者。不要把上游 Qwen 数字套到本库 2.1。

## 建议实验与否决

固定 weights、precision、scheduler、guidance、seed、reference；先验证 threshold=0 的同路数值控制，再比较默认 gate、保守阈值、连续跳过上限和校准范围拒绝。记录 signal/预测值/full/replay 的逐步日志与 actual block count。

独立 prompts 应包含文字、手、人脸、镜头运动和条件突变；拟合数据与 holdout 分开。换 LoRA、step 数、scheduler shift、image strength 时重查信号分布。零命中仍慢是有效负结果；SSIM 高也不能把文字改变叫无损。

论文：[TeaCache, arXiv:2411.19108](https://arxiv.org/abs/2411.19108)。作者扩展的校准、短日程和 MLX oracle 笔记见 [S03](sources.md#s03)。
