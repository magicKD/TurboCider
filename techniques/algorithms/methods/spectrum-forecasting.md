# Spectrum：Chebyshev ridge 特征预测与实际时间坐标

<!-- knowledge
id: technique-algorithms-methods-spectrum-forecasting
type: method
stage: dit
operator: sampling, gemm
backend: cpu-reference, mlx, metal
device: cpu, gpu
symptom: stale-cache, precision-loss
semantics: approximate
representation: bf16, fp32
boundary: producer-consumer, request
api: reference-only
-->

[算法入口](../README.md) · [TaylorSeer](taylorseer.md) · [状态反例](../case-studies/forecast-state-audit.md) · [实验](../../examples/README.md)

来源：作者固定版本 `basis_utils.py` 与FLUX接入 [S32](../sources.md#s32)。这是训练外特征预测，不是对latent做FFT，也不等同于 [SeaCache](spectral-magnitude-cache.md) 的光谱indicator。

## 核心数据流

把若干full计算的 `(time,feature)` 保存成历史窗口；时间映射到τ后构建Chebyshev设计矩阵X，求

`C = (XᵀX + λI)^-1 XᵀH`，预测 `H_hat = x(τ_future) C`。

源码用FP32 Cholesky解小型正规方程，支持很多feature RHS；coefficient和时间/部分basis按BF16保存。窗口变化才失效重拟合，cache hit重用coefficient。因此“每步做一次全特征SVD”不是它的计算结构。

这里的“spectral”指时间函数的多项式基展开，不是频域attention稀疏或权重低秩。M字段表示最高degree时，实际P=M+1列；不要把README的basis数量文字与源码维度混用。

## 实际接入并不只有一个多项式

FLUX forward保留embedding、当前time/condition与tail；full时执行block栈，非full时用预测替代主体后hidden。refresh由warmup、当前window size和flex增长控制；README还建议Chebyshev预测与局部趋势预测混合。它不是当前样本的误差gate决定hit。

`Spectrum.predict` 在该快照中要求显式w，用 `(1-w)*local + w*cheb`；一些 `prefer/alpha/ema` 字段存在不代表当前预测分支真的使用。研究实现时应按被调用语句追踪，不按constructor参数名想象能力。

## 四个容易被注释掩盖的条件

1. `_taus` 注释描述history endpoints，但实际固定使用0和50。新日程、反向sigma和超过50的index必须重做坐标合同。
2. `ready()`直接返回True，constructor的K≥M+2约束只是容量，不是当前已收集的观测数。早期fit依赖ridge，并未等待足量samples。
3. `_H_buf`首次保存没有明确clone；若调用者原tensor被原地复用，历史可变。后续cat分配不能证明首份快照独立。
4. BF16 coefficient、fixed clock、ridge和有限历史是数值配方；论文的理想近似误差分析不能直接替它们担保。

CPU固定源码探针确认t=75映射τ=2、T4(2)=97；这展示区间外基函数增长，不是证明当前50步模型一定失效。首份快照alias也有独立可重放反例，见[状态审计](../case-studies/forecast-state-audit.md)。

## Apple 实施取舍

H有F个元素、K个历史、P个basis：主要成本是K×F历史和P×F系数、构建XᵀH及prediction的读写，不是P×P Cholesky本身。可在CPU求小矩阵/投影权重，将最终线性组合留GPU；不要为小求解器把所有H回读CPU。

MLX/MPS的Cholesky、BF16 GEMM与所需shape需单独验证；原代码有CUDA入口不等于整个包可直接切mps。ANE可以尝试固定P的预测图，但full/predict图边界、state ownership与频繁调用成本要计；本轮没有其设备速度数据。

比较replay/一阶/Spectrum时先固定相同full-step表，再分别按相同质量预算优化refresh。否则无法判断收益来自更好的forecast，还是仅来自少跑几次网络。应记录reset、history bytes、fit/eval分别耗时，以及未参与调参的motion/detail压力集。

论文：[Adaptive Spectral Feature Forecasting, 2603.01623](https://arxiv.org/abs/2603.01623)。作者README的最高倍率不作为Apple预测。
