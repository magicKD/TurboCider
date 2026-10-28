# TaylorSeer：预测特征，而不只是复用旧特征

<!-- knowledge
id: technique-algorithms-methods-taylorseer
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

[算法入口](../README.md) · [DBCache](block-caches.md) · [PISA 的另一种 Taylor](piecewise-attention.md)

第二轮对照：[Spectrum的全窗拟合](spectrum-forecasting.md)、[FoCa/DMD的预测与fallback](forecast-calibrators.md)和[实际源码状态反例](../case-studies/forecast-state-audit.md)。数学公式、历史更新与fallback要分别验证。

来源：作者 `taylorseer_utils` [S07](../sources.md#s07)、Cache-DiT calibrator [S06](../sources.md#s06)、Diffusers hook [S25](../sources.md#s25)。分类：`approximate-train-free`。

## 数学与状态

用最近若干次完整计算的 feature，差分估计 `D1=(f_now-f_prev)/Δs`，再差分估计更高阶。跳过计算时预测

`f_hat(s+δ)=D0 + δ D1 + δ² D2/2! + …`。

feature 可以是 attention、MLP 或聚合 residual，边界取决于 adapter。需要真实 refresh 才能更新导数；不能把自己的预测当成新观测。每个 layer/module/CFG/stage stream 保留独立状态；m 阶大致要保留 O((m+1)ND) 数据，更新临时量还可能增加峰值。

作者 FLUX 实现按完整计算的 step 间隔 `activated_steps` 差分；代码注释给出 timestep 坐标的替代方式，但不是默认已启用。step-index 与 sigma/time 的平滑性不同，换非均匀 scheduler 后不能只更换一个 `δ` 而继续使用旧导数。

Cache-DiT 中 `update` 交换历史梯度 ladder，`approximate` 使用最近完整 anchor 与 elapsed；Diffusers 提供独立 cache context/hook。**三个实现的导数更新顺序、warmup、period 定义必须逐一核对，不只比较类名。** C++ 移植也要独立验证 anchor 与历史 buffer，不能因公式相同就跳过线性函数 oracle。

## Apple 路径

减乘加、difference 和绕过 block 都可在 MLX/MPS 执行；低阶 prediction 可以融合成一次 streaming kernel。瓶颈很可能是多份 feature 的统一内存带宽，不是多项式的乘法数。先试一阶加短 refresh，不默认阶数越高越可靠。ANE 固定 full/predict 分图的表达可行性需另测，频繁小图回调可能得不偿失。

与 DBCache 的组合可让 gate 决定是否省 M，calibrator 决定 R_M 用 replay 还是 extrapolation；两者不是二选一，也不是可无限叠加。cache 量化会在相邻值差分时放大舍入噪声，应把导数状态 dtype 作为独立变量。

## 优于 / 劣于 replay 的条件

变化平滑且有趋势时，一阶可减少 stale feature 的滞后；转折、快速运动、guidance 放大或稀疏切换时，高阶项可能外推爆炸。较短日程没有足够 warmup/history，预测不一定有意义。作者提到的“lossless”是任务指标接近，不是逐元素或同 seed 图像完全相同。

最小实验使用恒定、线性、二次函数与非均匀采样间隔验证更新和 anchor，再用突变/正弦轨迹展示预测反而更差。真实模型做 order=0/1/2、refresh 2/3/5、不同 state dtype 的冻结 sweep，记录 full count、内存、逐步误差与完整媒体指标。

论文：[From Reusing to Forecasting, arXiv:2503.06923](https://arxiv.org/abs/2503.06923)。注意本次读取的 Diffusers cache 文档曾把论文链接写成 `2403.06923`；这里以作者仓库/论文为准，不传播该笔误。
