# SeaCache / MagCache：光谱信号与幅值先验决定是否重算

配套：[机制 witness 与证据合同实验](../labs/algorithm-research.md)。这是支持性小实验，不是本方法的完整模型复现。

[算法入口](README.md) · [TeaCache](teacache.md) · [TaylorSeer](taylorseer.md)

来源：固定版本 Diffusers `sea_cache.py`、`mag_cache.py` 与 cache 文档 [S25](sources.md#s25)。分类：`approximate-train-free`。本轮核查的是 Diffusers 集成，不宣称完整复现原论文所有算法或支持所有视频模型。

## SeaCache：观察输出轨迹的视觉信号

Spectral-Evolution-Aware indicator 用 sigma 相关的频率响应过滤 raw vision latent，再比较跨 step indicator 的相对变化。低于累计阈值时复用或一阶外推 transformer residual，否则 full refresh。

本次 `_apply_sea_filter` 将 `(C,T,H,W)` 移到 `(T,H,W,C)`，FP32 FFT 只沿时空轴，不沿 channel；按频率的 clean-signal power prior 和 sigma 构造 gain，并密度归一为 unit mean 后 IFFT。不能对任意 flattened hidden 做 FFT 就认为是同一算法。

集成用 `cache_context(step_index,sigma,num_inference_steps)` 区分 trajectory。Cosmos3 adapter 保留输出轨迹中的 clean conditioning frames，却排除只作为额外 hint 的独立视觉序列；文字不进入视觉 indicator。它们仍会影响 transformer，不是从条件中被删除。

遇到 metadata 缺失、非连续 step、layout 不匹配时 fail-open 到 full。warmup、末步、max-consecutive 和0/1阶 residual prediction 都是独立约束。源码的主要优化对象是 Cosmos3，Wan T2V generic repeated-block 是另一条有限路径；不能外推到 Wan I2V 或所有模型。

## MagCache：先校准 residual 幅值的逐步比例

以完整执行得到的 `output-input` magnitude ratio 为先验，累计 ratio/error 指标，决定跳过 repeated blocks 的时机；保留 early steps 和最大连续 skip。不是每一步先完整计算输出、再决定自己是否可以省掉。

`MagCacheConfig` 要求显式 ratios 或 `calibrate=True`，校准模式不跳过。源码的 `nearest_interp` 可将 ratio 曲线映射到不同步数；实现能插值，不代表 scheduler/LoRA 改变后曲线仍然准确。FLUX 默认 ratios 不能直接套 FLUX2、Qwen 或 LTX。

## Apple 上各自的代价

MagCache 在线 scalar 控制开销低，但依赖离线先验与分支配对；SeaCache 更直接观察 latent 变化，但 FP32 FFT/complex 工作区、频率表、host gate 同步可能不小。MLX/MPS 能否高效执行目标 FFT shape 与 dtype 必须实测；不能因为都是 tensor 运算就认定 GPU 不会 fallback。

对 Core ML/ANE，建议先在 GPU 计算 indicator，full/predict 固定分图；这只是设计建议。不要将每步大 tensor 回读 CPU 用 FFT 后再传回 ANE 称为“统一内存所以没有成本”。

## 比较方法

在相同 residual 边界、full count 和质量预算下对比 latent L1、TeaCache signal、SEA indicator、固定步 mask、Mag prior。额外记录每步 FFT 时间、indicator 内存和 invalid-metadata 次数。若 host/controller 成本占短日程较大比例，负结果应保留。

特别做 condition-only hint 与真正 output frame 的负控、sigma 缺失/越界、重复 step、CFG 分支错配和高 motion holdout。累计指标是 heuristic budget，不是最终 latent/RGB 误差证书。

论文入口：[SeaCache 2602.18993](https://arxiv.org/abs/2602.18993)；MagCache 作者代码：[Zehong-Ma/MagCache](https://github.com/Zehong-Ma/MagCache)。后者作为溯源入口，本轮未单独固定/审计其仓库。
