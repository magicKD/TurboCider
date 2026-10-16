# 稀疏、时序重用与近似加速的验收

[总目录](../README.md) · [Attention](../metal/attention.md) · [VAE](../media/vae.md)

逐算法机制与开源实现见 [算法目录](README.md)、[模型sparsity地图](model-sparsity.md)与[固定来源](sources.md)；详细实验合同见 [evaluation](evaluation.md)。

## 先区分“不改语义”和“可接受误差”

改变dtype/reduction、W8A8、DBCache、稀疏attention、少参考分辨率、final-step FFN reuse、独立VAE tiles，都是不同近似。`allow_approximation=true`只代表允许尝试，不代表画质合格，更不代表可以漏步骤或输入。

固定noise/steps、同weights/adapter的优化作一类；更少steps/蒸馏checkpoint/更小参考图作另一类产品取舍。报告不可混成同语义kernel加速。

## 分层质量矩阵

| 层级 | 检查 | 不足 |
| --- | --- | --- |
| 数值安全 | shape、dtype、finite、padding、source | 不保证正常主体/细节 |
| component | abs/relative L2/cosine、独立FP32/FP64 witness | 单层误差不代表trajectory |
| trajectory | conditioning/noise、逐step velocity/latent | latent接近不保证decoded视觉/时间 |
| 图像 | decodedRGB/alpha、SSIM/PSNR/LPIPS、whole/detail观察 | 单prompt高分不代表语义一致 |
| 视频 | 每帧、最差帧、运动/光流、身份、边界/闪烁 | motion平均值不代表lip-sync |
| 音频 | duration、sample rate、波形/听感、AV同步 | mux成功不代表声音正确 |

阈值先固定，保留未通过指标。constant/near-zero tensor的相对误差与cosine要定义，不能NaN偷偷算pass。感知模型有自己的版本/权重/设备与离线依赖，不能在评测中静默下载。

## 近似组合不继承单项资格

DBCache×W8A8、prefix reuse×首步ANE、稀疏×ConvRot、FP16 LoRA×deferred join，都会改变误差和时序。先各自单变量，后固定组合重新测；不把单项速度倍率相乘，也不把单项质量预算相加当保证。

## 什么时候否决

- 组件快但完整phase/E2E没有达到预设净收益。
- 最终latent/RGB gate失败，即使视觉有限样本“很像”。
- 同prompt排序反转、竞争负载检查失败或样本不足却想自动推广。
- 媒体缺帧/变FPS、silent crop、掩盖失败/重试、source或shape不完整。

[LTX sparse](../../docs/design/ltx-stage2-sparse-results.md)是完整负结果：多个策略实际实现，core快但质量/阶段目标仍未同时达成。结论应是默认dense、保留研究，不是“所有稀疏无效”，也不是“已找到可上线稀疏”。

## 更有效的下一次实验

按layer/step/phase定位敏感位置，保留conditioning/anchors/exact边界；用真实中晚层输入筛查，再跑holdout。近似策略的目标应是一个清楚的速度/质量/内存Pareto点，而不是越过一个被不断放松的分数。

工具入口见 [质量与profile工具](../profiling/tools.md)，记录模板见 [experiment](../templates/experiment.md)。

## 实现参考与实验

可读实现：[参考代码](../labs/caches.py)；[实验步骤与边界](../labs/cache-and-composition.md)；[本项全部 case / controls](../labs/catalog.json)。

```sh
python3 -B techniques/labs/run.py token-merge
```

本项控制：mass/position and actual approximation error；off identity and combination seam conflicts。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：new error budget, holdout set, route or model topology。历史负结果限定于其来源/shape/边界，不关闭该方向。
