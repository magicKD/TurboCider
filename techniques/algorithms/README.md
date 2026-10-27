# 图像 / 视频生成的可控有损算法加速

<!-- knowledge
id: technique-algorithms-readme
type: navigation
-->

[百科入口](../README.md) · [实现技巧与最小验证](implementation-patterns.md) · [原始来源与源码阅读账本](sources.md) · [实验与质量合同](evaluation.md)

范围是 Apple Silicon 单机生成推理，以及可迁移的 CUDA 算法和 MLX/MPS/Metal 设计。这里保存方法、固定源码和历史案例，不分发隔离探针或实验证据，也不给近似方案授予上线资格。

## 怎样使用这些材料

本目录是一份学习手册，不是实验工作目录。推荐顺序：本页结论 → [实现与分析技巧](implementation-patterns.md) → 按问题选方法 → 用案例理解取舍。原始数据和实验重跑工具不随库提供。

| 目的 | 入口 | 材料定位 |
| --- | --- | --- |
| 学机制、比较取舍、寻找可迁移方案 | [方法综述](methods/README.md) | 按缓存/预测、attention、token、量化和采样组织，不按研究轮次读 |
| 学分析过程：正反控、强分母、状态和性能归因 | [分析案例](case-studies/README.md) | 保留完整推理链与负结果；不是preset推荐榜 |
| 实现一个优化或设计实验 | [16条实现技巧](implementation-patterns.md) · [评测合同](evaluation.md) | 说明省掉什么、依赖什么、怎样证伪，以及速度与质量如何分别验收 |
| 查论文、原作者源码、固定版本 | [来源账本](sources.md) | 追溯阅读身份与运行范围，不以仓库存在推导支持 |

## 先看结论

下面从源码与历史案例提炼可迁移的认识。案例数字按各自模型、环境和计时边界阅读，不是当前验证、通用阈值或方案排名。

1. **优先学习绕过整段计算，而非照搬某个阈值。** block/residual cache最容易保留现有优化GEMM/SDPA；原生Qwen完整请求和DeepCache MPS已有净收益，但都有显著质量取舍。[Qwen完整请求](case-studies/native-cache-request-study.md) · [DeepCache完整图像](case-studies/deepcache-sd15-mps-quality.md)。这是当前研究优先级，不是所有模型上最快的定理。
2. **有提速，不等于损失已受控。** DeepCache i5约2.91–3.00×完整请求收益，但32个有损计分输出全部未过两个预定RGB预算；full本身也不是语义groundtruth。控制器的threshold/ratio/period是调节参数，不是全输入误差上界。[质量拒绝与原图指标](case-studies/deepcache-sd15-mps-quality.md) · [幅值预算反例](methods/magcache-calibration-and-consumer.md)。
3. **先消除控制器的浪费，再增加近似。** 固定唯一时钟下，VDE去guard与DeepCache ordinal变体分别保留原近似轨迹的byte等值；收益只适用于各自已验证的日程/调用合同，不能迁移倍率，也不能自动推广到重复timestep、多阶段solver。[VDE host依赖](case-studies/vde-mps-controller-cost.md) · [DeepCache时钟](case-studies/deepcache-sd15-mps-quality.md)。
4. **稀疏和token减少必须真的少算，并覆盖恢复成本。** mask不保证跳过QK/PV；selector、summary、gather和restore都要进窗口。Sol真实Qwen短序列未取得稳定低误差收益；ToMeSD迁移到Qwen FFN虽能提速却有明显长尾，且不是原默认self-attention方案。[Sol组件](case-studies/sol-mps-component-study.md) · [token merge消费者](case-studies/token-merge-consumer-and-real-ffn.md)。
5. **容量优化与计算优化分开选。** 本轮warm resident真实BF16 FFN中，所有packed/低秩候选都慢于compiled dense。W4A16加分离低秩不等于Nunchaku CUDA W4A4融合；可借鉴outlier分解和校准，但Apple收益需重新测。[真实FFN负结果](case-studies/real-ffn-gpu-study.md) · [低秩/转换语义](methods/low-rank-quantization.md)。
6. **状态与测量设计本身就是可复用成果。** request-local、分支语义身份、owned history、完成后事务提交、强分母和配对比值，比复制preset更可迁移；组合必须另测，不能乘倍率。[实现技巧](implementation-patterns.md) · [评测合同](evaluation.md)。

## 按学习目标读源码

固定链接来自[来源账本](sources.md)，不用会漂移的`main`替代阅读版本。每条路线都同时给出“学什么”和“不能照搬什么”；作者实现、本库变体、CPU替身和本机结果在专题内分别标明。

| 学习目标 | 固定源码入口 | 本地解读 / 可借鉴设计 / 限制 |
| --- | --- | --- |
| 整块跳算与保留浅层更新 | [DeepCache helper](https://github.com/horseee/DeepCache/blob/1691a3ab22308fe06685e4ff151a9ab175be2dc4/DeepCache/extension/deepcache.py) · [S70](sources.md#s70) | [U-Net边界](case-studies/deepcache-unet-boundaries.md)：沿down/mid/up画出实际省算；先做interval1等值和enable/disable恢复，不复制branch编号到MLX |
| 低成本gate与生成事务 | [MLX TeaCache gate](https://github.com/IonDen/mlx-teacache/blob/b56f2882ca237b3b971c976f78804154cbeb1014/src/mlx_teacache/_kernel/gate.py) · [S42](sources.md#s42) | [CFG/lifecycle](case-studies/mlx-teacache-lifecycle.md)：两分支独立residual、active窗口、失败不提交；CPU包络不是完整Qwen画质 |
| 减少网络调用后的host成本 | [VDE FLUX](https://github.com/Tan-Junwen/VDE/blob/3d08e45de4378ba0e4848c0dfa6c7216b4311f2c/VDE4FLUX/inference_flux1.py) · [S67](sources.md#s67) | [分解/反例](methods/vde-input-adaptive-velocity.md) → [MPS成本](case-studies/vde-mps-controller-cost.md)：把唯一日程证明放到setup；重复时间与近零norm另测 |
| CFG与频谱补偿的正确对象 | [FasterCache CogVideoX](https://github.com/Vchitect/FasterCache/blob/02d05ef7edb48bdec01d2b4df6edfe7f472a41b9/scripts/cogvideox/fastercache_sample_cogvideox.py) · [S46](sources.md#s46) | [消费者](case-studies/fastercache-consumer-study.md) → [空间residual](case-studies/spatial-cache-mlx-study.md)：追实际uc/c身份、FFT轴与线性变换；不是完整视频GPU结果 |
| 稀疏内核与路由成本 | [Sol Metal](https://github.com/NVlabs/Sana/blob/d0c0a4685ab5dc2336d18b7213d85f13def92418/techniques/sparse_backends/sol_attn/mps/metal.py) · [S01](sources.md#s01) | [Sol机制](methods/sol-attention.md) → [真实成本](case-studies/sol-mps-component-study.md)：summary进入softmax分母、sink/tail与all-exact控制；不能只测mask密度 |
| merge→短计算→恢复 | [ToMeSD merge](https://github.com/dbolya/tomesd/blob/27a14a372beecaa85c101fb588631643f379b8ce/tomesd/merge.py) · [patch](https://github.com/dbolya/tomesd/blob/27a14a372beecaa85c101fb588631643f379b8ce/tomesd/patch.py) · [S48](sources.md#s48) | [路由/FFN研究](case-studies/token-merge-consumer-and-real-ffn.md)：区分共享网格与共享匹配、slot与原坐标；原默认只merge self-attention |
| outlier分解与量化部署 | [DeepCompressor校准](https://github.com/nunchux-ai/deepcompressor/blob/69f3473f5e1c1504bae35cc50c7858ef900a9b17/deepcompressor/calib/lowrank.py) · [S43](sources.md#s43) | [真实权重](case-studies/real-qwen-lowrank.md) → [完整FFN](case-studies/real-ffn-gpu-study.md)：分开fit目标、holdout与runtime；不从低bit推导更快 |
| Apple原生图与读回 | [Draw Things TeaCache.swift](https://github.com/drawthingsai/draw-things-community/blob/8d9c62ad38539ef8bb123f8b8e78cab76caafd0c/Libraries/SwiftDiffusion/Sources/TeaCache/TeaCache.swift) · [S53](sources.md#s53) | [原生控制器](methods/draw-things-caching.md) → [gate强对照](case-studies/nnc-gate-controller-cost.md)：合并小读回、不可变anchor/代际；少mean不保证净收益 |

更多方向按完整专题地图查找：预测见[Spectrum](methods/spectrum-forecasting.md)/[FoCa/DMD](methods/forecast-calibrators.md)，动态周期见[AdaCache](methods/adacache-motion-and-period.md)，空间降算见[HiDiffusion](methods/hidiffusion-spatial-compute.md)，attention/CFG阶段见[T-GATE](methods/tgate-phases-and-branches.md)。[实现技巧页](implementation-patterns.md)将这些材料转为“怎么借鉴、怎么验证”的检查表。

## Apple实测证据：哪些能用来作决策

下表各行的模型、精度和计时边界不同，**不能按倍率排序或相乘**。完整生成执行成功与质量预算通过是两个独立状态，详见[证据分层](evaluation.md)。

| 代表性研究 | 已有证据与结论 | 尚未证明 / 使用边界 |
| --- | --- | --- |
| [原生Qwen cache](case-studies/native-cache-request-study.md) | 真实完整生成约1.21–1.46×；自行车balanced/fast SSIM约.817/.803 | 两prompt不覆盖任务分布；alpha/语义预算仍需明确 |
| [DeepCache SD1.5 MPS](case-studies/deepcache-sd15-mps-quality.md) | 四cohort、48计分完整请求；i5约2.91–3.00×，全部有损输出拒绝预设预算 | fp16文件转FP32计算、特定DDIM；不是可直接采用的preset |
| [VDE MPS](case-studies/vde-mps-controller-cost.md) | 随机真实FLUX共同50步loop；去guard相对原版约1.41–1.55×且byte等值 | 无预训练/编码器/VAE；不是媒体E2E或画质资格 |
| [T-GATE MPS](case-studies/tgate-mps-resident-cost.md) | 随机SD完整latent窗口；相对CFG-only强分母fi2约1.047–1.069×，两组IQR跨1 | 部分单项反收益；同配方CPU parity不等于对full的质量 |
| [Sol attention](case-studies/sol-mps-component-study.md) | 原Metal组件与MPS/MLX dense对照；真实Qwen短序列无稳定低误差收益 | 合成长序列收益伴明显误差，未验证生成图像 |
| [ToMeSD/Qwen FFN-local](case-studies/token-merge-consumer-and-real-ffn.md) | 50%移除约1.81–1.86×组件收益，早/中层误差大 | 非默认ToMeSD attention；真实SD1.5媒体仍待验证，源码采集不算执行 |
| [真实BF16 FFN量化/低秩](case-studies/real-ffn-gpu-study.md) | 全部packed/低秩候选慢于同cohort compiled dense | 只否定此warm resident配方，不否定容量/offload或其它融合内核 |
| [频谱差→空间cache](case-studies/spatial-cache-mlx-study.md) / [NNC gate](case-studies/nnc-gate-controller-cost.md) | 前者约1.16–1.78×组件收益但保留高偏置拒绝；后者对强读回分母无稳定通用收益 | 固定线性变换/anchor合同；均非完整模型或峰值内存证明 |

Apple实现也有可读入口：[MLX TeaCache](methods/teacache.md)、[Sol Metal](methods/sol-attention.md)、[FastVideo MLX VSA](methods/structured-sparsity.md)。源码存在不等于本机质量通过；[MLX DeepCache port](case-studies/deepcache-unet-boundaries.md)固定tree没有LICENSE，许可待确认，仅静态阅读，不复制执行其body。

## 怎么选，先做什么

1. 先按 [模型地图](methods/model-sparsity.md) 确认 dense / joint / cross / linear attention、CFG、base/student、stage 和 latent 布局。不要凭名字套阈值。
2. 看 profiler 中 denoise、attention-core、FFN、VAE、权重 staging 的真实比例。若 attention 只占请求 20%，把 core 无限加速也只剩 1.25× 理论上限。
3. 多步 base 模型先试 request-local residual cache；短蒸馏日程从稀疏、量化或 VAE 入手，不能为制造 hit 一直提高阈值。
4. attention 占比高且长序列时，先 all-exact 内核控制，再逐步放松 density / tau，保留文字/参考/音频与敏感层。先比较局部窗口，再为动态补偿付额外工程成本。
5. 存不下模型时先研究 weight-only Q4/Q6/Q8 与敏感层回退；已经完全驻留后，内存下降不自动意味着更快。
6. 单项过门槛后再组合。缓存使稀疏少跑几次、量化改变路由输入、token merge 改变 shape，收益不能相乘。

若允许微调，训练型路由与sparse-linear是另一条路线，而不是base checkpoint上的免费开关。先绑定权重、feature map、归一化scope、bias和量化训练recipe；验证all-selected与native backward，再讨论训练预算、Apple后端和holdout媒体。

这是实现难度与证据驱动的研究顺序，不是保证最快的通用预设。ANE 更适合固定 shape 的计算段；动态选择与稀疏循环优先在 GPU 做。MLX GPU 不是 ANE。

## 案例与实现参考

- 优先阅读：[完整请求Qwen](case-studies/native-cache-request-study.md)、[DeepCache质量取舍](case-studies/deepcache-sd15-mps-quality.md)、[VDE控制器](case-studies/vde-mps-controller-cost.md)、[T-GATE强分母](case-studies/tgate-mps-resident-cost.md)、[FFN反收益](case-studies/real-ffn-gpu-study.md)。学习它们怎样构造假设、比较方案和限制结论，不要求拥有原实验包。
- 小型代码见[实现参考](../examples/README.md)，固定原作者实现见[来源索引](sources.md)。没有统一实验 runner 或原始 JSON。

材料范围与贡献方式见[维护规则](../MAINTENANCE.md)。

## 适用边界

本目录提炼的是机制与分析方法，不是跨模型的通用 preset。组件、随机模型、真实权重和完整媒体观察分别阅读；质量拒绝不会因更快而消失。迁移前提和重开条件见具体方法页，材料边界见[维护规则](../MAINTENANCE.md)。
