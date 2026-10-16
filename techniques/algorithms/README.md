# 图像 / 视频生成的可控有损算法加速

[百科入口](../README.md) · [原始来源与源码阅读账本](sources.md) · [实验与质量合同](evaluation.md)

研究日期：2026-10-16。范围是 Apple Silicon 单机上的生成推理；读过路线图第 11 节后，扩展至 CUDA 参考算法、MLX/MPS/Metal 实现及本库已有正反实验。这里交付的是学习报告与隔离探针，**没有修改产品默认或给任一近似方案授予上线资格**。

## 先看结论

最容易跨硬件迁移的是“少执行整个 block / 分支”的缓存策略，其次是 MLX 已有 packed QMM 上的逐层混合精度。稀疏 attention 的收益最依赖实际内核：把大量位置设成 mask，不等于跳过其矩阵运算；动态 gather、聚类、路由和小 kernel 提交也可能抵消节省。SVDQuant 的数学可移植，但 Nunchaku 的 CUDA W4A4 融合不能直接变成 Metal/ANE 支持。这是对下列源码和成本模型的综合判断，不是跨项目排行榜。

Apple 路径并非全都停留在设想：`mlx-teacache` 有纯 MLX 的 gate 与 mflux adapter；Sana Sol-Attn、ComfyUI-SolAttn-MPS 有 Metal 稀疏内核；FastVideo 有独立 MLX H3 VSA 路径。对应实现仍有版本、shape、质量和 fallback 限制，见各页；“源码存在”“作者实测”“本轮实测”分开记录。

## 技术报告地图

| 要解决的冗余 | 报告 | 省掉什么 | Apple 状态 / 首要风险 |
| --- | --- | --- | --- |
| 不变条件反复算 | [精确 cache 与身份](caches.md) | encoder、编译、条件 KV | 先核对依赖；不是跨步生成 KV 都不变 |
| timestep 相邻变化小 | [TeaCache 与 MLX 移植](teacache.md) | transformer 主体 | 已有 MLX；多项式换模型必须重校准 |
| 无离线拟合的整路重用 | [EasyCache](easycache.md) | denoiser / transform 向量 | 控制器可迁移；在线增益不是误差证明 |
| 前段预测后段 | [FBCache / DBCache](block-caches.md) | 中间 / 剩余 blocks | 本库 Qwen 有历史完整请求证据 |
| 历史特征预测未来 | [TaylorSeer](taylorseer.md) | attention / FFN / block | 数学通用；导数状态、坐标与外推漂移 |
| 分类型固定广播 | [PAB](pab.md) | 部分 attention / MLP | 易静态分图；长上下文缓存占内存 |
| U-Net 深层重用 | [DeepCache](deepcache.md) | 深层 down/mid/up 分支 | 保留浅层更新；不直接套到 DiT |
| CFG 双分支冗余 | [FasterCache](fastercache.md) | attention、部分 unconditional | FFT、guidance 放大和 branch 配对 |
| 光谱 / 幅值决定缓存 | [SeaCache / MagCache](spectral-magnitude-cache.md) | transformer 主体 | 先审 adapter / 校准；不移植默认阈值 |
| 稀疏性的不同含义 | [模型 sparsity 地图](model-sparsity.md) | 分类入口 | 不捏造“某模型固定 90% 稀疏” |
| 局部 / tile / learned routing | [STA / VSA / 窗口](structured-sparsity.md) | 未选择的 QK、PV tiles | MLX 原型可做；VSA gate 需对应权重 |
| 动态阈值 + 远端补偿 | [Sol-Attn / Metal](sol-attention.md) | 低分块精算 | 已有 Metal；精确 sink、tail、真实命中 |
| 精确或近似而非删除 | [PISA](piecewise-attention.md) | 非关键块 token 级精算 | 需新 Metal kernel；高阶摘要额外成本 |
| 时空 head / 语义聚类 | [SVG / SVG2 / SVG-EAR](svg.md) | 非关键连接 | 聚类、重排、变长块不能漏计 |
| 稀疏 + attention 量化 | [Sparge / SageAttention](quantized-attention.md) | QK、PV 的成本 | CUDA 低比特 ISA 不可照搬 |
| token 数目冗余 | [ToMeSD / pruning](token-reduction.md) | attention 及可选 FFN tokens | MPS workaround 存在；RoPE 与恢复非无损 |
| 权重/激活的数值冗余 | [PTQ / 混合精度](quantization.md) | 带宽、驻留、部分计算 | MLX / Core ML 路径分开；压缩≠整数加速 |
| outlier 导致低比特失真 | [SVDQuant / 低秩](low-rank-quantization.md) | 主分支低比特 | 分支融合、LoRA 与临时量不能漏计 |
| 求解器调用 / 训练结构 | [少步数、蒸馏与线性 attention](sampling-and-distillation.md) | NFE / 网络规模 | 改变采样或权重，单列产品取舍 |
| VAE 大模型解码 | [Tiny decoder](tiny-autoencoders.md) | 原 VAE 计算 | 预览优先；颜色、细节、闪烁与 ABI |
| 多技术组合 | [Sana sol-engine 源码导读](sana-sol-engine.md) | 模型各阶段不同部分 | 设计空间≠全部启用≠全部已测 |

## 怎么选，先做什么

1. 先按 [模型地图](model-sparsity.md) 确认 dense / joint / cross / linear attention、CFG、base/student、stage 和 latent 布局。不要凭名字套阈值。
2. 看 profiler 中 denoise、attention-core、FFN、VAE、权重 staging 的真实比例。若 attention 只占请求 20%，把 core 无限加速也只剩 1.25× 理论上限。
3. 多步 base 模型先试 request-local residual cache；短蒸馏日程从稀疏、量化或 VAE 入手，不能为制造 hit 一直提高阈值。
4. attention 占比高且长序列时，先 all-exact 内核控制，再逐步放松 density / tau，保留文字/参考/音频与敏感层。先比较局部窗口，再为动态补偿付额外工程成本。
5. 存不下模型时先研究 weight-only Q4/Q6/Q8 与敏感层回退；已经完全驻留后，内存下降不自动意味着更快。
6. 单项过门槛后再组合。缓存使稀疏少跑几次、量化改变路由输入、token merge 改变 shape，收益不能相乘。

这是实现难度与证据驱动的研究顺序，不是保证最快的通用预设。ANE 更适合固定 shape 的计算段；动态选择与稀疏循环优先在 GPU 做。MLX GPU 不是 ANE。

## 证据与复现

- [来源账本](sources.md)：固定 commit、阅读到的具体文件、论文及作者文档；[机器可读清单](sources.json)保留内容 SHA-256。采集不等于阅读，更不等于复现。
- [评测协议](evaluation.md)：GPU 空闲检查、匹配基线、消融、trajectory / RGB / 视频质量、预算与否决标准。
- [本轮实验](experiments-20261016.md)：CPU 数学负控、MLX 探针及明确的测量边界；不把合成 QKV 误差解释为生成画质。
- [原有近似验收](approximations.md)、[算法接缝实验](../labs/cache-and-composition.md)保持有效；新页细化机制，不替代原有 lifecycle 合同。

本目录不打包上游模型、媒体或整套推理引擎。源码采集器 [source_probe.py](source_probe.py) 只产生可追溯的来源 receipt；[source-plan.json](source-plan.json) 是选择范围，不是支持矩阵。
