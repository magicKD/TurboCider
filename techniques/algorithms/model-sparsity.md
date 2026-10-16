# 不同图像 / 视频模型里的 sparsity 究竟在哪里

配套：[机制 witness 与证据合同实验](../labs/algorithm-research.md)。这是支持性小实验，不是本方法的完整模型复现。

[算法入口](README.md) · [结构化稀疏](structured-sparsity.md) · [数值量化](quantization.md)

来源为 [S01/S13/S15/S20/S22/S25](sources.md) 的模型 adapter、attention 与作者说明。本表是结构及研究入口，不是未经测量的 density 配方；没有“某模型天生固定稀疏 90%”的承诺。

## 先拆开五种不同的稀疏

| 概念 | 代表 | 可以推导什么 | 不可以推导什么 |
| --- | --- | --- | --- |
| weight sparsity | 剪枝后零权重、N:M | 仅匹配稀疏 kernel 才省乘法 | 零很多就自动快；CUDA 2:4 指令收益能迁移 |
| attention connectivity | STA / Sol / PISA / SVG | 少算或近似 QK、PV block | FFN、投影、VAE 也同比减少 |
| token/update sparsity | ToMe / pruning / stale-token | 部分 tokens 少更新 | 简单 mask 了 token 就少跑 dense GEMM |
| timestep sparsity | TeaCache / DBCache | 主体少执行几次 | 原采样器少步；residual reuse 是精确缓存 |
| numerical / rank redundancy | Q4/Q8、SVDQuant | 降表示成本/低秩残差误差 | 参数少、文件小就一定快 |

结构上的 linear attention、压缩 latent 和蒸馏 student 是训练时设计，不属于 inference 时发现的近零 softmax。dense FP attention 的 softmax 通常并非含大量精确零；“稀疏”常表示可忽略的小质量，而不是 IEEE 数字等于零。

## 模型—可省工作—敏感边界

| 模型家族 | 结构和冗余入口 | 适合先研究 | 不能丢掉的语义 / 迁移风险 |
| --- | --- | --- | --- |
| SD1.x / SDXL | 多尺度 U-Net，逐步重复卷积/attention | DeepCache、ToMeSD、LCM、decoder 替换 | 多尺度 skip、ControlNet/inpaint；不能照搬 DiT block 边界 |
| FLUX.1 | joint double-stream 后 single-stream，图像/文字交互 | TeaCache、FBCache、SVDQuant、PISA | 文本查询和 K/V sink、RoPE、ControlNet 注入；schnell 与 dev 日程不同 |
| FLUX.2 Klein base / distilled | MLX adapter 已分 variant 与 CFG recipe | base cache、mixed precision；student 做 kernel/驻留 | 少步蒸馏中 cache 可能零命中；不能借 base 校准证明 student |
| Qwen-Image / edit / 2512 / 2.1 | 视觉、文字、参考条件；具体 prefix/target 路径依版本 | DBCache、精确 conditioning、PTQ | 版本不等价；prefix 的逐层 KV 与 timestep/LoRA/shape 相关 |
| Wan2.1 / Wan2.2 | 视觉 self-attention 与 text cross-attention；高/低噪声专家随版本不同 | EasyCache、SVG、低比特、分 stage cache | T2V/I2V、reference、专家切换；两个专家不是同时只用任意一个 |
| HunyuanVideo | 图文交互及长视频全连接注意力；多个变体 | SVG/STA/Sol、TeaCache、VSA 对应权重 | 准确识别 text/padding、3D 位置和非因果结构 |
| CogVideoX | 视频/文字联合序列、相关 temporal geometry | PAB、FasterCache、TeaCache | 不是把所有 joint heads 分成独立 spatial/temporal；时间重排要可逆 |
| LTX-2.x | 视频与音频联合处理、跨模态/文字 attention、多阶段细化 | stage-specific sparse、量化、少步 student | stage-1/2 校准分开、video-self 不代替 A/V 同步检查；2.3 配方不继承 2.5 |
| MiniMax-H3 / FastH3 | packed text/reference/audio/video；VSA 可需压缩 gate | exact packed sink、Sol-MPS、具备 gate 的 VSA | dense-only 权重不能凭开关变 VSA-capable；音频/参考 tokens 不是全部可剪 |
| SANA / SANA-Video | 高压缩 latent + 已训练的高效 attention 架构 | cache、投影/FFN、低比特 | linear attention 不是给 dense softmax 换 mask；SANA image/video 不能混同配方 |

这不是宣称上述所有方法在上述所有模型均已实现。具体证据：TeaCache/MLX 支持范围见 [TeaCache](teacache.md)；FastVideo VSA 的 gate 与质量限制见 [结构化稀疏](structured-sparsity.md)；Sana 模型选型见 [源码导读](sana-sol-engine.md)。

## 怎样实际测量“模型的稀疏性”

从真实模型各 step/layer/head 抽取少量 QKV，保留 dtype、scale、RoPE 后位置、valid lengths、mask 与所有条件 span。FP32/FP64 抽样行计算 dense 分布，记录：top-p 所需 key 数、top-k block 保留质量、attention entropy、局部/跨帧质量、head 间差异、极端最差行、遗漏 attention 的输出误差。

统计 token density 与实际 tile density：若一个 tile 内只有一个重要 key，整 tile 仍可能要算；统计 density 时分母要排除 padding 并说明是否含 exact sink。不能用平均 head 的稀疏率代表最敏感 head。

同时测 mask 稳定性：同 prompt 邻 step 的 overlap、不同 prompt/seed/分辨率的路由漂移。用当前 QKV 的离线 oracle 选预算只证明“有潜在空间”，不是不用 oracle 就能在线得到相同收益。固定参数必须在未参与选择的 prompt/seed 重跑。

## Apple 特有的选择压力

统一内存让大 cache 与权重争带宽；小 batch GEMM、离散 block gather、临时 FP32 summary 可能比 FLOPs 更重要。可优先用连续 tiles、有限 bucket 与 device-side routing。对 ANE 不规则连接，需要先证明静态图表达/后端真实计算/交接成本；不能把 dense 的 INT8 capability 当作 sparse capability。

本库 [LTX 负结果摘要](../../docs/design/ltx-stage2-sparse-results.md)提供一个明确反例：局部 core 很快，但完整阶段与质量预算未共同满足。该产品链接是可选溯源；本目录 [Sol 报告](sol-attention.md)保留关键结论。
