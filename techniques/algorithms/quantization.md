# 扩散 PTQ、SmoothQuant 与逐层混合精度

[算法入口](README.md) · [SVDQuant](low-rank-quantization.md) · [ANE 量化](../ane/quantization.md)

来源：MLX Metal/QMM 与量化 API [S19](sources.md#s19)，mflux [S20](sources.md#s20)，SmoothQuant [S18](sources.md#s18)，Apple Core ML Stable Diffusion [S27](sources.md#s27)，作者 Qwen 混合精度案例 [S03](sources.md#s03)。分类：PTQ 为 `approximate-train-free`；QAT 为 `requires-training`。

## 四种“低比特”不能混称

| 配方 | 存储 / 计算 | 收益来源 | 风险 |
| --- | --- | --- | --- |
| affine weight-only W4A16/W8A16 | packed codes+scale/bias，activation 浮点 | 权重带宽、驻留与融合解码 | 大 M 下解码/计算路径不一定快 |
| palettization / LUT | index 查浮点 palette | 压缩与 JIT 解码、memory-bound 层 | 不是相同 bit 的线性整数 GEMM |
| W8A8 / W4A4 | 权重、activation 都量化 | 设备适合时提高运算吞吐 | calibration、溢出和敏感层 |
| FP8/FP4 | 特定浮点编码及 scale recipe | 专门 kernel / 硬件路径 | format名≠所有 Metal/GPU/ANE 都支持原生加速 |

MLX `nn.quantize` 的 `class_predicate` 可返回 False 或不同 bits/group_size 参数，适合每层策略。当前源码中存在 affine/mxfp4/nvfp4/mxfp8、`QQLinear` 等 API 名，不等于所有模式、设备与 shape 都有同样 native kernel；要沿 backend/metal 与具体 primitive 的 dispatch 核实。不能写成“MLX 永远只有 W4A16”，也不能反过来说“有 QQLinear 即 Mac NVFP4 已验收”。

## SmoothQuant：先代数重参数化，再量化

按行向量约定 `Y=XW`，对角正 scale S 满足 `Y=(X S^-1)(S W)`。用激活通道范围与权重范围的 alpha 配比选择 S，把 activation outlier 压到权重一侧，便于 A8。原 PyTorch Linear 存储为 `(out,in)`，代码对权重列乘 scale，对 norm weight/bias 除 scale；转置约定不能弄反。

共享一份 norm 输出的 Q/K/V 或 gate/up 必须共同补偿。DiT 的 AdaLN modulation、LoRA 和多消费者接缝不能只改一个 linear。重参数化在实数下等价，量化之后仍然近似。原 SmoothQuant 仓库校准针对 LLM，不覆盖图像/视频 denoise 的 timestep、CFG 与参考条件分布。

## 扩散校准需要覆盖时间轴

至少采样 early/middle/late step、不同 block/heads、CFG 正负、真实参考数量、分辨率/帧桶与 LoRA。只用随机正态激活或第一步会漏掉后期细节/低噪声范围。可冻结一组 layer/timestep precision islands，保留 embedding、modulation、norm、final projection 或敏感 edge blocks，但哪些敏感必须来自该模型的消融，不把经验当定律。

作者 M1 Max 32GB 的 Qwen-Image 单 portrait/seed 案例，从 uniform-q4 改为 BF16 条件/输出、edge blocks Q8、中段 Q4，额外约1.9GiB改善皮肤纹理；作者明确没有更高精度完整参考和通用 sensitivity map。这证明混合配方值得测试，不证明所有 Qwen 更保真，更不能迁移为 Qwen-Image-2.1 固定层表。[S03](sources.md#s03)

版本差异很重要：本次固定的 mflux `QwenWeightDefinition.quantization_predicate` 已在 bits=4 时将 `.img_mod_linear` 保留8-bit，并跳过不支持量化的模块；不能沿用上述旧版本案例，把当前 mflux 一概称作 uniform-q4。当前策略也仍需完整模型/LoRA质量验证。[S20](sources.md#s20)

## Apple 官方的另一条路线

Apple 的 Stable Diffusion 实现有6-bit及更低 mixed-bit palettization：先逐层分析，对超预算层提高 bitwidth，再按 denoised latent PSNR 选择整套 recipe；还提供 activation sensitivity，使较敏感层 W8A16、其他 W8A8。它是实际 Apple 生态的算法/后端协同，不只是 CUDA 思路。

官方说明 JIT 解压依赖 compute unit、层类型、硬件代际；palettized storage 不等于整数计算。其 latent PSNR 也不直接证明 image-text alignment，最后仍要完整图像评测。[S27](sources.md#s27)

## 如何公平验证

保留原 BF16、dequantized fake-quant（表示误差）、真实 packed backend 三路；分别测离线压缩、cold load、warm QMM、完整 FFN/DiT、E2E 和峰值。记录 scales/biases、group、padding、dtype 与 LoRA 合成成本。

GPU/ANE 的同名 W8A8 不能只比各自最漂亮 microbench；交接和 dequant precision islands 一起计。低比特让 GPU 主分支更快后，ANE 最佳 share、cache 阈值与稀疏 head budget 都可能改变，组合需重新验收。

已有 [GPU affine 实测](../review/gpu-affine-20261016.md)与 [量化数学实验](../labs/quantization.md)可作为本库继续探索入口，不自动授予生成质量资格。
