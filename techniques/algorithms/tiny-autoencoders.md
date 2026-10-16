# Tiny AutoEncoders：把便宜预览与最终 VAE 质量分开

配套：[机制 witness 与证据合同实验](../labs/algorithm-research.md)。这是支持性小实验，不是本方法的完整模型复现。

[算法入口](README.md) · [VAE 完整阶段](../models/vae-and-media.md) · [媒体 tiling](../media/vae.md)

来源：TAESD 作者实现 [S23](sources.md#s23)，纯 MLX 移植 `mlx-taef` [S28](sources.md#s28)。分类：`requires-training` 的结构/蒸馏替代；使用发布权重时用户可以无需自己训练。

## 为什么快

用少量 Conv、ReLU residual blocks、upsample 与 clamp 近似原 VAE decoder。它减少计算/参数/中间激活，不是对原 VAE 的等价 kernel 优化。TAESD 各权重对应不同 latent API；SD1.x/SDXL/FLUX1/FLUX2/Qwen/SANA 的 channels、空间压缩率、归一化、RGBA 均可能不同。

作者 `taesd.py` 还包含32×/16×等空间变体，FLUX2 可有全局 pooling/norm。因此不能简单声称“所有 tiny decoder 都只有局部卷积、随便切块无接缝”。

## Apple 可用路径与边界

`mlx-taef` 提供纯 MLX decoder 与 mflux preview callback，是真正的 Apple 实现入口；本次只核对 README 的 API/支持与作者边界，没有逐行审计其模型或复现 decode benchmark。Core ML/ANE 的小 Conv 固定图也值得研究，但没有本轮设备放置或速度证据。

默认更适合中间预览：用 tiny decoder 看进度，最终结果仍用原 VAE。若以 tiny decoder 替代最终输出，必须作为独立产品质量档，测试纹理、脸、颜色、高动态范围、透明度和局部细节。对编辑，tiny encoder 的误差还会进入整个 denoising，不能拿 decoder-only 的收益和误差代替。

对视频逐帧独立2D解码可能闪烁，3D/causal tiny 模型则需匹配 history 与时间压缩规则。一次帧图像 SSIM 不证明全视频稳定，也不证明原音频对齐。

## latent ABI 最容易错在哪里

检查 generator callback 的 latent 是否 noisy/x0、packed/unpacked、subpixel fold 顺序、channel 数与 spatial divisor、scale/shift/mean/std、layout 和 alpha 语义。相同 shape、甚至共享原 VAE 权重，也不能证明 callback tensor 合同相同。

作者 MLX 项目记录自动套某些 VAE normalization 会令预览更差；应对 tiny 模型训练时的真实输入做 oracle，而不是见到统计量就机械反归一化。预览失败可禁用，但性能/质量测试必须 fail-fast 或明确报告 fallback。

## 测量

同一已 materialized latent 的 full decoder、tiny decoder、callback disabled 三组；分别记录 decode、GPU→image、PNG/视频写盘、callback 对 denoise 的扰动。只有原 VAE 是主要开销，替代最终解码才有大的请求收益；每步新增预览即使很快，也可能让原本无预览的总请求更慢。

作者公开过冷暖状态不对称造成过大 decoder 倍率并更正；因此不沿用旧“50–100×”宣传。本轮没有生成新媒体质量结果。[S28](sources.md#s28)
