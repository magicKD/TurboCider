# Apple Stable Diffusion：attention 表达、压缩与阶段生命周期

[ANE 目录](README.md) · [布局](layout-and-lowering.md) · [pipeline](../models/pipeline.md)

固定读本：[ml-stable-diffusion@ea2805dc](https://github.com/apple-aiml-research/ml-stable-diffusion/tree/ea2805dc1945be20561c77e5f6d1d9a5a637cda2)。重点读 [attention.py](https://github.com/apple-aiml-research/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/python_coreml_stable_diffusion/attention.py)、[转换器](https://github.com/apple-aiml-research/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/python_coreml_stable_diffusion/torch2coreml.py)、[Swift pipeline](https://github.com/apple-aiml-research/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/swift/StableDiffusion/pipeline/StableDiffusionPipeline.swift)和 [ManagedMLModel](https://github.com/apple-aiml-research/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/swift/StableDiffusion/pipeline/ManagedMLModel.swift)。本轮源码研究，不重跑 SD 全模型。

## 1. 三种 attention 表达

`ORIGINAL` 使用带 head 维的较大 einsum；`SPLIT_EINSUM` 将 Q/K/V 按 head 切开，以 BC1S 组织两次投影及 softmax；`SPLIT_EINSUM_V2` 再将 query 按 512 分块，减小中间 attention score 张量。源码注释分别推荐 GPU 原表达、ANE split 表达；这是候选建议，不是现代 GPU/所有 DiT 的固定分界。

query chunk 的数学前提是：每个 query 块仍访问完整、正确 masked K/V，只在 query 维拼接结果。若切 key 维，则必须合并 softmax 的归一化统计，不能把各块独立 softmax 后简单相加。

### 不能不看就复制的两处前提

所读 V2 用整除计算 `num_chunks = query_seq_length // 512`，构造等长切片；当长度大于 512 且不整除时，尾部没有相应切片。它接受 `mask` 参数，却没有像 V1 那样在 score 上加 mask。SD 的指定 shape/非因果调用可能不触发这些条件；**推广到任意长 prompt、带 mask 或 causal attention 前必须补 gate、tail 和数值回归**。这是源码层适用域分析，不冒充本轮运行上游后的 bug 复现。

另外，query 分块降低单个中间量，并不改变 attention 的 O(QK) 总数学工作。图内 chunk 与 host 多次预测也不是同一种调度成本。

## 2. Chunk UNet 不是“少跑半个模型”

`--chunk-unet` 把计算沿依赖切成多个模型，以降低编译/资源峰值；完整输出仍需要各段串联。它和 FFN 按通道并行、视频空间 tile、query chunk 是四种不同切分。收益要用完整模型加载、跨块 feature、驻留与完整 denoise 步骤衡量。

`ManagedMLModel` 通过自己的串行 queue 管理 lazy load 与 closure 内使用，避免 unload 与预测竞争。`reduceMemory` 在 encoder、UNet、decoder 等阶段后卸载模型，用额外重载换内存；资源工作没有消失。文本 encoder 只跑一次而 denoiser 多步重复，两者常驻的摊销逻辑不同。

## 3. Mixed-bit LUT，而不是“整个模型 INT4 算术”

转换器 `quantize_weights` 调用 `palettize_weights`，并可排除 embedding gather；README 的 SDXL 例子使用各层不同 bit 的 mixed-bit 配方。阅读时同时追踪精度、被压缩组件、模型版本、分辨率与 deployment target。官方入口：[Core ML SD研究](https://machinelearning.apple.com/research/stable-diffusion-coreml-apple-silicon)、[mixed-bit配方说明](https://github.com/apple-aiml-research/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/README.md#compression-lower-than-6-bits)。

质量校准不能只看单层误差。扩散误差会随 step 累积，需看条件编码、每步 velocity/latent、最终图像/视频及敏感层高精度例外。少步骤或缩小输出不属于同语义加速。

## 4. 给 TurboCider 的迁移建议

优先借鉴 BC1S、稳定阶段、有限 bucket、混合压缩和有界模型生命周期。不要把 SD1/2 的 UNet、77 文本 token 与现代 Qwen/FLUX 视频 DiT 混用，也不要用旧 iPhone 数字预测 M4 Max。

建议实验：完整 GPU → 相同模型的 Core ML → 单阶段 offload → 合法 hybrid；每臂保持同一步数、CFG、token、权重、seed、尺寸和输出完成边界。单独比较 first/warm、reduceMemory、UNet chunk、attention 表达、LUT，而不是一次打开全部开关后归因给 ANE。

对 VAE/视频还要保留 causal history、halo、全局 norm、色彩和最终媒体写出。小 Conv 图的胜利不授予完整 VAE 或视频请求资格，见 [既有阶段指南](../models/vae-and-media.md)。

## 本地学习入口与复查

- [operators.py](../labs/operators.py)：原创CPU合同/显式实验客户端，不是上游完整实现。
- [systems.py](../labs/systems.py)：原创CPU合同/显式实验客户端，不是上游完整实现。

CPU入口：`python3 -B techniques/labs/run.py online-attention`。只运行参考数学/合同，不自动触发Core ML、私有API或网络；实际设备命令与证据边界见[实验报告](experiments-20261016.md)。

SoC/OS/SDK、源码/编译入口、权重/LoRA、输入域/shape/layout/精度或实际消费者改变时重开；不继承旧数值、性能或物理设备资格。
