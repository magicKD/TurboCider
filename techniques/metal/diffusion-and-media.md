# 扩散与媒体 GPU 加速：模型图比单个 GEMM 更重要

[GPU 导航](README.md) · [阶段地图](../models/stage-map.md) · [VAE](../media/vae.md)

核查：2026-10-16。本文追踪 mflux、stable-diffusion.cpp、Draw Things/ccv 与 SolAttn 的不同层次；不复制已有 ANE/跨设备报告。

## mflux：用成熟 Metal backend，优化 denoise graph

固定 `8c00dab` 的 Qwen21 路线是一个清楚的调用链：

```text
Qwen21Transformer.__call__
  → geometry / timestep / modulation
  → mx.compile(_forward)，按 shape 重 trace
  → transformer_blocks
  → Q/K norm + RoPE + fast SDPA
  → output projection / residual / FFN
  → MLX Metal dispatch
```

[Transformer 源码](https://github.com/filipstrand/mflux/blob/8c00dab2505a96019df9d30bc9c223bf20d733c4/src/mflux/models/qwen21/model/qwen21_transformer/qwen21_transformer.py)将一个 denoise step 编译，保存 geometry cache。它并不使模型加载、所有外层 sampling 或 VAE 自动都成为单一 GPU graph；分辨率、文本长度、mask/LoRA 状态变化仍需检查 cache/specialization 与实际 graph。

[Attention 源码](https://github.com/filipstrand/mflux/blob/8c00dab2505a96019df9d30bc9c223bf20d733c4/src/mflux/models/qwen21/model/qwen21_transformer/qwen21_attention.py)在 padding-free 条件将 text causal 与 target full attention 分为两次调用，避免构造一张通用 block mask；有 mask 时保留通用路径。这是**利用模型 mask 结构选择更好的 backend 路线**，不是任意拆 attention 都能保持数学。text/target KV 范围、token 顺序与带 padding 条件必须与原式一一对应。

可迁移实验：同一 Q/K/V/原 mask，比较单次通用 mask、分段原式、保持分段但改善 layout；计入 concat/reshape/output projection。已有 [真实 Qwen attention](../review/qwen-attention-20261016.md)显示几个看似合理的候选并未稳定胜过原控制，因此不能凭减少调用数决定采用。

## stable-diffusion.cpp：GGML 算子与阶段/权重管理共同工作

固定 `2f88688` 的 [backend-fit](https://github.com/leejet/stable-diffusion.cpp/blob/2f886889e6e8b78738d6b87f7191f6018557c551/src/core/backend_fit.cpp)在设备预算与参数位置之间构造方案，区分 resident 与待缓存权重；Metal 算子执行则沿其实际 GGML 依赖继续追。

应拆开评估：text encoder、DiT/UNet、VAE、控制/超分、媒体 IO。某个 denoiser 的压缩支持不证明 VAE 或所有 Conv3D 都有同等 GPU 覆盖；节省权重驻留也可能换成每 step 上传。benchmark 必须包含 scheduler、权重切换与真实消费者，不只测一个事先 resident 的 Linear。

量化/Flash Attention/CPU offload/tiling 应分别控制；不把去掉高精度 VAE、换低分辨率或缩短视频当成同语义 backend 提速。项目级模型支持以固定版本文档和实际日志为准，本轮未下载/运行该引擎模型。

## Draw Things / ccv：应用约束进入 kernel descriptor

[ccv 固定读本](https://github.com/liuliu/ccv/tree/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/mfa)同时有传统 GEMM/attention、NA 核与 ANE bridge。学习重点是 precision/shape/LoRA/缓存/资源从应用层进入 descriptor，再由 kernel generator 形成 pipeline；只读一个 shader 无法解释内存与首调用成本。

本轮 GPU 选读 `NAAttentionKernel.cpp` 与 `NAInt8MatMulSmallMDescriptor.cpp`：前者按执行/精度分配内存、生成源码；后者包含 split-K scratch 与 binary archive。Draw Things 顶层调用链只作项目文档入口，未完整审计模型层到每个 kernel 的所有组合。公共 Core ML 与私有/未文档化假设分别见 [ccv bridge 导读](../sources/repos/ccv-coreml-rowwise.md)，不因为同在 MFA 目录就混为一个稳定接口。

## SolAttn：少做 attention，与“相同数学更快”分账

固定 `4507112` 的 tiled Metal 实现使用 block summaries 与精算/近似块。[代码入口](https://github.com/yshenaw/ComfyUI-SolAttn-MPS/tree/45071126b0c1ee30b0e6b7103fa9d70924828ba5)与文件 SHA 记录于[来源清单](research-sources.json)，避免活动分支变动导致学习断链。

适合长序列/视频的原因是可减少实际 attention 工作；代价是 routing/summary、稀疏模式对模型质量的影响，以及 PyTorch/ComfyUI patch 兼容性。先 dense-native / dense-custom / sparse-custom 三臂，再测 temporal consistency、运动、身份、场景切换和多 seed。一个 latent cosine 阈值不能替代视频质量。

## VAE、超分与图像格式是另一条关键路径

VAE 的卷积、upsampling、layout 和 tiling 与 DiT GEMM 的瓶颈不同。空间/时间 tile 要检查 overlap/halo、padding、接缝、时序状态和峰值；tile 小不等于请求快，重叠区重算可能抵消省内存收益。媒体 decoder、CVPixelBuffer/IOSurface、GPU tensor 之间的 BGRA/RGB、stride、alpha/色彩空间转换也常发生多余 CPU pass。

优先复用已有 [tensor/texture 交接](../media/tensor-texture-interop.md)、[真实媒体三路比较](../review/media-stage-20261016.md)、[VAE/模型生命周期](../models/vae-and-media.md)，再测试 MNN/ncnn 的辅助视觉网络。ncnn 的 Vulkan 路线不要重命名为原生 Metal；Core ML/媒体硬件也不要计作 GPU kernel 本身的收益。

## 接入时的验收顺序

固定 checkpoint/LoRA 与原始 dtype → 单算子全输出 → 完整 FFN/attention consumer → 完整 block → 多层/多 step → 图像/视频质量 → cold/warm/低内存/取消。cache key 包含 source generation 与图状态；异步输出发布前完成依赖，取消先 drain 再复用/释放。保留原路径，不把研究候选自动启用为产品默认。

## 可执行学习控制（不替代上游验证）

本页实现参考：[labs/model_stages.py](../labs/model_stages.py)、[labs/media_stage.py](../labs/media_stage.py)；case：`stage-dit`、`stage-vae`、`media-stage-contract`。CPU/旧 Metal 控制只验证对应数学与合同；`mlx:engine-mechanisms` 对应[显式组件 CLI](experiments-20261016.md)，不是自动启动 GPU 的普通单元测试，也不证明全部上游引擎或 PR 已运行。设备、source、shape、精度或消费者改变后重新验证。
