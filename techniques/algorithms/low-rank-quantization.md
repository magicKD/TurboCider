# SVDQuant / Nunchaku：用低秩分支吸收 outlier，低比特主分支做大头

配套：[机制 witness 与证据合同实验](../labs/algorithm-research.md)。这是支持性小实验，不是本方法的完整模型复现。

[算法入口](README.md) · [PTQ / 混合精度](quantization.md) · [GPU/ANE 切分](../gpu-ane/partitioning.md)

来源：Nunchaku 方法与 `SVDQW4A4Linear` [S17](sources.md#s17)，SmoothQuant [S18](sources.md#s18)。分类：`approximate-train-free`；低秩分解不代表纯粹把原模型裁成低秩网络。

## 算法分三步

先将 activation outlier 经平滑迁移到权重。再对变换后权重取 SVD 低秩部分 `L1L2`，主分支量化 residual `W_res=W-L1L2`。按 `X(M×K), W(K×N)` 约定，

`Y ≈ Q_A(X) Q_W(W_res) + (X L1) L2`。

高精度低秩分支吸收难量化结构；4-bit 主分支只需拟合更平滑的 residual。若只把原 W 整体 Q4，再额外加一个任意 SVD 分支，会重复计算而不是 SVDQuant。

## 为什么 Nunchaku 不能只看“两个 GEMM”

`SVDQW4A4Linear.quantize` 调用融合 activation quantization 和 low-rank down projection 的 kernel；`forward_quant` 将量化 GEMM、low-rank up projection 与输出融合。它复用输入/输出 IO，避免独立分支带来的完整 tensor 读写。

源码中的 rank、group_size、int4/nvfp4 packing、scales 与 pad_size 是实际 ABI。作者特别说明 naïve rank32 分支增加显著开销，融合才是系统实现的关键。减少权重 offload 的大收益与同驻留条件的计算提速也要分开，不直接引用作者最大倍率为 Mac 预期。

## Apple 可通用的内容与待补工程

数学可用 MLX/Metal 实现。实用第一版可比较“affine W4A16 residual + BF16 low-rank”与 uniform-Q4、Q8、混合精度；这叫 **SVDQuant-inspired W4A16 实验**，不是完成原 W4A4/Nunchaku kernel。

真正 W4A4 需要目标设备匹配的 packed dot/matrix、scale、activation quantization 与 fused epilogue。ANE 可探索固定 shape 的低秩/主分支，但 split 后仍要支付 layout、调度、共享带宽和 join；本轮未测，不能声称把低秩放 ANE 必然隐藏。

若主分支 `2MKN` FLOPs、低秩 `2Mr(K+N)`，算术比例约 `r(K+N)/(KN)`；当 K=N=4096、r=32 时约1.56%。**这仅是 FLOPs 比，不是延迟 overhead**：小矩阵利用率、重复 IO、launch 与缓冲区通常更关键。

## LoRA 不是低秩 outlier 分支的同义词

LoRA 是模型适配，SVD 分支是重构量化后的原权重，两者目标不同。可研究融合/合并 projection，但 scale、rank、权重顺序与 double-counting 都要验证；换 LoRA 后原校准可能失效，不能仅因低秩形状相同便共用输出。

## 推荐验证矩阵

固定校准集、真实 early/middle/late activation；rank=0/16/32/64、Q4/Q8、W-only vs W/A、fused vs unfused 比较。先检查分解重构/scale 代数，再测 fake-quant component error，最后整 trajectory 与 RGB/视频。

每组同时报权重 bytes、scale/low-rank bytes、workspace、warm GEMM、完整 FFN 和完整请求；保存零 rank、全高精度重构与错误转置负控。单个随机矩阵上“误差减半”不能推导图像保真。

论文：[SVDQuant, arXiv:2411.05007](https://arxiv.org/abs/2411.05007)；作者说明：[MIT Han Lab blog](https://hanlab.mit.edu/blog/svdquant)。该博客是补充学习入口；本轮核心实现结论来自固定代码/README，而非未读博客全文。
