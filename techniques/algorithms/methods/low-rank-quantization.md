# SVDQuant / Nunchaku：用低秩分支吸收 outlier，低比特主分支做大头

<!-- knowledge
id: technique-algorithms-methods-low-rank-quantization
type: method
stage: dit
operator: quantization, gemm
backend: mlx, metal, coreml
device: gpu
symptom: precision-loss, launch-bound
semantics: approximate
representation: bf16, q4, q8
boundary: request, producer-consumer
api: reference-only
-->

配套：[机制 witness 与证据合同实验](../../examples/README.md)。这是支持性小实验，不是本方法的完整模型复现。

[算法入口](../README.md) · [PTQ / 混合精度](quantization.md) · [GPU/ANE 切分](../../gpu-ane/partitioning.md)

来源：Nunchaku 方法与 `SVDQW4A4Linear` [S17](../sources.md#s17)，SmoothQuant [S18](../sources.md#s18)。分类：`approximate-train-free`；低秩分解不代表纯粹把原模型裁成低秩网络。

第六轮新增[实际MLX packed QMM＋低秩组件](../case-studies/lowrank-mlx-component-study.md)：32点、4608次计算求值及384次host control。只有合成FP16输入/权重，W4A16与独立低秩，不是Nunchaku W4A4、完整FFN或生成媒体；保留误差下降但净延迟变差的结果。

第十轮进一步在[真实Qwen完整BF16 FFN GPU](../case-studies/real-ffn-gpu-study.md)完成36点、4752段formal时间；warm resident下所有compiled packed/低秩候选仍慢于强compiled dense。实际低秩数值改善、逻辑weight压缩、GPU组件延迟和完整媒体必须分别验收，不把此W4A16负结果推广到作者W4A4或offload。

## 算法分三步

先将 activation outlier 经平滑迁移到权重。再对变换后权重取 SVD 低秩部分 `L1L2`，主分支量化 residual `W_res=W-L1L2`。按 `X(M×K), W(K×N)` 约定，

`Y ≈ Q_A(X) Q_W(W_res) + (X L1) L2`。

高精度低秩分支吸收难量化结构；4-bit 主分支只需拟合更平滑的 residual。若只把原 W 整体 Q4，再额外加一个任意 SVD 分支，会重复计算而不是 SVDQuant。

## 为什么 Nunchaku 不能只看“两个 GEMM”

`SVDQW4A4Linear.quantize` 调用融合 activation quantization 和 low-rank down projection 的 kernel；`forward_quant` 将量化 GEMM、low-rank up projection 与输出融合。它复用输入/输出 IO，避免独立分支带来的完整 tensor 读写。

源码中的 rank、group_size、int4/nvfp4 packing、scales 与 pad_size 是实际 ABI。作者特别说明 naïve rank32 分支增加显著开销，融合才是系统实现的关键。减少权重 offload 的大收益与同驻留条件的计算提速也要分开，不直接引用作者最大倍率为 Mac 预期。

进一步读`ops/quantize.py`：输入M会按pad_size默认256分配packed activation与FP32低秩hidden；INT4与NVFP4的group/scale dtype不同。读`ops/gemm.py`与`lora.cuh`可见up/down epilogue在tile accumulator边界工作，因子和hidden为专用packed布局，不是把row-major A/B改个名称即可兼容。此处`lora`命名同时服务低秩计算，不代表所有因子都是用户训练LoRA。

另一个重要代价是低秩hidden与reduction：原接口为FP32，当前MLX起点使用FP16常规GEMM。即使实数公式一致，激活padding、accumulator、量化/smoothing、输入/输出复用与具体融合边界也并不相同；没有执行原CUDA pipeline，不能继承作者倍率。

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

第六轮补读该blog的方法、融合与offload口径，源码阅读仍固定commit；没有逐式审计论文全部推导或重放作者模型。MLX affine解码的dtype/postcast反例见[组件报告](../case-studies/lowrank-mlx-component-study.md)，它修正的是本库非计时参考前提，不冒称上游解码bug。

## 第九轮：DeepCompressor里真正的校准循环

[S43固定源码](../sources.md#s43)补齐模型转换端，而不只看Nunchaku运行时。`LowRankBranch.reset_parameters`对原weight做double SVD，A取Vh、B取UΣ，最后cast回weight dtype；这与第六轮/本轮原创range80随机分解和sqrt(Σ)平分因子不是相同有限精度配方。实数的factor gauge等价，不代表舍入、临时量或转换成本相同。

必须分开三条路径：

| 路径 | 分解目标 | 主分支 | 不能混写的地方 |
| --- | --- | --- | --- |
| weight低秩＋residual quant | W或平滑后的W | Q(W−L) | 不能把L再加到原Q(W)上 |
| one-shot `compensate` | W−Q(W) | 固定Q(W) | 加的是量化误差低秩，不是原W的低秩，因此不必double-count |
| 多轮calibrator | 每轮W−QW_previous | 每轮重算Q(W−L_current) | 不能只保留第一轮SVD或把fixed-main旁路称为同一迭代 |

实际`QuantLowRankCalibrator._reset/_ask/_tell`保存best candidate，在所选objective下比较当前误差；启用early-stop后变差可提前结束。配置类默认num_iters=1，而所读SVDQuant默认YAML为rank32、OutputsError、迭代上限100、early-stop；两个“默认”层级不能混成一种配置。

diffusion driver的`compensate && num_iters<=1`会跳过前置branch calibration，在正式weight quant后加误差branch；其它情况先校准branch、从weight减去effective low-rank，再做weight range/kernel量化。`quantize_with_low_rank`也显式区分补偿与residual构造。因此“先Q(W)再加任意SVD分支会重复计算”中的“任意”很重要：**Q(W)+SVD(W−Q(W))是合法的另一近似，而Q(W)+SVD(W)通常不是**。

OutputsError还可能在attention或其并行parent block上评估，而非仅用weight Frobenius loss；input quantizer可以同时参与。smooth默认YAML包含AbsMax跨度、GridSearch/20格和allow_low_rank；INT4配置同时量化weight和input，并允许unsigned input。这里没有执行该平滑搜索、100轮循环、GPTQ、activation量化或作者模型转换，不能将本地weight-only数字挂上作者默认W4A4名称。

共享输入的QKV等可先沿输出维拼接，共享A、分片B，再分别量化residual。迁移时还要核对共享input的scale、通道布局及hook边界；相同rank不授予跨模块或跨LoRA兼容。

## 对角激活权重与完整FFN为什么要另验

对weight误差E，线性输出平方误差为`||X Eᵀ||²_F = tr(E XᵀX Eᵀ)`。按输入RMS对W列缩放只近似这个目标的对角部分，忽略通道相关性；而且后续residual量化、因子舍入和SwiGLU并不被未量化SVD的最优性覆盖。原创`lowrank-calibration-contracts`中，两个输入具有相同RMS/对角proxy，实际输出误差却分别为0与4。

再令z为SwiGLU hidden，D为down weight：`ΔY=Δz Dᵀ + z ΔDᵀ + Δz ΔDᵀ`。独立GU-only、down-only误差之和没有覆盖最后的cross term；标量witness中两个独立误差各1，组合为3。校准点选出的最优候选，也不自动在新prompt或后续timestep最优。

本轮将这些边界推进到[真实三层BF16权重、原始base与新独立prompt激活的126点全宽FFN采样](../case-studies/real-qwen-lowrank.md)；不是把合成第六轮速度与真实误差拼成模型排行榜。保留中层较高误差、RMS加权13/18点变差和3个rank非单调点；默认CPU数学case不读取权重或执行上游。
