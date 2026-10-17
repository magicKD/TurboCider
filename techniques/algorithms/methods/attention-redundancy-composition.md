# DiTFastAttn：空间窗口差分、跨步输出与 CFG 三种冗余一起搜索

<!-- knowledge
id: technique-algorithms-methods-attention-redundancy-composition
type: method
stage: dit
operator: attention, sampling
backend: cpu-reference, mlx, metal
device: cpu, gpu
symptom: stale-cache, launch-bound
semantics: approximate
representation: bf16, fp32
boundary: producer-consumer, request
api: reference-only
-->

[算法入口](../README.md) · [PAB](pab.md) · [FasterCache](fastercache.md) · [实验](../../examples/README.md)

来源：作者 `dit_fast_attention.py`、`fast_attn_processor.py` [S34](../sources.md#s34)。本轮只读，不运行其calibration/pipeline。属于 `approximate-train-free`。

## WA-RS 的差分到底是什么

在refresh步同时算full attention F_t与window attention W_t，缓存差 `R_t=F_t-W_t`。之后便宜地计算当前局部W，再加旧差：`F_hat_(t+1)=W_(t+1)+R_t`。

这与PISA/Sol在**当前步**给未选KV块做分子/分母补偿不同。R是两个已经归一化attention输出的差，依赖历史QKV与窗口。它不等于远端attention单独归一化后的输出，也不能把局部/远端SDPA直接相加当作full。

优点是充分保留当前局部更新，且可复用高性能window内核；代价是refresh额外算一次W，以及缓存完整attention输出差。若refresh频繁，保存差本身也可能抵消收益。

## 另外两种共享

`output_share`直接复用前步attention输出；`cfg_attn_share`只计算一个CFG半batch，并把结果复制给另一半。二者节省不同：前者复用时间，后者近似条件差。`cond_first`决定保留哪半，batch布局不是任意的。

共享发生在attention分支，不是无条件删除整个denoiser的CFG。每层后续residual/FFN/输出仍可能不同，因此最终CFG行为不能只从一处重复tensor推断。

## 校准器也值得学习

hook先获得完整transformer输出，再对某step/layer尝试候选方法，用整体输出损失决定选哪个；测试期间关闭保存residual/输出，避免候选试验污染正式state，重置每层step计数，最后恢复保存行为。这是“探索不更新正式缓存”的重要设计。

误差预算随层/step配置，最后得到静态compression plan。此过程要多次执行transformer，不是免费runtime gate；calibration成本、缓存plan身份与holdout都必须计。

本次代码的普通Diffusers分支入口仍有`breakpoint()`，部分loss分支会加载外部VAE/LPIPS；所以只研究源码，未将其当作可在Mac直接执行的稳定API。默认loss的元素比值也不是通用相对L2，不能与其他方法的threshold数值直接比较。

## Apple 迁移

已有MLX窗口切片可做W的参考，但要计真实concat/launch和refresh额外W。后续Metal fused window可能更合适。使用固定step/layer plan便于编译/ANE分图，但必须绑定checkpoint、LoRA、guidance/negative prompt、scheduler、geometry与窗口定义。

推荐消融：dense、只window、window+旧差、只output-share、只CFG-share、最终组合；每项保留相同强基线并重新做trajectory/RGB/视频质量。不要相乘三个单项倍率，也不要把“校准时整transformer误差小”当作完整视频必定稳定。

论文：[DiTFastAttn作者页面](https://openreview.net/forum?id=51HQpkQy3t)。
