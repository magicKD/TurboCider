# FasterCache：attention 外推与 CFG 分支近似

<!-- knowledge
id: technique-algorithms-methods-fastercache
type: method
stage: dit
operator: sampling, attention
backend: mlx, metal
device: gpu
symptom: stale-cache, precision-loss
semantics: approximate
representation: bf16
boundary: request, producer-consumer
api: reference-only
-->

配套：[机制 witness 与证据合同实验](../../examples/README.md)。这是支持性小实验，不是本方法的完整模型复现。

[算法入口](../README.md) · [PAB](pab.md) · [TaylorSeer](taylorseer.md)

来源：作者CogVideoX推理脚本[S10](../sources.md#s10)，Diffusers cache文档[S25](../sources.md#s25)，新增完整consumer与CPU接缝[S46](../sources.md#s46)。分类：`approximate-train-free`；不能简化为“每隔一步直接关CFG”。[第十二轮报告](../case-studies/fastercache-consumer-study.md)纠正局部变量与真实分支身份，保留421行完整向量重放，不是大模型/视频/GPU速度复现。

## 两种节省必须单独归因

第一部分在相邻完整 attention output 上作短距离外推，跳过部分 attention 前向。作者 CogVideoX 示例保留两次输出，并用差分乘一个权重补偿 stale 值。它与简单 PAB 的零阶广播不同，但也不等于任意高阶 TaylorSeer。

第二部分定期完整算CFG两路并保存频域差，期间只算一路、再用低/高频补偿重构另一路。**不能凭脚本的cond/uncond变量名断定身份**：固定CogVideoX配方的guider按[uc,c]拼batch，实际x[:1]/context[:1]保留unconditional，重构conditional；其它adapter需另审。原fft只变换H/W，fftshift/ifftshift却配对移动全部轴，不是THW视频频谱。delta_lf/hf在特定counter窗口每次复用乘1.1，直到full refresh重置；区间和放大系数不是通用常数。

对`y_cfg=y_u+g*(y_c-y_u)`，只近似unconditional的误差是`(1-g)*e_u`；只近似conditional则是`g*e_c`，两路均近似时相加。上述CogVideoX源码属于第二种，不能沿变量名套第一种。它们都是代数分析，不是作者给出的最终质量上界。

## Apple 移植优先拆解

先单测 attention output reuse，和每步完整 CFG 匹配；然后单测 CFG compensation；最后组合。MLX/MPS 的 FFT、complex tensor、dtype 支持与 layout 必须实际核实；可能要 FP32 精度岛，不能假设原 CUDA BF16 代码原样可用。若 FFT 回退 CPU 或引入大规模同步，会削弱整分支节省。

本轮实际CPU源码+明确模型替身显示：50-call单层attention branch计数，attention-only为56、CFG-only为74、组合为57，而不是56%×74%=41.44%；切回双路时batch守卫会额外刷新attention。这是调用链省算证据，不是GPU时间或可乘的完整请求收益。固定.3外推也未按真实anchor时间间隔归一；正常counter0 restart与中途异常/geometry变化必须分开验证。

[第十三轮MLX GPU实验](../case-studies/spatial-cache-mlx-study.md)进一步利用FFT线性性，在full refresh保存两band的空间residual，hit只作scalar更新/相加；实测组件收益包含refresh和强去shift/compiled分母。仅适用于固定mask/各band统一scalar，FP32与旧谱state不逐位等价，大偏置拒绝保留；没有真实模型/视频或每臂物理peak资格。

对已经 guidance-distilled、没有双路 CFG 的 student，第二部分无对象可省。对 joint audio/video 或不同 CFG branch 数的模型，分别定义图像、视频、音频输出与坐标，不能在错误轴上套 `fft2`。

## 需要记录的合同

cache key 包含 CFG branch、guidance、negative prompt、timestep/scheduler、geometry、LoRA；分支合并的 batch 位置必须有显式定义。fresh full 的频谱差才能更新状态，fallback 后先重新 seed，不继续用半更新历史。

固定三路/四路消融：全关、attention-only、CFG-only、组合。评测 RGB/运动/提示词依从性、negative prompt 有效性和极端 guidance，报告 FFT 本身、分支节省、内存及完整请求。不要把作者 CUDA 多卡的并行收益归给 cache。

论文：[FasterCache, arXiv:2410.19355](https://arxiv.org/abs/2410.19355)；作者展示：[项目页](https://vchitect.github.io/FasterCache/)。
