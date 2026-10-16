# FasterCache：attention 外推与 CFG 分支近似

配套：[机制 witness 与证据合同实验](../labs/algorithm-research.md)。这是支持性小实验，不是本方法的完整模型复现。

[算法入口](README.md) · [PAB](pab.md) · [TaylorSeer](taylorseer.md)

来源：作者 CogVideoX 推理脚本 [S10](sources.md#s10)，Diffusers cache 文档 [S25](sources.md#s25)。分类：`approximate-train-free`；不能简化为“每隔一步直接关 CFG”。

## 两种节省必须单独归因

第一部分在相邻完整 attention output 上作短距离外推，跳过部分 attention 前向。作者 CogVideoX 示例保留两次输出，并用差分乘一个权重补偿 stale 值。它与简单 PAB 的零阶广播不同，但也不等于任意高阶 TaylorSeer。

第二部分发现 conditional/unconditional 输出有重复结构：定期完整算两路并保存频域差，期间只算 conditional，再用低/高频补偿重构 unconditional。原脚本 `fft`、`delta_lf`、`delta_hf`、`ifft2` 以及 `counter` 窗口共同决定此路径；其硬编码 step 区间是特定日程配方，不是通用常数。

对 `y_cfg = y_u + g*(y_c-y_u)`，若只近似 unconditional，误差为 `(1-g)*e_u`；guidance 越强，分支误差可能越被放大。这是代数分析，不是作者给出的统一质量上界。

## Apple 移植优先拆解

先单测 attention output reuse，和每步完整 CFG 匹配；然后单测 CFG compensation；最后组合。MLX/MPS 的 FFT、complex tensor、dtype 支持与 layout 必须实际核实；可能要 FP32 精度岛，不能假设原 CUDA BF16 代码原样可用。若 FFT 回退 CPU 或引入大规模同步，会削弱整分支节省。

对已经 guidance-distilled、没有双路 CFG 的 student，第二部分无对象可省。对 joint audio/video 或不同 CFG branch 数的模型，分别定义图像、视频、音频输出与坐标，不能在错误轴上套 `fft2`。

## 需要记录的合同

cache key 包含 CFG branch、guidance、negative prompt、timestep/scheduler、geometry、LoRA；分支合并的 batch 位置必须有显式定义。fresh full 的频谱差才能更新状态，fallback 后先重新 seed，不继续用半更新历史。

固定三路/四路消融：全关、attention-only、CFG-only、组合。评测 RGB/运动/提示词依从性、negative prompt 有效性和极端 guidance，报告 FFT 本身、分支节省、内存及完整请求。不要把作者 CUDA 多卡的并行收益归给 cache。

论文：[FasterCache, arXiv:2410.19355](https://arxiv.org/abs/2410.19355)；作者展示：[项目页](https://vchitect.github.io/FasterCache/)。
