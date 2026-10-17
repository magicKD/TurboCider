# GPU affine实测：缓存收益、tile负结果与原始BF16控制

<!-- knowledge
id: technique-metal-case-studies-affine-ffn-cost-and-resident-tradeoff
type: case-study
aliases: evidence-gpu-affine-20261016
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[研究工作台](../../MAINTENANCE.md) · 全部数据 · [MLX来源](../../sources/repos/mlx-affine-dispatch.md) · [GGML来源](../../sources/repos/ggml-metal-layouts.md) · [上一组Core ML实验](../../gpu_ane/case-studies/qwen-ffn-tiling-and-concurrent-cost.md)

本轮结论：**在相同packed配方内，Q8驻留展开约快4.7%；与原始BF16完整GPU图比较则没有实质收益。原始路径保持不变。** Q4/Q8减小存储不等于整数MAC或更快；原创SG tile数值通过，却显著慢于现有MLX。

## 版本、数据和强基线

2026-10-16，M4 Max/40 GPU cores/64 GiB、已安装MLX0.32.0，device API的architecture为`applegpu_g16s`。新增只读v0.32.0 sparse checkout，固定`7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`，与既有`36969c...`源码对照；whole wheel来源仍由实际core/dylib/metallib SHA约束，版本号不是可复现构建证明。

真实输入沿用原生GPU轨迹capture，不随机生成权重、不重复小row：layer0，M1024/H4096/F12288，fused gate/up `[24576,4096]`与down `[4096,12288]`。step1用于discovery，step19另起进程确认。共1个synthetic tail/stride smoke、10个真实discovery、5个真实confirmation；另有2组独立诊断和1次已修复frontend初始化错误，全部保留。

两层分母分别回答不同问题：

- **同encoded storage比较**：所有方法使用相同uint32 codes、FP16 scale/bias、affine group64。它们可能改变decode/accumulation舍入，不承诺bitwise。FP32全输出参考与独立CPU FP64见证验证数值，不仅用同一个backend自比。
- **原始数据控制**：另从原始checkpoint/capture读取BF16 bits，验证选定tensor与原始输入SHA。基线用原始`[1,M,H]` BF16图；候选计入BF16→FP16输入转换与BF16输出恢复。输出ABI仍为`[1,M,H]` BF16。原始baseline对既有BF16 replica的全输出误差为0；它不是用FP16权重反推出来的替代基线，也不是整请求benchmark。

目标span包含完整fused gate/up→SiLU×up→down，直到所有输出`mx.eval`完成；CPU读回和metrics在计时外。权重已resident，quantize/decode setup另列，不声称含每层streaming或冷加载。每run有4次warmup、8个ABBA测量/4个pair，source snapshots与结束校验；background未隔离、频率未锁定，bootstrap只是小样本探索性区间。

固定同配方full/sample relative L2≤0.02；原BF16 component budget为relative L2≤0.03、cosine≥0.999。通过这个局部门槛不等于32层、多step、LoRA或生成画质通过，不能把PTQ改写为原模型无损优化。

## 所有discovery，不只保留一个最小耗时

单位ms，均为各run warm中位wall；倍率为median of paired ratios。

| 比较 | baseline | candidate | paired倍率 | 范围 |
| --- | ---: | ---: | ---: | --- |
| Q8 packed→驻留decoded cache | 22.713 | 21.739 | 1.045× | 同packed配方，不是原始模型收益 |
| Q8 packed→动态输入的decode表达式 | 22.753 | 23.044 | 0.986× | 返回decoded矩阵以materialize；实际dispatch次数未profile |
| Q8 packed→原创SG BM32/BN32/BK32 | 22.725 | 93.984 | 0.242× | 完整数值与tail guards通过，仍慢 |
| Q8 packed→BM64 | 22.709 | 66.742 | 0.340× | 整个配置改变，不单独归因某一资源 |
| Q8 packed→BM64连续K加载 | 22.707 | 66.731 | 0.340× | 源码访存假设未带来有效改善 |
| Q8 packed→连续加载+16-byte row skew | 22.700 | 66.633 | 0.341× | 没有bank counter，不虚构stall归因 |
| Q4 packed→驻留decoded cache | 22.665 | 21.761 | 1.041× | 需独立确认，不能当稳定赢家 |
| 原始BF16→Q8 packed | 21.703 | 22.773 | 0.952× | 包含输入cast/输出restore |
| 原始BF16→Q4 packed | 21.713 | 22.699 | 0.956× | 压缩没有改善此resident workload延迟 |
| 原始BF16→Q8 decoded cache | 21.705 | 21.727 | 0.999× | 基本持平，没有产品提速理由 |

原创核的加载/tiling空间和保护条件在[实现](../../examples/metal/affine_gemm.metal)中，不代表最优Metal写法。coalescing/padding无收益只否决这组实现和假设，不能说这些优化机制普遍无效。

## 独立step19确认

| 固定比较 | baseline ms | candidate ms | paired倍率 | 探索性95%区间 |
| --- | ---: | ---: | ---: | --- |
| Q8 packed→cache | 22.741 | 21.726 | 1.047× | [1.045,1.049] |
| Q4 packed→cache | 22.663 | 22.519 | 1.008× | [0.916,1.047]，覆盖1 |
| 原始BF16→Q8 packed | 21.681 | 22.792 | 0.952× | [0.949,0.956] |
| 原始BF16→Q4 packed | 21.695 | 22.688 | 0.957× | [0.955,0.960] |
| 原始BF16→Q8 cache | 21.696 | 21.741 | 0.998× | [0.996,1.000] |

Q8/Q4本轮均通过固定的局部original budget；Q4误差仍明显大于Q8，且随timestep变化。不能只因budget通过，就忽略完整扩散轨迹/更多层会放大误差的可能。这里没有新的生成候选图片/LoRA质量资格。

## cache不是免费收益：只列逻辑权重bytes

| 表示 | 本完整FFN权重 |
| --- | ---: |
| 原始BF16或展开FP16 | 288 MiB |
| affine Q8 + scale/bias | 153 MiB |
| affine Q4 + scale/bias | 81 MiB |
| 保留Q8并缓存FP16展开 | 441 MiB |
| 保留Q4并缓存FP16展开 | 369 MiB |

cache额外增加288 MiB；若假设32个相同shape FFN全部保留，就是额外9 GiB的**派生逻辑量**，不是实测request峰值。两arm共享准备好的candidate分配，未测隔离的每arm峰值。进程RSS/MLX allocator峰值包含大CPU/GPU参考数组，不能拿来宣称模型本身需要这些内存。

这些结果支持下一步研究memory-limited Pareto，而不支持默认换量化或展开缓存。原始GPU已经是dense图；“省去量化路径的一点decode税”不自动变成比它快。

## 格式与精度探针，和性能分开

stride/precision探针实际验证X/Q/S/B全部非连续：元素strides分别为`[192,2]`、`[24,2]`、`[6,2]`、`[6,2]`；M33/N41/K96，独立CPU参考与guard通过。shape正确不能替代完整packed/metadata stride。

同库Q4 tiny-scale控制中，公开dequantize与canonical FP16 storage一致；packed M1输出0.0019817352，M33为0.0019683838，原创float-decode为前者。正常控制与Q8控制均保留。这与所读half `scale/16`表达的模拟相容，但没有物理dispatch trace，不把source推断当指令归因或所有版本硬件定律。

GGML原函数摘录客户端真实执行保留许可的Q4_0/Q8_0两个模板体：正常nibble/signed控制匹配，subnormal Q4出现8.94e-7最大绝对差。CPU块解码和错误nibble交错对照在[quantization.py](../../examples/reference/quantization.py)。这不是整个GGML backend或模型速度验证。

第一次precision客户端把NumPy float16 scalar放进mx.array嵌套list，发生frontend ValueError；改用精确等值的Python float后完成采集。这个编程错误保留failed receipt，**不是硬件不支持FP16**。

## 复现与采用边界

通用bench、原始BF16有界reader、、超时监督均随techniques分发；真实模型/capture只留ignored路径，不新增生产依赖或默认路由。

当前仍保留GPU原路径。重新尝试的条件包括更合适的M/packing、原生消费者、编译模式、真实counter/指令分析、memory/IO约束或新GPU；需重建source/精度/所有权和完整请求证据。下一步还有GPU attention、native GPU/ANE交接、ANE图搜索与profiling、VAE/媒体/内存组合，不能把本页当整体研究完成。
