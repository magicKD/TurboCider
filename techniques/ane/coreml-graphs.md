# ANE / Core ML：图、shape 与布局

<!-- knowledge
id: technique-ane-coreml-graphs
type: method
stage: ffn
operator: swiglu, conv, layout
backend: coreml
device: ane, gpu
symptom: graph-interruption, wrong-layout, compile-bound
semantics: numerically-equivalent, approximate
representation: fp16, w8a8
boundary: gpu-ane, weight-runtime
api: public
-->

[总目录](../README.md) · [生命周期](lifecycle.md) · [共享 IO](../gpu-ane/shared-memory.md)

深入阅读：[ANE专题目录](README.md)、[Apple Transformer布局/lowering](layout-and-lowering.md)、[ANEMLL状态与公共部署](public-coreml-deployment.md)。版本能力和实际放置分别见[能力地图](capability-atlas.md)与[本轮压缩/计算计划对照](experiments-20261016.md)。

## 先选图合同，不先选 computeUnits

| 路线 | 权重在哪里 | 优点 | 主要代价 |
| --- | --- | --- | --- |
| checkpoint-bound frozen | 常量嵌在每层图 | 不必每层 stage，base 热路径可快 | 多图/多 bucket 的磁盘、load、驻留；绑定 checkpoint / LoRA |
| runtime FP16 weight | 运行时矩阵输入，同形状跨层复用 | 不为每层/adapter 编译；base slots 不含 LoRA | 每层权重转换/staging、host 边界、FP16 范围 |
| runtime W8A8 | INT8 code inputs + scales、固定变换图 | 减少传输；共享 GPU stager / bounded banks | 旋转/QDQ/restore 和误差；Public/Private ABI 不同 |

对应实现：[coreml.mm](../../native/backends/coreml.mm)、[ane_runtime.hpp](../../native/backends/ane_runtime.hpp)、[FP16 exporter](../../tools/coreml/export_runtime_ane.py)、[Public W8 exporter](../../tools/coreml/export_runtime_w8a8_ffn.py)。不能把一个新模板替换进旧 manifest 后继续沿用旧资格。

## ANE-friendly layout 的思路

Apple 的 [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)讨论了面向 ANE 的 tensor layout 与算子表达。工程上值得比较固定 shape 的 channel-first / 1×1 convolution 表达，避免多余 transpose 和不必要的 sequence-wide reduction；但不要把这篇文章的模型/设备收益直接移到视频 DiT。

本仓库 frozen FFN 采用 `[1,channels,1,rows]` 逻辑 shape 配合明确 strides，让 row-major 模型数据与 Core ML feature 对齐。runtime FP16 模板则是显式 `[rows,hidden]` matmul 输入。两种 ABI 不能混用，也不能只改 shape 不改实际 pitch。

内部 K/N tiling 应在一个 graph/prediction 中完成，不把每个 reduction chunk 拆成一次 host prediction。gate/up/激活/down 尽量保留在同一图，只有最后 hidden partial 跨设备；对 LoRA 则按完整 hidden ABI设计。

## 固定 bucket 与 padding

- 选能容纳实际 rows 的最小已验证 bucket；rows 来自分词/图像 patch，不由输出尺寸单独决定。
- padding 必须初始化，尾 rows 不能读旧输入；输出只发布真实 rows。检查 bias / mask / norm 是否使 padded 行影响真实行。
- `512→1024→512`、换长 prompt、多参考图和 partial final chunk 都是必要回归。
- flexible shape 未必比多 fixed buckets 快；需要测编译、第一次 prediction、steady prediction、缓存占用和 fallback。

## compute units 与实际设备

`CPUAndNeuralEngine` / `cpuAndNeuralEngine` 排除 Core ML 使用 GPU，但仍允许 CPU；不证明每个 op 跑在 ANE。[MLComputePlan.DeviceUsage](https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n/deviceusage)可检查支持/首选设备，是计划信息，不是连续硬件 occupancy trace。

执行验收需要三层证据：manifest/API 允许 → actual call/IO/recipe 确认真的执行 → 独立 device trace（若要声称物理占用/并发/INT8 MAC）。缺第三层时准确报告 host spans 和硬件放置未验证。

## FP16 范围与 headroom

BF16 和 FP16 同为 16 bit，但指数范围不同。把 BF16 FFN 简单转 FP16 可能 overflow；必须保留有限性、归一化 scale / headroom retry 和完整失败重算。缩放需要贯穿 gate/up/hidden/down 与 LoRA 修正，不能只缩输入后忘记恢复 bias/adapter。

不要用截断激活、跳过 SiLU、放松 finite 或直接更换 precision 来“修”慢图；那是新的数学配方，需要新的 quality gate。

## 有成本风险、可带对照重新探索的图形

- 每个 FC2 reduction chunk 单独 prediction：dispatch 和 partial reduction 税可能超过收益。
- 把长 attention/softmax 整个搬进 public Core ML：能 convert 不等于快。
- 把整个模型浮点展开以方便 staging，却仍称低内存 packed 路线。
- Q/DQ 图有 INT8 类型就声称硬件 INT8 compute；应区分 storage、activation、accumulator 和实际执行证据。

这些是实验风险而非永久禁区：新SDK/设备、融合段、更合适的bucket/K分段、sharedIO或消费者改变后，可以带完整traffic和质量对照重新评估；API/来源/finite/完成验证仍保留。

测试模板的 graph 结构、dynamic input、tail、headroom、错 shape/stride、corrupt artifact、base→A→base、晚 chunk failure。完整模型采用仍由 [测量门禁](../profiling/measurement.md) 决定。

## 实现参考与实验

### 图内K/N tiling：由真实消费者决定是否值得

[ccv来源卡](../sources/repos/ccv-coreml-rowwise.md)给出K拆分与surface政策；[vpipe来源卡](../sources/repos/vpipe-runtime-tiling.md)区分图内分块和host多次调用。核心代数是先完成每个投影的K归约，再做SiLU；N切分须保持gate/up配对以及down对应列。不要将非线性移进K partial，也不要重复bias/residual。

新的图构造与完整消费者驱动提供frozen/runtime、2D row-major/4D channel-major、sigmoid/exp、GPU-only/CoreML/serial/hybrid对照。4D matmul的`[1,1,C,M]`不是旧frozen Conv图的`[1,C,1,M]`，不能只改shape不改pitch。K求和改变舍入，属于需数值验证的等价变换，不声称bitwise。

首次[真实Qwen FFN实验](../gpu_ane/case-studies/qwen-ffn-tiling-and-concurrent-cost.md)使用完整1024×4096×12288与真实layer/timestep输入。CPU_AND_NE sigmoid路径被固定抽样门槛拒绝；同bundle CPU_ONLY对照后来满足门槛，不能将负结果泛化为所有CoreML图。exp通过组件数值，但完整CoreML消费者和半宽hybrid仍慢于compiled fused GPU。相同切分的并发相对串行有约1.16×收益，也不能据此替换更快的全GPU分母。

这些是FP16/Python bridge/resident-weight结果，不是原生shared IO、W8A8或LoRA结论。具体失败、全部样本与重开条件在证据中；优化图的最终目标是完整消费者/请求，而非提高计划中的ANE op占比。

可读实现：参考代码；[实验步骤与边界](../examples/README.md)；本项全部 case / controls。

本项控制：frozen/runtime inputs；real native INT8 IO rather than Python cast。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：new graph lowering/shape/compute policy/feature bridge。历史负结果限定于其来源/shape/边界，不关闭该方向。
