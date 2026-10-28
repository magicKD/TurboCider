# MNN 真实 Metal 引擎：卷积、Winograd、resize 与 replay

<!-- knowledge
id: technique-metal-case-studies-mnn-engine-convolution-and-resize
type: case-study
stage: vae, pipeline
operator: conv, gemm, layout, scheduling
backend: metal, cpu-reference
device: cpu, gpu
symptom: compile-bound, launch-bound, wrong-layout, precision-loss, measurement, stale-cache, lifetime
semantics: numerically-equivalent, contract-only
representation: fp32, fp16, packed
boundary: cpu-gpu, producer-consumer
api: public, private, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU 导航](../README.md) · [前轮源码深读](../mnn-metal-deepdive.md) · [固定来源](../sources.md#mnn-engine-sources-20261016) · [验证快照（过程原件已移出）](../../MAINTENANCE.md)

2026-10-16，第九轮。结论不是“支持 Metal”，而是：固定 MNN 引擎已实际执行完整的小型卷积消费者；**全部 Metal 输出通过预先设置的组件门禁，但整个矩阵没有全绿**。CPU 自动卷积分支保留了数值失败。Winograd 对中等规模有收益，对小块则可能更慢；热 Session resize 与新 Session 也不必重新选择同一种算法。

## 1. 本轮真正执行到哪一层

固定 `alibaba/MNN@dbf9ba61b08106a3b43426e295d1a29cb41cc8fe`，版本字符串3.6.1。源文件、源/探针/加载库的 SHA256、环境与构建选项见来源 JSON。不是 pip wheel，也不是重新实现几个原始 shader。

在独立任务副本以 Release 构建 `MNN_METAL=ON`，关闭本轮不需要的 TensorOps、SME2、转换器可执行工具、分离构建和内部功能。保留 CPU、Express、Session、形状/几何计算、Metal 注册和运行时。参考 checkout 未修改；不读取受限私有目录。未安装全局包、下载模型或改变系统设置。

本机只有 CLT，仍能构建和运行：[`MNNMetalContext.mm`](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/metal/MNNMetalContext.mm) 将嵌入的 shader 交给 `newLibraryWithSource`。缺少离线 `xcrun metal` 不是这条引擎路径必然不可用的证明。第一次稀疏副本遗漏 Express 所需的 `tools/converter`，配置失败；补齐**任务副本路径**后构建成功，未修改上游 CMake。

实际数据流：

```text
NCHW 输入 → Metal Raster/C4 → Conv(k×k) → ReLU → Conv(1×1, CO=5) → Raster/NCHW → CPU 校验
```

Express 只负责构造并序列化合成图；执行用原始 Interpreter/Session。每个 Session 的后端查询及每条 `executeBuffer.command` 的实际 `Execution::backend()` 都必须符合请求；五个执行节点逐一检查。还使用 `dladdr(getVersion)` 验证实际加载的库路径及其文件 SHA，避免只 hash 一个没有加载的 dylib。该证据比配置字符串强，但仍不是硬件 counter / 每 kernel 时间。

## 2. 矩阵、oracle 与失败账本

六臂：CPU High 自动 / 禁用 Winograd，Metal High 自动 / 禁用 Winograd，Metal Low 自动 / 禁用 Winograd。CPU 固定1线程；Metal 固定显式 tuning mode。High/Low 是 MNN 精度选项，逻辑 tensor 的 float32 type 不代表 Low 的物理存储仍是32位。

| case | N / CI / H / W → CO | kernel / stride | 要覆盖的分支 |
| --- | --- | --- | --- |
| pointwise_tail | 1 / 7 / 17 / 19 →13 | 1 /1 | 1×1、C4尾部、Winograd空臂 |
| winograd3_tail_batch | 2 /17 /15 /17 →19 | 3 /1 | batch=2、channel/tile双尾部 |
| winograd3_small | 1 /3 /9 /11 →5 | 3 /1 | 小空间、低channel的特殊准入 |
| general3_low_channels | 1 /7 /33 /35 →13 | 3 /1 | 空间较大且channel不足16，普通卷积空臂 |
| winograd5_tail | 1 /17 /9 /7 →19 | 5 /1 | 5×5变换与边界 |
| stride2_control | 1 /17 /17 /19 →19 | 3 /2 | Winograd不满足stride要求 |
| conv3_medium | 1 /32 /64 /64 →64 | 3 /1 | 中等工作量、完整消费者 |
| fresh_weight_control | 同第二行 | 3 /1 | 同shape、独立新权重/模型身份 |

权重和输入是有限、确定性的周期合成数据，不是自然图像或随机正态分布。seed17/29，输入A/B交替。每次运行前重新上传相同输入，避免 Session 内存复用使重复调用读到已变化的数据。每臂先保留first，再做5次不计入正式统计的warmup；正式每臂8次，轮间反转六臂顺序。同一seed的replay开/关采用17先开后关、29先关后开，禁止并行跑自己的GPU基准。

独立 NumPy FP64 直接卷积实现 SAME padding→ReLU→1×1，使用**原始完整 FP32 参数/输入**，不以另一引擎结果作为唯一oracle，也不只查少数像素。原始数组、模型和输出保存在忽略目录；可携带 JSON 保留每个输出的完整指标及SHA，不能在没有原始数组时假装重新校验了它们。

预设门禁为 High：relative L2≤`5e-5`且max abs≤`1e-4`；Low：relative L2≤`1e-2`且max abs≤`3e-2`。这些是本次合成组件的筛选线，不是图像质量标准。没有因CPU失败放宽它们。

- 正式四进程共1776个完整输出：1536个热态、192个first、48个resize/fresh对照。1200个Metal输出、288个CPU direct输出全部通过；288个CPU自动输出中82个未过严格线。
- 另四个带观察器/Fast tuning的进程共624个输出，含432个Metal输出全部通过；CPU自动分支另保留28个失败。这组绝对时间不作为正式基准。
- 全部八进程共448个负控：旧输入输出对新输入参考、元素数相同的错误NCHW/NHWC解释，都被拒绝。早期配置失败、编译探针的ARC ownership错误和严格验证失败日志保留，没有混入正式样本。

全输出最大relative L2：Metal High direct `1.84e-7`、High Winograd `6.56e-7`、Low direct `7.54e-4`、Low Winograd `2.16e-3`；CPU direct `2.35e-7`，CPU自动 `2.51e-4`。

### CPU失败不能被“CPU oracle”四个字抹掉

失败集中在三个3×3合成case的CPU自动分支；同输入/权重、同线程/High精度，设 `WINOGRAD_MEMORY_LEVEL=0` 后通过。固定 [`ConvolutionFloatFactory.cpp`](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/cpu/compute/ConvolutionFloatFactory.cpp) 的该门实际绕过Winograd，因而这个消融支持“差异与自动变换路径关联”。本轮没有定位到CPU具体变换单元/指令，也没有证明内存bug、上游回归或真实模型不可用；不能把关联升级为根因。

Supervisor返回0仅表示所有预定进程结束、证据保存成功。各 `verification.json` 仍是 `failed`，机器可读 capsule 保留82/28失败计数。比较Metal两条通过的臂，不等于把整套实验宣布合格。

## 3. Winograd加速的是乘法，新增的是变换与暂存

[`MetalConvolutionWinograd.mm`](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/metal/MetalConvolutionWinograd.mm) 使用输出unit2：3×3变为4×4域，5×5变为6×6域；执行source transform→`matmul4x4`→dest transform，再交给本轮真实ReLU/1×1消费者。权重预变换、C4/tile padding和暂存不能从成本账中删掉。

Metal的物理C4布局在这里是 `[ceil(C/4), N, H, W, 4]`，不是把batch放最外层的同名打包。`MetalRaster.mm` 的索引明确含 `c/4 * batch + n`；batch=1时两种错误解释可能碰巧相同，因此本轮必须有N=2、CI17/CO19。CPU学习合同单独验证这条映射，不声称已读取所有GPU padding guard。

生成并实际编进库的 [`AllShader.cpp`](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/metal/AllShader.cpp) 是这轮shader来源；不能只读 `shader/*.metal` 就忽略嵌入产物。Low下M/M4矩阵是half，`FLOAT4`累加器是float，但表达式先形成 `w4x4*i4x4[...]`再转换成FLOAT4；这不等于每个乘法/中间量都FP32。变换域存储和输出又有cast。我们测到了误差包络，没有通过IR证明每条算术的实际lowering。

### 正式、无观察器、replay开启：forward+完成等待的中位数

下表单位µs，分别列seed17/29。计时包括原始Session提交及输出完成等待，**不是kernel/GPU busy时间**；每次输入上传在该timer外，下载另计，完整upload→download样本也保留。

| case | High direct | High auto | Low direct | Low auto |
| --- | --- | --- | --- | --- |
| pointwise_tail（空臂） |112.3 /110.1|115.0 /123.9|111.0 /112.9|107.7 /110.3|
| winograd3_tail_batch |134.9 /129.9|141.6 /143.3|133.1 /132.1|136.9 /137.2|
| winograd3_small |96.1 /94.5|124.4 /122.9|110.5 /110.0|120.4 /116.5|
| winograd5_tail |159.8 /161.2|150.5 /150.0|157.3 /156.8|138.2 /141.2|
| conv3_medium |246.3 /250.6|197.9 /198.0|183.5 /186.4|170.5 /175.8|

中等3×3 High的本矩阵约1.24–1.27×，小3×3反而较慢。1×1空臂两配置本来不改变Winograd路径，仍可漂移约十余µs；因此5×5的小差距、replay的小差距尤其不能直接变成通用阈值。CPU direct的小块也只有数µs，不能用“GPU一定更快”的前提解释。这里只给固定同步边界的合成数据，不做跨引擎/完整VAE排名。

## 4. resize不会自动把creator重新跑一遍

先对CI3/CO5、9×11建Session，初次满足小空间Winograd准入。热态将同一Session改为19×21；后者若**新建**Session，不满足“channel都≥16或空间都≤16”的准入条件。

独立观察进程用Objective-C方法替换包装 `pipelineWithName:fp16:`，只记录原方法请求，不改shader、参数或返回值。活Session resize后仍请求Winograd source/matmul/dest；相同模型在第一次resize前设成19×21的新Session则请求普通 `conv_z2`。High/Low、两个seed、replay开/关均保留完整输出校验。

[`Pipeline.cpp`](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/core/Pipeline.cpp) 的 `executionCache` 会复用既有Execution，再调用其onResize；因此creator的shape准入条件不是每次resize的重新选型承诺。该现象本轮**数值正确**，不是已复现的bug。它说明动态shape实验要同时测“热resize”和“fresh初始化”，否则可能把不同算法的差异归给cache或调参。

暂存Tensor对象的持久化、buffer/offset改变与replay验证见原Winograd实现。这里覆盖的是有界空间变化，不是任意销毁/重用后raw Tensor指针安全，也不是LLM KV状态更新或fault恢复。

## 5. replay与tuning：观察器不是dispatch计数器

正式计时使用未profile的同一dylib。另组Fast tuning观察器保留2492个原方法请求事件的聚合计数；四进程与正式进程使用相同binary/library/probe身份。replay关闭时，每进程3×3 source kernel在正式两轮仍有24次方法请求；replay开启后该阶段为0，前几轮warmup仍有录制前调用。结合原始 `MetalExecution.mm` 的调用分支，支持正常onEncode被replay绕过，**不表示GPU少执行了24个dispatch**。

该机制仍是host encoder重发，不是ICB/GPU自行回放，也不证明不同语义op被融合。`MNN_METAL_OP_PROFILE`未启用；源码对该模式禁用replay的边界仍保留。观察器和debug打印的时间不混入前表。

正式每个seed的444个保留输出，在replay开/关两个独立进程间SHA全部相同。这是本矩阵的完整输出复现证据，不是所有shape、状态算子或设备fault下的逐位保证。

### 原始调优score甚至不是已经换算好的µs

`MNNMetalContext.mm::timeUsed` 实际取 `mach_absolute_time` 差再除1000，包含commit和wait；它没有在该函数换算 `mach_timebase_info`。本机numer=125、denom=3，因此一个score单位对应约41.67µs的host跨度量化，**不能读成1µs物理GPU时间**。同设备上的相对比较仍可能有用，但整数截断、候选平局和阈值的单位必须另审。

None调优路径未测量时保留 `UINT_MAX` sentinel，不能当巨大的真实时延。Fast进程的缓存条目保留GWS/group/local和原始score。进程first create在最终正式矩阵约8.8–11.9ms，而最早一次观察曾约2.08s；没有清驱动缓存，不能都称为“冷编译”。initial create、onResize/tune、first、warm/replay、D2H分别记录；本轮也没有加载/导出持久调参cache。

### architecture名字不等于可靠的SKU表

本机公开device名是Apple M4 Max，architecture为`applegpu_g16s`。同一引擎记录 `high_bandwidth_m4_heuristic=true`，而该读本的注释以M4 Pro校准解释`s`。因此不能仅凭注释将`s`硬编码为Pro，或将该heuristic当带宽实测。本轮`tensor_ops=false`，没有M5/NAX证据；`prefer_m64=true`也不证明当前每个卷积执行了M64 kernel。

## 6. 可复现入口与后续判据

本地CPU代码：labs/mnn_contracts.py；测试：test_mnn_engine_research.py。原创模型只检查C4布局、creator准入和timer单位，不替代原始引擎。设备CLI：Python驱动 / 原生harness；归档器：curate_mnn_engine.py。

使用新输出路径。prepare会显式联网取固定源码；compile不隐式下载；run/matrix不安装依赖。另开新的观察矩阵加`--observe --tune 16 --pairs 2`；不能和正式GPU矩阵并行。原始最终目录前缀为`outputs/metal-mnn-engine-20261016-final`及`-final-observe`，各有owned launch receipt。其他早期目录保留，未计为最终样本。

后续仍需研究：持久cache加载后的exact/close-shape grid合同；大暂存split；CPU自动分支具体误差来源；独立GPU时间/counter与原生完整VAE质量。backend自报内存本矩阵最大约8.13MB，但不是physical footprint或多轮teardown泄漏证明。下一轮应优先给close-shape hint加真实消费者/边界负控，而不是仅凭cache命中宣布新shape合格。

后续[第十轮持久cache实验](mnn-persisted-tuning-and-grid-coverage.md)已执行第一项：近似命中复用旧grid的尾部错误被完整消费者复现，exact/fresh/反向及无observer自然控制分账；不改变本页历史矩阵没有加载持久cache的事实。

本轮依据MNN Metal/测试技能分离观察与正式测量，按debug/复盘要求保留未过门禁，不编辑参考技能或上游实现。凡source、库/编译器、SDK/OS/SKU、shape/layout/dtype、缓存策略、Session生命周期或完整消费者改变，都要复验。本轮组件进展不代表持续研究结束，更不授予生产默认启用资格。
