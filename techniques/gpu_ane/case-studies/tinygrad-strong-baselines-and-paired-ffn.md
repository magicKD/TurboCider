# tinygrad强GPU基线与完整paired SwiGLU：少一个command不等于更快

<!-- knowledge
id: technique-gpu-ane-case-studies-tinygrad-strong-baselines-and-paired-ffn
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [前轮提交实验](tinygrad-metal-hcq-and-replay.md) · [来源](../sources.md#tinygrad-paired-ffn) · 确认计划 · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，继续固定tinygrad `3d057f05df532deca7661bfb2400c6564f96a04a`、M4 Max/Apple9、macOS26.6.2/SDK26.5。不改上游、旧轮源码/cache或产品后端。原Metal/TinyJit实际执行，CPU:CLANG仅编译host提交程序，是已声明的非默认override；PROFILE/DEBUG/VIZ/BEAM0、TC1。没有ANE或M5 NAX执行。

本轮补齐前轮的关键限制：8-kernel小链不是强GPU分母。先比较融合/代数表达，再比较含三次或两次matmul、SiLU、gate/up配对、bias和residual的**完整数学FFN**。使用生成权重的小形状，不是真实checkpoint或全模型质量/性能资格。

## 结论先行

- 同一数学改成单kernel后，eager/JIT差距显著小于原分阶段链。两者都用JIT时，8→1 command只有约1.1–1.2×的局部差距；不能用原约18×作异构或模型提速分母。
- 完整FP32 SwiGLU的gate/up可沿output维合并；selected GPU输出及去残差分支均通过预声明预算，最大绝对误差约1.48e-6。可复算的完整输出比只看kernel数更重要。
- 原ICB观察为split3 JIT4 command、paired2 JIT3 command、进一步融合2 command；M64的3-command版在两个进程都略快，M256接近持平且统计口径/窗口会改变小幅排序。不颁发“2<3<4所以更快”的通用排名。
- 主1440次计时＋独立确认1152次，每次完整readback/quality校验；确认是posthoc新进程、同一数据，不冒充新checkpoint或独立输入分布。四次五采样准入均0%，仍不能排除瞬时外部工作。

## 一、先把GPU分母做强

有界dyadic FP32输入上，四轮`x*2+.5`与`16*x+7.5`完全同值；本轮所有调用逐字节检查。任意FP32输入上改变运算顺序仍需重新检查舍入，不能从该格点域推导所有模型bitwise等价。

| arm族 | Tensor构造 | 原warm-phase ICB最大command数 |
| --- | --- | --- |
| staged | 每次乘和加都realize，共8边界 | JIT8；eager模板每次1 |
| fused | 同8个数学运算，最后才realize | 1 |
| folded | 直接`16*x+7.5` | 1 |

各族均比较eager和TinyJit。两个长度、两独立resident输入、12个平衡rotate/reverse block、每arm/block6次，共864次；不是把不同数学、精度或输入用同一倍率混在一起。

同步完成时间中位数如下，单位µs；全部输入构造、cold编译/加载和readback在计时区外：

| 元素数 | staged eager | staged JIT | fused eager | fused JIT | folded eager | folded JIT |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 3700.1 | 202.6 | 776.4 | 184.1 | 580.5 | 184.2 |
| 131072 | 3725.4 | 243.9 | 1235.5 | 209.9 | 649.3 | 208.5 |

单kernel的fused/folded eager仍会经历不同量的Python/Tensor构造与schedule处理；同为1 command不表示完整调用成本相同。JIT后两者接近，说明应区分数学/图融合、前端、提交与GPU执行。

证据JSON还保存每个block的配对比值，而不只给各arm总体中位数。这里的局部效果不是模型吞吐、TTFT、纯GPU纳秒时间、能耗或最佳实现证明；真实模型有不能代数折叠的矩阵/非线性/状态依赖。

## 二、完整SwiGLU的数学与paired布局

M为64或256，D128、F256，input/weight/bias/output均FP32，matmul显式指定FP32逻辑累积：

`g=XWg+bg; u=XWu+bu; h=SiLU(g)*u; Y=hWd+bd+X`。

| 对象 | separate布局 | paired布局/消费 |
| --- | --- | --- |
| gate/up weights | Wg[D,F]、Wu[D,F] | Wgu[D,2F]，每一row先gate再up，不能把两个平面简单串成一块 |
| gate/up bias | bg[F]、bu[F] | bgu=concat(bg,bu) |
| projection输出 | g/u各[M,F] | gu[M,2F]；g为前F列、u为后F列，保留正确row stride |
| hidden/down | h[M,F]、Wd[F,D] | 配对后计算同一SiLU(g)*u，down使用对应F轴 |
| 完整消费者 | down bias、residual | bd和X各加一次，不因合并/分支加两次 |

各参数、两seed输入、物理Wgu和全部reference均保存。权重稠密、正负混合；非线性破坏了精确格点结果，故FFN用预声明误差预算而不是宣称bitwise等价。

四种arm分别是：

1. split3_eager：gate、up、hidden分别realize，最后down+bias+residual。
2. split3_jit：相同完整工作，捕获到一个原ICB。
3. paired2_jit：合并gate/up，gu、hidden分别realize，再完整down消费者。
4. paired2_fused_jit：paired表达，仅最后realize；由原scheduler决定中间存储与融合。

实际warm-phase观察的JIT ICB最大command数为4/3/2；**一次realize不等于一个kernel**。这些是原`new_icb`软件层对象的观察，不是物理指令/driver trace。

## 三、GPU矩阵原语：看实际生成物，不只看TC开关

读本的Metal TC表提供8×8×8 fragment布局及F32→F32、FP16→FP32等dtype配对。默认TC1会尝试dtype/shape匹配，再做split/upcast；TC2则可只应用形状而不发WMMA。配置名、能力表与实际路径不能混为一谈。

本轮保留原`MetalRenderer.render`返回的完整MSL和IR观察：实际有WMMA UOps、`simdgroup_multiply_accumulate`与`simdgroup_float8x8`，全部热计时内没有新render或ICB创建。它是运行原Metal数值链＋生成物证据，不是M5 tensor accelerator、独立ANE、硬件accumulator轨迹或最优tile证明。

原SiLU经sigmoid/exp2表达；独立CPU oracle则从**实际F32文件**解码，以FP64直接contraction和exp计算，不调用BLAS。比较的是同一数学，允许F32累积与非线性近似的受控误差，而非伪装成同一步舍入。

## 四、质量门：residual不能遮住坏FFN

预声明三项同时成立：最大绝对误差≤2e-4、完整输出相对L2≤2e-4、`actualY-X`对独立FFN branch的相对L2≤3e-4，另必须全finite。branch包含down bias，但不含residual。

主所有FFN调用均通过；最大绝对误差约1.48e-6，branch相对L2最大约2.86e-7。独立确认的相同arm/seed输出与主payload逐字节相同，不把FFN不同arm之间强称bitwise相同。

若X很大、FFN分支很小，仅看完整Y相对误差可能掩盖分支全丢失；本轮另有标准库负控证明branch gate会拒绝这种情况。没有因此授予真实checkpoint、LoRA、FP16/INT8或图像生成质量资格。这个FP32配方也不能直接与[ccv W8A8边界](ccv-dense-precision-and-scale-floor.md)混用性能分母。

## 五、完整调用成本：更少command并非稳定赢家

主FFN共576次，两seed在每arm内交替；12个block平衡arm位置，每arm/block6次。中位同步完成时间为µs：

| M | split3 eager | split3 JIT（4command） | paired2 JIT（3command） | paired2 fused JIT（2command） |
| --- | --- | --- | --- | --- |
| 64 | 3469.6 | 272.6 | 252.7 | 258.6 |
| 256 | 3878.6 | 758.5 | 702.3 | 724.8 |

看到几％差异后，另开新进程/独立task cache，仅确认两个paired arm，12个ABBA block、每arm/block24次，共1152次，不修改主矩阵、不挑新的赢家输入。

以下统一定义ratio=`3-command / 2-command`，大于1才表示2-command更快：

| M | 主逐block配对ratio中位数 | 主范围/2-command胜出block | 确认配对ratio中位数 | 确认范围/胜出block |
| --- | --- | --- | --- | --- |
| 64 | 0.978 | 0.908–1.012；1/12 | 0.977 | 0.951–1.017；1/12 |
| 256 | 1.001 | 0.909–1.056；6/12 | 1.012 | 0.965–1.044；8/12 |

M64在所测条件下3-command版略快；M256只能记录为近乎持平/窗口敏感。没有GPU频率、带宽、寄存器或逐kernel时钟来证明具体瓶颈，不把差异直接归因于MMA退化、cache或某个编译器机制。

### “中位数之比”和“配对比值中位数”不能互换

M256主矩阵各arm总体中位数之比约0.969，表面上是3-command更快；但同block配对比值的中位数约1.001，已接近持平。独立确认的总体中位数252/258µs（M64）和776/773µs（M256）也不应被解释成全局稳定排序。

初版离线测试错误假定“总体排序反转”就等于“配对ratio也跨过1”，被实际数据否决；旧测试/失败日志保留，修正后分别检查两种口径，原始timers未改。所有block和两种统计均分发，不能事后挑一种口径宣传几％胜出。

## 六、计时、观测与完整数据分账

- 主向量864次＋主FFN576次＝1440次，70189056个F32值位置/280756224个actual字节完整检查。独立确认另1152次、23592960值/94371840bytes；pilot只有warmup/数值控制，不计主timers。
- 每次timer覆盖完整函数调用＋原`MetalDevice.synchronize`。输入resident；构造/编译/加载、完整readback/独立校验在区间外。确认的invoke还包含相同的own shape/dtype检查，两arm一致；不是模型request或持续满负载吞吐。
- actual按geometry/arm/seed去重保存，不能把重复值位置当新样本分布。确认仍是同一权重/输入，只是新进程、更长block，不是独立checkpoint确认。
- 所有主热区内新MSL render/ICB创建事件为0；main用同样的只读wrapper，确认无hot observer。外围readback和CPU校验会改变工作节奏，不能省略这项限制。
- 四个guard进程各五采样0%、owned terminal0；无全局GPU独占或未来外部活动保证。没有安装LLVM/其它依赖、修改系统安全、模型或默认产品后端。

## 七、可重建fixture与固定来源

标准库合同含参数、packing、输入与质量逻辑；可移植CPU生成器在已有NumPy环境下生成36份F32/FP64文件。FP64为`einsum(..., optimize=False)`和exp，另8个scalar fsum spot，完整生成文件已与设备所用fixture逐字节重建一致；不把重复同算法说成第二份独立硬件验证。

仅CPU准备，不触发设备：

```bash
python3 techniques/gpu_ane/generate_tinygrad_ffn_fixtures.py --output-dir NEW-FIXTURES
```
八文件新增读本与原237 blob复核见来源JSON，核心链可追溯：

- [dot/matmul和FP32累积](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/mixin/op.py)、[SiLU/sigmoid](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/mixin/elementwise.py)。
- [TC fragment/dtype表](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/renderer/tc.py)、[启发式选择](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/codegen/opt/heuristic.py)、[WMMA匹配/生成](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/codegen/opt/postrange.py)、[MSL renderer](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/renderer/cstyle.py)。

采用建议：先建立优化GPU分母，再把合法paired代数、完整bias/residual、数据范围和消费者同步放进比较；更少realize/command只是假设。这里没有新PR、PR parent/head A/B、全引擎最快、最佳tile、真实模型、tail/batch/LoRA、ANE/overlap、能耗、峰值内存或取消/drain资格。
