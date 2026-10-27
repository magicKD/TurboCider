# MLX编译与跨runtime FFN：完成、缓存和输出owner分别比较

<!-- knowledge
id: technique-gpu-ane-case-studies-mlx-compile-and-cross-runtime-ffn
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [同数学强基线](tinygrad-strong-baselines-and-paired-ffn.md) · [来源](../sources.md#mlx-compiled-ffn) · 实际数据 · nullary计划 · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17；本机M4 Max、macOS26.6.2，既有`mlx`/`mlx-metal`均0.32.0。官方tag v0.32.0指向`7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`，与本库reference目录名分别核实，不称最新版。tinygrad沿用固定3d057f0及支持的非默认CPU:CLANG host编译器。

本轮比较相同生成权重FP32 SwiGLU，在各框架自己的**同步完成边界**计时；并实际检查Python scalar缓存、输出保留及首次tracing抛错。不是所有引擎排名、真实模型、ANE或互操作零拷贝实验。

## 先读结果与限制

- 四个主进程按tiny-a→mlx-a→mlx-b→tiny-b顺序串行，各576次，合计2304次/47185920个F32值位置完整readback。全部完整输出/去残差branch满足前轮同一预算，最大绝对误差约1.48e-6。
- MLX的同进程`mx.compile`在这些形状有局部收益，但跨进程绝对时间明显变化。跨runtime只有两个独立窗口的范围，不造同窗口paired ratio，不由某一列宣布全局赢家。
- MLX显式Python scalar参数2→3→2的trace计数为1/2/2；selected旧输出引用保持旧值。tinygrad重放仍复用同一capture Tensor，旧引用读到新值。差别是调用合同，不是其中一方自动拥有并发/zero-copy资格。
- 首次trace失败需保留空cache资格；PR #3635的关键是nullary输入。安全Python analogue实测前两次均抛错并重trace，第三次返回3；非空输入恢复测试单独通过不足以覆盖该PR触发条件。

## 一、源码、tag、wheel是不同身份

稀疏/promisor参考库虽然有正确HEAD，但部分Python/docs blob并未本地存在。`git archive`触发隐式获取，代理连接失败；没有把目录名/可列出tree当完整内容已取得，也没有继续给参考库补对象。改在本轮新目录下载官方完整归档，按固定Git对象核全部918个blob，18文件人工读本另列。

既有`core.so`、libmlx/libjaccl、metallib及两distribution的METADATA/WHEEL/RECORD分别fingerprint，设备后重核字节未变。**version匹配＋组件hash不等于已证明wheel完整build由该commit产生**；本轮未重编MLX、不安装/升级依赖或checkpoint，不修改上游/参考源码、系统设置或产品后端。

同tag的lazy文档讨论默认动态造图，其中一段把compile写成未来方向；同tag却已有独立compile指南和明确实现。因此不能用孤立段落否定`mx.compile`存在，也不能把默认lazy与显式compile当同一入口。最新live文档和安装0.32.0的API/行为也不混用。

## 二、`mx.compile`做什么：有边界的图变换

| 层次 | 读本实现 | 必须分开的资格 |
| --- | --- | --- |
| Python入口 | pytree flatten；array成为输入，int/float/string/None进入constants key | 显式参数变化与mutable closure变化不同，不是任意Python都动态执行 |
| frontend cache | thread-local，以fun_id、default stream/device、shapeless、rank/shape/dtype及constants匹配 | 不是checkpoint内容hash、跨线程/跨进程性能或质量cache |
| graph simplify | scalar等价、公共工作合并，当前compile路径做3次simplify pass | 优化存在不等于所有表达都被融合 |
| fusion | 当前whitelist为unary/binary/ternary/broadcast；depth11、输入数24等边界 | MatMul/Reduce不在此通用whitelist，不能说整FFN变成一个shader |
| replay | `compile_replace`把trace placeholders替换成实际arrays，并创建新的输出描述 | Python函数不重跑，不表示没有runtime/分配/提交工作 |

benchmark显式`mx.enable_compile()`、`shapeless=False`，default device设GPU；记录Python body计数。每个geometry中lazy热区增加144次，compiled热区不增加；后者warm时只trace一次。tinygrad的两个JIT arm在hot阶段也不重跑Python。

`shapeless=True`仍检查rank/dtype，并非任意shape支持。本轮始终由own client先核精确shape/dtype，不提交缩小输入；只读源码不能替代[前轮发现的extent边界](tinygrad-metal-hcq-and-replay.md)验证，更不能给跨框架动态shape安全资格。

MLX提供显式captured `inputs/outputs`接口供state参与变换，但闭包中的Python状态不自动成为所有cache key。这里weights固定且先eval为可用arrays，不测试模型update、LoRA替换或captured state的所有生命周期。

## 三、完成边界：lazy造图时间不能与GPU完成时间比较

MLX `eval`对未调度输出运行`eval_impl(...,false).wait()`，对已有输出也调用array wait；array的available/evaluated状态与event分开。正常Metal路径等待共享event，并在wait前后检查错误。

本轮timer包围：

| runtime | 计时区内 | 计时区外 |
| --- | --- | --- |
| MLX | 函数调用＋原`mx.eval(output)`的同步完成 | resident准备、warmup、完整readback/oracle、下一轮前idle synchronize |
| tinygrad | 函数调用＋原`MetalDevice.synchronize()` | 相同类别的准备/readback/oracle |

没有只量`fn(x)`的lazy图构造，也没有机械给MLX在timed eval后再加一次可能提交空CB的queue synchronize。语义上都等所测输出完成，底层API、host工作与kernel实现并不相同，故这是完整调用控制，不是纯GPU kernel计时。

Metal错误路径有error poisoning及手工signal以防永等；signal不等于成功，tracing异常也不等于GPU失败。源event wait有无期限等待，本轮外围supervisor120s有界，但没有注入hang/GPU错误、取消或证明物理drain。正常调用的成功数值不授予这些边界。

## 四、同FP32数据，不冒充同shader或最佳storage

沿用前轮36份F32/FP64 fixture和预算：M64/256、D128、F256，seed41/53，完整数学为：

`Y=(SiLU(XWg+bg)*(XWu+bu))Wd+bd+X`。

Wgu[D,2F]每row按gate/up拼接，down bias/residual各一次。每次完整F32输出检查最大绝对误差≤2e-4、相对L2≤2e-4和去残差branch相对L2≤3e-4，全部finite。实际branch误差最大约2.86e-7。

- MLX两个arm使用native biased-linear配方`addmm`，SiLU为`g*sigmoid(g)`，分别lazy和显式compile。
- tinygrad两个arm沿用paired与进一步融合的Tensor表达；名字中的3/2来自前轮同图ICB观察，本轮没有重新采driver command trace。
- MLX显式`MLX_ENABLE_TF32=0`，tinygrad逻辑FP32/TC1；没有借F32输出dtype掩盖不同许可的TF32配方。M4下不把此配置当M5 NAX执行证据。
- 共同使用同一row-major Wgu/下投影storage。它借鉴Linear的addmm配方，**没有实例化NN Linear模块或采用其默认[out,in]权重再转置的所有storage选择**；也没有搜索各框架最佳layout/tile。

固定源码还按shape、dtype和NAX availability选择matmul路径；不能从source分支推导安装wheel真的执行了某条Split-K/NAX指令。软件版本相符与数值通过不是底层trace；所有参数是生成权重的小FFN，不是checkpoint/真实模型质量。

## 五、跨进程ABBA：给范围，不拼同窗口配对

两库分进程，避免同一进程混合编译器/全局状态。外层固定A/B/B/A，每个进程内部两arm以ABBA平衡，12block×每arm12次×两geometry＝576次；每次seed交替，先warm/capture，完整readback在timer外。外层调用间存在准备/检查间隔，ABBA不能消除温度、频率、CPU负载或外部瞬态；不是同时测量。

以下为两个进程各自调用时间中位数的[min,max]，单位µs；不是置信区间、每call极值，也不是cross-runtime paired ratio：

| M | tiny paired | tiny fused | MLX lazy | MLX compile |
| --- | --- | --- | --- | --- |
| 64 | 244.1–253.8 | 249.7–257.4 | 203.6–217.3 | 190.8–200.5 |
| 256 | 774.8–774.9 | 775.4–784.4 | 610.4–751.9 | 598.0–722.5 |

这个明确配方下MLX区间较低，但M256的MLX两窗口本身差异明显，不能推广到大D/F、量化、其他SoC/版本或全模型。CPU:CLANG host override、原生addmm与不同kernel算法也包含在成本里，不把差别全归因于Python/C++或某种矩阵单元。

同进程、同block的配对统计另算：MLX lazy/compile在A/B的中位ratio，M64约1.085/1.068、M256约1.043/1.024；tiny paired/fused分别约0.984/0.980和0.988/0.997。所有block/timers均保留，不挑最快一次、不把不同统计口径互换。

两个库均累计完整检查47185920个主F32值位置/188743680bytes；相同payload按runtime/process/arm/seed去重，不把2304次调用冒充2304份独立输入。未额外保留每次独立输出GPU快照，consumer是“本次读完再调用下一次”，不是多请求持续并发。

## 六、输出owner：同数学不代表相同保留合同

在相同有效shape上，先完成seed41并保留对象，再调用seed53，随后重新读取旧引用：

| 观察 | tinygrad capture返回 | 本轮MLX编译返回 |
| --- | --- | --- |
| old/new是否同一个Python对象 | 是 | 否 |
| 旧引用是否仍为seed41值 | 否 | 是 |
| 旧引用是否变成seed53值 | 是 | 否 |

完整before/new/after数组均保存并独立检查。它验证的是所选串行场景的**逻辑值保留**，不是driver backing identity、zero-copy、跨stream同步或并发consumer安全。

本benchmark每次立即消费完成的输出，没有给tinygrad加额外clone。若真实GPU→ANE→GPU消费者需要跨下一次调用持有旧值，应另计clone/独立slot和完成交接；不能不匹配owner合同便直接套这张表。反过来，MLX旧值保留也不表示任意captured outputs/state更新都无需owner/generation设计。

## 七、显式scalar与闭包：两种缓存语义

MLX的显式Python scalar参数2→3→2，各次输出都匹配请求；trace body计数1/2/2。Python binding把普通scalar值放入constants向量，所以新常量创建/选择不同cache entry，而不是每次把Python函数全部重跑。

这不同于前轮tinygrad普通Python scalar被固化到capture值的控制。不能把“compile/JIT”当统一接口；显式参数、Tensor/Variable、mutable closure及model state要各自验证。新常量的重trace成本也不能隐藏在warm计时外后宣称免费动态参数。

## 八、PR #3635：真正触发条件是空输入

[官方PR #3635](https://github.com/ml-explore/mlx/pull/3635)在2026-10-17 API观察为closed、merged=true，合并时间2026-06-09T23:56:04Z，merge`01368d8e7888d6989969aa82bc36f2ba09dc5ced`。两文件patch全文读取：把`entry.empty=false`从trace前移到trace/simplify/fuse全部完成后，再加原C++回归。

原测试是nullary function：第一次trace抛错后，半填entry的placeholder inputs仍为空；第二次同样空输入可能命中坏entry，跳过trace。非空输入重试反而可能因输入数量不匹配而新建entry，即使没有修复也能重trace。因此本轮早先的非空恢复控制**不足以证明这个PR触发条件**。

主矩阵后另跑安全own Python analogue：同一个无参compiled function连续两次在array数学之前抛错，两次均传播且trace计数1/2；允许成功后第三次重新trace并正确返回3，累计计数3。原C++回归只读，未编译运行；没有PR parent/head A/B或原套件通过声明。

scalar3可能被常量折叠，requested GPU也不能证明该scalar真的在GPU/ANE算。此结果只给当前wheel的所选frontend缓存恢复资格；不包括编译中任意错误、GPU执行失败、内存泄漏、取消或drain。

## 固定来源与复现

- [CompilerCache/fusion/replace](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/compile.cpp)、[Python编译入口](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/python/src/transforms.cpp)、[原C++回归](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/tests/compile_tests.cpp)。
- [eval/同步](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/transforms.cpp)、[array状态](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/array.cpp)、[Metal event](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/backend/metal/event.cpp)、[matmul选择](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/backend/metal/matmul.cpp)。
- own FFN client、nullary控制、离线比较合同、capture；fixture可用前轮CPU工具重建。

设备运行必须用新目录和`guarded_run.py`：五次低于5%采样、缺计数拒绝、有界supervisor。raw根`outputs/gpu-ane-mlx-ffn-round29-20261017`保留官方918blob归档、package组件指纹、六次guard的stdout/stderr/owned receipt、所有actual/timers/controls；30个preflight全0%不等于全局独占。本轮未采GPU编译/指令trace，warmup不能替代证明热区绝无底层driver活动。

采用建议：固定语义、精度许可、layout、完成和owner后，再比较原生强基线；把within-process优化和cross-process窗口范围分开。没有真实checkpoint/LoRA、最佳storage/tile、M5 NAX、ANE/overlap、跨runtime共享buffer、能耗、峰值内存或错误恢复/drain资格，不把本页数据写成全局调度常量。
