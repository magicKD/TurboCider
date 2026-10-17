# tinygrad Metal HCQ2：ICB提交、重放输入与profiling扰动

<!-- knowledge
id: technique-gpu-ane-case-studies-tinygrad-metal-hcq-and-replay
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [来源](../sources.md#tinygrad-metal-replay) · 计时计划 · 安全修订 · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17；固定tinygrad `3d057f05df532deca7661bfb2400c6564f96a04a`，不是最新版声明。本机M4 Max、Metal实际arch Apple9、macOS26.6.2/25G83、SDK26.5。该ref已经是`MetalQueue + HCQ2`，没有旧教程的`runtime/graph/metal.py`；不能把旧`MetalGraph`类的合同直接贴到当前读本。

本轮实际跑原始Metal/TinyJit，未改上游；own wrapper只记录原ICB创建和queue lowering元数据。显式使用支持的`HCQ_RUNTIME_DEV=CPU:CLANG`：默认CPU:LLVM因本机缺LLVM dylib失败，不安装依赖、不切生产后端，不称默认发行路径性能复现。这个CPU编译器生成提交程序，**不意味着下面的Tensor算术改在CPU计算**。

## 结论与本机边界

- JIT把原8-kernel链捕获到8-command ICB；pilot四次4096值输出均精确，Python函数体计数1/2/2/2。主两种长度、两独立输入、384次调用的25952256个F32值位置全部读回并与独立oracle逐字节相同。
- warm eager与JIT的同步完成计时在这个人为保留8-kernel的例子上差别明显，但这条链可以代数融合。它是**提交/前端开销演示，不是最优GPU分母或模型加速倍率**，更不是ANE贡献。
- PROFILE=1的同一小控制也正确，但8-command lowering多了16个timestamp指令；原代码会把一个CB拆成逐command的CB。profiling不是默认路径的被动旁观者，DEBUG≥2也会进入profile lowering。
- 原JIT会复用返回Tensor对象、冻结普通Python scalar；更小的bare1D输入也可能通过原输入metadata检查。后者只在安全preflight中保留，客户端拒绝实际重放；没有GPU访存安全或修复资格。

## 执行链：把Python提交程序也编译起来

| 层次 | 固定源码机制 | 适用/失效条件 |
| --- | --- | --- |
| Tensor/LINEAR | eager生成schedule；JIT收集多条LINEAR后统一memory plan/compile | 显式realize、复制及依赖影响融合/分批；不是任意Python都动态重放 |
| HCQ2 | `DepsTracker`按buffer/lane/byte区间跟踪读写；batch按device/queue与peer组织 | 有统一内存也不能跳过alias、生产者完成和host参数生命周期 |
| MetalQueue | 参数按alignment排布；GPU地址与scalar成为link-time或runtime patch | 输入地址/变量与固定链接资产分开；本轮没有symbolic launch尺寸测试 |
| ICB | 原`new_icb`建立pipeline、绑定一个args buffer、设置dispatch与barrier | pipeline须支持ICB，local-size受实际pipeline上限约束 |
| 提交/完成 | 编译的host代码发ObjC消息：CB/encoder、fence、execute ICB、signal shared event、commit | `_wait_signal`和synchronize是实际完成门；不能只计Python函数返回 |

`MetalRenderer`发出一个`args_t`，其中包括buffer地址及scalars，通过`constant args_t& [[buffer(0)]]`传入。ICB因此只需一个kernel buffer binding；运行时patch可以更新里面的地址，而不必每次在Python逐kernel重新建立全部绑定。

renderer在Apple family≥7提供SIMD-group MMA，当前所读代码是`simdgroup_multiply_accumulate`等路径；这不是独立ANE，也没有证明使用M5 GPU NAX/TensorOps。pre-Apple9另有pipeline-use workaround；它不表示Apple9之前不支持ICB，本机Apple9也没执行那个分支。

## 链接、池化和owner：不能只说“缓存了图”

HCQ2区分可在link时确定的buffer地址/常量与每次运行才知道的input地址/变量。`_is_link_patch`、`patch`、`hcq_link`以及`LinkCtx.refs`共同决定哪些写入可以提前、哪些refs必须跟着已链接程序存活。eager小schedule还会使用缓存模板、runtime地址patch和device ring，而不是每个op都从零构建所有对象。

JIT的`CapturedJit.linear`有自己的链接生命周期；`held_bufs`保护活Tensor/已分配常量，scratch单独标记；`free_intermediates()`会deallocate并要求后续重链接。Metal的ICB/command对象通过buffer关联，pipeline有cache；真正Metal buffer释放前原allocator调用synchronize。不能从weak字典、函数返回或Python对象失去引用推断任意失败下物理工作已drain。

residency可用时注册allocation，否则走`useResources`表；本轮未单列effective residency分支或验证driver zero-copy。共享storage和GPU地址的存在，不足以证明没有输入/输出copy、layout成本或内存压力。

### PR #18762：从专用nolru分配到共享bufferize

[官方PR #18762](https://github.com/tinygrad/tinygrad/pull/18762)在2026-10-17的API快照为closed、merged=true，合并时间2026-10-12T11:30:49Z，merge `b41c51306bb1f4228e5ecb5ba78b838363442724`。两文件commit patch全文已读，base/head/patch hash见来源JSON。

变更不是简单删除一个标志：它把`LinkCtx`传进bufferize matcher，让ICB使用统一分配出的buffer/视图，绑定时加入`buf.offset`，移除原专用`BufferSpec(nolru=True)`分配。需要一起理解ring、JIT保留链接、偏移及owner，不能只移植一行。上游关于ICB offset与64-bit地址的注释是设计解释，不是本轮已验证的Apple >4GB规格或driver限制。

本轮没有构建PR parent/head两套做性能A/B，下面的数字不能归因给该PR；没有提交上游PR/评论或修改参考checkout。

## barrier、fence和profiling不是同一件事

原`new_icb`给每个indirect command调用`setBarrier()`。Apple官方[setBarrier文档](https://developer.apple.com/documentation/metal/mtlindirectcomputecommand/setbarrier())说明它约束之前commands完成后才执行该command；本轮读取了官方DocC abstract/discussion并保存capture。这个局部顺序合同不能推导为GPU→ANE事件、不同请求隔离或取消后的drain。

encoder外层还wait/update同一个MTLFence；最后CB signal共享event，host的timeline wait才是本轮计时结束点。我们没有移除barrier做竞态负控，也没有采指令或物理command trace。

PROFILE改变了原lowering：

| 配置 | 原8-command队列的观察 | 源码选择的CB组织 |
| --- | --- | --- |
| PROFILE0、DEBUG0 | timestamp指令0；实际创建8-command ICB | 一次执行整个range |
| PROFILE1、DEBUG0 | timestamp指令16；四次完整输出仍精确 | 按command分CB，读取GPUStartTime/GPUEndTime |

这是原方法入口元数据＋源码分支解释，不是读回了8个实际driver提交。DEBUG≥2也能令`compile_linear`启用profile；不能把debug/搜索计时路径与未观测的默认路径直接比较。PROFILE进程正常退出时还有小型时钟对齐helper和task TMPDIR中的profile pickle，另列，不算四个主控制预测或主性能数据；VIZ0没有启动交互server/browser。

## 同输入、同8-kernel工作：局部计时与质量

每个stage显式realize `*2`再`+0.5`，共4stage/8kernel；数学为`16*x+7.5`。输入是两份独立resident FP32 buffer，seed41/53的有界dyadic值；每次交替输入，防止只验证第一次绑定。

两种长度各12个block，ABBA平衡两arm顺序，各arm/block8次，主384次。`perf_counter_ns`覆盖chain调用＋原`MetalDevice.synchronize()`；输入构造、cold compile/load、完整readback和Python oracle检查在计时区外。每次都完整校验，不只抽样；相同payload按geometry/arm/seed保存8个文件，记录384次独立readback的hash，不能误写成384份独立输入。

| F32元素数 | eager中位完成时间 | JIT中位完成时间 | 12个paired-block倍率中位数 | paired范围 |
| --- | --- | --- | --- | --- |
| 4096 | 3.7696ms | 0.2061ms | 18.34× | 18.08–19.47× |
| 131072 | 3.7774ms | 0.2305ms | 16.37× | 16.13–17.99× |

两个顺序的中位倍率分别18.36/18.30×和16.48/16.37×；完整timers和block统计在证据JSON。384次、25952256个值位置/103809024个actual字节均与独立标准库oracle精确相同。pilot与profile另各16384值；不是额外模型质量集合。

这些倍率刻意包含eager逐次前端/提交开销。此例可融合或代数改写为单kernel，未与该强基线、GEMM、attention或完整模型比较；readback及其间CPU校验还会改变工作节奏。它不是steady-state模型吞吐、cold启动、能耗、所有tinygrad调用或GPU/ANE收益。优化GPU分母后必须重新估计ANE share，不能用这个eager例子的分母宣传异构提速。

## 重放输入合同：正确重放和失败都要测

1. warm/capture/replay：Python函数体只在前两次执行；主hot阶段每geometry只有96个eager调用增加body计数。JIT重放返回同一capture Tensor对象，旧引用不是独立快照；异步消费者需独立clone/slot与完成owner，本轮每次读完再下一次。
2. 普通Python scalar：capture时scale2，重放请求scale3，实际完整输出仍2x；两个geometry都与capture数学相符、与请求数学不符。上游footguns明确记录Python常量冻结；应使用明确runtime Tensor/Variable合同或重建JIT，不能把Python参数默认为动态输入。
3. 同一buffer作为两个输入：原`JitError("duplicate inputs to JIT")`拒绝，两geometry都记录；不是支持任意alias的证明。
4. shape/extent：原2D shape-change测试只读，不能授予所有bare1D输入extent资格。实际较小1D输入通过了原metadata匹配；不能把“有expected_input_info”当完整尺寸保护。

### 可疑extent：只保留安全preflight，不重复提交

探索main最初期待缩小输入会抛JitError，但4096→2048、131072→65536都被接受。该控制没有读回输出，可能让已捕获kernel访问逻辑范围外的数据；不能据此宣布数值/访存安全或driver故障。进程已显式等待并terminal，原日志/客户端/384计时单列保存，**不进入正式性能集合**。

干净main只调用原`_prepare_jit_inputs`比较metadata，实际结果仍匹配；own client在提交前拒绝这个重放，不再调用原JIT。源码会把input base替换为NOOP以构造签名，这提示需要进一步核对bare-buffer extent，但本轮不把可能原因写成完整修复。真正caller合同还需shape、dtype、nb/布局、容量、变量边界和当前generation；单加一个字节数比较也不能宣称通用动态shape支持。

安全修订没有改变主计时矩阵和原引擎；它只移除了可疑dispatch。严格main全部输出重新核验。我们自己的array构造器误用、Target对象JSON序列化错误、默认LLVM缺失，以及sysctl在sandbox拒绝后的获准只读重试，均保留为不同失败，不归咎GPU数值。

## 来源、复现与尚未取得的资格

19文件读本，固定archive237个Git blob全部核验；不是全文审计整个tinygrad。实际新runtime、JIT、HCQ2、renderer、ObjC owner、cache、测试footguns与PR分别记录来源身份。

- [原Metal runtime](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/runtime/ops_metal.py)、[HCQ2](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/runtime/support/hcq2.py)、[JIT](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/engine/jit.py)、[footguns](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/test/runtime/test_jit_footguns.py)。
- own pilot、干净main、profile控制、离线合同、capture。这些依赖固定源码时才运行设备，独立Wiki测试仅标准库。
- raw根 `outputs/gpu-ane-tinygrad-metal-round27-20261017`：source/API、旧客户端、七次guard完整stdout/stderr/receipt、primary/provisional实际数组、timers、profile pickle和task caches保留；所有35个preflight采样0%，是粗粒度总占用准入，不排除瞬时外部工作。

设备复测必须用新输出目录、`guarded_run.py`的五次低于5%准入和有界supervisor，不能只运行主脚本绕过guard。没有清全局cache、安装LLVM、强开M5路径或修改安全设置；原Metal源码仍调用未文档化MTLCompiler service，不能称完全public-only编译链。

本轮资格只到固定原始Metal/JIT组件、受控重放和局部同步计时。没有ANE/overlap、M5 NAX、symbolic launch、完整模型/LoRA、最佳GPU基线、能耗、内存安全、故障取消/drain或产品启用资格；后续应先补强baseline与完整输入合同，再进入真实消费者。
