# GPU-resident完整SwiGLU：公共Core ML布局、双GPU基线与先验数值门槛

<!-- knowledge
id: technique-ane-resident-ffn-layout-and-gpu-baselines
type: method
stage: ffn
operator: gemm, swiglu, layout, scheduling
backend: coreml, metal, mlx, cpu-reference
device: cpu, gpu, ane
symptom: compile-bound, wrong-layout, precision-loss, measurement
semantics: numerically-equivalent, contract-only
representation: fp16, fp32, iosurface
boundary: producer-consumer, cpu-gpu
api: public, reference-only
-->

[ANE目录](README.md) · [固定来源](sources.md#resident-ffn-sources-20261016) · 全部样本/独立审计 · [此前pixel接缝](public-pixel-buffer-gpu-handoff.md)

2026-10-16，M4 Max / macOS26.6.2、SDK26.5、clang21、coremltools8.3.0、MLX0.32.0、NumPy2.0.2。本批从正确交接进一步测**完整dense paired SwiGLU→FP32 GPU消费者**，不把省copy直接当加速。

2026-10-17继续做结构/隔离验证；下述设备样本仍保留10月16日批次身份，不称10月17日重新执行的设备结果。

后续10月17日的[精度岛/观测图研究](precision-islands-and-observer-effects.md)另保存host诊断与新native重测：activation32 half-IO修复此large组件的数值，却比固定GPU基线更慢。它不覆盖本篇原half图、92份数值失败或原日期身份，也不提供物理累加器位宽证明。

四个正式进程，两shape×两seed；460份完整输出含400份正式timing output。独立CPU FP64复核全部64,061,440个Y元素及同数Z元素；92份不通过原先门槛的输出仍保留，**不能称460份全部数值合格**。没有原项目/真实checkpoint、完整block/request、物理ANE或ISA精度资格。

## 可复用的方法：按 GPU 常驻消费者选择 FFN 后端

1. 固定同一 FFN 的权重、非线性、残差和精度目标；同时建立可用的强 GPU 实现对照。
2. 从 ready GPU X 计到 ready GPU Z，把 pack、预测、restore 和同步留在分母，不能只比较 CPU-fed predict。
3. 先按完整输出拒绝不合格候选，再比较 warm 成本；精度岛或布局变化必须独立消融。

下文提供五臂设计与负结果，值得借鉴的是对照方法，不是“Core ML 一定胜过或输给 GPU”。

## 1. 相同函数、相同GPU输入与两个GPU库基线

固定FP16 source weights、逐generation变化的FP16 X：

```text
[G,U] = X · Wgu       # gate/up先配对concat成一个投影
H = SiLU(G) ⊙ U
Y = H · Wd
Z = 3·FP32(Y) − 0.5   # 实际GPU消费者
```

本例无bias/residual/LoRA/quantization，使用确定性的dense hash混合权重和输入，不是identity、零矩阵或低秩解析捷径。它是完整合成FFN组件，不是实际模型任务质量验证。

| shape | M,K,F,N | 输入/权重语义 |
| --- | --- | --- |
| small | 64,256,512,256 | 同一X[M,K]、Wgu[K,2F]、Wd[F,N] |
| large | 512,512,1024,512 | 相同公式，独立seed17/29 |

五臂都在同一个原创native进程中调用，无Python/ctypes跨臂边界差异：

| arm | 路线与计入的工作 |
| --- | --- |
| MLX compile API | 显式GPU及enable_compile；原生C++编译函数，paired matmul→gating→down→FP32 Z，eval并同步 |
| MPS | 两个原生MPSMatrixMultiplication，中间Metal gating，再Metal Z；一条owned command buffer完成 |
| matmul_ne | 同一frozen数学函数，row-major MLProgram，CPU_AND_NE，PB-backed IO，输出实际采用检查/必要时完整copy fallback，再GPU Z |
| conv_ne | 等价BC1S/1×1 Conv图，CPU_AND_NE；**GPU pack、完成等待、prediction、转置读取/FP32消费者**均计入 |
| conv_cpu | 同一compiled Conv资产的CPU_ONLY加载控制；仍包含相同GPU pack/consumer，并非整臂没有GPU工作 |

MPS和MLX都是库级优化路线，不是手写naive GEMM；但不宣称是所有Metal/MPSGraph/MFA/NAX实现中全局最优。MPS复用输出/intermediate，MLX保留自身分配与compile API行为；不把compile函数名当作固定kernel/dispatch数。

## 2. 边界是ready GPU X→ready GPU Z，不是predict时间

原生Metal producer先把同一owned GPU region写成当前generation并完成。三类frontend的输入wrapper、权重与对象都预建/准备；MLX通过严格buffer adoption持有整个page region，再建正确logical view，实际buffer/设备身份检查通过。

这里MLX低层buffer/Metal对象解释绑定原0.32.0 headers和安装库，不是Apple保证的通用ABI；移植到别的MLX版本/后端须重新验身份和owner，不能只因指针非空就直接采用。

timer开始后执行本arm的必要准备、调用、提交、完成及消费者；结束时Z已在GPU端完整就绪。Conv的layout税不被移到timer外，Core ML返回backing没有被采用时也不能偷偷消费旧scratch：client有显式实际返回copy路径，若发生会计时并标记。实际正式数据中276次Core ML返回全部采用原backing，未走该copy。

所有Y/Z原始snapshot、JSON/file IO、GPU diagnostic Y gather及CPU参考均在timer外；数值reference在整个native timing子进程退出后才计算。诊断仍可能影响相邻样本cache/host状态，未取得独占机器/频率控制，不能当纯kernel时间。

每进程first1、warmup2单列，再20个同输入五臂trial。轮转并反向顺序，每臂在每个位置恰好出现4次。不同seed不拼成一对；按每trial的固定GPU分母/candidate比值取中位，不用独立median之商，也不选每trial最小GPU时间组成不可部署的oracle。

首次预测前已导出、compile、查询compute plan并load，因此first不是真正cold。export/compile/load原始时间单列在JSON；权重驻留、prepare成本与一次性任务收益不能由warm表替代。

## 3. 实测wall：先看数值资格，再看时间

单位ms，每格为seed17 / seed29的正式中位。†仅为完整保存的诊断时间，**未通过数值门槛，不参与合格提速比较**。

| 路线 | small | large | 数值资格 |
| --- | ---: | ---: | --- |
| 原生MLX compile API | .2053 / .2041 | .4804 / .4858 | 两shape均通过 |
| 原生MPS paired FFN | .1835 / .1838 | .5264 / .5338 | 两shape均通过 |
| Core ML row-major，CPU+NE许可 | .1539 / .1575 | .4571† / .5703† | small通过；large拒绝 |
| Core ML Conv，CPU+NE许可 | .2559 / .2633 | .5408† / .6377† | small通过；large拒绝 |
| Core ML Conv，CPU_ONLY控制 | .2612 / .2673 | .6748 / .7092 | 两shape均通过 |

small的固定MPS基线比MLX快；row-major Core ML相对MPS的配对中位约**1.203× / 1.181×**，相对MLX约1.313×/1.311×。但其compute plan首选CPU，因此不能标题化为“ANE快20%”。Conv完整路径比两条GPU基线都慢。

large中MLX比MPS快，CPU_ONLY Conv也更慢。CPU+NE row-major有一个seed的诊断时间略低于MLX，另一个反而更高；两seed所有该路径输出都失败，**不发布其合格速度比**。未按更快seed、成功子集或每trial最佳arm选赢家。

这个反例说明：小任务的公共CPU路径可能在GPU-origin边界获益；Conv重写不自动更优；看似有价值的加速也可能先被数值预算否决。以上只属于这组resident合成组件，不能用于完整模型/媒体延迟或跨机器排名。

## 4. 原定门槛没有因为计划首选ANE而放宽

导出/首个pilot前就固定Y的relativeL2≤.015、maxabs≤.02，另要求完整Z精确等于`3·FP32(actual Y)−.5`。ideal reference使用源FP16 X/weights转CPU FP64，完整投影与非线性；不是拿MLX/MPS输出当Core ML oracle，也不是要求各有限精度实现bitwise相等。

| 数值观察（所有first/warm/timed） | 最大relativeL2 |
| --- | ---: |
| MLX / MPS，两shape | 约.00058 / .000541 |
| Core ML small，三个加载/布局arm | 约.00403 |
| Core ML large，CPU_ONLY Conv | 约.00595 |
| Core ML large，CPU+NE两图 | 约.02195–.02200，超过.015 |

large的两个CPU+NE arm各23输出×两seed，共92个拒绝，正式timed中80个。其maxabs仍低于.02，但两条门槛必须同时成立，不能只挑较好指标。summary在这些arm上将对两条固定GPU基线的ratio/median置null；原始wall仍可追溯，不包装成可采用增益。

两种Core ML布局每个trial的完整Y hash相同，共92对；small计划相同、large请求策略也相同。这不证明compiler内部具体采用哪一个kernel，更不证明所有Linear永远等价于Conv。CPU_ONLY large结果精度更高也只说明此处执行策略/路径影响数值；**不能据此认定ANE累加器位宽、内部量化或INT8 MAC**。

1.5%只是本次预声明组件gate，不是模型任务能接受1.5%/2.2%误差的通用规格。真实activation、outlier、bias、LoRA、精度岛及完整生成质量都需另验；并非宣告所有ANE计算错误。

## 5. compute plan怎样帮助解释，但不冒充测量

small两图非const op首选CPU，supported还列ANE；large两图非const op首选ANE；CPU_ONLY控制均为CPU。const的usage/cost缺失保持unknown。所有plan条目的source_kind都是predicted，physical_operation_timing=null。

固定[Apple ANE Transformers FFN](https://github.com/apple-aiml-research/ml-ane-transformers/blob/590eb35913622cdef217ebc5329047aab389ac98/ane_transformers/reference/ffn.py#L12)展示1×1 Conv投影及channel-first设计，但其原模型是ReLU/可选residual/norm，本批没有执行该PyTorch模块或继承2022倍率，只吸收布局问题。

本次row-major matmul large本身也在计划中首选ANE，不应照搬“必须先改Conv才能用ANE”的绝对说法。Conv与matmul之差包含不同图/runtime和pack/完成边界；没有分段消融，不能把全部差值归因于transpose。

[MLX v0.32.0 compiler](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/compile.cpp#L1100)缓存trace/tape后用当前inputs替换placeholder；新的函数调用不等于复用已求值旧output。这里预建input view却每次重新调用compile函数，完整generation检查通过。外部producer前后均按owned完成边界同步，不依赖新array名字来证明更新。

## 6. 两个pilot与独立参考的修正必须保留

首个small pilot只有2个正式trial，MLX每次还在window内make_buffer/view，而MPS/Core ML对象已预建，不作为正式速度证据。它的NumPy `@` FP64参考出现BLAS浮点状态warning，尽管数组finite且选定full-K/F scalar fsum匹配。原因未完全归因，不静默ignore。

随后改用unfused CPU FP64 `einsum(optimize=False)`并把warning作为error。large旧版native完整收集115份输出，但原driver在首个超门槛输出即停止oracle，因此wrapper为failed；没有覆盖这个失败或扩大gate。新read-only replay在另目录检查所有旧输出，旧small/large合计140份仍排除正式速度分析。

正式版预建MLX logical input view到window外，所有frontend共享ready同一X；函数每次fresh调用。oracle继续完整收集每份数值与pass/fail，而不是遇到首次拒绝就丢掉其余case。`协议完成`与`候选合格`分别记录；large exit0不等于其ANE候选通过。

第二套不导入原driver，不加载Core ML/MLX或GPU，分别计算gate和up再做down，对每个原始Y/Z全量复核；与原concat参考的metric最大差≤约3.34e-16。原driver另对两代、两个row、三个output column做完整K/F的math.fsum核对。FP64参考hash可因求和顺序不同而不同，不强求bitwise一致。

## 7. 可复现性、核心CPU范围与后续门槛

原生client、支持shader与显式driver为原创。driver拒绝旧output，保留2GiB reserve，使用已安装固定库及headers；不build上游、下载模型、安装依赖或改产品默认。数据包括实际库/header/metallib SHA，但不冒充loaded-image/source attestation。

标准库合同/统计检查和11个portable tests验证完整protocol、顺序、metadata绑定、数值拒绝与ratio屏蔽、plan标签和tiny有限错误。**它们不在没有raw文件/NumPy时假装重算64M个opaque设备Y值**；完整dense复核明确需要ignored原始数组及已有NumPy2.0.2。

`--audit-existing OWNED_RUN --output-dir NEW_AUDIT`只做CPU完整参考回放，不执行设备；不能覆盖原报告。原始weights/arrays/models/binary留本地；本批约640MiB，未删除旧实验。

尚未完成：实际checkpoint/activation和任务质量、bias/LoRA/量化、不同M/头尾shape、同预算memory/能耗、冷启动、硬件trace与原应用完整请求。重开条件：source/OS/SDK/MLX/CoreML或dtype、输入域/质量gate、shape/布局、weights、owner/stream/准备与完成边界改变；不能继承旧倍率或失败结论。
