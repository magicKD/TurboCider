# Draw Things：静态共享权重图、first-block gate 与多模态保护

<!-- knowledge
id: technique-algorithms-methods-draw-things-caching
type: method
stage: dit
operator: attention, sampling, layout
backend: cpu-reference, mlx, metal
device: cpu, gpu
symptom: stale-cache, lifetime
semantics: approximate
representation: bf16, fp32
boundary: producer-consumer, request
api: reference-only
-->

[算法入口](../README.md) · [TeaCache](teacache.md) · [DBCache](block-caches.md) · [实验](../../examples/README.md)

来源：第二轮固定源码/测试阅读 [S36](../sources.md#s36)；第十七轮进一步固定Draw Things、s4nnc、ccv [S53](../sources.md#s53)/[S54](../sources.md#s54)/[S55](../sources.md#s55)，实际编译原TeaCache控制器并运行NNC CPU/GPU接缝。原H3图、预训练权重、完整app/XCTest和媒体仍未执行；不引用未测速度。下文先讲设计，后列新增原生证据及限制。

## 类名不能替代算法边界

同一个TeaCache类包含不同模型策略。传统路径可以运行一个便宜inferModel产生调制信号，多项式累计相邻差；H3分支则接收**已经算完首块**的音频/视频信号，以各模态相对变化最大值对最近full anchor比较，并不使用同一多项式累计逻辑。

H3在hit时不推进signal anchor，避免每次很小的相邻变化掩盖累计漂移。若把不同长度模态拼成单一平均，庞大的稳定video可能掩盖小audio变化；max-of-modalities是一个保守保护。它不保证音画同步质量，但比混合平均更容易解释。

## full / tail / replay-head 三张图

H3模型可用 `startLayer`、`outputResidual`、`inputResidual`构造子图：首块输出当前hidden与首块residual；retained tail从后续global layer开始；replay-head把旧tail residual加到当前首块hidden后，仍执行当前norm/modulation和视频/音频投影。

权重按原global layer命名/mapper加载，静态辅助图通过parameters.share请求复用，而非显式再次加载整套参数；这是API层设计，实际物理backing/峰值驻留仍需测。不能把减掉prefix的子图重新从layer0编号，否则LoRA和权重映射可能串层。

cache的`marker`将独立轨迹的last signal、residual、skip计数分开；maxSkipSteps在实际replay调用处递增，而不在只查询gate时递增。调度计数必须按实际执行，而不是仅“gate说可以hit”。

## stream/host 同步是正确性的一部分

信号reduction的raw tensor同步toCPU用于gate；源码测试显式覆盖异步stream统计。统一内存不能省掉完成关系：如果host在reduction完成前读旧标量，控制器就会错误reuse。相应同步还必须计入整请求，不能称零开销。

同源码还按global layer选择FP16范围保护和Sol敏感层例外；这些是该模型/实现的校准策略，不是适用所有DiT的通用层号表。

## 这里的生命周期合同与本库并不相同

H3测试明确允许相同/递减step index在配置窗口内继续依据signal判断，并以marker、signal shape/数量、nonfinite与maxskip保护。TurboCider现有residual controller则对不连续step清状态。不能直接复制其中一个行为而宣称是所有TeaCache必须遵守的定义；应先定义sampler/多次评估的逻辑step合同。

ModelVersion枚举出现某模型，不证明该case的compile/replay路径已支持；某些case在compile分派仍会拒绝。能力查询应基于真实construct/compile/call闭环，而不是枚举列表。

## 可以迁移的设计技巧

显式model partition、global权重/LoRA身份、共享immutable参数、per-trajectory state、gate与cache提交分离、多模态最坏变化、完成后读统计。这些比“再加一个threshold”更接近可维护的Apple实现。

源代码测试涉及graph重建保权重、split/LoRA映射、input顺序/精度、async统计、多模态anchor与拒绝；第二轮只读，不当作通过。第十七轮仅执行下面明确列出的控制器/自有native图观察，不宣称原XCTest、原H3/LoRA图已全部通过。后续仍需same-source full模型与完整媒体复核。

## 第十七轮：原 Swift 控制器与真实 NNC，而非 tensor 接口桩

2026-10-17。固定 Draw Things `8d9c62ad38539ef8bb123f8b8e78cab76caafd0c`，其Package明确指向 s4nnc `f64e4a6ed2d656b5d169d39fc1316c8df793d09f` / ccv `66dc16a3a3b688adfa6dc1d19eeeb3071beeb701`。两依赖均以未修改的同ref源码构建；本地path override产生SwiftPM identity warning，保留而非消音。

准备器仅显式opt-in读取/核验Git对象，完整复制未改`TeaCache.swift`和选定原`ModelBuilderOrModel` enum到新的ignored任务目录；生成自有ModelVersion的36-case表面枚举与原生研究客户端。这个枚举只用于source switch编译，**不是36种模型的支持资格**。客户端使用真实NNC DynamicGraph/Tensor/Model/Dense，不提供伪toCPU/reduction/model实现。

模型为自有constant/affine heads及2×2 diagonal Dense，不是MiniMaxH3模型。没有加载真实权重或activation。cache/head/gate方法实际执行；`TeaCache.compile`被Swift编译但未运行其完整模型分派，不能把通用微型Model.compile成功升级为原H3三图或LoRA编译成功。

实际wrapper仅运行`.model`分支；ModelBuilder的Bool重建/完整模型装载与取消等路径仍未执行。source consumer阅读范围与native执行范围分别限定，不互相转授资格。

启动器绑定binary SHA与准备源文件；CPU默认，GPU须显式开关并重新采集3个Device≤5%样本，只记录`pid,pcpu,comm`派生的无argv进程信息。正式GPU窗口为`[0,0,0]`，完成后0；这不证明连续独占。新scratch正式构建成功；复用其它package root的旧scratch曾未编译新源文件，故未用该旧binary作为正式数据。源码、准备器、客户端、启动器、原始native JSON文本和owned-child终态分别绑定。

### 哪些行为已实际验证

分别含39个CPU / 40个GPU原生观察，同一binary、同一固定源码：

| 接缝 | 每设备观察 | 本轮证明的边界 |
|---|---:|---|
| H3多模态gate | 14 | audio/video分别mean-L1取max、对full anchor、严格阈值、marker隔离、坏shape/NaN/window拒绝；重复/递减step不自动失效 |
| 传统多项式gate | 12 | 相邻anchor与累计预算；零/非有限输入的NaN保留 |
| Wan/FLUX/带reference FLUX/H3 head | 4 | cold返回nil、真实参数输入位置与FP32大值保留 |
| 只查询gate | 4 | `[false,true,true,true]`，不消耗实际replay计数 |
| maxSkipSteps=0 | 4 | clamp到1；查询→replay→查询后为`[false,true,false]` |
| native Dense参数共享 | 1 | 只初次share后修改full权重，reduced结果从`[2,4]`变`[3,6]` |
| 显式GPU stream统计 | 仅GPU 1 | fill1→fill1.25后source gate为`[false,false,true]`，不是读到旧统计 |

39/40是功能观察条数，不是GEMM/shader/NFE或真实模型请求数。GPU tensor/mean路径用固定ccv的MPSGraph实现，CPU mean是其FP32 reference loop；没有做物理ISA/dispatch/ANE归因或训练验证。

表中14/12是两条主序列的query行数，budget和async分组内部另有query；机器计数字段也限定为`primary_*_query_rows`，不把它们冒充全部gate调用次数。

标准库整理器核对全部gate、完整private-anchor快照、head值与共享更新，独立标量oracle，不运行Swift、上游或GPU。source anchor通过只读Mirror检查，未修改private state；只读reflection/额外readback不是原应用工作量，因此本轮明确`performance_measured=false`。

### 新的精度/状态证据，不是泛化画质结论

缓存类的泛型为Float16，但明确传入的signal/head tensor为FP32；70000没有先缩到FP16。Wan/FLUX得到35003；H3两个不同head得到35005、17504。它验证这个typed caller边界，不是FP16/BF16全模型等价，也不证明所有其它wrapper都不cast。

H3以`max(mean(abs(delta))/max(mean(abs(anchor)),1e-6))`保护小模态，并检查统计是否finite。传统路径则把各tensor的mean相加再作ratio，始终更新相邻anchor；没有相同的shape/count/finite守卫。实验的线性外观系数`(0,0,0,1,0)`在零分母时仍算所有多项式项，`0*Inf`会使dist为NaN；与threshold作`>=`比较为false，query可以持续返回true。非有限初始anchor也可污染累计预算。**不能在oracle里把多项式先化简成r，再错误宣称已fail-open**。

这是明确的零/非有限人工输入，不证明正常Wan/FLUX调制信号会如此，也不是实际媒体失败。gate返回true且缺residual时，真实consumer仍可能因`callAsFunction`返回nil而走full；query资格与成功replay要分开。maxSkipSteps只在成功进入reduced call时递增，查询次数不等于省算次数。

`parameters.share`在此真实native路径有更新传播正控，但没有测物理pointer/backing/峰值，也不证明跨全部模型/LoRA/graph重建的参数映射正确。Model IO节点和参数共享不是同一件事：本客户端需要为不同Model新建Input，再按原API共享命名权重。

### 保留的构建/原生失败

首次SwiftPM在外层沙箱被拒绝执行manifest，获准后构建；不通过关闭上游断言制造成功。原测试样式`Model([input],[input])`触发当前ccv的非Input-output断言；复用两个Model的Input节点又触发input-bitmask断言。三个SIGABRT的终态、日志和当时客户端源文本均保留在公开证据history；前两个binary仅保留本地hash/文件，不分发。

修正仅限自有客户端：未执行的gate载体换为合法affine op，各Model用独立Input。原TeaCache、s4nnc和ccv均未改。这个结果是固定SwiftPM构建/fixture的前提，不是已确认商店app或所有Bazel配置故障；原仓库XCTest在不同构建模式下是否通过仍需单独运行。Package缺失exclude路径与local-identity警告同样保留。

### 还缺什么，怎样继续测成本

本轮跨过“只看Apple代码”的边界，但仍没有量化gate成本或跑完整H3。H3每步仍先算首块，source gate每模态至少有numerator/denominator reduction与host读统计；hit还要跑当前head。下一阶段应把首块、gate同步、tail refresh、cached head、CFG及完整请求拆开，同时保留strong full baseline，不按跳过层数直接计算倍数。

可以研究只在full更新anchor norm以减少分母重算，但必须保证anchor不可变/版本明确；若调用者原位修改借用tensor，缓存host denominator会改变策略。第十七轮未实现/测速；[第十八轮](../case-studies/nnc-gate-controller-cost.md)已测gate-only刷新/查询窗口，保留一次query亏损、相对批量读回强对照无稳定通用收益、真实mutable-anchor错误hit和代际保护。不包含首块/tail/head或原模型速度。MPS/MLX、CPU和ANE/CoreML各有不同交接成本，不能转授资格。

另一个[Draw Things→ccv异构导读](../../gpu_ane/case-studies/drawthings-to-ccv-hybrid-route.md)固定的是不同、更后的公共refs，且只做源码/selector研究；不能把那里的ANE/GPU选路与本轮旧ref gate功能数据拼成已验证异构模型。

### 重放

构建会执行依赖Package/编译工具，并有网络/SDK需求；默认检索和测试不会执行它们。没有完整编译闭包/发布binary attestation：只固定所列阅读文件、依赖refs/manifest及实际binary/客户端，不能把Git status clean当作对全部传递依赖的完整内容审计。
