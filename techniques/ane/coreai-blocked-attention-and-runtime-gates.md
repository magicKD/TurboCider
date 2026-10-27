# Core AI：分块attention、数值边界与运行时门槛

<!-- knowledge
id: technique-ane-coreai-blocked-attention-and-runtime-gates
type: method
stage: prefill, decode
operator: attention, gemm, layout
backend: coreai, cpu-reference
device: cpu, ane
symptom: precision-loss, measurement
semantics: contract-only
representation: fp16, fp32
boundary: producer-consumer
api: public, reference-only
-->

[ANE目录](README.md) · [源码阅读路线](../references/coreai.md) · [固定来源范围](sources.md#coreai-attention-sources-20261017) · [验证记录（过程原件已移出）](../MAINTENANCE.md)

2026-10-17，固定`apple/coreai-models@3b25d934ccaee2d05edc6e281c555a5a9a13a55c`。本批读取iOS/ANE导向的attention、Swift引擎/profiler入口，并在现有CPU Torch执行**整个未修改的原始iOS SDPA模块**。独立参考覆盖attention→非对称稠密投影→residual；没有Core AI运行时、导出、stock模型、物理ANE或速度资格，也未运行整套上游测试。

## 1. 平台门槛：Core AI不等于Core ML

固定[README](https://github.com/apple/coreai-models/blob/3b25d934ccaee2d05edc6e281c555a5a9a13a55c/README.md)要求macOS/iOS27及Xcode27，`Package.swift`的平台声明也是27。本机M4 Max为macOS26.6.2/build25G83、SDK26.5，SDK没有CoreAI.framework。四个新进程都实际编译执行自有Swift gate，`canImport(CoreAI)=false`，未执行推理；这是当前编译环境的结果，不是所有Mac或未来版本的结论。

| 路径 | 本批实际工作 | 不代表 |
| --- | --- | --- |
| Core AI Swift / `.aimodel` | 固定源码阅读、编译可导入性gate | 模型已prepare/运行/放置ANE |
| iOS BC1S primitives | 原SDPA/BlockedSDPA在CPU Torch执行 | 导出后Core AI或iOS/ANE数值通过 |
| macOS composite SDPA | 读取`coreai_torch.composite_ops.SDPA`委托 | 执行了macOS实现或fused GPU算子 |
| [既有Core ML实验](public-coreml-deployment.md) | 另一框架、另一批结果 | 代替Core AI缺失 |

只复用Python3.13.12 / Torch2.14.1（git`5c4886908584029761b579af026dcfb627c84070`）/ NumPy2.5.4，CPU一线程、无grad。项目Torch2.9.0 pin**不满足**；所选独立模块只依赖Torch，未导入`coreai`、`coreai_torch`或`coreai_models`，不等于满足整个工程依赖。没有安装、系统/SDK升级、package setup或模型下载。

## 2. 优化机制：布局、GQA复用与compiler表达

[原模块，完整256行](https://github.com/apple/coreai-models/blob/3b25d934ccaee2d05edc6e281c555a5a9a13a55c/python/src/coreai_models/primitives/ios/sdpa.py)接受`Q=[1,Hq·D,1,L]`、`K/V=[1,Hkv·D,1,N]`、mask`[1,N,1,L]`。本批batch1、合法整除GQA与无效caller负例分开；scale先乘K再做QK，实际buffer值另行记录，不把默认scale当模型scale。

flat逐Q head算QK、拼完整score、加mask并沿key轴softmax，再逐head算PV。blocked按KV head把共享它的`G=Hq/Hkv`个Q head折成`G·L`行，复用每个K/V block，避免复制KV到Hq份。每块只构造`G·L×B`score；super-chunk限制被compiler提升的transpose轴。这是源图materialization和复用机会，不是本批测得的峰值内存/带宽收益。

running maximum为m，缩放后的累计质量为den，**已归一化**输出为o，原表达的意图是：

```text
m'   = max(m, max(s))
corr = exp(m-m')
p    = exp(s-m') / sqrt(B)
prev = den*corr
den' = prev + sum(p)
o'   = (o*prev + p@V) / den'
```

统一`1/sqrt(B)`在理想最终比值中约去，降低累计量级，但不是任意N安全的证明：最坏den仍随`N/sqrt(B)`增长，QK、V与输出范围也需约束。上游的fp16/transpose/更快kernel注释是实现经验，不是本批验证的硬件规格。

N≤65536用`p@Vᵀ`，N>65536改成`(V@pᵀ)ᵀ`，转置较小的概率矩阵。用D16、G·L4区分实际Python操作数：65536为`[1,4,8192]@[1,8192,16]`；65537为`[1,16,8192]@[1,8192,4]`，末尾宽1也实际执行。16份观察行核对形状，最终输出全过原gate。**65536只是固定源码分支阈值，不是本机测得的ANE transpose极限。**

## 3. 完整实验：正常域、未定义参考和无效caller

两seed17/29×正反顺序独立进程×FP32/FP16×flat/blocked×observed/plain，共288次原forward尝试。正常shape返回后还执行完整稠密非对称投影及residual，不是identity或finite-only。全部输入字节保存在本地；公开配方可逐字节重建长K/V和mask，原调用前及全部调用后均核完整hash。

独立参考用Python双精度和`math.fsum`，K/V的31周期只用于**精确multiplicity分组**，不是抽样；独立逐key小例另验分组。原运行前冻结的逐元素gate不变：FP32 `1e-5+1e-4·abs(ref)`，FP16 `.002+.01·abs(ref)`；Y和完整投影/residual Z分别验收。

| fixture/契约 | flat | blocked | 解释 |
| --- | --- | --- | --- |
| causal start/extend | 通过 | 通过 | key0有效的普通global causal |
| N65536/65537与ragged tail | 通过 | 通过 | CPU源分支，不是设备阈值 |
| 前块全`-inf`，后块有合法window keys | 通过 | Y/Z全NaN | 有定义参考的primitive边界 |
| 同window但屏蔽值改`-10000` | 通过 | 通过 | 不同输入契约，不能称原mask已修复 |
| 有限QK=-51200、global causal mask | 通过 | Y/Z全NaN | 极端合成范围，不是预训练故障 |
| 整行所有key被`-inf`屏蔽 | NaN | NaN | 数学参考未定义，不能当零或普通误差 |
| Hq3/Hkv2 | IndexError | 只返回2个head | 无效caller，拒绝部分head消费者 |

528份Y/Z共29,952元素：384份有限且过gate；64份有定义参考的非有限拒绝；64份全mask未定义；16份无效GQA错误shape。另16次invalid GQA flat异常没有数组，不补零/计成功。共256次完整消费者。264对observed/plain和264对正反序数组逐字节一致，含NaN原字节；只覆盖本批CPU输入/观察器，不是普遍透明性证明。

## 4. 两个NaN机制：有限initializer不等于空状态

### 空前缀block污染后续合法结果

原blocked初始化`m=-40000`、`den=o=0`。首块mask全`-inf`时，m仍-40000，p、prev和新den均0，第一次更新就是`0/0`。实际小trace完整保存numerator/denominator/result，确为零/零→NaN。后面合法keys已使den正且有限，但`NaN*0`仍NaN，完整dense/residual继续传播。

不能只把initializer改`-inf`就称修复：空块还会遇到`-inf-(-inf)`。需要明确空质量状态转移、合法key合同及导出后语义；本批未修改上游或测试新修复，建议不等于已通过实现。

### 合法global causal也可低于有限哨兵

Q=160、K=-160、D4、实际scale=.5，原代码先缩K，CPU matmul在FP16/FP32都返回**有限-51200**，不是先溢出成`-inf`。running max仍-40000，`exp(-11200)`下溢0，所有块den仍0，结果NaN；flat围绕真实最大值归一化，Y/Z通过。全部96份小QK见证核对此有限值，原blocked首除法零den也保存。

输入满足普通global causal和整除GQA，但刻意构造极端范围；未经过Gemma4的Q/K norm、RoPE或真实activation分布，不能证明stock模型可达。

### 沿真实调用者限定结论

[Gemma4 Attention](https://github.com/apple/coreai-models/blob/3b25d934ccaee2d05edc6e281c555a5a9a13a55c/python/src/coreai_models/models/ios/gemma4_text.py#L139)明确把**global交BlockedSDPA、sliding交flat SDPA**，另有分离global flat/sliding ring cache和shared只读slot。因此window-prefix是primitive泛化/复用边界，**不是已复现Gemma4默认sliding故障**。所读上游flash-attention测试涵盖普通causal、ragged、GQA/MQA/MHA及65536/65537，正例都有最初合法key；未执行其测试/export，也未审计所有模型caller。

## 5. Swift引擎与profiling：对象和时间分别取证

后续已有[完整profiler/背压组件执行研究](coreai-profiling-reset-and-backpressure.md)，实际验证pending-write/reset与作用域/取消边界；采用明确的依赖适配器，不是完整Core AI引擎。本节保留原attention批次只读的历史范围，不回填运行资格。

[EngineFactory](https://github.com/apple/coreai-models/blob/3b25d934ccaee2d05edc6e281c555a5a9a13a55c/swift/Sources/CoreAILanguageModels/InferenceEngines/EngineFactory.swift)先prepare，再按structure选引擎：chunkedStatic默认staticShape，dynamic默认pipelined，不相容override明确拒绝。pipelined日志标记GPU。框架名、iOS目录、static shape或engine label不能单独证明物理ANE放置。

所读[静态引擎1–115行](https://github.com/apple/coreai-models/blob/3b25d934ccaee2d05edc6e281c555a5a9a13a55c/swift/Sources/CoreAILanguageModels/InferenceEngines/CoreAIStaticShapeEngine.swift#L11)展示`extend_<ctx>_<qlen>` / `prompt_opt_<ctx>_<qlen>`函数ladder、persistent states和预分配输入buffer。可对照[Core ML shape机制](dynamic-shapes-hints-and-failure-boundaries.md)，但不表示API兼容或整套调度/取消协议已验证。

[Profiler所读1–190行](https://github.com/apple/coreai-models/blob/3b25d934ccaee2d05edc6e281c555a5a9a13a55c/swift/Sources/CoreAILanguageModels/Profiling/InstrumentsProfiler.swift#L101)结合signpost与host Mach时钟，按timebase换算ns；`end()`显式结束后通过Task异步写StatsStorage。析构未显式end会关闭signpost并告警，但**不记录stat**。trace结束标记不等于summary已有同一笔记录，还需检查异步记录完成/覆盖；它不是ANE counter、每op设备时间或重叠证据。

完整`PerformanceMetrics.swift`按category统计，缺失category有0默认值，不应据此把缺测解释为零开销。extend注释是N-1 inter-token spans，decode/inference是N spans；未读完整strategy调用链、执行profiler或确认实际分母，不能仅据注释宣布off-by-one bug。后续同时保存span ID、explicit-end/记录完成、token事件、起止边界，区分TTFT、inter-token、logits、host准备和request总时间。

完整[BackpressuredStream](https://github.com/apple/coreai-models/blob/3b25d934ccaee2d05edc6e281c555a5a9a13a55c/swift/Sources/CoreAIShared/Runtime/BackpressuredStream.swift)用permit限制未pull元素，终止cancel producer并drain等待continuation。permit在pull后即归还，不代表消费者处理完或设备drain；yield调用前已构造的element及消费端持有元素也不等于被此计数覆盖。仅源码阅读，未执行慢consumer/取消测试，不称完整视频流水或总resident memory上界已验收。

## 6. 可迁移设计与复查条件

值得吸收的是BC1S/Conv投影、按KV分组GQA复用、受控transpose、静态函数菜单、完整消费者验证和显式profiling生命周期。须按shape、mask空行定义、数值范围、compiler、模型阶段重测；不当“所有ANE一律FP16/4bit必有固定PSNR”的定律。上游`model-authoring/SKILL.md`仅作经验来源，未安装/调用插件，其中典型压缩PSNR不是当前模型质量保证。

平台27可用后，应按原module/原mask导出正常及负域，再用真实Q/K分布和完整caller判断可达性，保存placement/分区、state binding、完整request质量/时延/峰值内存和profiling缺失状态。没有这些证据，不发布flat/blocked的ANE速度排名或由CPU总时长生成提速比。

## 实现参考与实验

独立checks与冻结配方/数学默认重放全部输出和小intermediate packet，并实际拒绝错误seed消费者参考、插入NaN两个负控。长intermediate只有shape/hash/分类，**不声称完整长中间值数值重放**；长最终输出及每字节输入则完整校验。

显式collector必须`--run-original`、固定只读checkout及已有选定Torch，仅写新owned目录。重核四份source/runner/math/gate、选定Torch Python/动态库、288个实际输入文件、终态receipt及全部数组；原上游body、binary和依赖不随Wiki分发。早期自有collector缺digest helper，在Swift/Torch/原forward前失败，失败码1与零调用保留，不混入主样本或归罪上游。

仅在已有上述精确依赖时显式重跑一个CPU进程；变量须由调用者设为现有解释器和固定checkout，不隐式创建环境：

完整curator需要四个seed/order目录及保留的失败pilot原件；单次重跑不假称复现了全部矩阵。默认CPU case仍仅需随Wiki分发的证据和标准库。

源码、Torch/NumPy、OS/SDK、dtype、GQA/head_dim、block/context、score范围、mask定义、导出算子或消费者变化均须复查。27平台可用不自动继承本批数值或ANE资格。研究继续。
