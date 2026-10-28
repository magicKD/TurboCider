# Metal 4 profiling的生命周期与范围失效：创建成功后仍须验退出

<!-- knowledge
id: technique-metal-case-studies-metal4-feedback-queue-lifetime-and-range-invalidation
type: case-study
stage: pipeline
operator: layout, scheduling, sampling
backend: metal, cpu-reference
device: cpu, gpu
symptom: measurement, lifetime, launch-bound
semantics: bitwise, contract-only
representation: fp32
boundary: cpu-gpu, producer-consumer, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [实际粒度/时间域](metal4-counter-heaps-and-time-domains.md) · [官方来源](../../sources/official/metal4-counter-heaps.md) · [前置阶段来源](../sources.md#metal4-counter-boundary-sources-20261017) · 后续范围失败/正式路径

2026-10-17，第二十六轮。计数器值以前，还要验证API对象、scope和process生命周期。当前M4 Max/25G83/SDK26.5的原创客户端中，出现两个不同问题：自定义feedback descriptor退出故障，以及非零location的heap invalidation未按请求归零。它们都保留为失败，不从后续正例推断已修复。

## 1. 把原smoke的清理故障缩到不提交GPU

原counter smoke返回native SIGTRAP，parent监督进程exit1。PID/parent匹配的OS报告写有dispatch object over-release；故障frame为`_os_object_release → main`，并非由该栈证明某个shader、counter硬件或所有Metal4错误。原始stderr为空，不能据此猜测被清理故障遮住的先前返回原因。

两个完整4099+17 guard输出在故障前已写出，8232个自有合成float由独立标准库逐值/byte重放通过。但counter/feedback最终report没有保存；它们不计成功计数器窗口或正式计时。原v1源码、plan和binary身份归档，未覆盖失败。

随后使用仅公开对象操作的缩减，没有encoder、pipeline、dispatch、commit或sharedEvent提交。v2显式`objc_precise_lifetime`，7模式各两次，第二遍反序；17/29仅顺序标签，不是随机输入seed或统计任务。

| 模式 | 建Metal4 queue | descriptor持有的非空字段在退出前清空 | 两次结果 |
| --- | --- | --- | --- |
| default | 是，默认反馈queue | 未设置自定义字段 | exit0 |
| custom | 是 | 否 | SIGTRAP |
| custom-clear-descriptor | 是 | 是 | exit0 |
| descriptor-only | 否 | 否 | SIGSEGV |
| descriptor-only-clear | 否 | 是 | exit0 |
| copy-only | 否，另复制descriptor | 否 | SIGSEGV |
| copy-clear | 否，另复制descriptor | 原件/副本均清 | exit0 |

14单元8正常退出、2 SIGTRAP、4 SIGSEGV；三个先行探索单元另存，不混入14。每个单元都先打印对象创建状态，只有8个打印scope-cleanup-returned并最终exit0。**打印成功、对象非nil与进程正常退出不是一回事**。

descriptor-only和copy-only不建Metal4 queue仍失败，说明在该缩减中GPU提交、callback调用及shader并非必要条件；不是“GPU kernel崩了”。OS报告只接受owned PID匹配，公开只含signal、符号/映像名和hash；报告覆盖不完整，不补造缺失栈，不扫描/公开无关崩溃或地址。

## 2. assign字段不提供内部ownership证明

SDK将`feedbackQueue`声明为assign，公开ObjC runtime属性为`T@"NSObject<OS_dispatch_queue>",N,V_feedbackQueue`，无strong/copy/weak标记。两者是声明/元数据，不是私有setter、copy和dealloc实现的完整审计。precise-local-lifetime控制后仍失败，能否定“只要补该scope annotation就解决本fixture”的说法，不能据此确定所有retain/release根因。

默认nil队列、提前清字段是本fixture的正控，不是通用workaround、安全发行方式或Apple bug修复。原counter客户端同时改变了反馈queue与日志方式，因此不把它与原崩溃做纯计时A/B，也不声称只一个修改已经修复原程序所有问题。未提交issue/PR、未改变系统或上游库。

## 3. 默认queue让原范围检查失败正常显现

另一个保留的原协议改用默认feedback queue，在GPU准入确实恢复后新建进程。此次不是观察超时重启，也没有复用旧活进程：native明确exit17，未超时，父receipt为failed/exit1。

已完成第0个cell=1（小N、relaxed、单encoder）后，8项原值的前三项为正。CPU执行`invalidateCounterRange(NSMakeRange(1,1))`，再resolve全部8项，结果是slot0归零，slot1仍保留原正值；原guard要求slot1为0，所以拒绝。完整raw/partial两数组在保留，不用“API返回”冒充范围正确。

当前只有这个被保留的非零location GPU案例，不能扩大成任意range/所有Metal4版本、内部实现根因或全设备故障。共享event和feedback完成均已等待；这只是该显式协议的观察边界，不是所有竞争路径已审计。普通CPU测试验证“请求和观察不一致应拒绝”，不会重新触发设备故障。

## 4. 独立整heap路径，而非抹掉失败

后续新客户端/[plan](../sources.md#metal4-counter-timing-sources-20261017)只在GPU完成后用`NSMakeRange(0,8)`清整heap，前后8项均必须为0。原partial-range失败依然记录exit17；旧源码/plan/负控没有就地改成通过。

该整heap复用协议运行720个窗口，全部完整输出/guards通过，1680个marker的24MHz映射也通过；具体relaxed合并和precise成本见[计时页](metal4-counter-heaps-and-time-domains.md)。它只验证整heap路径，**没有**给非零location失效资格或修复声明；不将放弃一个坏API使用范围伪装成其原检查通过。

一次早期v2启动因driver利用率快照高被准入拒绝，未创建native-start或native目录；之后利用率读数恢复且重新通过准入才启动新ignored实验。快照不是持续负载、忙碌归属或全程独占证据；该拒绝也不是一个仍存活job的timeout。所有进程terminal状态分开保存。

前置保存早期object/ownership研究时尚无成功counter窗口的历史阶段；当前整heap结果在独立。两份不是同一协议的成功/失败混样。

## 5. 可离线学习与复查

labs/生命周期合同区分table快照、end、GPU完成、reset与begin后residency重声明；边界tests完整解码8232值和14个终态，正式tests另重放整heap路径。普通学习不导入MLX/Metal、不查系统日志、不运行native。

仅在显式原件/owned crash目录读取，其OS报告按PID/parent筛选并去掉地址/用户路径；核对全部原始journal、outputs和时钟，默认不会执行任何上游或设备。

复查条件：SDK声明/实际runtime、编译器/ARC策略、property/queue/descriptor copy与析构顺序、OS/SoC、heap range/entry ABI、完成合同或consumer改变。需要原引擎、长期服务或生产使用前，继续作独立生命周期、数值、性能和恢复验证；不是用一次默认queue正控授予所有Metal4 profiler安全性。
