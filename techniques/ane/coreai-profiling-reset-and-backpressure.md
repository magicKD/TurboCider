# Core AI运行时支撑：异步统计、reset与背压所有权

<!-- knowledge
id: technique-ane-coreai-profiling-reset-and-backpressure
type: method
stage: decode, pipeline
operator: scheduling, layout
backend: coreai, cpu-reference
device: cpu, ane
symptom: measurement, lifetime
semantics: contract-only
boundary: producer-consumer, request
api: public, reference-only
-->

[ANE目录](README.md) · [前轮attention/平台门槛](coreai-blocked-attention-and-runtime-gates.md) · [完整来源](sources.md#coreai-runtime-sources-20261017) · 原生事件与完整数组 · [验证（过程原件已移出）](../MAINTENANCE.md)

2026-10-17，继续固定`apple/coreai-models@3b25d934ccaee2d05edc6e281c555a5a9a13a55c`。本批把**完整原始**`BackpressuredStream.swift`、`InstrumentsProfiler.swift`（878行）、`PerformanceMetrics.swift`直接编译执行，研究AI流水周围的host控制与测量正确性，不是Core AI模型、ANE/GPU计算或性能benchmark。

macOS26.6.2/build25G83、SDK26.5、Swift6.3.3。三个文件本身不导入CoreAI.framework，故能独立研究；整个项目的平台27要求和前轮真实canImport=false不因此改变。没有运行Swift package、安装插件/依赖、升级SDK、下载模型或改动上游。

## 可复用的方法：先明确统计和队列的完成边界

1. 将 span 结束、异步统计写入、reset 分开；reset 不应被假设为等待已有写入完成的屏障。
2. 保留 absent 与实测 zero、每项单位和 token 分母；host span 不包装成设备忙碌时间。
3. 背压同时核对 queued、producer 已构造和 consumer 持有对象；取消后继续验收任务退出及最后 owner 释放。

这些方法适用于流水线的内存和测量设计；下文是原 Swift 组件案例，不是 Core AI 模型或 ANE 性能复现。

## 1. 实验包络：原实现不动，依赖替代必须明示

原profiler和metrics导入`CoreAIShared`。本批没有拿到原Logger/Duration完整package；明确使用自有最小适配器：关闭CLILogger分支，并提供Duration到秒的转换。原StatsStorage、ProfileSpan、PerformanceMetrics和背压逻辑**未改写、未摘取替代函数**，但这仍是组件组装实验，不是完整原包验收。

四次独立编译/采集，seed17/29与正反case顺序，每次七个新原生进程：4个stats、24个stream，共28进程/816事件，全部正常退出。早期六case成功pilot另保留，不混入28分母。原件有源码、adapter/客户端、binary、module、stdout/stderr与receipt身份；`otool`显示adapter实际install-name为相对的`libCoreAIShared.dylib`，runner明确把cwd设为owned构建目录。它不是原完整CoreAIShared，也不是所有dyld已加载镜像/系统运行库的闭包证明。

Journal逐事件同步落盘，有序号和控制gate；这会扰动调度和时间。只验证必要的先后关系、状态、完整消费者，不要求全部线程事件的总顺序相同，更不把同步写盘后的host duration当性能数字。

## 2. end返回、统计可见、reset完成是三个时刻

原[ProfileSpan/StatsStorage](https://github.com/apple/coreai-models/blob/3b25d934ccaee2d05edc6e281c555a5a9a13a55c/swift/Sources/CoreAILanguageModels/Profiling/InstrumentsProfiler.swift#L108)的consuming `end()`先取host Mach duration、结束signpost，再创建Task写MainActor上的StatsStorage。**取duration不等于聚合已完成**，后台写入等待也不计入这笔已捕获duration。

测试在同一个MainActor同步片段里执行end→查询，随后才让出执行权。四轮的查询均无记录；待实际count达到1后记录到达。settle循环有明确次数上限，只是观察写入完成，不把“一次Task.yield”当通用barrier，也不把yield次数解释成ns。

更重要的控制是end→`PerformanceMetrics.reset()`→立即查询→让出执行权：

```text
旧span.end()       捕获duration，并安排稍后写入
metrics.reset()    清零tokens/time，清空当前StatsStorage
同步查询           category不存在
让出MainActor      旧写入到达，category重新有1条
再次查询           tokens仍0，旧generation统计已经回来
```

四个原生stats进程都重现。这里没有改StatsStorage、模拟假的异步writer或仅证明Python队列；真正的异步写入来自原ProfileSpan。它说明**reset不是pending-write drain/generation barrier**。不证明完整引擎已经发生跨请求污染：本批未取得/审计全部strategy调用者，也未执行模型，实际调用者可能有更强的完成约束。

可迁移的设计要求是明确“测量结束”和“账本可读取”的不同协议；若跨请求reset，应有可等待的写入完成或generation归属。仅等待推理结果、仅清dictionary、仅看到signpost结束，都不能自动证明旧统计不会再到达。本批没有实现或宣称上游修复。

## 3. 缺失、零、重叠与token分母不能压成一个数

| 原实现对象 | 实际控制 | 结论边界 |
| --- | --- | --- |
| category缺失与真实0ns记录 | 缺失count0/present=false；注入0ns后count1/present=true | metrics facade两者都给0；0属性不够判定覆盖 |
| 聚合单位 | 每轮向独立storage注入3个明确ns值，正反序重放 | total/min/max/avg和ns→s均通过；不是模型耗时 |
| 未end的span | 实际析构告警一次，warmup category仍不存在 | 关闭signpost不等于补一条有效统计 |
| mutating `end(storingInto:)` | 同一span调用两次，count从1到2，duration都从同一个start起算 | 此测试用overload不幂等，不能当重复调用自动去重 |
| consuming `end()` | 同一原文件、同一编译环境：调用一次编译成功，第二次使用触发noncopyable consumed-more-than-once诊断 | 默认所有权约束与上行mutating overload不同，不应混称“所有end都可重复” |
| token分母 | 自有5-token/4-extend-span注入，实际facade计算5/.01=500 | 是公式实验，**不是模型500tok/s或已证实N-1策略bug** |

独立storage使用原internal测试入口，ns注入也走原internal `recordSpan`，不冒充真实引擎生成的span。原MainActor shared storage、public begin/end及其异步写入则实际运行。ownership编译对照只增加一个`REPEAT_END`定义和更换object输出名，干净once baseline排除了缺module等泛化编译失败；不执行两个编译产物。

原[PerformanceMetrics](https://github.com/apple/coreai-models/blob/3b25d934ccaee2d05edc6e281c555a5a9a13a55c/swift/Sources/CoreAILanguageModels/Profiling/PerformanceMetrics.swift)按category取合计，metadata不成为StatsStorage的request分桶。本批保留实际count/present信息，未把缺测默认0、嵌套span求和或mock overall clock当硬件窗口。未调用原summary的速度评价阈值，也未跑Instruments/xctrace；os_signpost API被实际调用，不代表采到了可用的设备trace。

## 4. 背压限制的是未pull队列，不是全部live对象

原[BackpressuredStream](https://github.com/apple/coreai-models/blob/3b25d934ccaee2d05edc6e281c555a5a9a13a55c/swift/Sources/CoreAIShared/Runtime/BackpressuredStream.swift)先启动生产Task，yield闭包里先acquire permit，再向inner stream yield；consumer拿到元素即release permit。depth被clamp到至少1，传0不是rendezvous。

本批用有析构计数的不可变payload、明确gate和完整16维非对称投影/residual。控制边界的实际对象数为：

| 请求depth | 有效队列depth | 未首次pull：queue + 已构造待yield参数 | 已pull但尚未处理：再含consumer持有元素 |
| --- | --- | --- | --- |
| 0 | 1 | 2 | 3 |
| 1 | 1 | 2 | 3 |
| 3 | 3 | 4 | 5 |

队列数是由源式与已返回yield/尚无pull的协议推导，**没有给inner queue或actor waiter加内部探针**；live对象数是真实init/deinit计数。生产者在进入yield之前就构造下一个对象，consumer处理完之前permit已经归还，因此把depth直接乘帧大小当整个pipeline内存上界会漏算。它也不覆盖下游额外持有的结果、GPU/ANE in-flight资源、allocator或RSS。

三种正常depth各完整消费7个payload，producer-error模式先完整消费4个buffered payload，再传播原错误，不把错误改成nil成功。所有108次实际消费均保存原16值与完整16值projection/residual；共216数组/3456标量，独立标准库逐bit dyadic参考全部通过。两seed的正反顺序共108对数组逐字节一致；没有要求host duration、scheduler次数或线程交错逐字节一致。

## 5. cancel不是强制停止；in-scope保留不是泄漏

取消控制先消费一个元素，之后让consumer停在自有gate；生产端第三个yield调用尚未返回时发cancel，再打开gate，让已取消的consumer真正调用原iterator.next。八轮对应控制均返回nil，原生产Task看到cancelled=true，其待完成yield仍正常返回；原wrapper的drain用于唤醒continuation，不是物理设备drain。

两种自有producer只差是否在yield后调用`Task.checkCancellation()`：

- cooperative：构造/返回3次yield后抛CancellationError，consumer只拿到第一个完整payload。
- ignores：即使Task已cancel，仍完成有限的7次构造/yield；原闭包返回Void，忽略inner yield的terminated结果，不能替调用者强制中断任意工作。

这里的consumer在自有gate上等待时不会自动被取消打断，实验明确手动打开gate；不把这个安排当真实视频调用链。需在昂贵生产前/适当边界检查取消，并分别处理queue teardown和下游工作完成，本批没有验证无限忽略取消的producer能自行终止。

两种取消模式在wrapper仍被调用者持有的`stream_complete`点都有1个live payload；**离开stream作用域后的显式控制均为0**。不能用中途快照宣布内存泄漏或buffer永远不释放。producer body的defer事件也早于原wrapper后续finish，不等于拿到了隐藏Task句柄并join所有工作；本批能证明的是这些原生进程的终态与受测对象作用域释放，不是任意任务/设备的全局完成。

## 6. 对ANE推理/视频服务的吸收与未覆盖项

这些host协议决定统计是否可信、取消后是否继续生产、下游资源何时可复用，因而与ANE/GPU前后处理及视频流水相关；它们不需要先拿到物理ANE才能做有效研究。但本批没有实际视频帧、模型state/KV、多consumer并发next、错误恢复后的第二个请求、真实memory pressure、功耗或端到端速度证据。不能由Foundation组件通过推导完整Core AI引擎或TurboCider后端通过。

## 实现参考与实验

默认stdlib重放执行全部816事件、全部216数组、统计单位/状态与实际错误分类，并实际拒绝错误seed的完整消费者，以及把旧异步写入当作reset后空账本的反事实。它不运行Swift、Core AI或设备。

/Swift客户端、compile-only所有权对照/fixture及均只写新owned目录，不修改上游或安装包。初始curator误把adapter install-name假设为@rpath，实际otool是相对名字；只修正元数据验证期望，没有重采、改原binary/源码或调数值gate，失败发生在写curated artifact之前。

复查条件：固定源码、Swift/SDK/runtime、adapter/logger行为、actor isolation、stream ownership/consumer并发、取消检查位置、真实strategy计数或统计reset协议变化。实际模型、物理ANE和性能资格仍未取得；研究继续。
