# 私有 profiling 再取证：constructor、getter 与真实测量不是一回事

<!-- knowledge
id: technique-ane-profiling-constructor-provenance
type: method
operator: scheduling
backend: private-ane, cpu-reference
device: cpu, ane
symptom: measurement
semantics: contract-only
boundary: request
api: private, reference-only
-->

[ANE目录](README.md) · [工具地图与SOP](profiling.md)

2026-10-16，M4 Max / macOS 26.6.2。本轮只做 Objective-C 元数据与有界 constructor 诊断，**没有加载模型或执行推理，也没有取得有效 PMU capture**。原来的推理成功但 stats 缺失记录仍保持原身份；这轮不能替它补上硬件时间。

后续独立完成了[真实request双槽/生命周期实验](request-stats-lifecycle.md)：99次完整输出、26.6.2的`count`validation负例和seeded getter/held fallback对照；不回填为本篇constructor-only批次当时已验证。

## 1. 先把“有 stats”拆开

| 层次 | 需要的证据 | 本轮结果 |
| --- | --- | --- |
| 类/selector 可发现 | 当前进程的 class、selector、参数/返回类型 | 可发现，保留实际编码 |
| constructor 能返回对象 | 符合实际 ABI 的独立调用 | 零buffer factory/init均nil；seeded路线可返回对象 |
| 请求可携带对象 | 同OS上的request构造与validation通过 | 未运行模型请求，不知道 |
| capture有效且属于本次请求 | payload/schema、对象来源、实际route、输入输出代际与完成关联 | 未取得 |
| duration单位/含义可信 | 独立标定、明确measurement scope、非调用方预填 | 未标定，始终null |

这不是五个互相替代的成功标志。尤其一个 getter 可以返回由调用者提供的数字，并不需要发生任何设备计算。

## 2. 实际 ABI 和两个新进程重复

原创诊断入口是Python supervisor内的探针和原生实现。先检查 `NSMethodSignature`，类型不符就退出，不尝试“兼容性猜测”。这台机器实际返回：

| selector | 实际编码 | 调用所需的显式参数 |
| --- | --- | --- |
| `+statsWithRequestPerformanceBuffer:statsBufferSize:` | `@32@0:8^^v16^I24` | `void **`、`unsigned int *` |
| `-initWithRequestPerformanceBuffer:statsBufferSize:` | 同上 | 同上 |
| `+statsWithHardwareExecutionNS:` | `@24@0:8Q16` | `uint64`值，不是NSNumber指针 |
| `-hwExecutionTime` | `Q16@0:8` | 无显式参数，返回uint64 |
| `-perfCounterData` | `@16@0:8` | 无显式参数，返回对象 |

逐进程结果在证据JSON的 `profiling` 中，原始目录位于本地 `outputs/ane-followthrough-20261016/`，不作为便携文档依赖。

| 调用 | 重复 | 对象 | 返回buffer | payload | getter原值 |
| --- | --- | --- | --- | --- | --- |
| metadata | 1次成功，不调constructor | 不适用 | 不适用 | 未读 | 未读 |
| factory，初始`buffer=nullptr,size=0` | 2个新进程 | nil | null/0字节 | null | null |
| initializer，同一初始状态 | 2个新进程 | nil | null/0字节 | null | null |
| seeded，传入uint64 `123456` | 2个新进程 | `_ANEPerformanceStats` | null/0字节 | null | `123456` |

所有行都没有推理，`physical_duration_seconds=null`、`counter_capture_qualified=false`。最后一行是刻意设计的负对照，不把数字除以十亿变成“实测秒数”。nil只否定这台机器上此零buffer配方的对象结果，**不证明整台机器不支持PMU，也不证明合法request无法在执行后附上stats**。

最早一次metadata构建因漏了 `<initializer_list>` 失败。保留其compiler失败、stderr哈希和旧源码身份，修正后另建目录执行；没有把编译失败统计成一次native探针成功。总计7次成功的无推理诊断、1次保留的编译失败。

## 3. Espresso：请求 route 不等于实际 route

固定 [Espresso `ane_interop.m`](https://github.com/christopherkarani/Espresso/blob/05ea6a3da91be642abd3d4741c0341a941ce1064/Sources/ANEInterop/ane_interop.m#L413)，阅读413–494、2342–2445行，完整文件SHA和范围在证据JSON。

它的load准备阶段采用相同零buffer factory配方，并单独设置stats mask/load options。eval阶段则会根据client可用性、realtime加载状态和实际返回结果，退回标准in-memory入口。成功后依次寻找request上的stats、stats array、原保留对象。

因此应记录 `requested_route`、`attempted_route`、`successful_route` 和stats来自哪个对象；只有环境变量或一个合并后的success布尔值不足以区分。当前接口没有完整导出这些route身份，这是一条源码层面的可观测性缺口，不是本轮复现了fallback错误。也不能把“factory=nil”简化成“后续所有eval永远没有stats”。

没有启用上游的force-cold、power-saving、wired-memory、IO-fence或安全设置相关选项。诊断没有调用任意未知buffer指针、IOCTL、全局日志捕获或系统profiler session。

## 4. PMetal：默认0、版本gate与参数类型应分别核对

固定 [PMetal `runtime.rs`](https://github.com/Epistates/pmetal/blob/a225c0e2568f882573b05caeb42d94fed208fdd3/crates/pmetal-metal/src/ane/runtime.rs#L379)，阅读250–455、980–1010行。这个revision的路径是 `crates/pmetal-metal/src/ane/runtime.rs`；搜索引擎给出的另一个`pmetal-ane/src/runtime/perf_stats.rs`路径在此固定树中不存在，不混用其内容。

- availability检查是class加OS gate；unsupported或constructor=nil时可继续普通evaluation，并返回default stats，其中时间字段为0。这个0不能升级为“硬件实际耗时为0”。
- 此revision的兼容gate关闭的是**macOS 27及以后**，不是本机26.6.2；注释中的27请求validation问题属于上游兼容记录，不是本轮在26上复现。
- 该调用点给 `statsWithHardwareExecutionNS:` 传的是NSNumber引用；本机实际ABI要求uint64值。记为需要进一步检查objc2宏/包装层及目标OS的**潜在参数ABI不匹配**。未构建Rust项目、未执行该错误类型调用，不宣称观察到PMetal故障或计时损坏。

这三项分别涉及缺失值表示、版本兼容与调用类型；不能靠一个“支持stats”标志解决。

## 5. 怎样复跑，怎样继续

先运行不涉及任何设备的：

显式私有探针命令，输出目录必须不存在：

换OS/SDK/SoC必须重新读ABI，不能把这里的字节编码当作稳定公共合同。下一步若做自有模型request，要以原始输入输出、实际完成route、stats身份/执行前后变化为基础，再做单位标定；继续把缺失留为null，不能把seed或fallback默认0纳入硬件占用率、MAC吞吐或能效计算。公共部署仍优先参考[官方性能分析入口](https://developer.apple.com/documentation/coreml/analyzing-the-performance-of-your-core-ml-model)与[WWDC Core ML Instruments讲解](https://developer.apple.com/videos/play/wwdc2023/10049/)，其计划/trace/API时间也须分开。
