# Apple Metal 4 counter heap、反馈与资源生命周期来源

<!-- knowledge
id: source-official-metal4-counter-heaps
type: source
-->

[来源索引](../README.md) · [实际时钟/粒度](../../metal/case-studies/metal4-counter-heaps-and-time-domains.md) · [生命周期/范围失效反例](../../metal/case-studies/metal4-feedback-queue-lifetime-and-range-invalidation.md) · [十五份SDK header与八份官方JSON身份](../../metal/sources.md#metal4-counter-timing-sources-20261017)

2026-10-17读取SDK26.5及Apple官方DocC JSON；仅发布原创概括、URL和SHA，不镜像Apple headers/网页正文。SDK声明、源名、实际对象创建、设备执行和生产资格分层。

## 新API不是旧采样点的别名

`MTL4CounterHeap`由新的device工厂创建；`MTL4ComputeCommandEncoder`提供带granularity参数的timestamp命令。这些是macOS26.0等系统引入的API。旧`supportsCounterSampling(.atDispatchBoundary)`回答旧采样接口的能力，不直接决定新heap/encoder是否可用，也不能用新API回填旧MLX时间。

- [relaxed](https://developer.apple.com/documentation/metal/mtl4timestampgranularity/relaxed)优先降低观测干扰，可能只在encoder边界采样。
- [precise](https://developer.apple.com/documentation/metal/mtl4timestampgranularity/precise)请求更细的边界，但可能增加成本，例如内部拆分encoder；不是无扰动保证。
- [writeTimestamp](https://developer.apple.com/documentation/metal/mtl4computecommandencoder/writetimestamp(granularity:counterheap:index:))等待此前工作完成，但后续工作可能已经开始；相邻marker差不能直接叫独占shader busy。
- [queryTimestampFrequency](https://developer.apple.com/documentation/metal/mtldevice/querytimestampfrequency())返回GPU timestamp ticks/second。实际heap时间仍应对父提交的host seconds核查，不能继承另一接口的1ns/tick。
- [invalidateCounterRange](https://developer.apple.com/documentation/metal/mtl4counterheap/invalidatecounterrange(_:))是CPU时间线操作；调用方须确保GPU不在用该heap，失效项resolve为0。此合同与旧counter的全1错误sentinel要分开。本库实际非零location反例不会因为文档如此声明而被覆盖。
- [allocator reset](https://developer.apple.com/documentation/metal/mtl4commandallocator/reset())要求该allocator发出的GPU工作已完成；结束recording不等于完成。allocator在end后继续服务其他buffer与reset释放旧存储是两个动作。
- [commit feedback起点](https://developer.apple.com/documentation/metal/mtl4commitfeedback/gpustarttime)是host seconds，描述该次提交的buffer集合；不是一个旧`MTLCommandBuffer.status`记录或逐kernel时钟。

SDK另明确：每次`beginCommandBufferWithAllocator`会清掉该buffer以前的residency声明；argument table在dispatch/execute时快照资源绑定；依赖dispatch在同一pass用encoder barrier，跨pass的queue barrier不是同一个域。`feedbackQueue`声明为assign，默认nil使用内部反馈queue。`objc_precise_lifetime`、公开运行时property attributes与no-submission实验只能限定可观察生命周期，不能证明私有实现的retain/release细节。

## 当前实测与未知

M4 Max/25G83创建Metal4 queue、allocator、CB、argument table、timestamp heap成功；随后独立原创合成消费者取得720个提交窗口、1680个marker，queried frequency=24MHz，旧配对API仍校准为1ns/raw tick。该API/时钟分域结论见[实测页](../../metal/case-studies/metal4-counter-heaps-and-time-domains.md)，不由文档单独授予。

自定义feedback descriptor的scope退出故障、`NSMakeRange(1,1)`实际清错slot的拒绝记录见[负例页](../../metal/case-studies/metal4-feedback-queue-lifetime-and-range-invalidation.md)。这里只读固定资料，不报告最新bug/fix状态、不联系上游。Metal4语言/队列可用不等于M5、TensorOps/NAX、ANE或任何MLX模型已迁移。

编辑性迁移说明（2026-10-17）：复核本页仅调整了本地案例/来源链接并补此说明，Apple URL、来源快照、数值与能力结论未更新；未重新签发历史实验。迁移前原文 SHA256 为 `50c387dbc8cbb2a80eb5c86e86cdcc7994d735b4390690934ba851424856909a`，当前文档身份由来源登记提供；版本复查日期与窗口保持不变。
