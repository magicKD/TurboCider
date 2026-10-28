# MLX concurrent encoder-stage计数器：实际窗口与多dispatch聚合

<!-- knowledge
id: technique-metal-case-studies-mlx-concurrent-encoder-stage-counters
type: case-study
stage: prefill, decode, pipeline
operator: attention, gemm, norm, rope, layout, scheduling
backend: metal, cpu-reference
device: cpu, gpu
symptom: measurement, precision-loss, launch-bound
semantics: approximate, contract-only
representation: bf16, fp32
boundary: cpu-gpu, producer-consumer, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [能力与primitive边界](mlx-stage-sampling-capability-and-primitive-boundaries.md) · [前轮PSO关联](mlx-dispatch-pso-and-function-association.md) · [来源](../sources.md#mlx-stage-counter-sources-20261017) · 原值与校准 · 全部metadata · [终态审计（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第二十五轮。未改MLX0.32.3 wheel、原MLX-LM完整Qwen，在M4 Max/macOS26.6.2/build25G83/SDK26.5取得**1370对真实encoder-stage时间戳**，均通过同进程校准与实际父CB包含检查。四个stage进程中184个窗口属于模型forward；其中168个包含多次direct dispatch，不能拆成单kernel耗时。

这是从“有PSO/dispatch关联和父CB时钟”进一步到“有encoder边界时间”，**仍不是逐dispatch计时、shader busy或生产性能排名**。counter只枚举到timestamp；dispatch boundary不受支持，明确请求以exit3拒绝，不静默降为stage。

## 1. 为什么选择encoder attachment而非dispatch采样

[Apple公共合同](../../sources/official/metal-counters.md)要求分别查询采样点与counter set。新无提交能力探针实测stage=true、dispatch/draw/tile/blit=false，集合仅`timestamp/GPUTimestamp`；它不导入MLX、不创建应用计算工作。

[固定MLX Device提交链](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/device.cpp)使用`computeCommandEncoderWithDispatchType:`。本轮原创observer在选定具体类的该工厂处构造`MTLComputePassDescriptor`，并将原参数原样赋给`descriptor.dispatchType`。实际所有捕获encoder均为Concurrent=1，未将原并发encoder改成Serial。

stage模式绑定`sampleBufferAttachments[0]`的start/end encoder索引，使用shared timestamp buffer、容量4096、每timestamp结构8 bytes。每个encoder分配两个唯一连续slot，不重复使用或回绕；当前每进程使用694个BF16或676个FP32 slot。容量不足、无对应具体encoder类或创建失败都会失败，不静默漏采或用0补齐。

没有向不支持的dispatch点发送`sampleCountersInBuffer`命令，没有改MLX源文件、磁盘wheel、系统设置或其他进程。**保留Concurrent参数不等于观察透明**：换工厂/descriptor、counter、reference采样、锁、日志和对象持有均可能改变执行与生命周期。

## 2. 两条独立关联链

```text
encoder创建 → 固定(start_slot,end_slot) → resolved原值 → 同进程仿射时钟 → stage窗口
       └→ parent CB token → commit/完成 → SDK GPUStartTime/EndTime包含检查
       └→ 最新PSO绑定 → direct编码 → 创建时MTLFunction.name
```

完整44755条事件来自4个meta和4个stage进程；1733个已完成CB分别保留公开SDK seconds和host bracket，15840个Model direct编码保持前轮最近绑定/外键/PSO group上限合同。scope按实际编码/提交，不按可能更早发生的CB创建分摊工作。对象token只在各进程内有意义，不是地址、跨进程binary身份或IR层号。

meta使用原工厂、不挂counter；stage替换工厂并挂counter。24组Model/primitive阶段的有序`function_names/api/grid/group`签名逐组相同。**没有证明完整CB拓扑都相同**：例如FP32 startup等额外阶段使meta两个进程总完成CB数为82/79，stage均80。签名一致仅支持所检查的direct编码字段保值，不能证明屏障、资源访问或driver调度完全一致。

## 3. 不先假定timestamp单位

CPU单位候选分别测试ns与Mach ticks。按预先固定的lexicographic顺序选择第一组通过单位、bracket和单调检查的前后reference，不按最快stage、最短reference或期望结果择优。以Fraction保留有理数：

```text
scale = (CPU_after - CPU_before) × CPU_ns_factor / (GPU_after - GPU_before)
mapped(t) = CPU_before × CPU_ns_factor + (t - GPU_before) × scale
```

本轮四进程均选(0,0)，CPU候选为nanoseconds，GPU实测映射为1ns/raw tick；全部24个reference bracket最大1500ns。**不是因为CPU原值与GPU相等就宣告通用ns**，更不能再把其乘Mach的125/3。旧无reference事件、其他进程/OS/GPU不获得回填或外推资格。

原值必须正向、非0/全1 sentinel，并与native_valid一致；每行encoder/CB/scope/slot与工厂记录精确对应。derived窗口必须处在校准区间内，并被真正的父CB SDK窗口包含，容许预定2µs端点误差。所有1370对通过，转换结果仍标`source_kind=derived`；不可用保持null/issues。

## 4. 多dispatch窗口究竟知道什么

四个stage进程、16次Model forward的184个窗口：

| 每encoder实际direct编码数 | 窗口数 | 可得结论 |
| --- | --- | --- |
| 1 | 16 | 该边界只有一次被本观察器记录的direct编码；不是通用纯shader时间 |
| 2 | 8 | 两次编码共用窗口，不能独立分摊 |
| 13 | 8 | 聚合组，不是13个同耗时kernel |
| 51 | 120 | 多函数pipeline组，不能把整段归给最后绑定PSO |
| 52 | 32 | 同上，Concurrent参数不提供各child独立起止 |

一个实际BF16 scope100的51-call窗口约828.041µs，同时关联RMSNorm、RoPE、copy、full SDPA、fused/split-K GEMM、归并及生成elementwise函数。它是这个encoder的**整体边界跨度**，不是attention耗时或各函数GPU热点；不能按出现次数平均分配。更不能将多encoder的sum/union/span叫全设备GPU忙碌。

自检用三个自有PSO区分created-only/bound-only/used，两个完整128值输出由native端检查，两个dispatch共用**一对**stage计数器。公开仅保留native完整检查bool与全部API/counter metadata；CPU独立重放关联和时钟，不声称重新验算未保存的输出bytes。

## 5. 观察保值、数学与速度资格分账

正式16进程：stock/off/meta/stage×BF16/FP32×17/29；两篇文章32-token fresh prefill，各接一个固定真实continuation；另各有vector/full实际QKV primitive。每进程290个原BF16参数位展开SHA完整一致，FP32仅提升这些值，不是另一checkpoint。

64份151936维完整Model logits与32份primitive有效区域数组组成12个身份组，跨四模式/两重复逐位相同。重新独立算4个完整24层FP64-reduction/FP32-stage prefix参考：原`.005+.005*abs(ref)`且relative-L2≤.005不变，FP32 32/32通过、BF16 0/32通过。32个primitive条件参考均通过，不能覆盖完整模型失败；[另一页](mlx-stage-sampling-capability-and-primitive-boundaries.md)解释其独立域。

全部`qualified_performance_ms`和`per_dispatch_GPU_seconds`仍null。stock/off无counter窗口是unknown，不是0；meta与stage之间同时改变descriptor和counter，没有descriptor-only臂，因此也不能把时间差单独因果归给某一项。无任务质量、能耗、ISA、occupancy、长serving、M5/NAX或生产默认资格。

## 6. 离线重放与复查

标准库tests完整重放1370对counter及metadata，含sentinel、错误单位/CB/slot、超容量、不支持点、缺时钟/越界和聚合歧义负控；普通测试不导入MLX/NumPy、不加载动态库或启动GPU。需显式原资产及已有NumPy，只重算CPU数学；另逐一核对96个完整文件、所有终态receipt、源码和library/hash，不重复GPU实验。

复查条件：sampling-point能力、counter集合/ABI、SDK/OS/SKU、reference时钟、native/client/wheel/source、具体类/selector、资源生命周期、dispatch模式、shape/dtype/消费者或scope变化。新原生实验必须走显式supervisor、fresh ignored目录和GPU串行准入；不能让CPU学习自动触发设备。下一步是可信逐dispatch/IR关联或新设备控制，不是把本轮聚合窗口改名为kernel时间。
