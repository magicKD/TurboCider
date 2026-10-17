# Apple公共Metal counter与capture：先核查能力和时钟

<!-- knowledge
id: source-official-metal-counters
type: source
-->

[来源索引](../README.md) · [实验步骤](../../profiling/gpu-timeline-and-counters.md) · [本机结果](../../metal/case-studies/capture-counters-and-failure-boundaries.md)

2026-10-16读取Apple公开文档与本机macOS SDK 26.5 headers；这是官方API来源，不是开源仓库或完整Xcode资格。SDK文件SHA登记在[sources.json](../README.md)，不分发Apple headers。

## 已核查的合同

1. [GPUStartTime](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime)是host time、单位seconds；完成前可能为0。完成后读取，零/非正差值不能伪装成有效零耗时。
2. `MTLCounters.h`定义`MTLCounterErrorValue`为全1 sentinel，`MTLCounterDontSample`为不采样索引；缺失值不能填0。`resolveCounterRange`只能用于shared sample buffer。
3. `counterSets`与`supportsCounterSampling`必须实查。支持stage boundary不推出支持dispatch boundary，也不推出存在statistics/utilization各集合。
4. `MTLComputePass.h`的start/end encoder attachment只有支持stage boundary时才合法；否则须不采样。dispatch采样是另一API与能力位，不得悄悄替换。
5. [MTLCounterResultTimestamp.timestamp](https://developer.apple.com/documentation/metal/mtlcounterresulttimestamp/timestamp)给出GPU时间值；不能仅凭`uint64`假定它可直接与host seconds、Mach ticks或另一进程时钟比较。
6. [MTLTimestamp](https://developer.apple.com/documentation/metal/mtltimestamp)说明具体绝对时间表示可能随系统配置不同。[时钟转换指南](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time)要求至少两次CPU/GPU配对采样，并指出配对采样自身可能有运行时扰动。

因此首批collector保留counter原值、有效性、两端配对时钟，但`duration_seconds=null`，不假设ns、不用未知tick生成GPU/CPU重叠图。即使当前机器的CPU/GPU配对数值相同，也不是跨配置单位证明。

2026-10-16后续[校准实验](../../profiling/timestamp-calibration.md)增加`mach_absolute_time`、`mach_timebase_info`与`CLOCK_UPTIME_RAW`实测bracket，另核对每trial完成CB窗口。当前配置的ns候选与1 ns/raw GPU tick映射得到局部验证；转换保持derived，不回填缺参考的旧记录、不外推MLX queue/其他系统，也没有occupancy资格。Mach/libc headers SHA另登记，同样不分发其实现或覆盖原许可。

## 本机范围，不外推

M4 Max / macOS 26.6.2仅通过公共counter API枚举到timestamp集合；stage boundary支持、dispatch boundary不支持。小型真实计算取得有效stage timestamp；未测到occupancy、带宽或ALU利用率，不以FLOPs/逻辑字节推导替代。

CLT环境的`xctrace list templates`退出72，但公共programmatic `.gputrace`与stage counter仍可运行。没有完整Xcode不能直接推出所有Metal profiling入口不可用。原创客户端在labs/metal/bench.mm，不控制其他App、不改变developer directory、SIP、TCC或电源配置。
