# Metal 4 counter heap实测：24MHz时间域、relaxed合并与precise扰动

<!-- knowledge
id: technique-metal-case-studies-metal4-counter-heaps-and-time-domains
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

[GPU导航](../README.md) · [生命周期/范围失效](metal4-feedback-queue-lifetime-and-range-invalidation.md) · [官方合同](../../sources/official/metal4-counter-heaps.md) · [来源](../sources.md#metal4-counter-timing-sources-20261017) · 全部720窗口/输出packet

2026-10-17，第二十六轮。M4 Max/macOS26.6.2/build25G83/SDK26.5的旧dispatch counter能力为false，但独立Metal4对象/encoder接口可运行。新原创两dispatch消费者完成720个窗口，576个用于配对；1680个marker全部按设备查询的24MHz换算后被实际父提交窗口包含。**不能把上一轮旧API的1ns/raw tick迁移到新heap**。

本页是Metal4独立客户端，不是给未修改MLX wheel新增了逐kernel profiler，不是M5/TensorOps/NAX或ANE实验。精细marker仍会扰动执行，也不等于独占shader busy。

## 1. 三种时间不能共用一个单位字段

| 数据来源 | 当前观察/单位 | 验证方式 |
| --- | --- | --- |
| `MTLDevice.sampleTimestamps`前后配对 | CPU为ns，旧GPU原值仿射映射1ns/tick | 每进程3+3次Mach/UPTIME括号，原固定CPU策略 |
| `MTL4CounterHeap`实际marker | queried frequency=24,000,000 ticks/s，即候选125/3ns/tick | 1680个原值×该有理数，全部在父feedback host窗口±2µs内 |
| `MTL4CommitFeedback.GPUStartTime/EndTime` | host seconds | commit/callback以及CPU sharedEvent-ready括号，720个全部通过 |

这里新heap的绝对epoch与host窗口可比是**本轮观察检查通过**，不是仅靠frequency声明推导。错误地把heap tick当1ns，1680/1680均不被父窗口包含。旧配对API同时仍是1ns/tick；Mach timebase也恰为125/3，不能由数值相同就宣告所有timestamp API都是Mach ticks。

未采样的none臂没有marker，时间为unknown，不填0。失效heap的0是无数据，不是零耗时；时钟缺失、sentinel、倒序、未知slot或父窗口错误保持拒绝/不可用。旧实验未回填、新API不替代旧sampling-point能力位。

## 2. 真实执行的最小消费者

原创native使用Metal4 compiler/library/pipeline、语言4.0、mathMode=safe。两种N/R为4099/32与65539/128，group128，尾部17个guard；不是MLX源shader或真实模型权重。

```text
同一个argument table：slot0=x → dispatch A → 显式依赖barrier
                           slot0=z → dispatch B(a,z) → sharedEvent ready
```

输入是完整可生成的dyadic序列：`x=((13i+seed)%257-128)/64`、`z=((7i+3seed)%251-125)/128`。A逐次加交替x/32、x/16，共R轮；B从A逐次加z/64，共R轮，再加.5。独立闭式为`A=x*(1+3R/64)`、`B=A+z*R/64+.5`，这些数在本域FP32可精确表示。

每次GPU返回后核对全部x/z/A/B及guards，保存当前A/B完整SHA；只在整个bytes相同后复用文件。1440个完整输出出现值形成8个独立实际packet（2seed×2shape×A/B），标准库逐值重放全部packet，所有出现值hash均匹配。不是1440次独立数学参考，也不是预训练质量。

table在A编码后改slot0，B使用z；在GPU提交前两个编码都已完成。因此输出还验证了**dispatch时资源快照**，而非GPU执行时读取最后一份table地址。GPU地址不公开，JSON只用x/z等逻辑名。

## 3. 时间粒度会改变可识别边界

one-encoder在A前、A后、B后写3个marker；split-encoders为A/B各前后共4个marker。单encoder用intra-pass barrier；分encoder用consumer queue barrier，均显式Device visibility，不借timestamp代替数据依赖。

| 模式/布局 | 正式边界段 | 实测结果 |
| --- | --- | --- |
| relaxed / one encoder | 192段 | 前96段正差，后96段delta=0；后两个marker全部相同 |
| relaxed / split encoders | 192段 | 全部正差 |
| precise / one encoder | 192段 | 全部正差 |
| precise / split encoders | 192段 | 全部正差 |

含cold/warm时共有960段：840正差、120合并/分辨率受限段。relaxed单encoder的第一个差可能反映整pass，不能全归给A；B输出已完整正确，第二个差为0仍不能叫B零工作/零耗时。所有合并段`accepted_marker_interval_seconds=null`，保留raw_delta=0作事实。

precise请求更细边界，Apple声明可能内部拆encoder。本轮观测到更细marker和不同提交跨度，但没有内部encoder/ISA trace，**不将“可能拆分”写成已证明的内部机制**。公开命令也不保证后续工作一直等到marker写完，因此正差仍是instrumented marker span，`pure_shader_busy_seconds`及`unprofiled_dispatch_GPU_seconds`全部null。

## 4. 观察开销：同配方none对照，不给通用低开销承诺

4个顺序运行cohort，seed17/29各两组；12cell=两shape×两布局×none/relaxed/precise。每cohort先1cold block、2warm blocks，之后12次Latin rotation，使每个cell在12个位置各出现一次。保存全部720窗口，正式只用576；逐formal-round与同shape/布局none配对，不做不同样本中位数相除。

下表是四cohort**各自配对倍率中位数的范围**，倍率为observed/none（大于1表示观测模式更慢），不是置信区间或模型speedup：

| N | 布局 | 模式 | host ready倍率 | GPU feedback span倍率 |
| --- | --- | --- | --- | --- |
| 4099 | one | relaxed | .999–1.007 | 1.000–1.019 |
| 4099 | one | precise | 1.085–1.101 | 1.238–1.265 |
| 4099 | split | relaxed | 1.005–1.012 | .974–1.006 |
| 4099 | split | precise | 1.032–1.095 | 1.054–1.105 |
| 65539 | one | relaxed | .991–1.008 | .994–1.004 |
| 65539 | one | precise | 1.012–1.035 | 1.065–1.075 |
| 65539 | split | relaxed | 1.004–1.023 | 1.002–1.009 |
| 65539 | split | precise | 1.052–1.086 | 1.028–1.035 |

小N/one/precise的四组host IQR都高于1；大N/one/precise的四组host IQR均跨1，不能把各自中位数一概叫稳定参数。relaxed接近1或个别低于1不证明“加计数器提速”。cohort、时序、缓存、host同步噪声及被观察的调度都参与；没有自动采用门槛、通用百分比、模型或能耗资格。

host从`beginCommandBufferWithAllocator`前到sharedEvent-ready，含编码/提交/等待，不含pipeline建立、输入准备、CPU验证、resolve/hash/IO/allocator reset。feedback span为该次一个buffer提交的GPU跨度；两者不是同一边界。全表在相同native源码、pipeline及数学配方内比较，未给跨引擎或MLX overhead排名。

## 5. 生命周期与两个保留失败

每次begin重新声明residency；table/资源保持存活；先end encoder/CB，再commit，signal sharedEvent并有界等待，另外drain feedback。只有两种完成均成立才resolve、清heap和reset allocator。end recording允许allocator服务后续buffer，不意味着可reset仍在用的存储。

前置自定义feedback queue的scope崩溃已用14个无GPU提交单元缩减；随后默认queue揭露非零location失效范围检查exit17。本次正式路径是**独立的完成后整heap失效协议**，每次0..8全部resolve为0；原两类失败继续失败，没有包装成Apple修复或范围API通过。详情见[失败缩减](metal4-feedback-queue-lifetime-and-range-invalidation.md)。

旧MLX固定`64ea011c` Device提交选段仍用传统Concurrent encoder；三个选定文件无MTL4文本不是整个最新仓库不支持的证明。本轮没有注入/修改MLX、运行原Model或改变生产默认；MLX descriptor-only工厂扰动A/B仍是另一个未执行项。

## 6. CPU重放与复查

labs/clock合同/整heap decoder与完整tests不加载MLX/Metal或执行GPU。审计四个已完成原件、180行journal/进程、所有输出/输入hash、时钟和配对统计；新设备driver必须显式fresh ignored目录及GPU准入。

复查条件：SoC/OS/SDK/compiler、frequency/epoch/heap ABI、granularity、allocator/argument/residency/barrier/queue生命周期、shape/R/输出需求、完整消费者变化。需要实际MLX/其他引擎迁移、原模型与长期请求、能耗或ISA归因时另作证据，不把本配方结果继承为产品默认。
