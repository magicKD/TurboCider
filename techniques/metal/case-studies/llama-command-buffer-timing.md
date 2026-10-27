# llama.cpp完整模型的Metal命令缓冲计时：真实SDK窗口，不是逐kernel忙碌时间

<!-- knowledge
id: technique-metal-case-studies-llama-command-buffer-timing
type: case-study
stage: prefill, decode, pipeline
operator: scheduling, layout
backend: metal, cpu-reference
device: cpu, gpu
symptom: measurement, launch-bound
semantics: numerically-equivalent, contract-only
representation: q8, fp32, gguf
boundary: producer-consumer, cpu-gpu, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [前轮完整引擎](llama-pretrained-output-demand.md) · [observer反证](llama-metal-flash-and-eval-observer.md) · [来源/私有patch](../sources.md#cb-profile-sources-20261017) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第十八轮。同一官方Qwen2.5-0.5B Q8_0完整预训练引擎，新增**公开`MTLCommandBuffer.GPUStartTime/GPUEndTime`完成回调采集**。12个正式进程保存384个完整151936维logits，stock/诊断关闭/诊断开启逐位相同；开启诊断的4进程取得388个有效、时钟括号一致的CB窗口。这补上真实完整消费者的公开CB时间证据，**没有补上per-kernel时间、shader busy、occupancy、PMU或能耗**。

## 1. 私有诊断副本与13个真正CB提交站点

固定`ggml-org/llama.cpp@64e9bceb2c3a856efed96feda784a50947049feb`。[原始context实现](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/ggml/src/ggml-metal/ggml-metal-context.m)持有真正CB，但选定接口没有导出这些SDK时间。前轮`cb_eval`只得到GGML逻辑节点，且query-only也增加split同步，不能给节点冒填物理时间。

本轮新建ignored源码副本，仅改两个Metal `.m`和CMake。原创observer/私有接口给13处显式CB commit装完成回调，随后各调用一次原`commit`；不改数学、encoder、fence、shader，不增GPU wait。CMake增加ObjC++观察器，强制原`.m`保持Objective-C编译，不给整个引擎换语言。

| 审计站点 | ID | 解释边界 |
| --- | --- | --- |
| graph正常/abort后next-buffer | 7/4 | CB包含许多逻辑节点，不等于一个kernel |
| context async set/get/copy、event signal/wait | 1–3、5–6 | transfer/event不能全算shader算术 |
| device私有buffer memset/set/get/copy/clear | 9–13 | 当前shared-buffer配方未命中，不代表不必覆盖 |
| residency collection初始化dummy blit | 8 | 一次1-byte填充，与模型分账 |

特别注意：[device源码](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/ggml/src/ggml-metal/ggml-metal-device.m)的5ms后台循环调用`requestResidency`，**不是每5ms提交GPU dummy**；dummy在collection初始化时调用一次。另两处`[buf->rset commit]`是`MTLResidencySet`，不能按`commit`字符串统统hook。不做全局swizzle，reference及前轮stock源码/库不动，无上游提交。

## 2. 秒已经是秒；完成窗口须核对真实时钟关系

SDK26.5的`MTLCommandBuffer.h`已读取并SHA固定；[GPUStartTime](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime)/[GPUEndTime](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpuendtime)是**host time seconds**。0表示尚不可用，不是零耗时。只在完成回调读取，要求status=4、两端有限且正、end≥start；error/status5、缺失/非有限/反向值保持null。

SDK秒只乘1e9映射成ns；CPU独立用`mach_absolute_time × numer/denom`得ns。本机timebase=125/3，不能对SDK秒再乘125/3。沿用[既有时钟合同](../../profiling/timestamp-calibration.md)预置的2µs端点余量，检查SDK窗口在自身commit前括号/callback后括号之间，以及**同一次native调用**的host窗口内。它不宣称2µs绝对精度或任意设备恒定映射。

commit host时间在安装handler之前读取，不是精确的GPU enqueue时刻。native先截取host结束点，再等该scope的回调日志排空，退出时统一fsync；不假设`waitUntilCompleted`已保证自有日志写完。回调可在host结束前竞争CPU、锁和IO，窗外排空**不意味着观察器无扰动**。

## 3. 三组控制先比较完整输出

正式矩阵stock/off/on × Flash off/on × order replicas17/29，共12个串行进程。每进程2轮×2个历史39-token prompt×8次greedy forward，32份完整输出，无`cb_eval`。另3个smoke各2份输出，**不计入384份正式总数**。

原生客户端用同一stock public ABI，可选发现私有符号；驱动显式设置子进程库路径，移除继承的capture/debug覆盖，核对实际加载的每个llama/GGML dylib路径与SHA。stock无诊断符号，off有符号但不开日志，on必须成功创建唯一日志。同一模型SHA、原官方tokenizer/template IDs、KV F32、last-only、context请求128/实际256、全部层GPU offload等保持一致。

384份新输出全部保存、验证完整SHA/有限性及实际greedy choice；32个Flash×prompt×step分组跨三模式、两轮及两replicate各只有一个payload SHA。每份也与前轮同prefix的**完整独立CPU检查见证**逐位一致，继承原数学门槛：0通过/384拒绝。

这是**同prefix+完整payload身份转移**，不是又重算384次CPU模型。门槛仍是逐元素`.005+.005*abs(ref)`且relative-L2≤.005，未用引擎CPU量化forward、top1或“观察前后一致”替代gold。因此观察保值不授予模型质量通过，也没有新增任务/NLL/KL/BF16质量结论。

## 4. 388个窗口分别是什么

on的4进程共128次模型调用，每次恰好**2个graph CB + 1个async-get CB**，合计384。两个graph CB不是“两层”或“两kernel”，提交/编码打包了大量逻辑工作。另4个初始化dummy属于startup scope1；无未归属scope0。全部388个窗口有限、status4且commit/callback时钟括号通过，384个模型窗口也通过自身host括号。

每进程另有两次250ms idle控制。on的8次idle在13个已覆盖站点内无提交，记录集合可报告sum=0；**不等于整台GPU空闲**，其他queue/framework/进程和未覆盖路径仍未知。stock/off没采集，GPU时间是null而不是0。

保留三个量：`sum`逐CB持续时间之和，重叠时重复计；`union`窗口并集长度，仍可能含内部stall/event，不是shader busy；`span`最早开始到最晚结束，含CB间gap。不能把`span−union`直接命名为CPU时间。

本批每次调用的三个窗口不重叠，所以sum=union；decode常有gap。它不证明别的queue无重叠，也不证明gap可消除或优化后请求会加速。

## 5. 诊断结果，不发布倍率/排名

Apple M4 Max、macOS26.6.2/build25G83、SDK26.5、Apple clang21.0.0。下表只列observer开启、round1中位数；prefill各4样本，decode各28样本。完整三模式、round0、每次窗口和388条原始记录均保留于证据。

| Flash /阶段 | host ms | CB sum=union ms | CB span ms |
| --- | --- | --- | --- |
| off /prefill | 22.003 | 17.530 | 17.532 |
| off /decode | 3.954 | 2.464 | 2.786 |
| on /prefill | 20.944 | 16.884 | 16.886 |
| on /decode | 4.174 | 2.507 | 2.994 |

Flash是原attention选项，不是observer开关。host含scope begin、tokens/batch/reset、decode/sync及完整CPU logits getter；**不含**后续回调排空、持久vector复制、有限值/argmax、文件/SHA、chat render/tokenize/IPC。不能直接与前轮render-to-text或Candle请求wall相除。

stock/off/on诊断host中位数有时更大有时更小，如Flash-off decode为3.490/4.459/3.954ms。这些少量、非同时配对的进程样本不支持稳定observer开销倍率；开启回调的窗口不能当stock未观察窗口的无扰动替身。所有`qualified_model_speed_ms`仍null，不用原数学预算已拒绝的路径排名。

## 6. 合同、失败与后续

不启动任何引擎/GPU的CPU合同/测试：

负控覆盖重复Mach换算、0/缺失/NaN/overflow/error/反向端点、重叠窗口、缺失/重复回调、foreign scope/未审计site、disabled未知与有界空集合。CPU测试从保存的原始CB重新归一化全部正式模型scope，核对每份historical数学见证；不需要外部模型/build/integration。

设备实验必须显式ignored路径和supervisor，先核对来源capsule的三个文件patch及独立helper；普通检索/test不自动启动。构建参数见`llama_cb_profile_probe.py --help`，复现须使用全新root/run/log目录并串行GPU进程，不能覆盖sealed旧build或日志。只做CPU验证与来源封装。

两次自有构建失败保留：首次CMake把`.m`识别为ObjC++，第二次变量`id`遮蔽ObjC类型；修正后第三次构建、smoke、正式矩阵成功。这是诊断工具准备错误，不是引擎/驱动故障。未下载模型、清理用户cache或改系统设置。

复查条件：源码/13个站点、helper/ABI/dylib、SDK/OS/SoC/时钟、callback语义、buffer storage/transfer/event/capture、batch/输出需求/context/consumer任一变化。后续转held-out质量/NLL/KL、更大context/其他consumer或独立per-kernel/counter证据，不以重复固定prompt扩充资格。历史synthetic gputrace时间继续null，不能拿不同来源的新SDK窗口回填旧trace。
