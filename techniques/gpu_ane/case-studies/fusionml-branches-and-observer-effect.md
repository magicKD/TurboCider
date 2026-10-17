# FusionML 深读：分支身份、真实并行与探测本身的代价

<!-- knowledge
id: technique-gpu-ane-case-studies-fusionml-branches-and-observer-effect
type: case-study
stage: ffn
operator: swiglu, gemm
backend: cpu-reference, mlx, coreml
device: cpu, gpu, ane
symptom: straggler, precision-loss
semantics: numerically-equivalent
representation: fp16, fp32
boundary: gpu-ane, producer-consumer
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [关键路径](../partition-and-scheduling.md) · [续研证据](../../MAINTENANCE.md)

本轮分别浅克隆并固定两个分支：main为 [`2e36b64da1fc8e00ea90afa2ec7cc1b9744f0464`](https://github.com/ommo007/FusionML/tree/2e36b64da1fc8e00ea90afa2ec7cc1b9744f0464)，benchmarks为 [`4e7121a8f9c70fd6512b233c5705374626323471`](https://github.com/ommo007/FusionML/tree/4e7121a8f9c70fd6512b233c5705374626323471)。同时阅读[作者论文HTML](https://arxiv.org/html/2607.22785v1)。**没有运行上游、训练或复现其性能数字**；以下把实际代码、作者测量与自己的推导分开。

## 1. 为什么必须读两个分支

| 入口/版本 | 实际读到的路径 | 可以得出的结论 |
| --- | --- | --- |
| main `Fusion.linalg.matmul(Tensor,Tensor)` | IntelligentRouter→SmartScheduler | 默认是CPU+GPU row split，不是自动三设备同时执行 |
| main `CoreMLMatmul` / `ANEEngine` | create-model入口抛`modelNotAvailable`；部分op为`notImplemented` | 不能据类名/feature表宣布ANE后端已可用 |
| main `SmartScheduler3` | 需要外部模型缓存；所谓ANE slice里实际调用CPU matmul | 三个逻辑区段不等于三个物理设备 |
| benchmarks `TriComputeMatmul` | 显式ANE opt-in、shape状态、prewarm、完整dispatch/回读、回退 | 比旧Swift主线更完整，但有独立部署限制和成本 |
| benchmarks 的stream probe/MLX-LM patch | eager边界+CPU/GPU stream split | 这是另外的模型/测量入口，不能用它的结果为旧main背书 |

故main中的占位实现不否定论文分支的研究结果；反过来论文结果也不证明当前默认Swift入口已实现同样路径。仓库URL、branch名字或README更新都不足以绑定运行的实现。

## 2. 主线的“零拷贝”与输出覆盖

SmartScheduler的CPU支直接读共享指针并写最终输出；GPU支的`tensorView`实际创建新Tensor并`memcpy`输入row，GPU返回后又copy到最终输出。另一个WorkSplitter同样显式copy slice。统一内存、名为view、注释zero-copy都不能替代实现取证。

SmartScheduler3中，未校准的默认ratio仍给ANE份额；但只有对应模型存在才dispatch那一段，而且slice实际回CPU。**若模型不存在，所读代码没有为那段row补算。** 本轮没有执行该异常路径，故记录为静态覆盖风险，而非已复现输出错误。和vpipe的partial-write案例一样，join之后仍应检查所有预期row属于本代输出。

旧主线性能profile的key主要是size/M，而不是完整M/K/N、dtype、layout与设备环境；对相同M的不同projection不应自动继承最优share。其GPU MPS入口每次commit并wait，所以不能把该framework的同步成本当成所有Metal实现的下限。

## 3. benchmarks 中真正有用的改进

`smart_matmul.py` 默认关闭ANE；启用时显式警告开发环境出现过进程级崩溃。prewarm使用隔离子进程，key包含M/K/N和compute policy，当前process仅尝试加载已预热资产，并限制ANE shape变化。**但不能把其weight hash称为完整内容身份。**

继续追到`ane_backend._weight_hash`：它仅取每个数组序列化后的前1024字节加shape，再截短MD5。两个相同shape的FP16权重，即使第513个元素不同，也有完全相同的hash输入；对应矩阵乘法却可不同。新的CPU见证给出`[513,1]`权重尾部1→2、全1输入的513→514差别，没有执行上游cache或模型。迁移必须用完整不可变内容/generation，而不是仅加长这个sample hash的摘要。

磁盘命中路径`_ModelCache.get`又只调用`ct.models.MLModel(disk_path)`，没有显式传回key中的compute policy。不能仅据文件名里的`CPU_AND_NE`给实际预测归因；加载选项与plan需另核对。所存目录还是`.mlpackage`，不是注释所称已编译`.mlmodelc`的直接证明。

这些是作者针对其历史OS/CoreML/MLX组合的规避，不是“所有Apple平台只能载入一个ANE图”的硬件规则。该分支的readiness记录后来也报告新系统上的repro不再崩溃；迁移时必须重测，而不是把旧restriction固化成产品设计。

调度器按完整shape维护锁和状态；禁用ANE时把剩余row交给CPU/GPU，确保整数row总量守恒。GPU执行完成后，readback延后到join后的主线程；wall计时覆盖readback和concat，避免把最终输出准备成本藏到门槛外。

但本轮读到的`_dispatch`仍使用raw threads，worker错误没有通过future/result传回；最后只拼接out中存在的分支。单纯join不足以证明成功或完整shape。移植时应显式收集每支成功/异常并校验generation与row coverage；这仍是静态合同检查，不是故障复现。

## 4. lazy graph 跨stream不等于实际重叠

作者的最小probe区分：预先ready输入、同一eval graph内GPU producer→CPU/GPU分支、显式`mx.eval(producer)`后再分支。所读probe使用相同FP16 shape并分别完成所有输出；它是定位framework依赖/提交行为的方法，而非把“异步构图”当物理并行。

该probe按固定case顺序测量，未取得硬件重叠trace，也没有覆盖所有MLX版本。只能据此提出目标栈上的可检验假设，不能从一组wall time宣布所有lazy framework都有同一个全局锁。真实迁移应加入交替顺序、独立输入、无边界分母、eager-boundary分母与设备trace。

论文还指出ANE逐层offload常受其CoreML调用开销限制。benchmarks的`ane_matmul`实际将B固化为FP16常量、输入整理为host数组，再predict并新建FP32输出；旧main的逐元素boxing是另一条路径，不能混作它的成本解释。未经分项计时，作者某个路径的几十毫秒不能直接标为ANE硬件dispatch常数。本库上一轮的小图/封闭FFN边界又不同，不把两组数字互相除来排名。

## 5. 同一个仓库内也有不同校准/执行路径

`TriComputeScheduler.calibrate`的grid在同一compiled MLX函数内构造CPU/GPU两支；`tri_matmul`则可走NumPy/MLX/CoreML的多线程host路径。前者的ratio不天然适用于后者：backend、输入驻留、dtype、分支复制和同步不同。

`tri_matmul`所读调用点又以min-dimension查询ratio，不能仅因`get_ratios`支持完整shape key便认为实际调用总使用它。应沿真正模型入口核对，避免“具备校准API”被误写成“所有生产执行均按同一边界校准”。

比较方案时同时保留isolated与concurrent profile，并允许设备share为0。权重带宽主导的decode不一定能因增加计算设备获益；长prefill、大矩阵、封闭子图是另外的工作区间。

## 6. fallback不保证完整请求永不回退

wrapper的前3次probation既执行naive又执行split，最后返回naive；观察到split不够快会锁到naive。即便最终选择与baseline相同，用户仍支付了probe。动态策略的完整成本应是：

```text
总耗时 = calibration/probation + 正常调用 + 周期probe + 探测后的恢复/重载
```

因此“有baseline fallback”不是零regret的数学保证。作者readiness后续也修正README的never-regress措辞，记录小cell的探测开销、内存压力下probe把活跃工作集挤出、以及不稳定/排除结果。不要只取最上方headline，也不能把下方早期未更新的限制当最新全项目结论；按具体日期、case和版本追溯。

ThrottleGuard用wall-latency drift与迟滞决定减小/关闭ANE。它检测的是**变慢**，不是直接观测温度；CPU争抢、内存压力或后台I/O也会触发。二进制ANE tier反映当时编译/cache限制，不能推广为硬件只支持开/关。

## 7. 可迁移设计

保留：真实消费者边界、eager依赖消融、concurrent profile、显式ANE预热/禁用、整数row守恒、probe成本和内存扰动、失败分支覆盖。修正后再迁移：完整权重身份和disk-load policy。拒绝直接继承：旧主线占位后端、类名当物理设备、zero-copy注释、跨入口校准、任意shape都不会回退的宣传。

下一步可在本机受控复现stream-ready/dependent/eager-boundary三路，再决定是否值得移植；先检查其他GPU工作，分开正式计时和诊断trace。本轮只做源码分析与独立CPU状态/数学见证，不将其混入上一轮的设备测量数量。
