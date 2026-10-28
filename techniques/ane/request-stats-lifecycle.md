# 从 constructor 到真实请求：stats 双槽、seed 残留与版本门槛反例

<!-- knowledge
id: technique-ane-request-stats-lifecycle
type: method
operator: conv, scheduling, layout
backend: private-ane, cpu-reference
device: cpu, ane
symptom: measurement, lifetime
semantics: contract-only
representation: fp16, iosurface
boundary: request
api: private, reference-only
-->

[ANE目录](README.md) · [前批constructor诊断](profiling-constructor-provenance.md) · [固定源码范围](sources.md#request-stats-sources-20261016)

2026-10-16，M4 Max / macOS26.6.2（25G83）/ SDK26.5 / Apple clang21。本批不是再做一次无模型的constructor调用：**完成99次自有affine/conv请求的完整数值见证，但仍未取得有效counter或硬件时间**。另复现两次“raw stats对象被当作array carrier”的validation异常，均发生在evaluate之前。

新增的可靠结论是对象/slot/执行代际合同，不是PMU能力、物理ANE placement、完整模型速度或能耗资格。这里的mask0/15是**performance-stats mask，不是ANEForge的compute-device mask0x4**。

## 1. ABI正确，不等于对象schema正确

先单独获取本机class/selector元数据，再对实际调用逐参数检查。相关ABI包括：

- `statsWithHardwareExecutionNS:`接受uint64 `Q`，不是NSNumber指针。
- `statsWithRequestPerformanceBuffer:statsBufferSize:`接受`^^v`和`^I`，只使用已读的`buffer=nullptr,size=0`配方。
- 带`weightsBuffer:perfStats:`的request constructor有七个显式对象参数，实际编码`@72@0:8@16@24@32@40@48@56@64`。
- `perfStats`、`perfStatsArray`都是对象getter；`setPerfStats:`参数也只是`@`。这些编码本身不能表达“必须是数组、是哪种元素、是输入carrier还是解析后的结果”。

本机直接观察到两个不同slot：

| 自有调用方式 | 执行前getter | validation / 后续 |
| --- | --- | --- |
| constructor的`perfStats:`传nil | 两个getter都nil | 通过，能执行 |
| constructor的`perfStats:`直接传seeded `_ANEPerformanceStats` | `perfStatsArray`返回这个对象，`is_array=false`；`perfStats`为nil | validation向该对象发送`count`并抛异常；未evaluate |
| constructor传nil，随后`setPerfStats:seededObject` | `perfStats`返回seeded对象，`perfStatsArray`为nil | validation通过；执行后的结果取决于profile选项包，见下一节 |

这不是建议向array里塞未审计的raw object，也没有猜IOSurface底层指针。第三行是单独命名的原创诊断，不是第二行失败后在同一run静默fallback。`seed0`异常另有新进程确认。

## 2. 数值成功，getter仍可以只是调用方数字

使用两个已审核的合成MIL：affine `[1,32,1,32]`的`y=2x+.25`，以及dense1×1 conv `[1,128,1,512]`。输入随generation变化；conv使用固定dyadic权重。每个成功进程9次请求（含first/3 warmup），输出poison后按实际physical pitch取回并验证全部逻辑元素。

profiled配置同时设置model stats mask、`kANEFPerformanceStatsMask=15`、`enableProfiling=true`、`kANEFModelLoadPerformanceStats=true`；unprofiled配置使用mask0且不添加后二项。**本轮没有隔离这些选项各自的因果贡献**，不能声称某一bit单独清除了某个ivar。

| recipe | mask | 进程/完成evaluate | 执行后实际getter观察 |
| --- | --- | --- | --- |
| nil，无profile | 0 | 1 / 9 | 两个request getter均nil |
| nil，profiled | 15 | 2 / 18，affine+conv | 两个getter均nil |
| 零buffer factory，profiled | 15 | 2 / 18，affine+conv | factory每次nil，两个getter仍nil |
| raw seeded object直接传constructor参数 | 15 | 2 / 0，含新进程确认 | validation异常，不能填造执行后stats |
| `setPerfStats:`，seed0 | 0 | 1 / 9 | request getter保留原对象及0；无payload |
| `setPerfStats:`，varying sentinel | 0 | 1 / 9 | request getter保留123456…123464；无payload |
| `setPerfStats:`，seed0 | 15 | 1 / 9 | request getter变nil；held对象仍返回0 |
| `setPerfStats:`，varying sentinel | 15 | 3 / 27，affine、conv及新进程确认 | request getter变nil；held对象仍返回各次seed |

所有成功请求的request `perfStatsArray`执行后也为nil。未发现任何payload，所有`physical_duration_seconds`保持null。这里“getter变nil”是API观察，不是直接查看ivar、驱动内部状态或硬件trace。

### 2.1 身份和生命周期为什么重要

每次request都新建，seeded对象以进程内不含地址的`o0,o1…`标记，强引用保留以避免对象地址复用干扰身份判断。每次执行前后同时记录：

1. 客户端持有的constructor对象；
2. request的`perfStats`；
3. request的`perfStatsArray`及是否真是NSArray。

在54个parsed-slot请求中，held对象的seed一直不变；其中18个unprofiled请求的request getter执行后仍保留同一seed，36个profiled请求的request getter执行后为nil。由此得到两个明确负例：

- **inference成功 + stats对象存在 + 非零且随generation变化的getter**，仍可能只是调用方逐次预填的数字。
- **request getter执行后nil，转而读取旧held对象**，可能只读回旧seed；这不是有效capture。对象仍然存活，所以是语义/来源陈旧，不是一次UAF复现。

值为0也必须同样审查：不能把默认0、caller-seeded0或缺失nil叫作“硬件0ns”。没有进行单位标定，getter名字包含Execution/NS也不能代替来源和生命周期证据。

## 3. 版本门槛：本机26也有`count`异常

固定[PMetal runtime](https://github.com/Epistates/pmetal/blob/a225c0e2568f882573b05caeb42d94fed208fdd3/crates/pmetal-metal/src/ane/runtime.rs#L989)的compatibility gate在macOS27+停用stats，并在注释中描述单对象引发`count`的request-validation问题。本批正确uint64构造、经现场ABI检查的自有request，在**macOS26.6.2/build25G83**也复现相同selector类别的异常，两次均`evaluate_calls=0`且unload成功。

这提供了“只按major OS门槛不充分”的本机反例，**不是运行过PMetal的Rust代码，也不证明其全部旧OS都失败**。上游另有NSNumber-reference与本机Q参数的潜在ABI问题，仍只作源码风险，绝没有执行错误类型调用。这里观察的是本机正确类型对象经过自有构造/validate的行为，不把它冒充原项目完整故障复现。

本轮更合适的版本合同应至少包含OS build、实际ABI、carrier对象schema、slot角色、request构造方法、model/load配置与成功route；把“class存在”作为availability的充分条件也不够。

已在[版本复查登记](../guides/knowledge-base.md)绑定本批source/evidence SHA与本机context，7天后或提供的context/content改变时提示复查；缺context、精确匹配或复查工具通过都不授予运行时兼容/硬件资格。

## 4. Espresso：held fallback只能作候选，不能自证测量

固定[request builder](https://github.com/christopherkarani/Espresso/blob/05ea6a3da91be642abd3d4741c0341a941ce1064/Sources/ANEInterop/ane_interop.m#L505)在perf对象存在时走带`perfStats:`的constructor；[eval readout](https://github.com/christopherkarani/Espresso/blob/05ea6a3da91be642abd3d4741c0341a941ce1064/Sources/ANEInterop/ane_interop.m#L2390)依次查request stats、array元素、held对象。后者是一个候选来源选择策略，不自动证明最后选出的值在本次execution中更新过。

本批的setter路径说明这类fallback为何需要provenance；**没有复现Espresso完整引擎读到伪时间的bug**。本机所试零buffer factory实际为nil，且本库setter诊断与该项目constructor路径不同。这里提炼的是条件风险/可观测性合同，不给未跑项目挂上错误或速度标签。

所有本批evaluate均显式调用同一个`_ANEClient.evaluateWithModel:options:request:qos:error:`，不尝试其他route、不做client→inmem/realtime静默fallback。记录的route只说明客户端调用入口，不证明其内部物理调度。

## 5. 独立完整见证、保留失败和源码身份

调用完整FP16数学witness，按固定公式、`math.fsum`与FP16编码独立重建每次完整逻辑输入/输出SHA。99份输出共**1,843,200个元素**，均与对应generation吻合、最大误差0；不是只接受native的`numeric_pass`布尔值。输入X有17列周期，不冒充真实激活，也不据conv的源数学MAC数估算硬件TOPS。

检查16个完整进程记录：11个成功推理进程、2个metadata-only、2个validate异常、1个编译失败；源snapshot、MIL/weight blob、binary、native JSON和supervisor receipt均有hash。新进程确认不覆盖或替换原进程记录。

最早执行版因Objective-C++ ARC的strong/autoreleasing引用不匹配而编译失败，保留为**编译前置失败，不算一次失败推理**。初版metadata筛选没纳入`counter`方法名，后来筛选读到`perfCounterData`；旧JSON缺项不能解释成selector不存在。两个问题都保留旧源身份，修正后另用新目录。

无模型下载、安装、签名/entitlement变化、系统cache清除、安全/电源/fence修改或全局日志订阅，未直接调用kernel IOCTL。原始stderr可能包含系统地址，仅留本地ignored目录；公开证据不含raw pointer、counter buffer bytes、模型或binary。原生客户端只include本库原创旧probe来复用布局/hash工具，重命名的旧main不执行；其四个源依赖都进入每run身份清单，不分发第三方profiler实现。

## 6. 复跑与仍未取得的证据

只跑CPU合同和保存数据的验证：

可选私有客户端默认metadata-only，需要显式opt-in；实际请求还要`--execute`，输出目录必须不存在：

`seed0`是已知本机validation负例，不是可用于采集的推荐参数；`seed0-slot`/`sentinel-slot`也只是显式provenance诊断，**不是修复PMU的方案**。换OS/ABI/route/configuration必须重审；当前verifier按已观测配方fail-closed，不会把未来未知payload自动认证为有效counter。

仍未取得：有效的request-associated capture、已标定时间/事件单位、PMU schema/累积与reset/overflow合同、物理placement/占用/并发和能量。公共部署继续用计算计划、[官方Core ML Instruments入口](https://developer.apple.com/videos/play/wwdc2023/10049/)和完整wall-time消费者对照诊断；不为填满表格降低系统保护或把seed变成测量。
