# ggml-ane backend合同：缓存、stride与部分图失败

<!-- knowledge
id: technique-gpu-ane-case-studies-ggml-ane-backend-cache-and-failure
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [来源](../sources.md#ggml-ane-lifecycle) · CPU执行 · 正控审计 · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，固定 `arozanov/ggml-ane@fecc9c9ffc9fc60aa5cb569b8caadc000a07e0b0`。本轮将首轮的静态风险推进到**未改原始函数＋真实GGML scheduler/CPU join＋显式CPU bridge stand-in**。没有运行私有ANE bridge、GPU、模型或性能benchmark；下文负例均限定这个控制流/布局实验，不是ANE硬件错误复现。

## 结果与证据等级

| 问题 | 本轮实际证据 | 不授予的结论 |
| --- | --- | --- |
| 部分图失败是否自动回退 | 同一split第一项成功、第二项compile/eval失败，原graph函数仍返回0/SUCCESS；真实GGML继续CPU join，第二项及join各4096个NaN；全部失败时返回-1且join不执行 | 不等于原插件完整运行或私有ANE发生过该失败 |
| 常量缓存是否认识权重变化 | 原地把I改为3I，compile仍1次，before/after输出字节相同；after有3964个有限值不符、最大误差1.875 | 不是驱动compiled-artifact读回或真实LoRA/训练验证 |
| stride是否属于支持合同 | 合法padded weight/input view均被原supports_op接受，原compute按flat指针转换导致3966/3902个值错误 | 不推导所有view、dtype或模型都失败 |
| 维度上界是否一致 | M16385/K64/N64被supports接受；compute在attempt前跳过，status0、零compile/eval，1048640个目的值仍NaN | 这不是1048640个已执行预测值 |
| 原测试能否拒绝NaN | 调用三个未改原测试函数，bridge显式注入全FP16 NaN且eval=true；均报0 errors/PASSED，独立检查40960个F32值均NaN | 不证明作者曾在真实设备跑出NaN |

主实验有9条component观察、4个scheduler graph、3个原测试函数、21份目的数组，共1155136个F32值/4620544bytes完整检查；包含正确值、错误值、未计算poison和注入NaN，**不是21次成功预测**。独立strict-confirm进程的全部21份输出逐字节相同，单列为重复确认；两个早期无效正控cohort不混入主结果。

## 从GGML op到私有runtime：哪些成本真实存在

原backend只接部分2D `MUL_MAT`：GGML weight的`ne=[K,M,1,1]`、activation的`ne=[K,N,1,1]`，输出F32 `[M,N]`。supports要求M/K/N至少64、src1为F32或F16，decode的N1不在这条支持范围。

执行链是：

1. 从weight指针和M/K/N找kernel；F32转FP16，或经GGML量化type trait先转F32再FP16。
2. 生成128-byte header＋FP16 weights的blob，MIL把activation从`[1,N,1,K]`转成`[1,K,1,N]`，用常量权重Conv1×1，再转回GGML顺序。
3. 首次私有bridge compile/load后缓存kernel；每次CPU转换activation到FP16 vector、写IOSurface、eval、从IOSurface读FP16 vector，再CPU转F32输出。
4. 其它op交给GGML scheduler管理；backend本身不实现通用GPU补集或按row并行。

因此README的“no CPU involvement”不能作为完整路径事实：这里明确有CPU权重解码、精度转换、host调用和复制。CPU buffer type被支持，也不意味着GGML buffer直接变成了ANE IOSurface或有零拷贝。

原bridge源码使用 `_ANEInMemoryModelDescriptor/_ANEInMemoryModel/_ANERequest` 等私有入口，compile/load/eval的QoS为21，load失败有一次100ms重试；input/output copy按声明capacity截短，返回void。表面是IOSurface共享内存，仍须核实际capacity、pitch、锁与完成语义。该`.m`本轮只读，未编译、链接或执行；没有drain、取消、泄漏或私有API兼容性资格。

## 常量kernel缓存：地址身份不是内容身份

原key只有M/K/N＋`src0->data`。同形状不同地址的权重可区分，但同一地址原地更新、allocator地址复用、dtype/recipe变化都不自动改变key。MIL中的W是编译时常量，不是每次eval传入的runtime weight。

本轮先I→x，再在同一w对象/地址写3I→同一graph；实际保存了before/after物理weight storage、编译blob和桥接消费input。原函数第二次不compile，CPU stand-in仍持有首份blob，输出保持x，而独立正确答案应3x。该实验验证的是**原缓存把旧常量handle交给bridge的控制流**；未运行私有编译器，不能扩写成“本机ANE驱动缓存故障”。

该ref的上下文`compile_count`到4096后不再创建新key，没有普通eviction；backend free才遍历释放kernel。N也进入key，同一weight的不同N会形成独立编译资产。需要分别研究编译驻留、内容generation、动态权重与cache失效成本，不能只把首轮compile从warm计时删掉。

此固定commit的两文件patch改了cache double-check/duplicate free、FP16 denormal conversion、计算尺寸下限和MIL文本buffer。本轮完整读了官方API返回的patch文本；这些是**commit演进**，没有据此虚构PR号或最新状态。它解决了选定局部问题，不证明内容generation或全部graph失败合同已完备；本轮也没有并发race实测。

## supports_op、执行与完成必须同一套合法域

原supports_op没有16384上界，也没检查contiguous/nb；原compute却拒绝任意M/K/N>16384，输入/权重转换又按平面连续内存读取。

- 大尺寸例：合法GGML shape被宣称支持，但原compute跳过所有node，`n_attempted=0`，结尾仍SUCCESS。目的poison未写不是正确输出，更不应被称为fast path成功。
- stride例：K64、物理行宽128，logical view每行只用前64值，padding另填受控非零数据。正确结果仍是x；原flat读取把padding或别行当有效元素。两个例子都全finite，最大绝对误差1.875，有限性检查也无法替代正确性。
- dtype例：原compute只在node输出为F32时显式写回；正常GGML MUL_MAT的本轮输出确为F32。没有为手工改成F16的非法/非典型node制造“模型故障”结论。

移植时将dtype、stride、batch、上下限、buffer类型和真实可完成性一起放进capability合同。scheduler只依据declared support分配，不能指望它事后猜出某个被跳过的node。

## 部分失败：scheduler不是自动修复器

原graph循环对compile失败、input类型不支持或eval失败继续执行，只有`n_attempted>0 && n_computed==0`才返回FAILED。它没有要求每个assigned node都完成。

使用研究选择的固定GGML `64e9bceb…`，将两个MUL_MAT显式分给适配器、ADD分给真实CPU，且保留各中间结果：

| 故障注入 | 原graph/scheduler状态 | 第一项 | 第二项 | 真实CPU join |
| --- | --- | --- | --- | --- |
| all_ok | 0/SUCCESS | 逐值正确x | 逐值正确3x | 逐值正确4x |
| 第二项compile失败 | 0/SUCCESS | 正确x | 4096 NaN | 实际调用，4096 NaN |
| 第二项eval失败 | 0/SUCCESS | 正确x | 4096 NaN | 实际调用，4096 NaN |
| 两项eval均失败 | -1/FAILED | 未写poison | 未写poison | 未调用；目的仍poison |

选定GGML的`ggml_backend_sched_compute_splits`检查backend返回值；失败就返回该状态，成功就继续后续split，没有这里的“检测缺失输出并自动CPU重算”逻辑。本轮是强制placement的合同实验，不是自动route选择、并发调度、真实offload或设备性能证明。

可借鉴的设计要求是：assigned-node完成账本、全split成功的原子语义、清楚区分不支持/失败/成功、失败后由明确owner决定重算及发布。当前程序未修上游，不能宣称给出了已验证的runtime fallback实现。

## 原测试的“PASSED”不构成独立质量证据

原三个static测试忽略`ggml_backend_graph_compute`返回值，数值判定只用`fabsf(got-expected)>tolerance`，没有finite检查；scaled只检查前4/64个N行，cache测试只检查各自第一行。本轮不运行其动态加载main，只调用这三个原函数，并在own adapter外围保存完整F32输出。

CPU bridge注入FP16 NaN但返回eval=true后，原转换产生F32 NaN。NaN的`fabsf`比较不会为true，因此三个函数都返回0错误，四份完整输出共40960个NaN却被原测试判PASSED。独立checker明确拒绝；没有把原test文字、exit0或汇总数字当数值正确。

这不是修改测试、发起上游issue/PR或证明真实ANE曾返回NaN；它只展示验证器本身的一个具体盲点。测试oracle应同时核status、完整shape、所有值finite和受控误差，并能被预期坏结果证伪。

## 先否决自己的坏正控：allocator复用与poison

首轮收集器的诊断compile blob按`nan_success`命名，三个原测试之间有文件名碰撞；随后改唯一label，保留首轮binary/client和不完整blob记录，不把它归给上游。

更重要的是，独立逐值capture拒绝了前两次scheduler正控：all_ok本身全NaN。后续仅做allocation、不执行graph的诊断实测2个目的/leaf跨度重叠。GGML planner可以合法复用在**图执行顺序中已不再需要**的输入存储；本客户端在graph执行前poison所有未来输出，却破坏了尚待消费的输入。它是测试客户端错误，不是GGML allocator缺陷或ANE结果。

strict版本将诊断中需要保留的输入/输出都pin到结束，poison前拒绝任何跨度重叠；poison后保存完整native input/weights，另外保存每次bridge实际消费的FP16 input。strict与strict-confirm各4份重叠检查均0，all_ok三输出完整正确，故障负控才有归因资格。旧两cohort不删除、不重写、不再用于scheduler结论；正控audit记录额外source ranges与修正范围。

## 版本、构建与复现边界

仓库只有8个blob，没有AGENTS或GGML锁定依赖。官方归档全部核Git blob；研究选择的GGML固定ref从committed archive提取，1352个blob全核，不依赖dirty参考工作树。初始读本15文件，正控诊断另增加3个独立文件及既有文件的新范围；核字节不等于人读了1352文件。

原插件的positional `ggml_backend_i` initializer对所选GGML编译失败：新增2D async槽位让原graph_compute指针落到不同签名字段。保留terminal1，不修改原文件；own adapter用当前具名字段，调用exact source fragments，故本轮**没有完整插件构建资格**。API_VERSION2单值也不能替代精确header/ABI版本。

GGML原CMake在ignored快照里向上找到外层TurboCider Git，打印`ggml commit:451161f`；真正核心身份仍是固定64e9bce的1352个Git blob，不从banner猜版本。CPU-only、Metal/BLAS/Accelerate/OpenMP/native/all-variants均关闭，armv8-a；binary链接只见libSystem/libc++，原`ane_bridge.m`未编译。无需GPU-idle gate，因为没有提交GPU/ANE；未来设备工作仍必须先走本专题guard。

原始根 `outputs/gpu-ane-ggml-backend-round26-20261017` 保存官方API/归档、旧客户端/binary、编译拒绝、无效正控、alias-only诊断、两个strict进程、全量数组/input/blob、stdout/stderr/owned receipts和隔离结果。复现用own client、标准库合同、capture；原source fragments只在ignored目录生成，不向轻量Wiki复制上游实现，不安装系统依赖或修改产品后端。

## 固定一手来源与采用建议

- [完整小backend](https://github.com/arozanov/ggml-ane/blob/fecc9c9ffc9fc60aa5cb569b8caadc000a07e0b0/ggml/src/ggml-ane/ggml-ane.mm)：MIL/cache/graph_compute/supports/registration。
- [原bridge](https://github.com/arozanov/ggml-ane/blob/fecc9c9ffc9fc60aa5cb569b8caadc000a07e0b0/ggml/src/ggml-ane/ane_bridge.m)、[原测试](https://github.com/arozanov/ggml-ane/blob/fecc9c9ffc9fc60aa5cb569b8caadc000a07e0b0/test_ane_backend.c)、[固定修复commit](https://github.com/arozanov/ggml-ane/commit/fecc9c9ffc9fc60aa5cb569b8caadc000a07e0b0)。
- [研究所选GGML scheduler](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/ggml/src/ggml-backend.cpp)、[allocator](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/ggml/src/ggml-alloc.c)，不是插件上游锁定的依赖。

可迁移的是GGML backend接缝、常量图编译摊销和明确的能力/失败门；当前负例要求先补质量/完整完成资格，再讨论GPU补集、ANE驻留或更低功耗。README的3.5–4TFLOPS只保留为作者文案，不继承成本机数据或完整请求收益。无私有runtime、物理ANE/overlap、模型质量、内存安全、取消/drain或速度资格。
