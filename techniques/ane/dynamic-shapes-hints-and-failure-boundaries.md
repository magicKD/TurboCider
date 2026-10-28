# 动态shape与优化hint：默认形状、owner窗口和预测失败边界

<!-- knowledge
id: technique-ane-dynamic-shapes-hints-and-failure-boundaries
type: method
stage: pipeline, ffn, decode
operator: conv, scheduling
backend: coreml, cpu-reference
device: cpu, ane
symptom: compile-bound, measurement, graph-interruption
semantics: numerically-equivalent, contract-only
representation: fp16, fp32
boundary: request, producer-consumer
api: public, reference-only
-->

[ANE目录](README.md) · [固定源码/SDK](sources.md#shape-specialization-sources-20261017) · [验证（过程原件已移出）](../MAINTENANCE.md) · [前批工具纵切](coreml-experimental-profiling-and-inspection.md)

2026-10-17，固定coremltools8.3 `07ffd86005e385d8eddfd22072836cfb457f7350`、M4 Max / macOS26.6.2（25G83）、SDK26.5。四轮96个独立原生单元全部终止：固定/范围shape完成48单元，枚举shape的48单元均在**首次预测**SIGTRAP，没有完整输出或可比较的成功时间。

返回的3,240份完整最终数组（22,953,984元素）全部通过独立FP64参考；另保留120个shape拒绝、早期原型的402份完整输出及独立Python客户端的两次成功/两次同信号失败。**失败没有替换为范围shape成功、被跳过或计成0ms。** 所有成功单元的已知op计算计划均首选CPU；因此本批并未取得实际首选ANE的动态shape案例，更没有物理ANE或加速资格。

## 可复用的方法：先分清 shape 合同，再调 hint

1. 将允许形状集合、默认形状和优化 hint 分开建表；记录导出元数据及当前 API 的实际枚举值。
2. 先用固定 bucket 完成完整消费者控制，再单因素比较枚举、范围和 hint；覆盖切换顺序与尾长度。
3. admission、首次 prediction、输出质量和崩溃分别记录；进程隔离帮助保留 terminal 失败，不能把重试成功当首次成功。

适用于部署形状策略与失败缩减；下文局部结果不证明动态 shape 或某个 hint 普遍更快。

## 1. 三件不同的事：接受哪些shape、默认哪个、如何优化切换

| 机制 | 声明的合同 | 不能由此推断 |
| --- | --- | --- |
| 固定shape | 本模型只接收一个完整shape | 多个shape共用一个model owner、没有多份load成本 |
| EnumeratedShapes | 从有限完整shape列表中选；default用于初始化 | default就是唯一合法shape、列表每项均已实测或每个版本都更快 |
| RangeDim | 某维在有界区间中变化 | 每个尺寸都可用同一物理kernel或无需重优化 |
| reshapeFrequency | 对切换频率的预期；框架据此优化 | 强制ANE、保证快或改变feature合法集合 |
| specializationStrategy | 倾向specialization与prediction之间的取舍 | 低load、低峰值内存、低磁盘和低预测时间同时成立 |

固定8.3的[Flexible Input Shapes指南](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/docs-guides/source/flexible-inputs.md)推荐枚举shape，并描述default预分配、非default首调可能较慢。它是优化指导，不是当前任意导出图的可靠性/性能验收。本批枚举失败后，**无法给出其default与非default的成功时间对比**。

[ANEMLL原embedding转换](https://github.com/Anemll/Anemll/blob/f4ad26d061dc2e426faa19f81c367343bdeb9f0d/anemll/ane_converter/llama_converter.py#L553)就选`[1,1]`与`[1,batch_size]`两种完整token shape，以单token为default，再请求CPU_AND_NE。这是把decode和批量入口收敛到有限接口的实际设计；不是本批执行了其模型，也不能把这里的ReLU FFN失败套给其embedding、KV或完整LLM。

本批原Python constructor控制确认：单项枚举被拒绝、RangeDim的default越上下界被拒绝；129项枚举constructor却被接受。后者只说明**constructor这一层未拒绝**，没有转换/编译/执行129项模型，不能据此推翻指南的128形状建议或宣布runtime无上限。ML program的无界RangeDim另在converter中有检查，不能把构造对象成功当成功导出。

## 2. Python枚举不是native枚举的整数ABI

[8.3输入校验](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/models/model.py#L119)对**任何非空hint字典**统一要求macOS≥15.0。SDK26.5的头文件则区分：reshape提示从macOS14.4可用，specialization策略从15.0可用。不能由SDK可用性推断这个Python版本也接受同样的调用。

原函数的16个version/type fixture没有调用设备：14.4上None/空dict通过、非空hint先在版本处拒绝；15.0上合法enum通过，string、错enum类别、未知key和非dict按各自类型拒绝。

| 含义 | Python enum `.value` | SDK native enum |
| --- | ---: | ---: |
| Frequent | 1 | 0 |
| Infrequent | 2 | 1 |
| Default strategy | 1 | 0 |
| FastPrediction | 2 | 1 |

原Python wrapper传enum的**名字**，native桥再映射，不是直接转`.value`。自行写C/Swift/ObjC配置桥时把整数照搬，会改变或破坏请求。当前原生客户端使用SDK声明的named constants，另保存model报告的配置，不仅记录“调用方想设置什么”。

SDK说明Frequent偏向降低切换成本，Infrequent允许在切换时花时间优化后续同shape预测；FastPrediction可能增加specialization时间、内存和磁盘成本。[固定预测指南](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/docs-guides/source/model-prediction.md)还区分package compile、device specialization与persistent compiled路径缓存。本批只保留load/prediction的framework-call窗口和本次静态文件inventory，没有测RSS/系统缓存大小或宣称“FastPrediction无额外内存成本”。

## 3. 同一个完整消费者，四种shape协议与六种配置

```text
FP16 X [1,64,1,S]
  → dense 1×1 Conv 64→128 + bias
  → ReLU
  → dense 1×1 Conv 128→64 + bias
  → residual + X
  → 完整FP16 Y [1,64,1,S]
```

两seed17/29；矩阵全稠密、非identity且非对称，使用明确dyadic符号/偏置；不是预训练权重、语料或真实任务分布。输入各列有31种可重算相位，generation逐调用变化。独立参考对每种相位执行完整64→128→64数学，再铺回**全部列**；不是只采31个输出值或只检查一列。

每seed导出六份持久compiled资产：fixed16/fixed64/fixed256、enum64-default、enum16-default、range64-default。枚举完整集合均为S∈{16,64,256}，范围为[16,256]；所有MIL中非const操作均是conv→relu→conv→add。12个package中的48份完整FP16 weight/bias payload由标准库独立解析BLOBFILE、与参数配方匹配，不只信任requested dtype或源码banner。

研究的四个family是固定三模型bundle、enum64、enum16和range64。固定bundle为每个S持有一个owner；其他family仅一个owner。每family分别运行：CPU_ONLY省略hint，以及CPU_AND_NE下省略、显式Frequent、Infrequent、Fast+Frequent、Fast+Infrequent，共24单元。四轮改变seed与family/arm/visit正反顺序，**每单元另起进程**，故一个trap不会把剩余单元从账本抹掉。

正常访问S序列为64→16→256→64→256→16；反序轮颠倒访问。每次访问11次，首调用与后10次分开；canonical generation绑定visit/repeat，不因调换执行顺序换输入。之后在同owner尝试S8、257、32及恢复S64。枚举单元首次预测即trap，后续控制确实未执行，不能给它补写成功的拒绝/恢复记录。

## 4. 首次预测trap的证据与尚未证明的根因

最初把所有family放在一个进程：固定组六种配置已返回402份完整数组及18个正常shape拒绝，随后SIGTRAP，整轮不完整。保留其原二进制、源码、原journal和402数组独立审计，**没有把当时缺少stage breadcrumb的运行改写成已经定位了失败阶段**。

新版自有客户端在load、每次predict、control和plan前后记journal。四轮枚举单元的末事件均为首次`predict-begin`；load已有`load-end`，没有`predict-end`、没有plan查询，也没有返回值。正序首次S64，反序首次S16，两种default模型与所有hint均失败。监督器48份receipt均为signal退出-5，且确认owned进程终止；不是观察超时或失去句柄后盲重启。

按**本次owned receipt PID**匹配的系统crash报告：51个trap进程（含原型及Python对照）中采到28份，原始文件留ignored目录，只发布hash、signal和不含地址的frame image/symbol。所采样本有EXC_BREAKPOINT/SIGTRAP及BNNS执行栈；未采到的23份仍为unavailable，不称“没有崩溃”或假定相同完整stack。

独立现有coremltools8.3 `CompiledMLModel`侧客户端在同compiled资产、CPU_ONLY、S64上：enum64/enum16也SIGTRAP；fixed64/range64完整输出通过。原Python bridge会把NumPy FP16 input转FP32；两个成功对照的caller字典确实已变成float32，不能把其输入物理格式称为原生FP16路径。失败进程没有返回，转码后的实际输入没有额外capture，源码预期和现场见证分开。

这些对照削弱“只是自有C++数组索引/输出读取”的解释：trap发生在返回前，另一标准wrapper也复现。但**没有完成组件缩减、跨OS/新converter、ISA或系统源码分析**，不能认定具体哪个BNNS算子或8.3 exporter步骤是根因，不把SIGTRAP称为ANE硬件错误，也不宣传枚举shape在所有模型/版本都不可用。

## 5. 成功结果、正常拒绝与数值gate分别验收

| 主试结果 | 实际数量 | 数值/状态解释 |
| --- | ---: | --- |
| 固定/范围单元成功终止 | 48 | 不替代枚举单元 |
| 正常timed输出 | 3,168 | 包括首调，不用“warmup”藏掉 |
| 错shape后的恢复S64输出 | 48 | stateless owner的有界恢复控制，不是state atomicity |
| range合法但不在枚举菜单的S32 | 24 | 完整输出通过 |
| 最终完整数组合计 | 3,240 | 22,953,984元素，全通过原定逐元素门槛 |
| 正常shape拒绝 | 120 | 固定72，范围48；原NSError消息完整保留 |
| 枚举首次预测trap | 48 | 无输出/成功时间/plan，比较资格null |

固定的逐元素门槛为`abs(error) ≤ .002 + .01*abs(independent_F64_reference)`，没有放宽或用actual-feed参考替换请求目标。最大绝对误差/relative L2也保留，但不以一个整体指标替代逐元素gate。完整输出只有112份唯一bytes payload，lossless zlib去重；所有重复实际文件仍逐份核对，不把相同hash当少做了调用。

额外逐字节对照：每个CPU+NE配置与对应CPU_ONLY有528个timed数组对，五种配置均0差异；固定/range有1,584对，正反序有1,584对，也均0差异。这些重复对不是新的独立样本或物理放置证明，不能把同一baseline在多种比较中出现的次数当统计样本量。

120份NSError中，fixed消息描述与固定shape不匹配，range消息是dimension 3的Size不在16..256。首个自有curator误要求两者文本均含“shape”而失败；之后改为核对两种**完整原消息**，保留失败receipt，无设备重采、无数值门槛调整。文案/validator错误不能伪装成上游数值修复。

## 6. 计算计划同hint，不等于物理ANE；fresh owner不等于系统冷

成功owner完成所有预测与控制后，才请求其compute plan；将**同一MLModelConfiguration**（含reshape/specialization）传给plan。前批已知Python8.3 plan helper仅接受path/compute units，不能用它的默认配置计划解释另一hint的load。本批保存请求配置、model报告配置与plan查询配置，逐项核对。

96份成功模型plan的384个已知op均首选CPU。enum失败owner没有plan结果，不能用其余case的CPU计划替代它的放置。计划本身仍只是anticipated preference，不是CPU/ANE的物理dispatch证明；`ios18.conv`只是op命名，不代表进程运行在iOS18。

记录288份visit级诊断：首个framework-call interval、接下来10次median及是否本owner见过该shape。timer在输入准备/前置journal之后开始，返回即停止，输出snapshot/后置journal在窗口外；但这仍是带诊断IO的运行，窗口之间的状态已被扰动，**不提供无observer的生产性能排名、GPU对比或hint提速结论**。

读数示例：对四轮每个family/arm的24个visit，先各取后10次median，再取24份median的中位；fixed的省略hint约24.41µs，Fast+Infrequent约25.67µs，range相应约41.14/42.31µs。这是可回算的**诊断窗口汇总**，不是“Fast必然更快”的证据；两个hint同时改变、访问/缓存顺序及IO扰动都不支持把差值单独归因于specialization策略。无observer、单因素配对仍未运行。

固定bundle有三次load而range只有一次，不能只比较其hot predict就宣布完整请求胜负。持久compiled路径在不同hint/新进程中复用、早期pilot与plan也可能影响后续缓存；没有清系统缓存。所以“第一次在此owner见到S”不等于driver/OS冷启动，更不能把不同进程的最低值拼成速度比。

## 7. 本地实现、验证与继续入口

完整数学/输入/codec、、、测试。CPU命令不import框架、不启动设备、不读取系统crash目录：

显式设备代码：导出/prepare、原生公共API客户端、完整逐单元监督、独立Python对照、仅owned PID crash归档。只写新ignored目录，最低2GiB reserve；不删模型或缓存、不安装依赖、不改reference/安全设置/功耗策略或产品默认。不建议直接对任意用户模型套用此trap研究矩阵。

下一轮需要：枚举图的受控组件/前端/OS版本缩减（原失败仍保留）、真正计划首选ANE的较大动态消费者、无observer且同正确性gate的load/切换/稳态配对，以及RSS/磁盘与真实KV/训练请求。所有源码/wheel/native/SDK/OS、权重/shape/dtype/default、hint、owner/访问协议或消费者变动均重开。当前结果并未完成这些任务，整体ANE研究继续。
