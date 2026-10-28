# Core ML实验性profiling与inspection：时间来源、状态和诊断覆盖

<!-- knowledge
id: technique-ane-coreml-experimental-profiling-and-inspection
type: method
stage: pipeline, decode
operator: scheduling
backend: coreml, cpu-reference
device: cpu, ane
symptom: measurement, lifetime, compile-bound
semantics: bitwise, contract-only
representation: fp16, fp32
boundary: request, producer-consumer
api: public, reference-only
-->

[ANE目录](README.md) · [固定10文件来源](sources.md#experimental-tools-sources-20261017) · [验证记录（过程原件已移出）](../MAINTENANCE.md) · [profiling总览](profiling.md)

2026-10-17，固定coremltools **8.3 / `07ffd86005e385d8eddfd22072836cfb457f7350`**。四个新进程在M4 Max / macOS26.6.2（25G83）、Python3.11.16 / NumPy2.0.2中执行原`MLModelBenchmarker`、`MLModelInspector`与local async wrapper，156次成功预测、188份完整数组/6,016元素全部通过独立标准库精确参考。每seed的两种许可有39对完整结果一致，共78对。

本批研究的是**公开框架的实验性诊断入口**，不是私有PMU或真实模型速度比较。五个有cost的operation在两种许可下的计划均首选CPU；`CPU_AND_NE`许可和本机拥有ANE不构成物理ANE执行证据。没有更改依赖、模型下载、remote设备会话、上游文件或系统缓存；没有检查其他版本是否修复这些行为，不能称为所有/最新版本结论。

## 可复用的方法：让诊断工具成为可核查的助手

1. 给测量、静态估计和派生字段分别标记来源；检查单位、重复输出和无法取得的指标。
2. stateful warmup 使用与正式请求匹配的 state 协议；模型 warm 不等于请求状态已就绪。
3. inspector 导出的中间输出图是新模型；先比较主输出，另用完整非空样本验证 validator 真正做过检查。

适合搭建 profiling 工具链和诊断回归，不能把工具成功返回当成加速或逐算子归因。

## 1. 先辨认每个数字来自哪里

| 入口/字段 | 固定实现真正取值的位置 | 不应升级为 |
| --- | --- | --- |
| `benchmark_load` | wrapper底层model的native load-duration字段，转成ms | 完整Python构造、package创建、编译或冷启动成本 |
| `benchmark_predict` | 每次成功预测后的native predict-duration字段 | 输入准备、Python桥接完整wall、设备kernel时间 |
| `benchmark_operation_execution` | 同一prediction样本×plan的estimated cost weight | 逐op实测、物理设备dispatch或timeline |
| `Measurement.add_samples` | 两份同长度样本逐位相加，再重算统计 | 追加独立样本、有效并行时间或互斥op分区 |
| inspector输出 | 克隆spec、额外暴露输出、重新load后预测 | 原未改图内部无扰动trace |

[perf_utils原实现](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/models/ml_program/experimental/perf_utils.py)的op结果是估计；本批所有实际返回的估计样本均由保存的同次native interval与cost重新算出，16个无cost的constant路径保持空samples/`statistics=null`，不是零耗时。

[native bridge](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremlpython/CoreMLPython.mm#L358)使用mach时钟：普通路径先compile，再围住`modelWithContentsOfURL`记录load；predict先构造feature provider，再围住Core ML prediction方法，结束后才转Python输出。源码scope比简化的Python docstring更精确：不应把这段native framework call命名为“ANE执行时间”。本批未编译此native源码；wheel中的native文件SHA只是文件inventory，不是源码可重现构建、完整加载镜像或固件证明。

load原docstring描述逐轮unload，但函数实际保留第一份wrapper在`_loaded_model`，后续迭代才unload临时对象。四进程均记录两次原load并确认第一份被保留。即便有`gc.collect()`，也没有清OS/compiler/cache，更不能叫两次独立cold load。collector另存outer load wall，但这些带输出保留的诊断不发布性能排名。

### 多输出会让estimated sum超过整次prediction

[model_structure_path](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/models/ml_program/experimental/model_structure_path.py)对**每个operation output**创建path，而不是仅给每个op一个path。本批用真实protobuf/原path mapper/原benchmark函数加显式plan double验证：

```text
prediction samples = [2, 4] ms
split cost = .6, 两个输出left/right → 各[1.2, 2.4]
relu  cost = .4                  → [.8, 1.6]
unknown cost = unavailable       → [] / null
三份有值path逐位求和             → [3.2, 6.4]，不是[2, 4]
```

这是**helper fixture的真实运行**，不是本机split成本实测、硬件counter溢出或真实timeline重叠。当前实测小图的有值operation恰好单输出、weights和约为1，也不能据此替任意多输出/控制流模型保证可加性。去重需要保留原operation身份、映射覆盖和估计性质，不能只按相同cost数值去重。

## 2. warmup准备了model，却没有准备连续请求的state

[原预测循环](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/models/ml_program/experimental/perf_utils.py#L305)每次调用前都`make_state_if_needed()`。`iterations=3,warmup=True`实际做4次预测，只排除首个timing，**不复用warmup state**。

自有state模型为32元素FP16 accumulator：`acc ← acc + X`，返回FP32的完整acc。输入是可精确表示的dyadic值，不使用近似容差：

| 协议 | 实际完整输出序列 | state owner |
| --- | --- | --- |
| 普通MLModel连续4次 | X, 2X, 3X, 4X | 同一state |
| 原bench，3samples+warmup | X, X, X, X | 四个新state |
| 同一原wrapper显式连续3次 | X, 2X, 3X | 同一state |

这说明benchmark路径测的是fresh-state预测，不是同一KV cache持续增长、同一训练state或autoregressive decode。它没有测“state复用导致更快/更慢”。完整stateful inference语义与恢复另见[MLState研究](stateful-multifunction-and-retry.md)。

另执行原循环的CPU doubles：正常无warmup3次、warmup4次、第二次预测抛错、首次timer缺失。后两者原样抛出，分别停在2次和1次，不返回只含成功前缀的平均。不能把[其他benchmark吞错误](profiling.md)的行为套给这个helper；也不能把这些注入异常称为实际设备fault recovery。

## 3. inspector运行的是额外导出接口的新模型

[MLModelInspector](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/models/ml_program/experimental/debugging_utils.py#L154)复制spec，在block与description上增加支持的tensor输出，再通过async wrapper重新load；原final outputs仍保留。它不是读取原model某块内部buffer。

本批stateless完整图为`X → half → a=2X → b=a+.125 → squared=b*b → float32 Y`，shape `[1,4,2,4]`，EMPTY converter pipeline保留可指定的端口。每进程分别用batch1与batch3请求`a,b,y`，从**每次原预测返回**保存所有数组，不只保存API迭代器选出的tap。另有普通MLModel前后各4次无wrapper预测作控制；所有final Y及tap都精确符合独立数学。

这是本dyadic小图的保值正控制，不构成任意图/精度/tap无扰动保证；[前批inverse端口变化](division-variance-and-observer-boundaries.md)已经说明新输出接口可以改变编译结果。正式性能应回到未诊断图重测，当前全量复制/记录的运行不作benchmark。

### 四个容易被忽略的诊断边界

- **相同请求不命中cache**：key为requested names的frozenset，原lookup却用集合成员关系检查。两次请求`['a']`均新建wrapper，`same_wrapper=false`，字典仍只有1项；不能把这个长度叫“只load一次”。不据此断言底层compiler一定重新做了全部工作。
- **忽略constant可返回None**：`ignore_const_ops=True`下，本批合法constant请求保留key但value为None。unknown不能当数值0或已通过验证。
- **batch数缺少完整校验**：0在`range`处抛ValueError；-1对合法请求静默返回`{}`。调用成功不代表请求覆盖完整。
- **stateful inspection没有state**：inspect调用wrapper predict时不提供MLState。四进程真实尝试均为RuntimeError，提示acc需要MLState；这不是返回空tensor、模型数值不稳或整个Core ML不支持state。

编译/创建失败又是另一类：原`_generate_models_with_outputs`捕获大部分构造异常，随机分割输出集合重试，单元素仍失败后跳过；最终`inspect`可yield None。本批两端口fixture记录原始合并及两个单端口共3次构造拒绝、结果`{'a':None,'b':None}`；没有制造无效实际模型或掩盖错误日志。

## 4. validator返回空列表，不一定发生了数值验证

原validator使用PASS/FAIL/UNKNOWN内部状态；未取得值为UNKNOWN。对**没有op依赖的合成relu protobuf**，使用原`find_failing_ops`及明确的retrieval doubles：

| retrieval | 数值callback次数 | returned failing ops |
| --- | ---: | --- |
| None | 0 | [] |
| 有限[1] | 1 | [] |
| [inf] | 1 | [relu] |

两个`[]`证据完全不同。外层应额外验收requested/returned/missing、数值callback覆盖及构造失败，而不是以`len(failing_ops)==0`授予模型精度资格。这个fixture执行原遍历，不是实际inf设备图。

Comparator源码另有按输出名与cast前缀寻找对应、跳过某些op、沿依赖找候选failure source的规则；名字相近不自动建立两份模型的SSA/语义等价。本批**未运行完整comparator或Torch wrapper**；不能把validator fixture当作完整跨模型定位验证。

## 5. local async不等于异步执行或安全文件所有权

[LocalMLModelAsyncWrapper](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/models/ml_program/experimental/async_wrapper.py#L226)的loaded predict在async函数体中直接调用同步`MLModel.predict`。原wrapper上gather两次完整预测和一个只yield一次的ticker，四进程均得到：

```text
A:start → A:end → B:start → B:end → ticker → ticker:resumed
```

这是同一Python event loop未在这两次loaded预测中yield的证据，不是GIL释放实验、所有线程不能并行的证明，更不是物理CPU/ANE无重叠。remote路径另有真正的await/service协议，本批没有连接或构建它，不能沿用local结论。

local load把底层`package_path`记为`temp_asset_path`；cleanup按该路径递归删除，没有在此层区分“自己生成的临时包”和“caller提供的包”。`unload`本身仅清掉model引用。本批仅将**新建的自有disposable.mlpackage副本**交给原path wrapper，验证loaded存在、unload后仍存在、显式cleanup后不存在；每进程只删除这一份新副本，可由保留的stateless包精确再生成。没有对已有reference或用户模型运行这个配方。持有path wrapper前应核查ownership，而不是假定async/temporary字段名足以保证安全。

原collector开头“output目录内”的位置注释过窄，特此勘误：`from_spec`生成的全新包还会走coremltools自己的runtime临时目录，并由原wrapper清理；本批对**caller传入路径**的显式删除测试才严格限于新output下的disposable副本。两者均为本次新生成资产，不是existing/user包。本批保留实际执行的原collector SHA，不回写成历史运行已带后来更正的注释。

另有一个只读发现未运行：local compute-plan的无compiled-path fallback把字符串当Path调用resolve，并混用了compiled/compile变量。本批正常compiled-path均成功，**没有触发这个fallback**，不能称为实测路径错误或完整容错验证。optimization hints能传给load，但读取plan仅按其接口传path/compute units；本批没有变更hint或测试其对计划/性能的影响。

## 6. 输入生成与上游tests也要检查真实边界

原random multiarray helper查到声明dtype后没有用于cast：本机FLOAT16/FLOAT32/DOUBLE三例均生成float64，INT32生成int64，完整2×3数据另存。image feature dispatcher比较的是type对象而非已取出的feature-type字符串，本批2×2 RGB description在调用image生成器前即ValueError。正式实验明确提供typed、确定性、满足索引/范围约束的输入；不要把“自动随机输入”当对任意模型有效的输入域。

[原test_perf_utils.py](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/test/ml_program/experimental/test_perf_utils.py)前三个测试虽枚举compute_units，所读函数体并未把该参数传给model/benchmarker。这是source-only测试覆盖提醒，不是本批运行上游pytest的结果；本批四个正式进程都在实际构造处显式传入CPU_ONLY或CPU_AND_NE。

## 7. 本地复核、证据范围与继续问题

本地代码：标准库完整参考/合同、显式原helper collector、、26项正负测试。普通CPU入口不import Core ML/NumPy、不联网、不删除文件或触发设备：

显式设备入口需要既有固定依赖与read-only reference Git，写新ignored目录；先保证至少2GiB剩余空间：

七份实际installed Python文件逐字节核对固定Git后才import执行；10份来源文件、原collector副本、synthetic package全部文件与terminal receipts可追溯。188份主数组只做lossless去重，12个完整payload在wiki内可离线逐元素复算。较早成功pilot的39次/47数组另作完整审计，不补写成已具有后来per-call timer/validator元数据的正式过程。helper模拟调用、诊断重复返回、warmup、失败尝试和硬件dispatch都不混为同一分母。

本批26项专项、374项ANE、2,092项完整suite（2 skips）、只含techniques的同规模隔离suite及七项离线工具均通过；Python3.9/3.11 curator产物逐字节相同。完整数值与scope验收不代替真实模型或硬件证据，实际receipt/hash见验证记录。

下一步仍需：较大且实际首选ANE的representative图、hint/shape specialization的配对成本与完整输出、真实KV/训练状态的持续请求，以及独立Instruments/物理证据。改源码/wheel/native/OS、shape/精度/输入域、state协议、hint、输出接口或实际消费者均重开。本批没有发布speedup、能耗、真实checkpoint/媒体质量、完整训练或生产默认资格；整体研究继续。
