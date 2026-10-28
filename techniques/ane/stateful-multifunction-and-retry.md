# 公共MLState与多函数模型：状态所有权、重试提交点与warmup

<!-- knowledge
id: technique-ane-stateful-multifunction-and-retry
type: method
stage: prefill, decode
operator: layout, scheduling
backend: coreml, cpu-reference
device: cpu, ane
symptom: stale-cache, lifetime, precision-loss, measurement
semantics: numerically-equivalent, contract-only
representation: fp16, fp32
boundary: producer-consumer
api: public, reference-only
-->

[ANE目录](README.md) · [固定源码/SDK](sources.md#stateful-sources-20261016) · [输出backing前序实验](output-backing-and-stride-ownership.md)

2026-10-16，M4 Max / macOS26.6.2/build25G83、SDK26.5、clang21、既有coremltools8.3.0/NumPy2.0.2。本批真实执行自有公共Core ML **stateful multifunction** 模型，不安装Torch/ExecuTorch、不下载LLM，不执行原ANEMLL或ExecuTorch runtime。

两seed17/29×CPU_ONLY/CPU_AND_NE，共四个完整进程；成功pilot未改源码即作为矩阵的CPU/seed17单元，不为改名字重跑。68份实际输出、76份完整输入、100组八slot快照（800份完整state tensor）全部独立重算；状态共48,000元素。**其中4份INT32输出是数值负例，另4次故意的post-success盲重试重复更新状态；不是68份都符合“只执行一次逻辑请求”的原意。**

## 可复用的方法：将连续请求设计成明确的状态事务

1. 区分资产、函数、state schema 与 session slot 身份；显式定义 prefix、position 和 KV 的有效范围。
2. 使用独立 slot 和可重建的状态检查，验证跨函数接续，而不只比较单次预测返回值。
3. 失败后先判断 state 是否已提交，再选择恢复或重试；不能把 stateless fallback 套到有状态调用上。

适合设计 prefill/decode 和多函数部署；下文是公共小图合同，不替代真实模型或不同版本的验收。

## 1. 资产、函数、state handle不是同一个身份

Apple SDK26.5的`MLState.h`与`MLModel+MLState.h`给出三个关键合同：newState初始化为零；同一state上的prediction必须串行，in-flight时不可读写；回调中的mutable view不可逃逸，地址可在下一次访问改变。SDK没有给出“任何预测error都自动回滚”的事务保证。

[官方stateful说明](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)与[多函数说明](https://apple.github.io/coremltools/docs-guides/source/multifunction-models.html)是学习入口；本批另固定8.3 tag的文档/源码SHA与选段，不把活动网页当作固定运行版本。

| 身份 | 管理的东西 | 不自动管理什么 |
| --- | --- | --- |
| `.mlpackage/.mlmodelc`资产 | 图、常量、函数描述，可共享编译资产 | 不自动给每个用户请求创建隔离状态 |
| `MLModelConfiguration.functionName` | 此次加载的MLModel handle选择哪个函数 | 不自动切换到该请求应使用的MLState |
| `MLState` handle | 该逻辑会话的可变buffer | 不冻结快照，不承诺失败回滚，不是Adam moments |
| host checkpoint | 自有副本，能在明确静止边界写回 | 不替代模型外position/mask、采样器、取消和其他设备状态 |

本例从同一个compiled资产加载one、two、再次load one和nil默认函数的handle。原始调用只证明这些load和后续行为；没有以对象地址证明模型实例一定不同、没有推断内部cache或物理权重去重。

## 2. 真正运行了什么图

两个函数都有FP16 `acc:[1,4,3,5]`状态与FP16输入X；output描述是FP32：

```text
one: S' = FP16(S + FP16(X + B))
two: S' = FP16(S + FP16(2X + B))
B[c] = [0, .25, -.25, .5]
write_state(acc,S') → read_state(acc) → cast(FP32) → Y
```

只用精确dyadic输入域，不把这套累加器当作Transformer KV或训练收敛。`coreml_update_state`在固定[coreml dialect](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/converters/mil/mil/ops/defs/coreml_dialect/ops.py#L13)中是write_state后read_state的组合；[优化pass](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/converters/mil/mil/passes/defs/optimize_state.py#L19)会将更新靠近相应functional op，并让下游优先使用更新后的state。不能把Python里一个update调用或一个state名当作设备dispatch数。

用已安装8.3的`MultiFunctionDescriptor`/`save_multifunction`导出，default=one。只解析原模型的独立client检查12份原model message、IO/state、常量、连接和write/read/cast；one/two都引用同一bias blob的offset64，合并文件与两个原weight.bin内容SHA相同。证据证明这个小例的**文件资产共享**，不是加载后RAM/ANE SRAM共享或LoRA实际训练。

两个额外只读sparse clone分别固定coremltools8.3 `07ffd860...`和ANEMLL `f4ad26d...`；既有coremltools开发树不被拿来冒充8.3。六份相关installed Python文件逐字节匹配此tag；native wheel/Apple framework仍未做source或loaded-image attestation。

## 3. 八个独立slot与跨函数接续

每进程先创建A–H八个newState。每一个event之后都通过官方callback读取**全部八slot**的完整FP16逻辑字节与canonical FP32，不只读被调用的state或若干scalar。没有保留callback的MLMultiArray/pointer到外部，也没有并发调用同一state。

- A执行one两次，再交给two，再交给再次load one的handle：继续同一条完整状态轨迹，没有自动清零。
- B/C/D分别由不同handle或独立newState初始化，初始均零；A的更新不污染它们。B随后交给two也接续自己的轨迹。
- 显式将A写零后，下一次从零更新；写回第一段的owned checkpoint后，two从该旧轨迹继续。
- nil functionName按default=one执行，E仍是它自己的状态。不存在的函数名加载被拒绝，未执行预测。

只有本例一致的state名称/shape/dtype及图语义取得这些兼容性观察。不能据此跨任意模型/adapter/量化方式复用MLState，也不能只比较名字与shape就认定真实KV的layer顺序、token位置、RoPE、attention mask和head布局兼容。

不带state调用stateful图，本机返回“acc must be an MLState”错误。不存在函数名的NSError反而说functionName只允许MLProgram；但本资产已由protobuf确认是MLProgram，one/two均实际可运行。因此保留错误原话，不用单条错误文字推断模型真实类型。

## 4. 初始error、输出拒绝与状态提交点要分开

### 4.1 本例错误shape：没有推进，不等于所有error都不推进

F从零开始，提议错误shape的output backing。初次返回error；完整F和其他slot保持原值。去掉backing再执行同输入，得到一次更新的正确输出。这是本机这个validation失败的具体观察，不是所有Core ML失败、超时或取消的原子回滚承诺。

### 4.2 输出语义不合格时，state可以已经提交

G从零开始，给FP32 output提议INT32 backing。本机description检查为false，预测却成功，返回同backing的INT32，数值是正确FP16 state值的向零截断；**完整state已按浮点图推进**。

接着故意像“验证失败就重跑”那样，用无backing再次执行同一个逻辑输入。第二次的FP32输出数值对它实际执行的图完全正确，却已经是**两次累加**，不是第一次请求的正确输出。不能把它称作“重试修复了第一次结果”。

另外显式恢复G到调用前的owned全状态，再重跑，完整输出与fresh H的一次执行相同。该对照只验证单state、自有同步小图的受控rollback；真实模型还可能有多state、host position、GPU/cache/随机数或副作用，不能由恢复一块buffer就承诺事务安全。

这是原创client中的post-success数值拒绝/盲重放负控，**不是复现ExecuTorch在nil-error retry中重复更新的bug**。后者只在没有modelOutputs时重试；其prepareArgs还会按model constraint准备匹配dtype，不直接照搬本例INT32提议。任意nil失败是否已修改state，仍须在原路径单独取证。

## 5. 零输入warmup也可能改变状态

A在已有历史上执行one(zeros)。虽然输入全零，图有B，状态和输出仍改变；source-like清零会丢失已有会话。写回warmup前的owned checkpoint才恢复原轨迹。四进程都完整检查了这三个阶段。

固定[ExecuTorch prewarm](https://github.com/pytorch/executorch/blob/3f15285f3b668b0ecdac6fa985bc207edeb50efc/backends/apple/coreml/runtime/delegate/MLModel_Prewarm.mm#L110)生成零输入，走一次真实prediction；[ETCoreMLModel](https://github.com/pytorch/executorch/blob/3f15285f3b668b0ecdac6fa985bc207edeb50efc/backends/apple/coreml/runtime/delegate/ETCoreMLModel.mm#L409)之后将各state清零。这解释了为何warming本身不一定无副作用，也说明初始化时的zero reset不等于对已活跃会话做snapshot restore。本批没有执行其原prewarm函数，不能把本图bias作用扩成所有KV模型的结果。

有状态benchmark要保持各arm相同的起始state/history，或在测量窗口外恢复完整checkpoint。若A先执行、B接着用A推进后的state，数值/延迟都可能在比较不同请求。状态读回/写回及checkpoint验证也不应从完整采用成本中无理由删除。

## 6. ANEMLL与ExecuTorch：共享资产不自动共享会话state

固定[ANEMLL create_unified_state](https://github.com/Anemll/anemll/blob/f4ad26d061dc2e426faa19f81c367343bdeb9f0d/tests/chat.py#L1274)从prefill或infer模型建立state；一致时跨chunk/function传同一handle，不一致时建per-chunk list。所读一致性启发式只看state repr是否含`kv_cache_global`，**不是完整state schema检查**。固定8.3 Python的MLState包装本来就是opaque、没有列出state名称的repr实现，因此移植时应读model description/实际函数描述，不能只用repr猜兼容。本批未跑原chat或真实Gemma/LLM，未声称已复现其完整模型错误。

重要版本前置条件：该ANEMLL ref的[requirements.txt](https://github.com/Anemll/anemll/blob/f4ad26d061dc2e426faa19f81c367343bdeb9f0d/requirements.txt)明确要求`coremltools>=9.0`。本批8.3不在它的依赖合同内；8.3的opaque/read-write/repr差异是**向旧环境移植的兼容边界**，不能据此判ANEMLL在其受支持环境有bug，也没有为做对比而改动现有产品依赖。

requirements中关于ANE的版本注释只作该项目依赖说明，不升级为“ANE公共支持始于coremltools9.0”的平台规格。

[run_prefill/generate_next_token](https://github.com/Anemll/anemll/blob/f4ad26d061dc2e426faa19f81c367343bdeb9f0d/tests/chat.py#L1026)在prefill/infer/rotate函数间选择，同时携带position和mask。其optional update-mask探测会在包含`update_mask`的RuntimeError后换输入重试；这类兼容性fallback也依赖首次失败的state语义，不能只因调用抛错就假设没有任何提交。

另外[state_transition.py](https://github.com/Anemll/anemll/blob/f4ad26d061dc2e426faa19f81c367343bdeb9f0d/anemll/utils/state_transition.py#L62)对四维KV数组扩容/压缩、拷贝有效prefix与padding。这是数组变换，不会自动处理Core ML handle、RoPE/position或模型外状态。其示例的Python read_state/write_state不在本次8.3 opaque包装中；需版本适配或公共native view接口。函数的same-shape/full-valid early return还会返回原数组、跳过dtype转换；这只是已读分支，未在本批执行该原helper。

固定ExecuTorch每个`ETCoreMLModel`在初始化时自行newState；multimethod AOT将同位置分区放入一个资产，可共享文件/常量，但不会因此自动把两个runtime model owner的state合并。[take_over_mutable_buffer](https://github.com/pytorch/executorch/blob/3f15285f3b668b0ecdac6fa985bc207edeb50efc/backends/apple/coreml/partition/coreml_partitioner.py#L324)指导出mutable buffer为Core ML state，不是训练autograd或optimizer状态。更完整的AOT/估计profiling边界见[此前报告](executorch-coreml-aot-and-estimates.md)。

## 7. 本地复现、独立验证与剩余问题

标准库oracle独立计算每次FP16更新、函数scale、checkpoint和全部八slot，核对typed bytes及canonical hash；再逐一核对原始数组、模型、client/binary与terminal receipt。11个离线测试包括跨slot污染、错误函数、初始error造输出、盲重放、reset冒充restore、bool generation、路径逃逸及资格升级反例，不依赖Core ML/NumPy/兄弟源码。

原创native client与显式driver只构造自有模型、编译公共API客户端；必须显式device、固定既有依赖和全新owned目录，不自动安装/下载、清系统cache或改产品默认：

本批没有有效设备trace/compute-plan/PMU，不比较CPU与NE速度；CPU_AND_NE仅是许可策略。真实KV、多状态/异构事务、取消/异步错误、原wrapper的nil-error路径、请求质量、跨设备成本与能耗仍未验证。重开条件包括OS/SDK/source/graph、state schema/精度、函数/请求owner、layout/position/mask、checkpoint和实际消费者变化；推理state存在不能晋升为训练或收敛资格。
