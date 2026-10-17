# ONNX Runtime Core ML EP：真实派发、三层回退与陈旧模型缓存

<!-- knowledge
id: technique-ane-onnxruntime-coreml-partitions-and-cache
type: method
operator: conv, layout, scheduling
backend: coreml, cpu-reference
device: cpu, ane
symptom: stale-cache, measurement, compile-bound, wrong-layout
semantics: numerically-equivalent, contract-only
representation: fp32, fp16
boundary: producer-consumer
api: public, reference-only
-->

[ANE目录](README.md) · [固定源码](sources.md#delegate-sources-20261016) · 完整数据 · [ExecuTorch对照](executorch-coreml-aot-and-estimates.md)

2026-10-16，M4 Max / macOS26.6.2。ONNX Runtime **1.31.0**、ONNX1.23.2、NumPy2.0.2装在新的任务venv；原`.venv`和产品依赖未改动。wheel build info的`402979a`与`v1.31.0@402979ac...`对应，另新建匹配release的只读源码clone；没有用既有1.32开发树冒充运行版本，也没有把build info当作loaded-image attestation。

本批实际运行小ONNX模型并读取**自有ORT session profile**。主试117份输出；加上保留的partial client run共145份完整payload见证、9,784个输出元素。CoreML EP有108条实际软件provider派发事件，**不是108个ANE kernel**；没有硬件trace、能耗或完整模型速度资格。

## 1. 从ONNX节点到Core ML子模型，不是一个全模型开关

固定[GetCapability/Compile](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/core/providers/coreml/coreml_execution_provider.cc#L44)：

```text
ONNX graph → 每个builder按op/shape/dtype/initializer/format判断支持
  → 连通支持分区 → 原生builder生成Core ML模型 → compile/load
  → fused ORT kernel绑定host inputs → 持有该子模型mutex → Predict → host outputs
```

这里直接由C++ builder写Core ML表示，不经过本机coremltools的PyTorch转换器。支持判定不只看算子名；格式、常量权重、shape和细分参数都参与。一个model可以有多个CoreML分区，中间穿插CPU kernel；更多分区意味着更多包装/同步边界，不自动意味着更多ANE并行。

`EnableOnSubgraphs`涉及ONNX控制流的child graph，不是“是否允许把普通顶层graph切成EP分区”的同义词。少量shape/cheap op分区有trivial过滤，但不是所有便宜activation都被排除；实际runtime-weight NeuralNetwork案例仍留下单独Relu delegate。

## 2. 注册、派发、物理设备是三个问题

原生[factory/options](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/include/onnxruntime/core/providers/coreml/coreml_provider_factory.h#L18)的“ONLY_ENABLE_DEVICE_WITH_ANE”命名容易误读；它不保证整个模型只在ANE执行。现代`MLComputeUnits=CPUAndNeuralEngine`仍允许Core ML内部用CPU。

| 层 | 要观察什么 | 本批反例 |
| --- | --- | --- |
| wheel/EP注册 | available/registered providers、有效options | CoreML可用，不说明目标Conv被接管 |
| ORT节点/分区派发 | 自有profile中kernel的provider、对应原graph范围 | NeuralNetwork runtime-weight Conv在CPU，CoreML只执行Relu |
| Core ML内部设备 | compute plan作预测；实际设备另需trace | CoreML EP事件本身不能证明ANE |

本批`ORT_DISABLE_ALL`使这些小图的原节点映射容易追溯。复杂生产图会经过fusion/decomposition，不能把一个fused CoreML事件数当作原ONNX算子数或硬件指令数。

## 3. 实际小图的覆盖与失败

模型是`[1,4,4,4]`输入的1×1 Conv+Relu，dyadic权重/bias；dynamic案例在同一session内逐generation交替传入scale2/4的W。混合图在后面接NonZero，输出int64索引。全部采用独立完整参考，不是仅检查shape/finite。

| 控制 | 输出/实际软件provider观察 |
| --- | --- |
| CPU EP，ONNX省略默认pads | 9份正确输出，CPU Conv+Relu |
| CoreML MLProgram，同一合法implicit-pad语义 | session创建时Core ML compile拒绝：缺`pad`；0次Run |
| 显式`pads=[0,0,0,0]`，同一函数 | 9份正确输出，9条CoreML fused kernel事件 |
| Conv+Relu→NonZero | 9份正确int64输出；每次CoreML分区+CPU NonZero |
| runtime-weight Conv，MLProgram | W确实交替变化，9份正确输出；CoreML分区事件 |
| 同runtime-weight Conv，NeuralNetwork | 9份正确输出；Conv在CPU，Relu在CoreML |
| 非法provider option | Python constructor自动建成CPU-only session，仍有9份正确输出 |
| 混合图+禁用CPU EP分区回退 | session创建拒绝；0次Run |

### 3.1 合法ONNX默认值不一定被这个lowerer正确展开

ONNX checker和CPU运行都接受implicit zero pads。release的[AddPadTypeAndPads](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/core/providers/coreml/builders/impl/builder_utils.cc#L384)在NOTSET且未提供pads时走`valid`分支，只放pad_type；同文件SAME分支却显式放ignored pad。当前host的Core ML parser要求本次conv有pad输入。

显式零pads是单独记录、数学等价的控制，不是把失败原模型悄悄换掉。原型和campaign中的compile失败都保留；没有修改ORT/source或宣布所有Conv/所有OS不兼容。

### 3.2 “允许CPU回退”有多个不相同的层

1. 支持判定后将NonZero等节点分给CPU，是正常graph partition。
2. Python[constructor错误处理](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/python/onnxruntime_inference_collection.py#L682)对某些ValueError/RuntimeError重建CPU session；非法option案例实际走了这里。不是所有compile异常都这样恢复，implicit-pad案例仍抛Fail。
3. `disable_fallback()`关闭的是后续`session.run()`的EPFail恢复，不能撤回调用它之前已经发生的constructor fallback。
4. `session.disable_cpu_ep_fallback=1`拒绝ORT graph中落在CPU EP的节点，**不禁止Core ML内部选CPU**，所以也不是ANE-only证明。

适配器应核对actual providers、实际目标节点覆盖和完整输出。不能只看到CPU fallback后仍exit0/数值正确，就给CoreML/ANE成功标签。

## 4. 同路径换权重：明确的缓存内容身份合同

固定[cache文档化警告](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/include/onnxruntime/core/providers/coreml/coreml_provider_factory.h#L75)：CoreML EP不自动检测ONNX模型改变，缓存由caller管理。[builder](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/core/providers/coreml/builders/model_builder.cc#L415)先取user key，否则按model URL或有限graph名称生成key，再结合partition、static/dynamic、format组成路径；已有路径会复用，不重建权重。

自有目录内保存A/B两个原始snapshot，只覆盖我们自己的working leaf。A的diagonal weight=2，B=4；graph名称/shape/path相同。没有改用户模型或清全局cache：

| 调用 | 当前ONNX内容 | 实际输出匹配 |
| --- | --- | --- |
| cached A | A | A，正确 |
| 原路径替换B、复用默认cache | B | 仍为A，9次finite旧结果 |
| 新进程再次读这个B和同cache | B | 仍为A，9次；不是仅进程内旧session |
| B附正确content key | B | B，9次正确 |
| B关闭ModelCacheDirectory | B | B，9次正确 |
| B换另一个文件路径 | B | B，9次正确 |

这是**复现已声明的caller缓存误用合同**，不是声称底层Core ML随机算错。正确metadata键必须是`COREML_CACHE_KEY`，位于ONNX `metadata_props`，不是任意相似名称或provider option。

首次client误写为`CACHE_KEY`，该属性被忽略，所谓content-key控制仍读旧A；完整数值门槛当场拒绝。保留这次partial run及最后1份无导出profile的结果，修正helper并加精确键名guard后另开新目录。没有将它包装成“正确content key也无效”。

本例用不含自身key的model bytes及已记录编译spec生成56字符hex，再写入正确metadata。实际模型若有external weights、LoRA、graph属性、dtype/layout、opset等变更，也必须管理其内容身份；不要仅以shape或路径判断同一权重。in-memory模型的默认名称hash也不是完整权重digest。不同框架缓存的artifact层不同，不能盲目把所有执行选项塞进所有cache key，见ExecuTorch对照。

## 5. 独立审计与复跑

重建所有145个逻辑input/output payload SHA，含动态W和NonZero全部索引。117份主试输出中18份是故意复现的旧cache负例；保留partial run后总计28份旧A数值观察，其中1份没有完整provider profile，不能补造其派发证明。其余已导出profile合计108 CoreML/90 CPU事件；这些都是软件kernel边界，duration不叫ANE op时间。

另对raw NPY、受限ONNX protobuf字段/initializer、原始owned profile和snapshot SHA逐一核对。protobuf reader只抽取自有模型需要的subset，不替代ONNX checker。全部观测都未获得物理placement、完整模型质量或性能资格。

实际复跑用新的隔离venv安装固定binary wheels，不改产品环境；完整wheel URL/hash/dependency锁定在证据中。示例客户端要求显式device和新的output：

所有图都很小、值域为明确dyadics，不代表真实activation/LLM或大权重。未比较正式warm性能、峰值内存或能耗；profiling在正式执行中开启，也不把这些带instrumentation的时间拿作速度榜。下一步应在实际消费者中验证分区边界、shape/IO成本和模型质量，再考虑部署。
