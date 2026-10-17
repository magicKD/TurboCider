# ExecuTorch Core ML Delegate：AOT边界、共享权重、状态与估计式profiling

<!-- knowledge
id: technique-ane-executorch-coreml-aot-and-estimates
type: method
operator: quantization, layout, scheduling
backend: coreml, cpu-reference
device: cpu, ane
symptom: stale-cache, measurement, lifetime, precision-loss
semantics: contract-only
representation: fp16, fp32
boundary: producer-consumer
api: public, reference-only
-->

[ANE目录](README.md) · [ORT真实分区/缓存对照](onnxruntime-coreml-partitions-and-cache.md) · [固定源码](sources.md#delegate-sources-20261016)

后续[公共MLState/多函数实验](stateful-multifunction-and-retry.md)另做真实小图、完整state与warmup/重放控制；仍未运行原ExecuTorch delegate，不回填成本批已完成的范围。

固定`pytorch/executorch@3f15285f3b668b0ecdac6fa985bc207edeb50efc`，选定22文件source ledger中的ExecuTorch部分，不称最新或全仓库审核。**本轮没有安装Torch/ExecuTorch，也没有运行完整delegate/export/quantized模型**；实际执行仅为一个已审函数的六个mock-IR观察，profiling算术另外由独立CPU参考重建。与ORT的真实session结果不混为同级证据。

## 1. 与ORT不同：PyTorch export→分区→AOT字节→native runtime

[partitioner](https://github.com/pytorch/executorch/blob/3f15285f3b668b0ecdac6fa985bc207edeb50efc/backends/apple/coreml/partition/coreml_partitioner.py#L115)向coremltools查询FX node支持，再叠加本项目override/skip规则；CapabilityBasedPartitioner给支持分区打delegation tag。placeholder/output不自动当算子接管，get_attr常量可接管。

| 阶段 | 主要实现与合同 |
| --- | --- |
| export/partition | 尽量保留Core ML支持的高层op，不先强制分解SDPA等再期待同一优化；代码建议transform-and-lower流程 |
| conversion | [compiler](https://github.com/pytorch/executorch/blob/3f15285f3b668b0ecdac6fa985bc207edeb50efc/backends/apple/coreml/compiler/coreml_preprocess.py#L595)调用`ct.convert(source=pytorch, convert_to=mlprogram)`，精度/deployment/compute units由compile spec控制 |
| AOT资产 | 可保存mlpackage，或取得compiled mlmodelc；加有序IO metadata/debug mapping，再flatten到字节 |
| runtime | native解包内存文件系统、编译/取缓存、按metadata顺序绑定输入输出、选择function再predict |

当前源码默认model compute precision为FLOAT16、compute units为ALL；两者不等价，更不是FP16 ANE物理MAC证明。ONNX Runtime EP则走自己的原生builder，不经过这套PyTorch转换器，不能把一个frontend的兼容性故障直接迁给另一个。

## 2. 部分delegate不是任意失败都能自动落回CPU

`lower_full_graph=False`允许不支持节点不打delegate tag；True则对不支持op抛错，并要求常量/mutable-buffer takeover，不能同时配置skip ops。enumerated shape compile spec也要求full-graph模式。

支持查询之外还有明确override，例如add/sub的alpha条件、diagonal、特定随机或flattened argmin/max形态；这些是当前项目针对特定上游行为的规则，不写成Apple所有OS的永久限制。

即便支持predicate通过，后续ct.convert、压缩、序列化或runtime compile仍可能失败。所读路径不能据此承诺所有失败都会自动重新导出CPU图。需要区分：未接管节点、AOT lowering失败、delegate初始化失败、prediction失败；采用者应检查完整返回/输出，而不是只看delegation tag存在。

## 3. 共享embedding/lm-head为什么牵动压缩配置

原compiler可根据experimental `op_linear_quantizer_config`对权重做linear quantization，通常让gather opt-out，避免默认压缩embedding。如果同一个constant同时被gather和linear使用，不同消费者配置冲突可能使compression失败。

[_gathers_sharing_a_weight](https://github.com/pytorch/executorch/blob/3f15285f3b668b0ecdac6fa985bc207edeb50efc/backends/apple/coreml/compiler/coreml_preprocess.py#L39)找出这类gather名称，以op-name配置覆盖默认gather opt-out，使同一共享constant的消费者使用一致配置。这解释了“跳过embedding”不能只写一个op-type黑名单，必须看权重消费者关系。

受限source helper client只AST提取已读函数，未执行模块import/初始化：

| fixture | 原函数结果 |
| --- | --- |
| 缺MIL program | [] |
| gather-only constant | [] |
| gather+linear共享同一constant | 返回gather名称 |
| 两个不同weight名称 | [] |
| 两个gather与linear共享 | 两个gather名称 |
| 不同函数里不同constant但同var.name | 仍合并名称并返回gather；是helper scope边界 |

最后一例**不是已复现的完整模型bug**：当前caller在逐method转换/压缩时调用helper，之后才组装multifunction。不能脱离caller前置条件，把mock多函数边界当作真实export失败。helper按var.name而非对象身份聚合，也说明移植到别的IR/调用阶段需要重审。

初次helper probe用系统Python3.9解析整个上游文件，被`match`语法拒绝，函数未执行；这不是源文件在受支持Python上的语法错误。后续Python3.11执行六个fixture，并加了3.10+解析guard。没有安装Torch来“补”这个纯解析问题，也没有给缺失的旧source snapshot造hash。

权重压缩、导出精度、activation量化与实际整数MAC是不同层。本批只验证helper返回名称，没有验证真实quantized模型误差、INT8吞吐或INT4/LUT硬件算术。

## 4. 多method打包与cache key不要只照搬另一个框架

当前multimethod预处理默认POSITIONAL：各method必须有相同partition数，将同位置分区装到Core ML multifunction模型，NamedDataStore仅存一份共享资产，各method保存引用；DISABLED则独立处理。相同partition数是打包前置条件，不证明各method图相同或权重必定都被去重。

runtime还需要两个不同名字：methodName用于查该method的有序IO metadata，functionName用于`MLModelConfiguration`选择Core ML function。包含所有function的compiled资产不应每method再复制一遍。

在已读[manager](https://github.com/pytorch/executorch/blob/3f15285f3b668b0ecdac6fa985bc207edeb50efc/backends/apple/coreml/runtime/delegate/ETCoreMLModelManager.mm#L790)中，legacy asset路径在identifier后加compute-unit字符串；新cache路径明确不加，希望同一compiled model在不同unit配置下复用。这里缓存的artifact层与ORT的模型转换缓存不同：**内容身份与执行配置应分层设计，不是所有cache都必须按method/device复制**。

普通预处理为资产分配UUID。模型更新、外部权重或手工复用identifier仍需管理内容/版本身份；本批未复现ExecuTorch陈旧cache。其cache初始化有版本不符purge、过期temp/orphan清理，preprocess也会移动/删除自有临时目录；未来runtime测试必须指定新的owned cache，不把任意用户目录交给这些管理器。

## 5. 逐op profiler实际上如何生成区间

这是本批最需要区分的profiling层。[SDK profiler](https://github.com/pytorch/executorch/blob/3f15285f3b668b0ecdac6fa985bc207edeb50efc/backends/apple/coreml/runtime/sdk/ETCoreMLModelProfiler.mm#L171)取得MLComputePlan中的preferred/supported device与estimatedCost；测一次整模型`predictionFromFeatures`的mach_absolute_time窗口后，按遍历顺序分配：

```text
op_end = cursor + uint64(model_duration * estimatedCost)
记录被选择的path区间
cursor = op_end
```

它不是逐op读取设备时间戳。const被过滤，没有有效cost/usage的op可能被省略；selected-path过滤也不将剩余权重重新归一化。独立CPU公式见本地合同：101 ticks按.3333/.3333/.3334分配后只覆盖99 ticks，不能把剩余2 ticks叫stall。只选B时，仍保留A所占的前缀，不把B扩成整个窗口。

[event logger](https://github.com/pytorch/executorch/blob/3f15285f3b668b0ecdac6fa985bc207edeb50efc/backends/apple/coreml/runtime/sdk/model_event_logger_impl.mm#L91)将这些**estimated** ticks换成ns，送入通用delegate profiling事件。metadata包含estimatedCost、preferred/supportedComputeUnits等，消费者应保留这些来源标签。ns格式、漂亮时间线或preferred=NE均不使其变成物理ANE trace；顺序铺出的区间也不能证明真实硬件没有并发分支。

这与本批ORT profile中的实际软件provider kernel边界又不同，二者都不能自动提供内部ANE kernel/PMU。SDK profiler有OS/SDK可用性条件；model compiler还包含异步compile+semaphore超时，timeout不等于所有底层活动已经drain。本批没有运行这些SDK等待/清理/计时路径。

## 6. Mutable buffer不是训练optimizer state

partitioner默认可以take over mutated buffer作为Core ML state，源码提示这需要macOS15+/iOS18+。这对KV等有状态推理有意义；仍需每session/method的state所有权、reset/取消/重载合同。

不能由“PyTorch导出”“mutable buffer”或“支持Core ML”推出自动反向传播、Adam moments驻留或ANE训练。本批没有执行stateful模型；训练路线见[手工反向/动态参数/驻留方案](training.md)。

后续优先用真实导出小模型验证partition回退、multifunction IO/state映射、压缩与共享权重质量，再做完整消费者速度/内存/能耗对照。源码存在、mock helper通过、真实delegate数值通过和物理ANE资格继续分别记录。
