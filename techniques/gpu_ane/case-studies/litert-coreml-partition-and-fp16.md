# LiteRT旧Core ML delegate：分区、FP16常量与静态形状

<!-- knowledge
id: technique-gpu-ane-case-studies-litert-coreml-partition-and-fp16
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [张量交换与生命周期](litert-coreml-handoff-and-lifecycle.md) · [23文件读本](../sources.md#litert-coreml) · [第九轮范围](../../MAINTENANCE.md)

2026-10-16，固定 `google-ai-edge/LiteRT@b0983dc2dc27141a64fba7553f40299631e65937`。这次读的是保留在 `tflite/delegates/coreml` 的旧delegate与相关runtime，不是LiteRT V2全栈审计。没有构建、导入或运行LiteRT/TensorFlow，没有执行CoreML预测或上游测试。

## 1. 三种来源不能互相“补全能力”

| 来源 | 本次核对到的内容 | 正确的使用方式 |
| --- | --- | --- |
| 固定ref的[根README][root] | 新应用推荐Compiled Model API；iOS/macOS支持表的ANE带coming-soon星号 | 不把保留的旧delegate当成V2已接通ANE的证据 |
| [旧delegate README][legacy]与[官方iOS指南][guide] | 以Interpreter/manual delegate为背景的部署和op说明 | 用于理解旧路径；不是本页推荐新应用照抄的代码 |
| 固定ref的[实际入口][delegate]与[工具provider][provider] | 新系统用MLAllComputeDevices检测ANE；旧系统另有iPhone/iPad与Apple Silicon分支，工具构建门槛包含ARM64 Mac | 能证明选定源码存在Mac路径，不能证明本机模型可运行、V2已支持或已获性能资格 |

根README的AI coding directives要求新应用使用Compiled Model API，本轮只读runtime内部、编写独立CPU说明，没有生成旧Interpreter应用代码，也没有运行安装/构建脚本。仓库Git树未发现AGENTS.md；已完整读根README，未把其中的安装示例当成执行授权。

旧README对macOS的描述仍是“未按iOS同等方式测试”；不能用它否定后来实际代码，也不能反过来用代码分支更新成“所有Mac都已验证”。本页只给固定ref的源码深度，后续V2 Apple GPU/ML Drift与ANE集成另列为待研。

## 2. 能创建delegate，不等于图被放在ANE

[头文件][header]的enabled_devices控制**是否创建delegate**。零初始化枚举值为“仅在有ANE的设备创建”；[工具provider][provider]反而显式请求AllDevices。比较来自不同入口的benchmark，不能假定它们具有相同准入政策。

[入口实现][delegate]先做系统/设备检查；新系统查询实际compute-device列表，较老系统使用平台分支。后续[模型加载][executor]仍配置 `MLComputeUnitsAll`。所以即使“有ANE才创建”的准入通过，模型的具体层仍可被CoreML放在CPU/GPU；这个开关不是 `CPU_AND_NE`，也不是ANE-only或并行开关。

一个保留的接口检查点：外部C入口会解引用options，而内部C++构造器才有空指针默认处理。不能从内部构造器推导外部入口允许nullptr；本轮没有做空指针故障试验。

## 3. 量化、算子版本、参数常量是三道不同的门

[IsNodeSupportedByDelegate][delegate]先检查op version，再检查激活输入为FP32，最后进入op-specific validator。TransposeConv的激活在input[2]，不能把input[0]的int32 output-shape误认为激活dtype。

选定实现中，多数算子只接受version1；DepthwiseConv允许到2，FullyConnected允许到6。某个FC validator支持FP16权重，不代表整个delegate支持FP16激活入口；局部检查必须连同外层门槛阅读。全整数INT8/UINT8激活明确被挡住，不能因为硬件支持INT8就给这条路径赋予W8A8能力。

[卷积builder][conv]要求常量weights，普通卷积的bias及TransposeConv的output shape也有常量门槛。FP32权重转到protobuf floatvalue，FP16权重则以uint16 payload重排并保存在float16value；这与[MNN的压缩卷积先回float嵌入](mnn-coreml-graph-boundaries.md)不是同一表示合同。它仍不证明CoreML最终执行哪种物理MAC。

[FC validator][fc]要求浮点常量权重/可选bias，还无条件要求除末维外的输入维度乘积为1。注释虽然提CoreML2，这个限制并没有按conversion version分支。因此不能从“target CoreML3”推出已支持批量LLM projection/prefill。LoRA或逐层变化权重也不能直接复用这些静态常量图而不核对重建策略。

[binary util][binary]的broadcast比通用逐轴broadcast更窄：batch不同需一侧是常量标量；空间H/W不同需整对为(1,1)，而非各自任选一轴为1；channel不同需一侧为1。比如NHWC `[1,4,1,8]` 与 `[1,1,5,8]`在通用规则下可广播，这个检查却拒绝。不要把“支持Add/Mul”理解为支持全部numpy-style形状。

## 4. 分区先由runtime预览，再按节点数挑选

普通路径由[GraphPartitionHelper][utils]收集支持节点，交给runtime的 `PreviewDelegatePartitioning` 获得合法子集；本页没有重新实现这一步，也没有把“相邻支持op”当成完整的图分区算法。

之后的可见策略是：

1. 按每个候选分区的节点数降序排序；
2. 至多选max_delegated_partitions个；
3. 小于min_nodes_per_partition的后续分区不再选择；
4. 入口把非正的max解释为不限，非正的min设为2。

排序只比较节点数，相同大小没有稳定tie-break合同。它不是按MAC数、边界激活字节、CoreML编译成本或整请求收益排序；大分区也可能只是很多便宜操作。本库CPU见证对预算边界的同大小候选保留多个可能集合，不编造上游一定选谁。

[上游单元测试][utiltests]使用mock提供分区，检查最大的 `{0,3,7,8}` 等集合，不要求节点ID连续。这些测试只读未运行；本库的独立测试也不因此获得真实runtime预览或图执行资格。

### 一个重要例外：full coverage不走同一个门槛

CoreML入口实际使用FP16GraphPartitionHelper。其[实现][utils]检查：支持节点数+识别出的常量FP16 dequant节点数是否等于原execution-plan节点数。若相等，直接选原计划的全部节点；**不再调用同一套max/min分区筛选**。

这不要求图中真的有FP16节点：若所有普通节点都受支持，常量dequant计数为0也能满足条件。一个受支持单op图即使min=100，也可能走全部委托分支。反过来，加一个不支持op变成partial coverage后，单节点小岛又可能被min=2排除。

保存的离线数据明确展示这个不连续策略，而不是声称真实模型已经跑通。min=2不是所有场景都生效的全局保证；调参时要观察实际分支。入口日志中的num_partitions来自preview的候选分区数，亦不应直接当成筛选后实际CoreML模型个数或物理ANE任务数。

## 5. FP16 dequant绕过：先保住CPU消费者

这段优化处理的是**常量FP16→FP32的DEQUANTIZE**，不是通用INT8量化图，也不接受任意动态FP16产物。源码专门排除非常量，以避免DENSIFY等前驱仍需执行时错误跳过数据生产。

部分委托时的数据关系是：

```text
FP16常量W ────→ 已选中的CoreML消费者（输入边重映射）
    └→ CPU DEQUANTIZE → FP32 W ──→ 未委托CPU消费者
```

探测支持性时，helper暂时把某op输入改指原FP16常量，调用validator后恢复；最终只给选中的节点永久重映射。partial coverage保留dequant在CPU，不能为了省一次转换而让剩余CPU消费者读取没有生产的FP32值。

full coverage时可一并claim这些dequant节点；[GraphBuilder][builder]为它们创建[DummyOpBuilder][dummy]，后者不生成CoreML layer。常量FP16权重已经直接编码进相应layer，所以“被委托的节点数”不等于“实际执行了同样数量的CoreML转换算子”。

发生取消委托/重新分配时，[Subgraph::UndoAllDelegates][subgraph]还会把相应FP16输入边映回CPU所需的FP32版本，并跳过DEQUANTIZE节点本身。否则“恢复原execution plan”仍可能留下被改过的tensor边。CPU模型覆盖选中/未选中消费者和一对一映射恢复，但不覆盖所有复杂别名、多输出dequant或上游failure原子性。

## 6. 动态图被拒绝，与静态shape重新编译不同

CoreML delegate设置 `kTfLiteDelegateFlagsNone`，没有AllowDynamicTensors。[标志定义][flags]与[Subgraph调用链][subgraph]共同说明：含运行时动态tensor的图，普通delegate应用前就可能返回ApplicationError；这不是“动态节点自动CPU、其余仍随意委托”的承诺。

静态图的输入尺寸后来变化是另一件事。选定普通路径中：

```text
ResizeInputTensor（尺寸真的改变）
  → UndoAllDelegates：释放delegate kernels、恢复原计划/FP32边
  → 修改输入尺寸
  → AllocateTensors：RedoAllDelegates
  → 重新创建CoreML子图kernel、build/compile/load
```

相同shape且已有data指针有短路；不应把它与新shape的重建成本混在一起。这里没有声明CoreML flexible/enumerated模型，`Prepare`分配I/O向量不等于旧模型自动适应任何shape。VAE分辨率切换、tile尺寸、LLM上下文变化都需要把重新委托/编译时间单列。

[GraphBuilder][builder]的conversion version仅在CoreML2/spec3与CoreML3/spec4旧neural-network格式间选择，不是当前CoreML框架版本。更高target也不单调增加覆盖：[Reshape validator][reshape]在target>=3时直接拒绝，而target2只接受特定静态目标rank/形状参数。

## 7. 可迁移的设计，而非直接借到一个“ANE后端”

值得学习的是：合法分区与收益筛选分层、常量FP16绕过要维护所有消费者、探测阶段可恢复、Undo同时还原边和执行计划、静态shape变化重新取得图资格。值得改进/验证的则是节点数成本代理、full/partial门槛不连续、真实边界copy与编译摊销。

这些原则适用于TurboCider自己的后端实验；不意味着要引入整套旧Interpreter。真实采用还需要checkpoint/LoRA、输出质量、shape、强GPU分母及失败恢复实测。张量交换和2026年的生命周期修复见[下一篇](litert-coreml-handoff-and-lifecycle.md)。

[root]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/README.md
[legacy]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/coreml/README.md
[guide]: https://ai.google.dev/edge/litert/ios/coreml
[header]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/coreml/coreml_delegate.h
[delegate]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/coreml/coreml_delegate.mm
[provider]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/tools/delegates/coreml_delegate_provider.cc
[executor]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/coreml/coreml_executor.mm
[conv]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/coreml/builders/convolution_op_builder.cc
[fc]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/coreml/builders/fully_connected_op_builder.cc
[binary]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/coreml/builders/util.cc
[utils]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/utils.cc
[utiltests]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/utils_test.cc
[builder]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/coreml/builders/op_builder.cc
[dummy]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/coreml/builders/dummy_op_builder.cc
[subgraph]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/core/subgraph.cc
[flags]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/core/c/common.h
[reshape]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/coreml/builders/reshape_op_builder.cc
