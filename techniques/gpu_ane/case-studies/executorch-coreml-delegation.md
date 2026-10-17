# ExecuTorch Core ML：保留大算子、多方法共享与状态化交接

<!-- knowledge
id: technique-gpu-ane-case-studies-executorch-coreml-delegation
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [来源与PR状态](../sources.md#coreml-delegates) · [ORT对照](onnxruntime-coreml-partition-cache.md) · CPU合同

固定 [`pytorch/executorch@3f15285f3b668b0ecdac6fa985bc207edeb50efc`](https://github.com/pytorch/executorch/tree/3f15285f3b668b0ecdac6fa985bc207edeb50efc)，2026-10-16选读12个文件及两份相关PR的指定patch范围。没有安装/导入ExecuTorch、运行export、其测试或模型。不同stable网页可能对应不同文档版本，本文用固定源码解释，不混成本机现有coremltools环境已验证。

## 1. 先保护优化表达，再分区

`CoreMLPartitioner`通过coremltools的FX节点支持查询，再加项目自身的参数/已知问题guard，交给CapabilityBasedPartitioner构造尽可能大的合法分区；允许single-node partition。

`ops_to_not_decompose`将后端可接管的高层op保留到lowering，例如避免在通用导出阶段先把SDPA拆成更多基础op，丢掉后端优化入口。它还排除view op和部分已知lowering/perf不利的目标；不是不分条件地保留所有高层op。

这与ORT的“补小op来连接重计算分区”互补：**降低得太早可能丢大融合，支持得太碎又增加交接。** 原有`to_edge()+to_backend()`路径在源码中有明确deprecated提示，新路径建议是`to_edge_transform_and_lower()`；本轮未对二者计性能。

support还取决于参数：读本拒绝argmax/argmin的dim=None，明确dim不因此全部被禁；拒绝add/sub alpha!=1和特定已知不支持目标；rand被排除不代表randn等所有随机op都被排除。这些是该commit/coremltools上下文的资格规则，不是永久CoreML硬件能力表。

## 2. partial/full graph与state takeover是耦合选项

默认lower_full_graph=false，可把不支持部分留给外部ExecuTorch执行。它不是承诺剩余节点自动走优化Metal；actual backend/kernel取决于宿主计划。

full graph时，constructor要求无skip-op列表，接管constant和mutable buffer；unsupported op会抛出导出错误，不是等到手机上悄悄回CPU。enumerated-shape compile spec也要求full graph。

接管mutable buffer会将其作为CoreML state，所读路径提示需macOS15/iOS18+。它改变状态归属，而不只是一项性能flag；取消、请求隔离、预热和prefill/decode之间的状态语义都要随之确认。

## 3. AOT资产、加载配置与枚举shape

编译入口使用coremltools转MLProgram。`generate_compile_specs`默认compute-unit ALL、compute-precision FP16、model-type MODEL；不能将示例CPU_AND_NE或预编译选项视为默认。

MODEL保留.mlpackage供runtime编译；COMPILED_MODEL需要加载后取.mlmodelc。宿主CoreML specification/deployment target、runtime编译、模型加载和首个prediction应分别计时；编译物可加载不等于被ANE执行。

enumerated输入按**对应索引配对**，不是Cartesian product。tokens/mask各有[1,1]、[1,32]时，合法组是两组同序tuple，不应扩成四组交叉组合。源码还核对枚举数、静态维度、symbolic range和可接受dtype；同symbolic shape重复出现时须有一致枚举。多个枚举输入另有iOS18 deployment条件。

这种有限bucket给编译器更明确的形状候选，但不代替runtime ABI或证明任意动态形状都能执行。输出的wrapper resize也不能解除rank/预留存储容量限制。

## 4. prefill/decode多方法共享：共享资产，不共享请求状态

`preprocess_multimethod`的POSITIONAL默认按partition index把各method的.mlpackage组合成multifunction模型，要求各method分区数量一致；不一致应拒绝或选择DISABLED。数量一致也不替代函数/参数语义和实际去重效果的检查，不把position相同当作权重已匹配的证明。

合并模型每个partition只存一次NamedDataStore；各method的processed bytes是带CMJR magic的JSON引用，包含asset key和functionName。runtime检查引用、取得并保活NamedDataStore buffer，并按method metadata恢复相应I/O names；加载MLModel时设置真正functionName。artifact身份、ExecuTorch method名和CoreML function名不能合成一个字段。

这种设计避免为每个method重复放入同一导出资产；本轮没有实测.pte体积或加载内存。**磁盘共享不证明多个MLModel实例的全部resident权重物理共享，更不说明KV/state自动跨method共享**。有state时，所读Model实例各自newState，应用仍需确认跨prefill/decode的状态传递、位置与生命周期。

关联[PR #16552](https://github.com/pytorch/executorch/pull/16552)在2026-10-16的GitHub API状态为closed、merged=false；但固定代码有该功能，文件历史也有带此PR编号的提交。补丁、历史提交和GitHub治理状态分别记录。非空merge_commit_sha或提交标题中的PR号都不足以覆盖API的merged=false，本文不称该PR已合并。

## 5. 宿主张量→MultiArray：借用与转换分开

runtime把ExecuTorch Tensor的shape/真实strides传入MultiArray；scalar临时包装为rank1。`ETCoreMLModel::prepareArgs`若源dtype等于模型constraint dtype，就用原data pointer/strides包装MLMultiArray；借用buffer的owner仍由外层维持。

dtype不同则创建转换buffer，按shape+dtype缓存其存储，MultiArray释放时归还。输入需要copy/convert，输出backing准备不先复制旧输出。缓存去除in-use条目再在deallocator归还，是可借鉴的lease模式，不是“所有临时buffer都能马上复用”。

copy层依次尝试BNNS、同dtype/packed的memcpy、通用strided iterator；这与ORT所读single-gap helper的能力不同。本轮只读了这些入口和分支，不给所有rank/负stride/alias、整数溢出语义取得完整资格。

## 6. output backing是请求，不是采纳证明

manager给MLPredictionOptions提供outputBackings，预测后比较实际数据buffer；同pointer可跳copy，否则通过实际返回layout/dtype复制到目标。与“提供backing就跳所有回读”相比，这保留了fallback路径，但不能将内部复制数、pointer相等时的所有layout不变量或ANE共享行为推断为已测。

预测失败且曾提供backings时，manager将executor的ignoreOutputBackings置为YES后重试；default executor会清空该prediction option，后续该executor也继续忽略backing。它是**同一CoreML执行器换输出分配策略**，不是CPU/GPU模型重算fallback。

若模型有mutable state，重试还需要事务语义：只有能保证失败前未改state，或能恢复snapshot，才能把两次prediction视为一个逻辑请求。读本这个重试点没有状态snapshot/restore；本轮没有证明CoreML会在这类错误前改state。

CPU条件反例假定“第一次先state++、后报错”，无rollback重试得到2，有rollback得到1。它证明需审查的条件，不是上游故障复现。loading queue的序列化与map lookup锁也不是同一model/state的并发执行许可。

## 7. 两条缓存路径要分别阅读

`backend_delegate.mm`当前默认仍是旧asset-manager路径；内部runtime spec `_use_new_cache`才切换新cache。不要把新实现的全部性质归给默认部署。

| 身份层 | 所读实现 | 迁移时关注 |
| --- | --- | --- |
| 导出模型identifier | UUID写进AOT metadata；multifunction每组一个资产 | 不是全内容寻址或每次重导出必命中 |
| 旧compiled asset路径 | 加compute-units后缀，function/method不进入共享资产key | 完整multifunction编一次，加载另选function |
| 新compiled cache路径 | 只按模型identifier共享，加载时传当前configuration | 缓存无CU字段不自动等于错policy |
| loaded MLModel与MLState | 每次按function/config创建模型和state | 不能把共享磁盘asset当可共享请求state |

新cache使用同文件系统的temp目录及move，并有format版本、访问记录；但store中还包含remove-existing再move，不应把单次move的原子性扩大成跨进程完整事务。每identifier的loading queue只覆盖所读进程内加载。没有做并发writer、eviction/活跃asset或磁盘故障实验。

加载失败的某条新cache路径会删除entry，供下一次重新取得/编译；执行失败则向上返回错误。它不自动证明当前请求可以无损重跑或从预编译blob恢复原始模型。

## 8. #21969：tied embedding改变压缩策略的消费者关系

[PR #21969](https://github.com/pytorch/executorch/pull/21969) API显示于2026-08-28合并，三段patch最终形成模块级helper。若同一const被gather和linear等其他op共用，单纯让gather“不量化”、linear“量化”会给同一权重冲突策略，可能导致lowering失败。

读本为这样的shared gather加同一quantizer的op-name override，其余独立gather保留opt-out。这是按**权重消费者关系**处理量化，不是按op名称统一开关，也不是把同名Q4当所有模型都等价。

补丁测试只要求tied/untied两个模型都能完成lowering并生成program，没有验证推理输出、perplexity、ANE INT4执行或性能。现在可被压缩的embedding table还需要模型质量评估，不能把“导出成功”晋升为质量/加速资格。

## 后续验证

先固定导出图、coremltools/ExecuTorch版本、模型/adapter、shape tuples与真实provider路由，检查整个host张量→delegate→consumer。再区分package/compiled cache、stateful prewarm、backing被拒绝、retry、完整请求质量和物理placement。不要因两个框架都使用CoreML就把其cache、owner和GPU command timeline互换。
