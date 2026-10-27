# ONNX Runtime Core ML EP：分区质量、缓存身份和真实交接

<!-- knowledge
id: technique-gpu-ane-case-studies-onnxruntime-coreml-partition-cache
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [24文件/PR来源](../sources.md#coreml-delegates) · CPU合同 · [ExecuTorch对照](executorch-coreml-delegation.md)

固定 [`microsoft/onnxruntime@f33c1744e1e1064d3618d1d0ce533f0b946aed2d`](https://github.com/microsoft/onnxruntime/tree/f33c1744e1e1064d3618d1d0ce533f0b946aed2d)，2026-10-16选读12个文件。没有构建/运行ORT、加载ONNX模型或进行GPU/ANE benchmark。官方[使用入口](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)只作配置导航，以下实现结论以固定源码为准。

## 1. EP是子图委托，不是自动双设备并行器

`coreml_execution_provider.cc::GetCapability`查询op builder，按模型格式、opset、输入信息和具体算子限制得到supported nodes，再交给通用partitioning工具。`Compile`把每个接管分区变为一个CoreML Model对象，compute callback调用它的同步Predict。

`partitioning_utils.cc:60–255`使用贪心拓扑wave：supported节点可入当前组；消费当前组结果的unsupported节点成为下一组边界。组内不要求所有节点相连，但不能跨越会造成不合法依赖的unsupported节点；node-unit模式还要求完整QDQ单元。它不是简单按文本相邻或连通分量切图。

未被接管的节点由ORT其他已注册EP处理，常见是CPU；不能直接称“余下自动使用Metal”。执行失败返回Status，不在所读compute callback中把整个分区自动切回另一个设备重算。**图分区时的fallback和执行出错后的恢复不是同一条路径。**

每个Model的Predict外围有独占mutex，锁覆盖输出信息准备和预测/回读。它明确序列化同一Model的这些调用，但不能据此断言所有分区/设备/整个进程都串行，也不能凭多个EP注册就声称它们重叠执行。

## 2. #28293：补算子覆盖，同时拒绝独立的廉价小岛

[PR #28293](https://github.com/microsoft/onnxruntime/pull/28293) 的API记录显示于2026-05-20合并；四段patch中，第一段添加Identity/Ceil/Tile及trivial-only分区过滤，第三段把硬编码集合改为各builder的`IsTrivial(node)`。当前读本采用后者，默认marker=false，某些多op builder按具体node回答。

这两个动作必须一起理解：

```text
Conv → Identity/Tile/Ceil → Conv
  算子覆盖可减少图断点，并保留成有重计算锚点的分区

独立 Identity/Cast/Reshape/... 小岛
  即使技术上支持，也可能整个分区都不值得委托
```

源码在group关闭时要求至少一个non-trivial节点。不是全局禁止这些op，也不是“Tile对任意shape都便宜”的硬件定律。大tensor、expansion、共享带宽和不同CPU实现仍需成本模型。

作者给出M3 Max、YOLOv10n、MLProgram下的局部收益；本轮未复现，不把其数字迁到M4 Max或其他模型。补丁中的测试刻意固定优化级别，避免Identity先被优化消除、导致根本没测到partition heuristic。迁移时同样应确认被测变换真正进入了候选路径。

## 3. 四种compute policy仍不是物理放置

所读命名配置支持ALL、CPUOnly、CPUAndGPU、CPUAndNeuralEngine。`model.mm::LoadModel`在加载时将它们映射到MLModelConfiguration；source header也明确ANE相关flag不保证整个模型只在ANE运行。

实际defaults来自`coreml_options.h`：static-shape要求false、子图递归默认关闭、MLProgram=false（默认NeuralNetwork）、compute-units=ALL、低精度GPU累加=false。不要把文档示例中显式设置的MLProgram/ANE策略误当所有调用的默认值。

这是所读options类的字段默认，不代替外层wrapper可能显式传入的配置。

`ProfileComputePlan`只对MLProgram生效。实际实现读取preferred device和estimated cost；不是实时硬件执行轨迹，不能沿用header里“dispatched hardware”的简写作为实测证据。诊断会进入额外plan加载/等待，不能无条件混入正式性能计时。

CPUOnly也只是设备policy，不会恢复已经量化或改精度的模型权重/计算表达；它不是原FP32 checkpoint的自动精确oracle。

## 4. 输入借用、类型桥与输出复制

`CreateInputFeatureProvider`把ORT原buffer包装为MLMultiArray，deallocator不接管原内存；支持的直接dtype路径可避免这层host副本，但调用方/ORT上下文仍要保活。它构造的是dense strides，不是任意Metal view或IOSurface导入。

额外边界包括：

- int64输入收窄到int32，并做范围检查；不是按位reinterpret或静默截断。
- bool输入转换成int32 0/1；输出int32可再转换成bool。
- scalar用[1]的MultiArray表示，返回ORT时恢复[]。
- dynamic输入在runtime变成零元素时被明确拒绝；不承诺该case自动fallback。

另追到`include/onnxruntime/core/common/narrow.h`：正常异常构建使用GSL checked narrow；禁用异常时，范围失败会打印错误并`std::terminate`。因此不能把这类不合法输入统称为“返回Status后自动fallback”；CPU例子只检验范围数学，没有运行上游进程的终止路径。

输出通过prediction返回的MultiArray分配/填写ORT输出buffer，再做copy或类型转换。所读路径没有向MLPredictionOptions提供outputBackings，不能宣称是直接写入ORT buffer的普遍零拷贝。

`GetMLMultiArrayCopyInfo`还只接受一种uniform block-copy布局：至多一个不连续stride断点。例如：

| shape | source element strides | 这条copy路径 |
| --- | --- | --- |
| [2,3,4] | [12,4,1] | 一个24元素连续block |
| [2,3,4] | [16,4,1] | 两个12元素block，source stride16 |
| [2,3,4] | [32,8,1] | 两处padding断点，被拒绝 |

最后一行可由一般多维address oracle读取，并非数学上非法；只是当前这条实现没有通用gather fallback。不要把“支持非连续输出”扩成“支持所有strides”。输出shape还按runtime实际shape替换-1，并核对rank和固定维度，不以元素总数相同就放行。

## 5. 模型缓存不是内容寻址保险箱

`GetCapability`生成名称；`GetModelOutputPath`取key和partition id，加上static/dynamic、mlprogram/nn后缀。metadata中的COREML_CACHE_KEY优先；无key时使用路径或图名字相关hash。公开header直接说明**不会检测ONNX模型已变化**，由用户管理缓存失效。

`model_builder.cc:415–556`看见已存在的model目录/文件便认为cached；`model.mm`另查compiled_model.mlmodelc。换权重、LoRA、导出图/分区布局或其他语义后仍用同key/路径，不能靠“hash”这个词取得内容身份保证。

CPU见证只展示同user key/options slot没有权重内容这个参数：改权重仍可能指向同slot；它没有执行ORT真正的hash算法、cache或模型，不称本机stale模型故障复现。

需要区分两层：编译artifact缓存未含compute-units，不自动意味着错policy，因为LoadModel仍使用当前配置创建MLModel。真正要审计的是内容身份、编译上下文、加载配置和状态owner分别归谁；不能把一个cache key机械用于所有层。

## 6. 对TurboCider有用的设计与验证顺序

先数清partition边界及每条张量的dtype/shape/bytes，再比较整个消费者；op覆盖率不是优化目标本身。cache key绑定checkpoint/adapter/转换与分区版本；直接借内存也要满足实际consumer ABI和完成条件。

建议后续固定一个encoder/FFN图，逐项对照partial支持、补齐中间op、trivial过滤、cold/cache-hit和完整输出。先看实际路由与copy，再对优化GPU基线计时。本轮只是源码/CPU合同，不能据此给ORT、MLX或ANE排序。
