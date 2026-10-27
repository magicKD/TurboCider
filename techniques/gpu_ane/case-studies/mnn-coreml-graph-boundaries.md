# MNN CoreML：整图委托、回退与输出所有权的边界

<!-- knowledge
id: technique-gpu-ane-case-studies-mnn-coreml-graph-boundaries
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [Metal拷贝与卷积ABI](mnn-metal-copy-and-convolution-abi.md) · [21文件固定读本](../sources.md#mnn-boundaries) · [第八轮范围](../../MAINTENANCE.md)

2026-10-16，固定 `alibaba/MNN@dbf9ba61b08106a3b43426e295d1a29cb41cc8fe`。这是公开源码的机制研究；没有构建或运行 MNN、CoreML 模型、VAE，也没有故障注入。下文将明确可见的调用链、风险推断和独立CPU反例分开，不把发现可疑代码升级成已复现的上游故障。

## 1. “NPU后端”不等于“强制ANE”

[使用文档][npu]让调用方开启 `MNN_COREML`、链接 CoreML framework，并选择 `MNN_FORWARD_NN`。这个枚举还用于其他平台的 NNAPI，不是物理设备标识。[加载实现][executor]使用 `MLComputeUnitsAll`；Apple 的[公开合同][all]是允许全部可用计算单元，包括 Neural Engine，不承诺每个算子都在 ANE，也不承诺 GPU 与 ANE 同时计算。

[CoreMLBackend][backend]创建的是 specification version 4、`NEURAL_NETWORK` 类型、exact array shape mapping 的模型，不是现代 MLProgram。输入 shape 来自当前 MNN Tensor，resize 时重新构图/编译；这里没有声明 flexible/enumerated shape。**这是该后端、该ref的实现选择，不是 CoreML 框架缺乏动态 shape。**

[Schedule][schedule]的 `AUTO` 按 runtime creator 是否存在及固定优先级选择后端；其中 CoreML 优先于 Metal。没有在这段实现中逐 shape 比较完整消费者的延迟、内存或质量。因此“AUTO选了CoreML”不是“测得ANE最快”。本轮未改任何环境变量或调度默认值。

## 2. 一个执行单元在哪里变成整张CoreML图

关键顺序如下。箭头表示本读本的普通 Pipeline 路径，不是设备并发时间线。

```text
resize / allocation
  CoreML op::onResize → 把layer加入同一个CoreMLBackend模型
  CoreMLBackend::onResizeEnd → buildModel → 序列化/compile/load

run
  Pipeline逐项调用op::onExecute
    CoreMLCommonExecution → 直接NO_ERROR，不做逐op预测
  Pipeline::_exitExecute
    CoreMLBackend::onExecuteEnd → invokeModel → 同步CoreML prediction
```

来源：[CommonExecution][common]、[Backend][backend]、[Pipeline的resize与execute][pipeline]。把多个受支持层留在同一模型内，避免逐层跨框架往返，是可迁移的图级思路；但“一个op存在Execution对象”不等于该op在 `onExecute` 当下已产出数值。

这也影响 debug callback：围绕空的逐op `onExecute` 取时间或读Tensor，不能当成 CoreML 中该层的实际设备运行时或完成事件。要测的是整个同步预测、实际跨边界转换和消费者；逐层 CoreML placement 另需对应工具。

## 3. 四种“回退”必须拆开

| 情况 | 选定实现可见的动作 | 不能据此推导 |
| --- | --- | --- |
| 请求后端没有runtime creator | Schedule尝试 `config.backupType` | 并非CoreML预测失败后自动切Metal |
| 某op创建Execution返回空 | Pipeline `_createExecutions`尝试CPU backup | 不自动证明任意CoreML/CPU小岛可交错执行 |
| CoreML构图/编译失败 | `buildModel`返回 `NO_EXECUTION`，普通allocation/resize调用链返回错误 | 这里不是自动重建完整GPU图并重试 |
| CoreML运行时预测失败 | Objective-C方法返回false，wrapper打印日志后void返回 | 不能保证该失败传给最终Session返回值或触发恢复 |

来源：[Schedule][schedule]、[Session][session]、[Pipeline][pipeline]、[wrapper][wrapper]和[executor][executor]。Session中的备用执行后端由 CPU runtime 创建，不能把 `ScheduleConfig::backupType` 和逐op CPU backup 混成同一个策略。

### 为什么普通fallback代码不足以证明安全的子图分区

本读本看到以下组合：

- CoreML backend在一个终端hook中预测整图，逐op execute为空；
- `WrapExecution::needWrap`对目标 `MNN_FORWARD_NN` 直接返回false；
- CoreML的输入/输出map按Tensor的INPUT/OUTPUT usage收集，不因某个CPU不支持节点自动导出中间边界；
- 选定Pipeline命令循环中没有按“CoreML producer→CPU consumer”依赖插入预测flush、导出中间feature，再开始下一张CoreML模型的显式步骤。

来源：[WrapExecution][wrap]、[Backend][backend]及[Pipeline][pipeline]。因此，**不能仅凭“Try Backup”就宣称它具有已验证的任意混合分区执行能力**。这与 [ORT显式greedy分区](onnxruntime-coreml-partition-cache.md)是不同证据深度。显式配置的多个Pipeline也由 [Session][session]依次执行，不自动成为并行调度器。

CPU合同给一个理想化反例：输入3，CoreML应算 `t=x+1`，CPU消费 `2t`；若CPU在终端预测前消费初值0，则输出0而非8。它特意假设中间存储有效，只展示“缺少完成边界”的逻辑后果；没有证明这个图能在原MNN成功创建、编译或到达该执行状态，更没有运行它。

### 运行时错误有一处明确的类型边界

`invokeWithInputs`的bool失败被 `CoreMLExecutorWrapper::invokModel` 的void接口截断；后者只打印。Backend终端hook同样void；普通Pipeline在退出hook后返回 `NO_ERROR`，没有读取这次预测的bool。条件是没有其他前置/算子错误；这里不是说所有失败都静默、所有Session总成功。

保存的CPU见证只是按这个控制流说明状态为何可能丢失。没有真实CoreML错误、损坏输出、设备错误或重试测试。迁移时应让预测失败、output validity与消费者generation一起向上传播，并定义是否允许重建/回退；日志不是可组合的错误合同。

## 4. 普通I/O：host借用与owner，不是GPU零拷贝

[Executor的输入provider][executor]从Tensor shape反向乘出dense strides，并用外部 `host<float>()` 构造FP32 MLMultiArray，deallocator为空。**只说明这个包装步骤借用host存储**；不证明CoreML内部不转换、不复制，也没有传入MTLBuffer、IOSurface或跨队列事件。

[Backend的copy][backend]对NC4HW4输入暂以CAFFE/NCHW视图解释目标host区域，再通过 `MNNCPUCopyBuffer`转换。元数据叫NC4HW4，不代表交给CoreML的host字节仍是C4打包。输出方向做对应的临时NCHW视图转换；Metal侧的不同物理ABI见[下一篇](mnn-metal-copy-and-convolution-abi.md)。

普通输出路径更特殊：

1. 从返回的feature中取MLMultiArray；
2. 把它加入 `_outputArray`（header声明strong owner）；
3. 直接把MNN Tensor的host指针改成其 `dataPointer`，不在此处复制成MNN拥有的快照。

每次invoke开头先清空旧 `_outputArray`。所以Tensor裸指针本身不持有输出对象；下一次invoke或executor销毁可能结束旧owner的保活。不能在跨请求异步消费者仍在读时，无条件开启下一次预测并假定旧结果稳定。即使某次地址没有变化，也不提供跨代内容不变的保证。这里未验证同executor并发线程安全或实际释放时刻。

### dataPointer之后还有stride合同

选定输出分支未把返回MLMultiArray的实际shape/strides/dtype映射到Tensor，后续普通拷贝按MNN自身布局处理。Apple的[strides定义][strides]是**元素地址增量**，不是字节数。

离线反例使用shape `[1,3,1,2]`、strides `[12,4,2,1]`，逻辑值0到5。按真实stride读取正确；直接读前6个元素得到 `[0,1,-999,-999,2,3]`。这是人为构造的合法padded backing，不是观察到CoreML在这个模型上返回了该布局。它说明：安全复用需证明目标模型实际dense，或显式执行strided gather/copy；不能从裸指针推出dense。

与 [ExecuTorch](executorch-coreml-delegation.md)的实际backing采用检查、general copy和重试相比，本路径没有设置 `outputBackings` 的代码。两者不宜笼统归类成“统一内存，所以零拷贝”。

## 5. Precision_Low不是任意Tensor的INT8快路

[枚举定义][types]中 `Precision_Low=2`。这个CoreML路径用它开启**8-bit image I/O**：Backend的float I/O计1字节、feature类型改成image，输入输出copy走uint8 memcpy；不是给通用神经网络自动生成W8A8整数矩阵图。

两处实现还有待验证的兼容性差异：

- `setIO`按channel>1声明RGB，否则grayscale；
- provider却固定创建 `OneComponent8`，row bytes仅width；image输出分支也每row只复制width字节，虽然读了实际bytesPerRow。

来源：[Backend][backend]和[Executor][executor]。单通道和三通道的独立字段对照见 。单通道字段相符也不证明模型可用，batch、预处理、范围与输出格式尚未执行；三通道差异不是本机已复现的CoreML拒绝或坏图像。源码本身另有低精度NC4HW4 TODO。VAE的多通道浮点latent不能直接套用这个开关。

## 6. 卷积与形状：图内保留很好，权重语义要继续追

[CoreMLConvolution][conv]在resize中把group、stride、dilation、deconv输出形状、padding及bias写入CoreML layer；ReLU/ReLU6作为后接clip层，仍属于同一CoreML模型，并不因此发生一次host往返。某些CAFFE padding另加padding layer或选择SAME/top-left-heavy；是否被底层融合、是否放在ANE不由这些builder字段决定。

两个容易错误迁移的细节：

- 读到压缩卷积参数时调用load并使用 `weightFloat`，随后复制到protobuf的floatvalue。MNN文件曾经量化，不能据此说这张CoreML图仍用原始整数权重执行。
- 三输入卷积从input[1]/[2]读取host权重/bias，但仍在resize时复制嵌入，卷积layer只绑定激活输入。它不是 [vpipe](vpipe-runtime-weights.md)那种每次predict传入runtime weight的合同；若权重/LoRA变化，必须另验重建/resize，不能只更新host数组就假定已加载模型改变。

[Raster][raster]能识别部分reshape/transpose/pad/concat等结构；未命中时写成 `RasterLayer` custom layer。[选定custom实现][executor]提供CPU evaluate与region拷贝，没有实现相应GPU encode方法。图被CoreML接受不代表没有CPU段；自定义几何层也不能算作“全ANE图”。本轮没有审计所有Raster builder、padding数值或完整VAE。

另一个保留的静态检查点：[CoreMLMatMul][matmul]可选bias分支创建 `biasLayer`，但把ADD类型与参数写回 `mLayer_`。可见的目标对象不一致值得隔离验证；未构建、未检查CoreML实际报错，不把它写成已复现的matmul错误或已修复缺陷。

## 7. 冷启动、缓存与后续验收

选定[Executor][executor]把protobuf写进唯一临时文件，再调用CoreML compile与load；[wrapper][wrapper]成功后cleanup源模型和编译目录。Backend的runtime cache getter仍是初始空指针/零长度。这里未实现像ORT那样显式、持久、由调用者管理key的compiled artifact cache；**不等于系统内部绝无缓存**。没有清理系统缓存，也没有测cold load。

接入实验前应分别取得：构图/序列化/编译/加载时间、稳态预测、真实输出layout与owner、边界copy/等待、完整消费者误差。固定权重、bias、LoRA、shape/图像tile、精度和GPU基线；验证unsupported/compile/predict失败的明确恢复路径。做到这些之前，本页只授予“已读实现与风险清单”，不授予VAE加速、全ANE或GPU+ANE并发资格。

[backend]: https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/coreml/backend/CoreMLBackend.cpp
[executor]: https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/coreml/backend/CoreMLExecutor.mm
[wrapper]: https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/coreml/backend/CoreMLExecutorWrapper.mm
[common]: https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/coreml/execution/CoreMLCommonExecution.cpp
[conv]: https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/coreml/execution/CoreMLConvolution.cpp
[matmul]: https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/coreml/execution/CoreMLMatMul.cpp
[raster]: https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/coreml/execution/CoreMLRaster.cpp
[pipeline]: https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/core/Pipeline.cpp
[session]: https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/core/Session.cpp
[schedule]: https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/core/Schedule.cpp
[wrap]: https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/core/WrapExecution.cpp
[types]: https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/include/MNN/MNNForwardType.h
[npu]: https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/docs/inference/npu.md#coreml
[all]: https://developer.apple.com/documentation/coreml/mlcomputeunits/all
[strides]: https://developer.apple.com/documentation/coreml/mlmultiarray/strides
