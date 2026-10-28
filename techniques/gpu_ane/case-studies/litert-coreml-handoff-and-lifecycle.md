# LiteRT旧Core ML delegate：真实I/O、错误返回与析构修复

<!-- knowledge
id: technique-gpu-ane-case-studies-litert-coreml-handoff-and-lifecycle
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [分区与FP16常量](litert-coreml-partition-and-fp16.md) · [来源/PR身份](../sources.md#litert-coreml)

2026-10-16，固定 `google-ai-edge/LiteRT@b0983dc2dc27141a64fba7553f40299631e65937`。以下是旧CoreML delegate的源码机制与独立CPU见证，没有运行LiteRT/CoreML模型、Apple GPU/ANE或上游benchmark。不能把新Compiled Model API的异步/互操作宣传转移给这个同步旧delegate。

## 1. 子图kernel有明确的预测边界

[CoreMlDelegateKernel][kernel]将一个被替换子集封装成一个runtime op；Init阶段构图、save、compile/load，Prepare分配持久input/output `std::vector<float>`，Invoke完成实际预测和张量转换。不同于[MNN的终端整图hook](mnn-coreml-graph-boundaries.md)，这里替换节点的Invoke本身就是子图执行/完成边界。

输入/输出数据流是：

```text
TFLite host NHWC tensor
  → CPU transpose（必要时先FP16→FP32）
  → kernel拥有的dense FP32 NCHW向量
  → MLMultiArray包装此向量 → 同步prediction
  → 从返回MLMultiArray flat memcpy到kernel输出向量
  → CPU transpose回NHWC（必要时FP32→FP16）
  → TFLite输出tensor
```

来源：[kernel][kernel]与[executor][executor]。有共享物理内存，不代表以上逻辑步骤消失。多次小岛往返会重复承担host转换和预测边界；单看ANE层内时间，不能判断比完整GPU路径更快。

### rank/维度与权重布局不是同一个问题

kernel的GetDims把rank<=4右对齐为NHWC，缺少的前导维度补1，然后按 `[0,3,1,2]` 转成NCHW；输出用逆排列 `[0,2,3,1]`。例如二维 `[3,7]`在这个helper中解释为 `[1,1,3,7]`，不是任意batch×feature语义。

CoreML2的外部array shape只写CHW，CoreML3写NCHW；不应把缺省batch包装当作所有模型的batch支持。独立CPU检查覆盖48种正4D shape逐值往返，另检查较低rank；它没有执行库内optimized transpose或任意rank的模型。

[卷积builder][conv]的weight transpose又是另一套：普通卷积/TransposeConv读本按OHWI→OIHW，depthwise按 `[1,H,W,Cout]→[Cout,1,H,W]`；FP16分支移动uint16 payload，不先求FP32数值。CPU测试只检查这种排列保持bit pattern，包括负零/特殊位型的整数表示，不是FP16算术或真实卷积正确性证明。

## 2. FP16权重直通，不等于整条I/O是FP16

kernel内的TensorData仍使用FP32向量；[provider][executor]用dense strides构造FP32 MLMultiArray，并以空deallocator借用向量storage。它借用的是**已经转置/扩展后的内部host向量**，不是原始GPU buffer，也不是原TFLite tensor无复制地直达ANE。

Invoke中虽然有FP16激活扩展/压缩分支，外层[支持性检查][delegate]通常先要求激活入口FP32；不能只看某个分支就给全部FP16模型取得资格。常量FP16权重、FP16激活transport、内部算术精度和物理ANE执行仍是不同事实。

delegate注册的CopyFrom/CopyTo/FreeBufferHandle均为空，flags为None；实际Invoke调用同步prediction。选定路径未使用MTLBuffer、IOSurface、Metal fence或outputBackings。若后续要连接TurboCider GPU消费者，需要明确的新bridge和owner/stride/完成合同，不能把这些成本藏在“借用指针”四个字里。

## 3. 输出copy解决一类owner问题，未解决stride问题

与MNN借用返回MLMultiArray.dataPointer不同，这里[executor][executor]立即把结果memcpy到kernel自己的output vector，再转置写回runtime tensor。这个内部快照不再依赖CoreML返回对象持续保活；但runtime输出本身仍可在下一次Invoke/resize被覆写/重新分配，不能供异步消费者无限期保存裸指针。

选定memcpy分支没有依据返回MLMultiArray的实际shape、strides或dtype做general gather/转换，而是按预先分配的float数量平拷贝。安全性仍需目标模型实际返回的布局证据。

CPU反例人为放置逻辑NCHW shape `[1,3,1,2]`、element strides `[12,4,2,1]`；正确值0到5，flat memcpy却包含padding里的-999。**不是观察到CoreML实际返回了此布局，也不是已复现LiteRT输出错误。** 复制成owned buffer和正确解释布局是两个独立检查。

## 4. 回退与错误：图改写失败、预测失败分开

选定[registration][delegate]的Init/Prepare/Invoke均检查kernel有效性并返回TfLiteStatus。[kernel Invoke][kernel]看到executor预测返回false，就返回 `kTfLiteError`，不会继续做输出转置。这比void日志接口保留了更多可组合的错误信息。

这不等于无条件恢复或事务：

- 不支持节点：runtime保留在原执行计划中，所选合法子集被替换；
- delegate应用/静态分配失败：[Subgraph][subgraph]尝试RemoveAllDelegates并恢复原计划/内存，成功恢复后仍返回DelegateError；恢复步骤本身也可能失败；
- 运行时prediction错误：选定delegate/kernel返回Error，没有在其中自动重建CPU图并重新运行的代码；更高层是否另有显式fallback不在此处取得资格；
- [配置plugin][plugin]的GetDelegateErrno恒为0，不能拿它替代Invoke状态或输出有效性。

本轮没有测试半写输出、state mutation、retry、取消或设备错误。尤其不能用“返回了Error”推导所有资源已drain、所有输出保持上一代、或state可以无副作用地重跑。

## 5. PR #9762：路径owner与析构也属于可靠性合同

[LiteRT #9762][pr]的GitHub API在2026-10-16观察为closed、**merged=true**，merged_at为 **2026-09-04T21:43:19Z**；merge/patch commit `cad523c157861c4256f845d7f47e4bff308138ad`。实际读取一段152行patch，涉及四文件；来源清单分别保存metadata与patch SHA，不依据搜索索引日期或非空merge_commit_sha猜状态。

作者报告的问题是Foundation路径参数/对象寿命在清理时触发异常，且C++析构里未处理异常会终止进程。本轮没有复现该崩溃；可见修复机制是：

1. model用strong，两个路径property用copy，并通过setter赋值，跨autorelease pool保留有效对象；
2. saveModel早期就记录临时路径，使后续编译失败时也能找到源文件；
3. cleanup检查路径长度和文件存在性，分开处理两个删除结果，成功后清空对应property；
4. kernel析构对cleanup加Objective-C异常捕获并记录错误，避免异常直接越过析构边界。

来源：[PR][pr]、固定ref的[executor header][exec_header]、[cleanup实现][executor]及[kernel析构][kernel]。这不是“catch所有错误就恢复推理”：异常捕获对象、未成功删除的资产、运行中的prediction、队列drain与应用输出事务都需分别核查。也没有可继承的速度倍率。

LiteRT这段路径在kernel析构时清理临时源模型和编译目录，与MNN成功加载后立即cleanup的生命周期不同；本轮没有测其磁盘占用、系统内部缓存或load耗时。唯一临时文件+compile/load不能自动当作稳定内容key的持久cache。

## 6. 与三种已读委托实现对照

以下只比较各报告固定ref的选定路径，不是框架全局结论或性能排名。

| 路径 | 图执行边界 | 选定输出交换 | 需要继续验证 |
| --- | --- | --- | --- |
| [LiteRT旧CoreML](litert-coreml-partition-and-fp16.md) | 每个替换子集kernel的同步Invoke | flat copy到内部向量，再CPU transpose到runtime tensor | 实际strides/dtype、边界成本、静态resize重编译 |
| [MNN CoreML](mnn-coreml-graph-boundaries.md) | backend终端hook整图predict | 借用MLMultiArray裸指针，以强数组保活当前输出 | 跨代owner、stride、混合fallback与void错误边界 |
| [ORT CoreML EP](onnxruntime-coreml-partition-cache.md) | 明确分区model compute，选定model锁 | host copy与受限single-gap布局处理 | unsupported layout、缓存身份、narrowing失败 |
| [ExecuTorch CoreML](executorch-coreml-delegation.md) | delegate method/model executor | 尝试outputBacking，核实实际返回storage，否则copy/重试 | sticky retry、state事务、动态输出及shared-asset/resident区别 |

可迁移经验：不要把四者都写成“CoreML自动ANE+统一内存零拷贝”。分区算法、边界缓冲、owner、stride、错误与shape策略会改变整个成本/可靠性合同。正确比较须用同一真实消费者、相同权重/精度与质量门槛，统计compile/load、热预测、所有copy/转换、峰值驻留、失败路径和完整request。

## 7. 本地学习复算

这些命令只用标准库，没有LiteRT/TensorFlow/CoreML依赖或设备dispatch。全部24项离线测试通过不等于实际图已delegate、更不等于取得ANE placement、整数MAC、GPU+ANE物理overlap或VAE质量证据。完整范围见[第九轮](../../MAINTENANCE.md)。

[kernel]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/coreml/coreml_delegate_kernel.mm
[executor]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/coreml/coreml_executor.mm
[exec_header]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/coreml/coreml_executor.h
[delegate]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/coreml/coreml_delegate.mm
[conv]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/delegates/coreml/builders/convolution_op_builder.cc
[subgraph]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/core/subgraph.cc
[plugin]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/tflite/acceleration/configuration/coreml_plugin.cc
[pr]: https://github.com/google-ai-edge/LiteRT/pull/9762
