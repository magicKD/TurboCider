# LiteRT V2 Metal：请求异步、事件可达性与真实buffer成本

<!-- knowledge
id: technique-gpu-ane-case-studies-litert-v2-metal-events-and-buffers
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [ML Drift编码/布局](mldrift-metal-runtime.md) · [原始事件类探针](litert-original-event-probe.md) · [来源身份](../sources.md#litert-metal-runtime)

2026-10-16，固定LiteRT `b0983dc...`及其ML Drift依赖 `1b18771...`，完整40位身份与行范围在来源清单。这里是源码分析；原始event-class的小执行另列，未运行Compiled Model、完整delegate或模型。本页不把“存在异步方法”当作选定Metal路径已实测非阻塞。

## 1. 请求async、实际返回async与物理并行分开

选定[Swift dispatch][swift]调用新的C Async API；虽然Swift局部isAsync初值false，[C实现][capi]会先设true再调用runtime。这不是可以凭Swift变量初值判定的同步fallback。

[CompiledModel runtime][runtime]把请求传给buffer context，执行interpreter/signature runner后，再把返回async重算为“是否任一输出buffer携带event”。同步调用则对已附着的输出event做无限等待。这个返回值不是所有输出已ready，也不是GPU与ANE同时工作；已有旧event也不能单独证明本次生成了新的完成事件。

### 必须追到具体Metal上下文：一个静态可达性检查点

所读[Metal context header][context_h]把private `post_dispatch_event_`初始化为nullptr；[构造器][context]仅调用base constructor。完整选定class实现中：

| 步骤 | member为null时的分支 |
| --- | --- |
| PostConvert(output) | 调用WaitForCompletion，不创建event |
| 创建新CustomEventMetal的赋值 | 位于已经非null的else分支 |
| GetPostDispatchEvent(async=true) | null时返回NotFound |
| HandleOutputEvents请求async且NotFound | 记录不支持async并正常返回 |

因此，沿这个class的普通构造/调用路径，null状态不会自己进入创建事件的分支。新输出且没有其它事件来源时，不能从API名推导会附着Metal完成event。这个判断来自私有member及全部写入点，是**固定源码的可达性分析**，不是已运行LiteRT模型的故障复现，也不推广成其它backend/版本没有async。

本轮CPU状态见证只复述这个分支不变量；未强行设置私有member、修补上游或宣称修复。下文事件机制解释的是接口/条件分支怎样工作，不能越过上述可达性问题替普通Metal路径取得异步资格。

## 2. 条件分支中的事件桥如何设计

[输入处理][kernel]收集buffer上的事件，交给具体context。Metal的force_sync=false会把wait编码到command buffer，并按event对象去重；force_sync=true则调用host Wait。由GPU队列等待依赖，可避免CPU逐边界阻塞，但前提是event、threshold、queue、owner都正确。

在已经存在post-event的条件分支中，[PostConvert][context]结束output转换encoder，释放旧context引用、创建新的CustomEventMetal并编码signal；GetPostDispatchEvent提交command buffer并返回该event。[backend bridge][context]再把它装到LiteRtEvent/TensorBuffer。每代使用新对象是避免旧观察者的等待目标被随下一次提交修改的一种设计；并非本轮已证明普通入口会到达它。

`GetNative`也不是无类型共享fence：选定CustomEventMetal返回的是wrapper自身指针，不是裸MTLSharedEvent；对端必须理解这个custom-event ABI，不能将任意OpenCL/FD/CoreML事件强转成Metal对象。

## 3. wait返回、ready与成功是三个状态

[CustomEventMetal][event]的Wait callback是void：在支持的系统上调用有限/无限MTLSharedEvent wait，有限超时后可能仍未signaled。header明确允许再次等待。[LiteRtEventT::Wait][runtime_event]的custom分支调用此void callback后返回success，与FD分支的TimeoutExpired不是同一种超时状态合同。

所以有界调用方不能只检查“Wait API返回OK”，还应核对IsSignaled以及目标代次。这个语义由[原始类探针](litert-original-event-probe.md)实际验证：有限wait确实返回但未ready，host显式发信号后才ready；未执行LiteRtEventT自身，runtime层结论仍是源码证据。

ready也不等于数学结果有效：[base Metal WaitForCompletion][backend]在选定active/passive路径中等待/提交后直接返回OK，没有在该函数内将command-buffer error转成推理错误。callback发出、shared-event阈值到达、设备错误、输出误差及state事务需分别验收。本轮没有设备错误注入或取消/drain测试。

## 4. Buffer import真正复用了什么

[buffer handler][handler]保存MetalSpatialTensor，以及由ARC保活的device/queue；import根据buffer type把外部MTLBuffer或MTLTexture包成shared tensor。import字段 `owns_tensor=false`不应被解释成“所有资源都没有owner”：需看包装对象、ARC引用和调用者剩余责任。

[buffer utils][utils]会依rank构造BHWC：rank2是 `[dim0,1,1,dim1]`，rank1普通非零长度变成 `[dim0,1,1,1]`，与旧delegate一律右对齐的GetDims不同。逻辑float32配FP16 GPU memory时，descriptor物理dtype可改成half。不能跨两条路线复用同一个rank/stride或itemsize假设。

外部资源绑定后，[ML Drift SetTensor](mldrift-metal-runtime.md)仍要更新关联operation的资源参数。导入一块同大小buffer不证明offset、layout、精度、生产者完成事件或所有下游消费者都兼容。

## 5. host lock不是把MTLBuffer.contents原样借给应用

[TensorBuffer::Lock][tensor_buffer]先等待挂在buffer上的event（非AHWB），然后调用custom handler。选定[Metal lock/unlock][handler]还有内部host staging及descriptor转换：

| 操作 | 选定实现的成本 |
| --- | --- |
| 第一次host lock | 按packed_bytes分配对齐的host staging |
| write-only lock | 不下载旧数据，但仍返回staging |
| read/read-write lock，packed buffer | shared storage可直接memcpy；private storage需GPU blit、wait再copy |
| 普通typed buffer/texture读取 | ToDescriptor，再按逻辑元素类型DownloadData |
| read-only unlock | 不上传 |
| write/read-write unlock | ConvertDataToDescriptor后上传；shared buffer的优化分支直接CPU memcpy到实际GPU backing，否则走UploadDescriptorData |

shared-buffer写入优化省掉了某些临时queue、blit和wait，却没有消除host staging、格式/精度转换或CPU memcpy。旧GPU reader若还在读，直接写contents仍需完成依赖；不能因为统一内存就忽略最后reader/slot代次。

直接shared-buffer读取helper本身不再提交空command buffer，依赖更上层事件或调用者保证ready。DoNotWait、外部裸指针读写和同步API并非可互换的完成合同；本轮没有将这些路径实机接成模型生产者/消费者。

## 6. 真正的模型执行顺序与“零拷贝”分母

[Metal Invoke][delegate]的顺序是：必要时dequant→bind external buffers→上传中间CPU tensor→处理输入event→转换非external输入→dispatch→转换非external输出→处理输出event→必要的等待/中间CPU下载→quantize。external模式可绕过部分转换，但其它路径仍存在。

比较V2与旧CoreML delegate、MLX或自写Metal时，应分别报告：

- GPU-resident内部执行，不含host读写；
- 实际host lock/write/unlock到消费者可读的完整组件；
- 含模型加载、编译、shape变化与质量的完整request。

本轮无以上推理计时，没有把事件timeout的约束实验当GPU加速。将来还需核查完整async可达性、输入输出双代事件、旧event清理、多个in-flight slots、shared memory manager并发、state/取消与错误回收。ANE协同仍需另一套模型/资源/placement证据。

[swift]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/litert/swift/Sources/LiteRT/CompiledModel.swift
[capi]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/litert/c/litert_compiled_model.cc
[runtime]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/litert/runtime/compiled_model.cc
[context_h]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/ml_drift_delegate/delegate/gpu_backend_metal_litert.h
[context]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/ml_drift_delegate/delegate/gpu_backend_metal_litert.mm
[kernel]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/ml_drift_delegate/delegate/delegate_kernel_litert.cc
[event]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/ml_drift_delegate/delegate/custom_event_metal.mm
[runtime_event]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/litert/runtime/event.cc
[backend]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/ml_drift_delegate/delegate/gpu_backend_metal.mm
[handler]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/ml_drift_delegate/delegate/buffer_handler_metal.mm
[utils]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/ml_drift_delegate/delegate/buffer_handler_utils.cc
[tensor_buffer]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/litert/runtime/tensor_buffer.cc
