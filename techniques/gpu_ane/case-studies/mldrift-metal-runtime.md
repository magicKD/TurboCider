# ML Drift Metal：精确依赖、实际编码与内存规划

<!-- knowledge
id: technique-gpu-ane-case-studies-mldrift-metal-runtime
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [V2缓冲区与事件](litert-v2-metal-events-and-buffers.md) · [原始事件类实测](litert-original-event-probe.md) · [34文件来源](../sources.md#litert-metal-runtime)

2026-10-16。本页研究LiteRT V2实际使用的Metal GPU路径，不沿用[旧CoreML delegate](litert-coreml-partition-and-fp16.md)的图格式或I/O结论。没有构建/运行ML Drift引擎、模型、算子或GPU kernel；仅另一份报告中的LiteRT原始事件类有host-only执行。

## 1. 依赖链先固定，再讨论实现

```text
LiteRT b0983dc2dc27141a64fba7553f40299631e65937
  WORKSPACE::ML_DRIFT_COMMIT
    → ml-drift 1b18771f44c0be3d972847b31a80f6dc3d6964b5
```

[WORKSPACE][workspace]还声明archive SHA-256 `0a7384901754a4ee84220561bcf5447fde4363cbc8a42c94743f183d178feb30`，并有一条GL BUILD include-path补丁。此次只核Git commit与所读工作文件blob，没有下载核算归档字节，也没有把该补丁执行到参考checkout。Metal读本是依赖commit的原始源码，不是任意另一份最新main或已经构建成功的Bazel产物。

[ML Drift README][root]描述GpuModel、GpuOperation、UCL跨API代码生成、tensor virtualization和权重/内存优化。它是GPU引擎，Metal只是后端之一；不是独立ANE backend。README里的数量级性能宣传、Google博客搜索摘要和其它设备数字均不成为本机速度结果。博客正文直接打开返回HTTP406，论文只发现入口，本轮没有读到全文，不虚构算法/benchmark细节。

以上未读状态属于第十轮历史范围。第十一轮已[更正论文/博客身份并核对正文图表](mldrift-apple-mpp-and-routing.md)：此前未验证的候选关联撤回，正确论文为2505.00232v1，开源博客来自Google Developers Blog；旧source manifest保持冻结，不把旧候选当已读来源。

## 2. Compiled Model API仍需读实际平台路由

[LiteRT的Metal accelerator][accelerator]明确报告hardware support为GPU，创建ML Drift Metal delegate，注册Metal buffer/texture、FP16与packed类型的handler。它不是靠旧CoreML模型调用自动落到GPU，也没有在这条路径上调用ANE。

选定[Swift CompiledModel][swift]的无options构造明确建立CPU配置；根README概述的“自动选择accelerator”不能代替这个入口的默认值。GPU实验需明确选项、加载到的accelerator和实际委托范围，而不是只看到CompiledModel类名就认为用了GPU。

### 精度不是一个bool

[选项桥接][options]把FP16+FP32累加拆成基础precision=kFp16和独立 `use_f32_accum_for_fp16`；[Metal delegate][delegate]最终据此选择F32F16。default则按设备round-to-nearest能力选F16/F32F16。逻辑Tensor dtype、物理buffer精度、累加精度仍需分别记录。

同一入口还传播external tensors、weight loader/shared constants、argument buffers、residency、cache等选项。传入一个选项不证明所有路径都实际采用；未知条件、fallback和最终资源绑定仍要继续追，不能把“支持INT4/FP16”压缩成任意模型统一更快。

## 3. 图准备与热路径的工作不同

固定依赖的[InferenceContext][context]从GpuModel移动GPU operations，分配内存，再compile→bind tensors→update params→fast tune；可另序列化模型/程序信息。准备成本不属于只计热dispatch的分母，也没有因此取得严格cold-cache资格。

实际编码遍历node，按optional tags跳过未启用节点，然后调用[ComputeTask::Encode][task]绑定pipeline、参数并dispatch。active-wait路径通过[LiteRT backend][backend]和InferenceContext按flush period切command buffers；另一路在现有command buffer上编码。两者改变提交、CPU等待与暂存对象寿命，不能仅把task数量当实际GPU launch或完整时延。

### Argument buffer已接通，不等于ICB已启用

ComputeTask有两种普通Encode：

- direct bindings：逐项编码MetalArguments；
- argument buffer：显式声明间接资源，再绑定arg buffer；资源/参数重绑定时还更新其中内容。

[delegate][delegate]在argument-buffer tier1设备上会关闭这个选项。缓存参数减少host编码工作不等于消除resource声明、descriptor更新或shape变化验证。

同一源码有 `EncodeToICB`/`EncodeWithICB`，但[InferenceContext初始化][context]的创建条件以 **`false && ...`** 开头。所读普通dispatch调用EncodeWithCommandQueue/CommandBuffer，并不调用ICB执行入口。因此这个ref不能被列成“本机启用了ICB graph replay”，也不能把argument buffer与ICB混为同一功能。这里没有修改false门槛或尝试强启用。

## 4. Mutable tensor绑定是地址更新，不是重新建整图

[InferenceContext::SetTensor][context]更新memory manager中的external tensor，再只遍历引用该tensor的节点，调用SetSrc/SetDst。[ComputeTask][task]随之更新MetalArguments/argument buffer。可迁移的是把稳定图、可变资源和参数刷新分开；不是保证任意shape、dtype、view、并发slot都可无条件复用。

一个buffer对象相同，不等于其offset、逻辑布局、物理精度或代次相同。临时编译/tune时分配的external-mutable tensor会在初始化后解绑，不能拿准备阶段地址作为之后每次请求的永久地址。

## 5. 内存规划不是只跑一次GREEDY_BY_SIZE

[MemoryManager][memory]对选定buffer usage同时计算：

1. GreedyBest对象分配：多个可复用buffer对象；
2. GreedyBySize offset分配：一个较大buffer的子区间。

只有offset候选总字节不大于对象候选总量、backend总量门槛及设备max buffer size时才选它。不是看到GREEDY_BY_SIZE就断言所有模型都用单buffer，也不是本库CPU模型已经执行了上游allocator。

该allocator以macOS的256-byte常量地址空间对齐政策为起点；涉及buffer-backed纹理时，再与设备返回的linear-texture row-byte alignment取LCM。随后用真实base offset创建buffer或纹理view。这个256是**所读allocator的保守策略**，不是“所有Metal buffer通用256对齐”的硬件定律。

候选函数的bytes也不是整个模型resident/peak memory：其它资源、常量、队列in-flight和运行时缓存另有生命周期。共享allocator、共享底层buffer、共享模型权重与多请求并发安全是不同问题。

## 6. C4名称相似，batch顺序仍不同

[TensorDescriptor][desc]明确区分逻辑layout与物理layout，buffer默认physical layout为DCHWBC4。限定D=1、BHWC、buffer默认布局，令 `s=floor(c/4)`，标量lane地址为：

```text
offset = 4 × (((s×H+y)×W+x)×B+b) + (c mod 4)
```

而此前[MNN选定Metal转换](mnn-metal-copy-and-convolution-abi.md)是slice→batch→plane→lane。例：BHWC `[2,1,2,5]`、坐标 `[1,0,0,0]`，ML Drift默认地址4，MNN的对应公式得到8。只在B=1测试会漏掉差异。

此外还有另一种1D physical layout与多种2D纹理布局。[descriptor实现][desc]会按这些模式计算不同地址和storage dimensions；“PHWC4”注释不能代替实际descriptor。本文公式不推广到所有模式，buffer requirement中的 `{16}`也不应直接套成任意张量的完整stride表。

独立CPU数据与代码检查默认布局的48种正4D往返，并验证offset候选的三个门槛。未执行实际allocator、descriptor上传或数学算子，没有证明FP16舍入、量化、尾读/写和模型质量。

## 7. 采用时真正应比较什么

将UCL/kernel family、权重packing、argument binding、memory planner和request交接作为不同实验变量。固定checkpoint/量化数学、bias/LoRA、shape、external layout与resident budget，记录prepare/compile/tune、热GPU、CPU输入输出转换、完整消费者、峰值驻留及质量。新GPU基线改变后，再评估是否值得借ANE。

这轮取得的是GPU运行时与互操作机制的读本，不是ML Drift在M4/M5上的性能资格。实际async可达性、输出事件、host lock成本与失败返回在[下一篇](litert-v2-metal-events-and-buffers.md)展开。

[workspace]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/WORKSPACE#L497
[root]: https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/README.md
[accelerator]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/litert/runtime/accelerators/gpu/ml_drift_metal_accelerator.cc
[swift]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/litert/swift/Sources/LiteRT/CompiledModel.swift
[options]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/litert/runtime/accelerators/gpu/ml_drift_delegate_create.cc
[delegate]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/ml_drift_delegate/delegate/delegate_metal.mm
[backend]: https://github.com/google-ai-edge/LiteRT/blob/b0983dc2dc27141a64fba7553f40299631e65937/ml_drift_delegate/delegate/gpu_backend_metal.mm
[context]: https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/metal/inference_context.mm
[task]: https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/metal/compute_task.mm
[memory]: https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/metal/memory_manager.mm
[desc]: https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/common/task/tensor_desc.cc
