# 公共Core ML与Metal交接：CVPixelBuffer、真实IOSurface、pitch与别名完成边界

<!-- knowledge
id: technique-ane-public-pixel-buffer-gpu-handoff
type: method
operator: layout, scheduling
backend: coreml, metal, cpu-reference
device: cpu, gpu, ane
symptom: wrong-layout, stale-cache, lifetime, measurement
semantics: numerically-equivalent, contract-only
representation: fp16, fp32, iosurface
boundary: producer-consumer, cpu-gpu
api: public, reference-only
-->

[ANE目录](README.md) · [固定源码/SDK](sources.md#pixel-gpu-sources-20261016) · [先前heap backing](output-backing-and-stride-ownership.md)

2026-10-16，M4 Max、macOS26.6.2、SDK26.5、clang21、既有coremltools8.3.0/NumPy2.0.2。原创公共API客户端真实执行：**Metal纹理producer→CVPixelBuffer-backed MLMultiArray→Core ML→实际采用的output pixel buffer→Metal纹理consumer**。另有显式CPU copy-out/restore控制。

八个新进程覆盖width33/65×seed17/29×CPU_ONLY/CPU_AND_NE；48次实际预测、64组GPU consumer见证，共150,528个FP32见证元素及4MiB完整region字节可独立复算。额外16组是故意跳过预测的stale consumer，**不算16次新推理**。没有运行原ccv bridge、没有模型下载、物理ANE/完整模型速度或能耗资格。

## 可复用的方法：共享载体也要验证真实交接

1. 分开记录逻辑尺寸、row pitch 和 allocation；按实际 plane/stride 构造 Core ML 与 GPU 视图。
2. 先验证框架是否真正采用所提议 backing，再核对 GPU 完整消费者与 padding/tail；共享 ID 不替代这些检查。
3. 将 ready、consumer 完成和 release 分开，保活到最后读者；同时保留显式 copy 路线作正确性和成本对照。

适用于探索公共 pixel-buffer→GPU 交接，不保证任意模型都接受 backing，也不证明内部无 staging。

## 1. 这次跨的是公共接口，不是private request

先前[shared surface研究](../gpu-ane/shared-surface.md)验证private `_ANERequest`的surface与GPU消费者；[heap backing研究](output-backing-and-stride-ownership.md)只用host allocation。它们都不能代替本次公共CVPixelBuffer路径。

本例创建OneComponent16Half pixel buffer，按`[1,4,3,W]`包装为MLMultiArray。PB的width=W、height=4×3；GPU将channel/height折叠到二维texture的row，使用R16Float。它是**张量存储视图**，不是RGB image feature、颜色转换或图像预处理。

[CVMetalTextureCache](https://developer.apple.com/documentation/corevideo/cvmetaltexturecachecreatetexturefromimage(_:_:_:_:_:_:_:_:_:))建立live binding，SDK明确要求持有CVMetalTexture直到图像使用完成；只持有裸MTLTexture不是完整owner说明。[MLMultiArray pixel-buffer initializer](https://developer.apple.com/documentation/coreml/mlmultiarray/init(pixelbuffer:shape:))与outputBackings的SDK注释要求：预测前不要通过dataPointer/subscript等data访问让output PB处于CPU锁定状态。

主direct arm只读shape/stride/dtype/PB身份等metadata，不读其data pointer，不锁CVPixelBuffer；输入由GPU写，输出直接由GPU消费。所有GPU命令均检查完成。CVMetalTexture、CVPixelBuffer、MLMultiArray及raw mirror的owner保持到使用结束；未测试提前释放、跨进程pool复用或无同步竞争。

## 2. 逻辑尺寸、row pitch与allocation是三个数

| W、12个物理row | 逻辑row bytes | 实际row bytes | 逻辑FP16 tensor bytes | 本client allocation |
| --- | ---: | ---: | ---: | ---: |
| 33 | 66 | 128 | 792 | 32,768 |
| 65 | 130 | 192 | 1,560 | 32,768 |

实际MLMultiArray strides分别为`[768,192,64,1]`与`[1152,288,96,1]`，单位是**元素**；PB row bytes则是字节。这里batch=1、C=4、H=3，不外推任意offset/planar或动态输出布局。

首版使用普通`CVPixelBufferCreate`，被**本client的raw no-copy页长度门槛**拒绝。首个失败只记录合并门槛，没有细分值；后续独立诊断记录base对齐、allocation=2,112字节、page=16,384，不是page倍数。没有把后次诊断反填为首版原来已经记录的事实。

这是额外raw MTLBuffer mirror的保守admission，不是Core ML/纹理API拒绝普通CVPixelBuffer。我们没有冒险将allocation向外round-up去映射未证明拥有的地址范围，也没有尝试该non-page长度的raw wrap来声称驱动一定拒绝。普通PB仍可有合法texture路径，本批未给它做完整预测协议。

主试改用`IOSurfaceCreate`显式声明完整page region，再`CVPixelBufferCreateWithIOSurface`。row由IOSurface alignment API确定，allocation另保留至少一个page用于诊断。它不是最小内存实现，也不能拿这32KiB与普通PB的2,112字节直接算节省/峰值。

## 3. 完整数值与真正的外部backing证据

自有FP16图为`Y=2X+B[c]`，`B=[0,.25,-.25,.5]`；GPU consumer执行`Z=3Y-.5`。输入逐generation改变，值是可完整独立重算的dyadic，不用finite/shape替代数值检查。

24次direct预测同时观察到：

- 真实返回MLMultiArray等于提议对象；返回PB等于owned PB。
- 从返回PB取得的IOSurface ID与GPU texture实际`.iosurface`一致；raw mirror也指向该owned region。
- GPU producer的完整X、GPU从输出texture读取的完整Y、实际GPU消费者Z全量通过。
- 每次重新poison padding/tail后，完整raw diagnostic copy与预定字节配方一致；readback结果尾guard未变。

另24次copy_restore不提议output backing：读取**实际返回array**的真实dtype/stride，显式CPU copy到owned目标PB，再由同一GPU消费者处理。全部actual-array与GPU结果通过。这证明两条所测交付方式都正确，并非仅比较两份相同的host公式。

direct路径没有显式CPU feature staging；但每组另有两个**诊断GPU buffer copy**，以及GPU结果的CPU验证readback，全批共128次诊断copy。runtime/driver内部copy始终unknown，不能叫全系统零拷贝，也没有测得direct必然更快。

## 4. 共享ID、tracked与API成功仍可能全部不足

首个surface版本的collector exit0，Core ML预测也返回成功，identity检查成立；独立验证却发现GPU输入部分仍为`0x5a5a`对应的finite poison203.25，其他位置才是预期输入。模型对它**实际拿到的混合输入**计算正确，不是对预期输入计算正确。

八组失败GPU见证累计2,509个输入元素不符，原6次预测全部排除出48次主试。真实partial-write mask另存为观察证据：可离线重建该次完整失败，但不是一个可以预测GPU调度的确定性模型；未将它简化成“每个元素都还是poison”。

那版把raw-buffer fill与texture write放在同一个command buffer。主试控制先完成poison命令，再提交并完成texture producer，随后才predict。保守完成边界下，八进程完整通过。

本机实际记录texture.storageMode=Managed、raw buffer=Shared，两者hazardTrackingMode均为Tracked。**同IOSurface与tracked标签不是alias顺序/可见性证明**。我们没有设备trace来区分调度、cache或managed表示同步的具体原因，不将这次client协议失败命名为Apple driver bug，也不宣称所有相邻encoder不串行。新实验只改owned提交协议，不修改系统/产品fence策略；未再故意运行已被否决的顺序。

这些storage/hazard属性只在完成边界版采集；旧失败报告没有该属性快照，不反填为旧对象当时已经记录的事实。

## 5. 两项真正送进GPU的有限错误控制

每次consumer还通过raw buffer按紧密index错误读取输出。结果全部finite，却有34,496个元素与正确consumer不符：每组W33错363/396，W65错715/780。不能从`count*sizeof(dtype)`反推每行pitch，更不能因为共享了surface就假设紧密布局。

每arm在第0代正确预测后，GPU写第1代输入，**不调用Core ML**，继续消费旧output texture。16组真实stale consumer中9,408元素全部与当前第1代期望不符。之后恢复正常predict，当前代重新通过。stale与tight控制不计入性能或额外推理成功数。

## 6. 回到ccv：packed-row guard是设计合同，不是普遍能力限制

固定`liuliu/ccv@66dc16a3...`的[PB helper](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_coreml.mm#L160)明确拒绝`physical row bytes != width*element bytes`：Metal transform、weight blit和coherence ranges按packed rows设计，不满足就让GEMM回到Metal。

[调用链](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_gemm.cpp#L444)有准备fence/等待分支，再evaluate，然后从预定output scratch buffer做GPU恢复。此前[source卡](../sources/repos/ccv-coreml-rowwise.md)指出：evaluate检查返回array可读与总字节数，不等价于确认其就是预定scratch。该条件风险仍需原bridge现场验证。

本批有padding的公共图正确，恰好说明**Core ML可接受的布局比这个特定ccv消费者合同更宽**；并不证明ccv应删除guard，也不复现ccv bug。其原图包含INT8接口、量化/缩放与matmul；本批FP16 elementwise图不能为W8A8 MAC、原恢复kernel、异步fast fence或完整FFN取得资格。

## 7. 可复现资产与剩余门槛

标准库oracle重建全部X/Y/Z/tight payload、128个完整region及copy-out数组；再核对raw bytes、original model、client/shader/binary、监督receipt和来源SHA。独立protobuf client只检查自有八个模型的IO/FP16常量/完整连接，不compile/load/predict、不将raw SHA混成语义hash。

显式driver、原创native client与Metal shader只需既有固定依赖/macOS/CLT，必须device opt-in和全新owned output；不安装依赖、下载模型、清全局cache、运行上游demo或改产品默认：

本批没有强GPU-only分母、计时消融、hardware placement/PMU或能耗，故不计算GPU+ANE倍率。仍待真实matmul/FFN消费者、原ccv支持条件、stateful跨设备事务、资源池/取消/多队列、真实模型质量与端到端成本。任意OS/SDK/source、shape/dtype/pitch/allocator、owner/stream/完成边界或消费者变化均须重开，不能把这条公共stateless接缝与前一批stateful实验拼成已验证的完整KV流水。
