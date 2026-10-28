# MLC FFN 的中间观察与精度控制：条件通过不等于完整层通过

<!-- knowledge
id: technique-metal-case-studies-mlc-ffn-stage-observation-and-precision
type: case-study
stage: ffn, prefill, decode, pipeline
operator: gemm, gemv, norm, swiglu, quantization, scheduling
backend: metal, cpu-reference
device: cpu, gpu
symptom: precision-loss, measurement, launch-bound, graph-interruption
semantics: numerically-equivalent, approximate, contract-only
representation: fp16, fp32, q4, packed
boundary: producer-consumer, cpu-gpu
api: public, private, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [原始量化FFN失败](mlc-quantized-ffn-and-lowbatch-boundaries.md) · [原生VM部署](mlc-runtime-only-relax-vm.md) · [来源](../sources.md#mlc-vm-stages-sources-20261017) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第十三轮诊断部分。继续检查前轮严格失败，不改原权重、输入或`0.02+0.02×|ref|`门槛。原图buffer观察保存了输出，却明显改变同步/提交；实际feed的条件检查可以通过而完整层仍失败。**两次投影改为FP32输出、再转回FP16的任务控制，在18个选定完整FFN输入上全部通过原门槛，但改变了生成调度，尚无性能或模型质量资格。**

## 1. 先保留原图，再拆消费者

沿用MLC `8978ea9a...`、TVM `201fa000...`、FFI `df463f9b...`的固定源码和上一轮dylib；完整SHA见来源JSON。三组为H64/F96 KN、H65/F97 KN、H65/F97 NK。每组选seed17/M1、seed29/M8/M9及M9 codebook/zero/tiny，共18个独立输入；两种投影dtype控制各运行一遍，共36个case。

每个case首先从**同一个已导出库**执行：

```text
原VM无observer → 原VM instrument读N/GU/P/Y → 原VM无observer
                    ↓ 完成后的完整buffer
      原Norm、原quantized gate/up、原activation、原down+residual
                    ↓ 实际feed条件图
      相同packed/scales + typed-FP32投影控制 + 完整最终输出
```

原图符号为RMSNorm、dequantize-matmul、split-SiLU-multiply、dequantize-down-add。使用原[VM instrument接口](https://github.com/mlc-ai/relax/blob/201fa0001c81431ccce05419b442c47779ea917c/python/tvm/runtime/vm.py)，仅在实际调用完成后读取对应完整Tensor，callback永不`SKIP_RUN`，不改IR或添加原图输出。before/after用独立的无observer VM实例；没有把noop callback伪称为已关闭instrument。

按已知符号和完整shape选中最后一个匹配的输出参数，要求恰好获得N/GU/P/Y，否则拒绝采集。它是这个已审计图的有限观察器，不是任意模型通用profiler，也不凭shape就证明任意alias安全。

36个case的before、observed、after及捕获final都与旧完整输出逐位相同。三次原始完整调用共108个输出，其中18次仍重现原stress数值失败；**observer没有在本cohort改变值，不代表它没有扰动执行**。

## 2. 观察确实打断了batching

| 原runtime软件字段 | 无observer（含最终读回） | 每stage读回 |
| --- | --- | --- |
|dispatches|4|4|
|flushes|1|4|
|syncs|1|9|
|gpu_to_cpu|1|5|

该模式多次完成同步和CPU读回，最后Y又被读取一次；不能将读回后的软件计数写成无扰动kernel时间、DRAM流量或physical encoder/PMU清单。dispatch数量没变也不足以证明测量无影响。本轮不拿观察模式elapsed作原路径性能数据。

数据来自原[Metal runtime](https://github.com/mlc-ai/relax/blob/201fa0001c81431ccce05419b442c47779ea917c/src/backend/metal/runtime/metal_module.mm)与既有软件计数接口，不是新Instruments或硬件counter实测。真实driver内部行为仍需独立trace；MSL/IR只证明生成程序，不是执行ISA。

## 3. 完整oracle与实际feed oracle不能混用

N=RMSNorm(X)，GU=gate/up投影，P=SiLU(gate)×up，Y=down(P)+X。完整参考仍从X和**实际packed/scales独立重建的FP16权重**开始，用FP64 dense计算并保留前轮明确的阶段FP16舍入，不以原float权重替代encoded-weight oracle。

条件参考则故意输入实际GPU上游值，例如`down(P_gpu)+X`；它回答“给定这个P，down做得怎样”，不是“从原X出发的FFN是否通过”。新条件GPU图也调用原`RMSNorm`、原`GroupQuantizeLinear`、原QWen2 activation，采用完整原MLC pipeline，没有手写quantizer或替代shader。

默认FP16条件图的4个输出，在18个输入中都与捕获的原stage逐位相同；这种交叉控制排除了只看一个新观察图就替原pipeline结论的做法。

codebook的原始stage对实际feed参考的最大误差如下；这些max值不是同一个元素，不可直接相加：

| 配置 | N | GU（给定N_gpu） | P（给定GU_gpu） | Y（给定P_gpu） | 从原X算的完整Y |
| --- | --- | --- | --- | --- | --- |
|H64/F96 KN|0|0.03906|0.03125|0.08492，条件门槛通过|0.11255，完整门槛失败|
|H65/F97 KN/NK|0|0.03125|0.015625|0.03442，条件门槛通过|0.05005，完整门槛失败|

本cohort的N与舍入后的完整参考逐位一致，替换实际N不改变失败；将实际GU或P代入下游参考，最终检查却可以通过。**不能把后者改名为完整模型准确性恢复**。相对/绝对预算受幅值、非线性和residual cancellation共同影响；某个条件局部门槛通过并不自动传递到整个链。

同样不能把全部误差都归于一个projection或一个物理accumulator：P对高精度分阶段参考也有偏差，范数为0只适用于此批舍入后的输出。原[RMSNorm](https://github.com/mlc-ai/relax/blob/201fa0001c81431ccce05419b442c47779ea917c/include/tvm/topi/nn/rms_norm.h)会转换到FP32计算再返回输入dtype，但这不是所有shape/epsilon/设备的误差保证。

## 4. 一个保持低比特权重的精度控制

在新任务图里，仅将原gate/up与down模块的`out_dtype`设置为`float32`，每次投影后显式转回FP16；norm、SiLU/乘积、residual接口仍按原FP16边界。参数、packed/scales、X均不变，原库/原失败不重写。这是一个**改变浮点计算路径的控制候选**，不是原始完整MLP毫无修改，也不是改为FP32权重或新的量化方案。

| codebook配置 | 原始/FP16控制max abs | FP32投影控制max abs | 原定完整门槛 |
| --- | --- | --- | --- |
|H64/F96 KN|0.11255|0.01611|失败→通过|
|H65/F97 KN|0.05005|0.01282|失败→通过|
|H65/F97 NK|0.05005|0.01282|失败→通过|

FP16完整控制18个输出有3个stress失败；FP32投影控制18个全部通过。36个candidate输出全部保存并独立CPU重算原门槛，未用实际GU/P去替换candidate的完整参考，也未放宽线。

在相同实际feed的条件图里，FP32控制的norm/activation仍与原stage逐位相同，非零投影结果则改变。这为“投影精度/调度值得检查”提供证据，不是唯一因果归属。生成代码也变了：原FP16配置有2个SIMD-group MMA投影kernel；FP32配置的相关kernel不再出现该MMA，转而有thread float数组。默认LowBatch上下文每配置生成6个投影kernel，禁LowBatch的NK上下文是2个，不能按生成数量冒充调用数量。

原[MetalMatmul规则](https://github.com/mlc-ai/relax/blob/201fa0001c81431ccce05419b442c47779ea917c/python/tvm/s_tir/dlight/gpu/matmul.py)固定使用FP16 intrinsic，并有通用fallback；源码与MSL相符，但本轮未加hook捕获每次schedule异常，因此不把fallback原因推断称为动态调度trace。更不能据此宣称Apple硬件不支持FP32矩阵累加或直接断言其物理寄存器宽度。

两种完整控制仍均为4次软件dispatch。**相同dispatch不等于相同成本**；本轮没有合格latency/能耗/内存比较，FP32控制可能失去矩阵吞吐优势，需要另做同预算完整消费者测量，不能默认采用。

## 5. 证据数量、复现和剩余工作

36个case各保存11份完整GPU Tensor结果：before、3个中间buffer、observed final、after、4个条件输出和candidate final，共396份；捕获final与observed final的重复读回不计两份独立结果。独立CPU归档校验这些raw、旧输入/packed身份、全链/实际feed公式、所有接受/拒绝结果。不是把派生参考数组也加进设备输出数。

CPU学习：labs/mlc_vm_stage_contracts.py，case `mlc-stage-precision-contracts`，用half标量示例展示局部预算/完整residual预算不传递和observer负控；它不模拟真实MLC累加。设备入口为显式stage客户端，冻结数据与本轮原生部署客户端共用归档器和离线测试。

正式raw为`outputs/metal-mlc-stage-20261017-final-{full64,fulltail,pair-nk}-{float16,float32}`；smoke另列，不计本cohort。所有source与历史artifact保持身份，不改产品默认，不安装新依赖、下载模型或清cache。

后续应测更大/真实权重的FFN与强GPU基线、长序列与内存/能耗，并将条件诊断和原参考质量分开。源码/库、投影dtype/cast位置、schedule/融合、scale/packing、shape/幅度、observer、compiler/SDK/OS/SoC变化均需复查。这里没有完整checkpoint/LoRA/attention/图像视频质量或NAX资格，整体研究继续。
