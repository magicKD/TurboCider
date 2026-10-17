# ML Drift Apple MPP：能力门槛、量化选路与转换成本

<!-- knowledge
id: technique-gpu-ane-case-studies-mldrift-apple-mpp-and-routing
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [低比特packing与GPU解码](mldrift-quant-packing-and-decode.md) · [来源与论文身份](../sources.md#mldrift-quantization) · CPU合同

2026-10-16，固定ML Drift `1b18771...`，完整commit见来源清单。本页读到实际MPP generator和builder选路，但**没有执行MPP、完整GEMM或模型**；本机M4 Max运行的仅是另一报告中的通用MSL3.2 U4 decoder。

## 1. 硬件、语言、layout和builder条件缺一不可

[SupportsConvAppleMPP][mpp]要求Metal API、Apple vendor、family>=Apple10及MSL>=4.0。[该ref的设备分类][gpuinfo]将M3/M4归family9、M5类归family10；这是代码分类，不是所有SKU/驱动的硬件实测或发布支持表。

[Metal语言选择][metalcommon]还受SDK编译条件、系统availability及device支持Metal4 family检查影响。装了新SDK或运行在新macOS，不会单独满足Apple10门槛。更不能把Metal GPU TensorOps/NAX归类成独立ANE。

对某个external-weight直读分支，额外要求权重是OSpatialIOGroupI4O4、output_group_size等于输出slice数，且dtype属于支持集合。float-source builder还检查F16计算精度和输入C可被32整除。一个类存在或支持函数返回true，不等于任意VAE Conv或任意低比特layout自动取得该路径。

所读[普通float Apple convolution selector][selector]本身仍在WaveMatrix与Generic间选择；MPP还需要追到[GpuModelBuilder的特定调用点][builder]，不能只凭文件名推断覆盖。

## 2. 先按实际空间量分流，不按“prefill/decode”字符串

[低比特FC builder][builder]按BHWC推导total_spatial_size，再结合dtype相关cutoff、grouped量化、weights_sum、capability与hint选路。对于所读Apple F16条件，recommended cutoff算出16；这不是所有GPU、精度或batched-weight图的固定token门槛。

| 所读Q4条件 | 路线 | 不应误标 |
| --- | --- | --- |
| 小空间量不超过cutoff | fused FC读取压缩权重并解码 | 不自动执行整数activation MMA |
| 大空间、sum可用、支持且允许4bit conv | 单独activation量化→int4 conv路线 | 不是所有Apple GPU有此能力 |
| 否则非grouped、sum可用、支持且未禁用8bit conv | activation量化→int8 conv/MMA候选 | 权重仍可能以4bit存储、随后展开 |
| 其余条件 | float-source weight conversion或融合解码conv | 不能以模型文件bit数认定计算bit数 |

独立CPU策略见证只复述分支选择，不执行图构建。cutoff用真实空间量，group形状和weight batch可能改变计算；本页没有给任意prompt长度建立实机性能阈值。

## 3. 一份权重：融合解码还是先转成快核布局

`FullyConnectedSrcFloatExternalWeightsWithConversion`估计权重转换成本：

```text
conversion_cost_percent =
  100 × (压缩源权重字节 + 目标权重字节) / FLOPs × heuristic_FLOPs_per_byte
```

代码在所读Apple条件下使用32或110这样的校准系数，并以20为推荐融合门槛；有ringed weights时又有不同处理。**这些是实现启发式，不是峰值TOPS/带宽定律或本机测量。**

在单batch、1×1稠密投影的简化代数中，O/I相消后可看到：M越大，转换摊销比例越小。同样的W4、目标16bit、系数110，M=128的估计超过20，而M=1024低于20；CPU数据保存精确数值，但不将百分比叫成实测额外延迟。

若融合候选的MPP/WaveMatrix/Generic支持当前external layout，可在kernel中解码；否则创建目标conv布局并插入WeightsConversion。目标descriptor与原descriptor不同、或ring条件变化，都影响是否重排。迁移时必须把转换读写和额外驻留算入分母，不能只取转换后的快矩阵核。

## 4. MPP generator中的两种低比特含义

[ConvAppleMPP][mpp]生成包含MetalPerformancePrimitives header的MSL，构造matmul2d descriptor与cooperative destination tensor。两类路径应分开：

- float路线：U8/U4/U2权重解码成half向量，应用scale/bias，再供矩阵操作；支持低bit存储不证明原生同bit整数MMA。
- integer路线：可将biased-U4先XOR恢复signed nibble，再展开为signed INT8字节，供INT8矩阵路径；activation quant params、weight sum、scale与zero-point由builder的后续dequant逻辑处理。

本库CPU穷举65,536种四nibble输入，逐一检查该bit-expansion表达式等价于各nibble减8，共262,144个signed-byte值。它验证整数代数，不执行MPP硬件或完整量化dot。

对于权重转换/手工读取分支，源码把K tile设为32，external低比特case选择N tile64、4个SIMD groups，并将解码后的tile写入threadgroup memory，在复用前后放barrier。其它直接tensor路径可用不同K/N tile；构造器默认N128也可能因输出对齐回到64。不能把一个tile和scratch假设套给所有路径。

源输入不是合适buffer/CC4布局、融合softmax或非1×1卷积等条件，又会触发手工读取与threadgroup staging；输出cooperative tensor先store到局部tile，再按有效空间/slice写回及处理bias。这里的资源、padding与barrier均未由小decoder验证，不能据此声称已验证尾部安全、驻留或M5速度。

## 5. 论文、开源博客与当前源码分账

正确论文是[arXiv:2505.00232v1][paper]（2025-05-01）。本轮读取正文1–8页，并用PDFKit视觉核对第3、6、8页图表。图8在M4 Pro上用不同纵轴表示prefill/decode，不是所有模型/阶段都领先；协议含1024 prefill、256 generation，逐token同步，关闭speculative/FlashAttention。量化配方也不等同所有GGUF。它不是当前MPP路径或本机性能复现。

[2026-10-08官方博客][blog]是开源发布与能力概述，不能替代固定commit的算子/consumer证据。前轮未验证的Research博客地址和`2610.03826`候选已撤回，正确身份与原始字节SHA在[更正记录](../sources.md#mldrift-quantization)；冻结的历史manifest不被悄悄改写。

## 6. 下一步真正应测什么

先验证固定packing、group/zero-point与完整输出，再对相同权重/激活/累加精度比较融合解码与转换+快核，纳入prepare、conversion、scratch/驻留、冷/热和质量；针对实际device family/SDK/shape决定是否有资格运行MPP。M4 decoder通过、CPU bit-expansion通过、论文图表和一段supports函数都不能替M5、完整模型或GPU+ANE物理并发背书。

[mpp]: https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/common/kernels/conv_apple_mpp.cc
[gpuinfo]: https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/common/gpu_info.cc
[metalcommon]: https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/metal/common.mm
[selector]: https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/common/selectors/convolution_selector.cc
[builder]: https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/common/gpu_model_builder.cc
[paper]: https://arxiv.org/abs/2505.00232v1
[blog]: https://developers.googleblog.com/en/ml-drift-next-gen-gpu-aiml-inference-at-the-edge/
