# 完整 Native Metal encoder：单图编辑实现与验证

2026-10-08。完整Qwen3-VL视觉27层与语言36层已经由应用自有Metal graph执行，并完成一组单图编辑。独立完整encoder和实际Session encoder均在本组有净收益。保持显式实验，默认仍MLX；没有Native encoder ANE、W8A8或W2资格。

实现/约束见[设计22](../design/qwen-image-gpu-ane-command-graph/22-native-edit-encoder.md)，结构化结果见[JSON](2026-10-08-qwen-native-encoder.json)。本轮只用本地Qwen-Image-2.1、已有teapot参考和Viggle六步LoRA，没有下载模型、复制checkpoint或导出模型。

## 完整实现

- [NativeEncoder](../../native/models/qwen21/native_encoder.mm)：视觉patch projection、位置插值、LayerNorm、D72二维RoPE、FP32 attention、GELU、残差、final/DeepStack mergers；语言FP32 RMSNorm→BF16、Q/K norm、GQA、mRoPE、causal attention、BF16 FFN与残差，支持编辑的pre-final-norm输出。
- [NativeAttention](../../native/backends/metal_graph/native_attention.mm)：复用Steel online-softmax，新增Q32/KV8映射及FP32/D80视觉特化，保留80字节ABI、尾部和物理stride检查。视觉零补72→80后按1/sqrt72缩放，输出去掉8个尾部通道。
- [Session](../../native/models/qwen21/pipeline.cpp)：单图研究域选择、held来源、缓存、PSO复用、完整层数/物理完成检查及实际执行元数据。没有替换既有DiT、VAE或默认后端。

模型塔计算全部Native Metal；MLX仍处理原始reader、图像预处理、embedding/prompt assembly及输出边界。两层提交窗口先提交当前Native queue，再准备未来最多两层，完整33次提交、1053次dispatch、30个未来权重窗口。future源异常先wait当前GPU，再释放/抛错；unhealthy后端拒绝复用。host `mx::eval(next)`和每两层wait仍存在，不称为W2或完全设备驱动流水。

## 构建与正确性

最终[库、CLI、App和helper](../../build/qwen21-native-encoder-r3-20261008)身份为
`tc-runtime-build-v1-28d1c029316037cb0ddeb142671684b97089974a33ebbee62267df3ba6dd65d5`。
577个production identity source在generated、组件、完整encoder和E2E前后均一致。package和App/helper构建通过，只有已有ANE helper的deprecated API警告。

[小尺寸完整层](../../outputs/qwen21-native-encoder-20261008/generated-r3-supervisor.json)检查33/129行两层语言（GQA、mRoPE、两路DeepStack、raw/final RMSNorm）及4/8网格两层视觉（patch/position/attention/FFN/mergers）。语言最大RRMS0.84246%；FP32视觉最大0.000348%；BF16视觉GEMM最大1.5319%，均低于预声明2.5%。

[真实前两层](../../outputs/qwen21-native-encoder-20261008/components-r3-supervisor.json)使用原始真实权重、4096视觉patches、1066语言行；FP32/BF16 GEMM候选的retained输出RRMS分别0.2503%／0.3111%，低于同一2.5%线。该冷组件计时含新增shader编译，Native比GPU慢，因此不以它授性能资格。

第一版generated fixture用整数array做除法后得到FP32 positions，在任何原生模型执行之前被int32合同拒绝；修复工具显式cast，保留失败日志，未放宽门槛。新增元数据测试最初用prepare-only结果索取完整执行proof，被拒；改为实际完成结果再验证。生产实现没有因这两个工具问题放松检查。

65项CPU attention ABI/extent/GQA检查通过；129项已有API合同通过（3项跳过）；实际encoder元数据及DiT backend一致性检查通过。App/helper编译通过，本轮没有UI验收或替换安装版本。[API日志](../../outputs/qwen21-native-encoder-20261008/contract-tests.log)、[attention日志](../../outputs/qwen21-native-encoder-20261008/attention-contract-tests.log)、[构建日志](../../outputs/qwen21-native-encoder-20261008/native-build-r3.log)、[App日志](../../outputs/qwen21-native-encoder-20261008/app-build-r3.log)保留。

## 独立完整单图 encoder

[监督记录](../../outputs/qwen21-native-encoder-20261008/full-r3-supervisor.json)：单张已有512px teapot放大到1024 reference，4096视觉patches→1024视觉tokens、1066语言行、保留28行conditioning。视觉27层和语言36层全部执行。计时包含源加载、tokenizer、resize/assembly与本进程准备；没有DiT/VAE。预声明conditioning RRMS≤8%、最低净省时5%。

| 路径 | 视觉 | 语言 | 完整encoder | 相对本组MLX省时 | conditioning RRMS |
|---|---:|---:|---:|---:|---:|
| MLX GPU |1.689s|3.817s|5.768s|—|—|
| Native FP32视觉 |1.364s|2.663s|4.330s|24.93%|2.7913%|
| Native BF16视觉GEMM |1.309s|2.615s|4.224s|26.77%|3.0838%|
| Native BF16，第1次准备对象请求 |1.320s|2.608s|4.200s|27.18%|3.0838%|
| Native BF16，第2次复用请求 |1.302s|2.600s|4.171s|27.69%|3.0838%|

复用的两次均重新加载fresh SourceLease/Weights，禁用conditioning cache，两次输出逐位一致。第2次准备阶段约0.2µs，只复用PSO，未保留旧source closure。复用arm在timer前创建NativeEncoder，constructor耗时不计入该arm；首次两种独立Native arm则计入constructor。OS/Metal编译cache此前已被组件/旧进程使用，并未重置，因此以上不是操作系统冷缓存成绩，也不是稳定平均值。

FP32视觉是Native默认值。BF16视觉GEMM只比同组FP32多省约0.106s，精度更低，保留为显式候选，不据此改变默认或声称整图资格。

## 实际单图编辑

[E2E监督记录](../../outputs/qwen21-native-encoder-20261008/e2e/receipt.json)：512²输出、单张1024 reference、Viggle r128/strength1、六步、seed42、DiT cache关闭。两独立进程，MLX→Native FP32视觉，各一次。所有6步/227 LoRA projections、4096 reference latent tokens、无conditioning/vision hit和实际完整Native encoder均核对。

| 阶段 | MLX | Native encoder |
|---|---:|---:|
| encoder（加载/assembly/首次准备在内） |5.447s|5.039s|
| reference VAE |0.920s|0.927s|
| denoise |28.682s|28.549s|
| VAE decode |0.956s|0.965s|
| request wall |39.305s|38.680s|

本组encoder省时 **7.49%**、完整request少 **1.59%**。其余未修改阶段也有波动，不能将全部请求差值归于encoder。Native首次准备0.818s、host source-ready累计2.293s，GPU wait累计1.523s；这些host spans不能相加成设备算子耗时或宣称完全隐藏I/O。独立encoder的24.93%不能转述为App整图24.93%。

最终图像pixel RRMS **0.8825%**，平均绝对差0.854/255，低于预声明8%；[MLX图](../../outputs/qwen21-native-encoder-20261008/e2e/reference.png)和[Native图](../../outputs/qwen21-native-encoder-20261008/e2e/optimized.png)目视茶壶轮廓、位置、壶嘴/手柄/盖及光照接近，细微纹理有差异。这是一张图的筛选，不代表跨prompt语义质量或用户发布资格。

## 范围、资源及结论

本轮形成一组generated/真实组件筛选与一组完整encoder/有限E2E验证，未进行持续高频推理。监督器各arm有超时及进程组终止，完成后才核对来源stat/hash。完整encoder组swap在前后均4565.69MiB，无已记录thermal warning；这是端点观察，不是连续资源认证。

当前已完成**完整Native Metal模型塔和单图净收益筛选**。GPU/ANE原生encoder接线、设备事件W2、MemoryExecutionContext内存档位、encoder LoRA及App选择器仍不借此取得资格。当前1图1024-reference编辑的主要耗时已经是约28.55s的DiT；继续优化encoder无法消除这部分耗时。Native Shader首次准备与reader调度仍值得优化，持续worker可复用PSO，但条件cache命中应如实记录0层。

owned清理仅针对本轮build/probe的对象、archive、module cache和被新版本替代的初始工具；保留有效新库、CLI、App/helper、probe、manifest、最小conditioning比较文件、日志和图片。未清理模型、系统/driver cache或前轮有效private程序。清理详情见[回执](../../outputs/qwen21-native-encoder-20261008/cleanup.json)。
