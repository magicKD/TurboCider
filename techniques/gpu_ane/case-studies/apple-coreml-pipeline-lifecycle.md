# Apple Stable Diffusion：compute-units、模型生命周期与预览成本

<!-- knowledge
id: technique-gpu-ane-case-studies-apple-coreml-pipeline-lifecycle
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [attention实测边界](apple-split-einsum-tail-and-mask.md) · [SD.cpp对照](sdcpp-metal-residency-and-segments.md) · [来源](../sources.md#apple-coreml-pipeline)

2026-10-17，固定官方`apple-aiml-research/ml-stable-diffusion@ea2805dc1945be20561c77e5f6d1d9a5a637cda2`，tree `10ad12e4d029248e92a4253159ecdb0ace954e17`。这是2026-09-11的固定对象，不声称最新主线或release。23个选定源码/README范围核官方Git tree blob，完整tree无AGENTS文件；不下载模型、构建整个Swift包或运行Core ML图。

## 1. 这是公共框架的阶段协作，不是显式GPU/ANE并行调度器

Swift pipeline按text encoder→denoiser若干steps→VAE decoder工作，image-to-image另有VAE encoder，SD有可选ControlNet/safety，XL有base/refiner，SD3有多text encoders及MMDiT。`ManagedMLModel`包装每个Core ML模型；系统根据该模型的配置/图选择处理单元。

Apple的[computeUnits属性](https://developer.apple.com/documentation/coreml/mlmodelconfiguration/computeunits)是模型预测配置；[cpuAndNeuralEngine](https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine)允许CPU/NE而排除GPU，不是强制每个op都在ANE。`.all`也不证明所有设备同时参与。此项目没有在所读Swift链中把两条独立模型支路显式分别提交给GPU和ANE；同步`predictions`、批输入和模型分块都不自动成为物理overlap证据。

## 2. XL的VAE配置覆盖，不能只看CLI的一个开关

CLI `main.swift:74–75,112–172`默认`.all`，支持cpuOnly/cpuAndGPU/cpuAndNeuralEngine并传入resource factory。但固定版本的factory有区别：

| Swift资源构造器 | text/denoiser | VAE encoder/decoder |
| --- | --- | --- |
| SD | 传入config | 传入config |
| SDXL | 传入config | **复制config，再无条件设置cpuAndGPU** |
| SD3 | 传入config | 传入config |

`StableDiffusionXL+Resources.swift:94–104`的覆盖包含调用者传cpuOnly或cpuAndNeuralEngine的情形；不是Core ML忽略配置，而是此factory创建VAE时换了一份配置。cpuAndGPU只是允许GPU，并不证明实际一定用GPU；调用者若要求整条pipeline的设备政策一致，需要逐组件核查，而不是仅记录CLI参数。

转换端还有精度配方：`torch2coreml.py:548–605,700–760`默认XL VAE decoder使用FP32/CPU_AND_GPU；custom VAE存在时decoder分支可恢复F16输入及默认precision/传入compute-unit。但XL encoder分支仍无条件选FP32/CPU_AND_GPU，Swift XL factory的覆盖也仍在。不能把“custom FP16 VAE”直接解读成“本Swift pipeline的VAE已经上ANE”。源码关于ANE/FLOAT32的注释是该配方的设计理由，不外推为所有接口、I/O dtype和未来硬件的永久限制。

`--quantize-nbits`调用的是k-means **LUT palettization**，并对gather设置不压缩；不是通用W8A8整数MAC。storage bits、activation dtype、编译后放置和模型质量分别验证，本轮没有执行转换/压缩。

## 3. lazy load、预热与物理释放是不同边界

`ManagedMLModel.swift`里：

- init只保存URL/config，`loadedModel=nil`；load/perform都走单模型serial queue。
- `perform`在queue.sync与autoreleasepool中，必要时构造MLModel，再运行调用者closure；同一包装器的load、prediction和unload因此互斥。
- unload只是将包装器的`loadedModel`设nil，不证明驱动cache、所有外部引用或设备wired memory立即释放。
- closure约定只在其内部使用MLModel；把它保存在外部会延长owner。单模型锁不是整个pipeline的请求隔离，重入同一queue也不能当安全调用方式。

默认`ResourceManaging.prewarmResources`只做load→unload，**没有prediction**。UNet/MMDiT/ControlNet的具体方法逐模型load→unload；正常`loadResources`则可把多个模型同时保留。预热可能触发编译/准备并影响后续load，但不等于warm prediction已经跑过，也不能拿预热后的首次predict当严格冷进程性能。

本轮实际将未改的20行`ResourceManaging.swift`与本库Swift observer编译执行。四个假CPU资源对象得到：正常预热1load/1unload、重复两次2/2；load在改变假资源前抛错为1/0/未保留，在改变后抛错为1/0/保留，显式cleanup后1/1/未保留。它验证默认方法没有defer式错误cleanup；**不加载CoreML，不证明真实MLModel泄漏或设备释放行为**。binary声明依赖仅libSystem、libobjc、libswiftCore。

## 4. `reduceMemory`主要按阶段卸载，不是每step重载UNet chunks

SD的`loadResources`在reduceMemory时改为逐阶段prewarm；`generateImages`正常路径依次在prompt、image encoder、denoise、decode/safety阶段后unload相应模型。因而少些不同阶段同时驻留，代价是下一次需要时重新load。

UNet chunks由`ManagedMLModel`数组顺序预测：第一step会按需加载各chunk，随后**保留全部包装器的loadedModel**，直至正常denoise结束的unet.unload或显式unload。`Array.predictions`没有每个stage预测后unload的逻辑。分块可能改变编译/资产/框架内部内存，但源码不能支持“每个chunk运行完立即释放权重，所以每时刻只驻留一个”的宣传。

手工chunk链还把上一stage输出与**原始**输入字典合并，同名时输出胜出；不自动积累所有更早stage的输出。legacy chunk工具会找跨cut活跃变量并为FP16边界加FP32 cast；新coremltools分支委托bisect_model。切点按权重近似对半，不代表compute时间或跨cut激活大小平衡。本轮没有运行bisect/export或remove-original操作。

这与[SD.cpp显式pin/evict分段](sdcpp-metal-residency-and-segments.md)不同，也与[vpipe的runtime-weight双slot](vpipe-runtime-weights.md)不同；不能把一个系统的逐段回收资格套给另一个。

## 5. preview会把decoder带回denoiser窗口，且不计入普通step timer

`PipelineProgress.currentImages:423–425`不是缓存图片，而是`try! pipeline.decodeToImages(...)`。CLI的`saveEvery`在progress handler调用它，然后再保存PNG。降低显式同时驻留的配置下，每次预览都可能load decoder，decode后unload，而denoiser chunks还活着；这会增加load次数、预测次数和阶段交叠的内存需求。

独立CPU模型以最小SD text-to-image、两chunks、3steps、无ControlNet/safety为例：

| 逻辑路径 | decoder load次数（含一次prewarm） | decode调用 | 最大显式已load模型数 |
| --- | --- | --- | --- |
| reduceMemory、无预览 | 2 | 1 | 2 |
| reduceMemory、每step预览 | 5 | 4 | 3 |
| 第一个progress回调停止、无预览 | 仅预热的1 | 0 | denoiser两chunks仍保留 |

这只是独立生命周期见证，不是Core ML对象计数、内存测量或真实耗时；不同组件大小不等，模型数量也不能换算成bytes。

CLI在`pipeline.loadResources()`之后才启动SampleTimer；第一次progress stop之前包含prompt/latent准备、lazy-load及第一步，后续start在preview/save handler之后。最后VAE decode/最终保存也在最后step stop之外。因此step/sec不是整请求latency，preview/IO开销还可能被排除。README中的模型/设备对照主要是2023年的特定app/OS/模型/压缩结果，不升级为本机2026年M4/M5性能。

## 6. 取消与错误收尾并不完全一致

SD/XL/SD3的progressHandler返回false都在正常denoiser unload之前直接返回空数组；它表示停止后续steps，不是中断仍在跑的prediction或完成整pipeline资源回收。若需要可重复的cancel后内存状态，外层仍应显式管理unload并验证设备完成/owner。

decoder收尾也不同：SD在`try decoder.decode`成功后才卸载，XL/SD3用defer在抛错时也执行相应unload。`currentImages`的try!会将decoder错误变成trap风险，不能把“预览失败只是丢一帧”当既有合同。这些是源码控制流观察，不是本机OOM、cancel、crash或drain复现。

## 复算与采用边界

核心报告/测试不依赖CoreML、Swift、PyTorch、原仓库或outputs。原始Swift协议CPU执行和[原始PyTorch attention实验](apple-split-einsum-tail-and-mask.md)分别保存于；23文件/两份Apple正文与依赖身份在[来源清单](../sources.md#apple-coreml-pipeline)。原始代码、编译物、约437MiB的隔离PyTorch参考依赖只留本轮ignored outputs；没有修改项目/系统Python、下载模型或改产品默认后端。

真正采用仍需实际compiled模型、每组件effective config、允许设备与物理placement区分、完整TE→steps→VAE/preview计时、质量、cold/peak、取消和错误回收。设备实验按GPU guard连续5次一秒占用均低于5%准入，缺计数拒绝；本轮没有提交本应用GPU/ANE工作，也未声称对PyTorch进程所有动态框架加载做了attestation。

固定入口：[ManagedMLModel](https://github.com/apple-aiml-research/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/swift/StableDiffusion/pipeline/ManagedMLModel.swift)、[XL VAE override](https://github.com/apple-aiml-research/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/swift/StableDiffusion/pipeline/StableDiffusionXL+Resources.swift#L94)、[normal/early-return卸载](https://github.com/apple-aiml-research/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/swift/StableDiffusion/pipeline/StableDiffusionPipeline.swift#L337)、[preview/timer](https://github.com/apple-aiml-research/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/swift/StableDiffusionCLI/main.swift#L202)。范围与验证见[第十九轮](../../MAINTENANCE.md)。
