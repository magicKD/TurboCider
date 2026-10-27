# FastVLM：Core ML vision→MLX语言模型的实际接缝

<!-- knowledge
id: technique-gpu-ane-case-studies-fastvlm-coreml-vision-and-mlx-handoff
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [本轮摘要](../../MAINTENANCE.md) · [来源/行范围](../sources.md#fastvlm-handoff) · 原始元数据片段实测 · 计划 · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17。新增官方FastVLM的多模态stage案例：vision tower导出到Core ML，projector与语言模型留在MLX，图像特征注入文本embedding后才prefill/decode。**同一图像的这条依赖链不是GPU＋ANE并行证明**；Swift源码也没有显式设置Core ML compute units，不能直接称“ANE编码器＋GPU解码器”。

本轮只实际执行原导出函数内的JSON修改语句，不导入Torch/MPS/CoreML/MLX，不执行Swift、camera、完整export或任何模型。它发现一个具体元数据边界：已有`<image>`的ID若不是剩余token最大ID＋1，原语句可正常返回，却把model config写成另一个ID。该结果与未动态验证的stride/cancellation风险分别记账。

## 1. 固定身份、许可与两个不同的部署入口

固定[`apple-aiml-research/ml-fastvlm@6f7b13179bbc81e6aa7d921bc7c6335c43904778`](https://github.com/apple-aiml-research/ml-fastvlm/tree/6f7b13179bbc81e6aa7d921bc7c6335c43904778)，tree `0cc85a9ccaee5f0ae40cf51dfdb52d738bf06812`，commit日期2026-09-11。21份选定文本逐份核原Git blob，阅读范围在[清单](../sources.md#fastvlm-handoff)；不是整个FastViTHD网络/kernel或全仓库审计，tree无AGENTS。原文件只留ignored `outputs/gpu-ane-fastvlm-round37-20261017`。

许可不能按pyproject中的Apache classifier一概判断：root LICENSE为Apple软件条款，另有ACKNOWLEDGEMENTS中的子组件条款；LICENSE_MODEL单独限定Research Purposes，并明确不包含商业产品开发/使用。本轮只调研代码与自有JSON，未下载、分发或启用官方权重，不据公开下载链接推产品授权。

两个入口要分开：Python研究export说明要求固定`mlx-vlm@1884b551bc741f26b2d54d68fa89d4e934b9a3de`再应用仓库提供的6文件patch；Swift App则依赖mlx-swift-examples≥2.21.2、mlx-swift≥0.21.2、swift-transformers≥0.1.18的up-to-next-major范围。所捕获tree无Package.resolved；本轮不下载patch base、不apply、不解析Swift依赖，也不把本机现有MLX wheel当这套App的binary身份。

官方export目录自己注明是研究探索，并非官方推荐部署方案；Swift App与Python patch也不是完全相同的实现，不能互相继承功能或测试资格。

## 2. “hybrid vision”首先是网络结构，不是设备调度

[论文v1选段](https://arxiv.org/html/2412.13303v1)描述FastViTHD的卷积/Transformer结构和较激进的后段降采样：减少高分辨率视觉编码成本，同时减少送入LLM的图像token，从而影响prefill/TTFT。本文只读选定章节，不复现论文headline、精度表或Mac/iPhone速度。

代码链更具体：`MobileCLIPVisionTower.feature_select`取`image_embeddings`，将BCHW变为`[B,H×W,C]`；LLaVA的`encode_images`随后另调用mm_projector。`export_vision_encoder.py`只trace vision tower，**不把projector一并封进Core ML**。Swift/Python patch都在Core ML返回后再运行两层Linear→GELU→Linear projector，转到语言hidden size。

Swift语言层是attention→residual→MLP→residual的顺序block，不是可直接把attention/FFN拆成独立branches的parallel block。视觉stage与语言stage也有数据依赖；若将来要多帧pipeline，必须另建跨请求slot、事件和背压合同，不能从模型名称中的hybrid推出设备overlap。

## 3. export recipe与“only-llm”真正保留了什么

原export先通过MPS加载PyTorch模型，再补processor/tokenizer/config文件，取固定image resolution构造F32输入、trace并导MLProgram。调用明确设置输出`image_features`为F32、compute precision F32、compute units CPU_AND_GPU、最低iOS16。这里的转换/加载选项不证明最终App里实际GPU执行，更不能拿它证明ANE执行。

`--only-llm` patch会排除vision_tower权重，将mm_projector参数改名到`multi_modal_projector.linear_*`，其余改为`language_model.*`，并将Core ML package复制到导出目录。它不是丢掉所有视觉消费者，也不是把整VLM转换成Core ML。LLM的4/8-bit权重选项不代表vision模型INT4/INT8、实际整数MAC或ANE低比特加速。

Swift projector固定为两层结构，而原Pythonprojector builder还支持linear、identity和不同深度GELU MLP；换checkpoint/config时必须核对真正参数结构。`tie_word_embeddings`、lm_head存在与否、tokenizer和图像token index也属于同一导出配方，不能只对齐safetensors文件名。

完整export会直接改checkpoint目录中的JSON，再产生/保存Core ML package，还可能有模型加载、MPS和trace副作用。因此本轮没有直接运行它，也未运行带目标目录清空逻辑的权重下载脚本。下面的片段验证只写本库新建的、无权重的synthetic fixture目录。

## 4. Core ML→MLX的buffer、shape与owner

Swift `VisionModelCoreML`用lock保护惰性load，`sanitize`会尝试预加载vision模型，但load错误在那里被`try?`忽略；encode时又经`try!`调用。所选tree没有生成的`fastvithd`类/实际模型资产，因此不能从这段调用恢复完整MLModelConfiguration，也不把load lock当多请求预测互斥或所有错误可恢复的证明。

输入路径先`asType(.float32).asData(access:.noCopyIfContiguous)`，取得转换后arrayData的Data及strides，在Data的可变bytes作用域内包装MLMultiArray并同步调用prediction。这保持了所读调用点的输入Data作用域；“noCopyIfContiguous”仍是带条件的访问选项，不是任意dtype/stride/真实Metal backing都零copy的证据。

输出只precondition检查`shape=[1,256,3072]`及F32，随后在scoped bytes闭包中按线性buffer构建新MLXArray。3072是vision feature宽度，不是语言hidden size；单份逻辑输出恰为3MiB，再经MLX projector生成语言embedding。该桥没有显式消费输出strides。Apple的scoped bytes API说明buffer layout/strides可能与dataPointer属性不同，不能只凭shape/dtype判定所有输出dense；实际Core ML返回和resolved MLX initializer的copy/retain语义仍需探针。本轮未提交非dense buffer或复现输出错位。

图像预处理也不是零成本共享：Core Image做resize/crop/normalize，vImage将RGBAf→RGBf→planar RGB，另建Data后生成`[1,3,H,W]`MLXArray。1024²时按逻辑容量计算，RGBA Data为16MiB、planar Data为12MiB；这不是RSS/峰值实测，MLX构造和框架内部还可能有其他存储。每帧还调用该CIContext的clearCaches，本轮未调用这段API或清理用户/系统cache。

App的`MLX.GPU.set(cacheLimit:20MiB)`只限制对应buffer cache，不把全部模型、KV、Core ML、CI/vImage工作集限制在20MiB。

## 5. 图像token是关键消费者ABI：原语句的32次实跑

Swift processor按`(height/64)×(width/64)`等配置生成连续`<image>`占位；1024²对应256个。App单图/B=1，vision输出又固定256×3072，不自动支持任意resolution、多图、video batch或projector。merge扫描token IDs中等于`image_token_index`的位置，再将projected features赋给那些embedding位置；tokenizer ID错了，后端推理成功也不能证明接缝语义正确。

本库probe从原`export`函数选出47–68行的**未修改AST语句**，只注入stdlib os/json/copy与自有fixture路径；不导入上游module，不构造假的模型/Tensor。8类配置×两种字典顺序×连续两次调用，共32次，原始before/after JSON和AST hash全部保留。

| 配置控制 | 实际tokenizer的image ID | 原片段写入model config | 结果 |
| --- | --- | --- | --- |
| 原本无image，其他最大ID为7 | 新建8 | 8 | 对齐 |
| 已有image=8，其他最大7 | 8 | 8 | 对齐 |
| image在中间，ID=2，其他最大7 | 2 | 8 | 返回正常但不对齐 |
| image为最大ID=99，其他最大7 | 99 | 8 | 返回正常但不对齐 |
| 已有image=0，其他最大9 | 0 | 10 | 返回正常但不对齐 |
| 只有image token | 5 | 未改写 | ValueError控制 |
| 空added-token map | 不存在 | 未改写 | IndexError控制 |

原因不是密码学或数值误差：原逻辑从token_ids列表排除已有image ID，却始终按剩余最大ID＋1设置model config，没有使用已发现的实际image ID。主矩阵24次返回中12次不对齐；另4次ValueError、4次IndexError是声明的合成边界。第二次运行不修复这些错配。两种dict顺序还保留了原先“复制首条token的其他metadata”的行为；本轮不验证真实Tokenizer如何解释这些flag。

初版收集器只预期ValueError，在空配置的实际IndexError处中止；保留初版client、partial fixtures和failed receipt后，修正的是**本库异常分类**，不是上游语句或判定门槛。新目录完整执行32次，再用独立进程重复，完整JSON逐字节相同。

这证明的是给定JSON进入此片段后的实际行为，不证明这些合成配置能通过完整load/export，也不证明官方发布的checkpoint含此问题。正确迁移合同应绑定真实Tokenizer输出ID、model config、占位数量、projector输出和最终merge，而不只检验导出成功。

另有未动态测试的Swift text-only分支：processor允许无图输入，但`inputEmbeddings`的无图分支返回language logits，prepare随后又按embedding传给language model；Python patch的对应分支则返回token embedding。不能从共享接口名称宣布两路text-only等价或已能运行。本轮不执行它、不修改上游，也不把源码shape风险写成App故障复现。

## 6. 相机pipeline、取消与TTFT的实际边界

相机采集先产CMSampleBuffer，分发到显示/分析时取imageBuffer，三个stream均用bufferingNewest(1)。分析循环对每帧await完整generation task完成，再读取下一份可用帧；因此它偏向丢弃积压旧帧，而不是每个capture帧都做预测，也不是vision(N+1)与LLM(N)已并行。显式frame PTS/代际关联未一路传到生成报告，单个TTFT不能当端到端frame age。

`generate`在已有running task时直接返回旧task，不为每个新传入UserInput排队。`cancel()`标记Task、清引用和UI状态，不await该task/同步Core ML预测退役；生成callback也通过Task.isCancelled合作停止，并排队额外MainActor UI任务。所读代码没有统一request generation ID来丢弃旧UI完成，故快速切换/取消重启需要专门测试；这不是已复现竞态或device drain失败。

TTFT的Date从`_load()`之后、进入modelContainer.perform后的processor.prepare之前开始，在首token callback入口计算；排队等待perform、model load、相机capture不在同一计时跨度，图像预处理/vision/投影/语言prefill到首token则在内部依赖链上。它也不包含全部响应、UI实际显示完成或下一帧新鲜度。每4token更新UI的注释倍率仅是作者观察，不能继承为本机稳定收益。

## 7. 可迁移设计与验证顺序

值得借鉴：减少视觉token而非只换设备；封闭vision子图与语言消费者分层；一次图像embedding参与prefill、语言KV继续decode；有界最新帧队列和单flight。需要补的资格：完整依赖/资产身份、实际Core ML policy/physical placement、返回stride/容量/owner、image ID与token数量、text-only/多图/分辨率约束、取消后旧输出隔离，以及完整请求质量和峰值。

源码中的研究导出选项、App默认load、已有MLX wheel、作者论文倍率和本轮JSON结果不能互相拼接成GPU/ANE速度榜。本轮空间仍低于原10GiB设备/build门，只做源码/有界CPU metadata：raw≤16MiB、源文本≤1MiB、curated JSON≤2MiB、CPU snapshot≤64MiB、空闲≥2GiB。未安装依赖、下载权重、运行export/Swift/App/camera或GPU/ANE，不清用户文件/cache；20项新测试与隔离范围见[验证（过程原件已移出）](../../MAINTENANCE.md)。
