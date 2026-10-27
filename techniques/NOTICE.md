# 来源、署名与独立实现边界

<!-- knowledge
id: technique-notice
type: method
-->

算法第二十六轮[DeepCache](algorithms/case-studies/deepcache-unet-boundaries.md)/[真实SD1.5 MPS](algorithms/case-studies/deepcache-sd15-mps-quality.md)对Xinyin Ma等Apache-2.0代码、Apache-2.0 Diffusers/Transformers和既有BSD类PyTorch作原创导读/API实验；原DeepCache package及许可仅在ignored固定hash副本执行，未向Wiki复制上游函数body或改安装包。Barry0121 MLX port固定tree无LICENSE、仅部分Apple copyright头，修改授权未确认：只静态阅读，不运行/分发其代码。SD1.5公共非官方镜像按card CreativeML OpenRAIL-M来源保留13文件/LFS身份，仅本地研究，Wiki不分发权重、真实激活或生成图，也不将其许可改成MIT。公开代码、合成CPU数据、派生hash/统计为原创；private ordinal实例适配/有限值hook不代表上游修改、语义/安全/媒体或默认资格，产品默认未改。

[Metal4 heap与生命周期研究](metal/case-studies/metal4-counter-heaps-and-time-domains.md)只对Apple SDK/官方DocC和MIT MLX固定Device选段作原创概括，不分发Apple headers、库或网页正文。新ObjC++/MSL合成消费者、无提交对象缩减、独立dyadic数学/时钟/标准库工具为原创；公开完整合成packet、metadata/hash和owned OS报告的去地址符号摘要，无模型/用户数组或原始crash地址。默认queue/整heap正控不授予坏路径修复、通用ABI/ISA、NAX/ANE或生产背书；上游、Apple和模型原权利不变，未对外写issue/PR或改系统默认。

算法第二十五轮[VDE MPS控制器研究](algorithms/case-studies/vde-mps-controller-cost.md)继续使用作者/Apache-2.0 Diffusers及BSD类PyTorch/Apple归属的固定源码作原创分析。新共同去噪循环、公共模型API客户端、独立标准库审计、CPU观察器校准及合成数据为原创；原VDE脚本和LICENSE仅留显式核验后的ignored本地，不分发上游函数body、kernel、模型或binary。Tensor.item诊断patch仅在owned进程/正式窗外、finally恢复，安装包/上游/产品默认未改。首次自有客户端快照与raw结果留ignored作排除谱系，不把原/变体输出等值当质量/物理copy证明；第三方及SDK许可不由Wiki MIT覆盖。

[枚举feature/算子缩减](ane/enumerated-shape-metadata-and-operator-reduction.md)对固定Apple BSD-3-Clause coremltools schema/序列化路径及公开issue的相反报告作原创分析；issue/API正文仅留ignored目录，不分发上游函数body。独立公共API客户端、合成图/参数/数组、完整数学、有界wire reader与自有模型protobuf状态证书为原创。编译模型/native binary、原始crash地址留本地，仅公开无地址摘要/hash；SDK、上游和社区内容权利不由本库MIT覆盖，未改上游或发布任何GitHub写入。

算法第二十三轮[T-GATE MPS resident研究](algorithms/case-studies/tgate-mps-resident-cost.md)沿用MIT作者包/Apache2 Diffusers，只本地调用；PyTorch BSD类及Caffe2归属保留，以固定源码作原创同步/fallback/选择条件导读，不分发其kernel/模型body或binary。公开客户端、预登记计划、独立标准库完整payload/journal/parity/统计重放和随机合成数据为原创；原源码、Apple SDK/运行库许可继续有效。参数只发布fingerprint，不含预训练权重/真实activation/媒体，API同步/环境开关不冒充物理ISA、all-op GPU或质量证明。

[动态shape/hint研究](ane/dynamic-shapes-hints-and-failure-boundaries.md)对固定Apple BSD-3-Clause coremltools、ANEMLL源码及SDK声明作原创导读；自有dense ReLU residual图、参数/输入/FP64参考、原生公共API/独立Python客户端与无损codec为原创，不复制上游runtime或SDK函数body。模型、编译资产、二进制及仅本次owned PID匹配的OS原始crash文件留ignored目录；公开仅自有合成tensor、元数据、SHA及无地址stack image/symbol摘要。SDK/第三方权利不由Wiki MIT覆盖，未改上游或生产默认。

[原MLX-LM prefill分块](metal/case-studies/mlx-prefill-chunking-and-numerical-routes.md)与[KV/lookahead来源](metal/case-studies/mlx-prompt-cache-provenance-and-lookahead.md)对同一固定Apple MIT源码作原创导读；调用原API的observer、teacher-forced客户端、完整CPU数学/字节审计、无损JSON发布与合同/tests为原创，不复制原runtime/shader函数体。沿用既有Qwen与WikiText2授权/署名，真实完整logits/KV/权重留ignored本地，公开只含metadata/误差/hash。原贡献限制、SDK及上游权利不变，未发PR/issue或更换生产策略；collector有效区修正不冒充上游修复。
算法第二十二轮[T-GATE](algorithms/methods/tgate-phases-and-branches.md)对作者MIT实现、GPL3 ComfyUI_TGate/Comfy核心及Apache2 Diffusers作原创导读/API实验。上游完整包/所选源码仅在显式固定Git核验后于ignored目录准备/执行，未向Wiki复制其函数body、README图片、真实模型、视频或binary，未改安装包/上游/产品默认。公开probe、独立条件CFG-DDIM/cache/位置参考、随机合成latent与cache bytes/hash为原创派生；原码/本地链接代码继续遵原许可，本库MIT不覆盖GPL/MIT/Apache第三方或授予独立全模型/attention数学、GPU或媒体资格。

[MLX GQA/score精度](metal/case-studies/mlx-sdpa-gqa-fallback-and-score-precision.md)及[wide GEMV/head归因](metal/case-studies/mlx-small-m-wide-gemv-and-head-attribution.md)对Apple MIT固定源码/PR作原创分析，不复制其shader/runtime函数体或分发wheel/metallib。新公共API客户端、逐阶段数学、CPU合同、tests与脱敏发布器为原创；原始模型/激活向量和第三方source/API正文留ignored本地。沿用既有Qwen模型和WikiText窗口的授权/署名，公开仅完整身份、shape、误差、少量scalar及终态；既有上游AGENTS贡献约束不变，未提交issue/PR或修改其源码。Wiki许可不授予硬件/模型/性能保证或第三方额外权利。

算法第二十一轮的[完整Tiny U-Net消费者](algorithms/case-studies/hidiffusion-unet-consumer.md)使用原Apache-2.0 HiDiffusion包和Diffusers0.39.0公开构造器/真实模型方法；原包body与key表仅在显式固定hash后的ignored目录准备/导入，安装包和上游均未改。公开自有probe/独立shape及条件CFG-DDIM参考、随机合成张量hex/hash与派生状态不含预训练权重、真实activation/媒体、上游body或binary。随机网络param manifest是指纹而非权重分发；条件scheduler校验不授予独立全模型数学/媒体/Apple性能或完整native镜像资格，第三方与SDK许可继续有效。

算法第二十轮的[HiDiffusion空间降算](algorithms/methods/hidiffusion-spatial-compute.md)对作者Apache-2.0源码/论文作原创导读；原factory/forward只在显式固定Git读取后于本地执行，不向Wiki复制其函数body、模块key全文、图片、模型或binary。公开探针、独立坐标/conv/pool/cubic参考、微型uniform-SDPA/affine fixture与派生数值/状态为原创；原方法、Torch和依赖许可继续有效，不把论文图像/作者速度当本机结果，也未改上游或产品默认。

算法第十九轮的[packing签名诊断](algorithms/case-studies/nnc-packing-signatures.md)为原创只读C适配器、Swift反射客户端和标准库重放；固定BSD3 ccv私有头仅在ignored本地构建include，未向Wiki复制其struct/function body或binary。上游未改、未安装compiler hook；公开证据只有有名参数字段、合成scalar、hash/receipt及自有编译失败摘要，无地址、opaque padding、模型或activation。沿用第十七/十八轮原GPL3 TeaCache的本地构建边界，私有ABI仅诊断、无稳定接口/模型/性能资格。

[MLX-LM完整batch/缓存研究](metal/case-studies/mlx-lm-ragged-cache-and-padding.md)及[异步交接](metal/case-studies/mlx-lm-continuous-batching-and-cache-handoff.md)对Apple Inc. MIT源代码作原创导读和独立API调用；原模型/生成器/cache函数体、native库及完整PR diff只在ignored副本保存，Wiki不分发。独立client/CPU合同/参考与统计为原创，源码checkout仅补明确标注的本地generated版本元数据。原Qwen BF16继续沿Apache2资产身份复用；WikiText/Wikipedia短token窗口沿前轮CC-BY-SA3.0/GFDL及文章署名，未重新授权为本库原创语料。未向上游创建PR/issue/comment/commit。

[held-out分布评分](metal/case-studies/qwen-heldout-distribution-scoring.md)/[FP16-Q8配对](metal/case-studies/qwen-gguf-baseline-lineage.md)新增原官方Apache-2.0 BF16/FP16模型及WikiText archive，均只留ignored outputs，不分发权重、完整语料或完整logits。公开方案只保留原Wikipedia文章标题、行号、源SHA与短token-ID窗口；它们来自WikiText/Salesforce及原Wikipedia贡献者，沿固定数据卡CC-BY-SA3.0/GFDL，不将派生窗口称为本库原创语料或以Wiki MIT覆盖。Qwen2参数映射/codec/metrics客户端为原创分析实现；SmoothQuant MIT源码只读类比，不复制其函数body，也不声称实际执行了其校准。

[原始llama.cpp完整模型/输出需求](metal/case-studies/llama-pretrained-output-demand.md)和[Flash/observer](metal/case-studies/llama-metal-flash-and-eval-observer.md)以固定ggml-org MIT源码作原创分析/API执行，不复制或分发上游模型、scheduler、shader body。新C++/Python客户端、纯CPU合同和归档器为原创；可选构建引用上游vendored nlohmann/json3.12.0（Niels Lohmann，MIT）及Apple公共SDK，只在ignored私有source/build编译，不把其body/库纳入Wiki。官方Qwen资产沿前轮Apache-2.0身份只读复用；公开capsule含原创受控prompt/ID、形状/hash/误差/时间，不含权重、完整logits、raw日志指针或binary。遵守上游AI贡献政策：仅本地学习/私有API实验，未创建PR/issue/comment/commit或修改上游。

[完整预训练Candle Qwen](metal/case-studies/candle-pretrained-qwen-and-tokenizer.md)及[Q8片段/请求成本](metal/case-studies/candle-q8-fragments-and-request-cost.md)继续对Candle MIT OR Apache-2.0及其GGML来源作原创源码分析，不复制模型/Metal实现body。新Rust请求协议、公开API客户端、独立有界GGUF/Q8 reader、完整FP64-stage参考与统计为原创。Qwen官方GGUF、tokenizer/config/README/LICENSE按固定revision/Git/LFS身份留本地ignored目录，遵循其Apache-2.0，模型权利不由Wiki MIT覆盖；公开capsule仅发布身份、事实参数、原创受控prompt/ID、输出hash/误差/时间，不分发权重、完整logits、上游库/二进制或大段文档。CommonCrypto/Metal为Apple公共API，SDK及trace/编译产物不随Wiki分发。

[PyTorch MPS SDPA/cache研究](metal/case-studies/pytorch-mps-sdpa-dispatch-and-layout.md)和[真实Qwen/profiler](metal/case-studies/pytorch-mps-profiler-and-real-attention.md)对固定PyTorch BSD-style许可及Apple/MLX来源标注的选段作原创分析，不复制上游runtime/shader body或patch。新增公共API客户端、标准库合同、独立完整FP64参考和脱敏统计为原创；18个source/API/PR身份、guard包络差异与许可证hash分别保留。既有Qwen真实激活/权重、安装wheel/native库、raw capture和上游diff仅留本地ignored目录，公开JSON只含身份/形状/误差/时间/hash；Wiki MIT不覆盖模型、PyTorch各第三方声明、Apple SDK或trace格式权利。未向GitHub提交内容，也不以PR closure推断merged。

[Candle packed/view](metal/case-studies/candle-packed-views-and-copy.md)与[原始Qwen2/queue/capture](metal/case-studies/candle-qwen2-queues-and-capture.md)对固定huggingface/Candle源码作原创分析/API执行，不向Wiki复制runtime、模型类或shader body。上游仓库MIT OR Apache-2.0及所含GGML等文件原声明继续有效。新Rust/Python客户端、独立codec/完整数学参考为原创；Cargo.lock仅作依赖身份，合成GGUF/raw/binary/trace留ignored目录，不以本库MIT覆盖上游、Apple SDK或预训练模型权利。

[原生Relax VM部署](metal/case-studies/mlc-runtime-only-relax-vm.md)与[stage/投影精度研究](metal/case-studies/mlc-ffn-stage-observation-and-precision.md)对固定TVM/FFI/MLC Apache-2.0源码作原创分析；新增C++部署、API诊断和CPU验证器为本库原创客户端，不复制上游runtime或shader body。生成MSL/host库、合成packed参数、完整raw与二进制留在ignored任务目录，公开capsule只含身份、指标与hash；本库MIT不改变上游或Apple SDK条款，也不授予模型/硬件资格。

[总目录](README.md) · [设计样本](designs/README.md) · [机器可读吸收账本](designs/README.md)

本目录原创文档、Python/Metal/Objective-C++ 教学代码遵循本目录的 [MIT license](LICENSE)（保留仓库原版权声明）。它们是用于解释机制和拒绝反例的独立实现，不是把外部项目的完整 runtime、模型代码或权重打包进来；不宣称格式兼容或第三方 bit-exact 复现。

算法第十八轮的[gate成本研究](algorithms/case-studies/nnc-gate-controller-cost.md)是原创NNC公共API客户端、批量统计/分母缓存变体、合成signed fixture及独立标准库验证器；沿用固定GPL3 TeaCache与BSD3 NNC/ccv，仅本地ignored构建注入未改原body。公开证据含原创派生hash/scalars/timings/自有失败客户端，不含真实activation、模型、上游body或binary；原库、Apple SDK和构建依赖许可继续有效，组件倍率不授予模型/媒体/产品资格。

算法第十七轮的[Draw Things原生缓存研究](algorithms/methods/draw-things-caching.md)固定读取GPL3的Draw Things、BSD3的s4nnc与ccv代码/各自第三方许可，原TeaCache/wrapper仅在显式本地Git核验后复制到ignored编译目录，不向Wiki分发上游body或binary。新增公共NNC API客户端、准备/准入/独立scalar核查及合成图为原创；保留的失败客户端源文本是本库自有研究代码，不是上游模型。原代码/构建产物继续遵其许可，本库MIT不覆盖第三方应用、模型、Apple SDK或授予完整模型/速度/媒体资格。

算法第十六轮的[AdaCache/MoReg研究](algorithms/methods/adacache-motion-and-period.md)固定读取作者Apache-2.0/Open-Sora来源，原sampler/root/block/helper仅在显式本地Git对象核验后受限提取执行；不分发上游body、模型、VAE、真实activation、权重或视频。新增tiny affine/identity/reshape/direct-checkpoint fixtures、完整scalar FP32 oracle/独立审计和文档为原创；本库MIT不取代上游及模型许可，也不授予GPU/媒体/产品资格。

算法第十五轮的[MagCache校准/consumer研究](algorithms/methods/magcache-calibration-and-consumer.md)固定读取MagCache作者与Diffusers Apache-2.0源码；选定声明只在本地显式opt-in、核对Git blob后执行，不向Wiki分发上游body、模型、权重或activation。新增CPU affine/zero-time/metadata fixtures、独立标准库oracle/审计与报告为原创，Wan AMP/JSON替代和Qwen自有toy class patch均明确标注；本库MIT不覆盖第三方模型/代码或授予媒体/产品资格。

算法第十四轮的[ToMeSD consumer/真实FFN研究](algorithms/case-studies/token-merge-consumer-and-real-ffn.md)固定读取ToMeSD MIT源码（Copyright (c) 2023 Daniel Bolya）；原函数仅在显式本地Git对象上核验hash后提取执行，不向Wiki分发原body或Diffusers/MPS模型。新增group-map/均值/独立审计、公共MLX客户端与高风险FFN-local迁移为原创；公开数据仅含toy向量、真实输入派生地图/统计/身份，不含原始模型activation、权重或媒体。原模型/依赖及MLX各遵其许可，组件观察不授予质量/产品或模型分发资格。

算法第十三轮的[空间residual缓存研究](algorithms/case-studies/spatial-cache-mlx-study.md)为原创数学移植、公共MLX API客户端与标准库审计，不分发MLX/Apple shader、原FasterCache body、模型或视频。公开JSON为原创affine/harmonic fixture的身份、逐plane误差/energy、计时及终态，不含真实模型activation。上游算法、MLX及文档继续遵循各自条款，组件观察不授予第三方模型分发、视频质量或产品资格。

算法第十二轮的[FasterCache consumer研究](algorithms/case-studies/fastercache-consumer-study.md)仅在显式指定的本地Git对象上核对hash并受限提取原函数/方法，不向Wiki分发上游body、SAT/einops、checkpoint、CUDA库或生成视频。新增affine替身、reshape adapter、标准库DFT与verifier为原创；完整向量均为原创微型fixture，不是模型激活。上游README保留其借鉴项目署名，本目录许可不覆盖其模型/依赖或授予额外分发权。

算法第十一轮的[原生cache完整请求研究](algorithms/case-studies/native-cache-request-study.md)新增原创C ABI客户端、CPU像素诊断与标准库verifier；仅调用本地source-matched产品库，不打包原生runtime、模型、原始激活或生成PNG。Cache-DiT机制与本库native实现分开溯源；公开JSON仅含提示词/身份、timing/step、RGB/alpha统计及终态。生成媒体与组件模型仍遵循各自上游条款，两prompt测量不授予模型分发或质量/生产资格。

Public Core ML 和 Metal 示例使用 Apple 公共 SDK/API。计算设备策略不等于实际 ANE placement，INT8 特征/存储不等于 INT8 硬件算术；SDK 与运行时要求由脚本和编译器明确检查。

本轮新增固定版本来源卡：ccv/Liu Liu（K split、surface/actual backing，选定lib源码BSD-3-Clause）；vpipe/T-Go LLC（graph内tiling与runtime权重，Apache-2.0）；Apple coremltools（公共计算计划，BSD-3-Clause）；Apple ANE Transformers（FFN布局，Apple sample-code条款及ACKNOWLEDGEMENTS）；Anemll profiler（结构化诊断与需隔离的副作用，MIT）。具体文件SHA、阅读范围和许可边界在[来源索引](sources/README.md)与既有登记中保留。新MIL图、host probe、监督与汇总代码为独立API客户端/学习实现，不分发这些项目的runtime或手写compiled-bundle代码。

2026-10-16新增MLX/Apple affine dispatch与GGML Metal导读。原创affine SG核不是MLX/GGML kernel移植；唯一新增直接代码摘录是两段GGML dequantizer模板，来自`ggml-org/llama.cpp@64e9bceb...`，Copyright (c) 2023-2026 The ggml authors。其函数体未改、wrapper另写，完整MIT许可及PROVENANCE一同保留。不能将直接摘录称为本目录原创，也不能把所选模板实例执行当成整个上游engine验收。

若以后直接复制第三方代码/修改其具体实现，必须另行保留对应文件的版权、许可证和 NOTICE，并记录修改范围。不要因现在采用独立教学实现而推断原项目代码也可无条件改用 MIT；外部项目及权重仍受各自原许可约束。

2026-10-17的[llama.cpp CB诊断](metal/case-studies/llama-command-buffer-timing.md)在ignored私有副本给13处commit加原创公共SDK观察器，原文件MIT版权/许可原样保留；Wiki来源capsule只保存最小patch及原始/修改SHA，仍受ggml authors MIT许可，见既有完整许可。observer/客户端/CPU归一化为独立学习实现，不分发Apple SDK headers/framework或上游binary，不以本库许可覆盖它们。

GPU profiling新增[tmc/gputrace源码研究](sources/repos/gputrace-timing-provenance.md)（MIT，Travis Cline）与[Apple公共SDK合同](sources/official/metal-counters.md)。parser只在本地隔离构建/运行；Python adapter和公共Metal客户端为独立实现，不分发上游parser源码、Apple headers、framework或raw capture。官方文档/SDK不是MIT开源仓库，不以本目录许可覆盖其条款。

新增[MLX SDPA源码卡](sources/repos/mlx-sdpa-dispatch.md)继续以Apple MIT源码为阅读对象，不复制其shader实现。attention原配方replay复用本仓库MIT客户端/Transformer调用公式，产品库仅作显式可选本地接入；CPU oracle、候选API客户端和data verifier为原创工具，不打包MLX或模型。

[ANEForge来源卡](sources/repos/aneforge-rewrite-search.md)固定MIT源码与原作者Spencer H. Bryngelson署名；本机在ignored副本保留上游LICENSE/NOTICE并构建shim，不向Wiki分发该实现或binary。新增cache/gate教学代码、FP64 oracle、API probe和data verifier均为原创客户端；不把上游MIT许可扩大到Apple私有接口或系统软件。

[ane_pmu_profiler](sources/repos/ane-pmu-provenance.md)的固定tree未找到明确许可，只保留原创源码导读/独立API实验和脱敏结果，不分发上游source/binary/model。[measure_ane_capacity](sources/repos/ane-capacity-pmu.md)有BSD-3-Clause，Copyright (c) 2025 Koan-Sin Tan；本轮也是只读学习与本地编译，没有将App/模型/代码体纳入Wiki。PMU decoder、合成MIL/container writer、native客户端、独立数学witness与verifier为原创实现，格式/selector知识不等于Apple接口稳定或分发授权。

[真实Qwen private-prefix实验](gpu_ane/case-studies/qwen-private-ffn-copy-bridge.md)继续只发布原创API集成、稳定公式/切片、独立数学witness与data verifier；原始checkpoint/激活/私有程序binary保持本地。ANEForge/MLX继续遵循其原许可，模型权重许可不由本Wiki改变；公开证据的hash/范围/误差/时间不包含原始tensor或用户会话。

[actual surface实验](gpu-ane/shared-surface.md)的MIL、Metal/native客户端、独立全量witness与curator均为原创，未复制无明确许可的PMU实现。[Apple公共合同](sources/official/shared-memory.md)只作带URL/内容hash的概括，不分发Apple headers或网页正文；公开API条款不被本目录许可覆盖，也不授予私有接口稳定性/发行资格。

[MLX external-memory来源卡](sources/repos/mlx-external-memory.md)按Apple MIT固定源码做原创概括，不复制其实现或分发headers/library。新增native客户端、周期合成配方、完整hash oracle和curator为原创；原始安装库和实验binary保持本地，不以本目录许可覆盖Apple SDK/MLX自身条款。

[真实shared FFN集成](gpu_ane/case-studies/qwen-shared-ffn-consumer-cost.md)的ABI guard、Metal pack、完整MLX消费者、actual-part CPU join与data verifier为原创；复用的ANEForge MIL/blob保持来源/内容身份且不公开权重/编译产物。公开只含脱敏指标/hash。[clonefile官方来源](sources/official/clonefile.md)不镜像SDK/网页，SDK header的APSL2.0与Apple文档条款不由Wiki MIT覆盖；文件clone与私有接口没有新增分发/稳定性保证。

[ordinary完整block客户端](gpu_ane/case-studies/qwen-block-context-and-storage-boundaries.md)继续复用本仓库observer/调用配方作为显式可选integration；新增CPU oracle、codec与离线反例为原创工具。不分发模型激活、private常量或压缩BF16归档，合成字节fixture测试不作为实际模型交付资格。

[vImage来源](sources/official/vimage-stage.md)只概括Apple SDK选段，不分发headers/framework或宣称底层Accelerate开源。像素地址/PNG witness、公开API客户端、完整MLX阶段、curator与反例均为原创；实际链接的PE decoder/common来自本仓库MIT源码，产品配方重写注明来源。原图片/像素不发布，文件/alpha拒绝不当成完整媒体质量资格。

[ANE专题研究](ane/source-audit.md)补充Apple官方、maderix、ANEgpt、Orion、ANE-LM、Field Guide、ANEMLL、ANEForge等固定版本的原创综合，逐文件范围/哈希在[manifest](ane/sources.md#research-manifest)。新增压缩/手写反向API客户端、标量Lloyd示例、数学witness和数据验证器均为原创，不分发上游runtime、Apple headers、博客正文、模型或编译产物。根许可提示不代替逐文件授权；私有API、训练和设备状态能力仍需独立验证，不由Wiki的MIT许可授予。

ANE持续深化新增[Espresso/vpipe/ANEForge实现阅读](ane/sources.md#implementation-sources-20261016)。Espresso为MIT，Copyright2026 Christopher Karani；vpipe继续按Apache-2.0，ANEForge按其MIT/NOTICE。原NEON/Metal head和CPU worker仅在本地编译/执行，不分发上游函数体或binary；新的Swift/C++客户端、训练adapter、完整NPY数学复核与偏差反例为原创。仅收录指定原实现的观察，不把helper问题扩大为引擎或Apple硬件结论。

ANE后续[constructor/SGD窗口](ane/sources.md#followthrough-sources-20261016)另阅读PMetal固定Rust runtime选段，并复读Espresso/ANEForge相关分支；只发布原创分析、元数据/constructor客户端、同SGD四步图、独立标准库数学审计和脱敏逐样本数据。PMetal/ANEForge等上游实现、Apple framework、原始数组和编译产物不随Wiki分发，原项目条款和Apple接口权利不由本目录许可覆盖。

[同等分组训练与图身份](ane/grouped-training-and-graph-identity.md)继续以Apple MIT的MLX源码作原创选段导读；新SGD分组、AST受限提取、规范化DOT和完整数据审计为本库原创客户端，不复制MLX compiler实现或发布原始指针/权重/compiled binary。图中的synthetic操作实例也不构成硬件dispatch或私有接口发行资格。

[request stats生命周期](ane/request-stats-lifecycle.md)继续以固定Espresso/PMetal选段作原创分析。新客户端只复用本库原创MIL/layout/hash工具，未复制上游profiler/runtime函数体；重新命名的本库旧main不执行。公开证据仅含合成完整tensor的SHA、selector ABI、非地址对象token和脱敏生命周期，原stderr/模型/binary/counter bytes不发布，私有API权利不由本目录MIT许可授予。

[headroom/bias研究](ane/headroom-bias-and-bridge.md)在本地只读编译/调用Apache-2.0 vpipe emitter，未将上游函数体、bundle或binary纳入Wiki；新API client、受控slot变换、全矩阵数学/桥接/codec审计为原创。coremltools 8.3的BSD-3-Clause源码仅作带commit/hash的选段解释，Python文件相同不等于native image attestation；完整原始模型数组/源码抓取正文留ignored目录，原项目条款不被本目录许可替代。

[公共Core ML delegate研究](ane/sources.md#delegate-sources-20261016)对Microsoft ONNX Runtime的MIT和PyTorch ExecuTorch的BSD-3-Clause源码作固定选段分析；上游函数体只从显式指定、digest匹配的本地Git对象受限提取执行，不打包到Wiki。新增ONNX小图/API客户端、mock IR、受限protobuf/NPY读取、独立完整数学与profile审计为原创；wheel、原始数组、compiled cache及原函数源码留本地，原许可/版权不因AST提取或本目录MIT而改变。provider事件与估计op区间不授予物理ANE或第三方runtime发行资格。

[公共输出交接研究](ane/output-backing-and-stride-ownership.md)继续对vpipe Apache-2.0、ccv选定lib BSD-3-Clause、ORT MIT及ExecuTorch BSD-3-Clause源码作原创分析，未分发或执行这些原wrapper。Core ML SDK只保留相对路径/hash和原创概括，不镜像Apple headers。新增native/API客户端、完整数学/stride审计、protobuf身份复核与合成证据为原创工具；模型包、原始数组和binary仍在ignored本地目录，Wiki许可不改变Apple或各项目原条款。

[MLState/多函数研究](ane/stateful-multifunction-and-retry.md)使用既有coremltools8.3库的公共MIL/资产导出接口，按Apple BSD-3-Clause固定源码作原创说明；新小图、native client、完整FP16状态数学、parser与curator为原创工具。ExecuTorch仅源码分析；ANEMLL所固定根tree未据此确定统一LICENSE，不猜成MIT，不分发其chat/KV源码体。两份新source clone和原始模型/合成数组/binary留本地，SDK仅相对路径/hash与原创概括；Wiki许可不授予上游源码、Apple状态接口或任何模型的额外发行权。

[公共pixel-buffer/GPU研究](ane/public-pixel-buffer-gpu-handoff.md)对ccv选定lib BSD-3-Clause实现作原创分析，不执行或复制其原bridge/transform；API/native/Metal client、完整region数学、观测mask forensic和curator均为原创。Apple SDK仅保留相对路径/hash与概括，原始synthetic模型、buffer、binary和compiler输出留本地，不以Wiki MIT覆盖Core ML/CoreVideo/IOSurface/Metal条款；失败协议及tracked属性不构成底层同步或硬件placement保证。

[resident FFN五臂研究](ane/resident-ffn-layout-and-gpu-baselines.md)对Apple ANE Transformers sample-code条款及MLX MIT、coremltools BSD-3-Clause源码作原创导读；不分发原模块或shader。新dense配方、公共CoreML/MPS及MLX client、完整CPU FP64参考、gate/统计验证器均为原创；installed headers/library/metallib及模型/weights/数组不纳入Wiki。SDK只保存相对路径/hash；库名/计划/存储dtype不授予物理ANE、累加器或第三方代码的额外权利。

[精度岛与观测图研究](ane/precision-islands-and-observer-effects.md)对固定coremltools8.3 BSD-3-Clause源码/typed文档作原创分析，不镜像其converter/bridge实现。新MIL候选、桥接观测、标准库half舍入合同与独立完整参考为原创；native/helper复用本库既有原创工具。原始tensor、模型包、binary与日志留ignored本地目录，公开JSON只含合成metadata/metrics/hash；最低tensor精度、软件计划和Wiki许可不授予物理ISA、Apple接口或上游代码的额外保证/权利。

[校准顺序/QDQ研究](ane/calibration-order-and-qdq-patterns.md)对同一Apple BSD-3-Clause固定源码/文档作原创导读。五个原helper函数体仅从显式指定的固定Git对象受限提取执行，不向Wiki分发上游body；来源条款不因AST提取改变。新合成模型、临时观察wrapper、union/连通分量参考和标准库完整raw审计为原创；compiled包、raw和上游临时清理日志留本地。只发布合成metadata/metrics/hash，未修改或发行patched CoreML，也不授予最新bug状态、整数算术或原模型资格。

[三版本/所有权研究](ane/coreml-versioned-calibration-and-ownership.md)继续对Apple BSD-3-Clause固定源码及primary发布/PR资料作原创分析；selected原函数仅在显式固定的本地blob上受限执行，不打包其body或上游test/patch。来源抓取、CPU调用桩、owned sentinel guard、独立标准库审计均为原创；原始源码/API正文留ignored本地目录，wiki保留hash、状态和原创概括。未分发新wheel/native代码或改变其条款；CPU哨兵释放、PR状态及Wiki许可不授予native GIL/ANE/模型资格。

[LUT打包/十臂resident研究](ane/lut-packing-joint-compression-and-resident-cost.md)对coremltools BSD-3-Clause压缩/MILBlob源码作原创导读，不分发其packer/decompressor body、源码包或库。标量codec、weighted-Lloyd、递归常量parser、十臂client（派生自本库原创五臂）、独立完整参考与统计为原创工具；SDK/MLX source权限沿既有来源记录。真实raw/weights/compiled包/binary留本地，公开JSON只含合成metadata/hash/误差/时间；LUT index与量化centroid不授予Apple整数ISA、模型或第三方代码额外资格。

[语音阶段研究](ane/speech-stages-profiling-and-failure-boundaries.md)对whisper.cpp与WhisperKit/Argmax的MIT源码作原创调用链分析；原wrapper、Swift声明选段与许可证仅复制到ignored本机实验目录用于编译，仓库不分发第三方函数体、动态库、ASR模型或tokenizer。独立fake dispatch、观察/无observer client、Swift fixture、CPU复核与报告为原创；33文件manifest保留原commit/SHA和许可/AGENTS范围。公开tiny logical/受控数组完全来自自有公式；真实返回的未定义padding内容已从公开证据redact，只保留hash/诊断。学习实验不构成上游AI生成PR、修复提交或私有硬件性能背书。

[Apple原Transformer研究](ane/apple-transformer-norm-and-position.md)遵循固定树的Apple Software许可和ACKNOWLEDGEMENTS中的第三方声明，非简单统称MIT。原reference模块和一个原迁移hook只在ignored本机目录执行，不随Wiki分发第三方函数体、trace/compiled模型、库或checkpoint。collector、hash配方、scalar FP64/IEEE软件half oracle、portable tests及中文分析为原创；公开全部参数/输入/输出都是自有合成数据。CoreML8.3 Python loader只作BSD-3-Clause来源比对；原项目/Apple/第三方权利及不背书条款不因本地运行而改变。

[Adam状态/VJP研究](ane/adam-moments-loss-scaling-and-checkpoint.md)固定读取MIT ANEForge及其AGENTS/许可，原Python/native复制包、预建shim和MIL只留ignored本机目录，不分发上游函数体或库。原创collector/state adapter、完整FP64数学、bounded lossless codec、curator/tests和中文报告发布自有合成tensor与metadata/hash；无真实dataset/checkpoint或用户输入。48次原Trainer进程的数值观察不授予上游改动/PR、硬件placement、长期训练或性能背书，既有第三方许可和权利不变。

[Adam optimizer-only/切口研究](ane/adam-optimizer-cuts-and-finite-range.md)继续遵守同一MIT ANEForge固定树/AGENTS与既有shim来源；调用原helper的复制包和MIL仍只留ignored本机目录，未发布第三方函数体、动态库或补丁。新候选表达、collector、完整逻辑array/数学/±0 identity验证、curator/tests和报告为原创；公开数据均为前批自有合成状态及本批合成控制，不包含模型/用户数据。图级有限除法现象不授予私有硬件规格、训练或性能背书。

[独立除法/方差研究](ane/division-variance-and-observer-boundaries.md)对固定MIT ANEForge及Apple BSD-3-Clause coremltools源码作原创导读；第三方复制包、native扩展、compiled模型和cache仅留ignored本地，不分发其函数体/库。新collector、两套数学域参考、完整比较/curator/tests与自有合成数组为原创；发射MIL是这些自有数学图的可追溯结果，不是上游原训练实现。所有非有限输出仍为自有数据，cache缺失不回填或重建为旧镜像；Wiki许可不授予IEEE、硬件精度或上游背书。

[RMSNorm/VJP完整消费者研究](ane/rmsnorm-epsilon-floor-and-backward-contract.md)继续遵守固定MIT ANEForge源码/AGENTS及既有shim来源。PyTorch2.5.1三份文件仅作定义/多作者BSD式许可阅读，不运行、改写或分发其函数体/库；版权/不背书边界按原LICENSE保留。原创dense-adjoint、连续数学/FD/方向诊断、bounded codec/blob parser、collector/curator/tests与报告只发布自有合成tensor、生成MIL和metadata。原源码、cache字节副本与模型仍在ignored目录；归档file-at-read身份不授予native image、物理ANE、真实训练或性能资格。

[Orion LoRA/pass研究](ane/orion-lora-inputs-and-scalar-serialization.md)与[profiler来源分析](ane/orion-rewrites-and-profiler-metric-provenance.md)固定读取MIT Orion（Copyright (c) 2026 Murai Labs），原C/ObjC编译/调用副本只留ignored输出。vpipe Apache-2.0（Copyright 2026 T-Go LLC）的容器结构和CoreML BSD-3-Clause protobuf/API只作注明来源的格式导读，未调用vpipe的删除式writer或分发原函数体/库。新client、FP64/IR/varint/metric检查器及public sidecar实现为原创；公开仅为原工具在自有合成图/数据上的metadata、全部自有tensor、hash和明确注入的时长/RSS。第三方license、权利与不背书边界保留，不授予private/模型/训练/硬件性能资格。

算法第四轮新增[DiCache固定实现导读](algorithms/methods/dicache.md)与[Sana mask复用接缝](algorithms/case-studies/sparse-route-reuse.md)。上游函数body仅从用户显式指定、digest匹配的本地Git对象提取执行，不打包到Wiki；原创CPU替身、标量证据与独立verifier不是上游代码许可的替代。所读DiCache根tree没有独立LICENSE，Wan文件还保留原作者权利声明，因此不将上游实现改发为本目录MIT。论文只链接并原创概括；SVDCache同名候选的MIT LICENSE既不证明作者关联，也不授权另一项目/论文或不存在的代码。

算法第五轮对[Sage/Sparge](algorithms/methods/quantized-attention.md)的Apache-2.0固定源码作原创分析；源body仅在显式本地AST probe读取，不向Wiki分发CUDA/Triton代码或编译扩展。新增quantization参考、CPU参数桩、标量合同和独立curator为原创工具，不能冒充上游packed/ISA实现；公开JSON仅含输入身份、plan、标量误差/存储与调用元数据，不含QKV、checkpoint或用户媒体。NVIDIA PTX文档只作ISA语义链接，不授予Apple硬件或上游源码的额外权利。

算法第六轮的[MLX packed低秩组件](algorithms/case-studies/lowrank-mlx-component-study.md)为原创公共API客户端、合成矩阵与独立verifier，未复制Nunchaku CUDA body、MLX shader或分发二进制/模型；固定源码和作者blog只链接并原创概括，仍遵循各自上游条款。量化carrier的解码数学不是原项目完整packing兼容承诺；旧失败脚本仅在本地ignored归档，当前Wiki不授予其离线重放或reproducible-build资格。

算法第七轮的[SLA/SageSLA导读](algorithms/methods/sparse-linear-attention.md)、CPU包络、标量数学与verifier为原创研究；上游class/helper body只从显式指定、hash匹配的本地固定Git对象读取，不打包到Wiki。上游Apache-2.0声明和原文件头保持其效力，不因AST提取变为本库独占代码。[SpargeAttention2/SLA2](algorithms/methods/spargeattention2.md)论文只链接与原创概括，不分发训练代码、checkpoint或媒体；公开JSON只保留合成fixture身份、route、误差与小型梯度统计。

算法第八轮的[MLX TeaCache生命周期研究](algorithms/case-studies/mlx-teacache-lifecycle.md)为原创包络/独立verifier，原Apache-2.0源码只从显式指定、完整hash匹配的固定Git对象读取，不随Wiki分发或更改上游声明。公开数组仅为原创微型affine fixture的输入/输出/状态，没有原始模型activation、权重、媒体或mflux运行时；本库branch ledger和成本算术不是上游控制器复制品。

算法第九轮的[真实Qwen低秩研究](algorithms/case-studies/real-qwen-lowrank.md)仅在本地显式读取已有权重/原始activation与新owned capture，不分发模型、activation向量、媒体、因子或编译二进制。公开JSON为完整身份、统计、少量scalar与拒绝记录；原模型权利继续遵循其自身授权。[DeepCompressor](algorithms/sources.md#s43)的Apache-2.0源码只作固定版本阅读与原创分析，未复制其solver、CUDA body或执行其转换pipeline；本库随机分解、diagonal-RMS实验、数学合同与verifier为原创，不能冒充作者W4A4配方。

算法第十轮的[真实BF16 FFN GPU客户端](algorithms/case-studies/real-ffn-gpu-study.md)为原创公共MLX API实验，沿用已有本地原件，不分发weights、activation向量、低秩因子、native扩展或媒体。公开数据为全部timing、derived row误差、少量scalar与身份/终态。MLX固定源码仅链接和原创分析；source split-K参考是显式条件下的公式witness，不包装成上游shader或实际binary trace，各上游权利声明继续有效。
