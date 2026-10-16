# 来源、署名与独立实现边界

[总目录](README.md) · [设计样本](designs/README.md) · [机器可读吸收账本](references/absorbed-designs.json)

本目录原创文档、Python/Metal/Objective-C++ 教学代码遵循本目录的 [MIT license](LICENSE)（保留仓库原版权声明）。它们是用于解释机制和拒绝反例的独立实现，不是把外部项目的完整 runtime、模型代码或权重打包进来；不宣称格式兼容或第三方 bit-exact 复现。

思想和工程经验的明确出处：Splash/incoai（owned command payload、pack/consumer 协同）；带 ANE 扩展的 Splash snapshot（channel FFN、双 bank、rows calibration 与 handoff）；vpipe/tgo-app-dev（阶段内存协作、固定驻留子集和 typed refill）；Sana/NVlabs 的 sol-engine snapshot（capability/effect 接缝与 schedule）；KernelWiki/mit-han-lab（可搜索技术卡与证据等级组织）。另吸收mac_local_ai的消费者驱动ABI/精度岛、mac_transformer的拓扑与deadline planner、flux2-engine的persistent hybrid分工，以及mflux的FLUX.2单块图结构；[专项账本](references/gpu-ane-sources.json)记录实际读取文件SHA与工作树状态，新增demo仍为独立重写而非直接拷贝上述运行时。源码检查版本与原文件 SHA 保留在 [sources.json](references/sources.json)，**仅用于溯源，不是离线运行前置条件**。

Public Core ML 和 Metal 示例使用 Apple 公共 SDK/API。计算设备策略不等于实际 ANE placement，INT8 特征/存储不等于 INT8 硬件算术；SDK 与运行时要求由脚本和编译器明确检查。

本轮新增固定版本来源卡：ccv/Liu Liu（K split、surface/actual backing，选定lib源码BSD-3-Clause）；vpipe/T-Go LLC（graph内tiling与runtime权重，Apache-2.0）；Apple coremltools（公共计算计划，BSD-3-Clause）；Apple ANE Transformers（FFN布局，Apple sample-code条款及ACKNOWLEDGEMENTS）；Anemll profiler（结构化诊断与需隔离的副作用，MIT）。具体文件SHA、阅读范围和许可边界在[来源索引](sources/README.md)与既有登记中保留。新MIL图、host probe、监督与汇总代码为独立API客户端/学习实现，不分发这些项目的runtime或手写compiled-bundle代码。

2026-10-16新增MLX/Apple affine dispatch与GGML Metal导读。原创affine SG核不是MLX/GGML kernel移植；唯一新增直接代码摘录是[两段GGML dequantizer模板](assets/source-excerpts/ggml-dequant-64e9bce/dequantizers.metal)，来自`ggml-org/llama.cpp@64e9bceb...`，Copyright (c) 2023-2026 The ggml authors。其函数体未改、wrapper另写，[完整MIT许可](assets/source-excerpts/ggml-dequant-64e9bce/LICENSE)及[PROVENANCE](assets/source-excerpts/ggml-dequant-64e9bce/PROVENANCE.json)一同保留。不能将直接摘录称为本目录原创，也不能把所选模板实例执行当成整个上游engine验收。

若以后直接复制第三方代码/修改其具体实现，必须另行保留对应文件的版权、许可证和 NOTICE，并记录修改范围。不要因现在采用独立教学实现而推断原项目代码也可无条件改用 MIT；外部项目及权重仍受各自原许可约束。

GPU profiling新增[tmc/gputrace源码研究](sources/repos/gputrace-timing-provenance.md)（MIT，Travis Cline）与[Apple公共SDK合同](sources/official/metal-counters.md)。parser只在本地隔离构建/运行；Python adapter和公共Metal客户端为独立实现，不分发上游parser源码、Apple headers、framework或raw capture。官方文档/SDK不是MIT开源仓库，不以本目录许可覆盖其条款。

新增[MLX SDPA源码卡](sources/repos/mlx-sdpa-dispatch.md)继续以Apple MIT源码为阅读对象，不复制其shader实现。attention原配方replay复用本仓库MIT客户端/Transformer调用公式，产品库仅作显式可选本地接入；CPU oracle、候选API客户端和data verifier为原创工具，不打包MLX或模型。

[ANEForge来源卡](sources/repos/aneforge-rewrite-search.md)固定MIT源码与原作者Spencer H. Bryngelson署名；本机在ignored副本保留上游LICENSE/NOTICE并构建shim，不向Wiki分发该实现或binary。新增cache/gate教学代码、FP64 oracle、API probe和data verifier均为原创客户端；不把上游MIT许可扩大到Apple私有接口或系统软件。

[ane_pmu_profiler](sources/repos/ane-pmu-provenance.md)的固定tree未找到明确许可，只保留原创源码导读/独立API实验和脱敏结果，不分发上游source/binary/model。[measure_ane_capacity](sources/repos/ane-capacity-pmu.md)有BSD-3-Clause，Copyright (c) 2025 Koan-Sin Tan；本轮也是只读学习与本地编译，没有将App/模型/代码体纳入Wiki。PMU decoder、合成MIL/container writer、native客户端、独立数学witness与verifier为原创实现，格式/selector知识不等于Apple接口稳定或分发授权。

[真实Qwen private-prefix实验](review/qwen-private-ffn-20261016.md)继续只发布原创API集成、稳定公式/切片、独立数学witness与data verifier；原始checkpoint/激活/私有程序binary保持本地。ANEForge/MLX继续遵循其原许可，模型权重许可不由本Wiki改变；公开证据的hash/范围/误差/时间不包含原始tensor或用户会话。

[actual surface实验](gpu-ane/shared-surface.md)的MIL、Metal/native客户端、独立全量witness与curator均为原创，未复制无明确许可的PMU实现。[Apple公共合同](sources/official/shared-memory.md)只作带URL/内容hash的概括，不分发Apple headers或网页正文；公开API条款不被本目录许可覆盖，也不授予私有接口稳定性/发行资格。

[MLX external-memory来源卡](sources/repos/mlx-external-memory.md)按Apple MIT固定源码做原创概括，不复制其实现或分发headers/library。新增native客户端、周期合成配方、完整hash oracle和curator为原创；原始安装库和实验binary保持本地，不以本目录许可覆盖Apple SDK/MLX自身条款。

[真实shared FFN集成](review/qwen-shared-ffn-20261016.md)的ABI guard、Metal pack、完整MLX消费者、actual-part CPU join与data verifier为原创；复用的ANEForge MIL/blob保持来源/内容身份且不公开权重/编译产物。公开只含脱敏指标/hash。[clonefile官方来源](sources/official/clonefile.md)不镜像SDK/网页，SDK header的APSL2.0与Apple文档条款不由Wiki MIT覆盖；文件clone与私有接口没有新增分发/稳定性保证。

[ordinary完整block客户端](review/qwen-block-context-20261016.md)继续复用本仓库observer/调用配方作为显式可选integration；新增CPU oracle、codec与离线反例为原创工具。不分发模型激活、private常量或压缩BF16归档，合成字节fixture测试不作为实际模型交付资格。

[KernelWiki检索来源](sources/repos/kernelwiki-retrieval.md)固定MIT源码，Copyright (c) 2026 hanlab；本轮只读规则/查询/词表/版本检查器并独立实现本地合同，没有复制上游函数体、数据库、skill或CUDA资产，也未运行上游脚本。schema/主题词表不授予Apple API或硬件兼容资格。

[vImage来源](sources/official/vimage-stage.md)只概括Apple SDK选段，不分发headers/framework或宣称底层Accelerate开源。像素地址/PNG witness、公开API客户端、完整MLX阶段、curator与反例均为原创；实际链接的PE decoder/common来自本仓库MIT源码，产品配方重写注明来源。原图片/像素不发布，文件/alpha拒绝不当成完整媒体质量资格。

[ANE专题研究](ane/source-audit.md)补充Apple官方、maderix、ANEgpt、Orion、ANE-LM、Field Guide、ANEMLL、ANEForge等固定版本的原创综合，逐文件范围/哈希在[manifest](ane/research-manifest.json)。新增压缩/手写反向API客户端、标量Lloyd示例、数学witness和数据验证器均为原创，不分发上游runtime、Apple headers、博客正文、模型或编译产物。根许可提示不代替逐文件授权；私有API、训练和设备状态能力仍需独立验证，不由Wiki的MIT许可授予。
