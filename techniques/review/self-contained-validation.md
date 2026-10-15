# 自包含扩展：要求、证据与边界

[总目录](../README.md) · [设计样本](../designs/README.md) · [阶段方案](../models/stage-map.md) · [来源账本](../references/absorbed-designs.json)

2026-10-15，基于`193b356`，本轮只修改techniques，不改变产品runtime/default，不下载模型。本文对应本次自包含扩展；[上一轮硬件记录](labs-validation.md)与其中旧源码SHA保持历史含义，不重标为本轮测量。

## 对用户要求的完成判据

| 要求 | 实现与验证入口 |
| --- | --- |
| 不依赖其他项目阅读核心设计 | 9篇designs，问题/算法/数据布局/owner/失败/迁移/controls完整；absorbed-designs.json关联本地源码与测试 |
| 必需代码包含在本目录 | catalog必需code按techniques相对路径，禁止逃逸；原始产品入口仅在integration_code与integrations中可选保留 |
| Splash精髓 | owned graph、pack/consumer、完整FFN/LoRA代数、双bank、ready/done与校准设计；contracts、partition_axes、FFN样例及真实设备入口 |
| vpipe精髓 | typed refill、固定subset、防抖/ratchet、reserve减已分配、阶段alias预算，含反例和策略代码 |
| Sana精髓 | capability/effect、exclusive seam、逐step/stage策略、build/load与runtime区别、量化/复用组合质量边界 |
| Encoder / prefill / decode / DiT / VAE / pipeline | 8篇阶段地图/选择指南/方案，model_stages.py给出正负数学样本，完整pipeline配套IO/测量/预算工具 |
| 自包含不是只改检查器忽略错误 | check_standalone --isolate复制且仅复制techniques，实际执行完整CPU测试、所有轻量cases和本地审计 |
| 区分设备、模拟、近似和未实现边界 | 每个report明确scope/production_qualified；native INT8仅单matmul，CPU概率仅单token；完整产品fixture另保留 |

补充要求：已只读核查gpu_ane下mac_local_ai、mac_transformer、flux2-engine、mflux共4项目/17个选定文件；工作树dirty情况与SHA保留，不改参考工程。[专项综述](../references/gpu-ane-projects.md)关联3个新设计、模型选择指引和7个独立demo。

## 最终验证结果

| 检查 | 结果与边界 |
| --- | --- |
| 两套Python测试 | Python3.9.6 / 3.11.16各107个test methods通过、0skip；不是214项不同测试 |
| 逐章实现映射 | 45技术章节 / 45张卡 / 40个轻量case，全部必需code在techniques内 |
| 纯目录隔离 | 新临时目录只有techniques，未复制产品/兄弟源码；忽略继承PYTHON路径；完整107项测试、40 cases、wiki/catalog/依赖检查均通过 |
| 文档与来源 | 79篇Markdown，本地入链/链接/local source检查无错误；另显式核查105项完整来源、28项snapshot，无error/warning |
| 静态依赖 | 核查28个核心Python文件、3个native/Metal源码；必需资产、import、native quoted include无外部源码依赖；另有逃逸/缺阶段/动态加载反例测试 |
| 隔离Metal | M33/N41/K29真实GEMM，8 hot/4 warmup；CPU oracle最大绝对误差约1.11e-7，finite与tail guard通过 |
| 隔离Public Core ML | 本地frozen/runtime FP16均实际12次预测（含4 warmup），同一合成权重SHA、独立FP64 oracle；相对L2小于0.02预定toy gate |
| 隔离GPU/CoreML分工 | 真实MLX full GPU / GPU suffix + CoreML prefix的ABBA，6次CoreML调用（含warmup），完整restore/join边界；不称物理ANE overlap证据 |
| 隔离native INT8 shared IO | 两代输入，Metal producer -> public prediction -> Metal consumer；maxabs=0.0001220703125 < 0.003，tail guard通过，本次output fallback copies=0，private-class string检查为空 |
| 原完整W8A8产品接入 | 原脚本保留到integrations，只修正位置定位并标明可选；本轮未重跑这组产品fixture，不以新单matmul替代它的历史完整FFN结果 |
| 源码与原始报告 | [可移植摘要](self-contained-evidence-20261015.json)记录36个核心代码/测试SHA和原始报告SHA；设备报告中的5个直接源码绑定与当前内容逐项核对 |

真实设备为Apple M4 Max、macOS26.6.2；CoreML9.0仅通过既有隔离overlay使用，项目原环境未被替换。原始大报告留在ignored outputs，可移植摘要无个人绝对路径。CPU状态机/代数样本与设备样本分别标注范围，结构检查不替代语义审稿。

小M33/H128/F512边界的优化GPU host median约0.188ms，Python hybrid约0.233ms，**本次hybrid仍较慢**；这是继续研究host/restore/shared边界的负结果，不作加速承诺。不同phase/load/量化策略需要新matched trial，不能从编译成功或ANE policy推出收益。

第一次沙箱运行不能访问Metal，保留失败报告后获授权重跑才算设备通过。Metal的fastMathEnabled deprecation warning保留，不把它当数值失败或悄悄换fast-math配方。没有清系统cache、改wired/swap、电源或终止用户服务。

## gpu_ane补充验收

新增parallel-branches、attention-head-shards、consumer-strides、segmented-ffn、optional-devices、resource-hazards、precision-islands，按技术点独立运行；13项新test methods含正反例。完整107项测试与40 cases在techniques-only副本实际运行，没有导入被参考项目。三设备planner明确同时展示CPU退出、三设备参与与完整GPU回退，不预设设备必须参与。

这一补充仅增加CPU机制/设计与分类，未改变已实测的设备入口。设备5项结果沿用前段隔离实测，已重新核对其直接源码和local helper SHA完全未变；不把7个新demo称作新的设备/模型性能结果。摘要中snapshot_provenance明确区分两个验证范围。

## 完成范围与不冒充的能力

人工逐篇核对：9设计的算法/布局/生命周期/失败路径对应本地实现；各stage区分保持语义和近似，并提供错误反例、测量边界与重开条件。Splash/vpipe/Sana原文件路径只做可选出处，学习所需的核心内容已经在本目录。

没有实现整个新模型后端、完整多token speculative engine或通用2D/3D VAE；这些不是机制样本所声称的能力。也没有运行整产品make test、真实权重/LoRA画质、多机/低RAM媒体矩阵或物理ANE trace，不对上述范围给通过/速度资格。核心教学实现可独立复用，生产接入仍须真实adapter与完整请求验收。

## 复现

```sh
python3 -B -m unittest discover -s techniques/tests
python3 -B techniques/labs/check_catalog.py
python3 -B techniques/scripts/check_wiki.py
python3 -B techniques/scripts/check_standalone.py --isolate
```

默认不检查产品/兄弟仓库；若在完整仓库中审阅可选来源，额外运行`python3 -B techniques/scripts/check_wiki.py --include-integrations`。独立硬件CLI和依赖见[ANE](../labs/ane.md)、[GPU](../labs/gpu.md)；没有的硬件/工具不能算作通过。任何样本都不授予真实模型端到端收益、真实LoRA画质、低RAM资格或物理ANE占用证明。
