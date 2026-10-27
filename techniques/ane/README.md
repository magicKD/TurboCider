# ANE 加速研究：结论、优化方法与源码学习导航

<!-- knowledge
id: source-ane-survey-index
type: navigation
-->

[百科总目录](../README.md) · [来源阅读指南](source-audit.md) · [实验设计方法](experiments-20261016.md) · [源码阅读](sources.md) · [历史案例](case-studies/README.md)

整理日期：2026-10-17。本目录按“为什么可能有效 → 怎样实现 → 何时适用 → 如何验证”组织。专题保留固定commit、实现链接、实验与反例；**源码机制、作者报告、本机组件数值、完整任务收益、物理ANE证据分别阅读**。不把跨机器数字拼成排行榜，也不因测试通过就默认启用产品后端。

阅读时先选方法或学习路径，再读固定源码和案例。实验原件、逐批 JSON 和校验快照不随库分发，案例结果不是当前验证。

## 当前结论：哪些方法值得借鉴

1. **先优化图和边界，再追求更多offload。** 固定阶段、少量较大的prediction、持续复用模型/权重/IO，通常是值得先验证的方向。Linear→Conv或BC1S不是减少数学运算，而是改善映射及producer→consumer布局；见[布局](layout-and-lowering.md)、[性能边界](performance-limits.md)。
2. **压缩、低比特计算和运行内存是三个问题。** LUT4是码本索引，W8常量也不自动等于W8A8整数MAC。应分别核对文件、解压后的权重、实际图与执行、质量和峰值内存；见[权重压缩](weight-compression.md)、[W8A8](w8a8-int8.md)。M4/A17 Pro相关能力的官方出处和版本范围在后者，不外推为所有图或旧芯片的绝对结论。
3. **GPU+ANE的价值在合法的并行临界路径。** 先证明切分代数，再测并发带宽竞争、pack/restore和完整consumer；不能把attention→FFN依赖链当两个独立分支。见[跨项目实现对照](hybrid-runtime-comparison.md)。
4. **状态、缓冲区与参数代际也是优化的一部分。** 复用可以省加载和复制，但必须守住实际backing/stride、最后读者、KV位置、checkpoint与失败提交点；见[生命周期](lifecycle.md)、[输出所有权](output-backing-and-stride-ownership.md)、[MLState](stateful-multifunction-and-retry.md)。
5. **训练研究的关键是参数如何更新而不重编译。** runtime weights、delta reload、LoRA输入、resident optimizer各有不同的数据流和数值配方；梯度、norm导数及Adam完整状态必须独立验收。见[训练路线](training.md)。
6. **测量必须与采用分开。** compute plan是计划，host span不是硬件忙碌时间；缺失counter不能写成0。最终依据同输入、同质量预算、强GPU基线和完整请求成本决定是否采用，见[profiling](profiling.md)。

没有通用赢家。本库既有成功的组件/边界验证，也有“表达可运行但数值不合格”“压缩有效但完整消费者更慢”的结果；这些反例应转成回归用例，而不是抹掉整个优化方向。

## 优化方法速查：从机制找到可借鉴实现

| 想改善什么 | 可以借鉴的实现方法 | 迁移时要保留的条件 | 源码导读与验证 |
| --- | --- | --- | --- |
| Linear、attention的图映射 | 1×1 Conv投影、BC1S、按head组织attention，沿后续算子传播布局 | 权重方向、bias、norm/softmax轴、mask、最终consumer恢复 | [Apple原Transformer](apple-transformer-norm-and-position.md)、[布局配方](layout-and-lowering.md) |
| dispatch太多、中间张量太大 | gate/up→activation→down图内融合；图内K/N tiling；query分块 | K归约后再做非线性，tail完整；query块仍看全部合法K/V | [图合同](coreml-graphs.md)、[vpipe调用链](vpipe-callchain-and-failure.md)、[Core AI分块attention](coreai-blocked-attention-and-runtime-gates.md) |
| 权重带宽与资产体积 | grouped-channel LUT、mixed-bit、embedding/lm-head独立压缩 | 同一checkpoint；分组/码本开销；敏感层与decoded-weight参考 | [压缩方法](weight-compression.md)、[真实LUT打包与resident成本](lut-packing-joint-compression-and-resident-cost.md) |
| 激活/权重传输与整数路径 | W8A8、校准、必要时做等价basis变换 | 真正插入所需Q/DQ；样本覆盖、scale、饱和与输出质量 | [W8A8](w8a8-int8.md)、[校准调用链](calibration-order-and-qdq-patterns.md)、[量化组合](quantization.md) |
| 首次加载与重复请求开销 | 固定shape/bucket、session/backing复用、分阶段驻留 | graph/权重/adapter身份；first、warm和系统cache冷态分开 | [生命周期](lifecycle.md)、[shape与hint](dynamic-shapes-hints-and-failure-boundaries.md) |
| LLM prefill/decode与KV开销 | full-batch prefill+单token尾部、layer chunk、stateful KV | state schema、有效prefix、position/RoPE/mask；最终norm只处于正确边界 | [公共部署](public-coreml-deployment.md)、[MLState与重试](stateful-multifunction-and-retry.md) |
| CPU/GPU/ANE间的交接 | 实际output backing、IOSurface、stride-aware view、受控buffer复用 | 使用真正返回的对象；保活到最后consumer；ready与release分开 | [输出backing](output-backing-and-stride-ownership.md)、[公共pixel/GPU交接](public-pixel-buffer-gpu-handoff.md) |
| 同一次FFN的并行 | paired gate/up切分、对应down列、partial归约；校准ANE share | bias/residual只加一次；并发两支实测，不能取isolated时间的max | [异构方案](hybrid-runtime-comparison.md)、[GPU-resident五臂](resident-ffn-layout-and-gpu-baselines.md) |
| 参数更新引起的编译/搬运 | runtime参数、delta reload、冻结base+LoRA输入 | 参数generation、scale、量化语义、失败后的完整状态 | [动态权重](training-dynamic-weights.md)、[delta/LoRA](training-delta-and-lora.md)、[原LoRA frontend](orion-lora-inputs-and-scalar-serialization.md) |
| 训练step中的往返与调用 | VJP图、device optimizer、resident W/m/v、匹配K步窗口 | 完整梯度/更新、epsilon、loss scaling、checkpoint；同一GPU训练窗口 | [Autograd](training-autograd.md)、[驻留训练](resident-training-lab.md)、[匹配窗口](matched-training-windows.md) |
| 回退与测量失真 | plan找图岛，host时间线找边界，设备工具另做归因 | 每字段记录单位、时钟域、scope、availability；正式计时无observer | [工具与SOP](profiling.md)、[Core ML实验性工具](coreml-experimental-profiling-and-inspection.md)、[原Swift统计/背压](coreai-profiling-reset-and-backpressure.md) |

表中方法是实验候选，不是已对所有模型证明提速。尤其不要将LUT压缩率、CPU+NE许可、shared surface或源注释中的容量阈值直接当硬件/性能结果。

## 直接读哪些项目

以下为固定读本，便于从实现追溯，不声称是最新分支。

| 项目与源码链接 | 最值得学习的部分 | 本库阅读路径 |
| --- | --- | --- |
| [Apple ANE Transformers](https://github.com/apple-aiml-research/ml-ane-transformers/tree/590eb35913622cdef217ebc5329047aab389ac98) · [官方文章](https://machinelearning.apple.com/research/neural-engine-transformers) | Conv投影、BC1S、attention与LayerNorm参数迁移 | [布局](layout-and-lowering.md)、[原代码/完整合成图验证](apple-transformer-norm-and-position.md) |
| [Apple Core ML Stable Diffusion](https://github.com/apple-aiml-research/ml-stable-diffusion/tree/ea2805dc1945be20561c77e5f6d1d9a5a637cda2) | split-einsum、query chunk、mixed-bit与阶段资源管理 | [SD实现导读](stable-diffusion.md) |
| [ANEMLL](https://github.com/Anemll/anemll/tree/f4ad26d061dc2e426faa19f81c367343bdeb9f0d) | prefill/decode、KV/state、模型分块、分组件LUT | [部署](public-coreml-deployment.md)、[状态合同](stateful-multifunction-and-retry.md)；最新迁移实验仍见[未完批次](../MAINTENANCE.md) |
| [vpipe emitter](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/apple-silicon/coreml/ane-emitter.cc) · [ccv rowwise bridge](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_coreml.mm) | runtime权重、图内tile、持久worker、backing与GPU交接 | [vpipe纵切](vpipe-callchain-and-failure.md)、[headroom/bias](headroom-bias-and-bridge.md)、[方案对照](hybrid-runtime-comparison.md) |
| [ANEForge](https://github.com/sbryngelson/ANEForge/tree/896c4943c5ebf811c0e815f8d1381fd1e157fa7d) | 图搜索、VJP、动态参数、优化器与resident state | [搜索](graph-rewrite-search.md)、[训练实现](training-autograd.md)、[Adam完整状态](adam-moments-loss-scaling-and-checkpoint.md) |
| [maderix/ANE](https://github.com/maderix/ANE/tree/d91c9845c0784dec7753048954fc6d0e8411fe29) · [Orion](https://github.com/mechramc/Orion/tree/4dcc1c2a626c9ee1f35576bd0b290d489db17d08) | 私有MIL、动态训练、常量权重重载和LoRA输入 | [私有栈](private-runtime-stack.md)、[训练路线](training.md)、[Orion LoRA](orion-lora-inputs-and-scalar-serialization.md) |
| [whisper.cpp Core ML bridge](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/src/coreml/whisper-encoder.mm) | 选择稳定encoder边界，保留既有decoder与明确IO | [语音调用链/布局/计时](speech-stages-profiling-and-failure-boundaries.md) |
| [Apple Core AI Models](https://github.com/apple/coreai-models/tree/3b25d934ccaee2d05edc6e281c555a5a9a13a55c) | 分块attention、静态函数菜单、host span与背压 | [源码路线](../references/coreai.md)、[attention](coreai-blocked-attention-and-runtime-gates.md)、[host组件](coreai-profiling-reset-and-backpressure.md)；不等于完整Core AI运行 |
| [anemll-profile](https://github.com/Anemll/anemll-profile/tree/bf8528105a0e61fda2c80d2c9b19615ecab9b7fa) · [anemll-bench](https://github.com/Anemll/anemll-bench/tree/865e78bf8199188be612809356dc90f78b925957) | 机器可读诊断、计划、prediction计时及指标定义 | [profiling工具审计](profiling.md)，先区分测量、派生与估计字段 |

更多固定版本源码、官方资料和阅读重点见[source-audit](source-audit.md)及[源码索引](sources.md)。获取仓库不表示读完或运行过，作者结果也不等于本机复现。

## 按主题阅读全部专题

- **硬件与执行入口**：[能力地图](capability-atlas.md)、[公共部署](public-coreml-deployment.md)、[私有运行时栈](private-runtime-stack.md)、[私有/产品隔离](private-ane.md)、[Core AI平台门槛](coreai-blocked-attention-and-runtime-gates.md)。
- **图表达与精度**：[布局/lowering](layout-and-lowering.md)、[图与shape合同](coreml-graphs.md)、[Apple原Transformer](apple-transformer-norm-and-position.md)、[图搜索](graph-rewrite-search.md)、[精度岛](precision-islands-and-observer-effects.md)、[除法/方差/观察接口](division-variance-and-observer-boundaries.md)、[RMSNorm/VJP](rmsnorm-epsilon-floor-and-backward-contract.md)、[headroom/bias](headroom-bias-and-bridge.md)。
- **压缩与整数路径**：[权重压缩](weight-compression.md)、[LUT打包/联合压缩](lut-packing-joint-compression-and-resident-cost.md)、[W8A8](w8a8-int8.md)、[量化/basis变换](quantization.md)、[校准顺序与Q/DQ](calibration-order-and-qdq-patterns.md)、[Core ML版本与所有权](coreml-versioned-calibration-and-ownership.md)。
- **部署、状态与交接**：[生命周期](lifecycle.md)、[MLState/多函数/重试](stateful-multifunction-and-retry.md)、[backing/stride/owner](output-backing-and-stride-ownership.md)、[pixel buffer→GPU](public-pixel-buffer-gpu-handoff.md)、[shape/hint](dynamic-shapes-hints-and-failure-boundaries.md)、[枚举/算子缩减](enumerated-shape-metadata-and-operator-reduction.md)、[Apple SD](stable-diffusion.md)、[语音阶段](speech-stages-profiling-and-failure-boundaries.md)、[ORT分区与缓存](onnxruntime-coreml-partitions-and-cache.md)、[ExecuTorch AOT](executorch-coreml-aot-and-estimates.md)。
- **GPU+ANE与模型运行时**：[方案比较](hybrid-runtime-comparison.md)、[vpipe调用链](vpipe-callchain-and-failure.md)、[Espresso Swift](espresso-runtime-walkthrough.md)、[GPU-resident完整FFN](resident-ffn-layout-and-gpu-baselines.md)、[性能/带宽边界](performance-limits.md)。
- **训练与参数更新**：[训练总览](training.md)、[动态权重](training-dynamic-weights.md)、[delta/LoRA](training-delta-and-lora.md)、[Autograd](training-autograd.md)、[原LoRA frontend](orion-lora-inputs-and-scalar-serialization.md)、[驻留训练](resident-training-lab.md)、[匹配K步窗口](matched-training-windows.md)、[四步展开/图身份](grouped-training-and-graph-identity.md)、[Adam状态与loss scaling](adam-moments-loss-scaling-and-checkpoint.md)、[Adam optimizer cuts](adam-optimizer-cuts-and-finite-range.md)。
- **Profiling与诊断**：[工具/SOP](profiling.md)、[Core ML实验性工具](coreml-experimental-profiling-and-inspection.md)、[constructor来源](profiling-constructor-provenance.md)、[request stats生命周期](request-stats-lifecycle.md)、[Orion pass/profiler字段](orion-rewrites-and-profiler-metric-provenance.md)、[Core AI统计/reset/背压](coreai-profiling-reset-and-backpressure.md)。
- **验证方法与追溯**：[实验设计](experiments-20261016.md)、[负结果→回归](negative-results.md)、[来源阅读指南](source-audit.md)、[分类证据](../MAINTENANCE.md)、[研究过程归档](../MAINTENANCE.md)。

## 三条学习与迁移路径

**做推理后端**：能力地图 → 布局 → 公共部署 → 生命周期 → 实际输出/状态 → 完整模型。先固定一段encoder或FFN；同时保留原表达与优化表达，使用相同权重和输入，检查完整consumer。

**排查慢或回退**：profiling → 性能边界 → resident FFN对照 → 图缩减。分别测compile/load/first/warm、pack/predict/restore/consumer，查计划与实际执行边界；数值失败的候选不发布提速比。

**研究训练/LoRA**：训练总览 → 动态参数与delta/LoRA → Autograd → norm/VJP → Adam与恢复。先做单步梯度和完整W/m/v/t，再做同等K步窗口与真实任务；不要以finite loss代替正确更新。

## 迁移前的最小验收清单

- 固定SoC/OS/SDK、源码commit、权重/adapter、shape/layout、dtype、mask和消费者；版本变化后重新判断。
- 先测同语义强GPU基线、同图串行控制与候选，再判断并发；预处理、返回转换、同步与内存峰值不从分母消失。
- 数值覆盖完整输出、非整除tail、异常范围及真正下游；训练还要完整梯度、更新和checkpoint。
- 记录每次失败、缺失测量和未覆盖域；不要用成功前缀、平均值、exit0或更宽容的新参考掩盖拒绝。
- 区分组件收益和端到端质量/延迟/能耗；局部通过才进入真实请求验证，不自动成为默认后端。

## 学习样本与来源

小型数学与所有权例子见[实现参考](../examples/README.md)，具体历史观察见[案例索引](case-studies/README.md)。它们帮助形成设计假设，不提供旧设备环境或复算工具。固定版本读本见[源码索引](sources.md)，材料边界见[维护规则](../MAINTENANCE.md)。
