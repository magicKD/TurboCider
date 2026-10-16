# ANE 加速研究：从硬件能力到推理、训练与性能取证

[百科总目录](../README.md) · [来源与阅读范围](source-audit.md) · [本轮实验](experiments-20261016.md)

核查日期：2026-10-16。面向 Apple Silicon 的学习报告，不是产品后端启用说明。本文综合官方资料、固定版本源码及本库实验；**源码存在、上游性能报告、本机数值通过、完整模型提速、物理设备证据是不同维度**。原始模型、下载源码和编译产物不随文档分发。

## 最先弄清的五件事

1. **M4/A17 Pro 的 ANE INT8×INT8 吞吐提升有官方依据**；不是所有名叫 INT8 的模型都走这条路，也不据此断言旧芯片完全没有整数能力。只有 W8 常量、runtime INT8 surface 或 Q/DQ 节点，均不足以单独证明物理整数 MAC。见 [W8A8](w8a8-int8.md)。
2. **4-bit LUT 是码本索引压缩，不等于 INT4 乘加**；它在 M4 以前已有价值。JIT 解压、带宽、LUT 分组和精度才是主要考察对象。见 [LUT 与压缩](weight-compression.md)。
3. ANE 不只适合把整个模型搬过去。固定 encoder、FFN 子图、GPU/ANE 合法分担，各有不同的 setup、IO、内存和临界路径。见 [公共部署](public-coreml-deployment.md)、[异构对照](hybrid-runtime-comparison.md)。
4. **训练不是把 inference 开关改成 training**。研究项目将反向传播写成普通计算图；真正难点是参数更新后如何不重编译、梯度/优化器精度和跨设备状态。见 [训练路线](training.md)。
5. `CPU_AND_NE` 仍允许 CPU。本轮小图及手写反向实验的计划首选 CPU；大图才首选 ANE。不能把这些实验全部标为“ANE 实测”。见 [profiling](profiling.md)、[实验报告](experiments-20261016.md)。

上述能力结论的官方链接、版本条件和反例均在对应专题页；没有将社区 README 的硬件推断自动升级为规格。

## 按学习问题选报告

| 问题 | 阅读入口 |
| --- | --- |
| M1–M4 以后到底支持什么？BF16、INT8、LUT、SRAM？ | [能力地图与证据分级](capability-atlas.md) |
| Linear 为什么改成 1×1 Conv，BC1S 到底是什么？ | [布局与 lowering](layout-and-lowering.md) → [既有图合同](coreml-graphs.md) |
| W4、LUT4、W8A8、旋转量化有什么区别？ | [存储/解压方案](weight-compression.md) → [W8A8 路径](w8a8-int8.md) → [SmoothQuant/Hadamard](quantization.md) |
| Core ML LLM 怎么管理 prefill/decode、KV、分块？ | [ANEMLL、Whisper 与 Core AI](public-coreml-deployment.md) |
| 图像生成里怎么使用 ANE？ | [Apple Stable Diffusion 的 attention、分块与生命周期](stable-diffusion.md) |
| Private ANE 到底绕开哪一层？ | [低层运行时对照](private-runtime-stack.md) → [产品/研究隔离](private-ane.md) |
| ANE 能训练哪些部分？ | [训练总览](training.md) → [动态参数](training-dynamic-weights.md) → [delta/LoRA](training-delta-and-lora.md) → [autograd/驻留优化器](training-autograd.md) |
| GPU + ANE 为什么有时更慢？ | [vpipe/ccv/Splash/Draw Things 对照](hybrid-runtime-comparison.md) → [性能边界](performance-limits.md) |
| 怎样确认回退、瓶颈、占用、能耗？ | [profiling 工具分层与 SOP](profiling.md) |
| 有哪些值得复现的失败？ | [负结果与来源纠错](negative-results.md) → [图搜索失败缩减](graph-rewrite-search.md) |
| 怎样复跑本次实验？ | [命令、误差、全部结果与限制](experiments-20261016.md) |

## 项目定位：比较技术方案，不做跨机器速度排行榜

| 项目/项目族 | 主要路线 | 本次证据入口 |
| --- | --- | --- |
| Apple coremltools / ANE Transformers | 公共 MIL 转换、压缩；BC1S、Conv、attention 重写 | [布局](layout-and-lowering.md)、[量化](w8a8-int8.md)；源码/官方资料及原创实验 |
| Apple ml-stable-diffusion | 分阶段 Core ML、split-einsum、query chunk、混合 LUT | [SD 报告](stable-diffusion.md)；选定源码，不重跑整模型 |
| ANEMLL | Conv 投影、stateful KV、prefill/decode、分组 LUT | [部署](public-coreml-deployment.md)；模型/转换器选定范围 |
| whisper.cpp / WhisperKit | 稳定语音阶段或完整 Core ML 音频流水 | [部署](public-coreml-deployment.md)；前者 bridge 源码、后者文档入口 |
| Apple coreai-models | 新公共 Core AI 导出与 Swift runtime | [部署](public-coreml-deployment.md)；当前 Mac 缺 SDK/OS27，未运行 |
| maderix/ANE / MrZaKaRiA/ane fork | `_ANE*`、MIL、动态参数、手写反向、INT8 探针 | [动态训练](training-dynamic-weights.md)、[INT8 审核](w8a8-int8.md)；所选commit相同，不算两份独立证据 |
| ANEgpt | 动态参数训练、部分 CPU 算子继续移至 ANE | [训练](training.md)；代码/README 区分，未复跑 LLM |
| Orion | 常量图重载参数、delta patch、adapter 输入 | [delta 与 LoRA](training-delta-and-lora.md)；runtime/frontend/tests 选定范围 |
| ANEForge | e5rt、图搜索、VJP、runtime parameters、驻留 state | [运行时](private-runtime-stack.md)、[autograd](training-autograd.md)；已有推理实测，训练源码研究 |
| ANE-LM / Field Guide / ane.cpp | 常量 Conv、融合、FFN chunk、W-lane batching、负结果 | [运行时](private-runtime-stack.md)、[负结果](negative-results.md) |
| vpipe / ccv / 历史 Splash FFN / Draw Things | runtime weights、图内 tiling、GPU stager/shared IO、异构调度 | [异构](hybrid-runtime-comparison.md)；固定来源和已有真实 FFN 对照 |
| anemll-profile / anemll-bench | 计划/成本/日志与 prediction 计时；权重字节有效吞吐 | [profiling](profiling.md)；不能直接等同硬件 PMU |
| ane_pmu_profiler / measure_ane_capacity | 私有 counter、loader 与容量探针 | [profiling](profiling.md)；已有本机推理通过但 PMU 缺失的记录 |
| asitop / mactop / Perfetto | 系统采样、能耗上下文、时间线 | [profiling](profiling.md)；不等同逐 op ANE trace |
| hollance/neural-engine / iPhone ANE training | 历史问题地图 / 移动端训练探索 | [负结果与纠错](negative-results.md)；不照搬旧限制或宣传数字 |

来源身份、阅读范围、未运行项及抓取失败见 [source-audit](source-audit.md)，不是完整仓库安全审计。

## 建议的动手顺序

先复现相同权重/输入的 FP16 原图与布局变体；再做 W8、LUT4、W8A8 消融，同时读 compute plan。之后才研究 runtime weights、GPU 分担、共享 backing 和训练。每一步都保留优化 GPU 分母、全输出校验、compile/load/first/warm、外部转换及失败记录。

本次新增的 [压缩实验](../labs/ane/compression_survey.py)、[手写反向探针](../labs/ane/training_probe.py)需要显式设备选项，不被 CPU `all`、检索或 README 阅读自动触发。[来源抓取器](../labs/ane/research_sources.py)只下载小型文本，不执行上游脚本。标准库验证与复现说明见 [实验报告](experiments-20261016.md)。

没有通用赢家：优先公共 API 与稳定阶段边界；仅在测到完整收益且能承受系统兼容风险时研究私有路径。能效、前台延迟、后台吞吐和最低内存的最优方案也可能不同。
