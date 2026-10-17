# GPU + ANE 协同推理：优化方法与实现参考

<!-- knowledge
id: technique-gpu-ane-readme
type: navigation
-->

本目录按瓶颈和方法组织，回答：什么时候用、为什么有效、怎样实现、源码在哪里、如何验证。整理日期：2026-10-17。源码阅读、CPU见证、组件运行和完整模型结果分别标注；局部收益不等于端到端收益。

## 从这里开始

- 先读[14种优化技巧总览](optimization-playbook.md)，再按下表选择方法。
- 查具体实现读[项目案例](case-studies/README.md)；查全部主题读[主题索引](report-index.md)。不必了解研究批次。
- 学习代码读[小型实现参考](examples/README.md)；理解取舍读[历史观察与反例](experiments.md)、[来源阅读指南](sources-and-coverage.md)。

## 按瓶颈选择方法

| 想改善什么 | 机制与实现方法 | 最值得对照的案例 |
| --- | --- | --- |
| 小batch/低比特矩阵乘慢 | [按shape选核与解码复用](gpu-gemm-quantization.md) | [GGML few-row](case-studies/ggml-fewrow-mma-and-reuse.md) |
| attention中间流量大 | [分块attention与KV布局](gpu-attention-runtime.md) | [Split-Einsum](case-studies/coreml-split-einsum-export-and-runtime.md) |
| 小算子多、CPU提交高 | [完整子图融合与重放](fusion-and-replay.md) | [完整paired FFN](case-studies/tinygrad-strong-baselines-and-paired-ffn.md) |
| 希望同一FFN跨设备分担 | [合法切分](partition-and-scheduling.md) → [负载平衡](load-balancing.md) | [vpipe](case-studies/vpipe-controller-and-failure.md)、[ccv](case-studies/ccv-hybrid-cost-and-failure-boundaries.md) |
| 交接复制多、偶发旧输出 | [共享缓冲区、布局与同步](memory-and-handoff.md) | [Metal4消费者](case-studies/metal4-consumer-layout-and-handoff.md) |
| 编译/回退/ANE小图开销高 | [封闭子图与图表达](ane-graphs-and-quantization.md) | [ORT分区](case-studies/onnxruntime-coreml-partition-cache.md) |
| 权重能放下却被I/O拖慢 | [驻留、预取与专家bank](residency-and-prefetch.md) | [完整MoE消费者](case-studies/anemll-original-bank-gpu-consumer.md) |
| 单阶段快但完整请求不快 | [阶段划分与流水](stage-pipelines.md) | [Whisper](case-studies/whisper-coreml-metal-stages-and-real-asr.md)、[FastVLM](case-studies/fastvlm-coreml-vision-and-mlx-handoff.md) |
| 优化后数值或恢复不稳定 | [数值验收、测量与回退](validation-and-fallback.md) | [scale-floor](case-studies/ccv-dense-precision-and-scale-floor.md)、[融合精度](case-studies/metalpetal-render-graphs-and-pixelbuffer-handoff.md) |

## 当前结论

1. **先做强GPU分母，再研究ANE。** 根据小M/decode与大M/prefill选核，减少解码、score矩阵、中间量和提交；不能用低效分阶段GPU证明异构胜出。[GEMM](gpu-gemm-quantization.md) · [attention](gpu-attention-runtime.md) · [强基线实测](case-studies/tinygrad-strong-baselines-and-paired-ffn.md)
2. **优先尝试封闭子图与合法切分。** 完整FFN留在一侧；row、paired-F、独立branch、stage流水分别建模，顺序依赖不能假并行。[切分/成本](partition-and-scheduling.md) · [vpipe](case-studies/vpipe-runtime-weights.md) · [Splash](case-studies/splash-w8a8-prefill-and-layout.md)
3. **交接是ABI，不只是共享内存。** 实际stride/dtype、backing、完成事件、owner与generation一起验证；scheduled不等于completed。[共享合同](memory-and-handoff.md) · [真实Metal4消费者](case-studies/metal4-consumer-layout-and-handoff.md)
4. **数值语义和失败恢复是优化资格的一部分。** finite、同文本、矩阵可合并都不够；必须覆盖中间舍入/裁剪、bias/LoRA、递归state和完整输出。[数值/scale反例](case-studies/ccv-dense-precision-and-scale-floor.md) · [融合反例](case-studies/metalpetal-render-graphs-and-pixelbuffer-handoff.md) · [状态公式](case-studies/ane-lm-programs-state-and-numerics.md)
5. **硬件、框架、执行轨迹分开。** M5 GPU内NAX不是独立ANE；CoreML允许策略/compute plan不证明物理placement或overlap。本机M4不能验证M5快路径。[硬件/API](hardware-and-tensorops.md) · [ANE图与量化](ane-graphs-and-quantization.md)
6. **目前可给方法，不给通用赢家。** share、tile、量化和cache需随SoC/SDK、形状、模型/adapter、驻留及并发重验。现有证据尚不能宣称所有模型GPU+ANE都更快、全链物理零拷贝或取消/drain已合格。[证据边界](sources-and-coverage.md)

## 按目标选阅读路线

| 目标 | 机制入门 | 最值得对照的实现/案例 |
| --- | --- | --- |
| 小batch/量化矩阵更快 | [GEMM/GEMV与packing](gpu-gemm-quantization.md) | [GGML few-row](case-studies/ggml-fewrow-mma-and-reuse.md)、[MLX mid-M/NAX](case-studies/mlx-nax-mid-m-and-prefetch.md) |
| 降低attention中间流量 | [分块attention](gpu-attention-runtime.md) | [vLLM分页KV](case-studies/vllm-paged-nax.md)、[公共TensorOps层次](hardware-and-tensorops.md) |
| 减少launch与host成本 | [融合/提交](gpu-attention-runtime.md) | [tinygrad完整FFN](case-studies/tinygrad-strong-baselines-and-paired-ffn.md)、[MLX compile](case-studies/mlx-compile-and-cross-runtime-ffn.md) |
| GPU+ANE同block分担 | [切分与关键路径](partition-and-scheduling.md) | [vpipe控制器](case-studies/vpipe-controller-and-failure.md)、[ccv成本](case-studies/ccv-hybrid-cost-and-failure-boundaries.md)、[Splash配对布局](case-studies/splash-w8a8-prefill-and-layout.md) |
| 公共ANE图表达/委托 | [ANE图与量化](ane-graphs-and-quantization.md) | [CoreML实际小图](case-studies/coreml-split-einsum-export-and-runtime.md)、[ORT](case-studies/onnxruntime-coreml-partition-cache.md)、[ExecuTorch](case-studies/executorch-coreml-delegation.md) |
| 权重/专家驻留与I/O | [内存与交接](memory-and-handoff.md) | [SD.cpp分段](case-studies/sdcpp-metal-residency-and-segments.md)、[MoE GPU消费者](case-studies/anemll-original-bank-gpu-consumer.md) |
| 音频/视觉多stage | [项目分工地图](hybrid-projects-and-stages.md) | [Whisper真实ASR](case-studies/whisper-coreml-metal-stages-and-real-asr.md)、[完整encoder交接](case-studies/whisper-generation-and-complete-encoder-handoff.md)、[FastVLM](case-studies/fastvlm-coreml-vision-and-mlx-handoff.md) |
| 长时状态/失败/取消 | [owner与generation](memory-and-handoff.md) | [ggml-ane失败链](case-studies/ggml-ane-backend-cache-and-failure.md)、[ANE-LM状态](case-studies/ane-lm-programs-state-and-numerics.md)、[MetalPetal task](case-studies/metalpetal-render-graphs-and-pixelbuffer-handoff.md) |

## 怎样读案例结论

| 标记 | 能说明什么 | 不自动说明什么 |
| --- | --- | --- |
| 源码/官方正文 | 固定版本的实现、API合同或作者说法 | 本机已执行、实际快路径或收益 |
| 独立CPU数学/元数据 | 代数、索引、格式或协议边界 | 原框架、硬件或真实模型故障 |
| 原CPU片段 | 所选原函数/语句的实际行为 | 完整引擎/模型已运行 |
| GPU/CoreML组件 | 指定图、输入和完成边界的数值/计时 | 通用模型质量、物理ANE/overlap |
| 真实模型请求 | 指定checkpoint/输入的完整消费者观察 | 全验证集、所有模型、产品可启用 |

先读案例的范围说明，再沿固定源码链接理解实现；不提供 source/evidence/validation 实验包。典型正负对照见[技巧总览的反例表](optimization-playbook.md#四优先复查的反例)；PR按[优化PR读本](optimization-prs.md)追踪，治理状态与性能资格分开。

## 最小学习路径与采用边界

选择一个真实FFN或encoder，依次做：**强GPU baseline → 一项合法变换 → 完整数值 → 真实交接 → 独立确认 → 完整请求**。模板在[技巧总览](optimization-playbook.md#五把方法变成一个可验收的小任务)。

开展自己的实验时，应记录准备/首次/稳态/回读与失败；没有硬件 counter 就标未知。这里仅保存设计思路、小型代码和历史案例，不附实验启动器。

## 内容与维护

本层放方法和选型，[case-studies](case-studies/README.md)放实现分析与历史反例，[sources.md](sources.md)放固定源码入口，[examples](examples/README.md)放阅读练习。没有 JSON、实验归档或专用验证器。

通用 GPU 原理链接[Metal](../metal/README.md)，ANE 图与部署链接[ANE](../ane/README.md)，基础代数链接[gpu-ane](../gpu-ane/partitioning.md)。历史案例不等于当前设备或产品验证；新增笔记按[维护规则](../MAINTENANCE.md)提炼认识。
