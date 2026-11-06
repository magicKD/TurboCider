# TurboCider 设计文档

这里保存架构、接口和待验证方案。设计文档中的目标、伪代码和历史实验结果不等于
当前版本已交付的能力；运行方式以[用户指南](../public/README.md)为准，
实验记录从[实验归档](../experiments/README.md)进入，实施历史见[状态索引](../status/README.md)。

## 从主题开始

| 主题 | 入口 |
| --- | --- |
| 原生引擎职责、类型接口与重构验收 | [C++ 引擎边界](native-cpp-engine.md) |
| 任务路由与 GPU/ANE 分工 | [任务感知加速](task-aware-acceleration.md) · [并行加速](parallel-acceleration.md) |
| 受限内存、权重驻留与调度 | [驻留调度器](memory-constrained-residency-scheduler-20260915.md) · [Streaming 设计目录](streaming/README.md) |
| GGUF、量化执行与有限解码 | [量化执行目录](quantized-execution/README.md) · [量化与 streaming 对照](quantized-streaming-vpipe-comparison.md) |
| Qwen Native Metal / GPU+ANE command graph | [专项设计目录](qwen-image-gpu-ane-command-graph/README.md) |
| Apple 芯片与 kernel 优化 | [Mac kernel 优化](mac-kernel-opt.md) · [M5 ANE 适配](m5-ane-adaptation.md) |
| H3 C++/MLX INT6 | [加速方案与验收计划](h3-mlx-int6-fastvideo-parity-plan.md) |
| LTX 2.5 与 VDN H3 | [实施与验收方案](ltx25-vdn-h3-ane-implementation-plan.md) · [ltx-mac 对齐方案](ltx25-ltx-mac-acceleration-parity-plan.md) |
| LTX Video VAE 生命周期 | [生命周期与职责整理](ltx25-vae-lifecycle-and-code-cleanup-20260913.md) |

## Streaming 实施与验收

按[专题索引](streaming/README.md)选择具体阶段，不把未绑定的模板作为发布证据。
常用入口：

- [Public Adapter 代码规格](streaming/36-public-adapter-code-implementation-spec.md)
- [校准、性能与验收规格](streaming/37-public-streaming-calibration-performance-acceptance.md)
- [Framework 代码合同与工作台](streaming/38-framework-code-contracts-and-implementation-workbench.md)
- [验证与发布工作簿](streaming/39-validation-benchmark-and-release-workbook.md)

## Qwen command graph 与 encoder

[专题索引](qwen-image-gpu-ane-command-graph/README.md)保留设计顺序和实验边界。
以下文档分别讨论后端、LoRA 和 encoder，不能合并理解成已通过完整产品验收：

- [后端定位与 MLX 边界](qwen-image-gpu-ane-command-graph/10-native-metal-graph-backend.md)
- [完整 Native 后端规格](qwen-image-gpu-ane-command-graph/16-splash-native-backend-spec.md)
- [LoRA 接入](qwen-image-gpu-ane-command-graph/17-native-lora-integration.md)
- [Edit encoder GPU 融合](qwen-image-gpu-ane-command-graph/18-edit-encoder-acceleration.md)
- [Edit encoder ANE runtime](qwen-image-gpu-ane-command-graph/19-edit-encoder-ane-runtime.md)
- [Encoder runtime 优化](qwen-image-gpu-ane-command-graph/20-encoder-runtime-optimization.md)
- [Encoder 调度与校准](qwen-image-gpu-ane-command-graph/21-encoder-scheduling-and-calibration.md)
- [Native Metal 编辑 encoder](qwen-image-gpu-ane-command-graph/22-native-edit-encoder.md)
- [Native encoder 与编辑首步 ANE](qwen-image-gpu-ane-command-graph/23-native-encoder-prefill-ane.md)

对应的[Base E2E](../status/2026-10-08-native-register-w1-base-e2e.md)、
[LoRA 基础实现](../status/2026-10-08-native-lora-foundation.md)、
[SGMMA 组件筛选](../status/2026-10-08-native-lora-sgmma-screen.md)和
[整图配对](../status/2026-10-08-native-lora-sgmma-e2e.md)保留其各自结论。
Encoder 的[GPU 融合](../status/2026-10-08-qwen-edit-encoder.md)、
[ANE 初版](../status/2026-10-08-qwen-ane-encoder.md)、
[后续优化](../status/2026-10-08-qwen-encoder-followup.md)、
[调度校准](../status/2026-10-08-qwen-encoder-scheduling.md)和
[Native 实验](../status/2026-10-08-qwen-native-encoder.md)是独立历史记录。

## SeedVR2 专项设计

- [Block 权重驻留](seedvr2-block-weight-residency.md)
- [VAE 优化总览](seedvr2-vae-optimization.md)与[空间调度](seedvr2-vae-spatial-scheduler.md)
- [Halo 对齐分块](seedvr2-vae-halo-aligned-tiles.md)与[norm packing / 资源建议](seedvr2-vae-norm-pack-and-resource-advice.md)
- [ANE 可行性](seedvr2-vae-ane-feasibility.md)、[超分 causal convolution](seedvr2-vae-upscale-causal-conv.md)与[virtual causal convolution](seedvr2-vae-virtual-causal-conv.md)
- [阶段性能与超分历史](seedvr2-vae-phase-and-upscale-history.md)

## 历史实验与证据

- [LTX Stage-2 sparse 结果](../experiments/ltx-stage2-sparse/results.md)与[研究账本](../experiments/ltx-stage2-sparse/research.md)：保留速度与质量未同时达标的结论。
- [M4 Max reproduction](../../experimental/transformer/notes/M4_MAX_REPRODUCTION.md)：GPU/ANE、LTX Stage-2 与 VAE 生命周期的独立复验记录。
- [Transformer 异构报告](transformer-heterogeneous-report.md)与[2026-09-15 ANE 分阶段决策](ane-acceleration-status-2026-09-15.md)：历史设备、版本和计时边界不能外推到当前产品。
- [验证摘要索引](validation/README.md)与[VDN H3/MLX GPU 验证](validation/vdn-h3-mlx-gpu-validation-2026-09-12.json)：小型证据与依赖身份保留原路径。

新实验叙述及小型证据放入 `docs/experiments/<topic>/`。原始张量、生成媒体和过程日志
保留在忽略的 `outputs/` 中。已有 `validation/` 数据可能被工具或测试读取，迁移时必须
一并更新消费者；不要为了索引整理删除历史证据或复制出另一份结果。
