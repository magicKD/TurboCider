# TurboCider 状态文档入口

更新时间：2026-09-08

优先阅读 [2026-09-08 当前目标状态](current-status-2026-09-08.md)。它是 GGUF、LLaDA、LoRA 三模式、ConvRot 原生 ANE、真实性能和未完成门禁的最新统一快照。

按原始目标逐项核对请看 [原始目标完成度审计](objective-audit-2026-09-08.md)。

[2026-09-07 当前状态](current-status-2026-09-07.md)保留为合入 GGUF 和 LLaDA 之前后的阶段记录，不再作为最新结论。

专项材料：

- [Z-Image Turbo GGUF](z-image-gguf-2026-09-07.md)：GGUF 常驻 GPU、独立 LoRA、量化选择和未完成的 ANE/基准门禁。
- [LoRA 执行策略](../design/lora-execution-strategies.md)：三种公共策略、模型映射、GGUF 请求级路由和剩余 packed 低秩分支。

- [FLUX / Z-Image 优化验收](optimization-validation-2026-09-07.md)：本轮图像模型的最终 warm 数据和质量指标。
- [Transformer / Core ML / Streaming 技术报告](../design/transformer-heterogeneous-report.md)：异构并行、public/private ANE、启动开销和 vpipe/H3/LTX 低内存对照。
- [Private ANE 实验边界](../../experimental/private-ane/README.md)：研究入口、版本固定和正式构建隔离。
- [实现状态](implementation-status-2026-09-06.md)：合入阶段六个模型的详细能力矩阵和历史过程；当前八个注册模块以 2026-09-08 状态为准。
- [独立运行与外部依赖](independence-and-dependencies-2026-09-06.md)：源码、模型资产、系统框架和可选工具的边界。
- [版本准备度](release-readiness-2026-09-06.md)：可提交范围、禁止进入版本的产物和剩余发布风险。
- [main→dev 合并记录](main-dev-merge-2026-09-06.md)：合并决策和兼容处理，属于历史验收记录。

`2026-09-06` 文件保留原文件名以避免破坏已有链接；其中标注为历史数据的数字不应覆盖 2026-09-07 当前状态报告。
