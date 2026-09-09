# TurboCider 状态文档入口

更新时间：2026-09-08

优先阅读 [开源产品整理与验收](open-source-product-2026-09-08.md)，了解当前 App、CLI、模型库、下载、API、缓存和交付验证范围。
[2026-09-07 状态快照](current-status-2026-09-07.md) 保留此前的模型能力、真实性能、LoRA、GPU/ANE 门禁和独立运行边界；后续变更以新的专项记录为准。

专项材料：

- [Z-Image 可变长度 ANE](z-image-flexible-ane-2026-09-08.md)：512-token 容量、固定/枚举/范围输入实验、实际 App 验证与编译缓存复用。
- [App 控制与缓存](app-controls-and-cache-2026-09-07.md)：图片/任务删除、LoRA 开关与强度、GPU/ANE 选择。
- [FLUX / Z-Image 优化验收](optimization-validation-2026-09-07.md)：本轮图像模型的最终 warm 数据和质量指标。
- [实现状态](implementation-status-2026-09-06.md)：六个模型的详细能力矩阵和历史过程。
- [独立运行与外部依赖](independence-and-dependencies-2026-09-06.md)：源码、模型资产、系统框架和可选工具的边界。
- [版本准备度](release-readiness-2026-09-06.md)：可提交范围、禁止进入版本的产物和剩余发布风险。
- [main→dev 合并记录](main-dev-merge-2026-09-06.md)：合并决策和兼容处理，属于历史验收记录。

`2026-09-06` 文件保留原文件名以避免破坏已有链接；其中标注为历史数据的数字不应覆盖 2026-09-07 当前状态报告。
