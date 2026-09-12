# TurboCider 设计文档

当前任务感知路由与实时资源设计：[按任务选择计算路径](task-aware-acceleration.md)。
当前独立部署入口：[项目独立性与完整链路](standalone-project.md)。构建、转换工具链及加速产物均由 TurboCider 管理，原始模型路径可在外部。

最新磁盘管理：[Core ML 源模型、编译与缓存管理](coreml-artifacts-and-storage.md)，包含实际占用检查、safetensors 离线导出、CLI/API 和 App 清理入口。

最新实现：[模型准备、GPU/ANE 管理与性能](model-preparation-and-performance.md)，包含本机自动 GPU/ANE 适配、App 加载/预热、分区编译缓存及纯 GPU compile 实测。

**已实现的 Studio 首版与测试：[实现记录](app-studio-implementation.md)。** 包含原生 FLUX 三操作、图片输入、种子、load/unload 及实际验证；早期设计中的目录/能力以当前实现记录为准。

当前项目目录与旧代码退役：[项目重构](project-restructure.md)。

当前 main→dev 合并与阶段性六模型/App 回归见 [2026-09-06 合并验收](../status/main-dev-merge-2026-09-06.md)。当前状态、真实性能和剩余工作统一见 [2026-09-08 当前状态](../status/current-status-2026-09-08.md) 与 [状态文档入口](../status/README.md)，独立运行边界见 [外部依赖说明](../status/independence-and-dependencies-2026-09-06.md)，提交边界见 [版本准备度](../status/release-readiness-2026-09-06.md)。模型职责的详细说明见 [原生重构状态](rewrite-implementation-status.md)、[FLUX 同条件性能对比](flux-performance-comparison.md) 和 [H3/LTX 验收契约](video-model-acceptance.md)。以下长期设计不等于已交付能力。

当前各模型在 M4 Max/M4 Pro 及其他 Apple Silicon 上的 GPU/ANE fork-join 方案、自动门禁和实测口径见 [GPU/ANE 并行化方案](parallel-acceleration.md) 与 [并行化 SVG 图](parallel-acceleration.svg)。Transformer 的 CPU/GPU/ANE 轴、public/private ANE 对照和 vpipe/H3/LTX 结论见 [Transformer 异构并行技术报告](transformer-heterogeneous-report.md)；Core ML 启动生命周期见 [Core ML / ANE 启动开销](coreml-ane-startup.md)，量化和低内存对照见 [量化与 Streaming 对照](quantized-streaming-vpipe-comparison.md)。

跨后端不要求逐像素一致时的 RGB 回归、视频运动和公开 Vision feature-print 分层判定见 [感知质量诊断](vision-quality-diagnostics.md)。

## App 产品、界面与实现方案

2026-09-05 的 App 设计提案，基于现有 SwiftUI 界面、兼容控制平面与新原生纵切的能力差异。以下为待实施设计，不代表三模型原生功能或实时资源遥测已完成。

1. [App 整体产品与交互设计](app-product-design.md)：Studio、图像/编辑/视频流程、素材/任务/模型中心、首发范围及状态设计。
2. [视觉系统与组件规范](app-visual-system.md)：现代原生风格、明暗配色、窗口布局、组件、动效与可访问性。
3. [App 实现、契约与验收方案](app-implementation-plan.md)：SwiftUI 分层、运行时迁移、真实速度/ETA、模型加载与分阶段验收。

4. [首版多模态输入、种子与高级参数](app-inputs-and-parameters.md)：图片插入/拖放/粘贴、角色槽、有序多图、默认 42 与随机种子、单输出首版及按代码开放参数。

最新范围：每次一张图或一个视频，App 一次一个活动生成；多图输入不等于批量输出。原生代码的后续推进与现有文档差异见第 4 项的代码核对。

## 原生引擎与运行时设计

**后续整体重构主设计：[统一多模态框架](unified-multimodal-framework.md)。**

该文重新梳理旧系统、新 native 纵切、多模态输入、ModelModule、设备配置/分区策略及原生服务。以下文档保留为此前架构背景和实际实现证据；不能把旧纵切状态当成最终架构。

**模块职责、完整 App 与性能主线：[模块与高性能运行时详设](runtime-modules-and-performance.md)。**

该文细化分区构建、编译与会话缓存、预热调度、统一内存 offloading、复用边界和功能清单。默认行为为原生 GPU，混合路线由用户配置启用。

背景与验证文档：

1. [总体架构](native-engine-rearchitecture.md)：长期目标、Metal/ANE 边界、模块划分与迁移原则。
2. [本机实施设计](native-implementation-plan.md)：代码结构、共同抽象、依赖、真实 FLUX 路径，以及从纵切到完整 runtime 的迁移顺序。
3. [本机验证记录](native-validation-report.md)：实际完成项、逐张量证据、性能、App 与独立包测试，明确未完成范围。
4. [H3/LTX 迁移与验收](video-model-acceptance.md)：源文件迁移归属、无模型静态检查、权重到位后的数学和媒体退出条件。
5. [原生代码使用说明](../USAGE.md)：构建、打包、CLI/SDK 与离线验收复现命令。

H3 在 Apple Silicon 上迁移到 FastH3 C++/MLX、固定四步 affine INT6/g64，并对齐 FastVideo 性能/质量的具体实施合同见 [H3 C++/MLX INT6 加速方案与验收计划](h3-mlx-int6-fastvideo-parity-plan.md)。该文档包含当前原型边界、分阶段工作包、checkpoint/量化合同、同条件 ABBA benchmark 和发布门禁；在真实模型验收完成前不代表已交付能力。

[架构图](native-engine-architecture.svg) 表达最终方向，并非每个方框都已经实现。当前 FLUX、H3、FastMetal 和 LTX video-only 已有执行入口；LTX I2V/音频、完整 GPU+ANE 门禁、通用执行器与自有 allocator 仍待完成。

`validation/` 存放小型 JSON 证据和依赖身份。原始大张量、图像及过程日志保留在本机 `outputs/native-validation/`，不提交模型或大文件。

- [原生 C++ 引擎边界与重构验收](native-cpp-engine.md)：当前源码划分、类型化接口、驻留策略与性能验证。
