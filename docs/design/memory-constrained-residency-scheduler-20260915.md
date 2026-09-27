# TurboCider Streaming / 内存受限模式设计入口

2026-09-16：本设计已按主题重组，请从 [streaming/README.md](streaming/README.md) 阅读当前方案。

核心修订：采用 **layout-first** 框架。用户直接指定 block 分组、slot 数、常驻前缀与预取距离；框架统一描述资源、编译布局、管理 I/O/GPU 生命周期。内存上限是可选的独立 admission/guard，不再是选择执行布局的唯一入口。

| 主题 | 当前设计 |
|---|---|
| 架构与决策 | [01-framework.md](streaming/01-framework.md) |
| 参数、配置、预设、兼容规则 | [02-configuration.md](streaming/02-configuration.md) |
| Slot、lease、fence 与调度协议 | [03-runtime-protocol.md](streaming/03-runtime-protocol.md) |
| H3/LTX/Flux/Z-Image 接入 | [04-model-adapters.md](streaming/04-model-adapters.md) |
| 内存估算、上限和执行授权 | [05-memory-contract.md](streaming/05-memory-contract.md) |
| 逐文件改造、PR 顺序与回滚 | [06-migration.md](streaming/06-migration.md) |
| 工具链、仿真、性能与验收 | [07-tooling-and-validation.md](streaming/07-tooling-and-validation.md) |
| 当前实现状态与历史映射 | [08-status-and-history.md](streaming/08-status-and-history.md) |

旧稿 25,568 行已[归档](streaming/archive/memory-constrained-residency-scheduler-20260915.md)，未删除历史设计或实施记录；提交前仅整理了四处开发机绝对路径。旧的章节/锚点请在归档中查找。

实现已有配置解析、通用编译/执行底座和 LTX 内部实验 adapter，但正常 session 接线及生产资格仍未完成。
当前提交范围和待办见 [审阅与交接](streaming/21-review-and-handoff.md)，历史验证见
[实施进度](streaming/13-implementation-progress.md)。旧稿中互相冲突的草案，以新目录中的主题文档为准。
