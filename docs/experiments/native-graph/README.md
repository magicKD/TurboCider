# Native Metal / W1 实验归档

本目录集中保存 2026-10-08 在 Apple M4 Pro 上进行的独立 Native Metal
后端、private ANE W1 接线与尺度缓存实验。实现、合成数据、真实单层回放
和完整请求是不同证据范围；每份记录继续绑定原构建、设备与输入。

## 记录顺序与证据

| 记录 | 范围与结论 | 机器可读汇总 |
| --- | --- | --- |
| [Block 与 W0 交接](2026-10-08-native-block-and-ane-handoff.md) | 合成 block 与小型 GPU→ANE→GPU 正确性；没有整图或并行加速结论 | [JSON](../../status/2026-10-08-native-block-and-ane-handoff.json) |
| [完整去噪图与真实 block](2026-10-08-native-denoiser-and-real-block.md) | 小几何对照通过；真实 layer 0 首步／稳态组件记录，不授整图性能资格 | [JSON](../../status/2026-10-08-native-denoiser-and-real-block.json) |
| [Session 与首次 GPU 整图](2026-10-08-native-session-and-e2e.md) | 单组完整 Native 请求比 MLX 慢约 1.68%，保留准备成本及质量边界 | [JSON](../../status/2026-10-08-native-session-and-e2e.json) |
| [W1 并行 FFN 组件](2026-10-08-native-w1-parallel-ffn.md) | 合成小尺寸数值与生命周期检查；固定顺序、单样本，不授真实性能资格 | [JSON](../../status/2026-10-08-native-w1-parallel-ffn.json) |
| [W1S tiny 去噪与真实 FFN](2026-10-08-native-w1-session-components.md) | step 1 与 held-out step 19 分开记录；后者没有净收益，未追加整图 | [JSON](../../status/2026-10-08-native-w1-session-components.json) |
| [Register W1 Base 整图](2026-10-08-native-register-w1-base-e2e.md) | 同构建单组请求省时 9.97%；不是稳定统计或 generic/register 单项消融 | [JSON](../../status/2026-10-08-native-register-w1-base-e2e.json) |
| [请求内尺度缓存实现](2026-10-08-native-request-scale-cache.md) | 合成数据的精确复用与生命周期验证；预计整图计数保持为预计值 | [JSON](../../status/2026-10-08-native-request-scale-cache.json) |
| [真实尺度缓存 OFF／ON](2026-10-08-native-scale-cache-real-screen.md) | 两次正式记录随顺序反转，没有稳定净收益；保留默认关闭与后续 CPU 工具改进边界 | [JSON](../../status/2026-10-08-native-scale-cache-real-screen.json) |

不同构建与不同输入的计时不合并为排行榜。组件收益、tiny 输出误差、
实际完整请求及图像诊断分别保留，历史 W1S 不等于 W2 设备图、硬件
INT8／ANE placement／物理 overlap 或当前 App 的发布资格。

## 设计、后续专题与历史产物

[Command graph 设计目录](../../design/qwen-image-gpu-ane-command-graph/README.md)
继续保存架构和验收合同；[原生去噪资源审查](../../status/2026-10-08-native-denoiser-review.md)
继续作为独立检查点。后续专题见 [Native LoRA](../native-lora/README.md)
和 [Qwen encoder](../qwen-encoder/README.md)。当前产品入口见[用户指南](../../public/README.md)。

八份 JSON 保留原路径和原始字节。原始日志、图片、张量、构建产物及
清理回执继续指向原文的忽略目录，可能只在原实验机器可用；此次归档
没有复制或重建它们，没有重新运行模型或改写失败、未完成与待验证结论。

旧 `docs/status/2026-10-08-native-*.md` 中本专题的八个原路径保留章节跳转，
完整正文只维护在此处。新记录遵循[实验归档规则](../README.md)。
