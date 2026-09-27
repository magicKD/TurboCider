# TurboCider 状态文档入口

更新时间：2026-09-27

Qwen-Image-2.1 的当前入口是 [公开运行说明](../public/QWEN_IMAGE_21.md)；
逐项实验与可用开关见 [Qwen21 加速路径索引](qwen21-acceleration-routes-2026-09-27.md)。
索引明确区分 512px 缩图与 1024px 原尺寸参考、首请求与 resident
prefix-KV hit，以及保留和已撤回的候选。下方 9 月 25 日结论是历史快照，
不能覆盖 9 月 27 日的实测结果。

新增 [Qwen21 base DBCache 20/40 步实验](qwen21-dbcache-2026-09-27.md)：
记录 decode-only 请求级残差缓存的门禁、跳过计数、GPU／GPU+ANE
同条件时间和 PNG 观感；额外记录 40 步两张原尺寸参考图的四路
GPU／GPU+ANE 严格输入对照，以及三图编辑中连续跳过上限 2/4/8
的质量与速度取舍。此路由仍是显式诊断；已补 768×512／512×768
矩形画布的局部测量，尚未覆盖六步 LoRA。

[Qwen21 768×512／512×768 复用 512² W8A8 图](qwen21-rectangular-w8a8-2026-09-27.md)
记录固定 1024-row 图的 decode 序列切片、两种方向单次 warm 配对和
尚待完成的多种子门禁；没有将诊断路线设为默认。

[Viggle runtime LoRA + base ANE 复用诊断](qwen21-viggle-base-ane-reuse-2026-09-27.md)
记录 GPU 后缀低秩投影、未覆盖的 ANE 前缀 LoRA 非线性更新，及
六步对照；融合 Q/K＋FP16 低秩后文生图两次配对约 1.22×，
一图编辑收益不稳定，原尺寸双／三图单次配对略慢于优化 GPU。
额外 512px 缩图的双／三图各两次同输入 warm 配对，hybrid 对 GPU
中位约 1.139×／1.070×，主体与贴纸可辨，但只覆盖一类编辑和一个
种子，且未包含 ANE 前缀 LoRA 校正；另一个小贴纸贴壶的
编辑任务虽然肉眼合格，hybrid 比 GPU 略慢；仍不作为默认推荐。

Qwen-Image-2.1 的最新 W8A8/ANE 数值诊断见
[512² 单层诊断](qwen21-w8a8-ane-diagnostics-2026-09-25.md)；当前 W8A8
尚未通过质量和端到端速度门槛，生产运行仍保持既有路径。

最新 encoder-prefill 与 GPU/ANE 边界请看
[2026-09-15 Encoder prefill 与 GPU/ANE 可选路径状态](encoder-prefill-optional-2026-09-15.md)。
它记录默认 GPU、encoder/denoiser manifest 分离、实验 probe 的可选构建开关、
FLUX/Z-Image/H3/LTX 的资格结果、未通过的 LTX topology，以及 H3 exact-256
收益在本轮复跑中未稳定复现的限制。

跨 encoder、DiT、VAE 和 Core ML 生命周期的统一设计结论见
[ANE 加速现状与分阶段决策](../design/ane-acceleration-status-2026-09-15.md)。

本轮重构与提交范围以 [原生框架重构总结](native-refactor-summary-2026-09-09.md) 为准。
其中区分已完成的结构整理、实际运行结果和仍未证明的发布/性能范围。

优先阅读 [native-only GGUF 发行边界](native-gguf-boundary-2026-09-09.md)，它取代以下历史报告中的 sd.cpp 发行与 GGUF streaming 描述。

历史验证快照：[2026-09-09 当前目标状态](current-status-2026-09-09.md)。它是 GGUF CPU-staged streaming、近似质量门禁、LLaDA、LoRA 三模式、ConvRot 原生 ANE、真实性能和未完成门禁的最新统一快照。

按原始目标逐项核对请看 [原始目标完成度审计](objective-audit-2026-09-09.md)。

[2026-09-07 当前状态](current-status-2026-09-07.md)保留为合入 GGUF 和 LLaDA 之前后的阶段记录，不再作为最新结论。

专项材料：

- [Z-Image Turbo GGUF](z-image-gguf-2026-09-07.md)：GGUF 常驻 GPU、CPU-staged 低内存 streaming、独立 LoRA、量化选择和未完成的 ANE/基准门禁。
- [LoRA 执行策略](../design/lora-execution-strategies.md)：三种公共策略、模型映射、GGUF 请求级路由和剩余 packed 低秩分支。

- [Z-Image 可变长度 ANE](z-image-flexible-ane-2026-09-08.md)：512-token 容量、固定/枚举/范围输入实验、实际 App 验证与编译缓存复用。
- [App 控制与缓存](app-controls-and-cache-2026-09-07.md)：图片/任务删除、LoRA 开关与强度、GPU/ANE 选择。
- [FLUX / Z-Image 优化验收](optimization-validation-2026-09-07.md)：本轮图像模型的最终 warm 数据和质量指标。
- [Transformer / Core ML / Streaming 技术报告](../design/transformer-heterogeneous-report.md)：异构并行、public/private ANE、启动开销和 vpipe/H3/LTX 低内存对照。
- [Private ANE 实验边界](../../experimental/private-ane/README.md)：研究入口、版本固定和正式构建隔离。
- [实现状态](implementation-status-2026-09-06.md)：合入阶段六个模型的详细能力矩阵和历史过程；当前八个注册模块以 2026-09-08 状态为准。
- [独立运行与外部依赖](independence-and-dependencies-2026-09-06.md)：源码、模型资产、系统框架和可选工具的边界。
- [版本准备度](release-readiness-2026-09-06.md)：可提交范围、禁止进入版本的产物和剩余发布风险。
- [main→dev 合并记录](main-dev-merge-2026-09-06.md)：合并决策和兼容处理，属于历史验收记录。

`2026-09-06` 和 `2026-09-08` 文件保留原文件名以避免破坏已有链接；其中标注为历史数据的数字不应覆盖 2026-09-09 当前状态报告。2026-09-08 的 GGUF streaming 数字来自旧 disk-backend 配置，仅作历史记录。
