# Qwen encoder 实验归档

本目录统一保存 2026-10-08 在 M4 Pro / 48 GiB 上进行的 Qwen 编辑 encoder
实施和测量记录。MLX GPU 融合、Private ANE runtime、Native Metal encoder
与编辑首步 FFN 是不同执行范围；记录保留各自构建、精度、缓存和质量门槛，
不把组件结果外推为整请求收益或当前产品资格。

## 记录顺序与证据

| 记录 | 范围与结论 | 机器可读汇总 |
| --- | --- | --- |
| [1–3 图 GPU 融合](2026-10-08-qwen-edit-encoder.md) | 视觉/语言组件省时；实际三图请求中的 encoder 没有明显提速，保留不同误差门槛的结果 | [JSON](../../status/2026-10-08-qwen-edit-encoder.json) |
| [Private ANE runtime 初版](2026-10-08-qwen-ane-encoder.md) | 真实单层 FFN 更快，但完整三图 encoder 变慢且未通过预声明 5% 误差筛选 | [JSON](../../status/2026-10-08-qwen-ane-encoder.json) |
| [Runtime 后续优化](2026-10-08-qwen-encoder-followup.md) | 完整 encoder 省时 3.93%，未达到该轮 5% 性能筛选；单样本与质量范围保留 | [JSON](../../status/2026-10-08-qwen-encoder-followup.json) |
| [分区、预取与校准](2026-10-08-qwen-encoder-scheduling.md) | 准备复用改善固定三图输入；独立 S1 校准仅通过组件，没有整图或自动采用资格 | [JSON](../../status/2026-10-08-qwen-encoder-scheduling.json) |
| [Native Metal 单图 encoder](2026-10-08-qwen-native-encoder.md) | 视觉/语言完整 Native GPU 路径与一组单图编辑；保持显式实验，未授 ANE/W8A8/W2 资格 | [JSON](../../status/2026-10-08-qwen-native-encoder.json) |
| [Native encoder / 无 KV 编辑首步 ANE](2026-10-08-qwen-native-prefill-ane.md) | 共享 FFN、准备复用与 BaseOnly 首步诊断；保留冷启动负结果及预热成本边界 | [JSON](../../status/2026-10-08-qwen-native-prefill-ane.json) |

同一历史日期不代表同一构建、输入或计时窗口。单层 FFN、完整 conditioning、
编辑首步、去噪等待和整个请求需要分别比较；缓存命中和显式预热也不能代替
首次冷请求。原文的失败、预算保护、误差阈值及未完成验证均保持原样。

## 设计和历史产物

设计合同仍在 [Qwen command graph 专题](../../design/qwen-image-gpu-ane-command-graph/README.md)：
18–23 分别对应上述阶段。当前运行入口见 [Qwen 用户指南](../../public/QWEN_IMAGE_21.md)，
其他实施检查点见[状态索引](../../status/README.md)。

六份小型 JSON 汇总保留原路径和原始字节，正文链接直接指向这些文件。
历史请求、日志、回执、张量、PNG 和构建目录可能只存在于原实验机器；
本次迁移没有复制或重新生成这些产物，也没有重跑模型或更改默认。

旧 `docs/status/2026-10-08-qwen-*.md` 中本专题的六个路径保留原章节跳转。
新记录按[实验归档规则](../README.md)存放，避免复制正文。
