# Native LoRA 实验归档

本目录统一保存 2026-10-08 在 M4 Pro / 48 GiB 上进行的独立 Native Metal
后端 LoRA 实施与实验记录。不同构建、生成小张量、真实单层 FFN 和完整
请求分别保留身份与结论；任何一项通过都不自动授予其他阶段或当前代码资格。

## 记录顺序与证据

| 记录 | 范围与结论 | 机器可读汇总 |
| --- | --- | --- |
| [GPU 组件接入](2026-10-08-native-lora-gpu-components.md) | 生成小张量的算子、来源与生命周期验证；当时尚无真实模型请求或性能结论 | [JSON](../../status/2026-10-08-native-lora-gpu-components.json) |
| [GPU 六步整图](2026-10-08-native-lora-gpu-e2e.md) | 窄域 Viggle r128 请求成功，单组完整请求比同构建 MLX 慢 21.48% | [JSON](../../status/2026-10-08-native-lora-gpu-e2e.json) |
| [FP32 SGMMA 组件筛选](2026-10-08-native-lora-sgmma-screen.md) | 单一真实输入的完整 FFN 两顺序省时 7.26% / 3.59%；独立修正算子有顺序反转 | [JSON](../../status/2026-10-08-native-lora-sgmma-screen.json) |
| [SGMMA 六步整图](2026-10-08-native-lora-sgmma-e2e.md) | 同构建整请求仍比 MLX 慢 14.44%，未实现整图净加速 | [JSON](../../status/2026-10-08-native-lora-sgmma-e2e.json) |
| [Paired base gate/up](2026-10-08-native-lora-paired-base-screen.md) | 输出逐位相同，但两顺序没有净收益；保留原 `NoGain` 结论 | [JSON](../../status/2026-10-08-native-lora-paired-base-screen.json) |
| [Hybrid 基础接口](2026-10-08-native-lora-foundation.md) | ABI / MIL 和共享 low 验证通过，真实暖机对照因预算保护中止；尚未接通并行请求 | [JSON](../../status/2026-10-08-native-lora-foundation.json) |

GPU 与 SGMMA 的旧整图数字来自不同构建，不能相减后归因于新 kernel。
原记录中的单输入、单组配对、没有同构建 scalar 整图、预算保护和未完成
GPU/ANE、W2、App 检查均保留。原文所述默认与显式实验准入对应历史版本。

## 设计和历史产物

[独立 Native 后端规格](../../design/qwen-image-gpu-ane-command-graph/16-splash-native-backend-spec.md)
与 [LoRA 集成方案](../../design/qwen-image-gpu-ane-command-graph/17-native-lora-integration.md)
继续保存在设计目录。当前产品入口见[用户指南](../../public/README.md)，
其他实施检查点见[状态索引](../../status/README.md)。

六份小型 JSON 汇总保留原路径和原始字节，正文链接直接指向这些文件。
原始日志、回执、张量、PNG 及构建产物仍指向原文的忽略目录；它们可能仅
存在于原实验机器，本次归档未复制、重建或重新测量这些产物。

旧 `docs/status/2026-10-08-native-lora-*.md` 保留原章节的跳转入口。
新记录按[实验归档规则](../README.md)存放，避免再复制一份正文。
