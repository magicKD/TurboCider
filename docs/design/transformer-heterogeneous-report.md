# Transformer CPU / GPU / ANE 异构并行技术报告

这篇历史报告已统一归档到
[Transformer 实验目录：2026-09-08 技术报告](../experiments/transformer/2026-09-08-adoption-report.md)。

保留此入口以兼容已有文档和外部链接。归档保留原有结论、数值和来源路径，
并标明当前仓库缺失的历史证据。当前产品路由与支持范围请看[加速状态](../status/acceleration.md)。

Original section headings are retained below for existing deep links; the full
record and its evidence remain in the archive.

## 结论

[Read the archived section](../experiments/transformer/2026-09-08-adoption-report.md#结论).

## 1. 统一计时和正确性口径

[Read the archived section](../experiments/transformer/2026-09-08-adoption-report.md#1-统一计时和正确性口径).

## 2. 推荐的 FFN 分割

[Read the archived section](../experiments/transformer/2026-09-08-adoption-report.md#2-推荐的-ffn-分割).

### 2.1 fixed-shape micro 数据

[Read the archived section](../experiments/transformer/2026-09-08-adoption-report.md#21-fixed-shape-micro-数据).

## 3. 不同 Transformer 拓扑的并行窗口

[Read the archived section](../experiments/transformer/2026-09-08-adoption-report.md#3-不同-transformer-拓扑的并行窗口).

### 3.1 FLUX / Z-Image / LLaDA

[Read the archived section](../experiments/transformer/2026-09-08-adoption-report.md#31-flux--z-image--llada).

### 3.2 H3 / sequential pre-norm block

[Read the archived section](../experiments/transformer/2026-09-08-adoption-report.md#32-h3--sequential-pre-norm-block).

### 3.3 LTX video/audio 双流

[Read the archived section](../experiments/transformer/2026-09-08-adoption-report.md#33-ltx-videoaudio-双流).

## 4. Attention、QKV 和 sequence parallel 的负结果

[Read the archived section](../experiments/transformer/2026-09-08-adoption-report.md#4-attentionqkv-和-sequence-parallel-的负结果).

## 5. Core ML 启动、预加载和缓存

[Read the archived section](../experiments/transformer/2026-09-08-adoption-report.md#5-core-ml-启动预加载和缓存).

### 5.1 必须区分的阶段

[Read the archived section](../experiments/transformer/2026-09-08-adoption-report.md#51-必须区分的阶段).

### 5.2 实测启动数据

[Read the archived section](../experiments/transformer/2026-09-08-adoption-report.md#52-实测启动数据).

### 5.3 当前优化原则

[Read the archived section](../experiments/transformer/2026-09-08-adoption-report.md#53-当前优化原则).

## 6. Public Core ML 与 private ANE

[Read the archived section](../experiments/transformer/2026-09-08-adoption-report.md#6-public-core-ml-与-private-ane).

## 7. 与 vpipe 的低内存策略对照

[Read the archived section](../experiments/transformer/2026-09-08-adoption-report.md#7-与-vpipe-的低内存策略对照).

## 8. 模型级采用矩阵

[Read the archived section](../experiments/transformer/2026-09-08-adoption-report.md#8-模型级采用矩阵).

## 9. 尚未完成的验证

[Read the archived section](../experiments/transformer/2026-09-08-adoption-report.md#9-尚未完成的验证).
