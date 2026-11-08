# Qwen-Image-2.1 512² W8A8/ANE 诊断

本篇历史记录已迁入 [Qwen W8A8 实验归档](../experiments/qwen21-w8a8/qwen21-w8a8-ane-diagnostics-2026-09-25.md)。
记录顺序和证据边界见[专题索引](../experiments/qwen21-w8a8/README.md)。
以下保留原章节标题，供已有章节链接继续使用。

## 已测结果

[阅读归档章节](../experiments/qwen21-w8a8/qwen21-w8a8-ane-diagnostics-2026-09-25.md#已测结果)。

## 修正权重量化粒度后的后续证据

[阅读归档章节](../experiments/qwen21-w8a8/qwen21-w8a8-ane-diagnostics-2026-09-25.md#修正权重量化粒度后的后续证据)。

## GPU 后缀 W8A16 与 BF16 同条件对照

[阅读归档章节](../experiments/qwen21-w8a8/qwen21-w8a8-ane-diagnostics-2026-09-25.md#gpu-后缀-w8a16-与-bf16-同条件对照)。

## 参考图编辑：质量与加速并非随参考图数量单调改善

[阅读归档章节](../experiments/qwen21-w8a8/qwen21-w8a8-ane-diagnostics-2026-09-25.md#参考图编辑质量与加速并非随参考图数量单调改善)。

### 生产路径一致性与逐步耗时定位

[阅读归档章节](../experiments/qwen21-w8a8/qwen21-w8a8-ane-diagnostics-2026-09-25.md#生产路径一致性与逐步耗时定位)。

### 缩小参考前缀及少量 FFN 层回退：512² 三参考候选

[阅读归档章节](../experiments/qwen21-w8a8/qwen21-w8a8-ane-diagnostics-2026-09-25.md#缩小参考前缀及少量-ffn-层回退512²-三参考候选)。

### 相同双参考图的 resident 编辑条件复用

[阅读归档章节](../experiments/qwen21-w8a8/qwen21-w8a8-ane-diagnostics-2026-09-25.md#相同双参考图的-resident-编辑条件复用)。

### Resident Session 的 1–3 参考图实测

[阅读归档章节](../experiments/qwen21-w8a8/qwen21-w8a8-ane-diagnostics-2026-09-25.md#resident-session-的-13-参考图实测)。

### 狐狸文生图：29/32 层回退的另一题材复核

[阅读归档章节](../experiments/qwen21-w8a8/qwen21-w8a8-ane-diagnostics-2026-09-25.md#狐狸文生图2932-层回退的另一题材复核)。

### 正式 CLI 双参考 512² 端到端核验

[阅读归档章节](../experiments/qwen21-w8a8/qwen21-w8a8-ane-diagnostics-2026-09-25.md#正式-cli-双参考-512²-端到端核验)。

### Resident Session 双参考热请求

[阅读归档章节](../experiments/qwen21-w8a8/qwen21-w8a8-ane-diagnostics-2026-09-25.md#resident-session-双参考热请求)。

### 显式 6144-channel W8A8 分区复测

[阅读归档章节](../experiments/qwen21-w8a8/qwen21-w8a8-ane-diagnostics-2026-09-25.md#显式-6144-channel-w8a8-分区复测)。
