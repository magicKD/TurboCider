# Qwen base512：F32 down分解、tile筛选与FP16 gate/up负结果

本篇历史实验已迁入[实验归档](../experiments/qwen21-ffn/qwen-base512-ffn-down-tiles-2026-10-18.md)。
相关记录和证据边界见[专题索引](../experiments/qwen21-ffn/README.md)。
以下保留原章节标题，供已有章节链接继续使用。

## 保持什么、改变什么

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-ffn-down-tiles-2026-10-18.md#保持什么改变什么)。

## 独立分解：gate/up约8.28ms，down约4.27ms

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-ffn-down-tiles-2026-10-18.md#独立分解gateup约828msdown约427ms)。

## 12档tile：初始微弱差异未通过重复验证

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-ffn-down-tiles-2026-10-18.md#12档tile初始微弱差异未通过重复验证)。

## FP16 gate/up：每次转换也计入，结果更慢

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-ffn-down-tiles-2026-10-18.md#fp16-gateup每次转换也计入结果更慢)。

## 更严格的独立数值审计

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-ffn-down-tiles-2026-10-18.md#更严格的独立数值审计)。

## 工程、原件与后续

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-ffn-down-tiles-2026-10-18.md#工程原件与后续)。
