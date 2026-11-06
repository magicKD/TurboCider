<a id="transformer-experiment-document--moved"></a>

# Apple M4 Pro 上 Transformer 的 CPU/GPU/ANE 异构并行：切分、同步与编译边界

This document is maintained in the [M4 Pro experiment archive](../../../docs/experiments/transformer/m4-pro/REPORT.md).

This page preserves the original published path. New experiment narratives
belong under `docs/experiments/`; local generated reports use `REPORT.generated.md`.

Original section headings are retained below for existing deep links; the full
record and its evidence remain in the archive.

## 摘要

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#摘要).

## 1. 研究问题与相关工作

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#1-研究问题与相关工作).

## 2. 模型、数学拆分与依赖

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#2-模型数学拆分与依赖).

### 2.1 完整 block

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#21-完整-block).

### 2.2 FFN tensor parallel

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#22-ffn-tensor-parallel).

### 2.3 FFN sequence-row parallel

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#23-ffn-sequence-row-parallel).

### 2.4 Attention-head parallel

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#24-attention-head-parallel).

## 3. 实现与公平性控制

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#3-实现与公平性控制).

## 4. 实验方法

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#4-实验方法).

## 5. 性能结果

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#5-性能结果).

### 5.1 形状与切分比例

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#51-形状与切分比例).

### 5.2 独立 1–3 层确认

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#52-独立-13-层确认).

### 5.3 三设备 CPU 尾部

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#53-三设备-cpu-尾部).

### 5.4 完整 attention-head 并行

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#54-完整-attention-head-并行).

### 5.5 Sequence-row 的失败与编译分配

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#55-sequence-row-的失败与编译分配).

## 6. Public / private API 与重叠消融

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#6-public--private-api-与重叠消融).

## 7. 编译、加载和首次执行

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#7-编译加载和首次执行).

## 8. 数值可靠性与维度敏感性

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#8-数值可靠性与维度敏感性).

## 9. 成本模型与更高效的执行建议

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#9-成本模型与更高效的执行建议).

## 10. 局限与可复现性

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#10-局限与可复现性).

## 11. 缓存清理

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#11-缓存清理).

## 12. 结论

[Read the archived section](../../../docs/experiments/transformer/m4-pro/REPORT.md#12-结论).
