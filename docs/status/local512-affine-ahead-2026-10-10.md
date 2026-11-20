# 实际有界GPU ahead-decode：四层真实FFN链已跑，GGUF仍优先packed

本篇历史实验已迁入[实验归档](../experiments/quantized-execution/local512-affine-ahead-2026-10-10.md)。
相关记录和证据边界见[专题索引](../experiments/quantized-execution/README.md)。
以下保留原章节标题，供已有章节链接继续使用。

## 实现：两个逻辑job，不覆盖escaped物理目标

[阅读归档章节](../experiments/quantized-execution/local512-affine-ahead-2026-10-10.md#实现两个逻辑job不覆盖escaped物理目标)。

## 不同层真实source、依赖链、完整窗口

[阅读归档章节](../experiments/quantized-execution/local512-affine-ahead-2026-10-10.md#不同层真实source依赖链完整窗口)。

## 保留的失败与最终回归

[阅读归档章节](../experiments/quantized-execution/local512-affine-ahead-2026-10-10.md#保留的失败与最终回归)。

## 决策与完整目标

[阅读归档章节](../experiments/quantized-execution/local512-affine-ahead-2026-10-10.md#决策与完整目标)。
