# ConvRot512²：融合 typed decode 的 GPU F32 partial 实际盈利

本篇历史实验已迁入[实验归档](../experiments/quantized-execution/local512-convrot-mpp-partial-2026-10-08.md)。
相关记录和证据边界见[专题索引](../experiments/quantized-execution/README.md)。
以下保留原章节标题，供已有章节链接继续使用。

## 新 GPU kernel 与源码边界

[阅读归档章节](../experiments/quantized-execution/local512-convrot-mpp-partial-2026-10-08.md#新-gpu-kernel-与源码边界)。

## 实权重组件：F32 partial显著改善，不是packed QMM整体提速

[阅读归档章节](../experiments/quantized-execution/local512-convrot-mpp-partial-2026-10-08.md#实权重组件f32-partial显著改善不是packed-qmm整体提速)。

## 接入模型：不可变请求snapshot，默认不变

[阅读归档章节](../experiments/quantized-execution/local512-convrot-mpp-partial-2026-10-08.md#接入模型不可变请求snapshot默认不变)。

## 两方向实模型：原混合约5.70s →4.08s，优化GPU约4.50s

[阅读归档章节](../experiments/quantized-execution/local512-convrot-mpp-partial-2026-10-08.md#两方向实模型原混合约570s-408s优化gpu约450s)。

### 冷溢出：零重试窗口失败，不可追认正式资格

[阅读归档章节](../experiments/quantized-execution/local512-convrot-mpp-partial-2026-10-08.md#冷溢出零重试窗口失败不可追认正式资格)。

## 画面和质量范围

[阅读归档章节](../experiments/quantized-execution/local512-convrot-mpp-partial-2026-10-08.md#画面和质量范围)。

## 构建、回归和清理

[阅读归档章节](../experiments/quantized-execution/local512-convrot-mpp-partial-2026-10-08.md#构建回归和清理)。
