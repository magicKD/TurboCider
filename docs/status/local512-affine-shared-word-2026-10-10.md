# ConvRot/GGUF：共享 packed-word 解码，F32 partial 接近快速 BF16 路线

本篇历史实验已迁入[实验归档](../experiments/quantized-execution/local512-affine-shared-word-2026-10-10.md)。
相关记录和证据边界见[专题索引](../experiments/quantized-execution/README.md)。
以下保留原章节标题，供已有章节链接继续使用。

## Kernel：一个 packed word 解一次，供整个 TG 复用

[阅读归档章节](../experiments/quantized-execution/local512-affine-shared-word-2026-10-10.md#kernel一个-packed-word-解一次供整个-tg-复用)。

## 数值验证与保留失败

[阅读归档章节](../experiments/quantized-execution/local512-affine-shared-word-2026-10-10.md#数值验证与保留失败)。

## 真权重组件：胜过旧 F32，不等于全面胜过 packed QMM

[阅读归档章节](../experiments/quantized-execution/local512-affine-shared-word-2026-10-10.md#真权重组件胜过旧-f32不等于全面胜过-packed-qmm)。

## 接入既有 opt-in F32 API，保持快 BF16 路线不变

[阅读归档章节](../experiments/quantized-execution/local512-affine-shared-word-2026-10-10.md#接入既有-opt-in-f32-api保持快-bf16-路线不变)。

## 最新同库完整请求：不以慢原 F32 为唯一分母

[阅读归档章节](../experiments/quantized-execution/local512-affine-shared-word-2026-10-10.md#最新同库完整请求不以慢原-f32-为唯一分母)。

## 视觉、构建、清理与仍待推进

[阅读归档章节](../experiments/quantized-execution/local512-affine-shared-word-2026-10-10.md#视觉构建清理与仍待推进)。
