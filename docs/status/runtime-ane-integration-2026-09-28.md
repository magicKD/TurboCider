# Runtime-weight FFN：两模型集成与性能筛选

本篇历史记录已迁入 [Runtime ANE 实验归档](../experiments/runtime-ane/runtime-ane-integration-2026-09-28.md)。
记录顺序与证据边界见[专题索引](../experiments/runtime-ane/README.md)。
以下保留原章节标题，供已有章节链接继续使用。

## 已实现

[阅读归档章节](../experiments/runtime-ane/runtime-ane-integration-2026-09-28.md#已实现)。

## 第一版真实整请求对照（输出 SIMD 优化前）

[阅读归档章节](../experiments/runtime-ane/runtime-ane-integration-2026-09-28.md#第一版真实整请求对照输出-simd-优化前)。

## SIMD 输出恢复与 chunk 复测

[阅读归档章节](../experiments/runtime-ane/runtime-ane-integration-2026-09-28.md#simd-输出恢复与-chunk-复测)。

## 1024²、多 chunk 与候选切分修正

[阅读归档章节](../experiments/runtime-ane/runtime-ane-integration-2026-09-28.md#1024²多-chunk-与候选切分修正)。

## 本轮最终候选：两模型自动策略

[阅读归档章节](../experiments/runtime-ane/runtime-ane-integration-2026-09-28.md#本轮最终候选两模型自动策略)。

## 使用与重现

[阅读归档章节](../experiments/runtime-ane/runtime-ane-integration-2026-09-28.md#使用与重现)。

## BF16 最终候选的收尾状态（量化组件接入前）

[阅读归档章节](../experiments/runtime-ane/runtime-ane-integration-2026-09-28.md#bf16-最终候选的收尾状态量化组件接入前)。

## Q4/Q8 affine staging 组件（后续进展）

[阅读归档章节](../experiments/runtime-ane/runtime-ane-integration-2026-09-28.md#q4q8-affine-staging-组件后续进展)。

### 量化组件接入后的 BF16 回归

[阅读归档章节](../experiments/runtime-ane/runtime-ane-integration-2026-09-28.md#量化组件接入后的-bf16-回归)。

## Z 原生 GGUF Q8_0：整模型接入与 chunk 筛选

[阅读归档章节](../experiments/runtime-ane/runtime-ane-integration-2026-09-28.md#z-原生-gguf-q8_0整模型接入与-chunk-筛选)。

### 接入和正确性修正

[阅读归档章节](../experiments/runtime-ane/runtime-ane-integration-2026-09-28.md#接入和正确性修正)。

### 初步整请求筛选

[阅读归档章节](../experiments/runtime-ane/runtime-ane-integration-2026-09-28.md#初步整请求筛选)。

### c352 交错复测

[阅读归档章节](../experiments/runtime-ane/runtime-ane-integration-2026-09-28.md#c352-交错复测)。

### 1024² 长序列初测

[阅读归档章节](../experiments/runtime-ane/runtime-ane-integration-2026-09-28.md#1024²-长序列初测)。

### 同构建 BF16 / 冻结图回归

[阅读归档章节](../experiments/runtime-ane/runtime-ane-integration-2026-09-28.md#同构建-bf16--冻结图回归)。

### 可复用 CLI 入口

[阅读归档章节](../experiments/runtime-ane/runtime-ane-integration-2026-09-28.md#可复用-cli-入口)。

## 本次整理与验证

[阅读归档章节](../experiments/runtime-ane/runtime-ane-integration-2026-09-28.md#本次整理与验证)。

## 剩余工作

[阅读归档章节](../experiments/runtime-ane/runtime-ane-integration-2026-09-28.md#剩余工作)。
