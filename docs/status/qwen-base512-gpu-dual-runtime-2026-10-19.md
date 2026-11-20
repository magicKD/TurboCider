# Qwen base512：固定ANE图下的GPU complement融合验收

本篇历史实验已迁入[实验归档](../experiments/qwen21-ffn/qwen-base512-gpu-dual-runtime-2026-10-19.md)。
相关记录和证据边界见[专题索引](../experiments/qwen21-ffn/README.md)。
以下保留原章节标题，供已有章节链接继续使用。

## 固定图与动态长度

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-gpu-dual-runtime-2026-10-19.md#固定图与动态长度)。

## 本轮实现

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-gpu-dual-runtime-2026-10-19.md#本轮实现)。

## 同库stock完整请求，不是稳定加速结论

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-gpu-dual-runtime-2026-10-19.md#同库stock完整请求不是稳定加速结论)。

## 瓶颈：缩短FFN也会改变QKV准备窗口

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-gpu-dual-runtime-2026-10-19.md#瓶颈缩短ffn也会改变qkv准备窗口)。

## 四个独立timeline：变长不新建ANE图

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-gpu-dual-runtime-2026-10-19.md#四个独立timeline变长不新建ane图)。

## vpipe对照与灵活调度的边界

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-gpu-dual-runtime-2026-10-19.md#vpipe对照与灵活调度的边界)。

## 构建与复跑

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-gpu-dual-runtime-2026-10-19.md#构建与复跑)。

## 验收、内存与交付边界

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-gpu-dual-runtime-2026-10-19.md#验收内存与交付边界)。
