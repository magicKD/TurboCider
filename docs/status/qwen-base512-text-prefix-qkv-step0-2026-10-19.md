# Qwen base512：保留首步GPU QKV，修复text-prefix反例并保留收益

本篇历史实验已迁入[实验归档](../experiments/qwen21-cache/qwen-base512-text-prefix-qkv-step0-2026-10-19.md)。
相关记录和证据边界见[专题索引](../experiments/qwen21-cache/README.md)。
以下保留原章节标题，供已有章节链接继续使用。

## 为什么cache本身独立，结果却曾退化

[阅读归档章节](../experiments/qwen21-cache/qwen-base512-text-prefix-qkv-step0-2026-10-19.md#为什么cache本身独立结果却曾退化)。

## 实现：在stage之前路由，不新建ANE图

[阅读归档章节](../experiments/qwen21-cache/qwen-base512-text-prefix-qkv-step0-2026-10-19.md#实现在stage之前路由不新建ane图)。

## 完整请求性能：同库、stock强对照

[阅读归档章节](../experiments/qwen21-cache/qwen-base512-text-prefix-qkv-step0-2026-10-19.md#完整请求性能同库stock强对照)。

## 工作量、热编译与prompt长度

[阅读归档章节](../experiments/qwen21-cache/qwen-base512-text-prefix-qkv-step0-2026-10-19.md#工作量热编译与prompt长度)。

## 图像与原行为回归

[阅读归档章节](../experiments/qwen21-cache/qwen-base512-text-prefix-qkv-step0-2026-10-19.md#图像与原行为回归)。

## vpipe与灵活runtime的边界

[阅读归档章节](../experiments/qwen21-cache/qwen-base512-text-prefix-qkv-step0-2026-10-19.md#vpipe与灵活runtime的边界)。

## 构建、测试与范围

[阅读归档章节](../experiments/qwen21-cache/qwen-base512-text-prefix-qkv-step0-2026-10-19.md#构建测试与范围)。
