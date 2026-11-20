# Qwen base512：restore encoder实测与GPU complement消融

本篇历史实验已迁入[实验归档](../experiments/qwen21-ffn/qwen-base512-ffn-handoff-encoder-2026-10-18.md)。
相关记录和证据边界见[专题索引](../experiments/qwen21-ffn/README.md)。
以下保留原章节标题，供已有章节链接继续使用。

## 把event等待与encoder执行区间分开

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-ffn-handoff-encoder-2026-10-18.md#把event等待与encoder执行区间分开)。

## 真实完整block重放，不冒充40步图片

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-ffn-handoff-encoder-2026-10-18.md#真实完整block重放不冒充40步图片)。

## 结果：5120的设备侧restore区间也变长

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-ffn-handoff-encoder-2026-10-18.md#结果5120的设备侧restore区间也变长)。

## GPU gate/up打包合并：没有收益，不增加运行时负担

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-ffn-handoff-encoder-2026-10-18.md#gpu-gateup打包合并没有收益不增加运行时负担)。

## 独立审计与工程入口

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-ffn-handoff-encoder-2026-10-18.md#独立审计与工程入口)。
