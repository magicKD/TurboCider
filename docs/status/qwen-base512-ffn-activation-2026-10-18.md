# Qwen base512：定位SiLU近似，固定图内的等价改写

本篇历史实验已迁入[实验归档](../experiments/qwen21-ffn/qwen-base512-ffn-activation-2026-10-18.md)。
相关记录和证据边界见[专题索引](../experiments/qwen21-ffn/README.md)。
以下保留原章节标题，供已有章节链接继续使用。

## 为什么不能直接改成原生SiLU

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-ffn-activation-2026-10-18.md#为什么不能直接改成原生silu)。

## 原始FP16探针：融合后的结果与独立sigmoid不同

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-ffn-activation-2026-10-18.md#原始fp16探针融合后的结果与独立sigmoid不同)。

## 固定FFN互补激活：完整block约5.1%信号

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-ffn-activation-2026-10-18.md#固定ffn互补激活完整block约51信号)。

## 溢出：不以数值回退换掉固定Program合同

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-ffn-activation-2026-10-18.md#溢出不以数值回退换掉固定program合同)。

## 完整文生图与编辑：matched stock与反向/diagnostic复核完成

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-ffn-activation-2026-10-18.md#完整文生图与编辑matched-stock与反向diagnostic复核完成)。

## 同库五组质量配对

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-ffn-activation-2026-10-18.md#同库五组质量配对)。

## vpipe与运行时灵活性边界

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-ffn-activation-2026-10-18.md#vpipe与运行时灵活性边界)。

## 构建与范围

[阅读归档章节](../experiments/qwen21-ffn/qwen-base512-ffn-activation-2026-10-18.md#构建与范围)。
