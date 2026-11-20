# ConvRot512²：真实 LoRA、共享旋转和 MPP/F32 混合接续

本篇历史实验已迁入[实验归档](../experiments/quantized-execution/local512-convrot-lora-2026-10-08.md)。
相关记录和证据边界见[专题索引](../experiments/quantized-execution/README.md)。
以下保留原章节标题，供已有章节链接继续使用。

## 完整 ConvRot LoRA 路线，不是只开门/漏掉前缀

[阅读归档章节](../experiments/quantized-execution/local512-convrot-lora-2026-10-08.md#完整-convrot-lora-路线不是只开门漏掉前缀)。

## GPU helper：LoRA 也共享 frozen-base 的 H256

[阅读归档章节](../experiments/quantized-execution/local512-convrot-lora-2026-10-08.md#gpu-helperlora-也共享-frozen-base-的-h256)。

## 真实八步 LoRA：MPP 原混合约12s →9.1s，较GPU小幅快

[阅读归档章节](../experiments/quantized-execution/local512-convrot-lora-2026-10-08.md#真实八步-lorampp-原混合约12s-91s较gpu小幅快)。

## 画面与同图状态切换

[阅读归档章节](../experiments/quantized-execution/local512-convrot-lora-2026-10-08.md#画面与同图状态切换)。

## 回归、构建、保留

[阅读归档章节](../experiments/quantized-execution/local512-convrot-lora-2026-10-08.md#回归构建保留)。
