# Qwen-Image-2.1 runtime LoRA with an unfused adapter and a fused base FFN

本篇历史记录已迁入 [Runtime ANE 实验归档](../experiments/runtime-ane/qwen21-runtime-lora-fused-2026-09-28.md)。
记录顺序与证据边界见[专题索引](../experiments/runtime-ane/README.md)。
以下保留原章节标题，供已有章节链接继续使用。

### Shared-artifact switch check (2026-09-28)

[阅读归档章节](../experiments/runtime-ane/qwen21-runtime-lora-fused-2026-09-28.md#shared-artifact-switch-check-2026-09-28)。

## Measurements

[阅读归档章节](../experiments/runtime-ane/qwen21-runtime-lora-fused-2026-09-28.md#measurements)。

### Further profiling and partition screen (2026-09-28)

[阅读归档章节](../experiments/runtime-ane/qwen21-runtime-lora-fused-2026-09-28.md#further-profiling-and-partition-screen-2026-09-28)。

### Three-reference 512px edit check (2026-09-28)

[阅读归档章节](../experiments/runtime-ane/qwen21-runtime-lora-fused-2026-09-28.md#three-reference-512px-edit-check-2026-09-28)。

### Allocation experiment and shared-base timing (2026-09-28)

[阅读归档章节](../experiments/runtime-ane/qwen21-runtime-lora-fused-2026-09-28.md#allocation-experiment-and-shared-base-timing-2026-09-28)。

### Shared LoRA A experiment rejected (2026-09-28)

[阅读归档章节](../experiments/runtime-ane/qwen21-runtime-lora-fused-2026-09-28.md#shared-lora-a-experiment-rejected-2026-09-28)。

### Packed FP16 delta input screen rejected (2026-09-28)

[阅读归档章节](../experiments/runtime-ane/qwen21-runtime-lora-fused-2026-09-28.md#packed-fp16-delta-input-screen-rejected-2026-09-28)。

### Core ML call boundary breakdown (2026-09-28)

[阅读归档章节](../experiments/runtime-ane/qwen21-runtime-lora-fused-2026-09-28.md#core-ml-call-boundary-breakdown-2026-09-28)。

### Adapter-independent base graph and linear-down screen (2026-09-28)

[阅读归档章节](../experiments/runtime-ane/qwen21-runtime-lora-fused-2026-09-28.md#adapter-independent-base-graph-and-linear-down-screen-2026-09-28)。

### Direct-FP16 delta screen rejected for Qwen (2026-09-28)

[阅读归档章节](../experiments/runtime-ane/qwen21-runtime-lora-fused-2026-09-28.md#direct-fp16-delta-screen-rejected-for-qwen-2026-09-28)。

### Separate GPU suffix stream rejected (2026-09-28)

[阅读归档章节](../experiments/runtime-ane/qwen21-runtime-lora-fused-2026-09-28.md#separate-gpu-suffix-stream-rejected-2026-09-28)。

### Matched current GPU versus shared-base route (2026-09-28)

[阅读归档章节](../experiments/runtime-ane/qwen21-runtime-lora-fused-2026-09-28.md#matched-current-gpu-versus-shared-base-route-2026-09-28)。

### Distinct adapter-file identity check (2026-09-28)

[阅读归档章节](../experiments/runtime-ane/qwen21-runtime-lora-fused-2026-09-28.md#distinct-adapter-file-identity-check-2026-09-28)。

### Full-GPU FFN blocks 3/5/7 rejected for runtime LoRA (2026-09-28)

[阅读归档章节](../experiments/runtime-ane/qwen21-runtime-lora-fused-2026-09-28.md#full-gpu-ffn-blocks-357-rejected-for-runtime-lora-2026-09-28)。

### Deferring the GPU suffix rejected (2026-09-28)

[阅读归档章节](../experiments/runtime-ane/qwen21-runtime-lora-fused-2026-09-28.md#deferring-the-gpu-suffix-rejected-2026-09-28)。
