# FLUX / Z-Image 优化验收记录

更新时间：2026-09-07。硬件：Apple M4 Max，64 GB unified memory。所有数字均为同一 TurboCider 持久 C ABI 会话，首轮单列为 cold/warm-up，后续为 warm request wall；模型权重和 Core ML artifact 不进入 Git。

## 已完成

- MLX 全局 RMSNorm 使用 `mx::fast::rms_norm`，保留 FP32 累加和 BF16 输出边界。
- FLUX 4B 与 Z-Image 的 Q/K RoPE 使用一次原生 Metal dispatch；attention 使用 MLX fast SDPA，只有在 profile/geometry 适合时才允许显式 force-fused profiling。
- FLUX 4B GPU 默认使用 compiled single-block 图，step 内不再对每个 single block 做 host synchronization；hybrid 路径使用 compiled GPU complement 与 Core ML MLP prefix 并行。
- Z-Image 使用 compiled full block、compiled hybrid complement、BF16 modulation boundary 和 session-wide Core ML output backing。
- FLUX 4B 和 Z-Image 的自动 GPU+ANE 仍受 exact device、checkpoint、shape、artifact provenance、内存预算和质量门禁约束。

## FLUX.2 Klein 4B

工作负载为 512×512、4 steps、seed 42。最终 GPU warm 中位数为 **2.2663 s**；当前 direct MLX engine 为 **2.2642 s**，TurboCider/direct 比值 1.00094（约慢 0.09%，在测量噪声内）。GPU+ANE a6144 warm 中位数为 **1.6279 s**，相对 TurboCider GPU 为 **1.392×**。GPU+ANE 使用 ANE `[0,6144)`、GPU `[6144,9216)`，每请求 80 次 Core ML 调用，session output copy 为 0。

输出质量：GPU↔GPU+ANE PNG correlation 0.999262、cosine 0.999840；GPU↔direct correlation 0.998840、cosine 0.999751。Q/K RoPE 的 A/B warm 中位数由 2.3077 s 降至 2.2665 s；逐 block 同步恢复后为 2.2717 s，说明主要收益来自单 dispatch RoPE 与减少 host wait。强制 fused SDPA 的 warm 中位数为 2.2626 s，但在本机只比 heuristic 小幅波动，因此默认保留 MLX heuristic，force-fused 仅作为开关。

本轮没有宣称 FLUX 与 ComfyUI 的新鲜对照：本机 ComfyUI 0.32.0 没有与 TurboCider 同一 FLUX.2 Klein 4B 权重/工作流可复现。FLUX 的 GPU 门禁采用同一请求的当前 direct engine，Z-Image 则有 stock ComfyUI 对照。

## Z-Image Turbo

工作负载为 1024×1024、9 steps、seed 42。最终 GPU warm 中位数为 **36.3685 s**（39.4728 s cold）；stock ComfyUI 0.32.0、custom nodes disabled 的 warm 中位数为 40.110 s，TurboCider GPU 快 1.103×。a4096 GPU+ANE warm 中位数为 **30.0014 s**（37.5838 s cold），相对优化 GPU 为 **1.212×**；warm 范围 29.9956–30.0072 s，Core ML 每请求 288 次，output copy 为 0。

GPU↔ANE PNG correlation 0.999231、cosine 0.999903、MAE 0.879/255。与 ComfyUI 的共享噪声 parity 仍采用已有报告：PNG correlation 0.998665、cosine 0.999770；latent correlation 0.998697、cosine 0.998119。

## 当前默认与边界

| 路径 | 默认状态 | 结论 |
|---|---|---|
| FLUX 4B GPU | 自动 compiled GPU | 不慢于 direct；自动启用单 dispatch RoPE、compiled block 和 step 级同步 |
| FLUX 4B M4 Max GPU+ANE | exact profile 自动可选 | 1.392× warm；必须使用 a6144、同一 checkpoint，LoRA 需要绑定 artifact |
| Z-Image GPU | 自动 compiled GPU | 快于 stock ComfyUI 1.103× |
| Z-Image M4 Max GPU+ANE | exact base a4096 自动可选 | 1.212× warm；带 LoRA 仍不自动选择 |
| FLUX 9B | GPU-only | 无已验证 ANE partition，不误启用 |
| LTX/H3/FastMetal | 保持已有各自门禁 | 不因本轮 FLUX/Z-Image 结果扩大其自动策略 |

## 尚未完成

- FLUX 4B 多尺寸、多机器和严格 AB/BA p50/p95 矩阵；当前是 M4 Max 512² 的稳定复测。
- FLUX 4B fresh matched ComfyUI 对照；本机缺同一 4B checkpoint/workflow，当前不伪造结论。
- Z-Image LoRA-bound GPU+ANE 的优化后多轮 warm；基础 a4096 已通过门禁，LoRA 仍显式。
- FLUX 9B GPU+ANE、LTX/H3 的新 ANE 默认策略，以及 H3/LTX 逐层纯内存 LoRA。

原始性能和 parity 摘要：[`flux2-m4max-2026-09-07.json`](../design/validation/flux2-m4max-2026-09-07.json)、[`z-image-gpu-comfy-2026-09-07.json`](../design/validation/z-image-gpu-comfy-2026-09-07.json)、[`z-image-auto-m4max-2026-09-07.json`](../design/validation/z-image-auto-m4max-2026-09-07.json)、[`z-image-formats-lora-2026-09-07.json`](../design/validation/z-image-formats-lora-2026-09-07.json)。整体状态见 [`current-status-2026-09-07.md`](current-status-2026-09-07.md)。
