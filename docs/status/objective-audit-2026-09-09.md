# TurboCider 原始目标完成度审计

> 发行边界更新：本文的 sd.cpp/GGUF streaming 数据是历史实验记录。正式 App 已移除 sd.cpp，只支持 native MLX GGUF resident；当前规则见 [native-only GGUF](native-gguf-boundary-2026-09-09.md)。

更新时间：2026-09-09

本审计对应原始目标文件 `pasted-text-1.txt`，按“证据足够才算完成”的标准记录，不把局部 kernel、单次启动或旧配置结果冒充完整 E2E。

| 原始要求 | 当前证据 | 状态 |
|---|---|---|
| 自包含、整洁、可独立运行 | 本轮 `make build/package/test`、App 签名、源码外 `env -i` CLI `models/doctor`、独立性测试 | 基本完成；模型权重和系统媒体工具仍是明确外部资源边界；FastMetal worker 代码已打包但其模型/Python runtime 是可选资源 |
| GGUF 多量化 | pinned sd.cpp 覆盖混合 K-quants；native Q8/Q4 affine 分支 | 基础完成；只实测 Q3_K_S/Q4_K_M/Q8_0，其他 label 不能称为已验收 |
| 低内存 streaming/offload | CPU-staged sd.cpp layer prefetch/evict、mmap、VAE tiling；H3 BF16 双槽与 provenance-bound I8/F32 shard streaming；显式预算 | GGUF 256²、两个 base 1024² seed 和单 seed 1024² LoRA 可用；H3 量化缓存已通过无模型 schema/契约测试，但真实媒体 E2E 尚未测；未证明“不影响性能”，GGUF 当前明确存在 3.9–14.3% 回归 |
| Core ML/ANE 启动优化 | prepare/load-only、zero warmup、生命周期遥测和复用 | 机制完成；多机器、多轮 cold/cache-hit ABBA 仍缺 |
| private ANE 实验 | `experimental/` bridge、shipping static audit、正式 disabled stub | 正式隔离完成；研究性能不能作为产品承诺 |
| Transformer 并行探索 | tensor/sequence/head/CPU/GPU/ANE、1/multi-block、public/private report | 主要结论完成；更多机器/shape/block 矩阵仍缺 |
| 不要求逐 bit/逐像素一致 | quality gate、video gate、显式 approximation contract；`pixel_exact` 仅显式要求时启用 | 机制正确；允许 dtype/量化/融合/执行顺序差异，但近似仍需 shape/finite/provenance/质量/性能分别通过 |

## 本轮纠偏

原来的 `--params-backend disk --stream-layers` 不是有效的 layer streaming 配置，因为 pinned sd.cpp 只有在 diffusion 参数使用 CPU backend 时才启用 `--stream-layers`。当前正式路径改为：

```text
--params-backend diffusion=cpu,te=disk,vae=disk
--mmap --stream-layers --max-vram <GiB> --vae-tiling
```

benchmark 现在显式记录 `shape_equal`、`finite` 和 paired output count；缺少任一配对、shape/finite 失败或数值质量失败都不能通过。`pixel_exact` 仍是可选严格诊断，不再被错误地当作所有后端的普适正确性定义。

## 未完成的关键闭环

- GGUF 1024²更多 prompt/seed/LoRA adapter、更多量化、真实低内存物理机器和跨机器矩阵；当前已补两个 base seed 与一个 LoRA seed；
- Q4/Q8 GPU+ANE 的自动设备策略、LoRA 绑定 artifact 和多尺寸质量门禁；
- H3 完整媒体 E2E、量化 streaming 的真实 checkpoint 验收；LTX per-block streaming 和 hybrid 质量；
- LLaDA 独立 LoRA、FLUX 9B hybrid；
- 所有模型的多 adapter、取消、异常恢复和 cache invalidation 矩阵。

因此当前应标记为“核心架构和主要路径已实现，若干模型级生产门禁仍未完成”，不能标记为原始目标全部完成。
