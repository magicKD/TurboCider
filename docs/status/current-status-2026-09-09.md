# TurboCider 当前目标状态

更新时间：2026-09-09

验证主机：Apple M4 Max，64 GB unified memory，macOS 26.6.2；GGUF 使用固定版本 stable-diffusion.cpp Metal。

## 结论先行

当前 `dev` 已具备可构建、可独立启动的 Apple-native TurboCider runtime，正式构建不链接 private ANE API。FLUX、Z-Image、LLaDA、H3/LTX、GGUF 和统一请求/计划/生命周期均已接入，但原始四项目标尚未全部闭环。

本轮的关键修正是 Z-Image GGUF 低内存路径：旧的
`--params-backend disk --stream-layers` 会被 sd.cpp 忽略 layer streaming；现在改为
`diffusion=cpu,te=disk,vae=disk` 加 `--mmap --stream-layers`，因此真正启用 CPU-staged layer prefetch/evict。这个路径的目标是降低 footprint，不是自动加速。

## 跨后端正确性口径

TurboCider 不要求 GPU、MLX、Core ML/ANE、GGUF 和参考实现逐 bit 或逐像素相同。允许量化误差、dtype 差异、算子融合差异、执行顺序差异和随机数实现差异，但以下项目仍是硬门禁：

- shape、帧数、operation 和模型规定的 steps 正确；
- 输出可解码且 finite；
- checkpoint、VAE、text encoder、LoRA 文件和 strength 的 provenance 正确；
- 近似图片默认满足 correlation ≥ 0.99、cosine ≥ 0.995、MAE ≤ 5/255；视频还要单帧、帧数、帧率和运动门禁；
- 性能、内存和质量独立判定；近似质量通过不能掩盖性能回归；
- `pixel_exact` 只在 workload 需要时通过 `--require-pixel-exact` 显式启用。

`gpu_ane` 必须显式设置 `allow_approximation=true` 并绑定 manifest；近似路径不会自动冒充 exact GPU 路径。

## 已完成

| 项目 | 当前状态 | 证据/边界 |
|---|---|---|
| GGUF resident | 已完成 | Q3_K_S、Q4_K_M、Q8_0 真实生成；TurboCider 与同版本 sd.cpp direct 的已测请求 ratio ≤ 1.0、输出 exact |
| GGUF streaming | 已完成基础实现 | CPU-staged layer streaming；Q3/Q4/Q8 256²质量通过、footprint 降 33.3–45.9%；所有路线未过 1.02 性能门禁，因此只作显式 fallback |
| GGUF 1024² | 初步完成 | Q4_K_M 8 GiB hint：94.586 s → 107.956 s，footprint 降 35.2%，correlation 0.998117、cosine 0.999823、MAE 2.147/255；单 seed、非 exact |
| 独立 LoRA | 已完成基础请求期路径 | GGUF LoRA 保持独立文件，不生成 merged checkpoint；Q4 256² LoRA 4/4 exact，streaming footprint 降 37.6%，但慢 3.86% |
| Core ML/ANE 启动 | 机制已完成 | load-only prepare、zero-input warmup、first/subsequent prediction 分开；可把启动工作移到用户可见生成前，但不能减少总计算 |
| private ANE 隔离 | 已完成 | private bridge 仅在 `experimental/`；正式 dylib 无 `_ANE*` private symbols/framework |
| Transformer 异构实验 | 主要结论已完成 | FFN channel split 是当前有效方向；sequence/head split 的 Amdahl 负结果、public/private ANE 差异已记录 |
| H3 streaming | Transformer 级完成 | BF16 双 slot、background `pread`、budget-driven pinned-prefix、retained DiT 已验证；完整媒体 E2E 和量化 refill 仍缺 |

## GGUF 当前实测

修正后的 256²/9 steps/seed 42、ABBA×2、每路线四个计时样本：

| Variant | Resident | Streaming | Streaming/Resident | Footprint reduction | Quality |
|---|---:|---:|---:|---:|---|
| Q3_K_S | 9.551 s | 10.031 s | 1.0502× | 33.3% | 4/4 pixel exact |
| Q4_K_M | 9.497 s | 9.994 s | 1.0523× | 37.7% | 4/4 pixel exact |
| Q8_0 | 9.341 s | 10.679 s | 1.1432× | 45.9% | 4/4 pixel exact |
| Q4_K_M + LoRA | 13.365 s | 13.881 s | 1.0386× | 37.6% | 4/4 pixel exact |

1024² Q4_K_M 8 GiB hint的 streaming 质量通过默认近似门禁，但性能未通过；16 GiB 单次方向性 probe 仍为 1.1143×，说明增加 hint 尚未消除 staging/scheduling/VAE tiling 固定开销。`memory_budget_bytes` 是 sd.cpp `--max-vram` working-set hint，不是整个进程的物理内存上限。

完整摘要：[z-image-gguf-streaming-2026-09-09.json](../design/validation/z-image-gguf-streaming-2026-09-09.json)。

## 其他模型与异构路径

- Z-Image BF16 derived GPU+ANE 是当前图像全局最优候选；ConvRot native GPU+ANE 相对对应 GPU 约 1.249×，但仍慢于 BF16-derived ANE，所以不自动替换。
- LLaDA 原生 GPU+ANE 在单机 1024² warm 约 1.203×，但仍缺多机器、多 seed 和独立 LoRA，保持显式 `gpu_ane`。
- LTX GPU+ANE 的完整视频 RGB gate 当前未通过，不能把局部 kernel 或单次速度写成正式 E2E 加速；默认继续 GPU。
- FLUX 4B 已有验证 profile；FLUX 9B hybrid 尚未完成矩阵。
- private ANE 的 micro handoff/MLP 可快约 1.3–1.4×，但 H3 完整 E2E 仅约 1.041×，且 private API 仍只作研究证据。

## 尚未完成

1. GGUF 1024² 多 seed、LoRA 1024²、更多 Q-format 和真实 8/16/24 GB 物理机器矩阵。
2. GGUF Q4/Q8 GPU+ANE 多尺寸、多机器、LoRA 和自动策略门禁。
3. H3 完整 tokenizer/text encoder/VAE/MP4 E2E、量化 streaming refill；LTX per-block streaming 和 16/24/32 GB 验收。
4. LTX hybrid 质量修复、多 prompt/seed；LLaDA 独立 LoRA；FLUX 9B hybrid。
5. FLUX/Z-Image/GGUF 的多 adapter、取消、失败恢复、缓存失效和跨机器完整矩阵。

## 代码和提交状态

`.gitignore` 已排除模型权重、`.deps/`、Python 环境、build/dist、Core ML cache、outputs、日志和 machine-local profiles；curated validation JSON 与设计文档保留。本状态对应的源码、测试和文档组成同一个 scoped GGUF streaming 修正版本，生成物和原始模型没有进入提交。

本轮已通过：

```text
Python/bin/python3 -m pytest -q \
  tests/native/test_gguf_streaming_benchmark.py \
  tests/native/test_quality_gate.py \
  tests/native/test_video_quality_gate.py \
  tests/native/test_contract.py
68 passed, 49 subtests passed
```

此外，完整 `make test` 的所有 target 均通过（2 个 Python 测试因当前环境无 NumPy 而明确 skip），native/Swift build、public Vision helper、`make package`、App behavior test、App 深度签名、portable `env -i` CLI `models/doctor`、动态依赖和 private ANE symbol audit 均已通过。`git diff --check` 通过，索引中没有模型权重、`outputs/`、build 或 dist 产物。
