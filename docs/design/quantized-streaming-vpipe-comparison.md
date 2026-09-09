# 量化、Streaming Offload 与 vpipe/H3/LTX 对照

更新时间：2026-09-09

## 对照边界

vpipe 是独立的 Apple Silicon runtime，使用 custom Metal forward、4/8-bit 模型准备、真实 tensor table 的内存 floor、可增长的 block residency、双 reusable refill slot 和 `pread` 预取下一 block。它的公开 H3 数据是 960×544、124 frames、6 DiT steps，在 M4 Pro/M5/M5 Pro 上测得；TurboCider 当前 H3 是 4-step、M4 Max 数据，两者不是同一 workload，不能直接声称谁普遍更快。

能直接迁移的不是某个项目的类，而是以下原则：

```text
checkpoint tensor table
        │
        ├── trunk / text / VAE floor
        ├── per-block bytes
        └── activation + double-buffer scratch
                         │
                         ▼
             resident / pinned / streamed plan
                         │
        current block ──┴── pread next block into reusable slot
```

## TurboCider 当前实现

### Z-Image GGUF

TurboCider 已支持 Q2–Q8、IQ、F16/BF16/F32 descriptor 解析；mixed K-quant 由固定版本 stable-diffusion.cpp Metal 执行，Q8_0 还可切到 native MLX affine quantized matmul 和 checkpoint-bound GPU+ANE。LoRA 使用独立文件，混合 K-quant 的请求期 LoRA 由 sd.cpp 处理，不生成永久 merged checkpoint。

当前实现将 diffusion 参数放在 CPU backend、text encoder/VAE 放在 disk
backend；这是 sd.cpp 真正启用 `--stream-layers` 的前提。旧的全 disk backend
虽然降低了 footprint，但会忽略 layer streaming，现只保留为历史对照。

Q3_K_S、Q4_K_M、Q8_0 的 256²、9-step 重复 ABBA×2 对照（每条路线四个 warm 样本）如下：

| Variant | Resident warm median | Streaming warm median | Overhead | Physical footprint reduction | RGB |
|---|---:|---:|---:|---:|---|
| Q3_K_S | 9.551 s | 10.031 s | +5.02% | 33.3% | exact |
| Q4_K_M | 9.497 s | 9.994 s | +5.23% | 37.7% | exact |
| Q8_0 | 9.341 s | 10.679 s | +14.32% | 45.9% | exact |

所有十二个配对输出的 decoded RGB 都逐像素一致。这里的
`memory_budget_bytes=8 GiB` 只是 sd.cpp 的 `--max-vram` working-set hint；实际
child lifetime physical footprint 约为 10.00–10.01 GB，不应写成总进程被限制在
8 GiB。当前 1.02 material-regression gate 对三种量化都未通过，因此 streaming
是正确的低内存显式 fallback，而不是无代价优化。

Q4_K_M 1024²的 8 GiB 重复对照为 resident `94.586 s`、streaming
`107.956 s`，footprint 降低 `35.2%`。输出不逐像素一致，但 correlation
`0.998117`、cosine `0.999823`、MAE `2.147/255`，通过当前显式质量门禁。
第二个 base seed `314159` 复现为 resident `94.553 s`、streaming `107.816 s`
（`1.1403×`）、footprint 降低 `35.17%`、correlation `0.997056`、MAE
`2.679/255`。16 GiB 单次方向性 probe 仍慢 `11.43%`。Q4_K_M + 独立官方
LoRA 的 256²对照降低 `37.6%` footprint、慢 `3.86%`，4/4 decoded pixels exact；
1024² seed 42 首次 LoRA streaming 对照为 resident `135.442 s`、streaming
`148.673 s`（`1.0977×`）、footprint 降低 `34.85%`、correlation `0.998205`、
MAE `2.136/255`，质量通过但性能未过门禁。完整证据见
[`z-image-gguf-streaming-2026-09-09.json`](validation/z-image-gguf-streaming-2026-09-09.json)。

### H3

H3 现在有 resident 和 BF16 SSD streaming 两条正式路径。streaming 使用两个 Metal BF16 layer slot，在计算当前 block 时后台 `pread` 下一 block，并在 command boundary 等待；这已经避免了把完整 transformer 常驻 unified memory。`ssd_streaming` 与当前运行期 INT8 MLP/QKV 互斥，原因是量化 buffer、scale 和 reload 生命周期尚未统一。

当前还加入了动态 pinned-prefix：当 streamed H3 请求提供 `memory_budget_bytes` 时，运行时按实际请求的 activation geometry、两个 BF16 slot 和固定安全余量计算每个完整 block 的 BF16 payload，自动保留尽可能多的前置 active blocks，剩余 suffix 继续 disk streaming。显式 `ssd_pinned_prefix` 只能减少自动选择的前缀，不能让预算超限；策略始终保留至少一个 streamed block。没有预算时仍保持原来的两-slot 行为，避免改变既有默认性能。

这不是硬性进程内存上限：预算是 H3 DiT working-set target，系统 allocator、VAE、文本编码器和文件缓存仍可能产生额外 footprint。运行结果会记录 pinned/streamed block 数、估算 block/activation bytes、总读取和未隐藏等待时间。

真实 62 GiB Transformer 的 256²、22 帧、四步 A/B/B/A probe 中，16 GiB 预算选择 14 pinned/36 streamed blocks；四份最终 video/audio latent 字节完全一致。无预算与 16 GiB 的 denoise 中位数分别为 17.306 s 和 15.180 s（1.140×），每次读取由 154.91 GB 降到 111.75 GB，等待中位数由 10.738 s 降到 6.917 s；但 pinned 首次装载更慢，fresh 总时间中位数为 22.932 s 对 22.943 s，仅基本持平。

TurboCider Session 现已允许 streamed/pinned DiT 和 conditioning 跨请求保留，同时关闭 streamed 路线的 resident Video VAE/TAEH3 decoder，避免改变原有阶段式 VAE 内存生命周期。同一进程的两次 retained probe 中，四份输出 SHA-256 仍完全一致；第二次 16 GiB pinned denoise 为 11.109 s，无预算 streamed 为 15.469 s，读取量降低 28.0%、未隐藏等待降低 48.4%。这是单机单次 warm 对照，OS file cache 未清空，不替代完整 E2E ABBA。证据见 [fresh DiT probe](validation/h3-ssd-pinned-prefix-dit-2026-09-08.json)与 [retained DiT probe](validation/h3-ssd-pinned-prefix-retained-2026-09-08.json)。

### LTX 2.5

LTX 当前只有 resident/component-staged；component-staged 会按 text/transformer/VAE 生命周期释放组件，但尚未像 vpipe 一样完成每 block 双 slot streaming。LTX 的 non-block trunk（尤其 text connector）约数 GB，不能只按 Transformer block 大小估 floor；低内存计划还必须把 audio/video arena、VAE peak 和 stage boundary 纳入预算。

## vpipe 思路与 TurboCider 的差距

| 能力 | vpipe | TurboCider | 差距 |
|---|---|---|---|
| custom Metal forward | 全部 generative stack | H3/LTX custom Metal；图像模型主要 MLX/Metal | 图像模型仍依赖 MLX runtime |
| quantized preparation | 4/8-bit 预处理 | GGUF 原生、多种 Q-format；H3 运行期 INT8 | H3 streaming 还不能带量化 shard |
| block streaming | 通用双 slot + pread | H3 双 slot；GGUF 委托 sd.cpp | LTX 尚未 per-block streaming |
| dynamic residency | 依据 trunk、block bytes、scratch、RAM 增长/回收 | H3 budget-driven pinned-prefix；其余为 resident/component-staged/streamed | LTX/GGUF 尚无统一 tuner |
| low-memory E2E | 16 GB 工作流已有公开案例 | GGUF 8 GiB hint 已验证 256²、两个 base 1024² seed 和一个 LoRA 1024² seed；H3/LTX 仍需完整矩阵 | hint 不是 8 GB 物理机证明；单 prompt/机器不能外推 |

## 优化优先级

1. 让 H3 streaming slot 支持已量化的 FC1/FC2/QKV payload，必要时在 refill 阶段只转换 scale，不重新 materialize 全 BF16；
2. 恢复完整 H3 tokenizer/text encoder/VAE fixture，做 resident/streamed/pinned 的真实媒体 E2E ABBA；
3. 给 LTX 加 stage-aware 双 slot refill，明确 text connector、双流 audio/video 和 VAE 的 floor；
4. 对不能 raw-copy 的 F32 modulation tensor使用受控慢路径，不因少数 tensor 放弃整个 block streaming；
5. 将 quality gate 与 memory gate 同时纳入自动策略：低内存节省不能以 silent steps/shape/token 裁剪换取。

## 当前判断

TurboCider 已经具备可交付的 GGUF resident/streaming 路径；256² Q3/Q4/Q8
以及 Q4 独立 LoRA 的质量检查通过，Q4 1024²两个 base seed 和一个 LoRA seed
也通过近似图片质量门禁，
但所有 corrected streaming 路线均未通过 1.02 material-regression gate。因此它是
正确的显式低内存 fallback，而不是自动性能优化。H3 pinned-prefix 与 retained
DiT reuse 已完成真实 Transformer 验证，但量化 refill 和完整媒体 E2E 仍缺。
TurboCider 还不是 vpipe 那种覆盖所有 DiT 的通用低内存调度器；下一步重点是
H3 quantized refill、GGUF 1024²更多 prompt/seed/adapter，以及 LTX per-block streaming 的
16/24/32 GB 矩阵。

证据：[Z-Image GGUF streaming 2026-09-09](validation/z-image-gguf-streaming-2026-09-09.json)、[Z-Image GGUF 总结](z-image-gguf.md)、[Transformer 异构报告](transformer-heterogeneous-report.md)。vpipe 仅作为外部设计参考，不进入 TurboCider 构建或运行时依赖。
