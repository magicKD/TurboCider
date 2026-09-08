# 量化、Streaming Offload 与 vpipe/H3/LTX 对照

更新时间：2026-09-08

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

Q3_K_S 256²、9 steps 的真实 streaming 对照：

| 路径 | warm | physical footprint | RGB |
|---|---:|---:|---|
| resident sd.cpp | 9.534 s | 14.91 GB | — |
| streaming `--mmap --stream-layers --max-vram 8 --vae-tiling` | 9.995 s | 9.52 GB | decoded pixel exact |

physical footprint 降低 36.1%，warm 慢约 4.8%。但每条路线只有一个真实 warm 样本，不能称稳定 median；1024²和更多 quantization 尚未完成。

### H3

H3 现在有 resident 和 BF16 SSD streaming 两条正式路径。streaming 使用两个 Metal BF16 layer slot，在计算当前 block 时后台 `pread` 下一 block，并在 command boundary 等待；这已经避免了把完整 transformer 常驻 unified memory。`ssd_streaming` 与当前运行期 INT8 MLP/QKV 互斥，原因是量化 buffer、scale 和 reload 生命周期尚未统一。

### LTX 2.5

LTX 当前只有 resident/component-staged；component-staged 会按 text/transformer/VAE 生命周期释放组件，但尚未像 vpipe 一样完成每 block 双 slot streaming。LTX 的 non-block trunk（尤其 text connector）约数 GB，不能只按 Transformer block 大小估 floor；低内存计划还必须把 audio/video arena、VAE peak 和 stage boundary 纳入预算。

## vpipe 思路与 TurboCider 的差距

| 能力 | vpipe | TurboCider | 差距 |
|---|---|---|---|
| custom Metal forward | 全部 generative stack | H3/LTX custom Metal；图像模型主要 MLX/Metal | 图像模型仍依赖 MLX runtime |
| quantized preparation | 4/8-bit 预处理 | GGUF 原生、多种 Q-format；H3 运行期 INT8 | H3 streaming 还不能带量化 shard |
| block streaming | 通用双 slot + pread | H3 双 slot；GGUF 委托 sd.cpp | LTX 尚未 per-block streaming |
| dynamic residency | 依据 trunk、block bytes、scratch、RAM 增长/回收 | resident、component-staged、streamed 固定策略 | 缺 pinned-prefix tuner |
| low-memory E2E | 16 GB 工作流已有公开案例 | GGUF 8 GB budget 已验证 256²；H3/LTX 仍需完整矩阵 | 不能把 256²证据外推到大图/视频 |

## 优化优先级

1. 把 H3 当前全 resident/全 streamed 二选一升级为 pinned-prefix：根据真实 block bytes、trunk 和 scratch 预算决定保留前缀；
2. 让 H3 streaming slot 支持已量化的 FC1/FC2/QKV payload，必要时在 refill 阶段只转换 scale，不重新 materialize 全 BF16；
3. 给 LTX 加 stage-aware 双 slot refill，明确 text connector、双流 audio/video 和 VAE 的 floor；
4. 对不能 raw-copy 的 F32 modulation tensor 使用 `kUnservable` 式慢路径，不因少数 tensor 放弃整个 block streaming；
5. 优先做 resident/component-staged/streamed/pinned 的真实 E2E ABBA；不要只比较 SSD 带宽或单层 matmul；
6. 将 quality gate 与 memory gate 同时纳入自动策略：低内存节省不能以 silent steps/shape/token 裁剪换取。

## 当前判断

TurboCider 已经具备可交付的 GGUF resident/streaming 路径和 H3 BF16 streaming 原型，且通过了 Q3_K_S 256² decoded-RGB exact gate；它还不是 vpipe 那种覆盖所有 DiT 的通用低内存调度器。当前最现实的目标是先完成 H3 pinned-prefix + quantized refill，再补 LTX per-block streaming 和 16/24/32 GB 矩阵。

证据：[Z-Image GGUF streaming](validation/z-image-gguf-streaming-2026-09-08.json)、[Z-Image GGUF 总结](z-image-gguf.md)、[Transformer 异构报告](transformer-heterogeneous-report.md)。vpipe 仅作为外部设计参考，不进入 TurboCider 构建或运行时依赖。
