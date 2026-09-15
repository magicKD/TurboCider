# TurboCider ANE 加速现状与分阶段决策

更新时间：2026-09-15

验证主机主要为 Apple M4 Max、64 GB unified memory。本文汇总仓库已有
encoder、DiT、VAE、Core ML 启动和完整请求证据；本次代码整理没有重新运行模型
实验，也不把 operator microbenchmark 当作端到端加速结论。

## 结论

TurboCider 当前最有效的 ANE 用法是 Transformer FFN 的 intermediate-channel
split：GPU 保留 attention、RoPE、softmax、长序列归约和未分配的 FFN suffix，
Core ML/ANE 计算完整的 FFN prefix，最后只对 hidden-size partial 做一次 join。

产品和请求边界保持如下：

- 默认 execution 是精确 GPU；GPU/ANE 不作为所有模型、尺寸和设备的隐式默认；
- DiT/denoiser 使用 `ane_manifest`，encoder 使用独立的
  `encoder_ane_manifest`，两类 artifact 不复用；
- 所有显式 ANE 路径要求 `allow_approximation=true`，并验证 manifest ABI、
  checkpoint、shape/bucket 和 LoRA identity；
- FLUX 4B 和 Z-Image 仅在已验证的 M4 Max/M4 Pro profile 上允许自动选择
  DiT hybrid；其他 encoder/DiT 路径保持显式 optional；
- VAE 当前没有 ANE control-plane 或生产级资格结果，继续使用 MLX/Metal GPU；
- private ANE API 只保留在 `experimental/`，正式 runtime 只使用 public Core ML。

## 1. 分阶段状态

| 阶段 | 当前 GPU 路径 | ANE 探索 | 当前决定 |
| --- | --- | --- | --- |
| Tokenizer / prompt preparation | CPU 解析；GPU encoder 前处理 | 没有值得独立迁移的密集计算 | 保持现状 |
| Text / multimodal encoder | GPU attention、RoPE、MLP suffix、consumer taps | Qwen3、Qwen3-VL、Gemma4 的连续 MLP prefix | 显式 `encoder_ane_manifest`；未提供时精确 GPU |
| DiT / denoiser | Metal/MLX attention、projection、FFN complement | FLUX-like branch overlap；sequential block 的 FFN channel split | 仅通过模型、设备、shape 和质量门的 profile 可自动 |
| Latent upsampler / connector | GPU/Metal | 尚无可交付 ANE 结果 | GPU |
| Image VAE | MLX/Metal 2D conv、residual、mid attention | 理论可转换固定 shape Core ML，但没有资格数据 | GPU；不是当前优先项 |
| Video VAE | MLX/Metal 3D/causal conv、tile、chunk、overlap blend | 动态时空 shape 和因果依赖不利于 ANE | GPU；优先 tiling 和生命周期 |
| Audio VAE / vocoder | MLX/Metal 1D/2D conv 和 upsampling | 尚无完整质量/性能矩阵 | GPU |
| Export / mux | CPU/AVFoundation | 不适合 ANE | 保持现状 |

## 2. Encoder ANE

Encoder hybrid 与 denoiser hybrid 已在请求、descriptor、plan 和 runtime
telemetry 中分离。`supports_gpu_ane` 只描述 DiT/denoiser，
`supports_encoder_gpu_ane` 单独描述 encoder；缺失 encoder manifest 时不构造
Core ML session，也不改变 prompt conditioning 数学路径。

以下为 resident encoder-prefill 数据，不包含 tokenizer、模型冷加载、DiT、VAE
和媒体输出：

| Consumer | Encoder | 已测结果 | 结论 |
| --- | --- | --- | --- |
| FLUX Klein 4B/9B | Qwen3 | 保留 256/512/1024/2048 token buckets；warm speedup 最高约 `1.256x` | 质量和 runtime gate 已通过若干 bucket，仍需显式 manifest |
| Z-Image / GGUF | Qwen3 | 64–2048 token buckets 当轮 warm speedup 约 `1.315–1.445x` | 当前 encoder 结果最好，但不是完整 generation 的同倍率收益 |
| FastH3 / VSA / VDN | Qwen3-VL | exact-256 历史 hot-cache 为 `4.57608 / 5.11944 s = 1.119x`；整理后同口径长复跑为 `0.9962x` | 400 次 ANE 调用成功且质量通过，但收益未稳定复现；保持 explicit optional |
| LTX 2.5 | Gemma4 ConvRot | exact-128 GPU `0.508186 s`，8-layer hybrid `0.499790 s`，仅 `1.0168x` | 质量、backing 和 fallback 门通过，但低于 `1.10x` 门槛，不扩展 bank |

H3 exact-256 的 Core ML load 约 `29.606 s`，因此 resident prefill 的局部收益
不能表述成首请求收益。64/128 tokens 会因 `below_min_profitable_rows` 回 GPU，
512 tokens 会因现有 artifact 容量不足回 GPU。

各 encoder ABI 不可混用：

- FLUX/Z-Image Qwen3 4B：hidden 2560，intermediate 9728；
- FLUX 9B Qwen3：hidden 4096，intermediate 12288；
- H3 Qwen3-VL：hidden 5120，intermediate 25600，MRoPE；
- LTX Gemma4：hidden 3840，intermediate 15360，ConvRot、sliding/full attention。

更完整的 encoder 资格和本轮代码边界见
[Encoder prefill 与 GPU/ANE 可选路径状态](../status/encoder-prefill-optional-2026-09-15.md)。

## 3. DiT / denoiser ANE

### 3.1 FLUX-style 并行窗口

FLUX、Z-Image 和 LLaDA 的 attention 与 MLP 可读取同一份 normalized hidden：

```text
normalized hidden ──► GPU attention ──┐
                  └► GPU/ANE FFN ─────┴► hidden join
```

ANE MLP 能部分隐藏在 GPU attention critical path 中，是目前最适合产品化的
拓扑。已有 warm E2E 结果：

| 模型与 workload | GPU | GPU+ANE | 加速 | 质量/状态 |
| --- | ---: | ---: | ---: | --- |
| FLUX 4B，512²，4 steps | 2.2663 s | 1.6279 s | `1.392x` | PNG correlation 0.999262；精确设备 profile 可自动 |
| Z-Image BF16，1024²，9 steps | 36.3685 s | 30.0014 s | `1.212x` | PNG correlation 0.999231；M4 Max base profile 可自动 |
| LLaDA，1024²，4 steps | 17.971 s | 14.934 s | `1.203x` | correlation 0.995194；仍需显式 `gpu_ane` |

LoRA 只有在 Core ML artifact 绑定相同 adapter path、内容、role 和 strength 时才
可显式使用。未完成重复 warm 和质量矩阵的 LoRA hybrid 不进入 `auto`。

### 3.2 Sequential pre-norm 模型

H3/Wan 一类 block 的 MLP 输入依赖 attention residual，不能把同 block attention
与 MLP 画成并行：

```text
GPU attention ─► max(GPU MLP suffix, ANE MLP prefix) ─► join
```

因此 ANE branch 更容易成为 straggler，收益也更容易被 50 个串行 block、session
切换、AdaLN、VAE 和媒体输出稀释。

- H3 real-block MLP micro 可到约 `1.72x`，但 fully-warm 完整 H3 历史结果仅从
  `319.34 s` 到 `306.85 s`，约 `1.041x`；相对当前更快的 GPU INT8-QKV
  baseline 没有产品收益；
- Wan FastMetal 历史完整热路径约 `72.93 s` GPU 对 `69.97 s` GPU+ANE，约
  `1.04x`；现有 hybrid 不满足自动选择价值；
- H3 sequence-row split 的单 block 约 `1.282x`，但 all-50 最好仅约
  `1.009x`，所以 sequence split 不作为默认方向。

### 3.3 LTX 双流

LTX 同时存在 Video/Audio self stream、双向 cross attention 和两支 FFN，但完整
请求会叠加 Core ML session、双 GPU queue、UMA contention、VAE 和 mux 成本。

当前 704×448、97 frames、8+3 steps 记录：

- INT8 GPU+ANE denoise：`52.353 s → 45.761 s`，约 `1.144x`；
- 完整 warm request：`67.493 s → 65.646 s`，只有 `1.028x`；
- latent cosine 0.945149、RGB mean correlation 0.858582，质量门未通过；
- Stage-2-only 路径画质较接近 GPU，但 denoise 仅约 `1.0417x`，仍低于
  `1.10x` 目标。

因此 LTX GPU+ANE 保持显式实验候选，默认继续使用 GPU。完整证据见
[LTX ANE 与 720p tiled VAE](validation/ltx-ane-and-720p-tiled-2026-09-13.json)。

## 4. VAE 阶段

当前没有 `vae_ane_manifest`，也没有任一模型通过 production VAE ANE benchmark。
即使请求使用 `gpu_ane`，现有 VAE 仍在 MLX/Metal GPU 上运行。

### 4.1 图像 VAE

| 模型 | 权重大小 | 结构 | 当前判断 |
| --- | ---: | --- | --- |
| FLUX 4B/9B | 168.121 MB | latent 32；四级 2D conv/residual upsampling；mid attention；GroupNorm；quant/post-quant conv | GPU 很合适；一次 decode 难摊销 Core ML load/handoff |
| LLaDA | 168.121 MB | 与 FLUX2 VAE 基本相同 | GPU；SigVQ 是独立图像语义组件，不是 VAE |
| Z-Image / GGUF | 335.304 MB | latent 16；四级 2D conv/residual；mid attention；scale/shift | GPU；优先解决 allocator cache 和统一内存压力 |

Z-Image 1024² 的正常历史 VAE decode 约 `0.85–1.7 s`。一次内存压力诊断中
hybrid 请求的 VAE 曾升至 `6.277 s`，限制 MLX 空闲缓存后恢复到约 `1.7 s`；
这说明当前更重要的是权重/缓存生命周期，而不是把同一 VAE 搬到 ANE。

### 4.2 视频和音频 VAE

| 模型 | 权重大小 | 结构 | 当前判断 |
| --- | ---: | --- | --- |
| H3 Video VAE | 10.416 GB | 3D conv、时空 chunk、36-layer decoder Transformer、tiled video decode | GPU；动态 shape、大激活和串行依赖不适合当前 ANE bank |
| H3 Audio VAE | 605.429 MB | Conv1D/ConvTranspose1D、多级 residual upsampling | GPU；只有固定长度、高 batch 才值得重新研究 ANE |
| LTX Video VAE | 1.452 GB | 3D causal conv、五级 residual decoder、temporal/spatial depth-to-space、tile blending | GPU + tiling 是已证明的正确方向 |
| LTX Audio VAE | 364.867 MB | 2D causal conv、residual、三级 upsampling，后接 vocoder/BWE | GPU；尚无 ANE 质量/性能矩阵 |
| Wan source VAE | 507.592 MB | latent 16、causal temporal down/up sampling | native 使用转换后的 TAEHV；causal memory/temporal grow 更适合 GPU |

LTX 720p、121-frame 已有同 latent 对照：untiled Video VAE `412.936 s`，
tiled `26.146 s`，约 `15.794x`；完整请求 `635.719 s → 249.052 s`，
PSNR 41.99 dB、SSIM 0.9873。该收益完全来自 GPU tiling，不是 ANE。

512-pixel tile 的默认值继续保留：768-pixel standalone decode 曾从 `12.55 s`
降至 `10.11 s`，但 same-binary integrated helper 中反而从 `26.855 s` 增至
`27.613 s`。这说明 VAE 优化必须看完整生命周期，不能只采用孤立 kernel 数字。
详见 [LTX VAE tile 对照](validation/ltx-vae-tile-768-20260913.json)。

## 5. Core ML 启动、handoff 与内存

ANE operator 只有在 session、program、buffer 和权重可复用时才可能转化为模型
收益。已有 fixed-shape MLP 记录中，fresh process p50 约 `92.378 ms`，hot MLP
约 `1.353 ms`，setup proxy 约 `91.025 ms`。one-shot 小算子不应迁移到 ANE。

当前 runtime 已实现：

- manifest/checkpoint/LoRA provenance 验证；
- load-only prepare、zero-input warmup、first/subsequent prediction 分段 telemetry；
- caller-owned Core ML output backing，避免默认 CPU bridge copy；
- runtime failure、quality、bucket、padding 和 copy 计数；
- 显式 hybrid fail closed；LTX Gemma encoder 的单层 runtime failure 可精确回退
  GPU MLP，不输出缺失的 partial result。

Public Core ML 仍包含同步 prediction barrier。private IOSurface/ANE 数据只用来估计
调度上限，不进入正式构建，也不作为性能承诺。

## 6. 当前采用矩阵

| 模型 | Encoder ANE | DiT ANE | VAE ANE | 默认策略 |
| --- | --- | --- | --- | --- |
| FLUX 4B | explicit optional；若干 bucket 已通过 | M4 Max/M4 Pro 已验证 profile 可自动 | 无 | GPU；精确 profile 可 auto hybrid |
| FLUX 9B | explicit experimental | denoiser 未验证 | 无 | GPU |
| Z-Image BF16 | explicit optional；当前 encoder 数据最好 | M4 Max base a4096 可自动 | 无 | GPU 或资格匹配 auto hybrid |
| Z-Image GGUF | explicit optional | 仅 checkpoint-compatible Q8/Q4 candidate | 无 | native MLX GPU；hybrid 显式 |
| LLaDA | 当前无独立 encoder qualification | 约 1.203x，仍显式 | 无 | GPU |
| H3 / FastH3 / VSA / VDN | exact-256 optional，收益未稳定复现 | 历史完整约 1.041x；VDN 未达 1.20x 门 | 无 | GPU |
| Wan 1.3B QAD | 未形成 UMT5 encoder ANE qualification | 历史约 1.04x | 无 | GPU |
| LTX 2.5 | 8-layer optional，约 1.0168x | denoise 有局部收益但 E2E/质量未过门 | 无 | GPU |

## 7. 代码边界和可维护性

本轮整理后的代码状态：

- request parser 通过 `supports_encoder_gpu_ane` capability 判断，不维护另一份
  容易漂移的 encoder 模型白名单；
- `ane_manifest`、`encoder_ane_manifest`、`hybrid` 和 `encoder_hybrid`
  telemetry 独立，避免把 encoder、DiT 和 VAE 加速混报；
- Qwen3、Qwen3-VL、Gemma4 使用各自的 backend、precision、graph 和
  approximation 标签；
- H3 hybrid conditioning 不写入精确 GPU prompt cache，防止未绑定 Core ML
  artifact identity 的缓存交叉复用；
- LTX Gemma resident weights、ANE preload、bucket、fallback 和 backing 使用情况
  均进入 telemetry；
- 实验性 H3 encoder 与 LTX isolated MLP probes 不进入默认 native build。只有
  显式设置 `TURBOCIDER_BUILD_EXPERIMENTAL_PROBES=1` 才构建这些工具；
- exporter、sweep 和 probe 保留在 `tools/`，不进入 App/CLI 发行包；模型权重、
  Core ML artifact 和本机临时报告不提交。

## 8. 后续优先级

1. 保持默认 GPU 和 manifest-gated optional ANE，不扩大自动设备/shape 范围；
2. 为 H3 exact-256 建立可控缓存、系统负载和 cold/resident 分层复验，确认收益
   是否可稳定复现；
3. LTX encoder 只有在新 topology 先证明可能超过 `1.10x` full-encoder 门槛时
   才扩展 8-layer bank；
4. LTX DiT 优先修复完整视频质量，再讨论自动 hybrid；
5. 图像 VAE 优先 allocator/cache 生命周期；视频 VAE 优先 tile、chunk、
   clean finalizer 和 staged residency；
6. 暂不新增统一 VAE ANE 路径，除非固定 shape、预加载、handoff 和完整请求对照
   同时证明质量通过且端到端收益稳定。

Transformer 拆分原理、micro 数据和 public/private ANE 对照见
[Transformer CPU / GPU / ANE 异构并行技术报告](transformer-heterogeneous-report.md)；
Core ML 生命周期细节见 [Core ML / ANE 启动开销](coreml-ane-startup.md)。
