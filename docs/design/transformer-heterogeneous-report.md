# Transformer CPU / GPU / ANE 异构并行技术报告

更新时间：2026-09-08

验证主机：Apple M4 Max，64 GB unified memory，macOS 26.6.2。本文整合 TurboCider 当前 E2E 数据，以及 `gpu_ane/mac_transformer@9f322e1`、`gpu_ane/ANE@d91c984` 已完成的 fixed-shape、public Core ML 和 private ANE 实验。本文没有把 microbenchmark 冒充模型端到端结果。

## 结论

当前最有效、可发行的通用策略是：attention、softmax、RoPE 和长序列归约继续留在 Metal GPU；把 FFN 的成对 intermediate channels 分给 GPU 和 public Core ML/ANE，各分支完整执行 FC1、激活和 FC2，只在 hidden-size partial 上做一次 join。CPU shard 默认关闭。模型只有在 device、shape、checkpoint、LoRA、内存和质量门禁全部匹配时才能自动选择 GPU+ANE。

各并行轴的最终判断：

| 并行轴 | 代数可行性 | 实测结果 | TurboCider 决定 |
|---|---|---|---|
| FFN intermediate-channel tensor parallel | exact algebra；不同精度产生近似误差 | 最稳定，micro 可达 1.5–2.6× | 采用 |
| FLUX-style attention/MLP branch parallel | 两支同读 normalized input | FLUX 4B E2E 1.392× | 采用 |
| sequence/token-row split | 线性投影可行，完整 attention 需要大范围重排/归约 | H3 block 约 1.282×，all-50 约 1.009× | 拒绝默认 |
| head/QKV split | 必须按完整 head/output channels 切 | 去掉 concat 后仍受 ANE straggler 和数值敏感性限制 | 只保留研究 |
| public Core ML SDPA | 固定 shape 可表达 | H3 单 head 下界 291.1 ms，且不含 QKV/RoPE/output | 拒绝，使用 GPU fused/tiled attention |
| CPU tail | algebra 正确 | 512-row case 0.981×，CPU 成为 first-consumer straggler | 默认 CPU=0 |
| private ANE | 可降低 wrapper/handoff | H3 全模型只约 1.041×，且无稳定 ABI | 仅 `experimental/` |

## 1. 统一计时和正确性口径

异构实验必须分开报告三种时间：

1. hot operator/block：session、graph、buffer 和权重都已常驻；用于寻找切分比例；
2. fresh process：包含 executable、manifest、权重 I/O、Metal/Core ML runtime 和 wrapper binding；不等于首次 `.mlpackage` 编译；
3. E2E：包含 text encode、scheduler、全部 block、VAE、媒体输出以及必要的 load/warmup。

GPU/ANE 或量化路径不要求逐 bit/逐像素一致，但必须保持 shape、finite，并记录 correlation、cosine、MAE/relative-L2。`allow_approximation=true` 只是显式许可，不是质量通过证明。自动模式只允许已测设备和 workload；初始化失败回退 GPU，显式 `gpu_ane` 则直接报错。

## 2. 推荐的 FFN 分割

对于输入 `X[M,H]`、SwiGLU intermediate width `F`，按成对 gate/up channel 切成互不重叠集合：

```text
P_d = (SiLU(X Wg[:, I_d]) * (X Wu[:, I_d])) W2[I_d, :]
Y   = residual + P_gpu + P_ane (+ P_cpu)
```

每台设备都返回 `[M,H]` partial，而不是返回或拼接 `[M,F]` 激活。这样可以：

- 避免大 intermediate activation concat；
- 让 GPU 和 ANE 从同一输入真正 fork；
- 在一个 Metal epilogue 中完成 partial add、residual 和必要的下一层准备；
- 用 caller-owned Core ML output backing 避免默认 CPU bridge copy。

### 2.1 fixed-shape micro 数据

| Shape / plan | GPU p50 | GPU+ANE p50 | 加速 | 质量 |
|---|---:|---:|---:|---|
| `512x1024x4096` FP16，2048/2048 | 1.342 ms | 0.891 ms | 1.525× | cosine ≥ 0.9999929 |
| 同 shape，GPU 1024 / ANE INT8 3072 | 1.468 ms | 1.023 ms | 1.471× | cosine 0.9999965 |
| 同 shape，GPU/ANE/CPU 1792/1792/512 | 1.369 ms | 1.390 ms | 0.981× | CPU 约 1.20 ms，为 straggler |
| LTX Stage 1 `1001x4096x16384` FP16 | 258.817 ms | 147.544 ms | 1.754× | cosine 0.9999180 |
| LTX Stage 1 INT8 | 260.809 ms | 130.616 ms | 1.997× | cosine 0.9999020 |
| LTX Stage 2 `4004x4096x16384` FP16 | 971.685 ms | 564.292 ms | 1.722× | cosine 0.999999992 |
| LTX Stage 2 INT8 | 971.698 ms | 496.760 ms | 1.956× | cosine 0.9999841 |

LTX 的 join 只有约 1–2 ms，主瓶颈是 ANE branch 本身，而不是最后一次 add。继续拆 FC2 为多个 Core ML convolution 会增加 dispatch/reduction：H3 试验中 1 chunk 为 1.201×，2 chunk 降到 0.817×，4 chunk 只有 0.737×。

## 3. 不同 Transformer 拓扑的并行窗口

### 3.1 FLUX / Z-Image / LLaDA

FLUX-style block 的 attention 与 MLP 可同时读取同一个 normalized hidden：

```text
normalized X ──► GPU attention ──┐
             └► ANE/GPU FFN ────┴► hidden join
```

该拓扑允许把整个 MLP branch 隐藏在 GPU attention critical path 中。FLUX-like MLP micro 在 1088/4160 rows 上曾达到 2.391×/2.557×；模型 E2E 会受到 text encoder、attention 和 VAE 限制，因此当前正式结果为：

| 模型/workload | GPU warm | GPU+ANE warm | E2E 加速 | 质量 |
|---|---:|---:|---:|---|
| FLUX 4B，512²，4 steps | 2.2663 s | 1.6279 s | 1.392× | PNG corr 0.999262 |
| Z-Image BF16，1024²，9 steps | 36.3685 s | 30.0014 s | 1.212× | PNG corr 0.999231 |
| LLaDA，1024²，4 steps | 17.971 s | 14.934 s | 1.203× | corr 0.995194，MAE 2.649/255 |

FLUX 和 Z-Image 只对 exact M4 Max 64 GB profile 自动开放；LLaDA 仍需显式 `gpu_ane`，因为目前只有单机/单 seed 证据。

### 3.2 H3 / sequential pre-norm block

H3 的 MLP 输入依赖 attention residual，合法时间线只能是：

```text
GPU attention ─► max(GPU MLP shard, ANE MLP shard) ─► join
```

不能把 attention 与本 block MLP 虚假地画成并行。H3 real-block MLP micro 可达约 1.72×，但 full H3 还包含 50 个串行 block、session rotation、AdaLN、projection、VAE 和媒体输出。matched BF16 private MLP+QKV 的 fully-warm E2E 仅从 319.34 s 降到 306.85 s，即约 1.041×；与当前更快的 GPU INT8-QKV baseline 相比没有产品收益。

full-MLP sequence-row split 已经做到 suffix direct write，不再有 FC2 reduction、row copy 或 concat。block 0/20/40 micro 约 1.282–1.285×，但 all-50 最好 warm forward 只有约 1.009×。这证明 H3 的主要限制是依赖和生命周期，不是少一个 reshape 就能解决。

### 3.3 LTX video/audio 双流

LTX 额外暴露三类真实并行窗口：

1. Video self/text stream 与 Audio self/text stream；
2. A-to-V 与 V-to-A cross attention；
3. Video FFN 与 Audio FFN。

TurboCider 当前使用两个 GPU queue 交错音视频支路，并可让 Video FFN 使用 GPU+ANE channel split。48-block micro 中，Stage 1 从 3.064 s 降到 2.534 s（1.209×），Stage 2 从 10.182 s 降到 8.609 s（1.183×）。完整 fast-path 的 historical E2E 可到约 1.20×，但当前 TurboCider 真实完整请求仍未同时通过质量、启动、VAE 和 mux 门禁，所以不自动启用。

Video self-QKV sequence split 虽让 Stage-2 attention 从 62.148 ms 降到 52.431 ms，48-block 8+3 E2E 却是 54.848 s，慢于 dense GPU+ANE 的 49.467 s。原因是每 block session、load/switch、ANE queue 和 UMA contention 吃掉了局部收益。

LTX 的视频级近似也需要单独看，而不能只看 latent cosine。对同一 704×448、97 帧、24 fps、8+3 steps 的 GPU 与 GPU+ANE 输出，帧级门禁记录 mean correlation `0.8498`、minimum correlation `0.7751`、mean MAE `21.59/255`；虽然最大 frame-to-frame motion energy 相对误差为 `8.04%`，仍未通过 RGB 质量阈值。因此当前 LTX GPU+ANE 只能作为显式实验候选，不能自动替代 GPU。完整指标见 [视频质量门禁记录](validation/video-quality-gate-2026-09-08.json)。

## 4. Attention、QKV 和 sequence parallel 的负结果

QKV 只能按完整 head/output channel 切分。H3 原型已让 ANE prefix 与 GPU complement 直接写最终 Q/K/V head slot，并让 segmented norm/RoPE 原地消费，因此失败原因不是 concat 或 reshape。真实 864×480 FP16 16-head projection 中，GPU+ANE overlap 约 358 ms，完整 GPU 约 249 ms，pack+join 仅 3–4 ms；ANE projection 是 straggler。

public Core ML SDPA 对 H3 单 head `15405x128` 的 hot 下界约 291.1 ms，还未包含 QKV、RoPE 和 output projection。长序列 attention 继续使用 Metal fused/tiled kernel。sequence parallel 跨 GPU/ANE 还需要 sequence-to-head redistribution、全局 softmax/reduction 和回写，统一内存并不会让这些边界免费。

## 5. Core ML 启动、预加载和缓存

### 5.1 必须区分的阶段

TurboCider 分开记录：

- source `.mlpackage` compile 与内容寻址 `.mlmodelc` cache；
- manifest/checkpoint/LoRA provenance 验证；
- `MLModel` load；
- feature/interface 与 output backing 创建；
- zero-input warmup；
- first runtime prediction；
- subsequent prediction。

`prepare(load-only)` 会提前完成验证、权重和 session/backing 加载，但不 denoise、不 VAE decode、不写输出；`prepare(warmup)` 还会做无输出完整请求。它们能把用户可见首请求延迟移到后台准备阶段，不能减少总同步工作。

### 5.2 实测启动数据

`mac_transformer` 的已编译 `512x1024x4096` Core ML artifact，fresh-process p50 为 92.378 ms，hot MLP 只有 1.353 ms，setup proxy 约 91.025 ms。one-shot 小算子无法摊销这笔成本，session 必须跨 block、step 或请求复用。

Z-Image 256²、32 个 Core ML block 的一次真实拆分：

| 路径 | prepare | 随后 generate | 说明 |
|---|---:|---:|---|
| cold generate | — | 17.255 s | 包含首次构造/加载 |
| load-only + generate | 12.792 s | 11.201 s | 把约 6.05 s 从首请求移到 prepare |
| zero-input + full warmup + generate | 18.686 s | 10.895 s | 后续比 load-only 再快约 2.7%，但准备更贵 |

load-only 中 manifest/provenance 约 5.25 s，32 个 model load 合计约 6.88 s，output backing 约 4 ms；同一有序样本里 model load 也曾因 OS/Core ML cache 命中降至约 0.24 s。因此不能把单次 load 数字写成稳定编译耗时。

### 5.3 当前优化原则

- App/服务可在用户选择模型或编辑 prompt 时异步执行 load-only；
- 只有明确会连续生成时才做完整 warmup；
- session 以 checkpoint、LoRA identity、shape/bucket、precision 和 device profile 为 key；
- stage-scoped residency 优于把几十个大 program 永久全驻留；
- caller-owned `outputBackings` 必须验证对象和 pointer identity，避免隐藏 CPU copy；
- provenance cache 必须绑定文件 size、mtime/content hash，不能用跳过校验换启动速度。

## 6. Public Core ML 与 private ANE

public Core ML 的 `outputBackings` 已能避免默认 CPU bridge。`2048x512` Metal consumer 从默认 bridge 0.807 ms 降至 backing-direct 0.424 ms，约 1.929×；这仍是同步 prediction barrier，不是自动 GPU/ANE resource hazard tracking。

private IOSurface handoff 在相同 shape 上可进一步到 0.303 ms，约比 public direct 快 1.398×；真实 SmolLM2 up projection p50 为 0.611 ms 对 public 0.796 ms，约 1.302×。但 H3 整体只约 1.041×，而且 private API 没有系统兼容保证。因此：

- private 实现只保留在 `experimental/`；
- 正式构建删除 `_ANE*` bridge，并链接 fail-closed stub；
- 产品 GPU+ANE 只使用 public Core ML；
- private 数据只用于判断 public API 的潜在调度上限，不作为产品性能承诺。

隔离细节见 [Private ANE 实验说明](../../experimental/private-ane/README.md)。

## 7. 与 vpipe 的低内存策略对照

vpipe 的优势不是单个“神奇 kernel”，而是 custom Metal forward、模型准备期 4-bit/8-bit 转换、按真实 tensor table 计算 trunk/每 block/activation floor、双 slot reusable buffer、`pread` 直接填充目标 buffer，以及计算当前 block 时预取下一 block。其文档中的 H3 matched 结果使用 960×544、124 frames、6 DiT steps，并来自 M4 Pro/M5/M5 Pro；不能和 TurboCider 当前 M4 Max 四步 H3 数据直接做速度比。

TurboCider 当前对应能力：

| 能力 | vpipe | TurboCider 当前状态 |
|---|---|---|
| 自定义 GPU forward | custom Metal | H3/LTX custom Metal；图像模型主要 MLX/Metal |
| 量化模型 | 准备期 4-bit/8-bit | Z-Image GGUF 多 K-quant + native Q8；H3 运行期 INT8 kernel，但 streamed H3 仍 BF16 |
| block streaming | 动态 residency + reusable refill | H3 BF16 双 slot + background `pread`；Z-Image GGUF 由 pinned sd.cpp `--stream-layers` |
| 低内存 LTX | plugin 支持 16 GB | TurboCider 只有 `component_staged`，尚无 LTX per-block streaming |
| 自适应 pinning | 基于 trunk、真实 block bytes、scratch 和 RAM | TurboCider H3 目前是 resident/streamed 二选一，缺动态 pinned prefix |
| offload 质量 | 同模型专用准备 | GGUF Q3_K_S 256² streaming 与 resident decoded RGB 完全一致 |

Z-Image Q3_K_S 256²实测中，streaming physical footprint 从约 14.91 GB 降到 9.52 GB，降低 36.1%；单个 warm 样本为 9.995 s 对 resident 9.534 s，慢约 4.8%。这说明当前 low-memory 路径已经可用，但还不能宣称“不影响性能”，因为只有每路线一个 warm 样本且没有 1024²矩阵。

下一步最有价值的 vpipe-style 改进是：

1. 给 H3 加入基于真实 block bytes、activation scratch 和内存压力的动态 pinned prefix，而不是全 resident/全 streamed；
2. 让 H3 streaming 支持量化 block payload，避免目前 streaming 与 INT8 MLP/QKV 互斥；
3. 给 LTX 实现 stage-aware 双 slot per-block refill，并把 4.6 GB 级 non-block trunk、text connector 和 VAE 峰值纳入预算；
4. 用 `pread` 到已分配 Metal buffer，保留少数不可 raw-copy tensor 的原加载路径；
5. 用真实 E2E ABBA 比较 resident、component-staged、streamed、pinned-prefix，而不是只测 SSD GB/s。

## 8. 模型级采用矩阵

| 模型 | GPU | GPU+ANE | 低内存 | 当前产品状态 |
|---|---|---|---|---|
| FLUX 4B | compiled MLX/Metal | M4 Max a6144 / M4 Pro validated profile | component release | exact profile 可自动 |
| FLUX 9B | GPU-only | 未验证 | staged text | 不自动 ANE |
| Z-Image BF16 | compiled MLX/Metal | a4096 | component release | 当前全局最优图像路线 |
| Z-Image ConvRot | packed Q8 | native ConvRot a4096 | active MLX 约 6.5 GB | 显式候选 |
| Z-Image GGUF | sd.cpp Metal / native Q8 | native Q8 a4096 | disk/mmap/stream-layers | mixed K-quant 留 sd.cpp |
| LLaDA | native C++/MLX | a4096 prefix | large-image staged text | hybrid 仍显式 |
| H3 | custom Metal/MPS | public Core ML MLP/QKV | BF16 SSD double-buffer streaming | hybrid/streaming 仍需 E2E 矩阵 |
| LTX | custom Metal/MLX helper | public Core ML MLP/KV/QKV + dual GPU queue | component-staged | 缺 per-block streaming 与完整 hybrid 质量门禁 |

## 9. 尚未完成的验证

- Core ML cold process、disk cache hit、load、interface、zero warmup、first/subsequent prediction 的多轮 ABBA 矩阵；
- H3 与 LTX 在同一机器、同一输入、同一输出边界下对 vpipe 的 matched E2E；
- H3 动态 pinned-prefix 与量化 streaming；
- LTX per-block streaming、16/24/32 GB 低内存验收；
- LLaDA hybrid 的多机器、多 seed 自动门禁；
- GGUF Q2/Q3/Q4/Q5/Q6/IQ/F16/BF16/F32 的多尺寸、LoRA 和 1024² streaming；
- private procedure-bank 只可继续研究，不能改变正式发行边界。

结构化摘要见 [Transformer 异构验证记录](validation/transformer-heterogeneous-2026-09-08.json)。
