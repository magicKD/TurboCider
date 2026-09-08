# TurboCider GPU / ANE 并行化方案

更新时间：2026-09-08

这份文档描述当前 `dev` 合并版已经实现或验证过的异构执行方式。它把“能调用 ANE”与“端到端值得自动启用”分开：只有同一请求的 GPU 基线、输出质量和完整 request wall 都有证据时，才会进入自动策略；否则只能显式选择并保持 fail-closed。

并行执行总览见 [parallel-acceleration.svg](parallel-acceleration.svg)。图中的 `fork → join` 是真实的依赖边界，不代表把整个模型复制到两套设备；权重、临时 buffer 和 Core ML session 仍由同一个 TurboCider Session 管理。

## 机器策略

| 机器 | 自动 GPU+ANE | 当前默认 | 说明 |
|---|---|---|---|
| Apple M4 Max / 64 GB | FLUX.2 Klein 4B：ANE `[0,6144)` + GPU attention/后缀；Z-Image Turbo 基础模型：ANE `[0,4096)` + GPU attention/后缀 | 已验证模型走 `auto`；其余走 GPU | FLUX 4B 最终 GPU warm 中位数 2.2663 s、GPU+ANE 1.6279 s（1.392×）；Z-Image GPU warm 中位数 36.3685 s、stock ComfyUI GPU warm 中位数 40.11 s；a4096 GPU+ANE warm 中位数 30.0014 s（1.212×）。 |
| Apple M4 Pro / 48 GB | FLUX.2 Klein 4B 的完整 9216-channel ANE MLP profile | FLUX 4B 可按 profile/自动策略使用；其他模型 GPU | 该路线通过旧的 5% 回归门槛，但不能把它描述成普遍加速；FLUX 4B 的历史 warm 混合中位数约比原始框架慢 0.89%。 |
| 其他 Apple Silicon 或内存不匹配 | 不自动启用 | GPU | 不把芯片名称相近当作同一性能 profile；缺 manifest、shape、checkpoint 或内存预算时回退 GPU。 |

## 各模型的执行图与状态

### FLUX.2 Klein 4B

每个 DiT block 在 attention 和 MLP 输入处分叉：Core ML/ANE 计算 MLP 前缀，Metal GPU 同时计算 attention 以及剩余 MLP 后缀，最后在 residual join 汇合。M4 Max 使用 `[0,6144)` 前缀和 GPU `[6144,9216)` 后缀；M4 Pro profile 使用完整 MLP ANE 路线。

2026-09-07 的最终同入口持久会话复测（512×512、4 steps、seed 42）为：TurboCider GPU warm `2.2663 s`，当前 direct MLX engine warm `2.2642 s`，差约 `0.09%`；TurboCider GPU+ANE a6144 warm `1.6279 s`，相对 TurboCider GPU 为 `1.392×`。GPU+ANE 的 PNG correlation 为 `0.999262`、cosine 为 `0.999840`，Core ML output copy 为 0。纯 GPU 默认启用 compiled single-block、单 dispatch Q/K RoPE 和 step 级同步；MLX SDPA 保持默认 heuristic，强制 fused 与 heuristic 的差异约 0.2%，因此只保留为 profiling 开关。与 ComfyUI 的新鲜 FLUX 4B 对照本轮未宣称：本机 ComfyUI 0.32.0 没有同一 4B 权重/工作流可复现；因此 FLUX 的门禁采用同一请求 direct engine 对照，Z-Image 则有 stock ComfyUI 对照。

### Z-Image Turbo

32 个 S3-DiT block 保持串行 block 顺序。每个 block 先在 GPU 完成 modulation、attention 和 attention residual，物化共同的 FFN 输入后才分叉：gated FFN `[0,4096)` 前缀交给 ANE，GPU 异步计算 `[4096,10240)` 后缀，然后 join 回 residual。当前并行范围是 FFN intermediate-channel split，不是 GPU attention 与 ANE FFN 的重叠。纯 GPU 侧使用 MLX fused RMSNorm、单 dispatch Q/K RoPE、fused SDPA 与编译 block 图；混合路径的 GPU MLP 后缀也使用缓存的 `mx::compile` complement 图。Core ML 输出使用 session-wide shared backing，当前实测没有 output copy。

2026-09-07 的最终同条件复核（1024×1024、9 steps、seed 42）为：TurboCider GPU 首轮 39.4728 s，warm 36.3600/36.3769 s，中位数 `36.3685 s`；stock ComfyUI 0.32.0、custom nodes disabled 的 warm 中位数为 40.11 s，TurboCider GPU 快 `1.103×`。a4096 GPU+ANE 首轮 37.5838 s，warm 29.9956/30.0072 s，中位数 `30.0014 s`，相对优化 GPU `1.212×`；Core ML 每请求 288 次、output copy 0、GPU↔ANE PNG correlation `0.999231`、cosine `0.999903`。因此 M4 Max 基础模型继续允许自动 GPU+ANE；LoRA-bound 路线仍必须显式指定绑定同一 adapter 的 manifest，不能用基础 artifact 冒充稳定的自动 LoRA profile。

2026-09-08 新增 ConvRot 原生候选。GPU 把 signed tensor-wise INT8 映射成 packed affine Q8，并在投影前执行 grouped H256；Core ML 原生模式保留 rotated weights，在图内对 FFN 输入和 `w2` 输入执行 grouped H256。ConvRot GPU warm 中位数 `40.7288 s`，native a4096 GPU+ANE 为 `32.5979 s`，相对匹配 GPU `1.249×`；PNG correlation `0.999659`、cosine `0.999957`，active MLX 约 6.56 GB。但它仍比已有 BF16-derived a4096 的 `30.0014 s` 慢 `8.65%`，因此保持显式低内存候选，不替换全局最优自动路径。

### MiniMax H3 Turbo

H3 的 native runtime 以 Metal/MPS 为默认路径；在显式 manifest 存在时，block MLP/QKV 可按已验证的 Core ML 分区执行，GPU 负责其余算子并在 block join 汇合。H3 的 manifest、LoRA identity、取消和媒体生命周期已接入统一 Session，但当前没有足够跨请求/跨输入的稳定端到端 ANE 中位数，所以 App/API 不自动选择它。

### LLaDA-Image-Turbo

正式 LLaDA descriptor 当前只公开自包含的原生 C++/MLX `image.generate`。早期常驻 PyTorch/MPS worker 及单参考图 `image.edit` 只保留为显式开发诊断入口，不再自动发现兄弟 reference source，也不随正式 package 分发；它的历史性能不能代表当前发行能力。

现已增加显式 GPU+ANE 实验路径：2 个 `noise_refiner` 和 30 个主 `layers` 的 FFN 前缀 `[0,4096)` 调用 checkpoint-bound Core ML，原生 C++/MLX 计算后缀 `[4096,10240)`；context/sigvq refiner 保持 MLX。C ABI 的 FP16 输入边界已与直接 `.mlpackage` prediction 做 bitwise parity（MAE 0），正式路径不依赖 Python worker；旧 worker 仅保留为显式开发诊断工具。

旧 Python worker 的 256²测量（GPU+ANE 约 `0.873×`、图片 correlation `0.705–0.722`）不代表当前发行 executor，已从自动能力结论中移除。当前原生 C++/MLX 1024²、4 steps、resident 复测为 GPU warm `17.971 s`、GPU+ANE warm `14.934 s`，speedup `1.203×`，PNG correlation `0.995194`、cosine `0.998969`、MAE `2.649/255`。它满足本轮“可接受近似 + 至少 1.2×”的单机候选门槛，但仍只允许显式 `gpu_ane`；多机器、多 seed 和自动质量门禁尚未完成。

### LTX 2.5 Distilled

LTX 不是简单复制 FLUX 的 FFN split：它有 text/KV、video/audio 双流和 8+3 两阶段 schedule。已接入的候选 profile 让 GPU attention 与 Core ML MLP/KV/QKV 分区并行，并用两个 AV queue 交错 Stage 1/Stage 2；`preload_stage2`、`detach_stage1/2`、释放完整 GPU MLP 和 conditioning cache 都属于生命周期的一部分。当前 public executor 的 GPU+ANE 仍是显式候选，因为完整请求的 setup、VAE、mux 和音频边界还没有同时通过 parity 与稳定性能门禁；不能把 fixed-conditioning hot-path 的 1.196–1.266× 直接写成点击生成到 MP4 的保证。

### FastMetal 1.3B QAD

FastMetal 保留受控 Python/MLX worker，30 个 INT8 FFN block 固定按 4096/4864 intermediate channels 切分：MLX GPU 计算 residual/suffix，Core ML bridge 计算 ANE prefix，block join 后进入下一层。固定 832×480、81 帧、3 steps 的 base 路线已有逐元素 latent 与 MP4 parity；历史同热端到端约 69.97 s 对 72.93 s（约 1.04×），当前复测仍以不超过 5% 回归为门槛。LoRA 需要 provenance-verified premerged checkpoint，不能把任意独立 LoRA 动态塞进 ANE artifact。

### FLUX.2 Klein 9B 与未验证机器

9B 当前只开放 native Metal/MLX GPU；没有经过完整 ANE partition、质量和性能门禁的 profile。所有未匹配 M4 Max/M4 Pro 的设备也使用同一 fail-closed GPU fallback。这样可以保证 GPU 版本不会因为误选不适合的 Core ML shape 而变慢或改变模型语义。

## 统一判定规则

1. GPU+ANE 必须有显式 manifest，manifest 绑定 checkpoint、shape、分区范围、artifact 文件和设备 profile。
2. 带 LoRA 时，manifest 还必须绑定 adapter 的 canonical path、bytes、SHA-256、role 和 strength；否则 App 先切 GPU，native 仍会拒绝不匹配 artifact。
3. “稳定提升”按同一 workload 的完整 request wall 判断，不只看 denoise kernel；至少要有重复 warm 样本，并记录最慢样本的波动。
4. 纯 GPU 路径必须与原始框架或 direct baseline 做同条件比较；当前最终 FLUX 4B GPU 与 direct engine 差约 0.09%，H3/FastMetal 仍在既有门槛内，LTX GPU worker 热路径基本持平。Z-Image 优化后 TurboCider GPU warm 中位数 36.3685 s，stock ComfyUI warm 中位数 40.11 s，已通过“不慢于 ComfyUI”门槛。
5. 任何 shape、内存、artifact 完整性或已校准质量资格不匹配都回退 GPU（自动模式）；显式 `gpu_ane` 对运行时契约错误直接报错，并继续标记为实验路径，不静默裁剪 token、不减少 steps。离线 RGB/感知门禁不会在每次正式推理时重复生成参考结果。

## 证据索引

- `docs/design/validation/z-image-m4max-hybrid-lora-2026-09-07.json`：Z-Image LoRA-bound 与 base 分区数据。
- `docs/design/validation/z-image-convrot-native-ane-2026-09-08.json`：ConvRot packed Q8、原生 ANE、LoRA 内存和 parity 数据。
- `docs/design/validation/z-image-auto-m4max-2026-09-07.json`：最终构建的自动 4096-channel 重复 warm 复核。
- `docs/design/validation/z-image-gpu-comfy-2026-09-07.json`：TurboCider 优化 GPU、stock ComfyUI GPU 与 GPU↔ANE parity 摘要。
- `docs/design/validation/flux2-m4max-2026-09-07.json`：FLUX 4B 最终 GPU/direct、GPU+ANE、融合开关 A/B 与 parity 摘要。
- `docs/archive/control-plane/PERFORMANCE.md`：FLUX、H3、LTX、FastMetal 的历史 AB/BA 与阶段口径。
- `docs/design/flux-performance-comparison.md`：FLUX 原始框架与 TurboCider GPU/混合对照。
- `docs/status/implementation-status-2026-09-06.md`：当前实现边界、LoRA 和未完成项。
