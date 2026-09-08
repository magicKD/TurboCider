# TurboCider 当前状态与未完成项

> 此快照已由 [2026-09-08 当前目标状态](current-status-2026-09-08.md)取代。

更新时间：2026-09-07

验证机器：Apple M4 Max，64 GB unified memory，MLX 0.32.2

## 结论

当前 `dev` 是一个可独立编译和启动、并提供七个 executable model descriptor/执行路径的开发验收版本；新增的第七条是 `z-image-turbo-gguf`。mixed K-quant 使用常驻 sd.cpp Metal，Q8_0/Q4_0/Q4_1 可进入 native MLX；Q8_0 已完成 checkpoint-bound GPU+ANE 候选实测。实际生成仍要求相应模型资产和各模型列出的系统/可选依赖。FLUX.2 Klein 4B 和 safetensors Z-Image Turbo 的 native GPU 路径已经吸收 fused RMSNorm、Metal Q/K RoPE、MLX SDPA 和 compiled block 等优化；M4 Max 64 GB 的 FLUX a6144、Z-Image base a4096 与 GGUF Q8 a4096 GPU+ANE 路径都有显式 checkpoint/shape/manifest 门禁。

当前版本适合整理后提交为多模型 native integration / FLUX-Z-Image optimization milestone，但还不是“所有模型生产完成版”。GGUF Q8_0 base native GPU+ANE 已完成 1024² resident warm 验证；Q8 LoRA-bound 已完成导出、编译和真实启动，但其 1024² E2E warm 未通过不慢门禁。仍未完成的是 LTX 完整端到端 GPU+ANE、H3/LTX/FastMetal 纯内存 LoRA、FLUX 9B 的正式矩阵，以及 LoRA-bound 混合路径的优化和自动策略。

## 当前代码和仓库卫生

- `main` 的 C++ core/runtime 已合入 `dev`；模型实现位于 TurboCider 自己的 `native/`，运行时不从 `h3.c`、`ltx-mac` 或 `references/` 加载源码。
- 2026-09-07 已继续合入 `origin/main@7e84bbe` 的 task-aware acceleration、资源监控、磁盘 inventory 和实际路径显示；冲突处保留了 `dev` 的六模型、独立 LoRA、FLUX a6144 和 Z-Image a4096 实现。
- 通用模型数学使用 C++/MLX，Apple bridge 使用 Objective-C++/Metal/Core ML，App 使用 Swift；没有要求把所有功能重写成 Objective-C。
- `.gitignore` 已排除 `build/`、`dist/`、`models/`、`outputs/`、`artifacts/`、`Python/`、虚拟环境、Core ML 编译产物、缓存和本机研究记录。
- 当前 `git status` 中没有模型权重、生成图片/视频、构建对象、Python 环境或 Core ML artifact；未跟踪项均是本轮计划进入版本的正式 Markdown、SVG 和去路径化 validation JSON。
- `git diff --check` 无 whitespace 错误。源码中没有兄弟仓库运行路径。后续 2026-09-08 收口进一步把 private ANE bridge 从正式 `native/` 删除，仅在 `experimental/` 保留研究快照；本段其余结论保留为当日历史记录。

本机物理磁盘仍接近满载，当前可用空间约 8.3 GiB。被忽略的本地数据约包括：`models/` 106 GB、`outputs/` 3.0 GB、`Python/` 1.2 GB、`build/` 340 MB、`dist/` 401 MB。它们没有进入 Git，因此不影响提交卫生；若需要释放空间，优先候选是可重建的旧 `outputs/`、`build/` 和 `dist/`。模型与 Core ML artifact 仍是后续复测依据，本轮没有擅自删除。

## 模型实现状态

| 模型 | 当前可运行路径 | LoRA 状态 | 当前门禁/缺口 |
|---|---|---|---|
| FLUX.2 Klein 4B | native GPU；M4 Max a6144 GPU+ANE | 独立文件在 MLX 权重内存中融合；不写 merged checkpoint；LoRA-bound ANE manifest 可显式使用 | LoRA GPU+ANE 尚无优化后重复性能矩阵；多尺寸/多机器 AB/BA 待补 |
| FLUX.2 Klein 9B | native GPU | 与 4B 共用 load-time in-memory delta | 标准尺寸 parity、warm/resident 和 GPU+ANE 未完成 |
| Z-Image Turbo | Comfy split-files 或 Tongyi diffusers shards；native GPU；M4 Max base a4096 GPU+ANE | 独立 LoRA 在 MLX 内存中融合；官方 patch 的 238 个 projection 全部应用 | base ANE 为自动路线；LoRA-bound ANE 仍显式；逐 step oracle 待补 |
| Z-Image Turbo GGUF | Q3_K_S/Q4_K_M/Q8_0 已真实运行；Q8_0 native MLX GPU 与 GPU+ANE；mixed K-quant 常驻 sd.cpp Metal GPU | 独立 LoRA 已支持：sd.cpp 请求期加载、native MLX 内存 delta、Q8 LoRA-bound Core ML 可显式启动 | Q8 base GPU+ANE 1024² warm 37.5848 s，相对 native GPU 45.9634 s 为 1.223×；Q8 LoRA-bound E2E warm 65.02 s 对 GPU 55.76 s，未过不慢门禁，仍显式 |
| MiniMax H3 Turbo | native Metal/MPS；manifest-gated ANE | 独立文件接口已接入，但首次使用生成 repository-local Python 工具构造的可清理 runtime cache | 尚未做到逐层纯内存 LoRA；ANE 多轮矩阵待补 |
| LTX 2.5 Distilled | video-only public native 路径；GPU+ANE candidate | 独立文件接口已接入，但 refiner 仍通过 repository-local runtime bake/cache | 完整端到端 GPU+ANE 尚未过速度/数值门禁；I2V/音频 Session parity 未完成 |
| FastMetal 1.3B QAD | 受控持久 Python/MLX worker；固定 shape GPU/GPU+ANE | 只接受 provenance-verified premerged checkpoint | 独立 LoRA runtime bake 和对应 ANE artifact 未完成 |

“不依赖外部仓库”已经满足源码边界，但不等于没有任何运行依赖：模型权重由用户提供；Metal/Core ML/Foundation/ImageIO 是系统框架；MLX 随发行包；H3 媒体输入/输出仍可调用 `ffmpeg`/`ffprobe`；FastMetal 有意保留受控 Python worker；H3/LTX LoRA 仍需要 TurboCider 自带的 Python merge 工具和可清理缓存。

## FLUX.2 Klein 4B 性能

工作负载：512×512、4 steps、seed 42，同一持久 C ABI 入口。

| 路径 | Warm 中位数 | 结果 |
|---|---:|---|
| 当前 direct MLX engine | 2.2642 s | 对照基线 |
| TurboCider compiled GPU | 2.2663 s | TurboCider/direct = 1.00094，约慢 0.09%，属于测量噪声 |
| TurboCider GPU+ANE a6144 | 1.6279 s | 相对 TurboCider GPU 为 1.392× |

最优 M4 Max 分区是 ANE MLP `[0,6144)` 与 GPU attention + MLP `[6144,9216)` 并行，20 blocks × 4 steps 共 80 次 Core ML 调用，output copy 为 0。GPU↔ANE PNG correlation 为 0.999262、cosine 为 0.999840。

本轮没有同一 FLUX.2 Klein 4B checkpoint/workflow 的 fresh ComfyUI 对照，因此只宣称“不慢于当前 direct engine”，不宣称新的 ComfyUI 比值。

## Z-Image Turbo 格式、LoRA 和性能

两种模型布局均已真实运行：

- Comfy-Org split-files：BF16 单文件 DiT、Qwen3、VAE 和独立 distill patch LoRA。
- Tongyi-MAI diffusers：三分片 transformer index、text encoder、VAE 和 tokenizer；不要求先合并成单一 checkpoint。

Tongyi 三分片目录完成 1024×1024、9-step 真实生成。相同 prompt/seed 的 Tongyi 与 Comfy BF16 输出 PNG correlation 为 0.999556、cosine 为 0.999943。Comfy Qwen3 和 VAE 使用硬链接复用 Tongyi 文件，没有重复占用对应的 8.04 GB 与 335 MB；运行时更推荐 12.31 GB 的 Comfy BF16 DiT，而不是 24.62 GB 的 Tongyi F32 transformer shards。

官方 `z_image_turbo_distill_patch_lora_bf16.safetensors` 保持为独立 158,826,336-byte 文件。当前真实 GPU 请求报告 `lora_fusion=in_memory_delta`、`lora_applied_projections=238`，成功输出 1024×1024 PNG；模型目录中没有生成 merged/baked checkpoint。

工作负载：1024×1024、9 steps、seed 42。

| 路径 | Warm 中位数 | 结果 |
|---|---:|---|
| stock ComfyUI 0.32.0 GPU | 40.110 s | custom nodes disabled 对照 |
| TurboCider compiled GPU | 36.3685 s | 比 ComfyUI 快 1.103×；比旧 TurboCider GPU 快 1.243× |
| TurboCider GPU+ANE a4096 | 30.0014 s | 相对优化 GPU 为 1.212×；当前最优稳定分区 |

Z-Image a4096 使用 bucket 4128、32 blocks、INT8 per-channel FFN prefix、activation scale 8、output scale 32；GPU 负责 attention 和 `[4096,10240)` FFN suffix。每请求 288 次 Core ML 调用，output copy 为 0。GPU↔ANE PNG correlation 为 0.999231、cosine 为 0.999903、MAE 为 0.879/255。

已排除的更大分区：当前构建下 a5120 的单次 warm 为约 30.432 s，a6144/bucket 4608 为约 36.323 s，均不优于 a4096。扩大 ANE prefix 会增加 Core ML latency，不能直接达到 1.3×。

独立 LoRA 的当前最稳妥默认仍是 compiled GPU。已有 LoRA-bound a4096 artifact 能通过 path/bytes/SHA-256/role/strength 校验并显式运行，但它尚未完成优化 GPU 基线下的重复 warm 验收，因此 App `auto` 不会为带 LoRA 的请求选择该路径。

## 其他模型性能现状

| 模型 | 当前证据 | 判断 |
|---|---|---|
| H3 | direct 42.621 s；TurboCider 42.963 s；MP4 byte-exact | 慢约 0.80%，通过 5% GPU 回归门槛 |
| FastMetal GPU | direct 72.929 s；TurboCider 72.937 s | 差约 0.01%，输出一致 |
| FastMetal GPU+ANE | direct/TurboCider 均约 69.970 s | 热计算路径一致；相对 GPU 仅约 1.04× |
| LTX full-chain 历史对照 | 旧 direct 97.573 s；TurboCider 96.952 s | 同入口约快 0.64%，但不是当前稳定多轮结论 |
| LTX public cache-hit | 完整请求 86.284 s；pre-finalizer 81.918 s；clean VAE 3.410 s | conditioning cache 有效，但完整生命周期尚未稳定超越 mac-ltx |
| LTX GPU+ANE candidate | dynamic Gemma 请求约 125.90 s；dense GPU 约 119.08 s；latent cosine 约 0.945 | 当前更慢且偏差过大，不能自动启用 |

mac-ltx 的 59.483 s 是 conditioning 已准备、模型已加载的 decoded-pixel 热路径，不包含 TurboCider public 请求的模型建立和 conditioning。扣除 TurboCider 约 22.75 s setup 后的估算热链路约 58.61 s，接近并略快于该数字，但口径不同且缺少重复样本，不能据此宣称 TurboCider 完整端到端已经更快。

## 自动 GPU/ANE 策略

- Apple M4 Max / 64 GB：FLUX 4B base 仅对 512×512 / 4 步 / 1088 桶的 a6144 案例自动选择；Z-Image base 仅对 1024×1024 / 9 步 / 4128 桶的 a4096 案例自动选择。
- Apple M4 Pro / 48 GB：FLUX 4B full-MLP 保留 512 / 1088 桶和 1024 / 4160 桶两个独立实测案例；不复用 M4 Max Z-Image 策略。
- 其他芯片、内存容量不匹配、缺失/损坏 artifact、LoRA identity 不匹配或超出 token bucket：`auto` 回退 GPU；显式 `gpu_ane` 直接报错。
- FLUX/Z-Image 带 LoRA：只有绑定同一 adapter identity 的 artifact 才能显式使用；Z-Image LoRA 不自动选择 ANE。

详细 fork/join 图见 [GPU/ANE 并行化方案](../design/parallel-acceleration.md)。

## 已通过的正确性门禁

- native CLI、`libturbocider.dylib`、Swift App 和 integration runner 构建成功。
- native self-test 在 Apple M4 Max Metal 设备上通过。
- 46 项 contract、9 项 repository/boundary、5 项 Z-Image shard、4 项 Core ML LoRA 和 2 项 inventory 测试通过；其中需要 NumPy 的 1 项数值测试在缺依赖的系统 Python 下明确跳过。
- App 行为测试通过；受限会话无 pasteboard service 时只跳过系统剪贴板检查。
- validation JSON 和 GPU/ANE SVG 可解析；`git diff --check` 通过。

## 尚未完成，按优先级排序

1. LTX 在相同 fresh/cache-hit/loaded-model/resident 口径下完成 TurboCider GPU、TurboCider GPU+ANE 与 mac-ltx 的重复 AB/BA；当前 hybrid 更慢且 latent parity 不合格。
2. 将 H3/LTX 的 LoRA 从 runtime merged cache 改成真正逐层 native in-memory delta；FastMetal 仍需独立 LoRA runtime bake。
3. Z-Image LoRA-bound a4096 已完成一次三轮 GPU/GPU+ANE 实跑和 provenance 校验，但 E2E warm 为 65.02 s 对 55.76 s，未通过前保持显式并继续优化。
4. Z-Image 若要达到 1.3×，需要减少 288 次 block 级 Core ML 边界/输入打包开销，或改变分区粒度；扩大到 a5120/a6144 已证明无效。
5. FLUX 4B 补齐多尺寸、多 prompt、多机器和交错 AB/BA p50/p95；FLUX 9B 补齐标准尺寸 parity、warm/resident 与 GPU+ANE。
6. Z-Image 完成逐 step latent oracle 和多机器矩阵；LTX I2V/音频完成公开 Session parity。
7. Q8 GGUF LoRA 的 reference 对照工具已固定使用 Z-Image Turbo 正确的 CFG=1.0；仍需补更稳定的多轮/多尺寸 LoRA E2E 统计，不能把 denoise 约 1.04× 写成端到端加速。
8. 发布前补齐 FastVideo/TAEHV 第三方 NOTICE、Developer ID 签名和公证；当前只验证本地 ad-hoc package。

原始大文件报告位于被忽略的 `outputs/`。可提交的去路径化摘要位于 `docs/design/validation/`。
