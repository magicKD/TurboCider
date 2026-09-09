# Z-Image Turbo GGUF 状态

更新时间：2026-09-09（文件名保留以维持既有链接）

## 已完成

- 新增 `z-image-turbo-gguf` 注册模块，保留原有 safetensors `z-image-turbo` 路径不变。
- 支持目录或单文件输入；目录内可放多个 `.gguf`，`model_variant` 可按文件名/量化标签选择。
- 添加 GGUF magic/version 检查，避免把任意文件当作模型启动。
- 支持独立 VAE `ae.safetensors` 和独立 Qwen3-4B `qwen_3_4b.safetensors`；GGUF 与组件可以位于不同目录，benchmark 使用 `--component-root`，运行时也支持 `TURBOCIDER_Z_IMAGE_VAE`、`TURBOCIDER_Z_IMAGE_LLM` 覆盖路径。
- 通过受管、固定 SHA-256 的 macOS arm64 `sd-server` 实现常驻 GPU Metal 推理，避免每个请求重复加载约 5 GB GGUF。
- 支持独立 LoRA 文件，请求期加载，不生成 merged checkpoint；官方 LoRA 已完成与 direct sd-server 的 ABBA×2 像素/性能严格对照。
- `uses_parent_mlx=false`，外部 native server 仍受 TurboCider GPU lease 约束，但不污染父进程 MLX allocator。
- Q3_K_S、Q4_K_M、Q8_0 三个量化文件已经下载并真实生成；其余量化标签只完成选择/报告兼容，不能写成已验收。
- 直接 Unsloth sd-server 与 TurboCider 同时常驻、交错请求的性能/像素对照已经完成。
- build、47 项 contract、9 项 repository/boundary 测试通过，`git diff --check` 通过。

## Native MLX Q8_0 GPU+ANE 追加验证

同一 `z-image-turbo-Q8_0.gguf`、同一 Qwen3/VAE、同一 prompt/seed/1024²/9-step 请求下，32 个 Q8_0 Core ML 分区已经编译并接入 native MLX GGUF transformer。编译产物位于被忽略的本机目录，不进入 Git。

| 路径 | 首轮 request wall | warm request wall | warm denoise | 结果 |
|---|---:|---:|---:|---|
| native MLX Q8 GPU | 52.3429 s | 45.9527/45.9741 s，中位数 45.9634 s | 45.0422 s | 有效 1024² PNG |
| native MLX Q8 GPU+ANE | 49.0650 s | 37.6200/37.5497 s，中位数 37.5848 s | 36.6789 s | 有效 1024² PNG |

GPU+ANE 相对同一 native GPU session 的 resident warm 加速为 `1.2229×`。每次请求执行 32 个 block × 9 steps = 288 次 Core ML 调用；checkpoint SHA-256 校验通过，`output_copy_bytes_session_total=0`。GPU 与 GPU+ANE PNG 的 correlation 为 `0.9991566`、cosine 为 `0.9998930`，MAE 为 `0.9757/255`。这证明 Q8_0 的 GPU+ANE 路径可以启动、出图且不慢于同 checkpoint 的纯 GPU；但公开 Core ML API 不能直接证明每次调用都实际落在 ANE，报告中的 `compute_units=CPUAndNeuralEngine` 和 `observed_ane_residency=unknown` 必须保留。

去机器路径的证据见 [`z-image-gguf-native-q8-hybrid-2026-09-07.json`](../design/validation/z-image-gguf-native-q8-hybrid-2026-09-07.json)。

## 真实对照证据

模型：`z-image-turbo-Q4_K_M.gguf`，ModelScope 文件大小 `5,017,613,376` bytes，SHA-256 `e6494f87de6abaf6a561924f50317a5f271fc34bb4222aabbd801197df8f7daa`。

在当前 Apple Silicon 机器上，9 steps、seed 42：

| Variant/workload | direct median | TurboCider median | ratio | parity |
|---|---:|---:|---:|---|
| Q4_K_M 256²，ABBA×2 | 16.3675 s | 16.3467 s | 0.99872 | 4 组逐像素相同 |
| Q4_K_M 1024²，ABBA | 179.7119 s | 179.6199 s | 0.99949 | 2 组逐像素相同 |
| Q3_K_S 256² | 16.3700 s | 16.3493 s | 0.99874 | 逐像素相同 |
| Q8_0 256² | 15.9540 s | 15.9074 s | 0.99708 | 逐像素相同 |
| Q4_K_M + 官方 LoRA 256²，ABBA×2 | 13.1240 s | 13.1163 s | 0.99942 | 4 组逐像素相同 |

基础 Q3/Q4/Q8 与当前 Q4_K_M 官方 LoRA 样本均通过严格 `ratio <= 1.0` 的“不慢于 direct Unsloth runtime”门槛。LoRA 首次对照曾得到 23.5064/23.3321 s（ratio 1.00747），随后发现 TurboCider 内部 sd-server 漏传蒸馏 Z-Image 所需的 `--cfg-scale 1.0`，而 direct 侧已经使用该参数；这同时改变了工作负载和输出，旧数据只保留为配置缺陷诊断，不能作为性能结论。修复后两侧还统一为 100 ms polling，ABBA×2 的四组输出逐像素相同，TurboCider 中位数略低于 direct。两边使用同一固定版本 sd-server、模型、参数和 Metal flags；这证明桥接开销没有拖慢已验证 GGUF，不代表 sd.cpp Q4 比 TurboCider native BF16 快。后者 1024² GPU warm 约 36.37 s，而 Q4 GGUF base 约 179.62 s。

修复后的去机器路径证据见 [`z-image-gguf-q4km-lora-cfg1-2026-09-08.json`](../design/validation/z-image-gguf-q4km-lora-cfg1-2026-09-08.json)。

## 尚未完成

### 低内存 streaming/offload（2026-09-09 修正）

正式请求契约现支持 `residency: "streaming"` 或
`streaming_offload: true` 配合 `memory_budget_bytes`。运行时使用 pinned
sd.cpp 的 `diffusion=cpu,te=disk,vae=disk` parameter backend、mmap、真实 layer
prefetch/evict、max-vram 和 VAE tiling；native MLX/ANE 与 `in_memory_merge` LoRA
会明确拒绝，避免把全量常驻误报为低内存。旧的全 disk backend 会让 sd.cpp
忽略 `--stream-layers`，因此只保留为历史数据，不能再描述为有效 layer streaming。

Q3_K_S、Q4_K_M、Q8_0/256²/9 steps/8 GiB 已在 M4 Max 通过重复 ABBA×2
真实启动并出图。四个 measured warm 样本中，streaming 相对 resident 分别慢
5.02%、5.23%、14.32%，child lifetime physical footprint 分别降低 33.3%、
37.7%、45.9%，十二组 decoded RGB 全部逐像素相同。8 GiB 是 sd.cpp 的
`--max-vram` working-set hint，实际 child physical footprint 约 10.00–10.01 GB，
不是总进程硬上限。

Q4_K_M 1024²、8 GiB hint 的重复对照慢 14.14%，footprint 降低 35.2%；输出
并非 pixel exact，但 correlation 0.998117、cosine 0.999823、MAE 2.147/255，
通过显式近似质量门禁。16 GiB 单次 probe 仍慢 11.43%。Q4 官方独立 LoRA 的
256²对照保持 4/4 pixel exact，footprint 降低 37.6%，但慢 3.86%。所有路线均
未通过当前 1.02 performance gate，因此 streaming 是正确的显式低内存 fallback，
不是无代价优化；1024²多 seed、1024² LoRA、其它量化和真实低内存机器仍需补矩阵。
完整证据见
[`z-image-gguf-streaming-2026-09-09.json`](../design/validation/z-image-gguf-streaming-2026-09-09.json)。

1. GGUF GPU+ANE 已对 Q8_0 native MLX 路径完成真实候选验收；仍需把它从显式候选收口为经过设备/LoRA/多轮质量矩阵的自动策略，并对 Q4_0/Q4_1、F16/BF16/F32 和其他量化补齐矩阵。外部 sd-server 仍无法提供同样的 ANE fork/join，因此 mixed K-quant 继续使用 sd.cpp Metal。
2. 对尚未下载的 Q2、其余 Q3/Q4/Q5/Q6/IQ/F16/BF16/F32 变体补齐 load、出图、质量和性能矩阵；当前只实测 Q3_K_S、Q4_K_M、Q8_0。
3. 官方 LoRA 的 256² ABBA×2 已通过严格 `ratio <= 1.0`；仍需补 1024²、多 seed 和更长重复矩阵。
4. LLaDA-Image-Turbo 文生图已切换到原生 C++/MLX，1024² GPU+ANE 单机 warm 达到 1.203×；仍缺多机器/多 seed 自动门禁、LoRA 和原生 image.edit。
5. LoRA 的 `disk_premerge`、`in_memory_merge`、`inference_time` 三种策略已经统一进入请求、descriptor、plan/result、Swift 和 Studio；仍缺跨模型多 adapter、失败、取消与缓存失效完整矩阵。
6. `Comfy-Org/z_image_turbo` 的 `z_image_turbo_int8_convrot.safetensors` 不应与 GGUF Q4_K_M 混称；其原生 a4096 GPU+ANE 已达到相对 ConvRot GPU 的 1.249×，但仍比 BF16-derived a4096 慢 8.65%。
7. Q8_0 独立 LoRA 已完成 native MLX GPU 和 LoRA-bound Core ML 启动，但 GPU+ANE warm E2E 比 GPU 慢；仍需与 sd.cpp request-time reference 做统一 RNG/质量口径和继续优化混合路径。
