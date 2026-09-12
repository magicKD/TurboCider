# TurboCider H3 C++/MLX INT6 加速方案与验收计划

**状态：设计方案 + 实现快照（2026-09-12）**
**目标：**在 TurboCider 内增加一个 C++/MLX 原生的 MiniMax H3 FastH3 路径，首发固定为 T2VA（文本生成带音频视频，对外仍使用 `video.generate`）、4 steps、INT6，并在同一台 Apple Silicon Mac、同一模型/提示词/seed/分辨率/帧数/解码器条件下，性能不明显落后于 FastVideo 的 MLX 实现，质量接近现有 H3 C/Metal 基线和 FastVideo。

本文是实施合同，同时记录当前实现证据；它不把尚未完成的 ABBA 性能和完整媒体质量门禁标记为交付。TurboCider 现有 H3 C/Metal runtime 仍是默认路径；FastH3 MLX 是显式选择的候选 profile，不能通过环境变量静默替换默认行为。

## 0. 当前仓库快照与完成边界

截至 2026-09-12，工作区中已经有可编译的 H3 MLX 组件、真实 ModelScope 权重、候选 native profile、端到端 124 帧执行证据和一组 parity/契约测试。组件级数学、共享 conditioning/noise 下的最终 DiT latent、conditioner 严格逐元素 parity 和真实 prompt→MP4 链路已经闭环；单次冷/热性能与 FastVideo 接近，但正式多样本 ABBA median 和产品感知质量门禁尚未完成：

- 已有 `native/models/h3_mlx/geometry.*`、`dit.*`、`checkpoint.*`、`pipeline.*`：包含 T2VA 的 packed geometry、三轴 RoPE、Q/K/V + per-head QK RMSNorm、SwiGLU、双输出头、四步双 schedule、AdaLN cache 和显式 noise fixture 入口。
- 已扩展 `components::AffineMatrix` 支持 manifest 驱动的 affine INT6/g64，并对宽 packed sequence 保留临时 dequantize + dense GEMM dispatch；真实 checkpoint 已证明 bit-packing/loader/qmm 数值正确，dispatch 门槛和目标尺寸性能仍需单独 microbenchmark。
- checkpoint 的 safetensors/manifest 适配暂放在 `native/platform/apple/h3_mlx_checkpoint.mm`，模型数学和 MLX tensor 代码保持在纯 C++ 的 `native/models/h3_mlx/`；这是为了遵守仓库的 `native/models` 禁止 Objective-C++ 边界，后续可以替换成纯 C++ JSON reader，但不能把 Foundation 依赖重新塞回模型目录。
- 已完成 full video/audio VAE、streamed safetensors、raw tokenizer、prompt cache 和 streamed Qwen3-VL 前 50 层，并在真实 ModelScope 权重上完成 DiT、conditioner 和两个 VAE 的 GPU parity。
- 已新增 `minimax-h3-fasth3-mlx-int6` module 与 `h3_mlx_session.mm`，串联 ModelScope provenance 校验、conditioner/cache、四步 INT6 DiT、full video VAE、audio VAE 和 native H.264/AAC mux；CLI descriptor 和 execution plan 已可见，旧 `minimax-h3-turbo` C/Metal profile 保持不变。
- 22 帧 smoke 和正式 832×480×124 主 case 均已生成 H.264/AAC MP4；124 帧单次冷/热配对的端到端比值分别约 `1.0049×` 和 `0.9966×`，共享 noise 冷配对约 `1.0059×`。这些结果满足目标量级，但不是完整 ABBA median，不能替代第 5 节的发布性能门禁。
- 已增加显式 `noise_path` 和仅验证时启用的 `dump_tensors`。在修正 C++/NumPy ARM64 float32 MRoPE 数学实现并升级 prompt cache schema 后，英文和中文 probe 的 conditioner hidden 当前均达到 `max_abs=0`；124 帧共享 conditioning + 共享 video/audio noise 时，FastVideo 与 TurboCider 的 token tags、conditioning、最终 video/audio rows 也全部 `max_abs=0`。这证明此前的 `4.27e-4` 是已修复的实现差异，不再是当前候选链的结果。
- 当前仍缺媒体 metadata/PTS 自动化测试、多 prompt/seed 的冷/热 ABBA 和盲评结论。VSA 主样例 warm ABBA 已加入 process-tree RSS 和系统 swap 增量采样，Dense/VSA 也已完成共享-noise raw RGB/raw PCM 对比，但单个 prompt/seed 不能替代完整感知矩阵。MLX allocator 峰值已经按 conditioning、denoise、video decode、audio decode 分阶段记录。因此候选 profile 虽已可执行，仍不得作为默认或已发布路径。

因此当前可声明的结果是“设计文档 + ModelScope-only 资产 + 已通过的组件/共享输入 DiT parity + 主 case 单次性能证据 + 可执行的端到端候选链”；在完成第 5 节的多样本性能和产品质量验收前，不得把 `minimax-h3-fasth3-mlx-int6` 标记为默认、推荐或已达到完整端到端 FastVideo parity。

### 0.1 本轮真实证据（2026-09-12）

模型只从 ModelScope 获取：

```text
repository = FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree
endpoint   = https://modelscope.cn
revision   = master
```

下载脚本为 `tools/h3/download_fasth3_modelscope.py`，采用 ModelScope resolve URL、单文件 `.part` 续传和最终目录直写；H3 路径不调用 Hugging Face、hf-mirror 或 `snapshot_download`。当前本机已保留 text encoder、full video VAE、audio VAE、tokenizer、processor、scheduler 和 provenance，合计约 72.43 GiB；约 62 GiB 的 BF16 transformer 源权重已在 INT6 转换与校验完成后删除，仓库不提交任何权重。

已执行并通过：

| 验证 | 结果 |
|---|---|
| C++/MLX INT6/g64 四步 DiT vs FastVideo | 所有 step 的 video/audio velocity、sample、最终 latent `max_abs=0`；早期 micro-probe 为 `qmm_calls=1248`、`dq_gemm_calls=0`，最新 124-frame 主 case telemetry 为 `qmm_calls=48`、`dq_gemm_calls=1200`（宽矩阵走解量化 + dense GEMM dispatch） |
| streamed C++/MLX Qwen3-VL conditioner vs FastVideo | 英文/中文 probe 的 token tags 完全一致；hidden cosine 约 `1.0`、`max_abs=0`；C++ Cody–Waite/FMA MRoPE 与 NumPy ARM64 float32 结果逐元素一致；最新 124-frame run 的 native `13.435 s` vs FastVideo `12.867 s`，约 `1.044×` |
| C++/MLX audio VAE vs FastVideo | latent `max_abs=3.73e-9`；waveform `max_abs=2.24e-7`；cosine `0.9999999999995` |
| C++/MLX full video VAE vs FastVideo | latent `max_abs=0`；pixels `max_abs=3.64e-6`；cosine `0.999999999998` |
| tokenizer、streamed safetensors、prompt cache | 全部通过；tokenizer 覆盖中英文、特殊标记、Unicode 和换行样例 |
| native 端到端 22-frame cold smoke | prompt→conditioner→INT6 DiT→full video/audio VAE→H.264/AAC MP4 成功；总时长 `65.485 s`，text `13.440 s`、denoise `38.323 s`、video decode `12.550 s`、audio decode `0.251 s`、mux `0.390 s`；peak MLX `17,929,146,354 B` |
| native 端到端 22-frame prompt-cache hit | `prompt_cache_hit=true`，text encode `0.000945 s`，总时长 `46.865 s`；denoise `34.928 s`、video decode `11.565 s`、audio decode `0.145 s`、mux `0.129 s`；相对 cold smoke 约 `1.40×` 加速 |
| 分阶段 MLX 内存（cache-hit smoke） | condition `163,872 B`、denoise `17,929,146,354 B`、video decode `11,243,610,416 B`、audio decode `1,174,617,580 B`，结束 active `2,281,384 B`；不包含 process RSS/media encoder |
| 输出媒体 metadata | H.264 `832×480`、`24/1 fps`、`22` frames、`0.916667 s`；AAC `32000 Hz`、2 channels、`0.916656 s`；A/V 时长差约 `0.000011 s` |
| native contract + H3 source/geometry/cache/prepare/ModelScope/benchmark tests | `83 passed, 1 skipped, 45 subtests passed` |
| native build 与 profile plan | `TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh` 通过；plan 显示 `mlx_cpp_metal`、`int6_g64_bf16_activation`、4-step AV denoise、两个 full VAE 和 mux |
| 124-frame 共享 conditioning/noise DiT E2E tensor parity | conditioning、token tags、最终 video rows、audio rows 全部 `max_abs=0`；证明 native 长序列 INT6 DiT、scheduler 和 packed layout 与 FastVideo 一致 |
| 124-frame 各自 conditioner + 共享 noise 复验 | token tags、conditioning、最终 video rows、audio rows 均 `max_abs=0`；e2e tensor parity `passed=true`。该结果 supersede 早期未修复 MRoPE 的漂移诊断 |
| 124-frame 输出媒体复验 | H.264 `832×480`、24 fps、124 frames；AAC `32000 Hz`、2 channels；video duration `5.166667 s`、audio duration `5.166656 s`，A/V 差约 `11 μs`；输出可由 `ffprobe` 复核 |

最新 124-frame 复验记录 TurboCider text `13.435 s`、denoise `263.316 s`、video decode `76.976 s`、audio decode `0.542 s`、mux `0.322 s`、request wall `354.735 s`；denoise peak `20,754,200,586 B`。这些仍是单次测量；124-frame 已有严格同条件 tensor parity，但仍未达到多 prompt/seed、预热和 ABBA median 的发布统计要求。性能/质量门禁仍按第 5 节执行。

### 0.2 主 case 单次性能快照

固定环境与输入：Apple M4 Max、64 GiB unified memory、MLX `0.32.2`、FastVideo commit `a943220c115228ade5d57b3bab9a6a87fd600a10`；832×480×124、24 fps、4 steps、prompt `A red fox runs through fresh snow.`、seed 2026、INT6/g64、full video/audio VAE、VSA/TAEH3 disabled。

| 模式/阶段 | FastVideo (s) | TurboCider (s) | TurboCider / FastVideo |
|---|---:|---:|---:|
| cold total | 352.421246 | 354.132539 | 1.0049× |
| cold condition | 12.681988 | 13.484396 | 1.0633× |
| cold denoise | 261.101629 | 262.588704 | 1.0057× |
| cold video decode | 77.742781 | 77.055924 | 0.9912× |
| warm/cache-hit total | 339.028146 | 337.892556 | 0.9966× |
| warm/cache-hit condition | 0.004397 | 0.000964 | 0.2192× |
| warm/cache-hit denoise | 260.820133 | 260.159836 | 0.9975× |
| warm/cache-hit video decode | 77.426316 | 76.908036 | 0.9933× |

另一次显式共享 noise 冷配对为 FastVideo `352.418453 s`、TurboCider `354.498206 s`，比值 `1.0059×`；分阶段比值为 conditioner `1.0601×`、denoise `1.0067×`、video VAE `0.9946×`、audio VAE `1.0071×`。TurboCider 该次 denoise peak 为 `20,754,200,586 B`，FastVideo 为 `19.4337 GiB`，换算后接近；两者 video decode peak 约 11.1 GiB。

这些数字说明当前实现没有明显性能落后，且主 case 单次结果达到 `<=1.05×` 目标；但它们不具备正式 ABBA median 的统计效力。发布报告必须保留原始 stdout/stderr、每次结构化 JSON、运行顺序、温度/电源状态和 media probe，不能只引用本表。

### 0.3 共享输入质量定位结论

质量验收拆成两个互补问题：

1. **DiT/runtime 是否实现正确。** FastVideo 读取 TurboCider dump 的 conditioning，并与 TurboCider 读取相同 video/audio noise；最终 video/audio rows 全部 `max_abs=0`。这证明长序列 packed layout、RoPE、AdaLN、INT6 dispatch、四步双 scheduler 和输出头没有观测到数值漂移。
2. **TurboCider 自有 conditioner 是否产生等价产品质量。** MRoPE 数学修复后，当前英文/中文 probe 及 124-frame 主 case 的 conditioner/最终 rows 均达到逐元素 `max_abs=0`。仍需用多 prompt/seed 和 raw-frame/raw-PCM 指标确认这不是单一输入偶然现象；早期 `24.23 dB/0.8215` 媒体诊断只代表修复前版本，不应作为当前实现结论。

因此 conditioner 的局部阈值只作为组件健康检查，不再作为最终质量放行条件。正式发布必须满足第 5.2 节的“共享 conditioning/noise 严格数值门”以及“各自 conditioner 产品感知门”两层验收。工具链为：

- `tools/h3/create_fasth3_noise_fixture.py`：按 FastVideo MLX key split 生成 `video_noise`/`audio_noise`；
- `tools/h3/run_fastvideo_fasth3_noise_e2e.py`：离线运行本地 ModelScope 资产，支持 `--conditioning-input`、`--dump-tensors`、`--stop-after-denoise`；
- TurboCider request：使用 `noise_path` 和 `dump_tensors`；
- `tools/validation/h3_mlx_e2e_tensor_parity.py`：不需要 Metal，直接比较 safetensors 中 conditioning、token tags 和最终 rows。

### 0.4 VSA 实现快照（2026-09-12）

Dense 候选链闭环后，工作区已实现独立的
`minimax-h3-fasth3-mlx-int6-vsa` profile；它不会改变 Dense profile 或旧
`minimax-h3-turbo` C/Metal profile。真实 VSA checkpoint 已从 ModelScope 下载、
转换并在 M4 Max 上完成 22/124 帧 reference 路径验证。当前可以声明
“VSA C++/MLX INT6 数值正确且性能接近 FastVideo VSA”，但不能声明
“VSA 已比 Dense 更快”或“已经通过多样本产品发布门”：

- `native/models/h3_mlx/vsa.*` 实现 FastVideo 对应的 prefix segments、3D video
  tile geometry、64→`(4,4,4)` 和 256→`(4,8,8)` tile、ragged tile size、
  gather/untile index、top-k 和 dense-first-step/layer 规则。
- `native/models/h3_mlx/vsa_attention.*` 已接入 pooled Q/K routing、prefix
  `exempt`/`compete`、selected K/V `mx::gather`、memory-bounded query chunks、
  MLX batched SDPA 和 `attn.to_gate_compress.weight` 分支。Dense attention 继续
  走原 fused SDPA，不经过 VSA 代码。
- request/profile 支持 `vsa_sparsity`、`vsa_tile_size`、`vsa_prefix_mode`、
  `vsa_dense_first_n_steps`、`vsa_dense_layers` 和 `vsa_impl`；Dense profile
  拒绝这些 VSA 参数，VSA profile 要求显式 `vsa=true`。
- checkpoint loader 要求 `vsa.capable=true`、50 个 gate matrices、BF16
  attention activations，并核对真实量化 tensor 数量；Dense-only manifest
  `capable=false` 会 fail-closed。
- native result 已包含 configured/achieved sparsity、video keep、tile 数量、
  prefix mode、implementation、attention/sparse call 数和 fallback reason。
- `vsa_impl=simd` 当前会明确报告 reference fallback；尚未实现或验收 FastVideo
  的 SIMD/Metal 稀疏 kernel，不能把 reference 数据标成 SIMD 数据。
- VSA BF16 transformer 约 65 GiB，独立 INT6 checkpoint 约 17 GiB；manifest
  已验证 `vsa.capable=true`、50 个 gate matrices 和 ModelScope provenance。
- 真实 gate checkpoint 的 22/124 帧 MLX shape、top-k、索引和最终 latent 已与
  FastVideo 对齐。修复的两个根因是：routing score 必须保持
  `(q_pool @ k_pool.T) / sqrt(dim)` 的直接除法表达式；`vsa_sparsity` 必须使用
  `double`，否则 `0.9` 在 280 video tiles 上会错误保留 29 而不是 28 个 tile。
- `vsa_compute_topk(0.9, 280) == 28` 已加入 native regression test；最新 native
  build 和 VSA profile/source contract tests 通过。

VSA 模型只允许从 ModelScope 获取：

```text
FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree
```

准备顺序固定为：先做空间预检并下载 transformer；转换成独立
`vsa-int6/` 后验证 manifest/gate count；再复用或准备同一 VSA provenance 根下的
conditioner/VAE/runtime 组件。不得从 Hugging Face 或 hf-mirror 补文件，也不得让
VSA root 伪装成 Dense ModelScope provenance。建议命令为：

```bash
python tools/h3/download_fasth3_modelscope.py \
  --model-id FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree \
  --root <FastH3-VSA-ModelScope> --component transformer

python tools/h3/prepare_fasth3_mlx.py \
  --source <FastH3-VSA-ModelScope>/transformer \
  --repository FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree \
  --out <FastH3-VSA-ModelScope>/vsa-int6 --formats int6 --include-vsa
```

当前主配置固定为 832×480、4 steps、sparsity 0.9、tile 64、prefix exempt、
`reference`。下表是用于严格 tensor 定位的共享-noise 单次阶段计时；正式主样例
warm ABBA 结果单独列在表后：

| Case | FastVideo VSA | TurboCider VSA | TurboCider / FastVideo | 数值结果 |
|---|---:|---:|---:|---|
| 22 帧 denoise | 40.1127 s | 39.6495 s | 0.9885× | conditioning/tags/video/audio 最终 rows 全部 `max_abs=0` |
| 124 帧 denoise | 278.9233 s | 281.9010 s | 1.0107× | conditioning/tags/video/audio 最终 rows 全部 `max_abs=0` |

124 帧两侧均为 `video_keep=28/280`、`attention_calls=200`、
`sparse_calls=200`。TurboCider VSA denoise peak 为 `23,837,542,010 B`；FastVideo
记录约 `22.20245 GiB`。TurboCider/FastVideo 的单次速度差约 1.1%，达到
“不明显慢于 FastVideo VSA”的目标量级。

124 帧主 prompt/seed 的 warm-cache ABBA 使用顺序
FastVideo→TurboCider→TurboCider→FastVideo，先对每个 runtime 各预热一次；
结构化报告记录如下：

| 阶段 | FastVideo median | TurboCider median | TurboCider / FastVideo |
|---|---:|---:|---:|
| total | 356.0100 s | 357.7342 s | 1.00484× |
| condition/cache | 0.0050 s | 0.00117 s | 0.2346× |
| denoise | 278.6800 s | 280.7811 s | 1.00754× |
| video decode | 76.5650 s | 76.1518 s | 0.99460× |
| audio decode | 0.4300 s | 0.47885 s | 1.11360× |
| mux | 0.3300 s | 0.21049 s | 0.63784× |

两次正式 FastVideo run 为 `355.98/356.04 s`，两次 TurboCider run 为
`357.7934/357.6750 s`。process-tree peak RSS median 分别为
`18,065,137,664 B` 和 `18,094,972,928 B`，TurboCider/FastVideo 约
`1.00165×`；四次正式 run 的系统 swap 增量均为 0。预热阶段的 TurboCider
cold conditioner run 曾观察到 `634,199,736 B` 系统级 swap 增长，因此 cold
ABBA 仍需单独完成，不能用 warm 结果掩盖首次运行内存压力。

ABBA 首对产品媒体的 encoded PSNR 为 `43.804 dB`、SSIM `0.986254`；raw RGB
PSNR `39.523 dB`、cosine `0.999875`；raw PCM cosine `0.998241`、RMSE
`0.000316`，音频长度差 `0.017344 s`，仍小于一帧。该主样例的性能、媒体合同和
质量指标均通过当前候选门，但仍不是 3 prompts × 3 seeds 的最终产品结论。

22 帧产品链已经完成 prompt→conditioner→VSA INT6 DiT→full video/audio VAE→
H.264/AAC mux。输出为 832×480、24 fps、22 frames、H.264 High、AAC LC、
32 kHz stereo；视频/音频时长分别为 `0.916667 s` 和 `0.916656 s`，差约
`11 μs`。共享 noise 的 FastVideo/TurboCider 媒体对比结果为：encoded PSNR
`45.126 dB`、SSIM `0.988954`；raw RGB PSNR `40.695 dB`、cosine
`0.999925`、mean MAE `1.684/255`；raw PCM cosine `0.994376`、RMSE
`0.001467`，样本长度差 `0.011344 s`，小于一帧。配合最终 latent
`max_abs=0`，该样例通过当前 aligned-RGB 回归门。

Dense 与 VSA 使用不同的 DataFree student checkpoint，因此同 noise 下不会产生
像素对齐的同一视频，不能用低 PSNR 直接判断任一侧质量差。当前红狐样例中两侧都
能生成主体清晰、背景稳定且具有连续运动的雪地红狐；VSA 构图和运动轨迹与 Dense
明显不同。该结论仍只是工程人工检查，最终报告必须加入 3 prompts × 3 seeds
盲评和感知特征距离。

性能上，当前 `reference` VSA 在 M4 Max 上没有超过 Dense：22 帧同轮 TurboCider
denoise 为 VSA `39.6495 s`、Dense `37.2175 s`，VSA 慢约 `6.5%`；124 帧为
VSA `281.901 s`、Dense `263.316 s`，VSA 慢约 `7.1%`。FastVideo 自身也显示
近似比例，因此这不是 TurboCider 特有回归，而是 reference gather+SDPA 路径的
当前成本。若要获得相对 Dense 的真实加速，必须实现并单独验收 SIMD/Metal 稀疏
kernel；不得把 `reference` 或显式 fallback 数据标成 SIMD 加速数据。

剩余验证按以下顺序推进：cold ABBA；3 prompts × 3 seeds 产品矩阵；首次运行
RSS/swap 优化；感知特征和盲评。tile 256、prefix compete、dense-first 1 和
`simd` fallback 是扩展矩阵，不能替代主 case。

FastVideo 与 TurboCider 的 VSA 参数映射必须保持显式且可审计：

| FastVideo CLI | TurboCider request/profile | 当前状态 |
|---|---|---|
| `--vsa` | `model=minimax-h3-fasth3-mlx-int6-vsa` 且 `vsa=true` | 已接入；缺一即拒绝 |
| `--vsa-sparsity` | `vsa_sparsity` | 已接入，范围 `[0,1)`；主 case `0.9` |
| `--vsa-tile-size` | `vsa_tile_size` | 已接入，仅允许 `64`/`256` |
| `--vsa-prefix-mode` | `vsa_prefix_mode` | 已接入，`exempt`/`compete` |
| `--vsa-dense-first-n-steps` | `vsa_dense_first_n_steps` | 已接入，当前四步范围 `0…4` |
| `--vsa-dense-layers` | `vsa_dense_layers` 数组 | 已接入，层号 `0…49`、不可重复 |
| `--vsa-impl` | `vsa_impl` | `auto`/`reference`/`simd`；当前 `simd` 明确回退，未冒充实现 |
| `--mlx-checkpoint` | VSA ModelScope root 下受 manifest/provenance 约束的 `vsa-int6/` | loader fail-closed，禁止请求传入任意 checkpoint 绕过身份校验 |

`tools/native/benchmark_h3_mlx.py` 负责从一组参数同时生成 FastVideo CLI 和
TurboCider request：FastVideo 的 dense layer 列表使用逗号分隔，TurboCider 保持
JSON 数组。两侧参数必须写入同一个 benchmark record，禁止人工维护两套容易漂移
的配置。

## 1. 参考实现与范围冻结

参考实现是 `references/FastVideo/fastvideo/mlx_runtime/` 及其 `mlx_fasth3.py`、`convert_minimax_h3_mlx.py`。本项目实际冻结的 student checkpoint 是 ModelScope `FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree`（`master`，step 1000，transformer 13 shards，`66,245,985,792` bytes），不是普通 MiniMax-H3 或旧的 LightX2V 合并权重。Hao AI Lab 的 FastH3 本地方案给出的关键原则是：

- 四步 FastH3 student，而不是把普通 H3 强行减少到四步；视频/音频使用各自的 rectified-flow Euler schedule（video shift 12、audio shift 3）。
- Encode → Denoise → Decode → Export 分阶段运行；文本编码器只保留 Qwen3-VL 中间层所需 hidden features，跳过最后 14 层，并缓存 prompt embedding。
- DiT 采用 `[text | condition | audio | video]` packed sequence、三轴 RoPE、视频/音频双输出头、双模态 AdaLN；固定四步时预计算 AdaLN 表并可释放对应投影权重。
- 线性层采用 affine weight-only quantization，group size 64；INT6 作为本项目默认格式，输入/输出投影、norm、embedding、modulation 保持 BF16/FP32。FastVideo 的实测显示，INT6 在 M4 Max 上是速度/内存折中默认档；大矩阵上“量化权重临时解量化 + dense GEMM”可能比直接量化 GEMM 更快，必须保留 runtime dispatch。
- VSA 是独立可选能力：只有包含 `to_gate_compress` 权重的 checkpoint 才可启用；首版先完成 dense INT6，VSA 作为同一接口下的第二阶段实验开关。
- Full H3 VAE 是质量路径；TAEH3 是预览路径，不能混用质量结论。音频必须保留完整时长并与视频 24 fps 同步。

首发范围冻结为：

| 项目 | 首发 | 明确不纳入首发 |
|---|---|---|
| 任务 | T2VA `video.generate` | FL2VA、Ref2VA、keyframes、reference、LoRA |
| 采样 | FastH3 student，4 steps，固定 step ladder | 任意步数、普通 H3 多步兼容 |
| 精度 | DiT affine INT6/g64；激活 BF16；关键层 FP32 | INT4/INT8 产品开关、混合量化搜索 |
| 解码 | Full H3 video VAE + H3 audio VAE | 首发默认 TAEH3；TAEH3 仅预览验收 |
| 注意力 | dense MLX SDPA | VSA 产品默认；VSA 先做候选实验 |
| 平台 | Apple Silicon macOS，MLX C++ | CUDA、ANE 私有 API、跨平台抽象 |
| 服务 | 复用 TurboCider native CLI/API/App 生命周期 | 新的独立 Python 服务进程 |

## 2. 目标架构

### 2.1 运行时分层

新增 `native/models/h3_mlx/`，不把 H3 MLX 算子塞进 `h3_session.mm`，也不修改现有 C runtime 的数学实现。建议目录如下：

```text
native/models/h3_mlx/
  h3_mlx_dit.hpp/.cpp          # packed DiT、refiner、scheduler、AdaLN cache
  h3_mlx_conditioner.hpp/.cpp  # Qwen3-VL 截断、hidden rows、prompt cache
  h3_mlx_video_vae.hpp/.cpp
  h3_mlx_audio_vae.hpp/.cpp
  h3_mlx_pipeline.hpp/.cpp     # phase lifecycle and media handoff
  h3_mlx_checkpoint.hpp/.cpp   # manifest, safetensors, INT6 contract
  h3_mlx_parity.hpp/.cpp       # debug tensor dumps and counters

native/platform/apple/h3_mlx_checkpoint.mm # 仅负责 Apple/Foundation manifest 适配
```

`native/backends/mlx.*` 继续拥有 MLX tensor、safetensors、quantized matmul、stream、内存统计和通用 linear/attention helper；H3 只依赖这些接口，不新增第二套 allocator。媒体封装继续使用 `native/media`。`native/platform/apple/h3_session.mm` 只负责请求验证、backend selection、取消、结果翻译和现有 session 生命周期。

module registry 保留两个明确可观测的 backend profile：

- `minimax-h3-turbo`：现有 C/Metal 路径，保持默认兼容；
- `minimax-h3-fasth3-mlx-int6`：新 C++/MLX 路径，仅接受上述首发范围，descriptor 的 `backend`、`precision`、`checkpoint`、`decoder` 和 `schedule_id` 必须写入结果。

不得通过环境变量静默切换；请求必须显式选择 `model=minimax-h3-fasth3-mlx-int6`。MLX 不可用、manifest 不匹配或统一内存预估不足时应返回可诊断错误；不得在 benchmark/验收中静默回退到 C/Metal。

### 2.2 阶段式内存生命周期

每个阶段有单一 owner，并在边界 `mx::eval`、记录计时/峰值后释放：

1. **Conditioning**：加载 tokenizer 和 Qwen3-VL text encoder；按 FastVideo 的中间层约定执行截断，输出 hidden rows、token tags、prompt fingerprint；写入 identity-bound cache；释放 encoder。
2. **Denoising**：加载 INT6 DiT manifest/权重、建立 packed layout、预计算四步 AdaLN union ladder；视频/音频 latent 同时去噪；完成后导出 CPU/MLX tensor snapshot，释放 DiT。
3. **Decode**：加载 full video VAE，采用与 FastVideo 相同的 tiled decode；释放 video VAE 后加载 audio VAE；音频输出固定 32 kHz stereo（以模型 contract 为准）。
4. **Export**：复用 TurboCider native mux，检查 24 fps、音频采样数、PTS、H.264/AAC 编码错误；所有阶段的 peak RSS/MLX peak memory 写入 `RunResult.native_json`。

优先复用现有 `component_staged` 和 `block_residency` 语义。首版不做跨 C/MLX tensor 零拷贝；跨边界必须明确 dtype、shape、stride、storageMode 和 completion。后续如 profiling 证明拷贝是瓶颈，再做受控 bridge。

## 3. Checkpoint 与 INT6 合同

### 3.1 来源与转换

模型统一从 ModelScope `FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree` 获取；本方案不使用 Hugging Face 或其镜像。不得把权重提交到仓库。下载和转换必须分离：先用 ModelScope 下载器落盘并校验，再让准备工具只读取本地 transformer：

```bash
# 示例，实际路径通过环境变量传入，不写死本机目录
python tools/h3/prepare_fasth3_mlx.py \
  --source <FastH3-ModelScope>/transformer \
  --out <TurboCider model cache>/FastH3-MLX \
  --formats int6
```

在准备 TurboCider 自有转换器之前，先用仓库内 FastVideo 脚本建立可复现的参考资产；这样可以把“模型/转换问题”和“C++ runtime 问题”分开：

```bash
# 如网络需要代理，在当前 shell 中使用用户已有的代理函数；不要把代理写入 manifest。
source ~/.bashrc
setproxy

# 通过 ModelScope 逐 shard 直写；不要使用会保留第二份权重副本的 snapshot cache。
python tools/h3/download_fasth3_modelscope.py \
  --root <FastH3-ModelScope> --component transformer

# 在 references/FastVideo 下执行；实际输出目录由调用者决定
python scripts/checkpoint_conversion/convert_minimax_h3_mlx.py \
  --model-root <FastH3-ModelScope>/transformer \
  --out <FastH3-MLX> --formats int6
python examples/inference/basic/mlx_fasth3.py \
  --model-root <FastH3-ModelScope> \
  --mlx-checkpoint <FastH3-MLX>/int6 \
  --prompt '<fixed prompt>' --height 480 --width 832 \
  --num-frames 124 --seed 2026 --output-path <fastvideo-output>.mp4
```

使用 ModelScope 时，必须在实验记录中写清 repo、revision、下载工具和文件 hash；`setproxy` 只影响下载/依赖安装，不得进入运行时配置或 manifest。H3 的下载器、运行时和测试不得导入 `huggingface_hub`，不得访问 `hf-mirror`，也不得调用 `snapshot_download`。上游 `provenance.json` 中的 `base_model=hf://...` 只是发布方给出的血缘字符串，TurboCider 必须把它当作不可解引用的元数据，不能据此触发网络访问。TurboCider 的转换器应先对 FastVideo 产物做 manifest/key/shape/dtype 复验，再允许生成自己的 INT6 checkpoint。脚本默认 component 是 `transformer`，也支持 `text_encoder`、`vae`、`audio_vae`、`runtime` 和 `all`。当前资产已经按“下载/转换 transformer → 校验 INT6 → 删除约 62 GiB BF16 transformer → 下载 runtime 49 files”的顺序准备完毕；后续机器应沿用此顺序控制峰值磁盘占用。

### 3.2 磁盘空间与清理规则

- 下载器先计算所选组件的缺失字节和最小剩余空间；不足时 fail-fast，不边下载边赌空间。
- 只允许清理可重建缓存、失败的 `.part`、临时媒体、旧 benchmark 输出和已经由 hash/真实 parity 验证过的 BF16 transformer 源副本。当前约 62 GiB BF16 transformer 已按此规则删除。
- text encoder、full video VAE、audio VAE、tokenizer、processor、scheduler、ModelScope provenance 和 INT6 checkpoint 属于运行时必需资产，不得作为“缓存”误删。
- 不自动删除未知目录或用户文件；每次物质性清理应记录目标、大小、原因和是否可恢复。模型权重、生成媒体和缓存均不得提交 Git。

转换逻辑应与 FastVideo `convert_minimax_h3_mlx.py` 对齐：流式读取 diffusers safetensors，避免完整 BF16 DiT 同时驻留；保留模型配置、量化 key 表、源 checkpoint 文件 hash、转换器版本、MLX 版本、macOS/架构、schedule ladder 和 VSA capability。

### 3.3 manifest 必填字段

```json
{
  "format_version": 1,
  "model_id": "minimax-h3-fasth3-preview",
  "task": "t2v_audio",
  "steps": 4,
  "schedule": {"video_shift": 12.0, "audio_shift": 3.0},
  "quantization": {"mode": "affine", "bits": 6, "group_size": 64,
                    "dequantized_dtype": "bf16",
                    "scales_dtype": "fp32"},
  "adaln_cache": {"timesteps": [], "num_blocks": 0},
  "vsa": {"capable": false},
  "source": {"checkpoint_sha256": "...", "converter_git": "..."}
}
```

加载器必须拒绝：bits/group size 不匹配、缺少 scales/biases、step ladder 不一致、源模型不是 FastH3 student、tensor key/shape 不完整、VSA 请求但 `vsa.capable=false`、或 MLX 当前版本不支持该 quantized matmul。禁止静默反量化成另一种格式后继续运行。

量化 key 与 FastVideo 对齐：`attn.to_q/k/v/out`、`ff.net.0.proj`、`ff.net.2`；`proj_in`、`audio_proj_in`、`time_embedder`、`proj_out`、`audio_proj_out`、norm、embedding、AdaLN 相关权重保持高精度。对宽矩阵实现可配置的 affine dequant+GEMM dispatch，并在 telemetry 中记录实际走的是 quantized matmul 还是 dequantized dense GEMM。

**重要实现约束：INT6 的物理 packing 不能靠旧的 4/8-bit 经验猜测。** 当前工作区原型已扩展 `native/components/weights/affine.*` 接受 6 bit，并允许由 manifest 显式传入 logical input width；但这只是第一步，MLX affine INT6 的 packing、scales/biases shape 和 `mx::quantized_matmul` 参数仍必须以目标 MLX 版本的实际 probe/保存产物为准。FastVideo 当前转换器实际产生 `scales`/`biases` 为 FP32，较早内部产物可能为 BF16；loader 只接受这两种 dtype，不做隐式 INT8/FP16 替代。后续应把 `AffineMatrix` 稳定为 manifest 驱动的 `PackedQuantGeometry`，至少覆盖：

- 逻辑矩阵 `(out_channels, in_channels)` 与物理 packed shape 的独立记录，不从 `32 / bits` 猜测；
- `in_channels % group_size == 0`、scale/bias 行列数、packed dtype、尾部 padding 和 transpose 方向的校验；
- `in_channels` 不能整除 32、64、128、256 的随机/真实矩阵测试，确保 6-bit 跨 word packing 不错位；
- C++ 生成 checkpoint 后立即用 MLX Python/reference 重新读取并对随机输入比较 `max_abs/mean_abs/SNR`；
- 不缓存每层解量化副本，宽矩阵 dispatch 的 floor 必须可配置并记录；如目标 MLX 没有可用 INT6 kernel，加载器应明确失败而非退回 INT8/FP16。

## 4. 实施阶段与退出条件

### Phase 0：基线与资产（1 周）

- 固定一台基准机（建议 M4 Max 36 GB；另选 M4 Pro/M5 作为兼容机），记录 macOS、MLX、Xcode、CPU/GPU、内存。
- 准备 FastH3 student、INT6 MLX checkpoint、文本/视频/音频 VAE；校验 SHA-256。
- 建立同条件 FastVideo Python 基线和 TurboCider C/Metal 基线：832×480、124 frames、24 fps、4 steps、固定 prompt/seed，冷/热各 8 次。
- 产出基线 JSON、latent dump、短视频/音频样本和 peak memory 记录。

退出条件：两条基线都能重复生成；FastVideo 与自身重复运行的 latent 差异、音视频时长和 seed 规则有记录；未完成前不宣称性能比较。

### Phase 1：MLX C++ 最小闭环

- 先实现 checkpoint loader、INT6 affine matmul、manifest 校验和小型随机 tensor parity。
- 实现单 block、RoPE、RMSNorm、AdaLN、attention、SwiGLU、双输出头；用 Python reference 导出每层输入/输出，与 C++ 逐层对比。
- 实现 4-step scheduler、packed layout、video/audio latent shape。

退出条件：CPU/Metal 可运行的 tiny fixture 逐算子误差达标；真实 checkpoint 能 load/unload/cancel；不接媒体前的 DiT latent 与 FastVideo reference 在容差内一致。

### Phase 2：Conditioning 与 prompt cache

- 迁移 tokenizer、Qwen3-VL 截断层逻辑和 hidden-row 保留规则。
- cache key 必须绑定模型目录/manifest hash、tokenizer 文件身份、prompt、语言/对话控制标记和 encoder 版本；原子写入，损坏即删除重算。
- 记录 cache hit、encoder 实际层数、valid token rows 和 conditioning bytes。

退出条件：cache miss/hit 输出 hidden features 一致；第二次请求不加载 encoder；取消和异常不会留下半文件。

### Phase 3：视频/音频解码与 native mux

- 先接 full H3 video VAE tiled decode，再接 audio VAE；严格复用模型 normalization、latent scaling、tile overlap 和音频采样规则。
- 接入已有 media/mux，不复用 TAEH3 结果作为 full-VAE 质量证据。
- 允许 `decoder=taeh3` 作为明确的 preview profile，但结果和验收报告必须单独标注。

退出条件：视频帧数/尺寸/24 fps、音频采样数/声道、PTS 和 mux 后可播放性全部通过；full VAE RGB 与 C/Metal baseline 接近。

### Phase 4：端到端 backend/profile 接入

- 在 `h3_session.mm` 和 module descriptor 中接入显式 backend selection、能力检查、fallback reason、阶段计时和峰值内存。
- CLI/API/App 能显示 `FastH3 MLX INT6`、checkpoint identity、decoder、prompt-cache hit、quant dispatch。
- 默认保持现有 C/Metal 行为；MLX profile 未通过真实质量/性能门禁前不得标记 executable/recommended。

退出条件：native CLI、local API、App smoke、取消、重复请求、卸载/重载、模型缓存清理均通过。

### Phase 5：性能优化与 VSA 候选

- 先用 Instruments、MLX peak memory、阶段计时判断瓶颈，再优化：AdaLN cache、宽矩阵 dequant+GEMM、attention compile、prompt cache、磁盘预取、VAE tile 并行。
- Dense INT6 已达到单次性能量级与严格 tensor parity，因此 VSA 第一版 reference 路径已开始实现；只有真实 VSA checkpoint 的 22/124-frame 验证、FastVideo ABBA 和感知质量门通过后，才可把 VSA 从实验 profile 提升为候选。VSA 必须保留 dense-first-step/layer fallback 和稀疏命中率 telemetry。
- `--fast`/spatial fast/RIFE 属于预览档，不参与 full-quality 首发门禁。

FastVideo 当前源码还记录了一个必须复现而不能拍脑袋固定的 dispatch 结论：在 Apple M4 Max、MLX 0.32.2、INT6/g64、BF16 activation 的 H3 宽投影上，短序列更适合 `quantized_matmul`，超长 packed sequence 才考虑临时 `dequantize + dense GEMM`，门槛约为 `M=768`。TurboCider 必须把该门槛作为可观测配置，在目标机器上重新测量后再决定默认值；不能把某一台 Mac 的结果硬编码成普适规律。

### 4.1 可拆分任务与合并顺序

每一行应是一组可独立 review、可回退、不会把半成品标成可执行 profile 的变更：

| ID | 工作包 | 依赖 | 必须产物 | 合并门槛 |
|---|---|---|---|---|
| H3-000 | 固定 FastVideo commit、MLX 版本、模型 revision/hash 和 A–E 输入矩阵 | 无 | `baseline_manifest.json`、FastVideo 原始 benchmark JSON | 同条件基线可重复，不能只留截图 |
| H3-100 | INT6 capability probe、packed geometry、quantized/DQ-GEMM microbench | H3-000 | probe CLI、随机矩阵 golden、dispatch 表 | 目标 MLX 上 INT6/g64 可执行；误差和速度均有记录 |
| H3-110 | manifest/safetensors loader、完整 key/shape/dtype 校验 | H3-100 | loader、损坏/缺 key/错误 ladder tests | fail-closed、cancel/unload 不泄漏；模型目录无 ObjC++ |
| H3-120 | geometry、RNG/noise、scheduler、AdaLN union | H3-000 | Python/C++ golden tensors | `17n+5`、位置网格、四步双 schedule、noise fixture 一致 |
| H3-130 | refiner + 50-block DiT + 双输出头 | H3-110/H3-120 | per-layer dump、tiny/real latent parity | 四步 video/audio latent 通过 5.2 门槛 |
| H3-200 | streamed Qwen3-VL 前 50 层和 prompt cache | H3-000 | cache schema、conditioner parity dumps | miss/hit 一致；单层 residency；坏缓存可恢复 |
| H3-300 | full video VAE tiled decode、audio VAE、mux | H3-000 | decoder fixtures、MP4/WAV metadata report | frame/audio/PTS、RGB/audio parity 通过 |
| H3-400 | pipeline、session、profile、CLI/API/App、telemetry | H3-130/H3-200/H3-300 | 端到端 profile 与结果 schema | 不支持能力明确拒绝；旧 C/Metal 默认无回归 |
| H3-500 | ABBA profiling 与定向优化 | H3-400 | raw JSON、Instruments trace、决策记录 | 满足 5.3 性能/内存门槛 |
| H3-600 | 可选 VSA/预览档 | H3-500 | 独立 checkpoint/profile/report | dense 首发已通过；不得污染 full-quality 结论 |

严格数值 parity 不应依赖“两个 runtime 恰好实现相同 seed RNG”。H3-120 需要提供测试专用 `--noise-input`/latent fixture 入口，让 FastVideo 导出的初始 video/audio noise 被 TurboCider 原样读取；用户态 seed reproducibility 另做测试。性能测试则仍使用各 runtime 正常 seed 接口，避免 debug I/O 污染计时。

### 4.2 端到端数据合同

- 输入尺寸先向模型 32-pixel canvas 对齐；输出再中心裁剪到请求尺寸。帧数必须对齐为 `17n+5`，`124` 帧对应 video latent frame 数 `37`。
- video latent 逻辑布局固定为 `[24, T, H, W]`，patch `(1,2,2)` 后成为 row-major packed video tokens；audio latent 固定为 `[2, 32, T_audio]`，按 channel-major 变为 `2*T_audio` 行。
- T2V 首发没有 condition rows，packed 顺序是 `[text | audio | video]`，但内部 layout/indices 仍按通用 `[text | condition | audio | video]` 建模，避免未来扩展时重写索引规则。
- FastVideo 的初始噪声从一个 MLX seed key 分裂为 video key 和 audio key。TurboCider 必须锁定 key split 次序；parity 模式优先读取显式 noise fixture。
- 每一步分别计算 video/audio timestep，把每一行映射到 sorted unique timestep，再用 `(cache_position * 3 + modality_tag)` gather AdaLN；Euler blend 必须在 FP32 完成后 cast 回 activation dtype。
- 阶段边界不得传递未求值的 MLX graph；`mx::eval` 完成后才能记录计时、释放阶段 owner 或交给媒体线程。

### 4.3 当前工作包状态（2026-09-12）

| 工作包 | 当前状态 | 已有证据 | 下一退出条件 |
|---|---|---|---|
| H3-000 资产/基线 | 主 case 单次完成，统计未完成 | ModelScope repo/revision/content hash 已冻结；124-frame cold/warm/shared-noise 配对已运行 | 补齐 ABBA 多 prompt/seed 原始 JSON 与重复性统计 |
| H3-100/110 INT6 与 loader | 组件完成 | 真实 INT6/g64 QMM、manifest fail-closed、source contract 通过 | 补目标机器 dispatch microbench 表 |
| H3-120/130 geometry/DiT | 数值完成 | tiny 四步全部中间量 `max_abs=0`；124-frame 共享 conditioning/noise 最终 video/audio rows `max_abs=0` | 将共享输入 E2E tensor parity 固化为 fixture-backed 测试 |
| H3-200 conditioner/cache | 数值完成，产品多样本质量未完成 | MRoPE 修复后英文/中文 probe 与 124-frame 主 case 均 tags exact、hidden/最终 rows `max_abs=0`；prompt cache v2 命中正常 | 完成 3×3 prompt/seed 产品门；补坏缓存/并发 E2E |
| H3-300 VAE/mux | 主 case E2E 完成 | full video/audio VAE parity 通过；22/124-frame H.264/AAC metadata 与同步正确；临时 video-only 文件已清理 | 将帧率/采样率/PTS/临时清理固化为自动测试 |
| H3-400 session/profile | CLI 候选与重复请求完成 | profile/registry/plan/session、noise fixture、tensor dump 已接入；旧 C/Metal 默认未变 | API/App、cancel 和异常路径 smoke |
| H3-500 性能/质量 | 性能量级通过，发布门未完成 | 单次 cold `1.0049×`、warm `0.9966×`、shared-noise `1.0059×`；主 case tensor parity 已通过，但多样本感知门尚未完成 | 完成 5.1–5.3 的 ABBA、raw-frame/raw-PCM 指标和盲评 |
| H3-600 VSA/预览 | reference 候选链和 124-frame warm ABBA 已完成，发布统计仍未完成 | ModelScope VSA BF16/INT6、50 gate manifest、22/124-frame 最终 rows `max_abs=0`；22-frame 产品 MP4、raw RGB/PCM 单样例通过；124-frame warm ABBA TurboCider/FastVideo `1.00484×`，正式 run RSS 约 `1.00165×` | 完成 cold ABBA、3×3 感知/盲评、首轮 swap/内存压力处理；实现并单独验收 SIMD/Metal sparse kernel 才能宣称相对 Dense 加速 |

### 4.4 最小可观测结果格式

每次运行至少输出以下字段，benchmark 工具只读取机器生成的 JSON：

```json
{
  "profile": "minimax-h3-fasth3-mlx-int6",
  "backend": "mlx_cpp_metal",
  "checkpoint_sha256": "...",
  "schedule_id": "fasth3-v0.2-v12-a3-4step",
  "decoder": "full-h3-vae",
  "prompt_cache_hit": false,
  "quant": {"bits": 6, "group_size": 64, "dq_gemm_floor_m": 768,
            "qmm_calls": 0, "dq_gemm_calls": 0},
  "vsa": {"enabled": false, "configured_sparsity": 0.0,
          "achieved_sparsity": 0.0},
  "timings_seconds": {"request_wall": 0, "text_encode": 0, "denoise": 0,
                      "video_decode": 0, "audio_decode": 0, "mux": 0},
  "mlx_peak_bytes": 0,
  "mlx_active_bytes": 0,
  "memory": {"condition_peak_bytes": 0, "denoise_peak_bytes": 0,
             "video_decode_peak_bytes": 0, "audio_decode_peak_bytes": 0,
             "scope": "MLX allocator; excludes process RSS and media encoders"},
  "output": {"width": 832, "height": 480, "frames": 124,
             "fps": 24, "audio_rate": 32000, "audio_channels": 2}
}
```

`fallback_reason`、`cancelled_phase`、`vsa`、`preview_decoder` 等字段按运行情况追加；任何 fallback 都必须使 benchmark case 失败，不能拿 C/Metal 或 TAEH3 的结果冒充 MLX full-quality 结果。

## 5. 验收方案

### 5.1 固定测试矩阵

主矩阵至少包含：

| Case | 尺寸/帧数 | 目的 |
|---|---|---|
| A | 832×480×124，4 steps | 主性能与质量门禁，复现 FastVideo 文章口径 |
| B | 512×512×22 | 小尺寸启动、内存和 shape 边界 |
| C | 768×1344×124 | 大尺寸压力与 tiled VAE |
| D | 832×480×124，重复 prompt | prompt cache 命中与多请求驻留 |
| E | 832×480×124，TAEH3 | 仅预览速度/质量对照，不替代 full VAE |

每个 case 使用 3 个文本 prompt（普通动作、人物/细节、含英文对白控制标记）和 3 个 seed；每种 runtime 冷启动 3 次、预热后有效运行 8 次，采用 ABBA 交错顺序。报告 median、p95、阶段分解和 peak RSS/MLX memory；不得只报最快一次。

### 5.2 数值与质量门禁

分层验证，先定位算子误差，再看媒体质量：

1. tokenizer ids、conditioning hidden rows、packed layout、timesteps、noise 初值逐项一致。
2. conditioner contract：token tags 必须完全一致，hidden cosine `>= 0.99999` 且 `max_abs <= 1e-3`。该门槛只代表 conditioner 组件健康，不代表最终质量通过；当前英文/中文及主 case 真实结果为 `max_abs=0`。任何放宽都必须同时给出误差分布，不能只看 cosine。
3. 共享 conditioning/noise 的每个 denoise step 以及最终 video/audio latent 与 FastVideo Python reference 对比：默认 `max_abs <= 2e-2`、`mean_abs <= 2e-3`；主 case 124 帧应优先要求最终 rows `max_abs <= 2e-3`、cosine `>= 0.99999`，因为共享输入下当前实现已达到 `max_abs=0`。若 BF16/MLX fused kernel 造成稳定差异，必须给出逐层误差增长和最终媒体指标，不得只放宽总阈值。
4. 各自 conditioner 的产品质量不能只用 hidden cosine 放行：在至少 3 prompts × 3 seeds 的共享 noise/同编码器诊断中，最终 video/audio latent 必须达到 `max_abs <= 2e-2`、`mean_abs <= 2e-3`，并且统一 raw-frame/raw-PCM 的感知指标达到本节第 5、6 项门槛；否则继续修 conditioner 或将 profile 保持 candidate。
5. TurboCider MLX 与 TurboCider C/Metal 使用同一 FastH3 checkpoint/seed 时，最终 latent 的 `max_abs <= 3e-2`、`mean_abs <= 5e-3` 为建议门槛；任何超限需归因到量化、RoPE、scheduler、VAE 或编码器。
6. Full-VAE RGB 质量：统一解码 raw frames 后，固定帧抽样 PSNR ≥ 35 dB、SSIM ≥ 0.95 作为初始门槛；同时用 LPIPS/DINO/Vision feature distance 做感知回归，不能只依赖编码 MP4 的 PSNR/SSIM。
7. 视频专项：帧数、尺寸、黑帧率、平均亮度/色彩、运动连续性、音视频同步误差 ≤ 1 frame；对白样例人工检查口型/语音可懂度。任何 NaN、静帧、音频漂移或 mux 不可播放直接失败。

这些阈值是首轮工程门槛，不是宣称与 FastVideo 像素级相同。对于量化和不同 fused kernel，最终是否“质量接近”必须同时看数值、感知指标和盲评样本。

### 5.3 性能门禁

以同机、同 checkpoint、同解码器、同 prompt/seed、同 4-step recipe 的 FastVideo MLX median 为基准：

- TurboCider MLX INT6 端到端 wall time **不得慢于 FastVideo 的 1.15×**；主 case A 目标为 ≤ 1.05×，达到“接近”而非明显落后。
- denoise 阶段目标 ≤ FastVideo 的 1.10×；若 VAE/编码器差异导致总时长超过门槛，必须给出阶段归因和后续计划。
- 热 prompt-cache 命中时，conditioning 阶段不得重新加载 text encoder；端到端重复请求应显著低于冷启动，并记录 load/encode/denoise/decode/mux 分项。
- peak MLX memory 目标不高于 FastVideo 的 1.10×；36 GB 机器主 case 不得触发系统 memory pressure/swap。若为了速度临时保留大权重，必须显式标注为非低内存档。
- 任何优化不得通过减少 steps、跳过音频、切换 TAEH3、降低目标分辨率或改变 seed 来通过 full-quality 门禁。

### 5.4 回归与发布门

必须加入以下自动化测试：

- `tests/native/test_h3_mlx_modelscope.py` 与 `test_h3_mlx_prepare.py`：ModelScope-only 下载合同、manifest/schema、量化能力、step ladder、错误信息。
- `tests/native/h3_mlx_geometry_test.cpp` 与 validation probes：tiny fixture 逐算子/逐 step parity。
- `tests/native/test_h3_mlx_cache.py`：prompt cache fingerprint、原子写入、损坏恢复。
- `tests/native/test_contract.py` 与 `test_h3_mlx_source_contract.py`：请求校验、backend selection、fallback、noise/tensor dump、cancel、unload/reload。
- `tests/native/test_h3_mlx_media.py`：帧/音频/PTS/mux smoke；有模型时运行真实 fixture，无模型时明确 skip。
- `tools/native/benchmark_h3_mlx.py`：同条件 FastVideo/TurboCider ABBA、原始日志、JSON、media probe、质量诊断，禁止人工抄数；默认不下载任何资产。

发布前必须同时满足：C/Metal 旧路径回归不退化；MLX INT6 主矩阵质量门禁通过；性能门禁通过；所有模型/转换器/MLX 版本和 checkpoint hash 可复现；报告和示例命令不包含个人绝对路径、模型权重或凭据。

## 6. 风险、回退与决策点

1. **INT6 API/硬件差异**：MLX 版本可能不支持某种 affine dispatch。启动时 probe；不支持则拒绝 profile，不静默转换。
2. **FastH3 权重 provenance**：普通 H3 checkpoint 不能套用 FastH3 四步 schedule。manifest 必须绑定 student checkpoint 与 schedule。
3. **VAE 成为瓶颈**：先保留 full VAE 质量档；TAEH3 只作为显式 preview。可独立优化 tiled decode，不改变 DiT 质量结论。
4. **统一内存压力**：严格 phase release；必要时复用已有 streamed/pinned policy。请求预算是 working-set target，不冒充进程 RSS 硬上限。
5. **C/MLX 双实现漂移**：C/Metal 保持 reference/fallback；所有 H3 数学修正先在 Python reference + C++ tiny fixture 验证，再改生产路径。
6. **性能不达标**：按阶段回退到可测的混合策略（例如 MLX DiT + 现有媒体、或 C/Metal profile），但只有完整测试矩阵通过后才能改变默认推荐。

## 7. 交付物清单

- C++ MLX H3 runtime、checkpoint converter/validator、manifest schema。
- `FastH3 MLX INT6` CLI/API/App profile 与可解释 telemetry。
- tiny fixture、真实 checkpoint hash 清单、逐算子 parity 报告。
- A–E 测试矩阵的原始 JSON、性能表、内存表、质量指标、代表性 MP4/WAV（大媒体只保留在本机输出目录）。
- 使用说明：模型下载/镜像、`setproxy` 依赖安装、INT6 转换、冷/热 benchmark、full VAE/TAEH3 区别和故障排查。
- 发布结论必须明确写出：通过的机器和 case、未覆盖的操作（I2V/Ref/LoRA/VSA 等）、与 FastVideo 的版本/commit/checkpoint 对应关系，以及是否达到“性能不明显差、质量接近”。

## 参考资料

- Hao AI Lab：FastH3 本地 Mac/MLX 方案：<https://haoailab.com/blogs/fasth3-local/>。
- FastVideo MLX H3 实现：`references/FastVideo/fastvideo/mlx_runtime/minimax_h3.py`、`minimax_h3_pipeline.py`、`minimax_h3_vsa.py`。
- FastVideo checkpoint 转换：`references/FastVideo/scripts/checkpoint_conversion/convert_minimax_h3_mlx.py`。
- FastVideo 入口和参数：`references/FastVideo/examples/inference/basic/mlx_fasth3.py`、`docs/cookbook/minimax-h3.md`。
- TurboCider 当前 MLX 后端：`native/backends/mlx.*`、`tools/native/build.sh`；当前 H3 C/Metal runtime：`native/models/h3_runtime/`；共享驻留与现有验收边界：`docs/design/native-cpp-engine.md`、`docs/design/video-model-acceptance.md`、`docs/design/quantized-streaming-vpipe-comparison.md`。
