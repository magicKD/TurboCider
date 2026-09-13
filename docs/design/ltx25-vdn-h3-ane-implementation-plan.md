# LTX 2.5、VDN H3 与 ANE 加速实施及验收方案

**状态日期：2026-09-13**  
**目标平台：Apple Silicon macOS，优先 M4 Max 64 GiB，同时覆盖 36/48 GiB 内存档位**

## 1. 交付目标和不可缩减的门槛

本文同时约束三条工作线：

1. LTX 2.5 以端到端速度为第一选择标准：C/Metal、C++/MLX、MPSGraph 或
   Core ML/ANE 均可使用，不要求为了“MLX 化”而牺牲速度。当前生产实现冻结为
   C/Metal Fast A/V；C++/MLX 保留为显式实验路径，只有在相同模型、输入、shape、
   steps 和质量配置下不慢于 C/Metal 时才允许替换。组件/block loading/offloading
   只在真实生产 shape 上带来净收益时纳入默认路径。
2. 新增独立的 `minimax-h3-vdn` profile，移植 OpenVDN stage-DMD v2 的
   windowed-softmax + bidirectional Video DeltaNet，不把 `references/vpipe` 变成
   运行时或发行依赖。最终 960×544×124、6 steps 的匹配条件下，TurboCider
   median wall time 不得高于 vpipe；视频和音频质量接近即可，不要求 bit-exact。
3. LTX 2.5 的正式 GPU 路径必须同时支持文生视频、单首帧图生视频和音频输出；
   GPU 达到稳定性能后，再探索 LTX 2.5 和 VDN H3 的 ANE 分区。LTX 的最低成功
   门为完整 denoise ABBA `GPU / GPU+ANE >= 1.10`，VDN 仍以 `1.20` 为推荐门。
   ANE 不要求 bit-exact，但必须通过多帧感知、运动稳定性和人工 A/B 质量门；
   microbenchmark 或单个 MLP 的加速不能替代完整 denoise 结果。

旧 profile 必须继续存在且语义不变：

- `ltx-2.5-distilled`
- `minimax-h3-turbo`
- `minimax-h3-fasth3-mlx-int6`
- `minimax-h3-fasth3-mlx-int6-vsa`

VDN 不能通过环境变量或模型目录内容静默替换其中任何一条路径。

## 2. 当前证据与已知边界

### 2.1 LTX 2.5

当前 LTX 不是纯 C runtime：Transformer/conditioning/connector 仍以 C 为主，
Objective-C/Metal 层包含 MPSGraph、Metal kernel 和 Core ML/ANE，C++/MLX 已负责
upsampler、video/audio VAE、vocoder 与 BWE。Transformer 已有最多三槽的
look-ahead block streaming。

现有验证 `validation/ltx-streamed-lookahead-2026-09-09.json` 表明，小 shape 下：

- 12 GiB budget 相对旧双槽 denoise 约 1.34×；
- 16 GiB budget 相对旧双槽 denoise 约 1.23×；
- 历史 ANE MLP 候选 warm denoise 约 0.896×，并且 latent/RGB 质量门失败。

所以 LTX 下一步不是重新发明一个 loader，而是稳定 production shape 的 warm
residency、消除 block materialization 停顿，并选择足够粗粒度的 C++/MLX 或 ANE
分区。现有 ANE MLP 路径不得作为 1.2× 已达成的证据。

当前 C++/MLX Transformer 已能完成 Stage 1、latent upsample、Stage 2、video VAE
和 MP4 导出，但仍通过 `TURBOCIDER_LTX_MLX=1` 显式启用，默认 profile 仍是
C/Metal。匹配 `704×448×97、11 steps、seed 42、GPU、video-only` 的最新单次记录
为：C/Metal denoise `55.85 s`，MLX dense-ConvRot denoise 约 `68.80 s`，约慢
`23.2%`；可选的 MLX Metal ConvRot radix-4 butterfly 将 denoise 降到 `68.06 s`，
约比 MLX dense 快 `1.1%`，仍比 C/Metal 慢约 `21.9%`。因此该优化目前是实验
候选，不能宣称 LTX 已达到“不慢于当前路径”的性能门。

为定位数值漂移，MLX 和 C/Metal 现在都支持显式逐 scheduler step 的 BF16 dump。
同一输入的初始 latent 完全一致，MLX Stage 1 前 4 步仍在 `0.1%–0.3%`
relative L2，后段放大到 `21.5%`，Stage 2 最终到 `32.8%`。这说明首因不是初始
RNG 或 upsample，而是低 sigma 下多 block 数值/attention 误差累积。单 block parity
仍接近（video relative L2 约 `0.00485`），所以后续优先检查 long-sequence video
self-attention、SOL/tiled attention 等实现差异。

MLX block cache 已从纯 LRU 改为固定 prefix 加 refill slot；容量 4 的实跑记录为
`pinned_blocks=3、refill_slots=1、cache_loads=498、cache_hits=30、evictions=494`，
峰值工作集约 `2.60 GiB`。telemetry 现在报告真实加载字节数和加载耗时。预算推导
也已改用实测单 block 约 `436 MiB` 的保守 `512 MiB` 单位，避免 12 GiB 请求错误
地选择 48 blocks 全驻留。该路径能够在受限内存运行，但容量很小时加载成本仍高，
尚未达到生产性能门。

#### 2026-09-13 实机复验与新增结果

本轮使用同一份 TurboCider native binary、M4 Max、`704×448×97`、11 steps、seed
42、相同 prompt，分别运行两次并取第二次 warm request：

| 路径 | warm denoise | warm request | 结论 |
|---|---:|---:|---|
| C/Metal Fast A/V | 52.3528 s | 67.4933 s | 当前生产基线 |
| C/Metal + 48-block ANE MLP INT8 | 45.7609 s | 65.6461 s | denoise `1.1441×`，但质量门失败 |
| C/Metal + 48-block ANE MLP FP16 | 58.7807 s | 84.9235 s | 慢于 GPU，不保留为性能路径 |

INT8 ANE 的 Stage 1/Stage 2 分别为 `19.6300/26.1309 s`，速度目标（denoise
至少 `1.10×`）已达到，但 Stage-2 latent 相对纯 GPU 的 `rel-L2=0.3306`、
`cosine=0.9451`，逐帧 RGB 平均 correlation `0.8586`，未通过严格质量门。FP16
只把全量结果改善到 `rel-L2=0.2995`、`cosine=0.9550`，并仍比 GPU 慢，因此
ANE MLP 暂时保持显式实验候选，`auto` 继续选择 C/Metal。

分阶段实验进一步定位了风险：仅启用 Stage 2 INT8 ANE 时 Stage 1 latent
byte-exact，最终 Stage-2 latent `rel-L2=0.1066/cosine=0.9943`；仅启用 Stage 1
时最终 `rel-L2=0.3374/cosine=0.9429`。主要问题是 ANE MLP partial join 与低
sigma 多 block 累积，而不是 RNG 或 upsample。后续应先做 block-level MLP output
parity，再考虑任何默认化。

Video VAE tiled decode 已进入新 binary，默认仅对大 latent 空间自动启用；当前默认
tile 为 512 像素（可通过环境变量显式比较 768 像素），开关为
`TURBOCIDER_LTX_VAE_TILED`、`TURBOCIDER_LTX_VAE_TILE_PIXELS` 和
`TURBOCIDER_LTX_VAE_TILE_OVERLAP_PIXELS`。同一 5 秒 720p、121 帧、带音频请求的
结果如下：

| 指标 | 原始 VAE | tiled VAE |
|---|---:|---:|
| request wall | 635.72 s | 249.05 s |
| denoise | 194.50 s | 194.43 s |
| Video VAE | 412.94 s | 26.15 s |
| 总体加速 | — | `2.553×` |

输出为 `1280×704`、`121` 帧、24 fps、H.264；音频为 48 kHz stereo AAC，音视频
时长均为 `5.041667 s`。新旧 tiled/untiled 输出来自同一个 Stage-2 latent，整段
比较为 PSNR `41.99 dB`、SSIM `0.9873`，因此该优化可以进入 GPU 生产路径；MP4
只作为本机验收附件，不应提交到 Git。详细原始摘要见
`validation/ltx-ane-and-720p-tiled-2026-09-13.json`。

在同一 Stage-2 latent 上，768 像素 tile 相对 512 像素 tile 的独立 Video VAE
decode 为 `10.11 s` 对 `12.55 s`（`1.241×`），逐帧质量门为 mean correlation
`0.999101`、minimum correlation `0.998747`、mean MAE `1.559/255`、最大运动
能量相对误差 `1.12%`，通过 `video_quality_gate.py`。但是同一 native binary 的
完整 5 秒 720p 带音频对照为：512 tile 总墙钟 `249.86 s`、VAE `26.85 s`；
768 tile 总墙钟 `250.32 s`、VAE `27.61 s`。因此 512 仍是生产默认，768 只保留
为显式实验值，避免把独立解码微基准外推到 helper + 媒体链路。

#### 2026-09-13 当前生产路径复验（重建 binary）

本轮清理已拒绝的融合 FFN 候选后，用同一份重建的
`build/native/libturbocider.dylib` 重新验证 C/Metal 主线。结论是：当前正式后端
仍为 C/Metal Fast A/V；C++/MLX Transformer 继续作为显式实验后端，不能因为“使用
MLX”而替换更快的实现。

| 请求 | 规格 | warm/request wall | denoise | 媒体结果 |
|---|---|---:|---:|---|
| 文生视频 | 768×448×121、11 steps、seed 42、component_staged | `85.60 s`（首请求 `115.50 s`） | `68.73 s` | 5.0417 s、H.264 + 48 kHz stereo AAC |
| 图生视频 | 768×448×97、11 steps、seed 42、首帧强度 0.65 | `99.82 s` | `57.80 s` | 4.0417 s、H.264 + 48 kHz stereo AAC |

图生视频请求真实走 `video.image`、`first_frame_vae_encode` 和 stage-specific clean
prefix；将输入首帧与输出首帧作单帧 smoke 对照得到 PSNR `32.04 dB`、SSIM `0.9720`。
这证明链路和输入条件有效，但它不是完整参考质量门，正式质量仍需固定图片集和
多 seed 评估。完整摘要见
`validation/ltx-ane-and-720p-tiled-2026-09-13.json`，本机 MP4 放在
`/private/tmp/ltx-480-output-20260913/component_staged/` 与
`/private/tmp/ltx-i2v-output-20260913/component_staged/`，不进入 Git。

本次同时删除了 `native/models/ltx_runtime/ltx_gpu.m` 中在生产 shape 上慢约 2.9%
的融合 FFN 死代码；负结果保留在
`validation/ltx-fused-ffn-abba-20260913.json`，但源码不再保留可误启用的
`#if 0` 实现。

### 2.2 FastH3 MLX

FastH3 dense/VSA INT6 profile 已实现完整 T2VA 链路：ModelScope provenance、
Qwen3-VL conditioner/cache、4-step DiT、完整 video/audio VAE 与 H.264/AAC mux。
其设计和已有验证见 `h3-mlx-int6-fastvideo-parity-plan.md`。它是 VDN 可以复用的
MLX 数学、conditioner、VAE、媒体输出和 telemetry 基础，但 checkpoint 与 schedule
不能直接复用。

### 2.3 VDN H3

参考实现固定在 `references/vpipe` commit `db8960d`。VDN 不是 FastH3/VSA 的
另一组稀疏参数，而是普通 MiniMax H3 FL2VA backbone 上的第二条 attention 分支：

- softmax 分支按 5-frame chunk、radius 1 做精确 window attention；
- linear 分支使用双向 `vdn_solve` recurrence 携带 window 的补集；
- 第一帧和最后一帧为双向 anchor，离开 linear branch 输入；
- softmax gate 在 head 维缩放归一化后的 window attention；
- linear readout 经独立 `to_out_linear` 投影，只加到 video rows。

已从 ModelScope `OpenVDN/vdn-minimax-h3` 下载最小 stage-DMD 附件，位于本机
`models/VDN-H3-ModelScope`。模型目录不进入 Git。关键资产为：

| 资产 | 字节数 | SHA-256 |
|---|---:|---|
| linear branch | 4,279,428,112 | `dec6981c7874f5b3bc92d1a02e256b673a3b3499dc1a124714bb3b19da602855` |
| Turbo adapter | 851,452,696 | `24fc93c82fe84dc45d0627f4e72c637bc387d282ba18f60ed3b7f8c81089392c` |

`tools/h3/download_vdn_modelscope.py --verify-only` 目前会验证文件大小、SHA-256、
safetensors offset、50 blocks × 16 branch tensors、726 adapter tensors、所有
shape/dtype、stage-DMD v2 配置以及 `larryvrh_v4_step600_ema` adapter family。

普通 FL2VA base 另有 ModelScope-only 下载器
`tools/h3/download_h3_fl2va_modelscope.py`。它固定到
`MiniMax/MiniMax-H3` 的 `FL2VA/transformer`，只取 config、index 和 13 个 BF16
shard（约 61.7 GiB），逐文件校验 ModelScope 发布的大小和 SHA-256，并使用可续传
`.part` 文件，避免快照缓存造成第二份副本。当前本机 Comfy INT8 ConvRot pruned
checkpoint 和已合并 LightX2V checkpoint 都不能替代这份普通 base；在正式 converter
证明一种等价的 AdaLN+adapter 合成方法前，禁止用它们生成可执行 VDN manifest。

兼容性边界必须显式处理：当前 `minimax-h3-turbo` 是 LightX2V 4-step v0.1、
strength 0.0625 的预合并模型；VDN stage-DMD 自带另一套 Turbo adapter，而 vpipe
匹配基准使用 6 steps。二者不能直接叠加后宣称兼容。因此 VDN 必须有自己的
base/adapter/schedule manifest。

### 2.4 2026-09-12 MLX/Metal 复验结论（当前实现基线）

本机 MLX Python 包与 PyPI 在 **2026 年 9 月 12 日** 均为 `0.32.2`；没有更高版本
可用于替换当前依赖。MLX 的 GPU 路径仍不能直接执行 VDN 需要的
`cholesky/inv/solve/eigh`，因此当前方案保留自有的 MLX runtime Metal kernels：
panel Cholesky、triangular inverse，以及其后的普通 MLX matmul state scan。不要把
这个结论写成“MLX 没有 GPU”，准确说法是“MLX GPU 可用，但这些 LAPACK 类 primitive
没有可用的 Metal 实现”。

数值 probe 覆盖 `d = 7,31,32,33,64,128`，每个维度 3 个矩阵，GPU solve 相对 CPU
oracle 的最新结果为：最大绝对误差 `7.15e-7`、relative RMSE `2.16e-7`、最大残差
`1.31e-6`。这组结果是 GPU solve 的准入门，不替代端到端质量门。

VDN recurrence 必须严格遵循：

```text
S_out = (S_in · Diag(alpha) + B) · (I + A)^-1
transition = Diag(alpha) · inverse
```

在 MLX 张量 `[frames, heads, dim]` 与 `[frames, heads, dim, dim]` 上，
`expand_dims(alpha, 3) * inverse` 才是按 inverse 行缩放；
`expand_dims(alpha, 2) * inverse` 会变成 `inverse · Diag(alpha)`，必须由 source
contract test 拒绝。

阶段 profiling 通过 `TURBOCIDER_VDN_PROFILE=1` 显式开启，默认生成路径不插入同步。
22 帧 block-0 观测显示：window softmax、base QKV/FFN 和 branch readout 均可分别
量化，solve 本身不是当前第一热点；下一轮应优先移植 block-span online-softmax，
 再评估 base projection/FFN 的 graph 融合，不能先花时间重写已验证 solve。

本轮对两个 VDN 实验候选做了真实 GPU AB：MMA block-span attention 在
`384×384×22、6 steps` 上为 `25.8885 s`，当前 MLX SDPA/reference 路径为
`25.3664 s`，慢约 `2.1%`，且峰值显存略增；更小的 attention-only probe 也没有
出现反转。因此 `TURBOCIDER_VDN_EXPERIMENTAL_MMA_SPANS=1` 继续保持显式实验开关。
Affine INT6 的 DQ-GEMM crossover sweep（`TURBOCIDER_VDN_DQ_GEMM_MIN_ROWS`）在
同一负载上测得默认 `768` 为 `25.3664 s`，全量 quantized matmul 的 `0/2048` 档为
`25.7931/25.5100 s`；默认阈值暂不调整。为内存优先场景保留
`TURBOCIDER_VDN_COMPACT_LINEAR_OUTPUT=1`，它可少约 20 MB 临时显存，但本负载慢约
`2.2%`，不能作为默认性能路径。

2026-09-12 又加入了固定 window query/key index 的 MLX tensor cache，并保留
`TURBOCIDER_VDN_DISABLE_WINDOW_INDEX_CACHE=1` 作为严格 A/B 诊断开关。`384×384×22、6
steps、seed 42` 的五次启用中位数为 `24.7363 s`，四次禁用中位数为 `24.7400 s`，
四个输出 tensor 均逐元素完全一致，峰值内存也相同。该改动降低 host 侧重复构造，
但 GPU wall 收益低于噪声范围，不能作为 VDN 已达标的性能证据；详细数据见
`validation/vdn-h3-attention-candidates-2026-09-12.json`。

## 3. 目标架构

```text
Request / profile=minimax-h3-vdn
              │
              ├── ModelScope VDN stage manifest validator
              ├── FL2VA base + exact Turbo adapter contract
              ├── shared Qwen3-VL conditioner / prompt cache
              │
              └── C++/MLX H3 denoise pipeline
                    │
                    ├── base block weights (INT6 or validated INT8 route)
                    ├── one VDN branch block (~85 MiB BF16 source)
                    ├── raw Q/K/V ──┬── windowed softmax + softmax gate
                    │               └── VDN linear features / solve / readout
                    ├── two output projections + video-row residual add
                    └── block eviction / next-block prefetch
              │
              ├── shared full H3 video VAE
              ├── shared full H3 audio VAE
              └── native H.264/AAC mux → MP4
```

模型层仅依赖 TurboCider 自己的接口和随发行包携带的第三方许可证。可以阅读、
移植并重写 Apache-2.0 的 vpipe 语义与 kernels，但源码中不得出现 `../references`
路径，打包产物不得链接或调用 vpipe binary。

## 4. VDN 分阶段实现

### 4.1 WP-V0：资产和算法合同（已完成基础）

- ModelScope-only、`.part` 续传、固定大小和 SHA-256；
- 严格拒绝未知 `hybrid_attention` version、delta rule、bridge、anchor、conv targets；
- branch 必须为 50×16 BF16 张量；
- adapter 必须为 363 exact targets、726 BF16 张量；
- base config hash 必须为
  `fa5c3b3b3bc9f2e604f7a790d5a67369b975daeea3f0db7a6ceadbd5315cd3d0`；
- 纯 C++ `vdn_window_bounds`、anchor mask、gather、alpha bridge、Cholesky
  `vdn_solve` CPU oracle 已加入 tiny test。

### 4.2 WP-V1：独立 profile 与 checkpoint manifest

新增 `minimax-h3-vdn`，首个可验收范围固定为：

| 字段 | 合同 |
|---|---|
| operation | `video.generate` |
| input | text only |
| output | H.264/AAC MP4 |
| width/height | 32 的倍数；正式基准 960×544 |
| frames | `5+17n`；正式基准 124 |
| fps | 24 |
| steps | 首发固定 6；不得复用 FastH3 4-step ladder |
| video/audio shift | 12 / 3 |
| residency | `component_staged` 或 `streamed` |
| branch | stage-DMD step 250 |
| adapter | `larryvrh_v4_step600_ema`, scale 1 |

准备工具输出 `mlx_h3_vdn.json`，至少记录 base content hash、base tensor layout、
量化格式、adapter hash、branch hash、6-step AdaLN ladder、代码版本、MLX 版本和
完整 tensor key 集。只要任一 identity 不匹配，runtime 必须拒绝，不能退回 FastH3
或普通 H3。

### 4.3 WP-V2：base 与 adapter 准备

优先路线为把经验证的普通 FL2VA base 转为 affine INT6/g64，并在转换时合并
VDN 自带 Turbo adapter。转换必须逐 tensor/逐 block 流式执行：

1. 读取一个 base projection；
2. 读取对应 LoRA A/B，按 `alpha/rank` 合并；
3. 在 FP32 累加后量化为 INT6/g64；
4. 直接写最终 artifact，随后释放临时 tensor；
5. AdaLN 只为固定 6-step 双 schedule 烘焙 cache，不保留巨大 AdaLN weight；
6. 完成内容 hash 和 tiny projection parity 后才允许删除源副本。

若已有 Comfy INT8 base 被使用，必须显式记录 fused QKV 是 flat
`[all-q|all-k|all-v]`，并在合并 split LoRA 时重排；不能根据 tensor shape 猜布局。

量化验收先做 20 个 projection 的随机采样：FP32 merged oracle 与 INT6 输出
`cosine >= 0.995`，相对 L2 及最大误差进入报告。随后做 block-0 和完整 step-0
latent parity。若 INT6 质量无法过门，允许建立独立 INT8 候选，但不得让 benchmark
双方使用不同精度后仍声称等条件。

### 4.4 WP-V3：windowed softmax

先做可读的 MLX reference route，再做 production Metal route：

- 非 video rows 作为 global prefix，双向保持 dense；
- video query 只读取其 chunk window 与首尾 anchor frames；
- mask 以 frame spans/block spans 表示，禁止构造 `[seq, seq]` dense mask；
- 先对 2–12 latent frames 做 MLX/CPU/vpipe oracle parity；
- production 使用 block-sparse online softmax kernel；Q/K 仍使用 backbone 的
  QK RMSNorm 与 RoPE；
- softmax gate 为 `sigmoid(x @ W^T + b)`，按 head 广播，在 out projection 前应用。

### 4.5 WP-V4：linear branch

每个 block 在 QK norm/RoPE 改写 Q/K 前消费 raw Q/K/V：

1. 对 video K/V 做空间 5×5、时间 5 的 depthwise short-conv；
2. Q/K SiLU 后按 head L2 normalize，V 只做 SiLU；
3. `beta = sigmoid(beta_proj(x))`；
4. 每 frame/head 计算并对称化
   `A=sum(beta*k*k^T)`、`B=sum(beta*v*k^T)`；
5. FP32 计算
   `alpha=exp(-exp(A_log)*softplus(up(down(mean(x)))+dt_bias))`；
6. text rows 作为一个 delta chunk，得到两方向共享的 0.5× text state；
7. 前向/反向 `vdn_solve`，gather window 外最近状态并做 alpha bridge；
8. `q·state`、RMSNorm、output gate；
9. `to_out_linear(readout)` 仅加到 video rows。

production kernels 必须让 A/B/alpha/solve 保持 FP32；plain projections 和最终
readout projection可以用 BF16/INT6。首尾 anchor 的 linear readout 必须精确为零。

### 4.6 WP-V5：block streaming 与内存复用

VDN branch 单 block 约 85 MiB，必须跟随主 DiT 的 residency 决策：

- `resident`：base 与 branch 常驻，仅用于内存充足档位；
- `component_staged`：conditioner → denoise → VAE 分阶段独占；
- `streamed`：当前 block 计算时预取下一个 base+branch block；本 block完成后一起
  释放；
- 最多三槽 look-ahead，内存预算不足时降为双槽或单槽；
- QKV、attention、FFN、VDN A/B/state/readout scratch 做生命周期别名，禁止按峰值
  分别永久分配；
- 记录 `base_read_seconds`、`branch_read_seconds`、stall、prefetch hit、peak RSS、
  MLX active/peak 和 SSD bytes。

自动策略以可用内存而不是机器型号为唯一输入。任何降级只改变 residency，不改变
checkpoint、steps 或数学路径。

## 5. LTX 2.5 实现工作包

### 5.1 保持现有路径为基线

冻结当前 `ltx-2.5-distilled` 作为 A/B 基线，并保存：binary hash、model hash、
profile、shape、steps、seed、prompt、环境温度与内存压力。新实现先注册为显式
候选，未过门前不能替换当前 profile。

### 5.2 生产 shape 的 residency 稳定化

- 将已有 look-ahead planner 用于 Transformer block、upsampler、video/audio VAE；
- warm run 不重复解析 index、编译 Metal graph 或 materialize 不变 tensor；
- block prefetch 与当前 GPU command buffer 重叠，ABBA 中单独记录 SSD stall；
- 在 12/16/24 GiB budget 下分别验证单/双/三槽，选择最快且不过预算的方案；
- phase 结束时主动释放不可复用的大 tensor，同时保留可复用的编译 graph、tokenizer
  和小型 schedule cache。

### 5.3 C++/MLX 热点迁移顺序

使用完整 denoise profile，而不是代码语言作为选择标准：

1. checkpoint materialization 与 block prefetch；
2. Transformer linear/SwiGLU；
3. attention；
4. 只有 profile 显示收益时再迁移 elementwise/AdaLN。

每个候选都做 matched latent/media 与 ABBA。只要 wall median 变慢超过 2%，立即
回退该候选，不把更多迁移当作补救。

当前已落地但仍为 opt-in 的 kernel 候选：MLX ConvRot 256 维变换可使用
`TURBOCIDER_LTX_MLX_METAL_CONVROT=1` 的 radix-4 butterfly Metal kernel，避免每个
projection 构造 256×256 dense Hadamard matmul；RoPE 也有独立的
`TURBOCIDER_LTX_MLX_METAL_ROPE=1` paired kernel。两者均保留 eager/reference 回退，
必须分别通过 block parity、latent quality 和 ABBA 后才可成为默认路径。

本轮对“长视频行数达到门槛时先将 ConvRot Q8 权重反量化为 BF16，再执行 dense
GEMM”的候选做了同一构建的交错 ABBA。候选只作用于 video self-attention/FFN，
避免固定长度文本上下文污染小 shape；正式负载为 `704×448×97、11 steps、seed 42`，
component-staged、GPU、MLX + Metal ConvRot，4 个 baseline 与 4 个候选样本。基线
denoise median 为 `66.4894 s`，候选为 `65.8415 s`，denoise speedup 仅
`1.00984×`（约 `0.97%`），request wall speedup 为 `1.00928×`；4 个候选 latent
均与基线逐字节一致。由于收益低于 2% 的保留门槛，DQ-GEMM 开关已撤回，不进入
生产路径；完整原始数据见
`validation/ltx-dq-gemm-abba-2026-09-12.json`。

本轮还验证了“把 Q/K/V 的 affine-Q8 行拼成一次大 matmul”的候选。它在单个小尺寸
运行上看似略快，但生产尺寸的 Stage 2 latent relative L2 约 `0.67`，与逐投影路径
不具备数值等价性，候选已撤回，代码中不保留该开关。任何后续 QKV 融合必须先通过
同一输入、同一 block 的逐投影 parity，再进入端到端性能测试。

另外对 LTX 真实 video projection/FFN 形状做了 MLX 0.32.2 的 affine-Q8 group
size sweep。g128 在 6 个形状中的 5 个最快，跨 g32/g64/g128 的 checksum 和
逐元素输出一致；但正式 `704×448×97` ABBA 中，g64 denoise median `66.5785 s`，
g128 `66.5092 s`，仅 `1.00104×`（约 `0.10%`），因此不把 g128 设为默认。保留
`TURBOCIDER_LTX_MLX_CONVROT_GROUP_SIZE` 作为显式诊断/设备调优开关，默认仍为
g64；microbenchmark 与 ABBA 记录见
`validation/ltx-qmm-group-sweep-2026-09-12.json` 和
`validation/ltx-convrot-group-abba-2026-09-12.json`。

调试开关 `TURBOCIDER_LTX_PROFILE_BLOCK`、`TURBOCIDER_LTX_PROFILE_INVOCATION`
可输出 video self-attention、text cross-attention、AV cross 和 FFN 的 phase timing；
默认关闭，不在正常 denoise 中插入同步。`TURBOCIDER_LTX_MLX_DUMP_STEPS` 与
`TURBOCIDER_LTX_C_DUMP_STEPS` 用于导出逐步 BF16 latent，不能在性能基准中开启。

## 6. ANE 探索

### 6.1 分区原则

ANE 只适合形状稳定、计算足够粗、输入/输出拷贝相对小的分区。GPU 与 ANE 必须
真正重叠；串行地把一个 GPU GEMM搬到 ANE，通常会被 Core ML dispatch 和复制开销
抵消。

### 6.2 VDN 优先候选

按风险和潜在重叠排序：

1. `alpha.down/up + beta + output gates`：与 GPU window softmax 可并行；
2. K/V short-conv：与 QK norm/RoPE 或 softmax准备重叠；
3. A/B frame statistics 或 128×128 solve：仅当固定 geometry 的 Core ML graph
   能留在 ANE 且 FP32 语义可满足质量门时尝试；
4. `to_out_linear`：输出大，默认不优先。

每个 Core ML artifact 必须绑定 geometry bucket、weights hash、compute units 与
I/O dtype。首轮只做 960×544×124；达到门槛后再扩 bucket。

### 6.3 LTX 候选

现有 MLP-only INT8 路径已达到 denoise `1.144×`，但质量失败；FP16 路径质量仍
未达门且速度回退。因此它们都不能作为默认后端。下一轮优先：

- 双流或可独立的粗粒度 branch；
- 连续多个 linear + activation 的单 artifact；
- GPU 主干继续运行时 ANE 能并行执行的区段。

禁止把 warm microbenchmark 的加速外推到完整 denoise；LTX 的最低探索门是完整
denoise `≥1.10×`，质量和内存门必须同时通过，`1.20×` 作为推荐交付门。
由于当前 Stage-2-only INT8 ANE 的完整 97 帧视频检查为 warm denoise 约
`1.042×`、mean RGB correlation `0.9741`、mean MAE `8.26/255`，它在感知
质量上比全量 ANE 明显更接近 GPU，但还没有达到用户要求的 `1.10×`，所以继续
保持实验候选；全量 48-block INT8 虽达到 `1.144×`，但 mean correlation 仅
`0.8586`、mean MAE `20.51/255`，不得默认启用。
连续后 24 block（Stage 1/Stage 2 均启用，denoise `50.23 s`）的结果为 mean
correlation `0.8636`、mean MAE `20.52/255`，且相对 GPU 仅 `1.042×`；说明误差
主要来自 ANE 近似算子而非简单的 block 数量，后段 block-only 分区也拒绝。

## 7. 分层验收

### 7.1 L0：静态与构建

- repository independence：源码和产物不含 `../references`、`references/vpipe`；
- ModelScope manifest、SHA、shape、dtype、adapter family 全部通过；
- Release build 无 warning/error；
- 旧 profile contract tests 不变。

### 7.2 L1：算子 parity

VDN tiny fixture 至少覆盖：

- chunk bounds 和 anchor mask；
- short-conv 边界；
- beta、A/B 对称化、alpha；
- text state；
- forward/reverse solve；
- alpha bridge 与 gather；
- window softmax、softmax gate、linear readout；
- block-0 最终 residual。

CPU oracle/MLX/Metal 三者建议门：FP32 `max_abs <= 1e-4`；BF16 路径
`max_abs <= 2e-2`、`mean_abs <= 2e-3`。超过门槛必须定位，不允许只看最终视频。

### 7.3 L2：端到端 smoke

按顺序运行：

1. 22 帧、低分辨率、6 steps，保存 tensor dump；
2. 56 帧，验证 window 不再 full-cover；
3. 124 帧 960×544，生成带音频 MP4。

每个输出使用 `ffprobe` 验证帧数、24 fps、H.264、AAC、32 kHz stereo、PTS
单调。输出统一放到 `outputs/vdn-h3-validation-<date>/`，同时保存请求、日志、
telemetry JSON 和 SHA-256。不能只保存 MP4。

2026-09-12 已实际完成 22/56 帧修正版 smoke，并另行启动 124 帧修正版 smoke。旧的
`h3-vdn-22-gpu-solve.mp4`、`h3-vdn-56.mp4`、`h3-vdn-124.mp4` 是 alpha 广播轴修正
前的历史产物，不再作为质量基线；验收记录只引用带 `-gpu-fixed` 的新输出。22 帧
GPU solve 与 CPU solve 对照为 `29.99 dB / 0.9625 SSIM`，56 帧相对历史候选为
`25.86 dB / 0.9235 SSIM`，这两个数字只用于诊断，不能替代 matched vpipe A/B。

### 7.4 L2-App：App/SDK 路由验收

App 的 LTX 验收必须与 native CLI 使用同一请求语义：

- 文生视频：文字输入、11 steps、24 fps、480p/720p shape、可选音频；
- 图生视频：一张 `first_frame` 图片、strength 在 `[0,1]`、同样的音频和导出；
- 后端选择：`Auto` 明确解析为 C/Metal，`C/Metal` 可显式选择，`C++/MLX`
  只作为实验入口；图生视频选择 C++/MLX 必须给出明确错误；
- 输出历史、任务持久化、失败状态和媒体路径必须与其他模型一致。

2026-09-13 已完成 Swift App 和集成测试目标的重建；
`turbocider-studio-tests` 与 `turbocider-model-library-tests` 通过，覆盖 LTX
文生/图生能力、输入角色和默认值。首次运行的废纸篓删除检查受 sandbox 拒绝，
在测试临时目录允许外部权限后通过；`turbocider-studio-controls-tests` 未运行，
因为该目标需要独立的 LoRA 配置 fixture，与本轮 LTX 功能无关。

### 7.5 L3：性能 ABBA

同一台机器、同一模型内容、同一 prompt/seed、960×544×124、6 steps：

```text
warmup: vpipe, TurboCider
round 1: A(vpipe), B(TurboCider), B, A
round 2: B, A, A, B
round 3: A, B, B, A
```

记录端到端 wall、denoise、conditioner、base/branch I/O、VAE、mux、peak RSS、
MLX peak、SSD bytes 和 thermal state。主门槛：

- `median(TurboCider wall) <= median(vpipe wall) * 1.00`；
- p90 不超过 vpipe p90 的 1.05；
- denoise 不得靠跳过 audio/VAE 或改变 steps 获胜；
- 若模型量化不同，结果只能标记为探索，不能标记 matched acceptance。

当前修正版单次 124 帧记录为 denoise `528.34 s`、端到端 `624.90 s`、MLX peak
`29.55 GB`。参考资料中另一台 M5 Pro 的 vpipe hybrid 记录约 `278 s`，硬件、缓存、
编译和模型 artifact 尚未 matched，因此这里只能记录为“仍有约 2× 级别的探索缺口”，
不能宣称 TurboCider 已通过或精确失败于 vpipe ABBA 门。

LTX 以同样 ABBA 比较“当前稳定 profile”和“新候选”，主门槛为 candidate median
不高于 current；2%以内需扩大样本判断，超过 2%直接失败。最新 DQ-GEMM 负结果已
扩大到 4+4 样本，但只有 `0.97%` 收益，因此不能作为性能交付证据。相对于 C/Metal
的正式差距仍需以同条件 C/Metal 基准补齐；此前单次定位结果显示 MLX 仍慢约
20%–23%。质量回归报告 `mean correlation=0.8668、mean cosine=0.9798、mean
MAE=20.05/255、passed=false`；这只是定位证据，不是验收通过。

### 7.6 L4：质量

固定至少 12 个 prompt：人物、快速运动、镜头运动、动物、自然、城市、文字/标识、
低光、复杂纹理、动画风格、长景深和音画事件。每个 prompt 固定 2 个 seed。

自动指标同时看：

- raw RGB PSNR/SSIM（仅 matched checkpoint 时解释）；
- LPIPS/感知特征距离；
- optical-flow magnitude/distribution、flicker、warp error；
- CLIP/VideoCLIP prompt alignment；
- audio RMS、谱分布、静音比例、音视频时长差。

最终做双盲 A/B，至少 3 人，比较 prompt adherence、主体稳定、运动自然、细节、
闪烁和音画一致性。VDN 只要求与 vpipe 接近：多数维度不得出现统计显著且明显的
退化；任何严重 artifact 即失败。

### 7.7 L5：ANE 完整 denoise 门

同一 candidate 做 GPU 与 GPU+ANE ABBA，质量门通过后计算：

```text
speedup = median(GPU denoise) / median(GPU+ANE denoise)
```

LTX 的用户目标为 speedup `≥1.10`；VDN 推荐门为 `≥1.20`。不要求 ANE 与 GPU
bit-exact，但 p90 不得明显倒退、peak memory 不得破坏目标档位，并且必须同时通过
shape/finite、逐帧感知、运动稳定性、音频完整性和人工双盲 A/B。报告列出 Core ML
load/compile、输入输出 copy 和实际 overlap，避免把准备时间藏在 warmup 中。当前
全量 INT8 ANE 只通过 LTX 速度门，未通过质量门；Stage-2-only 更接近 GPU，但速度
未到 1.10×，两者均不能记为交付完成。

## 8. 磁盘与产物管理

当前 VDN 最小附件约 4.8 GiB，模型和输出都在 Git ignore 范围。删除任何大模型前：

1. 确认其不是当前 cold-start、多样本质量或回归验证的唯一来源；
2. 保留 manifest、SHA 和重新获取命令；
3. 优先删除可重建的 build/cache，再删除已完成转换且有校验证据的 BF16 源 shards；
4. 不删除用户未明确授权且仍在验收矩阵中的资产。

FastH3 VSA BF16 shards 仍可能用于 cold/multi-sample 验证；在验收范围未调整前不把
它当作无用缓存。VDN base 转换前至少保留“目标 artifact + 8 GiB 临时/系统余量”。

## 9. 完成定义

三条工作线分别完成且全部证据入库后，整个目标才完成：

- LTX 候选真实接入 TurboCider，内存档位可运行，matched ABBA 不慢于当前路径；
- `minimax-h3-vdn` 能独立生成带音频 MP4，matched ABBA 不慢于 vpipe，质量门通过；
- LTX ANE 路径完整 denoise ≥1.10×，VDN 以 ≥1.20× 为推荐门；两者都必须通过
  感知质量与媒体完整性门。若某模型经系统探索仍达不到，必须明确报告未达标，
  不能把探索本身记作成功；
- 所有输出、telemetry、原始日志、质量报告和二进制/model identity 可追溯；
- 旧 profile 与已有 FastH3 dense/VSA 回归测试保持通过。
