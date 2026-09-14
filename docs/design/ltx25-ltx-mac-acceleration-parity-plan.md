# TurboCider LTX 2.5 对齐 ltx-mac 的加速设计与验收方案

**状态日期：2026-09-13**  
**适用实现：`ltx-2.5-distilled`，Apple Silicon，C/Metal 主后端，C++/MLX 实验后端**

## 1. 结论与推荐路线

TurboCider 已经包含 `ltx-mac` 的大部分高价值底层能力。当前问题不是重新移植一套
Transformer，而是把已存在的能力整理成可理解、可复现、可回退的产品配置，并用同一
checkpoint、shape、prompt、seed 和媒体口径做严格验证。

推荐保留三层运行策略：

| 层级 | 默认性 | 计算路径 | 数学/质量性质 | 当前用途 |
|---|---|---|---|---|
| `quality` | 默认 | C/Metal dense + Fast A/V + Audio batching | 不引入新的算法近似 | T2V、I2V、带/不带音频的生产基线 |
| `quality_hybrid` | 显式 profile | C/Metal + 固定 shape INT8 ANE MLP + Stage-1 text K/V | ANE INT8 会改变生成轨迹，需感知质量门 | 480p-class 异构候选 |
| `sol` | 显式近似 | 上述 GPU 或 GPU+ANE 路径 + Stage-2 Sol | 近似 self-attention | 速度优先、短中序列候选 |
| `fast_approx` | 显式近似 | `sol` + Stage-2 text rows=256 | Sol + context pruning | 当前最激进候选，只用于通过质量套件的场景 |

当前不应把 C++/MLX Transformer 设为默认。已有相同工作负载记录中，MLX denoise 仍比
C/Metal 慢约 22%，而底层媒体组件继续使用 MLX 没有问题。选择标准是端到端速度和质量，
不是后端名称。

本轮已把 `sol`、`fast_approx`、Fast A/V 与后端选择接入 Request、C ABI 内部运行时、
Swift binding 和 App；Dense 仍是默认，近似模式必须显式设置
`allow_approximation=true`。

## 2. ltx-mac 与 TurboCider 能力对照

| ltx-mac 能力 | TurboCider 状态 | 默认/开关 | 决策 |
|---|---|---|---|
| C/Metal INT8 ConvRot Transformer | 已迁移 | `ltx_backend=c_metal/auto` | 生产主路径 |
| Video/Audio 第二 Metal queue 并行 | 已迁移 | `ltx_fast_av=true`，默认开启 | 保留；已是主要 exact 调度优化 |
| Audio self/text/FFN command batching | 已迁移 | 随 Fast A/V 开启 | 保留；不改变算术 |
| checkpoint mmap 载入后提前关闭 | 已迁移 | 运行时生命周期策略 | 保留；降低 VAE 前内存压力 |
| Stage-2 最后一步逐 block release | 已迁移 | component-staged/profile | 保留；降低最终解码干扰 |
| MLX spatial upsampler | 已迁移 | 内部媒体后端 | 保留 |
| MLX Video VAE、Audio VAE、vocoder/BWE | 已迁移 | 内部媒体后端 | 保留；支持 T2V/I2V 音视频输出 |
| 固定 shape INT8 ANE Video MLP split | 已迁移 | `gpu_ane` + manifest | 质量优先 hybrid 候选 |
| Stage-1 Video-text K/V on ANE | 已迁移 | profile `kv_stage_mask=1` | 保留；不启用 Stage-2 K/V reload |
| ANE V2A/QKV | 代码已迁移 | profile/编译候选 | 不进入推荐 profile |
| Sol tiled Video self-attention | 已迁移并公开 | `ltx_sol_stage1/2` | 仅推荐 Stage 2 |
| Stage-2 text context pruning | 已接入 | `ltx_stage2_text_rows=256` | 近似模式，默认关闭 |
| T2V | 已支持 | `video.generate` | 必须通过完整验收 |
| 单首帧 I2V clean-prefix | 已支持 | `video.image` | 仅 C/Metal；必须单独验收 |
| 720p tiled Video VAE | 已迁移 | 大 shape 自动/环境诊断 | 保留；Sol 当前不能用于 720p |
| persistent Audio worker | 已验证负结果 | 无产品开关 | 不继续投入 |
| Video attention command batch | 诊断能力 | `TURBOCIDER_LTX_VIDEO_ATTENTION_BATCH=1` | 默认关闭；Sol 启用时强制关闭 |
| Metal ConvRot MLP 分拆 | 实验代码 | `TURBOCIDER_LTX_GPU_METAL_CONVROT_MLP=1` | 704×448×97 仅约 1.00143×，不进入 profile |
| C++/MLX Transformer | 已有实验路径 | `ltx_backend=cpp_mlx` | 当前只支持 T2V；性能达标前不替换 C/Metal |

## 3. 已采用的 exact 加速

### 3.0 Video VAE 生命周期（2026-09-13 修正）

`ltx-mac` 与 TurboCider 的 MLX Video VAE 数学实现不是当前慢点；慢点来自 Transformer
父进程中的 Metal/MPSGraph/MLX allocator 和统一内存压力。TurboCider 现在对
`component_staged` 的 `video.generate` 与 `video.image` 统一使用 disposable worker + clean
`exec` finalizer，带音频请求不再走驻留 VAE 路径。

本机 704×448×97、11 steps、seed 42 的诊断数据如下：

| 路径 | Video VAE decode | 备注 |
|---|---:|---|
| `ltx-mac` clean finalizer | 约 2.77--2.95 s | 参考 |
| TurboCider spawned child（父进程仍存活） | 约 8.21 s | 仅回退/诊断 |
| TurboCider clean exec finalizer | 约 2.98--3.21 s | 默认目标 |

带音频 clean finalizer 还测得 Audio VAE/vocoder/BWE 约 0.62 s、AAC mux 约 0.04 s、
finalizer wall 约 3.88 s。Video-only 与 clean exec 的同一 latent 通过 `framemd5` 完全
一致；因此该优化释放的是进程级资源，不改变生成质量。详细生命周期、回退和门槛见
[`ltx25-vae-lifecycle-and-code-cleanup-20260913.md`](ltx25-vae-lifecycle-and-code-cleanup-20260913.md)。

### 3.1 Fast A/V 双队列

每个 LTX block 包含 Video 和 Audio 双流。以下区域存在天然并行：

```text
Video self/text  || Audio self/text
Audio -> Video   || Video -> Audio
Video FFN        || Audio FFN
```

TurboCider 在第二个 Metal context/queue 上运行 Audio 分支，buffer 仍位于统一内存。
`ltx_fast_av=false` 可得到串行诊断基线。正式基准必须同时记录
`gpu_parallel_av` 和 `gpu_batch_audio_commands`，避免把不同调度配置混在一起。

### 3.2 生命周期和 residency

生产优先级为：

1. 复制完 Transformer tensor 后关闭 checkpoint mmap；
2. ANE 完整覆盖 MLP 时，释放已被替代的 full GPU MLP；
3. Stage 2 最后一次更新后，逐 block 释放不会再使用的资源；
4. Transformer 退出关键路径后运行 MLX VAE/音频 finalizer；
5. 内存不足时才使用 block streaming；64 GiB、480p-class 不把 SSD streaming 当成加速。

streaming 的目标是避免 OOM/swap，不保证比 resident/component-staged 更快。任何 streaming
报告必须同时给出 bytes loaded、slot refills、load/wait 时间和 peak RSS。

### 3.3 Video VAE tiled decode

720p-class 必须使用 tiled Video VAE，避免完整 latent decode 的巨大中间张量。现有
1280×704×121 记录中 tiled 路径将 VAE 从约 412.94 秒降到约 26.15 秒，总请求约
2.553×；同 latent 的 PSNR 约 41.99 dB、SSIM 约 0.9873。512-pixel tile 保持默认，
768-pixel tile 虽然独立 decode micro 更快，但完整请求没有稳定改善，因此只保留诊断值。

## 4. GPU+ANE 质量优先 profile

新增 profile：

```text
profiles/ltx-ane-fast-int8-20260913.json
```

设计配置：

- fixed-shape `int8_pc` MLP，Stage 1/Stage 2 各 48 block；
- Video FFN intermediate 由 GPU 和 ANE 分片并行；
- Stage-1-only ANE Video-text K/V；
- A/V 双队列并行；
- Stage-2 artifact 预载；
- 完整 ANE 覆盖后释放 full GPU MLP；
- Stage-1/Stage-2 Core ML session 按生命周期 detach；
- 最终 step 逐 block release；
- 不启用 V2A、QKV、fused residual 或 fused AdaLN-pack。

profile 中的相对路径指向同一工作区 `ltx-mac/models/coreml` 下的本机可重建 artifact。
这些 artifact、编译后的 Core ML bundle 和模型权重不能提交 Git。正式发布应通过
TurboCider exporter 重新生成到用户缓存目录，并验证：

- source checkpoint canonical path、bytes、mtime/provenance；
- block index 0–47 完整；
- Stage-1/Stage-2 rows 与请求一致；
- variant 为 `int8_pc` 或质量诊断用 `fp16`；
- 当前设备能够加载 bundle，不出现 E5 recompile warning。

`ltx-mac` 的参考结果是 704×448×97、8+3 steps 下从 59.483 秒降到 49.467 秒，约
1.202×。该数字是目标参考，不是 TurboCider 已通过数据。TurboCider 当前复跑出现
`Must re-compile the E5 bundle`，因此 80–184 秒的请求不能用于判断算法或代码快慢。

## 5. 可选近似路径

### 5.1 Stage-2 Sol

推荐配置：

```json
{
  "allow_approximation": true,
  "ltx_backend": "c_metal",
  "ltx_sol_stage1": false,
  "ltx_sol_stage2": true,
  "ltx_sol_tau": 1.0,
  "ltx_sol_dense_edge_blocks": 1,
  "ltx_sol_dense_edge_steps": 0
}
```

语义为：Stage 1 全 dense；Stage 2 的 block 0 和 47 dense，block 1–46 使用 Sol，三个
Stage-2 diffusion step 都允许 Sol。Stage 1 不建议启用，因为它决定主体、构图和运动，
参考实验中收益约 0.5%，数值漂移却明显。

Sol route mask 目前最多支持 4096 个 video rows。704×448×97 的 Stage-2 rows=4004，
可以使用；1280×704×121 的 Stage-2 rows=11440，必须 fail closed 到 `quality`。

### 5.2 Stage-2 text256

`ltx_stage2_text_rows=256` 不是去 padding。connector 的 1024 rows 中，后部包含循环的
128-row learned registers，因此截断会改变模型条件。推荐只在 Stage 2 使用：Stage 1
仍保留完整文本和全局构图能力。

限制：

- 必须 `allow_approximation=true`；
- 非零值至少 128，最大不能超过实际 conditioning rows；
- 与固定 1024-row 的 Stage-2 ANE K/V 不兼容；推荐 profile 只启用 Stage-1 K/V；
- 长 prompt、多主体、复杂关系和字幕类 prompt 必须重点回归；
- 默认关闭，不能由 `auto` 静默选择。

`ltx-mac` 单 prompt/seed 参考值为约 47.708 秒、相对纯 GPU 1.247×，只说明值得验证，
不能替代 TurboCider 多样本质量验收。

### 5.3 App 模式映射

| App 选项 | Request 映射 |
|---|---|
| 画质优先（Dense） | 不设置 Sol，不裁剪文本 |
| Stage-2 Sol 近似 | `allow_approximation=true`、Stage-2 Sol、tau=1、edge blocks=1 |
| Sol + 256 行文本 | 上述设置再加 `ltx_stage2_text_rows=256` |

近似模式只允许 C/Metal。720p 超过 Sol row 上限时 App/contract 必须拒绝，而不是静默
产生一个配置名与实际计算不一致的结果。

## 6. T2V、I2V 与音频支持矩阵

| 能力 | C/Metal quality | C/Metal Sol/text256 | GPU+ANE fixed-shape | C++/MLX Transformer |
|---|---:|---:|---:|---:|
| T2V | 支持 | 支持，需质量门 | 支持，shape/profile 限定 | 实验支持 |
| 单首帧 I2V | 支持 | 支持，需单独质量门 | 可运行但必须做专门 AB | 暂不支持 |
| 联合音频 latent | 支持 | 支持 | 支持 | 仅按当前实现能力使用 |
| Audio VAE/vocoder/AAC | 支持 | 支持 | 支持 | 媒体后端可复用 |
| 720p tiled VAE | 支持 | Sol 不支持 | 当前 artifact shape 不支持 | 实验 |
| streamed | T2V video-only | 不作为推荐组合 | 不支持 | 受 block cache 策略限制 |

I2V 的 Stage 1 和 Stage 2 使用各自 latent 空间的 clean prefix。首帧编码只做一次并在
denoise 前释放 encoder。正式验收除了输出可播放，还必须验证首帧条件确实影响结果，
不能只检查操作字段。

## 7. 已否决方向与代码整理原则

以下方向已有 `ltx-mac` 完整或足够强的负证据，不应仅凭 microbenchmark 再次进入默认：

- persistent Audio worker；
- QKV sequence split；
- Core ML selector / multifunction；
- V2A ANE 与双 GPU queue 叠加；
- MLP boundary fusion；
- Metal SIMD INT8 projection；
- BF16 pre-expanded projection；
- 跨 diffusion step 的 Video-text K/V cache；
- packed QKV、fused output projection、简单 Sol command batching；
- dual-shape MLP 替代 fixed-shape MLP。

生产代码遵循以下整理规则：

1. 请求/API 只公开有明确语义、可记录 metadata、可做质量门的选项；
2. 仅在 micro 快、完整 E2E 不稳定的候选留作环境变量诊断，不放入 App/profile；
3. 已确认慢且维护成本高的实现直接删除，负结果保留在 validation 文档；
4. 任意近似必须进入 `algorithm_approximations`；
5. denoiser cache key 必须包括所有会改变图、shape 或生成轨迹的参数；
6. 不允许环境变量静默改变用户选定的后端或近似等级。

本轮已修正一个同步偏差：TurboCider 曾把 Video attention command batching 硬编码为
开启，而 `ltx-mac` 的 E2E 结果没有支持默认启用。现在该路径改为显式诊断开关，并在
Sol 模式下强制关闭，避免相互竞争的 command-batch 生命周期。

## 8. 实施阶段

### 阶段 A：接口与可观测性（已完成）

- Request/Swift binding 增加 backend、Fast A/V、Sol 和 Stage-2 text rows；
- parser/contract 校验 `allow_approximation`、参数范围和 Sol row 上限；
- C runtime options 接入 Sol；
- denoiser cache key 纳入 Sol 配置；
- result/plan 输出 backend、Fast A/V、ANE profile 和近似列表；
- App 暴露 `quality`、`sol`、`fast_approx`；
- T2V/I2V 均由正式 executor operation 暴露。

### 阶段 B：基准可复现性（本轮完成工具接入，实机重测待办）

`tools/native/benchmark_ltx_resident.py` 增加：

```text
--ltx-fast-mode quality|sol|fast_approx
--first-frame PATH
--first-frame-strength N
--audio
```

报告 workload 必须记录 operation、execution、fast mode、audio 和 first-frame 信息。

### 阶段 C：重建 Core ML artifact（待办）

1. 用当前实际 checkpoint 和当前设备重新导出 fixed-shape INT8 MLP；
2. 重新导出 Stage-1 1024-row text K/V；
3. 编译/首次加载完成后确认后续 warm run 不再出现 E5 warning；
4. 若清理缓存，只删除已确认属于此次 artifact 的具体 Core ML cache/bundle，不递归删除
   用户主目录、工作区、模型目录或共享缓存根；
5. 记录 exporter 命令、manifest hash、source checkpoint identity 和 artifact bytes。

### 阶段 D：严格性能 ABBA（待办）

对以下模式分别运行 A-B-B-A，再换 B-A-A-B；每个进程至少一轮 warmup、三轮计时：

1. C/Metal `quality`；
2. C/Metal `sol`；
3. C/Metal `fast_approx`；
4. fixed-shape INT8 GPU+ANE `quality`；
5. fixed-shape INT8 GPU+ANE + `sol`；
6. fixed-shape INT8 GPU+ANE + `fast_approx`；
7. C++/MLX dense，仅作后端竞争对照。

每个模式至少覆盖：

- T2V 704×448×97、video-only；
- T2V 704×448×97、audio；
- I2V 704×448×97、strength 0.65、audio；
- T2V 768×448×121、audio；
- T2V 1280×704×121、quality-only tiled VAE。

阶段计时必须分开记录 model load、conditioning/RNG、Stage 1、upsample、Stage 2、Video
VAE、audio decode、mux、request wall、client wall；同时记录 peak RSS、swap、ANE warning
数和 output path。

### 阶段 E：质量套件与发布策略（待办）

固定至少 12 个 prompt × 3 seeds，覆盖：

- 单主体动物和自然运动；
- 真人近景、手部、面部；
- 多主体交互和遮挡；
- 快速镜头/高速运动；
- 室内低光、室外高动态范围；
- 长 prompt、空间关系、颜色/数量约束；
- I2V 人像、风景、细线条和文字图各至少两张。

只有 `quality_hybrid` 通过后才可以作为硬件 profile 推荐；Sol/text256 即使通过，也保持
用户显式选择，直到更大 prompt family 验证完成。

## 9. 验收门槛

### 9.1 功能门

- T2V、I2V 均生成可播放 H.264 MP4；
- 音频开启时为 48 kHz stereo AAC，A/V duration 差不超过一帧；
- 无黑帧、NaN、Inf、崩溃、死锁、错误 cache 复用；
- 同 seed/同 exact 配置的 Stage-2 latent 可重复；
- I2V 输出首帧与输入保持显著结构相似，同时后续帧有非零运动；
- 取消请求能结束且下次请求不复用污染状态。

### 9.2 性能门

以同 binary、同 checkpoint、同进程策略、同输入、同输出媒体设置比较：

- C/Metal `quality` 不得比当前 production baseline 慢超过 3%；
- Fast A/V 相对串行 exact 路径，warm denoise 至少不回退，目标提升 ≥5%；
- GPU+ANE `quality_hybrid`：warm denoise speedup ≥1.10，目标接近 ltx-mac 1.20×；
- `sol`：相对相同 execution/profile 的 quality 模式 warm denoise 至少提升 1%；
- `fast_approx`：相对 `sol` 的 Stage 2 至少提升 2%；
- C++/MLX 只有在 median request wall ≤ C/Metal 1.03× 且质量通过时才有资格进入 auto；
- peak RSS 不超过设备 recommended working set；测试期间 swap 必须为 0；
- 任何收益必须大于 ABBA MAD/噪声，不能用单次最快值判断。

### 9.3 自动质量门

exact 调度候选要求最终 Stage-2 latent byte-exact，或在确认底层调度存在非确定性时满足：

- latent rel-L2 ≤ 1e-4；
- latent cosine ≥ 0.99999。

近似/ANE INT8 不能用 exact 阈值淘汰，应使用 paired decoded-video 指标：

- mean frame RGB correlation ≥0.95；
- minimum frame RGB correlation ≥0.90；
- mean RGB cosine ≥0.98；
- mean MAE ≤8/255；
- motion-energy relative error ≤20%；
- 无新增冻结、闪烁、主体破碎或音画错位。

这些阈值是筛查门，最终还需双盲人工 A/B。真人、I2V 首帧和长 prompt 任一类别明显退化，
该候选不得成为默认。

### 9.4 ANE 设备健康门

以下任一情况会使整轮性能结果无效：

- `Must re-compile the E5 bundle`；
- ANE model load failed；
- Core ML 回退到 CPU/GPU；
- 每轮重复 compile/load；
- artifact provenance 与 checkpoint 不一致；
- warm run 仍有大幅 model-load 抖动。

必须先修复 artifact/device cache，再重新开始 ABBA。无效轮次可以保留为诊断记录，但字段
必须标记 `not_a_performance_result=true`。

## 10. 建议验收命令

构建和契约测试：

```bash
cd TurboCider
MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" \
  TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
Python/bin/python -m pytest \
  tests/native/test_contract.py \
  tests/native/test_ltx_streaming_benchmark.py -q
MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" \
  tools/native/build_app.sh
make test-app
```

GPU quality：

```bash
Python/bin/python tools/native/benchmark_ltx_resident.py \
  --library build/native/libturbocider.dylib \
  --model models/LTX-2.5 \
  --cache /private/tmp/turbocider-ltx-quality-cache \
  --output /private/tmp/turbocider-ltx-quality \
  --report /private/tmp/turbocider-ltx-quality.json \
  --runs 4 --width 704 --height 448 --frames 97 --steps 11 \
  --residency component_staged --execution gpu \
  --ltx-fast-mode quality --audio
```

GPU Sol / fast approximation：

```bash
# 分别运行 --ltx-fast-mode sol 与 fast_approx；其余参数完全相同。
```

GPU+ANE quality：

```bash
Python/bin/python tools/native/benchmark_ltx_resident.py \
  --library build/native/libturbocider.dylib \
  --model models/LTX-2.5 \
  --cache /private/tmp/turbocider-ltx-ane-cache \
  --output /private/tmp/turbocider-ltx-ane \
  --report /private/tmp/turbocider-ltx-ane.json \
  --runs 4 --width 704 --height 448 --frames 97 --steps 11 \
  --residency component_staged --execution gpu_ane \
  --ane-manifest profiles/ltx-ane-fast-int8-20260913.json \
  --ltx-fast-mode quality --audio
```

I2V 在上述命令增加：

```text
--first-frame /absolute/path/to/input.png --first-frame-strength 0.65
```

正式 ABBA 应使用专门 orchestrator 将不同 mode 交错运行，并把 stdout/stderr 中的 ANE
warning 计数纳入 JSON；不要连续跑完一种模式后再跑另一种。

## 11. 当前验证状态

截至 2026-09-13：

- native build 通过；
- `test_contract.py` 与 `test_ltx_streaming_benchmark.py` 当前结果为 68 passed、1 skipped、
  62 subtests passed；
- Swift App 和 integration tests 构建通过，`make test-app` 在获得系统废纸篓权限后通过；
- TurboCider 首轮 fixed-shape ANE 实跑被 E5 bundle 重编译警告污染，不能形成性能结论；
- 已生成的 80–184 秒请求只保留为失败诊断；
- 正式 GPU／GPU+ANE／Sol／text256 ABBA 与多 prompt/seed 质量套件仍待完成。

本轮另完成了一次纯 GPU 单 prompt/seed、两次请求取第二次 warm 的初步 probe：

| 模式 | warm request | warm denoise | denoise 相对 Dense |
|---|---:|---:|---:|
| Dense quality | 69.641 s | 52.944 s | 1.000× |
| Stage-2 Sol | 67.073 s | 51.714 s | 1.024× |
| Sol + Stage-2 text256 | 65.472 s | 50.552 s | 1.047× |

接触图中三者均保持红狐、雪林和从左向右奔跑的主要语义及连续动作，但 aligned RGB 与
latent 门没有通过：Sol mean/min correlation 为 0.836/0.771、latent cosine 0.935；
`fast_approx` 为 0.816/0.747、latent cosine 0.924。运动能量最大相对误差分别约 7.9%
和 8.3%。因此这组结果只证明速度开关真实生效，不能证明质量已达标；两档继续默认关闭。

对应机器状态记录应放在：

```text
docs/design/validation/ltx-mac-acceleration-audit-2026-09-13.json
```

## 12. 模型、下载和磁盘约束

- LTX/H3 模型只从 ModelScope 获取；不要从 Hugging Face 或其镜像继续下载；
- 当前任务不需要重新下载 LTX checkpoint；
- 模型权重、生成视频、tensor dump、build objects、Core ML compiled artifact 和设备缓存
  不得提交 Git；
- 释放磁盘时优先删除明确可重建的旧输出、失败下载 `.part`、过期 build 目录和已经确认
  不再引用的 Core ML 编译缓存；
- 删除前必须列出精确路径和大小，禁止对 `~`、工作区根、模型共享根或通用 cache 根执行
  宽泛递归删除。

最终发布判定只有三种：

1. `quality`：默认、完整功能、无新增算法近似；
2. `quality_hybrid`：特定硬件/shape 推荐，必须有有效 ANE artifact 和质量报告；
3. `sol`/`fast_approx`：用户显式选择的速度优先模式，永远带近似 metadata 和回退路径。
