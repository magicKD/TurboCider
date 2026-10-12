# 本地 base 模型 benchmark

更新：2026-10-12，Asia/Singapore。当前机器为 Apple M4 Max、64 GiB unified
memory、macOS 26.6.2。本报告只比较不加载额外 LoRA 的执行；独立发布的
Z-Image Turbo / LTX distilled checkpoint 单列为 base，不拿不同蒸馏模型
互当分母。PE、Viggle、VDN Turbo adapter 和 distill-patch LoRA 不参加。

只看当前结论与最新同窗口数字，请先读[一页摘要](SUMMARY.md)。

先看结论：Qwen Image 2.1 原 Q4_K_M encoder + DiT / BF16 VAE 已有完整
40 步 GPU、GPU/ANE、原 Unsloth GPU 对照；encoder 和 DiT 均更快。
同观测内存的 BF16 streaming 对照中，GGUF 纯 GPU 仅小幅省时，GGUF
混合路径省时约 19%。这不是所有模型、格式或分辨率都更快的结论。
最新有效BF16/mflux对照中，Qwen GPU/frozen为43.239/31.059s，参考
48.512s；Z GPU/frozen为7.363/5.780s，参考9.520s。两者热态领先，
但不是所有模型/框架/精度的全局第一。FLUX stock匹配GPU8.042/参考
8.828s为1.098×；进一步cache0匹配GPU7.980/参考7.352s，参考反而更快，
不能只选较慢参考宣传胜率。LTX新完整配置中GPU quality63.848s、
GPU Sol+text256 61.869s，约3.1%小收益但明显视频漂移；INT8各recipe
整体更慢，clean-exec另列，不沿用旧0.86%当稳定收益。每组保留日期、构建、边界和
样本，不拼接不同实验的最快数值。
后续真实9B compiled图和encoder参数筛选已完成，最新同窗口native
7.374/mflux7.343s近似持平；对自身eager省约3.13%，仍不宣称普遍领先。
随后参考cache0完整对照为mflux45.968/GPU43.347/frozen31.251s，GPU/
frozen倍率缩至1.060×/1.471×、参考peak降到33–34GB；stock与调优
候选都保留，不为了胜率只选较慢参考，见[Qwen详细边界](QWEN_BF16_FRAMEWORKS.md)。

新增“各家族BF16/量化/参数与本机最佳GPU/ANE配置”的扩大目标仍在推进，
见[扩展覆盖与当前实验](EXPANSION_STATUS.md)。下面48组为已保留的完整
阶段，不把尚未成功的外部框架/未验证分区当作扩大目标已经完成。

10-12新增真实9B GPU/ANE全图四路线正反序：eager8.154/compiled7.844/
hybrid7.841/mflux-cache0 7.355s。每hybrid请求96真实calls、无fallback，
全部32PNG/8memory完整；hybrid对compiled只省0.031%，没有稳定fresh
收益，参考仍胜。cold15.73–16.03s更贵，repeat-cache-hit约11.5%收益
另列，不混入fresh排名。[完整分工、误差与复现](FLUX9_HYBRID.md)。

随后同payload DiT-only Q4/g32完整四路线完成：native BF16/Q4为
7.922/7.987s，mflux BF16/Q4为7.380/8.031s，32PNG/8memory完整。
Q4没有时延收益，但native peak降至18.16GB（自身BF16 30.53–30.65GB）；
它是较低内存候选，不是全组件量化。[严格匹配与误差](FLUX9_MATCHED_Q4.md)。

再后encoder252+DiT153线性层Q4/g32匹配也已完整完成：native BF16/Q4
7.742/8.005s、mflux BF16/Q4 7.346/8.043s，32PNG/8memory完整。
Q4慢约3.4%，但native peak11.99GB、对自身BF16约少60%；embedding/
VAE保持BF16，不冒充每个tensor4bit。[完整层数、误差、CoW与框架边界](FLUX9_FULL_Q4.md)。

最新提前解码候选五路线已完成：Q4源DiT→BF16 operands+compiled GPU
7.211s，对同窗口packed8.001s省9.88%，peak24.40–24.45GB而不是
packed11.95–11.97GB；40PNG/10memory与真实153次初始decode通过。
对参考原BF16 7.343s只是约1.8%名义差异且不同权重数学，不宣称严格
BF16/全局领先。[完整高内存候选与额外舍入](FLUX9_PREDECODE.md)。

10-11 已将实测快速组合接到合格 GGUF 请求的默认路径；不设置八个优化
override 的同库回归为 GPU44.449 / Private36.861 s，24对完整tensor、
12对PNG与显式组合一致。它使既有加速可直接使用，不是再一次新kernel
倍率。[默认条件、关闭开关与真实session回归](../status/qwen21-fast-defaults-2026-10-11.md)。

本轮测量和视频边界已整理：FLUX9B原BF16 native/mflux与Z五格式16配置的
正反序已完成；mlx-serve26.10.1对原FLUX布局拒绝、对原LTX组合在默认
memory preflight阶段拒绝，均不作成功性能分母。不拿第三方宣传值补数。

## 1. 当前本地资产

逻辑权重大小包含该条完整组件组合；共享 encoder/VAE 会在不同组合中
重复列出。它不是独占磁盘空间、驻留内存或峰值 physical footprint。
资产存在只证明 header/stat 可读，不代表已完成运行资格。

| 本地组合 | 逻辑权重 GiB | 本次资产检查 | benchmark 状态 |
| --- | ---: | --- | --- |
| Qwen Image 2.1 BF16 | 30.21 | encoder/DiT/VAE/tokenizer 在 | 40 步 resident/streaming/混合有实测 |
| Qwen Image 2.1 原 Q4_K_M | 9.22 | 原 GGUF encoder/DiT + 原 BF16 VAE 在 | GPU/Private W8A8/原 Unsloth 有同任务实测 |
| Z-Image Turbo BF16 | 19.27 | 完整组合在 | 当前512² GPU/frozen/Private正反序；1024²另保留历史 |
| Z-Image NVFP4 / INT8 ConvRot | 12.00 /13.58 | 完整组合在 | 当前GPU/ConvRot Public+Private正反序；NVFP4 GPU-only |
| Z-Image GGUF Q4_0 / Q8_0 | 12.00 /14.53 | DiT + 共用原BF16 encoder/VAE在 | 当前stock/fast GPU/Public/Private正反序；Q8另有1024²历史 |
| FLUX.2 Klein 9B BF16 | 32.32 | 两个 DiT、四个 encoder shards 与 VAE 在 | native/mflux与精确512/4实验9B混合完整实测；没有稳定fresh混合收益 |
| LTX 2.5 distilled INT8 ConvRot | 42.87 | 含外部软链接目标、connector、两个VAE在 | 当前完整GPU/INT8/FP16音视频实测，见第5节 |
| MiniMax H3 原 BF16 源 | 62.29（仅现存组件） | 缺原 video VAE 权重 | 不能按完整本地可执行模型报告 |
| FastH3 独立四步 INT6 | 72.40（仅现存组件） | encoder/VAEs 在，缺预期 INT6 DiT | 只保留历史结果，不标为本次可复跑 |
| Qwen3 4B GGUF Q4_K_M / Q8_0 | 2.33 / 3.99 | 两原文件在 | encoder 组件；不当成独立图像模型 |
| Qwen3 4B TC affine Q4 / Q8 | 2.84 / 4.53 | 权重与 tokenizer 在 | encoder 组件；与 DiT 组合需单独注明 |

[完整 header/stat 资产记录](local-model-inventory-20261011.json)由
`tools/validation/base_model_inventory.py` 产生，包含各组件相对路径、字节数、
tensor/dtype 统计与缺失项，不做全模型 SHA，不写模型或参考仓库。
Tongyi Z 根目录当前没有独立 DiT；别名目录、Core ML 图和 app-bindings
不是额外 base 模型。

## 2. 实测总览：每行只在自身 campaign 内比较

`倍速 = GPU 中位数 / 混合中位数`，省时为 `1 - 混合/GPU`。
最佳表示该组实际完整请求中的最快已测路线，不表示自动路由、普遍画质
资格或全局最优。Private 与 Public/Core ML 都保留，不能互换标签。

| 日期 / 模型与任务 | 纯 GPU s | 该组最快混合 s | 倍速 / 省时 | 样本与边界 |
| --- | ---: | ---: | --- | --- |
| 10-10 Qwen 原 Q4_K_M，512²，40 步 | 44.428 | 36.870，Private W8A8 通道分割 | 1.205× / 17.0% | 每路两 fresh warm；相同 BF16 I/O 前置负载 |
| 10-10 Qwen BF16，512²，40 步 | 43.353 | 33.121，checkpoint-bound frozen Core ML | 1.309× / 23.6% | 四 fresh warm/路；CPU busy diagnostic，非独占资格 |
| 10-11 Qwen BF16，512²，40步 | 43.250 | 31.008，Public Core ML frozen W6144 | 1.395× /28.3% | 原生完整ABBA；同窗口mflux噪声输出不作外部排名 |
| 10-11 Qwen BF16，最新有效五路线 | 43.239 | 31.059，Public Core ML frozen W6144 | 1.392× /28.2% | 四fresh/路，40PNG；mflux BF16 48.512s，内存压力保留 |
| 10-11 Qwen BF16，参考cache0独立候选 | 43.347 | 31.251，同配置frozen | 1.387× /27.9% | 四fresh/路，24PNG；mflux45.968s，原生对参考1.060×/1.471× |
| 10-11 FLUX9B BF16，旧配对 | 8.221 | N/A，9B DiT分区未验证 | 旧mflux11.263s | 四fresh warm/路；独立旧ABBA |
| 10-11 FLUX9B BF16，最新配对 | 8.042 | N/A，9B分区仍未验证 | 最新mflux8.828s，1.098× | 四fresh/路，16PNG/4memory；同源async ABBA |
| 10-11 FLUX9B BF16，参考cache0配对 | 7.980 | N/A，9B混合仍未验证 | mflux7.352s；native多用8.54%时间 | 四fresh/路，16PNG/4memory；负结果保留 |
| 10-11 FLUX9B，compiled+encoder候选 | 7.374，compiled GPU | N/A，9B混合仍未验证 | 同窗口mflux7.343s，近似持平 | 四fresh/路，24PNG/6memory；对eager7.612s省3.13% |
| 10-12 FLUX9B，实验四路线 | 7.844，compiled GPU | 7.841，CoreML INT8-weight/FP16-I/O prefix | 1.0003× /0.031%，无稳定fresh收益 | 四fresh/路，32PNG/8memory；同窗口mflux7.355s仍胜 |
| 10-11 Z BF16，512²，8步 | 7.551 | 6.187，frozen image-only W8A8 | 1.220× /18.1% | 四fresh warm/配置；完整正反序 |
| 10-11 Z BF16，最新七路线 | 7.363 | 5.780，frozen image-only W8A8 | 1.274× /21.5% | 四fresh/路，56PNG；mflux BF16 9.520s，cold另列 |
| 10-11 Z INT8 ConvRot，512²，8步 | 8.844 | 7.310，Private Comfy W8A8 | 1.210× /17.3% | fastest measured GPU denominator，非慢control |
| 10-11 Z Q4_0 GGUF，512²，8步 | 9.354 | 9.008，Private Sylvester W8A8 | 1.038× /3.7% | CPU-direct packed；GPU FP16 recipe另记 |
| 10-11 Z Q8_0 GGUF，512²，8步 | 9.504 | 9.302，Private Sylvester W8A8 | 1.022× /2.1% | 四fresh warm，非repeat cache hit |
| 10-11 LTX，704×448×97，11步/音视频 | 64.126 | 63.575，配置INT8 Core ML profile | 1.009× /0.86%，不作稳定收益 | 两warm/路；漂移较大，GPU仍推荐 |
| 10-11 LTX，新Sol+text256同recipe | 61.869 | 66.036，INT8 profile | 混合慢约6.74%，GPU胜 | 两warm/配置；共32MP4/20memory，quality GPU63.848s |
| 09-30 Qwen BF16，1024²，40 步 | 183.086 | 144.564，Public runtime v1 | 1.266× / 21.0% | 正反序两 warm/路；不是上一行的 encoder 生命周期 |
| 09 月 Z BF16，512²，8 步 | 6.997 | 5.344，frozen image-only W8A8 | 1.309× / 23.6% | 同库回文顺序四 warm/路；单狐狸 prompt |
| 09-30 Z BF16，1024²，8 步 | 31.429 | 28.400，Public runtime v1 | 1.107× / 9.6% | 灯塔四 warm/路；肖像为 31.401 / 28.388 |
| 09-30 Z Q8 GGUF，1024²，8 步 | 40.891 | 35.198，Public runtime v1 | 1.162× / 13.9% | 灯塔四 warm/路；肖像为 40.894 / 35.227 |
| 09-13 LTX distilled，704×448×97，11 步 | 67.493 | 65.646，实验 INT8 ANE | 1.028× / 2.7% | 单 warm；质量门失败，不作推荐收益 |

来源分别为 [Qwen 同前置负载](../status/qwen21-conditioned-reference-2026-10-10.md)、
[Qwen BF16 frozen 正反序](../status/local512-qwen-frozen-base-2026-10-10.md)、
[Qwen 1024 v1](../status/runtime-ane-v1-base-2026-09-30.md)、
[Z 512 同库三路](../status/runtime-ane-v2-matched.md)、
[Z BF16 1024](../status/runtime-ane-z-1024-v1-2026-09-30.md)、
[Z Q8 1024](../status/runtime-ane-z-gguf-q8-1024-2026-09-30.md)、
[LTX 历史记录](../design/validation/ltx-ane-and-720p-tiled-2026-09-13.json)、
[FLUX 当前对照](../status/flux9-bf16-framework-2026-10-11.md)、[LTX 当前结果](LTX_VIDEO.md)。
历史行不是 10 月 11 日新构建的速度承诺；不得把 M4 Pro/M5 数据并入本表。
Z五格式的新同库配置、完整16路线、cold/fresh/repeat/peak/source/calls
和图像差异见 [Z详细结果](Z_IMAGE_FORMATS.md)。NVFP4 GPU8.844s，无ANE列。
[Z最新完整框架对照](Z_IMAGE_FRAMEWORKS.md)与上述五格式配置是独立
campaign；[FLUX最新参考精度筛选](FLUX_PRECISIONS.md)不与旧原生样本拼倍率。
[Z Private14候选完整网格](Z_PRIVATE_PARTITIONS.md)的最快4096/F32为
6.147741s，同窗口frozen5.793671s仍胜；更大ANE份额和F16 join均未改善。

## 3. Qwen GGUF 与原 Unsloth：encoder 到完整生成

同源原 Q4_K_M DiT / Qwen3-VL-8B Q4_K_M encoder、BF16 VAE；512²、
40 步、seed29、相同三个 fox-light prompts；一 cold 两 fresh warm，
conditioning 全 miss，LoRA=0、reference=0。每一路前面都实际运行三次
同配置 BF16 streaming 40 步请求；整个窗口前后验源，前置负载到 target
之间不读 GGUF payload 做 SHA、不清系统缓存。这统一实际前置工作负载，
不保证 OS 每个 page 或所有 engine cache 完全相同。

| 路线 | cold encoder s | warm encoder s | warm DiT/sampling s | cold 完整生成 s | warm 完整生成 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| TurboCider GPU | 1.414193 | 0.091071 | 43.712706 | 46.171517 | 44.427732 |
| TurboCider GPU/Private ANE5120 | 1.254466 | 0.090424 | 36.168671 | 38.906136 | 36.870464 |
| 原 Unsloth GPU / pinned sd-server | 1.910000 | 0.130000 | 107.300000 | 128.730000 | 125.365000 |

原参考代码未修改，使用其 `metal_text_encoder_flags()` 与
`native_speed_flags("max")`、12 threads；conditioner compute/params
实际为 MTL0，而非拿 CPU encoder 作慢分母。参考完整生成是
`generate_image` span；HTTP warm 为 125.397347 s，readiness 0.213537 s
另外记录。Native 是 `request_wall`，含 VAE/导出；两者边界不是同一 API，
所以阶段数据和完整墙钟均列出，不伪装成逐算子等价。

Native GPU / 混合的 warm 完整生成分别约 2.822× / 3.400×；warm
encoder 分别约 1.427× / 1.438×，DiT 分别约 2.455× / 2.967×。
cold encoder 在这组前置条件下也领先。跨 engine 同 seed 不等于同 noise
或逐像素相同，不能据这些计时宣称质量 parity。

Native GPU encoder 使用实际 typed RMS、NeoX QK norm-RoPE、GQA Metal
kernel，packed QKV18/QK18/gate-up36 融合，embedding 只 gather 使用行。
CPU import 为 8 workers、两块 4 MiB 缓冲、一个有界 pread worker；
源首次完整 SHA 后跨进程复用，target model SHA bytes=0。
Private 为 Hadamard W8A8、ANE5120/GPU7168、IOSurface I/O、fixed async；
首步和后续都做 FFN 通道分工，每 40 步请求实际 1280 成功 calls，
failure/fallback/retry=0。首步不是全部交给 ANE，也没开启 temporal FFN 跳步。

[便携机器记录](../design/validation/qwen21-conditioned-reference-20261010.json)
保留构建/模型/参考二进制身份、各请求值、六个进程树窗口与输出证据。
这组仍是 diagnostic，不等于硬件级物理重叠 trace 或 arbitrary-shape 资格。

## 4. 相近观测内存：GGUF 是否胜过 BF16 streaming？

同 512²/40 步/seed29/三 fresh prompts，一 cold 两 warm；原 BF16
streaming encoder/DiT prefix28、两 slots、managed weight budget13 GiB。
budget 不等于 whole-process RAM cap。下表 physical peak 覆盖
load+cold+warm+exit，包含进程树但不归因所有外部服务/driver。

| 路线 | warm encoder s | warm denoise s | warm request s | tree peak GiB |
| --- | ---: | ---: | ---: | ---: |
| 原 BF16 streaming GPU | 1.504719 | 43.453257 | 45.532613 | 13.739 |
| 原 GGUF GPU，4 MiB read-ahead | 0.090683 | 43.740282 | 44.442035 | 13.611 |
| 原 GGUF GPU/Private ANE5120 | 0.090318 | 36.187572 | 36.879159 | 14.331 |
| 原 BF16 resident GPU | 0.584123 | 42.059459 | 43.346015 | 34.427 |

对 streaming，GGUF GPU 整请求少 2.395%（1.025×）；DiT 反而略慢，
主要省在 encoder 重载/计算。GGUF 混合少 19.005%（1.235×），同时
取得 DiT 收益；stream/mixed physical peak 比为 0.9587，落在 ±10%
观测匹配范围。原 BF16 resident GPU 比 GGUF GPU 更快，但峰值显著更大。
因此不能简化成“Q4 一定比 BF16 快”或“13 GiB 硬内存就能跑”。

streaming/resident 三对 PNG exact；不同量化源不是 bit-exact 同模型
表示。resident 窗口有约 13.80 GB 系统 compression，streaming 约
6.46 MB，均非某一进程独占指标；未隐藏内存压力。
[当前四路记录](../status/qwen21-bounded-import-read-ahead-2026-10-10.md)与
[完整样本](../design/validation/qwen21-bounded-import-read-ahead-20261010.json)的
`matched_memory`；[早期四路与候选筛选](../status/qwen21-matched-gguf-bf16-2026-10-10.md)
保留独立旧窗口，不混用两版结果。

## 5. 其它模型：实测赢家和负结果

- Qwen BF16：10-10 同库 frozen / Private / GPU 正反序中，frozen 热态
  最快，但 cold setup 约 8.38–11.60 s，Private 冷请求更低；frozen
  峰值约 36.1 GB、Private/GPU 约 30.6–31.0 GB。CPU load 检查失败，
  所以保留 busy diagnostic，不将最小单样本写成“最佳”。09 月更早
  同库 campaign 的 42.672 / 36.959 / 30.193 s 只在自身 campaign 解读。
  原10-11 native-only有效窗口43.249697/31.007695s仍单列，错误参考
  遗漏64个FFN权重、输出噪声，不作分母。修复后新十进程/40PNG有效
  五路线已完成：GPU43.238745/frozen31.058807/mflux BF16 48.511753/
  Q4 56.335344/Q8 55.100963s。GPU/frozen对参考BF16为1.122×/1.562×；
  参考peak约51GB且压缩/换页明显，不归因全部kernel；frozen为近似W8A8。
  [完整样本、冷启动、误差及失败保留](QWEN_BF16_FRAMEWORKS.md)。
- Z BF16：512² 已测 frozen 比 runtime v2 的 6.444 s 更快。1024²
  没有匹配的 frozen 分母，runtime v1 是该组最快已测；c704 微图
  少调用却只让完整请求名义省 0.38%，未替换 c352。
- Z Q8：混合可省约 14%，但另一组 BF16 GPU 的约 31 s 不应与它的
  40 s 跨源算 kernel 倍率。auto runtime 肖像正反序 PNG 不同，不能
  宣称完全确定性。这是历史1024²，不与当前512²结果混池。
  当前16配置新窗口中，BF16 fastest fresh6.187s；Q4/Q8的主要价值是
  fast packed GPU/混合的约8.5–11.8GiB tree peak，而非超过BF16时延。
  ConvRot compiled-dense仅名义比stock慢0.24%，未选作最快GPU；其低
  peak单列。全部32memory verifier complete，64对正反序RGB exact，
  BF16窗口的compression和一条ConvRot-fast swap-in保留，不声称无压力。
  后续新七路线/14进程/56PNG完整对照为GPU7.363311/frozen5.780123/
  Private6.139409/Public6.537484/mflux BF16 9.519604/Q4 9.971474/Q8
  10.080610s，frozen热态对参考1.647×，cold11.61–21.59s明显更贵。
  [最新框架与实际分工](Z_IMAGE_FRAMEWORKS.md)，不混池前一窗口。
- FLUX 9B：native 有 GPU module，正式build的DiT `gpu_ane` 拒绝未验证分区。
  新同源BF16、512²/4步、native→mflux→mflux→native对照，四fresh warm
  中位8.220596 /11.263455s，native省27.02%（1.370×）。cold分别
  8.163/10.132和8.792/9.947s，不是所有cold领先。重复prompt另列
  7.023124 /11.821010s，不能把native条件cache hit混进fresh排名。
  reference主计时无额外GPU同步，token校对在全部计时结束后。
  native/reference tree peak约30.1–30.5/40.5GB，两边系统compression、
  reference swap-out都保留；默认组件/allocator生命周期不同，不能全部
  归因kernel。warm图RGB correlation0.998518、SSIM0.990202，非byte-exact。
  [完整样本、阶段、压力和格式拒绝](../status/flux9-bf16-framework-2026-10-11.md)。
  后续stock参考六进程精度筛选BF16/Q4/Q8为8.962395/8.303592/9.466238s，
  明显快于旧参考；该窗口没有native，不跨窗口宣传1.370×。随后v5
  新同窗口native/mflux为8.042414/8.828487s、1.098×，16PNG/4memory
  完整，优势小于旧配对。v6 cache0新同窗口native7.979642/mflux7.351531s，
  mflux胜出，native多用8.544%时间；参考peak32.2–32.4GB、swap-out0，
  两边compression仍保留。匹配量化/9B compiled GPU与混合继续探索。
  [最新精度、表示限制与待补配对](FLUX_PRECISIONS.md)。
  10-12精确512/4实验9B混合已完整测量，7.841s与同窗口compiled7.844s
  持平，mflux7.355s仍胜；之前各日期的N/A保留其当时边界。
  [真实9B混合结果](FLUX9_HYBRID.md)，不把INT8-weight/FP16-I/O称W8A8。
  历史 **4B** 2.266 /1.628s与direct MLX2.264s仍不属于本地 **9B**。
- LTX：704×448×97 的 INT8 混合仅快 2.7%，latent rel-L2=0.3306、
  RGB correlation=0.8586，原质量门失败；FP16 混合 84.924 s 更慢。
  速度和误差都保留，当前推荐 GPU。720p 同 latent tiled VAE 的
  635.719→249.052 s 是另一 GPU 内存/解码优化（PSNR41.99/SSIM0.9873），
  不是 ANE 倍率。上述为09月历史。新完整音视频GPU/INT8正反序两warm
  中位64.126336 /63.574962s，名义差仅0.86%；FP16一warm78.114023s更慢。
  新97帧RGB correlation INT8/FP16为0.858522/0.876494，MAE20.49/19.02
  /255，不把较宽delta当无损。五进程memory verifier完整，compression/
  swap保留；[当前LTX全链路与框架拒绝](LTX_VIDEO.md)，当前仍推荐GPU。
  随后8配置/16arm/32MP4完整筛选：batch quality GPU63.847972/
  INT8 71.333099s；GPU Sol62.848950/text256 61.868682s，后者仅少3.1%，
  全97帧RGB correlation0.815742/MAE25.03/255，保存performance opt-in，
  不当无损默认。clean-exec GPU estimate62.638723s、peak37.60GB，
  视频RGB exact，但不能跨estimate/request_wall边界算倍率；INT8更慢且
  波动大，E5 recompile诊断与系统压力保留。[新完整八配置](LTX_VIDEO.md)。
- FastH3：历史 832×480×124、4 步、INT6/g64、完整音视频 VAE 的
  warm/cache-hit wall 为 native337.893 / FastVideo339.028 s，近似持平，
  不是多样本 ABBA；更不能将缺失的当前 INT6 资产标为已复跑。
  详见 [原设计与样本边界](../design/h3-mlx-int6-fastvideo-parity-plan.md)。

上述观察不要求不同精度逐像素相同，但也不会因允许较大 delta 就抹去
已有误差、失败或改变统计口径。

## 6. 常见框架的实际比较范围

| 框架 | 本地适用任务与来源 | 当前已有同任务证据 | 适用边界 / N/A |
| --- | --- | --- | --- |
| 原 Unsloth diffusion / stable-diffusion.cpp | 原 Qwen GGUF、GPU conditioner；只读参考代码 | 第 3 节 encoder/DiT/完整生成 | 不推广到所有 GGUF 或编辑 |
| mlx-serve | 官方26.10.1/release commit`02bee553f48cd3bc7d82aba0f8073820bd924738`；本地参考commit另记 | 原FLUX9B BF16 MissingFluxWeight；原LTX默认memory preflight拒绝；均无generation | N/A有实测原因，不换converted4/8bit拼等精度排名 |
| direct MLX / mflux | 只读mflux0.19.2 /commit`8c00dab2505a96019df9d30bc9c223bf20d733c4`；同源无损Qwen/Z布局与FLUX原BF16 | Qwen stock48.512 /GPU43.239 /frozen31.059s；cache0另组45.968 /43.347 /31.251s；Z9.520 /7.363 /5.780s；FLUX stock8.042 /8.828、cache0 GPU7.980 /mflux7.352s | FLUX调优参考胜出；Q4/Q8单列；内存/生命周期/近似不同；不外推所有尺寸或低内存设备 |
| FastVideo MLX | 独立 FastH3 INT6/g64、完整 VAE | 上节历史单 cold/warm 配对与 shared-input parity | 当前资产补齐后才能新复跑 |
| ComfyUI | 官方 Z workflow 历史观察 | 09-07 1024²/9步 40.110 s，但 seeds/边界不同 | 不作本报告公平框架排名 |

mlx-serve 的 media 路径不能用其 LLM decode tok/s 排图像生成延迟；
格式、checkpoint、encoder/VAE、分辨率、steps、prompt/cache、媒体输出
必须先对齐。官方 release 和本地源码 commit 分开记录，不能假定完全相同。
外部框架未测的格子保留为空缺；不引用其它机器的厂商 benchmark 补数。

## 7. 复现、审计与适用边界

Qwen 命令和 pinned sources 见第 3/4 节相应文档；Z 使用
`tools/validation/runtime_ane_model_screen.py` 的 gpu/runtime/frozen
回文顺序与原匹配 manifest。所有新 output 必须为空目录，模型任务串行。
`tools/validation/qwen21_gguf_hybrid_screen.py --fast-defaults` 专门审计
不设置八个优化 override 的实际默认路径，而不是再测一次显式 flags。

每个新 campaign 应记录：源与构建身份、精度/分区、request JSON、
warmup/cold 边界、每次各 stage 和完整 wall、实际 calls/fallback、
process-tree physical peak、系统 compression/swap 和输出/质量证据。
首次完整模型 SHA 持久化，后续 generation 未变时只做轻量身份检查；
不要在每个 inference 前重扫模型、用 I/O 预热制造冷启动优势。

`make test-base-benchmark` 使用项目 Python 环境运行定向无模型推理的
报告/样本/路线/质量/框架边界回归；raw目录在本机存在时还复核小型证据
与MP4摘要，不重新扫描模型权重。另已复核24项source注册、原SourceLease
fixture、Qwen options与GGUF只读snapshot。未运行全量`make test`。

当前可执行本地base家族已按任务/表示列出实测与历史，缺资产、未验证
分区、原格式不兼容或stock memory preflight拒绝的条目明确N/A。
额外转换不同精度不是本地原模型排名的隐含分母，未测/失败不会变通过。
这份测量报告不是所有设备、编辑/LoRA、所有尺寸或Public/hard-RAM发布
资格；[目标逐项审计](COMPLETION_AUDIT.md)区分已交付要求与这些限制。

[48 个完整已保留 campaign 的逐请求样本](results-20261011.json)单独导出
了 cold/warm arrays、各阶段时间、构建身份与 source summary SHA；它
是完整的本轮evidence catalog，不把明确N/A或未测扩展伪标为完成的运行。
第一阶段[逐项审计](COMPLETION_AUDIT.md)与各历史实验状态仍保留；扩大
目标的catalog为active/partial，`qualification_passed=false`不变，不代表
全场景发布资格或已经穷尽本机最佳参数。
