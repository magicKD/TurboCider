# ComfyUI Mac vs TurboCider 已测最优

2026-10-13整理；实测窗口2026-10-11至12，M4 Max64GiB /macOS26.6.2，base-only，无额外LoRA。
按最新要求：**已测TurboCider取保留的最优配置，不重复跑；Comfy已测的Z也不重跑。**
新补Qwen完整PNG和LTX完整AV MP4。Comfy为v0.39.0 /PyTorch2.12.1 MPS，GPU/SDPA
预设均实际执行；比值采用更快的Comfy预设，不拿慢配置放大优势。

| 完整任务 | Comfy最快已测 | TurboCider已有最优 | 每次节省 | 耗时差异 | 证据口径 |
| --- | ---: | ---: | ---: | --- | --- |
| Z-Image Turbo BF16源，512²/8步 | **8.535s**，默认MPS | **5.780s**，frozen W8A8 GPU/ANE | **2.755s** | **1.477×，省32.3%** | Comfy四fresh；TC原完整四fresh最优，跨窗口回用 |
| Qwen Image2.1 BF16源，512²/40步 | **48.350s**，GPU驻留/sub-quadratic | **31.059s**，W6144 GPU/ANE | **17.291s** | **1.557×，省35.8%** | Comfy两fresh/预设；TC原完整四fresh最优，跨窗口回用 |
| FLUX.2 Klein4B，512²/4步 | 当前本地saved-MLX Q4/g64未形成stock完整出图 | **2.574s**，Q4派生W8/F16 GPU/ANE；原packed GPU3.723s另列 | N/A | **N/A，不造Comfy分母** | 新派生视图16PNG/4memory验收，非原BF16/直接packed混合；Comfy whole index不在stock UNET choices |
| LTX2.5，704×448×97/11步，含音频MP4 | **112.230s**，SDPA/BF16 VAE，单warm观察 | quality GPU**63.848s**；opt-in近似**61.869s** | 质量**48.382s**；近似**50.361s** | **质量路线1.758×；最快近似1.814×** | 完整AV、复用原TC记录；Comfy只有1个warm，不伪称中位数 |

倍率为Comfy耗时 /TC耗时，节省比例为1−TC耗时 /Comfy耗时，
绝对差值由原始未舍入记录计算。完整PNG/含音频MP4导出计入；模型
构造/服务ready与cold另列。这里的“最优”是匹配来源、任务与缓存口径的
已验收最佳配置主记录，不是跨表示/缓存挑最低单次sample。

这里是用户指定的**“已有最佳记录对照”**，不是全新的同窗口配对实验：
明确允许回用TC旧窗口，保留来源、版本、配置与样本数。原严格同窗口Z
结果仍为8.535 /5.840s，1.462×，见[原配对报告](COMFYUI_MAC.md)，不改写它。
先前Unsloth/mflux的各自同cohort倍率仍在[总体性能报告](PERFORMANCE_REPORT.md)。
不能据此说四模型/所有Mac配置/插件/Torch版本都全面领先。

### 纯GPU与GPU/ANE一起看，不把更慢的混合路线当赢家

下面回用各自原记录，没有重新跑TC。Z/Qwen的GPU列取其保留混合最优
所在窗口的GPU control，避免把别的prompt/cache窗口最小值拼进去。

| 任务与表示 | TC纯GPU完整s | TC GPU/ANE完整s | Comfy最快已测s | 纯GPU对Comfy倍率 | 当前选择 |
| --- | ---: | ---: | ---: | ---: | --- |
| Z BF16源512²/8步 | 7.363 | **5.780** | 8.535 | **1.159×** | W5120 frozen混合 |
| Qwen2.1 BF16源512²/40步 | 43.239 | **31.059** | 48.350 | **1.118×** | W6144 frozen混合 |
| LTX quality，97帧AV | **63.848** | 71.333 | 112.230，单warm | **1.758×** | GPU，不强选ANE |
| LTX opt-in Sol+text256，97帧AV | **61.869** | 66.036 | 112.230，单warm | **1.814×** | GPU，MAE25.03/255需披露 |
| FLUX4当前Q4派生single40 BF16，512²/4步 | **3.484**，80实际compiled blocks | **2.574**，80实际runtime calls | 无当前stock Comfy完整生成 | N/A | W8/F16混合；不以旧BF16 1.628s代替当前来源 |

2026-10-13扩大goal续作只补FLUX4这个此前未测的派生候选，没有重跑
原packed GPU或任何外部框架。相同派生视图内混合比GPU快1.353×、
少26.1%；source链、BF16预解码成本与四fresh/route见
[FLUX4当前结果](FLUX4_CURRENT_STATUS.md)。它不改变本页的Comfy N/A。

Z纯GPU相对Comfy省13.7%时间，W8A8混合再比自身GPU省21.5%；Qwen
对应为10.6%和28.2%。因此Z/Qwen的额外优势主要来自混合配置，而不是
仅凭GPU后端就有1.5×。这是配置级差距，不是隔离ANE硬件贡献的因果
测量：混合也改变精度、调度和数据传输。LTX质量GPU已省43.1%，近似
GPU再比自身质量路线省3.1%；ANE路线更慢，当前不推荐为测速开启。

热态优势也不等于一次性请求优势：保留Z最优配置的cold为11.608–21.587s，
Comfy默认为10.679–11.839s，窗口不同，不能拿热态倍率推算cold赢家。
混合还增加驻留/启动成本；后续选择以完整任务的实际耗时为准。

Z同配置在Private网格control中也出现5.748s的四fresh中位，比保留主
cohort5.780s只低0.55%；不是新的配置/优化收益。历史5.344s是同prompt
conditioning-cache-hit窗口，不能替换本表fresh值去除Comfy fresh时间。
当前最优配置仍是W5120；完整control/原样本保留，不以噪声宣称新加速。
来源：[Z原完整框架组](../design/validation/z-valid-framework-current-20261011.json)、
[Private全网格](Z_PRIVATE_PARTITIONS.md)、[LTX完整八配置](../design/validation/ltx-full-configs-current-20261011.json)。

## Qwen：三种Comfy预设，不重复原生实验

三个独立owned实例，12个完整512² PNG，三份独立完整100ms/500ms内存。
每实例cold/两fresh/重复prompt；repeat换seed30确保40步/VAE重新执行。
三组件SHA与TC最优记录逐一相同，ManualSigmas对齐native40步Euler的
resolution mu与最后非零sigma0.02；CFG1、stock lossless auto prefix KV cache。

| Comfy v0.39 MPS预设 | 两fresh中位s | cold s | repeat新seed s | tree physical peak GB | swap-out GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 默认，CPU FP16 text /sub-quadratic | 49.013907 | 63.232 | 47.508 | 44.249 | 0.001 |
| **GPU驻留，BF16 text /sub-quadratic** | **48.349678** | 66.740 | 47.503 | 43.811 | 0.833 |
| SDPA + GPU驻留 /BF16 | 51.627773 | 67.732 | 50.531 | 43.637 | 0.546 |

GPU驻留比默认只快约1.4%；SDPA组合反而更慢。GPU-only是请求的策略，
不是“全程零offload”：日志实际有约2.3GB encoder部分卸载，源模型
加载/驻留与统一内存压力影响冷态和fresh。swap与累计压缩记录均保留，
不是硬RAM cap或所有设备上的最佳默认。

| fresh阶段host envelope | 默认 | GPU驻留 | SDPA/GPU |
| --- | ---: | ---: | ---: |
| encoder | 1.521s | 0.858s | 1.152s |
| sampler | 46.305s | 46.302s | 49.297s |
| VAE | 1.085s | 0.792s | 0.778s |
| SaveImage | 0.013s | 0.317s | 0.318s |

这是WebSocket节点边界，不是GPU-exclusive microkernel计时。GPU-only
把CPU回拷更多计入SaveImage，所以VAE单节点更短不等于整条decode/export
同比更快。主要耗时仍在DiT，不能把全部差异都归因于encoder。

TC回用[已验收最优记录](../design/validation/qwen21-valid-framework-current-20261011.json)：
纯GPU43.239s、mixed31.059s。第一步GPU，后39步GPU attention/caption
与并行ANE image-FFN6144/12288分工，实际1248 runtime calls/request、
初始32warmup calls；数值为W8A8近似，不等于严格BF16/全场景画质parity。
相对Comfy最快48.350s，旧pureGPU记录为1.118×，mixed为1.557×。

[新Qwen完整事件/history/媒体/内存与旧baseline绑定](../design/validation/qwen21-comfy039-reference-current-20261012.json)。
原始数据：`outputs/qwen21-comfy039-reference-only-v1-20261012/`。
早期五路线批次在默认Comfy四PNG之后因单次621ms采样gap被拒绝；
保留`outputs/qwen21-comfy039-five-route-v1-20261012/`失败，不降500ms门槛。
最新用户缩减原生重测后，新的三预设只执行Comfy；原失败时间不进入本表。

## Z：直接复用已经测过的Comfy

Comfy默认8.535s比SDPA/GPU8.840s更快；原8进程/32PNG/8memory完整
证据不重跑。TC复用[最优5.780s原记录](../design/validation/z-valid-framework-current-20261011.json)，
W5120/10240 image-only FFN、256calls/request。两边是同原BF16源和512²/8步，
但在不同窗口、各自原库/组件生命周期下测得，不拼成新paired cohort。
GPU/ANE冷启动与内存成本、非bit-exact误差仍保留在原记录中。

本页更新只补Comfy兼容路线，TC最佳配置与失败/更快参考不改写；不下载
新的大模型，不转换外部框架格式，不扩展LoRA、9B或新的参数网格。

## LTX：完整97帧音视频，不重跑TC

Comfy v0.39沿用同一本地distilled ConvRot DiT、Gemma4 ConvRot encoder、
BF16 **conv-only video VAE**、audio VAE和latent upscaler；源SHA与原TC
记录逐一相同，不替换为新pixel-diffusion VAE或别的checkpoint。
704×448 /97帧 /24fps，stage1八步+stage2三步Euler ancestral，两个CFG均1，
与native固定sigma序列一致；关闭LoRA、prompt enhancer和custom/API nodes。

| Comfy完整请求 | 耗时 | encoder缓存 | 采样/解码是否重新执行 |
| --- | ---: | --- | --- |
| cold，seed42 | 142.925s | 首次编码 | 8+3步、upscale、音视频decode、MP4导出全部执行 |
| warm，seed43 | **112.230s** | prompt/conditioning节点缓存 | 同上，未用整条DAG跳过作速度 |

warm host节点envelope：stage1 sampler42.986s、stage2 46.212s、upscaler
0.842s、video VAE21.015s、audio decode0.505s、SaveVideo0.500s。
这些含节点准备/加载，不是独占kernel时间。stock默认CPU text，DiT/VAE
为MPS；SDPA和BF16 VAE显式启用，没有启用隐藏MPS CPU op fallback。

两个MP4均H.264 /704×448 /97帧 /24fps，AAC双声道48kHz音频。
完整FFmpeg解码通过，全部audio samples finite，已检查开头/中段/结尾图。
tree physical peak39.097GB、swap-out0.995GB，max sample gap110.027ms，
独立memory complete，owned服务已退出；不是硬内存保证。

回用TC原完整batch记录：quality63.848s比Comfy少43.1%时间；最快
GPU Sol+text256为61.869s，少44.9%，但97帧RGB MAE约**25.03/255**，
不冒充无损或默认质量推荐。已有INT8 GPU/ANE quality71.333s更慢，
当前最优是GPU，不为“必须开启ANE”选一条较慢路线。clean-exec62.639s
为另一种estimate边界，也不拿来拼本表的完整warm wall。

Comfy是一个进程cold+一个新seed warm；TC旧窗口为自己的多样本中位数。
同来源/任务、不同窗口/生命周期/RNG/实现，不能宣称严格像素parity或
一次单warm证明稳定全局最优。按“已测不用充分重测”要求不扩统计网格。
[完整媒体、事件、内存与旧baseline绑定](../design/validation/ltx25-comfy039-reference-current-20261012.json)。
原始记录：`outputs/ltx25-comfy039-sdpa-reference-v2-20261012/`。
首次脚本把SaveVideo DynamicCombo写成nested dict，采样和decode成功但
导出失败；v2修正为stock API dotted inputs，没修改Comfy源码。旧失败保留，
其141.62s不是完整媒体，不进入速度表。

## FLUX4B：缺少当前stock同格式分母，不用不完整shard

本机当前4B是mflux-community原saved-MLX Q4/g64整包，DiT含
`0.safetensors`和`1.safetensors`，whole index共有387个source parameters。
实际stock `folder_paths`/UNETLoader choices只有两shard，**不包含whole
`model.safetensors.index.json`**。不把任一单shard当完整checkpoint测速，
也不把MLX UInt32 affine planes当Comfy GGUF/原BF16。
[本地入口与完整source描述符核验](../design/validation/comfy-flux4-local-format-admission-20261012.json)
是metadata admission检查，不是模型forward失败、性能证据或“ComfyUI
不支持FLUX4B”的声明；其它loader/插件没有本轮完整执行证据，不推测其速度。
原BF164B本地目录当前缺失，故暂不填倍率，也不为本次对照下载/转换模型。
