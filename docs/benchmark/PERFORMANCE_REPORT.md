# TurboCider 图像与视频性能评测

2026-10-13生成模型整理，2026-10-15补SeedVR2 · Apple M4 Max /64GiB · macOS26.6.2 · base-only，无额外LoRA。
生成模型范围为FLUX4B、Z-Image Turbo、Qwen Image2.1、LTX2.5；FLUX9B不参与排名。
SeedVR2作为图像／视频超分另列，不改写原四家族catalog或其完成口径。
本报告收敛已有完整实验与刚结束的Unsloth/ComfyUI对照，不再扩大参数网格。

**生成模型Comfy重点对照（10月13日）**按用户要求复用TC已测最优，不再重跑原生：
Z **1.48×**、Qwen2.1 **1.56×**、LTX质量GPU **1.76×**／最快近似 **1.81×**。
完整请求分别省**2.755s**、**17.291s**、LTX质量**48.382s**／近似**50.361s**。
这是已有最优记录的跨窗口比较，LTX Comfy只有单warm观察；下方原同窗口
倍率不改写。完整四模型表与来源/精度/缓存边界见[Comfy模型对照](COMFYUI_MODEL_COMPARISON.md)。

## SeedVR2：完整超分对ComfyUI（10月15日补录）

复用10月14日已完成的8组ABBA，不重跑实验。每任务选择更快的Comfy预设，
TC取该组配对结果；每路6个fresh warm，导入／resize／VAE encode／1步采样／
VAE decode／PNG或无音轨H.264导出全部计入，构造与cold另列。

| 超分任务 | 更快Comfy完整s | TC GPU完整s | Comfy/TC | 每次节省 | 耗时减少 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 图像256→512 | 3.978，SDPA/GPU-only | 0.828 | **4.81×** | 3.150s | 79.2% |
| 图像512→1024 | 15.535，默认attention | 3.506 | **4.43×** | 12.029s | 77.4% |
| 视频256→512，25帧/24FPS | 71.190，默认attention，MPS兼容 | 20.586 | **3.46×** | 50.604s | 71.1% |
| 视频512→1024，25帧/24FPS | 295.438，默认attention，MPS兼容 | 85.011 | **3.48×** | 210.428s | 71.2% |

TC为稳定FP16/FP32 **resident GPU**，不是ANE收益。Comfy0.39.0 /Torch2.12.1
MPS两种预设实际均记录FP16权重、BF16 manual cast，不宣称等精度／bit-exact。
视频参考只在任务副本加入non-CUDA缓存转移guard，不是未修改stock视频。
原记录速度门通过但`overall_qualification_passed=false`；无同口径Comfy物理
内存对比，不发布内存排名。全部默认／SDPA控制、cold、版本与原始样本绑定见
[SeedVR2专页](SEEDVR2_COMFYUI.md)和[机器证据](../design/validation/seedvr2-comfy-benchmark-current-20261015.json)。

## 生成模型：核心结论

**在已测完整任务中，TurboCider GPU/ANE相对固定Unsloth/Metal路径达到
3.40×（Qwen GGUF）和3.15×（Z-Image）；相对mflux0.22约快1.32–1.34×，
新增Z-Image对ComfyUI v0.39同机完整生成为1.46×。**
这是指定硬件、模型、任务和配置的实测优势，不是全部框架/设备的全局最优。

| 模型与完整任务 | TurboCider GPU | TurboCider GPU/ANE | 兼容开源参考 | 混合路线优势 |
| --- | ---: | ---: | ---: | --- |
| [Qwen2.1 GGUF Q4_K_M，512²/40步](HISTORICAL_RESULTS.md#3-qwen-gguf-与原-unslothencoder-到完整生成) | 44.428s | **36.870s** | 固定Unsloth/sd-server Metal 125.365s | **3.40×，耗时减少70.6%** |
| [Z-Image Turbo BF16源，512²/8步](Z_UNSLOTH_FRAMEWORKS.md) | 7.356s | **5.955s** | Unsloth Metal GPU-text 18.735s；mflux0.22 7.971s | **对Unsloth3.15×，减少68.2%；对mflux1.34×** |
| [Z-Image Turbo BF16源，独立ComfyUI组，512²/8步](COMFYUI_MAC.md) | 7.392s | **5.840s** | ComfyUI v0.39默认MPS 8.535s；SDPA/GPU-only 8.840s | **对更快默认路线1.46×，减少31.6%** |
| [Qwen2.1 BF16源，512²/40步](QWEN_BF16_FRAMEWORKS.md) | 43.309s | **31.810s** | mflux0.22 BF16 41.952s | **1.32×，耗时减少24.2%** |

各行使用自己同组的实际样本，不跨实验挑最快native/最慢reference拼倍率。
Qwen GGUF组每路两个fresh；其余主表每路四个fresh，正反序各两次。
GPU/ANE是source-bound近似W8A8 FFN分工，不等于严格BF16数值/画质parity。
Qwen纯GPU43.309s略输mflux41.952s，优势来自混合路线；这个负结果保留。

### Z-Image：完整五路线，不用较慢参考夸大优势

同一个原BF16 encoder/DiT/VAE、10进程、40PNG、10份完整内存采样。
Unsloth开启原`max`速度选项；默认CPU-text和可运行的GPU-text都实际测量。

| 路线 | 四fresh中位 | 首次请求，正/反 | 重复prompt中位 | tree physical peak |
| --- | ---: | --- | ---: | --- |
| TurboCider GPU | 7.356s | 8.695 /7.816s | 6.984s | 31.02–31.50GB |
| **TurboCider GPU/ANE frozen** | **5.955s** | 17.429 /14.206s | 5.331s | 31.75–31.89GB |
| mflux0.22 BF16 | 7.971s | 9.012 /8.329s | 7.984s | 22.48–22.52GB |
| Unsloth Metal，GPU-text | **18.735s** | 25.654 /25.146s | 18.734s | 23.86–23.92GB |
| Unsloth Metal，默认CPU-text | 22.600s | 29.347 /28.982s | 22.597s | 22.62–22.69GB |

宣传分母采用更快的18.735s，而不是默认CPU-text 22.600s：纯GPU也快
2.55×，混合快3.15×。对应MF优势只报1.34×，不隐藏更快的mflux参考。
混合路线冷启动更贵、内存更高；一次性请求不只看热态排名。
这些是pinned Unsloth native后端计时，不包含Desktop UI；噪声生成、
tokenizer/模板、内部算术与组件生命周期可能不同，不作跨engine逐像素parity。

### ComfyUI Mac：当前正式版，不把缓存跳过算加速

新组使用2026-10-12核验的ComfyUI v0.39.0，同一原BF16三组件源、
8进程/32PNG/8完整memory，每路四fresh，正反序。保留本机PyTorch2.12.1，
五个版本pin精确匹配的依赖装到隔离overlay，不升级用户Desktop或改原源码。

默认是PyTorch MPS /sub-quadratic attention，encoder在CPU；另测显式
SDPA/GPU-only/BF16路线。GPU encoder更快，但该组合的sampler更慢，故
**采用8.535s而不是8.840s作分母**；不是所有Mac配置/插件/Torch版本最优。
两边重复prompt都改seed43，核验Comfy每次8个真实sampler steps和VAE/
SaveImage执行，fresh必须重编码；没有把相同prompt+seed整图cache命中当推理。
混合cold11.689–17.664s、peak30.91–32.00GB，Comfy默认cold10.679–11.839s、
peak27.18–27.78GB，所有swap-out0。16对正反序同路线RGBA exact；完整图已检查。

随后新增Qwen三Comfy预设12PNG/3memory，最快GPU/sub-quadratic48.350s，
复用已有TC31.059s为1.557×；LTX同源两完整97帧AV MP4，warm112.230s，
复用quality63.848s为1.758×／opt-in61.869s为1.814×。它们是用户指定
“TC取已有最优”的跨窗口对照，不混进上述Z新paired cohort。
FLUX4B当前saved-MLX Q4/g64 whole index不在stock UNET choices，不能
用单shard当完整模型，暂无有效Comfy分母。实现与复现见[Mac专页](COMFYUI_MAC.md)
和[四模型重点表](COMFYUI_MODEL_COMPARISON.md)。

## FLUX4B与LTX：保留已测结果，不制造外部分母

| 模型与任务 | 已测性能 | 当前可用于报告的结论 |
| --- | --- | --- |
| [FLUX4B当前Q4来源，512²/4步](FLUX4_CURRENT_STATUS.md) | 原packed GPU3.723s；新single40预解码GPU**3.484s**、W8/F16 GPU/ANE**2.574s**；保留参考mflux3.811s、mlx-serve4.305s | 新派生视图内部完整配对混合**1.353×、省26.1%**；外部未重跑、不混为新配对排名；Comfy当前格式仍N/A |
| [LTX2.5 distilled，704×448×97/11步，含音频MP4](LTX_VIDEO.md) | GPU quality63.848s；INT8 quality71.333s | 当前推荐GPU；混合没有收益；新Comfy单warm112.230s成功，回用最佳记录另列，不称paired median |

FLUX4B不是从未测过：历史文档记录GPU/ANE warm1.628s，见
[2026-09-08状态](../status/current-status-2026-09-08.md#当前推荐配置)。它是
旧BF16窗口，不拿来替代当前Q4/g64两fresh结果，也不与今天的框架预检拼倍率。
2026-10-13只新增一个当前Q4来源的预解码候选：40个single投影转BF16，
其它267个DiT tensor、encoder/VAE/tokenizer不变，16PNG/4完整memory。
ANE图是INT8权重/FP16 IO与计算，**不是W8A8激活量化**；混合MAE最高
2.573/255、SSIM最低0.98059，peak8.06–8.25GB，cold5.362–9.081s。
原packed与参考均未重跑，跨窗口与表示的差距不升格为新paired框架优势。
LTX最快已测近似GPU Sol+text256为61.869s，只比自身quality省3.1%，
97帧MAE约25/255；配置保存为opt-in，不冒充无损或默认推荐。

## 计时与复现口径

- 主表为完整生成热态中位数，含encoder、DiT、VAE和输出；constructor、
  cold、立即重复prompt独立列出，不混入fresh。LTX为原conditioning-cache-hit
  warm窗口，不能与图像fresh吞吐混排。
- 固定Qwen GGUF参考保留原`generate_image` span，HTTP另存；native含export。
  新Z为native C API/PNG、mflux image/save、sd-server HTTP/PNG回复及导出。
  接口边界说明保留，不把denoise-only或LLM tok/s伪装成完整媒体速度。
- 只比较兼容且完整成功的路线；不兼容/启动拒绝只留审计，不强制适配。
  SHA首次source-generation校验后复用，没有每次inference重扫模型。
- 内存为load+cold+warm+exit的进程树physical peak，不是硬RAM cap。
  量化、混合和精度改变有数学误差；单机/有限prompt不代表全局速度或画质资格。

[逐请求样本catalog](results-20261011.json) · [已测最优配置](best-local-configs-20261011.json) ·
[Z完整机器证据](../design/validation/z-unsloth-mflux-current-20261012.json) ·
[框架身份与公开评测](OPEN_SOURCE_COMPARISONS.md)。
保留59组历史/当前记录，其中47组在当前四家族范围、12组9B为历史deferred。
