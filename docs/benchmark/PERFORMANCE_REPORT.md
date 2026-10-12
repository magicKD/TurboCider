# TurboCider 图像与视频性能评测

2026-10-12 · Apple M4 Max /64GiB · macOS26.6.2 · base-only，无额外LoRA。
范围仅FLUX4B、Z-Image Turbo、Qwen Image2.1、LTX2.5；FLUX9B不参与排名。
本报告收敛已有完整实验与刚结束的Unsloth对照，不再扩大参数网格。

## 核心结论

**在已测完整任务中，TurboCider GPU/ANE相对固定Unsloth/Metal路径达到
3.40×（Qwen GGUF）和3.15×（Z-Image）；相对mflux0.22约快1.32–1.34×。**
这是指定硬件、模型、任务和配置的实测优势，不是全部框架/设备的全局最优。

| 模型与完整任务 | TurboCider GPU | TurboCider GPU/ANE | 兼容开源参考 | 混合路线优势 |
| --- | ---: | ---: | ---: | --- |
| [Qwen2.1 GGUF Q4_K_M，512²/40步](HISTORICAL_RESULTS.md#3-qwen-gguf-与原-unslothencoder-到完整生成) | 44.428s | **36.870s** | 固定Unsloth/sd-server Metal 125.365s | **3.40×，耗时减少70.6%** |
| [Z-Image Turbo BF16源，512²/8步](Z_UNSLOTH_FRAMEWORKS.md) | 7.356s | **5.955s** | Unsloth Metal GPU-text 18.735s；mflux0.22 7.971s | **对Unsloth3.15×，减少68.2%；对mflux1.34×** |
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

## FLUX4B与LTX：保留已测结果，不制造外部分母

| 模型与任务 | 已测性能 | 当前可用于报告的结论 |
| --- | --- | --- |
| [FLUX4B Q4/g64，512²/4步](FLUX4_CURRENT_STATUS.md) | Native单进程cold3.959s、两fresh中位3.723s、repeat3.440s，4PNG及完整内存 | 已验证原生可运行；mflux/mlx-serve也可出图，但未形成当前配对排名，不声明领先倍率 |
| [LTX2.5 distilled，704×448×97/11步，含音频MP4](LTX_VIDEO.md) | GPU quality63.848s；INT8 quality71.333s | 当前推荐GPU；混合没有收益，没有兼容成功的同任务外部对照则不填分母 |

FLUX4B不是从未测过：历史文档记录GPU/ANE warm1.628s，见
[2026-09-08状态](../status/current-status-2026-09-08.md#当前推荐配置)。它是
旧BF16窗口，不拿来替代当前Q4/g64两fresh结果，也不与今天的框架预检拼倍率。
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
保留54组历史/当前记录，其中42组在当前四家族范围、12组9B为历史deferred。
