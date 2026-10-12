# 当前四家族 benchmark 摘要

2026-10-12，M4 Max64GiB/macOS26.6.2，无额外LoRA。
当前仅FLUX4B、Z-Image Turbo、Qwen Image2.1、LTX；9B暂缓，见
[历史扩展摘要](HISTORICAL_SUMMARY.md)，不进入本页排名。

## 已验证性能优势：只比较兼容且完整成功的路线

| 任务 | 开源参考 s | TurboCider GPU/ANE s | 加速比 | 耗时减少 |
| --- | ---: | ---: | ---: | ---: |
| [Qwen2.1 GGUF Q4_K_M，512²/40步](HISTORICAL_RESULTS.md#3-qwen-gguf-与原-unslothencoder-到完整生成) | 固定Unsloth/sd-server Metal 125.365 | Private W8A8 **36.870** | **3.40×** | **70.6%** |
| [Qwen2.1 BF16源，512²/40步](QWEN_BF16_FRAMEWORKS.md) | mflux0.22 BF16 41.952 | frozen W8A8 **31.810** | **1.319×** | **24.2%** |
| [Z-Image Turbo，512²/8步](Z_IMAGE_FRAMEWORKS.md) | mflux0.22 BF16 7.976 | frozen W8A8 **5.959** | **1.339×** | **25.3%** |

不兼容/启动拒绝的框架不进入主榜，也不强制转换适配；证据留在审计。
同一可比任务中参考胜出的路线仍保留：Qwen本组纯GPU43.309s略输
mflux41.952s；Z纯GPU7.354s领先，但参考更省内存。GGUF纯GPU44.428s，
对该固定Metal路径为2.82×；不扩大成Unsloth所有后端或所有模型的优势。

这是完整生成热态中位数；cold、constructor、repeat另列，各行自己的
cohort作分母。固定GGUF参考为原generate_image span，HTTP另存；
native包含export，边界差别保留。mixed/量化有数学变化，不称严格BF16画质parity。
Qwen/Z mixed冷启动较贵；LTX近似仅省3.1%、97帧MAE约25/255。
公开项目不同硬件/任务的数字只作[外部背景](OPEN_SOURCE_COMPARISONS.md)，
不与本机时间拼倍率；也不把LLM tok/s当图像/视频速度。

## 尚未进入开源对比榜的家族

| 家族 | 当前实测状态 | 下一步 |
| --- | --- | --- |
| [FLUX4B，512²/4步](FLUX4_CURRENT_STATUS.md) | Native Q4/g64四次PNG预检成功，两fresh中位3.723s；mflux/mlx-serve也可运行 | 补最佳GPU/mixed及兼容参考的完整正反序；不把预检拼成跨框架倍率 |
| [LTX2.5，704×448×97/11步/音视频](LTX_VIDEO.md) | GPU quality63.848s，INT8 quality71.333s未胜；暂无兼容成功的同任务外部分母 | 当前推荐GPU；不兼容参考略过，不强制改造 |

重点展示已证实的优势；“3.40×”只用于固定GGUF/Metal任务，“1.32–1.34×”
用于这两组mflux图像任务，不宣称四模型全面远超或全局最优。
[当前报告](README.md)、[逐请求catalog](results-20261011.json)、
[可复现配置](best-local-configs-20261011.json)、[完成度审计](COMPLETION_AUDIT.md)。
历史总53组中41组在当前四家族范围、12组9B暂缓；目标未完成。
