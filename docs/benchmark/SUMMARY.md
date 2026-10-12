# 当前四家族 benchmark 摘要

2026-10-12，M4 Max64GiB/macOS26.6.2，无额外LoRA。
当前仅FLUX4B、Z-Image Turbo、Qwen Image2.1、LTX；9B暂缓，见
[历史扩展摘要](HISTORICAL_SUMMARY.md)，不进入本页排名。

| 任务 | 原生GPU s | 最快已测GPU/ANE s | 同组开源参考 s | 当前结论 |
| --- | ---: | ---: | ---: | --- |
| FLUX4B，512²/4步 | 待当前重测 | 待当前重测 | mflux/兼容mlx-serve待测 | Q4整包已恢复、SHA验收；未作运行/性能通过，旧4B/9B不充当分母 |
| [Qwen2.1 BF16源，512²/40步](QWEN_BF16_FRAMEWORKS.md) | 43.309 | 31.810，近似frozen W8A8 | mflux0.22 BF16 41.952 | mixed1.319×；pureGPU略输参考 |
| [Qwen2.1 GGUF Q4_K_M，512²/40步](HISTORICAL_RESULTS.md#3-qwen-gguf-与原-unslothencoder-到完整生成) | 44.428 | 36.870，Private W8A8 | 固定Unsloth/sd-server125.365 | GPU2.82×/mixed3.40×，只针对这条路径 |
| [Z-Image Turbo，512²/8步](Z_IMAGE_FRAMEWORKS.md) | 7.354 | 5.959，近似frozen W8A8 | mflux0.22 BF16 7.976 | GPU1.085×/mixed1.339×；参考更省内存 |
| [LTX2.5，704×448×97/11步/音视频](LTX_VIDEO.md) | quality63.848 | INT8 quality71.333，未胜GPU | 暂缺成功外部完整对照 | 推荐GPU；近似GPU61.869s但漂移明显 |

这是完整输出热态中位数；cold、constructor、repeat另列，各行自己的
cohort作分母。mixed/量化有数学变化，不称严格BF16画质parity。
Qwen/Z mixed冷启动较贵；LTX近似仅省3.1%、97帧MAE约25/255。
公开项目不同硬件/任务的数字只作[外部背景](OPEN_SOURCE_COMPARISONS.md)，
不与本机时间拼倍率；也不把LLM tok/s当图像/视频速度。

可以说已测Qwen/Z的mixed领先，不能说四模型全局最优：FLUX4B当前
对照与LTX成功外部基线仍缺，更多precision/shape/share仍需实测。
[当前报告](README.md)、[逐请求catalog](results-20261011.json)、
[可复现配置](best-local-configs-20261011.json)、[完成度审计](COMPLETION_AUDIT.md)。
历史总53组中41组在当前四家族范围、12组9B暂缓；目标未完成。
