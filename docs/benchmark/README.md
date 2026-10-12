# 本地 base 模型 benchmark

更新2026-10-12，Apple M4 Max64GiB/macOS26.6.2。按用户最新要求，
当前只比较FLUX.2 Klein4B、Z-Image Turbo、Qwen Image2.1和LTX2.5。
base-only表示不加载额外LoRA；独立distilled checkpoint仍明确单列。
FLUX9B暂缓，仅保留在[历史报告](HISTORICAL_RESULTS.md)，不进入主榜，
不从9B或9月旧4B数据推算当前4B性能。

先读[一页摘要](SUMMARY.md)；[开源项目公开评测与可比性](OPEN_SOURCE_COMPARISONS.md)
区分同机完整实测、外部公开结果和未测/格式拒绝，不混用分母。

## 当前完整实测

| 模型 / 任务 | 原生GPU s | 本组最快GPU/ANE s | 同组开源参考 s | 判断 |
| --- | ---: | ---: | ---: | --- |
| FLUX4B，目标512²/4步 | 待当前重测 | 待验证 | mflux0.22/当前mlx-serve待测 | 上游Q4整包已恢复、SHA验收，运行/格式待测；旧4B不充当当前结果 |
| [Qwen2.1 BF16源，512²/40步](QWEN_BF16_FRAMEWORKS.md) | 43.309 | 31.810，近似frozen W8A8 | mflux0.22 BF16 41.952 | mixed1.319×；新版参考略胜pureGPU |
| [Qwen2.1原GGUF Q4_K_M，512²/40步](HISTORICAL_RESULTS.md#3-qwen-gguf-与原-unslothencoder-到完整生成) | 44.428 | 36.870，Private W8A8 | pinned Unsloth/sd-server125.365 | 该固定路径GPU2.82×/mixed3.40×；不同精度不混池 |
| [Z-Image Turbo BF16源，512²/8步](Z_IMAGE_FRAMEWORKS.md) | 7.354 | 5.959，近似frozen W8A8 | mflux0.22 BF16 7.976 | GPU1.085×/mixed1.339×；参考peak22.51GB更低 |
| [LTX2.5，704×448×97/11步/含音频](LTX_VIDEO.md) | quality63.848 | INT8 quality71.333，未胜GPU | 暂缺成功同任务外部完整对照 | 当前推荐GPU；不把失败启动当速度分母 |

全部时间为完整PNG/MP4 request wall，constructor/cold/repeat另列。
图像每路通常4个fresh，LTX部分为两次conditioning-cache-hit warm；
不跨campaign挑最小值。physical peak覆盖load+cold+warm+exit进程树，
不等于硬RAM cap或全部外部服务/driver归因。

## 当前推荐与限制

- Qwen/Z持续热态可选已测mixed；它们是部分近似，不宣称严格BF16。
  Qwen首步GPU、后续image-only FFN分工；本表不是Image Edit无KV-cache首步对照。
- Z Private14候选中4096/F32最快，但仍不及同窗口frozen；[分区网格](Z_PRIVATE_PARTITIONS.md)。
- LTX性能近似GPU Sol+text256为61.869s，对自身quality只省3.1%，
  97帧MAE约25/255；保存opt-in，不冒充无损或自动默认。clean-exec
  62.639s为不同estimate边界，不与persistent API wall拼倍率。
- FLUX4B历史BF16记录有完整source SHA，但对应目录当前未找到。
  原BF16恢复尝试的共享encoder精确SHA核验失败，不能混用Z encoder；
  失败receipt保留，明确标注的官方mflux-community Q4整包已恢复并
  完成全部文件的size/SHA验收（约4.62GB），runtime/performance仍false。
  在完成source/格式/实际生成验收后补同机
  GPU/最佳mixed/mflux/兼容mlx-serve及可用量化对照；不缩小目标为历史数字。

[原始样本catalog](results-20261011.json)保留53组历史/当前完整实验，
其中41组在当前四家族范围、12组FLUX9B为deferred；不把它们一起称为
“53组当前四模型结果”。[已测配置](best-local-configs-20261011.json)
明确active/deferred范围；[完成度](COMPLETION_AUDIT.md)、[后续覆盖](EXPANSION_STATUS.md)
保留缺口。原模型、失败日志、媒体和只读参考仓库均保留，SHA只在首次
source-generation验证，后续未变generation复用。整体goal保持active。
