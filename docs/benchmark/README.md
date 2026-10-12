# 本地 base 模型 benchmark

更新2026-10-12，Apple M4 Max64GiB/macOS26.6.2。按用户最新要求，
当前只比较FLUX.2 Klein4B、Z-Image Turbo、Qwen Image2.1和LTX2.5。
base-only表示不加载额外LoRA；独立distilled checkpoint仍明确单列。
FLUX9B暂缓，仅保留在[历史报告](HISTORICAL_RESULTS.md)，不进入主榜，
不从9B或9月旧4B数据推算当前4B性能。

先读[一页摘要](SUMMARY.md)；[开源项目公开评测与可比性](OPEN_SOURCE_COMPARISONS.md)
区分同机完整实测、外部公开结果和未测/格式拒绝，不混用分母。
主榜只纳入兼容且完整成功的开源路线；不兼容者略过，不强制转换或
改造它们，失败记录仅供审计。优先展示已验证优势，同任务参考胜出
的结果也保留，不把“最快已测”扩大成全部框架/任务全局最优。

## 兼容开源框架的完整实测：优势置顶

| 模型 / 任务 | 原生GPU s | 本组最快GPU/ANE s | 同组开源参考 s | 判断 |
| --- | ---: | ---: | ---: | --- |
| [Qwen2.1原GGUF Q4_K_M，512²/40步](HISTORICAL_RESULTS.md#3-qwen-gguf-与原-unslothencoder-到完整生成) | 44.428 | **36.870**，Private W8A8 | pinned Unsloth/sd-server125.365 | **3.40×，省70.6%**；纯GPU2.82×，限该固定Metal路径 |
| [Qwen2.1 BF16源，512²/40步](QWEN_BF16_FRAMEWORKS.md) | 43.309 | 31.810，近似frozen W8A8 | mflux0.22 BF16 41.952 | mixed1.319×；新版参考略胜pureGPU |
| [Z-Image Turbo BF16源，512²/8步](Z_IMAGE_FRAMEWORKS.md) | 7.354 | 5.959，近似frozen W8A8 | mflux0.22 BF16 7.976 | GPU1.085×/mixed1.339×；参考peak22.51GB更低 |

以上为完整生成热态时间，constructor/cold/repeat另列。固定GGUF参考
使用原generate_image span，HTTP端到端另存，不能与native export边界
称完全相同。其余图像为完整PNG request wall。
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
  [LTX完整结果](LTX_VIDEO.md)的GPU quality为63.848s，INT8 quality
  71.333s未胜GPU；没有兼容成功的外部分母，只报告内部优化，排除
  原组合启动拒绝的框架，不以强制转换作为benchmark完成要求。
- FLUX4B历史BF16记录有完整source SHA，但对应目录当前未找到。
  原BF16恢复尝试的共享encoder精确SHA核验失败，不能混用Z encoder；
  失败receipt保留，明确标注的官方mflux-community Q4整包已恢复并
  完成全部文件的size/SHA验收（约4.62GB）；下载receipt的runtime/
  performance=false只描述资产阶段，不改写。随后[当前mflux预检](../design/validation/flux4-mflux-saved-q4-precheck-20261012.json)
  已成功生成512²/4步完整PNG，单cold3.998s、constructor0.496s，
  387 DiT/905 encoder/266 VAE参数全覆盖；这是运行兼容性，不是
  fresh中位数或完整内存基线。继续补同机GPU/最佳mixed及兼容参考，
  不用旧4B/9B数据或单cold时间填写正式排名。
  当前mlx-serve26.10.1同一上游整包也实际返回512²/4步PNG，视觉
  检查正常、服务终态退出；仅兼容性通过，尚无配对计时或完整内存。
  随后[原生Q4/g64预检](FLUX4_CURRENT_STATUS.md)也完成四次完整PNG，
  两fresh中位3.723s、tree physical peak4.21GB，source payload和VAE
  卷积逆转轴序逐tensor位模式通过。VAE含8个Q4 attention线性层；
  尚无本组配对reference/混合结果，不宣传跨框架倍率。

[原始样本catalog](results-20261011.json)保留53组历史/当前完整实验，
其中41组在当前四家族范围、12组FLUX9B为deferred；不把它们一起称为
“53组当前四模型结果”。[已测配置](best-local-configs-20261011.json)
明确active/deferred范围；[完成度](COMPLETION_AUDIT.md)、[后续覆盖](EXPANSION_STATUS.md)
保留缺口。原模型、失败日志、媒体和只读参考仓库均保留，SHA只在首次
source-generation验证，后续未变generation复用。整体goal保持active。
