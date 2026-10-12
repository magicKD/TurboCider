# 本地benchmark一页摘要

更新2026-10-12。M4 Max64GiB/macOS26.6.2，base-only，无额外LoRA。
结论是Qwen/Z混合已测任务领先，Qwen纯GPU输给新版参考，LTX推荐GPU；不是
所有框架/尺寸/精度的全局第一。每行只用自身campaign的完整生成热态
中位数，不跨轮挑最快样本；cold、repeat、constructor和压力另列。

| 完成日期 / 任务 | TurboCider GPU s | 本组最快混合 s | 外部参考 s | 同组判断 |
| --- | ---: | ---: | ---: | --- |
| 10-10 [Qwen原GGUF Q4_K_M，512²/40步](README.md#3-qwen-gguf-与原-unslothencoder-到完整生成) | 44.428 | 36.870，Private W8A8 | pinned Unsloth/sd-server125.365 | GPU2.82×、混合3.40×；两fresh/路 |
| 10-11 [Qwen原BF16源，512²/40步](QWEN_BF16_FRAMEWORKS.md) | 43.347 | 31.251，frozen CoreML | mflux BF16/cache0 45.968 | GPU1.06×、混合1.47×；四fresh/路 |
| 10-12 [Qwen BF16源/cache1024完整候选](QWEN_BF16_FRAMEWORKS.md) | 43.377 | 32.393，frozen W8A8 | mflux BF16/cache1024 45.451 | GPU1.048×、混合1.403×；四fresh/路，24PNG/6memory |
| 10-12 [Qwen BF16源/cache4096完整候选](QWEN_BF16_FRAMEWORKS.md) | 43.331 | 32.407，frozen W8A8 | mflux BF16/cache4096 45.498 | GPU1.050×、混合1.404×；与1024参考只差0.1%，不证明速度优劣 |
| 10-12 [Qwen/mflux0.22完整BF16/FP16对照](QWEN_BF16_FRAMEWORKS.md) | 43.309 | 31.810，frozen W8A8 | BF16 41.952；partial FP16 42.000 | 新参考胜GPU约3.2%；mixed仍1.319×，32PNG/8memory；FP16无稳定收益 |
| 10-11 [Z原BF16源，旧参考512²/8步](Z_IMAGE_FRAMEWORKS.md) | 7.363 | 5.780，frozen W8A8 | 固定旧mflux BF16源9.520 | GPU1.29×、混合1.65×；旧cohort，不当当前分母 |
| 10-12 [Z/mflux0.22完整BF16/FP16对照](Z_IMAGE_FRAMEWORKS.md) | 7.354 | 5.959，frozen W8A8 | BF16 7.976；partial FP16 8.355 | 当前GPU1.085×、混合1.339×；参考peak22.51GB更低，32PNG/8memory |
| 10-12 [FLUX9B原BF16源，512²/4步](FLUX9_HYBRID.md) | compiled7.844 | 7.841，INT8-weight/FP16-I/O prefix | mflux BF16/cache0 7.355 | 混合对compiled仅省0.031%，参考胜；四fresh/路 |
| 10-12 [FLUX9B DiT-only Q4/g32，512²/4步](FLUX9_MATCHED_Q4.md) | Q4 7.987 | 未验证；该候选GPU-only | 同planes mflux Q4 8.031 | GPU仅名义0.5%差异；对自身BF16 7.922s不加速，但peak降至18.16GB |
| 10-12 [FLUX9B encoder+DiT线性Q4/g32，512²/4步](FLUX9_FULL_Q4.md) | Q4 8.005 | 未验证；该候选GPU-only | 同planes mflux Q4 8.043 | 对自身BF16 7.742s慢3.4%，native peak11.99GB、约少60% |
| 10-12 [FLUX9B Q4源/predecoded DiT+compiled GPU](FLUX9_PREDECODE.md) | 7.211 | 该候选GPU-only | 同源packed mflux Q4 8.028；原BF16 7.343 | 对自己packed8.001s省9.88%，peak24.4GB；不同BF16权重数学 |
| 10-11 [LTX原base，704×448×97/11步，含音频](LTX_VIDEO.md) | quality63.848 | INT8 quality71.333 | 暂缺成功同任务外部对照 | GPU胜；近似GPU61.869但仅省3.1%、漂移明显 |

倍率为参考耗时/本路线耗时，不是“少用相同百分比时间”。混合为部分
近似量化，不能宣称严格BF16画质parity；不同框架同seed不等于同noise。
Unsloth这里只指固定GGUF/sd-server路径。FLUX参考当前仍为
2026-09-20固定commit、包版本字段0.19.2；Qwen/Z新增0.22.0/独立依赖环境
完整实测，不代表所有版本/其它硬件的排名。不同checkpoint/quant recipe
不互当分母。当前主要是文生图/视频，不冒充Image Edit首步对照。

## 参数探索与推荐

- [Z Private完整14候选](Z_PRIVATE_PARTITIONS.md)：最佳ANE4096/GPU6144、
  F32 join，6.148s；其同窗口GPU7.345/frozen5.794s。多分给ANE和F16
  join均未改善；热态仍选frozen，单次冷启动未必选它。
  新0.22参考BF16对照中frozen5.959/reference7.976s，优势缩至1.339×；
  reference peak22.51GB比native约32GB更低，FP16没有时延收益。
- Qwen GGUF的GPU/Private快速组合已接到合格512/40 base请求默认路径；
  BF16源的frozen是独立近似候选，不能混称同一种默认精度。
  最新参考缓存1024/4096MiB完整对照约45.45s，GPU约43.35s、mixed
  约32.4s；保存1024的较低观测peak34.48–34.88GB候选，不合并旧样本。
  随后0.22同窗口参考41.952s、GPU43.309/mixed31.810s，最新mixed
  优势1.319×，纯GPU不领先；新版参考BF16与FP16仅差0.114%，不宣称加速。
- FLUX保留compiled GPU与新hybrid实验配置，但新hybrid不自动启用。
  cold15.73–16.03s更贵；native同prompt repeat-cache-hit省约11.5%
  单列，不混入fresh优势。物理ANE occupancy/并行重叠尚未观测。
- LTX仍推荐quality GPU。INT8加载/重编译抵消部分计算收益；batch和
  clean-exec estimate不混池，也不把较大delta当无损。

[详细总览](README.md)、[52组逐请求catalog](results-20261011.json)、
[可复现配置](best-local-configs-20261011.json)、[完成度审计](COMPLETION_AUDIT.md)
分别保留任务/精度/冷启动/实际调用/内存/媒体误差和缺口。更多尺寸、
FLUX匹配量化、进一步GPU/ANE分区及LTX成功外部对照仍未完成，扩大
目标保持active/partial，qualification_passed=false。
