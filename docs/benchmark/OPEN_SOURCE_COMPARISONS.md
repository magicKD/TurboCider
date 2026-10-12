# 开源项目评测：同机实测与公开背景分开

更新2026-10-12。当前仅FLUX4B、Z-Image Turbo、Qwen Image2.1、LTX2.5，
base-only。不是所有框架/版本/硬件的全局排行榜。

## 能作本机速度分母的结果

| 项目 / 已测路线 | 模型与任务 | 开源参考完整s | 原生本组最优完整s | 可比性与限制 |
| --- | --- | ---: | ---: | --- |
| mflux0.22，fixed release commit | Qwen2.1 BF16源512²/40步 | 41.952 | frozen31.810 | mixed1.319×；nativeGPU43.309略输；W8A8不是严格BF16 |
| mflux0.22，同release | Z-Image Turbo512²/8步 | 7.976 | frozen5.959 | mixed1.339×；referencepeak22.51GB比native约32GB低 |
| Unsloth权重+pinned sd-server/Metal | Qwen2.1原Q4_K_M encoder+DiT/BF16VAE，512²/40步 | 125.365 | Private36.870 | 该具体路径mixed3.40×，不代表Unsloth所有执行后端 |
| mflux/当前mlx-serve | FLUX4B512²/4步 | 待完整重测 | 待完整重测 | 不能拿9B数据、9月旧4B或公开GPU宣传值替代 |
| mlx-serve/ltx-2-mlx | LTX2.5 704×448×97/11步/stereoMP4 | 暂缺成功匹配基线 | qualityGPU63.848 | 已有stock mlx-serve memory preflight拒绝不是latency；converted/其他runtime仍待补 |

前三行只用自己的完整正反序cohort、原尺寸PNG、独立进程树内存和
实际参数/calls证据；[Qwen](QWEN_BF16_FRAMEWORKS.md)、[Z](Z_IMAGE_FRAMEWORKS.md)、
[固定GGUF路径](HISTORICAL_RESULTS.md#3-qwen-gguf-与原-unslothencoder-到完整生成)。
Model checkpoint/数学/组件生命周期、cold、repeat缓存边界另列。

## 上游公开评测能说明什么

| 上游固定来源 / 公开内容 | 属于什么证据 | 当前处理 |
| --- | --- | --- |
| BFL官方FLUX.2-klein-4B model card：4-step distilled，宣称sub-second生成，约13GB VRAM | 上游能力/性能宣传；未给出可复现的同机M4 Max完整任务计时 | 不拿“<1s”除本机时间；不混非蒸馏base-4B50步 |
| mflux官方0.22.0 release：Z默认隐流从FP32改BF16；新增局部FP16 compute及报告的潜在收益 | 同项目版本优化；不是它与TurboCider的同机胜率 | 本机Qwen/Z已实际重测，FP16在这些任务没有收益；FLUX9B旧实验当前deferred |
| mlx-serve README：相同MLX权重对LM Studio解码geomean约+26% | LLM token decoding，不是FLUX/Z/Qwen图像或LTX视频吞吐 | 不当媒体速度分母；媒体必须实际成功完整输出 |
| mlx-serve26.10.1官方release列出FLUX4B默认媒体路径、LTX2.5/Qwen2.1功能 | 版本能力声明，不证明本地原格式可运行或更快 | 从当前binary/对应tag源码核验，再作同机对照；旧只读source checkout日期不同，不冒充当前能力 |
| ltx-2-mlx上游公开benchmark：M2 Pro、主要2.3/streaming/不同pipeline | 不同设备、checkpoint和步骤/输出边界 | 仅作背景；当前2.5成功同任务完整MP4仍待实测，不跨条件拼倍率 |

截至本次查阅，上游LLM速度图、microkernel峰值、denoise-only时间、
冷启动失败和不同模型参数规模，都不能直接用于当前四模型完整生成
排名。优先比较同model/checkpoint、尺寸、步数、precision/recipe、
prompt/cache、完整输出边界与硬件；不同seed engine也不保证相同noise。

## 来源与版本身份

- BFL原4B固定repo/revision：`black-forest-labs/FLUX.2-klein-4B` /
  `e7b7dc27f91deacad38e78976d1f2b499d76a294`；官方model card：
  `https://huggingface.co/black-forest-labs/FLUX.2-klein-4B`。
- mflux已测0.22 commit：`31c9640399572ea394d07092763be9a36007a754`，
  release：`https://github.com/mflux-community/mflux/releases/tag/v.0.22.0`。
- mlx-serve当前官方release：`https://github.com/ddalcu/mlx-serve/releases/tag/v26.10.1`；
  README：`https://github.com/ddalcu/mlx-serve`。本地只读reference source
  是2026-09-01旧commit，不能据其不存在的模块推断26.10.1整体不支持。
- ltx-2-mlx源码/公开bench：`https://github.com/dgrauet/ltx-2-mlx`；
  本地只读0.15.2 snapshot commit `91e6f6c9bd621ff2ae31adfee643e113d67d6ae8`，
  2026-08-30；当前release需独立核验，不将此snapshot说成所有未来版本。

所有公开数字不进入[raw sample catalog](results-20261011.json)的本机
成功latency数组。保留框架自身负结果、完整source/实际执行/输出和
memory pressure，避免只展示较慢参考。目标仍active，FLUX4B与LTX
缺口不能用公开宣传数字填成完成。
