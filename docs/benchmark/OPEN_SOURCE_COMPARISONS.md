# 开源项目评测：同机实测与公开背景分开

更新2026-10-12。当前仅FLUX4B、Z-Image Turbo、Qwen Image2.1、LTX2.5，
base-only。主榜只比较兼容且完整生成成功的路线；不兼容者略过，不
强制转换或修改外部框架。不是所有框架/版本/硬件的全局排行榜。
收敛后的主报告见[性能评测](PERFORMANCE_REPORT.md)；本页只补框架身份与口径。

## 能作本机速度分母的结果

| 项目 / 已测路线 | 模型与任务 | 开源参考完整s | 原生本组最优完整s | 可比性与限制 |
| --- | --- | ---: | ---: | --- |
| Unsloth权重+pinned sd-server/Metal | Qwen2.1原Q4_K_M encoder+DiT/BF16VAE，512²/40步 | 125.365 | Private36.870 | **mixed3.40×，省70.6%**；只代表该固定路径，不代表Unsloth所有执行后端 |
| mflux0.22，fixed release commit | Qwen2.1 BF16源512²/40步 | 41.952 | frozen31.810 | mixed1.319×，省24.2%；nativeGPU43.309略输；W8A8不是严格BF16 |
| Unsloth native sd-server/Metal，最快已测GPU-text | Z-Image Turbo原BF16源512²/8步 | 18.735 | frozen5.955 | **mixed3.146×，省68.2%**；默认CPU-text22.600s更慢，未用作宣传分母 |
| mflux0.22，同release | Z-Image Turbo原BF16源512²/8步，同一新cohort | 7.971 | frozen5.955 | mixed1.338×；referencepeak22.48–22.52GB比native约32GB低 |
| ComfyUI v0.39.0 /PyTorch2.12.1 MPS | Z-Image Turbo原BF16源512²/8步，独立四路线cohort | 默认8.535；SDPA/GPU-only8.840 | frozen5.840；GPU7.392 | **对更快默认配置mixed1.462×，省31.6%**；8进程/32PNG/8memory，不代表全部Mac presets |

最新用户要求Comfy重点对照、TC直接取已有最优，不重复原生实验；
[四模型重点表](COMFYUI_MODEL_COMPARISON.md)另列**跨窗口已有最佳记录**：
Z8.535/5.780s为1.477×；Qwen最快Comfy GPU驻留48.350/旧mixed31.059s为
1.557×；LTX完整AV单warm112.230/旧quality63.848s为1.758×，对opt-in
近似61.869s为1.814×（MAE25.03/255）。Qwen12PNG/3memory与LTX2MP4/
完整memory已验收，TC没有再次重测；不把这些回用数字升级为上表的paired cohort。

每行只用自己的cohort和完整生成，不跨组拼最快分子/最慢分母。
Qwen/Z mflux组为完整正反序、原尺寸PNG、独立进程树内存和
实际参数/calls证据；固定GGUF组为同前置负载、各路两个fresh，
原generate_image span与native request/export边界的差别单列。
[Qwen](QWEN_BF16_FRAMEWORKS.md)、[Z最新Unsloth/mflux](Z_UNSLOTH_FRAMEWORKS.md)、
[固定GGUF路径](HISTORICAL_RESULTS.md#3-qwen-gguf-与原-unslothencoder-到完整生成)。
Model checkpoint/数学/组件生命周期、cold、repeat缓存边界另列。
[ComfyUI Mac完整对照](COMFYUI_MAC.md)另列当前v0.39正式版/精确依赖overlay、
MPS attention与encoder放置；不动本机原v0.32安装，强制核验实际sampler执行，
不把DAG整图cache命中计入速度。Comfy峰值更低、mixed cold更高也保留。

FLUX4B Q4/g64已经通过mflux0.22完整PNG单cold预检（3.998s），当前
mlx-serve26.10.1同一整包也实际生成512² PNG；[两项兼容证据](../design/validation/flux4-mflux-saved-q4-precheck-20261012.json)
仍不是与本机GPU/最佳mixed完整配对计时，暂不入速度榜。LTX原组合
mlx-serve memory preflight拒绝仍排除；新Comfy现已成功完整AV出图，
单warm观察与回用native中位数的边界单列，不拿失败时间当分母，不转换外部模型。

## Unsloth支持与本机覆盖不是一回事

2026-10-12核验：Unsloth Desktop官网明确列出FLUX和Z-Image图像生成，
官方Hugging Face仓库也提供`unsloth/FLUX.2-klein-4B-GGUF`和
`unsloth/Z-Image-Turbo-GGUF`。不能因本报告尚无某路线计时，就把它
写成Unsloth不支持。模型发布者、量化recipe与实际执行后端必须分开。

| 家族 | 已有Unsloth同机性能对照 | 当前处理 |
| --- | --- | --- |
| Qwen2.1 GGUF | 有：原Unsloth权重+固定sd-server Metal完整40步，125.365s | 对本组mixed36.870s为3.40×，限该后端/版本/任务，不代表全部Desktop路径 |
| Z-Image Turbo | 有：当前五路线10进程/40PNG/10memory完整正反序；Unsloth原native Metal BF16源GPU-text18.735s、defaultCPU-text22.600s | 用最快已测GPU-text对本组frozen5.955s为3.146×；不是整个Desktop UI开销或全参数最优 |
| FLUX4B | 暂无成功完整的Unsloth Desktop/sd-server配对cohort | mflux/mlx-serve单次兼容预检不能替代；在同checkpoint/recipe成立时再测，拒绝者略过 |

公开身份来源：`https://unsloth.ai/`、
`https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF`、
`https://huggingface.co/unsloth/Z-Image-Turbo-GGUF`。这里只核验能力与模型
存在，不把网页中的速度宣传当本机分母。当前catalog的Unsloth完整
对照已覆盖Qwen原GGUF与Z原BF16组；FLUX4B尚无当前Unsloth配对，不推算其胜率。

## 上游公开评测能说明什么

| 上游固定来源 / 公开内容 | 属于什么证据 | 当前处理 |
| --- | --- | --- |
| BFL官方FLUX.2-klein-4B model card：4-step distilled，宣称sub-second生成，约13GB VRAM | 上游能力/性能宣传；未给出可复现的同机M4 Max完整任务计时 | 不拿“<1s”除本机时间；不混非蒸馏base-4B50步 |
| mflux官方0.22.0 release：Z默认隐流从FP32改BF16；新增局部FP16 compute及报告的潜在收益 | 同项目版本优化；不是它与TurboCider的同机胜率 | 本机Qwen/Z已实际重测，FP16在这些任务没有收益；FLUX9B旧实验当前deferred |
| mlx-serve README：相同MLX权重对LM Studio解码geomean约+26% | LLM token decoding，不是FLUX/Z/Qwen图像或LTX视频吞吐 | 不当媒体速度分母；媒体必须实际成功完整输出 |
| mlx-serve26.10.1官方release列出FLUX4B默认媒体路径、LTX2.5/Qwen2.1功能 | 版本能力声明，不等于本机速度实测 | 当前binary的4B Q4实际PNG预检已成功；其他路线各自核验，旧source不冒充当前能力 |
| ltx-2-mlx上游公开benchmark：M2 Pro、主要2.3/streaming/不同pipeline | 不同设备、checkpoint和步骤/输出边界 | 仅作背景；本机暂无兼容成功的同任务2.5完整MP4分母，不跨条件拼倍率 |

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
memory pressure，同任务有更快参考时也如实列出。重点展示已测
Qwen/Z优势，不写“四模型全部远超”；未配对的FLUX4B与无成功参考的
LTX只报已测边界。按用户收敛要求，本轮不再扩大实验网格或强制适配。
