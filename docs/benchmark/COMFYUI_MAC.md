# ComfyUI Mac：当前正式版的同机完整出图对照

2026-10-12，M4 Max64GiB /macOS26.6.2，Z-Image Turbo，512²/8步，base-only。
最新已按用户要求补Qwen/LTX并回用TC最优，见[四模型重点对照](COMFYUI_MODEL_COMPARISON.md)。
本页保留原Z同窗口配对，不混成新的最佳记录比较。
**TurboCider GPU/ANE 5.840s，对本轮更快的ComfyUI路线8.535s为1.462×，
完整生成耗时减少31.6%；纯GPU为1.155×。** 仅限此已测任务，不是
所有Mac配置、插件、PyTorch版本或四模型的全局排名。

## 同一新cohort的实际结果

同一原BF16 encoder/DiT/VAE，四路线正反序，8进程、32PNG、8份完整独立
进程树采样。每进程cold/两fresh/重复prompt，主中位数来自每路四fresh。

| 路线 | 四fresh中位s | 首请求，正/反s | 构造/服务ready，正/反s | tree physical peak GB |
| --- | ---: | --- | --- | --- |
| TurboCider GPU | 7.391720 | 7.587 /7.969 | 0.143 /0.158 | 31.18–31.97 |
| **TurboCider GPU/ANE frozen W8A8** | **5.839860** | 17.664 /11.689 | 0.149 /0.154 | 30.91–32.00 |
| **ComfyUI v0.39默认MPS** | **8.535402** | 11.839 /10.679 | 3.828 /3.045 | 27.18–27.78 |
| ComfyUI v0.39 SDPA + GPU-only /BF16 | 8.840069 | 11.321 /11.360 | 3.032 /3.040 | 27.63–27.88 |

分母采用更快的默认配置，不采用较慢的8.840s；也不拿上一Unsloth组的
native5.955s拼倍率。混合ANE路线冷启动更贵、peak约32GB而非Comfy约28GB。
全部swap-out为0；GPU反序有约4.54MB进程压缩，原始内存记录保留。
这是完整PNG请求，不是denoise-only、模型加载速度或节点缓存命中时间。

## ComfyUI在Mac上实际怎么执行

| 项目 | 默认路线 | 显式SDPA/GPU路线 | 实测解释 |
| --- | --- | --- | --- |
| GPU后端 | PyTorch MPS /Metal | 同左 | 两路服务实际device均为mps，不是MLX或CUDA |
| 文本encoder | CPU /FP16，来自同一BF16源 | MPS /BF16 | fresh编码节点host envelope约0.348s →0.076s |
| DiT attention | sub-quadratic | PyTorch SDPA | sampler envelope约7.887s →8.443s，抵消encoder收益 |
| VAE | MPS /BF16，split attention | MPS /BF16，PyTorch attention | decode envelope约0.229s →0.166s |
| 驻留与中间结果 | stock shared-memory/offload策略 | `--gpu-only`，MPS驻留 | 不是GPU/ANE并行；调驻留不保证端到端更快 |

以上Comfy阶段是WebSocket节点边界的host envelope，包含节点准备与
生命周期成本，不能当GPU独占kernel时间；没有额外GPU fence扰动计时。
native同组denoise约6.760s，mixed约5.207s，优势主要在DiT，而不是HTTP/UI。
不是每个阶段都领先：native fresh text_encode约0.332s，比Comfy GPU编码
envelope0.076s长。权重驻留、短prefill与融合值得继续对照，但阶段边界不同，
不能仅凭这些envelope把差距全部归因于某个encoder kernel。
GPU-only也改变阶段间tensor放置，SaveImage envelope从约0.010s变为0.104s；
decode阶段变短不代表decode+CPU导出一定更快。
SDPA/GPU组合整体比默认慢约3.6%；不能据此把每个单独选项都判成无收益，
也未穷举“GPU encoder + sub-quadratic”等其他组合。

stock这条路径没有调用CoreML/ANE来分担FFN。v0.39源码的
`supports_int8_compute`对MPS返回false，并注明eager Kitchen的
`torch._int_mm`尚无MPS实现；**W8权重存储不自动等于Mac W8A8整数乘加，
更不等于ANE加速**。GGUF/FP8插件的decode、浮点matmul和缓存需要单独核验，
不能把插件存在或量化文件名直接当本轮完整性能证据。

只读SolAttn-MPS参考使用自定义Metal稀疏attention，属于GPU优化，不是
ANE路线。其默认`min_tokens=4096`，本512² Z任务image tokens约1024、
文本很短，不能直接套用它针对长视频/H3公开的kernel倍率。今后优化优先
继续针对DiT/FFN融合、解码摊销和GPU/ANE overlap；长序列attention另测。

## 版本、缓存与媒体证据

- 2026-10-12查询官方release：v0.39.0，2026-10-05发布；隔离只读checkout
  commit `b0b743566f65daafc423b4fea8a2fbda94b3384a`，dirty为空。
  本机原Desktop v0.32.0及原LTX VAE补丁未改，用户已有服务未关闭。
- 沿用本机Python3.13.12 /PyTorch2.12.1 /Transformers5.14.1。五个明确
  pinned包只装到隔离overlay：frontend1.53.10、templates0.11.76、
  embedded-docs0.5.13、Kitchen0.2.37、Aimdo0.5.5，实际installed/required
  全匹配。未把它说成PyTorch所有新旧版本的最优配置。
- stock Z workflow拓扑，Euler /simple /shift3 /CFG1；官方当前模板的
  res_multistep改为Euler以匹配native任务，不加载额外LoRA。fresh seed42。
  所有custom/API nodes、预览关闭；MPS自动CPU fallback环境变量未开启，
  默认CPU text属于显式放置，不是隐藏op fallback。
- 重复prompt改用seed43，两边都重新采样。每个Comfy请求核验8个actual
  steps以及KSampler/VAEDecode/SaveImage真的执行；fresh必须执行encoder。
  不关闭stock loader/node缓存，不把相同prompt+seed整张图跳过算成推理。
- owned loopback服务，listener PID逐请求验证；输出/user/temp/database
  隔离，readiness独立，8份100ms/500ms门槛内存均complete，服务均终态退出。
  初次预检发现仅base-directory仍会碰到原数据库锁；正式cohort已明确
  user-directory和in-memory SQLite，不用预检中的锁冲突路径作主榜。
  计时从HTTP提交到成功history及完整PNG下载/导出；native为C API/导出。
  浏览器UI不计入，接口边界仍不同。
- 16对正反序同路线/同请求完整RGBA全部exact。已检查四路线两fresh完整
  contact sheet，均为正常雪林狐狸。GPU/mixed八项RGB MAE约3.04–5.02/255，
  SSIM约0.938–0.972；W8A8不是bit-exact或跨engine严格画质parity。
  相同seed在不同RNG实现中不保证相同noise。

[完整便携机器证据](../design/validation/z-comfy039-current-20261012.json)绑定
实际请求/事件/history/媒体/内存链。原始记录在
`outputs/z-comfy039-four-route-v1-20261012/`；旧v0.32完整诊断在
`outputs/z-comfy-four-route-v1-20261012/`，不混进当前中位数。
source-generation首次SHA后只做metadata guard，不每次inference重扫权重。

## 四家族覆盖边界

| 模型 | 本轮ComfyUI比较处理 |
| --- | --- |
| Z-Image Turbo | 当前v0.39完整配对，采用更快实测MPS路线，入榜 |
| Qwen Image2.1 | 已补三Comfy预设12完整PNG/3memory，最快48.350s；回用TC最优31.059s，1.557×，非同窗口配对 |
| FLUX4B | whole saved-MLX Q4 index不在stock UNET choices；未形成完整stock生成，单shard/不同GGUF不填分母 |
| LTX2.5 | 已补同一704×448×97/11步/含音频MP4，cold142.925s、单warm112.230s；回用TC质量/近似最优，1.758×/1.814×，非paired median |

此前1024²/9步Comfy40.110s对mixed30.001s的1.34×见
[历史口径](../public/BENCHMARKS.md#z-image-on-m4-max--september-7-2026)。
它有不同seed/计时边界，不替代本次当前512²完整配对；其余三家族没有
Comfy倍数就不制造“全部远超”。本轮补上Comfy对照后继续收敛，不扩大网格。

复现入口：`tools/validation/z_comfy_framework_screen.py --help`，需要显式
指定已有Comfy checkout/Python、可选隔离package overlay、native Python
和source-matched library。工具不会安装/升级用户环境、下载新模型或关闭用户服务。

上游固定来源：
`https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.39.0`；
`https://github.com/Comfy-Org/ComfyUI/blob/b0b743566f65daafc423b4fea8a2fbda94b3384a/comfy/model_management.py`；
`https://github.com/Comfy-Org/ComfyUI/blob/b0b743566f65daafc423b4fea8a2fbda94b3384a/comfy/ldm/modules/attention.py`；
SolAttn只读commit `45071126b0c1ee30b0e6b7103fa9d70924828ba5`：
`https://github.com/yshenaw/ComfyUI-SolAttn-MPS`。
