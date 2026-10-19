# 当前最快已测配置

核对日期：2026-10-19。设备：Apple M4 Max /64GiB，macOS26.6.2。
本页集中索引已完成实验与本地产物，新实验的实际运行范围见各自独立报告。
App/default未改；原日期报告和原始计时不回填、不覆盖。
当前源码集成、构建和回归入口见[profiling工程索引](../../tools/profiling/README.md)。

## 先回答：Qwen Base最快是不是Public ANE约29秒？

不能直接这样记。**不使用DiT残差缓存、不减少40步实际block计算的512²文生图，
当前最强研究基线约28.7–28.8秒，是Private runtime ANE，不是Public frozen。**

| 容易混淆的记录 | 完整请求时间 | 实际含义 |
| --- | ---: | --- |
| Private runtime，base40文生图 | 28.705s；独立前看实验约28.824–28.835s | 固定FFN＋QKV、one-hop权重前看、text-prefix；无DiT残差复用 |
| Public frozen，base40文生图，DiT cache off | 约30.35–30.49s | checkpoint-bound逐层Core ML FFN；不是跨层动态W图 |
| Public frozen，base40编辑＋conservative cache＋prefix hit | seed73 29.266s，seed74 28.418s | 标准1024参考；复用了15步/360个middle blocks，不能当作完整计算文生图纪录 |
| 较早Public frozen记录中的约29秒 | 29.007–29.164s | 仅denoise；同窗口完整请求约33.0–33.3s，不是29秒整图 |

来源：[Private固定图完整请求](../status/qwen-base512-function-session-2026-10-19.md)、
[独立前看重复](../status/qwen-base512-qkv-lookahead-2026-10-19.md)、
[Public缓存/前缀实验](../status/qwen512-frozen-prefix-cache-2026-10-17.md)、
[Public无残差缓存控制](../status/qwen512-cache-hybrid-2026-10-17.md)、
[早期分阶段表](../status/local512-qwen-frozen-base-2026-10-10.md)。

## 计时和选择规则

- 以下是指定来源/任务的**已测配置**，不是跨模型画质等价排名或全局最优承诺。
- 完整请求包含实际采样、VAE与PNG/MP4导出；constructor、首次load/compile按原记录另列。
  fresh表示新prompt实际编码；repeat表示conditioning热命中，仍重新采样/解码/导出。
- 优先记录完整窗口中位数，不用最快单张。表内范围来自独立窗口，不是置信区间。
- 10月19日Qwen/新Z runtime性能窗有外部推理CPU负载，正式性能资格未过。
  数字保留作研究观察，不能称干净环境认证或据此自动选路；旧记录同样保留各自资格边界。
- Public/Private按**实际执行器**区分，不按library文件名中是否出现private判断。
  W8A8与Core ML调用成功也不是物理INT8 MAC或GPU/ANE硬件重叠的证明。
- 产品默认仍GPU。下面的研究选择不会自动改成App默认。

## 主要模型：当前应保留的速度配置

| 模型/工作负载 | 尺寸、实际步数 | 最强已测配方 | 完整请求观察 | 使用边界 |
| --- | --- | --- | ---: | --- |
| Qwen Image2.1 Base BF16文生图 | 512²，40 | Private固定FFN R1024/C6144＋Private QKV＋one-hop＋text-prefix，首步QKV留GPU | 热28.705s；独立窗口约28.8s | 研究基线；不启用DiT residual cache |
| 同Base单参考编辑 | 512²，40，参考1024 | 同固定FFN/QKV＋edit-prefix hit | 热约35.95–36.22s | 不把首次参考编码或prefix miss算作热命中 |
| Qwen Image2.1 Turbo BF16文生图 | 512²，作者8步 | Private FFN R1024/C6144，sigmoid-complement，F32 partial join | 热7.308s | 独立Turbo权重/采样器，不是Base任意截8步 |
| 同Turbo单参考编辑 | 512²，8，参考1024 | 同FFN＋runtime edit-prefix hit | 热8.275s | 未启用prefix的13.30s慢路线不选；GPU encoder |
| Qwen Base GGUF Q4_K_M文生图 | 512²，40 | Private W8A8 FFN C5120，FP16 IO、关闭F32 join；保留GGUF encoder并用GPU prefill | 热约36.86–36.96s | 同格式比较；不是原BF16模型的28.7s配方 |
| Qwen Base INT8 ConvRot文生图 | 512²，40 | Comfy H256→A8＋原旋转W8 codes，Private C6144，Exp，F32 join | 热39.660s | 保留原codes/scale基；可见局部几何变化，非无损 |
| Z-Image Turbo BF16源文生图 | 512²，8 | **Public frozen** image-only W8A8 R1024/C5120；GPU其余5120通道及全部caption | fresh 5.780s；同组repeat约5.333s | 当前绝对速度选择仍frozen，不是新runtime方案 |
| FLUX.2 Klein4B，当前Q4/g64来源 | 512²，4 | single40预解码BF16派生视图＋Public frozen W8/F16，R1536/C6144，GPU其余3072通道 | fresh 2.574s；repeat约2.301s | **历史配置产物已清理，需重建，不能直接运行** |
| LTX2.5 distilled，质量档 | 704×448×97帧，24fps，8＋3步，含音频 | 原生GPU，component_staged，quality，完整A/V/VAE/mux | batch热63.848s | 推荐质量档；不是同款跨层runtime ANE |
| 同LTX，速度近似档 | 同上 | GPU Sol＋text256，显式fast_approx | batch热61.869s | opt-in；97帧对quality MAE约25/255，非等价画质 |

Z的fresh/repeat来自同一份10月11日主记录；10月12日另组repeat约5.331s，
不是一个新配置突破。9月25日Public完整warm约5.393s属于较早构建/输入记录，
可作为历史佐证，不拿来替换fresh时间。LTX新进程clean-exec质量档另有62.639s，
其边界是finalizer的request_wall_estimate，不与上表batch中位数混成更快API结论。

主要依据：[Qwen扩展完整报告](../status/qwen-turbo-runtime-ane-2026-10-19.md)、
[Qwen GGUF快速配置](../status/qwen21-fast-defaults-2026-10-11.md)、
[Z原始主记录](../design/validation/z-valid-framework-current-20261011.json)、
[FLUX4B配方](FLUX4_CURRENT_STATUS.md)、[LTX完整配置](LTX_VIDEO.md)。

## 各模型的具体配置与复跑入口

### Qwen Base：固定Private FFN＋QKV是研究基线

原`models/Comfy-Org-Qwen-Image-2.1`，原BF16 transformer/encoder/VAE，
`execution=gpu_ane`、`hybrid_mlp_mode=runtime`、`allow_approximation=true`、
resident，base variant、40步、DiT cache off，无LoRA/encoder ANE。
FFN H4096/F12288，GPU/ANE各6144 intermediate channels；QKV固定512行。
动态W跨32层复用，各operator一个Program，不是每层一份常量模型。

28.705157s对应`build/qwen-function-session-v9-20261019/libturbocider.dylib`，
SHA256 `19bce742e096d846cbb84027672292e2e31502389b19b1eae43252232c617254`。
实际环境包括fixed-program、GPU IOSurface IO、F32 join、sigmoid-complement、
Private QKV和prefetch、text-prefix、text step0 GPU；GPU-dual与function family关闭，
W-code cache为0。不要把后来没有稳健净收益的实验开关一股脑叠加。

```sh
.venv/bin/python -B tools/profiling/qwen_base/collect.py \
  --library build/qwen-function-session-v9-20261019/libturbocider.dylib \
  --manifest examples/coreml/qwen21-runtime-shape-r1024.json \
  --route runtime --channels 6144 --fixed-program \
  --ffn-activation sigmoid-complement --private-qkv --private-qkv-prefetch \
  --case generate --mode stock --runtime-text-prefix-snapshot \
  --text-prefix-qkv-step0 gpu --output outputs/FRESH-base-fastest
```

输出必须用新目录。编辑改为`--case edit --runtime-prefix-snapshot`，去掉两个
text-prefix参数。该脚本是固定验证fixture，不是任意prompt/尺寸自动路由器。
复跑当前源码时将`--library`替换为自己的新构建；便携shape descriptor已随源码
提供，不依赖历史outputs中的小型配置。大图Turbo的R1536/R2048同目录另列，
使用`expanded_collect.py --manifest PATH`在初始化前选定。

原件：`outputs/qwen-function-session-gen-fixed-stock-v9-20261019/summary.json`；
完整审计：`outputs/qwen-function-session-audit-v2-20261019.json`。
首请求30.329s单列；热28.705s不代表fresh prompt或严格冷启动。
编辑最近ABBA原策略35.9518/35.9870s；GPU-step0新策略36.2194/36.1304s无提速，
继续默认关闭。后者改善的是seed73 miss/hit重放一致性，不能列为更快方案。
该轮独立审计：`outputs/qwen-edit-step0-independent-audit-v1-20261019.json`。

随后[共享compiled GPU complement的完整40步验收](../status/qwen-base512-compiled-complement-session-2026-10-19.md)
完成30张图片。首组小收益在反序基本消失，不把最低单窗更新为新最快配置；
上述原fixed FFN/QKV基线保持，不默认增加compiled-complement开关。

### Qwen Turbo：按尺寸在初始化前选固定FFN行数

原`qwen_image_2.1_turbo_bf16.safetensors`，明确Turbo variant、作者8步，
无LoRA；使用自己的Turbo scheduler。C6144、sigmoid-complement、F32 join，
GPU encoder；没有自动获得Base Private-QKV组合资格。

| 文生图尺寸 | 初始化固定rows | 热完整请求 | 正反/独立重复 |
| --- | ---: | ---: | --- |
| 512² | 1024 | 7.308s | 原三请求窗口 |
| 768×512 | 1536 | 11.156s | 独立重复11.033s |
| 1024² | 2048 | 35.767s | 反向重复35.795s |

大图仍只是一份初始化选定的固定图；不要在热owner里换descriptor后声称无新load。
同owner固定R1024变尺寸已验证功能，但不是每尺寸最快方案。多procedure提升的是
灵活性，尚无完整大图请求证据证明比此尺寸特定配方更快。

512生成原库`build/qwen-expanded-turbo-v1-20261019/libturbocider.dylib`，
prefix编辑原库`build/qwen-expanded-turbo-v2-20261019/libturbocider.dylib`；
大图原库`build/qwen-expanded-convrot-v4-20261019/libturbocider.dylib`，模型仍须
指向Turbo根目录，不能因库名convrot而换成ConvRot checkpoint。
完整环境/请求分别保存在`outputs/qwen-expanded-turbo-gen-runtime-v1-20261019/`、
`qwen-expanded-turbo-prefix-on-v2-20261019/`、
`qwen-expanded-turbo-768-runtime1536-v8-20261019/`和
`qwen-expanded-turbo-1024-runtime2048-v8-20261019/`。
768/1024完整审计为`outputs/qwen-expanded-size-lora-audit-v10b-20261019/audit.json`。
这些计时仍受外部CPU负载限制，不发布新正式加速比。

### Qwen GGUF与ConvRot：保持各自数值配方

GGUF的C5120、F32 join关闭、GPU encoder prefill/retention不可照抄BF16组合；
原文件是Q4_K_M DiT＋Q4_K_M encoder，VAE为原BF16。ConvRot则是原Comfy H256
旋转codes，Exp、F32 metadata/partial join；不重复旋转/重新量化原W8 codes。
两者最快记录不证明任意GGUF格式、LoRA、编辑、多尺寸或encoder ANE均获益。

最新完整两条配置都使用`build/qwen-expanded-convrot-v4-20261019/libturbocider.dylib`，
GGUF与ConvRot model根目录、descriptor/data path必须独立匹配。
原件分别为`outputs/qwen-expanded-gguf-runtime-v4-20261019/`、
`outputs/qwen-expanded-convrot-runtime-v7-20261019/`；ConvRot独立审计为
`outputs/qwen-expanded-convrot-audit-v4b-20261019/audit.json`。
GGUF当前记录36.959s，较早同配方约36.861s，不能把约0.1s窗口波动当新改进。

### Z-Image Turbo：当前最快仍Public frozen，不是新runtime

原Comfy BF16，8步、resident；image-only1024行、ANE5120/GPU5120 FFN通道，
caption整段GPU，每请求256次Core ML FFN。原主记录库为
`build/benchmark-async-v1-private/libturbocider.dylib`，但实际执行的是Public
frozen Core ML；Private-enabled库名不改变执行器身份。

已在本地的compiled manifest：
`models/coreml/z_image_w8a8_imageonly_hidden_c4_a5120_b1024_diverse8_compiled/manifest-0718a2ff4c18e5c283b91f45ec21d867cd6532e899e6b48520e8396efea20e72.json`。
请求`execution=gpu_ane`、`allow_approximation=true`、显式该manifest；复跑方法见
[Z CLI指南](../public/Z_IMAGE_ANE.md)。checkpoint和CLI/library必须匹配。

新Private runtime R1152/C5120的512²热观察约6.61–6.63s；同owner
512²→768×512→1024²→512²已过功能和零热compile/load检查，但没有干净速度资格，
也没有新的同窗口证据取代已有frozen基线。当前Z入口仅文生图，**不接受参考图**；
不能写作Z-Image Base、多参考编辑或任意尺寸都已验证最快。
其独立功能审计为`outputs/runtime-image-independent-audit-v2-20261019.json`。

压缩表示另保留[同库五格式对照](Z_IMAGE_FORMATS.md)，不是换格式即等价加速：

| Z-Turbo DiT格式，512²/8步 | 该完整格式窗口的最强配方 | 四fresh中位数 |
| --- | --- | ---: |
| INT8 ConvRot | Private Comfy H256 W8A8，C4096/F32 join | 7.310s |
| GGUF Q4_0 | Private Sylvester W8A8，C4096，保留packed bank | 9.008s |
| GGUF Q8_0 | 同上 | 9.302s |
| NVFP4 | GPU weight-only W4A16 | 8.844s |

BF16整体仍更快；GGUF较低内存不等于更低绝对延迟。这些格式配方不能套用到
Qwen GGUF或ConvRot，也不具有本轮Z多尺寸/参考图的新资格。

### FLUX4B：记录保留，重建前不可当即用配置

2.574s来自当前saved-Q4/g64来源派生：只将40个single投影预解码BF16，
再加载source-bound20个W8权重/FP16 IO与计算的FFN模型；不是W8A8激活量化，
也不是原packed UInt32直接ANE。完整对应请求在
[旧配置快照](best-local-configs-20261011.json)的4B条目。

已只读检查：原library仍在，但`outputs/flux4-single40-bf16-derived-v1-20261013`、
`outputs/flux4-single40-ane-compiled-v1-20261013`以及原packed native视图已在
之前获授权的清理中移除。原模型未删除；重建需重新生成派生视图并按源SHA导出/
编译，不能只复制manifest或使用旧hash。记录与恢复说明保存在
`outputs/qwen-expanded-retired-exports-v1-20261019.json`，本次没有再删除内容。
旧BF16模型的1.628s另属9月历史来源，不冒充当前Q4/g64可运行结果。

### LTX2.5：保留GPU，ANE不作为速度选择

原distilled ConvRot DiT/Gemma4 encoder、connector/upscaler、conv-only video VAE、
audio VAE，704×448×97帧、24fps、8＋3步、CFG1、完整有声MP4。
`execution=gpu`、`residency=component_staged`；quality批处理63.848s，
Sol＋text256显式近似档61.869s；quality clean-exec62.639s另列不同计时边界。
INT8 ANE quality批处理71.333s、Sol＋text25666.036s均更慢，不为开启ANE而选它。

原库及helpers为`build/benchmark-async-v1-private/`。完整参数/冷暖/媒体/原始环境
在`outputs/ltx-full-configs-v2-explicit-diagnostics-20261011/`，便携机器记录为
[ltx-full-configs-current](../design/validation/ltx-full-configs-current-20261011.json)。
当前逐层MLModel方案还有E5 resource/recompile记录；同款跨层动态W/GELU runtime
ANE尚未完成整视频验证，不能写成已有加速配置。

## LoRA与改变计算量的速度档，单独保存

10月19日新增[固定runtime ANE＋conservative残差缓存](../status/qwen-base512-runtime-conservative-cache-2026-10-19.md)：
Base40标准1024参考编辑热**26.510s**，独立反序**26.159s**；原runtime约35.86/35.80s。
本批对完整GPU三张编辑RGB SSIM0.9961–0.9964、alpha差1，过暂定数值预算，
但只有单参考/一个指令/两seed，且外部负载门未过，不改默认。
其生成约20.943s，却未通过对完整GPU的累计预算，不列为质量合格的最快生成档。
这两者每请求少算360个middle blocks，仍有40次采样；不替代上方完整计算表。

代码整理时补完三组编辑holdout，共九窗口/27张：添加柠檬、写TEA两组在暂定
RGB/alpha预算内，透明玻璃组有两张alpha差18/11，未过≤4的预算；对原runtime
的增量alpha也有5/7。不能继续把红茶壶结果推广为所有编辑均可接受。
缓存档继续默认关闭，完整计算配方不变；详见[集成验证](../status/runtime-ane-integration-2026-10-19.md)。

| 独立档位 | 当前配置与观察 | 为什么不混入Base完整计算冠军 |
| --- | --- | --- |
| Viggle v0.2.1 r128，作者6步 | 强GPU direct-QKV＋typed LoRA/SwiGLU；生成约7.208s，ref512 prefix-hit编辑约7.329s | 训练学生LoRA，不是Base40；不是任意普通LoRA资格 |
| 同adapter，作者可选5步 | GPU＋上述融合；生成约6.157–6.171s，编辑hit约6.237–6.245s | 少一次采样更新，是不同画质档 |
| 同6步LoRA，Private C4096 | 生成7.139s，只有约0.95%弱信号；编辑8.653s慢于GPU | 不默认选ANE；共享rank/split-down也未改善 |
| Public frozen＋conservative DiT cache，40次sampler更新 | 生成约22.21–22.46s | 15步复用360个middle blocks；对full GPU的整体误差仍需考虑 |
| Public frozen＋balanced /fast诊断 | 单样本生成19.511 /17.493s | 分别复用504/600个middle blocks；不是kernel同工作量提速，非公开named-preset资格 |

5步结果/完整矩阵见[typed LoRA/SwiGLU完整报告](../status/qwen512-lora-swiglu-request-2026-10-18.md)；
6步GPU/ANE比较见[LoRA扩展](../status/runtime-ane-lora-and-model-expansion-2026-10-19.md)。
不同研究开关并不都可组合，不把各轮增益相乘；普通风格LoRA、Turbo＋LoRA或
ConvRot＋LoRA没有因此自动获得速度/质量资格。encoder目前优先GPU。

## 其它模型与任务

SeedVR2为超分，不和生成时间混排。稳定FP16/FP32 resident GPU、单步、无LoRA：

| 完整超分任务 | 同配对原生GPU中位数 |
| --- | ---: |
| 图像256→512 | 0.828s |
| 图像512→1024 | 3.506s |
| 视频256→512，25帧/24fps | 20.586s |
| 视频512→1024，25帧/24fps | 85.011s |

这些是[已保存Comfy配对记录](SEEDVR2_COMFYUI.md)的原生配置，不声明穷尽所有
SeedVR2内存/分块候选；速度门通过，原整体资格仍false。

FLUX9B当前保留原BF16 compiled/deferred/fused GPU512²/4步的7.374s fresh
记录，原模型与库仍在；它是单独的9B历史条目，不替代4B的2.574s。
Wan、LLaDA与Z-Image Base没有在本轮建立统一最新最快配置表，暂不填写推测时延；
H3属于LLM任务，不混入图像/视频表。未测项不由组件GEMM、LLM prefill或其它芯片
数字外推。

## 本次整理和后续维护

已只读核对主summary/便携证据中的时间与实际backend，以及选定库、模型根目录、
manifest的存在状态；不是重新加载模型、全量复算每个历史SHA或重新认证性能。
本页链接到各自完整来源记录；旧的10月11–15日对Comfy/mflux倍率保持原窗口，
不使用本页更快的热命中时间回填外部分母。

中间日志、PNG/MP4、build、权重和raw profiling仍留在gitignored目录；本次仅加
汇总文档与入口，不生成大产物，不改自动路由。更新此页时必须同时保留模型来源、
实际步数/计算复用、尺寸/参考数、fresh/repeat、后端与产物可用性。
