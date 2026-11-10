# LTX 2.5 base：完整音视频与实际框架边界

## 2026-11-10：可选 PrunaVAED 原生解码

新增 `ltx_video_vae: "prunavaed"`，需要 `allow_approximation: true`；默认仍是
`"conv"`。v1 放在请求顶层，v2 放在 `execution` 中。示例见
[video-ltx25-prunavaed.json](../../examples/requests/video-ltx25-prunavaed.json)。
encoder、图生视频输入编码和 latent upsampler 继续使用原 Conv VAE。
支持普通 resident / component_staged / streamed residency；尚未资格化的
public streaming selector 和 memory_constrained 请求会显式拒绝。

App 操作：打开「模型库 → VAE 模型」，点击 PrunaVAED 的「从 ModelScope
下载」，完成后点击「用于 LTX 创作」；也可以在 LTX 创作页的「高级参数 →
视频 VAE」中选择 Conv VAE 或 PrunaVAED，并直接下载缺失的权重。选择随
草稿保存，复用历史任务时恢复，生成详情显示实际使用的 VAE。
PrunaVAED 暂需关闭校准内存档位，可使用普通分阶段或预算式流式加载。

App 下载器固定 ModelScope revision、大小和 SHA256，显示进度、支持取消，
通过校验后才登记到共享模型库。CLI 可调用相同下载器：

```bash
build/native/turbocider-library download-vae prunavaed
```

模型库权重独立于 LTX 基础模型。App 自动传递绝对路径
`ltx_video_vae_path`；CLI 也可显式设置此字段（v2 放在 `execution`），
仅在 `ltx_video_vae: "prunavaed"` 时接受。省略路径时读取
`<model-root>/vae/prunavaed.safetensors`。切回 Conv 时继续使用基础模型的
原始 VAE。

验证覆盖旧草稿默认值、选择持久化、历史复用、跨模型切换、v1/v2 请求、
下载完整性与取消/复用，以及 worker 拒绝解码器不匹配的结果。
另通过打包 CLI 和生产 App worker 完成 256×128×9、11 步、12 GiB
streamed 的实际生成；产物经 AVFoundation 验证，结果记录
`ltx_video_vae: "prunavaed"`。这条小尺寸冒烟不用于性能结论。

从 [ModelScope PrunaAI/PrunaVAED](https://modelscope.cn/models/PrunaAI/PrunaVAED)
下载、校验并安装到 LTX 模型目录：

```bash
.venv/bin/python tools/models/download_prunavaed_modelscope.py \
  --install-root models/LTX-2.5-INT8
```

权重为 1,327,909,418 bytes，SHA256
`4cbf0cbe6c185514d62c6c58c35dc42d7ea15924f34391e08be12f44bfccdf1d`。
原文件保留在 `models/PrunaVAED/vae/diffusion_pytorch_model.safetensors`，
安装位置为 `<model-root>/vae/prunavaed.safetensors`，同盘使用硬链接。
下载器固定 revision、校验大小/SHA256，支持断点续传，不执行下载的 Python。

原生 MLX 直接读取 Diffusers safetensors，校验全部 decoder tensor 的名称和
形状，只物化 decoder 与 latent 统计量。实现两个带 affine LayerNorm 的
残差投影、裁剪后的通道数、非因果时间 padding 和原始 patch 重排。
加速包括加载时预排布卷积权重、1×1×1 shortcut 使用 GEMM、融合 SiLU，
并复用现有空间分块路径。运行时无需 PyTorch / Diffusers。

M5 Pro / 24 GiB，同一个真实 LTX-2.5 latent，输出 704×448×97，关闭
额外近似；每种 decoder 预热一次，交替顺序各测三次，使用新进程。
下面仅统计 decode compute，不含权重加载、RGB 转换、编码和 DiT：

| Decoder | 解码中位数 | 相对 Conv |
| --- | ---: | ---: |
| 原 Conv VAE | 13.295587 s | 1.00× |
| PrunaVAED，未融合 SiLU | 5.981037 s | 2.22× |
| PrunaVAED，融合 SiLU | 5.912960 s | 2.25× |

主要收益来自剪枝网络；本次 SiLU 融合约多省 1.1%，不单独宣称稳定大幅收益。
`TURBOCIDER_PRUNAVAED_FUSED_SILU=0` 可用于 probe 对照。复现脚本为
`tools/validation/benchmark_prunavaed.py`，独立 probe 构建脚本为
`tools/native/build_ltx_video_vae_probe.sh`。

官方 patched Diffusers FP32 oracle 对照（`prunavaed_parity.py`）：
9×64×96 输出 RMSE 0.002930、最大绝对误差 0.016936；单帧96×64输出
RMSE 0.002538。原 Conv 小样本改动前后逐字节一致。
同一真实 latent 的全部97帧，PrunaVAED 相对原 Conv 的像素 PSNR 为
32.07 dB（[-1,1]范围），不是无损替代；融合相对未融合为57.61 dB。
上游训练/效果评估针对 LTX-2.3；本地确认 encoder、未剪枝 decoder 前段及
latent mean/std 共106个 tensor 与2.5 Conv逐字节一致，且完成这一个视频
样本的对照。更广泛的提示词、文字、细节和运动质量仍需额外评估。

本地原始记录：`outputs/prunavaed/benchmark/benchmark.json`、
`outputs/prunavaed/benchmark/pixel-comparison.json`、
`models/PrunaVAED/ltx25-conv-compatibility.json`。以下为历史完整AV基准，
不与本节的独立 VAE 测速混算。

最新2026-10-12已补Comfy v0.39同源97帧完整AV：cold142.925s、单warm
112.230s。按用户要求复用后续TC最佳quality63.848s／opt-in近似61.869s，
为1.758×／1.814×；[完整重点对照](COMFYUI_MODEL_COMPARISON.md)明确
跨窗口和近似漂移，不重跑TC。本页下方五进程是较早原GPU/ANE阶段，
不拿其64.126s替代后续最佳配置或拼成新paired median。

2026-10-11，M4 Max64 GiB/macOS26.6.2。当前本地LTX distilled INT8 ConvRot
DiT/Gemma、原connector/upscaler/video/audio VAE；无LoRA。原生C/Metal，
component_staged、704×448×97、24fps、8+3=11模型步、seed42，狐狸雪景
prompt，输出H.264 +48kHz stereo AAC、音视频时长约4.041667s。

GPU→INT8配置→FP16配置，各新进程一cold/一同prompt warm；之后独立
INT8→GPU反序。warm是实际condition disk cache hit；各arm独立空cache
开始，未偷用其它arm的conditioning。完整request_wall包含模型/文本/
两stage/upsample/两个VAE/音频vocoder/MP4输出，不只统计DiT。

## 五个完整进程、十段完整视频

| 路线 / 顺序 | cold s | warm s | tree peak GiB |
| --- | ---: | ---: | ---: |
| GPU forward | 72.195629 | 64.192437 | 47.106 |
| INT8 Core ML profile forward | 87.101623 | 64.103796 | 47.111 |
| FP16 Core ML profile forward | 115.819748 | 78.114023 | 47.142 |
| INT8 Core ML profile reverse | 75.202054 | 63.046128 | 47.079 |
| GPU reverse | 72.958291 | 64.060235 | 47.182 |

两warm中位GPU **64.126336s** /INT8 profile **63.574962s**，名义约
1.009×、少0.86%；仅两样本，不能称稳定的明显加速。FP16仅一warm，
78.114023s，比GPU慢约21.8%。两个INT8 cold都比自己的GPU方向更慢。
因此当前推荐仍GPU，不为了填“混合赢家”就把微小差值或慢路线推广。

profile与原本地48个Core ML block manifest严格配对，H4096/F16384、
ANE prefix6912/GPU complement9472，支持stage1 rows1001/stage2 rows4004。
INT8/FP16 profile只选择已有compiled artifacts，未导出/改权重；保留
parallel A/V、stage2 preloading、完整GPU MLP释放与stage detach/最终block
释放策略。receipt是native配置的gpu_ane与ane_profile，不伪造图像runtime
的call counter；没有物理NE占用/native integer MAC的trace资格。

## 图像/运动差异仍需保留

同源完整97帧RGB解码对照GPU与INT8/FP16 warm，不resize、不只看首帧：

| 对照 | 平均RGB correlation | 平均cosine | 平均MAE /255 | 最大motion相对误差 |
| --- | ---: | ---: | ---: | ---: |
| INT8 profile vs GPU | 0.858522 | 0.978402 | 20.4948 | 5.34% |
| FP16 profile vs GPU | 0.876494 | 0.981419 | 19.0182 | 5.99% |

这些轨迹差异明显超过Z/FLUX单图的几/255差异，不以用户允许较宽delta
宣称完全接近。历史latent门失败仍保留；这里是新的decoded-media对照，
不是新的严格latent或广泛感知通过。更快但误差较大的候选可以列入
performance表，不因此作为quality-safe默认。

## 内存范围与压力

五个独立100ms进程树verifier complete，含确切bundled
`ltx-video-vae-decode`子进程；不是只观察Transformer parent。scope是
load+cold+warm+helper+exit，仍不归因所有driver/外部Core ML服务。

forward GPU/INT8/FP16峰值约50.58/50.58/50.62GB（十进制）。对应系统
compression累计增量约22.91/29.27/43.39GB；FP16窗口有56,164,352bytes
swap-out，GPU/INT8为0，所有swap-in和reverse值都列入记录。累计活动不是
当前占用，不能说无压力或同低容量实机资格。VAE isolation实际为process。

## mlx-serve原组合：stock preflight拒绝，不虚构latency

26.10.1官方binary识别为native video engine，本地原config=AudioVideo、
model_version2.5，随后默认memory preflight报：media peak约55.54GB，
加warmup headroom需约62.5GB free，但当时available51.84GB。返回
InsufficientMemory/LoadFailed，exit1，无generation request，owned server
terminal。N/A是当前stock配置无法加载，不是“0.26s比GPU快”，也不声称
mlx-serve整个框架不支持LTX或其converted4/8bit pack也不能运行。

未为了制造对照关闭其它用户app、调整swap/系统RAM策略或强制跳过安全
预检。其converted布局还使用独立flat transformer/vae文件；如未来转换
权重/去重组件，必须单列来源、表示和media返回/客户端mux边界，不能
拼接第三方不同checkpoint、步数或raw-frame吞吐当本地原MP4对照。

## 证据与失败窗口

首个GPU窗口两段视频完成，但memory sampler未登记VAE helper角色，
报告unknown_child/inconclusive；该失败保留，不追认为accepted sample。
只增加精确resolved decoder executable角色后，新GPU/INT8/FP16与
反序全部重跑，仍拒绝其它unknown children和sampling gaps。

[机器记录](../design/validation/ltx-base-current-20261011.json)绑定原source/
library/helper/profile SHA、十个request/media、冷/热/stage、五条完整
memory链、原stock框架拒绝、完整97帧RGB/motion与旧失败。原始MP4、
dependencies/cache/binaries都不进Git。两种成功Core ML配置不当作
全设备/多prompt/多seed/physical-overlap或Public发布资格。

## 新完整配置筛选：32段完整MP4与20条memory完成

outputs/ltx-full-configs-v2-explicit-diagnostics-20261011已完整完成：
quality/Sol/text256的GPU与INT8 batch，加GPU/INT8 clean-exec，8配置
16arm正反序、每arm两段完整MP4；32段均完成，20个独立memory窗口
complete。源/步数/尺寸不变、无LoRA；Sol
和text256为另外的attention近似，不能与quality宣称同数学。
batch request_wall与clean finalizer request_wall_estimate/新进程
conditioning-hit分开，不为了更快数字混边界。原64.126/63.575s与
质量差异仍保留，不与下面新窗口混池。

| 路线 | 正/反warm s | 中位 s | peak GB（十进制） | 计时边界 |
| --- | --- | ---: | --- | --- |
| GPU quality batch | 63.769 /63.927 | 63.847972 | 50.58 | request_wall |
| INT8 quality batch | 67.681 /74.985 | 71.333099 | 50.54–50.77 | request_wall |
| GPU Sol batch | 62.856 /62.842 | 62.848950 | 50.58 | request_wall |
| INT8 Sol batch | 66.601 /67.730 | 67.165562 | 50.65–50.77 | request_wall |
| GPU Sol+text256 batch | 61.998 /61.740 | 61.868682 | 50.57–50.76 | request_wall |
| INT8 Sol+text256 batch | 70.637 /61.435 | 66.036188 | 50.70–50.76 | request_wall |
| GPU quality clean-exec | 62.633 /62.644 | 62.638723 | 37.60 | request_wall_estimate |
| INT8 quality clean-exec | 68.999 /62.481 | 65.740014 | 42.92–42.93 | request_wall_estimate |

batch最快为GPU Sol+text256，对自身quality少3.100%；同attention recipe
INT8比GPU慢6.736%。quality INT8比GPU慢11.723%，不沿用旧0.86%来
宣称当前稳定收益。INT8 text256的61.435s最快单样本仍保留，但与
70.637s一起计算中位数，不删除慢值。clean-exec同口径GPU最快，但
其新进程/estimate不能与batch算1.2s“API加速”。cold数组全部在记录中。

每候选warm MP4相对quality GPU的原尺寸97帧RGB/motion完整比较：

| 候选 | mean correlation | MAE /255 | 最大motion相对误差 |
| --- | ---: | ---: | ---: |
| GPU quality clean-exec | 1.000000 | 0 | 0 |
| GPU Sol | 0.836078 | 23.026 | 7.92% |
| GPU Sol+text256 | 0.815742 | 25.030 | 7.97% |
| INT8 quality（两边界） | 0.858522 | 20.495 | 5.34% |
| INT8 Sol | 0.807217 | 25.378 | 8.50% |
| INT8 Sol+text256 | 0.801190 | 26.193 | 6.71% |

已查看quality/text256首、中、尾帧，均是雪林奔跑狐狸，构图和树木/
身体细节有差异。允许较宽delta不意味着无损；仅3.1%收益伴随明显
视频漂移，保存performance opt-in，默认建议仍quality GPU。clean-exec
97帧视频RGB exact，不据此宣称AAC waveform/完整MP4 byte-exact。

可定位的瓶颈：GPU warm model_load约3.4–3.8s，INT8约8.5–17.7s。
INT8 stage1约20.9–21.5s比GPU23.5s短，部分stage2也更短，却被
加载/重编译、执行边界与波动抵消；不能说ANE数学完全没有收益。
raw里共76条E5 resource/recompile消息显式保留。system compression
每窗口累计0–37.59GB、swap-out总活动约5.33GB，不称全窗口无压力。
physical peak是process-tree范围，不包括所有外部Core ML服务/driver。

[完整配置便携记录](../design/validation/ltx-full-configs-current-20261011.json)
绑定514输入/所有helper/source/profile、32MP4/20memory/raw stdout、
诊断、cold/warm、每stage与97帧统计；完整逐帧数组由full_audit SHA绑定。
[已测配置](best-local-configs-20261011.json)分别保存batch performance
与clean-exec quality，不跨边界选赢家、不自动改生产默认。成功外部
框架/更多native profile与720p缺口继续pending，不因此完成扩大目标。

v1保留failed：GPU两段完成，INT8两段也完成且memory complete，但
Core ML的5条Missing E5 resource/recompile消息与首个JSON粘连，旧
采集器JSONDecodeError，不作为完整配置排名。v2只显式识别并记录
这类消息，raw stdout SHA保留；未知/截断文本仍拒绝。cold的recompile
开销不扣掉，也不把exit0/完整MP4当物理ANE执行或无警告资格。
Sol/tau/edge/text-row实际配置按native plan位置核验，而非猜顶层字段。
