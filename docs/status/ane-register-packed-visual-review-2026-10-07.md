# ANE/GGUF 寄存器 staging 与原尺寸视觉对照

2026-10-07，Asia/Singapore，M4 Max 64GB。接续
[ConvRot A8 staging](convrot-register-a8-staging-2026-10-06.md) 和
[FP32 partial join](native-fp32-partial-qwen-quality-2026-10-06.md)。

## 当前验收口径

按用户最新要求，最终目标是 **matched optimized GPU 之上有实际加速**，
不再强制 ≥1.2×；W8A8 以原尺寸生成图像视觉非常接近为目标，latent
N1 数值门槛保留为补充诊断，不再是唯一质量判据。历史报告、N1 阈值和
失败回执均未改写。正式 load / memory / artifact / actual-execution
检查继续保留。本轮没有完成全部 Z/Qwen、base/LoRA、GGUF/ConvRot、
Public/Private 的端到端资格，也没有宣布自动 per-operation 三后端路由完成。

## 已提交的实现

`0ac5a21` 将 Dense Sylvester H128/H512 的 W/A staging 移至单个
32-lane SIMD group。每 lane 持有4/16个值，低层 butterfly 用 shuffle，
高层交换 private registers；旋转不再使用 threadgroup scratch/barrier。
保留原 FP32 加减顺序、sign/normalization、normalized FP16 scales 和
RNE codes。原 `TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE=0` 保留 generic
control，1显式启用特化；默认 Public 路径不变。

`ec4d474` 把同一 kernel 扩展至 affine Q4/Q8 和 raw GGUF Q4_0/Q4_K/
Q8_0/Q6_K：直接解码压缩输入至寄存器，再 rotate/requantize。没有完整
dense 权重中间副本，没有 sidecar 或模型文件重写。直接 ConvRot signed/
packed W 已在 rotated basis，仍走原 direct kernel，不会再次旋转。
Comfy H256 A8 仍用上一轮独立 radix-4 寄存器实现，不混用 Sylvester。

pipeline key/上限26、W/A banks、scale-cache 容量、owners/shared events
均不扩张。后端失败仍需整个受影响操作 GPU 重算，不发布部分 scratch。
这些 GPU staging 优化不证明 ANE 原生 INT8 MAC 或物理 GPU/ANE overlap。

## Dense 同 binary 组件测量

保留回执 `outputs/sylvester-register-stage-components-31samples-20261006.json`。
BF16 合成输入，每臂10次 warmup、31个串行交替 samples；全部 codes、
scales、physical padding 按 bytes 相同。source 非 immutable，测量包括
scale 和 code 两个 pass，不是热 scale-cache hit。

| rows / columns / H / layout | generic ms | register ms | 局部倍率 |
| --- | ---: | ---: | ---: |
| 4096 / 3840 / 128 / W | 1.020333 | 0.476875 | 2.13962× |
| 3840 / 4096 / 512 / W | 1.206417 | 0.839208 | 1.43757× |
| 5120 / 4096 / 128 / W | 1.326958 | 0.591833 | 2.24212× |
| 4096 / 5120 / 512 / W | 1.647000 | 1.058542 | 1.55591× |
| 7168 / 4096 / 128 / W | 1.861459 | 0.743792 | 2.50266× |
| 4096 / 7168 / 512 / W | 2.148709 | 1.374625 | 1.56312× |
| 1056 / 4096 / 128 / A | 0.420333 | 0.284000 | 1.48005× |
| 4224 / 4096 / 128 / A | 1.176541 | 0.557750 | 2.10944× |

这是 generic 对 typed-load/function-constant/register **组合**特化，不能
把整个收益单独归因于寄存器交换。包含 command/host-ready，不是 GPU
timestamp 或完整 FFN/E2E。observer13次，max gap0.15434s，artifact未变。

## 压缩输入同 binary 组件测量

命令：

```sh
bash tools/native/build_private_ane_sylvester_stage_benchmark.sh
.venv/bin/python tools/native/run_gpu_component_screen.py \
  --output outputs/packed-register-stage-components-31samples-20261007.json \
  --interval .1 --timeout 900 -- build/sylvester-stage-benchmark/probe 31 packed
```

24个合成压缩输入 case，原 decoder metadata 精度不变，仍是10次 warmup/
31个串行交替 samples；每格完整 codes/scales/padding 相同。每种来源都测
4096×3840/H128、7168×4096/H128、3840×4096/H512、4096×7168/H512。
下面分别列这两个 H128 / 两个 H512 shape 的倍率，不是置信区间。

| source | H128 两格倍率 | H512 两格倍率 |
| --- | --- | --- |
| affine Q4 | 1.6522× / 1.7223× | 1.1513× / 1.1722× |
| affine Q8 | 1.6505× / 1.7538× | 1.1852× / 1.1809× |
| raw Q4_0 | 2.0052× / 2.0415× | 1.5033× / 1.4446× |
| raw Q4_K | 1.7117× / 1.7824× | 1.2702× / 1.2982× |
| raw Q8_0 | 2.1386× / 2.2383× | 1.4648× / 1.4830× |
| raw Q6_K | 1.9324× / 2.0486× | 1.3616× / 1.4259× |

35次 host observation，wall4.26857s，max gap0.15407s，无 observation
error，artifacts_unchanged=true。没有 owned build/test/model 与 timed
probe 重叠；未隔离桌面/外部工作。这仍不是正式 competing-load gate、
完整模型或硬件 overlap 资格。原始 samples 全保留，不挑样本。

## 四格实际 Qwen 生成与数值补充

使用 Dense-register v1 同一 Private CLI/dylib，GPU / generic / register
每臂独立进程，12个进程串行。fox prompt/seed42；base40步，真实 Viggle
rank256 LoRA6步、strength1、原 FP32 rank，227 applied projections。
512固定 ANE channels5120，1024固定7168；chunks1、fixed-async1、F32
partial join、scale-cache/launch-fence1；prefetch/lookahead/deferred0。
1024 LoRA 的既有显式 diagnostic 开关在三臂一致施加。

| cell | actual Private calls | final relL2 / cosine | 历史 N1 / actual ANE |
| --- | ---: | --- | --- |
| Q512 base | 1280 | 0.0226912881 / 0.9997425403 | pass / yes |
| Q512 LoRA | 192 | 0.0168074970 / 0.9998587563 | pass / yes |
| Q1024 base | 1280 | 0.0156643687 / 0.9998777995 | pass / yes |
| Q1024 LoRA | 192 | 0.0207968509 / 0.9997847861 | pass / yes |

四格 generic/register conditioning、initial/final latent、PNG 逐位一致。
四格候选 headroom1、overflow retry0、runtime failure0、fallback0。
GPU-reference 与 staging parity 是不同控制，均保留。Q1024 LoRA 这次
是实际固定 share 的 ANE execution，不是旧 auto decline 的 GPU-only。
但固定 share 可执行并不证明它盈利，也不表示 auto 应改为接受。

目录 `outputs/native-sylvester-register-q{512,1024}-{base,lora}-20261006/`
保留三臂 observed receipt/request/dumps/PNG、质量及 parity JSON。
`qualification_passed=false` 仍保留，不把一次 N1 pass 转成发行资格。

## 视觉证据：与数值报告分开

新增 `tools/validation/runtime_ane_visual_review.py`：只用 CPU/Pillow，
保存原尺寸 GPU/candidate side-by-side、同坐标 center/top-left/bottom-right
256×256（小图取原尺寸）裁剪，scale=1，无 resize/alpha 丢弃或 blending。
manifest 绑定原图和各 sheet SHA256；已有 output 拒绝覆盖、前后 hash
变化拒绝、geometry/mode/format 不符拒绝，review 初始为 pending。
脚本永远不根据 SSIM/N1 自动写 perceptual pass。

`outputs/visual-review-20261007/` 中准备并由本轮 agent 目视检查8组 whole
及每组三张 detail crops（32张 sheets）。Q 使用上面 register v1 GPU/
register；Z 使用历史 `native-final-quality-z*-20261006` GPU/runtime
W8A8，两臂相同旧 library，不冒称新 packed-register 模型运行。

| cell | 本轮 agent 视觉观察（单 fox / seed42） |
| --- | --- |
| Q512 base | 姿态、脸部、松针和积雪极接近；毛发局部纹理有细微变化 |
| Q512 LoRA | 五官、轮廓、尾部和背景一致性很高；细纹略变，无明显新伪影 |
| Q1024 base | 全图及胸部毛发裁剪非常接近；雪粒/细毛存在小差异 |
| Q1024 LoRA | 构图、眼鼻、松枝一致性很高；胸部细毛/地面纹理略变 |
| Z512 base | 整体很接近；脸部细毛、胸部纹理、背景虚化轻微改变 |
| Z512 LoRA | 整体很接近但不是不可区分；额头/耳部轮廓和尾毛有可见变化 |
| Z1024 base | 颜色、主体、姿态很接近；眼部/胡须/胸毛和尾毛局部有变化 |
| Z1024 LoRA | 主体与光照很接近；毛发纹理和轮廓、背景细节略变 |

本组未见明显新增棋盘格、色块、断裂肢体或全局色偏；这是 agent 的
有限视觉判断，不是用户审批、盲测或多 prompt/seed 的普遍质量保证。
自动 manifest 仍 pending，人工观察在本记录，不覆盖旧质量回执。
Z 原 latent N1 fail 仍是历史事实，但不能自动等同于最新视觉目标失败。
需要继续人物、文字、细小重复图案、高对比/极端 prompt 和多 seed 检查。

## GGUF 提前解码和混合后端：当前分析

当前两个优化方向不能混为一谈：

1. **本轮已做**：raw/affine→register rotate→W8 IOSurface，避免完整 dense
   解码 intermediate，缩短现有 ANE staging；不改变模型容器。
2. **既有研究窗口**：`AffineDenseWindow` 是同步 owner-only、一/两矩阵
   retention，保留 content ticket/generation/escaped-reader ledger claims，
   尚未自动接入跨 step 的模型消费/异步 predecode。

既有 [消费算子记录](convrot-register-dense-window-2026-10-06.md) 的回本
条件是 `decode/R + dense_GEMM < packed_QMM`：512 GGUF gate约23/28次，
down约46次，ConvRot约7/8次；1024大M约2–3次。逐层轮转、每步 eviction
并不构成同矩阵命中。不能用两槽、denoise steps 数或分离 timing 推断
已存在复用/overlap。每张 Z FFN projection dense75MiB，三个225MiB，
两个完整 FFN450MiB，原150MiB两矩阵 window不是两个完整 FFN。

因此目前优先保留压缩 master，直接 W8 staging；dense predecode 只适合
有预算且确有跨调用命中的热点。后续必须记录真实 hit/miss/eviction、
conversion成本和整请求 peak；异步需从 pager content generation/
ready event 接入，不允许先 publish 未完成或非有限 target。

Public/Private/GPU 共用 graph/weight/scheduler 已有实现；下面是下一步
**候选路由策略**，不是本轮已经实现的自动三选一。

| 操作条件 | 候选 consumer | 必须验证 |
| --- | --- | --- |
| 长行、允许 W8A8、已授权、转换可回本 | Private ANE + GPU channel complement | 完整 traffic calibration、视觉、matched E2E |
| 可分发/兼容性优先、FP16 模板合适 | Public Core ML + GPU row complement | 执行/转换成本、头部 GPU 重算、实际 residency |
| 短行、敏感激活、缺预算或未盈利 | 原 optimized GPU | 不产生桥接、完整 operation fallback |

Private→Public 的 capability fallback 当前在 factory 构建期；并不是已经
按每层性能自动动态切换。相同 weight slice 不应同时常驻 Public FP16
和 Private W8 两份；identity 要绑定 backend/ABI/recipe/LoRA/content。
Public 不承担私有客户端 API，Private 继续 opt-in、发行隔离。

### Public W8A8 不能简单等同于 Private runtime-weight 图

2026-10-07核对 Apple 官方
[Quantization Overview](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html)、
[API Overview](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-api.html)、
[Performance](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html)：
官方支持 W4/W8、A8；其 activation PTQ 使用样本校准与 per-tensor scales，
并说明 M4 上 NE 的 INT8×INT8 可获益。这支持继续评估 Public W8A8，
不支持把本项目当前 FP16 runtime template 或 Private normalized-I8 图
直接称为已执行该硬件路径。

官方也提醒 activation量化在CPU/部分GPU路径可能因runtime转换而变慢，
建议主要落在NE的模型使用；NE建议per-channel weight scales。下一轮
需分别验证 runtime输入weights的合法图、实际执行放置、数据binding/
compile摊销、真实LoRA换代和视觉校准，不盲套整模型静态压缩API，也
不为每层/adapter创建永久model副本。当前Public实现仍是FP16桥接。

同日核对 [MLX QMM API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html)：
affine QMM消费uint32 packed weights以及group scale/bias。这不等于所有
raw GGUF布局可以原样传入QMM；本轮raw GGUF→W8 decoder与既有
raw GGUF→MLX affine consumer是不同representation路径。

## 回归与 provenance

Dense-register v1 补跑：123项 host tests、3项 Private prepared/channel/
MLX tests、7项 Public graph/Core ML/MLX tests通过。Public实际库 release
guard通过，480 source inputs 独立匹配且 manifest seal一致。
上一轮测试选择把 prepared test 误写在 HardwareTests，得到6项实际通过
加1项 loader AttributeError，**不是7项绿色**；本轮用正确的
`PrivateAneCalibrationMlxTests` 显式环境开关重跑通过。

Packed-register：4项 actual Private hardware tests通过，包含 stager、
Comfy/direct独立路径、两bank actual W8A8、4224-row/多chunk/真实LoRA
hidden/失败恢复；最终又独立补跑stager1项，通过新增 raw/affine 每种
来源的非有限scale/normalized-scale overflow拒绝和clean refill。
所有9 encoding/dtype组合覆盖H128/H512、W/A transpose、独立非对齐
source offset/pitch、非零row/column slice和CPU oracle，pipeline仍18。
另2项 CPU视觉工具 tests通过。不是全仓/全部Private/Public/Swift验收。

最终Packed v1库又补跑13项Public graph/Core ML/MLX integration、3项
Private prepared/channel/MLX、125项host（包含2项visual工具）回归，
全部通过、无skip。配合上述4项hardware，共145个不同test case，额外
stager复跑不重复计数。对应日志为
`outputs/packed-register-{public,private-mlx,host}-regression-20261007.log`
与`outputs/packed-register-private-regression-20261007.log`。

Dense v1 Private library SHA256
`19cd011dd213df680acd23fd89172d0d8b74ea27177bec9bcf0cd814169e663b`，
build ID `tc-runtime-build-v1-84c65ace746cc6e0782538d145db2d8cc7818b35547cf17dbfeb6752fd1a3fbf`。
Dense v1 ordinary Public library SHA256
`878f6e6876e9083fc801410d39258249f4d61d67a506446ab174de616d88ee8d`，
build ID `tc-runtime-build-v1-b78976fa752b4603bbfa53cb2eec2a3ee4ac722c762446e73dda438688e86762`。
匹配检查发生在 packed扩展编辑之前，不冒称历史库匹配新增source。

Packed v1 Private native-only完整构建完成，480 source inputs独立匹配，
manifest seal一致；library SHA256
`60e6ffe99a8edcee5430cfa0258ea9f26d70af142408811ac3d3e28a95efb41c`，
build ID `tc-runtime-build-v1-78f28a03994e5f4e5a3f70dd45d1cb1bc1b27315a6ca063f6ae558f79d035464`。
Packed v1普通Public library-only构建成功，480 source inputs独立匹配、
manifest seal一致、actual binary release guard通过；library SHA256
`7f83e3fb58605d4fb05579382cc6b9b2eb46b90892897ee7a62c79d5b140f6dc`，
build ID `tc-runtime-build-v1-d3df3d9d0da34f986228e8454795baec4791c50bc34a6d6b3c3e61f117420244`。
Dense/packed组件回执SHA256分别是
`71d297559328bdbe462a194df993cac8f3eb4057fff4765e173a492eeb10bbce`、
`5153f95840059e5ae36862756d20b17e1bfd3b6fcbba98a6dd7556b9e847ccf3`。
既有 Metal `fastMathEnabled` deprecation warnings 保留，不改为fast math。
各构建包含用户原未提交 ConvRot草稿，不是clean staged-only构建；本轮
selective commits不夹带它们。models/adapters/references/原design只读。

## 正式速度边界

Dense v1之前的 Q512 base forward3 与 Dense v1 Q1024 LoRA reverse3
均被外部 ComfyUI CPU样本判污染；对应 summary incomplete/trials=[]。
未杀外部进程、未放宽门槛、未拼接GPU分母。它们不能提供有效倍率。

Packed v1 Q512真实LoRA一冷三热、GPU→runtime的严格load/memory screen
也拒绝了污染窗口：GPU四次请求完成，67个continuous load samples中
1个外部ComfyUI CPU样本触发拒绝，runtime臂尚未开始。
目录 `outputs/native-packed-register-q512-lora-forward3-20261007/` 保留
全部GPU请求/图片/stdout/load/memory及incomplete summary、零accepted
trials；run正确退出失败。即使launch前preflight空闲，也不替代连续检查。
局部 staging 加速不替代整模型快于GPU，也不证明两引擎物理重叠。
完整目标仍active。
