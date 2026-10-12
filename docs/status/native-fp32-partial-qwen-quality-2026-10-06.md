# 共用 F32 GPU partial：Qwen 最终 N1 改善，Z 仍未通过

2026-10-06，Asia/Singapore；接续 `0ebd1e0` 和
[上一轮最终八格](native-final-generation-quality-2026-10-06.md)。
完整目标仍 active：双后端、Z/Qwen 四格 base ≥1.2×、真实 LoRA
加速以及媒体/内存/device-trace 资格均未完成。新选项默认关闭；单
prompt 的局部 N1 不足以提升默认，不能把 GPU-only 输出列为 ANE 通过。

## 共享实现与不变边界

新增 `dense_gpu_projection.hpp`，共用原 dense MPP range arithmetic/
physical pitch。FP16/BF16 operands 不宽化，不构造 compact 或 F32 W bank；
可选 output 为 F32，仅省略 partial 的最后 narrowing。旧 Z range helper
转为薄包装，默认仍原输出 dtype。组件对照确认新 F32 再舍回原 dtype
与旧 range 输出逐位一致，涵盖 physical slices/pitch/masked tails。

`Weights::project_base_slice_fp32` 提供 dense/packed ConvRot 的 checkpoint
贡献；明确不包含 down-LoRA/bias，不支持普通 affine/raw GGUF/NVFP4
这一 F32 consumer。原 GGUF/Q4→W8 与 Public converters 不变。
Qwen request/calibration-local factories 共用新增 partial 参数，Z 的
真实运行/校准 callbacks 亦统一。Qwen 的 F32 head 用 MPP，而旧 BF16
head 是 MLX matmul；因此完整结果是 kernel+partial 配方的组合效果，
不把质量改善全部归因于单一 narrowing。W 和 input dtype 未改变。

Private W8 executor 的 Sylvester/Comfy GPU restore 均可返回 F32 base
partial；这**不是 F32 ANE arithmetic**。LoRA hidden 独立保留 FP16/BF16
ABI，拒绝 F32 hidden；字节计数/guards/alias/failure 校验不删除。
GPU+ANE base partial 先 F32 求和再一次 BF16 base 舍入；之后仍由 ONE
down-LoRA 消费 joined full BF16 hidden，保留 delta 的原 rank dtype 和
最终 BF16 边界。错误/迟到 chunk 不得发布 partial scratch，完整 GPU
重算契约保持。

显式选项 `TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN=1` 扩展到 Private
W8 BF16 channels、base/LoRA；Public/row/default 不静默采用它。
额外 GPU head/restore bytes 在 inference 与完整 independent calibration
预算中计入。校准的 recipe/graph ABI 增加 `fp32-partial-join-v1`，cache
不复用旧 BF16-partial 身份；浮点 target views 的 pitch/dtype 正确区分，
frozen ANE surfaces/内部 MIL 仍 F16。host admission 是估计，不是 RAM cap。

`runtime_ane_model_screen.py --fp32-channel-join` 校验实际 Private channel
receipt/source recipe，GPU/frozen 不继承选项；native-auto 校准与实际 join
标记必须匹配。非法组合在写 output 前拒绝，完整 GPU decline 明确区分。
quality comparator 仍检查 CLI/相邻 library 前后 hashes、真实 backend/
calls、精确输入和实际步骤；不放宽 N1。

## 最终生成八格：同一新库、严格原 N1

fox prompt/seed42、resident；Z8步，Qwen base40步/LoRA6步，strength1
inference-time Z distill patch / Qwen Viggle rank256，原 FP32 rank。
每格独立 GPU/runtime 进程、串行运行；same v2 Private CLI/library。
runtime W8A8/IOSurface/channels auto/chunks1/fixed-async1/scale-cache1/
launch-fence1/stage-specialize1/F32 join1，prefetch/lookahead/deferred0。
仅 Z LoRA GPU ordinal2；Q1024 LoRA 的既有诊断开关同时用于两臂。

所有 conditioning/initial payload 精确匹配，execution artifact identity
bound=true；GPU reference payload 与上一轮亦相同。门槛保持 rel L2≤.03、
cosine≥.999；独立 CPU FP64、Z完整8步轨迹、Qwen final endpoint、PNG
不 resize。`qualification_passed=false` 始终不变。

| cell | Fa / actual Private calls | final rel L2 | cosine | 实际 ANE N1 |
| --- | ---: | ---: | ---: | --- |
| Z512 base | 4096 / 256 | 0.1205874392 | 0.9927320235 | fail |
| Z512 LoRA | 4096 / 248 | 0.1514961901 | 0.9885429957 | fail |
| Z1024 base | 4096 / 256 | 0.1067203021 | 0.9943138605 | fail |
| Z1024 LoRA | 2560 / 248 | 0.0926983083 | 0.9956967450 | fail |
| Q512 base | 5120 / 1280 | 0.0226912881 | 0.9997425403 | pass |
| Q512 LoRA | 5120 / 192 | 0.0168074970 | 0.9998587563 | pass |
| Q1024 base | 7168 / 1280 | 0.0156643687 | 0.9998777995 | pass |
| Q1024 LoRA | 0 / 0 | 0 | 1 (rounded) | GPU-only，不通过 ANE |

七格真实 offload 均 zero fallback/retry/failure、headroom1、实际 F32
标记 true；Q1024 LoRA 的 bandwidth-fit rejection/false 标记是 GPU-only。
Q512 base/LoRA 上一轮 rel L2 为0.1009017823/0.0322923420，现在通过；
Z512两格反而更差，不能从组件 rounding 通过推断完整生成变好。
Z1024 LoRA 和 Q1024 base 的自动 Fa 亦变化，不作固定 share 的因果声明。
完整 N1 不等于 LPIPS/CLIP/多prompt/人审/device资格。

目录：`outputs/native-fp32-final-{z,q}{512,1024}-{base,lora}-20261006/`。
Qwen 首轮错误模型目录的8个失败 `generation-receipt.json` 保留；未
执行模型、不计成功。纠正为现有 Comfy-Org Qwen 目录和实际
`TURBOCIDER_QWEN21_LORA_1024_DIAGNOSTIC` 后，Qwen成功回执为
`generation-v2-receipt.json`；Z成功回执仍前名。没有修改 native schema
或覆盖旧最终八格。

## 性能尝试：正式门槛仍拒绝污染

新 Q512 base 同库 GPU→runtime、一冷三热、strict continuous load、
process-tree memory，目录`outputs/native-fp32-q512-base-forward3-20261006/`。
GPU臂完成，但 external competing inference CPU 门槛拒绝，summary 为
incomplete、零 accepted trials；runtime臂未启动。raw results/PNG/load/
memory 全保留，不终止外部进程、不放松门槛、不用旧分母拼接倍率。

独立 complete-FFN trial 的 Q512 base 七样本中位 GPU/candidate约
0.0836655/0.058274958s，候选 trial zero retry/fallback；这不是 E2E
≥1.2×。带dump质量运行包含cold calibration，不用于正式速度比。

## 验证、构建与后续

101项 shared host/parser/screen/load/memory/image/quality tests通过；
完整 Public native构建、13 Core ML/MLX/receipt tests、actual release
isolation通过。Private native构建及3次 channel integration tests通过，
最后一次含新 F32 base/A/B/base、17/67-row、ONE down、BF16 hidden/base
边界及 late chunk whole-GPU recompute。另4个 actual-driver tests通过，
含两种 lookahead、4224-row、raw/packed Comfy、group256及software BF16
carrier rejection；独立 Sylvester component也通过。
memory-plan host fixture检查 F32 target/head增量，不改变 frozen surface
预算。不是全仓/全部Private/Swift/physical trace验收。

两处最初夹具错误（变量重名、遗漏full gate/up callback）只修复夹具；
host verifier 的一次变量名错误已修复并重跑，未削弱 production 校验。
首个native构建在补齐source-recipe telemetry前停止，仅信号已确认的
owned process group；v2重新完整构建，两个477-source manifests逐项
hash无mismatch。保留原native ConvRot dirty drafts，不称clean staged-only。

Private library SHA256
`2a9893031ae4c2109cadbfb8748fe9cd9276a63fdf1de6e37bb3304f9355c0b5`；
Public `eac7c30f03867ea1efde9f66eb12f8743290f91d872eac92963ba3bccc502b8c`；
Private thin CLI `f09144aed59e0899e0f3cd06ec97e54958ac91094f7b54453aeb84faf62ea02f`。
Private build ID `tc-runtime-build-v1-d8706a142ba3c1cca5a4cfc4919f8cb2ce33e85628a70807a811df506e750de2`。

后续保留Qwen候选，补reverse/多prompt/正式load-memory/物理trace；Z须
进一步隔离activation/weight/中间rounding及GPU敏感activation与ANE
projection的完整成本。GGUF有界consumer真实reuse/eviction、Public/
Private逐operation选择和base四格≥1.2×仍在原范围内，不能以本轮三格
N1或组件试验替代完整目标。
