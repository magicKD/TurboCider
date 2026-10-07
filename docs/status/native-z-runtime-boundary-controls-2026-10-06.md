# Z-Image：共享 GPU 算术边界和 FP16 负对照

2026-10-06，Asia/Singapore，M4 Max 64GB。接续
[F32 partial 八格](native-fp32-partial-qwen-quality-2026-10-06.md)。
这是一次代码收敛、因果诊断和负结果保留，不是整模型加速资格。
完整双后端、四格 base ≥1.2×、真实 LoRA 加速以及媒体/内存/物理
trace 目标仍未完成。Public 保持默认；Private 仍需明确 opt-in。

## 实现与隔离

`z_dense_gpu_pre_body`、`z_dense_gpu_projection`、`z_dense_gpu_ffn_body`
现在由完整 GPU graph 和 dense runtime pre/完整 FFN 回退 graph 共用。
保留 shape/device/env 选择、原 MPP/QKV/SwiGLU、BF16 boundaries、
norm/gate reduction、raw final modulation 与 checkpoint weight arguments。
runtime post 使用同一 `z_modulate_norm`。不再复制另一份 dense graph
配方，也不把旧 frozen-hybrid graph 当作精确的 GPU 分段定义。
frozen hybrid、GGUF/ConvRot/runtime-LoRA compatibility 分支未改变。

新增默认关闭的 `TURBOCIDER_Z_DENSE_SPLIT_GPU_CONTROL=1`。
仅 experimental build 的 resident dense、GPU-only、无 LoRA/ANE
manifest/streaming/detail override 请求接受它；普通发行 build 在载入
权重前拒绝，非法非 0/1 值也拒绝。它执行 runtime pre → 完整 GPU FFN
→ runtime post，并在 input/feed 处 eval，**不构造 ANE executor、不
stage、不分配 surface、不调用 ANE**。实际 backend/graph 明确为
`mlx_cpp_metal_dense_split_gpu_control` / `compiled_split_gpu_ffn_control`，
不能拿它作 GPU+ANE 或性能推荐。

质量工具新增显式 Public/Private FP16 policy 与 GPU-only control binding。
仍检查匹配 binary/相邻 dylib 前后 hashes、几何/seed/实际 steps 和
完整 Z trajectory；GPU control 必须有专用实际 label、纯 GPU encoder、
无 hybrid/LoRA。兼容 native schema 的 plan.execution 和省略的零 LoRA
count，不允许非零/bool count。`candidate_coreml_executed` 与 Private
actual calls 分开，Public prediction 不代表物理 ANE 驻留。
`--gpu-only-control` 必须搭配 execution receipts、`--channel-auto 0`
和 `--device-io 0`，不修改原 N1 或 qualification 门槛。

## 同一新库的十次完整生成

fox prompt、seed42、8步、resident、无 LoRA，GPU/control/W8A8/
Private FP16/Public FP16 × 512/1024，十个模型进程串行，没有与 owned
build/test 重叠。每个保留 request、dumps、PNG、observed generation
receipt；候选有 quality.json。所有比较使用同一 v2 CLI/相邻 Private
library，包括在该 library 中明确选择的 Public executor。

| 形状/候选 | actual calls | final relL2 | cosine | 原 N1 |
| --- | ---: | ---: | ---: | --- |
| 512 GPU-only split | 0 | 0 | ≈1 | GPU 对照通过，非 ANE |
| 1024 GPU-only split | 0 | 0 | ≈1 | GPU 对照通过，非 ANE |
| 512 Private W8A8 | 256 | 0.1205874392 | 0.9927320235 | fail |
| 1024 Private W8A8 | 256 | 0.1067203021 | 0.9943138605 | fail |
| 512 Private FP16 | 259 | 0.0934619628 | 0.9956320612 | fail |
| 512 Public FP16 | 259 | 0.0934619628 | 0.9956320612 | fail |
| 1024 Private FP16 | 259 | 0.0779761958 | 0.9969570683 | fail |
| 1024 Public FP16 | 259 | 0.0779761958 | 0.9969570683 | fail |

原门槛仍为 relL2≤0.03、cosine≥0.999。两个 GPU control 的 conditioning、
initial、全部8步、final latent 和 PNG 均逐位一致；新普通 GPU 输出
也与上一轮对应 GPU dump 一致。两种 FP16 backend 的对应完整输出
逐位相同，但不是相对 GPU 通过，也不是物理引擎相同的证明。

W8A8 使用 auto Fa4096/Fg6144、GPU IO、F32 partial join、chunks1/
fixed-async1/scale-cache1/launch-fence1/stage-specialize1，prefetch/
lookahead/deferred0；两形状都是零 retry/failure/fallback，headroom1。
FP16 使用原 full-width H3840/F10240、352-row、K1024/N512 activation-
input 模板，base 的 corrections 为零；chunks1/fixed-async1，Private
GPU IO、Public host IO。每次都是3次 overflow recovery、最终 headroom64，
零 runtime failure/fallback。每形状实际 offload rows 都是90112；这不是
W8 channels 的同 share 对照，也不是零重试性能配方。

目录：`outputs/native-z-boundary-v2-z{512,1024}-base-20261006/`。
GPU PNG SHA256：512
`f872c1ad9beb5cea4e4a6d5fd691e65b07799d40700ecb0f71658f7223f14e18`；
1024 `3846868cf4197ba1b66a2b8740e591b0f1400865d74a0f7b3fec60a107c6bd6e`。
所有 quality 的 `qualification_passed=false`，没有整请求倍率声明。

## 被实测否定的简单解释

最初怀疑 legacy runtime pre/post 与当前 GPU kernel 算术不匹配。
v1 改成当前 GPU kernel 后，Z512 W8A8 与 FP16 输出却与旧库逐位相同。
随后 v2 收敛为真正共享 body，GPU-only split 在两种形状精确通过，
而实际 FP16/W8A8 仍失败。因此不能把边界重构包装为精度提升；它提供
已验证的维护和诊断边界。v1 的3次生成及旧库 Public/Private FP16 负
对照均保留，未替换或删除。

进一步以 Private FP16 352-row 模板将 K 从1024改为10240，移除显式
K-tile partial additions，N512、exp SiLU、weights/input 和其它参数不变。
Z512 真实执行259次、3次 recovery，final relL2=0.0937386878、
cosine=0.9956115459，仍失败。模板位于
`outputs/runtime-ane/z-fp16-fullk-c352-k10240-n512-v1/`，完整模型证据
`outputs/native-z-fp16-fullk-z512-private-20261006/`。这不足以支持“加大
K 就修复误差”，更不是速度推荐；其 Public/1024 请求尚未执行。

当前证据支持优先定位真实 FFN 的中间值/舍入和迭代误差放大，但没有
独立证明唯一根因。F32 GPU partial 不是 F32 ANE arithmetic，Full-K
也不证明物理 accumulation dtype。不能据此声称原生 INT8 MAC 或
GPU/ANE 物理 overlap。

## GPU / GGUF / 混合后端优化优先级

既有 ConvRot SIMD-register rotation/A8 staging 与 F32 partial 仍保留；
其局部收益见[staging 组件报告](convrot-register-a8-staging-2026-10-06.md)，
不能换算为本轮 ≥1.2×。应先保住已验证 GPU 算术与最终质量，不能仅
为了减少 dispatch 换 norm/hidden rounding 配方。

GGUF 提前解码应继续围绕已有 affine dense consumer window：限定 bytes/
槽数、按 source ticket/generation/bits/group/dtype/backend/recipe 隔离
缓存，owner/ledger claim 跟随逃逸 view，完成验证后才 publish。现有
window 是同步研究 consumer，**不是已接入的自动跨 step 预取/复用**。
先做 decode+真实 projection 的同预算消融，再把仅下一消费窗口的
decode 接入生产者事件和 GPU/ANE staging；不默认保留整模型 FP16
解码副本。详见[消费窗口报告](convrot-register-dense-window-2026-10-06.md)。

Public/Private/GPU 杂糅应按 operation/layer 的质量、shape、source 和
预算选择一种 consumer；同一片 W 不同时保留 Public FP16 和 Private
W8 两份。现有 capability fallback 不等于已实现的 per-operation
三选一路由。真实 chunk 失败仍整 operation GPU 重算，不发布 partial
scratch。本轮没有扩展该路由或给它加速资格。

## 构建与测试

Private v2 完整 native-only build 成功，Public ordinary library-only build
成功；各477个 native source inputs 与 current tree 独立匹配。
这两个 build 包含原有未提交 ConvRot/cache 等用户草稿，不是 staged-only
tree 的制品；本轮提交只选自己改动的 hunks，保留那些草稿。

Private library SHA256
`87045b2fbcc76e99dc66b9bbda31b5220a28b2d7d6832f97e103461b01ea44f2`，
build ID `tc-runtime-build-v1-c3a39445219cfab536c9e36905525e0933565421b52451498627327d720aa3bb`。
Public library SHA256
`7c71d39c974e7a0d20b4f88047500aebe26b4154f5f7bc3b66c004d21ef2c110`，
build ID `tc-runtime-build-v1-727cac0330bf07986058c32e39ab0a41a5caedc6b55dfb858a6d8a0c39fefa8b`。
薄 CLI hash 仍是
`f09144aed59e0899e0f3cd06ec97e54958ac91094f7b54453aeb84faf62ea02f`，
不能单独标识相邻 dylib。回执是前后 bytes 绑定，不是 loaded-image 或
物理引擎 trace。

117项 shared host/tool/source guards 通过；14项 Public graph、Core ML/
MLX/receipt integration 与新增 ordinary-control gate 通过，未跳过。
Public library 未发现 `_ANEClient`/`_ANERequest`/`_ANESharedEvents`/
`_ANEIOSurfaceObject` strings。另2项实际 Private channel/MLX 测试通过，
覆盖独立校准、raw GGUF/LoRA callback、原 hidden/ONE down、F32 partial、
typed ownership、late failure 和整操作重算。这不是全仓/Swift/App 或
全部设备资格。

## 正式性能尝试：仍拒绝污染数据

旧 F32 v2 的 Q512 LoRA runtime→GPU、cold+3hot 尝试保留于
`outputs/native-fp32-q512-lora-reverse3-20261006/`；本轮新 boundary v2
又在独立新目录
`outputs/native-z-boundary-v2-q512-lora-reverse3-20261006/` 重试相同
原 FP32 rank256/strength1/6步、Private W8A8 auto/F32 partial 配方。
两次均保留原 process-tree memory、continuous load、JSONL/PNG。

新一轮 runtime arm 完成，但严格 load gate 检出外部 ComfyUI Python
CPU 负载并拒绝；GPU arm 没有开始，summary 为 incomplete、trials=[]。
没有合并旧 GPU 分母、剔除热样本、放宽门槛或 signal 外部进程；因此
没有有效反序倍率，更不声称 ≥1.2×。这些是单项性能证据未成立，不
妨碍继续质量/kernel/有界消费实现，完整目标仍 active。
