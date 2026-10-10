# Public W8A8：ConvRot / GGUF Q4、Q8 的实际生成

2026-10-07，Asia/Singapore，M4 Max 64GB / macOS26.6.2。接续
[Public LoRA / full-K 控制](public-w8-real-lora-1024-fullk-2026-10-07.md)。
这是六格真实模型兼容性、视觉与诊断证据，不是完整性能资格。全部
Z/Qwen、base/真实 LoRA、512/1024、多 prompt/seed、正式 matched GPU
与双后端盈利调度仍未完成，整体目标保持 active。

## 两种来源，不能混称 raw GGUF 直接接入

ConvRot 的 GPU 基线是原 packed Q8 / BF16 metadata 路径，候选显式
`TURBOCIDER_Z_RUNTIME_CONVROT=1`，Public 图使用 Comfy H256。共享
stager 直接取已旋转 codes，不再 inverse/再次旋转，不混用 Sylvester。
GPU label 为 `mlx_cpp_metal_convrot_packed_q8`，候选保留
`mlx_cpp_metal_convrot+coreml_runtime_weight_experimental`。原 checkpoint
和 `int8_tensorwise_convrot_g256` 基线精度绑定，不改文件或 scale 来源。

本页 GGUF Q4/Q8 **仍经 `load_gguf_file` 转成 MLX affine packed planes**，
再由共享 GPU decoder → Sylvester rotation → normalized W8 staging。
不是 raw GGML blocks → ANE 的整模型接入，也不是 full dense predecode。
raw decoder 的硬件单测已存在，但不能用它替代这里的真实来源身份。
现有 CPU-direct packed bank 也输出 MLX affine，不因名字含 direct 就
称为 raw consumer。后续 raw 接入必须显式记录物理来源与有界 owner。

六格为 Z 8 步 / fox / seed42 / 无 LoRA，每臂独立进程、同一保留 Public
CLI/dylib、resident。512 使用352-row、1024 使用1056-row 模板；chunks1、
fixed-async1、GPU IOSurface IO、scale-cache/specialize/launch-fence1、
prefetch/lookahead0。模型、模板、request/receipt、逐步 dumps 和 PNG
保留，conditioning/initial latent exact、artifacts_unchanged=true。

## 真实调用与质量补充

每格均256次 Core ML/device-IO 调用、headroom1、overflow retry0、runtime
failure0、fallback0；Private ANE execution=false。Core ML API 成功不是
物理 NE 全驻留、native INT8 MAC 或 GPU/ANE overlap 的 trace 证明。

| cell | final relL2 / cosine | 原尺寸 RGB SSIM |
| --- | --- | ---: |
| ConvRot512 | .0708698476 / .9974882187 | .9814313 |
| ConvRot1024 | .0456726122 / .9989569103 | .9909037 |
| GGUF Q4 512 | .1525485953 / .9884180779 | .9393023 |
| GGUF Q8 512 | .1346975095 / .9909194047 | .9495464 |
| GGUF Q4 1024 | .0661297572 / .9978146929 | .9859574 |
| GGUF Q8 1024 | .0515516142 / .9986709078 | .9909381 |

六格历史 latent N1 均 fail，原失败和 qualification_passed=false 不改。
W8A8 最新目标以生成图像视觉非常接近为主，latent/SSIM 只是补充，
不据这些数值自动判视觉 pass。

本轮 agent 按原尺寸检查全部六格 whole 及三张同坐标 detail crops，
共24张 sheets。ConvRot 的主体、姿态、眼鼻、颜色很接近，胸毛/尾毛
有小差异；GGUF1024 整体很接近，毛发/胡须和雪粒略变；GGUF512 的
耳部轮廓、胸毛、背景虚化/枝条变化更明显，尤其 Q4，不是不可区分。
未见明显新增棋盘格、色块或断裂肢体。这是单 prompt/seed 的有限 agent
观察，不是盲测、用户批准或普遍质量保证；自动 visual manifests 仍 pending。

目录：`outputs/native-public-w8-convrot{512,1024}-20261007/` 与
`outputs/native-public-w8-gguf-{q4,q8}-{512,1024}-20261007/`。
按上表顺序，quality JSON SHA256：

```text
547fee3cac8e60196a9272ed1dab89c6bb19444d27029b0c4c9b2ea47f6ac02c
1a7c7239fd42a5d06b4d7ec2a75b8d64d48609cd8cbaa27919ddceb58ed107db
2fe0a4073fb8fdf821caf3da00aa48fc784c449b15cbf34dcae464be122de1b2
c9c184d74f821baa632a6d3a93aa30c45456608a5e716bbc1ad095342c5deb04
3f096093a9274469b5e4f9fade14fc7b155b63a12d264e8427862747c42b97db
2e1571b944e0838c42fc560f274efb20acdbc68aa24f52e499791e3d420ec252
```

## 诊断时间与下一步成本方向

| cell | GPU / Public denoise s | GPU / Public request-wall s |
| --- | --- | --- |
| ConvRot512 | 8.272513 / 6.941007 | 10.277896 / 9.597495 |
| ConvRot1024 | 35.885732 / 31.317696 | 37.996682 / 36.912221 |
| GGUF Q4 512 | 8.941317 / 8.261174 | 16.301827 / 12.033777 |
| GGUF Q8 512 | 8.947585 / 8.354984 | 19.163562 / 12.878107 |
| GGUF Q4 1024 | 39.472094 / 35.470263 | 48.064325 / 42.520784 |
| GGUF Q8 1024 | 39.674663 / 35.679650 | 51.024787 / 43.922937 |

这些带 dump/observer 的单次诊断没有 hot/reverse/strict load/memory
资格，也没有隔离 cold load 与 OS cache；不计算正式 speedup，不更新
默认 share。此前两次正式 Z512 测速仍因外部 ComfyUI busy sample 被
拒绝，不能拼接被拒绝的 arms、删样本、降低 gate 或终止外部任务。

GGUF 每格 scale-cache 为96 entries / 96 misses / 672 hits / 0 eviction，
1,556,480 bytes；这是 scale metadata 命中，不是 W8 codes cache 或
dense predecode 复用。ConvRot direct source 不使用该 scale-cache。
Q4/Q8 1024 累计 stage host span 分别7.243662/15.543232s，ConvRot 为
2.887680s；这些跨度不是独立 kernel 时间，不能与 denoise 直接相加或
推断重叠收益。raw→W8、next-layer prefetch 与有界热点复用值得继续
实测，但必须保留 read/convert/wait、真实 hit/eviction 和整个请求 peak。

## 回归与身份

质量比较工具新增独立 GGUF model identity / 完整 Z trajectory；ConvRot
仅在显式 route 下映射共享 validator 的 canonical label，结果保存原
backend、precision、checkpoint，拒绝普通 Z/GGUF/Qwen 冒充和来源变化。
86项 host tests通过：quality/model-screen/image/visual/component tools。
不是全仓、真实 raw 模型或全部硬件回归。

六格 observed receipts 的 adjacent library SHA256 是 opt-in Private-capable
v2 构建 `290c8a1f07181eadea1b5c92fc7379d5e2a5e23b8fb3818846691d218a51df61`，
build ID `tc-runtime-build-v1-010432fa7ae58d423bc2d163f6c896c3ad91ec704dac5cdc04b2dd6a93171b35`。
其实际选择的 executor 为 Public，不能把这个库身份与 ordinary Public
发行库 `f7cdd359…` 混淆。此处按原始 receipt 修正身份，生成结果不重写。
本页记录/工具改动没有替换 native library，没有夹带原未提交草稿。
