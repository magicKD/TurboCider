# Z-Image：同库五种本地表示，GPU / Public / Private

2026-10-11，M4 Max64 GiB/macOS26.6.2。原本地BF16、NVFP4、INT8 ConvRot、
GGUF Q4_0、Q8_0，均使用同一个原BF16 Qwen3-4B encoder、原BF16 VAE与
tokenizer；不加载LoRA。不同表示/算术配方不是逐位等价模型，不能把本表
当单个kernel的精度相同microbenchmark或普遍画质资格。

任务统一512²、8步、seed42，natural/soft/warm冬光狐狸提示词。每路线新
进程四请求：一cold、两fresh warm、一次立即重复warm prompt。16配置
正序后完整反序，32进程、128完整PNG；各配置四fresh样本池化取中位，
另外列两个repeat样本的中位。cold是首个生成请求，不清OS/driver cache。
计时为完整native request_wall，含encoder/VAE/PNG，不含parent验源。

## 最快已测配置与实际权衡

| 原DiT表示 | 最快已测GPU，fresh s | 最快已测混合，fresh s | 混合相对自己GPU省时 | 对应GPU tree peak GiB | 对应混合tree peak GiB |
| --- | --- | --- | ---: | --- | --- |
| BF16 | stock 7.551 | checkpoint-bound frozen 6.187 | 18.06% | 29.23–29.56 | 29.41–29.90 |
| NVFP4 | MLX weight-only W4A16 8.844 | N/A，当前GPU-only | — | 22.45–22.84 | — |
| INT8 ConvRot | stock packed 8.844 | Private Comfy W8A8通道 7.310 | 17.35% | 23.71–24.07 | 24.49–24.58 |
| GGUF Q4_0 | CPU-direct packed FP16 QMM 9.354 | Private Sylvester W8A8通道 9.008 | 3.70% | 8.58–8.81 | 8.53–9.13 |
| GGUF Q8_0 | CPU-direct packed FP16 QMM 9.504 | Private Sylvester W8A8通道 9.302 | 2.13% | 10.69–11.09 | 11.20–11.35 |

“最快”只在这16个实际完成配置内选择，不是穷尽所有kernel/share的全局
最优。GPU denominator同时含stock与显式fast候选，不能拿较慢GPU当
分母扩大混合倍率。4个fresh样本、单seed/狐狸家族不足以建立强置信区间。

本任务BF16最快；压缩表示不必更快。Q4/Q8的主要收益是直接compressed
bank的较低观测内存，而不是超过BF16的绝对时延。比BF16 frozen，Q4
混合fresh仍慢约45.6%；与Qwen Image2.1 GGUF对其BF16 streaming的
19%收益是不同模型/内存条件，不能互相替代。

## 全部16配置，不只保留赢家

时间单位s；cold列正序/反序。peak为整个load+cold+fresh+repeat+exit
期间进程树physical footprint，不是MLX allocator值或whole-machine cap。

| 配置 | cold 正/反 | 四fresh中位 | repeat中位 | tree peak正/反 GiB |
| --- | --- | ---: | ---: | --- |
| BF16 GPU | 8.341 /8.031 | 7.551090 | 6.995447 | 29.560 /29.235 |
| BF16 frozen | 11.620 /11.702 | 6.187363 | 5.346541 | 29.901 /29.412 |
| BF16 Private | 7.010 /8.244 | 6.536795 | 5.763445 | 29.568 /29.234 |
| NVFP4 GPU | 9.692 /9.715 | 8.843940 | 8.494185 | 22.841 /22.453 |
| ConvRot stock GPU | 10.284 /9.562 | 8.844211 | 8.488192 | 24.066 /23.705 |
| ConvRot compiled-dense GPU | 9.682 /9.635 | 8.865419 | 8.585120 | 16.359 /14.210 |
| ConvRot Private | 8.287 /8.386 | 7.310060 | 6.950776 | 24.487 /24.583 |
| ConvRot Public | 9.802 /10.406 | 7.509357 | 7.158510 | 24.605 /24.173 |
| Q4 stock GPU | 16.228 /16.621 | 9.470525 | 9.117806 | 24.127 /25.376 |
| Q4 fast GPU | 11.244 /10.875 | 9.353511 | 8.305088 | 8.579 /8.812 |
| Q4 Private | 10.721 /10.717 | 9.007673 | 7.945033 | 8.531 /9.134 |
| Q4 Public | 12.767 /12.740 | 9.760723 | 8.628825 | 9.035 /8.904 |
| Q8 stock GPU | 19.219 /19.254 | 9.571235 | 9.190422 | 27.019 /26.580 |
| Q8 fast GPU | 12.196 /11.860 | 9.504432 | 8.218018 | 10.686 /11.091 |
| Q8 Private | 12.006 /11.844 | 9.301974 | 7.986986 | 11.198 /11.351 |
| Q8 Public | 13.989 /13.785 | 10.034133 | 8.671330 | 11.332 /11.778 |

repeat是同prompt condition cache hit，不放入fresh样本池。GGUF retention
对repeat确实免去重新import；新prompt会按原政策释放/重建packed bank，
所以fresh多付实际source read/import费用。native verification bytes与
packed read bytes在机器记录逐请求另列，不以“零SHA”冒称零模型I/O。

ConvRot compiled-dense并未胜过stock的速度，本组名义差约0.24%，不
推广为更快；它的低峰值值得按内存场景单独衡量。Q4/Q8 fast路线仍是
packed QMM，dense_scope=none、dense capacity upper0，不创建全模型
dense sidecar。source表示、retention和allocator生命周期都参与结果，
不能把低peak或完整wall变化唯一归因一个GPU kernel。

## 配方与真实执行

- 同一`build/z-base-formats-v2-readonly-private`，Private1/probes1；514
  native输入独立与源码匹配。此库不是stable App自动默认或所有设备资格。
- BF16 frozen：已有checkpoint-bound image-only 5120-channel INT8图，
  GPU处理其complement与caption；原checkpoint SHA严格匹配。冷setup
  计入cold，不扣掉全源校验/Core ML装载后宣称cold胜出。
- Private：固定ANE4096/GPU6144 intermediate channels，IOSurface、
  fixed async1/chunks1、模板1152，actual rows按原匹配政策选择。BF16/
  ConvRot用F32 join；ConvRot GPU partial是已明确近似的BF16舍入后widen。
  ConvRot用Comfy H256直接codes配方，普通BF16/GGUF用Sylvester W8配方。
- Public：checkpoint-independent352-row W8图，单chunk/GPU complement。
  ConvRot用独立Comfy H256模板，不能误用Sylvester图。Public不是Private
  调用，成功不证明物理全NE放置/native INT8 MAC或hardware overlap trace。
- GGUF fast：CPU-direct/fused ARM NEON affine pack、retained packed bank、
  1GiB allocator hint；GPU compile packed与qmm_f16_down64额外FP16输入/
  FP32恢复仅用在GPU arm。runtime arm不混入此GPU-only compute recipe。
- ConvRot fast：compiled_dense、retained3GiB allocator hint；stock保留原
  compressed source/math。NVFP4仍是weight-only adapter，不启用A8或ANE。

所有混合实际完成，无runtime failure/error fallback。Private每请求实际
256成功channel blocks，四条Private每进程只在cold有2次headroom retry，
warm不再重试；真实calls严格按256×request+retry核对。cold恢复成本保留，
不宣称zero-retry整个窗口或以vacuous quality flag充当数值资格。

## 内存压力与图像差异

32份100ms进程树verifier全部complete。系统swap-out均0；forward
ConvRot-fast窗口有65,536bytes系统swap-in。六个BF16窗口出现约
0.279–1.481GB系统compression累计增量，其余列在机器记录。
system counters不归因单一进程；不承诺driver/外部服务字节或小容量实机。

正反序同路线/同prompt的64对原RGB全部exact。已查看四种最佳GPU/混合
warm-light whole512：狐狸姿态、眼耳鼻、尾巴、雪/背景构图很接近；
BF16 frozen的胸毛/眼鼻和背景局部变动更明显，ConvRot细纹变化，Q4/Q8
相对自身GPU主要是毛发/雪粒小差别，未见新增明显棋盘格或断裂。
有限agent观察不是blind/user approval或普遍视觉资格，不要求逐像素一致。

| 各格式 GPU vs自己的最快混合，warm-light | RGB correlation | RGB Gaussian11 SSIM | MAE /255 |
| --- | ---: | ---: | ---: |
| BF16 frozen | 0.984167 | 0.932267 | 4.9733 |
| ConvRot Private | 0.992737 | 0.959329 | 3.3510 |
| Q4 Private | 0.998282 | 0.985503 | 1.8584 |
| Q8 Private | 0.996163 | 0.975268 | 2.1782 |

## 来源、失败和复现

一窗口首次完整SHA后，parent只做轻量generation复验，不每inference
重扫。Z原stock GGUF的upstream可写mmap曾改变原文件ctime；修复为
held read-only source与owner-private压缩snapshot后本次原源保持。
原basis错误fallback、被source guard拒绝的8步/两步窗口都保留，未混入
本表；详见[读取边界修复](../status/z-gguf-readonly-source-2026-10-11.md)。

```sh
.venv/bin/python3 tools/validation/z_base_formats_screen.py \
  --cli build/z-base-formats-v2-readonly-private/turbocider --output outputs/new-z-forward
# 完成后串行另起fresh output，反序：同命令加 --reverse
```

`tools/validation/z_base_formats_evidence.py`独立审计两个complete窗口、
同SHA、完整反序、真实source/recipe/calls、raw memory链与完整PNG；
原始逐请求timings/cold/fresh/repeat/source bytes/precision/control/peak
全部在[机器记录](../design/validation/z-base-formats-20261011.json)。
模型、临时snapshots、raw PNG/日志/binaries均不进Git。未新增所有尺寸/
prompt/seed/低内存/Public发布资格；完整多模型目标继续推进。
