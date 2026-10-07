# Native W8A8：最终生成质量八格对照

2026-10-06，Asia/Singapore；接续 `93c6b16` 和
[compiled LoRA / 敏感层记录](native-compiled-lora-sensitive-layers-2026-10-06.md)。
双后端、四格 base ≥1.2×、真实 LoRA 加速和质量/内存/device-trace
目标仍未完成。本记录以最终生成对照补充之前的 FFN trial 和冷/热稳定性，
不将组件通过或同一候选冷/热一致视为 GPU-reference 质量通过。

## 输入和真实执行

八格 GPU/runtime 各用一个独立进程，串行运行。同一 v2 Private CLI
与相邻 library：CLI SHA256
`02c55923be1dff7fc8dfd5681fd26b1cfaf15d8553eec28d1d9ce09b16d11ebb`；
library SHA256
`f6197b29aba7b8922bed25ad51d5470a3c87c50667c7ce3394494b52c8b2035a`。
这些是保留原 ConvRot dirty drafts 的 working-tree builds，不是 clean
staged-only 构建。薄 CLI 的 hash 不能独自标识所加载的 runtime library。

fox prompt、seed42、resident；Z8步，Qwen base40步/LoRA6步。
Z distill patch / Qwen Viggle rank256 均 strength1、inference-time LoRA。
Qwen1024 LoRA 的既有显式诊断开关同时施加于两臂，保持 FP32 rank。
Private W8A8、GPU IOSurface IO、channels auto、chunks1、fixed-async1；
scale-cache/launch-fence/stage-specialize1，prefetch/lookahead/deferred0。
仅 Z LoRA 指定 GPU ordinal2（每请求8个完整 GPU blocks）；base 没有 override。

所有格 conditioning 和 initial latent payload 精确匹配。
`dump_tensors` 是 native request 正确字段；最初错误使用 `dump` 的 Z512
GPU 请求在 schema 校验处失败，旧回执保留，不纳入下表。成功运行均记录为
`generation-v2-receipt.json`，没有为迁就错误字段修改 native schema。

## 不变 N1 门槛及负结果

最终 latent：relative L2 ≤0.03、cosine ≥0.999。独立 CPU NumPy FP64
比较 safetensors，支持 F32/F16/BF16，限制 header/geometry/payload，拒绝
nonfinite、零能量、输入不匹配和 Z 缺失/不连续 trajectory；无 MLX/model load。
PNG 比较不 resize，RGB SSIM 不是感知/语义或设备资格。

| cell | ANE channels / actual Private calls | final rel L2 | cosine | N1 / actual ANE |
| --- | ---: | ---: | ---: | --- |
| Z512 base | 4096 / 256 | 0.0719742118 | 0.9974136529 | fail / yes |
| Z512 LoRA | 4096 / 248 | 0.1182747126 | 0.9930558803 | fail / yes |
| Z1024 base | 4096 / 256 | 0.1080132765 | 0.9941657786 | fail / yes |
| Z1024 LoRA | 3072 / 248 | 0.0852306902 | 0.9963615607 | fail / yes |
| Q512 base | 5120 / 1280 | 0.1009017823 | 0.9949063164 | fail / yes |
| Q512 LoRA | 5120 / 192 | 0.0322923420 | 0.9994787744 | fail L2 / yes |
| Q1024 base | 5120 / 1280 | 0.0169679899 | 0.9998565724 | pass / yes |
| Q1024 LoRA | 0 / 0 | 0 | 1 (rounded) | exact GPU-only / no |

Q1024 LoRA 的 auto calibration status 为 `rejected`：
`independent complete-traffic evidence rejected by bandwidth fit`。
不能把全 GPU 回退的零误差归为 ANE 成功。新工具将成功 native 回执的
backend/calls 和 workload 字段绑定，明确输出 `candidate_ane_executed`
与 `n1_with_actual_ane`；全八格只有 Q1024 base 的后者为 true。
它仍只是单 prompt 的局部 N1，`qualification_passed` 始终 false。
旧回执只绑定 CLI hash，尚不能自动证明每臂实际 loaded dylib 一致。

RGB SSIM 按表顺序：0.9808125355、0.9694294173、0.9735351854、
0.9839931322、0.9590923091、0.9891658865、0.9950490644、1。
Z512 step1…8 relative L2：base
0.001614/0.003996/0.006629/0.009798/0.016096/0.027255/0.044446/0.071974；
LoRA 0.002638/0.008050/0.012486/0.019816/0.031688/0.050800/0.079786/0.118275。
误差随生成积累；单 FFN trial 通过不足以提升默认。也不能未经实际对照
就假设仅在末步用 GPU 能修复之前累积的误差。

## 正式性能：拒绝污染窗口

`outputs/native-final-base-z512-forward3-20261006/` 使用同 v2 library，
GPU→runtime、一冷三热、严格 continuous competing-load 与 process-tree
memory sampler。GPU 的60个 load samples 中1个外部 ComfyUI CPU sample
触发污染门槛；保留全部 GPU 原始结果/PNG/load/memory，summary 为
`incomplete`、零 accepted trials，在 runtime 臂之前结束。
没有信号/终止外部 ComfyUI，没有放松门槛。不能拼接旧 GPU 分母，也不能
用本轮带 dumps 的质量运行时间计算 ≥1.2×。

## 工具和证据

`runtime_ane_model_screen.py --z-runtime-gpu-blocks` 规范化 ordinal 列表，
仅 Z BF16 runtime 生效；GPU/frozen 不继承该策略。根据 actual steps 和
cumulative forced complete-GPU counts 验证执行，而非仅采信环境开关。
非法 model/route/重复 ordinal 在输出写入前拒绝。

79项 quality/overflow/calibration/model-screen host tests 通过。
最初 execution-binding fixture 漏填 `failure_reason=""`；只修复 fixture，
production telemetry/fallback verifier 保持严格。不是全仓测试或新 native
构建验收。models、adapters、references、原设计文稿与已有 native drafts 未改。

原始目录：`outputs/native-final-quality-{z,q}{512,1024}-{base,lora}-20261006/`，
每格保留两臂 request/receipt/output/dumps 和 `quality.json`/`quality-bound.json`。
后者 SHA256 按上表顺序：

```text
407235f90929f6da2bf898b167665528d03e3a679e4f9ae48224fdf7e8462443
b5282c0fb924b059a04e1286a78ea561b7aefa4854a2f825830319a1f2de2ad6
2c169c675b2c75adac07dc24cf0a5cfe62687cafd0dd2117afb928dc5732a494
b2f65f20f04d4df044a1fb575372ee767b5c0c62b1a7bae6a179d93fcb418d25
caa99d6f7d212dafdc6f5ede61bc4ee1d69b4d1cea3b1f7db549584b56c8c594
3f0ea122f28a4e5edbb0d655e0e0fc84fd5b64f4fd8db82b562d69235f9e10b6
e4933005393b79927d1781238633a3ab2a00044d5abb60050a202bb9cbba2b3b
b4d77b5f2f21d6453b16c45953cc9e556bf5b41aecdf586c4bd64d564b745b7c
```

后续优先定位真实 W/A/中间舍入及完整 channel-join 误差，比较 GPU 敏感
激活和 Public/Private 高精度对照；GPU exact kernel 与 GGUF 消费布局优化
独立推进。不能靠放宽 N1、GPU-only 退路或减小目标来宣布完成。
