# Private W8A8 Executor 接续（2026-10-03）

接续 [W8 stager](private-ane-w8-stager-2026-10-03.md)。完整 SwiGLU 和统一
Executor 已进入真实模型初筛。目标仍 active：两模型 512²/1024² 四格
≥1.2×、层间 double-buffer prefetch、带宽校准、LoRA/画质与稳定性资格
没有完成，Public Core ML 仍是默认可分发后端。

## 实现

- `PrivateW8Graph` 实现共用 `Executor` 的 device weights/I/O 合同。
  `HybridFfn` 从已生产的 MLX source 提取真实 buffer、offset、physical
  pitch、dtype 和 allocation owner；Qwen 的 gate/up slices 不被压成错误
  的物理行。支持的 GPU decoder 见 stager 文档，不修改 base/LoRA 权重。
- 明确选择 `TURBOCIDER_PRIVATE_ANE_DATA_PATH=w8a8`；Public build 不编译
  私有类，Private factory 先通过数值 self-test 才交给模型。W8 路径只接
  SwiGLU，unsupported geometry 作为 capability failure，不静默冒充 W8。
- v2 recipe `sylvester-dh-b128-b512-rne-norm-f16-v2`：input signs 后 H。
  GPU gate/up/input 使用 H128，down/图内 hidden 使用 H512；normalized
  FP16 scales、RNE/clipping 与 CPU oracle 对齐。不是 Comfy H256 recipe。
- MIL 完成 gate/up normalized INT8 representation、scale、SiLU、LoRA
  correction、hidden H512/A8、down。单个 packed y 持有 normalized down、
  hidden row scale 和可选 corrected hidden；真实 shared-event done 覆盖
  整个输出。GPU FP32 epilogue 再恢复 BF16/FP16，保留 up-only headroom
  的溢出重试与后续层复用。不宣称已观测 native INT8 MAC。
- 恰好两套 W8 gate/up/down/scale bank，层之间 A/B 复用、不保留全模型
  W8 副本。当前只有 staging 与 attention 的重叠：worker 在下一层 stage
  前仍 join 当前层；**两套 bank 不等于当前 ANE 与下一层 staging 已重叠**。
- 错误、alias、非有限值、late-chunk 失败不能返回成功；HybridFfn 不消费
  部分 tail，使用原始 GPU callback 重算全部 tail。owner 保留到 completion。
- screen 增加 `--private-data-path w8a8`，强制 private/GPU-I/O 并校验实际
  data-path receipt；环境清理不允许外部变量污染纯 GPU 分母。

## 已执行验证与当前初筛

M4 Max 64GB、未改变发行 build/native：

- 12 项 private host/hardware、7 项 runtime host、47 项 screen host 通过。
  新回归涵盖完整 base/LoRA SwiGLU、两 bank 换权、padded GPU source/I/O、
  多 chunk、BF16 headroom、错误 staging 与 alias/nonfinite 拒绝/恢复。
  后续 SIMD/MIL 候选的 late-chunk/timing/recipe/K-tail 回归已重新通过；
  最新 public-only 库的 12 项 Core ML/MLX runtime 测试亦通过。
- 小图独立 dense source FFN oracle：base relative L2=0.020663；adapter
  relative L2=0.022322，hidden source L2=0.0106258。不是实模型质量验收。
- 初始 Private W8 库 SHA256：
  `7ffa7067272e728c38207c457213e44396c35ad3de298e79a5a2b5570b82a5a5`。
  Public-only 同阶段库：
  `39a46ac3b69d67bd0b653ceff24c778b438e7b315bd2239d82a804e93d173d4d`。
  完整 build 成功，public class strings/PrivateFrameworks link 隔离通过。
- Z BF16 512²/8 steps/fox/seed42，GPU→W8 auto，一冷两热。
  GPU hot median=6.991124 s；W8 session hot median=6.993521 s，0.999657×。
  W8 calls 累计 `[64,64,64]`：仅冷请求实际做 W8，热请求已被 auto 拒绝。
  零 error fallback 不等于热态 ANE 加速。原始证据：
  `outputs/private-ane-w8-z512-pilot-20261003/`。
- Qwen BF16 1024²/40 steps，同样 Q/K norm-RoPE、GPU→W8 auto/两热请求：
  GPU hot median=183.227665 s；W8=183.805384 s，0.996857×。
  W8 calls 累计 `[128,192,224]`，大多数 block 被 auto 拒绝，只有校准重试。
  全请求无 error fallback；仍未达到加速门槛。原始证据目录：
  `outputs/private-ane-w8-qwen1024-pilot-20261003/`。
- 像素检查只作为误差信号：初始 Z 冷请求 RGB MAE=5.288/255、
  PSNR=27.718 dB；Z 最后热请求已全 GPU，PNG identical 不是 W8 画质资格。
  Qwen 最后热请求 RGB MAE=0.403/255、PSNR=48.830 dB。未捕获 latent/
  FFN activation，也未做 holdout/media/多 adapter 完整画质验收。

## 正在推进的候选与尚缺内容

GPU Hadamard 的前五级改为 32-lane register shuffle，剩余级保留同序
threadgroup butterfly；scale reduction 使用 SIMD max。dispatch 前明确验证
GPU SIMD width。目标是减少权重转换的 barriers，不改变 recipe；候选已
重新通过 9 source formats 的 bit-exact oracle 和完整 Executor 回归。
MIL 同时改成先 slice INT8，紧接 dequantize → MatMul，不先解码全 FP16
权重再 slice；保留直接量化 operand pattern，不据此宣称硬件 INT8 MAC。
独立 `tools/native/private_ane_w8_stage_benchmark.mm` 可对两模型真实几何
测 gate/up/down staging，不以组件转换倍率替代整请求倍率。

standalone dense BF16 staging，一次 first-use 排除、8 热样本：

| 几何 | 原始 median | SIMD median | 转换倍率（非整请求） |
| --- | ---: | ---: | ---: |
| Z H3840/F10240 | 8.892 ms | 6.581 ms | 1.351× |
| Qwen H4096/F12288 | 11.311 ms | 8.361 ms | 1.353× |

SIMD/MIL 候选（报告层修复前）Private 库为
`d89b9ebdde0774a4595021cc545da4265ef06199d6fbdfb7cff46fb73a762ea3`，
对应 Public-only 为
`d18af6b5086d4bce7305af2acbbb021871c21fd72b097b8e269bba09b8847b73`。
构建、12 private、12 public、47 screen、public class/link 隔离检查均通过。
该候选 Z512 已完成运算，但 screen 因 JSON 详细回执缺失而拒绝完成：
`outputs/private-ane-w8-simd-z512-20261003/summary.json` 保持 incomplete。
旧 `results.mm` 按 `weight_variant == runtime_fp16` 判断 runtime，改用
`runtime_w8a8` 精度标签后丢了 payload/provenance。现改为 graph contract
（SwiGLU base/LoRA kind）判断，新增直接序列化回归；后续 chained cases
未启动，不把失败目录作为成功对照或将旧分母拼进新库。

报告修复后已重新完整 build：Private
`2666ca28597c22ad22ff1d95b74ffa75501e09442d23abc99afb8a22c3a3bbee`，
Public-only
`7815f00991661b3cff57ac8a093b783ad51cacdea211f8f6f1f89b248afb9e0a`。
直接 JSON 回归在 Private 库通过；13 项 Public Core ML/MLX/报告集成通过，
47 screen 和 7 host 通过，Public private-class/link 隔离仍通过。
新库串行 Z512/Z1024 base、Z patch/Qwen r256 LoRA、Qwen512 base 初筛已
启动，前缀 `outputs/private-ane-w8-receipt-*-20261003/`。未完成目录保留
incomplete；只有 completion/完整回执通过后才读取匹配性能，不借用旧分母。

新库 Z512 首格已通过完整回执校验：GPU median=6.989547 s，Private
session median=6.989986 s，0.999937×；每请求 W8 calls `[64,0,0]`。
热态仍被 auto 拒绝，没有转成真正的整模型收益。新库 Z1024 与后续
LoRA/Qwen512 共用同一串行 job；接续时先检查已启动句柄/原始 summary，
不要重新启动同一批请求或改写运行中的库/CLI/工具。

其后三格已完成（均同一 2666 库、GPU→Private、一冷两热、auto，error
fallback 为零；仍只是初筛）：

| 工作负载 | GPU hot median | W8 session hot median | 倍率 | 每请求 W8 calls |
| --- | ---: | ---: | ---: | --- |
| Z base 1024²/8 steps | 31.266972 s | 31.275825 s | 1.000× | 87/16/16 |
| Z distill patch LoRA 512²/8 steps | 8.559472 s | 8.593658 s | 0.996× | 68/8/8 |
| Qwen Viggle r256 LoRA 512²/6 steps | 8.151342 s | 8.947752 s | 0.911× | 98/33/0 |

真实 LoRA route 已成功执行 W8 activation corrections/hidden/down 回调，
没有合权，但仍无性能/广泛质量资格。Qwen 最后一热全 GPU，不能用它的
图像 parity 充作 W8 LoRA 画质证明。Qwen512 base/40步已完成：GPU median
41.718043 s，W8 session 42.449966 s，0.982758×，calls 128/64/32。
新的 Qwen1024、正反序 campaign 和层间 prefetch 尚未完成。
后续模型 channel candidate 与完整 hidden/down-LoRA join 已集成，见
[channel split 接续](private-ane-channel-split-2026-10-03.md)。

接着需要：候选实机回归、四格与两种现有 LoRA 的实跑、实际 activation/
latent/media 误差、channel split、下一层 prefetch/reuse fences、source leases/
scale cache、bandwidth calibration 和多热正反序 ≥1.2× 正式 campaign。
现有 row split 与银行轮换不是对原目标的替代。

```sh
.venv/bin/python tools/validation/runtime_ane_model_screen.py \
  --cli <private-enabled-build>/turbocider --model <checkpoint-root> \
  --model-id z-image-turbo --runtime-manifest <runtime-template>/manifest.json \
  --runtime-backend private --private-gpu-io --private-data-path w8a8 \
  --routes gpu,runtime --chunks auto --steps 8 --size 512 \
  --warm-repeats 2 --output <new-evidence-directory>
```

Public 默认、隔离 build、失败全 tail 重算和 non-merged LoRA 不变。
未 stage/commit，未修改参考仓库或设计稿。
