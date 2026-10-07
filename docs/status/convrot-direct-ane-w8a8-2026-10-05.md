# ConvRot direct-code ANE W8A8：实现与验收边界

2026-10-05。接续 `convrot-gguf-kernel-plan-2026-10-05.md`；本轮实现此前
Private W8A8 拒绝 Comfy transform 的缺口。默认 Public、默认 GPU 和原
checkpoint 不变。此路径是显式实验候选，不是新的生产加速资格。

## 已实现

Private 数据路径新增 `convrot_w8a8`，executor receipt 为 `w8a8_convrot`，
recipe 为 `comfy-h256-direct-q8-source-round-a8-rne-norm-f16-v1`。
它不能当作 Sylvester 的另一个 seed 使用；source/basis/block/seed
不匹配会拒绝 staging。原 `w8a8` 仍是 Sylvester H128/H512。

| 来源／步骤 | 直接路径 | 明确不做的转换 |
| --- | --- | --- |
| raw ConvRot | signed I8 原 codes + 每行 FP32 scale | 不裁掉 -128，不重量化 W |
| legacy packed ConvRot | q+128 → signed I8；使用实际存储的 scales | 不把 rounded BF16 scale 冒称原 FP32 scale |
| 输入 X | Comfy H4⁴，FP32 radix-4；先舍入到输入 dtype，再做 per-token RNE A8 | 不套 Sylvester random signs |
| FFN hidden | 独立 grouped Comfy H256 constant → A8 | 不复用 H512 constant；不覆盖未旋转 corrected hidden |
| down restore | FP32 epilogue，显式 finite signed/zero row-scale policy | 不放宽 token scale 的严格正数约束 |

原始数学是 `Y = (X H256) (diag(sw) C)^T`。ANE 输入表示为
`C/128`、`Qx/128`，staging scales 为 `128*sw`、`128*sx`；归一化
输出经 scale 恢复。normalized row scales 仍经过 FP16 RNE，因此
**原 codes 保持不等于完整算子逐位保持**。新增 A8、FP16 graph、hidden
rotation/quantization 的归约与舍入都需要整模型独立资格。
没有观察或宣称物理原生 INT8 MAC。

packed metadata 检查覆盖被选物理行的**全部** group，而非只检查
down channel slice：每组 scale 相同、offset 为 `-128*scale` 且有限。
非零 normalized scale 下溢为零、溢出或非有限是失败，不能通过给 W
套 quantizer floor 隐藏。A8 的零行／tiny 行沿用单独的安全 quantizer。

两套 W banks、可选两套 A8 slots、shared-event producer/consumer ABI
与 packed hidden 输出保持。future-bank identity 新增 basis，不能因
相同地址/geometry 误用另一基底。一个 W producer 抛出后仍 drain
先前已提交 producer；失败 bank 不 ready。任何晚期失败都返回失败，
model wrapper 必须重算整个 operation，不发布部分 scratch。

GPU `project_base_slice` 现在显式 dispatch 到 ConvRot `project_range`。
gate/up 的 `project_slice` 继续处理原低秩 correction；base-down channel
callback 不做 down-LoRA，完整 hidden 合并后只调用一次 down-LoRA。
不改变 adapter/master codes 或磁盘 model 格式。

## 已完成的组件检查

- 256 个独立 H4⁴ basis rows 验证 Comfy ordering；MIL constant blob 的
  每个系数也有独立 H4 matrix oracle，不只比较两个相同实现。
- 37 组 raw/packed Q8：signed -128、negative/zero/tiny stored scale、
  physical offset/pitch/slice、全部 row metadata、失败后 fresh refill。
- 9 组 A8：FP16/BF16/FP32、K512/3840/10240、strided source、source
  dtype rounding、RNE codes/scales 与 padding，generic/specialized 逐位一致。
- 实 GPU restore 验证 signed/zero policy；原 Positive policy 仍拒绝
  negative/zero W scale，所有 policy 仍拒绝 zero/negative/nonfinite token scale。
- 实 Private driver 的完整 FFN：raw/packed × serial/lookahead，共四格；
  三个 row chunks、非零 physical channel starts、A/B/A、zero/negative
  down scale、source 与 LoRA-hidden oracle、future-basis mismatch、晚期
  activation 失败、第二 W producer 失败及恢复。数值门槛未放宽。
- 24 组 GPU ConvRot base/channel slice：raw/packed、dense/Metal H256、
  三种 dtype、bias slice 与 base-only down，逐位对原 range oracle。
- 原 96 项 FFN、72 项 rotation、1056/4224 大尺寸 Quad exact parity 回归。

Private suite 16 项：14 pass，2 个 MLX/channel/calibration 显式集成开关
尚未在该次运行开启；Public factory 分发开/关都检查通过。另 8 个 host
memory/layout/conversion/scheduler 测试、5 个 GPU ConvRot integration
测试通过。不等于完整 `make test` 绿色，也不等于实际模型 LoRA 资格。

## 显式使用与混合 runtime

```sh
TURBOCIDER_ANE_BACKEND=private
TURBOCIDER_ALLOW_PRIVATE_ANE=1
TURBOCIDER_PRIVATE_ANE_DATA_PATH=convrot_w8a8
```

需要 Private-enabled 库、通过 actual-shape self-test、匹配 model geometry
的 runtime manifest，及 model 自身已有的 approximation/ConvRot opt-in。
channel share 仍显式选 512 对齐值；A8 lookahead/W prefetch 仍 default off。
此配方不能消费普通 BF16/affine/GGUF W；这些继续使用原 Sylvester W8
或 Public FP16 recipe，而不是默默 reinterpret 为 ConvRot。

Public 分发仍完全不链接 Private runtime。`auto` 的现有能力回退是
**构建 executor 时** Private → Public，不是每个 chunk 三路竞跑，也
不是 Private 运行失败后继续发布 Public/Private 的混合部分结果。
实际失败保持整 operation GPU fallback。默认 Public 的 ConvRot 路线
仍使用独立 inverse-H256 FP16 consumer。

后续可以按 operation/layer 的 source/shape/precision 选择 Public 或
Private，并各自搭配 GPU complement；避免对同一片 W 同时常驻两套
ANE 格式。选择与 calibration cache 必须隔离 backend/recipe/ABI。
Public CPU+NE policy 与 Private 异步提交计数均不是物理重叠证明。

## GPU／GGUF／提前解码的下一步

GPU Quad 已有同源、逐位保持的组件结果，但 butterfly vs dense-H
整轨迹仍有后段 block N1 失败，不能扩大默认。继续将旋转、QMM、
格式转换、restore/join 分开计时，再决定是否值得做融合。

GGUF 的 header/index 不是重复 layer decode。维持原文件与 packed
master source，按 M/K/N/encoding 比较 packed QMM、bounded predecode +
dense GEMM、fused decode/GEMM。decode 只有在实际复用时才可摊销；
不能把每 step eviction 后重解码当成一次成本。

Z FFN H3840/F10240 的三张 dense BF16 矩阵为 225MiB，两个 FFN-only
slot 是 450MiB，尚需 attention、metadata、allocator slack 与 ANE arena。
因此优先当前＋一层前瞻，不制造完整 dense sidecar。direct ConvRot W
避免原 inverse/compound rotate/requantize 的两遍 weight 计算，但仍有
code-bank copy、metadata validation、A8、restore 与统一内存带宽成本；
是否更快必须测完整窗口，不能从少一个转换步骤推断加速。

尚未完成：真实 GGUF bounded dense consumer A/B、拟合/缓存/自动 share
与 prefetch 接入、四格 matched optimized GPU ≥1.2×、multi-prompt/reverse
顺序、正式 latent/media/memory/device-trace 和实模型 LoRA 验收。本轮
组件结果不填补这些未完成项。原有负结果仍保留。

## 实模型结果：四步质量门槛未通过

同库、原 ConvRot checkpoint、legacy packed BF16 scale、相同 prompt/
seed42、GPU vs Private4096/GPU6144 channels，均跑四步并 dump 独立
trajectory。两种尺寸的 conditioning 与 initial latent 均逐位相同。
下列是**诊断数值对照**，不是 matched optimized GPU 性能资格。

| 尺寸 | Private channel blocks／calls | final latent rel L2 | cosine | N1 |
| --- | --- | --- | --- | --- |
| 512² | 128／248 | 0.0470397412 | 0.998893785 | fail |
| 1024² | 128／512 | 0.0872538294 | 0.996194781 | fail |

两个 case 均无 runtime failure／GPU fallback；是真实新数据路径，不是
能力失败后纯 GPU 完成。外层 backend/precision/selection 也修正为依据
actual executor receipt，不再把 Private ConvRot 写成旧 Public FP16 标签。

512 每步 latent rel L2 为 0.00194/0.00742/0.02192/0.04704；1024 为
0.00464/0.02346/0.05683/0.08725。它们说明误差随 trajectory 扩大，
**不能仅由 sparse self-test 或一步通过推断完整质量通过**。此前512²/
一步 preliminary 为 rel L2=0.0126453、cosine=0.999920、N1 pass，保留
但不替代四步失败。原 N1 门槛 rel L2≤0.03、cosine≥0.999 未放宽。

四步 PNG 的 CPU RGB SSIM 为0.990195/0.983672、PSNR 为38.7822/
35.6405dB；这些不是语义/感知资格，不能用它们抵消 latent gate 失败。
native quality_validation_enabled=false/calls=0 时的 vacuous passed
字段也不当作验收；这里使用独立 CPU FP64 tensor comparer。

下一步先拆分 full-GPU channel-join/source dtype boundaries、输入 A8、
hidden A8 的误差，再试 per-H256-group activation scales 与敏感层保持
GPU/Public FP16。它们是待验证方向，当前数据不能证明哪一项是主因，
也不能宣称更小 ANE share 或 Public 混合已改善质量/速度。四步失败前
不进入正式性能提升或扩大默认。

本轮 final Private 库：
`5015ce346df3e872fd2cb54fda75098cb0246b20361c3d7f4cc6e37c63a054a8`。
final Public 库：
`2ac32548989f3f2b3346716d794809a7649c36523a130da8ca8679ad7ef23246`。
Private=1/experimental-probes=1；Public 两项=0，actual release guard
通过。构建测试的是保留已有 ConvRot drafts 的 working-tree snapshot，
不是 clean staged-tree rebuild；原未提交代码没有混入本轮 selective commit。

证据目录：`outputs/convrot-direct-final-{512,1024}-gpu4-diagnostic/`、
`outputs/convrot-direct-final-{512,1024}-private4096-4-diagnostic/`；保留
原 PNG、每步 tensor、raw report、比较结果和 cold diagnostic wall。
没有计算或借用它们作为1.2×分母。最初使用缺少 transformer/ 的
Tongyi root 被创建校验拒绝，随后使用原 Comfy root；无 model 修改。

汇总见 `docs/design/validation/convrot-direct-ane-w8a8-20261005.json`。
