# ConvRot 分区误差定位与 FP32 partial 候选

接续 `77fb5da` 的四步失败。本轮仍不改变默认路由或原模型文件，也不
缩小 Z/Qwen 四格 base≥1.2×、LoRA、双后端与 GGUF 调度的完整目标。

## 因果对照

原 GPU 在512²、四步 trajectory 捕获全部32个 block 的真实 FFN 输入；
重放 block0/7/23/31，共16个 sample。1024²另捕获 block31 的四步。
capture 是 FP16；probe 逐位确认每个值可精确还原 BF16，才继续重放。
比较包括 physical padding，是组件诊断，不借作整模型质量/速度资格。

512的16项中，纯 GPU channel split 的 gate、up、hidden 均与完整 GPU
逐位一致，但 down 的两个 BF16 partial 再合并，FFN rel L2 约0.25–0.30%。
这是一项独立的舍入误差，不是 ANE 失败或 A8 独有问题。

直接把 X/scales/bias 升到FP32不是等价修复：无 split 的完整 down
也出现约0.10–0.22%的差异。MLX affine QMM 的 `QuantizedBlockLoader<T>`
先在输入/metadata dtype 解码 W；升 dtype 改变了这一步。

新增 `affine_gpu_fp32.hpp`：显式 MLX affine Q4/Q8，沿用原 FP16/BF16
coefficient decode，在浮点 SIMD-group 中累加，保留 FP32 partial。
它不消费 raw GGUF、不改变主 weights，也不做完整 dense bank。
24项FP16/BF16、Q4/Q8、group32/64/128、nonzero physical row/column
starts与masked tails，对独立 dequantized-dtype F32 GEMM oracle通过。

真实512重放中，新完整 down 再舍入BF16对原 GPU **16/16逐位一致**。
新 split 的 full-output rel L2 为0.00000134–0.00019386；余差来自分区
累加顺序与最终 rounding，不冒称 split 逐位一致。

Private 同步新增 FP32 base output epilogue；不宣称 FP32 ANE arithmetic。
round back BF16 对旧 Private 结果的对照、真实 multi-chunk byte receipt、
每个有限FP16编码的F32 restore、strided/tail、overflow与失败读抑制通过。
FP32+adapter明确拒绝，旧LoRA hidden ABI不被静默扩展/丢弃。

## 默认关闭的 model consumer

`TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN=1` 要求支持该 output 的
Private ConvRot channel executor、base-only BF16 inputs。GPU base-down
由保持原 decode dtype 的新 kernel 返回FP32；ANE normalized输出由GPU
以FP32恢复；join后只做一次BF16舍入。新暂存字节纳入memory admission。
开关进入 executor identity，receipt明确报告 `fp32_channel_join_enabled`，
source recipe后缀为`+fp32-partial-join-v1`。Public/default/Sylvester不变。

## 实模型：改善但仍未通过

同一新库、原source、seed42、四步、Private4096/GPU6144 channels：

| 尺寸 | 旧BF16 partial rel L2 | 新FP32 partial rel L2 | 新cosine | N1 |
| --- | --- | --- | --- | --- |
| 512² | 0.04703974 | 0.04336734 | 0.99905977 | fail |
| 1024² | 0.08725383 | 0.08632567 | 0.99627542 | fail |

新轨迹均128 channel blocks、248/512实际driver calls、0 fallback，
fp32 receipt=true。conditioning/initial latent由独立比较确认逐位一致；
既定N1 rel L2≤0.03、cosine≥0.999未放宽。零validation调用的 native
passed字段仍不算质量证明。没有因一项组件修复就提高默认或宣称1.2×。

row/group256 A8 GPU重放提示：group尺度能降低误差，但晚段block31
仍明显敏感。GPU normalized-FP16 emulation不等于实际ANE：保留两者
输出与hidden差异，不把其相似性当hardware-MAC或数学逐位资格。
下一步是实际graph的group input/hidden scales，再独立检查敏感层的
GPU/Public策略；正式性能前先通过四步source gate。

证据目录：`outputs/convrot-ffn-source-capture512*`、
`outputs/convrot-ffn-source-capture1024-block31*`、
`outputs/convrot-ffn-error-{replay,fp32-replay,decoded-f32-replay}512.json`、
`outputs/convrot-ffn-error-replay1024-block31.json`、
`outputs/convrot-fp32-join-{512,1024}-{gpu4,private4096-4}-diagnostic/`。

Private库SHA256：
`2c0c8bfe585db6aa6cdac4f671d836674991361b41f27ad76082dbd000411e46`。
Public库：
`922649a81d4ae4a94f238d06d60e835efcc13e3ec5f64d2f4dcd6bb84ed7ed08`。
Private14 pass/2 opt-in skip、Public13、host8、GPU6；默认Public actual
release guard通过。构建保留已有ConvRot drafts，是working-tree snapshot，
不是clean staged-tree rebuild；原未提交研究改动仍不混入本轮commit。

GGUF仍待实际bounded dense consumer与复用成本A/B；本轮Q4 kernel组件
通过不代表GGUF整请求加速。自动bandwidth fit/cache/share/prefetch、
Qwen/Z四格matched optimized GPU、正式LoRA/媒体/内存/物理trace仍未完成。

紧凑机器记录见 `docs/design/validation/convrot-fp32-join-20261005.json`。
