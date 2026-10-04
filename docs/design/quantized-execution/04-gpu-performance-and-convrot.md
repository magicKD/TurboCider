# 04 · GPU 性能候选与 ConvRot 加速

[目录](README.md) · [M5](05-m5-w8a8-backend.md) · [ANE](06-static-and-runtime-ane.md)

## 1. 先区分四个时间

`file parse/repack`、`read/decode each layer`、`GEMM`、`graph scheduling` 分开计时。
GGUF header 不会每个 denoise step 重新解析，不能把热态速度慢归因于“容器复杂”。
有界 streaming 会重复读/解码被驱逐的权重，这与一次性解析是不同成本。

理论比较以 `X[M,K] × W[N,K]^T` 为例：

```text
dense preparation work ~ O(KN)
GEMM work              ~ O(MKN)
quantized GEMM          = unpack/scales + actual matrix compute + scheduling
```

大 M 可以摊薄提前解码，但当目标浮点矩阵反复写回/读回、I/O 等待或带宽争用很大时
未必胜出。图片生成也有 M=1 调制层、小文本流、norm、attention、VAE 等不同算子，
不能只按模型名判断 compute-bound。

## 2. 当前慢因与解决方案矩阵

| 问题 | 可测证据 | 候选 A | 候选 B / 风险 |
| --- | --- | --- | --- |
| BF16 X + FP16 scales 提升到 FP32 | 逐投影 dtype trace | fused decoder 保持源 scale，以 FP32乘后一次落 BF16 | QMM 原生混合 dtype kernel；不先盲目降 scale 精度 |
| packed QMM 大 M 吞吐不佳 | matched-shape GPU time | 当前+前瞻层 dense BF16 GEMM | 优化 fused decode/MM 或 M5 真 INT8；二者并行比较 |
| 量化路线绕过 compiled block | graph/dispatch 数量 | 权重作参数的 fixed-shape compiled segments | 不把 decode IO/fence 塞入纯计算图导致失去生命周期控制 |
| 很多小 decode/cast/transpose | kernel trace、bytes | 融合 bit unpack、scale、target cast/layout | CPU SIMD 直填；避免多次全矩阵临时 |
| ConvRot 重复输入旋转 | gate/up graph | `project_many` 共享旋转 | 后续旋转+A8融合；LoRA fallbacks 不跳过 |
| dense H256 旋转 | 单独旋转时间 | 已有 radix-4 butterfly | 更改归约顺序需要质量资格，不能只按 O(KlogK) 宣称大幅提速 |
| 编码器短 prompt | actual token M | packed GEMV/GEMM | 只解码被访问 embedding rows，不套 DiT 大 M 策略 |
| I/O 或解码供给不足 | exposed ready wait | CPU预取/格式专用 SIMD | 2层前瞻只缓冲抖动，不能突破持续吞吐瓶颈 |
| 峰值突然抬高 | lifetime/backing trace | 固定槽和 lazy reader fence | 仅 `erase` 字典或清 allocator cache 不是释放证明 |
| GPU/ANE分配失衡 | full block control/split | 新GPU基线重新调 chunk | ANE局部更快但GPU分支未缩短，整层可能不变 |

同时比较至少三个 GPU backend：`packed_direct`、`bounded_dequant_dense`、
`native_w8a8`（仅经能力和质量验证的设备）。自动路线要按实际预算、M/K/N、
格式、dtype 和变换 recipe 做限定，未测量形状不能套最佳成绩。

## 3. 执行精度合同

Z DiT 的 BF16 执行是独立优化 profile，不是对所有 mixed 文件无条件的默认。
按 [09](09-release-scope-and-component-contracts.md)，首个正确性路径保留源浮点小
tensor dtype；另测 `z-dense-bf16-v1` 才显式转换它们。decoder 内 FP32 重建量化源值，
再一次舍入到 BF16；源 Q8_0 FP16 scales 与 ConvRot FP32 scales 都保留。

当前 compiled block 还检查 norm/bias 等字段，不能只解码大矩阵就宣称恢复 fusion。
Qwen3 Z encoder 的 FP32 residual/RoPE 则保持原约定，不继承 DiT 的 BF16 激活建议。

不可直接调用现有普通 `Weights::dequantize()` 然后认为完成优化：其普通 affine
分支当前输出 FP16，仍可能与 Z BF16 激活组合出 FP32。需要显式 `target_dtype`
并在所有 boundary 断言，而非输出末尾再 cast 掩盖中间 FP32。

FP16 分支单独比较，需 headroom/overflow 回归。尤其 SwiGLU hidden 可能超过
FP16 范围，不能将 BF16→FP16 当成纯性能实现细节。headroom 应在 gate/up/down
正确数学位置应用，并进入数值及缓存身份，不任意 clamp。

## 4. ConvRot 的存储数学

本 checkpoint 使用特定 Comfy H4 的 Kronecker H256，再沿 K 分组应用；不能换成
MLX 标准 Sylvester 排序。令 H 为所有组组成的正交矩阵：

```text
D = diag(weight_scale) * signed_int8_codes   // already in rotated basis
baseline Y = (X H) D^T
```

`pack_convrot_q8` 的 `q+128`、`bias=-128*scale` 是表示变换，不是 A8。
Z 的已有 BF16 scale 打包会舍入源 FP32 scale；新精度路线需说明是复现原生
checkpoint 还是复现旧 packed profile，不能混淆两个参考。

## 5. 低风险 ConvRot 优化顺序

### C1 · gate/up 共用旋转

```text
rotated = H256(x)
gate = project_rotated(rotated, Wg)
up   = project_rotated(rotated, Wu)
hidden = SiLU(gate) * up
output = project(hidden, Wd)     // its own rotation remains
```

复用 `Weights::project_many`，不拼接/常驻另一份完整权重，也不改变量化 scales。
保留每个投影 bias；若 runtime LoRA 不适合共享，按已有 API 回退，不丢修正。
本轮显式开关见 08。收益是减少一次旋转和表达式，不是 W8A8，也不保证同幅度
整请求提速。channel-sliced hybrid 路线需另做 `project_ranges_many`，不冒充已覆盖。

### C2 · butterfly 而非 dense H256

已有 `convrot_rotate_metal` 可复用。先验证 raw/packed、BF16/FP16/FP32、非连续
输入、1/33/1056/4128 rows，再决定 Z 是否接 opt-in。上一轮单算子测得约 14–15%
旋转时间改善，不是整图改善。共享旋转和 butterfly 的收益有重叠，不能直接相加。

### C3 · 有限槽解码，有两个分支

1. 保留旋转表示：`D→BF16`，在线 `XH` 后 dense GEMM。
2. 解码时逆旋转：`W_eff=D H^T→BF16`，在线普通 GEMM。

第二种在实数数学上满足 `X W_eff^T=(XH)D^T`；只是量化 checkpoint 的浮点
重建，不恢复原始未量化 W。FP32 变换后 BF16 舍入与旧路线不同，需独立质量资格。

**低内存下不能把逆旋转当成只付一次。** 一层被驱逐后，下步又要解码/逆旋转；
变换权重处理 N×K 元素，而激活旋转处理 M×K 元素，N 很大时方案2可能更慢。
优先以蝶形/有界 row scratch 做逆旋转，避免巨大 FP32权重 + BF16权重同时存活。
临时磁盘 dense cache 默认禁用：增加磁盘空间和每步读取流量，且不能假装仍有压缩
模型的 I/O 优势。若另行支持，必须显式授权与标注容量/生命周期。

## 6. ConvRot W8A8：可行但不是免费 A8

先用每 token 一个 activation scale、每输出通道一个 weight scale 的最小方案：

```text
R = X H
sx[m] = max(abs(R[m,:])) / 127        // zero-row has explicit safe scale
Qx[m,k] = clamp(round(R[m,k]/sx[m]), -127, 127)
I[m,n] = sum_k int32(Qx[m,k]) * int32(Qw[n,k])
Y[m,n] = float(I[m,n]) * sx[m] * sw[n] + bias[n]
```

保留源 `Qw∈[-128,127]`，无需再次 W8 量化。A8 的舍入模式、clipping、零行、
NaN/Inf 策略、scale dtype、INT32 overflow 上界、输出 dtype 均版本化。
不得把 signed -128 偷改成 -127 以迎合某个 API 后仍宣称权重未改变。

首个 per-row recipe 的 RNE、零行、安全累加和 epilogue oracle 已在
[11 第 2 节](11-acceptance-profiles-and-feasibility.md) 固定；更换 recipe 必须新资格。

Hadamard 只在每 256 通道内混合，并不保证跨 row/group 的 outlier 全部消失。
每 token A8、每 group A8、静态校准 A8 是不同误差/性能点：

- per-token：scale 可放完整点积后，整数 GEMM 简单，但 group outlier 可能影响整行。
- per-group：保真可能更好，但必须对各组 INT32 部分和分别 scale 后再浮点累加，
  不能全部加完只乘一个 scale。
- static per-tensor/region：适合部分 Core ML lowering；需要真实 prompts/steps/rows
  校准，caption/image 的分布差异单独处理。

先选单个 FFN，检查 gate、up、hidden、down 和最终 latent。敏感层/文本 rows
允许保留浮点，覆盖率和计算占比要报告，不把局部 W8A8 称全模型 W8A8。

## 7. 不互相替代的三种 oracle

1. source decoder oracle：源 codes/scales/rotation 是否解释正确。
2. execution oracle：相同 A8/scale 下，INT32 reference 与 GPU/ANE 结果是否匹配。
3. model oracle：额外 A8/变换顺序对完整生成是否可接受。

第一项错是 bug，不能通过放宽画质阈值；第二项通过不代表第三项通过。
性能比较必须保留当前量化 checkpoint 的对照，同时与原 BF16 的画质差异分别报告。
