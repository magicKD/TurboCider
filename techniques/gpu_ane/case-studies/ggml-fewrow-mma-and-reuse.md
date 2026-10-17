# GGML Metal：few-row MMA、量化解码复用与 GPU 分母

<!-- knowledge
id: technique-gpu-ane-case-studies-ggml-fewrow-mma-and-reuse
type: case-study
stage: ffn, prefill, decode
operator: gemm, gemv, quantization
backend: metal, mlx
device: cpu, gpu
symptom: measurement, precision-loss, launch-bound
semantics: contract-only
boundary: producer-consumer
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [GEMM 基础](../gpu-gemm-quantization.md) · [residual 回归实测](ggml-residual-identity-regression.md) · [固定源码/PR](../sources.md#ggml-fewrow)

核查日期 2026-10-16。本页读本为 llama.cpp `988190680d5a89fce97de3c20df2c2813731fd61`，是选定的 10 月 7 日合并版本，不声称它是当日最新 HEAD。此前 `64e9bce…` 的旧读本保留；这里解释新增 few-row 路径如何改变 speculative verification / 小 batch 的 GPU 分母，以及它与 ANE 分担的关系。

## 不只是“一个 token 用 GEMV，其余直接大 GEMM”

设权重 W 有 M 个输出行、K 列，激活有 N 个输入行，输出 Y 为 `[N,M]`。GGML 的 `ne00=K`、`ne01=M`、`ne11=N`，不要把这里的 M 与其他文献的 token 维 M 混用。

传统 mat-vec 每新增一个激活行都要增加读取/解码/归约工作；普通大 mat-mat tile 又可能在 N=2–16 时浪费 padded 列。新路径把 8×8 SIMD-group matrix 当成 few-row 计算单位，让同一份解码权重同时服务若干激活行。它改变完整 kernel 的读数、decode、累加、归约和写出，而不只是换一条乘法指令。

在[原始 dispatch 2588–2603](https://github.com/ggml-org/llama.cpp/blob/988190680d5a89fce97de3c20df2c2813731fd61/ggml/src/ggml-metal/ggml-metal-ops.cpp#L2588)中，FWHT 特例先检查，few-row MMA 再检查，之后才是既有小 batch `mul_mv_ext`、普通 `mul_mm` 与 mat-vec fallback。因此只读旧小 batch 分支会错判实际路径。

## 真正的 shape / dtype / layout 门槛

[`ggml-metal-common.cpp` 52–178](https://github.com/ggml-org/llama.cpp/blob/988190680d5a89fce97de3c20df2c2813731fd61/ggml/src/ggml-metal/ggml-metal-common.cpp#L52)把 broad `may_use_mma` 与精确 `use_mma` 分开。前者只描述潜在格式组合，不能授予 dispatch 或融合资格。最终使用还要求：

- 已合法的 GGML tensor、支持的权重 type、F32 激活、N≤16，且达到该 type 的最小行数。
- K 满足所选 kernel 的 step；权重内层 stride 为一个原格式单元，两个输入都不转置。
- 激活内层 stride 为 4 字节，外层三个 byte stride 各按 16 对齐。
- batch / broadcast ratio 可放进 int16 function constants。全局 quant-block、batch 可整除等仍由其他 GGML 检查负责。

| 最小 N | 本 ref 的类型 |
| --- | --- |
| 6 | F32 |
| 5 | TQ2_0 |
| 4 | BF16 |
| 3 | F16、Q2_0、Q4_K、Q5_0、Q5_1、Q2_K、IQ4_NL、MXFP4 |
| 2 | 其余 15 个支持类型，包括 Q4_0、Q8_0、Q5_K |

总计 26 种，不代表本机测试了 26 种，也不绕过 BF16 编译/设备支持。源码说阈值来自作者的 M3 Ultra 测量；它是版本相关经验，不是 Apple 硬件定律。

K step 随路线变：Q4_0 用 32；Q8_0 在 N≤8 用 32、N≥9 改 generic 用 64；Q5_K 和 generic 用 64。因此合法 Q8_0 的 K=96 在 N=8 可进入此路，N=9 却不满足该路。不能只按 quant block 大小推 dispatch。

## Tile、K split 与实际 pipeline 限制

[`ggml-metal-device.cpp` 807–925](https://github.com/ggml-org/llama.cpp/blob/988190680d5a89fce97de3c20df2c2813731fd61/ggml/src/ggml-metal/ggml-metal-device.cpp#L807)选择 `(NSG,NT,RT)`：

```text
一个 threadgroup 覆盖 (8*NT) 个权重行 x (8*RT) 个激活行
  NSG 个 SIMD-groups 各计算一份 K slice
  → FP32 partial 写到 threadgroup memory
  → threadgroup barrier → 跨 NSG 求和
  → 边界检查 → 最终结果/可选 residual
```

N≤8 时 RT=1，否则 RT=2；NT 从 4 向 2/1 缩。默认 NSG 随权重行数分为 32/16/8，再受 K steps 限制。partial reduction 用 `NSG*NT*RT*256` 字节，启发式上限 16KiB；还试图保留约 128 个 threadgroups 的并行度。

这不保证“至少 128 组”：权重行不足时，NT 已缩到 1 仍可能更少。真实 pipeline 的 max threads 不够时又只减半 NSG，不重新放大 NT。寄存器、编译限制、占用和并行度相互制约，不能只把 NSG 开到最大。

这里的 K split 在同一 threadgroup 内归约，不是跨 dispatch 的 split-K scratch/reduction。迁移时先确定自己复制的是哪一级切分。

## 三类 decode 复用

完整选定 [`mul_mv_mma.metal`](https://github.com/ggml-org/llama.cpp/blob/988190680d5a89fce97de3c20df2c2813731fd61/ggml/src/ggml-metal/kernels/mul_mv_mma.metal)已读。

Q4_0 / 小 N Q8_0 路线预取下一块 words/scale/activation，再对当前块做 MMA。Q4_0 high nibble 允许保持为 `16*q`，把对应 activation 除以 16；decode 与 activation schedule 必须配对。Q8_0 翻 sign bit 后减 unsigned bias，不把原始 signed byte 当成 0–255。

它们用 `0x6400 | q` 构造正常 half 数 `1024+q` 再减去 1024/bias，避免依赖可能被 flush 的 subnormal。独立 CPU bit-level 检查覆盖 1024 个构造值、256 个 Q4 byte 和全部 signed Q8 byte；不是 GPU subnormal 行为实测。

Q5_K 专门路径保留 sub-block scale/min，将 high nibble 比例与 scale 进入累加，并用激活和扣 min。漏 min 在接近零均值输入上可能不显眼，上游另外定义 nonzero-mean tests；本轮未执行这些额外用例。

generic 每 lane 从 64-weight chunk 解码 16 个连续权重，复用到全部 RT tiles。Q8_0 在 9–16 行也进入 generic；源码理由是此时逐 block 缩放不如展开成 F32。不同格式/行数不需要强行共用 schedule。

尾部 load clamp 到最后合法行，store 再检查真实坐标，重复计算不写回。lane、stride/broadcast 和后续 sum 是整体合同；只摘 decode 函数不是完整算子。

## SIMD-group MMA 不等于 M5 TensorOps，更不是 ANE

设备以 Apple7 判断 SIMD-group matrix 支持；Tensor API 的设备/SDK、默认 pre-M5 disable 和编译探测是另一条路径，见 [`ggml-metal-device.m` 1100–1217](https://github.com/ggml-org/llama.cpp/blob/988190680d5a89fce97de3c20df2c2813731fd61/ggml/src/ggml-metal/ggml-metal-device.m#L1100)。本机实际日志为 M4 Max、SIMD-group matrix=true、has tensor=false，MMA pipeline 确实运行。

来源差异值得注意：#29869 可变 PR 描述说“无 Tensor API 的设备”，并提到 Q4_0 N=2 的 mat-vec 特例；但实际 merge commit 与本轮 ref 的 host gate 仅检查 `has_simdgroup_mm` + shape predicate，没有 `!has_tensor` 排除。实测 Q4_0 N=2 也出现 MMA pipeline。以固定源码/执行路径为准，不把 PR 描述逐句当合并实现；本轮没有测 M5 选路或性能。

## PR 演进与可比较的结论

[清单](../sources.md#ggml-fewrow)保存五份官方 API/body、head/base/merge SHA 和 patch SHA；仅 #30065/#30100 两份共 234 行 patch 完整读，其余长 patch 只归档，机制从固定源码读。

| PR | 2026-10-16 API 状态 | 学习位置 |
| --- | --- | --- |
| [#25377](https://github.com/ggml-org/llama.cpp/pull/25377) | closed、merged=false | 64×8 mat-mat tile 提案，不当已合并能力 |
| [#29110](https://github.com/ggml-org/llama.cpp/pull/29110) | open、merged=false | 多列 mat-vec register reuse 提案，不混入本轮执行 |
| [#29869](https://github.com/ggml-org/llama.cpp/pull/29869) | 2026-10-05 合并，`a3a1c474…` | few-row MMA、融合与调度起点 |
| [#30100](https://github.com/ggml-org/llama.cpp/pull/30100) | 2026-10-07 合并，`7e8324f5…` | residual 身份修复及原始回归模式 |
| [#30065](https://github.com/ggml-org/llama.cpp/pull/30065) | 2026-10-07 合并，`98819068…` | 格式与阈值扩展，即本轮 ref |

作者报告使用不同机器、版本、形状、模型和采样口径，本页不转抄最大倍率排名。本机只取得[正确性与原始缺陷负对照](ggml-residual-identity-regression.md)，没有 perf，也不证明 MMA、另一提案、ANE 谁普遍更快。

对 GPU+ANE 的可迁移判断：先把真实 N 分布下的 GPU decode 复用、融合和提交成本算入强分母，再测 ANE 子图、packing、join 与共享带宽。旧 mat-vec 慢不能直接成为 ANE share 的理由；本页也不证明 ANE 分担失去价值。speculative acceptance、实际验证行数和完整请求耗时仍须一起测。
