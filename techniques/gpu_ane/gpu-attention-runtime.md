# GPU attention、融合与提交：减少中间数据和等待

[专题导航](README.md) · [硬件/API](hardware-and-tensorops.md) · [实验](experiments.md)

读本：[独立 MFA `8671cdd…`](https://github.com/philipturner/metal-flash-attention/tree/8671cddc38f19a6eadb804dee6a3ca2954b8bf32) README 1–130；[MLX `scaled_dot_product_attention.cpp`](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/scaled_dot_product_attention.cpp) 396–502；[WWDC26 TensorOps](https://developer.apple.com/videos/play/wwdc2026/330/)。下述 online-softmax 方程是算法说明，不是复制某个 shader。

## 不物化整张 score 矩阵

Dense attention 的 `S = QKᵀ` 有 O(Lq·Lk) 中间元素。分块 attention 对当前 query tile 维护 running max `m`、normalizer `l` 和未归一化输出 `O`。对新 score tile S：

```text
m_new = max(m, rowmax(S))
a     = exp(m - m_new)
P     = exp(S - m_new)
l_new = a*l + rowsum(P)
O_new = a*O + P@V_tile
结果  = O/l
```

在线 rescale 保持稳定 softmax，避免把完整 score/probability 存到 device memory。实际 FP16/FP32 累加、exp 近似、mask 与 all-masked row 的定义仍需验证，不承诺 bitwise 相同。

## 实现差异在哪里

MFA 的独立读本刻意聚焦 single-head 教学核心，讨论 tile aspect ratio、D 方向再分块、register pressure 与必要 spill；ccv/MFA 是应用整合版本，不能混用一个版本的 API/性能。单个高效核也不是完整模型 runtime。

MLX 读本有 vector/two-pass、Steel 和 NAX 等路径。特别值得学的是 D=72/80 的可选 zero-padding 到 96：保留原 scale，copy 回 caller 的真实 strides；仅在特定长、无 mask、half precision 条件默认启用，LSE 等条件又会关闭。padding 增加算术和 copy，但可解锁更快矩阵核；选择依据是完整耗时，不是最少 FLOPs。

WWDC26 将 cooperative tensor attention 用于减少 score/softmax→第二次 matmul 的共享内存交接。不能机械认为“都放寄存器”一定更快：长 head、大 tile、accumulator 会压低 occupancy。必须连 operand layout、合法 scope 和 barrier 一起检查。

## 精确、近似与 serving 优化分开

| 技术 | 改变什么 | 正确性/收益边界 |
| --- | --- | --- |
| Flash-style exact attention | tile 与归约/访存顺序 | 浮点误差；mask、GQA、tail、empty row |
| 稀疏/缓存 attention | 少算部分关系或复用旧值 | 近似模型；需要图像/视频质量，不只局部 L2 |
| Paged/varlen attention | KV 存储与不同请求长度 | page table、offset、request identity、吞吐/尾延迟 |
| KV quantization | cache 表示、解码流量 | 长上下文累计误差与 metadata 成本 |

SolAttn-MPS 属于稀疏/近似候选；[vLLM-Metal](inference-engines.md)的 request-aware paged/varlen 是 serving 层。不能把相同“attention”标题下的收益互相继承。

## 融合：不只是大矩阵

RMSNorm/modulation、RoPE、residual、activation 与 projection epilogue 可减少中间张量和小 dispatch。MLX `mx.compile` 是探索入口：把纯表达式交给编译图，允许融合；lazy 构图耗时不等于设备计算，计时必须 `mx.eval`/完成同步。

融合会改变原 buffer 的存在和生命周期。ANE stager 如果依赖被消除的 norm 输出，必须重建边界；compiled function 若捕获可变 shared output，而未把它作为运行时输入，可能重用旧结果。不能为了跨设备接缝把所有 fusion 都打散，也不能为了 fusion 忽略 consumer。

## 提交层另做消融

Metal 4 的 command allocator、argument table、residency 与 command 提交复用主要优化 host/resource 管理，TensorOps 优化矩阵运算层。先固定 kernel 比提交，再固定提交比 kernel，保留 cold pipeline、encode、submit、device、wait、完整 block 的分项。

多个 command queue 不保证并发；不同逻辑 tensors 若共用一个 tracked resource，hazard 也可能串行化。反过来 untracked resource 需要应用自己正确排序。`mx.async_eval` 要确实启动工作；紧随其后的 host 读取、`.item()` 或 `np.array` 可能提前 drain。

## 复现与反例

本轮实验包含 Q1/K4096/D64、Q512/K512/D64、Q192/K257/D80 三组 no-mask attention，使用独立 CPU FP64 reference；不覆盖 causal、GQA、all-masked row 或长视频。详见 [实验页](experiments.md)。既有 [真实 Qwen attention 负结果](../review/qwen-attention-20261016.md)保留更强模型消费者，不能被 toy 正结果覆盖。

建议增加 causal/noncausal、ragged K、GQA、任意 stride、mask 全空/极值、非有限输入和 head-D tail。每个优化都保留成熟 dense fallback；若变成近似，另设质量预算，不把少 token/少 steps 当等语义提速。
