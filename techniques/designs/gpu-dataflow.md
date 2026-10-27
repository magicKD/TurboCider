# GPU 优化样本：从读写量推到 kernel

<!-- knowledge
id: technique-designs-gpu-dataflow
type: method
stage: prefill, decode, dit, ffn
operator: gemm, gemv, attention, norm, rope, layout
backend: metal, mlx
device: gpu
symptom: memory-bound, launch-bound, wrong-layout
semantics: numerically-equivalent
representation: fp16, fp32, q4, q8, packed
boundary: producer-consumer
api: public
-->

[设计目录](README.md) · [阶段地图](../models/stage-map.md) · [本地 MSL](../examples/metal/kernels.metal)

## 问题与推导

一个 projection：`X[M,K] @ W[N,K]^T -> Y[M,N]`。先用近似工作量 `2MKN FLOPs` 和必须触达的输入、权重、输出字节计算算术强度。这里的字节只是理想下界：重复加载、scales、临时反量化、cache miss、布局变换都会提高实际流量。用实测可持续带宽和同精度算力估计 `max(FLOPs/P, bytes/B)`，不把标称峰值当测量。

M 大的 prefill/DiT 能复用 W，倾向 tiled GEMM；M 小的 decode 常更受权重流量影响，倾向 packed GEMV/QMM。先看具体形状和 trace，再决定共享内存、SIMD matrix、向量化还是少提交。不是“所有 matmul 都换同一个大 tile”。

## 样本的实现路径

本地 [kernels.metal](../examples/metal/kernels.metal)同时保留 scalar/vector、scalar/SG8 GEMM、独立/融合 epilogue、QK norm/RoPE、dense/online attention。配套 bench.mm生成固定输入、CPU 数值 oracle、ABBA 配对与尾部 guard，不依赖 Splash 或产品后端。

从 scalar 改 tiled GEMM 的步骤：

1. 定义物理 `row_stride`、元素类型、转置、padding 和逻辑 M/N/K。子矩阵保留母矩阵 pitch，不能拿局部宽度作 stride。
2. 一个 threadgroup 负责 `BM×BN` 输出；K 按 BK 遍历。加载 A/B 到寄存器或 threadgroup memory，越界填零；所有需要参与 barrier 的线程必须到达 barrier。
3. 累加器的面积约为 `BM×BN`，共享内存约为 `BK×(BM+BN)×element_bytes`；tile 增大同时增加复用与资源占用，occupancy 不一定提高。
4. 指定 FP16/BF16 输入、乘法及累加精度，最后一次 cast 的位置也属于合同。换归约顺序只保证误差门禁，不声称逐 bit 一致。
5. 在最后一次 store 融合 bias/activation/residual，前提是没有绕过必要的中间 rounding、全局统计或其他消费者。

本地 SG8 仅是小型教学入口，不声称实现了所有 BM/BN/BK、双缓冲或 MPS/MLX 的生产级调度。它提供真实 device 对照，便于逐项扩展。

## Pack 与消费端是一件事

Q4 权重理想数据量约为 FP16 的四分之一，但 scales/zeros、对齐和解码也有代价。若先把整个 W 解码成 FP16 再 GEMM，可能丢掉省流量收益。更合理的候选是直接读原 codes/scales，在 kernel 内按 consumer 的 K/N 分块解码，配合 fused epilogue。具体 GGUF K-quants 不是普通 nibble affine，不能只换扩展名。

[quantization.py](../examples/reference/quantization.py)给出 Q4/Q8、integer-dot、LUT 和“局部 pitch 错误”对照；[packed design](packed-streaming.md)解释 backing 与 bounds。它们独立实现格式概念，不假装能读取第三方权重包。

## Attention 的数据流优化

Dense attention 的 score 临时数组有 `Q×K` 个元素。Online softmax 按 K tile 维护 `(max, exp_sum, weighted_value_sum)`，每次新 max 到来时同步缩放旧累加器；最后除以 exp_sum。它减少中间 score 的驻留，不省略任何允许的 key。[operators.py](../examples/reference/operators.py)包括 causal/empty-mask/tail，[Metal attention](../examples/metal/kernels.metal)给出实际小型实现。

把 dense 换 sparse/merge 是另一类近似，不是上述等价变换。它需要位置、mask、token mass、路由成本和完整媒体质量门禁，见 [近似](../algorithms/methods/approximations.md)。

## 验收与移植接缝

必测：小/大 M、非 tile 倍数、特殊 mask、非有限值拒绝、完整 physical strides、cold/hot、完整 consumer 边界。把好 kernel 插在 projection/attention adapter，不重写模型的 timestep、position 或权重合同。重新探索条件：新精度、tile、pack、芯片、SDK、mask 或下游消费者。

思想归纳自本库的 GPU 实验、Splash 的 shape/pack 专用消费和 Sana 的融合接缝；这里只需要本地源码。可选出处见 [吸收账本](README.md)。
