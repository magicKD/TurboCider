# GEMM/GEMV 与低比特 GPU 数据流：相同位宽不代表相同方案

<!-- knowledge
id: technique-metal-gemm-and-quantization
type: method
stage: ffn, prefill, decode
operator: gemm, gemv, quantization
backend: metal, mlx
device: cpu, gpu
symptom: measurement, precision-loss, launch-bound
semantics: contract-only
boundary: producer-consumer
api: public, reference-only
-->

[GPU 导航](README.md) · [kernel 调优](kernel-tuning.md) · [TensorOps](tensorops-and-m5.md) · [实验](../MAINTENANCE.md)

核查：2026-10-16。本文综合 MLX/GGML/MFA/MNN/Candle 的已读实现；源码与范围见[来源清单](sources.md#research-sources)。不复制第三方 kernel，不按文件大小推算实际提速。

## 方法卡：小 M 时复用权重，大 M 时复用 tile

- **机制**：让一次加载的权重服务多个输入向量，或在 tile 内解码后复用；减少流量/无效计算，而非仅降低存储位宽。
- **何时考虑**：实际 M 很小、权重流量或 packed decode 成本突出；先记录 M/N/K、stride、dtype 和驻留条件。
- **代价/反例**：更多寄存器、scratch、reduction 或 dense materialization 可能抵消收益；量化更小不保证更快。
- **实现参考**：[small-M wide GEMV](case-studies/mlx-small-m-wide-gemv-and-head-attribution.md)包含固定 selector/guard；这是实现机制，不是已取得 wide 开关性能收益。
- **最小验证**：同输入/输出需求比较成熟 baseline 与一个可控候选，覆盖 M 阈值、尾部和布局，计入完整消费者成本。

先用这张卡形成假设，再读下文成本模型和具体证据；不能把固定 MLX 配置域当成所有后端的硬件定律。

## 怎样借鉴：先写成本假设，再选实现

2026-10-17 方法整理。建议按以下顺序做设计判断，后面的源码与实验分别提供机制和反例：

1. **先区分小 M 与大 M。** 记录实际 matmul 的 M/N/K，而非只记请求 batch。小 M 从成熟 GEMV/多向量复用开始；大 M 从成熟 tiled GEMM 开始。[MLX host selector](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/matmul.cpp)应与[对应 GEMV](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/kernels/gemv.h)连读。
2. **先估数量级，不把估算当实测。** 下文的理想 W 流量和 `2MNK` 只能帮助提出 bandwidth/compute 假设；packed metadata、重复读取、tail 与提交都可能主导。再用[profiling](profiling-playbook.md)与同 shape 对照检验。
3. **只在并行不足时试 split-K。** 先看输出 tile 网格是否足够，再对比不拆/拆 K 的完整输出；将 partial scratch、归并和不同求和顺序纳入，不能只计主 GEMM。
4. **量化解码放在哪里，按完整边界决定。** 阅读 [MLX quantized selector](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/quantized.cpp)，对照下文五条路线。常驻展开只适合用额外内存换重复收益的场景；逐次展开必须算 dense 写回和第二次读取。

两个可复用的决策公式：若每次完整消费者节省 `Δt>0`，额外准备成本为 `C`，估算回本调用数为 `ceil(C/Δt)`；若 `Δt≤0` 则无有限回本。常驻展开的逻辑额外存储先按 `N*K*dense_bytes` 估计，再另测实际 allocator/进程峰值。两者是规划模型，不是已观测收益，搜索成本的真实案例见 [BEAM 回本](case-studies/tinygrad-beam-search-and-amortization.md)。

## 从 M 决定问题类型

新增[MLX dense small-M wide GEMV](case-studies/mlx-small-m-wide-gemv-and-head-attribution.md)：M2..15多向量复用权重、≤5向量/register tile、vocab宽网格与alignment guard；真实Qwen固定hidden的全词表head控制同时展示为何shape相关不等于GEMV因果。不是本机wide开关性能A/B，也不是packed量化或M5/NAX资格。

令 `Y[M,N] = X[M,K] W[N,K]^T`，dense 乘加工作量按 `2MNK` FLOPs 计。若 W 每元素 b bits，仅一次读 W 的理想流量约 `NK*b/8`，还未包含 scale/zero、X/Y、重复加载、scratch 与 cache miss。M 增大可提高同一权重被多行复用的机会；这是算术强度模型，不是实测带宽/峰值。

**单路 decode / 极小 M**：权重读、metadata 和提交固定成本容易占主导。向量化加载、多输出共享 X、simd reduce、低比特直接消费常更重要；为了用矩阵原语硬补成大 tile 可能浪费计算。

**prefill / DiT 大 M**：矩阵 tile 的复用更有价值。packed decode 一次供多行、scale/bias 重用、输出融合，能摊薄解码成本；decode 速度、寄存器压力、TG memory 和占用仍会限制收益。

这解释为什么 MLX 的 `get_qmv_batch_limit` 按 architecture 和 D/O 改门槛，为什么一个固定 Q4 kernel 不适合所有 M。它是读本中的经验 dispatch，不是硬件定律。

## 五种实现方式及其真实代价

| 路线 | 在哪里解码/复用 | 适合验证的假设 | 必须计入的代价 |
| --- | --- | --- | --- |
| packed GEMV | 直接读取 codes，寄存器局部 decode+dot | 小 M 减少 W 流量 | metadata、位运算、reduction、launch |
| packed tiled GEMM | 解码到寄存器/TG，再供矩阵 tile | 同 W tile 被多行复用 | barrier、tile load、tail、TG 与 register 占用 |
| resident dense cache | 加载时解码一次，重复 GEMM | 相同权重频繁重用、内存充足 | 首次 decode、额外完整 W、cache 失效/LoRA 版本 |
| per-call dense materialization | 每次先写完整 dense W，再 GEMM | 现有 packed 核弱、GEMM 特别强 | 全局 dense write/read、分配、峰值与额外 dispatch |
| 原生量化 TensorOps/NAX | 合法量化 operand 送硬件/框架路线 | 新 GPU 对相应格式有高效支持 | repack/requant、动态 X quant、scale、精度预算、guard |

原生框架的 normal dense matmul 也可能有 SIMD split-K、矩阵专用路线或不同 accumulator，不可拿 scalar 教学核作“现有 GPU”分母。

## 三份源码怎么连起来读

**MLX**：`quantized.cpp` 的 shape/stride、QMV/QMM/NAX selector → `quantized.h` / `quantized_nax.h` 的 packed load 与 decode。affine 语义为 group 的 `scale*code+bias`，不是统一的 signed int4；scale/bias 的 storage dtype 与舍入顺序是合同的一部分。`ensure_row_contiguous_matrix` 也说明逻辑 shape 一样并不意味着没有复制。[详细旧版读本](../sources/repos/mlx-affine-dispatch.md)与主线新读本不可混作同一个 binary。

**GGML**：host 的 `MUL_MAT`/`MUL_MAT_ID` → dtype 专用 Metal decode/GEMV/GEMM → backend scheduler。Q4_0、Q8_0、K/IQ 等 block 格式不可互换，GGUF 是容器不是数学公式。[独立 Q4_0/Q8_0 摘录执行](../sources/repos/ggml-metal-layouts.md)已经覆盖 nibble/signed 和 subnormal 反例；不把两种模板的通过扩大成整个 GGML 引擎通过。

**ccv/MFA**：descriptor 构造 shape/precision/cache key → kernel descriptor → 生成源码与 binary archive。已读 `NAInt8MatMulSmallMDescriptor` 在满足对齐且 K 足够大时选择 split 4/8，按 split 数分配 int32 partial scratch；这些阈值属于该实现。读到它不能证明本机 M4 在运行这条 NAX 路径。

补充：MNN 的普通 matmul 读本提供容易理解的 stride、bias、onResize/onEncode 分层；Candle 的 Rust backend 将布局 stride/offset 传给 `call_mlx_gemm`。它们说明集成/dispatch 成本也值得读，不支持“哪个语言一定更快”的推论。

## split-K、预取与双 buffer：有收益也有账单

当 `ceil(M/BM)*ceil(N/BN)` 太少、K 很大时，沿 K 拆任务可增加并行；但多出 partial buffer、reduction、launch 和不同求和顺序。M/N 增大到本来就占满设备时，继续 split-K 可能只增加流量。

[MLX #4682](optimization-prs.md)进一步把 packed load/metadata fetch 与 decode/store 分开，预取下一 BK、双 buffer 权重，并限制 dtype/bit/shape/对齐后才走新调度；split-K 保留原 K 为物理行 stride。它是待审 patch，不是当前 wheel 的能力声明。实验要特别加入中等 M 和 K 尾部，验证“逻辑 split K”没有误当“物理 row pitch”。

MPP 的 dense device-direct 建议与这个 packed 案例应一起学习：避免把 CUDA shared-memory 流水机械搬来，也避免把“device-direct 常合适”误写成“所有压缩格式都不能暂存”。选择由相同 source、实际 shape 与 trace 决定。

## 已有更强反例

[真实 Qwen FFN 报告](case-studies/affine-ffn-cost-and-resident-tradeoff.md)中，Q8 同配方 resident decode 只快约 4.7%，额外增加 288 MiB；相对原始 BF16 GPU 图则基本持平。原创 SIMD-group 核比教学 scalar 好，不代表比成熟 MLX 好。新合成 M 扫描补充了更小权重/更多 M 的机制观察，但不覆盖真实图。

必须区分三个比较：同 packed 格式的实现质量、量化相对原模型的性能/质量，以及整个请求的内存/延迟 Pareto。只报告其中最容易取胜的一层，会得到误导性的采用结论。

## 最小实验合同

固定 code order、group、scale/bias dtype、signedness、source 权重、物理 pitch；验证 group 边界、全零/极值/subnormal、M/N/K 尾部、非连续 view。CPU 独立 decoder 与 FP64 dot 负责见证；finite 不等于正确。全输出通过再量 ABBA，分别列准备、热态、resident logical bytes、实际 footprint、完整消费者和取消 drain。
