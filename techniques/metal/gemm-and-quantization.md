# GEMM/GEMV 与低比特 GPU 数据流：相同位宽不代表相同方案

[GPU 导航](README.md) · [kernel 调优](kernel-tuning.md) · [TensorOps](tensorops-and-m5.md) · [实验](experiments-20261016.md)

核查：2026-10-16。本文综合 MLX/GGML/MFA/MNN/Candle 的已读实现；源码与范围见[来源清单](research-sources.json)。不复制第三方 kernel，不按文件大小推算实际提速。

## 从 M 决定问题类型

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

[真实 Qwen FFN 报告](../review/gpu-affine-20261016.md)中，Q8 同配方 resident decode 只快约 4.7%，额外增加 288 MiB；相对原始 BF16 GPU 图则基本持平。原创 SIMD-group 核比教学 scalar 好，不代表比成熟 MLX 好。新合成 M 扫描补充了更小权重/更多 M 的机制观察，但不覆盖真实图。

必须区分三个比较：同 packed 格式的实现质量、量化相对原模型的性能/质量，以及整个请求的内存/延迟 Pareto。只报告其中最容易取胜的一层，会得到误导性的采用结论。

## 最小实验合同

固定 code order、group、scale/bias dtype、signedness、source 权重、物理 pitch；验证 group 边界、全零/极值/subnormal、M/N/K 尾部、非连续 view。CPU 独立 decoder 与 FP64 dot 负责见证；finite 不等于正确。全输出通过再量 ABBA，分别列准备、热态、resident logical bytes、实际 footprint、完整消费者和取消 drain。

## 可执行学习控制（不替代上游验证）

本页实现参考：[labs/quantization.py](../labs/quantization.py)、[labs/metal/engine_probe.py](../labs/metal/engine_probe.py)；case：`quant-formats`、`mlx:engine-mechanisms`。CPU/旧 Metal 控制只验证对应数学与合同；`mlx:engine-mechanisms` 对应[显式组件 CLI](experiments-20261016.md)，不是自动启动 GPU 的普通单元测试，也不证明全部上游引擎或 PR 已运行。设备、source、shape、精度或消费者改变后重新验证。
