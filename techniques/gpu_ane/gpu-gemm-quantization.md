# GPU GEMM / GEMV：量化、tile、dispatch 与 source layout

<!-- knowledge
id: technique-gpu-ane-gpu-gemm-quantization
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

[专题导航](README.md) · [引擎地图](inference-engines.md) · [优化 PR](optimization-prs.md) · [本机实验](../MAINTENANCE.md)

核心读本：MLX `36969c094354086e15d2a327a9c50feb20b9cf11` 的 [`quantized.cpp`](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/quantized.cpp) 56–136、1056–1225；GGML `64e9bce…` 的 [`ggml-metal-ops.cpp`](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/ggml/src/ggml-metal/ggml-metal-ops.cpp) 2390–2555。这是所读路径，不代表完整 kernel 审计。

## 为什么 decode 与 prefill 不能共用一组参数

对 `X[M,K] @ W[K,N]`，dense 工作量约 `2MKN`。当 M 很小时，同一权重复用少，读取/解码权重和 launch 更重要；M 增大后，提高 tile 内复用与矩阵单元利用率更重要。这是成本分析，不是某台芯片的 counter 结论。

MLX `get_qmv_batch_limit` 按 architecture 和 D/O 选择不同阈值，而不是“batch=1 用 GEMV，其余一律 GEMM”。GGML 读本也有 `mul_mv_ext` 的小批参数、`use_mm` 条件与普通 `mul_mv` fallback，并在注释中保留 grid/tail 未充分解释的地方。工程上应学习分层 dispatch，而非把一个经验阈值当定律。

## 三条量化计算路线

| 路线 | 数据流 | 可能优势 | 代价 |
| --- | --- | --- | --- |
| packed 直接消费 | 每 tile 解码 codes/scales/bias 后复用 | 少常驻字节、避免完整展开 | 解码指令、metadata、寄存器/共享内存 |
| 预先展开常驻 FP16/BF16 | 一次 decode，后续用 dense GEMM | 多次使用摊销、成熟 dense 核 | 额外驻留、准备成本、带宽，可能挤出其他层 |
| 原生量化 TensorOps | repack 为受支持 tensor 后矩阵运算 | 特定新硬件/格式快路径 | 格式兼容、repack/动态量化和新质量配方 |

比较前两条时，用**同一份 packed source 解码值**作 reference，不将原 FP16 checkpoint 与量化权重互比。一次 decode 被排除的 warm resident 测试必须明确排除，不外推首请求或低 RAM。

MLX affine 可概括为 `w = scale[group] * code + bias[group]`；GGUF K-quants/IQ 又有不同的 metadata、符号和码本。文件容器 GGUF 不是量化数学定义。学习现有 [GGML layout 导读](../sources/repos/ggml-metal-layouts.md)，用独立 decoder 覆盖 group 边界和 scale 舍入。

## Tile、split-K 与局部数据复用

M/N tile 太小会让每次解码的权重复用不足，太大又增加 accumulator、寄存器和 threadgroup memory。搜索变量至少包括 BM/BN/BK、simdgroup 布局、向量宽、load order、decode 位置、tail 处理、accumulator dtype；不是单独调 threads。

输出 tile 数不足时，split-K 把 K 分段形成更多 threadgroups，再加和 partials。它提升并行度但增加 scratch、launch 和 reduction 流量。MLX 读本的非 NAX `qmm_splitk` 以约 512 threadgroups 为启发式，K 分区至少同时满足 BK=32 与 quant group 对齐，partial 用 FP32。若只按较小 group_size 切，可能越界读到下一组。

NAX 的 [PR #4682](optimization-prs.md)则是另一组 tile/split 限制，并尝试在当前 MMA 时预取下一 K 段 packed words，随后解码写入 threadgroup。prefetch/double-buffer 能隐藏 latency，也能增加 register pressure；不能仅凭“多 buffer”判优。

## Layout 是热路径的一部分

MLX `ensure_row_contiguous[_matrix]` 可能插入 copy 并保留 temporary 到完成。view 的 shape 正确但 physical stride 不同，不能假定免费。GGML 的 `nb*` stride 参数、buffer/view source 与 pipeline 选择也需对应。

跨 GPU/ANE 切 packed columns 时，要按完整 quant group/packed word 切，并对应 scales/bias；不能把 FP16 element offset 直接用于 packed bytes。down projection 的 F 分片需要区分逻辑 K 与物理 row pitch，尤其不能在 split 后把 K 错缩为 shard width 而继续原寻址。

## 推荐实验

冻结权重语义与 seed，扫 M=1/16/256/1024、N/K tail、group 和 dtype；增加非连续输入、偏移对齐、零/极值、部分 quant group 负控。分别报告准备、kernel、完整 FFN、peak/resident bytes 和请求。

本轮 [GPU 小实验](../MAINTENANCE.md)比较编译融合、packed-vs-predecoded 和 SDPA；预展开并未稳定胜过 packed，是该 1024×1024 warm 边界的观察，不是 dense GEMM 普遍无用。更强的真实 [Qwen GPU FFN 研究](../metal/case-studies/affine-ffn-cost-and-resident-tradeoff.md)单列，不把两组倍率拼接。

重开条件：新 NAX kernel、SDK、quant format、M 分布或驻留预算变化；GPU 提速后也要重测 hybrid 的最优 share，不能相乘加速比。

第十三轮另读[GGML few-row MMA](case-studies/ggml-fewrow-mma-and-reuse.md)：选定Oct7 ref在既有mul_mv_ext之前增加2–16行的SIMD-group矩阵路线，格式阈值、K step和decode复用均单列；不是M5 TensorOps专属。[原始residual回归与负对照](case-studies/ggml-residual-identity-regression.md)已在M4运行，但没有性能A/B或模型分母，不能把PR作者倍率当本机结果。

第十四轮另有[同图融合开关计时](case-studies/ggml-matched-fusion-timing.md)，不是MMA对legacy MV的比较：两进程使用相同packed weights和非dyadic F32输入，保留N=1漂移及短pilot，再作反序长块确认。小图N=8局部有收益，大矩阵多数接近持平；不把49,152次host同步调用当device timestamp或完整FFN/模型分母。

第十七轮进一步读[MLX #4682最终组合源码](case-studies/mlx-nax-mid-m-and-prefetch.md)：route helper为真仍可能split=1；新16-row schedule和split重路由分开，unaligned offset在split入口先copy，full K仍是原row pitch。packed权重解码到FP16/BF16再进float累加的MPP，并非直接INT4矩阵乘；两个原始CPU策略函数3430点已执行，M5 kernel和模型倍率仍未在本机验证。
