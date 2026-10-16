# Attention 实现对照：MFA、MLX、GGML、PyTorch MPS 与 SolAttn

[GPU 导航](README.md) · [数学/质量合同](attention.md) · [PR](optimization-prs.md)

核查：2026-10-16；已读范围见[来源清单](research-sources.json)。下面是机制比较，不是跨项目跑分。

## 共同问题：不把整个 score 写进内存

普通 `softmax(QKᵀ)*V` 会生成 `B*H*Lq*Lk` score。online softmax 用每行 running max、归一化和与未归一化 output，在遍历 KV tile 时修正旧累加，最后除以总和；避免完整 score 的全局驻留。这在实数数学上与 dense 相同，浮点求和/舍入顺序不同。

prefill 的多 Q 行可复用同一 KV tile；decode 的 Q 很短，常需要不同 vector/两 pass 路线增加并行。GQA 共享 KV 不能简单理解成提前 repeat 每个 KV head；repeat 会增加访存/内存，实际 kernel 的 head 映射和支持条件才是关键。

## 各项目具体怎么做

| 实现 | 已读入口 | 机制 | 需要特别验证 |
| --- | --- | --- | --- |
| 独立 MFA `8671cdd` | Swift AttentionKernel 的 softmax/caching 与 descriptor | 生成 tile 化源码，按 D/精度选择驻留与 spill 策略 | single-head 教学范围、多 head 扩展、D256 寄存器压力；README 的历史 counter 模型不是当前硬件保证 |
| MLX `36969c0` | `scaled_dot_product_attention.cpp`、Steel | full/vector/two-pass/NAX、mask/GQA、特定 D padding | actual wheel guard、LQ/LKV、dtype、输出物理 layout；`fast` 不保证每 shape 都 fused |
| GGML `64e9bce` | Metal `FLASH_ATTN_EXT` host/kernel | KV dtype/维度/形状专用 kernel 与 backend 图执行 | quantized KV、head stride、GPU table、mask/softcap 等操作语义 |
| PyTorch `eb429a0` | `Attention.mm` L790–865 | vector、prefill、MPP 与 MPSGraph 路径；按末维 stride 决定 contiguous | dropout、mask dtype/causal 组合、GQA、训练 backward 与 inference 条件 |
| ccv `66dc16a` | `NAAttentionKernel.cpp`、descriptor | NA 数据流、精度/内存布局生成、tile 与 scratch | 不把 NA 当 ANE；从上层调用追到设备 guard 和具体 variant |
| SolAttn `4507112` | `_metal_tiled_fwd.py`、routing | 块 summaries、重要块精算、其余近似贡献 | 数学已近似、routing 开销、阈值/模型质量/时序稳定性、可靠 dense fallback |

MFA 的[设计读本](https://github.com/philipturner/metal-flash-attention/tree/8671cddc38f19a6eadb804dee6a3ca2954b8bf32)、[MLX](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/scaled_dot_product_attention.cpp)、[PyTorch](https://github.com/pytorch/pytorch/blob/eb429a042a8f30a486e86eedaf140d1de53aa05b/aten/src/ATen/native/mps/operations/Attention.mm)是可追溯入口。SolAttn 包含 PyTorch 内部 `PrefillAttention.h` include；这种复用带来版本耦合，不能仅因当前 shader 编译成功就宣称长期可移植。

### PyTorch 的“math backend”不等于没有 fused Metal

固定读本中 `_scaled_dot_product_attention_math_for_mps` 的实现本身就可能选择 fused vector/prefill/MPP；真正退到 `sdpa_general_mps` 才是另一条图路径。因此 [#198564](optimization-prs.md)提议把已有 prefill kernel 暴露为 `SDPBackend::flash_attention`，不能解释成此前 MPS 完全没有高效 attention。它仍未合并，训练/LSE 合同须另审。

### 宽 tile 和 NAX 不是无条件赢家

MLX [#4476](optimization-prs.md)提议在 causal D256 的一段短 prefill 范围改变 dispatch，附 dtype/GQA 对照；已关闭未合并，不是当前主线已有能力，也不是一律把所有短 Q 送 NAX。GGML [#28439](optimization-prs.md)用 16-row query tile 提高 KV 复用，但按 SKU、可用 tile 数、TG memory 和尾部 padding 回退；它展示的是如何建立有边界的选择器，不是“tile 翻倍就更快”。

## 五个最容易忽略的正确性条件

1. **全空 mask 行**：有限最小 sentinel 与真正 -inf 不是同一语义；全空行可能有限但不为零，也可能 NaN。先约定零输出/拒绝等行为，再测内核。
2. **非方 causal**：左上或右下对齐会改变结果，decode/prefill cache 场景尤其敏感。不能只测方阵。
3. **Q/K/V strides**：末维连续不等于全 tensor contiguous；强制重排一个合法 view 可能只有成本没有收益。输出 token-major 布局也可能是下游有意设计。
4. **Dq/Dv、GQA 与 tail**：读 host guards，不能假设所有 D 都受支持；zero padding 必须保持原 scale 与 mask，并把裁剪/转换计入消费者。
5. **softmax 精度与范围**：输入、QK/PV accumulator、exponent/denominator 精度分开；随机温和输入通过不替代高幅值/长 KV/全空行控制。

本库已有[两个真实 seed 的 Qwen 对照和空行反例](../review/qwen-attention-20261016.md)，并保留 FP64 row witness 范围，不宣称整个生成质量已通过。新 [组件实验](experiments-20261016.md)只使用 maskless FP16、GQA1，并做全输出 CPU FP64；它不覆盖 causal、empty-row 或 quantized KV。

## 建议的三臂评价

先比较 dense-native 与 dense-custom，评价 kernel/dataflow；再比较 sparse-custom，单列近似程度、routing/summary 成本、误差与视频质量。加入 all-exact 控制证明 sparse wrapper 本身正确；晚层/晚 step、长短序列、多 seed、LoRA 与真实输出恢复均通过后，才讨论模型采用。不要把降低分辨率、裁 token 或少 steps 算作同语义 Metal 加速。

## 可执行学习控制（不替代上游验证）

本页实现参考：[labs/operators.py](../labs/operators.py)、[labs/metal/engine_probe.py](../labs/metal/engine_probe.py)；case：`online-attention`、`mlx:engine-mechanisms`。CPU/旧 Metal 控制只验证对应数学与合同；`mlx:engine-mechanisms` 对应[显式组件 CLI](experiments-20261016.md)，不是自动启动 GPU 的普通单元测试，也不证明全部上游引擎或 PR 已运行。设备、source、shape、精度或消费者改变后重新验证。
