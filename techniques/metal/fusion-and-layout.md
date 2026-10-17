# 融合与布局：少读写，但保留数值配方

<!-- knowledge
id: technique-metal-fusion-and-layout
type: method
stage: encoder, dit, ffn, vae
operator: norm, rope, swiglu, layout
backend: metal, mlx
device: gpu
symptom: memory-bound, launch-bound, wrong-layout, precision-loss
semantics: numerically-equivalent
representation: fp16, fp32, packed
boundary: producer-consumer
api: public
-->

[总目录](../README.md) · [GPU/ANE LoRA](../gpu-ane/lora.md)

## 怎样借鉴：沿 producer→consumer 找可消除的中间量

2026-10-17 方法整理。融合首先是一项数据流改造：若两段间的 tensor 没有其他消费者，可尝试在同一 kernel 内保留局部值，减少中间写回/再读取与 launch。不要先以“kernel 数越少越好”设目标；更大的寄存器/TG 占用、重复计算或新的重排可能抵消收益。

以 `norm → RoPE → attention` 为例：先列出 norm 的统计轴、epsilon、position、Q/K head mapping 与输出物理 strides，再决定融合 norm+RoPE；让 attention 直接消费该布局，比融合后又 `contiguous` 一次更值得检查。参考[本地 Q/K 实现](../../native/models/qwen21/metal/qk_norm_rope.hpp)与[独立数学控制](../examples/reference/operators.py)，不要把只适合该模型的 head/position 规则当通用 API。

可照搬的验证步骤：先比 separate/fused 的**完整边界输出**，再接真实下游；加入非零 offset、非紧凑 pitch、错误 position、LoRA 更新和明确 rounding 见证。计时保留需要的 cast/copy/layout 转换，并同时记录 scratch/峰值。只移走读回或把准备移到窗外，不算融合收益。

跨项目的两个反例尤其有用：[Candle packed view/copy](case-studies/candle-packed-views-and-copy.md)说明逻辑 contiguous/copy 不自动保证正确 origin 或独立 storage；[MLC 原图 stage 观察](case-studies/mlc-ffn-stage-observation-and-precision.md)说明中间值逐位相同也不保证同步/flush 成本不变。学布局与融合时，应一起学习这些否决条件。

## 最值得先测的边界

先检查消费者是否真的需要所有输出。若某些行在后续图中完全无人使用，可把需求向上游传播，避免“先算完整 tensor，最后才丢弃”。例如最后一层 attention/KV 已完成后，末层 FFN/head 有机会只计算所需位置；较早层仍用于建立后续 K/V 的 hidden 不能提前裁掉。参考 [GGML 输出需求案例](case-studies/llama-pretrained-output-demand.md)。

迁移时先画出依赖/输出集合，再比较完整请求，而不是按裁掉行数估算倍率。更小 M 会改变选核与浮点顺序；逐 token 评分可能确实需要所有 logits，DiT 也通常需要全部 image tokens。需求裁剪不是删除 prompt、降低分辨率或近似 attention。

| 候选 | 消除的成本 | 必须保留 |
| --- | --- | --- |
| Q/K norm + RoPE | 多次 dispatch、中间 Q/K 遍历 | 各自 norm 轴、epsilon、position、head layout |
| GEMM + bias/SiLU×up | gate/up 临时和 epilogue launches | gate/up 顺序、激活前 LoRA、dtype/舍入 |
| partial + residual / modulation | hidden 中间写回 | bias/residual 恰好一次、下游依赖 |
| typed affine decode + finite check | 多次完整 W 扫描 | 原 metadata dtype、完整非法输入拒绝 |
| global norm + SiLU + halo pack | 全尺寸 normalized tensor | 全局统计、halo/padding、temporal taps |
| shared LoRA A / paired B | 重复读取 X / rank dispatch | 每投影 alpha/scale 与有序 rounding |

实现可参考 [Qwen Q/K kernel](../../native/models/qwen21/metal/qk_norm_rope.hpp)、[dense base-LoRA](../../native/backends/dense_gpu_base_lora.hpp)、[SeedVR2 norm-pack](../../docs/design/seedvr2-vae-norm-pack-and-resource-advice.md)。每个是不同的验收范围，不组合成未经测试的通用开关。

## “逻辑拼接”不等于“物理紧凑”

FFN 的 `[gate|up]`、QKV 的 head 维和 packed W planes 都要有显式 mapping。做 channel split 时，gate 范围 `I` 对应 up 的同一 `I`，不是切 fused tensor 的前一半给 GPU、后一半给 ANE。

Down 的列切片是 full-row strided view；在 IO、Core ML stage 或 Metal kernel 中继续传 physical pitch。只有显式 compaction 后才能用局部宽度作 stride，并把 compaction 的时间/存储算进去。

共享 backing 的 view 会保住整块 allocation。表面只剩几行 history，也可能 pin 完整时序 concat；需要拥有独立紧凑存储并实测 active/footprint。反过来，随意 `copy()` 也不必然产生独立物理 storage，必须检查已 evaluated leaf 与 owner。

## 数值融合的四个陷阱

1. `cast(a+b)` 与 `cast(a)+cast(b)` 不同；FP16/BF16 scale 的 affine decode 先 widen 可能改变原系数。
2. 把 bias 合进 GEMM、让 FMA 或 reduction 顺序改变，可能改变既有 rounding。exact 优化应保留边界，近似优化需要独立 recipe/许可。
3. LoRA gate/up 必须在 SiLU 前；down 使用 corrected hidden。所谓 `lora_fused` 不是把 adapter 合入权重。
4. norm 局部 tile 统计不是 whole-frame norm；fusion 不能为省内存而偷换算子语义。

## MLX compiled graph 的实战规则

- 编译稳定 block / GPU complement / hybrid 交接前后段，避免强同步拆碎图。
- 按 shape、dtype、precision recipe、source/adapter generation 和路线建立有界 cache；变更 adapter 时不能继续捕获旧 A/B。
- `eval` / `async_eval` 是执行边界，普通表达式构建不是 device submission；定时前把上游依赖与目标操作区分。
- 不把 `pre_ffn` 主机时间全部叫 attention，也不把 lazy join 的 graph construction 变短解释为算术消失。

API 背景见 [MLX custom extensions](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)：MLX 自己管理执行图、stream 和 command buffer；需要精确 GPU/ANE event 时不能凭空把一次表达式构建当成外部 queue 已提交。

## 负结果如何迁移

[affine fused finite](../../docs/status/local512-affine-fused-finite-2026-10-10.md)降低转换准备时间，但 decode+GEMM 仍比直接 packed QMM 慢；不因 decode kernel 更快就默认预展开整个模型。

[native SGMMA E2E](../../docs/status/2026-10-08-native-lora-sgmma-e2e.md)说明单层完整 FFN 的改善也不足以证明独立 native 后端超过优化后的 MLX。继续优化应看真实 graph/adapter/同步成本，而不是用旧构建 scalar 时间作因果分母。

## 实现参考与实验

可读实现：[参考代码](../examples/reference/operators.py)；[实验步骤与边界](../examples/README.md)；本项全部 case / controls。

本项控制：separate/fused complete boundary；wrong positions and explicit rounding witness。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：new epilogue, tensor layout, precision or compiler contraction policy。历史负结果限定于其来源/shape/边界，不关闭该方向。
