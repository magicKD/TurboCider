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

物理拼接也不是融合 kernel 的必需条件。若一个 threadgroup 只处理某个投影，可按 uniform projection kind 选择原 Wq/Wk/Wv buffer，并把全局列坐标映射到该矩阵的局部列，保留实际 row pitch、offset 与 owner。[Qwen 原buffer直读案例](../../docs/status/qwen512-corrected-qkv-direct-2026-10-18.md)省去了额外的持久权重bank；更多buffer binding、选择逻辑和不同tensor extent仍需重新测完整请求。只说明未创建这份bank，不等于所有activation、框架或driver staging都零复制。

Down 的列切片是 full-row strided view；在 IO、Core ML stage 或 Metal kernel 中继续传 physical pitch。只有显式 compaction 后才能用局部宽度作 stride，并把 compaction 的时间/存储算进去。

共享 backing 的 view 会保住整块 allocation。表面只剩几行 history，也可能 pin 完整时序 concat；需要拥有独立紧凑存储并实测 active/footprint。反过来，随意 `copy()` 也不必然产生独立物理 storage，必须检查已 evaluated leaf 与 owner。

## 数值融合的四个陷阱

1. `cast(a+b)` 与 `cast(a)+cast(b)` 不同；FP16/BF16 scale 的 affine decode 先 widen 可能改变原系数。
2. 把 bias 合进 GEMM、让 FMA 或 reduction 顺序改变，可能改变既有 rounding。exact 优化应保留边界，近似优化需要独立 recipe/许可。
3. LoRA gate/up 必须在 SiLU 前；down 使用 corrected hidden。所谓 `lora_fused` 不是把 adapter 合入权重。
4. norm 局部 tile 统计不是 whole-frame norm；fusion 不能为省内存而偷换算子语义。

仅保持中间 dtype 还不够：F32 `a*c-b*s` 是否跨乘法与加减做 contraction，也可能在最终 BF16 舍入边界改变少量值。用 matched-input 的逐阶段反事实区分 base、B、norm 与 RoPE，再把数学控制限制到已验证的表达式 scope；全局改变快速数学会同时改变其他算子的 recipe，需要另一组对照。[Qwen RoPE 案例](../../docs/status/qwen512-corrected-qkv-rope-rounding-2026-10-18.md)展示了局部差异经多步放大、局部修复后再验证完整 final latents/RGBA 的过程；单个 block 相同不替代整图验证。

合成小矩阵也可能落入与生产shape不同的GEMM bucket，导致base本身的累积/舍入不同。可固定输入，只把融合体内base替换为原生base，观察其余阶段是否恢复，而不是提高误差容忍或换seed直到通过。保留反例，再按实际投影geometry检查真实消费者；不能把小shape失败一律归因cache，也不能用生产样本通过宣称所有shape等价。读取host完整数组前还须确认逻辑顺序：dtype转换未必消除转置/非紧凑layout，直接遍历data指针可能比较了不同存储顺序。对应诊断边界见[QKV前缀组合案例](../../docs/status/qwen512-qkv-prefix-composition-2026-10-18.md)。

typed数学调用本身也有边界：某版本BF16 sigmoid中的`exp(abs(T))`可以先返回T，再参与后续标量运算；不能替换成全F32 logistic最后一次cast并假定等价。应对gate/up修正、sigmoid、SiLU和最终multiply分别做实际库见证，再接完整down/residual。若要求bitwise，F32差值为0还不足以区分正负零，需要按原dtype、逻辑顺序比完整字节。验证虽放在timer外，也可能扰动下一样本的cache/时钟状态；保留相同数学的普通重放控制，不能只择有利窗口。[typed SwiGLU案例](../../docs/status/qwen512-lora-swiglu-block-2026-10-18.md)说明数值收敛和局部性能资格是两项独立问题。

若融合大base GEMM会改变选核或舍入，可先保持原生base，融合其后的成对LoRA-B与非线性，减少corrected gate/up的发布边界；这不保证所有逻辑concat原本都会物化。将range/index/up/scale解析一次，只共享已存在的factor owner；在线A、cast和完整down仍须每次计入。随后用连续完整请求验证，独立保留原始tensor、缓存恢复和无dump计时，不能把逐block同步窗口代替整图。[完整请求案例](../../docs/status/qwen512-lora-swiglu-request-2026-10-18.md)在有限上下文观察到小幅增量，不授予跨shape或独占设备保证。

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

成对dense投影可以直接读原gate/up两个物理row区间，在一个kernel内保留两份
accumulator，或用threadgroup暂存已舍入gate，再计算typed SiLU×up。它与先
拼接权重的方案不同，但共享X view也不证明物理load复用；双accumulator的
register占用与shared/barrier要独立比较。先用未融合MPP控制区分GEMM与epilogue
差异，再接原F32 down和同窗原hybrid强控制。一个小幅、尚未完成整图验证的
实例见[Qwen原buffer dual-GUP](../../docs/status/qwen-base512-dual-gup-block-2026-10-19.md)。
