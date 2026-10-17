# LoRA：共享图不能漏掉激活修正

<!-- knowledge
id: technique-gpu-ane-lora
type: method
stage: ffn
operator: gemm, swiglu
backend: cpu-reference, mlx, coreml
device: cpu, gpu, ane
symptom: precision-loss, wrong-layout
semantics: numerically-equivalent
representation: fp16, fp32
boundary: gpu-ane, producer-consumer
api: public, reference-only
-->

[总目录](../README.md) · [切分代数](partitioning.md)

## 完整数学顺序

对每个 projection，`Δ(x) = scale · (x Aᵀ) Bᵀ`；scale 包含实际 strength/alpha/rank 约定，多 adapter 按模型原顺序处理。

```text
g = x Wgᵀ + Δg(x)
u = x Wuᵀ + Δu(x)
h = SiLU(g) * u
y = h Wdᵀ + Δd(h)
```

gate/up 修正必须进入 SiLU 前。down-LoRA必须消费 corrected hidden，包括 ANE 的全部通道；只对 GPU suffix 做 down-LoRA 是错误的。base 请求提供零修正，不改 base slots。

## Frozen 图与 runtime adapter

adapter-bound frozen artifact 必须绑定实际 adapter 内容、role、strength；不能把 base 图冒充合并后的 LoRA。共图路线则让所有 checkpoint/ANE slots 只存 base，GPU 算 activation corrections，图提供 corrected hidden ABI。

`lora_fused` 在本仓库指 base FFN 算子融合，不是权重 merge。premerged 模型、derived dense view、runtime LoRA 是三种不同来源，报告和 cache key 不混用。

## 减少低秩重复工作

1. 同一 X / 同一 A 内容、dtype/recipe 可共享 input rank；gate/up 不一定天然同 A，先校验真实 bindings。
2. B epilogue 可融合 scale、cast、base add，保留 alpha/stacked ordering。
3. channel split 可用 `r = h_G A_Gᵀ + h_A A_Aᵀ`，再一次 B、一次 delta rounding、一次 base add，避免 full hidden concat。
4. 小 rank 不自动快：多个小 GEMM/dispatch、F32 operand cast、readiness fence 可能主导。计时必须在线重算 A→B，不能只拿 prepared rank B 时间充完整 LoRA。

第3项实数等价，但 FP32 reduction 次序与旧完整 hidden 路线不同，需另行数值/媒体验证。[split-down ranks](../../docs/status/local512-split-down-ranks-2026-10-08.md)中双参考局部改善、单参考变慢，仍没超过完整 GPU，因此不是默认推荐。

## 精度优化不能偷换名字

[联合 A/B](../../docs/status/local512-qwen-joint-ab-2026-10-09.md)用 BF16 operands/F32 rank 并保留特定 rounding，有 GPU 整图小幅收益；它改变所有符合条件的 LoRA projections，不只 FFN/ANE。不能把整图差值全归给 ANE，也不能标成原始 F32 operands 逐位等价。

规范记录：master A/B dtype、compute operands、rank dtype、B input narrowing、scale dtype、delta cast、base add 顺序；把有效选择绑定到请求 snapshot，别临时修改进程 env 导致同 Session 混用 recipe。

## 必测状态与来源

- 真实 adapter + 不同 rank/strength、正负 stacked terms、alpha、只有部分 projections 的 adapter。
- base→A→B→base；B 必须有内容 identity，合成 B 只能验证隔离，不代表第二个训练 adapter 质量。
- 非完整 A/B、orphan alpha、未知 keys、unused output rows、错 rank 在 loader 拒绝；失败后不残留部分 bindings。
- 全 hidden/row/channel callbacks、late chunk error、down callback 抛错、取消/重试、返回 tensor 后 executor 销毁。
- 性能用相同有效 LoRA recipe 的优化 GPU 为分母；base 40步收益不能借给六步编辑。

代码参考：[HybridFfn::Adapter](../../native/backends/ane_ffn.hpp)、[lora metadata](../../native/backends/lora_metadata.hpp)、[native LoRA contract](../../native/backends/metal_graph/native_ane_lora_contract.hpp)。

## 实现参考与实验

可读实现：[参考代码](../examples/split_ffn_reference.py)；[实验步骤与边界](../examples/README.md)；本项全部 case / controls。

本项控制：corrected hidden and one down B；native base/A/base and late-failure recovery。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：shared ranks, fused B, changed rank/strength or operator boundary。历史负结果限定于其来源/shape/边界，不关闭该方向。
