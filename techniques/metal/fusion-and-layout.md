# 融合与布局：少读写，但保留数值配方

[总目录](../README.md) · [GPU/ANE LoRA](../gpu-ane/lora.md)

## 最值得先测的边界

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

可读实现：[参考代码](../labs/operators.py)；[实验步骤与边界](../labs/gpu.md)；[本项全部 case / controls](../labs/catalog.json)。

```sh
python3 -B techniques/labs/run.py norm-rope
```

本项控制：separate/fused complete boundary；wrong positions and explicit rounding witness。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：new epilogue, tensor layout, precision or compiler contraction policy。历史负结果限定于其来源/shape/边界，不关闭该方向。
