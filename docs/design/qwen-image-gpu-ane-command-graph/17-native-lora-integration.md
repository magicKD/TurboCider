# 17 · Native GPU LoRA 全目标接入与后续 Hybrid ABI

更新：2026-10-08。本文区分**当前 Native GPU LoRA 候选实现**和后续 hybrid 设计。`61fe967c…` 的[生成组件证据](../../status/2026-10-08-native-lora-gpu-components.md)通过时请求门仍为 BaseOnly；其后新库 `6d914980…` 已开放显式 GPU Viggle r128／strength1／512²／六步／无参考图窄域，并完成一组 MLX→native 真实对照。227 个 target、物理完成与新增存储释放通过，最终图片目视接近；native request 25.437 s 对 MLX 20.939 s，慢 21.48%，保持 `experimental-unqualified`，没有自动采用或 App／hybrid LoRA 资格。详见[真实 LoRA 状态](../../status/2026-10-08-native-lora-gpu-e2e.md)。Base、缓存及 LoRA 的不同构建证据分别记录，不互相转授。

新构建 `77863b3f…` 的 SGMMA variant 已独立通过组件与一组真实六步执行；Native 23.987 s 对 MLX 20.960 s，仍慢 14.44%，默认 scalar／MLX 产品路线不变。组件收益和整图负结果见 §9 及[新 E2E 状态](../../status/2026-10-08-native-lora-sgmma-e2e.md)；未做同构建 scalar 整图，不把旧构建差异归因于新 kernel。

当前 `47d3ed80…` 增加 §5.3 的 ABI Spec／MIL emitter 与共享 low 编码接口，完整包构建、CPU 合同和微型共享 low 设备检查通过；没有接通 hybrid LoRA controller／模型／App。单层 MLX 暖机对照因内存预算保护提前结束，不能给予性能资格。准确实施边界和下一步见 §12 与[状态记录](../../status/2026-10-08-native-lora-foundation.md)。

核心决定：先实现 **完整 native GPU DiT 的 runtime LoRA**，覆盖实际模型所有线性投影；通过现有 MLX Transformer oracle 后，再增加独立的 native hybrid LoRA ABI。不能只接 FFN gate/up；不能把 weight merge、runtime activation correction 和已有 BaseOnly W8A8 当成同一精度路线。

## 1. 本地 adapter 的实际范围

只读取以下两个已有 safetensors 的 8 字节 header 长度及 JSON header；没有读取 tensor payload、导出权重、重新计算整文件 hash。下表 payload 是 header 的 `data_offsets` 差值求和，不是额外分配量。

本地目录：`/Users/kd/Documents/project/mac_image_generation/models/Qwen-Image-2.1/lora/`。

| 本地文件 | rank／dtype | 线性目标数 | alpha 的可观察事实 | tensor payload |
|---|---|---:|---|---:|
| `Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors` | 全部 r128／BF16 | 227，454 个 A/B tensor | metadata 声明 `r=128,lora_alpha=128,use_rslora=false`；没有 `.alpha` tensor | 648.0625 MiB |
| `Qwen-Image-2.1_NSFW_Image_Edit.safetensors` | 全部 r16／BF16 | 224，448 个 A/B tensor | 仅 `format=pt` metadata；没有 alpha tensor，不能从文件名猜 alpha | 80 MiB |

Viggle 的 227 个目标 = 32 层 × 7 + 3：

| canonical target | 个数 | A `[rank,input]` | B `[output,rank]` |
|---|---:|---|---|
| `transformer_blocks.i.attn.to_q / to_k / to_v / to_out.0` | 128 | `[128,4096]` | `[4096,128]` |
| `transformer_blocks.i.img_mlp.gate_layer / proj` | 64 | `[128,4096]` | `[12288,128]` |
| `transformer_blocks.i.img_mlp.out` | 32 | `[128,12288]` | `[4096,128]` |
| `time_text_embed.timestep_embedder.linear_1` | 1 | `[128,256]` | `[4096,128]` |
| `time_text_embed.timestep_embedder.linear_2` | 1 | `[128,4096]` | `[4096,128]` |
| `modulation.1` | 1 | `[128,4096]` | `[16384,128]` |

r16 adapter 只有上述 32×7 个 block 目标，将 rank 换为 16；使用 `.lora_A.default.weight`／`.lora_B.default.weight`，没有三个全局目标。两个本地文件均没有 `img_in`、`txt_in.*`、`norm_out.linear`、`proj_out` LoRA，但这些是真实模型的线性层，原生能力设计仍必须完整覆盖。

现有 `Weights::apply_loras` 的有效比例仍由 tensor 形式的 alpha 决定，不把 PEFT metadata 直接转换成另一数学规则。无 alpha tensor 时有效比例就是请求 `strength`；有 scalar alpha 时才是 `strength * alpha/rank`。因此本地 Viggle 在 strength=1 时恰好与 metadata 的 128/128 相容，但这不证明任意 PEFT alpha metadata 已受支持。新增 metadata compatibility 检查将非等价 alpha、RSLoRA／DoRA 等不支持变体记录为 native 拒绝理由，snapshot／importer 不得忽略；普通 MLX 路线的比例解释保持原合同。

当前 Qwen 请求层只允许一个 runtime adapter，普通 strength 范围 -8…8；已识别 Viggle 普通路线要求六步、strength=1，其他普通 adapter 使用 20…40 步 Base Euler。见 [qwen21_module.cpp](../../../native/models/qwen21_module.cpp) 264–286 行。原生首个模型资格建议仅开放本地 Viggle r128、strength=1、512²、无参考图、六步；rank16／其他 strength 可以先有算子和 tiny 能力，但不因此自动获得模型资格，更不能把所有 adapter 改成六步。

## 2. 现有代码的接入点与拒绝点

| 文件／边界 | 当前源码事实 | 尚待验证或接通 |
|---|---|---|
| [mlx.hpp](../../../native/backends/mlx.hpp)／[mlx.cpp](../../../native/backends/mlx.cpp) | 已新增 `runtime_lora_snapshot()`，保留 ordered A/B、dtype、原始 strength／有效 scale、output range、adapter ordinal、lease 和兼容性；`apply_loras_lease()` 与 pathname 共用 parser | real produced factors／generated oracle、失败与重载边界继续按合同验证；snapshot 不自动获得请求资格 |
| [mlx_native_denoiser.cpp](../../../native/models/qwen21/mlx_native_denoiser.cpp) | base importer 继续拒绝 adapter；独立 `import_runtime_lora_denoiser_weights()` 要求 verified held source 和 produced compact BF16／FP32 factors，逐 target／range 检查消费完整 | import 本身不 eval／cast／merge，不证明外部队列 readiness；Session 须持有并验证全部 adapter |
| [native_lora.hpp](../../../native/backends/metal_graph/native_lora.hpp)／[native_lora.mm](../../../native/backends/metal_graph/native_lora.mm) | 已有 `OrderedLoRATerm` 和 `NativeLoRA`：FP32 low／B、逐 term BF16 add、bias-last、独立 SiLU×up；默认五个通用 PSO、opt-in SGv2 另两个；新增 validated low／apply 拆分不加 PSO | Scalar／SGv2 的固定真实 r128 域、冷准备及整图成本已有测量，整图仍慢于 MLX；共享 low 目前只有 tiny 同图正确性，其他域与跨图 readiness 待验 |
| [native_block.hpp](../../../native/models/qwen21/native_block.hpp)／[native_block.cpp](../../../native/models/qwen21/native_block.cpp) | Projection 带 ordered terms；FFN 有 gate/up/down 三组 terms；attention 和 pre-SiLU／down 接线已存在；无 adapter 保留旧 fused FFN | 检查非交换舍入 fixture、全 target、alias、prefix／steady 和 Base 零额外分配；parallel plan 继续拒绝 LoRA |
| [native_denoiser.cpp](../../../native/models/qwen21/native_denoiser.cpp) | 八个全局投影统一 adapted helper；按实际最大 rank 准备一个 LoRA bundle；每请求共享 low／gate／up；prepare／begin 校验 adapter revision 和显式 schedule variant | 两层 generated 图及一组真实 32 层／六步窄域请求通过；性能未合格 |
| [native_memory_plan.cpp](../../../native/models/qwen21/native_memory_plan.cpp) | 已列 `lora-low`／`lora-gate`／`lora-up` sites，按 actual rank／FFN targets 加预算，同一 ledger；hybrid＋LoRA 明确拒绝 | Session 必须把 resolved rank／FFN requirements 送入 phase；全请求／driver 内存仍未认证 |
| [pipeline.cpp](../../../native/models/qwen21/pipeline.cpp)／[native_session.cpp](../../../native/models/qwen21/native_session.cpp) | native reload 强制重新 bind、same-fd 捕获及绑定后 materialize；Session 留存 lease，校验 strength／revision／variant，送入实际 rank phase／adapted import，发布实际执行证明 | `6d914980…` 一次真实请求通过；持久 worker 连续切换／故障回归仍需独立验收 |
| [native_route.hpp](../../../native/models/qwen21/native_route.hpp)／[native_denoiser_contract.hpp](../../../native/models/qwen21/native_denoiser_contract.hpp) | Base20 域保持；新增 GPU Viggle r128／strength1／六步精确候选，低层区分 Base20／ViggleLoRA6／GeneratedTiny，禁止 hybrid LoRA | 新库 12 项 metadata API 测试通过；filename 只选候选，实际内容由同一加载 FD 的 pin 验证 |

模型层应覆盖 232 个逻辑投影：32×7，加上 `img_in`、time linear1/2、txt linear1/2、`modulation.1`、`norm_out.linear`、`proj_out` 共 8 个全局投影。fused `gate_up` 是其中 gate/up 的共享物理存储，不是额外两个不同模型目标。

`txt_in.text_norm.weight`、`attn.norm_q.weight`、`attn.norm_k.weight` 是 norm 向量，不可当成普通二维 LoRA 投影；本章不新增任意 norm replacement/bias-training adapter。`norm_out.linear` 则是真实 H→H 线性层，必须支持。所有未消费的、形状不匹配的、未知变体 target 在产生 native submission 前拒绝；完整目标计数不能只用“应用数量 > 0”代替。

### 2.1 接入独立后端，而不是回落到 MLX projection

当前模型层把 LoRA correction 编入自身 `CommandGraph`，与 QKV／attention／FFN／Euler 一起由 native Backend queue 提交；MLX 仅在来源加载、物化与显式边界同步处参与。真实窄域请求没有用 MLX `Weights::project` 填补 native target；该函数在 generated 验证中是外部 oracle。encoder／VAE 仍使用既有 MLX，不代表全 App 已原生化。

对 [16 的后续接口](16-splash-native-backend-spec.md)，`ResolvedPlan` 的 variant 必须同时选定 schedule、ordered adapter revision、数学 recipe、全部 consumed targets 和 GPU／hybrid 能力。结构相容时可复用 `BackendGraph` 模板，但每个 `Invocation` 重绑实际 A/B owners、scale bits、output ranges、低秩 scratch 与 input generation；不用旧 Weights closure 或裸地址当作跨请求证据。模型本次运行的 adapter revision 必须等于 prepare 时 revision，改变 strength 或 source 后重新建立 invocation 和 prefix。

当前 GPU LoRA 的共享 low／gate／up 可行，是因为各投影在一个有序 native GPU 执行路径上消费，scratch 最后使用者已知。未来 W2 或 hybrid 分支不能直接继承这个复用结论：并发存活的低秩中间量需独立 slot／lease，交接到同一 CompletionDomain，预算进入原 NativePhaseMemory；BaseOnly 的资源表保持不变。新 GPU LoRA 数值、hybrid LoRA ABI、W2 窗口资格分别推进，不以一个开关同时开放。

## 3. 来源／重载问题与当前修复边界

### 3.1 native 重载与 `bind_lora` 的先后顺序

[pipeline.cpp](../../../native/models/qwen21/pipeline.cpp) 的原先顺序在清空／重载 Base 之前计算 `bind_lora`，可能因相同 `active_lora_identity_` 跳过新 Weights 的绑定。当前开发改动已令 native／layer-staged reload 强制重新绑定，并将 native active identity 更新移至 factor materialize、metadata 检查和 source revalidation 之后。

`61fe967c…` 阶段该分支仍受 BaseOnly gate 保护；`6d914980…` 后仅开放窄域显式候选，本次各路线是一个 fresh process。generated Base→LoRA→Base 已通过，但第二次相同 adapter 请求、持久 worker 切换、同路径替换和失败中断仍需独立回归：每次新 Weights 的全部 target 必须确实绑定；失败清空组件，不能用旧 active identity 证明新 source 已完成准备。

### 3.2 adapter 的 held-fd 与 produced owner

Base `Weights::load_lease` 已通过 [MlxLeaseFdReader](../../../native/backends/mlx_fd_reader.hpp) 和 `SourceLease::duplicate_fd()` 读取。新增 `apply_loras_lease` 已与原 pathname API 共用 parser，保留 lease／reader／A/B；pipeline 开发分支已用 `capture_verified` 在同一 held fd 上验 hash 并解析 adapter。原 `apply_loras` 仍可服务旧路径，不能把其 pathname 来源证明冒用给 native importer。

实现与后续 Session 接线须共同满足：

1. 打开有序 adapter 来源；从同一 held descriptor 验证所需 pinned digest，并保留身份。`SourceLease::capture` 的 stat snapshot 本身不是整文件内容 hash。
2. 用 duplicate fd 建 MLX reader，解析并绑定；reader 与 A/B 数组保留到 native 物理 drain。不可只保留一个 path／digest 字符串。
3. **在 adapter 绑定之后**调用 `Weights::materialize()`，它枚举 Base 与 runtime A/B。pipeline 当前开发改动已调整该顺序；仍需 generated 验证 importer 确实只接受 produced factors，不能由函数调用顺序推断 GPU 数值正确。
4. `mx::eval/synchronize`、source revalidate、`clear_cache` 后再采 NativePhase baseline。原始 Base＋adapter MLX storage 全在 baseline 中；native import 只 retain 真实 backing，不重复声称已由 native ledger 分配。
5. NativeSession 持有有序 adapter leases；step 前后、最终 publish 前及 drain 后按既有来源协议复核。异常隔离时 A/B、reader、lease、native wrapper 与 backend 一起保留。

当前 Session 逐 term 比对绑定时的 `source_strength` 与请求 strength 的原始 FP32 bits，不从 alpha 后的 scale 反推。Adapter revision 包含真实来源身份／内容证明、generation、顺序、原始 strength 和每个有效 scale 的 FP32 bits、resolved target 与 row range、rank、factor dtype、数学 recipe。Prefix K/V 是经过 LoRA 后的值，必须绑定该 revision；不能复用 Base prefix，也不能在 strength 改变后只更换投影参数而保留旧 prefix。

## 4. Native GPU 的数学合同

### 4.1 通用线性投影

正常 GPU oracle 是 [Weights::project](../../../native/backends/mlx.cpp) 576–615 行；切片 oracle 是 `apply_runtime_lora_slice` 764–827 行。令 `R` 为当前模型输出 dtype 的 BF16 rounding：

```text
y0 = R(X · Wᵀ)
for adapter j in resolved order:
    low_j   = FP32(X) · FP32(A_j)ᵀ
    delta_j = (low_j · FP32(B_j)ᵀ) * effective_scale_j
    y_j     = R(FP32(y_{j-1}) + delta_j)  // 只作用于 j 的 output range
y = y_last + BF16(bias)                 // 原模型 bias 运算边界，只加一次
```

不得改成先求所有 delta 的总和再 cast，不能让 rank 中间量提前变 BF16，也不能把 bias 留在当前 base NativeLinear 内提前相加。矩阵乘法 reduction、scale 和 add 的实际编译融合仍须用真实 MLX oracle 测试，不以纸面等式宣称 bit-identical。现有 native SiLU／norm 的已验证 BF16 recipe 不随 LoRA 改动。

当前实际接口摘录如下；adapter revision 属于上层 prepared model／invocation 身份，不在每个 kernel term 中重新推导：

```cpp
struct OrderedLoRATerm {
    MatrixView a, b;                 // b 已是 output_count × rank 的原 factor 切片
    float effective_scale;
    uint32_t first_output, output_count;
    size_t source_adapter_ordinal;   // provenance；vector 顺序决定实际算术顺序
};
// 仅摘录编码方法；prepare、geometry 和成员存储见 native_lora.hpp。
class NativeLoRA {
public:
    void encode_ordered_corrections(CommandGraph&, const MatrixView& input,
        std::span<const OrderedLoRATerm>, const MatrixView& base,
        const MatrixView& output, const MatrixView& low,
        std::optional<MatrixView> bias = {}) const;
    void encode_silu_up(CommandGraph&, const MatrixView& gate,
        const MatrixView& up, const MatrixView& hidden) const;
};
```

算子现已独立为 `native_lora.hpp/.mm` 和 [kernel／metadata contract](../../../native/backends/metal_graph/kernels/native_lora.hpp)，未给 DenseLinear 隐式增添 dtype／rank 假设。当前 DenseLinear 的 N/K 需要 64 对齐，input 必须 BF16，不能直接执行 FP32 low→B。新 LoRA bundle 使用通用 8×32 tile、FP32 FMA／low，支持 BF16 或 FP32 compact factors、rank≤128、K≤12288、N≤32768；r16/r128 的 generated device 数值已通过，真实 r128 窄域另完成一组配对，不代表所有形状均验证。prepare 一次建立 low／apply／copy／bias／SiLU-up 五个 PSO，不为每层或每个 rank 重新编译；Metal 4.0／32-lane／threadgroup 能力检查在本次设备路径通过。

当前两段执行为 `X×A→FP32 low`，再 `low×B→scale→有序 BF16 add`，第二段直接写 BF16 y，没有整张 `[M,N]` FP32 delta。允许 base/output exact logical in-place；其余 source／scratch overlap 被拒绝，所有 terms 先通过 shape／extent／range／alias 检查，再把私有 draft graph 原子追加到调用者图。空 terms 路线不准备或使用 LoRA PSO／scratch；有 bias 的 Base projection 仍走原 NativeLinear，不能借空 correction 路径重复加 bias。CPU／host 合同与 generated GPU Weights oracle 已通过；真实六步请求另获整图证据，不外推所有形状或 adapter。

### 4.2 FFN 必须在 SiLU 前加 gate/up LoRA

现有 [dense_ffn.hpp](../../../native/backends/metal_graph/kernels/dense_ffn.hpp) 112–151 行把 gate/up base matmul、BF16 cast、SiLU、hidden 相乘放在一个 kernel。不能在该 kernel 产生 hidden 后再加 gate/up delta。

```text
g = ordered_runtime_lora(X, W_gate, A_gate, B_gate)
u = ordered_runtime_lora(X, W_up,   A_up,   B_up)
h = BF16(BF16_SiLU(g) * u)
y = ordered_runtime_lora(h, W_down, A_down, B_down)
```

Gate、up 虽然都读 X，实际本地文件有**不同的 A**；不能仅按 input 相同就复用 low。复用键至少包含 source A 的真实 control-block owner、view offset/pitch、dtype、列范围和本次 input generation。仅当 A 确实相同才共享 `X×A`；同一 gate 的 GPU／ANE 输出行切片则可共享该 low。

当前已接线的纵向实现是：有 FFN adapter 的 block 用独立 NativeLinear gate/up＋ordered correction，经 `encode_silu_up` 后再 NativeLinear down＋down correction；无 adapter block 保留原 fused NativeFfn。NativeDenoiser 每请求只分配一套 gate/up BF16 workspace（M=1024 时共 48 MiB，实际按 prefill full rows），外加 `max(full_rows,2)×max_rank` FP32 low，在所有层与两个 phase 间复用；BaseOnly 不分配这些资源。真实配对表明当前 native LoRA 更慢；通用低秩 GEMM、拆分 FFN 与准备是待诊断候选，不能单凭源码分配比例归因。后续 tile epilogue fusion 可尝试减少 workspace，但必须保留全部 adapter 的顺序舍入。

Down 的 low 必须来自**修正后的 h**，不能来自 normalized X、未加 adapter 的 hidden、ANE down partial，或另一次纯 GPU 近似 FFN。`X×A_down` 的输入宽度是 12288，和 gate/up 的 4096 不同。

### 4.3 全局投影与 prefix

时间 linear1/2、modulation 的实际输入是两行：当前 sigma 和 0-time prefix。Viggle 的三个全局目标必须同时作用于两行；只处理当前 sigma 会改变首步 text/reference prefix K/V。`norm_out.linear` 在 native 中使用 target time row；`txt_in` 是 prefill 的条件文本通道；`img_in` 用于目标和参考图。无参考图的首个发布域也不能在通用 operator 实现中遗漏这些目标。

参考源：[transformer.cpp](../../../native/models/qwen21/transformer.cpp) 224–237、349–374、599–622、777–778 行。模型是 scale＋gate、无 shift 的 4H modulation；不要套其他 DiT 的六向 AdaLN 或 RMSNorm 替代 LayerNorm。

## 5. Hybrid LoRA：独立 ABI，不复用 BaseOnly 的输出解释

当前 NativeHybridFfn／PreparedAneStages 明确是 BaseOnly。Base W8 程序输出 `H+1` 行：normalized down 和 hidden scale；它没有能供 down LoRA 使用的 corrected hidden。当前 `plan_parallel` 只在 GPU 半区算 full FFN，再 FP32 partial join→BF16→gated residual。增加几个 delta 指针并不能使它支持完整 adapter。

建议分两阶段：

1. **GPU 全目标正确性先行。**LoRA Native GPU 可独立完成，hybrid LoRA 请求仍明确拒绝，不暗中让部分 layer 回到 MLX。
2. 新增 `RuntimeActivationLoRA` program/binding/phase recipe。优先一个 r128 adapter；BaseOnly 的 `H+1` ABI、surface 数、PSO 和无 adapter workspace 不变。新 ABI 的 header 明确 correction dtype／layout／headroom、hidden output 的含义、rank／输出范围和 rounding；程序 cache key 包含这些字段。

已有 [ane_mil.cpp](../../../native/backends/private/ane_mil.cpp) `w8_swiglu_program` 的 `lora_inputs=true` 增加 FP16 `dg/du`，SiLU 前相加，并把 `hsafe` 附加到输出 `H+1+A`。当前另加 `native_lora_program(Spec)`：保留旧 emitter，独立生成 Hidden／Rank ABI 和有限值 guard。两者**均未接入 native LoRA controller**。新候选仍使用 FP16 correction／FP16 SiLU；`bf16_value_boundaries` 继续拒绝 `lora_inputs`。它不是正常 GPU 的精确 oracle，也不继承已有 Base 的数值／设备资格。

首个兼容 ABI 可恢复 `hsafe * headroom` 到明确的 corrected-hidden 表示，再比较正常 GPU 与兼容 hybrid 两个 oracle；是否选择 BF16 restore、在哪个边界 cast，要写进新 recipe。不能忽略 headroom；不能把 quantized/rotated hidden 当原空间 hidden 用于 down A。更准确的 BF16 边界 emitter 是另一个候选，不在未经组件证明前替换旧 recipe。

### 5.1 图与 buffer 布局

```text
before_ffn producer:
    attention + norm
    low_gate = X × A_gateᵀ; low_up = X × A_upᵀ
    ANE-channel gate/up corrections → 新 dg/du surfaces
    W8/A8 staging → physical producer admission
submit ANE
gpu_ffn branch:
    同一 low 的 GPU B-row 部分 + BF16 base gate/up → h_G
    h_G × Wdown_Gᵀ → FP32 partial_G
consumer（获得真实 ANE read permit 后）:
    restore partial_A + corrected hidden h_A
    BF16(partial_G + partial_A) → base down output
    ordered down LoRA(corrected h) → bias → gated residual
```

现有 [native_denoiser.cpp](../../../native/models/qwen21/native_denoiser.cpp) 把 GPU hidden 绑定到完整 `[M,F]` arena 的前半列，保留 F pitch。可将 ANE restored hidden 写入该 arena 后半列，直接构成完整 corrected h，**不另建 concatenate tensor**。两个写入子范围必须在跨图 alias 合同中证明不重叠，down LoRA 在两边物理完成后读取。需要 [16 §4.2 的 StridedRows](16-splash-native-backend-spec.md#42-从模型算子到实际-command-buffer) 或等价逐行区间；两个包围 ByteRange 实际重叠，不能用不同 wrapper 假装不同 backing。该方案首先减少拼接分配；读取 down A 仍是完整宽度 GEMM。

下一步才比较分片低秩：`low_down = h_G A_Gᵀ + h_A A_Aᵀ`，FP32 join 后只执行一次 B；不把两个 BF16 down-LoRA 输出相加。由于 reduction 次序改变，这依然需要新数值验证。把 `h_A A_Aᵀ` 放到 ANE 并只输出 rank 行，会降低 output traffic，但引入 A 的 program 输入、额外计算与精度边界，不能作为首版必需优化。

M=1024、A=6144 时新 dg/du FP16 payload 合计 24 MiB，packed hidden 输出比 Base 多约 12 MiB（不含 pitch、allocator rounding）。rank128 的每份 FP32 low 约 0.5 MiB。这些都进入同一 NativePhase manifest、真实 surface hooks 与 owner 生命周期，不能借“共享内存”从预算中删掉。BaseOnly 不创建这些 buffer；当只有 attention/global LoRA、FFN 无 adapter 时，也无需换 FFN ABI。

### 5.2 cache 和完成域

Runtime activation correction 不修改 Base W，所以现有 request-local W row scales 可以继续按真实 Base backing＋view＋recipe 缓存；它不是 adapter output cache。adapter low／delta 必须按本层、本次 X、adapter identity 管理，不能跨步复用。若改为 merged effective W，必须使用新的有效权重 owner／revision／scale-cache key；旧 Base scale 绝不可命中。

新增 correction producer、ANE output hidden restore、down LoRA 都属于同一个 step deadline 和 completion domain。consumer 的有限值校验完成后才可发布本层结果；全部层／Euler／来源校验成功后才提交 latent/prefix。失败仍 retire 全 run，所有新 workspace／surface／A/B owner 保留到真实 drain；清理或人工释放事件不能成为读取 hidden 的许可。

### 5.3 后续最小实施包：HiddenReference 与 Rank128Candidate

**本节的 Spec／MIL emitter／low 拆分已实现，stages、controller 和 hybrid 请求仍待实现。** 当前 Base program、九个 surfaces、stages、预算和请求行为不变；不能据此打开 App／自动选择／native hybrid LoRA 路由。首次控制器实施沿用 [16 的 W1S](16-splash-native-backend-spec.md#61-当前-w1s保留作正确性与收益基线)：单 bank、逐层主机准入、同一步绝对 deadline，先提交 ANE 再提交 GPU 补集，真实完成后消费；它不实现 W1R／W2 或跨层设备窗口。先完成 `HiddenReference` 的组件闭环，再比较 `Rank128Candidate`，不把后者作为前者的前置条件。paired gate/up 筛选已经独立得到 NoGain，不混入本包的 hybrid 比较。

首个真实范围固定为已有单个 Viggle r128、strength=1、M=1024／H=4096／F=12288、GPU 与 ANE 各 C=6144 通道、headroom=1、无参考图。以下 `A_g/A_u/A_d` 是 LoRA 因子，`h_A` 的下标 A 指 ANE 通道；两者不要混淆。全局／attention LoRA 继续走现有 native GPU 全目标实现。未来整图仍首步完整 GPU、后五步 hybrid，不能在组件尚未证明正确前放开 Session／route。

#### ABI 与编码接口

已实现 [native_ane_lora_contract.hpp](../../../native/backends/metal_graph/native_ane_lora_contract.hpp) 的不可变 `Spec::make(Config)` 与 `Base / HiddenReference / Rank128Candidate` 枚举。Spec 返回精确 inputs／surfaces／packed sections、pitch、capacity upper、headroom、rank、writer masks、recipe 和 `abi_fingerprint()`。后者是无损规范文本，**不是内容 hash**；program cache 仍需连 MIL／constants／环境一起哈希，动态 factor 内容属于 binding。真实域固定 M1024／H4096／F12288／C6144、r128／一个 strength1 adapter／无 bias／headroom1；tiny M32／H128／F1024／C512 的 Hidden 另可 r16，Rank 只 r128。

`tc::ane::private_api::native_lora_program(spec)` 已生成两个 LoRA recipe，动态 `dg/du` 和 Rank 的 `ad` 不嵌入常量。Base 调用保持旧 emitter 路径；四组旧 MIL／rotation constants 的逐字节比较通过。新 MIL 的 dtype／shape／pitch 与 Spec、所有 finite token 的最终聚合均有 CPU 检查；**没有编译新 ANE program 或运行 ANE**。

下面前半段是**现行 NativeLoRA API 的示意调用**，后半段仍为拟议 stages：

```cpp
// op 是已 prepare 的 NativeLoRA；完整投影验证后才追加一个 dispatch。
auto full = op.validate_projection(input_bf16, full_term, output_bf16, low_fp32);
op.encode_low(producer_graph, full);
auto gpu_slice = op.validate_projection(input_bf16, gpu_b_row_term, output_bf16, low_fp32);
// 调用方还须证明 A/input/view/LowParameters/variant 相同及 low 已就绪。
// output_bf16 已含本投影的 BF16 base/上一 adapter 值；不是单独 delta 输出。
op.encode_apply_inplace(gpu_graph, gpu_slice);

// 以下仍待实现：与 recipe-specific writer mask、source/slot generation 一起验证。
encode_lora_delta_surface(graph, low_fp32, b_channel_slice, scale, dg_or_du, status);
encode_lora_down_a(graph, a_down_channel_slice, ad_surface, status); // 仅 Rank
encode_restore_hidden(graph, read_permit, producer_status, hidden_window, restore_status);
encode_restore_rank_join(graph, read_permit, producer_status, low_g_inout, restore_status);
```

`ValidatedLoRAProjection` 保留实际 views／owners、完整项的 Scalar／SG 选择及 LowParameters，不能靠 A 的形状单独选择 SG。它不证明 low 的内容属于本次 X／A，也不授跨图 readiness；后续 controller 必须绑定 source／slot generation 和完成域。完整项与 channel slice 要显式确认同一个 low 数学，切片若改变 SG／Scalar 选择不得共享。gate/up 的 A 不同，因此需要各自的 low；同一投影的 GPU／ANE B 行切片可共享对应 low。`encode_ordered_corrections` 保持原行为，直接重复调用仍会重复算 low。新增 delta surface writer 采用 FP32 B 累计后转换 FP16，它不是 `encode_apply_inplace` 的 ordered BF16 correction，仍需独立实现与 recipe。

[NativeAneHandoff](../../../native/backends/metal_graph/native_ane_handoff.hpp) 的通用输入／输出、sealed producer/read permit 和 owner 保留可复用；[NativeHybridFfn](../../../native/backends/metal_graph/native_hybrid_ffn.mm) 的七输入／H+1 断言、[stage contract](../../../native/backends/metal_graph/native_ane_stage_contract.hpp) 的固定 writer mask 与 restore 布局则需新增独立分支。不能在 native controller 中套一个 legacy `PrivateW8Graph`，从而另建队列、执行域或账本。`NativeFfn::encode_down` 可复用 corrected GPU hidden→FP32 partial；其 fused `encode_gate_up` 已执行 SiLU，不能用于需要先修正 gate/up 的分支。

#### 精确数据流与两个精度 recipe

```text
producer（两个候选共用）:
  native attention + norm → X BF16
  low_g = FP32(X × A_gᵀ); low_u = FP32(X × A_uᵀ)
  dg = FP16(scale_g × FP32(low_g × B_g,ANEᵀ))          [C,M] 转置 surface
  du = FP16(scale_u × FP32(low_u × B_u,ANEᵀ))          [C,M] 转置 surface
  Base W8/A8 staging；Rank 另将 A_d[:,C:F] stage 为 FP16 [128,C]
  所有 producer writes 完成且 finite/status 验证通过 → sealed producer permit
submit ANE，然后提交 GPU complement:
  GPU BF16 base gate/up → 各自有序 correction → BF16 SiLU×up → h_G BF16
  partial_G = FP32(h_G × Wdown_Gᵀ)
ANE（两个候选共用的 FFN）:
  W8/A8 gate/up + dg/du → FP16 corrected gate/up → FP16 hsafe
  hsafe → hidden rotation/A8 → normalized down partial + hscale
```

这里的 FP32 矩阵乘法仍绑定实际 scalar／SGMMA kernel variant 与 reduction recipe，不能仅由 dtype 名称推断逐位一致。GPU base gate/up 在修正前先形成 BF16 值；B 应用后按 adapter 顺序再形成 BF16 值，最后执行现有 BF16 SiLU/up 边界。不能把 gate/up delta 加到已相乘的 hidden。

| 已有 Spec／emitter recipe，consumer 待接 | ANE 输出及新增输入 | consumer 的数学与精度边界 |
|---|---|---|
| `native-hybrid-lora-hidden-fp16-correction-bf16-restore-v1`（HiddenReference） | 输出 `y=[down_norm(H);hscale(1);hsafe(C)]`，输入在 Base 七项上增加 `dg/du` | sealed read permit 后恢复 `partial_A FP32`；`h_A=BF16(headroom×FP32(hsafe))` 写入现有 `[M,F]` hidden arena 后半列；`base=BF16(partial_G+partial_A)`；完整 `[h_G,h_A]×A_dᵀ` 在 GPU 得到 FP32 low，随后一次 Bdown／scale／有序 BF16 加法，再 bias／gated residual |
| `native-hybrid-lora-rank128-fp16-correction-fp16-low-v1`（Rank128Candidate） | 再增加 `ad FP16[128,C]`；在 **hsafe 的 rotation/A8 之前**算 `ad×hsafe`，输出 `y=[down_norm(H);hscale(1);low_A_norm(128)]` | GPU branch 另算 `low_G=FP32(h_G×A_d,Gᵀ)`；permit 后恢复 FP16 low_A 到 FP32并加到 low_G；base partial 仍只 join／BF16 一次；对合并的 low 只执行一次 Bdown／scale／BF16 加法，再 bias／residual |

这些字符串已由 Spec 返回并进入规范 ABI 身份，尚未进入模型 serializer／资格表。Rank 的 `ad` 限定 BF16 factors 转 FP16；不可表示或非有限值必须在未来 writer 拒绝，不静默 clipping。emitter 已按 K≤2048 固定次序分块，用 FP16 partial／sum，并逐项检查有限值；tiny 为一块，真实 C6144 为三块。它不会经过 HiddenReference 的 BF16 hidden restore，因此同时改变 factor carrier、hidden 边界与累计精度；**FP16 low_A＋FP32 low_G 不是现有 GPU FP32 low 的等价替换**。未来 low_A restore 使用自己的 headroom／明确 factor scale；`sd` 和 `hscale` 只用于 Base down partial，不能误乘到 low_A。若后续引入 factor normalization，需要另记尺度和 recipe，不在此 v1 中隐含。

Rank 不得从 quantized `hq`、rotated hidden、normalized X 或 ANE 最终 down partial 计算／反推低秩结果。也不能分别形成 GPU／ANE 的 BF16 down-LoRA 输出再相加：应先在 FP32 中合并 low，Bdown 只应用一次。HiddenReference 的完整 hidden 写入保留 F pitch、列范围和原 backing；两半写窗口须证明不相交，最后 consumer 读取整个 hidden，无额外 concatenate tensor。

#### 提交范围、dispatch 与预算

按现有独立 gate/up base GEMM、无 bias 的首个 FFN、不开 scale-cache copy 计，payload dispatch 如下。该表**排除**共同 attention／norm、现有末尾 reset＋三个 finite 扫描、status 初始化／callback 和 MIL 内部算子；新 delta／factor writer、hidden／rank restore 应在本 dispatch 内写入 finite flags，不能以数量表省略校验。额外校验节点若实现时无法融合，应在真实 receipt 加计。

| 候选 | Producer | GPU branch | Consumer |
|---|---:|---:|---:|
| HiddenReference | 12：W staging 6＋A8 staging 2＋两个 low＋两个 delta surface | 6：两个 base GEMM＋两个 correction apply＋SiLU/up＋down partial | 6：partial restore＋hidden restore＋base join＋完整 down low＋B apply＋residual |
| Rank128Candidate | 13：前述 12＋Adown staging | 7：前述 6＋GPU 通道 down low | 5：partial restore＋rank restore/join＋base join＋B apply＋residual |

这里假定复用现有二遍 weight staging；若后续 paired gate/up 被单独验收，两候选 GPU branch 可各少一次 dispatch，但不能把那个收益归给 ANE ABI。ANE 提交后 GPU branch 才提交；consumer 只有同时获得真实 ANE read permit 和 GPU branch 物理完成后才可提交，仍是 W1S。

生产几何下当前九个 Base surfaces 的 **shape payload 是 85.00390625 MiB，逐 site 16 KiB capacity upper 是 85.03125 MiB**。tx 和 y 的页取整造成两者不同；它们都不是整体 phase／进程峰值。相对这些 Base surfaces 的新增预算为：

| 增量项 | HiddenReference | Rank128Candidate |
|---|---:|---:|
| dg＋du FP16 surfaces | 24 MiB | 24 MiB |
| y 新增输出行 | 12 MiB hidden | 0.25 MiB rank |
| Adown FP16 input surface | 0 | 1.5 MiB |
| 合计新增 surface payload | **36 MiB** | **25.75 MiB** |

以上新增项在本几何均已满足 64-byte pitch 和 16 KiB 对齐；原始 Base 上界仍保留。Rank 比 hidden 导出净少 **10.25 MiB**，不是只减去输出差额的 11.75 MiB，也不能由字节差直接推断耗时。

现有 full-row LoRA gate/up、完整 hidden arena 和一个 FP32 low workspace 继续按真实 prefill 上界计费。仅需再增加一个 target `[1024,128] FP32` low，**0.5 MiB**：producer 的 low0／low1 分别存 gate/up；GPU B 应用全部完成后，low0 可复用为 low_G；Rank consumer 直接从 y 恢复并加到 low0，不另分配 low_A FP32矩阵。HiddenReference 在两分支结束后也复用 low0 算完整 down low。attention／后层不得在旧 low 尚有 reader 时提前复用。所有新增 labels、surfaces 与实际 capacity 进入原 `NativePhaseMemory`／`MemoryBridge`，不增加第二账本；source baseline、driver/compiler guard、两个已有 FP32 partial 和状态页继续计入，不能把这里的增量当成完整请求上界。

#### 来源、校准与失败合同

正常 oracle 必须使用 [runtime_ffn::full](../../../native/models/qwen21/runtime_ffn_graphs.hpp) 与同一份实际 `Weights::project`，保持真正的 ordered BF16 adapter 边界。`corrections/down_add` 汇总 delta 后转换、再加 Base，连单 adapter 也可能增加舍入，不能拿它们冒充 normal oracle。两候选分别报告对正常 oracle 的完整 FFN 误差；另报告 Rank 对 HiddenReference 的额外误差，以及 down-low／down-output 的中间误差。阈值在运行前固定，不能看到结果后将 §7 的宽筛选范围套给所有投影或取消 finite／source／物理门槛。

真实输入优先复用已有 [LoRA capture provenance](../../../outputs/qwen-real-ffn-20261007/lora-captures/75991-1226054446574458-replay/complete.json) 对应的 layer0／step5，后续需要时再考虑同源 step1／3；Base 的 step1／10／19 捕获只能解释 Base 误差，不能替代 adapter 校准。历史 capture 不自动提供当前 held-FD 证明或整图资格。§12 的有限回放仅物化该单层 source／input，用于 GPU 对照；没有进行 hybrid LoRA 回放。

实际回放沿用 Base `SourceLease` 和 adapter `capture_verified` 同 FD pin；prepared bindings 持有真实 MLX Data control block、factor offset／pitch／范围、adapter revision／顺序／effective scale，不能只保留路径、裸地址或 wrapper。program cache key 包含上述 ABI／geometry／rank／recipe，具体 base/adapter 内容属于 binding 身份；动态 Adown staging 不要求每层重新编译。基于真实 Base W 的尺度缓存只有在原 source／view／quantization recipe 都匹配时才可能复用；首个比较先关闭该 cache，不缓存跨 step 的 low／delta。

fresh status 的 recipe-specific mask 必须覆盖 weight、activation、两个 correction writer，以及 Rank 的 Adown writer；Base 的固定 mask 不能放宽成“包含某几位即通过”。Spec 已声明 Base producer／restore 为 15／16，Hidden 为 111／272，Rank 为 239／528，**当前 AneStageStatus 尚不接受新 masks**。未来 restore status 同时绑定原 producer、program／slot generation、精确 y 行段及 captured scale。输出和 narrowing 前的模型值必须检查有限值，防止 quantization／cast 掩盖坏值。新 emitter 已在 corrected gate/up／hsafe、rotation 前后、hscale_positive、normalized down 和 Rank 每个 partial／sum 聚合 guard，包含合法最大 FP16 值 65504；失败输出 hscale=-1。GPU writer、sealed restore 和 status 仍需实现，MIL 哨兵不是物理 read permit。旧 emitter 没有被改写成这个新资格。记录前的 metadata／alias 失败零提交；提交后失败永久 retire invocation，不发布 latent／prefix。

一个 job 强持有 plan、factor/source owners、surfaces、status 和实际 tickets，直到 producer、ANE、GPU complement、consumer 全部物理完成。timeout 或逻辑 retirement 不授 read permit／bank reuse；未知物理完成时保留并隔离全部 owner，不能先覆盖单 bank 或启动原地 GPU fallback。即使已收到 physical receipt，callback 仍持有的最后 owner 和 allocation claim 也不能提前清零；释放／reap 遵守 [16 的同一生命周期合同](16-splash-native-backend-spec.md#7-准备缓存与内存统一管理)。同一步 deadline 只限制准入、等待和发布，不能硬中断阻塞 driver 调用。

#### 可直接拆分的实施工作包

1. **metadata／emitter 基础已落地。** Spec、input/output、writer masks、budget、rank／范围／pitch／generation 合同与 MIL CPU 检查通过，Base emitted bytes 不变。SourceView／AdapterBinding 是 metadata 校验，不验证 held FD；Spec 不创建 Surface／permit。跨图旧 permit、late owner、timeout／retired 和真实 slot 复用仍需在 stages/controller 中验证。
2. **完成组件闭环。** NativeLoRA validated low／apply 已拆分并通过 tiny 同图检查；下一步 PreparedAneStages 实现 delta、Adown、hidden／rank restore，controller 按显式 spec 分配和提交。先完成 HiddenReference，再接 Rank。复用现有 handoff、down FP32 kernel、统一 phase hooks，不引入 MLX 热路径或第二执行器。生成数据先检查边界和舍入；后续真实组件记录完整 norm→producer→ANE/GPU→down-LoRA→residual wall，而非只比 rank kernel。
3. **组件退出后才接模型／Session。** NativeBlock 将 adapter 分支明确接入 before_ffn／GPU／consumer；Denoiser 才分配新增 low／surfaces，扩展 schedule、manifest 和实际成功计数。首步 GPU、后五步共 160 FFN；227 target、adapter 来源、全部 physical users 与阶段账本释放都要通过。不能通过删除当前 `has_terms && hybrid_ffn`、GPU-only source 或 memory gate 来假装集成完成。
4. **最后单独决定整图与产品准入。** 只有完整组件路径有收益、数值与故障合同成立，才考虑有限的一组六步 native GPU／native hybrid 配对；新回执绑定同一构建和实际 recipe。§12 的两组轻量设备工作不包含 ANE 或整图，不开放 App／自动 route；硬件 INT8 placement、public／fixed Core ML 兼容及 W2 性能继续独立验收。

## 6. 可以复用什么，不能直接复用什么

| 既有实现 | 可复用 | 不可直接作为新路线证明 |
|---|---|---|
| `Weights::apply_loras[_lease]`、snapshot、`project`、`apply_runtime_lora_slice` | 共用命名／目标 parser、effective scale、ordered snapshot、held-fd reader 与正常 GPU oracle | pathname snapshot 不能冒充 verified lease；snapshot／CPU 结果不证明 produced readiness 或实际 GPU 算术 |
| `Weights::LoRAWorkspace` | 一次 FFN 内按 X/A/范围身份共享 low 的思想 | 其 MLX id／Tensor cache 不可直接成为 native backing readiness |
| [runtime_ffn_graphs.hpp](../../../native/models/qwen21/runtime_ffn_graphs.hpp) | full／channels／corrections／down_add 的职责拆分与测试输入 | `lora_delta_slice` 汇总 delta 后 cast，`down_add` 再加 Base，和正常多 adapter 有序舍入不同；连单 adapter 也多出 delta cast 边界 |
| private W8 emitter 的 `lora_inputs` | correction 输入及 hidden 输出布局、分块 MIL 组织 | FP16 correction/headroom/SiLU 与 native BF16 GPU recipe 不相同；当前 native stages 尚无此 ABI |
| vpipe pinned runtime LoRA／ANE staging（见 [16](16-splash-native-backend-spec.md)） | batch target、有效 rank、scratch 复用和 staged weight 组织 | 它的 staging-time weight merge 不是 activation correction；BF16／FP16 merged W 的舍入不能继承本地 runtime oracle |

不建议第一版预先 `B×A` 并产生完整 merged checkpoint：r128 factors 本地已约 648 MiB，Base DiT 很大，32 层都保留 merged dense bank 会显著扩大常驻内存与磁盘。若以后做 tile staging-time merge，应明确 FP32 merge→目标权重 dtype→quantization 的新 recipe，记录 adapter identity、准备成本与 scales 失效；组件证明完整路径收益后再考虑缓存，禁止默认导出整模型副本。

## 7. 分阶段实施与验证预算

当前已有的[模型 host／CPU 记录](../../../outputs/native-lora-model-20261008/host-cpu.log)包含 public/private 模型源码严格 host 检查，以及 schedule／adapter identity 合同的 12 个接受、32 个拒绝案例；记录明确 devices=0、models=0。kernel、source binding 与 importer 的 CPU／host 检查分别验证其边界；其后一次 generated GPU 记录见下节，仍不证明真实模型全请求成功。以下阶段仍按各自证据退出，不能以更多样本掩盖语义差异。

1. **完成并冻结 source／request 合同。**共用 fd loader／snapshot／importer 已有；补齐 Session leases、adapter revision、actual-rank phase、精确 variant／schedule 和 pipeline reload 的整体检查。继续测试两种 A/B 后缀、alpha scalar／默认比例、fused range、未知／遗漏 target、有序 adapter、同 path 替换、失败清空和第二次 native reload 必须重绑；request gate 在 generated 验证前保持关闭。
2. **一组 generated GPU 组件。**用真实 `Weights::project`／`apply_runtime_lora_slice` 作 oracle，覆盖 r16/r128、M=1/2/非整 tile、K=64/256/H/F、N=64/H/F/4H、有 bias 和有序多 adapter。包括两个 noncommuting BF16 rounding fixture、distinct gate/up A、负/零 strength、输入与 factor/padding 不变、invalid alias 零提交。BaseOnly buffer/PSO/dispatch 计数不增加。CPU 近似 GEMM 只测合同，不能代替 GPU oracle。
3. **generated 完整 DiT。**同一已有 MLX Transformer tiny 拓扑，对全部 8 global＋每层 7 logical target 建覆盖清单；分别触发 global 与 FFN target，避免全部同时开启后误差相互抵消。检查 prefix/steady、两行 time、最终 Euler、revision 失配、完整 drain；此时不必加载真实模型。
4. **至多一组真实 GPU LoRA 配对。**在前述通过后，用本地 Viggle r128 strength1、同 seed/prompt、512²六步比较正常 MLX runtime LoRA 与 native GPU。实际六步 schedule由原 pipeline传入；不是把20步Base缩短。记录准备/请求/去噪、peak、source provenance、图像目视与数值。r16编辑 adapter不因此获得edit/native资格。
5. **另一个阶段的 hybrid 组件，再决定 E2E。**优先已有真实 FFN输入回放一层/一两个时刻，比较全GPU normal oracle与同recipe hybrid oracle，独立统计 producer、完整wall及内存；只有完整路径有收益且错误边界清楚，才跑最多一组六步nativeGPU/nativeHybrid整图对照。新cache/ABI/recipe不得引用旧Base配对当新收益。

用户允许的 5–8% 数值差异是候选筛选范围，不能替代 finite、source、alias、生命周期的严格要求，也不能当所有投影自动合格的阈值；最终同 seed 图片仍需目视。GPU runtime LoRA 第一阶段应努力接近正常 GPU oracle；若 kernel 与 compiler 产生新差异，先定位中间边界，不直接用 8% 放过。Hybrid 的 FP16/W8A8 近似则单独报告。

首个 native LoRA hybrid 若保留首步完整 GPU、六步 schedule，成功计数应为 **32×5=160 个 hybrid FFN、5 个 hybrid step**。这是拟定合同，不是已测结果；现有 Base 的 608/19 断言必须按经验证的 variant 推导，不能简单删除，也不能误用到六步 LoRA。

完成第一轮纵向改造的标准是：同一实际 adapter 的全部 target 都消费，来源和 schedule 正确，全 DiT 无 MLX 回落，BaseOnly 没有无用 scratch，数值 oracle 与事务／内存合同一起通过。之后再优化 kernel fusion、weight staging、窗口并行与 prepared-session 复用。

## 8. Generated 阶段证据与后续真实请求

较早库 `61fe967c…` 完整编译通过，567 个构建源摘要一致。该阶段 NativeSession source／request 接线、实际 scratch phase 和 pipeline 重载顺序已冻结；当时外层 route 继续 BaseOnly，CLI／App 显式 Native LoRA 请求仍拒绝。

该 generated 阶段只做一组 GPU：算子 3 提交／14 dispatch、534 个 BF16 输出与实际 Weights oracle 零位差；完整 tiny H128／F256／2 层、3 文本＋4 目标行、两步，22 个目标混合 r16/r128。Base→LoRA→Base 三路最终 latent RRMS=0，LoRA 确实改变输出，恢复 Base 逐位相同；三个 phase 物理完成且新增存储账本归零，LoRA scratch 10,752 B，Base 为零。来源／padding 不变，失败 alias 不追加图。完整目标同时开启，没有独立逐目标消融；大几何或真实 32 层不得继承逐位结论。

该阶段未加载真实模型、运行 ANE 或生成整图，自有编译及生成 tensor 已清理，保留库和[历史回执](../../status/2026-10-08-native-lora-gpu-components.md)。随后 `6d914980…` 只增加一次真实 GPU LoRA 配对：195 physical bindings／227 logical targets、六步、source pin／revision、52,047,872 B scratch 和阶段释放通过，图片 RGB 相对 L2 1.035%；request 慢 21.48%，故不授性能资格。完整记录见[真实状态](../../status/2026-10-08-native-lora-gpu-e2e.md)。§7 的逐目标覆盖、持久请求回归、hybrid correction／hidden ABI、W1R／W2 继续独立推进。

## 9. Opt-in FP32 低秩矩阵乘法与独立筛选

由上一构建负性能结果提出的候选现已独立实现并在 `77863b3f…` 构建编译。默认仍为 scalar；严格 GPU Viggle 窄域可显式请求 `qwen21_native_lora_kernel="sgmma"`，schema 1 在顶层、schema 2 在 execution。新 variant 保留数据流／8×32 tile／128 threads／因子精确解码，只替换 low 和 apply 的归约；未同时改变大 tile、FFN fusion、ANE ABI 或 cache。完整构建及 13 项请求测试通过；组件结果见[本轮筛选](../../status/2026-10-08-native-lora-sgmma-screen.md)，不能继承 `6d914980…` 的 E2E 或其他路线资格。

保留的 scalar kernel 每线程仅两个输出，K32 tile 带两次 threadgroup barrier。H4096 的 low 需 128 个 K tile，down F12288 需 384 个；M1024／r128／N12288 的 B 应用每次 49,152 个 threadgroup。这些是源码几何事实，不是设备计时归因。现有 [NativeAttention](../../../native/backends/metal_graph/kernels/dense_attention.hpp) 已使用 FP32 8×8 matrix fragment；[NativeLinear](../../../native/backends/metal_graph/kernels/dense_linear.hpp) 仍限定 BF16 输入／权重，不能直接拿来处理 FP32 low 或 FP32 factors。

当前 opt-in variant 用四个 SIMD group 各计算一个 8×8 列块。shared tiles 仍为精确 FP32：BF16 factors 解码到 FP32，FP32 factors 不窄化；一个 K32 tile 内做四个 K8 矩阵乘加。概念流程如下，名称为伪代码，不是已编译的 Metal API：

```cpp
// 候选：一个 threadgroup 计算 [8 rows, 32 columns]。
acc = fp32_matrix_8x8_zero();
for (k32 : reduction_tiles) {
    stage_exact_fp32_input_and_factor_with_zero_mask();
    threadgroup_barrier();
    for (k8 : four_subtiles) {
        a = load_fp32_matrix_8x8(input_tile, k8);
        b = load_fp32_matrix_8x8(factor_tile, k8, simd_group_column);
        acc = fp32_matrix_multiply_accumulate(a, b, acc);
    }
    threadgroup_barrier();
}
// low: 写 FP32 scratch；apply: 独立 scale、读原 BF16、逐 adapter BF16 add。
store_with_existing_range_and_alias_contract(acc);
```

首次只选择大 M／对齐 r128 域；M1／2、r16、未支持的边界保留 scalar。必须 uniform 选择 SIMD 路径，不能在同一 group 的部分线程跳过 MMA 或 barrier。保持现有 masking、extent、owner／alias、提交前原子校验和 compact factor 布局，不展开整张 FP32 factors，不新增完整 `[M,N]` delta。

矩阵内部归约次序会变化，实际新 recipe 为 `runtime-lora-fp32-sgmma8x8-aligned-r128-scalar-tail-ordered-bf16-bias-last-v2`。实际 bundle 选择、CanonicalEncoder／adapter revision、prefix identity 和 result receipt 已绑定该 recipe；Session 检查 prepared model 的实际 recipe 后才报告，不能只使用计划字段证明执行。新资格独立，不沿用旧 `tiled8x32-fp32-fma…v1`。能力不成立时，显式新候选应拒绝；该 recipe 内已定义的小行／尾部 scalar 分支则如实报告为组合 recipe。

本轮固定 H→r128→F 与 F→r128→H 两个 generated correction（含一次 warm AB、一次 record AB／BA），再加小型两 adapter FP32 精度／有序 BF16 trap，共 14 提交。low／output 数值、来源、padding、physical completion 与账本释放通过；correction 耗时随顺序反转，不能宣称稳定提速。它们含 supplied base、low 与 apply 完整修正成本，不含 dense base GEMM，未分开计时 low／apply。

随后只复用已有真实 LoRA layer0／step5 输入，完整 FFN 最多六次。warm 后 AB wall 54.993→51.003 ms，BA 54.341→52.392 ms，分别省时 7.26%／3.59%；两 kernel 输出逐位相同，对实际 MLX FFN oracle 的 RRMS 均约 9.94e-7。6 次物理完成、来源／padding 与同 phase 释放通过。本次只有一层一个输入，不授全 DiT 数值或性能资格；随后只做一组同 seed 六步 MLX→SGMMA，227 targets／v2 recipe／来源／物理释放通过，图片 RGB 相对 L2 1.035%、肉眼接近。完整请求慢 14.44%、去噪慢 17.24%，仍不授性能资格；见[整图记录](../../status/2026-10-08-native-lora-sgmma-e2e.md)。32×32 low／32×64 apply tuning、gate/up fusion 与 hybrid ABI 留到独立候选，不追加当前模型实验。

FFN fusion 分为两个独立候选：paired gate/up base 投影分别写 BF16，仍保留全部 ordered correction；另一个候选把真正末项、覆盖完整 up 输出的 correction 与 SiLU×up 合并。前者不减少 gate/up 中间张量流量，后者才可能省去末项 up 的写入／再读取。下节记录前者的实际负性能结果，不能将两个候选合称已实现的 fusion 收益。

## 10. Paired base 原语：实现通过，当前输入没有净收益

新增 [NativeGateUp](../../../native/backends/metal_graph/native_gate_up.hpp) 和 [独立 shader](../../../native/backends/metal_graph/kernels/dense_gate_up.hpp)，用一个 prepared MPP PSO 同时执行两个 base matmul，分别形成 BF16 gate/up，再执行现有 SGv2 有序 LoRA。H／C 在准备期确定，M、两个 source/output pitch 和 first channel 在编码时校验；保留原 fused checkpoint 两个 view 的 offset／stride，不复制权重。它不是完整 FFN，也不包含 SiLU／bias。源和实际 device／alias 全部验证后，单个 dispatch 在私有 draft 中记录，再原子追加。

该原语只增加编译进库的可调用组件。原 `NativeFfn`、`NativeLoRA`、`NativeBlock`、Session／request／App 未切换；默认运行不创建额外 PSO 或 scratch。CPU contract 122 项、source 33 项、严格 host 编译通过；private package 库 `e6ea1909…` 完整编译，570 个身份源摘要匹配。不同于旧 SGv2 构建，新增 paired base 拥有 recipe `dense-bf16-mpp32x128-paired-base-separate-bf16-gate-up-v1`，不能继承旧整图数值或性能资格。

仅回放已有 layer0／step5 LoRA 输入，使用同一 held source、实际 MLX Data owner、同一 SGv2／down 实例和六个 scratch；一次 warm AB、一次记录 AB／BA，合计六次完整 FFN。两 arm 都先完成 base gate/up，再执行同一八个 correction／SiLU／down dispatch；A 用两个 NativeLinear，B 用一个 NativeGateUp，因此完整图为 10／9 dispatch。旧 SGv2 replay 在 base 与 correction 间交错，不能与旧耗时混算。

| 暖机后顺序 | Split 完整 FFN | Paired 完整 FFN | Paired 相对耗时 |
|---|---:|---:|---:|
| AB | 51.830750 ms | 51.881042 ms | +0.0970% |
| BA | 51.591916 ms | 51.885500 ms | +0.5691% |

所有对的两个输出逐位相同；对真实 `runtime_ffn::full` oracle 的 RRMS 为 9.935683e-7（4,194,304 个 BF16 元素中有 95 个位模式不同）。来源／padding 不变、全部物理完成、新增 phase 账本释放归零。源 payload 308 MiB，六份 scratch 88.875 MiB；准备单独计时，paired PSO 在本次进程耗时 203.25 ms，其他已命中系统／driver cache 的 PSO 不构成公平冷启动对照。MLX oracle 的首次执行也不是暖机 MLX 性能基线。

结论是**本输入 NoGain**：少一次 dispatch 和一个保守 buffer barrier 没有降低完整 wall。双 accumulator 压力、内存和 GEMM 时间仍是待测解释，不能由源码或这个结果认定主瓶颈。保持组件用于后续明确 recipe，不接模型默认路线，不追加当前候选 E2E；owned 编译缓存已清理，原模型／capture／库和回执保留。完整来源及样本限制见[本轮筛选](../../status/2026-10-08-native-lora-paired-base-screen.md)。

## 11. 下一候选：末项 up correction 与 SiLU 的明确边界

本节为**后续设计，尚未实现或验证**。先补充暖机 MLX／Native 相同完整 FFN 范围的对照，避免用 oracle 冷准备时间解释整图差距；§12 的首次尝试受 framework guard 拦住，须先修正可审核的预算与诊断，不能用不完整记录替代该前提。之后再单独比较下面的 fusion，不同时换 paired base、low tile、ANE ABI 或缓存。

保留现有三参 `NativeLoRA::prepare`，拟新增显式第四参 overload，选择 `FfnFusionVariant::Disabled / FinalFullUpSilu`。Disabled 不解析／准备新 shader，默认零额外 PSO；首包仅 SGv2 增加一个 apply-SiLU PSO，low 的 FP32 运算与旧 source body 不变。新增 v3 recipe 包含 fused 及 separate-tail 的政策；实际回执另外记录真正执行的 fused／fallback 数量，prepared variant 不能冒充每个 FFN 已融合。

```cpp
// 设计接口；当前 NativeLoRA 尚无此方法。
UpSiluDisposition encode_up_corrections_silu(
    CommandGraph&, const MatrixView& input,
    std::span<const OrderedLoRATerm> ordered_up_terms,
    const MatrixView& corrected_gate, const MatrixView& up_workspace,
    const MatrixView& hidden, const MatrixView& fp32_low,
    std::optional<MatrixView> up_bias = {});
```

up workspace 初始已经是未加 bias 的 BF16 base。新方法仅保证 hidden；融合时 up workspace 不包含最后一项校正，不能再作为最终 up 读取。旧 `encode_ordered_corrections` 的输出承诺不变。只能融合真正 vector 末项，且其 `first_output=0`、`output_count=N`、无 up bias、SGv2 eligibility 成立。此前 partial／多个 adapter 原样逐项 low→apply→BF16，不重排、不合并、不跳过零 scale；末项 partial、scalar/tail、空 terms、bias 或 Disabled 都走分离路径。空 terms＋bias 要独立编码已有 bias PSO→SiLU，不能直接调用会拒绝该组合的旧 correction API；空 terms 时 low 可空。

融合 shader 逐字保留 SGv2 的 FP32 tiles、四次 K8 MMA、result store 和 uniform barriers，只替换末尾 epilogue：先形成 `u=BF16(float(old_up)+float(result*scale))`，再逐字执行现有 BF16 exp→denominator→reciprocal→sigmoid→activated→hidden。gate 已完成全部 ordered correction；不能在未舍入 base 或 FP32 up 上直接激活。建议参数 ABI 保留原 apply 的 36 字节前缀，再增加独立 gate／hidden pitch，共 44 字节；绑定 low／B／up-readonly／gate-readonly／hidden／params，原始 offsets 只由 Backend 绑定一次。

全部 terms、实际 device、physical extent 和 alias 必须先校验，再构造 local draft 并一次 append。up 不得重叠 gate；hidden 不得重叠 input／up／gate／任何 A/B／bias；low 与 gate／hidden 共 backing 也拒绝。prepared PSO 不证明 gate readiness，仍由同一图中的 producer 顺序建立。失败或 fallback 不得留下半张图。

对当前单项 gate/up/down adapter 的适用 FFN，目标是 10→9 dispatch；M1024／F12288 时省去最后 up 的 24 MiB 写和随后 24 MiB 读。scratch 容量不因此减少，源码字节数也不等于实际 DRAM 流量或提速。CPU／source 合同之后先用生成张量验证所有边界，再复用同一真实输入进行最多六次完整 FFN 配对；有完整成本收益后，才考虑模型接线与有限 E2E。

## 12. 本轮基础实现与有界验证结论

构建 `tc-runtime-build-v1-47d3ed80c5de774df4ffa3ed634f9d362a0074f90c8b426a13df6f10abd4904a` 已完成 private package 编译，571 个 identity sources 摘要一致；package-only 不代表全套测试已运行。新增能力只在显式原语／工具中可调用，默认 Scalar／Base stages 和请求准入不变。

| 实施项 | 本轮证据 | 仍未证明的能力 |
|---|---|---|
| validated low／apply 拆分 | 新 split 合同 172 项；既有 contract／source／math 合计 1646 项；public/private host 检查通过，旧 shader 两个 body 字节不变 | 跨图共享 low 的 freshness／ready permit、多个并发 slot |
| 不可变 hybrid LoRA Spec | CPU 244 项，含 209 个拒绝案例；精确布局、预算、binding／generation 元数据 | held-FD 内容验证、物理访问许可、实际 Surface 分配／controller |
| native LoRA MIL emitter | 685 项 CPU 检查、4 组旧输出字节一致；input dtype／shape／pitch 和 guard 依赖图核对 | 新 ANE 编译、placement、硬件 INT8、运行误差或耗时 |
| shared low 微型设备检查 | M8／K128／N128／r128；旧 low＋apply 两节点，新 low＋两个 B 行窗三节点；共 2 Tickets／5 dispatch，low 和 output 与实际 MLX／Weights oracle 逐位一致 | 真实 FFN／ANE 并行／性能或模型资格 |
| warmed MLX／Native 完整 FFN 对照工具 | 同一 layer0／step5、相同 held Weights，输出实际 Data owner 新鲜与退休；计划 warm AB、record AB、BA，没有额外 oracle | 此次执行只完成 warm AB 和 record AB，framework guard 中止，没有 BA 或完整性能结论 |

微型检查的四个 native scratch 共 64 KiB capacity upper，实际生成 adapter 文件 65,783 B；两个成功 Ticket、padding／source 不变与新增 ledger 归零通过。它在**同一 graph 内**建立 low→两个 apply 的依赖，不能据此宣布跨图 GPU／ANE 共享 low 已安全。

真实对照只物化该层三个 base 权重、六个 factors 和输入，payload 308 MiB，native scratch 为六份。MLX 一次构造 `runtime_ffn::full` compiled closure，每 arm 重新 invoke，实际 `eval/synchronize` 后取输出；native 每次编码 10 节点并等 Ticket 物理完成。两者计时域明确不同，不能伪造 MLX Ticket 或相加 GPU span。当前 512 MiB framework allowance 是未认证的估算；第二对 checkpoint 报 `unattributed process footprint exceeds framework upper`，于是停止，未重试。所有已开始的四个 arm 都在各自实际完成边界结束，failure receipt 未保存完整 pairs／memory metrics，不能报告最终数值资格、账本归零或稳定性能差异。

接下来先补齐失败路径的 process baseline／current／peak、native ledger 与 MLX active／cache／peak 观测，再按真实 shape 和 liveness 计算 framework 上界。禁止简单关闭 guard 或在看过结果后任意抬限。之后完成 delta writer／recipe-specific status／hidden restore 的 HiddenReference 闭环，再比较 Rank；控制器、NativeBlock／Denoiser／Session、产品资格均继续分阶段接入。W1S 的逐层 host admission 和共同 IR／W2 的缺口仍在，不因增加 emitter 而消失。

本轮只运行上述两组轻量设备工作，没有 ANE、新整图或 App 验收。原模型与 capture 保留；已清理自有对象／archive／module caches、四组 rotation fixture 和微型 adapter 共 250 个文件、34,613,684 B（约 33.01 MiB），库／CLI／probe／原始日志和回执保留。详细身份、失败边界和清理见[本轮状态](../../status/2026-10-08-native-lora-foundation.md)。
