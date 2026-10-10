# 原生 block、ANE 桥与完整 DiT 接线合同

日期：2026-10-08。本章把 [10 的独立后端](10-native-metal-graph-backend.md) 落到当前组件和下一阶段的可实施合同。**基础 MLX 改造与独立原生后端继续是两条路线。** 这里的 N 路线直接拥有实际 Metal command graph；不会以每层回到 MLX attention 的 FFN island 作为完成定义。

当前源码新增默认关闭的 compact row-scale cache；生成数据及真实 FFN 的身份／精确复用已通过，完整组件耗时随顺序反转，没有稳定净收益，未追加整图，见[真实缓存筛选](../../status/2026-10-08-native-scale-cache-real-screen.md)。
当前源码已经实现原生 projection、Qwen pointwise/layout、分块 attention、FFN、完整 base block 编码和 private ANE W0 桥。后续新增完整 base DiT/Euler 图、共享 PSO/arena 与 request-local prefix 事务，实际接口见 [12](12-native-denoiser-runtime.md)；native phase 已接入同一账本，W1S 通道切分已接入稳态全层／Session。上一 register 整图验证构建（cache off）的一组完整 Base 对照去噪省时 13.60%、请求省时 9.97%，成功路径物理完成和阶段释放成立，但 App、泛化性能与异常设备路径仍待独立验收，见[最新状态](../../status/2026-10-08-native-register-w1-base-e2e.md)。本章组件阶段差距以 [13](13-native-session-and-memory-phase.md)、[15](15-native-w1-implementation.md) 和 [16](16-splash-native-backend-spec.md) 的当前记录为准。本章后面的 denoiser/prefix bank 类型仍是**设计草案**，不能当作当前 API；[前一轮组件记录](../../status/2026-10-08-native-block-and-ane-handoff.md) 保持其原始构建和验证范围。

## 1. 实际模块及责任

| 实际源码 | 组件阶段责任 | 当时未覆盖（当前状态见 12–16） |
|---|---|---|
| [backend.hpp](../../../native/backends/metal_graph/backend.hpp)、[backend.mm](../../../native/backends/metal_graph/backend.mm) | device/queue、retained buffers/events、准备 PSO、dispatch 参数副本、容量、物理完成 ticket；`append_graph` 使私有 draft 成功后一次追加 | 统一语义 IR lowering、生产 MemoryExecutionContext、读写集和跨设备事务 |
| [native_linear](../../../native/backends/metal_graph/native_linear.hpp) | BF16 `[M,K]×[N,K]`，FP32 累加，BF16/FP32 输出；K/N 支持 64 对齐及列尾；可选 BF16 bias | 任意 4304 尾部、LoRA epilogue、跨格式 kernel 选择 |
| [native_qwen_ops](../../../native/backends/metal_graph/native_qwen_ops.hpp) | LayerNorm/scale、tanh gated residual、QK RMSNorm/RoPE、head pack/unpack/copy；后续补齐 conditioning、SiLU/GELU 与调制展开；公开前置校验 | 其他模型的 norm/RoPE；采样/finite 由新 StepOps 承担 |
| [native_attention](../../../native/backends/metal_graph/native_attention.hpp) | BF16 D128、1–32 个相同 Q/KV heads、动态长度/stride、causal logical offset；分块在线 softmax，无全 N² score 缓冲 | GQA、D72、参考/target 双区间直接读取、encoder FP32 attention |
| [native_ffn](../../../native/backends/metal_graph/native_ffn.hpp) | BF16 gate/up→SiLU×up→down，原始 fused 权重布局和通道范围，明确 FP32 partial | bias、LoRA、ANE branch 的联合编码 |
| [native_block](../../../native/models/qwen21/native_block.hpp)、[实现](../../../native/models/qwen21/native_block.cpp) | 所有 block GPU commands；首步 segment attention/prefix 导出；稳态 prefix concat；私有 draft 与跨步骤 alias 检查 | 模型请求选择、32 层循环、prefix 身份/事务发布、异步 scratch 复用 |
| [native_ane_handoff](../../../native/backends/metal_graph/native_ane_handoff.hpp) | 导入同一 IOSurface backing；自己提交 ready producer；W0 确认 GPU 和 ANE 成功后授予只读结果许可 | 完整 FFN、A8/rotation/staging/restore/join 的原生 lowering、W1 并行及生产 slot ledger |

原生 attention 的必要 Steel 支持已内嵌，保留 [Apple/MLX 与 vpipe 声明](../../../THIRD_PARTY_NOTICES/mlx-steel-attention.md)，运行时不访问开发机的外部仓库。device→graph→ticket→ANE→资源/校准的组织继续以 [Splash #260](https://github.com/incoai/splash/pull/260) 为主要参考；Qwen Image 拓扑和数值边界以本地模型实现为准。

## 2. 先冻结数学，再扩大执行域

本地 [transformer.cpp](../../../native/models/qwen21/transformer.cpp) 的 block 是：

```text
x1 = BF16(LayerNorm_BF16(x) × BF16(1 + scale_attn))
q,k,v = BF16 native projections(x1)
q,k = RMSNorm + pair RoPE → head-major BF16
a = tiled SDPA(q,k,v) → BF16 → row-major
x2 = BF16(x + BF16(BF16(tanh(gate_attn)) × BF16(out_projection(a))))
x3 = BF16(LayerNorm_BF16(x2) × BF16(1 + scale_ffn))
f = down(BF16(SiLU_BF16(gate(x3)) × up(x3)))
out = BF16(x2 + BF16(BF16(tanh(gate_ffn)) × f))
```

`modulation.1` 输出 **4H**，顺序为 `scale_attn/gate_attn/scale_ffn/gate_ffn`；没有 shift。最后 `norm_out.linear` 是另一份 H scale，不能套用其他 AdaLN 模型的 6H 模板。

实现审查补齐的边界：

- LayerNorm 使用 FP32 mean 和 **中心方差**；`E[x²]-mean²` 可能消去，不作为同等替换。
- 普通 MLX RMSNorm 为 `BF16(BF16(x×rsqrt)×weight)`，乘权重前的 BF16 舍入必须保留，然后 FP32 pair RoPE→BF16。不能仅沿用此前 experimental fused kernel 的一次舍入并声称与普通 GPU oracle 一致。
- Linear 的 base bias 路线先按输出 dtype cast matmul，再加 bias、再 cast。以后带 LoRA 时 bias 放在按原顺序完成 adapter 累加之后。
- 当前 FFN 对本地 Qwen checkpoint 不漏 bias：只读 safetensors JSON 头确认 `gate_up [24576,4096]` 与 `out [4096,12288]` 均无 bias。其他 checkpoint 必须重新检查，不能静默丢掉未支持项。

这保留可对照的运算边界，**不承诺逐 bit 与 MLX 相同**；归约顺序、attention 算法和不同编译器仍可能造成差异。融合/FP16/W8A8/rotation 的新边界分别注册 recipe、重新校准。

## 3. 首步、稳态和 prefix bank

当前 block 有两种实际 phase：

- `PrefixBuild`：输入包含 text/reference/target，segments 必须完整连续覆盖；每段 Q 看 K/V `[0,end)`，causal text 使用逻辑 `start` offset。RoPE 后复制 prefix K/V。
- `SteadyDecode`：输入只含 target，K/V 仍须包括 prefix；当前实现分配两份 joined head-major K/V，分别复制 prefix 与 target 后执行 attention。**尚未消除 concat。**

上层 denoiser 需要下面的缓存合同；仅传两份 `HeadView` 不足以命中缓存：

```cpp
// 设计草案，尚未接线。
struct PrefixKey {
    SourceGeneration base;
    ConditioningIdentity text_and_references;
    SequenceGeometryDigest geometry;
    PrecisionRecipe normalization_rope_attention;
    FirstStepSigma sigma;
    AdapterIdentity adapters; // BaseOnly 时明确为空，后续包括顺序和 strength。
    uint64_t backend_generation;
};
struct PreparedPrefixBank {
    PrefixKey key;
    std::array<PrefixEntry,32> entries; // 每层 layer id、K/V storage 和 producer。
    TransactionEpoch epoch;
    std::bitset<32> tentative_layers;
    bool committed = false;
};
```

首步写入 tentative bank；完整 denoising step 成功、有限值和事务通过后才能发布。失败、取消、source/adapter/geometry/recipe 改变使相应 bank 失效，物理使用尚未结束的 storage 保留隔离。稳态只读取匹配且 committed 的 bank。现有 MLX cache 同内容也不能凭 shape 自动命中原生 bank。

参考图增加 prefix token，影响首步行数、后续 K/V 长度和每层缓存。保留 `SequenceGeometry` 的一至两张参考图排序、可见性和文本槽；先让通用 segment 路线正确，再验证双区间 attention 消除 concat 的收益。最后一层 target-only 优化属于后续可选优化；现有原生 block 仍计算全部首步行，不能因此声称已复用该优化。

## 4. 32 层的算子与内存复用

**同几何 operator bundle prepare 一次，32 层只更换权重视图。** 组件验收版本的 `NativeBlock::prepare` 是 16 个 PSO；补齐 conditioning 后当前 standalone prepare 为 4 Linear、2 Attention、11 QwenOps、3 FFN，共 20 项。每层各 prepare 一次会达到 640 项，超过 backend 默认 128 项。新的 `prepare_with_shared_ops` 和 [12 的 model bundle](12-native-denoiser-runtime.md) 已避免逐层准备。当前 `compile_library` 仍按调用创建 library，不能把 PSO map 误称为 source-level library cache。

推荐上层组织：

```cpp
// 设计伪代码：PreparedNativeDenoiser / StepTransaction 尚待实现。
auto ops = NativeBlock::prepare(backend, geometry);       // 一份算子。
auto workspace = reserve_workspace_for_largest_phase();  // 一份有预算的 arena。
for (uint32_t layer = 0; layer < 32; ++layer) {
    auto bindings = step.make_layer_bindings(layer, hidden_read, hidden_write);
    ops.encode(step.graph, bindings, workspace.for_phase(step.phase));
    std::swap(hidden_read, hidden_write);                 // 已排序的 ping-pong。
}
```

当前 block arena 的精确规划为每个 slot 按 256 bytes 对齐：

```text
13 × align256(M×H×2) + align256(M×F×2)
+ steady 且 P>0 时 2 × align256((M+P)×H×2)
```

Qwen `M=1024,H=4096,F=12288` 无 prefix 的 arena 为 **128 MiB**；`P=29` 的稳态约 **144.45 MiB**。32 层 prefix K/V 另约 **14.5 MiB**，也要入账；权重、hidden ping-pong、输入输出、encoder/VAE、ANE banks 和 driver 内存均不在这个 scratch 数字内。不能每层保留一份 arena，也不能漏算编辑增加的 prefix。

scratch 复用只允许同一队列严格有序的使用，或者已有最后消费者的物理完成证据；不可在前一次 step 未完成时从另一个请求覆盖。先共用一份保守无 alias workspace，再根据 dispatch live range 做别名/融合，逐项测量。

新增 library/PSO cache 需按 shader/source digest、语言标准、math recipe、function specialization、device/OS generation 区分，设置项数/内存上限。先共享 operator bundle，再考虑持久 metallib；不能用简单提高容量掩盖重复准备。

## 5. 完整 native DiT 的剩余接线

下表保留组件阶段的职责清单。前七行的 base 运算覆盖现在由 [12 的 NativeDenoiser](12-native-denoiser-runtime.md) 接线；共同 Session/source/阶段 memory/cancel 入口已在 [13](13-native-session-and-memory-phase.md) 实施。完整模型质量／性能、全请求内存认证、GPU/ANE 并行仍按独立证据验收，不能互相授予资格。

| 运算/职责 | 本地语义依据 | N 路线所需实现 |
|---|---|---|
| latent/input projection | `img_in [4096,64]`，patch/token 排序 | native Linear；输入 dtype、row shape、cast 与 sigma 一致 |
| text/reference conditioning | `txt_in.text_norm`、两次 projection/GELU；参考 latent→img_in；SequenceGeometry | 初期一次性预处理可在声明的准备阶段完成；要称完整 DiT 则本阶段的 GPU 运算由 N 控制，不能每 step 隐式回到 MLX |
| timestep/embedding | `Transformer::embedding`，两行当前 sigma 与 0，256→4096→4096，SiLU | native projections/activation；小型 CPU sigma/frequency 常量可显式准备并计时 |
| 调制 | SiLU(temb)→`modulation.1 [16384,4096]` | target 使用当前 timestep，prefix 使用 0；首步按区间广播四个 mods，稳态只保留 target |
| 32 blocks | 当前 `NativeBlock` 与 `Transformer` phase/cache 语义 | 共享算子、权重 leases、prefix bank、scratch 与 request identity；图中不逐层交回 MLX |
| output | target slice、`norm_out.linear`→LayerNorm×scale→`proj_out [64,4096]` | native ops/projection，保留输出 dtype；不能遗漏最终调制 |
| sampling/finite/cancel | [scheduler.hpp](../../../native/models/qwen21/scheduler.hpp)、[pipeline.cpp](../../../native/models/qwen21/pipeline.cpp) | 同 sigma schedule 和更新公式；finite flag、取消边界与完成事务，成功才发布 latent |
| 入口/出口 | `Session`/worker/文本或图像 encoder/VAE | 显式选择 backend 与 capability；阶段 boundary lease/readiness；结果仍走共同 Session 协议 |

`NativeBlock` 接收已经计算好的 mods/RoPE，不因此拥有完整 timestep/采样路径。生产入口必须校验精度、形状、base/adapter、mask/edit/cache/streaming 能力；选择不支持组合时在产生副作用前回到既有路线，不能忽略 adapter 或参考图以继续运行。

第一资格域仍为 512² BaseOnly text-to-image。之后接一至两张参考图的 DiT；视觉/文本 encoder 作为独立阶段验证。视觉 D72、GELU F4304，语言 causal GQA，当前 D128/SwiGLU primitives 未覆盖它们。

## 6. W0 桥与真正并行的分界

实际 W0 路径为：

```text
Native graph 写入 retained IOSurfaces → 自己追加 fresh ready signal
→ GPU Ticket 成功且物理完成 → prepare/submit 新 ANE request
→ ANE callback 成功、物理完成、未 retired、provenance 匹配
→ NativeAneReadPermit → 单独编码 restore/join GPU graph
```

这里没有预排 ANE output restore，没有失败时 force-signal，**也没有 GPU/ANE 并行性能资格**。当前 `NativeBlock` 的 FFN 仍调用完整 GPU `NativeFfn`，新 ANE bridge 与它尚未组合。

需要明确的身份及寿命约束：

1. ready/done 是新建的独立事件，target 均为 1，owner 强持有；不复用同一事件的旧 request 值。`MTLSharedEvent.device` 可合法为 nil，保留 creator-device metadata；非 nil 的 reported device 才额外匹配。输入/output Metal buffers 仍严格同设备，driver eventPort/selector 检查保持。
2. provenance 包括 program key/generation、domain、epoch、request、slot generation、producer generation 和 function index。event 数值及 metadata 不能替代真实 producer Ticket 成功。
3. adapter 自己提交 producer 图，但当前 `GraphStep` 没有读写集；**由 typed Plan builder 证明图覆盖全部输入/权重写入或已完成不可变输入**。不得接受外部任意 graph 与自报身份作为生产认证。
4. `NativeAneReadPermit` 仅授予输出读取资格。handoff 的 `all_work_physically_complete` 只覆盖 producer 与 ANE，**不包含后续 restore/join**；读取 graph 必须保留 view，并单独登记最后消费者。input/output/bank 可写槽只能在最后读取物理结束后复用。
5. 超时/取消/拒绝/部分提交导致逻辑退役；未确认停止访问的 backing、program、bank/event 继续隔离，不授予读取或复用许可。迟到 callback 不复活成功状态。

扩大为 W1 时，先拆分 block 的 attention/FFN-input prefix 与 FFN suffix，再原生编码 A8/rotation、W8 staging、GPU 通道补集、ANE restore/FP32 join。可提交 ready producer 后立即安排 GPU 补集，并在 W0 式有效结果许可后才提交 restore；先测安全 W1，不直接预排依赖输出读取的 W2/W3。slot/producer/ANE/consumer 必须进入同一个 completion domain。

带宽和同步成本仍须整段测量：不能把已与计算重叠的 staging 时间全部算成可消除开销，也不能只报 ANE alone time。新后端须同时对自己的 GPU-only 与既有最佳 GPU/hybrid 比较；更慢的 GPU kernel 不能制造表面 ANE 倍率。

## 7. LoRA、统一资源与验收顺序

BaseOnly 不预分配 LoRA 超集 scratch。后续 prepared factor view 复用既有 adapter target、顺序、strength 和 provenance；涵盖所有实际 target，包括 attention、FFN 及可能的输入/时间/调制/输出投影。gate/up 修正位于 SiLU 前，down 修正位于输出边界前；bias 位于 adapter 累加后。冻结编译、动态修正、SmoothQuant/Hadamard 是不同 recipe，分别校准 base/LoRA 并控制衍生缓存容量。

`AllocationHooks` 必须接生产 `MemoryExecutionContext`；SourceLease、StorageId、reservation、真实 allocated bytes 和 completion mailbox 统一管理。import 复用 existing lease，延长使用期仍计入峰值。native_handle 的 alias 检查不能识别两个 bytesNoCopy wrappers 共用同一地址，typed storage/range ledger 需要补齐，不维护另一套私有内存池。

验收按以下顺序推进，继续遵守笔记本 1–2 组 E2E 的预算：

1. 小张量投影、norm/QK、分段 attention、prefix/steady 和实际 W0 bridge；非法绑定/迟到完成/未完成槽不复用。
2. 少量真实 **block** 输入，对普通 GPU oracle 和当前最优路线比较完整命令段。已有 real FFN capture 是 post-attention 输入，不能充当完整 block 输入，也不能用 tiny H128/256 的时间推算 H4096 性能。
3. 32 层单步 prefix + steady，覆盖实际权重、缓存发布、有限值及峰值；可先 GPU-only。组件完整路径有收益后再安排同 seed 配对图。
4. 先 base，再 LoRA；Qwen edit、Z-image 专属拓扑和 encoder 独立资格。App 高级配置只暴露已经证明的组合，未验收路线保持实验性。
5. 每轮记录来源、build/recipe、shape、首步/稳态、GPU busy/host gaps、staging/restore/join、wall 与实际设备行为。清理本轮拥有的无用 cache/objects，不删模型或未归属的其他缓存。

用户接受约 5–8% 数值差异作为候选探索，并以同条件图像肉眼比较验收。它允许研究替代 precision/algorithm，不允许遗漏模型算子、忽略 LoRA/参考图，或放松同步和内存安全。
