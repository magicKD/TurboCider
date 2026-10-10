# Qwen Image：模型执行计划、LoRA 与量化合同

> 状态：改造设计，尚未实现。日期：2026-10-07。本文的接口和伪代码是拟议接口，不是已经存在的 API。性能结论仅来自文中引用的既有回执；编写本文不需要下载模型、运行生成或导出完整衍生模型。

本文规定 Qwen Image 2.1 如何映射到 `tc::hetero` 的 `ExecutionPlan`、`CommandGraphTemplate`、`GraphInstance`、`PreparedProgram`、`PreparedBinding` 和 `BufferLease`。模型层负责语义、精度和形状；通用执行层负责提交、事件、所有权及事务回退。不能为了减少同步而改变 Qwen 的 attention、LoRA、采样或缓存语义。

## 1. 范围与首版支持域

Base 阶段先独立优化本机现有 Qwen Image 2.1 BF16，不引入 LoRA。通过后再做已有 Viggle v0.2.1 r128 的 LoRA 阶段。本章保留两阶段的数学设计；下文 LoRA 小节中的“初版/首版”指 LoRA 首次接入，不是 Base 阶段的依赖。目标是把 Splash 的设备事件驱动通道并行用于真实 Qwen FFN，同时保留普通 GPU 路径作为完整参考和回退。图像编码、文本编码及 VAE 先沿用现有实现；不以重写这些组件为前置条件。

| 请求或实现 | 分阶段处理 | 扩展条件 |
| --- | --- | --- |
| 512×512、base、20 步、无参考图 | 主要筛选对象；首步与稳态分别选计划 | 组件、完整 block、E2E 同时通过 |
| 512×512、已绑定的 r128、strength=1、六步、无参考图 | 后续 LoRA 阶段；之前保留 GPU；保持 FP32 rank | Base 阶段通过后，单独校准、选择通道和提交策略 |
| 同一本地 adapter 的其他 strength、其他 rank | 保留原 GPU 能力，不自动套用已测策略 | 独立质量覆盖及完整路径收益 |
| 多 adapter、重叠目标范围 | 首版使用普通 GPU；不是模型能力限制 | 先解决有序舍入合同，再独立验收 |
| 长文本、参考图编辑、非方图、更大分辨率 | 保持原功能；没有合格计划则 GPU | 按真实几何、内存、前缀语义验收 |
| 原有 GGUF/ConvRot 源模型 | 按原 GPU 路径可用性运行 | 不从 BF16 的校准记录继承加速资格 |
| Private W8A8、公共 Core ML、冻结 Core ML | 各有独立能力和 recipe | 不能因为统一接口而宣称同步或精度相同 |

BaseOnly 需要精确的程序 ABI：没有 gate/up correction、down-LoRA 或额外 hidden 输出；不要用可选指针为 null 的 LoRA 超集图替代。现有代码已有纯 base 程序能力，但跨请求选择还需覆盖反向切换；具体审查见 [09](09-base-first-splash-review.md)。

这里的“主要筛选对象”也不等于发布验收通过。已有单提示词测试只证明该输入下有可继续研究的收益。六步计划只用于相应蒸馏 adapter，普通 LoRA 不会因为接入 ANE 自动成为六步模型。

## 2. 当前代码的具体差距

以下是改造必须覆盖的问题，不是另写一套 FFN 即可绕开的细节。

| 当前行为 | 影响 | 改造要求 |
| --- | --- | --- |
| 普通 GPU FFN 合并 `gate_up` GEMM；`runtime_ffn::channels` 对 gate、up 分别调用 `project_slice` | 通道并行可能损失原 GPU 算子的融合效率 | 增加保留物理步长的双区间 gate/up 投影；和现有融合 GPU 基线比较 |
| `corrections` 与 `channels` 是不同编译闭包，没有共享 `LoRAWorkspace` | 同一个 adapter 的 `xAᵀ` 可能被重复计算 | 将低秩结果变为显式图节点和 `BufferLease`，按消费者共享 |
| `HybridFfn::run_channels` 拼接 GPU/ANE 完整 hidden，再运行 down-LoRA | 额外张量、访存、等待；1024×12288 BF16 为 24 MiB | 对 down-LoRA 的输入维度分片，合并 rank 空间的结果 |
| runtime ANE correction 与普通 GPU 对多 adapter 使用不同舍入顺序 | 数学等价不代表数值等价；仅对比 hybrid 自身会漏检 | 明确 `RoundingContract`；首版拒绝未覆盖的叠加 adapter |
| 行数初始化取目标图像行数，不等于首步全部行数 | 1053 行落在 1024 桶时触发两次 ANE | 每个实际 block 选择 phase 和行数函数；不能据图片大小推断所有层 |
| 当前很多候选由环境变量组合形成 | 缺少可审计的完整语义身份，容易错误复用 | 构建不可变 `QwenExecutionPlan`，写入完整回执 |
| `runtime_ffn::full` 使用 `mx::split`，既有 `runtime_activation.hpp` 明确说明多输出 split 的图持有风险 | 编译闭包销毁后可能仍持有权重，需要回归验证 | 优先使用已有独立 slice helper；测试 Session/adapter 重绑释放 |

最后一项是可定位的所有权回归风险，当前文档不将其描述成已证实的泄漏。参考：[Qwen 图工厂](../../../native/models/qwen21/runtime_ffn_graphs.hpp)、[普通 Transformer](../../../native/models/qwen21/transformer.cpp)、[HybridFfn](../../../native/backends/ane_ffn.cpp)、[已有独立 slice helper](../../../native/models/qwen21/runtime_activation.hpp)。

## 3. 行数、前缀与 attention：图计划的输入

### 3.1 512×512 对应 1024 个目标行，不是 512 行

当前 latent/token 几何为：

\[
M_{target}=(H/16)(W/16),\qquad M_{prefix}=M_{text}+\sum_i M_{reference,i}.
\]

512×512 时 `M_target=1024`。没有可复用前缀时，普通 block 首步通常处理 `M_prefix + M_target`。已有样本的文本是 29 tokens，因此是 1053 行；这不是对所有 prompt 的固定数值。

前缀 KV 有效时，后续步只为目标图像产生 Q/K/V 并更新目标 hidden；attention 的 key/value 仍覆盖前缀和目标。**后续 FFN 是 1024 行，不意味着 attention 的 key 长度也只有 1024。** Qwen 代码中的 `decode` 是去噪稳态阶段，不是 LLM 的单 token 解码；不能套用 LLM decode 的带宽模型。

末层已有 target-only 优化，其 FFN 行数可能与前面各层不同。计划必须由真正的 `SequenceGeometry`、前缀复用状态和当前层输入形状产生，禁止写成 `step==0 ? total_rows : 1024`。

参考：[sequence.cpp](../../../native/models/qwen21/sequence.cpp)、[Transformer::forward](../../../native/models/qwen21/transformer.cpp)、[pipeline 的 total_tokens 与 prefix_runtime](../../../native/models/qwen21/pipeline.cpp)。

### 3.2 首步与稳态分开选择

首批候选仅需比较：

1. 首步 GPU、稳态 1024 行 channel split。
2. 首步选择一个覆盖真实行数的 ANE function，稳态仍用 1024。
3. 首步按行切成多个 FFN tile，仅在完整路径确有收益时采用。

优先保留少量有证据的 function，例如 1024 和一个实际需要的相邻桶；不能因 Splash 支持多 function 就无条件编译一整套最大形状。以 128 为步长的候选中，1053 可覆盖到 1152，额外行是 99；固定 1024 分两块则第二块大部分为空。1152 的 kernel 是否更快仍需测量。

Padding 只允许出现在逐行独立的 FFN 内部，必须满足：

- 原始真实行数、pad 行数分别记录；输出只消费真实行。
- gate/up LoRA 的 pad 修正为零，禁止读取未初始化 surface。
- per-row A8 统计不跨 token 混合；若 recipe 使用跨行统计，padding 必须显式 mask。
- 不将 padding token 塞入 attention 或 RoPE，不通过截断 prompt 缩小桶。
- 超过已编译容量时用受预算约束的 FFN tiling 或 GPU；不临时静默改变文本。

参考图或长文本可能使 attention 本身占比升高。FFN 变快不代表整个编辑请求同比变快；计划的成本输入必须同时包含 query/key 行数。

### 3.3 前缀缓存不等于近似去噪缓存

图模板、权重转换、前缀 KV、参考图 encoder、跨步 FFN/DBCache 必须分开管理：

- 图模板只复用拓扑，不意味着其输入输出张量仍有效。
- 前缀 KV 必须匹配 checkpoint、adapter 顺序与强度、文本和参考编码内容、参考图顺序/几何、RoPE、首步 sigma/调制合同及相关数值 recipe。
- 当前 `prefix_runtime` 已包含许多选项，迁移时应结构化保留，不能删掉看似影响不大的字段。
- 错误或取消的 graph 输出不能发布为前缀缓存；成功前保留 pending 状态。
- FFN/DBCache 是另一类可能改变结果的近似优化。首轮 command graph 性能验收关闭这些近似缓存，不把跳算收益算成调度收益。
- 前缀 KV 很大：当前 32 层、32 heads、128 head dimension 的 BF16 K/V 约为每个前缀 token 512 KiB。一个 1024 行参考前缀仅 K/V 就约 512 MiB，必须进入内存预算。

### 3.4 attention 的保留条件

A 基础路线首版继续使用 MLX 融合 SDPA；N 独立后端使用自己的 native attention，分别与同一 GPU oracle 对照。两路 CommandGraph 都必须保留文本 causal segment、参考图 segment、目标图 segment 及前缀可见范围。现有“局部参考 attention”等实验不能自动叠加。

后续可以开发接受分段 K/V 的 attention，避免稳态每层 `concatenate(prefix, target)`，但需要满足：

1. 不显式物化完整 `Q×K` score 矩阵。
2. 分段归约的 softmax 数值与现有 GPU oracle 对照。
3. 两个 K/V 区间不一定物理相邻，必须保留正确 stride 和 lifetime。
4. 和当前 MLX SDPA 对比完整 attention 路径，不只比较拼接耗时。

## 4. 模型计划与权重视图接口

以下结构表达所需字段，正式落地时可拆分到现有 runtime 类型，避免两套身份和内存系统。

```cpp
namespace tc::qwen21 {
enum class Phase { PrefixBuild, TargetDenoise, LastBlockTargetOnly };
enum class FfnRoute { Gpu, ChannelSplit, RowSplitReference };
enum class LoraMode { None, RuntimeActivation, StageMergedReference };

struct QwenShape {
    int layers = 32, hidden = 4096, width = 12288;
    int heads = 32, head_dim = 128;
    int target_rows, prefix_rows, query_rows, key_rows;
    SequenceGeometry sequence; // 源于实际 tokenizer/reference 编码
};

struct FfnPlan {
    FfnRoute route;
    int actual_rows, function_rows, tile_count;
    int gpu_channels, ane_channels;
    QuantRecipe recipe;
    RoundingContract rounding;
    LoraMode lora_mode;
    ProgramKey program;
    KernelVariant gpu_gate_up, gpu_down, join;
};

struct QwenExecutionPlan {
    QwenShape shape;
    SourceIdentity model;
    AdapterSetIdentity adapters; // 顺序、内容、目标范围、强度、alpha/rank
    RoundingContract oracle_contract;
    std::vector<FfnPlan> prefix_blocks, target_blocks;
    PrefixCacheContract prefix_cache;
    QualificationKey qualification;
    tc::hetero::ExecutionPlan execution;
};
} // namespace tc::qwen21
```

`WeightView` 必须表达逻辑和物理几何，不能只给裸指针与切片后的 `N×K`：

```cpp
struct WeightView {
    tc::hetero::BufferLease source;
    SourceIdentity identity;
    WeightEncoding encoding;       // DenseBF16 / MLXAffineQ4 / ...
    int logical_rows, logical_cols;
    int row_begin, row_count, col_begin, col_count;
    uint64_t byte_offset, physical_row_stride;
    QuantMetadataView metadata;     // scale/offset/packing/group/basis
};

struct GateUpView {
    WeightView gate;  // [0, F) 中选择该设备的通道
    WeightView up;    // [F, 2F) 中选择相同通道
};
```

双区间 gate/up kernel 可以在原 `2F×H` 权重上索引，不必为每个通道比例复制一份连续矩阵。down 是 `H×F` 的列切片，原始行跨度仍为完整 F（或对应 packed stride）。量化组、旋转块和切片边界需要联合对齐；首版只使用连续、对齐的通道范围，不同时引入任意敏感通道重排。

计划实例绑定实际 `BufferLease`，模板不捕获请求局部 `Weights&`。已有 `mx::compile` 闭包只能活在所属模型/adapter generation 内；在 adapter 重绑前 drain，销毁闭包和依赖张量，再释放源权重。禁止把现有捕获式闭包直接塞入跨请求全局 cache。

## 5. 通道并行 FFN 与 LoRA 数学

### 5.1 基本定义

以下按行表示 token，所有公式先描述实数运算；第 6 节单独规定有限精度行为。设：

\[
X\in\mathbb{R}^{M\times H},\quad W_g,W_u\in\mathbb{R}^{F\times H},\quad W_d\in\mathbb{R}^{H\times F}.
\]

LoRA 使用本地代码的 `down=A`、`up=B` 命名对应关系：

\[
A_j\in\mathbb{R}^{r_j\times K},\quad B_j\in\mathbb{R}^{N\times r_j},\quad
\Delta_j(X)=s_j(XA_j^T)B_j^T.
\]

`s_j` 必须来自已有 adapter 加载逻辑，包含用户强度及模型定义的 alpha/rank 归一化；图层不能重复乘一次 alpha/rank。

\[
G=XW_g^T+\Delta_g(X),\quad U=XW_u^T+\Delta_u(X),\quad
Z=\operatorname{SiLU}(G)\odot U,\quad Y=ZW_d^T+\Delta_d(Z).
\]

LoRA 必须先修正 gate/up，再执行非线性。`SiLU(base_gate)·base_up + LoRA_output` 不是同一个模型。

把中间通道分成 GPU 集合 P 和 ANE 集合 Q：

\[
Y_{base}=Z_PW_{d,P}^T+Z_QW_{d,Q}^T.
\]

GPU 和 ANE 可以同时计算各自分支，层与层仍由 attention/residual 依赖串联。可提前处理的是下一层只依赖权重的 staging，而不是凭空并行计算依赖当前 hidden 的下一层 FFN。

### 5.2 gate/up 的低秩结果只计算一次

同一 adapter 的 `L_j=XA_jᵀ` 不依赖输出通道，因此应由 GPU 计算一次，供 GPU gate/up 修正和 ANE 修正共同消费。注意：

- 分别训练的 gate 与 up 通常有不同 A，不能因为维度相同而复用数值。
- 多个 A 可以通过 grouped/batched/stacked 投影减少 dispatch，但每个 adapter 的身份、输出范围和舍入仍保持独立。
- 缓存键至少包含 graph instance、输入 generation、A 源身份、输入列区间、rank dtype；不能只用可重复分配的 GPU 地址。
- 首版低秩结果采用 FP32；不得照搬 vpipe 的 BF16 scratch 后仍标成原 FP32 recipe。
- 如果 ANE 输入采用 S1/旋转，LoRA 默认从原始 X 计算；不能误用已缩放/旋转的 X 乘未变换的 A。

```cpp
struct PreparedLoRA {
    AdapterSetIdentity identity;
    RoundingContract rounding;
    std::vector<AdapterProjection> gate, up, down;
    ScratchLayout rank_scratch; // 只保留有消费者的 rank 区间
};

struct LoraLowKey {
    GraphInstanceId graph;
    InputGeneration input;
    FactorIdentity a;
    ColumnRange input_columns;
    ScalarType accumulation; // 首版 FP32
};

// 构图：依赖明确存在于 graph，不靠 CPU 上的“最近一次输入”推断。
auto gate_low = graph.lora_low(original_x, prepared.gate, FP32);
auto up_low   = graph.lora_low(original_x, prepared.up, FP32);
auto ane_delta = graph.lora_expand({gate_low, up_low}, ane_range,
                                  plan.rounding.ane_correction_transport);
auto gpu_ffn = graph.gpu_ffn(original_x, gpu_weights,
                           {gate_low, up_low}, plan.rounding);
```

已有 `Weights::runtime_lora_low`/`LoRAWorkspace` 提供可参考的计算与 key 字段，但它目前按 MLX Tensor 身份管理，不能直接替代异步 graph 的 generation 和 lease 合同。参考：[mlx.cpp](../../../native/backends/mlx.cpp)。

按单个相同 rank 的 gate/up/down adapter 估算，三个 base GEMM 的乘加量为 `3MHF`，低秩分支为 `3Mr(H+F)`。H=4096、F=12288、r=128 时，低秩额外乘加约为 base FFN 的 4.17%。这不是耗时预测：FP32 rank、小 GEMM、dispatch、输出物化以及它们所处的关键依赖位置，可能使耗时比例大得多。因此不应只根据 FLOPs 决定把 LoRA 留到 join 后串行完成。

强度为零时可以完全省略对应低秩节点和修正运输，使用匹配的 base 计划，而不是执行两次 GEMM 后乘零。其基础模型、其余 adapter 和数值合同仍须匹配；不能借零强度特例绕过一般强度的校准身份。

### 5.3 分片 down-LoRA，避免完整 hidden 拼接

对 down adapter，把 A 按输入通道分成 `A_P`、`A_Q`：

\[
L_P=Z_PA_P^T,\qquad L_Q=Z_QA_Q^T,\qquad
\Delta_d=s(L_P+L_Q)B^T.
\]

执行顺序：

1. GPU hidden 就绪后马上计算 `L_P`，与 ANE 运算重叠。
2. ANE hidden 还原后计算 `L_Q`。
3. FP32 合并低秩结果，再进行一次 B 投影及规定的输出加法。
4. 释放分支 hidden，只保留后续消费者需要的结果。

在 M=1024、F=12288 时，完整 BF16 hidden 为 24 MiB；r128 的一份 FP32 low 是 0.5 MiB。实际节省受 scratch 复用、多个 adapter 和 ANE 输出 ABI 影响，不能把 24 MiB 全当作每层必然节省。LoRA 初版 ANE 仍需返回自己的 hidden；本优化首先消除 concat 和整块 low 输入，不承诺消除 hidden 的跨设备传输。BaseOnly 不需要这项导出。

```cpp
auto low_gpu = graph.lora_down_low(gpu_hidden, down_a.gpu_columns(), FP32);
// 许可依赖见第 9 节；禁止仅凭可人工放行的 done event 读取 ANE surface。
auto permission = graph.require_result_read_permit(ane_job, graph.window_certificate());
auto restored = graph.restore_when_permitted(permission, ane_job.outputs, plan.recipe);
auto low_ane = graph.lora_down_low(restored.hidden, down_a.ane_columns(), FP32);
auto low = graph.add(low_gpu, low_ane, FP32);
auto delta = graph.lora_expand(low, down_b, plan.rounding.down_delta);
auto base = graph.join_partials(gpu_base, restored.base, plan.rounding.base_join);
auto y = graph.add_down_delta(base, delta, plan.rounding);
```

约束：

- `restored.hidden` 必须是完成 headroom/量化缩放还原、处于定义坐标系的 hidden，不能拿缩小的 `hsafe` 直接乘原始 A。
- 两个 rank 部分的求和顺序不同于一次完整 FP32 GEMM，通常有小误差；这是新的实现变体，必须和完整 hidden GPU oracle 对比。
- 禁止先各自计算并 BF16 舍入两个完整 `s·L·Bᵀ` 再相加；这增加了输出舍入且可能重复执行 B。
- 多 adapter 的 rank 可以分配在同一 scratch arena，但其有序输出舍入不能被默认改成一次总和。
- 如果进一步把 `Z_Q A_Qᵀ` 放到 ANE，需要新的动态 A 输入、精度、staging 和 program ABI；首版不把该设想作为既定收益。

## 6. RoundingContract：精度边界必须成为接口

### 6.1 两种 oracle，解决不同问题

**产品 GPU oracle**：相同 checkpoint、adapter 列表/顺序/强度、prompt/reference、seed、采样计划和已接受选项下，当前普通 GPU 完整执行结果。它回答“模型行为是否仍然合格”。

**兼容 hybrid oracle**：当前已验证的 hybrid 路径，或将同一 `QuantRecipe`/精度边界显式模拟的参考实现。它回答“事件调度、布局、staging 和新融合是否破坏旧路线”。

必须同时对照两者。新实现与旧 hybrid 相同，不代表旧 hybrid 与普通 GPU 一样；只对照最终 PNG 也容易漏掉中间溢出、错误缓存或 adapter 遗漏。

**运行期 GPU 回放是第三个概念。** 它从失败事务的输入 checkpoint 重新执行原 GPU 算子；该 checkpoint 的 latent 或已提交 prefix 可能来自之前成功的 hybrid step，因此不等于“整请求从相同初始 noise 开始的纯 GPU oracle”。回执应写 replay 范围与此前实际路线，不能在一次回放后把整个请求标成纯 GPU。

首步失败时丢弃本事务新建的 pending prefix，重算相应事务；后续步可以读取先前已提交、未污染且被 fallback 合同允许消费的 prefix。若数值/布局合同不兼容，应执行明确的重新准备或请求重启策略，不能把其缓存身份简单改名成 GPU。质量测试中的产品 GPU oracle 始终从原始请求输入独立建立自己的前缀和轨迹。

### 6.2 当前存在的多 adapter 舍入差异

普通 `Weights::project`/`apply_runtime_lora_slice` 是有序更新：

\[
y_0=Q_b(XW^T),\quad y_{j+1}=Q_b(\operatorname{fp32}(y_j)+\Delta_j).
\]

`lora_delta_slice` 则在 FP32 中累加多个 adapter，再一次 cast；当前 ANE gate/up correction 和 hybrid down 路径使用这个接口。也就是近似如下合同：

\[
\delta=Q_b\left(\sum_j\Delta_j\right),\qquad
y=Q_b(\operatorname{fp32}(y_{base})+\operatorname{fp32}(\delta)).
\]

即使只有一个 adapter，提前把 delta 转 BF16 也不等于普通 GPU 直接把 FP32 delta 加到 base。ANE 上还有 FP16 运算/运输及 W8A8 误差。首版必须准确标记这些差别，不能宣传“LoRA 完全同精度”。

后续支持多 adapter 有两条明确路线：

1. 以原 GPU 有序舍入为目标，在 hybrid 中保留每个 adapter 的修正与 cast 序列，付出额外 ABI/带宽成本。
2. 提出新的“先聚合 delta”精度合同，经模型质量验收后启用；不能悄悄修改原 GPU baseline 以让对照通过。

### 6.3 合同字段

```cpp
struct RoundingContract {
    std::string version;
    ScalarType activation_storage;
    ScalarType base_accumulator;
    ScalarType rank_accumulator;
    AdapterReduction adapter_order; // OrderedCast / AggregateThenCast
    Boundary gate_base, gate_delta, gate_corrected;
    Boundary up_base, up_delta, up_corrected;
    Boundary silu, hidden, rotated_hidden;
    Boundary gpu_down_partial, ane_down_partial, base_join;
    Boundary down_delta, final_output;
    TransportContract ane_correction_transport;
    ReductionContract down_rank_partition; // full GEMM / split FP32 reduction
    BiasPlacement bias; // 加一次；不能在两路 partial 重复加
};
```

不是所有组合都应开放。使用注册过的、具备参考实现的合同 ID，未知组合直接拒绝构图。FP32 partial join、FP16 rank、BF16 值边界模拟、分片 down rank 都必须更新对应身份并重新验收，不能只改显示名称。

当前 Private MIL 还包含 BF16 value-boundary 的实验分支，其限制为特定 base-only Comfy recipe，不能推断所有 LoRA/分组量化都支持。FP16 carrier 不能覆盖 BF16 的完整数值范围；非有限值、溢出、headroom retry、sentinel 必须进入质量和失败回执。参考：[ane_mil.cpp](../../../native/backends/private/ane_mil.cpp)。

### 6.4 融合的保守落地

优先融合内存操作而保留数值边界，例如在同一 shader 中显式保留 base cast、adapter 有序 add/cast、SiLU 和 hidden cast。这样不必物化每个中间张量，但算术合同仍可追踪。

base GEMM＋LoRA B epilogue 需要另外比较。若将 base 的 FP32 accumulator 直接加 delta 再一次 cast，就移除了原 base BF16 边界；它可能更准确，也可能改变模型结果，不能作为“完全等价”的融合。应并存 conservative 和 altered-rounding 变体。

## 7. Runtime fused 与 weight merged：分别管理

`runtime fused` 指缩减算子/dispatch 或在 kernel 内合并低秩修正；`weight merged` 指构造 `W'=W+sBA`。两者没有相同的启动、内存和精度成本。

| 路线 | 优点 | 成本与限制 | 本方案定位 |
| --- | --- | --- | --- |
| Runtime activation LoRA | 源权重不变；强度切换便宜；低秩结果可共享 | 每步计算 A/B；需要修正 ABI | LoRA 阶段主路线 |
| 低秩 B 融入 GPU base epilogue | 可少写一次 base 输出、减少 dispatch | 占用寄存器；大 K/rank 可能变慢；舍入合同变化 | 按投影、shape 独立候选 |
| staging 时 merge 到 FP16 槽 | ANE 图不需要运行期 correction 输入 | 每次 staging 的 BA 成本；FP16/量化舍入；强度切换失效 | 参考 vpipe 的独立对照 |
| 预生成全模型合并权重 | 热态投影简单 | 磁盘、内存、准备时间，adapter 切换困难 | 当前约束下不做 |

vpipe 的共享 runtime-LoRA 实现提供紧凑 A/B 表示和预分配 scratch；其 Qwen ANE staging 接收 delta，将 BA 加到权重槽后转 FP16。应复用其**组织和带宽优化思路**，不能把这种实现写成和 TurboCider 的 pre-SiLU 激活修正完全相同。固定版本的对应位置：

- [runtime-lora.h：Applier::ensure_scratch/apply](https://github.com/tgo-app-dev/vpipe/blob/3e2682a0847c9e76dd37978f2be01f53195ae5e3/generative-models/shared/runtime-lora.h#L374)：每次 forward 按最大形状准备，apply 不分配；不同 adapter 有独立 scratch 区间。
- [runtime-lora.cc：Applier::ensure_scratch](https://github.com/tgo-app-dev/vpipe/blob/3e2682a0847c9e76dd37978f2be01f53195ae5e3/generative-models/shared/runtime-lora.cc#L934)：scratch 按每元素 2 字节 BF16 分配；不能直接替换当前 FP32 rank 合同。
- [Qwen ane_stage_](https://github.com/tgo-app-dev/vpipe/blob/3e2682a0847c9e76dd37978f2be01f53195ae5e3/generative-models/qwen-image/metal-qwen-image21-transformer.cc#L2549)：gate/up/down 分别把 runtime adapter delta 传入 staging；其源 gate/up 本来是独立矩阵，不能照搬成 TurboCider 已融合 `gate_up` 的物理布局。
- [ane-ffn.cc：1024 行 band 的 BA 合并](https://github.com/tgo-app-dev/vpipe/blob/3e2682a0847c9e76dd37978f2be01f53195ae5e3/generative-models/shared/ane-ffn.cc#L568)：使用 FP32 BLAS 批量准备 delta，再与解码 base 合并入 FP16 槽。它避免逐行小 BLAS，但仍需计入每次 staging 的准备成本。

如果源权重已量化或旋转，不能向 codes 直接加 BA。必须明确重建的是哪个数值域、何时施加旋转和量化、是否产生新的 scales。只允许在有界 slot/层内准备，并将 adapter/strength 纳入该权重准备结果的身份。

## 8. QuantRecipe：精度与存储 ABI 不能混为一谈

### 8.1 必须记录的字段

```cpp
struct QuantRecipe {
    std::string schema, id;
    WeightEncoding source_encoding;
    ScaleContract source_scales; // 粒度、dtype、zero/offset 的有无和含义
    RotationBasis input_basis, hidden_basis;
    int input_rotation_block, hidden_rotation_block;
    uint64_t rotation_seed;
    SmoothQuantContract s1, s2;  // 顺序、统计身份、clamp、alpha
    Quantizer weight, activation, hidden_activation;
    int input_group_size, hidden_group_size;
    int dot_segment_k;
    HeadroomContract headroom;
    RoundingContractId rounding; // 指向同一个合同注册项，不维护第二份可变副本
    std::string calibration_digest;
};
```

Program key 只包含真正影响程序结构/常量/ABI 的字段；含模型数值的 scale/权重准备缓存另行绑定源身份。质量资格必须绑定完整 recipe 和模型/adapter。不能为了增加程序缓存命中，把 source encoding 或 rounding 从资格身份中去掉。

### 8.2 旋转 W8A8 的数学位置

对正交 R：

\[
XW^T=(XR)(WR)^T.
\]

输入旋转作用于 GEMM 的收缩维度；不会把 FFN 输出通道任意混在一起。hidden 旋转必须在 gate/up 修正与 SiLU 乘法之后，不能移过非线性。

当前 Sylvester 路线使用有明确定义的符号、归一化和 block。Comfy H256 使用不同的 Hadamard 基、行列顺序与直接 codes 约定；它不是“把 Sylvester seed 改成 0”。

ANE 的动态权重 staging 与 GPU 激活量化必须使用同一方向、索引顺序和归一化。unit test 应验证小矩阵的正交性、轴方向和量化前恒等式，再验证真实张量。只检查 norm 不足以发现行列置换错误。

### 8.3 SmoothQuant 与旋转的组合

输入侧对角缩放 D1：

\[
X'=XD_1^{-1},\quad W'_g=W_gD_1,\quad W'_u=W_uD_1.
\]

再加入旋转时，输入为 `X D1⁻¹ R1`，权重为 `W D1 R1`。D 和 R 一般不可交换。GPU 补集可继续使用原权重与原 X；LoRA 修正也默认从原 X 计算。

hidden 侧若采用 `Z'=ZD2⁻¹`、`Wd'=WdD2`，可通过 up 分支缩放形成 Z'，但 up-LoRA delta 也必须除以 D2。down-LoRA 则需要原 Z，或对 A 做对应变换；不能用 Z' 直接乘原 A。

校准必须使用**实际 post-attention、modulation 后的 FFN 输入**。有 LoRA 时 hidden 统计必须包含 gate/up 修正；仅有 LoRA 请求产生的 X，但仍按 base gate/up 计算 hidden，不构成 LoRA hidden 校准。

首版使用既有捕获和 [LoRA calibration corrections 工具](../../../tools/coreml/qwen21_calibration_corrections.py)，保持 base、LoRA 轨迹分开。把训练/选择候选的样本和末期/其他深度验证样本分开，避免在单个早期 activation 上拟合后宣称整条轨迹可用。

### 8.4 编码支持矩阵

| 源数据 | GPU 补集 | ANE staging | 禁止事项 |
| --- | --- | --- | --- |
| Dense BF16 | 原生 BF16 GEMM/MPP 候选 | 源视图→旋转→W8，按预算双槽 | 先转完整模型 FP32 常驻 |
| 本地支持的 GGUF→MLX affine Q4/Q8 | 原 quantized GEMM，保留组和 packed stride | 解码、旋转、量化合并在 staging kernel 或有界块内 | 将 GGUF 文件容器名当唯一量化格式 |
| 已旋转 ConvRot signed Q8 | 对应 H256 激活旋转＋量化投影 | 符合 direct-code ABI 时直接 codes/scales | 再旋转一次 W；把 BF16 标成 direct Q8 |
| 已旋转 ConvRot packed affine Q8 | 保留 group、offset 和原尺度含义 | 验证 signed code 重构与 scales 后走 direct 路线 | 丢掉 offset 或更改 signed/zero scale 含义 |
| 未支持的位宽、布局或其他模型格式 | 原路径支持则原路径运行 | 拒绝 hybrid，给出具体原因 | 静默 reinterpret 或偷偷下载另一份模型 |

现有 `Weights::quantized_geometry` 明确支持的 native GGUF 范围是 Q4_0/Q4_1/Q8_0 对应表示；不能据此宣称通吃 GGUF。参考：[mlx.cpp](../../../native/backends/mlx.cpp)、[ane_gpu.mm](../../../native/backends/ane_gpu.mm)。

### 8.5 为什么量化格式可能在 GPU 更慢

量化减少权重字节，不保证 GPU 矩阵算术或完整路径更快。必须分别计入：

- packed 解码、scale/offset、旋转、dtype cast；
- 原 BF16 GEMM 的融合和 tile 效率是否丢失；
- 原始 leading stride、切片对齐导致的低效或拷贝；
- 独立旋转重复计算，例如 Q/K/V 或 gate/up 共同输入；
- GPU 本就受计算吞吐限制时，节省带宽不足以抵消额外算术。

因此量化 GPU 候选首先做“共享输入变换”和“解码融入 tile”，而不是默认整模型预解码。预解码仅允许有界的当前/下一层 scratch 或固定少量热层缓存。循环扫 32 层、容量远小于 32 时，普通 LRU 可能每轮零命中；需要固定子集或明确的 admission 规则。

即使预解码变快，也要报告新增常驻内存、首次准备和跨请求保留成本。它不能挤掉 KV/encoder 必需内存或触发压缩交换后仍算成功。

### 8.6 “W8A8”不等于物理 ANE INT8 已证明

Recipe 描述传入和图内量化数据。公共 Core ML placement preference、MIL 中出现 quantize/dequantize、名义 TOPS 都不能单独证明实际物理 ANE INT8 执行路径。回执应分别记录：

1. 请求的 backend 和 precision；
2. 编译/加载的真实程序及实际提交次数；
3. 可获得的计划归属或硬件轨迹证据及其局限；
4. fallback/retry/overflow；
5. 完整路径耗时与质量。

有 fallback 的候选输出零误差，不是 ANE 质量或加速证据。

## 9. 构图伪代码与关键依赖

下面展示单个 FFN 的模型依赖。事件、失效 output、失败事务和 slot 归还由通用执行层负责；模型层不得在事件被放行后直接认定结果成功。**结果读取许可独立于完成事件**：失败时人工提高 done event 可能只是排水，ANE 仍在写 output surface。此时“先 restore、最后不发布”仍有跨设备读写竞争，不能作为安全方案。

采用 [03 的窗口合同](03-command-graph-and-handoff.md)：W0/W1 只有确认该 ANE 访问物理完成后，才能 admission restore；W2 需要已认证的独立资源版本或提交/跳过协议。`WindowCertificate` 描述允许的协议，`ResultReadPermit` 授予某一次结果的合法读取。纯 `device_done` 事件不能构造后者。构图时只能创建等候许可的依赖，实际命令必须等许可条件成立才提交。

```cpp
FfnOutputs build_qwen_ffn(GraphInstance& graph, const FfnPlan& plan,
                         TensorLease x, WeightViews w,
                         const PreparedLoRA* lora) {
    switch (plan.route) {
    case FfnRoute::Gpu:
        return graph.qwen_gpu_ffn(x, w, lora, plan.rounding);
    case FfnRoute::RowSplitReference:
        // 独立的 rows/LoRA/精度合同；未实现时返回明确 Unsupported。
        return graph.qwen_row_split_reference_or_unsupported(x, w, lora, plan);
    case FfnRoute::ChannelSplit:
        break;
    default:
        return unsupported_route();
    }

    if (plan.variant == FfnVariant::BaseOnly) {
        require(lora == nullptr && plan.program.io == BaseOnlyABI);
        auto slot = graph.bind_program(plan.program);
        graph.ensure_weights(slot, w.ane, plan.recipe);
        auto a8 = graph.rotate_quantize(x, slot, plan.recipe);
        auto ready = graph.ready_after({a8, slot.weights_ready});
        HandoffTicket ane = graph.enqueue_ane(slot, ready);
        // 不返回对外可见的 hidden；内部计算 scratch 仍由 graph 持有。
        auto gpu_base = graph.qwen_gpu_base_channels(x, w.gpu, plan);
        graph.stage_next_weights_if_admitted();
        auto permit = graph.require_result_read_permit(
            ane, graph.window_certificate());
        auto ane_base = graph.restore_base_when_permitted(
            permit, ane.outputs, plan.recipe);
        auto output = graph.join_base_partials(
            gpu_base, ane_base, plan.rounding);
        return {output, graph.validity_dependency(ane)};
    }

    // 后续 LoRA 阶段；不以 nullptr + EmptyLows 模拟 BaseOnly。
    if (plan.variant != FfnVariant::ActivationLoRA || lora == nullptr)
        return unsupported_variant();
    auto slot = graph.bind_program(plan.program); // PreparedBinding + leases
    graph.ensure_weights(slot, w.ane, plan.recipe); // 可提前到上一层窗口
    auto lows = graph.prepare_low_rank(x, *lora);
    auto corrections = graph.expand_ane_corrections(lows, plan);
    auto a8 = graph.rotate_quantize(x, slot, plan.recipe);
    auto ready = graph.ready_after({a8, corrections, slot.weights_ready});
    HandoffTicket ane = graph.enqueue_ane(slot, ready);

    auto gpu = graph.qwen_gpu_channels(x, w.gpu, lows, plan);
    auto low_gpu = graph.down_rank_partial(gpu.hidden, lora, GpuRange, plan);
    graph.stage_next_weights_if_admitted(); // 不依赖下一层 activation

    // 返回一个有界的依赖 future，不是在构图时授予读取权限。
    // Lowering 保留 ticket/leases，W0/W1 由 completion reactor 允许提交；
    // W2 必须匹配已通过失败访问验证的 WindowCertificate。
    Pending<ResultReadPermit> permit =
        graph.require_result_read_permit(ane, graph.window_certificate());
    auto tail = graph.restore_when_permitted(permit, ane.outputs, plan.recipe);
    auto low_ane = graph.down_rank_partial(tail.hidden, lora, AneRange, plan);
    auto output = graph.join_and_finish(gpu.base, tail.base,
                                       low_gpu, low_ane, lora, plan.rounding);
    // output 仍属于当前事务；只有 status 与完成状态均有效才能发布。
    return {output, graph.validity_dependency(ane)};
}
```

Base 阶段只实现上面的精确 BaseOnly 分支。进入 LoRA 阶段后，可以先保持当前完整 hidden down oracle，只替换调度；共享 low、分片 down 和 epilogue 分别作为独立变更。这样每一步都能定位是布局/同步 bug，还是新的数值实现发生了变化。

## 10. 既有证据决定研究顺序

[真实 FFN 报告](../../status/2026-10-07-qwen-real-ffn-screen.md) 的本机结果：

- base 20 步：GPU 总耗时 48.400 s，当前最佳候选 43.044 s；去噪 42.155→35.268 s。
- r128 六步：GPU 总耗时 20.664 s，候选 21.292 s；去噪 14.183→13.637 s，但 setup 约 1.187 s 抵消收益。
- base 6144 通道提前提交的完整 FFN 组件约 27.9 ms，优于未提前提交约 39.8 ms；这部分已体现在现有候选，不能再作为新图的额外倍率。
- base 的 S1＋旋转、LoRA 的 S1＋旋转未胜过各自较好的旋转候选，首版不默认启用。
- 固定 SQ 单层 recipe 在未参与校准步骤的 RRMS 为 base 0.249、LoRA 0.112，超过该轮 0.03 门槛；不继续该 recipe 的全模型导出。这不证明所有 SmoothQuant 都无效。
- [冻结 FP16 回执](../../status/2026-10-07-qwen-frozen-laptop.json) 中全模型路线同时退化去噪与总时间，候选及编译缓存已清理；不能以“冻结必然更快”重新生成同一方案。

对于本模型，应先保证 GPU 补集足够快，再减少主线程交接、处理首步形状、复用 setup，最后才尝试更复杂量化。现有 base 与 LoRA 的最优通道数/提交 fence 不同，不能统一成一个最大 ANE 占比。

## 11. 验证与拒绝条件

### 11.1 先验证纯语义和内存，避免消耗笔记本算力

使用小矩阵、固定输入或已有 BF16 replay，不加载新的模型：

- 连续通道分区完整覆盖、不重叠，gate/up row offset 和 down column offset 正确。
- `WeightView` 的非紧密 stride、非零 offset、group/packed 对齐。
- 单 adapter 部分输出范围、gate/up 独立 A、strength=0/负值/非 1 的算术；这些测试不等于自动加速资格。
- 多 adapter 的有序 cast 与 aggregate cast 得到各自明确的期望值，不把两者误写成同一个合同。
- 分片 down 与完整 hidden 参考的 FP32 reduction 差异、headroom 还原。
- S1/S2/旋转轴向及 LoRA 修正坐标系。
- row padding、最后不足一桶、长文本/参考图几何及末层 target-only。
- 换 adapter、强度和输入 generation 后不复用旧 low；释放 graph 后闭包不持有整个权重银行。

### 11.2 复用真实输入，逐一筛选候选

复用已有层 0/15/31、早/中/末步骤的输入；base、LoRA 各自独立。一次只替换一个因素：

1. 原 rounding 的事件调度。
2. gate/up 联合投影。
3. 同 adapter low 共享。
4. 分片 down rank。
5. 选择性 epilogue 或新 quant recipe。

每个候选必须同时记录 GPU oracle 和兼容 hybrid oracle 的误差，测试包含权重 staging、A8、LoRA、输出还原及最后消费者完成。实际计时不能因 `async_eval` 提前返回就结束。

### 11.3 完整 block 后才进入两组 E2E

完整 block 包括 QKV、attention、residual、FFN 和下一权重 staging 的相互争用。收益门槛沿用总体实验方案；本章不以单 FFN 倍率替代整个 block 的收益。

Base 完整路径通过后先做 base 配对 E2E；LoRA 在后续独立阶段通过组件门后再做自己的配对 E2E，遵守 [06](06-calibration-performance-and-validation.md) 的每会话预算。相同源模型、prompt/reference、seed、采样和缓存状态配对，关闭近似 DiT cache；分别记录首次请求和可复用 worker 请求。长文本、编辑、多 adapter 没有覆盖时明确保留 GPU。

以下情况拒绝候选，而不是调低报告标准：

- 运行期 fallback、溢出 retry、未完成/取消输出被计入质量或性能成功样本。
- 只在 warm 组件快，但完整 block 或请求变慢；把候选限制为诊断用途。
- 私有 backend 不可用、SDK 能力缺失时，用 GPU 回退计时冒充该 backend 测量。
- RRMS 门槛、非有限值、像素误差或语义检查失败，或只挑有利层/步骤。
- 为优化增加整模型副本、大量磁盘缓存或持续压缩/swap。
- 用 base 校准替代 LoRA 校准，或用 strength=1 的资格覆盖其他强度。
- 将相互重叠的 stage、ANE wait、GPU wait、post-join 时间直接相加来推导收益。

在完整执行路径收益和质量都通过前，不导出 32 层新模型，不改变 App 默认精度或默认打开未验收的混合执行。

## 12. 上游复用边界

- **Splash**：采用通道并行、按实际 rows 选 function、同一图中的 ready/done、权重双槽和提前 staging；其 LLM prefill 样本/硬件上的通道比例不能直接作为 Qwen 策略。源版本固定为 [PR #260 的 AneFfn.cpp](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/ops/AneFfn.cpp)。该 FFN 接口没有直接提供本文所需的 Qwen runtime LoRA correction 合同，必须自行补充。
- **vpipe**：复用显式权重视图、紧凑低秩 metadata、预分配 scratch、按投影选择融合的组织方式；其 Qwen ANE staging 合并 LoRA、FP16 权重与低秩精度路线作为独立 recipe 研究。
- **TurboCider**：复用真实输入捕获、现有 source identity、量化布局校验、FP32 rank、Private W8A8、回退与内存预算；将散落在 graph factory、环境变量和 App profile 中的选择归入 `QwenExecutionPlan`。

最终应能回答每个请求的四个具体问题：本次使用什么模型/adapter 和精度合同；首步与稳态为何选该 GPU/ANE 划分；哪些程序与数据真正复用；完整请求是否在可接受质量和内存下变快。缺少其中任何一项，都不算完整的模型加速方案。
