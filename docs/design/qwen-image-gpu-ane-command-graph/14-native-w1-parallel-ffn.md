# Native W1：完整 GPU／ANE 并行 FFN 接线草案

日期：2026-10-08。状态：**原始接线设计保留；当前 W1S FFN／稳态 32 层／Session／phase 已实现。上一 register 整图验证构建（cache off）一组完整 Base 配对有净收益，仍未授泛化性能或产品资格，见[最新状态](../../status/2026-10-08-native-register-w1-base-e2e.md)。** [较早组件状态](../../status/2026-10-08-native-w1-session-components.md) 保留其原构建范围。实际 API、资源和验收边界以新增 [15](15-native-w1-implementation.md) 为准。本章中的草案 API／双 bank／预取等未由 15 明确列为实现的内容继续是提案。下文“尚待实施”表保留原始设计时的起点，不是当前缺口清单；当前完整后端差距见 [16](16-splash-native-backend-spec.md)，不能当作发布资格。未下载模型。

当前源码新增默认关闭的 compact row-scale cache；生成数据及真实 FFN 的身份／精确复用已通过，完整组件耗时随顺序反转，没有稳定净收益，未追加整图，见[真实缓存筛选](../../status/2026-10-08-native-scale-cache-real-screen.md)。
目标是将完整 FFN 的两条分支接入路线 N：原生 GPU 通道补集与 ANE 分支的 W8 staging、A8 rotation/pack、完整 SwiGLU、restore、FP32 join，以及共同的资源完成域。它不是交接一个小 matmul，也不是每层返回 MLX attention 的 FFN island。现有基础及数学合同见 [11](11-native-block-and-handoff-integration.md)，完整 GPU DiT 的实际接口与边界见 [12](12-native-denoiser-runtime.md)。

## 1. 首个资格域与实施顺序

首轮固定一个几何和一个分区，避免参数扫描：

| 项目 | 首轮候选 |
|---|---|
| 机器 | 本机 M4 Pro、48 GiB；仍须实际 capability 和资源准入通过 |
| 模型 | 已存在的本地 Qwen Image 2.1 dense BF16 checkpoint，BaseOnly |
| 图片／target 几何 | 512×512，packed latent 32×32，稳态 M=1024 |
| FFN | H=4096，F=12288；GPU G=6144，ANE A=6144 |
| 通道方向 | GPU [0,G)，ANE [G,F)，与现有 channel 路径一致 |
| phase | 首步 PrefixBuild 使用完整 native GPU；W1 初期仅 SteadyDecode |
| ANE recipe | 既有 SylvesterDH W8A8、H128 gate/up 与输入、H512 down、seed=20260930；headroom 初始固定 1 |
| 窗口 | 一个在途 ANE 请求，一组输入／输出请求槽 |
| 初期关闭 | LoRA、SmoothQuant S1、A8 single-pass、activation lookahead、权重预取、自动分区调整 |

“1024 行”来自 target token 几何，不是 prompt 长度上限。不同文本改变 prefix 和真实输入分布，仍须保留实际 sequence/source/conditioning 身份；该单候选的测试不能自动覆盖所有 prompt 或编辑输入。

初期不将 1053 等首步逻辑行强行填到某个历史 1056 程序中。首步已有完整 GPU 路径且负责 prefix K/V 事务，先保持它；之后若加入首步 ANE，必须定义 Mpad、GPU padding、逻辑行裁切、不同区段 modulation，以及首步专属验证。Z-image 的 H3840/F10240、post-FFN RMSNorm 和 residual 拓扑另做 adapter，不能套用 Qwen suffix。

W8A8 是 representation/recipe 的名称。程序使用 int8 输入、dequantize 和 FP16 运算图；**不能据此断言硬件实际使用 ANE INT8 arithmetic**。

## 2. 已有基础与真正缺口

| 现有模块 | 可以直接复用 | 本章尚待实施 |
|---|---|---|
| [NativeFfn](../../../native/backends/metal_graph/native_ffn.hpp) | BF16 gate/up、BF16 SwiGLU hidden、FP32 down partial；runtime first_channel，prepare-time channels；原始权重偏移和 pitch | partial join→BF16 与完整 hybrid 编码 |
| [NativeBlock](../../../native/models/qwen21/native_block.hpp) | 完整 projection、attention、norm/modulation、gated residual；validated draft；共享 PSO/arena | before-FFN 与 after-join 分段，完整跨图 validation |
| [NativeDenoiser](../../../native/models/qwen21/native_denoiser.hpp) | 完整 base DiT/Euler、prefix 和 latent 事务；一份 arena、hidden ping-pong | 按层 W1 调度，GPU／ANE／consumer 集合物理完成和取消 |
| [NativeAneHandoff](../../../native/backends/metal_graph/native_ane_handoff.hpp) | 相同 IOSurface backing import，自己提交 producer，fresh events，provenance，成功后 move-only read permit | producer admission／ANE submit／ANE finish 的分离，slot ledger |
| [PrivateW8Graph](../../../native/backends/private/ane_w8_executor.mm) | 完整 W8/A8/SwiGLU/FP32 restore，原 source selection，双权重 bank，S1/cache 规则 | 将现有 GPU stage/restore 算术作为 native graph nodes，统一 tickets |
| [ane_gpu](../../../native/backends/ane_gpu.mm) | W8/A8 kernel、完整 extent/alias/encoding 检查、FP16 scales、signed RNE codes、restore、4 MiB 弱身份尺度缓存 | prepared encode API，避免节点自行 commit |
| [private Program](../../../native/backends/private/ane_program.hpp) | 小型 checkpoint-independent MIL、compiled model 缓存、prepare/submit、callback 物理状态、外部事件绑定 | 接入完整 native W1 的共同身份与完成域 |

现有 PrivateW8Graph::launch_device()/finish() 已经执行完整 ANE FFN 分支。它可用于 native GPU complement 的完整参考路线，且运行时不需要 MLX 数学；但它仍拥有独立 staging/transfer queues 和 legacy 预排 restore。只调用它，或只拆 NativeAneHandoff::finish()，都不能宣称完成本章的 graph ownership、permit-checked restore 和共同完成域。

首个原生 W1 应复用现有 MIL emitter、量化表达式、FP16 scale 舍入、signed RNE、rotation signs 与 restore 乘法顺序。不要为接线重写一套算术、另建模型副本或重复实现 GGUF/affine 格式；首轮只支持 already-produced、immutable dense BF16 source。

## 3. 本地 Splash 完整 Runtime 参考

本轮只读参考仓库是 /Users/kd/Documents/project/splash，HEAD 为 0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7。路径和行号指该本地版本；它是设计依据，不是 TurboCider 的链接依赖。后续若移植源码，须保留对应许可和 NOTICE。

1. **完整模型调用和失败事务**：runtime/model/Runtime.cpp 的 Runtime::prefillAsync，2071–2125 行。它在模型图编码前 AneFfn::begin()，由 split 的 commit() 提交完整图；失败时 GPU 重算，并在结果发布前保持 RNG／KV／状态事务。这说明 FFN split 应属于完整 runtime transaction，不能只给单算子成功回执。TurboCider 应保留自己的 DiT/Euler/prefix 语义，不复制语言模型的 KV/RNG 回滚代码。
2. **完整 FFN 分支组织**：runtime/ops/AneFfn.cpp，addWeights 485–510 行，encode 530–565 行，commit 568–585 行。顺序是当前层 W staging、A8 rotate/pack、ready、GPU 完整补集、下一层 alternate bank staging、done、join；commit 先安排对应 ANE jobs，再提交 Metal graph。可参考其 ownership 和提交顺序，不将一个小桥测试当成完整执行。
3. **图参数、事件与 command-buffer 边界**：runtime/metal/CommandGraph.hpp，17–21、38–60 行；runtime/metal/MetalBackend.mm，commit 775–839 行。参数属于 graph，signal 切分 command buffers。后者 1071–1077 行明确解释 queue capacity 与先创建多 CB 的死锁风险。TurboCider 的 W1 应保持有界提交，而不是将 32 层含 waits 的深窗口一次铺满。
4. **程序源码和 compiled cache**：runtime/ane/Program.mm，constructor 612–651 行。检查 MIL/constants 文件内容，查询已有编译，再 load；它不将每层 master checkpoint 嵌入一个全模型导出。TurboCider 应沿用自己的 SHA256+OS/device/ABI 身份，不改回 Splash 该版本的 FNV key。
5. **计划、校准与真实内存分开**：runtime/engine/AneFfnStartup.cpp，decide 73–122、163–220 行；runtime/engine/RuntimeResources.mm，aneFfnModel 701–747 行、actualMemoryReport 674–698 行。它先资源可行性、再 recall/prepare/verify/measurement，并分别记录 actual allocated、backend/device peak。TurboCider 首轮采用已有历史候选，不开启新的自动扫描；缓存资格仍按自己的 source/recipe/build 身份。
6. **不能直接继承的失败释放行为**：runtime/ane/Handoff.mm，State::retire 42–50 行会提高 event；MetalBackend.mm 823–832 行也有失败时 CPU signal。该释放能让 waiter 退出，不能证明 ANE 已停止写 output，不能授予 TurboCider 的 read/reuse permit。新 W1 继续使用不 force-signal 的外部绑定路线。

Splash 在构造时还于 AneFfn.cpp 448–463 行预计算各层 row scales，降低后续 staging 的重复扫描。TurboCider 已有弱身份 row-scale cache；优先复用其有界缓存，不为模仿预计算流程新增全 checkpoint copies 或一次性全模型硬件扫描。

## 4. 完整数学、布局与 BF16 边界

令 G=F−A。GPU 分支读取 gate/up 的 [0,G) 行、down 的 [0,G) 列；ANE 分支读取 gate/up 的 [G,F) 行、down 的 [G,F) 列。Qwen fused gate_up 的 up 起点仍是原 F×H×2 bytes 偏移，不能在已有 up view 上再加一次 fused offset；down 源 row pitch 始终为完整 F×2 bytes。

NativeFfn 的 channel count 是 MPP specialization，不在每层重新 prepare。准备 G=6144 一套 operators，32 层仅替换源 views。GPU 通道数按 128 对齐；ANE 程序 width 当前按 512 对齐，Sylvester down selection 的 column_begin/count 也按 H512 对齐。6144/6144 满足这些边界。

~~~text
x3 = BF16(LayerNorm_BF16(x2) × BF16(1 + scale_ffn))

h_gpu = BF16(SiLU_BF16(BF16(x3 × Wgate_gpuᵀ))
             × BF16(x3 × Wup_gpuᵀ))
p_gpu = FP32_accumulate(h_gpu × Wdown_gpuᵀ)

p_ane = existing_W8A8_FFN_restore_to_FP32(x3, selected master weights)
f = BF16(FP32(p_gpu) + FP32(p_ane))
out = BF16(x2 + BF16(BF16(tanh(gate_ffn)) × f))
~~~

ANE partial 的 FP32 指恢复后的输出 dtype，不意味着 ANE projection/activation 使用 FP32。其 FP16/quantization/rotation 内部边界继续由独立 recipe 描述。GPU 不先窄化 partial，ANE 不先窄化 partial后再参加 FP32 join；一次 BF16 join 后才使用现有 gated residual。这是新混合 recipe，不能继承普通完整 GPU 的逐 bit 资格。

首轮 headroom=1 固定。若 restore/source status 不通过，不能在未记录 recipe 变化的情况下切换 headroom、重编程序或静默重试。首轮可以明确拒绝该候选或走完整 GPU 路由；如后续支持现有 headroom retry，需独立规定 job generation、程序身份、预算和重新验收。

## 5. 拟议文件与 API

以下是最小落点，不要求先扩展 Backend 的通用 IR：

| 新增或修改文件 | 拟议责任 |
|---|---|
| native/backends/metal_graph/native_ane_stages.hpp/.mm | prepare 既有 W8/A8/restore PSO；typed source/selection/target 验证；只 encode，不提交独立队列 |
| native/backends/metal_graph/native_hybrid_ffn.hpp/.mm | prepared Program、Surface/bank、分区、身份、slot ledger、完整两分支调度；统一退役与 drain |
| native/backends/metal_graph/native_ane_handoff.hpp/.mm | producer admission、ANE submit、ANE completion/read permit 分离；原 finish() 可保留包装 |
| native/backends/metal_graph/native_ffn.hpp/.mm、kernels/dense_ffn.hpp | encode_join_partials(FP32,FP32→BF16)，完整布局与 alias 检查 |
| native/models/qwen21/native_block.hpp/.cpp | plan_parallel 或同等分段 API；三张图事先完整验证 |
| native/models/qwen21/native_denoiser.hpp/.cpp | optional prepared hybrid plan，按层调度、共享 deadline、集合物理完成与整步事务 |
| ane_gpu 的共享小型 helper | 提取当前 W8Params/TransferParams 和 CPU validation plan；保持旧路径兼容，不复制量化数学 |

拟议接口草案如下。类型名只是说明边界；实施时可以收敛为一个专用 controller，但不能省略验证顺序。

~~~cpp
struct ParallelFfnWorkspace {
    MatrixView gpu_hidden;  // BF16 [M,G]，可为既有 F 宽 arena 的合法子视图
    MatrixView gpu_partial; // FP32 [M,H]
    MatrixView ane_partial; // FP32 [M,H]，首轮 M=1024，无 padding
};
struct ParallelBlockPlan {
    CommandGraph before_ffn; // attention/residual/norm；可追加本层 W8/A8
    CommandGraph gpu_ffn;    // 完整 GPU channel complement
    CommandGraph after_join; // FP32 join/gated residual；尚不含ANE输出restore
    MatrixView ane_input;    // 尚未就绪的 x3，不自带 producer 证明
};

// 返回前完成三图和跨图 alias/extent 校验，无设备副作用。
ParallelBlockPlan NativeBlock::plan_parallel(...);

// PSO 在 prepare 外层准备；这些方法只记录 native commands。
PreparedAneStages::encode_weight_bank(CommandGraph&, ...);
PreparedAneStages::encode_activation(CommandGraph&, ...);
PreparedAneStages::encode_restore(CommandGraph&, const NativeAneReadPermit&, ...);
NativeFfn::encode_join_partials(CommandGraph&, gpu_f32, ane_f32, output_bf16);
~~~

NativeBlock 的切点是现实现中 normalized_ffn 产生之后、gate/up 之前。after_join 只接 FP32 join 的 BF16输出及既有 gated residual，不能重写 LayerNorm 或把 FP32 branch直接送进 residual。现有 encode() 继续作为完整 GPU recipe。控制器在提交前校验拟用restore的几何、尺度与目标；该kernel只能在获得read permit后编码到新的consumer draft，再append已验证的after_join，不能在plan_parallel阶段假造permit。

stage bundle 沿用完整范围、source/metadata/target alias、device、pitch、overflow和 scale 检查。首轮不开放任意 encoding 或任意用户环境开关。source owner/generation 保持到全部设备消费者完成；cache identity 不能仅取 raw MTLBuffer地址。所有 library/PSO 在热路径外 prepare，数量及容量计入共享 operator bundle，不能每层各建一套。

## 6. W1 实际提交和 readiness

~~~text
1. acquire 单请求槽和可写 weight bank，检查最后消费者已物理完成。
2. 建立本层完整 typed plan；fresh status 在任何提交前初始化为 0。
3. before_ffn + 当前 W8 gate/up/down + A8 rotation/pack
   → handoff.submit_producer()，由 handoff 自己追加 fresh ready signal。
4. join handoff 实际 captured producer ticket：ok、physical、未退休。
5. 检查该 producer 写入的 W8/A8 status=0，bank/输入代际匹配。
6. prepare/submit 本层 ANE request；此时再提交 native GPU complement。
7. 收集 ANE callback和完整 provenance；成功才授予 read permit。
8. 在 permit后编码并提交 restore→FP32 join→BF16→gated residual。
9. 所有 GPU／ANE／consumer成功且实际物理完成，才使该槽可复用。
~~~

ANE 已提交与实际 ANE engine开始执行不是同一事实。第6步保证自己的提交顺序；真实重叠和关键路径须通过组件及整图记录验证，不能只凭 event或host时间名称声称已经重叠。

当前 NativeAneHandoff::finish() 将 producer等待、ANE submit和ANE等待合在一起。建议拆为 finish_producer()、submit_ane()、finish_ane()；producer permit由 handoff根据它自己提交的 ticket产生，stage controller核对 flags后生成已验证输入许可。最终实现也可在专用 start_ane() 内完成这些步骤，但不能接受一个任意外部“已完成 ticket”或未受控的验证回调。

GPU command成功与 GPU status=0是两项独立条件。ready=1不能替代它们；量化kernel遇到非有限源值或非法尺度时仍可能正常结束 commandbuffer并置status。状态buffer及检查必须与本次 producer/slot generation绑定，提交 ANE后才检查已经太晚。

不能把 GPU complement或等待 ANE输出的 restore放进 captured producer图。handoff等待的是整张ticket，midpoint signal不会缩短它；同时未提交的 signal与预排 wait可能造成死锁。首轮允许 producer成功后的host admission，然后GPU与ANE完整分支并行；这是 W1，不是无逐层host admission的 W2/W3。

单 native queue上，后续 join可以依靠已提交 GPU complement的队列顺序保证读取顺序；首个保守实现也可以显式等待两个分支。无论如何，最终成功与物理复用判断仍必须包含 complement ticket，不能因为 after_join完成就丢弃一个失败的前序 ticket。

准备／模型自检计时与请求计时分开，自检每个 prepared bundle执行一次。多个设备等待消费同一个 step绝对deadline，不能在32层逐次重新获得30秒预算。取消只在可证明安全的边界停止新的提交；logical取消不表示已提交ANE停止访问。

## 7. 槽位与集合物理完成

初期 slot状态：

~~~text
Free → Producing → Ready → ANEReading → Restoring → Joined → Free
任一步失败／超时／取消 → Retired（结果资格不可恢复）
~~~

这是一个在途请求窗口。现有双W bank只是允许将来准备alternate weights，**不代表具备两组可并发写入的 x/y/correction/output槽**。若之后增加一层W预取，只允许写alternate bank，保存其独立producer，并在使用前核对完整source/recipe/generation；当前bank直到ANE读取和sd restore消费者结束前不能被覆盖。首轮不启用该优化。

各资源的最后消费者不同：

| 资源 | 最少持有至 |
|---|---|
| source weights／signs／可选profile向量 | 所有 W8 producers与实际使用者物理结束 |
| x/tx、gate/up/down codes、sg/su | ANE request实际结束 |
| down sd尺度 | 对应restore GPU consumer物理结束 |
| ANE y／hs | 所有restore GPU consumers物理结束 |
| x3／GPU hidden／FP32 GPU partial | GPU补集和join最后消费者物理结束 |
| FP32 ANE partial／x2／FFN modulation | join/residual最后消费者物理结束 |
| prefix K/V、hidden/latent ping-pong | 整步成功，及所有已提交使用者实际结束 |

共同完成域记录 producer tickets、GPU complement ticket、private request ticket、restore/join ticket、source/bank/slot租约和epoch。status至少区分“所有已知提交物理完成”“结果可发布”“已退休”“未知物理状态被隔离”，不能用一个ok布尔值或cpu worker返回代替。

当前 NativeAneHandoff 的 all_work_physically_complete仅覆盖producer与ANE，不包含restore/join；Denoiser目前只有last_submission。W1必须改为集合状态；step成功后可折叠已完成历史记录，未完成或退休提交继续保留，不无限积累。

未知物理状态时隔离program、surfaces、events与bank，不reset status、不覆写、不发布部分结果。晚到callback可更新physical状态，不能复活许可。缺callback导致private健康域关闭后，不能再增加第二个被隔离请求。

首轮最简单的失败策略是退休该step并返回明确错误；preflight不合格可在提交前选择完整GPU。若增加提交后GPU fallback，必须完整FFN重算、使用明确安全且独立的scratch，保存所有旧消费者，记录实际fallback；不能将失败ANE partial加进GPU结果。native GPU backend若因提交/timeout已不健康，不能假定同一queue仍可重算。prefix/current仍只在整步成功、物理完成、finite检查后发布。

## 8. 内存：payload、实际分配与准入初估

下表是 M=1024/H4096/A6144 的布局计算，不是物理ANE scratch、进程峰值或已分配实测。IOSurface pitch、page capacity和driver/compiler workspace仍须单列。

| 项目 | payload／布局初估 |
|---|---:|
| 双W8 bank三份codes：2×3×H×A | 144 MiB |
| 双bank row-scale surfaces，以每行64bytes计算 | 2 MiB |
| A8 x int8 [H,M] | 4 MiB |
| tx FP16 [1,M] | 2 KiB逻辑payload |
| BaseOnly y FP16 [H+1,M] | 约8 MiB |
| 两块FP32 [M,H] partial | 32 MiB |
| GPU BF16 [M,G] hidden | 12 MiB逻辑payload；可复用现有24 MiB F宽slot |
| joined BF16 [M,H] | 8 MiB逻辑payload；复用既有ffn_output |
| rotation constants | A×512×2+128 = 6 MiB+128 B source constants；Program内部容量另计 |
| row-scale cache | 有界4 MiB，weak source identity |

现有PrivateW8Graph还使用保守估算 H×A×6 + M×(8A+12H) + 两份constants + 128 MiB workspace allowance + 4 MiB cache。以本几何计算为384 MiB+256 B，是准入allowance而非实测。原生W1另有两个FP32 partial；初期可按约416–512 MiB增量预算进行保守规划，并在分配时检查actual bytes，不能将公式直接当物理峰值或内存安全保证。

已有native block arena无prefix约128 MiB，P=29稳态约144.45 MiB；32层prefix另约14.5 MiB。该arena保留一份复用，W1 partial使用独立合法FP32 storage。imports复用同一个IOSurface/MLX backing，不能重复计费；保留owner延长生命周期仍可能提高峰值。master checkpoint、encoder/VAE、allocator cache和driver容量不在上述增量数字里。

MemoryExecutionContext/reservation接线要覆盖Surface factory及program allowances，并分别记录requested payload、实际allocated capacity、reservation、backend/device peak和进程footprint。DenoiserMemory目前只是payload统计，不能据此宣称48 GiB机器必然安全。source import不得在此阶段重新eval、dequantize或创建第二份完整checkpoint。

## 9. 本地缓存复用边界

本轮读取到的保留program source是：

~~~text
outputs/qwen-real-ffn-20261007/private-cache/
  3683c56e3f76fbeddadad16f318a0007cabb6fdd023246f88dfa6214a725622f/
    model.mil
    weights.bin
~~~

它的H4096/A6144/M1024正确，但含dg/du correction inputs和hidden输出，是LoRA超集ABI，不能作为精确BaseOnly cache hit。不要为了命中它让base默认分配LoRA correction/hidden scratch。保留weights.bin实际为6291584 bytes，即6 MiB+128 B；A4096才对应4 MiB+128 B。新的A6144 BaseOnly program只需小型MIL和同几何旋转常量，权重仍由现有source staging供应；这不是完整模型export。compiledModelExistsFor实际确认后才记录编译缓存命中，并检查精确MIL/constants、OS/device/ABI。

编译Program cache与执行资格必须分开：现有compiled key来自MIL/constants、OS build、device和ABI；不包含checkpoint内容、adapter strength或GPU侧S1向量。source/adapter/profile、native shader与stage recipe属于execution/stage/qualification身份。S1在GPU staging上变化而MIL/constants未变时，不要求强行生成不同compiled Program key，但必须使相应stage缓存及执行资格失效。LoRA启用若改变dg/du和hidden输出ABI，则因MIL变化产生不同Program key。

outputs/local-512-20261004、outputs/runtime-perf-20261005的旧临时program cache已经清理；保留inventory或历史receipt不代表文件当前存在，也不代表OS当前仍有可load的compiled model。W0 handoff小型测试cache也不是完整FFN program资格。

已有base-captures中的8行、3层、3时间点FP16输入和statistics可复用为校准线索／小型oracle。它们明确标记performance_sample=false、quantization_qualified=false，来源是旧forced-GPU runtime boundary；不能充当新native完整block、1024行或全模型分布证据。

S1完整profile的CPU连接记录不等于GPU/ANE质量和性能资格。初始W1不用S1；以后启用必须在execution/stage/qualification身份中包含profile digest/每层scale identity，W gate/up乘S1与A8输入除同一S1，down不使用S1，并更新stage缓存及receipt。只有程序内ABI/recipe改变时才相应更新compiled Program key。

row-scale cache仅保留尺度metadata，codes仍需每层重新生成。source generation、physicaloffset/pitch、selection、basis、S1及inverse都参与key；弱cache不能保留GPU input/profile tensor。严禁为省重复codes staging将32层全部W8 codes额外驻留：本几何约2.25 GiB codes，会扩大本轮低内存范围，也不是当前方案。

## 10. 历史收益不能授予 native W1

[refinement-screen.json](../../../outputs/qwen-real-ffn-20261007/refinement-screen.json)记录的旧MLX完整FFN回放是：

| 旧候选 | GPU均值 | hybrid均值 | 比率 |
|---|---:|---:|---:|
| Base Hadamard A4096，无launch fence | 41.895 ms | 40.880 ms | 1.025× |
| Base Hadamard A6144，有launch fence | 41.888 ms | 27.884 ms | 1.502× |
| Base Hadamard A6144，无launch fence | 41.871 ms | 39.835 ms | 1.051× |
| Base S1+Hadamard A6144，无launch fence | 41.895 ms | 40.572 ms | 1.033× |

这些完整FFN wall包含该probe每次weights staging与join，但不是attention/residual、整图或physical overlap证明；使用历史build及捕获输入，不能直接给当前native代码授予性能资格。相同6144分区的提交顺序有明显差异，支持优先保证A8/ANE先提交，再安排GPU complement；不证明ANE kernel单独更快，也不说明S1必然无效。

历史native真实block v3的约6.4% prefill、3.9% steady收益也不是hybrid结果，且不是最终v4或E2E资格。新方案必须至少比较同源native GPU-only与native W1，同时保留既有最佳GPU/hybrid结果作为参照；更慢的native GPU baseline不能制造表面ANE倍率。

## 11. Base之后的LoRA与量化扩展

顺序为：Base完整W1 → native GPU-only LoRA oracle → W1 LoRA → S1/其他量化recipe → 可选一层权重预取 → 更深窗口。Base不默认准备LoRA超集surfaces。

LoRA接线必须覆盖实际checkpoint的所有target。若仍不支持attention、输入／时间／调制／输出projection中的adapter，应明确拒绝组合，不能只处理FFN后宣称完整LoRA支持。

FFN LoRA保留三项基本规则：

1. 共享x·A属于本次FFN、此输入和ordered adapter集合；生命周期至本次所有consumer完成，不跨模型或输入误复用。gate/up的disjoint B范围按相同分区分别计算，保持Qwen fused/分离adapter的顺序、strength、逐adapter FP32 add/BF16 cast语义。
2. gate/up correction在SiLU之前。ANE recipe如经FP16 correction surfaces传入，需明示新增舍入域；不能用GPU correction终端hash冒充整个ANE分支等价。
3. down correction使用完整hidden、只加一次；不能在GPU和ANE各加一次。需要ANE hidden输出及恢复，与GPU hidden组成合法完整F宽view，再执行rank correction并按既有边界窄化。

先验证本地已存在的r128 adapter，不下载新LoRA。额外correction、hidden输出、rank scratch和租约全部入账。execution/stage/qualification身份包括base source、adapter源内容与顺序、strength、rank/target/layout、profile、recipe和geometry；compiled Program key依然按实际MIL/constants、OS/device/ABI区分。S1若启用，x·A仍用原始未缩放输入坐标；scale profile与adapter分布适用性另验收。

## 12. 轻量验收、计时与停止条件

本章只定义下一轮计划，不代表授权现在运行设备或模型。执行由主任务统一安排排它窗口，保持笔记本最多1–2组实验、少量E2E，不常驻高频校准。

1. **CPU／host先行**：几何、fused offset、原down pitch、overflow/alias；全计划validation失败不部分append；owner生命周期；producer flags；fresh event与代际；部分提交、取消、超时、晚callback；最后consumer未结束槽不复用。两块FP32 partial的CPU oracle验证一次BF16 join。纯CPU测试不证明Metal shader、实际ANE执行或设备并行。
2. **一次有界真实组件**：复用Session transient完整block observer、原compiled GPU callable和同一source权重，不落盘大张量。只取step1 steady block0实际input/mods/RoPE/prefix；1 warmup+2反向配对样本，比较native GPU-only与完整native W1 block。不把已有post-attention FFN capture称完整block输入；不只计ANE alone或restore小kernel。
3. **准备与热路径分开记录**：PSO/MIL编译、自检、首次cache miss单列；每次wall包含当前W staging、A8、host admission、GPU complement、ANE等待、restore/join/residual。同步与allocation按实际发生位置计时。记录实际layer/rows/prefix、input/cache logical SHA、selected与actual路线、fallback、原始samples、stage cachehits/misses。逐次native输出复现检查在计时外，不能仅warmup筛查后假定timed输出正确。
4. **组件通过后再E2E**：先保证完整native GPU-only请求基准和资源合同，随后同prompt/seed/20步比较GPU-only与W1；必要时GPU重复一张，总2–3张，不扩矩阵扫描。保存latent/image差异、实际build/recipe/source、首次与warm请求、thermal/swap和进程/allocator峰值。

出现timeout、非有限值、未预期fallback、身份/cache错误或physical复用疑点立即停止，不扩大实验或自动调参。沿用并明确记录现有研究数值门槛，不在新工具中静默放宽；约1% FFN误差是历史候选的可参考目标，不是全步／图像质量保证。用户接受候选探索的约5–8%差异也不免除算子完整性、同步和最终图像比较。

完整block收益不足约5%，或噪声大到不能确定收益时，先分析device spans/host gaps与资源竞争，不直接升为默认配置。各host等待span互相重叠，不能相加当GPU/ANE engine时间；更少A8 exposed wait也可能只是等待移动到其他span。任何结论以完整wall、实际执行和质量/内存证据为准。

首个W1通过后才考虑把一个next-layer W8 producer放在当前GPU complement之后、ANE执行期间，用alternate bank及独立ready ticket隐藏staging。这个优化仍不增加第二个在途输出请求。W2/W3、无逐层host admission和预排输出restore，需要更强的device-driven失败证书与slot完成域，不在本章首轮范围内。
