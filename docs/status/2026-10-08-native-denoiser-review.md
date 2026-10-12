# Native Qwen DiT 生命周期与资源审查（2026-10-08）

本次是当前工作树的静态审查，覆盖 `native_denoiser.hpp/.cpp`、`native_denoiser_math.cpp`、`NativeBlock::bind_workspace`／共享算子，以及关联的 native graph、MLX import 和现有内存账本接口。没有运行 GPU、ANE 或模型，没有新增下载。源码行号是本次审查时的位置，后续编辑可能移动。

**结论：当前请求内的单队列、逐步物理完成模式，没有发现共享 arena 提前覆盖、prefix bank 提前发布或最终输出被下一步重写的问题。** 这不是完整模型数值、性能、故障注入或生产内存约束认证。当前实现是 base BF16 native GPU DiT 基础路径；LoRA、完整 ANE 并行窗口及生产 `MemoryExecutionContext` 桥接不在已完成范围内。

## 已核对的不变量

| 项目 | 实现与判断 |
| --- | --- |
| 几何与权重 | `native_denoiser.cpp:18–40,159–185` 验证 BF16 类型、物理 stride／extent、所有层投影形状、target／text／reference 合计行数；最多两个 reference。`native_block.cpp:92–96` 限定 D128、相同 Q／KV heads 与 FFN 配方。 |
| reference 顺序 | 继续调用现有 `sequence.cpp:7–47`。`text_slot` 必须单调且位于文本范围；target 是最后一个非 causal segment。可接受同一 slot 插入两个 reference；奇偶尺寸偏移和 RoPE 位置使用原几何函数，不另外推导。 |
| causal offset | `native_block.cpp:221–225` 使用 segment 的物理起点作为 causal offset，而非其 RoPE 时间坐标。插入 reference 后二者可以不同；这与已有 Transformer 的分段注意力语义相符。 |
| arena 容量与布局 | `native_block.cpp:142–188` 的容量是 13 个普通矩阵、一个 FFN hidden 及 decode 可选的两个 joined K／V；`bind_workspace` 的 offset 推进与公式一致，末尾再断言相等。各可写区域互不重叠。 |
| 跨层复用 | `native_denoiser.cpp:215–221,289–302` 只分配一个取两阶段最大容量的 arena，所有层复用它。`backend.mm:444–448` 每次 dispatch 后编码 buffer barrier；图内顺序提供上一层读取完毕再覆盖的依赖。不能把此结论扩展到未来多队列或拆开的 ANE 窗口。 |
| prefix bank | `native_denoiser.cpp:223–231` 每层独立分配 K／V；首步构建后逐层 finite 检查。bank 不作为共享跨请求 cache 发布，`prefix_valid` 只在整步 ticket 成功、物理完成且 finite 通过后更新（313–319）。失败会退休该 run，已写入的部分 bank 不会进入下一步。 |
| 双 buffer 更新 | hidden ping-pong 与 latent ping-pong 的输出不 alias 输入。Euler 输出在步骤 `n` 写入 `latent[n % 2]`，上一成功步才改变 current。首步输入是 caller 提供的 retained latents。 |
| CPU 共享常量写入 | `native_denoiser.cpp:240–251` 持有 run mutex 后写入当前 time constants；前一步只有物理完成后才能成功返回，任何失败将 run 退休。因此同一 run 不会在前一步仍读取常量时覆盖它们。 |
| timeout／异常 | `native_denoiser.cpp:312–325` 不把 timeout 当作成功，不强拉事件，也不回放同一个 bank。已提交 graph／ticket 保留所有 BufferView；Backend 的 callback／quarantine 保留有实际或不确定提交的资源。逻辑退休与 backing 释放是两件事。 |
| 最终输出 | `final_latents()` 仅在 `Complete` 返回 retained MatrixView（328–332），后续 `step()` 拒绝（243–245）。run 内不再写输出。caller 仍须遵守输出不可外部改写的协定；类型本身不是只读 GPU capability。 |
| PSO 共享 | `native_denoiser.cpp:124–134` 只准备一组模型级算子；`native_block.cpp:130–139` 校验传入 square／pointwise 算子的几何并复用，循环 32 层时不重新编译 PSO。 |
| BF16 Euler | `native_step_ops.hpp` 对应的 kernel 在 `kernels/step_ops.hpp:120–126` 明确实现 `BF16(latent + BF16(noise * BF16(dt)))`，与现有 `pipeline.cpp:1305` 的 BF16 乘法／加法边界一致。不能改成一次 FP32 FMA 后只舍入一次。 |
| finite 标志 | graph 先 reset，再计算与 atomic-or 检查；`NativeStepOps::require_finite` 只在对应 ticket 成功且物理完成后读取 Shared flag。当前 denoiser 创建独占 flag，并传入自己这次提交的 ticket；通用 helper 本身不推断任意 ticket 与 flag 的关联。 |

外部源的“已产生、不可变、来源一致”仍由 importer／上层 lease 证明。`DenoiserIdentity` 是来源标签，不等于对文件、LoRA 或外部写入者的认证。Backend 必须活到 model／run 最后一次提交；BufferView 保持 backing 存活并不能让已销毁的 Backend 原始指针继续可用。

## 需要保留的限制与后续改进

1. **内存统计尚非物理预算。** 审查时 `allocate_matrix` 在 `native_denoiser.cpp:42–49` 累加逻辑矩阵长度，flag 计 4 字节，arena 计配方长度。Metal 的 `allocatedSize` 可能更大。`DenoiserMemory` 若只用于逻辑 workspace 诊断，应明确口径；若用于实际资源指标，应另外累加每个唯一 backing 的 `Buffer::allocated_bytes()`。这两者都不能替代生产账本 admission。
2. **返回 view 延长请求输出寿命。** `final_latents()` 的强 owner 是必要的。但现有 `MemoryExecutionContext::finish_success` 要求 `storage_count == 0`、`cached_bytes == 0`（`memory_execution.cpp:581–584`）。未来接线必须规定输出的 ownership transfer，或先完成后处理并释放所有请求内 owner，再结束账本事务；不能为了通过检查提前清空 claim。
3. **CPU 常量测试不证明设备 ULP 等价。** RoPE 与现有 Transformer 同用 host float `pow/cos/sin`；time embedding 的三角函数在原 Transformer 由 MLX 求值，而 native 目前由 host 求值。公式、频率精度与 BF16 舍入位置一致，但 CPU oracle 不是实际 MLX Metal trig 的数值证据。设备验证时应检查若干实际 schedule 的 time input／首层及完整 latent 误差，不把同公式 CPU 对照称为端到端通过。
4. **当前 retained cache 仅限一次 run。** 不同 request 的 conditioning、adapter、source generation、reference geometry 没有可复用缓存协议；本实现从头构建 bank，避免错误复用。若添加跨请求缓存，需新增完整 identity 和不可变 owner／失效协议，不能只比较 prefix 行数。
5. **存在可后续衡量的 CPU 开销。** 首轮指出的 `submit(graph)` 整图复制已在复核时改为 `submit(std::move(graph))`。`CommandGraph::append_graph` 每次精确 reserve 扩容；长期可做安全的几何容量增长或预留估计步数。它不是当前正确性缺陷，也没有在本次审查测量收益。
6. **GPU 性能没有由结构自动证明。** 每个 dispatch 的保守 barrier、decode 每层拼接 prefix／current K／V、prefill 全部 prefix 行 FFN 都会消耗带宽／计算。优化时先保留当前安全路径作基准，分别测量，再引入带资源读写元数据的更细依赖、直接双段 attention 或末层 target-only；不可通过删除必要的物理完成边来制造表面提速。

## AllocationHooks 接入现有账本的最小安全路径

当前 Native Backend 没有第二套全局预算。`backend.hpp` 的 `AllocationHooks` 允许调用方提供 reservation／commit token；`backend.mm:243–266` 在分配前以页对齐上界 reserve，拿到实际 `allocatedSize` 后 commit，Buffer 持有该 token。

建议保留现有 `MemoryAdmissionLedger`／`StorageLease` 为唯一物理存储账本：

1. 在请求 owner 线程构建 native allocation adapter，将明确的 allocation label 映射到已编译 manifest 的 site、class、epoch 与实例上限。token 包含 `optional<MemoryReservation>` 及 `optional<StorageLease>`，并携带稳定 allocator domain 和本次 generation。`MemoryExecutionContext::try_reserve_site`（`memory_execution.cpp:165–240`）访问事务状态，不应从 Metal callback 调用；也不能在 site 失败后回退通用 reserve 绕过约束。
2. `allocated` 回调同步把真实 native handle／capacity 提交为 `StorageId`。`memory_accounting.cpp:253–328` 按 domain、handle、generation 去重同一物理 backing，并验证 capacity／class。不能以 view 地址、逻辑矩阵长度或另一个新 domain 给同一导入 backing 重复计费。
3. Buffer 必须持有同一个 claim，直到最后一个 backing owner 释放。`MemoryStageLease::reset`（`memory_scheduler.cpp:71–79`）会立即清 active leases，因此不能让一个短作用域 stage 独占 claim，而 graph／Buffer 只保留裸地址。`StorageLease` 底层 ledger 状态由 mutex 保护，可安全在 callback 引发的最终 owner 析构时释放；析构 token 不应调用已经结束的 raw Context。
4. 如采用 pending retirement，owner 线程在提交前登记 completion token；实际完成 callback 只向 `MemoryStageScheduler::post_completion` 投递，owner safe point drain。`memory_scheduler.cpp:324–359` 即使 status 非零也会完成 pending release，所以**不能把主机 timeout／取消当作一次 completion 消息**。未证明设备完成的资源仍需保留／隔离。
5. Context、scheduler 以及任何捕获它们的回调桥必须活到真正 drain 完毕。当前 C API 成功路径在 drain 后 unload，再要求账本为空（`c_api.mm:261–288`）；失败路径无法 drain 会 quarantine（291–327）。不能让一个晚到 Metal／ANE callback 解引用已销毁的请求 Context。优先让 backing claim 只依赖 shared ledger state，并将 scheduler callback 生命周期单独显式拥有。

### 导入路径不能被一概视为“已计费”

- `mlx_bridge.cpp:53–57` 保留实际 `mx::array` owner。若源来自 `streaming/gguf_storage.hpp:13–42`、`affine_dense_window.hpp:64–74` 这类已经把 `StorageLease` 放进 Data deleter 的路径，原 claim 自然随 backing 保活；不要再 reserve 一个 native view claim。
- 普通 MLX 数组不保证拥有该 claim。`mlx_bridge.cpp:62–75` 的 `allocate_mlx_matrix` 目前直接调用 allocator，没有 admission。`ane_gpu.mm` 的 Surface 分配同样尚无生产账本桥；`import_ane_surface` 保留原 Surface owner、防止重复 wrapper，但不能因此声称 IOSurface 已计费。
- 原 MLX allocator 的 free 可能进入 allocator cache；声称“内存已经释放”时还须符合既有 cache／baseline 协议。回收 wrapper 不等于操作系统已经回收所有容量。编译器、PSO、framework 暂存等也不是 BufferHooks 自然覆盖的资源。
- NativeDenoiser 保留模型级权重 owner；将它们挂到短请求账本后跨请求保留，会违反当前成功收尾协议。必须选择已计费的持久 owner／基线，或沿用 drain 后清空策略，不能两边漏算或重复计算。

因此当前应明确标记：**生产 MemoryContext 桥接未完成；部分来源的已有 claim 能被原样保留，不代表全部 imports、Surface 或模型缓存已参与统一计费。**

## 已修复的真实析构顺序问题及证据

审查发现原 `Buffer::Impl` 的成员逆序析构会先释放 `accounting_lease`，再释放 native 和 external owner，短暂宣告预算空闲但仍保有 backing。本次经授权仅修该 backend 单点：`backend.mm:44–52` 使用 `noexcept` 析构，依次 `native = nil`、`external_owner.reset()`、`accounting_lease.reset()`。不改变 import、Surface lease 或设备执行路径。

新增 `tests/native/metal_graph_buffer_lifetime_test.mm` 直接实例化真实 `Buffer::Impl`，用 CPU NSObject 探针检查自分配 owner、外部 allocator owner、alias 和共享 claim 的析构先后；没有创建 MTLDevice／queue。`tools/native/build_metal_graph_buffer_lifetime_test.sh` 退出码 0。`backend.mm` 用 CLT、C++20、ARC、`-Wall -Wextra -Werror`、macOS minimum 26.2 做严格 host syntax 检查，退出码 0。

这些证据只证明 host ownership 合同与可编译性，不替代 GPU 提交异常、完整模型数值、峰值内存或 ANE placement 验证。

## 第二轮：design/12 对照、drain 可观测性与构建清单

复核 `12-native-denoiser-runtime.md` 后，未发现其 begin 行数上限、首步／稳态、request-local prefix、共享 arena、BF16 Euler 或失败退休描述与当前实现矛盾。该设计已明确 `DenoiserMemory` 是 payload 统计及生产内存接线未完成；源码计数字段仍不应被调用方当作实际容量。

`begin` 的总行数检查在 sequence／RoPE／cache 分配之前执行；text 零行被矩阵布局校验拒绝，故 prefix 至少包含文本，`prefix < full` 且 `full = prefix + target`。两个 reference 即使形状面积相同而宽高不同，仍通过原 `ReferenceGeometry` 生成各自位置；没有按面积复用旧 bank。分配中途异常由未发布的 `unique_ptr<Impl>` 回收已有 owner，未产生可消费的半初始化 run。

发现并经授权修复了一个**完成状态可观测性缺口**：原 Ticket 只保存在 `step` 局部变量中。timeout 后 run 虽然安全退休，Backend callback 继续保有所有已提交资源，但调用方没有该 Ticket 或 run-level drain 接口，不能证明稍后是否已经物理完成。当前 `Impl::last_submission`（`native_denoiser.cpp:145–148`）保存 move-only Ticket，`step` 通过该 Ticket 等待与检查 finite，失败后仍保留（316–329）。新增 `physical_status()` 与 `drain(timeout)`（333–354）：

- 尚无提交时返回 `ok=true, physically_complete=true, submitted_buffers=0`，只表示没有待回收的设备工作，不意味着完成 schedule 或拥有有效输出。
- `physical_status()` 不修改 Ticket 或 run；`drain(0)` 同样只读，避免把 `Ticket::wait(0)` 的真实超时语义误用为轮询。负 timeout 明确拒绝；正 timeout 调用现有 Ticket 的有界等待。
- wait 失败只使 run 进入／保持 Retired；成功或迟到 callback 不改变 Retired，不重新运行 finite，不发布 prefix／latent。`final_latents()` 仍只允许 Complete。若最后一个已成功的 submission 之后在下一步 encode 失败，drain 可能返回物理完成且 ticket ok，但 run 仍是 Retired，这正说明 drain 不能替代结果许可。
- 方法与同步 step 使用同一个 run mutex。正 timeout 约束拿到控制权后的设备等待，不是跨线程抢占正在执行的 step 的总墙钟截止时间；它不是异步取消 API。

新增 API 已通过 CLT C++20、`-Wall -Wextra -Werror`、macOS minimum 26.2 的 `native_denoiser.cpp` host syntax 检查，退出码 0。此修复没有设备实验；tiny fixture 的成功路径检查由主任务另行安排。未来 `Session::drain_memory_completions` 仍必须读取该物理完成证据并完成所有 owner／账本 drain，而非只看 Retired 或 Backend unhealthy。

同样，`step` 整段持有 run mutex，`state()` 和 `completed_steps()` 会等待同步 step 返回，无法在等待期间提供及时的 Running 状态或取消。对当前 W0 同步 API 这不是资源安全错误；接入 App／worker 时需要把 cancellation 与只读状态可见性纳入 Session 层协议，不能宣称已有非阻塞取消。

本次仅清点以下明确属于本轮的构建目录，没有删除。Git 中均无 tracked 文件，目录内没有 symlink，没有模型或推理张量。尺寸是清点时的文件长度合计；`du -sh` 的实际占块显示略大。

| 路径 | 可重建内容与逻辑大小 | 清理建议 |
| --- | --- | --- |
| `build/native-graph-20261008-denoiser/` | 249 文件，31,682,411 字节；`du` 约 33 MiB。包括 231 个 `.o` 共 18,096,224 字节，以及 `ltx-runtime/libltx-runtime.a` 1,616,896 字节。 | 首先可清各层 `.o` 和这一个静态库，保留已链接二进制与 build manifest；代价是下次重编。整个目录也可重建，但当前 component／replay 依赖其中的 `libturbocider.dylib`，继续同构建验证前应保留。 |
| `build/native-graph-20261008-denoiser/module-cache/` | 空目录。 | 可清，当前无空间收益。 |
| `build/native-graph-components-20261008/` | 7 文件，1,658,840 字节；其中 `native-qwen-block-replay.o` 和 `native-qwen-denoiser-component.o` 共 249,520 字节。 | 两个 `.o` 可重建，五个组件／合同可执行文件在证据记录完成且无需继续运行后可清。此目录总量很小，无需为了空间急删当前测试入口。 |
| `build/native-denoiser-cpu-20261008/` | 两个 CPU 合同可执行文件：`lifetime/native-buffer-lifetime-test` 114,728 字节、`step/native-step-ops-contract-test` 37,248 字节。 | 合同结果已经记录，可随时重建；停止使用后可清整个目录。 |

清点对应 runtime build ID 为 `tc-runtime-build-v1-013a2874857ed460bb295048cf4fe086a1fbb2b6639045990f65b007fbdd498d`。若删除整个 runtime-build 目录，先在正式证据中保留需要的 manifest／binary hash，不要只留下已不存在的磁盘链接。

“可清”表示确认是这些 owned 构建产物且可重建，不表示本次已经证明没有进程占用：受当前 sandbox 限制，`ps` 读取未成功。本轮没有为此请求更高权限或执行清理；最终清理由掌握测试窗口的主任务在构建／设备进程结束后进行。
