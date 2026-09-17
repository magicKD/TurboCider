# 38 · Streaming Framework 代码合同、实现工作台与逐文件施工设计

[目录](README.md) · [Adapter 代码规格](36-public-adapter-code-implementation-spec.md) · [校准与发布验收](37-public-streaming-calibration-performance-acceptance.md) · [验收工作簿](39-validation-benchmark-and-release-workbook.md)

修订日期：2026-09-17。状态：**实现工作台；其中“拟议”内容不代表已经存在的代码或 public 资格。**

本文把 01–37 中已经确定的原则进一步落到“应该创建哪些类型、谁拥有对象、谁可以在哪条线程调用、失败后怎样回收、每个 PR 修改哪些文件”的级别。实现者可以把本文当作 C1 之后的代码施工单；如果本文与现有源码不一致，应先更新本文的“当前事实”段，再修改代码，不得用文档掩盖代码差异。对于 C1/C2 的具体 API，本文是对 36 第 4–5 节概念草图的细化；发生冲突时，应先同步修改 36，再以本文件冻结的单一 fd lineage 和 receipt 接线为准。

## 1. 当前事实、目标与禁止事项

### 1.1 当前事实（2026-09-17）

当前 `feat/stream` 已经具备：

- `LayoutCompiler`、`StageExecutor`、`SlotSafetyTracker`、`IoExecutor` 和 C ABI v1/v2/v3；
- ordered multi-class barrier、single-pool K2 `carry_first_group`、claim 后 refill overlap 和 synchronous reader completion fast path；
- `StreamingSelector`、catalog record、canonical digest、deterministic resolver、internal-only authority；
- `PublicStreamingCoordinator`、immutable catalog snapshot、一次性 `PublicStreamingPreflight` ticket；
- `RunResult` 的 public selection metrics 和汇总型 actual-plan verifier；
- LTX、Z-Image、H3 Turbo、Flux 9B 的 private descriptor/streaming 实现和相应的 test/benchmark 证据。

当前**尚未完成**：

- 通用 fd-based source lease、value probe/snapshot 和 receipt v2；
- 四个真实 `ModelSession` 的 `probe_public_streaming`、`compile_public_streaming`、`generate_resolved` override；
- per-pass/group fill、logical bytes、reader fence、source generation 的硬校验；
- App engine-scoped options、`JobStore` v2、LTX worker 两阶段握手；
- 8/10/12/16/20 GiB 的完整 process-tree 校准、swap 对照和 reviewed production record；
- ANE streaming public 认证。

### 1.2 目标

目标是增加一条 opt-in public exact 路径：

```text
用户 selector -> request validator -> catalog snapshot -> probe
-> immutable snapshot + source lease -> authority
-> 既有 exact StageExecutor/reader/kernel -> actual receipt
-> independent verifier -> result
```

除上述路径外，resident、compiled、GPU+ANE、legacy `residency=streamed`、private candidate 和普通 H3/VDN 均保持原有入口和行为。

### 1.3 禁止事项

以下行为即使“能够跑起来”也不允许合入：

1. 在 `StageExecutor` 内读取 memory tier、catalog 或 App 状态；
2. public generate 重新根据用户 selector 编译一个未授权 layout；
3. adapter 在 `generate_resolved` 内按可替换 path 重新打开 checkpoint；
4. 为了 fit target 自动改变 dtype、shape、LoRA、quantized cache 或组件策略；
5. 将 `StreamingAuthority`、fd、GPU pointer、worker handle 写入 JSON/JobStore；
6. 发生 drain 不确定时直接析构 pool 或复用 engine；
7. 用 private candidate 的汇总 metrics 冒充 public receipt；
8. 把 simulator、单进程 MLX peak 或一次成功 smoke 当成 target 资格。

## 2. 目录与依赖方向

### 2.1 目标目录

以下是**拟议新增或逐步归并**的目录，不要求一次性创建所有文件：

```text
native/runtime/streaming/
  canonical_encoding.*        # 已有，所有 identity/digest 的唯一编码
  layout.*                    # 已有，纯 metadata compiler
  slot_pool.*                 # 已有，slot safety 与 ticket
  io_executor.*               # 已有，固定容量 refill workers
  context.*                   # 已有，StageExecutor owner pump
  c_bridge.*                  # 已有，C ABI adapter bridge
  public_request_validation.* # 已有，request-only validator
  preset_catalog.*            # 已有，record/schema/target resolver 基础
  preset_resolver.*           # 已有，select/authorize
  catalog_provider.*          # 已有，immutable snapshot provider
  public_runtime.*            # 已有，preflight/resolve/revalidate
  public_result.*              # 已有，汇总 verifier；待扩 receipt v2
  source_lease.hpp/.cpp        # C1 新增，fd + generation + TOCTOU
  value_probe.hpp/.cpp         # C1 新增，通用 metadata-only probe/snapshot
  actual_receipt.hpp/.cpp      # C2 新增，逐事件 receipt/expected verifier
  memory_scope.hpp/.cpp        # C4 新增，完整请求 process-tree scope
  trace.hpp/.cpp               # C4 新增，低开销事件采样（不进入 Off）
```

依赖必须单向：

```text
core Request/Recipe
        -> public_request_validation
        -> catalog_provider + preset_resolver
        -> PublicStreamingCoordinator
        -> ModelSession probe/snapshot hooks
        -> StageExecutor/slot_pool/io_executor
        -> actual_receipt/public_result
        -> C ABI/Swift/App serialization
```

`StageExecutor` 只能依赖 `Layout`、`ModelSlotAdapter` 和取消/完成协议；不能反向依赖 `ModelSession`、catalog、Swift 或 memory-tier selector。

### 2.2 值对象与可变对象边界

| 对象 | 可变性 | 所有者 | 生命周期 |
|---|---|---|---|
| `StreamingSelector` | 输入值 | Request/App | parse 到 resolve |
| `StreamingPresetRecord` | immutable snapshot | Catalog | 进程/请求 |
| `PublicStreamingPreflight` | move-only | Coordinator | preflight 到 resolve |
| `ModelStreamingProbe` | immutable | Resolved execution | resolve 到 generate 结束 |
| `SourceLease` | fd 状态不可替换 | Snapshot | probe 到 drain 后 |
| `ModelStreamingSnapshot` | immutable | Resolved execution | resolve 到 drain 后 |
| `StreamingAuthority` | immutable、不可复制 | Resolved execution | resolve 到 result |
| `StageExecutor` | owner-thread mutable | Model adapter | begin 到 finish/drain |
| slot/pool/ticket | owner/worker 协同 | StageExecutor | 一个 request |
| `ActualExecutionReceipt` | owner-thread append/update | StageExecutor | begin 到 verify |
| C/Swift result | immutable serializable | API caller | 返回后 |

核心原则：**只有 metadata 值对象跨线程共享；slot backing、fd duplication、GPU fence、worker mailbox 都不跨请求共享。**

## 3. C1：SourceLease 详细合同

### 3.1 数据结构

下面是建议的实现形态；字段名可以按现有命名调整，但语义必须保持：

```cpp
struct SourceFileIdentity final {
    std::string logical_id;          // stable: transformer.shard.0
    std::filesystem::path path;      // canonical absolute path
    uint64_t device = 0;
    uint64_t inode = 0;
    uint64_t bytes = 0;
    int64_t mtime_ns = 0;
    int64_t ctime_ns = 0;           // catches same-size payload rewrite
    std::string header_digest;       // optional, metadata/index header only
    std::string manifest_digest;     // optional, sorted manifest identity
    std::string content_digest;      // optional cached/import-time evidence
};

struct SourceLeaseDescriptor final {
    std::string source_snapshot_digest;
    std::vector<SourceFileIdentity> files; // sorted by logical_id
};

class OwnedSourceFd final {
public:
    explicit OwnedSourceFd(int fd) noexcept;
    OwnedSourceFd(OwnedSourceFd &&) noexcept;
    OwnedSourceFd &operator=(OwnedSourceFd &&) noexcept;
    ~OwnedSourceFd();
    int get() const noexcept;
    int release() noexcept;
};

class SourceLease final {
public:
    static std::shared_ptr<const SourceLease>
    open_and_verify(const SourceLeaseDescriptor &);

    SourceLease(const SourceLease &) = delete;
    SourceLease &operator=(const SourceLease &) = delete;
    OwnedSourceFd duplicate_fd(std::string_view logical_id) const;
    void revalidate_paths() const;
    void revalidate_open_files() const;
    void revalidate_after_drain() const;
    uint64_t generation() const noexcept;
    std::string_view digest() const noexcept;
};
```

`OwnedSourceFd` 是 move-only RAII 类型，析构关闭 fd；只有在调用现有 C reader API 的最后一层才允许 `release()` 为裸 `int`。禁止让 `duplicate_fd()` 返回一个所有权不清晰、异常路径可能泄漏的裸 fd。

`source_snapshot_digest` 是运行时 snapshot identity，不得伪装成全文件 content hash。`artifact_manifest_digest`/`content_digest` 负责“这是哪个模型内容”的资格锚点；device/inode/size/mtime/ctime 负责“本次运行读到的仍是同一文件代际”。full-file hash 可以在模型导入或首次受控 model-open 时计算并缓存，不能在每个 Off/default 请求中重复扫描大权重。

### 3.2 打开与校验顺序

`open_and_verify` 必须按以下顺序执行，任何一步失败都不返回部分 lease：

1. 校验 `logical_id` 唯一且按 canonical order 排序；
2. 对每个 path 使用 `O_RDONLY | O_CLOEXEC` 打开；
3. `fstat` 检查 regular file、device、inode、size、mtime、ctime；
4. 读取必要的 index/header 并核对 `header_digest`/manifest；
5. 计算 source snapshot digest；
6. 生成 request-scoped、非零、单调的 `generation`；
7. 所有 artifact 成功后才构造 immutable lease。

任何中途失败都必须关闭已经打开的 fd，并返回稳定错误（`source_open_failed`、`source_metadata_changed`、`source_digest_mismatch` 或 `source_duplicate_logical_id`）。

### 3.3 单一 fd lineage 与四个校验点

推荐的 production 实现不是“probe stat 一次、snapshot 再按 path 打开一次”，而是：

```text
probe capture/open lease once
-> metadata parser 只从 lease fd/pread 读取
-> probe 持有 shared immutable lease
-> snapshot 共享同一 lease
-> exact reader 从 lease duplicate fd
-> drain 后做最后一次 open-fd revalidate
```

这样同时减少重复 open/header parse，并消除 probe 与 snapshot 之间的 path reopen 缝隙。若 C1 第一小步暂时保留“probe descriptor -> compile open_and_verify”，它只能作为安全但较慢的过渡实现；进入第一个真实 public adapter 前必须收口为单一 fd lineage。

| 时机 | 校验 | 允许的副作用 |
|---|---|---|
| probe capture | path + fstat + metadata + header/index | 只读，不分配 GPU |
| snapshot compile | 同一 lease 覆盖全部 descriptor source range | 只构造 metadata/layout |
| pre-GPU revalidate | named path 与 opened fd 的 inode/size/mtime/ctime | 不新开 reader，不提交 GPU |
| post-drain revalidate | opened fd stat/content generation 未变化 | 只决定 result 是否可接受 |

same-size mutation 至少要由 ctime 变化捕获；若测试能恢复 mtime，也仍必须失败。若威胁模型要求抵抗能够伪造完整 stat tuple 的外部修改，则必须使用 import-time content digest/只读模型仓库或平台文件保护；不能只靠 header digest 声称完整 payload 内容不变。

实际 reader 必须使用 `duplicate_fd(logical_id)` 或现有 fd-backed API。若某个旧 C API 暂时只能接 path，只能在迁移分支用受控 `/dev/fd/<n>` 兼容层，并在 adapter record 中标记 `reader_revision`；production v1 不接受“重新按 path 打开”的实现。

`source_lease_verified=true` 的含义应冻结为：pre-GPU 和 post-drain 两次校验都成功，且 receipt 的 source generation 与 lease generation 相等；仅在 resolve 时做过一次 stat 不足以设置该字段。

### 3.4 两类 source closure

必须区分：

1. **streaming execution lease closure**：所有被 refill worker/pager 异步读取的 shard/index；这些必须 fd-leased；
2. **qualification source closure**：所有影响完整输出与内存峰值的 encoder、denoiser、VAE、upsampler、audio、LoRA/quant cache；它们必须进入 artifact manifest、workload/component policy identity，即使没有进入 streaming pool。

只租赁 transformer shard、却不绑定实际 VAE/encoder revision，不能生成 public record。

### 3.5 现有代码复用要求

C1 不应重新发明四套 stat 逻辑：

- `mlx_weight_pager.cpp` 的 open/fstat/device/inode/path replace 检查提取为共享 helper；
- LTX `StreamingMetadata` 的 descriptor snapshot 作为单文件 lease 参考；
- H3 shard identity、Z-Image 单文件、Flux 多 shard identity 统一转换到 `SourceLeaseDescriptor`；
- 原有 model-specific `check_unchanged()` 保留作内部防御，但 public revalidate 通过统一 lease 调用。

## 4. C1：Value Probe 与 Snapshot 详细合同

### 4.1 Probe 只做什么

Probe 是 metadata-only。它可以读取：

- model/index/header/manifest；
- source artifact stat 和 header digest；
- workload shape、tokens、frames/fps/steps/audio；
- runtime/adapter/reader/kernel/allocator revision；
- component policy revision。

Probe 不可以：

- 创建 MLX array、Metal buffer、Core ML compiled graph；
- 启动 I/O worker 或 `StageExecutor`；
- 调用 `synchronize`/`cache_clear`/`unload`；
- 修改 request shape/dtype；
- 创建 authority 或写 catalog。

### 4.2 值实现

```cpp
class ValueModelStreamingProbe : public ModelStreamingProbe {
public:
    struct Values {
        std::string model_id;
        PresetSourceIdentity source;
        PresetWorkload workload;
        PresetRuntimeIdentity runtime;
        std::string component_policy_revision;
        std::shared_ptr<const SourceLease> source_lease;
    };

    explicit ValueModelStreamingProbe(Values);
    const SourceLease &lease() const noexcept;
};

class ValueModelStreamingSnapshot : public ModelStreamingSnapshot {
public:
    struct Values {
        std::string model_id;
        PresetSourceIdentity source;
        PresetRuntimeIdentity runtime;
        Descriptor descriptor;
        Layout layout;
        std::string component_policy_revision;
        std::shared_ptr<const SourceLease> source_lease;
    };

    explicit ValueModelStreamingSnapshot(Values);
    void revalidate_source() const override;
    const SourceLease &lease() const noexcept;
};
```

### 4.3 Snapshot 不变量

构造 snapshot 时必须一次性检查：

```text
descriptor.model == model_id
descriptor.checkpoint_identity == source.source_snapshot_digest 的映射值
layout.digest 非空且已 finalized
layout.materializations_complete == true
source/runtime 与 probe 完全相等
lease.digest == source.source_snapshot_digest
layout 只含 public v1 允许的 streamed stage
retention == request
pass/pool policy 属于 adapter capability
```

Snapshot 构造失败时不得留下半初始化的 `StageExecutor`、pool 或 worker。建议采用“先构造纯值、最后 attach lease”的 builder，避免异常路径泄漏 fd。

Probe 与 Snapshot 应共享同一 `SourceLease`；Snapshot 不按 path reopen。模型专用 metadata（例如 H3 store view、Flux tensor index、LTX safetensors mapping）不得塞入 `void *`/`std::any`。通用 value 类可以作为非 final 基类，四模型使用显式 final 派生类型保存 model-specific immutable payload；`generate_resolved` 对预期 snapshot 类型做 fail-closed cast。这样既复用 identity/revalidation，又不牺牲类型安全或重复解析 header。

### 4.4 Cache 规则

允许缓存 probe metadata，但 cache key 至少包含：

```text
model root identity
all artifact (device,inode,size,mtime_ns)
manifest/header digest
build/runtime/adapter/reader/kernel/allocator revisions
execution container
workload feature digest
```

缓存不得包含 source fd、generation、authority、slot backing、worker 或 cancellation token。Off/default 分支必须在入口短路，保证上述 cache lookup 也为零。

## 5. C2：Actual Receipt v2 与 StageExecutor 接线

### 5.1 Receipt 结构

```cpp
struct ActualFillReceipt {
    uint32_t pass = 0, group = 0, pool = 0, slot = 0;
    uint64_t ticket_generation = 0;
    uint64_t logical_bytes = 0;
    uint64_t source_generation = 0;
};

struct ActualStageReceipt {
    std::string stage_id, layout_digest;
    uint32_t completed_passes = 0, completed_groups = 0;
    uint64_t fills = 0, groups_submitted = 0;
    uint64_t logical_read_bytes = 0;
    uint64_t reader_fences_issued = 0;
    uint64_t reader_fences_completed = 0;
    uint64_t source_generation = 0;
    bool drain_completed = false;
    std::vector<uint32_t> fills_per_pass_group;
    std::string event_digest;
};

struct ActualExecutionReceipt {
    uint32_t schema_version = 2;
    std::string implementation, layout_digest, component_policy_revision;
    std::vector<ActualStageReceipt> stages;
    std::string canonical_digest;
};
```

`fills_per_pass_group` 在 `begin()` 时根据 sealed layout 一次性分配；不能在 worker 线程中 `push_back`。如果模型 pass/group 数量超出 public v1 的上限，应在 compile 阶段拒绝，而不是扩大动态结构。

### 5.2 采集点

| StageExecutor/adapter 位置 | Receipt 更新 |
|---|---|
| `begin_fill` 成功 | ticket、pass/group/pool/slot、source generation |
| fill completion mailbox | fill status、logical bytes |
| `begin_use` | claim 事件，防止未 Ready 使用 |
| `encode_group` 返回 | groups submitted、reader fence issued |
| completion mailbox | reader fence completed |
| pool switch | class barrier 序号 |
| pass drain | pass completed、carry ticket |
| final finish | drain、event digest、canonical digest |

worker 只发布固定大小 POD completion；receipt 的 vector、digest 和错误字符串都由 owner thread 写入。这样可以保持现有 steady-state allocation/thread-create audit 为零。

### 5.3 Expected receipt

Expected receipt 从 sealed `Layout` 独立生成：

```text
每个 pass/group 的逻辑 fill = 1
expected groups_submitted = pass_count × group_count
expected logical_read_bytes = 每个 pass 所有 group.bytes 之和
expected pool transitions = layout class segment 数 × pass_count
expected source_generation = snapshot lease generation
drain_completed = true
```

`carry_first_group` 只改变时间顺序和 slot rotation，不减少逻辑 fill；验证器必须同时检查 carry event 的 pass/group/slot 坐标。

### 5.4 验证顺序

```text
authority
-> snapshot/layout/record digest
-> summary fields
-> receipt schema/stage
-> fills_per_pass_group == 1
-> fills/groups/logical bytes
-> fence issued == completed
-> source generation
-> event digest
-> drain_completed
-> canonical receipt digest
```

失败时不序列化 public success result；若不能证明 GPU/reader backing 安全，则将 engine 标为 quarantine。不要通过 `unload()` 猜测“应该已经安全”。

### 5.5 `StageExecutor` 的最小侵入式接线

不建议把 receipt 字段直接散落进 `ExecutionCounters`。增加一个仅 public active 路径创建的 recorder：

```cpp
struct ExecutionReceiptOptions final {
    std::string layout_digest;
    uint64_t source_generation = 0;
    std::string implementation;
};

class ActualReceiptRecorder final {
public:
    ActualReceiptRecorder(const StageLayout &, ExecutionReceiptOptions);
    void fill_submitted(const tc_stream_slot_ticket_v1 &);
    void fill_completed(const tc_stream_completion_v1 &);
    void readers_issued(const tc_stream_slot_ticket_v1 &, std::span<const tc_stream_reader_fence_v1>);
    void reader_completed(const tc_stream_completion_v1 &);
    void pool_barrier(uint32_t from, uint32_t to);
    void pass_completed(uint32_t, std::optional<tc_stream_slot_ticket_v1> carry);
    void drained();
    ActualStageReceipt finish();
};

class StageExecutor {
public:
    StageExecutor(uint32_t stage, uint64_t request_generation,
                  std::shared_ptr<ModelSlotAdapter>,
                  std::unique_ptr<ActualReceiptRecorder> = {});
};
```

旧构造路径传空 recorder，行为和内存布局以外的可观测语义不变。默认 resident/compiled 根本不创建 `StageExecutor`；private streaming 即使走 executor，也可以保持 recorder 关闭。所有 recorder callback 都由 owner thread 调用：fill worker 仍只写 mailbox，`consume()` 观察 completion 后再更新 receipt。

### 5.6 C ABI 的 additive receipt 接口

不要为了 receipt 重新定义全部 plan/callback ABI，也不要在 C callback 热路径传 JSON。建议增加 plan-version 无关的 opt-in：

```c
#define TC_STREAM_RECEIPT_ABI_V1 1u

typedef struct {
    uint32_t struct_size, version;
    uint64_t source_generation;
    char layout_digest[65];
    char implementation[64];
} tc_stream_receipt_config_v1;

typedef struct {
    uint32_t struct_size, version;
    uint32_t completed_passes, completed_groups;
    uint64_t fills, groups_submitted, logical_read_bytes;
    uint64_t reader_fences_issued, reader_fences_completed;
    uint64_t source_generation;
    uint8_t drained;
    char event_digest[65];
    char canonical_digest[65];
} tc_stream_receipt_v1;

int tc_stream_executor_enable_receipt_v1(
    tc_stream_executor *, const tc_stream_receipt_config_v1 *,
    char *error, size_t error_size);

int tc_stream_executor_receipt_v1(
    tc_stream_executor *, tc_stream_receipt_v1 *,
    char *error, size_t error_size);
```

`enable` 只能在 create 成功、第一次 `run_pass` 之前调用；`receipt` 只能在 `finish` 成功后调用。逐 pass/group matrix 留在 C++ handle 内部并由 generic verifier 校验，C 模型只拿固定大小 summary/digest，避免暴露动态数组所有权。test/evidence build 可以另提供内部导出，不进入 release public header。

Z-Image/Flux 的 C++ adapter 直接绑定 recorder；H3/LTX 的 C runtime 通过上述接口取得同一 generic executor 生成的 receipt，而不是在模型代码内另写一套计数器。

## 6. Scheduler 与多槽调度代码合同

### 6.1 参数语义冻结

| 参数 | 含义 | 运行时能否自动改 |
|---|---|---|
| `prefix` | 永久驻留的前缀 block 数 | 否 |
| `group_size` | 一次 fill 的连续 block 数 | 否 |
| `slot_count=K` | 精确 slot 数 | 否 |
| `distance=D` | 允许提前派发的距离 | 否 |
| `workers=Q` | refill worker 数 | 否 |
| `pass_transition` | reload 或 carry-first | 否 |
| `multi_pool_policy` | serial 或 retain_all | 否 |

任何不满足 capability 的布局都 fail-closed；不能为了 fit memory 自动降 K、缩 group 或跳过 block。

### 6.2 Owner pump 不变量

`StageExecutor` owner thread 是唯一可以修改 slot 状态的线程。worker 只能执行 `FillJob` 并发布 `TC_STREAM_FILL_COMPLETE`。reader callback 只能发布 `TC_STREAM_READER_COMPLETE`。

slot 状态转移必须满足：

```text
Vacant -> Filling -> Ready -> InUse -> Retiring -> Vacant
                              \\-> Ready(carry only at pass boundary)
```

禁止：

- Filling slot 被再次 claim；
- reader 未完成时复写 slot；
- pool 未 drain 就切换 class；
- cancel 后继续 enqueue 新 fill；
- mailbox overflow 后继续运行。

### 6.3 K2 稳态时序

```text
owner: enqueue A      worker: fill A
owner: enqueue B      worker: fill B
owner: claim A -> encode A
owner: claim B -> encode B
owner: enqueue next A while B reader is in flight
```

`overlap_next_fill_after_claim()` 只在 adapter 明确支持、`distance == 0`、非 carry 的简单 segment 中启用；它不能改变 layout digest 或 pass/group 顺序。

### 6.4 Multi-pool

- `serial`：前一 class 的所有 reader drain 后销毁 pool，再创建下一 class；峰值较低，可能有 setup allocation；
- `retain_all`：setup 时创建所有 class pool，运行期只切换 active pool；峰值为各 pool capacity 总和，但 steady-state 不再创建 pool；
- Flux 9B 只有在 dual/single class barrier 和 receipt 验证通过后才能使用 `retain_all`；其他模型默认 `serial`。

## 7. Public Adapter 统一模板

### 7.1 三个 hook 的固定职责

```cpp
probe_public_streaming(input)
    // validate route + metadata-only descriptor + source lease descriptor

compile_public_streaming(probe, record)
    // exact record -> descriptor/layout + lease + immutable snapshot

generate_resolved(execution, event, cancel)
    // revalidated binding -> existing exact core -> receipt -> RunResult
```

`generate_resolved` 不得调用普通 `generate(request)` 作为捷径，因为普通路径看不到 sealed snapshot/lease/authority，也无法证明 reader 使用正确 source generation。

### 7.2 Shared exact core

建议将每个模型内部拆为：

```text
run_impl(request, event, cancel, Optional<PublicExecutionBinding>)
  -> prepare components
  -> exact denoiser/transformer core
  -> VAE/upsampler/audio/finalizer
  -> finish/drain
  -> collect runtime + receipt
```

private candidate 传入手动 layout，public 传入 sealed snapshot；从 `exact denoiser/transformer core` 开始必须复用同一 reader、pool、kernel 和 compute 顺序。

### 7.3 Type-safe binding helper

四模型不要各自手写 authority/snapshot/lease 检查。新增内部模板 helper：

```cpp
template <class SnapshotT>
struct PublicExecutionBinding final {
    std::shared_ptr<const ResolvedRequestExecution> execution;
    std::shared_ptr<const SnapshotT> snapshot;
    const SourceLease *lease = nullptr;
};

template <class SnapshotT>
PublicExecutionBinding<SnapshotT> bind_public_execution(
    std::shared_ptr<const ResolvedRequestExecution> execution,
    std::string_view expected_model,
    std::string_view expected_container);
```

helper 固定检查：execution/probe/snapshot/authority 非空；request、record、probe、snapshot model 相同；container 相同；snapshot dynamic type 正确；authority 仍匹配 record/layout/runtime/device；exact selector resolution digest 匹配；lease 非空且 generation 非零。它只绑定，不重新 resolve、不做 GPU 工作。

### 7.4 `RunResult`/metrics 的拟议扩展

`StreamingRuntimeMetrics` 在 receipt v2 阶段增加：

```cpp
uint32_t receipt_schema_version = 0;
uint64_t source_generation = 0;
uint64_t logical_read_bytes = 0;
uint64_t reader_fences_issued = 0;
uint64_t reader_fences_completed = 0;
std::string receipt_digest;
```

`PublicStreamingSelectionMetrics` 增加 `verified_receipt_digest`，只在 common verifier 全部通过后赋值。完整 fill matrix 不进入普通 App result；test/evidence 路径可以保存独立 receipt artifact。`results.mm` 不序列化 path、fd、inode 或用户目录。

## 8. 四模型逐文件施工单

### 8.1 Z-Image Turbo（C3，推荐首个）

代码落点：

```text
native/models/z_image/z_image.hpp/.cpp
native/models/z_image/streaming_descriptor.hpp
native/platform/apple/z_image_streaming_descriptor.mm
native/platform/apple/z_image_weight_stream.mm
tests/native/z_image_public_streaming_test.cpp
tests/native/test_z_image_public_streaming.py
```

第一版只允许原始 BF16 GPU、单 streamed denoiser stage、single pool、无 LoRA/ANE/compiled/GGUF/ConvRot/image-edit。`ZImageWeightStream` 的 fd、size、mtime 校验迁移到 `SourceLease`，但旧 private 构造器继续保留。

### 8.2 Flux.2 Klein 9B（C4）

代码落点：

```text
native/models/flux2/flux.hpp
native/models/flux2/pipeline.cpp
native/models/flux2/flux_transformer.cpp
native/models/flux2/flux_streaming.hpp
native/platform/apple/flux_streaming_descriptor.mm
tests/native/flux_public_streaming_test.cpp
```

固定 8 dual + 24 single class boundary。`retain_all` 的 pool create/reuse、dual/single concatenate boundary、每一 class 的 source ranges 必须进入 receipt。Flux 4B compiled 不得被 public adapter 误匹配。

### 8.3 MiniMax H3 Turbo（C5）

代码落点：

```text
native/platform/apple/h3_session.mm
native/models/h3_runtime/h3_dit.c
native/models/h3_runtime/h3_streaming_descriptor.*
native/core/stream_slot_c.h
tests/native/h3_public_streaming_test.cpp
```

仅 `minimax-h3-turbo` original BF16 native Metal GPU。`experimental` bool、VSA、ordinary H3、VDN 和 FastH3 不是 public selector 的隐式开关。C bridge 需要增加固定大小 receipt 输出或 owner-side snapshot，carry-first-group 必须核对最后一个 pass 没有残留 carry。

### 8.4 LTX 2.5（C6）

代码落点：

```text
native/platform/apple/ltx_session.mm
native/models/ltx_runtime/ltx_streaming_descriptor.*
native/models/ltx_runtime/ltx_streaming_plan.*
tests/native/ltx_public_streaming_test.cpp
```

authority 和 source lease 必须在 request-scoped worker 内生成；parent 只持有 serializable JobEnvelope。视频 VAE、upsampler、audio、export 和 worker exit/drain 都属于同一 resource closure。worker SIGTERM/SIGKILL 或 drain 不确定时进入 quarantine。

## 9. App/C ABI/JobStore 接缝

### 9.1 C ABI

新增接口必须 additive，不改变旧函数布局：

```c
int tc_engine_streaming_options_json(
    tc_engine *, const char *request_json,
    char **result_json, char **error);

int tc_engine_resolve_streaming_json(
    tc_engine *, const char *request_json,
    char **result_json, char **error);
```

约束：

- `result_json`/`error` 由统一 allocator 分配，调用者用现有 free API 释放；
- busy、null、cancel、stale、quarantine 都返回结构化稳定错误；
- release header 不提供 catalog 注入、authority 构造或 fd 传入；
- resolve-only 不持有 global GPU lock，不创建 GPU backing。

### 9.2 JobEnvelope v2

```json
{
  "schema_version": 2,
  "request_nonce": "...",
  "original_selector": {"selection":"memory_tier","target_request_memory_bytes":10737418240},
  "exact_selector": {"selection":"preset","preset_id":"...","preset_revision":1},
  "resolution_digest": "...",
  "record_digest": "...",
  "layout_digest": "...",
  "source_digest": "...",
  "runtime_digest": "...",
  "device_digest": "...",
  "catalog_revision": "...",
  "state": "resolved"
}
```

不得保存 authority、fd、GPU pointer 或 worker private path。重放时 native 重新 resolve/revalidate；若 record 已 revoke，返回 `streaming_resolution_stale`。

## 10. 锁、取消、失败与 quarantine

### 10.1 固定锁顺序

```text
engine/session lock
  -> request metadata/probe lock（如有）
  -> global GPU lock
  -> adapter/StageExecutor owner thread
```

禁止在 global GPU lock 内等待 App/Swift、catalog provider 外部锁或启动新的 metadata scan。

### 10.2 取消

| 取消点 | 必须动作 |
|---|---|
| preflight 前 | 不创建 authority/snapshot |
| first fill 前 | 停止 enqueue，释放 metadata，返回 cancelled |
| fill 中 | worker 在 chunk 边界检查 cancel，join 后 drain |
| denoise 中 | 不发新 group，drain reader/fence |
| VAE/export 前 | 完成 transformer drain，再释放 component |
| worker kill | parent 等待超时，无法证明 backing 安全则 quarantine |

### 10.3 Quarantine

普通 source 变化（尚未提交 GPU）返回可重试错误；reader 已在飞行中而 drain 不确定、mailbox overflow、GPU callback 丢失、worker 非正常退出时 quarantine。quarantine session 只能销毁/重建，不得通过再次 `load()` 复用。

## 11. 逐文件修改矩阵与阶段停止条件

### 11.1 Common runtime 文件

| 文件 | 修改 | 对应测试 |
|---|---|---|
| `source_lease.hpp/.cpp` | RAII fd、canonical identity、pre/post revalidate | PUB-SOURCE-001…010 |
| `value_probe.hpp/.cpp` | 公共 identity/lease value base | PUB-HOST-103、model probe tests |
| `actual_receipt.hpp/.cpp` | expected/actual/event digest/verifier | PUB-RECEIPT-001…010 |
| `context.hpp/.cpp` | optional owner-thread recorder hooks | executor/audit/P1 |
| `io_executor.*` | 不改 worker ownership；只保留 POD completion | failure/audit |
| `stream_slot_c.h` | additive receipt config/result ABI | C ABI contract |
| `c_bridge.cpp` | enable/get receipt，内部 matrix verification | C bridge failure tests |
| `resolved_request.*` | snapshot/lease/authority identity 边界 | resolver tests |
| `public_runtime.*` | pre/post source revalidation 顺序 | coordinator tests |
| `public_result.*` | receipt v2 hard verification | actual mismatch tests |
| `session.hpp` | metrics/receipt 字段，不改变普通 virtual 顺序 | build/ABI/App |
| `results.mm` | 安全摘要序列化，不泄露 path/fd | contract/RunInsights |
| `tools/native/build.sh` | 新 source 加入 native build | native build |
| `Makefile`/Python wrappers | 新 host/source/receipt targets | CI |

### 11.2 阶段提交

| 阶段 | 主要文件 | 退出条件 |
|---|---|---|
| C1 | `source_lease.*`, `value_probe.*` + fixtures | source mutation/rename/short-read 全绿；Off open/fstat/hash=0 |
| C2 | `actual_receipt.*`, `context.*`, `c_bridge.*` | fake adapter v2 receipt 与 fault matrix 全绿 |
| C3 | Z-Image adapter + test catalog | exact generate、cancel、source lease、receipt 全绿 |
| C4 | Flux adapter | dual/single retained pool + class barrier 全绿 |
| C5 | H3 adapter/C receipt | carry 坐标、authority gate、ordinary H3 non-regression 全绿 |
| C6 | LTX worker/local authority | parent/worker/finalizer correlation、quarantine 全绿 |
| C7 | App/options/JobStore/tooling | UI 只显示五档；resolve→persist→generate 原子事务 |
| C8 | calibration/catalog/release | 至少一个 model+target reviewed record；再考虑 dev merge |

任何阶段若 source lease、receipt v2 或 Off audit 不满足，都不得进入真实 target 校准，更不得写 production catalog。

## 12. 代码审阅清单

### 控制面

- [ ] request validator 为纯函数且只调用一次；
- [ ] preflight/resolve 使用同一 catalog snapshot；
- [ ] exact selector 包含 record/revision/catalog/resolution digest；
- [ ] empty catalog 在 model probe/GPU 前 fail-closed；
- [ ] authority 不可复制、不可序列化。

### 数据面

- [ ] reader 使用 lease fd，不按 path 重开；
- [ ] owner thread 是唯一 slot 状态写入者；
- [ ] worker 固定容量、无 steady-state 分配；
- [ ] pool switch 前完成 drain；
- [ ] mailbox overflow/cancel/short read 走同一 cleanup。

### 结果与性能

- [ ] receipt v2 验证逐 group/pass/fence/source generation；
- [ ] mismatch 不返回 public success；
- [ ] Off provider/probe/source/receipt/worker 计数全为零；
- [ ] P0/P1/P2/P3/P4 分开统计；
- [ ] 没有未经证据的“比 swap 快”或“支持某档位”表述。

## 13. 本文完成定义

本文对应的“框架代码设计完成”只表示：所有类型、所有权、线程、错误、逐文件任务和验收 ID 已经明确，可以由实现团队按阶段施工。它不表示：

- 四模型 public adapter 已完成；
- 任一 8/10/12/16/20 GiB target 已校准；
- production catalog 已非空；
- App 已把 Streaming target 对用户开放。

以上四项必须分别由 39 的执行证据和 37 的 release gate 证明。
