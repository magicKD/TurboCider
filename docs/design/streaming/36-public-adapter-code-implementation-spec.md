# 36 · Public Streaming Runtime 与四模型 Adapter 代码实施规格

[目录](README.md) · [实施蓝图](35-public-streaming-implementation-blueprint-v2.md) · [代码合同工作台](38-framework-code-contracts-and-implementation-workbench.md) · [性能与发布验收](37-public-streaming-calibration-performance-acceptance.md) · [验收工作簿](39-validation-benchmark-and-release-workbook.md) · [当前进度](13-implementation-progress.md)

修订日期：2026-09-17。代码基线：feat/stream@60de338；对照分支：dev@02148b7。状态：**可执行的下一阶段代码规格，尚未开放 production public streaming。**

本文只回答一件事：基于当前已经提交的 selector、resolver、authority、coordinator 和 actual-plan verifier，接下来具体怎样修改代码，才能让 Z-Image Turbo、Flux.2 Klein 9B、MiniMax H3 Turbo 和 LTX 2.5 安全地进入 public streaming。

本文不重新设计 LayoutCompiler、StageExecutor、SlotPool 或模型数学。底层协议仍以 03、09、10 为准；public 产品语义仍以 23–26 为准；校准、swap 对照和发布门见 37。

## 1. 当前代码事实与本轮完成边界

### 1.1 已经提交的基础

当前 HEAD 已经具备：

- schema-v2 StreamingSelector，支持 Off、memory tier 和 exact preset replay；
- request/profile 合并、legacy streaming/budget/offload 冲突检查；
- canonical source/workload/runtime/device/record/resolution digest；
- deterministic PublicPresetResolver；
- internal-only、不可复制、不可序列化的 StreamingAuthority；
- immutable ResolvedRequestExecution；
- tc_engine_resolve_streaming_json；
- active generate 的 resolve → global GPU lock → revalidate → generate_resolved 调用链；
- PublicStreamingCoordinator 和 StreamingCatalogProvider；
- RunResult public selection metrics；
- public result 的 layout/P/G/K/D/Q/pool/worker/source/drain 硬校验；
- production catalog 为空时的 fail-closed 行为；
- host/contract/App build 回归。

### 1.2 当前仍然缺失

当前四个生产 ModelSession 均未 override：

~~~cpp
probe_public_streaming(...)
compile_public_streaming(...)
generate_resolved(...)
~~~

ModelStreamingSnapshot::revalidate_source() 仍有默认空实现。当前 verifier 只验证 adapter 自报的汇总字段，尚未验证每个 pass/group 的实际 fill、逻辑读取字节、reader fence、source generation 和结构化 receipt。

App 还没有 public StreamingChoice、engine-scoped exact options、JobStore v2 事务和 RunInsights public 指标。production catalog 仍为空，因此不能把任何档位标为 available。

### 1.3 本文定义的“代码完成”

框架代码完成不等于档位发布。本文的代码完成要求：

1. coordinator 不重复 preflight 或重复取 catalog；
2. source lease 从 probe、snapshot、reader 到 result 形成闭环；
3. actual receipt 能验证真实 fill/read/fence/drain，而不只是 adapter 自报参数；
4. 四模型可在 test-only reviewed catalog 下执行 public exact route；
5. private candidate 与 public route 共用同一个 exact execution core；
6. selector absent/disabled 的默认路径不创建任何新增对象、不查询 catalog、不做 source hash；
7. production catalog 继续为空，直到 37 的真实校准和 review 完成。

### 1.4 C0 已完成项

C0 已将本规格第 3 节的 coordinator 收口落到代码：

- `StreamingCatalogProvider` 改为返回 shared immutable snapshot；
- 新增不可复制的 `PublicStreamingPreflight` ticket，绑定 request digest 和 catalog snapshot；
- `c_api.mm` 在 make_plan 前只执行一次 preflight；planner 使用 `make_plan_after_public_streaming_preflight`，避免重复 public validator/selector 校验；
- `resolve_normalized` 消费 ticket 并在 probe 前检查 request digest；
- `revalidate` 每次读取最新 provider snapshot，用于检测 catalog revision/revoke；
- host test 覆盖 provider snapshot 稳定性、请求篡改、empty catalog 和 revalidate stale；
- default/Off 路径仍不创建 coordinator、snapshot、probe 或 receipt。

C0 没有改变 public catalog 为空、四模型 adapter 未完成和 production public 不可执行的边界。

## 2. 目标依赖结构

依赖只能单向：

~~~text
core request/schema
        |
        v
public_request_validation
        |
        v
catalog provider ---> preset resolver
        |                  |
        +-------> public coordinator
                         |
                         v
             probe -> snapshot -> authority
                         |
                         v
              model public adapter binding
                         |
                         v
       existing exact core / StageExecutor / C bridge
                         |
                         v
             actual receipt -> public verifier
~~~

禁止的依赖：

- model adapter 反向访问 App/Swift；
- StageExecutor 读取 memory tier 或 catalog；
- catalog record 保存 GPU pointer、file descriptor 或 session；
- App 构造 authority；
- public verifier 重新执行模型逻辑；
- private candidate 通过伪造 public metrics 绕过 authority。

## 3. R0：收口 PublicStreamingCoordinator

### 3.1 当前重复工作

当前 c_api.mm 的 resolve_public_streaming_locked 先调用 coordinator.preflight(request)，随后调用 make_plan，再调用 resolve_normalized；resolve_normalized 内部又调用一次 preflight。

production provider 当前返回静态空 catalog，因此重复调用暂时很便宜，但一旦 provider 引入 validated snapshot、revocation table 或测试注入，这会产生：

- 重复错误检查；
- 同一请求看到两个不同 catalog snapshot 的可能性；
- options/resolve/generate 计数难以审计；
- public active 路径不必要的额外工作。

### 3.2 推荐接口

新增一次性 preflight ticket：

~~~cpp
struct PublicStreamingPreflight final {
    Request request_before_planning;
    std::shared_ptr<const StreamingPresetCatalog> catalog;
    std::string catalog_revision;
};

class PublicStreamingCoordinator final {
public:
    PublicStreamingPreflight preflight(Request request) const;

    std::shared_ptr<const ResolvedRequestExecution>
    resolve_preflighted(
        PublicStreamingPreflight,
        Request normalized_request,
        const StreamingDeviceIdentity &) const;

    void revalidate(
        const ResolvedRequestExecution &,
        const StreamingDeviceIdentity &) const;
};
~~~

固定流程：

~~~text
parse
  -> engine try-lock
  -> coordinator.preflight（只一次）
  -> make_plan
  -> resolve_preflighted
  -> release resolve-only call
~~~

generate 仍在取得 engine lock 后 resolve，在取得 global GPU lock 后 revalidate。不能为了复用 App 的 resolve response 而绕过 native 重新 resolve。

### 3.3 CatalogProvider 生命周期

将 provider 从返回裸引用逐步演进为不可变 snapshot：

~~~cpp
class StreamingCatalogProvider {
public:
    virtual ~StreamingCatalogProvider() = default;
    virtual std::shared_ptr<const StreamingPresetCatalog>
    snapshot() const = 0;
};
~~~

规则：

1. production provider 返回进程生命周期的 const shared snapshot；
2. test provider 可以原子替换下一次请求使用的 snapshot，但已经开始的 resolve 始终持有旧 snapshot；
3. snapshot 构造时逐 record 执行 validate_streaming_preset_record；
4. revalidate 获取 current snapshot，并使用 exact selector replay；
5. catalog revision 或 record digest 改变时返回 streaming_resolution_stale；
6. release C header 不提供任意 catalog 注入入口。

若本轮不立即修改 provider ABI，也必须保证 preflight 和 resolve 使用同一 const catalog 引用，并在测试 provider 中禁止调用期间替换 value。

### 3.4 冻结错误顺序

active public selector 的错误顺序固定为：

~~~text
request/schema syntax
selector/legacy conflict
unsupported execution route
engine model mismatch
empty catalog
full make_plan/model validation
metadata probe
catalog exact identity/rank
snapshot compile
authority creation
GPU lock
source/catalog/device revalidation
model execution
actual receipt verification
result serialization
~~~

这样可同时满足：

- 配置错误不被空 catalog 遮蔽；
- 空 catalog 不触发模型文件扫描和 GPU 工作；
- shape/model 错误只在 catalog 非空且 route 有机会执行时进入 planner；
- actual mismatch 永远不会产生成功 result。

### 3.5 R0 测试

新增或扩展：

- PUB-COORD-001：一次 resolve 只调用一次 provider snapshot；
- PUB-COORD-002：一次 resolve 只调用一次 request-only validator；
- PUB-COORD-003：empty catalog 时 probe/compile/make GPU calls 均为零；
- PUB-COORD-004：preflight snapshot 在 resolve 期间保持稳定；
- PUB-COORD-005：revalidate 看到新 revision 时 stale；
- PUB-COORD-006：selector disabled 不构造 coordinator；
- PUB-COORD-007：engine busy 时 result/error ownership 正确；
- PUB-COORD-008：cancel before resolve 不产生 authority；
- PUB-COORD-009：cancel after revalidate 由模型 drain 后返回 status 2；
- PUB-COORD-010：resolve-only 不持有 global GPU lock。

## 4. R1：通用 Value Probe、Snapshot 与 Source Lease

### 4.1 为什么需要通用值类型

四个模型需要报告的 public identity 结构相同，区别主要在：

- 如何读取 checkpoint/index metadata；
- 如何编译 Descriptor/Layout；
- exact reader 如何消费 source；
- 如何把现有执行计数转换为 receipt。

如果每个模型都独立实现 probe/snapshot，会重复 canonical identity、source revalidation 和 component policy 校验，并容易出现某个模型少绑定一个 digest 的问题。

建议新增：

~~~text
native/runtime/streaming/value_probe.hpp/.cpp
native/runtime/streaming/source_lease.hpp/.cpp
native/runtime/streaming/actual_receipt.hpp/.cpp
~~~

### 4.2 ValueModelStreamingProbe

~~~cpp
class ValueModelStreamingProbe final : public ModelStreamingProbe {
public:
    struct Values {
        std::string model_id;
        PresetSourceIdentity source;
        PresetWorkload workload;
        PresetRuntimeIdentity runtime;
        std::string component_policy_revision;
        std::shared_ptr<const SourceLeaseDescriptor> source_descriptor;
    };

    explicit ValueModelStreamingProbe(Values);

    // ModelStreamingProbe overrides...
    const SourceLeaseDescriptor &source_descriptor() const noexcept;
};
~~~

构造后所有字段不可变。构造器必须校验：

- model_id 与 workload.model 相同；
- workload.execution_container 与 input 相同；
- source/runtime 的必需 digest 非空且格式合法；
- component policy revision 非空；
- source descriptor 覆盖 descriptor 将引用的所有 artifact；
- probe 不含 GPU/MLX/Metal backing；
- probe 不启动 I/O worker。

### 4.3 SourceLeaseDescriptor

~~~cpp
struct SourceFileIdentity {
    std::string logical_id;
    std::filesystem::path canonical_path;
    uint64_t device = 0;
    uint64_t inode = 0;
    uint64_t bytes = 0;
    int64_t mtime_ns = 0;
    std::string header_digest;
    std::string manifest_digest;
};

struct SourceLeaseDescriptor {
    std::string source_snapshot_digest;
    std::vector<SourceFileIdentity> files;
};
~~~

logical_id 必须稳定，例如 transformer.index、transformer.shard.0、h3.dit、ltx.transformer。canonical digest 按 logical_id 排序，不能按目录枚举顺序。

### 4.4 SourceLease

仅比较 path/mtime 仍有 probe 后替换文件的 TOCTOU 风险。public reader 必须从已验证的只读句柄读取：

~~~cpp
class SourceLease final {
public:
    static std::shared_ptr<const SourceLease>
    open_and_verify(const SourceLeaseDescriptor &);

    void revalidate_paths() const;
    int duplicate_fd(std::string_view logical_id) const;
    uint64_t generation() const noexcept;
    std::string_view digest() const noexcept;
};
~~~

实现要求：

1. 使用 O_RDONLY | O_CLOEXEC 打开；
2. 打开后立即 fstat，比对 device/inode/size/mtime；
3. reader 使用 lease fd 的 dup 或共享 pread handle，不能重新按 path 打开；
4. revalidate_paths 再比较 path 当前指向的 inode，检测 replace/rename；
5. descriptor/header digest 在 probe 或 model open 的受控缓存中验证；
6. lease 生命周期覆盖 snapshot 到 drain 完成；
7. cancel/failure 时先停止新 refill，再 drain GPU readers，最后关闭 fd；
8. 不把 fd 序列化进 result；
9. source generation 单调递增且 request-scoped；
10. full-file hash 不进入每次默认请求热路径。

如果现有 C/Metal reader 暂时只能接 path，第一阶段可以使用 open 后的 /dev/fd/N 兼容路径，但最终应直接接 fd/pread，避免 reader 再次解析可替换路径。

### 4.5 ValueModelStreamingSnapshot

~~~cpp
class ValueModelStreamingSnapshot final
    : public ModelStreamingSnapshot {
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
~~~

构造器硬校验：

- descriptor.model == model_id；
- layout digest 已 finalized；
- layout.materializations_complete；
- source/runtime 与 probe 完全相等；
- component policy 与 selected record 完全相等；
- lease digest 与 source.source_snapshot_digest 对应；
- layout 只有当前 public v1 允许的一个 streamed stage；
- retention 为 request；
- pass/pool policy 在 adapter capability 内。

### 4.6 Probe 缓存边界

允许 engine/session 缓存纯 metadata 结果，但 cache key 必须包含：

~~~text
model root identity
all source file stat tuples
source manifest digest
TurboCider build/runtime/adapter/reader/kernel revision
execution container
workload feature identity
~~~

缓存命中只能省 metadata parse/hash，不能跨请求复用：

- StreamingAuthority；
- ResolvedRequestExecution；
- mutable SourceLease generation；
- slot pool；
- worker；
- GPU backing；
- cancellation token。

### 4.7 R1 测试

- PUB-SOURCE-001：文件内容不变时 probe/snapshot identity 稳定；
- PUB-SOURCE-002：path replace 后 revalidate 失败；
- PUB-SOURCE-003：原 path 被 rename，lease fd 仍指向原 inode，但 path validation 失败；
- PUB-SOURCE-004：size/mtime/inode 任一变化均失败；
- PUB-SOURCE-005：short read 返回稳定错误并 drain；
- PUB-SOURCE-006：多 shard 顺序不影响 canonical digest；
- PUB-SOURCE-007：重复 logical_id 拒绝；
- PUB-SOURCE-008：未被 descriptor 引用的旁路文件不进入 lease；
- PUB-SOURCE-009：descriptor 引用未租赁 artifact 时 compile 失败；
- PUB-SOURCE-010：selector absent 时 open/fstat/hash 计数为零。

## 5. R2：Actual Execution Receipt v2

### 5.1 当前汇总 metrics 的不足

当前 PublicStreamingSelectionMetrics 和 StreamingRuntimeMetrics 可以发现 layout 参数、pool 数、worker 数、source/drain flag 不一致，但不能证明：

- 每个 pass 的每个 group 正好 fill 一次；
- fill 的 pool/slot/ticket generation 正确；
- logical read bytes 与 descriptor source ranges 一致；
- reader fence 数量与完成数量一致；
- carry_first_group 没有漏 fill 或重复消费；
- dual/single pool 的 class barrier 真正发生；
- source lease generation 与 reader 使用的 generation 相同。

因此新增内部 receipt，不直接把大数组暴露给 App。

### 5.2 类型设计

~~~cpp
struct ActualFillReceipt {
    uint32_t pass;
    uint32_t group;
    uint32_t pool;
    uint32_t slot;
    uint64_t ticket_generation;
    uint64_t logical_bytes;
    uint64_t source_generation;
};

struct ActualStageReceipt {
    std::string stage_id;
    std::string layout_digest;
    uint32_t completed_passes;
    uint32_t completed_groups;
    uint64_t fills;
    uint64_t groups_submitted;
    uint64_t logical_read_bytes;
    uint64_t reader_fences_issued;
    uint64_t reader_fences_completed;
    uint64_t source_generation;
    bool drain_completed;
    std::vector<uint32_t> fills_per_pass_group;
    std::string event_digest;
};

struct ActualExecutionReceipt {
    uint32_t schema_version = 2;
    std::string implementation;
    std::string layout_digest;
    std::string component_policy_revision;
    std::vector<ActualStageReceipt> stages;
    std::string canonical_digest;
};
~~~

fills_per_pass_group 在 begin 时一次性分配，大小为 pass_count × group_count。public v1 模型 block 数较小，开销可控；private/default 路径不创建该数组。

### 5.3 StageExecutor 采集点

在以下位置更新 receipt：

| 位置 | 记录 |
|---|---|
| begin_fill 成功后 | pass/group/pool/slot/ticket/source generation |
| fill completion | logical bytes、完成状态 |
| begin_use | group submitted |
| seal_readers | issued fence count |
| complete_reader | completed fence count |
| pool switch | class barrier 序号 |
| pass drain | completed pass、carry ticket |
| finish | final drain、event digest |

receipt 更新必须由 owner thread 完成。worker completion 仍只写固定 mailbox record，避免在 worker 热路径加锁或分配。

### 5.4 Expected receipt 编译

从 sealed Layout 独立生成 ExpectedStageReceipt：

~~~text
expected fills per pass/group = 1
expected logical bytes = sum(group.bytes over all passes)
expected submitted groups = pass_count × group_count
expected pool transitions = layout class segments × passes
expected source generation = snapshot lease generation
expected drain = true
~~~

carry_first_group 只是改变 fill 的时间和下一 pass slot rotation，不减少逻辑 fill 数。验证器应额外核对 carry event 坐标和 K2 slot rotation。

### 5.5 Receipt 验证顺序

verify_and_attach_public_streaming_result 固定按以下顺序：

1. authority 非空；
2. snapshot/layout/record digest 一致；
3. runtime summary 与 layout 一致；
4. receipt schema/implementation/stage 数正确；
5. fills_per_pass_group 每项等于 1；
6. fills、groups submitted、logical bytes 与 expected 一致；
7. reader issued == reader completed；
8. source generation == snapshot lease generation；
9. event digest 可重算；
10. drain_completed；
11. receipt canonical digest；
12. 最后才附加 public_streaming metrics。

任一失败：

~~~text
result 不序列化
output 不标记为成功
engine/session 按 drain 结果决定是否 quarantine
返回 streaming_actual_plan_mismatch 或更具体稳定错误
~~~

### 5.6 向后兼容

第一阶段允许 receipt v1/v2 双轨：

- fake host adapter 必须先实现 v2；
- Z-Image/Flux 使用 C++ v2；
- H3/LTX 的 C bridge 增加固定 ABI receipt；
- 所有模型进入 production record 前必须是 v2；
- v1 summary 只能用于 private candidate，不可用于 public-stable。

### 5.7 R2 测试

- PUB-RECEIPT-001：完整 reload receipt PASS；
- PUB-RECEIPT-002：某 pass/group fill=0 失败；
- PUB-RECEIPT-003：重复 fill 失败；
- PUB-RECEIPT-004：logical bytes 少一个 source range 失败；
- PUB-RECEIPT-005：reader completion 少一个失败；
- PUB-RECEIPT-006：wrong source generation 失败；
- PUB-RECEIPT-007：wrong pool/class transition 失败；
- PUB-RECEIPT-008：carry slot rotation 错误失败；
- PUB-RECEIPT-009：drain false 触发 quarantine；
- PUB-RECEIPT-010：receipt mismatch 不留下 public success metrics。

## 6. 通用 Public Adapter Binding

### 6.1 不新增第二套执行器

每个模型只增加 public 的“授权入口”，不复制数学和 slot executor：

~~~text
private candidate request --+
                           +--> exact execution core --> receipt
public resolved request ----+
~~~

private route从 manual StreamingConfig 编译 plan；public route直接消费 sealed snapshot。两者在进入 exact execution core 后必须共享同一个 reader、pager、pool 和 block compute。

### 6.2 PublicExecutionBinding

~~~cpp
struct PublicExecutionBinding final {
    std::shared_ptr<const ResolvedRequestExecution> execution;
    std::shared_ptr<const ValueModelStreamingSnapshot> snapshot;
    const SourceLease *source_lease = nullptr;
};

template <class Snapshot>
PublicExecutionBinding bind_public_execution(
    std::shared_ptr<const ResolvedRequestExecution> execution,
    std::string_view expected_model);
~~~

binding 校验：

- execution 非空；
- request.model、snapshot.model_id、record.workload.model 全相同；
- exact selector 的 expected resolution digest 存在；
- authority matches 已由 coordinator revalidate；
- snapshot dynamic type 正确；
- request.streaming canonical config 与 snapshot layout 一致；
- source lease 非空且刚完成 revalidate。

### 6.3 Session 方法模板

~~~cpp
std::shared_ptr<const ModelStreamingProbe>
ModelSessionX::probe_public_streaming(
    const PublicResolveInput &input) const {
    validate_public_model_route(input);
    return build_value_probe(input);
}

std::shared_ptr<const ModelStreamingSnapshot>
ModelSessionX::compile_public_streaming(
    std::shared_ptr<const ModelStreamingProbe> probe,
    const StreamingPresetRecord &record) const {
    return compile_value_snapshot(probe, record);
}

RunResult ModelSessionX::generate_resolved(
    std::shared_ptr<const ResolvedRequestExecution> execution,
    const Event &event,
    std::atomic<bool> &cancel) {
    auto binding = bind_public_execution<SnapshotX>(
        std::move(execution), kModelId);
    return run_exact(binding.execution->request, event, cancel, &binding);
}
~~~

generate_resolved 不能简单调用 generate(request)，因为普通 generate 只看到 exact config，看不到 snapshot、lease 和 authority，也无法证明 reader 使用的是已租赁 source。

## 7. Z-Image Turbo Public Adapter

### 7.1 首批资格范围

首批仅允许当前 exact adapter 已验证的 Z-Image Turbo 原始 BF16 GPU 路线。明确拒绝：

- GGUF；
- ConvRot；
- NVFP4；
- 未纳入 descriptor 的 Diffusers shard 组合；
- LoRA；
- GPU+ANE；
- compiled GPU；
- quantized cache；
- image edit/transform，除非单独 workload card 认证。

### 7.2 文件改动

~~~text
native/models/z_image/z_image.hpp
native/models/z_image/z_image.cpp
native/models/z_image/streaming_descriptor.hpp
native/platform/apple/z_image_streaming_descriptor.mm
native/platform/apple/z_image_weight_stream.mm
tests/native/z_image_public_streaming_test.cpp        新增
tests/native/test_z_image_public_streaming.py         新增
~~~

### 7.3 代码重构

在 ZImage 中新增三个 override，并把 run 拆成：

~~~cpp
RunResult run_impl(
    const Request &,
    const Event &,
    std::atomic<bool> &,
    bool warmup,
    const PublicExecutionBinding *public_binding);
~~~

ZImageExactStream 新增从 sealed plan 和 lease 构造的入口：

~~~cpp
ZImageExactStream(
    std::shared_ptr<const ZImagePublicSnapshot>,
    Weights &resident,
    const Event &,
    std::atomic<bool> &,
    uint64_t request_generation);
~~~

public path不得再次从 request 重新编译 descriptor。private candidate 保留当前构造器，但最终二者都调用统一的 Impl 构造函数。

### 7.4 Probe

probe 只读取：

- transformer format/metadata；
- safetensors header 和 block source ranges；
- model variant；
- width/height/steps/dynamic text/token shape；
- runtime/adapter/reader/kernel revisions；
- component policy revision。

不加载 text encoder、transformer tensor、VAE，不创建 MLX array。

### 7.5 Snapshot 与执行

snapshot 保存 z_image::StreamingMetadata、Descriptor、Layout 和 transformer SourceLease。执行时：

1. 清理与 exact request 不兼容的 resident transformer；
2. 保留 conditioning 仅在 component policy 明确允许时复用；
3. MlxWeightPager 从 lease fd 构造；
4. begin exact stream；
5. 每 pass 运行现有 block compute；
6. finish 并取 receipt；
7. clear exact resident fixed tensors；
8. drain；
9. 加载 VAE；
10. 返回 runtime summary + receipt。

### 7.6 必须补齐的 metrics

- component_policy_revision；
- multi_pool_policy=serial；
- pool_count=1；
- slot_bundle_count=K；
- refill_worker_count=Q；
- source_lease_verified=true；
- source_generation；
- logical_read_bytes；
- fills_per_pass_group；
- reader fences issued/completed；
- drained=true。

### 7.7 Z-Image 验收

- public 与 private exact 同一 sealed layout 输出 byte-exact 或现有质量门等价；
- test catalog exact replay；
- source path replace；
- cancel before first fill、mid-fill、mid-denoise、before VAE；
- short read；
- stale resolution；
- wrong component policy；
- result receipt mismatch；
- selector Off 的 resident audit 全零；
- 首批真实校准前 production catalog 仍为空。

## 8. Flux.2 Klein 9B Public Adapter

### 8.1 首批资格范围

仅 Flux.2 Klein 9B BF16 eager GPU、8 dual blocks + 24 single blocks、无 LoRA、无 reference/edit、完整 denoise schedule。Flux 4B compiled graph 不进入本框架。

### 8.2 文件改动

~~~text
native/models/flux2/flux.hpp
native/models/flux2/pipeline.cpp
native/models/flux2/flux_streaming.hpp
native/models/flux2/flux_transformer.cpp
native/models/flux2/streaming_descriptor.hpp
native/platform/apple/flux_streaming_descriptor.mm
native/runtime/streaming/mlx_weight_pager.*
tests/native/flux_public_streaming_test.cpp             新增
tests/native/test_flux_public_streaming.py              新增
~~~

### 8.3 保留 multi-pool 语义

当前 proven seed 为双 pool retain_all、K2/G1/D1/Q2。public snapshot 必须显式包含：

- dual pool 和 single pool 的独立 slot capacities；
- group → pool 映射；
- retain_all；
- dual→single class barrier；
- single block 8 前的 concatenate boundary；
- 两 pool request lifetime；
- release-before-VAE component boundary。

不能把两个 pool 合并为最大 slot 的单池来“简化”public adapter；这会改变峰值、allocation 和 evidence identity。

### 8.4 代码重构

Flux::run 增加 public binding 参数；FluxExactStream 增加 snapshot/lease 构造器。当前从 root/StreamingConfig/StreamingWorkload 构造 plan 的逻辑下沉为 private plan factory，public 不再重复编译。

MlxWeightPager 增加：

~~~cpp
MlxWeightPager(
    const SourceLease &,
    const Descriptor &,
    const Layout &,
    ...);
~~~

每个 shard 的 logical id 必须由 descriptor 指向 lease file，禁止根据目录再次 discover。

### 8.5 Receipt 专项

除通用字段外必须记录：

- dual fills；
- single fills；
- dual reader completions；
- single reader completions；
- pool selection sequence；
- class barrier count；
- pool create count=2；
- slot bundle count=4（当前 seed）；
- 两 pool 在第二 pass 不重建；
- context concatenate 只在规定边界发生。

### 8.6 Flux 验收

- 复用当前 same-layout direct/generic P1 证据作为 seed，不把它当 public 资格；
- test catalog public route 与 private exact layout digest 相同；
- dual/single 任一 shard replace 都失败；
- pool transition 错误被 receipt 捕获；
- cancel/drain 后两 pool 安全销毁；
- default Flux 4B compiled 和 Flux 9B resident 路径 audit 零回归；
- full request target 校准单独按 37 执行。

## 9. MiniMax H3 Turbo Public Adapter

### 9.1 唯一支持范围

只做 minimax-h3-turbo。普通 H3、FastH3、VDN-H3、其他 checkpoint、量化变体不进入 public catalog。

首批只允许当前 exact route 已验证的 original BF16 native Metal GPU、G1/K2/D1/Q1、carry_first_group、无 LoRA、无 GPU+ANE。

### 9.2 文件改动

~~~text
native/platform/apple/h3_session.mm
native/models/h3_runtime/h3_streaming_descriptor.*
native/models/h3_runtime/h3.h
native/models/h3_runtime/h3.c
native/models/h3_runtime/h3_dit.h
native/models/h3_runtime/h3_dit.c
native/core/stream_slot_c.h 或独立 receipt C ABI
tests/native/h3_public_streaming_test.mm              新增
tests/native/test_h3_public_streaming.py              新增
~~~

### 9.3 移除 public 对 experimental bool 的依赖

当前 exact H3 由 allow_experimental_streaming_ 保护。正确的 public 放行方式不是把 production session 的 bool 改成 true，而是：

~~~text
ordinary generate + exact config + no public binding
    -> 仍拒绝

private candidate engine
    -> allow_experimental_streaming_ 允许

generate_resolved + valid authority/snapshot/lease
    -> 独立 public_authorized 分支允许
~~~

run_impl 接收 enum：

~~~cpp
enum class H3ExactAuthority {
    none,
    private_candidate,
    public_resolved,
};
~~~

任何 exact route 必须是 private_candidate 或 public_resolved，不能由 request 字段自行决定。

### 9.4 C receipt ABI

扩展 h3_dit_exact_streaming_info 或新增版本化结构：

~~~c
typedef struct h3_stream_receipt_v2 {
    uint32_t struct_size;
    uint32_t version;
    uint64_t source_generation;
    uint64_t logical_read_bytes;
    uint64_t reader_fences_issued;
    uint64_t reader_fences_completed;
    uint32_t pass_count;
    uint32_t group_count;
    const uint32_t *fills_per_pass_group;
    uint32_t fills_per_pass_group_count;
    int drained;
} h3_stream_receipt_v2;
~~~

数组生命周期至少持续到 h3_result_free；C++ H3Session 在返回 RunResult 前复制为内部 receipt。ABI 使用 struct_size/version，旧 private 工具保持兼容。

### 9.5 Carry 验证

必须验证：

- K=2；
- 单 pool；
- 每 group 单 block；
- 每 pass group 0 的 fill 可在前一 pass 尾部发生，但坐标属于下一 pass；
- slot rotation 与 layout 算法一致；
- completed passes == request steps；
- fill 总数 == streamed groups × passes；
- carry drain 只保留一个 ready ticket；
- final pass 不留下 carry ticket。

### 9.6 H3 验收

- ordinary production engine 手工 exact config 仍拒绝；
- public resolved route只对 minimax-h3-turbo放行；
- private candidate仍工作；
- carry first group receipt 完整；
- cancel 在 carry fill 前后均安全；
- native GPU drain 失败时 quarantine；
- H3 cache 在 public exact route 禁用，不污染下一 resident request；
- output/quality、P0/P1 和 full request memory 按 37 验收。

## 10. LTX 2.5 Public Adapter 与 Worker-local Authority

### 10.1 首批资格范围

仅 ltx-2.5-distilled 当前原生 C/Metal exact route及其已认证 checkpoint/component policy。GPU+ANE Gemma、其他 LTX checkpoint、不同 audio/upsampler/VAE policy 必须是独立 record。

### 10.2 文件改动

~~~text
native/platform/apple/ltx_session.mm
native/models/ltx_runtime/ltx_streaming_descriptor.*
native/models/ltx_runtime/ltx_streaming_plan.*
native/models/ltx_runtime/ltx_streaming_adapter.inc
native/models/ltx_runtime/ltx_safetensors.*
apps/cli/*
apps/macos/LTXWorker.swift
apps/macos/JobStore.swift
tests/native/ltx_public_streaming_test.mm              新增
tests/native/test_ltx_public_streaming.py              新增
tests/TurboCiderKitTests/LTXStreamingWorkerTests.swift 新增
~~~

### 10.3 为什么 authority 必须在 worker 内生成

LTX App 路径会启动独立 CLI worker。authority 绑定：

- worker 进程看到的 model source；
- worker runtime/build；
- worker execution container；
- worker device identity；
- worker 内部 source lease 和 session。

因此 App 主进程不能 resolve 后把 authority 传给 worker。App 只持有用户 selector、resolution summary 和 expected resolution digest；worker 必须自己重新 exact resolve。

### 10.4 Worker 两阶段握手

建议协议：

~~~text
App:
  create pending job
  spawn worker with v2 request

Worker:
  open engine
  resolve selector
  emit streaming_resolved event + exact selector
  wait for ACK

App:
  validate response belongs to current request nonce
  persist JobEnvelope v2 atomically
  send ACK

Worker:
  generate using exact selector + expected resolution digest
  native re-resolves and creates worker-local authority
  emit result
~~~

若 App 在 persist 前崩溃，worker因无 ACK 超时退出且不提交 GPU。若 catalog/source 在 resolve 与 generate 之间改变，exact replay失败，不静默选择新 plan。

### 10.5 Worker 协议字段

~~~json
{
  "schema_version": 2,
  "kind": "streaming_resolved",
  "request_nonce": "...",
  "resolution": {},
  "exact_request": {},
  "ack_deadline_ms": 10000
}
~~~

ACK：

~~~json
{
  "schema_version": 1,
  "kind": "persisted",
  "request_nonce": "...",
  "resolution_digest": "..."
}
~~~

不要通过环境变量传 authority 或 layout。临时文件权限必须仅当前用户可读写，worker exit 后删除。

### 10.6 LTX exact core

public snapshot 保存 C descriptor/plan 的拥有型拷贝、SourceLease 和 component policy。ltx_exact_stream 使用 lease fd，不能再次按 checkpoint path 打开。

现有 quarantine 机制保留。public adapter补齐：

- source generation；
- fill matrix；
- reader fence counts；
- exact plan digest；
- worker exit/drain；
- denoiser→upsampler→VAE/audio component boundary；
- process-tree memory sample correlation ID。

### 10.7 LTX 验收

- worker无 ACK 不执行 GPU；
- App persist失败会终止 worker；
- worker本地 resolve与App summary digest一致；
- stale exact selector失败；
- SIGTERM先 cancel/drain，超时SIGKILL被记录为非成功；
- quarantine不能被 unload 强行清除；
- video finalizer输出与当前路径一致；
- parent+worker+finalizer完整内存进入37的校准峰值。

## 11. Engine、锁、取消和 Quarantine

### 11.1 固定锁顺序

~~~text
engine mutex
  -> public resolve/probe/snapshot
  -> global execution mutex
  -> DeviceLease
  -> source revalidate
  -> model generate_resolved
  -> receipt verify
  -> result serialize
~~~

禁止：

- 持有 global GPU lock 时运行 App I/O；
- refill worker获取 engine mutex；
- receipt verifier等待 worker；
- cancel线程等待 engine mutex后才设置 atomic cancel；
- source lease destructor在未 drain 时关闭 fd/backing。

### 11.2 取消阶段

| 阶段 | 行为 |
|---|---|
| preflight/probe | 抛 Cancelled，不创建 pool |
| snapshot/lease | 关闭已打开 fd，无 GPU 资源 |
| first fill 前 | shutdown worker，destroy empty pools |
| fill 中 | 停止新任务，等待/取消 read，drain |
| GPU readers 中 | 不复写 backing，等待 fence/drain |
| pass boundary | 不启动下一 pass |
| VAE/export | exact pool已释放；按模型现有取消处理 |

### 11.3 Quarantine 决策

以下情况必须 quarantine：

- drain 无法证明完成；
- reader completion 丢失；
- backing 仍可能被 GPU 使用；
- source lease 与实际 reader generation 不一致；
- receipt 表明 slot safety 不完整；
- C executor destroy 返回不安全。

配置错误、catalog stale、source 在执行前改变但尚未提交 GPU，不需要 quarantine。

## 12. App 与 Native API 必需接缝

完整 App 设计见 37，本节只冻结 native 接口。

当前 tc_streaming_options_json 没有 engine/model artifact，因此只能返回 tentative_without_artifact_identity。public App 应新增 engine-scoped exact options：

~~~c
int tc_engine_streaming_options_json(
    tc_engine *,
    const char *request_json,
    char **result_json,
    char **error);
~~~

它复用 probe 和 catalog filtering，但不 compile GPU payload。返回每个 target：

~~~text
available
unavailable
unsupported_workload
artifact_mismatch
unvalidated_device
catalog_empty
~~~

全局 tc_streaming_options_json 可保留给未打开模型时的 tentative UI，但不得把 tentative 显示成“可运行”。

Swift 需增加：

~~~swift
public func streamingOptions(
    _ request: NativeRequestV2
) async throws -> NativeStreamingOptions
~~~

该实例方法通过 engine queue 串行；静态 tentative 方法改名或在类型中明确 query_status。

## 13. 逐文件修改矩阵

| 文件 | 修改 | 不允许影响 |
|---|---|---|
| public_runtime.* | 单次 preflight ticket、snapshot provider | Off 路径 |
| catalog_provider.* | immutable snapshot/test provider | release 任意注入 |
| resolved_request.* | lease/receipt binding identity | authority 可序列化 |
| source_lease.* | fd-based source lease | 默认路径 full hash |
| actual_receipt.* | expected/actual receipt、digest | worker heap allocation |
| session.hpp | receipt/result字段、helper forward declarations | 普通 virtual 调用顺序 |
| c_api.mm | engine-scoped options、协调锁/错误/ownership | resident planner行为 |
| results.mm | receipt摘要/public result | 暴露 path/fd |
| z_image.* | 三 hook、shared exact core | GGUF/ConvRot默认路径 |
| flux2/* | 三 hook、multi-pool receipt | Flux 4B compiled |
| h3_session/runtime | public authority route、C receipt | 其他 H3变体 |
| ltx_session/runtime | public snapshot、fd reader、worker receipt | legacy component staged |
| TurboCiderNative.swift | engine options、resolution/result值类型 | legacy request兼容 |
| App files | 由37实施 | 默认Off |

## 14. 建议提交序列

### C0 · Coordinator cleanup

- 去掉重复 preflight；
- provider snapshot 稳定性；
- C ABI null/busy/cancel/serialization 测试；
- production catalog 仍为空。

### C1 · Source lease + value probe/snapshot

- 通用值类型；
- fd-based fixture；
- source mutation/failure tests；
- 无模型 public override。

### C2 · Receipt v2

- StageExecutor receipt；
- C bridge ABI；
- fake adapter hard verification；
- default audit zero。

### C3 · Z-Image public adapter

- test-only catalog；
- model hooks；
- lifecycle/quality；
- 不加入 production record。

### C4 · Flux 9B public adapter

- dual/single retained pool；
- class receipt；
- test-only catalog。

### C5 · H3 Turbo public adapter

- public authority与experimental gate分离；
- carry receipt；
- 仅 minimax-h3-turbo。

### C6 · LTX public adapter/worker

- worker两阶段握手；
- JobStore接缝；
- process quarantine tests。

### C7 · App/tooling/calibration

按37完成后，才允许 record-only production catalog commit。

### C8 · Merge dev

合并 dev@02148b7 或届时最新已审阅 dev；保留 dev 默认优化，重跑 Off/default、GPU+ANE、compiled、legacy、public fail-closed 和四模型 test catalog。

## 15. 自动化验收矩阵

### 15.1 Host

| ID | 内容 |
|---|---|
| PUB-HOST-101 | preflight exactly once |
| PUB-HOST-102 | immutable catalog snapshot |
| PUB-HOST-103 | value probe canonical identity |
| PUB-HOST-104 | source lease replace/rename |
| PUB-HOST-105 | expected receipt compiler |
| PUB-HOST-106 | reload receipt |
| PUB-HOST-107 | carry receipt |
| PUB-HOST-108 | multi-pool receipt |
| PUB-HOST-109 | receipt digest mutation |
| PUB-HOST-110 | Off audit zero |

### 15.2 C ABI/Swift

| ID | 内容 |
|---|---|
| PUB-ABI-101 | engine-scoped options |
| PUB-ABI-102 | null result/error ownership |
| PUB-ABI-103 | busy |
| PUB-ABI-104 | cancel status 2 |
| PUB-ABI-105 | stale exact selector |
| PUB-ABI-106 | result serialization |
| PUB-ABI-107 | receipt mismatch no result |
| PUB-SWIFT-101 | Codable semantic parity |
| PUB-SWIFT-102 | error envelope |
| PUB-SWIFT-103 | engine queue serialization |

### 15.3 Model

| ID | 内容 |
|---|---|
| PUB-Z-101…110 | Z probe/snapshot/execute/source/cancel/receipt/default |
| PUB-FLUX-101…112 | Flux dual/single pool/source/class barrier/receipt/default |
| PUB-H3-101…112 | H3 authority gate/carry/C receipt/cancel/cache/default |
| PUB-LTX-101…114 | LTX worker handshake/source/C receipt/quarantine/finalizer/default |

### 15.4 必跑命令

每个阶段至少运行：

~~~sh
env TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
make test-streaming-host
make test-streaming-contract
make test-streaming-campaign
make test-streaming-source-identity
make test-streaming-audit
make test-streaming-pager
tools/native/build_app.sh
python3 -B tests/native/test_streaming_preset_resolver.py
git diff --check
~~~

涉及真实 Metal/C executor 时再运行对应模型 test catalog smoke、fault 和 campaign。没有模型 fixture 时必须报告 SKIP 原因，不能记为 PASS。

## 16. 性能保护的代码级要求

selector absent/disabled 时：

~~~text
provider snapshot calls          0
public validator active calls   0
probe calls                     0
source open/fstat/hash calls    0
snapshot/authority allocations  0
slot/pool/worker creates        0
new synchronize calls           0
new cache clear/unload calls    0
receipt allocations             0
~~~

实现方式必须是入口处分支，而不是创建 coordinator 后让其内部快速返回。

public active 路径允许 request-scoped metadata/receipt开销，但：

- receipt 容器在 begin 一次分配；
- worker不做动态分配；
- canonical receipt digest在finish计算；
- source full hash使用 model-open/probe cache，不逐 block计算；
- options查询不加载tensor或创建GPU backing；
- actual verifier不触发synchronize。

## 17. 代码 Definition of Done

### Framework

- [ ] coordinator只preflight一次；
- [ ] catalog snapshot在单次resolve内稳定；
- [ ] source lease reader不按可替换path重开；
- [ ] receipt v2覆盖fill/read/fence/source/drain；
- [ ] actual mismatch不产生成功result；
- [ ] Off/default audit全零；
- [ ] production catalog仍可保持空而全套测试通过。

### 每个模型

- [ ] 三个public hook override；
- [ ] public/private共享exact core；
- [ ] unsupported format/route明确拒绝；
- [ ] snapshot source/layout/runtime/component identity sealed；
- [ ] reader消费lease；
- [ ] receipt v2；
- [ ] cancel/failure/drain/quarantine；
- [ ] default路径无新增行为；
- [ ] test-only catalog端到端通过。

### 进入真实校准前的停止条件

只有在某模型上述全部完成后，才可进入37的真实8/10/12/16/20 GiB探索。若 source lease、receipt或默认零开销仍未完成，即使private candidate能生成，也不得构建production record。
