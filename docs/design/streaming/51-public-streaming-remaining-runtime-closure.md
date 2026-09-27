# 51 · Public Streaming 剩余 Runtime 闭环：多阶段、H3、LTX 与请求生命周期

修订日期：2026-09-18。状态：**代码实施规格；不代表 public 已开放**。

本文在 [49 代码合同](49-public-streaming-code-contracts-and-execution-blueprint.md) 的基础上，只处理当前代码尚未闭环的部分：

- common runtime 的 request-scoped context 与 multi-stage result/receipt；
- MiniMax H3 Turbo 的 public adapter 和 C bridge 完整 receipt；
- LTX 2.5 的 worker-local authority、多阶段 streaming 和原子输出；
- public 失败、取消、drain、quarantine 的统一生命周期；
- 不影响 Off/default/resident 路径的代码隔离和性能验收。

五档候选搜索、process-tree/swap 实验和 catalog evidence 仍以 [50](50-public-streaming-calibration-and-release-evidence.md) 为准；App 产品化与配置文件见 [52](52-public-streaming-app-config-and-model-tier-spec.md)。

## 1. 当前代码基线与本文完成目标

### 1.1 已经存在的基础

截至 `9a351c9`，仓库已经有：

- schema-v2 public selector、immutable catalog snapshot、resolver 和 authority；
- request-scoped fd-backed `SourceLease`；
- `ValueModelStreamingProbe` / `ValueModelStreamingSnapshot`；
- common `StageExecutor`、slot/pool 状态机和 bounded I/O；
- `ActualExecutionReceipt` v2、C bridge summary receipt 和 common verifier；
- Z-Image Turbo 第一阶段 public lease adapter；
- Flux.2 Klein 9B 第一阶段 public lease adapter，以及 transformer/text/VAE 的同一 lease lineage；
- Flux MLX lazy safetensors 的 fd-backed reader 真实 Metal fixture；
- private LTX、H3 exact candidate 和同布局 P1 工具链。

仍需明确：以上只是实现基础。production catalog 仍是 `tc-streaming-catalog-empty-v1`，没有任何 App 可执行的 production target。

### 1.2 当前结构上的四个阻断点

| 阻断点 | 当前事实 | 必须完成的结构变化 |
|---|---|---|
| H3 actual receipt | C ABI 只返回 summary，`h3_result` 只有 counters | 把真实 group/fence/carry matrix 安全传回 C++ common verifier |
| LTX multi-stage | `Layout`/receipt 能容纳多个 stage，但 `public_result.cpp` 强制 `stages.size()==1` | 增加 boundary receipt、逐 stage metrics 和 multi-stage verifier |
| request cleanup | Z-Image/Flux 仍含迁移期 session binding | 增加 request-scoped run context，明确 drain/quarantine/release 顺序 |
| App/worker | desktop 仍主要生成 legacy `NativeRequest`；LTX worker 只识别 `component_staged` | On 使用 V2 selector，LTX 在 worker 内重新 resolve authority |

### 1.3 本文的 Definition of Done

本文对应代码全部完成时，应满足：

1. H3 Turbo 和 LTX 均实现 `probe_public_streaming`、`compile_public_streaming`、`generate_resolved`；
2. H3 receipt 来自运行中的 C executor，而不是从 counters 推导；
3. LTX 的 stage1、upsampler boundary、stage2、VAE/export 顺序被 actual receipt 证明；
4. 所有 public request 使用 request-local context，不依赖上一请求残留的 lease、target 或 authority；
5. 取消、reader timeout、worker EOF、source mutation 均不会产生可见成功输出；
6. Off/default 路径的 streaming allocation、thread、catalog、probe、lease 和 audit counter 全部为零；
7. host、sanitizer、真实 GPU smoke、P0/P1 都通过后，才进入五档校准。

## 2. 目标对象图与所有权

### 2.1 一个 public 请求的唯一对象图

```text
API request
  |
  +-- PublicStreamingPreflight
  |     +-- immutable catalog snapshot
  |     `-- request digest
  |
  +-- ResolvedRequestExecution (shared immutable)
        +-- exact selected record
        +-- authority
        +-- model probe ---------+
        +-- model snapshot ------+---- shared SourceLease
        `-- request digest
  |
  `-- PublicStreamingRunContext (one generate stack only)
        +-- cancellation reference
        +-- stage executors[]
        +-- stage receipts[]
        +-- boundary receipts[]
        +-- model adapter request state
        +-- quarantine transfer hook
        `-- final ActualExecutionReceipt
```

规则：

- `ResolvedRequestExecution` 可以跨 API resolve/generate 边界共享，但不可序列化、不可复制 authority；
- `PublicStreamingRunContext` 只能存在于一次 `generate_resolved()` 调用栈；
- session 可以持有默认 resident cache，但不能把 public lease、authority、slot pool 或 receipt 留给下一请求；
- quarantine 是失败后的安全所有权转移，不是 cache，也不能被下一请求复用。

### 2.2 建议新增的 request context

新增：

```text
native/runtime/streaming/run_context.hpp
native/runtime/streaming/run_context.cpp
```

建议接口：

```cpp
namespace tc::streaming {

enum class PublicRunPhase : uint8_t {
    created,
    gpu_revalidated,
    running,
    draining,
    receipt_sealed,
    source_revalidated,
    completed,
    quarantined,
};

class PublicStreamingRunContext final {
public:
    PublicStreamingRunContext(
        std::shared_ptr<const ResolvedRequestExecution>,
        std::atomic<bool>& cancel);
    ~PublicStreamingRunContext();

    const ResolvedRequestExecution& execution() const noexcept;
    const SourceLease& lease() const;
    uint64_t source_generation() const;

    StageExecutor& attach_stage(
        uint32_t stage_index,
        std::shared_ptr<ModelSlotAdapter> adapter);
    void finish_stage(uint32_t stage_index);
    void record_boundary(ActualBoundaryReceipt);

    void mark_gpu_revalidated();
    void mark_running();
    void drain_all();
    std::shared_ptr<const ActualExecutionReceipt> seal_receipt(
        std::string implementation,
        std::string component_policy_revision);
    void revalidate_source_after_drain();
    void complete();

    void quarantine(std::string reason) noexcept;
    bool quarantined() const noexcept;
    PublicRunPhase phase() const noexcept;

private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace tc::streaming
```

这里的 `attach_stage()` 只负责持有 executor 和 stage receipt，不负责选择 layout。layout 已经由 snapshot/authority 冻结。

### 2.3 析构与显式 finalize

成功路径必须显式完成，不依赖析构函数“猜测成功”：

```text
finish last stage
  -> drain all model readers/GPU callbacks
  -> seal stage and boundary receipts
  -> build complete execution receipt
  -> lease.revalidate_after_drain()
  -> common public result verifier
  -> release request backing
  -> mark completed
```

异常路径：

```text
set cancel
  -> stop new fills
  -> drain each started stage in reverse order
  -> if every drain is proven safe: destroy pools/readers
  -> otherwise transfer adapter + mailbox + backing to quarantine
  -> never seal a success receipt
  -> never expose output as committed
```

析构函数只做最后的安全网：若 phase 不是 `completed`，调用 `drain_all()`；若无法证明 safe drain，则 quarantine。析构函数不能把异常伪装成成功，也不能释放仍被 GPU 或 worker 引用的 backing。

## 3. Multi-stage result 与 receipt v3 扩展

### 3.1 为什么不能继续复用单 stage summary

当前 `RunResult` 有一个 `streaming_runtime` 和一个 `streaming_receipt`。`ActualExecutionReceipt` 已经含 `stages` vector，但 `public_result.cpp` 仍要求一个 stage，且没有表达组件边界。

LTX 至少需要表达：

```text
stage1 denoiser
  -> drain stage1 readers
  -> release stage1 slot/prefix backing
  -> latent spatial upsampler
  -> stage2 denoiser
  -> drain stage2 readers
  -> release stage2 backing
  -> video VAE / optional audio / export
```

如果没有 boundary receipt，仅从两个 stage receipt 不能证明 stage1 backing 在 upsampler/VAE 前已释放，也不能证明没有跨 stage reader 悬挂。

### 3.2 Additive `RunResult` 结构

保留旧字段用于单 stage/兼容输出，增加：

```cpp
struct StreamingStageRuntimeMetrics {
    uint32_t stage_index = 0;
    StreamingRuntimeMetrics runtime;
};

struct StreamingBoundaryRuntimeMetrics {
    uint32_t boundary_index = 0;
    std::string id;
    std::string from_stage;
    std::string to_stage;
    bool source_stage_drained = false;
    bool source_stage_backing_released = false;
    uint64_t live_slot_bytes_before = 0;
    uint64_t live_slot_bytes_after = 0;
    uint64_t pending_readers_before = 0;
    uint64_t pending_readers_after = 0;
    std::string event_digest;
};

struct RunResult {
    // Existing fields remain.
    std::optional<StreamingRuntimeMetrics> streaming_runtime;
    std::vector<StreamingStageRuntimeMetrics> streaming_stages;
    std::vector<StreamingBoundaryRuntimeMetrics> streaming_boundaries;
    std::shared_ptr<const streaming::ActualExecutionReceipt>
        streaming_receipt;
};
```

兼容规则：

- 单 stage adapter 同时填写 `streaming_runtime` 和一个 `streaming_stages[0]`，过渡期 verifier 接受两者完全相等；
- multi-stage adapter 不得把 stage2 覆盖到 legacy `streaming_runtime`；legacy 字段可以为空，或由 serializer 输出明确的 `summary_only=true`；
- public verifier 以 `streaming_stages[]` 为权威，不从 legacy summary 反向构造 stage；
- default/private 路径无需创建新 vector 内容。

### 3.3 Boundary receipt

在 `actual_receipt.hpp` 中 additive 增加：

```cpp
struct ActualBoundaryReceipt final {
    uint32_t boundary_index = 0;
    std::string id;
    uint32_t from_stage_index = 0;
    uint32_t to_stage_index = 0;
    uint64_t source_generation = 0;
    uint64_t last_reader_sequence = 0;
    uint64_t completed_reader_sequence = 0;
    uint64_t live_slot_bytes_before = 0;
    uint64_t live_slot_bytes_after = 0;
    uint64_t released_slot_bytes = 0;
    uint64_t pending_readers_before = 0;
    uint64_t pending_readers_after = 0;
    bool source_stage_drained = false;
    bool source_stage_backing_released = false;
    bool next_stage_started = false;
    std::string event_digest;
    std::string canonical_digest;
};

struct ActualExecutionReceipt final {
    uint32_t schema_version = actual_receipt_schema_v3;
    // existing fields
    std::vector<ActualStageReceipt> stages;
    std::vector<ActualBoundaryReceipt> boundaries;
    std::string canonical_digest;
};
```

注意：`live_slot_bytes_after == 0` 只表示 streamed slot/prefix backing 已释放，不表示进程总内存为零。VAE、latent、text cache、driver allocation 应进入 memory ledger。

### 3.4 Boundary recorder 的触发点

boundary 不能在结果组装时补写，必须由实际 lifecycle 事件触发：

```cpp
auto boundary = context.begin_boundary(
    "ltx-stage1-to-upsampler", stage1_index, stage2_index);
boundary.observe_before(stage1.live_slot_bytes(),
                        stage1.pending_readers());
stage1.finish();
adapter.release_stage_backing(stage1_index);
boundary.observe_after(stage1.live_slot_bytes(),
                       stage1.pending_readers());
run_upsampler();
boundary.mark_next_stage_started();
context.commit_boundary(std::move(boundary));
```

`commit_boundary()` 至少检查：

- from/to stage index 合法且单调；
- from stage receipt 已 sealed；
- reader fences issued == completed；
- pending readers after == 0；
- 若 policy 要求 release，则 `live_slot_bytes_after == 0`；
- source generation 与整个 request 相同；
- next stage 尚未提交首个 fill 时 boundary 已记录 release。

### 3.5 Multi-stage public verifier

`verify_and_attach_public_streaming_result()` 改为：

```text
1. authority/snapshot/layout/source 基础校验
2. layout.stages.size == result.streaming_stages.size
3. receipt.stages.size == layout.stages.size
4. 对每个 stage 调 verify_actual_stage_receipt
5. 验证 stage metrics 与 layout/record 一致
6. 验证 boundaries.count == required boundaries.count
7. 验证每个 boundary 的 drain/release/reader/source generation
8. lease.revalidate_after_drain
9. 验证 execution receipt canonical digest
10. 填写 public selection metrics 和安全摘要
```

单 stage record 要求 `boundaries.empty()`；多 stage record 的 required boundary 应进入 plan identity，而不是由 adapter 自由决定。

### 3.6 Catalog plan 的 multi-stage 表达

当前 `PresetPlan` 只有一个 `canonical_config/layout_digest`，仍可以覆盖整个多 stage layout；但为了审阅和工具链，建议 additive 增加只读摘要：

```cpp
struct PresetStageSummary {
    std::string id;
    std::string stage_layout_digest;
    uint32_t prefix = 0, group = 0, slots = 0;
    uint32_t distance = 0, workers = 0;
    std::string pass_transition, multi_pool_policy;
};

struct PresetBoundaryPolicy {
    std::string id, from_stage, to_stage;
    bool require_drain = true;
    bool require_release = true;
    bool allow_component_overlap = false;
};
```

这些摘要必须从 canonical layout 生成并由 catalog builder 校验，不能成为第二份可编辑 plan。

## 4. H3 Turbo public adapter 详细设计

### 4.1 首版 public card

首版只支持：

```text
model: minimax-h3-turbo
operation: video.generate
backend: native C/Metal GPU
precision: original BF16
inputs: text only
audio: false
active blocks: all H3_DIT_BLOCKS
core reuse interval: 1
token reduction / first-block cache / tea cache / step gate: off
LoRA / quantized cache / ANE / hybrid / approximation: off
streamed stages: DiT denoiser only
```

普通 H3、VDN/VSA、FastH3、量化 cache、有音频、多模态 reference、image-to-video 都是不同 workload card，首版必须返回 `streaming_route_unsupported`。

### 4.2 两层 source closure

H3 需要区分：

1. **request execution closure**：影响完整结果的 tokenizer、text encoder、FL2VA transformer、video VAE、scheduler/config；
2. **streamed source closure**：本次 slot reader 实际读取的 transformer shards/ranges。

设计要求：

- `PresetSourceIdentity.artifact_manifest_digest` 覆盖完整 execution closure；
- `Descriptor.artifacts` 和每个 `SourceRange` 覆盖 streamed closure；
- 两者共享一个 request `SourceLease`；
- 当前参考模型若是 13 个 transformer shard，probe 必须从 manifest/index 得到“恰好 13”并记录，不能把 13 硬编码为所有 checkpoint 的通用事实；
- 任意 config/index/shard/tokenizer/VAE 文件不在 closure 中，public probe 失败；
- `StreamingMetadata` 不再通过目录扫描后按 path reopen payload。

建议新增 model-specific closure builder：

```text
native/models/h3_runtime/h3_public_streaming_source.hpp
native/models/h3_runtime/h3_public_streaming_source.cpp
```

```cpp
struct H3PublicSourceClosure final {
    std::vector<streaming::SourceFileIdentity> execution_files;
    std::vector<std::string> transformer_logical_ids;
    streaming::PresetSourceIdentity identity;
};

H3PublicSourceClosure inspect_h3_public_source(
    const std::filesystem::path& model_root,
    const Request& normalized_request);
```

### 4.3 `StreamingMetadata` lease 化

当前 `StreamingMetadata(transformer_directory)` 使用目录扫描、`stat` 和 `h3_weight_store_open(path)`。public 路径应增加独立 constructor：

```cpp
StreamingMetadata(
    std::shared_ptr<const streaming::SourceLease> lease,
    std::vector<std::string> transformer_logical_ids);
```

要求：

- metadata header/parser 从 `duplicate_fd(logical_id)` 读取；
- tensor source range 绑定 logical id、offset、bytes、dtype、shape；
- `check_unchanged()` 在 public 路径调用 lease revalidation，不再重新扫描目录；
- private candidate constructor 保留，避免影响已有 P1；
- 两个 constructor 必须生成相同 canonical descriptor，增加 golden equality test。

H3 C weight store 若只能接受 path，应增加 fd/source callback 版本，而不是使用 `/dev/fd/<n>` 后提前关闭 fd：

```c
typedef struct {
    void *user;
    int (*pread)(void *user, uint32_t artifact,
                 uint64_t offset, void *dst, size_t bytes,
                 char *error, size_t error_size);
    uint64_t (*size)(void *user, uint32_t artifact);
} h3_weight_source_v1;
```

public `h3_weight_store_open_source_v1()` 持有 source user 至最后一个 reader fence 完成；private/default 的 path API 保持不变。

### 4.4 H3 session 三段式 hook

在 `H3Session` 中增加：

```cpp
std::shared_ptr<const streaming::ModelStreamingProbe>
probe_public_streaming(const streaming::PublicResolveInput&) const override;

std::shared_ptr<const streaming::ModelStreamingSnapshot>
compile_public_streaming(
    std::shared_ptr<const streaming::ModelStreamingProbe>,
    const streaming::StreamingPresetRecord&) const override;

RunResult generate_resolved(
    std::shared_ptr<const streaming::ResolvedRequestExecution>,
    const Event&, std::atomic<bool>&) override;
```

`probe`：

- 只做 normalized request route validation；
- 构建完整 source closure 并 `SourceLease::capture()`；
- 从 lease 构建 workload/runtime/source/component policy identity；
- 不创建 `h3_ctx`、Metal buffer、refill worker 或 VAE。

`compile`：

- dynamic cast/验证 probe 类型；
- 用同一 lease 的 metadata 编译 descriptor/layout；
- 要求 record model/source/workload/runtime/component/layout 全等；
- public plan 允许先上 `reload/K1`，再单独 record 上 `carry/K2`；
- 不因现有 private gate 允许而自动授予 public。

`generate_resolved`：

- 先验证 authority、snapshot、lease 指针和 source generation；
- 使用 RAII request binding，把 exact config/lease/receipt sink 绑定到一次 `generate()`；
- 调用内部共享实现 `generate_impl(request, public_context?)`，不复制完整 H3 pipeline；
- public context 非空时绕过 `allow_experimental_streaming_` 的 private constructor gate，但只能接受 authority 匹配；
- 返回真实 receipt，并在 common verifier 前保持 lease/context 存活。

推荐重构：

```cpp
RunResult generate_impl(
    const Request&, const Event&, std::atomic<bool>&,
    streaming::PublicStreamingRunContext* public_run);
```

`generate()` 始终传 `nullptr`；`generate_resolved()` 构造 context 后传非空。这样 default 热路径只有一次静态分支，不创建 public 对象。

### 4.5 H3 C bridge 完整 receipt

当前 `tc_stream_receipt_v1` 只有 summary，不能让 common verifier检查每个 group/fence/carry。不能从 `exact_fills` 等 counters 合成。

建议 additive opaque receipt ABI：

```c
#define TC_STREAM_RECEIPT_ABI_V2 2u

typedef struct tc_stream_owned_receipt_v2 tc_stream_owned_receipt_v2;

int tc_stream_executor_clone_receipt_v2(
    tc_stream_executor *, tc_stream_owned_receipt_v2 **out,
    char *error, size_t error_size);

int tc_stream_owned_receipt_shape_v2(
    const tc_stream_owned_receipt_v2 *,
    tc_stream_receipt_shape_v2 *out,
    char *error, size_t error_size);

int tc_stream_owned_receipt_copy_groups_v2(...);
int tc_stream_owned_receipt_copy_readers_v2(...);
int tc_stream_owned_receipt_copy_pools_v2(...);
int tc_stream_owned_receipt_copy_carries_v2(...);
void tc_stream_owned_receipt_free_v2(tc_stream_owned_receipt_v2 *);
```

合同：

- clone 只允许 successful finish 后调用；
- opaque handle 内部持有 immutable `ActualStageReceipt`；
- copy API 使用 caller-owned buffer 和明确 capacity/count，不跨 ABI 返回 STL；
- 每个 group 带 reader offset/count，所有整数做 checked conversion；
- free 可从任意 cleanup path 调用且不触碰 GPU；
- clone/copy 失败不能退回 summary-only success。

H3 侧：

```c
struct h3_result {
    /* existing fields */
    tc_stream_owned_receipt_v2 *exact_receipt;
};
```

- `h3_generate()` 在 DiT finish 后、destroy executor 前 clone receipt 到 result；
- `h3_result_free()` 释放 opaque receipt；
- `H3Session` 把 C POD arrays 转成 common `ActualStageReceipt`；
- 转换后重新计算 event/canonical digest，并与 C 侧 digest 全等；
- public run 任何 receipt 缺失/shape mismatch 都失败；private candidate 可继续只读 counters，避免改变旧接口行为。

### 4.6 Reload 与 carry 的 record 隔离

H3 至少保留两个候选族：

```text
h3-reload-k1: K1 / D0 / Q1 / reload
h3-carry-k2:  K2 / D1 / Q1或Q2 / carry_first_group
```

二者必须有不同：

- preset id/revision；
- layout digest；
- pass transition；
- implementation/runtime identity（若 adapter revision 不同）；
- P1 evidence。

如果 K1/reload 只作为 correctness baseline、性能不达标，可以保持 candidate，不必发布。不能让 resolver 在一次请求中从 carry 自动退到 reload。

### 4.7 H3 专项测试

新增建议：

```text
tests/native/h3_public_streaming_test.cpp
tests/native/test_h3_public_streaming.py
tests/native/h3_stream_receipt_bridge_test.c
```

最小测试 ID：

```text
H3-PUB-001 exact Turbo route accepted
H3-PUB-002 ordinary/quant/ANE/audio/input/approx route rejected
H3-PUB-003 full execution closure complete
H3-PUB-004 probe/snapshot/reader share one lease
H3-PUB-005 path replacement after resolve fails pre-GPU
H3-PUB-006 same inode mutation fails post-drain
H3-PUB-007 reload K1 full receipt verified
H3-PUB-008 carry K2 carry event verified
H3-PUB-009 C receipt group/reader/carry copy bounds
H3-PUB-010 cancel during first fill drains or quarantines
H3-PUB-011 failure request does not poison next default request
H3-PUB-012 default constructor does zero public work
H3-PUB-013 real full request output equals resident baseline
H3-PUB-014 P0/P1 pass
```

## 5. LTX 2.5 public multi-stage adapter

### 5.1 首版 public card

建议先冻结最小可闭环 workload：

```text
model: ltx-2.5-distilled
operation: video.generate
backend: c_metal
execution: gpu
input: text only
audio: false
steps: distilled 8 + 3
frames/fps/shape: record exact match
approximation/Sol/sparse: off
ANE/MLX backend/LoRA: off
worker: per request
```

音频、image-to-video、Sol/fast_approx、ANE 和 C++/MLX 是独立 workload/card，首个 public 闭环不应同时承担这些分支。

### 5.2 Layout 的 stage 划分

建议 descriptor 至少产生：

```text
stage[0] id = "ltx-stage1-denoiser"
stage[1] id = "ltx-stage2-denoiser"
```

upsampler、VAE、export 不是 block streaming stage，但属于 component policy 和 memory ledger。record 的 boundary policy：

```text
boundary[0]: stage1 -> upsampler
  require_drain = true
  require_release = true
  allow_component_overlap = false  // v1

boundary[1]: stage2 -> video_vae
  require_drain = true
  require_release = true
  allow_component_overlap = false  // v1
```

高内存候选将来允许 overlap 时，必须生成新的 component policy revision 和 record；不能运行时自行开启。

### 5.3 LTX source closure

完整 closure 至少包括：

- stage1/stage2 transformer checkpoint、header/index、量化 metadata（若 card 允许）；
- Gemma/tokenizer/template/connector；
- spatial upsampler 权重/config；
- video VAE 权重/config；
- audio VAE/vocoder/BWE（仅 audio card）；
- scheduler、latent stats、export/ffmpeg route manifest；
- 影响 conditioning cache 的 source identity。

要求：

- worker 内 capture lease；desktop 只保存路径和 selector intent；
- conditioning cache 若命中，cache manifest 必须包含 source/workload/runtime digest；
- cache 文件不是 authority，source identity 不匹配即拒绝；
- stage1 与 stage2 reader 只使用 lease duplicate fd；
- private `StreamingMetadata(checkpoint)` 保留，public 增加 lease constructor。

### 5.4 Worker-local authority

desktop 到 worker 只发送可序列化 intent：

```swift
struct LTXJobEnvelopeV2: Codable, Sendable {
    let schemaVersion: Int
    let request: NativeRequestV2
    let modelPath: String
    let catalogHintRevision: String?
    let jobID: UUID
    let outputTransactionID: UUID
}
```

禁止发送：

- exact authority object；
- source fd；
- `ResolvedRequestExecution` 指针；
- desktop resolve 得到的 lease generation；
- Metal buffer/MLX array；
- 把 exact preset 当作已经授权执行的 JobStore 状态。

worker 内部顺序：

```text
decode envelope
  -> validate request-only fields
  -> open native engine in worker
  -> query immutable catalog snapshot
  -> probe + capture worker-local SourceLease
  -> resolve exact record + authority
  -> acquire worker GPU lock
  -> revalidate authority/source/device
  -> execute stage1/boundary/stage2/boundary/VAE/export
  -> common result verifier
  -> fsync temporary result and media
  -> atomic rename to final output
  -> write terminal success envelope
```

desktop 的 `resolveStreaming()` 可用于 UI 预览和错误提示，但 worker 必须在 generate 时重新 resolve；两次结果不同则以 worker 为准，并把 stale/revoked 明确返回给 UI。

### 5.5 原子输出协议

worker 不直接写用户最终路径：

```text
final:    /path/video.mp4
staging:  /path/.video.mp4.tc-<transaction>.partial
result:   temp/result.json.partial
```

成功提交：

1. encoder/ffmpeg 正常退出；
2. media file 可打开且基础 metadata 正确；
3. public result verifier 通过；
4. staging media `fsync`；
5. result JSON `fsync`；
6. 同一 filesystem 原子 rename media；
7. 写 terminal success。

任一失败：

- final output 不可见；
- partial 文件进入可清理状态；
- JobStore 不写 `succeeded`；
- receipt/authority/source 错误不允许 fallback 到 legacy component-staged。

### 5.6 Worker 取消、EOF 与 quarantine

| 故障 | worker 行为 | desktop 状态 | 是否允许同进程重试 |
|---|---|---|---|
| 用户取消，drain 成功 | 停止新 fill，完成 drain，删除 partial | cancelled | 新 worker 可以 |
| 5 秒 grace 后 SIGKILL | OS 回收 worker；partial 不提交 | cancelled/worker_killed | 必须新 worker |
| IPC EOF，进程仍在 | supervisor terminate，超时 kill | worker_lost | 必须新 worker |
| reader timeout/drain unknown | worker 标记 quarantine 后退出 | failed/quarantined | 不允许 |
| source stale | GPU 前拒绝或 post-drain fail | failed/source_stale | 重新 resolve 新 worker |
| output rename failure | 保留 verified temp 供人工恢复标记 | failed/output_commit | 可显式重试提交，不自动生成 |

由于 LTX 本来就是 per-request worker，process-local quarantine 的最安全策略是让 worker 退出；desktop 进程不能接管不透明 GPU backing。

### 5.7 LTX session 改造

`native/platform/apple/ltx_session.mm` 建议分为：

```cpp
RunResult generate_impl(
    const Request&, const Event&, std::atomic<bool>&,
    streaming::PublicStreamingRunContext*);

LtxExactStageResult run_exact_stage(
    uint32_t stage_index,
    const streaming::StageLayout&,
    LtxStageInputs&,
    streaming::PublicStreamingRunContext&);

void run_public_boundary(
    const PresetBoundaryPolicy&,
    LtxComponentState&,
    streaming::PublicStreamingRunContext&);
```

迁移原则：

- 复用现有 `ltx_streaming_adapter.inc` 的真实 Metal fill/encode；
- 不另建一套 slot scheduler；
- `LtxExactRequestState` 改成由 run context 所有；
- process quarantine registry 继续服务 private fault tests，但 public worker 遇到 quarantine 应退出；
- stage-specific P/G/K/D/Q 来自 layout，不从环境变量或 session 成员二次解析；
- stage1 和 stage2 分别产生 receipt；
- upsampler/VAE 前明确调用 release 并记录 boundary。

### 5.8 LTX 专项测试

```text
LTX-PUB-001 text-only GPU route accepted
LTX-PUB-002 audio/i2v/ANE/MLX/Sol/LoRA rejected by v1 card
LTX-PUB-003 worker captures authority locally
LTX-PUB-004 desktop resolution cannot authorize worker execution
LTX-PUB-005 stage1 receipt verified
LTX-PUB-006 stage1 boundary drain/release verified
LTX-PUB-007 stage2 receipt verified
LTX-PUB-008 stage2->VAE boundary verified
LTX-PUB-009 source replacement between desktop and worker fails
LTX-PUB-010 worker EOF and SIGKILL never commit output
LTX-PUB-011 cancel grace drains or kills worker safely
LTX-PUB-012 output rename failure does not mark success
LTX-PUB-013 next job unaffected after failed worker
LTX-PUB-014 real full request output/quality parity
LTX-PUB-015 P0/P1 pass
```

## 6. 统一错误与重试合同

### 6.1 稳定错误 envelope

所有 public path 错误最终映射为：

```json
{
  "schema_version": 1,
  "code": "streaming_source_stale",
  "phase": "pre_gpu_revalidate",
  "message": "模型文件在解析后发生变化，请重新开始任务。",
  "retryable": true,
  "retry_scope": "new_resolution",
  "safe_to_reuse_engine": true,
  "safe_to_reuse_worker": false,
  "request_digest": "sha256:...",
  "resolution_digest": "sha256:..."
}
```

建议错误类别：

| code | retry scope | engine/worker |
|---|---|---|
| `streaming_request_invalid` | none until user changes request | reusable |
| `streaming_target_unavailable` | select another explicit target | reusable |
| `streaming_catalog_stale` | new options + new resolve | reusable |
| `streaming_source_stale` | new source probe/resolve | depends on phase |
| `streaming_authority_mismatch` | new resolve | reusable before GPU |
| `streaming_reader_timeout` | new process/session | quarantine |
| `streaming_drain_unknown` | new process/session | quarantine |
| `streaming_receipt_mismatch` | code/catalog defect | do not retry silently |
| `streaming_worker_lost` | new worker | desktop reusable |
| `streaming_output_commit_failed` | explicit commit/re-run | no fake success |

### 6.2 禁止 fallback 的位置

以下错误不得自动切回 resident、private candidate、另一档位或旧 `residency=streamed`：

- record/identity/layout mismatch；
- source lease stale；
- actual receipt mismatch；
- drain unknown/quarantine；
- selected target 超限；
- worker lost；
- catalog revoked/stale。

用户可以在新请求中主动选择 Off；这是新的用户决策，不是当前失败请求的 fallback。

## 7. 默认路径性能隔离

### 7.1 代码结构要求

Off/default 请求必须在 `tc_engine_generate()` 的 selector 分支处完全旁路。禁止为了统一调用链：

- 构造空 `PublicStreamingRunContext`；
- 查询 catalog；
- capture source lease；
-扫描 model files；
- 创建 receipt vector；
- 启动 I/O worker；
-读取 physical/free memory；
- 安装 audit hook；
- 改变 default session cache 生命周期。

模型内部推荐：

```cpp
RunResult generate(const Request& r, ...) override {
    return generate_impl(r, ..., nullptr);
}

RunResult generate_resolved(... execution, ...) override {
    PublicStreamingRunContext context(std::move(execution), cancel);
    return generate_impl(context.execution().request, ..., &context);
}
```

`generate_impl()` 的 default 热循环不应反复检查 context。进入模型前解析一次布尔值，把 public-specific setup 放在分支外侧；block kernel 内不得增加 virtual call 或 authority check。

### 7.2 Audit 必须新增的计数

```text
public_catalog_snapshots
public_model_probes
public_source_leases
public_run_contexts
public_stage_executors
public_receipt_recorders
public_io_workers
public_boundary_recorders
public_quarantine_transfers
```

P0 Off/default 每项都必须为 0。已有 generic/private streaming 的计数应分开，避免 private P1 被误判为 public work。

## 8. 分阶段实施与停止条件

### R3 · Common multi-stage receipt

改动：

```text
session.hpp
actual_receipt.hpp/.cpp
public_result.cpp
run_context.hpp/.cpp
serialization/to_dictionary
tests/native/streaming_actual_receipt_test.cpp
```

完成条件：single-stage v2 replay 完全兼容；multi-stage/boundary host tests 和 ASan/UBSan/TSan 通过。

停止条件：default audit 非零、boundary 可以在 stage2 开始后补写、失败 receipt 仍能 seal success。

### R4 · H3 receipt bridge 与 public hooks

改动：

```text
stream_slot_c.h / c_bridge.cpp
h3_dit.h / h3_dit.c
h3.h / h3.c / h3_result_free
h3_streaming_descriptor.*
h3_session.mm
H3 public tests
```

完成条件：reload 和 carry receipt 都由真实 C executor导出；真实 H3 Turbo full request 通过 common verifier。

停止条件：receipt 从 counters 合成、path reader fallback、普通 H3/ANE/quant route 可进入。

### R5 · LTX worker-local multi-stage

改动：

```text
ltx_streaming_descriptor.*
ltx_streaming_plan.*
ltx_streaming_adapter.inc
ltx_session.mm
LTXWorker.swift
CLI schema-v2 generate
LTX public/worker tests
```

完成条件：两个 stage 和两个 boundary verified；worker kill/EOF/rename failure 均无可见假成功。

停止条件：desktop 持有 authority/fd、worker 复用 stale exact selector、stage1 backing 跨 boundary 未声明保留。

### R6 · App 与校准

R6 在 [52](52-public-streaming-app-config-and-model-tier-spec.md) 中展开。只有 R3–R5 合并并取得真实 GPU smoke 后，才允许生成 staging record。

## 9. 验收矩阵

### 9.1 Host/contract

```bash
env TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
make test-streaming-host
make test-streaming-contract
make test-streaming-source-lease
make test-streaming-audit
python3 -B tests/native/test_h3_public_streaming.py
python3 -B tests/native/test_ltx_public_streaming.py
```

要求：

- 不是 skip 的项目全 PASS；
- production catalog 仍可保持 empty；测试使用 test catalog/provider；
- `git diff --check` 通过；
- release binary 不导出 test-only catalog setter 和 fault hook。

### 9.2 Sanitizer

至少覆盖：

```text
ASan+UBSan: group/reader copy bounds, short read, cancel, exception cleanup
TSan: mailbox post/consume, cancel race, receipt owner-only, worker event reader
```

TSan 环境若不能运行 Metal，使用 synthetic adapter，但真实 Metal 的生命周期仍需单独 smoke。

### 9.3 真实 GPU smoke

每模型至少保存：

```text
request JSON
exact selected record
source/runtime/device/workload digests
full actual receipt
output digest/quality report
process-tree timeline
failure/cancel result
binary and commit digest
```

H3 和 LTX 的 smoke 必须使用真实 checkpoint、真实 output component 和 common public generate，不接受只运行 descriptor 或 block probe。

### 9.4 P0/P1

沿用正式门槛：

```text
P0 default resident:
  median ratio <= 1.02
  P95 ratio <= 1.05
  public allocation/thread/hook = 0

P1 same exact layout generic/public vs direct/private:
  median ratio <= 1.02
  P95 ratio <= 1.05
  output equivalent
  receipt/layout semantic identity equal
```

H3 合理运行波动可以存在，但正式非劣门仍按统计规则判断；若样本方差导致 inconclusive，应增加预先规定的样本，不可事后放宽阈值。

## 10. Code review checklist

### Common runtime

- [ ] authority 只由 resolver 创建；
- [ ] request context 不可复制且不跨请求；
- [ ] receipt/boundary 事件发生时记录，不在结果层补写；
- [ ] multi-stage verifier 检查全部 stage，不只检查 summary；
- [ ] source revalidate 在 drain 后、release 前完成；
- [ ] drain unknown 转 quarantine；
- [ ] Off 路径零 public work。

### H3

- [ ] 只允许 MiniMax H3 Turbo original BF16 GPU card；
- [ ] execution closure 与 streamed closure 都完整；
- [ ] metadata/reader 使用 lease fd lineage；
- [ ] C receipt 保留 group/fence/carry，不从 counters 合成；
- [ ] private gate 与 public authority gate 分离；
- [ ] cancel/failure 后下一 default request 不受污染。

### LTX

- [ ] authority/lease 在 worker 内创建；
- [ ] stage1/stage2 plan 独立；
- [ ] boundary drain/release 有 receipt；
- [ ] output 只在 verifier 后原子提交；
- [ ] worker EOF/SIGKILL 不产生 success；
- [ ] audio/i2v/ANE/approx 未认证 route fail-closed。

## 11. 完成后仍不能宣称的内容

即使 R3–R5 全部完成，也只能说明模型 runtime public 闭环完成。以下仍需 [50](50-public-streaming-calibration-and-release-evidence.md) 和 [52](52-public-streaming-app-config-and-model-tier-spec.md) 的证据：

- 8/10/12/16/20 GiB 中哪些档位真的 fit；
- streaming 是否比自然 swap 更快；
- App 推荐是否适合具体物理内存；
- production catalog record 是否可发布；
- ANE/hybrid 是否支持；
- 未校准 shape/audio/input/LoRA 是否支持。

只有真实 P0/P1/P2/P3、App 事务、catalog review 和 revoke 演练全部完成后，才可以把对应“模型 + workload + target”标为 public available。

## 12. 文件级代码修改矩阵

| 文件/目录 | 当前职责 | 建议修改 | 关键验收 |
|---|---|---|---|
| `native/runtime/session.hpp` | 单 stage runtime metrics/result | additive stage/boundary metrics；保留 legacy optional | single-stage serialization 不变 |
| `native/runtime/streaming/actual_receipt.*` | stage receipt 和 execution digest | receipt v3、boundary receipt、v2 replay | digest golden、missing boundary fail |
| `native/runtime/streaming/context.*` | 单 `StageExecutor` owner pump | 保持 executor 单 stage，不塞入 request orchestration | executor P1 不回退 |
| `native/runtime/streaming/run_context.*` | 尚不存在 | request-scoped stage/boundary/drain/quarantine owner | exception/cancel cleanup |
| `native/runtime/streaming/public_result.*` | v1 单 stage hard gate | stage loop、boundary verifier、legacy summary compatibility | single/multi-stage host tests |
| `native/runtime/streaming/resolved_request.*` | immutable authority/snapshot | additive component/boundary identity（若 record schema需要） | authority digest stable |
| `native/runtime/streaming/preset_catalog.*` | plan/calibration record | additive stage summary/boundary policy/headroom revision | canonical digest golden |
| `native/core/stream_slot_c.h` | C executor + summary receipt | opaque full receipt clone/copy/free ABI v2 | bounds/ABI C tests |
| `native/runtime/streaming/c_bridge.cpp` | C→C++ StageExecutor wrapper | immutable receipt handle、checked POD copy | ASan/UBSan/TSan |
| `native/models/h3_runtime/h3_streaming_descriptor.*` | path metadata descriptor | shared lease constructor、execution/streamed closure | path-vs-lease descriptor equality |
| `native/models/h3_runtime/h3_weights.*` | path-based C store | additive source callback/fd-backed store | short read/stale/no fallback |
| `native/models/h3_runtime/h3_dit.*` | private exact C executor | enable receipt、clone before destroy、public source generation | carry/group/fence receipt |
| `native/models/h3_runtime/h3.*` | full H3 pipeline/result | result owns opaque receipt；free path | no leak/double free |
| `native/platform/apple/h3_session.mm` | default/private generate | public hooks、request binding、receipt conversion | route gate/full request/P0 |
| `native/models/ltx_runtime/ltx_streaming_descriptor.*` | single checkpoint descriptor | lease constructor、stage1/stage2 descriptor | stage-specific digests |
| `native/models/ltx_runtime/ltx_streaming_plan.*` | private exact plan | multi-stage canonical plan/boundaries | compiler goldens |
| `native/models/ltx_runtime/ltx_streaming_adapter.inc` | C exact slot adapter | request-context ownership、stage receipt export | lifecycle/fault tests |
| `native/platform/apple/ltx_session.mm` | monolithic LTX generate | public hooks、`run_exact_stage`、boundary recorder | full worker smoke/P0/P1 |
| `apps/macos/LTXWorker.swift` | legacy component-staged worker | V2 envelope、terminal envelope、atomic output supervision | EOF/SIGKILL/cancel |
| `native/api/c_api.mm` | resolve/generate transaction | multi-stage serializer/error envelope；Off 旁路不变 | contract/audit |
| `bindings/swift/TurboCiderNative.swift` | V2 types/options/resolve/generate | optional selector、multi-stage safe summary、typed errors | Swift round-trip |
| `apps/macos/StudioState.swift` | legacy residency fields | Off/五档 UI state 与 migration | old draft fixtures |
| `apps/macos/JobStore.swift` | legacy request persistence | V2 intent、安全 summary、stale/revoke transaction | App integration |
| `tools/native/run_streaming_campaign.py` | P0/P1 ABBA campaign | P2 memory/P3 four-arm metadata和receipt/boundary digest | synthetic verifier fixtures |
| `tools/native/verify_streaming_campaign.py` | performance/quality verifier | process-tree/gap/swap/target/boundary gate | tamper/failure tests |

实施 review 时逐行确认：模型文件不能 include App/catalog implementation；common executor 不应依赖 H3/LTX；Swift 不得持有 native authority；工具链不能修改 production runtime 的选择结果。

## 13. 每个阶段的最小提交边界

建议保持可回滚的小提交：

```text
R3a  receipt v3 data model + v2 compatibility tests
R3b  boundary recorder + public multi-stage verifier
R3c  request-scoped run context + failure/quarantine tests
R4a  C full-receipt opaque ABI
R4b  H3 lease metadata/source callback
R4c  H3 public hooks + host tests
R4d  H3 real GPU smoke/P0/P1 evidence（证据提交与代码分开）
R5a  LTX multi-stage descriptor/compiler
R5b  LTX session stage/boundary adapter
R5c  worker-local V2 transaction + atomic output
R5d  LTX real GPU smoke/P0/P1 evidence
R6   App/config/calibration，按文档52再拆
```

每个代码提交必须满足：

1. 本提交的 host/contract 测试通过；
2. `git diff --check` 通过；
3. Off/default audit 不增加；
4. production catalog 仍为空，除非该提交本身是经 review 的 release commit；
5. 不把 evidence、生成媒体、model weights 或机器绝对路径混入代码提交；
6. 下一提交可以在不破坏 default path 的情况下独立回滚。

dev 合并安排在 H3/LTX/App主要接口稳定后。合并改变 kernel/reader/source/runtime identity 时，旧 evidence 必须全部标记 stale 并重跑，不能只解决编译冲突后沿用旧成绩。
