# 54 · Public Streaming 深度实施、代码接口与验收附录

修订日期：2026-09-18。状态：实施规格，尚未完成 public release。

本文是 [53 Final Implementation Closure](53-public-streaming-final-implementation-closure.md) 的代码级补充，不建立第二套 scheduler、第二套内存定义或第二套验收门槛。目标是把设计继续下钻到对象字段、调用顺序、线程边界、错误处理、模型 adapter 接线、工具输入输出、实验记录和每个 PR 的停止条件。

本文状态标签固定为：`DONE`（已提交且有证据）、`WIP`（已有部分实现）、`NEXT`（可编码但未实现）、`BLOCKED`（依赖设备/真实模型/evidence）、`NOT_PUBLIC`（private 可运行但不得进入 public catalog）。除非明确标为 `DONE`，否则不得将接口、record、性能数字或模型组合描述为已支持。当前 production catalog 仍为 `tc-streaming-catalog-empty-v1`。

---

## 1. 产品目标和不可改变的边界

### 1.1 用户只选择 target

App 高级设置只暴露：

```text
Off / 8 GiB / 10 GiB / 12 GiB / 16 GiB / 20 GiB
```

用户选择的是“完整进程树的内存目标”，不是物理显存容量、不是操作系统 hard cap，也不是 `slot_count`。native 根据 reviewed catalog 为 `(model, source, workload, device, target)` 选择 exact record；用户看不到 `P/G/K/D/Q`、pool policy、pass transition 或 source range。

物理内存只用于推荐和 eligibility 检查，不能用当前 free memory、swap 活跃度或一次运行的偶然可用空间推导 plan。若没有精确 record，必须返回 unavailable；不得自动换邻近档位、resident 或自然 swap。

### 1.2 三条路径

| 路径 | 入口 | public 对象 | 约束 |
|---|---|---:|---|
| Off/default | 原 `generate()`/`prepare()` | 否 | 不读 catalog、不 probe、不创建 context、不启动 sampler |
| legacy/private | `residency=streamed`、`component_staged` 或内部 exact | 否/旧合同 | 保留旧 gate，不得伪装为 public |
| public target | `streaming_selector.active()` | 是 | 每请求 resolve/revalidate/receipt，未认证组合 fail-closed |

如果 public selector 与旧 manual streaming config 同时存在，解析阶段直接拒绝，不能猜优先级。旧 `residency` 和 `memory_budget_bytes` 的迁移语义仍由 legacy 路径负责。

### 1.3 性能承诺

“性能不能差”拆成三个独立门：

1. **默认零开销**：Off/default 不增加 selector/catalog/probe/context/sampler 热路径工作。
2. **同布局非劣**：相同 layout、checkpoint、workload 的 public executor 与 reference runner 通过 P1。
3. **低内存收益实测**：streaming 与 resident、bounded、natural-swap 的峰值和 wall time 通过同机四臂实验确认。

沿用 [12 Acceptance Playbook](12-acceptance-playbook.md) 的门槛：

```text
P0 default median ratio <= 1.02, P95 <= 1.05
P1 same-layout median ratio <= 1.02, P95 <= 1.05
P2 process-tree peak P95 + max(512 MiB, 10%) <= target
P3 streaming vs natural-swap 必须实测，不从设计推导
```

---

## 2. 当前代码对象图和所有权

### 2.1 对象图

```text
Request
  └─ StreamingSelector                       用户意图
      └─ immutable StreamingPresetCatalog    reviewed record 快照
          └─ StreamingPresetRecord
              └─ ResolvedStreamingSelection
                  └─ StreamingAuthority      不可构造/不可序列化

ModelStreamingProbe                          metadata-only
  └─ SourceLease                             request-scoped source lineage
ModelStreamingSnapshot                       descriptor + layout + same lease
  └─ ResolvedRequestExecution                immutable execution ticket
      └─ PublicStreamingRunContext            mutable owner
          ├─ StageExecutor[0..N)
          ├─ ActualReceiptRecorder
          ├─ ActualBoundaryReceipt[]
          └─ quarantine state
```

以下对象不能跨请求共享：`ResolvedRequestExecution`、`StreamingAuthority`、`SourceLease`、slot backing、completion mailbox、receipt recorder。immutable layout、catalog snapshot 和 identity 可在同一请求内跨线程只读；ticket、slot state、receipt vector 只能由 owner 线程修改。

### 2.2 `ModelSession` 三个 public 接口

当前唯一允许的模型接缝是：

```cpp
virtual std::shared_ptr<const ModelStreamingProbe>
probe_public_streaming(const PublicResolveInput &) const;

virtual std::shared_ptr<const ModelStreamingSnapshot>
compile_public_streaming(
    std::shared_ptr<const ModelStreamingProbe>,
    const StreamingPresetRecord &) const;

virtual RunResult generate_resolved(
    std::shared_ptr<const ResolvedRequestExecution>,
    const Event &, std::atomic<bool> &);
```

实现合同：

1. `probe_public_streaming()` 只读 checkpoint index/header、manifest、shape 和 request identity；不能分配 GPU buffer、编译 graph 或启动 worker。
2. `compile_public_streaming()` 生成 descriptor、layout、source ranges、component policy 和 request-scoped `SourceLease`；不能提交 GPU work。
3. `generate_resolved()` 只消费已授权 execution，不能根据 free memory 重新选档或 fallback。
4. adapter 必须返回真实 `RunResult.streaming_receipt`；不得从 counters 合成 receipt。

### 2.3 Resolved execution 不变量

调用 `generate_resolved()` 前必须成立：

```text
probe != null, snapshot != null
probe.model_id == engine_model_id
probe.workload_identity == catalog.record.workload
snapshot.layout.digest == record.plan.layout_digest
snapshot.component_policy_revision == record.plan.component_policy_revision
probe.source_lease == snapshot.source_lease
source_lease.generation != 0
authority.matches(record, snapshot, current_device)
```

`PublicStreamingCoordinator::resolve_normalized()` 会清除 request 中的 selector，写入 catalog record 的 canonical `StreamingConfig`。这样 adapter 不会二次解释 target，result 也不会把 selector 当作 execution authority。

---

## 3. Public transaction 的精确顺序

### 3.1 C API

`native/api/c_api.mm` 的 public 分支必须维持以下顺序：

```cpp
request = request_from_json(parse_json(input));
if (request.streaming_selector && request.streaming_selector->active()) {
    require(!engine.streaming_quarantined);
    public_execution = resolve_public_streaming_locked(
        engine, std::move(request));
    request = public_execution->request;
}

lock engine;
lock process_wide_execution;
acquire DeviceLease;

if (public_execution)
    revalidate_public_streaming_locked(engine, *public_execution);

result = public_execution
    ? session->generate_resolved(public_execution, event, cancelled)
    : session->generate(request, event, cancelled);

if (public_execution)
    verify_and_attach_public_streaming_result(*public_execution, result);

serialize_result_atomically(result);
```

catalog preflight 必须早于 model materialization；global GPU lock 必须早于 source/device revalidation 和 reader open；receipt verification 必须早于输出 JSON 写入。失败时不产生半成品结果。

### 3.2 Swift/JobStore

App 侧只生成 selector：

```swift
enum StreamingTarget: Codable, Equatable {
    case off, gib8, gib10, gib12, gib16, gib20
}

struct PublicStreamingSelector: Codable, Equatable {
    var enabled: Bool
    var targetRequestMemoryBytes: UInt64?
    var catalogRevision: String?
}
```

JobStore 顺序：

```text
draft snapshot
→ local validation
→ native streaming_options (read-only)
→ reject unavailable/revoked
→ persist request + selector atomically
→ start generate
→ verify job id/request digest
→ persist result/error atomically
```

native 发现 source、device、record 或 authority 变化时 fail-closed；JobStore 不能捕获错误后自动改成 Off 重试。用户必须明确重新选择 Off 或另一个可用 target。

建议的 read-only options 响应：

```json
{
  "schema_version": 1,
  "model": "flux2-klein-9b",
  "catalog_revision": "tc-streaming-catalog-empty-v1",
  "targets": [
    {"target_gib": 8, "state": "unavailable", "reason": "no_reviewed_record"},
    {"target_gib": 10, "state": "unavailable", "reason": "no_reviewed_record"},
    {"target_gib": 12, "state": "unavailable", "reason": "no_reviewed_record"},
    {"target_gib": 16, "state": "unavailable", "reason": "no_reviewed_record"},
    {"target_gib": 20, "state": "unavailable", "reason": "no_reviewed_record"}
  ],
  "recommended_target_gib": null,
  "recommendation_basis": "physical_memory_only"
}
```

catalog 为空时可以显示五档，但必须 disabled；不能显示 private exact plan，也不能暗中执行 legacy streaming。

### 3.3 `StudioDraft` 迁移规则

当前 draft 仍有 `residency` 和 `zImageStreamingBudgetGiB`。建议 additive 增加：

```swift
var streamingTargetGiB: Int? = nil  // nil == Off
var streamingCatalogRevision: String? = nil
```

解码规则：

1. 新字段存在时只验证 `nil/8/10/12/16/20`；
2. 新字段缺失时保持 `nil`，不能把旧 `zImageStreamingBudgetGiB=10` 自动提升成 public 10 GiB；
3. 旧 `residency=streamed` 继续走 legacy/private 路径，直到用户主动选择 public target；
4. 用户选择 target 后，请求 builder 生成 selector，并拒绝同时生成 manual `streaming`；
5. 用户切回 Off 时删除 selector，但不改模型、prompt、输出或 legacy draft 字段；
6. Job history 保存 requested target、catalog revision、record/resolution digest，便于重放时检测 stale，而不是静默重选。

UI 只把 native options 中 `available` 的档位设为可点击。recommended 只是默认高亮建议，不自动修改 draft，也不在物理内存变化时替用户换档。

---

## 4. Layout compiler 和 memory contract

### 4.1 字段语义

| 字段 | 语义 | 是否是进程峰值 |
|---|---|---:|
| `SourceRange.bytes` | checkpoint logical read bytes | 否 |
| `FieldSpec.bytes` | materialized logical content bytes | 否 |
| `SlotLayout.capacity_bytes` | 对齐后的单 slot backing | 否 |
| `StageLayout.peak_pool_bytes` | stage pool 静态 backing 上界 | 否 |
| `calibrated_request_bytes` | 完整 process-tree 实测/审查值 | 目标证据 |

slot capacity、logical source bytes 和 `K × slot` 绝不能直接写成“请求内存上限”。activation、text encoder、VAE、MPSGraph、driver、output、allocator cache、worker 和 parent 都必须单独记账。

### 4.2 `compile_layout()` 确定性算法

```text
1. 检查 block id 严格递增且唯一
2. 检查 FieldSpec name/storage_id/alignment
3. direct materialization 必须有 source reads
4. derived materialization 必须引用同 block 更早字段且不重复读源
5. 按 layout_class 分 pool，保留 block 顺序
6. 按 canonical config 形成 group，禁止自动调整 G/K/D/Q
7. 固定 group→pool 和 group ordinal→slot 映射
8. 计算每个字段 max capacity、resident/prefix/pool/source bytes
9. canonical encode 后生成 layout digest
```

必须拒绝：重复/越界 block、`G/K/Q == 0`、非法 `D`、未声明 capability 的 `carry_first_group`、未 opt-in 的 `retain_all`、source range 越界、materializations incomplete、digest 与 record 不一致。

### 4.3 Request memory ledger

至少记录：

```text
target_request_bytes, calibrated_request_bytes
resident_weight, prefix, active_pool, retained_pool
activation_reserve, conditioning, text_encoder, vae
output_backing, allocator_cache
parent_peak, worker_peak, process_tree_peak, peak_scope
```

静态诊断上界：

```text
static_upper = resident + prefix + max_live_pools(policy)
             + conditioning + text + vae + output + activation_reserve
```

P2 只使用 sampler 得到的 `observed_tree_peak = max_t(sum(process_tree.rss[t]))`。无法测量的 driver/Metal bytes 标为 `unknown`，不能填零后宣称 hard cap。

### 4.4 物理内存推荐

```text
eligible = records matching model/source/workload/runtime
           and physical_memory in qualification
           and release.revoked == false

recommended = highest target where
              calibrated_request_bytes + reserve_margin
              <= physical_memory_bytes
```

推荐不能读取瞬时 free pages、swap in/out 作为 plan 输入。无 eligible record 时返回 nil。

---

## 5. StageExecutor、slot/pool 和 owner pump

### 5.1 Slot 状态机

```text
Empty → Filling → Ready → Claimed → Reading → Retiring → Empty
                          └── any failure → Poisoned → Quarantined
```

状态只能由 owner pump 修改。I/O worker 和 GPU callback 只发布 `tc_stream_completion_v1` POD 到 mailbox，不得修改 slot state、vector、allocator 或 Objective-C 对象。

不变量：

```text
一个 (pool, slot) 同时只有一个 content_generation
Ready => fill_completed && actual_bytes 合法
Claimed => backing 不可被 refill 覆盖
Retiring => 所有 reader fence 已完成或在 mailbox 等待
Empty => pending reader 为 0 且无 worker 持有 source buffer
```

### 5.2 Owner pump 参考实现

```cpp
void StageExecutor::run_pass(uint32_t pass, uint32_t step,
                             std::atomic<bool>& cancel,
                             std::chrono::milliseconds timeout) {
    owner();
    check_cancel(cancel);
    adapter_->encode_prefix(pass);

    for (const Group& group : groups_for(pass)) {
        check_cancel(cancel);
        Slot& slot = claim_expected_slot(group);
        wait_until_ready(slot, timeout);
        adapter_->prepare_group(group, slot.ticket);

        ReaderSet readers = adapter_->encode_group(
            group, slot.ticket, *mailbox_);
        recorder_->readers_issued(slot.ticket, readers.fences);
        slot.state = Reading;

        if (adapter_->overlap_next_fill_after_claim())
            issue_next_fill_if_safe(pass, group, cancel);

        wait_for_last_readers(slot, timeout);
        retire_slot(slot);
        consume_mailbox_or_throw();
    }

    if (pass_transition == carry_first_group)
        retain_first_next_pass_ready_only_if_declared();
    recorder_->pass_completed(pass, carry_ticket_if_any());
}
```

禁止以 timeout 到期 free pending reader、动态增加 slot、worker 继续读取已结束 lease、或把 `fills/groups_submitted` 当成 reader fence receipt。内存不够时只能拒绝当前 exact plan，不能自动切换 swap。

### 5.3 Prefetch、carry 与 multi-pool

`P/G/K/D/Q` 固定表示 prefix、group size、slot count、prefetch distance、I/O workers。安全下界是 `K1/D0/Q1`；只有 adapter 的 backing、reader queue、source buffer 都明确支持时才增加 overlap。`carry_first_group` 只允许保留一个已完成 fill 的下一 pass 首组，不能省略该 pass 的 source accounting。

`serial` 一次只保留 active layout class，class barrier 执行 drain → destroy → create/select。`retain_all` setup 时创建所有 pool，峰值是所有 pool capacity 之和；只有 adapter opt-in 且 P2 通过才能发布。切换时必须证明 pending reader 为 0 并记录 pool event。

---

## 6. Receipt、boundary 和结果验证

### 6.1 Receipt 来源

`ActualExecutionReceipt` 必须由真实 executor 事件产生：fill submitted/completed、reader fence issued/completed、pass completed、pool selected、carry、stage drained、boundary release。`streaming_runtime` 只是兼容摘要，`streaming_stages[]` 是 stage 摘要，`streaming_receipt` 是 public proof。

### 6.2 v2/v3 规则

| schema | 内容 | 限制 |
|---|---|---|
| v2 | 单 stage | boundaries 为空，layout 只能一个 stage |
| v3 | 多 stage + ordered boundaries | boundary 数为 `stage_count - 1` |

每个 boundary 必须证明：

```text
from_stage_drained == true
source_stage_backing_released == true
pending_readers_after == 0
live_slot_bytes_after == 0（或 record 明确的 retained bytes）
released_slot_bytes == live_before - live_after
completed_reader_sequence >= last_reader_sequence
next_stage_started 只能在 boundary event 之后为 true
```

### 6.3 `results.mm` 输出边界

允许输出 schema、stage id、layout digest、counts、bytes、digest、source generation 和 verifier revision。禁止输出 fd、pointer、authority object、ticket matrix 和秘密路径 token。public verifier 失败时，整个请求失败，不返回“生成成功但 receipt 缺失”的结果。

---

## 7. 四模型 adapter 施工规格

### 7.1 Z-Image Turbo（`WIP/NOT_PUBLIC`）

相关文件：`native/models/z_image/*`、`native/platform/apple/z_image_streaming_descriptor.mm`、`z_image_weight_stream.mm`。现有 probe/snapshot/lease/pager 接缝和 public host contract 不能替代 full request qualification。

必须完成：

1. probe 锁定 artifact、BF16/GGUF variant、token shape、resolution、steps、conditioning policy；
2. snapshot 覆盖 transformer、text、VAE、output 前组件的 source closure；
3. `generate_resolved()` 禁止读取旧 `zImageStreamingBudgetGiB` 反推 slot；
4. weight reader 只使用同一 `SourceLease`；
5. actual receipt 进入 common verifier。

首版建议 GPU-only、BF16、text-to-image、无 LoRA、无 ANE、固定已校准 shape；未进 catalog 的组合 unavailable。

### 7.2 Flux 9B（`WIP/NOT_PUBLIC`）

相关文件：`native/models/flux2/pipeline.cpp`、`flux_streaming.hpp`、`streaming_descriptor.hpp`、`native/platform/apple/flux_streaming_descriptor.mm`。`9a351c9` 已完成 lease-backed reader 的重要生命周期修正，但尚无 full-request public record。

必须保证 transformer、text encoder、VAE 使用同一 request source lineage；snapshot 明确 dual/single block、retained K2、Q2 refill、conditioning/VAE policy；`FluxExactStream` 不得自行按 path 打开 reader。若 text/VAE 不能纳入一个 stage，拆 ordered stages 并使用 v3 boundary。首版限制 GPU-only、BF16、固定 image workload；ANE、LoRA、不同 VAE/aspect ratio 单独认证。

### 7.3 MiniMax H3 Turbo（`NEXT/NOT_PUBLIC`）

范围冻结：仅 `minimax-h3-turbo`、original BF16、GPU-only、text-to-video、无 audio/reference/input/LoRA/quantized cache/ANE/approximation。普通 H3 和其他变体不在本轮。

现有 `native/platform/apple/h3_session.mm` private exact 使用 K2/G1/BF16 block streaming，不能直接写成 public。需要新增 metadata probe、lease-backed snapshot、`generate_resolved()` 和真实 C receipt。

建议 H3 receipt ABI：

```c
typedef struct {
    uint32_t struct_size, version;
    uint32_t stage_index, pass, group, pool, slot;
    uint64_t request_generation, content_generation;
    uint64_t expected_bytes, actual_bytes;
    uint32_t fill_completed, group_submitted, reader_count;
    h3_reader_fence_v1 readers[H3_MAX_READER_QUEUES];
} h3_stream_group_receipt_v1;

typedef struct {
    uint32_t struct_size, version;
    uint32_t stage_index, completed_passes, completed_groups;
    uint64_t fills, logical_read_bytes;
    uint64_t reader_fences_issued, reader_fences_completed;
    uint64_t source_generation;
    uint32_t drain_completed;
    const h3_stream_group_receipt_v1 *groups;
    uint32_t group_count;
    char event_digest[65];
} h3_stream_receipt_v1;
```

ABI 必须检查 `struct_size/version`；native C executor 填真实 group/fence；ObjC++ 不得从 `ssd_streamed_blocks` 合成 receipt；`h3_drain()` 成功且 pending reader 为 0 后才可标记 drained。失败时保留 context/reader/adapter 进入 quarantine。

H3 public hooks 的实现顺序：

```cpp
probe_public_streaming(input)
  → capture manifest/index lease
  → build source/workload/runtime identity

compile_public_streaming(probe, record)
  → project h3_streaming_descriptor
  → compile exact layout
  → verify record digest/component policy
  → return ValueModelStreamingSnapshot with same lease

generate_resolved(execution, event, cancel)
  → construct PublicStreamingRunContext
  → revalidate H3 component and source
  → attach denoiser stage
  → run C executor with receipt enabled
  → drain/finish/seal/revalidate source/complete
  → attach RunResult receipt and metrics
```

### 7.4 LTX 2.5 Distilled（`WIP/NOT_PUBLIC`）

当前 `apps/macos/LTXWorker.swift` 的 `component_staged` CLI 是 legacy worker，不是 public multi-stage。public graph 建议：

```text
connector/text → boundary
→ denoiser stage1 → boundary
→ latent upsampler → boundary
→ denoiser stage2 → boundary
→ VAE/decode/export
```

worker 必须自己重新读取 catalog、device、source 并 resolve；parent 只传 request JSON、model path、job id 和临时目录，不能传 fd、GPU pointer 或 authority。每个 boundary 先 drain/release 再启动下阶段；最终 `result.tmp` fsync+rename，parent 只有在 job id、request digest、receipt digest 和 output envelope 一致时提交 JobStore。EOF、SIGTERM、SIGKILL、cancel、timeout 都进入 quarantine，部分 mp4 不算成功。process-tree sampler 必须包含 parent、worker 和 finalizer。

建议保留两个明确不同的入口：

```swift
static func accepts(_ request: NativeRequest) -> Bool
// legacy component_staged only

static func acceptsPublic(_ request: NativeRequest,
                          options: StreamingOptions) -> Bool

static func generatePublic(
    model: URL,
    request: NativeRequest,
    selector: PublicStreamingSelector,
    onEvent: @escaping @Sendable (NativeEvent) -> Void
) async throws -> Data
```

worker response envelope 至少包含：

```json
{
  "schema_version": 1,
  "job_id": "<uuid>",
  "request_digest": "<sha256>",
  "record_digest": "<sha256>",
  "resolution_digest": "<sha256>",
  "receipt_digest": "<sha256>",
  "output_path": "<temporary-path>",
  "output_bytes": 0,
  "completed": true
}
```

---

## 8. Catalog、model card 和 record

每条 record 必须绑定：model/source variant、artifact content/manifest identity、operation/shape/frames/steps/token shape、device qualification、TurboCider build/runtime/adapter/reader/kernel revisions、canonical config、layout digest、component policy、process-tree calibration、performance evidence 和 reviewed commit/revoke state。

snapshot/stat/inode/time 只能用于失效检测；没有内容 hash 或可信 artifact manifest 不能授予 public qualification。

拟议 candidate record（不能直接发布）：

```json
{
  "id": "flux9b-gpu-bf16-512x512-s4-12g-k2-q2-v1",
  "catalog_revision": "tc-streaming-catalog-candidate-v1",
  "workload": {
    "model": "flux2-klein-9b",
    "operation": "image.generate",
    "execution": "gpu",
    "width": 512,
    "height": 512,
    "frames": 1,
    "steps": 4,
    "audio": false
  },
  "plan": {
    "layout_digest": "<sha256>",
    "component_policy_revision": "flux-public-v1",
    "pass_transition": "carry_first_group",
    "multi_pool_policy": "retain_all"
  },
  "calibration": {
    "complete": false,
    "scope": "process_tree",
    "calibrated_request_bytes": 0,
    "evidence_digest": ""
  },
  "release": {"channel": "candidate", "revoked": false}
}
```

`complete=false`、`channel=candidate` 或 evidence digest 为空时，builder 必须拒绝写入 production catalog。独立 verifier 再次检查 record digest、layout digest、source/runtime/workload identity、P0–P3、quality、reviewed commit 和 release state。

### 8.1 Target resolution 的确定性

相同 catalog snapshot、probe、device 和 selector 必须得到同一 record。排序键建议冻结为：

```text
exact target
→ exact source/workload/runtime/device match
→ release channel priority
→ lowest performance.rank
→ highest preset revision
→ lexicographically smallest record id
```

任何并列仍未消除时返回 ambiguity error，不能依赖 vector 遍历顺序。requested target 不得扩成 target range。

### 8.2 Revocation

revoked record 不能用于新 resolve。正在运行的 request 使用已绑定 authority，但 GPU work 前 revalidate 若发现 catalog 已撤回同 record，应 fail-closed。result 必须记录 catalog revision 和 record digest，便于回溯。回滚 production catalog 不删除 evidence，只改变 release/revocation 状态。

### 8.3 五档 target 的候选探索，不把 target 倒推成运行计划

五档只是搜索桶。离线探索器可以为每个模型生成候选 family，但只有经过真实 full-request 校准的 record 才能进入 catalog。建议的搜索边界如下：

| 模型 | 当前 adapter revision 允许探索的 exact family | 五档处理原则 |
|---|---|---|
| Z-Image Turbo | `G1/K2/D0/Q1/reload`，只 sweep 合法 prefix | 某 target 找不到满足 P2 的 prefix 就 unavailable；不能临时改成 K1/K3 |
| Flux 9B | `P0/G1/K2/D0..1/Q1..2/retain_all/reload` | 只在当前 family 内比较 D/Q；低 target 不 fit 就 unavailable |
| H3 Turbo | `G1/K2/D1/Q1/carry_first_group`，sweep 合法 prefix | 保持 current exact ABI；K1 或其他 transition 必须新 adapter revision |
| LTX 2.5 | `G1/K1..3` 与已声明 pass/upsample boundary，sweep prefix/K | public multi-stage graph 固定后逐 target 校准，不在运行时改变 stage topology |

上表是**搜索顺序和候选族**，不是支持矩阵，也不是已测配置。每个单元要执行：

```text
enumerate explicit (P,G,K,D,Q,pool,transition)
→ compile deterministic layout
→ reject static upper > target - margin
→ run synthetic executor
→ run real full request
→ sample complete process tree
→ compare quality and P0/P1/P2/P3
→ write candidate evidence only
```

候选生成器不得修改用户 request，也不得在运行时尝试多个 plan。对某个 target 找不到可行 candidate 时，record 缺失就是正确结果；App 显示 unavailable，而不是把 12 GiB 请求降级到 10 GiB 或升级到 16 GiB。

如果研究发现 K1、K3、serial/retain-all 切换或新的 pass transition 有明显收益，应先发布新的
`adapter_revision`/descriptor 测试，再进入另一轮 candidate campaign；不能在同一 record 下扩大当前
descriptor 的 capability。

### 8.4 候选 family 的排序和淘汰

候选排序先按安全性，再按性能：

```text
1. source/materialization closure PASS
2. static upper <= target - margin
3. full-request tree peak P95 <= target - margin
4. quality PASS
5. P1 PASS
6. logical read bytes / wall time / refill wait 最小
```

任何候选在 P2 或 quality 失败时立即淘汰；不得用较好的 median 抵销 P95 超标。候选 record 保留淘汰原因和 evidence digest，方便以后重新校准。

---

## 9. 工具链设计

建议统一入口 `tools/streaming/streamingctl`（语言可按仓库现状决定）：

```text
inspect       --model --checkpoint --request
compile       --descriptor --config --out-plan
simulate      --plan --fault cancel|eof|reader|source-change
campaign      --model --target 8,10,12,16,20
              --arms resident,streaming,bounded,swap
sample-tree   --pid --interval-ms 20 --out samples.jsonl
verify        --evidence bundle.json
build-catalog --candidates candidates/ --out catalog.json
```

每个命令支持 `--json`、`--schema-version`、deterministic seed，并输出 git commit、binary hash、OS build、device identity、artifact identity 和 command line。

### 9.1 inspect

只读取 metadata，输出 source identity、stages/blocks/fields、dtype/shape/ranges、workload、component policy 和 closure failures。不得创建 GPU buffer 或修改 catalog。

### 9.2 compile

只接受显式 `StreamingConfig`，不从 target 倒推。输出 layout digest、stages、slot capacities、logical source bytes、`materializations_complete` 和 `execution_qualification=none|candidate|public`。compile 成功不等于 public。

### 9.3 simulate

fake adapter 覆盖 fill/reader delay、queue reorder、各状态 cancel、EOF/source replacement、mailbox/drain failure、serial/retain-all、v2/v3 mutation。simulate 只能证明状态机/verifier，不能产生 P2/P3 evidence。

### 9.4 campaign 和 sampler

sampler 每 20 ms 记录 parent/worker descendants：

```text
monotonic_ns, pid, ppid, role
rss_bytes, compressed_bytes
phase, stage_index, pass, group, pool, slot
fills, pending_readers, output_bytes
swap_in_bytes, swap_out_bytes
```

采样缺口超过 record 的 `maximum_sample_gap_ns` 时，`calibration.complete=false`。未知 driver/Metal bytes 标为 unknown，不能补零。

### 9.5 verifier 和 builder

verifier 在独立进程重算 source/workload/runtime/layout/receipt/evidence digest；builder 只接受 verifier PASS 的 immutable bundle。任何 record 失败都拒绝整次 production catalog build，避免 App 暴露部分错误档位。

---

## 10. 四臂实验、统计和质量

| arm | 含义 | 用途 |
|---|---|---|
| resident | 原始常驻路径 | 默认速度/质量基线 |
| streaming | exact slot/pool plan | public 候选 |
| bounded | explicit memory guard | 预算合同诊断 |
| natural-swap | 不改模型、允许 OS 换页 | P3 对照 |

streaming arm 失败不能自动切到 swap；swap arm 不能调用 streaming executor。

每个 `(model,target,workload,device)` 使用同一输入、seed、build、thermal protocol，ABBA 顺序：

```text
A resident warmup
→ B candidate × N
→ B candidate × N
→ A resident × N
```

至少记录 wall、first output、denoise/decode、tree peak、parent/worker peak、swap bytes、logical read bytes、quality digest、cancel status、receipt digest 和 output size。每臂至少 20 个有效请求，报告 median、P95、bootstrap 95% CI、失败率和样本数；采样缺口、温度异常、cancel 单独报告，不能静默删除。

质量至少检查 output decode、frame/dimension/fps、PNG/latent deterministic digest 或 perceptual metric，以及音频路径的长度/存在性。只比较“文件存在”不算 PASS。

### 10.1 “是否比 swap 快”的判定

只能在同机同 workload 下比较：

```text
speedup = natural_swap.wall / streaming.wall
peak_reduction = natural_swap.tree_peak - streaming.tree_peak
swap_reduction = natural_swap.swap_in_out - streaming.swap_in_out
```

若 confidence interval 跨 1，结论写 `INCONCLUSIVE`，不能写“更快”。即使 wall 更快，P2 不通过也不能发布该 target；即使 P2 通过，质量或 failure rate 回退也不能发布。

### 10.2 Default/Off 性能双重守卫

静态审计检查：

```text
selector inactive 时不调用 catalog_provider.snapshot()
selector inactive 时不调用 probe_public_streaming()
selector inactive 时不构造 PublicStreamingRunContext
selector inactive 时不创建 SourceLease/StageExecutor/sampler
default ModelSession::generate() 调用图与 dev 基线一致
```

动态审计使用同一 binary 中的 feature counter 或 test hook（release 构建不暴露）：

```text
public_preflight_calls == 0
public_probe_calls == 0
public_context_creates == 0
public_sampler_starts == 0
```

随后对 merge 前 dev binary 与 candidate binary 做默认 workload ABBA。只有静态零调用和 P0 同时满足，才能写“内存充足机器不受影响”；单纯 source inspection 或单次 smoke 都不够。

---

## 11. 测试矩阵

### 11.1 Common PR

```bash
git diff --check
env TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
make test-streaming-host
make test-streaming-contract
make test-streaming-audit
```

修改 receipt/C ABI/slot ownership/callback 时追加：

```bash
make test-streaming-sanitizer
make test-streaming-tsan
```

覆盖 empty catalog、selector conflict、source generation/path replacement、duplicate/missing stage、boundary digest/order、pending reader/release mismatch、legacy summary mismatch、drain failure/quarantine 和 retry drain。

### 11.2 Adapter 四层

| 层 | 能证明什么 | 不能证明什么 |
|---|---|---|
| metadata | identity/source closure/compiler 输入 | GPU 质量/峰值 |
| synthetic | executor/receipt/fault 状态机 | 真实模型性能 |
| Metal smoke | fence/buffer/reader 生命周期 | full request P2/P3 |
| full request | quality、峰值、性能、record 资格 | 其他未测 shape/variant |

前三层不能替代 full request；full request 不能省略 common verifier。

### 11.3 App/JobStore

覆盖 legacy draft decode、Off 无 selector、unavailable/revoked disabled、stale digest、error 不改 Off、cancel 无半成品、worker envelope/job/request digest mismatch、jobs.json atomic recovery。

### 11.4 H3/LTX 专项

H3：ABI size/version、真实 group/fence clone、native drain pending、source replacement、各位置 cancel、H3 Turbo full request。

LTX：worker-local re-resolve、parent 不传 authority/fd/pointer、boundary release bytes、upsampler failure quarantine、EOF/nonzero/SIGKILL/timeout、atomic rename、parent+worker sampling。

### 11.5 Fault injection 期望

| 故障 | 期望结果 |
|---|---|
| fill EOF | stage fail，未提交 reader，context drain/quarantine |
| reader callback error | slot Poisoned，不复用 backing |
| source path replaced | GPU work 前或 drain 后 revalidate fail |
| cancel during fill | worker cooperatively stop，unsafe backing 不 free |
| cancel during reader | 等待/重试 drain，超时 quarantine |
| boundary pending reader | verifier reject，不启动 next stage |
| LTX worker EOF | parent 拒绝 result，删除临时输出 |
| receipt digest mutation | public result reject |

---

## 12. PR 施工顺序和停止条件

### R3 · Common multi-stage runtime（`DONE`）

`aea2cf2`、`30fefe4` 已完成 receipt v3、boundary verifier、`PublicStreamingRunContext`、legacy summary consistency、results serialization。它只构成 common 基线，不授予任何模型 public 资格。

### R4 · H3 Turbo（`NEXT`）

拆为：

1. `R4a` metadata/lease/snapshot；
2. `R4b` C receipt ABI 和真实 group/fence export；
3. `R4c` `H3Session::generate_resolved()`；
4. `R4d` host/synthetic/Metal/full request/P0–P2；
5. `R4e` candidate record、P3 和 review。

任一 source closure 或 receipt 缺口都停止，不能从 counters/summary 补齐。

### R5 · LTX worker（`NEXT`）

拆为 worker-local resolve、stage graph/boundary、cancel/EOF/quarantine/atomic output、process-tree sampler/full quality、P0–P3/record review。`R5c` 前不能把 `component_staged` label 改成 public target。

### R6 · App/catalog（`NEXT`）

统一 target enum、native options、unavailable UI、JobStore transaction、builder/verifier、release/revoke/rollback。production catalog 非空前，UI 只显示 disabled 状态。

### R7 · Z-Image/Flux full qualification（`BLOCKED`）

在已有 lease adapter 上补真实 full request、P0/P1/P2/P3、quality 和 model card。host/synthetic PASS 不能直接升级为 public record。

### 合并 `dev` 前

记录 worktree 状态，非破坏性 merge，逐文件审阅冲突；重跑 native build、host/contract/audit、adapter、App 和 default performance audit。若 `dev` 改变 allocator、MLX、Metal、Swift schema 或 model route，必须重新确认 Off 零开销和 selector digest。

---

## 13. 代码修改矩阵

| 文件/目录 | 修改目标 | 主要测试 |
|---|---|---|
| `native/runtime/streaming/*` | resolver/context/receipt/verifier/catalog | host、sanitizer、TSan |
| `native/core/stream_slot_c.h` | 仅 additive ABI，保持旧布局 | ABI/static assertions |
| `native/platform/apple/h3_session.mm` | public hooks/context/result | H3 host/Metal/full |
| `native/models/h3_runtime/h3_*.{c,h}` | lease reader、C receipt、drain | C fixture/fault |
| `native/platform/apple/ltx_session.mm` | multi-stage receipt/boundary | LTX model/full |
| `apps/macos/LTXWorker.swift` | worker-local resolve/envelope/atomic output | worker protocol |
| `apps/macos/StudioState.swift` | target enum/legacy migration | Codable tests |
| `apps/macos/JobStore.swift` | options/transaction/stale/cancel | App integration |
| `native/platform/apple/results.mm` | safe summaries only | contract snapshots |
| `tools/streaming/*` | inspect/campaign/sampler/verifier/builder | golden/e2e |
| `docs/design/streaming/13*` | 只记录真实完成状态 | doc checks |

所有 ABI 改动必须 additive；不能为了 public streaming 改变 default/private executor 的结构布局或调用顺序。模型 adapter PR 不应同时修改 App 产品文案和 production catalog，便于回滚与性能归因。

---

## 14. 错误码和诊断

稳定错误码建议：

```text
catalog_has_no_public_records
streaming_target_unavailable
streaming_source_lease_mismatch
streaming_authority_mismatch
streaming_actual_plan_mismatch
streaming_receipt_incomplete
streaming_drain_failed
streaming_worker_protocol_error
streaming_quality_mismatch
```

这些错误不能转换成自动 resident/swap fallback。诊断可包含 target、record id、layout digest、source generation、stage/boundary index、pending readers、peak scope，但不输出 fd、pointer 或秘密路径 token。

---

## 15. Release reviewer checklist

### 15.1 代码

```text
[ ] Off/default 不创建 selector/catalog/probe/context/sampler
[ ] public request 每次 resolve/revalidate
[ ] probe/snapshot/executor 使用同一 SourceLease
[ ] layout digest 与 record 一致
[ ] slot state 仅 owner 修改，callback 只发 POD
[ ] cancel/drain/quarantine 无 unsafe free
[ ] v2/v3 receipt 来自真实事件
[ ] multi-stage boundary 在 backing release 后记录
[ ] LTX worker 不接 authority/fd/pointer
[ ] H3 不从 counters 合成 receipt
```

### 15.2 证据

```text
[ ] source/runtime/workload/device identity 完整
[ ] real checkpoint full request
[ ] output quality PASS
[ ] process-tree sampler 无大采样缺口
[ ] resident/streaming/bounded/natural-swap 四臂
[ ] P0/P1/P2/P3 PASS
[ ] evidence/record digest 一致
[ ] builder/verifier 独立通过
[ ] rollback/revoke 已演练
```

### 15.3 产品

```text
[ ] App 只显示 Off/五档，不暴露 layout
[ ] unavailable/revoked 不可点击
[ ] physical memory 只用于推荐
[ ] native 不自动 fallback 或换档
[ ] legacy draft 可解码且不改变旧语义
[ ] 错误/取消不提交半成品
```

---

## 16. 最终完成定义

只有以下条件同时满足，框架才能从 private/candidate 变成 public：

1. common runtime、receipt v3、boundary、source lease、quarantine 有 host/sanitizer/Metal 证据；
2. LTX、H3 Turbo、Z-Image Turbo、Flux 9B 各自有独立 model card、layout、runtime identity 和 reviewed record；
3. 每个开放 target 有真实 full-request process-tree peak、quality、P0/P1/P2/P3 evidence；
4. ANE/hybrid 未认证组合保持 fail-closed，不能复用 GPU-only record；
5. App/JobStore 只暴露 target，native 每请求 exact resolve；
6. production catalog 非空、可独立 verify、可 revoke/rollback；
7. Off/default 路径无可测新增热路径开销；
8. 合并 `dev` 后完整回归仍通过。

在这些条件满足前，正确状态是：**统一 streaming 框架和 common runtime 已在实施，部分模型有 private/candidate adapter，但 public streaming 尚未发布。**
