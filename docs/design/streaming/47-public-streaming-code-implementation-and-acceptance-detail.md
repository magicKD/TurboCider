# 47 · Public Streaming 代码实施与验收细化规格

修订日期：2026-09-18。状态：**待实施/实施中规范；不是 public 支持声明**。

本文是 [46 实施总手册](46-public-streaming-implementation-handbook.md) 的代码级附录，专门回答以下问题：

1. 当前 common runtime 已经具备哪些接缝，模型代码应从哪里接入；
2. request、source lease、snapshot、slot backing、reader fence、receipt 和结果分别由谁拥有；
3. Z-Image Turbo、Flux.2 Klein 9B、MiniMax H3 Turbo、LTX 2.5 应如何逐文件实现；
4. 8/10/12/16/20 GiB 档位如何从候选布局变成可发布 record；
5. 如何证明默认路径不回归、generic executor 没有显著开销、低内存路径不依赖非计划 swap；
6. 每批代码在什么条件下可以提交、合并、发布或必须回滚。

当前实现事实只以 [13 实施进度](13-implementation-progress.md) 为准。本文中的新增类型、文件、测试和命令如果尚不存在，均属于设计合同，不能据此宣称已经支持。

---

## 1. 冻结决策

### 1.1 产品层只暴露内存档位

App 高级设置只暴露：

```text
Off / 8 GiB / 10 GiB / 12 GiB / 16 GiB / 20 GiB
```

用户不直接编辑 `P/G/K/D/Q`、pool policy 或 carry policy。public selector 表达目标，catalog record 才携带经过校准的 exact layout。

### 1.2 布局是确定性执行合同

一次 resolve 后，以下字段在请求执行期间不可改变：

```text
resident_prefix_blocks (P)
block_group_size       (G)
slot_count             (K)
prefetch_distance      (D)
io_workers             (Q)
multi_pool_policy
pass_transition
component_policy_revision
```

运行时不能因为“当前内存看起来紧张”而偷偷减少 slot、改 prefix、降精度、缩 shape 或切换 backend。布局不 fit 时必须拒绝或让用户重新选择另一个已发布档位。

### 1.3 默认路径必须物理隔离

当 selector 为 Off 或不存在时：

- 不创建 `PublicStreamingCoordinator`；
- 不读取 production catalog；
- 不捕获 `SourceLease`；
- 不编译 public layout；
- 不创建 slot pool、I/O worker 或 receipt recorder；
- 不新增默认热路径分支内的 heap allocation、thread creation 或 source metadata scan；
- 继续使用当前 resident/private/compiled/GPU+ANE 路径。

“代码逻辑没有改变”不足以通过 P0；必须用 audit counter 和 ABBA 性能样本证明上述隔离成立。

### 1.4 public success 必须是闭环事实

一次 public success 必须同时拥有：

```text
immutable catalog snapshot
exact source/workload/runtime/device identity
shared request-scoped SourceLease
compiled layout digest
unforgeable in-process authority
actual receipt v2
reader drain proof
post-drain source revalidation
verified RunResult summary
完整请求 memory/performance evidence（用于发布，不要求每次用户运行都落盘）
```

只要其中一项缺失，结果就不能标为 `actual_plan_verified=true`。

### 1.5 GPU-only v1 边界

首个 public 版本只认证 GPU 路径。任何 `ane_manifest`、`encoder_ane_manifest`、模型私有 ANE/VDN/VSA route 都必须返回稳定的 unavailable/unsupported reason。未来 hybrid streaming 使用独立 adapter revision、component policy 和 catalog card，不能继承 GPU-only record。

---

## 2. 当前代码基线与本轮目标差距

### 2.1 已存在的 common runtime 接缝

当前代码已经有以下核心对象：

| 对象 | 位置 | 当前职责 |
|---|---|---|
| `StreamingSelector` | `native/core/streaming_contracts.hpp` | schema-v2 用户意图：target 或 exact preset |
| `StreamingConfig` | 同上 | schema-v1 exact manual layout |
| `StreamingPresetRecord` | `native/runtime/streaming/preset_catalog.hpp` | source/workload/runtime/device/layout/calibration/performance/release 合同 |
| `SourceLease` | `native/runtime/streaming/source_lease.*` | 单一 fd lineage、generation/digest、执行前/后 revalidation |
| `ModelStreamingProbe` | `native/runtime/streaming/resolved_request.hpp` | metadata-only identity 和 lease |
| `ModelStreamingSnapshot` | 同上 | descriptor、compiled layout、同一 lease |
| `StreamingAuthority` | 同上 | record/snapshot/device/source generation 的内部授权 |
| `ResolvedRequestExecution` | 同上 | 一次 resolved generate 的 immutable 输入 |
| `PublicStreamingCoordinator` | `native/runtime/streaming/public_runtime.*` | preflight、resolve、GPU lock 后 revalidate |
| `StageExecutor` | `native/runtime/streaming/context.*` | owner pump、multi-slot、multi-pool、carry、drain |
| `ActualReceiptRecorder` | `native/runtime/streaming/actual_receipt.*` | owner-thread actual execution receipt v2 |
| public result verifier | `native/runtime/streaming/public_result.*` | receipt、layout、source、结果摘要核对 |
| C ABI/Swift types | `native/api/c_api.mm`、`bindings/swift/TurboCiderNative.swift` | options、resolve、generate JSON 边界 |

这些对象是唯一控制面和执行面。模型 adapter 不得建立第二套 selector、catalog、scheduler、receipt 或 source identity。

### 2.2 仍需完成的最小闭环

| 闭环 | Z-Image | Flux 9B | H3 Turbo | LTX 2.5 |
|---|---|---|---|---|
| public probe/snapshot | 实施中 | 待实现 | 待实现 | worker 内待实现 |
| shared source lease reader | 实施中 | 待实现 | 待实现 | 待实现 |
| `generate_resolved()` | 实施中 | 待实现 | 待实现 | worker 内待实现 |
| 真实 receipt v2 | 实施中 | 待实现 | 待实现 | 待实现 |
| full-request P0/P1 | 待完成 | 待完成 | private 证据不可直接代替 public | 待完成 |
| target P2 | 无 production evidence | 无 | 无 | 无 |
| App/JobStore | 共用能力待实现 | 共用 | 共用 | 另需 worker transaction |

### 2.3 不应继续扩大的临时模式

模型 session 上用成员变量临时保存 `public_stream_lease_`、target 或当前 resolved layout，可以作为短期接线方式，但不是最终框架形态。其风险包括：

- exception/cancel 路径漏清理后污染下一请求；
- 同一 session 的 reentrant 调用难以证明安全；
- private generate 与 public generate 共享隐式状态；
- 单元测试只能观察结果，难以检查状态生命周期；
- LTX worker、多 stage 或未来并发 session 难以复用。

目标架构应使用 request-scoped execution context，见第 5 节。若某个模型在迁移期间必须使用 session binding，必须用 RAII 对象恢复全部旧值，并在进入/退出时断言无重入和无残留。

---

## 3. 模块依赖与目录边界

### 3.1 允许的依赖方向

```text
App / Swift
    -> C ABI
        -> public coordinator / catalog / verifier
            -> ModelSession public hooks
                -> model descriptor + exact adapter
                    -> StageExecutor / SourceLease / receipt
                        -> Metal / MLX reader and kernels
```

反向依赖禁止：

- common runtime 不包含 Z-Image/Flux/H3/LTX 头文件；
- model adapter 不解析 App UI 状态；
- executor 不读取 catalog 或 selector；
- I/O worker 不调用 resolver、allocator policy 或 App callback；
- catalog builder 不链接模型 GPU runtime；
- Swift 不反序列化完整内部 layout/ticket/receipt matrix。

### 3.2 common runtime 推荐新增文件

为避免四模型复制 glue，建议新增：

```text
native/runtime/streaming/model_adapter_helpers.hpp
native/runtime/streaming/model_adapter_helpers.cpp
native/runtime/streaming/execution_view.hpp
native/runtime/streaming/execution_view.cpp
native/runtime/streaming/error_codes.hpp
```

职责限制：

| 文件 | 可以做 | 不可以做 |
|---|---|---|
| `model_adapter_helpers.*` | 构造/比较 canonical identity；检查 exact record | 打开模型、分配 GPU、改布局 |
| `execution_view.*` | 从 `ResolvedRequestExecution` 提供只读 typed view 和统一 precondition | 持有 mutable slot/worker 或 fallback |
| `error_codes.hpp` | 稳定 code 常量与分类 | 拼接含路径/fd 的用户消息 |

### 3.3 模型侧目录边界

每个模型保持三层：

```text
model session / pipeline
  - request route、public hooks、完整组件生命周期、RunResult

model streaming descriptor
  - source metadata -> blocks/fields/passes/classes
  - 纯 metadata，不创建 GPU payload

platform reader / exact adapter
  - lease fd -> backing fill
  - binding、kernel encode、reader fence、drain
```

不要把 safetensors/Metal/MLX reader 细节塞进 coordinator，也不要把 catalog resolution 放进 descriptor。

---

## 4. 一次 public 请求的精确事务

### 4.1 端到端时序

```text
App draft
  -> query options
  -> user selects target
  -> encode schema-v2 request
  -> tc_engine_resolve_streaming_json
       parse
       request-only validation
       immutable catalog snapshot
       normalize/make_plan
       model probe + SourceLease capture
       exact record selection
       descriptor/layout snapshot
       authority creation
       return safe resolution summary
  -> JobStore stores resolved token summary
  -> tc_engine_generate_json
       repeat parse/preflight/resolve for execution authority
       acquire engine lock
       acquire global GPU lock
       revalidate catalog/source/device/runtime/layout
       ModelSession::generate_resolved
       exact adapter executes StageExecutor
       final drain
       post-drain lease revalidation
       common receipt/result verification
       serialize verified result
  -> write output atomically
  -> JobStore verified -> committed
```

App 的 resolve summary 不是可跨进程反序列化的 authority。真正 generate 必须由 native 重新创建 request-scoped authority，防止 catalog/source/runtime 在用户点击与执行之间变化。

### 4.2 API 层顺序不变量

必须保持：

1. 空 production catalog 在模型 metadata probe 和 GPU lock 前 fail-closed；
2. public request normalize 只能发生在一次性 preflight ticket 之后；
3. global GPU lock 后再次检查 catalog、device、lease 和 authority；
4. revalidate 通过前不创建 GPU buffer、不启动 refill worker；
5. `generate_resolved()` 失败时绝不调用普通 `generate()`；
6. common verifier 通过前不返回 public success；
7. result 序列化失败也必须先完成 drain/RAII cleanup。

### 4.3 准备与生成的边界

public streaming v1 不支持“resolve 一次后长期 prepare 一个可变 pool”。`prepare()` 对 active public selector 返回 `streaming_prepare_unsupported`，直到以下条件都实现：

- prepared authority 有明确有效期；
- source lease 跨 prepare/generate 的更新语义已定义；
- catalog revoke 能使 prepared session stale；
- backing retention 纳入完整内存校准；
- App 能原子废弃旧 prepared state。

---

## 5. 对象所有权与 request-scoped execution context

### 5.1 所有权表

| 对象 | 创建者 | 最长生命周期 | 可变性 | 结束条件 |
|---|---|---|---|---|
| catalog snapshot | provider/preflight | resolve | immutable | resolved object 释放 |
| `SourceLease` | model probe | 整个 resolved generate | immutable identity，held fd | post-drain revalidate 后释放 |
| probe | model session | resolve + generate | immutable | resolved object 释放 |
| snapshot/layout | model session | resolve + generate | immutable | resolved object 释放 |
| authority | resolver | resolve + generate | immutable/nonserializable | resolved object 释放 |
| model run context | `generate_resolved()` | 一次请求 | owner-thread mutable summary | result/exception cleanup |
| slot backing | exact adapter | stage begin -> verified drain | GPU/reader owned | safe destroy 或 quarantine |
| fill job | executor | submit -> completion consumed | worker-local/POD output | owner consumes completion |
| receipt recorder | executor | begin -> finish | owner-thread only | sealed immutable receipt |
| App resolution | Swift | draft/resolved job | serialized safe summary | stale/commit/cancel |

### 5.2 推荐的只读 execution view

建议新增只读 wrapper，统一模型 precondition：

```cpp
class PublicStreamingExecutionView final {
public:
    explicit PublicStreamingExecutionView(
        std::shared_ptr<const ResolvedRequestExecution>);

    const Request &request() const noexcept;
    const StreamingPresetRecord &record() const noexcept;
    const ModelStreamingSnapshot &snapshot() const noexcept;
    const SourceLease &lease() const;
    const Layout &layout() const noexcept;
    uint64_t target_bytes() const;
    uint64_t source_generation() const;
    std::string_view component_policy_revision() const noexcept;

    void require_model(std::string_view) const;
    void require_implementation(std::string_view) const;
};
```

构造函数一次性检查：

```text
execution/probe/snapshot/authority 非空
probe lease == snapshot lease
record layout digest == snapshot layout digest
record component policy == snapshot component policy
exact selector target 存在且属于公开档位
request.streaming 是 record canonical config
```

它只保存 `shared_ptr<const ResolvedRequestExecution>`，不能缓存路径、复制 fd 或修改 layout。

### 5.3 模型 run context

每个模型可以有轻量 request-scoped context：

```cpp
struct ZImagePublicRunContext final {
    PublicStreamingExecutionView execution;
    uint64_t request_generation = 0;
    std::shared_ptr<ZImageExactStream> stream;
    bool drained = false;
};
```

目标调用形式：

```cpp
RunResult ZImage::generate_resolved(... execution ...) {
    ZImagePublicRunContext run{PublicStreamingExecutionView(execution)};
    return generate_impl(run.execution.request(), event, cancelled, &run);
}
```

`generate_impl(..., nullptr)` 是原默认路径；默认路径不构造 view/context。不要让普通 `generate()` 先检查一组 public session 成员再决定行为。

### 5.4 迁移期 RAII binding

若一次性重构 `generate_impl` 风险过大，可暂时使用：

```cpp
class ScopedZImagePublicBinding final {
public:
    ScopedZImagePublicBinding(ZImage &, PublicStreamingExecutionView);
    ~ScopedZImagePublicBinding();
    ScopedZImagePublicBinding(const ScopedZImagePublicBinding &) = delete;
};
```

要求：

- 构造时断言 session 没有 active binding；
- 保存并恢复所有被修改字段；
- 析构 `noexcept`，不能在清理失败时吞掉 unsafe backing；
- slot/worker 不由 binding 拥有，仍由 exact stream/drain 管理；
- 增加异常、取消、receipt mismatch 后第二次普通请求测试；
- 在 C3 后续收敛 PR 中删除隐式 binding，不能复制到 Flux/H3/LTX。

---

## 6. Common runtime 代码设计

### 6.1 identity helper

建议接口：

```cpp
PresetSourceIdentity make_public_source_identity(
    const SourceLease &, std::string_view variant,
    std::string_view weight_format,
    std::string_view artifact_manifest_digest);

PresetRuntimeIdentity make_public_runtime_identity(
    std::string_view build_id,
    std::string_view runtime_revision,
    std::string_view adapter_revision,
    std::string_view reader_revision,
    std::string_view kernel_revision,
    std::string_view allocator_policy_revision);

void require_probe_record_identity(
    const ModelStreamingProbe &,
    const StreamingPresetRecord &);

void require_snapshot_record_identity(
    const ModelStreamingSnapshot &,
    const StreamingPresetRecord &);
```

helper 只能 canonicalize/compare。模型仍负责构造准确 workload，例如 token rows、VAE policy、audio、frames、conditioning revision。

### 6.2 source closure

`SourceLease` 的文件集合必须覆盖 public card 在一次完整请求中可能读取的所有模型 artifact，而不只是 streamed denoiser：

```text
transformer / DiT
text encoder(s)
VAE
upsampler
audio encoder/vocoder
connector/projection
tokenizer/config/index 中会影响 tensor/shape/route 的文件
```

对于目录型或 sharded checkpoint：

1. 先读取 manifest/index 并得到稳定的逻辑文件列表；
2. 将 manifest/index 自身也纳入 lease；
3. 每个 shard 使用稳定 logical id；
4. source identity 包含 ordered artifact manifest digest；
5. reader 只能通过 lease duplicate fd 访问 shard；
6. 禁止 snapshot 只租 index，而执行时按路径打开 shard。

首个 Z-Image public card可严格限制 single-file transformer/text/VAE，以缩小 C3 风险；Flux/H3/LTX 不能因此省略自身真实 closure。

### 6.3 receipt 组装

每个 stage 的 receipt 必须来自其 `StageExecutor`。完整模型 receipt 只允许做确定性组合：

```cpp
auto receipt = make_actual_execution_receipt(
    implementation_revision,
    view.layout().digest,
    view.component_policy_revision(),
    collect_finished_stage_receipts());

verify_actual_execution_receipt(
    view.layout(), implementation_revision,
    view.component_policy_revision(),
    view.source_generation(), receipt);
```

禁止从 `ExecutionCounters`、日志或计划预期值拼出 actual group/fence/carry。若某个 resident component 没有 StageExecutor，它仍必须出现在 component policy 和 memory evidence 中，但不伪造 streamed stage receipt。

### 6.4 finalization 顺序

建议统一 helper：

```cpp
void finalize_public_streaming_run(
    const PublicStreamingExecutionView &,
    std::span<const std::shared_ptr<const ActualStageReceipt>>,
    RunResult &);
```

内部顺序必须是：

```text
all StageExecutor::finish()
-> adapter drain completed
-> receipt sealed
-> SourceLease::revalidate_after_drain()
-> execution receipt verify
-> attach receipt/runtime summary
```

如 finish/drain 不可证明安全：设置 engine/session quarantine，并保留 unsafe backing/adapter/mailbox 到 session 销毁；不能为了释放内存强行析构。

### 6.5 稳定错误分类

建议至少冻结以下 code 类别：

| 类别 | 示例 | retryable |
|---|---|---|
| selector/request | `streaming_target_unsupported`、`streaming_route_unsupported` | 否，需改设置 |
| availability | `catalog_has_no_public_records`、`streaming_preset_unavailable` | 可能，取决于 catalog |
| stale | `streaming_resolution_stale`、`streaming_source_changed` | 是，需重新 resolve |
| authority | `streaming_authority_mismatch`、`streaming_layout_digest_mismatch` | 否，内部错误/版本不匹配 |
| execution | `streaming_fill_short_read`、`streaming_scheduler_no_progress` | 视错误而定 |
| safety | `streaming_actual_receipt_mismatch`、`streaming_drain_unverified` | 否，engine quarantine |
| cancellation | `cancelled` | 是，新请求/新 session |

用户消息不能包含 fd、inode、绝对路径、指针或内部 ticket。完整诊断只进入本地 evidence，且需要脱敏。

---

## 7. StageExecutor 与 multi-slot 详细调度合同

### 7.1 owner 单线程原则

只有调用 `begin/run_pass/finish` 的 owner thread 可以修改：

- slot state；
- active pool；
- group/pass cursor；
- carry ticket；
- receipt vector/digest；
- sticky failure/quarantine 标志。

I/O worker 和 GPU completion callback 只向 bounded mailbox 写固定 POD completion，不持有 `StageExecutor*` 的可变引用。

### 7.2 slot 状态与合法事件

| 当前状态 | 允许事件 | 下一状态 | 拒绝条件 |
|---|---|---|---|
| `Vacant` | owner dispatch fill | `Loading` | slot 仍有 reader/carry |
| `Loading` | fill completion | `Ready` | generation/ticket/bytes 不匹配 |
| `Ready` | owner claim | `InUse` | group/pass/pool/slot 不匹配 |
| `InUse` | reader fences issued | `AwaitingFence` | 无 reader 且未声明 sync complete |
| `AwaitingFence` | 所有最后 reader 完成 | `Vacant` | fence 重复/未知/缺失 |
| 任意非终态 | cancel/failure | drain path | 新 dispatch 必须停止 |

carry 是 `Ready` content 跨 pass 保留，不新增一个隐藏状态；receipt 中必须显式记录 from/to pass、group、pool、slot、content generation。

### 7.3 refill 窗口

对当前待执行 group ordinal `i`：

```text
eligible_fill_ordinals <= i + D
inflight_fills <= Q
live_content <= K slots per pool
```

`D < K` 且 `Q <= K` 已由 config validation 保证，但 executor 仍要对 compiled layout/ticket 防御性检查。`overlap_next_fill_after_claim()` 只能改变 dispatch 时点，不能放宽 ownership。

### 7.4 no-progress 判定

owner 每轮至少发生一个 progress event：

```text
completion consumed
fill dispatched
group claimed/submitted
reader retired
pool barrier advanced
pass completed/carry sealed
```

若没有 progress，但存在合法 inflight completion，则在带 deadline 的 mailbox/fence wait 上阻塞；若没有 inflight 且无法 dispatch/claim，则立即报 `streaming_scheduler_no_progress`。stall timeout 到期后停止新 dispatch、drain 并 quarantine，而不是继续无限等待。

### 7.5 multi-pool

#### serial

```text
pool A create -> execute all A groups -> drain A -> destroy A
-> pool B create -> execute all B groups -> drain B -> destroy B
```

优点是峰值接近 `max(pool_capacity)`；缺点是 class barrier 有 create/destroy 成本。serial adapter 不允许任何 A reader 越过 B create。

#### retain_all

```text
setup create A+B+...
select A -> execute/drain reader ownership
select B -> execute/drain reader ownership
finish -> destroy all pools
```

峰值计入 `sum(pool_capacity)`，但可消除稳态 allocation 和 class recreate。adapter 必须证明不同 pool backing 地址和 binding namespace 不冲突。

### 7.6 pass transition

`reload`：每个 pass 从 group 0 正常 fill。

`carry_first_group`：pass N 末尾提前填充 N+1/group 0，并保持 `Ready` 到下个 pass。它仍是 N+1 的真实 fill，不能降低 `fills` 或 logical read bytes。最后一个 pass 必须无 carry-out；cancel 在 carry 已 fill 未 claim 时仍需安全 drain。

### 7.7 取消与失败矩阵

| 时点 | 行为 |
|---|---|
| fill 未提交 | 不再提交，直接进入 drain |
| fill 进行中 | 设置 worker cancel；仍 join 并消费终态 completion |
| group Ready 未 claim | 丢弃 content 前确认没有 reader |
| group InUse | 等最后 reader fence；不能覆盖 backing |
| pass carry Ready | 取消 carry，记录未执行但已 fill 的失败证据，不生成 success receipt |
| pool barrier | 不创建下一 pool；完成当前 pool drain |
| final drain | 取消不能跳过安全 drain |
| callback/mailbox 失联 | backing ownership unknown，quarantine |

取消请求可以返回 cancellation，但不能返回 `actual_plan_verified=true`。

### 7.8 内存模型

布局编译器能精确计算的是 backing 下界/组成，不是完整请求上限：

```text
stage_layout_bytes = resident_bytes + prefix_bytes + pool_policy_bytes

serial:     pool_policy_bytes = max(pool.capacity_bytes)
retain_all: pool_policy_bytes = sum(pool.capacity_bytes)
```

真实完整请求仍需加入：

```text
process/runtime baseline
text encoder and conditioning
activation/latent
VAE/upsampler/audio/export
allocator cache and fragmentation
I/O buffers, command buffers, callback/mailbox
worker process footprint（LTX）
reviewed headroom
```

因此 slot 参数是布局输入，档位资格必须来自完整 process-tree 校准，不能只用上式倒推。

---

## 8. Selector、profile 与 catalog schema

### 8.1 用户请求

启用：

```json
{
  "schema_version": 2,
  "execution": {
    "policy": "gpu",
    "streaming": {
      "schema_version": 2,
      "enabled": true,
      "selection": "memory_tier",
      "retention": "request",
      "target_request_memory_bytes": 12884901888
    }
  }
}
```

关闭必须只编码 `schema_version` 和 `enabled=false`；不能残留 target/preset 字段。

### 8.2 内部 candidate profile

profile 是离线实验输入，建议增加完整版本字段：

```json
{
  "schema_version": 2,
  "profile_id": "z-image-turbo.512x512.12gib.candidate-001",
  "profile_revision": 1,
  "model_card": "z-image-turbo-comfy-bf16-gpu-t2i-v1",
  "target_request_memory_bytes": 12884901888,
  "headroom_policy": {
    "revision": "tree-peak-v3",
    "ratio": 0.10,
    "minimum_bytes": 536870912
  },
  "workload": {
    "width": 512,
    "height": 512,
    "frames": 1,
    "steps": 9,
    "batch": 1,
    "dynamic_text": true,
    "audio": false
  },
  "layout": {
    "stages": {
      "denoiser": {
        "residency": "streamed",
        "resident_prefix_blocks": 14,
        "block_group_size": 1,
        "slot_count": 2,
        "prefetch_distance": 1,
        "io_workers": 2
      }
    }
  },
  "policies": {
    "multi_pool": "serial",
    "pass_transition": "reload",
    "retention": "request"
  },
  "qualification": {
    "gpu_only": true,
    "allow_lora": false,
    "allow_approximation": false,
    "allow_compiled_graph": false
  }
}
```

profile 中的数字是候选，不是 public 支持声明。

### 8.3 catalog record 必备字段

在现有 `StreamingPresetRecord` 基础上，builder 必须拒绝缺少以下语义的 evidence：

```text
model card id
source manifest/snapshot digest
workload identity（含 token shape/feature digest）
runtime/adapter/reader/kernel/allocator revisions
device qualification and execution container
canonical config + layout digest
component policy + pool/pass policy
calibrated full-request bytes + scope + sample gap
P0/P1/P2/P3 evidence digest
output parity evidence
fault/cancel/source mutation evidence
reviewed commit/reviewer/review digest
release channel + revoked flag
```

### 8.4 deterministic record selection

同一个 query 匹配多个 record 时，选择顺序必须完全确定：

1. exact target；
2. exact source/workload/runtime；
3. device qualification；
4. 非 revoked 且允许的 release channel；
5. `performance.rank`；
6. calibrated bytes 更低；
7. record id/revision 作为稳定 tie-break。

如果两个 production record 在所有排序字段上冲突或不可区分，catalog build 应失败，而不是让运行时依赖 vector 顺序。

### 8.5 stale 规则

以下任一变化使旧 App resolution stale：

- catalog revision/record digest/revoked 状态；
- source snapshot/manifest；
- workload shape、steps、token rows、audio、input、LoRA；
- runtime build、adapter、reader、kernel、allocator policy；
- device class、OS build family、execution container；
- component policy 或 layout digest。

stale 只能重新 query+resolve，不能就地更新旧 exact selector。

---

## 9. 五档候选搜索与选择算法

### 9.1 不为档位手写单一布局

每个 model card 的每个 target 应先生成一组有限候选，再通过 simulator、real campaign 和 Pareto 选择产生 record。不能先拍脑袋写“8 GiB = K1，12 GiB = K2”后只验证一个点。

### 9.2 候选生成约束

```text
P in adapter-supported safe prefixes
G in adapter-supported group sizes
K in [min_slots, max_slots]
D in [0, K-1]
Q in [1, K]
pool policy in adapter-supported set
pass transition in adapter-supported set
```

先用 descriptor 精确排除：容量溢出、unsafe boundary、字段位置不一致、source materialization 不完整、suffix groups 少于 K、pool namespace 冲突。

### 9.3 候选排序

对同一 target，建议按以下目标做约束优化：

```text
硬约束：
  full_request_peak + headroom <= target
  nonplanned_swap_out == 0
  output parity == pass
  receipt/fault/source tests == pass

主目标：minimize wall median
次目标：minimize wall P95
再其次：minimize logical reads / fill wait / energy proxy
稳定性：minimize run-to-run CV and failure rate
```

只保留 Pareto frontier。两个候选性能置信区间重叠时，优先选择更简单、slot 更少、pool policy 更保守的布局。

### 9.4 目标档位与物理内存

target 是请求 envelope，不等于设备物理内存。App 推荐条件：

```text
target + system_reserve <= physical_memory
&& production record exact match
```

`system_reserve` 由 device/OS qualification 离线确定；不能用瞬时 free memory 作为 silent selection。瞬时压力只能在执行前导致安全拒绝或系统级 admission，不能换 record。

### 9.5 每个模型的候选族

以下是搜索优先级，不是已发布配置。

#### Z-Image Turbo

```text
stage: denoiser
pool: single class
G: first {1}; only benchmark >1 after correctness
K: {1,2}; K3 only if descriptor and measured overlap justify
D: K1->0, K2->{0,1}
Q: {1..K}
P: safe prefix set, from low-memory to high-prefix frontier
transition: reload
```

8/10 GiB 优先搜索低 P、K1/K2；12/16/20 GiB 允许提高 P 来减少每 pass logical read，但必须确认 activation/VAE 峰值没有被 prefix 挤压。20 GiB 也不能直接假定 resident 可用。

#### Flux.2 Klein 9B

```text
stages/classes: dual blocks -> single blocks
G: first 1
K: {1,2} per class
D/Q: within each pool
pool: serial first; retain_all only after exact peak accounting
transition: reload
P: dual/single 分别从 adapter safe boundary 搜索
```

低档位优先 serial，较高档位可比较 retain_all 是否用更多 backing 换取 class transition 性能。dual 与 single 的 P/K 不应被强制相同；catalog config 需要能表达两个逻辑 stage 或同 stage multi-pool 的确定性配置。

#### MiniMax H3 Turbo

```text
model: minimax-h3-turbo original BF16 only
stage: active DiT blocks
G: 1
K: {1,2}
D: {0,1} where D<K
Q: first 1, then 2 only if reader/I/O evidence supports
transition: reload baseline, carry_first_group optimized candidate
P: active-block safe prefix set
```

先用 reload 建 correctness baseline，再让 carry 与 reload 做同布局/同 logical fill 对照。普通 H3、量化、ANE、VDN/VSA 不进入候选生成器。

#### LTX 2.5

```text
components: text/connector, stage1, upsampler, stage2,
            video VAE, optional audio/vocoder, export
streamed candidates: stage1 and stage2 independently
pool policy: stage boundary serial first
retention: request; no implicit cross-job cache
K: stage-specific {1,2,3} only where native descriptor supports
P/G/D/Q: stage-specific
```

8/10 GiB 的首要目标是 worker 整体闭包与 stage boundary 峰值；16/20 GiB 才评估跨 stage retention。音频开关、frames/fps、upsampler route、token rows 都是独立 workload card，不能共用无区分 record。

---

## 10. Z-Image Turbo 逐文件实施规格

### 10.1 public card

首卡固定：

```text
model: z-image-turbo
layout: Comfy BF16 single-file transformer/text/VAE
operation: text-to-image
backend: eager GPU
shape: 512x512 first
LoRA/ANE/GGUF/NVFP4/ConvRot/compiled: disabled
```

### 10.2 descriptor 和 reader

`native/models/z_image/streaming_descriptor.hpp` 与 `native/platform/apple/z_image_streaming_descriptor.mm`：

- public constructor 接收 `shared_ptr<const SourceLease>`；
- header parser 使用 `duplicate_fd("transformer")`；
- descriptor 的 artifact identity 来自 lease，不重新 stat path 生成第二身份；
- 每个 block field 记录 source range、dtype、shape、storage id、alignment；
- resident text/VAE 仍进入 source closure 和 component policy；
- metadata parser 对 header length、offset overflow、overlap、short read、duplicate tensor 做严格拒绝。

`native/models/z_image/weight_stream.*`：

- public constructor只接收同一 lease；
- exact fill 只能 `pread` held/duplicate fd；
- 每个 fill 返回实际 logical bytes；
- short read、offset drift、record mismatch 直接失败；
- worker job 不捕获 path/string-heavy model state；
- private path constructor保留时应有显式 legacy 注释和测试，public call graph 不可达。

### 10.3 session hook

`probe_public_streaming()`：

- 在捕获 lease 前完成便宜 route 拒绝；
- tokenization 可以产生 workload identity，但不得分配 GPU；
- lease 覆盖 transformer/text/VAE；
- source/workload/runtime/component policy 均使用冻结 revision；
- 返回 `ValueModelStreamingProbe`。

`compile_public_streaming()`：

- dynamic cast/typed view 检查；
- record 与 probe exact identity；
- 从同一 lease 构造 `StreamingPlanView`；
- 用 record canonical config 编译；
- layout digest exact match；
- 返回持有同一 shared lease 的 snapshot。

`generate_resolved()`：

- 构造 request-scoped view/context；
- 不再按 path 重开 artifact；
- exact stream 在 begin 后首 pass 前启用 receipt；
- text conditioning、denoise、VAE、PNG export 全部走同一完整请求；
- exact stream finish 后 post-drain revalidate；
- 返回真实 execution receipt；
- 不允许 catch 后 fallback 普通 `generate()`。

### 10.4 Z-Image 测试

建议新增：

```text
tests/native/z_image_public_streaming_test.cpp
tests/native/test_z_image_public_streaming.py
```

最低用例：

| ID | 用例 | 期望 |
|---|---|---|
| ZIMG-SRC-001 | probe/snapshot lease 指针相同 | PASS |
| ZIMG-SRC-002 | reader 不按 path reopen | audit 0 |
| ZIMG-SRC-003 | path replace | pre-GPU/post-drain reject |
| ZIMG-SRC-004 | same-size mutation | open-fd/identity reject |
| ZIMG-SRC-005 | header short read/overflow | deterministic error |
| ZIMG-ID-001 | source/workload/runtime mismatch | reject |
| ZIMG-ID-002 | layout digest mismatch | reject before GPU |
| ZIMG-ROUTE-001 | GGUF/NVFP4/ConvRot | unsupported |
| ZIMG-ROUTE-002 | LoRA/ANE/compiled/image input | unsupported |
| ZIMG-RUN-001 | actual receipt matches layout | PASS |
| ZIMG-RUN-002 | receipt missing/tampered | no public success |
| ZIMG-RUN-003 | fill short read | drain/quarantine |
| ZIMG-RUN-004 | cancel at fill/reader/export | safe terminal state |
| ZIMG-RUN-005 | public failure后普通请求 | 无状态泄漏 |
| ZIMG-P0-001 | resident ABBA | P0 pass |
| ZIMG-P1-001 | direct vs generic same layout | P1 pass |
| ZIMG-P2-* | 每个候选 target完整请求 | target fit |

---

## 11. Flux.2 Klein 9B 逐文件实施规格

### 11.1 public card

首卡只覆盖 reviewed BF16 eager GPU text-to-image 9B。Flux 4B compiled graph、image input、LoRA、量化和 ANE 使用独立 card 或保持 unavailable。

### 11.2 descriptor

`native/models/flux2/streaming_descriptor.hpp` 与 `native/platform/apple/flux_streaming_descriptor.mm`：

- 从真实 metadata 枚举 dual/single block 数，不硬编码 8/24 作为 authority；
- dual/single 使用不同 layout class 和 storage namespace；
- common fixed tensors进入 resident fields；
- descriptor 声明 serial/retain-all 支持能力；
- source closure覆盖 transformer、text encoders、VAE 和影响 route 的 manifest/index/shards。

### 11.3 exact adapter

`native/models/flux2/flux_transformer.cpp`：

- pool-addressed backing，不能只保存一个隐式 active buffer；
- `select_pool()` 只切换已安全创建的 pool；
- dual reader 全完成后才能进入 single class barrier；
- binding key 包含 pool/layout class/field position；
- retain-all 不得在切换时释放上一 pool；
- receipt 记录每次 pool selection 和所有 group/fence。

`native/models/flux2/pipeline.cpp`：

- 实现三 public hooks；
- VAE/export 纳入完整请求与 component policy；
- public exact 与 resident 使用同 seed/conditioning/output route；
- 结果必须可做 PNG byte hash 或定义好的数值质量比较。

### 11.4 Flux 测试

| ID | 用例 |
|---|---|
| FLUX-DESC-001 | dual/single block/class/field closure |
| FLUX-POOL-001 | serial A drain 后才 create B |
| FLUX-POOL-002 | retain-all setup 无稳态 allocation |
| FLUX-POOL-003 | wrong pool selection/ticket reject |
| FLUX-POOL-004 | dual reader 延迟时禁止进入 single |
| FLUX-POOL-005 | pool address/namespace 不混淆 |
| FLUX-REC-001 | receipt pool sequence exact |
| FLUX-FAULT-001 | class barrier cancel/fault safe drain |
| FLUX-OUT-001 | VAE/PNG parity |
| FLUX-P0/P1/P2-* | 默认、同布局、target 档位 |

serial 与 retain-all 是两个不同 layout identity，必须独立 P1/P2，不能只测一个后共享 record。

---

## 12. MiniMax H3 Turbo 逐文件实施规格

### 12.1 public card

严格限定：

```text
minimax-h3-turbo
original BF16
native Metal GPU
quality-preserving exact route
no ordinary H3 / quant / ANE / VDN / VSA
```

### 12.2 descriptor 与 C bridge

`native/models/h3_runtime/h3_streaming_descriptor.*`：

- descriptor 根据 workload/route 给出 active block，不假定固定总数；
- pass count/step mapping 明确；
- source spans 与 native tensor binding 一一对应；
- carry policy进入 layout digest。

`native/models/h3_runtime/h3_dit.c`：

- C ABI ticket/generation 原样传递；
- fill/reader completion 都通过现有固定 POD bridge；
- 不从 C callback 修改 C++ scheduler 状态；
- synchronous completion 必须明确设置 `already_complete`，仍生成 reader identity；
- short read、Metal command failure、callback 丢失进入 sticky failure。

`native/platform/apple/h3_session.mm`：

- 实现三 public hooks和完整 source closure；
- request-scoped exact adapter产生 receipt；
- ordinary H3 route继续原逻辑并必须 P0；
- public H3 Turbo失败不污染后续 resident session。

### 12.3 carry 测试

| ID | 用例 | 不变量 |
|---|---|---|
| H3-CARRY-001 | N->N+1 group0 | 一次真实 fill，一次 claim |
| H3-CARRY-002 | last pass | 无 carry-out |
| H3-CARRY-003 | carry Ready 时 cancel | 安全 drain，无 success receipt |
| H3-CARRY-004 | carry generation mismatch | reject |
| H3-CARRY-005 | K1 请求 carry | compile reject |
| H3-CARRY-006 | delayed reader | 不覆盖 slot |
| H3-CARRY-007 | reload vs carry logical bytes | 相等 |
| H3-P1-001 | direct vs generic reload | framework gate |
| H3-P4-001 | reload vs carry | 策略收益，不能冒充 P1 |

---

## 13. LTX 2.5 worker-local 实施规格

### 13.1 authority 必须在 worker 内创建

parent 不能把以下对象发送给 worker：

```text
SourceLease/fd
StreamingAuthority pointer
ModelSession pointer
Metal/MLX object
compiled mutable pool
```

parent 只发送 canonical `JobEnvelope`：request、model root identity hint、target、catalog revision hint、correlation id。worker 自己 snapshot catalog、capture lease、probe、compile、authorize、GPU lock revalidate 和 execute。

### 13.2 JobEnvelope 建议字段

```json
{
  "schema_version": 1,
  "correlation_id": "uuid",
  "request": {},
  "requested_target_bytes": 12884901888,
  "expected_catalog_revision": "...",
  "parent_build_id": "...",
  "output_temp_path": "..."
}
```

worker 首先检查 build/container 协议兼容；hint 不构成 authority。最终响应只返回 verified summary、临时输出 hash、evidence path 和稳定 error envelope。

### 13.3 组件闭包

component policy 至少列出：

```text
tokenizer/Gemma text encoder
connector/projection
stage1 denoiser
latent upsampler
stage2 denoiser
video VAE
audio encoder/vocoder（若 audio=true）
export/finalizer
```

即使只 streaming stage1/stage2，其余组件也必须计入 source closure、workload identity 和 process-tree memory calibration。

### 13.4 worker 状态机

```text
spawned -> validating -> resolving -> authorized -> running
-> draining -> verified -> response_sent -> exited

任何阶段 -> cancelled/failed
running/draining + ownership unknown -> quarantined -> process exit
```

parent 只有收到 `verified` 响应、校验 correlation/request/resolution/output digest 并原子 rename 临时输出后，才能 JobStore commit。

### 13.5 故障测试

| ID | 故障点 | 期望 |
|---|---|---|
| LTX-WORKER-001 | resolve 前 SIGKILL | 无输出、可重新创建 worker |
| LTX-WORKER-002 | stage1 fill 中 SIGKILL | temp output 清理、旧 worker不复用 |
| LTX-WORKER-003 | stage boundary IPC EOF | quarantined/retryable |
| LTX-WORKER-004 | stage2 reader pending | parent不提交 |
| LTX-WORKER-005 | receipt 缺失/篡改 | reject |
| LTX-WORKER-006 | output hash mismatch | reject/清理 |
| LTX-WORKER-007 | catalog/source stale | 重新 resolve |
| LTX-WORKER-008 | audio component failure | 整体请求失败，不提交视频半成品 |
| LTX-WORKER-009 | normal cancel | worker drain/exit，JobStore cancelled |
| LTX-P0/P1/P2-* | 默认/同布局/target | 含完整 worker tree |

---

## 14. App、Swift 与 JobStore 实施规格

### 14.1 UI 状态模型

建议 App 内部使用：

```swift
enum StreamingPreference: Equatable, Codable {
    case off
    case target(UInt64)
}

enum StreamingAvailability: Equatable {
    case loading
    case available(NativeStreamingTargetOption)
    case unavailable(code: String)
    case stale
}
```

不要把 `slotCount/groupSize/prefix` 加入 `StudioDraft`。诊断页如需展示，只显示 native 返回的安全摘要。

### 14.2 options cache key

key 至少包含：

```text
model/model variant/model root identity
operation and input roles
width/height/frames/fps/steps/audio/batch
dynamic text and token shape-affecting options
LoRA list/strength/role
ANE/compiled/quantization/approximation route
execution container/device class/OS build family
catalog revision
```

prompt 文本本身如果影响 token rows/feature digest，就不能只按字符数缓存。可以由 native 返回 normalized workload digest 作为最终 key。

### 14.3 自动推荐

默认值始终是 Off。App 可以标记一个“推荐档位”，但只有在：

- options 对该 exact workload 返回 production `available`；
- target 加 reviewed system reserve 不超过物理内存；
- 没有 stale catalog/source；
- 用户尚未明确选择其他值。

推荐不能自动写入已存在的 draft，也不能因为内存压力变化而在运行时跳档。

### 14.4 JobStore v2

建议保存：

```text
requested selector
resolved exact selector safe fields
catalog/record/revision/resolution digest
workload/source/runtime/device digest摘要
state + stable error code
verified result/receipt摘要
temp/final output URL and hash
created/resolved/started/finished timestamps
```

禁止保存 authority pointer、fd/path identity细节、GPU pointer、worker PID、完整 receipt ticket matrix。

### 14.5 原子提交

```text
resolve success -> persist resolved job
run -> write temp output
native verified result -> hash temp output
single DB transaction:
  verify job still resolved/running and not stale/cancelled
  rename temp -> final
  persist verified summary
  state = committed
```

任何失败都不能留下被 UI 当作成功的最终文件。若 rename 与 DB 不可放进同一事务，使用可恢复 journal，并在 App 启动时 reconciliation。

### 14.6 App 测试

```text
APP-STREAM-001  Off 编码无 target/preset残留
APP-STREAM-002  五档顺序和字节值准确
APP-STREAM-003  空 catalog 全部 unavailable
APP-STREAM-004  options key 任一 workload 字段变化即失效
APP-STREAM-005  resolve 后 catalog revoke -> stale
APP-STREAM-006  source/build/device变化 -> stale
APP-STREAM-007  stale job 不调用 generate
APP-STREAM-008  internal layout 不出现在普通 UI/JobStore
APP-STREAM-009  native error code映射为明确 action
APP-STREAM-010  cancel 与 verify race 不能 commit
APP-STREAM-011  verified response + rename/DB failure可恢复
APP-STREAM-012  LTX worker EOF不提交
APP-STREAM-013  App重启恢复 running/stale/temp文件
APP-STREAM-014  Off/default生成路径无 options query副作用
```

---

## 15. 工具链详细设计

### 15.1 inspect

输入：model root、model card、canonical workload。输出：artifact closure、source identity、block/field/class、safe boundaries、resident fields、passes、adapter revisions。

硬要求：host/metadata-only；不得分配 GPU。输出可作为 descriptor golden，但不能成为 source authority。

### 15.2 compile

输入：candidate profile + inspect descriptor。输出：canonical config/layout、pool capacities、logical read bytes、digest、静态 rejection。

必须重新使用生产 `compile_layout()`，不能在 Python 重写一套略有差异的编译器。Python CLI 可以调用 host binary/C ABI。

### 15.3 simulate

使用离散事件队列模拟 fill、claim、reader、pool barrier、carry、cancel 和 timeout。每个 trace 必须能由 checker 验证：

```text
no overwrite
no claim before ready
no pool transition before required drain
bounded inflight fills/readers
all passes/groups reachable exactly once
cancel reaches terminal state
```

Simulator 只淘汰明显错误/低 overlap 候选，不认证真实内存或性能。

### 15.4 campaign

建议 policy 包含：

```text
model card/source/workload/device
candidate ids
arm order and randomized seed
warmup/measured/fault counts
fresh-process flag
sampler interval
output parity method
pressure scenario
timeout and cancellation injection points
```

每个请求生成独立 correlation id，所有 native event、sampler、worker、output 和 verifier report 用该 id join。

### 15.5 process sampler

macOS 首选稳定、无需 root 的进程 API/系统命令采样组合；必须记录 sampler 自身开销和最大 sample gap。parent/worker tree 根据 pid+start time 跟踪，避免 PID reuse。

至少采集：

```text
monotonic timestamp
pid/ppid/start identity/role
phys footprint/resident/virtual（按可用性）
MLX active/peak/cache（模型进程）
system swapin/swapout delta
memory pressure state
stage/pass/group/pool/slot counters
terminal sample marker
```

采样器崩溃、gap 超门限、未知 worker 或 terminal sample 缺失都使 run INCONCLUSIVE。

### 15.6 verifier

独立 verifier 不链接模型执行 session，输入 evidence bundle 并重算：

- manifest 文件 hash；
- canonical request/source/workload/runtime/device/record/layout digest；
- receipt schema/event/canonical digest；
- group/fill/fence/carry/pool completeness；
- synchronized process-tree peak；
- headroom fit；
- paired performance statistics；
- output parity；
- fault/cancel denominators。

verifier 版本和二进制 hash 进入 evidence。

### 15.7 catalog builder

builder 只接受 verifier 状态 PASS 的 evidence，并执行：

```text
all mandatory test IDs present
sample denominator exact
commit/build/runtime matches
target fit all measured runs
P0/P1 pass
review metadata present
record canonical digest recomputed
no conflicting production record
```

不提供 `--force-production`。实验开发如需绕过，只能输出 `candidate` 到 test-only catalog。

---

## 16. streaming 与自然 swap 的实验设计

### 16.1 四臂定义

| Arm | 执行路径 | 压力 | 用途 |
|---|---|---|---|
| A resident | 当前默认 | 正常 | 最佳性能/默认 P0 |
| B streaming | exact reviewed candidate | 正常 | 主候选 |
| C bounded | memory guard/admission | 正常 | 判断 guard 本身影响 |
| D resident+pressure | 当前默认 | controlled host pressure，允许自然 swap | swap 对照 |

不能用不同 dtype、shape、steps、source cache 状态或 output route 比较 B 与 D。

### 16.2 压力注入器

压力进程必须是独立、可终止、非特权工具：

1. 以固定 chunk 逐步 allocate+touch；
2. 记录已触碰字节和 pressure state；
3. 达到 policy 指定的 available-memory/pressure 条件后保持；
4. campaign 结束或 watchdog 超时立即释放；
5. 不使用无限分配、fork bomb、root-only tuning 或修改系统 swap 配置；
6. 如果系统进入严重不稳定状态，整组 run 中止并标为环境失败。

压力工具不能被算进模型 process-tree peak，但必须作为 host environment evidence 单独记录。每个 paired run 使用相同压力 policy，实际 swap delta仍可能不同，必须如实报告。

### 16.3 cache 与顺序控制

不建议用 `purge` 伪造冷缓存。采用 randomized/Latin-square arm 顺序，并同时报告：

- fresh-process cold-ish 首次；
- 同 source 的 steady warm cache；
- logical read bytes；
- observed I/O wait。

如果 B 获得 warm cache 而 D 没有，run 不可作为配对样本。

### 16.4 统计

每个 tuple 至少 20 paired measured runs，输出：

```text
median/P95 wall and denoise
paired ratio distribution
bootstrap confidence interval
run-to-run CV
peak/headroom margin
swapin/swapout delta
failure/cancel/quarantine rate
output parity
```

P0/P1 仍使用 2% median、5% P95 门；P3 不预设 streaming 必须胜过 swap。若 streaming 更慢但无 swap 且可稳定 fit，应把性能代价和内存收益同时写入产品说明。

---

## 17. 分层测试与门禁

### 17.1 L0：纯 host 单元测试

覆盖 canonical encoding、selector/config、catalog resolution、source lease fixture、layout compiler、receipt verifier、error priority。无模型权重、无 GPU。

门禁：每个 PR 必跑；sanitizer 通过；测试确定性且秒级/分钟级完成。

### 17.2 L1：scheduler synthetic

fake adapter 注入：延迟 fill、乱序 completion、multi-reader、pool barrier、carry、cancel、mailbox overflow、drain false。验证无死锁、无 overwrite、quarantine 规则。

门禁：任何 executor/receipt/C ABI 改动必跑 Address/Undefined/Thread sanitizer。

### 17.3 L2：模型 metadata/adapter host test

使用最小 safetensors/index fixture 验证 source closure、descriptor、layout digest、reader exact span，不执行完整 kernel。

门禁：每个模型 public hook PR 必须新增对应 source mutation 和 route rejection。

### 17.4 L3：真实模型 smoke

真实权重、最小但代表性的 workload，1–2 measured run。目标是接线、输出、receipt、cancel/fault，不用于性能发布。

### 17.5 L4：同布局性能 P1

direct exact 与 generic executor 必须使用同 P/G/K/D/Q、同 pool/pass policy、同 source/workload。策略变化另列 P4。

### 17.6 L5：完整档位 P2/P3

fresh-process 完整生成、process-tree sampler、20 paired run、fault/source mutation、swap arm、独立 verifier。只有 L5 PASS 才能产生 staging/production record。

### 17.7 L6：App/release

options、resolve、stale、JobStore transaction、revoke、升级/降级 catalog、App 重启恢复、production build 空/非空 catalog 行为。

---

## 18. 验收阈值与失败判定

### 18.1 P0 默认非回归

```text
wall median ratio <= 1.02
wall P95 ratio <= 1.05
output/quality parity
default path streaming hook = 0
default path lease capture = 0
default path pool/worker/receipt allocation = 0
```

至少对每个模型做 A-B-B-A；merge dev 后重跑。若性能 CI 噪声使置信区间跨线，标 INCONCLUSIVE，不可用单个更快样本覆盖。

### 18.2 P1 框架开销

```text
generic/direct wall median <= 1.02
generic/direct wall P95 <= 1.05
generic/direct denoise median <= 1.02
same output
same logical layout and reads
steady allocation/thread-create = 0
```

### 18.3 P2 target fit

对全部 measured run：

```text
tree_peak + max(512 MiB, ceil(0.10 * target)) <= target
swap_out_delta attributable to model pressure == 0
terminal sample present
maximum sample gap <= record threshold
no unknown child process
```

任何一个 measured run 超限则该候选 FAIL，而不是只比较 median peak。

### 18.4 功能与安全

- receipt/source/layout mismatch：FAIL；
- drain ownership unknown：QUARANTINED；
- sampler/evidence缺失：INCONCLUSIVE；
- output mismatch：FAIL；
- user cancel但安全 drain完成：cancel test PASS，不算生成成功；
- source mutation被拒绝且安全 drain：mutation test PASS；
- production catalog缺 record：正确状态是 unavailable，不是 FAIL。

---

## 19. Evidence Bundle 与可追踪性

### 19.1 目录

```text
evidence/<model-card>/<target>/<record-candidate>/<campaign-id>/
  manifest.json
  environment.json
  command.json
  request.canonical.json
  source.summary.json
  descriptor.json
  compiled_layout.json
  actual_receipts/
  events/
  process_samples/
  outputs/
  memory_summary.json
  performance_raw.jsonl
  performance_summary.json
  fault_matrix.json
  verifier_report.json
  review.md
```

### 19.2 manifest 必备字段

```text
schema/revisions
git commit + dirty flag
build/runtime/adapter/reader/kernel identities
device/OS/execution container
model card/source/workload/layout/record digests
target/headroom/sampler interval/max gap
warmup/measured/fault/mutation denominators
arm order/randomization seed
verifier binary/version/hash
overall state and per-gate state
```

dirty worktree evidence只能用于开发诊断，不能生成 production record。

### 19.3 测试 ID 到证据映射

`verifier_report.json` 必须列出每个 mandatory test ID、输入文件 hash、结果和失败原因。reviewer 不接受只有一行“all tests passed”的报告。

---

## 20. 分阶段提交与停止条件

### C3：Z-Image 首闭环

建议拆分：

```text
C3.1 shared lease metadata/reader + host tests
C3.2 public probe/snapshot + identity/layout tests
C3.3 request-scoped generate + real receipt + failure cleanup
C3.4 real model P0/P1 smoke/evidence
```

停止条件：任何 path reopen、lease不一致、receipt伪造、普通请求状态污染或 drain unknown。

### C4：Flux 9B

```text
C4.1 dual/single descriptor and source closure
C4.2 serial multi-pool exact adapter
C4.3 retain-all optional candidate
C4.4 public hooks/receipt/P0/P1
```

先让 serial正确，再研究 retain-all。不能为了性能把两个 pool 隐式融合。

### C5：H3 Turbo

```text
C5.1 public source/descriptor/hooks with reload
C5.2 C/Metal reader receipt
C5.3 carry-first-group candidate
C5.4 ordinary H3/Turbo default P0
```

carry 属于策略优化，不能阻塞 reload correctness baseline。

### C6：LTX worker

```text
C6.1 JobEnvelope/response protocol
C6.2 worker-local resolve/lease/authority
C6.3 stage1/stage2 exact receipt
C6.4 full component closure and atomic output
C6.5 kill/EOF/quarantine suite
```

### C7：App

```text
C7.1 options/query model
C7.2 advanced settings UI
C7.3 JobStore v2 transaction/stale
C7.4 LTX worker integration and restart recovery
```

### C8/C9：工具、校准、发布

工具必须先在 test-only catalog 上端到端跑通，再生成 staging record。production catalog仍保持空，直到至少一条 exact model card/target 完成 review。

### dev 合并

在模型 adapter 阶段提交并保持工作树清晰后再合并 `dev`。冲突原则：

- 保留 dev 的模型功能/性能修复；
- 保留 feat/stream 的 lease/authority/receipt 安全合同；
- 不以“能编译”为完成，合并后重跑 L0–L4 和所有默认 P0；
- 若 dev 改变 source/runtime/kernel/component identity，旧 evidence 和 record全部 stale。

---

## 21. 代码审阅逐项清单

### 21.1 控制面

- [ ] selector 与 exact config 没有混用；
- [ ] preflight ticket一次性且绑定同一 catalog snapshot；
- [ ] metadata probe不创建 GPU/worker；
- [ ] probe/snapshot持有同一 lease对象；
- [ ] record/source/workload/runtime/device/component/layout exact match；
- [ ] public unsupported无 fallback；
- [ ] error priority稳定且测试覆盖。

### 21.2 reader/scheduler

- [ ] public reader无 path reopen；
- [ ] source offset/bytes/shape有 overflow和short-read检查；
- [ ] worker只发布 bounded POD completion；
- [ ] owner独占状态和receipt；
- [ ] `D<K`、`Q<=K`、pool/ticket/generation运行时检查；
- [ ] reader完成前slot不复写；
- [ ] multi-pool barrier和carry进入receipt；
- [ ] failure停止新dispatch并安全drain/quarantine。

### 21.3 生命周期

- [ ] request-scoped context不泄漏到下一请求；
- [ ] exception/cancel每个出口都有RAII清理；
- [ ] unsafe backing不强行free；
- [ ] success前post-drain source revalidate；
- [ ] receipt只能在finish后读取；
- [ ] default/private路径不构造public对象。

### 21.4 App/release

- [ ] UI只暴露Off和五档；
- [ ] options cache key覆盖完整workload/source/catalog；
- [ ] stale/revoke不能执行旧job；
- [ ] temp output与JobStore原子提交；
- [ ] production record只能由verified evidence builder产生；
- [ ] 单record可撤回且Off/default不受影响。

---

## 22. 建议命令与 CI 分层

每次 common/model adapter PR 的基础命令：

```bash
env TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
tools/native/build_app.sh
make test-streaming-host
make test-streaming-contract
make test-streaming-audit
python3 -B tests/native/test_streaming_layout.py
python3 -B tests/native/test_streaming_preset_resolver.py
python3 -B tests/native/test_streaming_source_lease.py
python3 -B tests/native/test_streaming_actual_receipt.py
git diff --check
```

修改 executor/C ABI/receipt 时增加：

```bash
env TC_STREAMING_SANITIZER=address,undefined \
  python3 -B tests/native/test_streaming_layout.py
env TC_STREAMING_SANITIZER=thread \
  python3 -B tests/native/test_streaming_layout.py
```

模型 PR 增加对应 public test wrapper；真实模型和性能 campaign 不放进每次普通 CI，而进入带权重/设备标签的 release gate，并把 evidence digest回填到状态文档。

---

## 23. 完成定义

整体 public streaming 只有在以下条件全部满足时才算完成：

1. Z-Image Turbo、Flux 9B、H3 Turbo、LTX 2.5 都使用同一 common control/executor/receipt 框架；
2. 每个 public model card 有完整 source closure 和 request-scoped lease；
3. 每次 public success都有真实 receipt、drain和post-drain revalidation；
4. App只暴露Off与五个target，旧resolution可正确stale/revoke；
5. production record全部来自独立 verifier通过的 evidence；
6. 每个已发布 target满足P0/P1/P2，并完整报告P3；
7. cancellation、fault、source mutation、worker kill不会产生假成功或unsafe free；
8. 默认 resident/private/compiled/GPU+ANE 路径无新热路径工作且性能不回归；
9. merge dev 后重新完成 build、host/contract/audit、模型和性能签核；
10. catalog可以单record发布、撤回和回滚。

在这之前，正确表述始终是：**common framework 已具备部分基础，模型 public adapter、App、校准和 production records 仍按阶段完成；空 production catalog 表示用户尚不可用。**
