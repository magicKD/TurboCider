# 53 · Public Streaming 最终收口实施稿：代码设计、四模型接线与验收

修订日期：2026-09-18。本文是当前 `feat/stream` 工作树的实施收口稿，目标是把
`block/slot/stage streaming` 从 private/candidate 能力收敛为 App 可选择、native 可验证、
catalog 可发布的 public 产品能力。

本文不是“已经 public”的声明。当前 production catalog 仍然为空
（`tc-streaming-catalog-empty-v1`），因此任何模型、任何档位都必须在真实 evidence、
独立 verifier 和 release review 通过后才可以进入 public registry。

本文与主题文档的关系：

| 主题 | 规范来源 |
|---|---|
| layout、slot、pool、pass、retention 的定义 | [02 Configuration](02-configuration.md)、[09 Compiler](09-layout-compiler-spec.md) |
| executor、owner pump、I/O 与 fence | [10 Executor](10-executor-implementation.md)、[41 Scheduler](41-scheduler-multi-slot-and-multi-pool-implementation.md) |
| public selector/catalog/resolver | [23 Presets](23-public-memory-tier-presets.md)、[25 Preset Implementation](25-public-preset-implementation-spec.md) |
| source lease、receipt、authority | [49 Code Contracts](49-public-streaming-code-contracts-and-execution-blueprint.md)、[51 Runtime Closure](51-public-streaming-remaining-runtime-closure.md) |
| App 五档、模型 card、校准和发布 | [52 App/Config](52-public-streaming-app-config-and-model-tier-spec.md)、[50 Evidence](50-public-streaming-calibration-and-release-evidence.md) |
| 当前真实完成情况 | [13 Implementation Progress](13-implementation-progress.md) |

本文只补齐一个缺口：把这些规范串成可直接开工的文件级实现、状态机、配置产物、验收
矩阵和停止条件。

---

## 1. 结论、范围与不可改变的约束

### 1.1 最终产品形态

用户不选择 `P/G/K/D/Q`，也不选择 block layout。App 高级设置只显示：

```text
Streaming：Off / 8 GiB / 10 GiB / 12 GiB / 16 GiB / 20 GiB
```

这六个值是“目标 request memory tier”，不是 GPU 总显存、不是进程 hard RSS limit，也不是
由 native 根据当前 free memory 临时倒推的 slot 数。用户选择的 target 只参与确定性
catalog lookup；真实的 `resident_prefix_blocks`、`slot_count`、`block_group_size`、
`prefetch_distance`、`io_workers`、stage 划分和 pool 策略全部来自经过校准的 record。

### 1.2 三条路径

```text
Off
  -> 既有 resident/component_staged/legacy streamed 路径
  -> 不创建 public coordinator/context，不读取 production catalog

Public target (8/10/12/16/20 GiB)
  -> request-only validation
  -> catalog snapshot
  -> metadata probe + SourceLease
  -> exact record resolve
  -> immutable snapshot + authority
  -> GPU-lock 下 revalidate
  -> request-scoped RunContext
  -> stage/slot execution + boundary drain
  -> actual receipt verify
  -> result serialize

Private/manual exact
  -> 仍然保留给内部/测试
  -> 不得被 App public selector 隐式调用
  -> 不得把 private evidence 当 public record
```

### 1.3 必须保持的性能承诺

1. `Off` 不执行 probe、catalog snapshot、resolver、lease、receipt recorder、context、
   process-tree sampler 或任何新增锁操作。
2. 内存充足且用户未选择 public streaming 时，模型继续使用原有 resident 或
   `component_staged` 优化；不得因为 catalog 或 public runtime 已编译进 binary 就改变
   原路径的 allocator、线程数、Metal command buffer 或 graph cache。
3. public streaming 的“同布局框架开销”与“低内存收益”分开验收：不能用降低 slot 数、
   改变 dtype 或改变模型质量来掩盖框架回退。
4. 任何未认证的 model × target × workload × device 组合必须 fail-closed，不能 fallback
   到邻近档位、当前 free memory 推算的临时 layout 或自然 swap。

### 1.4 当前状态标签

本文使用以下标签，避免把设计和实现混淆：

| 标签 | 含义 |
|---|---|
| `DONE` | 已提交代码/测试或现有代码已明确支持 |
| `WIP` | 当前工作树已有部分实现，但尚未完成测试/提交 |
| `NEXT` | 可直接按本文编码的下一步 |
| `BLOCKED` | 依赖真实模型、Metal 权限、catalog evidence 或外部 release review |
| `NOT_PUBLIC` | 即使 private/candidate 可运行，也不能向 App public 暴露 |

截至 2026-09-18：common selector/catalog/resolver、`SourceLease`、receipt v2、Z-Image
第一阶段 public lease adapter、Flux 9B lease adapter已有基础；receipt v3、multi-stage
`RunContext`、boundary verifier、result 序列化已在 `aea2cf2` 落地，生命周期/失败回归在
`30fefe4` 补齐。H3 Turbo、LTX multi-stage public adapter、App 五档迁移、真实 catalog
仍未完成，因此不能宣称 public。

---

## 2. 端到端对象图与调用顺序

### 2.1 控制面和数据面

```text
App StudioDraft
    │ 只携带 streaming_mode / target GiB
    ▼
NativeRequest
    │ streaming_selector = {enabled, target_request_memory_bytes}
    ▼
PublicStreamingCoordinator
    ├─ preflight(request)
    │    ├─ validate_public_streaming_request
    │    ├─ catalog_provider.snapshot()
    │    └─ fail if catalog empty
    ├─ resolve_normalized(request, device, preflight)
    │    ├─ ModelSession::probe_public_streaming
    │    ├─ PublicPresetResolver::select
    │    ├─ ModelSession::compile_public_streaming
    │    └─ PublicPresetResolver::authorize
    └─ revalidate(execution, current_device)
         ├─ SourceLease paths/open files
         ├─ snapshot::revalidate_source
         ├─ catalog replay
         └─ authority.matches

ResolvedRequestExecution (immutable)
    ├─ Request (normalized)
    ├─ SelectedStreamingPreset
    ├─ probe (metadata + same SourceLease)
    ├─ snapshot (layout + same SourceLease)
    └─ authority
         │
         ▼
PublicStreamingRunContext (request-scoped mutable data plane)
    ├─ StageExecutor[0..N-1]
    ├─ ActualReceiptRecorder per stage
    ├─ boundary events
    ├─ cancellation / drain / quarantine
    └─ final ActualExecutionReceipt v2/v3
         │
         ▼
verify_and_attach_public_streaming_result
    ├─ layout vs runtime metrics
    ├─ actual receipt vs layout/source generation
    ├─ boundary count/order/pending readers
    └─ PublicStreamingSelectionMetrics
```

### 2.2 关键代码接口

当前接口的职责必须保持窄而明确：

```cpp
class ModelSession {
public:
    virtual std::shared_ptr<const ModelStreamingProbe>
    probe_public_streaming(const PublicResolveInput &) const;

    virtual std::shared_ptr<const ModelStreamingSnapshot>
    compile_public_streaming(
        std::shared_ptr<const ModelStreamingProbe>,
        const StreamingPresetRecord &) const;

    virtual RunResult generate_resolved(
        std::shared_ptr<const ResolvedRequestExecution>,
        const Event &, std::atomic<bool> &);
};
```

`probe` 只能读取 metadata/index/manifest，不得创建 GPU payload、worker 或 slot；
`compile_public_streaming` 只能产生 immutable descriptor/layout/source lease view；
`generate_resolved` 才允许在 revalidate 后创建 GPU executor 和 adapter。任何模型如果只
实现了 probe 或只返回合成 metrics，都不能进入 public catalog。

### 2.3 Public generate 的建议模板

下列代码是实现模板，不要求所有模型完全复制，但所有 public adapter 必须满足同样的
生命周期和失败语义：

```cpp
RunResult ModelSession::generate_resolved(
    std::shared_ptr<const ResolvedRequestExecution> execution,
    const Event &event, std::atomic<bool> &cancel) {
    PublicStreamingCoordinator coordinator(
        *this, engine_model_id(), execution_container(), catalog_provider());

    // API 层已经调用 preflight/resolve；这里仍然必须在 GPU lock 下 replay。
    coordinator.revalidate(*execution, current_streaming_device());

    PublicStreamingRunContext context(execution, cancel);
    context.mark_gpu_revalidated();

    try {
        // 单 stage:
        auto &stage = context.attach_stage(
            0, make_model_slot_adapter(*execution, event, cancel),
            adapter_implementation_id());
        stage.run_all_passes(event, cancel);
        context.finish_stage(0);

        // 多 stage 模型在这里显式 finish/drain/release，再启动下一 stage。
        // context.record_boundary(boundary) 必须发生在下一 stage attach 之前。

        context.drain_all();
        auto receipt = context.seal_receipt(
            adapter_implementation_id(),
            execution->model_snapshot->component_policy_revision());
        context.revalidate_source_after_drain();
        context.complete();

        RunResult result = make_model_result(...);
        result.streaming_receipt = std::move(receipt);
        result.streaming_stages = collect_stage_metrics(...);
        result.streaming_boundaries = collect_boundary_metrics(...);
        verify_and_attach_public_streaming_result(*execution, result);
        return result;
    } catch (...) {
        // 任何不确定的 GPU/worker 状态都进入 quarantine；不能在异常路径
        // 直接 reset slot backing，也不能把失败伪装成 resident 重试。
        context.quarantine(current_exception_detail());
        throw;
    }
}
```

实际代码不应让 adapter 自己创建第二个 `StageExecutor`、第二套 slot pool 或第二个
receipt recorder。模型 adapter 只负责：把模型 kernel 绑定到 common ticket/fence ABI、
声明真实 reader queue、在指定 boundary 调用 drain/release，并填充模型专属 metrics。

---

## 3. 请求生命周期、状态机和失败处理

### 3.1 `PublicStreamingRunContext` 状态机

当前 `run_context.hpp/.cpp` 已引入以下状态；后续实现必须保持单向迁移：

```text
created
  -> gpu_revalidated
  -> running
  -> draining
  -> receipt_sealed
  -> source_revalidated
  -> completed

任意 running/draining/receipt_sealed 阶段发生未知状态：
  -> quarantined
```

状态规则：

| 状态 | 允许操作 | 禁止操作 |
|---|---|---|
| `created` | `mark_gpu_revalidated`, attach 之前的只读查询 | 创建 slot、提交 GPU |
| `gpu_revalidated` | `mark_running`, attach stage | 重新 resolve、改 selector |
| `running` | attach 未创建 stage、run pass、finish stage、record boundary | 重复 stage、跳过 boundary |
| `draining` | retry drain、seal receipt | 新建 stage、写入旧 slot |
| `receipt_sealed` | source revalidate | 改 receipt、重新 submit |
| `source_revalidated` | complete | 重新执行任何 GPU work |
| `completed` | 只读结果 | 任何 executor 操作 |
| `quarantined` | 由 registry 保留待后续显式回收 | 自动重试、静默释放不确定 backing |

析构函数只能做两件事：已完成则直接返回；未完成则尝试 `drain_all()`，失败则
`quarantine(reason)`。不能在 destructor 中调用 catalog resolver、重新打开文件或启动线程。

**已落地的硬化项。** `aea2cf2` 已把下列前置条件编码为显式
`require_context()`/receipt verifier，避免调用者绕过 GPU revalidation：

```text
attach_stage       仅允许 gpu_revalidated 或 running
mark_running       只能从 gpu_revalidated 进入
finish_stage       只能从 running 进入，且该 stage 只能 finish 一次
record_boundary    必须在 from-stage finish/drain 后、to-stage attach 前
drain_all          只能从 running/draining 进入，不能从 created 伪造 drain
seal_receipt       所有 stage receipt 存在，boundary 数量严格为 N-1
revalidate_source  只能在 receipt_sealed/draining 进入
complete           只能在 source_revalidated 进入
```

`30fefe4` 已覆盖 duplicate stage、missing stage receipt、重复 source revalidate、drain
failure/quarantine 和 multi-stage legacy summary 拒绝；不能只依赖 adapter 遵守调用顺序。
这样可以避免未来 H3/LTX adapter 通过“先 attach、后 revalidate”而意外绕过 authority。

### 3.2 单请求唯一性

以下对象必须是 request-scoped，不得放进 model/session 全局缓存：

- `request_generation`；
- mutable `StageExecutor`；
- ticket/content generation；
- actual receipt recorder；
- boundary list；
- cancel pointer；
- source lease 的 revalidation 状态。

可以跨请求共享的只有 immutable metadata、compiled kernel、manifest digest、只读 catalog
和经过严格 identity binding 的 graph cache。共享 cache 不得拥有本请求的 fd、slot backing
或 reader fence。

### 3.3 取消与 timeout

取消必须有四个阶段：

1. 设置 request cancel flag，停止产生新 ticket；
2. owner pump 停止 claim，等待已发出的 reader fence；
3. 调用 adapter drain，确认 pending readers 为 0；
4. 成功则返回 cancellation error，失败则 quarantine。

不允许使用 `kill -9` 作为 common runtime 的正常取消手段。LTX 独立 worker 可以在
5 秒协作取消窗口后被终止，但 worker 的临时目录、子进程 PID、result receipt 和失败
原因必须由 desktop 侧原子收集，不能把子进程异常映射为成功。

---

## 4. Receipt v3 与 multi-stage boundary 合同

### 4.1 版本兼容

`ActualExecutionReceipt` 采用 additive schema：

```cpp
inline constexpr uint32_t actual_receipt_schema_v2 = 2;
inline constexpr uint32_t actual_receipt_schema_v3 = 3;

struct ActualExecutionReceipt {
    uint32_t schema_version = actual_receipt_schema_v2;
    std::string implementation;
    std::string layout_digest;
    std::string component_policy_revision;
    std::vector<ActualStageReceipt> stages;
    std::vector<ActualBoundaryReceipt> boundaries;
    std::string canonical_digest;
};
```

- v2 只能表示单 stage，`boundaries` 必须为空；
- v3 可以表示 N 个有序 stage，必须满足 `boundaries.size() + 1 == stages.size()`；
- v3 不改变 stage receipt 的 v2 语义；
- result 端必须拒绝 v2 receipt 搭配多 stage layout；
- canonical digest 的 domain separator 必须随 schema 版本变化，不能让 v2/v3 产生相同
  digest。

### 4.2 Boundary 的实际触发点

Boundary 不是“stage0 返回了所以推断 stage1 可以开始”，而是 adapter 在真实事件点
显式记录：

```text
stage N pass complete
  -> all submitted reader fences complete
  -> source stage drain helper returns success
  -> old slot/prefix backing release returns success
  -> pending_readers_after == 0
  -> record boundary event/digest
  -> attach stage N+1
```

`ActualBoundaryReceipt` 至少包含：

```cpp
struct ActualBoundaryReceipt {
    uint32_t boundary_index;
    std::string id;
    uint32_t from_stage_index;
    uint32_t to_stage_index;
    uint64_t source_generation;
    uint64_t last_reader_sequence;
    uint64_t completed_reader_sequence;
    uint64_t live_slot_bytes_before;
    uint64_t live_slot_bytes_after;
    uint64_t released_slot_bytes;
    uint64_t pending_readers_before;
    uint64_t pending_readers_after;
    bool source_stage_drained;
    bool source_stage_backing_released;
    bool next_stage_started;
    std::string event_digest;
    std::string canonical_digest;
};
```

验证必须拒绝：boundary 顺序错、stage index 错、source generation 不一致、pending reader
非零、release 标志为假、live bytes 增长、event/canonical digest 被修改。

### 4.3 Result 汇总规则

`RunResult` 同时保留：

```cpp
std::optional<StreamingRuntimeMetrics> streaming_runtime; // legacy single stage
std::vector<StreamingStageRuntimeMetrics> streaming_stages;
std::vector<StreamingBoundaryRuntimeMetrics> streaming_boundaries;
std::shared_ptr<const ActualExecutionReceipt> streaming_receipt;
```

规则：

1. 单 stage adapter 可以只填 `streaming_runtime`，common verifier 会投影到
   `streaming_stages[0]`；
2. 多 stage adapter 必须填完整 `streaming_stages`，不能把 stage2 覆盖到旧 summary；
3. legacy `streaming_runtime` 在多 stage 结果中必须为空；
4. `results.mm` 只序列化已验证 receipt 派生出的 receipt summary，不能从 counters 重新
   合成一个“看起来像 receipt”的对象；
5. `public_streaming.actual_plan_verified` 只有 `verify_and_attach_public_streaming_result`
   全部通过后才能为 true。

### 4.4 `results.mm` 的实现与后续校验

`aea2cf2` 已让 `results.mm` 输出单 stage/multi-stage 的安全摘要和 public selection；实现由
以下三个 helper 组成：

```objc
static NSDictionary *streaming_stage_result(
    const StreamingStageRuntimeMetrics &stage);

static NSDictionary *streaming_boundary_result(
    const StreamingBoundaryRuntimeMetrics &boundary);

static NSDictionary *streaming_receipt_result(
    const ActualExecutionReceipt &receipt);
```

输出形状：

```json
{
  "streaming_stages": [
    {
      "stage_index": 0,
      "stage": "denoiser",
      "receipt": {
        "schema_version": 3,
        "fills": 88,
        "groups_submitted": 88,
        "reader_fences_issued": 176,
        "reader_fences_completed": 176,
        "source_generation": 41,
        "event_digest": "..."
      }
    }
  ],
  "streaming_boundaries": [
    {
      "boundary_index": 0,
      "id": "denoiser-to-upsampler",
      "source_stage_drained": true,
      "source_stage_backing_released": true,
      "pending_readers_after": 0,
      "released_slot_bytes": 805306368,
      "event_digest": "..."
    }
  ],
  "streaming_receipt": {
    "schema_version": 3,
    "canonical_digest": "..."
  }
}
```

禁止序列化 fd、裸指针、Metal object、authority 私有字段和临时路径；这些只能留在
native/evidence 内部。

后续仍需补一层 Objective-C result fixture，验证 `native_json`、prepared 和普通 result
三条序列化分支都包含同样的 `streaming_stages`、`streaming_boundaries` 和
`streaming_receipt` 字段；当前 native build 已覆盖编译接线，但没有把真实 model result
反序列化作为独立 host test。

---

## 5. Scheduler 与 slot/pool 的统一实现规则

### 5.1 Layout 字段的含义不能混用

| 字段 | 作用 | 不允许的替代解释 |
|---|---|---|
| `resident_prefix_blocks` | 永久保留的前缀 block 数 | 不能当作 slot 数 |
| `block_group_size` | 一次 I/O/绑定的逻辑 block 数 | 不能跳过模型 block |
| `slot_count` | 精确的 reusable slot 数 | 不能自动缩减 |
| `prefetch_distance` | look-ahead 距离 | 不能隐式改变 pass 顺序 |
| `io_workers` | bounded refill worker 数 | 不能每个 request 动态无限创建 |
| `pass_transition` | reload 或 carry-first-group | 不能由 runtime counters 推断 |
| `multi_pool_policy` | serial 或 retain_all | 不能按当前内存偷偷切换 |

### 5.2 Owner pump 时序

每个 stage 的 owner pump 必须保持如下顺序：

```text
owner claims ticket
  -> fill request enters bounded I/O queue
  -> optional prefetch for distance d
  -> fill completion mailbox
  -> owner binds slot/content generation
  -> model kernels submit on one or more Metal queues
  -> reader fences recorded in receipt
  -> reader completion mailbox
  -> slot returns Ready/Reusable
```

I/O worker 不能直接修改 slot state；GPU completion callback 不能直接释放 source
backing；所有权转移必须回到 owner pump。这样才能在取消、重复 ticket、reader queue
乱序时保持可验证性。

### 5.3 Slot 状态机

```text
Empty -> Filling -> Ready -> InUse -> Retiring -> Empty
                         \
                          -> Poisoned (任何不可恢复错误)
```

- `Ready` 只能被当前 request generation 的 owner claim；
- `InUse` 期间不能被下一 pass 提前复写；
- `Retiring` 必须等所有 reader fence complete；
- `Poisoned` sticky，不能通过减少计数器伪装恢复；
- request 结束必须保证所有 slot 不是 `Filling/InUse/Retiring`，否则 quarantine。

### 5.4 Overlap 的安全边界

允许的 overlap：

- 当前 pass 的 Metal kernel 与下一 group 的 bounded pread；
- 不同 pool 之间的 independent prefetch（仅当 record 明确 `retain_all`）；
- 当前 group 的 reader fence 与下一 group 的 fill。

首版不允许：

- 跨 stage 的 weight backing overlap，除非 record 的 boundary plan 明确计入峰值；
- 任意 DAG 跨 component 的 speculative prefetch；
- 让 worker 直接持有下一 request 的 slot；
- 用自然 swap 代替 scheduler 的 release 事件。

---

## 6. 五档 target 到 model card/catalog 的设计

### 6.1 Target 解析规则

native 只接受以下 canonical target：

```cpp
constexpr uint64_t kTargets[] = {
    8 * gib, 10 * gib, 12 * gib, 16 * gib, 20 * gib,
};
```

`Off` 不进入 resolver。target 必须满足 `supported_streaming_target()`；8/10/12/16/20
之外的值一律返回 `unsupported_memory_target`，不四舍五入、不选最近值。

resolver 先用用户 target 做“可容纳”过滤，而不是要求 record 内再保存一个重复的 target
字段；一个更小的 plan 可以被多个更大的 target 复用。过滤后的排序键必须与当前
`preset_catalog.cpp` 一致：

```text
1. model/source/workload/runtime/device identity exact match
2. release.channel 是允许的 public channel，且 revoked == false
3. calibrated_request_bytes <= target - target_margin
4. performance.rank ascending
5. calibrated_request_bytes ascending
6. logical_read_bytes ascending
7. record id、revision ascending（deterministic tie-break）
```

选择完成后生成的 exact selector 必须钉住 `preset_id`、`preset_revision`、
`catalog_revision` 和 `expected_resolution_digest`；generate 前 replay 不允许重新选择另一个
“也能 fit”的 plan。

如果没有 exact record，返回明确错误：

```text
no_preset_fits_target
catalog_has_no_public_records
artifact_verification_required
unvalidated_workload
unvalidated_device
```

禁止 fallback 到：

- 较大或较小 target；
- 当前 physical free memory；
- private candidate；
- legacy `residency=streamed`；
- natural swap。

### 6.2 Model card 必须包含的字段

每个模型的 card 是 catalog builder 的输入，不是 App 运行时自由配置：

```json
{
  "model": "flux2-klein-9b",
  "public_variant": "bf16",
  "operation": "image.generate",
  "backend": "mlx_fd_reader",
  "execution_container": "engine_session_v2",
  "supported_targets_gib": [8, 10, 12, 16, 20],
  "unsupported": ["lora", "ane", "quantized", "i2v"],
  "identity": {
    "source_manifest_digest": "sha256:...",
    "reader_revision": "tc-mlx-fd-reader-v1",
    "kernel_revision": "flux9-public-v1"
  },
  "quality": {
    "reference": "resident_bf16",
    "max_abs_error": 0,
    "png_byte_exact": true
  },
  "stage_plan": ["denoiser"],
  "exploration_channel": "candidate"
}
```

card 不能声明“8 GiB 支持”而没有对应 record；`supported_targets_gib` 只是候选探索
范围，production 支持由 record + evidence 决定。

### 6.3 四模型首版 public scope

| 模型 | 首版目标 | stage | 首版限制 | 当前状态 |
|---|---|---:|---|---|
| Z-Image Turbo | 8/10/12/16（按真实 calibration） | 1 | BF16、纯 GPU、无 LoRA、无 ANE | `WIP/NOT_PUBLIC` |
| Flux.2 Klein 9B | 8/10/12/16/20（按真实 calibration） | 1 | BF16、纯 GPU、text-to-image | `WIP/NOT_PUBLIC` |
| MiniMax H3 Turbo | 12/16/20 起步，低档需实测 | 1 | original BF16、GPU-only、text-only | `NEXT/NOT_PUBLIC` |
| LTX 2.5 Distilled | 12/16/20 起步，multi-stage | 3+ | video-only、无 audio/I2V/LoRA | `NEXT/NOT_PUBLIC` |

这里的目标范围不是承诺。每个 target 必须独立完成完整请求校准和 quality/performance
evidence；若 8 GiB 在某模型上无安全 plan，catalog 不得填充该档位。

### 6.4 Catalog record 示例

```json
{
  "id": "flux9-bf16-1024x1024-s8-v3",
  "catalog_revision": "tc-streaming-catalog-2026-09-r1",
  "revision": 3,
  "source": {
    "model_variant": "flux2-klein-9b-bf16",
    "weight_format": "bf16",
    "artifact_manifest_digest": "sha256:...",
    "source_snapshot_digest": "sha256:..."
  },
  "workload": {
    "model": "flux2-klein-9b",
    "operation": "image.generate",
    "execution": "gpu",
    "device_class": "apple_metal_gpu",
    "execution_container": "engine_session_v2",
    "width": 1024,
    "height": 1024,
    "frames": 1,
    "steps": 4,
    "batch": 1,
    "audio": false,
    "dynamic_text": false,
    "conditioning_revision": "flux9-text-v2",
    "vae_policy_revision": "flux9-vae-v1"
  },
  "plan": {
    "canonical_config": {
      "schema_version": 2,
      "stages": {
        "denoiser": {
          "residency": "streamed",
          "resident_prefix_blocks": 0,
          "block_group_size": 2,
          "slot_count": 2,
          "prefetch_distance": 1,
          "io_workers": 1,
          "pass_transition": "carry_first_group",
          "multi_pool_policy": "serial",
          "retention": "request"
        }
      }
    },
    "layout_digest": "sha256:...",
    "component_policy_revision": "flux9-public-policy-v1"
  },
  "calibration": {
    "complete": true,
    "calibrated_request_bytes": 10500000000,
    "scope": "mlx_process_tree_gpu_request_v1",
    "confirmation_sample_count": 30,
    "evidence_digest": "sha256:..."
  },
  "release": {
    "channel": "public-experimental",
    "revoked": false,
    "reviewed_commit": "<commit>",
    "review_digest": "sha256:..."
  }
}
```

`calibrated_request_bytes` 必须是完整请求 process-tree 的统计值，不得只写 denoiser
weight bytes。stage/slot 的静态值和实际运行 receipt 必须一致。

---

## 7. 四模型代码实施清单

### 7.1 Z-Image Turbo

**已有基础：** `native/platform/apple/z_image_streaming_descriptor.mm`、
`native/models/z_image/streaming_descriptor.hpp`、MLX weight stream/pager 和第一阶段
lease adapter。

**下一步：**

1. 将 probe、snapshot、reader、result 统一绑定同一 `SourceLease` shared pointer；
2. `ZImage::generate_resolved` 创建 `PublicStreamingRunContext`，不要在 result 层补
   receipt；
3. 在每个 denoise pass 的 group fill/reader completion 调用 common receipt recorder；
4. 真实记录 `streaming_stages[0]`、receipt v2；
5. public card 明确拒绝 LoRA、ANE、GGUF、非认证尺寸和 dynamic text 组合；
6. 输出与 resident reference 做 PNG byte-exact 或明确的数值质量对照；
7. 将现有 6/8/10/12 GiB legacy UI 迁移为五档 selector，但只在 catalog 有 record 的
   target 显示可用。

**文件：**

```text
native/models/z_image/z_image.cpp
native/models/z_image/z_image.hpp
native/platform/apple/z_image_streaming_descriptor.mm
native/platform/apple/z_image_weight_stream.mm
native/runtime/streaming/run_context.*
native/runtime/streaming/public_result.cpp
native/platform/apple/results.mm
apps/macos/StudioState.swift
apps/macos/App.swift
```

### 7.2 Flux 9B

**已有基础：** `9a351c9 streaming: add flux public lease adapter`，transformer/text/VAE
已具备同一 lease lineage，fd-backed MLX reader 和 Metal fixture 已有。

**下一步：**

1. 将 transformer、text encoder、VAE 的 source identity 统一纳入一个 public request
   snapshot；
2. 只允许 text-to-image、BF16、纯 GPU、无 LoRA 的首版 public workload；
3. `Flux::generate_resolved` 使用 single-stage context，所有 pool/slot/carry 通过
   receipt recorder 记录；
4. 结果中的 `streaming_runtime` 只作为兼容 summary，正式数据来自 `streaming_stages[0]`；
5. 通过 process-tree sampler 记录 MLX active/peak、Metal resident、native RSS、
   swapin/out、I/O wait；
6. 将当前 18.3 GiB resident vs 11.7 GiB candidate 的 internal observation 只作为校准
   起点，不能直接写入 production catalog；
7. 复跑同布局 direct/generic P1，确保 common context 不造成额外回退。

**文件：**

```text
native/models/flux2/pipeline.cpp
native/models/flux2/flux.hpp
native/platform/apple/flux_streaming_descriptor.mm
native/runtime/streaming/run_context.*
native/platform/apple/results.mm
```

### 7.3 MiniMax H3 Turbo

H3 首版只做 H3 Turbo，不接普通 H3、quantized、audio/reference/LoRA、ANE 或
approximation。

**必须修改：**

```text
native/platform/apple/h3_session.mm
native/models/h3_runtime/h3_streaming_descriptor.*
native/models/h3_runtime/h3_dit.*
native/models/h3_runtime/h3.h
native/models/h3_runtime/h3_weights.*
native/core/stream_slot_c.h
native/runtime/streaming/c_bridge.cpp
```

**实现要求：**

1. `probe_public_streaming` 只读 original BF16 checkpoint metadata；
2. descriptor 中显式保存 block geometry、source manifest digest、reader/kernel revision；
3. source closure 和 streamed closure 使用同一 `SourceLease`；
4. C executor 导出每个 group 的 fill bytes、pool/slot、reader queues、completion sequence、
   carry，以及最终 drain；
5. H3 native legacy `allow_experimental_streaming_` gate 与 public authority gate 分开；
6. H3 public result 不从 `BlockResidencyMetrics` 反推 receipt；
7. 任何 audio/reference/LoRA/quantized request 在 public preflight 阶段拒绝；
8. private K2/G1 工程性能结果仅是 adapter 起点，public release 仍需完整 model card 和
   target record。

### 7.4 LTX 2.5 Distilled

LTX 是首个必须使用 receipt v3 的 multi-stage 模型。首版 public 只考虑 video-only、
quality mode、纯 GPU；audio、I2V、LoRA、ANE/hybrid 暂不放行。

**目标 stage graph：**

```text
stage0 connector/text conditioning
  -> boundary0: drain/release connector backing
stage1 denoiser
  -> boundary1: drain/release denoiser backing
stage2 upsampler/second denoiser
  -> boundary2: drain/release stage2 backing
stage3 VAE/decode/export (可作为 resident helper 或独立认证 stage)
```

最终 stage 数必须以真实 layout/catalog record 为准。不能为了复用单 stage verifier 把
stage2 的内存计入 stage1，也不能把 LTX worker 的进程退出当作 boundary receipt。

**worker 合同：**

```text
desktop -> serialized request envelope（无 fd/authority/GPU pointer）
worker  -> 自己 probe -> SourceLease -> resolve -> authority -> GPU execute
worker  -> 输出 result.json + verified receipt + evidence summary
desktop <- 原子读取 result.json；失败则保留错误和子进程状态
```

worker 必须使用自己的 catalog snapshot 和 device identity，desktop 不能把 authority
对象跨进程传递。临时目录采用随机 UUID、原子 rename、完成后清理；取消时保留短期
diagnostic manifest，避免丢失失败原因。

**文件：**

```text
native/platform/apple/ltx_session.mm
native/models/ltx_runtime/ltx_streaming_descriptor.*
native/models/ltx_runtime/ltx_streaming_plan.*
native/models/ltx_runtime/ltx_streaming_adapter.inc
native/models/ltx_runtime/ltx_blocks.c
apps/macos/LTXWorker.swift
```

---

## 8. App、JobStore 和配置迁移

### 8.1 新的 Swift 状态

建议在 `StudioDraft` 增加 public selector，而不是继续扩展模型专属字段：

```swift
enum PublicStreamingTarget: Int, Codable, CaseIterable, Sendable {
    case off = 0
    case gib8 = 8
    case gib10 = 10
    case gib12 = 12
    case gib16 = 16
    case gib20 = 20
}

struct PublicStreamingChoice: Codable, Sendable, Equatable {
    var target: PublicStreamingTarget = .off
    var explicitlySelected = false
}

struct StudioDraft: Codable, Sendable {
    var publicStreaming = PublicStreamingChoice()
    // legacy fields remain decodable during migration but are not emitted for
    // newly saved drafts.
    var residency = "resident"
    var zImageStreamingBudgetGiB = 10
}
```

迁移规则：

| 旧状态 | 新状态 |
|---|---|
| `residency=resident` | `publicStreaming=off` |
| `residency=component_staged` | `publicStreaming=off`，继续走 LTX worker/legacy stage path |
| Z-Image `residency=streamed`, budget 6 | `publicStreaming=off`，显示“旧实验配置需重新选择 public 档位” |
| Z-Image `residency=streamed`, budget 8/10/12 | 转为对应 target，但必须重新 native resolve |
| profile/ANE/LoRA 与旧 streamed 组合 | `off`，拒绝静默迁移 |

旧字段不能在新 public request 中同时出现。native request parser 必须把冲突报告为
`streaming_config_conflict`，并指出字段来源（draft/profile/request）。

### 8.2 App 只显示可用 target

App 需要一个只读 query：

```text
queryPublicStreaming(model, workload, device) ->
  [{target, available, reason, recommended, catalog_revision}]
```

注意：physical memory 只用于 UI recommendation；`available` 必须来自 native catalog
identity match。query 返回 unavailable 时，用户不能通过勾选强制运行；App 不自动选邻近
档位。

### 8.3 JobStore 事务

`JobStore.generate` 的顺序：

```text
1. lock busy/reentrancy
2. persist draft/job with public selector
3. native plan(request)      // request-only parse/validation
4. if target != off:
     native preflight + resolve
     show resolved target/model/availability
5. acquire session or spawn LTX worker
6. native generate_resolved
7. verify result public_streaming.actual_plan_verified
8. persist result/evidence summary
```

失败时必须保存：`error_code`、`model`、`target`、`catalog_revision`（若已读取）、
`resolution_digest`（若已生成）、`job_id`。不能只保存一段本地化字符串，否则无法
复盘 catalog/release 问题。

### 8.4 UI 文案原则

推荐文案：

```text
流式加载（高级）
目标内存：10 GiB
仅在当前模型、尺寸和 GPU 有已验证方案时可用。
目标内存用于选择已验证执行计划，不是整个应用的硬上限。
```

不应向普通用户显示：slot、group、prefix、prefetch、worker、layout digest、fd、
authority、swap。诊断页可以显示经过脱敏的 stage/target/peak/receipt digest。

---

## 9. 内存 ledger、process-tree 采样和四臂实验

### 9.1 统一内存定义

每次 calibration 必须同时记录：

```text
process_tree_peak_bytes
  = parent + descendants 的峰值 RSS/phys_footprint（按平台定义）
mlx_peak_bytes
metal_resident_peak_bytes
native_heap_peak_bytes
slot_backing_peak_bytes
activation_peak_bytes
vae_or_decode_peak_bytes
swap_in_bytes / swap_out_bytes
io_wait_seconds
wall_seconds / denoise_seconds
```

`calibrated_request_bytes` 默认使用 process-tree scope；如果某个平台只能可靠获得
`phys_footprint`，必须在 `scope` 明确写出 estimator revision，不能与 MLX allocator
统计混为一谈。

### 9.2 Sampler 设计

新增工具建议：

```text
tools/native/streaming_process_tree_sampler.py
tools/native/run_streaming_campaign.py
tools/native/verify_streaming_campaign.py
```

sampler 每 10–50 ms 采样一次，事件边界额外采样：

```text
request_start
probe_done
snapshot_done
gpu_revalidated
stage_start(stage)
pass_start(stage, pass)
boundary_begin(id)
boundary_release(id)
stage_end(stage)
request_end
```

输出 JSONL 必须包含 monotonic timestamp、PID tree、model/source/runtime identity、
target、plan digest、catalog revision 和 tool version。sampler 失败时 campaign 失败，
不能用“无数据”代替 0。

### 9.3 四臂实验

每个 model × workload × device × target 至少执行：

| Arm | 目的 |
|---|---|
| `resident` | 质量、性能和完整进程峰值 reference |
| `streaming_same_layout` | 固定同一 layout，测框架 overhead |
| `streaming_bounded` | 目标 target 下真实 public plan |
| `natural_swap` | 不使用 scheduler，观察 OS swap 的真实代价 |

公平性要求：同一模型源、同一 prompt/seed、同一温度窗口、同一 warmup、同一输出路径
策略；四臂不能混用已缓存的旧 worker 或不同的 text cache。`natural_swap` 不能写入
production plan，只作为性能/风险对照。

### 9.4 统计门槛

沿用文档 12/37/50 的门槛：

- P0 Off/default：median ratio ≤ 1.02，P95 ratio ≤ 1.05；
- P1 same-layout generic/direct：median ≤ 1.02，P95 ≤ 1.05；
- P2 bounded：`tree_peak_p95 + max(512 MiB, 10%) <= target`，且无非计划 swap；
- P3 streaming vs natural swap：必须实测 wall、tail、swap、I/O wait，不能预设 streaming
  一定更快；
- 质量：PNG byte-exact 或模型 card 指定的数值误差、seed/output hash 和 stage output
  对照全部通过。

bootstrap 结果为 `INCONCLUSIVE` 时，不能写成 PASS；可以记录为诊断信息，但不能替代
独立 release review。

---

## 10. 校准工具链和生成物

### 10.1 工具阶段

```text
inspect  -> 读取 model card/checkpoint geometry，不分配 GPU
compile  -> 生成所有合法 P/G/K/D/Q 候选和 layout digest
simulate -> 静态 slot/activation/boundary ledger，筛掉明显超预算候选
campaign -> 真实四臂运行 + process-tree 采样 + output comparison
verify   -> 独立校验 receipt/identity/statistics/quality
build    -> 只把 reviewed records 编译为 catalog
revoke  -> 生成新 catalog revision，将 record 标记 revoked
```

### 10.2 候选生成约束

候选生成器可以搜索 `P/G/K/D/Q`，但必须满足：

```text
P >= 0
G >= 1
K >= 1 且为固定候选集合
D >= 0 且不改变 pass 顺序
Q >= 1 且受 adapter worker 上限约束
retention=request
```

候选器不能使用本机当前 free memory 推断布局，也不能把非法布局“修复”为最近合法
值。每个候选输出 rejected reason，便于审计搜索空间是否被错误收窄。

### 10.3 Evidence bundle

每个 record 至少包含：

```text
record.json
model-card.json
layout.json
probe.json
source-lease.json（不含敏感路径内容）
campaign-manifest.json
process-tree.jsonl
resident/*.json
same-layout/*.json
bounded/*.json
natural-swap/*.json
quality/*.json
receipt/*.json
verifier-report.json
review.md
```

Evidence 必须绑定 `reviewed_commit`、catalog revision、tool revision、OS build、GPU
identity 和 source digest。缺任一 identity 字段时，builder 拒绝生成 production record。

### 10.4 Catalog builder 防线

builder 必须拒绝：

- `release.channel=public-stable/public-experimental` 但 `calibration.complete=false`；
- `canonical_record_digest` 不匹配；
- record 的 layout digest 与 evidence 不一致；
- quality hash 缺失；
- `tree_peak_p95` 超过 target margin；
- receipt schema/implementation 不匹配；
- stage boundary 数量不匹配；
- device/source/runtime identity 为空；
- 同一 exact key 有多个不同 plan 且没有明确 rank。

---

## 11. 分阶段实施计划与提交边界

### R3 · Common multi-stage runtime（`DONE`：`aea2cf2`、`30fefe4`）

**R3a receipt v3：**

- `actual_receipt.hpp/.cpp`：boundary 数据结构、digest、v3 verifier；
- 保证 v2 single-stage tests 不变；
- 新增 boundary mutation/order/pending reader tests。

**R3b public result：**

- `session.hpp` 增加 stage/boundary metrics；
- `public_result.cpp` 支持 N stage，legacy summary 只允许单 stage；
- `results.mm` 输出 stage/boundary/receipt v3。

**R3c run context：**

- `run_context.hpp/.cpp` 接入 build；
- attach/finish/drain/seal/revalidate/complete/quarantine；
- duplicate stage、missing receipt、drain failure 测试。

完成证据：native-only build、`make test-streaming-host`、`make test-streaming-contract`、
`make test-streaming-audit` 全部通过；receipt/context targeted tests 的 ASan/UBSan 与 TSan
通过；default audit counters 未增加。H3/LTX public generate 可以开始接线，但不能跳过各自
真实模型和性能验收。

### R4 · H3 Turbo public adapter

1. metadata probe/source lease；
2. exact snapshot/layout；
3. C bridge receipt v2；
4. H3 session `generate_resolved`；
5. host + synthetic + Metal smoke；
6. 真实四臂 calibration；
7. candidate record review。

停止条件：任何 audio/reference/quant/LoRA/ANE 请求必须 fail-closed；receipt 必须来自真实
ticket/fence；不能从旧 block counters 合成。

### R5 · LTX worker multi-stage

1. worker-local probe/resolve/authority；
2. stage graph 和 boundary receipt；
3. atomic result envelope；
4. cancellation/quarantine；
5. process-tree 采样包含 desktop + worker；
6. video-only candidate calibration。

停止条件：worker 不能接受 desktop 传入 fd/authority/GPU pointer；boundary pending readers
必须为 0；worker 异常不能生成成功 result。

### R6 · App/catalog

1. Swift public selector 和旧字段迁移；
2. native query/resolve 错误映射；
3. JobStore request transaction；
4. model × target availability UI；
5. catalog staging/release/revoke；
6. public smoke 和 upgrade/downgrade migration。

停止条件：production catalog 为空时所有 target fail-closed；`Off` 不触发任何 public
control-plane 调用；不可用 target 不允许自动 fallback。

---

## 12. 代码级测试矩阵

### 12.1 Host/contract tests

| ID | 场景 | 预期 |
|---|---|---|
| `R3-001` | v2 single-stage receipt | PASS，digest 与旧结果一致 |
| `R3-002` | v3 two-stage receipt | PASS |
| `R3-003` | boundary count mismatch | reject |
| `R3-004` | boundary order mismatch | reject |
| `R3-005` | pending readers after boundary | reject |
| `R3-006` | backing release=false | reject |
| `R3-007` | boundary digest mutation | reject |
| `R3-008` | multi-stage `streaming_stages[]` | accept |
| `R3-009` | legacy summary + multi-stage | reject |
| `R3-010` | duplicate context stage | reject |
| `R3-011` | missing stage receipt | reject |
| `R3-012` | source revalidate after drain | called exactly once |
| `R3-013` | drain failure | quarantine |
| `R3-014` | Off/default | no context/coordinator |
| `PUB-001` | empty production catalog | `catalog_has_no_public_records` |
| `PUB-002` | unsupported target 6/14 GiB | `unsupported_memory_target` |
| `PUB-003` | selector exact replay | same record digest |
| `PUB-004` | source generation mutation | reject before GPU |
| `PUB-005` | device mismatch | reject before GPU |

### 12.2 Synthetic executor tests

- K=1/2/3，G=1/2/4，D=0/1，Q=1/2/3；
- reload/carry-first-group 两种 transition；
- serial/retain-all 两种 multi-pool；
- 两个 reader queue、重复 reader、乱序 completion；
- short fill、EOF、cancel、timeout、partial create；
- ASan/UBSan/TSan；
- allocation/thread-create audit：setup 允许、steady state 不新增。

### 12.3 真实 GPU tests

每个模型最少：

```text
resident reference
private exact candidate
public candidate record
cancel during first fill
cancel at boundary (LTX)
source mutation before revalidate
output quality/hash comparison
receipt JSON round-trip
```

### 12.4 App tests

- 旧 draft 解码不崩溃；
- 旧 Z-Image 6 GiB streamed 配置不静默 public；
- Off 请求 JSON 不包含 active public selector；
- target unavailable 显示原因，不自动降级；
- JobStore 失败后保留 error code/catalog revision；
- LTX worker 结果包含 receipt/boundary summary；
- native generate 再次 resolve，不能信任 UI 缓存的 plan。

---

## 13. 验收命令和人工签字单

### 13.1 每个 common/runtime PR

```bash
git diff --check
env TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
make test-streaming-host
make test-streaming-contract
make test-streaming-audit
```

如果对应 PR 改动了 C ABI、slot ownership 或 receipt verifier，还必须运行：

```bash
make test-streaming-sanitizer
make test-streaming-tsan
```

### 13.2 每个模型 adapter PR

```bash
make test-streaming-host
make test-streaming-contract
make test-streaming-audit
make test-<model>-streaming-layout
make test-<model>-streaming-model
```

真实 Metal 测试必须在有权限的 host 外运行；沙箱内的合成 fixture 只能证明 ABI/状态机，
不能证明模型质量、峰值或 public 资格。

### 13.3 Release reviewer 必须签字

```text
[ ] model card 与 source/runtime identity 完整
[ ] exact record digest 与 evidence 一致
[ ] catalog builder 独立 verifier 通过
[ ] resident/streaming/bounded/natural-swap 四臂齐全
[ ] process-tree peak scope 明确
[ ] P0/P1/P2/P3 门槛通过
[ ] 输出质量和 receipt 通过
[ ] Off/default audit 无新增热路径工作
[ ] 未认证组合 fail-closed
[ ] App migration 与错误文案通过
[ ] rollback/revoke 方案已演练
```

只有所有框都完成，且 `release.channel=public-stable`（或经过明确实验渠道审批的
`public-experimental`）record 被编译进非空 catalog，
才能在 App 中显示对应 target 为 public 可用。

---

## 14. 待办清单与当前最优先顺序

按依赖排序，禁止跳过 common runtime 直接做 UI：

1. **当前：** H3 Turbo C receipt、lease-backed public hooks 和 `generate_resolved`；
2. **随后：** LTX worker-local authority、multi-stage boundary graph 和原子结果协议；
3. **同时准备：** 四模型 target calibration tooling，产出 candidate evidence，但不提前放行；
4. **再做：** App 五档迁移、query/resolve UI、JobStore transaction；
5. **校准发布：** 真实 8/10/12/16/20 GiB process-tree campaign、P0–P3、reviewed catalog；
6. **兼容扩展：** 对每个模型单独认证 ANE/hybrid，不复用 GPU-only evidence；
7. **发布前：** 合并 `dev` 并重新跑 default/P0/P1/P2/P3 全套，确认没有性能回退。

如果任何一步只能通过“自动换档”“自然 swap”“从 counters 合成 receipt”或“在 Off 路径
额外初始化框架”来完成，应立即停止并回到对应合同修复，而不是扩大 public 范围。

---

## 15. Definition of Done

本设计真正完成的标准不是“代码能跑一次”，而是同时满足：

1. App 只暴露 Off/五档，用户不需要理解 layout；
2. native 对每次 public request 重新 resolve，且 source/catalog/device 全部 revalidate；
3. 每个真实模型 adapter 使用同一 SourceLease、common executor、真实 receipt；
4. multi-stage 模型有可验证的 drain/release boundary；
5. 低内存 target 的 process-tree 峰值和 swap 行为有真实 evidence；
6. 默认路径没有新增可测热路径开销；
7. 未认证组合 fail-closed，失败不会静默 fallback；
8. catalog builder、独立 verifier、review、revoke 都可重复执行；
9. 生产 catalog 非空且每条 record 都能追溯到 commit/evidence；
10. 文档 13、README 和 App 文案都反映真实状态，而不是候选计划。

在达到上述 Definition of Done 之前，正确的产品状态仍然是：**框架实现中、候选能力
存在、production public streaming 尚未发布。**
