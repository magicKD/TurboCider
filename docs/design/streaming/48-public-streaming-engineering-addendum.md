# 48 · Public Streaming 工程实施附录：档位求解、代码接线与验收

修订日期：2026-09-18。状态：**工程设计与实施合同；尚未完成 public 发布**。

本文在 [47 代码实施与验收细化规格](47-public-streaming-code-implementation-and-acceptance-detail.md) 的基础上，进一步把以下内容写成可以直接分工、编码和验收的工程合同：

1. App 只显示 `Off / 8 / 10 / 12 / 16 / 20 GiB` 时，native 如何解析成一个确定的内部 layout；
2. `P/G/K/D/Q`、multi-pool、carry、retention 和 source closure 如何参与候选搜索，而不是由用户手工填写；
3. 当前 `PublicStreamingCoordinator`、`SourceLease`、`StageExecutor`、`ActualReceipt` 和四个模型 session 如何形成一条完整 request-scoped 链路；
4. 如何在内存充足机器上保持 resident 默认路径零新增工作，同时在低内存机器上避免不可控的系统 swap；
5. 如何通过 simulator、process-tree sampler、四臂实测和独立 verifier 证明“能运行、没有假成功、没有 unsafe free、性能代价可接受”；
6. 哪些代码已经存在、哪些只是拟议接口、哪些证据不足以称为 public 支持。

本文不把 private exact candidate、host fixture、tiny smoke 或单次较快的 raw wall time 写成 public 成绩。当前 production catalog 仍为空，正确产品状态仍然是：**框架控制面、source lease、receipt v2 和 Z-Image 第一阶段接线已存在；Flux/H3/LTX public 闭环、档位校准、App 开关和 reviewed catalog 仍未完成。**

为了避免本文件继续膨胀，两个可执行分册进一步展开本文：

- [49 代码合同与执行蓝图](49-public-streaming-code-contracts-and-execution-blueprint.md)：接口所有权、include 依赖、线程/事件/错误合同、Flux lease loader、逐模型文件施工、测试 ID 和 PR 停止条件；
- [50 档位校准、性能实验与发布证据](50-public-streaming-calibration-and-release-evidence.md)：五档候选搜索、process-tree sampler、四臂 swap、P0–P3、evidence bundle、独立 verifier、catalog builder 和发布签字。

49/50 只展开实现与验收，不改变本文的产品边界、主题规范和当前完成状态。

---

## 1. 冻结的产品边界

### 1.1 用户看到的是目标，不是实现参数

高级设置只允许：

```text
Streaming: Off | 8 GiB | 10 GiB | 12 GiB | 16 GiB | 20 GiB
```

用户不能编辑以下内部参数：

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

这些参数由经过校准的 `StreamingPresetRecord` 提供。一个 target 可能有多个 candidate layout，但一个已经发布的 record 必须只包含一个确定 layout。运行过程中不允许“看起来内存紧张”就偷偷减少 `K`、缩小 `G`、改变 dtype、改变 shape、切换 backend 或回退到普通 generate。

### 1.2 target 不是硬件总显存，也不是系统级强制上限

`8 GiB` 表示本次请求的进程树预算目标，不表示 GPU 物理显存一定是 8 GiB，也不表示 runtime 能控制系统的全部内存。预算验收使用：

```text
tree_peak + safety_margin <= target

safety_margin = max(512 MiB, ceil(0.10 * target))
```

其中 `tree_peak` 必须来自完整请求的 process-tree 采样，至少包括主进程、LTX worker、模型子进程和由模型创建的 helper；不能只读一个 RSS 或只读 Metal allocation counter。未测量时，target 只能显示为 tentative/unavailable，不能自动生成 public record。

### 1.3 默认路径物理隔离

当 selector 为 `Off`、缺省，或请求不使用新 public streaming：

- 不读取 production catalog；
- 不创建 `SourceLease`；
- 不执行 metadata probe 或 layout compiler；
- 不创建 `StageExecutor`、slot pool、I/O worker 或 receipt recorder；
- 不在默认热路径增加新的 heap allocation、thread creation、file scan 或 callback；
- 继续使用当前 resident、private exact、compiled graph、GPU+ANE 等既有路径。

默认路径的“无影响”必须由 audit counter、ABBA 配对性能和 build-time/assembly review 共同证明，不能只靠代码阅读。

### 1.4 GPU-only v1

第一版 public catalog 只允许已认证的 GPU route。以下路由必须 fail-closed：

- `execution=ane` 或 `gpu+ane`；
- encoder/decoder 使用私有 ANE manifest；
- compiled graph、GGUF、NVFP4、LoRA、ConvRot 等未加入 source/layout identity 的变体；
- 无法证明完整 source closure 的 checkpoint；
- `prepare-only` 或跨请求持久 backing。

未来 hybrid streaming 必须拥有独立 adapter revision、component policy、source closure、layout digest 和独立 catalog record，不能复用 GPU-only record。

---

## 2. 总体架构与职责

### 2.1 三个平面

```text
控制面 Control plane
  App/Swift -> C ABI -> request validator -> catalog snapshot
             -> model probe -> source lease -> snapshot/layout -> authority

数据面 Data plane
  request-scoped run context -> StageExecutor owner pump
  -> slot pool -> bounded I/O -> reader fence -> model kernel

证据面 Evidence plane
  actual receipt v2 -> process-tree samples -> output digest
  -> independent verifier -> catalog builder -> staging/release/revoke
```

职责必须单向流动：

```text
App 不解析内部 layout；
coordinator 不分配 GPU；
descriptor 不决定用户档位；
executor 不读取 catalog；
I/O worker 不调用 resolver；
receipt verifier 不补造实际执行事件；
catalog builder 不链接模型 GPU runtime。
```

### 2.2 当前代码对象映射

| 层 | 当前对象/文件 | 必须承担的职责 |
|---|---|---|
| selector | `native/core/streaming_contracts.hpp` | disabled、memory tier、exact preset 的 schema-v2 |
| catalog | `native/runtime/streaming/preset_catalog.*` | immutable reviewed records；production 为空时 fail-closed |
| preflight/resolver | `public_runtime.*`, `preset_resolver.*`, `resolved_request.*` | request-only validation、catalog snapshot、exact authority |
| source | `source_lease.*` | 单一 fd lineage、generation、digest、前后 revalidation |
| layout | `layout.*`, descriptor files | source metadata -> block/group/pass/pool layout |
| executor | `context.*`, `slot_pool.*`, `io_executor.*` | owner pump、slot state、reader fence、drain/quarantine |
| receipt | `actual_receipt.*` | owner-thread actual event matrix 和 canonical digest |
| verifier | `public_result.*` | authorized layout 与实际 receipt/result 的一致性 |
| C ABI | `native/api/c_api.mm` | lock、错误 envelope、所有权、JSON serialization |
| Swift | `bindings/swift/TurboCiderNative.swift` | safe summary、options、request selector |
| App | `apps/macos/JobStore.swift`, 高级设置视图 | tier selection、stale job、atomic commit |

模型侧保持三层边界：

```text
ModelSession / pipeline
  route、request context、完整 component 生命周期、RunResult

Streaming descriptor
  config/index/header -> source artifact、block、field、pass、class

Platform reader / exact adapter
  lease fd -> bounded fill -> binding -> kernel encode -> fence -> drain
```

---

## 3. 用户 selector、catalog record 与内部 layout

### 3.1 selector schema-v2

外部请求只允许如下结构：

```json
{
  "streaming": {
    "schema": 2,
    "selection": "memory_tier",
    "target_request_memory_bytes": 8589934592
  }
}
```

`target_request_memory_bytes` 只接受：

```text
8 * 2^30, 10 * 2^30, 12 * 2^30, 16 * 2^30, 20 * 2^30
```

内部 replay 或测试可以使用：

```json
{
  "streaming": {
    "schema": 2,
    "selection": "preset",
    "preset_id": "z-image-turbo-bf16-512x512-8g-p14-g1-k2-d0-q1"
  }
}
```

生产 App 不显示 `preset_id`，也不接受用户输入的 `P/G/K/D/Q`。

### 3.2 catalog record 最小结构

建议冻结如下逻辑结构；字段名可按当前 `StreamingPresetRecord` 实现调整，但语义不能减少：

```json
{
  "schema": "tc-streaming-preset-v3",
  "id": "flux2-klein-9b-bf16-512x512-12g-p8-g1-k2-d1-q2",
  "revision": 3,
  "catalog_revision": "tc-streaming-catalog-staging-2026-09-xx",
  "model": "flux2-klein-9b",
  "target_request_memory_bytes": 12884901888,
  "source": {
    "model_card": "flux2-klein-9b-bf16",
    "format": "diffusers-bf16-sharded",
    "source_digest": "sha256:...",
    "artifact_logical_ids": [
      "config.json",
      "diffusion_pytorch_model.safetensors.index.json",
      "transformer/...",
      "text_encoder/...",
      "vae/..."
    ]
  },
  "workload": {
    "operation": "image.generate",
    "width": 512,
    "height": 512,
    "frames": 1,
    "steps": 20,
    "batch": 1,
    "conditioning_revision": "...",
    "feature_digest": "..."
  },
  "runtime": {
    "runtime_revision": "turbocider-streaming-2026-09-18",
    "implementation": "generic_stage_executor_v2",
    "reader_revision": "...",
    "kernel_revision": "...",
    "cache_policy_revision": "..."
  },
  "plan": {
    "resident_prefix_blocks": 8,
    "block_group_size": 1,
    "slot_count": 2,
    "prefetch_distance": 1,
    "io_workers": 2,
    "multi_pool_policy": "retain_all",
    "pass_transition": "reload",
    "component_policy_revision": "...",
    "layout_digest": "sha256:..."
  },
  "calibration": {
    "scope": "full_process_tree",
    "calibrated_request_bytes": 10900000000,
    "headroom_bytes": 1073741824,
    "peak_p95_bytes": 10200000000,
    "sampler_interval_ms": 20,
    "max_sample_gap_ms": 100
  },
  "release": {
    "channel": "staging",
    "revoked": false,
    "reviewed_commit": "...",
    "review_digest": "sha256:..."
  }
}
```

上例中的值是结构示例，不是已认证 record。尤其不能因为某个 layout 在 host test 中成立，就把 `release.state` 写成 `production`。

### 3.3 identity 组合规则

record 选择至少要求以下 identity 全部相等：

```text
model_id
source identity / source digest
workload identity（shape、steps、token shape、conditioning）
runtime identity（reader/kernel/cache）
device class / execution container
component policy revision
layout digest
target tier
```

只要一项不匹配，返回稳定错误而不是选择“最接近”的 record。错误优先级固定为：

```text
request/schema conflict
-> route unsupported
-> catalog unavailable
-> source/workload/runtime mismatch
-> target unavailable
-> layout/authority mismatch
-> actual receipt mismatch
```

这样既不把用户请求错误隐藏在空 catalog 后面，也不让一个旧 record 静默套用到新 checkpoint。

---

## 4. 从目标档位到 exact layout 的确定性算法

### 4.1 分层流程

```text
1. normalize request
2. validate selector/legacy conflict
3. query immutable catalog snapshot
4. model metadata probe + SourceLease capture
5. derive workload/source/runtime identity
6. enumerate candidates allowed by model profile
7. static fit filter（descriptor bytes、slot capacity、known component bytes）
8. lookup measured records matching all identities
9. choose deterministic best record
10. compile snapshot and exact layout
11. create non-serializable authority
12. execute only after lock-time revalidation
```

### 4.2 候选枚举约束

每个模型 profile 提供候选族，而不是由通用 resolver 猜测。候选生成器应使用：

```cpp
struct StreamingCandidateProfile {
    std::vector<uint32_t> prefix_values;
    std::vector<uint32_t> group_values;
    std::vector<uint32_t> slot_values;
    std::vector<uint32_t> prefetch_values;
    std::vector<uint32_t> worker_values;
    std::vector<MultiPoolPolicy> pool_policies;
    std::vector<PassTransition> pass_transitions;
    uint32_t min_slots = 1;
    uint32_t max_slots = 8;
    uint64_t max_inflight_bytes = 0;
};
```

候选必须满足：

```text
1 <= G
1 <= K
0 <= D < K
1 <= Q <= K
每个 layout class 的 suffix group_count >= K
每个 pool 的容量/对齐可计算
pass transition 与 adapter 能力匹配
multi-pool policy 已被 adapter 显式声明支持
```

不能在 resolver 中自动改变 block 顺序；`G` 只控制批量加载/驻留，不跳过模型 block。

### 4.3 内存模型

对一个 stage，静态上界至少包括：

```text
resident_prefix_bytes
+ pool_count * K * aligned_slot_capacity
+ active bindings / descriptor tables
+ reader scratch and completion mailbox
+ kernel scratch / activation allowance
+ component handoff overlap
+ allocator fragmentation allowance
```

完整 request 上界：

```text
request_peak_upper =
    max(stage_peak_i)
  + max(component_overlap_boundary)
  + process_overhead
  + sampler_uncertainty
```

`source_read_bytes`、`destination_bytes`、`logical_content_bytes` 必须分别记录。不能用 destination bytes 冒充文件读取量，也不能用 descriptor-only projection 冒充完整进程上界。

### 4.4 候选排序

resolver 不直接按 `K` 越小越好排序。推荐顺序：

```text
1. measured fit：tree_peak + margin <= target
2. release state：production > staging > tentative
3. output/quality parity：exact > approximate
4. performance rank：经 campaign 的 rank
5. calibrated peak：越低越好
6. logical read bytes：越低越好
7. fewer pools / workers：仅作为最后 tie-break
8. preset id / revision：字典序确定性 tie-break
```

如果没有 measured fit record，返回 `target_unavailable`；不要用公式估算后伪造可执行 record。物理内存推荐只影响 UI 初始选中项，不改变用户明确选择的 tier，也不能绕开 record gate。

### 4.5 物理内存推荐

推荐器只读取可用物理内存和系统保留量，输出 tentative suggestion：

```text
usable = physical_memory - os_reserve - app_reserve
recommend highest tier T where T + safety_margin <= usable
```

推荐结果必须标记 `recommended=true` 而不是 `selected=true`。若用户选择的档位没有 record：

- UI 显示“此模型/工作负载尚未验证该档位”；
- native 返回 unavailable；
- 不自动切到另一个档位，除非用户明确允许“自动选择已验证档位”。

### 4.6 buffer/headroom policy

用户最初提出的 `x%` buffer 不应成为每次执行时可随意变化的隐藏参数。public v1 建议：

```text
headroom_policy_revision = tc-streaming-headroom-v1
percentage_margin        = 10%
minimum_margin           = 512 MiB
```

离线探索工具可以比较 5%/10%/15% 等策略，但一个 catalog record 必须固定：

```text
target bytes
headroom policy revision
measured maximum sample gap
process-tree scope
calibrated peak
confirmation sample count
```

若未来 App 允许用户输入任意 `Y` 和 `X%`，它应作为 experimental/manual 模式，与 public memory tier 分开；不能用用户输入的 buffer 修改 reviewed record 的资格。系统内存 guard 仍然是独立 admission 层：它可以在执行前安全拒绝，但不能在运行中改 layout 或证明 OS 绝不会 swap。

### 4.7 四模型初始搜索网格

以下只是进入 simulator/campaign 的搜索网格，不是档位映射或已发布配置：

| 模型 | stage/class | baseline | 扩展候选 | 低档位优先方向 | 高档位优先方向 |
|---|---|---|---|---|---|
| Z-Image Turbo | single denoiser class | `G1, K1, D0, Q1, reload` | `K2, D0/1, Q1/2`，再搜索 safe prefix | 低 P、K1/serial，保证 VAE boundary | 提高 P 或 K2，减少 wait/read |
| Flux 9B | dual + single pools | `G1, K1, serial, reload` | 每 pool K2、`retain_all` | serial、低 P、逐 pool release | retain-all 与较高 prefix 做峰值/切换对照 |
| H3 Turbo | active DiT blocks | `G1, K1, D0, Q1, reload` | `K2/D1/Q1-2`、carry-first-group | reload correctness、低 P | carry 与 reload 同布局对照 |
| LTX 2.5 | stage1/stage2 独立 | stage boundary serial、stage-specific K1 | K2/K3、stage-specific P/D/Q | 先消除 stage overlap 峰值 | 只在 multi-stage receipt 完成后研究 retention |

8/10/12/16/20 GiB 不能简单固定为 K1/K2/K3。对每个 model card、shape、steps、audio、token rows 和 component route，候选都要重新编译、模拟和完整采样；只把 Pareto frontier 上、满足 target 的候选写入 catalog。

---

## 5. request-scoped 执行上下文与生命周期

### 5.1 不再依赖 session 隐式状态

当前 Z-Image/Flux 迁移期可能在 session 上保存 `public_stream_lease_` 和 target。最终形态应引入：

```cpp
struct PublicStreamingRunContext final {
    std::shared_ptr<const ResolvedRequestExecution> execution;
    std::shared_ptr<const SourceLease> lease;
    std::unique_ptr<ModelStreamingSnapshot> snapshot_view;
    std::unique_ptr<StageExecutor> executor;
    std::atomic<bool> cancelled{false};
    bool gpu_locked = false;
    bool drained = false;
    bool source_revalidated = false;
    bool receipt_sealed = false;
};
```

要求：

- context 只能由一次 `generate_resolved()` 创建；
- 不可复制，不可跨 request/session 缓存；
- `lease`、snapshot、reader、executor、receipt 的生命周期必须包含在 context 内；
- destructor 必须按照 drain -> revalidate（若可行）-> release/quarantine 顺序执行；
- cancel、exception、serialization failure 都走同一 cleanup；
- context 结束后 session 不得残留当前 target、lease、layout 或 receipt。

迁移期若必须使用 session 成员，必须使用 RAII binding：构造时保存旧值，析构时恢复旧值，并在进入时拒绝 reentrant public request。

### 5.2 public 请求事务

```text
App draft
  -> query options
  -> select target
  -> encode Request schema-v2
  -> tc_engine_resolve_streaming_json
       request-only validate
       immutable catalog snapshot
       model probe + shared SourceLease
       record select
       descriptor/layout snapshot
       non-serializable authority
  -> JobStore stores safe resolution summary only
  -> tc_engine_generate_json
       parse and preflight again
       acquire engine lock
       acquire global GPU lock
       catalog/source/device/layout revalidate
       model generate_resolved(context)
       exact StageExecutor
       final drain
       post-drain lease revalidate
       actual receipt verify
       result verify
       atomic output commit
```

resolve 返回的 summary 不是可以跨进程恢复的 authority。用户点击 Generate 后 native 必须重新建立 execution authority，防止 catalog revoke、source replacement、runtime update 或 GPU device change。

### 5.3 终结顺序

成功与失败都遵循：

```text
stop new dispatch
-> request cancellation to I/O and GPU adapter
-> consume completion mailbox
-> wait for all reader fences
-> adapter.drain()
-> if drain false: quarantine adapter + backing + mailbox
-> if drain true: seal actual receipt
-> SourceLease::revalidate_after_drain()
-> common verifier
-> destroy safe pool / release lease
-> serialize verified result or error
```

任何路径不得先 free slot/backing 再等待 reader。`quarantined=true` 时禁止 retry generate；只能由明确的 quarantine owner 在 drain 成功后回收。

### 5.4 LTX 之前必须完成 multi-stage result/receipt 扩展

当前 `public_result.cpp` 明确要求 `layout.stages.size() == 1`，`RunResult` 也只有一个 `StreamingRuntimeMetrics`。这对 Z-Image、Flux 和 H3 的首个单 denoiser card 足够，但不能完整表达 LTX 的 stage1、upsampler boundary、stage2 和后续 component handoff。LTX public 接入前必须先做 additive 扩展，不能把多个阶段伪装成一个 stage。

建议新增：

```cpp
struct StreamingStageRuntimeMetrics final {
    std::string stage_id;
    std::string implementation;
    std::string layout_digest;
    uint32_t resident_prefix_blocks = 0;
    uint32_t block_group_size = 0;
    uint32_t slot_count = 0;
    uint32_t prefetch_distance = 0;
    uint32_t io_workers = 0;
    uint32_t group_count = 0;
    uint32_t pass_count = 0;
    std::string pass_transition;
    std::string multi_pool_policy;
    uint32_t pool_count = 0;
    uint32_t slot_bundle_count = 0;
    bool drained = false;
};

struct StreamingComponentBoundaryReceipt final {
    std::string from_stage;
    std::string to_component;
    uint64_t event_ordinal = 0;
    bool readers_drained = false;
    bool pools_released = false;
    std::string canonical_digest;
};
```

`RunResult` additive 增加：

```cpp
std::vector<StreamingStageRuntimeMetrics> streaming_stages;
std::vector<StreamingComponentBoundaryReceipt> streaming_boundaries;
```

兼容规则：

- 单 stage 模型继续填当前 `streaming_runtime`，同时可填 `streaming_stages[0]`；
- multi-stage 模型必须填 `streaming_stages`，不能只填 legacy summary；
- `ActualExecutionReceipt::stages` 已是 vector，verifier 应按 layout stage 顺序逐项核对；
- stage id 必须唯一，receipt/runtime/layout 的 stage 顺序和数量必须完全相等；
- stage1 receipt finish 和 backing release 成功后，才能记录 `stage1 -> upsampler` boundary；
- stage2 开始前不得保留未纳入 record 的 stage1 pool；
- boundary receipt 若需要改变 actual receipt schema，新增 schema v3；不可在 schema v2 下偷偷增加 canonical 字段导致旧 digest 失效；
- C ABI/Swift 只公开 aggregate 和安全的 stage/boundary summary，完整 group matrix 仍留在 native/evidence。

建议文件改动：

```text
native/runtime/session.hpp
native/runtime/streaming/actual_receipt.hpp/.cpp
native/runtime/streaming/public_result.cpp
native/api/c_api.mm
bindings/swift/TurboCiderNative.swift
tests/native/streaming_actual_receipt_test.cpp
tests/native/streaming_preset_resolver_test.cpp
```

验收增加：two-stage ordered success、stage count/order mismatch、stage1 drain false、boundary missing、stage1 pool 未释放、stage2 source generation 不一致、cancel at boundary、旧 single-stage v2 replay。multi-stage verifier 没有完成前，LTX catalog record 必须保持 unavailable。

---

## 6. StageExecutor、slot 与 overlap 的实现合同

### 6.1 slot 状态机

每个 slot 必须可以表示以下状态：

```text
vacant
fill_submitted
fill_ready
claimed
encode_submitted
reader_pending
reader_complete
reusable
quarantined
```

合法迁移：

```text
vacant -> fill_submitted -> fill_ready -> claimed
claimed -> encode_submitted -> reader_pending -> reader_complete -> reusable
任何状态 --cancel/fault--> drain_pending -> reusable | quarantined
```

禁止：

- `fill_submitted` 直接复写成另一个 group；
- 未完成最后 reader 就变成 `vacant`；
- worker 线程直接修改 owner-only receipt vector；
- completion mailbox 未消费就销毁 adapter；
- pass barrier 未完成就切换 pool。

### 6.2 owner pump 伪代码

```cpp
void StageExecutor::run_pass(pass, step, cancel, timeout) {
    owner_thread_only();
    check_layout_invariants();
    encode_prefix(pass);

    GroupCursor cursor = first_group(pass);
    while (cursor.has_value()) {
        check_cancel(cancel);
        consume_completions_until_refill_window(cursor, timeout);

        SlotTicket ticket = choose_vacant_slot();
        if (ticket.has_unretired_reader())
            wait_and_consume(ticket, timeout);

        submit_fill(cursor.group, ticket);
        if (receipt_enabled()) receipt.fill_submitted(ticket);

        if (overlap_next_fill_after_claim())
            mark_claimable_after_content_ready(ticket);
        else
            wait_fill_completed(ticket, timeout);

        prepare_group(cursor.group, ticket);
        ReaderSet readers = encode_group(cursor.group, ticket, mailbox);
        seal_reader_events_before_commit(readers);
        if (receipt_enabled()) receipt.readers_issued(ticket, readers);
        cursor.advance();
    }

    drain_pass_readers(timeout);
    record_pass_completed(pass, carry_if_allowed());
}
```

上面是行为合同，不要求所有 adapter 使用相同函数名。关键是不变量：owner 独占 slot 状态和 receipt；worker 只返回 bounded POD completion；reader fence 注册必须发生在 GPU commit 前；最后一个 reader 未完成时 slot 不可复用。

### 6.3 D/K/Q overlap

对 `K` 个 slot、prefetch distance `D`、I/O worker `Q`：

```text
0 <= D < K
Q <= K
最多 K 个 slot backing 同时 live
最多 Q 个未消费 fill completion
最多 D 个尚未 encode 的预取 group
```

推荐 baseline 是 `D=0` 或 `D=1`，先用 serial pool 验证正确性，再开启 retain-all 或更大的 `D`。增大 `D` 只能在完整 process-tree 校准后发布，因为 I/O 队列、compressed page、decoder scratch 和 kernel staging 可能使理论 slot 上界失真。

### 6.4 multi-pool 与 pass transition

`serial`：前一个 pool drain 完成后销毁，再创建下一个 pool；峰值低、正确性简单，作为所有模型 baseline。

`retain_all`：多个 layout-class pool 同时 live，由 adapter 显式实现 `select_pool(pool)` 和 `destroy_pool(pool)`；只能用于经过完整峰值校准的 record。

`reload`：pass 结束后 reader drain，再按正常顺序加载下一 pass。

`carry_first_group`：只有最后一个 group 的 reader、binding、content generation 都满足 carry 合同时，才允许将它移交下一 pass；receipt 必须记录 `ActualCarryReceipt`，否则按 reload 处理。

优化不能改变模型数学顺序，也不能用 carry 事件掩盖未完成 reader。

### 6.5 no-progress 与死锁保护

executor 应记录：

```text
last_owner_progress_time
last_completion_time
last_reader_completion_time
mailbox_depth
pending_fill_count
pending_reader_count
```

在 `stall_timeout` 内没有任何状态进展时，执行：

```text
stop dispatch -> request cancel -> drain -> quarantine if unknown
```

不允许无限等待，也不允许超时后直接 free GPU backing。

---

## 7. source closure 与 SourceLease

### 7.1 closure 原则

public adapter 必须在 probe 阶段列出所有会影响执行结果或 parser/layout 的 artifact。至少包括：

```text
model config
index/manifest
所有 weight shards
tokenizer/config（若 tokenization 影响 workload identity）
text encoder config/index/weights
VAE config/weights
adapter-specific kernel/manifest（若参与 route）
```

若一个文件未进入 closure，却在运行时被 path reopen，public 资格无效。可选组件（LoRA、controlnet、audio、ANE manifest）必须要么被明确拒绝，要么加入 source/workload/component identity；不能隐式忽略。

### 7.2 单一 fd lineage

推荐调用顺序：

```cpp
auto lease = SourceLease::capture(files);       // 打开并持有原始 fd
parse_config(lease->duplicate_fd("config.json"));
parse_index(lease->duplicate_fd("...index.json"));
auto snapshot = compile_descriptor(lease);      // 只使用 duplicate fd
reader = AdapterReader(lease);                  // 禁止按 path reopen
```

`SourceLease::revalidate_paths()` 检查 named path 是否仍指向同一 canonical target，`revalidate_open_files()` 检查持有 fd 的 stat identity，`revalidate_after_drain()` 在 reader 完全结束后再次执行。generation/digest 要进入 authority 和 receipt。

### 7.3 short read、overflow 和 parser 防护

所有 source range 读取必须检查：

```text
offset <= file_size
length <= file_size - offset
offset + length 不溢出 uint64_t
read 返回值 == requested bytes
dtype/shape/stride 与 descriptor 一致
artifact logical id 唯一
index 中引用的 shard 都在 lease closure
```

任何失败都返回稳定错误并进入 cleanup，不允许继续使用部分填充 slot。

---

## 8. 四个模型的 public adapter 施工单

### 8.1 Z-Image Turbo

**public route（首版建议）**：Comfy BF16、single-file transformer/text/VAE、eager GPU、text-to-image、无 LoRA/ANE/compiled graph/GGUF/NVFP4/ConvRot。

**source closure**：

```text
transformer config/index/weights
text encoder config/index/weights
VAE config/weights
tokenizer files/config
```

**代码落点**：

```text
native/models/z_image/streaming_descriptor.hpp
native/models/z_image/weight_stream.hpp
native/models/z_image/z_image.cpp/.hpp
native/platform/apple/z_image_weight_stream.mm
tests/native/z_image_public_streaming_test.cpp
tests/native/test_z_image_public_streaming.py
```

**实现不变量**：

- probe、snapshot、weight reader 使用同一 `shared_ptr<const SourceLease>`；
- denoise exact stream 结束后先 finish/drain、封存 receipt、释放 transformer backing，再进入 VAE；
- VAE 不得持有 transformer slot 的隐式 reader；
- 非法 target/authority 在修改 session 临时状态前拒绝；
- private candidate 和 default resident 时序保持不变。

**验证**：host source closure、route rejection、path replacement、receipt digest、真实 512×512 GPU full request、五档候选校准。当前已有 host/synthetic 与接线验证，但没有 production record。

### 8.2 Flux.2 Klein 9B

**public route**：BF16 eager GPU、text-to-image、无 LoRA/ANE/compiled graph；首版只允许 Klein 9B，不扩展 Flux 4B compiled graph。

**source closure**：

```text
transformer/config.json
transformer/diffusion_pytorch_model.safetensors.index.json
transformer/*.safetensors（按 index 权威校验）
text_encoder config/index/weights
vae config/weights
tokenizer/config
```

**当前工作树接线要求**：

1. `MlxWeightPager` 的 lease-backed constructor 以 logical id 映射 artifact，禁止 path reopen；
2. `StreamingMetadata` 的 config/index/shard parser 全部从 lease duplicate fd 读取；
3. 删除 probe 中无实际用途的 index `ifstream`，由 lease-backed metadata 做权威 index/shard 校验；
4. `FluxExactStream` public constructor 使用 `generic_stage_executor_v2` 并启用 receipt；
5. `Flux::run()` 将 exact receipt 组装成 `RunResult.streaming_receipt`，public result 通过 common verifier；
6. denoiser 完成后释放 transformer pool，再进入 VAE/upsample 边界；
7. public runtime metrics 必须包含 `implementation/layout_digest/stage/P/G/K/D/Q/group/pass/pass_transition/retention/component_policy/multi_pool/pool/slot/worker/drained`。

**candidate 顺序**：先 `serial + reload`，再 `retain_all + reload`，最后才考虑 carry 或更激进 prefetch。当前 private Flux 已有同布局 direct replay P1，但该证据不授予 public 资格。

### 8.3 MiniMax H3 Turbo

**public 范围**：只做 original BF16 MiniMax H3 Turbo GPU；不扩展普通 H3、量化变体或 ANE route。

**代码落点**：

```text
native/platform/apple/h3_session.mm
native/models/h3_runtime/h3_streaming_descriptor.*
native/models/h3_runtime/h3_dit.c
native/models/h3_runtime/h3_streaming_policy.*
```

**关键要求**：

- descriptor 必须列出 denoiser active block、prefix 和每个 safetensors range；
- C/Metal reader 使用统一 generation、读完后通过 reader fence 通知 owner；
- 先实现 reload baseline，再实现 carry-first-group；
- C bridge 的 destroy/status 必须能区分 safe drained 与 quarantined；
- `exact_streaming_finished`、`exact_streaming_poisoned` 等 legacy metrics 不能代替 actual receipt v2；
- H3 普通 default/resident P0 单独复跑，不能用 exact candidate 性能覆盖默认回归。

### 8.4 LTX 2.5

LTX 的 streaming 生命周期必须在 worker 内闭合，不能让 desktop App 持有 native authority 或 GPU backing。

**worker 内顺序**：

```text
JobEnvelope
  -> parse/validate
  -> query catalog
  -> capture SourceLease
  -> probe/snapshot/authority
  -> acquire worker GPU lock
  -> stage1 executor
  -> upsampler boundary: drain/release
  -> stage2 executor
  -> video/audio VAE/vocoder
  -> atomic output commit
  -> verified response
```

**必须覆盖的 component closure**：

```text
stage1 transformer
latent spatial upsampler
stage2 transformer
video VAE
audio VAE / vocoder / BWE（若 route 使用）
conditioning/tokenizer/Gemma cache identity
export/ffmpeg route manifest
```

**故障合同**：worker SIGKILL、IPC EOF、cancel grace timeout、source mutation、output rename failure 都必须让 JobStore 进入 failed/cancelled/quarantined，不得写入 verified success。

---

## 9. App、Swift 与 JobStore

### 9.1 options API

`tc_streaming_options_json` 返回当前模型/workload/device 的安全摘要：

```json
{
  "schema": 2,
  "catalog_revision": "...",
  "physical_memory_bytes": 17179869184,
  "recommended_target_bytes": 12884901888,
  "options": [
    {"target_bytes": 8589934592, "state": "unavailable", "reason": "no_record"},
    {"target_bytes": 12884901888, "state": "staging", "preset_count": 1},
    {"target_bytes": 21474836480, "state": "unavailable", "reason": "no_record"}
  ]
}
```

Swift 只接收 `state/reason/recommended` 等安全字段，不接收 slot matrix、fd、path、reader ticket 或内部 authority。

### 9.2 UI 行为

1. 默认值是 `Off`，保持与当前 App 一致；
2. 物理内存推荐值可展示“推荐”，不自动覆盖用户选择；
3. unavailable 档位可显示但不可提交；
4. catalog revision 变化时清除旧选项并重新 query；
5. 选择 streaming 后，JobStore 保存 requested target 与 resolution summary digest；
6. generate 时 native 重新 resolve，summary 仅用于 UI/诊断；
7. resolve/generate 任一步 stale，JobStore 原子转为 `stale`，不自动 fallback 到 resident；
8. 用户选择 Off 时，彻底走原有 generate/worker 逻辑。

### 9.3 JobStore 状态

```text
draft
 -> resolving
 -> resolved
 -> running
 -> cancelling
 -> verified
 -> committed

任意阶段 -> stale | failed | cancelled | quarantined
```

输出文件采用临时路径 + fsync/atomic rename；只有 common verifier 通过、输出 digest 生成、source revalidation 完成后，才可把 JobStore 标记为 `verified`。LTX worker 的事件日志不能直接把 job 标成成功。

---

## 10. 工具链设计

工具链必须支持 test-only catalog，先验证闭环，再生成 staging/production record。建议接口如下：

### 10.1 inspect

输入：模型目录、route、workload、device。输出：source closure、metadata identity、descriptor block count、可用 candidate profile。

```text
turbocider-stream inspect \
  --model flux2-klein-9b \
  --root <model-root> \
  --width 512 --height 512 --steps 20 \
  --json-out inspect.json
```

inspect 不创建 GPU backing，不授予 execution authority。

### 10.2 compile

输入：inspect snapshot + candidate `P/G/K/D/Q`，输出 canonical layout digest、静态容量、source ranges 和 simulator input。

```text
turbocider-stream compile \
  --inspect inspect.json \
  --prefix 8 --group 1 --slots 2 --prefetch 1 --workers 2 \
  --pool retain_all --transition reload \
  --out candidate.json
```

### 10.3 simulate

使用 fake reader/fence 注入：延迟、乱序、short read、cancel、mailbox overflow、pool barrier 和 carry。输出：

```text
simulation_summary.json
actual_receipt.json
state_trace.jsonl
```

模拟器必须能证明无 overwrite、无 double destroy、无 reader-before-commit、无 owner race，但不能替代真实 Metal/MLX 证据。

### 10.4 campaign

campaign 每个 candidate 运行：

```text
warmup >= 3
measured >= 20 paired requests
fresh-process arm
steady-cache arm
fault/cancel/mutation subset
```

每次运行输出：request canonical、source/record/layout digest、actual receipt、process samples、output digest、wall/denoise、logical reads、swap delta、failure state。

### 10.5 process-tree sampler

采样器必须：

- 记录 root pid 与所有 descendants；
- 记录 RSS/VM/physical footprint（平台支持时）、swap in/out、thread count、GPU allocation counter（若可得）；
- 记录采样 monotonic timestamp；
- 记录 child appearance/disappearance；
- 终止时再采样一次；
- 若 sample gap 超过 record threshold，结果为 `INCONCLUSIVE`，不能生成 PASS。

### 10.6 verifier

verifier 独立于运行时实现，检查：

```text
manifest/source/workload/runtime/device identity
layout digest
per-stage/per-pass/per-group receipt
reader fence issued == completed
fills == groups_submitted
source generation and post-drain revalidation
output parity / finite / expected dimensions
tree_peak + margin <= target
swap policy
failure/cancel/quarantine semantics
```

### 10.7 catalog builder

只有 verifier 对 mandatory tests 全部 PASS，且 review signer 确认 source closure、性能和产品边界后，builder 才能把 candidate 转为 staging record。production builder 拒绝：

- dirty worktree evidence；
- missing environment/source digest；
- INCONCLUSIVE sampler；
- 任意 P0/P1/P2 FAIL；
- 未签名或不匹配的 verifier revision。

---

## 11. streaming 与自然 swap 的四臂实验

目标不是预设 streaming 一定更快，而是回答三个独立问题：

1. 在同一 target 下，streaming 是否避免非计划 swap；
2. 避免 swap 的 wall/denoise 代价是多少；
3. resident、streaming、bounded resident、系统 swap 哪个在具体模型/机器上最优。

四臂：

```text
A resident：当前默认 resident 路径
B streaming：public exact layout + StageExecutor
C bounded：内存 guard/显式 bounded allocation（若实现）
D natural swap：关闭 streaming，让系统自然处理压力
```

实验控制：

- 相同 source、request、seed、steps、output format；
- randomized/Latin-square arm 顺序；
- fresh-process cold-ish 与 steady warm-cache 分开报告；
- 不使用 `purge` 冒充冷缓存；
- 同一温度/电源/后台进程策略；
- 记录 logical read bytes、I/O wait、swap delta、tree peak、失败和 output parity。

报告至少包含：

```text
wall median/P95
denoise median/P95
tree_peak median/max/P95
swap_in/out delta
logical read bytes
output parity
failure/cancel/quarantine rate
paired ratio + bootstrap interval
```

判定规则：streaming 只有在“目标 fit、无非计划 swap、输出正确”后才有资格与 swap 比 wall；如果 streaming 比 swap 慢，但稳定 fit 且避免 swap，应如实记录为内存收益换性能，不能宣称加速。若 streaming 和 swap 都超 target 或 output 不一致，均为 FAIL/INCONCLUSIVE，不得用 median 掩盖单次峰值。

---

## 12. 分层验收矩阵

### 12.1 L0 host contract

每个 common/runtime PR 必须通过：

```text
selector/config canonical encoding
request conflict/error priority
preset resolver deterministic ordering
SourceLease generation/digest/path replacement
layout compiler golden fixtures
receipt verifier v2
public result verifier
```

### 12.2 L1 executor synthetic

覆盖：

```text
K=1/2/3/8
D=0/1/(K-1)
Q=1/K
serial/retain_all
reload/carry_first_group
delayed/乱序 fill completion
reader fence delayed
cancel at first fill / middle group / pass boundary
mailbox overflow
drain false -> quarantine
wrong-thread destroy rejected
```

门禁：ASan、UBSan、TSan；无死锁、无 slot overwrite、无 unsafe free、receipt 与 trace 一致。

### 12.3 L2 model metadata/adapter host

每个模型必须覆盖：

```text
source closure exact set
index/shard validation
missing/duplicate artifact
short read/overflow
route rejection
source path replacement
same-size content mutation
layout/record/runtime mismatch
```

### 12.4 L3 real-model smoke

每个模型至少一个真实 checkpoint、最小代表性 workload、1–2 次完整请求：检查 output、receipt、drain、cancel 和 source revalidate。L3 只证明接线，不产生性能发布资格。

### 12.5 L4 P0/P1

P0 默认路径：

```text
wall median ratio <= 1.02
wall P95 ratio <= 1.05
output/quality parity
default public hook/lease/pool/worker/receipt allocation = 0
```

P1 同布局 direct vs generic：

```text
wall median <= 1.02
wall P95 <= 1.05
denoise median <= 1.02
same layout/read bytes/output
```

### 12.6 L5 P2/P3

P2 target fit 要求所有 measured request：

```text
tree_peak + margin <= target
无非计划 swap out
terminal sample 存在
max sample gap <= threshold
无未知 child process
```

P3 四臂对照不预设 streaming 胜过 swap；必须报告完整 trade-off。

### 12.7 L6 App/release

覆盖：

```text
Off 默认路径
五档 options/recommendation
unavailable/staging/production state
catalog revoke/revision stale
App 重启恢复
JobStore atomic commit
LTX worker SIGKILL/EOF
单 record rollback
```

---

## 13. 代码实施顺序与停止条件

### C3 · Z-Image 收口

1. 提交 shared lease metadata/reader；
2. request-scoped generate context；
3. actual receipt 接线；
4. full-request GPU smoke；
5. 五档候选 campaign。

停止条件：任何 path reopen、receipt 合成、session 状态污染、VAE 边界 reader 未 drain。

### C4 · Flux 9B

1. 修复显式 include 与 probe index 读取；
2. descriptor/index/shard lease closure；
3. public exact constructor；
4. `Flux::run()` receipt/result 接线；
5. serial/reload baseline；
6. retain-all candidate；
7. P0/P1/P2。

停止条件：metadata 与 pager lease logical id 不一致、public route 进入 direct benchmark executor、receipt 缺少 per-group/fence、VAE/upsample 前未释放 transformer pool。

### C5 · H3 Turbo

1. source closure 与 descriptor；
2. C/Metal reader generation/fence；
3. reload baseline；
4. carry candidate；
5. public receipt/result；
6. default/resident P0。

停止条件：C bridge destroy 无状态返回、reader fence 无法绑定 group、普通 H3/ANE/量化 route 混入 public card。

### C6 · LTX worker

1. JobEnvelope/response schema；
2. worker-local resolve/lease/authority；
3. stage1/upsampler/stage2/VAE component closure；
4. atomic output/JobStore；
5. kill/EOF/quarantine；
6. full target campaign。

停止条件：desktop 持有 worker authority、IPC EOF 写 verified、component boundary 未 drain、任意 child 未纳入 sampler。

### C7 · App

1. options query；
2. advanced settings UI；
3. recommendation/unavailable state；
4. JobStore v2；
5. stale/revoke/restart；
6. default path allocation audit。

### C8/C9 · calibration/release

1. test-only catalog end-to-end；
2. 8/10/12/16/20 GiB candidate search；
3. resident/streaming/bounded/swap 四臂；
4. independent verifier；
5. staging record；
6. review、publish、revoke、rollback drill。

`dev` 合并应在 C4–C6 主要 adapter 接线后进行。冲突处理优先保留 SourceLease/authority/receipt 安全合同，同时吸收 dev 的模型功能修复；合并后重新跑 L0–L4。若 dev 改变 source/runtime/kernel identity，旧 evidence 全部 stale。

---

## 14. 代码审阅清单

### 控制面

- [ ] selector 与 exact preset 没有混用；
- [ ] empty catalog 不会吞掉 request conflict；
- [ ] preflight 与 resolve 使用同一 immutable catalog snapshot；
- [ ] authority 不可复制、不序列化；
- [ ] probe/snapshot/reader 共享同一 lease；
- [ ] public unsupported 不 fallback 到普通 generate；
- [ ] device/source/catalog 在 GPU lock 后再次 revalidate。

### 数据面

- [ ] owner thread 独占 slot state 与 receipt；
- [ ] worker 只发布 bounded POD；
- [ ] reader fence 在 GPU commit 前注册；
- [ ] slot 复用前最后 reader 完成；
- [ ] `D < K`、`Q <= K`、pool barrier 有运行时检查；
- [ ] cancel/fault 停止新 dispatch 并 drain/quarantine；
- [ ] public adapter 不 path reopen。

### 证据面

- [ ] actual receipt 来自 executor，不在 result 层合成；
- [ ] receipt digest、source generation、layout digest 一致；
- [ ] post-drain revalidation 成功后才标 verified；
- [ ] process-tree sampler 有 terminal sample 和 gap 检查；
- [ ] dirty worktree evidence 不可生成 production record；
- [ ] 每个 mandatory test ID 都有输入 hash、命令、结果和失败原因。

### App/发布

- [ ] UI 只暴露 Off/五档；
- [ ] unavailable 不自动 fallback；
- [ ] JobStore 只保存安全 summary，不保存 fd/authority；
- [ ] stale/revoke 不能执行旧 job；
- [ ] output 只有 atomic commit 后才可见为 verified；
- [ ] 单 record 可撤回且 Off/default 不受影响。

---

## 15. 当前状态与完成定义

### 15.1 当前状态（2026-09-18）

| 能力 | 状态 |
|---|---|
| selector schema-v2 / 五档 target | 已有控制面基础 |
| immutable catalog snapshot / authority | 已有并测试 |
| fd-backed SourceLease | 已提交并测试 |
| actual receipt v2 | 已提交并测试 |
| Z-Image public lease adapter | 第一阶段已提交 `9b806f3`，host/synthetic 通过，待真实校准 |
| Flux 9B public adapter | 工作树已通过 native build、descriptor/public host contract、lease pager 与 fd-reader 真实 Metal；full request、receipt/evidence 和校准未完成 |
| H3 Turbo public adapter | 未完成 |
| LTX worker-local public adapter | 未完成 |
| App 五档 UI / JobStore v2 | 未完成 |
| 8/10/12/16/20 GiB full-request calibration | 未完成 |
| resident/streaming/bounded/swap 四臂 | 未完成 |
| production catalog | 为空 |
| ANE streaming | 未认证 |

### 15.2 整体完成定义

只有在以下条件全部满足时，才能说某个模型/档位 public：

1. source closure、workload/runtime/device identity 完整且可复现；
2. public generate 使用 request-scoped lease、exact layout 和 common StageExecutor；
3. 每次成功都有真实 actual receipt、drain 和 post-drain revalidation；
4. 完整请求 process-tree peak + margin 在 target 内；
5. P0/P1/P2 通过，P3 四臂 trade-off 有数据；
6. cancel/fault/source mutation/worker kill 不产生假成功或 unsafe free；
7. App 只暴露目标档位，stale/revoke/rollback 可用；
8. production record 来自独立 verifier 通过的 clean evidence；
9. merge dev 后默认路径和模型路径重新验收；
10. catalog 可按单 record 发布、撤回和回滚。

在此之前，文档和 UI 必须使用“未发布 / unavailable / staging / tentative”等准确状态，不能把 private exact、host PASS 或单次 raw timing 表述为 public 支持。

---

## 16. 文件级实施矩阵与测试 ID

下面的矩阵用于把设计拆成可独立 review 的 PR。路径和函数名以当前 `feat/stream` 为基线；如果 `dev` 合并后移动文件，必须在 PR 描述中保留旧路径到新路径的映射。

| PR | 代码范围 | 主要实现 | 必须新增/更新的测试 | 停止条件 |
|---|---|---|---|---|
| C3.1 | `source_lease.*`, model descriptor | shared lease、duplicate fd、logical id closure | `streaming_source_lease_test`, model descriptor wrappers | 任意 reader path reopen |
| C3.2 | `resolved_request.*`, model session hooks | probe/snapshot/authority exact match | `streaming_preset_resolver_test`, route rejection | authority 可复制或 request 状态泄漏 |
| C3.3 | `context.*`, model exact adapter | receipt enable/finish/drain 接线 | `streaming_actual_receipt_test`, executor integration | receipt 在 result 层合成 |
| C3.4 | Z-Image files | transformer→VAE boundary cleanup | `test_z_image_public_streaming.py`, real smoke | source mutation 未拒绝 |
| C4.1 | Flux descriptor/pager | config/index/shard lease-backed parse | `test_flux_streaming_descriptor.py`, `test_mlx_weight_pager.py` | logical id 或 index authority 不一致 |
| C4.2 | Flux pipeline/exact stream | public constructor、receipt、RunResult | `test_flux_public_streaming.py`（新增） | public route 进入 direct benchmark |
| C4.3 | Flux campaign | serial/reload → retain-all | campaign verifier + P0/P1 | full request 未采样 |
| C5 | H3 session/C bridge/Metal | reload baseline、carry candidate、fence | `test_h3_streaming_descriptor.py`, `test_h3_candidate_streaming_gate.py` | destroy 状态不可证明 |
| C6 | `LTXWorker.swift`, LTX session | worker-local authority、component handoff | lifecycle/EOF/SIGKILL wrappers | desktop 持有 authority |
| C7 | Swift/App/JobStore | options、tier UI、stale/atomic commit | App unit/UI integration | unavailable 静默 fallback |
| C8 | tools/sampler/verifier | inspect/compile/simulate/campaign | `test_streaming_campaign_verifier.py` | sampler gap/unknown child 被忽略 |
| C9 | catalog/release | staging、revoke、rollback | release fixture + clean-tree check | 未签名 evidence 生成 production |

### 16.1 每个模型 adapter 的最小接口

```cpp
std::shared_ptr<const ModelStreamingProbe>
probe_public_streaming(const PublicResolveInput &) const;

std::shared_ptr<const ModelStreamingSnapshot>
compile_public_streaming(
    std::shared_ptr<const ModelStreamingProbe>,
    const StreamingPresetRecord &) const;

RunResult generate_resolved(
    std::shared_ptr<const ResolvedRequestExecution>,
    const Event &, std::atomic<bool> &);
```

probe 只能做 metadata/source capture；compile 只能生成 descriptor/layout snapshot；generate_resolved 才能创建 GPU payload、slot pool、reader 和 receipt。三个接口缺一不可；默认 `ModelSession` 实现必须继续抛出 `streaming_public_adapter_unsupported`。

### 16.2 合并前命令包

common/runtime 或 C ABI 改动：

```bash
env TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
make test-streaming-host
make test-streaming-contract
make test-streaming-audit
python3 -B tests/native/test_streaming_layout.py
python3 -B tests/native/test_streaming_preset_resolver.py
python3 -B tests/native/test_streaming_source_lease.py
python3 -B tests/native/test_streaming_actual_receipt.py
git diff --check
```

executor/receipt/C bridge 改动增加：

```bash
env TC_STREAMING_SANITIZER=address,undefined \
  python3 -B tests/native/test_streaming_layout.py
env TC_STREAMING_SANITIZER=thread \
  python3 -B tests/native/test_streaming_layout.py
```

模型 adapter 改动增加对应 descriptor、candidate gate、public streaming host wrapper；真实模型、process-tree、swap 四臂和 20-pair campaign 不放入普通 host CI，只能在带设备/权重标签的 release gate 执行，并把 evidence digest 回填到进度文档。

### 16.3 失败结果的统一映射

| 事实 | API/UI 状态 | 是否可重试 |
|---|---|---|
| no catalog record | `unavailable/no_record` | 用户换档位或等待发布 |
| source/path mutation | `failed/source_stale` | 重新 resolve 后可重试 |
| target peak 超限 | `failed/target_not_fit` | 选择更高档位 |
| sampler gap/unknown child | `inconclusive/evidence_incomplete` | 重新 campaign |
| reader drain unknown | `quarantined` | 不能复用当前 context |
| user cancel 且 drain 成功 | `cancelled` | 新建 request |
| output/verifier mismatch | `failed/actual_plan_mismatch` | 修复 adapter/record 后重试 |

任何失败映射都不能转为 `verified`，也不能隐式切换到 resident；只有用户明确改选 `Off` 或另一个可用目标，才进入另一条请求路径。
