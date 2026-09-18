# 44 · 下一阶段实现收口规格：Receipt v2、四模型接入、App 事务与低内存发布

修订日期：2026-09-17。状态：可直接施工的设计规格，尚未表示代码已完成或 production catalog 已开放。

本文是对 [38 代码合同](38-framework-code-contracts-and-implementation-workbench.md)、[40 产品化合同](40-public-productization-and-app-contract.md)、[41 调度器实现](41-scheduler-multi-slot-and-multi-pool-implementation.md)、[42 模型施工手册](42-model-adapter-playbooks.md) 的下一阶段收口；逐项验收、数据字段和证据打包见 [45 验收追踪](45-acceptance-traceability-and-evidence-spec.md)。它解决一个具体问题：在不改变 resident/default 热路径的前提下，如何把当前已经提交的 source lease、public resolver、coordinator 和 generic executor 收口成可审计、可验证、可回滚的 public streaming 产品。

## 1. 当前基线与本阶段边界

### 1.1 已存在的代码接缝

当前工作树已经具备以下基础：

| 层 | 当前代码 | 事实边界 |
|---|---|---|
| 请求 | `StreamingSelector`、五档 target、schema v2 校验 | 只表达用户意图，不授予布局权限 |
| 解析 | `PublicStreamingCoordinator`、`PublicPresetResolver` | production catalog 仍为空时 fail-closed |
| 身份 | `SourceLease`、probe/snapshot lease 强校验 | 已有 fd lineage；内容 hash 和完整 artifact registry 仍需补齐 |
| 执行 | `StageExecutor`、`SlotSafetyTracker`、`IoExecutor`、C ABI v1/v2/v3 | private/generic executor 可运行，receipt v2 尚未完成 |
| 结果 | `StreamingRuntimeMetrics`、`PublicStreamingSelectionMetrics` | 只有摘要字段；不能证明每个 pass/group/fence 的实际执行 |
| 模型 | LTX exact private 垂直切片、Flux private candidate、H3/Z-Image descriptor/candidate | 还没有四模型 public hook 完整闭包 |
| App | C API options/resolve 的基础输出 | JobStore 原子事务、正式 Swift/App public 入口尚未完成 |

### 1.2 本阶段只做什么

本阶段包含：

1. `ActualExecutionReceipt` v2：把布局预期与真实执行逐 pass/group/fence 对齐。
2. generic executor 的 recorder 接线，并保持 worker 只发布固定大小 POD completion。
3. C bridge additive receipt API，不破坏既有 v1/v2/v3 plan ABI。
4. 四个模型的 public adapter 逐一接入：Z-Image → Flux 9B → H3 Turbo → LTX worker。
5. App 只暴露 `Off / 8 / 10 / 12 / 16 / 20 GiB`，后台完成 options→resolve→persist→generate 事务。
6. 完整 process-tree 校准、resident/streaming/bounded/swap 四臂对照，以及 catalog builder 的证据门。

本阶段不做：

- 不把任意 manual layout 放入 production catalog；
- 不把 slot 数自动解释成全进程 hard cap；
- 不让运行时因压力偷偷减少 slot、改变 block group、改变 dtype/shape/pass 或切换 pool policy；
- 不把 tiny smoke、合成 fake receipt 或候选 binary 的 raw wall 数据写成 public 性能承诺；
- 不为 H3 普通版、H3 其他量化版或未冻结的变体扩展资格；本阶段 H3 仅为 MiniMax H3 Turbo。

## 2. 非回归原则与执行资格

### 2.1 三条独立的性能承诺

“性能不能差”必须拆为三个可测命题：

| 命题 | 比较 | 目标 |
|---|---|---|
| P0 默认非回归 | dev/resident 对当前 feat/stream resident | median wall ≤ 1.02，P95 ≤ 1.05，质量一致 |
| P1 框架成本 | 同一 source/layout 下 direct exact 对 generic executor | median wall ≤ 1.02，P95 ≤ 1.05，denoise ≤ 1.02 |
| P2/P3 低内存收益 | public streaming 对 bounded/swap pressure | 必须真实测量；不预设“一定更快” |

P2/P3 的收益不能抵销 P0。改变 P/G/K/D/Q 带来的收益属于策略实验，不得冒充 generic framework overhead。

### 2.2 public 资格的必要条件

一个 record 只有在下列条件全部满足时，才允许 `release.channel=production`：

```text
catalog snapshot immutable
source lease capture + pre-GPU/post-drain revalidate
probe identity == snapshot identity == record identity
layout digest exact
actual receipt v2 complete and independently verified
full request process-tree peak calibrated
P0/P1 non-regression passed
target-memory samples all fit with H(T) margin
cancel/fault/quarantine passed
model quality/output parity passed
App transaction and stale-resolution tests passed
reviewed evidence bundle and signed catalog record
```

任一项缺失时，状态只能是 `candidate`、`unavailable` 或 `INCONCLUSIVE`。

## 3. C2：Actual Receipt v2 的代码设计

### 3.1 文件与依赖

新增：

```text
native/runtime/streaming/actual_receipt.hpp
native/runtime/streaming/actual_receipt.cpp
tests/native/streaming_actual_receipt_test.cpp
tests/native/test_streaming_actual_receipt.py
```

修改：

```text
native/runtime/streaming/context.hpp/.cpp
native/runtime/streaming/slot_pool.hpp/.cpp
native/runtime/streaming/io_executor.hpp/.cpp   # 仅必要的 completion 字段检查
native/runtime/streaming/c_bridge.cpp
native/core/stream_slot_c.h                     # additive receipt ABI
native/runtime/streaming/public_result.cpp
native/runtime/session.hpp                       # result 摘要字段
native/platform/apple/results.mm                 # JSON 摘要
tools/native/build.sh / Makefile                 # source/test 接入
```

`actual_receipt.*` 只依赖 `layout.hpp`、`stream_slot_c.h` 和 canonical encoder；不能依赖 Metal、MLX、Foundation 或具体模型，以保证 host-only 测试可编译。

### 3.2 数据结构

建议采用以下内部结构。所有动态容器在 `begin()` 前按 sealed layout 一次性分配，worker 线程永远不触碰容器扩容。

```cpp
struct ActualFillReceipt final {
    uint32_t pass = 0;
    uint32_t group = 0;
    uint32_t pool = 0;
    uint32_t slot = 0;
    uint64_t ticket_generation = 0;
    uint64_t logical_bytes = 0;
    uint64_t source_generation = 0;
    int32_t status = 0;
};

struct ActualGroupReceipt final {
    uint32_t pass = 0;
    uint32_t group = 0;
    uint32_t pool = 0;
    uint32_t slot = 0;
    uint32_t fill_count = 0;
    uint64_t expected_bytes = 0;
    uint64_t actual_bytes = 0;
    uint64_t reader_issued = 0;
    uint64_t reader_completed = 0;
    uint64_t first_enqueue_ns = 0;
    uint64_t ready_ns = 0;
    uint64_t claim_ns = 0;
    uint64_t submit_ns = 0;
};

struct ActualStageReceipt final {
    std::string stage_id;
    std::string layout_digest;
    std::string implementation;
    uint32_t completed_passes = 0;
    uint32_t completed_groups = 0;
    uint64_t fills = 0;
    uint64_t groups_submitted = 0;
    uint64_t logical_read_bytes = 0;
    uint64_t reader_fences_issued = 0;
    uint64_t reader_fences_completed = 0;
    uint64_t source_generation = 0;
    bool drain_completed = false;
    std::vector<ActualGroupReceipt> groups;
    std::vector<std::pair<uint32_t, uint32_t>> pool_barriers;
    std::string event_digest;
};

struct ActualExecutionReceipt final {
    uint32_t schema_version = 2;
    std::string implementation;
    std::string layout_digest;
    std::string component_policy_revision;
    std::vector<ActualStageReceipt> stages;
    std::string canonical_digest;
};
```

生产 result 不序列化 `groups` 的全部时间戳；时间戳只进入 evidence bundle。App 结果只拿 schema、digest、计数和 verifier revision，避免暴露内部 pool/slot 实现细节。

### 3.3 Expected matrix 的索引规则

`begin()` 依据 `pass_count × group_count` 建立确定性矩阵：

```text
index(pass, group) = pass * group_count + group
```

矩阵每格必须记录：

- expected pool、expected block list、expected logical bytes；
- 是否为 carry-first-group；
- 允许的 slot 集合（通常为 executor rotation 计算出的单一 slot）；
- expected source generation。

不能通过“只比较总 fills”放宽验证。一个 group 被错误 pool/slot 填充，即使总字节数正确，也必须拒绝 public success。

### 3.4 事件顺序与线程所有权

```text
owner: begin -> fill_submitted -> completion consume -> ready
owner: begin_use -> readers_issued -> reader completion consume
owner: pool barrier -> pass complete -> carry transition
owner: shutdown/join -> adapter drain -> source post-drain revalidate
owner: finish -> canonical digest -> public verifier
```

worker 只执行：

1. 读取 `FillJob` 中已经冻结的 fd/span；
2. 在 chunk 边界检查 cancel；
3. 返回固定大小 `tc_stream_completion_v1`。

worker 禁止写 receipt vector、计算 SHA-256、分配字符串、调用 MLX/Metal global synchronize 或修改 slot state。`CompletionMailbox` overflow 是 sticky failure，必须进入 quarantine。

### 3.5 Digest 规范

`event_digest` 使用 canonical encoder，字段顺序固定：

```text
schema_version
stage_id
layout_digest
pass,group,pool,slot
fill_count,expected_bytes,actual_bytes
reader_issued,reader_completed
carry_flag,pool_barrier_index
source_generation
```

禁止把绝对时间、线程 id、指针、fd、path 或随机数放入 digest。时间戳只用于性能诊断，不影响语义 digest。`canonical_digest` 再对 `event_digest`、summary、component policy 和 implementation revision 编码。

### 3.6 失败优先级

验证器按以下顺序返回第一个稳定错误码：

```text
receipt_schema_mismatch
receipt_stage_mismatch
receipt_layout_mismatch
receipt_group_missing
receipt_group_duplicate
receipt_pool_slot_mismatch
receipt_logical_bytes_mismatch
receipt_source_generation_mismatch
receipt_fence_incomplete
receipt_carry_mismatch
receipt_drain_incomplete
receipt_digest_mismatch
```

错误返回前必须完成安全 cleanup；如果 GPU/reader 是否仍引用 backing 无法证明，则 `streaming_quarantined=true`，禁止同一 engine 再次 generate。

### 3.7 C ABI additive 接口

不能修改已有 plan struct 的字段布局。新增：

```c
#define TC_STREAM_RECEIPT_ABI_V1 1u

typedef struct {
    uint32_t struct_size;
    uint32_t version;
    uint64_t source_generation;
    char layout_digest[65];
    char implementation[64];
} tc_stream_receipt_config_v1;

typedef struct {
    uint32_t struct_size;
    uint32_t version;
    uint32_t completed_passes;
    uint32_t completed_groups;
    uint64_t fills;
    uint64_t groups_submitted;
    uint64_t logical_read_bytes;
    uint64_t reader_fences_issued;
    uint64_t reader_fences_completed;
    uint64_t source_generation;
    uint8_t drained;
    char event_digest[65];
    char canonical_digest[65];
} tc_stream_receipt_v1;

int tc_stream_executor_enable_receipt_v1(
    tc_stream_executor *, const tc_stream_receipt_config_v1 *,
    char *, size_t);
int tc_stream_executor_receipt_v1(
    tc_stream_executor *, tc_stream_receipt_v1 *, char *, size_t);
```

`enable` 只能在 create 成功、首个 `run_pass` 之前调用；`receipt` 只能在 `finish` 成功后调用。C ABI 不拥有字符串内存，也不暴露内部矩阵。

### 3.8 Receipt 测试矩阵

至少加入：

| ID | 故障 | 预期 |
|---|---|---|
| RCP-001 | zero fill | reject，不能 serialize success |
| RCP-002 | duplicate fill | reject |
| RCP-003 | logical bytes 少一段 | reject |
| RCP-004 | wrong pool/slot | reject |
| RCP-005 | issued fence 多于 completed | reject + quarantine |
| RCP-006 | source generation 变化 | reject |
| RCP-007 | carry rotation 错 | reject |
| RCP-008 | mailbox overflow | reject + quarantine |
| RCP-009 | `drain=false` | reject + quarantine |
| RCP-010 | event digest 被改写 | reject |
| RCP-011 | C ABI struct_size 太小 | reject，不越界读 |
| RCP-012 | receipt 在 finish 前读取 | reject |

## 4. Public generate 事务与代码接线

### 4.1 状态机

```text
Received
  -> Parsed
  -> Preflighted(catalog snapshot)
  -> Planned(normalized request)
  -> Probed(source lease + workload identity)
  -> Snapshotted(descriptor/layout)
  -> Authorized(exact selector + authority)
  -> Locked(global GPU lock)
  -> Revalidated(path/fd/catalog/device)
  -> Executing(StageExecutor/adapters)
  -> Drained(reader/worker/source)
  -> Verified(receipt v2 + quality)
  -> Committed(result/job evidence)
```

可失败节点：

- `Preflighted` 之前失败：无 engine quarantine；
- `Probed/Snapshotted` 失败：释放 lease，不能创建 GPU backing；
- `Revalidated` 失败：普通 stale/source-changed 错误，不执行 GPU；
- `Executing` 中 cancel：协作取消，安全 drain 后返回 cancelled；
- `Executing` 中 callback/mailbox/fence 不确定：quarantine；
- `Verified` 失败：不得产生 public success output，必要时删除临时导出物。

### 4.2 `c_api.mm` 目标伪代码

```cpp
int tc_engine_generate(...) {
    // 1. Existing mutex and cancellation checks.
    auto parsed = request_from_json(parse_json(r));

    std::shared_ptr<const ResolvedRequestExecution> resolved;
    if (parsed.streaming_selector && parsed.streaming_selector->active()) {
        PublicStreamingCoordinator coordinator(...);
        auto preflight = coordinator.preflight(parsed);       // no GPU
        auto normalized = make_plan_after_public_streaming_preflight(parsed);
        resolved = coordinator.resolve_normalized(
            std::move(normalized.request), device_identity(),
            std::move(preflight));                             // metadata only
        parsed = resolved->request;
    }

    std::unique_lock global(execution_mutex(), std::try_to_lock);
    require(global.owns_lock(), "native GPU runtime busy");
    DeviceLease device;
    if (resolved) coordinator.revalidate(*resolved, device_identity());

    RunResult result = resolved
        ? session->generate_resolved(resolved, event, cancelled)
        : session->generate(parsed, event, cancelled);

    if (resolved)
        verify_and_attach_public_streaming_result(*resolved, result);
    serialize_result(result);
}
```

关键点：

1. public adapter 不支持时返回稳定 `streaming_public_adapter_unsupported`，不能 fallback 到普通 `generate()`。
2. `PublicStreamingCoordinator` 不获取 global GPU lock；lock 由 API 层统一管理。
3. `Request` 在 resolve 后只保留 canonical `StreamingConfig`，清除 selector，防止 adapter 再次自行解析。
4. receipt verifier 必须在 source post-drain revalidate 之后执行。

### 4.3 默认路径零成本证明

默认 `Off` 或已有 resident 路径必须满足：

```text
catalog snapshot calls = 0
source lease capture/open/fstat/hash = 0
probe/snapshot calls = 0
receipt recorder allocations = 0
new worker threads = 0
new pool allocations = 0
memory guard lookups = 0
```

通过 `AuditCounter`、编译期 symbol review 和 P0 ABBA campaign 三重证明。不得只通过一次 wall benchmark 推测零成本。

## 5. 四模型 public adapter 施工顺序

### 5.1 Z-Image Turbo：首个 public gate

冻结第一张 public card：512×512、原始 BF16 GPU、single pool、无 LoRA、无 ANE、无 GGUF/compiled graph。adapter 实现：

```text
ZImage::probe_public_streaming()
  -> SourceLease::capture(shard files)
  -> metadata/index/workload identity

ZImage::compile_public_streaming()
  -> descriptor + LayoutCompiler
  -> exact record layout digest
  -> ValueModelStreamingSnapshot(shared lease)

ZImage::generate_resolved()
  -> create fixed pool/backing
  -> StageExecutor with receipt recorder
  -> bind block tensors by storage_id
  -> complete denoise + VAE/export
  -> drain + post-drain lease check
```

必须补：shard 顺序稳定性、same-size mutation、short read、cancel、output SHA/quality、完整 process-tree sampler。

### 5.2 Flux.2 Klein 9B：dual/single 异构 pool

冻结结构：8 个 dual block、24 个 single block、两个 retained K2 pool、Q2 refill。adapter 必须显式声明 `supports_multi_pool_policy(retain_all)=true`，否则 resolver 只能选择 serial。

实现注意：

- dual 与 single 的 `storage_id` 命名空间必须独立；
- class barrier 前等待上一 pool reader 全部完成；
- retain-all 的 calibrated bytes 必须包含两个 pool 的总 capacity；
- concat/binding 不得依赖“上一次 pool 的地址仍然相同”；
- Flux 4B compiled graph 保持原默认路径，不进入 9B public record。

### 5.3 MiniMax H3 Turbo：carry-first-group

范围只覆盖 `minimax-h3-turbo` original BF16 native Metal。每个 pass 仍对每个 suffix group 产生一次 logical fill，carry 只改变时序：

```text
pass N: ... encode(last group)
pass N+1: fill(first group) 已在上一个 pass 尾部发起并 Ready
         -> claim(first group)
```

receipt 必须记录 carry 的 pass/group/slot；不能以少一次 fill 的方式实现“优化”。额外验收 active block、cross-block fusion、跨 forward prefetch 和 ordinary H3 默认回归。

### 5.4 LTX 2.5：worker-local authority

LTX 的 parent process 不能把 fd、`ModelSession*`、Metal object 或 authority pointer 发送给 worker。推荐协议：

```text
parent: parse + validate user request
parent: create serializable JobEnvelope v2
worker: create model session
worker: open/capture SourceLease
worker: probe -> snapshot -> exact resolve
worker: revalidate under worker GPU lock
worker: execute denoiser/upsampler/VAE/audio/export
worker: receipt + source post-drain revalidate
worker: return result/evidence summary
parent: commit JobStore only after verified response
```

worker 被 SIGTERM/SIGKILL 或 pipe 断开时，parent 将 job 标记 `quarantined`/`retryable`，不复用旧 engine。必须覆盖视频帧数、音频、upsampler、VAE、export 以及 finalizer 的资源释放。

## 6. App、Swift 与 JobStore 事务

### 6.1 用户配置与内部配置分离

用户可见：

```json
{
  "streaming": {
    "enabled": true,
    "target": "12GiB"
  }
}
```

内部解析后才生成：

```json
{
  "schema_version": 2,
  "selection": "memory_tier",
  "target_request_memory_bytes": 12884901888,
  "preset_id": "z-image-turbo.p1.12gib.r1",
  "preset_revision": 1,
    "catalog_revision": "catalog-2026-09-17-a",
  "resolution_digest": "sha256:...",
  "layout_digest": "sha256:..."
}
```

UI 不显示 `slot_count`、`block_group_size`、`prefetch_distance`、`io_workers`、`layout_digest`，诊断页可显示 record id 和状态。

### 6.2 Options API 的语义

`tc_streaming_options_json` 只能回答“当前 model/workload 在每个 target 是否有 reviewed candidate”，不能授予执行资格。options key 必须包含：

```text
model id + model variant + operation + execution backend
width/height/frames/fps/steps/audio/dynamic_text
LoRA identities and strengths
ANE/compiled/approximation flags
source artifact identity if available
catalog revision
```

上述任一字段变化，Swift cache 必须失效。结果为 `available` 才能让 UI 勾选；`unavailable` 必须带稳定 reason code。

### 6.3 JobStore v2 原子提交

Job 状态：

```text
draft -> resolving -> resolved -> running -> verified -> committed
                         \-> stale
running -----------------> cancelled
running -----------------> quarantined
```

事务规则：

1. `resolve` 成功前不能写入可运行 job；
2. 保存 selector、exact selector、record/revision/catalog/resolution digest 和 workload digest；
3. 不保存 fd、inode、canonical path、GPU pointer、worker PID 或 authority object；
4. generate 前重新 resolve/revalidate；catalog revoke、source mutation 或 build mismatch 只允许变为 `stale`，不能静默执行旧布局；
5. result receipt verifier 成功后才写 `verified/committed`。

## 7. 低内存校准与配置文件

### 7.1 统一 profile 文件

建议 profile schema：

```json
{
  "schema_version": 2,
  "profile_id": "flux9b-macos-gpu-12gib-v1",
  "model": "flux2-klein-9b",
  "workload": {
    "operation": "text_to_image",
    "width": 1024,
    "height": 1024,
    "steps": 20,
    "audio": false,
    "backend": "gpu_bf16"
  },
  "target": {"request_memory_bytes": 12884901888},
  "layout": {
    "stages": {
      "denoiser": {
        "residency": "streamed",
        "block_group_size": 1,
        "slot_count": 2,
        "resident_prefix_blocks": 8,
        "prefetch_distance": 1,
        "io_workers": 2
      }
    }
  },
  "policy": {
    "multi_pool": "retain_all",
    "pass_transition": "reload",
    "retention": "request"
  },
  "calibration": {
    "scope": "process_tree",
    "headroom_bytes": 536870912,
    "estimator_revision": "peak-v3"
  }
}
```

profile 不是用户输入，也不是生产 catalog 本身。catalog builder 必须从 profile、完整 evidence 和 review 重新生成 canonical record。

### 7.2 target 判定

对于目标 `T`，定义：

```text
H(T) = max(512 MiB, ceil(0.10 × T))
eligible = every measured sample complete
         && tree_peak + H(T) <= T
         && no unexpected swapout/failure
```

`tree_peak` 是请求进程树在同一时间点的进程 footprint 之和，不是各进程独立 peak 的相加。采样最大间隔、漏采样和 process exit 事件都必须进入 evidence；无法证明完整覆盖则 `INCONCLUSIVE`。

### 7.3 用户 target、buffer 与执行预算的关系

用户看到的 `8/10/12/16/20 GiB` 是“整个请求允许占用的 process-tree envelope”，不是 denoiser slot 的 GPU hard cap，也不是允许写入系统 swap 的额度。内部将 target 和 buffer 分开保存：

```text
T              = target_request_memory_bytes
b              = reviewed buffer ratio（首版默认 10%，不由普通 UI 编辑）
M              = max(512 MiB, ceil(b × T))
usable_ceiling = T - M
```

校准与执行必须满足：

```text
tree_peak + M <= T
process_baseline + component_reserve + denoiser_budget <= usable_ceiling
```

实现要求：

1. `physical_memory_bytes` 只用于 resolver 的设备资格和 target 可用性判断，不当作瞬时 free-memory；
2. `target_request_memory_bytes`、`buffer_ratio`、`headroom_revision` 都进入 calibration id 和 record digest；
3. 运行中不根据 `free` 或 pressure 动态减少 slot、改变 block group 或切换 dtype。若 clean baseline 已经超过 `usable_ceiling`，在创建 GPU backing 前返回 `memory_budget_too_small`；
4. swap/compression 计数是验收指标，不属于 target envelope 的“可用余量”；发生不可预期 swapout 时 sample 失败或 `INCONCLUSIVE`；
5. 目标档位之间不能线性推导布局或性能。每个 target 必须有独立 reviewed record，除非同一 record 明确声明其 device/workload range。

这一定义保留了“用户只选内存档位”的产品体验，同时避免把 slot 数量错误地当成全进程内存上限。

### 7.4 参数探索顺序

固定 workload 后按以下顺序搜索，避免组合爆炸：

```text
1. K = 1,2,3；保持 G=1、D=min(K-1,2)、Q=K
2. 调整 G=2,4，只在单组 capacity 已明显大于 activation 时尝试
3. 调整 prefix，优先增加能减少重复读取的稳定前缀
4. 调整 Q/D，观察 fill wait 与 reader wait 的重叠
5. Flux 只在完成 dual/single pool 后比较 serial/retain_all
6. H3 只在 carry receipt 正确后比较 reload/carry
```

每个候选必须生成 simulator 结果；只有 simulator 不超预算、无死锁且有足够 overlap 的候选才进入真实测量。

## 8. 分阶段提交与回滚点

| 阶段 | 代码范围 | 必须通过 | 回滚点 |
|---|---|---|---|
| C2 | receipt v2、C ABI、verifier | RCP-001…012、sanitizer、Off audit | 恢复 recorder=null 的 executor |
| C3 | Z-Image public hook | source/receipt/full output/P0/P1 | catalog 仍为空，保留 candidate |
| C4 | Flux 9B dual/single | pool barrier、retain-all memory、VAE | 只撤销 Flux record |
| C5 | H3 Turbo carry | carry receipt、普通 H3 regression | 只撤销 H3 record |
| C6 | LTX worker | worker kill/quarantine、全组件输出 | 回退 private exact worker |
| C7 | App/JobStore | options、stale、原子 commit | UI 隐藏 streaming 入口 |
| C8 | 校准/catalog | 完整 evidence、review、builder | production catalog 保持空 |

任何阶段发生 P0 回归、source lease 漏洞、receipt mismatch 可绕过、quarantine 泄漏或默认路径新增 allocation，都必须停止后续阶段。

## 9. 实现完成定义

本文件对应的“实施规格完成”只表示代码文件、函数边界、状态机、失败语义、测试 ID、配置和回滚路径已经冻结。真正完成必须另有证据：

- C2 receipt v2 代码与故障矩阵已实现；
- 至少一个模型完成完整 public adapter 和真实 receipt；
- 四模型各自的 source closure、完整输出和 process-tree 峰值均有记录；
- 8/10/12/16/20 GiB 的 reviewed record 只从真实测量生成；
- resident/default P0 和 same-layout P1 不回退；
- swap 对照结论由统计证据决定；
- production catalog 非空前，App 只能显示 `unavailable` 或 private candidate 状态。
