# 27 · Public Streaming 端到端施工蓝图与验收追踪

[目录](README.md) · [产品语义](23-public-memory-tier-presets.md) · [候选探索](24-memory-tier-exploration-and-acceptance.md) · [代码规格](25-public-preset-implementation-spec.md) · [发布验收](26-public-preset-acceptance-and-release.md) · [代码落地](28-public-runtime-code-design.md) · [实施计划](29-public-implementation-and-acceptance-plan.md)

日期：2026-09-17。分支：`feat/stream`。设计前置提交：`a76c414`；控制面及本文提交：`63b73d9`。

状态：**实施蓝图及已提交控制面审阅记录**。本文不新增另一套配置、scheduler 或验收等级；23 定义产品语义，24/26
定义实验和发布资格，25 定义接口原则。本文负责把这些要求落实成可按 PR 执行的代码改造、所有权、状态机、测试与交接清单。

当前 production preset catalog 仍为空，App 尚未公开该开关，普通 engine 尚不能执行 public selector。
文中所有 `illustrative.*` ID、JSON 数值和伪代码都不是已发布 record。

## 1. 最终交付形态

Public streaming 的最终形态不是“把 P/G/K/D/Q 暴露给用户”，而是一个三层系统：

```text
产品层
  App 高级开关 + 生成内存目标 + 可用性/原因/证据展示
                     |
控制面
  selector parser → workload/source/device identity → catalog resolver
                  → exact preset → immutable resolved execution + authority
                     |
数据面
  existing descriptor → LayoutCompiler → StageExecutor/SlotPool/Pager
                      → existing model kernels/readers/fences
```

需要同时保证：

1. streaming 关闭时仍走原有 `NativeRequest`、原 engine/session、原 resident/legacy-streamed 路径。
2. selector 只表达用户意图，不直接授予执行资格，也不允许用户伪造 `authorized=true`。
3. catalog 只选择经过完整证据审核的 exact preset，不在运行时搜索任意布局。
4. resolved plan 在一次请求中不可变；adapter 不得临时缩槽、换 prefix、换 backend 或回退 resident。
5. slot/block 数据面继续复用当前框架；public 功能只增加控制面和模型接线。
6. 8/10/12/16/20 GiB 是经验校准目标，不是 hard cap、零 swap 保证或设备物理内存规格。
7. 首版 public route 只做 GPU；GPU+ANE 保持原路径和性能，不因 public streaming 合并而被隐式改变。

## 2. 当前代码事实与完成边界

截至本文审阅时，`63b73d9` 已包含以下实现。它们已通过 `tools/native/build.sh` 的 native 与 Swift/App 编译，
并通过 `make test-streaming-contract`、`make test-streaming-host`；这只能证明控制面基础和既有框架回归，不能证明 public
模型可执行、内存档位成立或 App 已可用。

| 能力 | 当前实现 | 当前结论 |
|---|---|---|
| selector 合同 | `StreamingSelector`、schema v2、8/10/12/16/20 GiB target | duplicate-key raw scanner已有；仍缺target lexical integer严格限制 |
| request/profile 合并 | selector/manual 原子替换、provenance、legacy 冲突检查 | 已有基础；继续补 profile/disabled/golden 边界 |
| plan-only 报告 | active selector 返回 `streaming_preset_resolution_required` | 已实现；正确地不授予执行资格 |
| catalog/resolver | `preset_catalog.*`、整数 margin、排序、release/revoke/device/workload 过滤 | 仅 host skeleton；record identity 还不足以发布真实模型 |
| production catalog | `tc-streaming-catalog-empty-v1` | 正确 fail-closed；没有任何 public record |
| options C ABI | `tc_streaming_options_json` 返回五个 target 的 tentative 状态 | metadata-only 基础已存在；还没有 artifact/token exact resolve |
| Swift binding | `NativeRequestV2`、selector/options 类型和 overload | 可编译的第一版；还不是完整 job/request envelope |
| App state/UI | 无 `StreamingChoice`、无 picker/query state | 未实施 |
| engine exact resolve | 无 `tc_engine_resolve_streaming_json`、authority、resolution digest | 未实施 |
| generate public route | 无 `generate_resolved` 和 public session owner | unresolved selector 的generate/prepare早拒绝已实现；exact执行未实施 |
| 四模型 public adapter | 只有 private/manual exact adapter 和性能锚点 | 未接 public authority；不能称 public 支持 |
| memory calibration | 无 fresh-process tree sampler/verifier/catalog builder | 未实施 |
| reviewed records | 无 full-request、device/container 认证记录 | 未实施 |

当前测试结果：

- request contract 12 项通过，四模型普通 public gate 仍 fail-closed；
- layout compiler 3535 个有效布局、14 组 K/D/Q、multi-class/fault/cleanup 通过；
- preset resolver 的整数余量、确定性排序、exact replay、device/calibration/revocation fail-closed 通过；
- LTX、H3 Turbo、Z-Image、Flux 9B descriptor host tests 通过；
- 没有在本轮执行 GPU sweep、系统压力、swap 对照或 public full-pipeline campaign。

可用于实施规划的既有 private/same-layout 锚点如下；它们不是 public memory-tier 认证：

| 模型/冻结tuple | 现有结论 | 不能外推到 |
|---|---|---|
| LTX P8/G1/K3/D2/Q3 | generic/legacy wall `0.98810`，denoise `0.97249` | normal workload、任意target、App public |
| Z-Image P14/G1/K2/D0/Q1 | wall `0.99986`，denoise `1.00313` | 9-step/1024²整请求、旧budget迁移 |
| H3 Turbo P0/G1/K2/D1/Q1 | wall `1.01066`，denoise `1.01185`；工程接受，统计仍INCONCLUSIVE | full video memory-tier、普通H3 |
| Flux9 P0/G1/K2/D1/Q2 | generic/direct wall `0.99971`，denoise `0.99816` | resident等速、P>0、任意target |
| Flux9 resident→streaming | MLX peak `18,303,578,036 → 11,693,804,356` bytes；wall `1.338198 → 1.951123s` | 进程总峰值、swap优势、其他shape |

这些数字的用途是保护现有 tuple 和估计实验成本；catalog 只能引用后续 full-request、同scope、独立确认的数据。

## 3. 已完成第一优先级：消除 selector 的潜在静默回退

原 `tc_engine_generate` 只检查 `request.streaming.active()`，未检查 active `request.streaming_selector`；
`preparation_call` 也只拒绝 manual streaming。当前工作树已经修复这一问题：解析发生在全局GPU锁和DeviceLease之前，
active selector 在任何普通session调用前统一拒绝。

当前临时 fail-closed gate 等价于：

```cpp
static bool has_active_public_selector(const tc::Request &request) {
    return request.streaming_selector && request.streaming_selector->active();
}

tc::require(!has_active_public_selector(request),
            "streaming_preset_resolution_required: "
            "public selector must be resolved by the engine before execution");
```

当前覆盖：

- `tc_engine_generate`；
- `tc_engine_prepare`；
- 现有App/CLI/service最终使用的同一C API入口；后续新增worker public入口仍须复用该规则；
- cache/load 入口若不接收 request，不得假装为 selector prepare；
- selector disabled 继续与 absent 相同，不触发该拒绝。

新增contract使用普通 public constructor和private candidate active selector，已断言：

1. 返回稳定错误码；
2. 普通engine和candidate的`generate/prepare`均不能绕过；
3. 源码gate位于全局GPU锁、DeviceLease和session调用之前；
4. 返回resolution-required而不是resident、legacy-streamed或candidate路线错误；
5. disabled selector 的plan仍与absent相同。

后续实现 resolver 后，只能把这条拒绝替换为显式的 `resolve → authority → generate_resolved` 分支，不能直接删除。

## 4. 模块边界与依赖方向

建议最终目录职责如下：

```text
native/core/
  streaming_contracts.hpp          wire intent、manual config；无模型/GPU依赖

native/runtime/streaming/
  preset_catalog.*                 immutable records/index/revocation
  preset_resolver.*                exact filtering/ranking/digest/authority
  resolved_request.*               snapshot ownership and immutable execution
  layout.*                         existing compiler
  slot_pool.* / io_executor.*      existing bounded data plane
  mlx_weight_pager.*               existing MLX backing/pread implementation

native/platform/apple/
  request.mm/profile.mm            parsing/merge only
  *_streaming_descriptor.*         model metadata projection only
  *_session.mm                     consume resolved authority and execute
  results.mm                       additive reporting

bindings/
  c/include/turbocider.h           additive public ABI
  swift/TurboCiderNative.swift     v1 preserved; v2 explicit wire types

apps/macos/
  StudioState.swift                user choice and validation
  StreamingOptions.swift           async metadata query state/cache
  JobStore.swift                   frozen payload/resolve/execute transaction
  RunInsights*.swift               target/preset/scope/actual layout display

tools/native/
  inspect/calibrate/explore/verify/build catalog tools
```

依赖必须单向：

```text
App → Swift ABI → C API → resolver/catalog → descriptor/compiler → session/adapter
                                                               → executor/pager
```

禁止反向依赖：

- executor 不读取 App target 或 catalog；
- adapter 不根据空闲内存选择 P/K；
- catalog 不包含可执行代码、动态库路径或 callback；
- App 不复制模型 block 数、P 上界或合法 K/D/Q 规则；
- `make_plan` 不读取 engine root、模型文件或授予 public authority；
- benchmark 工具不能直接把本地 JSON 注入 production authority。

## 5. 控制面类型与所有权

### 5.1 Intent、record、resolution、authority 四类对象

必须分成四层，不用一个含大量 optional 字段的对象贯穿全链路：

```cpp
struct StreamingSelectorIntent {
    SelectionKind selection;              // memory_tier | preset
    uint64_t target_request_memory_bytes;
    std::optional<ExactPresetSelector> exact;
};

struct StreamingPresetRecord {
    PresetIdentity preset;
    Applicability applicability;
    CanonicalStreamingPlan plan;
    CalibrationEvidence calibration;
    PerformanceEvidence performance;
    ReleasePolicy release;
};

struct ResolvedStreamingSelection {
    PresetIdentity preset;
    CanonicalStreamingPlan plan;
    WorkloadIdentity workload;
    SourceIdentity source;
    DeviceIdentity device;
    std::string resolution_digest;
};

class StreamingAuthority {
    friend class PublicPresetResolver;
    // private constructor; never decoded from JSON
};
```

语义：

- selector 可以持久化和跨进程传输，但没有权限；
- record 是 build 内只读数据，不能由请求提供；
- resolution 是 native 对当前 artifact/workload/device 的确定结果；
- authority 是进程内能力对象，只能由 resolver 构造并由 session 验证。

### 5.2 Resolved request 的生命周期

建议 `ResolvedRequestExecution` 持有：

```cpp
struct ResolvedRequestExecution {
    ExecutionPlan model_plan;
    std::shared_ptr<const ModelSourceSnapshot> source;
    std::shared_ptr<const streaming::Descriptor> descriptor;
    std::shared_ptr<const streaming::Layout> layout;
    ResolvedStreamingSelection selection;
    std::shared_ptr<const StreamingAuthority> authority;
};
```

一次 generate 再创建 request-scoped owner：

```text
ResolvedRequestExecution（只读，可由一次resolve返回）
  └─ RequestExecutionOwner（每次运行独立）
       ├─ cancellation generation
       ├─ adapter/pager
       ├─ slot pools and tickets
       ├─ I/O workers/mailbox
       ├─ reader/fence completions
       └─ terminal report/quarantine state
```

不可跨请求复用 ticket、slot state、cancel flag、mailbox 或 generation。若 P/K/source/layout/component revision 改变，
旧 resolved snapshot 立即失效。owner 只有在停止 dispatch、join worker、drain reader/callback 后才能释放 source/layout/authority。

### 5.3 Digest 边界

至少保留以下独立 digest：

| Digest | 绑定内容 | 不负责 |
|---|---|---|
| `artifact_digest` | 已验证权重/manifest 内容 | 当前打开文件仍未变化 |
| `source_snapshot_digest` | 文件集合、索引、size/mtime/inode/open identity | 恶意同 identity 原地篡改的密码学证明 |
| `workload_digest` | shape/token/backend/feature semantics | seed/prompt 隐私持久化策略 |
| `layout_digest` | descriptor + P/G/K/D/Q/pool/pass/retention | component live order |
| `component_policy_digest` | text/denoise/VAE/export 生命周期 | slot ticket 状态 |
| `resolution_digest` | 上述 identity + record/release/build | bearer token或外部授权 |

任何一项变化都不应由 resolver“找一个差不多的 preset”继续执行 exact replay。

## 6. 请求与状态机

### 6.1 三条请求路径

```text
Absent / v2 disabled
  → legacy/default request
  → no catalog, no resolver, no new sampler

Manual v1
  → existing exact layout parser
  → private/candidate gate
  → current adapter/compiler/executor

Public selector v2 active
  → syntactic validation
  → options（可选、tentative）
  → engine exact resolve
  → freeze exact selector + digest
  → generate_resolved
```

manual 和 public selector 互斥。public selector 与 legacy `residency`、`memory_budget_bytes`、`streaming_offload`、
未认证 `memory_constrained` 组合时早拒绝，不能把几个策略拼成未测试的新路线。

### 6.2 Profile 合并

合并规则继续采用完整对象替换：

| Profile | Request | Effective |
|---|---|---|
| 无 | 无 | off/default |
| selector on | 无 | profile selector |
| selector on | selector off | off；不继承 profile 的 target |
| selector on | selector on | request selector 整体替换 |
| manual | manual partial | 现有 stage-aware overlay |
| manual | selector | selector 整体替换 manual |
| selector | manual | manual 整体替换 selector |

合并后再检查 legacy residency/budget/offload/guard 冲突。provenance 只用于诊断，不进入 resolution identity。

### 6.3 运行状态机

```text
Draft
  → OptionsTentative
  → Resolving
  → Resolved
  → Persisted
  → Admitted
  → Running
  → Draining
  → Succeeded
       or Cancelled / Failed / Quarantined / Stale
```

关键约束：

- options 不是 resolve；
- resolve 成功后 jobs 持久化失败则不得运行；
- 排队到执行前重新检查撤回/source/build identity；
- stale 不自动重选，必须回到 Resolving 并让用户看到变化；
- failure 不自动 resident retry，否则可能越过用户目标并污染峰值；
- quarantine 后同 engine 的任何 generate/prepare 明确拒绝并要求重建。

## 7. Public API 设计

### 7.1 Options：便宜、只读、可 tentative

当前 `tc_streaming_options_json` 可作为第一阶段接口，但最终 query 至少需要：

```json
{
  "schema_version": 1,
  "request": { "schema_version": 2, "model": "...", "execution": {} },
  "artifact_hint": {
    "model_root": "logical-local-reference",
    "verified_manifest_digest": null
  },
  "execution_container": "embedded_app",
  "draft_revision": 42
}
```

返回：

- device/physical memory identity；
- catalog/runtime revision；
- 每个 target 的 `available|tentative|unavailable`；
- stable reason code；
- evidence scope/release label；
- recommended target（只作为 UI 建议）；
- `requires_exact_resolve=true`。

options 禁止 materialize 权重、创建 GPU pool、启动 I/O worker、跑 benchmark 或修改 MLX process-global policy。
未知 token/source 只能 tentative；production 空 catalog 应返回明确 `catalog_has_no_public_records`，而不是让用户误解成模型名不支持。

### 7.2 Resolve：engine/source 级精确冻结

新增：

```c
int tc_engine_resolve_streaming_json(tc_engine *engine,
                                     const char *request_json,
                                     char **result_json,
                                     char **error);
```

resolve 允许：

- 校验 engine model/source；
- 读取受限 metadata/index；
- 运行 tokenizer count/prepare-tokens；
- 生成 descriptor、compile 最终 chosen layout；
- 校验 record identity；
- 返回 exact preset selector、resolution digest、展示摘要。

resolve 禁止：

- 创建 fill worker/slot pools；
- 加载完整 tensor 或执行 encoder forward；
- 修改全局 GPU/MLX policy；
- 根据临时空闲内存改 record；
- 接受外部 catalog path 或测试 authority。

如果 artifact 尚未经过可信安装验证，返回 `artifact_verification_required`，由独立安装工作流完成，不在每次 UI 输入变化时 hash TB 级文件。

### 7.3 Generate：只消费冻结结果

有两种安全实现方式，建议优先 A：

```text
A. App提交exact selector + expected_resolution_digest
   engine内部重新resolve并立即generate_resolved

B. engine保存短生命周期resolved handle
   App提交opaque handle；engine仍重新核验source/revocation
```

A 更适合跨进程和重启，不依赖 bearer handle；B 可减少同进程 metadata 重建，但要处理 TTL/engine identity。
两者都不能把 digest 当权限。首版可只实现 A，并用 bounded metadata cache 优化。

`ModelSession` 增加：

```cpp
virtual RunResult generate_resolved(
    std::shared_ptr<const ResolvedRequestExecution>,
    const Event &, std::atomic<bool> &);
```

默认实现必须抛 `streaming_public_adapter_unsupported`，不可调用普通 `generate`。四模型逐一 override 后才可发布 record。

### 7.4 ABI 内存与并发合同

- 调用开始将 `*result/*error` 置 NULL；成功只返回 result，失败只返回 error；
- 所有字符串由 `tc_string_free` 释放；异常不得穿过 C ABI；
- resolve 使用 engine try-lock，busy 时立即返回，不影响当前 generate 的 cancel flag；
- options 使用只读 catalog snapshot，不持有 engine/GPU lock；
- callback 中禁止递归调用同一 engine；
- engine free 要求无 active call；
- stable error code 与 message 分离，Swift 不解析中文 message 做控制逻辑。

## 8. Catalog 的可实现数据模型

当前 `StreamingPresetRecord` 只够验证 resolver skeleton。真实 public record 至少拆成以下部分：

```json
{
  "schema_version": 1,
  "id": "illustrative.zimage.bf16.512sq.p14.k2",
  "revision": 1,
  "catalog_revision": "illustrative-catalog-r1",
  "release": {
    "channel": "proposed",
    "revoked": false,
    "review_commit": null,
    "evidence_digest": null
  },
  "applicability": {
    "model": "z-image-turbo",
    "variant": "original-bf16",
    "artifact_manifest_digest": null,
    "operation": "image.generate",
    "execution": "gpu",
    "execution_container": "embedded_app",
    "device_class": "illustrative-only",
    "runtime_revision": "illustrative-only",
    "reader_revision": "illustrative-only",
    "kernel_revision": "illustrative-only",
    "workload": {
      "width": 512,
      "height": 512,
      "frames": 1,
      "steps": 9,
      "audio": false,
      "text_valid_rows": {"min": 1, "max": 512},
      "text_compute_rows": [512],
      "batch": 1,
      "loras": [],
      "approximation": false
    }
  },
  "plan": {
    "component_policy_revision": "illustrative-only",
    "stages": {
      "transformer": {
        "residency": "streamed",
        "block_group_size": 1,
        "slot_count": 2,
        "resident_prefix_blocks": 14,
        "prefetch_distance": 0,
        "io_workers": 1,
        "pass_transition": "reload",
        "multi_pool_policy": "retain_all"
      }
    }
  },
  "calibration": {
    "complete": false,
    "scope": "execution_process_tree_v1",
    "estimator_revision": "observed_tree_max_v1",
    "calibrated_request_bytes": null,
    "coverage": "incomplete",
    "device_samples": []
  },
  "performance": {
    "rank": null,
    "full_request_wall_median_seconds": null,
    "full_request_wall_p95_seconds": null,
    "logical_read_bytes": null,
    "comparison_kind": null
  }
}
```

该示例故意是 `proposed` 且 calibration incomplete，builder 必须拒绝编入 public index。

### 8.1 Catalog 生成和嵌入

推荐两步生成：

```text
evidence bundles
  → build_streaming_catalog.py propose
  → proposed catalog + verifier report
  → independent review/review commit
  → build_streaming_catalog.py compile-reviewed
  → generated immutable C++ table + digest
```

source JSON 可以进入仓库，runtime 只加载生成后的 immutable table。CI 检查生成文件没有漂移、排序确定、重复
ID/revision 被拒绝、public record 均有 evidence/review/release 字段。release binary 不读取任意用户路径作为 catalog。

### 8.2 Resolver 过滤顺序

先使用廉价索引缩小集合，再对最终候选 compile descriptor/layout：

```text
1. validate selector and merged request conflicts
2. identify model/variant/operation/container/release channel
3. identify verified artifact and runtime/reader/kernel revisions
4. derive exact workload and token rows
5. filter device class and physical-memory qualification
6. filter calibration completeness and M + H(T) <= T
7. filter exact preset selector, if replay
8. deterministic rank: performance → memory → reads → ID/revision
9. compile one final canonical plan
10. compare all identities and mint authority
```

不得为所有 rejected candidates 创建 descriptor 或读完整 metadata。任意 required field unknown 都是 unavailable，不是 wildcard。

## 9. App 配置、推荐与任务冻结

### 9.1 StudioDraft

建议新增：

```swift
enum StreamingMode: String, Codable, Sendable { case off, memoryTier }

struct StreamingChoice: Codable, Sendable, Equatable {
    var mode: StreamingMode = .off
    var targetBytes: UInt64?
}

struct StudioDraft {
    var streamingChoicesByModel: [String: StreamingChoice] = [:]
}
```

旧 draft 缺字段时为 off。损坏字段必须提示，不静默选一个 target。按 model 保存“目标偏好”，不保存 exact P/G/K/D/Q。
checkpoint、shape、steps、prompt/token、backend、LoRA、ANE、audio 或 profile 变化时，现有 resolution 失效并重新 query。

### 9.2 物理内存推荐只负责 UX

推荐不能代替 catalog eligibility。UI 可使用保守的展示算法：

```text
system_reserve(P) = max(4 GiB, ceil(20% × physical_memory P))
recommendation_cap = P - system_reserve(P)
recommended = largest native-returned available target <= recommendation_cap
```

按该初值，16/24/32 GiB 机器的通用展示上限分别约为 12/16/20 GiB；8/10 GiB 机器可能没有可推荐 public target。
这只是产品默认值，不是运行资格，也不是实测结果。最终 resolver 仍要求该 record 在精确 device/container 上有校准。

UI 规则：

- 首次开启时预选 `recommended`；无可用档则保持未选择并显示原因；
- 用户可查看其他 native 返回的 available 档，但 device-ineligible 档不能强行选择；
- 不用 `ProcessInfo.physicalMemory` 自己维护 model→P 表；
- 临时 pressure 只影响 admission 提示，不产生新的 exact layout；
- 用户关闭时恢复旧 residency/budget 控件值；
- Z-Image 旧 6/8/10/12 GiB sampling budget 与新整请求 target 分开保存和展示。

### 9.3 StreamingOptionsStore

新增独立 view model：

```swift
@MainActor
final class StreamingOptionsStore: ObservableObject {
    @Published private(set) var state: QueryState
    private var task: Task<Void, Never>?
    private var revision: UInt64 = 0
    private var cache: [OptionsKey: NativeStreamingOptions] = [:]
}
```

要求：

- 开关 off 不创建 query task；
- draft 变化约 300 ms debounce；
- 每次请求带 revision，旧返回丢弃；
- task cancel 不调用生成 engine 的 `tc_engine_cancel`；
- cache key 包含 model/source/workload/container/catalog/runtime；
- query 不启动 benchmark、hash 全模型或加载权重；
- UI 只显示 target/status/reason/evidence，layout 放在只读详情。

### 9.4 请求 payload 与 NativeJob

不要把所有旧请求无条件升级到 v2。建议：

```swift
enum NativeRequestPayload: Sendable {
    case legacy(NativeRequest)
    case publicStreaming(NativeRequestV2)
}
```

当前 `NativeRequestV2(legacy:targetBytes:)` 是可编译骨架，正式接入前必须完成 semantic parity audit：

- v1 所有 input 字段、audio/input audio 语义；
- `encoder_ane_manifest`、warmup、profile、noise、compile、LTX flags；
- explicit false/0/absent 的保留；
- unsupported 字段必须拒绝，不能因 v2 类型未覆盖而丢失；
- `CodingKeys` 和 golden JSON，避免依赖 Swift synthesized enum shape。

`NativeJob` 改成 versioned envelope：

```text
legacy job: existing request field, no job_schema_version
new off job: still legacy payload
public job: job_schema_version=2
            request_payload=<full v2 request>
            streaming_submission=<intent + exact selector + digest + evidence summary>
```

历史 v1 job 不自动升级。output 删除、参数复用、route summary、preview、worker routing 都改用共同 accessor，
不能再假定 `job.request` 永远是 v1。未知新 job schema 不得启动时覆盖原 jobs 文件。

### 9.5 提交事务

```text
1. MainActor 设置 busy，复制 immutable draft
2. 构造 legacy 或 v2 payload
3. off：沿用原 plan/acquire/generate
4. on：options状态校验 → acquire engine → exact resolve
5. 比较 draft revision、target、source、catalog
6. 持久化 frozen job
7. 持久化成功后 generate exact request
8. 写 actual result/terminal state
```

resolve 成功但持久化失败不得执行 GPU。运行中 UI 改 target 只影响下一任务。历史 exact replay 遇到 revoked/stale 应失败；
“复用参数并重新推荐”是新任务，允许重新 resolve，但必须让用户看到选择变化。

## 10. Runtime 接线与模式切换

### 10.1 C API generate 分支

伪代码：

```text
parse request
if no active selector:
    run existing path byte-for-byte equivalent
else:
    require public resolver compiled and production policy enabled
    resolve exact execution before configure_streams/pool allocation
    acquire GPU/global execution locks in frozen order
    recheck source/revocation/device baseline
    session.generate_resolved(snapshot)
```

off 分支要在访问 catalog singleton、tokenizer、descriptor、sampler和新 cache 之前分开。selector path 的 preflight wall
属于整请求成本，性能报告不能把它移到计时窗外。

### 10.2 Prepare 策略

首版建议明确不支持 public prepare：

```text
active selector + prepare → streaming_prepare_unsupported
```

App public streaming 直接 generate。这样避免 prepare 先全驻留加载再切 streaming、重复 resolve 或在 generate 前留下未计入
峰值的 cache。将来支持 prepare 时，必须由同一 authority 构造相同 backing plan，并重新完成 full-request/mode-transition 校准。

### 10.3 模式切换

至少测试：

```text
resident → public streaming → resident
legacy streamed → public streaming → legacy streamed
public preset A → public preset B → A
public streaming cancel/fail → off/default
```

每次切换明确处理：模型权重、MLX cache、native Metal buffers、I/O workers、source fd、Core ML resources、allocator policy。
不能为降低峰值在默认路径全局 `clear_cache`。若旧 cache 无法安全释放，resolver 要拒绝当前 baseline，而不是边运行边赌。

## 11. 继续复用的 slot/block 调度框架

Public preset 只生成 canonical manual plan，实际调度继续使用：

- `ResourceDescriptor`：block/field/source ranges/reader semantics；
- `LayoutCompiler`：P/G/K/D/Q、pool class、capacity、pass transition；
- `StageExecutor`：owner-pump、bounded dispatch、mailbox、class barrier；
- `SlotPool`：exact K slots、ticket generation、reader completion；
- `MlxWeightPager` 或原生 adapter：预分配 backing、worker pread、binding；
- model kernel：按原顺序执行，不知道 target/catalog。

`K` 是每个实际 pool 的精确 slot 数，`D<K`，`Q<=K`。prefix 是每 pass 计算、每 request 一次加载的权重；
不能误写成 prefix block 只算一次。multi-pool 的 `retain_all|serial` 是 layout identity，切换策略需要新 record 和 P4 证据。

Public resolver 不允许：

- suffix groups 少于 K 时自动降低 K；
- target 较小时改变 dtype/quantization；
- 运行时根据 wait 动态扩大 slot；
- 遇到 I/O 慢时回退 resident；
- 在 block hot path 查询 catalog或系统内存。

## 12. 四模型实施设计

### 12.1 共同准入

每个模型 public override 必须验证：

1. model/variant/format/artifact identity；
2. exact workload/token/backend/features；
3. descriptor/layout/component/reader/kernel revision；
4. actual P/G/K/D/Q、pool count、pass transition 与 record 相同；
5. full request 的 component live order；
6. cancellation/read failure/drain/quarantine；
7. result 报告 authority source 为 `public_preset_registry`。

### 12.2 LTX 2.5 distilled

首版范围：ConvRot INT8 checkpoint、GPU C/Metal、dense quality、T2V、512×320×33、11 steps、无 audio/I2V/LoRA/ANE/Sol。

代码改造：

- `ltx_session.mm` 增加 `generate_resolved`，消费 frozen LTX source/plan；
- `ltx_streaming_descriptor.cpp` 的 metadata identity 进入 record match；
- `ltx_streaming_plan.cpp` 不再从 target 推导参数，只验证 canonical config；
- 现有 exact owner/pager 生命周期扩展 authority 和 result report；
- `LTXWorker.accepts` 不用伪造 `component_staged`，container 由 record 显式决定。

候选仍按 24：P1/4/8 的 K2、P4/8/12/16 的 K3，P1/K1 只作最低占用研究。P8/G1/K3/D2/Q3
是同布局性能锚点，不是任何 target 的认证。audio、97 frames、I2V 必须独立 workload card。

### 12.3 Z-Image Turbo

首版范围：original/Comfy BF16、GPU、T2I、无 LoRA/ANE/GGUF，512²×9 steps；1024²独立卡。

代码改造：

- `z_image.cpp`/session 增加 public frozen plan 入口；
- `z_image_streaming_descriptor.mm` 保持 P 可变、K2/G1/D0/Q1；
- `z_image_weight_stream.mm` 验证 prefix/fixed/lazy array 无整模型隐藏 materialization；
- record 固定 text valid/padded/compute rows；
- denoise→VAE 的 pool release 若改变，bump component policy 并重新做 P1/P4。

候选 P0/4/8/12/14/18/22/26/28；P29 拒绝。旧 6/8/10/12 GiB sampling budget 绝不能转换成这些
整请求 target。现有 P14 一步 P1 只证明框架成本；public 需要 9-step full request。

### 12.4 MiniMax H3 Turbo

只做 H3 Turbo original BF16，不做普通 H3、FastH3、INT6/量化缓存或其他 checkpoint。

代码改造：

- `h3_session.mm` 增加 public authority 消费，不永久打开 private bool；
- descriptor/source identity 绑定原始 BF16 shards 和 50 blocks；
- 保持 K2/G1/D1/Q1、四 pass、`carry_first_group` 和大矩阵单次大 `pread`；
- full session 覆盖真实 Qwen3 text、DiT、video VAE、MP4/export；
- P0/2/4/8/12/16/24 分别验证奇/偶 suffix carry。

第一张卡为512²×22帧×4步、无 audio；39/73帧独立。当前 denoiser probe 工程结果允许合理 I/O 波动，
但 bootstrap `INCONCLUSIVE` 仍如实保留，不能冒充 full-pipeline public PASS。

### 12.5 Flux.2 Klein 9B

首版范围：9B、Diffusers BF16 two-shard、GPU eager、无 LoRA/ANE；Flux 4B compiled graph 不在范围。

P0/G1/K2/D1/Q2 可以先做较高 target 的 full-request 认证。P>0 需要额外实现：

- pager resident prefix arrays；
- global dual[0..7]+single[8..31] block index；
- block 8 concatenate 每 pass 精确一次；
- P8 时删除空 dual pool；
- P7 因 dual suffix 不足 K2 拒绝；
- adapter 使用 compiled pool IDs，不混用 class enum/vector index；
- P0 路径保持不变，作为回归基线。

P2/P4/P6、P8、P12 分阶段验收。低于现有 P0 全流程峰值的 target 不能靠增加 prefix 实现；需要 text-stage、
serial pool 或 component release 的新策略时，建立新 component/backing revision 并按 P4 验证。

### 12.6 首批范围表

| 模型 | 首批 public 候选 | 明确排除 |
|---|---|---|
| LTX | 2.5 distilled C/Metal dense T2V | audio、I2V、Sol、ANE、LoRA、MLX route |
| Z-Image | Turbo original BF16 T2I | GGUF、LoRA、ANE、旧budget语义 |
| H3 | MiniMax H3 Turbo original BF16 | 普通H3、量化、FastH3、audio |
| Flux | Klein 9B BF16 eager | Flux4B compiled、LoRA、ANE |

某模型无合格 target 时保持 unavailable，不要求四模型同时或五档全部上线。

## 13. GPU 与 ANE 的边界

用户最初要求先只考虑 GPU，因此 public catalog v1 冻结 `execution=gpu`。当前 selector validation 已允许语法上的
`gpu_ane` 一致性检查，但 production resolver 在没有 GPU+ANE 专属 record 时必须返回
`streaming_route_unsupported`，App 不应显示可选档。

这样设计有三个原因：

1. ANE manifest、partition rows、近似许可和 Core ML cache 会改变 workload/component identity；
2. GPU+ANE 的峰值可能发生在模型切换、CPU/ANE/GPU buffer 重叠阶段，不能沿用 GPU-only calibration；
3. `dev` 分支已有 M5/ANE 优化，过早把 streaming 和 ANE 耦合会扩大 merge 与性能回归面。

未来接 ANE 时：

- record 的 execution/container/device/manifest digest 必须独立；
- options 只展示本机已验证 manifest 的档位；
- GPU-only record 不允许自动关闭 ANE，也不允许 ANE request 自动转 GPU；
- 做 GPU+ANE default P0、same-plan P1、整请求 memory calibration；
- 确认 encoder-only ANE 与 denoiser ANE 分开建 identity；
- streaming off 的现有 ANE 性能必须先通过 dev 基线对照。

## 14. 内存档位、swap 与模拟

### 14.1 为什么可能比自动 swap 更快

可控 streaming 的优势来自：

- 只读取下一批所需的连续权重，而不是让 OS 在压力下随机淘汰活跃页；
- slot 数有界，GPU reader completion 后才复用；
- 可提前 prefetch，与 GPU compute overlap；
- component 生命周期显式，避免 text/DiT/VAE 无意重叠；
- I/O bytes、wait、fill、peak 可以测量并优化。

但它不保证总比 swap 快。SSD 顺序读、page cache、统一内存带宽、activation、VAE、prefix 和 worker concurrency 都会影响结果。
高内存机器 resident 往往更快；因此 UI 默认仍为 off，public streaming 的目标是可预测地降低峰值并避免失控 paging。

### 14.2 模拟的正确用途

离散事件模拟输入：

- 每 group source ranges/bytes；
- read 服务时间分布；
- kernel时间；
- K/D/Q；
- pool class transition；
- reader completion；
- component live intervals。

输出：预计 wall、wait、I/O worker 利用率、pool live bytes、候选排序。模拟只用于 E0/E2 剪枝，不生成 public record。
至少保留 held-out layout，建议 wall 中位误差≤10%、P95≤20%；不满足时只作 sensitivity 分析。

### 14.3 与 swap 的实测

按 26 的 P3 顺序：

1. 无压力：resident vs streaming，建立质量和策略成本；
2. 自然低内存实体机：记录 footprint、pressure、swap delta、fault、wall/tail；
3. 经单独授权的 bounded pressure：冻结最大分配、timeout、stop、cleanup，不关闭系统 swap。

比较 full request success rate、wall median/P95、swap in/out delta、major faults、thermal/pressure、输出质量。
不能用 64 GiB 机器仅设置软件 target 代替 8/12/16 GiB 实体机，也不能把一次没有 swap 当作保证。

## 15. 性能零回归设计

### 15.1 Off fast path

代码审查要求：

- request 无 active selector 时不调用 catalog/resolver/tokenizer/descriptor；
- App 开关 off 时不发 options query；
- 不启动 sampler、I/O worker 或额外线程；
- 不添加每 block public flag/string/hash/JSON 分支；
- 不为 default route 增加 `clear_cache`、unload 或同步；
- production empty catalog 不在进程启动时扫描资源；
- result 未启用 streaming 时省略新增报告对象。

### 15.2 必测性能层级

| 层级 | 回答问题 | 比较 |
|---|---|---|
| P0 | 新代码是否伤害现有路径 | 合并前/后 default resident、legacy streamed、ANE |
| P1 | 通用框架本身是否有开销 | direct/legacy 与 generic 完全相同布局/生命周期 |
| P4 | 不同 preset 的内存/速度取舍 | 独立确认后的布局策略比较 |
| P3 | 受压力时是否优于 OS paging | resident pressure vs streaming pressure |

默认 non-regression 继续使用文档12的 median 2%、P95 5%阻断线和 bootstrap 规则。不同 P/K/retention 的收益不能抵消 P0 回归。

### 15.3 控制面预算

建议冻结初值：

| 操作 | 目标 |
|---|---|
| off request 新增控制面 | audit为零；wall无可测回归 |
| cached options | 主线程无同步I/O；P95 < 10 ms |
| uncached metadata options | 不加载权重；P95 < 100 ms，超出则异步展示 |
| exact resolve | 有界 metadata/tokenizer；单独报告，不隐藏于计时外 |
| per-block public overhead | 0 catalog lookup、0 JSON、0 allocation、0 thread create |

以上是拟议工程目标，不是当前成绩。设备达不到时先分析协议，不能删除 preflight 安全检查换取数字。

## 16. 可观测性和错误合同

### 16.1 Result

active public request 的 result 增加：

```text
streaming_selection:
  requested target
  exact preset/revisions/digest
  release/evidence/container/device
streaming_actual:
  P/G/K/D/Q
  pool count/class/retention/pass transition
  source/fill/prefix bytes
  waits/fills/readers/drain
streaming_memory:
  calibrated request bytes/scope/coverage
  current observed values if sampled
  memory_enforcement=none|bounded
```

catalog calibration 与本次 sampled peak 分开；unknown 为 null，不显示 0。actual 与 resolved 不一致时请求失败并报告
`streaming_actual_plan_mismatch`，不能成功后只写 warning。

### 16.2 稳定错误码

| 类别 | 代表错误码 |
|---|---|
| 配置 | `streaming_selector_invalid`、`streaming_config_conflict` |
| 可用性 | `unsupported_model_or_format`、`unvalidated_workload`、`no_preset_fits_target` |
| 身份 | `artifact_verification_required`、`artifact_changed`、`streaming_resolution_stale` |
| 发布 | `catalog_has_no_public_records`、`preset_not_public`、`preset_revoked` |
| 路由 | `streaming_route_unsupported`、`streaming_public_adapter_unsupported` |
| 执行 | `streaming_prepare_unsupported`、`streaming_quarantined`、`engine_busy` |
| 内存 | `component_floor_exceeds_target`、`temporary_memory_pressure` |
| 内部 | `streaming_actual_plan_mismatch`、`streaming_authority_mismatch` |

Swift 使用 code 决定 UI，message 仅展示。错误必须标 `retryable` 与建议动作；不能统一提示“显存不足”。

## 17. 实施测试设计

### 17.1 无 GPU 的 host/contract

必须覆盖：

- selector absent/off/manual/memory_tier/preset；
- profile/request 原子替换与 legacy 冲突；
- bool/负数/0/小数/指数/字符串/`2^53`/duplicate keys；
- target margin 边界差1 byte；
- deterministic rank、device/container/calibration/revocation；
- production empty catalog；
- unresolved selector generate/prepare 零 session/GPU 调用；
- resolution digest 对 JSON key order 稳定，对 layout/source 变化敏感；
- test fixture catalog 不进入 release binary。

### 17.2 App/Swift

- v1 encoder golden byte/semantic 不变；
- v2 所有字段 round-trip；
- 旧 draft/job decode 默认 off；
- mixed v1/v2 job、unknown future schema 不破坏文件；
- Z legacy budget 新开关 on/off 后恢复；
- debounce/revision 逆序响应；
- options off 零调用；
- resolve 后 persist failure 不 generate；
- replay revoked/stale 不自动换 preset；
- output 删除/preview/route summary/参数复用保持安全。

### 17.3 Runtime/lifecycle

- public authority model/source/workload/layout 绑定；
- allocation/read/cancel/fence/class/pass/VAE/export 故障；
- unsafe drain 进入 quarantine；
- request owner 晚于 callback/source/pager；
- 模式切换后 default allocator/cache policy 恢复；
- active reader/inflight fill/worker 在 success 后为零；
- quarantine retained object/bytes 有明确报告。

### 17.4 模型与实机

每条 public record 都需要：

- full output quality；
- full-request memory tree coverage；
- fresh/warm/mode-transition；
- P0/P1/P4，P3如要宣称 swap 优势；
- 本机 device/container 资格；
- cancel/read/source mutation/cleanup；
- App ordinary constructor 重放；
- catalog revoke 演练。

## 18. PR 拆分与依赖

### PR-A：安全闸门和控制面收口

内容：

- unresolved selector generate/prepare fail-closed；
- 保留现有 duplicate-key raw scanner，并补 target 的无指数十进制 lexical integer 校验；
- stable error envelope；
- 现有 selector/catalog/options/Swift skeleton 测试补齐；
- production catalog 继续为空。

完成门：build、contract/host、普通 public constructor 零执行；default gate 不变。

### PR-B：Catalog v1 完整 schema 与 resolver

内容：artifact/workload/device/container/revision/calibration/performance/release identity；canonical digest；
fixture builder/verifier；metadata options reason taxonomy。

完成门：全部 host tests；release binary 无 test authority；仍不可生成。

### PR-C：Engine resolve 与 immutable execution

内容：source lease、token count、descriptor/layout compile、authority、`tc_engine_resolve_streaming_json`、
`generate_resolved` 默认拒绝、request owner/quarantine。

完成门：fake session 证明 compile once、TOCTOU、stale/revoke、ownership/fault；四模型仍未自动开放。

### PR-D：Swift/App opt-in 控制面

内容：完整 v2 parity、StreamingChoice、OptionsStore、Job envelope、UI、Insights；使用 fixture/mocked options。

完成门：default off 序列化和行为不变；App release build 无 candidate API；没有 production record 时显示 unavailable。

### PR-E：校准和 catalog 工具

内容：fresh-process sampler、PID/tree/time schema、candidate inspect、explore、independent verifier、propose/compile-reviewed。

完成门：CAL 反例全通过；默认 runtime 零 sampler；预算/停止/持久化安全。

### PR-F：模型 public adapter

建议顺序：LTX/Z-Image → H3 Turbo → Flux P0 → Flux prefix。每个模型独立 PR，可只产生 reviewed candidate，
不必同时编入 public catalog。

### PR-G：首批 records 与发布

只提交独立审核通过的 records、evidence digest、UI release labels 和撤回演练。一个模型/一个 workload/一个 target
也可以是首批；不要为了“五档齐全”加入无证据映射。

### PR-H：合并 dev 与 ANE 回归

完成 public GPU 控制面和默认性能防线后再合并 `dev`。解决 request/Swift/App/ANE profile 冲突时：

- 保留 dev 的已验证 ANE device scope；
- public v1 仍 GPU-only；
- 重跑 default GPU/ANE P0；
- selector off 不改变 ANE route；
- selector on + gpu_ane 无 record 时明确拒绝。

## 19. 验收追踪矩阵

| 交付 | 代码完成 | Host | App | GPU full | Memory | Performance | Public |
|---|---:|---:|---:|---:|---:|---:|---:|
| selector/parser/plan report | `63b73d9` | PASS基础 | NOT_RUN | N/A | N/A | off需P0 | 否 |
| catalog/resolver skeleton | `63b73d9` | PASS基础 | NOT_RUN | N/A | fixture | N/A | 否 |
| options C ABI/Swift types | `63b73d9` | 编译/PASS基础 | 未接UI | N/A | N/A | query未测 | 否 |
| unresolved selector fail-closed | 已实现 | PASS contract | N/A | 普通/candidate均早拒绝 | N/A | off仍需P0 | 否 |
| exact engine resolve/authority | 未实施 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | 否 |
| App开关/job迁移 | 未实施 | N/A | NOT_RUN | NOT_RUN | N/A | off需P0 | 否 |
| calibration toolchain | 未实施 | NOT_RUN | N/A | NOT_RUN | NOT_RUN | sampler开销未测 | 否 |
| LTX public records | 未实施 | private descriptor PASS | NOT_RUN | NOT_RUN | NOT_RUN | 仅旧P1锚点 | 否 |
| Z public records | 未实施 | private descriptor PASS | NOT_RUN | NOT_RUN | NOT_RUN | 仅旧P1锚点 | 否 |
| H3 Turbo public records | 未实施 | private descriptor PASS | NOT_RUN | NOT_RUN | NOT_RUN | 旧工程锚点 | 否 |
| Flux 9B public records | 未实施 | private descriptor PASS | NOT_RUN | NOT_RUN | NOT_RUN | 旧P0/P1锚点 | 否 |

表中`63b73d9`只表示控制面代码已经提交，不等于发布或模型资格。每次实现 PR 更新真实路径/命令/结果，
不用完成百分比替代证据。下一阶段的类型和逐文件设计以[28](28-public-runtime-code-design.md)为准，实验和发布执行以
[29](29-public-implementation-and-acceptance-plan.md)为准。

## 20. Definition of Done

框架层完成：

- [ ] off/default 代码路径和性能签核；
- [ ] unresolved selector 无任何执行回退；
- [ ] options/resolve/generate 权限边界明确；
- [ ] immutable snapshot/authority/owner 生命周期通过 fault tests；
- [ ] catalog build/review/revoke 可追溯；
- [ ] App v1/v2 job 安全迁移；
- [ ] default release binary 无 candidate/test authority；
- [ ] GPU-only 与 ANE 路线不混淆。

单个 preset 完成：

- [ ] exact model/artifact/workload/device/container identity；
- [ ] descriptor/layout/component/reader/kernel identity；
- [ ] full output quality；
- [ ] full-request memory calibration complete；
- [ ] `M + H(T) <= T`；
- [ ] P0/P1/P4 结论及置信状态；
- [ ] fault/cancel/source mutation/drain/quarantine；
- [ ] ordinary App/public constructor staging；
- [ ] evidence/review/catalog digest；
- [ ] revoke/rollback 演练。

产品层完成：

- [ ] 高级开关默认关闭；
- [ ] 只显示 native 返回的已验证/实验档位；
- [ ] 推荐与资格分开；
- [ ] 不暴露可编辑 P/G/K/D/Q；
- [ ] unavailable 原因清楚；
- [ ] 结果显示 target、preset、scope、actual layout、证据；
- [ ] 不承诺 hard cap、零 swap 或无条件加速。

## 21. 实现审阅清单

提交前逐项回答：

1. selector off 是否在 catalog/tokenizer/descriptor 前分支？
2. active selector 未 resolve 时是否一定拒绝 generate/prepare？
3. resolver 是否只接受 embedded reviewed records？
4. record 是否绑定 artifact/workload/device/container/runtime/reader/kernel？
5. exact replay stale/revoked 时是否拒绝而非换 preset？
6. App 是否保存完整冻结 payload，而不是只保存 preset ID？
7. v2 是否保留 v1 全部语义或明确拒绝？
8. session 是否消费同一个 compiled snapshot，避免二次选 plan？
9. owner 是否覆盖 worker/reader/callback 的完整生命周期？
10. actual layout 与 resolved record 是否逐项核对？
11. 日常路径是否没有密集 memory sampler/audit？
12. model candidate 是否只覆盖文档声明的 variant/workload？
13. ANE/LoRA/approximation 是否被准确拒绝或有独立 record？
14. 性能比较是否区分 P0/P1/P4/P3？
15. memory 数值是否 full-request、同 scope、coverage complete？
16. 失败/高峰样本是否保留，未做 post-hoc 删除？
17. test fixture/authority 是否不能进入 release binary？
18. catalog 置空时 default 是否仍完全可用？
19. 合并 `dev` 后是否重跑 GPU 与 ANE default P0？
20. 文档状态是否与代码/证据一致，没有把 private anchor 写成 public 资格？

满足上述清单后，TurboCider 才拥有一个可对 LTX、H3 Turbo、Z-Image、Flux 9B 统一管理、可逐模型扩展、
默认无性能代价，并且能用证据安全公开的 slot/block streaming 产品框架。
