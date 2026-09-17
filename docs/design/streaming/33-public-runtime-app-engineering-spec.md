# 33 · Public Streaming Runtime 与 App 工程实施规格

[目录](README.md) · [收口规格](32-public-streaming-completion-spec.md) · [代码设计](28-public-runtime-code-design.md) · [模型与档位验收](34-model-tier-calibration-and-release-spec.md) · [实施蓝图 v2](35-public-streaming-implementation-blueprint-v2.md) · [当前进度](13-implementation-progress.md)

日期：2026-09-17。状态：**工程施工规格；production catalog 为空，App 尚未开放 public streaming。**

本文把 28–32 中的架构结论进一步收敛为可直接分配、编码和验收的 runtime/App 工作包。本文不会把 private candidate、设计中的候选档位或一次真实请求写成 public 支持。
当前工作树对应的 coordinator/source lease/actual receipt/test catalog 进一步拆分和 R0–R8 顺序见
[35](35-public-streaming-implementation-blueprint-v2.md)。

## 1. 目标与完成边界

本阶段的目标不是先加入 production record，而是先完成一条能够安全承载 reviewed record 的公共执行链：

~~~text
App user intent
  → request-only preflight
  → exact native resolve
  → durable job snapshot
  → generate 内部重新 resolve
  → source/runtime/device revalidation
  → authorized adapter execution
  → actual-plan hard verification
  → public metrics/result
~~~

只有下列三层全部完成，才可以说某个模型的某个档位“公开可用”：

| 层 | 完成定义 |
|---|---|
| 公共框架 | schema、validator、resolver、authority、C ABI、Swift、App transaction、result verification 全部通过 |
| 模型 adapter | probe、snapshot、source lease、generate_resolved、failure drain、actual-plan report 全部通过 |
| 单个 record | 完整请求内存、性能、质量、生命周期、App replay、独立复核全部通过并进入 production catalog |

本文明确不把以下内容视为完成：

- production catalog 为空时 resolve 能正确失败；
- private exact adapter 已经能执行；
- denoiser-only MLX peak 低于目标；
- 同布局 P1 已通过；
- App 中已经存在旧 `residency` 或 Z-Image budget UI；
- native options 返回五个目标但都为 unavailable。

## 2. 当前代码事实与立即修复项

截至 2026-09-17，当前工作树已经具备 exact resolver、authority、immutable resolved request、C ABI resolve 和 Swift binding 骨架。仍需按下表处理：

| 编号 | 当前事实 | 处理 |
|---|---|---|
| RT-BOOT-001 | `public_request_validation.cpp` 已抽出共享 validator | 保持无 catalog/GPU/model module 依赖；已修正为 include `core/streaming_contracts.hpp` |
| RT-BOOT-002 | `resolve_public_streaming_locked` 已在空 catalog 前调用 validator | 增加错误优先级与 planner/API parity 测试 |
| RT-BOOT-003 | session public hooks 默认拒绝 | 四模型逐一 override，不能在基类中自动转发 private route |
| RT-BOOT-004 | generate 内部会重新 resolve | 补 source lease、actual-plan verifier、public result |
| RT-BOOT-005 | App 仍直接持久化 `NativeRequest` | 引入 job schema v2 和 exact resolution snapshot |
| RT-BOOT-006 | LTX component-staged 走独立 CLI worker | public resolve 必须在最终 worker 内完成 |
| RT-BOOT-007 | production catalog 为空 | 在所有 runtime/App/adapter/tooling 验收前保持为空 |

共享 validator 的最终依赖边界为：

~~~text
public_request_validation.cpp
  may depend on: core/contracts, core/streaming_contracts, core/common
  must not depend on: preset_catalog, preset_resolver, session, Metal, MLX,
                      filesystem, tokenizer, model module
~~~

## 3. 模块边界和依赖方向

建议保持以下单向依赖，禁止反向 include：

~~~text
core request/schema
       ↓
request-only public validator
       ↓
catalog value types + canonical identity
       ↓
model probe → deterministic resolver → immutable snapshot → authority
       ↓
engine orchestration
       ↓
model adapter execution core
       ↓
actual-plan verifier + RunResult serialization
       ↓
C ABI → Swift binding → App state/job store/UI
~~~

具体规则：

1. Catalog 不拥有 session、GPU buffer、file descriptor、worker 或 mutable slot。
2. Resolver 不调用模型 kernel，不创建 reader，不持有 GPU lock。
3. Probe 可以读取小型 manifest/index/header，但不能 materialize tensor。
4. Snapshot 可以持有只读 source lease 和 compiled descriptor/layout；不能创建 request-scoped slot content。
5. Authority 只能由 resolver 构造，不可 JSON deserialize，不可 copy。
6. Adapter 只消费 `ResolvedRequestExecution`，不得重新解释用户 target 并自行改 P/G/K/D/Q。
7. App 永远不生成 layout，也不把 native options 当 execution authority。

## 4. Request-only validator 的最终合同

### 4.1 API

保留当前简单接口：

~~~cpp
void validate_public_streaming_request(const Request &request);
~~~

engine model identity 和 execution container 不属于 request-only validator，由 engine orchestration 单独检查：

~~~cpp
require(request.model == engine.model_id,
        "streaming_engine_model_mismatch");
require(engine.execution_container == expected_container,
        "streaming_execution_container_mismatch");
~~~

这样可以避免 validator 需要 engine 类型，也避免 planner 为了调用 validator 伪造 container。

### 4.2 Validator 必须检查

- selector 存在并 active；
- selector schema、selection、retention、target、exact preset 字段合法；
- manual `streaming` 与 public selector 冲突；
- explicit residency、legacy memory budget、legacy offload 与 selector 冲突；
- `memory_constrained.enabled` 与 public v1 冲突；
- execution 必须严格等于 `gpu`；
- ANE manifest 和 encoder ANE manifest 必须为空；
- approximation、compiled graph、LoRA、quantized cache 必须关闭。

### 4.3 Validator 不得检查

- production catalog 是否为空；
- model ID 是否存在于 module registry；
- checkpoint 是否存在；
- prompt/token shape；
- width/height/steps 的模型专属约束；
- GPU availability；
- source digest；
- layout 是否 fit；
- target 是否对该 workload 有 record。

### 4.4 错误优先级

最终顺序冻结为：

~~~text
ABI pointer/ownership
→ quarantine
→ engine busy
→ JSON/schema/duplicate key
→ selector validation
→ engine model mismatch
→ public config conflict
→ public route unsupported
→ empty catalog / no public records
→ full request/model validation
→ probe/source/workload/runtime/device match
→ layout compile/authorize
~~~

planner 与 exact resolve 对同一个 request-only 错误必须返回同一稳定 code 前缀。消息文本可补充细节，但 App 不应依赖完整英文句子。

### 4.5 Validator 测试

新增或扩展：

| ID | 场景 | 预期 |
|---|---|---|
| PUB-VAL-001 | selector + legacy budget + empty catalog | `streaming_config_conflict` |
| PUB-VAL-002 | selector + GPU+ANE + empty catalog | `streaming_route_unsupported` |
| PUB-VAL-003 | selector + compile_gpu + empty catalog | `streaming_route_unsupported` |
| PUB-VAL-004 | 合法 selector + empty catalog | `catalog_has_no_public_records` |
| PUB-VAL-005 | engine model 与 request model 不同 | `streaming_engine_model_mismatch` |
| PUB-VAL-006 | target 不是五个公开值之一 | selector validation error |
| PUB-VAL-007 | selector disabled + legacy request | 完全走旧 plan 路线 |
| PUB-VAL-008 | planner/resolve 同一 conflict | code 相同 |
| PUB-VAL-009 | active selector + LoRA | route unsupported，且不读取 LoRA 文件 |
| PUB-VAL-010 | active selector + quantized cache | route unsupported，且不打开 cache |

## 5. Exact resolve 的代码结构

### 5.1 建议拆分

`native/api/c_api.mm` 不应长期承载全部 resolve 细节。建议将纯 C++ orchestration 下沉为：

~~~text
native/runtime/streaming/public_resolution.hpp
native/runtime/streaming/public_resolution.cpp
~~~

接口示意：

~~~cpp
struct PublicResolutionContext {
    std::string_view engine_model_id;
    std::filesystem::path model_root;
    std::string_view execution_container;
    ModelSession &session;
    StreamingDeviceIdentity device;
    const StreamingPresetCatalog &catalog;
};

std::shared_ptr<const ResolvedRequestExecution>
resolve_public_streaming(
    PublicResolutionContext,
    Request);
~~~

C API 只负责：pointer contract、engine lock、JSON decode、调用 helper、JSON encode。这样 host tests 可以使用 fake session/catalog，不需要 Objective-C runtime。

### 5.2 Resolve 固定阶段

| 阶段 | 输入 | 允许副作用 | 输出 |
|---|---|---|---|
| R0 | parsed request | 无 | request intent |
| R1 | request + engine ID | 无 | identity checked |
| R2 | request | 无 | preflight passed |
| R3 | catalog | 无 | availability checked |
| R4 | request | 仅普通模型校验 | complete plan |
| R5 | model root + metadata | 小型只读文件 I/O | probe |
| R6 | probe + selector + device | 无 | selected record |
| R7 | probe + record | 只读 source lease/descriptor compile | snapshot |
| R8 | record + probe + snapshot + device | 无 | authority |
| R9 | frozen request + selection | 无 | immutable execution |

R5–R7 仍不得 materialize weight payload、创建 Metal buffer、启动 I/O worker 或取得 global GPU lock。

### 5.3 Exact selector 语义

memory-tier resolve 输出 exact selector：

~~~json
{
  "schema_version": 2,
  "enabled": true,
  "selection": "preset",
  "retention": "request",
  "target_request_memory_bytes": "12884901888",
  "preset_id": "...",
  "preset_revision": 1,
  "catalog_revision": "...",
  "expected_resolution_digest": "..."
}
~~~

约束：

- byte count 在 JSON 中使用十进制字符串，避免跨 Swift/JSON/JS 的 53-bit 风险；
- exact replay 不得在 record 消失时自动选择同 target 的另一个 record；
- target 保留为用户选择和展示字段，不能改写为 legacy memory budget；
- exact selector 不携带 authority；
- App 保存 exact selector 只用于确定性重放，generate 内仍重新铸造 authority。

## 6. Source lease 与 TOCTOU 防护

### 6.1 Snapshot 合同扩展

给 `ModelStreamingSnapshot` 增加明确的执行前验证接口：

~~~cpp
struct SourceLeaseValidation {
    bool valid = false;
    std::string observed_source_digest;
    std::string failure_code;
};

virtual SourceLeaseValidation revalidate_source_lease() const = 0;
~~~

每个模型 snapshot 至少固定：

- normalized root；
- artifact relative paths；
- file identity（size、mtime ns、platform file ID，按可用性）；
- safetensors/GGUF/index header digest；
- shard list 和顺序；
- dtype/shape/range table digest；
- model variant；
- tokenizer/template/component policy revision。

### 6.2 三次校验点

| 时点 | 校验 |
|---|---|
| resolve probe | 建立 source identity |
| 获得 global GPU lock 后、创建 slot 前 | source lease + runtime/device/catalog revalidation |
| 第一次 refill 打开 artifact 时 | reader 对 snapshot 中的 file identity 再确认 |

第三次不是重新 hash 整个模型，而是阻止路径替换、shard 改动或 range table 与打开文件不一致。

### 6.3 失败行为

- 任何 mismatch 在创建可见 GPU content 前失败；
- 已经创建 request-scoped pool 时必须 poison、drain、destroy；
- engine 只在 cleanup 无法证明完成时进入 `streaming_quarantined`；
- source mismatch 不自动回退 resident；
- App 显示“模型文件已变化，请重新打开模型并重新解析”。

## 7. Generate 锁、取消与状态机

### 7.1 固定锁顺序

~~~text
engine mutex (try_lock)
  → parse + internal resolve
  → global execution mutex (try_lock)
  → DeviceLease
  → revalidate authority/source/runtime/device
  → configure backend
  → create request-scoped pools/workers
  → execute
  → drain/destroy
  → release DeviceLease/global mutex
  → release engine mutex
~~~

禁止：

- options/resolve 获取 global execution mutex；
- adapter 内反向获取 engine mutex；
- refill worker 获取 App/engine/global mutex；
- source hashing 在持有 global GPU lock 后才首次执行；
- cancel handler 销毁仍可能被 GPU reader 使用的 slot。

### 7.2 取消语义

1. Resolve 不读取、不清除 generation cancel flag。
2. Generate 取得 global lock 并通过 revalidation 后，才开始本次 request 的 cancel epoch。
3. owner pump 在 block/group 边界检查 cancel。
4. refill reader 在 artifact/range 边界检查 cancel。
5. 已提交的 GPU reader 必须等待真实 completion fence。
6. cancellation 期间禁止 slot reuse；先停止新 dispatch，再 drain 已提交 reader，再 destroy。
7. cleanup 成功返回 cancelled；cleanup 失败返回原始错误并 quarantine。

### 7.3 状态机

~~~text
Idle
  → Resolving
  → WaitingForGPU
  → Revalidating
  → SettingUp
  → Running
  → Draining
  → Completed

任意活动态 → Cancelling → Draining → Cancelled
任意活动态 → Failing → Draining → Failed
Draining cleanup failure → Quarantined
~~~

每次 terminal transition 只能发生一次；result/error ownership 也只能交付一次。

## 8. Actual-plan hard verification

### 8.1 为什么必须硬校验

Public authority 授权的是 exact layout，而不是“允许 adapter 自行选择一个大致相似的 streaming 模式”。若 adapter 实际创建 K3、遗漏 prefix 或把 dual/single pool 合并，即使输出成功也必须失败。

### 8.2 通用执行回执

建议新增：

~~~cpp
struct ActualStreamingExecution {
    std::string implementation_id;
    std::string layout_digest;
    std::string component_policy_revision;
    std::string pass_transition;
    std::string multi_pool_policy;
    uint64_t pool_count = 0;
    uint64_t slot_bundle_count = 0;
    uint64_t refill_worker_count = 0;
    uint64_t logical_read_bytes = 0;
    uint64_t refill_count = 0;
    bool drained = false;
    bool source_lease_verified = false;
};

void verify_actual_streaming_execution(
    const ResolvedRequestExecution &authorized,
    const ActualStreamingExecution &actual);
~~~

必须比较：

- authorized layout digest 与 actual layout digest；
- component policy revision；
- pass transition；
- multi-pool policy；
- expected pool/slot class 数；
- implementation ID 是否属于 record 的 runtime identity；
- source lease 是否已验证；
- terminal drain 是否完成。

不一致返回 `streaming_actual_plan_mismatch`，不得只写 warning 或继续返回成功图片/视频。

### 8.3 RunResult 新字段

建议在公共路径输出：

~~~json
{
  "public_streaming": {
    "schema_version": 1,
    "target_request_memory_bytes": "12884901888",
    "calibrated_request_bytes": "...",
    "preset_id": "...",
    "preset_revision": 1,
    "catalog_revision": "...",
    "record_digest": "...",
    "resolution_digest": "...",
    "source_digest": "...",
    "workload_digest": "...",
    "runtime_digest": "...",
    "device_digest": "...",
    "authorized_layout_digest": "...",
    "actual_layout_digest": "...",
    "component_policy_revision": "...",
    "execution_container": "embedded_app",
    "memory_scope": "full_process_tree_request",
    "actual_plan_verified": true
  }
}
~~~

Streaming off 时不创建此对象，保证默认 result 生成不需要 catalog/digest 工作。

## 9. C ABI 完整合同

### 9.1 `tc_engine_resolve_streaming_json`

必须满足：

- 入口先将 `*result_json` 和 `*error` 置空；
- null engine/input/output 返回稳定错误且不崩溃；
- engine busy 使用 try-lock，不等待长任务完成；
- 成功时仅 result 非空；失败时仅 error 非空；
- 所有字符串均由现有 free API 释放；
- resolve 不 reset cancel；
- resolve 不调用 session generate/prepare/load；
- resolve 不获取 global GPU lock；
- test hooks 不进入 release symbol table。

### 9.2 结构化错误

第一阶段可以保持 C ABI 的 error string，但 public 路线必须统一编码：

~~~json
{
  "schema_version": 1,
  "domain": "public_streaming",
  "code": "catalog_has_no_public_records",
  "message": "...",
  "retryable": false,
  "action": "disable_streaming"
}
~~~

Swift decoder 先尝试 envelope，再兼容 legacy plain text。不能把所有旧 native 错误全量迁移后才开放 public 路线。

### 9.3 C ABI 测试 ID

| ID | 检查 |
|---|---|
| PUB-ABI-001 | null engine |
| PUB-ABI-002 | null request |
| PUB-ABI-003 | null result output |
| PUB-ABI-004 | success ownership |
| PUB-ABI-005 | failure ownership |
| PUB-ABI-006 | engine busy immediate failure |
| PUB-ABI-007 | resolve 不改变 cancel epoch |
| PUB-ABI-008 | resolve 不获取 global GPU lock |
| PUB-ABI-009 | release binary 无 test hooks |
| PUB-ABI-010 | malformed structured error fallback |

## 10. Swift binding 设计

### 10.1 值类型

Swift 侧应保持以下层次：

~~~text
NativeStreamingSelector          user/exact wire intent
NativeStreamingOption           tentative UI option
NativeStreamingResolution       exact native resolution
NativePublicStreamingResult     observed execution result
NativeFailure                   decoded stable native failure
~~~

所有 byte count 使用能够无损处理 64-bit 的自定义 Codable wrapper；编码为十进制字符串，解码兼容历史 number，但重新编码必须规范化为字符串。

### 10.2 NativeEngine 方法

~~~swift
func streamingOptions(_ request: NativeRequest) async throws
    -> NativeStreamingOptions

func resolveStreaming(_ request: NativeRequest) async throws
    -> NativeStreamingResolution

func generate(_ request: NativeRequest, ...) async throws -> Data
~~~

`streamingOptions` 是无 engine 的 tentative query；`resolveStreaming` 必须针对已打开的 engine/model root；generate 自己再次 resolve。

### 10.3 并发要求

- Swift actor/serial queue 只保护 wrapper 生命周期，不替代 native try-lock；
- options 查询可以取消且不影响 engine；
- stale options response 不得覆盖新 workload；
- resolve 与 generate 之间 request 必须 immutable；
- exact resolution 不缓存 authority pointer。

## 11. App 状态模型与高级设置

### 11.1 用户可见状态

只暴露：

~~~swift
enum StreamingChoice: Codable, Equatable {
    case off
    case memoryTier(bytes: UInt64)
}
~~~

不暴露 P/G/K/D/Q、prefix、pass transition 或 multi-pool policy。

### 11.2 `StudioDraft` 迁移

新增字段：

~~~swift
var streamingChoice: StreamingChoice = .off
~~~

迁移规则：

1. 旧 draft 没有字段时为 `.off`。
2. 旧 Z-Image `residency=streamed` 继续表示 legacy private/旧 UI 路线，不自动升级 public。
3. Public choice active 时，构造 request 必须保持 legacy residency/budget/offload 未显式指定。
4. GPU+ANE、LoRA、compiled GPU 或不支持的 workload 下 UI 仍可显示 Streaming，但 option 为 unavailable 并给出原因；不得静默关闭用户选择后继续 resident。

### 11.3 `StreamingOptionsStore`

建议独立 `@MainActor` store：

~~~swift
@MainActor
final class StreamingOptionsStore: ObservableObject {
    enum State {
        case idle
        case loading(queryID: UInt64)
        case loaded(queryID: UInt64, NativeStreamingOptions)
        case failed(queryID: UInt64, NativeFailure)
    }
}
~~~

Query key 至少包含：

- model ID 和 normalized model path identity；
- operation、width、height、frames、fps、steps、audio；
- execution、ANE、approximation、compile、LoRA、quantized cache；
- dynamic text 和影响 token shape 的输入类别；
- execution container；
- native build/catalog revision；
- physical memory/device class。

采用 150–300 ms debounce。每次 workload 变化增加 monotonically increasing query ID；只有最新 ID 可以更新 UI。

### 11.4 推荐规则

推荐只基于 native 返回 `available` 的档位：

~~~text
headroom = max(2 GiB, ceil(15% × physical memory))
recommended = 最大的 available target，且 target <= physical - headroom
~~~

若没有满足项：默认 Off，并显示“当前模型/尺寸暂无已验证档位”。推荐不是自动启用；用户必须显式选择。

### 11.5 UI 行为

- 高级设置默认折叠；
- 默认 Off；
- 每档显示 target、native availability 和简短原因；
- exact preset ID 不作为主要 UI 文案，只放诊断详情；
- workload 变化后旧选择若不再 available，阻止提交并要求重新选择；
- 不能在提交时偷偷改为更高 target、不同 preset 或 resident。

## 12. JobStore schema v2 与提交事务

### 12.1 Job envelope

不要继续只保存裸 `NativeRequest`。建议：

~~~swift
struct NativeJobEnvelope: Codable, Sendable {
    var schemaVersion: Int = 2
    var userRequest: NativeRequest
    var streamingResolution: NativeStreamingResolution?
    var effectiveRequestDigest: String?
    var resolutionTimestamp: Date?
}
~~~

旧 `NativeJob` decode 时迁移到 schema 1/off，不合成 resolution。

### 12.2 原子提交顺序

~~~text
freeze draft
→ validate draft
→ resolve acceleration（public streaming 必须 GPU-only）
→ build immutable NativeRequest
→ if streaming active: engine.resolveStreaming
→ replace memory-tier selector with exact selector in frozen request
→ create pending job envelope
→ atomically persist jobs.json
→ only then start generate/worker
~~~

若 persist 失败，不允许 generate。若 generate 启动失败，保留 failed job 和 exact resolution 供诊断。

### 12.3 重放

- “再次生成”默认重放 exact selector；
- record revoked/stale 时明确失败，并允许用户点击“重新解析当前档位”；
- 重新解析会形成新 job，不原地篡改历史 job；
- 历史结果继续显示当时 target/preset/catalog/result，不根据当前 catalog 重解释。

## 13. LTX worker-local authority

当前 LTX component-staged 使用独立 CLI。Public streaming 下必须采用：

~~~text
App tentative options
→ App 打开/确认模型路径
→ App 可做 exact resolve 仅用于提交检查
→ persist job
→ launch worker with exact selector, not authority
→ worker create engine
→ worker exact resolve again
→ worker generate, revalidate and verify actual plan
→ worker returns public result
~~~

原因：authority 不可跨进程；worker 的 runtime build、container、source file identity、device identity 才是最终执行身份。

Worker 协议新增：

- request envelope schema version；
- execution container=`ltx_worker`；
- exact selector；
- optional expected App resolution digest，仅用于发现差异；
- worker resolution/result digest；
- structured error line，不能只依赖 stderr 尾部文本。

若 App resolution 与 worker resolution 不一致，worker 必须 fail closed，不能自动挑选别的 record。

## 14. App 与 Swift 测试矩阵

| ID | 场景 | 期望 |
|---|---|---|
| PUB-APP-001 | 新安装 | choice=Off |
| PUB-APP-002 | 旧 jobs/draft | 可解码且不自动启用 |
| PUB-APP-003 | 快速改变尺寸 | stale option response 被丢弃 |
| PUB-APP-004 | available 档位提交 | resolve→persist→generate 顺序正确 |
| PUB-APP-005 | persist 失败 | generate 未调用 |
| PUB-APP-006 | generate 前 record revoked | job failed，不 resident fallback |
| PUB-APP-007 | GPU+ANE + active public choice | UI/resolve 拒绝 |
| PUB-APP-008 | LoRA + active public choice | UI/resolve 拒绝 |
| PUB-APP-009 | old exact job replay | exact replay 或稳定 stale error |
| PUB-APP-010 | LTX worker | authority 在 worker 内生成 |
| PUB-APP-011 | cancel options | 不取消 generation |
| PUB-APP-012 | cancel generation | job terminal 一次且 cleanup 完成 |
| PUB-APP-013 | result mismatch | 显示失败，不展示为成功输出 |
| PUB-APP-014 | catalog empty | 高级设置显示暂无档位，默认路径可用 |

## 15. 默认性能保护

### 15.1 结构性 fast path

Streaming off/absent 时必须在第一个 public helper 前分流：

~~~cpp
const bool public_active = request.streaming_selector &&
                           request.streaming_selector->active();
if (!public_active) {
    return existing_generate_path(...);
}
return public_generate_path(...);
~~~

旧路径不得：

- 构造 production catalog view；
- probe/hash source；
- canonical encode record；
- 创建 slot/pool/worker；
- 增加 block-level public condition；
- 增加 synchronize/unload/cache clear；
- 改变 default MLX/Metal compiled graph 地址生命周期。

### 15.2 审计计数

为 default route 保留零计数断言：

~~~text
public_validator_calls = 0
public_catalog_queries = 0
public_probe_calls = 0
public_snapshot_compiles = 0
public_pool_creates = 0
public_worker_creates = 0
public_source_hashes = 0
~~~

计数器仅在 audit/test build 启用，release hot path 不做原子加法。

### 15.3 P0 门

每个受影响模型与默认 route 都必须通过：

- wall median ratio ≤ 1.02；
- wall P95 ratio ≤ 1.05；
- 输出相同或在既定质量等价门内；
- 默认 audit 零新增 pool/worker/hash/clear/unload；
- 置信区间和样本数量遵循文档 12。

## 16. 分批施工和提交边界

### R0：共享 validator 收口

- 修正 include 和 build list；
- planner/C API 共用函数；
- 完成 PUB-VAL-001…010；
- native-only、host、contract、App build 全绿。

### R1：C ABI 与 pure C++ resolution helper

- 下沉 orchestration；
- fake session/catalog unit tests；
- 完成 PUB-ABI-001…010；
- production catalog 保持为空。

### R2：Source lease 与 actual-plan result

- snapshot revalidation；
- common actual verifier；
- RunResult serialization；
- mismatch/fault/quarantine tests。

### R3：Swift/App

- StreamingChoice/OptionsStore；
- JobStore schema v2；
- resolve→persist→generate；
- RunInsights；
- 完成 PUB-APP-001…014。

### R4：模型 adapter

按 Z-Image → Flux 9B → H3 Turbo → LTX 顺序。每次只接一个模型，复用 private exact core，不加入 production record。

### R5：工具、校准和 reviewed record

按照 [34](34-model-tier-calibration-and-release-spec.md) 执行。一个 record 一个 evidence bundle、一个 review、一个可独立撤回的 catalog diff。

### R6：合并最新 dev

在 R0–R4 稳定后合并最新 dev；保留 dev 默认/ANE/M5 优化，同时保持 public v1 GPU-only。合并后重新做完整 P0 与 public fail-closed 回归。

## 17. 工程 Definition of Done

公共 runtime/App 框架只有在以下全部满足时才完成：

- [ ] 共享 validator 无错误依赖并通过全部错误优先级测试；
- [ ] resolve 无 GPU lock、无 payload、无 cancel reset；
- [ ] authority 不可序列化且 generate 内重新生成；
- [ ] source lease 在执行前和 reader open 时验证；
- [ ] actual plan mismatch 硬失败；
- [ ] public RunResult 信息完整；
- [ ] C ABI ownership/busy/null/cancel 全部通过；
- [ ] App 默认 Off，只展示 native options；
- [ ] exact resolve 在 persist 前完成，persist 失败不 generate；
- [ ] LTX worker-local resolution 完成；
- [ ] streaming off 路径 audit 为零；
- [ ] production catalog 仍可为空且默认功能完全正常；
- [ ] 四模型 adapter 和每个 target 的完成状态分别可追踪；
- [ ] 没有把未校准 target 显示为 available。

即使本清单全部完成，只能称为“public 框架 ready”；至少一个 reviewed record 加入 production catalog 并完成普通 App 重放后，才可以称为“public streaming 已有可用档位”。
