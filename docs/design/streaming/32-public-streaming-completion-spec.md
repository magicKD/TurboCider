# 32 · Public Streaming 收口实现、模型接入与验收规格

[目录](README.md) · [Runtime 代码设计](28-public-runtime-code-design.md) · [详细集成](30-public-streaming-detailed-integration.md) · [配置与校准](31-public-streaming-config-calibration-runbook.md) · [Runtime/App 工程规格](33-public-runtime-app-engineering-spec.md) · [模型档位与发布](34-model-tier-calibration-and-release-spec.md) · [当前进度](13-implementation-progress.md)

日期：2026-09-17。分支：<code>feat/stream</code>。状态：**以当前工作树为基线的收口实施规格；尚无 public record，App 尚未开放。**

本文不再重新设计第三套 streaming runtime，而是回答当前代码距离可公开使用还差什么，以及实现者应当按什么顺序修改、测试和发布。
本文是 28–31 的收口补充；若这些文档中的“当前状态”与本文冲突，以本文和 13 的最新进度为准。

共享 validator、C ABI、source lease、actual-plan、Swift/App/JobStore 和 LTX worker 的逐任务合同见
[33](33-public-runtime-app-engineering-spec.md)；四模型候选族、完整请求采样、swap 对照、evidence、reviewed record、
发布与撤回见[34](34-model-tier-calibration-and-release-spec.md)。32 保留总体验收边界，33/34负责可执行细节。

## 1. 最终结论

Public streaming 应采用以下产品与框架边界：

1. 用户只看到“关闭”以及 native 返回可用的 8、10、12、16、20 GiB 请求档位，不直接编辑 P/G/K/D/Q。
2. P/G/K/D/Q、pass transition、multi-pool policy 和 component release policy 属于 reviewed catalog record。
3. selector 只是用户意图；真正执行权限由 native resolver 根据 source、workload、runtime、device、layout 和证据即时铸造。
4. streaming off/absent 必须继续走当前默认路径，不访问 catalog、不 probe、不 hash source、不创建线程或 slot。
5. public v1 只支持 GPU。MiniMax H3 只支持 H3 Turbo original BF16；Flux 只支持 Klein 9B BF16 eager；Z-Image 只支持 original BF16；LTX 只支持首批明确冻结的 LTX 2.5 workload card。
6. production catalog 在完整请求内存、质量、生命周期和性能证据完成前保持为空。框架代码可完成，但不能因此宣称任一档位已公开。
7. streaming 通常会比“内存不足后由系统被动 swap”更可控，在严重压力下也很可能更快；但它会增加显式权重读取，在内存充足时通常不应替代 resident。该结论必须通过 P3/swap 对照验证，不能仅凭原理发布速度声明。

## 2. 当前代码基线

### 2.1 已经实现

| 层 | 当前能力 | 代码位置 |
|---|---|---|
| Wire | schema-v2 selector、五档 target、profile/request 原子覆盖、严格十进制 byte 解析 | <code>streaming_contracts.hpp</code>、<code>streaming_config.mm</code>、<code>json_keys.cpp</code> |
| Catalog | 结构化 source/workload/runtime/device/plan/calibration/performance/release record；production catalog 为空 | <code>preset_catalog.hpp/.cpp</code> |
| Canonical identity | typed、length-prefixed canonical encoder；record/source/workload/runtime/device/resolution digest | <code>canonical_encoding.hpp/.cpp</code>、<code>resolved_request.cpp</code> |
| Resolver | deterministic select、exact replay、authorize、不可从 JSON 构造的 non-copyable authority | <code>preset_resolver.hpp/.cpp</code> |
| Session contract | probe、compile snapshot、generate_resolved 三个默认拒绝 hook | <code>native/runtime/session.hpp</code> |
| Engine identity | model ID、normalized model root、execution container、streaming quarantine | <code>native/api/c_api.mm</code> |
| Exact C ABI | <code>tc_engine_resolve_streaming_json</code>；active generate resolve 后只调用 <code>generate_resolved</code> | C header、<code>c_api.mm</code> |
| Swift | resolution/result 类型、error envelope decoder、<code>NativeEngine.resolveStreaming</code> | <code>TurboCiderNative.swift</code> |
| Safety gate | 空 catalog 时 resolve/generate 在普通 session/GPU 执行前失败；active prepare 明确不支持 | C API 与四模型 candidate gate tests |
| Data plane | layout compiler、StageExecutor、slot safety、bounded I/O、multi-class/multi-pool、MLX pager | 现有 streaming runtime |
| Private adapters | LTX、H3 Turbo、Z-Image、Flux 9B 均已有 candidate-only exact execution 和同布局性能锚点 | 各模型 adapter |

### 2.2 仍未完成

| 缺口 | 为什么阻断 public |
|---|---|
| selector 冲突/路由检查与空 catalog 的错误优先级尚未冻结 | 当前空 catalog 可能遮蔽更准确的 config/route 错误 |
| C ABI null/busy/ownership/cancel/error envelope 测试不完整 | 新接口尚未形成稳定可依赖合同 |
| execution-time source lease revalidation 尚无统一接口 | authority 目前主要核对 immutable identity，尚未完整证明执行使用同一未变化 source |
| RunResult 尚无 public selection/actual-plan 完整结果 | App、evidence 和 reviewer 无法从结果确认“授权计划就是实际计划” |
| App advanced setting、options state、job v2 transaction 尚未实现 | 用户侧还不能安全 opt-in |
| LTX worker-local final resolution 尚未实现 | App 进程不能替独立 worker 铸造 authority |
| 四模型未 override public probe/compile/generate_resolved | 当前 exact 路线仍是 private candidate 权限 |
| full-request memory sampler、verifier、catalog builder 尚未实现 | 不能证明任一 target 满足完整请求内存目标 |
| reviewed records 为零 | production catalog 必须继续为空 |
| 当前 <code>dev@02148b7</code> 尚未再次合并 | M5/ANE/default 兼容性还没有在最终 public 控制路径上复核 |

### 2.3 三种“完成”必须分开

~~~text
框架完成
  = wire + resolver + authority + C ABI + App transaction + common result verification

模型 adapter 完成
  = public probe + immutable snapshot + source lease + generate_resolved
    + actual-plan verification + lifecycle/fault/quality

单个 public 档位完成
  = adapter 完成 + full-request calibration + P0/P1/P4
    + ordinary App/worker replay + independent review + reviewed record
~~~

任何 private candidate P1、denoiser-only peak、一次成功请求或 metadata estimate 都不能替代第三层完成。

## 3. 端到端执行架构

### 3.1 Off/default 路线

~~~text
parse request
  → selector absent/disabled
  → 原有 engine lock / global GPU lock / DeviceLease
  → 原有 session.generate
  → 原有 result
~~~

该路线不得构造 public catalog、resolver、probe、snapshot、authority、metadata cache、slot pool 或 sampler。新增代码审查时，
应把“off 分支发生在第一个 public helper 之前”作为结构性要求，而不是依赖编译器优化。

### 3.2 Public resolve

~~~text
App/worker frozen v2 request
  → engine try-lock
  → parse/profile/selector validation
  → engine model identity check
  → cheap public conflict/route validation
  → production catalog availability
  → full request/model validation
  → metadata-only model probe
  → deterministic preset select
  → immutable model snapshot compile
  → layout/source/runtime/device authorization
  → exact selector + resolution summary
~~~

Resolve 不取得 global GPU lock，不创建 DeviceLease，不 configure streams，不 materialize payload，不创建 pool/worker，也不重置 generation cancel flag。

### 3.3 Public generate

~~~text
active memory-tier/exact selector
  → engine try-lock
  → same exact resolve path
  → acquire global GPU lock
  → DeviceLease
  → catalog/source/runtime/device revalidation
  → configure resolved backend
  → session.generate_resolved
  → drain
  → compare actual plan with authorized plan
  → attach public result metrics
~~~

App 在 generate 前调用 resolve 是为了展示与持久化 exact selector；真正 authority 仍必须在 generate 调用内部重新生成。不能从 App 传回 authority，
也不能添加可序列化的 authorization token。

## 4. 共享 public 请求校验与错误优先级

### 4.1 当前问题

<code>resolve_public_streaming_locked</code> 当前先检查空 production catalog，再调用 <code>make_plan</code>。因此一个同时包含
legacy memory budget、GPU+ANE 或 compile route 的 active selector 请求，可能先收到
<code>catalog_has_no_public_records</code>，而不是更直接的 conflict/route 错误。

直接把完整 <code>make_plan</code> 移到空 catalog 之前也不理想：

- 它会调用 model recipe/module validation；
- 合成 gate fixture 可能先因模型 shape 或缺少真实文件失败；
- options/resolve 会不必要地执行与 catalog availability 无关的工作；
- API 与 planner 容易继续形成两套错误顺序。

### 4.2 新增纯函数

建议新增：

~~~cpp
// native/runtime/streaming/public_request_validation.hpp
namespace tc::streaming {

struct PublicRequestValidation {
    const StreamingSelector *selector = nullptr;
    std::string execution_container;
};

PublicRequestValidation validate_public_streaming_request(
    const Request &request,
    std::string_view execution_container);

} // namespace tc::streaming
~~~

实现文件不得 include Metal、MLX、catalog、model session 或 filesystem。该函数只做：

1. selector 存在、active、schema/selection/retention/target 合法；
2. manual layout 与 selector 互斥；
3. legacy residency、memory budget、streaming_offload 与 public selector 冲突；
4. <code>memory_constrained.enabled</code> 与 public v1 冲突；
5. execution 必须严格为 GPU，ANE manifests 必须为空；
6. approximation、compiled graph、LoRA、quantized cache 等 v1 未认证特性必须拒绝；
7. execution container 必须由 engine/worker 提供，不能由请求覆盖；
8. 不调用 <code>module_for</code>，不检查模型文件，不 tokenize，不读取 catalog。

<code>make_plan</code> 的 active-selector 分支调用同一个函数；C API exact resolve 也调用同一个函数。删除 planner 中重复的 conflict/route 条件，
避免两个入口产生不同错误。

### 4.3 冻结错误优先级

建议 public resolve/generate 使用下表顺序：

| 优先级 | 类别 | 例子 |
|---:|---|---|
| 1 | ABI 指针/输出所有权 | missing engine/input/output |
| 2 | engine terminal state | streaming quarantined |
| 3 | engine busy | engine busy |
| 4 | JSON/schema/duplicate key | invalid JSON、unknown field、decimal lexical |
| 5 | selector intent | unsupported target、incomplete preset selector |
| 6 | engine/request identity | engine model mismatch |
| 7 | public conflict | legacy budget/manual/memory guard conflict |
| 8 | public route | GPU+ANE、approximation、compiled graph、LoRA |
| 9 | production feature availability | empty catalog |
| 10 | complete model request validation | shape/steps/operation/model constraints |
| 11 | artifact/workload/device/runtime | source、token shape、container、device mismatch |
| 12 | layout compile/authorize | no fit、layout mismatch、stale/revoked |

这样空 catalog 仍然在 source probing 和模型执行之前 fail-closed，同时不会遮蔽用户能够立即修正的 selector conflict。

### 4.4 必须新增的错误顺序测试

| ID | 输入 | 期望 |
|---|---|---|
| PUB-VAL-001 | active selector + legacy budget + empty catalog | config conflict，不是 empty catalog |
| PUB-VAL-002 | active selector + GPU+ANE + empty catalog | route unsupported |
| PUB-VAL-003 | active selector + compile_gpu + empty catalog | route unsupported |
| PUB-VAL-004 | valid selector/route + empty catalog | catalog_has_no_public_records |
| PUB-VAL-005 | wrong engine model + empty catalog | engine model mismatch |
| PUB-VAL-006 | malformed selector target | selector validation error |
| PUB-VAL-007 | selector disabled + legacy request | 与旧 plan/result 相同 |
| PUB-VAL-008 | planner 与 resolve 对同一 conflict | error code 相同 |

## 5. Resolver、authority 与 source 生命周期

### 5.1 Authority 绑定

当前 authority 已绑定 record、source、workload、runtime、layout、component policy 和 device digest。保持以下规则：

- 构造函数 private，仅 <code>PublicPresetResolver</code> 可创建；
- 不可 copy，不可从 JSON deserialize；
- <code>expected_resolution_digest</code> 只做 stale/replay 检查；
- private candidate 的 <code>allow_experimental_streaming</code> 不能被转换成 public authority；
- exact selector 重放失败时不自动挑选同 target 的新 record；
- authority 不能跨进程；LTX worker 必须自己 resolve。

### 5.2 Snapshot 增加 source revalidation 合同

仅比较 snapshot 中保存的 source digest 不足以证明执行前 source 没有变化。建议扩展：

~~~cpp
struct SourceRevalidation {
    PresetSourceIdentity identity;
    bool unchanged = false;
    std::string diagnostic;
};

class ModelStreamingSnapshot {
  public:
    virtual SourceRevalidation revalidate_source() const = 0;
    virtual bool uses_same_execution_source() const noexcept = 0;
};
~~~

模型 snapshot 必须持有真实执行将使用的 fd/mapping/manifest lease，而不是只保存 path。执行前：

1. 对持有的 fd 做 device/inode/size/mtime/ctime 或模型等价 quick validation；
2. 核对安装期 verified manifest/content identity；
3. 确认 generate_resolved 不会按 path 重新打开另一份文件；
4. snapshot/source lease 生命周期覆盖 I/O workers、GPU last readers 和 drain；
5. revalidation 失败返回 <code>artifact_changed</code>，不重新 probe 并静默换 source。

### 5.3 Catalog 与 runtime revalidation

取得 global GPU lock 后复核：

- production catalog revision 和 record digest；
- record 未 revoked；
- device identity 未改变；
- runtime/reader/kernel/allocator revision 与 authority 一致；
- exact selector 中 expected resolution digest 一致；
- source quick identity 未变化；
- engine/session/model/container 未变化。

临时 memory pressure 只能返回 retryable 的 admission 错误，不能临时缩 K、改 P 或换 record，否则实际计划将不再等于 reviewed record。

## 6. C ABI、Swift 与结构化错误

### 6.1 C ABI 完成项

<code>tc_engine_resolve_streaming_json</code> 还需要补齐：

- null engine/request/result/error combinations；
- result/error 在所有返回路径先置空；
- 成功 result 与失败 error 都由 <code>tc_string_free</code> 独立释放；
- busy engine 不阻塞；
- resolve 不重置 <code>cancelled</code>；
- resolve 不取得 global GPU lock，不创建 DeviceLease；
- release dylib 不导出 test-only mint/lifecycle symbols；
- unknown exception 返回稳定 public error envelope。

### 6.2 只对 opt-in public 路线增加稳定 error envelope

建议增加：

~~~cpp
enum class PublicStreamingErrorCode {
    invalid_request,
    engine_busy,
    streaming_quarantined,
    streaming_engine_model_mismatch,
    streaming_config_conflict,
    streaming_route_unsupported,
    catalog_has_no_public_records,
    unsupported_model_or_format,
    unvalidated_workload,
    artifact_verification_required,
    artifact_changed,
    unvalidated_device,
    no_preset_fits_target,
    streaming_resolution_stale,
    preset_revoked,
    streaming_public_adapter_unsupported,
    streaming_actual_plan_mismatch,
    streaming_drain_failed
};
~~~

Native error JSON：

~~~json
{
  "schema_version": 1,
  "code": "catalog_has_no_public_records",
  "message": "当前版本尚无通过审核的 Streaming 档位。",
  "retryable": false,
  "action": "disable_streaming"
}
~~~

旧 v1/off 路线继续允许纯字符串，避免一次性改变全库错误 ABI。Swift 已能先 decode envelope、失败后回退 legacy message；下一步要让 native
新路线实际输出 envelope，并为每个 stable code 建立映射测试。

### 6.3 Swift semantic parity

当前 <code>NativeRequestV2</code> 仍是 public streaming 专用骨架。必须增加 v1/v2 semantic snapshot 测试，至少核对：

- model、operation、prompt/media inputs、strength；
- output path/kind/width/height/frames/fps/audio；
- seed、steps、dynamic text、noise；
- execution/profile/ANE manifests/approximation/compile；
- LTX backend/fast AV/Sol/text rows/video attention；
- LoRA paths/roles/strength/strategy；
- explicit false/0 与 absent；
- selector off/memory-tier/exact preset。

字段缺失不能通过“首批 record 不支持该特性”来掩盖；应先保证两种 request 表达语义一致，再由 public validator 明确拒绝未认证特性。

## 7. Result 与实际计划验证

### 7.1 新增 public result 类型

建议在 <code>session.hpp</code> 增加：

~~~cpp
struct PublicStreamingSelectionMetrics {
    uint64_t target_request_memory_bytes = 0;
    uint64_t calibrated_request_bytes = 0;
    std::string memory_scope;
    std::string preset_id;
    uint32_t preset_revision = 0;
    std::string catalog_revision;
    std::string record_digest;
    std::string resolution_digest;
    std::string source_digest;
    std::string workload_digest;
    std::string runtime_digest;
    std::string device_digest;
    std::string authorized_layout_digest;
    std::string actual_layout_digest;
    std::string component_policy_revision;
    std::string release_channel;
    std::string execution_container;
    bool actual_plan_verified = false;
};
~~~

<code>RunResult</code> 增加 optional 字段，<code>results.mm</code> 以 additive 的
<code>public_streaming</code> object 输出。不得输出 raw model path、inode 列表、prompt 或 authority 内部值。

### 7.2 Common actual-plan verifier

当前 <code>StreamingRuntimeMetrics</code> 适合 private same-layout 性能统计，但 public 需要通用核对：

~~~cpp
void verify_public_streaming_actual_plan(
    const ResolvedRequestExecution &resolved,
    const RunResult &result);
~~~

至少比较：

- implementation family；
- layout digest；
- stage/pool class 和顺序；
- P/G/K/D/Q；
- group/pass count；
- pass transition、retention、multi-pool policy；
- reader/kernel/weight format revision；
- conditioning/upsample/component boundary；
- source/runtime/device digest；
- request generation 和 terminal drain state。

如果模型结果只包含单 stage 字段，应扩展成 per-stage/per-pool 数组，不能仅凭一个 layout digest 跳过 runtime counters。任何不一致都返回
<code>streaming_actual_plan_mismatch</code>，不能把请求标记成功后只写 warning。

## 8. App 高级设置与任务事务

### 8.1 App 状态模型

建议新增：

~~~swift
enum StreamingChoice: Codable, Sendable, Equatable {
    case off
    case recommended
    case target(UInt64)
}

struct StreamingDraftState: Sendable {
    var choice: StreamingChoice = .off
    var options: NativeStreamingOptions?
    var resolution: NativeStreamingResolution?
    var queryRevision: UInt64 = 0
    var isQuerying = false
    var error: NativeFailure?
}
~~~

默认必须是 <code>.off</code>。旧 draft decode 不存在该字段时也必须得到 off，不能因机器内存或历史 Z-Image streamed 设置自动开启 public selector。

### 8.2 Options 查询

OptionsStore 的 key 至少包含：

~~~text
model ID + model root installation identity
operation/shape/frames/fps/steps/audio
token bucket/template/features
execution container
device class
catalog revision
~~~

UI 修改会影响 workload identity 的字段后，采用 150–300 ms debounce，并给每次 query 单调 revision。旧 revision 返回时不得覆盖新状态。
关闭 streaming 后取消展示结果，但不需要强制取消正在执行的纯 metadata query。

### 8.3 物理内存与档位推荐

8/10/12/16/20 GiB 是“完整请求内存目标”，不是独立显存容量，也不是物理内存 SKU。Apple GPU 使用 unified memory，App 不能简单做
“16 GB 机器选择 16 GiB target”。

推荐必须分两层：

1. native/catalog 判断 record 是否对 exact device/workload 可用；
2. App 只在 available records 中展示 native 给出的 recommended 标记。

建议 options response 增加：

~~~json
{
  "recommendation": {
    "target_request_memory_bytes": 10737418240,
    "reason_code": "device_policy_recommended",
    "system_reserve_bytes": 4294967296
  }
}
~~~

推荐 policy 必须由 reviewed device profile 产生，并留出系统/App/driver reserve。没有 verified device recommendation 时，App 可展示 available
档位，但默认保持 off，不自行用当前 free memory 推导 target。

### 8.4 Resolve → persist → generate

~~~text
freeze draft and seed
  → build NativeRequestV2(memory-tier)
  → engine.resolveStreaming
  → replace selector with exact_selector
  → persist versioned job + resolution summary atomically
  → generate exact job
  → native re-resolve and execute
~~~

persist 失败时 generate 次数必须为零。generate stale/revoked 时，JobStore 保留原 job 和 error，不自动改 target 或回 resident。

### 8.5 Job envelope v2

~~~json
{
  "schema_version": 2,
  "request_schema_version": 2,
  "created_by": "TurboCider",
  "streaming": {
    "requested_selector": {},
    "exact_selector": {},
    "resolution_summary": {},
    "resolved_at": "..."
  },
  "request": {}
}
~~~

旧 job 无损 decode 为 legacy/off；未知新版本只读展示并拒绝覆盖。保存使用临时文件 + fsync + atomic rename。不要把 native authority 或 model
本机绝对路径复制进可分享的证据摘要。

### 8.6 LTX worker

LTX 的 final resolution 必须在真实 <code>ltx_cli_worker</code> container 完成。推荐首版使用两阶段进程：

~~~text
App → worker --resolve-only
worker constructs engine, probes ltx_cli_worker identity, returns exact selector, exits
App atomically persists job
App → fresh worker --generate-exact
worker reconstructs engine, re-resolves exact selector, generates
~~~

resolve-only worker 禁止创建 GPU backing。若未来改为长驻 ACK 协议，也必须在 App 确认持久化成功前不进入 GPU execution。

## 9. 通用模型 adapter 合同

### 9.1 不新增第二套 executor

四模型 public adapter 必须复用现有 descriptor、layout compiler、StageExecutor、SlotPool、IoExecutor、MLX pager 和 candidate exact kernel。
public 工作只增加：

- exact metadata projection；
- source/runtime/workload identity；
- immutable snapshot/source lease；
- authority 检查；
- candidate/public 共用的 exact execution helper；
- actual result verification。

不得为每个模型复制 target→P/G/K/D/Q 逻辑，也不得在 block hot path查询 catalog。

### 9.2 推荐共享 helper

~~~cpp
template <class Snapshot>
std::shared_ptr<const Snapshot> checked_public_snapshot(
    const ResolvedRequestExecution &,
    std::string_view expected_model);

void validate_public_execution_common(
    const ResolvedRequestExecution &,
    const ModelStreamingSnapshot &);

void attach_public_streaming_result(
    RunResult &,
    const ResolvedRequestExecution &);
~~~

共享 helper 只检查身份、authority、actual plan 和结果，不知道模型 tensor 名称或 kernel。模型特定 helper 负责 source lease、PlanView 和 execution owner。

### 9.3 Candidate 与 public 共用执行核心

每个模型重构成：

~~~text
private manual candidate
  → compile candidate snapshot
  → run_exact_streaming(snapshot, PrivateCandidateAuthority)

public resolved route
  → checked public snapshot
  → run_exact_streaming(snapshot, StreamingAuthority)
~~~

两种 authority 不能相互构造，但必须进入同一 exact execution core，从而避免 public 和 P1 已验证 candidate 在 kernel/slot 调度上漂移。

## 10. 四模型逐项接入

### 10.1 LTX 2.5

首批范围：

- dense T2V、C/Metal；
- 512×320×33、11 steps、24 fps；
- no audio、I2V、LoRA、Sol、ANE；
- execution container = <code>ltx_cli_worker</code>。

主要修改：

| 文件 | 修改 |
|---|---|
| <code>ltx_session.mm</code> | override public probe/compile/generate_resolved；把 candidate exact owner 抽成共用 helper |
| <code>ltx_streaming_descriptor.*</code> | 输出 verified artifact/source/workload/runtime identity |
| <code>ltx_streaming_plan.*</code> | public snapshot 持有 metadata/header/fd、Layout、PlanView、component policy |
| <code>ltx_streaming_adapter.inc</code> | 不区分 candidate/public 调度，只消费 validated plan |
| <code>LTXWorker.swift</code> | resolve-only 与 generate-exact 两阶段协议 |

专项验收：

- worker 内 resolve，embedded App resolution 不可用于 worker；
- exact source fd 在 compile 到 destroy 期间保持同一；
- Stage1→upsampler→Stage2→VAE/export 完整请求；
- cancel first fill、mid-stage、VAE/export failure、destroy retry/quarantine；
- actual layout、517 类 fill/counter、component boundary 与 record 一致；
- full process-tree calibration 包含 worker launch/exit 和 export。

### 10.2 MiniMax H3 Turbo original BF16

首批范围：

- 只支持 <code>minimax-h3-turbo</code> original BF16；
- GPU native Metal；
- 512×512×22、4 steps、无 audio/LoRA/ANE；
- 不包含普通 H3、VDN、FastH3、MLX 或量化变体。

主要修改：

| 文件 | 修改 |
|---|---|
| <code>h3_session.mm</code> | public hooks；candidate/public 共用 H3 exact execution spec |
| <code>h3_streaming_descriptor.*</code> | verified BF16 source、39 blocks、四 pass/workload identity |
| <code>h3_streaming_policy.*</code> | 保持 active block 与 canonical layout 同源 |
| H3 runtime schedule | actual carry-first-group、pass/group counter 输出 |

专项验收：

- P/G/K/D/Q 与 record 一致；
- K2/G1 carry-first-group 每 pass 每 suffix group恰好读取一次；
- 四 pass 无跨 generation completion；
- cancel/fill/fence/destroy 故障 drain；
- 当前约 1% private P1 波动可作为工程锚点，但 public 仍需 ordinary constructor 和 full request 证据。

### 10.3 Z-Image Turbo original BF16

首批范围：

- original BF16 safetensors；
- T2I 512²、9 steps、GPU eager；
- no GGUF、LoRA、ANE。

主要修改：

| 文件 | 修改 |
|---|---|
| <code>z_image.hpp/.cpp</code> | public hooks；抽取 <code>ZImageExactStream</code> 共用入口 |
| <code>streaming_descriptor.hpp</code>、<code>z_image_streaming_descriptor.mm</code> | source/token/workload/runtime identity |
| <code>weight_stream.hpp</code>、<code>z_image_weight_stream.mm</code> | snapshot source lease、actual reads/fills |
| <code>results.mm</code> | public selection 与 actual plan |

首轮固定 G1/K2/D0/Q1，只扫描 prefix，降低搜索维度。必须验证：

- 30 blocks，suffix 必须至少容纳 K2；P29 非法；
- token valid/padded/compute rows进入 workload identity；
- text encoder release 和 VAE peak纳入完整内存；
- legacy <code>residency=streamed + memory_budget</code> 与 public selector 完全分离；
- 512² record 不复用于 1024²。

### 10.4 Flux.2 Klein 9B BF16 eager

首批范围：

- Klein 9B BF16 eager T2I；
- 512²、4 steps、GPU；
- no 4B compiled graph、LoRA、ANE。

主要修改：

| 文件 | 修改 |
|---|---|
| <code>flux.hpp</code>、<code>pipeline.cpp</code> | public hooks；共用 <code>FluxExactStream</code> |
| <code>streaming_descriptor.hpp</code>、<code>flux_streaming_descriptor.mm</code> | dual/single class descriptor 与 source identity |
| <code>flux_transformer.cpp</code> | actual multi-pool/class barrier/counter 输出 |
| MLX pager | source lease、reader completion、pool retention evidence |

专项验收：

- dual 0–7、single 8–31 的 class 顺序；
- retain_all 两 pool、K2/G1/D1/Q2；
- dual→single→dual 跨 pass 不重建 pool；
- block 8 concatenate 每 pass恰一次；
- current private same-layout generic/direct P1 作为框架锚点；
- current resident→streaming peak下降不能替代 8/10/12/16/20 GiB full-request qualification。

## 11. Multi-slot / multi-pool 调度设计

### 11.1 Slot 状态机

~~~text
Vacant
  → Filling(generation, content)
  → Ready
  → Claimed(group, pass)
  → Reading(sealed reader set)
  → Retiring(last readers)
  → Vacant(next generation)
~~~

Slot 的 backing 生命周期和 content generation 必须分离。Vacant 只表示可覆盖内容，不表示 backing 已释放。跨 generation completion、重复 claim、
reader 未完成就 refill 都是 correctness failure。

### 11.2 参数语义

- P：resident prefix，减少每 pass I/O，但提高常驻峰值；
- G：每次 refill 的 block group，增大可降低调度开销，但增大 slot capacity 和尾部浪费；
- K：slot 数，决定可并行驻留的 streamed groups；
- D：lookahead 距离，必须小于 K；
- Q：I/O worker 数，首版不超过 K。

Public v1 的这些值完全固定在 record 中。runtime 不根据瞬时 SSD 延迟或 free memory动态改变参数。

### 11.3 K2 标准时序

~~~text
owner fill g0 synchronously
owner claim g0
I/O   fill g1 into slot1
GPU   encode g0 from slot0
owner retire g0 after all reader fences
owner claim g1
I/O   fill g2 into slot0
GPU   encode g1 from slot1
...
~~~

overlap 的目标是使：

~~~text
steady group time ≈ max(GPU compute time, next-group fill time)
~~~

而不是 compute + fill。若 fill 长期大于 compute，应优先检查 physical I/O、group size、Q、page cache 和 source layout；盲目增加 K 会提高峰值，
不一定降低 wall。

### 11.4 Multi-pool

Flux dual/single 采用 retain_all 时：

- request setup 一次性创建两个 pool；
- class barrier 只 drain/readjust active pool，不销毁 backing；
- request结束统一销毁；
- 峰值包含两个 pool 之和；
- steady allocation/thread create 必须为零。

serial policy 则在 class barrier 后释放旧 pool再创建新 pool，峰值更低但可能增加 allocation 和 barrier cost。policy 是 layout identity，
不能在 runtime 自动切换。

## 12. 内存模型与档位选择

### 12.1 估算只用于候选剪枝

对一个 compiled plan，静态估算至少包含：

~~~text
M_weight =
  resident prefix backing
  + sum(active pool slot capacities)
  + resident helper/component weights

M_request_upper =
  M_weight
  + activation upper by workload
  + text/conditioning live intervals
  + VAE/upsampler/audio/export live intervals
  + allocator/cache/driver reserve
  + process/container overhead
~~~

静态 upper 可排除明显不可能候选，但 public eligibility 必须来自完整请求 process-tree 实测与独立 upper 的较大值。

### 12.2 Target 判定

保持整数规则：

~~~text
H(T) = max(512 MiB, ceil(10% × T))
M_catalog = max(M_confirmed_full_request_peak, independent_upper_bound)
eligible(T) = calibration.complete && M_catalog + H(T) <= T
~~~

这不是 macOS hard cap，也不保证系统 swap delta 恒为零；它表示该 record 在指定 workload/device/container 的完整请求证据满足目标与内部余量。

### 12.3 候选选择

同一 target 的候选按：

1. exact identity；
2. public release 且未 revoked；
3. calibration complete；
4. target fits；
5. reviewed performance rank；
6. calibrated bytes；
7. logical read bytes；
8. stable preset ID/revision。

排序不读取瞬时 free memory、温度或 page cache 状态，保证同一 identity 可重复 resolve。

## 13. Streaming 与系统 swap 的比较

### 13.1 原理判断

显式 block/slot streaming 相对被动 swap 的潜在优势：

- 权重读取顺序由模型 block 顺序决定，可做连续 pread 和 bounded prefetch；
- backing 数量有界，不依赖 VM 随机回收哪些页；
- GPU 尚在读取的 slot 受 fence/generation保护；
- 可以在 component/pass 边界主动释放；
- I/O 与 GPU compute 可重叠；
- failure/cancel 可以由 runtime drain，而不是等待系统压力自行缓解。

潜在劣势：

- 每个 pass 可能重复读取 suffix 权重；
- 小模型或内存充足时 resident 无 page fault，streaming 额外 I/O 必然可能更慢；
- SSD、文件缓存、热状态和其他进程会影响 refill；
- 过小 target 可能迫使 P 下降，读取量和 wait显著增加；
- macOS compression/swap 可能在某些 workload 上比预期高效。

因此产品策略应是：默认 resident/off；用户明确选择并且 catalog 有证据时才启用 streaming。

### 13.2 可证伪实验

每个 workload 固定 artifact、prompt/seed、shape、steps、container，运行：

| 路线 | 目的 |
|---|---|
| resident / normal memory | 内存充足性能与质量基线 |
| public exact streaming / normal memory | 显式 I/O 的纯代价 |
| resident / controlled pressure | 系统 compression/swap 行为 |
| public exact streaming / same pressure | 低内存稳定性与速度 |
| legacy streamed（若有） | 兼容路线参考 |

压力工具必须在独立 harness 中显式启动，普通测试不能悄悄制造内存压力。报告：

~~~text
wall/denoise median and P95
process-tree physical footprint peak
MLX/Metal accounted peak
swap in/out delta
compression and page-fault delta
logical/physical I/O
GPU idle and refill wait
thermal/pressure state
quality/failure/cancel/quarantine
~~~

只有 streaming 相对 pressure-resident 的 wall speedup 95% 区间下界大于 1，才能声明“比 swap 更快”。否则应如实表述为
“峰值更低/成功率更高，但速度代价为 X”。

### 13.3 简化 break-even 模型

用于模拟器排序，而非发布证明：

~~~text
T_stream ≈ T_fixed + sum_groups max(T_gpu_group, T_fill_group)
           + uncovered_startup + barriers + decode/export

T_swap ≈ T_resident_no_pressure
         + major_fault_stall + compression_cpu + swap_io + retry/jitter
~~~

当 swap fault stall 和抖动大于 streaming 未被 overlap 的显式 fill 时，streaming 才有速度优势。模拟器必须用实测 read/GPU/fence 分布校准，
不能只用 SSD 标称吞吐。

## 14. 工具链设计

| 工具 | 输入 | 输出 | 是否进入 release |
|---|---|---|---|
| candidate inspector | model metadata + workload card | 合法 P/G/K/D/Q 候选与 rejection | 否 |
| layout compiler CLI | descriptor + candidate | canonical compiled layout/digest | 可共享 host code，CLI 否 |
| schedule simulator | layout + measured distributions | peak/read/wait/wall prediction | 否 |
| campaign runner | frozen policy + binary/artifact | raw per-request events | 否 |
| process-tree sampler | pid tree + phase events | footprint/swap/pressure samples | 否 |
| independent verifier | raw bundle | P0/P1/P3/P4、coverage、quality | 否 |
| catalog builder | verified bundle + review | proposed/reviewed record | 否 |
| production catalog | reviewed records only | runtime immutable index | 是 |

所有工具生成文件都带 schema、tool revision、git commit、binary digest、artifact identity、device/OS/container 和 SHA-256 manifest。
builder 默认只能生成 proposed，缺 independent review 时不得输出 public channel。

## 15. 验收矩阵

### 15.1 Host/config/resolver

- canonical C++/Python golden；
- absent/false/0、map order、UTF-8、overflow；
- selector conflict/error priority；
- deterministic rank；
- stale/revoked/source/runtime/device/layout mismatch；
- authority non-copy/non-serializable；
- empty catalog complete fail-closed。

### 15.2 C ABI/Swift/App

- null/ownership/busy/cancel；
- resolve 零 GPU lock/DeviceLease/session execution；
- structured error round-trip；
- v1/v2 semantic parity；
- options debounce/revision；
- off 不 query/resolve；
- resolve→persist→generate transaction；
- persist failure 不 generate；
- old/unknown job schema；
- LTX worker-local resolution。

### 15.3 Runtime/lifecycle

- allocation、first fill、mid-fill short read；
- fence timeout、mailbox overflow；
- cancel before dispatch/during prefetch/during GPU read；
- component/VAE/export failure；
- unsafe drain quarantine；
- repeated engine request；
- resident→public→off transition；
- active readers/workers/tickets/fds 为零。

### 15.4 模型与性能

每个模型至少通过：

1. private candidate 与 public ordinary constructor 的 exact-plan parity；
2. fixed seed quality；
3. same-layout framework P1；
4. selector-off/default P0；
5. full-request memory calibration；
6. strategy P4；
7. 若声称优于 swap，再通过 P3；
8. actual authorized plan verification；
9. ordinary App 或真实 worker replay。

默认性能门继续使用：

~~~text
wall median ratio <= 1.02
wall P95 ratio <= 1.05
对应置信区间不越门
~~~

selector-off audit 还必须证明 catalog access、source hashing、new pool/thread/sampler、cache-clear/unload、额外 synchronization 全部为零。

## 16. 分阶段代码实施

### C1：收紧 exact runtime 合同

修改：

- 新增 <code>public_request_validation.hpp/.cpp</code>；
- <code>plan.cpp</code> 和 <code>c_api.mm</code> 共用 validator；
- 补 C ABI null/busy/ownership/cancel/error priority tests；
- native 输出 public error envelope；
- snapshot source revalidation interface；
- RunResult public metrics 与 common actual verifier。

完成门：host/contract/native-only/App build 全通过；production catalog 仍为空；off audit 无新增工作。

### C2：App 控制面

修改：

- <code>StudioState.swift</code>：StreamingChoice；
- 新增 <code>StreamingOptionsStore.swift</code>；
- <code>AccelerationView.swift</code> 或高级设置入口：target picker；
- <code>JobStore.swift</code>：job v2 和原子事务；
- <code>RunInsights*</code>：public selection/actual/memory scope；
- <code>LTXWorker.swift</code>：worker-local resolve。

完成门：catalog-empty App 安全可用；默认 off；旧 job 无损；persist failure 不生成。

### C3：四模型 public adapter

顺序建议：

1. Z-Image：单 component、现有 exact stream 简单，先验证通用 public helper；
2. Flux 9B：验证 multi-pool；
3. H3 Turbo：验证 carry-first-group/multi-pass；
4. LTX：最后验证 worker/container 和完整 component pipeline。

每个 adapter 独立提交，production catalog 仍为空。不得在同一提交加入 record。

### C4：工具与校准

实现 inspector、simulator、process-tree sampler、campaign runner、verifier、builder。先生成 proposed/rejected evidence，不修改 production catalog。

### C5：单 record 发布

每次只发布一个 model/workload/device/target record。record-only 提交不改 kernel/executor/App default。先 public-experimental，完成 staged replay
和 revoke drill 后再决定 stable。

### C6：合并最新 dev 与 GPU/ANE 回归

在 C1 默认路径防线与 App off-by-default 完成后，再合并当前 <code>dev@02148b7</code>。冲突处理原则：

- 保留 dev 的 M5/ANE 优化与默认 resident 路线；
- 保留 public selector 的 GPU-only fail-closed 边界；
- selector off 重跑 GPU、ANE、Core ML cache、Z-Image legacy streaming；
- active public selector + GPU+ANE 仍返回 route unsupported；
- 不因合并把 production catalog 填入未复核记录。

## 17. Release checklist

- [ ] shared public request validator 已替代重复逻辑；
- [ ] error priority 和 stable envelope 有测试；
- [ ] resolve ABI null/busy/ownership/cancel 完整；
- [ ] source lease 与 execution-time revalidation 完整；
- [ ] RunResult public selection 和 actual-plan verification 完整；
- [ ] App 默认 off、options、target、job v2、RunInsights 完整；
- [ ] LTX worker-local final resolution 完整；
- [ ] 四模型 public hooks 完整；
- [ ] full-request process-tree sampler/verifier/builder 完整；
- [ ] 至少一个 reviewed record 的 P0/P1/P4/quality/lifecycle/memory 完整；
- [ ] 若宣传优于 swap，P3 置信结论完整；
- [ ] latest dev merge 后 GPU/default/ANE 回归通过；
- [ ] ordinary App/worker resolve→persist→generate 通过；
- [ ] revoke/replay drill 通过；
- [ ] selector-off audit 为零新增热路径工作。

只有全部满足某一个具体 record 的相关项后，App 才可把该档位从 unavailable 改为 available。框架完成、private adapter 完成和 public record
完成必须继续分别报告。
