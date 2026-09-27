# 28 · Public Streaming Runtime 代码落地设计

[目录](README.md) · [端到端蓝图](27-public-streaming-delivery-blueprint.md) · [代码原则](25-public-preset-implementation-spec.md) · [实施与发布验收](29-public-implementation-and-acceptance-plan.md) · [详细集成规格](30-public-streaming-detailed-integration.md) · [Runtime/App 工程规格](33-public-runtime-app-engineering-spec.md)

日期：2026-09-17。分支：`feat/stream`。已提交控制面基线：`63b73d9`；设计基线：`f7bf60f`。

状态：**R1/R2 已完成工作树接缝，R3 及模型发布尚未完成**。本文把文档 25/27 中的原则收敛为类型、调用顺序、文件修改点、
所有权和错误合同。当前工作树已经落地 canonical encoder、扩展 catalog、resolver、authority、resolved request、engine identity、
exact engine resolve C ABI、active public generate 分支和 Swift resolution/error 类型；App 事务、四模型 public override、完整
source revalidation、RunResult public metrics、校准工具和 reviewed records 仍未实现。
本文不授予任何模型 public 执行资格。production catalog 继续为空；空 catalog 时 active selector 的 resolve/generate 仍在 GPU 锁和
session 执行前 fail-closed，active prepare 仍明确不支持。

本文冻结的核心实现选择是：

1. public selector 是可序列化意图，不是权限；
2. engine 在本地 metadata 上解析出不可变 snapshot；
3. authority 只存在于 native 进程内，不能从 JSON 构造；
4. generate 对 exact selector 重新核验并立即执行，不信任 App 保存的 digest；
5. 模型只负责 metadata probe、exact snapshot 和 kernel 接线，catalog 选择仍由通用 resolver 负责；
6. off/default 分支在 catalog、tokenizer、descriptor 和新缓存之前返回原路径；
7. 首版只开放 GPU，prepare 明确不支持 public streaming；
8. LTX 的独立进程必须在 worker 内完成最终 resolve，App 不能替 worker 授权。

## 1. 目标、非目标和完成边界

### 1.1 本阶段必须交付

- 完整 catalog record identity 和确定性 canonical digest；
- `tc_engine_resolve_streaming_json`；
- internal-only `StreamingAuthority`；
- immutable `ResolvedRequestExecution`；
- `ModelSession` 的 metadata probe、snapshot compile 和 `generate_resolved` 默认拒绝入口；
- active selector 的 generate 分支，且无 silent resident/manual fallback；
- Swift resolve API、结构化错误和 v2 semantic parity；
- App `StreamingChoice`、options state、任务冻结和 versioned job envelope；
- 四模型 public adapter 的可独立接入点；
- catalog 为空时的完整 App/engine fail-closed 行为；
- default/off 路径的零新增审计和性能基线。

### 1.2 本阶段明确不做

- 不从实时 free memory 自动求解 P/G/K/D/Q；
- 不把 8/10/12/16/20 GiB 解释成系统 hard cap；
- 不为 public selector 自动开启 `memory_constrained` guard；
- 不允许外部 JSON、profile、环境变量或 App 构造 authority；
- 不把 private candidate constructor 改名后直接公开；
- 不开放 GPU+ANE、LoRA、近似算法、量化变体或未校准 workload；
- 不在 block hot path 读取 catalog、JSON、设备信息或系统内存；
- 不引入第二套 slot executor、第二个 layout compiler 或模型私有 target→layout 选择器；
- 不让 `prepare` 先全驻留再切换成 public streaming；
- 不在本阶段承诺 streaming 一定快于 resident 或系统 swap。

### 1.3 “代码完成”与“模型可公开”分开

框架完成后 production catalog 仍可以为空。单个模型/档位只有同时满足下列条件才可加入 catalog：

```text
代码闭环
  + exact source/workload/device/runtime identity
  + full-request quality/lifecycle
  + full-request memory calibration
  + P0/P1/P4 证据
  + ordinary App/public constructor replay
  + independent review and revocation drill
= public record
```

private P1 anchor、denoiser probe、metadata estimate 或一次 MLX peak 都不能单独生成 public record。

## 2. 不可破坏的系统不变量

### 2.1 配置不变量

1. schema v1 manual layout 与 schema v2 selector 互斥；
2. selector disabled 与 absent 都走旧路径；
3. `memory_tier` 必须使用五个固定 target 之一；
4. `preset` 必须同时指定 preset ID/revision/catalog revision/target；
5. public selector 与 legacy residency、legacy memory budget、streaming offload、未认证 guard 组合早拒绝；
6. profile/request 对 selector 采用完整对象替换，不逐字段拼接两个 selector；
7. exact replay stale/revoked 时拒绝，不自动换到另一个 record；
8. adapter 不得在执行期修改 P/G/K/D/Q、retention、pool policy 或 component order。

### 2.2 权限不变量

- `expected_resolution_digest` 只是 stale 检查，不是 bearer token；
- 请求不能出现 `authorized`、`public=true`、`authority_id` 一类可伪造权限字段；
- authority 构造函数 private，只允许 resolver factory 创建；
- authority 必须绑定 catalog record、source、workload、device、runtime、layout 和 component policy；
- session 必须校验 authority 与自身 model family 和 snapshot 相同；
- test fixture authority 只能进入 test target，不导出 release symbol；
- catalog builder 不能将 `proposed`、incomplete 或无 review 的 record 编入 public index。

### 2.3 生命周期不变量

- metadata probe 不创建 GPU buffer、MLX array、slot pool 或 I/O worker；
- resolved snapshot 可以只读复用，但 slot/ticket/mailbox/cancel generation 不能跨请求复用；
- source lease 的寿命必须覆盖 pager、fill worker、GPU reader callback 和 drain；
- success、cancel 和 failure 都必须停止 dispatch、join worker、drain reader，再释放 backing；
- unsafe drain 必须 quarantine engine/request owner，不能假装清理成功；
- public execution 失败后不能自动 resident retry；
- result 中 actual layout 与 resolved layout 不一致时请求失败。

### 2.4 默认性能不变量

当 selector absent/disabled 时：

- 不访问 production catalog singleton；
- 不计算 source/workload/resolution digest；
- 不运行 tokenizer 的第二次 pass；
- 不创建 metadata cache entry；
- 不增加 worker/thread/pool/memory sampler；
- 不增加 cache clear、unload 或 GPU synchronize；
- 不添加每 block public 条件分支；
- 继续调用原 `ModelSession::generate(Request,...)` 或现有 worker 路线。

## 3. 目录和依赖结构

建议新增或扩展以下文件：

```text
native/core/
  streaming_contracts.hpp                 已有；只保留 wire intent/manual config
  streaming_public_types.hpp              新增；无 Foundation/MLX，保存 identity 值类型

native/runtime/streaming/
  preset_catalog.hpp/.cpp                 扩展；record/index/release/revocation
  preset_resolver.hpp/.cpp                新增；filter/rank/compile/authority
  resolved_request.hpp/.cpp               新增；immutable snapshot、digest、revalidation
  canonical_encoding.hpp/.cpp             新增；确定性 length-prefixed digest
  layout.*                                已有；唯一 layout compiler
  context.* / slot_pool.* / io_executor.* 已有；唯一 data plane
  mlx_weight_pager.*                      已有；MLX 模型复用

native/runtime/session.hpp
  ModelSession metadata probe / compile / generate_resolved 默认拒绝

native/api/c_api.mm
  engine identity、options/resolve/generate public 分支、结构化错误

native/platform/apple/
  *_streaming_descriptor.mm               模型 metadata/source projection
  *_session.mm                            原生模型 public snapshot/execute
  results.mm                              additive selection/actual/memory report

bindings/c/include/turbocider/turbocider.h
  tc_engine_resolve_streaming_json

bindings/swift/TurboCiderNative.swift
  完整 v2 wire、resolve response、JSON-or-legacy error

apps/macos/
  StreamingOptions.swift                  新增；query state/debounce/cache
  StudioState.swift                       StreamingChoice 和 validation
  JobStore.swift                          v1/v2 envelope、resolve→persist→generate
  LTXWorker.swift                         worker 内 final resolve/generate
  RunInsights*.swift                      target/preset/scope/actual 展示
```

依赖只允许：

```text
wire types
   ↓
catalog + canonical identity
   ↓
resolver ──→ ModelSession metadata hooks
   ↓                 ↓
resolved request → model snapshot
   ↓
existing executor/pager/kernel
```

`layout/context/slot_pool` 不得 include catalog 或 App 类型；`preset_catalog` 不得 include 模型 kernel；App 不得复制
模型 block 数和 prefix 合法区间。

## 4. 通用类型设计

下面是实施级草案；命名可在 patch 中按仓库风格微调，但字段和边界不能弱化。

### 4.1 基础 identity 值类型

建议 `native/core/streaming_public_types.hpp`：

```cpp
namespace tc::streaming {

struct PresetKey {
    std::string id;
    uint32_t revision = 0;
    std::string catalog_revision;
};

struct RuntimeIdentity {
    std::string turbocider_build_id;
    std::string runtime_revision;
    std::string adapter_revision;
    std::string reader_revision;
    std::string kernel_revision;
    std::string allocator_policy_revision;
};

struct DeviceIdentity {
    std::string gpu_name;
    uint64_t physical_memory_bytes = 0;
    std::string device_class;
    std::string os_build_family;
};

struct SourceFileIdentity {
    std::string logical_role;
    uint64_t size = 0;
    uint64_t device = 0;
    uint64_t inode = 0;
    uint64_t mtime_ns = 0;
    std::string content_digest; // only when installation verification provides it
};

struct SourceIdentity {
    std::string model_id;
    std::string variant;
    std::string format;
    std::string artifact_manifest_digest;
    std::string snapshot_digest;
    std::vector<SourceFileIdentity> files;
};

struct TokenShape {
    std::string encoder;
    uint32_t valid_rows = 0;
    uint32_t padded_rows = 0;
    uint32_t compute_rows = 0;
    std::string tokenizer_revision;
    std::string template_revision;
};

struct WorkloadIdentity {
    std::string model_id;
    std::string operation;
    std::string execution;
    std::string execution_container;
    uint32_t width = 0, height = 0, frames = 1, fps = 0, steps = 0;
    uint32_t batch = 1;
    bool audio = false;
    bool dynamic_text = false;
    bool approximation = false;
    std::vector<TokenShape> token_shapes;
    std::string conditioning_revision;
    std::string vae_policy_revision;
    std::string feature_digest; // LoRA/input/refiner flags; no raw prompt
};

} // namespace tc::streaming
```

实现要求：

- path 只用于本机诊断，不直接进入跨机器 catalog key；
- raw prompt、输出路径和 seed 不进入 layout eligibility；token shape/template revision 进入；
- LoRA 列表即使为空也要以 canonical empty identity 表达，不能省略成 wildcard；
- `artifact_manifest_digest` 缺失时 public exact resolve 返回 verification-required；
- inode/mtime 是 TOCTOU 快速检查，不代替经过验证的内容 digest；
- runtime/kernel/reader revision 必须来自编译代码，不能由请求提供。

### 4.2 Catalog record

把当前扁平 `StreamingPresetRecord` 扩展为组合结构：

```cpp
struct Applicability {
    SourceIdentityConstraint source;
    WorkloadConstraint workload;
    DeviceConstraint device;
    RuntimeConstraint runtime;
};

struct CanonicalPlanRecord {
    StreamingConfig config;
    std::string layout_schema_revision;
    std::string component_policy_revision;
    std::string pass_transition;
    std::string multi_pool_policy;
};

struct CalibrationEvidence {
    bool complete = false;
    std::string calibration_id;
    std::string scope;
    std::string estimator_revision;
    std::string execution_container;
    uint64_t calibrated_request_bytes = 0;
    uint64_t confirmation_sample_count = 0;
    uint64_t maximum_sample_gap_ns = 0;
    std::string evidence_digest;
};

struct PerformanceEvidence {
    std::string profile_id;
    uint32_t rank = 0;
    uint64_t logical_read_bytes = 0;
    std::string comparison_kind;
    std::string confidence_status;
    std::string evidence_digest;
};

struct ReleasePolicy {
    std::string channel; // proposed/public-experimental/public-stable/revoked
    bool revoked = false;
    std::string reviewed_commit;
    std::string review_digest;
};

struct StreamingPresetRecord {
    PresetKey key;
    Applicability applicability;
    CanonicalPlanRecord plan;
    CalibrationEvidence calibration;
    PerformanceEvidence performance;
    ReleasePolicy release;
    std::string canonical_record_digest;
};
```

旧 skeleton 字段可在一次内部 refactor 中迁移；这是未发布 C++ 内部类型，不需要维持其源码 ABI。测试 fixture builder
应改用命名 helper，避免 aggregate initializer 在新增字段时误填。

### 4.3 Model metadata probe

通用 resolver 不应知道 safetensors、MLX arrays、LTX C handle 或 H3 shard 细节。建议增加只读基类：

```cpp
class ModelStreamingProbe {
  public:
    virtual ~ModelStreamingProbe() = default;
    virtual std::string_view model_id() const noexcept = 0;
    virtual const SourceIdentity &source_identity() const noexcept = 0;
    virtual const WorkloadIdentity &workload_identity() const noexcept = 0;
    virtual const RuntimeIdentity &runtime_identity() const noexcept = 0;
    virtual std::string_view component_policy_revision() const noexcept = 0;
};
```

模型实现的 probe 可以持有：

- 已打开的只读文件描述符或稳定 file identity；
- 解析后的 checkpoint index；
- tokenizer 结果和 geometry；
- descriptor 构造所需的小型 metadata；
- verified install manifest 引用。

probe 不能持有：

- materialized model weights；
- GPU/Metal/MLX backing；
- slot pool、I/O workers、command buffer；
- mutable request execution state。

### 4.4 Model compiled snapshot

record 选定后，由同一个 session 把 probe 编译成模型 snapshot：

```cpp
class ModelStreamingSnapshot {
  public:
    virtual ~ModelStreamingSnapshot() = default;
    virtual std::string_view model_id() const noexcept = 0;
    virtual const SourceIdentity &source_identity() const noexcept = 0;
    virtual const streaming::Descriptor &descriptor() const noexcept = 0;
    virtual const streaming::Layout &layout() const noexcept = 0;
    virtual const RuntimeIdentity &runtime_identity() const noexcept = 0;
    virtual std::string_view component_policy_revision() const noexcept = 0;
};
```

各模型 subclass 负责持有现有 `StreamingPlanView`、source lease 和 model-specific immutable metadata。不要把这些对象
复制到 generic map，也不要让 generic snapshot 使用无类型裸 `void *`。session 可以用 checked `dynamic_cast` 取回自己的类型；
类型不匹配返回 `streaming_authority_mismatch`。

### 4.5 Authority 和 resolved execution

```cpp
class StreamingAuthority final {
  public:
    StreamingAuthority(const StreamingAuthority &) = delete;
    StreamingAuthority &operator=(const StreamingAuthority &) = delete;

    const PresetKey &preset() const noexcept;
    std::string_view resolution_digest() const noexcept;

  private:
    friend class PublicPresetResolver;
    StreamingAuthority(PresetKey, std::string source_digest,
                       std::string workload_digest,
                       std::string device_digest,
                       std::string runtime_digest,
                       std::string layout_digest,
                       std::string component_digest,
                       std::string resolution_digest);
    // all fields immutable
};

struct ResolvedRequestExecution final {
    ExecutionPlan model_plan;
    StreamingSelector requested_selector;
    StreamingSelector exact_selector;
    StreamingPresetRecord record;
    DeviceIdentity device;
    std::shared_ptr<const ModelStreamingProbe> probe;
    std::shared_ptr<const ModelStreamingSnapshot> model_snapshot;
    std::shared_ptr<const StreamingAuthority> authority;
    std::string request_digest;
    std::string resolution_digest;
};
```

`record` 可以保存为 shared catalog snapshot/reference，但必须保证 catalog 对象在 resolved execution 期间不可变。首版 catalog
嵌入 binary，复制小 record 简单且安全。内部 `model_plan.request` 必须先清除 `streaming_selector`，再把 record 的 canonical
manual config 填入 `streaming`，且该内部 request 永不重新走 public wire parser、永不序列化回 App。原始和 exact public
selector只保存在 `ResolvedRequestExecution` 中用于报告、stale核对和复放，不能让同一个 `Request` 同时携带 manual 与 selector。

## 5. `tc_engine` 的身份和缓存

当前 `tc_engine` 只保存 session、mutex、cancel/quarantine。建议增加：

```cpp
struct tc_engine {
    std::string model_id;
    std::filesystem::path model_root;
    std::unique_ptr<tc::ModelSession> session;
    std::mutex mutex;
    std::atomic<bool> cancelled{false};
    std::atomic<bool> memory_quarantined{false};
    std::atomic<bool> streaming_quarantined{false};
    std::optional<tc::MemoryExecutionReport> last_memory_report;
    tc::streaming::MetadataResolutionCache streaming_metadata_cache;
};
```

### 5.1 engine identity 的设置

`tc_engine_create_model` 在 session 创建成功后设置：

- exact registered `model_id`；
- absolute/normalized root，仅用于本地一致性和 diagnostics；
- session 自身仍是 source truth；root 字符串不能替代 fd/source snapshot；
- candidate constructor 同样设置 identity，但 authority source 仍不同。

若 model root 是 symlink，identity 记录 normalized requested path 和 resolved path；record 不按用户路径匹配。文件在 engine 生命周期中
变化时 probe cache 失效，generate revalidation 返回 `artifact_changed`。

### 5.2 metadata cache

cache 只缓存无 GPU 的 probe/snapshot 构造中间结果：

```text
key = model_id
    + verified artifact digest
    + source snapshot digest
    + workload digest
    + adapter/runtime revision
    + catalog revision
```

要求：

- fixed entry count 和估算 bytes 上限；
- LRU eviction 只释放 metadata/fd lease，不触发 GPU synchronize；
- active resolved execution 使用 shared ownership，evict 不影响它；
- cache hit 仍复核 source quick identity、record revoke 和 expected digest；
- off/default 永不查询该 cache；
- `unload()` 是否清 metadata cache由 source identity决定，但绝不能把 GPU arrays放入该 cache；
- cache metrics 只在 public resolve result 中报告，不污染 default result。

首版可以把 cache 上限设为每 engine 2–4 个 snapshot，具体数值在实现 PR 中按 metadata 实测冻结；未测前不写成
production 常量。

## 6. ModelSession 接口

建议在 `native/runtime/session.hpp` 增加 forward declarations，避免把全部 streaming 实现 include 到所有模型：

```cpp
namespace streaming {
struct PublicResolveInput;
struct StreamingPresetRecord;
class ModelStreamingProbe;
class ModelStreamingSnapshot;
struct ResolvedRequestExecution;
}

class ModelSession {
  public:
    virtual std::shared_ptr<const streaming::ModelStreamingProbe>
    probe_public_streaming(const streaming::PublicResolveInput &) const {
        throw std::runtime_error("streaming_public_adapter_unsupported");
    }

    virtual std::shared_ptr<const streaming::ModelStreamingSnapshot>
    compile_public_streaming(
        std::shared_ptr<const streaming::ModelStreamingProbe>,
        const streaming::StreamingPresetRecord &) const {
        throw std::runtime_error("streaming_public_adapter_unsupported");
    }

    virtual RunResult generate_resolved(
        std::shared_ptr<const streaming::ResolvedRequestExecution>,
        const Event &, std::atomic<bool> &) {
        throw std::runtime_error("streaming_public_adapter_unsupported");
    }
};
```

### 6.1 为什么分 probe 和 compile

- probe 提供 artifact/token/workload/runtime identity，供 catalog 初筛；
- compile 只对最终选中的一个 record 构造 descriptor/layout；
- 避免为每个 rejected candidate 解析/编译一次完整 layout；
- 避免 resolver include 四模型头文件；
- 允许 options 保持 tentative，而 engine resolve 获得 exact 结果；
- model snapshot 可持有 existing PlanView，保证执行和 resolve 使用同一 metadata。

### 6.2 默认实现必须抛错

默认 `generate_resolved` 不能调用 `generate(execution->model_plan.request,...)`。否则新模型即使没有审阅 public adapter 也会
静默获得权限。三个默认方法的错误码固定为 `streaming_public_adapter_unsupported`，并在 host fake-session 测试中验证零
session generate 调用。

### 6.3 private candidate 与 public authority 分离

当前 engine 上的 `allow_experimental_streaming` 只服务 private/manual candidate。public 路线：

- 不设置该 bool；
- 不读取环境变量来授予权限；
- session 只接受 resolved authority；
- result authority 写 `public_preset_registry`；
- private route 仍写 `private_candidate_constructor`；
- 同一请求不能同时携带 manual config 和 public selector。

## 7. Resolver 的精确算法

### 7.1 输入

```cpp
struct PublicResolveInput {
    Request request;                       // profile 已合并并验证
    StreamingSelector selector;           // active schema v2
    std::string execution_container;       // host policy注入，不信任JSON
    DeviceIdentity device;                 // native读取
    std::string engine_model_id;
    std::filesystem::path engine_root;     // local only
    ResolvePurpose purpose;                // options_preview | exact_preview | execute
};
```

`execution_container` 由入口固定：embedded App、CLI worker、service worker分别使用不同值。请求不能声称自己是
`cli_worker` 以套用更低 baseline 的 calibration。

### 7.2 解析阶段

```text
R0 parse JSON + duplicate-key + lexical integer validation
R1 resolve profile + validate conflicts
R2 verify engine model ID matches request model
R3 call session.probe_public_streaming(request)
R4 validate source/workload/runtime identities are complete
R5 catalog cheap index filter
R6 target/calibration/margin/release/device filter
R7 deterministic rank or exact replay lookup
R8 session.compile_public_streaming(probe, selected record)
R9 compare compiled descriptor/layout/component/runtime against record
R10 canonical encode all identities and compute resolution digest
R11 compare expected_resolution_digest if present
R12 mint StreamingAuthority and immutable execution
```

任何阶段失败都不能创建 slot pool、配置 GPU 或调用 ordinary `generate`。

### 7.3 过滤顺序和错误优先级

推荐稳定顺序：

1. invalid selector/config conflict；
2. model/route/format unsupported；
3. artifact verification/source mismatch；
4. workload/token/features not validated；
5. device/runtime/container mismatch；
6. release/revocation；
7. calibration incomplete；
8. target margin 不满足；
9. exact preset stale/missing；
10. compiled actual plan mismatch。

错误优先级必须由测试冻结，不能因为 catalog record 排序变化而改变用户看到的 reason。resolver 可以内部收集全部 rejected
reason，但 public response 只给一个 primary code 和 bounded diagnostics。

### 7.4 deterministic rank

memory-tier selector 在所有 exact-compatible candidates 中按：

```text
performance.rank ascending
calibrated_request_bytes ascending
logical_read_bytes ascending
preset_id bytewise ascending
preset_revision ascending
```

排序字段全部来自 reviewed catalog。当前 skeleton 已实现此原则；扩展 schema 时保持顺序并增加 golden tests。不得把本次临时
free memory、SSD cache hit、当前温度或随机采样放入 rank，否则 replay 不可重复。

### 7.5 余量规则

保持现有整数算法：

```text
H(T) = max(512 MiB, ceil(T / 10))
record fits iff calibrated_request_bytes + H(T) <= T
```

实现继续使用防溢出的减法形式。`calibrated_request_bytes=0` 不是 unknown 的合法替代；incomplete record 在 fits 前拒绝。

## 8. Canonical encoding 和 digest

不要直接对 Foundation/Swift/Python 重新序列化后的 JSON 求 hash。不同 number/string/key order 容易产生不必要漂移。

### 8.1 编码格式

建议 `tc-streaming-canonical-v1` 使用显式 type tag + big-endian length/value：

```text
magic utf8("tc-streaming-canonical-v1")
record begin(tag, field_count)
field(tag, utf8 length + bytes)
field(tag, u32 big endian)
field(tag, u64 big endian)
field(tag, bool 0/1)
list(tag, count, ordered elements)
record end
```

- map 必须按 UTF-8 byte order 排序；
- list 的顺序有语义，不排序；
- optional 使用 absent/present tag，不能把 absent 与0/empty混同；
- 不编码 locale、pointer、filesystem path separator 或浮点格式；
- workload 若必须编码浮点，先改成受审整数/枚举；
- SHA-256 输出 lower-case hex；
- C++ builder 和 Python catalog builder 共享 golden vectors。

### 8.2 Digest 层级

| 字段 | 输入 | 用途 |
|---|---|---|
| `record_digest` | catalog record除自身digest | 检测生成/嵌入漂移 |
| `source_snapshot_digest` | verified manifest + file identities | TOCTOU和artifact绑定 |
| `workload_digest` | geometry/token/backend/features | workload exact匹配 |
| `layout_digest` | compiler canonical output | P/G/K/D/Q/pools/pass identity |
| `component_digest` | text/denoise/VAE/export live order | 全请求生命周期绑定 |
| `resolution_digest` | 上述digest + preset/device/runtime/container/build | stale检查和结果追踪 |
| `request_digest` | 完整语义请求，排除输出临时路径策略按协议 | job冻结/复放诊断 |

`resolution_digest` 不包含随机进程 nonce，否则同一合法 replay 无法比较；authority 对象本身不可序列化已经提供权限隔离。

### 8.3 Golden tests

至少覆盖：

- JSON key order 改变，digest 不变；
- explicit false 与 absent，digest 不同；
- source inode/mtime/content digest 任一变化，source/resolution digest 变化；
- P/K/retention/pool policy 任一变化，layout/resolution digest 变化；
- seed/output path 改变不影响 layout eligibility，但 request digest 按冻结规则变化；
- C++ 与 Python 对同 fixture 产生相同 hash；
- malformed UTF-8、embedded NUL、超长字段被拒绝。

## 9. C ABI 设计

### 9.1 新接口

```c
int tc_engine_resolve_streaming_json(
    tc_engine *engine,
    const char *request_json,
    char **result_json,
    char **error_json);
```

不新增“set authority”接口，不返回指针 handle 给 App。resolve 是预览/冻结 exact selector；generate 时 native 重新 resolve 并
在同一次调用中持有真正 authority。

### 9.2 Resolve 返回结构

```json
{
  "schema_version": 1,
  "status": "resolved",
  "request_digest": "...",
  "resolution_digest": "...",
  "catalog_revision": "...",
  "exact_selector": {
    "schema_version": 2,
    "enabled": true,
    "selection": "preset",
    "retention": "request",
    "target_request_memory_bytes": 12884901888,
    "preset_id": "illustrative.only",
    "preset_revision": 1,
    "catalog_revision": "illustrative.only",
    "expected_resolution_digest": "..."
  },
  "summary": {
    "model": "z-image-turbo",
    "execution_container": "embedded_app",
    "memory_scope": "execution_process_tree_v1",
    "calibrated_request_bytes": 0,
    "release_channel": "public-experimental",
    "layout_digest": "...",
    "component_policy_revision": "..."
  }
}
```

示例数值/ID 非真实 record。实际 resolved response 不返回 raw model path、prompt、文件 inode 列表或 authority 内部对象。

### 9.3 结构化错误

新 endpoint 和 active-selector generate 错误采用：

```json
{
  "schema_version": 1,
  "code": "no_preset_fits_target",
  "message": "当前模型与任务没有满足该内存目标的已验证方案。",
  "retryable": false,
  "action": "choose_another_target"
}
```

现有 v1/off 请求继续允许旧纯字符串错误。Swift decoder 先尝试 JSON，再回退 legacy message。不要把本次全库 error ABI 一次性
重写；只对新 opt-in 路线提供 stable code。

### 9.4 Lock 和调用约束

`tc_engine_resolve_streaming_json`：

1. `*result/*error=nullptr`；
2. try-lock engine mutex，busy 立即失败；
3. 不重置 engine cancel flag；
4. 不拿 global GPU mutex/DeviceLease；
5. 不调用 `configure_streams`；
6. 不执行 `session->load/unload/prepare/generate`；
7. 完成后释放 local lock；
8. 所有返回字符串用 `tc_string_free`。

options 无 engine，继续是 catalog/device tentative query。options 不接受请求提供的 container override；C API 按入口固定
`embedded_app`，CLI/service 若需要选项应增加 host-owned variant，而不是在 JSON 中信任字符串。

## 10. Generate 调用链改造

### 10.1 目标结构

`tc_engine_generate` 中 active selector 不能简单删除现有 fail-closed gate。建议重构成两个私有 helper：

```cpp
int generate_legacy_or_private(...); // 原路径，尽量机械移动
int generate_public_streaming(...);  // 新 opt-in 路径
```

入口：

```cpp
auto request = request_from_json(parse_json(r));
if (!(request.streaming_selector && request.streaming_selector->active()))
    return generate_legacy_or_private(...);
return generate_public_streaming(...);
```

真正实现时要避免在分支前做新的 catalog/device/token工作；parse 本身是原本已有成本。

### 10.2 public 路径顺序

```text
engine try-lock
  → reject memory/streaming quarantine
  → parse/profile/conflict validation
  → exact resolver（无GPU）
  → validate public adapter available
  → acquire global GPU execution try-lock
  → DeviceLease
  → revalidate source quick identity + catalog revocation + device/runtime
  → reset request-local cancel flag
  → configure only the resolved backend
  → create request execution owner
  → session.generate_resolved(snapshot)
  → validate actual == resolved
  → drain/release/report
```

若 exact resolve 较慢，engine local lock期间 App 不能对同一 engine prepare/load/generate，这是预期；cancel 不应被 resolve 重置。
后续若需要可取消 resolve，应增加独立 resolution cancellation token，不能复用生成 cancel flag造成并发歧义。

### 10.3 执行前 revalidation

获得 global GPU lock 后至少复核：

- catalog revision/record 未撤回；
- source lease/file quick identity 未变化；
- engine model/session 未被替换；
- device identity 与 resolve 时相同；
- runtime/build identity 相同；
- expected resolution digest 相同；
- temporary admission 基线满足 record 的设备/容器策略。

临时系统压力只能返回 `temporary_memory_pressure`，不能更换布局或把 target 降低。用户可重试；若选择另一个 target，应产生新任务。

### 10.4 actual plan 核对

模型返回 `StreamingRuntimeMetrics` 后，C API/通用 helper 比较：

- stage/pool count/class；
- P/G/K/D/Q；
- group/pass/retention/pass transition；
- reader/kernel/weight format；
- layout digest；
- component policy/conditioning/upsample boundary；
- authority source。

任一不一致：终态失败 `streaming_actual_plan_mismatch`，保留 diagnostics 并按生命周期规则 drain。不能成功返回后只记录 warning。

### 10.5 Prepare 和 load

- active public selector + `tc_engine_prepare` → `streaming_prepare_unsupported`；
- `tc_engine_load` 不接受 request，不能为未来 public request预先全驻留加载；
- App public generate 前不调用 load/prepare；
- 从 resident session 切 public 时显式 unload/clear 的需要由模型 component policy 决定，并计入 full request；
- public 完成后是否保留 metadata 不等于保留 GPU weights；首版 retention=request。

## 11. Request execution owner

建议每次 generate 构造：

```cpp
class PublicStreamingExecutionOwner {
    std::shared_ptr<const ResolvedRequestExecution> resolved_;
    std::unique_ptr<ModelRequestAdapter> adapter_;
    std::unique_ptr<streaming::StageExecutor> executor_;
    std::vector<std::shared_ptr<streaming::SlotPool>> pools_;
    std::unique_ptr<streaming::IoExecutor> io_;
    CompletionMailbox mailbox_;
    uint64_t request_generation_;
    TerminalState state_;
};
```

不要求强行新增这一公共类；各模型已有 owner 可通过共同 helper 满足相同生命周期。但必须能在测试中证明下面顺序：

```text
construct immutable snapshot
construct backing/pools
start bounded workers
run passes
stop new dispatch
join fills
wait GPU readers
drain callbacks/mailbox
destroy pools/prefix/resident backing
release source lease/snapshot/authority
publish terminal result
```

析构函数不能依赖抛异常完成安全清理。显式 `finish()`/`cancel_and_drain()` 返回状态；析构只做最后防护。若底层 executor 的
unsafe destruction 当前会 terminate，public path 必须保证在 owner 离开前已调用安全终止 API，失败则 quarantine 并保留 owner。

## 12. Swift binding 设计

### 12.1 v2 类型必须完成 semantic parity

当前 `NativeRequestV2` 是骨架。实施前建立 `NativeRequestSemanticSnapshot`，从 v1/v2 解析成共同测试表示，逐项核对：

- model/variant/operation；
- text/media inputs、audio path、include embedded audio、strength；
- output kind/path/shape/frames/fps/audio；
- seed/steps；
- execution/profile/ANE manifests/approximation；
- residency/budget/guard conflict语义；
- dynamic text/compile/noise/VSA；
- LTX 所有 flags；
- LoRA path/strength/role/strategy；
- explicit false/0 与 absent。

public record不支持的字段在 validation 返回明确 code。不能因为 Swift v2 漏字段而不发送。

### 12.2 新 response 类型

```swift
public struct NativeStreamingResolution: Codable, Sendable {
    public let schema_version: Int
    public let status: String
    public let request_digest: String
    public let resolution_digest: String
    public let catalog_revision: String
    public let exact_selector: NativeStreamingSelectorV2
    public let summary: NativeStreamingResolutionSummary
}

public struct NativeAPIErrorEnvelope: Codable, Sendable, Error {
    public let schema_version: Int
    public let code: String
    public let message: String
    public let retryable: Bool
    public let action: String?
}
```

`NativeEngine.resolveStreaming(_:)` 与 generate 使用同一个 serial queue，防止同一 engine source/session 并发变化。options 是 static
metadata query，不使用 engine queue。

### 12.3 错误解码

```text
if error bytes decode NativeAPIErrorEnvelope:
    throw NativeFailure(code/message/retryable/action)
else:
    throw NativeFailure(message: legacy string)
```

不要在 Swift 根据中文 message contains 判断。现有 `NativeFailure` 可 additive 增加 optional code/retryable/action，并保留原
initializer，避免旧调用点大面积改写。

## 13. App 产品状态和任务事务

### 13.1 Draft 只保存用户目标

```swift
enum StreamingMode: String, Codable, Sendable {
    case off
    case memoryTier
}

struct StreamingChoice: Codable, Sendable, Equatable {
    var mode: StreamingMode = .off
    var targetBytes: UInt64?
}
```

建议 `StudioDraft` 按 model ID 保存最近 target 偏好；source/shape变化后重新 query，不复用旧 availability。不要保存 P/G/K/D/Q。
旧 draft decode 为 off；损坏 target 提示用户重选，不向最近档位取整。

### 13.2 Options state machine

`apps/macos/StreamingOptions.swift`：

```text
off
loading(revision)
available(revision, options, recommendedTarget)
unavailable(revision, reason)
failed(revision, code, retryable)
```

- off 时 task/caches均不触发 native query；
- draft 影响 key 的字段变化后约300ms debounce；
- revision 不同的旧响应丢弃；
- UI task cancel 不调用 engine.cancel；
- options 只是 tentative；提交时必须 exact resolve；
- production catalog为空时展示“当前版本没有已发布方案”，而非“模型损坏”。

### 13.3 推荐算法

App 只在 native 返回 available targets 内推荐：

```text
reserve(P) = max(4 GiB, ceil(20% × physical memory P))
cap = P - reserve(P)
recommended = largest available target <= cap
```

这是 UI 初值，不是执行准入。若 exact resolve 因 token/source/container 变为 unavailable，显示 exact reason，不自动换档。

### 13.4 Job envelope

避免把 `NativeJob.request` 直接改成无法向后兼容的 enum synthesized JSON。建议持久化：

```json
{
  "job_schema_version": 2,
  "id": "...",
  "createdAt": "...",
  "request_payload": {
    "wire_schema": 2,
    "request": {}
  },
  "streaming_submission": {
    "requested_target_bytes": 12884901888,
    "exact_selector": {},
    "request_digest": "...",
    "resolution_digest": "...",
    "catalog_revision": "...",
    "evidence_summary": {}
  },
  "state": "preparing"
}
```

历史无 `job_schema_version` 的对象继续 decode `request: NativeRequest`。新 off 任务仍可写旧格式，减少迁移面。unknown future schema
必须保留原 jobs 文件并报只读错误，不能启动时重写为空。

统一 accessor：

- `job.modelID`；
- `job.outputURL`；
- `job.inputPaths`；
- `job.requestPayload`；
- `job.canReplayExactly`；
- `job.streamingSummary`。

输出删除、预览、route summary、参数复用、LTX worker routing 不再直接假定 `job.request` 为 v1。

### 13.5 提交事务

```text
T0 MainActor 设置 busy，并复制 immutable draft revision
T1 构造 v1(off) 或 v2(on) payload
T2 off：原 plan/acquire/generate
T3 on：确认 options revision仍匹配
T4 acquire正确execution container的engine/worker
T5 exact resolve
T6 比较draft revision、target、model/source/catalog
T7 构造frozen job并atomic persist
T8 persist成功后generate exact selector
T9 terminal result/error atomic persist
```

T5成功但T7失败：不启动GPU。T7后T8发现stale：job写failed/stale，不自动换preset。用户选择“重新推荐并生成”时创建新job。

## 14. LTX 独立 worker 的特殊设计

当前 `LTXWorker.accepts` 只接 v1 `component_staged`，并把 request JSON写入临时目录后启动 CLI。public LTX 不能在 App
主进程 resolve 后把 authority 传给 worker，因为 container、process baseline、source lease和runtime identity属于 worker。

### 14.1 首版协议

建议 worker 支持单次 `generate` 命令内的 resolve+execute：

```text
App options: tentative embedded_app/worker-capability display
App freeze: public v2 selector + target
Worker starts fresh
Worker creates ordinary model engine
Worker exact resolve with execution_container=ltx_cli_worker
Worker writes one bounded "streaming_resolved" event
Worker executes same exact selector immediately
Worker result carries actual resolution
App persists returned exact selector/digest/evidence summary
```

为了满足“GPU前先持久化 frozen job”，App 可先持久化 pending selector job；worker resolved event到达后，在 worker开始 GPU
allocation 前通过一条 ACK 协议让 App 更新/持久化 exact submission。若不实现双向 ACK，则首版 LTX public 必须改为两阶段 worker：

1. `turbocider resolve-streaming model request` 返回 exact payload后退出；
2. App persist；
3. 新 fresh worker `generate` 重新核验 exact selector并执行。

推荐两阶段方案，简单、可审计，代价是 metadata解析两次；可用verified metadata cache文件优化，但不能持久化 authority。

### 14.2 Worker 安全要求

- 临时 request/result/events 文件权限受限；
- request/result 大小有上限；
- stderr event JSONL 与错误文本分离，避免把任意日志当事件；
- exact resolve和generate都返回 stable code；
- App cancellation 对 worker发TERM，grace后KILL；worker未drain时记录unsafe termination，不能生成成功result；
- calibration record必须匹配 `ltx_cli_worker`，不能沿用embedded App；
- worker executable/build digest进入runtime identity。

## 15. 四模型 snapshot 接入

### 15.1 共同模式

每个模型实现三个方法：

```text
probe_public_streaming(request)
  → source/workload/runtime/component identity + model metadata owner

compile_public_streaming(probe, record)
  → existing StreamingPlanView + generic descriptor/layout + source lease

generate_resolved(execution)
  → checked model snapshot → existing exact owner/adapter → kernel
```

compile 必须验证 record canonical config，不从 target 推导。generate 不重新构建另一个不同 PlanView；如果模型 API要求构造执行对象，
它必须引用 snapshot 内同一 source/layout identity并逐项核对。

### 15.2 LTX 2.5 distilled

文件：

- `native/platform/apple/ltx_session.mm`；
- `native/models/ltx_runtime/ltx_streaming_descriptor.*`；
- `native/models/ltx_runtime/ltx_streaming_plan.*`；
- `native/models/ltx_runtime/ltx_streaming_adapter.inc`；
- `apps/macos/LTXWorker.swift`。

`LtxPublicProbe` 持有 ConvRot/source manifest、48-block metadata、Gemma token shape、geometry、backend flags。
`LtxPublicSnapshot` 持有 existing `ltx::StreamingPlanView`、fd-only source owner、canonical layout和component policy。

执行要求：

- 首版只 C/Metal dense T2V；
- 512×320×33、11 steps、24fps、无audio/I2V/Sol/LoRA/ANE；
- exact P/G/K/D/Q 不由现有 budget heuristic重算；
- connector、stage1/stage2、upsampler、VAE/export都在component identity；
- request retention不继承legacy retained cache；
- worker container独立校准；
- existing private P8/K3锚点只作回归，不自动成为target record。

### 15.3 Z-Image Turbo

文件：

- `native/models/z_image/z_image.hpp/.cpp`；
- `native/models/z_image/weight_stream.hpp`；
- `native/platform/apple/z_image_streaming_descriptor.mm`；
- `native/platform/apple/z_image_weight_stream.mm`。

`ZImagePublicProbe` 绑定 original BF16 source/index、Qwen token valid/padded/compute rows、shape、9-step schedule。
`ZImagePublicSnapshot` 持有 existing `z_image::StreamingPlanView` 和 exact weight stream source。

执行要求：

- GGUF、LoRA、ANE、legacy budget不进入public v1；
- fixed K2/G1/D0/Q1 candidate族；
- P29对30 blocks/K2拒绝；
- VAE前释放策略进入component revision；
- encoder manifest变化触发的weight release行为在合并dev后重新核对；
- 512²与1024²是不同workload，不共享record。

### 15.4 MiniMax H3 Turbo

文件：

- `native/platform/apple/h3_session.mm`；
- `native/models/h3_runtime/h3_streaming_descriptor.*`；
- `native/models/h3_runtime/h3_streaming_policy.*`；
- H3 source/pass hooks。

`H3TurboPublicProbe` 必须拒绝普通H3、MLX H3、FastH3、quant cache和非original BF16 shards。
`H3TurboPublicSnapshot` 持有 transformer source、50-block descriptor、四pass/carry policy。

执行要求：

- 固定 K2/G1/D1/Q1；
- `carry_first_group` 和大矩阵单次大pread不改变；
- 512²×22 frames×4 steps首卡；39/73另卡；
- full Qwen3 text→DiT→video VAE→MP4/export；
- public authority路径不能永久打开 `allow_experimental_streaming_`；
- existing bootstrap INCONCLUSIVE如实保留。

### 15.5 Flux.2 Klein 9B

文件：

- `native/models/flux2/flux.hpp`；
- `native/models/flux2/flux_streaming.hpp`；
- `native/models/flux2/flux_transformer.cpp`；
- `native/models/flux2/pipeline.cpp`；
- `native/platform/apple/flux_streaming_descriptor.mm`；
- `native/runtime/streaming/mlx_weight_pager.*`。

先接 P0/G1/K2/D1/Q2 snapshot，保持两个 retained class pools。P>0 第二阶段增加prefix arrays和统一block boundary helper。

执行要求：

- 只9B BF16 eager，排除4B compiled/LoRA/ANE；
- global block ID dual 0..7、single 8..31；
- concatenate在block8每pass恰一次；
- P8消除空dual suffix pool；
- P7因dual suffix不足K2拒绝；
- compiled pool ID不能等同class enum/vector index；
- P0代码路径和direct reference保持可对照；
- prefix权重每request加载一次，prefix block每pass仍计算一次。

## 16. Result 和可观测性

`RunResult` 建议新增 optional：

```cpp
struct PublicStreamingSelectionMetrics {
    PresetKey preset;
    uint64_t requested_target_bytes = 0;
    std::string resolution_digest;
    std::string selection_source; // public_preset_registry
    std::string release_channel;
    std::string execution_container;
    std::string memory_scope;
    uint64_t calibrated_request_bytes = 0;
};
```

保持现有 `StreamingRuntimeMetrics` 表达 actual layout。`results.mm` 输出：

```text
plan.streaming.requested_selector
plan.streaming.exact_selector
plan.streaming.selection
plan.streaming.resolved_layout
plan.streaming.actual_layout
plan.streaming.memory_calibration
plan.streaming.authority=public_preset_registry
```

规则：

- off/default不构造也不输出这些对象；
- catalog calibrated peak 与本次 observed peak分开；
- 未启用日常sampler时 observed=null，不写0；
- private candidate仍保留原authority标签；
- public success必须包含exact/actual digest；
- error diagnostics不得泄露raw prompt或本机绝对model路径。

## 17. 稳定错误和 UI 动作

| Code | Retryable | UI动作 |
|---|---:|---|
| `streaming_selector_invalid` | 否 | 修复配置/重选 |
| `streaming_config_conflict` | 否 | 关闭冲突profile或旧budget |
| `catalog_has_no_public_records` | 否 | 显示当前版本未发布 |
| `unsupported_model_or_format` | 否 | 使用支持的原始模型格式 |
| `unvalidated_workload` | 可能 | 改为已验证shape/steps或关闭 |
| `artifact_verification_required` | 是 | 启动独立安装验证 |
| `artifact_changed` | 是 | 重建engine/重新验证 |
| `preset_not_public` | 否 | 选择公开档位 |
| `preset_revoked` | 否 | 重新推荐；不exact replay |
| `no_preset_fits_target` | 否 | 选更高target或关闭 |
| `streaming_resolution_stale` | 是 | 重新resolve并确认 |
| `streaming_route_unsupported` | 否 | 切GPU或关闭不兼容特性 |
| `streaming_public_adapter_unsupported` | 否 | 当前模型未接执行入口 |
| `streaming_prepare_unsupported` | 否 | 直接generate |
| `temporary_memory_pressure` | 是 | 关闭其他任务后重试 |
| `engine_busy` | 是 | 等待当前操作完成 |
| `streaming_actual_plan_mismatch` | 否 | 内部错误；保留证据 |
| `streaming_authority_mismatch` | 否 | 内部错误；重建engine |
| `streaming_quarantined` | 否 | 重建engine/worker |

同一错误在 options/resolve/generate 中的 code 保持一致；message可本地化，code不可随文案变化。

## 18. 逐文件施工顺序

### 18.1 R1：类型、canonical encoding、catalog schema

修改/新增：

- `native/core/streaming_public_types.hpp`；
- `native/runtime/streaming/canonical_encoding.*`；
- `native/runtime/streaming/preset_catalog.*`；
- resolver fixture tests。

完成门：纯C++ host build；cross-language digest golden；production catalog仍为空；无C ABI执行变化。

### 18.2 R2：session metadata hooks 和 resolver

修改/新增：

- `native/runtime/session.hpp`；
- `native/runtime/streaming/preset_resolver.*`；
- `native/runtime/streaming/resolved_request.*`；
- fake session/probe/snapshot tests。

完成门：fake模型完整resolve；默认方法拒绝；TOCTOU/stale/revoke/identity mismatch覆盖；无GPU。

### 18.3 R3：engine identity 与 C ABI

修改：

- `native/api/c_api.mm`；
- `bindings/c/include/turbocider/turbocider.h`；
- `native/platform/apple/results.mm`；
- C ABI contract tests。

完成门：options/resolve可用；active generate只有fake/public test adapter可执行；普通模型仍unsupported；off audit零。

### 18.4 R4：Swift/App

修改：

- `bindings/swift/TurboCiderNative.swift`；
- `apps/macos/StreamingOptions.swift`；
- `StudioState.swift`、`JobStore.swift`、`RunInsights*.swift`；
- integration tests。

完成门：catalog为空时UI只显示unavailable；off序列化和行为不变；v1 jobs无损；persist failure不generate。

### 18.5 R5–R8：模型 adapter

建议顺序：

1. Z-Image 或 LTX：验证一原生/一MLX接入形态；
2. 另一者；
3. H3 Turbo；
4. Flux P0；
5. Flux prefix。

每个模型可先让 test catalog record执行，不立即加入production catalog。普通 App release binary不能加载test catalog。

### 18.6 R9：production records

只提交reviewed evidence对应的generated catalog diff。record PR不同时改kernel/executor；这样出现回归可仅撤回record。

## 19. 代码级测试清单

### 19.1 Host unit

- canonical encoder所有类型/optional/list/map；
- C++/Python digest一致；
- record duplicate ID/revision、排序、revoked、release channel；
- five targets和margin边界±1 byte；
- options tentative与exact resolve reason差异；
- probe incomplete identity拒绝；
- compile actual layout mismatch拒绝；
- authority不可copy/不可deserialize；
- expected digest stale；
- source mutation、runtime/device/container mismatch；
- metadata cache hit/evict/active shared owner；
- default `ModelSession` 三个入口拒绝。

### 19.2 C ABI contract

- null engine/input/output/error组合；
- success只result、failure只error；
- error JSON schema和legacy fallback；
- engine busy；
- resolve不改变cancel flag；
- resolve零global GPU lock/DeviceLease/configure/session generate；
- active public prepare固定拒绝；
- off generate字节/语义路径不变；
- candidate bool不能授予public authority；
- release binary无test catalog setter/authority mint symbol。

### 19.3 Swift/App

- v1 golden完全不变；
- v2每字段round-trip和semantic snapshot；
- explicit false/0/absent；
- error JSON/legacy string；
- options debounce、cancel和out-of-order revision；
- old/new/unknown job schema；
- persist failure零generate；
- stale/revoked不自动重选；
- output delete/preview/replay/LTX routing；
- streaming off零query；
- catalog empty和unsupported workload文案区分。

### 19.4 Lifecycle/fault

- failure at probe/compile/pool create/first fill/mid fill/encode/fence/VAE/export；
- cancel before resolve、after resolve、during fill、during GPU reader、during VAE；
- source changed between resolve和global lock；
- callback throws/returns slowly；
- mailbox overflow；
- join/drain timeout；
- quarantine后所有入口拒绝；
- resident→public→resident和preset A→B→A；
- terminal时active reader/inflight fill/worker为0。

## 20. 实施审阅检查表

每个代码 PR 必须逐项回答：

1. off 分支是否在新控制面调用之前？
2. 是否新增了默认路径静态初始化、锁、线程、hash或I/O？
3. selector是否仍只是意图？
4. authority是否只能由resolver构造？
5. resolve和execute是否使用同一source/workload/layout identity？
6. session是否可能重新选择另一布局？
7. exact replay stale/revoked是否拒绝？
8. source在resolve后变化是否被执行前复核？
9. metadata probe是否意外materialize权重？
10. snapshot是否比pager/worker/callback活得久？
11. failure是否安全drain或quarantine？
12. result actual是否逐项对照resolved？
13. LTX是否在真正worker container内授权？
14. App persist失败是否阻止GPU运行？
15. test fixture/catalog/authority是否与release隔离？
16. 新模型是否只能通过显式override获得资格？
17. GPU+ANE/LoRA/approximation是否准确拒绝？
18. 文档/测试是否没有把private anchor写成public record？

## 21. 本文的完成定义

本文对应代码全部完成时，应能够做到：

- production catalog保持为空也能完整编译、运行App和执行所有旧请求；
- App高级设置可以展示“当前无可用public方案”，但不会误导用户；
- test-only reviewed fixture可通过 ordinary engine 完成 selector→resolve→authority→generate_resolved；
- 任意伪造 selector/digest/record都不能绕过native authority；
- 四模型能按独立PR接入，不修改通用executor协议；
- default/off audit为零且P0不回退；
- 之后只需通过文档29的实机证据和record review，便可逐模型、逐workload、逐target发布。

真正的 public 完成仍以 production record 和实机验收为准；“resolver编译成功”不是发布终点。
