# 30 · Public Streaming 详细集成、代码实现与验收规格

[目录](README.md) · [Runtime 代码设计](28-public-runtime-code-design.md) · [实施与实验计划](29-public-implementation-and-acceptance-plan.md) · [Runtime/App 工程规格](33-public-runtime-app-engineering-spec.md) · [配置与校准手册](31-public-streaming-config-calibration-runbook.md) · [当前框架说明](22-current-framework-guide.md)

日期：2026-09-17。分支：feat/stream。

状态：**实现伴随规格；public streaming 尚未发布**。本文建立在现有 schema-v2 selector、空 production catalog、
通用 layout compiler、StageExecutor、slot pool、MLX pager 和四模型 private adapter 之上，把下一步工作细化到
类型、函数、锁、所有权、状态机、逐文件修改和可自动验收的 test ID。本文不授予任何模型 public 资格，也不把
private 性能锚点改写成 8/10/12/16/20 GiB 档位证据。

与 28/29 的分工：

- 28 定义 public runtime 的安全边界和主要接口；
- 29 定义 PR、实验、发布和撤回流程；
- 本文定义实现者在每个调用点应写什么、如何复用现有 executor、如何证明默认性能不受影响。

## 1. 最终产品形态和关键判断

用户只选择：

1. Streaming 关闭；或
2. Streaming 开启，并选择 8、10、12、16、20 GiB 中 native 明确返回可用的请求目标。

用户不直接编辑 P/G/K/D/Q：

- P：resident prefix blocks；
- G：block group size；
- K：slot count；
- D：prefetch distance；
- Q：I/O worker count。

P/G/K/D/Q 仍然是框架和性能工程的核心，但它们属于 reviewed preset record，而不是产品 UI。这样可以同时满足：

- UI 简单，不要求用户理解模型 block 结构；
- runtime 得到确定的、可复放的布局；
- 每个布局都能绑定完整的 source/workload/device/runtime 证据；
- 新模型只需实现 descriptor/snapshot/adapter，不复制 target 到布局的启发式；
- production catalog 为空时完整 fail-closed；
- streaming 关闭时继续走当前默认路径。

Public v1 冻结为 GPU-only。MiniMax H3 范围只包含 MiniMax H3 Turbo original BF16，不包含普通 H3、MLX H3、
FastH3、量化变体或其他 checkpoint。

## 2. 当前代码检查点和真实缺口

当前工作树已经完成 R1/R2 基础并接入 exact engine C ABI，但仍处于未发布阶段：

| 层 | 当前已有 | 尚缺 |
|---|---|---|
| Wire | schema-v2 selector、五个 target、profile/request 合并、冲突检查 | App choice、job envelope、完整 v1/v2 semantic parity |
| Catalog | 结构化 source/workload/runtime/device/plan/calibration/performance/release record；production 为空 | builder、reviewed record、revocation 发布流程 |
| Identity | typed length-prefixed canonical encoder；record/source/workload/runtime/device/resolution digest | 跨语言 golden、request/component 完整 digest 边界确认 |
| Resolver | select、authorize、internal-only authority、snapshot identity 检查，已接 engine exact resolve | 执行前 source lease revalidation、错误优先级收口 |
| Session | probe/compile/generate_resolved 默认拒绝 hook | 四模型 override 和共享 helper |
| C API | options query；engine model/root/container identity；resolve API；public generate 分支；prepare unsupported | null/busy/ownership/cancel测试、stable结构化错误、RunResult public metrics |
| Swift | selector/options/v2 request、resolution/error类型、resolve API | semantic parity、App transaction、旧 job 迁移 |
| Data plane | compiler、StageExecutor、slot safety、multi-pool、MLX pager | public authority 接线；不新增第二套 executor |
| Model | 四模型 private descriptor/PlanView/candidate adapter | public probe/snapshot/generate_resolved 和 full-request record |
| Evidence | private P1 anchors、若干 memory peak | full-request tree peak、五档筛选、swap 对照、独立 review |

生产状态必须继续表述为：框架施工中，production catalog 为空，没有公开可执行档位。
当前 exact C ABI 之后的收口设计、共享轻量校验和错误优先级以
[32](32-public-streaming-completion-spec.md)为准。

## 3. 不可破坏的实现不变量

### 3.1 默认路径不变量

selector absent 或 disabled 时，tc_engine_generate 必须在任何 public helper 之前选择原路径。该分支不得：

- 访问 production catalog；
- 创建 probe、snapshot、authority 或 metadata cache key；
- hash model source；
- 再次 tokenize；
- 创建线程、slot、pool、sampler；
- 调用 unload、clear cache 或额外 GPU synchronize；
- 在 block hot path 增加 public 条件判断；
- 改变 legacy resident、legacy streamed、memory_constrained 或 GPU+ANE 行为。

### 3.2 权限不变量

- JSON selector 只是 intent；
- expected_resolution_digest 只是 stale/replay 检查；
- StreamingAuthority 只允许 native resolver 私有构造；
- authority 不序列化、不持久化、不跨进程传递；
- candidate constructor 的 allow_experimental_streaming 不能转换成 public authority；
- public execution 必须同时持有 record、probe、snapshot、device 和 authority；
- generate_resolved 不能根据 target 重新挑布局；
- public 失败不能自动回 resident。

### 3.3 生命周期不变量

- probe 不创建 GPU/MLX backing 和 worker；
- snapshot 只保存 immutable metadata、layout、source lease 和模型 PlanView；
- pool、tickets、mailbox、cancel generation 和 counters 每请求创建；
- source lease 必须晚于 fill worker、reader fence、completion callback 和 pager 销毁；
- success、cancel、failure 都要 drain；
- drain 不安全时保留 owner 并 quarantine，不能释放可能仍被 GPU/worker 引用的内存；
- result 的 actual layout 与 authorized layout 不同即失败。

### 3.4 内存语义不变量

Public target T 是经过校准的完整请求目标，不是：

- 机器物理内存；
- MLX active bytes；
- denoiser-only budget；
- macOS hard cap；
- 零 swap 保证。

记录可用的必要条件是：

~~~text
M_catalog + H(T) <= T
H(T) = max(512 MiB, ceil(10% × T))
~~~

M_catalog 必须覆盖从 engine/worker 创建到安全清理结束的完整 process-tree peak。

## 4. 端到端调用图

### 4.1 Streaming off

~~~text
App NativeRequest v1 / selector disabled
  → tc_engine_generate
  → parse request
  → old global GPU lock / DeviceLease / memory_constrained logic
  → ModelSession::generate
  → existing result
~~~

这条调用链不构造任何 public streaming 对象。

### 4.2 Options 查询

~~~text
App draft
  → NativeEngine.streamingOptions(v2)
  → tc_streaming_options_json
  → request-level cheap validation
  → production catalog tentative filtering
  → available/unavailable target list
~~~

Options 不持有 engine，不读取 payload 权重，不授予 authority。它可以因为缺少 exact source/token identity 返回 tentative。

### 4.3 Exact resolve

~~~text
App frozen draft
  → engine.resolveStreaming(v2 intent)
  → engine try-lock
  → parse + validate selector/model/container
  → session.probe_public_streaming
  → resolver.select
  → session.compile_public_streaming
  → resolver.authorize
  → return exact selector + summary
~~~

Exact resolve 不取得 process-global GPU mutex，不创建 DeviceLease，不调用 configure_streams，不启动 sampler/pool/worker，也不调用
load、prepare、generate。

### 4.4 Public generate

~~~text
exact selector or memory-tier selector
  → engine try-lock
  → exact resolve in same engine/container
  → acquire global GPU lock
  → execution-time source/revocation/device revalidation
  → build immutable ResolvedRequestExecution
  → session.generate_resolved
  → model adapter uses authorized canonical manual config
  → existing StageExecutor/pager/kernel
  → actual-plan verification
  → result
~~~

首版允许 generate 内部重新 resolve；App 先调用 resolve 是为了展示、持久化和 stale 反馈，不是把 authority 从一次 C 调用携带到另一次。
每次 generate 都必须在当前 engine 内重新铸造 authority。

## 5. Engine、锁和 helper 拆分

### 5.1 tc_engine 新字段

建议最终结构：

~~~cpp
struct tc_engine {
    std::string model_id;
    std::filesystem::path model_root;
    std::unique_ptr<tc::ModelSession> session;
    std::mutex mutex;
    std::atomic<bool> cancelled{false};
    std::atomic<bool> memory_quarantined{false};
    std::atomic<bool> streaming_quarantined{false};
    bool allow_experimental_streaming = false;
    std::optional<tc::MemoryExecutionReport> last_memory_report;
    tc::streaming::MetadataResolutionCache streaming_metadata_cache;
};
~~~

model_id 和 model_root 必须在普通 constructor 与 private candidate constructor 中都赋值。public source identity 仍来自 session
probe 和 verified artifacts，不能仅凭 path 字符串授权。

### 5.2 锁顺序

全项目冻结以下顺序：

~~~text
engine mutex
  → metadata cache mutex（若实现）
  → global execution mutex
  → model/session internal mutex
  → executor owner-thread state
~~~

禁止在持有 global execution mutex 时做 options、catalog scan、tokenization 或文件 digest。禁止从 completion callback 反向获取 engine
mutex。resolve 只获取 engine mutex；generate 在 resolve 完成后再获取 global execution mutex。

### 5.3 C API 内部 helper

不要继续把所有分支堆入 tc_engine_generate。建议拆出：

~~~cpp
Request parse_engine_request(const char *json);

ResolvedRequestExecutionPtr resolve_public_streaming_locked(
    tc_engine &, Request, ResolvePurpose);

RunResult execute_public_streaming_locked(
    tc_engine &, ResolvedRequestExecutionPtr,
    const Event &, std::atomic<bool> &);

RunResult execute_legacy_locked(
    tc_engine &, Request, const Event &, std::atomic<bool> &);
~~~

ResolvePurpose 至少区分 metadata_response 与 immediate_generate。两者使用相同选择和 authorize 代码；immediate_generate 额外执行
revalidation，并把 resolved internal request 交给 generate_resolved。

### 5.4 分支位置

建议 tc_engine_generate 的顶层形状：

~~~cpp
auto request = parse_engine_request(json);
const bool public_active = request.streaming_selector &&
                           request.streaming_selector->active();

if (!public_active)
    return generate_legacy_exactly_as_before(engine, std::move(request), ...);

auto resolved = resolve_public_streaming_locked(engine, std::move(request),
                                                ResolvePurpose::immediate_generate);
return generate_public(engine, std::move(resolved), ...);
~~~

关键不是函数名，而是 off 分支必须在 catalog、probe、digest、cache 和 public audit 之前成立。

## 6. Request 冻结和 authority 设计

### 6.1 三份 request 表示

一次 public 请求需要严格区分：

1. requested wire request：用户选择 memory tier；
2. exact public selector：包含 preset/revision/catalog/resolution digest；
3. internal execution request：清除 selector，写入 reviewed canonical manual config。

internal request 只能存在于 ResolvedRequestExecution，不能返回 App、不能保存为普通可编辑 request、不能再次进入 public parser。

### 6.2 冻结顺序

~~~text
copy parsed request
  → validate model equals engine model
  → probe exact workload/source/runtime
  → select record
  → compile snapshot from record canonical config
  → authorize snapshot
  → clear internal_request.streaming_selector
  → clear internal_request.streaming_selector_requested
  → set internal_request.streaming = record.plan.canonical_config
  → freeze all fields in shared immutable execution
~~~

若现有 Request 包含 legacy residency、memory budget、memory_constrained 或 manual streaming，schema parser 应在 probe 前返回
streaming_config_conflict。

### 6.3 Digest 覆盖矩阵

| Digest | 必须覆盖 | 不应覆盖 |
|---|---|---|
| record | record 全字段，除自身 digest | JSON key order、本机 path |
| source | variant/format/manifest/snapshot | raw prompt、output path |
| workload | shape/frames/fps/steps/token shape/feature flags/container | seed、输出文件名 |
| runtime | build/adapter/reader/kernel/allocator revisions | UI version string |
| layout | canonical P/G/K/D/Q、pool、pass、field/materialization identity | runtime counters |
| resolution | record/source/workload/runtime/device/layout/component | 可变 telemetry |
| request | wire semantic fields | JSON whitespace/key order |

当前 authority 的 matches 实现至少要补足 workload/device/component 的明确绑定测试，不能只依赖 record 已经间接包含这些字段。

### 6.4 TOCTOU revalidation

执行前按成本从低到高重验：

1. record 仍在当前 catalog 且未 revoked；
2. engine model 与 snapshot model 相同；
3. device identity 未变化；
4. source quick identity 未变化；
5. opened file descriptor 与 named path 仍指向同一 inode/size/mtime；
6. snapshot layout/component/runtime digest 与 authority 相同。

若安装系统能提供 content digest，content digest 是信任依据；stat/inode/mtime 只用于快速发现变化。

## 7. Catalog、选择和配置文件

### 7.1 Production catalog 形态

production catalog 应由工具生成 C++/JSON 双产物，不接受手写 public record。推荐目录：

~~~text
configs/streaming/catalog/
  schema.json
  proposed/
  reviewed/
  revoked/

generated/streaming/
  production_catalog.inc
  production_catalog.json
  catalog_manifest.json
~~~

若仓库不希望新增 generated 顶层，可放到 native/runtime/streaming/generated，但必须与工具临时输出和 evidence 分离。

### 7.2 Record 示例

~~~json
{
  "id": "zimage-original-512-s9-gpu-12g-p14-k2-r1",
  "revision": 1,
  "catalog_revision": "tc-streaming-public-2026-09-r1",
  "source": {
    "model_variant": "z-image-turbo-original-bf16",
    "weight_format": "safetensors-bf16",
    "artifact_manifest_digest": "...",
    "source_snapshot_digest": "..."
  },
  "workload": {
    "model": "z-image-turbo",
    "operation": "image.generate",
    "execution": "gpu",
    "execution_container": "embedded_app",
    "width": 512,
    "height": 512,
    "frames": 1,
    "fps": 24,
    "steps": 9,
    "batch": 1,
    "audio": false,
    "dynamic_text": true,
    "approximation": false,
    "token_shapes": []
  },
  "device": {
    "minimum_physical_memory_bytes": 17179869184,
    "maximum_physical_memory_bytes": 0
  },
  "plan": {
    "component_policy_revision": "zimage-components-v1",
    "pass_transition": "reload",
    "multi_pool_policy": "serial",
    "canonical_config": {
      "schema_version": 1,
      "enabled": true,
      "retention": "request",
      "stages": {
        "transformer": {
          "residency": "streamed",
          "resident_prefix_blocks": 14,
          "block_group_size": 1,
          "slot_count": 2,
          "prefetch_distance": 0,
          "io_workers": 1
        }
      }
    }
  },
  "calibration": {
    "scope": "full_request_process_tree",
    "calibrated_request_bytes": 0,
    "complete": false
  },
  "release": {
    "channel": "proposed",
    "revoked": false
  }
}
~~~

以上只是 schema 例子，calibrated_request_bytes 为 0 且 complete=false，因此 builder 必须拒绝进入 production。

### 7.3 选择顺序

Memory tier 选择必须稳定：

1. exact model/source/workload/runtime；
2. device class/container；
3. public release 且未 revoked；
4. calibration complete；
5. M + H(T) <= T；
6. performance rank；
7. calibrated bytes；
8. logical read bytes；
9. stable ID/revision。

Preset selector 是 exact replay。任一 identity 不匹配返回 stale、revoked 或 unsupported，不回退到同 target 的另一个 record。

## 8. 通用模型 adapter 架构

### 8.1 复用方式

不建议为每个模型复制 resolver glue。增加仅供 native runtime 使用的共享 helper：

~~~cpp
template <class Probe, class Snapshot>
std::shared_ptr<const Snapshot> checked_public_snapshot(
    const ResolvedRequestExecution &, std::string_view expected_model);

void validate_public_execution_common(
    const ResolvedRequestExecution &,
    const ModelStreamingSnapshot &,
    std::string_view expected_component_revision);

void attach_public_streaming_result(
    RunResult &, const ResolvedRequestExecution &,
    const Layout &, const ExecutionCounters &);
~~~

共享 helper 只做类型/identity/result 检查，不知道 safetensors、MLX、LTX C handle 或 H3 pass。

### 8.2 Probe 基类约束

每个 probe 必须回答：

- exact model variant 和 weight format；
- verified artifact manifest 与 source snapshot；
- expanded workload，包括 token valid/padded/compute rows；
- runtime/adapter/reader/kernel/allocator revisions；
- execution container；
- component policy revision。

Probe 构造期间允许读取 JSON/index/header/tokenizer metadata；禁止读取完整 payload、创建模型 arrays 或启动 worker。

### 8.3 Snapshot 基类约束

Snapshot 必须持有：

- 与 probe 相同的 source/runtime identity；
- generic Descriptor；
- compile_layout 产生的 Layout；
- 现有模型 PlanView；
- source lease/opened descriptors；
- component policy revision。

Snapshot 不持有正在运行的 StageExecutor、ticket、mailbox 或 cancel flag。

### 8.4 generate_resolved 固定模板

~~~text
checked downcast model snapshot
  → authority.matches(record, snapshot, device)
  → snapshot.metadata.check_unchanged
  → verify internal request canonical config
  → create request-scoped exact adapter/owner
  → run existing pipeline
  → finish/drain
  → compare actual layout/counters/source reads
  → attach public selection metrics
~~~

异常路径必须在 snapshot/source lease 析构前完成 drain。若 drain 失败，将 owner 转移到 session/process quarantine 容器。

## 9. Multi-slot、multi-pool 和 overlap 运行时

### 9.1 Slot 状态机

每个 slot 必须经历：

~~~text
Vacant
  → Filling(ticket generation)
  → Ready(content identity)
  → Claimed(group/pass)
  → Reading(reader set sealed)
  → Retiring(all last readers complete)
  → Vacant(next generation)
~~~

任何跨 generation completion、重复 claim、未完成 reader 的 overwrite 都是 correctness failure，不是性能降级。

### 9.2 K、D、Q 的约束

首版 compiler/runtime 继续冻结：

~~~text
1 <= K <= adapter.max_slots
0 <= D < K
1 <= Q <= K
suffix group count >= K
~~~

通常：

- K=1：无法稳定 overlap，只用于边界/诊断；
- K=2：当前四模型最实用的 ping-pong 基线；
- K=3：允许更深预取，但增加 backing peak；
- D=0：仅在同步/claim-after-fill 模式合法；
- D=1：K2 的标准下一组预取；
- Q>K 没有意义，且增加线程/竞争。

Public v1 不允许运行时根据 I/O 延迟自动改变 K/D/Q；动态调整会破坏 record memory/performance identity。

### 9.3 稳态时序

K2/G1/D1 的目标时序：

~~~text
owner: fill g0 synchronously
owner: claim g0
io:    fill g1 into other slot
gpu:   encode g0
owner: retire g0 after all reader fences
owner: claim g1
io:    fill g2 into vacant slot
gpu:   encode g1
...
~~~

对于同步 backend，overlap_next_fill_after_claim 允许 claim 后、encode 阻塞前启动下一次 fill。该优化只能由 adapter 显式声明，
且必须保留 slot safety tracker 的 reader 生命周期。

### 9.4 pass transition

reload：每个 pass 从第一组重新 fill，适合默认模型。

carry_first_group：仅在单 pool、K2、G1、多个 pass、稳定 cyclic mapping 下，把下一 pass 第一组在上一 pass 末尾预取并保持 Ready。
H3 Turbo 当前使用该合同。carry 不是隐式 cache hit；每 pass 的每个 suffix group 仍恰好读取一次。

### 9.5 multi-pool

Flux 9B 有 dual/single 两类布局：

- serial：class barrier 后销毁旧 pool、创建新 pool；峰值低，但 steady allocation 可能增加；
- retain_all：setup 时创建所有 pool，请求结束统一销毁；steady allocation 为零，峰值为 pool 之和。

策略属于 layout identity。adapter 必须通过 supports_multi_pool_policy 显式支持，不能在运行时自行切换。Public record 必须同时记录：

- pool 数和 class order；
- 每 pool slot capacities；
- serial/retain_all；
- barrier 数；
- setup/steady allocation audit。

## 10. 四模型代码接入设计

### 10.1 LTX 2.5 distilled

首卡：C/Metal dense T2V、512×320×33、11 steps、24 fps、无 audio/I2V/Sol/LoRA/ANE。

代码落点：

- native/platform/apple/ltx_session.mm：override 三个 public hook；
- native/models/ltx_runtime/ltx_streaming_descriptor.*：probe metadata/source；
- native/models/ltx_runtime/ltx_streaming_plan.*：snapshot 持有 PlanView；
- native/models/ltx_runtime/ltx_streaming_adapter.inc：复用 exact native execution；
- apps/macos/LTXWorker.swift：两阶段 worker resolve/generate 或 ACK 协议。

LtxPublicProbe：

- original distilled checkpoint；
- 48 blocks 和字段/materialization；
- Gemma tokenizer/template/token rows；
- connector、stage1/stage2、upsampler、VAE/export component identity；
- execution_container=ltx_cli_worker。

LtxPublicSnapshot：

- ltx::StreamingPlanView；
- StreamingMetadata/source fd lease；
- exact canonical config；
- request generation；
- component policy revision。

执行注意：App 进程不能替 worker 铸造 authority。首选两阶段：worker resolve 退出，App 持久化 exact selector，再启动 fresh worker generate，
worker 重新核验。后续若引入 ACK，worker 必须在 App 持久化成功前不创建 GPU backing。

### 10.2 Z-Image Turbo

首卡：original BF16、GPU、T2I、512²、9 steps、无 GGUF/LoRA/ANE。

代码落点：

- native/models/z_image/z_image.hpp/.cpp；
- native/models/z_image/streaming_descriptor.hpp；
- native/platform/apple/z_image_streaming_descriptor.mm；
- native/models/z_image/weight_stream.hpp；
- native/platform/apple/z_image_weight_stream.mm。

固定首轮族：G1/K2/D0/Q1，只扫描 prefix。Snapshot 复用 z_image::StreamingPlanView 和现有 exact weight source。

必须验证：

- 30 blocks；
- P29 因 suffix<K2 拒绝；
- token valid/padded/compute rows进入 workload；
- VAE 前权重释放进入 component revision；
- 512²/1024²不同 record；
- old zImageStreamingBudgetGiB 只属于 legacy 路线，不映射 public target。

### 10.3 MiniMax H3 Turbo

首卡：original BF16 C/Metal、512²×22 frames、4 steps、GPU-only。

代码落点：

- native/platform/apple/h3_session.mm；
- native/models/h3_runtime/h3_streaming_descriptor.*；
- native/models/h3_runtime/h3_streaming_policy.*；
- H3 transformer source/pass hooks。

固定 G1/K2/D1/Q1 和 carry_first_group。Probe 必须拒绝普通 H3、MLX H3、FastH3、quant cache、非 original BF16 shards。
Snapshot 记录 50 blocks、四 pass、单次大 pread、reader revision 和 component lifecycle。

Public generate 不能永久设置 allow_experimental_streaming。private candidate bool 继续只服务开发验证。

### 10.4 Flux.2 Klein 9B

首卡：9B BF16 eager、GPU T2I 512²、4 steps、无 LoRA/ANE；4B compiled graph 不在范围。

代码落点：

- native/models/flux2/streaming_descriptor.hpp；
- native/platform/apple/flux_streaming_descriptor.mm；
- native/models/flux2/flux_streaming.hpp；
- native/models/flux2/flux_transformer.cpp；
- native/models/flux2/pipeline.cpp；
- native/runtime/streaming/mlx_weight_pager.*。

第一阶段只公开候选 P0/G1/K2/D1/Q2，保留 dual/single 两个 class pool。第二阶段实现 prefix：

- P2/P4/P6/P8/P12；
- P7 因 dual suffix<K2拒绝；
- P8 消除空 dual suffix pool；
- block 8 concatenate 每 pass 恰一次；
- prefix arrays 每请求加载一次，prefix block 每 pass 仍计算一次；
- pool ID 不等同 class enum/vector index。

## 11. App 高级设置和任务事务

### 11.1 UI 状态

~~~swift
enum StreamingMode: String, Codable, Sendable {
    case off
    case memoryTier
}

struct StreamingChoice: Codable, Sendable, Equatable {
    var mode: StreamingMode = .off
    var targetBytes: UInt64?
}
~~~

高级设置只展示：开关、native 返回的可用 target、推荐标记、不可用原因和“会降低峰值但可能变慢”的说明。

### 11.2 物理内存推荐

推荐只在 available targets 中选择：

~~~text
reserve(P) = max(4 GiB, ceil(20% × physical memory P))
recommendation_cap = P - reserve(P)
recommended = largest available target <= recommendation_cap
~~~

用户可以选其他 available target。App 不显示 catalog 中不适用于当前 exact workload 的档位，也不根据 free memory 临时生成布局。

### 11.3 Options cache key

至少包含：

- model/operation；
- width/height/frames/fps/steps/audio；
- execution policy/container；
- dynamic text/approximation；
- model installation identity revision；
- active LoRA/input feature digest；
- catalog revision；
- physical device class。

任一字段变化即使 debounce，也必须使旧 response revision 失效。

### 11.4 提交事务

~~~text
capture immutable draft revision
  → validate App fields
  → build v2 intent
  → acquire correct embedded engine or LTX worker
  → exact resolve
  → confirm draft/target/source/catalog unchanged
  → atomically persist frozen job
  → generate using exact selector
  → atomically persist terminal result/error
~~~

persist 失败不得启动 GPU。generate stale/revoked 后 job 进入 failed，不自动换档；用户重试会创建新 resolution 和新 job attempt。

### 11.5 Job schema

NativeJob 需要自定义 Codable，兼容当前 request: NativeRequest。新 schema 建议保存 request_payload、streaming_submission 和 terminal
result。unknown future schema 只读失败，不能把 jobs 文件重写为空。

公共 accessor 必须替换 App 各处对 job.request 的直接依赖：modelID、outputURL、inputPaths、routeSummary、requestPayload、
streamingSummary、canReplayExactly。

## 12. Calibration、候选探索和 catalog 工具链

### 12.1 工具分层

建议实现以下独立命令：

| 工具 | 输入 | 输出 | 是否可发布证据 |
|---|---|---|---:|
| inspect_streaming_candidates | descriptor/workload | legal candidate list | 否 |
| simulate_streaming_schedule | candidate + measured distributions | predicted wait/peak/wall | 否 |
| explore_streaming_presets | bounded policy | raw candidate runs | 否 |
| calibrate_streaming_memory | exact candidate/card | timestamped process-tree samples | 是，需验证 |
| verify_streaming_calibration | raw samples/result | independent peak/coverage verdict | 是 |
| verify_streaming_performance | ABBA raw results | P0/P1/P4 statistics | 是 |
| build_streaming_catalog | verified bundles + review | proposed/reviewed catalog | 是 |

工具不能链接进 release runtime，也不能在普通 App 请求中启动 sampler。

### 12.2 Candidate ID

候选 ID 必须可由 canonical fields 重建：

~~~text
model/workload/container
  + P/G/K/D/Q per stage
  + pass transition
  + multi-pool policy
  + component policy revision
  + adapter/reader/kernel revision
~~~

不要把运行编号、时间戳或机器用户名作为 candidate identity。

### 12.3 Process-tree sampler

采样记录至少包括：

- monotonic timestamp；
- pid、start identity、role、parent；
- physical footprint/resident/compressed；
- swapin/swapout counters；
- MLX active/peak/cache（若进程使用 MLX）；
- native Metal accounted backing；
- phase/event sequence；
- pressure/thermal state；
- sample gap 和 sampler reset epoch。

Coordinator 不加载模型。worker launch/exit 事件补齐短命进程；最大 sample gap 超过 record 上限则 calibration incomplete。

### 12.4 候选剪枝

先静态拒绝：

- suffix<K；
- D>=K 或 Q>K；
- adapter 不支持的 multi-pool/pass transition；
- pool slot capacity overflow；
- source/materialization incomplete；
- component floor 已超过 target；
- workload/source/runtime 与 card 不匹配。

再保留 Pareto candidate：更低 predicted peak、更少 logical reads、更低 predicted wait，且至少一项严格更好。

### 12.5 Swap 对照

按三层运行：无压力、真实低内存设备、受控 pressure。受控 pressure 需要单独授权、固定 helper 上限/timeout/cleanup，不能关闭系统 swap，
不能无限分配。比较必须同一设备、同一 source/card/container/seed/output。

只有 wall speedup 的 95%区间下界大于 1，且质量/成功率不差，才可声称比 swap 更快。否则准确报告：峰值、swap、tail 和速度的实际取舍。

## 13. 逐 PR 代码实施单

### 13.1 R1/R2 收口：identity、resolver、authority

当前相关文件：

- native/runtime/streaming/canonical_encoding.*；
- native/runtime/streaming/preset_catalog.*；
- native/runtime/streaming/preset_resolver.*；
- native/runtime/streaming/resolved_request.*；
- native/runtime/session.hpp；
- tests/native/streaming_preset_resolver_test.cpp。

收口要求：

1. record validation 覆盖全部 mandatory identity；
2. catalog revision 与 record revision 一致；
3. canonical encoder 有 C++/Python golden；
4. authority 绑定 workload、device、component 和完整 runtime digest；
5. empty catalog 稳定返回 catalog_has_no_public_records；
6. default ModelSession hooks 全部拒绝；
7. release build 无 authority/test catalog mint symbol。

### 13.2 R3：Exact C ABI

新增 header symbol：

~~~c
int tc_engine_resolve_streaming_json(
    tc_engine *engine,
    const char *request_json,
    char **result_json,
    char **error);
~~~

响应最少包含：schema_version、status、request_digest、resolution_digest、catalog_revision、exact_selector、selection summary、
source/runtime/layout/component digests、execution_container 和 memory calibration summary。

prepare 对 active public selector 固定返回 streaming_prepare_unsupported。load/unload 不获得 public authority。

### 13.3 R4：Generate 分流

1. 提取 legacy helper，保持现有代码顺序；
2. active selector 走 resolve；
3. 空 production catalog 在 global GPU lock 前失败；
4. valid resolution 后才 acquire DeviceLease/configure MLX；
5. revalidate；
6. 调 session.generate_resolved；
7. actual plan mismatch 失败并 drain；
8. public failure不调用普通 generate。

### 13.4 R5：Swift/App

新增 NativeStreamingResolution、NativeAPIErrorEnvelope、NativeFailure.code/retryable/action、NativeEngine.resolveStreaming。
App 新增 StreamingOptionsStore、StreamingChoice、job schema v2、RunInsights public streaming summary。

先完成 Codable/transaction tests，再显示 UI；catalog 为空时 UI 可进入但只能显示无已发布方案。

### 13.5 R6：模型 adapter

每个模型一个独立 PR，包含：probe、snapshot、generate_resolved、test-only record、GPU smoke、fault/lifecycle、actual-plan result。
不得同 PR 修改 production record、kernel 优化和默认 App 选择。

### 13.6 R7：工具与 reviewed record

工具先输出 proposed；独立 verifier 和 reviewer 通过后才生成 public-experimental/public-stable。record-only PR 不修改 executor/kernel。

### 13.7 R8：合并 dev 和 ANE 回归

public v1 继续 GPU-only。解决冲突时保留 dev 的 validated device profile 和 ANE scope；active public selector + gpu_ane 无 record 时返回
streaming_route_unsupported。重跑 GPU default、ANE default、Core ML cache、Z-Image manifest release 和 App selector-off。

## 14. 自动化测试设计

### 14.1 Host tests

| ID | 场景 | 通过条件 |
|---|---|---|
| PUB-H001 | canonical primitive/optional/list | digest golden 一致 |
| PUB-H002 | JSON order/whitespace | record digest 不变 |
| PUB-H003 | false/0/absent | digest 三者不同 |
| PUB-H004 | record mandatory fields | 任一缺失 fail-closed |
| PUB-H005 | deterministic rank | 输入乱序仍选同 record |
| PUB-H006 | exact replay stale | 不替换另一个 preset |
| PUB-H007 | revoked/incomplete record | 不可选择 |
| PUB-H008 | source/workload/runtime/device mismatch | stable code 正确 |
| PUB-H009 | snapshot layout mismatch | streaming_actual_plan_mismatch |
| PUB-H010 | authority copy/deserialize | 编译或 API 层不可用 |
| PUB-H011 | default session hooks | 全部 unsupported |
| PUB-H012 | empty catalog | catalog_has_no_public_records |

### 14.2 C ABI tests

| ID | 场景 | 通过条件 |
|---|---|---|
| PUB-A001 | null engine/input/out/error | 无崩溃，ownership 正确 |
| PUB-A002 | engine model mismatch | GPU lock 前失败 |
| PUB-A003 | resolve busy | engine_busy，cancel flag 不变 |
| PUB-A004 | resolve empty catalog | stable code，session/GPU 未进入 |
| PUB-A005 | active prepare | streaming_prepare_unsupported |
| PUB-A006 | active generate empty catalog | global GPU lock/DeviceLease 为零 |
| PUB-A007 | off generate | legacy path/result/audit 不变 |
| PUB-A008 | exact stale/revoked | 无 ordinary generate fallback |
| PUB-A009 | result/error memory | success 只 result，failure 只 error |
| PUB-A010 | release symbols | 无 test setter/authority mint |

### 14.3 Swift/App tests

| ID | 场景 | 通过条件 |
|---|---|---|
| PUB-S001 | v1/v2 semantic snapshot | 支持字段逐项等价 |
| PUB-S002 | explicit false/0/absent | round-trip 不丢语义 |
| PUB-S003 | options debounce/out-of-order | 旧 revision 不覆盖新状态 |
| PUB-S004 | streaming off | 零 options/resolve 调用 |
| PUB-S005 | catalog empty | 文案准确，App 可继续默认生成 |
| PUB-S006 | persist failure | generate 调用次数为零 |
| PUB-S007 | stale/revoked | 不自动换 target/preset |
| PUB-S008 | legacy job migration | 无损读取，必要时保持旧格式 |
| PUB-S009 | unknown future job | 不覆盖 jobs 文件 |
| PUB-S010 | LTX worker resolve | authority 不跨进程，container 正确 |

### 14.4 Executor/lifecycle tests

| ID | 注入点 | 必须验证 |
|---|---|---|
| PUB-L001 | pool create | 无 worker/ticket 泄漏 |
| PUB-L002 | first fill | source lease 保持，安全 drain |
| PUB-L003 | mid fill/read | slot generation 不错乱 |
| PUB-L004 | encode/reader fence | 未完成 reader 不释放 backing |
| PUB-L005 | mailbox overflow | 请求失败并 quarantine/drain |
| PUB-L006 | cancel before dispatch | 零 GPU group submitted |
| PUB-L007 | cancel during prefetch | join worker，pending fill=0 |
| PUB-L008 | cancel during GPU read | 等 fence 后销毁 |
| PUB-L009 | VAE/export failure | transformer backing 已按 policy 安全释放 |
| PUB-L010 | unsafe drain | engine/process quarantined |
| PUB-L011 | mode transition | resident→public→resident 无残留 |
| PUB-L012 | target A→B→A | 无 cache/layout authority 串用 |

## 15. 性能和内存验收

### 15.1 P0 默认路径

每个 runtime PR 都做 default resident、legacy streamed、memory_constrained、GPU+ANE（合并 dev 后）的 ABBA 对比。

阻断线沿用文档 12：

- wall median ratio ≤1.02；
- wall P95 ratio ≤1.05；
- bootstrap 区间跨线为 INCONCLUSIVE，不记 PASS；
- default public audit counters 为零。

除 wall 外还比较 engine create、load、first token/conditioning、denoise、VAE/export、unload 和 process peak。

### 15.2 P1 同布局框架成本

Direct 与 generic 必须保持完全相同 P/G/K/D/Q、source、kernel、reader、component policy、cold/warm、seed/output。
已有 private anchors继续作为回归下限，但 public adapter 接线后需要用 ordinary public constructor/test authority 重跑。

### 15.3 P4 策略取舍

每个 target 输出：

- full-request peak 和余量；
- wall/denoise median、P95、置信区间；
- logical/physical reads；
- refill wait、GPU idle、worker utilization；
- pool/slot backing；
- swap/fault/pressure；
- quality、成功率、取消/失败率。

P4 不允许用一个统一 2% 门槛掩盖低内存收益；是否发布由 target-specific product policy 决定，但证据必须完整。

### 15.4 当前锚点的使用边界

当前 private 同布局结果可保护框架：

- LTX P8/G1/K3/D2/Q3；
- Z-Image P14/G1/K2/D0/Q1；
- H3 Turbo P0/G1/K2/D1/Q1；
- Flux 9B P0/G1/K2/D1/Q2。

Flux resident→streaming 的 MLX peak下降也只是候选证据。这些数据没有完整 public source/workload/container/calibration/review，
不得直接生成 production record。

## 16. 单模型/单 target 验收卡

每一条 public record 都要有独立卡：

~~~text
identity
  model/variant/format/source digest
  workload/token/component/container
  build/adapter/reader/kernel/allocator/device

layout
  stage P/G/K/D/Q
  pass transition/multi-pool
  prefix/resident/pool/source bytes

correctness
  matched outputs
  actual layout digest
  source-read and pass counts

memory
  fresh/warm/transition full-request tree peak
  coverage/sample-gap verifier
  M + H(T) <= T

performance
  P0/P1/P4
  swap comparison if claimed

lifecycle
  cancellation/read/fence/drain/quarantine

product
  ordinary App/worker resolve→persist→generate
  unavailable/stale/revoke UX

release
  evidence digest/reviewer/commit/channel/revoke drill
~~~

同一模型的 512/1024、22/39/73 frames、不同 token bucket、不同 container 都是不同卡。

## 17. Result、诊断和隐私

Public success result 至少输出：

- requested target；
- exact preset key/catalog revision；
- resolution digest；
- execution container；
- memory calibration scope/value；
- canonical resolved layout summary；
- actual layout digest；
- pool/fill/read/wait counters；
- selection_source=public_preset_registry。

off/default 不输出 public selection 对象。observed peak 未采样时为 null，不写 0。错误/telemetry 不记录 raw prompt、本机绝对模型路径、
输出路径或用户文件名；可记录不可逆 feature/source digest。

## 18. 错误优先级和恢复动作

建议优先级：

~~~text
JSON/schema/duplicate key
  → config conflict
  → engine/model/container mismatch
  → empty/no public record
  → source verification/change
  → workload/device/runtime mismatch
  → target fit
  → compile/layout mismatch
  → stale/revoked
  → runtime execution/lifecycle failure
~~~

同一输入在 options/resolve/generate 返回同类 stable code。UI 根据 code 决定重选 target、重新验证安装、重建 engine 或关闭 streaming，
不能通过 message substring 判断。

## 19. 回滚和发布策略

回滚层级从小到大：

1. revoke 单条 record；
2. 删除某 target 的 availability；
3. 隐藏 App advanced 开关；
4. 禁用 exact resolve C ABI；
5. 回滚单模型 public adapter；
6. 回滚通用 resolver runtime；

无论回滚到哪层，active selector 都只能明确失败，不能 silent resident fallback。off/default 和 legacy 路线必须继续可用。

首次 public release 建议只发布一个模型、一个 workload、一个 container、一个或少量 target。五档不齐全不是失败；填入未验证档位才是失败。

## 20. Definition of Done

### 20.1 框架完成

- [ ] selector→probe→select→snapshot→authority→generate_resolved 闭环；
- [ ] authority 不可从 wire 构造；
- [ ] off/default 零 public 控制面和 P0 不回归；
- [ ] exact C ABI/Swift/App transaction 完成；
- [ ] source/revoke/device/runtime/layout revalidation 完成；
- [ ] success/cancel/failure 安全 drain 或 quarantine；
- [ ] LTX 在实际 worker container 内授权；
- [ ] release binary 无 test authority/catalog 注入；
- [ ] dev 合并后 GPU/ANE default 通过。

### 20.2 单模型完成

- [ ] 三个 public session hook；
- [ ] ordinary constructor，不依赖 permanent candidate bool；
- [ ] full pipeline 和 actual plan result；
- [ ] model-specific fault/lifecycle；
- [ ] P0/P1/P4；
- [ ] 至少一个完整 workload card。

### 20.3 单 target 完成

- [ ] complete full-request calibration；
- [ ] M + H(T) <= T；
- [ ] output quality/success；
- [ ] target-specific performance取舍；
- [ ] evidence bundle和独立review；
- [ ] proposed→experimental/stable catalog；
- [ ] replay和revoke drill。

### 20.4 产品完成

- [ ] 高级设置默认 off；
- [ ] 只显示 native available targets；
- [ ] 推荐与资格分离；
- [ ] 用户不编辑 P/G/K/D/Q；
- [ ] unavailable/stale/revoked 文案准确；
- [ ] 旧 draft/job 无损；
- [ ] catalog 为空时 App 安全可用；
- [ ] 不宣传 hard cap、零 swap 或未经统计证明的提速。

## 21. 下一步建议顺序

1. 收口当前 R1/R2 canonical catalog/resolver/authority，并补齐 workload/device digest 测试；
2. 实现 tc_engine model identity 和 exact resolve C ABI；
3. 将 active generate 改为 resolve-before-GPU，并保持空 catalog 早拒绝；
4. 完成 Swift resolution/error 与 v1/v2 semantic parity；
5. 完成 App off-by-default transaction 和 job schema；
6. 先接一原生模型和一 MLX 模型，建议 LTX/Z-Image，验证 adapter 形态；
7. 接 H3 Turbo 和 Flux 9B P0，再做 Flux prefix；
8. 建设 calibration/explorer/verifier/builder；
9. 合并 dev，重跑默认 GPU/ANE；
10. 逐 card/target 发布 reviewed records。

该顺序确保在没有任何 public record 的阶段，系统仍能完整编译、测试和运行所有现有路径；真正开放能力只发生在模型、内存、性能、
生命周期和产品事务都具有可复算证据之后。
