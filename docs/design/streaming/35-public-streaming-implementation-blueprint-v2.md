# 35 · Public Streaming 框架实施蓝图 v2

[目录](README.md) · [收口规格](32-public-streaming-completion-spec.md) · [Runtime/App 工程规格](33-public-runtime-app-engineering-spec.md) · [模型档位与发布](34-model-tier-calibration-and-release-spec.md) · [当前进度](13-implementation-progress.md)

修订日期：2026-09-17。分支：`feat/stream`。代码基线：`fa1ecd0`。状态：**实施蓝图；production catalog 为空，public streaming 尚未开放。**

本文是对 32、33、34 号文档的工程化补充，目标是让 runtime、模型 adapter、App、工具链、性能和 release owner 可以按照同一份施工图实现和验收。本文不改变已经冻结的用户语义、默认路径和 public v1 边界：用户只选择 Off 或内存档位；后台根据 reviewed catalog 选择完整布局；没有 reviewed record 时 fail-closed。

## 0. 阅读方式与事实边界

### 0.1 文档职责

| 文档 | 唯一职责 |
|---|---|
| 01–20 | 通用 layout/compiler/executor 的基础设计、历史实施记录和底层协议 |
| 22 | 面向使用者和实现者的现有框架总览 |
| 23–26 | public memory-tier 产品、探索、校准和发布基础规格 |
| 27–31 | public 控制面、runtime/App 接线、工具和实验执行计划 |
| 32 | 收口语义和最终完成边界 |
| 33 | Runtime、C ABI、Swift、App、JobStore 的逐接口工程合同 |
| 34 | 四模型档位校准、swap 对照、evidence、catalog review 和 release gate |
| **35** | **把上述内容串成可分配的代码任务、状态机、数据结构、测试矩阵和 DoD** |
| 36 | 最新代码基线上的 source lease、receipt v2、四模型逐文件 public adapter 实施规格 |
| 37 | 五档真实校准、性能/Swap 对照、App 事务、evidence/catalog/release 验收规格 |

如果本文与底层协议冲突，以 02、03、09、10 为准；如果本文与当前事实记录冲突，以 13 的最新日期段落为准。本文的“计划”“应实现”“拟议”不能被解释为“已经支持”。

### 0.2 当前绝不能声称的内容

- production catalog 仍为空，App 没有可用的 public target；
- 四模型已有 private exact/candidate，但尚未完成 public probe/compile/generate_resolved；
- actual-plan 汇总 verifier 已在 `c6cba54` 提交，coordinator/provider 已在 `fa1ecd0` 提交并完成 host/contract/App 回归；但四模型仍未生成真实 public receipt；
- 没有完整请求 process-tree calibration，就不能宣称 8/10/12/16/20 GiB 任一档位可用；
- 没有 P3 pressure/swap 对照，就不能宣称显式 streaming 比系统 swap 更快；
- GPU-only public v1 不自动扩展到 GPU+ANE、LoRA、量化缓存、compiled graph 或任意尺寸。

## 1. 目标、非目标和不可破坏的原则

### 1.1 目标

1. 将 block/slot/layer streaming 抽象为模型无关的 control plane、通用 executor 和模型 adapter。
2. 让 App 只暴露 `Off / 8 / 10 / 12 / 16 / 20 GiB`，不暴露 P/G/K/D/Q、pool、pass transition 等实现参数。
3. 让 reviewed catalog record 成为唯一的自动布局来源；同一 workload、source、runtime、device 和 target 必须确定性地得到同一 plan。
4. 在低内存机器上通过显式 slot/pool 调度降低 live working set，避免把所有压力交给系统 swap。
5. 在内存充足且 streaming 关闭时保持当前 default GPU、GPU+ANE、compiled graph、resident cache 路径不变。
6. 让每个成功 public request 都能证明“授权布局 = 实际布局”，并留下可审计的 source/runtime/device/result identity。
7. 让单个 model/target 可以独立校准、review、staged、production、revoke 和 replay。

### 1.2 非目标

- 不在 runtime 中根据瞬时 free memory 动态改变已经授权的 P/G/K/D/Q；
- 不根据 target 直接反推未经校准的 slot 数；
- 不把系统物理内存、GPU VRAM、swap 上限、denoiser-only peak 混为同一语义；
- 不在 public v1 中自动组合 ANE 和 block streaming；
- 不通过 public selector 静默回退到 legacy `residency=streamed`、memory budget、resident 或 swap；
- 不新增一套只服务 public 的 model executor；private candidate 和 public route 必须共享 exact execution core；
- 不允许 App 序列化 authority、GPU 指针、slot backing、file descriptor 或 worker 句柄。

### 1.3 性能不可退化原则

性能分成三个独立问题：

| 维度 | 验收对象 | 规则 |
|---|---|---|
| 默认非侵入 | selector absent/disabled | public validator/catalog/probe/slot/worker/hash/sync 计数均为零；无新增热路径副作用 |
| 同布局框架开销 | direct exact vs generic executor | P0/P1 wall median ≤ 1.02、P95 ≤ 1.05，按 12 的 bootstrap 规则报告 |
| 低内存收益 | resident/streaming/swap pressure 对照 | 分别报告峰值、swap、wall、I/O、失败率；收益不能抵消默认路径回归 |

## 2. 当前代码地图与目标落点

### 2.1 已存在并应复用的模块

| 层 | 当前类型/文件 | 继续承担的职责 |
|---|---|---|
| request/schema | `native/core/contracts.hpp`、`streaming_contracts.hpp`、`request.mm` | v1/v2 request、selector、legacy 冲突标记和严格 byte 解析 |
| public validator | `public_request_validation.*` | request-only 检查；不触碰 catalog、GPU、filesystem、model module |
| layout/catalog | `layout.*`、`preset_catalog.*` | descriptor、StageLayout、PoolLayout、record 和 canonical digest |
| executor | `slot_pool.*`、`io_executor.*`、`context.*` | claim/fill/read/retire、mailbox、multi-pool barrier、quarantine |
| model pager | `mlx_weight_pager.*` | MLX source read、backing、reader completion 和计数 |
| identity | `canonical_encoding.*`、`resolved_request.*` | source/workload/runtime/device/resolution digest、authority |
| API | `native/api/c_api.mm`、C header | resolve/generate/ownership/error conversion |
| Swift | `bindings/swift/TurboCiderNative.swift` | v2 value types、native error、options/resolve/generate async wrapper |
| App | `StudioState.swift`、`JobStore.swift`、`LTXWorker.swift`、`RunInsights.swift` | 草稿、任务事务、worker、结果展示 |
| evidence | `run_streaming_campaign.py`、`verify_streaming_campaign.py`、`run_streaming_audit.py` | P0/P1/audit evidence 和独立 verifier |

### 2.2 已完成拆分与下一步新增模块

`fa1ecd0` 已从 `c_api.mm` 下沉：

~~~text
native/runtime/streaming/public_runtime.hpp/.cpp
  PublicStreamingCoordinator
native/runtime/streaming/catalog_provider.hpp/.cpp
  StreamingCatalogProvider
  production provider
~~~

下一阶段按 36 新增或扩展：

~~~text
native/runtime/streaming/source_lease.hpp/.cpp
  fd-based SourceLeaseDescriptor / SourceLease

native/runtime/streaming/value_probe.hpp/.cpp
  common immutable probe/snapshot

native/runtime/streaming/actual_receipt.hpp/.cpp
  ActualStageReceipt
  ActualExecutionReceipt
  receipt canonicalization/digest
~~~

`c_api.mm` 只保留 C ABI 参数转换、engine lock、错误转换和结果 JSON 拷贝；实际流程由纯 C++ helper 承担，以便 host test 在没有 Metal/App 的环境中覆盖。

## 3. 统一数据模型与不变量

### 3.1 用户意图与 exact plan

用户请求只表达意图：

```json
{
  "schema_version": 2,
  "model": "z-image-turbo",
  "execution": {
    "policy": "gpu",
    "streaming": {
      "schema_version": 2,
      "enabled": true,
      "selection": "memory_tier",
      "retention": "request",
      "target_request_memory_bytes": 10737418240
    }
  }
}
```

native 内部形成不可变链：

```text
StreamingSelector
  → PublicResolveInput
  → ModelStreamingProbe
  → StreamingPresetRecord
  → ModelStreamingSnapshot
  → ResolvedStreamingSelection
  → StreamingAuthority
  → ResolvedRequestExecution
```

必须保持以下不变量：

1. `StreamingSelector` 不含 P/G/K/D/Q；
2. `ModelStreamingProbe` 不创建 GPU backing、pool、worker 或 payload；
3. `StreamingPresetRecord` 是 catalog value，不持有 session 或 mutable state；
4. `ModelStreamingSnapshot` 的 layout/source/runtime identity 与 record 完全一致；
5. `StreamingAuthority` 只能由 `PublicPresetResolver::authorize` 构造，不能 copy、JSON deserialize 或由 App 传入；
6. `ResolvedRequestExecution` 是单次请求快照，不能跨请求复用；
7. adapter 只能消费 snapshot，不能重新解释 target 或修改 layout。

### 3.2 Target 与内存语义

固定 target：

```text
8 GiB  =  8  × 2^30
10 GiB = 10  × 2^30
12 GiB = 12  × 2^30
16 GiB = 16  × 2^30
20 GiB = 20  × 2^30
```

统一 guard：

```text
H(T) = max(512 MiB, ceil(10% × T))
eligible(T) = calibration.complete
           && calibrated_full_request_peak + H(T) <= T
```

`calibrated_full_request_peak` 必须是完整进程树窗口（App/worker、text encoder、denoiser、upsampler/VAE、export 和退出清理），不能填 denoiser-only `mlx_peak`。若独立 upper bound 大于采样峰值，则取较大者。

### 3.3 Layout 参数的严格含义

| 符号 | 含义 | 禁止行为 |
|---|---|---|
| P | 常驻前缀，降低每 pass I/O | target 不够时运行时偷偷缩小 |
| G | 一次 refill 的连续 block 数 | 解释成跳过 block |
| K | 精确 live slot 数 | 当 upper bound 或自动减槽 |
| D | group lookahead，必须 `< K` | 制造未知读者或跨 barrier 预取 |
| Q | refill worker 数，首版不超过 K | 根据瞬时 I/O 无限扩容 |
| pass | denoise pass 及跨 pass 语义 | 把 carry 解释成省略 source fill |
| pool | serial 或 retain_all backing policy | 按瞬时内存自动切换 |

## 4. 端到端事务和锁顺序

### 4.1 Off/default fast path

```cpp
auto request = request_from_json(parse_json(r));
if (!request.streaming_selector ||
    !request.streaming_selector->active()) {
    return existing_generate_path(...);
}
return public_generate_path(...);
```

该分支不得查询 catalog、probe/hash source、创建 resolver/slot/worker、增加 `unload`/cache clear/`synchronize`，也不得改变 default MLX/Metal/ANE/compiled graph 的 resident 生命周期。

### 4.2 Resolve-only 事务

```text
R0  parse request
R1  engine try-lock
R2  request-only validator
R3  engine.model_id / execution_container check
R4  empty-catalog check
R5  make_plan，只做 request normalization
R6  metadata-only model probe
R7  deterministic catalog select
R8  compile immutable snapshot
R9  authorize source/workload/runtime/device/layout
R10 serialize resolution summary
R11 release engine lock
```

Resolve 禁止取得 `tc::execution_mutex()`、`DeviceLease`、configure streams、GPU allocation、reader open、worker create 或 cancel reset。

### 4.3 Generate 事务

```text
G0  out/error pointer 置空
G1  engine try-lock；quarantine 先检查
G2  parse + request-only validator
G3  在 engine 内重新 probe/select/compile/authorize
G4  获取 global GPU lock + DeviceLease
G5  source/runtime/device/catalog revalidation
G6  bind request-scoped streaming context
G7  adapter.generate_resolved
G8  adapter finish/drain；GPU reader 和 I/O mailbox 全部结束
G9  actual receipt hard verify
G10 attach public metrics
G11 serialize result；最后才写 out
G12 success cleanup；失败则 drain/quarantine
```

锁顺序固定为：

```text
engine.mutex
  → tc::execution_mutex()
    → DeviceLease/backend lock
      → adapter session state
        → slot/pool/mailbox internal mutex
```

禁止反向获取 `engine.mutex`，禁止 callback 同步等待持有上层锁的线程。`resolveStreaming` 与 `generate` 都使用 try-lock；busy 返回结构化错误，不能隐式排队。

### 4.4 Cancel、异常和 quarantine

| 事件 | 必须动作 | 允许结果 |
|---|---|---|
| cancel before first fill | 停止新 ticket；等待已提交 reader/fill；drain | cancelled，无 success result |
| cancel during encode | 停止后续 group；等待最后 reader fence | terminal cancelled，slot 不复用 |
| short read/checksum fail | 关闭 reader；保留 backing 到 drain | stable error code |
| drain timeout | 不 free 仍可能被 GPU/worker 访问的 backing | session/engine quarantine |
| authority mismatch | result serialization 前失败 | `streaming_actual_plan_mismatch` |
| source changed | 不使用新 source；销毁或 quarantine 旧 lease | `artifact_changed` |

失败路径必须满足：

```text
result_json == null
&& no successful output is exposed
&& no slot is refilled after failed drain
&& quarantine state is observable
```

## 5. PublicStreamingCoordinator 代码设计

### 5.1 推荐接口

```cpp
class PublicStreamingCoordinator final {
public:
    PublicStreamingCoordinator(
        ModelSession &,
        CatalogProvider &,
        StreamingDeviceIdentity,
        std::string execution_container);

    std::shared_ptr<const ResolvedRequestExecution>
    resolve(Request request) const;

    void revalidate(const ResolvedRequestExecution &) const;
};
```

Coordinator 不拥有 engine/global GPU lock，也不执行 kernel。C ABI 在锁内调用 `resolve`；generate 总是从原始 request 重新 resolve，不能接受 App 传入的 authority。测试使用 fake `ModelSession`、probe、snapshot 和 catalog 覆盖错误排序。

### 5.2 CatalogProvider 注入

```cpp
class CatalogProvider {
public:
    virtual ~CatalogProvider() = default;
    virtual const StreamingPresetCatalog &catalog() const = 0;
};

class ProductionCatalogProvider final : public CatalogProvider {};

#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
class TestCatalogProvider final : public CatalogProvider {
    StreamingPresetCatalog catalog_;
};
#endif
```

要求：

1. production catalog 只能来自编译进来的 reviewed data，不允许 App/环境变量覆盖；
2. test-only injection 不导出到 release C header；
3. 注入 catalog 逐 record 调 `validate_streaming_preset_record`，并校验 canonical digest；
4. provider 不创建 model session 或访问 GPU；
5. `tc_streaming_options_json` 仍只查询 production catalog；engine-scoped test provider 只服务测试 resolve/generate。

### 5.3 C ABI 错误和所有权

所有导出函数入口先执行：

```cpp
if (result) *result = nullptr;
if (error) *error = nullptr;
```

错误 envelope：

```json
{
  "schema_version": 1,
  "code": "streaming_actual_plan_mismatch",
  "message": "slot_count differs",
  "retryable": false,
  "action": "recreate_engine"
}
```

- status `0` 才允许 `result != null`；
- status `1/2` 必须 `result == null`；
- cancellation status `2` 不当成普通失败重试；
- C string 由 native `strdup`，Swift 只 `tc_string_free` 一次；
- C ABI 不导出 authority、slot pointer、GPU handle 或 raw file descriptor。

## 6. Source lease、snapshot 和 actual receipt

### 6.1 SourceLeaseToken

```cpp
struct SourceLeaseToken {
    PresetSourceIdentity identity;
    std::vector<SourceArtifact> artifacts;
    std::string manifest_digest;
    uint64_t generation = 0;
    bool immutable = false;
};

class SourceLeaseVerifier {
public:
    static void verify_metadata(const SourceLeaseToken &);
    static void verify_before_open(const SourceLeaseToken &, const Descriptor &);
    static void verify_before_execute(const SourceLeaseToken &);
};
```

三次校验点：

1. probe 后：manifest/index/header 与 source identity 一致；
2. snapshot compile 和 first reader open 前：文件 stat/size/mtime 或 content digest 不变；
3. execution revalidation：同一 source lease、runtime revision、device digest 和 catalog record 仍有效。

source identity 只有 snapshot/stat 时不能宣称 content hash；record 必须明确证据范围。

### 6.2 ActualExecutionReceipt

当前 `StreamingRuntimeMetrics` 作为兼容层；public 逐步增加结构化 receipt：

```cpp
struct ActualStageReceipt {
    std::string stage, layout_digest;
    uint32_t prefix = 0, group_size = 0, slots = 0;
    uint32_t distance = 0, workers = 0;
    uint32_t groups = 0, passes = 0;
    std::string pass_transition, multi_pool_policy, retention;
    uint32_t pool_count = 0, slot_bundle_count = 0;
    uint64_t total_fills = 0, total_groups_submitted = 0;
    uint64_t logical_read_bytes = 0, source_generation = 0;
    bool source_lease_verified = false, drained = false;
};

struct ActualExecutionReceipt {
    uint32_t schema_version = 1;
    std::string implementation;
    std::vector<ActualStageReceipt> stages;
    std::string component_policy_revision;
    std::string source_digest, runtime_digest, device_digest;
    std::string receipt_digest;
};
```

Verifier 必须检查：

- stage/layout/component policy 完全相等；
- `groups == layout.groups.size()`、`passes == stage.pass_count`；
- 每个 suffix group 每 pass 恰好一次 fill；
- carry-first-group 只 carry Ready content，不能省略该 pass 的 source fill；
- pool、slot bundle、worker 与 layout 精确相等；
- source lease 已校验、所有 reader 已 drain；
- implementation identity 属于 record 允许集合；
- mismatch 在 result serialization 前失败。

### 6.3 Public result schema

`results.mm` 至少输出 target、calibrated bytes、preset/catalog/record/resolution/source/workload/runtime/device digest、authorized/actual layout digest、component policy、execution container、memory scope 和 `actual_plan_verified`。

只有 `actual_plan_verified=true` 才能标记 success；历史 job 可以展示 revoked record，但不能直接重新提交。

## 7. 通用 StageExecutor 与 multi-slot 调度

### 7.1 生命周期

```text
Vacant
  → Filling(generation, group)
  → Ready(generation, group)
  → Claimed(pass, group)
  → Reading(reader-set sealed)
  → Retiring(last-reader-complete)
  → Vacant(next generation)
```

不变量：ticket generation 单调递增；reader-set 在 encode 前注册；reader 未完成不得 refill；`Vacant` 不表示 backing 已释放；`finish()` drain prefix、所有 pool 和 callback；failed drain 时 executor/adapter 一起 quarantine。

### 7.2 K2 overlap

```text
owner fill g0(slot0) → claim g0 → register readers → encode g0
I/O                         fill g1(slot1)
GPU                         encode g0
owner retire g0 → claim g1 → register → encode g1
I/O                              refill g2(slot0)
GPU                              encode g1
```

目标为 `steady_group_time ≈ max(gpu_encode_time, next_group_fill_time)`。如果 I/O 长期落后，先诊断 source layout、G、page cache、reader completion 和 Q；K 增大前必须重新做 peak calibration。

### 7.3 Multi-pool

- `serial`：class barrier 后销毁旧 pool，再创建新 pool；峰值低但会增加 setup allocation；
- `retain_all`：setup 创建全部 pool，barrier 只切 active pool；steady allocation 可为零，但峰值为所有 pool capacity 之和；
- Flux 9B dual/single 是 retain_all 候选；其他 adapter 默认 serial；
- policy 是 layout identity，runtime 不得自动切换。

### 7.4 Actual counters

每个 public request 至少记录：`pool_creates`、`slot_bundles`、`fills`、`groups_submitted`、`logical_read_bytes`、`wait_seconds`、source lease verification、reader fence completed 和 drain completed。这些计数进入 result/evidence；default route 的 release 热路径不增加 audit 原子计数。

## 8. 四模型 public adapter 施工单

四个 adapter 都遵守同一个模板：

```cpp
probe_public_streaming(input)       // metadata-only
compile_public_streaming(probe, record)
  -> immutable snapshot + source lease + layout
generate_resolved(execution, event, cancel)
  -> existing exact execution core
  -> actual receipt
```

禁止在 `generate_resolved` 中重新根据 target 选择布局；如果 record 与真实 descriptor 不匹配，必须失败。

### 8.1 Z-Image Turbo（第一优先）

范围：original BF16 safetensors、GPU eager、T2I 512²、9 steps、无 LoRA/ANE/GGUF。

| 文件 | 任务 |
|---|---|
| `native/models/z_image/z_image.hpp` | override 三个 public hook；增加 snapshot/source lease 成员的生命周期说明 |
| `z_image.cpp` | 将现有 `ZImageExactStream` 作为唯一 execution core；public route 只传 validated snapshot |
| `streaming_descriptor.hpp`、`z_image_streaming_descriptor.mm` | 输出 30 blocks、token shape、conditioning/VAE policy、source/runtime identity |
| `weight_stream.*` | first-open source lease、logical read/fill counters、reader completion |
| `native/api/c_api.mm` | 仅做 generic coordinator，不新增 Z-Image 特例分支 |
| `results.mm` | actual receipt 和 public selection 输出 |

首个候选族固定 `G1/K2/D0/Q1`，只扫描 P；P 必须保证 suffix 至少容纳两槽。512² record 不得复用于 1024² 或不同 token padding。

验收：Z-PUB-001 metadata no GPU；Z-PUB-002 source stale；Z-PUB-003 layout digest；Z-PUB-004 9 pass fills；Z-PUB-005 VAE/export peak；Z-PUB-006 cancel；Z-PUB-007 output parity；Z-PUB-008 actual receipt。

### 8.2 Flux.2 Klein 9B

范围：BF16 eager、512²、4 steps、GPU；不包含 4B compiled graph、LoRA、ANE。

- `flux.hpp`/`pipeline.cpp`：public hooks 与 `FluxExactStream` 复用；
- `flux_streaming_descriptor.mm`：dual blocks 0–7、single blocks 8–31、class identity；
- `flux_transformer.cpp`：retain_all dual/single pools、class barrier、carry/reader counters；
- `mlx_weight_pager`：source lease、reader completion、logical reads、pool retention evidence；
- result verifier：验证 dual→single→dual 顺序、block 8 concatenate 恰一次。

候选族先冻结 `retain_all + K2/G1/D1/Q2`，再根据完整请求 peak 分别评估 8/10/12/16/20 GiB。已有 private same-layout P1 只能作为框架性能锚点，不是 public qualification。

### 8.3 MiniMax H3 Turbo

范围：只支持 `minimax-h3-turbo` original BF16 native Metal、512×512×22、4 steps、无 audio/LoRA/ANE。普通 H3、VDN、FastH3、MLX、量化变体不在本轮。

- `h3_session.mm`：probe/snapshot/generate_resolved；
- `h3_streaming_descriptor.*`：39 blocks、four-pass workload、source identity；
- `h3_streaming_policy.c`：确保 active block 与 canonical layout 同源；
- H3 result bridge：carry-first-group、每 pass/group fill counter、poison/drain 状态；
- 保留现有 `h3_exact_generation_` 和 cancellation query，不能跨 generation 复用 ticket。

验收：H3-PUB-001 constructor identity；H3-PUB-002 K2/G1 exact；H3-PUB-003 carry fill count；H3-PUB-004 four-pass drain；H3-PUB-005 cancel/fault；H3-PUB-006 output/quality。H3 Turbo 之外不创建 public record。

### 8.4 LTX 2.5

范围：首批 dense T2V、C/Metal、512×320×33、11 steps、24 fps、no audio/I2V/LoRA/Sol/ANE；execution container 为 `ltx_cli_worker`。

- `ltx_session.mm`：将现有 exact owner 拆成 public/private candidate 共享 helper；
- `ltx_streaming_descriptor.*`：transformer/upsampler/VAE/Gemma source closure 和 component policy；
- `ltx_streaming_plan.*`：snapshot 持有 metadata/header/fd 的 request-scoped lease；
- `ltx_streaming_adapter.inc`：只消费 validated plan，不重新选择 P/G/K/D/Q；
- `apps/macos/LTXWorker.swift`：worker 自己执行 resolve→generate，App 只转发 frozen request；
- LTX campaign：包含 worker launch/exit、finalizer、export 和清理。

验收：LTX-PUB-001 worker-local authority；LTX-PUB-002 first-fill cancel；LTX-PUB-003 mid-stage/VAE failure；LTX-PUB-004 destroy retry/quarantine；LTX-PUB-005 component boundary；LTX-PUB-006 full process-tree peak；LTX-PUB-007 actual result。

## 9. App、配置和 JobStore 事务

### 9.1 用户可见模型

```swift
enum StreamingChoice: Codable, Sendable, Equatable {
    case off
    case memoryTier(UInt64)
}
```

UI 只展示 Off 和 native 返回 available 的 8/10/12/16/20 GiB。P/G/K/D/Q、layout digest、worker 数只在 Run Insights 的诊断展开中显示，不能作为设置项。

### 9.2 Options 查询

`NativeEngine.streamingOptions` 返回每个 target 的 status/reason/preset/revision/calibrated bytes/memory scope/release channel。

1. query 带 request generation/digest；
2. 150–300 ms debounce；
3. 新 query 返回后丢弃旧 response；
4. `catalog_has_no_public_records` 显示“暂无可用档位”，不影响 Off；
5. 不把 tentative target 写成 available；
6. physical memory recommendation 只做提示，不偷偷改变选择。

### 9.3 Resolve → persist → generate

```text
用户选 target
  → 生成 NativeRequestV2
  → plan/普通字段校验
  → native resolveStreaming
  → requested selector + exact selector + resolution digest 写入 JobEnvelope v2
  → persist 成功
  → generate 原始 request（native 内部再次 resolve）
```

任何一步失败都不能生成；persist 失败不能把 result 写入 JobStore。JobStore 只保存 JSON-safe identity，不保存 authority。

### 9.4 JobEnvelope v2

```json
{
  "schema_version": 2,
  "request": { "…": "frozen request" },
  "streaming": {
    "choice": "memory_tier",
    "target_request_memory_bytes": 10737418240,
    "requested_selector": { "…" : "…" },
    "exact_selector": { "…" : "…" },
    "resolution_digest": "…",
    "catalog_revision": "…"
  },
  "state": "queued"
}
```

重放时重新验证 source/runtime/device/catalog；record 被 revoke 或 source 变化时失败，不能静默换档。

### 9.5 LTX worker-local

- App 不能用 embedded engine 的 authority 代替 worker authority；
- worker 收到 frozen request 后自己创建 engine、probe、resolve、generate；
- worker result 的 execution container 必须是 `ltx_cli_worker`；
- worker 退出前完成 result 写入、event/source/reader drain；
- App 只消费 JSONL event 和 result，不能接触 slot/pool。

## 10. Catalog record 和配置文件

### 10.1 Record 最小字段

record 至少包含 `id/catalog_revision/revision`、source/workload/runtime/device、plan、calibration、performance 和 release。plan 必须包含 canonical config、layout digest、component policy、pass transition、multi-pool；calibration 必须包含完整请求 bytes、scope、evidence digest、sample count。

真实 public record 的 `calibrated_request_bytes` 禁止为零；未经校准的 record 只能留在 test/experimental catalog。

### 10.2 配置文件分层

```text
profiles/streaming/
  workload_cards/        # 模型/尺寸/steps/input 闭包
  candidate_search/      # 离线候选范围
  reviewed/              # signed/reviewed exact records
  examples/              # 非可执行示例
```

App 不读取 candidate search；production build 只打包 reviewed record；当前仍打包空 catalog。

## 11. 校准与工具链实施方案

### 11.1 工具分层

| 工具 | 输入 | 输出 | 是否可授予 public |
|---|---|---|---|
| `descriptor_inspector` | model path | descriptor/source/workload summary | 否 |
| `layout_compiler` | descriptor + candidate config | canonical layout/digest | 否 |
| `schedule_simulator` | layout + timing model | overlap/queue/peak estimate | 否 |
| `run_streaming_audit.py` | audit dylib + request | hook/allocation/thread counters | 否 |
| `run_streaming_campaign.py` | frozen policy + dylib + model | raw samples/evidence | 否 |
| `verify_streaming_campaign.py` | policy + raw samples | independent P0/P1 verdict | 否 |
| `process_tree_sampler` | pid/worker | RSS/compressed/swap/pressure timeline | 否 |
| `catalog_builder` | verified evidence + review | reviewed record/catalog diff | 仅生成候选 |
| `catalog_linter` | catalog JSON | schema/identity/release validation | 否 |

只有 review owner 将 catalog diff 合并到 production source 后，record 才具备 public 资格。

### 11.2 Candidate 搜索顺序

```text
1. 编译 descriptor，确认 block/pool/class/source ranges 完整
2. 静态估算剪掉明显超 target 的候选
3. 优先扫描较小 G、K2、Q1/Q2 和 model-approved prefix
4. simulator 检查 slot capacity、D<K、pool peak
5. 真实短 smoke 排除读错、stall、poison、output mismatch
6. 完整 process-tree campaign
7. resident/streaming/swap 对照
8. independent verifier + model/performance/release review
```

不得先选一个“看起来能跑”的布局，再把结果倒填为 target。

### 11.3 完整请求采样

每个 sample 至少保存 request/resolved/result/event、process-tree、pressure、swap、stdout/stderr、source/runtime/device identity。采样窗口覆盖 setup、denoise、pass/class transition、VAE/upsampler/audio/export、drain/worker exit。

同时记录 MLX active/peak、进程 RSS、compressed memory、swap in/out、pressure warning、logical/physical reads、wall/denoise、失败和取消。

### 11.4 Simulator 不能替代真实证据

simulator 只能回答 slot capacity、明显 starvation、class barrier pool peak 和 overlap 上界，不能回答 OS page cache、Metal driver peak、完整进程树、swap pressure、quality 或 source drift。

## 12. Streaming 与系统 swap 的实验设计

### 12.1 四路矩阵

| 路线 | 说明 |
|---|---|
| R | resident/default，内存充足基线 |
| S | explicit streaming exact layout |
| B | bounded-memory admission（若独立实现） |
| W | 内存压力下系统 swap 对照 |

W 必须证明产生 swap in/out 或 pressure evidence，不得把普通 page-cache miss 称为 swap。

### 12.2 采集指标

`wall median/p95`、`denoise median/p95`、first-output latency、MLX active/peak、process-tree RSS/peak、swap in/out、compressed bytes、logical/physical reads、stall/wait、success/cancel/failure、output hash/quality distance。

### 12.3 可证伪结论

只有同时满足以下条件，才能写“streaming 比 swap 更快/更稳定”：

1. W 确实有 swap/pressure evidence；
2. S/W output 和 quality 等价；
3. S 在预注册的 wall/P95/failure/swap 门槛上更好；
4. 至少 20 个匹配样本，ABBA/blocked bootstrap 通过；
5. 目标 physical memory class 可复现。

否则只能写“显式 streaming 提供可控峰值，swap 性能关系待验证”。

## 13. 默认路径保护和性能验收

### 13.1 结构性审计

audit build 导出以下计数：

```text
new_memory_probes
new_public_catalog_queries
new_source_hashes
new_pool_allocations
new_worker_threads
new_cache_clear_or_unload_calls
steady_framework_allocations
steady_framework_thread_creates
```

默认请求全部为 0；public exact 请求至少有 framework-active 证据，但不能以 audit counter 代替性能通过。

### 13.2 P0/P1/P2/P3/P4

- P0：default resident vs candidate/public route，wall median ≤ 1.02，P95 ≤ 1.05；
- P1：同一 exact layout 的 direct/generic executor 对照；
- P2：quality/output parity；
- P3：memory/swap pressure；
- P4：生命周期、取消、故障和重复请求。

统计必须保存样本排除原因、随机种子、bootstrap unit、multiplicity policy 和 evidence digest。没有分母的“平均快了 1%”不能进入 release note。

### 13.3 内存门

```text
max(process_tree_peak_samples) + H(target) <= target
```

发生 swap、OOM、driver allocation failure、worker crash 或 telemetry gap 时，不能静默删除 sample 后继续声称通过；必须按 exclusion policy 记录并由 reviewer 决定整组是否 invalid。

## 14. 验收矩阵

### 14.1 Host/compiler/resolver

| ID | 验收 |
|---|---|
| PUB-HOST-001 | canonical source/workload/runtime/device digest 稳定 |
| PUB-HOST-002 | layout、multi-class/multi-pool、K/D/Q 回归 |
| PUB-HOST-003 | selector conflict/route/model mismatch 错误优先级 |
| PUB-HOST-004 | empty catalog fail-closed；不调用 model probe/GPU |
| PUB-HOST-005 | test catalog injection 只在 test build 生效 |
| PUB-HOST-006 | authority non-copy/non-serializable |
| PUB-HOST-007 | actual receipt mismatch（slot/source/drain/layout）硬失败 |

### 14.2 C ABI/Swift

| ID | 验收 |
|---|---|
| PUB-ABI-001 | null engine/request/result/error pointer |
| PUB-ABI-002 | busy try-lock 返回稳定 code |
| PUB-ABI-003 | error/result ownership 各 free 一次 |
| PUB-ABI-004 | cancel status 2 与普通 error 分离 |
| PUB-ABI-005 | resolve 不取得 global GPU lock |
| PUB-ABI-006 | generate 内部重新 resolve |
| PUB-SWIFT-001 | Codable round-trip 与 native JSON semantic parity |
| PUB-SWIFT-002 | stale options response 不覆盖新 query |

### 14.3 Runtime/lifecycle

| ID | 验收 |
|---|---|
| PUB-RT-001 | first-fill/claim/refill/retire 顺序正确 |
| PUB-RT-002 | reader 未完成不得 refill |
| PUB-RT-003 | cancel before first fill |
| PUB-RT-004 | cancel mid-pass |
| PUB-RT-005 | short read/checksum/source drift |
| PUB-RT-006 | drain timeout 进入 quarantine，不 free live backing |
| PUB-RT-007 | retained multi-pool 无 steady allocation |
| PUB-RT-008 | generation 不跨请求复用 |

### 14.4 App/worker

| ID | 验收 |
|---|---|
| PUB-APP-001 | 默认 Off 且旧任务 JSON 兼容 |
| PUB-APP-002 | 只显示 native available targets |
| PUB-APP-003 | resolve→persist→generate 原子事务 |
| PUB-APP-004 | persist 失败不 generate |
| PUB-APP-005 | catalog empty 显示 unavailable，不影响 Off |
| PUB-APP-006 | revoked/stale 不静默换档 |
| PUB-APP-007 | LTX authority 在 worker 内铸造 |
| PUB-APP-008 | result mismatch 不展示成功输出 |
| PUB-APP-009 | RunInsights 展示 target/preset/actual layout/memory scope |

### 14.5 模型与发布

| ID | 验收 |
|---|---|
| PUB-MODEL-001 | Z-Image public hook + 512² smoke |
| PUB-MODEL-002 | Flux 9B dual/single retained pools |
| PUB-MODEL-003 | H3 Turbo carry-first-group |
| PUB-MODEL-004 | LTX worker-local exact route |
| PUB-CAL-001 | 每个 target 至少一个完整 workload card |
| PUB-CAL-002 | process-tree peak + memory margin |
| PUB-CAL-003 | resident/streaming/swap 四路对照 |
| PUB-REL-001 | evidence/catalog/review digest 一致 |
| PUB-REL-002 | staged→production→revoke→replay 演练 |

### 14.6 本机回归命令和判定

提交 R0 或任何 runtime/result 变更后，至少执行以下顺序：

```sh
env TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
make test-streaming-host
make test-streaming-contract
tools/native/build_app.sh
git diff --check
```

独立的 host resolver/result 测试可单独复现：

```sh
python3 -B tests/native/test_streaming_preset_resolver.py
python3 -B tests/native/test_z_image_candidate_streaming_gate.py
```

判定规则：

1. 任一命令非零退出，R0 不得标记完成；
2. deployment-target warning 可以记录，但不能隐藏编译 error；
3. host/contract 通过只说明框架接缝和 fail-closed，不说明真实模型 public eligibility；
4. `tools/native/build_app.sh` 通过只说明 Swift/App 可编译，不说明 App 已经展示 available target；
5. 真实模型 campaign 必须额外上传 raw sample、process-tree、swap/pressure 和 independent verifier 输出；
6. release note 必须引用 commit、catalog revision、record digest 和 evidence digest，不得只引用“测试通过”。

CI 建议拆成四个 job，避免把默认回归和 public candidate 混在同一个绿色状态中：

```text
streaming-host      compiler/layout/executor/resolver/result fake tests
streaming-contract  C ABI/Swift semantic/fail-closed tests
default-regression  resident GPU/GPU+ANE/compiled/legacy route
public-evidence     only on reviewed test catalog or physical model runner
```

其中 `public-evidence` 默认不在普通 PR 上运行，也不能向 production catalog 写入记录；只有 release campaign 明确指定 reviewed test catalog 时才允许执行。

## 15. 分阶段实现与提交边界

### R0 · Actual result 收口（已完成）

文件：`public_result.*`、`session.hpp`、`results.mm`、`c_api.mm`、build/test wiring。

结果：`c6cba54` 已提交 actual plan verifier success/failure、public result serialization 和完整 native host/contract/App build 回归；production catalog 仍为空、无 public record。

### R1 · Pure coordinator/provider（主体已完成，仍需收口）

文件：`public_runtime.*`、`catalog_provider.*`、C ABI 下沉。`fa1ecd0` 已完成 pure coordinator、provider、fake session resolve/revalidate 和 production provider；仍需消除重复 preflight、补完整 C ABI ownership/busy/cancel/result serialization，并决定 immutable catalog snapshot/test injection 的最终形态。

### R2 · Source lease + receipt v2

文件：`source_lease.*`、`value_probe.*`、`actual_receipt.*`、四模型 metrics/receipt bridge。完成标准：source changed、runtime/device changed、per-pass/group actual mismatch、drain timeout 全部 fail-closed；result 无半成功状态。

### R3 · Z-Image public adapter

先完成最小 512²/9 steps GPU workload card，只在 test catalog 运行；通过模型与生命周期验收后再进入 calibration。

### R4 · Flux 9B / H3 Turbo

分别接 multi-pool 和 carry-first-group；每个模型独立 candidate/evidence，不共享未经验证的 record。

### R5 · LTX worker-local + App

完成 JobStore v2、options picker、resolve transaction、worker-local authority、RunInsights。

### R6 · Toolchain/calibration

实现 inspector/compiler/simulator/process sampler/campaign/verifier/catalog builder，先用 test model/fake catalog 做端到端演练。

### R7 · 单 record staged/release

按模型→target→workload card 单调推进；每次只加入一个 reviewed record；完成灰度、revoke、replay 后再加入下一个。

### R8 · 合并最新 dev

当前 `dev` 为 `02148b7`，尚未在本轮重新合并。runtime/App/model 稳定后执行：

```sh
git fetch --all --prune
git merge dev
```

保留 dev 的 default GPU、ANE、M5 优化；streaming Off 不查 catalog；public v1 未认证 GPU+ANE 继续拒绝。冲突解决后重跑 default GPU、GPU+ANE、Core ML cache、legacy streaming、public fail-closed、actual receipt 全套回归。

## 16. 失败注入和回滚设计

### 16.1 必须可注入的故障

```text
catalog revision mismatch
record revoked
source mtime/size/digest changed
reader short read
reader completion lost
slot double claim
slot refill before retire
GPU command failure
worker SIGTERM/SIGKILL
VAE/export failure
drain timeout
```

故障注入只允许 test/audit build；release binary 不导出测试 hook。

### 16.2 Record 撤回

1. 将 record `revoked=true`，catalog revision +1；
2. options 立即返回 unavailable；
3. 新 request 不得 select revoked record；
4. 已运行 request 由 authority/source lease 决定继续或失败，不强制换布局；
5. 历史 JobEnvelope 保留原 record/resolution/result identity；
6. 记录 review digest、撤回原因和 rollback commit。

### 16.3 Engine quarantine

```json
{
  "code": "streaming_quarantined",
  "retryable": false,
  "action": "recreate_engine"
}
```

App/worker 必须释放旧 engine 并创建新实例，不能通过 `unload` 强行复用不安全 session。

## 17. Definition of Done

### 框架级

- [ ] Off/default 结构性 fast path 和 audit zero 通过；
- [ ] request-only validator 无 catalog/GPU/filesystem 副作用；
- [ ] resolver deterministic，authority 不可序列化；
- [ ] source lease 三次校验和 execution revalidation 完成；
- [ ] actual receipt mismatch 在 result 前硬失败；
- [ ] C ABI null/busy/ownership/cancel 全绿；
- [ ] production catalog empty 时所有 public route 明确 unavailable。

### 模型级

- [ ] probe/compile/generate_resolved 三个 hook 均 override；
- [ ] private candidate/public route 共用同一 exact execution core；
- [ ] P/G/K/D/Q、pool/pass/component identity 从 descriptor 到 result 全链路一致；
- [ ] cancel、short read、source drift、worker exit、drain timeout 全部可验证；
- [ ] output hash/quality 和 full request peak evidence 完整。

### App 级

- [ ] 默认 Off；只显示 target，不显示内部 layout 参数；
- [ ] options debounce/stale guard；
- [ ] resolve→persist→generate 事务原子；
- [ ] LTX worker-local resolve；
- [ ] result/replay/revocation 状态可理解且不静默 fallback。

### 单 record 级

- [ ] workload/source/runtime/device/container exact identity；
- [ ] 完整 process-tree calibration、margin 和 independent verifier；
- [ ] P0/P1/P2/P3/P4 证据齐全；
- [ ] 至少 20 个匹配性能样本和失败/取消证据；
- [ ] model/performance/release owner review 签字；
- [ ] staged、production、revoke、replay 演练通过。

## 18. 当前下一步执行清单（按优先级）

1. 消除 `c_api.mm` → `resolve_normalized()` 的重复 preflight，冻结单次 catalog snapshot；
2. 补 C ABI ownership/busy/cancel 和 public result serialization 测试；
3. 按 36 实现 ValueModelStreamingProbe/Snapshot、fd-based source lease 和 receipt v2；
4. 先接 Z-Image public hooks，使用 test-only reviewed catalog 做完整 replay；
5. 按相同模板接 Flux 9B、H3 Turbo、LTX worker；
6. 按 37 完成 process-tree sampler、simulator、campaign/verifier 和 catalog builder；
7. 对每个模型按 workload card 扫描 8/10/12/16/20 GiB 候选，先记录 unavailable 也可以，不得伪造 available；
8. 做 resident/streaming/bounded/swap pressure 四路对照，结果不足时只报告“待验证”；
9. 完成 App StreamingChoice、engine-scoped options、JobStore v2 和 LTX worker-local authority；
10. runtime/App/model 稳定后合并 `dev@02148b7`，再做完整回归；
11. 通过 reviewed record release gate 后，才在 App 中显示第一个 available target。

本文完成后，团队可以用同一套接口和验收 ID 扩展新模型；但在 catalog、adapter、calibration 和 review 全部完成前，TurboCider 仍应把 public streaming 视为 **框架施工中、默认功能不受影响**。
