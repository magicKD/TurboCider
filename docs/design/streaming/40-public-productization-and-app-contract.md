# 40 · Public Streaming 产品化、App 合同与端到端事务

[目录](README.md) · [代码合同](38-framework-code-contracts-and-implementation-workbench.md) · [验证工作簿](39-validation-benchmark-and-release-workbook.md)

修订日期：2026-09-17。状态：**产品化与代码实施规格；不代表任一模型或档位已经 public**。

本文回答一个产品层问题：用户只选择“目标内存档位”，不直接看到 `prefix/group/slot/prefetch/worker/pool` 等内部布局；TurboCider 在后台根据模型、工作负载、设备和 reviewed catalog 选择一个精确 layout。该设计必须满足：

1. 不开启 add-on 时，默认 resident/compiled/GPU+ANE 路径不进入 streaming 控制面；
2. 开启后，用户输入仍是稳定、少量、可解释的目标档位；
3. native 端执行的是 catalog 中审核过的 exact record，而不是运行时猜测或偷偷降级；
4. 任何 source、runtime、device、workload 或 catalog 变化都 fail-closed；
5. App 能区分“暂不可用”“没有匹配档位”“正在校准”“执行失败”，不能把 private candidate 显示成可用。

## 1. 产品边界

### 1.1 用户看到的概念

高级设置增加一个独立的 Streaming 选择器：

```text
Streaming：关闭 / 8 GiB / 10 GiB / 12 GiB / 16 GiB / 20 GiB
```

显示值是 `target_request_memory_bytes` 的人类化别名。它不是：

- GPU 显存硬上限；
- 当前瞬时空闲内存；
- 系统 swap 大小；
- 单个 transformer 权重池大小；
- 对所有模型都保证成功的承诺。

catalog 记录的 `calibrated_request_bytes` 是完整请求 process-tree 的已测峰值，判定时还要加 `H(T)` safety margin。用户选 10 GiB，实际执行资格要求的是：

```text
tree_peak + H(10 GiB) <= 10 GiB
```

其中 `tree_peak` 覆盖 parent、worker、encoder、denoiser、VAE/upsampler/audio、export 和 drain；不能只看 slot backing。

### 1.2 用户看不到的概念

下列字段留在 native catalog/record 中，不进入 App 的普通编辑表单：

```text
resident_prefix_blocks
block_group_size
slot_count
prefetch_distance
io_workers
pass_transition
multi_pool_policy
component_policy_revision
reader_revision
source_snapshot_digest
layout_digest
```

高级诊断页可以只读地显示摘要，例如“已选择 10 GiB 档位 · 预计峰值 8.7 GiB · 12 个分组”，但不能允许用户直接编辑内部布局并绕过 authority。

### 1.3 明确不做的事

- 不根据当前空闲内存每次动态改 `K/G/P/D/Q`；
- 不把 swap 当作 streaming 的实现细节；swap 只在性能实验中作为对照臂；
- 不因某个档位不 fit 就自动换到更小 slot、减少 block 或改变 dtype；
- 不将旧 `residency=streamed`、旧 `memory_budget_bytes` 和新 public selector 静默合并；
- 不在 production 目录放入未经完整请求校准的 record；
- 不让 App 侧自带 catalog、路径、fd、authority 或 GPU 指针。

## 2. 选择器、选项和 exact record

### 2.1 请求模型

Swift 与 C++ 共用 schema v2；推荐的最小请求如下：

```json
{
  "schema_version": 2,
  "model": "z-image-turbo",
  "operation": "image.generate",
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

`StreamingSelector` 只表达意图；`preset_id`、`preset_revision`、`catalog_revision` 和 `expected_resolution_digest` 只在 native 返回 exact selector 后回填，不能由 App 伪造。

### 2.2 `Off` 的规范语义

以下情况都表示关闭新 public add-on：

```text
streaming selector absent
selector.enabled == false
legacy request without selector
```

关闭时必须满足：

```text
catalog_provider.snapshot()        == 0 calls
probe_public_streaming()            == 0 calls
SourceLease open/fstat/hash          == 0
receipt allocation                   == 0
streaming worker/pool                 == 0
```

旧路径仍可按自己的 `residency`、profile、ANE 和 memory policy 工作；“新 selector 关闭”不能破坏旧 `residency=streamed` 语义。

### 2.3 options 查询

options 是只读查询，不创建 GPU backing：

```c
int tc_streaming_options_json(
    const char *request_json,
    char **result_json,
    char **error);
```

返回形态：

```json
{
  "schema_version": 1,
  "query_status": "available|unavailable|tentative",
  "catalog_revision": "tc-streaming-catalog-...",
  "execution_container": "embedded_app",
  "device": {
    "gpu": "Apple M4 Max",
    "physical_memory_bytes": 68719476736,
    "device_class": "..."
  },
  "targets": [
    {
      "target_request_memory_bytes": 8589934592,
      "status": "available|not_calibrated|no_fit|unsupported",
      "reason_code": null,
      "preset_id": "...",
      "preset_revision": 1,
      "calibrated_request_bytes": 7516192768,
      "memory_scope": "execution_process_tree_v1",
      "release_channel": "public-stable"
    }
  ]
}
```

`tentative` 只适用于 test-only catalog 或开发构建，不能出现在 release App。production catalog 为空时，所有 target 返回 `not_calibrated` 或 `catalog_has_no_public_records`，不伪装成 available。

### 2.4 exact resolve

```c
int tc_engine_resolve_streaming_json(
    tc_engine *, const char *request_json,
    char **result_json,
    char **error);
```

resolve 的输出包含：

```text
requested_selector       # 用户意图
exact_selector           # native 选出的 preset + revision + digest
selection                # record/layout/calibrated bytes
identity                 # source/runtime/device digest
request_digest
resolution_digest
catalog_revision
```

resolve-only 阶段不得：

- 创建 MLX array、Metal buffer、compiled graph；
- 启动 refill worker；
- 修改 request shape、dtype、steps 或 prompt；
- 占用 global GPU lock；
- 把未审核 candidate 写入 production catalog。

## 3. 物理内存与档位匹配策略

### 3.1 只使用物理内存，不使用瞬时空闲值

设备发现使用 `StreamingDeviceIdentity.physical_memory_bytes`。一次 resolve 中设备 identity 必须冻结；运行期间若设备 identity 改变，应返回 `streaming_authority_mismatch`，不能切换到另一档位。

推荐的显示策略：

| 物理内存 | UI 默认推荐 | 可见档位 | 说明 |
|---:|---:|---|---|
| 8–9 GiB | 8 GiB | 8 | 只显示有 reviewed record 的模型/工作负载 |
| 10–11 GiB | 10 GiB | 8/10 | 10 GiB 仍需完整请求 margin |
| 12–15 GiB | 12 GiB | 8/10/12 | 不根据空闲值临时升级 |
| 16–19 GiB | 16 GiB | 8/10/12/16 | 由 catalog device range 过滤 |
| ≥20 GiB | 20 GiB | 8/10/12/16/20 | 20 GiB 仍不是“任意请求” |

这张表只是 UI 推荐，不是资格表。真正可用性由 `minimum_physical_memory_bytes`、`maximum_physical_memory_bytes`、workload/source/runtime exact match 和 `fits(record, target)` 共同决定。

### 3.2 选择排序

同一 target 有多个 reviewed record 时，排序必须稳定：

```text
performance.rank
calibrated_request_bytes（小者优先）
performance.logical_read_bytes（小者优先）
preset_id
preset_revision
```

不得按当前空闲内存、历史最近使用时间或随机值排序。若用户显式提供 `preset_id + revision`，只允许 exact replay；找不到则返回 stale，不自动换另一个 record。

### 3.3 目标 margin

当前合同：

```cpp
H(T) = max(512 MiB, ceil(0.10 * T));
```

当 `calibrated_request_bytes + H(T) > T` 时 record 不 fit。未来若改变 margin 算法，必须提升 `estimator_revision`、catalog revision 和所有 evidence digest，不得在原 record 上原地修改。

## 4. App 端实现合同

### 4.1 Swift 类型

建议保留现有 `NativeStreamingSelectorV2`，在 App 层增加不暴露内部布局的类型：

```swift
enum StreamingChoice: Codable, Equatable, Sendable {
    case off
    case targetGiB(Int)

    var targetBytes: UInt64? {
        if case let .targetGiB(gib) = self {
            return UInt64(gib) << 30
        }
        return nil
    }
}

struct StreamingAvailability: Codable, Sendable {
    let targetGiB: Int
    let status: String
    let reasonCode: String?
    let presetID: String?
    let calibratedBytes: UInt64?
}
```

`StudioDraft` 只保存 `streamingChoice`；旧 `zImageStreamingBudgetGiB` 在迁移期可读但不再作为 public selector 的唯一来源。保存草稿时不要持久化 `layout_digest` 作为用户配置。

### 4.2 UI 状态机

```text
model/workload changed
    -> invalidate options + resolution
    -> query options (resolve-only)
    -> show Off + available targets + disabled reason
user selects target
    -> build request with selector
    -> resolve
    -> persist exact JobEnvelope
    -> generate
```

以下状态必须可见且可解释：

| 状态 | UI 文案建议 | 是否允许 Generate |
|---|---|---:|
| `available` | “10 GiB · 已验证” | 是 |
| `not_calibrated` | “10 GiB · 尚未验证” | 否 |
| `no_fit` | “10 GiB · 当前请求放不下” | 否 |
| `unsupported` | “10 GiB · 此模型不支持” | 否 |
| `stale` | “配置已过期，请重新解析” | 否 |
| `tentative` | “开发测试档位” | 仅 test build |

禁用项不能通过修改 JSON、重试 generate 或切换 UI 页面绕过；native resolver 仍是最终 authority。

### 4.3 请求事务

App 生成必须是两阶段事务：

```text
draft -> request digest
     -> native resolve
     -> exact envelope persisted atomically
     -> native generate using same request/exact selector
     -> public result verified
     -> JobStore terminal state
```

失败规则：

- resolve 失败：不写 `resolved` Job，不创建 authority；
- persist 失败：不调用 generate；
- generate 前 stale/revoke：Job 标为 failed/stale，不产生 output；
- generate 中 cancel：等待 drain；若 drain 未知，session quarantine；
- result mismatch：不写成功 output，不把作业标为 succeeded。

### 4.4 JobEnvelope v2

```json
{
  "schema_version": 2,
  "job_id": "uuid",
  "request_nonce": "random-128-bit",
  "model": "ltx-2.5-distilled",
  "original_selector": {
    "selection": "memory_tier",
    "target_request_memory_bytes": 17179869184
  },
  "exact_selector": {
    "selection": "preset",
    "preset_id": "ltx-...",
    "preset_revision": 1,
    "catalog_revision": "...",
    "expected_resolution_digest": "..."
  },
  "resolution_digest": "...",
  "record_digest": "...",
  "layout_digest": "...",
  "source_digest": "...",
  "runtime_digest": "...",
  "device_digest": "...",
  "state": "resolved"
}
```

禁止写入：fd、inode、canonical model path、GPU pointer、worker PID、authority 对象、临时 source path。重放时 native 重新 probe、revalidate 和 resolve。

## 5. Native 代码修改清单

### 5.1 当前代码可复用部分

```text
native/core/streaming_contracts.hpp       StreamingSelector / 五档常量
native/runtime/streaming/config.cpp       selector validation
native/runtime/streaming/preset_catalog.* record validation / fit
native/runtime/streaming/preset_resolver.* stable selection / authority
native/runtime/streaming/public_runtime.* preflight / resolve / revalidate
native/runtime/streaming/public_result.* actual-plan verification
native/api/c_api.mm                       engine/global lock + JSON boundary
bindings/swift/TurboCiderNative.swift    Codable + async resolve
```

### 5.2 必须补齐的接缝

1. `PublicStreamingCoordinator::preflight()` 的 ticket 必须覆盖整个 resolve；禁止 resolve 内再次取得不同 catalog snapshot；
2. `ModelStreamingProbe::source_lease()` 与 `ModelStreamingSnapshot::source_lease()` 必须对 public adapter 非空；
3. `PublicPresetResolver::authorize()` 必须校验 probe/snapshot 同一 lease、同一 digest、同一非零 generation；
4. `PublicStreamingCoordinator::revalidate()` 必须在 pre-GPU 前调用 `revalidate_paths()` + `revalidate_open_files()`；
5. adapter drain 后必须调用 `revalidate_after_drain()`，再把 `source_lease_verified=true` 写入 runtime metrics；
6. `PublicStreamingSelectionMetrics` 增加 receipt digest、source generation 和 verifier revision；
7. Swift `NativeJobStore` 增加 options/resolution 缓存失效和 JobEnvelope 原子提交；
8. LTX worker 必须在 worker 内完成 resolve/generate，parent 仅传递可序列化 envelope；
9. release C header 只增加 additive options/resolve API，不暴露 catalog provider 注入。

### 5.3 伪代码：一次生成

```cpp
RunResult generate_public(tc_engine &engine, Request request) {
    require(request.streaming_selector->active(), "selector_required");
    auto preflight = coordinator.preflight(request);              // no GPU
    auto plan = make_plan_after_public_streaming_preflight(request);
    auto execution = coordinator.resolve_normalized(
        std::move(plan.request), device_identity(), std::move(preflight));

    lock_global_gpu();
    coordinator.revalidate(*execution, device_identity());        // source + catalog
    auto result = session.generate_resolved(execution, event, cancel);
    verify_and_attach_public_streaming_result(*execution, result); // receipt + drain
    return result;
}
```

`generate_public` 不能调用普通 `session.generate(request)` 作为 fallback。若 adapter 不支持 exact public route，必须返回稳定的 `streaming_public_adapter_unsupported`。

## 6. Public、private、legacy 三条路由

| 路由 | 入口 | authority | 是否可出现在 App public |
|---|---|---|---:|
| legacy/default | `generate(Request)` | 无新 authority | 是，保持现状 |
| private candidate | `tc_engine_create_model_candidate` + exact layout | test/internal flag | 否 |
| public exact | selector → catalog → probe/snapshot → resolved generate | immutable authority + lease + receipt | 仅 reviewed record |

`allow_experimental_streaming` 只能启用 private candidate，不能被请求 JSON、环境变量或 Swift 选项映射为 public release channel。

## 7. 性能保护

### 7.1 默认路径编译期/运行时闸门

```cpp
if (!request.streaming_selector || !request.streaming_selector->active()) {
    return session.generate(request, event, cancel); // 原有路径
}
```

不要在默认路径上：

- 查询 catalog；
- 计算 source digest；
- 初始化 receipt recorder；
- 创建 `PublicStreamingCoordinator`；
- 读取模型 header；
- 改变 allocator/cache policy。

审计计数器必须验证以上调用为零。只有显式 selector active 时才允许新增工作。

### 7.2 Public exact 的性能门

同布局 direct 与 generic executor 的 P1 门：median wall ratio ≤ 1.02、P95 ≤ 1.05、denoise median ≤ 1.02、输出/质量一致。默认 P0 仍要与 dev 基线 ABBA 对照；P1 不能替代 P0。

### 7.3 为什么不在 UI 显示布局

布局是实现 identity 的一部分。让用户编辑布局会导致：

- catalog record 无法与请求一一对应；
- 内存校准结果失效；
- source/reader/kernel 组合可能不受支持；
- 任意用户输入可触碰 slot safety 和 worker 生命周期。

因此 UI 只选择目标档位，后台保留可审计的 exact selector 和 record digest。

## 8. 发布前产品验收

### App/API

- [ ] Off/default 路径的 provider/probe/lease/receipt/worker 计数为零；
- [ ] 五档显示值固定，非五档请求返回 `unsupported_memory_target`；
- [ ] model/workload/shape/LoRA/ANE 改变会使 options/resolution 失效；
- [ ] disabled target 不能通过 JSON 重放绕过；
- [ ] resolve 失败不产生 resolved Job；
- [ ] result mismatch 不产生成功 output；
- [ ] JobStore v2 不保存 fd/path/GPU 指针；
- [ ] LTX parent/worker 两阶段事务可恢复或 quarantine；
- [ ] revoke 后旧 Job 只能 stale/重解析；
- [ ] UI 文案不出现“保证低于显存”“一定比 swap 快”等未经证据的承诺。

### Release gate

必须同时具备：

```text
reviewed catalog record
source lease + receipt v2
20 matched model runs
完整 process-tree peak + H(T)
P0/P1 non-regression
fault/cancel/quarantine
independent verifier
App transaction test
```

其中任一项缺失，状态只能是 `candidate`、`unavailable` 或 `INCONCLUSIVE`，不能把选项标为 available。
