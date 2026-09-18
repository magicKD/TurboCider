# 52 · Public Streaming App、配置与模型档位规格

修订日期：2026-09-18。状态：**产品化和校准实施规格；当前 production catalog 为空**。

本文定义用户如何只选择 `Off / 8 / 10 / 12 / 16 / 20 GiB`，以及 TurboCider 如何为不同模型、shape、组件和物理内存选择已认证的内部 streaming plan。用户不应看到 `P/G/K/D/Q`，也不应承担 layout 调参责任。

本文依赖：

- [23 public memory tier](23-public-memory-tier-presets.md)：产品目标；
- [40 App contract](40-public-productization-and-app-contract.md)：App/JobStore 边界；
- [50 calibration/release](50-public-streaming-calibration-and-release-evidence.md)：实机校准和证据；
- [51 runtime closure](51-public-streaming-remaining-runtime-closure.md)：H3/LTX/common runtime 闭环。

## 1. 产品决策

### 1.1 用户层只有一个 streaming selector

App 高级设置显示：

```text
内存模式：
  Off（默认）
  8 GiB
  10 GiB
  12 GiB
  16 GiB
  20 GiB
```

语义：

- `Off`：保持当前模型默认/常驻路径；不构造 selector；
- 数值档位：请求一个完整生成过程的目标内存档位，由 native catalog 选择已认证 plan；
- 档位不是物理显存容量，不是 OS hard cap，也不是任意 workload 的保证；
- target 无 record 时显示 `当前配置不可用`，不能自动换档或回退 resident；
- 只有用户显式改回 `Off`，才允许走默认路径。

### 1.2 为什么不能显示 P/G/K/D/Q

`P/G/K/D/Q` 是实现细节，并且不同模型含义不同：

```text
P = resident prefix blocks
G = block group size
K = slot count
D = prefetch distance
Q = I/O workers / queue depth（历史资料命名需由 schema 固定）
```

直接显示会导致：

- 用户误以为同一 K 在 H3、Flux、LTX 上内存和性能等价；
- 用户组合出未校准的 plan；
- catalog/evidence identity 被 App 参数覆盖；
- 出错时 native 无法判断是产品选择还是任意 manual layout。

保留 manual exact layout 作为开发/测试 API，但 release App 不暴露，也不能由用户 JSON 绕过 catalog。

## 2. Swift 数据模型和兼容迁移

### 2.1 建议的 App 状态

在 `apps/macos/StudioState.swift` 中把旧字段收敛为：

```swift
enum StreamingSelection: String, Codable, Sendable, CaseIterable {
    case off
    case tier8
    case tier10
    case tier12
    case tier16
    case tier20

    var targetBytes: UInt64? {
        switch self {
        case .off: return nil
        case .tier8: return 8 << 30
        case .tier10: return 10 << 30
        case .tier12: return 12 << 30
        case .tier16: return 16 << 30
        case .tier20: return 20 << 30
        }
    }
}

struct StreamingUIState: Codable, Sendable {
    var selection: StreamingSelection = .off
    var optionsCatalogRevision: String?
    var optionsRequestDigest: String?
    var lastStatus: String = "unknown"
}
```

`StudioDraft` 增加：

```swift
var streaming = StreamingUIState()
```

旧字段迁移：

| 旧值 | 新值 |
|---|---|
| `residency=resident` | `streaming.selection=.off` |
| `residency=streamed` + 6 GiB | `.tier8`，并提示旧 6 GiB 不再是合法档位，需要重新解析 |
| `residency=streamed` + 8/10/12 GiB | 对应 tier |
| `residency=component_staged` | `.off`；LTX worker policy 单独迁移 |
| GGUF/旧 manual profile | `.off`；保留 profile 文件但不自动启用 public streaming |

迁移不能静默把旧 `zImageStreamingBudgetGiB=6` 当作已支持 6 GiB。

### 2.2 发送的 V2 request

`bindings/swift/TurboCiderNative.swift` 保留 `NativeStreamingSelectorV2`，release App 只构造：

```swift
func streamingSelector(_ state: StreamingUIState) -> NativeStreamingSelectorV2? {
    guard let bytes = state.selection.targetBytes else { return nil }
    return NativeStreamingSelectorV2(targetBytes: bytes)
}
```

`NativeRequestV2.execution.streaming` 应改成 optional：`nil` 表示 Off；On 才发送 `{enabled:true, selection:"memory_tier", target...}`。

禁止：

- Off 发送 `enabled:true,target:0`；
- App 写入 `preset_id` 作为执行授权；
- App 写入 exact `P/G/K/D/Q`；
- App 缓存 resolution 后直接发 exact selector 绕过 native resolve；
- legacy `residency=streamed` 与 V2 selector 同时表达两个冲突意图。

native validation 顺序：

```text
schema/duplicate key
  -> legacy/V2 conflict
  -> target whitelist
  -> model/workload route
  -> catalog resolve
```

### 2.3 Options API 的状态模型

```swift
enum StreamingTargetStatus: String, Codable, Sendable {
    case available
    case unavailable
    case stale
    case revoked
    case unsupported
    case catalogEmpty = "catalog_empty"
}
```

空 catalog 输出示例：

```json
{
  "schema_version": 2,
  "catalog_revision": "tc-streaming-catalog-empty-v1",
  "query_status": "catalog_empty",
  "execution_container": "gpu-only",
  "device": {
    "gpu": "Apple GPU",
    "physical_memory_bytes": 68719476736,
    "device_class": "apple-unified"
  },
  "targets": [
    {"target_request_memory_bytes":8589934592,"status":"catalog_empty"},
    {"target_request_memory_bytes":10737418240,"status":"catalog_empty"},
    {"target_request_memory_bytes":12884901888,"status":"catalog_empty"},
    {"target_request_memory_bytes":17179869184,"status":"catalog_empty"},
    {"target_request_memory_bytes":21474836480,"status":"catalog_empty"}
  ]
}
```

App 必须能显示 unavailable，而不是隐藏所有选项，也不能把空 catalog 当作自动使用 legacy streaming。

## 3. 物理内存推荐策略

### 3.1 推荐只读物理内存

App 启动使用：

```swift
let physical = ProcessInfo.processInfo.physicalMemory
```

不使用：

- 当前 free memory；
- 单次 `mx::get_memory()`；
- 过去一次任务 peak；
- 把 unified memory 错称为独立显存后做错误换算。

下面的区间只产生一个 **raw candidate**，不能直接成为 UI 最终推荐：

```text
physical memory < 8 GiB  -> Off
8.. <10 GiB              -> 8 GiB
10.. <12 GiB             -> 10 GiB
12.. <16 GiB             -> 12 GiB
16.. <20 GiB             -> 16 GiB
>=20 GiB                 -> 20 GiB
```

最终推荐算法：

```text
raw_target = physical-memory band
options = native.streamingOptions(exact workload)
eligible = options.targets where
  status == available
  && record.minimum_physical_memory_bytes <= physical
  && record.headroom_policy is verified for this device class
recommended = highest eligible target <= raw_target
if none: Off
```

因此 8 GiB 物理内存机器不一定推荐 8 GiB：如果该 record 要求 10 GiB 最低物理内存以给 OS 留空间，最终必须推荐 Off。native options 仍可因为 model/workload/device qualification 把 raw candidate 标记 unavailable。瞬时系统压力只允许执行前拒绝或 runtime guard 安全失败，不允许静默换 record。

### 3.2 UI 文案

需要区分：

- `推荐 12 GiB（基于本机物理统一内存）`；
- `12 GiB 当前模型/尺寸无已认证 plan`；
- `请使用 Off 或选择有记录的档位`；
- `已选择 8 GiB（低内存优先，可能增加 I/O 时间）`。

未取得 P2/P3 evidence 前不能写“保证不 swap”“一定比 swap 快”。

## 4. Model × target catalog 设计

### 4.1 记录粒度

一个 production record 的主键至少包括：

```text
model_id
model_variant/source_digest
operation
execution_container/device_class
shape/frames/fps/batch
steps/scheduler
conditioning/feature digest
component policy
target tier
runtime/adapter/kernel/reader revisions
layout digest
```

同一模型不能用一条“通用 8 GiB”记录覆盖：

- 512 与 1024；
- 单图与 97 帧；
- text-to-image 与 image-to-video；
- 有音频与无音频；
- BF16 与量化；
- eager GPU 与 compiled/ANE；
- LoRA on 与 off。

### 4.2 Record 示例

```json
{
  "schema": "tc-streaming-preset-v3",
  "id": "h3-turbo-gpu-card-12g-reload-k1",
  "model": "minimax-h3-turbo",
  "target_request_memory_bytes": 12884901888,
  "workload": {
    "operation": "video.generate",
    "width": 512,
    "height": 512,
    "frames": 1,
    "steps": 20,
    "audio": false,
    "execution_container": "gpu-only"
  },
  "device": {
    "device_class": "apple-unified",
    "minimum_physical_memory_bytes": 17179869184
  },
  "plan": {
    "stages": [{
      "id": "denoiser",
      "resident_prefix_blocks": 8,
      "block_group_size": 1,
      "slot_count": 1,
      "prefetch_distance": 0,
      "io_workers": 1,
      "pass_transition": "reload",
      "multi_pool_policy": "serial"
    }],
    "layout_digest": "sha256:...",
    "component_policy_revision": "h3-gpu-bf16-v1"
  },
  "calibration": {
    "scope": "full_process_tree",
    "peak_p95_bytes": 0,
    "swap_out_bytes": 0,
    "sample_count": 20,
    "evidence_digest": "sha256:..."
  },
  "release": {"channel":"candidate","revoked":false}
}
```

示例数字不是认证数据。catalog builder 必须拒绝零 peak、空 evidence 或未经 verifier 的手写 production record。

### 4.3 四模型首批 card

#### Z-Image Turbo

```text
image.generate / BF16 eager GPU / 512x512 / batch 1
无 LoRA、ANE、compiled graph、GGUF、ConvRot
single denoiser + VAE boundary
```

候选顺序：8 GiB serial/K1 → 10 GiB K2 → 12/16 GiB increase prefix → 20 GiB compare higher prefix/overlap。

#### Flux.2 Klein 9B

```text
image.generate / BF16 eager GPU / 512x512 / batch 1
无 LoRA、ANE、compiled graph
dual + single class pool
```

8/10 GiB 先 serial；12/16/20 GiB 才比较 dual/single retain-all，denoiser→VAE 必须 drain/release。

#### MiniMax H3 Turbo

```text
video.generate / original BF16 C/Metal GPU
text only / no audio / no reference
all active blocks / core reuse 1
```

8/10 GiB 先 reload/K1；12 GiB 起搜索 K2/carry；16/20 GiB 比较 prefix/carry I/O 收益。只支持 H3 Turbo。

#### LTX 2.5

```text
video.generate / C/Metal GPU worker
text only / audio off / fixed 8+3 steps
exact shape/frame card
stage1 + stage2 independent pools
```

8/10 GiB 禁止跨 stage overlap；12/16 GiB 搜索 stage-local K/D；20 GiB 才考虑有证据的 component overlap。frames/fps/音频必须独立 record。

## 5. Target 到 exact plan 的确定性选择

### 5.1 Resolver 输入

```cpp
struct PublicResolveQuery final {
    std::string model_id;
    PresetWorkload workload;
    PresetSourceIdentity source;
    PresetRuntimeIdentity runtime;
    StreamingDeviceIdentity device;
    uint64_t target_bytes = 0;
    std::optional<std::string> preset_id;
    std::optional<std::string> catalog_revision;
};
```

### 5.2 算法

```text
1. validate target ∈ {8,10,12,16,20} GiB
2. validate model/operation/execution container
3. snapshot catalog once
4. filter revoked/channel/target/device qualification
5. filter source/workload/runtime/component identity
6. require calibration.complete and verified evidence
7. sort: explicit preset(test only), target, device, workload, revision
8. none -> target_unavailable
9. freeze record/layout/authority
```

resolver 不允许：

- 根据当前空闲内存改 target；
- 12 GiB 不 fit 后自动选 16 GiB；
- K2 failure 后改 K1；
- 混拼不同 revision 的 plan/calibration；
- 把 candidate/staging 当 production；
- 接受 App exact P/G/K/D/Q 覆盖 record。

### 5.3 Target margin 与 guard

```text
allowed_peak = target_bytes - max(512 MiB, ceil(target_bytes * 0.10))
```

这里的 `x=10%` 是 public v1 的固定 buffer policy，不是 App 用户参数。record 必须保存 `headroom_policy_revision`；将来若某模型需要 15% 或其他最小 reserve，必须发布新 policy revision 并重新校准，不能运行时偷偷改变公式。

record fit：

```text
tree_peak_p95 <= allowed_peak
swap_out_bytes == 0（非计划 swap）
unknown_child == false
sample_gap <= 100 ms
```

runtime guard 只允许：

```text
停止新 prefetch -> 完成当前 group -> drain -> target_exceeded
```

不能运行时改变 layout，否则 authority/receipt 失配。

## 6. 内存 ledger 与 process-tree sampler

### 6.1 统一 ledger

native 每次 public request 记录：

```cpp
struct StreamingMemoryLedger final {
    uint64_t descriptor_bytes = 0;
    uint64_t resident_prefix_bytes = 0;
    uint64_t slot_backing_bytes = 0;
    uint64_t slot_scratch_bytes = 0;
    uint64_t active_binding_bytes = 0;
    uint64_t kernel_scratch_bytes = 0;
    uint64_t component_handoff_bytes = 0;
    uint64_t io_buffer_bytes = 0;
    uint64_t allocator_fragmentation_bytes = 0;
    uint64_t driver_observed_bytes = 0;
    uint64_t process_tree_rss_bytes = 0;
    uint64_t process_tree_peak_bytes = 0;
    uint64_t swap_out_bytes = 0;
    bool complete = false;
};
```

字段含义不能混淆：

- logical read bytes 不是 slot backing bytes；
- MLX/Metal driver bytes 不是 process tree RSS；
- RSS 不是统一内存的完整 GPU allocation；
- `swap_out_bytes == 0` 不是没有 pressure；
- sampler gap/unknown child 存在时 ledger 只能是 `inconclusive`。

### 6.2 Sampler 设计

建议新增：

```text
tools/native/process_tree_sampler.py
tools/native/collect_streaming_memory.py
tests/native/test_process_tree_sampler.py
```

sampler 每 20 ms：

1. 读取 root pid 和已知 child pid；
2. 记录 pid/ppid、RSS、compressed、swap、phase/stage；
3. 读取 native runtime memory snapshot（若可用）；
4. 处理 child spawn/exit；
5. 写 append-only JSONL，并 fsync 每个 sample block；
6. 结束时记录 sampler errors、unknown child、最大 gap。

sample：

```json
{
  "timestamp_ns": 0,
  "pid": 123,
  "ppid": 1,
  "role": "ltx-worker",
  "rss_bytes": 0,
  "compressed_bytes": 0,
  "swap_out_bytes": 0,
  "driver_bytes": 0,
  "phase": "denoise",
  "stage": "ltx-stage1-denoiser"
}
```

独立 verifier 计算：

```text
tree_total(t) = Σ rss(pid,t) + driver_normalized(t)
tree_peak = max_t tree_total(t)
max_gap = max(sample[i+1].timestamp - sample[i].timestamp)
```

`driver_normalized` 的定义必须固定在 evidence runtime revision 中；不同机器不能使用不兼容公式后混合统计。

### 6.3 Runtime guard 状态机

```text
green
  -> warning (peak >= target - margin)
  -> stop_prefetch (no new fill)
  -> drain_current_group
  -> fit_success | target_exceeded | drain_unknown
```

guard 是安全响应，不是调度器第二入口。所有 layout 参数只来自已解析 record。

## 7. 校准工具链施工设计

### 7.1 CLI

建议固定为：

```bash
tc-stream inspect   --model-root ... --model ... --request request.json
tc-stream compile   --descriptor inspect.json --profile profile.json
tc-stream simulate  --candidate candidate.json --ledger ledger.json
tc-stream campaign  --policy policy.json --out evidence/
tc-stream memory     --pid PID --out memory.jsonl
tc-stream verify    --evidence evidence/
tc-stream catalog   --evidence evidence/ --channel staging
tc-stream revoke    --catalog catalog.json --record ID --reason ...
```

每一步只消费上一步的 immutable artifact，禁止使用未记录的环境变量隐式改变结果。

### 7.2 inspect

检查：

- model variant/route；
- source closure 文件集合、logical IDs、stat/content digest；
- index→shard 引用；
- workload/shape/step/audio/input identity；
- adapter/runtime/kernel/reader revision；
- route unsupported reason。

失败时输出 `unsupported_reasons[]`，不生成 candidate。

### 7.3 compile

输入是 model profile，不是 App target。生成 exact candidates：

```json
{
  "candidate_id": "flux9-dual-serial-k2-d1",
  "target_hint": 10737418240,
  "plan": {
    "stages": [{"id":"dual","slot_count":2,"prefetch_distance":1}]
  },
  "layout_digest": "sha256:...",
  "static_ledger": {
    "known_peak_bytes": 0,
    "unknown_fields": ["driver_observed_bytes"]
  }
}
```

compile 必须做 checked arithmetic、alignment、group/slot bounds 和 capability checks；静态 unknown 不能填成 zero。

### 7.4 simulate

模拟器至少输出：

```text
prefix_bytes
slot_backing_bytes
scratch_bytes
pool_live_intervals
component_overlap_bytes
io_queue_bytes
estimated_peak_lower_bound
unknown_upper_bound_fields
```

模拟器只用于剪枝，不是认证。lower bound 已超过 target 时丢弃；lower bound 未超不代表 fit。

### 7.5 campaign 与 verifier

复用现有 `tools/native/run_streaming_campaign.py` 和 `verify_streaming_campaign.py`，新增：

- process-tree memory JSONL；
- swap counters；
- actual receipt digest；
- stage/boundary semantic digest；
- output quality/artifact digest；
- candidate target and allowed peak；
- unknown child/gap failure。

正式 campaign：

```text
warmup: 3 per variant
paired measured: >=20 per candidate
ABBA/BAAB randomized blocks
quality replay: >=3
cancel/fault probes: >=3
```

P2 fit 和 P3 swap 对照不能共用一份只含 wall time 的 bundle。

## 8. resident / streaming / bounded / natural-swap 四臂

### 8.1 四臂定义

| arm | 配置 | 目的 |
|---|---|---|
| resident | Off/default | 默认性能基线 |
| streaming | explicit layout，无 guard | 纯 framework/streaming 成本 |
| bounded | catalog plan + target guard | 低内存 fit 与失败行为 |
| natural-swap | resident + 可控系统压力 | 与 OS swap 的实际比较 |

`natural-swap` 不能在普通 runner 中偷偷制造；必须由单独授权的压力 harness 执行并记录压力来源。

### 8.2 必须报告的指标

```text
success_rate
wall median/p95
denoise median/p95
first_output latency
tree_peak p50/p95/max
driver_peak p50/p95/max
swap_out_bytes
logical_read_bytes
reader_wait_seconds
GPU idle / I/O wait（若可测）
output quality/digest
failure/quarantine count
```

不能只报告“streaming 比 swap 快了 X%”；若成功率、peak、quality 或尾延迟不同，必须完整呈现。

### 8.3 允许的实验结论

允许记录：

- streaming 在某 target 上避免 swap，但 wall 更慢；
- streaming 在 swap arm 失败时仍成功；
- natural swap 在热 cache 下更快，但 p95/峰值更差；
- 某 model/shape 低档位不可用；
- 需要更大 target 或 Off。

禁止从设计直接推导“必然比 swap 更快”。只有 P3 paired evidence 才能回答。

## 9. App JobStore 事务

### 9.1 Job 保存字段

JobStore 可以保存：

```json
{
  "streaming_selection": "tier12",
  "target_request_memory_bytes": 12884901888,
  "request_digest": "sha256:...",
  "catalog_revision_seen": "...",
  "resolution_status": "available|stale|unavailable",
  "safe_summary": {
    "preset_id": "...",
    "layout_digest": "...",
    "target": 12884901888
  }
}
```

不能保存/恢复执行：fd、authority pointer、source generation、GPU pointer、receipt 作为新请求授权或可编辑 exact layout。

### 9.2 Generate 前事务

```text
validate draft
  -> query options (optional cached UI)
  -> construct V2 intent
  -> native generate revalidates and resolves
  -> unavailable/stale: no output
  -> accepted: create pending job
  -> success: attach public summary
  -> atomic persist JobStore
```

options cache 只用于 UI；catalog revision、model path、workload digest 任一变化都应刷新显示。generate 不依赖旧 cache 的 available 状态。

### 9.3 UI 状态机

```text
unknown -> loading_options -> available / unavailable / catalog_empty / stale

available + submit -> resolving -> running
                   -> succeeded | cancelled | failed | quarantined
```

unavailable 的 Generate 按钮 disabled，并给出换档位/Off 建议；不在后台自动选别的档位。

## 10. 分阶段实施计划

### A1 · App 数据模型迁移

文件：`apps/macos/StudioState.swift`、`apps/macos/App.swift`、`apps/macos/JobStore.swift`、`bindings/swift/TurboCiderNative.swift`。

验收：旧草稿可加载；Off 仍调用 legacy；五档显示 unavailable；不再只给 Z-Image 显示 streaming 控件。

### A2 · Native options/query 接入

文件：`native/api/c_api.mm`、`native/runtime/streaming/public_runtime.*`、Swift binding。

验收：physical memory 只做 recommendation；空 catalog 下 target 全 unavailable；无 selector 的 request 不查询 catalog。

### A3 · JobStore V2 generate

文件：`apps/macos/JobStore.swift`、`apps/macos/LTXWorker.swift`、`native/api/c_api.mm`。

验收：四模型统一构造 V2 intent；LTX worker 只接 envelope；desktop 不携 authority/fd。

### A4 · Model tier calibration

顺序：Z-Image 512 → Flux 9B 512 → H3 Turbo frozen card → LTX fixed frame card。

每个模型先 8 GiB，再 10/12/16/20 GiB；每个 target 至少选一个最简单 candidate 和一个性能 candidate。失败档位也记录 unavailable evidence，不删除。

### A5 · Catalog staging/release

只允许：`verified evidence -> catalog builder -> staging -> App QA -> production review`。

运行时或 source identity 变化时：`revoke record -> invalidate options -> preserve audit bundle`。

## 11. 验收矩阵

### App/UI

```text
APP-STREAM-001 Off is default for new draft
APP-STREAM-002 five tiers shown for all eligible models
APP-STREAM-003 unsupported model shows unavailable, not hidden
APP-STREAM-004 recommendation uses physicalMemory only
APP-STREAM-005 P/G/K/D/Q absent from release UI
APP-STREAM-006 legacy draft migration preserves safe intent
APP-STREAM-007 stale catalog refreshes before submit
APP-STREAM-008 failed resolve creates no output file
APP-STREAM-009 cancelled worker leaves no verified output
```

### Native control plane

```text
APP-CTRL-001 Off does not query catalog/probe/lease
APP-CTRL-002 target whitelist rejects 6/14/24 GiB
APP-CTRL-003 empty catalog returns catalog_empty for all tiers
APP-CTRL-004 exact resolve is deterministic
APP-CTRL-005 revoked record rejected
APP-CTRL-006 source/workload/runtime mismatch rejected
APP-CTRL-007 target unavailable never auto-fallbacks
```

### Calibration/release

```text
CAL-001 inspect closure complete
CAL-002 compile checked arithmetic
CAL-003 simulate unknown fields explicit
CAL-004 process-tree sampler max gap enforced
CAL-005 P0/P1 ABBA verifier
CAL-006 P2 allowed_peak and no unplanned swap
CAL-007 P3 four-arm evidence
CAL-008 catalog builder rejects incomplete evidence
CAL-009 revoke invalidates resolver
CAL-010 dev/runtime identity change marks evidence stale
```

## 12. 性能验收与不回退保证

### 12.1 Off/default

```text
median <= 1.02 × baseline
P95 <= 1.05 × baseline
new public allocations = 0
new public threads = 0
new catalog/probe/lease work = 0
```

四模型默认 resident 单独运行，不能用低内存收益抵消 P0 回归。

### 12.2 Streaming same-layout

public generic executor 与 direct/private exact route 的同布局对照：

```text
median <= 1.02
P95 <= 1.05
output/quality equivalent
actual receipt semantic identity equal
```

H3 carry 的收益若来自额外 overlap，应单独作为 layout optimization evidence，不写成 generic framework overhead 改善。

### 12.3 Low-memory target

```text
tree_peak_p95 + max(512 MiB, 10% target) <= target
no unplanned swap
no unknown child/gap
quality/output pass
success rate 100% on measured set
```

失败档位记录 unavailable；不自动减少 slots 或偷偷换 layout。

## 13. 发布前 checklist

```text
[ ] App 默认 Off，旧草稿迁移安全
[ ] 五档固定为 8/10/12/16/20 GiB
[ ] 物理内存只用于推荐，未绕过 native resolve
[ ] P/G/K/D/Q 未出现在 release UI
[ ] Z-Image/Flux/H3/LTX route identity 各自独立
[ ] H3 只允许 MiniMax H3 Turbo original BF16 GPU
[ ] LTX authority/lease 在 worker 内创建
[ ] multi-stage receipt/boundary verifier 通过
[ ] production catalog 空时所有 target fail-closed
[ ] 每个 available record 有 source/workload/runtime/device identity
[ ] P0/P1/P2/P3 evidence 完整
[ ] natural swap 对照按独立 harness 完成
[ ] catalog builder/revoke/review 演练完成
[ ] dev merge 后重新跑 identity 和 P0
[ ] 未认证 ANE/hybrid 组合明确 unavailable
```

在 checklist 全部完成前，App 可以作为内部 staging UI 开发，但不得把任何档位显示为生产“已支持”，也不得承诺“避免 swap”或“比 swap 更快”。

## 14. 配置文件、生成物与目录规范

### 14.1 三类配置必须分开

```text
User intent config
  只含 Off/target，不含 layout

Candidate search profile
  开发/校准使用，声明可枚举的 P/G/K/D/Q 范围和能力

Released preset catalog
  由 verified evidence 生成，包含唯一 exact layout 和 release identity
```

三者不能共用一个 JSON：如果 candidate range 被误当作 release record，resolver 就可能在用户机器上现场调参，破坏确定性和 evidence。

### 14.2 建议目录

```text
profiles/streaming/
  schemas/
    model-card-v1.schema.json
    candidate-profile-v2.schema.json
    campaign-policy-v2.schema.json
    evidence-manifest-v2.schema.json
    preset-catalog-v3.schema.json
  models/
    z-image-turbo-bf16-gpu.json
    flux2-klein-9b-bf16-gpu.json
    minimax-h3-turbo-bf16-gpu.json
    ltx-2.5-distilled-c-metal-gpu.json
  campaigns/
    staging/
  catalog/
    staging.json
    production.json

results/streaming/
  evidence/<calibration-id>/
  rejected/<calibration-id>/
  revoke/<record-id>/

native/runtime/streaming/generated/
  production_catalog.inc
  production_catalog.digest
```

`results/streaming/evidence` 是否入库由仓库 artifact policy 决定；至少 manifest、摘要、签核和外部 artifact digest 必须可追踪，模型权重和巨大 raw media 不入库。

### 14.3 Model card

model card 只定义能力和候选空间，不定义 production target：

```json
{
  "schema": "tc-streaming-model-card-v1",
  "model_id": "minimax-h3-turbo",
  "route": {
    "operation": "video.generate",
    "execution": "gpu",
    "precision": "bf16",
    "audio": false,
    "inputs": ["text"]
  },
  "stages": [{
    "id": "denoiser",
    "prefix_candidates": [0, 2, 4, 8, 12],
    "group_candidates": [1],
    "slot_candidates": [1, 2],
    "distance_candidates": [0, 1],
    "worker_candidates": [1, 2],
    "pass_transitions": ["reload", "carry_first_group"],
    "pool_policies": ["serial"]
  }],
  "exclusions": [
    "audio",
    "reference-input",
    "quantized-cache",
    "ane",
    "approximation"
  ]
}
```

数组是搜索空间，不表示所有组合有效。compiler 仍需 capability、bounds、memory lower bound 和 layout identity 剪枝。

### 14.4 LTX multi-stage profile

```json
{
  "schema": "tc-streaming-candidate-profile-v2",
  "model_id": "ltx-2.5-distilled",
  "stages": [
    {
      "id": "ltx-stage1-denoiser",
      "prefix_candidates": [0, 4, 8],
      "group_candidates": [1],
      "slot_candidates": [1, 2, 3],
      "distance_candidates": [0, 1, 2],
      "worker_candidates": [1, 2, 3]
    },
    {
      "id": "ltx-stage2-denoiser",
      "prefix_candidates": [0, 4, 8],
      "group_candidates": [1],
      "slot_candidates": [1, 2, 3],
      "distance_candidates": [0, 1, 2],
      "worker_candidates": [1, 2, 3]
    }
  ],
  "boundaries": [
    {
      "id": "stage1-to-upsampler",
      "require_drain": true,
      "require_release": true,
      "allow_overlap_candidates": [false]
    },
    {
      "id": "stage2-to-vae",
      "require_drain": true,
      "require_release": true,
      "allow_overlap_candidates": [false]
    }
  ]
}
```

首版 `allow_overlap_candidates` 只有 `false`。未来增加 `true` 时要变更 component policy revision 并重新跑 P1/P2，不能只改 profile。

### 14.5 Campaign policy

现有 campaign schema 应 additive 升级，至少包含：

```json
{
  "schema_version": 2,
  "status": "frozen",
  "comparison_kind": "P2",
  "model_card_digest": "sha256:...",
  "candidate_digest": "sha256:...",
  "target_request_memory_bytes": 12884901888,
  "headroom_policy": {
    "revision": "tc-public-headroom-v1",
    "fraction_ppm": 100000,
    "minimum_bytes": 536870912
  },
  "sampler": {
    "scope": "full_process_tree",
    "interval_ms": 20,
    "maximum_gap_ms": 100,
    "unknown_child_policy": "fail"
  },
  "protocol": {
    "warmups_per_variant": 3,
    "measured_pairs": 20,
    "order": "randomized-abba-baab",
    "quality_replays": 3,
    "fault_probes": 3
  }
}
```

policy 一旦 `status=frozen`，runner 不得自动补字段、修改阈值或增加排除样本规则。

### 14.6 Catalog 生成与编译

推荐流程：

```text
verified evidence manifests
  -> catalog builder validates schemas/digests/signatures
  -> canonical production.json
  -> generate production_catalog.inc + digest
  -> native build embeds immutable snapshot
```

`production_catalog.inc` 是生成物，不能手改。CI 重新生成后必须与工作树一致；不一致说明 source catalog 或 generator 未提交。

release binary 只读取 embedded production snapshot。开发/测试可以通过 test-only provider 注入 staging catalog，但相关 setter 不得导出到 release binary。

### 14.7 配置优先级

```text
release user selector (Off/target)
  -> immutable production catalog exact record
  -> compiled canonical layout
```

开发环境允许：

```text
test exact preset selector
  -> test/staging catalog
```

不允许：

```text
environment variable -> silently override production layout
profile file -> override public exact record
App cached resolution -> override current catalog
manual P/G/K/D/Q -> enter public route
```

### 14.8 Config 验收

```text
CFG-001 all JSON schemas reject unknown required-field omissions
CFG-002 canonical encoding is byte deterministic
CFG-003 model card cannot be loaded as preset record
CFG-004 candidate ranges cannot enter production catalog
CFG-005 generated catalog matches source digest
CFG-006 release binary has no staging override symbol
CFG-007 headroom policy revision participates in record digest
CFG-008 runtime/kernel/reader change marks record stale
CFG-009 duplicate preset identity is rejected
CFG-010 revoked record remains auditable but cannot resolve
```
