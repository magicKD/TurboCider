# 46 · Public Streaming 实施总手册：框架、代码、档位校准与验收

修订日期：2026-09-18。状态：**实施总手册；不是 public 能力声明**。

本文把 [44 下一阶段实现收口规格](44-next-implementation-code-and-integration-spec.md)、[45 验收追踪与 Evidence](45-acceptance-traceability-and-evidence-spec.md)、[41 调度器实现](41-scheduler-multi-slot-and-multi-pool-implementation.md) 和 [42 四模型施工手册](42-model-adapter-playbooks.md) 收敛为一份可以直接分配给实现、App、性能和 release 负责人的执行手册。

本文特别区分三种状态：

1. **已实现事实**：当前分支中已有代码和测试证明的内容。
2. **设计合同**：已经冻结接口、状态机或验收规则，但尚未全部编码。
3. **发布证据**：必须在真实模型、真实设备和完整请求上取得，不能由 host test、tiny smoke 或文档示例替代。

当前 production catalog 仍为空。即使 private candidate 可以运行，也不能将其写成 App `available` 或 public 支持。

---

## 1. 目标、边界与不可妥协的约束

### 1.1 目标

将 block/slot/layer streaming 变成 TurboCider 的通用能力：

```text
用户只选择：Off / 8 / 10 / 12 / 16 / 20 GiB
                         |
              Native options / resolver
                         |
      model + workload + device + source exact match
                         |
      reviewed preset record（内部 P/G/K/D/Q/pool）
                         |
          descriptor -> layout compiler
                         |
        owner pump + slot pool + bounded I/O
                         |
       model-specific exact core + receipt verifier
```

用户不需要知道 `resident_prefix_blocks`、`block_group_size`、`slot_count`、`prefetch_distance`、`io_workers` 或 pool 拓扑。它们是 catalog record 的内部实现，不是稳定的用户 API。

### 1.2 范围

本阶段只考虑单机、单 GPU/Apple unified-memory GPU、单活动作业；四个模型按以下顺序接入：

1. Z-Image Turbo；
2. Flux.2 Klein 9B；
3. MiniMax H3 Turbo；
4. LTX 2.5（独立 worker）。

H3 范围严格限定为 **MiniMax H3 Turbo**。普通 H3、其他量化版本、ANE/VDN/VSA 变体不继承本方案资格。

### 1.3 不可妥协的约束

| 约束 | 设计要求 |
|---|---|
| 默认性能 | `Off`/resident/compiled/GPU+ANE 原路径不进入 resolver、lease、receipt、pool 或新线程热路径 |
| exact 性能 | 同 source、同 layout、同 workload 下 generic executor 相对 direct exact 的 wall median ≤ 1.02、P95 ≤ 1.05；denoise 另测 ≤ 1.02 |
| 安全 | source lease、layout digest、receipt、reader fence、drain 任一不完整即 fail-closed |
| 内存 | 目标是完整请求 process-tree envelope；不是只测 denoiser backing，也不是允许 swap 的额度 |
| 可回滚 | 每个模型/档位独立 catalog record，可单独 revoke，不需要回滚整个框架 |
| 确定性 | 不因运行时压力偷偷缩 K、改 G、改 dtype、跳 block 或切换 pool |
| 可审计 | 每次 public success 都能回答“哪个 source、哪个 layout、哪个 runtime、哪些 group/fence 实际执行” |

---

## 2. 当前基线：已实现、已验证和明确缺口

### 2.1 已提交代码事实

截至 2026-09-18，`feat/stream` 已提交：

- `SourceLease`：请求级 fd lineage、path/open-fd/post-drain revalidation、generation/digest；
- `PublicStreamingCoordinator`：单次 preflight/catalog snapshot、normalized resolve、execution-time revalidate；
- `PublicPresetResolver`：record/device/workload/runtime exact authority；
- `StageExecutor`：ordered multi-class barrier、K2 carry-first-group、claim 后 refill overlap、reader completion fast path；
- `ActualExecutionReceipt` v2：逐 pass/group/pool/slot/fill/fence/source generation 记录、canonical digest、独立 verifier；
- C ABI receipt additive 接口，不改变现有 v1/v2/v3 plan/callback 布局；
- `RunResult` 和 Objective-C JSON 的安全 streaming/receipt 摘要字段；
- host、contract、audit、Address/Undefined/Thread sanitizer 回归。

C2 的测试命令和结果记录在 [13 实施进度 §13.25](13-implementation-progress.md)。

### 2.2 仍未完成

| 区域 | 当前状态 | 放开 public 前的硬条件 |
|---|---|---|
| Z-Image | private exact stream 可运行；尚无完整 public hook | 共享 lease、真实 receipt、完整 PNG 请求、P0/P1、target calibration |
| Flux 9B | private dual/single candidate | 两类 pool barrier、retain-all 预算、VAE/output parity、P0/P1 |
| H3 Turbo | private K2/G1/carry candidate | carry receipt、native source closure、普通路径 P0 |
| LTX | private worker/exact 设计与局部实现 | worker-local authority、SIGKILL/quarantine、视频/音频/upsample/export 全闭包 |
| App | Swift C ABI options/resolve 基础类型存在 | 高级设置、stale guard、JobStore 原子事务、只显示五档 |
| 校准 | 无 reviewed 8/10/12/16/20 record | fresh-process process-tree 采样、四臂 swap 对照、独立 verifier |
| release | production catalog 为空 | reviewed evidence bundle、catalog builder、revoke drill |

禁止将上表中的 `private candidate`、tiny smoke 或 fake adapter 测试写成“已支持”。

---

## 3. 分层框架：控制面、数据面和证据面

### 3.1 控制面

控制面只决定“是否有资格执行、应该使用哪个 exact record”：

```text
Request JSON
  -> parse/normalize
  -> public request validation
  -> immutable catalog snapshot
  -> metadata-only model probe
  -> exact source/workload/runtime identity
  -> record selection
  -> snapshot descriptor/layout
  -> authority token
```

控制面禁止：

- 分配完整模型权重；
- 创建 Metal/MLX GPU buffer；
- 启动 refill worker；
- 以瞬时 free memory 改写 layout；
- 发现不 fit 后静默降级到普通 `generate()`。

### 3.2 数据面

数据面在 API 层获得 global GPU lock 后运行：

```text
authority revalidate
  -> create fixed/prefix backing
  -> create slot pool
  -> owner pump dispatch fill jobs
  -> worker pread into preallocated backing
  -> owner consumes POD completion
  -> GPU binds group and emits reader fences
  -> next fill overlaps current compute when safe
  -> pass/pool barrier
  -> final drain
```

每个 stage 只有一个 owner thread 修改 slot state、receipt vector 和 digest。I/O worker 只接收已经冻结的 fd/span/ticket，并返回固定大小 completion。

### 3.3 证据面

证据面不参与热路径决策；它记录：

- source identity、catalog/runtime/build identity；
- compiled layout digest；
- actual receipt v2；
- process-tree footprint/MLX/swap samples；
- wall/denoise/I/O/fence/queue timings；
- output hash/质量比较；
- cancel/fault/quarantine 结果。

证据面必须能由独立 verifier 在不重新信任 adapter 自报的前提下重算。

---

## 4. 用户 API 与内部计划的双层契约

### 4.1 用户可见请求

App 高级设置只暴露：

```json
{
  "streaming": {
    "enabled": true,
    "target": "12GiB"
  }
}
```

`Off` 的规范语义是：

```json
{
  "streaming": { "enabled": false }
}
```

它不是“选择最小 streaming preset”，而是保留当前 resident/default 路径。

### 4.2 Native selector

解析到 C++ 后使用现有 `StreamingSelector` schema v2：

```json
{
  "schema_version": 2,
  "enabled": true,
  "selection": "memory_tier",
  "retention": "request",
  "target_request_memory_bytes": 12884901888
}
```

selector 只表达用户意图。它不能携带内部布局，也不能直接授权执行。

### 4.3 Resolved execution

只有 resolver 成功后，内部才形成：

```json
{
  "selector": {
    "target_request_memory_bytes": 12884901888
  },
  "selection": {
    "preset_id": "z-image-turbo.comfy-bf16.512x512.12gib.r1",
    "preset_revision": 1,
    "catalog_revision": "catalog-2026-09-18-a",
    "record_digest": "sha256:...",
    "layout_digest": "sha256:..."
  },
  "exact_layout": {
    "denoiser": {
      "resident_prefix_blocks": 14,
      "block_group_size": 1,
      "slot_count": 2,
      "prefetch_distance": 1,
      "io_workers": 2
    }
  }
}
```

内部 JSON 只用于 debug/evidence，不作为稳定 App schema。App 结果只展示档位、状态和安全摘要。

### 4.4 物理内存自动预选

默认仍为 `Off`，以保护内存充足机器的当前表现。可选的“推荐”逻辑只能改变高级设置中的推荐值，不能静默切换执行路径：

```text
published_targets = {8, 10, 12, 16, 20 GiB}
physical_memory = device_info().physical_memory
system_reserve = reviewed device/system reserve

recommended = largest T where
    T <= physical_memory - system_reserve
    && at least one reviewed record matches model/workload/device
```

如果没有匹配 record，显示 `unavailable` 和稳定 reason code；不能因为“物理内存足够”而推导出布局可用。

---

## 5. 统一布局与调度模型

### 5.1 参数语义

| 参数 | 语义 | 运行中是否可变 |
|---|---|---|
| `P` / `resident_prefix_blocks` | 直接常驻且跨 pass 复用的前缀 block 数 | 否 |
| `G` / `block_group_size` | 一个 fill/绑定/计算 group 包含的连续 block 数 | 否 |
| `K` / `slot_count` | 同一 pool 中可循环使用的 backing slot 数 | 否 |
| `D` / `prefetch_distance` | 当前 ordinal 之后最多提前准备的 group 距离 | 否 |
| `Q` / `io_workers` | 并发 I/O worker 数 | 否 |
| pool policy | `serial` 或 `retain_all` 等 pool 生命周期策略 | 否 |
| pass transition | `reload` 或显式 `carry_first_group` | 否 |

`slot_count` 是精确值，不是 upper bound；不 fit 时拒绝。`G` 不能跳过 block，也不改变模型计算顺序。

### 5.2 Slot 状态机

```text
Vacant
  -> FillQueued
  -> Filling
  -> Ready
  -> InUse
  -> ReaderPending
  -> Releasable
  -> Vacant
```

不变量：

1. `Filling` slot 不能被 GPU bind；
2. `InUse/ReaderPending` slot 不能被 refill 覆写；
3. 一个 `(pass, group)` 只能有一个 active ticket；
4. reader fence 未完成时不能释放 backing；
5. owner 未消费 completion 前不能推进依赖该 completion 的状态；
6. 任意 sticky failure 后停止新 dispatch，并完成 join/drain/quarantine 判断。

### 5.3 Owner pump 伪代码

```cpp
while (!finished) {
    consume_fill_completions();
    consume_reader_completions();
    if (cancelled()) stop_new_dispatch();
    if (failure()) break;

    while (ready_to_dispatch_fill() && inflight_fills < Q) {
        auto ticket = next_ticket();
        require(slot_is_vacant(ticket.slot));
        mark_filling(ticket);
        io.submit(make_fill_job(ticket));
    }

    if (next_group_ready()) {
        auto ticket = claim_next_group();
        require(ticket.layout_digest == layout.digest);
        adapter.prepare_group(group(ticket), ticket);
        auto readers = adapter.encode_group(group(ticket), ticket, mailbox);
        recorder.readers_issued(ticket, readers);
        mark_reader_pending(ticket, readers);
    } else if (progress_possible()) {
        wait_for_completion_or_fence();
    } else {
        fail("streaming_scheduler_no_progress");
    }
}
```

不能使用无界等待、周期性全局 `synchronize()` 或在 worker 中直接修改 scheduler 状态。`D` 只增加已知安全窗口内的 overlap，不得突破 slot ownership。

### 5.4 Multi-pool

`serial`：切换 pool 前旧 pool 所有 reader 必须完成，峰值较低但 overlap 较少。

`retain_all`：多个 pool 同时保留；峰值必须把所有 capacity、固定组件、activation 和 allocator cache 一并计入。class barrier 只允许在 receipt 记录完整后推进。

Flux 的 dual/single pool、LTX 的 stage1/stage2、以及未来 VAE/upsampler 边界都必须显式声明 pool policy，不能通过对象析构时机隐式决定。

### 5.5 Pass 与 carry

`pass` 表示一次完整 denoise pass；`step` 是模型采样语义，二者不能混用。`carry_first_group` 只改变下一个 pass 首组的准备时序，不减少 logical fill 数：

```text
pass N 末尾：提前填充 pass N+1 group 0
pass N+1 开始：验证 carry ticket -> claim group 0
```

receipt 必须记录 from/to pass、group、pool、slot、content generation；最后一个 pass 必须没有 carry-out。

---

## 6. Source lease、snapshot 和 authority 的实现合同

### 6.1 单一 fd lineage

public adapter 必须采用：

```text
SourceLease::capture(files)
  -> open + fstat exactly once
  -> metadata/header parser uses duplicate_fd()
  -> probe and snapshot retain same shared lease
  -> reader uses duplicate_fd(), never path open()
  -> pre-GPU revalidate paths + open fds
  -> post-drain revalidate paths + open fds
```

Z-Image 当前缺口是 `StreamingMetadata` 和 `ZImageWeightStream` 各自按 path 打开；C3 必须改为共享 `std::shared_ptr<const SourceLease>`。

### 6.2 Probe

Probe 只能产生 immutable identity：

```cpp
auto lease = streaming::SourceLease::capture(source_files);
auto source = make_source_identity(*lease, manifest_digest);
auto workload = make_workload_identity(request, component_policy);
auto runtime = make_runtime_identity(build_id, adapter_rev,
                                     reader_rev, kernel_rev);

return std::make_shared<ValueModelStreamingProbe>(
    ValueModelStreamingProbe::Values{
        model_id, source, workload, runtime,
        component_policy_revision, lease});
```

Probe 阶段不能构造 MLX array、GPU buffer、worker 或 compiled graph。

### 6.3 Snapshot

Snapshot 使用 selected record 的 canonical config 编译 layout，并检查 digest：

```cpp
auto descriptor = make_descriptor_from_lease(*probe->source_lease(), workload);
auto layout = compile_layout(record.plan.canonical_config, descriptor);
require(layout.digest == record.plan.layout_digest,
        "streaming_layout_digest_mismatch");
return std::make_shared<ValueModelStreamingSnapshot>(
    ValueModelStreamingSnapshot::Values{
        model_id, probe->source_identity(), probe->runtime_identity(),
        std::move(descriptor), std::move(layout),
        component_policy_revision, shared_lease});
```

Snapshot 阶段不能创建运行时 pool 或 worker。

### 6.4 Authority revalidate

在 global GPU lock 后、任何 GPU work 前重新核对：

```text
request digest
catalog revision/record digest
source lease generation/digest
device identity
runtime/build identity
layout digest
component policy revision
```

任一变化返回稳定错误；不自动重选另一个 record。用户若重试，必须重新完成 preflight/probe/snapshot/resolve。

---

## 7. Receipt v2 与结果一致性

### 7.1 必须记录的实际事件

每个 `(pass, group)` 至少记录：

- expected/actual logical bytes；
- request/content/source generation；
- pool/slot；
- fill completion；
- group submitted；
- reader fences issued/completed；
- carry 和 pool selection；
- stage drain 状态。

`event_digest` 只编码确定性语义字段，禁止时间戳、指针、fd、路径和线程 id。时间戳进入 evidence 性能记录，不进入语义 digest。

### 7.2 验证顺序

```text
schema
-> stage/layout
-> group existence/duplicate
-> pool/slot
-> logical bytes/source generation
-> fence completeness
-> carry/pool barrier
-> drain
-> event digest
-> execution canonical digest
```

验证失败时：

1. 停止新增 dispatch；
2. join worker、等待可证明的 reader completion；
3. 无法证明 backing 安全时设置 `streaming_quarantined=true`；
4. 不产生 public success result；
5. 同一 engine 不得继续 generate，除非重新创建 session。

### 7.3 结果输出边界

App/JSON 只输出：schema、implementation revision、layout/record digest、计数、source generation、verified flag、memory scope 和安全错误摘要。

不输出：fd、path、inode、GPU pointer、完整 ticket matrix、worker PID 或未审阅的内部 P/G/K/D/Q。

---

## 8. 四模型 adapter 实施细节

### 8.1 Z-Image Turbo（C3，首个 public gate）

#### 冻结卡片

```text
model              = z-image-turbo
source             = Comfy BF16 single-file safetensors
operation          = image.generate
execution          = eager GPU Metal
shape              = 先冻结 512x512；其他 shape 独立 card
LoRA/ANE/compiled  = disabled
GGUF/NVFP4/ConvRot = disabled
```

当前 descriptor 为 30 个 dense blocks、每块 13 个 tensor；public adapter 仍必须从 lease metadata 重新确认，不得硬编码信任。

#### 候选搜索

首轮只搜索：`G=1`、single pool、`pass_transition=reload`、`K∈{1,2}`、`D<K`、`Q<=K`，逐步改变 `P`。候选必须满足 suffix group 数足够支撑 K 个 slot；不满足则拒绝，而不是自动降 K。

#### 代码修改

```text
native/models/z_image/z_image.hpp/.cpp
  probe_public_streaming()
  compile_public_streaming()
  generate_resolved()
  ZImageExactStream::enable_receipt()/receipt()

native/models/z_image/streaming_descriptor.hpp
  StreamingMetadata(SourceLease, logical_id)
  StreamingPlanView(SourceLease, config, workload)

native/platform/apple/z_image_streaming_descriptor.mm
  metadata parser uses lease->duplicate_fd("transformer")
  check_unchanged() delegates lease revalidate

native/platform/apple/z_image_weight_stream.mm
  constructor accepts shared lease
  fill_exact() uses duplicated fd; public path forbids open(path)

tests/native/z_image_public_streaming_test.cpp
tests/native/test_z_image_public_streaming.py
```

#### 完成条件

- probe/snapshot/generate 使用同一 lease；
- exact stream 产生真实 receipt，不能在 result 层合成；
- 512×512 完整生成、PNG export、cancel/fault/source mutation 全通过；
- resident P0、direct-vs-generic P1、至少一个真实 target P2 通过；
- production catalog 仍可为空，直到 review 完成。

### 8.2 Flux.2 Klein 9B（C4）

#### 冻结卡片

```text
model             = flux2-klein-9b
source            = reviewed BF16 eager weights
execution         = GPU Metal
first workload    = text-to-image；image-input/LoRA 独立 card
pool topology     = dual class + single class
```

当前 private candidate 研究结构是 8 个 dual block + 24 个 single block，但 public record 必须从 descriptor 取得实际计数。

#### 调度要求

- dual/single 使用不同 `storage_id` namespace；
- class barrier 前上一 pool 所有 reader 必须完成；
- `retain_all` 必须把两个 pool 总 capacity 计入校准；
- concat/binding 不能依赖上一个 pool 的地址稳定；
- 4B compiled graph 保持现有默认路径，不映射到 9B record。

#### 代码修改

```text
native/models/flux2/flux_transformer.cpp
  exact dual/single group bind、class transition、receipt hooks

native/models/flux2/pipeline.cpp
  public probe/snapshot/generate_resolved、VAE boundary、metrics

native/platform/apple/flux_streaming_descriptor.mm
  SourceLease closure、dual/single descriptor、pool policy

tests/native/flux_public_streaming_test.cpp
  class barrier、retain_all、VAE/output parity、fault/cancel
```

### 8.3 MiniMax H3 Turbo（C5）

#### 冻结卡片

```text
model          = minimax-h3-turbo
checkpoint     = original BF16
execution      = native Metal GPU
approximation  = false
ANE/VDN/VSA    = disabled
```

H3 的 active block 数由 descriptor/workload 决定；resolver 不能假设固定 block count。

#### Carry 实现

H3 首选 `G=1, K=2, D=1, Q=1`，在 receipt 正确后再评估其他 Q/D。carry 仅提前启动下一 pass 的首组 fill，不能减少 logical fill 计数。

#### 代码修改

```text
native/platform/apple/h3_session.mm
  public probe/snapshot/generate_resolved、source closure、receipt summary

native/models/h3_runtime/h3_streaming_descriptor.*
  active block、pass、carry metadata

native/models/h3_runtime/h3_dit.c
  exact group bind、reader fence、C bridge completion

tests/native/h3_public_streaming_test.cpp
  carry rotation、last-pass、ordinary H3 non-regression
```

### 8.4 LTX 2.5（C6，独立 worker）

LTX 不能让 parent 把 fd、Metal object、`ModelSession*` 或 authority pointer 发送给 worker。协议必须是可序列化 JobEnvelope：

```text
parent: validate request + create correlation id
parent: spawn worker with model root and JobEnvelope
worker: capture lease -> probe -> snapshot -> resolve
worker: revalidate under worker GPU lock
worker: execute stage1 -> upsample -> stage2 -> VAE/audio/export
worker: receipt + post-drain lease revalidate
worker: return verified summary/evidence path
parent: commit JobStore only after verified response
```

worker kill、pipe 断开、receipt 缺失或 drain unknown 都标记 `quarantined/retryable`，不能复用旧 session。

必须独立覆盖：Gemma/tokenizer、connector、Stage1 denoiser、latent upsampler、Stage2 denoiser、video VAE、audio/vocoder、export/finalizer。

---

## 9. 内部 profile、catalog 和版本管理

### 9.1 Profile 是实验输入，不是用户配置

推荐 profile：

```json
{
  "schema_version": 2,
  "profile_id": "z-image-turbo-512-12gib-candidate-v1",
  "model": "z-image-turbo",
  "source": {
    "model_variant": "comfy-bf16",
    "artifact_manifest_digest": "sha256:...",
    "source_snapshot_digest": "sha256:..."
  },
  "workload": {
    "operation": "image.generate",
    "width": 512,
    "height": 512,
    "frames": 1,
    "steps": 9,
    "batch": 1,
    "audio": false,
    "dynamic_text": true
  },
  "layout": {
    "stages": {
      "denoiser": {
        "residency": "streamed",
        "block_group_size": 1,
        "slot_count": 2,
        "resident_prefix_blocks": 14,
        "prefetch_distance": 1,
        "io_workers": 2
      }
    }
  },
  "policy": {
    "pass_transition": "reload",
    "multi_pool": "serial",
    "retention": "request"
  },
  "target": {
    "request_memory_bytes": 12884901888,
    "buffer_ratio": 0.10,
    "headroom_revision": "tree-peak-v3"
  }
}
```

Profile 允许离线工具搜索和仿真；未经完整 evidence 和 review，不能被 resolver 直接加载。

### 9.2 Catalog record

record 必须包含：

```text
record id/revision/catalog revision
source identity
workload identity
runtime/device qualification
canonical layout digest
component policy revision
calibrated process-tree bytes
memory scope/headroom/estimator revision
performance evidence digest
release channel/review commit/review digest
```

`release.channel` 只能是 `candidate`、`staging` 或 `production`；production 的必要条件见第 12 节。

### 9.3 版本兼容

以下任一变化必须产生新 record revision：

- source manifest/content；
- model adapter revision；
- reader/kernel revision；
- MLX/Metal/runtime build；
- component policy；
- layout P/G/K/D/Q；
- target/buffer/estimator；
- workload shape/steps/token rows/audio/LoRA；
- device class/OS family。

旧 record 不得通过“兼容推测”继续执行；catalog revoke 后已保存 JobStore 只能变为 `stale`。

---

## 10. 内存预算、采样与档位校准

### 10.1 预算公式

用户目标 `T` 是整个请求 process-tree envelope。reviewed headroom：

```text
H(T) = max(512 MiB, ceil(0.10 × T))
usable_ceiling = T - H(T)
```

候选必须满足：

```text
tree_peak + H(T) <= T
process_baseline + fixed_components + layout_working_set <= usable_ceiling
```

`T` 不是物理显存容量、GPU allocator hard cap 或 swap 额度。系统 swapout、采样缺失或未知子进程会使样本 FAIL/INCONCLUSIVE，而不是被当成“用了 buffer”。

### 10.2 采样边界

fresh process 的一次完整请求必须覆盖：

```text
process startup
model/session construction
tokenizer/text conditioning
prefix/fixed component load
every denoise pass
upsampler/VAE/audio/vocoder/export
worker join + reader drain
terminal sample before process exit
```

每个 sample 至少记录：

```json
{
  "mono_ns": 0,
  "pid": 0,
  "ppid": 0,
  "role": "denoiser-worker",
  "phys_footprint_bytes": 0,
  "mlx_active_bytes": 0,
  "mlx_peak_bytes": 0,
  "gpu_bytes": null,
  "swap_in_bytes": 0,
  "swap_out_bytes": 0,
  "pool": 0,
  "slot": 0,
  "fills": 0,
  "fences_issued": 0,
  "fences_completed": 0,
  "correlation_id": "..."
}
```

`tree_peak` 是同一时刻进程树 footprint 总和的最大值，不能把每个子进程独立峰值相加。

### 10.3 每个候选最低样本

```text
3 warmup（不计入统计）
20 matched measured requests
10 cancel/fault requests
2 source mutation requests
1 fresh-process rerun
1 independent verifier rerun
```

20 个 measured 请求中任何一个 output、receipt、peak 或 terminal sample 不完整，都不能发布该 record。

### 10.4 候选搜索顺序

为避免组合爆炸，固定 workload 后按以下顺序：

1. `K=1,2,3`，保持 `G=1`、`D=min(K-1,2)`、`Q=K`；
2. 只有单组容量明显大于 activation 时才试 `G=2,4`；
3. 调整 prefix，寻找减少重复读取但不挤压 activation 的位置；
4. 调整 D/Q，观察 fill wait、reader wait 和 overlap；
5. Flux 先完成 dual/single barrier，再比较 serial/retain_all；
6. H3 先完成 carry receipt，再比较其他 pass transition；
7. LTX 先保证 stage boundary drain，再研究跨 stage retention。

simulator 不通过的候选不进入真实 campaign。

---

## 11. 工具链与代码目录规划

### 11.1 工具链职责

```text
inspect   读取 source/descriptor/workload，不分配 GPU
compile   profile -> canonical layout，输出 digest
simulate  离散事件模拟 slot/pool/fence/内存峰值
campaign  在真实设备运行 matched requests 并采样
verify    独立验证 receipt、source、peak、输出和统计
builder   仅从完整 evidence 生成 catalog record
revoke    标记 record revoked，触发 App stale 行为
```

建议 CLI 约定：

```text
tools/streaming_inspect --model ... --workload ...
tools/streaming_compile --profile profile.json --out compiled.json
tools/streaming_simulate --compiled compiled.json --trace trace.json
tools/streaming_campaign --policy policy.json --fresh-process
tools/streaming_verify --evidence evidence/manifest.json
tools/streaming_catalog_builder --evidence evidence/ --out catalog.json
```

工具输出必须包含 build id、catalog revision、source digest、workload digest 和 command line，保证可复现。

### 11.2 推荐新增目录

```text
tools/streaming/
  inspect.py
  compile.py
  simulate.py
  campaign.py
  verify.py
  catalog_builder.py
  process_sampler.py
  schemas/

tests/native/
  streaming_simulator_test.cpp
  streaming_process_tree_fixture_test.cpp
  z_image_public_streaming_test.cpp
  flux_public_streaming_test.cpp
  h3_public_streaming_test.cpp
  ltx_public_streaming_test.cpp

tests/integration/
  StreamingOptionsTests.swift
  StreamingJobStoreTests.swift
  StreamingReleaseTests.swift
```

具体实现可以复用现有 `tools/native`、`tests/native` 和 `apps/macos` 结构，但不能另建第二套 executor 或第二套 selector 语义。

### 11.3 Simulator 最小模型

Simulator 的事件：

```text
fill_submit -> fill_ready -> claim -> gpu_submit
reader_issue -> reader_complete
pool_barrier -> pass_transition -> carry
cancel -> worker_join -> drain
```

输入是 compiled layout、每个 group 的 logical bytes、I/O latency 分布、GPU compute duration、fence delay 和 cancellation point。输出：

- 是否 deadlock/no-progress；
- slot overwrite/overcommit；
- 最大 inflight fill/reader；
- 估算 backing peak；
- fill wait、reader wait、overlap ratio；
- carry/pool barrier trace。

Simulator 不能替代真实 MLX/Metal 性能或 process-tree 内存证据。

---

## 12. 性能、swap 对照与发布门禁

### 12.1 四臂实验

固定 source、workload、seed、dtype、backend、OS/build、组件 route 和输出质量要求，比较：

1. **resident**：当前默认路径；
2. **streaming**：同一 workload 的 exact streaming layout；
3. **bounded**：启用内存 admission/guard，但不主动 swap；
4. **swap**：允许 OS/系统自然发生 swap 的压力对照。

每臂都要 fresh-process、warmup 分离、至少 20 个 matched measured requests。报告 wall、denoise、I/O wait、reader wait、peak、swap in/out、失败率、输出 hash/质量。

### 12.2 不预设 streaming 一定快

Streaming 是否优于 swap 只能由实验回答：

```text
same workload + same source + same device
  -> resident baseline
  -> controlled memory pressure
  -> streaming target
  -> natural swap control
  -> paired bootstrap / confidence interval
```

如果 streaming 较慢但能稳定 fit target，应如实标记为“内存收益、性能代价”；如果 swap 反而更快，也不能为了发布叙事隐藏该结果。性能收益不能抵销 P0 默认非回归失败。

### 12.3 P0–P4 门禁

| 门 | 内容 | 发布条件 |
|---|---|---|
| P0 | 默认 resident 非回归 | median wall ≤ 1.02、P95 ≤ 1.05、质量一致、audit 新工作为零 |
| P1 | 同布局 direct-vs-generic | median wall ≤ 1.02、P95 ≤ 1.05、denoise ≤ 1.02 |
| P2 | 目标档位 fit | 所有完整样本 `tree_peak + H(T) <= T`，无非计划 swap/failure |
| P3 | streaming-vs-swap | 只报告真实配对统计，不设先验赢家；swap/streaming 失败率也必须报告 |
| P4 | 策略探索 | 记录候选族、搜索空间、淘汰原因，不把策略收益冒充框架收益 |

### 12.4 Production record 准入

以下条件必须全部满足：

```text
catalog snapshot immutable
source lease pre-GPU/post-drain revalidate
probe == snapshot == record identity
layout digest exact
receipt v2 complete + independent verification
full process-tree calibration
P0/P1 pass
target fit pass
cancel/fault/quarantine pass
output quality parity pass
App stale/transaction tests pass
reviewed evidence bundle + signed catalog record
```

任一缺失时，record 只能是 `candidate`、`staging`、`unavailable` 或 `INCONCLUSIVE`。

---

## 13. App、JobStore 与失败恢复

### 13.1 App UI

高级设置建议显示：

```text
Streaming：关闭 / 8 GiB / 10 GiB / 12 GiB / 16 GiB / 20 GiB
状态：当前模型与工作负载可用 / 不可用
说明：档位是完整请求目标，不是显存硬上限
```

内部 P/G/K/D/Q 只允许在诊断 evidence 页面显示摘要，不进入普通 UI。

### 13.2 Options 查询

`tc_streaming_options_json` 对每个 target 返回：

```text
status: available | unavailable | stale | experimental
reason_code
preset_id/revision（仅安全摘要）
calibrated_request_bytes
memory_scope
release_channel
```

options cache key 必须包含 model、variant、operation、shape、steps、frames/fps/audio、dynamic_text、LoRA、ANE/compiled/approximation、source identity、catalog revision。任一改变都使旧 options 失效。

### 13.3 JobStore 状态机

```text
draft -> resolving -> resolved -> running -> verified -> committed
                         |                       |
                         v                       v
                       stale                 quarantined
running -> cancelled
```

JobStore 保存 selector、exact selector、record/catalog/resolution/workload digest 和安全 result 摘要；不保存 fd、inode、authority pointer、GPU pointer 或 worker PID。

`resolve` 成功前不能写入可运行 job。generate 前必须再次 revalidate；catalog revoke/source mutation/build mismatch 只能转为 `stale`，不能静默使用旧布局。

### 13.4 LTX worker 失败

父进程收到 worker 非零退出、SIGKILL、IPC EOF 或 receipt 缺失时：

1. 标记当前 job `quarantined` 或 `retryable`；
2. 删除不完整临时输出；
3. 不复用旧 worker/session；
4. 只有重新创建 worker 并重新 resolve 后才能重试。

---

## 14. 逐阶段代码实施计划

| 阶段 | 主要文件/工作 | 完成条件 | 回滚策略 |
|---|---|---|---|
| C2 | receipt v2、C ABI、verifier | 已完成；见 13.25 | recorder opt-in 关闭 |
| C3 | Z-Image lease 化、public hooks、真实 receipt | host + real 512×512 + P0/P1/P2 | catalog 仍空，仅保留 private |
| C4 | Flux dual/single adapter、pool barrier | retain-all/serial、VAE parity、P0/P1 | 仅撤销 Flux record |
| C5 | H3 Turbo adapter、carry receipt | last-pass/cancel/ordinary P0 | 仅撤销 H3 record |
| C6 | LTX worker-local adapter | 全组件、kill/quarantine、P0/P1/P2 | 回退 private worker |
| C7 | Swift/App/JobStore | options、stale、原子 commit、UI 五档 | 隐藏高级 streaming 入口 |
| C8 | sampler/campaign/catalog builder | 四臂、P0–P4、reviewed evidence | production catalog 保持空 |
| C9 | staging/release/revoke drill | 记录可发布、可撤回、可回滚 | restore previous catalog snapshot |

每阶段提交必须同时更新：

1. 对应主题文档的“已实现/待实现”；
2. 测试 ID 和命令；
3. evidence manifest（即使状态是 FAIL/INCONCLUSIVE）；
4. 回滚说明。

---

## 15. 代码审阅清单

### 15.1 默认路径

- [ ] `Off` 不调用 `production_streaming_catalog_provider()`；
- [ ] 不捕获 `SourceLease`；
- [ ] 不创建 receipt recorder、slot pool、worker；
- [ ] audit counters 的 hook/allocation/thread 为零；
- [ ] resident/compiled/GPU+ANE 的输出和 wall 与基线一致。

### 15.2 Public resolve

- [ ] preflight 只执行一次并绑定 immutable catalog snapshot；
- [ ] request digest 在 normalize 前后可验证；
- [ ] probe/snapshot 使用同一 lease；
- [ ] layout digest 与 record exact match；
- [ ] 不允许 adapter 自行降 K/G/P/Q 或 fallback 普通 generate。

### 15.3 Scheduler/receipt

- [ ] worker 只发布固定 POD completion；
- [ ] owner 独占 slot/receipt 状态；
- [ ] reader issued/completed 成对；
- [ ] carry/pool barrier 有确定性坐标；
- [ ] mailbox overflow、fence missing、drain unknown 会 quarantine；
- [ ] successful result 前完成 post-drain source revalidate 和 receipt verify。

### 15.4 App/release

- [ ] UI 不暴露内部布局；
- [ ] options 与 workload/source/catalog 完整绑定；
- [ ] JobStore 不保存不可序列化运行时对象；
- [ ] production catalog 由 builder 从 evidence 生成，不手写 available；
- [ ] revoke 后旧 job 变 stale，不能静默重跑。

---

## 16. 验收命令与最小 Evidence Bundle

### 16.1 Host/contract 回归

```text
env TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
tools/native/build_app.sh
make test-streaming-host
make test-streaming-contract
make test-streaming-audit
python3 -B tests/native/test_streaming_layout.py
python3 -B tests/native/test_streaming_preset_resolver.py
python3 -B tests/native/test_streaming_source_lease.py
python3 -B tests/native/test_streaming_actual_receipt.py
env TC_STREAMING_SANITIZER=address,undefined python3 -B tests/native/test_streaming_layout.py
env TC_STREAMING_SANITIZER=thread python3 -B tests/native/test_streaming_layout.py
git diff --check
```

### 16.2 Evidence Bundle

```text
evidence/<model>/<record>/<run>/
  manifest.json
  request.json                 # canonical request，不含 secret
  source.json                  # digest/identity 摘要
  device.json                  # GPU/OS/build 摘要
  compiled_layout.json
  actual_receipt.json
  process_tree_samples.jsonl
  memory_summary.json
  performance_summary.json
  output_hash.json
  fault_results.json
  verifier_report.json
  reviewer.md
```

manifest 至少包含：schema、record/catalog/revision、source/workload/runtime/device digest、command、git commit、sample counts、sampler interval、verifier revision、状态和失败原因。

### 16.3 证据状态

```text
NOT_STARTED / IMPLEMENTED / PASS / FAIL / INCONCLUSIVE
QUARANTINED / REVOKED
```

`IMPLEMENTED` 不是 `PASS`；一个 tuple 的 `PASS` 不能外推到其他 shape、LoRA、dtype、ANE、compiled route 或机器。

---

## 17. 下一步执行顺序（当前工作树）

按以下顺序推进，不跳过 source lease 和 receipt：

```text
C3 Z-Image：共享 lease -> public probe/snapshot/generate -> 真实 receipt
  -> host/synthetic/real 512x512 验收
C4 Flux 9B：dual/single pool -> retain_all/serial -> VAE/output parity
C5 H3 Turbo：carry-first-group -> receipt -> ordinary path P0
C6 LTX：worker-local authority -> full component closure -> kill/quarantine
C7 App：options -> Swift selector -> JobStore atomic transaction -> stale/revoke
C8 工具链：sampler/simulator/campaign/verify/catalog builder
C9 校准与发布：8/10/12/16/20 GiB、四臂 swap 对照、review、staging/release
最后：合并 dev，完整回归并重新签核 P0
```

如果任一步发现默认路径新增分配、source path reopen、receipt 可绕过、quarantine 泄漏、target peak 超限或 output parity 失败，立即停止后续阶段；production catalog 继续为空。

---

## 18. 完成定义

本手册对应的工程目标只有在以下条件全部满足时才算完成：

1. 四模型各有统一 probe/snapshot/generate adapter，且没有第二套 executor；
2. App 只需选择 Off/8/10/12/16/20 GiB，内部自动匹配 reviewed record；
3. 每个 public success 都有 source lease、layout digest、receipt v2、drain 和 process-tree evidence；
4. 每个已发布 target 都有独立校准，不从其他档位线性推导；
5. resident/default P0 和 same-layout P1 通过；
6. streaming 与 swap 的结论来自固定变量、配对样本和独立 verifier；
7. catalog 可发布、可撤回、可回滚；
8. merge dev 后再次通过完整 host/contract/audit/model/App/P0 回归。

在这些条件之前，正确的产品状态是“框架已实现、模型逐步验收中、public catalog 尚未开放”，而不是“所有模型都支持 streaming”。

---

## 19. 首个 adapter 的代码骨架（以 Z-Image 为准）

本节不是要求一次性复制完整实现，而是冻结最小接口形状，避免 C3 实现时重新发明一套控制流。函数名可以按现有命名微调，但语义不能改变。

### 19.1 `StreamingMetadata` 和 `ZImageWeightStream`

头文件建议改为：

```cpp
// native/models/z_image/streaming_descriptor.hpp
class StreamingMetadata {
public:
    explicit StreamingMetadata(
        std::shared_ptr<const streaming::SourceLease> lease,
        std::string logical_id = "transformer");
    explicit StreamingMetadata(const std::string &checkpoint); // private/legacy only

    const streaming::SourceLease &lease() const;
    void check_unchanged() const;       // lease->revalidate_*()
    streaming::Descriptor describe(const StreamingWorkload &) const;
};

class StreamingPlanView {
public:
    StreamingPlanView(
        std::shared_ptr<const streaming::SourceLease> lease,
        const StreamingConfig &, const StreamingWorkload &);
};
```

```cpp
// native/models/z_image/weight_stream.hpp
class ZImageWeightStream {
public:
    ZImageWeightStream(
        std::shared_ptr<const streaming::SourceLease> lease,
        unsigned pinned_blocks, uint64_t budget,
        uint64_t activation_reserve, Weights &fixed,
        const Event &, std::atomic<bool> &);

    // Existing path constructor remains private/legacy-only. Public generate
    // must not call it because it would create a second fd lineage.
    uint64_t fill_exact(uint32_t slot, uint32_t block,
                        const std::atomic<bool> *worker_cancel);
};
```

实现规则：

1. `StreamingMetadata` 保存 `std::shared_ptr<const SourceLease>` 和一个 `OwnedSourceFd`；解析 header 使用 `lease->duplicate_fd("transformer")`。
2. `ZImageWeightStream` 只使用来自同一 lease 的 duplicate fd；public fill callback 内禁止 `open(path)`、`ifstream` 或重新构造 metadata。
3. `check_unchanged()` 先校验 held fd，再校验 named path/canonical target；post-drain 使用 `revalidate_after_drain()`。
4. private/legacy path constructor 可以暂时保留，但必须在代码注释和测试中标记为非-public；不能由 `generate_resolved()` 走到。

### 19.2 `ZImage` 三个 public hook

`native/models/z_image/z_image.hpp`：

```cpp
std::shared_ptr<const streaming::ModelStreamingProbe>
probe_public_streaming(
    const streaming::PublicResolveInput &) const override;

std::shared_ptr<const streaming::ModelStreamingSnapshot>
compile_public_streaming(
    std::shared_ptr<const streaming::ModelStreamingProbe>,
    const streaming::StreamingPresetRecord &) const override;

RunResult generate_resolved(
    std::shared_ptr<const streaming::ResolvedRequestExecution>,
    const Event &, std::atomic<bool> &) override;
```

`probe_public_streaming()` 伪代码：

```cpp
auto files = z_image_source_files_for_public_card(root_, transformer_path_);
auto lease = streaming::SourceLease::capture(std::move(files));
auto workload = make_z_image_public_workload(request, lease->descriptor());
auto source = make_z_image_source_identity(*lease, "comfy-bf16");
auto runtime = make_z_image_runtime_identity();
return std::make_shared<streaming::ValueModelStreamingProbe>(
    streaming::ValueModelStreamingProbe::Values{
        "z-image-turbo", std::move(source), std::move(workload),
        std::move(runtime), "z-image-components-v1", std::move(lease)});
```

`compile_public_streaming()` 必须检查：

```text
probe.model_id == "z-image-turbo"
record.source == probe.source_identity
record.workload == probe.workload_identity
record.runtime == probe.runtime_identity
record.release.channel ∈ {candidate, staging, production}
```

然后从共享 lease 构造 descriptor，用 `record.plan.canonical_config` 调 `compile_layout()`，最后要求 `layout.digest == record.plan.layout_digest`。任一不匹配都返回稳定错误，不改写 record 或 selector。

`generate_resolved()` 的最小结构：

```cpp
auto &snapshot = *execution->model_snapshot;
snapshot.revalidate_source();
require(snapshot.model_id() == model_id_);
require(snapshot.layout().digest == execution->selection.record.plan.layout_digest);

ZImageExactStream stream(
    snapshot, execution->selection.record, transformer_,
    event, cancelled, next_request_generation());
stream.enable_receipt({
    snapshot.layout().digest,
    "generic_stage_executor_v2",
    snapshot.source_lease()->generation()});

run_text_conditioning_and_denoise(stream, execution->request, event, cancelled);
stream.finish();                         // joins workers + final drain
snapshot.source_lease()->revalidate_after_drain();

RunResult result = make_z_image_result(...);
result.streaming_receipt = stream.receipt();
result.streaming_runtime = stream.runtime_metrics();
return result;                            // common verifier runs in c_api.mm
```

这里的 `ZImageExactStream` 应增加：

```cpp
void enable_receipt(streaming::ExecutionReceiptOptions);
std::shared_ptr<const streaming::ActualStageReceipt> receipt() const;
StreamingRuntimeMetrics runtime_metrics() const;
```

它内部只允许在 `StageExecutor::begin()` 之后、首个 pass 之前启用 recorder；`finish()` 之后才允许读取 receipt。`stream.receipt()` 返回 executor 的真实 receipt，不得用 `ExecutionCounters` 在外层拼装。

### 19.3 `c_api.mm` 的一次性事务

public 分支应保持以下顺序：

```cpp
auto parsed = request_from_json(parse_json(raw));
std::shared_ptr<const ResolvedRequestExecution> resolved;
std::optional<PublicStreamingCoordinator> coordinator;

if (parsed.streaming_selector && parsed.streaming_selector->active()) {
    coordinator.emplace(...);
    auto preflight = coordinator->preflight(parsed);      // no GPU
    auto normalized = make_plan_after_public_streaming_preflight(parsed);
    resolved = coordinator->resolve_normalized(
        std::move(normalized.request), device_identity(), std::move(preflight));
}

std::scoped_lock engine_lock(e->mutex);
std::scoped_lock gpu_lock(global_gpu_mutex());
if (resolved) coordinator->revalidate(*resolved, device_identity());

RunResult result = resolved
    ? e->session->generate_resolved(resolved, event, e->cancelled)
    : e->session->generate(parsed, event, e->cancelled);

if (resolved)
    verify_and_attach_public_streaming_result(*resolved, result);
*out = copy(json(to_dictionary(result)));
```

禁止的实现：

```cpp
// 不允许：public unsupported 时偷偷走普通路径
try { return generate_resolved(...); }
catch (...) { return generate(parsed, ...); }

// 不允许：adapter 内重选布局
if (!fits) layout.slot_count = 1;

// 不允许：result 层伪造 receipt
result.streaming_runtime->receipt_fills = counters.fills;
```

### 19.4 四模型统一 adapter helper

建议把下列纯函数放在 `native/runtime/streaming/model_adapter_helpers.*`，避免每个模型复制 identity/digest 逻辑：

```cpp
PresetWorkload make_public_workload(const Request &,
                                    std::string_view component_policy);
PresetRuntimeIdentity make_runtime_identity(std::string_view adapter,
                                             std::string_view reader,
                                             std::string_view kernel);
PresetSourceIdentity make_source_identity(const SourceLease &,
                                          std::string_view variant,
                                          std::string_view format);
void require_exact_record_identity(const StreamingPresetRecord &,
                                   const ModelStreamingProbe &,
                                   const ModelStreamingSnapshot &,
                                   const StreamingDeviceIdentity &);
```

这些 helper 只能比较和构造 identity，不能持有 MLX/Metal 状态；这样 host-only 测试可以覆盖四模型的公共逻辑。

---

## 20. 实施与验收追踪矩阵

### 20.1 C3–C9 代码出口

| 阶段 | 代码出口 | 主测试 | 停止条件 |
|---|---|---|---|
| C3-A | Z-Image shared lease metadata/reader | `ZIMG-SRC-001..010` | 任意 path reopen、generation mismatch 或 short read |
| C3-B | Z-Image probe/snapshot | `ZIMG-PROBE-001..006` | probe 分配 GPU、layout digest mismatch 未拒绝 |
| C3-C | Z-Image exact receipt | `ZIMG-RUN-001..012` | receipt 缺失、fence 未完成、output 非完整 |
| C4 | Flux dual/single adapter | `FLUX-POOL-001..012` | class barrier/retain-all/地址复用错误 |
| C5 | H3 Turbo carry adapter | `H3-CARRY-001..010` | carry 坐标错误、last-pass 有 carry-out |
| C6 | LTX worker protocol | `LTX-WORKER-001..016` | SIGKILL/IPC EOF 后复用旧 session |
| C7 | Swift options/JobStore | `APP-STREAM-001..014` | stale selector 被静默执行、UI 泄露内部布局 |
| C8 | sampler/simulator/campaign/verify | `TOOL-STREAM-001..020` | 漏采样、未知子进程、证据无法重放 |
| C9 | catalog builder/revoke/release | `REL-STREAM-001..012` | 无 review/evidence 生成 production record |

### 20.2 每个真实 model card 的最小分母

```text
20 matched measured runs
10 cancel/fault runs
2 source mutation runs
1 fresh-process rerun
1 independent verifier rerun
```

卡片必须报告成功数、失败数、INCONCLUSIVE 数及其原因；不允许只报告“成功的平均速度”。

### 20.3 默认路径审计

对每个模型至少做以下 ABBA 运行：

```text
A = resident/default request
B = explicit streaming candidate request
B = repeat streaming candidate request
A = resident/default request
```

默认路径两次 A 的输出、peak、wall 和 audit counters 必须保持在 P0 门内；B 的结果不得污染下一次 A 的 session/cache/allocator 状态。若 A 的第二次运行被 B 改变，优先判定为生命周期回归，而不是接受“缓存差异”。

### 20.4 发布前人工签核问题

reviewer 必须逐项回答：

1. 这个 record 精确绑定了哪些 source/workload/device/runtime？
2. 真实执行的 receipt 是否与 compiled layout 一致？
3. process-tree 采样是否覆盖 model load、每个 pass、VAE/export 和 terminal sample？
4. 是否有任何非计划 swap、未知子进程或采样 gap？
5. resident/default 是否仍通过 P0？generic executor 是否通过 P1？
6. streaming 与 swap 的结论是否来自固定变量和配对样本？
7. revoke 后旧 JobStore 是否变 stale，且没有静默 fallback？
8. 是否可以只撤销这一条 record，而不影响其他模型和 Off/default 路径？

全部回答为“是”并附 evidence digest 后，才允许将 record 从 `staging` 提升为 `production`。
