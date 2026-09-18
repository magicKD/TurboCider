# 49 · Public Streaming 代码合同与执行蓝图

修订日期：2026-09-18。状态：**实施设计；不是 public 资格声明**。

本文把 [48 工程实施附录](48-public-streaming-engineering-addendum.md) 再向下展开到“开发者可以按文件施工、reviewer 可以按函数审阅、测试可以按 ID 验收”的层级。本文只定义代码合同和实现顺序，不把已有 host fixture、private candidate 或 tiny GPU smoke 误写成生产支持。

当前事实仍以 [13 实施进度](13-implementation-progress.md) 为准：production catalog 为空；Z-Image 只有第一阶段 lease adapter；Flux 的 text/VAE lease lineage 已通过本轮 native build 和 host contract，但真实 Metal pager/full request 仍待验证；H3 Turbo、LTX worker、App 五档 UI、完整档位校准和 swap 四臂仍未完成。

## 1. 实现原则

### 1.1 一个请求只有一个 authority

一次 public generate 必须有唯一的 `ResolvedRequestExecution` 和唯一的 `StreamingAuthority`。任何模型 helper、worker 或 Swift 层都不能自行重新解析档位、读取 catalog 或生成另一个 layout digest。

```text
Request JSON
  -> request-only validation
  -> immutable catalog snapshot
  -> model probe + SourceLease
  -> exact record + ModelStreamingSnapshot
  -> non-serializable StreamingAuthority
  -> GPU-lock revalidation
  -> one request-scoped executor
```

以下对象不是 authority：

- App 的 `streamingSummary`；
- JobStore 中保存的 preset id；
- model session 上一次请求的成员变量；
- descriptor-only 的估算结果；
- receipt 中的 `layout_digest` 字符串。

它们只能作为输入或证据，不能授权一次新的 GPU 执行。

### 1.2 `Off` 必须是一条真正的旁路

`Off`/缺省路径必须在 API 边界完成分支，并且不能调用：

- `StreamingCatalogProvider::snapshot()`；
- `SourceLease::capture()`；
- `probe_public_streaming()`；
- `StageExecutor`/`CompletionMailbox` 构造；
- streaming audit counter；
- public receipt recorder。

实现上推荐让请求对象保留明确的 variant，而不是用空字符串表达关闭：

```cpp
struct StreamingIntent final {
    enum class Kind : uint8_t { off, memory_tier, exact_preset };
    Kind kind = Kind::off;
    uint64_t target_bytes = 0;
    std::string preset_id;
};
```

`kind == off` 时，默认模型入口直接调用原有 resident/private route。不得为了“统一代码”创建一个空的 `ResolvedRequestExecution`，否则很容易在内存充足机器上增加 allocation、锁或文件扫描。

### 1.3 任何优化都必须先保留安全合同

允许优化：

- claim 后预取下一 group；
- 同 class pool 的 retain-all；
- H3 `carry_first_group`；
- MLX `mmap`/fd duplicate 的 reader 复用；
- stage boundary 的流水化。

不允许以优化为名省略：

- 最后 reader fence；
- source generation revalidation；
- pass/class barrier；
- drain/quarantine；
- actual receipt 的 group/fence 事件；
- default route allocation audit。

## 2. 代码分层和 include 依赖

### 2.1 依赖方向

```text
core contracts / stream_slot_c.h
        |
        v
runtime/streaming (layout, lease, resolver, executor, receipt)
        |
        +--> model descriptor / adapter
        |       |
        |       +--> platform reader (Metal/MLX)
        |       +--> model kernel/session
        |
        +--> native/api C ABI
                    |
                    +--> Swift/App
```

禁止反向依赖：

- `runtime/streaming` 不包含 Swift/Foundation/MLX/模型头文件；
- `core` 不包含 `SourceLease` 或 `StageExecutor`；
- C ABI 不把 C++ `shared_ptr`、fd、Metal object 或 authority 指针暴露给 Swift；
- model kernel 不读取 catalog 或修改 JobStore。

### 2.2 推荐 include 规则

| 文件 | 可以 include | 不应 include |
|---|---|---|
| `source_lease.hpp` | `<filesystem>`, STL | model/platform header |
| `resolved_request.hpp` | `layout.hpp`, `preset_catalog.hpp`, `source_lease.hpp` | `c_api.h`, Swift |
| `context.hpp` | `actual_receipt.hpp`, `slot_pool.hpp`, `io_executor.hpp` | catalog/resolver |
| `public_runtime.hpp` | `catalog_provider.hpp`, `resolved_request.hpp` | Metal/MLX |
| `flux_streaming.hpp` | common runtime + Flux descriptor | App/Swift |
| `mlx.hpp` | MLX/base types | streaming runtime implementation types |
| `c_api.mm` | runtime + model public API | model-specific private fields |

如果 backend 只需要 lease 类型，应使用 forward declaration；不要让 `mlx.hpp` 依赖整个 streaming runtime，避免默认 MLX 编译路径发生头文件扩散和 ABI 变化。

## 3. 精确接口合同

### 3.1 public coordinator

推荐保持以下生命周期：

```cpp
PublicStreamingPreflight preflight(const Request &) const;

std::shared_ptr<const ResolvedRequestExecution> resolve_normalized(
    Request,
    const StreamingDeviceIdentity &,
    PublicStreamingPreflight) const;

void revalidate(const ResolvedRequestExecution &,
                const StreamingDeviceIdentity &) const;
```

合同：

1. `preflight()` 只做 request-only validation 和一次 catalog snapshot，不 probe 模型、不打开权重、不拿 GPU lock。
2. `resolve_normalized()` 只能消费 preflight ticket 中的 request digest；digest 不匹配立即失败。
3. `resolve_normalized()` 生成的 probe/snapshot 必须共享一个 `SourceLease`，不能 probe 一次、reader 再按 path 打开一次。
4. `revalidate()` 必须在 global GPU lock 已持有、任何 GPU payload 创建前调用。
5. `revalidate()` 失败时不得创建 slot pool、MLX pager、Metal buffer 或 worker。

### 3.2 模型 adapter 三段式接口

每个 public 模型都实现同一语义，函数名可以按现有 session 风格适配：

```cpp
std::shared_ptr<const ModelStreamingProbe>
probe_public_streaming(const PublicResolveInput &) const;

std::shared_ptr<const ModelStreamingSnapshot>
compile_public_streaming(
    std::shared_ptr<const ModelStreamingProbe>,
    const StreamingPresetRecord &) const;

RunResult generate_resolved(
    std::shared_ptr<const ResolvedRequestExecution>,
    const Event &, std::atomic<bool> &);
```

`probe` 禁止：GPU allocation、slot creation、异步 I/O、改变 session 当前模型状态。

`compile` 禁止：GPU allocation、worker creation、按 path reopen；允许解析 lease duplicate fd、构造 immutable descriptor/layout。

`generate_resolved` 必须：

- 检查 authority 和 snapshot identity；
- 在 context 内构造 pool、pager、reader、receipt；
- 只使用 snapshot 的 lease；
- 在每个 stage 完成后 drain；
- 在 result 返回前执行 common verifier；
- 任何异常都走同一 cleanup。

### 3.3 request context

最终实现建议增加一个只在 generate 栈上存在的对象：

```cpp
class PublicStreamingRunContext final {
public:
    PublicStreamingRunContext(
        std::shared_ptr<const ResolvedRequestExecution> execution,
        std::atomic<bool> &cancel);
    PublicStreamingRunContext(const PublicStreamingRunContext &) = delete;
    PublicStreamingRunContext &operator=(const PublicStreamingRunContext &) = delete;
    ~PublicStreamingRunContext();

    void attach_stage(std::unique_ptr<StageExecutor>);
    void mark_gpu_locked();
    void drain_or_quarantine() noexcept;
    void seal_receipt();
    bool quarantined() const noexcept;
};
```

析构顺序必须是：

```text
cancel dispatch
 -> drain mailbox/readers
 -> adapter drain
 -> seal receipt (仅成功 drain)
 -> source revalidate_after_drain
 -> release safe backing
 -> keep quarantined objects alive when drain unknown
```

不得在析构函数中吞掉“无法 drain”的状态。析构函数可以记录错误并把对象转移给 quarantine owner，但不能对仍可能被 GPU/reader 使用的 backing 调用 free。

### 3.4 executor 与 adapter 的线程合同

| 操作 | owner thread | I/O worker | GPU callback |
|---|---:|---:|---:|
| 修改 slot 状态 | 是 | 否 | 否 |
| 写 receipt vector | 是 | 否 | 否 |
| 读取 source fd | 否 | 是 | 否 |
| 提交 fill completion | 否 | 是（POD mailbox） | 否 |
| 提交 reader fence | 是/adapter | 否 | 否 |
| 标记 reader complete | owner consume | 否 | 只能 post POD |
| 触碰 Metal/MLX backing | adapter owner | 仅允许受控 fill | callback 不直接 free |
| destroy pool | owner | 否 | 否 |

worker 只能向固定容量的 completion mailbox 写入 POD 记录。mailbox overflow 是 fatal；不能无限增长 vector 作为“临时修复”。

## 4. 统一事件和错误传播

### 4.1 事件顺序

每个 group 的最小事件序列：

```text
group_planned
 -> fill_submitted
 -> fill_completed
 -> slot_claimed
 -> group_submitted
 -> reader_fence_issued[n]
 -> reader_fence_completed[n]
 -> slot_reusable
```

以下事件必须按实际发生记录，而不是在 `finish()` 时补写：

- `pool_selected`；
- `fill_completed.actual_bytes`；
- 每个 fence 的 queue、event value、generation；
- carry 的 from/to pass、slot、content generation；
- drain completion。

### 4.2 错误分类

native 错误应包含稳定 code、阶段和可安全重试性：

```cpp
enum class StreamingErrorCode {
    request_invalid,
    catalog_empty,
    no_record,
    route_unsupported,
    source_stale,
    source_short_read,
    descriptor_mismatch,
    layout_mismatch,
    authority_mismatch,
    target_not_fit,
    io_failed,
    reader_timeout,
    drain_unknown,
    actual_receipt_mismatch,
    output_verification_failed,
    worker_lost,
};
```

每个 error envelope 最少包含：

```json
{
  "code": "source_stale",
  "phase": "pre_gpu_revalidate",
  "retry": "resolve_again",
  "request_digest": "sha256:...",
  "catalog_revision": "...",
  "source_generation": 17,
  "safe_to_retry_same_context": false
}
```

映射规则：

- `request_invalid`、`route_unsupported`：不创建任何 streaming runtime；
- `no_record`、`target_not_fit`：可由 UI 引导用户换档位，但不得自动换档；
- `source_stale`、`authority_mismatch`：当前 context 不可重试，必须重新 resolve；
- `reader_timeout`、`drain_unknown`：quarantine，禁止复用当前 context；
- `actual_receipt_mismatch`：视为 adapter/record 缺陷，不 fallback 到 resident；
- 用户取消且 drain 成功：`cancelled`，可以新建请求。

## 5. 文件级施工建议

### 5.1 common runtime

| 文件 | 施工内容 | 先决测试 |
|---|---|---|
| `resolved_request.hpp/.cpp` | authority identity、request digest、source generation | resolver mismatch、copy/delete contract |
| `public_runtime.*` | preflight/resolve/revalidate 三段事务 | empty catalog、stale catalog、lock-time revalidate |
| `source_lease.*` | capture、duplicate fd、path/open revalidate | replace、same-size mutation、short read |
| `actual_receipt.*` | stage/group/fence/carry canonical digest | missing group、wrong slot、wrong generation |
| `public_result.*` | result 与 receipt 双向核对 | digest mismatch、drain false |
| `context.*` | cleanup、quarantine、cancel | exception、cancel、mailbox overflow |
| `slot_pool.*` | generation、state transition、reader count | early reuse、double release |
| `io_executor.*` | bounded workers/mailbox/deadline | queue full、worker exit、partial read |

每次 common runtime 改动都必须同时跑默认 contract。不能只跑 streaming fixture，因为 include、锁或计数器的变化可能影响 Off 路径。

### 5.2 Flux 9B lease lineage

当前工作树已开始把 text encoder/VAE 也改成 lease fd lineage。实施时必须明确区分两条路径：

```cpp
// default/private path
Weights::load(model_path);

// public exact path
Weights::load_lease(lease, logical_ids, event, cancel);
```

`load_lease()` 的要求：

1. logical id 必须在 probe closure 中存在；
2. 通过 `SourceLease::duplicate_fd()` 得到 owned fd；
3. 将 owned fd 包装成实现 MLX `io::Reader` 的只读 reader，顺序读取使用受锁 cursor，随机读取使用 `pread`；
4. `Weights` 持有 reader，MLX lazy array 也持有同一 reader，使 fd 生命周期覆盖所有 array 消费；
5. 不得把 fd reader 失败静默 fallback 到 path reopen；
6. default `Weights::load(path)` 不应创建 lease、reader 或执行额外 probe。

Flux public route 的 source closure 至少包含 transformer config/index/shards、text encoder config/weights、VAE config/weights 和 tokenizer JSON。denoiser drain 后释放 transformer backing，再进入 VAE；receipt 必须包含两个 component boundary 的真实顺序。

建议新增测试：

```text
FLUX-LEASE-001  text encoder fd-reader safetensors load
FLUX-LEASE-002  VAE fd-reader safetensors load
FLUX-LEASE-003  same-size source mutation rejected
FLUX-LEASE-004  missing tokenizer/text/VAE closure rejected
FLUX-LEASE-005  loader lazy-read fd lifetime
FLUX-LEASE-006  default path does not call load_lease
```

在 `FLUX-LEASE-005` 不能证明 fd 生命周期前，不得把 lease-backed text/VAE 标记为 public ready。

### 5.3 Z-Image

Z-Image 的第一阶段实现应与 Flux 使用相同的 request context，但不复制 Flux 的 component policy。必须明确：

- transformer stage 的 descriptor/layout digest；
- text encoder 和 VAE 是否由同一 closure 管理；
- transformer→VAE 边界是否有 pending reader；
- GGUF、LoRA、ANE、compiled graph 是否拒绝；
- output digest 是否由 public result verifier 计算。

如果 Z-Image 当前 adapter 仍有 session 成员保存 lease/target，必须在最终 PR 中改成 context-local；迁移期可以用 RAII restore，但不能让并发请求看到上一请求的 target。

### 5.4 H3 Turbo

H3 public 只覆盖 MiniMax H3 Turbo 原始 BF16 GPU route。代码施工顺序：

1. descriptor 通过 lease 读取 config/index/shards；
2. C/Metal reader 为每个 active block 绑定 `content_generation`；
3. 先实现 reload + K1 serial；
4. 接入 K2/G1 的 carry-first-group；
5. actual receipt 记录 carry 和跨 pass pool selection；
6. 完成 default resident P0 后再开放 public candidate。

普通 H3、量化 H3、ANE/hybrid、未声明的 multimodal branch 必须在 route validation 阶段拒绝，而不是走 H3 Turbo 的 record。

### 5.5 LTX 2.5

LTX 是唯一必须先升级 multi-stage result/receipt 的模型。推荐的 worker 事务：

```text
desktop creates JobEnvelope only
  -> worker validates request
  -> worker captures SourceLease
  -> worker resolves exact record
  -> worker acquires GPU lock
  -> worker runs stage1
  -> drain + boundary receipt
  -> upsampler boundary
  -> stage2
  -> VAE/audio/export
  -> verify receipt + output
  -> atomic output commit
```

desktop 不得持有 fd、authority 或 Metal backing。IPC EOF/SIGKILL 后，worker supervisor 必须把 job 标成 `worker_lost`，不能把临时输出标成 verified；drain unknown 时保留 quarantine 句柄并阻止同 GPU 立即重试。

## 6. 内存 ledger 和预算守卫

### 6.1 统一 ledger 字段

每次 request 的 ledger 至少记录：

```text
descriptor_bytes
resident_prefix_bytes
slot_backing_bytes
slot_scratch_bytes
active_binding_bytes
kernel_scratch_bytes
component_handoff_bytes
io_buffer_bytes
allocator_fragmentation_bytes
driver_observed_bytes
process_tree_rss_bytes
process_tree_peak_bytes
swap_out_bytes
```

这些字段不能相互替代。尤其：

- `logical_read_bytes` 不是 `slot_backing_bytes`；
- `driver_observed_bytes` 不是完整 process-tree peak；
- `RSS` 不是 GPU allocation；
- `swap_out_bytes == 0` 不是没有 pressure；
- descriptor 估算不能生成 production record。

### 6.2 admission 与 runtime guard 分离

推荐分为三层：

```text
static admission: record 的已校准 peak + margin 是否 <= target
runtime guard:    process-tree/driver 采样是否接近红线
safety response:  停止新 prefetch、drain 当前 group、失败并报告
```

runtime guard 不能在中途改变 `K/G/P/D/Q`。如果当前 layout 已经超出 target，必须停止派发并安全失败；不能偷偷把 K2 改成 K1，因为这样会使 receipt/layout digest 与 authority 不一致。

### 6.3 采样与峰值定义

采样器每 20 ms 读取主进程及所有已知子进程，允许最大 gap 100 ms。每个样本保存：

```json
{
  "timestamp_ns": 0,
  "pid": 0,
  "ppid": 0,
  "rss_bytes": 0,
  "compressed_bytes": 0,
  "swap_out_bytes": 0,
  "metal_bytes": 0,
  "phase": "denoise",
  "stage": "transformer"
}
```

`tree_peak` 是同一时间窗口内所有纳入 process set 的 RSS/driver 归一化值的最大值；如果存在未知 child、终止样本缺失或 gap 超限，结果为 `inconclusive`，不能自动放行。

## 7. 验收和测试矩阵

### 7.1 L0：纯 host 合同

```text
L0-REQ-001  selector schema / duplicate keys / legacy conflict
L0-REQ-002  Off path no catalog/probe/lease allocation
L0-REQ-003  empty catalog fail-closed
L0-REQ-004  exact selector digest stable
L0-SOURCE-001 source lease capture/open/revalidate
L0-RECEIPT-001 canonical digest deterministic
L0-AUTH-001 authority non-copyable and mismatch rejection
```

命令：

```bash
make test-streaming-host
make test-streaming-contract
python3 -B tests/native/test_streaming_source_lease.py
python3 -B tests/native/test_streaming_actual_receipt.py
```

### 7.2 L1：synthetic executor

覆盖 K1/K2/K3、D0/D1、Q1/Q2、two readers、cancel、short fill、mailbox overflow、pass barrier、carry、quarantine。必须在 ASan/UBSan/TSan 下运行，且每个失败都检查：

- slot/backing 是否仍由 owner 持有；
- receipt 是否没有伪造完成事件；
- 重复 destroy 是否安全；
- quarantine 是否阻止 retry。

### 7.3 L2：模型 metadata/adapter host

每个模型至少有：

```text
descriptor identity
source closure completeness
wrong component policy
same-size mutation
path replacement
unsupported route
invalid target
bound-after-failure repeated call
```

### 7.4 L3：真实 GPU smoke

真实模型 smoke 只证明功能接线，不证明 public 资格。必须保存：

- model/source/runtime/device identity；
- request digest/layout digest；
- actual receipt；
- output digest/质量比较；
- process tree sample；
- 是否使用冷/热 cache；
- build commit 和 binary hash。

### 7.5 P0/P1/P2/P3

| 门 | 目标 | 通过条件 |
|---|---|---|
| P0 | Off/default 不回退 | median ≤ 1.02、P95 ≤ 1.05；新增 allocation/thread/hook 为 0 |
| P1 | 同布局 generic vs direct | wall/denoise median ≤ 1.02、P95 ≤ 1.05；输出一致 |
| P2 | target 预算 | `tree_peak + max(512 MiB, 10%) ≤ target`，无非计划 swap |
| P3 | streaming vs natural swap | 四臂 paired 数据；不预设谁更快，报告成功率/峰值/吞吐/尾延迟 |

P0/P1 必须使用 ABBA 或随机交错顺序、至少 20 个 paired request、冷启动和热 cache 分开报告。单次 raw wall 不能替代统计门。

## 8. PR 分批和停止条件

### PR-A：common lease/receipt

只修改 runtime/common 和 host tests。停止条件：Off path audit 非零、source identity 不稳定、receipt verifier 只能从 result 合成。

### PR-B：Flux public lease

包含 Flux transformer/text/VAE/pager 的同一 lease lineage。停止条件：任意 fd-reader 失败后静默 path fallback、reader 生命周期短于 lazy array、VAE 边界未 drain。

### PR-C：H3 Turbo

先 reload/K1，再 carry/K2；两个 candidate 不能混入同一 record。停止条件：普通 H3/ANE route 未拒绝、C destroy 无法证明安全。

### PR-D：LTX multi-stage

先 additive receipt/result，再 worker transaction，最后档位 campaign。停止条件：stage boundary receipt 缺失、worker kill 后临时文件可见、stage1 pool 未释放。

### PR-E：App/catalog

只在 staging catalog 有真实 evidence 后显示可用档位；UI 仍可显示 unavailable。停止条件：空 catalog 下出现 Generate enabled、stale job 静默执行、Off 路径触发 streaming runtime。

## 9. 合并 dev 后的重新验证

`dev` 合并不是文档动作，而是 identity 变化风险。若以下任一项变化，旧 evidence 全部标记 stale：

- model kernel/reader revision；
- descriptor/index parser；
- source logical id 或 artifact closure；
- Metal/MLX runtime；
- output post-processing；
- C ABI receipt schema；
- Swift/JobStore request serialization。

合并后最小命令包：

```bash
env TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
make test-streaming-host
make test-streaming-contract
make test-streaming-audit
tools/native/build_app.sh
git diff --check
```

任何默认路径性能回归都必须先修复或显式回滚该合并，不能用 streaming 低内存收益抵消 P0 回归。

## 10. Definition of Done

单个“模型 + workload + target tier”只有在以下都满足时才可以写入 production catalog：

1. adapter 的 source closure 可列举、可重放；
2. resolver、authority、snapshot、reader 共用一条 lease lineage；
3. actual receipt 由 executor 实际记录，并通过独立 verifier；
4. cancel/fault/source mutation/worker loss 不产生假成功；
5. P0、P1、P2 通过，P3 有四臂数据；
6. 输出质量/字节或明确的容差证据已签核；
7. clean-tree evidence bundle 可重建；
8. App options/recommendation/unavailable/revoke/rollback 已验证；
9. `dev` 合并后重新验证 identity；
10. catalog builder 只接受 `verified` evidence，不接受手写 record。

在此之前，文档必须使用 `design`、`candidate`、`staging` 或 `unavailable`，不能使用“已支持 public”“保证不 swap”或“比 swap 更快”等未被证据支持的表述。
