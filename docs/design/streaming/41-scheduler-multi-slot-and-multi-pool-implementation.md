# 41 · Block/Slot 调度器、Multi-slot 与 Multi-pool 实现细节

[目录](README.md) · [框架合同](17-layout-first-framework.md) · [代码工作台](38-framework-code-contracts-and-implementation-workbench.md) · [模型施工单](42-model-adapter-playbooks.md)

修订日期：2026-09-17。状态：**调度实现设计；所有参数仍需按模型和 workload 校准**。

本文把“block/slot streaming”收敛成一个可复用的单 GPU 调度器。重点不是再定义一套模型逻辑，而是明确在当前 `StageExecutor`、`SlotSafetyTracker`、`IoExecutor` 和 C bridge 之上，怎样安全地实现多槽、跨 pass、异构 pool、I/O/Metal overlap，并保证内存充足时不影响默认路径。

## 1. 术语与职责

```text
Descriptor   模型 adapter 提供的逻辑 block/field/source 描述
Layout       compiler 根据显式 config 生成的不可变布局
Stage        一个有序的计算阶段（例如 denoiser）
Group        一次加载/绑定的连续 block 集合
Pool         同一 layout class 的 slot 集合
Slot         一个可复用的物理 backing bundle
Pass         一次完整 block 序列执行；可能对应一个 denoise step/stage
Ticket       (request, stage, pass, step, group, pool, slot, generation)
Fence        GPU reader 对 slot 内容的异步使用完成信号
```

职责必须单向：

```text
Model adapter -> Descriptor / source ranges / binding
LayoutCompiler -> deterministic groups/pools/capacities
StageExecutor -> owner-thread state machine and schedule
IoExecutor -> bounded refill, no slot-state mutation
GPU reader -> fence completion callback, no scheduling decisions
Receipt verifier -> expected vs actual execution proof
```

`StageExecutor` 不知道具体模型 tensor 名称；adapter 不得自行修改 slot 状态；App/selector/catalog 不得进入 data-plane 热路径。

## 2. 布局参数的精确定义

每个 streamed stage 的核心参数：

| 参数 | 符号 | 语义 | 运行时是否可改 |
|---|---:|---|---:|
| `resident_prefix_blocks` | P | 永久驻留、按 block 顺序位于 suffix 之前的 block 数 | 否 |
| `block_group_size` | G | 每次 fill 包含的连续 block 数 | 否 |
| `slot_count` | K | 精确 slot 数，不是上限 | 否 |
| `prefetch_distance` | D | owner 允许提前派发的 group 距离 | 否 |
| `io_workers` | Q | 固定 refill worker 数 | 否 |
| `pass_transition` | — | `reload` 或 `carry_first_group` | 否 |
| `multi_pool_policy` | — | `serial` 或 `retain_all` | 否 |

必须满足：

```text
1 <= K <= max_slots
1 <= G <= adapter.max_group_size
0 <= D < K
1 <= Q <= K
suffix_group_count(pool) >= K
```

如果某个候选不满足这些条件，compiler 直接拒绝；不能在执行时偷偷把 K 从 3 降到 2，也不能把 G 改为 1 来“救活”请求。

## 3. 内存模型

### 3.1 阶段峰值

对于一个 stage，估算值不是只有 `K × slot_bytes`：

```text
stage_peak =
    resident_stage_bytes
  + prefix_bytes
  + live_pool_backing_bytes
  + per_slot_scratch_bytes
  + activation_bytes(stage/workload)
  + kernel_workspace_bytes
  + reader_command_bytes
  + allocator/cache reserve
```

`live_pool_backing_bytes`：

```text
serial      = max(pool.capacity_bytes)
retain_all  = sum(pool.capacity_bytes)
```

完整请求峰值还要加上组件重叠：

```text
request_peak = max_t(
  parent + worker + encoder + stage_i + VAE/upsampler/audio + export + drain
)
```

因此 `Layout::peak_pool_bytes` 只用于 compiler/simulator 过滤，不能直接写入 `calibrated_request_bytes`。

### 3.2 各类内存的生命期

| 类别 | 典型 owner | 释放时机 |
|---|---|---|
| prefix backing | adapter/session | stage 或 request 结束，取决于 component policy |
| slot backing | adapter pool owner | pool drain 后显式 destroy |
| slot scratch | adapter slot owner | 与 slot backing 同生命周期 |
| activation | model kernel/runtime | pass/step 结束或 backend fence 完成 |
| reader buffer | `FillJob` | fill completion 或 worker join |
| VAE/upsampler | component owner | denoise drain 后按 policy 释放 |
| output/export | media owner | 输出落盘后释放 |

任何 backing 的释放都必须在最后一个 GPU reader fence 完成之后；仅看到 CPU fill 完成不代表 GPU 不再读。

## 4. Slot 状态机与所有权

### 4.1 状态

推荐使用当前 `ContentState` 的语义扩展：

```text
Vacant
  -> Loading       owner 已发出 ticket，worker 正在填充
  -> Ready         fill 完成，bytes 与 expected 匹配
  -> InUse         adapter 已绑定该 group 并提交 reader
  -> AwaitingFence reader 未完成，禁止复写
  -> Vacant        所有 fence 完成且 owner 回收 ticket
```

carry 场景的唯一例外：

```text
AwaitingFence -> Ready(carry)
```

此状态只能跨一个明确的 pass boundary 保存一个 carry group；任何其他“缓存命中”都必须重新 fill 或另建 layout policy。

### 4.2 线程所有权

```text
owner thread:
  begin_fill / accept_ready / begin_use / seal_readers
  complete_reader 消费 / pool switch / destroy / receipt append

refill worker:
  pread/convert/fill fixed backing
  post FillComplete 或 error

GPU callback:
  post ReaderComplete(ticket, fence)
```

worker 绝不直接调用 `SlotSafetyTracker`；completion mailbox overflow 是 sticky failure。owner 发现 overflow 后停止新 enqueue，进入 drain/quarantine。

### 4.3 Ticket 内容

当前 C ABI ticket 已经包含所需坐标，字段名必须保持 ABI v1 兼容：

```cpp
struct tc_stream_slot_ticket_v1 {
    uint32_t struct_size;
    uint32_t version;
    uint32_t pool;
    uint32_t slot;
    uint64_t request_generation;
    uint64_t content_generation;
    tc_stream_work_item_v1 item; // stage, pass, step, group
};
```

`request_generation` 隔离请求，`content_generation` 解决同一 slot 在不同 group/pass 中的 ABA 问题。任何 completion ticket 与当前 slot generation 不一致时拒绝并 quarantine；receipt v2 复用这些字段，不改变 v1 struct prefix。

## 5. Owner pump 算法

### 5.1 基本循环

```text
run_pass(pass, step):
  verify owner thread and lifecycle
  select/activate pool segment
  optionally consume incoming carry
  next = first uncompleted group
  dispatch = first group not yet enqueued

  while not boundary_complete:
    check cancel / mailbox overflow / timeout
    consume all available completions

    while dispatch < end
          and dispatch <= next + D
          and slot(dispatch) == Vacant:
        claim slot with ticket
        enqueue bounded FillJob
        dispatch++

    if prefix_not_encoded:
        adapter.encode_prefix(pass)

    if carry_next and last group reached:
        enqueue next-pass first group into free slot

    if group(next) == Ready:
        adapter.prepare_group(group, ticket)
        begin_use(ticket)
        adapter.submit_group(group, ticket)
        seal_readers(ticket, fences)
        next++

    if no progress:
        wait mailbox until timeout / completion

  adapter.drain()
  consume remaining completions
  verify quiescent (or carry exception)
```

`D` 只控制 owner 的最早派发窗口，不改变计算顺序。`next` 始终按 group id 单调前进，不能因为某个后续 group 先 ready 就先提交计算。

### 5.2 进度条件

一次 loop 至少有以下进展之一：

```text
completion consumed
fill enqueued
prefix encoded
group submitted
carry enqueued
```

超过 `stall_timeout` 且无进展，返回 `streaming_progress_timeout`；如果 GPU/worker 状态无法证明安全，随后进入 quarantine，而不是继续等待无限时间。

### 5.3 Backpressure

`IoExecutor::enqueue()` 返回 false 表示没有 queue credit，不是工作丢失。owner 应：

1. 先消费 mailbox；
2. 再检查 slot 是否因 fence 释放；
3. 重新尝试 enqueue；
4. 超时则失败并 drain。

禁止增长 `std::vector`、扩展 job ring 或在 steady-state 线程中动态分配。

## 6. 多槽 overlap 设计

### 6.1 K=1

```text
fill A -> wait ready -> use A -> wait fence -> refill A
```

K1 的内存最低，但几乎没有 I/O/compute overlap；适合作为低内存与安全基线，不应默认宣称最快。

### 6.2 K=2

```text
slot0 fill group0
slot1 fill group1 (可与 group0 compute overlap)
use slot0
while slot0 reader in flight: refill slot1 for group2
use slot1
```

K2 的关键是“先 claim 再发后续 fill”时，slot1 的 fence 已完成；`overlap_next_fill_after_claim()` 只能由 adapter 明确授权。

### 6.3 K=3/Q=3

K3 允许更深 I/O 窗口，但会增加常驻 backing、worker 和读带宽。调度器仍然使用同一 owner pump；不增加第二套线程模型。

### 6.4 D 与 Q 的关系

推荐初始约束：

```text
Q <= K
D < K
D=0 仍允许 claim-overlap（仅 adapter opt-in）
```

`D=K-1` 会尽早占满所有 slot，可能减少 compute starvation，但也可能把错误/取消推迟到更多 I/O 已发出之后。候选校准必须同时记录 fill wait、reader wait、logical read bytes 和取消延迟。

## 7. Multi-pool 调度

### 7.1 serial

适用于 layout class 之间字段签名不同、物理 backing 不能别名的模型：

```text
pool0 setup -> groups0 -> drain -> destroy pool0
pool1 setup -> groups1 -> drain -> destroy pool1
```

优点：峰值低；缺点：class barrier 有 setup/destroy 成本。`activate_pool()` 在切换前必须确认 active pool `drained=true`。

### 7.2 retain_all

适用于 class 数量有限、总 backing 可接受且 setup churn 明显的模型：

```text
setup pool0..N-1 once
select pool0 -> drain boundary
select pool1 -> ...
destroy all after request drain
```

`retain_all` 的容量是所有 pool 之和，不得把“steady-state 无分配”误写成低内存。Flux 9B 的 dual/single 是首个适合验证该策略的模型，但必须单独测 class barrier 和完整 VAE/export 峰值。

### 7.3 跨 pool carry 禁止

`carry_first_group` 首版只支持单 pool K2。禁止在 multi-pool 中把 carry ticket 从 dual pool 带到 single pool；需要这种优化时新增显式 pass transition 和 adapter capability，不能复用旧字段。

## 8. Pass、step 与 carry

### 8.1 pass 与 step 分离

`pass` 表示布局中的逻辑遍历；`step` 表示用户可见 denoise/采样步。一个 step 可包含多个 pass，receipt 必须同时记录二者。

```text
step 0: pass 0(stage1) -> pass 1(stage2)
step 1: pass 0(stage1) -> pass 1(stage2)
```

不得用 `pass_count` 替代用户 `steps`，也不得将 callback 的 step 数量直接当作 fill 次数。

### 8.2 carry-first-group

H3 Turbo 等可使用：

```text
pass N 最后一个 group 正在 encode
→ 在另一空闲 slot 中 fill pass N+1 的第一个 group
→ drain pass N 时保留该 Ready slot
→ pass N+1 首 group 直接 claim
```

receipt 必须证明：

- carry ticket 的 pass/step/group 坐标正确；
- carry slot 在边界处除 Ready 外无其他 InUse/Loading；
- 下一 pass 仍有其他 suffix group 的完整 fill；
- 最后一个 pass 没有残留 carry。

## 9. 取消、超时和 quarantine

### 9.1 取消路径

```text
cancel flag set
→ owner stops new dispatch
→ workers observe chunk boundary and finish/abort
→ IoExecutor shutdown_and_join
→ adapter drain GPU readers
→ consume mailbox
→ all slots quiescent?
    yes: destroy pools, return cancelled
    no: quarantine session
```

### 9.2 超时策略

建议分三类 timeout：

| timeout | 目的 | 超时后 |
|---|---|---|
| fill chunk deadline | 检测磁盘/reader 卡住 | 停止后续 fill，尝试 drain |
| progress stall timeout | owner 长时间无完成事件 | 取消并 drain |
| process/worker join deadline | 防止 App 永久挂起 | 进程隔离或 quarantine |

没有安全 drain 证据时，不能简单 `delete` executor；当前析构 `retry_drain()` 失败会 terminate，public worker 应改为显式 quarantine/finalizer 路径。

## 10. Receipt 与审计

每个 pass/group/fence 至少记录到内部固定容量 recorder：

```text
expected group id / pool / slot / bytes
actual fill bytes
logical source bytes
reader fences issued/completed
source generation
carry in/out
timestamps (enqueue, ready, claim, submit, fence)
```

worker 只发 POD completion；owner 在 finish 时一次性计算 digest。默认 Off 路径不创建 recorder。

## 11. 新增/修改代码建议

### 11.1 Common runtime

```text
native/runtime/streaming/context.hpp/.cpp
  - 将 owner pump 拆成可测试的 dispatch/consume/boundary helpers
  - 增加 optional ReceiptRecorder*，默认 nullptr

native/runtime/streaming/slot_pool.hpp/.cpp
  - 复用现有 request/content generation，增加 expected logical bytes 校验
  - 统一 carry 与 normal-ready 的状态转移

native/runtime/streaming/io_executor.hpp/.cpp
  - 固定 ring capacity
  - 增加 fill error/status/short-read completion
  - shutdown 后禁止 enqueue

native/runtime/streaming/actual_receipt.*
  - expected/actual matrix、digest、verifier

native/core/stream_slot_c.h
  - additive completion/receipt v2 POD，不破坏 v1 struct prefix
```

### 11.2 adapter 侧

adapter 只实现：

```cpp
create_pool(PoolLayout)
select_pool(PoolLayout)
make_fill_job(Group, Ticket)
prepare_group(Group, Ticket)
submit_group(Group, Ticket)      // 内部生成 reader fences
drain()
destroy_pool(pool_id)
```

adapter 不得：

- 直接修改 `ContentState`；
- 在 `make_fill_job` 中解析 JSON/manifest；
- 在 worker 内调用 MLX/Metal global synchronization；
- 绕过 executor 直接按 path 打开 source；
- 根据内存压力临时改变 layout。

## 12. 调度器单元测试

最小 fake adapter 必须覆盖：

```text
K1/K2/K3，G1/G2，D0..K-1，Q1..K
single pool / serial multi-pool / retain_all
reload / carry_first_group
reader fence delay 0/1/N
completion out-of-order
mailbox overflow
short read / zero fill / duplicate completion
cancel before fill / during fill / during reader / at boundary
stall timeout / worker join timeout
```

每个测试都要断言最终状态：`success`、`cancelled`、`failed` 或 `quarantined`，并检查 backing 是否仍被引用。

## 13. 调度性能验收

至少报告：

```text
wall median/P95
first-group startup
steady group interval
fill wait / reader wait
logical read bytes / fill count
pool create/destroy
steady allocation count
worker CPU time
tree peak / MLX peak / swap delta
```

同一 layout 的 direct 与 generic executor 才能计算 P1 framework ratio；改变 P/K/G/D/Q 的结果属于策略实验 P4。任何新 overlap 若导致默认 resident 路径有额外工作，直接违反 P0。

## 14. 完成定义

调度器实现完成不是“某次 K3 demo 跑通”，而是：

1. 所有状态转移有 owner/fence 证据；
2. single/multi-pool、reload/carry 有确定性 receipt；
3. cancel/fault/timeout 都能 drain 或 quarantine；
4. steady-state 无未计划分配；
5. Off/default 审计为零；
6. 通过 fake、sanitizer、真实模型 adapter 和完整请求验收；
7. 任意 target 的 public record 都能回溯到 exact layout digest 与 scheduler revision。
