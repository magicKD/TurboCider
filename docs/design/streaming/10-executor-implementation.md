# 10 · Executor 实施：有界流水线与热路径

[目录](README.md) · [协议](03-runtime-protocol.md) · [编译器](09-layout-compiler-spec.md) · [验收细则](12-acceptance-playbook.md)

状态：目标规格；当前 executor 已支持单活动 pool 的 streamed 执行，以及 ordered multi-class
barrier（按 class drain、销毁、重建），并有 host/合成测试；resident、模型/guard 接入尚未完成，见
[13](13-implementation-progress.md)。
03 定义生命周期合同；本文定义实现选择、进展条件与性能边界。首发单 GPU 作业不等于只能一个 GPU command queue。

## 1. 组件划分与职责

| 组件 | 拥有的状态 | 不得做什么 |
|---|---|---|
| `StreamingExecutionContext` | request generation、取消、stage cursor、终态 | 改 K/P、切换模型算法、自动重试 resident |
| `StageExecutor` | demand cursor、pass、提交/退休进度 | 接管 sampler 数值公式 |
| `SlotPool` | backing bundle、content generation、writer/reader 权利 | 独立创建 reader 线程、按压力扩容 |
| `IoExecutor` | Q 个 worker、固定任务/结果队列、固定 scratch | 配置重选、模型 context 分配、无限 future |
| `UseFenceTracker` | 每个 slot 的所有最后读者完成条件 | 把 scheduled/encode-return 当 completed |
| `BudgetBridge` | 可选现有 MemoryExecutionContext 窄接口 | 新建第二套与 ledger 不一致的全局账 |
| `ModelAdapter` | format reader、kernel binding、backend completion | 私有预取池、隐式全模型 materialize |

顶层采用 C++ orchestration、C ABI 接 LTX/H3；函数表按 stage 一次绑定。每 group 一次调度，不给每个 GEMM 增加虚函数。
无新 framework 时不实例化这些对象，legacy loop 继续直接调用当前 kernel helper。

### 1.1 现有 bridge 与后续接口草案

当前`stream_slot_c.h`/`c_bridge.cpp`已有allocate_slot/fill/prefix/prepare/encode/drain/destroy回调，
桥接同一个C++ StageExecutor；`create → run_pass(pass,step) × S → finish/destroy`支持跨pass保留池和线程。
它仍不提供真实模型descriptor、field级construction view或全请求guard；下面是职责草案，不是现有签名抄录。
现有fill无worker ID，按Q份scratch需要补worker context；首个adapter可按K份显式计账，见 [15](15-adapter-implementation-plan.md)。

建议 `stream_slot_c.h` 冻结以下签名职责，不把 C++ executor 传入 C kernel：

```text
describe(model_metadata, workload, descriptor_sink, error) -> status
create_pool(resolved_pool_view, owner_allocation_ops, out_pool, error) -> status
make_read_plan(pool, group_view, writable_spans, out_read_plan, error) -> status
fill(read_plan, worker_scratch, cancel_view, out_fill_result) -> status
prepare_group(pool, ticket, step_inputs, error) -> status
encode_group(pool, ticket, compute_inputs, completion_sink, out_submission) -> status
drain(pool, deadline, out_drain_result) -> status
destroy_pool(pool, out_release_result) -> status
```

所有 input view 在调用/异步消费期间保持有效；read_plan 存入固定 job record，直到 worker join/完成才销毁。
completion_sink 指向稳定 mailbox 控制块而不是栈上 callback context。create/destroy 仅 owner 使用 allocation ops，
fill 不得到这些 ops。status/error/cancel/结果结构带 version/size，不能返回可悬空的内部 error 字符串。
executor负责验票与发布状态；adapter不能绕过accept_ready直接调用encode。

## 2. Owner 驱动的 pump，而非逐层全局同步

定义 `j = 下一个尚未提交 compute 的 group`。它不是 GPU 已完成的 group，也不是刚读完的 group。
每次 compute 成功提交后才推进 j；任何新 fill 必须满足 `[j,j+D]`，并且目标 slot 无 writer/reader。

```text
while pass not complete:
    accept bounded completion records; validate generation and status
    retire only contents whose entire last-reader set has completed
    enqueue never-started fills in [j, min(j+D,last)], ascending group order
        - current demand has priority
        - stop if the next required dispatch has no slot/queue credit
    if group j is Ready and adapter submission credit exists:
        owner prepares step-derived values
        register ticket reader intent; encode and commit bounded work
        seal last-reader fence set; advance j; continue pump
    if an error/cancel occurred: enter cleanup
    if no progress: wait on completion/worker/cancel condition with deadline
```

ordered enqueue 是首发确定性策略：不跨过一个不能派发的较近 group 去抢占远期 I/O。
它可能牺牲少量吞吐，但容易复现和认证；后续 out-of-order dispatch 必须升级 schedule revision 并重新测。
完成可以乱序，compute 严格顺序。prefetch window 只决定是否允许开始，不保证磁盘能同时服务这些任务。

### 2.1 Startup / steady state / drain

- Startup：池和 prefix 在 owner 创建；prefix 开始计算前可启动前 D+1 个 suffix fill，Q 限制并发。
- Steady state：能推进就继续提交；只在 current demand、slot reuse、submission credit 或阶段依赖上等待。
- Drain：最后 group submit 不是完成；等待全部 reader、worker、回调，再允许 stage handoff/free。
- 下一 pass：backing 保留；首发先 drain 上一 pass，再按计划使内容失效并重读。不能保留尾块临时改变读序列。
- D=0 仍可在当前 group 已提交后读取下一 group 的另一槽；它禁止“当前需求之前的远期预读”，不等价于所有 GPU/I/O 完全串行。
- K=1 要等唯一 slot 最后 reader 完成才 refill，因此提供可验证的串行基线。

## 3. 防止 batch/fence 死锁的必要合同

如果所有 group 被编码进同一个未提交 command buffer，同时 owner 等 slot fence 再编码下一组，就会形成循环等待。
adapter 必须提供 `commit_safe_boundary`，在 slot 将复用前确保其最后读者已经提交。

首发策略：

1. 每 group 定义合法的结束边界；可以保留组内现有 batching/fusion。
2. `encode_group` 返回时已提交读者，或返回一个显式可提交的 batch handle；不能返回无法等待的伪 fence。
3. submission credits 与 K、backend 队列资源上限一起静态确定，作为 resolved plan 内部字段，不新增用户旋钮。
4. commit boundary 若破坏原算法或依赖未来 group，不强制切断；compiler 拒绝，或需要用户显式选择满足闭包的 G。
5. command buffer 数/commit 次数变化必须进入性能对照；“没有 CPU wait”不等于没有 GPU 批次退化。

新路径不能用每组 `device-wide synchronize` 替代此设计。同步-only adapter 可用于最早 correctness 实验，但不能因此通过性能发布。

## 4. Slot 写入权、derived 数据与 GPU 可见性

一个 content generation 只有一个 fill writer。worker 完成后 release publication，owner acquire 并验证 Ready；
owner 可在 compute 前写入已声明的 step-derived 数据，之后直到所有 reader 完成禁止任何 CPU 写入。
generation 和 sequence 使用 checked 64-bit 单调计数；将要回绕时拒绝新任务并 clean drain，不重置为0继续复用。

LTX 当前 `apply_scalar_conditioning_one()` 位于 fill 完成和 `run_block()` 之间。接入时必须审计它写哪些对象：

- 从 checkpoint 来的不可变参数，下一次可以重读，不需写回磁盘。
- 根据 timestep/conditioning 产生的派生表属于本次内容或独立 workspace，计入容量和写入阶段。
- 跨 step 必须保留的状态另建资源，不能归入“随时可丢弃的 immutable weights”。

adapter 的 `prepare_group` 明确限定在 owner 上执行，位于 Ready→InUse 之前。它不构成第三套 loader，不能隐式分配无限 scratch。
shared storage 不等于没有 CPU/GPU 数据依赖；private storage 另有 copy-complete→compute 依赖，staging 在 copy reader 结束后才可复用。

## 5. Fence 聚合、回调与 mailbox

`SlotUseFence` 是框架抽象，不是要求使用 `MTLFence` 对象。需要证明“CPU 可以改写”的完成条件，而不仅是 GPU pass 之间排序。
Metal bridge 首发优先使用已提交 command buffer 的真实 completion；注册点、错误状态与所有实际 queue 必须审核。
completion handler 在 commit 前注册；owner 的 reader intent 也在提交前登记，不能提交后再补回调。

- 单 queue：同序提交且已证明 ordering 时，可用最后读取该 content 的 sequence 代表之前全部读取。
- 多 queue：每个读取 queue 都有最后 reader，全部满足才 reusable；video/audio 任一未完成都不能复写。
- GPU callback 只投递 ticket、queue、sequence、status；不能操作 ledger、slot state 或释放 model context。
- 在提交前登记 reader intent；以 `fences_sealed` 位阻止“completion 太快、读者集合尚未完整”时提前退休。
- 错误提交需区分未提交与可能已提交，后者必须 drain/quarantine，不能回滚成从未使用。

队列容量按计划中的最大 outstanding 证明：ready records 上限 K，每 slot 最多 R 个合并后的 queue completion，
则 mailbox 至少预留 K+K×R 条普通记录；取消/error 用独立固定 latch，不能依赖一个已满的普通队列才能报告溢出。
如果 backend 不能将 completion 合并到 R，按实际最大数另算容量；不把该公式硬套所有实现。
回调不可阻塞等待 owner 腾位，overflow 触发 sticky failure/latch；最终 drain 用 backend 独立 completion 状态证明安全。

## 6. 有界 I/O 的具体实现

- Q 个 persistent worker 在新 context 启动一次；不为每个 block 建立/销毁 pthread。
- pending jobs+running jobs+ready content 不超过 K；任务 payload 使用预分配记录和 borrowed spans。
- `read_plan` 在 owner 固定，包含 fd identity、offset/length、destination、decode scratch 上限。
- 使用显式 offset reader，不能共享有竞争的全局文件 seek cursor；处理 EINTR、short read、EOF、overflow。
- scratch固定预分配并声明归属：有worker context时按Q份；现有C bridge首个adapter可按K份独占scratch，必须按K计账。格式只支持整shard解包时如实报告峰值，不能把“按group API”当按group实现。
- fill 分 chunk 检查 cancel，chunk 大小属于 versioned reader 策略并记录；不要暴露几十个默认用户参数。
- 首版 I/O/解包共用一个任务生命周期；以后分离 read/decode 时必须加入第二类队列和 staging credits 的计划。
- worker 创建失败返回 clean setup error；不能临时降 Q 或同步执行并仍报告原计划。

## 7. 现有 LTX 并发与对照的关键修正

当前 `run_streamed_block_stack()` 在 depth>1 时会为最初 depth 个 block 分别启动 loader thread，随后 refill 也创建线程。
所以现有 K=3 不能自动描述成 Q=1。新方案三槽单 worker 示例是独立布局候选，不能把它与旧三槽对比称为纯框架开销。

先记录旧路线的实际 initial/steady concurrency、D、P、retention、缓存与 command boundary，再选择：

1. 真正能对齐语义的候选进行同布局 A/B；persistent worker 对 per-fill pthread 的变化明确标为 executor 实现差异。
2. 无法对齐的组合标 `comparison_kind=strategy_change`，不能通过此成绩替代同布局 gate。
3. Q=1 与 Q=K 单独做调度策略实验；证明较少 CPU/I/O 竞争能否补偿 startup 延迟。

## 8. 默认与新路径的性能预算

| 位置 | 首发要求 |
|---|---|
| legacy 入口 | 一次 route 判断，不创建 descriptor/context，不扫 checkpoint |
| legacy kernel 内循环 | 新 hook/probe/锁/计时/等待/缓存清理=0 |
| new steady state | framework allocation/thread-create=0；固定数组/索引和小型状态转换 |
| new per-group dispatch | 不格式化字符串，不查询系统 memory/swap，不重算 digest |
| guard 检查 | 纳管分配前 reservation；owner 安全点观测，频率作为认证合同，不拿关 guard 的速度顶替 |
| report/trace | 常量 counters 默认；详细 trace 显式 campaign 启用、固定上限 |
| model kernel | 同 dtype、same input、same fusion policy；有差异单独申明并认证 |

“零动态分配”针对新 framework 热路径；不能把它扩大成 backend/driver 永不分配的未证实承诺。
首次初始化和文件 I/O 不能从 end-to-end wall 中删掉。CPU 开销、commit 数和 file bytes 要和 denoise 时间同时报告。

## 9. 与 strict memory schedule 的连接

slot content events 不改变 backing live intervals；只有真正 reserve/create/destroy 才改变资源账。
owner 的窗口开放 intent/compute 序列由模板决定，实际 I/O start/ready/completion arrival 单独作为 telemetry。
每次进入 j 的固定边界，按顺序为新进入窗口的 group 记录一次 eligibility intent，即使目标槽仍在使用；
此时不宣称任务已开始，也不创建新的 backing lease。实际派发等 slot/worker credits，就绪由 SlotSafetyTracker 验证。
因此 strict cursor 不需要按磁盘或 GPU 的真实完成顺序前进，不把 callback 到达顺序编码成严格 epoch。
新 codec 必须区分 `PREFETCH_ELIGIBLE` 与 telemetry `READ_STARTED`，不能让旧 v1 PREFETCH 的含义在新模型里悄悄变化。
需要 v2 event 时新增 versioned bridge，同时保留 v1 ABI；不得修改已有 v1 struct 大小并假称兼容。
模型外层 TEXT/UPSAMPLE/VAE/EXPORT 事件由 session/adapter 发一次，slot 事件由 executor 发一次。
F5 必须提供 event producer ownership 表和全请求 expected sequence，不以 denoiser terminal 代替整请求 terminal。

## 10. Cleanup 状态机

以下是目标合同。当前实现仍使用阻塞worker join和无deadline参数的adapter drain，stall timeout不等于安全清理的硬时限。
接service前必须补非协作I/O/driver阻塞时的生命周期和隔离策略，见 [14](14-layout-first-integration.md) 第7节。

```text
Running -> StopDispatch -> CancelIo -> JoinWorkers -> DrainGpu
        -> ConsumeCallbacks -> DestroyPools -> Unbind -> CleanFailed/Succeeded
                                             |
                         cannot prove drain -> Quarantined
```

取消不是立即 free。join 超时或 GPU 失败时保留被使用资源，worker 不再接新请求；由 service 走隔离/受控退出。
保持 primary failure 和 cleanup failure 分离；任何已发生的安全错误不能因为最后成功释放而变成 success。
每个 create 的第 n 次失败都要有 injection case，覆盖 prefix/slot/scratch/thread/callback registry 的部分初始化。

## 11. 完成标准

- 独立 fake GPU 在 fence 前持续读取 checksum，能发现提前覆盖；不只观察 executor 自报状态。
- K=1/2/3、D=0/K−1、Q=1/K、乱序 ready、延迟多 queue completion 均有可重现 trace。
- pass wrap、prefix compute、class barrier、stage handoff、warm request 顺序全部测试。
- 无 steady-state 线程创建/池扩容；默认路径 instrumentation 新增计数为 0。
- 必须通过 12 的真实 GPU 和性能 gate，才可获得 layout_validated；fake 成功仅代表 F2 完成。

## 12. Metal 语义核对依据

2026-09-16 核对 Apple Developer Documentation：`addCompletedHandler(_:)`、`Synchronizing CPU and GPU work`。
实现 reviewer 应按实际 SDK/command queue 版本复核；不要将不同 queue/API 版本的同步规则直接混用。
本设计只采用最小合同：CPU 写与 GPU 读不能冲突；completion 必须对应真正执行完的工作；多份 backing 用于 overlap。
这不证明任何具体 K 的性能收益，收益仍按 12 实测。

官方来源标识（便于后续实现复核）：

```text
https://developer.apple.com/documentation/metal/mtlcommandbuffer/addcompletedhandler(_:)
https://developer.apple.com/documentation/metal/synchronizing-cpu-and-gpu-work
```

## 13. 已实现的 cross-pass carry 数据面（2026-09-17）

`StageExecutor` 现在把 pass boundary 从隐式“总是清空”提升为 immutable plan 字段。实现仍复用同一套
`SlotSafetyTracker`、`IoExecutor`、completion mailbox 和 failure/quarantine 状态机，没有创建第二套 pager。

关键实现点：

1. `State::carry_group` 在 `begin()` 时复制一次 group metadata；热循环只修改 slot 标量，避免每 pass 复制
   `std::vector` 或重新构造 source span。
2. `State::carry_ticket` 只允许保存一个 Ready ticket。`quiescent_except_ready()` 同时验证 ticket identity、
   content state 和其余 slot 全部 Vacant。
3. pass 开始按 cyclic offset 旋转 group→slot 映射；incoming ticket 必须匹配新的 pass、step、group、slot 和
   content generation，随后才可进入 prepare/use。
4. pass 结尾只有在下一 pass 首组目标 slot Vacant 时才 dispatch carry；最后一组在 carry ticket 建立前不会
   encode，避免最后读者占用同一 slot 时提前覆盖。
5. `CAdapter::Job` 自持 blocks vector；异步 fill callback 不再引用临时 `Group` 的 blocks 指针。每个 slot 的
   job storage 预先存在，稳定态同尺寸 assignment 复用 capacity。
6. C ABI v3 只扩展 plan 的 `pass_transition`；adapter callback ABI 继续使用 v1。v1/v2 的 struct size、入口和
   reload 行为不变。

当前 begin-time fail-closed 条件为单 pool、K=2、每 group 一个 block、pass_count≥2、首组静态 slot=0，且
所有 group 都能放入两个 slot capacity。该限制是当前证据边界，不是框架长期上限。

已覆盖的 host/fake-backend 证据包括奇偶 suffix rotation、carry dispatch 早于上一 pass 最后一组 encode、
source fill/encode exactly once、取消、carry fill failure、错误 step、C ABI v3 和 pool 精确释放。fake callback
使用 reader fence，但仍不等于真实 H3 Metal command-buffer completion；真实 adapter 必须另外证明最后读者。
