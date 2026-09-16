# 03 · Slot / Scheduler 运行时协议

[目录](README.md) · [参数](02-configuration.md) · [模型 adapter](04-model-adapters.md)

状态：待实现接口草案。以下新类型与 C ABI 不代表已有代码；已有 `MemoryLedger`、`MemoryExecutionContext`、
`MemoryCompletionToken` 应复用，不能制造第二套相互独立的内存账。

## 1. 逻辑对象

```cpp
struct WorkItemKey {
    uint32_t stage, pass, step, group;  // descriptor 定义局部稳定 ID
};
struct SlotTicket {
    uint32_t pool, slot;
    uint64_t request_generation, content_generation;
    WorkItemKey item;
};
struct SlotBacking {
    StorageBundleHandle storage;      // 拟议 opaque backend bundle
    uint64_t capacity_bytes;
    OptionalBudgetLease lease;        // bounded 时由现有 ledger 管理
};
struct SlotUseFence {
    SlotTicket ticket;
    uint64_t queue_id, submission_sequence;
};
```

slot 是可复用 storage bundle，不要求一块连续 tensor 或一个 MTLBuffer。
layout class 包括每个 tensor 的 dtype/shape/alignment、packed scale、绑定布局；相同总 bytes 不代表可互换。
需要多类 pool 时 capacity 分别计算；首版单活动 pool，可在 layout-class barrier 释放旧池并建立新池，不偷偷扩容。

## 2. 两层生命周期

```text
Backing:
  Reserved -> Allocated -> Committed ----------------> Retiring -> Released
                              |                               ^
Content:                      v                               |
  Vacant -> Loading -> Ready -> InUse -> AwaitingFence -> Vacant
          (同一物理 backing 可重复上述 content cycle)
```

`Vacant` 表示内容可重写，不表示内存已归还。budget lease 从池初始化一直持有到物理 backing 销毁。
每次 refill 递增 content_generation，不重复 reserve/commit 一份同样的 storage。

已有 `MemoryCompletionToken` 用来结束 backing 的 pending-release；新增 `SlotUseFence` 只说明内容可复写，不能扣减 ledger。
把两个 token 合并会导致“池还在但账本归零”的错误。

reserved-but-not-allocated 和 pending-release 都占 guard 容量；warm retained pool 也不能隐藏在 baseline 之外。

## 3. Adapter operations（拟增内部 C ABI）

共同约定：size/version 前缀、opaque handles、fixed-size errors、回调不跨 C ABI 抛异常；不传 STL 类型。

| 操作 | 线程 | 合同 |
|---|---|---|
| describe | preflight owner | 读取 metadata，返回 items/layout/文件 range/envelope，不分配 GPU tensor |
| create_pool | owner | 按编译计划与 reservation 创建 storage bundle，不自行调整容量 |
| make_read_plan | owner | 冻结 fd identity、ranges、destination spans、unpack scratch |
| fill | bounded I/O worker | 只写 ticket 独占的已分配 storage；不访问 context/cursor，不创建模型资源 |
| accept_ready | owner | 验 item/generation/bytes/status，发布 Ready |
| prepare_group | owner | 在 Ready→InUse 前写已声明的 timestep/conditioning 派生数据，不任意扩容 |
| encode_group | owner | 使用 Ready content，执行原有 kernels，返回所有最后读取者的 fence |
| post_use_complete | GPU callback | 向有界 mailbox 投递 POD；不 emit semantic event，不释放 pool |
| drain_stage | owner | join I/O、证明 GPU 使用完成，消费 mailbox |
| destroy_pool | owner | 实际销毁，结束 ledger lease；失败要保留诊断身份 |

形状相关 workspace 由 adapter 通过 guard-aware allocation sites 获取，不能“在 compute 回调里任意分配再补报”。
layout-only 可不执全进程预算，但也必须遵守固定 pool 容量和错误传播；若 backend operation 不支持安全复用，descriptor 拒绝。

## 4. 所有权与并发

- Request owner 是唯一 slot 状态、cursor、epoch、lease 变更者。
- Worker 可以写指定 backing 内容与自己的 bounded result message；不能改变 content_generation 或 published state。
- 完成发布使用明确的 release/acquire 关系或 join，不能只靠 `ready=true` 的普通变量。
- GPU callback 不保留会悬空的 raw context；mailbox owner 生命周期覆盖全部 callback，解绑前 drain。
- worker stack、queue、read buffers、trace/mailbox 都有固定上限并进入 control-memory 估算。
- 一个 GPU request 持有一个 executor。现有 service GPU 串行化维持不变；多 worker process 的系统竞争仍需单独 admission。

首次创建 slot 在 owner 完成，worker 不在旧 epoch 并发调用 `try_reserve_site()`。
这是 LTX 初次 `load_block_weights()` 在后台分配与严格 owner scheduler 集成时的关键调整。

## 5. 调度伪代码

```text
resolve all groups and capacities
create exactly K backings; no resizing in loop
for pass in model_defined_passes:
    reset content identities as specified by plan
    for group j in compute order:
        enqueue eligible fills in [j, j+D], current demand first
        wait/consume READY for group j (completion arrival order may differ)
        emit planned bind/compute boundary
        encode group j using slot assigned by immutable plan
        register every queue's last reader as SlotUseFence
        consume completion mailbox at planned points
        reuse slot only when all its use fences have completed
drain I/O and GPU, release pool at stage's planned release boundary
```

首版每个 pass 使用相同 group 顺序；suffix slot 跨 step 保持 backing，但逻辑内容是否重读按 plan 固定。
不临时把上一步残留的最后几块当成缓存命中改变事件序列；未来优化需要新的 plan revision。

## 6. 三槽不是三个计算分支

K=3、G=1、D=2、Q=1 示例：

```text
slot0: load B0 -> GPU B0 -> fence -> load B3 -> GPU B3
slot1:     load B1 -----------> GPU B1 -> fence -> load B4
slot2:          load B2 -----------------> GPU B2 -> fence
```

图仅表示依赖，不是实测时间。I/O/GPU 能否 overlap 取决于 shared bandwidth、解包成本、cache 和真实 fence。
计算依然 B0→B1→B2→B3；不因多槽改变 attention/conditioning/sampler 顺序。

D=0 即使 K=3 也只 just-in-time 使用固定池；布局合法但可能浪费内存，plan 可给 warning，不擅自减 K。
若当前 group 只有一个 slot 可用，不允许提前写入未完成读取的 slot 来“避免等待”。

## 7. Schedule 与 telemetry 分离

strict semantic sequence 表达 owner 的逻辑操作顺序；read/fence 实际完成时间是 telemetry，不驱动 epoch。
乱序 ready message 按 ticket 存入固定记录，owner 依编译 sequence 消费。每个可分配 site 在执行前都已进入合法 epoch。
窗口开放的逻辑 intent 和实际 I/O start 分离，不能让异步调度时序决定 strict memory cursor；详见 [10](10-executor-implementation.md) 第9节。

已有 schedule v1 无 pass/group/content generation；有唯一编码时用明确映射，不把临时序号塞进 branch 或 tile。
无法无歧义编码 LTX 多 pass、group refill 时，新增 versioned v2 codec，包含 stage/pass/step/group/operation/slot。
禁止 stage BEGIN 和 step BEGIN 映射到同一个 key。content_generation 用于运行时安全，可与静态 pass/item identity 分开。

框架生成的 slot 事件只证明框架动作，不能自证 kernel 已完成：adapter 返回的真实 fences、独立 fake-backend access log
以及 backend allocation-site audit 共同验证。模型负责准确发 text/latent handoff/VAE/export 的外层 semantic boundary。

新 layout-only context 也校验 slot protocol；bounded 时再把同一 semantic event 交给现有 strict memory cursor。
一个事件只有一个 producer，不允许框架和模型 wrapper 重复 emit。

## 8. Offload 的实际动作

不可变 weight 完成读取后复用 slot，下次从 checkpoint 重读，不把同样的 weight 再写回磁盘。
shared-memory backend 的 upload 可以是 bind/visibility 阶段，不强造 H2D 副本；private destination 则把 staging 和 copy 同时计入。

activation、latent、conditioning 是有状态数据：除非已经无后续消费者，否则不能直接丢弃或从 weight 文件恢复。
step-derived weight table 同样需要明确 owner 写入阶段与生命周期，不能因其位于 weights 结构就假定完全不可变。
第一版不实现隐式 activation spill/SSD 写回，不能把这类数据套进 weight pager。

`erase_prefix()`/`clear_cache()`/slot Vacant 都不等于实际释放；MLX graph 引用和 GPU readers 必须结束。
VAE tiling 管 activation/output，decoder weights 通常在 VAE 阶段复用，不默认每 tile 重载。

## 9. 错误、取消与隔离

| 场景 | 处理 |
|---|---|
| reservation/allocate 失败 | 不填 READY，回滚已分配 pool；返回具体 site/capacity |
| short read / EOF / layout mismatch | 停止派发；清理 partial fill，不能继续旧内容 |
| I/O cancel | chunk 边界检查取消；join 后再释放目标 buffer |
| GPU fence error / late completion | sticky failure；停止新 compute，不因超时直接标已完成 |
| duplicate / stale generation | 记录拒绝并失败，不把旧 completion 用于新内容 |
| mailbox overflow | hard failure；不得丢弃后继续运行 |
| drain 无法证明 | quarantine，不能 free in-flight storage 或接新请求 |
| telemetry buffer 满 | 声明数据不完整；认证 run 失败，不把丢事件当成零开销 |

固定 deadline 控制等待，但 deadline 不赋予提前释放权。无活动 GPU 的 clean failure 可回收，未证明 clean 的 worker 不复用。
后台 worker 不直接 emit error schedule；owner 收到失败后保存 primary error，cleanup error 单独输出。

## 10. 不支持请求内改布局

Tight 可暂停未启动的未来 I/O，只要不改变已编译 key 顺序、live intervals 和 slot mapping；否则候选不支持这种 suppression。
Critical 或 guard 越界立即停止新派发，在安全边界清理并失败。不能短时回落后把失败改为成功。
要减少 K/G/P 或选择 resident，必须新请求重新 resolve/admit。
