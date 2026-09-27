# 22 · 当前 Slot/Streaming 框架使用与实现指南

[目录](README.md) · [配置规范](02-configuration.md) · [运行时协议](03-runtime-protocol.md) ·
[实现进度](13-implementation-progress.md) · [验收手册](12-acceptance-playbook.md)

修订日期：2026-09-17。

App 按内存目标选择后台 preset 的下一步设计见 [23 产品与 Public 接线](23-public-memory-tier-presets.md) 和
[24 分模型探索与验收](24-memory-tier-exploration-and-acceptance.md)；进一步代码接缝与模型算法见
[25 实施规格](25-public-preset-implementation-spec.md)，采样工具/测试矩阵/发布见
[26 验收与发布](26-public-preset-acceptance-and-release.md)。这些是待实施方案，不改变本文当前 private/candidate 状态。

本文集中介绍 TurboCider 当前已经实现的通用 block/slot streaming 框架，包括使用方式、配置语义、
布局编译、slot 调度、模型 adapter、错误处理、性能证据和发布边界。本文是当前实现的总览；字段冲突以
[02](02-configuration.md) 为准，底层状态机以 [03](03-runtime-protocol.md) 和
[10](10-executor-implementation.md) 为准，最新实测数字以 [13](13-implementation-progress.md) 为准。

## 1. 当前结论和适用范围

TurboCider 已经把原来分散在模型内部的 block streaming 能力抽象为统一框架：

```text
用户请求 / profile / 冻结 preset
              |
              v
      StreamingConfig
              |
              v
模型 ResourceDescriptor ---- checkpoint/source identity
              |
              v
        LayoutCompiler
              |
              v
       immutable Layout
              |
              v
 StageExecutor + SlotPool + bounded I/O
              |
              v
     ModelSlotAdapter
              |
              v
  Metal / MLX model kernels
```

当前已经有真实 adapter 和真实 GPU 证据的范围是：

| 模型 | 当前范围 | 状态 |
|---|---|---|
| LTX | LTX-2.5 当前冻结 checkpoint、原生 C/Metal 路径 | private candidate，冻结同布局 P1 已通过 |
| Z-Image | Z-Image Turbo、BF16、MLX block 路径 | private candidate，冻结同布局 P1 已通过 |
| H3 | MiniMax H3 Turbo original BF16 | private candidate，工程验收完成；严格统计仍为 `INCONCLUSIVE` |
| Flux | Flux.2 Klein 9B、Diffusers 双分片 BF16、eager GPU、无 LoRA | private candidate，冻结同布局 P1 已通过 |

这里的“已支持”表示 adapter、统一 executor、真实权重、真实 GPU、质量比较和指定 tuple 的性能证据已经存在；
不表示 public production registry 已开放。当前 production registry 仍为空，普通 H3、H3 量化变体、Flux 4B
compiled graph、LoRA、GPU+ANE 和任意 shape/checkpoint 不自动获得资格。

新框架是显式 opt-in add-on。没有开启 `execution.streaming.enabled=true` 时：

- 默认 resident 路径保持不变；
- 现有 legacy streamed 路径保持不变；
- 不创建通用 descriptor、layout、executor、slot pool 或 I/O worker；
- 不因为存在新框架而自动改变 residency、slot 数或预算。

## 2. 为什么使用 block/slot streaming

模型全驻留时，所有权重长期留在 GPU 可见内存中，运行速度通常最好，但大模型可能挤压 activation、VAE、
text encoder、driver cache 和媒体缓冲，最终触发分配失败或系统 swap。

Slot streaming 不等待操作系统被动换页，而是显式控制权重生命周期：

```text
从 checkpoint 读取下一组权重
          |
          v
写入固定 GPU-visible slot
          |
          v
绑定到当前 block 并执行
          |
          v
等待所有 GPU reader 完成
          |
          v
复用同一个 slot backing
```

它的主要收益是：

- 权重峰值由“完整模型权重”下降为“resident 权重 + 若干 slot”；
- backing 数量和容量在布局编译后确定；
- 可以让 SSD/文件读取与 GPU 计算重叠；
- 不依赖系统随机选择哪些页面进入 swap；
- slot 复用受 generation 和 GPU last-reader fence 保护。

代价是权重需要重复读取。模型计算时间不足以隐藏 I/O 时，streaming 一定可能慢于 resident。因此框架分别验收：

1. 默认路径是否回退；
2. 同一 streaming 布局下通用框架是否增加明显开销；
3. streaming 相对 resident 节省多少内存、损失多少速度；
4. 真实低内存和 swap 条件下是否优于系统换页。

当前第 1、2 项已有部分或正式证据，第 3 项已有 Flux 实测，第 4 项尚未完成正式 P3 campaign。

## 3. 核心术语

### 3.1 Block

Block 是模型 adapter 定义的最小有序计算单元，例如 transformer block。框架不会改变 block 的数学实现，
也不会通过 streaming 跳过 block。

### 3.2 Group

Group 是一次 fill 进入 slot 的连续 block 集合。`block_group_size=G` 表示每组最多包含 G 个 block。

```text
N=10, P=2, G=3

resident prefix: block 0, 1
group 0:         block 2, 3, 4
group 1:         block 5, 6, 7
group 2:         block 8, 9
```

Group 不能跨 layout class 边界。G 只改变加载和驻留粒度，不表示 kernel fusion，也不改变 block 的计算顺序。
当前四个已验收模型 tuple 都使用 `G1`。

### 3.3 Slot

Slot 是预先分配、可循环复用的一份物理 backing。slot 中的内容会变化，但 backing 地址在正常执行期间保持稳定。

`slot_count=K` 是每个 pool 的精确槽数，不是上限。增加 K 通常可以提高 lookahead 能力，但也线性增加 slot backing、
转换 scratch 和部分绑定对象的占用。

### 3.4 Pool

同一 layout class 的 slot 构成一个 pool。不同 class 的 field 数量、shape 或 capacity 不同，不能假定共享相同 backing。

例如 Flux 有两类 block：

```text
dual block class   -> dual K2 pool
single block class -> single K2 pool
```

因此 Flux 的 K2 是“每个 pool 两槽”，总计四个 slot bundle。

### 3.5 Stage、Pass 和 Step

- stage 是 adapter 暴露的执行阶段，例如 `denoiser`；
- pass 表示同一有序 group 列表的一次完整消费；
- step 是模型/采样器对外的步编号；
- pass 和 step 会同时写入 slot ticket，不能互相替代。

### 3.6 Reader

Reader 表示最后仍可能读取某个 slot 内容的 GPU queue/fence。一个 slot 可以有多个 reader；只有所有已声明 reader
都完成后，slot 才能重新进入 `Vacant`。

## 4. 布局参数 P/G/K/D/Q

当前手动布局由五个核心参数组成：

| 符号 | 配置字段 | 含义 | 主要影响 |
|---|---|---|---|
| P | `resident_prefix_blocks` | 开头常驻的 block 数 | 增大 P 可减少每 pass I/O，但增加常驻权重 |
| G | `block_group_size` | 每次 fill 的 block 数 | 增大 G 可减少调度次数，但增加单 slot 容量和等待粒度 |
| K | `slot_count` | 每个 pool 的物理 slot 数 | 增大 K 可增加流水深度，但增加 backing 内存 |
| D | `prefetch_distance` | 相对当前需求可提前派发的最远 group | 决定 lookahead 窗口，不直接等于线程数 |
| Q | `io_workers` | 持久 fill worker 数 | 决定并发读取上限，增加 Q 不保证磁盘更快 |

基础约束：

```text
G >= 1
K >= 1
0 <= D < K
1 <= Q <= K
0 <= P < block_count       # streamed stage
K <= 每个 pool 对应的 group 数
```

Adapter 可以增加更严格限制。例如当前 `carry_first_group` 只认证 single-pool、K2、G1；某个模型也可以只接受
一个冻结 P/K/D/Q 组合。

### 4.1 D 和 Q 的区别

`D` 控制允许发出多少未来工作，`Q` 控制多少 worker 能同时执行 fill。

```text
K3 / D2 / Q1
```

表示可以持有三份不同 group 内容，也允许当前 group 之外再看两组，但实际只有一个线程顺序读取。

```text
K3 / D2 / Q3
```

才允许最多三个 fill worker 同时工作。是否真正并行还取决于文件、SSD、page cache 和 adapter 的读取方式。

### 4.2 为什么不只暴露 slot_count

仅设置 K 无法唯一决定调度行为：

- K3/D0 仍可能接近逐块同步加载；
- K2/D1/Q1 可以形成典型双缓冲；
- K3/D2/Q1 是深缓冲、单读取线程；
- K3/D2/Q3 是深缓冲、多读取线程；
- 相同 K 在不同 P、G、pool policy 下内存占用不同。

普通 UI 可以只向用户展示“低内存/平衡/高吞吐”等 preset，但 preset 最终必须展开为完整 P/G/K/D/Q，
resolved report 也必须显示完整实际布局。

## 5. 配置和使用方式

### 5.1 手动 streaming 配置

当前 schema 使用 request v2 下的 `execution.streaming`：

```json
{
  "execution": {
    "policy": "gpu",
    "streaming": {
      "schema_version": 1,
      "enabled": true,
      "selection": "manual",
      "retention": "request",
      "stages": {
        "denoiser": {
          "residency": "streamed",
          "block_group_size": 1,
          "slot_count": 3,
          "resident_prefix_blocks": 8,
          "prefetch_distance": 2,
          "io_workers": 3
        }
      }
    }
  }
}
```

当前字段约束：

- `schema_version` 必须为 `1`；
- `selection` 当前只接受 `manual`；
- `retention` 当前只接受 `request`；
- streamed stage 必须完整提供 P/G/K/D/Q；
- 不认识的 stage、字段或 schema 必须拒绝；
- runtime 不根据实时空闲内存自动改变显式参数。

上述结构已经进入 parser、profile merge 和 plan/report；模型是否允许真实执行还要通过 private/production
execution registry 和 adapter capability 检查。

### 5.2 显式 resident

框架配置可以显式描述 resident stage：

```json
{
  "execution": {
    "policy": "gpu",
    "streaming": {
      "schema_version": 1,
      "enabled": true,
      "selection": "manual",
      "retention": "request",
      "stages": {
        "denoiser": {
          "residency": "resident"
        }
      }
    }
  }
}
```

Resident stage 不接受 P/G/K/D/Q。需要最高兼容性和现有 warm-cache 行为时，推荐完全不设置新 streaming，
继续使用原默认路径；显式框架 resident 是独立生命周期，不自动继承旧默认路径的发布资格。

### 5.3 与 memory-constrained guard 组合

Slot 布局和内存上限是两个正交功能：

```json
{
  "execution": {
    "policy": "gpu",
    "streaming": {
      "schema_version": 1,
      "enabled": true,
      "selection": "manual",
      "retention": "request",
      "stages": {
        "denoiser": {
          "residency": "streamed",
          "block_group_size": 1,
          "slot_count": 3,
          "resident_prefix_blocks": 8,
          "prefetch_distance": 2,
          "io_workers": 3
        }
      }
    },
    "memory_constrained": {
      "enabled": true,
      "limit_bytes": 21474836480,
      "buffer_percent": 15,
      "min_free_bytes": 2147483648
    }
  }
}
```

这里用户上限 Y 为 20 GiB，`buffer_percent=15` 后的主要可用预算约为 17 GiB，同时还要求保留至少 2 GiB。

Guard 不会把 K3 自动改成 K2。正确行为是：

```text
编译用户指定布局
      |
计算已知资源需求
      |
需求 <= admission budget ?
      | yes                 | no
      v                     v
继续执行               返回缺口和失败阶段
```

当前 whole-request resource closure 尚未覆盖所有 MLX、MPSGraph、Metal、driver、媒体和 host allocation，
因此不能把这个示例解释为已经获得“进程绝不超过 Y”或“绝对零 swap”的 production 保证。

### 5.4 关闭和覆盖规则

- 不提供 `execution.streaming`：完全沿用旧行为；
- `streaming.enabled=false`：关闭新框架，不会把旧 streamed 强制改为 resident；
- request 显式改变 residency：该 stage 整体替换 profile 中旧 stage 配置；
- residency 不变：各字段按 presence overlay；
- K 从 3 改成 1、却继承旧 D=2：必须拒绝，不自动修正为 D=0；
- 新 manual streaming 与用户显式 legacy residency/budget/offload 权威冲突：必须返回 config conflict；
- profile、preset 和 request 的最终 resolved 值及字段来源必须进入报告。

合并顺序是：

```text
descriptor fixed policy
        < selected model profile
        < explicit request fields
```

### 5.5 当前可参考的模型配置

仓库中的以下文件给出了当前冻结 tuple 的完整请求或性能 policy：

| 模型 | 示例 | 用途 |
|---|---|---|
| LTX | [ltx-p1-same-layout-policy.json](examples/ltx-p1-same-layout-policy.json) | P8/G1/K3/D2/Q3 candidate 与 legacy 同布局对照 |
| Z-Image Turbo | [z-image-p1-same-layout-policy.json](examples/z-image-p1-same-layout-policy.json) | P14/G1/K2/D0/Q1 candidate 与专用 streaming 对照 |
| H3 Turbo | [h3-p1-same-layout-policy.json](examples/h3-p1-same-layout-policy.json) | original BF16 P0/G1/K2/D1/Q1 probe 对照 |
| Flux.2 Klein 9B | [flux9-k2-q2-request.json](examples/flux9-k2-q2-request.json) | P0/G1/K2/D1/Q2 private candidate 请求 |
| Flux.2 Klein 9B | [flux9-p1-same-layout-policy.json](examples/flux9-p1-same-layout-policy.json) | direct replay 与 generic executor 正式 P1 对照 |

这些文件用于开发和验收，不代表 public constructor 已经放行。H3 当前通过专用 probe 和
`TURBOCIDER_H3_EXACT_STREAM=1` 进入 candidate；不能把该环境变量当成普通用户 API。LTX、Z-Image 和 Flux
示例中的 `constructor=candidate` 同样是受限资格入口。

## 6. LayoutCompiler 的职责

模型 adapter 生成 `Descriptor`，至少描述：

- model、backend 和 checkpoint identity；
- source artifact 及 tensor range；
- resident field；
- block field、shape、dtype、alignment；
- materialization/conversion；
- layout class；
- pass count 和 workload identity；
- adapter 支持的 P/G/K 范围和固定策略。

`compile_layout()` 随后生成不可变 `Layout/StageLayout`：

```text
resident prefix
ordered groups
group -> pool mapping
group -> physical slot mapping
per-field slot capacity
pool capacity
peak pool bytes
logical source bytes per pass
pass transition
multi-pool policy
canonical layout digest
```

Compiler 是纯 metadata 操作：

- 不访问 GPU；
- 不启动 worker；
- 不探测当前 RAM；
- 不自动调整用户参数；
- 不把权重布局错误地宣传为 whole-request memory upper。

同一个 descriptor 和同一组配置必须产生确定性的 canonical layout 和 digest。实际执行报告中的 layout digest
必须与 resolved plan 一致，否则属于 lifecycle/configuration failure。

## 7. Slot 状态机和安全身份

每个 slot 使用以下状态机：

```text
Vacant
  | begin_fill
  v
Loading
  | exact fill completion
  v
Ready
  | begin_use
  v
InUse
  | seal all last readers
  v
AwaitingFence
  | every reader completed
  v
Vacant
```

每次 fill 产生不可复用的 ticket：

```text
pool
slot
request_generation
content_generation
stage
pass
step
group
```

状态机保证：

- 只有 `Vacant` slot 可以开始 fill；
- fill 字节数必须精确等于 layout 预期；
- 只有 `Ready` 内容可以绑定和计算；
- encode 后必须声明所有 last-reader fence；
- 所有 reader 完成前不能覆盖 slot；
- stale、重复、跨 request/pass/step/group completion 会 poison tracker；
- owner 线程之外不能直接修改 slot authority。

SlotSafetyTracker 只追踪内容生命周期，不负责释放物理 backing。backing 由 adapter/pool owner 持有；如果 drain
无法证明 GPU、callback 和 worker 已经退出，executor 必须保留完整对象进入 quarantine，而不是提前 free。

## 8. StageExecutor 调度算法

### 8.1 Owner-pump

每个 stage 只有一个 owner 线程修改调度状态。I/O worker 和 GPU callback 只向有界 completion mailbox 投递事件。

简化算法：

```text
next = 当前尚未提交 compute 的 group
dispatch = 下一个尚未派发 fill 的 group

while stage/pool 尚未完成:
    consume fill completions
    consume reader completions
    retire 已完成 reader 的 slot

    while dispatch <= next + D:
        如果目标 slot 不是 Vacant: break
        begin_fill
        enqueue bounded I/O job
        dispatch++

    如果 prefix 尚未提交:
        encode_prefix

    如果 group[next] 已 Ready:
        prepare_group / bind
        begin_use
        encode_group
        seal_readers
        next++

    如果本轮没有进展:
        等待 mailbox 或 stall timeout
```

`next` 是下一个未提交计算的 group，不是 GPU 已完成的位置。Compute 必须按顺序提交；fill completion 可以乱序，
但未来 group Ready 不能越过当前 group 执行。

### 8.2 K2/D1 双缓冲

```text
时间 ---------------------------------------------------->

slot 0: load G0 -> compute G0 ------------> load G2 -> compute G2
slot 1:          load G1 -> compute G1 ------------> load G3 -> compute G3

                  GPU compute
                  与下一 slot fill 重叠
```

复用 slot 0 前必须等待 G0 的所有 reader 完成，而不只是等待 `encode_group()` 返回。

### 8.3 K3/D2 深流水

```text
initial window: G0, G1, G2

slot 0 <- G0 -> G3 -> G6 ...
slot 1 <- G1 -> G4 -> G7 ...
slot 2 <- G2 -> G5 -> G8 ...
```

K3/D2 允许更深 lookahead，适合 LTX 这类 block compute 足以覆盖读取、且 adapter 已认证三槽容量的模型。
它并不意味着 K3 一定快于 K2；如果 SSD 已饱和或额外内存挤压 activation，K3 可能没有收益。

### 8.4 D0 同步 backend 的 claim-overlap

按普通规则，D0 只允许派发当前 group。Z-Image 的同步 `mx::eval` 会使简单实现退化为 load/compute 串行。

框架提供 adapter opt-in 的 `overlap_next_fill_after_claim()`：

```text
当前 group Ready
      |
begin_use，当前内容已被安全 claim
      |
如果另一个 slot Vacant，派发下一 fill
      |
进入同步 encode/mx::eval
```

这样仍保持 startup 只发一个 fill，同时恢复 K2 稳态 overlap。该优化默认关闭，只有明确需要并通过验证的
同步 adapter 才能启用。

### 8.5 同步和异步 reader

原生 Metal adapter 通常返回真实异步 command-buffer reader fence，由 callback 稍后写入 mailbox。

MLX adapter 在同步 `mx::eval` 返回时设备读取已经完成，可以返回 `ReaderSet::already_complete=true`。Executor
仍执行 ticket 匹配、`begin_use` 和 `seal_readers`，随后直接退休 reader，省去一次无意义的 mailbox round trip。

## 9. I/O 执行器和 Pager

`IoExecutor` 在 stage setup 时创建固定 Q 个 worker：

- worker 数在 steady state 不变化；
- job queue 和 completion mailbox 都是有界结构；
- enqueue 失败是 backpressure/合同错误，不允许静默丢任务；
- worker 只执行读取和填充，不改变 slot 状态；
- cancel 是协作式的；
- finish/失败路径必须 join worker。

MLX 模型共用 `MlxWeightPager`：

- owner 线程创建 MLX shared arrays；
- worker 只做 ranged `pread` 和受控填充；
- 支持多个 safetensors artifact；
- 支持 resident field、pool fill、bind、短读和取消；
- 检查 snapshot/path identity，避免使用明显不一致的 source；
- 当前 Flux/Z-Image 认证主要集中在 BF16、G1。

读取粒度属于 adapter 性能合同。H3 曾将大矩阵切成 8 MiB 小 `pread`，导致 wall ratio 退化到 `1.14662`、
denoise ratio 退化到 `1.22061`；恢复为每个大矩阵一次大 `pread` 后回到约 1.1% 波动。这说明通用调度正确
并不自动保证 I/O 实现高效。

## 10. Pass transition

### 10.1 `reload`

默认策略。每个 pass 完整消费所有 group，边界 drain 到所有 slot `Vacant`，下一 pass 从 group 0 重新开始。

适用当前 LTX、Z-Image 和 Flux tuple。

### 10.2 `carry_first_group`

用于减少多 pass 模型边界气泡：

```text
pass N 尾部:
    fill pass N+1 / group 0
    compute pass N / last group

pass boundary:
    只允许一个 Ready carry ticket
    其他 slot 必须 Vacant

pass N+1:
    直接消费已经 Ready 的 group 0
```

它不是 cache hit：每个 pass 仍然对每个 group 执行精确一次 fill。当前实现要求 single pool、K2、G1，并按照：

```text
(pass * group_count + group_index) mod K
```

旋转物理 slot，避免跨 pass 时固定映射与 carry slot 冲突。MiniMax H3 Turbo 使用该策略。

## 11. Multi-pool 策略

### 11.1 `serial`

```text
drain pool A
destroy pool A
create pool B
execute class B
```

优点是峰值只需要一个 class 的 pool；缺点是 class/pass 切换可能产生 steady-state allocation 和额外同步。

### 11.2 `retain_all`

```text
setup:
    create pool A
    create pool B

runtime:
    select A -> select B -> select A -> select B

finish:
    drain and destroy all pools
```

优点是稳态不重建 pool；代价是峰值为所有 retained pool capacity 之和。Flux 使用 `retain_all`，创建 dual K2
和 single K2 两个 pool，共四个 slot bundle。

Multi-pool policy 是 compiled layout identity 的一部分，adapter 不能在运行时静默改用另一种策略。

## 12. ModelSlotAdapter 接口

所有模型通过同一个 `ModelSlotAdapter` 接口连接 executor，主要职责如下：

| 接口 | Adapter 责任 |
|---|---|
| `supports_multi_pool_policy` | 明确声明是否支持 `retain_all` |
| `create_pool` | 按编译容量创建稳定 backing 和绑定对象 |
| `select_pool` | 在 ordered class barrier 切换 active pool |
| `make_fill_job` | 生成不依赖临时对象的固定 fill job |
| `encode_prefix` | 执行 resident prefix |
| `prepare_group` | 将 ticket 对应内容绑定给当前 block/group |
| `overlap_next_fill_after_claim` | 可选启用 D0 同步 backend overlap |
| `encode_group` | 提交计算并返回所有 last reader |
| `drain` | 等待 prefix、GPU reader、callback 和 adapter 工作退出 |
| `destroy_pool` | 只在已证明安全后释放 backing |

Adapter 不应：

- 自己维护第二套 slot authority；
- 在 steady refill 中重新解析整个 checkpoint；
- 在 worker 中创建/销毁模型 pool；
- 隐式改变 group、slot 或 pass 顺序；
- 在 GPU reader 未完成时复写或释放 backing；
- 用全设备同步掩盖缺失的 reader 生命周期，除非该同步本身就是已验收 backend 语义。

## 13. 各模型当前布局和调度

### 13.1 LTX

冻结 tuple：

```text
P8 / G1 / K3 / D2 / Q3
48 total blocks
8 resident prefix
40 streamed groups
11 passes
reload
single pool
440 fills/request
```

特点：

- 原生 C/Metal adapter，经版本化 C bridge 接统一 executor；
- 三槽轮转，D2 允许当前、下一组、下下组进入 dispatch window；
- 真实 Metal reader fence；
- Stage 1、upsample、Stage 2 边界保持原模型语义；
- setup 创建 3 workers、1 pool；
- setup 后 steady framework allocation/thread-create 为 `0/0`。

### 13.2 Z-Image Turbo

冻结 tuple：

```text
P14 / G1 / K2 / D0 / Q1
30 total blocks
14 resident prefix
16 streamed groups
1 pass
reload
single pool
16 fills/request
```

特点：

- 每个 streamed block 有 13 个 BF16 tensor；
- 两个 MLX shared slot；
- 复用已有 ranged `pread`、`Weights::bind_arrays` 和 `z_block` 数学实现；
- 使用 claim-overlap 恢复 K2/D0 overlap；
- 同步 `mx::eval` 使用 already-complete reader 快路径；
- setup 创建 1 worker、1 pool；稳态 allocation/thread-create 为 `0/0`。

### 13.3 MiniMax H3 Turbo

冻结 tuple：

```text
original BF16 only
P0 / G1 / K2 / D1 / Q1
50 streamed blocks
4 passes
carry_first_group
single pool
200 fills/request
```

这里的 P0 表示 resident prefix 为零，不是性能等级 P0。

每个 block streaming：

- QKV；
- attention output；
- FC1；
- FC2。

Norm、AdaLN、conditioning 和 workspace 保持常驻。两个 Metal shared slot 通过真实 command-buffer callback
完成 reader retirement；pass N 尾部预取 pass N+1 的 block 0。Exact 模式开启后 legacy slot authority 明确失效，
避免两套调度器同时控制 backing。

### 13.4 Flux.2 Klein 9B

冻结 tuple：

```text
Diffusers two-shard BF16
eager GPU
no LoRA

P0 / G1 / K2 / D1 / Q2
8 dual blocks
24 single blocks
2 passes
reload
retain_all
64 fills/request
```

布局：

```text
9 fixed tensors resident

dual class:
    8 groups
    16 fields/group
    K2 pool

single class:
    24 groups
    4 fields/group
    K2 pool
```

两个 pool 在 setup 时一次创建，dual→single→下一 pass dual 只切换 active pool，不重建 backing。Q2 worker
可并行执行 ranged fill；同步 `mx::eval` 使用 already-complete reader。

## 14. 当前性能表现

### 14.1 如何解释性能数字

必须区分三种比较：

| 比较 | 回答的问题 |
|---|---|
| 默认路径 before/after | 新代码是否拖慢未开启 streaming 的用户 |
| 相同布局 direct/legacy vs generic | 通用框架抽象本身是否增加开销 |
| resident vs streaming | 用 SSD I/O 换内存后，整体策略代价是多少 |

不能用 streaming 相对 resident 的慢，证明 executor 抽象很慢；也不能用 generic 相对旧专用实现略快，证明
streaming 会比 resident 更快。

正式 P1 当前采用固定 10 个 ABBA block、20 matched pairs、40 measured requests，并检查：

- wall median ratio 上限 1.02；
- wall P95 ratio 上限 1.05；
- denoise median ratio 上限 1.02；
- 输出逐对一致；
- fault 为零；
- semantic layout 完全匹配；
- steady framework allocation/thread-create 为零。

### 14.2 同布局通用框架结果

| 模型 | baseline | wall median ratio | wall P95 ratio | denoise median ratio | 质量 | 结论 |
|---|---|---:|---:|---:|---|---|
| LTX | legacy streamed | 0.98810 | 0.98953 | 0.97249 | 20/20 Stage-2 BF16 byte-exact | P1 PASS |
| Z-Image Turbo | 专用 streaming | 0.99986 | 0.99913 | 1.00313 | 20/20 PNG byte-exact | P1 PASS |
| H3 Turbo | legacy K2/G1 | 1.01066 | 0.99511 | 1.01185 | 20/20 artifact byte-exact | 工程验收完成；bootstrap `INCONCLUSIVE` |
| Flux.2 Klein 9B | direct same-layout replay | 0.99971 | 1.00325 | 0.99816 | 20/20 PNG byte-exact | P1 PASS |

对应绝对数：

| 模型 | baseline wall median | generic wall median | baseline denoise median | generic denoise median |
|---|---:|---:|---:|---:|
| LTX | 15.318877 s | 15.136558 s | 7.652459 s | 7.441930 s |
| H3 Turbo | 22.023271 s | 22.258101 s | 15.284899 s | 15.466089 s |
| Flux.2 Klein 9B | 1.946221 s | 1.945657 s | 1.336996 s | 1.334537 s |

Z-Image 当前提交中的正式摘要以 ratio、bootstrap 区间和 matched-pair 质量结论为主，因此不在上表混入来自
不同阶段 bundle 的绝对时间。

结论是：在当前冻结 tuple 上，统一 StageExecutor 没有观察到明显的框架级性能回退。H3 median 约慢 1.1%，
属于已接受的 I/O 波动范围，但严格 bootstrap 区间跨过门槛，因此不能写成统计 P1 PASS。

### 14.3 Flux resident 与 streaming 的策略成本

Flux 当前有一组直接展示内存/速度交换的结果：

| 路径 | wall | denoise | MLX peak |
|---|---:|---:|---:|
| resident | 1.338198 s | 0.859606 s | 18,303,578,036 bytes |
| generic streaming | 1.951123 s | 1.383473 s | 11,693,804,356 bytes |

Streaming：

- MLX peak 减少 6,609,773,680 bytes，约 6.15 GiB；
- wall 增加约 46%；
- denoise 增加约 61%；
- 输出保持 byte-exact。

这说明该 workload 中 resident 明显更快，而 streaming 显著降低了权重相关峰值。它是策略选择，不是框架失败；
同样的 streaming 布局下，generic/direct ratio 为 0.99971，说明 executor 抽象本身接近零额外成本。

### 14.4 默认路径保护

当前设计通过入口隔离保护默认路径：未启用新 streaming 时不构造通用对象。Flux 默认 resident 审计观察到
新框架相关计数为零，小型 ABBA 对照也未观察到回退。

但是仍需保持准确表述：

- 部分模型只有 tiny/default 审计，不等于所有 normal-target 已完成 production P0；
- 任何改变默认入口、planning 次数、allocator、kernel 或 session lifecycle 的提交都必须重跑 P0；
- streaming 的正向收益不能补偿默认 resident 的回退。

### 14.5 尚未证明比 swap 更快

当前尚未完成正式低内存 P3：

- 没有在受控 memory pressure 下比较 resident+swap 和 explicit streaming；
- 没有完整记录 swap-in/swap-out、memory pressure、page fault、GPU idle 和 SSD traffic；
- 没有证明所有机器、SSD 和 workload 上 streaming 都更快；
- 没有 whole-request RSS hard cap 证据。

理论上 explicit streaming 更可预测，因为它读取已知 tensor range、复用固定 backing，并能与 GPU compute overlap；
系统 swap 则可能换出 activation、runtime metadata 或无关进程页面。但该判断必须由 P3 campaign 验证，不能直接作为发布结论。

## 15. 可观测性和运行报告

一次真实执行至少应报告：

```text
requested streaming config
field provenance
resolved P/G/K/D/Q
group count and slot assignment
pool count and pool policy
layout digest
adapter/backend/checkpoint identity
actual semantic layout
actual slot fills
bytes loaded
pool creates and slot bundles
I/O wait seconds
steady framework allocations
steady framework thread creates
execution implementation identity
eligibility and enforcement level
failure stage / rejection code
```

关键 counter 解释：

| Counter | 含义 |
|---|---|
| `pool_creates` | request setup/切换期间实际创建的 pool 数 |
| `slot_bundles` | 所有 pool 中创建的物理 slot 总数 |
| `fills` | 成功完成的 group fill 数 |
| `bytes_loaded` | 逻辑 source bytes；不等于 SSD 物理流量 |
| `groups_submitted` | 已提交 compute 的 group 数 |
| `wait_seconds` | owner 无进展时等待 completion 的时间 |
| `steady_framework_allocations` | setup 完成后框架新增 allocation 数 |
| `steady_framework_thread_creates` | setup 完成后新增线程数 |

`actual` 与 `resolved` 不同不是自动优化，而是错误。Campaign verifier 必须同时检查 semantic layout、implementation
identity、build identity、输出质量和环境信息，避免 baseline/candidate 意外走同一路径仍被误报为 PASS。

## 16. 错误、取消和资源释放

框架采用 fail-closed：

- fill 短读、字节数错误或 artifact identity 失效：失败；
- completion mailbox overflow：失败；
- stale/duplicate ticket 或 reader：poison；
- owner 线程错误：拒绝；
- stall 超时：失败并尝试 drain；
- request cancel：停止派发新工作，协作取消 I/O，drain GPU/callback；
- drain 无法证明安全：进入 quarantine，保留 adapter、mailbox、pool 和 callback owner；
- 安全 drain 后才能 destroy pool 和释放 backing。

`StageExecutor` 析构时如果仍持有不安全 pool 且重试 drain 失败，会选择终止保护，而不是静默制造 use-after-free。
这是一条安全底线，不应为了让失败路径“看起来成功”而删除。

## 17. 当前源码结构

| 模块 | 主要文件 | 职责 |
|---|---|---|
| 配置合同 | `native/core/streaming_contracts.hpp` | presence-aware config、P/G/K/D/Q 类型 |
| 配置合并/验证 | `native/runtime/streaming/config.cpp` | overlay、schema、字段约束 |
| 布局模型 | `native/runtime/streaming/layout.hpp/.cpp` | descriptor、compiler、group/pool/slot capacity、digest |
| Slot 安全 | `native/runtime/streaming/slot_pool.hpp/.cpp` | generation、状态机、reader fence、poison |
| 有界 I/O | `native/runtime/streaming/io_executor.hpp/.cpp` | persistent worker、job ring、completion mailbox |
| 通用执行器 | `native/runtime/streaming/context.hpp/.cpp` | owner-pump、lookahead、pass、multi-pool、drain |
| MLX pager | `native/runtime/streaming/mlx_weight_pager.hpp/.cpp` | safetensors ranged fill、MLX shared backing |
| C bridge | `native/runtime/streaming/c_bridge.cpp`、`native/core/stream_slot_c.h` | 原生 C/Metal adapter ABI |
| Audit | `native/runtime/streaming/audit.*` | setup/steady allocation 和线程计数 |
| 模型 adapter | 各模型 runtime 目录 | descriptor、pool、fill、bind、encode、drain |
| Campaign | `tools/native/run_streaming_campaign.py`、`verify_streaming_campaign.py` | ABBA runner、evidence、verifier |

通用层不能依赖具体模型 tensor 名称；模型 adapter 不能复制一套私有调度状态机。共享的应是生命周期和安全协议，
保留在模型侧的应是 tensor 解释、binding 和 kernel 语义。

## 18. 推荐使用方式

### 18.1 内存充足机器

优先保持现有默认 resident/legacy 路径，不开启新框架。只有为了验证、统一部署或显式限制权重驻留时，才选择
已认证 streaming preset。

### 18.2 内存紧张但尚未触发 swap

选择对应模型已经验证的冻结 tuple，不要任意改变 K/G/P/D/Q。运行前检查模型 checkpoint、backend、shape、
step、LoRA/ANE 等是否在资格范围内。

### 18.3 已发生明显 swap

可以实验性比较 explicit streaming，但应同时采集：

- wall、denoise 和 stage timing；
- MLX peak、process RSS；
- system memory pressure；
- swap-in/swap-out；
- page fault；
- logical bytes loaded 和实际磁盘吞吐；
- GPU idle / owner wait；
- 输出质量。

在正式 P3 完成前，不应只根据一次 wall time 就宣布 streaming 普遍优于 swap。

### 18.4 自定义 slot 数

当前不建议普通用户对 production workload 自由搜索 K。正确流程是：

```text
inspect descriptor
    -> compile candidate layouts
    -> reject unsupported/capacity-invalid layouts
    -> synthetic schedule simulation
    -> real model smoke
    -> quality parity
    -> same-layout P1
    -> 独立内存校准和策略 P4 确认
    -> publish reviewed layout-only preset（不承诺 hard cap）
    -> 独立后续：guard P2 / L3、需要加速声明时的 swap P3
```

## 19. 新模型接入清单

一个新模型要接入通用框架，至少需要完成：

1. 枚举真实 block、field、source range、dtype、shape 和 layout class。
2. 明确 resident field、prefix 和 suffix 边界。
3. 明确跨 block fusion 是否允许在 group 边界切断。
4. 实现固定 pool backing 和无稳态分配的 fill/bind 路径。
5. 明确 GPU last-reader，不能只根据 CPU 函数返回推断安全。
6. 明确同步 reader 还是异步 callback reader。
7. 明确 pass transition：`reload` 或受限的 `carry_first_group`。
8. 多 layout class 时选择 `serial` 或显式支持 `retain_all`。
9. 实现 cancel、短读、错误 completion、drain 和 quarantine。
10. 报告 actual layout、fills、bytes、pool/slot 和 implementation identity。
11. 完成 host、sanitizer、真实 GPU、质量和性能验收。
12. 只有冻结 tuple 证据完成后才加入 registry/preset。

## 20. 验收层级

建议继续使用 P0–P4 分层：

| 层级 | 目的 | 关键问题 |
|---|---|---|
| P0 | 默认路径非回退 | 不开启新功能时是否与现有版本等价 |
| P1 | 同布局框架开销 | generic executor 是否不劣于 direct/legacy scheduler |
| P2 | 同布局 guard 开销 | guard off/on 的增量成本；内存覆盖/峰值校准另列 L2/L3 证据 |
| P3 | 低内存/swap | explicit streaming 是否优于 resident+swap/失败 |
| P4 | 策略搜索 | 不同 P/G/K/D/Q 和 retention 的 Pareto 前沿 |

任何发布候选至少还应通过：

- layout/compiler golden tests；
- K1/K2/K3 slot 状态机和 reader fence；
- ASan/UBSan、TSan；
- 短读、取消、stale completion、duplicate completion；
- 多 pass 和 multi-pool 边界；
- 真实模型输出一致；
- setup 后 steady allocation/thread-create 审计；
- 固定 policy、build identity 和可重放 raw evidence。

## 21. 当前限制和下一步

当前仍未完成：

1. Public production registry 和普通用户执行资格。
2. 全请求 resource closure 和可信 hard memory upper。
3. 真实低内存/swap P3 对照。
4. 任意 shape/checkpoint/K/G/P/D/Q 的自动资格扩展。
5. Session retention、跨 component 任意 DAG 预取。
6. 非协作 OS I/O 的硬 deadline 和完整 service 级 quarantine/eviction。
7. Flux 4B compiled graph、LoRA、GPU+ANE 等分支。
8. 普通 H3 和其他 H3 变体；当前计划只保留 H3 Turbo original BF16。

下一阶段优先级应是：

```text
保持默认路径 P0
    -> layout-only preset 的实测内存校准 / 策略 P4 / public registry
    -> App 显式内存目标选择（不承诺 hard cap）
    -> 独立后续：完整 resource ledger / bounded guard / L2-L3 / guard P2
    -> 对“优于 swap”的宣传另做真实 pressure + swap P3
```

Public layout-only 不以所有 bounded/P3 工作完成为前置；23/24 定义这条独立产品化路线。

## 22. 最终判断

当前通用框架已经证明：

- 同一套 slot 状态机和 owner-pump 可以同时服务原生 Metal 与 MLX 模型；
- LTX、Z-Image Turbo、H3 Turbo 和 Flux.2 Klein 9B 不需要各自维护完整调度器；
- 通过 adapter 可以保留各模型的 tensor、kernel 和 fence 语义；
- 冻结同布局上，通用框架开销处于目标范围内；
- multi-slot、cross-pass carry 和 retained multi-pool 已有真实模型用例；
- 默认路径可以通过入口隔离避免新增热路径工作。

但 slot 数不是完整内存上限，streaming 也不是无条件加速。当前合理定位是：

> 一个显式、可验证、可复用的权重驻留与调度框架，用固定 slot backing 和受控 I/O 替代不可预测的权重常驻或系统换页；
> 模型通过 adapter 接入，具体布局通过版本化 tuple/preset 认证；内存 hard cap 需完整资源闭包与 L2/L3，guard 开销由 P2 验证，优于系统 paging 的收益由 P3 独立验证。
