# 19 · 布局优先架构：可执行计划、调度算法与性能边界

[目录](README.md) · [框架合同](17-layout-first-framework.md) · [代码矩阵](18-code-change-matrix.md) · [实施与验收](20-layout-first-implementation-and-acceptance.md)

日期：2026-09-16。状态：**实施设计，不是完成或性能认证**。
17 负责总体合同，本文补充可直接实现的对象关系、编译/执行算法和算例；20 给出下一批 PR 和验收。
02 是配置唯一规范，09/10 是 compiler/executor 规范，05 是内存合同，12 是性能阈值唯一规范。
新增类型名、报告字段若未在现有头文件出现，均为拟议内部接口，不是已实现 API。

## 1. 结论：用户选布局，框架验证并执行布局

可以把内存受限的主要调优入口改成 slot/block 参数，而不先让 runtime 根据 Y 搜索参数。
但必须区分三个承诺：

| 能力 | 由谁决定 | 能保证什么 |
|---|---|---|
| layout | 用户显式参数或显式选用 preset | 分组、slot 数、prefix、预读距离和 worker 数可复现 |
| slot safety | executor + adapter 的最后 reader 证明 | 不在读者结束前复写，不把半成品内容提交计算 |
| whole-request bound | 独立 resource plan + 资格 + admission/guard | 在明确纳管/envelope 合同内约束整个请求；不是操作系统独占 RAM |

只给 K 可以简化 UI，不能简化下层合同；UI/preset 仍必须展开完整 K/G/P/D/Q。
框架既不按空闲 RAM 偷改这些值，也不通过自动降精度、少算 block、缩 shape 来“适配”。

两条路线在入口分开：

- **legacy**：未启用新 streaming，继续现有 resident 或旧 streamed；未启用 guard 时也不创建 guard。
- **explicit layout**：metadata→compile→资格→可选预算→construct→run→drain；unsupported 必须拒绝。

旧 guard-only 功能保持既有路由；不能因为新框架关闭就把用户独立启用的 guard 关掉。
GPU-only 指模型计算后端为 GPU，CPU 文件读取、转换与提交仍是必需辅助工作。

## 2. 模块边界与依赖方向

```text
request/profile ──> StreamingConfig
                         │
ModelSession.describe ──> OwnedPreflight(snapshot + descriptor + construction metadata)
                         │
                   compile_layout (pure)
                         │
                   ResolvedStreamingPlan
                         │
               capability / registry / optional admission
                         │
                StreamingRequestContext
                         │
                request coordinator
            text ─> denoiser ─> VAE ─> export
                        │
                   StageExecutor
             ┌──────────┼──────────────┐
        SlotSafety   IoExecutor    ModelSlotAdapter
        generation   bounded Q     fill / bind / encode / drain
             └──────────┴──────────────┘
                       backend
```

**不另造一个完整模型执行器。** 现有 sampler/数值 kernel 保留；coordinator 管资源边界，
StageExecutor 管单一兼容权重域；模型 adapter 管字段语义。组件生命周期也能共用 request context，
但当前 StageExecutor 不支持 resident/multi-class，不能硬套进去。

| 层 | 允许依赖 | 禁止依赖/行为 |
|---|---|---|
| core/contracts | 纯值类型、版本化 POD | Foundation/Metal/模型 loader |
| runtime/compiler | config、descriptor、checked arithmetic | RAM probe、文件加载、GPU 分配、预算搜索 |
| runtime/executor | immutable StageLayout、窄 adapter、mailbox | 模型名分支、JSON、文件路径、每模型线程池 |
| model adapter | descriptor 投影、reader、已有 kernel/backend | 自行重分组、修改 K、绕过统一 safety |
| platform/session | model/component lifecycle、GPU/MLX handles | 在 legacy 每层插入新框架探测 |
| tools/verifier | 导出的 plan/actual/raw evidence | Python 重写生产 compiler 后用它自证 |

构造用字符串 ID，热路径用有界整数索引。request-level plan 保持不可变；计数器、ticket、采样属于 mutable execution state。
基础对象建议放在现有 runtime，而不引入新 plugin 系统或第二套 memory scheduler。

## 3. 四张布局表与一个资源表

```text
source_table     artifact + byte range + dtype/shape + identity/reader revision
backing_table    destination storage + alignment + capacity + ownership/domain
binding_table    block/field -> group/slot/span + conversion/derived dependency
access_table     ordered blocks + pass/step/phase + safe boundary + last reader
resource_plan    non-slot lifetimes + baseline/reservation/envelopes + unknowns
```

当前 `layout.hpp` 已有 SourceArtifact/SourceRange/Materialization/PassSpec/workload，
但还没有完整 request resource plan 或完整 C construction view。不要因 `materializations_complete=true`
就把 unknown activation/VAE/driver 开销标为已闭包。

### 3.1 分组和容量的确定性算法

1. 固定 workload：shape/token bucket、格式、真实 pass、数值选项，禁止执行后才发现另一个 shape。
2. 从 ordered blocks 取前 P 个为 prefix，不参与 suffix 分组。
3. 在连续同 class 且安全边界内按最多 G 个分组；尾组可短，不跨 class/fusion 边界。
4. 单 class 池中按局部 ordinal `slot(g)=ordinal(g)%K`；必须满足 K≤该池的 group 数。
5. 每个 slot、每个 block position、每个兼容 field 分别取最大容量并对齐，再求和。
6. 填写 source reads、转换/派生、access/pass 和 construction view；读量未知保持 null。
7. canonical digest 绑定行为、source 和 workload，不包含 Y/X/S、probe、wall 或用户机器绝对路径。
8. 能力检查拒绝当前 executor/adapter 不支持的 layout；compiler 能生成不代表 runtime 能执行。

对 slot s 的字段 f：

```text
cap[s,f] = align_up(max(required_bytes[g,f] for g mapped to s), alignment[f])
pool_bytes = sum_s sum_unique_backing_f cap[s,f]
```

需 checked overflow；真实 alias 按 destination storage 去重；两个独立分配即使来自同一个 tensor 也要分开计费。
字段 `(320,64)` 与 `(64,320)` MiB 映到同槽时，固定字段容量是 640 MiB，不是 max(group total)=384 MiB。
offset/shape 相同但 storage mode/packing 不兼容时不能只按字节数复用。

### 3.2 三种“字节”不能混用

- backing capacity：实际预分配上界，包含对齐和尾组空位。
- content bytes：本次有效写入，通常小于或等于 capacity。
- logical source reads：pread 源范围的读取量；不等于 content，更不等于 SSD physical traffic。

当前 executor 的 `bytes_loaded` 从 completion.bytes 累加，只能按 adapter 现有口径解释。
后续独立增加 source/content 计数，版本化报告，不能重命名旧字段后声称历史性能可直接比较。

### 3.3 快照与执行身份

preflight 持有 header/open fd；metadata-only 不创建 GPU/整块权重。fd 防路径重开换源，
但不是防同 inode 原地修改的隔离快照。首发要求输入 artifact 请求期间不可变；stat 检查只检测变化，
可信 content identity/注册流程另行提供。hash 成本显式计入注册或 preflight，不每次偷偷读完整 checkpoint。

所有读取 checkpoint 的 helper 都在表中声明来源：denoiser、connector、upsampler、text 和 VAE 可来自不同 artifact，
但各自必须绑定自己的 snapshot。不能只让 slot fill 用同 fd，而让 connector 继续按可变路径打开却声称全请求同源。

## 4. 请求所有权与事务接线

建议持有关系：

```text
EngineOwner（仍按现有 GPU 单作业串行）
  └─ StreamingRequestContext
       ├─ OwnedPreflight: header/fd/descriptor/construction metadata
       ├─ shared<const ResolvedStreamingPlan>
       ├─ model exact state + sampler workspaces/conditioning/latents
       ├─ StageExecutor + adapter + callback-visible state
       ├─ optional guard + resource ledger
       └─ terminal report / primary and cleanup errors
```

metadata 的借用必须持续到最后 worker/adapter 退出；sampler 临时量如被异步 GPU 引用，也要属于可保留的 owner。
guard 最后释放，不能在回调还可能访问时析构。失败只销毁已经安全的子集，其余移交整个 quarantine owner。

### 4.1 接线顺序

```text
parse/resolve → snapshot/describe → compile → capability/qualification
 → [bounded: resource qualification → documented clean boundary → admission]
 → construct → bind → model run/pass calls → drain
 → verify actual → finalize output/report → unbind/release
```

资格/配置失败不卸载旧 session。bounded admission 如需清旧 cache，应在资格通过后执行并报告副作用；
后续失败不承诺恢复旧 cache。construction 失败逆序回收，不自动 fallback legacy，不减 K 重试。

prepare(load-only)、prepare(warmup)、generate 分别有访问模板与资格；首批未支持的 prepare 明确拒绝。
公开 `streaming_layout_not_certified` gate 应由真正的 preflight/registry 取代，不能直接删除放行。

### 4.2 Exact handle 与失败销毁

LTX legacy 的 `unique_ptr<...,ltx_native_free>` 不变。
新 request owner 使用 `ltx_native_streaming_destroy(&handle,...)`，只有返回安全成功后才能把 handle 置空。
构造失败也可能返回 quarantine handle，错误分支不能丢弃它。

`StageExecutor::retry_drain()` 当前会调用阻塞 join；析构时 unsafe pool 会 terminate。
因此当前代码不是“超时 N 秒内一定安全退出”的实现。下一步需落实外层可保留 owner/service eviction 规则；
需要硬恢复时限的服务再独立设计进程隔离，不用强杀线程假装安全。

## 5. Multi-slot 的唯一调度语义

沿用 02：G≥1，0≤P<N，1≤K≤groups，0≤D<K，1≤Q≤K，retention **仅 request**。
stage barrier 是内部生命周期边界，不是新增 `retention=stage`；session retention 留给 F9b。
P/K/G 还受 adapter 限制，例如当前 LTX 原型 P≥1/G=1/K≤3。

### 5.1 Backing 与内容分离

```text
backing: allocated ─────────────────────────────────── released
content: Vacant → Loading → Ready → InUse → AwaitingFence → Vacant
request: running → draining → safe terminal | quarantined
```

状态名对应现有 `slot_pool.hpp`。poison 是 sticky failure，不新增一个可以随意释放的 content state。
InUse 的 reader 尚未 seal 时不可复用；AwaitingFence 要等所有 queue 的最后 reader。Vacant 只说明可重填，
不说明内存归还。只读权重 offload 通常是丢弃旧内容/重用 backing，不是每块做一次 GPU→CPU 权重回拷；
需保存的可变状态必须单独建模，不能随权重一起丢弃。

### 5.2 Owner pump（与当前 context.cpp 顺序一致）

```text
consume completions and check sticky error/cancel
while dispatch <= next_submit + D and mapped slot is Vacant:
    create generation-bound ticket
    enqueue bounded fill job; advance dispatch
if prefix not yet submitted:
    encode prefix once for this pass
if next_submit slot is Ready:
    prepare derived values on owner
    begin_use → encode_group → seal all reader fences
    advance next_submit only after successful submission
if no progress:
    wait on mailbox; apply stall detection (not a hard kill deadline)
after all groups: drain pass readers before next pass
```

先派发首批 fill 再执行 prefix，才能把 prefix compute 用作 startup overlap。
fill worker 只做已声明 read/convert/write，不提交 GPU、不调用 owner ledger。
completion 包含完整 request/pool/pass/step/group/slot/generation；先验证后发布 Ready。
短读、重复/旧 ticket、mailbox overflow、失败 submit 都进入 sticky failure。

当前 LTX adapter 的 encode 边界可能同步完成原有 GPU 工作；不要假设函数返回就代表一个
全异步 backend。改成异步提交需单独 reader/credit 设计及 P1 验证，不能顺手改旧 batch。

### 5.3 有界控制内存与公平性

jobs、mailbox、slot records 的容量随 K/Q/最大 reader queues 有界，不随 diffusion steps 增长。
现有 mailbox 用 `K*(1+TC_STREAM_MAX_READER_QUEUES)`；保持一 fill completion/每 queue 最后一 reader
的合同才可沿用此容量，扩事件种类必须重新证明容量并测试 overflow。full trace 单独有上限/丢弃计数，
不能因调优日志无限增长破坏低内存模式。

当前需求优先；不让未来 group 占用本组必须的 source scratch 或 owner credit。
Q>1 允许乱序完成，但 compute 仍按模型顺序。不通过增隐式 slot 或 fallback pthread 化解阻塞。
D=0 不等于绝不 overlap：next 在 compute 提交后推进；K=1 则必须等唯一 slot reader 后重填。

### 5.4 可手算的时间线（合成，不是 LTX 成绩）

条件：6 个同构 group、无 prefix、Q=1、read+convert=8 ms/group、GPU compute=12 ms/group；
owner/通知零耗时、单 queue、计算提交与实际开始分开、无共享带宽竞争。K2 取 D1。

| group | slot | fill 区间 ms | GPU compute 区间 ms |
|---|---:|---|---|
| g0 | 0 | 0–8 | 8–20 |
| g1 | 1 | 8–16 | 20–32 |
| g2 | 0 | 20–28 | 32–44 |
| g3 | 1 | 32–40 | 44–56 |
| g4 | 0 | 44–52 | 56–68 |
| g5 | 1 | 56–64 | 68–80 |

K1 总时间 6×(8+12)=120 ms；K2=80 ms；理想 K3 仍是 80 ms。
第三槽可吸收部分抖动，但没有平均吞吐的必然收益。g2 不能在 16 ms 就覆写 s0，必须等 20 ms 的 g0 最后 reader。
若改 read=18 ms、compute=12 ms，则 K1=180 ms、理想 K2=120 ms；更多槽不能突破读取吞吐瓶颈。
真实测量要补 prefix、owner、multi-queue tail、每 pass drain、text/VAE/输出，不能将该比值外推到完整请求。

### 5.5 Pass 与 class

同一 compatible domain 的 pool/workers 跨 LTX 11 个 pass 保留，内容首版每 pass 重填；
upsample 只在 pass drain 后交接 activation/workspace，不重建 weight pool。
新 timestep 必须更新 derived conditioning；同地址不代表同内容。当前 P=1/G=1 的 48-block、11-pass
用例应是 (48−1)×11=517 次 suffix fills；prefix 读取与 helper 读取另计，不能把 517 当任意布局常数。

多 class 目前仅 compiler 能表达。首版扩展按 barrier 先释放旧 pool 再建新 pool；
每个 generation 独立计数，不谎报 whole-request 只建池一次。跨 pass 预取、同时保留多 class 池、
session cache 都需改访问/资源合同并重新验收，而不是 adapter 私自优化。

## 6. 内存核算、overlap 与 swap

### 6.1 先选 layout，再计算需要多少

```text
request_upper = max_epoch(
  residual baseline + unique live backings + outstanding reservations
  + pending releases + owned scratch/control + non-overlapping framework upper
)
B = floor(Y * (100 - X) / 100)
admit iff request_upper <= B and reliable_system_headroom permits increment + S
```

reservation 转 allocation 不重复计；已分配但 pending release 的 backing 仍计一次。
CPU shared buffers/CPU scratch/MLX/driver 部分依 05 统一计量，不把 GPU-only 理解为只计算 MTLBuffer。
buffer X 是保留百分比；S 是系统余量，不重复从 B 扣一次。

算例（完全合成，单位 GiB，假设各行含全范围且无 unknown）：

| epoch | prefix+pool | activation/conditioning | VAE+output | baseline/control/envelope | 合计 |
|---|---:|---:|---:|---:|---:|
| denoise | 4+5 | 3 | 0 | 1 | 13 |
| denoiser 已安全释放后的 decode | 0 | 1 | 8 | 1 | 10 |
| 若改为重叠 decode/denoiser | 4+5 | 3 | 8 | 1 | 21 |

Y=20 GiB、X=15% 得 B=17 GiB。串行方案 upper=13 GiB，可做后续 admission；重叠方案 21 GiB 拒绝。
不能把互不重叠的最大项盲加成 21 GiB，也不能未经 drain 就按串行的 13 GiB 放行。
实际系统若还有 unknown 就不能 bounded；以上只是 ledger verifier 的手算 fixture。

### 6.2 优化目标与性能模型

优化完整 wall、P95、峰值与成功率，不最大化 overlap 百分比。
稳态简化吞吐下界为 max(每组 IO/convert 服务时间、owner 服务时间、GPU 服务时间)，
K1 通常暴露读+算串行；K≥2 仅在服务资源可并行且安全 credit 足够时接近下界。
Q 加大可能增加 CPU/共享带宽竞争，G 加大增加 startup 和尾组浪费；P 增大减少重复读取但增加固定峰值。
这些是测试假设，不是自动选择参数的 runtime 规则。

建议优化顺序：去掉重复 metadata/分配→持久 workers→保证 demand priority→定位 starvation→对齐 batch/fence
→实测有限 K/D/Q 候选→最后研究 range 合并/跨 pass 等语义扩展。
先修实现开销，再调布局；不能用增大 P 掩盖 P1 回退。

### 6.3 与 swap 的判断边界

显式布局可能让读取更可预测、降低活跃权重，并避免一部分 paging 等待，但仍支付源读取/转换/同步。
若 resident 已放得下，streaming 反复读权重没有必胜理由；若主要峰值是 VAE，减 K 也未必有用。
OS clean file page 回收、compression、swap in/out 与本进程 logical pread 不能混成一个“IO bytes”。
这些机制按 16 的假设通过 P3 验证，不把当前 tiny smoke 写成快于 swap 的证据。

大 RAM 上设置小 Y 只模拟 admission，不能模拟物理低内存。仿真先用上述手算 oracle，
再校准真实读取/转换/compute/通知分布，另用未参与校准的数据验证预测误差和排序。
真实压力需另行授权、限时限额和停止条件；不修改系统 swap，不 purge 用户缓存。

## 7. 模型接入与配置产品化

| 模型 | 首个可实现范围 | adapter 必须补充 | 后续扩展 |
|---|---|---|---|
| LTX | native GPU、G1/P≥1、单 class、request retention | 同 snapshot 构造、真实 conditioning、两 stage/upsample、完整输出/cleanup | P=0、G>1、多格式；各自测试 |
| H3 | 新路由 K2/G1 | active block ID、norm/AdaLN 常驻、跨 block reader/fusion、跨 forward prefetch 语义 | K1/更多 K，不能只去掉 xor |
| Flux | component-staged | 输出 materialize、graph/compiled cache 引用、text/denoiser/VAE handoff | 格式 reader、异构 block class、稳定绑定 |
| Z-Image | component-staged | 同上，加实际量化/shard 生命周期 | 经过独立资格的 block streaming |

共用 descriptor/compiler/executor/报告；不强迫 kernel 统一，不为每模型新建一套 scheduler。
每个 adapter 附 metadata fixture、construction/fill/bind、最后 reader 证明、完整 workload 和证据包。

配置沿用 02/examples，不新增 slot_num、auto_slots、memory_limit 等别名：

| 用户意图 | 提交方式 | 框架动作 |
|---|---|---|
| 大内存保持现状 | 不启用 streaming；guard 保持用户原设置 | 原路线，无新热路径工作 |
| 固定 2 槽 | 完整 K2/G/P/D≤1/Q≤2 tuple | 正向编译、资格检查，不猜 D |
| 固定布局加 Y | 同一 tuple + memory_constrained.limit_bytes/buffer_percent/min_free_bytes | 同布局 admission，失败不降 K |
| 不懂高级参数 | 显式选 preset | UI 展开完整字段并显示实际解析值/来源 |
| 改 K3→K1 | 同时选择合法 D0/Q1（或完整 preset） | 若只改 K 而继承 D2 则拒绝 |

preset 是配置，不是授权；性能/资格记录在只读 sidecar/registry。
同 RAM 不等于同 GPU/SSD/shape；不发布无证据的“16 GiB 就选 K1”通用映射。
新显式 resident 仍有独立资格与 request-retention 成本，不能冒充 default resident 的 warm-cache 性能。

## 8. 性能保护与本轮边界

结构要求：legacy 不创建 descriptor/context/pool/worker/probe；新 steady state 不进行框架分配、线程创建、parse/hash。
保持 kernel、提交批次和已有 cache 策略，优化仅在入口新路由。用 audit build 验证这些为零，再用 release build 测 P0。

统计与发布完全遵循 12：P0 默认 median 比值区间上界≤1.02/P95≤1.05，
P1 同布局 median≤1.03/P95≤1.05；P2 为 guard 开销，P3 为 OS-managed 对照，P4 为布局策略变化。
不是“测 5 次就通过”，也不允许通过 P3 抵消 P0。每个正式 workload 独立判断。

目前 compiler/executor/LTX 内部原型存在，但 session/资源闭包/production 资格未完成，详见13/20。
本文未给出新的模型性能成绩；没有 normal-target ABBA，就不能承诺已实现性能非劣。
