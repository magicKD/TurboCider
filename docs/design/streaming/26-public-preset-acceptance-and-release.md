# 26 · 内存档位工具链、实施与发布验收

[目录](README.md) · [候选和实验原则](24-memory-tier-exploration-and-acceptance.md) · [代码规格](25-public-preset-implementation-spec.md) · [性能统计规范](12-acceptance-playbook.md)

日期：2026-09-17。状态：**待实施验收规格**。新增测试名、文件名、工具参数均为建议；本轮未运行 GPU、压力或档位 sweep。
文档 12 是 P0–P4/L0–L3 的唯一规范；本文使用 `PUB-*` 子测试 ID，不重定义这些级别。

## 1. 完成标准：不只“生成成功”

一个档位可以发布，当且仅当下列交付完整：

1. 普通 public constructor 能执行已审核 record，无记录/错误 source/撤回 record 被拒绝。
2. actual layout、component plan、reader/kernel/format、retention 与审核记录一致。
3. full request（包括 setup、文本、VAE、export、cleanup）经验峰值完整，`M+H(T)<=T`。
4. 对声明支持的设备/container 做独立确认；64 GiB 机器的软件 target=8 GiB 不算物理低内存证明。
5. 质量、取消、读失败、source mutation、reader completion 和 quarantine 通过。
6. 默认路径 P0 通过；同布局 P1 和策略 P4 各有明确结论；INCONCLUSIVE 不改成 PASS。
7. App 旧 draft/job、开关切换、v2 序列化、public API、报告和撤回一致。
8. release 卡标明 `memory_enforcement=none`；不承诺 hard cap、零 swap 或未验证的速度。

当前四个模型的 private adapter/P1 锚点不等于这些新条件已全部通过。尤其 H3 仍缺 public full-pipeline 档位证据；
Flux 新 prefix 尚未实现，不能先配置一个 8 GiB public record 再尝试凑证据。

## 2. 工具模块：复用已有 runner，分离测量目的

### 2.1 建议目录和职责

```text
tools/native/
  run_streaming_campaign.py                已有，P0/P1/P4 release timing 主协议
  verify_streaming_campaign.py             已有，统计/identity/quality 校验
  capture_streaming_source_identity.py     已有，源码构建证据
  inspect_streaming_candidates.py          新增，只读 metadata + candidate 去重
  calibrate_streaming_memory.py            新增，fresh-process/连续请求/切换采样
  explore_streaming_presets.py             新增，有界 E0–E3 orchestration
  build_streaming_catalog.py               新增，离线候选编译/审核发布分离
  verify_streaming_calibration.py          新增，独立 raw→calibration verifier
  streaming_memory_schema.py              新增，纯数据合同/校验，无 native import
  streaming_memory_worker.py              新增，child native runner/自观测
```

没有理由复制所有 P1 ABBA 逻辑。新 calibration coordinator 与 worker IPC、source identity 可以复用小模块；
若抽取已有 runner 的辅助函数，必须保持原 policy/schema/样本/统计输出兼容并重跑 synthetic tests。

coordinator 不加载 native dylib，不在自身统计 MLX；只有 worker 或实际 App 执行进程加载 native。
一台 GPU 同时只允许一项实验。即使 host metadata 可并行，也不并行跑模型、采样确认或 ABBA 两个 variant。

### 2.2 模式不能混用

| 工具模式 | 进程/engine | 用途 | 不可声称 |
|---|---|---|---|
| metadata | 不创建模型 engine 的 GPU 资源 | 校验布局、source bytes、候选剪枝 | 实测内存/速度 |
| fresh_process | 每样本新 worker，engine 从零建立 | 构造至销毁的整进程范围 | 冷 SSD/page cache |
| warm_request | 同 engine 连续请求，按 request time window 分隔 | cache state、增长/泄漏、热性能 | 用 ru_maxrss 得到每请求峰值 |
| mode_transition | 同真实容器 resident→streaming→off | 旧 cache 重叠、释放与 reload 成本 | 与原 P1 同布局等价 |
| release_timing | 既有固定 lifecycle，关闭密集 sampler/audit | P0/P1/P4 wall/denoise 非劣 | 由无采样请求证明峰值 |
| audit | hook/fault/trace build | 分配/线程/reader/queue 安全 | 正式 release timing |

`ru_maxrss` 是进程 lifetime high-water；创建新的 engine 不会重置进程高水位。保留该字段并明确 `scope=process_lifetime`，
不能减去上次 ru_maxrss 或把其下降假想为每请求峰值。每请求主域由时间窗采样得到。

## 3. 实验输入合同与资源限额

### 3.1 Policy 必填字段

新 exploration policy 独立版本化，不把未知字段塞进现有 P1 policy。以下结构是**草案模板，不可直接执行**：

```json
{
  "schema_version": 1,
  "status": "draft",
  "experiment_kind": "memory_tier_exploration",
  "model_id": "flux2-klein-9b",
  "artifact_manifest_digest": null,
  "execution_container": "cli_worker",
  "workload_card": "illustrative.flux9.512sq.4step",
  "targets_bytes": [8589934592,10737418240,12884901888,17179869184,21474836480],
  "candidate_policy": {
    "max_coarse_candidates": 8,
    "max_finalists": 3,
    "deduplicate_by": "layout_and_component_identity"
  },
  "measurement": {
    "sample_interval_ms": 20,
    "max_tree_skew_ms": 10,
    "max_sample_gap_ms": 60,
    "scope": "execution_process_tree_v1",
    "estimator": "observed_tree_max_v1",
    "max_sampler_overhead_ratio": 1.02
  },
  "limits": {
    "max_requests": null,
    "max_wall_seconds": null,
    "max_logical_read_bytes": null,
    "max_output_bytes": null,
    "min_free_disk_bytes": null,
    "min_system_available_bytes": null,
    "request_timeout_seconds": null,
    "cleanup_timeout_seconds": null,
    "max_stop_wait_seconds": null
  },
  "protocol": {
    "launch_pressure": false,
    "allow_gpu_execution": false,
    "exclusion_policy": "retain-all-raw-no-posthoc-exclusions"
  }
}
```

`null` 故意阻止实际启动，操作者需在 freeze 前填写真实模型/设备/limits。`allow_gpu_execution=false` 默认只做 E0。
20/10/60 ms 与 2% sampler overhead 是初始工程目标，不是已有成绩；平台不满足时先调整并重新冻结 protocol，不能测后改口径。
archive 保存原 policy bytes 和 SHA-256；resume 必须 identity/limits/protocol 相同，预算累计而不是从零重置。

### 3.2 Workload 卡

卡片包含 request payload、artifact/config/tokenizer digests、GPU/backend、exact shape/frames/steps、prompt/seed set 的本地引用、
token rows、operation/audio/inputs、kernel/reader/format、expected output modalities、container/cache policy。
测试 prompt 可用仓库小型合成文本；用户私有 prompt、绝对模型路径不提交。bundle 中的可分享部分用逻辑 model alias。

首批 normal cards 按文档 24：LTX 512×320×33、11 steps；Z 512²、9 steps；H3 Turbo 512²×22、4 steps；
Flux9 512²、4 steps。1024²/97frames/39frames 等都是额外卡，不能从第一张成功外推。

### 3.3 请求数和读取量必须先算

E2 每 candidate 的初值是 3 个 fresh +10 个 warm 请求；8 个候选已经是 104 个请求/卡，还未含 baseline、smoke、
mode transition、audit 和 E3。工具先显示总数/预计读量/耗时并按 limit 截止，不能把“最多8候选”误解为总共8次运行。

```text
planned_requests = smoke + exploration_cold + exploration_warm + transitions
                   + confirmation + timing + quality + audit
logical_reads_per_request = sum(stage.resident_source_read_bytes)
                          + sum(stage.prefix_source_read_bytes)
                          + sum(stage.pass_count * stage.source_read_bytes_per_pass)
                          + declared text/VAE/other component reads
planned_reads = sum(all scheduled requests' logical_reads_per_request)
```

unknown component reads 不能当0。先用保守上界；若无法给出可靠上界，先执行更小的授权 probe 补齐，再重新 freeze 大 campaign。
请求启动前做 admission reservation，完成后记实际 counter；运行中超预算停止新 dispatch/新请求并安全 drain。
硬件 page cache 可能让物理 reads 不同，logical read budget 仍按上述消费，避免随 cache 状态无限重跑。

## 4. 内存观测：时间轴、scope 和有效性

### 4.1 主域与辅助域

主域 `execution_process_tree_v1`：实际执行 root +生成相关 helpers。embedded App 含其基线；cli/service worker 含 worker/helper，
App shell 单列。coordinator/sampler 自身单列 experimental overhead；不把 external pressure helper 加入请求主域。

主域统一使用选定的 footprint 口径，各 PID 同一 metric；缺失时保持 null，不能悄悄用 RSS 替代后拼接。
可另报告完整 RSS tree，但 metric 改变需新 estimator/calibration revision。MLX active/peak/cache、Metal live/allocated、
纳管 allocator、compression、page faults、system swap 分开保存，不相加成总内存。

跨 PID 共享内存不能天然去重：保守相加记录 `aggregation=conservative_no_shared_dedup`，且不能称真实唯一物理字节。
footprint 是观察而非资源形式化上界；有 paging/compression 造成指标口径冲突时不选更小值完成分档。

### 4.2 worker 自观测与 PID 身份

建议 worker 用现有 Apple memory probe 逻辑的只读接口采自身，在 coordinator 合并；避免依赖默认不存在的跨进程 task 权限。
App embedded calibration 需 App 内的显式测试通道自报；不能用 CLI 的 RSS 代替 App footprint。

每个 PID 使用 `(pid, process_start_identity, role)`，不能只用 PID；helper launch 前注册、退出时注销并有最终读数。
fork/exec/重启要更新身份并保留前后关系。短命 helper 只靠轮询容易漏，launch lifecycle hook 必须纳入 coverage。
不同 Mach/进程时钟先验证可对齐，记录 shared monotonic origin；不确定时保留 interval/skew，不能强行当同时值。

### 4.3 Raw sample schema

下面是合成 fixture，只示范 null 和分类，不对应真实模型结果：

```json
{
  "schema_version": 1,
  "run_id": "synthetic-run-1",
  "request_id": "synthetic-request-1",
  "sequence": 12,
  "timestamp_monotonic_ns": 1000000000,
  "collection_start_ns": 998000000,
  "collection_end_ns": 1002000000,
  "phase": "denoise",
  "execution_container": "cli_worker",
  "processes": [
    {"pid":123,"start_identity":"synthetic-start","role":"root", "footprint_bytes":null, "rss_bytes":null}
  ],
  "mlx": {"active_bytes":null,"peak_bytes":null,"cache_bytes":null,"reset_epoch":0},
  "metal": {"observed_allocated_bytes":null,"tracked_live_backing_bytes":null},
  "system": {"swapin_delta_bytes":null,"swapout_delta_bytes":null,"pressure":"unknown"},
  "coverage": "incomplete",
  "missing": ["root_footprint"]
}
```

不要向 final metadata JSON 写每 block trace；sampling stream 单独 bounded JSONL。coordinator 一次读取/写入 bounded batch，
buffer overflow 必须记录 dropped samples 并令校准失效，不能持续增长 RAM 或静默丢峰值。

### 4.4 同时峰值算法

```text
for aligned sampling window t:
    require every live generation PID has valid footprint within max_tree_skew
    tree(t) = sum(footprint(pid,t) for all generation PIDs alive at t)
F(request) = max(tree(t) for t from construct-start through safe-cleanup)
M_observed = max(F(request) for all independent confirmation samples)
M = max(M_observed, complete same-scope stage prediction if available)
```

禁止 `sum(max_each_pid)` 伪装同一时刻 tree peak；若保守采用该上界要另起 estimator，清楚标注并重新审核。
缺 PID/样本间隔超限/启动或 cleanup 未覆盖/metrics reset 未跟踪 → incomplete；辅域已知 live backing 明显超过主域时调查。
若在 20 ms 采样间发生短峰值，边界 hooks 和 5 ms 复测有助于发现，但仍是 sampled maximum，不是 hard upper。

sampler 在 worker ready handshake 后、engine construct 前开启，generation phase 与实际 helper PID 一起打点。
构造本身过短时仍需起止事件；service 老模型 cached baseline 也要采。MLX peak reset epoch 每次记录，不能跨 epoch 取差。

### 4.5 Baseline-to-peak 和额外准入量

每个样本配对 `B_i=该请求构造/转换前同scope baseline` 与 `F_i`；记录缓存状态及 baseline 范围。
候选额外需求可用 `A_i=max(0,F_i−B_i)` 的保守确认值，但不能用 `max(F)−max(B)`，也不能跨 PID/container 相减。
idle resident cache 在 F 里重叠时不能忽略，request target 的 M 仍取绝对 tree peak，不是增量 A。

正式执行的当前 baseline 超出校准范围，则 preflight 拒绝或要求显式重新建立合适会话；不为了通过准入自动清掉所有默认 cache。
系统 available-memory 数据不可靠/不可用时，返回观察不可用而非按 free pages=available 推导安全。

### 4.6 完成与异常样本

全部 raw request 有终态 `success|cancelled|timeout|failed|quarantined|interrupted`。失败的 timing/peak 仍保存，未完成输出不能算质量通过。
目标适配器不得跳过出错/偏高样本取成功最大值；修复后开新 campaign，旧 evidence 留存并说明不采纳原因。
thermal/系统压力污染时 campaign 为 INCONCLUSIVE，不能将被污染样本删除后继续原冻结统计计划。

对仅标注 empirical-target 的实验发布不要求宣称零 swap；但不能利用明显 paging/compression 导致变小的 footprint 去认定低档适配。
需要诊断主域与已知 backing/外部压力，符合预注册“干净校准”条件后重测；bounded/no-swap 验收仍更严格。

## 5. 候选生成、仿真和独立确认

### 5.1 E0 metadata 剪枝

`inspect_streaming_candidates.py` 使用模型真实 descriptor 和同一个 layout compiler。输出每个 candidate 的：

- canonical config、source/layout/component digests、支持/拒绝理由；
- resident/prefix 字节、每 class pool/K、peak live pool 字节；
- pass 数、每 pass suffix source bytes、setup reads、预估总逻辑读取；
- activation/text/VAE/export 未知项、是否只有 denoiser scope；
- 预计可探索目标（unknown 也保留），不得输出 public available。

候选按 layout+component identity 去重，不能因为 target 不同重复测相同布局。metadata 只判断合法和明显不合适，
不把 prefix+slot bytes 当 full-request peak。低档超 text/VAE floor 时先停止该档 sweep，而不是继续减少 P 寻找不可能结果。

初轮最多8个 coarse candidate/卡，确认最多3个；在 memory target 边界附近细化 P，但每次增补仍计入冻结预算。
若改变 candidate 集或策略，需要开新探索 revision；确认集不得因看过赢家结果再回流到探索。

### 5.2 仿真的正确用途

新增/扩展仿真仅用于排名和解释，不产 public record。用实际 block trace 提取：source range/read bytes、Q 下读服务时间分布、
prefix/kernel/reader 延迟、D 窗口、pool class transition、allocation 和 pass carry 行为。

最小离散事件模型应包含：

```text
fill(g) earliest start:
    source ready ∧ one I/O worker available ∧ its exact slot vacant
    ∧ dispatch window permits ∧ class/pool policy permits
encode(g) earliest start:
    fill(g) complete ∧ all preceding model dependencies complete
slot(g) reusable:
    fill producer ended ∧ all declared GPU readers complete
request peak:
    baseline + simultaneously live components/prefix/pools/activations/workspace
```

不要用 `max(sum IO,sum GPU)` 作为真实预测，它只能表示理想重叠下界；startup/drain、串行 stage、slot 竞争、Q 读盘竞争
以及统一内存带宽争用会改变服务时间。K3/Q3 不一定快过 K2/Q1。page cache 和 OS paging 不在简单 slot 模型里自动出现。

校准至少留一组未参与拟合的 layout/shape trace，报告 wall/wait 误差和 candidate 排名命中率。
预注册建议目标为：held-out wall 相对误差中位≤10%、P95≤20%；不达标仍可做 sensitivity 图，但不能自动淘汰性能接近候选。
这些是工具目标不是实际准确度；硬件/SSD/runtime 改变要重新校准。不能用 MLX synthetic peak 仿真替代 full-pipeline 实测。

### 5.3 探索到确认的状态机

```text
proposed → metadata-valid → smoke-valid → explored
         → independent-confirmed → reviewed-release-candidate
         → staging public-route verified → public-experimental/stable
```

每一步保留失败原因；“reviewed-release-candidate”只是 staging 构建可测试的记录，不是用户 App 已可选。
E3 冻结 finalists、独立 prompt/seed（在合法 workload 范围内）、样本数、最大预算、统计协议。
如果必须改变 token 条件，先把新条件加入拟议 applicability 并独立覆盖，不可让确认请求意外落到另一 bucket。

E2 的3 fresh +10 warm只是筛选；E3每待发布 tuple至少10 fresh，另有 repeated/mode transition全集及正式配对 time确认。
样本数按卡片预注册且受预算约束，不能无止境“跑到PASS”。warm至少10请求用于初步增长观察，不等于证明长期无泄漏。

### 5.4 Catalog 生成算法

独立 verifier 读取 raw，不信任 worker 提供的 `passed=true` 或预计算最低 peak：

```text
verify policy/build/artifact/source digests and expected sample membership
verify full-request coverage, all terminal states, quality and resource safety
derive M from confirmation raw only; preserve exploration raw separately
for each target T:
    eligible = confirmed execution + complete calibration + correct deployment
    feasible = [p in eligible where checked(M[p] + H(T)) <= T]
    if empty: emit unavailable with specific reason
    else: select frozen performance rank with deterministic tie-break
emit proposed catalog + validation report; never self-enable public
```

扩大 target 必须扩大或保持可行集合；catalog 验证器对相邻 targets 做单调性 property test。rank 以整请求 wall 为主，
不能按 denoise 最快值选一个 text/VAE 更慢的候选。性能无显著差异时保留更省内存/更少读量/稳定ID的候选。

`build_streaming_catalog.py` 分两种明确操作：

- `propose`：evidence→仅 proposed mappings，允许输出未支持目标的原因；不能调用运行时。
- `compile-reviewed`：已有 reviewer/review commit、完整证据、revocation/compatibility 都校验后生成 embedded table。

发布输入不接受未跟踪 arbitrary local JSON 作为 authority；生成器禁止执行 record 中的命令、脚本或任意动态库路径。
发布时 private/test records 不进入 public index；每条 public record 可追溯到不可变 evidence digest。

## 6. 性能验收：默认、框架成本、策略成本分开

### 6.1 必测对照

| 问题 | 实验 | 条件 |
|---|---|---|
| 关闭后是否伤害现有性能 | P0 before/after | 同默认 resident、原 legacy streamed、相同设备/shape/cache |
| 新抽象的额外开销 | P1 same-layout | 相同 P/G/K/D/Q/pool/reader/kernel/retention/lifecycle，actual逐项匹配 |
| 哪个目标选择哪个布局更好 | P4 independent finalists | 不同 layout/component policy，质量相同 |
| public resolver/query 成本 | P1 控制面分项 + full wall | preflight不能移到总计时之外；off audit仍零 |
| hard-memory guard的额外开销 | P2（后续独立） | 相同 layout guard off/on；不与本轮 calibration混淆 |
| 是否快于OS paging | P3（需另授权） | 两侧同机器/外部压力和完整输出，实际观察 paging |

P0/P1 使用文档12默认 median 2%/P95 5%非劣线及bootstrap置信区间，点估计在范围内不自动PASS。
各 workload/cache 条件单独签核，不能用LTX改善抵消Z回归。没有可信before build只能报告当前版本数据，不能宣称对dev零回归。

public 控制面测 `parse/profile`、`identity/tokenize`、`resolve/compile`、`resource transition`、`load/text/denoise/VAE/export/drain`；
分项有重叠不可机械相加。完整 wall 包含用户真实需等待的 preflight，App queue wait 单列；内存采样和release timing分别跑。

### 6.2 公共 UI 不能承诺无条件加速

用户选择低目标通常是在内存与I/O间交换；大内存无压力时 resident 可能最快。保持off作为默认，用户明确开启后不悄悄转resident。
目标16与20可共享一个preset，没有必要为了“高档更快”提高prefix。收益很小或尾延迟不稳定时优先稳定低读量方案。

同布局框架成本、相对resident策略成本、相对OS paging收益分别显示。Flux既有小型anchor的resident→streaming
MLX peak节省和wall增加只能作为该anchor的策略结果，不能推出App 512²/1024²的倍数或12GiB整请求可用。

### 6.3 Audit 的范围

独立audit应记录setup worker/pool/arrays和steady framework backing/thread counts；不把模型activation分配算成框架违约，
也不把pool重建藏进“模型allocation”。off新增hook/probe/worker/pool/cache-clear/unload为0。
Flux新prefix retain_all的前缀load只在setup；serial pool方案的稳态重分配如实报告并作为不同策略，不能通过改计数范围变成零。

sampler overhead以相同tuple有/无采样独立配对测试，初值目标≤2%；不达标降低频率或优化采样后重新freeze。
release timing永远不使用instrumented请求时间减估计开销来“校正”结果。

## 7. 为什么可能比 swap 快，以及怎样验证

### 7.1 机制判断而非结论先行

block streaming在访问顺序已知时能提前读取下一块，用固定K槽限制同时活跃权重；只读权重通常可以丢弃并在下一pass重读。
这里的“offload”主要指释放/复用backing，不必把刚用完的权重再写一份到磁盘；不要实现无意义的GPU→CPU→磁盘权重回写。
Apple共享内存路径也不能照搬独立显卡的每层PCIe复制模型。

但OS也可能直接回收干净的file-backed页，并不总发生匿名页swapout。streaming可能每pass重复读取大量suffix、挤压文件cache，
或者与GPU争用统一内存带宽；权重本就能驻留时可能更慢。text/VAE峰值主导时，减少denoiser slot也未必能避免paging。
所以“更可预测的驻留/预取”是设计意图，“更快”必须通过P3在声明条件下验证。

### 7.2 三层验证顺序

1. 无压力metadata/trace仿真：确定哪一阶段可能超物理余量，估计逻辑读取和暴露I/O等待；只作假设。
2. 专用低物理内存实机：同artifact/workload/default与preset，记录全stage/系统压力，先完整输出与安全，再比较时间。
3. 大内存机器受控pressure：只能作补充，需用户单独授权，有限内存/时长/stop/cleanup；不等价于同容量低端GPU。

本轮设计任务不启动pressure、不关闭swap、不purge系统缓存、不改系统VM配置。未来工具必须默认`launch_pressure=false`，
GPU benchmark授权也不自动包含系统压力授权。

### 7.3 P3 条件冻结

- 预注册压力来源/最大字节/触页方式/持续时间与恢复时间、最小磁盘/available floor、thermal stop。
- 两侧相同非模型压力，但不能动态把candidate省下的内存额外占满来“匹配剩余free RAM”。
- 记录物理内存、GPU/SSD带宽类、page cache状态、file-backed faults、compressed memory、swapin/out增量、物理读写与来源口径。
- baseline和candidate交替配对，上一侧安全退出、压力状态恢复后再开始下一侧；无界触页/盲目强杀不可作为默认清理。
- supervisor停止接新请求→请求取消→有限drain→记录隔离/必要进程终止；保护专用机器、输出和其他进程。

系统counter是全局，外部任务会污染归因；检测到污染按protocol标INCONCLUSIVE而非归因TurboCider。
如果baseline没有可观察paging，只能称同压力下OS-managed对照，不可称“战胜swap”。
baseline OOM/timeout时报告完成率和超时下界，不能用timeout秒数当精确wall算倍数。只有两侧都成功且质量通过的匹配样本
能报告conditional speedup，并同时显示全部失败率，避免幸存者偏差。

## 8. Host/API/App 验收矩阵

以下文件名为建议，新增后接入Makefile/build_app；不得把测试名称存在当已PASS。输入fixture用小型合成metadata，不下载真实权重。

| ID | 测试/注入 | 必须结果 | 建议位置 |
|---|---|---|---|
| PUB-CFG-01 | absent、v1、v2 off；profile on→request off | 原路径/原overlay；零catalog/probe/worker | `test_streaming_selector_contract.py` |
| PUB-CFG-02 | v2混stages/legacy/preset字段；unknown/duplicate key | typed reject，首个GPU allocation前 | 同上 |
| PUB-CFG-03 | target bool/负/零/小数/指数/string/2^53边界 | 无舍入，合法整数精确传递 | 同上 |
| PUB-CFG-04 | manual↔selector原子替换；profile残留legacy | 不拼接出混合配置；残留冲突拒绝 | 同上 |
| PUB-RES-01 | 同inputs/catalog键乱序重复resolve | 相同preset/layout/resolution，排序确定 | `streaming_preset_resolver_test.cpp` |
| PUB-RES-02 | T=10GiB,M=9GiB；M增加1byte | 前者fits，后者reject；整数margin精确 | 同上 |
| PUB-RES-03 | 增大T，两个targets同preset | 可行集合单调；不强制增大P | 同上 |
| PUB-RES-04 | token unknown→exact、不匹配shape/audio/container | tentative不可执行；精确匹配后授权 | 同上 |
| PUB-RES-05 | 没有模型/设备/版本/校准数据 | 具体不可用原因；unknown不当0 | 同上 |
| PUB-ID-01 | source同名替换、mtime/size变化、无verified manifest | invalidate/reject；snapshot不当content hash | `test_streaming_public_gate.py` |
| PUB-ID-02 | request伪造hash/authority/发布状态/catalog路径 | 不授权、不执行外部命令 | 同上 |
| PUB-ID-03 | replay旧record、撤回record、跨engine/model复用 | stale/revoked/model mismatch；不换preset | 同上 |
| PUB-ABI-01 | NULL参数、JSON异常、分配失败、重复调用 | 指针初始化/free正确，异常不穿ABI | `test_streaming_public_abi.py` |
| PUB-ABI-02 | generate中resolve/query/cancel并发 | busy或只读响应；query不重置generate cancel | 同上 |
| PUB-PLAN-01 | public generate_resolved | compile一次，实际消费相同snapshot | `streaming_resolved_request_test.cpp` |
| PUB-PLAN-02 | default generate与plan-only v2 | 默认无新preplan；plan-only不给authority | 同上 |
| PUB-APP-01 | 旧draft/jobs无新字段，decode/encode golden | 默认off，旧字段语义不变 | `StreamingChoiceTests.swift` |
| PUB-APP-02 | 混合v1/v2历史、输出删除/input引用/replay | 完整payload保留，managed-output安全不变 | `StreamingJobTests.swift` |
| PUB-APP-03 | Z旧6/8/10/12预算→开新target→关闭 | 旧值恢复，新/旧预算含义不混用 | 同上 |
| PUB-APP-04 | 慢query返回时切模型/prompt/profile | draft revision丢弃旧响应；不覆盖新状态 | `StreamingOptionsTests.swift` |
| PUB-APP-05 | resolve成功后jobs持久化失败 | 不启动GPU；可恢复错误 | `StreamingJobTests.swift` |
| PUB-APP-06 | 运行中取消/撤回、重启、engine quarantine | 正确终态和重建要求；无隐式resident重试 | 同上 |
| PUB-APP-07 | advanced关闭且打开设置/修改prompt | 不触发query/hash/tokenizer/sampler | `StudioBehaviorTests.swift`扩展 |
| PUB-REP-01 | MLX/Metal缺失、本次未采样、仅catalog estimate | null/范围/scope准确，不能显示0GiB | `RunInsightsTests.swift`扩展 |
| PUB-REL-01 | empty catalog / release无测试记录 | 普通public selector拒绝，default可用 | `test_streaming_public_gate.py` |

原有layout/executor/fake fence/streaming contract tests继续全跑；新API不能弱化原manual的精确值/拒绝行为。
test registry正向fixture与release空registry负向fixture必须是不同受控构建，普通请求不能选择test registry。

## 9. 模型专项验收

### 9.1 LTX

每个候选至少记录：prefix/group/slot/prefetch/worker、每 pass suffix fill 数、audio/VAE/export live order、source read bytes、
latent/video/audio parity。P8/G1/K3/D2/Q3 是现有锚点，不自动代表 8/10/12/16/20 任何档。

```text
LTX-01  P8 tiny: descriptor/layout/source identity
LTX-02  P8 512×320×33 11-step full output
LTX-03  P4/P6/P8/K2/K3 candidate matrix (E2 only)
LTX-04  97-frame full card and VAE/export peak coverage
LTX-05  cancel during fill, pass transition, audio finalization
LTX-06  resident→streamed→off mode transition and reuse
```

首个 public record 需要 E3 独立 prompt/seed、至少一组 mode transition 和完整 execution-tree coverage；仅 denoise/warm P1 不够。

### 9.2 Z-Image Turbo

保持 original BF16、K2/G1/D0/Q1；候选 P0/4/8/12/14/18/22/26/28 按24的合法性和目标邻近原则筛选。
记录 text rows、9-step denoise、VAE、PNG、释放策略；legacy 采样预算另测，不纳入新 target 的 proof。

```text
ZIMG-01  P0/P14 metadata and P29 reject
ZIMG-02  512² 9-step full request
ZIMG-03  1024² 9-step full request
ZIMG-04  prompt token 32/512/1024 boundary
ZIMG-05  VAE release policy revision and mode transition
```

1-step anchor 只能作 executor P1/diagnostic；新 public record 需9-step full request。GGUF、不同 tokenizer 或 text >1024 的请求应保持明确不可用。

### 9.3 H3 Turbo

只覆盖 MiniMax H3 Turbo original BF16；记录 frames `5+17n`、22..362、Qwen3 text、DiT、video VAE、export。保留大矩阵 pread，
不能用较慢小块读替代。当前 denoiser probe 和 bootstrap INCONCLUSIVE 都不能升级成 public statistics pass。

```text
H3-01  original BF16/source identity; other H3 variants reject
H3-02  512²×22 4-step full video
H3-03  39/73 frame full video and VAE peak
H3-04  odd suffix/carry boundary and pass transition
H3-05  cancellation during pread, DiT, VAE, export
H3-06  public authority reaches h3_session; test env alone does not
```

H3 档位是否 public 取决于 full-pipeline memory/quality/lifecycle；工程上接受当前固定 tuple 的合理 I/O 波动，不等于免除这些条件。

### 9.4 Flux 9B

先用 P0/G1/K2/D1/Q2 作为 private/reference，明确当前 direct executor 与 generic executor 的 same-layout 适用范围。
新 prefix 需完成 P2/4/6、P7 reject、P8、P12 的 host/Metal/PNG/fill/pool 验收。记录 dual/single pool 数、prefix source bytes、
concatenate count、release-before-vae 和 retained/serial policy。

```text
FLUX-01 P0 existing private tuple, exact output and source identity
FLUX-02 P2/P4/P6 prefix dual-only, no duplicate concatenate
FLUX-03 P7 reject with K2 suffix grouping
FLUX-04 P8 cross-class boundary, no empty dual pool
FLUX-05 P12 prefix includes first single, suffix starts at 12
FLUX-06 retain_all vs serial pool lifecycle and allocation audit
FLUX-07 cancel/fill error at dual→single boundary
FLUX-08 512²/1024² full 4-step memory and P4 strategy comparison
```

Flux 的 existing P1 `0.99971` wall / `0.99816` denoise 只说明冻结 P0 同布局 framework 开销；不能用作 P2/P8 或 12 GiB target 的证据。
新 prefix 若无法维持 same-layout direct reference，应标策略 P4 或 NOT_RUN，而不是回避比较。

## 10. 校准工具与失败清理的反例验收

### 10.1 CAL：独立 verifier 必须拒绝的证据

建议新增 `tests/native/test_streaming_memory_calibration.py`、`test_streaming_catalog_builder.py`，用 fake worker/clock/PID
和小型 JSONL fixture，无真实压力、无大 allocation。verifier tests 不能只测 happy path：

| ID | 合成输入 | 必须结果 |
|---|---|---|
| CAL-01 | 同worker先高峰后低峰，ru_maxrss维持高值 | 两个request采样窗分开；lifetime字段不作低峰证据 |
| CAL-02 | root/helper峰值发生在不同时间 | 同时tree求和；不把各自峰值之和称观察值 |
| CAL-03 | helper未登记、采不到、PID复用、exec切换 | 不错配identity；missing coverage不能发布 |
| CAL-04 | footprint=null，RSS/MLX存在 | 主域不填0、不换口径凑总量 |
| CAL-05 | construct前缺采样、cleanup缺终态、sample gap/skew超限 | incomplete，明确缺失阶段 |
| CAL-06 | MLX reset epoch变化，counter overflow/negative/NaN | 分epoch解释或拒绝，不能减出负峰值 |
| CAL-07 | 顺序颠倒/重复sequence、buffer drop、最后一行截断 | 明确invalid/interrupted；恢复可读前缀但不当完整 |
| CAL-08 | 高峰请求failed，其他请求成功 | 保留失败；不能删除高峰后分到低档 |
| CAL-09 | policy/source/workload/build/request digest不匹配 | reject，不能只信summary passed |
| CAL-10 | cache baseline不同，maxF−maxB产生虚假低增量 | 只允许每样本配对增量；不影响绝对M |
| CAL-11 | 自报physical reads=0但source reads>0 | 合理cache情形；两域分开，不抹掉logical budget |
| CAL-12 | resume累计请求/读量/时间超limit | 停止新请求，状态budget_exhausted，不重置预算 |
| CAL-13 | 新增较高T、同一个preset覆盖多个T | feasible集合单调，candidate去重，只测一次身份 |
| CAL-14 | null evidence、伪造review、proposed/test record | proposal可留未支持原因，compile-reviewed拒绝public |
| CAL-15 | 只修改JSON键顺序/manifest生成顺序 | canonical digest稳定；修改layout/component则变 |
| CAL-16 | source bytes缺text/VAE，只有denoiser峰值 | scope partial，不允许full-request tier |

### 10.2 PUB-FLT：运行期生命周期故障矩阵

在现有 `streaming_executor_test.cpp`、C bridge failure tests、pager和模型lifecycle tests上扩展；每项既检查错误，也检查
已创建资源的最终处置。真实 GPU fence故障使用独立test-hook构建和专用worker，不在release注入。

| ID | 注入点 | 必须结果 |
|---|---|---|
| PUB-FLT-01 | 第n个resident/prefix/pool分配失败 | primary error保留；已分配backing只在无reader时释放 |
| PUB-FLT-02 | short read/EINTR/EOF/文件改动/worker异常 | 合法EINTR重试；其余停止新dispatch，不能使用半填slot |
| PUB-FLT-03 | cancel于prefix、fill、claim、encode、class/pass边界 | 无后续新工作；所有reader与callback结束后释放 |
| PUB-FLT-04 | reader回调立刻完成、晚到、旧generation、重复完成 | ticket/seal合同成立，无early overwrite/ABA |
| PUB-FLT-05 | drain timeout/MLX同步失败/Metal completion异常 | quarantine保持snapshot/pager/mailbox，禁止后续generate |
| PUB-FLT-06 | callback引用source/plan，外层resolve对象先销毁 | request owner维持依赖生命周期，无UAF |
| PUB-FLT-07 | VAE/export失败、输出磁盘满 | 不标success，不自动换preset；保存峰值与cleanup状态 |
| PUB-FLT-08 | cancel后立即同engine请求/切off | safe drain后才允许；隔离engine明确拒绝/重建 |
| PUB-FLT-09 | worker被中断、coordinator退出、IPC断开 | bounded停止协议，保留interrupted记录，不留无主实验 |
| PUB-FLT-10 | 模式切换后allocator/cache policy残留 | off恢复原配置；不保留错误cap、authority或后台worker |

成功terminal之后检查：active reader=0、inflight fill=0、callback可达对象无悬空、executor pool/prefix按retention释放、
新增线程终止。allocator cache不必物理归零，必须区分保留cache与live backing；不能在默认路径加clear_cache使测试“好看”。
隔离terminal不要求假装资源已释放，需列 retained bytes/objects、后续动作与最大等待；不能把quarantine当成功cleanup。

### 10.3 工具自身停止和持久化

日志达到空间上限/磁盘不足时停止新请求，已有请求走安全取消；主错误和日志错误分别记录，不覆盖原始I/O/模型错误。
request原始记录在下一请求发出前持久化，避免只剩汇总；memory高频流可批量flush，但文件尾损坏必须被verifier识别。
停止最多等待冻结deadline，不能因为统计未收口继续读TB级权重。未完整写入的bundle不可由catalog builder采纳。

## 11. Public 发布和回滚清单

### 11.1 Staging gate

发布前 CI/人工 checklist：

- [ ] release build 中 test authority/candidate constructor 不可由 App 引用；strings/symbol audit 通过。
- [ ] embedded catalog 仅包含 reviewed records；未完成/撤回/illustrative ID 不进入 public index。
- [ ] catalog source、生成表、review commit、evidence digest 和 runtime build identity 一致。
- [ ] `make test-streaming-host`、`test-streaming-contract`、`test-streaming-campaign`、`test-streaming-source-identity`、
      `test-streaming-audit`、`test-streaming-pager` 通过；需要 Metal 的目标在真实机执行或明确 SKIP。
- [ ] 新 selector parser/resolver/C ABI/Swift/App 测试全部通过；历史 job golden 与 Z legacy tests 不回归。
- [ ] 每条 record 具备模型专项 L0/L1/L2、P0/P1/P4（如适用）、完整 memory calibration 和独立 confirmation。
- [ ] 未完成的 bounded guard/P3 明确显示为 none/unsupported，不被 public 选择流程暗示。
- [ ] App default off 在新鲜安装、升级旧 jobs、切换模型和无 GPU 时行为与旧版一致。
- [ ] UI 不暴露 P/G/K/D/Q；details 仍可审计 actual layout 和 evidence revision。

### 11.2 发布最小范围

先发布少量 exact records（模型×artifact×workload×设备/container），不追求五个 target 全覆盖。
建议状态名为“Streaming（实验）— 已验证内存档位”；只有在多批独立数据和稳定 P0/P1/P4 后才升级 stable。
缺少本机设备校准时显示“此设备暂无已验证档位”，不要泛化到同品牌所有 GPU/RAM。

### 11.3 撤回和回滚

出现质量差异、source/reader错误、峰值越界、异常失败率或性能回归时，优先把对应 record 标 `revoked` 并发布 catalog revision，
不修改旧 record 语义、不让 resolver 自动挑另一个 layout。App 重新 query 后显示撤回原因；历史已完成 job 可查看结果，重跑需重新 resolve。

catalog rollback 必须是原子 embedded resource 替换；runtime、App、CLI/service 同步使用同一 revision。若 public gate wiring 有问题，
可以把 public index 置空，保留 private/manual 测试路径；不能通过关闭所有 streaming validation 来“紧急恢复”。

## 12. 代码验收、证据和交接工件

每个实现 PR 至少提交：

```text
docs/design/streaming 更新（已实现/待实现/限制）
build-identity.json
catalog-source.json 或空 catalog 声明
compiler/resolver golden report
unit/fault/race test output
native/API/Swift schema examples
若涉及 GPU：quality.json、raw samples、memory calibration、performance report
```

证据目录建议：

```text
results/streaming/public/<model>/<catalog-revision>/<workload>/
  policy.json
  environment.json
  source-identity.json
  raw-samples.jsonl
  memory-samples.jsonl
  quality.json
  performance.json
  verifier-report.json
  review.md
```

结果中使用相对逻辑 artifact 名和 digest；不要提交模型权重、视频/图片大文件、个人路径、凭据或未经授权的系统 dump。
`review.md` 必须说明 sample membership、排除策略（默认无 post-hoc exclusion）、coverage、未支持分支、
P0/P1/P4/P3 结论和下一步，不用一个“PASS”字段掩盖 INCONCLUSIVE。

## 13. 可追踪的实施顺序与完成门

S0–S10是24中MT批次的代码子任务，不是新的资格级别；目前全部是待实施设计。可按表独立拆PR，不能在基础PR偷偷附带public record。

| 子任务 / 对应MT | 依赖 | 代码交付 | 完成门与可合并范围 |
|---|---|---|---|
| S0 / MT-05 | 无 | selector contracts；`request.mm/profile.mm/streaming_config.mm`严格解析 | PUB-CFG；v1 golden不变；public仍拒绝 |
| S1 / MT-05 | S0类型 | catalog schema/index、builder、test fixtures | PUB-RES的catalog部分/CAL-14/15；production空表 |
| S2 / MT-05 | S0/S1 | resolver、snapshot、metadata token/source接口、plan-only | PUB-RES/PLAN/ID；无GPU分配；不改旧make_plan权限 |
| S3 / MT-05/06 | S2 | C header/API、Swift typed payload/query/error消费 | PUB-ABI；老symbol/serializer不变 |
| S4 / MT-06 | S3；mock query可先行 | draft开关、Job Codable迁移、JobStore、Insights、UI | PUB-APP/REP；default off；test records不随App发布 |
| S5 / MT-05 | S2/S3 | `generate_resolved`、authority、session owner、撤回检查 | PUB-PLAN/ID/FLT/REL；实际释放与quarantine验证 |
| S6 / MT-01/02 | 可与S0并行；复用当前private route | calibration worker/schema/verifier、candidate inspect | CAL全套；fake先过；默认路径零观察器 |
| S7 / MT-03/07 | S5/S6 | LTX/Z source和component接线、full cards、独立确认 | LTX/ZIMG、P0/P1/P4；仅产review candidate |
| S8 / MT-03/07 | S5/S6 | H3 Turbo full-pipeline；session级public入口 | H3、quality/memory/取消；INCONCLUSIVE如实保留 |
| S9 / MT-04/07 | S6；public验证需S5 | Flux prefix pager/边界helper/pool消除/reference | FLUX、pager/fault、P0/P1或NOT_RUN、P4；旧P0不回退 |
| S10 / MT-08 | 至少一模型S7/S8/S9合格 | compile-reviewed catalog、App/CLI/service staging验收、撤回演练 | 独立review后仅开放覆盖tuple；无需等所有模型/目标 |

资源闭包/hard-memory guard仍属MT-09，不隐藏在S6经验校准里。S6可先做，尽早回答text/VAE floor是否使低档不可能，
避免等UI写完才发现8/10GiB不可行。S9 prefix是额外优化，不阻止现有Flux P0获得其证据覆盖的较高目标。

每一门的停止条件：

- contract/resolver 任一 fail → 不进入 GPU；
- adapter/quality/lifecycle fail → 不进入 memory calibration；
- calibration incomplete → 不进入 catalog proposal；
- independent confirmation inconclusive → 保持 experimental/not-public；
- P0 default regression → 阻止合并并修复/回滚代码；关闭public index不能修复已经污染off路径的开销；
- release 后单 record 异常 → 撤回 record，不改其他模型/档位。

### 13.1 现有命令与待新增目标

下列是已存在的入口，后续实现可用来验证没有破坏旧合同；本轮文档编辑未执行这些runtime测试：

```sh
make test-streaming-host
make test-streaming-contract
make test-streaming-campaign
make test-streaming-source-identity
make test-streaming-audit
make test-streaming-pager
```

contract需要已构建native库；pager/Metal相关测试需要相应依赖和GPU权限，缺失时SKIP不是PASS。
真实模型测试另按已存在的policy/文档12执行，不能通过`make test`隐式下载权重或启动pressure。

实施后建议增加的目标（**当前不存在**）：

| 拟议target | 内容 | 默认可在无权重环境运行 |
|---|---|---:|
| `test-streaming-public-host` | CFG/RES/catalog/digest/golden | 是 |
| `test-streaming-calibration` | fake worker/time/PID/CAL/verifier | 是 |
| `test-streaming-public-api` | tiny fixture C ABI、门禁、ownership | 需native build，无大模型 |
| `test-streaming-public-app` | Swift Codable/legacy job/query/Insights mocks | 需Swift/App test build，无模型 |
| `test-streaming-public-model` | 显式MODEL/卡片/预算/full GPU | 否；必须opt-in |

新增目标要接入CI并保留真实stdout/exit status。GPU inaccessible、缺weights或权限受限时测试明确SKIP并保持发布阻断，
不能以host tests全绿替代model/内存/性能验收。

## 14. 当前状态记录方式

当前已有测试和性能锚点仍集中在 [13](13-implementation-progress.md)、[21](21-review-and-handoff.md) 和 [22](22-current-framework-guide.md)。
本文件新增的 PUB/CAL/MODEL ID 是后续实现的验收编号，不表示测试已经存在或通过。新增测试文件落地后，把建议名替换为真实路径并把
CI 命令写入对应表；没有真实 artifact 的条目保留 `NOT_RUN`。
