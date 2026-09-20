# 07 · 工具链、仿真、swap 对比与验收

[目录](README.md) · [预算合同](05-memory-contract.md) · [状态](08-status-and-history.md)

本文定义实验原则与工具输出；具体 P0–P4 gate、统计判定、测试 ID 和发布执行顺序统一在 [12](12-acceptance-playbook.md)。

2026-09-20 口径补充：12 保留早期 gate 定义；当前 public P2 指完整请求内存校准，实施入口见 [56 第 5、8 节](56-production-readiness-review-and-acceptance-plan.md)，正式工具证据合同参见 [50](50-public-streaming-calibration-and-release-evidence.md)。56 的 calibrated release policy 尚未实现，不能当作现有 builder 已允许跳过 P3。
当前 campaign 只接受 `artifact_sha256_equal`。Hybrid 的迁移正确性、相对 exact GPU 的近似质量、same-plan 性能与完整请求收益必须分别判定；拟议新 quality mode、逐样本规则、独立 verifier、H 包测试矩阵见 [57 第 7–9 节](57-gpu-ane-integration-and-delivery-plan.md)。禁止为支持 ANE 而放宽旧 GPU exact policy。

## 1. 先有可解释工具，再有自动调参

拟议统一工具 `tools/native/streaming_tool.py`（当前不存在），子命令如下。Python 负责输入/报告编排，
通过小型 native plan utility 调用同一个 C++ descriptor/compiler，不复制第二套 layout 算法。

| 子命令 | 输入 | 输出 | 会运行 GPU 吗 |
|---|---|---|---|
| inspect | model metadata/checkpoint identity | stage/layout class/支持的参数/格式 | 否 |
| plan | 显式配置、shape、descriptor | resolved groups、pool bytes、whole-request upper/unknown/reject | 否 |
| simulate | resolved plan + calibration | 预测 trace、瓶颈、敏感性区间 | 否 |
| sweep | 用户指定有限 K/G/P/D/Q 集 | 候选排序和未执行的建议配置 | 默认只仿真；真实运行需显式参数 |
| benchmark | 实际支持的 layout + 对照 | 冷/热 timing、quality、slot/fence、memory、swap | 是 |
| verify | 原始记录/evidence bundle | PASS/FAIL/SKIP/INCONCLUSIVE + 原因 | 否 |

默认 inspect/plan 不 hash 全权重、不分配 GPU；需要内容验证时显式输出 validation cost/status。
输出 plan-only 不应改变 session residency 或清 cache。recommend/sweep 只生成建议文件，不能自动回写用户 profile。

## 2. 工具输出标准

```text
input.json                    原始意图、shape、seed、sampling、模式
descriptor.json               格式/layout/resource 描述和 identity
resolved-plan.json            exact groups/K/G/P/D/Q + digest + capacity
trace.jsonl                   owner sequence + ticket + backend timestamps
memory.json                   managed/reserved/pending/framework/observed 峰值
performance.json              冷/热耗时、I/O/GPU、质量与对照信息
environment.json              GPU/物理RAM/OS/SDK/backend/存储/缓存/温度状态
manifest.json                 bundle 文件名、字节数、hash、验证器版本
verification.json             每条 gate 的结果，不能由 candidate 自报 PASS
```

证据类型：model_only / synthetic / real_gpu_layout / bounded_l2 / bounded_l3 / release_reviewed。
测试 build 输出 synthetic record 永远不能被生产 loader 自动发现。
trace 必须有限容量或向预留文件流式写出，日志本身占用内存/I/O需算入测试与成本；trace overflow 的认证 run 不通过。

## 3. 离散事件仿真器

输入不是一个固定 `swap_penalty_ms`：至少包括每个 group 的文件 ranges、logical bytes、cache-hit 参数、read/decode/bind/compute 服务时间、
queue 数、worker 数、slot mapping、真实 fence 延迟分布，以及 activation/framework timeline。
文件缓存、SSD 带宽、CPU 解包、GPU memory bandwidth 分开建模，先给无争用基线，再加入同机器校准的竞争曲线。

事件：reserve pool、enqueue read、read start/end、decode done、ready accepted、compute submit/start/end、fence consume、reuse、stage drain。
模拟器执行与 runtime 同一资源约束；不能只按 sleep duration 相加。
相同输入/seed 生成相同 logical timeline；随机参数仅用于延迟敏感性，不改变安全判断。

对单读资源、串行 compute 的简化模型，设 load_i 包含 read/decode/bind：

```text
load_start(i) = max(reader_available, slot_reusable(i), release_time_from_lookahead(i))
load_end(i) = load_start(i) + load_i
compute_start(i) = max(compute_end(i-1), load_end(i))
compute_end(i) = compute_start(i) + compute_i
slot_reusable(i+K) >= compute_end(i) + relevant_fence_latency
```

忽略 fill/drain/fence/竞争、每组耗时相同时：串行约 n(L+C)，理想双缓冲约 L+C+(n−1)max(L,C)。
这只是计算下界，不是预测保证。第三槽可以吸收抖动、减少两槽中的 dispatch gaps，但不能把慢盘的持续吞吐变成快盘。
增加 K 在固定物理内存下还可能挤掉 file cache；不同 K 的 L 不能假设完全不变。

验收至少包含 K=1 serial、K=2 overlap、K=3 jitter、G>1、异构 groups、pool OOM、取消、长尾 fence。
没有真实参数时报告 sensitivity/model_only，不能拿人为设定的 swap 惩罚宣布加速。

## 4. Streaming 与 swap 的正确比较

系统 VM paging、文件 clean-page 回收、匿名页压缩、swap 文件 I/O 不相同。不可把每次 checkpoint read 都算成 swap，也不能假设
resident 超 RAM 的全部 bytes 都会立即写 swap。共享物理内存的模型搬到 CPU 副本仍会占 RAM。

对 VM baseline 的仿真用明确标注的 reference trace + LRU/工作集敏感性模型，区分 file-backed/anonymous/dirty 页；
它不是 macOS pager 仿真器。分别改变 cache hit、可用物理池、compression ratio/CPU cost，找结论在哪些区间反转。

需要检验的假设，而不是预设的结论：

- 如果所有资源 resident 且不分页，streaming 通常增加 read/bind，不能承诺更快。
- 如果可重建 weights 占满内存并出现反复换页，显式按未来访问顺序预取可能减少等待和尾延迟。
- 如果 checkpoint 大多命中文件缓存，历史 warm read 秒数不能用于预测真实低内存 SSD 成本。
- 如果解包/内存带宽主导，增槽可能让 compute 更慢；Q=1/2 必须测竞争。

只批准 document/simulation 的任务不运行全系统 pressure，也不关闭 swap、不调系统 VM 参数、不无界申请内存。
真实 pressure baseline 需在专用测试机由操作者显式启动，设字节上限、最长时间、free-space 和系统压力停止条件。

## 5. 配对实验设计

### 5.1 拆开两种实验

默认回归：A=修改前 legacy，B=修改后同一 legacy；checkpoint/shape/dtype/seed/cache/编译优化一致，交错 ABBA。

框架成本：A=旧手工布局，B=新框架同一 K/G/P/D/Q 和 retention。先不改变参数；否则不能把差异归于 framework。

策略效果：A=正常默认路径，B=显式 streaming，C=专用压力条件下的默认 OS-managed 路径。
C 不强加虚假的进程硬 cap；B/C 在相同物理机器和可用内存条件下比较，同时另报 B 是否满足 Y。
pressure 试验独立批次运行，每批结束等待内存/温度恢复至记录基线，不把 C 直接夹在零回归 ABBA 中污染结果。

### 5.2 冷/热与样本数

- process-cold：新进程，不代表 OS 文件缓存冷；必须记录 file-cache 状态未知/热/按协议冷。
- warm process 与 retained model 是两类条件；首版 request retention 不能拿默认 retained model 的 warm 时间冒充同一条件。
- 3 cold + 5 warm 只做 smoke；不据此声称 P95/P99 稳定。
- 初次性能评估建议至少 20 个 matched pairs；正式 tail campaign 建议至少 50 次有效请求并给区间，P99 样本不足则不发布结论。
- 长视频代价高时预先声明样本预算和无法证明的分位数，不用低样本“PASS”代替证据。
- 同时报告热降频、外部 I/O、后台内存压力和失败/截尾样本；不能只保留成功快样本。

### 5.3 指标定义

```text
speedup = baseline_wall / candidate_wall
exposed_io_wait = owner/GPU关键路径等待必须数据的时间（区间去重）
io_hidden_fraction = duration(union(IO intervals) ∩ union(GPU compute intervals))
                     / duration(union(IO intervals))
gpu_busy_fraction = duration(union(GPU compute intervals)) / request_wall
pool_reuse_count != backing_allocation_count
```

host encode 时间不能冒充 GPU compute；并行 worker 耗时不能简单相加当 wall 或 overlap。
logical read bytes、物理 disk bytes、swap bytes 分开报告；无法归因的 disk counter 标记 system-scope。
峰值至少有 managed、reserved、pending、sampled process、framework upper、system available、swap delta，并注明观察盲区。

## 6. 自动验收矩阵

| 层级 | 必测项 | 失败判定 |
|---|---|---|
| Config | 缺省/false/0、profile覆盖、legacy冲突、unknown/duplicate/type | 任意 silent coercion/布局变化 |
| Compiler | groups/classes、tail、P、精确 K、capacity/overflow/digest、resident K0 | 二次隐藏规划、未知 required envelope 放行 bounded |
| Pool | backing一次分配、多次content generation、no early overwrite、多queue fences | fake backend记录 fence前写入即FAIL |
| Fault | short read、cancel、thread失败、late/duplicate/stale completion、mailbox满 | use-after-free、账本提前归零、poison后继续 |
| Adapter | 模型事件独立期望序列、actual==resolved、disabled callback=0 | 框架自造event被当成真实kernel证明 |
| Real GPU | 固定 seed parity、completion、live storage、cleanup、多次请求 | 无设备只能SKIP，不能通过synthetic代替 |
| Bounded | 每阶段whole-request upper、sites闭包、swapouts=0、低内存实机 | 超限/观测不可信/unknown required 一项即FAIL |
| Default performance | resident及已有streamed ABBA、cache/线程/command数 | 新增probe/unload是硬失败；性能不等价则不发布 |

纯调度/数据搬运预期同输出；若 kernel 精度不变但平台存在非确定性，预先固定 latent/RGB 等数值容差和质量门，
不能看完结果再放宽。tiling 不是天然无损，需要模型自己的 parity 验证。

## 7. 默认性能发布目标

关闭新 framework/guard 的新增 probe、hook、pool allocation、worker thread、cache-clear/unload 次数=0。
原来已存在的设备查询、cache clear、等待不算“新增”，不顺手删除。
建议 wall median regression 上限 2%、P95 上限 5%，作为预注册非劣目标；区间跨过门槛则继续采样或 INCONCLUSIVE。
这不是“低于2%就允许新增全局probe”；行为隔离和统计非劣都要通过。

streaming 优化用同布局对照和真实低内存条件评估，不给所有模型统一加速承诺。
bounded guard overhead 与布局 I/O overhead 单独 A/B；不能以 guard=false 成绩替代 bounded 成绩。

## 8. 现有可执行检查（不是未来工具）

```sh
python3 -B tests/native/test_memory_schedule_adapter.py
python3 -B tests/native/test_memory_execution.py
python3 -B tests/native/test_memory_plan_compiler.py
python3 -B tests/native/test_contract.py
python3 -B tests/native/test_h3_gpu_memory_hooks.py
python3 -B tests/native/test_ltx_gpu_memory_hooks.py
```

它们验证当前底座，不验证本文待实现的 slot framework。不要把 `test_ltx_streaming_benchmark.py` 的 metrics 单元测试
当成真实模型速度测试。新增通用编译器/slot tests 在 F1/F2 中实现，命令完成后再加入本节。
