# 12 · 验收执行书：正确性、性能、内存与发布

[目录](README.md) · [工具与实验原则](07-tooling-and-validation.md) · [实施任务](11-work-packages.md)

状态：拟议验收规格。本文阈值是开发前预注册的发布目标，不是已有测量结果，也不是承诺任意低内存配置都比 resident 快。
首要目标：默认性能不退化；新框架成本可控；低内存布局在同等资源/质量条件下可证实有用。

## 1. 五类问题用五类实验回答

| ID | Baseline A | Candidate B | 可以得出的结论 |
|---|---|---|---|
| P0 | 改动前 legacy | 改动后同 legacy | 是否伤害默认/原有 streamed |
| P1 | 旧 executor 的明确布局 | 新 executor 同布局与资源策略 | framework/executor 实现成本 |
| P2 | 新 manual guard off | 同 layout guard on | guard 增量成本；clean-boundary 差异另报 |
| P3 | 同机器默认 OS-managed | 新 streaming，匹配外部压力条件 | 指定条件下比 VM paging 更好/更差 |
| P4 | 新已验证 K/G/P/D/Q 组合A | 另一显式组合B | 策略选择收益，不归因于框架重构 |

P0 必须在大内存/no-pressure resident 和已有 streamed 上分别通过。P3 的收益不能抵消 P0 回归。
同布局不仅是 K 一样：prefix、group、lookahead、I/O 并发、格式、pass、数值优化、retention 都必须核对。
P2 中 guard 所需的额外 clean boundary 属于用户可见成本，既报告 total，也另报相同生命周期执行段成本。

## 2. 建议的发布门槛与统计判断

| Gate | 时间指标的非劣界（候选/基线） | 附加硬门 |
|---|---|---|
| P0 默认 | end-to-end median ≤1.02；P95 ≤1.05；denoise median ≤1.02 | 新增 hook/probe/thread/pool/clear/unload=0 |
| P1 同布局 | end-to-end 与 denoise median ≤1.03；P95 ≤1.05 | kernel/format一致，steady framework alloc/thread-create=0 |
| P2 guard | 同执行段 median ≤1.05；P95 ≤1.10 | 守卫覆盖不可删减；总 wall 单列，不能只公布执行段 |
| P3/P4 优化 | 不统一承诺百分比；加速声明要求 wall speedup 的95%区间下界>1 | 同质量、失败率不增加、预算/安全通过 |

这些上限是发布阻断线，不是鼓励用满的额外开销预算。默认仍以无可检测回退为目标。
单个发布 workload 超限不得用其他更快 workload 的平均值掩盖；按 model/shape/cache condition 分层签核。
有明确业务更严格要求时可在 campaign 前降低上限，不能测完后调宽。

判定规则：

- 预先选择 statistic、bootstrap 方法、样本上限、noise exclusion 条件和 candidate 集。
- 使用 matched ABBA blocks；按整个 block 重采样保留配对和时间相关性，不把每个 denoise step 当独立请求样本。
- 对 candidate/baseline 统计量比值计算 95% interval；上界≤门槛才 PASS；下界>门槛则 FAIL；其余 INCONCLUSIVE。
- median 使用各条件 request wall 的中位数比；P95 使用预注册 quantile estimator 的分位数比，二者均通过 block bootstrap 取区间。
- 20 matched pairs 是首次分析起点，不保证足够；tail 至少50次有效请求仍可能不够。P99 不给小样本结论。
- 不反复看结果直到偶然 PASS；预注册一次最终分析，或使用明确的序贯检验方法和停止边界。
- 多 tuple 搜索先用探索集；选定后用独立确认集。需要同时推断多个性能指标时预注册 multiplicity 处理。

若 runtime 优化值得发布但 P1 未通过，只能保留 experimental，修复瓶颈后再测；不能改称“内存省了所以性能通过”。

## 3. 固定 workload 卡片

每个 model/backend/format 至少准备以下卡片。具体 checkpoint hash、shape、steps、seed 必须在执行前填实，空值不允许认证：

| 卡片 | 目的 | 可替代真实目标场景吗 |
|---|---|---|
| tiny-smoke | 快速 parity、错误定位、全 trace | 否 |
| normal-target | 实际常用图片/视频 shape、steps | 是，限本卡片 |
| largest-certified | 发布支持范围边界，activation/VAE峰值 | 是，限本边界及明确证明范围 |
| repeated-request | 连续成功、取消后重试、shape A→B→A | 覆盖生命周期，不替代单次性能 |

LTX 至少含两 stage/upsample；H3 包含 text/DiT/VAE/export。Flux/Z-Image 先以 component candidate 为单位，不虚构 block streaming 成绩。
audio/I2V/LoRA/不同量化格式分别列 supported/unsupported；未运行不默认覆盖。
首版 exact tuple registry 不靠“测试最大 shape 所以所有小 shape 安全”自动扩范围。

## 4. 构建与环境封存

1. 封存 before/after 源码 identity：commit + dirty diff hash + untracked source 清单/hash + 编译器/flags。
2. baseline 从独立已保存构建或隔离 worktree 生成，不对用户 dirty tree 执行 reset/checkout。
3. checkpoint/adapter/backend/OS/SDK/GPU/RAM/SSD/电源和 thermal 状态入 `environment.json`。
4. 先验证 quality，再启动长性能批次；ASan/TSan/详细 trace 构建与 release timing 构建分开。
5. P0/P1 的两侧 instrumentation 一致；另跑 audit build 检查新增行为=0，避免 audit 自身扭曲正式 timing。
6. 参数不从可用内存动态变化；保存 requested/resolved/actual 和所有 digest，actual 不一致直接失败。

没有可信 before build，只能报告当前候选之间对比，不能宣称“对改动前零回归”。

## 5. 冷热与时间范围

- `process_cold`：新进程；记录 OS cache 为 unknown/warm/protocol-cold，不把冷进程叫冷 SSD。
- `request_warm`：复用进程，但首版新框架 request pools 仍释放；prefix/slot 重载计时。
- `retained_model`：独立条件。不能拿旧 retained baseline 对新 request retention 后声称纯 framework 开销。
- wall 从服务接受执行/排队后执行起点按预注册定义计到可用输出+必要 terminal cleanup；queue latency 独立记录。
- 同时报告 preflight/load/text/denoise/upsample/VAE/export/drain；合计与 wall 重叠关系注明，不能简单相加并行区间。
- 冷启动 first load、失败 cleanup、未成功的超时请求均保留，不只算 denoise 或删慢样本。

## 6. L0–L3 正确性与资源验收

此处为新框架统一命名；不把归档中同名层级的历史通过迁移成当前通过。

| 层级 | 内容 | 通过证据 |
|---|---|---|
| L0 | schema/compiler/golden/fuzz/fixture | 无 GPU host tests；独立 reference verifier |
| L1 | fake I/O/GPU/多 queue/race/fault/property | fence前读内容校验、bounded queues、无泄漏/死锁 |
| L2 | 真实 GPU 质量/lifetime + 纳管预算/whole-request upper | checkpoint运行、真实completion、allocation/site审计、观测可靠 |
| L3 | 专用低内存实机、no-swap campaign、长请求/系统余量 | 全阶段资源与系统观察、失败/停止记录、同条件baseline |

layout_validated 要求 L0/L1、L2 中质量/lifetime/cleanup部分以及 P0/P1；不需要假装自己已有全局 cap。
bounded_certified 要求完整 L2/L3、P2 和 whole-request required-site closure。P3 胜过 swap 不是内存安全认证的必要条件。

## 7. 可直接分配的测试 ID

| ID | 输入/注入 | 必须观察到 |
|---|---|---|
| CFG-01 | 缺省/false/profile true→request false | legacy path；新对象/线程=0 |
| CFG-02 | duplicate/unknown/bool-as-int/overflow/旧字段冲突 | 精确 error，无卸载和分配 |
| CMP-01 | compiler-golden fixture | groups/capacity/read bytes/digest逐项一致 |
| CMP-02 | field大小反转、alias、class barrier、K>groups | 正确unique bytes或明确拒绝 |
| CMP-03 | 同layout换Y、JSON键乱序 | layout digest不变；admission独立变化 |
| RUN-01 | K1、K2、K3；D0和K−1；Q1和K | 工作项次数/顺序不变；实际并发不越界 |
| RUN-02 | read乱序、GPU长尾、多queue最后reader迟到 | 无early overwrite；无device-wide等待替代 |
| RUN-03 | callback提交后立即到达、generation溢出/旧回调 | seal前不复用；stale拒绝；generation不回绕 |
| RUN-04 | 多pass/class边界/shape变化 | 明确重读与drain；无跨request旧内容命中 |
| FLT-01 | 第n次alloc/thread-create/read/submit失败 | primary error保存，已创建资源可安全清理 |
| FLT-02 | 每种状态cancel、mailbox满、drain timeout | 不再派发，安全回收或quarantine |
| MEM-01 | upper=B、B−1、baseline变化、pending release | 同布局允许/拒绝，无偷偷减K/P |
| MEM-02 | 未知required envelope、swap counter失效/增长 | bounded fail-closed，不能success |
| GPU-01 | 固定输入/seed/kernel的latent与输出对照 | 预注册quality门；真实completion/cleanup |
| DEF-01 | 默认resident/旧streamed repeated run | 新hook/probe/thread/cacheclear全部为0 |
| PERF-01 | P0/P1/P2/P3/P4独立campaign | 原始样本、置信区间、明确comparison_kind |

quality 容差必须按模型/精度预注册；确定性相同 kernel 优先逐位一致，非确定性路径同时报告 max abs/relative L2/成品指标。
不能统一用一个宽松 SSIM 阈值掩盖 latent/帧序/audio 错误；LoRA/量化变化不属于 slot 框架优化。

## 8. 真正低内存与 swap 对比操作边界

先做无压力 baseline 和 model-only 仿真；再在专用低内存机器执行 L3。大内存机器上的软件 budget 只限制候选，不能等同物理低内存。
全系统 pressure 不是普通 benchmark 默认行为，需操作者明确启动；本轮文档任务不启动它。

pressure campaign 预注册：maximum allocated bytes、duration、free disk floor、最低 system available、thermal/pressure stop、cleanup deadline。
stop 触发即停止新请求并回收辅助压力资源；不关闭 swap、不运行无界 allocator、不清未知用户缓存。
辅助压力程序/计数器开销与 memory.scope 必须记录，两侧一致；候选额外节省的 RAM 可以被 OS 用作 file cache，这是实际效果，不应强行抹平。

分开观察：physical read/write、file-backed fault、anonymous/compressed memory、swapin/swapout、process footprint、managed peak。
系统 swap counter 不能自动归因给 TurboCider；bounded run 检测活动仍失败，若证据表明外部污染则 campaign 标 INCONCLUSIVE，不能把失败改 PASS。

baseline timeout/OOM：报告 completion rate/timeout threshold，而不是把超时阈值当精确 baseline wall。
两侧都完成且 quality合格的匹配数据可算条件 speedup，但必须同时公布失败率，避免幸存者偏差。

## 9. 仿真校准与工具验收

- simulator 复用 plan/resource constraints，而非复制 runtime 错误；保留独立 reference recurrence 验证小例子。
- 用一组真实 traces 校准 read/decode/compute/fence/竞争，用未参与拟合的 shape/layout traces 验证预测。
- 报告 wall误差、exposed wait误差、候选排序命中、误差区间；没有独立验证时仅 sensitivity，不给生产 preset。
- K3 可能降低 tail，也可能因共享带宽/file-cache缩小而变慢；仿真必须允许这种结果。
- `sweep` 默认 plan/simulate；只有显式 GPU experiment flag 和资源上限才执行有限候选集。
- 自动推荐不改已有请求，不能凭模拟结果写 production registry。

## 10. Evidence bundle 与独立 verifier

除 07 文件外，加入 `campaign-policy.json`、`raw-samples.jsonl`、`quality.json`、`faults.json`、`build-identity.json`。
每个样本有 run/pair/block ID、mode、cache condition、status、wall、完整布局 identity、peak/bytes、排除原因。
policy 在运行前保存 hash；原始失败/超时样本不删除。verifier 不接受只有 aggregate summary 的 release bundle。

verifier 检查：manifest hash→schema→identity对齐→event/lifetime→quality→memory→performance→eligibility建议。
建议不等于自动签发：生产 registry 修改仍需 review，避免 benchmark 自签资格。
示例 [performance-policy.json](examples/performance-policy.json) 只演示冻结阈值，不是可直接运行的 campaign，也未授权压力测试。

## 11. 一次验收的执行顺序

1. F0/F1 host tests通过，生成 plan-only报告并确认 exact tuple。
2. F2 fake/race/fault通过；GPU设备不可用则明确停在这一层。
3. LTX或H3最小真实运行，先quality/completion，再normal-target。
4. 独立audit检查默认零新增行为和new steady-state约束。
5. P0/P1正式配对性能，任何回归先分析wait/read/commit/CPU时间，不立即增加slot掩盖问题。
6. F5闭包后进行P2、完整L2；再经授权做L3/P3。
7. 独立verify，记录PASS/FAIL/SKIP/INCONCLUSIVE；只发布证据覆盖的tuple。
8. 发布后仍保留旧路径；回滚撤新record，失败preset明确报错不自动改布局。

## 12. 规格与实际证据的边界

本文是验收规格；后续已经有compiler/executor/C bridge基础和部分测试，但真实model adapter与性能campaign尚未完成，准确范围见 [13](13-implementation-progress.md)。
本次布局优先深化新增的施工与工具方案见 [14](14-layout-first-integration.md)、[15](15-adapter-implementation-plan.md)、[16](16-performance-toolchain-plan.md)，不是新性能成绩。
后续每个PR应把测试ID映射到真实测试文件/命令及artifact；在映射之前不能声称该gate已自动化。
框架级整合的当前施工矩阵与新增测试映射见 [18](18-code-change-matrix.md) 第9–10节；本文件仍是性能阈值与统计规则的唯一规范。
