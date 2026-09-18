# TurboCider 通用 Streaming 框架

修订日期：2026-09-18。状态：设计规格及实施中，尚未发布。最新代码/测试证据见 [13 实施进度](13-implementation-progress.md)。

提交前审阅、当前发布阻断项和本次重跑范围见 [21 审阅与交接](21-review-and-handoff.md)。

## 一句话决策

将 block/slot streaming 做成独立执行框架：**用户指定布局，adapter 描述模型，框架编译和执行；内存上限只作为可选校验与守卫。**

可以不从 Y 倒推 slot，但不能由 slot 数单独推断完整进程的内存上限。三槽 weights 很小的模型仍可能被 activation 或 VAE 撑满。

## 新的职责划分

```text
用户手动布局 / 显式选用的 preset / 可选离线调优器
                         |
模型 ResourceDescriptor -> LayoutCompiler -> ResolvedStreamingPlan
                                                 |
                            可选 MemoryGuard ----+---- execution eligibility
                                                 |
                             StageExecutor / SlotPool / bounded I/O
                                                 |
                                  模型 kernel + backend fence
```

关闭新 streaming 且关闭 memory guard：继续调用原有执行路径，不创建上述对象。用户原有 `residency=streamed` 也属于原路径，不能因新框架默认关闭而失效。

## 文档地图

| 文档 | 回答的问题 |
|---|---|
| [01 Framework](01-framework.md) | 框架统一什么，不统一什么；manual/auto/guard 的边界 |
| [02 Configuration](02-configuration.md) | slot/group/prefix 的精确定义，JSON 示例和冲突处理 |
| [03 Runtime](03-runtime-protocol.md) | backing 与 content 生命周期、异步协议和错误恢复 |
| [04 Adapters](04-model-adapters.md) | 各模型需要提供什么、当前缺口是什么 |
| [05 Memory](05-memory-contract.md) | 给定布局的内存计算、预算 guard 和认证 |
| [06 Migration](06-migration.md) | 具体修改哪些文件、如何分 PR、怎样不影响默认路径 |
| [07 Tooling / Validation](07-tooling-and-validation.md) | inspect/plan/simulate/campaign、swap 比较和验收 |
| [08 Status / History](08-status-and-history.md) | 当前真实状态、历史原稿与旧章节映射 |
| [09 Compiler Spec](09-layout-compiler-spec.md) | 数据类型、分组/容量/身份算法、异构布局与 golden fixture |
| [10 Executor Implementation](10-executor-implementation.md) | owner pump、线程/队列、真实 fence、进展条件与热路径预算 |
| [11 Work Packages](11-work-packages.md) | F0–F9 每批交付物、代码切入点、依赖与停止条件 |
| [12 Acceptance Playbook](12-acceptance-playbook.md) | 测试 ID、性能统计、真实机器操作与发布门槛 |
| [13 Implementation Progress](13-implementation-progress.md) | 当前已实现基础、真实测试证据、剩余模型/性能验收 |
| [14 Layout-first Integration](14-layout-first-integration.md) | 端到端职责、四张计划表、持久池生命周期、配置档位和当前缺口 |
| [15 Adapter Implementation](15-adapter-implementation-plan.md) | LTX/H3/Flux/Z-Image逐helper施工单、C bridge接线、专项验收 |
| [16 Performance Toolchain](16-performance-toolchain-plan.md) | 仿真/benchmark/verifier实施、性能诊断、当前测试与本机验收路线 |
| [17 Layout-First Framework](17-layout-first-framework.md) | 框架级控制面、数据面、状态机、multi-slot 与模型统一接入合同 |
| [18 Code Change Matrix](18-code-change-matrix.md) | 按目录/文件拆分的实施顺序、接口草案、回滚点与验收映射 |
| [19 Layout-first Architecture](19-layout-first-architecture.md) | 对象/依赖、确定性分组与容量算法、owner pump、双槽时序和峰值算例 |
| [20 Implementation and Acceptance](20-layout-first-implementation-and-acceptance.md) | 当前 v2 接缝、下一批 PR、失败/所有权测试、模型接入与 P0–P4 交付单 |
| [21 Review and Handoff](21-review-and-handoff.md) | 提交范围、审阅发现、当前验证与按优先级排列的未完成项 |
| [22 Current Framework Guide](22-current-framework-guide.md) | 当前 slot/streaming 框架的完整使用、配置、调度、模型接入、性能与限制说明 |
| [23 Public Memory-Tier Presets](23-public-memory-tier-presets.md) | App 高级开关、8/10/12/16/20 GiB 目标、只读 catalog、public 准入与配置/代码改造（待实施） |
| [24 Memory-Tier Exploration](24-memory-tier-exploration-and-acceptance.md) | 四模型候选布局、整请求内存测量、离线探索/独立确认、分批实施与验收（待实施） |
| [25 Public Preset Implementation Spec](25-public-preset-implementation-spec.md) | 将 selector、catalog、authority、snapshot、C ABI、Swift、resolver、pager 和四模型 adapter 接到当前代码的实施规格 |
| [26 Public Preset Acceptance and Release](26-public-preset-acceptance-and-release.md) | memory calibration 工具链、swap 对照、PUB/CAL/model 测试矩阵、发布/撤回/回滚验收 |
| [27 Public Streaming Delivery Blueprint](27-public-streaming-delivery-blueprint.md) | 基于已提交控制面的端到端施工顺序、所有权、App/engine/四模型接线、GPU/ANE边界、PR拆分与验收追踪矩阵 |
| [28 Public Runtime Code Design](28-public-runtime-code-design.md) | resolver/probe/snapshot/authority、engine/C ABI、Swift/App事务、LTX worker与四模型逐文件代码设计 |
| [29 Public Implementation and Acceptance](29-public-implementation-and-acceptance-plan.md) | 可回滚PR计划、工具链、候选矩阵、内存/swap实验、性能门、evidence/review和发布回滚 |
| [30 Public Detailed Integration](30-public-streaming-detailed-integration.md) | 基于当前代码的精确调用链、锁/所有权、adapter模板、multi-slot时序、逐PR代码任务和自动化验收矩阵 |
| [31 Public Config and Calibration Runbook](31-public-streaming-config-calibration-runbook.md) | selector/device/workload配置、四模型候选矩阵、完整内存校准、swap对照、命令模板与签字验收单 |
| [32 Public Streaming Completion Spec](32-public-streaming-completion-spec.md) | 以当前 exact C ABI 工作树为基线的收口规格：共享校验、错误优先级、source revalidation、RunResult、App事务、四模型接入、swap实验与 release checklist |
| [33 Public Runtime/App Engineering](33-public-runtime-app-engineering-spec.md) | 将共享 validator、exact resolve、source lease、actual-plan、C ABI、Swift、App/JobStore、LTX worker 和默认性能保护拆成可直接编码的工程合同 |
| [34 Model Tier Calibration and Release](34-model-tier-calibration-and-release-spec.md) | 四模型候选族、8/10/12/16/20 GiB 完整请求校准、resident/streaming/swap 对照、evidence、catalog review、发布与撤回 |
| [35 Public Streaming Implementation Blueprint v2](35-public-streaming-implementation-blueprint-v2.md) | 将当前代码接缝、coordinator、test-only catalog、source lease、actual receipt、四模型 adapter、App 事务、工具链、故障注入、验收矩阵和提交边界串成可直接施工的蓝图 |
| [36 Public Adapter Code Implementation](36-public-adapter-code-implementation-spec.md) | 基于当前代码的 coordinator 收口、value probe/snapshot、fd source lease、receipt v2、四模型逐文件 public adapter、LTX worker 协议和代码级测试 |
| [37 Calibration, Performance and Acceptance](37-public-streaming-calibration-performance-acceptance.md) | 五档候选探索、完整 process-tree 采样、resident/streaming/bounded/swap 四路对照、P0–P4、App/JobStore、evidence/catalog/release 验收 |
| [38 Framework Code Contracts](38-framework-code-contracts-and-implementation-workbench.md) | 将 source lease、probe/snapshot、receipt、owner pump、multi-pool、C ABI、JobStore 和四模型逐文件改造落到代码合同与 PR 停止条件 |
| [39 Validation / Benchmark Workbook](39-validation-benchmark-and-release-workbook.md) | 测试分层、故障注入、真实校准、四臂性能/压力对照、统计门、evidence bundle 和 release 签字工作簿 |
| [40 Public Productization and App Contract](40-public-productization-and-app-contract.md) | 用户只选内存档位、options/resolve、App/JobStore 事务、public/private/legacy 边界与默认路径性能保护 |
| [41 Scheduler / Multi-slot / Multi-pool](41-scheduler-multi-slot-and-multi-pool-implementation.md) | owner pump、slot 状态机、K/D/Q overlap、carry、serial/retain-all pool、内存模型与取消/quarantine |
| [42 Model Adapter Playbooks](42-model-adapter-playbooks.md) | Z-Image、Flux 9B、H3 Turbo、LTX 的 source closure、候选族、代码落点、错误矩阵与统一 adapter 模板 |
| [43 Toolchain / Simulation / Release Gates](43-toolchain-simulation-and-release-gates.md) | inspect/compile/simulate/campaign/verify/builder 工具链、process-tree 采样、swap 四臂、P0–P4 门禁与撤回 |
| [44 Next Implementation / Integration](44-next-implementation-code-and-integration-spec.md) | 记录 C2 receipt v2 已实现基线，并规定 public generate、四模型 hooks、App/JobStore、profile 与分阶段回滚的逐文件接线 |
| [45 Acceptance Traceability / Evidence](45-acceptance-traceability-and-evidence-spec.md) | 将 host/synthetic/real-model、source/receipt、process-tree、swap 四臂、P0–P4、evidence/catalog/revoke 映射为可执行测试 ID 和签核项 |
| [46 Public Streaming Implementation Handbook](46-public-streaming-implementation-handbook.md) | 基于 C2 已完成基线，收敛用户五档、控制面/数据面/证据面、调度算法、四模型逐文件改造、工具链、App、性能与发布验收的实施总手册 |
| [47 Public Streaming Code / Acceptance Detail](47-public-streaming-code-implementation-and-acceptance-detail.md) | 46 的代码级附录：request-scoped所有权、common runtime接口、multi-slot状态机、四模型逐文件施工、五档搜索、App事务、工具链、swap实验和分层验收 |
| [48 Public Streaming Engineering Addendum](48-public-streaming-engineering-addendum.md) | 以当前代码为基准的工程实施附录：selector→catalog→exact layout 求解、StageExecutor 状态机、source closure、四模型代码接线、App/JobStore、工具链、四臂 swap 对照和逐层验收 |
| [49 Public Streaming Code Contracts](49-public-streaming-code-contracts-and-execution-blueprint.md) | 将 48 下钻为可编码合同：依赖方向、接口职责、线程/事件/错误协议、Flux lease lineage、逐模型文件施工、ledger、测试 ID 与 PR 停止条件 |
| [50 Calibration / Release Evidence](50-public-streaming-calibration-and-release-evidence.md) | 五档候选生成、四模型搜索策略、process-tree/swap 四臂、P0–P3 统计、evidence bundle、独立 verifier、catalog builder/revoke 和发布签字手册 |
| [51 Remaining Runtime Closure](51-public-streaming-remaining-runtime-closure.md) | 当前剩余 runtime 闭环：request-scoped context、multi-stage/boundary receipt、H3 C receipt/public hooks、LTX worker-local authority、错误/quarantine 与验收 |
| [52 App / Config / Model Tiers](52-public-streaming-app-config-and-model-tier-spec.md) | App 五档与迁移、物理内存推荐、model×target record、内存 ledger/sampler、工具链、四臂实验、JobStore 和发布验收 |
| [53 Final Implementation Closure](53-public-streaming-final-implementation-closure.md) | 将当前代码接缝收敛为可直接施工的文件级方案：R3 receipt/context、scheduler、四模型 adapter、App/JobStore、catalog、工具链、提交边界和 Definition of Done |
| [54 Deep Implementation Spec](54-public-streaming-deep-implementation-spec.md) | 代码级补充：对象所有权、精确调用顺序、compiler/ledger、owner pump、H3 C receipt、LTX worker 协议、工具链、四臂实验、PR 停止条件和 release checklist |

架构阅读：01 → 02 → 03 → 04/05 → **17**。实现阅读：09 → 10 → 06/11 → **18** → 12；实验工具原则见 07。查事实和历史先读 08。

布局优先整合评审先读 **14 → 15 → 16**：先定边界和持久生命周期，再按真实模型切入，最后用可复现证据决定发布。
本轮布局优先评审建议先读 **19 → 20**；总体合同查17，逐文件查18，正式性能签核查12。准备按当前工作树直接施工时，继续读 **35**。
13是当前实施事实来源；08第1–8节是历史设计快照，不能当作当前尚无代码的结论。

不必顺序阅读全部文档：架构评审看19，实施负责人看20第3–7节，测试负责人看20第8–10节和12。
希望从使用方式一路理解到当前四模型表现时，直接阅读22；它是面向使用者和实现者的当前总览，不替代各主题规范。
希望在 App 中仅选择内存目标、不暴露槽位参数时，阅读23 → 24。这是新的产品化设计，尚未运行档位探索或放开 public gate；
目标档位不是物理显存容量或 hard cap，未测数据不得填成已支持。
希望直接实施代码接线时，阅读25；希望安排实验、验收和发布时，阅读26。25/26中的新增文件/API/测试编号均为拟议项，
没有实现与 evidence 之前不能称为已支持或已公开。
需要从当前已提交控制面基础继续施工时，优先阅读27：它记录 selector/catalog/options/Swift 已有部分、
unresolved selector 的立即安全闸门、query→resolve→generate 的所有权链、App任务迁移以及逐PR完成门。
控制面基础现已提交到`63b73d9`。准备直接编码 resolver/authority/App transaction 时阅读28；安排工具开发、
四模型档位实验、swap对照和逐record发布时阅读29；需要按当前文件和函数逐项施工、检查锁/生命周期、编写 test ID 时阅读30。
当前工作树已经进一步接入 exact engine resolve/public generate authority 和 Swift resolve；继续收口实现时应先读32，
其中冻结了空 catalog 前的共享请求校验、错误优先级、source lease revalidation、public result、App事务和模型接入顺序。
准备直接拆解 runtime/App 工单时继续读33；准备执行四模型候选探索、完整请求内存校准、swap对照和 record 发布时读34。
33/34把既有结论变成任务和验收合同，不改变 production catalog 为空、public streaming 当前不可执行的事实。
35进一步把 32–34 的合同落到当前文件、类型、状态机、测试 ID 和 R0–R8 提交边界；它仍是实施蓝图，不代表任何 public record 已发布。
从当前 `33bd9ea` C2 基线继续直接实现时，优先阅读 46 → 47 → 42 → 44 → 45：46 给出收敛后的实施总手册，47 冻结 request-scoped 生命周期、代码接口、模型施工和逐层验收，42/44 给出四模型 adapter 和逐文件接线，45 提供真实档位校准、swap 对照、App 事务、证据和 release gate；需要追溯 common runtime 设计时再查 36/38/41。
如果从产品/App 视角评审，先读 40；如果从工具和发布视角评审，先读 43。40–43 均是设计与实施合同，不表示 production catalog 已非空或任何 target 已 public。
当前 C2 Actual Receipt v2 已在 `33bd9ea` 提交并通过 host/sanitizer/build 验收，详见 13.25。继续编码时先读
**46 第6–8节与第14–17节**，再按 **44 第5节（四模型）→ 第6节（App）** 施工；实现后按 **45** 的测试 ID、
采样字段、P0–P4 门槛和 evidence bundle 逐项验收。44–46 仍是规格，不改变 production catalog 为空的事实。

如果需要把当前设计直接拆成代码工单和评审 checklist，继续阅读 **48**：它冻结了用户只选目标档位、内部
record 确定性选择、`P/G/K/D/Q` 候选约束、request-scoped 生命周期、StageExecutor owner pump、四模型 source
closure、App/JobStore 事务、inspect/compile/simulate/campaign/verifier/builder 工具链以及停止条件。48 是
实施合同，不代表 Flux/H3/LTX public 已完成，也不代表任何 production record 已发布。
需要直接修改 common runtime、Flux lease loader、H3/LTX adapter 或组织 PR review 时，阅读 **49**；需要执行
8/10/12/16/20 GiB 实机探索、process-tree 采样、swap 四臂、证据签核和 catalog 发布时，阅读 **50**。
49/50 是 48 的分册，不另起一套 executor、内存定义或性能门槛；冲突时仍以 02/05/09/10/12 的主题规范为准。
在 Flux public lease adapter `9a351c9` 之后继续收口代码时，优先阅读 **51 → 52**：51 给出 H3 完整 C receipt、
LTX multi-stage/boundary、worker-local authority 和 request cleanup 的具体接口与测试；52 给出 release App 的 Off/五档状态、
旧草稿迁移、模型 card、process-tree sampler、四臂实验和 catalog 发布合同。51/52 仍是设计规格，不表示 production
catalog 已非空，也不替代 50 的真实 evidence 要求。

如果需要按当前工作树直接排工单和做代码评审，再阅读 **53**：它把 51/52 的合同串成一个 request
状态机，标出 `DONE/WIP/NEXT/BLOCKED/NOT_PUBLIC`，给出 `results.mm`、H3、LTX、Z-Image、Flux、
Swift/JobStore、catalog builder 和 R3–R6 的逐文件施工顺序。53 仍是实施稿，不改变 production
catalog 为空的事实。
如果需要进一步下钻到“这一行代码由谁拥有、哪个线程可以修改、怎样生成真实 receipt、怎样把 H3/LTX
接到 common runtime、怎样执行四臂实验和逐 PR 停止”，阅读 **54**。54 是 53 的代码级附录，不新增
第二套 scheduler、预算定义或性能门槛；其中所有未标记 `DONE` 的接口和 record 都仍是待实施合同。
28/30不新增第二套executor，29/30不重新定义文档12的P0–P4阈值。
参数冲突以02为准，预算以05为准，compiler/executor以09/10为准，性能阈值以12为准；
19/20是实施展开，不新增 retention 值、配置别名、F/L/P 编号或另一套调度器。

## 最小落地路线与性能原则

先完成配置/纯 compiler（plan-only），再完成 fake slot/fence，随后按 LTX、Z-Image、H3、Flux 推进真实 adapter；
预算 guard 在完整资源闭包后独立放行。LTX 与 Z-Image 已各关闭一个冻结初始 tuple 的严格 P1；H3 只覆盖
MiniMax H3 Turbo，K2/G1 冻结 tuple 的工程验收已经完成。H3 支持范围到此为止，不计划接普通 H3、其他
checkpoint或量化变体。严格 bootstrap 统计仍为 `INCONCLUSIVE`，作为 I/O 方差
诊断保留但不再阻断本轮收口，也不等同于 production P1 PASS。Flux.2 Klein 9B 已完成 private BF16
execution adapter：8个dual block、24个single block、两个retained K2 pool和Q2 refill；默认resident审计为零，
两步输出与resident一致。Flux private同布局direct replay已完成正式10-block/20-pair P1：generic/direct wall
median ratio `0.99971`、wall P95 ratio `1.00325`、denoise median ratio `0.99816`，对应bootstrap上界均低于
2%/5%门槛；40/40请求成功、20/20输出一致，framework overhead已在该冻结tuple上签核。

“性能不能差”拆成三件事：默认路径零新增热路径工作、同布局框架开销通过非劣验收、低内存策略收益单独实测。
12 给出默认 median 2%/P95 5% 的非劣发布阻断线和置信区间规则；LTX P8/G1/K3/D2/Q3 和 Z-Image
P14/G1/K2/D0/Q1 的冻结tiny初始tuple已取得P1证据，其他shape、模型和bounded-memory仍必须独立验收。
改变 P/Q/retention 的加速不能冒充纯 framework 收益，也不能补偿默认回归。

## 本次冻结的决定

1. `slot_count` 是精确值，不是 upper bound；manual 模式拒绝时不擅自减槽、缩小 shape 或改 dtype。
2. `block_group_size` 控制加载/驻留分组，不是跳过 block，不改变模型计算顺序。
3. `resident_prefix_blocks`、`slot_count`、`prefetch_distance`、`io_workers` 是四个独立参数。
4. 新 streaming 的功能安全必须一直开启：slot 不可提前复写、故障必须 drain。它不依赖 memory guard 是否开启。
5. 关闭 budget guard 只表示没有用户指定的全进程上限保证，不表示允许越过 adapter 能力、文件安全或执行资格校验。
6. 新 layout-only release 和 bounded-memory release 需要不同的证据；不能把关闭 guard 当成绕过未完成实现的捷径。
7. 首版单 GPU、单作业、单活动 streaming slot pool，component 间保守串行；不实现任意 DAG 跨 component 预取。已纳入计划与预算的 resident/helper 权重可同时存在，“单池”不代表整次请求只能有这一份权重。
8. 内存充足时默认路径保留全部既有优化；用户若精确选 streamed，不自动改 resident。可显式选 resident 或使用未来 opt-in recommendation。
9. H3/LTX 使用原生 Metal adapter；Z-Image及Flux 9B已分别完成MLX block/group candidate，Flux 4B compiled
   graph保持原默认路径且明确不进入当前资格。
10. 新资格只在真实模型/性能/资源验收通过后加入production；当前基础实现不等于资格放行。

## 文档维护规则

- 各主题文件是唯一规范来源，不再向巨型文档不断追加“最新一节”。跨主题用链接。
- 新结论修改对应章节；当前实际测试记录放 13 或独立带日期的 evidence；08仅保留历史，不混入拟议配置。
- API/config 未实现时明确标记；示例目录不是可直接加载的 production profile。
- `archive/` 只保留历史，不作为当前实现依据。原稿中旧统计、草案、相对链接及格式问题保留；提交前的路径整理见归档 README。
- 实现 PR 同时更新该主题的“已实现/待实现”与测试链接，不用无分母的完成百分比。

## 示例

[三槽手动布局](examples/manual-ltx.json) · [同一布局加预算](examples/bounded-ltx.json) · [显式 resident](examples/resident-ltx.json) · [拟议 profile v2](examples/profile-v2.json)

配置示例现已接入schema2 request的plan-only解析；模型adapter与执行资格尚未完成，generate/prepare仍明确拒绝新manual路线。
示例文件存在不意味着模型已经获得执行授权。
另有 [compiler golden fixture](examples/compiler-golden.json) 与 [性能 gate 示例](examples/performance-policy.json)，不是生成请求或可执行 campaign。
LTX 正式同布局 P1 使用 [可执行 opt-in policy](examples/ltx-p1-same-layout-policy.json)；它需要仓库根目录下的
release dylib、LTX 模型和实体 Metal 权限，不进入默认测试，也不会主动创建内存压力。
Z-Image 对应 policy 为
[z-image-p1-same-layout-policy.json](examples/z-image-p1-same-layout-policy.json)，同样只用于显式私有验收。
H3 Turbo 对应 policy 为
[h3-p1-same-layout-policy.json](examples/h3-p1-same-layout-policy.json)；它只覆盖原始 BF16 P0/G1/K2/D1/Q1
denoiser tuple，不代表其他 H3 变体或 public production 资格。
Flux 9B 的可复现 private candidate 请求见
[flux9-k2-q2-request.json](examples/flux9-k2-q2-request.json)；同布局direct/generic正式P1 policy见
[flux9-p1-same-layout-policy.json](examples/flux9-p1-same-layout-policy.json)。后者会硬校验baseline必须为
`flux_direct_same_layout_v1`、candidate必须为`generic_stage_executor_v1`。

## 当前代码检查点（2026-09-18）

统一 executor 已支持 ordered multi-class barrier、单 pool K=2/G=1 的显式 cross-pass
`carry_first_group`（C ABI v3）、claim后fill overlap和同步reader completion快路径。H3 Turbo K2/G1真实
Metal candidate adapter 已完成20-pair工程验收；wall/denoise median ratio为`1.01066/1.01185`，独立 audit
steady allocation/thread-create为0/0。strict bootstrap verifier仍为INCONCLUSIVE，但按已确认的合理 I/O
波动口径不再要求继续跑盘；public gate未开放，其他 H3 变体也不在计划内。
LTX/Z-Image/Flux 9B exact也仍是内部candidate，production registry为空。
LTX campaign 已改用规范化 actual semantic layout 和显式 per-request engine lifecycle，避免拿 legacy
retained cache 与 request-scoped exact 直接比较。冻结10-block/20-pair initial tuple已通过P1；同时移除
C bridge refill的block-vector复制，并增加setup后稳态allocation/thread audit。Z-Image冻结20-pair P1也已
通过，wall median `0.99986`、denoise median `1.00313`。Flux 9B K2/G1/D1/Q2两步请求相对resident将
MLX peak从`18,303,578,036`降至`11,693,804,356` bytes，PNG byte-exact；真实audit为setup worker/pool
`2/2`、steady allocation/thread-create `0/0`，默认resident五类计数全零。最新实现事实见
[13 第13.14–13.15节](13-implementation-progress.md)，exact public control-plane 工作树接线与本轮 validator/设计验证见 [13 第13.17–13.18节](13-implementation-progress.md)，协议见 [03 第11节](03-runtime-protocol.md)，执行细节见
[10 第13–14节](10-executor-implementation.md)。

Public preset 控制面基础从`63b73d9`继续推进，并已在`46a3e97`收口 exact resolution 基础：schema-v2 selector、
request/profile 合并、空 production catalog、typed canonical encoder、完整 record identity、deterministic resolver、
internal-only authority、immutable resolved request、默认拒绝的 session public hooks、engine model/root/container identity、
`tc_engine_resolve_streaming_json`、active generate 的 resolve→GPU-lock→revalidate→`generate_resolved` 分支、Swift
resolution/error 类型，以及共享 request-only validator 和错误优先级测试。空 production catalog 下，四模型普通/候选
resolve/generate 均在 session/GPU 执行前返回`catalog_has_no_public_records`，active prepare 返回
`streaming_prepare_unsupported`。

actual-plan 汇总 verifier 已在 c6cba54 提交，coordinator/provider 已在 fa1ecd0 提交并完成 host/contract/App 回归。
随后 C0 已完成单次 preflight/catalog snapshot 收口：API 只在 preflight ticket 上调用一次 request-only validator 和一次 catalog snapshot，
make_plan 使用专用的 prevalidated 入口，resolve 校验 request digest，revalidate 重新读取最新 catalog。
随后 C1 fd-backed `SourceLease` 已在 `2878d21` 提交，C2 `ActualExecutionReceipt` v2 已在 `33bd9ea` 提交；common
verifier 现已核对 per-pass/group fill、logical bytes、reader fence、source generation、carry/pool 选择和 canonical digest。
R3 multi-stage common runtime 已在 `aea2cf2` 提交，并由 `30fefe4` 补齐生命周期/失败测试：receipt v3、ordered
boundary、request-scoped `PublicStreamingRunContext`、multi-stage result verifier 和 Objective-C 安全摘要均已接线，
host/contract/audit 与 targeted ASan/UBSan/TSan 通过。Z-Image、Flux 9B 已有 lease-backed public adapter 基础；
H3 Turbo 和 LTX 尚未接入新的 public run context，四模型仍都缺少完整五档 process-tree/release evidence。

下一步按 [53](53-public-streaming-final-implementation-closure.md) 和 [42](42-model-adapter-playbooks.md) 实现
H3 Turbo public C receipt/hooks，再完成 LTX worker-local multi-stage；同时对 Z-Image/Flux 9B 做真实 full-request
复核。随后按 [37](37-public-streaming-calibration-performance-acceptance.md) 建设完整 process-tree calibration、
swap 四路对照、App/JobStore 事务和 reviewed records。production catalog 仍为空，当前仍不可 public 执行。
