# TurboCider 通用 Streaming 框架

修订日期：2026-09-17。状态：设计规格及实施中，尚未发布。最新代码/测试证据见 [13 实施进度](13-implementation-progress.md)。

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

架构阅读：01 → 02 → 03 → 04/05 → **17**。实现阅读：09 → 10 → 06/11 → **18** → 12；实验工具原则见 07。查事实和历史先读 08。

布局优先整合评审先读 **14 → 15 → 16**：先定边界和持久生命周期，再按真实模型切入，最后用可复现证据决定发布。
本轮布局优先评审建议先读 **19 → 20**；总体合同查17，逐文件查18，正式性能签核查12。
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

## 当前代码检查点（2026-09-17）

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
[13 第13.14–13.15节](13-implementation-progress.md)，协议见 [03 第11节](03-runtime-protocol.md)，执行细节见
[10 第13–14节](10-executor-implementation.md)。

Public preset 控制面基础已提交到`63b73d9`：schema-v2 selector、request/profile 合并、plan-only 报告、空 production
catalog、host resolver、metadata-only options C ABI、Swift v2/options 类型，以及 active selector 在普通/candidate
generate/prepare 中的早期 fail-closed gate。当前工作树又加入 typed canonical encoder、完整 record identity、
resolver/authority/resolved request 和默认拒绝的 session public hooks；resolver host test 与 native-only build 已通过。
这些 R1/R2 增量尚未形成阶段提交，exact engine C ABI、public generate、Swift/App 事务、四模型 override、校准工具和 reviewed records
仍未完成。下一步按[28](28-public-runtime-code-design.md)和[30](30-public-streaming-detailed-integration.md)完成 exact runtime，
再按[29](29-public-implementation-and-acceptance-plan.md)与[31](31-public-streaming-config-calibration-runbook.md)建设工具、
模型证据和 reviewed records；production catalog 仍为空，当前不可 public 执行。
