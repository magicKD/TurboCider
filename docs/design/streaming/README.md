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

架构阅读：01 → 02 → 03 → 04/05 → **17**。实现阅读：09 → 10 → 06/11 → **18** → 12；实验工具原则见 07。查事实和历史先读 08。

布局优先整合评审先读 **14 → 15 → 16**：先定边界和持久生命周期，再按真实模型切入，最后用可复现证据决定发布。
本轮布局优先评审建议先读 **19 → 20**；总体合同查17，逐文件查18，正式性能签核查12。
13是当前实施事实来源；08第1–8节是历史设计快照，不能当作当前尚无代码的结论。

不必顺序阅读全部文档：架构评审看19，实施负责人看20第3–7节，测试负责人看20第8–10节和12。
参数冲突以02为准，预算以05为准，compiler/executor以09/10为准，性能阈值以12为准；
19/20是实施展开，不新增 retention 值、配置别名、F/L/P 编号或另一套调度器。

## 最小落地路线与性能原则

先完成配置/纯 compiler（plan-only），再完成 fake slot/fence，随后接一条 LTX exact layout，验证后接 H3 双槽；
预算 guard 在完整资源闭包后独立放行。Flux/Z-Image 从 component staged 开始，不一次替换所有模型内循环。

“性能不能差”拆成三件事：默认路径零新增热路径工作、同布局框架开销通过非劣验收、低内存策略收益单独实测。
12 给出默认 median 2%/P95 5% 的非劣发布阻断线和置信区间规则；这是验收目标，不是本轮已取得的数据。
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
9. H3/LTX 先做原生 Metal adapter；Flux/Z-Image 先 component-staged，再独立验证 block/group streaming。
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

## 当前代码检查点（2026-09-17）

统一 executor 已支持 ordered multi-class barrier，以及单 pool K=2/G=1 的显式 cross-pass
`carry_first_group`（C ABI v3）。H3 已完成 metadata → layout → v3 plan → fake executor 证据，但尚未接真实
Metal block adapter；LTX exact 仍是内部 candidate，production registry 为空。最新实现事实见
[13 第13.9节](13-implementation-progress.md)，协议见 [03 第11节](03-runtime-protocol.md)，执行细节见
[10 第13节](10-executor-implementation.md)。
