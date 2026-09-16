# 06 · 框架级整合与逐文件改造

[目录](README.md) · [架构](01-framework.md) · [验收](07-tooling-and-validation.md)

本文是迁移目标图；其中config/compiler/executor/C bridge已有部分基础实现，准确范围见 [13](13-implementation-progress.md)。
下方“新增”表示相对legacy的新增模块，不代表全都尚不存在；模型adapter/registry等未落地项仍是建议。
本文维护逐文件变更总图；每批任务/停止条件见 [11](11-work-packages.md)，接口算法见 [09](09-layout-compiler-spec.md)/[10](10-executor-implementation.md)，最终 gate 见 [12](12-acceptance-playbook.md)。

## 1. 目录方案（拟议）

```text
native/core/
  streaming_contracts.hpp             新增：typed user intent / presence flags
  stream_slot_c.h                     新增：C POD adapter bridge
native/runtime/streaming/
  descriptor.hpp/.cpp                 metadata descriptions / validators
  layout.hpp/.cpp                     manual layout compiler / canonical digest
  context.hpp/.cpp                    request owner / stage lifecycle
  c_bridge.cpp                       已有：版本化C adapter调用同一C++ executor
  slot_pool.hpp/.cpp                  backing/content state + generations
  io_executor.hpp/.cpp                bounded workers + read/ready mailbox
  use_fence.hpp/.cpp                  last reader fence，区别于 backing release
  registry.hpp/.cpp                   layout execution eligibility
native/platform/apple/
  streaming_config.mm                新增：JSON parse/profile bridge
native/models/ltx_runtime/
  ltx_streaming_adapter.c             新增：LTX layout/fill/kernel bridge
native/models/h3_runtime/
  h3_streaming_adapter.c              新增：H3 dual-slot bridge
```

既有 memory_accounting/manifest/plan/execution/schedule/scheduler/watchdog 不整体搬家、不重命名公共符号。
streaming 通过窄接口复用它们；避免一次 PR 同时做目录移动、语义重构和模型优化。
当前`context.*`仅有StageExecutor；完整request orchestration仍待实现，不能由上述目标目录推断已具备整请求调度。
逐helper施工和persistent bridge接入细节见 [15](15-adapter-implementation-plan.md)。

## 2. 配置到请求

| 文件 | 修改建议 | 测试要求 |
|---|---|---|
| `native/core/contracts.hpp` | Request 增加 typed StreamingConfig；跟踪旧 residency/budget 显式性 | 缺省与显式 false/0 不混淆 |
| `native/core/memory_contracts.hpp` | 保留当前 Y/X/S 字段；显式 max_refill_slots 作为兼容 cap | 不修改旧默认请求语义 |
| `native/platform/apple/request.mm` | 新增 execution.streaming parsing，当前未知字段拒绝改为严格新 schema | duplicate/unknown/type/overflow/conflict |
| `native/platform/apple/profile.mm` | 分离 v1/v2 reader，v2 model.streaming + request overlay | v1 golden fixtures 不变 |
| `native/platform/apple/results.mm` | 输出 requested/resolved/actual 布局、来源、资格、资源/指标 | 默认结果旧字段不改义，新字段只按需追加 |

不要继续使用字符串 `residency` 推导所有 streaming 细节；也不在 `make_plan()` 内改写用户 manual K/P。
对外 C API 如仍传 JSON，可不增加单个参数函数；结构体 ABI 扩展必须 versioned，不能静默改变未带 size 的二进制布局。

## 3. Plan 与 session

`native/runtime/session.hpp`：

- `ExecutionPlan` 增 optional streaming intent/preview，exact resolved plan 由 session preflight 后附加；
- ModelSession 新增显式 `describe_streaming()`、`create_streaming_adapter()`，默认返回 unsupported；
- 不让默认空 virtual implementation 假装支持真实 drain；capability 必须显式声明。

`native/runtime/plan.cpp`：

- 先根据字段 presence 选择 legacy/new-framework；legacy 完整保留；
- 新 manual 做 syntax/route validation，跳过旧 `resolve_memory_constrained_candidate()` 的 budget→K/P 重写；
- model validator 分清新 streaming candidate 与旧 residency，不直接绕过现有 dtype/LoRA/shape 校验；
- heuristic memory estimate 仍只能展示，不具执行授权。

`native/api/c_api.mm`：

```text
if no explicit framework config:
    existing prepare/generate paths (including legacy constrained)
else:
    exact metadata describe + compile layout
    layout preflight eligibility (no unload yet)
    if bounded:
        memory certification preflight
        clean boundary + baseline
        authorize same layout against budget (no parameter rewrite)
    create streaming context, bind adapter, execute
    join + drain + destroy + terminal report
```

无 layout record 时拒绝应发生在 session unload、cache clear 和 checkpoint materialization 之前。
`generate()` 和 `prepare()` 的事件序列分别建 plan，不能让 load-only prepare 假装走了完整 denoise schedule。

## 4. 与现有 memory runtime 的融合

| 现有模块 | 复用什么 | 避免什么 |
|---|---|---|
| `memory_accounting.*` | site/instance/storage、reservation、pending-release | 为 slot 再造独立“无限预算账本”并宣称全局覆盖 |
| `memory_manifest.*` | required sites、provenance、checkpoint identity | adapter 自签 certified record |
| `memory_plan.*` | epoch live set/alias peak | compiler 把 content Vacant 当 backing free |
| `memory_execution.*` | bounded admission、strict cursor、failure disposition、terminal | layout-only 构造假无限 budget context |
| `memory_scheduler.*` | backing release completion/mailbox | 用同一 token 表示 reuse fence 和 storage release |
| `memory_schedule.*` | codec/sequence/flags validation | 模型与框架双重 emit；用 v1 key 硬编码无法表达的新事件 |
| `memory_watchdog.*` | owner checkpoint observer | 默认路径启动线程/读系统 counters |

新 `StreamingExecutionContext` 是执行对象，MemoryExecutionContext 是可选 guard。先通过组合接入，
后续可抽出纯 ScheduleCursor，但须保持现有 poison/reentrancy/terminal 测试，不一次重写所有 runtime。

## 5. 模型代码改动

LTX：新增 exact-layout create 路径，复用 checkpoint reader/`run_block()`；保留旧 `run_streamed_block_stack()` 作为默认实现与对照。
把 first allocation 从 loader 移到 owner；按 scope 暴露内部 kernel helper，不将整个巨型 C 文件重新组织为前置条件。

H3：新增 adapter 包住现有双槽，替换的只是新分支的 slot selection/fence ownership。默认 xor 双槽先不改；
单槽作为后续独立候选，不把新 API 暴露给未迁移的所有 load overload。

Flux/Z-Image：先加 metadata/component lifecycle，保留默认 compiled graph/Weights 容器；block-level reader 另 PR。
不能把加 `mx::eval()`/`clear_cache()` 当通用解法，也不在高内存默认路径引入每层 barriers。

## 6. Service 集成

`services/turbociderd/service.mm` 增加 normalized streaming identity 到新框架的 worker/session cache key，
保持默认 key 不变；用户手动 P/K 不同不共享未验证 pool。单 GPU job 排队/取消保持原调度。
failed/quarantined worker 不能被复用；config reload 只影响下一请求，不能修改正在运行的 plan。
服务提交请求不被视作批准全系统压力测试或修改 swap 设置。

## 7. 分 PR 实施顺序

| PR | 范围 | 完成门 |
|---|---|---|
| F0 | 新 config/presence/结果 schema，全部 plan-only | v1/default golden 不变；所有冲突有测试 |
| F1 | descriptor + manual compiler + fake layout registry | roundtrip/digest 稳定，actual 不可静默改 K/P |
| F2 | slot backing/content machine + bounded I/O + use fence | synthetic property/fault tests；TSan 生命周期测试 |
| F3 | LTX adapter G=1、P≥1、K=1/2/3 支持子集 | fake exact sequence + 对齐旧布局的真实 Metal parity/cleanup |
| F4 | H3 现有双槽 adapter/coarse semantic events | 双槽质量和 use fence；单槽仍 unsupported |
| F5 | BudgetGuard 组合、site closure、capability identity | same-layout admission、peak verifier、zero unknown required |
| F6 | LTX/H3 real-device campaigns/default ABBA | layout_validated 签核；bounded 必须再过 L3 |
| F7 | H3 serial、G>1、更多 reader等优化候选 | 每次单独 layout revision + 证据；不顺手改默认 |
| F8 | Flux/Z-Image component，随后 group streaming | 框架引用/envelope/lifetime 闭环，未通过就 plan-only |
| F9 | 推荐工具/preset、可选 retained-root 扩展 | 工具不自授权；retained 独立 accounting/验收 |

F2 依赖 F1；F3/F4 都依赖 F2；F5 在对应模型 adapters 基础上逐个完成；F6 的 bounded 部分依赖 F5。
不把原稿里的旧 PR 编号重新解释成这些编号；本表是新框架的施工队列。

## 8. 性能隔离与发布/回滚

- 入口 dispatch 一次；默认内循环继续原样，不走 nullable callback 的每层判断。
- 框架热路径固定容量容器，error message 在失败路径格式化；耗时 trace 在诊断或 guard 需要时开启。
- 不在用户首个 generate 时自动 benchmark/tune/扫描整机器 profile；推荐在离线或显式命令执行。
- 内存充足默认机器继续 legacy resident；高内存模式不是自动试验新框架的理由。
- 合入代码不等于启用 release record；布局与内存资格分别签核。
- 回滚只撤对应新 registry entry 或禁用显式 framework route；既有默认模型路径不被修改。
- 不支持的已保存 preset 返回明确错误，不回退未知 resident，避免突然 OOM/swap。

## 9. 实现验收与文档维护

每个 PR 更新 [当前状态](08-status-and-history.md) 的真实完成项、命令和 evidence，不能只更新“代码已接线”。
架构决策改 01、schema 改 02、协议改 03、模型差异改 04；不要重新在归档尾部累积数百节。
新工具、新配置、新 registry 的实现未落地前，文档中的命令均标拟议，保持 reviewer 能区分现状与目标。
