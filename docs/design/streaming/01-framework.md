# 01 · 框架决策：布局优先，而非预算反推

[目录](README.md) · [配置](02-configuration.md) · [迁移](06-migration.md)

## 1. 用户的方案可行，但要拆开两个问题

“我要几槽、每槽几块、留多少常驻 block”是执行策略；“进程不能超过 Y 的某个比例”是资源约束。
先指定执行策略再正向计算资源消耗，比在每个模型内部从预算反推一套隐藏策略，更容易解释、复现和调优。

本设计不删除内存估算，而是删除 manual 路径中的参数搜索：

```text
旧的主要入口：预算 -> 模型特定启发式 -> slot/prefix -> 运行
新的主要入口：显式布局 -> metadata layout -> memory requirement -> 验证 -> 运行
可选调优入口：预算/机器事实 -> 搜索布局 -> 推荐配置 -> 用户接受 -> 同一执行入口
```

manual compiler 的输出只能是“原布局的确定性展开”或“拒绝及原因”，不能悄悄改布局。
编译算法/类型见 [09](09-layout-compiler-spec.md)，executor 实施见 [10](10-executor-implementation.md)，
逐批交付和验收执行书见 [11](11-work-packages.md)、[12](12-acceptance-playbook.md)。
跨层整合、持久pass调用、配置档位与当前工程缺口见 [14](14-layout-first-integration.md)；模型施工单和性能工具分别见 [15](15-adapter-implementation-plan.md)、[16](16-performance-toolchain-plan.md)。

## 2. 两个正交维度

| Streaming 配置 | Memory guard | 路径与承诺 |
|---|---|---|
| 无新配置/显式关闭 | 关闭 | legacy；包括现有用户选择的 legacy streamed，不新增框架成本 |
| 显式 manual | 关闭 | layout-only；保证已验证的 slot 生命周期/布局，不承诺 Y 上限或零 swap |
| 显式 manual | 开启 | bounded-manual；同一布局通过全请求 admission，否则拒绝 |
| 无新配置 | 开启 | 迁移期保留现有 memory_constrained 入口；未认证仍 fail-closed，不偷开新框架 |
| 未来显式 recommended | 开启或关闭 | 可选工具先产出具体布局，再进入完全相同的编译与执行路径 |

新 streaming 开关不等于 residency：`enabled=true` 内仍可选 resident stage（slot=0），用于统一管理 component 生命周期。
关闭 streaming 也不等于请求 resident：旧 `residency` 继续按原逻辑解释。

## 3. 六个层级

### 3.1 Policy：只表达意图

`StreamingConfig` 表达手动布局、retention 和阶段配置；`MemoryConstrainedConfig` 保留 Y/X/S。
profile/preset 提供固定值，request 覆盖；不把 “certified”、framework upper 或 checkpoint digest 作为用户可写字段。

### 3.2 Description：模型声明可做什么

`ModelResourceDescriptor` 由 session adapter 根据 checkpoint metadata、shape 和 backend 生成：

- 稳定 stage IDs、依赖和 pass 顺序；
- block 顺序、layout class、tensor ranges/format/alignment；
- 哪些 block/group 可流式加载，哪些小参数必须 resident；
- activation/conditioning/output/lifetime 与 framework envelope；
- 支持的 K/G/P/D/Q 组合、允许的 stage boundary、fence 语义；
- 版本和 checkpoint identity。

这是能力描述，不是 release 授权。descriptor 无法描述的模型保持 unsupported/plan-only。

### 3.3 Compiler：唯一布局权威

`StreamingLayoutCompiler` 展开 groups、slot assignment、pool capacity、pass schedule，生成 immutable `ResolvedStreamingPlan`。
底层 H3/LTX 禁止再调用自己的 budget→prefix 算法重选布局；只核对实际 tensor layout 与计划一致。

纯 `make_plan()` 仍可不持有模型根目录；它只做语法/路由计划。exact resolve 在 session metadata preflight 执行，
不要在通用 planner 中扫描目录或实例化 GPU。

### 3.4 Executor：统一推进生命周期

`StreamingExecutionContext` 管理 owner-thread 顺序、SlotPool、bounded I/O、UseFence mailbox、取消和终止。
每个 component 的数值算法仍由 adapter 执行；框架不接管所有 tensor op，不重写 diffusion sampler。

外层 pipeline 先维持现有 ModelSession 调用顺序，在显式阶段 boundary 调用 executor；稳定后才能扩大统一调度范围。

### 3.5 Resource control：按需附加

两层 observer：

- 必需的 `SlotSafetyTracker`：所有新 streaming 路径都有，检查 ticket、reader fence 和生命周期；
- 可选 `BudgetGuard`：复用 `MemoryExecutionContext`/ledger/manifest/watchdog，验证全请求上限和 required-site closure。

不创建一个假“无限预算的 MemoryExecutionContext”来支持 layout-only；那会混淆授权和指标。新 context 可持有 nullable guard pointer。
固定池 capacity 统计在 layout-only 中仍有，process/swap probe 只在 guard 或显式 diagnostic campaign 中打开。

### 3.6 Evidence：验证之后才授权

执行资格分开标记：`layout_plan_only`、`layout_validated`、`bounded_certified`。这些是拟议字段，不是现有枚举已支持。
layout_validated 至少有真实设备质量/lifetime/cleanup 证据；bounded_certified 还需 whole-request memory closure、低内存和 swap 证据。
现有 production memory registry 保持原限制。新 layout registry 同样先空；离线实验使用测试 build，不能靠请求开关绕过 release gate。

## 4. 框架统一和模型差异

| 统一能力 | 保留在 adapter 的差异 |
|---|---|
| stage/group ID、slot ticket、调度和指标协议 | H3/Flux 的 block 排列、LTX 双 stage 逻辑 |
| backing capacity 和 content generation | BF16/INT8/GGUF layout、scale、dequant scratch |
| I/O admission、bounded queue、read ranges | checkpoint reader、合法格式转换 |
| last-use fence 和 cleanup | Metal queue/MPSGraph/MLX 的具体 completion |
| 预算/报告/故障代码 | activation 与 VAE tiling 的精确公式 |

不要统一成一个 `void* load_block(i)` 然后假设所有模型等大小、同生命周期。
也不允许 adapter 内又启动一个不受框架约束的预取池；旧 pager 仅在 legacy 路径保留。

## 5. 默认性能与大内存模式

无新配置时只增加一次入口选择，不创建 descriptor/pool/guard、不开新线程、不发新 callback、不额外 unload。
共享数值函数不加逐 op 锁和时钟；需要不同循环时用 legacy wrapper 与 streaming wrapper 复用 kernel。

开启手动三槽意味着用户选择 I/O tradeoff，即使机器有足够 RAM，也不能把三槽自动变 resident。
高内存用户可以继续完全不启用新框架，或显式选 resident。bounded resident 首发仍 request-scoped；
若要保留跨请求 cache，必须另行完成 session-root accounting，不能为了 warm 性能删 terminal gate。

“不影响性能”验收是代码路径隔离 + 配对统计，而不是承诺任意新模式零时间开销。

## 6. 暂不做的内容

- 不支持 CPU/ANE 模型执行分配、多 GPU、多作业并行调度；CPU 读文件/解包仍是 GPU pipeline 的辅助步骤。
- 不隐式改变分辨率、帧数、steps、精度、LoRA、attention 算法。
- 不在请求中途改 K/G/P、换 plan、改 slot assignment 或自适应展开任意 DAG。
- 不保证任意 block 都可从 checkpoint 独立读取；需要 offline pack 的路线先完成 pack 工具和 identity 验证。
- 不以“无 swap”替代质量、lifetime 或全进程预算校验。

## 7. 请求生命周期（拟议）

```text
parse intent
  -> choose legacy OR explicit framework
  -> metadata describe
  -> resolve exact layout + preview resource requirements
  -> check layout execution eligibility
  -> if bounded: check memory certification + clean baseline + recheck same layout
  -> create fixed pools / execute passes / stage handoff
  -> stop I/O / join / drain GPU / release request pools
  -> immutable result + optional memory report
```

如果 baseline 重测后不足，只能拒绝同一布局，不能偷偷重新分配 prefix。结果必须同时记录 requested/resolved/actual 三份信息。
