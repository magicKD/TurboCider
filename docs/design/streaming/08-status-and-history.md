# 08 · 当前实现状态与历史索引

[目录](README.md) · [迁移队列](06-migration.md) · [验收](07-tooling-and-validation.md)

注意：本文第1–8节保留设计阶段的核对快照。之后已开始代码实施；最新事实、测试命令和缺口以 [13 实施进度](13-implementation-progress.md) 为准。

## 1. 本轮范围

2026-09-16，本轮只阅读当前工作树并重组/完善设计文档。未实现新的 StreamingConfig、通用 executor 或配置 v2，
未运行真实 GPU 性能测试，未修改 production registry。源码工作树有大量先前改动，不将这些改动全部归因于本轮。

## 2. 源码核对结果

| 项目 | 当前事实 | 与新框架的差距 |
|---|---|---|
| `MemoryConstrainedConfig` | 已有 enabled/Y/X/min_free/max_slots/tiling；presence flags | 无独立 manual StreamingConfig |
| profile/request | v1 profile、execution.memory_constrained | 不识别本文 streaming/profile v2 |
| `make_plan()` | constrained candidate 会按模型归一化旧 residency/budget | 需要 layout-first 分支，不再二次推 K/P |
| memory ledger/manifest/plan | site、instance、epoch、alias、reservation、release、digest 底座 | 模型全 required-site closure 未完成 |
| strict schedule | owner/cursor/poison/terminal/ABI 已实现 | 模型 semantic 事件覆盖不全，通用 group/pass 协议未实现 |
| shared adapter helper | `memory_schedule_adapter.h` 已实现 | 不能代替模型事件生产者 |
| LTX | 三槽上限；prefix+suffix、异步 loader、refill；粗 load/stage 事件 | 无 exact manual layout ABI、后台初分配、缺 step/block/slot 闭环 |
| H3 | `stream_slots[2]` 固定双槽与 xor；hook plumbing | 尚无完整 semantic emits，未有单/多槽通用映射 |
| Flux | existing MLX compiled/eager 路线 | component/group streaming 仍为设计 |
| Z-Image | existing MLX/格式-specific resident 路线 | ranged reader、lazy graph release、slot 接入待实现 |
| watchdog | owner checkpoint 驱动，不是独立后台采样线程 | 不能用“启动watchdog”措辞暗示持续无盲区 |
| production memory registry | `production_memory_capability_registry()` 仍为空 | 无真实 bounded release candidate |

参考源码：[Request](../../../native/core/contracts.hpp)、[memory config](../../../native/core/memory_contracts.hpp)、
[plan](../../../native/runtime/plan.cpp)、[session](../../../native/runtime/session.hpp)、
[profile](../../../native/platform/apple/profile.mm)、[API orchestration](../../../native/api/c_api.mm)、
[LTX](../../../native/models/ltx_runtime/ltx_blocks.c)、[H3](../../../native/models/h3_runtime/h3_dit.c)。

## 3. 验证结果的归属

前轮记录过 memory accounting/manifest/schedule/adapter/compiler/scheduler/watchdog/trace/execution/probe、H3 schedule 等逻辑测试 PASS，
native-only build PASS；最近 contract 输出是 **81 tests run，80 passed + 1 fixture skipped**，不是“81通过另加1跳过”。
H3/LTX Metal memory hook tests 在前轮执行环境返回 `SKIP: no Metal device is available`。
这只能说明当时执行环境不可用，不能据此断言实体机器没有 GPU。

本轮文档任务只验证文档结构、链接、JSON 例子和归档完整性；没有重新执行上述 runtime 矩阵。
上述历史测试也不覆盖新设计的 F0–F9 功能。不存在“新通用框架已通过 L0/L1”的结论。

## 4. 历史性能证据

| 记录 | 数据 | 重要限制 |
|---|---|---|
| [LTX lookahead](../validation/ltx-streamed-lookahead-2026-09-09.json) | 12 GiB target：denoise 11.030→8.233s；16 GiB：10.063→8.191s | 64 GiB M4 Max；64×64×9 retained-engine 小样本；同时改了 P；不是新框架/低内存认证 |
| [LTX residency](../validation/ltx-streamed-residency-2026-09-09.json) | 12 GiB denoiser target，worker peak RSS 约24.8 GiB | 不是进程 cap；独立 decoder 不在该 RSS 中 |
| [H3 pinned prefix](../validation/h3-ssd-pinned-prefix-dit-2026-09-08.json) | DiT denoise 17.306→15.180s；load+denoise约22.94s不变 | 64 GiB；256×256×22；每路线2次；不含文本编码/VAE/export；OS文件缓存未清 |

这些证据支持继续研究显式多槽/prefix，但不证明对 swap 的通用加速，更不证明新框架关闭时零回归。
数据口径和下一实验协议见 [07](07-tooling-and-validation.md)。

## 5. 文档迁移记录

旧入口保留：[memory-constrained-residency-scheduler-20260915.md](../memory-constrained-residency-scheduler-20260915.md)。
原稿迁到 [archive 原文](archive/memory-constrained-residency-scheduler-20260915.md)，完整 25,568 行，移动前后 SHA-256 相同：

```text
826b9ce3089a5878a84533fc1336ab92043d6c12367af33311772a1face3e6cf
```

原稿只作历史快照，不修正其中反复覆盖的结论、长段落、旧相对链接或未闭合 code fence；新规范已经独立整理，
不需要按原稿后缀章节逐层判定“哪个最新”。历史具体章节编号仍可在 archive 文本中搜索。

| 旧稿主题/章节 | 新的规范来源 |
|---|---|
| 预算/政策/默认路径（1–11、183、237、280、296、310） | [01](01-framework.md)、[05](05-memory-contract.md) |
| 配置/API/profile（267、308、318） | [02](02-configuration.md)：manual streaming 与 guard 分离 |
| schedule/slot/owner/completion（273–276、288–292、299、307、314） | [03](03-runtime-protocol.md) |
| H3/LTX/Flux/Z-Image（277–279、293–295、313） | [04](04-model-adapters.md) |
| 逐文件/PR/发布（284–285、303–304、311–312） | [06](06-migration.md) |
| 性能/仿真/验收（283、302、306、309、315–317） | [07](07-tooling-and-validation.md) |
| 实施记录（271、287、305） | 本文 + archive，不能当作新框架完成记录 |

## 6. 本次决策取代的旧草案

- 新框架不再只挂在 memory_constrained=true 分支；streaming layout-only 是独立模式，但仍需自己的执行资格。
- manual mode 不从 Y 搜索 K/P；recommendation 是未来可选离线工具。
- 不实现多套 `slot_policy/memory_tuning` 与 `execution.streaming` 并行权威；统一采用 02 的字段。
- “slot Vacant→ledger释放”被禁止；backing 与 content 有两套生命周期。
- D 不等于 K；新定义为最远未来 group 距离，范围 0…K−1。
- 低频 footprint 采样不是 OS 级连续硬上限证明；关闭 guard 不承诺零 swap。
- 不能从 tiny warm benchmark 推导所有模型/机器的加速，也不使用无需求分母的完成百分比。

## 7. 下一实际实现起点

先做 F0 配置/报告与 F1 metadata/manual compiler，全部保持 plan-only；然后做 F2 fake slot/fence。
有了可复现 exact layout，才把 LTX 当第一个真实 adapter 接入，H3 固定双槽作为第二个；不先全局替换当前 pager。
每个阶段只在对应主题文件更新，避免再次增长为几万行单文件。

## 8. 2026-09-16 本次深化（仍为设计，不是实施完成）

新增 [09 compiler](09-layout-compiler-spec.md)、[10 executor](10-executor-implementation.md)、
[11 实施任务](11-work-packages.md)、[12 验收执行书](12-acceptance-playbook.md)，以及两个合成/性能合同 JSON 示例。
补齐异构槽逐字段容量、class barrier、pass身份、固定大小控制面、预取游标、batch死锁防护、derived数据、mailbox容量与cleanup。
配置补充 residency 切换时的 stage replacement，避免沿用不合法的旧 streamed 字段。

本次源码复查特别确认：LTX initial loaders 可多线程并发且每次 refill 创建线程，旧三槽不能当作 Q=1；
`apply_scalar_conditioning_one()` 在 refill 后写派生数据；H3 含 active-block 与跨 block fusion，adapter 需要访问闭包限制。
这些是下一实现的约束，不代表已经修改旧实现。

本次不运行 runtime/GPU/pressure benchmark，不增加生产资格；新增性能数字是验收门槛，不是实测成绩。
后续从 [11](11-work-packages.md) 的 F0/F1 plan-only 开始；测试ID在对应实现PR中绑定真实文件与命令。

本次文档检查：16份当前 Markdown（含旧入口、不含归档长文）、119个本地链接、3段内嵌JSON、6份JSON文件通过；
独立复算compiler fixture的分组/槽容量/读入字节，确认manual与bounded布局完全相同，归档SHA-256保持不变，
code fence/尾随空白检查与`git diff --check`通过。该只读检查不是新框架runtime测试或性能验收。
