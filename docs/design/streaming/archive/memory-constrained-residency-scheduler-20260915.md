# TurboCider 内存受限模式：GPU 分层 Streaming/Loading/Offloading 设计稿

最新阅读入口（2026-09-16，multi-slot 推广修订）：先读第 306–312 节的总体方案，再读第 313–319 节的源码审计、
收敛后的 slot/lease 协议、planner、仿真/实测工具链、配置迁移及逐 PR 验收。该组章节修正前文的完成度、
swap 性能假设与配置草案；发生冲突时以第 313–319 节为准。本轮仅更新设计文档，不实施运行时改造。
通用 pager、simulator、配置 v2、resident-retained constrained candidate 均是待实现设计，不是已可调用功能。

状态：Proposal + scaffolding（配置、计划、首版 MemoryLedger/MemoryAdmission、LTX 原生 Metal buffer hooks v2、LTX 阶段级 host reservation、H3 GPU/host allocator hooks 与 Video VAE options 传播已部分落地；H3 classified upload API、Video VAE required GPU tensor 分类、text encoder 分类、DiT common-path 首批分类、Euler previous video/audio velocity 分类以及 decoded-host 按真实输出 commit 已进入当前工作树；planner 已移除 heuristic-fit 自动执行，H3/LTX 未命中 verified manifest 时固定输出 `hook_bridged + plan_only`。纯 C++ `MemoryManifest`/`MemoryCapabilityRegistry`、live-interval `compile_memory_plan()`、`MemoryStageScheduler`、capability preflight/authorization、请求级 `MemoryExecutionContext`、两阶段 `MemoryAllocationTxn`、site token、generation/domain 校验、有界 completion mailbox、H3 schedule gate-score host readback closure、metadata-only `memory_probe`、H3/LTX probe override、成功前 completion drain、terminal report/C API 和对应逻辑测试已经进入当前工作树，`c_api.mm` 已在 constrained 清理和大分配之前执行 capability preflight 并绑定 context admission。2026-09-16 当前工作树 native build 已 PASS；memory probe/manifest/plan compiler/accounting/scheduler/execution 测试全部 PASS，contract 为 81 PASS/1 fixture SKIP；H3/LTX 真实 Metal hook 测试因当前环境无 Metal device 明确 SKIP。生产 registry 仍为空，真实 checkpoint/device/shape capability 尚未认证；真实 Metal completion、所有 required-site closure、conditioning/control host allocation、真实 shape/framework envelope、LTX direct MTLBuffer/MLX/graph 与全阶段逐 allocation 闭环、service evidence writer、swap telemetry、Flux/Z-Image staged route 仍未完成，仍无 L2/L3 或可发布的真实成功证据）

日期：2026-09-16（本轮续修）

状态增补（2026-09-16）：request 级 swap counter baseline/checkpoint/terminal 采样、swapout fail-closed、terminal
metrics/JSON 字段，以及 capability 绑定的 certified explicit-epoch gate 已进入当前工作树。首页长状态行中把 swap
telemetry 列为“仍未完成”的描述已由本增补取代；尚未完成的是低内存机器逐请求 `swapouts_delta == 0` campaign、
service evidence writer、模型 required-site closure 和 H3/LTX 真实 Metal L2/L3 认证。

状态增补（2026-09-16，schedule PR-S1a/PR-R1）：共享 schedule C ABI、compiler、plan digest 绑定、strict cursor、sticky
poison/reentrancy gate、terminal helper、结果字段、request checkpoint 去重，以及 capability-record 到 authorization 的 schedule
传递已经进入当前工作树。第 287.2 节记录的 strict execution fixture 失败已通过 deterministic observer 修复；最新 schedule、
manifest、plan compiler、execution、contract 等 L0 测试和 native-only build 均 PASS。H3/LTX 目前只完成 schedule hook
POD 的 session/runtime 传递与 ABI 校验，尚未在真实 step/block/tile 循环发 semantic event；production registry 继续为空。

范围：Apple Silicon GPU/Metal/MLX 推理；必须计入 CPU staging、媒体输出及 worker 内存；不设计 CPU/ANE 推理分配与多作业并行
修订：本轮同步 LTX allocator hooks、阶段级 host reservation、H3 allocator/refill-slot hooks 与 effective-plan 服务路由的当前状态，并新增第 66–91 节实施规格：运行时对象、调度状态机、确定性 planner、逐模型改造、逐文件 patch 任务、预算实例、状态不变量、manifest、三队列 overlap、API/运维兼容、证据包与发布门禁。本轮继续新增第 92–111 节：当前缺口审计、可直接落地的 C++/C 接口、确定性候选算法、H3/LTX 逐调用链 patch、pager 兼容层、manifest 生成、服务生命周期、可观测性、patch stack、L0–L3 验收矩阵、Flux/Z-Image 逐文件接入、校准版本治理、参考事务伪代码与风险登记。第 112–120 节进一步按当前工作树补充：allocation call-site 注册表、H3 VAE/host 收尾、LTX route/vector/helper 收尾、manifest capability 硬门禁、调度器事务接口、逐模型接入模板、故障注入与本机/低内存实机验收方案。第 121–132 节继续固定工程实现蓝图：默认路径隔离、计划 IR 与精确峰值编译、调度器/权重 pager 事务、allocator 封装、H3/LTX/Flux/Z-Image 逐文件施工图、结果协议、基准工具和可拆分 work package。第 133–142 节记录本轮 hard gate 的真实落地状态，并补充 manifest schema、执行入口顺序、异步 ledger、H3/LTX 收尾施工、pager 并发算法、Flux/Z-Image 边界和可直接执行的验收矩阵。第 143–150 节进一步把当前 scaffolding 收敛为可执行的代码接线合同：capability probe 的准确时序、`c_api.mm`/`ModelSession` 改造、H3 schedule/host allocation closure、LTX direct Metal/vector closure、Flux/Z-Image 的 staged 边界、scheduler/pager token 语义、错误协议与逐 PR 验收门禁。第 151–160 节依据 2026-09-15 源码复核继续细化 H3 API 迁移、请求级 runtime context、allocation-site 可达性注册、manifest 生成器、scheduler/pager ABI、模型首发 patch 模板、自动化 evidence runner 以及需求到测试的追踪矩阵；其中第 151 节记录的构建失败已由当前工作树修复。第 161–172 节进一步固定 context 绑定与失败 disposition、单请求/未来 broker 边界、planner/live interval 规则、H3/LTX/Flux/Z-Image manifest、pager 并发、逐文件改造、结果协议、fault campaign、回滚和签核合同。第 173–185 节基于最新工作树新增 metadata-only probe 的真实接线、sidecar 安全与 TOCTOU 合同、H3/LTX 精确 patch、候选降级图、completion drain、逐模型资源图、公式版本、性能 SLO、probe 专项验收和实际证据记录。第 186–196 节固定容量合同、runtime token/mailbox、H3/LTX 当前差距与发布 DoD。本轮续修新增第 197–208 节，进一步补齐 control/data plane 边界、拟增 C++ 接口、两阶段 admission、manifest closure report、H3/LTX/Flux/Z-Image 施工矩阵、MLX/Metal 统一内存约束、并发/取消不变量、结果协议、L0–L3 测试、性能调优、feature gate 和 handoff 清单。配置/计划脚手架的真实状态见第 39、91、107、112、121、133、151、161、173、185、197 节；其余明确标记为“建议/拟增”的接口仍为设计合同。源码核对基线为 `804925c` 加当前工作树改动。

最新修订说明：第 209–226 节进一步审计真实 allocation enforcement、framework upper、capability identity、
evidence 和逐文件收尾；第 227–236 节将当前 `memory-runtime-v3` 已落地事实与后续设计分开，新增 adapter
reserve/commit/retire/completion 接线合同、显式 execution state/watchdog 建议、三队列 overlap、H3/LTX
阶段事务、Flux/Z-Image plan-only 边界、结果协议、逐 PR 修改清单、L0–L3 验收和默认路径回滚策略；第 237–253 节
同步 Y/X/B/S 数字合同、explicit epoch、LTX hook v2、capability identity、required-site closure 和 terminal
report 的当前实现与发布阻塞；第 254–270 节进一步补齐 schedule schema v2、site registry、swap telemetry、
evidence writer、H3/LTX 首发 candidate、Flux/Z-Image 认证阶梯、逐文件 patch 队列、性能模型、quarantine runbook、
API/schema、CI/L0–L3 验收和 release/rollback 门禁；第 271 节记录已落地的 request 级 swap telemetry 与
certified explicit-epoch gate；第 272–286 节进一步冻结共享 schedule ABI、严格 cursor、explicit allocation
instance、H3/LTX 函数级接线、Flux/Z-Image opaque framework envelope、evidence writer、逐 PR 施工单和验收硬门；
第 287–304 节基于 2026-09-16 当前代码和新增测试继续补齐 schedule 事务中毒语义、request 编排、H3/LTX
事件矩阵、预算/overlap 控制、估算校准、故障恢复、证据字段、CI/实机 campaign 和逐 PR 文件级修改建议。

阅读导航：第 1–11 节为方案总览；第 12–16、21–22 节为接口和模型接入；第 28–36 节为账本、MLX/worker 生命周期和调度算法；第 39–51 节为当前实现状态和 patch 级接入；第 52–61 节为 allocator、pager、服务和性能细化；第 66–78 节给出基于当前工作树的下一阶段实施规格、逐模型改造和可执行验收；第 79–91 节进一步固定状态不变量、manifest/plan 数据结构、三队列 overlap、H3/LTX 具体代码合同、跨模型认证门禁和证据包；第 92–111 节是完整实现合同；第 112–120 节是基于当前工作树的收尾施工图；第 121–132 节是接近代码 review 和任务拆分的工程蓝图；第 133–142 节是 hard gate 落地事实、接口细化和验收清单；第 143–150 节是 capability 到真实执行闭环的施工顺序；第 151–160 节是基于实际 allocation site 的 review-ready 增补；第 161–172 节是 context、planner、pager 和验收施工合同；第 173–185 节是 probe/sidecar/接线、completion drain、验收与证据增补；第 186–196 节是容量合同与源码差距；第 197–208 节是接口、模型、并发、测试和发布 handoff；第 209–226 节是 enforcement、framework upper、identity 和 evidence 审计；第 227–236 节是当前应优先阅读的 runtime-v3 事实、模型接线、代码修改和验收版本；第 237–253 节是 Y/X/B/S、explicit epoch、LTX v2、closure 和 report 状态；第 254–270 节是下一阶段 schedule/site/swap/evidence 施工、模型 patch、性能、runbook 与发布门禁；第 271 节是最新已实现 swap/epoch 门禁记录；第 272–286 节是当前最细的代码级施工与验收版本。建议实现者优先读 227、237、239、249、254、271、272–285 节。若前文的“当前状态”与后文冲突，应以后者为准。第 9、17–20、25、37、62–65 节为原始验收与交付框架。第 38 节是平台/源码依据。

最新状态入口：当前 runtime 已实现 instance/epoch/alias enforcement、record 注入的非零 framework upper、
constrained generic allocation 禁止、site `Reserved/Active/Pending/Cached` 生命周期、显式 execution state、
owner-polled bounded watchdog 和
`memory-runtime-v3`；LTX hook v2、terminal report/C API、request 级 swap telemetry 和 certified explicit-epoch
gate 已实现；生产 registry 仍为空，H3/LTX 的真实
required-site closure、Metal completion 和低内存 L2/L3 证据尚未完成。详细事实以第 227、249、252、254 节为准。
第 271 节记录最新已落地门禁，第 272–285 节定义下一轮实现合同；第 287–304 节是基于当前源码和新增
strict-schedule 测试结果形成的实施修订，出现冲突时应以后者为准。

实现状态声明：本文件不是“已支持所有模型”的发布说明。当前仓库已经有
`MemoryConstrainedConfig`、profile/request 解析、预算计算、plan JSON 输出、
首版 `MemoryLedger`/`MemoryAdmission`、阶段边界 footprint 采样，以及 H3/LTX
受限请求的 candidate normalization。LTX 原生 C/Metal 的集中 `MTLBuffer`
分配路径已经接入 reserve/commit/release hooks，并开始标注 weights/activation
等类别；H3 session 已能同时安装 `h3_gpu_memory_hooks` 与
`h3_host_memory_hooks`，把共同的 allocator domain/generation 传入 C runtime，
并已把 `h3_gpu_options` 传播到 DiT、纯文本编码器和 Video VAE 的 resident/
chunked/single-shot 路径。H3 host 侧目前纳管 joint video/audio latent、RGB8 输出，
并用 `h3.vae.decoded_host_envelope_v1` 包住 Video VAE 的 F32/tile 主峰值；decode
返回后已开始按真实 frame shape 做 checked commit，但该 envelope 仍是启发式聚合
上界，不等于逐 allocation 闭环。Video VAE required GPU tensor 和 text encoder
required GPU tensor 已改用 classified API，DiT common path 也已迁移一批 activation/
conditioning/converted-weight 调用点；Euler previous video/audio velocity 也已经改为
classified activation。剩余量化/ANE/Core ML 可选分支与其他旧调用点仍需做“分类或首发
不可达”证明。conditioning/layout/tokenizer
等 host allocation 也尚未全部纳管。LTX MLX VAE/upsampler、MPSGraph temporary、
direct MTLBuffer、MLX cache、
全部 host staging 和媒体输出同样没有形成可认证的全阶段闭环，真实 H3/LTX L3
全流程证据尚未完成。因此第 39 节以前出现的“建议/拟增/应”均为设计合同；第 39 节
列出的已实现项才是当前代码行为。任何模型在没有真实 `must_succeed` 证据前
都不能把 `memory_policy` 标记为 stable，也不能仅凭 planner 数字宣称不会进入
swap。当前 H3/LTX 路线在无 verified manifest 时明确输出
`execution_supported=false`、`capability_level=hook_bridged`、
`certification_state=plan_only`；planner 的启发式 estimate 不再具有执行放行权。
当前工作树还包含 `native/runtime/memory_manifest.*`、`memory_plan.*`、
`memory_scheduler.*`：它们已经提供稳定 manifest/candidate digest、release/experimental
registry gate、half-open live interval 峰值编译、alias backing 去重、stage lease、pending GPU
completion 和 normal/tight/critical 压力状态机。它们目前仍是独立 scaffolding：生产 registry
为空，`tc_engine_generate()`/`tc_engine_prepare()` 尚未从持有模型根目录的 session 取得 exact
checkpoint manifest，因此受限执行继续在第一道 hard gate 处拒绝。这一状态是安全的，不能
为了演示执行而用 synthetic record 或环境变量绕过。

## 1. 摘要与结论

建议新增一个显式 add-on：`memory_constrained`。关闭时完全沿用当前本机默认策略、默认 residency、缓存和调度路径；开启时由 TurboCider 自己做预算、准入、分层驻留和回收，禁止把“等待系统进入 swap”当作内存管理策略。

核心结论：

1. **优先采用显式逐层 streaming/loading/offloading，swap 只作为操作系统最后兜底，不应作为产品能力。** Apple Silicon 是统一内存，GPU buffer、MLX array、CPU staging 和文件缓存最终竞争同一物理内存；把对象转成 host 或依赖压缩 swap 不等于降低峰值，且会产生不可预测的 page-in/page-out 延迟。
2. 本文约定 `X` 是**预留比例**，不是使用比例：`B = floor(Y * (100-X)/100)`。例如 Y=16 GiB、X=15%，使用目标为 13.6 GiB；如果用户希望“最多使用 Y 的 85%”，对应 X=15%。`B` 约束完整推理进程集合的预算，包括已有会话基线；`Y-B` 不可拿来继续放 pinned 权重。大分配前的准入可以硬拒绝，但 macOS 上的瞬时进程 footprint/全系统 swap 不能靠这个开关获得操作系统级绝对保证。
3. 预算必须覆盖**全请求峰值**：模型权重、activation/workspace、输入/输出、Metal/MLX cache、staging/refill slots、编译/加载临时峰值，而不只是当前 H3/LTX denoiser block。
4. 调度采用三级时间尺度：作业级 admission、阶段级组件生命周期、block/partition 级双/三槽预取。预取只能在预算和 last-use 证明安全时进行；内存不足时先停止 look-ahead，再缩短 pinned prefix，最后降级到更小 tile/更少并发，仍不可行则 fail-closed。

## 2. 现状与问题边界

本机验收基线（2026-09-15）：Apple M4 Max，`hw.memsize=68719476736`（64 GiB），macOS 26.6.2；系统 swapfile 总量 4 GiB、当时已使用约 2.83 GiB。该 swap 使用量可能来自更早的任务，属于历史采样，不是当前实时值。验收应比较 `vm_stat` swapins/swapouts 单调计数的增量、压缩与压力；`vm.swapusage.used` 的净变化只是辅助，净值不变也可能发生换入换出。

仓库已有可复用基础：

- `native/runtime/block_residency.{h,c}`：按 `activation_reserve + (pinned + slots) * block_bytes` 构造计划；允许 full-resident 的调用可返回零 refill slot，H3 wrapper 则保持至少一个 streamed block。
- H3 C runtime：BF16/I8 shard、SSD `pread`、双 typed refill slot、动态 pinned-prefix。
- LTX C/Metal：denoiser block streaming，最多三个 look-ahead refill slot；C++/MLX 另有 `BlockCache` 的 pinned prefix + rotating refill 和显式权重输入的 `CompiledBlockForward`。两条路线均未等价于全请求 cap。
- `native/runtime/residency.{hpp,cpp}`：resident/component-staged 组件策略和估算校验。
- `Request.memory_budget_bytes`：现为模型/denoiser hint；`Request.residency` 由模型模块验证。
- `RunResult.BlockResidencyMetrics`：已能记录 block/slot/读取/等待指标。
- `native/runtime/memory_accounting.*`：首版 reservation/lease、唯一 backing 去重、pending-release/cache 和阶段边界 footprint 观测；尚未逐 allocation 接入模型后端。

现状缺口：

- budget 不是全进程 RSS/物理内存的 OS 硬 cap；现有 `MemoryAdmission` 已将进程 baseline 与 planned increment 组合成 concrete-allocation ceiling，并提供阶段采样，LTX 原生 Metal buffer 主路径已开始逐 allocation admission，但尚未覆盖 H3、全部 Metal/MLX/native malloc、graph temporary 与 host/media arena。
- H3/LTX 能 block-stream，Flux/MLX 和 Z-Image GGUF 不能统一响应 budget；部分模块明确只接受 resident。
- 特别注意：VDN H3 的模块契约虽接受 `streamed`，当前 MLX checkpoint loader 仍创建并持有整套 safetensors 的 tensor map；在完成 pager 化之前不能把它视为真正的低内存 streaming 能力。
- 已有统一 `MemoryConstrainedConfig` 与基础诊断，但尚无持续 sampler、watermark pressure controller、运行中回退状态机和完整 trace。
- load/compile/warmup 的临时峰值可能与 denoise 同时发生，导致看似满足 block budget 但仍触发 swap。

本设计不改变模型数学、采样步数、默认精度和默认执行路径。整文件 `mx::load_safetensors` 可能涉及 mmap/lazy loading，不能直接断言立即读入全部物理页；这里的问题是没有按 block 限制持有、materialize 和释放范围。

## 3. 用户接口与语义

当前沿用现有 JSON CLI，没有新增第二套命令行参数解析：schema-v1 顶层接受下列可选对象；schema-v2 放在 `execution.memory_constrained`。以下字段是本设计唯一规范命名。

```json
{
  "memory_constrained": {
    "enabled": true,
    "limit_bytes": 17179869184,
    "buffer_percent": 15,
    "min_free_bytes": 1073741824,
    "max_refill_slots": 3,
    "allow_quality_preserving_tiling": true
  }
}
```

- `limit_bytes=Y`：JSON 只接受正整数 bytes，拒绝 bool/小数/NaN/负数，沿用当前 `byte_count` 的范围（不超过 2^50）；App 的 GiB 输入在 SDK 边界转换。Y 大于本机物理内存时拒绝，不静默截断。
- `buffer_percent=X`：整数 5–30，默认 15；硬件最终推荐值需验收，不是理论最优常数。
- `min_free_bytes`：全系统准入的额外保留目标，不属于进程内存，不在 B 内重复扣除。它也不是承诺系统 free-pages 计数永远大于该值，见第 28 节。
- `max_refill_slots`：1–3，默认 3，仅为上限。backend 声明的已验证槽位集合才可选；H3 现有路线只认证双槽。`slots=1` 在没有对应执行器时应返回不可行，不能假装已实现。
- `allow_quality_preserving_tiling`：只授权已有质量证据且保持请求语义的 decode/workspace 分块，不授权切割全局 DiT attention 或修改图片尺寸。
- 不公开 `fail_if_infeasible=false`：受限模式一律 fail-closed，不提供转回 swap/unbounded 的后门。
- 对象缺失等同关闭。请求显式 `enabled=false` 覆盖 profile；附带字段仍做类型校验，但不修改执行行为。`enabled=true` 缺 Y 则拒绝。
- 同一有效请求同时设置 add-on 和非零旧 `memory_budget_bytes` 时返回 `memory_policy_conflict`。旧字段是 denoiser hint，新字段是全请求目标，不能静默互相覆盖。若来自 profile，错误列出来源并建议移除旧字段。
- add-on 下 `residency` 是候选偏好，实际策略写入 `effective_residency`；原始请求不覆盖。模型数学校验与有效策略能力校验要分开，避免先被旧 resident/streamed 白名单拦截。
- GPU-only 包括 encoder：明确的 `gpu_ane`、ANE/encoder manifest 或 hybrid profile 与首版 add-on 冲突应拒绝。`auto` 在此模式只从 GPU 候选选择；不能仍由 `select_acceleration` 偷偷切到 ANE。

`EffectiveMemoryPolicy` 在请求准入时冻结；系统压力变化可生成新的 schedule revision，但不能改变 Y/X、模型身份、采样步骤和质量合同。所有有效字段参与 policy digest；原始字段来源另存用于解释。

## 4. 预算模型

### 4.1 全请求模型与阶段峰值

```text
B = floor(Y * (100-X) / 100)
R_proc = 控制进程基线 + 校准后的未归因/框架开销上界
L(t) = W(t) + S(t) + A(t) + C(t) + V(t) + O(t) + K(t) + T(t)
M_plan = max_t [R_proc(t) + L(t)]
准入要求：M_plan <= B，且独立的系统可用内存/GPU工作集检查通过
```

W=非 block trunk 与 pinned 权重；S=refill backing 和 I/O staging；A=激活/workspace；C=conditioning/latent；V=独立 VAE 阶段权重与 arena；O=输出及媒体队列；K=allocator 空闲缓存/编译缓存；T=转换/重载临时副本。按 backing 唯一身份去重，不能同时把同一 VAE 权重算入 W/V，或把 MLX buffer 又加一次 Metal allocated size。

峰值是**同时存活集合的最大值**，不是所有阶段各自峰值之和。若 encoder、denoiser、VAE 串行，不必将三套权重同时预留；但跨阶段保留的 conditioning、latent、host output 必须持续计入。预加载下一阶段时才对重叠存活集合求和。

B 内还包括 `R_proc`；它不是 `min_free_bytes`，也不是 Y-B。用户预留、进程开销、系统余量三者不得重复扣除。具体账本与例子见第 28 节。

### 4.2 估算来源与置信度

真实 tensor table 给出精确存储大小；allocator alignment、运行时 pack/dequantize、attention 算法、graph workspace 则需解析模型配置及校准。每项记录 `exact|validated_envelope|heuristic`，上下界、样本数、shape/backend/runtime 身份与有效范围。有限样本的 P95 不是硬上界，不能单独用于受限模式准入。

`plan REQUEST.json` 不带模型路径，当前只能做合同校验和粗估；它不得报告已经准入。带 checkpoint 的 metadata-only probe 在生成前建立完整计划；拟增带模型路径的 estimate 入口见第 21 节。不为估算先整包加载模型，也不自动运行未准入的 warmup。

### 4.3 水位、反馈与保护

使用当前受控承诺量 `M_committed` 和当前观测值；历史 peak 只用于报告，不能拿历史峰值持续触发回收。普通计划可接近 B，但仅在**观测异常、系统压力或下一项分配无余量**时收缩 pinned，避免最大化 pinned 后立即因阈值反复逐出的振荡。

| 水位（默认实验值） | 附加行为 |
|---|---|
| <0.80B | 正常执行已准入计划 |
| ≥0.80B | 禁止新的可选 warmup/额外缓存；既定必需预取不必取消 |
| ≥0.90B | 禁止额外阶段并载，重新检验下一次 reservation；有压力时缩小预取 |
| ≥0.97B、pressure warning/critical | 暂停可选提交，安全点 drain，必要时收缩或退出 |
| 预测下一项会超过 B、观测超过 B | 分配前拒绝；已发生的观测越线记为违约，后续回收不能改写为通过 |

上升用当前/预测值，恢复至少低于原水位 5 个百分点并稳定三个安全点；系统 critical 不等待样本。任何一个 poll 都不能保证两个采样之间没有瞬时超限，因此主要防线仍是分配前 reservation + bounded graph。

## 5. 总体架构

```text
Request + DeviceInventory + ModelManifest + Shape
                  |
         Cost/Residency Planner
                  |
        EffectiveMemoryPolicy
                  |
       ResourceCoordinator admission
          /          |            \
   StageScheduler  WeightPager   PressureController
      |                |              |
  text/DiT/VAE   pinned + refill   watermarks/events
                  |
              GPU queues/events
```

新增/扩展模块：

- `native/runtime/memory_policy.{hpp,cpp}`：配置解析、B、watermark、预算校验。
- `native/runtime/memory_accounting.{hpp,cpp}`：分类 ledger、lease、peak/attributed/unattributed bytes。
- `native/runtime/residency_planner.{hpp,cpp}`：组件+block 联合规划；复用 `make_block_residency_plan`。
- `native/runtime/weight_pager.{hpp,cpp}`：统一 source、双/三槽、`pread`/mmap、prefetch ticket、event pin。
- `native/runtime/pressure_controller.{hpp,cpp}`：水位状态机、停止预取、边界重规划、诊断原因。
- `native/runtime/schedule_trace.{hpp,cpp}`：阶段/block 的 load、compute、wait、release、bytes 事件。

`ModelSession` 增加可选能力查询：`memory_capabilities()`、`estimate_memory(shape, policy)`、`build_residency_plan()`。旧 session 若未实现完整估算/受控执行能力，add-on 一律 fail-closed；即使 resident 粗估低于 B 也不能假装已有硬约束。关闭 add-on 时直接走原执行路径，零逐层账本开销。

## 6. 调度算法

### 6.1 作业级 admission

1. 解析模型、operation、shape、steps、audio、LoRA、backend。
2. 计算全请求 upper-bound；并行枚举 resident、component-staged、streamed、tiling 等合法候选。
3. 对每个候选计算分阶段存活峰值、首次装载、逐步 I/O、compute、未隐藏等待和输出质量合同。固定 resident、staged、1/2/3-slot、若干 group-size 枚举即可，无需并行代理或通用图编译器。
4. 过滤全请求 `M_plan>B`、系统准入失败或能力未验证的候选；只在相同质量合同内按端到端成本排序。
5. 建立 B 容量的 job ledger，逐阶段从中获得子 reservation；不要把 B 整体计费后又将子分配加一遍。沿用现有同 GPU generation 互斥，不为本功能增加多作业并发。

### 6.2 阶段级调度

默认顺序：text/image encode → denoise → VAE/decode → export。只在 `next_stage_load + current_stage_peak <= B` 时提前加载下一阶段；否则在 current stage 完成 event 后释放权重再加载。

- text/vision encoder：完成后保留 conditioning，释放 encoder weights；缓存命中仍要计入 C，不重复占用 W。
- denoise：保留必要 trunk/prefix，block suffix 走 pager；最后一个 denoise step 完成后按 last-use 释放 transformer，再加载 VAE。
- VAE/audio/video：视频和音频 arena 不同时创建，除非静态证明峰值满足 B；优先 tiling/chunking，保持输出质量契约。

### 6.3 block/partition 级调度

对每个 block 建立 `use[i][step]` 和 bytes。规划：

```text
capacity = floor((B - R_proc - live_non_block_floor) / uniform_block_bytes)
if capacity >= all_blocks: fully resident
else choose pinned prefix + N refill slots, N<=max_refill_slots
```

上式只适用于等大 block 且 floor 已覆盖该阶段 scratch/转换的简化情形。真实模型按逐 block bytes 和类型化 slot envelope 计算；full-resident 使用零 refill slots，streamed 才要求 suffix 非空。H3 固定双槽、LTX 最多三槽需要各自能力验证后选择。GPU 计算 block i 时：

1. slot `current` 被 event pin，不能回收；
2. I/O queue `pread` block i+1 到 free slot；
3. 转换/pack 与 GPU compute overlap；
4. i 完成且下游 event signal 后释放 slot；
5. 若 wait 时间连续超过阈值，增加 look-ahead 直到预算水位允许；若内存上升则反向减小窗口。

不得在每层同步读取 host 标量；只在 block/stage boundary 记录 trace。

### 6.4 压力与回退状态机

`NORMAL → THROTTLED → DRAINING → REPLAN → RESUME/ABORT`。先停止可选预取/warmup，释放已无引用的 cache/组件，再在完成边界减少 pinned 或选择该 backend 已认证的更少 slots。不是所有 backend 都能在线 resize；首版只在 step/stage 边界变更，不能安全恢复则 abort。tile 方案在 decode 开始前选择；正在解码中不得直接切换 halo/时间状态。内存模式不自动启用新量化或稀疏，即使用户允许 approximation，也先锁定同一模型变体并单独验证数值路径。

## 7. 模型适配矩阵（先 GPU）

### H3 / FastH3 / VDN

- 第一优先级：复用 `native/models/h3_runtime` 的 shard/block streaming；把现有 `ssd_memory_budget_bytes` 映射到统一有效预算中的 denoiser 子预算。FastH3 MLX/VDN 在 checkpoint loader 改为按 block 读取前，只能使用 component-staged 或报告型 dry-run，不能宣称已满足全请求上限。
- 新 planner 先建立 text/denoise/decode 各阶段 floor，只从 denoise 子预算扣除当时仍存活的 conditioning、trunk、scratch、output 等；不把已经释放的 encoder 与尚未加载的 VAE 权重重复扣入。全请求仍必须证明每一个阶段均可行。
- C/Metal H3 BF16 双槽最先接入。C/I8 quant-cache 与 MLX affine INT6 是不同格式/内核合同，不能共用一张量化 bytes 表；M4 Max 不自动开放要求 M5 能力的 I8 路线。VDN 的同 block branch 与 INT6 base、norm、modulation 一起 lease；solve 的跨 token 双向状态属于当前层 workspace，不可误解为可改层序的流水线。
- 默认 add-on 下建议 `component_staged + streamed DiT`；无 add-on 的 `resident/component_staged` 原行为不变。

### LTX 2.5

- 复用 `native/models/ltx_runtime/ltx_blocks.c` 的 48 blocks 与最多三 refill slots。
- 预算分区：text connector、stage1/stage2 activation、video/audio VAE、输出 arena；streamed 当前仅 video-only text-to-video，I2V/audio 仍 fail-closed，直到有完整证据。
- Stage1/Stage2 共享同一组 48-block denoiser 权重，不要人为复制两套。scratch/latent geometry 不同；Stage1→upsample→Stage2 的过渡峰值也需预算。首版共用较保守 prefix，后续才验证按阶段调整；不要无理由在阶段间重载全部权重。
- 允许 quality-preserving VAE tiling 作为内存降级；Sol/sparse 仍是独立 approximation 开关，不能由内存模式偷偷开启。

### Flux 2 Klein 4B/9B

- 当前 `flux2/pipeline.cpp` 只有 resident/component-staged 语义；第一阶段只实现 component-staged：text encoder 完成即 release，denoiser 完成再加载 VAE。
- 第二阶段才做 block streaming：将 dual/single transformer layers 切成合法 layer groups，使用两槽 Metal buffer；需要验证 attention KV、RoPE、conditioning 是否可跨 group 保持，不能按单 tensor 机械切片。
- 4B 先于 9B；预算不足先降低 pinned groups/安全并发，再选择已有证据的 decoder tiling。全局 DiT attention 不能靠任意切 latent tile 保持等价；activation floor 仍超预算时拒绝，并建议用户另行降低画布。

### Z-Image / Z-Image GGUF

- 原生 MLX GGUF 当前明确拒绝 streaming，普通 Z-Image 也未开放 component-staged；须先完成组件拆分与能力声明，才可选择 staged，不能认为添加字段即获支持。
- 第二阶段在当前 native MLX/GGUF 后端新增按 tensor offset 的 group reader；保留自有执行器，不引回旧 sd.cpp worker，不通过改名 checkpoint 扩展类型支持。
- 当前 native module 接受 Q8_0/Q4_0/Q4_1 或 floating GGUF，mixed K-quants 不属于本功能。按真实 tensor type 验证，不能照搬旧文档的 Q3_K_S/Q4_K_M 成绩。先做 Q8_0，再 Q4_0/Q4_1；记录原始 bytes、runtime pack 和 dequantized 临时峰值。

### 其他模型

模块若未声明完整受控执行能力，add-on 返回 `memory_policy_unsupported`。有已验证 resident envelope 的模块可以只支持 constrained-resident；不要求所有模型必须 streaming。

## 8. 代码修改清单

1. `native/core/contracts.hpp`：引用新增 `memory_contracts.hpp` 的输入配置；`EffectiveMemoryPolicy`/capability 等执行类型放 runtime（第 12 节）。保留旧 JSON/C 函数兼容，默认 disabled。内部 C++ 布局/虚表改变必须整体重编。
2. CLI/服务 JSON parser：校验 Y/X、单位、错误码；在 `RunResult` 输出 policy digest、effective budget、peak estimate、watermark transitions。
3. `native/runtime/plan.cpp`：把现有模型估算升级为分项 `MemoryEstimate`，保留旧 `memory_estimate_bytes` 作为兼容总值。
4. `native/runtime/residency.cpp`：实现组件 floor 与 block plan 联合；`validate_budget` 从“physical >= estimate+4GiB”改为 add-on 专用 admission，默认路径保持原逻辑。
5. `native/runtime/block_residency.*`：增加可选 `reserved_non_denoiser_bytes`、动态 slots 上限和解释字段；旧 C ABI 函数行为不变，新增 v2 函数。
6. H3/LTX：接入统一 pager/accounting；保留现有双/三槽和 pinned-prefix 结果，增加 full-request peak 统计。
7. Flux/Z-Image：先实现 component-staged adapter 与 capability declaration；未完成 block streaming 时拒绝“budget 伪装成已支持”。
8. MLX/Metal backend：封装 allocator cache trim、buffer byte ledger、event-safe release；任何 release 必须等待最后使用 event。
9. `native/runtime/execution.cpp`/service：MemoryLease admission、取消时 drain、失败时 deterministic cleanup；同 GPU 大作业互斥策略保持默认。
10. 测试：新增 policy 单测、planner golden、模型 capability contract、压力注入、ABBA 质量/峰值矩阵。

## 9. 验收方案

### 9.1 单元与静态验收

- `B=floor(Y*(100-X)/100)` 的边界、整数溢出、X=0/100 的非法值拒绝、Y 小于 minimum 的错误码。
- block plan：budget 足够时 full resident；不足时 pinned prefix + 至少一个 streamed block；slots 不超过上限；显式 prefix 不得超预算。
- 生命周期：in-flight event 未完成时不得释放；取消/异常后所有 leases 归还。
- add-on disabled 的 golden：同 request/seed 输出、选择、时间路径和日志字段不改变。

### 9.2 真实性能/内存矩阵

每个模型至少测试：resident baseline、component-staged、constrained（Y=设备物理内存的 60/75/90%，X=10/15/20%）、冷启动和 warm cache、最小/默认/大画布或帧数。记录：wall、denoise、load、I/O bytes、unhidden wait、peak attributed bytes、process footprint、swap in/out、GPU busy、输出 hash/质量指标。

### 9.3 硬门禁

- 受控账本：任一分配点 `committed <= B`，没有测后回收来掩盖越线。
- 进程观测：完整作用域 sampled footprint 目标 <=B；任何采样超 B 判本次失败，即使仍小于 Y。采样无超限不证明所有瞬间受 OS 强制 cap；报告 `enforcement_scope`、采样间隔和未知项。
- swap：安静环境中 swapouts 增量目标为零，swapins 也需报告；存在已有换出页、外部任务干扰时标记环境受污染/需重测，不将全局活动强行归因本请求，也不因净 swap.used 不变判通过。
- 数值：只改 residency 的同 backend 路线原则上 latent/decoded RGB exact；若不同等价 kernel 的浮点舍入改变，必须预先单独认证 tolerance，不能临时放宽。量化模型与它自身同 checkpoint baseline 对比；MP4/PNG 文件 hash 只作辅证，编码 metadata 不代表像素变化。
- 默认关闭：同 workload ABBA 至少 3 轮（每配置 6 样本）median wall 回归 <=2%，同时检查输出、route、cache/compile 与 load 顺序；有噪声则加样本，不用单次结果过门。
- 支持矩阵应区分 `must_succeed` 和 `must_reject`。安全拒绝不是模型低内存运行能力验收成功。

### 9.4 本机命令与执行状态

当前存在且可单独运行的无权重测试（不要误当成 pytest 测试函数）：

```sh
python3 -B tests/native/test_h3_streaming_policy.py
python3 -B tests/native/test_ltx_streaming_benchmark.py
python3 -B tests/native/test_native_gguf.py
```

`test_staged_residency.py` 是需要 `--model --manifest --output` 的集成脚本，还涉及 hybrid；本 GPU-only 受限模式应新增独立脚本，不能直接丢给 pytest 或将它当成受限模式证据。

当前 CLI 是 JSON 文件入口，后续扩展仍沿用下列命令结构。parser 已接受新字段；H3/LTX 的匹配请求会进入粗粒度 `streamed_adapter_v1`，不匹配、估算未知或预算不足的请求仍按稳定错误 fail-closed。只有完成 allocator ledger 和 L3 证据后，才可把该实验路径描述为正式受限能力：

```sh
build/native/turbocider plan tests/fixtures/memory/constrained-flux4b.json
build/native/turbocider generate /absolute/model/root tests/fixtures/memory/constrained-flux4b.json
```

`plan` 是静态说明；生成前还要基于真实 checkpoint 准入。不存在 `./build/turbocider --plan --memory-limit ...` 的现有接口，不将它列为可执行验收命令。

每份 benchmark JSON 绑定源码、OS/MLX/设备、checkpoint、shape/token 数、policy 与 plan revision、冷热状态、分项峰值、swap 单调计数、质量和 timing。测试工具退出码必须区分 PASS/FAIL/SKIP/CONTAMINATED。

## 10. 分阶段落地

**Phase 0（不改默认）**：实现 config/ledger/planner 单测和 dry-run explain；内部诊断只观测。此时对外 `enabled=true` 未支持路线仍必须拒绝，不能报告模式已生效却只监控不约束。

**Phase 1（安全可用）**：component-staged admission + H3 C/LTX C 的分配前约束；目标不引发新增 swap、不可行则 fail-closed；Flux/Z-Image 在 adapter 完成后只开放 staged。

**Phase 2（性能）**：prefetch autotune、Metal/MLX cache trim、LTX VAE tiling、H3 retained session；完成冷/热矩阵。

**Phase 3（扩展）**：Flux layer groups、Z-Image GGUF stream-layers；每个模型独立质量/性能 gate。

发布原则：add-on 关闭时零行为变化；add-on 开启时宁可提前拒绝，也不能让系统先分配到 swap 再“自动恢复”。

## 11. 关键风险与未决项

- Apple/MLX/Metal 对 allocator cache、统一内存 pressure 和 GPU 工作集的可见性并非完全等价；因此“进程绝不超过 Y”只能在完成平台实测后只能描述已测试范围内满足目标，不能升级成操作系统级绝对保证。
- mmap/file cache 可能制造额外物理压力；应优先 `pread` 到受控 shared buffer，并把 page cache 视为不可控余量。
- MLX graph evaluation 可能延迟分配；planner 需要 warmup fixture 或保守上界，不能只看 host-side tensor size。
- 编译/首次加载峰值可能高于稳态；必须单独 admission，不能与生成共享同一未计量预算。
- 多任务并发、外部 GPU/视频应用和系统内存压力会改变可用量；MemoryLease 需要在阶段边界重新检查，必要时暂停/取消。

最终产品文案应使用“预算约束的低内存模式、避免主动触发 swap、不可行时安全失败”，不要使用未经证据支持的“绝对内存上限”或“零物理内存增长”。

## 12. 建议的类型与接口（可直接作为实现起点）

### 12.1 配置、估算与分层头文件

新增 `native/core/memory_contracts.hpp` 放不依赖 backend 的输入配置，供 `contracts.hpp` 包含；实际策略、estimate、lease 放 `native/runtime/`。避免 `contracts.hpp → memory_policy.hpp → session.hpp → contracts.hpp` 循环。

下面是接口草案，不是可直接编译的完整头文件；引用的 `StageId/CompletionToken/MemoryError` 等类型也需新增。

```cpp
struct MemoryConstrainedConfig {
    bool enabled = false;
    uint64_t limit_bytes = 0;
    unsigned buffer_percent = 15;
    uint64_t min_free_bytes = 1ull << 30;
    unsigned max_refill_slots = 3;
    bool allow_quality_preserving_tiling = true;
};
struct ByteEnvelope {
    uint64_t lower = 0;
    std::optional<uint64_t> upper; // 缺失 != 0
    std::string provenance;       // tensor_table / validated_envelope / heuristic
    std::string qualification_id;
};
struct StageMemoryEstimate {
    StageId stage;
    ByteEnvelope persistent, activations, transition;
    ByteEnvelope conversion, allocator_cache, output;
};
struct EffectiveMemoryPolicy {
    uint64_t user_limit_bytes = 0;
    uint64_t effective_budget_bytes = 0;
    uint64_t process_reserve_bytes = 0;
    uint64_t system_reserve_bytes = 0;
    std::string scope; // inference_process_tree_v1
    std::string enforcement; // controlled_allocations+validated_graphs
    std::string digest;
};
```

逐分量保留置信度，不能用一个 `Exact` 标签覆盖动态 graph 的不确定性。所有 upper 加法/乘法使用 checked uint64；`upper=nullopt` 的候选不具备 constrained admission 资格。

### 12.2 双层 reservation、唯一 backing 与异步释放

```cpp
class AllocationReservation { // move-only；未提交析构只归还 reservation
public:
    AllocationReservation(AllocationReservation&&) noexcept;
    AllocationReservation& operator=(AllocationReservation&&) noexcept;
    AllocationReservation(const AllocationReservation&) = delete;
    AllocationReservation& operator=(const AllocationReservation&) = delete;
    ~AllocationReservation() noexcept;
    StorageLease commit(StorageId unique_backing, uint64_t actual_capacity);
};
class MemoryLedger {
public:
    ReserveResult try_reserve(MemoryClass, uint64_t upper, AllocationTag);
    void retire(StorageLease, CompletionToken last_use);
    MemorySnapshot snapshot() const;
};
class StageEnvelope { // 是父容量，不是又一份物理分配
public:
    ReserveResult try_reserve_child(MemoryClass, uint64_t upper);
};
```

- `StorageId` 表示实际 allocator backing（含 generation），不是 tensor view。别名、reshape、CPU/Metal 同一 shared buffer 计一次。
- `commit` 不可无声扩容；实际容量必须 <= reservation；平台分配尺寸无法预知时事先预留 alignment envelope。
- `retire` 先转 `pending_release`，completion 后删除最后所有者。若 buffer 进入 allocator cache，账本从 active 迁到 cache，不能当作已返还 OS。
- 只保留 `observe_actual` 诊断不能称为分配前约束。MLX 无公开逐次分配拦截时，通过 Stage/Block graph envelope 包住其内部上界，见第 30 节。
- 析构不在任意 callback 上阻塞整个 GPU；运行上下文持有 deferred-release 队列，显式 drain 完成后再销毁 ledger。错误也不能在析构中 throw。

### 12.3 ModelSession 的最小兼容接入

保留旧 `generate(Request, Event, cancelled)`，新增可选受限入口；关闭 add-on 时不经过新的热循环。

```cpp
virtual MemoryCapabilities memory_capabilities(const Request&) const;
virtual ProbeResult probe_memory_layout(const Request&, const MetadataContext&);
virtual RunResult generate_constrained(
    const Request&, const PreparedMemoryPlan&, MemoryRunContext&,
    const Event&, std::atomic<bool>&);
```

默认 capability 为空、受限入口明确返回 unsupported，而不是默默转回旧 generate。类型包含 `supports_constrained_resident`、`component_staged`、`block_streaming`、支持的槽数集合、safe-replan 边界、graph envelope 版本；tiling 需要绑定具体 decoder/shape/质量证据，不是一个万能 bit。

模块 descriptor 用于 UI 的静态声明；session/probe 用真实 checkpoint、backend、operation、LoRA 和 runtime 版本复核。内部 C++ 虚表改变必须全量重编；现有对外 opaque C handles/JSON 函数签名保持兼容。

## 13. 配置解析、计划解释与错误契约

### 13.1 解析位置

- `native/platform/apple/request.mm`：schema-v1 顶层、schema-v2 execution 内分别 whitelist/解析，保留字段来源；两种非零预算冲突时拒绝，不静默覆盖。
- `native/platform/apple/profile.mm`：新对象按“默认 → profile 对象 → 请求对象”逐字段合并；仅为新对象规定此优先级，既有字段保持原 profile 语义。请求 `enabled=false` 必须胜出，记录来源以便解释冲突。
- `services/turbociderd/service.mm`：服务计划和执行请求都要透传，不在服务层自行推导 B。
- `bindings/swift/TurboCiderNative.swift`：增加可选 `MemoryConstrainedOptions`，默认 nil；旧客户端编码结果必须与旧 JSON 相同。
- `apps/cli/main.mm`：`plan` 输出候选列表、拒绝原因和最终 policy；不要只输出单一 estimate。

### 13.2 错误码

建议稳定错误码：

| 错误码 | 含义 | 必须包含的字段 |
|---|---|---|
| `memory_policy_invalid` | Y/X/单位非法 | 原始值、合法范围 |
| `memory_policy_unsupported` | session 没有所需能力 | model、capabilities |
| `memory_budget_too_small` | B 低于最小 floor | required、budget、largest contributor |
| `memory_admission_conflict` | 与其他已准入作业冲突 | holder job、requested bytes |
| `memory_pressure_abort` | 运行中无法安全降级 | watermark、last action |
| `memory_observation_unreliable` | 实际分配不可观测 | backend、missing counter |

静态配置/能力/容量错误尽量在大权重 load 前返回；I/O、运行压力、设备失败属于运行时错误。后者附最后完成的 stage/block、是否可重试与部分输出状态，不能承诺所有错误都发生在 load 前。错误分类同时加入 `memory_policy_conflict`、`memory_estimate_unknown`、`memory_estimate_invalid`、`weight_io_error`、`memory_lifetime_violation`。

### 13.3 计划结果的合同与示例

在现有 `plan REQUEST.json` 结果里加入解释对象，不假设已有 `--explain` flag。下例为无真实权重的 synthetic planner fixture，不是 H3 测量值；其 10 个等大 block、6 pinned+2 slots 的关系必须满足账本恒等式。

```json
{
  "memory_policy": {
    "enabled": true, "limit_bytes": 17179869184,
    "buffer_percent": 15, "effective_budget_bytes": 14602888806,
    "admission_state": "metadata_verified", "scope": "inference_process_tree_v1"
  },
  "memory_plan": {
    "fixture": "synthetic_10x1GiB", "process_reserve_bytes": 1073741824,
    "stage_floor_bytes": 4294967296, "pinned_bytes": 6442450944,
    "slot_capacity_bytes": 2147483648, "peak_upper_bytes": 13958643712,
    "headroom_bytes": 644245094, "active_blocks": 10,
    "pinned_blocks": 6, "streamed_blocks": 4, "refill_slots": 2,
    "candidate": "double_slot_streamed", "plan_revision": 1
  }
}
```

上例总量=1+4+6+2=13 GiB，低于 B≈13.6 GiB；全部驻留需 15 GiB，拒绝。运行前还需系统准入，`metadata_verified` 不等于 `admitted`。不带路径时输出 `admission_state=estimate_only`、实际 block table unknown，不能伪造上例。

## 14. 详细时序和并发规则

### 14.1 阶段时序

```text
admit(B)
  -> reserve text + conditioning
  -> load text / encode / event
  -> release text weights, retain conditioning
  -> reserve denoise floor + pager slots
  -> load pinned prefix; pread first streamed block
  -> denoise loop (compute[i] || pread[i+1])
  -> drain pager; release transformer
  -> reserve VAE/output; decode tiles; release latent/workspace
  -> export; release all leases; emit final trace
```

阶段间默认不重叠。只有 planner 明确证明 `current_peak + next_load_peak <= B` 时才允许下一阶段 load 与当前阶段 compute 重叠；compile/warmup 永远使用独立低优先级 lease。

### 14.2 Pager 状态机与提交序

每槽：`FREE → RESERVED → IO_INFLIGHT → READY → GPU_INFLIGHT → RETIRING → FREE`；CPU unpack/GPU convert 可以各自添加子状态。标识为 `(job_id, session_generation, slot_epoch, block_id)`。

```text
预先：reserve 所有必需槽/转换临时空间，读取第一个 group
for each step:
    for each block/group in 原模型层序:
        current = await_ready(required_group)
        若有 FREE 槽与已准入 reservation：启动下一 group 的 pread
        在 GPU/MLX owner 线程绑定 current（权重作为显式输入）
        构建/提交当前 group；登记所有读取者的 CompletionToken
        current -> RETIRING
        处理已完成 tokens；仅最后读者完成后才允许槽覆写
        在认证的 step/stage 边界响应压力或取消
    按原实现执行 sampler 更新（恰好一次）
最后：drain I/O、GPU、转换；清理引用；回收/迁移 cache 账本
```

没有可用槽时走有界等待/串行，不分配“临时第三槽”绕过预算。预取发生在当前计算期间，但下一层 compute 不能在当前层输出尚未就绪时执行。一个槽有多个 GPU queue/CPU conversion 读者时必须等待全部 token。

I/O worker 只读文件到已获 lease 的 backing，不在任意后台线程操作全局 MLX graph/cache。首版 MLX 可在 owner 线程执行 `eval + synchronize` 作为保守完成边界；后续事件优化必须针对安装的 MLX 版本证明真实设备完成，不能把 `async_eval` 返回当作完成。

### 14.3 取消和异常

先停止新提交，再标记取消并等待在途 `pread` 返回（短分段读取检查取消），等待已提交 GPU/转换结束，销毁所有 tensor/compiled input 引用，再迁移/清理 cache，最终归还子 reservation。取消不能即时抢占所有 Metal/系统 I/O 调用。

**忽略旧 generation 回调不够**：I/O 仍可能直接写旧 slot 的内存。在途任务必须持有 backing/context 的强所有权；join/完成前绝不能把该 backing 分配给新请求。旧回调只丢弃发布动作，实际内存要等写入结束才能释放。

逻辑 reservation 不足可在提交前退让；GPU device error/MLX eval 异常后不假设状态可继续。隔离污染 session，清理或结束 disposable worker，再由用户重试；不自动重做已执行的 Euler/随机采样。I/O 短读/EINTR 可在提交前受控重试，EOF/身份改变直接报错。

`@autoreleasepool` 要包住 loader 迭代与媒体 chunk，避免 NSData/Objective-C 临时对象延迟到整个请求末尾才释放。

## 15. 估算实现细节和校准流程

### 15.1 Weight index：以真实 tensor 布局为依据

拟增可缓存的 `residency.json`，不是手填某个模型的平均 block bytes。用 metadata-only 扫描生成：

```text
schema_version, model_id, source_identity, runtime_layout_version
source_files[]: relative_path, byte_size, content_identity
storage_records[]: id, source_file, offset, length, dtype,
                   shape, alignment, runtime_layout, pack_upper_bytes
units[]: id, stage_mask, family, storage_ids[], last_use_rule
quantization: format, bits, group_size, scale_dtype, zero_point/layout
```

同一 storage 被多个 unit 引用需去重；每个 unit 分别记录磁盘 payload 与驻留容量，不假设二者相等。H3 50 blocks、LTX 48 blocks、Flux dual/single 和 Z-Image refiner 不能共用一个 `block_count=48` 模板。

safetensors header/JSON 长度设上限；所有 offset+length、shape乘积、alignment 使用 checked arithmetic，验证范围在文件内、无未声明重叠。GGUF 读取已有 `native/core/gguf.hpp` 的类型/格式验证，不用文件名猜测格式。转换后的 packed 工件需绑定源 hash、converter 版本与布局；不改写用户原 checkpoint。

mismatch 时 invalid/rebuild metadata 或返回错误；不降成未经验证的 heuristic 后继续执行。第一版优先使用原文件 offset/index，非必须不复制整模型为新 shards。需要 packing cache 时先给出磁盘空间估算，以有界内存离线准备、原子提交 sidecar；准备失败不留下貌似 ready 的半成品。

### 15.2 shape、算法和阶段估算

估算 key 包括模型/adapter/quant-layout identity、backend/OS/MLX 版本、实际 text tokens、宽高帧数、参考图数量、音频长度、attention/compile/tile 方案。视频像素体积只能作为粗估代理；显式 NxN attention 与 fused attention 的 memory envelope 完全不同。

LTX 使用已有 workload geometry：Stage1 rows=`latent_frames*(W/64)*(H/64)`，Stage2 rows=`latent_frames*(W/32)*(H/32)`；同时考虑 audio rows，即使最终 `audio=false` 也不能假设联合 denoiser 不计算 audio。Flux/Z-Image 根据现有 tokenizer padding/图像 patch 数计算真实 tokens；长 prompt 和多图 reference 是独立压力用例。

`steps` 主要改变 I/O/计算次数，只有跨步缓存/诊断保留才会抬高峰值。scheduler 更新、text hidden 输出、RGB/PCM 转换、LoRA merge 的临时副本、quantized->BF16 dense GEMM 必须单列。

### 15.3 校准不能替代上界

不自动在新硬件上运行无预算的大 warmup。离线 calibration 先对已可准入 fixture 测量多组 shape，保存样本数、实测最大值、误差 guard 与适用范围。P95 仅用于性能/经验预测，不称硬内存保证；一个样本不能计算可信 P95。

运行中发现 actual 超过已认证 envelope：本次失败、记录 estimate_invalid，并将该 identity/shape 的 constrained 资格失效；不自动增加预算继续，也不只修正下一次而把当前判通过。超出校准范围可采用可证明上界或拒绝，不外推一个像素比例就认定可行。

## 16. 各模型的代码落地拆分

### 16.1 H3 C runtime

优先修改：`native/models/h3_runtime/h3_dit.c`、`h3_gpu.m`、`h3_weights.c`。

- 将 `ssd_memory_budget_bytes` 改为由 `EffectiveMemoryPolicy` 计算出的 denoiser sub-budget，同时保留旧字段兼容。
- 在 `h3_dit` context 保存 C-compatible opaque budget callbacks、generation、slot completion 和 last-use 信息；C 代码不直接依赖 C++ RAII 类型，异常不得穿越 C ABI。
- 首版以 adapter 接入预算/完成回调，保留现有成熟双槽 loader 与 Metal 内核；后续共用 WeightPager 协议，不为抽象统一重写全部 I/O 热路径。
- final-pass eviction 必须以 GPU completion event 为条件；不能仅按 block index 释放。

### 16.2 H3 MLX / VDN

优先修改：`native/platform/apple/h3_mlx_checkpoint.mm`、`native/models/h3_mlx/checkpoint.*`、`pipeline.cpp`。

- 当前 loader 持有整个 tensor map，不能把 lazy/mmap 当成 bounded residency；新路径用 header/index + unit-range 读取，不必强制先拆成 50 个实体文件。
- `Checkpoint` 从 `unordered_map<string, Tensor>` 改为 resident block map + non-block map；`linear()` 接受当前 block lease。
- VDN branch 的 800 个 tensor 按当前代码 block 前缀分组，与同层 base/attention 权重同 lease；重视 solve 的状态/workspace，不能把加载 branch 文件当作异步完成。
- `mx::set_memory_limit` 是 guideline，不是硬 cap：仅作为 scoped 辅助配置。为每个 group 的 graph 建立认证的分配 envelope；公有 allocator API 无逐分配 veto 时不得声称全 MLX 分配已硬拦截。快照 active/cache/peak 用于核对，不直接与 Metal/进程 footprint 相加。

### 16.3 LTX

优先修改：`native/models/ltx_runtime/ltx_blocks.c`、`ltx_weights.*`、`native/platform/apple/ltx_session.mm`。

- 现有 `tc_block_residency_plan_build` 继续作为 block planner 核心；输入 budget 改为扣除 text/latent/output floor 后的 sub-budget。
- `LTX_MAX_REFILL_SLOTS` 变为配置上限，运行时 slot 数由 planner 给出。
- 为 stage1→upsample→stage2 建立存活/过渡计划，复用同一权重池；需包含 upsample 的 latent/部分 VAE 权重以及旧新 latent 同时存活的峰值。
- ANE preload 在 GPU-only add-on 一律不选。`parallel_av` 的 GPU 双队列是内存/性能候选，不一律关掉；先认证 serial，再对有余量的双队列验证共享带宽与 arena 峰值。LTX C++/MLX 的 BlockCache 另设 adapter，不能复用 C 路线的槽位能力声明。

### 16.4 Flux

优先修改：`native/models/flux2/pipeline.cpp`、`flux_transformer.cpp`、`flux.hpp`。

- 先把当前 `ResidencyPolicy` 的 release 点包装成 `MemoryLease`，确保 text/transformer/VAE 分阶段可观测。
- 保留旧全量 `Weights` 路径，新增 unit-backed view/reader；4B 为 5 dual+20 single、9B 为 8 dual+24 single（以配置校验为准）。两类 group 大小不同，scratch 与 slot layout 分别计量；第一版不切 projection 内部。
- denoise loop 在 group 边界插入 `WeightPager`，并为 dual/single layer 顺序建立 golden trace。
- `mx::set_cache_limit()` 由 policy 的 K budget 驱动；不能固定 512 MiB 而忽略请求预算。

### 16.5 Z-Image

优先修改：`native/models/z_image/gguf.cpp`、`z_image.cpp`、`native/backends/mlx.cpp`。

- Z-Image 已会清理 text encoder，但 `load()` 仍把 transformer/VAE 一起载入；新 staged 路径拆分这两者，还要在新 prompt text encode 前逐出前一请求遗留的 transformer/VAE。
- GGUF layer streaming 需要读取 tensor metadata/offset，建立可 seek 的 layer table；LoRA inference-time 分支必须证明不会隐式复制整份 GGUF。
- 30 个 `layers` 与两组各 2 个 noise/context refiner 分开成 unit，global embedding/norm/output 放 trunk；caption/联合 token 激活不可误逐出。QKV fuse/remap、convrot/NVFP4 packing 与 scales/bias 转换计临时容量；后两种未认证时 constrained 明确拒绝。

## 17. 测试文件与测试用例建议

新增或扩展：

- `tests/native/test_memory_policy.py`：配置解析、B 公式、错误码、JSON round-trip。
- `tests/native/memory_accounting_test.cpp`：checked arithmetic、lease 生命周期、重复 release、event 延迟释放；由 `tests/native/test_memory_accounting.py` 编译运行。
- `tests/native/test_residency_planner.cpp`：H3/LTX/Flux fixture 的 pinned/slot golden。
- `tests/native/test_memory_pressure.cpp`：注入 allocation failure、yellow/orange/red 状态转换和 deterministic fallback。
- `tests/native/test_memory_capability_contract.py`：模块 descriptor 与 session capability 一致性。
- 扩展 `tests/native/test_h3_streaming_policy.py`：constrained policy 生成的 sub-budget 和全请求 budget 对照。

每个集成测试都应输出 `before/after` 两组：`active_bytes, peak_bytes, cache_bytes, process_footprint, swap_delta, gpu_wait, io_bytes`。测试失败时保留 schedule trace，而不是只返回 exit code。

## 18. 验收矩阵：容量探索不等于成功承诺

以下 Y 是待测试的预算点，不是声称该机器/模型一定能够完成。先 metadata-probe，记录每阶段 floor 和主要限制；实现支持矩阵要进一步指定至少一个真实 `must_succeed` 点。

| 模型/执行器 | 合法起始 shape | Y 探索点 | 第一阶段预期 |
|---|---|---|---|
| `minimax-h3-turbo` C/Metal BF16 | 256×256×22，4 steps | 16/24/32 GiB | 已有两槽适配；全流程 text/VAE floor 仍须准入 |
| `minimax-h3-fasth3-mlx-int6` | 512×512×22，4 steps | 16/24/32 GiB | 未完成 pager 前只接受认证 staged-resident 或明确拒绝 |
| `minimax-h3-vdn` | 960×544×22，6 evaluations | 16/24/32 GiB | 必测旧 streamed 字段不误报受限能力；branch workspace 独立核算 |
| LTX C/Metal | 256×256×9；704×448×97，11 steps | 16/24/32 GiB | T2V video-only 首批；8+3 输出和 upsample/decoder 过渡 |
| LTX C++/MLX | 同合法 T2V geometry | 16/24/32 GiB | BlockCache/ref绑定/周期 eval 独立认证，不以 C 结果替代 |
| Flux 4B/9B | 256²、512²、1024²，4 steps | 8/12/16/24 GiB | 先 staged，后分别认证 dual/single streaming |
| Z-Image native Q8_0/Q4_0/Q4_1 | 256²、512²、1024²，9 steps | 8/12/16/24 GiB | staged/streamed 分阶段开放，mixed K-quant 必须拒绝 |

每个模型另测长文本、变更 prompt、相同 prompt cache hit、模型切换、较小 Y 的后继请求；LoRA/I2V/audio 在模型能力允许的组合中独立测试，不自动扩大首发范围。

对每条支持路线测 `minimum-1 byte`（must_reject）、`minimum`（检查估计与运行相符）、中档预算、完全驻留预算。至少测试 X=10/15/20。64 GiB 上人工预算只能验证 scheduler/账本，不能证明真实 8/16 GiB 机器的文件缓存、带宽、swap 行为。

Cold/fresh-process、warm-file-cache、retained-session 三种状态分别记录；ABBA 至少 3 轮，P95 在足够样本后再报告（建议 >=20），少量样本只报 median/min/max。任何不能解释的越线不可被平均值掩盖；安全拒绝的点不得计入 success throughput。

## 19. 实施顺序与提交拆分建议

1. `memory_policy` + parser + plan explain + 单测（不改变执行）。
2. `MemoryLedger/Lease` + telemetry，先只观测。
3. H3 C runtime 接入全请求 admission 和 pager lease。
4. LTX 两阶段 sub-budget、动态 slots、取消清理。
5. Flux/Z-Image component-staged 能力声明与预算拒绝。
6. H3 MLX/VDN 真正按 block 读取。
7. Flux layer-group streaming、Z-Image GGUF streaming。
8. pressure autotune、calibration 和 release gates。

每一步单独提交、单独 benchmark；默认模式的 regression gate 在每一步执行。任何阶段若需要改变默认 residency、默认 cache limit 或默认 slot 数，都必须另开变更并更新兼容性说明。

## 20. 完成定义（Definition of Done）

只有同时满足以下条件，才能把 `memory_constrained` 标为 stable：

- 配置、plan、运行 trace、错误码和 Swift/CLI 接口版本化；
- 至少 H3 C runtime 与 LTX 具备实测全请求预算约束；
- 每条宣称支持的模型/backend 有 `must_succeed` workload 实际通过峰值/数值/性能门；不可行/未支持组合按 `must_reject` 测试，两类不得混计；
- add-on 关闭的输出、选择和默认性能通过 ABBA 回归；
- 取消、异常、服务重启后无 lease/slot/cache 泄漏；
- 每个支持模型都有 manifest/estimate 版本、硬件校准记录和可复现 JSON 证据；
- 文案明确区分 working-set target、可观测进程 footprint 与操作系统最终 swap 行为。

## 21. 逐文件改造草案（接近 patch 级别）

以下不是要求一次性提交的完整代码，而是实现者应遵循的最小改造边界。

### 21.1 `native/core/contracts.hpp`

不要删除或重命名已有 `memory_budget_bytes`；它仍服务于旧 H3/LTX CLI。新增字段建议放在 `Request` 尾部，降低源码 aggregate 初始化的影响；这不保证内部 C++ ABI：

```cpp
MemoryConstrainedConfig memory_constrained;
```

解析时只得到用户配置和来源，profile 合并后纯函数验证；设备/metadata 准入在执行入口完成。旧 budget 路径不自动转换为新 B，关闭 add-on 时旧字段原义不变。避免 `Request` 的构造/解析隐式初始化大 backend 或加载权重。

### 21.2 `native/platform/apple/request.mm`

增加：

```objc
keys(memory, @[ @"enabled", @"limit_bytes", @"buffer_percent",
                @"min_free_bytes", @"max_refill_slots",
                @"allow_quality_preserving_tiling" ]);
```

推荐同时接受 schema-v1 顶层 `memory_constrained` 和 schema-v2 `execution.memory_constrained`，规范化到同一 C++ 字段；序列化时只输出用户实际提供的 schema 位置。禁止把 `NSNumber` 的 boolean 当作 number，沿用现有 `numeric/boolean/byte_count` 校验函数。

### 21.3 `native/runtime/plan.cpp`

将目前按模型写死的 `memory_estimate_bytes` 保留为总值，同时新增：

```cpp
MemoryEstimate estimate_request_memory(const Request&, const DeviceInfo&);
std::vector<ResidencyCandidate> enumerate_residency_candidates(...);
```

模型分支只负责提供 shape-dependent facts（block count、block bytes、VAE geometry、是否支持 tiling）；公共 planner 负责组合、排序和错误解释。不要在 `plan.cpp` 直接调用 MLX/Metal 分配，dry-run 必须无副作用。

### 21.4 `native/api/c_api.mm`

`tc_plan_json` 为 enabled 请求增加说明对象，但当前函数无模型路径，返回 estimate_only。拟增 `tc_engine_estimate_memory(engine, request_json, ...)`（metadata-only，不 materialize）供 CLI `estimate MODEL REQUEST.json`/服务使用；这是新接口，必须新增 C header、导出和 SDK 测试。

`tc_engine_generate` 现已持有 `e->session`，不是在其中重新创建 session。保持 engine mutex→execution_mutex→DeviceLease 顺序；解析有效策略后：检查/逐出现有 residency → 真实 metadata probe → admission → 选择 constrained 入口 → drain → 恢复全局 limits → 返回结果。所有新 model 构造函数必须仅持有元数据；如果构造本身会大分配，需要先拆出轻量 probe。

`tc_engine_load`/prepare/warmup 也不能绕开约束。给新配置提供带 request 的 prepare/estimate 接口；无 request 的旧 load 默认行为保持，但 App 启用受限模式时不调用无预算整模 preload。

### 21.5 `native/runtime/execution.hpp/.cpp`

现有 `DeviceLease` 是跨进程 GPU 互斥，不应承担 bytes 计数。新增进程内 `MemoryAdmission`，并由服务持有一个全局实例。两者顺序固定为：先获取 GPU `DeviceLease`，再申请 `MemoryAdmission`；失败时按相反顺序释放，避免另一个作业观察到半准入状态。

不要持有 GPU DeviceLease 再无限等待内存，当前跨进程 GPU lease 本就是 try-lock。服务在未持有 GPU 资源时排队；获得执行锁后做一次 try_admit，失败释放所有执行锁并返回可重排队原因。压力暂停设置有界截止时间/安全点，首版可直接 fail-closed，不为此引入死锁式等待。

### 21.6 `native/backends/mlx.cpp/.hpp`

增加 backend 级 RAII：

```cpp
struct MlxMemorySnapshot {
    size_t active, cache, peak, limit;
};
MlxMemorySnapshot memory_snapshot();
ScopedMlxLimits scoped_memory_limits(uint64_t graph_limit,
                                     uint64_t cache_limit);
```

`scoped_memory_limits` 在已有 execution_mutex 下保存旧值，通过 setter 返回值保存 cache limit；并在所有 MLX 队列 drain、持有对象释放后恢复。它仅是辅助 guideline，不能替代 graph envelope。错误/取消/prepare 路径都要恢复；若 session 被污染无法证明安全则隔离 worker。默认请求不调用新 limits，不继承受限请求的设置。

### 21.7 `native/models/h3_runtime/*`

当前 C runtime 已有最接近目标的实现。建议先不改 kernel，只改上下文和 admission：

1. `h3_dit` 初始化前接收 `effective_denoise_budget`；当前 adapter 只接受已验证双槽，不在未改 loader 时传任意槽数；
2. 将 block/slot allocation 通过 callback 交给 `MemoryLedger`；
3. `pread` 完成后只在 `READY` 状态发布 slot；
4. 最后一个使用者的 completion 发布 retirement token，由 ledger owner 完成计账，不在 C/Metal callback 中抛 C++ 异常；
5. 在 `h3_dit` 结果中回填 `peak/bytes_loaded/wait_seconds`。

### 21.8 `native/models/ltx_runtime/ltx_blocks.c`

当前代码在加载 block 0 后才知道 `block_resident_bytes`，这会造成 planner 先分配再发现预算不足。应增加 manifest-only 查询函数：

```c
int ltx_native_probe_block_layout(const char *checkpoint,
                                  ltx_block_layout *out, ...);
```

生成前先 probe header/offset/单 block bytes，再计算 pinned/slots；正式 load 时验证 probe 与实际 block bytes 一致。`LTX_MAX_REFILL_SLOTS` 只作为编译期最大数组长度，实际 `ctx->refill_slots` 必须来自 policy。

### 21.9 `native/models/flux2/flux_transformer.cpp`

当前 4B single-block compiled graph **已将权重作为显式输入**，不要错误地将它描述为 checkpoint 常量捕获。真正需修改的是其连续 20 层通常不逐层 eval 的依赖链：受限路径在 group 边界物化/确认完成并断开旧引用，否则 C++ map erase 仍不能释放设备 backing。dual layers 与 eager/9B 路径边界不同，分别适配。默认模式保留当前减少 host waits 的优化。

LTX `CompiledBlockForward` 同样已有显式权重输入，可复用接口理念但不共享模型数学；compiled executable cache key 绑定结构/shape/layout，checkpoint/LoRA 实例不做闭包常量。

### 21.10 `native/models/z_image/z_image.cpp` 和 `gguf.cpp`

现有 `load()` 同时加载 transformer 与 VAE。component-staged 改造应拆成 `load_text() / load_transformer() / load_vae()`，并在 `run()` 的明确边界调用 `clear()`。GGUF streaming 的 reader 需独立解析/验证 GGUF offset/type，复用公共 storage/ledger/slot 协议，不直接拿 safetensors 的 offset 布局读取；从已有 `native/core/gguf.hpp` 验证能力扩展。

## 22. 线程、事件和资源所有权规则

### 22.1 线程模型

| 线程 | 允许操作 | 禁止操作 |
|---|---|---|
| planner/service | 估算、准入、取消等待 | 直接创建 MLX tensor、触碰 Metal queue |
| model/GPU | encode、submit、event 注册 | 阻塞式大文件扫描、修改全局 policy |
| pager I/O | `pread`、slot 状态更新 | 创建 MLX array、调用 `mx::eval` |
| telemetry | 采样、写 trace | 触发回收或改变调度 |

所有跨线程 callback 带 `job_id + generation`。session 重用时 generation 递增；旧 job 的 I/O 或 Metal completion 不得释放新 job 的 slot。

### 22.2 所有权图

```text
JobLease owns StageLease
StageLease owns ComponentLease / PagerLease
PagerLease owns SlotLease
SlotLease owns MetalEvent pin
```

父上下文只能在所有子任务/lease 退役后销毁；容量 envelope 不再次计为物理 bytes。调试版加引用/slot epoch assert，release build 检测违反时拒绝继续执行；异常从 owner 线程报告，析构和 GPU completion 不抛异常。

### 22.3 文件映射与 page cache

受限权重优先 `pread` 到有界 slot；mmap 并非禁止，也不等于零成本。VMS 映射长度、被触碰 clean pages、private/dirty backing 分别记录；若计划会扫过整个映射，不能只用一个 P95 低估它。`F_NOCACHE`/`madvise` 等仅作平台支持的性能提示，既不能当硬上限，也不能提前 invalidate GPU 正在读取的 backing。不得无界 WILLNEED 整模型；`pread` 本身仍可能经过 OS page cache。

## 23. 自适应策略参数与默认值

首版应使用固定、可解释参数，避免“自动调参”掩盖问题：

| 参数 | 首版默认 | 说明 |
|---|---:|---|
| `buffer_percent` | 15% | 用户 Y 到 TurboCider B 的保护带 |
| `min_free_bytes` | 1 GiB | 独立系统准入余量；不与进程 R_proc 重复扣除 |
| yellow watermark | 0.80B | 停止新增可选缓存/warmup，保留已准入必要预取 |
| orange watermark | 0.90B | 有压力时在认证边界缩小窗口，不假定单槽可用 |
| red watermark | 0.97B | drain 后重规划 |
| I/O wait trigger | 连续 2 个 block | 仅触发评估；扩窗还需 capacity/带宽收益和滞回证明 |
| prefetch max | H3=2、LTX=3 | constrained 可进一步降低 |
| pre-submit I/O retry | 有界 1 次策略 | 仅可重试的读失败；GPU/eval 错误不自动重放采样 |

调参只允许改变 refill slots、pinned prefix 和 tiling；不允许自动改变 dtype、steps、sampler、稀疏模式或近似开关。

## 24. 可观测性字段与日志规范

每个 request 的 trace 至少包含：

```text
policy_digest, model_revision, shape_signature
budget_Y, budget_B, reserve_R
estimate_upper, actual_peak, actual_active, cache_peak
stage, block, slot, generation
io_bytes, io_seconds, wait_seconds, compute_seconds
watermark_before, watermark_after, recovery_action
swap_in_delta, swap_out_delta, process_footprint_peak
```

受限模式日志默认 stage summary + 聚合 block 指标；每个 slot 的详细事件仅在内部诊断 trace 打开时写入有界 JSONL 队列。关闭 add-on 时不启动此采样/日志线程。`RunResult` 是内部 C++ 类型，可以扩展；对外仍走现有 C ABI 返回 JSON 字符串，不将可变长 trace 塞进 C 结构体。

UI 区分“估算可行”“已准入”“受控分配未越线”“本次观测在目标内”；即使采到 actual_peak，也不显示“操作系统强制绝对上限”。缺某个计数显示 unknown，而不是 0。

## 25. 故障注入和恢复测试

实现阶段必须提供仅测试可用的注入开关（环境变量或 mock）：

- 第 N 次 slot allocation 返回 `ENOMEM`；
- 第 N 个 block 的 `pread` 短读/延迟；
- 在测试 backend 的 graph-envelope 超限点注入异常；另测真实 MLX 内存异常，不假设 set_memory_limit 会硬触发；
- GPU completion 延迟，验证 slot 不提前 free；
- generation 取消后旧 I/O completion 到达；
- VAE load 失败、导出失败、服务进程重启。

每个注入都要求：无死锁、无 double free、无残留 lease、下一请求可正常运行，并输出最后一个安全边界和恢复动作。

## 26. 性能目标的分层定义

不要用单一“比 resident 慢多少”评价 constrained：

1. **可行性层**：在 B/Y 内完成、无新增 swap、质量不回退。
2. **调度层**：分别报告 GPU-starvation 时间与时间线交集 `T(IO∩GPU)/T(IO)`，该值才描述 I/O 与计算的时间重叠；收益还需和关闭预取的同预算对照比较。`1-wait/(wait+compute)` 只是近似 compute 占比，不能称为 I/O 隐藏率。首版不设不顾 SSD/shape 的统一 50% 硬门槛，避免 I/O-bound 模型被误判。
3. **端到端层**：相对 component-staged baseline 的 wall overhead；冷启动和 warm 分开报告。
4. **默认兼容层**：add-on disabled 的 wall、peak、输出 hash 与基线一致。

任何性能目标必须绑定硬件、OS、模型 hash、shape、cache 状态和 seed；不允许跨机器比较绝对秒数。

## 27. 与现有文档/实现的同步要求

实现后必须同步更新：

- `docs/design/quantized-streaming-vpipe-comparison.md`：区分 denoiser working-set 与全请求 constrained budget；
- `docs/design/runtime-modules-and-performance.md`：补充 MemoryCoordinator/WeightPager 职责；
- `docs/design/video-model-acceptance.md`：增加 constrained matrix 和 fail-closed 条件；
- `docs/design/README.md`：加入本文入口和状态（Proposal/Implemented）；
- `tests/native/test_contract.py`：schema/旧请求兼容性；
- `README.zh-CN.md`：仅在稳定 gate 通过后宣传用户可见能力。

文档状态必须与代码真实能力一致：尤其不能因为 `Request` 已接受 `memory_constrained` 字段，就宣称所有模型都支持 block streaming。

## 28. 账本计算示例与去重规则

以 `Y=16 GiB, X=15%, min_free=1 GiB` 为例：

```text
B = floor(16 GiB * 0.85) = 13.6 GiB
```

以下是假想 unit 大小的算术例子，不是 H3/LTX 实测权重大小。假设某一时刻同时存活：

```text
process baseline (R_proc)       0.8 GiB
text conditioning (C)           0.5 GiB
non-block trunk (W_trunk)       1.2 GiB
6 pinned blocks                 6.0 GiB
2 refill slots                  2.0 GiB
denoise workspace (A)           2.2 GiB
latent/output (O)               0.4 GiB
MLX/Metal cache (K)             0.2 GiB
-----------------------------------------
planned upper                  13.3 GiB <= B
```

`min_free=1 GiB` 用于系统 admission/pressure 监测，不能再次从上述 13.3 GiB 中扣除；`Y-B=2.4 GiB` 也不是可自由分配的预算。若下一阶段 VAE 需 4 GiB，而 denoise 尚有 10.1 GiB 存活，则 planner 必须先释放 transformer/工作区，或拒绝并行加载，不能把 VAE 4 GiB 直接加到一个“阶段平均值”上。

### 28.1 唯一 backing 去重

同一物理 backing 可能有多个视图：MLX tensor、Metal buffer、C++ `Tensor` wrapper、LoRA slice。ledger 以 `StorageId` 去重：

```text
StorageId = (allocator_domain, pointer/handle, byte_capacity, session_generation)
```

逻辑 view 的 shape/stride 不增加 bytes；copy、dtype cast、dequantized dense matrix 产生新 StorageId，必须另计。`mx::get_active_memory()`、`MTLDevice.currentAllocatedSize`、process footprint 是不同观测域，不能相加；统一 ledger 只对自己负责的 backing 计 committed，其他观测用于差异分析。

### 28.2 三个数字必须同时输出

- `committed_bytes`：planner 已承诺且可能尚未 materialize 的容量；决定是否允许下一次分配。
- `active_bytes`：backend 当前认为有引用/活跃的容量；受 MLX lazy graph 影响，不能单独作为安全判据。
- `observed_footprint`：进程或系统采样值；可包含映射页、缓存和非 TurboCider 分配。

通过条件不是 `active==committed`，而是 `committed<=B`、已认证 graph envelope 未越界，并且 observed footprint/pressure 没有超过运行门禁。三者差异超过校准阈值时标记 `memory_observation_unreliable`。

## 29. LTX 阶段与 finalizer 的特殊预算

LTX 当前可能使用独立 Video VAE helper/exec finalizer。受限调度必须把它视为新的资源域，而不是把 child 的内存“从 parent 报表中删除”：

1. parent 在启动 helper 前完成 latent staging file/共享输入 reservation；
2. helper 获得独立 `MemoryLease`（或显式声明“不受 TurboCider 账本观测”，则 constrained 模式禁用该路线）；
3. parent 释放 transformer/workspace 后再启动 helper，避免两进程峰值叠加；
4. helper 退出并确认输出文件完整后，parent 才释放 staging reservation；
5. 失败/取消删除临时目录，不能留下 latent/PCM/RGB 大文件占用后续预算。

如果 helper 的 process footprint 无法纳入 admission，只能选择另一条已经认证的同进程路径，或返回 `memory_observation_unreliable`。不能因为 child 是独立 PID 就宣称 parent 未超预算。

这里还要区分 `posix_spawn` 和 `exec`：spawn 期间 parent 仍存在，峰值是同一时刻 parent+child；exec 替换 worker 进程镜像，旧堆不会继续作为完整 resident session 存活，但常驻服务和 staging/cache 仍需计入。服务自身绝不能被 exec 掉。现有 CLI/service 根据请求 residency 选择 disposable LTX worker，新增策略必须使用规范化的有效计划传递相同路由，不能只改 model session 而漏掉 service 层分支。

建议 parent 预留 child 的 envelope，传递 `job_id/policy_digest/stage_budget/plan_revision`；child 在分配前 ACK，监控覆盖 bootstrap 到退出。子 reservation 消耗父 reservation，不是再额外获得 B。exec 不能依赖原 C++ RAII 析构来通知服务释放；生命周期由父服务 wait/终态事件统一回收。worker 崩溃、ACK 超时、取消时同样执行回收协议，且须复用现有 GPU lock 的所有权/移交，避免 child 再获取父持有的锁而死锁。

临时 latent 文件本身占磁盘配额，不是整文件长度都算活跃 RAM；写入缓冲、文件缓存和读端新 tensor 仍需计入。文件放专属随机临时目录，失败只清理本 job 文件，不能按广泛通配符删除用户输出。

## 30. MLX 延迟执行与 envelope 的实现合同

MLX 的 array 构造、graph 编译、`eval` 和缓存回收可能在不同时间发生。首版要采用保守合同：

- planner 为每个 `eval` 单元记录输入 backing、预计输出 backing、最大临时 workspace 和 graph cache 增量；
- 只有在输入/输出/临时总和已被 reservation 覆盖时才调用 `mx::eval`；
- `mx::async_eval` 后不得立即释放权重，至少等显式 `mx::synchronize` 或版本化 completion API；
- `mx::clear_cache()` 只能释放 MLX cache，不代表 Metal/OS backing 立即归还，也不能在有 in-flight graph 时调用；
- 每个 session/shape/compiled graph 使用唯一 cache key。受限和默认模式的 key 可以不同，但不能让受限 graph 污染默认 session 的 cache limit；
- 如果无法从 MLX API 获得 graph 临时上界，受限模式使用经过实测的 whole-graph envelope；没有 envelope 就拒绝该 graph，而不是拿输入 tensor bytes 近似。

`mx::set_memory_limit`/`set_cache_limit` 作为 scoped 辅助配置，drain 后恢复。首版**不主动调用 `set_wired_limit`，不提升 wired 限额**：wired 是保持页驻留的能力，不是减少总内存的方法。大额 wiring 会压缩系统余量，不能拿它作为避免 swap 的捷径。所有这些 API 都不构成整个进程的硬 cap。

## 31. 压力采样、系统干扰和测试环境判定

采样器每 100–250 ms 读取一次 process footprint/pressure；阶段边界强制采样一次。采样线程只上报，不触发任意 free。下列情况将本轮标记 `CONTAMINATED`：

- 测试期间出现非 TurboCider 大型进程/视频导出任务；
- swap 单调计数上涨但无法关联到本 job，且系统 memory pressure 为 critical；
- 用户在 Activity Monitor/其他工具手动触发 purge 或改变内存状态；
- 外接显示器/媒体解码或系统更新造成明显额外峰值。

污染条件在测试前登记；污染轮次单独列数并重测，不根据性能差或 swap 上涨后临时选择性删除样本。所有已发起轮次仍进入执行总数，报告 PASS/FAIL/SKIP/CONTAMINATED 各自数量；无干扰性能汇总可以排除预先定义的污染，但本进程账本/footprint 已越线的事实不能因此消除。验收脚本记录开始/结束及周期 `vm_stat`、`vm.swapusage`、进程 footprint、各 backend counters 和 ledger snapshot。无法读计数时标 unknown/未资格化，不假设为零。

## 32. 安全降级的决策表

| 发现情况 | 可做动作 | 不可做动作 |
|---|---|---|
| pinned + slots 有余量但 I/O 等待高 | 在下一个认证边界增加一个已支持 slot | 临时创建未预留第三槽 |
| cache 超过 K budget | 在无 in-flight graph 时 trim/clear cache | 直接清理仍被 event 引用的 buffer |
| denoise 后 VAE 不满足 | 释放 transformer，选择已认证 tiling | 同时保留 DiT+VAE 并希望系统 swap |
| 单槽可行但执行器未认证 | fail-closed 或保持双槽并拒绝 | 只改 planner 数字让实现“看起来”单槽 |
| MLX actual 超 envelope | abort 当前 job，标记 estimate invalid | 提高 Y/B 后继续并把本次算成功 |
| 仅另一近似变体满足预算 | 报不可行并建议用户显式换变体后重新准入 | 内存模式自动打开 sparse/quantization |

## 33. pinned、slot 与 tiling 的成本模型

planner 不应单纯“尽量多 pinned”。对候选 c 估算：

```text
T(c) = T_load_initial(c)
     + sum_over_steps [T_compute_step(c) + T_unhidden_io_step(c)]
     + T_decode(c) + T_export(c) + T_replan(c)
```

失败风险在可行性过滤时处理，不把“不安全但可能很快”的计划用一个惩罚秒数混入排序；未知 envelope 直接拒绝。增加一个 pinned block 通常减少重复 I/O，但会改变冷启动 load 和持久容量；增加一个 refill slot 可能隐藏 I/O，也会占用原本可放 pinned 的容量。比较的是完整候选 `ΔT`，不能只看单 block wait。

单请求按 N=1 的完整成本排序，**不预设较少/较多 pinned 一定更快**；只有用户明确队列批次或实际 retained session 才计算跨请求收益。静态 tensor 可以在通过兼容校验后跨 shape 复用，但 workspace/compiled graph/condition cache 必须分别校验，不用一个粗略 session key 代替所有缓存身份。扩窗需要连续 starvation、可用预算和预期收益；首版仅在请求/step 边界重选，不在每层抖动。

tiling 的排序也要把边界复制、halo、额外临时 output 和多次 kernel launch 计入；只比较 tile 内 activation bytes 会系统性低估峰值。quality-preserving tiling 需要同一 decoder、相同累积顺序或已验证 tolerance；不能把任意 latent tile 作为通用降级。

## 34. 交付检查清单（实现 PR 模板）

每个实现 PR 至少回答：

- 是否触碰默认模式代码路径？若是，给出 ABBA 证据；
- 新增的每个 backing/slot/workspace 在哪个 ledger 类别计账？是否去重？
- reservation 的上界来自哪一个 manifest/fixture/校准证据？适用 shape 范围是什么？
- 哪些 event 保证最后使用完成？取消和旧 generation 如何处理？
- 发生 ENOMEM/pressure/短读时是 retry、replan 还是 reject？是否会重复 sampler step？
- 输出质量合同是 exact 还是 approximate？对应哪一个现有 gate？
- 是否包含 cold/warm、minimum-1/minimum、must-succeed/must-reject 和污染环境处理？
- 是否更新 model descriptor、CLI/Swift/service schema、trace JSON 和相关 docs？

没有这些信息的实现不应合并为“低内存支持”，最多标为实验性 instrumentation。

## 35. 分配闸门、系统准入和会话切换

### 35.1 精确的账本状态

对 ledger 自管分配：

```text
M_committed = process_baseline_envelope
            + sum(unique live backing capacities)
            + sum(pending-release capacities)
            + sum(cached capacities not yet returned/reused)
            + sum(unconsumed reservations)
```

live、pending-release、cached 是互斥状态。一个 reservation commit 后，将预留空间迁为 live，并归还未用部分，不能两边都保留。复用 cached backing 时迁为 live，不再新加一份。MLX graph envelope 是父容量，其内子分配只消耗该容量；若内部不可逐个观察，保留整个 envelope 直到 group 完成，勿再叠加同范围的 active_memory 快照。

CPU `vector` 计 capacity 不只计 size；Metal heap 按 heap backing 计，suballocated buffer 不重复计；ObjC autorelease 临时对象、高水位字符串/日志、tokenizer 输出至少纳入基线/过渡 envelope。`get_active_memory` 看不到 native C 的 malloc，C/Metal runtime 的预算回调不可遗漏 host scratch。

分配事务顺序：`checked size → try_reserve → allocate → verify capacity → commit → publish`。任何失败在 publish 前 rollback。不得先 allocate 再判断 budget；callback 返回 failure 时 C loader 沿已有 cleanup 标签退出，不留下半初始化权重。

### 35.2 系统准入是另一个检查，不是第二套 GPU RAM

Y 是作用域内目标，不等于“本机总物理内存”。Apple Silicon 上 CPU/GPU 共用物理容量，不能再给 CPU offload 一个等大的独立池。

平台 sampler 提供当前 pressure、压缩/换页趋势、已有 scope footprint，以及保守的新增可用容量估计 `H_system`。运行前检查 `planned_increment + min_free_bytes <= H_system`；planned_increment 只对相对当前 scope 的新增存活集合计算，不能把已有 live backing 再当新增占一遍。H_system 不简单等于 free pages，也不能把所有 inactive/file cache 页都视为立即可回收。估计来源/时间戳必须输出；当前平台无法认证该估计时，采用保守资格范围和压力检查，并明确保证边界。

另检查本进程 GPU resources/heaps envelope 相对 `recommendedMaxWorkingSetSize`；这只是性能工作集参考，不是可分配 RAM 的许可证。不要以 `B <= recommended` 代替全进程预算，也不要用提高 wired limit 绕过。

### 35.3 作用域与旧会话接管

首版 scope 为“承载此次推理的进程以及由它/服务创建的推理 worker/helper”；常驻服务基线明确分摊，PID 列表写入结果。embedded App 模式必须包括承载推理的整个 App 进程基线，不能只报告模型数组；如果 UI/preview 内存无法建立可靠 envelope，应使用已认证 disposable worker 或报告该运行方式未支持。独立 UI 进程不计入 worker scope，但仍属于系统外部占用。

从默认 resident 任务切到小 B 时，旧会话可能已经超过 B。执行进入 `normalizing_residency`：无新大分配，等待旧工作完成并 unload/trim；低于 B 后才进入 `memory_scope_ready/admitted`。报告过渡前的 footprint，不掩盖曾高于 B 的事实。若用户要求“从 API 调用第一瞬间就绝不超过 B”，已有超标进程无法满足，必须拒绝或改用独立干净 worker；普通用户态配置不能追溯消除旧分配。

相反，从 constrained 回到 default，只恢复 backend global limits 与 legacy route；已经释放的 cache 不会凭空重新出现。首次 default 可能有 cold reload，单独计入切换延迟；稳态默认性能门禁使用匹配 cache state，不虚假保证跨模式切换零开销。

## 36. v2 block planner：非等大分层与确定性选参

### 36.1 保留 v1，新增独立接口

`tc_block_residency_plan_build` 现有代码将 `slot_bytes/minimum_bytes` 固定按两个 block 计算，预算容量不足两 block 时会拒绝；adaptive 分支也不是通用单槽调度器。不要修改这些 legacy 语义后让旧默认性能变化。

拟增 `native/runtime/block_residency_v2.{h,c}`，并在 `tools/native/build.sh` 加入构建。接口输入显式包含：

```c
typedef struct {
    uint64_t disk_bytes;
    uint64_t resident_bytes;
    uint64_t conversion_scratch_upper;
    uint32_t layout_family;
} tc_residency_unit_v2;
/* build_v2 输入另含：stage floor、unit array、支持槽数位集合、
 * grouping 边界、full-resident 能力、成本样本。
 * 输出：pinned 集合/首版 prefix、slots 各自 capacity/layout、minimum、peak。
 * 采用调用者提供的输出 buffer + capacity，不跨 C ABI 返回 std::vector。 */
```

首版 pinned 只选 prefix，避免顺序层遍历的普通 LRU 在每步出现循环失效；以后有证据再按单位 bytes 的未隐藏成本选非连续热点。少数 F32 modulation/scale 张量用自己的 typed slice 计入，不能把整个混合层错误按 BF16 memcpy。

### 36.2 候选构造与 lower bound

```text
for allowed backend/runtime/quality-preserving execution variants:
  for group size in certified group sizes (初始通常 1，后续 2/4):
    for k in backend certified slot counts intersect user maximum:
      for p in allowed pinned prefix counts:
        compute actual pinned bytes = sum(unit[i].resident_bytes, i<p)
        compute k slot envelopes by layout/payload assignment
        compute overlapping conversion scratch and stage live-set maximum
        reject if any stage/transition exceeds B or capability unavailable
        simulate ready/read/convert/compute/release events
select feasible candidate with lowest predicted end-to-end cost
```

50 层×3 槽×少量 group 的枚举足够便宜，不需要大规模最优化框架。所有候选使用 metadata/既有校准，不以全模加载 benchmark 搜索参数。

可运行最低预算为 `max(各阶段最小合法活跃集合, 各阶段过渡最低峰值)+R_proc`。对 streamed denoise，最小集合含 trunk、conditioning、当前 unit、认证槽位、scratch；只能认证双槽的 backend 不能用单槽理论下界准入。activation 或 VAE floor 超 B 时，减少 pinned 再多也无效。

full-resident 另枚举，slots=0。不使用“所有 layers pinned + 仍分配 2/3 refill slots”的重复容量；promotion/demotion 若无法直接转移 backing 所有权，旧新副本共存峰值需预算，禁止通过短暂超限完成迁移。

### 36.3 I/O 和 overlap 模型

固定 prefix、相同权重的简化重复读取量：

```text
read_bytes ≈ initial_pinned_read
           + actual_layer_traversals * sum(streamed_unit_disk_bytes)
           + component_reload_bytes
T_io_unit = seek/syscall_cost + disk_bytes/BW_io_under_compute + unpack_time
T_compute_unit = measured_GPU_time_at_same_shape_and_IO_state
```

actual_layer_traversals 根据真实 forward schedule，不盲目等于 UI steps（尤其多阶段/多 evaluation profile）。I/O+GPU 共享 DRAM 带宽，后台读也可能让 compute 变慢，BW 必须来自两者同时工作的测量。理论下界 `max(total_compute, total_io)` 仅作 sanity check；真实串行依赖、首尾填充和空槽等待需离散事件模拟。

建议 loader 默认一个顺序 reader、有限 8–32 MiB 读取块、最多认证槽数的在途任务；chunk 大小是 benchmark 参数，不是格式对齐保证。相邻 offset 合并成有界 range 读取，但不能为一个小 tensor 读整个 shard。先 CPU-read/GPU-compute overlap，再考虑额外 I/O 队列；SSD 慢时宁可承认 I/O-bound，也不靠无界 file cache 假装完全隐藏。

## 37. 具体测试规约与新增工具合同

### 37.1 必须覆盖的边界用例

| ID | 输入/注入 | 预期断言 |
|---|---|---|
| MP-01 | 无对象 / enabled=false | 旧输出字段与 route 不变；无 pager/telemetry 线程启动 |
| MP-02 | Y=0、bool、负数、小数；X=4/31 | parser 拒绝；尚无权重大分配 |
| MP-03 | Y=16GiB、X=15 | B=14602888806，checked 算术与显示一致 |
| MP-04 | add-on + 旧非零 budget / ANE manifest | 明确 conflict，不自动覆盖 |
| MP-05 | v1/v2/profile/Swift round-trip | 同一有效 policy digest；request false 覆盖 profile true |
| ML-01 | 两个 tensor view 指向同 backing | 唯一 capacity 只计一次 |
| ML-02 | cache->active / active->pending_release | 不重复计费，不提前归还可用容量 |
| ML-03 | 突发两个并发 reserve 都接近 B | 原子准入最多一个成功；无 TOCTOU 越线 |
| ML-04 | parent stage envelope + child allocation | 父容量不和子物理容量重复相加 |
| BP-01 | 等大 / 大小交替 / 混合 dtype | slot capacity 不小于实际填充，首层 bytes 不代表全部层 |
| BP-02 | minimum-1 / minimum / 全驻留 | 分别拒绝/计划可行/slots=0；运行能力另测 |
| BP-03 | H3 未认证单槽且 max=1 | 拒绝，不创建两个槽后谎报一个 |
| IO-01 | 短读、EINTR、EOF、源文件截断 | 完整读取或可解释失败；不向 GPU 发布半层 |
| IO-02 | 取消旧 job 后旧 read 继续写 | backing 未复用，generation 无跨请求污染 |
| GPU-01 | 延迟两条 queue 的 completion | 最后读者完成后才可覆写槽 |
| MX-01 | compiled 显式权重但未 eval | 旧引用不误判释放；group barrier 后才能 retire |
| MX-02 | 全局 cache/limit 请求后恢复 | success/cancel/error/prepare 后新 default 请求不受污染 |
| ST-01 | LTX upsample / spawn / exec | 同时存活峰值计入；worker budget ACK/退出回收无泄漏 |
| ST-02 | 旧 resident 会话切小 B | 规范化阶段显式显示；未就绪前不宣称已准入 |
| QT-01 | 各层 latent、最终 RGB/PCM | 同 backend 同 dtype 的 residency-only parity；采样步骤恰好一次 |
| DF-01 | VAE tiling halo/边界/长视频 | 边缘和时间状态质量单测；不等价路线不自动启用 |

### 37.2 新工具与运行约定（待实现）

拟增 `tools/native/benchmark_memory_constrained.py`：读取 campaign JSON，执行同预算串行/双槽/三槽/自动/resident baseline，对每轮创建独立输出目录，解析 trace 与计数，生成证据汇总。不能无参数自动遍历所有本机模型、下载 checkpoint 或制造系统压力。

campaign 配置至少包含模型路径、请求 JSON、预算点、期望 must_succeed/must_reject、rounds、cache_state、质量合同、timeout、监控 scope；路径只保留本机配置，不提交绝对模型路径。输出包含 `runs.jsonl`、`summary.json` 和 per-run trace；不把 prompt/用户输入原文默认写入公开性能报告。

建议 `tests/native/test_memory_policy.py` 等无权重脚本与 C/C++ fake backend 一起接入 `Makefile:test`；C/C++ 二进制编译放 TemporaryDirectory，跨平台/权限缺失明确 SKIP。`tools/native/build.sh` 有显式 source list，新增 `.cpp/.c/.mm` 必须分别加入，不假设自动 glob 生效。

阶段性交付必须至少完成一条真实成功路线（优先 H3 C 或 LTX C），不能所有 case 都靠 admission reject 获得“100%安全成功”。实际小内存硬件复验是 release gate：64 GiB 机器预算 16 GiB 只是 synthetic capacity test；冷 OS file cache 不能靠乱用系统 purge 在用户工作环境制造。

### 37.3 本次文档工作验证记录

当前工作树除本文外已包含配置/计划脚手架、首版 ledger/admission、LTX 原生
Metal buffer hooks 与 H3/LTX 受限路径接线；H3/MLX/graph/host-media 的完整逐
allocation 闭环、完整 pager 证据和真实受限模型生成仍未完成，也不声称低内存
性能已达标。2026-09-15 已执行的测试：

- `python3 -B tests/native/test_h3_streaming_policy.py`：PASS，编译并运行 shared H3/LTX policy 测试。
- `python3 -B tests/native/test_ltx_streaming_benchmark.py`：2 tests，PASS。
- `python3 -B tests/native/test_native_gguf.py`：4 tests，PASS。
- `MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" python3 -B tests/native/test_contract.py`：81 tests PASS，1 SKIP，包含新字段解析、预算算术、冲突、schema-v2、受限入口规范化、effective-plan 服务路由和生命周期接线。
- `python3 -B tests/native/test_memory_accounting.py`：PASS，覆盖 reservation、唯一 backing 去重、pending-release/cache 和 footprint admission。
- `python3 -B tests/native/test_ltx_gpu_memory_hooks.py`：当前环境无 Metal device，合法 `SKIP`；测试源码覆盖 reserve/commit/release、retain 不重复释放和超预算拒绝。
- `MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh`：PASS，native CLI/dylib 构建成功。

原生 CLI/动态库曾使用相同 `MLX_ROOT` 成功完成编译；之后的 Swift 集成构建长时间无输出并被中止，因此不能记为完整 `make build` PASS。

这些测试证明现有基础行为、首版 ledger 和 LTX 集中 allocator hook 可复用，不证明
底层 GPU allocation 已全部受控、MLX graph envelope 已认证、或新 pager 的全请求
预算合同已完成。后续实施必须新增上述测试并保留实际模型 evidence。

## 38. 平台语义依据与源码证据索引

2026-09-15 核实的官方语义：MLX memory API 将 memory limit 定义为 graph 执行 guideline，active counter 不包含空闲 cache；`get_array_buffer_size` 对已求值数组的底层 backing 去重，但新增 API 是否存在要以项目实际链接版本为准。Metal `recommendedMaxWorkingSetSize` 是性能影响的近似工作集阈值，不是 OS 硬配额。查验的是官方源码/文档，未将 master API 存在等同于本机旧库可链接。

可复查来源（地址置于代码块，实施时记录实际版本/commit）：

```text
MLX 官方 memory API 源码：
https://raw.githubusercontent.com/ml-explore/mlx/main/mlx/memory.h
Apple Metal recommendedMaxWorkingSetSize 官方文档：
https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize
```

关键本地证据：

- `native/runtime/block_residency.c`：v1 两槽 minimum、adaptive/full-resident 规则。
- `native/models/h3_runtime/h3_streaming_policy.c`：H3 wrapper 固定两槽且不走 full-resident。
- `native/models/ltx_runtime/ltx_blocks.c`：先 load block 0 再计算 layout、48 blocks、top/conditioning 随后加载。
- `native/models/ltx_mlx/model.cpp`、`block.hpp`：BlockCache prefix/refill、显式权重编译图与 eval 生命周期。
- `native/platform/apple/h3_mlx_checkpoint.mm`、`h3_mlx_session.mm`：whole-map 权重持有、VDN branch、stage cleanup/完整 RGB 输出。
- `native/models/flux2/flux_transformer.cpp`：dual/single 层序、4B single compile 输入与减少 host waits 的默认优化。
- `native/models/z_image/z_image.cpp`、`z_image_gguf_module.cpp`：transformer/VAE 联载、refiner/main 层序与 GGUF 类型/驻留白名单。
- `native/api/c_api.mm`、`request.mm`（位于 `native/platform/apple/`）、`services/turbociderd/service.mm`：锁顺序、schema、旧 plan 缺路径和 worker 路由边界。

实施者以当前源码和本设计的能力门禁为准；历史 streaming benchmark/旧 worker 文档只作为背景，不作为当前后端已支持的证明。

## 39. 当前仓库实现状态、已知缺口与禁止误报

本节把“文档设计”与“当前代码”明确分开，避免在实现过程中因为字段已经能解析就误报功能完成。

### 39.1 已落地的脚手架

截至 2026-09-15，仓库中已经存在以下改动：

| 区域 | 已实现行为 | 仍然不是的东西 |
|---|---|---|
| `native/core/memory_contracts.hpp` | 定义 `MemoryConstrainedConfig`、字段来源 bitmask 和默认值 | 不是运行时账本，也不持有任何 GPU buffer |
| `native/platform/apple/request.mm` | 解析 schema-v1 顶层与 schema-v2 `execution.memory_constrained`，做类型和白名单校验 | 不读取 checkpoint metadata，不做实际准入 |
| `native/platform/apple/profile.mm` | 按 profile → request 逐字段 overlay，request 的 `enabled=false` 可覆盖 profile | 不会自动改变旧 `residency` 或旧 `memory_budget_bytes` |
| `native/runtime/memory_policy.*` | 计算 `B=floor(Y*(100-X)/100)`、冲突检查、digest 和 estimate 状态 | `set_memory_limit`、OS footprint、Metal heap 均未被此类逐分配拦截 |
| `native/runtime/memory_accounting.*` | 首版 `MemoryLedger`、reservation/lease 生命周期、唯一 backing 去重、macOS footprint 观测和阶段 checkpoint；`MemoryAdmission` 已改为 baseline + concrete allocation ceiling，避免 root/child 双重计费 | 仍是过渡模型；H3/LTX 仅部分 allocation 接入，MLX/graph、其余 host/media 与 required unknown 尚未闭环，root 尚未升级为正式可分割 `MemoryBudgetScope` |
| `native/models/h3_runtime/h3_gpu.*`、`h3_dit.c`、`h3_text_encoder.c` | 可选 GPU hooks、classified tensor/upload、mmap/load reservation、双 refill-slot tag 已接入；VAE/text/common-DiT/Euler velocity 的 required 路径已开始分类；旧 API 保持 `NULL` fast path | 量化/ANE/Core ML 可选分支和其他旧 API 调用仍需分类或首发不可达证明；vision/audio encoder 未认证 |
| `native/models/h3_runtime/h3_video_vae.*`、`h3.c` | 已新增 `_with_options` API，resident decoder、chunked decode、single-shot decode 均传播 `h3_gpu_options`；required GPU tensor 已迁移 classified API | options 贯通和首批分类仍不等于全 decoder allocation registry、exact upper 或 L3 完成 |
| `native/models/h3_runtime/h3.h`、`h3.c`、`native/platform/apple/h3_session.mm` | 已新增独立 host hooks 和统一 domain/generation；joint video/audio latent、RGB8 输出、decoded F32/tile 聚合 envelope 已接入 reserve/commit/release | conditioning/layout/tokenizer/control 与 VAE 内部小对象尚未完整纳管；`decoded_host_envelope_v1` 的 `4x` 上界尚未由 manifest 校准 |
| `native/models/ltx_runtime/ltx_gpu.*`、`ltx_blocks.c`、Gemma GPU wrapper | 可选 C ABI memory hooks；原生 Metal buffer 执行 reserve→allocate→commit，并在 wrapper/deferred batch 生命周期结束后 release；部分 weights/activation 已分类 | 尚未证明所有 LTX 分配均分类；MPSGraph temporary、MLX VAE/upsampler/cache、host vector 和输出 arena 不在该 hook 的完整控制范围内 |
| `native/platform/apple/ltx_session.mm` | `uses_parent_mlx(const Request&)` 已使受限请求固定走 C/Metal，并对 operation/input/audio/LoRA/residency 做执行时二次 gate | vector capacity、helper 子进程、VAE graph envelope 和全部 cancel 路径仍需独立认证 |
| `native/runtime/plan.cpp` | 受限请求输出 `EffectiveMemoryPolicy`，H3/LTX candidate 规范化；candidate 初始为 `hook_bridged + plan_only`，不再由 estimate 自动标记 `execution_supported` | 仍未完成 manifest/L3 认证，不能执行 |
| `native/platform/apple/results.mm` | plan JSON 返回 policy、estimate、reason、scope 以及 capability/certification/digest 字段 | 这是解释信息，不是“本次运行未越线”的证明 |
| `native/api/c_api.mm` | `generate`/非 cache `prepare` 统一执行受限 guard、清理旧 session、建立 `MemoryAdmission`、按阶段采样 footprint，并把规范化 request 传给 session | admission 目前仍以过渡型 allocation ceiling 计账，尚未覆盖全部 native/MLX/Metal allocation；失败 JSON 仍有稳定前缀而非完整结构化对象 |

因此当前受限生成的正确行为是（H3/LTX candidate 可生成 plan，但在无 verified manifest 时
执行入口 fail-closed；其余模型同样 fail-closed）：

```text
parse -> make_plan -> normalize effective request -> capability match
      -> plan_only（当前默认；不创建 checkpoint/模型大 allocation）
      -> verified manifest + envelope => admission/execute
      -> unsupported/unknown/over-budget => stable memory_* error
```

这个结果是安全的实验状态，但不能作为发布完成。`plan.cpp` 已移除
`route_available && estimate_fits` 自动设置 `execution_supported=true` 的逻辑；后续
实现 PR 必须在真实 H3 C/Metal 或 LTX C/Metal 路线通过全请求证据后，才可以把该路线的
capability 从 `hook_bridged/plan_only` 升级为 `allocation_guarded`、`envelope_validated`
或 `l3_certified`。

### 39.2 必须先修正的调用链问题

后续实现不得只在 `plan.cpp` 中修改 `r.residency`，而忘记执行入口仍使用原始 request。建议新增一个集中入口，避免 `generate`、`prepare`、服务 worker 三处各自复制逻辑：

```cpp
struct PreparedExecution {
    Request effective_request;          // 规范化后的 residency/budget/backend
    ExecutionPlan plan;                 // 含 candidate、estimate、policy
    std::unique_ptr<MemoryAdmission> admission;
};

PreparedExecution prepare_execution(tc_engine &engine,
                                    const Request &requested,
                                    ExecutionIntent intent);
```

推荐顺序：

```text
request_from_json
  -> make_plan(requested)                 // 无权重、副作用为零
  -> session.probe_memory_layout(plan)    // metadata-only；必要时验证 hash
  -> plan_residency_candidates
  -> choose candidate / normalize request
  -> system admission + MemoryLedger root lease
  -> session.generate_constrained(effective_request, context)
```

`ExecutionPlan.request` 必须保存规范化后的 request；原始用户字段另存为 `requested_request` 或 trace 字段。不得让 session 再次根据环境变量偷偷推导不同的预算。

### 39.3 已完成的入口修正与仍需推进的事项

以下三个容易被遗漏的调用链问题已经在当前工作树修正，后续改动不得回退：

1. `native/runtime/plan.cpp` 先对 H3 C/Metal、LTX C/Metal video-only 做 candidate normalization，再调用 `module_for(...).validate(r)`；因此旧的 `resident`/`streamed` 白名单不会提前拦截可规范化请求。
2. `native/api/c_api.mm` 的 `generate` 与非 cache `prepare` 均把 `request_plan.request` 作为 effective request 传给 session，并在 session 前执行受限 guard/admission。
3. `finalize_memory_policy_estimate` 在估算超预算时输出 `rejected`，在 candidate fit 但
   无 capability 时保留 `plan_only` reason；`require_memory_constrained_execution_supported`
   先区分 `memory_budget_too_small` 与 `memory_policy_unsupported`。

仍需推进的调用链工作是：

- 将 `PreparedExecution` helper 提升为真正的共享实现，服务 worker、CLI 和 C API 都只调用这一条路径；服务路由已经改为读取 effective plan，但 submit/planning/admission 仍需进一步统一，避免服务层复制准入逻辑。
- 在 `ExecutionPlan` 中保存 `requested_request`、`effective_request`、`candidate_id` 和 `planner_revision`，避免后续 trace 只能看到规范化后的请求。
- 将 `MemoryAdmission` 的 root envelope 改为可分割的 parent scope，避免未来把 root reservation 与每个 child allocation 重复计费。

这些事项不能通过把 `execution_supported` 默认改成 true、或修改测试期望来绕过；必须有明确的 allocation/worker 证据。

## 40. 端到端调度器的实现合同

### 40.1 入口统一与能力分层

`MemoryConstrained` 不是一个独立 backend，而是包在现有 model session 外部的策略层。每个 session 需要声明三层能力：

```cpp
enum class ConstrainedCapability {
    None,
    ResidentEnvelope,       // 只证明完整组件能在 B 内
    ComponentStaged,         // 阶段间释放/重载
    BlockStreaming,          // 已验证 block pager + overlap
    BlockStreamingTiled      // 另有 decoder/workspace tiling 证据
};

struct MemoryCapabilities {
    std::set<ConstrainedCapability> levels;
    std::set<unsigned> certified_refill_slots; // 例如 H3={2}, LTX={1,2,3}
    std::set<unsigned> certified_group_sizes;
    std::set<std::string> operations;
    std::string envelope_revision;
    bool supports_runtime_replan = false;
    std::vector<std::string> unsupported_reasons;
};
```

静态 `ModelDescriptor` 只用于列出大致路线；真实 `MemoryCapabilities` 必须结合 checkpoint hash、LoRA 身份、shape、backend 和 MLX/Metal 版本复核。静态 descriptor 与动态 probe 不一致时，以动态 probe 的 fail-closed 结果为准。

### 40.2 `PreparedExecution` 的伪代码

以下伪代码描述 C++ 层面的唯一准入路径，实际名称可以调整，但不能省略步骤：

```cpp
PreparedExecution prepare_execution(tc_engine &engine,
                                    const Request &requested,
                                    ExecutionIntent intent) {
    auto plan = make_plan(requested);
    if (!plan.memory_policy) {
        // 保持旧路径：不创建 ledger、pager、sampler 或 scoped MLX limit。
        return {plan.request, std::move(plan), nullptr};
    }

    auto &policy = *plan.memory_policy;
    auto probe = engine.session()->probe_memory_layout(plan.request, intent);
    require(probe.complete, "memory_estimate_unknown: " + probe.reason);

    auto candidates = enumerate_candidates(probe, policy,
                                           engine.device_inventory());
    auto candidate = choose_lowest_cost(candidates);
    require(candidate.has_value(), explain_infeasible(candidates, policy));

    plan.request.residency = candidate->residency;
    plan.request.memory_budget_bytes = candidate->denoiser_sub_budget;
    policy.execution_supported = candidate->capability_verified;
    policy.admission_state = "candidate_selected";
    policy.reason.clear();

    auto root = engine.memory_admission().try_admit(
        candidate->full_request_upper_bytes, policy, intent);
    require(root.ok(), root.error_message());
    policy.admission_state = "admitted";
    return {plan.request, std::move(plan), std::move(root.lease)};
}
```

其中 `denoiser_sub_budget` 仅供已有 H3/LTX C runtime 使用，不能被解释成全进程 cap。全请求上限由 `full_request_upper_bytes` 和 root lease 保证；两者必须在结果中同时输出。

### 40.3 生成、prepare、worker 的一致性

- `tc_engine_generate`：调用 `prepare_execution`，随后将 `effective_request` 传入 `generate_constrained`；结果中合并 root ledger snapshot 和 pager trace。
- `tc_engine_prepare`：warmup 也必须先 admission。若 warmup 只建立编译图，应使用单独的 `CompileLease`，不能把编译临时峰值隐藏在生成阶段。
- `tc_engine_cache`：缓存管理操作不是推理请求，不应继承上一个受限请求的全局 MLX limit；清理动作要在无 in-flight graph 时执行。
- `tc_engine_load`：保持 ABI 和旧行为。它没有 request，不能凭空加载一个受限会话；文档和 SDK 应明确“受限客户端先调用 request-aware prepare/generate”。
- `services/turbociderd/service.mm`：服务层只做排队、worker 路由和进程树生命周期，不自行计算 B；所有字段原样透传给 native plan。

## 41. MemoryLedger 的可执行实现细节

### 41.1 数据结构与状态不变量

账本只对“TurboCider 能识别、能在分配前拿到容量上界”的对象负责。建议实现以下最小类型：

```cpp
enum class MemoryClass {
    ProcessBaseline, Weights, Activation, Conditioning,
    RefillSlot, ConversionScratch, Output, AllocatorCache,
    CompileTemporary, ChildProcessEnvelope, UnknownExternal
};

struct StorageId {
    uint64_t allocator_domain;
    uint64_t handle;
    uint64_t capacity;
    uint64_t session_generation;
    bool operator==(const StorageId &) const = default;
};

enum class LeaseState { Reserved, Committed, PendingRelease, Cached, Returned };

struct MemorySnapshot {
    uint64_t budget = 0;
    uint64_t committed = 0;
    uint64_t active = 0;
    uint64_t pending_release = 0;
    uint64_t cached = 0;
    uint64_t observed_footprint = 0;
    uint64_t peak_committed = 0;
    uint64_t peak_observed = 0;
    std::string pressure;
};
```

必须保持的恒等式：

```text
committed = process_baseline
          + unique(live backings)
          + pending_release
          + cached_not_returned
          + unconsumed reservations
```

`live`、`pending_release`、`cached` 三种状态互斥；父 `StageEnvelope` 只提供容量，不再作为一份 backing 加入物理总量。相同 MLX backing 的多个 view、Metal suballocation 和 C++ wrapper 通过 `StorageId` 去重。

### 41.2 分配事务与异常安全

所有受控分配必须遵守：

```text
checked_size
  -> try_reserve(class, upper, tag)
  -> platform_allocate(upper)
  -> verify(actual_capacity <= upper)
  -> commit(StorageId, actual_capacity)
  -> publish_to_model
```

任何一步失败都要在 publish 前 rollback。`commit` 不能静默扩容；如果平台返回的 alignment/pack 尺寸超过 reservation，记录 `memory_lifetime_violation` 并终止当前 job。C callback 不抛 C++ 异常，错误由 owner thread 汇总。

建议 `MemoryLedger` 内部使用一把短持有时间的 mutex；不要在 ledger 锁内执行 `pread`、`mx::eval`、Metal wait 或文件系统操作。`try_reserve` 必须在同一锁下检查和扣减，避免两个线程 TOCTOU 越线。

### 41.3 异步释放与 allocator cache

`retire(storage, CompletionToken)` 只把对象转为 `PendingRelease`；最后一个 GPU/CPU 读者完成后，才允许迁移为 `Cached` 或 `Returned`。MLX `clear_cache`、Metal heap trim、C malloc free 都必须由 owner thread 在 drain 点调用。缓存仍可被复用时，其容量继续计入 `committed`。

取消路径先停止新提交，再等待所有 token；不能为了快速响应而释放仍被 GPU 读取的 slot。跨请求复用必须增加 `session_generation`，旧 generation 的 completion 永远不能回收新 generation 的 backing。

## 42. WeightPager、预取与双/三槽执行

### 42.1 Pager 接口

建议增加与模型无关的 reader/pager 接口：

```cpp
struct WeightUnit {
    uint32_t id;
    uint64_t file_offset;
    uint64_t disk_bytes;
    uint64_t resident_bytes;
    uint64_t conversion_scratch;
    uint32_t family;
};

class WeightSource {
public:
    virtual ~WeightSource() = default;
    virtual uint32_t unit_count() const = 0;
    virtual WeightUnit unit(uint32_t) const = 0;
    virtual void read(uint32_t unit, void *dst, uint64_t capacity,
                      std::atomic<bool> &cancel) = 0;
};

class WeightPager {
public:
    PrefetchTicket prefetch(uint32_t unit, SlotId slot);
    ReadyWeight await_ready(PrefetchTicket);
    CompletionToken submit_use(const ReadyWeight &, GpuQueue &);
    void retire_after(const ReadyWeight &, CompletionToken);
    void cancel_and_drain();
};
```

首版只允许顺序层序和 prefix pinned；不要一开始引入通用 LRU。每个 slot 至少有以下状态：

```text
FREE -> RESERVED -> IO_INFLIGHT -> READY -> GPU_INFLIGHT
     -> RETIRING -> FREE
```

状态标识必须包含 `job_id/session_generation/slot_epoch/unit_id`。任何 I/O 完成回调先核对 generation 和 epoch，再发布 READY；旧任务的回调只能丢弃发布动作，不能释放新任务内存。

### 42.2 overlap 提交顺序

以当前 unit `i` 为例，允许的顺序是：

1. `current` slot 已 READY，并登记 GPU 使用 token。
2. 在仍有 reservation 的 FREE slot 中启动 `pread(i+1)`；I/O 线程只能写 slot，不得构建 MLX graph。
3. 当前 block/group 的 Metal/MLX compute 提交后，立即把下一 unit 的 conversion 任务排入受控队列。
4. 当前 GPU event 与 conversion event 均完成后，slot 进入 FREE；未完成时禁止覆写。
5. 连续两个 block 出现 starvation，且预算处于 green/yellow，才在下一个边界评估增加 look-ahead。

没有 FREE slot 时只能有界等待或退回串行；不得临时 malloc “第三槽”。H3 的 C/Metal 能力初始固定 `certified_refill_slots={2}`，LTX C/Metal 根据实际 `LTX_MAX_REFILL_SLOTS` 和 envelope 逐项开放 1/2/3。

### 42.3 读文件安全

`pread` 必须循环处理 EINTR 和短读；EOF、offset+length 越界、文件身份改变均为不可重试错误。读取目标的 capacity 来自 reservation，不能根据磁盘文件大小在 worker 线程随意增长。受限模式优先有界 buffer + `pread`，mmap 只在能给出 clean-page/working-set 证据时采用。

## 43. 规划器的确定性算法与模型专用预算

### 43.1 候选枚举

规划器输入是 `(model_id, checkpoint_identity, backend, operation, shape, steps, policy, device_inventory)`，输出候选列表和每项拒绝原因。候选枚举顺序固定，保证同一输入得到相同 plan digest：

```text
for capability in [resident, component_staged, streamed, tiled]:
  for group_size in certified_group_sizes:
    for slots in certified_slots ∩ [1, max_refill_slots]:
      for pinned_prefix in legal_prefixes:
        calculate exact/validated upper envelope
        reject if any stage or transition > B
        simulate IO/compute events
        retain only quality-equivalent candidates
sort by (feasible, predicted_wall, io_wait, fewer_slots)
```

候选成本不允许用“失败概率惩罚秒数”掩盖未知上界；上界缺失直接 reject。排序结果必须包含 `planner_revision`、`estimate_revision`、`candidate_id`，供 trace 和复现使用。

### 43.2 H3 C/Metal 首条垂直切片

H3 C runtime 已有 SSD block 读取和双槽，因此首条实现只接 `minimax-h3-turbo`、`execution=gpu/auto`、C/Metal、video/audio 由现有 executor 真实支持的组合。planner 先计算：

```text
H3_denoise_floor = trunk + conditioning + activation + conversion_scratch
H3_minimum       = R_proc + H3_denoise_floor + 2 * block_bytes
H3_slots         = 2                         (固定认证能力)
H3_pinned        = min(active_blocks-1,
                       floor((B-R_proc-H3_denoise_floor)/block_bytes)
                       - H3_slots)
```

`memory_budget_bytes` 传给 H3 的值应是扣除 non-denoiser 存活集合后的 `H3_sub_budget`；全请求 B 仍由 root lease 负责。H3 不能因为理论上 1 slot 能装下就把 `max_refill_slots=1` 映射给未修改的双槽 loader。

H3 适配完成的标志：

- 生成前无需加载整套 denoiser 才能计算 pinned/slot plan；
- 每个 slot allocation 经过 ledger reservation；
- `h3_result` 返回 bytes loaded、refill、wait、actual peak；
- text、denoise、VAE、export 全阶段都有 envelope；
- 同 checkpoint 的 resident 与 streamed 输出通过既定质量门，且 constrained 运行没有 ledger 越线。

### 43.3 LTX C/Metal 首条垂直切片

首批只开放 `ltx_backend=c_metal`、`execution=gpu/auto`、`residency=streamed`、`operation=video.generate`、`audio=false`、非 I2V。LTX 的 48 blocks 和最多三个 refill slot 作为独立能力声明，不与 H3 共用 block bytes。

规划至少包含：

```text
text_floor + conditioning
stage1_latent + stage1_workspace
upsample_old_latent + upsample_new_latent
stage2_latent + stage2_workspace
video_vae_decode_tiles + output_arena
```

Stage1→upsample→Stage2 的瞬时旧/新 latent 重叠是最容易漏计的峰值。只有通过该过渡 envelope，才可以把 `ltx_blocks.c` 的 streaming 结果标成 `execution_supported=true`。I2V、audio、C++/MLX BlockCache 继续单独认证，不能复用 C/Metal 结果。

### 43.4 Flux、Z-Image 与 FastH3/VDN

- Flux 4B/9B：在没有 layer-group pager 和显式 graph 输入之前，只允许 `component_staged`；`memory_constrained` 下如果 staged envelope 仍超过 B，返回 `memory_budget_too_small`。
- Z-Image/GGUF：先拆分 text/transformer/VAE 生命周期，再实现 offset reader；mixed K-quant、QKV remap 或 dequant 临时峰值未验证时拒绝。
- FastH3 MLX/VDN：当前 whole-map tensor 持有与 branch workspace 不满足全请求合同，只能报告 `memory_policy_unsupported` 或支持已验证 resident envelope，不能把现有 `residency=streamed` 当作证据。

## 44. MLX/Metal 的延迟分配、cache 与硬上限边界

### 44.1 MLX envelope

MLX 的 `set_memory_limit`/`set_cache_limit` 只能作为 scoped guideline。受限路径必须为每个 graph/eval 预留：

```text
graph_input_backings
+ graph_output_backings
+ max temporary/workspace
+ compiled executable cache delta
```

在 envelope 覆盖前禁止 `mx::eval`；若无法取得运行时上界，使用经过固定 shape/backend 校准的 whole-graph envelope，否则 fail-closed。`async_eval` 返回不是 GPU 完成事件；首版可使用 owner thread `mx::synchronize()` 作为保守边界。

### 44.2 scoped limits 的恢复

`ScopedMlxLimits` 必须保存旧 `memory_limit`、`cache_limit`、shape/cache key，并在 success、cancel、exception、prepare、worker exit 五条路径恢复。恢复顺序：drain graph → 释放受控引用 → trim cache（若允许）→ 恢复旧 limit。受限请求不能污染后续默认请求。

不要主动提高 wired limit，也不要把 `recommendedMaxWorkingSetSize` 当成可分配许可。Metal/MLX/进程 footprint 的快照用于差异诊断，不能相加形成“总内存”。

### 44.3 观测不等于硬拦截

当 backend 没有逐次分配 veto 时，系统能保证的是：planner 预留 + 已认证 graph envelope + 可观测异常时 fail-closed。结果中必须明确：

```json
{
  "enforcement": "ledger+validated_graph_envelope",
  "os_hard_cap": false,
  "observed_scope": "inference_process_tree_v1",
  "unknown_bytes": 0,
  "observation_quality": "qualified"
}
```

`unknown_bytes>0` 或 graph envelope invalid 时，`observation_quality` 只能是 `unreliable`，不得显示“已满足上限”。

## 45. 服务、worker 与 GPU 锁的生命周期

### 45.1 锁顺序

保持现有顺序并新增 bytes admission：

```text
engine mutex (try-lock)
  -> process-wide execution_mutex (try-lock)
  -> DeviceLease (cross-process GPU lock)
  -> MemoryAdmission root lease
  -> ModelSession / pager
```

不要持有 GPU `DeviceLease` 无限等待内存；如果 root admission 失败，应按相反顺序释放并把原因交给服务队列。MemoryLedger 不能代替跨进程 GPU 互斥。

### 45.2 LTX disposable worker

如果 constrained LTX 需要独立 worker，parent 必须先预留 child envelope，再 spawn；spawn 期间 parent+child 同时存在，峰值按两者相加。worker 启动参数至少包含：

```text
job_id, policy_digest, stage_budget, plan_revision,
checkpoint_identity, request_path, parent_pid
```

child 在首个大分配前回传 `admission_ack`；ACK 超时、崩溃、取消都由 parent 回收 child lease。exec 替换进程镜像并不自动释放 parent 的 staging/cache；服务仍需 wait/终态事件回收。临时 latent/PCM 文件放每 job 专属目录，失败只清理该目录。

### 45.3 会话切换

默认 resident → constrained 小 B：进入 `normalizing_residency`，等待旧工作完成并 unload/trim，低于 B 后才准入；期间不得声称“从 API 第一瞬间开始就在 B 内”。constrained → default：恢复全局 limits，允许 cold reload，默认 ABBA 必须用匹配 cache state 比较。

## 46. 对外 API、版本兼容和错误 JSON

### 46.1 schema 兼容

保留 schema-v1 顶层和 schema-v2 `execution.memory_constrained` 两种输入，规范化为同一内部对象。新增字段不得改变旧 request 的序列化；旧客户端不发送对象时，`memory_policy` 字段不出现在 plan/result 中。

建议未来增加 metadata-only API（不改变已有 ABI）：

```c
int tc_engine_estimate_memory(tc_engine *engine,
                              const char *request_json,
                              char **out_json,
                              char **error);
```

该 API 只允许读取 header/index/manifest，不 materialize 权重；若模型没有 metadata probe，返回 `memory_estimate_unknown`。不要通过“先完整 load 再测 active bytes”实现 estimate。

### 46.2 统一错误结构

C API 仍返回非零状态，但 `error` 字符串建议稳定为 JSON：

```json
{
  "schema_version": 1,
  "code": "memory_budget_too_small",
  "message": "effective budget is below the certified minimum",
  "model": "minimax-h3-turbo",
  "required_bytes": 18000000000,
  "budget_bytes": 14602888806,
  "largest_contributor": "stage2_workspace",
  "retryable": false,
  "plan_revision": 3
}
```

`memory_policy_unsupported`、`memory_estimate_unknown`、`memory_observation_unreliable`、`memory_pressure_abort` 和 `weight_io_error` 必须可由客户端区分；不要把所有失败压成“out of memory”。

### 46.3 计划输出字段

受限 plan 至少增加：

```text
requested_limit_bytes, effective_budget_bytes, buffer_percent,
process_reserve_bytes, system_reserve_bytes,
estimate_upper_bytes, estimate_provenance,
candidate_id, effective_residency, certified_refill_slots,
admission_state, execution_supported, reason,
policy_digest, planner_revision, envelope_revision
```

`estimate_only`、`candidate_selected`、`admitted`、`running`、`completed` 是不同状态；不能在 `plan` 命令阶段输出 `admitted`。

## 47. 追踪、诊断和性能归因

### 47.1 事件格式

建议使用有界 JSONL 队列，每条事件包含：

```json
{
  "ts_ns": 0,
  "job_id": "…",
  "generation": 4,
  "stage": "denoise",
  "unit": 17,
  "slot": 1,
  "event": "io_ready",
  "committed_bytes": 13200000000,
  "active_bytes": 12700000000,
  "observed_footprint": 14100000000,
  "budget_bytes": 14602888806,
  "watermark": "yellow",
  "io_bytes": 458752000,
  "wait_ns": 0,
  "plan_revision": 3
}
```

生产默认只写 stage summary 和聚合计数；slot 级事件由诊断开关启用，并限制队列长度、单 job 文件大小和 prompt/路径脱敏。trace writer 不能阻塞 GPU owner thread；队列满时丢弃低优先级 detail，但保留错误、越线和终态事件。

### 47.2 必须区分的指标

`committed_bytes` 决定准入；`active_bytes` 表示 backend 引用；`observed_footprint` 反映进程采样。I/O overlap 使用时间线交集：

```text
overlap_ratio = duration(IO ∩ GPU) / duration(IO)
gpu_starvation = time(GPU queue waits for READY slot)
```

不要用 `1 - wait/(wait+compute)` 命名为 I/O 隐藏率，也不要只报告 MLX active memory 而忽略 C/Metal staging 和 child worker。

## 48. 验收分层、测试夹具与 CI 门禁

### 48.1 无权重纯单测

先加入不需要模型文件的测试，确保调度器本身可证明：

- `B` 公式、checked overflow、X/Y 类型和 schema round-trip；
- profile overlay 与旧字段冲突；
- `MemoryLedger` reservation/commit/retire/pending-release/cache 去重；
- 父 envelope 与 child reservation 不重复计费；
- H3/LTX fixture 的 minimum、pinned、slot、full-resident golden；
- pager generation、短读、取消和延迟 completion；
- pressure state machine 的 deterministic fallback。

测试应使用 fake allocator/fake GPU event/fake reader，不以 sleep 模拟正确性。每个 fake backing 都有固定 `StorageId`，可检验重复 view 不增加 capacity。

### 48.2 真实模型分级门禁

每个宣称支持的 adapter 至少提供以下 workload：

| 层级 | 条件 | 通过标准 |
|---|---|---|
| L0 | metadata-only | candidate 与 minimum 计算可复现，未知项为零或明确 reject |
| L1 | synthetic/fake backend | ledger 与 pager 无越线、无泄漏、取消可恢复 |
| L2 | 小真实 checkpoint | constrained 成功，peak/质量/trace 齐全 |
| L3 | 目标 checkpoint + 代表性 shape | 冷/热、最小/默认/大 shape、swap/pressure 证据通过 |
| L4 | 长视频/LoRA/worker 路径 | helper/导出/异常/重启后仍无 lease 泄漏 |

只有 L2/L3 通过后，`execution_supported` 才能在该模型/shape/backend 组合中置 true。L0/L1 通过不能写成“已支持低内存生成”。

### 48.3 minimum-1 / minimum / baseline 三点

每条路线都必须测：

1. `B = minimum - 1`：must reject，且不发生大权重分配；
2. `B = minimum`：must succeed 或按能力声明明确拒绝，不能出现偶发越线；
3. `B = baseline`：与 resident/component-staged 做质量和性能比较。

预算点要固定记录 Y、X、B，不能只记录一个“memory limit”字符串。任何采样超 B、ledger commit 超 B、或 unknown envelope 都使该轮 FAIL/UNQUALIFIED，而不是通过后再回收。

### 48.4 默认模式 ABBA

每次修改 pager/ledger/MLX wrapper 后，对 add-on disabled 执行至少三轮 ABBA：

```text
A = 旧默认 request
B = 同 request + memory_constrained.enabled=false
A = 旧默认 request
B = 同 request + memory_constrained.enabled=false
```

比较输出像素/latent、route、cache/compile、wall、peak；稳态 median 回归目标 ≤2%，冷启动单独报告。关闭路径不能创建 sampler 线程、调用新 scoped limit 或写受限 trace。

### 48.5 本机运行约定

64 GiB 开发机上的 8/16 GiB Y 只能证明 scheduler synthetic capacity，不等价于真实 8/16 GiB 物理机。测试脚本必须记录 macOS、MLX、Metal device、checkpoint hash、swap 单调计数和外部干扰；污染轮次保留为 `CONTAMINATED`，不可选择性删除。

## 49. 发布、回退和运维策略

### 49.1 功能开关

用户可见开关只有 request 的 `memory_constrained.enabled`。实现阶段可以有内部 capability gate（例如 `TC_MEMORY_CONSTRAINED_H3_C=1`），但 gate 关闭时应返回 `memory_policy_unsupported`，不能转回无限制旧路径。不得提供 `fail_if_infeasible=false` 之类的绕过开关。

### 49.2 运行中回退

受限任务发生 pressure/ENOMEM 时，回退顺序固定：

```text
stop optional prefetch
  -> drain safe events
  -> release dead cache/component
  -> shrink pinned/look-ahead at certified boundary
  -> enable pre-approved decoder tiling
  -> abort with memory_pressure_abort
```

不自动切换 dtype、steps、sampler、sparse/sol 或另一个 checkpoint。质量变化必须由用户显式请求并重新生成 digest/admission。

### 49.3 版本化与回滚

`policy_digest`、`planner_revision`、`envelope_revision`、`checkpoint_identity` 必须写入结果和 benchmark。升级 MLX/Metal、模型转换器或权重 sidecar 时，旧 envelope 自动失效并回到 estimate-only；不能沿用旧校准数字冒险运行。回滚实现时保留 schema 解析和错误码兼容，但删除未认证的 adapter capability。

## 50. 实现 PR 顺序与逐项完成清单

推荐按可回滚的提交拆分，每个提交都要有独立测试：

1. **Policy contract**：配置、profile overlay、B 公式、错误码、plan JSON；默认 ABBA。
2. **Ledger core**：`MemoryLedger`、`MemoryAdmission`、fake allocator/event、lease 生命周期。
3. **Planner v2**：`MemoryEstimate`、metadata probe、H3/LTX fixture、候选排序和 explain。
4. **H3 C/Metal adapter**：双槽 pager、sub-budget、全阶段 floor、真实 must-succeed workload。
5. **LTX C/Metal adapter**：48-block pager、stage1/2 transition、video-only T2V、output/export 预算。
6. **MLX envelope**：scoped limits、graph envelope、generation/compile cache 隔离；先不开放模型 capability。
7. **Flux/Z-Image staged**：组件生命周期和明确拒绝矩阵；后续再做 layer/tensor streaming。
8. **Pressure/telemetry**：sampler、watermark、trace、污染环境报告、故障注入。
9. **Service/worker**：parent-child reservation、ACK、重启/取消清理、SDK/CLI round-trip。
10. **Release gates**：L0–L4、ABBA、swap/pressure、文档状态同步。

每个 PR 的描述必须回答第 34 节清单，并附：

```text
model/checkpoint hash
request + Y/X/B
planner/envelope revision
candidate and rejection table
committed/active/observed peak
swap delta and contamination status
quality contract and output hash
default-mode ABBA summary
```

当且仅当 H3 C/Metal 与 LTX C/Metal 至少各有一条 L3 真实成功路线、Flux/Z-Image 的拒绝矩阵可复现、默认路径 ABBA 通过、取消/异常无泄漏时，才将文档顶部状态从 Proposal + scaffolding 改为 Implemented（仍需保留“非 OS 硬上限”措辞）。

## 51. 最小可编译改造样板（按当前源码落地）

本节给出比前文更接近 patch 的实现顺序。代码是结构化样板，不要求逐字复制；关键是保持调用顺序、错误语义和默认路径零开销。

### 51.1 `ExecutionPlan` 增量字段

第一步不改变现有 aggregate 初始化之外的公开 ABI，只在 C++ 内部扩展：

```cpp
struct ExecutionPlan {
    Request requested_request;             // 新增：原始规范化前输入
    Request request;                      // 现有字段：最终 effective request
    Recipe recipe;
    std::optional<uint64_t> memory_estimate_bytes; // 兼容总值
    std::optional<MemoryEstimate> memory_estimate; // 新增分项 envelope
    std::optional<EffectiveMemoryPolicy> memory_policy;
    std::optional<ResidencyCandidate> residency_candidate;
};
```

若暂时不引入 `MemoryEstimate` 类型，至少不要复用 `memory_estimate_bytes` 承载“全请求 B”和“denoiser working-set”两个语义。向后兼容的 JSON 继续输出旧字段，同时新增 `estimate_upper_bytes`、`estimate_scope`。

### 51.2 `make_plan` 的正确顺序

现有 `make_plan` 中，模型模块 `validate` 早于受限 residency 映射。建议拆出两个纯函数：

```cpp
static void normalize_constrained_request(Request &r,
                                          EffectiveMemoryPolicy &policy) {
    if (!policy.enabled) return;
    require(r.execution != "gpu_ane" && r.encoder_ane_manifest.empty(),
            "memory_policy_conflict: GPU-only constrained mode");

    if (r.model == "minimax-h3-turbo" &&
        (r.execution == "auto" || r.execution == "gpu")) {
        require(r.ltx_backend == "auto", "unexpected backend for H3");
        r.execution = "gpu";
        r.residency = "streamed";
        policy.admission_state = "candidate_pending_h3_c_metal";
        return;
    }

    if (r.model == "ltx-2.5-distilled" &&
        (r.execution == "auto" || r.execution == "gpu") &&
        (r.ltx_backend == "auto" || r.ltx_backend == "c_metal") &&
        r.operation == "video.generate" && !r.audio &&
        r.inputs.empty()) {
        r.execution = "gpu";
        r.ltx_backend = "c_metal";
        r.residency = "streamed";
        policy.admission_state = "candidate_pending_ltx_c_metal";
        return;
    }

    policy.reason = "no certified constrained execution adapter";
}
```

`make_plan` 中的建议顺序：

```text
copy requested -> build policy -> normalize candidate -> validate recipe/module
-> build stage recipe -> estimate -> finalize policy -> return plan
```

如果规范化需要读取 checkpoint header，不要在 `make_plan` 中做；由 `probe_memory_layout` 补充 metadata 后重建 candidate，避免 `plan` 命令产生大内存副作用。

### 51.3 `tc_engine_generate` 的最小替换

当前入口已经做了 capability guard，但仍把原始 `request` 传给 session。最小改造应将有效 request 贯穿后续逻辑：

```cpp
auto requested = tc::request_from_json(tc::parse_json(r));
auto prepared = tc::prepare_execution(*e, requested,
                                      tc::ExecutionIntent::Generate);
const auto &effective = prepared.plan.request;

const bool parent_mlx = e->session->uses_parent_mlx(effective);
// configure streams / scoped MLX limits / event callback ...
auto result = e->session->generate(effective, event, e->cancelled);
result.plan = prepared.plan;
result.memory_ledger = prepared.admission->snapshot();
```

若第一阶段还没有 `generate_constrained` 虚函数，可暂时由 H3/LTX session 识别 `effective.residency=="streamed"` 进入现有 C/Metal path；但必须在注释和结果中标记 `experimental_adapter_v1`，并在 adapter 真实证据前保持 capability gate 关闭。不要让 Flux/Z-Image 由于 `effective.residency` 被强行改为 streamed。

### 51.4 `preparation_call` 的防绕过改造

非 cache 分支应与 generate 使用同一 helper：

```cpp
auto requested = std::move(*parsed_request);
auto prepared = tc::prepare_execution(*e, requested,
                                      warmup ? ExecutionIntent::Warmup
                                             : ExecutionIntent::Prepare);
auto effective = prepared.plan.request;
effective.dump.clear();
auto result = e->session->prepare(effective, warmup != 0, event,
                                  e->cancelled);
```

如果 warmup 的 envelope 与 generate 不同，`ExecutionIntent` 必须让 planner 返回不同的 `CompileTemporary` 上界；不能直接复用 generate 的 root lease。`cache=true` 的 coreml cache 管理动作不调用该 helper，但必须先 drain 现有受限 graph 并恢复 scoped limit。

### 51.5 H3 C adapter 的字段传递

在 `native/platform/apple/h3_session.mm` 中，受限 candidate 生成后按以下方式传值：

```cpp
const auto &candidate = *plan.residency_candidate;
parameters.ssd_streaming = candidate.residency == "streamed";
parameters.ssd_memory_budget_bytes = candidate.denoiser_sub_budget;
parameters.ssd_pinned_prefix = candidate.pinned_prefix;
parameters.ssd_refill_slots = 2; // 首版固定，除非能力声明新增
```

当前 H3 C ABI 没有 `ssd_refill_slots` 时，不要通过改 `max_refill_slots` 伪造单槽；应先扩展 ABI 或保持固定双槽。扩展 C ABI 时新增字段放结构体尾部、增加 `struct_size`/version，旧 caller 传旧大小时使用默认双槽。

### 51.6 LTX C adapter 的 probe-before-load

在 `native/models/ltx_runtime/ltx_blocks.c` 中新增只读 probe：

```c
typedef struct {
    uint32_t block_count;
    uint64_t block_resident_bytes;
    uint64_t conversion_scratch_bytes;
    uint64_t stage1_floor_bytes;
    uint64_t stage2_floor_bytes;
    uint32_t certified_refill_slots_mask;
} ltx_block_layout;

int ltx_native_probe_block_layout(const char *checkpoint,
                                  ltx_block_layout *out,
                                  char *error, size_t error_size);
```

`ltx_native_create` 先调用 probe，再按 `B - R_proc - floor` 计算 pinned/slots，最后才读取 block 0。正式读取时重新检查 header identity 和 block bytes，与 probe 不一致立即 abort；不允许“先读取 block 0，超预算后再决定”。

### 51.7 结果和错误的兼容写法

在现有 `results.mm` 中保留：

```json
{
  "memory_estimate_kind": "conservative_heuristic_not_hard_limit"
}
```

受限路线通过真实 admission 后再增加：

```json
{
  "memory_policy": {
    "admission_state": "admitted",
    "execution_supported": true,
    "enforcement": "ledger+validated_graph_envelope",
    "effective_residency": "streamed",
    "denoiser_sub_budget_bytes": 0,
    "estimate_upper_bytes": 0,
    "actual_peak_committed_bytes": 0,
    "observation_quality": "qualified"
  }
}
```

错误字符串若暂时无法升级为 JSON，至少保持稳定前缀（`memory_budget_too_small:`、`memory_policy_unsupported:` 等），并在 C API 文档中声明客户端必须按前缀解析；后续再无破坏性地切换为错误 JSON。

### 51.8 构建与测试接入

`tools/native/build.sh` 使用显式 source list。新增 `.cpp/.c/.mm` 时分别加入对应编译段，并在 `tests/native/test_contract.py` 中增加“文件已进入构建清单”的静态断言，避免本地 IDE 能编译但发布脚本漏编译。

每个真实 adapter PR 至少执行：

```sh
MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" make build
MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" \
  python3 -B tests/native/test_contract.py
python3 -B tests/native/test_memory_accounting.py
python3 -B tests/native/test_residency_planner.py
python3 -B tools/native/benchmark_memory_constrained.py \
  --campaign tests/fixtures/memory/h3_c_metal.json
```

其中 `make build` 的 Swift 集成阶段如果因环境/权限长时间无输出，应记录为 `INCOMPLETE`，不能只凭 native CLI 编译通过就宣称整库构建成功。benchmark 不得在没有 checkpoint 或没有 `must_succeed` 证据时把 reject 轮次统计成成功。

## 52. 从当前脚手架到真实受控执行器的差距

本节把实现工作拆成“必须改变的语义”和“可以暂时保守的工程选择”。实现者应先完成语义，再做性能优化；不能通过增加 buffer、频繁 `clear_cache` 或等待 swap 来掩盖账本没有接入的问题。

### 52.1 当前首版可以保证什么

当前代码已经能保证以下行为：

- 请求 schema、类型、范围、profile overlay 和旧字段冲突是 fail-closed 的。
- `B=floor(Y*(100-X)/100)` 的计算使用 checked arithmetic，`limit_bytes` 不会静默截断。
- H3/LTX 受限 candidate 会规范化为有效 GPU/C/Metal streamed request；session 收到的是 effective request，而不是原始 resident request。
- `MemoryAdmission` 在创建时采样进程 footprint，检查 `baseline + planned_increment <= B` 和系统可用内存，随后在阶段边界采样。
- `MemoryLedger` 的 reservation、唯一 backing 去重、pending-release、cache 计费和 generation 字段已有独立测试。
- add-on 关闭时不创建 `MemoryAdmission`，不调用受限 scoped limit，也不改变默认 session 的生命周期。

### 52.2 当前首版不能保证什么

以下能力在底层 callback 完成前不得写入 release note 或用户文档：

- 每个 `MTLBuffer`、MPSGraph 临时 backing、`malloc` staging 或 mmap clean page 都经过同一个分配闸门。
- `ledger_peak_committed_bytes` 等于进程物理 footprint；两者目前只是互补指标。
- MLX `set_memory_limit` 能拒绝任意一次 graph 分配；官方 API 的 limit 只能作为 graph guideline。
- macOS 在两个 footprint checkpoint 之间绝不短暂超过 B；没有 OS hard cap 的前提下无法作此承诺。
- worker 子进程、helper、编码器和外部媒体工具的峰值已被父进程 admission 覆盖。

因此，当前未认证 candidate 必须保持：

```text
execution_supported=false
capability_level=hook_bridged
certification_state=plan_only
```

只有在 52.3 的准入门槛全部满足，并且 capability/manifest identity 精确命中后，才可
把 capability 升级为 `block_streaming_ledger_v2` 并开放执行。

### 52.3 capability 升级门槛

每个 `(model, checkpoint_identity, shape_bucket, backend, operation)` 维护独立 capability key。升级顺序固定为：

1. `estimate_only`：metadata probe 可复现，所有 unknown 项显式列出。
2. `envelope_validated`：小 checkpoint 或 fake allocator 对每一类分配给出 upper bound。
3. `allocation_guarded`：native allocator callback 已接入，越界 reservation 在分配前失败。
4. `must_succeed_l2`：真实小模型成功，输出质量、ledger、footprint、取消均通过。
5. `must_succeed_l3`：目标 checkpoint 和代表性 shape 冷/热运行通过，swap/pressure 证据合格。

任何升级失败只回退到上一等级；不能保留旧 `execution_supported` 标记并在运行时“尽量尝试”。

## 53. 层级 reservation：避免 root envelope 与 child allocation 重复计费

早期脚手架曾把 `planned_increment_bytes` 作为一笔 `UnknownExternal` root reservation；当前工作树已采用 53.2 的过渡方向：root 不作为一项 storage 长期占用，ledger budget 设为 `initial_process_footprint + planned_increment`，真实 child allocation 直接进入 ledger。这样避免了已接入的 LTX child buffer 与 root envelope 双重计费，但 `budget_bytes`、`ledger_budget_bytes` 和 planned ceiling 的语义仍需要正式化。最终实现必须在以下模型中固定一种，不得又把 root reservation 加回同一个 `committed`。

### 53.1 推荐：可分割的 `MemoryBudgetScope`

root admission 表示“可用容量”，child scope 从 root 中分割额度；child 的实际 storage 计账，root 的剩余额度不再作为第二份 backing：

```cpp
class MemoryBudgetScope {
public:
    std::optional<MemoryBudgetScope> split(
        MemoryClass cls, uint64_t upper_bytes, std::string tag);
    std::optional<StorageLease> commit(
        const StorageId&, MemoryReservation&& reservation);
    uint64_t remaining_bytes() const noexcept;
    void cancel() noexcept;
};

struct MemoryAdmission {
    MemoryLedger ledger;          // baseline + unique storage
    MemoryBudgetScope root;       // capacity only, not another storage
};
```

推荐的不变量为：

```text
ledger.committed = baseline + unique(active/pending/cached backing)
root.remaining   = planned_increment - sum(child reservations)
ledger.committed + root.remaining <= B
```

`split()` 必须在 ledger mutex 下完成，child scope 析构时只归还未消费额度；已经 `commit()` 的 storage 由 `StorageLease` 生命周期负责。父 scope 销毁前必须确认 `children==0`，否则记录 `memory_lifetime_violation`。

### 53.2 过渡方案：root 只作为 upper-bound sentinel

当前过渡实现已经把 root reservation 从 `MemoryLedger` 中移出，并令 concrete-allocation ceiling 为 `initial_process_footprint + planned_increment_bytes`。后续若继续沿用该方向，应把 ceiling 作为一等字段，而不是依赖 `MemoryLedger::budget_bytes` 的隐含解释；每次 child allocation 调用 `try_reserve`，root 不再调用 `ledger_.try_reserve`。若未来允许多个 stage scope 并发，还必须防止两个线程同时消耗剩余额度：

```cpp
bool try_child(uint64_t upper) {
    std::lock_guard lock(mutex);
    if (child_reserved > planned_increment - child_committed) return false;
    child_reserved += upper;
    return true;
}
```

建议将指标改名/扩展为：

```text
effective_request_budget_bytes = B
initial_process_footprint_bytes = F0
planned_increment_bytes         = E
allocation_ceiling_bytes        = F0 + E
ledger_known_storage_bytes      = unique known backing（含 baseline）
unattributed_footprint_bytes    = max(observed_footprint-ledger_known, 0)
```

当前兼容字段 `ledger_budget_bytes` 暂时等于 `allocation_ceiling_bytes`；新旧字段至少保留一个 schema revision，防止采集工具把它误解为用户输入的 B。该过渡方案不适合在没有额外 child-scope 互斥的情况下做跨 stage 并行 admission，也不能把 `planned_increment` 当作一项 `UnknownExternal` storage 输出。正式版仍建议迁移到 53.1。

### 53.3 分配事务的强制顺序

所有 C/Objective-C/C++ allocator 必须遵循同一事务：

```text
checked_size
  -> scope.split(class, upper, tag)
  -> platform_allocate(upper)
  -> actual_capacity <= upper
  -> scope.commit(StorageId{domain, handle, actual, generation})
  -> publish wrapper
```

失败处理：

- `platform_allocate` 返回空：取消 reservation，转为 `memory_budget_too_small` 或 `platform_allocation_failed`，不调用任何模型 fallback。
- 实际容量大于 upper：立即释放刚分配 backing，抛出 `memory_lifetime_violation`；不得扩容 reservation。
- wrapper publish 失败：先销毁 platform object，再取消 reservation。
- 取消信号到达 I/O 线程：保留 reservation 直到 reader 返回；不能在仍可能写入的 buffer 上调用 free。

## 54. LTX C/Metal allocator 接入设计

LTX 是首个适合完成逐 allocation 接入的后端，因为 `ltx_gpu_buffer_new()` 集中创建 `MTLBuffer`，`ltx_gpu_buffer_free()` 集中释放 wrapper。实现者应先接入这里，再扩展 graph workspace 和 H3。

### 54.1 C ABI 扩展

不要在现有函数签名中加入 C++ 类型。新增一个尾部可扩展的 hooks/options 结构：

```c
typedef struct {
    uint32_t struct_size;
    uint32_t version;
    void *user;
    /* Return non-zero and fill token before platform allocation. */
    int (*reserve)(void *user, uint32_t memory_class,
                   uint64_t upper_bytes, const char *tag,
                   void **token, char *error, size_t error_size);
    /* Called after MTLBuffer is created. actual_bytes must be <= upper. */
    int (*commit)(void *user, void *token, uint64_t allocator_domain,
                  uint64_t handle, uint64_t actual_bytes,
                  uint64_t generation, char *error, size_t error_size);
    /* Called when platform allocation/wrapper publication/commit fails. */
    void (*cancel)(void *user, void *token);
    /* The token remains opaque to the C backend; the bridge owns the
     * allocator-domain/handle/generation identity captured at commit.
     * Called after the last safe wrapper/deferred-command reference. */
    void (*release)(void *user, void *token);
} ltx_gpu_memory_hooks;

```

当前工作树把 `max_refill_slots`、`memory_hooks`、`memory_allocator_domain` 和
`memory_generation` 直接追加在现有 `ltx_native_options` 与
`ltx_gemma_encoder_options` 中；后续若继续扩展，必须保持这些字段的尾部追加方式，
不要重排已有字段。旧 caller 的 `memory_hooks=NULL` 时保持现有无 hook 行为。受限
session 必须传完整 hooks、非零 allocator domain 和 generation，并在 hook 缺失时
fail-closed；不能把旧 ABI 的 `memory_budget_bytes` 当作逐 allocation 约束。
`struct_size/version` 位于 hooks 本身；若未来需要 options 版本门禁，应在 options
结构末尾新增独立 size/version，不能改变现有字段偏移。

### 54.2 `ltx_gpu_buffer` 字段

建议在 `ltx_gpu_buffer` 中增加以下 opaque 字段：

```c
struct ltx_gpu_buffer {
    void *buffer;
    size_t bytes;
    uint32_t references;
    void *memory_token;       /* reservation/lease owned by this wrapper */
    uint64_t allocator_domain;
    uint64_t generation;
    uint64_t handle;
    uint8_t accounted;
};
```

`handle` 应来自稳定的单调计数器或 `MTLBuffer` 指针转整数后的 generation-bound identity；不能只用裸指针，因为地址可能被 allocator 复用。`newBufferWithLength` 成功后使用实际 `buffer.length` 作为 `actual_bytes`，而不是假设等于请求值。

### 54.3 分配与释放伪代码

```c
ltx_gpu_buffer *ltx_gpu_buffer_new(ltx_gpu *gpu, size_t bytes,
                                   char *error, size_t error_size) {
    if (!gpu || !bytes) return fail_invalid(...);
    void *token = NULL;
    if (gpu->hooks.reserve && !gpu->hooks.reserve(
            gpu->hooks.user, LTX_MEMORY_CLASS_BUFFER,
            (uint64_t)bytes, "ltx_gpu_buffer", &token,
            error, error_size))
        return NULL;

    id<MTLBuffer> native = [device newBufferWithLength:bytes
        options:MTLResourceStorageModeShared];
    if (!native) {
        if (gpu->hooks.cancel) gpu->hooks.cancel(gpu->hooks.user, token);
        return fail_platform(...);
    }

    ltx_gpu_buffer *result = calloc(1, sizeof(*result));
    if (!result) {
        if (gpu->hooks.cancel) gpu->hooks.cancel(gpu->hooks.user, token);
        return fail_host_tracking(...);
    }
    result->handle = ++gpu->next_buffer_handle;
    result->bytes = [native length];
    if (gpu->hooks.commit && !gpu->hooks.commit(
            gpu->hooks.user, token, gpu->allocator_domain,
            result->handle, result->bytes, gpu->generation,
            error, error_size)) {
        CFBridgingRelease((__bridge void *)native);
        free(result);
        return NULL;
    }
    result->memory_token = token;
    result->accounted = 1;
    ...
}
```

`ltx_gpu_buffer_free()` 在 `references` 降到零后，先释放 Objective-C strong/bridged object，再调用 ledger `release`；若 GPU command buffer 仍引用该 object，不能直接降到零。当前 LTX deferred command array 已在 `ltx_gpu_submit()` 等待最后一个 command buffer，建议将释放动作挂到 command buffer completion handler，只有没有 in-flight token 时才调用 `release`。

### 54.4 command buffer 与 lease 的关系

每次 kernel/graph 提交都要把 buffer 的 in-flight 引用计数加一，并把 `MTLCommandBuffer` completion handler 作为 release token。推荐新增：

```c
int ltx_gpu_buffer_use_begin(ltx_gpu_buffer *buffer,
                             id<MTLCommandBuffer> command);
int ltx_gpu_buffer_use_end(ltx_gpu_buffer *buffer,
                           id<MTLCommandBuffer> command);
```

如果不想暴露 Objective-C 类型，可由 `ltx_gpu_begin()` 返回 opaque command epoch，内部在 `ltx_gpu_submit()` 统一批量完成。关键不变量是：

```text
wrapper references == 0  不代表可归还 ledger
可归还条件 = wrapper references == 0 && gpu_inflight == 0 && cpu_readers == 0
```

这正是 `StorageLease::retire()` + `complete_pending()` 的使用场景。不要在 `ltx_gpu_continue_releasing()` 中直接 `free` 仍被 GPU 使用的 buffer。

## 55. H3 Metal allocator 与 mmap 权重接入设计

H3 的分配入口分为三类，不能只修改 `h3_gpu_tensor_new()`：

| 类别 | 当前位置 | 账本类别 | 处理方式 |
|---|---|---|---|
| 可写 activation/output | `h3_gpu_tensor_new` | Activation/Output | 分配前 reservation，tensor 持有 lease |
| `newBufferWithBytesNoCopy` mmap 权重 | `h3_gpu_tensor_load_file`、`h3_gpu_tensor_map_bf16` | Weights | 记录 Metal backing + mmap range；clean page 不重复计为第二份 storage |
| ANE/IOSurface bridge | `h3_gpu_tensor_wrap_f32` 及 pack/unpack | ConversionScratch/UnknownExternal | 若无法取得容量上界则受限模式拒绝 hybrid；不能标记为已计账 |

### 55.1 H3 hooks

为了保持旧 C ABI，新增 `h3_gpu_create_with_options()`，旧 `h3_gpu_create()` 调用默认 options：

```c
typedef struct {
    uint32_t struct_size;
    uint32_t version;
    uint64_t generation;
    uint64_t allocator_domain;
    void *user;
    int (*reserve)(void *, uint32_t, uint64_t, const char *, void **,
                   char *, size_t);
    int (*commit)(void *, void *, uint64_t, uint64_t, uint64_t, uint64_t,
                  char *, size_t);
    void (*cancel)(void *, void *);
    void (*release)(void *, void *);
} h3_gpu_memory_hooks;
```

`H3Tensor` 增加 opaque token、`gpu_inflight` 和 `read_only` 标识。`h3_gpu_tensor_load_file()` 中 mmap 失败后调用 reservation cancel；`newBufferWithBytesNoCopy` 成功但 callback commit 失败时必须先释放 MTLBuffer，再 `munmap`。

### 55.2 mmap 与 clean-page 语义

同一文件范围可能同时存在 mmap page cache 和 MTL shared backing。账本至少要区分：

- `StorageId(allocator_domain=metal, handle=...)`：Metal 对象容量。
- `StorageId(allocator_domain=mmap, handle=file+offset+length, generation=...)`：mmap 虚拟范围 upper bound。

如果平台实测表明 `newBufferWithBytesNoCopy` 不复制 payload，可把两者绑定为一个 composite allocation，并在结果中标记 `mmap_aliasing=validated`。在证据出现前，保守做法是两项都计入 planned envelope；不能把文件大小全部当作物理常驻，也不能直接假设零拷贝。

### 55.3 最终释放与 H3 pipeline

H3 已有 `h3_gpu_continue_releasing()`，应扩展为：

1. 将 tensor wrapper 置为 `retiring`，阻止新的 kernel 使用。
2. 将最后一个 command buffer completion token 绑定到 tensor。
3. completion 回调中调用 `h3_gpu_tensor_free` 和 ledger `complete_pending`。
4. 若进入可复用 cache，状态为 `Cached` 且 bytes 继续计入；只有真正销毁才转 `Returned`。

`h3_gpu_stats.live_bytes` 只能作为 backend 统计，不再单独代表受限模式的 committed bytes；结果同时输出 `ledger_peak_committed_bytes` 和 `h3_gpu_stats.peak_live_bytes`。

## 56. LTX block pager 的真实调度实现

### 56.1 block metadata probe

`ltx_native_probe_block_layout()` 必须只读取 checkpoint header、tensor index 和 block offset，不创建 MTLBuffer。probe 输出至少包含：

```c
typedef struct {
    uint32_t layout_version;
    uint32_t block_count;
    uint32_t stage_count;
    uint64_t checkpoint_size;
    uint64_t block_payload_bytes[48];
    uint64_t block_runtime_bytes[48];
    uint64_t conversion_scratch_bytes;
    uint64_t stage_floor_bytes[2];
    uint32_t certified_refill_slots_mask;
    uint8_t checkpoint_identity_sha256[32];
} ltx_block_layout;
```

如果 block payload 不是等大，planner 必须按 `max(window)` 而不是 `count * average` 计算；`average` 只能放在诊断信息中。probe 与正式 load 之间重新 `fstat` 并校验 checkpoint identity，任何变化都返回 `weight_io_error`。

### 56.2 双/三槽环

首版 LTX C/Metal 允许 `slots ∈ {1,2,3}`，但每个槽都必须有独立 reservation。推荐 ring 状态：

```text
FREE
  -> RESERVED(unit, epoch)
  -> IO_INFLIGHT
  -> READY
  -> GPU_INFLIGHT(command_token)
  -> RETIRING
  -> FREE
```

调度循环：

```cpp
for (uint32_t i = 0; i < block_count; ++i) {
    auto current = pager.await_ready(i);
    pager.mark_gpu_use(current, gpu_command_epoch);

    if (i + 1 < block_count && pressure.green())
        pager.prefetch(i + 1, pager.find_free_slot());

    submit_block(current);
    pager.retire_after(current, gpu_command_epoch);
    pressure.observe_boundary(i);
}
```

`find_free_slot()` 没有结果时只允许：等待最早 completion、减少 look-ahead、或串行读取；严禁临时 `malloc` 第四槽。两个连续 starvation 且水位为 green 时才增加 look-ahead；yellow/red 水位只能保持或减少。

### 56.3 LTX graph workspace

MPSGraph 可能创建 scheduler 不可见的 temporary backing。首版应为每个 `(stage, shape, precision, graph_revision)` 建立校准 envelope：

```text
graph_upper = max(observed_peak - known_inputs, 0) + alignment_margin
```

校准数据必须绑定 runtime/Metal/MLX 版本和 shape bucket。若 graph revision 改变、输入 shape 超出 bucket 或校准样本不足，candidate 状态改为 `estimate_unknown`，而不是沿用旧 upper bound。

## 57. H3/LTX 全阶段预算拆分与重规划

### 57.1 子预算不是固定 4 GiB

当前实现为 H3/LTX 使用 `non_denoiser_reserve_bytes=4 GiB` 的保守值。后续 planner 应按阶段动态生成：

```text
non_denoiser(stage) = process_baseline_delta
                    + conditioning
                    + latent_or_input
                    + output_arena
                    + conversion_scratch
                    + graph_or_compile_temp(stage)
                    + safety_margin(stage)
```

`denoiser_sub_budget = B - non_denoiser(stage)` 仅在 denoise 阶段使用；进入 VAE 时必须释放 denoiser reservation，再为 VAE weights/workspace 建立新的 child scope。不能把全请求 B 直接写入 H3 的 `ssd_memory_budget_bytes`。

### 57.2 边界状态

每个阶段通过显式状态对象传递 live set：

```cpp
struct StageLiveSet {
    uint64_t conditioning_bytes;
    uint64_t latent_bytes;
    uint64_t output_bytes;
    uint64_t required_weight_bytes;
    uint64_t scratch_upper_bytes;
    bool next_stage_preload_allowed;
};
```

切换前执行：

```text
stop optional prefetch
-> submit final current-stage commands
-> wait/drain completion tokens
-> retire dead weights and complete_pending
-> checkpoint footprint + ledger
-> admit next-stage live set
-> load next-stage weights
```

只要 `current_stage_live + next_stage_load > B`，就不得预加载；必须等旧阶段释放完成后再加载。预加载优化不是正确性要求，不能让它改变 candidate 的 feasibility。

### 57.3 运行中重规划边界

首版只允许在以下安全点重规划：

- denoise step 完成后，所有当前 step command buffer 已完成；
- LTX Stage1→upsample→Stage2 transition；
- H3 text→denoise、denoise→VAE；
- VAE tile 之间且 tile 状态可序列化。

禁止在单个 attention kernel 中途、正在写入输出的 command buffer 中途或 VAE halo 处理中途更换 tile/slot 数。重规划失败直接 `memory_pressure_abort`，不能偷偷回到 resident。

## 58. MLX envelope 与受限模式隔离

### 58.1 scoped limit 的职责

`ScopedMlxLimits` 只负责把受限 graph 的 guideline 和 cache limit 设置为计划值，并在所有退出路径恢复旧值。它不承担逐分配 veto：

```cpp
class ScopedMlxLimits {
public:
    ScopedMlxLimits(uint64_t graph_limit, uint64_t cache_limit,
                    std::string policy_digest);
    ~ScopedMlxLimits() noexcept;
    void drain_and_restore();
};
```

构造和析构必须运行在已持有 `execution_mutex()` 的线程；不得让多个线程并发修改 process-global MLX limit。恢复顺序固定：`synchronize → release graph refs → trim optional cache → restore old limits`。

### 58.2 graph envelope 认证

每个 MLX graph 建立 manifest：

```json
{
  "graph_revision": "mlx-graph-v3",
  "runtime_version": "0.29.x",
  "device_family": "Apple-M4-Max",
  "shape_bucket": [512, 512, 25],
  "known_input_bytes": 123456789,
  "temporary_upper_bytes": 987654321,
  "cache_delta_upper_bytes": 456789,
  "samples": 12,
  "status": "qualified"
}
```

manifest 缺失、版本不匹配或 `samples==0` 时，MLX 路线只能返回 `memory_estimate_unknown`。不能把某次 `mx::get_active_memory()` 的结果直接升级为 hard upper bound。

### 58.3 默认路径零污染

add-on disabled 时：

- 不创建 `ScopedMlxLimits`；
- 不调用 `mx::clear_cache()`；
- 不把 graph 缓存转移到 constrained ledger；
- 不改变 `force_eval_each_block`、compile cache key 或 resident session reuse；
- 不启动 footprint sampler 线程。

受限请求结束后必须恢复全局状态；随后执行一个默认请求，结果中的 route、cache hit、compile key 和 wall time 应与 ABBA 的 A 轮一致。

## 59. 服务层与 worker 路由改造

当前工作树的 `services/turbociderd/service.mm` 已将受限 LTX 路由切换为读取 plan 的
`memory_policy`，并持久化 `service_route`；后续仍需把 submit/planning/admission
状态统一收敛到 `PreparedExecution`，避免服务、CLI 和 C API 各自复制准入逻辑。

### 59.1 路由原则

```text
request JSON
  -> tc_plan_json
  -> inspect plan.memory_policy.effective_residency
  -> inspect plan.memory_policy.adapter_candidate
  -> choose native session / disposable worker
```

路由只读取 plan 中的 `effective_*` 字段，不自行复制模型白名单。建议增加：

```json
{
  "memory_policy": {
    "effective_residency": "streamed",
    "adapter_candidate": "ltx_c_metal_streamed_video_v1",
    "worker_scope": "same_process",
    "parent_child_envelope_bytes": 0
  }
}
```

受限 LTX C/Metal 首版默认 `worker_scope=same_process`，因为当前受限路径已经禁用 helper process finalizer 以避免 parent+child 峰值无法计账。只有实现 45 节的 child reservation/ACK 后，才能将某个受限 candidate 路由到 disposable worker。

### 59.2 submit 前 admission

服务队列不能把“排队成功”误写成“内存已准入”。建议状态拆分为：

```text
queued -> planning -> waiting_for_memory -> admitted -> running
      -> succeeded / failed / cancelled / interrupted
```

`waiting_for_memory` 只表示 GPU execution mutex 或 system reserve 暂时不可用；如果 checkpoint envelope 本身大于 B，应直接 `failed(memory_budget_too_small)`，不能无限重试。

### 59.3 worker 子进程合同

若未来启用 child worker：

1. parent 先为 `ChildProcessEnvelope` 建立 reservation；
2. spawn 后 child 读取同一个 `policy_digest`、`plan_revision`、checkpoint identity；
3. child 在第一笔大分配前回传 `admission_ack`；
4. parent 收到 ACK 才释放“spawn temporary” reservation；
5. child 退出后 parent 等待并完成 child lease；超时/崩溃都算失败；
6. parent 的 latent/staging 不因 `exec` 自动消失，必须显式清理。

## 60. 错误、取消和恢复语义

### 60.1 错误分类

错误码必须能指导客户端下一步，而不是统一返回 ENOMEM：

| code | 含义 | 是否重试 | 客户端建议 |
|---|---|---:|---|
| `memory_policy_invalid` | Y/X/slots/schema 非法 | 否 | 修正请求 |
| `memory_policy_conflict` | 与旧 budget/ANE/hybrid 冲突 | 否 | 移除冲突字段 |
| `memory_budget_too_small` | upper envelope 或 stage minimum 大于 B | 否 | 增大 Y、降低 shape 或换模型 |
| `memory_estimate_unknown` | metadata/graph envelope 不完整 | 否 | 使用已认证路线或等待支持 |
| `memory_policy_unsupported` | 模型/operation/backend 无受限 adapter | 否 | 关闭 add-on 或换已支持路线 |
| `memory_pressure_abort` | 运行中 pressure/观测越界/无法重规划 | 视环境 | 先清理其它任务后重试 |
| `memory_observation_unreliable` | footprint/system sample 不可用 | 否 | 在支持的 macOS 环境重试 |
| `weight_io_error` | offset/EOF/identity/短读错误 | 否 | 检查 checkpoint |
| `memory_lifetime_violation` | allocator 实际容量或 lease 生命周期违约 | 否 | 作为实现 bug 处理 |

### 60.2 取消顺序

```text
cancel requested
-> stop new prefetch and graph submission
-> mark pager cancel flag
-> wait current GPU command/event
-> drain IO/conversion queue
-> retire slots and complete pending leases
-> release stage scopes and restore MLX limits
-> checkpoint("cancelled")
-> return status=2
```

取消不能跳过 completion token；否则下一次 job 可能复用仍被 GPU 写入的 backing。所有取消路径都要检查 `snapshot().storage_count==0` 或明确记录跨请求 cache 的容量。

### 60.3 失败恢复

错误发生后必须保证：

- 当前 `policy_digest` 的 scoped limits 已恢复；
- `MemoryAdmission` metrics 仍可序列化到 error JSON（至少包含 phase、ledger peak、observed peak）；
- pager 的 I/O 线程已 join，不能在 session 销毁后回调悬空 user pointer；
- H3/LTX session 可安全重新 `load` 或被 `unload`；
- 默认模式的下一次请求不继承 constrained cache 或 sampler。

## 61. 性能实现与 overlap 调优

### 61.1 三条独立队列

受限路径推荐逻辑上拆成：

```text
IO queue       pread / mmap / checksum
Convert queue  dequantize / pack / dtype cast
GPU queue      MTL command buffer / MPSGraph evaluation
```

队列之间只通过 bounded slot 和 event/token 传递，不在每个 block 上执行全局 synchronize。若 backend 只能使用单线程 owner，仍可以把 I/O 放在 worker thread，把 convert 任务推入受控 serial queue。

### 61.2 look-ahead 自适应规则

每个安全点统计最近窗口：

```text
io_wait_ratio = gpu_starvation_ns / gpu_active_ns
overlap_ratio = duration(IO ∩ GPU) / duration(IO)
```

规则建议：

- green 且 `io_wait_ratio > 0.10` 连续两个窗口：slots 或 look-ahead +1（不超过认证上限）。
- yellow：不增加 slots；若 `committed > 0.92B`，优先释放可选 cache。
- red/critical：停止预取，等待当前 slot 完成；安全边界减少 pinned prefix。
- `io_wait_ratio < 0.02` 连续三个窗口：可减少一个 slot 以降低统一内存压力，但只在该 slot 不影响当前 step 的情况下执行。

所有阈值写入 trace 和 plan revision；不能让自适应导致同一 request 的 candidate digest 不稳定。digest 绑定初始 candidate，运行时只记录 `schedule_revision`。

### 61.3 I/O 与文件缓存

`pread` 必须循环处理 EINTR/短读；每次读取前检查目标 slot 的 capacity。对大顺序权重流，建议使用 `fcntl(F_NOCACHE)` 或等价的已验证路径，但必须通过 benchmark 证明不会降低小 block 命中率。不能无条件对所有 checkpoint 开启 NOCACHE；默认模式也不能被影响。

### 61.4 性能门槛

受限模式的性能不以 resident 的绝对速度作为硬性要求，而以同一预算下的可预测退化为准：

- H3 双槽相对同 shape resident baseline 的 denoise wall 退化目标 ≤35%；
- LTX 三槽 video-only 相对 component-staged baseline 的 denoise wall 退化目标 ≤25%；
- 连续 starvation 超过 10% 的运行只能标记 `performance_unqualified`，不能发布为“充分 overlap”；
- 内存越界、unknown bytes 或 swap 污染优先级高于速度，速度达标但安全不达标仍 FAIL。

这些是首轮实验门槛，不是 API 合同；硬件、checkpoint 和 runtime 版本必须写入 benchmark。

## 62. 代码修改清单（按文件和依赖顺序）

| 顺序 | 文件/模块 | 具体修改 | 依赖 | 验收 |
|---:|---|---|---|---|
| 1 | `native/runtime/memory_accounting.*` | 增加 `MemoryBudgetScope` 或 transition sentinel；分类统计、child scope、error snapshot | 无 | fake ledger 单测 |
| 2 | `native/runtime/session.hpp` | `ExecutionPlan` 增加 requested/effective request、candidate、estimate revision；声明 `probe_memory_layout`/`memory_capabilities` | 1 | contract + ABI 静态检查 |
| 3 | `native/runtime/plan.cpp` | 统一 candidate enumeration、stage floor、确定性 digest；未知 upper 直接 reject | 2 | planner golden |
| 4 | `native/models/ltx_runtime/ltx_gpu.h/.m` | hooks/options、buffer token、actual capacity、completion-safe release | 1 | fake allocator + native build |
| 5 | `native/models/ltx_runtime/ltx_blocks.c` | probe-before-load、每槽 reservation、generation/epoch 校验、无第三槽 | 3/4 | pager fake + short-read/cancel |
| 6 | `native/platform/apple/ltx_session.mm` | 将 policy scope 传入 options；stage transition 释放/重建；结果附 ledger/pager metrics | 5 | LTX L2 |
| 7 | `native/models/h3_runtime/h3_gpu.h/.m` | hooks、tensor lease、mmap alias 计账、completion release | 1 | H3 allocator unit |
| 8 | `native/models/h3_runtime/h3.c` | 将 denoiser sub-budget/pinned/slots 与 scope 对接；首尾 stage checkpoint | 3/7 | H3 L2 |
| 9 | `native/platform/apple/h3_session.mm` | 传递 generation、scope、streaming metrics；禁用未认证 cache reuse | 8 | H3 L2/L3 |
| 10 | `native/backends/mlx.*` | `ScopedMlxLimits`、graph manifest、cache delta envelope；默认路径 no-op | 3 | MLX fake/qualified graph |
| 11 | `native/api/c_api.mm` | 抽取 `prepare_execution`，统一 generate/prepare/error metrics；保证异常恢复 | 2/3/6/8/10 | C API contract |
| 12 | `services/turbociderd/service.mm` | 依据 effective plan 路由；queued/waiting/admitted 状态；child envelope | 11 | service integration |
| 13 | `tools/native/benchmark_memory_constrained.py` | campaign、monitor、swap delta、质量 hash、污染状态 | 11/12 | benchmark smoke |
| 14 | `tests/native/*` | L0-L4 fixture、minimum-1/minimum/baseline、ABBA、故障注入 | 全部 | release gate |

实现提交应按表中顺序拆分，避免 allocator、planner、service 同时变更导致无法定位越界来源。

## 63. 测试夹具与故障注入细节

### 63.1 fake allocator

fake allocator 不应真的申请 GiB 内存，而应维护：

```cpp
struct FakeBacking {
    uint64_t handle;
    uint64_t capacity;
    uint64_t generation;
    bool gpu_inflight;
    bool cpu_inflight;
};
```

可注入行为：分配失败、actual>upper、重复 release、延迟 completion、旧 generation 回调、短读、EOF、取消和 cache retain。每个注入都要验证 ledger 不泄漏、root scope 可复用、错误码稳定。

### 63.2 planner golden

为 H3/LTX 各准备至少三组不等大 block fixture：

```text
uniform:        48 × 256 MiB
large_tail:     40 × 256 MiB + 8 × 512 MiB
scratch_heavy:  48 × 256 MiB + stage scratch upper
```

golden 内容包括 minimum B、合法 slots/pinned、每阶段 peak、候选排序和拒绝原因。任何 planner revision 改变 golden 必须在 changelog 说明，而不是无提示更新期望文件。

### 63.3 minimum-1/minimum/baseline

每个 candidate 固定三点：

```text
B = certified_minimum - 1       -> must_reject, no large allocation
B = certified_minimum           -> must_succeed or explicitly capability reject
B = baseline                    -> compare quality/performance
```

“minimum” 必须来自同一 checkpoint/shape/precision/runtime。不能用 heuristic minimum 测试真实路径后宣称 hard cap。

### 63.4 memory pressure 注入

本机 macOS 不应通过破坏性地填满系统内存来做常规 CI。推荐两种方式：

1. fake `ProcessMemoryObservation` 返回 green/yellow/red/critical 序列，测试状态机确定性；
2. 专用人工验收机运行受控压力进程，记录 `vm_stat` 单调计数和环境污染状态。

压力注入必须覆盖：预取停止、slot 缩减、stage boundary replan、abort、取消后 cleanup 和下一 job 恢复。

## 64. 实际本机验收脚本建议

建议新增 `tools/native/benchmark_memory_constrained.py`，命令格式如下：

```sh
MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" \
python3 -B tools/native/benchmark_memory_constrained.py \
  --campaign tests/fixtures/memory/h3_c_metal_l3.json \
  --output results/memory/h3-20260915
```

campaign 示例：

```json
{
  "schema_version": 1,
  "model": "minimax-h3-turbo",
  "model_path": "${TURBOCIDER_H3_MODEL}",
  "request": {
    "model": "minimax-h3-turbo",
    "operation": "video.generate",
    "width": 512,
    "height": 512,
    "frames": 25,
    "steps": 6,
    "execution": "gpu",
    "memory_constrained": {
      "enabled": true,
      "limit_bytes": 17179869184,
      "buffer_percent": 15,
      "max_refill_slots": 2
    }
  },
  "budget_points": ["minimum_minus_one", "minimum", "baseline"],
  "rounds": 3,
  "cache_state": ["cold", "warm"],
  "must_succeed": ["minimum", "baseline"],
  "quality": {"reference": "resident_baseline", "max_abs": 0.002},
  "timeout_seconds": 1800
}
```

脚本执行前后记录：

- `uname -a`、macOS build、Metal device name、MLX version、native git commit；
- checkpoint 文件 identity/hash、sidecar/envelope revision；
- Y/X/B、process baseline、system reserve、planned upper；
- ledger committed/active/pending/cached peak、进程 footprint peak、unknown bytes；
- `vm_stat` swapins/swapouts 单调计数增量、`vm.swapusage` 辅助值；
- 输出像素/latent hash、质量指标、wall、I/O wait、overlap ratio；
- `PASS`、`FAIL`、`SKIP`、`CONTAMINATED`、`INCOMPLETE` 终态。

如果 GPU unavailable、checkpoint 缺失或 graph envelope 未认证，脚本必须输出 `SKIP`/`UNQUALIFIED`，不能把安全拒绝计为成功。

## 65. 发布前 DoD 与文档同步规则

### 65.1 H3/LTX 发布门槛

至少同时满足：

- H3 C/Metal 和 LTX C/Metal 各有一条目标 checkpoint 的 L3 `must_succeed`；
- 每个大 buffer allocation 有 reservation/commit/release 证据；
- `unknown_bytes=0` 或明确排除在 capability 之外且请求被拒绝；
- minimum-1 必拒绝、minimum 稳定成功、baseline 质量/性能通过；
- 冷/热、取消、异常、重复请求无 lease 泄漏；
- service 路由使用 effective plan，worker parent-child scope 有证据；
- swap/pressure 证据未污染，默认模式 ABBA wall 回归 ≤2%。

### 65.2 Flux/Z-Image 发布策略

在完成组件拆分、graph envelope 和真实 staged 证据前，Flux/Z-Image 继续返回 `memory_policy_unsupported` 或 `memory_estimate_unknown`。不能因为 resident 估算小于 B 就开放 constrained resident；这会让用户误以为已有逐层 offload。

### 65.3 文档状态更新

只有满足 65.1 后，才把顶部状态改为：

```text
Implemented for certified H3/LTX C/Metal routes; not an OS hard cap.
```

同时保留：

- capability key、checkpoint identity、planner/envelope revision；
- 已认证 shape/slots/operation；
- 未支持模型和拒绝原因；
- 本机验收环境及污染说明。

代码、测试和本文档必须在同一 PR 更新；若实现只完成脚手架或 fake backend，顶部状态仍保持 Proposal + scaffolding，不能提前改成 Implemented。

## 66. 基于当前工作树的实施规格

本节是后续实现的“单一施工图”。它不改变前文的产品结论，只把已经存在的
脚手架、下一步代码修改和验收证据连接起来。实现者应优先完成语义闭环，再做
性能优化；任何优化都不能绕过 reserve、completion-safe release 或 fail-closed。

### 66.1 当前代码与目标代码的边界

当前工作树可按下面四层理解：

```text
配置/计划层       MemoryConstrainedConfig + EffectiveMemoryPolicy
                  已可解析、校验、计算 B、规范化 H3/LTX candidate

准入/账本层       MemoryAdmission + MemoryLedger
                  已有 reservation/lease、unique backing、footprint checkpoint

原生分配层        LTX C/Metal buffer hooks
                  已开始 reserve -> MTLBuffer -> commit -> deferred release

完整执行闭环      H3 tensor + MLX graph/VAE/cache + host/media + stage planner
                  尚未完成，不能把现状描述成 OS hard cap
```

因此下一阶段的成功定义不是“再增加一个字段”，而是让同一条请求在以下四个
层面拥有同一个 `policy_digest`、同一个 `generation`、同一个错误生命周期：

```text
request -> plan -> admission -> allocator/pager -> stage scheduler -> result
```

### 66.2 建议新增的运行时对象

`MemoryConstrainedConfig` 保持为解析层对象，不把运行时 token 塞回 JSON 配置。
建议在 `native/runtime/memory_policy.hpp` 增加以下只读运行时结构：

```cpp
struct MemoryPolicyRuntime {
    uint64_t effective_budget_bytes;       // B = Y * (100-X) / 100
    uint64_t process_baseline_bytes;       // admission 时 F0
    uint64_t planned_increment_bytes;      // E
    uint64_t allocation_ceiling_bytes;     // F0 + E，过渡实现
    uint64_t system_reserve_bytes;
    uint64_t safety_margin_bytes;
    uint32_t planner_revision;
    uint32_t schedule_revision;
    std::string policy_digest;
};
```

`ExecutionPlan` 应增加（或等价地保留）以下字段：

```cpp
Request requested_request;                 // 用户原始请求，便于审计
Request request;                           // normalize 后的 effective request
std::string candidate_id;                  // 稳定能力 key
std::string planner_revision;
std::string estimate_revision;
std::optional<MemoryPolicyRuntime> memory_runtime;
```

约束如下：

1. `requested_request` 只读保存原始字段，不能被 session 再次修改。
2. `request` 是所有执行入口唯一使用的规范化请求；服务、CLI、worker 和 C API
   不得自行根据环境变量重算 residency 或 slots。
3. `candidate_id` 必须包含模型、backend、precision、operation 和认证的槽位集合，
   例如 `ltx_c_metal_video_streamed_v1_slots2`。
4. `planner_revision` 与 `estimate_revision` 进入结果 JSON 和 benchmark 文件名，
   防止新旧 envelope 被误混在同一张图中。

### 66.3 账本不变量和指标语义

当前过渡实现应明确采用以下不变量：

```text
effective_request_budget_bytes = B
allocation_ceiling_bytes       = F0 + E
ledger.committed_bytes         = F0 + unique known backing
ledger.reserved_bytes          = 尚未 commit 的 child upper bound
unattributed_bytes             = max(observed_footprint - ledger known, 0)
```

其中 `F0` 不能被模型代码重复 reservation。`MemoryAdmission` 构造时采集一次
进程 baseline；之后模型 allocator 只提交新增 backing。`StorageId` 去重必须包含
`allocator_domain + handle + generation`，不能只比较裸地址。

结果字段建议保持向后兼容，同时新增：

```json
{
  "memory": {
    "effective_request_budget_bytes": 14602888806,
    "process_baseline_bytes": 3219128320,
    "planned_increment_bytes": 11383760486,
    "allocation_ceiling_bytes": 14602888806,
    "ledger_known_storage_peak_bytes": 10928312320,
    "unattributed_footprint_peak_bytes": 183500800,
    "unknown_bytes": 0,
    "observed_within_budget": true,
    "enforcement_scope": "ltx_native_metal_buffers+stage_boundaries"
  }
}
```

上例中的 `allocation_ceiling_bytes` 由 `process_baseline + planned_increment` 得出；
实际输出必须由运行时计算，禁止在 fixture 中复制示例常量。`unknown_bytes > 0` 时，若该值来自
必需路径，candidate 必须降级为 `memory_estimate_unknown`；只有明确排除且不影响
准入的诊断对象才可以出现在非零 unknown 中。

### 66.4 `PreparedExecution` 的统一入口

不要扩张当前仅负责 GPU 锁的 `native/runtime/execution.hpp`。建议新建
`native/runtime/prepared_execution.hpp/.cpp` 实现唯一准备入口。伪代码如下：

```cpp
PreparedExecution prepare_execution(
    tc_engine& engine, const Request& requested, ExecutionIntent intent) {
    validate_request_schema(requested);
    auto plan = make_plan(requested);                 // no allocation
    if (plan.memory_policy) {
        require(plan.memory_policy->route_available,
                plan.memory_policy->reason);
        probe_checkpoint_metadata(plan);             // header/index only
        choose_memory_candidate(plan);               // deterministic
        require_memory_constrained_execution_supported(*plan.memory_policy);
    }

    auto admission = prepare_memory_admission(engine.session, plan,
                                               /*parent_mlx=*/false);
    bind_effective_request(engine.session, plan.request);
    return PreparedExecution{std::move(plan.request), std::move(plan),
                             std::move(admission)};
}
```

实现要求：

- `make_plan`、metadata probe、candidate 选择必须无副作用；不能先创建完整模型
  再检查预算。
- `prepare_memory_admission` 失败时，不得调用旧的 resident/unbounded 路径。
- `generate` 和 `prepare` 只调用 `prepare_execution`，服务层只消费 plan JSON；
  不能保留第二份准入逻辑。
- `MemoryAdmission`、session binding、scoped MLX limits、pager worker 必须由
  RAII 对象持有，异常、取消和 `longjmp` 之外的所有 C++ 退出路径都能清理。

## 67. 调度状态机、锁顺序与事件合同

### 67.1 作业状态机

受限作业的持久化状态必须区分“等待 GPU 锁”和“等待内存准入”：

```text
queued
  -> planning
  -> waiting_for_gpu
  -> waiting_for_memory
  -> admitted
  -> loading
  -> running
  -> draining
  -> succeeded

任何阶段 -> failed / cancelled / interrupted
```

状态语义：

- `planning`：只读 checkpoint header/index、计算候选和 envelope。
- `waiting_for_gpu`：同 GPU generation 的互斥锁未获得；不持有 child reservation。
- `waiting_for_memory`：系统 reserve、process baseline 或 scope 容量暂不可用；
  允许退避重试，但不能无限重试一个静态 `memory_budget_too_small`。
- `admitted`：B、system reserve、candidate capability 都已通过；可以创建模型对象。
- `loading`：允许 weights、slot、graph compile temporary 的 bounded allocation。
- `running`：只允许计划中的阶段和 block reservation。
- `draining`：停止新 prefetch，等待 GPU/IO/convert queue，完成 pending release。

服务恢复历史 job 时，若旧记录没有 `service_route` 或 `candidate_id`，必须重新
planning；不能依据旧 request 的原始 `residency` 直接恢复 worker 拓扑。

### 67.2 锁顺序

为避免 allocator callback 与服务线程死锁，固定顺序为：

```text
service job mutex
  -> execution/GPU mutex
    -> MemoryLedger mutex
      -> pager slot mutex
        -> IO queue condition variable
```

禁止的反向顺序：

```text
pager slot mutex -> GPU mutex
MemoryLedger mutex -> wait GPU command
allocator callback -> file IO / mx::synchronize
```

`reserve/commit/cancel/release` callback 必须是短临界区，不做 GPU wait、文件读取、
Objective-C autorelease pool drain 或 MLX eval。需要等待时先把 token 标记为
`retiring`，释放 ledger mutex，再在外层等待 completion。

### 67.3 统一事件格式

`native/runtime/schedule_trace.*` 的事件至少包含：

```json
{
  "ts_ns": 123456789,
  "job_id": "...",
  "policy_digest": "...",
  "schedule_revision": 4,
  "stage": "denoise",
  "block": 17,
  "event": "reserve|load_begin|load_end|gpu_begin|gpu_end|retire|release",
  "memory_class": "weights",
  "bytes": 268435456,
  "slot": 1,
  "watermark": "green",
  "ledger_committed": 9876543210,
  "ledger_reserved": 268435456,
  "process_footprint": 11223344556
}
```

事件记录失败不能改变执行结果；trace buffer 满时丢弃低优先级 timing 事件，但
`reserve denied`、`observed over budget`、`cancel`、`lifetime violation` 必须保留。

## 68. 确定性 planner 与运行时 schedule

### 68.1 候选枚举顺序

对相同质量合同，候选按以下固定顺序生成，保证不同入口得到相同 plan：

```text
resident
component_staged
streamed(slot=1)
streamed(slot=2)
streamed(slot=3)
component_staged + certified_tiling
streamed(slot=N) + certified_tiling
```

候选过滤顺序固定为：

1. schema/model/backend/precision 合法性；
2. checkpoint identity 与 metadata probe；
3. quality contract（不能偷偷改变尺寸、采样步数、全局 attention 语义）；
4. stage floor 和全请求 upper-bound；
5. allocator/graph/pager capability；
6. `B` 与 `min_free_bytes` admission；
7. 性能代价排序。

任何一个必须的 upper bound 未知，候选直接进入 `memory_estimate_unknown`，不能
用下一候选“试运行看看”。

### 68.2 峰值计算

对每个 stage 建立 live interval：

```text
resource = {id, class, lower, upper, first_use, last_use, reuse_policy}
```

阶段峰值使用区间扫描，而不是把所有资源简单相加：

```text
peak(stage) = max_t sum(upper(resource) where first_use <= t < last_use)
M_plan      = max_stage peak(stage) + cross_stage_overlap
```

`cross_stage_overlap` 只在 planner 明确允许下一阶段 preload 时出现。默认情况
Stage1→VAE 采用 strict handoff：旧阶段 GPU event 完成、旧 scope 释放、checkpoint
通过后才创建新阶段权重。这样牺牲一部分 overlap，换取确定的峰值边界。

### 68.3 planner 伪代码

```cpp
for (auto candidate : enumerate_candidates(model, request)) {
    if (!candidate.capability_verified) continue;
    auto resources = probe_and_build_resources(candidate, shape);
    if (resources.has_unknown_required_upper()) {
        reject(candidate, "memory_estimate_unknown");
        continue;
    }
    auto peak = interval_peak(resources);
    if (peak > policy.effective_budget_bytes) {
        reject(candidate, "memory_budget_too_small");
        continue;
    }
    if (!system_admission_possible(peak, policy.min_free_bytes)) {
        reject(candidate, "memory_admission_conflict");
        continue;
    }
    candidate.score = estimate_wall_time(candidate, resources)
                    + 1000.0 * estimate_unhidden_io_wait(candidate);
    feasible.push_back(candidate);
}
if (feasible.empty()) fail_closed(best_rejection_reason());
return stable_sort_by_score_then_candidate_id(feasible).front();
```

### 68.4 运行中重规划

运行中只允许在安全点改变 `pinned_prefix`、`lookahead`、`refill_slots`：

- denoise step 完成且当前 command buffer 全部完成；
- block boundary 且没有 reader/convert writer；
- Stage1、upsample、Stage2、VAE tile 之间的显式 transition。

禁止在 attention kernel 中途、VAE halo 写入中途或正在使用的 MTLBuffer 上切换
slot。重规划只允许向更保守方向变化；若保守候选也不 fit，返回
`memory_pressure_abort`。

## 69. LTX 的落地顺序与代码修改建议

### 69.1 当前已完成的 LTX 部分

当前工作树已经完成：

- `ltx_gpu_memory_hooks` C ABI 及旧 ABI 兼容入口；
- `ltx_gpu_buffer_new_classified` / copy classified；
- reserve → Metal allocation → commit；失败时 cancel；
- retain 不重复释放；deferred command batch 在安全点 release；
- native denoiser 与 Gemma GPU context 安装 hooks；
- 受限 LTX 默认走 `memory_constrained_native_session`，不使用未计账 helper。

这些改动证明了“集中 Metal buffer 入口可接入 ledger”，但不等于 LTX 全链路已
受控。下一步必须按下面顺序推进。

### 69.2 `ltx_native_probe_block_layout` 必须先于 block 0 加载

当前 `ltx_native_create()` 仍存在“先创建部分对象，再从 block 0 推导布局”的
历史路径。应增加：

```c
int ltx_native_probe_block_layout(
    const char *checkpoint,
    ltx_block_layout *out,
    char *error, size_t error_size);
```

probe 只允许执行 `open/fstat/pread(header/index)`，不允许：

- 创建 `MTLBuffer`；
- 创建 MPSGraph；
- 读取完整权重 payload；
- 修改全局 MLX cache。

正式 load 再次 `fstat`，并比较 `checkpoint_identity_sha256`、文件大小、mtime/size
或更强的 inode+hash 组合；不一致返回 `weight_io_error`。不等大的 block 必须按
窗口最大值规划：

```text
window_bytes(i, slots) = max(runtime_bytes[i .. i+slots-1])
slot_upper            = max(payload_bytes, runtime_bytes, conversion_bytes)
```

### 69.3 LTX stage reservation

在 `native/platform/apple/ltx_session.mm` 建立显式 stage scope：

```cpp
struct LtxStageLease {
    std::optional<MemoryReservation> conditioning;
    std::optional<MemoryReservation> latent;
    std::optional<MemoryReservation> output;
    std::optional<MemoryReservation> graph_workspace;
    std::optional<MemoryReservation> vae_workspace;
};
```

推荐顺序：

```text
text encoder
  -> commit conditioning, release Gemma weights if not reusable
denoise stage1
  -> reserve latent/output/workspace + block pager
upsample transition
  -> wait stage1 GPU event; release stage1-only buffers
denoise stage2
  -> reserve stage2-specific block/workspace
VAE/decode
  -> stop pager; wait transformer; release denoiser scope
  -> reserve VAE graph/arena; decode tiles
export
  -> reserve bounded output/media queue; flush and release
```

受限模式下 `ltx_mlx_video_vae_create`、`ltx_mlx_upsampler_create` 前必须有
metadata/envelope reservation；无法得到上界时 fail-closed。`std::vector`、RGB
转换、mp4 muxer queue 也要么纳入 `Output/ConversionScratch`，要么建立经过校准
的 coarse envelope，不能默认为零。

### 69.4 MLX denoiser 分支

LTX MLX 分支暂时仍属于 `memory_estimate_unknown`，除非同时满足：

1. BlockCache 的每个 active/refill tensor 有可识别 backing 和 release event；
2. MPSGraph temporary 有已认证 `(runtime, device, shape, revision)` envelope；
3. `mx::set_cache_limit`、`mx::clear_cache` 的前后 delta 已计入 K；
4. `force_eval_each_block` 和 graph cache key 固定在 candidate 中；
5. 真实冷/热两组 L3 运行通过。

在这些条件完成前，不能因为 `mlx_cache_capacity < 48` 就把 MLX 路线标成
`allocation_guarded`。

## 70. H3 的落地顺序与特殊语义

### 70.1 三类 allocation 必须分开

H3 的 allocator 接入不能照搬 LTX 的“单一 MTLBuffer”假设：

| 资源 | 典型入口 | 账本类别 | 关键风险 |
|---|---|---|---|
| activation/output | `h3_gpu_tensor_new`、F32 wrap | Activation/Output | shape 对齐和临时 pack |
| streamed weight slot | `h3_gpu_tensor_load_file`、`h3_gpu_tensor_map_bf16` | Weights/RefillSlot | pread、短读、completion release |
| mmap/no-copy alias | `newBufferWithBytesNoCopy` | Weights + mmap composite | clean page 与 Metal backing 是否复制 |

H3 wrapper 现有双槽约束必须保留：

```text
certified_slots = {2}
request.max_refill_slots >= 2 只表示上限足够
planner 不得把 slots=1 当成 H3 双槽已实现
```

### 70.2 H3 hooks 设计

在 `native/models/h3_runtime/h3_gpu.h/.m` 增加与 LTX 等价但独立命名空间的
`h3_gpu_memory_hooks`，旧 `h3_gpu_create()` 委托到 `h3_gpu_create_with_options()`。
每个 `H3Tensor` 保存：

```c
void *memory_token;
uint64_t allocator_domain;
uint64_t generation;
uint64_t handle;
uint32_t references;
uint32_t gpu_inflight;
uint8_t read_only;
uint8_t accounted;
```

`h3_gpu_tensor_map_bf16()` 需要返回 composite identity：如果后续证据证明 Metal
只持有 mmap clean pages 的 alias，可在 manifest 中写 `mmap_aliasing=validated`，
并取消重复计费；在证据出现前，planner 按保守上界计入两者。

### 70.3 H3 pipeline 改造顺序

```text
probe shard index
  -> reserve pinned prefix
  -> reserve exactly two refill slots
  -> stream block i
  -> mark GPU use and attach completion
  -> retire/reuse slot
  -> final step release suffix/prefix as last-use permits
```

`h3_gpu_continue_releasing()` 只能推进状态，不能直接释放仍有
`gpu_inflight > 0` 的 tensor。`h3_session.mm` 需要把 `generation`、candidate、
scope 和 streaming metrics 传入 C runtime，并在异常时显式 drain。

H3 当前 MLX/VDN whole-map 路线不能复用 C/Metal 的认证结果；在 checkpoint
pager 化和 whole-map backing 可观测前，受限请求必须返回
`memory_policy_unsupported` 或 `memory_estimate_unknown`。

## 71. Flux、Z-Image 与后续模型的适配门禁

### 71.1 Flux

Flux 的第一阶段不应直接做“任意层逐层换入”。建议先按已知执行拓扑划分：

```text
text/conditioning -> transformer single blocks -> transformer dual blocks
                   -> latent/output -> VAE
```

第一版只认证：

- 单一 GPU backend；
- 固定 shape bucket；
- 固定 precision/dequant 格式；
- single/dual block 的完整 metadata index；
- VAE 与 transformer strict handoff；
- 已测 graph workspace envelope。

若 dual block 的输入需要同时持有两组权重，planner 必须以两组实际 upper 计算，
不能用 single block 平均值。未完成 block pager 前，Flux constrained 只能使用
`component_staged` 且必须有真实阶段释放证据；不能开放 streamed。

### 71.2 Z-Image

Z-Image 需要区分 main transformer、refiner、VAE 和 GGUF 类型：

```text
main transformer -> latent handoff -> refiner（可选） -> VAE
```

`refiner` 未启用时不应预留其权重；启用时必须把 latent 保留期间的重叠计入
`cross_stage_overlap`。GGUF dequant、pack 和 CPU staging 统一归入
`ConversionScratch`，并且每种 quant type 建立独立 envelope。只要某一 quant
type 没有 upper，整个 constrained candidate 拒绝，而不是退回 resident。

### 71.3 能力矩阵

模型 capability 必须由代码返回，不由文档或 model name 推断：

```json
{
  "model": "flux",
  "backend": "mlx",
  "operation": "image.generate",
  "capabilities": {
    "resident": true,
    "component_staged": false,
    "streamed": false,
    "quality_preserving_tiling": false,
    "graph_envelope": "missing"
  }
}
```

只有 capability、checkpoint identity、shape bucket、runtime/device 版本全部匹配
时，planner 才允许 `execution_supported=true`。

## 72. MLX、MPSGraph、cache 与 host memory 的受控边界

### 72.1 `ScopedMlxLimits` 的最小实现

`native/backends/mlx.*` 建议提供：

```cpp
class ScopedMlxLimits {
public:
    static ScopedMlxLimits install(const MlxEnvelope&, const std::string& digest);
    void drain_and_restore() noexcept;
    ~ScopedMlxLimits() noexcept;
private:
    uint64_t old_memory_limit_ = 0;
    uint64_t old_cache_limit_ = 0;
    bool active_ = false;
};
```

实现顺序：

```text
capture old limits
 -> set scoped graph/cache guideline
 -> run stage
 -> synchronize
 -> release graph refs
 -> trim optional cache
 -> restore old limits
```

该对象只能在 execution mutex 持有线程构造/析构。add-on 关闭时不构造它，
不调用 `mx::clear_cache()`，不改变默认 cache key。

### 72.2 未归因内存处理

每个 checkpoint 计算：

```text
unattributed = max(process_footprint - ledger_known_storage, 0)
```

如果 unattributed 超过 `safety_margin_bytes` 连续两个安全点：

1. 停止新 graph compile 和可选 prefetch；
2. trim 可选 cache；
3. 在 stage boundary 重新采样；
4. 仍超出则 `memory_pressure_abort`。

不能用 `mx::get_active_memory()` 单独替代进程 footprint，也不能把一次成功
的 cache trim 当作之后所有 shape 的硬上界。

### 72.3 host/staging/output

下列对象必须有明确归属：

- `std::vector<uint16_t>` prompt/audio/video staging：Conditioning 或 ConversionScratch；
- latent 和 RGB frame arena：Activation/Output；
- ffmpeg/mp4 muxer queue：Output；
- checkpoint `pread` 临时页：RefillSlot 或 ConversionScratch；
- Python/Objective-C bridge copy：ChildProcessEnvelope 或 UnknownExternal。

如果对象不能在创建前给出上界，受限路径拒绝；不能通过把它留在 host 就认为
脱离 GPU 预算。

## 73. 服务层、C API 与持久化改造

### 73.1 plan JSON 最小合同

`results.mm` 和服务层使用以下字段，不读取内部 C++ 指针：

```json
{
  "memory_policy": {
    "enabled": true,
    "user_limit_bytes": 17179869184,
    "effective_budget_bytes": 14602888806,
    "buffer_percent": 15,
    "effective_residency": "streamed",
    "adapter_candidate": "ltx_c_metal_video_streamed_v1_slots2",
    "planner_revision": "planner-v3",
    "estimate_revision": "ltx-layout-v2",
    "execution_supported": false,
    "admission_state": "planning",
    "reason": "memory_estimate_unknown"
  }
}
```

`execution_supported=false` 时必须附稳定 `reason`；服务不应把它解释为普通
resident candidate。`enabled=false` 时，plan 中可省略 runtime-only 字段，保证
默认模式 JSON 和 route 不出现额外行为变化。

### 73.2 route 与 job 持久化

服务层路由依据 `memory_policy.effective_residency`、`adapter_candidate`、
`worker_scope`，而不是原始 `request.residency`。job 表至少持久化：

```text
policy_digest
candidate_id
planner_revision
service_route
checkpoint_identity
admission_state
terminal_reason
```

恢复时 digest 或 checkpoint identity 不匹配，作业回到 `planning`，不能直接
复用旧 pager/worker。

### 73.3 统一错误 JSON

C API 仍可保持 `char** error` ABI，但错误字符串应由内部结构化对象生成：

```json
{
  "code": "memory_budget_too_small",
  "phase": "denoise_stage1",
  "message": "required upper bound exceeds effective budget",
  "effective_budget_bytes": 14602888806,
  "required_upper_bytes": 15100000000,
  "ledger_peak_bytes": 10300000000,
  "policy_digest": "...",
  "retryable": false
}
```

客户端仍可先按 code 前缀兼容；新工具优先解析 JSON。错误中禁止写“自动切换
到 swap/resident”之类暗示性文案。

## 74. 测试与本机验收的实施矩阵

### 74.1 L0/L1：无 GPU 也必须通过

L0 单测至少覆盖：

- Y/X 边界、整数溢出、bool/float/NaN/负数拒绝；
- profile overlay 与 request `enabled=false` 覆盖；
- 旧 `memory_budget_bytes` 冲突；
- candidate deterministic digest；
- uniform/large-tail/scratch-heavy block planner golden；
- ledger reserve/commit/cancel/release/retire/cache 去重；
- `actual_bytes > upper_bytes`、重复 release、旧 generation、pending completion；
- service route 只看 effective plan。

L1 fake backend 必须不申请真实 GiB，只模拟容量和延迟，覆盖：

```text
allocation fail
short read / EOF / EINTR
delayed GPU completion
cancel during IO
cancel during graph eval
unknown envelope
pressure green -> yellow -> red -> critical
```

每个 case 断言：`storage_count`、`pending_release_count`、`reservation_count` 最终
回到零（若有显式跨请求 cache，则按预期 cached bytes 断言），且错误码稳定。

### 74.2 L2：真实 allocator 和短流程

有 Metal device 时运行：

```sh
MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" \
TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh

python3 -B tests/native/test_memory_accounting.py
python3 -B tests/native/test_ltx_gpu_memory_hooks.py
python3 -B tests/native/test_contract.py
python3 -B tests/native/test_h3_streaming_policy.py
python3 -B tests/native/test_ltx_streaming_benchmark.py
```

无 Metal device 时，allocator integration 测试只能输出 `SKIP`，不能伪造 PASS。

### 74.3 L3：真实模型 must-succeed

每条认证路线至少执行：

```text
minimum-1 byte  -> must_reject，禁止创建第一个大 backing
minimum         -> must_succeed，完成完整输出
baseline        -> 与 add-on disabled 做 ABBA
```

每个预算点包括 cold cache、warm cache、最小 shape、默认 shape 和一组大 shape；
至少三轮。必须保存：

- checkpoint hash、Metal device、OS/MLX/native commit；
- Y/X/B、F0、E、allocation ceiling；
- ledger known peak、unknown/unattributed peak、process footprint peak；
- pager slot/pinned/refill、I/O bytes、GPU starvation、overlap ratio；
- `vm_stat` swapins/swapouts 单调计数 delta；
- 输出 hash、质量指标、wall/denoise/load 时间；
- `PASS/FAIL/SKIP/CONTAMINATED/INCOMPLETE`。

### 74.4 通过/失败规则

通过必须同时满足：

1. minimum-1 在大分配前拒绝；
2. minimum 全流程完成且输出质量满足合同；
3. `unknown_required_bytes == 0`；
4. 进程 footprint checkpoint 不超过 B；
5. 安静环境下 swapouts 增量为零，swapins 只报告并调查；
6. 取消、异常、重复请求后没有 lease 泄漏；
7. 默认模式 ABBA wall 回归不超过 2%，且 route/cache/compile key 不变化。

任一安全条件失败，即使 wall time 达标也标记 FAIL。外部压力或已有 swap 无法
隔离时标记 CONTAMINATED，保留样本并按计划重测，不能删掉失败轮次。

## 75. 分阶段发布、回退与 DoD

### 75.1 开关与能力分级

建议分三档 feature flag：

```text
TC_MEMORY_CONSTRAINED_PLAN_ONLY=1
  只解析/规划/输出，不执行受限模型

TC_MEMORY_CONSTRAINED_LTX_C_METAL=1
  仅打开已认证 LTX C/Metal candidate

TC_MEMORY_CONSTRAINED_H3_C_METAL=1
  仅在 H3 allocator + L3 证据完成后打开
```

生产默认均为关闭。模型 capability 以 manifest/代码返回为准；环境变量只能
缩小能力集合，不能把未认证模型强行打开。

### 75.2 回退原则

受限执行失败时只允许：

- 同一 policy 下更保守的已认证 candidate；
- 直接 fail-closed 并返回稳定错误。

不允许自动回退到 resident/unbounded/swap。用户若要普通模式，必须显式关闭
`memory_constrained.enabled` 并重新提交请求，防止错误路径悄悄改变语义。

### 75.3 发布 DoD

H3/LTX 各自发布前必须具备：

```text
[ ] metadata probe-before-load
[ ] 所有必需大 allocation 有 reserve/commit/release
[ ] completion-safe retire，取消无悬挂回调
[ ] stage reservation + strict handoff
[ ] unknown bytes 为零或被 capability 拒绝
[ ] minimum-1/minimum/baseline 证据
[ ] cold/warm + pressure + cancel + failure campaign
[ ] service effective-plan route 与 job persistence
[ ] default ABBA 零行为变化
[ ] docs 状态、capability key、revision、证据路径同步
```

Flux/Z-Image 在 graph envelope、组件拆分、真实 staged/streamed 证据完成前，
继续返回 `memory_policy_unsupported` 或 `memory_estimate_unknown`。顶部状态只有
在 H3/LTX 目标路线达到 L3 后才能改成“Implemented for certified routes; not an
OS hard cap”。

## 76. 逐文件 patch 级任务单

下面的任务单以当前工作树为基线；“状态”栏用来区分已落地、部分落地和拟增，
实现 PR 可以直接按编号拆分。每个编号都应有独立测试或可审阅的证据，避免一次
提交同时改 allocator、planner 和服务路由而难以定位问题。

| 编号 | 文件 | 修改内容 | 当前状态 | 完成证据 |
|---:|---|---|---|---|
| 76-01 | `native/runtime/memory_accounting.hpp/.cpp` | 保留当前 baseline + allocation ceiling 过渡模型；新增 `allocation_ceiling_bytes`、`unattributed_peak_bytes`、`unknown_required_bytes` 指标；为正式版预留 `MemoryBudgetScope` | 部分落地 | root/child 不双计费单测；字段在 success/error/cancel 均序列化 |
| 76-02 | `native/runtime/session.hpp` | `ExecutionPlan` 保存 requested/effective request、candidate、planner/estimate revision；`RunResult` 保留 plan 快照 | 拟增 | ABI/contract test 验证原字段布局和 JSON 字段 |
| 76-03 | `native/runtime/prepared_execution.hpp/.cpp` | 实现统一 `prepare_execution()`，集中做 plan、probe、admission、session binding 和 scoped cleanup | 拟增 | C API、CLI、service 三入口共享同一 digest |
| 76-04 | `native/runtime/plan.cpp` | candidate 枚举、metadata probe、interval peak、确定性 tie-break；把旧固定 reserve 改为 stage-derived | 部分落地 | planner golden；minimum-1/minimum 结果稳定 |
| 76-05 | `native/runtime/residency_planner.*` | 合并组件级和 block 级计划，输出 `StageLiveSet`、slot/pinned、cross-stage overlap | 拟增 | uniform/large-tail/scratch-heavy golden |
| 76-06 | `native/runtime/weight_pager.*` | 统一 `pread`/mmap source、slot 状态机、短读/EINTR、generation、completion ticket | 拟增 | fake pager + delayed completion + cancel |
| 76-07 | `native/models/ltx_runtime/ltx_gpu.h/.m` | 已完成 hooks、classified buffer、deferred release；补充每个 buffer 的 in-flight 计数和 completion callback | 部分落地 | `test_ltx_gpu_memory_hooks.py`；多 queue stress |
| 76-08 | `native/models/ltx_runtime/ltx_native.h/.c` | 增加 `ltx_native_probe_block_layout`；正式 load 校验 header identity；每槽接入 reservation | 拟增 | probe 不创建 GPU 对象；identity race 失败 |
| 76-09 | `native/models/ltx_runtime/ltx_blocks.c` | block payload/runtime bytes 分离；refill slot 为 `RefillSlot`；无第四槽；last-use 后 retire | 部分落地 | slot ring fake + native streaming info |
| 76-10 | `native/platform/apple/ltx_session.mm` | stage1/upsample/stage2/VAE stage lease；VAE/输出/host staging envelope；取消 drain | 部分落地 | LTX L2/L3 trace；cancel 后 storage_count 为 0 |
| 76-11 | `native/models/ltx_runtime/ltx_mlx_video_vae.cpp`、`ltx_mlx_upsampler.cpp` | 提供 metadata-only estimate、ScopedMlxLimits 接口、cache delta 观测；未知 graph fail-closed | 拟增 | graph manifest + cold/warm envelope |
| 76-12 | `native/models/h3_runtime/h3_gpu.h/.m` | 新增 `h3_gpu_memory_hooks`；tensor token、mmap alias、completion-safe release | 部分落地 | H3 allocator fake/native 单测 |
| 76-13 | `native/models/h3_runtime/h3_dit.c`、`h3.c` | 把 `ssd_memory_budget_bytes` 改为 denoiser child budget；双槽固定、动态 pinned 只能向保守方向变化；slot tensor 使用 `REFILL_SLOT` 分类和稳定 tag | 部分落地 | H3 short run + minimum boundary |
| 76-14 | `native/platform/apple/h3_session.mm` | 传入 allocator domain/generation；text→denoise→VAE stage checkpoint；禁用未认证 whole-map reuse | 部分落地 | H3 route/metrics contract |
| 76-15 | `native/backends/mlx.*` | `ScopedMlxLimits`、cache trim/restore、graph manifest lookup；默认模式 no-op | 拟增 | ABBA cache/limit 恢复 |
| 76-16 | `native/api/c_api.mm` | 改用 `PreparedExecution`；统一 error JSON；异常/取消 RAII；保存 requested/effective plan | 部分落地 | C API contract + failure injection |
| 76-17 | `services/turbociderd/service.mm` | 已按 effective plan 路由；补 queued/planning/waiting/admitted 状态和 admission retry 分类 | 部分落地 | service integration；旧 job recovery |
| 76-18 | `native/platform/apple/results.mm` | 输出 allocation ceiling、unknown/unattributed、schedule trace 摘要、terminal reason | 部分落地 | JSON schema fixture |
| 76-19 | `tools/native/benchmark_memory_constrained.py` | campaign runner、vm_stat 单调计数、污染判断、质量/wall/overlap 汇总 | 拟增 | smoke campaign；退出码矩阵 |
| 76-20 | `tests/native/*`、`Makefile` | L0-L3、fake allocator、planner golden、ABBA、minimum-1/minimum、cancel/EOF/pressure fixture | 部分落地 | CI 输出 PASS/FAIL/SKIP/CONTAMINATED |
| 76-21 | `native/models/h3_runtime/h3_video_vae.h/.c`、`h3.c` | `_with_options` API、旧 API `NULL` 委托、resident/chunked/single-shot options 传播 | 已落地，待分类收尾 | contract + native build；所有 VAE required GPU allocation 不再是 unknown |
| 76-22 | `native/models/h3_runtime/h3.h`、`h3.c`、`h3_session.mm` | host hooks、domain/generation、joint latent/RGB8/decoded envelope | 部分落地 | host hook fake/failure injection；actual output commit；required host unknown=0 |
| 76-23 | `native/platform/apple/ltx_session.mm`、`session.hpp` | request-aware `uses_parent_mlx`，受限请求强制 C/Metal，执行时二次 gate | 已落地，待 route 证据 | contract、service route、环境变量冲突测试 |
| 76-24 | `native/runtime/plan.cpp`、`memory_policy.*` | 删除 `route_available && estimate_fits` 自动开放执行；增加 capability/certification 字段并保持无 manifest 时 plan-only | 已落地 hard gate；registry/签名 manifest 仍拟增 | heuristic fixture `execution_supported=false`；签名 manifest fixture 才可 true |

### 76.1 每个 allocation call site 的改造模板

对所有新增或迁移的 GPU/MLX/host allocation，统一采用以下模板；不能只在
调用者外面包一个“估计值”而让底层继续无界分配：

```cpp
auto reservation = admission.try_reserve(
    MemoryClass::Activation, checked_upper, "ltx.latent.stage1");
require(reservation, "memory_budget_too_small: latent.stage1");

auto native = allocate_platform_object(checked_upper);
if (!native) {
    reservation->cancel();
    fail("platform_allocation_failed: latent.stage1");
}

auto lease = reservation->commit(StorageId{
    allocator_domain, next_handle(), actual_capacity(native), generation});
publish_owned_object(std::move(native), std::move(lease));
```

审查时逐个确认：

- `checked_upper` 来自 metadata/shape，不是 `size_t` 乘法溢出的结果；
- `actual_capacity <= checked_upper`；
- wrapper 发布失败先销毁 native object，再 cancel；
- 异步 GPU 使用结束前不 release；
- tag 包含模型、stage、block/tile、dtype，便于 trace 归因。

## 77. 预算实例、资源归属与调度例子

### 77.1 16 GiB/15% buffer 的计算示例

若用户输入 `Y=16 GiB`、`X=15`，则：

```text
Y = 16 * 2^30 = 17,179,869,184
B = floor(Y * 85 / 100) = 14,602,888,806
```

假设 admission 时 `F0=3,219,128,320`，则 concrete allocation ceiling 为：

```text
E = B - F0 = 11,383,760,486
F0 + E = 14,602,888,806
```

`min_free_bytes` 例如 1 GiB 只参与 system admission，不再从 B 中重复扣一次。
如果当前系统可用内存不足以同时覆盖 `E + min_free_bytes`，作业进入
`waiting_for_memory` 或返回 `memory_admission_conflict`；不能消耗 Y-B 的 buffer。

### 77.2 LTX video-only 的示例 live set

以下数值仅用于说明计算方法，不是任何 checkpoint 的认证值：

| 阶段 | 同时存活资源 | 示例上界 |
|---|---|---:|
| text | Gemma GPU weights + prompt staging | 4.0 GiB |
| denoise-1 | conditioning + latent + graph temp + pinned + 2 refill slots | 10.8 GiB |
| upsample | stage1 latent + upsample scratch | 3.2 GiB |
| denoise-2 | conditioning + stage2 latent + pinned + 2 refill slots | 10.5 GiB |
| VAE | latent + VAE graph temp + RGB tile + output queue | 7.4 GiB |

若 text 完成后 Gemma weights 可安全释放，planner 不把 4.0 GiB 与 denoise 峰值
相加；若 conditioning 由 Gemma 输出保留，则只保留 conditioning backing。若
upsample 必须同时保留旧/新 latent，则 3.2 GiB 已包含重叠，不能只算新 latent。

### 77.3 H3 双槽的调度例子

H3 的 slot 数固定为 2，某个窗口的合法顺序是：

```text
slot0: reserve -> pread(block i)   -> convert -> GPU use -> retire
slot1: reserve -> pread(block i+1) -> convert -> READY
GPU 完成 block i 后，slot0 才能复用为 block i+2
```

若 block i+2 上界大于 slot0 容量，必须在 pread 前拒绝或重建更大 slot；不能
临时分配 slot2。pressure yellow 时先停止 slot1 的 look-ahead，再缩短 pinned
prefix；pressure red 时保持当前 block，等待最早 completion。

### 77.4 资源归属表

| 资源 | 归属类别 | 可否跨阶段保留 | 释放条件 |
|---|---|---:|---|
| prompt conditioning | Conditioning | 是，直到最后一个 denoise stage | 最后一次 connector/denoise 使用完成 |
| denoiser block | Weights/RefillSlot | 否，除非 pinned prefix 合同允许 | GPU completion + last-use |
| latent | Activation | 是，直到 VAE 输入完成 | VAE 读取和导出完成 |
| VAE graph temp | CompileTemporary/Activation | 否 | graph synchronize + refs 清空 |
| RGB/PCM output | Output | 是，直到 mux 写完 | muxer ack 或文件关闭 |
| MLX cache | AllocatorCache | 可选 | scoped trim 或会话销毁 |
| mmap clean page | Weights composite | 视 alias 认证 | 文件映射解除且无 Metal alias |

任何“可跨阶段保留”的资源都必须有 `last_use`；“缓存命中”不等于免费，仍需在
ledger 中保留 cached bytes，直到真正 trim/drop。

## 78. 安全、鲁棒性与可诊断性要求

### 78.1 模型元数据是不可信输入

checkpoint header、tensor index、GGUF quant type、block count、shape 和 offset
都必须经过 checked arithmetic：

```text
offset + payload <= file_size
count * stride 不溢出 uint64
rows * cols * dtype_bytes 不溢出 size_t
block_count <= 编译期上限
```

不满足时返回 `weight_io_error` 或 `memory_policy_invalid`，不能让 planner 生成
异常大的 `size_t` 后交给 Metal/MLX。probe 与正式 load 之间必须重复检查文件身份，
防止 checkpoint 被替换造成 TOCTOU。

### 78.2 generation 与 stale callback

每次 constrained execution 分配新的 `memory_generation`。旧 session、旧 worker、
旧 I/O 线程回调到新 admission 时必须被拒绝并记录：

```text
callback_generation != active_generation
  -> no ledger mutation
  -> memory_lifetime_violation
```

这样可以避免取消后旧线程释放新作业同地址 backing。generation 不应复用为裸 PID
或时间戳；采用进程内单调计数并在 worker 边界传递 policy digest。

### 78.3 诊断输出的隐私边界

trace 默认不写入完整 prompt、输入图片路径或用户输出内容；只写 job id、模型
checkpoint identity、stage/block、bytes、timing 和错误码。调试模式如需保存输入，
必须由显式开关开启，并在 benchmark 汇总中脱敏。

### 78.4 可观测性最低要求

每次受限请求必须能回答以下问题：

1. B、F0、E、allocation ceiling 分别是多少？
2. 哪个 allocation/graph/stage 使 planner 拒绝或运行中 abort？
3. 峰值是 ledger known、unattributed 还是 OS pressure？
4. 哪些 block 被 pinned、哪些 slot 被预取、GPU 等待了多少时间？
5. 失败后所有 reservation、pending release、cache 是否清零？

如果结果 JSON 无法回答这些问题，则该路线只能处于 `estimate_only` 或
`envelope_validated`，不能升级为 `allocation_guarded`。

## 79. 实现不变量、状态转移和审查准则

前面的章节描述了模块边界；本节把实现时必须保持的“不变量”写成可逐条审查的
合同。任何 PR 只要破坏其中一条，即使短样本能够跑通，也不能进入受限模式的
下一等级。

### 79.1 四个独立的数字

实现中不得把以下四个数字混用：

```text
Y  = 用户输入的 limit_bytes
B  = floor(Y * (100-X) / 100)，请求的有效预算
F0 = admission 时整个进程的 process footprint baseline
E  = 本次请求允许新增的 concrete allocation ceiling
```

在当前单 GPU 串行模型下，过渡实现采用 `E = B-F0`，账本预算采用
`F0+E=B`。后续若允许同一进程多个独立 scope，也必须让 root scope 为唯一的
物理 ceiling；child scope 只能从 root 的剩余容量借用，不能再次把 `F0` 加入
child。下列表达式应在单元测试中作为不变量断言：

```text
0 < B <= Y
0 <= F0 <= B
0 <= E <= B-F0
ledger.committed = F0 + sum(unique committed backing)
ledger.reserved   = sum(uncommitted upper bounds)
ledger.committed + ledger.reserved <= B
```

`reserved` 是准入承诺，不是已经发生的物理 footprint；但它必须计入后续
reservation 的可用容量，防止两个线程同时通过检查。进程观测值与账本不一致时，
差值只能被称为 `unattributed`，不能偷偷记为“已释放”。

### 79.2 reservation/lease 状态机

每个 allocation token 只能沿着一条合法路径转移：

```text
ABSENT
  -> RESERVED
       -> COMMITTED(active)
       -> CANCELLED
COMMITTED(active)
  -> RETIRING(pending GPU/IO completion)
  -> CACHED(optional reusable backing)
  -> RELEASED
RETIRING
  -> RELEASED
  -> CACHED       （仅显式 retain_as_cache）
```

规则：

1. `commit`、`cancel`、`release` 都必须幂等；第二次调用不得改变计数，调试构建
   可记录 `duplicate_transition`。
2. `commit` 只接受 `actual_bytes <= upper_bytes`，并验证 domain、handle、generation
   非零且与当前 session 匹配。
3. 异步 command 尚未完成时只能 `retire`，不能直接 `release`；completion callback
   必须携带原始 generation 和 token 地址，过期 callback 不得修改新作业账本。
4. `CACHED` 仍占用 ledger storage；只有 `drop_cached` 或 cache trim 成功后才减少
   known bytes。
5. 对象发布失败的清理顺序是“先销毁平台对象，再 cancel reservation”；已
   `commit` 的对象则“先使平台对象不可见，再 release lease”。不得反过来，以免
   allocator 观察到一个仍可被使用的空 backing。

### 79.3 scope 和 generation

`MemoryScope` 建议包含：`policy_digest`、`allocator_domain`、单调
`generation`、父 scope 指针、关闭状态和 trace sink。一个请求最多拥有一个活动
   root scope；阶段 scope 的生命周期嵌套在 root 内：

```text
root(admitted)
  ├─ text
  ├─ denoise.stage1
  ├─ denoise.stage2
  ├─ vae
  └─ export
```

阶段 scope 关闭前必须满足：无 active lease、无 pending completion、无 pager
worker 引用。若取消发生在阶段中间，scope 进入 `draining`，拒绝新 reservation，
等待已有 GPU/IO 事件后再关闭；不要依靠 Objective-C ARC 的最终析构时机猜测
GPU 是否已经停止访问。

### 79.4 代码审查红线

以下模式一律要求修改或明确豁免说明：

- 在 `memory_constrained.enabled` 路径中调用未包 reservation 的
  `newBufferWithLength`、`mx::array`、`malloc`、`std::vector::resize` 或媒体队列
  扩容；
- 以 `device.currentAllocatedSize` 代替分配前的 upper bound；该值只能用于观测；
- `catch (...)` 后继续进入普通 resident 路径；受限请求只能转入已认证保守 candidate
  或返回稳定错误；
- 把 mmap 文件大小、Metal buffer size、MLX logical bytes 机械相加而没有 backing
  identity；
- 依靠 `setPurgeableState:`、`clear_cache()` 或系统 swap 作为成功条件，而没有
  对应的 lease/release 记录；
- 用一个全局静态的“剩余字节数”代替带 generation 的 ledger；这会在取消和重用
  worker 时触发 ABA 错误。

## 80. Planner 输入、输出和 manifest 版本

### 80.1 模型内存 manifest

每个可认证模型应随 checkpoint 或代码生成一份不含用户输入的 manifest。建议
版本化为 `memory-manifest-v1`，最小结构如下：

```json
{
  "schema": "memory-manifest-v1",
  "model": "ltx-2.5-distilled",
  "checkpoint_identity": "sha256:...",
  "backend": "c_metal",
  "precision": "bf16",
  "runtime_revision": "ltx-layout-v2",
  "components": [
    {
      "id": "denoiser.block",
      "count": 48,
      "payload_bytes": {"min": 123, "max": 456},
      "runtime_bytes": {"min": 789, "max": 1000},
      "alignment": 4096,
      "class": "weights",
      "confidence": "exact"
    }
  ],
  "envelopes": {
    "graph_workspace": {"upper_bytes": 2147483648, "confidence": "validated_envelope"},
    "decode_tile": {"upper_bytes": 805306368, "confidence": "validated_envelope"}
  },
  "capabilities": {
    "component_staged": true,
    "streamed_slots": [2, 3],
    "quality_preserving_tiling": ["vae.video.bf16"]
  }
}
```

manifest 中每一个 `upper_bytes` 必须绑定 shape bucket、dtype、Metal device family、
OS/MLX/native revision。`confidence=heuristic` 的条目可以用于普通模式的性能
提示，但不能单独支撑受限准入。manifest 的 parser 要使用 checked arithmetic，
并拒绝未知的必需 component、重复 id、负数、超大 count 和不在文件范围内的
offset。

### 80.2 Planner 输出

`ResidencyPlan` 建议拆成机器可执行部分和解释部分，避免执行器重新推导：

```cpp
struct StageLiveSet {
    std::string stage_id;
    std::vector<ResourceInterval> resources;
    uint64_t upper_bytes = 0;
    uint64_t cross_stage_overlap_bytes = 0;
};

struct BlockWindow {
    uint32_t first_block = 0;
    uint32_t block_count = 0;
    uint32_t pinned_prefix = 0;
    uint32_t refill_slots = 0;
    uint64_t slot_upper_bytes = 0;
};

struct ResidencyPlan {
    std::string candidate_id;
    std::string manifest_revision;
    std::vector<StageLiveSet> stages;
    std::vector<BlockWindow> windows;
    uint64_t peak_upper_bytes = 0;
    uint64_t required_unknown_bytes = 0;
    uint64_t estimated_io_bytes = 0;
    uint64_t estimated_unhidden_wait_ns = 0;
    bool strict_stage_handoff = true;
};
```

执行器只读取 `windows` 和 resource lease 模板，不再根据 `max_refill_slots` 自行
创建更多槽位。解释部分应包括“哪个资源决定峰值”“为什么没有选择更快 candidate”
和“哪些 overlap 被禁止”，便于服务返回可操作的拒绝原因。

### 80.3 确定性 tie-break

相同请求在 CLI、C API 和服务 worker 上必须产生 byte-for-byte 相同的
`candidate_id`、`plan_digest` 和 slot/pinned 选择。排序键固定为：

```text
(quality_contract, peak_upper_bytes, unhidden_wait_ns,
  estimated_io_bytes, candidate_id)
```

禁止用哈希表迭代顺序、当前 wall clock、线程完成先后或随机 seed 参与 planner
选择。planner golden 应覆盖等峰值不同 I/O、等 I/O 不同 slot 和大尾 block，确保
升级后差异能被审阅。

## 81. 端到端执行时序和资源 overlap

### 81.1 三队列模型

受限执行器至少区分三个逻辑队列，底层可以映射到已有线程/command queue：

```text
IO queue       : pread/mmap、短读重试、checksum/pack、slot READY
Convert queue  : dequant、dtype cast、layout transform、graph input binding
GPU queue      : encode/commit、kernel/MPSGraph、completion event
```

合法 overlap 是 `IO(i+1)` 与 `GPU(i)`、`Convert(i+1)` 与 `GPU(i)` 的并行；禁止
同一个 slot 在 `GPU(i)` 完成前被 `IO(i+2)` 覆写。每个 slot 使用单调 sequence：

```text
EMPTY(seq=n)
  -> LOADING(seq=n+1)
  -> READY(seq=n+1)
  -> IN_FLIGHT(seq=n+1, gpu_event)
  -> RETIRING(seq=n+1)
  -> EMPTY(seq=n+1)
```

`READY` ticket 必须同时记录 source offset、payload bytes、actual bytes、checksum
（如果 manifest 要求）和 generation。GPU 发现 ticket generation 不匹配时必须
返回 `memory_lifetime_violation`，不能尝试“看起来相同”的下一块。

### 81.2 预取窗口控制

预取窗口不是越大越好。控制器每个安全点计算：

```text
free_budget = B - committed - reserved
slot_need   = max(next_window.slot_upper_bytes)
```

只有 `free_budget >= slot_need + safety_margin` 时才增加 look-ahead。若连续两个
窗口 GPU 等待 I/O，且水位低于 0.80B，可从 1→2→3 槽逐步增加；一旦水位达到
0.90B、unattributed 上升或系统 pressure warning，先停止新增预取，再释放
READY-but-not-required 的 ticket。控制器不得在同一安全点同时扩容和缩容。

### 81.3 阶段 handoff

默认采用 strict handoff，明确牺牲少量 preload overlap 换取峰值可证明：

```text
current stage GPU event complete
 -> stop pager and drain IO/convert
 -> retire/release stage-only backing
 -> checkpoint footprint + ledger
 -> reserve next stage envelope
 -> create next stage objects
```

只有 manifest 明确声明 `cross_stage_overlap_bytes` 且 planner 证明仍小于 B 时，
才允许提前加载下一阶段。Stage1→upsample→Stage2 尤其不能因为两阶段复用同一
denoiser 权重就假设可以原地覆盖 latent；必须以实际 last-use event 决定何时释放。

## 82. 代码修改建议：从当前脚手架到可执行闭环

下表按依赖顺序细化比第 76 节更接近实现任务。每项修改都应带一个失败注入或
golden，避免“接口先加、语义后补”。

| 顺序 | 文件/符号 | 修改建议 | 关键断言 |
|---:|---|---|---|
| 82-01 | `native/core/memory_contracts.hpp` | 增加 `MemoryManifest`、`ResourceInterval`、`ResidencyPlan`、`MemoryCapability` 的 POD/序列化结构；所有长度用 `uint64_t` | JSON round-trip 不改变 digest |
| 82-02 | `native/runtime/memory_policy.*` | 将 `B/F0/E`、planner/manifest revision、candidate digest 组装成不可变 `MemoryPolicyRuntime` | disabled 不创建 runtime 对象 |
| 82-03 | `native/runtime/memory_accounting.*` | 在现有 ledger 上增加 scope id、tag 索引、reservation 状态和 peak-by-class 计数；保留现有 API | double release/旧 generation 不改变 bytes |
| 82-04 | `native/runtime/residency_planner.*` | 读取 manifest 和 shape，做 interval peak、slot window、strict handoff 规划 | minimum-1 与 minimum 结果稳定 |
| 82-05 | `native/runtime/weight_pager.*` | 实现 `PagerSource`、`RefillSlot`、`PrefetchTicket`；I/O 只向已 reserve 的 slot 写 | EOF/EINTR/cancel 后无 pending |
| 82-06 | `native/runtime/prepared_execution.*` | 合并 probe→plan→admission→session bind；返回 move-only `PreparedExecution` | 三入口 digest 相同 |
| 82-07 | `native/runtime/schedule_trace.*` | lock-free/有界 ring buffer；错误事件不可丢 | trace 满不改变执行 |
| 82-08 | `native/platform/apple/results.mm` | 输出每类 peak、unknown、watermark、terminal reason 和 plan digest | 旧字段值保持兼容 |
| 82-09 | `native/api/c_api.mm` | C ABI 只持有 opaque prepared handle；取消先 drain 后释放 | C 异常不泄漏 scope |
| 82-10 | `services/turbociderd/service.mm` | job 持久化 planning/admission/running/draining 状态和 candidate | 重启不恢复过期 generation |

### 82.1 C callback bridge

H3/LTX 的 C ABI callback 不应直接捕获 C++ `MemoryReservation`。建议在
`native/platform/apple/bridge.hpp` 增加一个小的 opaque bridge：

```cpp
struct MemoryHookBridge {
    tc::MemoryAdmission *admission;
    uint64_t allocator_domain;
    uint64_t generation;
};
```

`reserve` callback 在 bridge 中创建 `new HookToken{MemoryReservation, class, tag}`；
`commit` 校验 domain/generation 并把 token move 到 `StorageLease`；`cancel` 删除
未提交 token；`release` 调用 lease.release。C 回调不能抛异常，所有异常都转成
固定错误字符串。`HookToken` 的析构只允许发生在 owner thread 或明确的 completion
线程，避免在任意 autorelease pool 中触发大对象回收。

### 82.2 默认模式零行为变化

所有新增 wrapper 必须采用“旧函数委托到新函数、hooks 为空即 fast path”的形式：

```c
h3_gpu_create(...) {
    return h3_gpu_create_with_options(..., NULL, ...);
}
```

disabled 路径不得调用 `observe_process_memory`、`mx::clear_cache`、额外的
`MTLCommandBuffer`、文件 `fstat` 或 JSON digest 计算。ABBA 测试不仅比较 wall time，
还要比较 route、cache key、command 数量和输出 hash，防止“性能没变但执行路径已变”。

## 83. H3 C/Metal 具体落地合同

### 83.1 当前已落地的 allocator 方向

`h3_gpu.h/.m` 已加入独立的 `h3_gpu_memory_hooks`、
`h3_gpu_create_with_options`、classified tensor API、mmap/load reservation 和
completion-safe release。旧 `h3_gpu_create` 保持兼容。当前
`h3_session.mm` 已安装 bridge，并通过 `h3_params.gpu_options` 传到
`h3_dit.c`、纯文本 `h3_text_encoder.c` 与 Video VAE；
`h3_video_vae_decoder_load_with_options` 和 `h3_video_vae_decode_with_options`
已覆盖 resident decoder、chunked decode 与 single-shot decode，旧 API 委托
`NULL`。此外 H3 已有独立 `h3_host_memory_hooks`，与 GPU hooks 共享 allocator
domain/generation，首批覆盖 joint video/audio latent、RGB8 output 和
`h3.vae.decoded_host_envelope_v1`。

下一步不再是“传 options”，而是把 VAE/DiT/text upload 的 required unknown tensor
逐个改为 classified allocation，并把 conditioning/layout/tokenizer/control host
allocation 纳入逐 allocation hook 或 validated stage envelope。音频/视觉 encoder
继续排除在首发 candidate 之外。这些已接入的 hooks 仍只代表“部分请求账本”，不能
升级为端到端 L2/L3。

### 83.2 tensor 分类规则

分类不是装饰性 tag，而是 planner 反馈所需的计费维度：

| H3 对象 | C API 入口 | memory class | tag 格式 |
|---|---|---|---|
| denoiser activation | `h3_gpu_tensor_new_*` | `ACTIVATION` | `h3.activation.<name>` |
| conditioning/rope/row map | classified new | `CONDITIONING` | `h3.conditioning.<name>` |
| BF16/I8 streamed weight | load/read file | `WEIGHTS` | `h3.weight.block<idx>.<field>` |
| 双 refill slot | `allocate_stream_slot` | `REFILL_SLOT` | `h3.stream.slot<0|1>.<field>` |
| cast/dequant 临时 | classified new | `CONVERSION_SCRATCH` | `h3.convert.<name>` |
| 最终视频/音频输出 | classified new | `OUTPUT` | `h3.output.<name>` |

当前 `h3_dit.c` 的两个 refill slot 应保持固定为 2；slot tensor 采用
`h3_gpu_tensor_new_classified` 并使用稳定的 `h3.stream.slotN.field` tag。即使
quantized 和 BF16 slot 的字段不同，也不能回到 unknown allocation。slot 数量变化
只能在 planner capability 与 L3 证据同时更新后进行。

### 83.3 mmap composite accounting

`h3_gpu_tensor_map_bf16` 同时涉及文件映射页和 Metal no-copy alias。实现上先以
保守方式把 manifest 的 `mmap_payload_upper + alias_runtime_upper` 计入，等真实
采样证明 Metal 没有复制 backing 后，才在 manifest 中把 `mmap_aliasing` 标成
`validated` 并使用唯一 `StorageId` 去重。`munmap` 必须在 MTLBuffer deallocator
执行；ledger token 则在 tensor/command 最后使用完成后 release，二者不能由两个
独立路径各自释放。

### 83.4 H3 runtime 传参顺序

推荐调用链：

```text
h3_session.mm
  -> h3_params.memory_hooks/domain/generation
    -> h3.c prepare/load
      -> h3_dit_load_* / h3_gpu_create_with_options
        -> tensor/slot reserve-commit-release
```

`ssd_memory_budget_bytes` 在受限模式下只表示 denoiser child budget，不能再被当作
全请求预算。H3 text encoder、video/audio VAE、输出 arena 必须由父 scope 单独预留；
若这些模块仍使用旧的无 hooks allocator，candidate 必须保持 unsupported，而不是
把 denoiser 的成功误报为全请求成功。

## 84. LTX 具体落地合同和当前实现对齐

当前 `ltx_session.mm` 已有阶段级 host reservation：conditioning、Stage1/Stage2
latent、audio latent、Video VAE coarse graph envelope、BF16 planar pixels 和
RGB24 output，并在 last-use 处释放。该实现应继续遵守以下顺序：

```text
raw/conditioning reservation
 -> stage1 latent
 -> stage1 denoise + block slots
 -> upsample transition（释放 stage1-only）
 -> stage2 latent + denoise
 -> decode（释放 denoiser/graph/cache）
 -> planar -> RGB conversion
 -> video-only export 后释放 RGB
```

当前 Video VAE coarse graph envelope 使用 `2 GiB` 只是本机脚手架中的保守占位，
不是所有 shape 的认证上界。发布前必须按 `(model identity, width, height, frames,
dtype, OS, MLX/native revision)` 建立 envelope 表；任意未命中的 shape 返回
`memory_estimate_unknown`。

LTX native block pager 允许最多三个 refill slot，但 planner 只能选择 manifest
声明且 `max_refill_slots` 允许的槽位。video-only text-to-video 之外的 I2V/audio
路径，在 conditioning、audio latent、媒体队列和 VAE graph 未形成完整闭环前继续
fail-closed。Stage1/Stage2 共享权重 backing 时可以复用 lease，但每个阶段仍需
独立记录 last-use event，不能用“同一 pointer”替代生命周期证明。

## 85. Flux、Z-Image 以及新增模型的认证门禁

新模型接入不以“能把权重放进 Metal”作为完成标准，而以能力矩阵和 evidence bundle
作为完成标准。每个 candidate 至少要提交：

```text
manifest + checkpoint identity
shape bucket 列表
每个必需 allocation 的 call-site 清单
L0/L1 fake planner/pager 结果
L2 allocator/graph 结果
L3 minimum-1/minimum/baseline 结果
quality diff、swap delta、cancel cleanup 结果
```

### 85.1 Flux

第一阶段只认证 component-staged：text encoder 完成后释放其权重，transformer
完成后 strict handoff 到 VAE。只有 single/dual transformer group 的权重、attention
workspace、RoPE/conditioning 和 VAE envelope 都可按 group 读取时，才开放
`streamed(slot=2)`。不允许把 dual block 拆成任意 tensor 片段；若 kernel 需要两组
权重同时存在，`slot_upper` 必须取两组实际最大值。

### 85.2 Z-Image/GGUF

按 `Q8_0`、`Q4_0`、`Q4_1` 分别建立 manifest；mixed K-quants 不能复用任何一张
bytes 表。dequant/pack/CPU staging 归入 `CONVERSION_SCRATCH`，其上界必须在
创建 GGUF executor 前已知。refiner 启用时把 latent 跨阶段保留计入 overlap；未启用
时不预留 refiner 权重。普通 Z-Image、GGUF 和 Flux 都不能因 resident 估算恰好小于
B 就自动获得 constrained capability。

### 85.3 新模型模板

新增模型 PR 应先实现 `memory_capabilities()` 和 metadata-only
`estimate_memory()`，返回 `execution_supported=false` 的 dry-run plan；通过
L2/L3 后再切换 capability。模型名字符串、环境变量或用户传入 `residency` 都不能
直接绕过 capability gate。

## 86. API、版本、配置和运维兼容

### 86.1 配置版本

schema-v1 保持顶层兼容；若 schema-v2 使用 `execution.memory_constrained`，服务端
要同时保存原始 JSON 和规范化 plan。未知字段默认拒绝（避免客户端误以为新字段已
生效），但 `enabled=false` 时可以接受已知的实验字段并只做类型校验。`limit_bytes`
和所有 bytes 字段必须是 JSON 整数；App/Swift 的 GiB 输入在边界一次性转换，内部
不使用浮点。

### 86.2 feature flag

环境变量只能缩小能力集合，不能把未认证 candidate 打开：

```text
TC_MEMORY_CONSTRAINED_PLAN_ONLY=1       # 只生成 plan
TC_MEMORY_CONSTRAINED_LTX_C_METAL=1     # 允许 LTX 已认证候选
TC_MEMORY_CONSTRAINED_H3_C_METAL=1      # 允许 H3 已认证候选
TC_MEMORY_CONSTRAINED_TRACE=1           # 增加 trace，不改变调度
```

生产默认均为 0。开关状态和 capability revision 必须进入结果 JSON，便于发现
“同一请求在不同 worker 上走了不同实现”。

### 86.3 重启和恢复

job 恢复时只恢复请求和 checkpoint identity，不恢复旧的 Metal object、pager slot、
lease token 或 generation。服务把状态退回 `planning`，重新生成 digest；若 manifest
或 runtime revision 变化，则旧 evidence 不能继续用于 admission。

## 87. 验收扩展：从单测到证据包

### 87.1 L0/L1 必测断言

在无 Metal 设备也能运行的 fake backend 中增加：

```text
reserve upper == budget+1      -> memory_budget_too_small
commit actual > upper          -> memory_lifetime_violation
cancel twice                   -> reservation_count 不变
retire then completion         -> pending_count 从 1 到 0
old generation release        -> ledger bytes 不变 + violation trace
unknown required envelope     -> memory_estimate_unknown
pressure red                   -> 停止新 prefetch，不创建新 slot
```

fake pager 要模拟短读、EINTR、EOF、延迟 GPU completion、取消时 I/O 正在阻塞，且在
每个终态断言 `reservation_count=0`、`pending_release_count=0`、`storage_count=0`
（显式跨请求 cache 除外）。

### 87.2 L2 真实 allocator

有 Metal 设备时，除现有测试外增加 H3 hooks 测试：

```sh
TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
python3 -B tests/native/test_h3_gpu_memory_hooks.py
python3 -B tests/native/test_ltx_gpu_memory_hooks.py
```

测试只创建小 buffer，重点验证 callback 次序、异步 release 和分类 tag，不把小样本
误写成大模型成功。无 Metal 设备时只能输出 `SKIP`，不能伪造 `PASS`。

### 87.3 L3 campaign 结果格式

每次真实运行生成一条不可覆盖的 JSONL 记录，至少包含：

```json
{
  "status": "PASS",
  "case": "ltx.video.default.minimum",
  "checkpoint_identity": "sha256:...",
  "policy_digest": "...",
  "Y": 17179869184,
  "X": 15,
  "B": 14602888806,
  "F0": 3219128320,
  "E": 11383760486,
  "peak_process_footprint": 14490000000,
  "peak_unattributed": 120000000,
  "swapins_delta": 0,
  "swapouts_delta": 0,
  "gpu_starvation_ratio": 0.031,
  "output_hash": "sha256:..."
}
```

记录文件名必须包含日期、native commit、manifest revision 和 case id；重跑不能覆盖
旧样本。`CONTAMINATED`（外部进程、系统 pressure、已有 swap 变化无法隔离）与
`FAIL` 必须保留，汇总脚本不能只输出最好的一轮。

### 87.4 质量和默认回归

质量比较分三层：输出 hash（确定性路径）、像素/latent 最大绝对误差、感知/任务
指标。受限模式若使用同一 precision 和 sampler，默认阈值优先采用 baseline 的
逐元素容差；tiling 或 dequant 路径必须有单独阈值和人工审阅。默认模式做 ABBA：

```text
A = disabled warmup
B = constrained off（同一模型/shape）
B = disabled 再跑
A = disabled 再跑
```

若 route、cache key、command count 或 wall 的中位数回归超过 2%，即使输出正确也
要阻止合入，除非提交明确的性能解释和豁免。

## 88. 性能模型、调优和容量建议

### 88.1 overlap 指标

定义：

```text
overlap_ratio = hidden_io_or_convert_time / max(total_io_or_convert_time, 1ns)
gpu_starvation = gpu_wait_for_ready_slot / total_gpu_compute_window
```

目标不是盲目追求最大 `overlap_ratio`，而是在不触碰 B 的前提下使
`gpu_starvation` 下降。若从 2 槽增加到 3 槽只减少等待却使 peak 增加超过
`safety_margin`，应选择 2 槽。planner 的 score 同时考虑峰值、I/O、未隐藏等待和
候选稳定性，不以单一 wall 预测决定。

### 88.2 pinned prefix 策略

优先 pinned 访问频率最高、跨 step 重用且 payload 较小的 block；不要简单固定为
文件前 N 层。H3/LTX 若 kernel 对层顺序有严格约束，planner 可只在合法 prefix
边界选择 N。pressure 收缩时先减少 look-ahead，再减少 pinned prefix；恢复时顺序
相反，且必须经过三个稳定安全点，避免水位振荡。

### 88.3 低内存机器的用户体验

请求提交阶段应尽量快速返回 plan/rejection，而不是加载数百 MB 后才发现不 fit。
服务可在 `waiting_for_memory` 状态指数退避（例如 100ms、250ms、1s、2s，上限
30s），但静态的 `memory_budget_too_small` 不重试。UI/CLI 应显示：`Y`、`X`、`B`、
当前 baseline、候选、预计峰值、是否允许 tiling，以及明确的下一步建议；不能只显示
“内存不足”。

## 89. 发布阶段和回滚策略

建议把能力分为四个等级，结果 JSON 中只出现实际等级：

```text
L0 contract_only       只有解析/错误合同
L1 planned             有 metadata planner 和 fake evidence
L2 allocation_guarded  真实 allocator/pager 已受 admission 保护
L3 certified            目标 checkpoint 的完整 must-succeed + quality/swap 证据
```

发布时先默认 `L0/L1`，通过本机和 CI 的 L2 后按模型/shape 白名单打开，完成 L3
后才允许普通用户使用 `enabled=true`。回滚只需把 candidate capability 降级或关闭
feature flag，不应改变 disabled 路径。任何 allocator 发现 unknown required bytes、
generation violation 或 observed over-budget，都应自动降级为 fail-closed，并保留
完整 trace 供修复；不能静默回退到 swap 或 resident。

## 90. 明确非目标与后续扩展

本版本不承诺：

- macOS 进程的绝对 RSS/phys_footprint hard cap；
- 阻止系统对其他进程、文件 cache 或统一内存做任何 swap/compression；
- 在一个 GPU 上同时公平调度多个大模型作业；
- 自动改变模型量化、采样步数、画布尺寸或全局 attention 数学；
- 让未认证的 MLX、Flux、Z-Image、ANE/hybrid 路径因“估算看起来够小”而通过。

后续可扩展：多 GPU 独立 ledger、CPU/ANE 分区预算、跨请求权重 cache quota、
checkpoint page cache、在线 cost model 学习。但这些扩展必须先定义新的
`MemoryClass`、scope 和 evidence，不应把本版本 GPU-only 合同悄悄泛化。

## 91. 本轮实施结果与文档同步要求

本轮已补充并与当前工作树对齐的实现细节包括：

- H3 `h3_gpu` hooks 的 reserve/commit/cancel/release、classified tensor、mmap
  weight reservation 和 deferred completion release；
- H3 session bridge 已安装 allocator domain/generation，同时安装 GPU/host hooks；
  `h3_params.gpu_options` 已传播到 DiT、纯文本 encoder 和 Video VAE 的 resident/
  chunked/single-shot 调用链；音视频 encoder 仍未贯通；
- H3 双 refill slot 已使用 `H3_GPU_MEMORY_REFILL_SLOT` 与
  `h3.stream.slot<0|1>.<field>` 稳定 tag；
- H3 joint video/audio latent、RGB8 output 已使用 host reserve/commit/release，Video
  VAE F32/tile 暂由 `h3.vae.decoded_host_envelope_v1` 聚合覆盖；该 envelope 的
  `4x` 上界仍是启发式 scaffolding，且 commit 必须改为真实 frame shape；
- LTX 已实现 request-aware `uses_parent_mlx(const Request&)`，受限请求固定 C/Metal，
  candidate 与 session 均排除 input/audio/LoRA/非 streamed 路径；
- MemoryLedger 的 baseline、allocation ceiling、known/unknown/unattributed 和
  peak observed over-budget 指标语义；
- LTX 阶段级 host reservation、Video VAE coarse envelope、last-use release 和
  strict handoff 的后续认证要求。

这些内容仍不代表 H3 全阶段、LTX MLX/VAE、Flux 或 Z-Image 已达到 L3。实现 PR 合并
前必须同步更新第 39、65、75、76 和本节的“当前状态”，并附上实际测试输出路径。
文档中的数值示例、2 GiB envelope、16 GiB budget 以及候选名都只是合同示例，不能
直接复制成运行时常量或 capability 白名单。

## 92. 当前代码缺口审计与首个可认证范围

本节以当前工作树为准，目的是把“已经有接口”“已经接入某一组件”和“完整请求已
受控”三个概念分开。实现时不得因为某一列完成，就提前提升整条 candidate 的能力
等级。

| 能力 | 当前状态 | 首个发布版本要求 | 未完成时行为 |
|---|---|---|---|
| 配置解析与默认隔离 | 已有 `MemoryConstrainedConfig`、profile/request overlay | disabled 路径字节级兼容 | 关闭 add-on 继续走旧路径 |
| 预算计算与 plan JSON | 已有 `EffectiveMemoryPolicy` 和粗估 | 用 manifest interval peak 替换常量 reserve | `estimate_only`，不可执行 |
| 进程 baseline/footprint | 已有 `MemoryAdmission` checkpoint | admission、stage boundary、terminal 全采样 | observer 不可用则目标 shape 未认证 |
| Metal allocator admission | LTX 主路径与 H3 `h3_gpu` 已有 hooks | 所有首发 candidate 的必需大分配必须先 reserve | 任一必需 unknown 则拒绝 |
| H3 hook 传播 | DiT、纯文本 encoder 与 Video VAE options 已接入 | 清除首发路径 required unknown；音视频/vision 继续排除 | H3 仍不能标为全请求 L2 |
| H3 多模态/音频 | vision/video/audio encoder 仍使用旧 create | 首发先排除；后续逐组件接入 | capability gate 拒绝 |
| H3 host memory | joint video/audio latent 与 RGB8 已逐 allocation 纳管；Video VAE F32/tile 有启发式聚合 envelope | conditioning/control 补 envelope；decoded envelope 用真实 shape commit 并完成校准 | required unknown 非零或 envelope 未验证则拒绝 |
| LTX host memory | conditioning、latent、VAE/output 已有阶段 reservation | 将 2 GiB graph 占位替换为 shape envelope | 未命中 envelope 则拒绝 |
| MLX allocator/graph | 未形成通用逐 allocation guard | 首发 C/Metal candidate 不进入 MLX denoiser；LTX request-aware parent route 已强制 false | MLX candidate unsupported |
| cache 生命周期 | 受限 H3 已禁用主要 cache；LTX 有阶段清理 | cache 必须有 lease/quota/last-use | 不可证明则禁用 cache |
| L3 真实证据 | 尚无完整 must-succeed evidence | 每个 checkpoint/shape bucket 独立证据包 | 不能宣称 certified |

### 92.1 首发 candidate 必须进一步收窄

当前工作树已经把 `memory_policy.cpp` 的 H3 candidate 收窄到 model、GPU、
`video.generate`、无 audio、无 input、无 LoRA、无 quantized cache，并在
`h3_session.mm` 执行入口做相同二次校验。首个真正可执行 candidate 的最终白名单
保持为：

```text
model        = minimax-h3-turbo
operation    = video.generate
execution    = gpu | auto（规范化后为 gpu）
inputs       = empty
references   = empty
audio        = false
preview      = false
retain       = false
backend      = H3 C/Metal
residency    = streamed
refill_slots = 2
```

具体字段以 `Request` 已有语义为准；`preview_denoise`/`retain_decoded` 当前由 H3 C
参数控制，受限模式已经在 `h3_valid_params()` 中拒绝非零值，session 默认传 0。
后续若把这两个字段暴露为 request 配置，candidate matcher 也必须显式检查，不能只
依赖 C runtime 的晚失败。直到 Video VAE required unknown、host conditioning/control
与真实 manifest 闭环完成前，即使满足上述条件也只能
`route_available=true`；`execution_supported` 不得由启发式 estimate 自动置 true。

LTX 首发范围保持当前约束：`video.generate`、video-only、无 input、`c_metal`、GPU、
streamed。I2V、audio、`cpp_mlx` 均不得借用此 capability。建议把 route 判定抽成具名
函数，而不是继续增长一条条件表达式：

```cpp
CandidateMatch match_h3_streamed_video_v1(const Request&);
CandidateMatch match_ltx_streamed_video_v1(const Request&);
```

`CandidateMatch` 返回 `matched`、`candidate_id`、`rejection_code`、`rejection_detail`。
planner 和执行入口调用同一函数，避免 plan 能选中、run 又走到另一条路径。

### 92.2 当前两个占位常量必须退出执行判定

`memory_policy.cpp` 当前给 H3/LTX 都设置 `4 GiB` non-denoiser reserve；LTX session
当前给 Video VAE graph 使用 `2 GiB` coarse envelope。二者只能作为 scaffolding，不能
成为 `execution_supported=true` 的依据。替换顺序如下：

1. manifest 给出 text/conditioning/latent/VAE/output/compile temporary 的资源区间；
2. planner 对实际 shape 求区间峰值，得出 `planned_upper_bytes`；
3. 缺失任何 required 项时返回 `memory_estimate_unknown`；
4. 常量只可保留在 plan-only 调试模式，且 provenance 必须是
   `heuristic_not_executable`；
5. 代码中不允许出现“estimate fits 因而 execution supported”的隐式升级。

### 92.3 首发完成的精确定义

某 candidate 达到 `allocation_guarded` 至少同时满足：

- 从 request 进入 session 到输出完成，所有大于等于 manifest `individual_guard_threshold`
  的 allocation 都经过 ledger；
- 小于阈值的必需 allocation 被一个已验证的阶段控制面 envelope 覆盖；
- `required_unknown_bytes == 0`，而不是把 unknown 重命名成 activation；
- 每个 async GPU backing 在 completion 后才 release；
- cancel、错误、短读、allocator failure 后 ledger 回到预期终态；
- 实际 route 与 plan 的 `candidate_id`、digest、generation 完全一致；
- 真实 allocator 测试通过，但尚未等同 L3 大模型认证。

## 93. 运行时对象与接口的增量改造方案

不建议一次性替换现有 `MemoryLedger` API。更稳妥的做法是在保持
`try_reserve`/`commit`/`retire`/`release` 行为的基础上，逐步增加 scope、实际大小和
错误原因。这样 H3/LTX 已接入的 hooks 不需要重写。

### 93.1 拟增核心类型

建议在 `native/runtime/memory_accounting.hpp` 增加：

```cpp
using MemoryScopeId = uint64_t;

enum class MemoryRequirement : uint8_t {
    Required,      // 不可计费或被拒绝时终止请求
    OptionalCache, // 可丢弃，不能影响数学结果
    OptionalPrefetch,
};

struct MemoryReservationRequest {
    MemoryScopeId scope_id = 0;
    MemoryClass memory_class = MemoryClass::UnknownExternal;
    MemoryRequirement requirement = MemoryRequirement::Required;
    uint64_t upper_bytes = 0;
    uint64_t alignment = 1;
    std::string tag;
};

struct MemoryReservationFailure {
    std::string code;
    uint64_t requested_bytes = 0;
    uint64_t free_budget_bytes = 0;
    MemoryScopeId scope_id = 0;
    std::string tag;
};
```

新增 overload，旧 API 委托到新 API，保证当前调用点可渐进迁移：

```cpp
ReservationResult try_reserve(const MemoryReservationRequest& request);
```

`ReservationResult` 可以是项目已有 expected 风格，或者一个含
`optional<MemoryReservation>` 与 failure 的简单结构；不要靠解析异常字符串获取
requested/free bytes。

### 93.2 scope 设计

每个 request 一个 root scope，每个阶段和 pager 一个 child scope：

```text
request/<job-id>
  text
  conditioning
  denoise/stage1
  upsample
  denoise/stage2
  video_vae
  audio_vae
  export
  pager/<component>
```

建议增加 move-only RAII：

```cpp
class MemoryScope {
public:
    MemoryScope child(std::string name);
    std::optional<MemoryReservation> try_reserve(
        MemoryClass, uint64_t upper, std::string tag,
        MemoryRequirement = MemoryRequirement::Required);
    void checkpoint(std::string_view phase);
    void close();
};
```

`close()` 不是强制释放仍在 GPU in-flight 的 backing，而是把 active lease 转为
pending，并检查“不应跨出本阶段”的对象。scope 析构时若仍有 required active lease，
记录 `memory_scope_leak`；debug/test 构建可以终止，生产构建应 fail 当前请求并完成
drain，不能导致 C++ 析构抛异常。

### 93.3 reservation 与实际 allocation 大小

当前 `StorageId.capacity` 已可表示真实 backing capacity。需要进一步固定以下规则：

```text
reserve upper U
allocate actual A
要求 A <= U
commit StorageId(capacity=A)
ledger 计费采用 backing capacity A
在 allocation/commit 间继续占用 reservation U
```

若 allocator 返回 `A > U`：立即销毁尚未暴露给执行器的 backing、cancel reservation、
返回 `memory_estimate_violation`。不能临时追加第二次 reserve，因为这会让 planner 的
上界失去意义。alignment、Metal `allocatedSize`、vector capacity 而非 logical bytes
都应作为 A。

同一 backing 的 alias 使用相同 `(allocator_domain, handle, capacity, generation)`；
新 view 增加引用，不新增 storage bytes。不同 allocator 恰好返回相同 pointer 时，
domain/generation 防止误去重。

### 93.4 allocation ceiling 的精确定义

令：

```text
B  = 用户有效预算
F0 = admission 时进程 footprint
P  = planner 给出的完整进程峰值上界
C  = min(B, P)                         // root allocation ceiling
I  = max(0, C - F0)                    // 本请求可新增的 backing 上界
```

root ledger 应按当前实现构造为 `MemoryLedger(C, F0)`：budget/ceiling 是完整进程上界
C，baseline F0 只计一次，具体新 backing 最多消耗 I。F0 在 checkpoint 时用于判断
进程总 footprint，不能再作为 storage 记一次。正常情况下 planner 已要求 `P <= B`，
所以 C=P；保留 `min` 只是防御性校验。若 P 本身是增量上界而非完整进程峰值，上式
必须改用 `P_increment`，并在字段名上显式区分。建议废弃语义含混的
`planned_upper_bytes`，新增：

```cpp
uint64_t planned_process_peak_upper_bytes;
uint64_t planned_increment_upper_bytes;
uint64_t admission_baseline_bytes;
uint64_t allocation_ceiling_bytes;
```

过渡期 JSON 同时输出旧字段与新字段，但执行代码只读取新字段；contract test 断言两者
关系，避免未来再次把 full peak 和 increment 混用。

### 93.5 必需 unknown 的处理

`UnknownExternal` 不能笼统地一律拒绝，因为 footprint observer 会看到系统/框架的
不可细分波动；需要把它拆成两种语义：

- `required_unknown_bytes`：已知是当前请求导致、但没有 upper bound 的必需分配；只要
  非零就禁止执行 candidate；
- `observed_unattributed_bytes`：`process_footprint - baseline - known backing` 的观测
  差值；它有校准 envelope，但任何越过 envelope 的样本都使本次运行失败。

建议在 metrics 中同时保留二者。禁止通过把默认 tensor 统一标成 `Activation` 来让
unknown 归零；class 解决“是什么”，upper-bound provenance 才解决“能否准入”。

## 94. 确定性 planner 与分层策略算法

planner 的输入必须只依赖 request、checkpoint metadata、manifest、device/runtime
identity 和冻结的 policy。运行中 pressure 可以选择 plan 预先声明的降档，但不能
临时创造未进入 digest 的新 tile、精度或 block 切分。

### 94.1 候选枚举空间

每个模型 adapter 提供有限候选，不实现任意组合搜索：

```cpp
struct ResidencyCandidate {
    std::string id;
    ResidencyMode mode;
    uint32_t pinned_prefix_blocks;
    uint32_t refill_slots;
    uint32_t block_group_size;
    TileShape decode_tile;
    bool strict_stage_handoff;
    bool allow_cross_stage_preload;
};
```

候选顺序由 adapter 固定，例如：

```text
resident
component_staged
streamed(prefix=N, slots=3)
streamed(prefix=N, slots=2)
streamed(prefix=N, slots=1)  // 只有 executor 认证后才枚举
tiled_decode + streamed
```

同一候选中 N 可以通过单调搜索求最大安全值，但最终 N 必须写入 plan；运行时不得因
“看起来还有空间”擅自增加。

### 94.2 非均匀 block 的精确峰值

现有 uniform `block_bytes` 适合首版 H3/LTX，但 manifest 应支持每层不同大小。对按
顺序执行的层 `i`：

```text
resident_prefix = sum(bytes[0..N-1])
slot_window(i)  = sum(largest required slot payloads in window i)
stage_live(i)   = non_block_floor(i) + resident_prefix + slot_window(i)
candidate_peak  = max_i(stage_live(i))
```

当一个 layer 需要 Q/K/V 或 dual/single group 多个 shard 同时存在时，manifest 用
atomic `load_group` 表达；planner 不得按单 tensor 切开来虚假降低峰值。尾层大于普通层
时，slot capacity 取每个并发 slot 可能承载的最大 payload，而非平均 block 大小。

### 94.3 资源区间求峰值

每项资源映射到逻辑事件区间 `[acquire_event, release_event)`：

```cpp
struct ResourceInterval {
    std::string id;
    MemoryClass memory_class;
    uint64_t upper_bytes;
    EventId acquire;
    EventId release;
    std::string backing_group;
    EstimateProvenance provenance;
    bool required;
};
```

planner 先按 `backing_group` 去重，再对事件扫描：release 必须先于同一事件的 acquire
处理，以表达 strict handoff；只有 manifest 明确声明 overlap 时才反过来。实现要用
checked add/sub，任何溢出均是 `memory_manifest_invalid`。

### 94.4 候选可行性和排序伪代码

```text
best = none
for candidate in adapter.enumerate(request, capability):
    if candidate changes quality contract: continue
    intervals = adapter.materialize_intervals(candidate, shape, manifest)
    if any required interval lacks executable upper bound: reject UNKNOWN
    peak = sweep_peak(intervals)
    if peak > B: reject BUDGET
    if peak + system_reserve conflicts with current system availability:
        reject TRANSIENT_SYSTEM_PRESSURE
    cost = estimate_cost(candidate, calibration)
    score = cost.wall_upper
          + lambda_io * cost.unhidden_io_upper
          + lambda_risk * candidate.uncertainty_penalty
          + lambda_peak * peak / B
    choose minimum (score, peak, bytes_read, candidate.id)
```

最后一行给出完全确定的 tie-break。`system availability` 只影响当前 admission，不进入
稳定 policy digest；它进入 admission record。静态 budget 不可行与瞬态系统压力必须
使用不同错误码。

### 94.5 运行时允许的降档集合

plan 内预先列出 `fallback_ladder`，仅允许质量不变且已估算的动作：

```text
level 0: configured slots/prefix/tile
level 1: stop optional look-ahead
level 2: slots 3 -> 2（slot 释放完成后）
level 3: reduce pinned prefix to declared boundary
level 4: smaller certified VAE tile
level 5: fail closed
```

H3 当前执行器固定双槽，因此首版 H3 只允许 level 0、1、3、5；不能生成虚假的
`slots 2 -> 1`。LTX 只有已有执行器与 evidence 支持 2/3 槽时才把对应降档放进 plan。

## 95. Scheduler 状态机、压力控制与安全点

### 95.1 作业状态机

建议统一为：

```text
CREATED
 -> PROBING
 -> PLANNED
 -> ADMITTED
 -> BOUND_TO_SESSION
 -> RUNNING(stage, revision)
 -> DRAINING
 -> COMPLETED

任意非终态 -> CANCELLING -> DRAINING -> CANCELLED
任意非终态 -> FAILING    -> DRAINING -> FAILED
```

只有 `ADMITTED -> BOUND_TO_SESSION` 后 allocator callback 才可 reserve。session 解绑
必须发生在所有 completion callback 已结束之后。`DRAINING` 有独立超时和诊断，但
超时不能通过释放仍在 GPU 使用的 backing 来“修复”账本；应终止 worker 或保留泄漏
证据并使该 generation 永不复用。

### 95.2 唯一合法重规划点

运行中只在以下 safe point 读取 pressure 并选择 fallback ladder：

- 组件加载前；
- 每个 denoise step 之间；
- block group 的 GPU completion 后、下一个 group reserve 前；
- Stage1/upsample/Stage2 的 handoff；
- VAE tile 或 temporal chunk 之间；
- export chunk 完成后。

kernel 已编码、graph 已提交或 slot 处于 `IN_FLIGHT` 时禁止改变其 payload、释放 backing
或修改 generation。pressure callback 只设置原子 flag，实际回收由 owner scheduler 在
safe point 执行。

### 95.3 水位输入与去抖

每个 safe point 采集：

```text
ledger_committed
ledger_reserved
pending_release
process_footprint
observed_unattributed
system_available
OS pressure level
next_required_reservation_upper
```

决策优先级固定：

1. generation/lifetime violation：立即 fail；
2. 下一必需 reservation 超 budget：执行已计划降档，仍失败则 fail；
3. observed process footprint 超 B：记本次违约，停止新工作并 drain；
4. OS critical：停止 optional prefetch，在下一个 safe point shrink；
5. yellow/red watermark：按第 4.3 节 hysteresis 调整；
6. green 且稳定三个 safe point：最多提升一级 optional look-ahead，不能增加 pinned
   prefix 超过初始 plan。

为了避免小层高频 safe point 引起振荡，同一方向变更至少间隔一个完整 block group 或
一个 VAE tile；critical 和预算拒绝不受间隔限制。

### 95.4 停止与回收顺序

cancel 或错误后的通用 drain 顺序：

```text
stop accepting new prefetch
cancel QUEUED/LOADING tickets（等待 I/O 回调退出）
release READY-but-never-submitted slots
wait/observe submitted GPU completion
complete pending releases
destroy stage object and graph
drop request-owned cache
close child scopes
checkpoint terminal footprint
unbind session admission
close root scope
```

`ScopedMemoryAdmissionBinding` 析构只能做 unbind，不能隐式替代 drain；否则 C runtime
异步 release 会落到空 bridge 或下一个 request generation。

### 95.5 “不能超过 X%”的产品语义

运行时能承诺的是：

- TurboCider 已知的大分配在分配前受 B 准入；
- 所有认证 graph/allocator 的未知开销被验证 envelope 覆盖；
- 观测到超预算会使运行失败且 candidate 证据失效；
- 不主动依赖 swap，不会在失败后切到 unbounded 模式。

不能承诺的是 macOS 对整个系统的绝对 hard cap。UI、CLI、文档和错误信息统一使用
“TurboCider 受控预算目标/validated envelope”，不要写成“操作系统保证 RSS 永不超过”。

## 96. H3 逐文件改造清单（GPU 与 host 两条账本同时闭环）

H3 的正确接入顺序不是先把 `h3_gpu_create` 全局替换，而是先建立生命周期边界，
再迁移每类 allocation。这样可以在任一步失败时保持旧 API 可编译，并通过
`h3_gpu_options == NULL` 保持默认模式的原路径。

### 96.1 `h3_video_vae` 的 options 传播

当前工作树已完成 `_with_options` API 和旧函数 `NULL` 委托；resident decoder、
chunked decode、single-shot decode 已改用 `h3_gpu_create_with_options`。接口合同如下：

```c
h3_video_vae_decoder *h3_video_vae_decoder_load_with_options(
    const char *weight_directory, const char *shader_source_path,
    int latent_height, int latent_width,
    const h3_gpu_options *gpu_options,
    h3_video_vae_progress progress, void *progress_opaque,
    char *error, size_t error_size);

int h3_video_vae_decode_with_options(
    const char *weight_directory, const char *shader_source_path,
    const float *normalized_latent, int latent_time,
    int latent_height, int latent_width,
    const h3_gpu_options *gpu_options,
    h3_video_vae_progress progress, void *progress_opaque,
    h3_video_frames *output, char *error, size_t error_size);
```

已完成步骤：

1. `h3_video_vae_decoder` 保存经过 ABI 校验的 options 副本；context 使用同一份配置。
2. resident decoder、chunked decode 和 single-shot decode 均创建带 options 的 GPU。
3. `h3.c:h3_acquire_video_decoder` 与最终一次性 decode 均传递
   `params->gpu_options`。
4. 旧 `h3_video_vae_decoder_load`/`h3_video_vae_decode` 委托 `_with_options(..., NULL)`。

仍需完成的步骤：

1. 增加 `h3_gpu_tensor_from_f32_classified`（以及确有需要时的 BF16/U32 classified
   upload），使 upload 与空 tensor 使用同一分类入口。
2. 将 `prepare_input_batch`、`prepare_rope`、`allocate_activations`、`run_decoder`、
   `run_resident_tile` 中的 required allocation 全部标成 activation、conditioning 或
   conversion scratch；首发路径不得残留 `H3_GPU_MEMORY_UNKNOWN`。
3. 为这些调用点增加 hook fixture，断言 tag、class、reserve/commit/release 次数和
   allocation failure 的清理顺序。
4. 继续只认证“无 preview、无 retain、video-only”；resident preview decoder 与
   TAEH3 在拥有独立 output/host envelope 前保持 unsupported。

旧函数合同必须保持：

```c
h3_video_vae_decoder_load(...) {
    return h3_video_vae_decoder_load_with_options(..., NULL, ...);
}
```

这可避免 disabled 模式新增一次 options 分支以外的行为变化，也让现有 H3 诊断工具
无需修改。

### 96.2 H3 host-memory hooks

GPU hooks 不能覆盖 `malloc/calloc/realloc`。当前工作树已在 `h3.h` 增加独立 C ABI，
并在 `h3_session.mm` 复用同一个 `MemoryAdmission`、allocator domain 和 generation；
该 ABI 不复用 `h3_gpu_memory_hooks`，因为 host allocation 的释放时机和 Metal
completion 语义不同：

```c
typedef enum {
    H3_HOST_MEMORY_CONDITIONING = 1,
    H3_HOST_MEMORY_LATENT = 2,
    H3_HOST_MEMORY_DECODED_F32 = 3,
    H3_HOST_MEMORY_OUTPUT = 4,
    H3_HOST_MEMORY_STAGING = 5,
    H3_HOST_MEMORY_CONTROL = 6,
    H3_HOST_MEMORY_UNKNOWN = 255
} h3_host_memory_class;

typedef struct {
    uint32_t struct_size;
    uint32_t version;
    void *user;
    int (*reserve)(void *user, uint32_t memory_class,
                   uint64_t upper_bytes, const char *tag,
                   void **token, char *error, size_t error_size);
    int (*commit)(void *user, void *token, uint64_t allocator_domain,
                  uint64_t handle, uint64_t actual_bytes,
                  uint64_t generation, char *error, size_t error_size);
    void (*cancel)(void *user, void *token);
    void (*release)(void *user, void *token);
} h3_host_memory_hooks;
```

bridge 可先把 `DECODED_F32` 与 `OUTPUT` 映射到现有
`tc::MemoryClass::Output`，`LATENT` 映射到 `Activation`，但 trace/tag 保留更细的 H3
分类；如果后续需要按类独立 peak，再在一次 schema revision 中扩展 `MemoryClass`，
不要在不同 adapter 中私自使用不一致的枚举值。

`h3_params` 增加：

```c
const h3_host_memory_hooks *host_memory_hooks;
uint64_t memory_allocator_domain;
uint64_t memory_generation;
```

当前实现已经提供 `h3_accounted_host_malloc/free`、
`h3_reserve_host_envelope`、`h3_commit_host_envelope` 和
`h3_release_or_cancel_host_envelope`。后续如果迁移调用点继续增多，建议再收敛成带
pointer/capacity/token 的 `h3_host_buffer`，避免调用者分别保存裸指针与 token：

```c
typedef struct {
    void *ptr;
    size_t capacity;
    void *token;
    h3_host_memory_class memory_class;
    int hooked;
} h3_host_buffer;

int h3_host_buffer_alloc(h3_host_buffer *, h3_host_memory_class,
                         size_t upper_bytes, const char *tag,
                         const h3_host_memory_hooks *, char *, size_t);
void h3_host_buffer_release(h3_host_buffer *,
                            const h3_host_memory_hooks *);
```

规则：

- `upper_bytes` 必须是 checked multiplication 结果；溢出直接返回
  `memory_manifest_invalid`/`h3_host_allocation_overflow`。
- hook 存在时，先 reserve，再 `malloc`，再以真实 `capacity` commit；malloc 失败必须
  cancel exactly once。
- hook 不存在时走旧 malloc，但只允许在 disabled/非受限路径；受限路径发现 hook
  缺失时返回 `memory_policy_unsupported`，不得静默无账本分配。
- `realloc` 禁止直接使用。实现为 reserve 新 buffer → copy → release 旧 buffer，
  或在调用点预先计算最大容量并只分配一次。
- 小于 `H3_HOST_CONTROL_THRESHOLD` 的字符串、索引和错误缓冲可以由一个
  `CONTROL` envelope 聚合；阈值、聚合上界和调用点列表必须进入 manifest。

### 96.3 H3 host allocation 迁移表

| 文件/位置 | 现有对象 | 目标 class/tag | 生命周期 | 当前状态 |
|---|---|---|---|---|
| `h3.c` input copy/prepare | video/audio/reference copy | `LATENT` / `h3.input.*` | request end 或 denoise last-use | 首发 candidate 无 input；后续拟增 |
| `h3.c` conditioning arrays | token/vision/audio rows | `CONDITIONING` / `h3.conditioning.*` | encoder/conditioning last-use | 未完成；首发需 stage envelope |
| `h3.c` `h3_rgb_f32_to_u8` | RGB8 output | `OUTPUT` / `h3.output.rgb8` | export 完成 | 已逐 allocation 纳管 |
| `h3.c` result/frame object | frame metadata + pointers | `CONTROL/OUTPUT` / `h3.output.frames` | result free | 小对象未独立纳管 |
| `h3_video_vae.c` `extract_latent_tile` | tile input | `STAGING` / `h3.vae.tile_input` | tile GPU completion | 由 decoded-host 聚合 envelope 临时覆盖 |
| `h3_video_vae.c` `unpack_frames` | F32 decoded tile | `DECODED_F32` / `h3.vae.tile_rgb` | stitch 完成 | 由 decoded-host 聚合 envelope 临时覆盖 |
| `h3_video_vae.c` `stitch_tiles` | final RGB arena | `DECODED_F32` / `h3.vae.final_rgb` | RGB conversion/export | 由 decoded-host 聚合 envelope 临时覆盖 |
| `h3_video_vae.c` rope/stat arrays | cosine/sine/normalization | `CONDITIONING` / `h3.vae.rope` | decoder free | host 未逐 allocation 纳管；GPU 仍有 unknown |
| `h3.c` `video/audio` joint latent | denoiser input/output | `LATENT` / `h3.latent.video/audio` | VAE last-use | 已逐 allocation 纳管 |

迁移一项后要加 allocation-failure test；不能只靠最终 footprint 采样推断它已受控。
对于 frame 数量很大的输出，优先使用 temporal chunk + encoder/export consumer，避免
一次性保留整个 F32 与 RGB8 双份；若当前 API 要求完整 `h3_video_frames`，首发 manifest
应明确把双份输出上界计入，而不是偷偷改变结果接口。

### 96.4 H3 encoder/decoder 的 capability 分层

建议把 H3 candidate 拆成以下能力，而不是一个布尔值：

```text
h3.dit.gpu_hooks                 = implemented
h3.text_encoder.gpu_hooks        = implemented（纯文本路径）
h3.video_vae.gpu_options         = implemented
h3.video_vae.required_classified = partial
h3.host.latent_hooks             = implemented（joint latent）
h3.host.output_hooks             = partial（RGB8 + decoded envelope）
h3.host.conditioning_hooks       = planned
h3.multimodal.vision_hooks       = unsupported
h3.audio_vae.gpu_hooks            = unsupported
h3.preview.full_output            = unsupported
```

planner 只在 capability 的 required 集合全部为 `implemented+validated` 时返回
`execution_supported=true`。多模态请求不应因为它最终也会进入同一个 DiT，就复用
`h3_c_metal_streamed_v1`。

## 97. LTX 原生、MLX 与阶段 handoff 的代码合同

### 97.1 `uses_parent_mlx(request)` 必须与 candidate 一致

`ModelSession` 的默认实现保留旧行为；当前 LTX session 已增加 request-aware override：

```cpp
bool uses_parent_mlx(const Request& request) const override {
    if (request.memory_constrained.enabled)
        return false; // 首发 constrained 只允许 C/Metal
    return ltx_mlx_requested(request);
}
```

`uses_parent_mlx()` 无参版本继续由基类提供兼容语义；新执行入口必须优先调用
request-aware 版本。后续仍必须满足：同一个 constrained request 的 `plan.backend`、
`uses_parent_mlx(request)`、session 内 `use_mlx` 三者一致。contract test 应同时检查
这三个值，服务层不得为了复用旧 worker 再调用无参版本覆盖结果。

`TURBOCIDER_LTX_MLX=1` 是环境级旧开关，不能覆盖 add-on 的 GPU-only gate；受限请求
发现该开关时返回 `memory_policy_conflict` 或 candidate unsupported，而不是在
session 内悄悄切到 MLX。

### 97.2 LTX vector 容量与 reservation 对齐

当前阶段已经有 `reserve_host_memory`，但 `std::vector` 的自动增长可能在 reserve
后产生临时旧/新双份。所有受限路径改为：

```cpp
template <class T>
HostVector<T> allocate_host_vector(MemoryClass cls,
                                   size_t logical_count,
                                   std::string tag) {
    const uint64_t upper = checked_vector_upper<T>(logical_count, tag);
    auto reservation = reserve_host_memory(cls, upper, tag);
    HostVector<T> value;
    value.reserve(logical_count); // capacity 在 commit 前固定
    value.resize(logical_count);
    auto lease = commit_host_memory(reservation,
                                    value.capacity() * sizeof(T), tag);
    return {std::move(value), std::move(lease)};
}
```

如果某个算法必须增长，先为目标容量做第二个显式 reservation，提交成功后再切换，
旧 backing 的 release 必须发生在 copy 完成且 GPU 无引用之后。禁止 `push_back` 在
受限阶段隐式扩容。

### 97.3 LTX native/MLX 两条路径的边界

首发 constrained LTX 只认证：

```text
LTX C/Metal denoiser block pager
host conditioning/latent/output reservation
strict transformer -> VAE handoff
video-only text-to-video
```

以下对象暂不视为受控：`ltx_mlx_*` denoiser cache、MLX VAE/upsampler allocator、
MPSGraph 未公开 temporary、audio BWE/vocoder、I2V encoder。即使这些对象在某次运行
没有导致超预算，也只能记录为 observation，不能写入 certified manifest。

### 97.4 LTX 阶段代码检查点

按 `ltx_session.mm` 当前流程，每一个点都应有 reservation/lease 对：

```text
load_conditioning        -> conditioning reservation
stage1 vectors            -> stage1 video/audio lease
stage1 denoise            -> native pager leases
upsample                  -> stage2 vector reservation
stage2 denoise            -> stage2 pager leases
transformer cleanup      -> stage2/conditioning last-use release
video VAE graph           -> compile temporary envelope
decoded planar pixels     -> output lease
planar -> RGB24           -> second output lease
export                    -> RGB lease release
```

`video_vae_graph_reservation` 的 2 GiB 值只能在 plan-only 模式出现。认证前应为
`ltx_video_vae_estimate(shape, dtype, runtime_revision)` 建立表格；未命中就返回
`memory_estimate_unknown`。如果 `exec_ltx_video_finalizer` 继续保留，必须独立记录为
process-isolation candidate，不得与 in-process 的 `peak_process_footprint` 混在同一
证据集。

## 98. Model Manifest 与 Plan JSON 的可执行合同

### 98.1 manifest 最小结构

manifest 是随模型 adapter 发布、由 checkpoint identity 绑定的静态文件；用户请求不
能修改其中的上界。建议版本化为 `memory-manifest-v1`：

```json
{
  "schema": "memory-manifest-v1",
  "model": "minimax-h3-turbo",
  "adapter": "h3_c_metal_streamed_v1",
  "checkpoint": {"sha256": "sha256:...", "bytes": 123},
  "runtime": {"metal_family": "apple7", "native_revision": "..."},
  "shape_buckets": [
    {"width": 544, "height": 544, "frames": 49, "steps": 4,
     "dtype": "bf16", "quality_contract": "default"}
  ],
  "components": [
    {"id": "text", "class": "conditioning", "upper_bytes": 123456,
     "provenance": "validated_envelope", "required": true,
     "acquire": "text.load", "release": "text.last_use"},
    {"id": "dit.block.0", "class": "weights", "upper_bytes": 789,
     "provenance": "exact", "load_group": "dit.group.0",
     "acquire": "dit.block.0.ready", "release": "dit.block.0.complete"}
  ],
  "slots": {"count": 2, "upper_bytes": 456,
            "generation_checked": true},
  "envelopes": {"compile_temporary": 123, "host_control": 456},
  "capabilities": {"preview": false, "audio": false,
                    "references": false, "tiling": false}
}
```

必填检查：schema、model/adapter、checkpoint identity、runtime identity、shape bucket、
每个 required component 的 interval、slot 上界、所有 envelope 的 provenance。字段
缺失或重复 `id/backing_group` 都是 `memory_manifest_invalid`；manifest 不是可信输入，
解析必须限制字符串长度、数组数量和总 bytes 上界，防止恶意 JSON 溢出或 DoS。

### 98.2 Plan JSON 与 execution result 的字段分离

`plan` 表示准入前冻结的决定；`result` 表示实际执行证据。不要把观测 peak 回写到
原 plan：

```json
{
  "memory_policy": {
    "enabled": true,
    "limit_bytes": 17179869184,
    "buffer_percent": 15,
    "effective_budget_bytes": 14602888806,
    "planned_process_peak_upper_bytes": 14400000000,
    "planned_increment_upper_bytes": 11200000000,
    "candidate_id": "h3_c_metal_streamed_v1",
    "manifest_revision": "h3-memory-v1@...",
    "plan_digest": "...",
    "fallback_ladder": ["stop_prefetch", "fail_closed"]
  },
  "memory_result": {
    "status": "PASS",
    "peak_process_footprint_bytes": 14310000000,
    "peak_committed_bytes": 10980000000,
    "peak_reserved_bytes": 11800000000,
    "peak_unattributed_bytes": 91000000,
    "required_unknown_bytes": 0,
    "observed_over_budget_bytes": 0,
    "watermark_transitions": 3,
    "swapins_delta": 0,
    "swapouts_delta": 0,
    "terminal_reason": "completed"
  }
}
```

任何字段的单位统一为 bytes；百分比只在配置中使用整数。`peak_process_footprint` 与
`peak_committed` 不应互相替代。结果必须含 `route`、`execution`、`uses_parent_mlx`、
`generation`、`native_revision`，方便定位 plan/执行分叉。

### 98.3 错误码和用户可执行建议

建议固定错误码，UI 只根据 code 做本地化，detail 保留诊断：

| code | 含义 | 是否重试 |
|---|---|---|
| `memory_policy_invalid` | 配置类型/范围错误 | 修改配置 |
| `memory_policy_conflict` | 与 legacy budget/ANE/MLX 冲突 | 修改请求 |
| `memory_manifest_invalid` | adapter manifest 不可信 | 更新模型/软件 |
| `memory_estimate_unknown` | required upper bound 缺失 | 换已认证 shape/model |
| `memory_budget_too_small` | 静态候选超过 B | 提高 Y/降低 shape |
| `memory_system_pressure` | 当前系统不可安全准入 | 关闭其他任务后重试 |
| `memory_policy_unsupported` | route 尚未认证 | 关闭 add-on 或换模型 |
| `memory_lifetime_violation` | stale generation/错误释放 | 不重试同一 worker |
| `memory_observed_over_budget` | 运行中实际越过 B | 保留 evidence，修 manifest |
| `memory_cancelled` | 用户取消，drain 已完成 | 按需重试 |

静态不可行不应自动反复重试；系统压力可有限指数退避。`memory_observed_over_budget`
和 lifetime violation 应标记 worker 为 `tainted`，该 worker 不再接受 constrained job，
直到进程重启，避免旧 Metal/MLX 对象污染后续证据。

## 99. 调度 trace、诊断与服务生命周期

### 99.1 bounded trace 事件

每条 trace 事件至少包含：

```text
monotonic_ns, job_id, scope_id, generation, stage, block, slot,
event_kind, upper_bytes, actual_bytes, committed, reserved,
process_footprint, system_available, queue, result_code
```

`event_kind` 固定枚举：`reserve_begin`、`reserve_denied`、`commit`、`prefetch_begin`、
`prefetch_ready`、`gpu_submit`、`gpu_complete`、`retire`、`release`、`watermark`、
`fallback`、`cancel`、`error`。trace 使用有界 ring buffer；满时允许丢弃普通采样，
但不得丢弃 `error`、`over_budget`、`generation_violation` 和 terminal 事件。trace
写入本身不得申请未受控的大块内存。

### 99.2 进程与 job 状态持久化

服务持久化的是 request、plan digest、checkpoint identity、manifest revision、状态和
错误；不持久化 `MTLBuffer`、Metal event、pager slot pointer、reservation token。
推荐状态字段：

```text
planning -> admitted -> running -> draining -> completed|failed|cancelled
```

服务重启、manifest 变化、native revision 变化或 worker 被标记 tainted 时，所有旧 job
回到 `planning`，重新做 metadata probe。恢复不得直接跳到 `running`，也不得复用旧
generation。

### 99.3 结果收集与隐私

默认记录模型/shape/candidate/bytes/time/错误码，不记录 prompt、原始媒体或 token。若
debug 需要保存 prompt hash，也只能保存不可逆 digest。evidence bundle 中的
checkpoint identity、output hash 和 trace 文件路径必须可关联，但不应把模型绝对路径
或用户目录泄漏给远端服务。

## 100. 测试与验收：实现者可直接执行的矩阵

### 100.1 L0/L1（无 Metal 设备也必须通过）

```sh
python3 -B tests/native/test_memory_accounting.py
python3 -B tests/native/test_contract.py
python3 -B tests/native/test_h3_streaming_policy.py
python3 -B tests/native/test_ltx_streaming_benchmark.py
git diff --check
```

另外增加纯 C++ fake scheduler 测试，覆盖：预算恰好等于上界、上界多 1 byte、checked
add 溢出、非均匀 block、release/acquire 同事件的 strict handoff、旧 generation、
unknown required、可选 prefetch 拒绝和 fallback ladder 的确定性 tie-break。

### 100.2 L2（有 Metal 时执行，无 Metal 时只能 SKIP）

```sh
MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" \
TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
python3 -B tests/native/test_h3_gpu_memory_hooks.py
python3 -B tests/native/test_ltx_gpu_memory_hooks.py
```

测试输出必须区分 `PASS`、`FAIL`、`SKIP_NO_METAL`；无设备不能把编译通过写成 allocator
认证。新增 H3 VAE options 测试后，还要检查：options 从 session 到每个 create call
的传播、commit failure 的 exactly-once cancel、异步 completion 后 release、cache
disabled 时没有 stale token。

### 100.3 L3 真实模型 campaign

每个 checkpoint/shape bucket 至少跑以下三类：

```text
minimum-1：B 比 planner minimum 少 1 byte，必须在 admission 失败
minimum：B 恰好满足 validated upper，必须成功
headroom：B 比 upper 多 5%/10%，比较吞吐与 watermark
```

每类至少 warmup 1 次、正式 3 次；任何外部进程、系统 critical pressure、swap 计数
无法隔离的样本标 `CONTAMINATED`，不能选择性丢弃。L3 must-succeed 还需要：输出质量
阈值、`required_unknown_bytes=0`、`observed_over_budget=0`、generation violation 为 0、
cancel cleanup 通过、swapins/swapouts 增量满足门禁。

### 100.4 disabled ABBA 回归

对 H3、LTX 各选一个默认 shape，运行：

```text
A0 disabled warmup
B0 constrained=false（显式关闭 add-on）
B1 disabled
A1 disabled
```

比较 route、cache key、command count、输出 hash、wall 中位数和峰值。受限代码不得在
disabled 时调用 process observer、ledger、额外 `fstat`、cache clear 或 command queue。
若 wall 中位数回归超过 2% 或输出 hash 改变，阻止合入，除非给出可复现实验和豁免。

### 100.5 验收门槛表

| 等级 | 必须满足 | 可对外状态 |
|---|---|---|
| L0 | schema/error/disabled contract | plan-only |
| L1 | metadata manifest、fake planner/pager、确定性 digest | 实验开关 |
| L2 | 真实 allocator hooks、async cleanup、unknown=0 的小流程 | 内部白名单 |
| L3 | 目标 checkpoint 全流程、quality、swap、cancel、ABBA | 默认可用 |

L2/L3 只对具体 `candidate_id + checkpoint + shape bucket + runtime revision` 生效；
不能因为 H3 544p 通过就放行 H3 768p，也不能因为 LTX C/Metal 通过就放行 LTX MLX。

## 101. 分阶段 patch stack 与每步 DoD

建议将实现拆成互相可审查的 PR/commit；每一步都能编译并运行已有测试：

| Patch | 内容 | DoD |
|---|---|---|
| P0 | 文档、错误码、manifest schema、feature flag | parser/contract 通过，无执行变化 |
| P1 | `MemoryScope`/reservation failure/metrics 字段 | accounting 单测与旧 API 兼容 |
| P2 | planner interval sweep、非均匀 block、fallback ladder | golden digest/overflow/unknown 测试 |
| P3 | pager queue/ticket/generation、fake I/O | 短读、EINTR、cancel、stale 测试 |
| P4 | H3 VAE `_with_options` 传播 | 编译、options propagation、旧 API 测试 |
| P5 | H3 host hooks 与 latent/output arena | allocation failure、cleanup、unknown=0 |
| P6 | H3 full video-only C/Metal stage handoff | L2 小 shape；无 preview/audio |
| P7 | LTX native stage manifest 替换 4 GiB/2 GiB 占位 | L2 allocator + shape envelope |
| P8 | LTX constrained candidate/parent-MLX route 修正 | route contract、disabled ABBA |
| P9 | trace/result/service persistence | evidence bundle、tainted worker |
| P10 | H3/LTX L3 campaign 与 capability whitelist | must-succeed、quality、swap 门禁 |
| P11 | Flux/Z-Image component-staged 设计实现 | 仅在独立 L3 后打开 |

P4/P5 未完成前，P6 不应把 H3 candidate 标为 `execution_supported`。P7 未完成前，LTX
受限请求即使原生 pager 成功，也只能返回 `estimate_only`。每个 patch 应包含：
变更文件清单、失败注入、测试命令、性能/默认模式影响说明和 evidence 路径。

### 101.1 建议的 code review checklist

- 是否新增了旧路径调用点但漏传 options？
- 是否在 reserve 前已经发生了 malloc、vector growth、Metal allocation 或 graph compile？
- `actual_bytes` 是否使用 capacity/allocatedSize，而非 logical bytes？
- 同一 backing 的 alias 是否使用稳定 `StorageId` 去重？
- async callback 是否检查 domain + generation？
- cancel/error/destructor 是否 exactly-once release/cancel？
- constrained request 是否可能经由 `auto`、环境变量、cache 或 parent MLX 绕过 gate？
- `execution_supported` 是否仅由 validated manifest/capability 设置？
- disabled 路径是否增加 observer、fstat、cache clear、queue 或 digest？
- 结果 JSON 是否同时报告 full process peak、known/unknown/unattributed 和 terminal reason？

## 102. 性能实现细节：在低内存下保持吞吐

### 102.1 I/O 与转换 overlap

pager 的 I/O、convert、GPU 三队列只做可证明安全的 overlap：

```text
IO(i+1) -> Convert(i+1) -> GPU(i+1)
             与 GPU(i) 并行
```

I/O 使用 page-aligned offset、短读循环和有限大小的 staging buffer；不要为了“顺序读”
建立整文件缓存。对于 mmap，manifest 需要区分 mapped virtual bytes、resident observed
bytes 和 Metal alias capacity；没有唯一 backing 证明时采用保守上界。prefetch ticket
达到 `READY` 后立即开始 convert，不能再额外复制一份“方便调试”的 payload。

### 102.2 Metal/Objective-C++ 生命周期

受限阶段每个 command buffer 必须绑定完成回调 token；回调只做轻量的
`complete_pending`/原子计数，实际大对象销毁在 owner scheduler 的 safe point 执行。
建议每个 block group 建立局部 `@autoreleasepool`，但不得把 pool drain 当作账本 release
的唯一证据。Metal heap、MPSGraph cache、MLX allocator cache 的 trim 都必须有明确
`MemoryClass::AllocatorCache` lease 或阶段边界事件。

### 102.3 预取窗口与 pinned prefix 的初始值

首版不从在线学习开始，以确定性保守值启动：

```text
H3 C/Metal       slots=2，pinned prefix=manifest 最大安全 N
LTX C/Metal      slots=manifest 认证值（通常 2 或 3）
look-ahead       首次 window=1，连续两次 GPU starvation 才升一级
shrink           先停 optional prefetch，再降 prefix
restore          连续三个 green safe point 才恢复一级
```

如果 I/O latency 高导致 GPU starvation，优先扩大 staging 的并行度而非盲目增加 slots；
staging 扩容也必须先 reserve。任何调优结果都作为新的 manifest calibration，不通过
环境变量在生产中绕过 plan digest。

### 102.4 低内存机器的容量建议

对用户暴露的 Y 应至少保留 `X=15%` 默认 buffer，并根据 checkpoint/shape 的
validated envelope 建议最小 Y。若 `B < F0 + minimum_increment`，提前给出静态不可行
原因，不要先启动 encoder 再失败。若系统可用内存小于 `min_free_bytes`，状态设为
`waiting_for_memory`；但等待期间不持有已加载的模型权重，避免“排队 job”本身造成压力。

## 103. 风险登记与处理优先级

| 风险 | 触发条件 | 影响 | 处理 |
|---|---|---|---|
| graph temporary 未知 | MPSGraph/MLX 私有 workspace 无 upper | 可能瞬时超 B | 未纳管即拒绝；建立 shape envelope |
| mmap alias 复制 | Metal no-copy 假设不成立 | 重复计费/峰值低估 | 保守计双份，实测后才去重 |
| vector 隐式增长 | `push_back/resize` 造成双份 | 账本漏记 | 固定 capacity 或 HostBuffer |
| stale callback | worker/session generation 复用 | 旧对象释放新对象 | domain+generation 校验，worker tainted |
| cache 残留 | stage handoff 后 allocator cache 不清 | VAE 与 denoiser overlap | strict handoff + cache lease |
| system pressure 抖动 | pressure 在 safe point 间变化 | 吞吐波动/反复 shrink | hysteresis + bounded transitions |
| 模型 manifest 漂移 | checkpoint/shape/runtime 变化 | 上界失效 | identity digest，未命中拒绝 |
| disabled 回归 | 新路径在默认模式执行 | 充足内存机器变慢 | wrapper fast path + ABBA |
| 误标 unknown | 全部 tensor 粗暴标 activation | 失去 fail-closed | required unknown 独立字段审计 |
| output 双份 | F32、RGB8、编码队列同时存活 | export 峰值过高 | chunk/export consumer 或计入完整 envelope |

优先级固定为：lifetime correctness > hard admission > unknown elimination > quality
parity > overlap > wall-time。任何性能优化都不能逆转前四项。

## 104. 实现完成后的最终验收清单

合入前，负责人应逐项勾选并在 PR 中附证据路径：

```text
[ ] disabled/default route、cache、output hash、wall 通过 ABBA
[ ] schema/overlay/conflict/error contract 通过
[ ] B=floor(Y*(100-X)/100) 且 bytes 全程整数 checked arithmetic
[ ] plan 使用完整进程峰值与增量峰值的明确字段
[ ] required_unknown_bytes=0 或请求被 fail-closed
[ ] H3/LTX 每个 required allocation 有 call-site 清单与 lease
[ ] reserve→allocate→commit→retire→release 生命周期无 double release
[ ] GPU completion、I/O cancellation、generation stale 均有测试
[ ] stage handoff 不存在未经 manifest 证明的 overlap
[ ] pressure/fallback 只在 safe point，且 ladder 确定
[ ] trace 错误/terminal 事件不丢，result JSON 可重放 plan
[ ] L2 allocator 测试真实运行；无 Metal 时明确 SKIP
[ ] L3 minimum-1/minimum/headroom、质量、swap、cancel 证据齐全
[ ] capability whitelist 精确到 model/checkpoint/shape/runtime
[ ] rollout flag 默认关闭，rollback 不改变 disabled 路径
```

在上述清单全部满足前，文档顶部状态保持 `Proposal + scaffolding`，H3/LTX 的
`execution_supported` 不得被解释为已完成 L3。完成后再把对应 candidate 从 L2/L3
白名单逐步打开，并同步更新第 39、65、75、76、91 节的实现状态和测试输出路径。

## 105. 设计决策记录（ADR 摘要）

### ADR-01：显式 pager 优先于 swap

选择显式 streaming/loading/offloading，因为它能在分配前知道上界、利用 last-use
和 GPU event 做确定性释放，并能把 I/O/compute overlap 纳入 planner。swap 只作为
OS 级最后兜底，不作为 API、fallback 或验收通过条件。

### ADR-02：受限模式 opt-in 且 fail-closed

充足内存机器不应为新增账本和 observer 付成本；低内存机器宁可明确拒绝，也不能静默
切回 resident/unbounded。该选择使性能回归和错误语义都可审查。

### ADR-03：先认证 GPU C/Metal，再处理 MLX/ANE

Apple Silicon 统一内存下，MLX/MPSGraph 的私有 cache/workspace 需要额外 envelope。
首发只开放已经能逐 allocation 保护的 native C/Metal；MLX/ANE 另建 capability，
避免把“能执行”误当成“能受控”。

### ADR-04：manifest upper 是准入合同，不是在线猜测

planner 可以使用 heuristic 生成 plan-only 结果，但只有 exact/validated envelope
才能进入执行。真实运行的 peak 用来校准新 manifest，不得反向放宽当前请求。

## 106. 后续扩展边界

完成本 GPU-only 方案后，才考虑：多 GPU 独立 ledger、CPU/ANE 分区、跨请求权重 cache
quota、checkpoint page cache、在线 cost model。每项扩展都必须新增 MemoryClass、
scope、manifest schema revision、错误码和 evidence，不得把当前 `inference_process_tree_v1`
语义无声扩展为多租户全机硬限制。

## 107. 本轮补充后的状态同步

本轮新增的实施级约束是：

- H3 candidate 收窄到已实际具备调用链基础的 GPU video-only 方向；reference/audio/
  preview/multimodal 不得借用同一 capability；
- H3 视频 VAE `_with_options` API、decoder/一次性/resident options 传播和旧 API
  `NULL` 委托已落地；下一步是清除 VAE required GPU unknown，而不是继续添加入口；
- H3 独立 host hooks 已落地并覆盖 joint latent、RGB8 和 decoded-host 聚合 envelope；
  conditioning/control、小对象阈值和真实 decoded commit 仍需补齐；
- LTX request-aware `uses_parent_mlx(request)` 已落地，受限路径固定 C/Metal；仍需把
  环境变量冲突、session `use_mlx` 与 service route 固定为同一合同测试；
- 4 GiB non-denoiser 与 2 GiB VAE graph 只能作为 plan-only scaffolding，认证前不得
  触发 `execution_supported=true`；
- manifest、plan、result、trace、worker taint 和错误码均有可持久化合同；
- patch stack、L0–L3 命令和最终 DoD 已明确，任何缺口都保持 fail-closed。

这些补充不会改变 disabled/default 模式，也不宣称 H3/LTX 当前已经完成全阶段 L3。
本轮静态合同测试为 81 项通过、1 项因缺少 Wan fixture 跳过；该结果只证明接口和路由
合同未回退，不是 Metal allocator L2 或真实模型 L3 证据。

## 108. Flux 与 Z-Image 的具体接入设计

这两个模型当前不是 H3/LTX 那种已经存在 native block pager 的路径，因此不能只在
`memory_policy.cpp` 增加 candidate 名称。首阶段目标应是 component-staged：文本编码、
transformer、VAE 严格分段；之后再按量化格式和 kernel 的真实同时存活集合增加 block
streaming。

### 108.1 Flux：先拆组件，再谈 transformer 分层

当前 `native/models/flux2/pipeline.cpp` 的 `Flux::load()` 会同时加载并 materialize
transformer 与 VAE；`conditioning()` 另有 Qwen/MLX 文本路径和 cache。受限模式建议
增加显式组件 API：

```cpp
enum class FluxComponent { TextEncoder, Transformer, Vae };

LoadResult load_component(FluxComponent, const Request&,
                          const Event&, std::atomic<bool>&);
void release_component(FluxComponent);
MemoryEstimate estimate_component(FluxComponent, const Shape&,
                                  const FluxManifest&);
```

受限 text-to-image video-free candidate 的最小顺序：

```text
load TextEncoder -> encode prompt -> release TextEncoder/cache
load Transformer (+ LoRA baked copy if requested) -> denoise -> release Transformer
load VAE -> decode -> release VAE
export -> release output
```

第一阶段只允许 `image.generate`、无 input、无 reference、无 text-encoder LoRA、GPU
执行；`image.transform/edit` 的 reference latent、多个图片和 image encoder 需额外
manifest。`Flux::load()` 旧行为保留给 disabled/resident；constrained 入口调用新的
`load_component`，禁止先调用旧 `load()` 再清理，因为旧路径已经造成 transformer+VAE
同时 materialize 的峰值。

必须特别处理当前 `conditioning()` 中的：

- `cached_conditioning_` 的 cache lease；受限模式默认不跨 request 保留，命中也要计入
  conditioning bytes；
- `mx::clear_cache()` 的时机；只能在 stage handoff 调用，且 cache bytes 要有 envelope；
- `select_loras()` 产生的 load-time baked 权重；LoRA merge 临时副本计入
  `ConversionScratch`/`CompileTemporary`，不能只计最终 transformer bytes；
- `select_acceleration()` 的 ANE/hybrid fallback；constrained 必须在 candidate gate
  前拒绝 `gpu_ane`，不能沿用 automatic fallback。

Flux 的后续 streamed candidate 需要在 `flux_transformer.cpp` 将 dual/single block
按 `load_group` 组织；若某个 attention kernel 同时需要 dual 与 single 权重，manifest
必须以 atomic group 计费。未经此改造，`component_staged` 是唯一可认证的 residency。

### 108.2 Z-Image：safetensors 与 GGUF 分开认证

当前 `native/models/z_image/z_image.cpp` 的 `load()` 会把 transformer、VAE 一起载入，
`text_encode()` 再单独载入 text encoder；GGUF 由 `native/models/z_image/gguf.cpp`
选择 checkpoint。受限模式改造为：

```cpp
LoadResult load_text_component(...);
LoadResult load_transformer_component(...);
LoadResult load_vae_component(...);
void clear_text_component(...);
void clear_transformer_component(...);
void clear_vae_component(...);
```

每个组件的 `Weights::load/materialize/clear` 都要包在对应 scope reservation 中。对
diffusers safetensors：

```text
text encode -> clear text weights/cache
transformer denoise -> clear transformer
VAE decode -> clear VAE
```

对 GGUF：

- `Q4_0/Q4_1/Q8_0/mixed K` 必须各自有 manifest；
- `load_gguf_file` 的 mmap、metadata、dequant/pack、CPU staging、Metal tensor 分开
  计费；
- GGUF tensor 的 dequant buffer 若由 kernel 需要跨多个 block，使用 `load_group`，不能
  假定每层独立；
- `nvfp4`/`convrot` 的 `pack_*` 临时副本计入 `ConversionScratch`，pack 完成并验证
  last-use 后才释放源 tensor；
- `z_transformer()` 的手工 BF16 cast 和 VAE `mx::astype` 产生的 array 不得标成
  logical zero-copy，必须以实际 allocator backing 计费。

首个 Z-Image candidate 建议只做 `z-image-turbo` safetensors、text-to-image、无输入、
GPU、component-staged；GGUF 单独建立 `z-image-turbo-gguf` candidate，未有 dequant
上界前保持 unsupported。这样不会把一个“文件格式不同但模型名相近”的 checkpoint
误用同一份 bytes 表。

### 108.3 Flux/Z-Image capability 表

| 模型 | 首阶段模式 | 首阶段输入 | 当前可执行性 | 后续门禁 |
|---|---|---|---|---|
| Flux.2 Klein 4B | component-staged | text-only image.generate | plan-only | MLX allocator/cache envelope、LoRA merge |
| Flux image.transform/edit | component-staged | image inputs | unsupported | image encoder/reference latent 全闭环 |
| Z-Image safetensors | component-staged | text-only | plan-only | text/DiT/VAE 分开 materialize |
| Z-Image GGUF | component-staged | text-only | unsupported | 每量化格式 dequant/staging manifest |
| Z-Image refiner/LoRA | staged + overlap | 依请求 | unsupported | merge/cross-stage upper bound |

`plan-only` 仅允许生成 estimate/拒绝原因，不应让 `execution_supported` 为 true。只有
当 MLX cache、graph temporary、output arena 和 cancel cleanup 都有 evidence 后，才
可从 H3/LTX 之外独立提升等级。

## 109. Manifest 生成、校准与版本治理

### 109.1 metadata-only manifest probe

manifest 生成工具不得执行完整模型推理，也不得为了探测而把整套权重 materialize 到
GPU。建议新增：

```sh
tools/native/memory_manifest_probe \
  --model minimax-h3-turbo \
  --checkpoint /path/to/checkpoint \
  --shape 544x544x49 --steps 4 --dtype bf16 \
  --backend c_metal --output manifest.json
```

probe 读取 safetensors/GGUF index、tensor dtype/shape、layer group、VAE config 和已
注册 calibration table；输出 `exact` 项与待校准的 `validated_envelope` 项。若必须
运行小型 dry-run，使用 fake/最小 shape 并把结果标为 calibration，不得直接写入生产
capability whitelist。

### 109.2 校准流程

每个 envelope 按以下流程生成：

1. 固定 checkpoint SHA、native/MLX revision、Metal family、OS、shape、steps、dtype；
2. 在 clean worker 中运行三次 warmup + 三次正式，记录每个 class 的 peak、graph
   compile、allocator cache、process footprint 和 swap delta；
3. 对可重复字段取最大实测值，再加固定 alignment/allocator margin；不得用 P95 直接
   当硬上界；
4. 对只观测到一次的临时对象，标 `heuristic_not_executable`，直到失败注入或更大
   shape 证据覆盖；
5. manifest revision 递增，生成 digest，更新 capability whitelist；旧证据不覆盖。

### 109.3 版本失效条件

以下任何变化都使旧 manifest 对 constrained execution 失效：

```text
checkpoint SHA / file size / index 改变
native revision、shader source、Metal family 改变
MLX version、allocator/cache policy 改变
OS major/minor 改变（若 envelope 未声明兼容范围）
shape bucket、dtype、steps、LoRA merge identity 改变
pager slot、tile、block group、kernel compile flag 改变
```

失效后允许保留旧 manifest 做比较，但 plan 必须 `execution_supported=false`，不能
自动复用“相近版本”的证据。

## 110. 参考实现伪代码与数据流

下面伪代码描述 request 入口应达到的事务语义。它不是要求一次性新增同名类，而是
给实现和 code review 一个统一行为基准：

```cpp
RunResult run_constrained(const Request& request) {
    const auto policy = freeze_policy(request);       // Y/X/B/digest
    const auto manifest = load_verified_manifest(request);
    const auto candidate = choose_candidate(request, manifest, policy);
    const auto plan = planner.plan(request, manifest, candidate, policy);
    require(plan.required_unknown_bytes == 0,
            "memory_estimate_unknown");
    require(plan.process_peak_upper <= policy.effective_budget_bytes,
            "memory_budget_too_small");

    auto admission = prepare_admission(plan, policy); // F0 + increment ceiling
    ScopedMemoryAdmissionBinding binding(session, &admission);
    Scheduler scheduler(plan, admission, trace);
    scheduler.start();
    try {
        scheduler.stage("text", [&] { adapter.encode(request); });
        scheduler.stage("denoise", [&] { adapter.denoise(request); });
        scheduler.stage("vae", [&] { adapter.decode(request); });
        scheduler.stage("export", [&] { adapter.export_result(request); });
        scheduler.drain();
        admission.checkpoint("complete");
        require(admission.metrics().observed_within_budget,
                "memory_observed_over_budget");
        return make_result(plan, admission.metrics(), trace);
    } catch (...) {
        scheduler.cancel_and_drain();
        admission.checkpoint("failed");
        throw;
    }
}
```

实现注意事项：

- `freeze_policy`、manifest identity、candidate 和 plan 必须在任何模型权重加载前完成；
- `prepare_admission` 失败时 session 不得绑定，避免半初始化对象；
- `scheduler.stage` 只能在内部建立 child scope，不能让 adapter 私自创建 root ledger；
- `adapter` 抛出异常后，scheduler 负责 drain，C API 负责转成固定错误码；
- `make_result` 只能读取冻结 plan + metrics + trace，不修改 plan 或重新估算；
- 任何 fallback 都是 plan 内已声明 ladder 的 revision，revision 改变要写 trace/result。

## 111. 文档与实现同步规则

每次代码修改只要触及以下任一项，就必须同步本文档的“当前状态”和 evidence 记录：

```text
MemoryClass / reservation 状态 / StorageId 去重语义
h3_gpu_options、h3_params、host hooks、LTX allocator bridge
candidate 匹配条件、uses_parent_mlx、cache 开关
manifest schema、shape bucket、slot/tile/group size
错误码、result JSON、trace event、worker taint
L0/L1/L2/L3 通过范围和测试命令
```

PR 描述必须注明：

1. 修改了哪一个章节和哪一个 capability；
2. 是否改变 disabled/default 路径；
3. 新增/失效了哪些 manifest/evidence；
4. 运行了哪些命令，`PASS/FAIL/SKIP/CONTAMINATED` 各多少；
5. 若只完成脚手架，明确写 `Proposal + scaffolding`，不得写“已支持内存硬上限”。

本文档的“方案完整”不等于“实现已完成”；完成度以代码、测试和真实 evidence 三者
同时存在为准。当前状态仍保持 fail-closed：任何模型、shape 或 backend 未被认证，
受限模式应返回可解释的拒绝，而不是通过 swap、resident 或未受控 MLX 路径继续运行。

## 112. 当前工作树收尾基线与能力成熟度

本节记录截至 2026-09-15 当前工作树的实现事实，并给后续 PR 一个不会误报的成熟度
模型。它优先于前文中仍以“计划新增 Video VAE/host hooks”描述的旧段落。

### 112.1 已经可以依赖的事实

| 项目 | 当前事实 | 可以据此做什么 | 不能据此宣称什么 |
|---|---|---|---|
| 配置与策略 | `MemoryConstrainedConfig`、profile overlay、Y/X/B、冲突校验已经存在 | 做 schema/plan/错误合同测试 | OS hard cap |
| H3 candidate | 已收窄为 H3 Turbo、GPU、T2V、video-only、无 input/LoRA/quantized cache；session 有二次 gate | 防止明显未接入路径绕过 planner | H3 全流程 allocation guarded |
| H3 GPU bridge | DiT、纯文本 encoder、Video VAE 已能收到同一 generation 的 hooks；classified upload、VAE/text/common-DiT/Euler velocity 首批调用点已进入工作树 | 继续逐调用点分类，不必再改顶层 options ABI | H3 首发全路径 required unknown 已清零 |
| H3 host bridge | joint video/audio latent、RGB8、decoded-host 聚合 envelope 已纳管；decode 后开始按真实 frame shape commit | 做 host reserve/commit/release 失败注入 | conditioning/control/VAE host 已逐 allocation 完成或 4x envelope 已验证 |
| LTX route | request-aware `uses_parent_mlx` 已让 constrained request 固定 C/Metal | 避免 parent MLX 路由分叉 | MLX VAE/upsampler 已受控 |
| LTX allocator | 原生 Metal 主 buffer 路径和阶段 host reservation 已部分接入 | 做真实 allocator L2 测试 | 所有 vector/graph/helper 峰值都已闭环 |
| Plan | 能返回 candidate、estimate、policy 字段；H3/LTX 无 manifest 时固定 plan-only | 解释为什么选择/拒绝并阻止误执行 | 该 candidate 已可发布或运行 |
| 测试 | 最新 native build 通过；`tests/native/test_contract.py` 81 项通过、1 项缺 Wan fixture 跳过；memory accounting/H3 residency/LTX benchmark 通过；H3/LTX Metal hook 因无 Metal device 跳过 | 证明最新工作树可编译且纯逻辑/静态合同未回退 | 证明 Metal L2、真实 checkpoint L3 或无 swap |

### 112.2 五级能力成熟度

不要继续用一个 `bool execution_supported` 同时表达“路由存在”“allocator 已接入”和
“真实机器认证完成”。建议内部引入以下单调成熟度；对外仍可保留 bool，但 bool 只能
由规定等级派生：

```cpp
enum class MemoryCapabilityLevel : uint8_t {
    Declared = 0,             // 有 candidate 名称和拒绝矩阵
    HookBridged = 1,          // hooks/options 已到达组件，但可能有 unknown
    AllocationGuarded = 2,    // required allocation 全部 guard，unknown=0
    EnvelopeValidated = 3,    // shape/runtime manifest 有验证上界
    L3Certified = 4,          // 真实目标模型 campaign 通过
};
```

建议字段语义：

```text
route_available       = level >= Declared 且 request matcher 命中
execution_supported   = level >= EnvelopeValidated 且 manifest identity 精确命中
release_stable        = level >= L3Certified 且 rollout whitelist 命中
```

`AllocationGuarded` 不代表可以执行任意 shape：它只证明所有必需 allocation 会在分配前
被拒绝或提交到账本。没有 verified envelope 时，planner 仍不知道 graph/workspace 的
可靠 upper。`L3Certified` 也不是全模型通配符，而是绑定 checkpoint、shape bucket、
backend、dtype、slot/tile 策略、runtime revision 和设备族的能力记录。

### 112.3 当前 H3/LTX 的建议等级

```text
H3 C/Metal T2V        HookBridged（VAE/text/common DiT/Euler velocity 已分类；剩余 GPU/host 缺口）
LTX C/Metal T2V       HookBridged～AllocationGuarded 之间（需完成 vector/graph/helper 审计）
H3/LTX MLX            Declared 或 unsupported
Flux/Z-Image          plan-only Declared；没有独立认证不得执行
```

在第 114–118 节的缺口完成前，代码保持 `execution_supported=false`。结果 JSON 已增加
`capability_level`、`certification_state`、`manifest_digest`、`evidence_digest` 和
`release_stable`，调用方不得再从一个 bool 猜测成熟度。

## 113. Allocation Call-Site 注册表与完整性证明

“所有大分配都包了 hook”必须可以由代码和测试机械验证，不能依赖人工记忆。建议建立
稳定的 allocation-site 注册表，将代码 tag、manifest 资源项和测试期望连接起来。

### 113.1 稳定 site id

每个受限路径的必需 allocation 使用稳定 `site_id`；动态 block/tile 序号放在 trace
字段，不拼进 manifest 主键。建议格式：

```text
<model>.<component>.<stage>.<object>

h3.vae.decode.input_upload
h3.vae.decode.rope_cos
h3.vae.decode.activation.qkv
h3.host.denoise.video_latent
ltx.stage1.latent.video
ltx.vae.decode.graph_temporary
flux.transformer.block_group.weight_slot
zimage.vae.decode.output_tile
```

若每层大小不同，manifest 在 `instances[]` 里记录 block index/bytes，而不是生成几十个
不稳定字符串。日志可以输出 `site_id + instance_id`。

### 113.2 建议的数据结构

在 `native/runtime/memory_manifest.hpp` 中增加跨模型只读结构；C runtime 只接收
`site_id` 字符串和算好的 upper，不依赖 C++ 类型：

```cpp
enum class AllocationLifetime : uint8_t {
    Request, Stage, Block, Tile, CommandBuffer, Export
};

enum class UpperProvenance : uint8_t {
    ExactMetadata,
    ExactShapeFormula,
    ValidatedEnvelope,
    HeuristicNotExecutable,
};

struct AllocationSiteSpec {
    std::string site_id;
    std::string component;
    std::string stage;
    MemoryClass memory_class;
    AllocationLifetime lifetime;
    UpperProvenance provenance;
    bool required = true;
    bool asynchronous = false;
    bool aliasable = false;
    uint64_t guard_threshold_bytes = 0;
};

struct AllocationInstance {
    std::string site_id;
    uint32_t instance_id = 0;
    uint64_t upper_bytes = 0;
    uint64_t live_begin = 0;
    uint64_t live_end = 0;
    std::string last_use_event;
};
```

`AllocationSiteSpec` 随代码发布，描述语义；`AllocationInstance` 由 checkpoint metadata
和 request shape 生成，描述本次请求的数值。两者共同进入 plan digest。

### 113.3 代码与 manifest 的双向校验

metadata probe 结束后执行：

```cpp
for (const auto& site : code_registry.required_sites(candidate)) {
    require(manifest.contains(site.site_id),
            "memory_estimate_unknown: missing required allocation site");
    require(manifest.provenance(site.site_id) !=
                UpperProvenance::HeuristicNotExecutable,
            "memory_estimate_unknown: unvalidated allocation site");
}
for (const auto& item : manifest.required_items()) {
    require(code_registry.contains(item.site_id),
            "memory_manifest_invalid: orphan required allocation site");
}
```

运行时 hook 事件也要校验 `site_id`：

- 未注册 required tag：记 `unknown_required_bytes`，立即拒绝或在尚未分配时 fail；
- manifest 有 site，但 actual allocation 从未发生：允许，记录为条件分支未命中；
- actual bytes 超 upper：`memory_envelope_violation`，当前请求失败、worker tainted、该
  manifest 失效；
- 同一 `site_id + instance_id` 同时存在多个 backing：只有 spec 明确允许 alias/copy
  才能发生，否则记生命周期违约。

### 113.4 注册表建议文件

为避免把模型细节塞进通用 runtime，建议分层：

```text
native/runtime/memory_manifest.hpp/.cpp
native/models/h3_runtime/h3_memory_sites.h/.c
native/models/ltx_runtime/ltx_memory_sites.h/.c
native/models/flux2/memory_manifest.cpp
native/models/z_image/memory_manifest.cpp
```

H3/LTX 的 C header 只暴露常量字符串或小枚举，Objective-C++ bridge 负责映射到
`MemoryClass`。不要在 C 与 C++ 两边维护两份数值不同的 memory-class enum。

### 113.5 完整性测试

新增 `tests/native/test_memory_site_registry.py` 和一个纯 C++ registry test，至少断言：

1. 所有 required site id 唯一且稳定排序；
2. 每个 executable candidate 的 required site 都有 executable provenance；
3. `H3_GPU_MEMORY_UNKNOWN`/`UnknownExternal` 在首发正常请求结束时为 0；
4. 条件分支（preview/audio/input/LoRA）未被 candidate 允许时，不需要借用首发 manifest；
5. manifest 删除任意一个 required site 后，第一笔模型大分配前失败；
6. 修改 site class、upper 或 lifetime 后，plan digest 必须变化。

## 114. H3 首发路线的具体收尾 patch

H3 下一阶段应拆成四个小 PR：VAE GPU 分类、decoded-host 真实 commit、
conditioning/control host envelope、capability gate。不要把四项和 planner 重写放在同一
提交中。

当前工作树校准：前两项中的 classified upload、VAE required GPU 分类和 decoded actual
commit 已经开始落地；本节仍保留完整合同，作为构建验证、host requirements 纯函数和
故障测试的验收依据。它们尚未因为“代码已出现”自动视为完成。

### 114.1 增加 classified upload API

当前工作树已具有本节列出的三个 classified upload API。剩余工作是完成调用点审计、
构建/Metal hook 验证和 registry 对齐，而不是重复新增另一组 API。

`h3_gpu_tensor_new_classified` 目前只能创建未初始化 tensor；
`h3_gpu_tensor_from_f32` 会固定使用 `H3_GPU_MEMORY_UNKNOWN`。建议在
`h3_gpu.h/.m` 增加：

```c
h3_gpu_tensor *h3_gpu_tensor_from_f32_classified(
    h3_gpu *gpu, const float *values, size_t elements,
    h3_gpu_memory_class memory_class, const char *tag);

h3_gpu_tensor *h3_gpu_tensor_from_bf16_classified(
    h3_gpu *gpu, const uint16_t *values, size_t elements,
    h3_gpu_memory_class memory_class, const char *tag);

h3_gpu_tensor *h3_gpu_tensor_from_u32_classified(
    h3_gpu *gpu, const uint32_t *values, size_t elements,
    h3_gpu_memory_class memory_class, const char *tag);
```

实现直接委托已有 `h3_gpu_tensor_new_classified_internal`；旧 API 继续传
`H3_GPU_MEMORY_UNKNOWN`，保证默认 ABI 和行为不变。不要先调用 classified new 再做
额外 staging copy，否则同一 upload 会临时出现两个 GPU backing。

### 114.2 Video VAE 调用点迁移

当前 `h3_video_vae.c` 的 input upload、RoPE、activation arena 与 suffix scratch 已按
本节方向改为 classified API；仍需由静态 registry 和运行时 required-unknown 指标证明
首发 route 完整。

按当前 `h3_video_vae.c` 的函数，建议固定如下：

| 函数/对象 | class | site id | 说明 |
|---|---|---|---|
| `prepare_input_batch` 的 `vae->latent` | `ACTIVATION` | `h3.vae.decode.input_upload` | host rows 可由 host staging envelope 覆盖，GPU upload 单独计费 |
| `prepare_rope` 的 `rope_cos/sin` | `CONDITIONING` | `h3.vae.decode.rope_cos/sin` | decoder shape 生命周期；resident decoder cache 禁用时 stage end 释放 |
| `allocate_activations.post` | `ACTIVATION` | `h3.vae.decode.activation.post` | patch rows |
| `patch_hidden/hidden/norm` | `ACTIVATION` | `h3.vae.decode.activation.*` | sequence shape |
| `qkv/query/key/value/heads` | `ACTIVATION` | `h3.vae.decode.attention.*` | attention 峰值必须同时计入 |
| `branch/ff1/activated/projected` | `ACTIVATION` | `h3.vae.decode.mlp.*` | 不能只用最大单 tensor 代替 live set |
| `run_decoder` suffix zero | `CONVERSION_SCRATCH` | `h3.vae.decode.suffix_zero` | submit 完成后释放 |
| `run_resident_tile` suffix zero | `CONVERSION_SCRATCH` | `h3.vae.tile.suffix_zero` | tile command 完成后释放 |

`F32(field, elements)` 宏应改为带 site id 的显式表，避免所有 activation 共用一个 tag：

```c
#define H3_VAE_F32(field, elements, site)                                  \
    (vae->field = h3_gpu_tensor_new_classified(                            \
        vae->gpu, (elements), H3_GPU_F32,                                  \
        H3_GPU_MEMORY_ACTIVATION, (site)))
```

但建议最终删除宏，改为一个数组描述或逐行创建，因为调用点需要 checked multiplication
和精确错误信息。任何一个创建失败都必须走统一 `vae_context_cleanup()`，并让已经提交的
token exactly-once release。

### 114.3 Video VAE host 峰值从 `4x` 启发式转为资源公式

当前 `decoded_output_bytes * 4` 只能临时保护大致峰值。正式 manifest 应至少拆出：

```text
final_f32_bytes        = F * W * H * 3 * sizeof(float)
rgb8_bytes             = F * W * H * 3
temporal_overlap_bytes = overlap_frames * W * H * 3 * sizeof(float)
tile_input_bytes       = batch * latent_channels * tile_t * tile_h * tile_w * sizeof(float)
tile_output_bytes      = batch * tile_frames * tile_w_px * tile_h_px * 3 * sizeof(float)
tile_pointer_bytes     = tile_count * sizeof(float*)
tile_metadata_bytes    = tile_count * sizeof(h3_video_frames)
axis_metadata_bytes    = starts/overlaps 的 checked sum
```

对 chunked/tiled 路径，以 live interval 求峰值，不把互斥的两种策略相加。若当前 API
必须在 RGB8 转换前保留完整 F32，则 VAE→export 的峰值至少是
`final_f32_bytes + rgb8_bytes + conversion_scratch`；只有引入流式 encoder consumer 并
更改 last-use 后才能减少。

建议新增纯函数：

```c
int h3_video_vae_host_requirements(
    const h3_video_vae_shape *shape,
    const h3_video_vae_tiling *tiling,
    h3_video_vae_host_requirements_v1 *out,
    char *error, size_t error_size);
```

该函数只做 checked arithmetic，不创建 GPU 或 malloc。planner/probe 与实际 decode
共用它，防止计划公式和执行公式漂移。

### 114.4 decoded envelope 必须按真实输出 commit

当前 `h3.c` 已新增 checked frame-byte 计算并在 decode 后按真实 shape commit；本节中
“应修改”的代码示意保留为合同。尚未完成的是用资源公式替换预测值 4x 的启发式 upper，
以及证明聚合 envelope 覆盖实际 tile/chunk live set。

当前 reservation 按预测 `temporal.frame_count/render_width/render_height` 计算，decode
成功后 commit 也使用预测值。应在 `frames` 返回后重新做 checked multiplication：

```c
uint64_t actual_f32_bytes = 0;
if (!h3_checked_frame_bytes(frames.frames, frames.width, frames.height,
                            3, sizeof(float), &actual_f32_bytes)) {
    h3_set_error(ctx, "h3_host_allocation_overflow: decoded frames");
    goto cleanup;
}
if (actual_f32_bytes > decoded_host_upper_bytes) {
    h3_set_error(ctx, "memory_envelope_violation: decoded frames exceed plan");
    goto cleanup;
}
if (!h3_commit_host_envelope(..., actual_f32_bytes, ...))
    goto cleanup;
```

`decoded_host_upper_bytes` 必须保存 reservation 时的完整 upper，而不是只保存
`final_f32_bytes`。如果使用聚合 envelope，commit 的 actual 应定义为“当前实际持有的
聚合 backing 上界”而不是假装只有最终 frames；更推荐把最终 F32 逐 allocation 纳管，
其余 tile scratch 用独立 stage envelope。无论采用哪种方式，manifest、账本和 result
字段必须使用相同定义。

### 114.5 H3 conditioning/control 首发 envelope

首发 T2V 虽然排除了 reference/audio output，仍有文本 embedding、token tags、layout、
sigma schedule、key strings 和缓存 key。短期可先用一个经过验证的 stage envelope，
避免一次性迁移几十个小 malloc：

```text
h3.host.text_conditioning_v1
  live: text_encode_begin -> denoise_conditioning_last_use
  includes:
    text.values capacity
    token tags/row maps
    layout positions/segments
    sigma schedule
    conditioning/prepared/resident/decoder key capacity
    allocator alignment margin
```

实现建议：

1. `h3_generate()` 进入文本准备前 reserve；
2. 使用 manifest 对 `max_tokens + width/height/frames` 求 upper；
3. conditioning 构造完成后按真实 `capacity` 汇总 commit；
4. 受限模式已禁用 conditioning/DiT cache，因此 denoise last-use 后 release；
5. 任一内部 `realloc` 超过 envelope upper 时 fail-closed，不允许无界增长。

长期应把大于 `individual_guard_threshold_bytes` 的 `text.values`、video/audio rows 单独
迁移到 `h3_accounted_host_malloc`，只让字符串、索引和错误缓冲留在 control envelope。

### 114.6 H3 缓存与 decoder 策略

受限 H3 session 当前关闭主 cache，但 `h3_cache_set_decoder_enabled()` 仍由 resident
residency 决定。candidate 已规范化为 streamed，因此首发自然关闭 decoder cache；
仍应在执行入口加显式断言，避免未来 residency 规则变化：

```cpp
require(!r.memory_constrained.enabled ||
            (!h3_cache_enabled(context) &&
             !h3_decoder_cache_enabled(context)),
        "memory_lifetime_violation: constrained H3 cache enabled");
```

不要在 constrained request 完成后把 GPU/host token 绑定的 decoder 保存到 session。
将来若开放 cache，必须使用 `StorageLease::cache()`、cache quota 和新 request generation
的显式 reactivate，而不是复用旧 callback token。

### 114.7 H3 patch 验收

建议新增/扩展：

```text
tests/native/h3_gpu_memory_hooks_test.mm
  - classified upload class/tag
  - reserve denied before MTLBuffer
  - commit actual > upper
  - partial VAE allocation cleanup

tests/native/h3_host_memory_hooks_test.cpp 或 Objective-C++
  - latent/RGB8 reserve/commit/release
  - decoded envelope cancel on decode failure
  - real frame bytes commit
  - stale generation commit rejected
  - retain/preview rejected before large allocation

tests/native/test_h3_memory_manifest.py
  - required site completeness
  - text/control upper checked arithmetic
  - manifest missing site -> execution unsupported
```

H3 `AllocationGuarded` 的退出条件是：首发正常请求结束时 GPU/host
`required_unknown_bytes=0`，所有 token 清零，且没有靠最终 footprint 才发现的未归因
大峰值。达到该条件仍需 validated envelope 和 L3 campaign 才能发布。

## 115. LTX 首发路线的具体收尾 patch

LTX 当前受限路线的 denoiser 已固定 C/Metal，但 Stage1→upsample→Stage2→Video VAE 并非
全部原生 Metal：upsampler/Video VAE 仍可能进入 MLX/MPSGraph。首发能力应明确命名为
“C/Metal denoiser + bounded MLX component envelope”，不能简称“纯 native”。

### 115.1 route 不变量

建议在 `ltx_session.mm` 将分散条件收敛为一个纯函数，并由 planner、service、session
共同调用：

```cpp
struct CandidateMatch {
    bool matched = false;
    std::string candidate_id;
    std::string rejection_code;
    std::string rejection_detail;
};

CandidateMatch match_ltx_c_metal_streamed_video_v1(
    const Request& request) {
    // video.generate, GPU, c_metal, streamed, no input/audio/LoRA
}
```

执行入口必须再次断言：

```text
request.memory_constrained.enabled == true
request.ltx_backend == c_metal
request.execution == gpu
request.residency == streamed
uses_parent_mlx(request) == false
use_mlx_denoiser == false
image_to_video == false
audio == false
inputs.empty && loras.empty
```

这里 `uses_parent_mlx=false` 只表示 TurboCider 不把整个请求交给 parent MLX worker，不
表示后续 VAE helper 内部没有 MLX。plan/result 要分别输出：

```text
denoiser_backend = c_metal
upsampler_backend = native_or_mlx（必须冻结为 candidate 字段）
video_vae_backend = mlx_in_process
parent_worker_backend = in_process_native_session
```

否则“route 看似 C/Metal、实际 decode 用 MLX”的 graph envelope 无法审计。

### 115.2 host vector 必须绑定实际 backing

当前 stage latent、planar pixels、RGB 已先 reserve 再构造 vector，并以
`capacity*sizeof(T)` commit，这是正确方向；但 dynamic Gemma 的 `raw_video/raw_audio/
raw_mask`、connector 的 connected vectors 和 clean-prefix vectors 仍需确认被
conditioning envelope 完整覆盖。

建议将 `ltx_host_vector_upper` 和 `ltx_host_vector_capacity_bytes` 收敛成一个 RAII 类型：

```cpp
template<class T>
class AccountedHostVector {
public:
    static AccountedHostVector allocate(
        MemoryAdmission* admission, MemoryClass cls, size_t count,
        std::string site_id, uint64_t domain, uint64_t generation);

    std::vector<T>& value();
    const std::vector<T>& value() const;
    uint64_t committed_bytes() const;
    void release_storage_at_safe_point();
private:
    std::vector<T> value_;
    StorageLease lease_;
};
```

Darwin 上分配前 upper 可用 `malloc_good_size(requested_bytes)`，commit 后实际容量优先用
`malloc_size(vector.data())`；不可用时才退到 `capacity*sizeof(T)` 加 manifest alignment
margin。必须检查：

```text
requested <= capacity*sizeof(T) <= actual_allocator_capacity <= reserved_upper
```

不要对 `std::vector` 的 move/swap 假设 lease 自动跟随；RAII wrapper 必须把 backing 和
lease 一起 move。当前 `video.swap(upsampled)` 后手工 release Stage1 lease 的模式容易在
后续重构中错配，建议改成 `AccountedHostVector::swap_backing_and_lease()`。

### 115.3 conditioning overlap 的精确公式

当前 conditioning envelope 以“两份最大 conditioning + margin”覆盖 raw 与 connected
短时重叠。正式 manifest 至少拆出：

```text
raw_video    = raw_rows * kLtxVideoDim * sizeof(bf16)
raw_audio    = raw_rows * kLtxAudioDim * sizeof(bf16)
raw_mask     = raw_rows * sizeof(bf16)
connected_*  = padded_rows 对应三项
connector_gpu_scratch
gemma_output_staging
cache_serialization_scratch（若写 cache）
```

受限首发建议禁止 `write_ltx_conditioning_cache()`，因为 cache serialization 和文件写入
不是生成数学的一部分，却会增加 host buffer、文件缓存和跨请求生命周期。若保留写
cache，必须作为 `optional` reservation：yellow/red 水位可跳过，不能因 cache 写失败让
已完成 conditioning 的请求失败。

### 115.4 Stage1→Stage2 latent 交接

当前代码明确保留 Stage1 video 与 Stage2 `upsampled` 两份直到 upsample 完成，这是正确
峰值。planner 的 interval 应为：

```text
stage1_video:   denoise1_begin ---------------- upsample_complete
stage2_video:                         upsample_begin ---------------- vae_read_done
overlap:                              [upsample_begin, upsample_complete]
audio_latent:   denoise1_begin ---------------- denoise2_complete（video-only 后释放）
conditioning:   text_complete ----------------- denoise2_complete
```

`video.swap(upsampled)` 之后：

1. 验证 `video` 的 lease 已切换到 Stage2 backing；
2. 销毁现在位于 `upsampled` 中的 Stage1 vector；
3. 释放 Stage1 lease；
4. 才能开始 Stage2 pager optional prefetch。

该顺序需通过 fake vector/backing id 测试，不能只检查 bytes 最终为零。

### 115.5 Video VAE/upsampler MLX envelope

`kLtxVideoVaeGraphEnvelopeBytes = 2 GiB` 只能是
`HeuristicNotExecutable`。建议给 MLX 组件增加只读 probe：

```c
typedef struct {
    uint64_t weight_bytes_exact;
    uint64_t input_bytes_exact;
    uint64_t output_bytes_exact;
    uint64_t graph_temporary_upper;
    uint64_t allocator_cache_upper;
    uint64_t compile_temporary_upper;
    uint32_t envelope_revision;
} ltx_mlx_component_requirements;

int ltx_mlx_video_vae_probe_requirements(...);
int ltx_mlx_upsampler_probe_requirements(...);
```

probe 只读取 checkpoint index/config 与已安装的 calibration table，不调用
`mx::load_safetensors` 或 `mx::eval`。执行时由 `ScopedMlxMemoryContext` 记录：

```text
active_memory_before/after/peak
cache_memory_before/after/peak
process_footprint_before/after/peak
manifest graph/compile upper
```

如果 MLX 版本没有可靠的 allocator hard limit API，也不能伪造逐 allocation guard；
首发使用 validated whole-component envelope，在进入组件前一次性 reserve，并在 stage
完成后 `clear_cache + destroy arrays + checkpoint`。任何观测超过 envelope 都使该
manifest 失效。

### 115.6 helper/exec 子进程策略

当前 constrained streamed 路径应默认禁用 spawned helper 和 `exec` finalizer，直到
parent/child reservation 协议完成。未来若启用：

```text
parent reserve ChildProcessEnvelope
parent serialize frozen plan/digest/generation
child ACK baseline + accepted ceiling
parent release in-process component backing
child materialize VAE
child return terminal metrics/output ack
parent release child envelope
```

不能同时让 parent 按完整 VAE envelope 计费、child 又创建独立 B；两者必须共享一个
process-tree root budget。child 崩溃或超时后，parent 先确认 PID/IPC 终止，再 release
envelope。`exec` 替换进程会丢失内存中 ledger，因此需要把 plan、已持有 output/latent
大小和 generation 写入受控 handoff 文件或 pipe，并由新进程重建账本。

### 115.7 LTX 收尾测试

至少增加：

- dynamic Gemma raw/connected 双份峰值测试；
- vector move/swap 后 StorageId 与 lease 匹配测试；
- Stage2 reservation 失败时 Stage1 仍可安全清理；
- VAE graph reservation 失败时不创建 `ltx_mlx_video_vae`；
- constrained request 在 `TURBOCIDER_LTX_MLX=1` 下仍不走 MLX denoiser；
- helper/exec 环境变量不能绕过 candidate；
- `pixels`→`rgb` 双份输出峰值和 last-use release；
- cancel 发生在 Gemma、Stage1、upsample、Stage2、VAE、RGB、export 各点后，ledger 回到
  baseline，pending=0。

## 116. Manifest Capability 硬门禁与 `execution_supported` 修正

历史上最重要的 planner 风险是：`route_available && estimate_fits` 会自动把
`execution_supported` 设为 true。该逻辑已在当前工作树删除；本节保留风险背景和后续
registry 实现合同，防止未来重构时回退。

### 116.1 CapabilityRecord

建议新增：

```cpp
struct CapabilityRecord {
    std::string candidate_id;
    MemoryCapabilityLevel level;
    std::string model_identity;
    std::string checkpoint_digest;
    std::string backend_revision;
    std::string runtime_revision;
    std::string os_compatibility;
    std::string device_family;
    std::string shape_bucket;
    std::string dtype;
    std::vector<uint32_t> certified_refill_slots;
    std::vector<std::string> required_site_ids;
    std::string manifest_digest;
    std::string evidence_digest;
    bool enabled_for_release = false;
};
```

该记录由只读 capability registry 提供。生产 registry 应由代码随版本发布或加载只读
签名/哈希绑定文件；不能由 request 自行提供 `enabled_for_release=true`。

### 116.2 新的开放条件

`plan.cpp` 改为：

```cpp
auto match = match_candidate(r);
policy.route_available = match.matched;
if (!match.matched) {
    policy.reason = match.rejection_detail;
} else {
    auto manifest = load_memory_manifest(match.candidate_id, checkpoint);
    auto capability = capability_registry.lookup(
        match.candidate_id, checkpoint, request_shape, runtime_identity);
    auto estimate = planner.plan(manifest, request_shape, policy);
    finalize_memory_policy_estimate(policy, estimate.upper, estimate.complete);

    const bool allocation_guarded =
        capability && capability->level >=
            MemoryCapabilityLevel::AllocationGuarded;
    const bool envelope_validated =
        capability && capability->level >=
            MemoryCapabilityLevel::EnvelopeValidated;
    const bool identity_matches =
        capability && capability->manifest_digest == manifest.digest &&
        capability->checkpoint_digest == checkpoint.digest;

    policy.execution_supported =
        estimate.complete && estimate.fits && allocation_guarded &&
        envelope_validated && identity_matches &&
        capability->enabled_for_release;
}
```

若希望在开发阶段运行未 release 的路线，使用编译期或测试 harness 的
`experimental_allow_unreleased_capability`，结果必须标 `experimental_unqualified`，且
默认生产构建关闭。环境变量只能关闭已认证能力，不能创建能力记录。

### 116.3 estimate 状态与 execution 状态分离

建议 policy/result 使用两个枚举：

```cpp
enum class EstimateState {
    Unavailable, Heuristic, CompleteExactOrValidated, ExceedsBudget
};

enum class ExecutionState {
    Unsupported, PlanOnly, ExperimentalGuarded, Certified
};
```

典型映射：

| 情况 | estimate | execution |
|---|---|---|
| 没有 manifest | Unavailable | Unsupported |
| 4 GiB/2 GiB 常量 | Heuristic | PlanOnly |
| required site 完整、allocator guard 完成、未校准 graph | CompleteExactOrValidated/部分 | ExperimentalGuarded |
| manifest/evidence/identity 全命中 | CompleteExactOrValidated | Certified |
| upper > B | ExceedsBudget | Unsupported（错误为 budget too small） |

`require_memory_constrained_execution_supported` 的错误优先级应为：

```text
invalid/conflict
 -> route unsupported
 -> manifest/estimate unknown
 -> estimate exceeds budget
 -> capability not certified
 -> system admission conflict
```

这样不会把“没有可靠估计”错误地报告成“预算太小”。

### 116.4 checkpoint identity 与 TOCTOU

probe 时记录每个权重文件：canonical path、device/inode、size、mtime、header/index hash、
必要时完整 SHA。正式 load 前再次 `fstat`；大文件完整 SHA 可由离线 install 阶段生成，
运行时验证 sidecar + size/header hash。若身份变化：

```text
memory_manifest_stale
 -> 不创建 GPU device/buffer
 -> 清除 candidate execution_supported
 -> 要求重新 probe/calibrate
```

不要在已开始 allocation 后“重新估一下”并继续，因为原 plan digest 已失效。

### 116.5 contract test 修改建议

当前静态测试已经禁止 H3/LTX heuristic candidate 出现 `execution_supported=true`；
后续 manifest registry 实现后继续扩展为：

1. 默认无 verified manifest fixture：`route_available=true`、
   `execution_supported=false`、`ExecutionState::PlanOnly`；
2. 注入完整 test manifest + capability：true；
3. 修改 checkpoint digest、shape、runtime revision、slot 数任一字段：false；
4. 仅把 estimate 调小：仍 false；
5. 删除 required site：`memory_estimate_unknown`；
6. `B=upper-1`：`memory_budget_too_small`，且没有创建大 backing。

## 117. Scheduler 事务接口与运行时重规划

现有代码主要由 session 自己串联阶段。为了逐步接入而不重写所有模型，建议先实现一个
薄的事务调度层：它管理阶段 scope、reservation 和 safe point，但不接管模型数学。

### 117.1 最小接口

```cpp
class ConstrainedScheduler {
public:
    StageToken enter_stage(StageId stage, const StagePlan& plan);
    void checkpoint(StageToken&, SafePointKind);
    PrefetchPermit request_prefetch(StageToken&, const ResourceSpec&);
    void retire(StageToken&, StorageLease&&, CompletionTicket);
    void leave_stage(StageToken&);
    void cancel_and_drain(std::string reason) noexcept;
    ScheduleSummary summary() const;
};
```

`StageToken` 是 move-only，析构时若阶段未正常离开则进入 cancel/drain；但析构不能阻塞
任意回调线程。owner session 仍负责调用 GPU wait/event，scheduler 只记录 pending ticket
并验证关闭条件。

### 117.2 StagePlan

```cpp
struct StagePlan {
    StageId id;
    uint64_t fixed_floor_bytes;
    uint64_t optional_prefetch_bytes;
    uint64_t maximum_stage_upper;
    std::vector<ResourceInterval> resources;
    std::vector<FallbackAction> fallback_ladder;
    bool may_overlap_next_stage = false;
};
```

`fixed_floor_bytes` 不是另一次 reservation；它是 planner 对 resources 的聚合结果。运行时
仍逐 allocation reserve。`maximum_stage_upper` 用于进入阶段前快速 fail，不能替代底层
guard。

### 117.3 三条队列和 ticket 状态

```text
I/O queue:       EMPTY -> RESERVED -> READING -> READ_READY
convert queue:   READ_READY -> CONVERTING -> GPU_READY
GPU queue:       GPU_READY -> SUBMITTED -> COMPLETED -> RECLAIMABLE
```

每个 `PrefetchTicket` 包含：site/instance、slot id、upper/actual、generation、file identity、
I/O error、GPU event、取消标记。状态转移只允许单向 CAS；cancel 不直接把 SUBMITTED 改成
RECLAIMABLE，而是停止后续提交并等待 completion。

### 117.4 look-ahead 决策

在 block safe point 计算：

```text
headroom = B - (ledger.committed + ledger.reserved)
next_required = next block/group required upper
optional_window = min(candidate.max_window, user.max_refill_slots)
```

决策：

1. `headroom < next_required`：不预取，若当前计划没有更保守 revision 则 abort；
2. pressure yellow 或 unattributed 上升：窗口最多 1，不增加 pinned；
3. green 且连续两个 block `gpu_wait_for_weight > target`，且新 slot upper 可 reserve：窗口
   加 1；
4. 连续三个 block I/O 完全隐藏且水位高于 0.80B：窗口减 1，避免无收益驻留；
5. runtime window 不得超过 capability certified slots。

调节只影响 optional prefetch，不改变当前/下一必需 block 的可执行性。pinned prefix 首发
只允许在 denoise step 边界缩小，不在线增大；增大需要重新计算 live-set 并生成 schedule
revision。

### 117.5 阶段 handoff

统一 handoff：

```cpp
scheduler.checkpoint(current, SafePointKind::StageComputeComplete);
current.stop_optional_prefetch();
current.drain_io_and_gpu();
current.release_dead_resources();
admission.checkpoint("stage.<name>.released");
require(current.required_active_leases() == expected_cross_stage_leases);
scheduler.leave_stage(current);
scheduler.enter_stage(next, next_plan);
```

跨阶段只允许 manifest 列出的 conditioning/latent/output lease。Transformer weight/cache
若不在列表中，VAE 进入前仍 active 即 `memory_lifetime_violation`，而不是依赖系统压力
再回收。

### 117.6 观测超线处理

若 `process_footprint > B`：

1. 当前 run 立即标 `budget_violation_observed=true`；即使后续降回 B 也不能改成 PASS；
2. 停止 optional prefetch 和 cache 写；
3. 在最近 safe point drain；
4. 保存最近 N 条 allocation/trace 事件；
5. 返回 `memory_observed_over_budget`；
6. 使当前 manifest/capability 进入 quarantine，后续默认 plan-only，直到重新校准。

不要在异步 sampler 线程直接释放模型对象。sampler 只更新原子 pressure/violation flag。

## 118. H3、LTX、Flux、Z-Image 的统一接入模板

四类模型的内部结构不同，但对调度器应提供相同六步合同。

### 118.1 Adapter 接口

```cpp
class MemoryConstrainedAdapter {
public:
    virtual CandidateMatch match(const Request&) const = 0;
    virtual ModelMemoryProbe probe(const Request&, const ModelIdentity&) = 0;
    virtual ResidencyPlan plan(const Request&, const ModelMemoryProbe&,
                               const EffectiveMemoryPolicy&) = 0;
    virtual void bind(MemoryAdmission&, ConstrainedScheduler&) = 0;
    virtual RunResult run(const Request&, const ResidencyPlan&,
                          const Event&, std::atomic<bool>& cancel) = 0;
    virtual void unbind_after_drain() noexcept = 0;
};
```

默认 session 不实现该接口；add-on 开启且 adapter 缺失时 fail-closed。disabled 请求继续
直接调用旧 `ModelSession::generate`，不先构造 adapter 或 probe。

### 118.2 模型映射

| 模型 | 首个 adapter | component stage | block strategy | decoder strategy | 当前 gate |
|---|---|---|---|---|---|
| H3 Turbo | `h3_c_metal_t2v_streamed_v1` | text→DiT→Video VAE→export | DiT 双 refill slot + pinned prefix | Video VAE chunk/tile，首发无 cache/preview | HookBridged，待 VAE/host/manifest |
| LTX 2.5 | `ltx_c_metal_t2v_streamed_v1` | Gemma→Stage1→upsample→Stage2→VAE→export | native pager 认证 slots | MLX Video VAE whole-component envelope | 待 vector/graph envelope/L3 |
| Flux | `flux_component_staged_t2i_v1` | text→transformer→VAE→export | 首版不做 block stream | VAE tile only if quality-exact | plan-only |
| Z-Image safetensors | `zimage_staged_t2i_v1` | text→transformer→VAE→export | 首版不做 block stream | exact tile/whole component | plan-only |
| Z-Image GGUF | 独立 candidate | 同上 | 按 quant load group | format-specific dequant envelope | unsupported |

### 118.3 Flux 代码改造顺序

基于 `native/models/flux2/pipeline.cpp` 当前 `Flux::load()` 同时加载 transformer 与 VAE，
建议：

1. 在 `flux.hpp` 增加组件状态枚举和 `load/release_component`；
2. 将 `Flux::load()` 保留为 disabled 路径的组合调用；
3. constrained `generate()` 禁止调用旧 `load()`，而是 strict handoff；
4. `conditioning()` 的 Qwen weights/cache 纳入 Text stage；prompt cache 默认 request-local；
5. transformer `Weights::load/materialize/cast/pack_convrot_q8` 分别标记 weights、
   conversion scratch 和 cache；
6. `flux_transformer.cpp` 后续按 dual/single block 的实际共享依赖定义 `load_group`，不能
   假设每个 C++ block 对应一个独立权重分层；
7. VAE 的 full image、tile、RGB output 同时存活集合进入 manifest。

首个 Flux candidate 不支持 input/edit/LoRA/ANE/hybrid。LoRA 即使是离线预合并，也需要
checkpoint identity 指向合并后 artifact；运行时 merge 会产生额外 dense copy，另建
candidate。

### 118.4 Z-Image 代码改造顺序

基于 `native/models/z_image/z_image.cpp` 与 `gguf.cpp`：

1. 将 text/transformer/VAE 的 load/clear 拆成组件 API；
2. safetensors 与 GGUF probe 分开，manifest 不共用 quant bytes；
3. `Weights::load`、BF16 cast、GGUF dequant/pack 的 source/destination overlap 进入
   conversion interval；
4. prompt conditioning cache 默认 request-local；
5. refiner/main 同时存在时以两个 transformer component 计，不复用单模型 envelope；
6. GGUF 的 atomic `load_group` 以 kernel 一次需要的 tensor 集合定义；
7. VAE output 与导出 queue 逐 allocation guard 后才开放 execution。

首个 safetensors candidate 只做 text-to-image、无 input/LoRA、GPU、component-staged。
GGUF 在每种量化格式有独立 graph/dequant evidence 前保持 unsupported。

### 118.5 接入新模型的 DoD

新增模型不能只提交 candidate matcher。最小 PR 集合为：

```text
[ ] model-specific site registry
[ ] metadata-only probe + identity validation
[ ] stage/resource interval manifest
[ ] disabled path ABBA
[ ] constrained route matcher + execution-time duplicate gate
[ ] allocator/host/graph coverage or fail-closed envelope
[ ] cancel/error cleanup tests
[ ] L2 allocator evidence
[ ] L3 shape/checkpoint evidence before release whitelist
```

## 119. 故障注入、测试工具与证据文件

验收要能主动证明失败发生在正确位置，而不是只测试成功路径。

### 119.1 统一 fault injector

建议只在测试构建增加：

```cpp
struct MemoryFaultRule {
    std::string site_id;
    uint32_t fail_on_reserve_n = 0;
    uint32_t fail_on_allocate_n = 0;
    uint32_t fail_on_commit_n = 0;
    uint32_t delay_completion_n = 0;
    uint64_t force_actual_bytes = 0;
};
```

规则由测试直接注入对象，不从生产环境变量读取。每次 fault run 必须断言：

```text
terminal error code 正确
没有后续 site allocation
reservation_count == 0
pending_release_count == 0（drain 后）
storage_count 回到 baseline 允许值
generation 未被旧 callback 修改
partial output 被删除或标记 invalid
下一次普通/受限请求可正常运行
```

### 119.2 必测故障位置

| 阶段 | 故障 | 预期 |
|---|---|---|
| plan/probe | manifest missing/stale/overflow | 第一个模型大分配前拒绝 |
| reserve | budget 恰少 1 byte | `memory_budget_too_small`，allocator 未调用 |
| platform allocate | malloc/MTLBuffer 返回空 | cancel reservation，错误稳定 |
| commit | actual > upper/stale generation | 销毁 backing，cancel 或隔离 token，worker tainted |
| I/O | short read/EINTR/EOF/checksum mismatch | ticket failed，未提交 GPU，slot 可回收 |
| GPU | completion 延迟/错误 | lease 保持 pending，drain 后 release |
| stage handoff | dead cache 未释放 | 进入下一阶段前 lifetime violation |
| VAE | graph peak 超 envelope | run FAIL、manifest quarantine |
| export | encoder/mux 失败/取消 | output lease 在文件关闭后 release，partial 删除 |

### 119.3 evidence bundle 目录

建议工具输出：

```text
results/memory-constrained/<candidate>/<checkpoint>/<run-id>/
  request.json
  environment.json
  model-identity.json
  manifest.json
  plan.json
  result.json
  trace.jsonl
  memory-samples.jsonl
  vm-stat-before.json
  vm-stat-after.json
  quality.json
  stdout.log
  stderr.log
  verdict.json
```

`verdict.json` 只由工具根据固定规则生成，人工备注放 `notes.txt`，不能手工把 FAIL 改为
PASS。目录名不包含 prompt 或用户路径。大模型输出本身默认不复制进仓库，只记录 hash
和外部受控 artifact URI。

### 119.4 campaign runner

新增 `tools/native/benchmark_memory_constrained.py`，最小参数：

```sh
python3 tools/native/benchmark_memory_constrained.py \
  --request request.json \
  --budgets 12GiB,16GiB,24GiB \
  --buffers 10,15,20 \
  --cold-runs 3 --warm-runs 5 \
  --output results/memory-constrained/...
```

runner 应：

1. 保存 immutable request/plan；
2. 采样进程 footprint、system available、pressure、swap counters；
3. 每轮使用新 job generation；cold run 可使用新进程，warm run 固定 session；
4. 记录外部活跃进程/pressure，污染轮次标 `CONTAMINATED`；
5. 自动生成 minimum-1、minimum、headroom、resident baseline；
6. 计算 wall、stage time、I/O overlap、GPU starvation、quality；
7. 返回非零退出码，只要存在未豁免 FAIL/INCOMPLETE。

### 119.5 本机测试与真实低内存机器分工

64 GiB 开发机上设置 Y=8/12/16 GiB 可以验证 planner、ledger、allocation failure、
窗口收缩和默认路径隔离，但不能模拟较小机器的内存带宽、file cache、压缩器行为和
系统后台负载。发布前至少需要：

```text
开发机：synthetic budget sweep + fault injection + ABBA
目标低内存机：minimum/minimum+headroom + cold/warm + swap/pressure
目标常规机：disabled 默认性能/质量回归
```

实机 evidence 必须记录 physical memory；不得只在 64 GiB 机器上通过人为 B 后写
“16 GiB Mac certified”。

## 120. 推荐实现顺序与最终验收门禁

### 120.1 下一组可审查 patch

按风险和依赖排序：

| Patch | 内容 | 完成后等级变化 |
|---|---|---|
| M1 | H3 classified upload API + VAE required tensor 分类 | H3 GPU unknown 显著收敛 |
| M2 | H3 decoded actual commit + host requirements 纯函数 | host envelope 可审计 |
| M3 | H3 text/conditioning/control envelope + fault tests | H3 接近 AllocationGuarded |
| M4 | LTX AccountedHostVector + conditioning/vector 审计 | LTX host backing 闭环 |
| M5 | LTX MLX upsampler/VAE probe + validated envelope | LTX 可进入 EnvelopeValidated 候选 |
| M6 | CapabilityRecord + plan 硬门禁 | 消除 heuristic 自动执行风险 |
| M7 | Scheduler stage token/safe-point 薄层 | 统一 handoff/cancel/pressure |
| M8 | campaign runner + H3/LTX L2/L3 evidence | 决定是否进入 release whitelist |
| M9 | Flux component-staged adapter | 独立 plan-only→guarded 路线 |
| M10 | Z-Image safetensors adapter；GGUF 后续独立 | 独立 plan-only→guarded 路线 |

M1–M5 可以保留当前实验执行开关，但 M6 合入后默认没有 verified capability 的路线必须
回到 plan-only。不要为了保持临时 demo 可运行而跳过 M6。

当前进度校准：M1 的 API/VAE 分类和 M2 的 decoded actual commit 已部分进入工作树，
但 M1 仍缺 H3 DiT 剩余 reachable audit 与新一轮构建验证，M2 仍缺 host requirements
纯函数和 validated upper；因此二者均不能标记 complete。

### 120.2 每个 patch 的提交说明模板

```text
Scope:
  candidate / stage / allocation sites

Default-path impact:
  none / measured delta and reason

Memory contract:
  new/changed site ids, classes, uppers, lifetimes

Failure behavior:
  reserve/allocate/commit/cancel/cleanup paths

Tests:
  commands and PASS/SKIP counts

Evidence:
  manifest/evidence digest and artifact directory

Remaining unsupported:
  explicit request variants and reason
```

### 120.3 发布前硬门禁

某条 H3/LTX candidate 只有同时满足下列条件才能 `enabled_for_release=true`：

```text
[ ] route matcher 与 session duplicate gate 完全一致
[ ] metadata probe 在任何大 allocation 前完成
[ ] manifest identity 与 checkpoint/runtime/device/shape 精确命中
[ ] required site registry 完整，required_unknown_bytes == 0
[ ] 每个大 GPU/host allocation reserve-before-allocate
[ ] graph/private allocator 有 validated envelope
[ ] async backing completion-safe release
[ ] stage handoff 只保留声明的 cross-stage lease
[ ] minimum-1 预分配拒绝；minimum/headroom 稳定成功
[ ] observed footprint 无越 B；安静环境 swapouts delta == 0
[ ] cold/warm/cancel/fault campaign 全部通过
[ ] 输出质量与 disabled baseline 满足合同
[ ] disabled ABBA route/cache/output 不变，稳态性能回归 <= 2%
[ ] evidence bundle 可重放且无污染
```

如果任何一项缺失，正确状态是 `PlanOnly`、`ExperimentalGuarded` 或明确 unsupported，
不是放宽预算、依赖 swap 或回到 resident。文档顶部仍保持 `Proposal + scaffolding`，直到
至少一条 H3 与一条 LTX 目标路线分别达到 L3 并进入精确 capability whitelist。

### 120.4 本轮文档校准时的验证记录

以下命令已在当前工作树执行，用于确认本文引用的接口和静态合同仍成立：

```sh
MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" \
TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh

python3 -B tests/native/test_contract.py
python3 -B tests/native/test_memory_accounting.py
python3 -B tests/native/test_h3_gpu_memory_hooks.py
python3 -B tests/native/test_ltx_gpu_memory_hooks.py
python3 -B tests/native/test_h3_streaming_policy.py
python3 -B tests/native/test_ltx_streaming_benchmark.py
```

结果：native build 通过；contract 81 项通过、1 项因缺少 Wan fixture 跳过；memory
accounting、H3 shared residency policy、LTX streaming benchmark 通过；H3/LTX Metal
hook 测试因当前测试进程无可用 Metal device 明确 SKIP。该记录只证明代码可编译、纯
逻辑/静态合同通过；它不提升 capability 等级，也不替代有 Metal 设备的 L2 与真实
低内存机器上的 L3。

本次文档扩展后已对包含 H3 classified upload、Video VAE/text encoder/DiT common-path
分类和 decoded actual commit 的最新工作树重跑上述命令：native build 通过；contract
81 项通过、1 项因缺 Wan fixture 跳过；memory accounting、H3 shared residency policy、
LTX streaming benchmark 通过；H3/LTX Metal hook 因当前测试进程无可用 Metal device
明确 SKIP。该结果仍然只属于 L0/L1 和 build 证据，不能替代 Metal L2、真实 checkpoint
L3、目标低内存机器或 swap campaign。

## 121. 面向实现的最终蓝图与默认路径隔离

本节开始给出“可以直接拆任务、写接口、做 code review”的规范。它不是新增第二套架构，
而是把前 120 节压缩成一条可执行主线。若本节之后的接口与前文示意代码冲突，以本节
之后为准；若当前源码尚无对应类型，则均标为“拟增”。

### 121.1 当前工作树事实与剩余缺口

截至本次源码复核，当前实现可按下表理解：

| 层级 | 已在工作树中的事实 | 仍缺少的发布条件 |
|---|---|---|
| 请求/config | `MemoryConstrainedConfig`、profile overlay、Y/X/B、冲突校验、GPU-only 约束 | schema version、capability digest 与错误合同的最终固化 |
| plan | H3/LTX candidate normalization、粗粒度 estimate、plan JSON；无 manifest 时固定 plan-only | 接入 verified manifest、精确 peak 和 release whitelist |
| job accounting | `MemoryLedger`、`MemoryAdmission`、baseline/footprint 观测 | scope/site/lifetime、required unknown、异步 retire 完整性与 stage token |
| H3 bridge | GPU/host hooks、同 generation/domain、VAE options 传播 | conditioning/control host、剩余 DiT 调用点、全路径 unknown=0 |
| H3 GPU | classified upload API；VAE、text encoder、DiT common path 和 Euler previous velocity 已分类 | 量化/可选分支分类或证明不可达 |
| H3 host | latent、RGB8、decoded envelope；真实 frame bytes commit | 从启发式 4x envelope 迁移到 shape 公式和实际 backing |
| LTX native GPU | 集中 `ltx_gpu_buffer` 分配已能 reserve/commit/release | Video VAE/upsampler direct `MTLBuffer`、MPSGraph temporary、剩余 unknown |
| LTX host | 若干阶段级 reservation | `malloc`/`vector`/conditioning backing 与 helper 子进程闭环 |
| Flux/Z-Image | 当前 MLX resident/component-staged 代码可作为适配起点 | 没有统一 pager、逐阶段 envelope 或可执行 capability |

因此近期目标不是“把所有模型打开”，而是先建立如下严格顺序：

~~~text
route matcher
  -> metadata-only probe
  -> verified capability lookup
  -> deterministic plan compile
  -> job admission
  -> stage/block allocation guard
  -> runtime evidence
~~~

其中任一步失败都必须在下一笔模型大分配之前结束请求。

### 121.2 默认路径的结构性隔离

不能仅依赖每个热点函数内部的 `if (memory_constrained)`，否则即使关闭 add-on 也会
引入原子计数、字符串 tag、锁、footprint syscall 或不同的 cache 生命周期。建议在
`native/runtime` 和 `ModelSession` 边界建立一次性分叉：

~~~cpp
RunResult execute_request(ModelSession &session, const ExecutionPlan &plan, ...) {
    if (!plan.memory_policy || !plan.memory_policy->enabled)
        return session.generate(plan.request, event, cancelled); // 原路径

    auto context = RequestMemoryContext::admit(plan, device_snapshot);
    MemoryAdmissionBinding binding(session, context.admission());
    return session.generate_constrained(plan.request, context, event, cancelled);
}
~~~

若不希望立刻增加第二个 virtual，可保留现有 `generate`，但必须保证：

1. `set_memory_admission(nullptr)` 时所有 native callback 指针均为空；
2. disabled 请求不创建 `MemoryAdmission`、`StageScheduler`、trace ring 或 sampler；
3. disabled 请求不禁用原缓存、不改变 `residency`、不强制 C/Metal route；
4. disabled 请求不读取 memory manifest、不计算 checkpoint SHA；
5. 编译器可以内联掉 C runtime 中 `hooks_enabled == false` 的短分支；
6. ABBA 测试验证 A=原始 commit、B=功能代码但 disabled、A'=功能代码关闭，三者 route/
   cache/output/性能等价。

建议新增轻量 RAII，而不是在每个 session 手写清理：

~~~cpp
class MemoryAdmissionBinding {
  public:
    MemoryAdmissionBinding(ModelSession &session, MemoryAdmission *admission)
        : session_(session) {
        session_.set_memory_admission(admission);
    }
    ~MemoryAdmissionBinding() noexcept {
        session_.set_memory_admission(nullptr);
    }
  private:
    ModelSession &session_;
};
~~~

异常、取消、编码失败、输出路径失败均必须经过析构清除 binding。session 下一请求开始时
再断言 admission 为空，防止持久 daemon 复用旧 generation。

### 121.3 顶层执行顺序

建议将现有 plan 和 generate 的重复规划收敛成以下顺序：

~~~text
parse request/profile
  -> normalize model shape/backend
  -> make_plan(request, PlanMode::MetadataOnly)
  -> validate capability and budget
  -> acquire DeviceLease
  -> capture process/system admission snapshot
  -> create RequestMemoryContext
  -> bind session callbacks
  -> execute exact frozen effective request
  -> finalize pending GPU releases
  -> capture result/evidence
  -> unbind callbacks
~~~

当前多个 session 在 `generate()` 内再次调用 `make_plan(r)`。短期允许二次调用，但
必须让 effective request 幂等；中期应把 `ExecutionPlan` 传入执行器，避免第一次 plan
与 session 内第二次 plan 因环境变量、cache、checkpoint 或 auto route 变化而不一致。
推荐新增：

~~~cpp
virtual RunResult generate_planned(
    const ExecutionPlan &plan,
    RequestMemoryContext *memory,
    const Event &event,
    std::atomic<bool> &cancelled);
~~~

迁移期默认实现可委托旧 `generate(plan.request, ...)`；H3/LTX constrained adapter
首先覆盖。完全迁移后，执行器不得自行更改影响内存的字段。确需选择 tile/slot 时，只能
从 `plan.allowed_runtime_revisions` 中选，并记录 revision。

### 121.4 只允许四种状态标签

所有接口、日志和 UI 使用统一状态，避免“支持”一词含义漂移：

~~~text
unsupported          没有合法 route 或明确排除
plan_only            可估算，但不允许运行
experimental_guarded required allocation 已 guard，envelope 尚未完成 L3
certified             精确 capability whitelist 命中且 campaign 通过
~~~

`execution_supported` 只在后两种为 true；`release_stable` 只在 `certified` 为 true。
默认发布开关可进一步要求 `certified`，内部开发开关才允许
`experimental_guarded`。无论内部开关如何，`plan_only` 不得进入模型分配。

## 122. Policy、Capability 与 Manifest 的代码级改造

### 122.1 拆分当前 `EffectiveMemoryPolicy`

当前结构把用户合同、候选、估算结果和运行状态混在一起。建议保持 JSON 兼容，同时在
C++ 内部拆成四层：

~~~cpp
struct MemoryBudgetContract {
    uint64_t user_limit_bytes = 0;       // Y
    uint64_t effective_budget_bytes = 0; // B
    uint64_t system_reserve_bytes = 0;
    unsigned buffer_percent = 15;
    unsigned max_refill_slots = 3;
    bool allow_quality_preserving_tiling = true;
};

struct MemoryCandidateKey {
    std::string adapter;
    std::string model_id;
    std::string checkpoint_digest;
    std::string backend;
    std::string dtype;
    std::string model_variant;
    std::string operation;
    std::string shape_bucket;
    std::string sampler_mode;
    std::string runtime_revision;
    std::string device_family;
};

enum class CertificationState : uint8_t {
    Unsupported,
    PlanOnly,
    ExperimentalGuarded,
    Certified,
};

struct MemoryCapabilityRecord {
    MemoryCandidateKey key;
    CertificationState state = CertificationState::Unsupported;
    MemoryCapabilityLevel level = MemoryCapabilityLevel::Declared;
    std::string manifest_digest;
    std::string evidence_digest;
    std::vector<unsigned> verified_refill_slots;
    std::vector<std::string> verified_tiling_modes;
    uint64_t maximum_validated_upper_bytes = 0;
    bool release_enabled = false;
};
~~~

为降低首个 PR 风险，可以暂不移动现有字段，而是在 `EffectiveMemoryPolicy` 尾部增加
`capability_level`、`certification_state`、`manifest_digest`、
`evidence_digest` 和 `release_stable`。本轮已完成这些字段和结果 JSON 输出；后续
registry/service 完成后再把 `release_stable` 与正式 whitelist 绑定。

### 122.2 移除启发式自动执行

历史上的 `native/runtime/plan.cpp` 逻辑（当前已删除）：

~~~cpp
if (memory_policy->route_available && memory_policy->estimate_fits)
    memory_policy->execution_supported = true;
~~~

必须替换为显式 capability 命中：

~~~cpp
const auto capability = capability_registry.lookup(candidate_key);
memory_policy->route_available = candidate_match.matched;
memory_policy->estimate_fits =
    compiled_plan && compiled_plan->peak_bytes <= budget.effective_budget_bytes;
memory_policy->execution_supported =
    memory_policy->estimate_fits &&
    capability &&
    capability->level >= MemoryCapabilityLevel::EnvelopeValidated &&
    capability->manifest_digest == compiled_plan->manifest_digest &&
    capability->release_enabled;
~~~

内部实验运行可以由编译时或仅测试环境的 `allow_experimental_guarded` 放宽到
`AllocationGuarded`，但该字段不得进入公共 request schema，production daemon 默认
false。禁止用环境变量在用户不知情时把 plan-only 变成 executable。

### 122.3 Capability registry 文件布局

建议新增：

~~~text
native/runtime/memory_capability.hpp
native/runtime/memory_capability.cpp
native/runtime/memory_manifest.hpp
native/runtime/memory_manifest.cpp
native/models/h3_runtime/h3_memory_manifest.c
native/models/ltx_runtime/ltx_memory_manifest.c
native/models/flux2/memory_manifest.cpp
native/models/z_image/memory_manifest.cpp
resources/memory-capabilities/*.json
~~~

职责边界：

- code registry：candidate matcher、required site、shape 公式版本；
- generated manifest：checkpoint tensor 表、分区、exact bytes、validated graph upper；
- capability JSON：哪些 manifest/evidence 可以执行或发布；
- request/result JSON：只引用 digest 和状态，不复制完整 tensor table。

production 只接受随二进制签名/打包的 capability 文件；开发态外部 manifest 只能
`plan_only`，除非显式测试 build。这样可以防止用户修改 JSON 绕过 hard gate。

### 122.4 Candidate key 必须冻结的字段

以下任一字段改变都必须 cache miss 并重新 plan：

~~~text
model id / variant / checkpoint SHA-256
所有权重 shard/index 的 path+size+digest
backend、dtype、quantization layout、kernel family
GPU family、OS major、Metal/MLX runtime revision
operation、input modality、audio、LoRA identity
width/height/frames/fps/token bucket/steps
sampler CPU/GPU、reuse/cache/sparse/tome 等算法开关
residency、pinned set、slot count、tile/chunk strategy
allocator cache limit、graph compile mode
site registry version、shape formula version
~~~

“steps”是否影响 peak 不能只凭理论判断。即使单步 shape 不变，GPU sampler velocity、
cache/reuse、trace、preview 或跨步 graph retention 都可能改变 live set，因此首版仍
进入 key。以后只有证据证明等价后才能把它移出。

### 122.5 Request normalization 必须幂等

现有 `MemoryConstrainedConfig` 中 runtime-only 字段是合理过渡方案，但要补以下断言：

~~~cpp
require(!config.normalized ||
        (config.effective_budget_bytes == policy.effective_budget_bytes &&
         config.denoiser_budget_bytes == policy.denoiser_budget_bytes &&
         config.refill_slots == policy.refill_slots),
        "memory_policy_invalid: normalized request drift");
~~~

第二次 `make_plan` 不得再次从 B 扣除 non-denoiser reserve，也不得把已规范化的
`memory_budget_bytes` 当作用户 legacy 字段触发冲突。最终应在 `ExecutionPlan` 保存
`requested_request` 和 `effective_request`，不再把 runtime-only 字段写回公共
`Request`。

### 122.6 Digest 生成规则

当前 FNV digest 只能用于日志相关性，不适合作为 capability 身份。建议：

- `policy_digest`：canonical JSON 后 SHA-256；
- `manifest_digest`：稳定排序的资源/site/upper/lifetime 后 SHA-256；
- `checkpoint_digest`：所有参与文件的相对路径、size、SHA-256 构成 Merkle-like digest；
- `plan_digest`：policy + candidate + manifest + exact stage plan；
- `evidence_digest`：campaign summary + plan digest + binary/runtime identity。

数值全部以无符号十进制 byte 输出；浮点质量阈值使用稳定十六进制 bit pattern 或固定
decimal schema，不依赖 locale。

## 123. Plan IR、精确峰值编译与预算求解

### 123.1 为什么需要独立 Plan IR

不能让 H3/LTX 各自返回一个“总字节数”，因为 planner 必须知道资源何时同时存活、
何时可释放、预取是否形成重叠峰值。建议拟增以下最小 IR：

~~~cpp
using ResourceId = uint32_t;
using Epoch = uint32_t;

enum class ResourceKind : uint8_t {
    Weight,
    Activation,
    Conditioning,
    RefillSlot,
    HostStaging,
    GraphTemporary,
    Output,
    AllocatorCache,
    ProcessBaseline,
};

struct ResourceInterval {
    ResourceId id = 0;
    std::string site_id;
    ResourceKind kind;
    uint64_t upper_bytes = 0;
    Epoch acquire_epoch = 0;
    Epoch release_epoch = 0; // half-open [acquire, release)
    uint64_t alias_group = 0;
    bool required = true;
    bool asynchronous_release = false;
};

struct ScheduleEpoch {
    Epoch id = 0;
    std::string name;
    std::string required_completion_event;
    bool runtime_safe_point = false;
};

struct CompiledMemoryPlan {
    std::vector<ScheduleEpoch> epochs;
    std::vector<ResourceInterval> resources;
    uint64_t planned_peak_bytes = 0;
    Epoch peak_epoch = 0;
    std::vector<ResourceId> peak_live_set;
    std::string manifest_digest;
    std::string plan_digest;
};
~~~

`peak_live_set` 必须输出，便于审查“为什么 16 GiB 不可行”，而不是只给一个数字。

### 123.2 Epoch 与异步完成语义

epoch 不是 wall-clock 时间，而是有全序的生命周期边界。建议统一顺序：

~~~text
0 request_admitted
10 text_load_begin
20 text_compute_begin
30 text_gpu_complete
40 denoiser_load_begin
50 denoise_step_N_block_M_submit
60 denoise_step_N_block_M_complete
70 denoise_complete
80 decoder_load_begin
90 decode_complete
100 export_complete
110 request_cleanup_complete
~~~

真实实现可用紧凑递增 id；示意中的间隔用于插入 block/tile epoch。GPU 资源
`release_epoch` 必须对应 command buffer completion，不是 CPU 编码函数返回。阶段间
有显式 barrier 且旧资源 completion 后才加载新资源时，旧资源 interval 在新 acquire
之前结束；若允许 preload overlap，则两个 interval 必须相交。

### 123.3 Alias 与唯一 backing

同一 `MTLBuffer` 被多个 tensor view 引用时只计一次。规则：

1. manifest 资源可以有相同 `alias_group`；
2. 组内 upper 取唯一 backing capacity，不求和；
3. 若执行时出现不同 handle，必须分别计费，即使 site 声称 alias；
4. in-place 更新只有输入 last-use 已发生且 kernel 合同允许时才共享 alias；
5. mmap 文件大小不等于 resident physical bytes，但 mapped region 的 materialized
   upper 和目标 GPU buffer 均要进入各自 interval；
6. `Weights::bytes()`、`mx::get_active_memory()` 与底层 Metal buffer 不能跨来源重复
   相加；需要通过 source ownership 标记选择一个权威计费源。

### 123.4 Peak sweep 算法

所有 arithmetic 使用 checked uint64。按 half-open interval 生成 delta：

~~~cpp
for (const auto &resource : resources) {
    add_delta(resource.acquire_epoch, +resource.upper_bytes, Acquire);
    add_delta(resource.release_epoch, -resource.upper_bytes, Release);
}
sort(events, epoch, sequence);
~~~

不能仅用同 epoch “release first”规则掩盖真实重叠；epoch 编译器必须先把 barrier 和
preload 决策展开为不同 sequence。alias group 在 sweep 前折叠。每个 epoch 保存 live
resource ids，得到：

~~~text
peak(epoch) = process_baseline
            + sum(unique live backing upper)
            + calibrated unattributed/framework upper
~~~

可行条件：

~~~text
planned_peak <= B
next_required_allocation <= B - current_committed - current_reserved
system_available_at_admission >= min_free_bytes + planned_increment
capability covers every required resource
~~~

系统 available 检查与进程 B 是两个独立门，不做相加后重复扣除。

### 123.5 Candidate 枚举与确定性排序

候选只枚举离散、后端已实现的组合：

~~~text
component residency:
  resident / component_staged
block residency:
  full / pinned-prefix+streamed-suffix
refill slots:
  backend verified set 的 1/2/3
decode:
  full / verified spatial tile / verified temporal chunk
stage preload:
  none / decoder-after-last-step-submit / other certified overlap
~~~

排序建议：

1. certified 优先于 experimental；
2. 同质量合同内 planned peak 更低且满足 latency SLO；
3. 预测 wall time 最短；
4. unhidden I/O bytes 更少；
5. 较少 runtime replan、较少 graph compile；
6. candidate id 字典序，保证完全确定。

不能因为 resident 更快就选择 `planned_peak > B`。不能因为更低精度更省内存就自动
改模型变体；量化只有用户原本选择且 capability 精确匹配时才是候选。

### 123.6 成本模型

候选延迟初始可使用可解释模型：

~~~text
T_total =
  T_metadata
  + sum(T_stage_load + T_stage_compute + T_stage_export)
  + max(0, T_refill_io + T_convert - T_overlap_compute)
  + T_barrier
  + T_replan_expected
~~~

参数按 `device_family + checkpoint_digest + candidate` 校准，保留样本数、P50/P95。
成本模型只负责在都安全的候选中排序，绝不能参与放宽 memory upper。缺性能样本时选择
更保守的 2-slot/无 preload 方案，而不是猜测 3-slot 一定更快。

### 123.7 计划解释输出

plan JSON 至少增加：

~~~json
{
  "memory_policy": {
    "certification_state": "plan_only",
    "capability_level": "allocation_guarded",
    "manifest_digest": "...",
    "plan_digest": "...",
    "planned_peak_bytes": 0,
    "peak_epoch": "decode.output_and_rgb8",
    "peak_live_set": [
      {"site_id": "h3.vae.decoded_f32", "upper_bytes": 0},
      {"site_id": "h3.host.rgb8", "upper_bytes": 0}
    ],
    "candidate_rejections": [
      {
        "candidate": "h3_c_metal_streamed_2slot",
        "code": "memory_budget_too_small",
        "required_bytes": 0,
        "available_bytes": 0
      }
    ]
  }
}
~~~

真实数值由 probe 填充，禁止示例常数进入执行。

### 123.8 H3 阶段 live-set 示例

H3 首发 T2V 可先编译成如下抽象 epoch；这是资源关系示例，不是固定数值：

| Epoch | 新增资源 | 释放资源 | 跨阶段继续存活 |
|---|---|---|---|
| metadata | tokenizer/header/index 控制 envelope | probe 临时映射 | manifest |
| text-load | text encoder pinned + slots | 无 | prompt tokens |
| text-compute | encoder activation | 每层 activation/slot | final embedding/layout |
| denoiser-load | DiT trunk、pinned blocks、2 slots | text encoder weights | conditioning、joint latent |
| denoise-block | block activation、current/next slot | previous block event-complete slot | latent、conditioning |
| denoise-end | sampler scratch | DiT weights/slots/activation | final video/audio latent |
| vae-load | VAE weights/activation | audio latent 若输出合同不需要 | video latent |
| decode | tile/chunk scratch、decoded F32 | prior tile scratch | decoded F32 |
| convert/export | RGB8 + encoder staging | decoded F32 last-use 后 | output file buffer |
| finalize | 无 | 全部 job resource | 无 |

在每个交接点都要明确“释放发生在 CPU 返回、GPU completion 还是媒体 consumer ack”。
若不能证明 last-use，就按更长 interval 计费。

### 123.9 LTX 两阶段 live-set 示例

LTX 的关键不是把 Stage1 和 Stage2 权重算两遍，而是正确处理共享权重和 latent：

~~~text
Gemma weights -> release
conditioning -------------------------------> Stage2 last use
shared denoiser pinned blocks ---------------> Stage2 denoise end
Stage1 latent -> upsampler input
upsampler workspace -> Stage2 latent
Stage1 latent release only after upsampler GPU completion
Stage2 latent -------------------------------> VAE input completion
VAE weights/workspace -> decoded frames
decoded frames + export staging -> mux completion
~~~

若当前实现为 Stage1 后销毁并重建 denoiser，plan 应准确加入 reload 转换峰值；不能为了
表面上“共享”而忽略真实代码。优化共享生命周期必须作为独立性能 patch。

## 124. Ledger v2、StageScheduler 与 allocation 事务

### 124.1 `RequestMemoryContext` 是唯一 job owner

建议新增：

~~~cpp
class RequestMemoryContext {
  public:
    static RequestMemoryContext admit(
        const ExecutionPlan &,
        const ProcessMemoryObservation &);

    MemoryAdmission &admission();
    StageScheduler &scheduler();
    ScheduleTrace &trace();
    const CompiledMemoryPlan &plan() const;
    void checkpoint(std::string_view phase);
    void finalize();
};
~~~

它拥有 job generation、ledger、stage state、trace、pressure 状态和 capability lease。
模型 session 只持非 owning 指针；任何 tensor/buffer token 都不得比 context 活得更久。

### 124.2 Reservation 状态机

现有 `MemoryReservation -> StorageLease -> PendingRelease` 方向正确，建议显式固化：

~~~text
Reserved
  -> Cancelled                         allocation 失败/分支未使用
  -> Active                            commit 成功
Active
  -> Released                          同步资源最后引用释放
  -> PendingCompletion                 GPU 已提交但未完成
  -> Cached                            仅 capability 允许
PendingCompletion
  -> Released
  -> Cached                            completion 后可缓存
Cached
  -> Active                            同 generation 合法 reactivate
  -> Released
~~~

非法转换立即报 `memory_lifetime_violation`；析构只能做不抛异常的保底 release，并在
metrics 中增加 `implicit_cleanup_count`。认证 campaign 要求正常成功路径该值为 0，
因为所有生命周期应显式结束。

### 124.3 Allocation ticket

为了让 C、Objective-C 与 C++ 使用同一合同，建议增加 POD 风格 ticket：

~~~cpp
struct AllocationRequest {
    std::string_view site_id;
    MemoryClass memory_class;
    uint64_t upper_bytes;
    uint32_t instance_id = 0;
    uint64_t expected_generation = 0;
};

class AllocationTicket {
  public:
    void commit(StorageId actual);
    PendingReleaseId retire();
    void cancel() noexcept;
};
~~~

C callback 仍传 opaque token；Objective-C++ bridge 内部持有 `AllocationTicket`。
`upper_bytes` 来自 plan 中本次 instance，runtime 调用点不得自行换一个更宽松常数。
commit 时校验：

~~~text
actual.capacity > 0
actual.capacity <= upper
allocator_domain/handle/generation 合法
site/instance 尚未 commit
相同 backing 的 capacity 一致
~~~

### 124.4 Scope 与 stage token

拟增 `MemoryScopeId`：

~~~cpp
struct MemoryScopeId {
    uint64_t job_generation;
    uint32_t stage;
    uint32_t step;
    uint32_t block_or_tile;
};
~~~

每个 reservation 绑定 scope。`StageScheduler::end_stage` 只允许在：

1. 该 stage 所有 command completion 已发生；
2. 所有 required reservation 已 commit 或 cancel；
3. 所有 block/tile scope 无 active/pending backing；
4. 仅 manifest 声明的 cross-stage resource 仍 active。

条件不满足时返回具体 live site，而不是强行清账。

### 124.5 StageScheduler 最小接口

~~~cpp
class StageScheduler {
  public:
    StageToken begin_stage(StageId);
    LoadPermit request_component_load(ComponentId, uint64_t upper);
    PrefetchPermit request_prefetch(PartitionId, uint64_t upper);
    void record_submit(CommandId, std::span<const StorageId>);
    void record_complete(CommandId);
    SafePointDecision safe_point(const RuntimeObservation &);
    void end_stage(StageToken);
};
~~~

`LoadPermit`/`PrefetchPermit` 是 reservation 的高层封装。scheduler 不直接创建
`MTLBuffer`；模型 adapter 负责实际 load。这样通用 runtime 不依赖 H3/LTX 的 tensor
格式。

### 124.6 Pressure 输入与 replan

首版 pressure controller 只在以下边界运行：

- 组件 load 前；
- denoise step 开始/结束；
- block group completion；
- VAE tile/chunk completion；
- export queue flush；
- request finalize。

输入：

~~~text
ledger committed/reserved/pending/cache
process physical footprint
system available/pressure level
next required allocation upper
recent refill wait and I/O throughput
~~~

输出只能从预先认证的 revision 中选择：

~~~text
Continue
StopOptionalPrefetch
ShrinkLookahead
EvictOptionalCache
SwitchAtNextStageToCertifiedTile
AbortBeforeNextAllocation
~~~

首版不允许运行中改变 dtype、模型、分辨率、steps、attention 数学或已经开始的 tile halo。

### 124.7 越线与 worker taint

出现以下任一情况，当前 context 标记 tainted：

~~~text
actual capacity > manifest upper
required unknown allocation
stale generation commit/release
stage 结束仍有未声明 live resource
observed process footprint > B
child/helper 超出声明 envelope
~~~

tainted 后：

1. 禁止创建新的可选 allocation；
2. 等待已提交 GPU command 到安全 completion；
3. 释放当前 job 所有可释放资源；
4. 请求以原始错误失败；
5. persistent session 执行 `unload()`；
6. 若未知资源无法证明释放，worker 退出，由 service 重建；
7. evidence 写入 manifest quarantine 列表，后续请求不再使用该 capability。

### 124.8 Baseline、planned increment 与 ledger budget

当前 `MemoryAdmission` 用进程初始 footprint 和 planned increment 构造 allocation
ceiling。最终合同应明确：

~~~text
B                       用户有效预算
P0                      admission 时 process physical footprint
R_controlled_existing   P0 中已经由当前 session ledger 识别的可复用 backing
U0                      max(0, P0 - R_controlled_existing)
I_plan                  本请求相对 P0 的计划新增峰值
allocation_ceiling      min(B, checked_add(P0, I_plan))
ledger_budget           allocation_ceiling
ledger_baseline         P0
~~~

若 session 在请求之间保留权重，必须先把它们作为 cached backing 注册，不能既包含在 P0
又按新 load 再计一次。首发受限模式建议禁用跨请求模型 cache，降低这类双重身份复杂度。

### 124.9 Unknown 的双门禁

`UnknownExternal` 分成：

~~~text
optional_unknown        debug/小控制对象；总量受 tiny-control envelope 限制
required_unknown        任何影响成功的未分类大 allocation
~~~

默认阈值建议不是一个全局固定 1 MiB，而是 manifest 字段：

~~~text
individual_guard_threshold_bytes
aggregate_control_upper_bytes
~~~

低于单项阈值的对象可以进入聚合 control envelope；累计超过 envelope 仍失败。任何
`MTLBuffer`、MLX graph temporary、完整 tensor row 或媒体 frame backing 不得以“小对象”
名义进入 control envelope。

## 125. WeightPager、三队列 overlap 与反抖动策略

### 125.1 先统一合同，不强行统一权重格式

H3 shard、LTX safetensors、Flux/Z-Image MLX `Weights` 和 GGUF 的物理布局不同。
`WeightPager` 应统一生命周期和调度，不应假设所有模型都能按相同 offset 读取：

~~~cpp
struct PartitionDescriptor {
    PartitionId id;
    uint32_t logical_order = 0;
    uint64_t resident_bytes = 0;
    uint64_t staging_bytes = 0;
    uint64_t converted_bytes = 0;
    bool already_device_ready = false;
};

class PartitionSource {
  public:
    virtual ~PartitionSource() = default;
    virtual std::vector<PartitionDescriptor> describe() const = 0;
    virtual ReadResult read(
        PartitionId, MutableBytes destination, const CancelToken &) = 0;
    virtual void verify_identity() const = 0;
};

class PartitionMaterializer {
  public:
    virtual MaterializedPartition materialize(
        const ReadResult &, AllocationTicket &, GpuTransferQueue &) = 0;
};
~~~

H3/LTX 首期 adapter 可继续使用已有 loader/thread，只需把 `PrefetchPermit`、site id、
completion 事件接入；不要求先重写为通用 C++ pager。通用 pager 在两个 adapter 的
生命周期合同稳定后再抽取，避免把成熟的 H3/LTX 读取路径一起重写。

### 125.2 Slot 状态机

每个 refill slot 只有以下状态：

~~~text
Empty
  -> Reserved
  -> Reading
  -> Converting
  -> Ready
  -> Submitted
  -> InUse
  -> Retiring
  -> Empty
~~~

允许 `Reading -> Empty`（I/O 取消）和任意未提交状态到 `Failed`；`InUse` 只能在 GPU
completion 后进入 `Retiring`。slot 记录：

~~~cpp
struct RefillSlotState {
    SlotId slot;
    PartitionId partition;
    uint64_t generation;
    AllocationTicket ticket;
    StorageId backing;
    IoTicket io;
    CommandId last_use_command;
    SlotPhase phase;
};
~~~

分区 identity 和 generation 都要匹配；旧请求的异步读完成后不得覆盖新请求 slot。

### 125.3 三条队列

逻辑上分为：

1. I/O queue：读取、校验长度、可选 checksum；
2. transfer/convert queue：dtype 转换、pack、上传；
3. compute queue：执行当前 block。

并不要求建立三个 `MTLCommandQueue`。首发可以是一个 I/O worker + 一个 Metal queue，
只要用 event/ticket 表达依赖。额外 Metal queue 只有在实测能并发且不增加不可控
driver workspace 后才开放。

标准时序：

~~~text
compute block i on slot A
    overlaps
read block i+1 to host/shared staging B
    then
convert/upload i+1 in B
GPU complete i
retire A
submit i+1 using B
~~~

三槽只在 `read + convert` 明显长于单 block compute 且第三槽不会把水位推过 high
watermark 时有意义。否则第三槽只是增加内存。

### 125.4 Non-uniform block 规划

不能继续用单一 `block_bytes` 代表所有 block。对每个 partition 计算：

~~~text
slot_upper(partition) =
  max(source_read_bytes,
      converted_device_bytes,
      conversion_overlap_live_bytes)
~~~

若 slot backing 可复用，槽容量取该槽可能承载 partitions 的最大值。为了减少浪费，
可按大小分桶：

~~~text
small slot class
medium slot class
large slot class
~~~

但首版 H3/LTX 建议维持现有固定 typed slot，manifest 写入真实最大值。只有当 padding
浪费超过目标预算 10% 且 profiling 显示切换槽类型收益明显时再引入分桶。

### 125.5 Look-ahead 决策

每个安全点计算：

~~~text
free_budget = B - committed - reserved
next_upper = next_partition.slot_upper
can_prefetch = free_budget >= next_upper
               && process_footprint + next_upper <= B
               && pressure < warning
               && an empty certified slot exists
~~~

性能反馈：

~~~text
stall_ratio = refill_wait_seconds /
              max(block_compute_seconds + refill_wait_seconds, epsilon)
~~~

- 连续 3 个 block group `stall_ratio > 0.15`，且 free budget 足够：增加一个已认证
  look-ahead；
- 连续 3 个安全点到达 0.90B 或 pressure warning：减少一个 look-ahead；
- 恢复必须低于 0.85B 且稳定 3 个安全点；
- candidate 没有对应 slot-count capability 时不得在线切换。

这些阈值是初始实验参数，最终写入 capability/evidence，不作为跨机器常量硬编码。

### 125.6 Pinned prefix 选择

选择 pinned blocks 时不应只取“尽可能多的前缀”。初期可兼容 H3/LTX prefix 约束，
但成本函数应包含：

~~~text
benefit(block) =
  expected_uses * load_latency
  - resident_bytes * memory_shadow_price
~~~

若 loader 只能 prefix，则枚举 prefix 长度 0..N 并求总 benefit。若某些首尾 block
明显更常用或转换更贵，后续可开放显式 pinned set，但必须确认执行顺序和 slot loader
支持非连续分区。

`memory_shadow_price` 随水位提高：

~~~text
<0.80B 低
0.80–0.90B 中
>0.90B 高
~~~

因此同一 checkpoint 在更小 Y 下自然选择更短 prefix，而不是依赖压力发生后再驱逐。

### 125.7 I/O 与 page-cache 现实

- `pread` 到 shared buffer 的 bytes 是受控 backing，但文件页缓存仍由 OS 管理；
- mmap 不代表零内存，首次触页可能提升 physical footprint；
- 禁止用 `posix_fadvise` 或 `F_NOCACHE` 的存在宣称 page cache 为零；
- manifest 的 framework/unattributed envelope 必须覆盖测得的 page-cache/driver 差值；
- cold campaign 要在可重复条件下记录首次读取，warm campaign 记录缓存命中；
- 不通过清空全系统 cache 的特权操作伪造生产行为。

### 125.8 取消与 I/O 错误

取消顺序：

1. 停止发出新 prefetch；
2. 标记尚未提交的 I/O ticket cancelled；
3. 让进行中的 `pread` 返回后丢弃结果并 cancel reservation；
4. 已提交 GPU command 等 completion；
5. retire/release slot；
6. 清理 stage/component；
7. 验证 ledger token 全部归零。

短读、checksum 不符、checkpoint 发生 TOCTOU 变化都返回 checkpoint/IO 错误，不能用旧
slot 数据继续计算。失败 slot 在重新 verify identity 前不能复用。

## 126. H3：基于当前代码的逐文件实现方案

### 126.1 首发 candidate 必须进一步具体化

当前 `h3_c_metal_streamed_v1` 仍隐藏 sampler、模型变体和若干环境开关。建议拆为：

~~~text
h3_c_metal_bf16_streamed_cpu_sampler_t2v_v1
h3_c_metal_bf16_streamed_gpu_sampler_t2v_v1
~~~

首批优先认证 CPU sampler 或当前目标设备实际使用的单一路线；另一条保持 plan-only。
两者共享 DiT/VAE manifest，但 sampler resource 不同。任何
`H3_GPU_SAMPLER`、`H3_CPU_SAMPLER`、reuse、cache、token reduction、Sol、Core ML/ANE
环境开关必须被 adapter 固定或纳入 candidate key，不能让用户环境偷偷改变 live set。

### 126.2 `h3_gpu.h/.m`

当前工作树已增加 classified `from_f32/from_bf16/from_u32`。后续收尾：

1. 为 C header 添加 API 注释：旧 API 永远标记 UNKNOWN，仅供 disabled/legacy；
2. classified create/upload 在 reserve 失败时必须发生在 `newBufferWithLength` 之前；
3. commit 使用 Metal buffer `length`，不是请求 elements 推算值；
4. allocation 失败调用 cancel；tensor free exactly-once release；
5. GPU command 引用 tensor 时支持 pending release，避免 CPU free 早于 completion；
6. hook test 检查 class、tag、domain、generation、actual capacity。

建议在 constrained build/test 中加静态扫描：首发 reachable 文件禁止调用旧
`h3_gpu_tensor_new_*` 和 `h3_gpu_tensor_from_*`，除非调用点带
`H3_LEGACY_UNREACHABLE_IN_CONSTRAINED` 注释并由 route test 证明不可达。

### 126.3 `h3_video_vae.c/.h`

当前 worktree 已完成 required GPU tensor 分类。下一步不再是“继续换 API”，而是把
shape 和生命周期公式独立：

~~~c
typedef struct {
    uint64_t input_upload_bytes;
    uint64_t rope_bytes;
    uint64_t activation_peak_bytes;
    uint64_t suffix_scratch_bytes;
    uint64_t decoded_f32_bytes;
    uint64_t tile_host_peak_bytes;
    uint64_t rgb8_overlap_bytes;
    uint32_t tile_count;
} h3_video_vae_requirements_v1;

int h3_video_vae_requirements(
    const h3_video_vae_shape *,
    const h3_video_vae_options *,
    h3_video_vae_requirements_v1 *,
    char *error, size_t error_size);
~~~

planner 和 runtime 必须调用同一函数。`allocate_activations()` 创建的多个 tensor 不是
简单相加所有字段，而应由函数按实际同时存活集合返回 peak；同时 manifest 仍列出每个
site upper 用于 runtime guard。

decoded-host 当前已按真实 `frames * width * height * 3 * sizeof(float)` commit。还要
把“reserve upper”从 `predicted * 4` 改为上面的资源公式，并区分：

~~~text
final frame backing
tile accumulation backing
temporal overlap
pointer/metadata
RGB8 overlap
~~~

如果短期继续聚合 envelope，字段名和 provenance 必须明确为
`ValidatedEnvelope`，实际 campaign 覆盖所有支持 shape/tile；未经验证仍是
`HeuristicNotExecutable`。

### 126.4 `h3_text_encoder.c` 与 tokenizer

GPU weights/ids/RoPE/deepstack/activation 已有分类，但 host 侧仍需：

- tokenizer JSON 文件 size 和 Foundation 对象峰值 envelope；
- prompt UTF-8 byte upper、token count upper；
- `output->values` 的 `token_count * 5120 * sizeof(uint16_t)` checked formula；
- `output->tags` capacity；
- deepstack host/source staging（若存在）；
- encoder result 到 DiT last-use 的 conditioning interval。

不能把 `strlen(prompt)` 当作 tokenizer 对象峰值；它只能作为 token 数的一个保守输入。
首发 manifest 应设置最大 prompt bytes/token bucket，超过 bucket 在 metadata phase
拒绝，不允许 tokenizer 先扩容后再报错。

建议新增：

~~~c
int h3_text_requirements(
    size_t prompt_bytes,
    uint32_t max_tokens,
    h3_text_requirements_v1 *out,
    char *error, size_t error_size);
~~~

Foundation/tokenizer 无法逐 malloc hook 的部分进入经过实机验证的
`h3.host.tokenizer_control_v1` envelope。

### 126.5 `h3_dit.c` 剩余分类

当前 common path 已覆盖 refiner、RoPE、maps、final converted weights 和主 activation
arena，但源码仍有旧创建调用。处理原则：

| 调用组 | 首发处理 |
|---|---|
| `ensure_previous_velocities()` | 必须分类为 `ACTIVATION`，site 为 previous video/audio velocity；或首发强制 CPU sampler 并证明不可达 |
| dynamic INT8 quantized weights/scales | 当前 candidate 禁用 quantized cache；加不可达断言，未来单独作为 `WEIGHTS`/`CONVERSION_SCRATCH` candidate |
| Core ML/ANE qkv/mlp/attention-out buffers | GPU-only candidate 必须在 load 前拒绝；仍建议分类，避免未来扩展遗漏 |
| benchmark/dump/verification buffers | constrained candidate 默认禁用；请求/环境命中时 fail-closed |
| cache/Sol/token reduction optional arena | candidate 固定开关并纳入 manifest；未认证默认禁用 |

`ensure_previous_velocities()` 若保留 GPU sampler，建议修改为：

~~~c
dit->previous_video_velocity = h3_gpu_tensor_new_classified(
    dit->gpu, video_elements, H3_GPU_BF16,
    H3_GPU_MEMORY_ACTIVATION,
    "h3.dit.activation.previous_video_velocity");
~~~

audio 对象同理。即使用户最终只导出视频，只要 H3 数学路径创建 audio velocity，也必须
计入。

### 126.6 `h3.c` conditioning/control envelope

建议把 `h3_generate()` 分为显式 scope：

~~~text
request control
text/tokenize
conditioning/layout
joint latent
denoise
video decode
RGB conversion/export
cleanup
~~~

新增纯函数汇总 upper：

~~~c
int h3_request_host_requirements(
    const h3_params *,
    const h3_model_info *,
    const h3_video_vae_requirements_v1 *,
    h3_request_host_requirements_v1 *,
    char *error, size_t error_size);
~~~

逐 allocation 纳管优先级：

1. final text embedding、layout positions/segments；
2. video/audio latent；
3. decoded F32、RGB8；
4. tile/chunk staging；
5. tokenizer/Foundation 聚合 envelope；
6. key strings、sigma、小数组的 aggregate control。

cache 在首发 constrained request 中必须显式关闭并断言：

~~~text
conditioning cache disabled
resident DiT cache disabled
decoder cache disabled
preview cache disabled
~~~

### 126.7 `h3_session.mm`

当前 duplicate gate 是必要的。建议进一步：

1. `generate` 入口比较 `plan.adapter_candidate` 与 session adapter；
2. 只从 frozen plan 设置 H3 env/options，不读取外部变化；
3. `set_memory_admission` 清空时同时清空 native hooks；
4. generation 在每个 constrained job 增长，overflow 后拒绝新 job；
5. context identity 包含 checkpoint SHA、sampler mode、VAE strategy；
6. 异常后若 bridge 仍有 token，`context_.reset()` 并 taint worker；
7. result 返回 H3 GPU/host required unknown、active、pending、peak。

### 126.8 H3 完成定义

H3 进入 `AllocationGuarded` 前必须满足：

~~~text
首发 route reachable 的旧 GPU allocation API 为 0
host 大 allocation 全部逐项或 validated aggregate guard
required_unknown_bytes == 0
所有 reservation 在大 allocation 前发生
所有 GPU free 与 command completion 顺序正确
success/cancel/fault 后 token/pending/cache 符合预期
sampler/cache/optional env 已冻结
~~~

进入 `EnvelopeValidated` 还需要所有 shape bucket 的 metadata formula 与 observed upper
一致，并通过 minimum/minimum+headroom campaign。

## 127. LTX：native、MPSGraph、MLX 与 helper 的闭环方案

### 127.1 首发 route 的真实组成必须写清

`ltx_c_metal_streamed_video_v1` 不能只说明 denoiser 是 C/Metal。完整 candidate 要列出：

~~~text
tokenizer/Gemma backend
conditioning connector backend
Stage1 denoiser backend
upsampler backend
Stage2 denoiser backend
video VAE backend
video conversion/export backend
是否启动 helper 子进程
~~~

其中任何一步仍走未受控 MLX 或 unbounded helper，整体只能 plan-only。建议 candidate id
包含 `native_upsampler`、`mpsgraph_vae` 或 `mlx_vae` 等关键差异。

### 127.2 `ltx_gpu.m/.h`

集中 `ltx_gpu_buffer_new_classified` 已提供主要 guard。需要补：

- 所有首发 reachable `ltx_gpu_buffer_new` 旧调用迁移或证明不可达；
- storage mode、actual `[buffer length]`、domain/generation 进入 commit；
- shared buffer 的 host-visible backing 不重复以 host allocation 计费；
- MPSGraph tensor data 对同一 `MTLBuffer` 只计唯一 backing；
- command completion 前使用 pending release；
- `calloc(ltx_gpu_buffer)` 等小 control allocation纳入 aggregate envelope。

### 127.3 direct `MTLBuffer` 的统一封装

`ltx_video_vae.m` 和 `ltx_upsampler.m` 直接调用
`newBufferWithLength`，绕过 `ltx_gpu_buffer`。建议不要在两个文件复制 callback 逻辑，
而是在 `ltx_gpu.h/.m` 增加 external buffer helper：

~~~c
typedef struct ltx_gpu_external_allocation {
    void *native_buffer;
    void *memory_token;
    uint64_t handle;
    uint64_t bytes;
    uint64_t generation;
} ltx_gpu_external_allocation;

int ltx_gpu_external_buffer_new(
    ltx_gpu *gpu,
    uint64_t bytes,
    uint32_t storage_mode,
    ltx_gpu_memory_class memory_class,
    const char *site_id,
    ltx_gpu_external_allocation *out,
    char *error, size_t error_size);

void ltx_gpu_external_buffer_release(
    ltx_gpu_external_allocation *);
~~~

Objective-C model 可将 `native_buffer` bridge 为 `id<MTLBuffer>`，但 token 所有权仍由
external allocation struct 持有。数组中保存 wrapper 对象而不是裸 buffer，防止
NSArray 释放 buffer 后 ledger token 泄漏。

分类：

| 文件/对象 | class | site |
|---|---|---|
| Video VAE `weightBuffers` | weights | `ltx.vae.weight` + instance |
| Video VAE split graph `workspaceBuffers` | graph temporary | `ltx.vae.graph.workspace` |
| Upsampler `weightBuffers` | weights | `ltx.upsampler.weight` |
| Upsampler input/output workspace | activation/graph temporary | 对应 stage site |

MPSGraph 内部 private temporary 无法由 wrapper 直接观察，必须另有 validated graph
envelope；外部 workspace 被 guard 不代表 graph 闭环。

### 127.4 LTX host allocation 迁移

`ltx_blocks.c`、`ltx_conditioning.c`、`ltx_connector.c`、
`ltx_gemma_encoder.m` 和 `ltx_transformer_io.c` 有多处直接 `malloc`。建议增加 C
wrapper：

~~~c
void *ltx_host_malloc_classified(
    const ltx_native_options *,
    uint64_t bytes,
    ltx_host_memory_class,
    const char *site_id,
    ltx_host_allocation *allocation,
    char *error, size_t error_size);
~~~

迁移优先级：

1. noise/final latent host arrays；
2. conditioning embedding/combined/positions/RoPE；
3. transformer I/O flat arrays；
4. Gemma projected embedding；
5. verification/fixture-only arrays 标记首发不可达；
6. 小频率表/字符串进入 aggregate control envelope。

所有 `elements * sizeof(T)`、rows*hidden、video/audio patch 公式在 malloc 前 checked；
`realloc` 采用“先 reserve 新 capacity 或增量、再 realloc，成功后切换 lease，最后释放
旧 lease”的两阶段协议。若无法安全保留旧/新 backing 的短暂重叠，则按两者之和 reserve。

### 127.5 Stage1→upsampler→Stage2

必须定义精确交接：

~~~text
Stage1 output lease active
  -> reserve upsampler output/workspace
  -> submit upsampler
  -> completion
  -> commit Stage2 latent actual
  -> release Stage1 output and temporary
  -> start Stage2
~~~

如果 upsampler 输出复用 Stage1 backing，则 manifest 声明 alias 和 in-place 约束；
否则两者在 completion 前同时存活。planner 不得假设调用函数返回就可以释放。

### 127.6 MPSGraph envelope

每个 graph key（shape、split/full、dtype、OS/runtime）记录：

~~~text
external input/output bytes
explicit workspace bytes
weight bytes
process-footprint delta upper
Metal allocated-size delta upper（若可观测）
warm/cold compile temporary upper
steady execution temporary upper
~~~

冷编译和稳态必须分开。若 plan 允许首次请求编译，使用 cold upper；若 capability 只认证
预编译/缓存状态，服务启动必须先在独立预算内建立 cache，且请求不能碰到未认证 shape。
缓存丢失时 fail-closed，不在请求中临时冷编译。

### 127.7 MLX VAE/upsampler 和 helper

当前存在 `mx::load_safetensors` 路线。选项只有三种：

1. constrained candidate 明确固定 native/MPSGraph 路线，MLX 不可达；
2. 为 MLX 路线建立独立 validated envelope 和 worker 隔离 candidate；
3. 保持 plan-only。

不能在 native denoiser 通过后默认落入 MLX finalizer。若使用子进程：

- parent 在 spawn 前 reservation `ChildProcessEnvelope`；
- child 回报 PID、阶段和 footprint；
- parent/child 总和与共享映射去重规则写入 capability；
- child 超 upper 或异常退出时 job 失败；
- stdout/stderr/pipe buffer 和输出媒体 staging 也计入；
- helper 结束后验证进程回收，再 release envelope。

### 127.8 `ltx_session.mm`

建议将当前长 `generate` 路径切成不改变数学的 stage helper：

~~~cpp
Conditioning run_text_stage(...);
StageLatent run_stage1(...);
StageLatent run_upsampler_handoff(StageLatent &&);
StageLatent run_stage2(...);
DecodedFrames run_video_vae_stage(StageLatent &&);
void run_export_stage(DecodedFrames &&);
~~~

每个返回对象持有 backing lease 和 completion event，move 后旧对象失效。这样
`StageScheduler` 可以在 helper 边界验证 live set；无需一次性重写所有 kernel。

session 入口继续保留 duplicate route gate，并增加：

- effective candidate id 与后端对象类型断言；
- constrained route 下 `uses_parent_mlx(request)==false`；
- audio/I2V/LoRA/ANE 仍在任何模型 load 前拒绝；
- helper path 必须与 plan 完全一致；
- success/failure 后 `set_memory_admission(nullptr)`；
- result 输出 stage-specific peak、graph envelope、host unknown。

### 127.9 LTX slot 与性能策略

LTX 支持最多三槽不等于所有 Y 都应选三槽：

~~~text
1 slot：只有执行器确实支持“当前层算完再 refill”并完成认证时可选
2 slot：默认安全 overlap candidate
3 slot：SSD/convert 长尾明显且预算足够时的性能 candidate
~~~

Stage1 与 Stage2 可以有不同性能参数，但如果 runtime 不能安全重配 slot，首版取两阶段
共同认证的较保守配置。pinned blocks 可共享；stage-specific activation reserve 必须
分别计算。

### 127.10 LTX 完成定义

除了 H3 同类门禁外，LTX 特别要求：

~~~text
48-block streaming slot count 与 capability 一致
Stage1/Stage2 shared weight 不重复计费
direct MTLBuffer 全部封装
MPSGraph cold/steady envelope 分别认证
MLX/helper route 不可隐式出现
所有 host vector/malloc backing 纳管
upsampler 交接无提前释放
video-only candidate 不创建 audio backing
~~~

## 128. Flux：从 component-staged 到 block-streamed 的分阶段接入

### 128.1 当前代码特征

Flux 当前：

- `Flux::load()` 同时加载 transformer 和 VAE；
- text encoder 在 `conditioning()` 中按 residency 做阶段化；
- `run()` 已有 denoise 后清 transformer、decode 后清 VAE 的策略；
- `transformer_`、`vae_` 是整体 `Weights` map；
- dual-stream 与 single-stream block 在 `Flux::denoise()` 中逐层访问；
- compiled/hybrid graph 可能捕获或显式接收大量权重；
- session 有 prompt/conditioning、graph、LoRA 等跨请求 cache；
- 尚未覆盖 `set_memory_admission`。

因此 Flux 不应直接宣称“已经能按层 offload”。现有 component-staged 可作为 F1，真正
block streaming 是 F2。

### 128.2 F0：plan-only metadata

新增 `native/models/flux2/memory_manifest.cpp`，只读：

~~~text
transformer/vae/text encoder safetensors index
dual_layers、single_layers、hidden、heads
每个 component 和 block prefix 的 exact storage bytes
request token/image latent geometry
~~~

不调用 `Weights::load/materialize`。输出 component-stage peak 的 heuristic/metadata
计划，但 capability 为 plan-only，主要用于发现最低预算和设计分区。

### 128.3 F1：component-staged guarded

适用于 B 能容纳整套 transformer + denoise activation，但不能同时容纳 text/VAE。
代码建议：

1. `Flux` 增加 `set_memory_admission` 和 request-local cache policy；
2. 拆分 `load_transformer_stage()`、`load_vae_stage()`；
3. constrained route 禁止 `Flux::load()` 同时 load 两者；
4. text 完成后保留 request-local conditioning，清 text weights/cache；
5. transformer load 前 reserve exact storage + materialize/graph envelope；
6. denoise completion 后 clear transformer、graph、allocator cache；
7. VAE load/decode/export 后 clear；
8. constrained request 不把 conditioning/compiled graph 留给下一请求。

`Weights::bytes()` 是 logical storage 数，不足以单独作为实际 upper；commit 使用 MLX
active/allocator delta 与 validated envelope，且避免和 logical tensor table 双计。

### 128.4 F2：transformer block pager

按权重 key 分区：

~~~text
trunk:
  x/context embedder
  timestep/guidance/modulation
  final projection
dual block i:
  transformer_blocks.i.*
single block i:
  single_transformer_blocks.i.*
~~~

拟增：

~~~cpp
class FluxPagedWeights {
  public:
    const Weights &trunk() const;
    BlockWeightLease acquire_dual(int index, PrefetchTicket);
    BlockWeightLease acquire_single(int index, PrefetchTicket);
};
~~~

`Flux::denoise()` 的循环改成显式 lease：

~~~cpp
for (int i = 0; i < dual_layers_; ++i) {
    auto next = pager.prefetch_dual(i + 1);
    auto weights = pager.acquire_dual(i);
    run_dual_block(..., weights.view());
    materialize_block_output();
    weights.retire_after(last_command);
}
~~~

不能让 lazy MLX graph 保留已 offload block 的 weight array。每个 block 输出在权重释放前
必须 `eval`/提交，compiled graph 的权重必须为显式参数，不可 capture `Weights&`。

### 128.5 Flux compile cache

每种 dual/single block geometry 可以共用 compiled function，但：

- function graph 不持有 checkpoint tensor；
- explicit inputs 的 shape/dtype 固定；
- compile temporary 有独立 upper；
- cache 数量和 bytes 有 quota；
- constrained request 默认只使用预认证 compiled graph；
- cache miss 若没有 cold-compile capability，则 fail-closed；
- LoRA identity 变化必须使 graph/weight binding digest 失效。

### 128.6 VAE tiling

Flux VAE 先实现 component-stage；当 decode peak 仍超过 B 时才开放 tile。要求：

~~~text
tile/halo 公式来自卷积 receptive field
边界拼接与 full decode 质量通过
输出 consumer 支持逐 tile 或明确累积峰值
tile 顺序固定且可取消
每个 tile workspace completion 后释放
~~~

如果最终 PNG 保存仍需要完整像素 tensor，tile 只能减少 VAE workspace，不能消除最终
输出 backing；plan 必须保留完整像素 + encoder staging 峰值。

### 128.7 Flux 首发限制

建议 F1/F2 首发只开放：

~~~text
text-to-image
execution=gpu
无 input image
无 LoRA
无 ANE
固定 BF16 model variant
固定 token bucket / width / height / steps
无 dump
~~~

image edit/transform 会引入 VAE encode、reference latent 和更长 token sequence；LoRA
可能产生合并副本；这些应分别认证。

### 128.8 Flux 文件修改清单

| 文件 | 修改 |
|---|---|
| `flux.hpp` | memory binding、stage helper、paged weights 成员 |
| `pipeline.cpp` | frozen plan、component handoff、cache disabled 分支 |
| `flux_transformer.cpp` | block weight view、显式 prefetch/acquire/retire |
| `flux_text.cpp` | text stage requirements、conditioning lease |
| `flux_vae.cpp` | VAE requirements、可选 certified tiling |
| `memory_manifest.cpp` | metadata probe、resource intervals、candidate |
| `flux_module.cpp` | capability/限制对外描述 |

默认 `memory_constrained.enabled=false` 时仍走当前 `Weights transformer_, vae_` 路径，
不构造 `FluxPagedWeights`。

### 128.9 Flux 验收阶梯

~~~text
F0: metadata bytes 与 safetensors index 对齐
F1: component-staged required unknown=0，full transformer fit
F2: block lease/retire 正确，所有 step 可重复 paging
F3: overlap 性能达标，compiled graph 不保留 weight
F4: 指定 shape/device L3 certified
~~~

F1 证据不能开放 F2；F2 的 4B/9B 也分别认证。

## 129. Z-Image：safetensors、GGUF 和量化变体的独立认证

### 129.1 当前分区结构

`z_transformer()` 有天然分区：

~~~text
trunk/embedder/timestep/RoPE
2 x noise_refiner
2 x context_refiner
30 x layers
final layer
~~~

这为 pager 提供了 34 个逻辑 group，但不同 group 大小/输入不同，不能用一个 uniform
block bytes。VAE、Qwen text encoder 和 transformer 当前仍是整体 `Weights`。

### 129.2 Z1：component-staged

现有 `encode_text()` 已在完成后清 text encoder，是良好起点。还需：

1. constrained route 不在 `load()` 同时 materialize transformer + VAE；
2. conditioning 为 request-local，禁止跨请求 prompt cache；
3. transformer denoise 后清理，再加载 VAE；
4. `mx::clear_cache()` 后同步并观测 allocator residual；
5. full transformer fit 才允许 Z1；
6. BF16、ConvRot packed Q8、NVFP4、GGUF 分别生成 manifest/candidate。

### 129.3 Z2：34-group pager

建议分区 id：

~~~text
z.trunk
z.noise_refiner.0..1
z.context_refiner.0..1
z.layers.0..29
z.final
~~~

noise/context refiner 目前交替运行，planner 可以用同一 slot class，但不能假设两者权重
同大小。30 个 main layer 使用 2/3-slot overlap。`z_block`/
`z_context_block` 接收 `BlockWeightView`，不再从全局 `Weights` 通过字符串查找。

`z_transformer` 每层输出需确保 materialized/提交后才能 retire 当前权重；纯 GPU
compiled path 不必 host synchronize，但必须拿到可用于 completion 的 command/event。
若 MLX API 无法暴露足够的 completion 语义，首版使用每 block/group `mx::eval` 的保守
路径，性能优化作为后续 patch，不能提前释放。

### 129.4 GGUF 单独实现

GGUF 不是“天然低内存”。即使文件可随机访问，当前 adapter 若把所有 tensor view/
packed backing 保留并 materialize，仍可能超过 B。GGUF candidate 必须记录：

~~~text
quantization type per tensor
file offset/alignment/byte length
dequant/packed device bytes
per-block conversion scratch
mmap/pread strategy
GGUF metadata/control envelope
~~~

`NativeGGUF::select()` 在选择 variant 后先 metadata probe 和 SHA；`ZImage` 接收
`GGUFPartitionSource`，按 group 映射 tensor。禁止先构造完整 unbounded session 再判断
预算。

### 129.5 ConvRot、NVFP4 与 LoRA

- ConvRot 的 pack 会产生转换临时和 packed backing；计划要表达 old+new 短暂重叠；
- NVFP4 是 checkpoint-defined approximation，不能自动替换 BF16 candidate；
- inference-time LoRA 每层需要 base + low-rank tensor；作为独立 block manifest；
- in-memory merge 会产生复制峰值，首发 constrained 禁止；
- hybrid/Core ML 在 GPU-only first 中拒绝；
- debug dump 保存每步 tensor，会引入无界 I/O/staging，首发拒绝。

### 129.6 Z-Image 文件修改清单

| 文件 | 修改 |
|---|---|
| `z_image.hpp` | request memory binding、component/pager 成员 |
| `z_image.cpp` | stage helper、34-group weight view、cache 分支 |
| `gguf.cpp` | metadata-first selection、partition source、身份冻结 |
| `memory_manifest.cpp` | variant-specific resource/shape 公式 |
| `z_image_module.cpp` | capability 和 limitation |
| `z_image_gguf_module.cpp` | GGUF 独立 capability |

### 129.7 Z-Image 首发 candidate

建议按风险排序：

~~~text
Z1 BF16 safetensors component-staged plan-only -> guarded
Z2 BF16 safetensors block-streamed
Z3 GGUF fixed quant variant block-streamed
Z4 ConvRot packed Q8
Z5 NVFP4
LoRA / image input / hybrid 后续
~~~

每条路线独立 L2/L3，不得用 BF16 证据替 GGUF/ConvRot/NVFP4 背书。

## 130. API、结果协议、错误合同与服务生命周期

### 130.1 公共输入保持最小

首版公共输入继续只暴露：

~~~json
{
  "memory_constrained": {
    "enabled": true,
    "limit_bytes": 17179869184,
    "buffer_percent": 15,
    "min_free_bytes": 1073741824,
    "max_refill_slots": 3,
    "allow_quality_preserving_tiling": true
  }
}
~~~

用户不应选择 pinned block 数、stage reserve、manifest path、unsafe fallback 或
`allow_experimental_guarded`。这些属于 planner/capability，不是稳定 API。UI 可以展示
“高性能/平衡/更低内存”预设，但最终仍映射到 Y/X/slot 上限，不能直接绕过能力门禁。

### 130.2 输入语义补充

- `limit_bytes=Y` 是该请求相关进程集合的目标上限，不是“额外可用内存”；
- `buffer_percent=X` 是从 Y 中留出的用户冗余，effective B 为 `Y*(100-X)/100`；
- `min_free_bytes` 是系统 admission 条件，不从 B 再扣一次；
- `max_refill_slots` 是上限，effective slots 由 planner 选择；
- `allow_quality_preserving_tiling=false` 时，依赖 tile 才能 fit 的 candidate 被拒绝；
- `enabled=false` 时其他字段只做格式校验，不改变 route；
- profile 和 request 合并后输出每个字段的 source，便于解释；
- 相同 request 的 frozen effective plan 可序列化，执行时不得重新解释 profile。

### 130.3 Plan 与 result 分开

plan 输出“准备怎么做”，result 输出“实际做了什么”。建议字段：

~~~json
{
  "memory_policy": {
    "enabled": true,
    "policy_digest": "...",
    "candidate": "...",
    "certification_state": "experimental_guarded",
    "capability_level": "allocation_guarded",
    "execution_supported": true,
    "release_stable": false,
    "manifest_digest": "...",
    "evidence_digest": "",
    "planned_peak_bytes": 0,
    "planned_increment_bytes": 0,
    "effective_refill_slots": 2,
    "effective_tiling": "none",
    "peak_epoch": "...",
    "enforcement_scope": "inference_process_tree_v1"
  }
}
~~~

result 的 `memory_admission` 增加：

~~~json
{
  "actual_plan_revision": 0,
  "ledger_peak_committed_bytes": 0,
  "peak_process_footprint_bytes": 0,
  "peak_unattributed_process_footprint_bytes": 0,
  "required_unknown_bytes": 0,
  "optional_unknown_bytes": 0,
  "pending_release_peak_bytes": 0,
  "implicit_cleanup_count": 0,
  "replan_count": 0,
  "pressure_state_peak": "normal",
  "swapins_delta": 0,
  "swapouts_delta": 0,
  "observed_within_budget": true,
  "envelope_violations": []
}
~~~

swap delta 属于 evidence/diagnostic，不应由 library 在每个请求强制采样；service/campaign
可采集。若无权限或平台不支持，字段为 unavailable，不伪造 0。

### 130.4 计划 revision

runtime 只能从 plan 中声明的 revision 切换：

~~~json
{
  "allowed_runtime_revisions": [
    {
      "revision": 0,
      "slots": 3,
      "pinned_blocks": 8,
      "tiling": "none",
      "planned_peak_bytes": 0
    },
    {
      "revision": 1,
      "slots": 2,
      "pinned_blocks": 4,
      "tiling": "none",
      "planned_peak_bytes": 0
    },
    {
      "revision": 2,
      "slots": 2,
      "pinned_blocks": 2,
      "tiling": "spatial_v1",
      "planned_peak_bytes": 0
    }
  ]
}
~~~

revision id、切换安全点、触发原因写入 trace。没有更低 revision 时必须 abort，不创建
未计划的第四种策略。

### 130.5 错误码

对外错误使用稳定 code + 可读 message + structured details：

| Code | 阶段 | 含义 |
|---|---|---|
| `memory_policy_invalid` | parse/normalize | 字段非法、overflow、Y/X 无效 |
| `memory_policy_conflict` | normalize | legacy budget、ANE、LoRA 等冲突 |
| `memory_policy_unsupported` | route/capability | 模型/后端/shape 无执行能力 |
| `memory_manifest_invalid` | metadata | manifest/site/identity 不一致 |
| `memory_estimate_unknown` | plan | required upper 缺失或 heuristic-only |
| `memory_budget_too_small` | plan/admission | 所有合法 candidate 均不 fit |
| `memory_system_pressure` | admission/runtime | 系统余量/pressure 不允许继续 |
| `memory_policy_allocation_failed` | runtime | reservation 成功但实际 allocator 失败 |
| `memory_envelope_violation` | runtime | actual/observed 超 validated upper |
| `memory_lifetime_violation` | runtime/cleanup | token、generation、last-use 违规 |
| `memory_checkpoint_changed` | metadata/load | TOCTOU 或 digest 变化 |

details 示例：

~~~json
{
  "code": "memory_budget_too_small",
  "candidate": "ltx_...",
  "required_bytes": 14800000000,
  "available_bytes": 14600000000,
  "peak_epoch": "stage1_to_stage2_upsample",
  "largest_resources": [
    {"site_id": "ltx.stage2.latent", "bytes": 0}
  ],
  "suggestions": [
    "increase limit_bytes",
    "increase buffer budget by lowering buffer_percent within allowed range",
    "enable quality-preserving tiling if available"
  ]
}
~~~

建议必须真实可执行；没有已认证 tile 时不得建议打开 tiling。

### 130.6 Trace 事件 schema

trace 使用固定容量 ring，事件不保存 prompt、文件绝对路径或 tensor 数据：

~~~cpp
struct MemoryTraceEvent {
    uint64_t monotonic_ns;
    uint64_t job_generation;
    uint32_t sequence;
    TraceEventKind kind;
    uint32_t stage;
    uint32_t instance;
    uint64_t bytes;
    uint64_t committed_after;
    uint64_t process_footprint;
    FixedString<64> site_id;
    FixedString<32> reason;
};
~~~

release build 默认只输出聚合指标；失败时保存最后 N 个事件。测试/benchmark build 可
打开完整 trace。固定字符串避免 trace 本身产生不可控 heap allocation。

### 130.7 Service 调度

当前 service 已串行 GPU job 并区分 resident session/worker。受限模式建议：

1. 在 job 入队时只做 schema 校验；
2. 真正执行前重新采集 system/process admission；
3. constrained job 不与另一个 resident model session 共存，除非其 backing 已入账；
4. 首发优先 dedicated disposable worker，成功后也退出，获得最清晰的进程回收边界；
5. 当某 candidate 已证明 persistent session cache accounting 正确后，才开放复用；
6. job 取消时向 child 发协作取消，超时后终止 child，并记录非优雅清理；
7. worker 退出码、signal、peak footprint、stderr 尾部进入 evidence；
8. service 自身 baseline 和 worker baseline 分别计量，process tree 口径明确。

受限 LTX 请求不得被现有 resident-candidate 环境开关吸入常驻路线；route inspector 必须
以 frozen plan 的 candidate 为准，而不是再次解析原 JSON。

### 130.8 Cache 策略

首发统一原则：

~~~text
disabled 请求：完全保持现有 cache
constrained certified 请求：默认 request-local cache
跨请求 cache：必须有 StorageLease::cache、quota、identity 和 eviction
~~~

允许缓存时，plan 必须同时给出 cold 和 warm peak。warm request admission 要把已有 cache
作为 process baseline 中的 controlled backing；budget 变小或 checkpoint 变化时先驱逐，
不能为了命中 cache 拒绝一个本可通过冷加载执行的请求。

### 130.9 输出/媒体生命周期

视频和图片保存是峰值的一部分：

- H3/LTX MP4：decoded frame、RGB/YUV conversion、编码器 input、mux buffer；
- Flux/Z-Image PNG：decoded tensor、NHWC transpose/materialize、PNG row/filter/compress
  buffer；
- dump/preview：首发 constrained 禁止或单独 capability；
- 输出文件写入失败仍必须释放媒体 backing；
- 流式媒体 consumer 只有在 backpressure、最大队列深度、frame ownership 都有界时才
  能降低 output upper；
- 磁盘空间不足不是 memory fallback 理由，请求正常失败。

## 131. 测试、基准、故障注入与验收方法

### 131.1 测试金字塔

| 层级 | 目的 | 是否需要真实模型/GPU |
|---|---|---|
| L0 schema/static | 字段、route、旧 API、site registry | 否 |
| L1 pure planner/ledger | arithmetic、live interval、事务、fault | 否 |
| L2 allocator/backend | Metal buffer、completion、MPSGraph envelope | 需要 Metal，部分需小 fixture |
| L3 end-to-end | checkpoint、质量、内存、swap、性能 | 需要目标模型和目标机器 |

任何 L2 因无 Metal 而跳过必须显示 SKIP，不得记 PASS。L3 没有模型资产同理。

### 131.2 新增测试文件

建议新增：

~~~text
tests/native/memory_policy_test.cpp
tests/native/memory_manifest_test.cpp
tests/native/memory_plan_compiler_test.cpp
tests/native/memory_scheduler_test.cpp
tests/native/memory_pager_test.cpp
tests/native/test_memory_site_registry.py
tests/native/test_memory_default_abba.py
tests/native/test_memory_campaign.py
tests/native/h3_host_memory_hooks_test.cpp
tests/native/ltx_external_mtl_buffer_test.mm
tests/native/ltx_mpsgraph_memory_envelope_test.mm
tests/native/flux_memory_manifest_test.cpp
tests/native/z_image_memory_manifest_test.cpp
~~~

现有 `memory_accounting_test.cpp`、H3/LTX GPU hook test 和 contract test 继续扩展，不
另建重复测试框架。

### 131.3 Planner/ledger 必测边界

至少覆盖：

1. Y*multiplier overflow、零值、5/30 边界；
2. resource size 乘法/加法 overflow；
3. alias 折叠与实际不同 handle 拒绝；
4. half-open interval、preload overlap、completion 延长；
5. 相同 epoch 的 barrier/non-barrier 顺序；
6. reserve 恰好等于剩余预算成功，+1 byte 失败；
7. commit actual==upper 成功，upper+1 失败；
8. duplicate commit/release、stale generation；
9. pending completion 在 stage end 被拒绝；
10. cache reactivate identity mismatch；
11. required manifest site 删除后 execution false；
12. heuristic provenance 永远不能打开 release；
13. candidate 排序在输入顺序变化后仍确定；
14. plan digest 任一关键字段改变后变化；
15. disabled request 不创建 ledger/trace。

### 131.4 Source audit

`test_memory_site_registry.py` 可做静态检查，但不要只用脆弱的字符串总数。建议：

- 扫描 H3/LTX 首发 reachable 文件中的旧 allocation API；
- 允许列表包含函数名、原因和过期日期/issue；
- 每个 classified site 必须在 registry 存在；
- registry required site 必须在源码或 aggregate envelope 中有 owner；
- `newBufferWithLength` direct call 只允许出现在统一 wrapper 或明确 unsupported 文件；
- `malloc/realloc` 大对象路径需使用 wrapper 或有 aggregate envelope 注释；
- Flux/Z-Image pager route 禁止 block graph capture 全局 `Weights`。

静态扫描只证明调用点形态；最终仍需运行时 required unknown=0。

### 131.5 Synthetic backend

建立不依赖 Metal 的 fake adapter：

~~~cpp
class FakeMemoryAdapter {
    // 可配置 partition bytes、I/O 延迟、compute 延迟、失败序号、
    // actual>upper、stale completion、pressure 事件。
};
~~~

用虚拟时钟验证：

~~~text
双槽 overlap
三槽无额外收益时不选择
预算下降后 shrink lookahead
cancel while reading
cancel while GPU in use
stage handoff live-set
fault 后 exactly-once cleanup
~~~

这样 scheduler 核心不依赖昂贵实机测试。

### 131.6 Metal L2

在有 Metal 的机器运行：

- reserve denied 时 `MTLBuffer` allocation counter 不增加；
- actual `buffer.length` 与 commit 一致；
- shared/private storage mode 均纳管；
- command buffer completion 前 release 进入 pending；
- completion 后 storage bytes 下降；
- H3 classified upload 内容正确；
- LTX external buffer wrapper NSArray/ObjC 生命周期正确；
- graph plan cache clear 后 private footprint 回落到 envelope；
- fault/cancel 后下一请求不复用 tainted buffer。

若系统 API 不能直接读取所有 private allocation，使用 ledger + process footprint delta
双证据，并在 manifest 标注来源限制。

### 131.7 Budget boundary 搜索

对每个 candidate/shape，不建议连续二分任意 byte，因为 slot/pinned/tile 是离散跳变。
流程：

1. 枚举 planner 产生的所有 plan revisions；
2. 计算每个 revision 理论 minimum；
3. 测试 `minimum-1 byte`：必须在大 allocation 前拒绝；
4. 测试 `minimum`：允许出现 allocator alignment 导致的显式 envelope violation，但
   不能越 B 后继续；
5. 测试 `minimum + alignment/headroom`：必须稳定成功；
6. 测试用户典型 Y 档位；
7. 记录实际 selected revision，而不是假设预算单调映射到相同策略。

若 minimum 恰好成功率不稳定，说明 upper/alignment 未建模，不能通过提高隐藏 margin
掩盖；应修正 manifest。

### 131.8 L3 campaign 矩阵

H3/LTX 每条准备发布的 candidate 至少覆盖：

| 维度 | 样本 |
|---|---|
| cache | cold process、第二次 warm（若允许 cache） |
| budget | minimum-1、minimum、minimum+headroom、典型 Y |
| shape | 每个认证 bucket 的最小/典型/最大 |
| prompt | 较短、token bucket 边界、UTF-8 长 prompt |
| run | success、cancel text、cancel denoise、cancel decode、output failure |
| pressure | normal、warning 注入、真实后台压力的受控场景 |
| I/O | cold-ish、warm page cache、短读/文件变化 fault |
| repeat | 至少 5 次；边界预算建议 20 次 |

Flux/Z-Image 在 plan-only 阶段只做 metadata/planner；进入 guarded 后采用对应 image 矩阵。

### 131.9 内存与 swap 采样

campaign runner 建议同时采集：

~~~text
process physical footprint（主 worker/child）
ledger committed/reserved/pending/cache
MLX active/peak/cache（适用时）
Metal 可观测 allocated size（适用时）
system available/pressure
vm_stat swapins/swapouts 起止增量
vm.swapusage 辅助快照
stage/block trace
~~~

采样间隔建议 50–100 ms 作为观测工具，但不能把“没有采到尖峰”当作硬证明。硬证明来自
reserve-before-allocate 和 validated graph envelope。采样线程自身内存固定、低优先级，
结果写入预分配 ring，避免影响被测峰值。

### 131.10 性能指标

至少计算：

~~~text
end-to-end wall
text/load/denoise/decode/export 分段
block compute P50/P95
read/convert P50/P95
refill wait ratio
overlap hidden fraction
bytes read per denoise step
slot utilization
pinned hit ratio
replan count
cache cold/warm delta
energy（若稳定可采）
~~~

overlap：

~~~text
hidden_fraction =
  1 - unhidden_refill_wait / max(total_refill_io_convert, epsilon)
~~~

不使用“GPU utilization 单一百分比”评价，因为 unified memory、I/O 和 command 提交可能
让该指标误导。

### 131.11 性能门槛

硬门禁：

~~~text
disabled ABBA 稳态 wall 回归 <= 2%
disabled 路径无额外模型 load/cache clear/route 变化
constrained ledger peak <= B
required_unknown_bytes == 0
observed footprint 不得已知越 B
安静基线 campaign swapouts delta == 0
质量合同通过
~~~

性能目标（需按模型证据固化，不作为首次代码合入硬常量）：

~~~text
full-resident revision 相对 disabled 同 route 回归 <= 3%
streamed denoise unhidden refill wait <= denoise wall 的 15%
稳定区间 hidden_fraction >= 0.70
取消到安全退出 P95 <= 一个 block/tile 最长时间 + 2 s
连续 5 次运行无进程 footprint 单调泄漏
~~~

若受限模式与系统 swap baseline 比较，报告两者完整内存状态；不能只挑一次 swap 抖动很大
的结果宣称加速。产品价值首先是可预测和避免 swap，性能其次是在该约束下尽可能好。

### 131.12 质量验收

- 相同 seed/request/model variant；
- H3/LTX 比较 latent/checkpoint 或既有视频质量 gate；
- Flux/Z-Image 比较像素、LPIPS/SSIM/现有质量脚本；
- tiling 做 seam、边界、首尾 frame 专项；
- streaming 本身不应改变数值；若 dtype conversion 已存在，必须和同 route resident
  基线比较；
- 不把量化 candidate 与 BF16 baseline 混为“内存调度无损”；
- 输出容器/帧数/尺寸/fps/audio contract 均一致。

### 131.13 Evidence bundle

每次认证产生不可变目录：

~~~text
evidence/memory/<candidate>/<timestamp>/
  request.json
  effective-plan.json
  manifest.json
  capability-candidate.json
  build.json
  device.json
  samples.jsonl
  trace.jsonl
  result.json
  quality.json
  summary.json
  stdout.log
  stderr.log
~~~

`summary.json` 包含所有 run 的 PASS/FAIL/SKIP、最大峰值、swap delta、性能统计和 digest。
日志去除 prompt、用户名、绝对模型路径；checkpoint 只保留受控相对路径/digest。

## 132. 可直接执行的 Work Package、依赖与最终 DoD

### 132.1 总体依赖图

建议按下列顺序开发，避免模型 patch 等待“大一统 scheduler”：

~~~text
W0 default-path baseline/ABBA
 |
 +--> W1 capability hard gate
 |
 +--> W2 site registry + plan IR + pure planner
       |
       +--> W3 ledger v2/scope/pending
             |
             +--> W4 H3 GPU/host closure
             |     |
             |     +--> W7 H3 L2/L3
             |
             +--> W5 LTX GPU/host/MPSGraph closure
             |     |
             |     +--> W8 LTX L2/L3
             |
             +--> W6 scheduler/pager transaction
                   |
                   +--> W9 Flux F1/F2
                   +--> W10 Z-Image Z1/Z2/GGUF

W11 service/trace/campaign 与 W3–W10 并行接入
W12 release whitelist 在 W7/W8 后
~~~

### 132.2 W0：默认路径基线

修改/新增：

~~~text
tests/native/test_memory_default_abba.py
tests/native/benchmark_paired_app.py（扩展）
docs/evidence/default-memory-path/
~~~

任务：

1. 固定 H3/LTX/Flux/Z-Image 各一个 disabled 请求；
2. 记录 route、residency、cache、output digest、wall/stage timing；
3. 在功能分支上确认没有 manifest probe、SHA、ledger、trace；
4. 加 counter 证明 callback reserve 调用次数为 0；
5. 保存后续每个 work package 都可复用的 baseline。

DoD：

~~~text
所有默认 request JSON 行为不变
原有单测通过
稳态性能差异 <= 2%
无新增 mandatory artifact
~~~

### 132.3 W1：Capability hard gate

修改：

~~~text
native/runtime/memory_policy.hpp/.cpp
native/runtime/plan.cpp
native/platform/apple/results.mm
tests/native/test_contract.py
tests/native/memory_policy_test.cpp
~~~

任务：

- 增加 capability/certification 字段；
- 删除 heuristic 自动 `execution_supported=true`；
- 默认 registry 为空或只含测试 fixture；
- route available/estimate fits/execution supported 分离；
- H3/LTX 当前生产状态回到 plan-only；
- 错误明确说明缺 manifest/certification，而不是伪报预算不足。

DoD：没有 verified test record 时，任何 constrained execute 都在模型大分配前拒绝；
disabled 完全不受影响。

### 132.4 W2：Site registry、manifest 与 plan compiler

修改/新增：

~~~text
native/runtime/memory_manifest.hpp/.cpp
native/runtime/memory_plan.hpp/.cpp
native/runtime/memory_capability.hpp/.cpp
tests/native/memory_manifest_test.cpp
tests/native/memory_plan_compiler_test.cpp
tests/native/test_memory_site_registry.py
~~~

任务：

- 定义 resource/lifetime/provenance；
- 实现 checked arithmetic、alias folding、epoch sweep；
- metadata-only test manifests；
- candidate rejection reason；
- SHA-256 digest；
- determinism/fuzz/property tests。

DoD：纯测试可生成 H3/LTX synthetic plan，minimum-1 拒绝，输入顺序改变不影响 digest。

### 132.5 W3：Ledger v2 与 scheduler 薄层

修改/新增：

~~~text
native/runtime/memory_accounting.hpp/.cpp
native/runtime/memory_scheduler.hpp/.cpp
native/runtime/schedule_trace.hpp/.cpp
native/runtime/session.hpp
native/runtime/execution.cpp/.hpp
tests/native/memory_accounting_test.cpp
tests/native/memory_scheduler_test.cpp
~~~

任务：

- scope/site/instance；
- required/optional unknown；
- explicit implicit-cleanup metrics；
- stage token 与 pending completion；
- RAII session binding；
- taint/unload；
- disabled fast path。

DoD：synthetic success/cancel/fault 后没有 token 泄漏，stage end 能报告具体 live site。

### 132.6 W4：H3 closure

当前工作树中 classified upload、VAE/text/common DiT 和 decoded actual commit 可作为
起点。剩余拆为：

| 子包 | 代码 |
|---|---|
| W4a | `ensure_previous_velocities` 分类；旧 API reachable audit |
| W4b | `h3_video_vae_requirements` 纯函数，替换 4x heuristic |
| W4c | text/tokenizer/layout/conditioning host envelope |
| W4d | cache/sampler/env fingerprint 与 session gate |
| W4e | allocation registry、fault tests、result metrics |

DoD：

~~~text
H3 首发 candidate GPU/host required unknown == 0
reserve-before-allocate
actual<=upper
success/cancel/fault cleanup 完整
仍保持 experimental_guarded，未有 manifest/L3 不发布
~~~

### 132.7 W5：LTX closure

| 子包 | 代码 |
|---|---|
| W5a | 旧 `ltx_gpu_buffer_new` reachable audit |
| W5b | Video VAE/upsampler external MTLBuffer wrapper |
| W5c | host malloc/vector/conditioning 纳管 |
| W5d | Stage1→upsampler→Stage2 lease |
| W5e | MPSGraph cold/steady envelope |
| W5f | helper/MLX route 固定与拒绝 |

DoD：

~~~text
首发完整 backend chain 可由一个 candidate 描述
direct MTLBuffer required site 全部 guard
host required unknown == 0
MPSGraph envelope 有精确 identity
audio/I2V/LoRA/ANE/未认证 MLX 不可达
~~~

### 132.8 W6：Pager 与 overlap

不要先删除 H3/LTX 现有 loader。步骤：

1. 提取通用 slot state、ticket、trace；
2. H3 adapter 复用现有双槽；
3. LTX adapter 复用现有 2/3 槽；
4. synthetic virtual-clock 验证 look-ahead；
5. 实机调优阈值；
6. 最后决定是否抽取统一 `PartitionSource` 实现。

DoD：

~~~text
每个 prefetch 先获得 permit
slot generation 正确
GPU completion 后才复用
压力下降只切已认证 revision
I/O fault/cancel exactly-once cleanup
~~~

### 132.9 W7/W8：H3/LTX 认证

每个 candidate 分三步：

1. AllocationGuarded test capability，仅 CI fixture；
2. EnvelopeValidated development capability，绑定 manifest；
3. L3Certified release capability，绑定 evidence。

promotion 通过 code review 修改打包 whitelist，不由本机自动写入。认证 PR 必须附：

~~~text
checkpoint/runtime/device/shape identity
minimum-1 和 minimum+headroom
cold/warm/repeat/cancel/fault
peak/swap/quality/performance
remaining unsupported
~~~

### 132.10 W9：Flux

顺序：

~~~text
F0 metadata plan-only
F1 component-staged guarded
F1 L2/L3
F2 dual/single block pager
F2 compiled explicit-weight graph
F2 L2/L3
VAE tiling
LoRA/input variants
~~~

只有 F1 full transformer fit 的机器才能先获益；低于 full transformer floor 的机器等待
F2，不能让 F1 回落 unbounded。

### 132.11 W10：Z-Image

顺序：

~~~text
BF16 safetensors component-staged
BF16 34-group pager
GGUF metadata/pager
ConvRot packed Q8
NVFP4
LoRA
~~~

每个 variant 有独立 manifest/site/capability。`NativeGGUF` 的 variant selection 在
metadata phase 冻结。

### 132.12 W11：Service、campaign 与 evidence

修改/新增：

~~~text
services/turbociderd/service.mm
tools/native/memory_campaign.py
tools/native/memory_manifest_probe*
tools/native/memory_evidence_summary.py
tests/native/test_memory_campaign.py
~~~

任务：

- frozen plan 传给 worker；
- disposable constrained worker；
- process-tree sampler；
- fault env 仅测试 build；
- evidence 去隐私、digest；
- campaign resume，不覆盖旧 run；
- SKIP 与 PASS 分开；
- summary 可供 capability PR 引用。

### 132.13 W12：Release whitelist 与回滚

release capability 文件只允许：

~~~text
exact model/checkpoint
exact candidate/backend/dtype
有限 shape/token/step buckets
device family + runtime range
manifest/evidence digest
release_enabled=true
~~~

回滚只需删除/禁用 record，不需要删除实现代码。若线上发现 envelope violation：

1. 立即 quarantine 对应 digest；
2. 所有匹配请求回到 plan-only/unsupported；
3. 保留 disabled default path；
4. 分析 evidence 后重新认证；
5. 不通过增加隐藏 margin 静默恢复。

### 132.14 Code review 清单

每个内存 patch reviewer 必查：

~~~text
[ ] disabled fast path 是否保持原逻辑
[ ] allocation 前是否 reserve
[ ] upper 来源是否 executable provenance
[ ] actual capacity 是否 commit
[ ] backing 是否重复计费或漏计
[ ] GPU last-use/completion 是否正确
[ ] error/cancel 是否 cancel/release
[ ] generation/site/instance 是否稳定
[ ] cache 是否有 quota/identity
[ ] plan 与 runtime 使用同一 shape 公式
[ ] route/capability 是否在 load 前 gate
[ ] tests 是否覆盖 deny/fault/cleanup
[ ] 文档状态是否没有误报 maturity
~~~

### 132.15 最终 Definition of Done

本功能整体完成必须同时满足：

~~~text
产品：
  add-on 显式开启，输入 Y/X，错误可解释
  关闭时当前路径和性能不变
  开启时无 resident/swap fallback

规划：
  每个可执行 candidate 有 metadata-only plan
  全阶段 live-set 与 peak 可解释
  required resource 全部 executable provenance

执行：
  大 allocation reserve-before-allocate
  GPU/host/graph/output/process tree 纳入
  async release、cancel、fault 无泄漏
  observed 越线立即失败并 taint

模型：
  H3、LTX 至少各一条 GPU route L3Certified
  Flux、Z-Image 未认证路线保持 plan-only，不影响前两者发布

性能：
  默认 ABBA <=2%
  constrained overlap 达各 candidate evidence SLO
  无持续 footprint 增长

验收：
  minimum-1 早拒绝
  minimum+headroom 稳定成功
  安静环境 swapouts delta=0
  质量合同通过
  evidence 可重放、capability 可回滚
~~~

在上述条件满足之前，文档状态继续保持 `Proposal + scaffolding`。即使 H3 或 LTX 单条
route 达到 certified，也只更新该 record，不把未认证 shape、模型或后端概括成“内存
受限模式已全面支持”。

### 132.16 当前工作树的下一步建议

结合当前源码，下一组最小、可审查的提交顺序建议为：

1. 先完成 W1 capability hard gate，消除 heuristic execute 风险；
2. 完成 H3 `ensure_previous_velocities` 与剩余旧 API reachable audit；
3. 保持每个后续 patch 重跑已通过的 native build/contract/logic 套件，并在有 Metal
   device 的进程中补跑当前明确 SKIP 的 H3/LTX hook tests；
4. 实现 `h3_video_vae_requirements`，移除 4x executable provenance；
5. 实现 H3 conditioning/control host envelope；
6. LTX external MTLBuffer wrapper；
7. LTX host malloc/vector 审计；
8. 再落 plan IR/stage scheduler，不把前述闭环阻塞在大重构之后。

## 133. 本轮实现校准（2026-09-15）

本节记录本轮已经进入工作树的代码行为，作为后续实现和验收的事实基线。它覆盖的是
“硬门禁和首批 allocation hook”而不是完整的 L3 发布能力；文档整体状态仍然是
`Proposal + scaffolding`。

### 133.1 已落地的行为

| 位置 | 改动 | 运行时意义 |
|---|---|---|
| `native/runtime/memory_policy.hpp` | 增加 `MemoryCapabilityLevel`、`MemoryCertificationState`、manifest/evidence/release 字段 | 将“能估算”与“能执行”分离 |
| `native/runtime/memory_policy.cpp` | H3/LTX candidate 初始为 `hook_bridged + plan_only` | 没有验证过的 manifest 不得执行 |
| `native/runtime/plan.cpp` | 删除 `route_available && estimate_fits -> execution_supported=true` | 防止启发式误把 resident/unbounded 路径放行 |
| `native/platform/apple/results.mm` | 输出 capability/certification/digest/release 字段 | UI、service、evidence 使用同一状态合同 |
| `native/models/h3_runtime/h3_dit.c` | Euler previous velocity 使用 classified activation allocation | joint latent 路线的隐含 allocation 纳入 ledger |
| `tests/native/test_contract.py` | H3/LTX 期望 `plan_only`，并静态禁止旧晋升逻辑 | 把 fail-closed 变成回归测试 |

### 133.2 当前状态机的可观察结果

规划阶段允许返回候选，但执行入口必须遵守以下规则：

~~~text
route=false, estimate=false     -> unsupported/rejected
route=true,  estimate=false     -> plan_only (estimate unavailable)
route=true,  estimate=true,
capability=hook_bridged          -> plan_only
route=true,  estimate=true,
capability=allocation_guarded    -> experimental_guarded (仅内部开关)
route=true,  estimate=true,
capability=envelope_validated,
release whitelist 命中           -> certified
~~~

其中 `execution_supported` 只能由 capability registry 的精确匹配函数设置，而不能由
planner 的粗略公式设置。`release_stable` 只能由 `certified + release_enabled=true`
设置。任何 `plan_only` 请求在 checkpoint、MLX array、Metal buffer 或 graph 编译前都应
失败；错误必须包含 `memory_policy_unsupported` 和候选的 identity 摘要。

### 133.3 示例结果 JSON

下面是一个估算 fit 但尚未认证的 H3 规划结果。该结果可以用于 UI 预览、预算调整和
campaign 生成，但不能传给执行器：

~~~json
{
  "memory_policy": {
    "enabled": true,
    "user_limit_bytes": 51539607552,
    "effective_budget_bytes": 43808666419,
    "buffer_percent": 15,
    "route_available": true,
    "estimate_fits": true,
    "execution_supported": false,
    "capability_level": "hook_bridged",
    "certification_state": "plan_only",
    "release_stable": false,
    "adapter_candidate": "h3_c_metal_streamed_v1",
    "manifest_digest": "",
    "evidence_digest": "",
    "admission_state": "plan_only",
    "estimate_provenance": "conservative_heuristic_not_hard_limit",
    "reason": "H3 C/Metal route is plan-only: no verified capability manifest matches this checkpoint, shape, device, and runtime"
  }
}
~~~

## 134. Capability manifest 的最小可实现规范

### 134.1 Manifest 的职责边界

Manifest 不是模型权重索引，也不是用户可编辑的“强制放行开关”。它是一个由构建工具
和认证 campaign 生成、由 runtime 只读验证的 capability record，回答四个问题：

1. 这条 route 是否覆盖当前 checkpoint、backend、dtype、shape 和 runtime？
2. 所有 required allocation site 是否都有 executable upper、lifetime 和 backing 归属？
3. 认证过程中是否观测到 `actual <= upper`、无 swap 增量、质量合同通过？
4. 该 record 是否被 release whitelist 启用，是否可以在异常时回滚？

manifest 不应保存 prompt、输出路径或其他隐私数据；请求级动态字段只参与 candidate key
和 plan digest，不写入持久 evidence。

### 134.2 推荐 JSON schema（v1）

~~~json
{
  "schema": "turbocider.memory_capability.v1",
  "record_id": "h3_c_metal_streamed_v1.m5.max.512x512x22.s4",
  "state": "certified",
  "level": "l3_certified",
  "release_enabled": true,
  "adapter": "h3_c_metal_streamed_v1",
  "identity": {
    "model": "minimax-h3-turbo",
    "checkpoint_digest": "sha256:...",
    "model_variant": "bf16",
    "backend": "c_metal",
    "dtype": "bf16",
    "operation": "video.generate",
    "device_family": "AppleGPU-M5",
    "runtime_revision": "tc-native-<git-sha>"
  },
  "shape_domain": {
    "width": {"min": 512, "max": 512, "step": 64},
    "height": {"min": 512, "max": 512, "step": 64},
    "frames": {"min": 22, "max": 22, "step": 1},
    "steps": {"min": 4, "max": 4, "step": 1},
    "audio": false,
    "inputs": "empty",
    "loras": "empty"
  },
  "budget": {
    "user_limit_bytes": 51539607552,
    "buffer_percent": 15,
    "effective_budget_bytes": 43808666419,
    "system_reserve_bytes": 1073741824,
    "non_denoiser_reserve_bytes": 4294967296,
    "planned_upper_bytes": 40265318400,
    "validated_peak_bytes": 38923141120
  },
  "resources": [
    {
      "site": "h3.dit.activation.previous_video_velocity",
      "class": "activation",
      "upper_expr": "video_rows * VIDEO_PATCH * 2",
      "lifetime": ["denoise[0]", "denoise[last]"],
      "backing": "metal_buffer",
      "required": true
    }
  ],
  "evidence": {
    "manifest_digest": "sha256:...",
    "evidence_digest": "sha256:...",
    "campaign_id": "2026-09-15-h3-m5-512",
    "quality_contract": "sha256:...",
    "swap_delta_bytes": 0,
    "fault_cleanup_pass": true
  }
}
~~~

实现上不要求第一版支持任意表达式。`upper_expr` 应由受信任的 C++ calculator 解析为
有限的变量乘加树，拒绝函数调用、文件读取、浮点 NaN 和未声明变量。manifest 中的
`shape_domain` 也应在解析时归一化成闭区间，避免 `max`、`step` 的边界歧义。

### 134.3 Candidate key 和 digest

candidate key 必须由以下字段按固定顺序拼接并做 SHA-256：

~~~text
schema | adapter | model | checkpoint_digest | model_variant | backend |
dtype | operation | width | height | frames | steps | audio |
input_kinds | lora_identity | device_family | runtime_abi | scheduler_revision
~~~

`manifest_digest` 绑定资源 site 的排序结果、upper/lifetime、slot 数量、tiling revision
和 allocator domain。`evidence_digest` 绑定 campaign summary、二进制 revision、设备
family、系统版本和采样器版本。任何一个字段改变都必须使 record miss，而不是自动扩大
上限。

### 134.4 Runtime 匹配伪代码

~~~cpp
std::optional<MemoryCapabilityRecord> CapabilityRegistry::match(
    const FrozenPlan &plan, const DeviceIdentity &device) const {
    for (const auto &record : records_) {
        if (!record.release_enabled && !options_.allow_experimental)
            continue;
        if (!record.identity.matches(plan, device))
            continue;
        if (!record.shape_domain.contains(plan.shape))
            continue;
        if (record.manifest_digest != plan.manifest_digest)
            continue;
        if (record.level < MemoryCapabilityLevel::EnvelopeValidated)
            continue;
        return record;
    }
    return std::nullopt;
}
~~~

匹配失败时必须返回可诊断原因（例如 checkpoint digest mismatch、shape outside
domain、device family mismatch、manifest digest missing），但不能透露完整本地路径。

## 135. Planner 与执行器的代码修改建议

### 135.1 将四类状态拆开存储

短期可以继续使用 `EffectiveMemoryPolicy` 保持 ABI 兼容；中期应按下列结构重构：

~~~cpp
struct MemoryBudgetContract { /* Y/X/B、system reserve、slot limit */ };
struct MemoryCandidate { /* adapter + normalized request + estimate */ };
struct MemoryCapability { /* level/state/digests/release whitelist */ };
struct MemoryRuntimeState { /* admitted/tainted/actual peak/overflow */ };

struct RequestMemoryContext {
    MemoryBudgetContract contract;
    MemoryCandidate candidate;
    MemoryCapability capability;
    MemoryRuntimeState runtime;
    MemoryLedger ledger;
};
~~~

`plan.cpp` 只构造前三者，不得修改 `execution_supported=true`。`c_api.mm` 的
`prepare_memory_admission()` 在大 allocation 前执行 capability match、clean-baseline
采样和 ledger 创建；只有 match 返回 `EnvelopeValidated` 或 `L3Certified` 时才创建
admission binding。

### 135.2 执行入口的顺序约束

所有 GPU-only constrained 请求必须遵循以下严格顺序：

~~~text
parse -> normalize -> metadata plan -> estimate -> capability match
  -> acquire device -> clean baseline -> reserve process envelope
  -> bind admission hooks -> load checkpoint through guarded pager
  -> allocate stage resources -> execute -> drain completions
  -> verify actual/upper and swap delta -> finalize evidence -> unbind
~~~

不能在 capability match 之前调用 `module_for(...).load()`、MLX `array` 构造、
`newBufferWithLength`、MPSGraph compile 或 helper 子进程。对于 plan-only 请求，入口应在
“acquire device”之前失败，从而避免一次失败请求污染 page cache 或模型缓存。

### 135.3 请求幂等和冻结

当前 session 仍可能在 `generate()` 内再次调用 `make_plan()`。迁移期间必须确保：

- 二次 plan 只接受第一次生成的 `normalized_request` 和 `plan_digest`；
- 二次 plan 不重新选择 backend、residency、slot 数或 tile revision；
- 环境变量只能在 metadata 阶段读取一次，并写入 `runtime_revision`；
- 如果二次 plan 的 digest 不一致，直接报 `memory_plan_changed`，不尝试合并。

建议新增接口：

~~~cpp
RunResult generate_planned(const ExecutionPlan &frozen_plan,
                           RequestMemoryContext &,
                           const RunEvent &,
                           std::atomic<bool> &cancelled);
~~~

旧 `generate(Request, ...)` 只作为 disabled fast path 和迁移期兼容入口。受限模式下不
允许旧入口隐式回退。

## 136. 预算、ledger 和异步 allocation 的实现细节

### 136.1 Y/X/B 的统一公式

用户输入 `Y` 是进程级软上限，`X` 是冗余百分比，`B = floor(Y * (100-X)/100)`。
运行时还必须计入 `P`（clean process baseline）和 `R`（非 denoiser 组件），因此
denoiser 可用的进程内预算为：

~~~text
D = B - P - R
~~~

`S`（system minimum free）是独立的全系统准入条件，不从 `B` 再次扣除；准入时另行要求
`system_available_after_commit >= S`。这样避免把用户 buffer、进程预算和系统余量重复
扣除。若 `D <= 0` 或独立系统准入失败，必须在任何 checkpoint allocation 前拒绝。当前
`plan.cpp` 的
`memory_estimate_bytes` 仍是 heuristic upper，只能用于 metadata plan；真正可执行的
candidate 必须由 manifest 提供 `planned_upper_bytes`，并由 admission 在运行时重新扣除
实际 `P`。

### 136.2 资源分类和重复计费规则

统一分类如下：

| class | 典型资源 | 归属原则 |
|---|---|---|
| `weights` | checkpoint page、packed block | 同一 backing 只计一次；映射和 tensor view 不重复计费 |
| `activation` | DiT hidden、velocity、attention scratch | 按 live interval 计费 |
| `conditioning` | text/context/rope | prompt-bound cache 必须有 quota 和 identity |
| `refill_slot` | pager 双/三槽 | slot backing 归 slot，不再记作 weights |
| `conversion_scratch` | fp32↔bf16、quant unpack | completion 前保持 live |
| `output` | decoded frame、host envelope | 以真实 shape commit，不能使用最大 shape 猜测 |
| `graph_workspace` | MPSGraph cold/steady workspace | cold 与 steady 分开记录，不能用一次 warm 值替代 cold |

每个 reservation 必须有 `site_id`、`instance_id`、`generation`、`upper_bytes`、
`backing_id` 和 `scope_token`。同一 `backing_id` 的多个 view 通过 alias 表共享一项；
不同 generation 即使物理地址复用，也必须先完成旧 generation 的 GPU event。

### 136.3 Reserve/commit/release 状态机

~~~text
EMPTY -> RESERVED -> COMMITTED -> PENDING_RELEASE -> RELEASED
             |             |                |
             +-> CANCELLED +-> TAINTED      +-> FORCE_RELEASE (fault only)
~~~

- `reserve` 只增加 `reserved_bytes`，不增加 actual/live；失败不允许触发 allocator。
- `commit` 必须携带实际 backing handle 和 actual bytes，校验 `actual <= upper`。
- `release` 只能由 owner 或完成事件触发，重复 release 必须幂等并计数。
- `TAINTED` 资源不能回收到正常 slot pool，必须销毁或隔离到 fault cleanup。
- 发生 upper violation 时，先标记 job/worker tainted，再异步清理；不能继续下一 stage。

### 136.4 Pending completion 与 Metal event

Metal command buffer 未完成前不能释放或复用 slot。推荐每个 commit 返回
`AllocationTicket`，其中保存 `MTLCommandBuffer` completion handler 或共享 event value：

~~~cpp
struct AllocationTicket {
    uint64_t token_id;
    uint64_t generation;
    uint64_t backing_id;
    uint64_t upper_bytes;
    uint64_t actual_bytes;
    std::shared_ptr<Completion> completion;
};
~~~

`WeightPager` 的 `acquire(slot)` 必须等待 `slot.generation-1` 的 completion；仅在 CPU
对象析构并不代表 GPU 使用结束。测试中要注入“CPU release 早于 GPU completion”，确保
ledger 仍然保持 live，避免复用导致数据竞争或漏计。

## 137. H3 首发路线的剩余施工图

### 137.1 当前已完成与待完成

已完成：classified upload API、Video VAE required GPU tensor、text encoder 主要
allocation、DiT common path、joint latent/RGB8/decoded host envelope、按真实 frame shape
commit、Euler previous velocity 分类。本轮新增的两个 site 为：

~~~text
h3.dit.activation.previous_video_velocity
h3.dit.activation.previous_audio_velocity
~~~

仍待完成：

1. `h3_video_vae_requirements()` 纯函数和 decoder/encoder 的 exact upper；
2. conditioning、tokenizer、layout/control host malloc/vector envelope；
3. H3 cache、sampler、zero-copy/map weight 的 backing identity；
4. `h3_gpu_tensor_new_bf16()` 剩余 reachable call-site audit；
5. 首发 candidate 的 required unknown 必须变成 0；
6. manifest digest、device family 和 MPSGraph/Metal runtime identity 绑定。

### 137.2 旧 API reachable audit 方法

不能只用 `grep` 证明安全。建议构建一个 constrained symbol audit：

~~~text
1. 编译 H3 candidate 时定义 H3_MEMORY_CONSTRAINED=1
2. 通过 linker map 导出所有 h3_gpu_tensor_new_* 调用者
3. 对每个调用点记录 stage、条件分支、memory class、site id
4. 对 CoreML/ANE、debug、benchmark、fallback 分支标注不可达证明
5. 若无法证明不可达，必须迁移到 classified API 或将 candidate 降级 plan_only
~~~

审计输出建议为 `build/evidence/h3/reachable-sites.json`，内容包含 source revision、
编译 flags、调用点列表和 `required_unknown_count`。CI 只允许首发 manifest 在该字段为 0
时进入 release whitelist。

### 137.3 VAE exact upper 公式

推荐把 `predicted * 4` 替换成按 decoder block 的 live interval 公式：

~~~text
decoded_host_upper = align64(frame_count * decoded_width * decoded_height * channels * bytes)
                   + row_scratch_upper
                   + color_convert_upper
                   + encoder/decoder overlap upper
~~~

`decoded_host_upper` 必须由纯函数计算，输入只包含 normalized shape、pixel format、
tiling revision 和 output mode。真实 commit 使用 allocator 返回的 frame shape；若实际
超过 upper，立即 taint，不得扩大 reservation 继续运行。

### 137.4 H3 pager 初始策略

首发不应一次性重写所有 loader。先把现有双槽抽象为：

~~~cpp
struct H3BlockSource {
    uint32_t block_id;
    uint64_t file_offset;
    uint64_t upper_bytes;
    h3_gpu_dtype dtype;
    std::string site_id;
};

class H3WeightPager {
    SlotLease prefetch(const H3BlockSource &, uint64_t generation);
    void mark_last_use(const SlotLease &, MTLCommandBuffer *);
    void evict_after_completion(const SlotLease &);
};
~~~

look-ahead 默认 1 个 block；只有 observed I/O latency 大于 GPU compute gap 时才提升到
2。slot 数受 `max_refill_slots` 和 manifest verified slots 双重限制，不能因为当前空闲
就动态创建第 4 个槽。

## 138. LTX 首发路线的剩余施工图

### 138.1 Direct `MTLBuffer` 统一包装

当前重点不是增加更多 loader，而是消灭不受 admission 管理的 direct allocation。建议
在 `ltx_gpu.m/.h` 增加：

~~~c
ltx_gpu_buffer *ltx_gpu_buffer_new_classified(
    ltx_gpu *, size_t bytes, ltx_gpu_memory_class, const char *site);
ltx_gpu_buffer *ltx_gpu_buffer_new_copy_classified(
    ltx_gpu *, const void *, size_t, ltx_gpu_memory_class, const char *site);
~~~

Video VAE、upsampler、MPSGraph binding 和 Stage2 text context 均不得直接调用
`newBufferWithLength`。source audit 要把 direct call 计数降为 0，或给每个不可达分支附
静态证明。

### 138.2 LTX host allocation envelope

必须纳入 ledger 的 host 资源包括 Gemma tokenizer 输出、conditioning vectors、stage1/2
latent、upsampler temporary、音频禁用路径的 mux scratch。`std::vector` 不可直接按
`capacity()` 估算后忘记增长；推荐使用带 owner 的 allocator：

~~~cpp
template<class T>
using AccountedVector = std::vector<T, LedgerAllocator<T>>;
~~~

扩容流程必须是 `reserve(new_capacity)` -> ledger reserve -> allocator allocate -> commit，
不能先让 STL 扩容再补账。失败时保留旧 vector，避免半扩容状态破坏 stage handoff。

### 138.3 Stage1→upsampler→Stage2 lease

LTX video-only 首发建议固定为：

~~~text
text_encode
  -> stage1_denoise (latent lease A)
  -> stage1_end checkpoint
  -> upsampler (lease A read-only, scratch B)
  -> stage2_denoise (lease C; A only if manifest says alias-safe)
  -> video_vae_decode
  -> output commit
~~~

若 Stage2 允许复用 Stage1 backing，必须在同一 command queue 上插入 completion fence，并
在 manifest 中声明 alias relation；否则默认复制到新 backing 并将复制成本计入 upper。

### 138.4 MPSGraph cold/steady envelope

LTX 的 graph workspace 必须有两个独立数字：

- `cold_compile_upper`：第一次 graph compile、constant folding、pipeline cache 建立；
- `steady_dispatch_upper`：后续 dispatch 的临时 workspace。

受限模式不能用 warm benchmark 代替 cold upper。若当前请求要求 cold compile 而 manifest
只验证 steady，必须返回 `plan_only` 或先在受控的 disposable worker 中完成 compile，再
重新建立 baseline。

## 139. StageScheduler 与 WeightPager 的并发算法

### 139.1 三队列模型

每个 stage 使用三条逻辑队列：

~~~text
I/O queue       读取/解压/转换下一 slot
encode queue    将 slot 绑定到 command buffer
completion queue 等待 GPU 完成并释放上一 slot
~~~

队列之间只通过 `AllocationTicket` 和 generation 传递，不直接共享裸指针。I/O queue
拿不到 permit 时必须阻塞或取消预取，不能把数据先读进未计账的 host 临时 buffer。

### 139.2 Look-ahead 决策伪代码

~~~cpp
while (!stage.done()) {
    auto next = stage.peek_next_block();
    if (pager.can_prefetch(next) &&
        ledger.available_for(next.upper_bytes, stage.guard_bytes())) {
        auto lease = pager.reserve_slot(next);
        io.submit(next, lease);
    }

    auto ready = pager.ready_for_encode();
    if (ready) {
        auto ticket = ledger.commit(ready.token(), ready.actual_bytes());
        encode.submit(ready.buffer(), ticket);
    }

    completion.poll([&](auto ticket) {
        pager.release_after_completion(ticket);
        ledger.release(ticket);
    });
}
~~~

### 139.3 Pressure controller

压力控制器每 50–100 ms 采样一次，输入包括 ledger live/reserved、Metal resident bytes、
process footprint、free memory、I/O queue depth 和最近三次 stage latency。状态分为
`normal`、`tight`、`critical`：

| 状态 | 进入条件 | 动作 |
|---|---|---|
| normal | `live + reserved <= 0.85B` 且无系统压力 | 保持 manifest revision 和 look-ahead |
| tight | `>0.85B` 或 free 低于 `S + guard` | 停止新增预取，等待 completion，必要时降为单槽 |
| critical | `>B`、allocation fail 或 swap delta 非零 | 立即 taint，取消未提交 I/O，失败请求 |

必须使用双阈值和最小保持时间（建议 3 个采样周期）反抖，不能在 normal/tight 间来回
切换导致 slot churn。压力控制器只能选择 manifest 已验证的 revision；不得临时改变
dtype、tile 或 block 数来“自救”。

### 139.4 取消和 fault cleanup

取消顺序固定为：停止接受新 stage -> cancel I/O -> 标记未提交 token cancelled -> 等待
已提交 command buffer completion -> release committed tokens -> destroy tainted resources。
超时后由 disposable worker 退出兜底；daemon 不得把未完成 token 留给下一请求。

## 140. Flux 与 Z-Image 的落地边界

### 140.1 Flux

Flux 先做 F0 metadata-only，原因是 transformer、text encoder、VAE 和 compile workspace
的 combined upper 尚未有完整 manifest。F1 只允许 component-staged，要求 transformer
本身在有效预算内完整驻留；低于该 floor 的机器保持 `plan_only`，不能回到旧 resident
路径。F2 才实现 block pager：

~~~text
text encoder -> transformer block group pager -> VAE tiled decode
~~~

每个 block group 独立记录 quantization、rope cache、attention scratch 和 compile
revision。`compile_gpu=true` 只在 graph 支持 explicit weight binding 且 manifest 声明
workspace upper 时可用，否则切到 eager block pager，而不是静默 resident。

### 140.2 Z-Image

Z-Image safetensors、GGUF、ConvRot packed Q8、NVFP4 必须是独立 candidate；不能共用一个
“Z-Image supports constrained”布尔值。variant selection 在 metadata 阶段冻结，随后
manifest 必须同时匹配：

~~~text
checkpoint format + quant scheme + group size + packed layout + VAE revision + LoRA mode
~~~

GGUF 的 mmap、dequant scratch 和 block cache 需要明确 backing；如果无法证明 mmap page
不会形成额外 resident 峰值，则使用 explicit read/stream buffer 并把 page-cache 风险写入
evidence。NVFP4 先保持 plan-only，直到有真实 device 上的 quality、peak、swap campaign。

## 141. 测试与验收矩阵（实现者可直接照此执行）

### 141.1 L0：纯逻辑和契约

~~~sh
python3 -B tests/native/test_contract.py
python3 -B tests/native/test_memory_accounting.py
git diff --check
~~~

必须覆盖：Y/X/B 边界、`D<=0`、disabled fast path、plan-only candidate、manifest miss、
double release、cancel/fault cleanup、旧 heuristic 晋升逻辑静态不存在。当前 16 GiB Flux
示例因保守 upper 超预算应为 `rejected`，不能再标成 `estimate_only`。

### 141.2 L1：源码和构建完整性

~~~sh
env MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" \
  TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
python3 -B tests/native/test_h3_streaming_policy.py
python3 -B tests/native/test_ltx_streaming_benchmark.py
~~~

要求 native build 成功，disabled 请求的 plan/result JSON 与改造前 ABI 字段保持兼容，且
未打开 constrained 时不创建 `MemoryAdmission`、不清空 resident session、不改变 cache
策略。

### 141.3 L2：Metal 设备测试

~~~sh
python3 -B tests/native/test_h3_gpu_memory_hooks.py
python3 -B tests/native/test_ltx_gpu_memory_hooks.py
~~~

无 Metal device 时只能输出明确 `SKIP: no Metal device is available`，不能伪造 PASS。
有设备时必须验证 reserve-before-allocate、actual commit、GPU completion 后 release、
slot generation、upper violation 和 cancel cleanup。

### 141.4 L3：认证 campaign

每个 candidate 至少运行以下矩阵：

| 维度 | case |
|---|---|
| 预算 | `minimum-1`、`minimum`、`minimum+headroom` |
| 生命周期 | cold、warm、repeat、cancel、I/O fault、allocation fault |
| 设备 | 最低支持 GPU、目标 GPU、相邻 device family |
| 负载 | 最小 shape、最大已声明 shape、边界 steps/frames |
| 质量 | seed 固定输出、PSNR/SSIM/latent checksum 或模型专用合同 |
| 系统 | 安静环境、并发 1/2、swap 监测、page cache 预热/冷启动 |

L3 通过条件：`actual_peak <= planned_upper`、`swapouts_delta == 0`（安静环境）、无
resident fallback、无 token 泄漏、质量合同通过、性能达到 candidate SLO，并生成可重放
evidence bundle。

### 141.5 当前工作树实测记录

本轮已验证：

~~~text
native build                         PASS
contract suite                       81 tests PASS（1 个 Wan fixture SKIP）
memory accounting                    PASS
memory manifest/registry             PASS
memory plan compiler                 PASS
memory stage scheduler               PASS
H3 streaming policy                  PASS
LTX streaming benchmark              PASS
H3 Metal hook                        SKIP（当前无 Metal device）
LTX Metal hook                       SKIP（当前无 Metal device）
~~~

实际日志必须保存在 `build/evidence/local/`，并标注测试进程、git diff digest、设备是否
存在和 SKIP 原因；SKIP 不得计入 release evidence。

## 142. 发布、回滚与代码审查补充规则

### 142.1 Release whitelist

release 文件只允许列出 exact identity：checkpoint digest、adapter、dtype、shape bucket、
device family、runtime range、manifest/evidence digest、`release_enabled`。不允许写
“所有 H3”“所有 48 GiB 机器”这类模糊范围。新增 shape 或 runtime 必须新增 record 并重新
campaign。

### 142.2 线上越线处理

若 observed peak 越过 upper、出现 swap delta、GPU fault 或质量回归：

1. 立即 quarantine 对应 evidence/manifest digest；
2. 后续匹配请求回到 `plan_only` 或 `unsupported`；
3. 保留默认模式和旧实现，不删除代码；
4. 分析具体 site/backing/generation 后修正 manifest 或实现；
5. 重新执行 L3 campaign，通过 review 后再启用 whitelist。

不能通过“偷偷增加全局 margin”掩盖 site 漏记或 double-count 问题；margin 必须有来源、
版本和 evidence。

### 142.3 每个内存 patch 的审查问题

~~~text
[ ] disabled fast path 是否字节级保持原有选择和 cache 行为？
[ ] capability match 是否发生在第一笔大 allocation 前？
[ ] required resource 是否都有 executable upper 和稳定 site id？
[ ] reserve 是否早于 allocator，commit 是否记录 actual？
[ ] GPU completion 前是否禁止 release/reuse？
[ ] host vector/mmap/graph workspace 是否纳入 process-tree ledger？
[ ] alias/backing 是否避免重复计费？
[ ] cancel/fault/exception 是否 exactly-once cleanup？
[ ] plan 与 runtime 是否使用同一 normalized shape 和 digest？
[ ] unknown required 是否为 0，optional unknown 是否显式列出？
[ ] 测试是否包含 minimum-1、minimum+headroom、swap、质量和性能？
[ ] 文档是否准确标注 maturity，没有把 plan-only 写成 supported？
~~~

### 142.4 交付判断

在 H3/LTX 至少各有一条 `L3Certified` route 之前，本功能只能以实验性 add-on 形式
交付；Flux/Z-Image 的 plan-only 或 component-staged 能力不应改变 H3/LTX 的发布判断。
“内存受限模式已实现”必须解释为：

~~~text
实现了显式配置、规划、ledger、hook、fail-closed 和可认证基础设施；
只有命中 release whitelist 的具体 candidate 才允许运行。
~~~

这一定义既保证低内存机器不会掉入 swap，也保证内存充足机器在不开启 add-on 时维持
当前默认路径和性能。

该顺序与第 120 节的 M1–M10 对齐，但反映当前 worktree 中 M1、M2 的部分内容已经进入
代码、尚未完成新一轮完整验证的现实。

## 143. 从当前 scaffolding 到真实执行闭环

本节是对第 133–142 节的代码级收敛。前文描述了目标架构，本节明确“现在的代码能做
什么、下一步必须在哪个文件接线、什么顺序才不会在 capability 认证前产生大分配”。
实现者应把本节当作 code review 的顺序合同，而不是新增的用户配置。

### 143.1 当前调用链和安全边界

当前 `tc_engine_generate()` 的实际顺序是：

```text
request_from_json
  -> make_plan
  -> require_memory_constrained_execution_supported
  -> 使用 plan.request 的 normalized request
  -> uses_parent_mlx / GPU lock
  -> prepare_memory_admission
  -> set_memory_admission
  -> session.generate / session.prepare
```

当前 `make_plan()` 已经完成以下安全工作：

| 步骤 | 当前实现 | 不能误解为 |
|---|---|---|
| 配置校验 | `validate_memory_constrained_request()` 校验 Y/X、slot、GPU-only | 已经有 OS hard cap |
| candidate normalization | H3/LTX 受限候选转成 `gpu + streamed` | 该 route 已经认证 |
| estimate | `memory_estimate_bytes` 产生保守数值 | 可替代 allocation ledger |
| hard gate | `execution_supported` 初始为 false | 可以由 estimate 自动打开 |
| admission | `MemoryAdmission` 采用 clean process footprint 建立 ledger ceiling | 已覆盖 MLX/graph/所有 malloc |

由于生产 registry 还没有 exact record，当前任何 H3/LTX constrained execute 都应在
checkpoint、MLX array、Metal buffer 和 graph compile 之前返回
`memory_policy_unsupported`。这不是测试失败，而是 fail-closed 的预期行为。实现者不得
为了跑通一个示例而将 `execution_supported=true` 写回 `plan.cpp`，也不得在测试环境变量
中伪造 release record。

### 143.2 Capability probe 必须发生在 session 侧

`tc_plan_json` 没有模型路径和 checkpoint identity，因此纯 plan 阶段无法进行正式
capability 认证。模型路径只存在于 `ModelSession` 创建时的私有 `root_`，并且不能把
绝对路径写入结果 JSON。建议新增以下只读对象（字段名可按现有命名调整，但语义不能
改变）：

```cpp
struct DeviceIdentity {
    std::string family;          // AppleGPU-M4 / AppleGPU-M5 ...
    std::string metal_feature_set;
    std::string os_build;
    std::string mlx_revision;
    std::string runtime_revision;
};

struct MemoryCapabilityProbe {
    MemoryCandidateKey key;
    MemoryManifest manifest;
    std::string checkpoint_identity; // digest only, never an absolute path
};

class ModelSession {
  public:
    virtual std::optional<MemoryCapabilityProbe>
    probe_memory_capability(const ExecutionPlan &frozen_plan,
                            const DeviceIdentity &device) const {
        return std::nullopt; // old/unknown session remains plan-only
    }
};
```

推荐把 registry 作为 engine 启动时加载的只读对象：

```cpp
class MemoryCapabilityCatalog {
  public:
    MemoryCapabilityMatch lookup(const MemoryCandidateKey &key,
                                 std::string_view manifest_digest,
                                 bool allow_experimental) const;
};
```

`probe_memory_capability()` 的实现只允许读取以下内容：模型目录中的受信任
metadata/index、manifest sidecar、runtime revision、设备只读属性和已经冻结的 request
shape。它可以对 checkpoint 文件做流式 SHA-256；不能 materialize tensor、创建
MTLBuffer、调用 `mx::eval()`、编译 MPSGraph 或启动 worker。manifest sidecar 的路径可以
来自模型目录，但 sidecar 本身必须通过签名/白名单目录验证，不能由 request 中的任意路径
覆盖。

建议将 `ModelSession` 的默认实现设计成“无能力”而不是“resident 能力”。这样 Flux、
Z-Image、FastH3/VDN 或未完成 closure 的 H3/LTX 分支会自然保持 plan-only，不会因为旧
session 没有 override 而意外使用不受控的 resident 路径。

### 143.3 `c_api.mm` 的推荐改造顺序

当前 `tc_engine_generate()` 在 probe 之前直接调用 hard gate。迁移时应把“构造 candidate”
和“验证 exact capability”拆成两个函数，避免在 C API 中复制模型逻辑：

```cpp
PreparedMemoryExecution prepare_constrained_execution(
        ModelSession &session, ExecutionPlan plan,
        const DeviceIdentity &device, MemoryCapabilityCatalog &catalog) {
    auto probe = session.probe_memory_capability(plan, device);
    require(probe, "memory_policy_unsupported: capability probe unavailable");
    probe->manifest.validate();
    const auto manifest_digest = probe->manifest.digest();
    auto match = catalog.lookup(probe->key, manifest_digest,
                                /*allow_experimental=*/false);
    require(match.matched,
            "memory_policy_unsupported: exact capability record not matched");

    // clean baseline 和 full plan compile 在下一步完成；此处仍无大分配。
    return prepare_admitted_execution(session, std::move(plan),
                                      std::move(*probe), std::move(match));
}
```

推荐的最终顺序是：

```text
parse -> normalize -> metadata probe -> candidate key -> manifest digest
  -> registry lookup -> acquire GPU/execution lease
  -> clean/ledger-bound baseline observation -> exact plan compile
  -> MemoryAdmission -> bind H3/LTX hooks
  -> load through guarded adapter -> execute
  -> completion drain -> final observation -> result
```

必须满足：

1. registry miss、manifest mismatch、required unknown、`minimum-1` 或 system reserve 失败均在
   第一笔大分配之前返回；
2. capability match 之后才允许 `prepare_memory_admission()` 清理旧 session；失败的
   plan-only 请求不应主动 `unload()`，避免无谓地破坏默认缓存；
3. `prepare_memory_admission()` 成功后才可以让 session 安装 callbacks。callback 的生命期
   覆盖一次 admitted call，调用结束必由 RAII 清空；
4. 归一化后的 `plan.request`、`plan_digest` 和 `policy.digest` 作为唯一执行输入，session 内
   不得再次基于原始 request 选择 resident、backend、slot 或 tile；
5. 任何旧入口 `generate(Request, ...)` 在 constrained 模式收到未冻结 plan 时必须拒绝，
   而不是隐式重新 plan。

### 143.4 clean baseline 与 exact plan 的两道门

Capability identity 可在不清理 session 的前提下匹配，但动态 process baseline 只有在决定
执行后才能准确获得。建议将执行准入拆为两道门：

```text
Gate A（只读）：checkpoint/device/runtime/candidate/manifest/release record 精确匹配
Gate B（动态）：清理或登记 retained backing，采样 baseline/system available，编译 full plan
```

Gate A 不通过时不改变 session。Gate B 的具体计算为：

```text
P = clean 或 ledger-bound process baseline
F = framework/cold-compile validated upper
M = manifest resource live-set peak
planned_total = max_epoch(P + F(epoch) + M(epoch))
require planned_total <= B
require planned_increment <= system_available - system_reserve
```

`framework_upper` 不能继续来自自由浮动的 heuristic。第一版应把 framework/cache/compile
作为 manifest resource 实例；若暂时保留 `compile_memory_plan()` 的独立参数，该字段必须
带 `ValidatedEnvelope` provenance 和 exact runtime/device identity。

### 143.5 `prepare`、cache 和 retained session 的边界

当前 `prepare_memory_admission()` 会先 synchronize/unload/clear cache，再采样进程
footprint。这对第一版安全，但不是最终性能形态。后续实现应分三种明确状态：

| 状态 | constrained 行为 | 允许复用 |
|---|---|---|
| `clean` | 无已知模型 backing，直接 admission | 无 |
| `ledger_bound` | retained weights/condition/cache 均有 `StorageId` 和 scope | 仅 manifest 声明的 alias/cache |
| `tainted` | 曾有 upper violation、GPU fault 或未知外部 allocation | 不得复用，销毁/worker 退出 |

在 `ledger_bound` 完成前，constrained `prepare(warmup=1)` 不应写入默认 resident cache。
`tc_engine_cache()` 目前是独立 cache 管理入口；在 constrained 模式下若 cache action 没有
对应 scope/quota，应返回 `memory_policy_unsupported`，不能把 cache clear 当作通用内存
算法。disabled 请求继续沿用现有 cache 行为，不创建 `MemoryAdmission`，也不调用
manifest SHA。

### 143.6 `EffectiveMemoryPolicy` 的兼容重构

短期继续使用现有字段，避免破坏结果协议；中期把语义拆成四个内部对象：

```cpp
struct MemoryBudgetContract { /* Y/X/B、system reserve、slot 上限 */ };
struct MemoryCandidate { /* normalized request、route、plan digest */ };
struct MemoryCapability { /* level/state/digests/release record */ };
struct MemoryRuntimeState { /* actual peak、unknown、taint、revision */ };
```

`plan.cpp` 只能构造 contract/candidate，不能写 `execution_supported=true`；catalog/probe
负责 capability；admission/scheduler 负责 runtime state。这样可以避免把一个可解释的
estimate 当成执行授权。

## 144. H3 C/Metal 首发路线：把 schedule 也纳入 allocation closure

### 144.1 当前 H3 首发范围

首发只考虑以下精确组合：

```text
model=minimax-h3-turbo
operation=video.generate
execution=gpu
backend=C/Metal
audio=false, inputs=empty, loras=empty, quantized_cache=empty
residency=streamed, refill_slots=2
```

`h3_gpu_options` 与 `h3_host_memory_hooks` 已经能从 session 传播到 runtime，且旧 API 在
hooks 为 `NULL` 时保留 fast path。首发认证不能因此覆盖以下分支：audio、reference/image
condition、runtime LoRA、I8 quant cache、ANE/CoreML、preview、TAEH3、vision encoder。
这些分支要么单独建 candidate，要么写出可执行的不可达证明。

### 144.2 `h3_dit_schedule.c` 的可达调用闭包

源码已经证明 H3 T2V common path 会从 `h3_dit.c` 调用：

```text
h3_dit_schedule_precompute_active()
h3_dit_schedule_precompute()
```

因此不能只审计 `h3_dit.c` 的主 block loader。`h3_dit_schedule.c` 中以下 allocation 必须
加入 manifest/site registry：

| 代码对象/调用 | 归类 | 建议 lifetime | 处理方式 |
|---|---|---|---|
| `video_rows/audio_rows` | `conditioning` | request→schedule free | host reserve/commit |
| `visual_condition_rows/audio_condition_rows` | `conditioning` | request→schedule free | 仅对应条件存在时实例化 |
| `times/features` | `conversion_scratch` | schedule precompute | host reserve；GPU upload 另计 |
| `input/hidden/activated/output` | `conversion_scratch`/`activation` | one projection | classified GPU tensor |
| `bf16/silu` | `conditioning` | one projection→schedule use | classified GPU tensor |
| `schedule->blocks[]` | `conditioning` | denoise request | classified GPU tensor，按 active mask |
| `schedule->final` | `conditioning` | denoise request | classified GPU tensor |
| `values` 临时数组 | `conversion_scratch` | one conversion | host reserve 或证明小于 fixed floor |

`weight_f32_*`/`weight_bf16_*` 通过 `h3_weight_load_*`，其底层已经标记为
`H3_GPU_MEMORY_WEIGHTS`；不要在 schedule 层再次把同一 backing 加入 ledger。若同一权重
同时作为 MTLBuffer view 和 tensor wrapper 出现，必须使用相同 allocator domain/handle，
由 `StorageId` 去重。

建议先把 `time_embeddings()` 改成显式分类 API：

```c
h3_gpu_tensor *input = h3_gpu_tensor_from_f32_classified(
    gpu, features, (size_t)rows * TIME_INPUT,
    H3_GPU_MEMORY_CONVERSION_SCRATCH, "h3.schedule.time_input");

h3_gpu_tensor *hidden = h3_gpu_tensor_new_classified(
    gpu, (size_t)rows * TIME_HIDDEN, H3_GPU_F32,
    H3_GPU_MEMORY_ACTIVATION, "h3.schedule.time_hidden");
```

如果某个 schedule tensor 在多个 denoise step 之间存活，site 的 lifetime 必须是
`denoise[0]→denoise[last]`；不能因为每个 projection 已经单独 submit，就把 schedule
backing 错当成 command-buffer 临时对象。

### 144.3 `prepare_rows()` 的 host accounting

`prepare_rows()` 当前有多组 `calloc/malloc`，它们会在 GPU schedule 完成前一直存活，且
不能依赖 `h3_gpu_memory_hooks` 被观察。建议在 `h3.h` 的 `h3_host_memory_hooks` 之上增加
一个小型内部 helper：

```c
void *h3_host_alloc_classified(const h3_host_memory_hooks *hooks,
                               h3_host_memory_class memory_class,
                               size_t bytes, const char *site,
                               h3_host_allocation *out,
                               char *error, size_t error_size);
void h3_host_free_classified(h3_host_allocation *allocation);
```

helper 规则：

1. `bytes` 经过 checked multiply/alignment；
2. 先 `reserve(upper=bytes)`，再 `calloc/malloc`，成功后以指针值作为 host handle
   `commit(actual=bytes)`；
3. malloc 失败调用 `cancel`，不留下 reservation；
4. free 只在最后一次 host use 之后进行；
5. hooks 为 `NULL` 时走旧路径，且 helper 不增加任何默认模式计数。

首发可以把 schedule host arrays 合并为一个 validated envelope，前提是 manifest 明确
记录数组的 shape/steps 上界和一次性峰值；不能只写一个“schedule small”注释。

### 144.4 H3 Video VAE 上界收敛

当前 `decoded_host_envelope_v1` 仍包含启发式 `4x` 上界。应新增纯函数：

```cpp
struct H3VideoVaeShape {
    uint32_t frames;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t tile_width;
    uint32_t tile_height;
    uint32_t tile_batch;
};

uint64_t h3_video_vae_required_bytes(const H3VideoVaeShape &,
                                     std::string_view revision);
```

公式至少拆出：

```text
decoded_output
+ temporal_overlap (最多 5 帧窗口)
+ per_tile input/output
+ color conversion scratch
+ final frame aggregation
+ allocator alignment
```

`h3_video_vae_required_bytes()` 必须和执行时的 `tile_count/tile_batch/frame_elements`
使用同一套 normalized shape；执行中返回的实际 frame shape 用于 commit。若 actual 大于
upper，立即标记 request/worker `tainted` 并停止后续 export，不得把 reservation 扩大后
继续运行。

### 144.5 H3 pager 与 schedule 的边界

首发不重写既有双槽 SSD loader，只包一层 adapter：

```cpp
struct H3BlockSource {
    uint32_t block_id;
    uint64_t file_offset;
    uint64_t upper_bytes;
    h3_gpu_dtype dtype;
    const char *site_id;
};

struct H3SlotTicket {
    uint64_t generation;
    unsigned slot;
    MemoryStageLease stage;
    std::shared_ptr<Completion> completion;
};
```

流程固定为：

```text
next block source
  -> ledger reserve(refill_slot)
  -> SSD pread/convert into reserved slot
  -> commit actual backing
  -> encode GPU commands
  -> mark command completion
  -> retire slot
  -> completion callback -> ledger complete_pending
```

H3 现有 `ssd_pinned_prefix` 仍可由 planner 计算；但 constrained 模式必须把 schedule
conditioning、activation、host staging、output 和 process baseline 从同一个 full-request
plan 中扣除。不能只把 `ssd_memory_budget_bytes` 写成 B 就宣称全请求受限。

## 145. LTX C/Metal 首发路线：消灭 direct allocation 和 host vector 黑洞

### 145.1 Direct Metal allocation 收口

当前 `ltx_gpu_buffer_new()` 已有集中 hook，但 `ltx_video_vae.m`、`ltx_upsampler.m` 等
仍有 direct `newBufferWithLength:` 路径。受限首发必须达到以下条件之一：

```text
所有 constrained-reachable direct call -> ltx_gpu_buffer_new_classified()
或
该分支有编译/运行时不可达证明，并在 reachable-sites.json 登记
```

建议新增：

```c
ltx_gpu_buffer *ltx_gpu_buffer_new_classified(
    ltx_gpu *, size_t bytes, ltx_gpu_memory_class, const char *site);
ltx_gpu_buffer *ltx_gpu_buffer_new_copy_classified(
    ltx_gpu *, const void *, size_t bytes,
    ltx_gpu_memory_class, const char *site);
```

`ltx_video_vae.m` 和 `ltx_upsampler.m` 不应自己创建另一套 reservation；所有 reservation
必须经由 `ltx_gpu` context 传递同一 `allocator_domain/generation`，否则无法检测跨模块
alias。对于无法复用 `ltx_gpu` 的 helper，给 create options 加同样的 hooks/domain/generation，
而不是引用一个进程全局裸指针。

### 145.2 LTX host allocation registry

首发 video-only 路线必须审计并分类：

| 文件/对象 | 资源 | 归类 |
|---|---|---|
| `ltx_blocks.c` 4970 附近 | stage1/2 video/audio、patch、mask、reference | activation/conditioning |
| `ltx_blocks.c` 5355 附近 | noise/final host buffers | conversion_scratch/output |
| `ltx_conditioning.c` | embedding、parameter arrays | conditioning |
| `ltx_gemma_encoder.m` | tokenizer output、states、rope | conditioning/conversion_scratch |
| `ltx_gemma_tokenizer.m` | ids/mask | conditioning |
| `ltx_connector.c` | combined/positions/sin/cos | conditioning |
| `ltx_session.mm` | stage vectors、upsampled latent、pixels、RGB | activation/output |

推荐新增 `tc::LedgerAllocator<T>` 或 C ABI 等价物，但实现时必须注意 STL 异常安全：

```cpp
template <class T>
class LedgerAllocator {
  public:
    T *allocate(std::size_t n); // reserve -> malloc -> commit
    void deallocate(T *p, std::size_t n) noexcept;
};
```

对 `std::vector` 扩容要遵守“先申请新 backing，成功后 swap”：

```text
old vector 保持不动
  -> reserve(new_capacity * sizeof(T))
  -> allocate/commit new backing
  -> move/copy
  -> GPU/CPU last-use fence
  -> release old backing
```

不能先让 STL 自己扩容，再从 ledger 补记；也不能在扩容失败后把旧 vector 和新半成品
同时视为可回收。`capacity()` 不是 actual bytes 的唯一来源，必须使用 checked multiply 与
allocator 对齐后的 capacity。

### 145.3 Stage1→upsampler→Stage2 的严格 lease

LTX video-only 首发建议先使用 strict handoff，牺牲少量 overlap 换取容易认证的峰值：

```text
text/gemma encode
  -> stage1 conditioning + latent lease
  -> stage1 denoise
  -> stage1 GPU completion
  -> allocate stage2 latent
  -> upsample
  -> release stage1 latent/clean prefix
  -> stage2 denoise
  -> release denoiser/conditioning/audio latent
  -> Video VAE graph + decode
  -> release latent/decoder graph
  -> RGB conversion
  -> export
```

当前 `ltx_session.mm` 已在 stage transition 处释放部分 vector 和 decoder cache；正式
manifest 还必须记录 stage1 latent、stage2 latent、upsampler scratch 的 overlap interval。
若将 stage1 backing alias 给 stage2，必须同时满足：

1. 两个 stage 使用相同 layout/dtype；
2. 同一 command queue 上的 completion fence 已 signal；
3. manifest 明确写出 alias group；
4. `MemoryPlan` 的 alias folding 和 runtime `StorageId` 都使用同一 identity。

否则默认分配新 backing，并把复制成本纳入 upper。

### 145.4 MLX VAE/upsampler 和 MPSGraph

LTX C/Metal allocator hook 不会自动观察 MLX array、MPSGraph temporary 或 MLX allocator
cache。首发 constrained candidate 要求：

- C/Metal denoiser 可执行，但 MLX VAE/upsampler 只能在有独立、已验证 envelope 时执行；
- `cold_compile_upper` 与 `steady_dispatch_upper` 分开记录；
- 只验证 warm path 时，cold request 必须 plan-only 或由 disposable worker 预编译后重新
  建立 clean baseline；
- `mx::clear_cache()` 只能作为 stage boundary 的已记录动作，不能替代 reservation；
- 任何 MLX route 的 required unknown 不为 0 时，不得升级 `EnvelopeValidated`。

如果使用 `exec` finalizer，parent 在 spawn/exec 期间的生命周期必须按第 29、45 节计账。
constrained 首发默认 `worker_scope=same_process`；只有 child envelope、ACK、退出回收和
process-tree evidence 完成后，才允许启用 disposable worker。

### 145.5 现有 2 GiB graph envelope 的处置

`ltx_session.mm` 当前使用 `kLtxVideoVaeGraphEnvelopeBytes = 2 GiB` 包住 Video VAE graph。
该值可以继续作为 `HookBridged`/开发诊断的保守 reservation，但在 release manifest 中必须：

1. 绑定 LTX VAE checkpoint、MLX/MPSGraph revision、shape/tile 和 device family；
2. 分开记录 cold create、first decode、steady decode、cache clear 后 residual；
3. 证明实际峰值不超过 envelope；
4. 若一次 decode 触发更大临时对象，能够在 allocation 前拒绝而不是只在阶段末观察；
5. 未达到上述条件时维持 `HeuristicNotExecutable`，不能让 2 GiB 常量获得放行权。

## 146. Flux 与 Z-Image：先分层 staged，再逐模型开放 block pager

### 146.1 Flux 的两阶段路线

Flux 4B/9B 在 transformer、text encoder、VAE 和 compile workspace 的 combined upper
尚未有完整 manifest，因此按以下阶段推进：

```text
F0 metadata-only plan
  -> F1 component-staged（transformer 完整驻留）
  -> F1 L2/L3
  -> F2 layer-group pager
  -> F2 explicit-weight graph/eager fallback
  -> VAE tiling、LoRA、reference input 独立认证
```

对应文件施工建议：

| 文件 | 改造 |
|---|---|
| `native/models/flux2/flux_transformer.cpp` | 将 block group 权重入口改成 pager lease；graph key 绑定 shape/layout/checkpoint |
| `native/models/flux2/flux_text.cpp` | text context、rope、padding 计入 conditioning manifest |
| `native/models/flux2/flux_encode.cpp` | image/reference encode 作为独立 stage，不假设与 transformer 不重叠 |
| `native/models/flux2/flux_vae.cpp` | decode workspace/output tiling envelope |
| `native/models/flux_module.cpp` | candidate identity、明确拒绝矩阵，不把 descriptor executable 当作 constrained certified |
| `native/platform/apple/results.mm` | 输出 F0/F1/F2 capability/certification 字段 |

F1 的最低要求是 transformer full-resident floor 在 B 内；如果 B 低于该 floor，必须返回
`memory_budget_too_small` 或 `plan_only`，不能回到旧 resident/unbounded 执行。F2 的每个
layer group 需要单独记录 attention scratch、RoPE cache、MLP conversion 和 compiled
workspace；不能只按参数 bytes 估算。

### 146.2 Z-Image 的 variant 隔离

`z-image-turbo`、`z-image-turbo-gguf`、ConvRot packed Q8、NVFP4 和混合 K-quant 不是一个
candidate。对应改造点：

| 文件 | 改造 |
|---|---|
| `native/models/z_image/z_image.cpp` | text/transformer/VAE 生命周期拆分；BF16 staged candidate |
| `native/models/z_image/gguf.cpp` | GGUF offset reader、dequant scratch、block cache identity |
| `native/models/z_image_module.cpp` | variant/candidate key 与拒绝原因 |
| `native/models/z_image_gguf_module.cpp` | quant scheme/group size/layout 绑定 manifest |
| `native/platform/apple/results.mm` | 输出 variant-specific memory evidence |

在 metadata 阶段冻结 `checkpoint_format + quant_scheme + group_size + packed_layout + VAE
revision + LoRA mode`。若 GGUF mmap 的 page-cache 峰值无法被证据包中的 process footprint
解释，使用 explicit read/stream buffer；不得把 mmap 的“懒加载”当作自动 streaming 证明。
NVFP4、mixed K-quant、未验证 QKV remap 和 request-time LoRA 暂时保持 plan-only。

### 146.3 staged 和 streamed 的能力边界

四类模型不能共用一个布尔 `supports_streaming`。建议 capability 明确声明：

```cpp
enum class MemoryExecutionKind {
    ResidentGuarded,
    ComponentStaged,
    BlockStreamed,
    TiledDecode,
};
```

H3/LTX 可以先认证 `BlockStreamed` denoiser；Flux/Z-Image 可以先认证
`ComponentStaged`。如果 component-staged 的单个最大组件仍不适合 B，应明确拒绝并等待
block pager，而不是把“阶段之间释放”宣传成任意低内存支持。

## 147. StageScheduler/WeightPager 的最终事务模型

### 147.1 当前 `MemoryStageScheduler` 的定位

当前实现已经提供：

- `begin_stage()` 返回 RAII `MemoryStageLease`；
- `allocate()` 执行 ledger reserve→commit；
- `retire_all()` 产生 pending release token；
- `complete()` 显式确认 pending GPU completion；
- `observe()` 提供 normal/tight/critical 三态；
- critical sticky，tight 恢复需要连续三个低于 80% 的样本。

需要特别指出：当前 `MemoryStageLease::allocate()` 接收的是已经存在的 `StorageId`。因此
它适合 synthetic test 或“allocator 已被另外一层 guard”的 adopt 场景，但不能单独证明
生产代码中的 reserve 发生在真实 `MTLDevice`/`malloc` 之前。生产 adapter 必须使用下面的
两步接口，禁止先分配再补账：

```cpp
auto reservation = admission.try_reserve(
    MemoryClass::RefillSlot, upper_bytes, "h3.dit.refill_slot.1");
require(reservation, "memory_budget_too_small: refill slot");

void *backing = allocator_allocate(upper_bytes); // MTLBuffer/malloc/pread target
if (!backing) {
    reservation->cancel();
    throw allocation_error();
}

auto lease = reservation->commit(StorageId{
    allocator_domain, backing_handle, actual_bytes, generation});
```

下一步不应删除 `MemoryStageScheduler`，而应加一个 adapter 层，显式携带 site、generation
和 command-buffer completion：

```cpp
struct AllocationTicket {
    uint64_t token_id;
    uint64_t generation;
    uint64_t backing_id;
    std::string site_id;
    MemoryClass memory_class;
    uint64_t upper_bytes;
    uint64_t actual_bytes;
    PendingReleaseId pending;
    std::shared_ptr<Completion> completion;
};

class WeightPager {
  public:
    std::optional<AllocationTicket> reserve_slot(
        const BlockSource &, MemoryStageScheduler &, bool optional_prefetch);
    void submit_gpu(AllocationTicket &, MTLCommandBuffer *);
    void on_completion(AllocationTicket &);
    void cancel(AllocationTicket &) noexcept;
};
```

### 147.2 Slot 状态和 generation

每个 slot 必须经历如下状态，状态迁移由单一 owner 执行：

```text
FREE
  -> RESERVED       ledger token 已取得
  -> LOADING        host I/O/解压进行中
  -> COMMITTED      actual backing 已提交
  -> IN_FLIGHT      GPU command buffer 引用
  -> RETIRING       CPU 已请求回收，等待 completion
  -> FREE

RESERVED/LOADING -> CANCELLED -> FREE
COMMITTED/IN_FLIGHT -> TAINTED -> destroy/quarantine
```

slot reuse 必须满足 `new_generation > old_generation`，且旧 generation 的 completion 已
signal。物理地址复用不等于 ledger identity 可复用；`StorageId` 至少包含
`allocator_domain + handle + capacity + generation`。

### 147.3 三队列和 bounded overlap

每个 stage 使用三条逻辑队列：

```text
I/O queue:       读取、解压、转换到已预留 host/Metal slot
encode queue:    将已 commit slot 绑定到 GPU command buffer
completion queue:等待 command completion，完成 pending release
```

推荐 look-ahead 算法：

```cpp
while (!stage.done()) {
    if (!pressure.stop_optional_prefetch()) {
        auto next = stage.peek_next_block();
        if (next && pager.can_prefetch(*next) &&
            ledger.available_for(next->upper_bytes + guard_bytes)) {
            auto ticket = pager.reserve_slot(*next, scheduler, true);
            if (ticket) io.submit(*ticket, *next);
        }
    }

    if (auto ready = pager.ready_for_encode()) {
        pager.submit_gpu(*ready, encode.current_command_buffer());
    }

    completion.poll([&](AllocationTicket &ticket) {
        pager.on_completion(ticket);
        scheduler.complete(ticket.pending);
    });
}
```

I/O queue 没有 permit 时必须阻塞或取消，不得先把数据读进一个未计账的临时 malloc。
没有 free slot 时只能等待 bounded timeout 或退化到已认证的 serial revision；不得临时
malloc “第四个槽”。

### 147.4 pressure 阈值的版本化

当前 scheduler 代码使用 `>85%` 进入 tight、低于 `80%` 连续三个样本恢复；allocation
fail、observed over budget 或 system reserve 不足进入 sticky critical。前文的 0.90/0.97
水位是目标调度策略，不应继续以散落常量存在。最终实现应把阈值放入
`MemoryPolicyRevision`：

```cpp
struct PressureThresholds {
    unsigned tight_percent = 85;
    unsigned recover_percent = 80;
    unsigned recover_samples = 3;
    unsigned critical_percent = 100;
};
```

manifest/candidate 绑定 threshold revision 后，campaign 才可比较性能。pressure controller
可以减少 look-ahead、释放 optional cache、等待 completion，但不能在线改变 dtype、模型
variant、全局 attention 语义或未经认证的 tile。进入 critical 后 request tainted，不能
靠后续回收把结果改写成 PASS。

### 147.5 锁顺序和回调禁止事项

固定锁顺序：

```text
service/job mutex -> execution/GPU mutex -> ledger mutex
  -> pager slot mutex -> I/O condition variable
```

allocator callback 内禁止：

- `mx::synchronize()`；
- 等待 GPU completion；
- 文件 I/O；
- Objective-C autorelease pool drain；
- 触发新的 model load 或 cache trim。

callback 只做 checked state transition，外层 scheduler 再等待。否则容易形成
`ledger mutex -> GPU wait -> callback -> ledger mutex` 的死锁。

## 148. 错误协议、结果字段和可观测性

### 148.1 错误码优先级

受限模式必须让用户知道是“预算不足”“能力未知”还是“运行时越线”：

| 错误码 | 触发 | 是否允许重试 |
|---|---|---|
| `memory_policy_invalid` | Y/X/slot/schema 非法 | 修改 request 后重试 |
| `memory_policy_conflict` | 与旧 hint、ANE、hybrid 冲突 | 修改 route 后重试 |
| `memory_policy_unsupported` | route/manifest/capability 未认证 | 只能换已认证 candidate；不得自动 fallback |
| `memory_estimate_unknown` | required upper 缺失 | 只能完成 manifest/代码 closure |
| `memory_budget_too_small` | minimum-1、D<=0、plan peak>B | 提高 Y、降低 shape 或使用已认证 tiling |
| `memory_admission_conflict` | system reserve/进程基线无法满足 | 等待外部进程结束或提高余量 |
| `memory_pressure_abort` | observed footprint 越线或 critical | 需清理 tainted worker 后重试 |
| `memory_lifetime_violation` | stale token、double release、alias mismatch | 视为实现 bug，隔离 candidate |

错误消息应包含 `adapter_candidate`、`plan_digest`、`manifest_digest`（如有）、预算 B、
observed/upper bytes 和具体 site；不能打印 prompt、绝对 checkpoint 路径或用户目录。

### 148.2 结果 JSON 的最小闭环

受限成功结果至少包含：

```json
{
  "memory_policy": {
    "enabled": true,
    "user_limit_bytes": 17179869184,
    "effective_budget_bytes": 14602888806,
    "planned_upper_bytes": 13958643712,
    "actual_peak_bytes": 13690208256,
    "required_unknown_bytes": 0,
    "adapter_candidate": "h3_c_metal_streamed_v1",
    "manifest_digest": "sha256...",
    "evidence_digest": "sha256...",
    "capability_level": "l3_certified",
    "certification_state": "certified",
    "release_stable": true,
    "refill_slots": 2,
    "schedule_revision": 3,
    "pressure_transitions": 1,
    "resident_fallback": false,
    "swapouts_delta": 0,
    "ledger_peak_unknown_bytes": 0
  }
}
```

当前 `RunResult` 尚未包含全部字段，迁移可以先在 `memory_admission` 和
`block_residency` 下增加可选字段，保持旧客户端反序列化兼容。字段不存在表示旧/disabled
结果，不得解释为 constrained success。

### 148.3 Trace 事件

`schedule_trace` 建议采用预分配 ring buffer，事件至少包括：

```json
{
  "ts_ns": 123,
  "stage": "denoise",
  "block": 17,
  "slot": 1,
  "event": "reserve|load_begin|load_end|gpu_begin|gpu_end|retire|release|deny",
  "site_id": "h3.dit.refill_slot.1",
  "generation": 42,
  "upper_bytes": 268435456,
  "actual_bytes": 267386880,
  "ledger_committed": 9876543210,
  "ledger_reserved": 268435456,
  "process_footprint": 11223344556,
  "pressure": "normal",
  "plan_digest": "..."
}
```

trace 满时可以丢弃普通 timing 事件，但以下事件必须保留：`reserve denied`、
`observed over budget`、`upper violation`、`cancel`、`fault cleanup`、`lifetime violation`。

## 149. 实现与验收：从单元到本机 L3

### 149.1 当前工作树可重复命令

每个后续 patch 至少运行：

```sh
env MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" \
  TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
python3 -B tests/native/test_contract.py
python3 -B tests/native/test_memory_accounting.py
python3 -B tests/native/test_memory_manifest.py
python3 -B tests/native/test_memory_plan_compiler.py
python3 -B tests/native/test_memory_scheduler.py
git diff --check
```

当前已验证结果：native build PASS；contract 81 项通过、1 项 Wan fixture SKIP；accounting、
manifest、plan compiler、scheduler 均 PASS。Metal hook 测试在无 Metal device 时只能明确
SKIP，不能写入 release evidence。

### 149.2 新增的纯逻辑测试

建议继续增加以下测试（不依赖权重和 Metal）：

| 测试 | 必须证明 |
|---|---|
| `test_memory_capability_probe.py` | registry miss、manifest digest mismatch、exact match |
| `test_memory_plan_identity.py` | shape/backend/slot/tiling/runtime 任一改变都会改变 digest |
| `test_memory_stage_fault.py` | reserve fail、commit fail、cancel、double release 无泄漏 |
| `test_memory_scheduler_revision.py` | threshold revision 冻结、critical sticky、serial fallback |
| `test_memory_default_abba.py` | disabled A/B/B/A 与基线 route/cache/结果一致 |
| `test_memory_reachable_sites.py` | H3/LTX constrained-reachable unknown site 数为 0 |

必须包含 property/fuzz 样例：随机 lifetime 不应触发整数溢出；输入顺序改变不应改变
manifest/plan digest；alias group 不能让 non-aliasable site 被折叠；minimum-1 只能拒绝而
不能丢失完整 plan。

### 149.3 L2 Metal 设备测试

有真实 Metal device 时，对 H3/LTX 各执行：

```text
reserve-before-allocate
actual <= upper commit
CPU release before GPU completion
pending release after completion
slot generation reuse
allocation fault / I/O fault / cancel
observed upper violation -> taint
```

测试需要检查底层 `MTLBuffer` 的 allocated length、ledger `StorageId` 和 completion handler
一致。仅看 `MemoryLedger` 数字而不验证真实 allocator callback 不足以通过 L2。

### 149.4 L3 campaign 的最小矩阵

每个 release candidate 至少包含：

| 维度 | 必测值 |
|---|---|
| budget | `minimum-1`、`minimum`、`minimum+headroom` |
| lifecycle | cold、warm、repeat、cancel、I/O fault、allocation fault |
| shape | 最小、最大声明、steps/frames 边界 |
| scheduler | serial、认证 slot 数、压力触发后的 bounded drain |
| system | 安静环境、已有 page cache、swap 计数 delta |
| quality | 固定 seed、latent/output checksum、媒体尺寸/fps/audio contract |
| performance | end-to-end、denoise、I/O wait、overlap hidden fraction、cache cold/warm |

通过条件：`actual_peak <= planned_upper`、required unknown 为 0、安静环境
`swapouts_delta==0`、无 resident/unbounded fallback、cancel/fault 无 token 泄漏、质量合同
通过、性能达到 candidate SLO，并生成可重放 evidence bundle。任何一次 observed 越线都
记为 FAIL；不能通过删除污染样本或事后扩大 margin 抹掉。

### 149.5 默认路径 ABBA

每个模型至少做：

```text
A: disabled request，冷启动
B: disabled request，热启动
B': disabled request，热启动（代码已接入 constrained scaffolding）
A': disabled request，冷启动（代码已接入 constrained scaffolding）
```

比较 route、residency、cache hit、输出 digest、stage timing 和 native allocation counter。
目标是稳态 wall 回归不超过 2%，且 disabled 路径没有 manifest probe、SHA、ledger callback、
cache clear、worker 拓扑或 model unload 新动作。若为了兼容编译必须引入符号，运行时计数
仍应为零。

### 149.6 本机低预算测试不能替代低内存实机

在 64 GiB 机器上把 Y 设置为 8/12/16 GiB，只能证明 planner、ledger 和 fail-closed 合同，
不能证明 8/12/16 GiB 机器上的系统行为。发布前仍需在目标低内存设备重复 L3 campaign，
至少记录：

```text
hw.memsize / device family / OS build
进程初始与峰值 physical footprint
vm_stat swapins/swapouts 的增量
memory pressure 事件
文件 page cache 冷/热状态
后台进程污染标记
```

不执行系统级 `purge` 或创建不可控压力来“制造冷环境”；应使用独立测试机、重启后的规定
窗口或可重复的文件副本策略，并在 evidence 中登记。

## 150. 逐 PR 实施顺序和最终 DoD

### 150.1 推荐 PR 顺序

```text
PR-0  disabled ABBA 基线与证据目录
PR-1  capability probe/catalog 接口；保持所有 constrained plan-only
PR-2  H3 schedule + host allocation closure
PR-3  H3 VAE exact upper + reachable-site audit
PR-4  LTX direct Metal/vector/MPSGraph closure
PR-5  scheduler ticket/completion/pager adapter
PR-6  H3 C/Metal allocation-guarded campaign
PR-7  LTX C/Metal allocation-guarded campaign
PR-8  H3/LTX manifest promotion 到 L3 whitelist
PR-9  Flux F1 staged（独立证据）
PR-10 Z-Image BF16/GGUF staged（独立证据）
PR-11 layer-group pager、tiling、LoRA/input variants
```

每个 PR 必须单独说明：改动的 allocation site、upper provenance、alias/generation 规则、
取消/异常清理、disabled fast path 影响、测试命令和未覆盖分支。不要把 H3/LTX/Flux/Z-Image
一次性合成一个“统一 pager”大 PR；不同 backend 的 storage、graph 和质量合同不同。

### 150.2 Release promotion 条件

从 `plan_only` 升级到 `experimental_guarded`：

```text
required site closure = 0 unknown
allocation guard 已覆盖 constrained-reachable path
minimum-1 / minimum / cancel / fault 单测通过
```

从 `experimental_guarded` 升级到 `envelope_validated`：

```text
真实 device 上 actual<=upper
cold/steady envelope 分离
GPU completion/pending release 无泄漏
```

从 `envelope_validated` 升级到 `l3_certified`：

```text
完整 L3 campaign、swap/quality/performance evidence
exact checkpoint/device/runtime/shape/slot/tiling identity
review 后 release whitelist 启用
```

任何 promotion 都由 code review 修改 registry/whitelist 完成；本机 campaign 不得自动把
自己写成 release record。发现越线时只 quarantine record 并回到 plan-only，不删除默认实现。

### 150.3 最终 Definition of Done

本设计真正完成，不是“JSON 接受了 `memory_constrained`”或“某个 block 可以 streaming”，
而是同时满足：

```text
配置：Y/X/B 语义清晰，enabled=false 完全走旧路径
规划：exact checkpoint manifest、live interval、alias folding、plan digest
准入：capability 在大分配前匹配；D<=0/minimum-1 早拒绝
执行：GPU/host/graph/output/worker allocation 全部 reserve->commit->completion release
调度：slot 有界、generation 正确、压力只选择已认证 revision
模型：H3/LTX 各至少一条真实 GPU L3 candidate；Flux/Z-Image 未认证则拒绝
可靠性：cancel/fault/upper violation taint 且无 token/cache 泄漏
性能：disabled ABBA <=2% 回归；constrained overlap 与 candidate SLO 达标
证据：swap delta、质量、峰值、trace、manifest/evidence digest 可重放
运维：release record 可回滚，失败 candidate 不影响默认路径
```

在 H3/LTX 两条真实 GPU 路线达到 `l3_certified` 之前，文档顶部必须继续保持
`Proposal + scaffolding`。本节的代码片段是实现建议和接口合同，不能被解读为当前已经
存在的生产 capability。

## 151. 2026-09-15 最新源码复核：先恢复可编译基线

本节是对第 133、144、149 节的最新事实校正。它只记录本轮实际执行的构建结果和紧接着
应完成的 patch，不把建议状态误写成已完成状态。

### 151.1 历史构建失败来自接口迁移未闭合；当前工作树已修复

执行：

```sh
env MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" \
  TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
```

该次复核的历史结果为 FAIL。`native/models/h3_runtime/h3_dit.h` 已给以下三个 API 增加
`const h3_host_memory_options *host_memory`，但当时调用方仍按旧签名传参：

```text
native/models/h3_runtime/h3.c:h3_parallel_prepare_main
    h3_dit_load_t2va_core()  少 1 个参数

native/models/h3_runtime/h3.c:h3_generate conditioned branch
    h3_dit_load_conditioned() 少 1 个参数

native/models/h3_runtime/h3.c:h3_generate text-only branch
    h3_dit_load_t2va() 少 1 个参数
```

此外，当时 `tools/native/h3_dit_streaming_probe.c` 仍按旧签名调用 `h3_dit_load_t2va()`；主库修复
后该工具也必须传入显式 `NULL`，否则后续 probe 构建仍会失败。当前工作树已经完成这些调用点
迁移，并恢复 native build PASS。保留本段是为了说明失败根因和回归要求，不应再解读为当前构建
仍失败；任何 capability promotion 仍需引用当前 HEAD 的独立 build/test evidence。

### 151.2 H3 host options 的唯一构造规则

不要在三个调用点分别手写结构体字段。建议在 `h3.c` 增加一个无分配 helper：

```c
static h3_host_memory_options h3_host_options_from_params(
        const h3_params *params) {
    h3_host_memory_options result = {0};
    if (!params || !params->host_memory_hooks) return result;
    result.struct_size = sizeof(result);
    result.version = 1u;
    result.hooks = params->host_memory_hooks;
    result.allocator_domain = params->memory_allocator_domain;
    result.generation = params->memory_generation;
    return result;
}

static const h3_host_memory_options *h3_optional_host_options(
        const h3_host_memory_options *options) {
    return options && options->hooks ? options : NULL;
}
```

`h3_generate()` 在参数校验通过后只构造一次：

```c
h3_host_memory_options host_memory = h3_host_options_from_params(params);
```

随后将 `h3_optional_host_options(&host_memory)` 传给 conditioned/text-only 两个 load 分支。
这样可以保证：

- disabled 路径仍传 `NULL`，不会让 schedule 误以为 constrained 已启用；
- domain、generation 与 `h3_gpu_options` 使用同一请求快照；
- 后续版本字段升级只改一个 helper；
- 不使用指向临时 compound literal 的指针跨异步边界。

### 151.3 `h3_parallel_prepare` 必须按值持有 options

`h3_parallel_prepare_main()` 在 worker thread 上执行，不能保存指向 `h3_generate()` 栈上临时
options 的裸指针。建议给 `h3_parallel_prepare` 增加：

```c
h3_host_memory_options host_memory;
```

在 `h3_parallel_prepare_start()` 中按值复制由 `params` 构造的 options，再在 worker 中传：

```c
prepare->host_memory = h3_host_options_from_params(params);

prepare->dit = h3_dit_load_t2va_core(
    ...,
    prepare->gpu_options,
    h3_optional_host_options(&prepare->host_memory),
    ...);
```

当前 constrained route 因 `parallel_prepare_eligible` 明确要求 `!params->gpu_options`，实际上不会
进入这个并行分支；但 API 仍需完成一致迁移，避免普通路径构建失败。不要为了通过编译删除
参数或放宽 constrained 并行资格。若未来允许 constrained parallel prepare，必须先把 text
placeholder、layout、schedule、worker stack 和同时加载峰值加入 manifest，再独立认证新的
candidate id。

### 151.4 工具和测试调用点迁移

`tools/native/h3_dit_streaming_probe.c` 属于 disabled/传统 streaming 工具，应在
`gpu_options` 后传 `NULL`：

```c
h3_dit_load_t2va(path, shader, NULL, NULL, &text, ...);
```

全仓库迁移完成条件：

```sh
grep -R -n "h3_dit_load_t2va\|h3_dit_load_t2va_core\|h3_dit_load_conditioned" \
  native tools tests
```

人工核对每个结果均满足新签名，不能只依赖编译到的 target，因为有些 probe/benchmark 不在
默认 native target 内。

### 151.5 恢复 build green 的最小验收顺序

修复只应做签名传播，不应顺手改变调度或缓存行为。推荐顺序：

1. 修改 `h3.c` helper、parallel struct 和三个调用点；
2. 修改 `h3_dit_streaming_probe.c` 的显式 `NULL`；
3. native build 必须 PASS；
4. 运行 contract、H3 streaming policy 和 H3 GPU hook 测试；
5. 执行 `git diff --check`；
6. 只有上述步骤通过，才继续 gate-score host closure。

该 patch 的 disabled AB 验收要求：`params->host_memory_hooks == NULL` 时所有 H3 load API 收到
`host_memory == NULL`，无额外 reserve/commit/release 回调，无 cache key 变化，无额外线程，
输出和 route 与修改前一致。

## 152. 请求级 `MemoryExecutionContext`：统一对象所有权和 API 生命周期

当前代码已经有 `MemoryAdmission`、`MemoryStageScheduler`、`CompiledMemoryPlan` 和 capability
resolution，但它们仍散落在 API helper、session callback 和模型私有 bridge 中。继续增加
零散裸指针会使 cancel、exception、GPU completion 和 session reuse 很难证明正确。建议新增
请求级 RAII 对象，把所有 constrained-only 状态绑定到一个 generation。

### 152.1 推荐对象图

```text
tc_engine
  └── ModelSession                         长生命周期，默认路径可复用
       └── [non-owning binding while call is active]
             MemoryExecutionContext        单请求、不可复制
               ├── EffectiveMemoryPolicy   冻结值
               ├── MemoryCapabilityProbe   metadata-only 结果
               ├── CompiledMemoryPlan      确定性 live interval
               ├── MemoryAdmission         root budget + footprint sampler
               │    └── MemoryLedger       unique backing/tokens
               ├── MemoryStageScheduler    stage/revision/pending completion
               ├── ScheduleTrace           预分配 ring
               ├── cancellation flag       借用 API 的 atomic
               └── generation/domain       所有 backend hook 共用
```

`ModelSession` 不拥有该 context；session 仅在一次 `generate()`/`prepare()` 调用期间借用。context
析构前必须先解除 session binding，并 drain/判定所有 completion。默认 session cache 仍由原路径
拥有，不被 constrained context 隐式接管。

### 152.2 建议新增接口

建议在 `native/runtime/memory_execution.hpp` 中增加：

```cpp
struct MemoryExecutionIdentity {
    uint64_t allocator_domain = 0;
    uint64_t generation = 0;
    std::string candidate_digest;
    std::string plan_digest;
};

class MemoryExecutionContext {
  public:
    MemoryExecutionContext(EffectiveMemoryPolicy,
                           MemoryCapabilityResolution,
                           CompiledMemoryPlan,
                           ProcessMemoryObservation,
                           uint64_t allocator_domain,
                           uint64_t generation);

    MemoryExecutionContext(const MemoryExecutionContext &) = delete;
    MemoryExecutionContext &operator=(const MemoryExecutionContext &) = delete;

    MemoryAdmission &admission();
    MemoryStageScheduler &scheduler();
    const CompiledMemoryPlan &plan() const;
    const MemoryExecutionIdentity &identity() const;
    void checkpoint(std::string_view phase);
    void mark_tainted(std::string reason) noexcept;
    void finish_success();
    void finish_failure() noexcept;
};
```

`finish_success()` 应断言：

```text
reservation_count == 0
pending_release_count == 0
required unknown bytes == 0
observed_within_budget == true
context 未 tainted
```

`finish_failure()` 不得伪造成功；它执行 best-effort cancellation/drain，记录仍未释放的 token，
并通知 session 丢弃 constrained generation 创建的 cache。

### 152.3 `ModelSession` 接口演进

当前只有 `set_memory_admission(MemoryAdmission *)`，只能让 backend 访问 ledger，不能访问 plan、
scheduler、trace 和 generation。建议兼容演进：

```cpp
class ModelSession {
  public:
    virtual void set_memory_execution_context(MemoryExecutionContext *ctx) {
        set_memory_admission(ctx ? &ctx->admission() : nullptr);
    }

    virtual void set_memory_admission(MemoryAdmission *) {}
};
```

首轮保留旧 virtual，H3/LTX 逐步改写为新接口；所有模型迁移后再考虑删除旧接口。binding 必须
满足：

- 同一 session 同时最多一个 context；
- 重复绑定非空 context 返回 `memory_lifetime_violation`；
- 清空错误路径为 `noexcept`/best-effort；
- backend 不缓存 context 裸指针到请求结束之后；
- completion handler 不直接解引用已析构 context，而持有线程安全 ledger state 或独立 token。

### 152.4 C API 的精确顺序

`prepare_memory_admission()` 建议重命名为 `prepare_memory_execution()`，返回
`std::unique_ptr<MemoryExecutionContext>`。顺序必须固定为：

```text
1. if (!memory_policy || !memory_policy->enabled) return nullptr
2. device identity（只读，不创建 command queue）
3. session.probe_memory_capability()（metadata-only）
4. production registry exact lookup
5. 确认 probe/candidate 后，停止旧 constrained generation
6. session.unload()；parent MLX synchronize + clear cache
7. 采 clean process baseline
8. compile exact manifest plan + validated upper gate
9. 再跑 model validate(runtime-adjusted sub-budget)
10. 构造 context/admission/scheduler/trace
11. bind context
12. 进入 session.generate()/prepare()
13. checkpoint final，收集 metrics
14. 解除 binding
15. finish_success 或 finish_failure
```

第 1 步必须是立即返回。虽然 `make_plan()` 当前只有 `enabled=true` 才创建 optional policy，API
仍应明确写成 `!policy || !policy->enabled`，防止后续 profile merge 或反序列化改变这一隐含条件。

### 152.5 `prepare` 与 `generate` 的 context 边界

首发 constrained route 不允许把请求级 context 跨 API 调用保存到 session。原因是 `prepare()`
返回后，调用方可能隔很久才 `generate()`，期间系统压力、baseline、checkpoint 和 device state
均可能变化。两种可接受方案：

- 首发简单方案：constrained `prepare` 只做 metadata probe/plan，不 materialize；`generate` 再完整
  admission；
- 后续 retained 方案：返回一个显式 `PreparedMemoryGeneration`，其所有 retained backing 继续
  被独立 persistent ledger 计费，并在 generate 前重新做 system admission。

禁止让普通 session cache 在 `prepare()` 后留下未计费 GPU/host backing，再在 `generate()` 创建
一个“干净”ledger。没有 persistent-cache protocol 前，受限模式应保持 clean-start。

### 152.6 allocator domain 和 generation 分配

推荐由 `tc_engine` 持有两个原子计数：

```cpp
std::atomic<uint64_t> next_memory_generation{1};
const uint64_t memory_allocator_domain; // engine 创建时随机/单调分配
```

每个 admitted request 获取唯一 generation；0 永远非法。`StorageId` 的 key 仍为
`(allocator_domain, handle, session_generation)`。同一 Metal buffer 的 pointer/Objective-C handle
在释放后可能被系统复用，因此不能仅以 address 去重；generation 是防止 ABA 的必要部分。

domain 应区分至少：

```text
engine-owned host malloc
H3 Metal device
LTX Metal device
MLX allocator bridge
child-process shared memory（未来）
```

如果多个 backend 实际共享同一个 `MTLBuffer`，只能由明确的 alias adapter 赋同一 domain/key；
不得仅凭地址相同自动折叠。

## 153. Allocation-site 注册与 constrained-reachable closure

仅靠 `grep malloc` 不能证明内存闭环：有些 allocation 在 disabled-only 分支，有些隐藏在
`std::vector`、Objective-C collection、MLX eval、MPSGraph 编译或第三方库内。需要同时维护
“静态 site 注册表”和“candidate 可达集合”，并在 capability promotion 前证明两者闭合。

### 153.1 site id 是稳定 ABI，不是随意日志字符串

建议 site id 命名：

```text
<model>.<component>.<operation>[.<variant>]

h3.schedule.object
h3.schedule.row_map.video
h3.schedule.gate_readback
h3.dit.block.17.refill.qkv
ltx.stage1.latent.video
ltx.video_vae.tile.accumulation
flux.transformer.group.03.weights
z_image.gguf.refill.slot.1
```

动态 block/tile index 不应导致 manifest site 数无限膨胀。稳定 `site_id` 表示代码位置/语义，
`AllocationInstance.instance_id` 表示本次计划中的 block/tile/slot 实例。site id 改名会改变 manifest
digest，应按 schema/revision 变更处理。

### 153.2 建议的源码注册形式

对于 C/C++/Objective-C 自有 allocator，建议显式声明静态 spec：

```cpp
constexpr AllocationSiteSpec kH3ScheduleGateReadback{
    .site_id = "h3.schedule.gate_readback",
    .component = "h3_dit_schedule",
    .stage = "dit_prepare",
    .memory_class = MemoryClass::ConversionScratch,
    .lifetime = AllocationLifetime::CommandBuffer,
    .provenance = UpperProvenance::ExactShapeFormula,
    .required = true,
    .asynchronous = false,
    .aliasable = true,
};
```

C runtime 不必依赖 C++ struct，可在 `h3_memory.h`/`ltx_native.h` 使用稳定字符串常量，平台
session 的 manifest builder 再映射到 `AllocationSiteSpec`。不要用预处理宏隐藏真实 allocator；
wrapper 的参数必须仍显式携带 class、site id、upper 和 generation。

### 153.3 candidate 可达性描述

建议新增 `MemoryReachabilityDescriptor`：

```cpp
struct MemoryReachabilityDescriptor {
    std::string candidate_id;
    std::vector<std::string> required_sites;
    std::vector<std::string> forbidden_sites;
    std::vector<std::string> disabled_features;
};
```

H3 首发 candidate 可先锁定：

```text
model=minimax-h3-turbo
operation=video.generate
execution=gpu
audio=false
inputs=[]
loras=[]
preview_denoise=false
retain_decoded=false
quantized_cache=""
parallel_prepare=false（由 constrained eligibility 保证）
```

在这个范围内，多模态 reference、audio VAE、preview decoder、Core ML/ANE、retained output 等
site 可列为 forbidden/unreachable；但 tokenizer、text encoder、layout、schedule、DiT、video VAE、
RGB8/export 仍然 required。不能因为产品请求没有 references，就忽略普通 text path 中的 host
分配。

### 153.4 reachable-site audit 的自动化

建议新增 `tools/native/audit_memory_sites.py`，输入 candidate descriptor 和源码目录，输出：

```json
{
  "candidate": "h3_c_metal_streamed_v1",
  "registered_sites": 87,
  "reachable_sites": 61,
  "guarded_sites": 58,
  "unknown_reachable_sites": [
    "h3.schedule.gate_readback",
    "h3.layout.segments",
    "h3.tokenizer.ids"
  ],
  "forbidden_hits": []
}
```

脚本可以组合三类证据：

1. allocator wrapper 注册表；
2. candidate-specific allow/deny path 表；
3. instrumented dry-run/fixture 的 site trace。

它不是通用静态分析器，不能证明任意 C 控制流；其价值是让“已知 allocation site 清单”和
“代码改动后新增裸 allocator”进入 code review。CI 还应做简单 source guard：在首发可达文件
新增 `malloc/calloc/realloc/newBufferWithLength/std::vector(size)` 时，若未同时更新注册表就失败。

### 153.5 unknown 与小对象阈值

不要用“少于 4 KiB 不计费”作为闭环捷径。大量小对象也可能积累，且 allocator page/zone
开销会反映在 footprint 中。可采用两层策略：

- 单个长期或 shape-dependent allocation：无论大小均注册；
- 高频固定小控制对象：可以按阶段 control-arena 聚合，一次 reserve arena upper，arena 内分配；
- 第三方无法逐对象 hook：使用 validated component envelope，但必须有独立 site、upper provenance、
  cold/steady 区分和 L2 实测；
- 无法给 exact/envelope 的 required site：`HeuristicNotExecutable`，candidate 保持 plan-only。

`guard_threshold_bytes` 只能表示 instrumentation/trace 的采样阈值，不能表示预算豁免。ledger 必须
按 arena 或真实 backing 的完整容量计费。

### 153.6 allocation closure 的 review 表

每个 required site 在 PR 中必须填写：

| 字段 | 说明 |
|---|---|
| site id | 稳定唯一标识 |
| source function | 实际 allocation 所在函数 |
| class | weight/activation/conditioning/refill/staging/output/cache/compile |
| upper formula | 用哪些 shape、dtype、alignment 计算 |
| provenance | exact metadata/exact formula/validated envelope |
| lifetime | acquire epoch、last-use、release epoch |
| async | 是否等待 GPU completion |
| alias | 与哪个 backing 共用，为什么安全 |
| fault cleanup | reserve/allocate/commit/submit/cancel 各失败如何清理 |
| disabled behavior | hooks 为 NULL 时是否完全走旧实现 |
| tests | minimum-1、exact、fault、double release、ABBA |

只有表内 required site 全部 `guarded` 且 runtime trace 未出现 unknown，才能从
`hook_bridged` 晋级 `allocation_guarded`。

## 154. Exact manifest 生成器与 metadata-only capability probe

`MemoryManifest` 和 `MemoryCapabilityRegistry` 已有基础实现，下一步不是手工在 session 中拼一份
常量 manifest，而是建立可重复的 checkpoint inspection 与 shape instantiation 流程。生产 probe
必须快、无 GPU 副作用，并且不能由模型代码自行授予 release capability。

### 154.1 将 manifest 拆成三层输入

建议把运行时 manifest 构造拆成：

```text
StaticSiteCatalog
    编译进二进制的 allocation-site 定义、公式版本、backend revision

CheckpointLayout
    从受信 metadata/index/sidecar 读取的 tensor 名称、dtype、shape、shard、offset、bytes

RequestInstantiation
    width/height/frames/steps/audio/input/slot/tiling 对应的 instance 和 live interval
```

最终 `MemoryManifest` 是三者确定性组合。这样 checkpoint 变化只影响 layout digest，请求 shape
只影响 instance/plan digest，代码 site 变化影响 backend/runtime revision。不要把 device family、
release state 写进模型侧 manifest；它们属于 `MemoryCandidateKey` 和外部 registry。

### 154.2 checkpoint identity 不能依赖绝对路径

同一 checkpoint 放在不同目录必须得到相同 identity。推荐 identity 输入：

```text
model format/version
canonical tensor index content
每个 shard 的相对逻辑名称
每个 shard 的 content SHA-256
必要的 tokenizer/config content digest
转换/量化 recipe digest
```

禁止把以下字段纳入 canonical digest：绝对路径、mtime、inode、下载时间、用户名。mtime/size 可以
用作本机 hash cache 的失效键，但不能替代 content identity。

对数十 GiB checkpoint 每次请求全量 SHA-256 会破坏首 token/首帧性能。推荐在模型准备阶段生成
只读 sidecar：

```json
{
  "schema": "turbocider.checkpoint_identity.v1",
  "model_id": "minimax-h3-turbo",
  "files": [
    {"logical_name": "dit/model-00001.safetensors",
     "size": 123,
     "sha256": "..."}
  ],
  "canonical_digest": "...",
  "created_by": "turbocider-model-prepare-vN"
}
```

runtime probe 读取 sidecar、检查 schema、逻辑文件存在和 size；是否每次重新全 hash 由 trust policy
决定。release evidence 必须记录采用的校验级别。若文件变化或 sidecar 缺失，受限执行 fail-closed；
默认路径不受影响。

### 154.3 tensor bytes 的精确计算

每个 checkpoint tensor 必须验证：

```text
element_count = checked_product(shape)
raw_bytes = checked_mul(element_count, dtype_storage_bytes)
offset + raw_bytes <= shard_size
```

量化格式不能简单使用“bits/element”估算，需加 scales、zero points、group metadata、alignment 和
pack padding。若 runtime 会将 INT8/BF16 转成另一种 GPU layout，manifest 应有两个 site：source
staging 与 converted GPU backing，并用实际 overlap lifetime 计峰值。

`MTLBuffer` upper 应按 backend 实际 alignment 公式计算，不把 tensor logical bytes 当作 allocated
capacity。建议把 alignment revision 纳入 `backend_revision`，例如：

```text
h3-metal-allocator-v3/alignment-256
ltx-metal-allocator-v2/storage-shared
mlx-runtime-<exact version>/allocator-envelope-v1
```

### 154.4 shape bucket 应是可判定合同

不要使用 `small/medium/large` 但没有边界的字符串。推荐 canonical form：

```text
w768_h1344_f22_s4_audio0_input0
w768_h1344_f45_s4_audio0_input0
w1024_h1024_f22_s8_audio0_input0
```

若为范围 bucket，必须编码闭区间和对齐条件：

```text
w[512,768]/32_h[512,1344]/32_f{22,45}_s[1,8]_audio0
```

manifest instance 必须用本次 exact shape 生成；bucket 只用于 registry 选择，不得以 bucket 最大值
之外的请求复用 record。若 upper 公式在范围内单调，可认证整个 bucket；若 attention/workspace
存在离散算法切换，应拆 bucket。

### 154.5 epoch builder 的阶段命名

当前 `compile_memory_plan()` 返回 `epoch_<id>`。后续 manifest builder 应维护确定性的语义 epoch：

```text
0 request_begin
10 text_load_begin
20 text_compute_begin
30 conditioning_ready_text_release
40 dit_trunk_load_begin
50 dit_step_0_block_0
...
900 dit_last_completion
910 vae_load_begin
920 vae_tile_0
...
990 export_complete
1000 request_end
```

epoch id 不必等同 block index，可以预留区间，便于插入事件而不重排全部 id。计划 digest 应包含
语义 epoch 表，否则相同数字 lifetime 在不同 builder 版本中可能被错误解释。

建议将 `MemoryManifest` schema 升级到 v2 时增加：

```cpp
std::vector<ScheduleEpoch> epochs;
std::string site_catalog_digest;
std::string shape_formula_revision;
```

v1 继续用于现有单元测试；release registry 只接受明确列出的 schema/revision。

### 154.6 probe 的纯度测试

每个 session 的 `probe_memory_capability()` 必须在测试中验证以下 counter 在前后不变：

```text
Metal device buffer allocation count
Metal command queue/buffer submission count
MLX active/cache memory
session resident weight count
worker/thread count（允许短暂同步 metadata reader 需单独声明）
process footprint 超过噪声阈值的增长
```

probe 允许：打开文件、`pread` 小型 header/index、解析 JSON/GGUF directory、读取 sidecar、计算
小型 canonical digest。probe 禁止：`mx::load_safetensors` 后 materialize、`mx::eval`、MPSGraph compile、
`newBufferWithLength`、启动长期 worker、清理现有 session。

### 154.7 registry 的生成和审查

production registry 继续以代码/只读 artifact 提供，不从环境变量、request 或本地 benchmark
自动写入。推荐将 release record 存在：

```text
native/runtime/memory_capabilities/
  h3_c_metal_streamed_v1.inc
  ltx_c_metal_streamed_video_v1.inc
```

每条 record 旁边写注释或生成 metadata：evidence bundle digest、测试设备、OS build、runtime revision、
验证日期、review ticket。生成脚本输出 patch，但不直接改 production registry；reviewer 需要确认
manifest/evidence digest 和 candidate key 后合入。

## 155. Scheduler/Pager 可实现 ABI：修正“先有 StorageId 再 reserve”的缺口

当前 `MemoryStageLease::allocate()` 接受已经存在的 `StorageId`，内部才
`try_reserve()->commit(storage)`。它适合逻辑测试，却不能证明真实 allocator 满足
`reserve -> allocate -> commit`，因为调用者可能已在传入 `StorageId` 前创建 backing。生产模型
adapter 必须使用两阶段 transaction。

### 155.1 C++ allocation transaction

建议新增：

```cpp
class MemoryAllocationTxn {
  public:
    MemoryAllocationTxn() = default;
    MemoryAllocationTxn(MemoryAllocationTxn &&) noexcept;
    ~MemoryAllocationTxn() noexcept; // 未 commit 自动 cancel

    explicit operator bool() const;
    uint64_t upper_bytes() const;
    StorageLease commit(const StorageId &, uint64_t actual_bytes);
    void cancel() noexcept;
};

class MemoryStageLease {
  public:
    std::optional<MemoryAllocationTxn> reserve(
        MemoryClass, uint64_t upper_bytes, std::string site_id,
        uint32_t instance_id);
    void adopt(StorageLease);
};
```

典型调用：

```cpp
auto txn = stage.reserve(MemoryClass::Weights, upper,
                         "flux.transformer.group.weights", group);
if (!txn) return budget_denied();

auto buffer = metal_allocate(upper); // 真正 allocation 在 reserve 之后
if (!buffer) return allocation_failed(); // txn 析构 cancel

const uint64_t actual = buffer.allocated_size();
auto lease = txn->commit(storage_id(buffer, generation), actual);
stage.adopt(std::move(lease));
```

`commit` 需要显式 `actual_bytes`，并验证 `0 < actual <= upper`；`StorageId.capacity` 可以设为 actual，
但不能让 allocator 伪报 logical size。Objective-C wrapper 应使用 `[buffer allocatedSize]` 或经验证的
等价值，而非请求长度。

### 155.2 C hook ABI 保持三步事务

H3/LTX 现有 hook 方向正确：

```text
reserve(user, class, upper, tag, &token, error)
commit(user, token, domain, handle, actual, generation, error)
cancel/release(user, token)
```

建议 v2 增加 `site_id` 与 `instance_id` 的独立字段，避免把可解析身份塞进 tag：

```c
int (*reserve_v2)(void *user,
                  uint32_t memory_class,
                  const char *site_id,
                  uint32_t instance_id,
                  uint64_t upper_bytes,
                  void **token,
                  char *error, size_t error_size);
```

v1 继续兼容首轮 patch；manifest trace 可暂时以稳定 tag 映射 site。ABI version 不匹配必须拒绝
constrained route，不能静默忽略 hooks。

### 155.3 Slot 和 pager 状态机

每个 refill slot 建议有显式状态：

```cpp
enum class RefillSlotState {
    Empty,
    Reserved,
    Reading,
    Converting,
    Ready,
    Submitted,
    Retiring,
    Faulted,
};

struct RefillSlot {
    uint32_t index;
    uint64_t generation;
    uint32_t logical_block;
    RefillSlotState state;
    StorageLease host_staging;
    StorageLease gpu_backing;
    PendingReleaseId pending;
    uint64_t ready_event_value;
    uint64_t done_event_value;
};
```

合法转移：

```text
Empty -> Reserved -> Reading -> Converting -> Ready
Ready -> Submitted -> Retiring -> Empty
任意未提交状态 -> Faulted -> Empty
Submitted -> Retiring 后必须等 GPU done，不能直接 Empty
```

slot 被复用时 generation 必须递增。`logical_block` 只是语义身份，不能作为 storage identity。
收到迟到 completion 时若 generation 不匹配，只记录 stale event，不释放新一代 backing。

### 155.4 Pager ticket

建议 I/O/pager 对 scheduler 暴露：

```cpp
struct PageRequest {
    uint32_t block;
    uint32_t slot;
    uint64_t generation;
    uint64_t source_offset;
    uint64_t source_bytes;
    uint64_t converted_upper_bytes;
    std::string site_id;
};

class PageTicket {
  public:
    bool ready() const;
    void wait();
    void cancel() noexcept;
    uint64_t bytes_read() const;
    double read_seconds() const;
};
```

ticket 必须拥有 host staging reservation，直到 conversion/upload 不再读取它。取消 ticket 需要区分：

- 还未开始 I/O：直接 cancel reservation；
- `pread` 进行中：标记 cancelled，I/O 返回后释放；
- GPU 已提交：进入 pending completion，不能同步 free；
- partial read/CRC mismatch：slot faulted、candidate request abort，不允许用旧 slot 内容继续计算。

### 155.5 bounded overlap 的选择公式

对 block `i`，只有以下条件全部满足才预取 `i+d`：

```text
candidate 认证 slots >= d+1
free slot 存在
reservation upper <= ledger.remaining
pressure state == Normal，或该预取是避免立即 stall 的 required depth
current block 的 GPU completion 不会与新 slot alias 冲突
source range 已通过 checkpoint manifest 验证
```

首发采用静态认证深度，不做复杂在线学习：

```text
H3: depth=1，两个 alternating slots
LTX: depth=min(request.max_refill_slots, verified_slots)-1
Flux/Z-Image F1 staged: 无 block look-ahead
```

后续自适应只能在已认证 revision 集合内选择。例如 registry 仅认证 2、3 slots，就不能因为 I/O
慢临时切成 4 slots。

### 155.6 pressure revision 与 safe point

`MemoryPressureDecision` 当前只有 Normal/Tight/Critical。建议增加 plan revision 输出：

```cpp
struct MemoryPressureDecision {
    MemoryPressureState state;
    bool stop_optional_prefetch;
    bool drain_completions;
    bool abort_request;
    std::optional<unsigned> target_refill_slots;
    std::optional<unsigned> target_pinned_blocks;
    uint64_t revision;
    std::string reason;
};
```

revision 只能在 manifest 标记的 `runtime_safe_point` 应用。denoise block 内只允许停止尚未开始的
look-ahead；减少 pinned prefix 需等该 block 的最后 use completion；VAE tile size 在 decode 开始前
冻结。Critical 为 sticky，首发直接 abort，不尝试在未知 footprint 越线后继续。

### 155.7 锁、线程和回调规则

建议固定锁顺序：

```text
request/context mutex
  -> scheduler mutex
    -> pager slot mutex
      -> ledger mutex
```

但是 allocator、`pread`、Metal API、user progress callback 和 trace flush 均不得在 ledger mutex 内
调用。reserve 在短锁内更新计数后返回 transaction；真实 allocation 在锁外；commit 再短锁。
completion handler 只做无阻塞 token transition，昂贵清理交给 scheduler thread。

用户取消与 pressure abort 统一进入 cancellation state，但错误码不同：前者返回 Cancelled，后者
返回 `memory_pressure_abort`。两者都必须 drain 已提交 GPU work 或让独立 completion owner 保持
ledger state 存活，绝不能因为 API stack unwound 就销毁 token 所依赖的对象。

## 156. H3 C/Metal 首发 patch：逐函数施工与验收

H3 已经最接近可认证路线，但“存在 hooks”与“整个 candidate allocation closure”仍有明显差距。
首发目标继续限定 `minimax-h3-turbo`、GPU、text-to-video、无 audio/inputs/LoRA/preview/retained
decoded、BF16 streamed DiT、两个 refill slots。

### 156.1 Patch H3-0：签名迁移

修改文件：

```text
native/models/h3_runtime/h3.c
tools/native/h3_dit_streaming_probe.c
```

按第 151 节完成 host options helper、parallel struct 按值复制、三个主调用点和 probe 的 `NULL`。
验收：native build、所有旧 probe 编译、disabled hook counter 为 0。

### 156.2 Patch H3-1：schedule gate readback closure

`h3_dit_schedule_gate_score()`、`h3_dit_schedule_gate_scores()`、
`h3_dit_schedule_gate_branch_scores()` 目前均按完整
`time_rows * BLOCK_OUTPUT * sizeof(uint16_t)` 裸 `malloc`。建议抽取：

```c
static uint16_t *schedule_readback_allocate(
    const h3_dit_schedule *schedule,
    size_t count,
    void **memory_token) {
    size_t bytes;
    if (!checked_mul_size(count, sizeof(uint16_t), &bytes)) return NULL;
    return host_allocate(
        schedule->host_memory.hooks ? &schedule->host_memory : NULL,
        H3_HOST_MEMORY_STAGING,
        bytes, 0,
        "h3.schedule.gate_readback",
        memory_token, NULL, 0);
}
```

三个 API 都使用同一 helper，并在 `h3_gpu_tensor_read_bf16` 成功或失败后执行 `host_release`。
若 public API 没有 error buffer，可保持原返回值语义，同时 trace 记录 reserve deny；不要为绕过错误
返回未初始化 score。

manifest 只需一个 aliasable site；由于三个 API 不应并发调用，可以让实例共享同一 alias group。
若实际调用允许并发，则必须移除 alias 或增加并发上界。

### 156.3 Patch H3-2：`prepare_rows()` 与 schedule object 验证

当前 schedule object、video/audio/condition row arrays、features 已开始使用 host helper。需要补测试：

- `sigmas->steps` 每个合法边界下 upper 精确；
- visual/audio condition on/off 时只创建可达 row array；
- 第 N 次 reserve、malloc、commit 故障时，之前 token 全部释放；
- features 在 time embedding upload 完成后释放，不跨 block projection 生命周期；
- `h3_dit_schedule_free(NULL)` 和部分构造对象安全；
- active mask 未选择的 block 不创建 `schedule->blocks[block]`；
- final AdaLN 总是 required，并有 conditioning class；
- hooks 为 NULL 时 object/rows 的分配和释放行为与原路径一致。

建议新增 `tests/native/h3_schedule_memory_test.mm`，用 fake GPU tensor allocator 或现有 hook probe
验证调用序列；无 Metal device 时，纯 host 部分仍应 PASS，真实 GPU 部分明确 SKIP。

### 156.4 Patch H3-3：普通 text-to-video 可达 host 清单

即使首发排除了 references，`h3_generate()` 仍有以下 host 生命周期需要审计：

```text
tokenizer ids/token buffers
text embedding output
h3_layout segments/maps
sigma schedule
condition row pointers（text-only 应为 0/NULL）
joint video/audio latent
Euler previous velocity
Video VAE decoded F32/tile arena
RGB8 conversion/output
media writer staging
cache key strings/control objects
```

固定小 control 对象可放入 request control arena；text embedding、layout maps、latent、decoded output
必须逐 site 或 exact envelope。`strdup` 的 key 不应在 constrained clean-start 下被 session cache
长期保留；若 cache disabled，确保 cleanup 不遗漏。

当前 candidate 设置 `audio=false` 只表示不导出音频，不一定表示 H3 内部联合 audio latent 完全不存在；
manifest 应依据实际 `h3_dit_audio_elements()` 和 denoise contract 计入内部 audio latent，不能因产品输出
关闭就扣掉模型需要的 audio branch。

### 156.5 Patch H3-4：GPU allocation 分类收尾

对 `h3_dit.c`、`h3_text_encoder.c`、`h3_video_vae.c`、`h3_gpu.m` 做 reachable audit：

```text
h3_gpu_tensor_new(
h3_gpu_tensor_from_*
h3_weight_load_*
newBufferWithLength
temporary MTLBuffer
converted INT8/BF16 weights
attention workspace
readback buffer
```

每个可达调用要么改用 `*_classified`，要么在上层已有准确 component envelope 且底层 backing 可用
唯一 handle commit。不能同时按 component envelope 和逐 tensor 重复计费。推荐首发优先逐 backing，
只对 Metal driver/PSO cache 使用 validated envelope。

需要特别验证 schedule projection 的临时 weight tensor：加载 block AdaLN weight/bias、提交 linear、
释放 weight/bias。若 `h3_gpu_submit()` 只提交不等待，`free_tensor` 必须确认底层 runtime 会延迟释放到
completion；否则现有代码本身存在 lifetime 风险，ledger 不能提前 release。manifest 的
`asynchronous=true` 和 last-use event 必须与真实实现一致。

### 156.6 Patch H3-5：session capability probe

在 `native/platform/apple/h3_session.mm` 实现：

```cpp
std::optional<MemoryCapabilityProbe> probe_memory_capability(
    const ExecutionPlan &plan,
    const MemoryDeviceIdentity &device) const override;
```

probe 只接受首发 scope，并从 session 已知 model root 读取 checkpoint identity/静态 layout。key 至少
固定：

```text
adapter=h3_c_metal_streamed_v1
backend=h3_c_metal
dtype=bf16
model_variant=base_no_lora
operation=video.generate
shape_bucket=<exact/bounded canonical shape>
sampler_mode=<steps+flow-shift contract>
refill_slots=2
tiling_mode=<certified video VAE tile revision>
runtime_revision=memory-runtime-v1 + h3 backend revision
device_family=<normalized Apple GPU family>
```

device name 的营销字符串不能直接作为 family；应归一化到 capability/evidence 使用的稳定标识。
checkpoint 内部有 quantized cache、Ref2VA 或不同 VAE 时必须产生不同 variant/digest。

### 156.7 Patch H3-6：真实执行 context 接线

`h3_session.mm` 在 `set_memory_execution_context()` 中创建稳定 bridge，填充：

```text
h3_gpu_options.memory_hooks
h3_gpu_options.memory_allocator_domain
h3_gpu_options.memory_generation
h3_params.host_memory_hooks
h3_params.memory_allocator_domain
h3_params.memory_generation
```

GPU 和 host domain 可以不同，但 generation 必须相同；若当前 H3 validator 要求同一 domain，则先
保持同一 engine-H3 domain，等 ledger alias 设计完善后再拆。bridge 生命周期覆盖完整
`h3_generate()`，解除 binding 后不再接收新 reserve；迟到 GPU completion 通过独立 shared ledger
state 完成。

### 156.8 H3 首发验收表

| 测试 | 预期 |
|---|---|
| disabled cold/warm | hook counter 0，route/cache/output 不变 |
| registry empty | 在 unload/大分配前 `memory_policy_unsupported` |
| exact record + minimum-1 | plan compile 后早拒绝，无 allocator call |
| gate readback reserve fail | API 返回失败，token=0，无泄漏 |
| active block mask | 未选择 block allocation count=0 |
| slot I/O fault | request abort，旧内容不提交 GPU |
| cancel during block | submitted backing pending 到 completion |
| VAE tile upper | actual allocated <= exact/validated upper |
| final result | unknown=0、pending=0、无 resident fallback |

H3 达到 L2 前，production registry 仍为空；达到 L3 后也只加入 exact checkpoint/device/shape record，
不使用通配符开放所有 H3 请求。

## 157. LTX C/Metal 首发 patch：统一 Metal、host vector 与 MLX 边界

LTX 的复杂度高于 H3：同一请求可能经过 Gemma/text、connector、Stage1 DiT、upsampler、Stage2
DiT、video VAE、audio VAE/vocoder，并混合 C、Objective-C/Metal、C++/MLX 与可选子进程。首发
candidate 已限定为 `video.generate + audio=false + inputs=[] + loras=[] + c_metal`；实现必须继续利用
这个窄范围，而不是一次认证所有 LTX 功能。

### 157.1 先冻结真实 backend 组合

`ltx_c_metal_streamed_video_v1` 的名字只说明 denoiser 路线，不足以说明 video decode/upsampler
实际走 Metal 还是 MLX。capability key 和 manifest 必须增加/编码完整 pipeline revision，例如：

```text
text_backend=ltx_native_gemma_gpu_v1
connector_backend=ltx_native_connector_v1
stage1_backend=ltx_c_metal_streamed_v1
upsampler_backend=ltx_metal_v1 或 ltx_mlx_v1
stage2_backend=ltx_c_metal_streamed_v1
video_vae_backend=ltx_metal_v1 或 ltx_mlx_tiled_v1
audio_backend=disabled
export_backend=rgb8_v1
```

如果 runtime 因文件可用性或环境变量切换 backend，probe 必须得到与实际执行完全相同的选择；
执行时再次校验，不允许 probe 认证 Metal VAE 后运行时回退 MLX VAE。建议将 backend selection
提前冻结进 `EffectiveMemoryPolicy`，session 只执行已冻结的 route。

### 157.2 Patch LTX-0：集中 direct `MTLBuffer` 创建

当前至少以下文件有 direct buffer 路径：

```text
native/models/ltx_runtime/ltx_gpu.m
native/models/ltx_runtime/ltx_video_vae.m
native/models/ltx_runtime/ltx_upsampler.m
```

建议所有 constrained-reachable `newBufferWithLength` 通过同一个内部 helper：

```objc
static id<MTLBuffer> ltx_new_accounted_buffer(
    id<MTLDevice> device,
    NSUInteger requested_bytes,
    MTLResourceOptions options,
    const ltx_memory_options *memory,
    ltx_memory_class memory_class,
    const char *site_id,
    uint32_t instance_id,
    void **token,
    char *error,
    size_t error_size);
```

事务顺序：checked requested upper → reserve → `newBufferWithLength` → 读取 allocated size → commit。
任何失败均 release Objective-C object 并 cancel token。free helper 的顺序为：先确保 GPU last-use
完成，再释放 Metal buffer，最后 release token；若只是提交了 command buffer，则 token 进入 pending，
不能在 ARC/local scope 结束时从 ledger 中扣除。

普通路径 `memory == NULL` 时 helper 直接调用原 `newBufferWithLength`，不生成 site string、不锁
ledger、不采样 footprint。

### 157.3 Patch LTX-1：C host allocator 注册表

当前 source audit 显示 host allocation 集中在：

```text
ltx_blocks.c
ltx_conditioning.c
ltx_connector.c
ltx_transformer_io.c
ltx_gemma_tokenizer.m
ltx_gemma_encoder.m
ltx_safetensors.m
ltx.c
```

不要逐个复制 H3 helper。建议在 `ltx_native.h`/新 `ltx_memory.h` 定义统一 hooks 和 options，在
`ltx_memory.c` 实现：

```c
void *ltx_host_alloc(const ltx_memory_options *, ltx_memory_class,
                     size_t bytes, int clear, const char *site_id,
                     uint32_t instance_id, void **token,
                     char *error, size_t error_size);
void ltx_host_free(const ltx_memory_options *, void *, void **token);
```

首发 required site 至少覆盖：

```text
conditioning embedding/parameters
connector combined/positions/rope scratch
transformer input/output host arrays
stage1 video/audio/mask latent buffers
stage1 noise/final BF16 conversion
upsampler input/output
stage2 video latent
video VAE tile input/output/accumulation
RGB8 output
tokenizer ids/mask
Gemma output embedding
safetensors header/index control arena
```

`ltx_blocks.c` 中 benchmark/reference-only 的大批 `actual_*`/`reference_*` allocation 若首发执行
不可达，应在 reachability descriptor 中列 forbidden，并用运行时 route assert 证明没有进入；不要
为了首发把测试-only allocation 计入生产峰值。

### 157.4 Patch LTX-2：`std::vector` 必须按 capacity backing 计费

`ltx_session.mm`、`ltx_mlx/native.cpp`、`ltx_mlx_video_vae.cpp` 等文件使用大量 shape-dependent
`std::vector`。计费 logical `size()*sizeof(T)` 可能低估 capacity；建议提供 constrained-only RAII：

```cpp
template <typename T>
class AccountedVector {
  public:
    bool resize_exact(size_t count, MemoryExecutionContext *,
                      MemoryClass, std::string_view site, uint32_t instance);
    std::vector<T> &value();
    void release();
};
```

实现可采用以下之一：

1. 自定义 allocator，在 `allocate(n)` 前 reserve，deallocate 后 release；
2. 已知只分配一次时，reserve `count*sizeof(T)+allocator_alignment_upper`，`vector.reserve(count)`，
   以 `capacity()*sizeof(T)` commit；
3. 多个小 vector 放进预留的 stage host arena。

禁止在默认 `std::vector` 已分配后才登记 `StorageId`。禁止假设 `shrink_to_fit()` 必然释放。需要
确定 stage boundary 时释放时，可用空 vector swap，但 ledger release 必须跟真实 allocator
deallocate 绑定，而不是仅跟 `size()==0` 绑定。

首发优先改造以下 shape-dependent 大对象：

```text
stage1_clean_prefix / stage2_clean_prefix
raw_video / raw_audio / raw_mask
video / audio latents
connected_video / connected_audio / connected_mask
upsampled stage2 video
decoded pixels / RGB
mel / waveform16 / waveform48（audio=false 时应不可达）
MLX video VAE accumulation/tile buffers（若选择 MLX VAE）
```

### 157.5 Patch LTX-3：Stage1→upsampler→Stage2 的 last-use 表

建议 manifest builder 明确下面的生存期，而不是只给“非 denoiser 4 GiB”：

| 资源 | acquire | last-use/release |
|---|---|---|
| text embedding | text complete | Stage1/Stage2 conditioning 最后一次读取后 |
| connector output | connector complete | 两阶段最后使用后，若 Stage2 可重建则比较重建成本 |
| Stage1 video latent | Stage1 init | upsampler 输入提交完成 |
| Stage1 audio latent | Stage1 init | audio=false 首发仍按内部 Stage1 contract 判断 |
| clean prefix | I2V only | 首发 inputs=[]，应不可达 |
| upsampler output | upsampler begin | Stage2 latent 初始化完成 |
| Stage2 latent | Stage2 init | video VAE input copy/command completion |
| transformer weights | 各 stage load | 对应 stage 最后 denoise completion |
| video VAE weights | VAE load | decode final completion |
| decoded pixels | tile/output | export callback/write complete |

默认 serial 计划应在 Stage1 transformer completion 后释放其非共享权重，再加载 upsampler；Stage2
同理。只有 manifest 证明 overlap 峰值小于 B 时才预加载下一组件。首发不为追求 load overlap
保留 Stage1 和 Stage2 两套完整 transformer。

### 157.6 Patch LTX-4：MLX/MPSGraph 采用独立 component envelope

如果首发 route 必须使用 `ltx_mlx_video_vae.cpp`、`ltx_mlx_upsampler.cpp` 或 MPSGraph，需单独
完成：

```text
cold compile upper
steady execute upper
MLX active bytes
MLX cache bytes
host tile/accumulation bytes
graph temporary bytes
weight materialization bytes
clear_cache 后仍保留的 framework bytes
```

当前“固定 2 GiB graph envelope”只能作为待验证 upper，不能直接赋予执行能力。建议 candidate
拆成：

```text
ltx_c_metal_streamed_video_metal_vae_v1
ltx_c_metal_streamed_video_mlx_vae_v1
```

分别认证。若 native Metal VAE 能完成同质量合同，首发优先选择全 C/Metal 路线，因为 allocator
可见性更好；但最终选择应以实际性能、质量和完整 closure 为准，不能仅凭技术偏好。

### 157.7 Patch LTX-5：streaming slot 的真实并发上限

LTX 当前允许最多三个 look-ahead slot，但 `request.max_refill_slots` 只是用户上限。planner 必须从
registry 的 `verified_refill_slots` 取交集：

```text
requested_max={1,2,3}
backend_supported={2,3}
record_verified={2}
=> effective=2
```

如果预算只能容纳 1 个 slot，而执行器未认证 1-slot 串行模式，应返回 infeasible；不能分配 2 个
slot 后声称按 1-slot 计划运行。每个 slot 的 host read buffer、converted GPU buffer和 alignment
分别计费，只有明确 alias 才合并。

### 157.8 LTX 首发验收表

| 测试 | 预期 |
|---|---|
| backend freeze | probe 与 execute 的每个 component backend 完全一致 |
| direct Metal fault | reserve/newBuffer/commit 任一点失败均无 token 泄漏 |
| vector growth | 实际 capacity 不超过 reservation，超出立即 fail |
| Stage1 release | Stage2 load 前旧 transformer backing 已完成释放 |
| 2/3 slot identity | 不同 slot count 产生不同 key/plan/evidence |
| audio=false | audio export/vocoder site 不可达；内部 audio latent按实际模型计费 |
| MLX cold/steady | 分开 upper；cold 不能复用 steady record |
| output | 视频尺寸/fps/seed contract 与默认路径一致 |

## 158. Flux 与 Z-Image：首发 staged 方案和后续 layer-group pager

Flux/Z-Image 当前主要依赖 MLX tensor map、compiled graph 和整模型 class，对它们直接套 H3 的 C
refill slot 会形成“接口看起来统一、底层仍整包 materialize”的假 streaming。首发应以 component
staging 为目标，证明 text/transformer/VAE 不同时常驻；只有完成权重索引和 block forward 解耦后
再开放 layer-group pager。

### 158.1 共同的 F1 staged candidate 合同

F1 仅允许：

```text
同一 checkpoint/precision
GPU/MLX
无 LoRA
固定 text encoder route
固定 transformer route
固定 VAE/decode route
固定 shape bucket
阶段串行，禁止 transformer 与 VAE 隐式并载
```

F1 的 `effective_residency=component_staged`，不能返回 `streamed`。即便 stage 之间 unload 成功，如果
单独 transformer stage 峰值仍超过 B，则 candidate infeasible，不依赖 swap。

### 158.2 Flux F1 文件级改造

建议涉及：

```text
native/models/flux_module.cpp
native/models/flux2/pipeline.cpp
native/models/flux2/flux_encode.cpp
native/models/flux2/flux_transformer.cpp
native/models/flux2/flux_vae.cpp
native/models/flux2/flux.hpp
对应 Apple session/module factory
```

将 pipeline 拆成明确生命周期：

```cpp
TextConditioning load_and_encode_text(...);
void unload_text_encoder();

Latent run_transformer(TextConditioning &&, ...);
void unload_transformer();

Pixels decode_latent(Latent &&, ...);
void unload_vae();
```

实际对象可以复用已有 class，但必须暴露 unload 后验证：相关 tensor map 不再持有 backing、
`mx::eval` 已完成、MLX cache 的 retained bytes 被独立计入或清理。`pipeline.cpp` 中对 latent/noise
的 `mx::eval` 是自然 safe point，但不能把 `eval` 返回当作所有 allocator cache 已释放。

F1 manifest 至少有：text weights/activations/output、transformer weights、compiled graph cold/steady、
latent/noise、VAE weights/tiles/pixels、MLX cache envelope。

### 158.3 Flux F2 layer-group pager

F2 前置条件：

- safetensors index 能按 block/group 映射 tensor 到 shard+offset；
- transformer forward 接受外部 `BlockWeights`，不从整模型 map 随机访问；
- compiled block graph 不 capture 全模型 tensor；
- group unload 后无 hidden reference；
- LoRA 首发禁用或已预合并成独立 checkpoint identity。

推荐 group size 枚举 `{1,2,4}`，而不是任意 N；manifest 根据每组最大 weight bytes 和 workspace
选择。MLX lazy array 不能直接作为“已 offload”证明，必须在执行 trace 中观察 materialized backing
和 cache 行为。

### 158.4 Z-Image BF16 F1

`native/models/z_image/z_image.cpp` 当前有 `mx::load_safetensors`、多个 compiled graph 与逐 block
`mx::eval`。BF16 F1 与 Flux 类似，先拆 text/transformer/VAE stage。静态 `z_gpu_block_graph()` 等
compiled graph 可能跨请求保留 framework 内存；默认路径可继续保留，constrained candidate 必须：

- 在 clean baseline 前明确清理，或
- 将其 cold/steady retained upper 作为 process/framework envelope 计入。

不能把静态 C++ function object 本身的大小当 graph footprint；需要真实 runtime calibration。

### 158.5 Z-Image GGUF F1/F2 的映射风险

GGUF 量化能减少权重存储，但 `mmap`/lazy page 不等于可控内存。文件页、统一内存和压缩/swap
仍可能竞争。GGUF candidate 必须明确使用哪种 source：

```text
bounded pread -> host staging -> GPU converted slot（最可控）
只读 mmap + bounded materialization（需额外 page-residency 证据）
整文件 MLX load（不能称为 F2 streaming）
```

F1 可以使用 component staged GGUF，但要把整 transformer stage 的实际 footprint 纳入 upper。F2
建议为每个 layer group 建立 GGUF tensor directory，按 offset 读取到固定 staging/converted slots。
source file length 不计为 committed RSS，但实际 materialized/mapped resident pages必须通过 process
footprint residual/envelope 覆盖；不能简单计 0。

BF16 与 GGUF 使用不同 `model_variant`、dtype、manifest、quality evidence。任何 GGUF approximation
都必须是用户选定的模型变体，内存模式不能自动把 BF16 请求降成 GGUF。

### 158.6 Flux/Z-Image probe 和 registry

建议 candidate id：

```text
flux2_klein_4b_mlx_component_staged_v1
flux2_klein_9b_mlx_component_staged_v1
z_image_bf16_mlx_component_staged_v1
z_image_gguf_mlx_component_staged_v1
```

不同模型规模不能共享 validated upper。9B record 不能由 4B 结果外推；不同 GGUF quant type 也不能
共享。F1 达到 L3 后才进入 production registry；F2 pager 是新 adapter/revision，重新认证。

### 158.7 staged route 的验收重点

| 风险 | 必须观察 |
|---|---|
| hidden tensor capture | stage unload 后 MLX active/cache 和 process footprint 是否下降 |
| graph cold peak | 第一次 compile 峰值是否独立覆盖 |
| static graph cache | repeat request 是否被 baseline/envelope 正确计费 |
| safetensors lazy load | 实际 materialize 时点和峰值，而非仅文件大小 |
| VAE overlap | transformer last-use 后才加载 VAE，除非 plan 显式重叠 |
| quality | 与相同 checkpoint/precision 默认路径固定 seed 对齐 |

## 159. 自动化验收与 evidence runner 设计

设计能否落地，取决于是否把“预算不超限、无 swap 增量、无 fallback、质量相同、默认路径无回归”
变成自动化产物，而不是人工看 Activity Monitor。建议增加统一 runner 和分层测试目录。

### 159.1 建议文件布局

```text
tests/native/
  memory_execution_context_test.cpp
  memory_allocation_txn_test.cpp
  memory_reachability_test.cpp
  h3_schedule_memory_test.mm
  ltx_host_memory_test.cpp
  ltx_stage_lifetime_test.mm
  test_memory_execution_context.py
  test_memory_reachability.py
  test_h3_schedule_memory.py
  test_ltx_stage_lifetime.py

tools/native/
  audit_memory_sites.py
  build_memory_manifest.py
  run_memory_campaign.py
  compare_memory_abba.py

results/memory-constrained/            # 默认 gitignore，仅 evidence 发布流程选取
  <candidate>/<timestamp>/
```

测试 binary 可复用现有 Python wrapper 编译方式。真实设备 campaign 不应混入普通 unit suite，以免
无 Metal/无 checkpoint 环境误报失败；但缺设备必须清楚输出 SKIP reason。

### 159.2 逻辑测试必须覆盖的状态空间

`MemoryAllocationTxn`：

```text
reserve deny
allocator returns null
actual=0
actual=upper
actual=upper+1
commit duplicate StorageId
alias same capacity
alias changed capacity
cancel twice
release twice
retire then complete
retire then stale generation completion
```

`MemoryStageScheduler`：

```text
Normal -> Tight -> 3 samples recovery -> Normal
Normal/Tight -> Critical sticky
safe point 前 revision 不应用
pending completion 阻止 slot reuse
cancel during Reading/Converting/Submitted
budget arithmetic overflow
system reserve boundary equal/less/greater
```

`MemoryManifest`：

```text
canonical input ordering independence
duplicate site/instance rejection
required site no instance rejection
heuristic required site rejection
half-open release-before-acquire
alias max not sum
non-aliasable instance declares alias rejection
epoch semantic digest change
checkpoint/candidate/runtime/device 任一变化不匹配
```

### 159.3 fault injection 接口

建议仅 test build 启用确定性 fault plan：

```cpp
struct MemoryFaultPlan {
    std::optional<uint64_t> fail_reservation_number;
    std::optional<uint64_t> fail_allocation_number;
    std::optional<uint64_t> fail_commit_number;
    std::optional<uint64_t> fail_io_number;
    std::optional<uint64_t> delay_completion_number;
    std::optional<std::string> fail_site_id;
};
```

不要用随机 OOM 作为主要 fault 测试；确定性序号/site 让每个 cleanup 分支可重放。release build 不从
普通环境变量接受 fault plan，避免成为生产绕过或不稳定源。

### 159.4 evidence bundle 规范

每次 L3 campaign 输出：

```text
request.redacted.json
effective_policy.json
candidate_key.json
checkpoint_identity.json
memory_manifest.json
compiled_plan.json
allocation_sites.json
schedule_trace.jsonl
footprint_samples.jsonl
swap_before_after.json
timings.json
quality.json
fault_results.json
default_abba.json
system.json
summary.json
```

`summary.json` 包含所有文件 SHA-256，形成 `evidence_digest`。prompt 和绝对模型路径必须脱敏；保留
shape、seed、steps、模型 digest。evidence bundle 可重放 plan/quality 检查，但不包含模型权重。

### 159.5 process footprint 与 swap 采样

建议 runner 在以下边界强制采样，并在运行中以 10–50 ms 低优先级采样补充：

```text
before preflight
after clean unload/cache clear
after admission
text peak/end
each component load end
denoise first/middle/last step
before/after each stage transition
VAE each tile group
export begin/end
after final drain
```

高频 sampler 本身有开销，ABBA 中必须关闭或保持两侧一致。硬判断以 allocation guard 和强制边界
为主，sampler 用于发现 framework/unattributed gap。

swap 记录至少包含 `vm_stat` swapins/swapouts 增量、压缩页变化和 `sysctl vm.swapusage` 前后值。
安静专用机要求 `swapouts_delta==0`；共享开发机出现系统其他进程污染时标记 inconclusive，不把失败
改写成 PASS。

### 159.6 性能指标和 SLO

每个 candidate 报告：

```text
T_total
T_text / T_load / T_denoise / T_vae / T_export
I/O bytes、read bandwidth、read wait
GPU busy/idle（可用时）
prefetch hit rate
stall_fraction = refill_wait / denoise_wall
hidden_fraction = 1 - exposed_io_wait / total_io_time
peak_known / peak_unattributed / process_peak
```

建议初始 release SLO：

- disabled ABBA 稳态回归 ≤2%；
- constrained 相对同 backend unbounded baseline 的总耗时开销按 candidate 单独设定，不统一承诺；
- H3/LTX streaming 的 exposed refill wait 应有明确上限，例如 denoise wall 的 15–25%，实际阈值由
  目标 SSD/device campaign 决定；
- actual process peak ≤ planned upper ≤ B；
- required unknown=0、resident fallback=false、swapouts delta=0；
- 固定 seed 输出满足模型既有 parity/quality threshold。

不能只优化 tokens/s 或 block compute；低内存模式的主要用户体验是端到端稳定、不抖动、不因 swap
长尾卡死，因此至少报告 P50/P95/P99 stage latency 和 cold/warm 分布。

### 159.7 minimum budget 二分测试

runner 可对 exact candidate 进行离线 planner 二分，找到 `B_min_plan`，然后测试：

```text
B_min_plan - 1 byte       必须在任何 allocation 前拒绝
B_min_plan                必须成功或暴露 upper 公式 bug
B_min_plan + 5%           观察是否减少 stall
B_min_plan + 15%          验证更多 pinned/look-ahead candidate
```

这里测试的是 effective B，不是用户 Y。runner 根据 X 反推最小 Y 时必须向上取整，防止整数除法导致
实际 B 少 1 byte。报告同时列 Y、X、B，不使用“16GB 模式”这种含糊标签。

### 159.8 低内存实机矩阵

至少准备 16/24/32 GiB 目标机（取决于 candidate minimum），并覆盖：

```text
冷启动/热启动
内部 SSD/目标支持的外置 SSD
最小/最大 shape bucket
低/高 steps
取消、I/O fault、allocation fault
系统安静、受控轻后台负载
重启后规定窗口的冷 page cache
连续 5–10 次请求的 steady state
```

64 GiB 机器设置较小 Y 仍是必要逻辑测试，但不能替代这些实机。设备证据绑定具体 family、OS build、
Metal/runtime revision；升级系统或 MLX 后需按 change matrix 决定重跑 L2/L3。

### 159.9 最新回归命令模板

在第 151 节签名修复并恢复 build green 后，至少执行：

```sh
env MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" \
  TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
python3 -B tests/native/test_contract.py
python3 -B tests/native/test_memory_accounting.py
python3 -B tests/native/test_memory_manifest.py
python3 -B tests/native/test_memory_plan_compiler.py
python3 -B tests/native/test_memory_scheduler.py
python3 -B tests/native/test_memory_execution.py
python3 -B tests/native/test_h3_schedule_memory.py
python3 -B tests/native/test_h3_streaming_policy.py
python3 -B tests/native/test_ltx_streaming_benchmark.py
python3 -B tests/native/test_h3_gpu_memory_hooks.py
python3 -B tests/native/test_ltx_gpu_memory_hooks.py
git diff --check
```

当前工作树已经恢复 native build green，并通过 summary 中列出的逻辑回归；真实 Metal hook 测试
由于当前环境没有 Metal device 只能 SKIP。后续任何 release evidence 仍需在相同 HEAD/binary hash
下重新执行整套命令，不能把无设备 SKIP 或不同代码状态的历史结果拼成 L2/L3 PASS。

## 160. 需求追踪矩阵、实施优先级与最终决策

本节将产品要求映射到代码和验收。实现团队可按 requirement id 在 PR、测试和 evidence 中引用，
减少“某项原则写过但没人负责”的风险。

### 160.1 核心需求追踪

| ID | 需求 | 主要代码 | 必须测试 |
|---|---|---|---|
| MEM-001 | add-on 显式开启 | request/profile、`plan.cpp` | disabled ABBA、enabled validation |
| MEM-002 | B=`floor(Y*(100-X)/100)` | `memory_policy.*` | overflow、边界、rounding |
| MEM-003 | 不以 swap 为调度机制 | planner/scheduler/model adapters | swap delta、无 unbounded fallback |
| MEM-004 | capability 在大分配前 gate | `memory_execution.*`、`c_api.mm` | registry miss 无 unload/allocation |
| MEM-005 | exact checkpoint/device/runtime | manifest/registry/session probe | 任一 identity 变化拒绝 |
| MEM-006 | 全请求 live peak | manifest/plan compiler | alias、half-open、stage overlap |
| MEM-007 | reserve-before-allocate | allocator wrappers/transaction | fault injection sequence |
| MEM-008 | GPU completion 后释放 | scheduler/Metal hooks | delayed/stale completion |
| MEM-009 | unknown required site 为 0 | reachability/site registry | source guard、runtime trace |
| MEM-010 | bounded overlap | pager/slot scheduler | slot count、generation、I/O stall |
| MEM-011 | pressure fail-closed | pressure controller | Tight recovery、Critical sticky |
| MEM-012 | 默认路径零影响 | all adapters/C API | A/B/B/A、hook/probe counter 0 |
| MEM-013 | H3 首发 | H3 session/C runtime | L0–L3 H3 campaign |
| MEM-014 | LTX 首发 | LTX session/C/Metal/MLX | L0–L3 LTX campaign |
| MEM-015 | Flux/Z staged | Flux session/pipeline | stage unload、cold/steady upper |
| MEM-016 | Z GGUF variant 隔离 | GGUF loader/pager | BF16/GGUF key/quality 分离 |
| MEM-017 | 可重放证据 | runner/result protocol | evidence digest/replay |
| MEM-018 | 错误可解释 | C API/result JSON | error precedence/redaction |

### 160.2 推荐实施优先级（按阻塞关系）

```text
P0-A  修复 H3 host-options 签名迁移，恢复 native build
P0-B  保持 production registry 为空，确认 capability miss 早拒绝
P0-C  引入真正的 MemoryAllocationTxn / reserve-before-allocate API

P1-A  H3 schedule gate readback + host reachable closure
P1-B  H3 session probe + exact manifest builder
P1-C  H3 Metal completion/pending release + L2 tests
P1-D  H3 低内存 L3 campaign + whitelist review

P2-A  LTX direct Metal wrapper closure
P2-B  LTX host/vector/stage lifetime closure
P2-C  冻结 video VAE/upsampler backend，分开 MLX/Metal manifest
P2-D  LTX L2/L3 + whitelist review

P3-A  Flux 4B staged F1
P3-B  Z-Image BF16/GGUF staged F1
P3-C  Flux 9B 独立认证
P3-D  layer-group pager F2
```

P0 未完成前，不应并行推广更多 candidate；否则所有模型都会复制一个不完整的 transaction/context
接口。P1/P2 的模型内 allocation audit 可以并行准备，但 promotion 必须独立。

### 160.3 每个 PR 的强制说明模板

```text
Scope:
Candidate ids:
Changed allocation sites:
New/changed upper formulas:
Lifetime/last-use changes:
Alias/generation rules:
Default-path behavior:
Failure cleanup:
Tests run:
Metal/device availability:
Known unknown sites:
Capability level before/after:
Registry change: none/experimental/release
Evidence digest:
```

若 PR 改了 allocator、shape formula、backend selection、compiled graph、runtime/MLX version或
checkpoint conversion，必须明确 capability record 是否失效。默认答案应是失效/需复核，而不是
假设旧 evidence 仍适用。

### 160.4 发布前 go/no-go 清单

Go 需要同时满足：

```text
[ ] 当前 HEAD native build 和逻辑回归全部通过
[ ] production registry 只含 reviewed exact records
[ ] probe metadata-only，registry miss 在大分配前拒绝
[ ] planned upper 来自完整 manifest，无 required heuristic/unknown
[ ] 所有真实 backing reserve-before-allocate
[ ] 所有异步 backing completion 后释放，pending 最终为 0
[ ] actual peak <= planned upper <= B
[ ] 安静低内存实机 swapouts_delta == 0
[ ] cancel/fault 不泄漏，不回退 resident/unbounded
[ ] disabled ABBA 稳态回归 <=2%
[ ] quality/媒体合同通过
[ ] evidence bundle 完整、digest 可重算
[ ] quarantine/rollback 已演练
```

任一项不满足即 No-Go。尤其不能用以下理由放行：

```text
“大多数时候没 swap”
“Activity Monitor 看起来低于 Y”
“planner 估算能放下”
“Metal 测试在无设备环境 SKIP 了”
“H3/LTX 某个 block 已有 streaming”
“低预算跑在 64 GiB 机器成功”
```

### 160.5 最终架构判断

对于 TurboCider 的内存受限 add-on，正确产品策略仍是：

1. **TurboCider 主动做 component/block/tile residency 调度；**
2. **用完整 manifest + allocation guard 把受控进程峰值约束在 B 内；**
3. **把 swap 视为系统外部风险指标和失败信号，不视为容量扩展；**
4. **先认证 H3/LTX 的窄 GPU candidate，再做 Flux/Z-Image staged，最后 pager 化；**
5. **enabled=false 在入口立即走原路径，所有 probe、unload、ledger、trace 和 cache 改动均为零。**

这比“内存不足后让 macOS 自动 swap”更适合作为可承诺、可测试的 TurboCider 模式。swap 对普通
桌面应用是合理的 OS 兜底，但对长时间 GPU 生成会引入不可预测的 page-in/page-out、统一内存争用、
尾延迟和失败放大，无法构成稳定产品合同。与此同时，设计只能承诺 TurboCider 自身已知 allocation
和经验证 framework envelope，不应宣称能对 macOS 全系统提供绝对硬内存上限。最终对外文案应使用
“预算受控、在认证设备/模型/shape 下验证无 swap 增量”，而不是“操作系统级绝不超过 Y”。

## 161. 2026-09-15 追加施工合同：把 scaffolding 变成可交付实现

本节不是重新定义架构，而是把当前工作树中已经存在的对象、尚未接线的边界和每个 patch 的退出条件
固定下来。实现者应把“当前事实”和“建议行为”分开看：当前代码可以编译和通过纯逻辑测试，不代表
真实模型已经具备可发布的 constrained capability。除非本节明确写成“当前已验证”，否则均属于待实现
合同。

### 161.1 当前代码到目标闭环的差距表

| 位置 | 当前事实 | 必须补齐的目标 | 退出条件 |
|---|---|---|---|
| `native/api/c_api.mm::prepare_memory_execution` | 已做 probe→clean unload→baseline→plan→context 的主顺序 | 将 probe 失败、unload 失败、baseline 失效、plan 失败映射成稳定错误码，并确保任何大分配前都没有模型 materialize | fault test 能证明失败前 allocation counter=0 |
| `native/runtime/session.hpp` | 有 `set_memory_admission(MemoryAdmission*)` 和 metadata-only probe 默认接口 | 增加 context 绑定的过渡 API；旧 admission 只作为兼容桥，不再成为新 adapter 的主接口 | H3/LTX adapter 只从 context 取得 admission/scheduler/generation |
| `native/platform/apple/h3_session.mm` | 生成路径安装 GPU/host hooks；未实现 exact probe | 从 checkpoint index/sidecar 生成 manifest，不创建 GPU buffer、不加载 tensor、不编译 graph | probe purity test、manifest digest 稳定、registry miss 仍拒绝 |
| `native/platform/apple/ltx_session.mm` | C/Metal allocation hooks 已有部分接线；MLX/graph/host 仍有缺口 | 冻结首发 candidate 为 C/Metal、GPU、video-only、text-to-video；补齐 direct buffer 和阶段 envelope | reachable-site unknown=0，L2 actual<=upper |
| `native/models/h3_runtime/h3.c` | GPU/host hook 和 schedule readback 已有事务调用 | 所有 constrained-reachable host/GPU site 使用同一 domain+generation；禁止异步 worker 借用栈指针 | sanitizer/故障序列无 stale token、无悬空 options |
| `native/models/ltx_runtime/ltx_gpu.m` | direct buffer 已有 reserve/commit/cancel/release hook | 所有 MTLBuffer 创建都通过一个 wrapper，实际 `allocatedLength` 回传 commit | 源码审计无裸 `newBufferWithLength`（白名单除外） |
| `native/runtime/memory_execution.*` | context 能冻结 plan、观察 pressure、校验成功释放 | 失败时区分“等待 GPU completion”和“已损坏 worker”；增加 bounded drain/quarantine 语义 | cancel/fault 后 pending 不被伪造为 complete |
| `native/platform/apple/results.mm` | 已输出基础 policy/admission 字段 | 增加 candidate、plan、trace、peak、unknown、fallback、swap delta 字段，并保持旧客户端可解析 | schema-v1 客户端忽略新字段，schema-v2 能重放 evidence |

### 161.2 context 绑定 API 的过渡设计

新 adapter 不应继续扩散裸 `MemoryAdmission*`。建议在 `ModelSession` 增加以下可选接口；这是一份
实现建议，不要求一次性修改所有旧模型：

```cpp
class MemoryExecutionContext;

class ModelSession {
public:
    virtual void bind_memory_context(MemoryExecutionContext *context) {
        // Default is deliberately inert. The C API installs the old
        // admission bridge separately during the migration period.
        (void)context;
    }
    virtual void unbind_memory_context() noexcept {
        // New adapters override this to clear their non-owning context.
    }
    virtual void memory_checkpoint(std::string_view phase) {
        (void)phase;
    }
};
```

`c_api.mm` 的新顺序应为：

```cpp
auto context = prepare_memory_execution(*session, plan, parent_mlx);
ScopedMemoryContextBinding binding(*session, context.get());
// Binding happens exactly once and remains valid until all model calls return.
auto result = session->generate(request, event, cancelled);
context->checkpoint("generate_returned");
context->finish_success();
```

过渡期允许 `ScopedMemoryAdmissionBinding` 继续存在，但必须满足：

1. 同一请求不能同时安装两个不同 admission；若 adapter 检测到 generation 不一致，立即返回
   `memory_lifetime_violation`。
2. 过渡期的 `ScopedMemoryContextBinding` 先调用
   `set_memory_admission(context ? &context->admission() : nullptr)`，再调用
   `bind_memory_context(context)`；析构按相反顺序 unbind/clear。这样 header 不需要在
   `MemoryExecutionContext` 仍为 incomplete type 时内联调用 `context->admission()`，避免
   `session.hpp` 与 `memory_execution.hpp` 的循环依赖。
3. `unbind_memory_context()` 和 `set_memory_admission(nullptr)` 必须在 context 析构前执行；
   析构顺序不能依赖 Objective-C
   autorelease pool 的偶然 drain。
4. `tc_engine_load()` 和无 request 的 `tc_engine_unload()` 不能偷偷创建 constrained context。
   它们要么保持旧路径，要么新增带 request 的显式 API；第一版推荐保持旧路径，避免“load 阶段
   未知预算、generate 阶段重新 admission”的双重计费。
5. `tc_engine_cache(action=clear)` 属于显式缓存管理，不得在 disabled 路径因为引入 context 而
   自动执行。constrained generate 的 clean boundary 可以调用 `unload()+mx::clear_cache()`，
   但该动作必须计入 trace，并只发生在 enabled 请求。

### 161.3 单请求并发合同与未来扩展

首发只承诺一个 GPU constrained request。当前 `execution_mutex()` 已将 GPU runtime 串行化，
因此不会把两个请求各自的 B 相加后误以为可以并行。文档和结果字段应明确：

```text
global_gpu_execution = 1
constrained_inflight = 1
default_path_inflight = 0 while constrained request owns the runtime
```

未来若要并行，必须先引入 `MemoryBudgetBroker`，不能简单删除全局 mutex：

```cpp
struct JobSlice {
    uint64_t budget_bytes;          // <= global B - process baseline
    uint64_t system_reserve_bytes;
    unsigned priority;
    uint64_t admission_sequence;
};

std::optional<JobSlice> broker_admit(
    uint64_t planned_peak, uint64_t system_reserve,
    unsigned priority, uint64_t deadline_ns);
```

并发扩展的最低要求是：每个 job 有独立 ledger/generation，shared immutable weights 以 backing
identity 去重，任一 job 的取消不能释放另一个 job 仍在 GPU 使用的 backing；否则宁可继续串行。
在没有这套 broker 和真实 overlap evidence 前，`parallel_jobs>1` 应返回
`memory_policy_unsupported`，不能把 OS swap 当作隐式共享池。

### 161.4 prepare/generate/cache 的生命周期边界

请求级 context 只覆盖一个有明确 request 的 `prepare` 或 `generate` 调用。建议固定以下三种情况：

| 调用 | 是否可创建 context | 预算归属 | cache 行为 |
|---|---:|---|---|
| `tc_engine_generate(request)` | 是，唯一首发入口 | 完整请求 B | enabled 下先清理旧 resident；成功后不得把未知 backing 留在 cache |
| `tc_engine_prepare(request)` | 是，但只能在 `prepare` 自己完成并释放 | 完整 prepare peak | 若 warmup 会编译 graph，compile temporary 必须在 manifest 中 |
| `tc_engine_load()` | 否（第一版） | 无 constrained 合同 | 保持旧路径，不读取 memory_constrained 配置 |

`prepare` 成功后如果 session 保留权重，必须把这些 backing 纳入可重用 capability 的 cache identity；
否则下一次 constrained generate 必须按“clean process baseline”处理并主动 unload。不能出现第一次
请求以为权重已释放、第二次请求却从 session cache 继承未计账 resident 的情况。

### 161.5 失败路径：drain、quarantine、taint 的精确定义

当前 `finish_failure()` 不能只设置一个 bool 就结束。建议增加一个更强的替代接口，并由旧函数
作为兼容 wrapper 调用它：

```cpp
enum class FailureDisposition : uint8_t {
    Clean,             // 所有 reservation/release 已完成
    NeedsGpuDrain,     // 有 pending completion，必须在外层等待
    QuarantineWorker,  // 无法证明 GPU 已停止使用 backing
};

FailureDisposition MemoryExecutionContext::finalize_failure(
    std::string reason, uint64_t drain_deadline_ns) noexcept;
```

这不是对现有 `void finish_failure(std::string) noexcept` 的同签名重载；过渡期可保留旧接口，
让它调用 `finalize_failure()` 并把 disposition 写入 metrics。所有 C API 入口迁移后再删除旧 wrapper。

规则：

- 仍在 CPU 阶段、没有 pending token：取消所有 reservation，释放 active lease，状态为 `Clean`。
- 有 pending GPU completion：不得调用 `complete_pending()` 假装已完成；外层在非 callback 线程
  poll event，直到完成或 deadline 到期。
- deadline 到期、GPU queue 状态未知、或出现 stale generation：状态为 `QuarantineWorker`，
  关闭该 session/worker，错误码为 `memory_lifetime_violation` 或
  `memory_pressure_abort`，后续请求不得复用该 worker。
- `tainted=true` 一旦设置不可清除；后续成功回收只能让资源可回收，不能把本次结果改成 PASS。

这使“无泄漏”与“无伪造 completion”同时成立。只有 `Clean` 才允许把请求错误归类为普通
`memory_budget_too_small`；`NeedsGpuDrain`/`QuarantineWorker` 必须在 evidence 中单独记录。

### 161.6 本轮当前 HEAD 验证结果

2026-09-15 本轮重新执行 native build 与逻辑回归，结果如下：

```text
native build                              PASS
contract suite                           81 PASS / 1 Wan fixture SKIP
memory accounting                        PASS
memory manifest                          PASS
memory plan compiler                     PASS
memory scheduler                         PASS
memory execution context                 PASS
H3 schedule memory                       PASS
H3 shared streaming policy               PASS
LTX streaming benchmark                  PASS
H3 Metal memory hooks                    SKIP: no Metal device
LTX Metal memory hooks                   SKIP: no Metal device
git diff --check（本文档）                 PASS
```

这里的两个 Metal SKIP 只表示当前执行环境没有可用 Metal device，不是 L2 验收通过。production
registry 仍为空，不能因为 native build/逻辑测试通过而开放 constrained 执行。

## 162. 端到端时序与状态机（实现者必须保持的顺序）

### 162.1 请求时序

```text
parse JSON/profile
  -> validate Y/X/route/legacy conflict
  -> make_plan + normalize candidate
  -> metadata-only probe
  -> registry lookup (exact identity)
  -> clean session/cache boundary (enabled only)
  -> observe process baseline/system headroom
  -> compile exact live-interval plan
  -> admission(context + ledger + scheduler)
  -> bind context to session
  -> stage: reserve -> real allocate -> commit -> submit GPU
  -> safe point: poll completion -> release/retain cache
  -> next stage
  -> final synchronize/drain
  -> checkpoint + finish_success
  -> unbind context
```

任何一步失败都沿反向边执行 cleanup。特别是：

- registry miss 发生在 clean unload 之前，避免把失败请求当作一次 cache-mutating 操作；
- plan 编译发生在任何模型 tensor/GPU buffer 创建之前；
- `reserve` 必须早于真实 `malloc`、`MTLDevice newBuffer`、MLX materialize、MPSGraph compile；
- GPU backing 的 ledger release 必须晚于 completion event，而不是晚于 command encode；
- `finish_success` 之前必须为零的集合是 `reservation_count`、`pending_release_count`、
  `unknown_bytes`，不是只有 `active_bytes`。

### 162.2 状态转换表

```text
NEW
  -> VALIDATED              (schema/route/Y/X passed)
  -> PROBED                 (metadata-only probe returned exact manifest)
  -> AUTHORIZED             (registry + identity + plan passed)
  -> ADMITTED               (baseline/system headroom passed)
  -> RUNNING                (context bound; at least one stage active)
  -> DRAINING               (cancel/fault/final completion)
  -> SUCCEEDED              (all releases complete, tainted=false)
  -> FAILED                 (known clean failure)
  -> QUARANTINED            (completion/lifetime cannot be proven)
```

禁止的边：`VALIDATED -> RUNNING`、`PROBED -> RUNNING`、`FAILED -> RUNNING`、
`QUARANTINED -> RUNNING`。只有新请求和新 generation 才能重新开始。

### 162.3 每个阶段的最小事务

```cpp
MemoryExecutionContext &execution = *context;
MemoryStageLease stage = execution.scheduler().begin_stage("denoise");
auto txn = stage.reserve(MemoryClass::RefillSlot, upper, site_id);
if (!txn) return deny("memory_budget_too_small", site_id, upper);

RawBacking raw = allocator.allocate_after_reserve(upper);
if (!raw) { txn->cancel(); return fail("allocation_failed"); }

StorageId id = raw.storage_id(/*domain=*/domain, /*generation=*/generation);
auto lease = txn->commit(id, raw.actual_bytes());
stage.adopt(std::move(lease));

auto command = encode_and_submit_gpu(raw);
auto pending = stage.retire_all();
auto *scheduler = &execution.scheduler();
attach_completion(command, [ids = std::move(pending), scheduler]() mutable {
    for (auto id : ids)
        scheduler->complete(id, /*retain_as_cache=*/false);
});
```

`MemoryExecutionContext` 必须活到上述 callback 全部完成；若 deadline 前不能完成，就按第 161.5 节
进入 `QuarantineWorker`，不能让 callback 捕获悬空 scheduler。

真实实现可以用 C hook 包装上述逻辑，但语义不能变成“先 allocate 再询问 ledger”。任何无法
提供 `actual_bytes` 或 completion token 的 backing 都属于 `UnknownExternal`，在首发 constrained
路径必须拒绝，而不是用一个大概的上界蒙混过关。

## 163. 统一内存账本与 planner 的细节补充

### 163.1 四个不同的数字必须同时保留

实现和结果 JSON 不得把以下数字合并成一个 `memory_used`：

```text
Y                 用户输入上限
B                 floor(Y * (100-X) / 100)
planned_peak      manifest/live-interval 编译出的最坏同时存活上界
actual_peak       运行期间 ledger + process observation 的观测峰值
```

另外保留：`process_baseline`、`framework_upper`、`unknown_peak`、`pending_release_peak`、
`system_available_min`。推荐定义：

```text
planned_increment = planned_peak - process_baseline
allocation_ceiling = process_baseline + planned_increment = planned_peak
admission: planned_peak <= B
runtime guard: ledger_known + ledger_reserved <= allocation_ceiling <= B
system guard: system_available >= min_free + planned_increment
```

当前 `MemorySnapshot::known_bytes` 已经包含 `process_baseline_bytes`，因此 runtime guard 不能再次
加 baseline；否则会双重计费并错误拒绝所有接近预算的请求。

`planned_peak` 不应把 `B` 当作自身上界；它必须来自 manifest 的 live set。`actual_peak` 若高于
`planned_peak`，记录 `upper_violation`，立即 taint；不能事后把 `planned_peak` 改大后重新判定。

### 163.2 最小 Y 与 shape bucket 的计算

对于固定 candidate，若需要达到某个 `P=planned_peak`，仅从进程预算反推用户 Y：

```text
Y_min = ceil(P * 100 / (100-X))
```

如果系统 reserve 约束也不满足，还要单独检查：

```text
available_at_admission - min_free >= planned_increment
```

因此“把 Y 调到 `Y_min` 就一定成功”不成立；报告必须同时给出 B、process baseline 和系统
可用内存。shape bucket 选择只允许向上取整到 manifest 已认证 bucket，不能因为实际输入更小
就复用更小 bucket 的 upper。

### 163.3 live interval 编译的确定性规则

对所有 `AllocationInstance`：

1. 使用半开区间 `[acquire_epoch, release_epoch)`；同一 epoch 释放再获取不算重叠。
2. `release_epoch=UINT64_MAX` 表示请求结束前一直存活，不能被阶段边界隐式释放。
3. 相同 `alias_group` 只有在 `aliasable=true` 且 backing layout/dtype/device 完全相同时才折叠。
4. `asynchronous_release=true` 时，release epoch 必须晚于 completion epoch；不能以 CPU scope 结束
   代替 GPU last-use。
5. 峰值相同时按 `(epoch id, site_id, instance_id)` 排序，保证 plan digest 与输入枚举顺序无关。

planner 的 tie-break 固定后，`manifest_digest` 或 `plan_digest` 不能因为 C++ `unordered_map`
遍历顺序变化而改变，否则 capability evidence 无法重放。

### 163.4 allocator cache 和 framework temporary 的分层

第一版将 cache 分为三类：

| 类别 | 是否算进 B | 是否允许 retained | 处理规则 |
|---|---:|---:|---|
| 必需 backing（weights/activation/output） | 是 | 只在 manifest 明确允许时 | completion 后按 last-use 释放 |
| 可丢弃 cache（MLX allocator cache、prompt cache） | 是 | constrained 默认否 | safe point 可 trim；trim 不计为“峰值下降证明” |
| 未知 framework temporary/OS file cache | 不可用未知值放行 | 否 | 无 validated envelope 时 registry miss |

`mx::clear_cache()` 是释放可丢弃 cache 的实现动作，不是 planner 的容量来源；清理后仍无法
证明 graph temporary/allocator watermark 的 candidate 不能升级。

## 164. 模型级实施细节与 manifest 设计

### 164.1 H3 C/Metal 首发 candidate

首发继续限定为：`minimax-h3-turbo`、`video.generate`、GPU、C/Metal、text-to-video、无 audio、
无 input reference、无 LoRA、无 quantized-cache variant。首发沿用当前
`native/runtime/memory_policy.cpp` 已写入的 candidate id：

```text
h3_c_metal_streamed_v1
```

推荐 manifest stage 表：

| stage | 必需资源 | 主要上界来源 | last-use |
|---|---|---|---|
| `text_encode` | tokenizer host、text weights、embedding、attention workspace | checkpoint metadata + validated workspace envelope | text output copied into conditioning 后 |
| `conditioning` | joint video/audio latent metadata、schedule rows、gate-score readback | checked shape formula；schedule readback 已闭环 | DiT 首个 command buffer encode 后 |
| `denoise` | pinned blocks、refill slots、latent、activation、Euler previous velocity | block bytes + activation formula + slot count | 最后一步 GPU completion |
| `decode` | VAE weights、tile workspace、decoded host arena | VAE tile envelope；不能与 denoise weights 错算重叠 | output copy 完成后 |
| `export` | encoded media/output backing、temporary conversion | output dimensions/fps/audio contract | file write/consumer acknowledgement |

H3 latent 的上界只应使用 checked symbolic formula，不应写死某个当前 512×512 样本：

```text
latent_elements = frames * ceil(height / spatial_stride)
                  * ceil(width / spatial_stride) * latent_channels
latent_bytes = align_up(checked_mul(latent_elements, dtype_bytes), tensor_alignment)
```

`frames/height/width/steps` 必须来自已认证 shape bucket。任何 prompt tokenizer 的动态 host
vector、reference input、LoRA merge、audio path 都会改变 candidate key；未有独立 manifest 时
必须早拒绝。

H3 的 `h3_dit_schedule_gate_score()` readback 需注册为独立 site，例如：

```text
h3.conditioning.schedule_gate_score.readback
h3.conditioning.schedule_branch_score.readback
```

这样 schedule 读回失败时能定位具体 site，不会把失败归因成笼统的 activation overflow。

### 164.2 H3 代码修改顺序

建议按以下顺序提交，避免一次 patch 同时改算法和账本：

```text
H3-A  h3_host_options_from_params() + worker 按值复制 options
H3-B  schedule readback reserve/commit/cancel/release（当前已部分落地）
H3-C  text/conditioning/layout/tokenizer host site 注册
H3-D  Video VAE tile/workspace exact formula + completion release
H3-E  h3_session.mm metadata-only probe + candidate key
H3-F  H3 generate 全路径绑定 MemoryExecutionContext
H3-G  真实 Metal L2；随后才允许 capability record review
```

每个 patch 必须有一个“不可达证明”：如果某旧 allocation site 没有 hook，它要么被证明在
首发 request 组合不可达，要么必须把 candidate 留在 `plan_only`。禁止以“该分配很小”作为
漏记账理由，除非它小于统一 `guard_threshold_bytes` 且 manifest 明确记录小对象累计 envelope。

### 164.3 LTX C/Metal 首发 candidate

继续限定为：`ltx-2.5-distilled`、`video.generate`、GPU、C/Metal、text-to-video、video-only、
无 LoRA、无 audio、无 image-to-video。首发沿用当前
`native/runtime/memory_policy.cpp` 已写入的 candidate id：

```text
ltx_c_metal_streamed_video_v1
```

阶段表：

| stage | 资源 | 处理 |
|---|---|---|
| `text_encode` | Gemma/tokenizer host、text embedding | 统一 host registry；embedding 复制后释放 tokenizer scratch |
| `stage1_denoise` | Stage-1 block cache、latent、MPS/Metal activation | block slot bounded；每步完成后只保留必要 latent/velocity |
| `upsampler` | upsampler weights、temporary、video latent copy | 与 stage1 串行，除非 manifest 明确允许一槽 look-ahead |
| `stage2_denoise` | Stage-2 blocks、conditioning、latent | 使用独立 stage lease，不能复用 stage1 的 generation |
| `video_vae` | decoder weights、tile workspace、decoded host | `allow_quality_preserving_tiling=true` 时只使用已验证 tile bucket |
| `export` | output/codec backing | 输出完成后 release；不得把文件系统 page cache 当作已释放 |

LTX MLX 仍应与 C/Metal 分成不同 candidate。MLX cache、lazy array、MPSGraph temporary 无法
从 C/Metal manifest 继承 upper；在其 envelope 经过真实设备验证前，`TURBOCIDER_LTX_MLX=1`
的 constrained 请求必须返回 `memory_policy_unsupported`。

### 164.4 Flux 与 Z-Image 的 staged F1

Flux F1 只做 component-staged：

```text
text encoder -> transformer denoise -> VAE decode -> export
```

组件之间必须有显式 `retire_all()+completion drain`；不得在 transformer 仍有 command buffer
使用 embedding 时卸载 text encoder 的 backing。F1 期间不切 transformer block，避免首次发布
同时承担 MLX lazy graph、block pager、LoRA 和 tiling 四类变量。

Z-Image 必须按 variant 分开：

```text
z_image_bf16_staged_v1
z_image_gguf_staged_v1
```

GGUF 的 dequant scratch、per-channel metadata、mmap page fault 和 BF16 tensor 不能共用
同一 upper。LoRA、不同量化组大小、不同 runtime revision 均改变 candidate key。只有当 F1
在真实设备上证明 staged component 峰值后，才设计 layer-group pager F2。

### 164.5 预算驱动的 candidate/schedule 选择算法

模型 adapter 不应直接用 `B / block_bytes` 粗暴计算 pinned blocks。由于 clean baseline 只有在
unload/cache-clear 之后才知道，而 capability miss 又必须在 clean mutation 之前拒绝，完整选择应
分成 metadata authorization 和 clean-baseline selection 两段：

```text
Phase A（任何 session/cache mutation 前）
1. 过滤 request 语义不匹配的 route（backend/dtype/audio/input/LoRA/operation）。
2. metadata-only 构造所有允许的 exact manifest variant。
3. coordinator 用 candidate key + manifest digest 查询 registry，丢弃未认证 variant。
4. 若认证集合为空，立即拒绝；不 unload、不 clear cache、不分配模型 backing。

Phase B（clean boundary 和 baseline observation 后）
5. 对 Phase A 的每个已认证 variant 编译带真实 process baseline 的 live-interval plan。
6. 丢弃 planned_peak>B、system reserve 不满足或超过 validated upper 的 variant。
7. 从剩余集合选预计端到端 wall 最小方案；同分时选择更少 slot和更大剩余 headroom。
8. 若无方案，返回 minimum required Y/B 和 peak live set，不执行模型 allocation。
```

伪代码：

```cpp
struct AuthorizedVariant {
    MemoryCapabilityProbe probe;
    MemoryCapabilityRecord record;
    uint64_t framework_upper = 0;
};

std::vector<AuthorizedVariant> authorize_variants_before_mutation(
    const Request &request, const EffectiveMemoryPolicy &policy,
    const Metadata &metadata, const MemoryCapabilityRegistry &registry) {
    std::vector<AuthorizedVariant> result;
    for (auto &probe : build_exact_variants(request, policy, metadata)) {
        auto match = registry.lookup(
            probe.key, probe.manifest.digest(), /*experimental=*/false);
        if (match.matched)
            result.push_back({std::move(probe), *match.record});
    }
    require(!result.empty(), "memory_policy_unsupported: no exact variant");
    return result;
}

std::optional<AuthorizedSchedule> choose_after_clean_baseline(
    const std::vector<AuthorizedVariant> &authorized,
    const EffectiveMemoryPolicy &policy, uint64_t process_baseline) {
    std::optional<AuthorizedSchedule> best;
    for (const auto &variant : authorized) {
        auto plan = compile_memory_plan(
            variant.probe.manifest, process_baseline,
            variant.framework_upper, policy.effective_budget_bytes);
        if (plan.complete && plan.fits_budget &&
            plan.planned_peak_bytes <=
                variant.record.maximum_validated_upper_bytes)
            consider(best, variant, std::move(plan));
    }
    return best;
}
```

注意 singular `probe_memory_capability()` 的当前接口一次只返回一个 manifest。首发可以先让
normalization 根据 B/slot 选择唯一 revision，再由 probe 构造该 exact manifest；这会安全地
fail-closed，但在 clean baseline 较高时不能自动尝试第二方案。需要真正枚举多个认证方案时，新增
`probe_memory_capability_variants()` 让 session 只返回 metadata，registry 查询仍由 coordinator
完成。pressure 期间只能切换到 admission 时已经预授权且已编译 fit 的 serial/少 look-ahead
revision；不能临时生成一个未匹配 manifest 的新 schedule。

## 165. WeightPager/StageScheduler 的可实现并发算法

### 165.1 slot 状态机

每个 refill slot 只能按以下顺序转换：

```text
FREE(g)
 -> RESERVED(g+1)
 -> READING(g+1)
 -> READY(g+1)
 -> IN_FLIGHT(g+1)
 -> PENDING_RELEASE(g+1)
 -> FREE(g+1)
```

任何回调都必须携带 `(slot_id, generation, ticket_id)`。回调看到旧 generation 时只记录
`stale_completion` 并返回 false，不能释放当前 slot。`slot_id` 单独不足以防止 ABA。

### 165.2 look-ahead 选择

建议对候选 block 计算一个可解释分数，而不是只按“下一个 block”盲目预取：

```text
score = reuse_probability * expected_stall_saved
        - io_bytes * io_cost
        - slot_bytes * pressure_penalty
        - generation_risk
```

首发只允许 `reuse_probability=1` 的确定性顺序 block；不允许 speculative branch、跨 sampler
分支或未认证的动态 tile。`pressure_state!=Normal` 时分数只用于 optional prefetch，必需 slot
仍按 plan reservation 提交；无法拿到 permit 就阻塞在安全点或退化到已认证 serial revision。

### 165.3 overlap 的必要条件

只有同时满足以下条件，I/O 与 GPU compute 才能算“hidden overlap”：

1. 下一 slot 的 reservation 已成功并 commit；
2. 文件读取不使用未计账的临时 buffer；
3. 当前 GPU command buffer 与下一 slot 使用不同 backing/generation；
4. completion callback 能在下一次 reuse 前完成；
5. trace 中 `load_end <= next_gpu_wait_deadline`，且没有因 pressure 被强制串行化。

只要有一项不满足，指标应记为 `exposed_io_wait`，不能用平均 wall time 推断 overlap 成功。

### 165.4 无“第四槽”原则

`max_refill_slots=2` 的 candidate 永远最多有两个真实 backing。I/O queue 没有空槽时必须：

- 等待现有 slot completion；或
- 取消 optional prefetch；或
- 选择 manifest 已认证的 serial revision。

禁止临时 `malloc`、`NSData dataWithContentsOfFile`、MLX array copy 或 Objective-C autorelease
对象来偷建第四槽。若 framework 强制产生额外副本，必须将其加入 framework envelope；不能用
“通常很快释放”豁免。

## 166. 代码修改建议：按文件和符号拆成可 review patch

### 166.1 Runtime 层

| 文件 | 建议修改 | 关键断言 |
|---|---|---|
| `native/runtime/memory_accounting.hpp/cpp` | 增加 `MemorySiteToken`、generation 校验、unknown site 计数 | token 只能 commit/cancel/release 一次 |
| `native/runtime/memory_scheduler.hpp/cpp` | 增加 `reserve_slot()`、`on_completion()`、threshold revision | 无 permit 不得真实 allocate |
| `native/runtime/memory_execution.hpp/cpp` | context 绑定、failure disposition、schedule revision | success 时 reservation/pending/unknown 全为 0 |
| `native/runtime/memory_manifest.*` | manifest builder、canonical sort、shape bucket | digest 与枚举顺序无关 |
| `native/runtime/memory_plan.*` | stage overlap、alias fold、framework upper | unknown required site 拒绝 compile |
| `native/runtime/session.hpp` | context 过渡 API、probe contract | default implementation 不授予 capability |

建议的 token 结构：

```cpp
struct MemorySiteToken {
    uint64_t reservation_id = 0;
    uint64_t allocator_domain = 0;
    uint64_t generation = 0;
    std::string site_id;
    uint64_t upper_bytes = 0;
    bool committed = false;
    bool cancelled = false;
};
```

token 的 `site_id` 不能只存在于日志；它必须参与错误和 evidence，便于把“第几个槽位超限”
映射回源码注册点。

### 166.2 C/Objective-C bridge 层

| 文件 | 建议修改 | 兼容要求 |
|---|---|---|
| `native/api/c_api.mm` | 把 `ScopedMemoryAdmissionBinding` 升级为 context binding；统一 success/failure drain | disabled 请求不调用 probe、ledger、trace |
| `native/platform/apple/h3_session.mm` | `probe_memory_capability()`、context 到 h3 hooks 的 bridge | options 按请求快照复制 |
| `native/platform/apple/ltx_session.mm` | C/Metal candidate probe、direct buffer wrapper、stage lease | MLX route 不得复用 C/Metal record |
| `native/platform/apple/results.mm` | policy/trace/evidence 字段和错误 precedence | 新字段可选；旧客户端不崩 |
| `native/platform/apple/request.mm/profile.mm` | schema-v2 memory object 的来源记录 | profile overlay 不覆盖 request 显式值 |

C bridge 应保留当前 H3/LTX v1 opaque-token ABI，不做没有必要的破坏性重写：

```c
int reserve(void *user, uint32_t memory_class, uint64_t upper,
            const char *site_id, void **token,
            char *error, size_t error_size);
int commit(void *user, void *token, uint64_t allocator_domain,
           uint64_t handle, uint64_t actual, uint64_t generation,
           char *error, size_t error_size);
void cancel(void *user, void *token);
void release(void *user, void *token);
```

ABI 约束：`site_id` 在 reserve 时传入并复制；callback 不保存调用者栈指针；`commit` 失败后
调用者不得继续使用 backing；v1 `release` 只能在 GPU last-use 已确认后调用。若某 backend 必须
在 CPU owner 先结束时产生异步 pending token，应通过 `struct_size/version=2` 增加明确的
`retire/complete` ABI，或由 Objective-C completion closure 延迟调用 v1 `release`；不得偷偷改变
现有 v1 callback 的含义。

### 166.3 control-plane token arena

当前 H3/LTX bridge 的 reserve callback 会创建小型 C++ token 对象。完整 closure 不能假设这些
对象永远“足够小”而忽略。建议在 `MemoryExecutionContext` admission 后建立有界 token arena：

```text
capacity = max_live_reservation_count(manifest.instances)
bytes = align_up(capacity * sizeof(BackendMemoryToken), arena_alignment)
site = runtime.control_plane.token_arena
lifetime = Request
```

arena 自身计入 `framework_upper` 或 manifest；allocator callback 只从 freelist 取 token，不再调用
普通 heap。arena 耗尽是确定性的 `memory_lifetime_violation`，不能退回 `new`。同时把 trace ring、
pending-id vector 和 callback closure storage 放入同一 control-plane envelope。因为当前 baseline
是在 context 构造前采样，这些后续 control-plane 分配必须显式计入 planned increment。

### 166.4 模型 native 层

H3：

```text
native/models/h3_runtime/h3.c
  - host_options_from_params()
  - reserve_host / commit_host / release_host site labels
  - schedule readback site labels
  - worker options by-value copy

native/models/h3_runtime/h3_gpu.m
  - every new MTLBuffer path -> h3_gpu_allocate_classified()
  - completion handler carries domain/handle/generation

native/models/h3_runtime/h3_video_vae.c
  - tile workspace upper formula
  - decoded host arena actual-size commit
```

LTX：

```text
native/models/ltx_runtime/ltx_gpu.m
  - one direct buffer allocator wrapper
  - actual allocated length commit
  - completion handler -> pending release

native/models/ltx_runtime/ltx_blocks.c
  - slot/generation/ticket propagation
  - no load into unreserved temporary

native/platform/apple/ltx_session.mm
  - Stage1/upsampler/Stage2 lease table
  - host vector capacity accounting
  - MLX/MPSGraph route rejection until separately certified
```

### 166.5 `plan.cpp` 与旧字段兼容

`Request.memory_budget_bytes` 仍是旧 denoiser hint。enabled constrained 请求在 normalization
后可以把它设置成 `denoiser_budget_bytes`，但结果必须同时保留：

```text
request.memory_constrained.limit_bytes = Y
policy.effective_budget_bytes = B
request.memory_budget_bytes = denoiser sub-budget (compatibility only)
```

任何新代码都不得使用 `request.memory_budget_bytes` 重新推导全请求 B；全请求合同只从
`EffectiveMemoryPolicy` 读取。`memory_constrained.normalized=true` 只用于避免重复 make_plan，
不允许 parser 接受该字段作为用户输入。

## 167. 可观测性、错误优先级和运维接口

### 167.1 结果 JSON 最小字段

建议在现有 `memory_policy` 下新增：

```json
{
  "schedule_revision": 3,
  "context_generation": 42,
  "planned_increment_bytes": 123,
  "actual_peak_bytes": 456,
  "peak_unknown_bytes": 0,
  "pending_release_peak": 2,
  "prefetch_submitted": 17,
  "prefetch_cancelled": 3,
  "exposed_io_wait_seconds": 0.12,
  "hidden_io_fraction": 0.81,
  "resident_fallback": false,
  "worker_quarantined": false,
  "swapins_delta": 0,
  "swapouts_delta": 0,
  "system_available_min_bytes": 987,
  "failure_disposition": "clean"
}
```

字段含义必须稳定：分别保留 `process_peak_bytes`、`ledger_peak_known_bytes` 和
`ledger_peak_reserved_bytes`；若兼容层必须输出单一 `actual_peak_bytes`，定义为前两者的最大值，
不能相加重复计费，也不能只写 `mx::get_active_memory()`。`swapouts_delta` 来自前后 `vm_stat`
单调计数，而不是 swap 文件净大小。

### 167.2 错误 precedence

若一个请求同时发生多个错误，按以下优先级报告，并在 trace 保留其余事件：

```text
memory_lifetime_violation
  > memory_pressure_abort
  > memory_budget_too_small
  > memory_admission_conflict
  > memory_policy_unsupported
  > memory_estimate_unknown
  > memory_policy_invalid
```

例如 allocation 越线后又因 cancel 触发，主错误仍是 `memory_lifetime_violation`，不能降级成
“用户取消”。错误信息包含 site、upper、actual、B、plan digest 和 candidate id；禁止 prompt、
绝对路径、LoRA 私有路径和完整环境变量。

### 167.3 quarantine 运维动作

worker 被 quarantine 后：

1. 立即停止接受新请求；
2. 若 parent MLX，外层尝试 synchronize，但不得在 allocator callback 内同步；
3. 关闭模型 session，清理可确认的 cache；
4. 将 generation、pending token、最后 trace 事件和错误摘要写入 evidence；
5. 新 worker 必须使用新 generation，不能复用旧 `StorageId`。

quarantine 不是自动重试。服务层可以由上层策略在新 worker、同一 candidate、同一预算下重试一次；
若再次失败，返回错误并保留两次 evidence。不得因为重试成功就删除第一次越线记录。

## 168. 验收方案增补：测试向量和可重复实验

### 168.1 单元测试向量

必须加入固定向量（所有 bytes 采用十进制或明确二进制单位，不能混用）：

| 向量 | 输入 | 预期 |
|---|---|---|
| P-01 | Y=16GiB, X=15% | B=14,602,888,806 bytes；按 floor 结果精确断言 |
| P-02 | Y=1, X=30% | B=0，拒绝 |
| P-03 | limit=2^50, X=5% | 不溢出，B 正确 |
| P-04 | `minimum-1` | 在任何 allocator 前拒绝 |
| P-05 | alias 两个相同 backing | 只计一次；不同 generation 不折叠 |
| P-06 | stale completion | 不释放新 generation；返回 false |
| P-07 | reserve 成功、allocate 失败 | reservation cancel，unknown=0 |
| P-08 | commit actual>upper | 主错误 lifetime violation，context tainted |
| P-09 | critical 后恢复 | pressure 仍 sticky，不能改成成功 |

P-01 的具体 B 必须由测试用 calculator/常量生成，不能在文档中手抄近似值。测试同时覆盖
`uint64_t` 边界和 `checked_multiply/checked_add` 溢出。

### 168.2 probe purity 测试

对 H3/LTX probe 运行前后记录：

```text
MTLBuffer create count
MLX active bytes
MLX cache clear count
graph compile count
file descriptor count
process footprint
```

要求除读取 checkpoint metadata/index 和 hash 所需的文件读取外，所有 materialization/compile/
GPU allocation/cache mutation 计数为零。probe 返回的 manifest digest 在两次独立进程中一致。

### 168.3 fault sequence 测试

用确定性 `(site_id, ordinal)` 注入，不从普通环境变量读取随机 fault：

```text
F-01 first weight reserve denied
F-02 second refill commit fails
F-03 I/O read short
F-04 GPU completion delayed beyond timeout
F-05 cancel between encode and submit
F-06 stale completion after slot reuse
F-07 host output actual bytes exceeds upper
F-08 pressure critical at stage boundary
```

每个序列都要检查：reservation_count、pending_count、unknown_bytes、session generation、
worker disposition 和错误 precedence。release build 不开放随机 fault 环境变量。

### 168.4 disabled ABBA 的强断言

除现有 wall/quality 比较外，建议在测试 build 增加计数器：

```text
memory_probe_calls == 0
memory_reserve_calls == 0
memory_commit_calls == 0
memory_release_calls == 0
memory_trace_events == 0
memory_cache_clear_extra_calls == 0
```

在 disabled A/B/B/A 中，模型 route、residency、cache key、worker 拓扑和 output digest 必须一致。
若编译器引入了未执行的符号引用，不算行为变化；运行时计数必须为零。

### 168.5 L2/L3 campaign 的停止规则

以下任一事件出现，立即停止当前 candidate campaign，并回到 `plan_only`：

- 任何 required site unknown；
- actual>upper 一次；
- 发生 resident/unbounded fallback；
- pending completion 在最终 drain 后非零；
- swapouts_delta>0（安静专用机）；
- output/quality contract 失败；
- trace 丢失 error event 或 evidence digest 无法重算。

不要继续跑更多样本来“稀释”失败。修复后新建 evidence digest；旧失败记录保留并关联
`superseded_by`。

## 169. 低内存本机实验脚本设计

### 169.1 runner 输入

建议增加 `tools/native/memory_constrained_campaign.py`（名称可调整），输入只引用已存在的
checkpoint root 和脱敏 request：

```text
--candidate h3_c_metal_streamed_v1
--request request.redacted.json
--budget-bytes Y
--buffer-percent X
--repetitions 5
--cold-start | --warm-start
--fault-sequence F-03
--output evidence/<campaign-id>
```

runner 不负责修改 production registry。它只能生成 `candidate_record.proposed.json`，由人工 review
后复制到受控 registry source。runner 启动前检查 git SHA、native binary hash、MLX/runtime revision、
device family 和 OS build；任一缺失就标记 `inconclusive`。

### 169.2 采样边界

所有 candidate 至少采样：

```text
before_parse
after_probe
after_clean_boundary
after_admission
each stage load end
first/middle/last denoise step
each tile group
before export / after export
after final drain
```

高频 sampler 只做诊断；硬判定依赖 reserve/commit guard 和边界 observation。采样线程不能持有
ledger mutex 等待 GPU，也不能调用 cache clear。

### 169.3 结果归档和脱敏

evidence 中允许保留：shape、seed、steps、dtype、candidate、digest、timing、allocation site。
必须脱敏：prompt、绝对路径、用户名、LoRA 文件名、任意原始 token。模型权重不进入 bundle，
只记录 checkpoint digest 和 sidecar manifest digest。

## 170. 实施顺序、回滚和签核模板

### 170.1 建议的增量提交

```text
PR-161A  context binding compatibility bridge + disabled counters
PR-161B  failure disposition / bounded drain / quarantine
PR-162A  H3 metadata-only probe + manifest builder
PR-162B  H3 reachable host/GPU closure
PR-163A  LTX C/Metal exact probe + direct buffer closure
PR-163B  LTX stage lease/vector/MPSGraph envelope
PR-164A  pager ticket/generation/serial fallback
PR-164B  trace/evidence runner
PR-165A  H3 L2/L3 evidence review
PR-165B  LTX L2/L3 evidence review
PR-166A  Flux staged F1
PR-166B  Z-Image BF16/GGUF staged F1
```

每个 PR 必须可独立回滚。registry 变更与 runtime 代码变更分离提交：先让代码支持并保持
`release_enabled=false`，再由 evidence review PR 打开 candidate。发现越线时只回滚 registry
record/flag，不回滚默认 resident/streamed 行为，也不删除失败 evidence。

### 170.2 PR 签核模板

```text
Candidate id:
Request/shape bucket:
Checkpoint/device/runtime identity:
Changed files and allocation sites:
New exact/envelope formulas:
Unknown required sites before/after:
Reservation-before-allocation proof:
Completion/pending-release proof:
Failure disposition proof:
Disabled ABBA result:
L0/L1/L2/L3 commands and outputs:
Swap/quality/performance evidence digest:
Registry change: none / proposed / release-enabled
Rollback owner and procedure:
```

### 170.3 Go/No-Go 的实现者版检查

```text
[ ] 当前 HEAD build green，且不是引用旧 binary
[ ] probe metadata-only；无 GPU allocation/cache mutation
[ ] exact manifest/candidate digest 可重放
[ ] required reachable site unknown=0
[ ] planned_peak 来自 live interval，不是 heuristic
[ ] reserve 在真实 allocate 之前
[ ] commit actual <= upper；StorageId 含 domain+generation
[ ] GPU completion 后才 release；pending 最终为 0
[ ] cancel/fault 进入 clean 或 quarantine，不伪造完成
[ ] pressure critical sticky；无 resident/unbounded fallback
[ ] disabled ABBA 运行时 memory counters 全为 0
[ ] 真实 Metal 设备 L2/L3 通过；无设备只能 SKIP
[ ] 安静低内存机 swapouts_delta=0
[ ] quality/output/media contract 通过
[ ] registry 仅由 review PR 开启，且可回滚
```

## 171. 仍需显式决策的事项

以下事项不能由实现者自行“顺手决定”，需要在 PR 或产品评审中记录：

1. **系统 reserve 的默认值。** 目前默认 1 GiB 只是实验起点；不同 macOS/内存容量可能需要
   分档。改变默认值会改变 policy digest 和最低 Y，应使旧 evidence 失效。
2. **allocator cache 是否纳入 candidate upper。** 如果只能通过 `clear_cache` 降低峰值，必须把
   清理动作和时序写入 candidate，不得把 cache 视为永远免费。
3. **framework envelope 的有效期。** MLX、Metal、MPSGraph 或 OS build 改变时，默认使 record
   失效；只有有对照 campaign 才能延长有效期。
4. **低内存实机的目标档位。** 至少选择一个 16/24/32 GiB 档位作为 L3 目标；64 GiB 机器的
   小 Y 逻辑测试不能替代真实物理内存压力测试。
5. **质量保持型 tiling 的边界。** 只能接受已有 checksum/感知质量证据的 VAE/export tile；
   不得在没有质量合同的情况下切分全局 attention 或隐式改变采样语义。
6. **并发请求。** 首发继续单 GPU request；若产品需要并发，先设计 broker 和 shared-backing
   计费，再修改全局执行锁。

在这些决策明确前，正确的默认行为都是保守拒绝，而不是利用 swap、环境变量或 synthetic
capability record 让请求“先跑起来”。

## 172. 本轮追加内容的验收结论

本轮文档补充的目标是让实现团队能够从当前源码继续施工，而不是宣称新增生产能力。交付判断保持：

- 当前已落地的 runtime scaffolding、H3 schedule readback、部分 H3/LTX hooks 和纯逻辑测试可继续
  作为基础；
- H3/LTX metadata-only probe 接线和 fixture 已落地；完整 key equality、checkpoint trust/TOCTOU、allocation closure、真实 Metal L2/L3 和 production registry 仍是硬门；
- Flux/Z-Image 先按 staged F1 独立认证，不能继承 H3/LTX record；
- enabled=false 必须保持旧路径零 probe/zero ledger/zero trace；
- swap 仍只作为系统外部观测指标，不能成为 TurboCider 的容量扩展机制。

在 H3/LTX 至少各有一条真实 GPU `l3_certified` candidate、完整 evidence bundle 可重放、低内存
实机 `swapouts_delta==0` 且默认 ABBA 稳态回归不超过 2% 之前，文档顶部状态继续保持
`Proposal + scaffolding`，不升级为“已支持内存受限模式”。

## 173. 最新源码差距与下一条最短可交付路径

本节以后以当前工作树为准。当前已经有请求级 `MemoryExecutionContext`、context/admission 双绑定、
failure disposition、manifest builder、checkpoint identity digest，以及已进入主构建并接入 H3/LTX 的
`native/platform/apple/memory_probe.{hpp,mm}`。因此下一步不应继续增加平行抽象，而应先完成下面这条
最短闭环：

```text
sidecar fixture
  -> metadata-only load/validation
  -> H3/LTX ModelSession::probe_memory_capability()
  -> production registry lookup（当前仍故意 miss）
  -> no-mutation fail-closed
  -> probe purity/identity/error tests
```

### 173.1 当前文件级状态

| 文件/能力 | 当前状态 | 下一步 | 未完成时的正确行为 |
|---|---|---|---|
| `memory_execution.*` | context、binding、failure disposition、成功前 completion drain 已存在 | 增加真实异步 backend drain evidence | 失败后保守 quarantine |
| `memory_manifest.*` | builder、digest、checkpoint identity 已存在 | 增加 schema limits、epoch/公式版本 | registry 不得放行 |
| `memory_probe.*` | 已编译；sidecar/checkpoint/manifest 基础校验和 fixture 已通过 | 完整 key equality、TOCTOU trust policy、字段 hard limits、purity counters | probe 返回空或抛 typed error |
| `h3_session.mm` | admission/context hooks 与 metadata-only probe override 已绑定 | 构造并校验 H3 expected key；补真实 manifest generator | registry miss 时 constrained 拒绝 |
| `ltx_session.mm` | admission/context hooks 与 C/Metal probe override 已绑定 | 固化 expected key；MLX route 继续隔离 | MLX constrained 拒绝 |
| `tools/native/build.sh` | 已编译 `memory_probe.mm` | 用 contract test 防止接线回归 | 不得借 probe 接线宣称 capability 已认证 |
| production capability registry | 空 | 继续保持空，等待真实 evidence review | 所有 constrained 执行 fail-closed |

### 173.2 依赖顺序

```text
P0  probe 能编译并通过纯 metadata 单测
 |
 +--> P1  H3/LTX session 接入，真实 model root 上 registry miss 但无副作用
       |
       +--> P2  allocation-site closure + exact manifest generator
             |
             +--> P3  真实 Metal L2/L3 evidence
                   |
                   +--> P4  review 后增加 release record
```

P0/P1 已在当前工作树完成并通过 build/fixture 测试；P2 可以用 synthetic allocator 和小 checkpoint 验证大部分事务；
只有 P3/P4 需要真实目标机器。不得反转 P3/P4：先写 release record 再补 evidence 会把验证机制变成
自我授权。

### 173.3 本轮 Definition of Done

本轮若只完成 probe 接线，交付措辞只能是：

```text
metadata-only probe path integrated and fail-closed;
no production candidate enabled.
```

不能写“低内存模式可用”。至少还要满足：required allocation site closure、completion drain、真实设备
峰值和 swap evidence，才进入下一成熟度。

## 174. `memory_probe` sidecar 的可执行规格与安全边界

### 174.1 v1 完整示例

首版 sidecar 固定放在：

```text
<model-root>/memory-capabilities/<adapter>/<shape-key>_slots<N>.json
```

建议内容如下；所有省略字段都视为错误，不设置隐式默认值：

```json
{
  "schema": "turbocider.memory_probe.v1",
  "key": {
    "adapter": "h3_c_metal_streamed_v1",
    "model_id": "minimax-h3-turbo",
    "checkpoint_digest": "<64-lowercase-hex>",
    "backend": "c_metal",
    "dtype": "bf16",
    "model_variant": "fl2va_t2v_video_only",
    "operation": "video.generate",
    "shape_bucket": "w512_h512_f22_s4_audio0_input0_lora0",
    "sampler_mode": "h3_distilled_euler_4step",
    "refill_slots": 2,
    "tiling_mode": "vae_tile_v1",
    "runtime_revision": "memory-runtime-v1",
    "device_family": "Apple-M4-Max"
  },
  "checkpoint_files": [
    {
      "logical_name": "FL2VA/transformer/model-00001-of-00013.safetensors",
      "size_bytes": 123456789,
      "sha256": "<64-lowercase-hex>"
    }
  ],
  "manifest": {
    "schema": "turbocider.memory_manifest.v1",
    "candidate_id": "h3_c_metal_streamed_v1",
    "checkpoint_digest": "<same canonical digest>",
    "backend_revision": "h3-c-metal-allocator-v1",
    "runtime_revision": "memory-runtime-v1",
    "sites": [
      {
        "site_id": "h3.dit.refill_slot.bf16",
        "component": "dit",
        "stage": "denoise",
        "memory_class": "refill_slot",
        "lifetime": "block",
        "provenance": "exact_metadata",
        "required": true,
        "asynchronous": true,
        "aliasable": false,
        "guard_threshold_bytes": 0
      }
    ],
    "instances": [
      {
        "site_id": "h3.dit.refill_slot.bf16",
        "instance_id": 0,
        "upper_bytes": 123456,
        "live_begin": 100,
        "live_end": 180,
        "alias_group": 0,
        "last_use_event": "dit.block.0.complete"
      }
    ]
  }
}
```

sidecar 是“模型准备工具产生的不可变描述”，不是运行时 profile，也不是用户可调配置。用户不能通过
request 覆盖其中任何 identity、upper、lifetime 或 device 字段。

### 174.2 严格解析规则

loader 必须一次性执行以下检查，任一失败都在 session unload/cache clear 之前拒绝：

1. root、`key`、`manifest` 是 object；文件/site/instance 是 array；拒绝 `null` 替代空集合。
2. 所有枚举只接受已知值；required site 不接受 `heuristic_not_executable`。
3. SHA-256 必须为 64 个小写十六进制字符；不自动 lower-case。
4. `refill_slots` 必须在 `[1,3]` 且等于 effective policy；`instance_id` 必须能放入 `uint32_t`。
5. v1 数值必须是不超过 `2^53` 的非负 JSON integer。v2 若要覆盖更大整数，改用规范十进制字符串，
   不要让 Foundation `NSNumber` 经 `double` 丢精度。
6. logical checkpoint name、site id、instance id、logical file 必须唯一。
7. `live_begin < live_end`；`last_use_event` 非空；asynchronous site 必须有可关联的 completion event。
8. 拒绝 sidecar 中未知的安全相关字段；允许扩展时必须通过显式 `extensions` namespace，而不是默默忽略
   拼写错误的 `required`/`upper_bytes`。

建议把严格字段检查集中到 helper，避免 H3/LTX 各写一份宽松 parser：

```cpp
void require_exact_keys(NSDictionary *object,
                        std::initializer_list<NSString *> required,
                        std::initializer_list<NSString *> optional,
                        std::string_view context);
```

### 174.3 路径与符号链接规则

当前 checkpoint logical path 已按 model root 解析；还应补充对 sidecar 自身的独立验证。即使 session
通常由安全 helper 构造路径，loader 也不能假设所有未来调用者都正确：

```cpp
ResolvedModelFile resolve_model_file(
    const std::filesystem::path &model_root,
    const std::filesystem::path &logical_or_candidate,
    FileKind kind);
```

要求：

- model root 先 canonicalize，并确认是 directory；
- sidecar 和 checkpoint 的最终 canonical path 都必须位于 root 下；
- logical path 拒绝绝对路径、`.`、`..`、NUL、空 component；
- production 可选择拒绝所有 symlink，或允许 symlink 但最终目标必须仍在 root 下；策略必须进入
  runtime revision；
- 不在错误信息中输出完整绝对路径，只输出脱敏 logical name；
- `memory-capabilities` 目录不可写时不影响读取；runtime 不尝试现场生成或修复 sidecar。

`path_has_prefix()` 必须按 path component 比较；不能用字符串前缀，否则 `/model-a` 会错误匹配
`/model-ab`。

### 174.4 TOCTOU 与 hash cache

仅比较 `size + mtime` 的 hash cache 适用于“由应用拥有且推理期间不可变”的模型根目录；对外部可写
目录不足以构成内容证明。建议引入：

```cpp
enum class ModelRootTrust {
    OwnedImmutable,
    ExternalMutable,
};

struct FileSnapshot {
    uint64_t device;
    uint64_t inode;
    uint64_t size;
    timespec change_time;
    timespec modify_time;
};
```

校验流程应使用打开后的 fd：

```text
open(O_RDONLY | O_CLOEXEC [| O_NOFOLLOW])
 -> fstat(before)
 -> stream SHA-256 from same fd
 -> fstat(after)
 -> require snapshots equal
 -> close
```

`OwnedImmutable` 可以缓存 `(device,inode,size,ctime,mtime)->digest`；`ExternalMutable` 每次 constrained
请求重算，或先复制/导入到应用管理的 immutable store。mtime 只能是 cache hint，不是 trust anchor。
sidecar 自身也需要 content digest，最终写入 evidence；否则 sidecar 在 probe 与 registry lookup 之间被
替换时难以定位。

### 174.5 checkpoint identity 的边界

checkpoint 文件清单必须覆盖会影响执行结果或 allocation shape 的所有 artifact：

```text
transformer/text encoder/VAE weights
config and tensor index
tokenizer model（若会改变 token shape）
quantization/dequant metadata
conversion or merge provenance manifest
```

shader binary、TurboCider 代码和 runtime 行为由 `backend_revision/runtime_revision` 绑定，不必伪装成
checkpoint file。LoRA 不允许沿用 base digest；首发直接拒绝 LoRA，后续 key 使用：

```text
effective_checkpoint_digest = H(base_digest | lora_digest | strength_bits |
                                merge_revision | target_projection_set)
```

## 175. H3/LTX probe 接线的逐文件代码修改建议

### 175.1 公共 helper API 收敛

当前 loader 接收 `(plan, device)` 并从 sidecar 读 `MemoryCandidateKey`。为了确保 sidecar 不能声明一个
session 实际不会执行的 backend/dtype/sampler，建议由 adapter 先生成 expected key，再做完整相等比较：

```cpp
std::optional<MemoryCapabilityProbe> load_memory_capability_probe(
    const std::filesystem::path &model_root,
    const std::filesystem::path &sidecar,
    const MemoryCandidateKey &expected,
    ModelRootTrust trust,
    const MemoryCheckpointHashCache &hash_cache);
```

最低限度也必须在现有函数内检查全部 key 字段，而不只是 adapter/model/operation/shape/slot/runtime/device：

```text
backend
dtype
model_variant
sampler_mode
tiling_mode
checkpoint_digest（由文件清单重算后比较）
```

shape key 只能有一个 canonical helper。sidecar 文件名、`MemoryCandidateKey::shape_bucket`、测试 fixture、
registry record 都调用同一个函数，避免 `_input0` 与 `.input-none` 这类表示漂移。

### 175.2 H3 session

在 `native/platform/apple/h3_session.mm`：

```cpp
#include "memory_probe.hpp"

class H3Session final : public tc::ModelSession {
    // ...
    mutable tc::MemoryCheckpointHashCache memory_probe_hash_cache_;

    std::optional<tc::MemoryCapabilityProbe> probe_memory_capability(
        const tc::ExecutionPlan &plan,
        const tc::MemoryDeviceIdentity &device) const override {
        if (!plan.memory_policy || !plan.memory_policy->enabled)
            return std::nullopt;
        tc::require(plan.memory_policy->adapter_candidate ==
                        "h3_c_metal_streamed_v1",
                    "memory_policy_unsupported: H3 probe adapter mismatch");
        const auto sidecar = tc::memory_capability_probe_path(
            root_, plan.memory_policy->adapter_candidate, plan.request,
            plan.memory_policy->refill_slots);
        return tc::load_memory_capability_probe(
            root_, sidecar, plan, device, memory_probe_hash_cache_);
    }
};
```

进一步要求：

- probe 不得调用 `read_json()` 后顺带创建 H3 context；
- 读取 merge provenance 可以，但不能调用 `h3_load()`、`h3_gpu_create()` 或任何 warmup；
- expected variant 应固定 `FL2VA` text-to-video/video-only，不接受 Ref2VA、audio、LoRA 或 quantized cache；
- H3 当前只允许已认证双槽；如果 policy 是一槽/三槽，不应寻找名字相似的 sidecar；
- hash cache 属于 session，但不持有 fd、mapped weights 或 Objective-C GPU object。

### 175.3 LTX session

在 `native/platform/apple/ltx_session.mm` 同样包含 helper，并增加 cache 与 override：

```cpp
mutable MemoryCheckpointHashCache memory_probe_hash_cache_;

std::optional<MemoryCapabilityProbe> probe_memory_capability(
    const ExecutionPlan &plan,
    const MemoryDeviceIdentity &device) const override {
    if (!plan.memory_policy || !plan.memory_policy->enabled)
        return std::nullopt;
    require(plan.memory_policy->adapter_candidate ==
                "ltx_c_metal_streamed_video_v1" &&
                plan.request.ltx_backend == "c_metal",
            "memory_policy_unsupported: LTX probe route mismatch");
    const auto sidecar = memory_capability_probe_path(
        root_, plan.memory_policy->adapter_candidate, plan.request,
        plan.memory_policy->refill_slots);
    return load_memory_capability_probe(
        root_, sidecar, plan, device, memory_probe_hash_cache_);
}
```

LTX 文件清单至少覆盖 transformer、Gemma、upsampler、video VAE 及影响 shape 的配置。helper executable
本身不放进 checkpoint digest，但 helper binary hash/runtime revision 必须进入 evidence 和
`backend_revision`。C/Metal sidecar 绝不能授权 MLX route；`uses_parent_mlx()` 的结果、key.backend 和
request backend 必须三者一致。

### 175.4 build 与链接

下列源文件已经加入 `tools/native/build.sh` 的 `SOURCES`，后续由 contract test 防止接线回归：

```text
native/platform/apple/memory_probe.mm
```

它使用 Foundation 和 CommonCrypto；主 dylib 已链接对应平台 framework 时不应增加新的 runtime
依赖。构建必须开启现有 `-Wall -Wextra`，probe 专项测试使用 `-Werror`。若 CommonCrypto 的旧 C API
产生 deprecation warning，优先使用项目部署目标支持的稳定 digest API，并把实现 revision 记入代码，
不要全局关闭 warning。

### 175.5 contract test

`tests/native/test_contract.py` 增加静态接线断言：

```text
build.sh 包含 memory_probe.mm
H3/LTX include memory_probe.hpp
H3/LTX override probe_memory_capability
两者都有独立 MemoryCheckpointHashCache
disabled 分支在 path/hash 前返回
production registry 仍为空
```

静态断言不能替代行为测试，但能防止后续重构把源文件从 build 或 session vtable 中意外移除。

## 176. 从 probe 到执行结束的精确生命周期

### 176.1 成功路径顺序

当前 `prepare_memory_execution()` 的“先 probe、后 unload”的方向正确；建议最终固定为：

```text
1  parse + validate request
2  make_plan/normalize route
3  if disabled: 直接旧路径，后续 4–16 全不执行
4  metadata-only probe + checkpoint identity
5  immutable registry lookup
6  若 miss：立即返回，session/cache/worker 不发生 mutation
7  synchronize existing parent runtime（如适用）
8  session.unload + constrained-specific cache clean boundary
9  observe clean process baseline
10 compile/authorize exact live-interval plan
11 create MemoryExecutionContext/control-plane arena
12 bind admission，再 bind context
13 execute stages and allocation transactions
14 wait/drain all GPU completion tokens
15 final checkpoint + finish_success
16 unbind context/admission，销毁 context
```

第 14 步必须位于 `finish_success()` 之前。仅在函数返回时由局部 `Drain` 析构同步是不够的，因为
`finish_success()` 已经检查 pending token；更不能先宣告成功，再由析构发现 completion failure。

### 176.2 建议新增的 session completion API

不同 backend 的同步原语不同，不能让 `MemoryExecutionContext` 直接依赖 Metal/MLX。建议：

```cpp
struct MemoryDrainResult {
    bool completed = false;
    uint64_t completions = 0;
    uint64_t pending_after = 0;
    std::string failure;
};

class ModelSession {
  public:
    virtual MemoryDrainResult drain_memory_completions(
        MemoryExecutionContext &, std::chrono::milliseconds timeout) {
        return {true, 0, 0, {}};
    }
};
```

默认实现只适用于没有异步受控 backing 的 legacy adapter；声明 asynchronous manifest site 的 adapter
必须 override。H3 用已提交 command buffer/event，LTX 用自身 command queue/helper completion，MLX
route 用 `mx::synchronize()`，但不得在 allocator callback 内调用 drain。

API 层伪代码：

```cpp
auto result = session.generate(request, event, cancelled);
if (memory_execution) {
    auto drained = session.drain_memory_completions(
        *memory_execution, policy.completion_timeout);
    require(drained.completed && drained.pending_after == 0,
            "memory_lifetime_violation: GPU completion drain failed");
    memory_execution->checkpoint("complete");
    memory_execution->finish_success();
}
```

### 176.3 失败路径与 disposition 消费

`finalize_failure()` 的返回值不能只写入 metrics 后丢弃。API/service 应执行：

| disposition | 动作 | 是否可复用 session/worker |
|---|---|---:|
| `Clean` | 正常 unbind，保留失败诊断 | 是 |
| `NeedsGpuDrain` | bounded drain；成功后重新评估 ledger | 仅在 drain 后完全 clean 时 |
| `QuarantineWorker` | 标记 engine unavailable、unload 可确认资源、由 owner 重建 | 否 |

建议 `tc_engine` 增加：

```cpp
std::atomic<bool> quarantined{false};
std::string quarantine_reason; // 受 mutex 保护，已脱敏
```

新请求在 parse/model mutation 前检查 `quarantined`。`tc_engine_free()` 仍允许释放 quarantined engine；
`tc_engine_generate()` 不自动悄悄重建，因为这会隐藏第一次 lifetime violation。daemon 可以在进程级
policy 下显式创建新 worker，并把两次 attempt id 都写进结果。

### 176.4 control-plane 内存放置

baseline 在 context 创建前采样，因此下列 context 自身开销必须计入 `framework_upper` 或 manifest：

```text
MemoryLedger maps/vectors
reservation token arena
pending completion table
trace ring
callback closure storage
stage/slot metadata
```

首发不要求逐个 `std::string` 单独 hook，但要求有经过测试的固定 upper，并禁止运行中无界增长。
所有 tag/site string 应引用 intern table；trace 使用有界 ring，溢出时 candidate 失败或只丢弃非关键
sample，绝不能丢失 allocation/error/completion 事件。

## 177. 多候选选择、降级图与 schedule revision

### 177.1 候选不是一个布尔开关

同一个模型/shape 可以有多个独立认证的候选，例如：

```text
H3-S2-P8   双槽 + pinned 8 blocks
H3-S2-P4   双槽 + pinned 4 blocks
H3-S1-P0   单槽串行 + 无 pinned suffix（需独立执行器/evidence）
H3-S2-P4-T VAE tile_v1（质量保持，独立 manifest）
```

每个候选有独立 key、manifest digest、maximum upper 和 performance evidence；不能只修改 planner
数字就把 S2 candidate 当作 S1 执行。

建议接口从 singular probe 演进为：

```cpp
struct MemoryCandidateDescriptor {
    MemoryCandidateKey key;
    MemoryManifest manifest;
    CandidateCost cost;
    ScheduleRevision revision;
};

virtual std::vector<MemoryCandidateDescriptor>
probe_memory_capability_variants(const ExecutionPlan &,
                                 const MemoryDeviceIdentity &) const;
```

首版保留 singular wrapper，内部返回一个元素；待多个真实 candidate 存在后再启用枚举，避免提前增加
无证据分支。

### 177.2 选择目标函数

只有通过 registry、预算和 system reserve 的候选才参加排序：

```text
feasible(c) = registry_match(c)
           && planned_peak(c, clean_baseline) <= B
           && planned_peak(c) <= validated_upper(c)
           && system_available - min_free >= planned_increment(c)
```

排序建议使用字典序，而不是难以解释的单一魔法分数：

```text
1. quality_contract_id（必须与请求一致；不同质量不互相排序）
2. predicted_end_to_end_ms
3. exposed_io_ms
4. planned_peak_bytes
5. refill_slots（更少优先，作为稳定 tie-break）
6. candidate key canonical string
```

预测误差只影响性能，不影响安全；安全始终由 upper/ledger 决定。

### 177.3 合法降级图

运行中只能走 admission 时已经认证和预编译的边：

```text
normal S3/P8
   | pressure at safe point
   v
tight S2/P4
   | pressure at step boundary
   v
serial S1/P0
   | no feasible reservation
   v
abort
```

tile candidate 只能在 decode stage 开始前切换；sampler、dtype、量化、分辨率、帧数不属于内存降级，
不得修改。若 S1 backend 根本不存在，图直接从 S2/P4 到 abort。

### 177.4 revision identity

`ScheduleRevision` 至少绑定：

```cpp
struct ScheduleRevision {
    uint32_t revision = 0;
    std::string candidate_digest;
    std::string manifest_digest;
    std::string plan_digest;
    unsigned refill_slots = 0;
    unsigned pinned_blocks = 0;
    std::string tiling_mode;
    uint64_t planned_peak_bytes = 0;
};
```

每次合法切换发出 `schedule_revision_changed` trace，包含 old/new digest、safe point、原因和切换前后
headroom。结果 JSON 输出最后 revision 及 revision history digest。

## 178. Manifest upper 公式与容量计算规范

### 178.1 所有公式使用 checked arithmetic

统一提供：

```cpp
uint64_t checked_add_bytes(uint64_t, uint64_t, std::string_view site);
uint64_t checked_mul_bytes(uint64_t, uint64_t, std::string_view site);
uint64_t checked_align_up(uint64_t, uint64_t alignment,
                          std::string_view site);
uint64_t checked_product(std::span<const uint64_t>, std::string_view site);
```

alignment 必须是非零 2 的幂；overflow 返回 `memory_estimate_unknown`，不能饱和到 `UINT64_MAX` 后继续
排序候选。

### 178.2 常见 backing 公式

```text
tensor_logical = product(shape) * storage_bytes(dtype)
metal_shared    = align_up(tensor_logical, metal_buffer_alignment)
metal_private   = align_up(tensor_logical, metal_buffer_alignment)
                + upload_staging_if_live
host_vector     = capacity * sizeof(T)            // 不是 size
rgb8_output     = frames * height * row_stride
f32_decode      = frames * height * width * channels * 4
refill_slot     = max(bytes(block_i) for i assigned to slot type)
```

Metal private buffer 从 host 上传时，经常同时存活 source staging、private target 和 command-buffer
tracking；manifest 必须画出三者 overlap。`newBufferWithBytes` 可能隐含复制，不允许只算目标 logical
bytes。若统一内存使用 `storageModeShared`，CPU/GPU 指向同一 backing 时用 StorageId 去重，但仍按
`allocatedSize` 计费。

### 178.3 attention/workspace envelope

attention workspace 可能随 kernel/shape 离散切换，不能仅用 `tokens^2` 理论式。建议：

```text
upper = max(exact_formula(shape, selected_kernel),
            validated_runtime_workspace_upper(kernel_revision, shape_bucket))
```

若 runtime 无法提前报告 workspace，必须为该 kernel/shape/device 建立 validated envelope；runtime 或
OS build 改变使 record 失效。不能从上一请求 observed peak 临时学习一个 upper 并在同一进程授予
执行权。

### 178.4 child helper 与 IPC

LTX video VAE/finalizer helper 若以子进程运行，预算是 worker 集合预算：

```text
process_tree_peak = parent_footprint
                  + child_footprint
                  - verified_shared_file_pages_adjustment
```

首发建议不做 shared-page 精细扣减，直接使用 child process validated envelope，避免把共享 page 的
统计差异误作可用容量。父进程在 child 存活期间持有的 input/output pipe、mapped file 和 decoded frame
也计入 live set。helper timeout 或异常退出按 failure disposition 处理，不能只释放 parent ledger token。

## 179. 模型级资源图与首发 site 清单

### 179.1 H3 C/Metal

H3 首发最小资源图：

```text
text weights + tokenizer scratch
          |
          +--> conditioning -----------------------------+
                                                        |
DiT trunk + pinned blocks + slot[0..N) + latent + Euler velocity
                                                        |
                                                        v
                                      VAE weights + tile workspace
                                                        |
                                                        v
                                             decoded/output/export
```

建议 required site 至少包括：

| site prefix | class | lifetime | upper 来源 |
|---|---|---|---|
| `h3.text.weights.*` | weights | stage | safetensors metadata |
| `h3.text.tokenizer.*` | conversion scratch | stage | bounded token/vector capacity |
| `h3.conditioning.*` | conditioning | request/denoise | checked shape formula |
| `h3.schedule.*.readback` | conditioning | command buffer | exact scalar/vector bytes |
| `h3.dit.trunk.*` | weights | denoise | checkpoint metadata |
| `h3.dit.pinned_block.*` | weights | denoise | per-block metadata |
| `h3.dit.refill_slot.*` | refill slot | block | max typed block bytes |
| `h3.dit.activation.*` | activation | command buffer/step | exact or validated kernel envelope |
| `h3.euler.previous_*` | activation | step | latent formula |
| `h3.vae.weights.*` | weights | decode | checkpoint metadata |
| `h3.vae.tile_workspace.*` | activation | tile | tile formula/envelope |
| `h3.vae.decoded_host.*` | output | decode/export | row-stride-aware formula |
| `h3.export.*` | output | export | media contract/envelope |

H3 schedule gate-score host readback 必须与 GPU source completion 绑定。CPU vector 已释放不等于 GPU
buffer 可释放；两者如为不同 backing 应有不同 site/StorageId。

### 179.2 LTX C/Metal

LTX 至少有五个阶段，不能把两个 denoiser 和 upsampler 合并成一个粗略 reserve：

```text
Gemma text
 -> Stage-1 denoise
 -> latent upsample
 -> Stage-2 denoise
 -> video VAE
 -> finalizer/export
```

required site 建议：

```text
ltx.text.gemma.weights.*
ltx.text.tokenizer.host.*
ltx.conditioning.embedding.*
ltx.stage1.trunk.* / pinned.* / refill_slot.* / activation.*
ltx.upsampler.weights.* / graph_temporary.* / latent_copy.*
ltx.stage2.trunk.* / pinned.* / refill_slot.* / activation.*
ltx.video_vae.weights.* / tile_workspace.* / decoded_host.*
ltx.finalizer.child_envelope / encoded_output.*
```

Stage-1 和 Stage-2 即使 shape 相同也使用不同 stage/generation，除非它们确实共享同一 backing，且
manifest 以 alias group 和 source identity 证明。`std::vector::reserve()`、Objective-C `NSData`、
MPSGraph executable temporary 和 helper IPC buffer 都要进入 registry 或 framework envelope。

### 179.3 Flux F1

Flux 首发只开放 component-staged，不开放 block pager：

| stage | 必须释放后才能进入下一阶段的对象 | 允许跨阶段保留 |
|---|---|---|
| text encode | text weights、tokenizer scratch | text embeddings |
| transformer | transformer weights、graph workspace | final latent |
| VAE decode | VAE weights、tile workspace | decoded image |
| export | encoder/conversion scratch | 最终输出 |

若 MLX lazy graph 让 text/transformer backing 实际同时存活，则 manifest 必须按真实 overlap 计峰值，
不能按逻辑 Python/C++ scope 假定已释放。F1 只在明确 `eval/synchronize/clear` safe point 后转阶段。

### 179.4 Z-Image BF16/GGUF

两种 candidate 独立：

- BF16：source tensor、MLX/Metal target、graph temporary；
- GGUF：mmap/source pages、quant metadata、dequant scratch、converted tile/block target。

GGUF layer group 的 slot upper 使用同组最大 packed+metadata+alignment 大小；不同 quant type 不放进同一
slot class。若某层只能 whole-model materialize，首发退回 component-staged 或拒绝，不能把 mmap 当成
“已经 streaming”。

## 180. WeightPager/I/O/GPU overlap 的工程实现

### 180.1 slot 对象

```cpp
struct PagerSlot {
    SlotId id;
    uint64_t generation = 0;
    SlotState state = SlotState::Free;
    uint64_t capacity_bytes = 0;
    StorageLease backing;
    std::optional<ReadTicket> read;
    std::optional<GpuCompletionToken> completion;
    BlockKey assigned_block;
};
```

状态迁移必须在一个 owner queue/锁域内完成；文件读取 completion 和 GPU completion 只投递事件，不在
任意 callback 线程直接重写多张表。每次迁移验证 expected generation/state，失败记
`stale_completion` 或 `memory_lifetime_violation`。

### 180.2 双槽时序

```text
slot A: reserve -> read block i   -> upload/ready -> GPU block i -> pending -> free
slot B:                    reserve -> read block i+1 -> upload/ready -> GPU block i+1
GPU:                                    [compute i]                 [compute i+1]
I/O:             [read i]               [read i+1]
```

对于 private Metal buffer，如果 read 先进入 host staging，再 blit 到 private buffer，则“A/B”每个逻辑槽
可能包含两个物理 backing；manifest 必须把两者都计入，而不是把 staging 称为临时实现细节。

### 180.3 backpressure

```cpp
PrefetchDecision decide_prefetch(const PagerState &state,
                                 const MemoryPressureDecision &pressure,
                                 const CompiledSchedule &schedule) {
    if (pressure.abort_request) return Abort;
    if (pressure.stop_optional_prefetch) return DemandOnly;
    if (!state.has_free_certified_slot()) return WaitForCompletion;
    if (!schedule.next_block_is_deterministic()) return DemandOnly;
    return PrefetchNext;
}
```

等待策略必须有 deadline/cancellation 检查。取消后停止新 read，但已经提交的 I/O/GPU 必须 drain 到明确
终态；不能为了快速 cancel 直接复用 slot。

### 180.4 overlap 指标

每个 block 记录：

```text
reserve_start/end
read_start/end
upload_start/end
gpu_wait_start/end
compute_start/end
completion_time
release_time
```

定义：

```text
exposed_io_wait = max(0, compute_start(i) - compute_end(i-1)) 中由 I/O 引起的部分
hidden_io       = read/upload duration - exposed portion
hidden_fraction = hidden_io / total_read_upload_duration
```

性能验收同时看端到端 wall 和 exposed wait；只提高 hidden fraction 但增加总 I/O、导致 wall 更慢，不算
优化成功。

## 181. 调度器、ledger 与 completion 的并发不变量

### 181.1 单 owner 与 callback mailbox

首版建议 scheduler 单线程拥有 mutable slot/stage state，callback 只写有界 mailbox：

```cpp
struct CompletionMessage {
    CompletionKind kind;
    uint64_t token;
    uint64_t generation;
    int status;
};
```

优点是避免 allocator callback、Metal completion queue、API thread 三方锁反转。mailbox 容量来自
manifest 最大 live completion 数；满时是 lifetime violation，不退回 heap。

### 181.2 锁顺序

若仍使用 mutex，固定顺序：

```text
engine mutex
 -> process GPU execution mutex
 -> execution-context scheduler mutex
 -> ledger mutex
 -> per-slot mutex（最好不存在）
```

禁止在持有 ledger/scheduler mutex 时调用：Metal wait、`mx::synchronize()`、file read、user callback、
event callback、session unload。completion handler 也不能取得 engine mutex。

### 181.3 成功不变量

`finish_success()` 前同时断言：

```text
reservation_count == 0
pending_release_count == 0
scheduler.pending_count == 0
unknown_bytes == 0
all slots == FREE or retained-as-explicit-cache
mailbox empty
no I/O ticket outstanding
no child helper alive
actual_peak <= B
observed_over_budget == false
```

retained cache 首发建议为零。若以后允许，cache 必须有 request-external owner、独立 ledger 和下一请求
baseline/identity 规则，不能把 request context 的 lease 移出生命周期。

### 181.4 ABA 与 handle identity

`StorageId{domain,handle,capacity,generation}` 中：

- domain 标识 allocator/session；
- handle 必须在 generation 内唯一，不直接信任可快速复用的裸 pointer；
- capacity 为实际 allocated size；
- generation 每次 context/admission binding 增加且跳过 0。

若 Metal object pointer 被 allocator 复用，旧 completion 携带旧 generation，必须无法释放新 lease。

## 182. Probe 与 sidecar 专项测试设计

### 182.1 新增测试文件

建议增加：

```text
tests/native/memory_probe_test.mm
tests/native/test_memory_probe.py
```

测试构造临时模型根目录，不引用真实权重：

```text
tmp-root/
  checkpoint/shard-0.safetensors      # 几十 bytes 固定内容
  config/model.json
  memory-capabilities/synthetic_streamed_v1/
    w64_h64_f1_s1_audio0_input0_lora0_slots2.json
```

Python wrapper 用 `xcrun clang++` 编译 Objective-C++ test，链接 Foundation 与 CommonCrypto，并包含：

```text
memory_probe.mm
memory_manifest.cpp
memory_policy.cpp
common.cpp
```

### 182.2 正向向量

```text
MP-01 valid fixture -> key/manifest 与 expected 完全相同
MP-02 file list reorder -> canonical checkpoint digest 不变
MP-03 site/instance reorder -> manifest digest 不变
MP-04 repeated probe on immutable root -> hash cache hit，结果不变
MP-05 missing sidecar -> nullopt；不创建 sidecar
```

### 182.3 负向向量

```text
MP-10 sidecar path outside root
MP-11 logical path contains ..
MP-12 symlink escapes root
MP-13 checkpoint missing/empty
MP-14 size mismatch
MP-15 sha mismatch
MP-16 file changes while hashing
MP-17 duplicate logical file
MP-18 duplicate site or instance id
MP-19 required site uses heuristic provenance
MP-20 live_begin >= live_end
MP-21 upper=0 or integer >2^53
MP-22 adapter/model/backend/dtype/variant/operation mismatch
MP-23 shape/sampler/slot/tiling mismatch
MP-24 runtime/device mismatch
MP-25 manifest checkpoint/candidate mismatch
MP-26 malformed JSON/invalid UTF-8/unknown enum
```

每个错误断言稳定的 error code prefix，不断言整段本地路径或 Foundation 文案。

### 182.4 purity instrumentation

测试 build 提供计数器或 link seam：

```text
metal_buffer_creates
metal_command_submits
mlx_load_calls
mlx_eval_calls
graph_compile_calls
session_unload_calls
cache_clear_calls
worker_start_calls
```

所有 probe test 前后必须为零。允许的变化只有文件 open/read/close 与短期 JSON/parser host allocation；
这些 host allocation 的峰值另设合理测试上限，避免恶意 sidecar 造成无界内存。sidecar 大小、array 长度、
string 长度都要有 hard limit，例如 16 MiB、10 万 instances、4 KiB 单字符串；实际值根据最大模型
manifest 校准并写入 revision。

### 182.5 集成测试

使用临时 H3/LTX model root 创建 session，并打开 constrained request：

1. sidecar 缺失：在 unload/cache clear 前返回 `memory_policy_unsupported`；
2. valid sidecar + 空 production registry：仍在 mutation 前 registry miss；
3. test-only registry 精确 record：进入 plan compile，但不运行真实 model allocation；
4. disabled：probe/path/hash counter 为零。

第 3 项只能通过测试注入 registry，不能给 production registry 增加 synthetic record。

## 183. 分层验收、性能目标与发布判定

### 183.1 验收层级重新明确

| 等级 | 环境 | 证明内容 | 不证明什么 |
|---|---|---|---|
| L0 | 纯 C++/ObjC++、无 GPU | parser、digest、plan、ledger、状态机 | Metal lifetime/峰值 |
| L1 | API/session synthetic fixture | probe purity、fail-closed、disabled isolation | 真实模型 allocation closure |
| L2 | 目标 Metal 设备 + 单 candidate | hook、completion、峰值、fault cleanup | 低物理内存长期稳定 |
| L3 | 16/24/32 GiB 低内存实机 | swap、质量、重复运行、冷/热启动性能 | 其他 device/runtime |

没有 Metal 设备时 L2 只能 SKIP；SKIP 既不是 PASS，也不计入 release gate。

### 183.2 默认路径性能 SLO

disabled 是最严格的兼容目标：

```text
probe/hash/manifest/ledger/trace/cache-clear/unload extra calls = 0
route/residency/cache key/thread topology = baseline identical
steady-state wall median regression <= 2%
peak memory regression <= measurement noise envelope
output digest/quality contract unchanged
```

用 A/B/B/A 或 A/B/A/B 交错运行，避免温度、文件缓存和首次 shader compile 把时间趋势误判成回归。

### 183.3 constrained 性能目标

安全门优先于性能门。在 `actual_peak<=B`、swapout=0、quality pass 后，再评估：

```text
end-to-end slowdown versus same route unconstrained
time-to-first-denoise-step
exposed I/O wait per block
hidden overlap fraction
SSD bytes read per generated frame
slot reuse count and stall distribution
stage transition drain time
```

不预设所有模型都必须达到同一 slowdown。每个 candidate 在 evidence 中声明目标，例如：

```text
H3 dual-slot: median wall <= unconstrained_streamed * 1.15
LTX triple-slot: median wall <= unconstrained_streamed * 1.20
component-staged Flux/Z-Image: 单独基线，不与 resident 假装等价
```

上述数字是建议的首轮工程目标，不是已验证事实；真实 threshold 由 L2/L3 数据审定。P95 和最慢一次
也必须报告，避免平均值掩盖 page-in 或 drain 长尾。

### 183.4 swap 判定

在安静专用机上记录请求前后：

```text
vm_stat swapins/swapouts page counters
compressor pages/bytes
memory pressure state
process and process-tree footprint
vm.swapusage（辅助）
```

首发 L3 hard gate 是 `swapouts_delta==0`。已有 swap.used 不为零不自动失败，但必须记录起点；系统有
其他进程造成的 swapout 时该 run 标记 inconclusive，不把噪声归功或归罪给 candidate。若 TurboCider
本身观察到 `actual_peak>B`，即使 swapout 恰好为零也失败。

### 183.5 质量和确定性

内存模式默认不改变 sampler、seed、steps、dtype、模型变体、分辨率和帧数。允许 tile 时按模型定义
数值容差：

```text
exact route: output hash 或中间 tensor parity
floating-point reorder: max_abs/relative_l2/cosine
media output: frame count/尺寸/fps/audio contract + perceptual metric
```

performance 通过但质量失败，candidate 仍为 no-go。不能用缩短视频或减少 steps 让低内存结果通过。

## 184. 可直接执行的 patch 队列与最终签核增补

### 184.1 建议 patch 队列

```text
MC-01  [完成] memory_probe.mm 加入 build；修齐 include/warning
MC-02  [部分完成] sidecar strict parser、path root check、limits；剩余完整 key equality/TOCTOU
MC-03  [完成] memory_probe_test.mm + Python compile/run wrapper
MC-04  [完成] H3 probe override + immutable hash cache；disabled counter 仍需增强
MC-05  [完成] LTX C/Metal probe override + route identity；disabled counter 仍需增强
MC-06  [完成] API completion drain + disposition consumption + quarantine bit（真实异步 drain 仍需 backend-specific evidence）
MC-07  H3 required allocation-site closure + exact sidecar generator
MC-08  LTX required allocation-site closure + helper/process-tree envelope
MC-09  pager mailbox/generation/backpressure + trace
MC-10  evidence runner + fault campaign
MC-11  H3 L2/L3 record proposal（不与 runtime patch 混合）
MC-12  LTX L2/L3 record proposal
MC-13  Flux component-staged F1
MC-14  Z-Image BF16/GGUF component-staged F1
```

每个 patch 保持 production registry 空或 release-disabled，直到 MC-11/12 的独立 review。MC-01–06 已在
没有模型权重、没有 Metal device 的环境完成。

### 184.2 每个 patch 的 review 证据

```text
changed symbols/files
default-disabled call-count proof
new/changed site ids and formula revision
manifest/candidate/plan digest changes
unit/integration commands and exact PASS/SKIP count
fault cleanup result
known unguarded sites
whether registry changed（默认 none）
rollback scope
```

若 manifest digest 因纯排序或 JSON pretty-print 改变，说明 canonicalization 有 bug；若因公式/site/lifetime
语义改变，必须提升 revision 并使旧 evidence 失效。

### 184.3 首个 probe patch 的验收命令建议

```sh
env MLX_ROOT=<mlx-root> TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
python3 tests/native/test_memory_probe.py
python3 tests/native/test_memory_manifest.py
python3 tests/native/test_memory_execution.py
python3 tests/native/test_contract.py
```

真实命令中的 `<mlx-root>` 由本机环境提供，不写入仓库。输出需明确区分：PASS、fixture SKIP、no Metal
SKIP 和 FAIL；不能只看进程 exit code 后概括为“全部通过”。

### 184.4 最终签核新增问题

在第 170 节模板基础上，reviewer 还必须回答：

```text
[ ] sidecar 自身是否被限制在 model root 内？
[ ] expected key 是否由执行 adapter 生成并逐字段匹配？
[ ] checkpoint hash cache 的 trust policy 是否明确？
[ ] probe 是否在任何 unload/cache clear/GPU allocation 前完成？
[ ] context/control-plane allocation 是否进入 framework upper？
[ ] finish_success 是否发生在 completion drain 之后？
[ ] failure disposition 是否真正影响 engine/worker reuse？
[ ] 每个在线 schedule revision 是否在 admission 时已认证？
[ ] helper/child process 是否计入同一个预算域？
[ ] disabled 路径是否有运行时零调用证明？
```

只要其中一项为否，candidate 就不能进入 `release_enabled=true`。本轮追加仍不改变总体结论：
TurboCider 应优先实现显式、可证明、可回收的分层 streaming/loading/offloading；操作系统 swap 仅作
外部安全信号和失败证据，不是调度器的容量层。

## 185. 本轮实际落地与证据记录

本轮除了文档补充，还落地了以下代码闭环：

```text
native/platform/apple/memory_probe.mm
  -> 已加入 tools/native/build.sh 主构建
  -> sidecar 路径必须位于 model root
  -> sidecar 大小、数组长度、字符串长度和整数范围受限
  -> checkpoint 使用同一 fd 的 inode/ctime/mtime/size 前后快照 + SHA-256

H3Session/LTXSession
  -> metadata-only probe override
  -> session-owned hash cache
  -> constrained route 与 C/Metal candidate identity 检查

ModelSession/c_api.mm
  -> 统一 drain_memory_completions()
  -> finish_success() 前强制 completion drain
  -> failure disposition 消费和 engine quarantine

tests/native
  -> memory_probe_test.mm + test_memory_probe.py
  -> manifest/execution/contract 测试扩展
```

已执行并通过：

```text
native build                                  PASS
test_memory_probe.py                          PASS
test_memory_manifest.py                       PASS
test_memory_execution.py                      PASS
test_memory_scheduler.py                      PASS
test_memory_accounting.py                     PASS
test_memory_plan_compiler.py                  PASS
test_contract.py                               81 PASS / 1 fixture SKIP
test_h3_gpu_memory_hooks.py                   SKIP: no Metal device
test_ltx_gpu_memory_hooks.py                  SKIP: no Metal device
```

这些结果只证明 probe/ledger/context/scheduler 的逻辑和接线；它们不证明真实 GPU allocation closure、
Metal command completion、低内存物理机 swap 行为或任何 candidate 的 release capability。当前
production registry 仍为空，因此真实 constrained 请求仍应在 capability preflight 处拒绝。这是预期的
安全状态，不是测试失败。

仓库聚合入口还观察到两个环境边界：默认 `make test` 使用的 `python3.11` 在本机不存在；用
`make test PYTHON=python3` 后，内存相关测试均通过，随后既有 `test_video_timing.py` 因
`video writer start failed` 退出。该失败发生在本轮 memory patch 之外，不能被记为 memory scheduler
通过或失败；提交 evidence 时必须单独标记为环境失败并保留原始日志。

后续完成顺序固定为：

```text
1. 完整 expected-key equality 与 sidecar trust policy
2. H3/LTX 全 required allocation-site closure
3. 真实 Metal 设备 L2 completion/peak/fault evidence
4. 16/24/32 GiB 低内存机 L3 swap/quality/performance evidence
5. review-only registry record；之后才允许 release_enabled=true
```

## 186. 追加设计决策：把“内存上限”变成可执行的容量合同

本节是对前述 proposal/scaffolding 的进一步收敛，目的是让实现者能够在不改变默认路径的前提下，
直接开始编写 scheduler、pager 和 model adapter。下文中的“GPU”特指 Metal/Apple GPU 路径；CPU、
ANE、独立服务进程和系统级 swap 不属于本轮执行容量层，只有在模型的 manifest 明确将它们纳入预算域时，
才能作为被观测的外部资源。

### 186.1 最终选择：显式分层调度优先，swap 只做安全信号

两种方案的工程结论如下：

| 方案 | 能否给出硬上限 | 峰值是否可证明 | 是否可做 overlap | 失败是否可定位 | 本项目结论 |
|---|---:|---:|---:|---:|---|
| 依赖操作系统 swap | 否 | 否 | 不可控 | 很差 | 只记录 swapout，不能作为容量层 |
| 模型层 streaming/loading/offloading | 是 | 是 | 是 | 可定位到 site/stage/slot | constrained 首选 |
| 两者混合 | 只有显式层可证明 | 仅显式层 | 显式层可做 | 取决于未知页 | 允许作为防线，但 swap 触发即失败 |

原因不是 swap 一定会导致崩溃，而是 swap 的换入/换出时机由系统全局压力、压缩器、其他进程和文件缓存
共同决定，scheduler 无法把一次 page-out 映射回某个权重 slot，也无法保证 `Y` 与实际物理峰值的关系。
相反，显式 pager 可以在真实 GPU allocation 之前取得 permit，并在 GPU completion 后释放 permit。这样
`planned_peak`、`actual_peak` 和 `swapouts_delta` 可以分别回答“计划是否安全”“运行是否越界”“系统是否
已经开始用 swap”。

因此 constrained 模式的硬规则是：

```text
允许显式的 GPU 分层 backing；
允许系统仍有既存 swap.used，但本请求新增 swapout 必须为零；
一旦出现未知 GPU backing、未计账的 framework temporary 或实际峰值超过 B，立即失败并按 disposition 处理；
绝不因为 swap 可用就把 candidate 视为可执行。
```

### 186.2 Y、X、B 的精确定义

请求字段的语义固定如下：

```text
Y = memory_constrained.limit_bytes                 # 用户输入的总预算
X = memory_constrained.buffer_percent              # 冗余百分比
B = floor(Y * (100 - X) / 100)                     # 可用于执行的有效预算
F = memory_constrained.min_free_bytes               # 系统级最低可用内存
```

`Y` 是产品层的承诺边界，`B` 才是 planner 和 ledger 的硬上限。`X` 不是“可以超出 B 的额度”，而是
在 `Y` 内保留给系统、窗口抖动、Metal bookkeeping 和未纳入 candidate 的短期开销。除非经过新的证据
审查，首发建议把 `X` 限制在 `0..50`，并拒绝负数、浮点、整数溢出和会使 `B==0` 的值。

预算分解统一写成：

```text
B >= process_baseline
  + framework_upper
  + weights_live
  + activation_live
  + conditioning_live
  + refill_slots_live
  + output_live
  + child_process_envelope
```

其中 `process_baseline` 在 constrained request 的 clean boundary 之后采样；不能把模型已经驻留的旧
session 当作 baseline 隐式带入。`framework_upper` 必须是 manifest 或独立 validated envelope 中的
固定上界，不能用一次运行的 observed peak 反推。所有加法和乘法都使用 checked arithmetic，overflow
转为 `memory_estimate_unknown` 并拒绝执行。

### 186.3 三个容量域和一个禁止域

实现上将内存分成三个必须有 owner 的容量域，以及一个禁止放行的未知域：

```text
GPU domain       Metal private/shared buffers、GPU activation、refill slot、GPU graph backing
Host domain      tokenizer/conditioning、upload staging、decoded frame、codec/output
Process domain   process baseline、helper/child envelope、allocator/framework watermark
Unknown domain   无 site_id 或无 completion/lifetime 证明的任意 backing
```

GPU-only constrained 的“只考虑 GPU”并不意味着可以忽略 host 和 process：host staging 很可能与 private
buffer 同时存活，MLX/MPSGraph 也可能在 process footprint 中产生无法单独观察的 temporary。正确做法是将
它们纳入 `framework_upper` 或 candidate manifest；无法建立 envelope 时进入 `Unknown domain`，candidate
保持 `plan_only`。

### 186.4 默认路径的零行为变化合同

当 `memory_constrained.enabled == false` 时，以下行为必须保持基线等价：

```text
不调用 probe_memory_capability；
不读取 capability sidecar 或计算 checkpoint hash；
不创建 MemoryExecutionContext/MemoryLedger/MemoryStageScheduler；
不调用 session unload、mx::clear_cache、额外 synchronize 或 completion drain；
不改变 residency、backend、candidate、线程数量、队列拓扑和 cache key；
不新增每 allocation 的 mutex、trace、atomic 或 callback；
结果 JSON 中既有字段的值和类型保持兼容。
```

实现者不得用“构造一个空 context 再在 disabled 分支跳过 reserve”的方式满足兼容性，因为这仍可能
引入分配、锁竞争和线程拓扑变化。disabled 分支必须在 API 入口处直接沿旧调用图运行。

## 187. 运行时合同：从 request 到 GPU backing 的完整 API 设计

### 187.1 建议补充的类型

当前 `MemoryReservation`、`StorageLease` 和 `MemoryStageLease` 已经形成基础闭环。下一步应补充
site/generation/completion 三类身份，避免 C/Objective-C callback 只能拿到裸指针：

```cpp
struct MemorySiteToken {
    uint64_t reservation_id = 0;
    uint64_t allocator_domain = 0;
    uint64_t generation = 0;
    std::string site_id;
    MemoryClass memory_class = MemoryClass::UnknownExternal;
    uint64_t upper_bytes = 0;
    bool committed = false;
    bool cancelled = false;
};

struct MemoryCompletionToken {
    uint64_t value = 0;
    uint64_t allocator_domain = 0;
    uint64_t generation = 0;
    uint32_t slot_id = 0;
    uint32_t stage_id = 0;
};

enum class MemoryCompletionKind : uint8_t {
    GpuCommand,
    BlitUpload,
    FileRead,
    ChildProcess,
};
```

`MemorySiteToken` 只允许一次 `commit` 或 `cancel`；`MemoryCompletionToken` 只允许一次
`complete`。token 的字段应在 debug/test build 中完整保留，在 release build 中可以用 interned
site id 降低字符串开销，但不能删除 generation 和 allocator domain。

建议的 scheduler API 增量：

```cpp
std::optional<MemorySiteToken> MemoryStageLease::reserve_site(
    std::string site_id, MemoryClass klass, uint64_t upper_bytes);

StorageLease MemoryStageLease::commit_site(
    MemorySiteToken &&token, StorageId storage, uint64_t actual_bytes);

MemoryCompletionToken MemoryStageLease::retire_async(
    StorageLease &&lease, MemoryCompletionKind kind);

bool MemoryStageScheduler::on_completion(
    MemoryCompletionToken token, int status,
    bool retain_as_cache = false) noexcept;
```

现有 `reserve()`/`commit()`/`retire_all()` 可以继续保留为兼容 helper，但新 allocation bridge 不应再
直接传空 tag。空 `site_id` 只允许在测试 fake allocator 中使用；生产 candidate 对空 site 必须拒绝。

### 187.2 API 入口的顺序约束

`native/api/c_api.mm` 的 constrained 路径应保持以下顺序，代码 review 以这些边界为准：

```cpp
// 1. 只读解析和 route normalization
ExecutionPlan plan = make_plan(request);

// 2. metadata-only probe；registry miss 时此处抛出
auto capability = preflight_memory_capability(
    *engine->session, plan, device_identity,
    production_memory_capability_registry());

// 3. 通过 exact manifest 后才允许清理旧 session/cache
engine->session->unload();
if (parent_mlx) mx::clear_cache();

// 4. clean baseline + live interval plan
auto context = prepare_memory_execution(*engine->session, plan, parent_mlx);

// 5. context 绑定有效期覆盖完整 generate + completion drain
ScopedMemoryExecutionBinding binding(*engine->session, context.get());
auto result = engine->session->generate(plan.request, event, cancelled);
drain_memory_execution(*engine->session, *context);
context->checkpoint("complete");
context->finish_success();
```

任何代码路径都不得在第 2 步之前调用 `unload()`、`clear_cache()`、`load()`、`prepare()`、
`mx::eval()`、`MTLDevice newBuffer*` 或 graph compile。这样 registry miss 是纯只读拒绝，不会改变
本机的 warm cache 或 session 复用状态。

### 187.3 allocator bridge 的“先 reserve，后 allocate”模板

Metal/C allocator、MLX host vector 和 Objective-C buffer 都必须遵循同一模板：

```cpp
auto token = stage.reserve_site(
    "h3.dit.refill_slot.0", MemoryClass::RefillSlot, slot_upper);
if (!token) return fail_budget("h3.dit.refill_slot.0", slot_upper);

RawBacking backing = allocator.allocate_after_admission(slot_upper);
if (!backing) {
    token->cancel();
    return fail_runtime("allocation_failed");
}

auto lease = stage.commit_site(std::move(*token),
    backing.storage_id(domain, generation), backing.actual_bytes());
auto completion = stage.retire_async(std::move(lease),
    MemoryCompletionKind::GpuCommand);
command.add_completed_handler([mailbox, completion](int status) {
    mailbox->push({MemoryCompletionKind::GpuCommand,
                   completion.value, completion.generation, status});
});
```

以下写法禁止进入 constrained candidate：

```cpp
auto buffer = device.newBuffer(length, options); // 先分配
ledger.try_reserve(...);                         // 事后记账
```

也禁止在 permit 不足时偷偷降低 `upper_bytes` 并继续分配，因为实际 allocator 可能按 alignment、
page granularity 或 kernel workspace 产生更大的 backing。需要降级时，必须切换到 admission 阶段已
认证且已编译的 schedule revision。

### 187.4 completion mailbox 和锁边界

Metal completion handler、文件读取线程、helper waiter 只向有界 mailbox 投递消息，不直接修改
ledger 或 slot 表。scheduler owner 在 safe point 批量消费消息：

```text
callback thread: 读取 status + token -> mailbox.push()
owner thread:    pop -> 校验 generation/state -> ledger.complete_pending()
                 -> FREE/RETAINED -> 发出 trace
```

mailbox 容量取 manifest 的 `max_live_async_completions + safety_margin`，且 safety margin 是固定值，
不能随运行中的消息数量无限增长。mailbox 满时按 `memory_lifetime_violation` 处理并 quarantine；不能
退回普通 heap 作为“临时扩容”。

锁顺序固定为：

```text
engine mutex -> execution mutex -> scheduler mutex -> ledger mutex
```

持有 scheduler/ledger mutex 时不得调用 Metal wait、`mx::synchronize()`、文件 I/O、session unload、
用户 event callback 或 helper wait。这样可以避免 completion callback 与 API cancel 路径互相等待。

## 188. Scheduler 算法：预算、slot、pressure 和安全降级

### 188.1 两阶段 admission

调度器必须把“能力认证”和“本次机器基线适配”分开，避免为了观测 baseline 而先改变 session：

```text
Phase A: request 规范化 -> exact key/manifest -> registry lookup
         失败：直接拒绝，零 GPU/MLX/cache mutation

Phase B: unload/cache clean -> process baseline -> plan compile
         失败：返回 B、baseline、planned_peak、peak_live_set
         成功：创建 context、绑定 adapter、开始 stage
```

当未来有多个已认证 candidate 时，Phase A 返回候选集合，Phase B 对每个候选分别编译计划；只在
`planned_peak <= B`、`system_available - F >= planned_increment` 且不超过 record 的
`maximum_validated_upper_bytes` 时参加排序。当前 singular probe 接口可以继续工作，但必须明确它
只返回一个已归一化 candidate，不能在运行中临时生成第二个未认证方案。

### 188.2 stage/slot 的调度伪代码

```cpp
for (const auto &stage_spec : compiled_plan.stages()) {
    auto stage = scheduler.begin_stage(stage_spec.name);
    pressure = scheduler.observe(sample_process_and_ledger());
    if (pressure.abort_request)
        return abort("critical pressure before stage");

    for (const auto &block : stage_spec.required_blocks) {
        auto slot = pager.acquire_demand_slot(block, stage_spec.deadline);
        if (!slot)
            return abort_or_preauthorized_downgrade("no certified slot");

        slot.read_from_checkpoint();       // 只使用已 reserve 的 staging
        slot.commit_read();
        slot.upload_or_bind_gpu();
        slot.submit(block.command);

        if (stage_spec.allow_lookahead &&
            pressure.state == MemoryPressureState::Normal)
            pager.prefetch(next_deterministic_block());

        pager.retire_after_last_use(slot);
        scheduler.consume_mailbox_at_safe_point();
        execution.checkpoint(stage_spec.name + ".block");
    }
    stage.retire_all();
    execution.checkpoint(stage_spec.name + ".complete");
}
```

`prefetch()` 只能使用 manifest 已声明的 optional slot；当 pressure 为 `Tight` 时停止 optional
prefetch，当为 `Critical` 时停止所有新读并进入 drain/abort。必需 block 在没有 permit 时不能
“先读到 host 再说”，只能等待已提交 completion 或走预授权 serial revision。

### 188.3 pressure 状态机的动作表

| 状态 | 进入条件 | 新分配 | optional prefetch | cache | 降级 | 退出 |
|---|---|---|---|---|---|---|
| Normal | accounted < 80% 且系统 headroom 正常 | 允许 | 允许一个确定性 look-ahead | 默认不保留 | 无 | - |
| Tight | accounted ≥ 85% 或系统 headroom 接近 F | 仅 manifest-required | 停止 | safe point 可 trim | 可切换已预授权较少 slot | 连续 3 次恢复样本 |
| Critical | allocation failure、over-budget、系统低于 F | 禁止 | 禁止 | 只做必要 drain | 不能临时降级；无 serial 则 abort | 本请求终止 |

critical 一旦进入不可自动回到 Normal；即使后续 observed footprint 下降，也必须完成当前请求的
failure cleanup 并让 engine 根据 disposition 决定是否 quarantine。这样可以避免一次越界之后复用
可能仍持有旧 command buffer 的 worker。

### 188.4 合法降级和切换点

降级图只能在 admission 时编译并通过 registry 的边：

```text
S3/P8 -> S2/P4 -> S1/P0 -> abort
```

每条边绑定：`old_plan_digest`、`new_plan_digest`、`safe_point_kind`、`quality_contract_id`、
`planned_peak`。允许的切换点只有：

```text
denoise block boundary
stage boundary
decode tile boundary
```

不允许在一个 GPU command buffer 中途切换 slot 数量、dtype、sampler、分辨率或帧数。pressure
回调不得直接调用 planner；它只能向 owner 线程投递 `request_revision_change`，由 owner 在 safe
point 校验预授权计划并执行。

### 188.5 I/O 设计与 overlap 约束

首发 pager 使用 direct file descriptor + `pread`/等价的有界读取，避免 `NSData dataWithContentsOfFile`
或无界 autorelease 对象产生隐藏副本。每个 slot 至少有以下时间戳：

```text
reserve_start/end
read_start/end
upload_start/end
gpu_submit
gpu_wait_start/end
completion
release
```

只有满足 `read_end <= gpu_wait_deadline` 且期间没有额外 staging allocation，才计入
`hidden_io_bytes/hidden_io_ms`。否则计入 `exposed_io_wait_ms`。当磁盘吞吐不足时，scheduler 应降低
look-ahead，而不是增加第四个 slot；增加 slot 必须由新的 manifest、plan digest 和 capability
evidence 共同证明。

## 189. 模型适配施工稿：从当前源码到首发 candidate

### 189.1 H3：C/Metal streamed v1

当前代码入口是 `native/platform/apple/h3_session.mm` 和 `native/models/h3_runtime/*`；首发
candidate 仍为 `h3_c_metal_streamed_v1`。施工时按下面的资源图拆分，而不是把整个 H3 当成一个
`denoiser_bytes`：

```text
text weights/tokenizer
        -> conditioning + schedule readback
        -> pinned DiT blocks + refill slots + latent/velocity
        -> VAE weights + tile workspace
        -> decoded host + export
```

必须逐点闭环的 site 前缀：

```text
h3.text.weights.*
h3.text.tokenizer.host.*
h3.conditioning.embedding.*
h3.conditioning.schedule_gate_score.readback
h3.conditioning.schedule_branch_score.readback
h3.dit.trunk.*
h3.dit.pinned_block.*
h3.dit.refill_slot.*
h3.dit.activation.*
h3.euler.previous_velocity
h3.vae.weights.*
h3.vae.tile_workspace.*
h3.vae.decoded_host.*
h3.export.*
```

代码修改建议：

1. 在 `h3_runtime/h3.h` 的 options 结构中继续按值携带 `h3_gpu_memory_hooks` 和
   `h3_host_memory_hooks`，禁止 worker 保存调用方栈地址；`h3_session.mm` 创建新 generation 时
   更新 domain/generation。
2. 在 `h3_gpu.m` 的每一个 `newBuffer`/tensor constructor 处集中经过
   `h3_gpu_reserve_memory()`；source staging、private target、command tracking 若重叠，分别登记
   site，不得只登记 target。
3. 在 `h3_runtime/h3.c` 的 host envelope、tokenizer、conditioning vector 和 schedule readback
   处使用统一 host hook；`h3_video_vae.c` 的 decode tile 使用 tile-specific upper。
4. 将 VAE 的 `h3_gpu_tensor_from_f32` 等旧调用全部收敛到带分类的 helper；任何未分类 helper 在
   constrained build 中直接触发 `unknown_external`。
5. `H3Session::drain_memory_completions()` 在真实 Metal 版本必须等待 command buffer/event，而不
   只是读取 ledger pending count；如果底层路径同步提交，仍需提供“已同步”的明确证据字段。
6. H3 v1 继续拒绝 audio、reference input、runtime LoRA、quantized cache 和非 `video.generate`。
   这些组合必须拥有独立 candidate key，不能复用 v1 的 manifest。

H3 的最低验收是：每个 required site 在 manifest 有 instance；运行中 `unknown_bytes==0`；每个 slot
completion 能通过 `(slot_id,generation,ticket_id)` 找到唯一 lease；VAE decode 的 source/target
overlap 与 export output 的 lifetime 在 live interval 中可见。

### 189.2 LTX：双 denoiser + helper 的 staged/streamed v1

当前代码入口是 `native/platform/apple/ltx_session.mm`、`native/models/ltx_runtime/ltx_gpu.m`、
`ltx_blocks.c` 和 video VAE/helper。首发 candidate 为 `ltx_c_metal_streamed_video_v1`，只支持
video-only text-to-video，不继承 MLX route 的隐式 cache upper。

资源图必须保持五个边界：

```text
Gemma text
  -> Stage-1 denoise
  -> latent upsample
  -> Stage-2 denoise
  -> video VAE/helper
  -> finalizer/export
```

必需 site 前缀：

```text
ltx.text.gemma.weights.*
ltx.text.tokenizer.host.*
ltx.conditioning.embedding.*
ltx.stage1.trunk.* / pinned.* / refill_slot.* / activation.*
ltx.upsampler.weights.* / graph_temporary.* / latent_copy.*
ltx.stage2.trunk.* / pinned.* / refill_slot.* / activation.*
ltx.video_vae.weights.* / tile_workspace.* / decoded_host.*
ltx.finalizer.child_envelope / ltx.export.output.*
```

代码修改建议：

1. `ltx_gpu.m` 中的 `ltx_gpu_buffer_new_classified` 和 copy variant 作为唯一 GPU allocation
   入口；所有调用都必须传入 stage/site tag。禁止新代码直接调用旧 `ltx_gpu_buffer_new`。
2. `ltx_blocks.c` 的 stage1/stage2 context 分别保存 generation 和 stage id；即使两个 stage 的
   tensor shape 相同，也不能复用旧 stage token，除非 manifest 明确写 alias group。
3. `ltx_session.mm` 的 `reserve_host_memory`/`commit_host_memory` 覆盖 Gemma token vector、
   conditioning、latent copy、VAE decoded host 和 IPC buffer；`std::vector::reserve()` 的 capacity
   而非 size 写入 upper。
4. helper 进程在 parent 启动前取得 `ChildProcessEnvelope` reservation；helper alive、pipe、mapped
   input/output、decoded frame 都在同一 live interval；timeout/异常退出产生 `QuarantineWorker`。
5. `uses_parent_mlx(request)` 在 constrained request 必须返回 false，除非未来有单独 MLX manifest 和
   registry record；当前 C/Metal v1 不能从 MLX observed peak 借用 upper。
6. Stage-1/Stage-2 的 sparse patterns、audio、image-to-video、LoRA 和不同 helper revision 各自
   绑定 candidate key；不允许用 `allow_approximation` 绕过 capability lookup。

LTX 的性能目标优先顺序是：先确保 stage boundary 不重叠、helper peak 被纳入 B，再尝试 stage1
最后一个 block 与 upsampler 准备的 overlap。任何 overlap 都必须以独立 alias/lifetime evidence
证明，不能仅凭 wall time 下降推断。

### 189.3 Flux：先做 component-staged F1，再做 layer-group pager

Flux 当前 `native/models/flux2/pipeline.cpp` 的 `load()` 会 materialize transformer/VAE，
`conditioning()` 可能保留 image weights；这与 constrained 的 clean baseline 和 staged contract
不同。首发建议：

```text
text_encode -> completion drain -> transformer denoise
            -> completion drain + unload transformer
            -> VAE decode -> completion drain -> export
```

F1 修改范围：

- 在 `Flux::load`/`Flux::conditioning`/denoise/vae decode 周围引入 stage lease，但不切 transformer
  block；
- `mx::eval`、`mx::synchronize`、`clear_cache` 必须位于 manifest 声明的 safe point；
- text embedding 只有在复制到已登记的 conditioning backing 后才能跨 stage 保留；
- LoRA in-memory merge 的 temporary、compiled graph 和 image input backing 必须进入 framework
  envelope；无 envelope 时 F1 拒绝 constrained；
- Flux 9B、4B、不同 encoder manifest 和 GPU+ANE route 不共享 record。

F2 layer-group pager 需要新增 `flux.denoise.layer_group.*` manifest，并证明 MLX lazy array、
compiled graph 和 source weight staging 的真实 overlap。F2 之前不得把 `residency=component_staged`
伪装成“已支持 block streaming”。

### 189.4 Z-Image：BF16 与 GGUF 分开，先按组件分段

`native/models/z_image/*` 和 `z_image_gguf_module.cpp` 当前均以 resident 为主。constrained 的 F1
应新增两个独立 candidate：

```text
z_image_bf16_staged_v1
z_image_gguf_staged_v1
```

BF16 资源：Qwen/text embedding、transformer weights/graph temporary、VAE weights、decoded output。
GGUF 资源还必须包含 mmap/source pages、quant metadata、dequant scratch 和 packed layer backing。
不能因为 GGUF 文件是 mmap 就把 source pages 从预算删除；必须使用 process observation 或 validated
file-page envelope 记录它们。

F1 修改建议：

1. 在 `ZImage::load`/prepare/denoise/decode 的组件边界插入 stage lease 和 explicit drain；
2. `z-image-turbo-gguf` 的 Q8/Q4 不同 quant type 使用不同 slot class，禁止混用最大值掩盖布局差异；
3. inference-time LoRA 的 packed low-rank branch、in-memory merge temporary 和 Core ML artifact
   作为独立 candidate；
4. 只有在真实设备上证明 whole-model materialize 不发生时，才进入 F2 layer-group pager；否则保留
   component-staged 或拒绝。

## 190. 失败、取消和 worker 生命周期

### 190.1 错误分类

错误字符串必须有稳定前缀，便于 API、测试和 evidence parser 使用：

```text
memory_policy_invalid          # Y/X/F、字段类型或组合非法
memory_policy_unsupported      # 没有 exact route/manifest/registry record
memory_budget_too_small        # B、baseline、planned peak 或 system reserve 不足
memory_estimate_unknown        # required upper 无法证明或 arithmetic overflow
memory_allocation_failed       # reserve 成功但真实 allocator 失败
memory_pressure_abort          # 运行中进入 Critical
memory_lifetime_violation      # completion、generation、slot 或 release 不一致
memory_observation_unreliable  # process/system footprint 无法观测
memory_worker_quarantined      # 之前请求已污染 worker，必须重建
```

禁止在失败时静默回退到 resident、parent MLX、系统 swap 或缩短 steps/frames。若请求选择了
constrained，就只能在已授权降级图内运行；否则返回可操作的最小预算建议和失败原因。

### 190.2 cancel/fault 的反向 cleanup

取消或异常时按以下顺序执行：

```text
stop new prefetch/read
 -> stop enqueueing new GPU commands
 -> wait/drain submitted I/O and GPU completion with deadline
 -> consume mailbox and release/retire every lease
 -> stop/reap child helper
 -> final process observation
 -> disposition = Clean / NeedsGpuDrain / QuarantineWorker
```

deadline 到期时不能直接清空 slot vector 或复用 generation。`NeedsGpuDrain` 只有在 bounded drain
后 ledger 的 pending/unknown/reservation 全部为零才可降为 `Clean`；否则设置 engine quarantine。
`tc_engine_free()` 仍必须允许释放 quarantined engine，但 `tc_engine_generate()` 应在真正开始
model mutation 前返回 `memory_worker_quarantined`。

### 190.3 child/helper 处理

helper 的生命周期必须有 owner：parent context 创建 envelope reservation，启动 helper 后记录 pid、
generation、pipe bytes 和 deadline；helper 正常退出后才完成 token。SIGKILL、崩溃、超时或无法读取
最终状态统一按 `QuarantineWorker`，不能仅凭 parent 的 `waitpid` 返回值把 GPU/host lease 标记为已释放。

## 191. 结果 JSON、trace 和调试接口

### 191.1 结果字段建议

在既有 `memory_admission` 下补充以下稳定字段；未知值用 `null` 或空字符串，不用 `0` 冒充已测得：

```json
{
  "budget_bytes": 0,
  "user_limit_bytes": 0,
  "buffer_percent": 0,
  "planned_peak_bytes": 0,
  "actual_peak_bytes": 0,
  "process_baseline_bytes": 0,
  "framework_upper_bytes": 0,
  "unknown_peak_bytes": 0,
  "pending_release_peak_bytes": 0,
  "system_available_min_bytes": 0,
  "swapouts_delta": null,
  "candidate_key_digest": "",
  "manifest_digest": "",
  "plan_digest": "",
  "schedule_revision": 0,
  "schedule_revision_history_digest": "",
  "slot_count": 0,
  "slot_refills": 0,
  "exposed_io_wait_ms": 0,
  "hidden_io_ms": 0,
  "pressure_transitions": 0,
  "failure_disposition": "none"
}
```

`actual_peak_bytes` 必须取 process observation 与 ledger peak 的较大者，不能只取 ledger；
`unknown_peak_bytes` 非零时 `execution_supported` 必须为 false 或 request 失败。`swapouts_delta`
如果平台不可用应为 `null`，不能写 0 造成误判。

### 191.2 trace event

trace 使用有界 ring，关键事件不可丢失。建议事件集合：

```text
memory_admission_start/end
memory_probe_start/end
memory_plan_compiled/rejected
stage_begin/end
site_reserve/commit/cancel
slot_read_begin/end
slot_upload_begin/end
gpu_submit/completion
pressure_transition
schedule_revision_changed
memory_checkpoint
memory_failure_disposition
```

每个事件至少带 `request_id`、`engine_generation`、`candidate_key_digest`、`manifest_digest`、
`plan_digest`、`site_id`、`stage`、`slot_id`、`generation`、`timestamp_ns`。路径、prompt 和 LoRA
内容只记录脱敏 digest，避免把用户数据写进 evidence。

### 191.3 调试命令

建议增加只读调试接口（不改变 runtime 状态）：

```text
tc_memory_plan_json(request)       # 只做 plan/probe，不 unload/cache clear
tc_memory_manifest_json(request)   # 返回 canonical manifest/digest
tc_memory_snapshot(engine)         # 当前 ledger/process observation
tc_memory_trace_dump(engine)       # 导出 bounded trace
```

这些接口必须显式标记 `read_only=true`，且不能因为调试调用而启动 worker、materialize 模型或授予
capability。production registry 仍只能由 reviewed build artifact 注入。

## 192. 按文件拆分的新增 patch 队列

在已有 MC-01～MC-14 之后，建议追加以下小步 patch。每个 patch 都应保持可独立回滚，且默认
`release_enabled=false`：

```text
MC-15  MemorySiteToken/MemoryCompletionToken 与一次性状态校验
MC-16  Scheduler mailbox + owner-thread drain；禁止 callback 直改 ledger
MC-17  H3 全 allocation-site inventory 与 required closure report
MC-18  H3 Metal completion token；移除仅凭 pending count 的假 drain
MC-19  LTX stage-specific generation、helper envelope 和 timeout disposition
MC-20  Flux component-staged F1（只改 stage boundary，不改 block residency）
MC-21  Z-Image BF16/GGUF component-staged F1，variant 分离
MC-22  多 candidate probe API（当前 singular wrapper 返回单元素）
MC-23  预授权降级图与 safe-point revision switch
MC-24  process-tree/swap evidence runner 和结果 JSON 字段
MC-25  fault campaign：allocation fail、read fail、GPU error、cancel、helper timeout
MC-26  真实设备 L2 evidence review；只生成 review-only registry record
MC-27  低内存 L3 16/24/32 GiB 重复运行与性能/质量签核
```

建议的文件与符号映射：

| Patch | 主要文件 | 关键符号/输出 |
|---|---|---|
| MC-15/16 | `native/runtime/memory_accounting.*`, `memory_scheduler.*` | token、mailbox、generation、pending count |
| MC-17/18 | `h3_session.mm`, `h3_gpu.m`, `h3.c`, `h3_video_vae.c` | site inventory、Metal event completion |
| MC-19 | `ltx_session.mm`, `ltx_gpu.m`, `ltx_blocks.c` | stage1/2 generation、child envelope |
| MC-20 | `flux2/pipeline.cpp`, `flux.hpp` | stage lease、component drain |
| MC-21 | `z_image.cpp`, `z_image_gguf_module.cpp` | BF16/GGUF manifest、staged boundaries |
| MC-22/23 | `session.hpp`, `memory_execution.*`, `memory_policy.*` | variant list、revision switch |
| MC-24/25 | `tests/native`, `tools/native` | evidence JSON、fault matrix |
| MC-26/27 | `docs/design`, review artifact | release gate，不直接改生产 registry |

每个 patch 的 PR 描述必须附带：修改 symbol、manifest/revision 变化、disabled 零调用证明、测试命令、
PASS/SKIP/FAIL 原始摘要、已知未覆盖 site、回滚范围。若 patch 只改变 canonical JSON 排序却改变
manifest digest，应视为 bug；若改变 lifetime/formula，必须提升 manifest/backend revision 并使旧 record
失效。

## 193. 详细验收方案和可重复实验

### 193.1 L0/L1：无 GPU 的逻辑验收

必须通过：

```text
memory_policy_test
memory_manifest_test
memory_plan_compiler_test
memory_accounting_test
memory_scheduler_test
memory_execution_test
memory_probe_test（临时 fixture）
test_contract.py
```

负向向量至少包含：Y/X 非法、baseline 大于 B、framework upper 溢出、unknown required site、
duplicate site/instance、generation ABA、重复 completion、pending release 未 drain、registry key
任一字段不匹配、sidecar 路径逃逸和 checkpoint TOCTOU。所有失败断言使用稳定 error prefix。

### 193.2 L1：默认路径隔离

为 disabled request 加入 link seam/counter，验证以下计数全部为零：

```text
probe_open、checkpoint_hash、manifest_parse、ledger_construct、scheduler_construct、
session_unload_extra、mx_clear_cache_extra、mx_synchronize_extra、completion_drain_extra、
memory_trace_event、memory_worker_thread
```

同时进行 A/B/B/A 交错基准：A 为原始 disabled，B 为包含新代码但仍 disabled 的 build。验收门：

```text
输出/质量契约不变；
steady-state wall median 回归 <= 2%；
峰值变化不超过测量噪声；
线程数、cache key、route/residency 与 baseline 相同。
```

### 193.3 L2：真实 Metal 设备

每个 candidate 至少运行：

```text
冷启动：新进程 + 空 cache + 单请求
热启动：同一进程连续 10 请求
shape 边界：认证 bucket 的最小/中间/最大 shape
slot：每个已认证 refill_slots
故障：第 N 个 block allocation/read/GPU completion 注入失败
取消：text、denoise、decode、export 各阶段取消
```

通过条件：

```text
planned_peak <= B；actual_peak <= B；unknown_peak == 0；
成功请求 finish_success 前 reservation/pending/mailbox/I/O/helper 全部为零；
失败请求有正确 disposition，quarantine 后不能复用同一 worker；
quality contract 通过；无未解释的 output/shape/fps/audio 变化；
Metal hook 的真实 completion evidence 存在；不能以“无 device”结果冒充 PASS。
```

当前没有 Metal device 时，命令仍应执行并输出 `SKIP: no Metal device is available`，该结果只能
计入环境说明，不计入 release gate。

### 193.4 L3：低内存机器和 swap 证据

在 16/24/32 GiB 机器分别记录：

```text
请求前后 vm_stat swapins/swapouts
memory pressure state
process/process-tree footprint
compressor pages
system available min
TurboCider actual_peak/planned_peak/unknown_peak
```

每个 `(model, shape, Y, X, slots)` 至少冷启动 3 次、热启动 10 次；专用机上其他进程固定，若外部
进程造成 swapout，该 run 标记 `inconclusive` 而不是 PASS/FAIL。首发 hard gate：

```text
TurboCider actual_peak <= B
TurboCider unknown_peak == 0
swapouts_delta == 0
quality contract == PASS
```

已有 `swap.used` 不为零不自动失败，但必须写入起点；系统 swapout 为零也不能掩盖 TurboCider 自身
`actual_peak > B`。性能报告同时给 median/P95、time-to-first-step、exposed I/O wait、SSD bytes
per frame、slot refill count 和 drain tail，避免只报平均 wall。

### 193.5 建议的本机命令清单

```sh
env MLX_ROOT=<mlx-root> TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
python3 tests/native/test_memory_probe.py
python3 tests/native/test_memory_manifest.py
python3 tests/native/test_memory_plan_compiler.py
python3 tests/native/test_memory_accounting.py
python3 tests/native/test_memory_scheduler.py
python3 tests/native/test_memory_execution.py
python3 tests/native/test_contract.py
```

默认 `make test` 若绑定不存在的 `python3.11`，应使用本机可用解释器显式运行，并在 evidence 中把
后续既有 `video writer start failed` 等环境失败与 memory 测试分开记录。任何“全部通过”的摘要都必须
列出 PASS、fixture SKIP、no Metal SKIP 和环境 FAIL 的数量。

## 194. 发布判定、回滚和未决风险

### 194.1 Release gate

候选只有在下列条件全部满足后，才能把 `release_enabled` 从 false 改为 true：

```text
[ ] exact sidecar/manifest 已绑定 model root、checkpoint digest、runtime revision、device family
[ ] required allocation sites 全部有 closure；unknown_bytes 在真实运行中始终为零
[ ] plan/live interval 与实际 completion/lifetime 一致，且 actual_peak <= B
[ ] 所有 GPU/host/helper completion 有可消费 token，finish_success 前 mailbox 为空
[ ] cancel/fault/read error/GPU error 的 cleanup 和 quarantine 已通过
[ ] disabled 路径零 probe/ledger/trace/clear/unload/线程额外调用
[ ] L2 真实 Metal 证据存在；无 device 只能 SKIP
[ ] L3 低内存重复运行 swapouts_delta==0、质量通过、性能达到 candidate 自己的目标
[ ] evidence digest、manifest digest、plan digest 可重放
[ ] registry record 经过独立 review，且不是由 request/env/synthetic fixture 生成
```

在这些条件满足以前，production registry 保持空；请求可返回“route available / plan only”，但不能
进入真实 constrained execution。这是 fail-closed 的预期结果，不是功能缺失的临时 workaround。

### 194.2 回滚策略

每个 candidate、backend revision、manifest revision 和 registry record 都应可独立回滚：

```text
发现质量/峰值/lifetime 回归 -> 删除或禁用对应 record
发现 scheduler bug -> 关闭 constrained route，disabled 路径继续工作
发现单一 shape bucket 失败 -> 只禁用该 bucket，不影响其他 bucket
发现 helper 泄漏 -> 禁用包含 helper 的 candidate，保留纯 GPU candidate
```

回滚不能通过把 `release_enabled=false` 后仍让 runtime 忽略 registry miss 来实现；registry miss 必须
真正阻断 constrained allocation。也不能回滚到系统 swap 兜底后继续声称满足 Y/B 合同。

### 194.3 仍需在实现前确认的风险

以下问题需要在真实设备或对应 runtime 源码审查中确认，当前设计不假设它们已经解决：

1. Metal command buffer 的 completion handler 是否覆盖所有 private/shared backing 的最后一次使用；
2. MLX/MPSGraph 是否存在无法在 clean baseline 后稳定复现的 temporary watermark；
3. unified memory 下 process footprint 与 GPU allocatedSize 的重叠统计如何避免双计费或漏计费；
4. LTX helper 的共享文件页、pipe buffer 和 decoded frame 的 process-tree 统计是否足够保守；
5. H3/LTX 所有 required allocation site 是否真的经过当前 hook，尤其是异常和 warm-cache 分支；
6. 低物理内存机器上的系统 compressor 是否会在 swapout 为零时仍造成不可接受的长尾；
7. 不同 macOS/Metal/runtime revision 是否需要新的 `device_family` 或 `runtime_revision` record。

风险未闭环时的处理原则仍是：缩小 candidate scope、增加 manifest upper、禁用 overlap 或直接拒绝，
而不是放宽硬上限。这样才能在内存充足机器上保持默认性能，同时让低内存机器得到可解释、可测量、
可回滚的 constrained 行为。

## 195. 当前源码差距审计：下一轮应该修改什么

本节基于当前仓库已有 scaffolding，列出“现状、风险、目标修改、最小测试”。它比架构图更接近下一轮
施工清单；未完成项不能因为已有类型或测试文件存在就标记为 production-ready。

### 195.1 Runtime 核心差距

| 当前文件/符号 | 当前行为 | 风险 | 建议修改 | 最小验收 |
|---|---|---|---|---|
| `memory_execution.cpp::checkpoint` | 读取 `MemoryPressureDecision`，只消费 `abort_request` | `stop_optional_prefetch`、`drain_completions` 没有传给 pager/adapter | context 保存原子/owner-thread pressure hints，并在 safe point 调用 adapter `apply_memory_pressure()` | Tight 后 optional prefetch 计数立即停止；Critical 不再提交新 read/GPU command |
| `memory_scheduler.cpp::observe` | 85%/80% 与连续 3 次恢复写死 | threshold 变化不会进入 schedule identity/evidence | 把 threshold 封装为 `MemoryPressurePolicy`，绑定 runtime revision；首发仍使用固定默认值 | policy digest 改变时 candidate record 不匹配；边界值单测 |
| `MemoryStageScheduler::pending_` | 普通 `std::vector`，`complete()` 直接修改 | Metal callback 线程直调时存在数据竞争和锁反转风险 | owner-thread mailbox；callback 只 push token，owner 消费并调用 ledger | TSAN/并发 fake completion；旧 generation 不释放新 lease |
| `MemoryStageLease::~/reset` | `active_.clear()` 同步 release | 异步 GPU last-use 的 backing 若未显式 `retire_all`，可能提前释放 | required async site 在析构时若仍 active，标记 lifetime violation/taint；同步 site 才允许直接 release | 忘记 retire 的 synthetic async stage 必须失败并 quarantine |
| `MemoryExecutionContext::finish_success` | 检查 reservation/pending/unknown | 未显式拒绝 cached/active storage、mailbox、I/O ticket、helper | 加 `storage_count==0` 或“仅允许 reviewed retained cache”；检查 mailbox/I/O/helper 全空；检查最终 observation | 每种遗留资源各一个负向测试 |
| `MemoryExecutionContext::finalize_failure` | 一次判定后把 context 标记 finished | `NeedsGpuDrain` 后无法在同一 context 正式重新评估为 Clean | 拆为 `begin_failure_cleanup(reason)` 和 `complete_failure_cleanup(drain_result)`；只有第二步结束生命周期 | pending -> drain success -> Clean；drain timeout -> Quarantine |
| `authorize_memory_capability` | `framework_upper_bytes` 调用处当前传默认 0 | context、trace、MLX/MPSGraph temporary 可能漏计 | 从 capability record/manifest 读取 validated framework upper，禁止 request/env 设置 | framework upper 缺失时 candidate 拒绝；非零值参与 peak/live set |
| `production_memory_capability_registry` | 空 registry | 真实 constrained 请求 fail-closed | 在 L2/L3 和 review 完成前保持为空 | synthetic record 只能测试注入；release binary registry size 仍为 0 |

### 195.2 Session/API 差距

| 当前文件/符号 | 当前行为 | 目标修改 |
|---|---|---|
| `session.hpp::probe_memory_capability` | 单一可选 probe | 增加 `probe_memory_capability_variants()`；默认 wrapper 把 singular 结果包装成单元素，保证迁移兼容 |
| `session.hpp::drain_memory_completions` | 默认同步空结果 | constrained adapter 必须 override；capability manifest 包含 async site 时禁止使用默认实现 |
| `c_api.mm::prepare_memory_execution` | preflight 后 clean session，随后 baseline/plan | 保持顺序；未来多 candidate 时 Phase A 授权全集，Phase B clean 后选最优，不重复读取权重 |
| `c_api.mm::finalize_memory_failure` | disposition 非 Clean 即 quarantine | 接入两阶段 drain；只有 drain 后仍不 clean 才 quarantine，且保留原始 failure 与 drain failure |
| `ScopedMemoryExecutionBinding` | enabled context 才绑定 admission/context | 保持；增加 binding generation，unbind 前必须 drain 或显式 quarantine |
| `results.mm` | 已有 memory admission 基础字段 | 补 candidate/manifest/plan/revision、actual peak、unknown peak、swap delta nullable、overlap 指标 |

API 的异常处理不要用嵌套 catch 吞掉 cleanup failure。建议保存两个错误：

```cpp
struct MemoryFailureReport {
    std::string primary_failure;  // generate/read/allocation/cancel 的原始错误
    std::string cleanup_failure;  // drain/reap/release 的错误
    MemoryFailureDisposition disposition;
};
```

返回给用户时以 `primary_failure` 为主，metrics/evidence 同时记录 `cleanup_failure`。cleanup failure
不能覆盖原始错误，也不能被忽略；它通常决定 worker 是否 quarantine。

### 195.3 H3/LTX 当前 drain 的具体修改

`H3Session::drain_memory_completions()` 与 `LTXSession::drain_memory_completions()` 当前主要比较
ledger/scheduler pending count；这能发现“已有 pending 未清空”，但不能主动等待底层 command buffer，
也不能证明没有尚未登记的 GPU work。目标实现至少需要：

```cpp
MemoryDrainResult H3Session::drain_memory_completions(
        MemoryExecutionContext &context,
        std::chrono::milliseconds timeout) {
    auto deadline = Clock::now() + timeout;
    while (has_submitted_gpu_work()) {
        if (!wait_next_metal_completion(deadline))
            return {false, completed, pending_count(), "timeout"};
        context.scheduler().consume_completion_mailbox();
    }
    context.scheduler().consume_completion_mailbox();
    return validate_clean_pending_state(context);
}
```

LTX 还要同时等待 helper/upsampler/VAE 的异步边；H3 要覆盖 schedule readback、DiT block 和 VAE
decode。若底层函数本来就是同步的，adapter 仍应从底层返回 `submitted/completed` sequence 或 event id，
让 evidence 能证明“同步完成”，而不是把 pending count 为零当作唯一证据。

### 195.4 Probe/sidecar 后续差距

当前 probe 已有 model-root 限制、fd 级读取、`fstat` 前后快照、checkpoint SHA-256 和完整 candidate
字段比对。下一步测试和设计仍应补：

1. 对 backend、dtype、model variant、sampler、tiling 分别做独立 mismatch 测试，避免一个综合 case
   无法证明每个字段都参与比较；
2. `ExternalMutable` trust 下明确禁用跨请求 hash cache，或将 cache key 绑定 dev/inode/ctime/mtime/
   size 并在使用前重新校验；
3. sidecar 自身增加 canonical content digest/signature 字段时，digest 必须覆盖除 self-digest 外的全部
   security-relevant 字段；
4. sidecar schema revision、runtime revision、backend revision 任一变化都使旧 registry record 失效；
5. probe 的 host allocation 也设上界，尤其是 JSON object count、nested depth 和 canonical sort temporary。

这些修改仍必须保持 probe metadata-only：不 load 模型、不创建 Metal buffer、不 compile graph、
不 unload session、不 clear cache。

### 195.5 Model adapter 的最短闭环顺序

下一轮不要同时实现四个模型的 block pager。建议最短路径：

```text
1. H3：枚举并关闭所有 required GPU/host allocation site
2. H3：真实 Metal completion + 双槽 pager L2
3. LTX：复用 runtime token/mailbox，关闭 stage1/stage2/VAE/helper site
4. LTX：真实 Metal/helper completion L2
5. Flux：只做 component-staged F1
6. Z-Image BF16/GGUF：分别做 component-staged F1
7. H3/LTX L3 通过后，再评估 Flux/Z-Image layer-group pager
```

原因是 runtime token/mailbox 和 completion 语义应先由 H3 一个 candidate 验证；LTX 可复用同一基础设施，
但增加双 denoiser 和 helper。Flux/Z-Image 依赖 MLX lazy/runtime envelope，先做 component-staged 能减少
同时变化的维度。这个顺序是风险收敛，不改变各模型最终都可支持 constrained 的目标。

## 196. 实现 PR 的 Definition of Done

每个 constrained-memory PR 在合并前必须回答以下问题；不能只写“unit tests passed”：

```text
Scope
[ ] 本 PR 新增/改变了哪些 candidate、stage、site、slot 或 schedule revision？
[ ] 是否改变 manifest/plan/candidate digest？为什么？

Safety
[ ] 所有真实 allocation 是否发生在 reserve 成功之后？
[ ] actual allocated size 是否 <= upper，alignment/隐式副本是否计入？
[ ] GPU last-use 是否由 completion 而不是 CPU scope 证明？
[ ] cancel/exception 是否能反向清理，无法证明时是否 quarantine？

Compatibility
[ ] disabled 路径是否保持零 probe/ledger/trace/额外 unload/clear/drain/线程？
[ ] 既有 backend/residency/cache key/输出是否不变？

Evidence
[ ] L0/L1 的命令和 PASS/SKIP/FAIL 数量是否完整列出？
[ ] 若声称 L2，是否来自真实 Metal device，而不是 fixture 或 no-device SKIP？
[ ] 若声称 L3，是否包含 process-tree、swapout、quality、median/P95？

Release
[ ] production registry 是否保持不变？如改变，是否有独立 evidence review？
[ ] rollback 是否只影响该 candidate/record，不影响 disabled 默认模式？
```

任何一项无法回答，就把 PR 定义为 scaffolding、instrumentation 或 plan-only，而不是 constrained
production support。文档状态也应同步更新，避免“已有 route 名称”等同于“已可安全执行”。

## 197. 本轮增补：从设计合同到可直接施工的接口级方案

本节是对前面 1–196 节的实现者版补充。前文回答“为什么采用显式 streaming/loading/offloading、预算
如何定义、哪些模型先做”；本节进一步固定“哪一层负责什么、每个 API 在哪个线程调用、哪些字段必须进入
digest、失败时如何退出”。如果本节与更早的建议性文字冲突，以本节的 fail-closed 规则和当前源码的真实
状态为准。

当前工作树仍然处在 `scaffolding + hook_bridged + plan_only` 阶段：

```text
已落地：MemoryLedger/Admission、live-interval planner、site validation、generation/token/mailbox、
        H3/LTX 部分 reserve/commit hooks、failure disposition、metadata-only probe 接线
尚未闭环：真实 Metal command completion、所有 required site closure、MLX/MPSGraph temporary envelope、
          Flux/Z-Image constrained adapter、真实设备 L2、低内存 L3、reviewed production registry
```

因此新增接口先以“建议 ABI/实现模板”出现，不能被解读为已经具有生产执行能力。任何测试 fixture 生成的
record 只能注入测试 registry，不能写入 `production_memory_capability_registry()`。

### 197.1 分层职责：control plane 与 data plane 必须分离

实现中最容易发生的错误，是让模型 adapter 自己决定预算、让 callback 线程直接改 ledger，或者让
`MemoryLedger` 反过来调用 Metal/MLX。正确的职责边界如下：

| 层 | 允许做的事 | 禁止做的事 | 主要文件 |
|---|---|---|---|
| Control plane | 解析请求、probe、registry match、编译 plan、选择 revision、输出 evidence | 创建模型 tensor、触发 GPU、在 registry miss 时放行 | `native/runtime/memory_policy.*`, `memory_manifest.*`, `memory_plan.*`, `c_api.mm` |
| Admission/ledger | 记录 reservation、storage、pending release、unknown、peak | 等待 GPU、读文件、调用 `mx::eval`、决定模型算法 | `native/runtime/memory_accounting.*` |
| Scheduler | stage/slot 生命周期、pressure 状态、completion token 消费 | 在 callback 线程执行阻塞 wait 或分配 backing | `native/runtime/memory_scheduler.*` |
| Pager/adapter | 按已授权 site 做有界 read/upload/submit，提交 completion token | 修改 Y/X/B、生成未认证的 fallback、绕过 reserve | `h3_runtime/*`, `ltx_runtime/*`, 未来 `flux2/*`, `z_image/*` |
| Observation/evidence | 采样 footprint、swap、I/O、trace，标记 PASS/SKIP/FAIL | 用观测到的低峰值反向扩大预算或替代 manifest upper | `memory_probe.*`, `results.mm`, `tools/native/*` |

control plane 可以创建 `CompiledMemoryPlan`，但只有在 Phase A registry match 和 Phase B clean baseline
之后，data plane 才能创建第一个真实 backing。这样可以证明“拒绝发生在有副作用之前”。

### 197.2 建议补充的公共类型（保持旧 ABI 兼容）

以下类型建议先放在 `native/runtime/memory_scheduler.hpp` 或独立的
`native/runtime/memory_schedule.hpp`。它们是 C++ 内部 ABI，不直接暴露给现有 C API；现有字段不删除，
新字段增加 revision 后再进入结果 JSON。

```cpp
struct MemoryPressureHints {
    MemoryPressureState state = MemoryPressureState::Normal;
    bool stop_optional_prefetch = false;
    bool stop_new_reads = false;
    bool stop_new_gpu_submissions = false;
    bool request_drain = false;
    uint64_t observed_committed_bytes = 0;
    uint64_t observed_budget_bytes = 0;
    uint64_t sequence = 0;
};

struct MemorySlotSpec {
    uint32_t slot_id = 0;
    std::string site_id;
    uint64_t upper_bytes = 0;
    bool optional_prefetch = false;
    bool reusable = true;
};

struct MemoryStageSpec {
    uint32_t stage_id = 0;
    std::string name;
    std::vector<std::string> required_sites;
    std::vector<MemorySlotSpec> slots;
    bool allows_lookahead = false;
    bool requires_gpu_drain = false;
    std::string safe_point_kind;
};

struct MemoryDrainRequest {
    std::chrono::steady_clock::time_point deadline;
    bool wait_gpu = true;
    bool wait_file_io = true;
    bool wait_helpers = true;
};
```

`MemoryPressureHints` 是 owner-thread 快照，不是跨线程可变对象。adapter 每次 safe point 读取一次，
不能保存引用跨越下一个 stage。`MemoryStageSpec` 必须由 manifest/plan digest 产生，运行中只能选择
已存在的 spec；不能根据 pressure 临时 append 一个 site 或 slot。

建议在 `MemoryExecutionContext` 增加以下方法，旧调用保持不变：

```cpp
MemoryPressureHints pressure_hints() const;
void set_pressure_hints(MemoryPressureHints hints); // 仅 owner thread
void checkpoint(const std::string &phase,
                const MemoryDrainRequest *drain = nullptr);
bool request_schedule_revision(std::string_view revision_id);
const std::string &active_schedule_revision() const;
```

`request_schedule_revision()` 只登记请求，不立即切换。真正切换必须由 owner 在
`denoise_block_boundary`、`stage_boundary` 或 `decode_tile_boundary` 完成，并将旧/新 plan digest 和
切换原因写入 trace。

### 197.3 现有接口与拟增接口的兼容规则

当前源码已经有 `ModelSession::bind_memory_context()`、`unbind_memory_context()`、
`drain_memory_completions()` 和 `probe_memory_capability()`。建议按以下顺序演进：

1. 保留 singular `probe_memory_capability()`；新增 `probe_memory_capability_variants()` 默认将 singular
   结果包装成一个元素，避免一次改动所有模型。
2. 保留 `drain_memory_completions(context, timeout)`；新增带 `MemoryDrainRequest` 的内部 helper，旧
   override 由 session wrapper 转换。
3. 保留 `MemoryStageLease::reserve()` 作为测试/兼容 helper；生产 adapter 强制使用
   `reserve_site()`，使 site id 成为类型的一部分。
4. 保留 `MemoryStageScheduler::complete(PendingReleaseId)` 供同步 fake 使用；真实 callback 只能
   `post_completion(MemoryCompletionToken, status, retain)`，不得直接调用 `complete()`。
5. 结果 JSON 增加字段时采用 nullable/optional 语义。旧消费者遇到未知字段必须忽略，现有字段名称和
   类型不可改变。

`MemoryCompletionToken` 的 `value`、`allocator_domain`、`generation`、`pending_release`、
`stage_id`、`slot_id` 共同形成唯一身份；任何字段为零、旧 generation、错误 stage/slot 或重复消费，
都应返回 `memory_lifetime_violation`，而不是静默忽略。

## 198. 精确的 admission、plan 和 schedule 编译算法

### 198.1 输入冻结与单位规范

在 `make_plan()` 之后创建不可变的 `MemoryPolicyInput`（建议只在内部使用）：

```cpp
struct MemoryPolicyInput {
    uint64_t Y_limit_bytes;
    uint32_t X_buffer_percent;
    uint64_t min_free_bytes;
    unsigned requested_slots;
    std::string model_id;
    std::string operation;
    std::string backend;
    std::string dtype;
    std::string variant;
    std::string sampler;
    std::string tiling;
    std::string shape_bucket;
};
```

所有内部计算均以 `uint64_t bytes`、`uint32_t percent`、`uint32_t count` 为单位；禁止在 planner 中使用
`double` 表示 bytes。`B = floor(Y * (100-X) / 100)` 的实现必须先做 checked multiplication：

```cpp
uint64_t budget_after_buffer(uint64_t y, uint32_t x) {
    require(x <= 100, "memory_policy_invalid: buffer percent out of range");
    require(y <= UINT64_MAX / (100u - x),
            "memory_estimate_unknown: budget multiplication overflow");
    return (y * (100u - x)) / 100u;
}
```

实际代码还必须显式处理 `x == 100`、`B == 0`、`Y > physical_memory`、`B < process_baseline`；示例
代码只用于说明顺序。`min_free_bytes` 不从 B 再扣一次，而是在 Phase B 以独立系统 headroom 条件
检查：`system_available >= min_free_bytes + planned_increment`。无法获得可信系统观测时，constrained
请求拒绝，不把“未知”当作足够 headroom。

### 198.2 两阶段准入伪代码

```cpp
ExecutionPlan plan = make_plan(request);                 // no side effect
if (!plan.memory_policy || !plan.memory_policy->enabled)
    return legacy_execute(plan);                          // zero memory hooks

const auto identity = device_identity();
auto candidates = session.probe_memory_capability_variants(plan, identity);
require(!candidates.empty(), "memory_policy_unsupported: no exact candidate");

auto authorized = match_registry_exact(candidates, production_registry());
require(!authorized.empty(), "memory_policy_unsupported: registry miss");

session.unload();                                         // only after Phase A
if (session.uses_parent_mlx(request)) mx::clear_cache();
const auto baseline = observe_process_memory_or_reject();

for (const auto &candidate : authorized) {
    auto compiled = compile_memory_plan(candidate.manifest,
        baseline.process_footprint_bytes,
        candidate.record.framework_upper_bytes,
        plan.memory_policy->effective_budget_bytes);
    if (!compiled.complete || !compiled.fits_budget) continue;
    if (!system_headroom_allows(compiled, baseline, plan.memory_policy)) continue;
    feasible.push_back({candidate, std::move(compiled)});
}
require(!feasible.empty(), "memory_budget_too_small: no certified plan fits");
auto selected = deterministic_best(feasible);                // no new probe/load
auto context = make_context(selected.plan, baseline);
bind_context(session, *context);
```

`deterministic_best()` 的排序键固定为：

```text
quality_contract descending
planned_peak ascending
refill_slots ascending
exposed_io_wait_estimate ascending
candidate_key.digest ascending
```

排序不能使用 wall-clock、随机数或本次观测的瞬时低峰值。未来支持多个 candidate 时，所有候选都必须
在同一次 clean baseline 上编译；不能加载第一个候选后再决定第二个候选。

### 198.3 live interval 与 slot 上界的编译规则

每个 `AllocationInstance` 使用半开区间 `[live_begin, live_end)`。编译器按以下顺序处理：

1. 先按 `site_id, instance_id, live_begin, live_end` 排序，保证跨平台确定性；
2. 在同一 epoch 先 release、后 acquire；需要物理重叠的资源必须使用不同 epoch 或不同 alias group；
3. 同一 `alias_group` 只计同时存活资源中的最大 upper，不把可证明复用的 backing 重复计费；
4. `asynchronous=true` 的资源 `live_end` 必须晚于最后一个 GPU/file/helper completion epoch，不能用
   CPU scope 结束替代；
5. 每个 site 记录 `maximum_instance_upper_bytes`、`aggregate_instance_upper_bytes`、
   `instance_count`，运行时 upper 不得超过前者，累计 live 实例也不能超过 manifest 的实例闭包；
6. `UpperProvenance::HeuristicNotExecutable` 可以出现在 plan-only 输出，但只要 required site 使用该
   provenance，registry lookup 后也不得进入真实执行。

建议为每个 stage 编译两个 peak：

```text
peak_without_lookahead
peak_with_certified_lookahead
```

`with_lookahead` 只有在额外 slot、host staging 和 upload backing 都有显式 instance 时才有效。运行中
遇到 Tight pressure 只能从前者和后者之间切换到已认证的低峰 revision，不能直接从 3 slot 改为任意
1.5 slot 或偷偷缩小 upper。

### 198.4 candidate 选择、revision 与 digest

以下字段必须进入 `policy.digest`：

```text
Y, X, B, min_free_bytes, requested/effective slots,
candidate key, checkpoint digest, manifest digest, plan digest,
framework upper, process baseline, schedule revision, runtime/backend revision
```

process baseline 是本次 Phase B 的事实输入；因此相同请求在不同 baseline 上可能得到不同 `plan_digest`，
这是预期的。`manifest_digest` 不能包含 baseline，便于证明 checkpoint/site 结构没有改变；
`plan_digest` 必须包含 baseline 和 budget，便于重放本次准入。

## 199. allocation-site manifest 的编写、生成与闭包证明

### 199.1 site 命名规范

site id 使用稳定的点分隔路径，不允许把地址、线程 id、随机序号或临时 tensor 名称放入 identity：

```text
<model>.<component>.<resource>.<lifetime>.<variant>
```

推荐示例：

```text
h3.dit.refill_slot.block
h3.dit.activation.latent
h3.vae.tile_workspace.decode
ltx.stage1.refill_slot.block
ltx.upsampler.graph_temporary.forward
flux.vae.output_tile.decode
zimage.gguf.dequant_scratch.block
```

同一逻辑 site 的不同 block/tile 用 `AllocationInstance.instance_id` 区分；不要为每个运行时 block
生成不同 site id，否则 registry 无法对模型版本做稳定认证。

### 199.2 manifest 生成器的输入和输出

建议新增 `tools/native/generate_memory_manifest.py` 或等价 C++ 工具，但它只消费已导出的 metadata：

```text
checkpoint index / tensor dtype / shape / byte offset
model config / shape bucket / sampler / tiling
allocator alignment table / validated workspace envelope
adapter/backend/runtime revision
```

它不能 import 模型并执行 forward，也不能通过一次运行的 peak 自动猜 upper。输出必须 canonicalize：

1. site 按 `site_id` UTF-8 字节序排序；
2. instance 按 `site_id, instance_id, live_begin, live_end` 排序；
3. 整数统一十进制，无前导零；
4. 缺省布尔值仍输出；
5. digest 字段不参与自身 digest；
6. 生成器版本、schema、backend revision、runtime revision 都写入 sidecar。

### 199.3 required closure report

每个 candidate 必须生成一份 closure report，而不是只给 manifest digest：

```json
{
  "candidate": "h3_c_metal_streamed_v1",
  "required_sites": 42,
  "guarded_sites": 39,
  "synchronous_sites": 7,
  "asynchronous_sites": 35,
  "unreachable_sites": 3,
  "unknown_sites": [],
  "unreachable_justification": [
    {"site": "h3.audio.vocoder", "reason": "route rejects audio"}
  ],
  "completion_sources": ["metal_command_buffer", "host_io"],
  "status": "plan_only"
}
```

`unknown_sites` 非空时 status 必须是 `plan_only` 或 `unsupported`。`unreachable_sites` 必须关联到
request validator 的稳定错误/能力条件，不能只写“当前没测到”。closure report 的 digest 应进入
evidence，但不替代真实 L2/L3。

### 199.4 sidecar 与 checkpoint identity

checkpoint digest 只证明文件内容身份，不证明 runtime 会按正确 site 使用它。manifest digest 只证明
site/interval 结构，不证明文件仍然是同一 checkpoint。registry match 必须同时比较：

```text
candidate key + checkpoint digest + manifest digest + backend revision + runtime revision + device family
```

任何一个字段变化，都应导致旧 record miss。`ExternalMutable` 模型根目录禁止无条件跨请求复用 hash
cache；若保留 cache，key 至少包含 dev/inode/ctime/mtime/size，并在使用前重新做 fd 级快照校验。

## 200. 模型适配矩阵与当前源码施工顺序

### 200.1 H3：先完成 C/Metal required closure，再接真实 completion

| 阶段 | 目标 | 主要文件 | 通过条件 |
|---|---|---|---|
| H3-0 | 统一 hook bridge、domain/generation、site 前缀 | `h3_session.mm`, `h3_gpu.{h,m}`, `h3.h` | 所有 constrained allocator 入口可定位 site |
| H3-1 | DiT 双槽 pager 与 host readback | `h3.c`, `h3_dit.c`, `h3_dit_schedule.c` | `unknown_bytes==0`，slot upper 与 manifest 一致 |
| H3-2 | VAE tile/source/target/export closure | `h3_video_vae.c`, `h3_session.mm` | decode 与 export overlap 在 live interval 中可重放 |
| H3-3 | Metal command completion | `h3_gpu.m`, `h3_session.mm` | token 只在真实 command completion 后 post |
| H3-4 | fault/cancel/quarantine | `c_api.mm`, H3 session | read/GPU/cancel 失败 cleanup 结果可预测 |

当前已完成 H3-0 的大部分桥接和若干 H3-1/H3-2 分类；H3-3 仍需把“pending count 变为零”提升为
“对应 command/event 已完成”的证据。`h3_gpu_drain()` 必须返回提交序列、已完成序列、超时和错误，
不能只返回一个布尔值。

建议 H3 C API 最终形态（名称可调整，但语义不变）：

```c
typedef struct {
    uint64_t command_sequence;
    uint64_t completed_sequence;
    uint32_t stage_id;
    uint32_t slot_id;
    int status;
} h3_gpu_completion;

int h3_gpu_wait_completion(h3_gpu_context *, uint64_t sequence,
                           uint64_t deadline_ns, h3_gpu_completion *out);
```

Objective-C bridge 收到 completion 后只调用 `post_completion()`；C++ owner 在 checkpoint/drain safe
point 消费 mailbox。不要让 Metal handler 捕获 `MemoryExecutionContext &` 或 `MemoryStageLease &`，
避免 callback 晚于 context 生命周期。

### 200.2 LTX：五个资源边界、两个 denoiser、多个 helper

LTX 的难点不是 block pager 本身，而是 stage1/stage2、MLX VAE/upsampler、Gemma、direct Metal 和
child/helper 可能在统一内存中重叠。首发 candidate 继续限制为 video-only、固定 shape bucket、无 audio、
无 runtime LoRA 的 `ltx_c_metal_streamed_video_v1`。

施工顺序：

1. `ltx_session.mm` 先把 conditioning、stage1 latent、upsampler graph temporary、stage2 latent、
   decoded host、output 逐一变成 named host sites；
2. `ltx_gpu.m` 的每个 MTLBuffer 经过 site-specific reserve/commit，并把 allocator domain 与 generation
   写入 `StorageId`；
3. `ltx_blocks.c` 只接受已获得的 slot handle，不自行创建额外 scratch；
4. helper/upsampler/VAE 产生 completion ticket，session drain 同时等待这些 ticket 和 GPU event；
5. `mx::clear_cache()` 只能发生在 preflight match 后、context 建立前或已完成 failure cleanup 后，不能
   在 live stage 中作为“回收手段”；
6. 如果无法证明 MLX graph temporary 的 envelope，candidate 降级为 plan-only，而不是把 observed
   temporary 当作零。

### 200.3 Flux：只先做 component-staged F1

当前 `native/models/flux_module.cpp` 仍主要接受 resident/component-staged。首发不做 block-level pager，
先把 `native/models/flux2/pipeline.cpp` 的生命周期拆成：text encoder → transformer → VAE → output。
每一段需要：

```text
component begin -> reserve fixed component envelope -> load/eval -> explicit sync/drain -> release
```

F1 的认证限制：单请求、无 LoRA、固定 dtype、固定 shape bucket、无音频、无并行 batch。只有在 component
stage 的真实峰值和 completion 证据稳定后，才评估 transformer layer-group pager。Flux 的 `resident` 旧
路径绝不能因 add-on 失败而自动改为“半 streaming”；失败即返回 unsupported/plan-only。

### 200.4 Z-Image：BF16 与 GGUF 必须拆成不同 candidate

`z_image_module.cpp` 当前明确拒绝 component-staged，`z_image_gguf_module.cpp` 也明确拒绝 streaming。
因此首发方案是：

```text
zimage_bf16_component_staged_f1
zimage_gguf_component_staged_f1
```

两者分别建立 manifest、dequant/scratch upper、dtype/quantization revision 和质量 contract。GGUF 的
dequant scratch 不能复用 BF16 的 weight upper；如果 GGUF runtime 没有 allocator hook，candidate 只能
保持 unsupported。任何“请求写 `residency=streamed` 就绕过 module validator”的改动都属于安全回归。

## 201. MLX/Metal 统一内存的特殊实现约束

### 201.1 三种数字不能混为一个 `memory_bytes`

每个结果和 trace 至少同时保留：

```text
ledger_committed_bytes     # 已知 reservation + storage + process baseline
process_footprint_bytes    # OS/process-tree 观测值
gpu_allocated_bytes        # Metal 可取得时的 device allocation 观测
```

Apple unified memory 上这三个数可能重叠，也可能观察窗口不同。planner 用 manifest upper，ledger 用
reservation/lease，OS footprint 用 evidence。不能把 `gpu_allocated_bytes` 再无条件加到
`process_footprint_bytes`；结果中应同时报告来源、采样时间和是否可能重叠。

### 201.2 MLX lazy evaluation、cache 和 graph temporary

MLX 的 lazy array 可能在构造时没有立即 materialize，在 `mx::eval` 或隐式同步时才产生 backing。因此：

1. reserve 必须发生在触发 `eval`/compile 前；
2. graph compile temporary 作为独立 `CompileTemporary` site 进入 manifest；
3. `mx::clear_cache()` 只能减少 reviewed cache，不能替代 unknown allocation 的 accounting；
4. 因为一个 API 调用可能产生多个隐式 buffer，单一 output tensor 的 upper 不足以证明峰值；
5. 若无法通过 runtime hook 拿到 temporary 的最后使用 completion，必须增加 `framework_upper` 并在
   registry 中绑定 runtime revision；无法建立 validated envelope 时拒绝。

### 201.3 mmap、文件 cache 与显式 pager 的关系

使用 mmap 不等于 streaming，也不等于 zero-copy。manifest 必须区分：

```text
file_mapping_bytes       # 映射地址空间/文件页，不直接等于 resident bytes
materialized_tensor      # 真正参与 compute 的 backing
host_staging             # read/dequant/convert 的显式临时 backing
```

首发 pager 建议使用有界 `pread` 或等价 API，将 read buffer 纳入 `HostStaging/RefillSlot`。若使用 mmap，
仍须限制同时 materialize 的 block 数量，并在证据中记录 page-in 行为；不能用“文件映射很小”作为满足
`actual_peak <= B` 的证明。

## 202. 并发、生命周期和取消的强不变量

### 202.1 owner-thread 模型

一个 admitted request 只有一个 scheduler owner，通常是 generate/prepare 所在线程。Metal、I/O 和 helper
线程是 producer，不是 owner：

```text
producer callback -> bounded mailbox -> owner safe point -> token validation -> ledger transition
```

owner 退出前必须执行：`adapter drain -> mailbox drain -> session unload -> ledger snapshot -> finish`。
任何 producer 仍然可能投递消息时，context 不得析构；无法停止 producer 时 worker quarantine。

### 202.2 状态机

```text
NEW -> PREFLIGHTED -> BASELINE_CAPTURED -> ADMITTED -> RUNNING
RUNNING -> DRAINING -> SUCCEEDED
RUNNING -> FAILING -> (NEEDS_GPU_DRAIN | CLEAN | QUARANTINED)
NEEDS_GPU_DRAIN -> DRAINING -> CLEAN
NEEDS_GPU_DRAIN -> DRAIN_TIMEOUT -> QUARANTINED
```

`finish_success()` 只允许 `DRAINING` 且所有计数清零时调用；`finalize_failure()` 不能把仍有 pending
的 context 直接伪装成 finished-clean。当前实现中的 `finished_` 是兼容字段，后续应增加内部 enum
状态，避免 `NeedsGpuDrain` 后不能继续记录 cleanup 结果。

### 202.3 cancel/error 的顺序

取消并不是立即释放 GPU backing 的许可。正确顺序是：

1. owner 设置 `cancel_requested`，停止新 prefetch/read/submit；
2. 等待已经提交的 GPU/file/helper completion，或到 deadline；
3. 消费 mailbox，释放 pending lease；
4. 释放同步 active lease、drop reviewed cache；
5. 做最终 snapshot；
6. clean 则 worker 可复用，否则 quarantine。

GPU error、I/O error、allocation failure 和用户 cancel 都要保留 primary reason；drain/release 错误写入
`cleanup_failure`，不能覆盖 primary reason。任何 token generation 不匹配都视为 lifetime violation，
不能降级成普通 budget miss。

### 202.4 锁与阻塞调用

锁顺序继续固定为：

```text
engine -> execution -> scheduler -> ledger
```

持有 scheduler/ledger mutex 时禁止调用：Metal wait、`mx::synchronize()`、`mx::eval()`、文件 I/O、
`session.unload()`、helper wait、用户 callback。completion mailbox 满时也不能拿普通 heap 无限扩容；
直接进入 quarantine，保留 overflow evidence。

## 203. API 错误、结果协议和可观测性

### 203.1 稳定错误分类

对外错误字符串保持已有 prefix；内部可以再有 machine-readable code：

| code | 含义 | 是否可重试 | 是否 quarantine |
|---|---|---:|---:|
| `memory_policy_invalid` | Y/X/slots/字段类型非法 | 否，修请求 | 否 |
| `memory_policy_unsupported` | route、registry、runtime 或 device 不匹配 | 换 candidate/设备 | 否 |
| `memory_budget_too_small` | plan 或 reservation 超出 B | 调高 Y/降低 shape | 否 |
| `memory_estimate_unknown` | required upper、framework envelope 或 site 不可证明 | 需要新 manifest | 否 |
| `memory_pressure_abort` | 进入 Critical、系统 headroom 不足或观测越线 | 可换更小预授权 plan | 视 cleanup |
| `memory_lifetime_violation` | token、completion、storage、generation 违反不变量 | 不应直接重试 | 通常是 |
| `memory_cleanup_failed` | drain/release 后仍有资源 | 不复用当前 worker | 是 |

### 203.2 结果 JSON 最小增量

在既有 `memory_admission` 下建议增加以下 nullable 字段，不改变已有字段类型：

```json
{
  "policy_revision": "memory-policy-v1",
  "schedule_revision": "h3-s3-p2-r1",
  "candidate_key_digest": "...",
  "manifest_digest": "...",
  "plan_digest": "...",
  "budget_bytes": 14602888806,
  "user_limit_bytes": 17179869184,
  "buffer_percent": 15,
  "planned_peak_bytes": 13254000640,
  "actual_peak_bytes": 13190123520,
  "unknown_peak_bytes": 0,
  "swapouts_delta": null,
  "gpu_completion_evidence": "metal-sequence-verified",
  "overlap_ratio": 0.71,
  "exposed_io_wait_ms": 184,
  "drain_tail_ms": 9,
  "failure_disposition": "none"
}
```

没有可用 swap 计数时输出 `null`，不能输出 0。`actual_peak_bytes` 的定义必须在结果中固定为“本次
request 期间 ledger committed peak 与可信 OS observation 的保守合成值”，并同时记录各来源，避免
不同 backend 用不同口径比较。

### 203.3 trace 事件和采样频率

必需事件：`preflight_start/end`、`baseline_capture`、`plan_compiled/rejected`、`stage_begin/end`、
`site_reserve/commit/cancel`、`slot_read/upload/submit/completion`、`pressure_transition`、
`revision_requested/changed`、`memory_checkpoint`、`failure_disposition`。

每个事件至少带：`request_id`、`engine_generation`、`candidate_key_digest`、`manifest_digest`、
`plan_digest`、`stage_id`、`slot_id`、`site_id`、`allocator_domain`、`generation`、`timestamp_ns`。
prompt、文件绝对路径和 LoRA 内容只写脱敏 digest。trace 必须有容量上限；overflow 是 evidence，不能
悄悄丢弃后声称 trace 完整。

建议采样策略：

```text
每次 reserve/commit/retire/completion：事件级，不能采样丢弃
process footprint：stage boundary + 每 250ms 至多一次
system pressure/swap：stage boundary + failure/cancel + campaign runner 定时采样
```

## 204. 测试矩阵：从纯逻辑到低内存实机

### 204.1 L0 纯 C++ 单元测试

必须新增或补齐以下向量：

```text
Y=0、X=100、Y*percent overflow、baseline>B、framework>B-baseline
site 缺失、class mismatch、per-instance upper 超限、重复 instance、空 lifetime
alias group 复用、same-epoch release/acquire、async live_end 太早
completion 重复、旧 generation ABA、错误 allocator domain、错误 stage/slot
mailbox overflow、pending 未 drain、cached/storage/mailbox 残留
finalize_failure: clean / needs_drain / drain_timeout / quarantine
```

每个负向测试都应断言稳定 prefix 和 ledger 最终状态，避免只测试抛异常而没有验证资源是否泄漏。

### 204.2 L1 默认路径零行为测试

建议在 `ModelSession` seam 中加计数器或 fake hooks，disabled request 必须满足：

```text
probe == 0, manifest_parse == 0, checkpoint_hash == 0,
ledger_construct == 0, scheduler_construct == 0,
unload_extra == 0, clear_cache_extra == 0, synchronize_extra == 0,
drain == 0, trace_event == 0, worker_thread == 0
```

执行 A/B/B/A 交错基准，至少覆盖 H3、LTX、Flux、Z-Image 的默认 resident/component-staged 路径。门槛为
质量/输出逐字节或容差等价、线程数量不变、cache key 不变、steady-state wall median 回归不超过 2%。

### 204.3 L2 真实 Metal 测试

每个已认证 candidate、shape bucket、slot 数量至少执行：冷启动 3 次、热启动 10 次、最大 shape 3 次。
故障注入点包括第 N 个 block read、upload、GPU submit、completion、decode tile 和 export。通过条件：

```text
planned_peak <= B
actual_peak <= B
unknown_peak == 0
finish_success 前 reservation/pending/storage/mailbox/helper/I-O ticket 全部清零
真实 Metal completion sequence 证据存在
quality contract、输出 shape、fps、采样步数保持不变
```

无 Metal device 时只能输出 `SKIP: no Metal device is available`，不能折算成 PASS。

### 204.4 L3 低内存 campaign

在 16/24/32 GiB 机器上固定外部进程、macOS 版本、runtime revision 和输入 shape。每个组合记录：

```text
vm_stat swapins/swapouts delta
compressor pages
system available min
process/process-tree footprint
ledger planned/actual/unknown peak
slot refill count、exposed I/O wait、drain tail、median/P95 wall
quality metrics
```

外部进程导致 swapout 的 run 标记 `inconclusive`。TurboCider 自身 `actual_peak > B` 或
`unknown_peak > 0` 直接 FAIL；即使系统 swapout 为零也不能覆盖这两个失败。性能目标按 candidate 单独
定义，至少报告相对默认路径的 TTFS、每帧 SSD bytes、P95 和失败率。

### 204.5 fault sequence 与重复运行

单点故障不足以证明 cleanup。runner 应执行序列：

```text
success -> cancel -> success
read_fail -> new_request
gpu_fail -> new_request
helper_timeout -> new_request
mailbox_overflow -> process recycle
```

`new_request` 若复用同一 worker，必须证明旧 generation 的 token 不会释放新请求 backing；quarantine
路径则必须证明 worker 不再接收新请求。所有 run 归档 `primary_failure` 与 `cleanup_failure`。

## 205. 性能、overlap 和调优方法

### 205.1 不以“slot 越多越快”为默认目标

slot 数量增加会同时增加：refill backing、host staging、upload in-flight 和 completion 复杂度。调优
目标应是最小满足 `read_end <= compute_deadline` 的 slot 数，而不是追求最大 slot：

```text
slots=1：基线串行，验证正确性
slots=2：首发 overlap candidate
slots=3：只有在 manifest/evidence 证明 slot-3 不越界且 I/O 有收益时认证
```

每次改变 slot、pinned prefix、tile 或 prefetch distance 都要提升 `schedule_revision`，重新编译 plan 并
使旧 registry record 失效。不能只改常量而复用旧 evidence。

### 205.2 overlap 指标定义

建议统一计算：

```text
compute_window_ms = GPU compute start -> GPU compute end
io_overlap_ms = overlap(read/upload, compute_window)
overlap_ratio = io_overlap_ms / max(compute_window_ms, 1)
exposed_io_wait_ms = time compute queue blocked by missing slot
```

`overlap_ratio` 只用于性能分析，不参与硬安全判定。若 I/O 慢，优先降低 look-ahead 或切换已认证 serial
revision；增加第四槽属于新的 candidate，不能作为运行时自适应应急方案。

### 205.3 热缓存与公平比较

报告必须区分：

```text
cold: 新进程、空 allocator/cache、首次 checkpoint access
warm: 同一进程、相同 candidate、连续请求
reused-session: 只有默认 disabled 路径允许；constrained 首发不继承未记账 session
```

constrained 与默认路径比较时，分别给出 cold/warm，不要把默认路径的长期 warm cache 优势与 constrained
的 clean baseline 混为单一平均值。若未来认证 retained cache，cache 必须也有 site/lease/eviction 证明。

## 206. 分阶段落地、feature gate 与回滚

### 206.1 推荐提交顺序

```text
P0  文档/错误码/结果字段（无行为变化）
P1  runtime ledger + token/mailbox + 逻辑测试
P2  metadata-only probe + manifest closure tooling
P3  H3 reserve/commit 全 required site（仍 plan-only）
P4  H3 真实 completion + 双槽 L2
P5  LTX host/GPU/helper closure + L2
P6  Flux component-staged F1
P7  Z-Image BF16/GGUF 分离 candidate
P8  L3 低内存 campaign、review-only registry
P9  独立 review 后才允许 release_enabled=true
```

每个 PR 都必须只推进一个 capability level，避免一个大 PR 同时改变 planner、adapter、registry 和结果
协议。`production_memory_capability_registry()` 在 P8 之前保持空。

### 206.2 gate 设计

建议 gate 的优先级为：

```text
TC_MEMORY_CONSTRAINED_REQUEST=0/1 仅用于本地构建验证，不授予生产 capability
registry release_enabled 决定是否允许真实 constrained execution
candidate/shape/slot 级 record 决定精确范围
worker quarantine 决定单进程是否继续接收请求
```

环境变量、请求字段和 profile 只能选择已存在的 route/candidate，不能改变 `framework_upper`、verified
slots、quality contract 或 `release_enabled`。本地实验若使用 synthetic record，结果必须包含
`experimental=true`，并禁止上传为 release evidence。

### 206.3 回滚动作

```text
单 candidate 失败：禁用该 record，其他 candidate 不受影响
某 schedule revision 失败：只禁用 revision，保留旧 revision（若其 evidence 仍有效）
runtime token/mailbox bug：关闭整个 constrained route，disabled 路径不变
worker cleanup 不可信：隔离并回收 worker，不尝试“继续用 swap 跑完”
```

回滚操作要记录 operator、时间、record digest、原因和恢复条件。删掉 release record 后，registry miss
必须真的阻止执行；不能仅把结果字段改成 `release_stable=false` 仍继续分配。

## 207. 施工验收清单（实现者 handoff 版）

### 207.1 Runtime patch 完成条件

```text
[ ] reserve -> allocate -> commit 的顺序可由测试 seam 证明
[ ] 所有 async lease 都有 completion token，callback 不直改 ledger
[ ] generation/domain/stage/slot 校验覆盖成功、取消、异常、重复 completion
[ ] mailbox 有界，overflow 进入 quarantine，不使用无界 heap 扩容
[ ] finish_success 前 reservation/pending/storage/cache/unknown/mailbox/I-O/helper 全清零
[ ] failure cleanup 能保留 primary_failure 与 cleanup_failure
[ ] context 析构不会静默吞掉 lifetime violation
```

### 207.2 Model adapter 完成条件

```text
[ ] manifest 中 required site 与代码调用点有逐项映射
[ ] 每个未接 hook 的分支都有 validator unreachable 证明
[ ] Metal/MLX/host/helper backing 都有 class、upper、lifetime、completion 来源
[ ] 实际 completion 不是仅凭 pending count 推断
[ ] shape、dtype、sampler、tiling、LoRA、audio 等候选字段都参与 key equality
[ ] 失败路径不会遗留 command、file read、helper、cache 或 decoded output
```

### 207.3 Evidence 完成条件

```text
[ ] L0/L1 命令、版本、PASS/SKIP/FAIL 原始摘要已归档
[ ] L2 仅来自真实 Metal device，包含 completion/peak/quality/fault 证据
[ ] L3 包含 swapouts_delta、process-tree、compressor、性能 P50/P95 和质量
[ ] manifest/plan/evidence digest 可重放
[ ] disabled ABBA 回归在门槛内
[ ] registry record 由独立 reviewer 签核，且不是 synthetic fixture 生成
```

### 207.4 当前下一步（按风险排序）

结合当前工作树，下一轮最值得先做的不是继续扩展更多模型，而是：

1. 已在 2026-09-15 重新构建并运行完整 native/memory test；后续代码改动仍需重复同一命令集；
2. 已补 mailbox overflow、duplicate completion、遗留 cache/storage 的主要负向测试；下一步优先补
   `framework_upper` 非零和运行时 instance/epoch enforcement；
3. H3 adapter 已接入 retire → callback post → owner mailbox drain，下一步是在真实 Metal 设备验证
   command completion sequence 和 late callback；
4. LTX 已接入 `ltx_gpu_drain()` / `ltx_native_drain()` 以及 Stage 2 边界 drain，下一步是完成
   helper/VAE/process-tree envelope 和是否需要 async hook 的结论；
5. 输出 H3/LTX required-site closure report，明确哪些调用点仍属于 plan-only；
6. 只有在真实设备 L2 通过后，才开始 L3 低内存矩阵；
7. 在 L2/L3 和独立 review 之前保持 production registry 为空。

### 207.5 2026-09-15 最新本机验证记录

本轮续修后执行：

```sh
env MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" \
  TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh

python3 tests/native/test_memory_probe.py
python3 tests/native/test_memory_manifest.py
python3 tests/native/test_memory_plan_compiler.py
python3 tests/native/test_memory_accounting.py
python3 tests/native/test_memory_scheduler.py
python3 tests/native/test_memory_execution.py
python3 tests/native/test_contract.py
python3 tests/native/test_h3_schedule_memory.py
python3 tests/native/test_h3_gpu_memory_hooks.py
python3 tests/native/test_ltx_gpu_memory_hooks.py
```

结果：

```text
native build                         PASS
memory probe tests                  PASS
memory manifest tests               PASS
memory plan compiler tests          PASS
memory accounting tests             PASS
memory scheduler tests              PASS
memory execution tests              PASS
contract                            81 PASS / 1 fixture SKIP
H3 schedule memory                  PASS
H3 GPU memory hooks                 SKIP: no Metal device is available
LTX GPU memory hooks                SKIP: no Metal device is available
```

这组结果只证明 L0/L1 的编译、逻辑合同和无设备 skip 行为；它不证明真实 Metal completion、
`actual_peak <= B`、`swapouts_delta == 0` 或模型质量，因此不能记为 L2/L3，也不能据此向 production
registry 添加 record。后续任何 scheduler、adapter、manifest、completion 或 allocator 修改都必须重跑
同一命令集，并把新的 PASS/SKIP/FAIL 摘要覆盖到 evidence，而不是继续引用本次结果。

## 208. 本轮文档更新后的最终判断

对本项目而言，最合适的产品模式仍是：

```text
默认模式：完全沿用现有本机 residency/cache/调度，零新增 memory scheduler 行为
内存受限 add-on：以 B 为硬预算，显式分层 loading/offloading/streaming，swap 只作观测和失败信号
```

这不是简单地把“内存不够时自动 swap”替换成“多几个 buffer”。可交付的核心是完整的容量合同：
准确的 candidate/manifest、先 reserve 后 allocate、可证明的 live interval、真实 completion、bounded
mailbox、pressure 安全点、可回滚 registry 和可重复 evidence。任何一环缺失，都只能称为
instrumentation、scaffolding 或 plan-only，不能对用户承诺“运行过程中绝不超过 Y 的 X% buffer 后预算”
或“必然不会进入 swap”。

## 209. 2026-09-15 续修：按当前工作树收敛实现边界

前面的章节已经定义了目标架构。本节把目标架构映射到当前工作树，明确哪些部分已经有代码闭环，哪些
部分仍然只能作为 plan-only。后续实现、review 和验收都应以本节为准，避免把“接口存在”误认为“模型
已经获得执行能力”。

### 209.1 当前已落地的真实闭环

当前代码已经具备以下可复用基础：

| 能力 | 当前落点 | 当前状态 | 仍缺什么 |
|---|---|---|---|
| 请求级 add-on 开关 | `MemoryConstrainedConfig`、`memory_policy.cpp`、`plan.cpp` | 已接线 | 还没有 production candidate |
| `Y/X/B` 预算计算 | `effective_budget()`、`EffectiveMemoryPolicy` | 已接线 | `framework_upper` 仍有路径为零 |
| metadata-only capability probe | `memory_probe.mm`、session `probe_memory_capability()` | 已接线 | 需要真实 sidecar 与 reviewer 签核 |
| manifest/interval plan | `memory_manifest.*`、`memory_plan.*` | 已接线 | 仍需逐 allocation-site closure |
| reserve → allocate → commit | `MemoryReservation`、`MemoryAllocationTxn`、H3/LTX bridge | 逻辑闭环 | 需要真实 Metal peak 证据 |
| bounded completion mailbox | `MemoryStageScheduler` | 已接线 | 需真实设备压测 overflow/late callback |
| H3 异步 retire/complete | `h3_gpu_continue_releasing()`、`H3MemoryBridge` | 已接线 | 需真实 Metal command completion |
| H3 runtime drain | `h3_drain()`、DiT/VAE/TAEH3 drain | 已接线 | 需证明所有 helper/缓存边界 |
| LTX deferred command drain | `ltx_gpu_drain()`、`ltx_native_drain()` | 已接线 | 当前主要是 owner-thread drain，尚无通用 async hook |
| success/failure cleanup | `MemoryExecutionContext`、`c_api.mm` | 已接线 | 建议改成显式内部状态机 |
| default disabled 零调用路径 | API 入口提前返回空 context | 逻辑已覆盖 | 需继续做 ABBA 性能回归 |
| production registry | `production_memory_capability_registry()` | **刻意为空** | L2/L3 完成后才允许添加 record |

特别需要强调：当前 H3/LTX 路由虽然能够生成 manifest、编译 plan、安装 allocator hook，但
`preflight_memory_capability()` 仍会要求 registry 中存在完全匹配的 record。因此没有经过真实设备验证的
checkpoint、shape、dtype、slot 数或 runtime revision 不会被“自动降级”为 resident 模式；它们会在
capability preflight 失败，这正是 constrained add-on 的 fail-closed 设计。

### 209.2 当前两条模型路线的有效含义

当前 `resolve_memory_constrained_candidate()` 的语义不是“所有模型都支持低内存”，而是只声明两个
候选入口：

```text
H3:
  candidate        = h3_c_metal_streamed_v1
  backend          = c_metal
  variant          = fl2va_t2v_video_only
  sampler          = h3_distilled_euler_v1
  tiling           = h3_video_vae_tiled_v1
  refill_slots     = 2（即使请求 max_refill_slots=3 也不会偷偷增加）
  non_denoiser     = 4 GiB

LTX:
  candidate        = ltx_c_metal_streamed_video_v1
  backend          = c_metal
  variant          = ltx_2_5_distilled_t2v_video_only
  sampler          = ltx_distilled_euler_v1
  tiling           = ltx_video_vae_helper_v1
  refill_slots     = request.max_refill_slots
  non_denoiser     = 4 GiB
```

这里的 4 GiB 不是测量结果，而是当前候选的保守组件保留量。正式发布前必须把它拆成
`process_baseline + framework_upper + output/helper envelope` 三项，并在 sidecar 中写入来源和版本。
如果某个候选仍依赖“4 GiB 魔数”才能通过 plan，候选应保持 `plan_only`，不得标记为
`EnvelopeValidated` 或 `release_enabled`。

## 210. 从请求到 GPU backing 的强制执行顺序

实现者不得通过“先创建模型，失败后再补记账”的方式接入 constrained 模式。唯一允许的顺序如下：

```text
JSON/profile merge
  -> validate_memory_constrained_request()
  -> make_effective_memory_policy()
  -> resolve_memory_constrained_candidate()
  -> metadata-only probe（不创建 GPU buffer）
  -> registry exact lookup（miss 即拒绝）
  -> parent-MLX synchronize（仅 parent-MLX 路径）
  -> session.unload() + cache boundary
  -> observe_process_memory() 取得 clean baseline
  -> compile_memory_plan(manifest, baseline, framework_upper, B)
  -> validate model-specific runtime sub-budget
  -> bind MemoryExecutionContext
  -> 建立 GPU/host allocator hooks
  -> 每次 reserve(upper) 之后才允许真实 allocate
  -> commit(actual) 后才把 backing 纳入 active lease
  -> stage safe-point / pressure checkpoint
  -> retire_async 或 owner-thread deferred release
  -> backend drain + mailbox drain
  -> unload/cache clear
  -> finish_success() 或 finalize_failure()
```

其中有四个不可交换的边界：

1. `session.unload()` 必须发生在 constrained baseline 采样之前，否则上一请求驻留的模型会被错误地
   当作本请求的 process baseline。
2. `reserve()` 必须发生在 Metal/MLX/Core ML allocation 之前；仅在 allocation 之后更新 ledger
   不能证明峰值，也不能防止 allocator 先把系统推入 swap。
3. callback 只能发布 value-type completion token；不能在 Metal callback 中直接改 ledger、释放
   session-owned 对象或触发下一轮 loading。
4. `finish_success()` 之前必须先完成 backend drain，再消费 mailbox，最后再做 ledger zero-state
   检查。任何 pending/storage/cache/unknown/mailbox 残留都必须令成功路径抛出
   `memory_lifetime_violation`。

### 210.1 建议把 `MemoryExecutionContext` 改成显式状态机

目前 `finished_`、`tainted_` 和 `failure_disposition_` 能覆盖主要路径，但实现继续扩展后容易出现
“已经 finish 但仍可以 drain”或“failure cleanup 被重复调用”的隐式状态。建议增加：

```cpp
enum class MemoryExecutionState : uint8_t {
    Admitted,
    Running,
    Draining,
    Succeeded,
    FailedNeedsDrain,
    FailedClean,
    Quarantined,
};

MemoryExecutionState state() const noexcept;
void begin_running();
void begin_draining();
void mark_success();
void mark_failure(MemoryFailureDisposition, std::string reason);
void assert_zero_state() const;
```

合法转移应固定为：

```text
Admitted -> Running -> Draining -> Succeeded
Running  -> FailedNeedsDrain -> FailedClean
Running  -> FailedNeedsDrain -> Quarantined
Draining -> FailedNeedsDrain -> Quarantined
```

禁止 `Succeeded -> Running`、`Quarantined -> Running`、`FailedClean -> Succeeded`。状态转移本身应在
owner thread 上完成，并在 debug build 记录 phase、thread id、plan digest 和最后一个 allocation site。

## 211. Scheduler 的可实施算法（v1）

### 211.1 预算、slot 和 live interval

对每个候选，planner 先把 manifest 的 instance 转成半开区间 `[acquire_epoch, release_epoch)`。同一
`alias_group` 的区间只计一次最大 upper；不同 alias group 必须按物理重叠相加。计算过程使用 checked
arithmetic：

```text
base = process_baseline + framework_upper
peak(epoch) = base + sum(max(upper(instance)) for every live alias_group)
fits = max(peak(epoch)) <= B
```

`refill_slots` 不应被当作单纯的整数开关，而应显式出现在峰值中：

```text
slot_upper = max(block_upper[i] for i in streamed blocks)
slot_bytes = refill_slots * slot_upper
streamed_peak = pinned_prefix_bytes + slot_bytes + activation_peak
```

当 `adaptive_refill_slots=0` 时，先固定申请请求指定数量的 slot，再选择 pinned prefix；不能因为某一
个 shape 的 block 较小就临时增加 slot。只有经过独立 evidence 的 adaptive candidate 才允许动态减少
slot，且减少动作只能发生在 stage safe-point。

### 211.2 两阶段 admission

第一阶段是静态准入，第二阶段是运行时准入。两者都必须通过：

```cpp
CompiledMemoryPlan plan = compile_memory_plan(..., B);
require(plan.complete && plan.fits_budget,
        "memory_budget_too_small: plan does not fit B");

auto reservation = context.try_reserve_site(site_id, klass, upper);
require(reservation.has_value(),
        "memory_budget_too_small: runtime admission denied");

auto storage = allocate_real_buffer(upper); // reserve 之后
auto lease = reservation->commit(storage_id(storage), actual_bytes(storage));
```

运行时 `actual_bytes` 可以小于 upper，但不能大于 upper；upper 不能由本次 observed allocation 反推。
如果真实 allocator 返回的 capacity 无法取得，必须把该 backing 归类为 `UnknownExternal` 并拒绝执行，而
不是用 `bytes_requested` 代替 capacity。

### 211.3 stage/slot 调度伪代码

以下伪代码适用于 H3 DiT 和 LTX denoiser 的 streamed block 路线：

```text
stage_begin(stage_id)
  reserve activation + conditioning + output envelopes
  reserve all fixed refill slots
  load pinned prefix

for logical_step in steps:
  for block in schedule[logical_step]:
    slot = choose_reusable_slot(block, last_use_epoch)
    wait_until(slot.last_completion <= current_epoch)
    reserve_site(block.site_id, RefillSlot, slot_upper)
    pread_exact(block.offset, slot.host_staging)
    commit(slot.gpu_buffer, actual_capacity)
    encode_block(slot, activation, conditioning)
    retire_async(slot.lease, stage_id, slot.id) // last GPU use only
    enqueue_completion_callback(slot.token)
    if pressure_state != Normal:
      stop optional lookahead
      drain at next safe point
  checkpoint("block_boundary")

drain backend queues
drain owner mailbox
release pinned/activation/output/host leases
stage_end(stage_id)
```

`pread_exact`、Metal command encode 和下一 slot 的预取可以 overlap，但同一个 slot 的生命周期必须满足：

```text
reserve -> host read complete -> GPU commit -> last command encoded
        -> retire -> command completed -> mailbox consumed -> release
```

不能在“开始下一个 block”时直接释放上一个 slot；必须使用最后一次实际 GPU use 的 command completion。

### 211.4 pressure 控制器动作

建议保留当前 `Normal/Tight/Critical` 三态，但把动作写成不可歧义的表：

| 状态 | 进入条件 | 允许动作 | 禁止动作 | 退出 |
|---|---|---|---|---|
| Normal | accounted < 85% B 且 system headroom 足够 | 固定 lookahead、双 slot overlap | 动态增加 slot | 直接继续 |
| Tight | accounted ≥ 85% B 或 headroom 低 | 停止 optional prefetch，减少 lookahead，优先 drain | 新增非必要 cache、扩大 batch | 连续 3 次低于 80% B |
| Critical | allocation fail、observed over budget、system reserve 破坏 | 停止提交新工作，drain 当前 queue，失败当前请求 | 依赖 swap 继续、偷偷切 resident | 不自动恢复；请求结束后重建 |

Critical 必须 sticky 到当前 request。即使后续 observed footprint 下降，也不能在同一个 request 内重新
放行新 allocation，因为失败时可能已经存在未观测的 framework temporary 或 allocator cache。

## 212. H3 适配施工稿：当前代码、manifest 和下一步修改

### 212.1 已接入的 allocation class

H3 GPU C ABI 目前支持：`WEIGHTS`、`ACTIVATION`、`CONDITIONING`、`REFILL_SLOT`、
`CONVERSION_SCRATCH`、`OUTPUT`。host bridge 另外把 conditioning、latent、decoded F32、output、
staging、control 映射到 runtime `MemoryClass`。这比只统计 DiT 权重更严格，因为 H3 的 VAE/解码输出
可能与 DiT 的最后一个 command 同时存活。

建议把 H3 manifest 的 site 命名稳定为以下前缀，并禁止在 release record 中出现未归类的动态字符串：

```text
h3.text.weights.*
h3.text.conditioning.*
h3.dit.pinned.block.<n>.*
h3.dit.refill.slot.<s>.*
h3.dit.activation.*
h3.dit.conversion.*
h3.vae.activation.*
h3.vae.conditioning.*
h3.vae.output.*
h3.host.decoded.*
h3.host.output.*
```

每个 site 至少必须提供：`component`、`stage`、`memory_class`、`lifetime`、`upper_bytes` 来源、
`asynchronous`、`last_use_event`。例如：

```json
{
  "site_id": "h3.dit.refill.slot.0.weights",
  "component": "dit",
  "stage": "denoise",
  "memory_class": "refill_slot",
  "lifetime": "command_buffer",
  "provenance": "exact_metadata",
  "required": true,
  "asynchronous": true,
  "instances": [
    {"instance_id": 0, "upper_bytes": 50331648,
     "live_begin": 120, "live_end": 121, "alias_group": 7000,
     "last_use_event": "dit.block.0.last_kernel"}
  ]
}
```

### 212.2 H3 completion 约束

当前 `h3_gpu_continue_releasing()` 已经采用：

```text
owner thread retire(StorageLease)
 -> scheduler pending-release + completion token
 -> commit Metal command
 -> Metal callback h3_memory_complete() 只 post mailbox
 -> owner thread drain_completion_mailbox()
 -> ledger complete_pending()
```

这条链路必须继续保持。后续不得把 `h3_memory_complete()` 改成直接调用 `ledger.complete_pending()`，
也不得在 callback 中调用 `session.unload()`、`mx::clear_cache()` 或阻塞等待其他 command。

下一步建议在 `h3_gpu.m` 增加 debug-only completion provenance：为每一个 deferred tensor 记录
`stage_id`、`slot_id`、`command_index` 和 `last_use_tag`，在 callback 和 mailbox drain 时比较并输出
不匹配信息。该信息不进入 release key，但可显著降低真实设备上的 ABA 排查成本。

### 212.3 H3 分阶段预算建议

H3 首个 candidate 只支持无 audio、无 input、无 runtime LoRA 的 text-to-video。建议将有效预算拆成：

```text
B_h3 = process_baseline
     + framework_upper
     + text_conditioning_upper
     + dit_pinned_upper
     + dit_refill_slots * slot_upper
     + dit_activation_peak
     + vae_activation_peak
     + decoded/output_upper
```

如果 VAE 和 DiT 无法做到物理不重叠，就必须将两者 upper 相加；只有在 `h3_drain()` 已经等待 DiT
最后一个 command 且 mailbox 完成后，VAE 才可以使用同一 alias group 或复用同一 refill slot。

## 213. LTX 适配施工稿：stage boundary、helper 和 async gap

### 213.1 当前真实边界

LTX 当前已经把 GPU buffer 按 `WEIGHTS/ACTIVATION/CONDITIONING/REFILL_SLOT/CONVERSION_SCRATCH/OUTPUT`
分类，并通过 `ltx_native_drain()` 等待主 GPU 与 audio GPU 队列。constrained video-only 路径在 Stage 2
完成后会：

```text
drain denoiser queue
 -> destroy denoiser/streaming slots
 -> clear streamed helper/cache
 -> release conditioning + unused audio latent host lease
 -> admit Video VAE graph envelope
 -> run VAE/helper
```

这条边界是当前避免 Transformer 与 VAE 峰值重叠的关键。任何将 `denoiser_.reset()` 提前到 drain 之前的
改动都属于生命周期回归。

### 213.2 LTX 当前缺口和建议修改

`ltx_gpu_memory_hooks` 当前只有同步 `release()`，没有 H3 那样的 `retire()/complete()`。批处理路径通过
`deferred_memory_tokens` 在 `ltx_gpu_drain()` 的 owner thread 上释放 token，这对单队列 v1 足够，但有
三个限制：

1. 不能在多个独立 GPU queue 之间表达“最后一个使用者”；
2. callback 失败状态和 token identity 没有统一进入 scheduler mailbox；
3. 如果未来 VAE/helper 与 denoiser overlap，单一 batch drain 不能提供细粒度回收。

建议下一版 ABI 追加尾字段（保持现有 struct_size 兼容）：

```c
int  (*retire)(void *user, void *token,
               uint32_t stage_id, uint32_t slot_id,
               char *error, size_t error_size);
void (*complete)(void *user, void *token, int status);
```

实现顺序与 H3 完全一致：`retire` 仅做 ledger pending transition，`complete` 仅 post mailbox，
`ltx_gpu_drain()` 或真实 Metal callback 不直接修改 ledger。LTX v1 在这个改动完成前可以继续执行
single-queue owner drain，但 candidate 的 `asynchronous` site 必须保持 false；不能在 manifest 中把
当前 deferred release 伪装成 callback-complete。

### 213.3 LTX 五类 resource site

LTX manifest 建议固定包含：

```text
ltx.transformer.pinned.<block>.weights
ltx.transformer.refill.slot.<slot>.weights
ltx.transformer.stage1.activation.*
ltx.transformer.stage2.activation.*
ltx.conditioning.video/audio/mask
ltx.audio_gpu.*                       # 即使 video-only 也要显式为 zero/absent
ltx.video_vae.graph_envelope
ltx.video_vae.activation.*
ltx.video_vae.output
ltx.host.stage1_latent / stage2_latent
ltx.host.decoded / ltx.host.output
```

Stage 1 与 Stage 2 的 latent 不能仅凭“同一个 vector”自动 alias；必须在 manifest 中写出
`last_use_event=stage1.export` 和 `live_begin=stage2.input`，并由编译器确认两者不重叠后才能共享
alias group。

## 214. Flux、Z-Image 以及后续模型的接入顺序

### 214.1 Flux：先做 component-staged F1

Flux 当前有更多 backend/LoRA/ANE/MLP 变体，直接做逐 transformer layer pager 会把 candidate key、
quality gate 和 allocator closure 复杂度同时推高。建议 F1 只支持：

```text
text encoder（固定 resident 或独立 host envelope）
 -> transformer component
 -> VAE component
```

每个 component 之间必须有显式 drain 和 cache clear；只有 F1 在 L2/L3 证明 `component_peak <= B`
后，才允许把 transformer block 拆成 refill slots。Flux 的 LoRA 必须绑定 checkpoint/content digest，
不能在 constrained request 内临时 merge。

### 214.2 Z-Image：BF16 与 GGUF 分离 candidate

Z-Image 的 BF16 和 GGUF 不能共享同一 manifest：权重上界、dequant scratch、kernel workspace 和质量
阈值均不同。建议至少生成：

```text
z_image_c_metal_bf16_component_v1
z_image_c_metal_gguf_component_v1
```

GGUF candidate 必须额外记录 dequant buffer 的 `ConversionScratch` upper，以及每个量化 shard 的
`last_use_event`；如果 dequant workspace 不能在 command completion 后及时回收，则该 candidate 只能
作为 resident/plan-only，不得宣称硬上限。

## 215. overlap、I/O 和资源利用率设计

### 215.1 推荐的三条队列

在不改变默认模式的前提下，constrained candidate 可以使用三条逻辑队列：

```text
Read queue       pread/mmap-independent file read -> host staging
Upload/encode    host staging -> Metal shared/private buffer + command encode
Compute queue    DiT/denoiser/VAE kernels
```

实际实现可以复用线程池，但 trace 中必须能区分三类时间。允许的 overlap 是：

```text
Compute(block n) || Read(block n+1) || Upload(block n+1)
```

不允许的 overlap 是：

```text
释放 slot n 的 backing || block n 的最后一个 GPU command
VAE allocation || DiT 尚未 drain 的 command
增加 refill slot || 已经达到 B 的 Tight/Critical 状态
```

### 215.2 lookahead 的预算约束

lookahead 不是免费优化。若预取 `k` 个未来 block，则必须把 `k * slot_upper` 或相应 host staging
upper 加入 manifest 的 live set。建议每个 candidate 显式记录：

```text
lookahead=0：最低峰值，适合 Critical/Tight
lookahead=1：默认 constrained v1
lookahead=2：只有 evidence 证明 I/O 能被完全隐藏时启用
```

当 `accounted/B >= 0.85` 时，pressure controller 自动把 lookahead 降到 0；不要等 allocator 失败后
才停止预取。

### 215.3 评价 overlap 的指标

必须同时记录：

```text
read_seconds
upload_seconds
encode_seconds
compute_seconds
wait_for_slot_seconds
wait_for_completion_seconds
idle_gap_seconds
```

建议定义：

```text
overlap_ratio = 1 - idle_gap_seconds / wall_seconds
gpu_utilization = compute_seconds / wall_seconds
```

`overlap_ratio` 不能用“总阶段耗时减去 read 耗时”估算，必须来自带 sequence id 的 trace event，
否则无法识别 read 与 compute 的真实重叠。

## 216. 结果 JSON、trace 和故障诊断字段

constrained 请求的结果应在现有字段之外增加一个稳定的 `memory_admission` 对象。建议最小字段如下：

```json
{
  "enabled": true,
  "user_limit_bytes": 25769803776,
  "buffer_percent": 15,
  "effective_budget_bytes": 21904333209,
  "planned_upper_bytes": 21139292160,
  "actual_peak_bytes": 20803747840,
  "process_baseline_bytes": 4294967296,
  "framework_upper_bytes": 536870912,
  "refill_slots": 2,
  "peak_epoch": 173,
  "manifest_digest": "...",
  "plan_digest": "...",
  "pressure_state": "normal",
  "swapouts_delta": 0,
  "completion_consumed": 104,
  "pending_after": 0,
  "failure_disposition": "none",
  "worker_quarantined": false
}
```

字段语义要求：

- `planned_upper_bytes` 来自 manifest/plan，不能被 observed peak 覆盖；
- `actual_peak_bytes` 来自过程观测和 ledger peak，两者取更严格的一方并同时保留原始值；
- `swapouts_delta` 只用于证据和失败判定，不参与预算加法；
- `pending_after` 必须在 backend drain 与 mailbox drain 之后采样；
- `worker_quarantined=true` 时，engine 必须拒绝后续请求直到重建。

建议 trace event 至少包含：`request_admitted`、`manifest_loaded`、`stage_begin`、`reserve_begin`、
`allocate_commit`、`file_read_begin/end`、`command_submitted`、`retire`、`completion_posted`、
`completion_consumed`、`pressure_transition`、`stage_drain`、`request_finished`。每个 event 携带
`request_id`、`generation`、`allocator_domain`、`stage_id`、`slot_id`、`site_id` 和 `sequence`。

## 217. 失败、取消和 worker quarantine 的代码建议

### 217.1 错误优先级

实现中必须保留 primary failure，不得被 cleanup failure 覆盖：

```text
primary_failure = allocation_failed | command_failed | cancelled | pressure_abort
cleanup_failure = drain_failed | stale_completion | mailbox_overflow | cache_residual
```

结果中同时输出两者；如果 `cleanup_failure` 非空，disposition 最低为 `QuarantineWorker`，除非有
独立证据证明所有 queue、token、storage 和 helper 已清零。

### 217.2 cancel 的反向顺序

取消不是“立即 free 所有对象”。安全顺序为：

```text
set cancellation flag
stop submitting new command
stop optional prefetch
finish/drain already committed command
post/consume all completions
release host staging and inactive reservations
clear component cache
assert zero-state
```

如果 command buffer 永久不完成、Metal 返回 error 或 mailbox overflow，必须隔离 worker；不能为了返回
更快而强制释放仍可能被 GPU 使用的 backing。

### 217.3 `finalize_failure()` 的具体修改建议

建议将当前实现拆成三个内部函数，减少异常路径重复判断：

```cpp
MemoryFailureDisposition classify_failure() const noexcept;
bool has_gpu_work_or_pending_tokens() const noexcept;
bool verify_failure_zero_state(std::string *why) const noexcept;
```

`finalize_failure()` 只负责记录 primary failure 和分类；`complete_failure_cleanup()` 负责 drain 后
再次验证；两者不能在同一个调用中隐式吞掉 backend drain 异常。这样可以让 API 层明确决定：返回错误、
重建 worker，还是在确认 clean 后复用 worker。

## 218. 详细验收矩阵（实现者可直接执行）

### 218.1 L0：纯逻辑和 ABI

必须通过：

```text
memory_probe / manifest / plan_compiler / accounting / scheduler / execution
contract / h3_schedule_memory
```

必测负向向量：

```text
buffer_percent=0 或 31
limit_bytes=0 或 overflow
max_refill_slots=0 或 4
manifest 缺 required site
instance live interval 反转
actual_bytes > upper_bytes
allocator_domain/generation/stage/slot 不匹配
duplicate/stale completion
mailbox overflow
finish_success 时残留 storage/cache/pending/unknown
```

### 218.2 L1：默认路径零行为

对同一个 request 做 disabled/enabled=false 的 ABBA 测试，验证：

```text
没有 probe/hash/registry lookup
没有 MemoryExecutionContext/MemoryLedger 构造
没有额外 drain/unload/cache clear
没有 allocator hook 安装
既有结果字段值/类型不变
P50/P95 wall time 与峰值在预设门槛内
```

建议首发门槛是 disabled 路径 wall-time P95 增量不超过 1%，但最终数字应由真实设备基线和 reviewer
签核决定；没有基线时不能宣称“零开销”。

### 218.3 L2：真实 Metal 设备

每个 candidate、shape bucket、dtype、slot revision 至少运行：

```text
warmup >= 3
measurement >= 10
成功、取消、Metal command error、重复 completion、stage boundary
```

必须保存：

```text
device identity / OS / runtime revision
manifest/plan/evidence digest
planned_peak / actual_peak / ledger_peak
command completion sequence
swapouts_delta / compressor / process-tree envelope
output quality hash 或固定质量指标
```

L2 的硬条件：`actual_peak <= B`、`unknown_bytes==0`、`pending_after==0`、`mailbox_overflow=false`、
成功输出质量达标。任一失败都只能标记为 `ExperimentalGuarded` 或 `PlanOnly`。

### 218.4 L3：低内存 campaign

低内存 campaign 不应直接把系统推到不可恢复状态。建议使用隔离 worker，在 16/24/32 GiB 物理内存
机器上分别运行：

```text
Y = 0.50 * physical_memory
Y = 0.65 * physical_memory
Y = 0.80 * physical_memory
X = 5, 10, 15, 20, 30
refill_slots = 1, 2, 3（候选支持时）
```

每个组合至少重复 5 次，并记录：

```text
swapouts_delta == 0
actual_peak <= B
quality pass rate
wall time P50/P95
worker quarantine rate
completion/mailbox residual
```

出现新增 swapout、未知 backing 或 worker quarantine 时，该组合不得写入 release evidence；可以作为
失败向量保留，用于调节 manifest upper 或禁用 candidate。

### 218.5 当前无 Metal 环境的本机记录

截至 2026-09-15，本机已重跑：

```text
native build                         PASS
memory probe                         PASS
memory manifest                      PASS
memory plan compiler                 PASS
memory accounting                    PASS
memory scheduler                     PASS
memory execution                    PASS
contract                             81 PASS / 1 fixture SKIP
h3 schedule memory                  PASS
h3 GPU memory hooks                 SKIP: no Metal device is available
ltx GPU memory hooks                SKIP: no Metal device is available
git diff --check                     PASS
```

这只能作为 L0/L1 证据。不得把 Metal hook 的 SKIP 解读成 completion 正确，也不得把逻辑 plan 通过解读成
真实 GPU 峰值通过。

## 219. 分阶段提交与发布门禁

建议按以下 patch 队列实施，任何一步失败都可以独立回滚：

```text
P0  文档、schema、错误码和 trace 字段冻结
P1  MemoryExecutionState 显式状态机与 zero-state assert
P2  MemoryStageScheduler 的 stage/slot/lookahead API
P3  H3 required-site closure + 真实 retire/complete evidence
P4  LTX required-site closure + async hook ABI（或明确 single-queue 限制）
P5  H3/LTX 真实 Metal L2 campaign
P6  Flux component-staged F1 plan-only
P7  Z-Image BF16/GGUF 分离 candidate plan-only
P8  低内存 L3 campaign 与独立 reviewer 签核
P9  只添加已签核 record 的 production registry
```

每个 patch 必须附：

```text
代码 diff + manifest/plan digest
新增/修改的单元测试
禁用路径 ABBA 结果
失败路径和 cleanup 证据
明确的 PASS/SKIP/FAIL 原因
```

Release gate 固定为：

```text
required-site closure = complete
capability level >= EnvelopeValidated
certification state = Certified
L2 真实 Metal 证据存在
L3 证据存在（若产品声称低内存不会 swap）
disabled 回归通过
production registry record 经过独立 review
```

任何 gate 未满足时，registry lookup 必须 miss，constrained request 必须拒绝；不能自动回退到 resident
路径，也不能仅在结果 JSON 中写 `release_stable=false` 后继续执行。

## 220. 本轮增补后的实施结论

当前最优路线不是继续增加更多模型入口，而是完成两条已接入路线的闭环：

```text
先把 H3 的 required-site、真实 Metal completion 和 L2 证据做完整
再把 LTX 的 stage/helper/VAE 边界和 async gap 做完整
随后才扩展 Flux/Z-Image 的 component-staged candidate
```

产品语义保持不变：

```text
disabled：完全沿用本机默认 residency/cache/调度，零额外 scheduler 行为
enabled ：B 是硬预算；reserve-before-allocate；显式 streaming/loading/offloading；
          swap 只用于观测和失败信号；未知 backing、completion 残留、实际越界即 fail-closed
```

只要 production registry 仍为空，当前代码就是“安全的 plan-only/scaffolding 状态”；这比在没有真实
Metal/L3 证据时宣称“任何低内存机器都不会 swap”更符合工程事实，也为后续实现者提供了清晰、可复现、
可回滚的施工和验收边界。

## 221. 当前源码审计发现的关键 enforcement 缝隙

这部分是本轮最重要的代码级补充：当前 planner 已经能计算 interval peak，但 runtime 还没有完整执行
instance、epoch 和 aggregate upper 约束。如果不补齐这些约束，一个调用点即使使用了合法 `site_id`，仍
可能在错误 epoch 重复分配，或者同时创建超过 manifest 声明数量的 backing。

### 221.1 `try_reserve_site()` 当前只验证单实例 upper

当前 `MemoryExecutionContext::try_reserve_site()` 会验证：

```text
site_id 存在
memory_class 匹配
upper_bytes <= maximum_instance_upper_bytes
```

但它还没有验证：

```text
当前 epoch 是否允许此 site 存活
active instance 数是否超过 instance_count
同 site active upper 总和是否超过 aggregate_instance_upper_bytes
调用点对应哪个 manifest instance_id
该 instance 是否已经被另一个 storage 占用
alias_group 是否真的在上一 instance release 后才复用
```

因此正式 candidate 进入 registry 之前，必须增加 runtime site guard。推荐的数据结构为：

```cpp
struct CompiledAllocationInstance {
    std::string site_id;
    uint32_t instance_id = 0;
    MemoryClass memory_class = MemoryClass::UnknownExternal;
    uint64_t upper_bytes = 0;
    uint64_t live_begin = 0;
    uint64_t live_end = 0;
    uint64_t alias_group = 0;
    bool asynchronous = false;
};

struct RuntimeSiteState {
    uint32_t active_instances = 0;
    uint64_t active_upper_bytes = 0;
    uint32_t peak_active_instances = 0;
    uint64_t peak_active_upper_bytes = 0;
};

struct MemorySiteIdentity {
    std::string_view site_id;
    uint32_t instance_id = 0;
    uint64_t epoch = 0;
    uint32_t stage_id = 0;
    uint32_t slot_id = 0;
};
```

建议修改 API：

```cpp
std::optional<MemoryReservation> try_reserve_site(
    const MemorySiteIdentity &, MemoryClass, uint64_t upper_bytes);

void enter_epoch(uint64_t epoch, std::string_view event);
void leave_epoch(uint64_t epoch);
```

ledger 在 reserve 时验证 `live_begin <= epoch < live_end`，在 commit 时绑定 instance 与
`StorageId`，在 release/pending completion 时降低 active counters。重复占用同一 instance 或在 interval
之外 reserve 都返回 `memory_lifetime_violation`。

### 221.2 manifest instance 表示物理 backing，不表示每次 refill 内容

实现时必须区分：

```text
allocation instance = 一个有容量、有 allocator handle 的物理 backing
refill event         = 把不同 block 内容写入同一个 backing
```

如果两个 refill slot 在整个 denoise stage 内复用，则 manifest 应只有两个 slot instance；每次 block
切换只生成 trace event，不应为 48 个 block * steps 伪造数百个 allocation instance。反之，如果当前实现
确实每个 block 都创建新 Metal buffer，则 manifest 必须逐次建模，或者先改成固定 slot pool 后才能发布。

### 221.3 alias group 的运行时证明

planner 允许同一 alias group 只计最大 upper，但 runtime 必须证明复用合法。建议 ledger 维护：

```cpp
std::unordered_map<uint64_t, StorageKey> active_alias_owner;
```

新 allocation commit 时，如果 alias group 已有 active owner，则拒绝；只有同步 release 或
`complete_pending()` 消费 completion 后才能转移 owner。这样可以防止 manifest 声明“不重叠”，但后端
实际 command 仍在使用旧 backing 的情况。

### 221.4 required-site closure 生成器

建议新增只读工具：

```text
tools/memory/generate_site_inventory.py
tools/memory/compare_site_inventory.py
```

生成器扫描受支持 allocator 入口，例如：

```text
h3_gpu_tensor_new_classified*
h3_gpu_tensor_load_*
h3_accounted_host_*
ltx_gpu_buffer_new_classified*
reserve_host_memory
MLX/Core ML/helper envelope bridge
```

输出：

```json
{
  "source_revision": "...",
  "sites": [
    {
      "site_id": "h3.dit.activation.core_input",
      "file": "native/models/h3_runtime/h3_dit.c",
      "line": 5406,
      "allocator": "h3_gpu_tensor_new_classified",
      "class": "activation",
      "literal": true
    }
  ]
}
```

`compare_site_inventory.py` 做 expected-key equality：源码 required site 必须全部出现在 manifest；manifest
required site 必须全部能映射回源码或明确的 generated site。dynamic tag、空 tag 和 `UNKNOWN` allocation
在 release candidate 中一律失败。

## 222. `framework_upper`、unattributed memory 和 process sampler

### 222.1 移除可执行路径上的默认零值

当前 `authorize_memory_capability(..., framework_upper_bytes = 0)` 的默认参数容易让调用方忘记提供真实
framework envelope。建议删除默认值，并将 upper 放入经过签核的 capability record：

```cpp
struct MemoryCapabilityRecord {
    // existing fields...
    uint64_t framework_upper_bytes = 0;
    uint64_t unattributed_upper_bytes = 0;
    std::string framework_provenance;
    std::string framework_profile_digest;
};
```

然后授权流程固定为：

```cpp
auto compiled = compile_memory_plan(
    manifest,
    baseline.process_footprint_bytes,
    record.framework_upper_bytes,
    policy.effective_budget_bytes);
```

release record 中 `framework_upper_bytes==0` 只有在后端完全不创建任何未入 ledger 的 allocator/cache/
graph temporary 且有专项证据时才合法。H3/LTX 首个 candidate 预计不能假设为零。

### 222.2 framework upper 的校准方法

建议在干净 worker 中运行同一 candidate 的固定 campaign：

```text
unattributed(t) = process_footprint(t)
                - clean_process_baseline
                - ledger_known_bytes(t)
```

对每个设备族、OS/runtime revision、shape bucket 和 schedule revision，记录多次运行的
`max(unattributed(t))`。签核 upper 应满足：

```text
framework_upper >= campaign_max + deterministic_guard
```

`deterministic_guard` 必须是明确字节数或明确公式，不能只写“P99 + 一点余量”。任何 runtime、Metal、
MPSGraph、MLX 或 graph compilation 策略变化都使该 evidence 失效。

### 222.3 enabled-only process sampler

ledger 是预防越界的第一防线；process sampler 只能检测未知项，不能代替 reserve-before-allocate。建议在
constrained context 内启用专用 sampler，disabled 路径不创建线程、不注册 timer：

```cpp
struct MemoryWatchdogConfig {
    std::chrono::milliseconds interval{20};
    uint64_t budget_bytes = 0;
    uint64_t framework_upper_bytes = 0;
    uint64_t system_reserve_bytes = 0;
};
```

watchdog 只做无锁/低锁采样并写入 bounded ring；owner thread 在 safe-point 消费。它发现
`footprint > B`、`unattributed > framework_upper` 或 system reserve 被破坏时，将 pressure 设置为 sticky
Critical。watchdog 不调用 allocator、不 unload session、不抛跨线程异常。

### 222.4 swap 观测

swap 观测至少记录 request 开始和结束的累计 swapout 计数以及 compressor/available memory。判定固定为：

```text
swapouts_delta = swapouts_end - swapouts_begin
```

机器在请求开始前已有 swap.used 不应直接判失败；本请求新增 swapout 才是失败信号。由于系统计数是
全局的，L3 campaign 必须在隔离、低背景噪声的 worker/机器上运行，并把并发进程清单纳入 evidence。

## 223. capability identity 还需要绑定的维度

当前 key 已包含 adapter、model/checkpoint、backend、dtype、variant、operation、shape、sampler、slots、
tiling、runtime revision 和 device family。正式发布还建议显式绑定：

```cpp
std::string schedule_revision;
std::string framework_profile_digest;
std::string helper_mode;
std::string process_topology;
std::string platform_build_class;
```

原因如下：

- 同样的 `refill_slots=2`，lookahead、pinned prefix 或 command batching 不同会改变 live set；
- 同一 VAE tiling 名称，in-process helper 与 exec finalizer 的 process envelope 不同；
- `memory_runtime_revision()` 当前是常量字符串，若不绑定 build/source revision，代码变化后旧 evidence
  仍可能错误匹配；
- Metal/macOS runtime 变化可能改变 graph temporary 和 allocator watermark。

推荐 `memory_runtime_revision()` 返回由以下内容组成的 build-time digest：

```text
memory schema revision
scheduler source revision
allocator bridge ABI revision
model adapter revision
build configuration
```

OS build 与 device family 不一定要直接拼入 runtime digest，但必须进入 `MemoryDeviceIdentity` 和 evidence
lookup 策略。任何模糊匹配必须先经过独立兼容性审查。

## 224. 按源码文件拆分的具体修改建议

### 224.1 Runtime 核心

| 文件 | 建议修改 | 验收点 |
|---|---|---|
| `native/runtime/memory_manifest.hpp/.cpp` | 增加 schedule/framework identity；明确 slot backing 与 refill event | canonical digest 随任一字段变化 |
| `native/runtime/memory_plan.hpp/.cpp` | 保留 `CompiledAllocationInstance`；输出 epoch/live-set/site aggregate | 重复 instance、alias overlap 负向测试 |
| `native/runtime/memory_accounting.hpp/.cpp` | ledger 绑定 site/instance/alias；输出 per-class/per-site peak | release 后 counter 精确归零 |
| `native/runtime/memory_scheduler.hpp/.cpp` | 增加 epoch、stage、slot、lookahead hint API | Tight 只降级，Critical sticky |
| `native/runtime/memory_execution.hpp/.cpp` | 显式 execution state；移除 framework default zero；zero-state report | success/failure/cancel 全状态覆盖 |
| `native/api/c_api.mm` | 调用 begin_running/begin_draining；保留 primary/cleanup failure | 任何 cleanup 异常都能 quarantine |
| `native/platform/apple/results.mm` | 输出 plan/actual/framework/swap/completion 字段 | JSON 类型稳定、disabled 向后兼容 |
| `native/platform/apple/memory_probe.mm` | 校验新增 identity 和 upper provenance | sidecar 缺字段 fail-closed |

### 224.2 H3

| 文件 | 建议修改 | 验收点 |
|---|---|---|
| `h3_gpu.h/.m` | 保持 struct_size ABI；debug completion provenance；所有 required allocation 分类 | v1 老 hook 兼容、v2 retire/complete 成对 |
| `h3_dit.c` | 为 pinned/refill/activation 建立稳定 site/instance mapping | 50/35 layer 分支都有 closure |
| `h3_dit_schedule.c` | host staging、time embedding、block AdaLN upper 精确公式 | shape 边界测试 |
| `h3_text_encoder.c` | text weights/conditioning/activation 生命周期分离 | text encoder free 后 ledger 清零 |
| `h3_video_vae.c` / `h3_taeh3.c` | decoder queue drain 与 tile/output upper | DiT/VAE 不非法 alias |
| `h3.c` | cleanup 前统一 drain；local/cached component 去重 | cancel/错误不会 double drain/free |
| `h3_session.mm` | completion failure sticky；context unbind 只能在 drain 后 | late callback 导致 quarantine |

### 224.3 LTX

| 文件 | 建议修改 | 验收点 |
|---|---|---|
| `ltx_gpu.h/.m` | 追加 async hook 或明确 single-queue v1；token 与 command identity 绑定 | active batch drain 返回 violation |
| `ltx_native.h` / `ltx_blocks.c` | stage1/stage2/refill slot instance mapping；主/audio queue drain | 两 queue 均无 pending |
| `ltx_gemma_encoder.*` | encoder weights/activation/conditioning closure | encoder phase结束归零 |
| `ltx_session.mm` | Transformer→VAE 边界、helper envelope、child process topology | helper 失败可回收或 quarantine |
| VAE/helper bridge | graph cache、decoded frame、output buffer upper | process-tree peak 纳入 B |

### 224.4 Flux 与 Z-Image

首个 patch 只增加 metadata probe、component envelope 和 drain hook，不直接添加 production record。实现者
应先证明 component-staged 路线，再讨论逐层 pager。Flux 的 LoRA/ANE 组合、Z-Image 的 BF16/GGUF
分别是不同 candidate，不能用一个“大而全”的 manifest 包含互斥分支。

### 224.5 测试文件

建议补充或扩展：

```text
tests/native/memory_plan_compiler_test.cpp
  - runtime epoch/instance/alias vectors

tests/native/memory_scheduler_test.cpp
  - Tight hint、Critical sticky、late completion、mailbox saturation

tests/native/memory_execution_test.cpp
  - explicit state transition、primary+cleanup failure、framework upper

tests/native/h3_gpu_memory_hooks_test.mm
  - ABI v1/v2 struct_size、retire failure、late callback、drain idempotence

tests/native/ltx_gpu_memory_hooks_test.mm
  - active batch violation、two-queue drain、deferred token order

tests/native/test_contract.py
  - disabled zero-call source contract、production registry empty、source inventory closure
```

## 225. Evidence artifact 和 reviewer 工作流

### 225.1 建议目录结构

每次 L2/L3 campaign 生成不可变 artifact：

```text
results/memory/<candidate>/<device>/<run-id>/
  request.json
  device.json
  checkpoint.json
  manifest.json
  plan.json
  result.json
  trace.jsonl
  process-memory.csv
  vm-swap.json
  quality.json
  summary.md
```

`evidence_digest` 对上述 canonical 文件清单、每个文件 SHA-256 和工具版本计算；不能只对 summary 文本
计算。原始 trace/CSV 不允许在 review 后被覆盖。

### 225.2 reviewer 必答问题

```text
1. request key 是否和 capability record 完全相等？
2. checkpoint identity 是否来自同一 fd 的稳定快照？
3. required-site closure 是否为 100%？
4. framework_upper 是否非启发式并绑定当前 runtime？
5. planned peak 是否包含 helper/output/process-tree？
6. actual peak 是否在 B 内且无新增 swapout？
7. 所有 completion 是否来自真实 backend completion？
8. success/cancel/error 后是否 zero-state？
9. disabled ABBA 是否在门槛内？
10. record 是否由 source/build artifact 生成而非 request/env 注入？
```

只有十项全部为“是”才能将 record 设为 `release_enabled=true`。

## 226. 代码实现优先级的最终收敛

结合本轮源码审计，runtime 正确性底座已经进入当前工作树；后续最短可发布路径应改为：

```text
1. 以当前 runtime-v3 API 为冻结基线，补齐 H3 required-site closure report
2. 在真实 Metal 设备完成 H3 completion/peak/fault L2
3. 生成 LTX required-site closure report，并补 helper/process-tree envelope
4. 在真实 Metal 设备完成 LTX L2
5. 完成 trace artifact 导出与稳定字段（runtime-v3 已落地 bounded trace buffer、显式 execution state 和 owner-polled watchdog）
6. 做 16/24/32 GiB L3 campaign
7. 独立 review 后添加第一条 production registry record
8. 再开始 Flux/Z-Image 的 plan-only adapter
```

其中第 1、2、3、4 项是模型适配闭环；第 5 项是可观测性和维护性增强；第 6、7 项是产品承诺门槛。runtime
instance/epoch/alias、framework upper、constrained generic allocation 禁止和 completion lifecycle 已在当前工作树
实现，但还没有真实 Metal 设备上的闭环证据。只做 streaming 性能优化而跳过这些 enforcement，仍然不能证明硬预算，
也不能证明“不会因为内存不足进入 swap”。

## 227. 本轮源码落地事实：runtime-v3 已经提供的硬约束

本节是实现者必须以代码为准的事实清单。若本节与前文“建议增加”的描述冲突，以本节和当前头文件签名为准。

### 227.1 编译计划已经从 aggregate upper 细化到 allocation instance

`native/runtime/memory_plan.hpp` 的 `CompiledMemoryPlan` 现在同时保存：

```cpp
uint64_t process_baseline_bytes;
uint64_t framework_upper_bytes;
std::vector<CompiledAllocationSite> allocation_sites;
std::vector<CompiledAllocationInstance> allocation_instances;
```

每个 `CompiledAllocationInstance` 包含 `site_id`、`instance_id`、`upper_bytes`、半开区间
`[live_begin, live_end)`、`alias_group` 和 `asynchronous`。这使计划能够回答三个不同的问题：

1. 某个 site 总共有多少个合法 instance；
2. 在当前 epoch 哪个 instance 可被占用；
3. 一个 alias backing 是否仍被上一个逻辑 view 占用。

`compile_memory_plan()` 采用半开区间，在相同 epoch 先 release、后 acquire。manifest 侧还禁止同一
`alias_group` 的 lifetime 重叠；如果真实 backend 需要多个逻辑 view 同时指向同一 backing，必须新增
`StorageKey`/shared-view 语义，不能直接删除 overlap 校验。

### 227.2 framework upper 已从 capability record 注入 admission

可执行 capability record 必须提供非零 `framework_upper_bytes` 和非空 `framework_provenance`，且不能超过
`maximum_validated_upper_bytes`。请求字段、普通环境变量和 synthetic test record 都不能授予生产执行能力。

当前 admission 的额度关系是：

```text
planned_process_upper = process_baseline + planned_increment
concrete_ledger_ceiling = planned_process_upper - framework_upper
```

因此 ledger 只为真实、可归因 allocation 留额度，未归因 process footprint 必须在 checkpoint 时满足：

```text
unattributed_process_footprint <= framework_upper
```

超过时失败并保持 tainted；不能在后续释放缓存后把这次越线重写成成功。

### 227.3 constrained execution 已拒绝未声明 allocation

`MemoryExecutionContext` 构造时执行：

```cpp
admission_->ledger().set_site_constraints_required(true);
```

此后普通 `MemoryLedger::try_reserve()` 只能用于 unconstrained/test ledger。在 constrained context 中，
adapter 必须使用：

```cpp
context.try_reserve_site(site_id, memory_class, upper_bytes);
```

如果调用 generic reserve，必须失败并报告 `memory_estimate_unknown: unmanifested allocation is forbidden in constrained execution`。
这个 fail-closed 行为是防止旧 allocator call-site 绕过 manifest 的最后一道保护；不能为了兼容旧代码而把错误吞掉。

### 227.4 site instance 生命周期已和 lease 状态绑定

一个 site reservation 的 runtime 状态只能按以下方向变化：

```text
Reserved -> Active -> Pending -> (Free | Cached)
```

`StorageLease::retire()` 只进入 pending，不代表 GPU 已经不再访问 backing；只有真实 completion 调用
`complete_pending()` 后才能释放或转 cache。`cache()` 仍然计费，`finish_success()` 会拒绝 active、pending、cached、
unknown 或 mailbox 未清空的 context。

### 227.5 epoch、execution state 与失败 disposition 已经有可观测语义

`MemoryExecutionContext::enter_epoch()` 禁止 epoch 倒退、缺失 epoch 和已有 allocation 越过 `live_end`；
`try_reserve_site()` 在存在未来合法 instance 时可以自动推进到最早 epoch，但生产 adapter 仍应在 stage boundary
显式调用 `enter_epoch(epoch, event)`，避免自动推进掩盖错误的调用顺序。

失败结束分为：

```text
Clean              没有 GPU/ledger 残留，可复用 worker
NeedsGpuDrain      需要 backend drain 后再判定
QuarantineWorker   completion、unknown memory 或 cleanup 不可证明安全
```

`finalize_failure()` 只记录 primary failure 和 disposition；GPU drain 必须由拥有 backend queue 的上层执行，随后调用
`complete_failure_cleanup()`。这避免 runtime 线程直接对 Metal queue 做不可控同步。

当前代码还已经落地显式 `MemoryExecutionState`：

```text
Admitted -> Running -> Draining -> Succeeded
                              \\-> Failed
                              \\-> Quarantined
Admitted/Running -> Failed | Quarantined
Running -> Draining -> Failed | Quarantined
```

`finalize_failure()` 发现 pending GPU work 时将状态置为 `Draining`，只有 cleanup 证据完整后才进入 `Failed`；
`finish_success()` 只接受 `Draining`，成功后进入不可逆的 `Succeeded`。c_api generate/prepare 入口显式调用
`begin_running()` 和 `begin_draining()`；结果 JSON 同时输出 `execution_state` 与旧的 disposition/tainted 字段。

### 227.6 当前验证结果（2026-09-16）

当前工作树已重新构建并执行：

```text
native build                         PASS
memory probe                         PASS
memory manifest                      PASS
memory plan compiler                 PASS
memory accounting                    PASS
memory scheduler                     PASS
memory watchdog                      PASS
memory execution                     PASS
contract                             81 PASS / 1 fixture SKIP
H3 schedule memory                   PASS
H3 GPU memory hooks                  SKIP: no Metal device is available
LTX GPU memory hooks                 SKIP: no Metal device is available
```

这些结果证明的是纯逻辑约束和源码 contract；不能替代真实 Metal completion、真实 allocator peak、真实
process-tree footprint 或低内存机器上的新增 swapout 证据。生产 capability registry 仍为空。

## 228. adapter 接线合同：从两阶段 admission 到真实 GPU backing

所有 H3/LTX/未来模型 adapter 应遵循同一顺序。下面的伪代码是施工合同，不是可直接复制的单一实现：

```cpp
auto resolution = preflight_memory_capability(
    session, plan, device_identity, production_registry);
auto compiled = authorize_memory_capability(
    plan, resolution, baseline.process_footprint_bytes);
MemoryExecutionContext context(
    *plan.memory_policy, std::move(compiled), baseline, observer);
ScopedMemoryExecutionBinding binding(session, &context);

context.enter_epoch(kStageEpoch, "stage.begin");
auto reservation = context.try_reserve_site(
    "model.stage.buffer", MemoryClass::Activation, upper_bytes);
require(reservation.has_value(), "memory_budget_too_small: reservation denied");

// 真实 allocator 只能放在 reservation 成功之后。
StorageId storage = backend_allocate_after_admission(upper_bytes);
auto lease = reservation->commit(storage);
stage.adopt(std::move(lease));

submit_gpu_work();
// retire 只表示提交了 release，不表示 GPU 已经完成。
auto pending = stage.retire_all();
post_real_backend_completion_for_each(pending);
require(context.drain_completion_mailbox().ok(),
        "memory_completion_failure: completion drain failed");
context.checkpoint("stage.complete");
```

施工时必须满足：

1. **reserve-before-allocate**：任何 `MTLBuffer`、MLX materialized array、转换副本、host staging 和输出 arena
   都要先 reserve；无法在分配前提供 upper 的路径必须暂时标记为不可执行；
2. **commit-after-real-allocation**：commit 的 `actual_bytes` 是 allocator 返回的真实 capacity，不是请求 upper；
3. **retire-before-publish-token**：异步 release 先进入 ledger pending，再把 token 捕获到 Metal completion closure；
4. **drain-at-safe-point**：只在 queue/event 已达到约定安全点后 drain mailbox；不要让 callback 线程直接修改 ledger；
5. **stage boundary checkpoint**：每个 encoder、denoiser、VAE 和 export boundary 至少 checkpoint 一次；长循环按
   block/step 周期 checkpoint，但不得每个 kernel 都同步采样。

当前 `MemoryStageLease::reserve_site()` 仍保留仅传 `site_id` 的兼容接口，并且在 constrained context 中会因为
generic reserve 禁止而失败；同时已经新增以下两个安全重载，生产 adapter 应使用它们：

```cpp
// 路径 A：stage 从 context 获取 site reservation
std::optional<MemorySiteToken> reserve_site(
    MemoryExecutionContext &, std::string site_id,
    MemoryClass, uint64_t upper_bytes);

// 路径 B：显式传入已由 compiled plan 导出的 constraint
std::optional<MemorySiteToken> reserve_site(
    MemoryClass, uint64_t upper_bytes,
    const MemorySiteReservationConstraint &);
```

路径 A 已经实现并且更不容易被 adapter 错传 epoch/instance；路径 B 也已实现，适合 plan compiler 和纯 C bridge
测试。两者都不能重新开放无 site 的 generic reserve。

## 229. 显式 execution state（已实现）与 watchdog（owner-polled 已实现）

当前代码已经用 `MemoryExecutionState` 表达终态和 drain 过程；`tainted_` 与 `failure_disposition_` 仍保留用于
兼容结果协议和错误分类。当前实现的枚举为：

```cpp
enum class MemoryExecutionState : uint8_t {
    Admitted,
    Running,
    Draining,
    Succeeded,
    Failed,
    Quarantined,
};
```

允许的状态转移：

```text
Admitted -> Running -> Draining -> Succeeded
                             \-> Failed
                             \-> Quarantined
Admitted/Running -> Failed | Quarantined
Running -> Draining -> Failed | Quarantined
```

实现要求：

- `finish_success()` 只接受 `Draining`，并要求 reservation/pending/storage/site/unknown 全为零；
- `finalize_failure()` 把任意未完成状态置为 `Failed` 或 `Quarantined`，不得再次进入 `Running`；
- `complete_failure_cleanup()` 只允许 `Failed + NeedsGpuDrain`，cleanup 证据完整时才降为可复用 `Failed/Clean`；
- destructor 若仍不是终态，保守走 `finalize_failure()`；
- 结果 JSON 同时输出旧 `finished/tainted/failure_disposition` 和新 `execution_state`，至少保留一个版本周期。

为减少误用，`MemoryStageLease` 还新增了携带 `MemorySiteReservationConstraint` 的 `reserve_site` 重载；该重载
直接调用 `MemoryLedger::try_reserve_site()`，不会绕回 generic reserve。旧的仅传 `site_id` 的兼容重载在 constrained
context 中仍会失败，生产 adapter 应优先使用 context/compiled constraint 路径。

watchdog 只在 enabled/constrained request 创建，disabled request 必须完全不创建线程、timer、probe 或 ring buffer。
当前已经实现为 owner-thread safe-point 轮询的 bounded ring，不启动后台线程；这是有意选择：后台线程在
Metal teardown、MLX cache 操作和 worker quarantine 时会引入难以证明的竞态。未来若要异步采样，必须保留同样的
“不分配、不抛异常、不操作 queue”合同，并单独升级 runtime revision。

```cpp
struct MemoryWatchdogSample {
    uint64_t monotonic_ns;
    uint64_t process_footprint;
    uint64_t system_available;
    uint64_t committed;
    uint64_t reserved;
    uint64_t unattributed;
    uint32_t phase_id;
};
```

watchdog 只采样和设置 sticky pressure flag；不得分配内存、抛异常、调用 `unload` 或直接操作 Metal queue。
owner 在 `MemoryExecutionContext::checkpoint()` 中复用 admission 已取得的 process observation，按
`Normal/Tight/Critical` 调用 scheduler。ring 满时记录 `watchdog_dropped_samples` 并进入保守状态；不能静默
丢掉唯一的越线证据。`watchdog_critical` 会被合并为 scheduler 的 observed-over-budget 信号，因而不会在
后续样本恢复后自动清除。

当前代码位置：`native/runtime/memory_watchdog.hpp/.cpp`、`MemoryExecutionContext::checkpoint()`、
`MemoryAdmissionMetrics` 和 `native/platform/apple/results.mm`。纯逻辑覆盖位于
`tests/native/memory_watchdog_test.cpp`，包括正常采样、framework/budget/system reserve 越线、ring overflow
和显式 failure。

### 229.1 bounded trace buffer 已实现

`native/runtime/memory_trace.hpp/.cpp` 提供固定容量的 `MemoryTraceBuffer`，当前 context 默认容量为 4096。
它只保存数值化的 phase/site hash、epoch、upper/actual、committed/reserved/pending、stage/slot、status 和
monotonic timestamp；不在 callback 线程写文件，也不在 ring 中保存可变长字符串。当前已记录：

```text
admitted/running/draining/succeeded/failed/quarantined state
explicit epoch transition
site reserve
site commit / async retire / completion
checkpoint / pressure
failure
```

ring 覆盖会设置 `trace_overflowed=true` 并累计 `trace_dropped_events`，但 trace 本身不会改变 admission 结果。
`MemoryExecutionContext::drain_trace()` 供 owner/evidence runner 在 context 销毁前导出；成功的 generate/prepare
结果现在也会携带 `memory_trace` 数组，失败请求仍由错误/disposition 路径返回，不能假设失败 trace 已自动持久化。

## 230. 可执行的 scheduler 事务与 overlap 设计

### 230.1 三条逻辑队列和依赖关系

即使底层只有一个 Metal command queue，也要在 trace 中区分：

```text
Q_io       文件读取、解压、pack、host staging
Q_gpu      upload、compute、command buffer
Q_reclaim  completion 消费、lease retire/cache/free
```

推荐依赖图：

```text
Q_io:       read(i+1) -> decode(i+1) -> ready(slot)
                                      |
Q_gpu:      compute(i) --------------> wait(ready) -> upload/compute(i+1)
                 |
Q_reclaim:       event(i) -> retire(slot i) -> complete(slot i)
```

`Q_reclaim` 不是第三个 GPU queue；它是 owner thread 对真实 completion 的消费路径。若 backend 使用多个 Metal
queue，必须把 queue identity 加入 completion token，并在 `generation` 不匹配时拒绝 stale callback。

### 230.2 预取事务的准入公式

预取不以“当前空闲 bytes”判断，而以未来同时存活集合判断：

```text
prefetch_upper = upload_upper + converted_upper + staging_upper
future_live = current_live - proven_last_use + prefetch_upper
allow_prefetch iff future_live + framework_upper + baseline <= B
```

若 `future_live` 只能通过启发式 last-use 得出，则不得在 certified route 使用；可在 plan-only route 记录
`prefetch_disabled_reason=unproven_last_use`。当水位进入 Tight，先将 lookahead 从 2/3 降到 1；进入 Critical，停止
可选 prefetch 并 drain 已提交 completion；不能直接 free event 尚未完成的 slot。

### 230.3 H3 的建议事务边界

H3 首发 candidate 应固定为：

| 阶段 | 必需 site 类别 | 释放条件 | 首发策略 |
|---|---|---|---|
| text encoder | weights、conditioning、activation | encoder completion + host conditioning commit | encoder weights 不跨阶段保留 |
| DiT prefix | pinned weights、conversion scratch | prefix upload/last-use event | pinned prefix 只取 manifest 允许的最大值 |
| DiT suffix | refill slot、activation、command temporary | 每个 block event | 双槽优先，三槽需独立 capability record |
| latent handoff | latent、Euler velocity、audio/video state | 最后 denoise step event | 不与 VAE 权重重复计 backing |
| Video VAE | weights、tile scratch、decoded host envelope | decoder queue drain | tile 方案在 decode 开始前冻结 |
| export | output、media staging | file write completion | 输出必须按真实 shape commit |

H3 代码接入顺序建议为 `h3_session.mm` 建 context → `h3.c` 建阶段 → `h3_dit.c`/`h3_dit_schedule.c`
为每个 block/slot 提供 site → `h3_video_vae.c` 和 `h3_taeh3.c` 补 queue drain → `h3_gpu.m` 回传真实
completion。先闭合 common path，再处理量化、ANE/Core ML 等可选分支；不可达分支应有 source contract。

### 230.4 LTX 的建议事务边界

LTX 必须把 stage1/stage2、video/audio queue、Gemma encoder、VAE/helper 视为不同 owner：

- stage1 和 stage2 不共享同一 instance id；如果共享 backing，manifest 必须给出不重叠 alias interval；
- audio/video latent 的 queue event 分开记录，任何一个 queue pending 都不能 finish success；
- MLX array 只有在实际 materialize 后才 commit，不能把 safetensors mmap 的虚拟地址当作 GPU backing；
- helper/child process 的 footprint 进入 `ChildProcessEnvelope` 或 capability 的 framework upper，helper 失败时优先回收，
  无法证明回收完成则 quarantine worker；
- Transformer→VAE 边界必须先完成 stage2 completion drain，再切换 context epoch 和 VAE site。

## 231. 模型 manifest 与 plan 的具体写法

manifest 不是“所有可能 allocation 的列表”，而是**某个 candidate key、shape bucket、schedule revision 下的可执行
allocation 闭包**。建议每个模型至少生成以下字段：

```json
{
  "schema": "turbocider.memory_manifest.v1",
  "candidate_id": "h3-gpu-bf16-streamed-v1",
  "checkpoint_digest": "sha256:...",
  "backend_revision": "h3-metal-...",
  "runtime_revision": "memory-runtime-v3",
  "sites": [
    {
      "site_id": "h3.dit.refill.slot0",
      "component": "dit",
      "stage": "denoise",
      "class": "refill_slot",
      "lifetime": "block",
      "provenance": "exact_metadata",
      "required": true,
      "asynchronous": true,
      "aliasable": false
    }
  ],
  "instances": [
    {
      "site_id": "h3.dit.refill.slot0",
      "instance_id": 0,
      "upper_bytes": 8388608,
      "live_begin": 200,
      "live_end": 201,
      "alias_group": 0,
      "last_use_event": "h3.dit.block.0.complete"
    }
  ]
}
```

字段命名以 C++ 结构为准；JSON serializer 可以使用 snake_case，但必须保持 canonical 排序。`upper_bytes` 的 provenance
必须来自 exact tensor metadata、确定性 shape formula 或经过签核的 envelope；`HeuristicNotExecutable` 只能出现在
plan-only manifest，不能进入 production registry。

每个 candidate 还需要生成 `required-site-closure.json`：

```json
{
  "source_revision": "git:...",
  "manifest_digest": "sha256:...",
  "required": ["h3.dit.refill.slot0", "h3.dit.refill.slot1"],
  "observed": ["h3.dit.refill.slot0", "h3.dit.refill.slot1"],
  "missing": [],
  "unexpected": [],
  "dynamic_unknown": 0,
  "status": "closed"
}
```

closure 报告必须从源码 inventory、manifest 和 trace 三方交叉生成；只检查 manifest 自洽不够，因为一个漏标的
allocator call-site 仍可能让计划看起来很漂亮。

### 231.1 H3、LTX、Flux、Z-Image 的首发策略

| 模型 | 第一阶段可执行路线 | 暂不承诺 | 必须先完成的证据 |
|---|---|---|---|
| H3 | text→DiT 双槽→VAE component-staged | 三槽、ANE 混合、未闭合量化分支 | required-site closure、真实 Metal completion、VAE/output peak |
| LTX | Gemma→stage1/2→VAE，原生 Metal buffer admission | MLX graph temporary 全自动精确归因、helper 共享 cache | 双 queue drain、helper envelope、process-tree peak |
| Flux | plan-only metadata probe、component-staged 候选 | 未验证逐层 pager、LoRA/ANE 组合硬预算 | checkpoint probe、LoRA identity、quality/peak campaign |
| Z-Image | GGUF shard/component-staged plan-only | mmap 等价于 streaming、在线切换量化 | shard identity、GPU-only route、VAE tile envelope |

Flux 和 Z-Image 即使模型权重理论上可分片，也不能因为“能读取 shard”就登记为 executable。必须先证明每个
materialization、conversion、LoRA merge 和 output path 都受同一 ledger 约束。

## 232. 结果协议、trace 和诊断要求

`result.json` 应将“计划承诺”“账本事实”“进程观测”“swap 观测”“失败状态”分开，避免一个 `peak_bytes`
字段被误解：

```json
{
  "memory_policy": {
    "enabled": true,
    "user_limit_bytes": 17179869184,
    "buffer_percent": 15,
    "effective_budget_bytes": 14602888806,
    "planned_process_upper_bytes": 13958643712,
    "framework_upper_bytes": 268435456,
    "allocation_ceiling_bytes": 13690208256,
    "manifest_digest": "sha256:...",
    "plan_digest": "sha256:...",
    "runtime_revision": "memory-runtime-v3"
  },
  "memory_actual": {
    "ledger_peak_committed_bytes": 0,
    "peak_process_footprint_bytes": 0,
    "peak_unattributed_process_footprint_bytes": 0,
    "peak_observed_over_budget_bytes": 0,
    "pending_release_count": 0,
    "site_allocation_count": 0,
    "observed_within_budget": true
  },
  "swap": {
    "swapouts_begin": 0,
    "swapouts_end": 0,
    "swapouts_delta": 0,
    "source": "vm_stat"
  },
  "failure": {
    "execution_state": "succeeded",
    "failure_disposition": "none",
    "worker_quarantined": false
  }
}
```

示例数值只表示字段关系，不是 H3/LTX 的认证结果。`allocation_ceiling_bytes` 是扣除 framework 后的 concrete
ledger ceiling；`planned_process_upper_bytes` 才包含 baseline + framework + concrete。序列化时保持整数类型，
disabled 模式保留兼容字段但不创建 sampler/ledger，额外字段可为 `null` 或 `disabled`，不能报告伪造的零测量。

trace 每一行至少包含：`monotonic_ns`、`request_id`、`phase`、`epoch`、`site_id`、`instance_id`、`event`、
`upper_bytes`、`actual_bytes`、`committed`、`reserved`、`pending`、`process_footprint`、`pressure_state`、
`queue`、`generation`。推荐事件集合：`admission.begin/end`、`reserve`、`commit`、`upload.begin/end`、
`compute.begin/end`、`retire`、`completion.post/drain`、`cache`、`release`、`checkpoint`、`pressure.transition`、
`failure`、`quarantine`。

## 233. 代码修改清单（按 PR 可拆分）

下面的拆分让每个 PR 都有独立验收点，且不会把低内存模式的风险扩散到默认路径。

### PR-1：runtime-v3 API 固化（核心已落地，后续补 schema/version）

- 冻结 `CompiledAllocationInstance`、site constraint、completion token 的字段顺序和 canonical digest；
- 给 `MemoryCapabilityRecord` 增加 schema/version 校验，拒绝 framework upper 缺失；
- 增加显式 `MemoryExecutionState`，保留旧结果字段（已落地）；
- 补充 `memory_execution_test.cpp` 的所有状态转移和 destructor 未完成测试（状态转移已覆盖，destructor 负向测试待补）；
- 增加 context-aware `MemoryStageLease::reserve_site()` 和 constraint-aware overload（已落地）；
- 验收：disabled path 编译产物和 call count 与基线一致。

### PR-2：H3 required-site closure

- 在 `h3_dit.c`、`h3_dit_schedule.c`、`h3_text_encoder.c`、`h3_video_vae.c`、`h3_taeh3.c` 建立稳定 site ID；
- `h3_gpu.m` 为每个 async release 携带 allocator domain、generation、stage、slot；
- `h3_session.mm` 在 stage boundary 显式 `enter_epoch()`、drain completion、checkpoint；
- 自动生成 source inventory 与 closure report；
- 验收：无 Metal 设备时只能跑 fixture/contract；有设备时每个 required site 至少出现一次 reserve→commit→completion。

### PR-3：LTX 双队列和 helper envelope

- `ltx_gpu.m` 绑定 command identity 与 completion token；
- `ltx_blocks.c` 为 stage1/stage2/refill/audio/video 建立分离 site；
- `ltx_session.mm` 将 Transformer→VAE 边界设为 drain-safe-point；
- helper process 输出 `ChildProcessEnvelope`，失败时进入 worker quarantine；
- 验收：任意一个 queue 有 pending 都不能 finish success。

### PR-4：watchdog/trace/evidence runner（watchdog 与 bounded trace 已落地）

- 新增 enabled-only sampler 和 bounded ring（已落地，当前为 owner-polled watchdog）；
- 新增 `MemoryTraceBuffer` 以及 state/epoch/site/completion/checkpoint 事件（已落地）；
- 统一生成 `request/device/checkpoint/manifest/plan/result/trace/process-memory/vm-swap/quality` artifact；
- `evidence_digest` 绑定每个文件 SHA-256、工具版本、source/build revision；
- 验收：采样器异常、ring overflow、swapout 增量和 completion failure 都能在结果中定位。

### PR-5：Flux/Z-Image plan-only adapter

- 只实现 metadata-only probe、manifest 生成、candidate key 和 rejection reason；
- 不把任何 synthetic record 放进 production registry；
- 先完成 component envelope 和质量/峰值 campaign，再决定是否实现逐层 pager；
- 验收：未命中 verified record 时固定 `execution_supported=false`。

## 234. 验收矩阵与通过定义

### 234.1 L0：静态和纯逻辑门禁

每次提交必须运行：

```sh
env MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" \
  TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
python3 tests/native/test_memory_probe.py
python3 tests/native/test_memory_manifest.py
python3 tests/native/test_memory_plan_compiler.py
python3 tests/native/test_memory_accounting.py
python3 tests/native/test_memory_scheduler.py
python3 tests/native/test_memory_execution.py
python3 tests/native/test_contract.py
git diff --check
```

L0 通过条件：全部 exit 0；contract 的 fixture skip 必须有明确原因；没有把 SKIP 计为 Metal 通过；production
registry 仍为空或变更经过单独 review。

### 234.2 L1：synthetic/fault campaign

必须注入并验证：

| 故障 | 预期结果 |
|---|---|
| site 缺失/类别错误/upper 超限 | reserve 失败，错误可定位到 site |
| instance 重复/epoch 倒退/越过 live_end | manifest 或 execution 失败 |
| alias lifetime overlap | manifest 拒绝 |
| generic reserve bypass | constrained context fail-closed |
| actual_bytes > upper | commit 回滚，reservation 不泄漏 |
| stale generation/domain completion | mailbox 标记 stale，context 不成功 |
| mailbox overflow | sticky failure/quarantine，不静默丢 callback |
| framework unattributed overage | checkpoint 失败并 tainted |
| primary failure + cleanup failure | 保留 primary，disposition 至少 quarantine |

### 234.3 L2：真实 Metal 单请求

在每个目标 device family、runtime revision、shape bucket 上，至少完成：

1. 20 次 warm request + 100 次 measured request；
2. resident、1-slot、2-slot（以及模型声明的其他候选）分别记录端到端时间；
3. 每次 request 的 plan peak、ledger peak、process peak、unattributed peak 和 swapout delta；
4. 正常、cancel、allocation failure、completion failure 四种结果；
5. success/cancel/error 后 zero-state 与 worker reuse/quarantine 证据。

L2 通过：每次 measured request 都满足 `actual_process_peak <= B`、`swapouts_delta == 0`、required-site closure
100%、completion 计数闭合、质量结果在合同阈值内。任何一次越线都不能用均值或 P99 覆盖。

### 234.4 L3：低内存 campaign

在隔离 worker、低背景噪声下使用 16/24/32 GiB 等目标预算（实际设备可用值以 campaign 记录为准），对每个
candidate 运行 shape/steps/LoRA/audio/tiling 组合。必须记录 OS 版本、device、并发进程、swap counter 初值和终值。

L3 通过要求：

- 所有成功 request 均无新增 swapout；
- 预算越小只导致确定性降级或 fail-closed，不出现随机 OOM/swap 抖动；
- disabled ABBA 性能和 footprint 在既有门槛内；
- 与默认模式相比质量合同不变，除非请求明确选择已认证 tiling；
- evidence digest 可从原始 artifact 重算，reviewer 的十项问题全部回答“是”。

## 235. 默认路径隔离与回滚策略

### 235.1 disabled add-on 的零副作用合同

`enabled=false` 或 memory object 缺失时：

- 不调用 `preflight_memory_capability()`、`probe_memory_capability()`、`observe_process_memory()`；
- 不创建 `MemoryExecutionContext`、ledger、scheduler、watchdog、trace ring；
- 不增加 H3/LTX drain、cache clear、synchronization 或 allocator branch；
- 不改变默认 residency、dtype、采样步数、queue topology 和 worker reuse；
- 结果只保留向后兼容字段，不报告“受限模式成功”。

静态 contract 要把这些条件写成源码可搜索的门禁；性能 ABBA 还要用相同模型、shape、seed、warmup 和 measured
次数测量 wall time、process footprint、GPU/Metal command 数。

### 235.2 constrained route 的回滚

发现 runtime 逻辑 bug 时，允许回滚到上一个 runtime revision，但不允许把 constrained request 自动改成默认
unbounded route。安全行为是：

```text
capability mismatch / plan missing -> reject request
completion uncertain               -> quarantine worker
watchdog unavailable               -> reject constrained request
disabled request                   -> unchanged default route
```

feature gate 只控制是否展示候选和是否允许实验 record，不能绕过 site、framework、completion 或 swap 门禁。

## 236. 最终实现建议

当前最稳妥的工程路线不是“在现有 swap 行为上加一个更激进的阈值”，而是把 memory-constrained 作为一个
独立、可审计的 execution contract：

1. 请求层只定义 Y、X、slot 上限和质量允许的 tiling；
2. probe 层从真实 checkpoint 建立 candidate key、manifest 和 framework envelope；
3. planner 用 live interval、instance 和 alias 计算全请求峰值；
4. admission 将 framework upper 与 concrete ledger 分离；
5. adapter 以 site reservation 包住每一次真实 GPU/host allocation；
6. scheduler 以 epoch、event、completion token 管理 overlap 和 offload；
7. pressure controller 只做停止预取、边界重规划和 fail-closed，不把 swap 当容量；
8. result/evidence 同时证明计划、实际峰值、completion closure、zero-state 和 swapout delta；
9. 只有真实 Metal L2/L3 和独立 review 完成后，才向 production registry 添加 record。

这样做的代价是首发模型和 shape 范围会比较窄，但能够保证低内存机器上的行为是“可预测地降级或拒绝”，而不是
偶发进入 swap；内存充足机器在 add-on 关闭时仍保持现有路径，在 add-on 开启且计划选择 full-resident 时也不会
强制逐层搬运。该边界正是“低内存可用”和“高内存零回归”同时成立的必要条件。

## 237. 2026-09-16 续修：把 Y/X 预算合同、代码接线和模型估算写成可施工规范

前面的章节已经确定了方向，但实现团队还需要一份不会产生歧义的“数字合同”。本节把请求中的 `Y`、冗余比例
`X`、系统保留量、framework envelope、具体 allocation ledger 和 denoiser 子预算分开定义；同时给出当前工作树
中各个文件应该如何继续改造。本文中带有“建议新增”字样的类型或字段尚未表示已经存在，不能在 reviewer 报告中
当成已实现事实。

### 237.1 Y、X、B、S 四个数字必须分层

约定：

```text
Y = 用户输入的进程内存上限（bytes）
X = buffer_percent，内部冗余百分比，当前合法范围 5..30
B = effective_budget = floor(Y * (100 - X) / 100)
S = min_free_bytes，系统级最低可用内存保留量
```

`Y` 是用户意图，不等于允许 allocator 直接吃满的数；真正交给 admission/ledger 的预算是 `B`。`S` 不从 `B`
中重复扣减，而是一个独立的系统压力门槛：当 process footprint 尚未越过 `B`，但系统可用内存低于 `S` 时，
watchdog 仍然必须进入 `Critical`，停止预取并在安全边界拒绝继续运行。

当前代码对应关系如下：

| 概念 | 当前字段/函数 | 语义 | 是否允许 adapter 覆盖 |
|---|---|---|---|
| Y | `MemoryConstrainedConfig::limit_bytes` | 请求级上限 | 只能来自请求/profile overlay |
| X | `MemoryConstrainedConfig::buffer_percent` | 预留冗余 | 只能来自请求/profile overlay |
| B | `EffectiveMemoryPolicy::effective_budget_bytes` | 受限模式可执行预算 | 不允许模型自行修改 |
| S | `MemoryConstrainedConfig::min_free_bytes` | 系统保留量 | 不允许模型自行降低 |
| planned upper | `EffectiveMemoryPolicy::planned_upper_bytes` | baseline + 计划增量的保守上界 | 只能由 plan compiler 产生 |
| framework upper | `EffectiveMemoryPolicy::framework_upper_bytes` | 未归因 framework/process envelope | 只能由 capability record 注入 |
| denoiser budget | `EffectiveMemoryPolicy::denoiser_budget_bytes` | 扣除 baseline、VAE、输出等后的子预算 | 由 policy 归一化，不由 adapter 重写 |

应在文档、结果 JSON 和测试中统一使用 bytes 的整数值；GiB 只用于 UI/日志展示。所有乘法、加法、减法都要经过
checked arithmetic，溢出必须返回 `memory_estimate_unknown` 或 `memory_budget_too_small`，不能 wrap 成一个较小的
预算。

### 237.2 admission 的最终不等式

对一个已经命中 capability record 的 constrained request，必须同时满足：

```text
baseline_process + planned_increment <= B
planned_process_upper                         <= B
concrete_ledger_ceiling = planned_process_upper - framework_upper
concrete_ledger_ceiling                      >= 0
unattributed_process_footprint                <= framework_upper
system_available                             >= S
actual_process_peak                          <= B
swapouts_delta                                == 0
```

其中第一、二行是计划阶段条件，第三、四行是 runtime accounting 条件，第五行是 watchdog 条件，第六行是实机
验收条件。`framework_upper` 不能被当成“额外可分配空间”；它只是允许 framework 自身、allocator metadata、
Metal/MLX 临时对象等未能逐项归因的 envelope。任何 concrete site 都必须进入 ledger，不能因为有较大的
framework upper 就绕过 `reserve_site()`。

### 237.3 add-on 开关与默认路径的判定顺序

建议把判定顺序固定为下面的单向流程，避免某个模型 adapter 先创建了缓存又发现自己不支持 constrained：

```text
parse request
  -> overlay profile
  -> validate raw constrained fields
  -> if !enabled: return legacy/default plan immediately
  -> resolve candidate key
  -> metadata-only probe
  -> capability lookup
  -> compile manifest/plan
  -> observe clean baseline
  -> authorize plan against B/S/framework
  -> create MemoryExecutionContext
  -> bind adapter hooks
  -> begin_running
  -> run staged schedule
  -> begin_draining + backend drain
  -> finish_success or fail/quarantine
```

`!enabled` 分支不得执行 probe、`session.unload()`、`mx::clear_cache()`、watchdog、trace ring 或额外的
`checkpoint()`。当前 `prepare_memory_execution()` 已经在 enabled 路径建立 clean model/cache boundary；后续
修改必须保持该调用只位于 constrained 分支。

## 238. 计划编译器的详细实现建议

### 238.1 推荐增加的内部类型（非 ABI 公共字段）

以下类型可放入 `native/runtime/memory_plan.hpp` 的 `detail` 命名空间，避免影响现有 C ABI：

```cpp
struct MemoryBudgetVector {
    uint64_t process_baseline = 0;
    uint64_t framework_upper = 0;
    uint64_t concrete_upper = 0;
    uint64_t system_reserve = 0;
    uint64_t user_limit = 0;
};

struct CompiledStageBoundary {
    uint64_t epoch = 0;
    std::string stage_id;
    bool requires_gpu_drain = false;
    bool permits_prefetch = false;
    bool permits_cache = false;
};

struct MemoryPlanRejection {
    std::string code;
    std::string site_id;
    uint64_t requested_bytes = 0;
    uint64_t available_bytes = 0;
    uint64_t epoch = 0;
    std::string explanation;
};
```

这些字段只用于内部诊断和 evidence；对外结果仍使用稳定的 `reason`、`admission_state` 和 trace 数值字段。
如果最终需要将 rejection 传出 API，先升级 result schema，再加入兼容字段，不要直接暴露 C++ `std::string` 布局。

### 238.2 live interval 编译算法

`compile_memory_plan()` 应按以下顺序实现，所有排序都使用 canonical 字典序，以保证相同输入产生相同
`plan_digest`：

1. 校验 manifest schema、runtime revision、checkpoint digest 和 backend revision；
2. 按 `(site_id, instance_id)` 排序并拒绝重复 instance；
3. 校验 `live_begin < live_end`，并把 epoch 0 保留为 admission/initial boundary；
4. 对每个 alias group 按 `live_begin` 排序，若半开区间重叠则拒绝；
5. 对每个 epoch 先应用 `live_end == epoch` 的 release，再应用 `live_begin == epoch` 的 acquire；
6. 计算每个 epoch 的 active set、aggregate upper、最大 simultaneous instance 数；
7. 将 required site 的缺失、未知 provenance、实例数不足分别编码为不同 rejection；
8. 加上 `process_baseline` 与 `framework_upper`，得到 `planned_process_upper_bytes`；
9. 比较 `planned_process_upper_bytes <= B`，生成 `fits_budget` 和 `peak_live_set`；
10. 对 canonical manifest 和所有编译字段做 digest，digest 输入必须包含 runtime revision。

伪代码：

```cpp
for (const auto& epoch : epochs) {
    for (auto id : ending_at[epoch.id])
        live.erase(id);                         // release first
    for (auto id : starting_at[epoch.id])
        live.insert(id);                        // acquire second

    uint64_t concrete = 0;
    for (auto id : live) {
        concrete = checked_add(concrete, instance[id].upper_bytes);
    }
    const uint64_t process_upper = checked_add(
        checked_add(process_baseline, concrete), framework_upper);
    update_peak(epoch.id, process_upper, live);
}
```

实际实现中要按 alias group 去重 backing；逻辑 view 若确实共享 backing，manifest 必须明确表达 shared storage，
不能简单把两个 instance 的 upper 相加，也不能通过删除 overlap 校验来“修复”峰值。

### 238.3 candidate 选择与确定性降级

candidate 选择必须是纯函数：同一个 `MemoryCandidateKey`、预算和 runtime revision 只能得到一个结果。推荐顺序：

```text
full_resident
  -> streamed slots = min(request.max_refill_slots, verified_slots)
  -> quality-preserving tiling (only if request allows and verified)
  -> lower temporal tile / spatial tile bucket
  -> reject memory_budget_too_small
```

每次降级都产生 `degradation_reason`、`from_candidate`、`to_candidate` 和新的 manifest digest。禁止在 runtime
压力升高时静默改变 dtype、采样步数或随机种子；这些属于质量合同，必须作为 candidate key 的一部分。runtime 只可
减少 lookahead、减少 refill slots 或采用已认证 tiling。

## 239. scheduler v1 的逐步执行合同

### 239.1 scheduler 输入与输出

建议把 `MemoryStageScheduler::observe()` 继续保持无 I/O、无 allocator 副作用，并在未来增加一个纯数据输出结构：

```cpp
struct MemoryScheduleAction {
    MemoryPressureState state = MemoryPressureState::Normal;
    unsigned target_refill_slots = 0;
    unsigned target_lookahead = 0;
    bool stop_optional_prefetch = false;
    bool request_stage_boundary = false;
    bool request_gpu_drain = false;
    bool abort_request = false;
    const char* reason = nullptr; // static string or interned table entry
};
```

v1 可以继续用现有 `MemoryPressureDecision` 对外，内部由 adapter 把动作解释为 schedule 参数。不要让 scheduler
直接调用模型的 `unload()` 或 Metal API；queue owner 必须执行这些动作并在完成后回报 token。

### 239.2 水位状态与滞回

建议使用预算百分比和系统可用量双重判定：

```text
Normal:   known <= 0.80*B and process <= 0.85*B and available >= S + margin
Tight:    known <= 0.92*B and process <= 0.97*B
Critical: known > 0.92*B OR process > 0.97*B OR available < S
Abort:    process > B OR framework overage sticky OR watchdog critical
```

百分比只用于 pressure controller，不改变硬 admission 上限。为避免在临界值附近来回切换：

- 从 Tight/ Critical 恢复到 Normal 至少需要连续 3 个 safe-point 样本；
- `watchdog_critical`、ring overflow、swapout 增量和 framework overage 是 sticky，不能靠恢复样本清除；
- Critical 时只允许完成已提交的当前 stage，不允许创建新的可选预取；
- Abort 时不再提交新的 GPU work，转入 `Draining`，然后根据 completion 结果进入 `Failed` 或 `Quarantined`。

### 239.3 owner-thread schedule loop

模型 adapter 的主循环应接近下面的结构：

```cpp
for (uint32_t step = 0; step < request.steps; ++step) {
    context.enter_epoch(epoch_for_step(step), "denoise.step.begin");
    context.checkpoint("denoise.before_prefetch");

    auto action = choose_action(context.scheduler().pressure_state());
    if (action.stop_optional_prefetch)
        prefetch_window = 0;
    else
        prefetch_window = std::min(prefetch_window, action.target_lookahead);

    for (auto block : blocks_for_step(step)) {
        context.enter_epoch(epoch_for_block(step, block), "block.begin");
        auto slot = reserve_and_load_block(context, block, prefetch_window);
        submit_block_compute(slot);
        retire_previous_slot_after_last_use(step, block);
        if (block % checkpoint_stride == 0)
            context.checkpoint("denoise.block");
    }

    submit_step_epilogue();
    context.checkpoint("denoise.step.complete");
}
```

`reserve_and_load_block()` 必须先建立 `MemorySiteToken`，再执行文件读取后的真实 GPU allocation；如果 I/O 读取
本身需要大块 host staging，也必须用 host memory site 先 reserve。一个 block 的 token 未进入 `retire_async()` 前，
不能因为“逻辑上已经不用”而直接 free backing。

### 239.4 overlap 的安全上限

首发只推荐：

| 资源 | Normal | Tight | Critical |
|---|---:|---:|---:|
| refill slots | 2（或 candidate 上限） | 1 | 0/当前 slot |
| I/O lookahead | 1 | 0 | 0 |
| 可缓存权重 | 认证值 | 仅当前 stage | 禁止新 cache |
| completion drain | stage boundary | 每 block/小批次 | 立即 safe point |

三槽、lookahead=2、跨 VAE/DiT 的持久权重缓存都必须是独立 capability record，不能因为机器有更多空闲内存就
自动开启。这样可以把“更快”与“更安全”拆成可测量、可回滚的 candidate。

## 240. 源码级修改清单（函数粒度）

下面的清单面向实现者，不要求一次性提交；每一项都应有对应测试或 source contract。

### 240.1 请求、policy、plan

| 文件 | 函数/区域 | 修改建议 | 验收 |
|---|---|---|---|
| `native/platform/apple/request.mm` | `parse_memory_constrained` | 拒绝负数、浮点、未知字段；bytes 统一转 `uint64_t`；记录 `specified_fields` | 非法 JSON 全部 fail-closed |
| `native/runtime/memory_policy.cpp` | `validate_memory_constrained_request` | 保持 X=5..30、slots=1..3；增加 checked `Y*(100-X)`；Y 不得大于物理内存 | overflow、Y=0、S>=physical 全覆盖 |
| `native/runtime/memory_policy.cpp` | `resolve_memory_constrained_candidate` | candidate key 必须包含 checkpoint/device/runtime/shape/sampler/tiling；禁止按 model 名称单独命中 | 同模型不同 checkpoint 不串用 |
| `native/runtime/memory_plan.cpp` | `compile_memory_plan` | 添加 source/manifest/plan digest；输出 peak live set 和 rejection site | 相同输入 digest 稳定 |
| `native/runtime/plan.cpp` | `make_plan` | enabled 路径先 normalize 再 module validation；disabled 路径不触发 memory probe | disabled call-count 为零 |

`parse_memory_constrained()` 当前只接受白名单字段。若未来增加 `lookahead`、`temporal_tile` 等字段，必须先加入
`MemoryConstrainedField`、overlay、digest 和测试，不能让 adapter 从普通环境变量读取隐式开关。

### 240.2 execution、accounting、scheduler

| 文件 | 函数/区域 | 修改建议 |
|---|---|---|
| `native/runtime/memory_execution.cpp` | `checkpoint` | 保持 admission 与 watchdog 复用同一 observation；记录 pressure/action trace |
| `native/runtime/memory_execution.cpp` | `enter_epoch` | 生产 adapter 在 stage boundary 显式调用；自动推进仅保留兼容和测试用途 |
| `native/runtime/memory_accounting.cpp` | `try_reserve_site` | 错误中带 site、epoch、instance upper；actual>upper 时保证 reservation 可回滚 |
| `native/runtime/memory_scheduler.cpp` | `observe` | 保持无分配/无阻塞；ring overflow、swapout、framework overage 作为 sticky signal 输入 |
| `native/runtime/memory_trace.cpp` | `record/drain` | 只写定长数值事件；字符串 hash 必须稳定，禁止 callback 中写文件 |
| `native/runtime/memory_watchdog.cpp` | `sample` | owner-polled；不要新增后台线程；critical 首次原因只写一次 |

建议为 `MemoryAdmissionMetrics` 再增加以下字段，但在 schema 升级前只作为内部候选：

```text
peak_epoch
peak_site_count
peak_reserved_bytes
peak_pending_release_bytes
degradation_reason
schedule_action_count
completion_generation_mismatch_count
```

### 240.3 C API 与失败 evidence

`native/api/c_api.mm` 当前已经在 generate/prepare 成功路径导出 metrics 和 trace。下一步要把失败路径也纳入
evidence，建议修改为：

```cpp
try {
    result = session.generate(...);
    execution->begin_draining();
    drain_memory_execution(session, *execution);
    execution->checkpoint("complete");
    execution->finish_success();
    evidence.write_success(execution->metrics(), execution->drain_trace());
} catch (const std::exception& error) {
    const auto primary = error.what();
    finalize_memory_failure(engine, &session, execution.get(), primary);
    evidence.write_failure(
        primary, execution ? execution->metrics() : std::nullopt,
        execution ? execution->drain_trace() : std::vector<MemoryTraceEvent>{});
    throw;
}
```

`evidence.write_failure()` 不应在 constrained request 失败时重新 probe 或清 cache；它只能消费已经采样的 metrics、
trace、process memory 和 swap counter。若 evidence 写入失败，不得把原始 runtime failure 改写为 success；结果中
保留 `evidence_write_failure` 诊断并按 cleanup 结果决定 worker 是否 quarantine。

### 240.4 H3 文件级接线

H3 当前已经具备 GPU/host memory hooks。后续应按下面的稳定 site 族完成 closure，而不是继续使用
`h3_gpu_tensor` 这种兜底 tag：

| 源文件 | site 族 | 接线规则 |
|---|---|---|
| `h3_text_encoder.c` | `h3.text.weight.*`、`h3.text.activation.*`、`h3.text.conditioning.*` | encoder 开始前 epoch；输出 conditioning commit 后释放 encoder temporary |
| `h3_dit_schedule.c` | `h3.schedule.*` | host control/sigma/time buffer 必须走 host hooks；schedule 销毁前完成 release |
| `h3_dit.c` | `h3.dit.weight.*`、`h3.dit.activation.*`、`h3.dit.refill.*`、`h3.dit.convert.*` | block/slot instance 与 `stage_id/slot_id` 一一对应 |
| `h3_video_vae.c` | `h3.vae.decode.*`、`h3.vae.tile.*` | DiT→VAE 先 drain，再进入新 epoch；tile scratch 不得跨 tile 泄漏 |
| `h3_taeh3.c` | `h3.taeh3.*` | TAE 输出与 RGB decode envelope 分开计费 |
| `h3_gpu.m` | backing/completion | `allocator_domain + generation + handle` 必须在 callback 中原样校验 |

在 required closure 完成前，任何未分类的 quant cache、Core ML conversion、ANE prefix/suffix 或 debug readback
都应使 candidate 保持 `plan_only`，而不是标记为 `execution_supported`。

### 240.5 LTX 文件级接线

| 源文件 | site 族 | 额外要求 |
|---|---|---|
| `ltx_gemma_encoder.m` | `ltx.gemma.weight.*`、`ltx.gemma.activation.*` | 动态 prompt 的 raw/packed/mask capacity 按最大 rows 计 upper |
| `ltx_blocks.c` | `ltx.stage1.*`、`ltx.stage2.*`、`ltx.refill.*` | stage1/stage2 不复用 instance id；每个 block completion 后才能 retire |
| `ltx_gpu.m` | upload/compute backing | video/audio queue 的 generation 分开；stale callback 必须可定位 |
| `ltx_session.mm` | session boundary | Transformer→VAE、helper 启停、MLX materialize 都是 checkpoint |
| `ltx_mlx_*` | graph/cache envelope | 未能精确归因的 graph temporary 只能进入 framework upper，且必须有 validated envelope |
| `ltx_video_vae.h/.m` | `ltx.video_vae.*` | helper/child process footprint 进入 `ChildProcessEnvelope` |

当前 LTX 还有 MLX cache、audio/video parallel、helper process 等复杂分支。首发 candidate 应固定为
`c_metal + video_only + component_staged`，其它分支维持 `plan_only` 或显式 unsupported。

## 241. 模型内存估算与 manifest 生成模板

### 241.1 通用估算公式

任何 site 的 upper 都必须由下列一种来源产生：

```text
ExactMetadata:      checkpoint tensor metadata × bytes_per_element + alignment
ExactShapeFormula:  shape dimensions × dtype size + deterministic scratch margin
ValidatedEnvelope:  measured p99 + signed safety margin + framework evidence
Heuristic:          仅用于 plan-only，不得执行
```

建议统一使用：

```cpp
bytes = align_up(checked_mul(element_count, element_size), allocator_alignment);
upper = checked_add(bytes, site_margin_bytes);
```

`site_margin_bytes` 必须按 backend/allocator 固定，不得按当前 free memory 动态放大。对于异步 command buffer、
Metal heap、MLX lazy graph，margin 进入对应 site 或 framework envelope，不能无证据地写成 0。

### 241.2 H3 估算维度

H3 的 candidate key 至少包含：`width`、`height`、`frames`、`fps`、文本 token rows、audio rows、dtype、quant
mode、refill slots、VAE tile。建议按以下 site 计算：

```text
text conditioning = rows × hidden × bf16 + rope/maps + host copies
denoiser activation = max(video_tokens, audio_tokens) × hidden × dtype × live buffers
refill slot = block_weight_upper + conversion_scratch + upload staging
latent = video_latent + audio_latent + velocity/state
VAE tile = tile_pixels × channels × f32/bf16 × encoder/decoder live factor
output = decoded RGB + export staging + ffmpeg handoff envelope
```

H3 的 `h3.dit.activation.*` 数量较多，不能简单把所有历史 tag 的 upper 相加；应通过 block live interval 和
alias group 计算 simultaneous set。`h3.dit.convert.*` 若与 activation 发生短暂重叠，必须在 manifest 中显式
重叠；不能因为转换很快就假设零峰值。

### 241.3 LTX 估算维度

LTX 需要额外纳入：stage1/stage2 block 数、video/audio token 数、Gemma rows、convrot group、MLX cache capacity、
VAE tile、helper process envelope。建议把以下变量固化到 shape bucket：

```text
video_tokens = ceil(width / patch_w) × ceil(height / patch_h) × temporal_frames
audio_tokens = ceil(audio_length / audio_stride)
conditioning_rows = min(max_prompt_rows, actual_rows + padding)
refill_upper = max(block_weight, converted_block, command_staging)
vae_graph_upper = validated_envelope(shape_bucket, tiling_mode)
helper_upper = measured_child_process_p99 + safety_margin
```

对于 MLX lazy evaluation，`mmap` 文件大小不代表 materialized GPU/Unified-memory footprint；必须在实际 materialize
后用 probe/ledger 对账。无法把 graph temporary 分解到 site 时，candidate 只能在 `ExperimentalGuarded`，不能进
release registry。

### 241.4 Flux 和 Z-Image 的接入策略

Flux 和 Z-Image 在当前工作树没有完整的 production memory hooks，建议先完成 metadata-only manifest generator：

1. 读取 checkpoint tensor metadata、dtype、shape 和模块边界；
2. 生成 component-level site（text encoder、transformer、VAE、control/LoRA、output）；
3. 估算 resident、single-component-staged、double-buffer 三个候选；
4. 只输出 plan、peak、rejection reason，不创建 GPU backing；
5. 在真实 Metal completion、quality parity 和 swapout=0 证据齐全后再实现 adapter。

## 242. 完成、取消、错误和 quarantine 的详细时序

### 242.1 成功时序

```text
Admitted
  -> begin_running
  -> enter_epoch(stage0)
  -> reserve -> allocate -> commit
  -> submit GPU work
  -> retire_async(token)
  -> backend completion callback post_completion
  -> safe-point drain_completion_mailbox
  -> checkpoint(last stage)
  -> begin_draining
  -> backend drain_memory_completions
  -> unload/cache teardown while context bound
  -> final mailbox drain
  -> finish_success
```

成功条件是所有 reservation、storage、pending、completion、cache、unknown 和 site allocation 都为零；
`finish_success()` 的失败必须视为实现 bug 或 backend contract violation，不能转成“部分成功”。

### 242.2 cancel 时序

取消只阻止新工作提交，不立即 free 正在被 GPU 使用的 backing：

```text
cancel flag observed
  -> stop prefetch/new block
  -> retire logically dead leases
  -> wait/drain all submitted queues
  -> post/drain completion tokens
  -> release remaining host/GPU storage
  -> finalize_failure("cancelled")
  -> complete_failure_cleanup(true)
```

如果 backend 不支持在取消后证明所有 command buffer 已完成，必须 `QuarantineWorker`。不能为了复用 worker 而调用
全局 `synchronize()` 之外的未验证释放路径。

### 242.3 allocation failure 与 completion failure

| 故障 | primary reason | 允许的后续动作 | 最终状态 |
|---|---|---|---|
| reserve denied | `memory_budget_too_small` | 不分配 backing，清空已提交资源 | `Failed/Clean` 或 `Quarantined` |
| allocator returns null | `memory_policy_allocation_failed` | cancel reservation，排空已提交 queue | 同上 |
| actual > upper | `memory_lifetime_violation` | 立即 taint，禁止继续 reserve | 通常 `Quarantined` |
| stale completion | `memory_lifetime_violation` | 不消费 stale token，隔离 worker | `Quarantined` |
| mailbox overflow | `memory_lifetime_violation` | 停止新提交并 drain；证据不完整则隔离 | `Quarantined` |
| process/framework overage | `memory_pressure_abort` | 停止预取，进入 draining | `Failed` 或 `Quarantined` |

primary failure 必须保留，即使 cleanup 也失败。结果中同时输出 `failure_reason` 和 `cleanup_failure`，reviewer 才能
区分“预算不够”和“回收不可信”。

## 243. evidence artifact 设计和 reviewer 操作手册

### 243.1 目录与最小文件

建议每次 constrained request 生成一个不可变 artifact 目录（失败也生成）：

```text
evidence/<request_id>/
  request.json
  normalized_request.json
  device.json
  capability.json
  manifest.json
  plan.json
  result.json
  memory_metrics.json
  memory_trace.jsonl
  process_memory.jsonl
  vm_swap.json
  quality.json
  cleanup.json
  digest.json
```

`memory_trace.jsonl` 可以由固定大小事件数组序列化生成；写文件发生在 owner thread/context 销毁前或专用
evidence runner 中，不能在 Metal callback 中做文件 I/O。失败请求即使没有 result，也必须有 `result.json`，其
`execution_state` 至少为 `failed` 或 `quarantined`。

### 243.2 digest 和重放

`digest.json` 至少绑定：

```text
runtime_revision
source_revision
device_identity
checkpoint_digest
manifest_digest
plan_digest
request_digest
result_digest
trace_digest
toolchain_version
```

reviewer 应能删除 `digest.json` 后从其余文件重新计算相同摘要。任何依赖当前机器路径、时间、指针地址的字符串
都不能进入 canonical digest；trace 中的 site/phase 使用稳定 hash，原始名称只在符号表中保存。

### 243.3 reviewer 必答问题

每个 candidate 发布前必须逐项回答：

1. Y、X、B、S 的数值和单位是否清楚？
2. process baseline 是否在 clean boundary 采样？
3. manifest 是否覆盖所有 required allocation call-site？
4. 是否存在 generic reserve 或 unknown external 绕过？
5. `framework_upper` 的来源、版本和上限是什么？
6. 所有 async release 是否有 generation/domain/token？
7. success/cancel/error 后是否 zero-state？
8. 低预算是否只确定性降级或拒绝？
9. swapout delta 是否逐请求为 0？
10. disabled ABBA 是否无可测额外开销？

## 244. 测试与验收扩展方案

### 244.1 L0：新增纯逻辑测试

除已有测试外，建议新增以下 case：

| 测试 | 输入 | 预期 |
|---|---|---|
| budget arithmetic | `Y=max_uint64`, `X=15` | overflow rejection |
| buffer boundary | X=4、31 | request validation failure |
| baseline exclusion | baseline > planned upper | admission failure |
| framework separation | concrete + framework > B | plan rejection |
| digest stability | 相同 manifest 不同 insertion order | digest 相同 |
| candidate isolation | 同 model 不同 checkpoint/device | 不命中旧 record |
| deterministic downgrade | 相同压力序列重复 100 次 | action 序列完全相同 |
| trace overflow | capacity=1，多事件 | overflow sticky 且结果可见 |
| destructor failure | context 未 finish | failure/quarantine 可见 |

### 244.2 L1：fault injection 序列

不要只注入单点故障，还要验证组合顺序：

```text
reserve(slot0)
  -> commit(slot0)
  -> retire(slot0)
  -> stale completion(slot0)
  -> primary allocation failure(slot1)
  -> cleanup failure
```

预期是保留 primary allocation failure、记录 stale completion 和 cleanup failure，并最终 quarantine；不能因为
slot0 的旧 callback 到达而误释放 slot1 或把请求标记 success。

### 244.3 L2/L3 实机指标

每个 request 必须记录至少以下列：

```text
request_id, candidate, device, shape_bucket, Y, X, B, S,
plan_peak, ledger_peak, process_peak, unattributed_peak,
min_system_available, swapouts_delta, pressure_transitions,
io_wait_seconds, gpu_wait_seconds, completion_count,
quality_digest, execution_state, failure_disposition
```

通过标准是逐请求硬约束，不使用均值覆盖异常：

```text
process_peak <= B
unattributed_peak <= framework_upper
min_system_available >= S   （或按明确的 Critical/Fail 规则记录）
swapouts_delta == 0
completion_count_posted == completion_count_drained
success => all zero-state counters == 0
```

### 244.4 本机无 Metal 时的边界

当前环境没有可用 Metal device，因此只能宣称 L0/L1 和源码 contract 通过；H3/LTX GPU hooks、真实 allocator peak、
completion latency、swapout delta 和 L3 低内存 campaign 必须标记 `SKIP: no Metal device is available`。文档和
release note 不得把 skip 改写成 pass。

## 245. 性能回归保护与调优顺序

### 245.1 disabled ABBA

默认模式回归测试要使用相同：checkpoint、shape、seed、warmup、测量次数、并发度和 cache 状态。至少比较：

- wall time 与 p50/p95；
- process footprint peak；
- Metal command buffer 数；
- model load/unload 次数；
- host↔GPU copy bytes；
- worker reuse rate。

disabled 路径不应出现 `MemoryExecutionContext`、`MemoryWatchdog`、trace ring、额外 `session.unload()` 或
`mx::clear_cache()`。如果编译器无法完全消除 branch，至少通过 call-count contract 和 ABBA 证明没有行为变化。

### 245.2 enabled 模式调优顺序

调优顺序固定为：

1. 先保证 zero-state 和 swapout=0；
2. 再测 resident 与 single-slot 的质量/速度基线；
3. 再开启双槽 overlap，观察 I/O 与 GPU wait；
4. 再调 checkpoint stride、lookahead 和 tile size；
5. 最后才考虑三槽、跨 stage cache 或 MLX graph reuse。

任何优化如果增加 `framework_upper`、unknown bytes、pending release 或 completion mismatch，都必须退回 plan-only，
不能以 wall time 改善为理由进入 release。

## 246. 建议的实现提交顺序和每个提交的验收门

```text
M0  文档/字段/错误码冻结
    -> schema、digest、Y/X/B/S 单测

M1  runtime-v3 enforcement
    -> plan instance、site constraint、execution state、watchdog、trace

M2  H3 common path closure
    -> text/DiT/VAE/output source inventory + host/GPU completion

M3  H3 real-device L2
    -> 20 warm + 100 measured，quality/swap/zero-state

M4  LTX C/Metal video-only closure
    -> stage1/stage2、audio/video queue、helper envelope

M5  LTX real-device L2
    -> same evidence contract

M6  failure evidence runner
    -> success/cancel/error artifact 可重算 digest

M7  low-memory L3 campaign
    -> 16/24/32 GiB 或实际设备等价预算

M8  capability registry review
    -> 只加入真实 evidence 对应的第一条 production record

M9  Flux/Z-Image plan-only
    -> 未命中 verified record 永不执行
```

任何提交都不应同时做 runtime enforcement、模型结构重构和性能优化。这样可以在出现失败时快速定位是 budget、
manifest、backend completion 还是模型质量问题。

## 247. 最终操作结论（本轮续修）

对 TurboCider，推荐的内存受限模式不是“让操作系统更积极地 swap”，而是一个 add-on execution contract：

```text
Y/X -> B
  -> clean baseline + framework envelope
  -> manifest/live interval plan
  -> reserve-before-allocate
  -> epoch/slot/event scheduling
  -> owner-polled pressure/watchdog
  -> completion-safe offload
  -> zero-state + swapout=0 evidence
```

内存充足机器在 add-on 关闭时保持旧路径；add-on 开启但选择 resident candidate 时也不应被强迫逐层搬运；只有
命中已验证的 streamed candidate 才启用逐层 loading/offloading。任何估算不完整、completion 不可证明、framework
envelope 未签核或真实 Metal 证据缺失的路径，都应返回 plan-only/unsupported，而不是自动退回 unbounded 默认路径。

## 248. 显式 epoch 映射：避免 adapter 依赖“自动推进”

当前 `MemoryExecutionContext::try_reserve_site()` 在当前 epoch 不匹配、但未来存在合法 instance 时，会自动进入最早
合法 epoch。这个行为适合兼容和单元测试，却不应成为 production adapter 的正常控制流，否则错误的调用顺序会被
“刚好存在的未来 instance”掩盖。

### 248.1 semantic epoch key

不要让模型代码硬编码容易冲突的整数。建议新增一个只在 plan/adapter 内使用的 semantic key：

```cpp
enum class MemoryStageKind : uint8_t {
    TextEncoder,
    DenoiserLoad,
    DenoiserBlock,
    DenoiserStepEnd,
    LatentHandoff,
    VideoVae,
    AudioVae,
    Export,
};

struct MemoryEpochKey {
    MemoryStageKind stage;
    uint32_t step = 0;
    uint32_t block = 0;
    uint32_t tile = 0;
    uint16_t subphase = 0;
};
```

plan compiler 对所有 `MemoryEpochKey` 做 canonical 排序后分配单调递增 `uint64_t epoch`，并输出
`MemoryEpochTable`。adapter 只能通过 resolver 查询：

```cpp
const uint64_t epoch = compiled_plan.epochs.resolve({
    MemoryStageKind::DenoiserBlock, step, block, 0, kComputeBegin});
context.enter_epoch(epoch, "h3.dit.block.compute.begin");
```

这样可以保证数字 epoch 仍适合 ledger/trace，而模型代码使用稳定的语义 key。若 resolver 找不到 key，应立即
`memory_plan_invalid`，不能退回自动推进。

### 248.2 H3 建议 epoch 图

```text
text.load
  -> text.encode
  -> text.conditioning.handoff
  -> dit.load.prefix
  -> for each step:
       dit.step.begin
       -> for each block:
            block.prefetch
            -> block.upload
            -> block.compute
            -> block.retire
       -> dit.step.epilogue
  -> latent.handoff
  -> vae.load
  -> for each tile:
       vae.tile.upload -> vae.tile.compute -> vae.tile.retire
  -> export
```

`block.prefetch` 与前一个 block 的 `compute` 可以时间重叠，但其 live interval 必须重叠计入 peak。prefix weight 的
`live_end` 是其真实最后使用 event，不是“suffix 开始”的逻辑猜测。

### 248.3 LTX 建议 epoch 图

```text
gemma.load -> gemma.encode -> conditioning.handoff
  -> stage1.load
  -> for each stage1 step/block: prefetch -> video/audio compute -> retire
  -> stage1.to.stage2.handoff
  -> stage2.load
  -> for each stage2 step/block: prefetch -> video/audio compute -> retire
  -> transformer.drain
  -> video_vae.load -> tile loop
  -> audio_vae.load -> chunk loop             (audio candidate only)
  -> helper.drain
  -> export
```

首发 video-only candidate 可以不生成 audio VAE epoch，但必须让 candidate key 和 manifest 明确 `audio=false`；不能
生成 audio site 后在运行时静默跳过 required closure。

### 248.4 自动推进的收敛策略

当前已按 capability gate 收敛：

1. `MemoryCapabilityRecord` 已增加 `require_explicit_epoch`，所有可执行 record 注册时必须为 true；
2. `compile_memory_plan()` 把该布尔值纳入 plan digest，认证 plan 遇到 allocation 驱动的未来 epoch 会直接拒绝；
3. 非认证 compatibility plan 仍可使用自动推进，但 metrics 会记录 `automatic_epoch_transition_count`；
4. compatibility flag 不进入请求 JSON，也不能来自普通环境变量；production registry 当前为空，因此不会误放行尚未
   接入显式 stage/block epoch 的 H3/LTX adapter。

## 249. LTX GPU hook ABI v2：补齐 retire/complete 生命周期

截至 2026-09-16，下面描述的 ABI v2 已在当前工作树实现：`ltx_gpu_memory_hooks` 已追加 `retire/complete`，
`ltx_gpu_set_memory_hooks_for_queue()` 已提供逻辑 queue identity，batch drain 会在真实 Metal completion 之后调用
`complete()`，并由 owner thread drain scheduler mailbox。version 1 前缀仍为默认/同步调用方保留。

这里的“已实现”只表示源码、编译与无 GPU 单测合同已闭合；当前机器没有 Metal device，尚不能据此把 LTX 标为
L2/L3 certified。required-site closure、stage1/stage2 显式 epoch、helper/framework envelope 和低内存 swapout evidence
仍是发布阻塞项。

### 249.1 ABI 追加方式（已落地）

保留现有前缀字段不变，在结构尾部追加：

```c
typedef struct {
    uint32_t struct_size;
    uint32_t version;
    void *user;
    int (*reserve)(...);
    int (*commit)(...);
    void (*cancel)(...);
    void (*release)(...);          /* v1 prefix */

    int (*retire)(void *user, void *token,
                  uint32_t queue_id, uint32_t stage_id,
                  uint32_t slot_id, char *error, size_t error_size);
    void (*complete)(void *user, void *token,
                     uint32_t queue_id, int status); /* v2 tail */
} ltx_gpu_memory_hooks;
```

兼容规则：

- `struct_size` 小于 v2 tail 或 `version < 2` 时，默认模式继续使用 v1 release；
- constrained execution 必须要求完整 v2，并校验所有函数指针非空；
- `release()` 只允许用于从未提交给 GPU 的 committed backing，或已经同步完成的 CPU/host storage；
- 已提交 GPU 的 token 必须 `retire()` 恰好一次，然后由真实 command completion 调用 `complete()` 恰好一次；
- `complete()` 只 `post_completion()`，不释放 model state、不等待 queue、不抛异常。

### 249.2 `ltx_gpu_buffer` 状态机（已落地的语义）

`native/models/ltx_runtime/ltx_gpu.m` 当前按下面的 token 生命周期处理 batch 最后 owner：

```text
Reserved -> Committed -> Submitted -> Retired -> CompletionPosted -> Released
      \-> Cancelled
Committed(no submit) -------------------------------> Released
```

`ltx_gpu_buffer_free()` 遇到 `Submitted` 状态时不能直接调用 v1 `release()`；它必须把 token 交给最后引用该 buffer 的
command buffer completion handler。若一个 buffer 被多个 command buffer/queue 使用，要维护 last-use sequence，只有
最大 sequence 完成后才能 complete。

### 249.3 video/audio queue identity（已落地）

当前代码定义的稳定 queue id 为：

```c
enum {
    LTX_GPU_MEMORY_QUEUE_MAIN = 1,
    LTX_GPU_MEMORY_QUEUE_VIDEO = 2,
    LTX_GPU_MEMORY_QUEUE_AUDIO = 3,
    LTX_GPU_MEMORY_QUEUE_TEXT = 4,
    LTX_GPU_MEMORY_QUEUE_VAE = 5,
};
```

queue id 必须进入 completion token 或 trace event。即使 video/audio 当前落在同一个 Metal command queue，也要保留
逻辑 identity，避免未来并行化后复用旧 evidence。

### 249.4 LTX ABI v2 测试（已落地，实机部分待补）

新增或扩展：

```text
tests/native/ltx_gpu_memory_hooks_test.mm
  - v1 默认模式兼容
  - constrained 拒绝 v1
  - reserve-before-MTLBuffer
  - submitted buffer 不直接 release
  - real completion 后 complete
  - video/audio queue id 不串线
  - stale generation/domain quarantine
  - drain 后 token/ledger/mailbox 全为零
```

## 250. capability identity v2：需要继续绑定的维度

当前 `MemoryCandidateKey` 已包含 adapter、model/checkpoint、backend、dtype、operation、shape、sampler、slots、tiling、
runtime revision 和 device family。发布前建议增加：

```cpp
std::string schedule_revision;
std::string framework_profile_digest;
std::string helper_mode;
std::string process_topology;
std::string platform_build_class;
std::string allocator_revision;
std::string graph_cache_mode;
```

字段语义：

| 字段 | 为什么影响认证 |
|---|---|
| `schedule_revision` | last-use、slot 数、prefetch 顺序改变会改变峰值 |
| `framework_profile_digest` | MLX/Metal 未归因 envelope 随版本变化 |
| `helper_mode` | in-process 与 child-process VAE 的计费方式不同 |
| `process_topology` | 单 worker、helper tree、sidecar 的 process peak 不同 |
| `platform_build_class` | macOS/Metal framework 的 allocator 行为可能变化 |
| `allocator_revision` | alignment、heap/cache policy 改变 actual capacity |
| `graph_cache_mode` | MPSGraph/MLX cache 是否跨 block/stage 保留 |

因为 production registry 当前为空，现在仍是修改 canonical key 的低风险窗口。若这些字段加入 digest 或 capability
匹配语义，应升级 key schema，并视兼容范围决定是否把 `memory-runtime-v3` 升为下一 revision；绝不能在保留旧 digest
的同时改变字段含义。

不建议把精确 OS patch、机器序列号、绝对模型路径写入 candidate key。它们应进入 evidence device record；capability
匹配使用经过 campaign 定义的 `platform_build_class` 和 `device_family`，既避免过度碎片化，也不跨越未验证边界。

## 251. required-site closure 生成器的可实现方案

### 251.1 不依赖正则猜测 allocator call-site

只扫描字符串字面量容易漏掉宏拼接、动态 tag 和 wrapper。建议为每个 adapter 建一个可编译的 site registry：

```c
#define H3_MEMORY_SITES(X) \
    X("h3.text.conditioning.token_ids", CONDITIONING, STAGE, REQUIRED) \
    X("h3.dit.refill.slot0", REFILL_SLOT, BLOCK, REQUIRED) \
    X("h3.dit.refill.slot1", REFILL_SLOT, BLOCK, REQUIRED) \
    X("h3.vae.decode.input_upload", ACTIVATION, TILE, REQUIRED)
```

同一个 X-macro 同时生成：

1. C/Objective-C 中的稳定 tag 常量；
2. probe/manifest builder 的 `AllocationSiteSpec`；
3. source inventory JSON；
4. tests 中的 expected required set。

LTX 使用相同模式。禁止 allocator 调用直接手写新的字符串；新增 site 必须先进入 registry，否则编译或 contract test
失败。动态 block/slot 不应动态拼 site id，而应固定 site id + instance id/slot id。

### 251.2 closure 三方集合

closure 工具读取：

```text
S_source    编译生成的 source inventory
S_manifest  candidate manifest 中 required sites
S_trace     campaign 中解析 symbol table 后实际观察 sites
S_exempt    明确不可达或 optional 分支，带 source revision 和理由
```

判定：

```text
missing_manifest = S_source(required) - S_manifest - S_exempt
unknown_manifest = S_manifest - S_source
unobserved        = S_manifest(required) - S_trace - S_exempt
unexpected_trace  = S_trace - S_manifest
closed iff 四个集合均为空且 dynamic_unknown == 0
```

`unobserved` 不能只靠一次 happy path 清零。campaign 必须覆盖 candidate key 允许的所有 optional branch；如果 candidate
明确禁止 LoRA/audio/quant/control，则这些分支应在 request validation 阶段拒绝，而不是用 exemption 掩盖。

### 251.3 建议工具与输出

建议新增：

```text
tools/memory/generate_site_inventory.py
tools/memory/build_manifest.py
tools/memory/verify_required_site_closure.py
tests/native/test_memory_site_inventory.py
```

输出 `required-site-closure.json` 除已有字段外再包含：

```json
{
  "candidate_key_digest": "sha256:...",
  "schedule_revision": "...",
  "source_inventory_digest": "sha256:...",
  "trace_count": 100,
  "covered_shape_buckets": ["..."],
  "covered_branches": ["video_only", "two_slot"],
  "exemptions": [],
  "status": "closed"
}
```

## 252. 失败报告的一次性快照接口

截至 2026-09-16，一次性、幂等 terminal report 已经实现。成功和失败路径都可在 context 进入终态后调用
`take_report()`；第一次调用 drain bounded trace 并缓存，后续调用返回同一快照，不重新 probe 或修改 scheduler。
当前结构为：

```cpp
struct MemoryExecutionReport {
    MemoryAdmissionMetrics metrics;
    std::vector<MemoryTraceEvent> trace;
};

class MemoryExecutionContext {
  public:
    MemoryExecutionReport take_report();
  private:
    std::optional<MemoryExecutionReport> terminal_report_;
};
```

已实现规则：

- 第一次 `take_report()` 在 owner thread 抓取 metrics 和 bounded trace；watchdog 摘要已经并入 metrics；
- 后续调用返回同一份 immutable report，不重新 probe；
- report 只能在 `Succeeded/Failed/Quarantined` 终态或 cleanup 已完成后生成；
- report 生成失败不覆盖 primary runtime error；
- report 内存上限由 trace capacity 决定，禁止包含任意大小的日志字符串。

### 252.1 C API 向后兼容方案（已落地）

现有 `tc_engine_generate()` 的 `error` 继续保持普通字符串；当前已新增可选查询接口，没有改变旧 ABI 的错误格式：

```c
int tc_engine_take_last_memory_report_json(
    tc_engine *engine, char **out, char **error);
```

行为：

- 只有最近一次 enabled constrained request 生成 report；
- disabled/default 请求不创建也不覆盖 report；
- take 后由 engine 清除，避免跨请求误读；
- engine quarantine 后仍允许读取 report；
- service 层收到 generate/prepare 错误时，立即调用该接口并落 evidence artifact。

如果不希望扩展 C ABI，也可以由 `services/turbociderd/service.mm` 注入内部 evidence sink；但 sink 同样必须是
request-scoped、bounded、失败不改写 primary error。

### 252.2 敏感信息与稳定性

failure report 不写 prompt、绝对 checkpoint 路径、指针地址或模型权重内容。允许写 request id、shape bucket、digest、
site/phase hash、字节数、状态、时间戳和错误分类。site/phase hash 的算法和 seed 必须固定并记录 revision，否则不同
build 的 trace 无法对比。

## 253. 发布前的最终阻塞清单

截至 2026-09-16，下面任一项未完成都不能加入 production capability record：

- H3 required-site closure 不是 `closed`；
- H3 真实 Metal completion、cancel、allocation/completion failure 没有 L2 artifact；
- LTX hook v2 虽已接入，但尚无真实 Metal completion/failure L2 artifact；
- LTX stage1/stage2、video/audio queue 或 helper envelope 未闭合；
- failure request 虽可读取 terminal metrics/trace，但 service 尚不能持久化完整 evidence artifact/cleanup/swap 文件集；
- candidate key 未绑定 schedule/framework/helper/process topology；
- 16/24/32 GiB 或等价低内存 campaign 没有逐请求 `swapouts_delta == 0`；
- disabled ABBA 发现额外 unload、cache clear、probe、watchdog 或性能回归；
- 任何 synthetic capability record 出现在 production registry。

当前可继续宣称的是：runtime-v3 的纯逻辑 enforcement、owner-polled watchdog、bounded trace、execution state 和
现有 contract/unit tests 已落地；当前不能宣称 H3/LTX 已在真实 GPU 和低内存机器上 release-ready。

## 254. 2026-09-16 实现状态同步：后续施工必须以源码事实为基线

本节覆盖前文中已经被当前工作树实现推进所超越的描述。它不改变“production capability registry 必须保持为空”
这一发布边界，只把后续工程任务建立在准确的源码状态上。

| 子系统 | 当前状态 | 可以宣称 | 仍不能宣称 |
|---|---|---|---|
| 请求/policy | `memory_constrained`、Y/X/B/S、candidate normalization 已接入 | 配置解析和 fail-closed 路由存在 | 任意模型都可执行受限模式 |
| manifest/plan | site、instance、live interval、alias、peak 编译已存在 | 纯逻辑 peak 可确定性重算 | manifest 已覆盖真实 allocator 全部 call-site |
| ledger/scheduler | reserve/commit/retire/completion mailbox、site constraint 已存在 | allocation 前 admission 和异步释放状态机已存在 | 所有 framework/MLX allocation 都能逐项硬拦截 |
| execution/watchdog | explicit state、owner-polled watchdog、bounded trace 已存在 | 失败可进入 clean/quarantine 终态 | 已在真实低内存机器证明不 swap |
| H3 hooks | GPU/host hooks、async completion bridge 已接入 | H3 allocator 有受控入口 | H3 required-site closure、真实 Metal L2 已完成 |
| LTX hooks | v2 `retire/complete`、queue identity、sticky completion failure 已接入 | LTX C/Metal token 生命周期可表达 | LTX stage/helper/framework envelope 已认证 |
| terminal report | `take_report()` 和 additive C API 已接入 | 成功/失败终态可消费 metrics+trace | service 已自动落完整 evidence artifact |
| capability registry | release registry 为空 | 不会误放行 synthetic record | 任一 candidate 已 release-ready |

当前本机可重复验证状态：

```text
native build                                      PASS
tests/native/test_memory_execution.py             PASS
tests/native/test_contract.py                     81 PASS / 1 fixture SKIP
tests/native/test_ltx_gpu_memory_hooks.py          compile PASS / runtime SKIP(no Metal)
git diff --check                                  PASS
```

因此下一阶段的核心不是继续添加更多宽泛的 `reserve()` 调用，而是完成五条闭环：

1. semantic epoch 从 manifest 一直贯通到 H3/LTX 真实 stage/block 边界；
2. source inventory、manifest、trace 形成 required-site closure；
3. capability key 绑定 schedule/framework/helper/process topology；
4. terminal report、swap telemetry 和服务层 evidence writer 形成可审查证据包；
5. 在真实 Metal 与低内存机器上完成 L2/L3 campaign，之后才允许增加 production record。

## 255. 下一阶段总体施工图与依赖关系

后续 patch 不应按“模型名”平铺，而应按下列依赖图施工。上层 patch 在下层合同未冻结前不能自行定义另一套 key、
site 或 evidence schema。

```text
P1 schedule/epoch schema v2
   ├── P2 capability identity v2
   ├── P3 H3/LTX explicit epoch bridge
   └── P4 plan digest / schedule revision

P5 compiled source-site registry
   ├── P6 H3 required-site closure
   ├── P7 LTX required-site closure
   └── P8 closure verifier + symbol table

P9 swap/process telemetry
   └── P10 terminal evidence writer
         └── P11 L2/L3 campaign runner

P3 + P6 + P9  -> H3 experimental guarded candidate
P3 + P7 + P9  -> LTX experimental guarded candidate
L2 + L3 + independent review -> production capability record
```

推荐把 patch 控制在以下边界：

- P1/P2 只改 runtime schema、digest 和纯单测，不碰模型数学；
- P3 只传递 stage/block/tile 事件，不改变计算顺序；
- P5/P6/P7 只把 allocation 分类为稳定 site，不顺便做性能重构；
- P9/P10 只采样和落证据，不影响 disabled 路径；
- overlap、slot 数和 cache 生命周期优化必须在 closure 之后独立提交。

这样 reviewer 可以明确区分：错误来自 schema/digest、估算、allocator 接线、completion，还是性能优化。

## 256. schedule schema v2：semantic epoch 的完整代码设计

### 256.1 目标和非目标

目标是让“当前执行到哪个模型语义边界”成为 planner、adapter、trace 和 capability identity 共同理解的事实，避免
`try_reserve_site()` 根据未来 instance 自动猜测 epoch。semantic epoch 只描述资源生命周期和安全点，不描述模型
数学，也不改变采样器输出。

首版不需要支持任意 DAG。H3/LTX 当前单请求 GPU execution 可以编译为全序 epoch；I/O、upload、compute 的并行性
通过 live interval 重叠和 queue event 表达，而不是让 epoch 自身变成并发图。

### 256.2 建议新增类型

在 `native/runtime/memory_plan.hpp` 中增加：

```cpp
enum class MemoryStageKind : uint8_t {
    Admission,
    TextEncoder,
    ConditioningHandoff,
    DenoiserLoad,
    DenoiserBlock,
    DenoiserStepEnd,
    LatentHandoff,
    VideoVae,
    AudioVae,
    Export,
    TerminalDrain,
};

enum class MemorySubphase : uint8_t {
    Begin,
    Prefetch,
    Upload,
    Compute,
    Retire,
    Drain,
    End,
};

struct MemoryEpochKey {
    MemoryStageKind stage = MemoryStageKind::Admission;
    MemorySubphase subphase = MemorySubphase::Begin;
    uint32_t step = 0;
    uint32_t block = 0;
    uint32_t tile = 0;
    uint32_t branch = 0;

    bool operator==(const MemoryEpochKey &) const = default;
    bool operator<(const MemoryEpochKey &) const;
    std::string canonical() const;
};

struct ScheduleEpoch {
    uint64_t id = 0;
    MemoryEpochKey key;
    std::string name;
    bool runtime_safe_point = false;
    bool requires_gpu_drain = false;
    bool permits_prefetch = false;
};
```

`branch` 用稳定枚举值表示 video/audio/text/helper，而不是使用线程 id 或 queue pointer。所有字段都必须进入 plan
canonical digest。`MemoryEpochKey::canonical()` 只使用固定 ASCII 字段和值，不受 locale、路径或指针影响。

`CompiledMemoryPlan` 再增加：

```cpp
std::string schedule_revision;
bool require_explicit_epoch = false;
uint64_t resolve_epoch(const MemoryEpochKey &) const;
const ScheduleEpoch &epoch(uint64_t id) const;
```

实现可继续使用按 key 排序的 `std::vector` 和 binary search；adapter 热路径不需要 `std::map` 动态分配。编译完成后
epoch table 不可变，可安全被 owner thread 和只读 callback bridge 引用。

### 256.3 manifest schema 的迁移

当前 `AllocationInstance` 只有数值 `live_begin/live_end`。建议引入
`turbocider.memory_manifest.v2`：

```cpp
struct AllocationInstance {
    std::string site_id;
    uint32_t instance_id = 0;
    uint64_t upper_bytes = 0;
    MemoryEpochKey acquire;
    MemoryEpochKey last_use;
    uint64_t alias_group = 0;
    std::string last_use_event;
};
```

编译步骤固定为：

1. 收集所有 instance 的 `acquire/last_use` key，以及 adapter 声明的 drain/safe-point key；
2. 按模型 schedule template 的全序约束排序；
3. 拒绝缺失 step/block/tile 范围、重复 key、反向 lifetime；
4. 分配从 0 开始的单调 epoch id；
5. 把 semantic lifetime 编译为现有 ledger 使用的半开区间 `[begin, end)`；
6. 对编译结果计算 `schedule_revision` 和 `plan_digest`。

v1 数值 manifest 可以继续服务纯单测或 migration tool，但 production capability record 必须声明 v2 和
`require_explicit_epoch=true`。不能把 v1 数字重新解释为新语义而保留原 digest。

### 256.4 execution context 的严格模式

`MemoryExecutionContext` 建议增加：

```cpp
void enter_epoch(const MemoryEpochKey &, std::string_view event = {});
bool at_epoch(const MemoryEpochKey &) const;
```

`try_reserve_site()` 的生产语义调整为：

```text
if current epoch matches at least one legal instance:
    reserve
else if require_explicit_epoch:
    throw memory_lifetime_violation
else:
    compatibility auto-advance + trace + metric
```

新增 metrics：

```text
explicit_epoch_transition_count
automatic_epoch_transition_count
epoch_resolution_failure_count
last_epoch
peak_epoch
```

任何用于认证的 evidence 必须满足 `automatic_epoch_transition_count == 0`。自动推进分支不得由请求 JSON、profile 或
环境变量打开，只允许测试 fixture 或旧 manifest compatibility 使用。

### 256.5 C/Objective-C adapter bridge

H3/LTX 的 C runtime 不应知道 C++ `MemoryEpochKey` 布局。建议新增一个共享、只追加字段的 C ABI：

```c
typedef struct {
    uint32_t struct_size;
    uint32_t version;
    void *user;
    int (*enter)(void *user, uint32_t stage, uint32_t subphase,
                 uint32_t step, uint32_t block, uint32_t tile,
                 uint32_t branch, char *error, size_t error_size);
    int (*checkpoint)(void *user, const char *phase,
                      char *error, size_t error_size);
} tc_memory_schedule_hooks_v1;
```

接线建议：

- 新建 `native/core/memory_schedule_c.h`，只包含 POD/整数，不依赖 C++；
- 在 `h3_gpu_options`、`h3_host_memory_options`、`ltx_native_options` 和 `ltx_gemma_encoder_options` 尾部追加可选 pointer；
- disabled/default request 传 `NULL`，C runtime 使用当前路径，不产生任何额外 callback；
- constrained request 要求非空、完整 v1；任一 callback 失败立即停止新提交；
- callback 只进入 context/checkpoint，不做文件 I/O、Metal wait 或模型释放。

不能复用用户进度 `Event` 作为唯一 schedule 信号。进度事件可能被 UI 节流、重命名或只在阶段首尾触发，不足以证明
block/slot 的最后使用。schedule hook 必须位于真实 acquire、submit、retire 附近。

### 256.6 H3 与 LTX 的最小 epoch 模板

H3 首发模板至少包含：

```text
text/begin -> text/compute -> conditioning_handoff/end
dit_load/begin
for step:
  for block:
    denoiser_block/prefetch
    denoiser_block/upload
    denoiser_block/compute
    denoiser_block/retire
  denoiser_step_end/drain
latent_handoff/drain
for vae tile:
  video_vae/upload -> compute -> retire
export/begin -> terminal_drain/end
```

LTX 首发模板至少包含：

```text
text/begin -> text/compute -> conditioning_handoff/end
stage1 load
stage1 step/block: prefetch -> video compute -> retire
stage1 drain -> latent_handoff
stage2 load
stage2 step/block: prefetch -> video compute -> retire
transformer drain
video VAE tile loop
helper drain
export -> terminal drain
```

video-only candidate 不生成 audio key。audio candidate 必须是另一条 capability key，并为 audio queue、audio VAE 和
video/audio join 生成独立 epoch；不能在同一个 certified key 下按请求布尔值静默改变 schedule。

### 256.7 schedule v2 测试

新增：

```text
tests/native/memory_epoch_test.cpp
tests/native/test_memory_epoch.py
```

必须覆盖：

- 相同 semantic key 集合，不同 insertion order 得到相同 id/digest；
- step/block/tile 变化会改变 schedule revision；
- production strict 模式拒绝 allocation 驱动的自动推进；
- 向后进入 epoch、未知 key、重复 key、空 lifetime 全部 fail-closed；
- async resource 的 `last_use` 早于 completion/drain 时拒绝编译；
- H3/LTX fixture 全程 `automatic_epoch_transition_count == 0`；
- disabled request 不构造 epoch table，不调用 schedule hook。

## 257. required-site registry：从宽泛 tag 收敛到可证明闭包

### 257.1 registry 文件布局

建议使用可被 C、C++ 和工具共同消费的 `.def` X-macro 文件：

```text
native/models/h3_runtime/h3_memory_sites.def
native/models/ltx_runtime/ltx_memory_sites.def
native/runtime/memory_site_registry.hpp
native/runtime/memory_site_registry.cpp
tools/memory/generate_site_inventory.py
tools/memory/verify_required_site_closure.py
```

`.def` 每行字段固定为：

```c
TC_MEMORY_SITE(
    SYMBOL,
    "stable.site.id",
    MEMORY_CLASS,
    LIFETIME,
    SYNC_MODE,
    REQUIREMENT,
    BRANCH,
    UPPER_FORMULA_ID)
```

示例：

```c
TC_MEMORY_SITE(H3_DIT_REFILL_SLOT,
               "h3.dit.refill.slot",
               REFILL_SLOT, BLOCK, ASYNC, REQUIRED,
               COMMON, "h3.block_weight_upper.v1")
```

动态 block/step/slot 只进入 `instance_id/stage_id/slot_id`，不拼接到 site string。这样 site set 有限、可编译，trace
仍可通过数值 identity 区分具体实例。

### 257.2 编译期使用规则

registry 同时生成：

1. C 常量或 enum，供 allocator call-site 使用；
2. `AllocationSiteSpec` 模板，供 probe/manifest builder 使用；
3. site-id 到 memory class/lifetime 的只读表，供 runtime 交叉校验；
4. inventory JSON，供 evidence 和 closure verifier 使用；
5. 测试 expected set。

模型代码禁止直接把任意字符串传给 classified allocator。迁移完成后可使用 wrapper：

```c
h3_gpu_tensor *h3_gpu_tensor_new_site(
    h3_gpu *gpu, size_t elements, h3_gpu_dtype dtype,
    h3_memory_site site, uint32_t instance_id);
```

wrapper 根据 registry 自动提供 class 和稳定 tag；debug build 对 caller 传入的 class 再做一次一致性断言。若为兼容保留
字符串 API，production constrained path 遇到 registry 未知 tag 必须返回 `memory_estimate_unknown`。

### 257.3 H3 首批 site 族

至少拆分：

| component | required site 族 | 关键 lifetime |
|---|---|---|
| text | token ids、mask、embedding weight、encoder activation、conditioning output | stage/request |
| schedule | sigma、timestep、rope/layout control | request/stage |
| DiT persistent | norms、modulation、pinned prefix weight | denoiser/request |
| DiT streamed | refill slot、converted weight、upload staging | block/command buffer |
| DiT activation | latent、velocity、attention/qkv、MLP scratch、residual | block/step |
| VAE | latent upload、tile input、graph scratch、tile output | tile/command buffer |
| export | decoded f32、RGB bytes、audio PCM、container staging | export/request |

`h3_gpu_tensor`、`h3_host_memory` 这类兜底 tag 只允许出现在 disabled 路径或开发断言中；任何 production trace 观察到
它们，closure 立即失败。

### 257.4 LTX 首批 site 族

至少拆分：

| component | required site 族 | 关键 lifetime |
|---|---|---|
| Gemma | token ids、mask、packed inputs、weights、activations、conditioning | text stage |
| stage1 | video latent、audio latent、block weight、refill、qkv/attention/MLP scratch | block/step |
| handoff | stage1 clean/output、stage2 input、conditioning cache | stage boundary |
| stage2 | 独立 block/refill/activation site，不复用 stage1 instance | block/step |
| video VAE | graph inputs、tile/chunk scratch、decoded output | tile/helper |
| audio VAE | audio latent、mel/PCM/vocoder staging | chunk/helper |
| helper | IPC payload、child envelope、result import | helper/request |

当前 `ltx_activation_tensor` 等宽泛 tag 必须逐步消失。stage1/stage2 即使 shape 相同也不能用同一 instance id，因为其
last-use、queue 和 helper handoff 不同。

### 257.5 closure verifier 的精确输入/输出

输入：

```text
source-inventory.json       编译自 .def
manifest.json               candidate 实际 manifest
trace.jsonl                 campaign 全部成功/失败路径 trace
trace-symbols.json          hash -> site/phase/revision
closure-exemptions.json     有期限、带理由和 source revision 的豁免
candidate.json              允许的 branch/shape/schedule
```

除前文四个集合外，还要检查：

```text
class_mismatch        同一 site 在 source/manifest/trace 的 memory class 不同
lifetime_mismatch     manifest lifetime 与 registry 不同
branch_escape         trace 出现 candidate key 禁止的 branch
upper_violation       observed actual > manifest instance upper
instance_overflow     同 epoch 活跃 instance 数超过计划
generic_site_seen     观察到兜底/unknown site
auto_epoch_seen       观察到自动 epoch 推进
```

最终 `status=closed` 的必要条件：以上集合全部为空、trace 无 overflow、symbol table revision 匹配，并且 campaign 覆盖
candidate key 声明的所有 branch 和 shape bucket。

### 257.6 exemption 规则

exemption 不能用于掩盖“尚未测试”。每条必须包含：

```json
{
  "site_id": "...",
  "candidate_key_digest": "sha256:...",
  "source_revision": "...",
  "reason": "compile-time unreachable because audio=false",
  "proof": "request validator rejects audio=true before session creation",
  "expires_after_revision": "..."
}
```

允许豁免的是由 candidate validation 明确禁止的分支；仅仅在本次请求没有走到的 LoRA、audio、ANE、debug readback
不能获得长期豁免。

## 258. capability identity v2：避免证据跨 schedule/平台误复用

### 258.1 key schema

建议把 `MemoryCandidateKey` 升级为显式 schema，而不是在旧 canonical string 后静默追加字段：

```cpp
struct MemoryCandidateKey {
    std::string schema = "turbocider.memory_candidate.v2";
    // existing fields...
    std::string schedule_revision;
    std::string source_inventory_digest;
    std::string framework_profile_digest;
    std::string helper_mode;
    std::string process_topology;
    std::string platform_build_class;
    std::string allocator_revision;
    std::string graph_cache_mode;
};
```

`operator==`、`operator<` 和 `canonical()` 必须同一 patch 修改，测试要检查三者字段集合一致。禁止出现 digest 已包含某
字段但 equality 未包含，或 lookup 比较字段而 canonical 未绑定的情况。

### 258.2 字段取值规范

| 字段 | 推荐取值示例 | 禁止取值 |
|---|---|---|
| `schedule_revision` | `h3-stream-2slot-v2:<digest>` | 当前时间、随机 UUID |
| `source_inventory_digest` | `.def` canonical SHA-256 | 文件绝对路径 |
| `framework_profile_digest` | 经过 campaign 签名的 envelope digest | “latest” |
| `helper_mode` | `none`、`in_process`、`disposable_child_v1` | PID |
| `process_topology` | `single_worker`、`worker+vae_child` | 机器用户名 |
| `platform_build_class` | `macos26-metal4-apple9` 这类审核枚举 | 精确序列号 |
| `allocator_revision` | `h3-metal-shared-v2` | pointer/address |
| `graph_cache_mode` | `disabled_per_request`、`stage_local_v1` | 隐式环境变量值 |

`platform_build_class` 由编译时/平台 probe 的白名单函数产生。未知 OS/Metal/device family 返回 unknown，production
lookup fail-closed；不能自动回落到最近的已认证平台类。

### 258.3 capability record 增补

建议增加：

```cpp
std::string candidate_key_digest;
std::string schedule_digest;
std::string source_inventory_digest;
std::string closure_digest;
std::string framework_profile_digest;
std::string campaign_digest;
uint64_t maximum_validated_process_peak_bytes = 0;
uint64_t maximum_validated_unattributed_bytes = 0;
uint64_t minimum_validated_system_available_bytes = 0;
bool swapout_zero_required = true;
```

`MemoryCapabilityRegistry::add()` 必须验证所有 digest 格式、cross-field 一致性和 state/level 组合。certified record 还要
要求 `closure_digest`、`campaign_digest`、`swapout_zero_required=true`。experimental record 可以没有 L3 campaign，
但仍必须有 source closure 和 L2 allocator evidence；普通请求默认不允许 experimental。

### 258.4 revision 迁移策略

因为当前 production registry 为空，推荐一次性完成 v2 key 和 `memory-runtime-v4`，避免未来维护双 lookup。迁移顺序：

1. 先更新 key/schema/digest 单测；
2. 更新 sidecar loader，旧 v1 sidecar 只允许 plan-only；
3. 更新 H3/LTX probe 生成 v2 key；
4. 重新生成 synthetic fixtures，不复用旧 digest；
5. 保持 production registry 为空，直到新 evidence 完成。

若必须保留 runtime-v3 读取能力，也只能用于显示 rejection 原因；绝不能把 v1 record 自动提升为 v2 certified。

## 259. swap telemetry：把“不进入 swap”变成逐请求证据

### 259.1 swap 不是调度资源

runtime 的正确顺序始终是：先用 B 和 S 做 admission，再用 explicit streaming/offload 控制 live set。swap counter 只
承担两个角色：

1. campaign 的硬验收信号；
2. 运行时观察到系统已经发生 swap 压力时的 sticky abort/quarantine 信号。

它不能增加 ledger budget，也不能触发“允许多分配一点，交给系统换页”的分支。即使 swap 压缩看起来暂时没有明显
延迟，也不应被当作成功。

### 259.2 当前平台接口（已落地）

为复用现有 Mach VM probe 和保持 dependency-free native tests，当前实现位于
`native/runtime/memory_accounting.hpp/.cpp`，而不是单独的 Objective-C 文件。接口为：

```cpp
struct SwapActivityObservation {
    bool available = false;
    uint64_t swapins = 0;
    uint64_t swapouts = 0;
    uint64_t compressed_pages = 0;
    std::string source;
};

using SwapActivityObserver =
    std::function<SwapActivityObservation()>;
SwapActivityObservation observe_swap_activity();
```

Apple 实现使用 `host_statistics64(HOST_VM_INFO64)` 读取 `swapins`、`swapouts` 和
`compressor_page_count`；若调用失败则返回 `available=false`，不伪造零值。执行测试通过注入
`SwapActivityObserver`，不依赖真实机器产生 swap；accounting test 另外验证本机 counter 可读且单调。

全局 `swapouts` 是系统计数，不完全等价于“本进程导致”。因此解释规则为：

- `delta == 0`：可以证明观测窗口内系统没有新增 swapout；
- `delta > 0`：该次 request/campaign 不能通过，不能宣称全部由本进程造成，也不能忽略；
- counter 回退、wrap 或 source 变化：观测无效，certification fail-closed；
- 共享机器上若存在并发高内存进程，campaign 标记 environment invalid 并重跑，不用均值抵消。

### 259.3 采样时机和开销（已落地）

只在 constrained enabled 路径采样：

```text
admission 前/clean baseline 后       begin sample
每个 stage drain boundary            optional checkpoint sample
terminal cleanup 完成后              end sample
```

不建议每个 block 调 Mach API；block 级硬约束由 reserve-before-allocate 和 process watchdog 提供。stage 边界采样足以
记录 swap 首次出现的阶段，并避免影响 hot loop。

`MemoryExecutionContext` 构造时接收 begin observation 和 observer；`checkpoint()`、`finish_success()`、failure 和
failure cleanup 边界进行 owner-thread 采样。`take_report()` 只消费已冻结的 terminal metrics/trace，不重新 probe，
因此二次读取 report 不会改变 counter。

### 259.4 metrics 与错误语义（已落地）

`MemoryAdmissionMetrics` 增加：

```text
swap_observation_available
swapins_begin
swapins_end
swapins_delta
swapouts_begin
swapouts_end
swapouts_delta
swap_counter_invalid
swap_activity_detected
swap_sample_count
compressed_pages_begin
compressed_pages_end
swap_first_observed_phase
swap_observation_source
```

运行中 `swapouts_delta > 0` 时：

1. watchdog 标记 sticky critical；
2. scheduler 停止 optional prefetch 和新 stage；
3. 当前已提交 GPU work 进入 drain；
4. request 返回 `memory_swap_activity_detected`；
5. 若 drain/zero-state 成功，可为 `Failed/Clean`；否则 `Quarantined`。

不能在首次 swapout 后继续跑完整 request 只为收集性能数据；专用 campaign 若要研究失败之后的行为，应使用单独的
诊断模式，且该模式永不生成 release evidence。

### 259.5 swap 测试（逻辑与本机 probe 已落地）

当前测试分别并入 `memory_accounting_test.cpp` 和 `memory_execution_test.cpp`，覆盖：

- 正常 counter delta；
- counter 回退/source 改变；
- unavailable；
- begin=0/end=1 触发 sticky failure；
- failure 后仍完成 GPU drain，primary reason 不被 cleanup 覆盖；
- disabled request 的 observer call-count 为 0；
- report 二次读取不重新采样 swap。

## 260. evidence writer：把 report 基础升级为完整可复核 artifact

### 260.1 当前缺口

`tc_engine_take_last_memory_report_json()` 已能提供最近一次 constrained terminal report，但当前接口本身不包含 request、
device、candidate、manifest、plan、swap 原始样本和文件摘要，也没有自动写入 service 的 evidence 目录。因此它是
证据采集基础，不是完整认证产物。

### 260.2 建议对象模型

新增：

```text
native/runtime/memory_evidence.hpp
native/runtime/memory_evidence.cpp
services/turbociderd/memory_evidence.mm
tools/memory/verify_evidence.py
```

核心接口：

```cpp
struct MemoryEvidenceInput {
    std::string request_id;
    std::string normalized_request_json;
    std::string device_json;
    MemoryCandidateKey candidate_key;
    MemoryManifest manifest;
    CompiledMemoryPlan plan;
    MemoryExecutionReport report;
    SwapObservation swap_begin;
    SwapObservation swap_end;
    std::string primary_error;
};

struct MemoryEvidenceResult {
    bool written = false;
    std::filesystem::path directory;
    std::string digest;
    std::string diagnostic;
};

MemoryEvidenceResult write_memory_evidence(
    const std::filesystem::path &root,
    const MemoryEvidenceInput &input) noexcept;
```

writer 必须是 terminal、owner-thread、bounded。它不允许重新访问模型 allocator、重新 probe checkpoint 或调用
`session.unload()`；所有输入都应在运行时已经冻结。

### 260.3 原子落盘与目录安全

写入流程：

```text
validate request_id and root
  -> create root/.tmp-<random>
  -> write bounded files
  -> fsync/close as configured
  -> compute per-file digest and root digest
  -> write digest.json last
  -> atomic rename to root/<request_id>
```

安全规则：

- request id 只允许 `[A-Za-z0-9._-]` 且长度有上限；
- 禁止符号链接逃逸、`..`、绝对子路径；
- 文件已存在时不覆盖，追加稳定 suffix 或返回 diagnostics；
- artifact 不包含 prompt、绝对 checkpoint 路径、指针、用户目录、原始权重；
- `memory_trace.jsonl` 和 samples 均有硬上限，overflow 明确写入 status；
- writer 失败不能覆盖 primary runtime error，也不能把 request 从 failure 改为 success。

### 260.4 文件 schema

建议最小集合：

```text
request.json                 去敏后的原始字段
normalized-request.json      实际执行字段和 shape bucket
device.json                  device family/platform class/physical memory
candidate.json               complete v2 key
capability.json              matched record；未命中时写 rejection
source-inventory.json        registry digest 和 site symbol table
manifest.json                canonical manifest
plan.json                    epoch、live interval、peak set、digest
result.json                  success/failure/quarantine
memory-metrics.json          admission/ledger/watchdog/swap summary
memory-trace.jsonl           bounded numeric trace
swap.json                    begin/end/delta/source
cleanup.json                 drain、mailbox、zero-state
quality.json                 parity digest/thresholds
digests.json                 全部文件摘要和 root digest
```

所有 JSON 使用 sorted keys、整数 bytes、明确 schema/version。时间戳只作为非 canonical metadata；root digest 排除
临时时间和绝对目录。

### 260.5 C API 和 service 接线

短期保持 additive C API：

1. generate/prepare 返回失败；
2. service 在持有 engine 生命周期期间立即调用
   `tc_engine_take_last_memory_report_json()`；
3. 把 report 与 service 已知的 request/device/plan 合并；
4. 写 artifact；
5. 返回原 primary error，并附加非破坏性的 evidence id。

中期建议增加内部 C++ sink，使 API 层能直接保存 manifest/compiled plan，而不是让 service 从 result JSON 反解析。
外部 C ABI 不必暴露 C++ 对象，可以新增一个只读的完整 terminal evidence JSON 查询接口。

### 260.6 report API 的直接测试

除源码 contract 外，增加一个真正调用 C API 的测试 executable：

```text
tests/native/memory_report_c_api_test.mm
tests/native/test_memory_report_c_api.py
```

用 fake `ModelSession` 覆盖：

- constrained success 可读取一次完整 JSON；
- failure/quarantine 仍可读取；
- 第二次 take 返回 unavailable；
- disabled/default request 不清除也不覆盖上一份 constrained report；
- enabled 新请求开始时清除旧 report；
- engine busy 时查询失败但 report 未丢失；
- 序列化失败不修改 primary failure。

## 261. H3 首发适配方案：从已有 hooks 到显式 schedule/closure

### 261.1 首发范围必须冻结

H3 第一条 experimental candidate 应明确固定：

```text
adapter                 h3_c_metal_streamed_v2
backend                 C/Metal
residency               streamed
refill slots            1 或 2（分别认证）
quant cache             固定模式或关闭
operation/shape/audio   由 candidate key 精确绑定
ANE/LoRA/debug dump     首发关闭，除非独立 closure/evidence
graph cache             per-request disabled
```

不要尝试用第一条 record 覆盖所有 H3/FastH3/VDN 变体。MLX、ANE、quantized cache、conditioned generation、audio
分支只要 allocation graph 不同，就应是独立 candidate。

### 261.2 逐文件修改建议

| 文件 | 修改点 | 目标 |
|---|---|---|
| `h3_memory_sites.def` | 新增全部 host/GPU stable sites | source inventory 唯一来源 |
| `h3_memory.h` | 在 options 尾部追加 schedule hooks | host allocation 前进入正确 epoch |
| `h3_gpu.h/.m` | classified API 接受 site enum/instance；completion 记录 queue/stage/slot | 消灭 generic GPU tag |
| `h3_text_encoder.c` | text begin/compute/handoff，临时 activation 及时释放 | conditioning lifetime 可证明 |
| `h3_dit_schedule.c` | sigma/time/control 全部 classified | host control 不再漏账 |
| `h3_dit.c` | block prefetch/upload/compute/retire hook | 两槽 live interval 与真实时序一致 |
| `h3_video_vae.c` | tile begin/upload/compute/retire | VAE tile upper 可认证 |
| `h3_taeh3.c` | TAE scratch/output 独立 site | 不与 full VAE envelope 混淆 |
| `h3.c` | stage drain、handoff、cancel 顺序 | 顶层生命周期闭环 |
| `h3_session.mm` | bridge semantic epoch、site enum、evidence identity | C runtime 接入 C++ context |

### 261.3 DiT 双槽 schedule

对 block `b`，推荐时序：

```text
CPU/SSD: pread block b+1 into slot[(b+1)%2]
GPU:     compute block b from slot[b%2]
after compute b last-use:
         retire slot[b%2] lease
queue completion:
         post completion token
owner safe point:
         drain mailbox, slot becomes reusable
```

关键不变量：

- 只有 completion 已 drain 的 slot 才可 refill；
- slot upper 包含 weight backing、转换 scratch、必要的 upload staging；
- prefetch 的 reservation 在真实 pread/MTLBuffer 前完成；
- prefetch 与 compute 重叠时两个 slot 都计入 peak；
- I/O 失败 cancel 尚未 commit 的 reservation；
- GPU submit 失败只 release 从未提交的 backing，已提交的走 retire/complete；
- block 最后使用事件改变时必须更新 `schedule_revision`。

### 261.4 pinned prefix 的决策

planner 枚举 pinned prefix `p`：

```text
candidate peak(p, slots) = baseline + framework
                         + persistent activations
                         + prefix_weight_upper(p)
                         + refill_slots_upper(slots)
                         + conversion/upload overlap
                         + output/conditioning live set
```

在同等质量下排序：

1. peak 必须 `<= B`；
2. 优先更多 pinned blocks；
3. 在性能证据支持时优先 2 slots，否则 1 slot；
4. tie-break 使用固定数值顺序，不根据瞬时 free memory 改变。

运行中只允许 `2 slots -> 1 slot -> no optional lookahead`，不允许增加 pinned prefix，也不允许把未认证 block 临时常驻。

### 261.5 H3 stage handoff

Text -> DiT：

- conditioning output 的 request/stage lease 保留；
- text weights/temporary completion 后 offload；
- mailbox 清零后才进入 DiT load epoch。

DiT -> VAE：

- denoiser queue drain；
- streamed/pinned weights 全部 release；
- latent/output handoff backing 继续存活；
- process checkpoint 验证 framework upper 已回落到 VAE 允许 envelope；
- 再创建 VAE graph/tile resources。

VAE -> Export：

- 每 tile scratch completion 后释放；
- decoded output 与 export staging 分开计费；
- FFmpeg/container 若由子进程执行，必须是独立 helper/process topology candidate。

### 261.6 H3 验收重点

- 每个 required site 至少在对应 branch trace 出现；
- slot0/slot1 的 instance、stage、completion 不串线；
- 1-slot 与 2-slot 输出质量完全一致或满足现有 deterministic tolerance；
- injected pread、allocation、submit、completion、cancel 故障都 zero-state 或 quarantine；
- disabled H3 的 command 数、allocation tag、load/unload 行为与修改前相同；
- 真实 Metal L2 证明 `retire -> queue completion -> mailbox drain -> reuse` 顺序。

## 262. LTX 首发适配方案：利用已完成 hook v2 收敛 stage/helper 边界

### 262.1 首发 candidate

推荐第一条仅支持：

```text
adapter             ltx_c_metal_video_streamed_v2
backend             native C/Metal denoiser
output              video only
audio               false
component staging   enabled
stage1/stage2       固定 schedule
refill slots        1 或 2，分别认证
Gemma               native path，固定 max token bucket
video VAE           固定 helper_mode/tiling
ANE/Sol/sparse      disabled
resident cache      disabled per request
```

MLX denoiser、audio、ANE MLP/KV、Sol/sparse attention、LoRA 和 conditioning cache reuse 都改变 allocation graph，首发
保持 plan-only 或独立 candidate。

### 262.2 queue identity 和 barrier

当前 enum 已包含 MAIN、VIDEO、AUDIO、TEXT、VAE。下一步要把 logical queue identity 写入 completion token 和 trace；
即使底层暂时共享一个 `MTLCommandQueue`，也不能在 evidence 中丢失逻辑 owner。

必须存在以下 barrier：

```text
TEXT drain  -> conditioning handoff
VIDEO stage1 drain -> stage1/stage2 handoff
VIDEO stage2 drain -> transformer teardown
VAE drain   -> decoded output/export
AUDIO drain -> audio/video join（仅 audio candidate）
MAIN drain  -> terminal zero-state
```

barrier 不一定每次都创建新的 Metal queue，但必须等待相应 batch 的真实 completion，并在 owner thread drain mailbox。

### 262.3 逐文件修改建议

| 文件 | 修改点 | 目标 |
|---|---|---|
| `ltx_memory_sites.def` | Gemma/stage1/stage2/VAE/helper 全部 site | closure 唯一来源 |
| `ltx_gpu.h/.m` | token trace 增 queue id；classified API 使用 site enum | v2 completion 可审计 |
| `ltx_native.h` | options 尾部追加 schedule hooks/revision | native block loop 可显式进入 epoch |
| `ltx_blocks.c` | stage/step/block/prefetch/retire hook | live interval 与真实 refill 一致 |
| `ltx_gemma_encoder.h/.m` | text queue、packed input、activation site | 文本阶段闭包 |
| `ltx_session.mm` | stage handoff、helper envelope、VAE queue | 全请求 lifecycle |
| video/audio VAE 文件 | tile/chunk site 与 completion | helper/graph upper 分开 |
| service worker | helper mode/process topology 进入 key | evidence 不跨进程拓扑复用 |

### 262.4 stage1/stage2 资源隔离

stage1 与 stage2 应使用不同 semantic site instance 和 alias group。允许复用同一物理 refill slot 的条件是：

1. stage1 queue 已完成；
2. stage1 completion token 已被 owner drain；
3. ledger 中旧 storage 已 release 或明确 reactivate 为同一 storage identity；
4. stage2 重新 reserve/commit 或走经过验证的 cached reactivation；
5. trace 能证明两个 lifetime 不重叠。

禁止只修改一个 `stage` 字段继续沿用旧 token。跨 stage 的真实 backing reuse 会影响 peak 和 schedule revision，必须
进入 manifest/plan。

### 262.5 helper 和 VAE envelope

若 video VAE 在独立进程：

- parent 不把 child footprint 当成自己的 framework upper；
- capability key 使用 `helper_mode=disposable_child_v1` 和对应 process topology；
- admission 同时检查 parent B、child envelope 和系统 S；
- evidence 分别记录 parent/child peak 和 combined system impact；
- parent 只在 child 完成并退出后释放 IPC/handoff storage；
- child crash、timeout、无法取得 terminal metrics 时 request quarantine/失败，不能以输出文件存在视为成功。

若 VAE 在同进程，使用另一条 `helper_mode=in_process` record，不得复用 child evidence。

### 262.6 LTX overlap 策略

首版 overlap 只允许：

- 当前 block compute 与下一 block SSD read/upload；
- video/audio 在独立且已认证 queue 时并行；
- Transformer drain 后再启动 VAE，不做跨 component graph overlap。

当压力进入 Tight：lookahead=0、refill slots 降为 1、每小批次 drain。Critical：不启动下一 block/stage，仅完成当前已提交
work。任何 audio/video 并行 candidate 都必须按两个 queue 同时 live 的峰值计算，不能分别取 max 后相加遗漏共享资源。

### 262.7 LTX 验收重点

- v1 默认路径兼容；constrained 强制完整 v2；
- retire 失败不发布不存在的 completion token；
- completion callback status 非零使 drain 失败并 sticky；
- stage1/stage2、TEXT/VIDEO/VAE queue identity 不串线；
- helper timeout/crash 后 parent 不保留 handoff backing；
- video-only candidate 运行时拒绝 audio=true，而不是动态扩展 manifest；
- 真实 Metal L2 和低内存 L3 通过前保持 production registry 为空。

## 263. Flux、Z-Image 与新增模型：plan-only 到可执行的认证阶梯

### 263.1 为什么不能直接复用 H3/LTX adapter

Flux 和 Z-Image 当前已有 resident、GGUF、LoRA、Core ML/MLX 等多个执行分支，但这些分支的权重布局、graph cache、
LoRA 融合时机和 VAE 生命周期不同。即使两个模型的“总权重大小”相近，也不能共享 H3/LTX 的 refill slot、framework
upper 或 completion evidence。

因此新增模型一律按以下阶梯推进：

```text
P0 metadata-only probe
  -> P1 manifest/plan golden
  -> P2 all allocation call-site registry
  -> P3 reserve-before-allocate hooks
  -> P4 async retire/complete and stage drain
  -> P5 quality parity + deterministic output
  -> P6 real Metal L2
  -> P7 low-memory L3 + swapout=0
```

缺少任一步时，candidate 只能返回 `plan_only` 或 `unsupported`。不能因 resident 路径在当前机器能成功，就把
streamed candidate 标成 executable。

### 263.2 Flux 施工建议

Flux 首发可拆成两个独立 candidate：

```text
flux_gpu_resident_v1          只描述现有 resident 路径（若无 hooks 则仍为 plan-only）
flux_gpu_component_staged_v1  text -> transformer -> VAE，组件间 drain
```

site 族：

| 组件 | site | 说明 |
|---|---|---|
| text encoder | tokenizer/embedding/hidden states | dynamic text bucket 绑定 key |
| transformer | persistent norms、block weight、QKV/MLP scratch、latent | block 级 last-use |
| LoRA | base/adapter/delta/fused output | strategy、strength、checkpoint digest 绑定 |
| VAE | latent upload、decoder graph、tile scratch、RGB output | tile/graph cache 独立 |
| export | image output、file staging | 进程/子进程 envelope 独立 |

逐文件建议：

```text
native/models/flux2/*                 加入 metadata-only component manifest
native/runtime/memory_manifest.*      生成 Flux site/instance/alias
native/platform/apple/flux_session.mm 仅在 context 已 bind 后创建 classified backing
native/platform/apple/results.mm      输出 candidate/plan/closure status
tests/native/test_flux_memory_plan.py 只测 deterministic plan，不声称 GPU pass
```

Flux 的 LoRA 不能在 planner 之后偷偷 materialize。三种策略必须分别建 key：

1. inference-time low-rank：base 权重与 adapter 权重同时 live；
2. in-memory delta：融合 scratch 与 fused backing 同时 live；
3. pre-merged checkpoint：checkpoint digest 改变，不能复用前两者 evidence。

如果当前 Flux adapter 没有逐 allocation hooks，允许输出 resident/staged 的估算，但 `execution_supported=false`，
并在结果中明确 `reason=memory_hooks_incomplete`。

### 263.3 Z-Image/GGUF 施工建议

Z-Image/GGUF 还要区分：

- resident GPU GGUF；
- sharded/mmap host backing；
- request-time LoRA；
- Core ML/ANE hybrid；
- VAE helper。

每种组合都要单独绑定：`checkpoint_digest`、GGUF quant layout、shard count、mmap policy、LoRA strategy、helper mode。

GGUF mmap 的文件大小不是 GPU live upper。manifest 必须包含：

```text
mapped file logical bytes
clean host page envelope
materialized GPU tile/row upper
dequant scratch
QKV/MLP activation
graph/cache envelope
```

`madvise`、`posix_fadvise` 或关闭 mmap 只能改变 I/O/page-cache 行为，不能替代 reservation。若无法在 materialize 前
得到 deterministic upper，Z-Image 只能 plan-only。

### 263.4 新模型模板

每个新模型 PR 必须新增 `docs/design/<model>-memory-adapter.md` 或在本文件追加对应小节，至少包含：

```text
candidate key/schema
supported operations/shapes
all required site families
memory formulas and provenance
epoch graph
queue/helper topology
resident/streamed/downgrade candidates
quality invariants
failure/quarantine behavior
L0/L1/L2/L3 test commands
known unsupported branches
```

模板必须先由 reviewer 确认，再写代码；否则容易出现“先接 generic allocator、后补 manifest”的不可审计路径。

## 264. 逐文件 patch 级施工单（当前工作树下一轮）

以下任务单把前述设计压缩到实现者可以直接领取的粒度。每个任务的“完成”都要求代码、测试、contract 和文档四者
同步，不允许只改注释或只加源码断言。

### 264.1 Runtime core

| 任务 | 文件 | 具体修改 | 必测 |
|---|---|---|---|
| R1 epoch key | `native/runtime/memory_plan.hpp/.cpp` | `MemoryEpochKey`、resolver、strict flag、schedule digest | insertion-order/digest/unknown key |
| R2 context strictness | `native/runtime/memory_execution.*` | explicit vs automatic counters；production strict | auto advance rejected |
| R3 key v2 | `native/runtime/memory_manifest.*` | schema、new identity dimensions、lookup equality | equality/canonical/digest field parity |
| R4 closure registry | `native/runtime/memory_site_registry.*` | X-macro loader、symbol digest、generic-site rejection | source/manifest/trace closure |
| R5 swap probe | `native/platform/apple/swap_probe.*` | injectable observer、counter validity、delta | unavailable/rollback/delta |
| R6 report schema | `native/runtime/memory_evidence.*` | bounded files、atomic directory、root digest | path traversal/symlink/partial write |

### 264.2 H3

| 任务 | 文件 | 具体修改 | 必测 |
|---|---|---|---|
| H1 sites | `h3_memory_sites.def`, `h3_memory.h` | site enum + schedule hooks | unknown tag rejected |
| H2 text | `h3_text_encoder.c` | epoch + classified host/GPU allocations | conditioning closure |
| H3 DiT | `h3_dit.c`, `h3_dit_schedule.c` | slot instance/stage/retire | one/two slot peak golden |
| H4 VAE | `h3_video_vae.c`, `h3_taeh3.c` | tile/TAE/output sites | tile alias/zero-state |
| H5 bridge | `h3_session.mm`, `h3_gpu.m` | semantic epoch, queue/stage/slot | stale completion/quarantine |
| H6 evidence | service + results | H3 candidate/closure/swap fields | failed artifact |

### 264.3 LTX

| 任务 | 文件 | 具体修改 | 必测 |
|---|---|---|---|
| L1 sites | `ltx_memory_sites.def` | Gemma/stage/VAE/helper site set | closure branch matrix |
| L2 epoch | `ltx_blocks.c`, `ltx_native.h` | stage1/stage2 semantic events | no auto epoch |
| L3 queue | `ltx_gpu.m`, `ltx_gpu.h` | queue identity in token/trace | video/audio/text isolation |
| L4 session | `ltx_session.mm` | stage handoff/helper barriers | pending=0 after drain |
| L5 evidence | service/results | helper topology and child peak | child crash/timeout |

### 264.4 API/测试

| 任务 | 文件 | 具体修改 | 必测 |
|---|---|---|---|
| A1 C report test | `tests/native/memory_report_c_api_test.mm` | 真调用 take API | one-shot/busy/disabled |
| A2 epoch tests | `tests/native/memory_epoch_test.cpp` | strict compatibility | deterministic resolver |
| A3 closure tool | `tools/memory/verify_required_site_closure.py` | 四集合+扩展错误集 | fixture pass/fail |
| A4 evidence verifier | `tools/memory/verify_evidence.py` | recompute file/root digests | tamper detection |
| A5 Makefile | `Makefile` | 加入 L0/L1 tests，Metal 无设备时明确 skip | CI no silent skip |

### 264.5 推荐提交顺序

```text
R1 -> R2 -> R3 -> R4 -> A2/A3
R5 -> R6 -> A1/A4
H1/H2/H3/H4/H5 -> H6
L1/L2/L3/L4 -> L5
Flux/Z-Image plan-only fixtures
真实 Metal L2 -> 低内存 L3 -> capability review
```

不同模型 adapter 可并行施工，但不能同时修改同一份 registry、digest schema 或 evidence schema 而不先合并合同 patch。

## 265. planner 与 scheduler 的可计算性能模型

### 265.1 基本时间模型

对一个 streamed block，定义：

```text
T_io(b)       = SSD read + decode/dequant + host staging
T_upload(b)   = host->shared/private GPU transfer
T_compute(b)  = Metal encode + GPU execution
T_retire(b)   = command completion wait/queue drain cost
```

双槽理想稳态吞吐近似：

```text
T_step ≈ T_warmup + Σ_b max(T_compute(b), T_io(b+1)+T_upload(b+1))
        + boundary_drain_cost
```

这只是调优排序模型，不是内存上界。planner 必须独立计算：

```text
peak = max_epoch(baseline + framework + all live concrete/alias backings)
```

不要用“读取被 GPU 隐藏”推导“读取 buffer 不占内存”；I/O staging 在重叠窗口内仍是 live resource。

### 265.2 slot 数选择

候选 `k` 的约束：

```text
peak(k) <= B
slots(k) <= verified_refill_slots
completion_bound(k) <= timeout
quality(k) == candidate_quality_contract
```

性能排序可使用：

```text
score(k) = gpu_utilization_weight * overlap_ratio(k)
         - io_wait_weight * normalized_io_wait(k)
         - memory_pressure_penalty(k)
```

但 score 只能在满足硬约束的候选间排序。`memory_pressure_penalty` 不允许把超预算候选“罚分后仍选择”。

### 265.3 lookahead 自适应

建议初始 `lookahead=1`，并按 pressure action 调整：

```text
Normal   -> min(1, candidate_lookahead)
Tight    -> 0
Critical -> 0 + drain at next safe point
Abort    -> no new work
```

若未来认证 lookahead=2，必须在 candidate key 和 plan digest 中绑定，并增加 I/O staging 与两个 future slot 的 live
interval。不能只修改运行时整数。

### 265.4 利用率指标

每个 request 建议记录：

```text
gpu_busy_seconds
gpu_wait_seconds
io_read_seconds
io_decode_seconds
upload_seconds
prefetch_overlap_seconds
boundary_drain_seconds
slot_reuse_count
cache_hit_count
```

派生指标：

```text
gpu_utilization = gpu_busy / wall
io_hidden_ratio = prefetch_overlap / io_read
drain_overhead = boundary_drain / wall
slot_reuse_efficiency = useful_refills / slot_reuse_count
```

这些指标用于比较同一 candidate 的参数，不可跨不同 schedule/framework identity 直接合并。

## 266. 失败恢复和 worker quarantine 的可执行 runbook

### 266.1 失败分类

所有失败必须映射到有限枚举：

```text
request_invalid
memory_observation_unreliable
memory_capability_missing
memory_manifest_invalid
memory_budget_too_small
memory_policy_allocation_failed
memory_lifetime_violation
memory_pressure_abort
memory_swap_activity_detected
backend_gpu_failure
helper_failure
cancelled
evidence_write_failure
```

字符串可包含 site/phase 诊断，但分类字段必须稳定，供 service/UI/统计使用。不要直接把 Objective-C/Metal 原始错误
作为跨版本分类。

### 266.2 clean vs quarantine 决策表

| 条件 | cleanup 可证明 | 结论 |
|---|---|---|
| reserve denied，未提交 GPU | 是 | `Failed/Clean`，worker 可复用 |
| actual > upper | 否 | `Quarantined` |
| completion status 非零，但 queue drain 成功且 storage zero | 是 | `Failed/Clean` |
| stale generation/domain | 否 | `Quarantined` |
| mailbox overflow | 否 | `Quarantined` |
| helper timeout/crash，child 已确认退出 | 视 handoff zero-state | clean 或 quarantine |
| swap delta > 0 | 视 GPU cleanup | 不能成功；clean 仅表示可复用 |
| report/evidence 写失败 | 不影响 runtime cleanup | 保留 primary 状态，另记 evidence failure |

`Failed/Clean` 不是成功；它只表示 worker 的 allocator/queue 状态经过验证，可以接收下一次 request。`Quarantined`
必须阻止 engine 再次 generate，并要求 service 销毁/重建 worker。

### 266.3 取消顺序

取消路径必须满足：

1. 置 `cancelled`，禁止新的 prefetch/reserve/submit；
2. 当前线程完成 callback-safe point；
3. retire 所有逻辑上不再使用的 lease；
4. 逐 queue drain，不使用未验证的直接 free；
5. drain mailbox，检查 pending/completion/storage；
6. 释放 host/cache/helper；
7. 生成 terminal report/evidence；
8. 根据 zero-state 决定 clean/quarantine。

如果 cancel callback 自身抛异常，必须转换为 `memory_lifetime_violation`，并保留原始 `cancelled` 作为 cause chain 的
前一个节点。

### 266.4 worker 重建

service 发现 `memory_quarantined=true` 时：

```text
stop accepting new requests for worker
wait current caller return
collect evidence/report
destroy engine/session/helper
release process-wide locks
spawn/create fresh worker
re-run metadata-only preflight
```

禁止在同一个 quarantined engine 上调用 `unload()` 后“试试看能否继续”。如果平台资源不能证明已回收，继续复用会
把 stale token 误配到新 request。

## 267. API、配置和结果 schema 的具体建议

### 267.1 请求示例

schema v1（当前兼容）：

```json
{
  "schema_version": 1,
  "model": "h3",
  "prompt": "...",
  "output": "/tmp/out.mp4",
  "memory_constrained": {
    "enabled": true,
    "limit_bytes": 17179869184,
    "buffer_percent": 15,
    "min_free_bytes": 1073741824,
    "max_refill_slots": 2,
    "allow_quality_preserving_tiling": true
  }
}
```

规范化后结果必须同时显示：

```json
{
  "memory_policy": {
    "enabled": true,
    "user_limit_bytes": 17179869184,
    "buffer_percent": 15,
    "effective_budget_bytes": 14602888806,
    "system_reserve_bytes": 1073741824,
    "planned_upper_bytes": 14000000000,
    "framework_upper_bytes": 1073741824,
    "denoiser_budget_bytes": 9000000000,
    "refill_slots": 2,
    "effective_residency": "streamed",
    "execution_supported": false,
    "admission_state": "plan_only",
    "reason": "capability manifest missing"
  }
}
```

示例中的数值仅用于说明 schema，不能当作任何机器的认证上限。特别是 `effective_budget_bytes` 的计算必须使用
checked integer arithmetic，并明确是否是 decimal percent floor。

### 267.2 配置 precedence

优先级固定为：

```text
request.memory_constrained fields
    > named profile overlay
    > application default
```

但只能覆盖“用户可配置字段”：enabled、Y、X、S、max_refill_slots、allow_tiling。以下字段永远由 probe/manifest/plan
决定，用户和环境变量不能覆盖：framework upper、schedule revision、candidate backend/dtype、site upper、helper
topology、capability level。

profile overlay 必须保留 `specified_fields`，以便区分“用户明确设 0”与“未指定使用默认”。merge 后再做一次统一
validation，不能在 profile 和 request 两处各自实现边界检查。

### 267.3 结果字段兼容

在现有 result JSON 上只追加字段，不删除/重命名旧字段：

```text
memory_policy
memory_admission
memory_trace
memory_evidence_id
memory_failure_class
memory_failure_disposition
```

旧客户端忽略未知字段即可。C API 的旧 `error` 保持普通字符串；新的 report/evidence 查询接口使用独立函数和明确
返回码。所有新接口要有 `tc_abi_version` 或 struct_size/version 检查。

### 267.4 错误 JSON（service 内部）

建议 service 内部统一：

```json
{
  "schema_version": 1,
  "error_class": "memory_budget_too_small",
  "message": "planned peak exceeds effective budget",
  "request_id": "...",
  "candidate_key_digest": "sha256:...",
  "plan_digest": "sha256:...",
  "execution_state": "failed",
  "failure_disposition": "clean",
  "evidence_id": "..."
}
```

`message` 可以供人阅读；机器逻辑只依赖 `error_class`、state、disposition 和 digest 字段。不要把完整 prompt、路径或
Metal pointer 放入错误。

## 268. 本机验收和 CI 设计：从无 Metal 到真实低内存

### 268.1 L0（任何 macOS/CI）

必须执行且不能因无 Metal 静默跳过：

```sh
python3 tests/native/test_memory_accounting.py
python3 tests/native/test_memory_manifest.py
python3 tests/native/test_memory_plan_compiler.py
python3 tests/native/test_memory_scheduler.py
python3 tests/native/test_memory_watchdog.py
python3 tests/native/test_memory_trace.py
python3 tests/native/test_memory_execution.py
python3 tests/native/test_contract.py
git diff --check
```

这些测试使用 fake observer/allocator，不创建 GPU buffer，必须在无 Metal device 上通过。

### 268.2 L1（有编译器但无 Metal device）

```sh
python3 tests/native/test_h3_schedule_memory.py
python3 tests/native/test_h3_gpu_memory_hooks.py
python3 tests/native/test_ltx_gpu_memory_hooks.py
```

允许 runtime `SKIP: no Metal device is available`，但 wrapper 必须先成功编译 Objective-C/Metal 源码；skip 必须打印
固定 reason 并被 CI 统计，不能返回 PASS 或吞掉 stderr。

### 268.3 L2（真实 Metal）

每个 candidate 至少：

```text
5 warmup + 20 measured（短 shape）
10 allocation/submit/completion fault injections
cancel at text/dit/vae/export safe points
1-slot 和 2-slot（若声明）
resident baseline 对照
```

每个 request 保存完整 evidence。L2 通过条件：

- source/manifest/trace closure closed；
- no generic/unknown site；
- completion order 正确；
- process/framework upper 没有 sticky overage；
- successful request zero-state；
- 输出质量达到 candidate contract；
- disabled baseline 没有额外行为。

### 268.4 L3（低内存 campaign）

优先在真实 16/24/32 GiB Apple Silicon 机器上，或使用经过验证的等价预算做 campaign。每个预算至少覆盖：

```text
minimum-1   计划刚好低于 B 的可成功点
minimum     最小已认证成功点
baseline    默认 resident/streamed 对照
```

每个点重复至少 3 次；不以均值掩盖单次超预算。硬门禁：

```text
max process_peak <= B
max unattributed_peak <= framework_upper
min system_available >= S（若低于 S，必须按规则失败）
every successful request swapouts_delta == 0
every failed request has terminal report
```

如果系统已有 swap 历史，记录 campaign 前后 counter；并发干扰或 counter 无效时标记 environment invalid，重新运行。

### 268.5 CI gate 伪代码

```python
def assert_evidence(e):
    assert e["plan"]["complete"]
    assert e["closure"]["status"] == "closed"
    assert e["memory_metrics"]["peak_process_footprint_bytes"] <= e["policy"]["effective_budget_bytes"]
    assert e["memory_metrics"]["peak_unattributed_process_footprint_bytes"] <= e["policy"]["framework_upper_bytes"]
    assert e["swap"]["swapouts_delta"] == 0
    if e["result"]["execution_state"] == "succeeded":
        assert e["cleanup"]["zero_state"]
```

CI 只接受机器可重算的 JSON；人工截图、日志片段或“运行很顺利”不能替代 evidence。

## 269. 发布和回滚的最终门禁

### 269.1 release checklist

加入 production registry 前，release owner 必须签核：

```text
[ ] candidate key v2、manifest、plan、closure、framework digest 一致
[ ] schedule_revision 与源码/tag 一致
[ ] H3/LTX 所有 required site closure=closed
[ ] LTX v2 retire/complete 的真实 Metal completion evidence 完整
[ ] H3/LTX allocation/submit/completion/cancel fault injection 通过
[ ] helper/process topology 与 evidence 一致
[ ] L2 quality parity 与 performance baseline 通过
[ ] L3 每请求 swapouts_delta=0
[ ] disabled ABBA 无额外 probe/watchdog/clear_cache/unload
[ ] evidence digest 可独立重算
[ ] production registry 只加入真实 review 批准的 record
```

### 269.2 回滚触发器

下列任一事件应立即从 registry 移除/禁用对应 record，并回到 plan-only：

- 任意 request actual > manifest upper；
- stale completion、mailbox overflow、generic site；
- framework overage 或 swapout delta；
- OS/Metal/framework/allocator revision 不匹配；
- 输出质量回归超阈值；
- evidence 缺失或 digest 无法重算；
- disabled ABBA 出现统计显著回归。

回滚只改变 capability lookup 的 release_enabled/state，不删除旧 evidence。旧 evidence 用于定位差异；重新认证必须生成新
manifest/plan/schedule/campaign digest。

### 269.3 用户体验

低内存用户应看到明确的候选和原因：

```text
可执行：streamed 1-slot，预计峰值 13.2 GiB，预计额外 I/O 1.8 s
可降级：streamed 2-slot -> 1-slot，原因 memory pressure
不可执行：需要 18.4 GiB，当前有效预算 14.6 GiB，原因 memory_budget_too_small
```

不要显示“系统会自动 swap，所以可以继续”；swap 是失败/诊断信号。默认模式用户不应看到 constrained 的额外提示、
probe 或 trace。

## 270. 本轮增补结论

本次续修把方案从“已有 runtime 脚手架”进一步收敛为可施工闭环：

1. 代码事实与文档事实已对齐：LTX hook v2、terminal report/C API 标记为已落地，但真实 Metal/L3 仍未认证；
2. semantic epoch、required-site closure、capability identity v2 成为下一轮的三条基础主线；
3. swap telemetry 和 evidence writer 把“不进入 swap”和“失败也可审查”变成机器可验证合同；
4. H3/LTX 有明确的首发 candidate、逐文件 patch、stage/queue/barrier 和故障验收；
5. Flux/Z-Image 明确保持 plan-only，按同一认证阶梯逐步推进；
6. scheduler 的性能优化受硬预算、completion、zero-state、quality 和 disabled ABBA 共同约束；
7. 在真实 Metal 与低内存 campaign 完成前，production capability registry 必须为空，不能用 synthetic record 代替。

实现者完成下一轮后，应重新运行文档顶部的当前状态审计，并把新的源码事实、测试命令、skip 原因和 evidence 路径继续
写回本文，保证这份设计稿始终是可执行的施工与验收合同，而不是脱离代码的愿望清单。

## 271. 2026-09-16 继续实施记录：swap telemetry 与 certified explicit epoch

本轮在不增加任何 production capability record、也不改变 disabled/default 路径的前提下，继续落地了两项 runtime
硬门禁。

### 271.1 request 级 swap telemetry 已实现

代码修改：

```text
native/runtime/memory_accounting.hpp/.cpp
  - SwapActivityObservation / SwapActivityObserver
  - host_statistics64(HOST_VM_INFO64) counter probe
  - MemoryAdmissionMetrics 的 swap begin/end/delta/status 字段

native/runtime/memory_execution.hpp/.cpp
  - constrained baseline、checkpoint、finish_success、failure cleanup 采样
  - counter unavailable/regression/source change fail-closed
  - swapouts_delta > 0 -> memory_swap_activity_detected
  - terminal report 只读取冻结指标，不重新 probe

native/api/c_api.mm
  - clean process baseline 后建立 swap baseline
  - constrained execution 要求 swap counter 可用
  - enforcement_scope 增加 swapout_delta

native/platform/apple/results.mm
  - 序列化 swap counter、delta、sample count、source 和首次异常 phase
```

语义边界：

- counter 是全系统单调值，正增量使 request/campaign 失败，但不声称一定由本进程造成；
- swap counter 从不增加预算，也不选择更大的 residency candidate；
- unavailable、counter 回退或 source 改变视为不可靠观测，不能产生成功 evidence；
- swapins 被记录，当前硬失败条件是 `swapouts_delta > 0`；
- disabled request 不创建 `MemoryExecutionContext`，因此没有新增 swap probe。

### 271.2 certified explicit epoch gate 已实现

代码修改：

```text
native/runtime/memory_manifest.hpp/.cpp
  - MemoryCapabilityRecord::require_explicit_epoch
  - executable record 注册时必须为 true

native/runtime/memory_plan.hpp/.cpp
  - CompiledMemoryPlan::require_explicit_epoch
  - explicit-epoch requirement 进入 plan digest

native/runtime/memory_execution.hpp/.cpp
  - strict plan 禁止 allocation 自动进入未来 epoch
  - compatibility plan 保留自动推进并单独计数
  - metrics 输出 explicit/automatic transition、current/peak epoch

native/platform/apple/results.mm
  - 输出 epoch transition metrics
```

这一步故意没有伪造 H3/LTX semantic stage hook。当前 H3/LTX 尚未把真实 stage/block/tile event 映射到 plan epoch，
所以即使未来错误地加入 capability record，也会在第一次需要未来 epoch 的 allocation 上 fail-closed。只有完成第 256、
261、262 节的 adapter schedule bridge 并证明 `automatic_epoch_transition_count == 0` 后，才允许认证。

非认证 compatibility plan 仍能自动推进，主要用于旧 manifest migration 和纯逻辑测试；该模式不能来自 request JSON、
profile 或普通环境变量。

### 271.3 新增/扩展测试

```text
tests/native/memory_accounting_test.cpp
  - 本机 Mach swap counter 可读、source 稳定、counter 单调

tests/native/memory_execution_test.cpp
  - zero swapout success
  - swapout delta fail-closed
  - counter regression fail-closed
  - terminal report 不重新 probe
  - certified plan 必须显式 enter_epoch
  - compatibility plan 自动推进并计数

tests/native/memory_plan_compiler_test.cpp
  - explicit epoch requirement 改变 plan digest

tests/native/memory_manifest_test.cpp
  - executable capability record 拒绝 implicit epoch

tests/native/test_contract.py
  - swap/epoch API 与 JSON 字段源码合同
```

### 271.4 本轮验证结果

```text
native-only full build                           PASS
memory accounting                               PASS
memory manifest                                 PASS
memory plan compiler                            PASS
memory scheduler/watchdog/trace                 PASS
memory execution                                PASS
memory probe                                    PASS
contract                                        81 PASS / 1 fixture SKIP
H3 GPU memory hooks                             compile PASS / runtime SKIP(no Metal)
LTX GPU memory hooks                            compile PASS / runtime SKIP(no Metal)
```

仍未完成且继续阻塞 release 的事项：

- H3/LTX semantic epoch table 与 C schedule hook 的真实接线；
- H3/LTX required-site source/manifest/trace closure；
- capability identity v2 的 schedule/framework/helper/platform 字段；
- service evidence artifact writer；
- 真实 Metal completion/fault L2；
- 16/24/32 GiB 或等价低内存机器的逐请求 `swapouts_delta == 0` L3 campaign；
- Flux/Z-Image allocation hooks 和 staged executable adapter。

production capability registry 因此继续保持为空，本轮结果不能解释为任一模型已经 release-ready。

## 272. 2026-09-16 实施冻结增补：把下一轮改造拆成可独立评审的合同

第 256–264 节已经定义了 semantic epoch、required-site registry、capability identity v2 和模型接入方向。本节开始
进一步固定实现顺序和接口所有权，目标是避免下一轮同时修改 manifest schema、C ABI、模型循环、evidence schema
和 production registry，导致问题无法归因。

### 272.1 下一轮只解决四个 release blocker

实现顺序固定为：

1. 先增加共享 schedule event ABI 和 C++ resolver，但不改变任何 disabled/default 调用路径；
2. 再让 H3、LTX 在真实 owner-thread 语义边界发事件，证明 certified plan 的
   <code>automatic_epoch_transition_count == 0</code>；
3. 再把 allocator 的宽泛字符串收敛为 required-site registry，并补 explicit instance identity；
4. 最后升级 capability key 和 service evidence writer，仍保持 production registry 为空。

以下工作不能混入上述提交：

- 不在同一 patch 开启 ANE、LoRA、Sol、sparse、audio 或动态 gate-skip 新分支；
- 不在 schedule hook patch 中改变 kernel、数值精度、block 顺序或 sampler；
- 不在 site registry patch 中顺便改变 allocation 大小或缓存策略；
- 不在 evidence writer patch 中改变成功/失败判定；
- 不在无真实 Metal/L3 证据时加入 executable production record。

### 272.2 两阶段 schema 迁移，避免一次性重写 manifest

推荐先做 bridge schema，再做最终 schema：

| 阶段 | manifest | schedule 表 | executable 条件 |
|---|---|---|---|
| A：bridge | 保持现有 v1 数值 <code>live_begin/live_end</code> | plan 中新增 semantic event 到数值 epoch 的映射 | experimental fixture；必须 explicit event |
| B：native semantic | 升级 v2，instance 直接保存 acquire/last-use key | compiler 从 key 生成 epoch | 新认证 record 只允许此阶段 |

阶段 A 的价值是把 adapter 接线、线程约束、事件遗漏和错误传播先独立验证。阶段 A 不能生成 production certified
record，也不能把旧 v1 manifest 自动提升为 v2。阶段 B 必须重新计算 manifest、schedule、plan、closure 和 campaign
digest。

### 272.3 对象所有权

| 对象 | 创建者 | 生命周期 | 可写线程 | 禁止行为 |
|---|---|---|---|---|
| <code>MemoryExecutionContext</code> | API request owner | admission 到 terminal report | request owner | 被 Metal callback 直接销毁 |
| compiled schedule table | plan compiler | 整个 request，不可变 | 构造期后只读 | adapter 修改 epoch id |
| C schedule hooks | H3/LTX session bridge | bind 到 unbind | owner 调用 | 背景 I/O 线程推进 epoch |
| allocation token | reserve callback | reserve 到 cancel/release/complete | 明确的状态 owner | 同时走 release 和 complete |
| completion token | retire 创建 | retire 到 mailbox drain | callback 只发布，owner 消费 | callback 直接改 ledger |
| evidence snapshot | terminal owner | take_report 后到 writer 返回 | owner/writer | 重新 probe allocator 或模型 |

session 的 <code>bind_memory_context()</code> 只发布 non-owning pointer；<code>unbind_memory_context()</code> 必须在所有模型
线程 join、GPU completion drain 和 hook detach 之后执行。任何 callback 在 unbind 后到达都属于 stale generation，
worker 必须 quarantine。

## 273. 共享 schedule event C ABI 的最终建议

### 273.1 文件与命名

新增：

~~~text
native/core/memory_schedule_c.h
native/runtime/memory_schedule.hpp
native/runtime/memory_schedule.cpp
tests/native/memory_schedule_test.cpp
tests/native/test_memory_schedule.py
~~~

<code>memory_schedule_c.h</code> 只能依赖 <code>stddef.h</code> 和 <code>stdint.h</code>，可以被 C、C++、Objective-C 和
Objective-C++ 同时包含。它不能暴露 <code>std::string</code>、C++ enum layout、Objective-C object 或 Metal pointer。

### 273.2 ABI 建议

~~~c
#ifndef TC_MEMORY_SCHEDULE_C_H
#define TC_MEMORY_SCHEDULE_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TC_MEMORY_INDEX_NONE UINT32_MAX

typedef enum {
    TC_MEMORY_STAGE_ADMISSION = 1,
    TC_MEMORY_STAGE_TEXT = 2,
    TC_MEMORY_STAGE_CONDITIONING_HANDOFF = 3,
    TC_MEMORY_STAGE_DENOISER_LOAD = 4,
    TC_MEMORY_STAGE_DENOISER = 5,
    TC_MEMORY_STAGE_LATENT_HANDOFF = 6,
    TC_MEMORY_STAGE_VIDEO_VAE = 7,
    TC_MEMORY_STAGE_AUDIO_VAE = 8,
    TC_MEMORY_STAGE_EXPORT = 9,
    TC_MEMORY_STAGE_TERMINAL_DRAIN = 10
} tc_memory_schedule_stage_v1;

typedef enum {
    TC_MEMORY_ACTION_BEGIN = 1,
    TC_MEMORY_ACTION_PREFETCH = 2,
    TC_MEMORY_ACTION_UPLOAD = 3,
    TC_MEMORY_ACTION_COMPUTE = 4,
    TC_MEMORY_ACTION_LAST_USE = 5,
    TC_MEMORY_ACTION_RETIRE = 6,
    TC_MEMORY_ACTION_DRAIN = 7,
    TC_MEMORY_ACTION_END = 8
} tc_memory_schedule_action_v1;

typedef enum {
    TC_MEMORY_BRANCH_COMMON = 1,
    TC_MEMORY_BRANCH_TEXT = 2,
    TC_MEMORY_BRANCH_VIDEO = 3,
    TC_MEMORY_BRANCH_AUDIO = 4,
    TC_MEMORY_BRANCH_VAE = 5,
    TC_MEMORY_BRANCH_EXPORT = 6
} tc_memory_schedule_branch_v1;

enum {
    TC_MEMORY_EVENT_SAFE_POINT = 1u << 0,
    TC_MEMORY_EVENT_GPU_DRAINED = 1u << 1,
    TC_MEMORY_EVENT_PREFETCH_ALLOWED = 1u << 2,
    TC_MEMORY_EVENT_TERMINAL = 1u << 3
};

typedef struct {
    uint32_t struct_size;
    uint32_t version;
    uint32_t stage;
    uint32_t action;
    uint32_t step;
    uint32_t block;
    uint32_t tile;
    uint32_t branch;
    uint32_t slot;
    uint32_t flags;
} tc_memory_schedule_event_v1;

typedef struct {
    uint32_t struct_size;
    uint32_t version;
    void *user;
    int (*emit)(void *user,
                const tc_memory_schedule_event_v1 *event,
                char *error,
                size_t error_size);
} tc_memory_schedule_hooks_v1;

#ifdef __cplusplus
}
#endif

#endif
~~~

约束：

- 所有未使用坐标必须写 <code>TC_MEMORY_INDEX_NONE</code>，不能用 0 同时表示“第 0 个”和“不适用”；
- adapter 不传 sequence；plan 内 sequence 由 C++ cursor 自己维护，下一 semantic key 不匹配即可发现漏发、重复发和乱序；
- adapter 只发 semantic event，不传 epoch id；epoch id 始终由 compiled plan resolver 决定；
- <code>slot</code> 用于 trace 和实例校验，默认不参与 semantic epoch identity；
- flags 必须与 plan 中该 event 的 flags 完全一致，adapter 不能自行宣称已经 drain；
- ABI v1 只保留一个同步 <code>emit</code>；checkpoint 由 SAFE_POINT event 驱动，避免两个回调顺序不一致。

### 273.3 线程与重入合同

<code>emit</code> 只能由 request owner thread 调用。具体规则：

1. H3/LTX owner 在启动异步 prefetch 线程之前先发 PREFETCH event；
2. I/O 线程只填充已经 reserve 的 slot，并写入自己的 completion/result 字段；
3. owner join I/O 线程后发 UPLOAD 或 COMPUTE event；
4. Metal completion handler 只调用现有 <code>complete()</code> 发布 mailbox，不发 schedule event；
5. owner drain mailbox 成功后才发 DRAIN 或下一 block 的 PREFETCH event；
6. <code>emit</code> 不得递归进入模型、等待 Metal、写文件或取得 service lock。

如果未来必须让多个 owner 执行 video/audio 并行，不能直接把 context 改成多线程可写。应由各执行线程把
<code>ScheduleMessage</code> 写入有界 mailbox，再由 request owner 按 plan barrier 合并和推进 cursor。

### 273.4 错误传播

callback 返回 0 后：

- adapter 必须停止创建新 allocation、prefetch 和 command buffer；
- 已 reserve 未 allocate 的 token 走 cancel；
- 已 allocate 未 submit 的 token可同步 release；
- 已 submit 的 token走 retire/completion/drain；
- 顶层把 callback error 保留为 primary memory error；
- error buffer 截断时仍要返回固定错误类前缀，例如
  <code>memory_lifetime_violation:</code>，不能只留下半段自由文本。

## 274. C++ schedule resolver、cursor 和 plan 编译细节

### 274.1 建议数据结构

在 <code>native/runtime/memory_schedule.hpp</code> 增加：

~~~cpp
struct MemoryScheduleEventKey {
    uint32_t stage = 0;
    uint32_t action = 0;
    uint32_t step = TC_MEMORY_INDEX_NONE;
    uint32_t block = TC_MEMORY_INDEX_NONE;
    uint32_t tile = TC_MEMORY_INDEX_NONE;
    uint32_t branch = TC_MEMORY_BRANCH_COMMON;

    bool operator==(const MemoryScheduleEventKey &) const = default;
    bool operator<(const MemoryScheduleEventKey &) const;
    std::string canonical() const;
};

struct CompiledScheduleBinding {
    uint64_t sequence = 0;
    MemoryScheduleEventKey key;
    uint32_t expected_slot = TC_MEMORY_INDEX_NONE;
    uint32_t required_flags = 0;
    uint64_t epoch = 0;
    bool checkpoint_after = false;
};

struct MemoryScheduleCursorMetrics {
    uint64_t accepted = 0;
    uint64_t duplicate = 0;
    uint64_t out_of_order = 0;
    uint64_t unknown = 0;
    uint64_t flag_mismatch = 0;
    uint64_t last_sequence = 0;
};
~~~

在 <code>CompiledMemoryPlan</code> 增加：

~~~cpp
std::string schedule_schema;
std::string schedule_revision;
std::vector<CompiledScheduleBinding> schedule_bindings;
bool require_explicit_schedule = false;
~~~

不要在每次 callback 中构造 canonical string 或查 <code>std::map</code>。plan compile 后按 sequence 保存紧凑 vector；
runtime 正常路径只比较下一个元素。key 的 map/binary search 仅用于诊断未知事件。

### 274.2 compiler 输入

每个 adapter 提供一个 metadata-only schedule template：

~~~cpp
struct MemoryScheduleTemplateInput {
    std::string adapter;
    std::string template_revision;
    uint32_t steps = 0;
    uint32_t blocks = 0;
    uint32_t video_tiles = 0;
    uint32_t audio_chunks = 0;
    uint32_t refill_slots = 0;
    bool audio = false;
    bool dynamic_block_skip = false;
};
~~~

首发 H3/LTX candidate 必须设置 <code>dynamic_block_skip=false</code>。如果实际 runtime 启用了 first-block cache、gate
skip、token merge、Sol/sparse 或其他改变 block sequence 的模式，candidate key 必须不同；没有独立 template 和
evidence 时 preflight 直接拒绝。

### 274.3 阶段 A 的 event 到数值 epoch 映射

在 manifest v1 过渡期，每个 schedule binding 明确保存 epoch id。编译算法：

1. 从 adapter template 生成有序 event list；
2. 收集 manifest 中所有 instance 的 <code>live_begin/live_end</code>；
3. 要求每个 allocation acquire 前至少有一个 event 映射到对应 begin epoch；
4. 要求 asynchronous instance 的 last-use event 早于或等于 RETIRE event，release epoch 不早于 DRAIN；
5. 要求所有 non-empty epoch 至少被一个 schedule event 可达；
6. 计算 schedule digest，再把 digest 纳入 plan digest；
7. certified bridge fixture 设置 <code>require_explicit_schedule=true</code>。

阶段 A 仍然是 migration artifact。真实 production record 只能使用 manifest v2 semantic lifetime。

### 274.4 cursor 状态机

<code>MemoryExecutionContext::emit_schedule_event()</code> 的顺序固定为：

~~~text
validate execution state == Running
validate cursor state == Healthy and not reentrant
validate event struct_size/version
validate cursor.next.sequence == cursor.accepted_count
validate semantic key/slot/flags
validate epoch >= current_epoch
if SAFE_POINT:
    drain completion mailbox
    if GPU_DRAINED: validate pending/outstanding/mailbox state is empty
admission.validate_site_epoch(target_epoch)
enter_epoch(target_epoch, explicit)
if checkpoint_after:
    sample process footprint
    sample swap activity
    watchdog evaluate
advance cursor exactly once
record trace
~~~

SAFE_POINT 的 completion drain 必须发生在 epoch 推进之前。否则 runtime 可能先让旧 epoch 的 async backing 在
ledger 中变成“已过 lifetime”，然后才发现 GPU 实际仍在使用它。checkpoint 失败时虽然 cursor 尚未 advance，epoch
可能已经推进，因此不能把 callback 当成可重试事务；context 必须进入 sticky poisoned 状态，后续只允许 cleanup、
terminal report 和 quarantine 判断。相同 sequence 的重复事件默认 fail-closed；首版不支持 adapter retry。若某些模型
函数确实需要可重入 retry，应把 retry 作为新 event/action 和新 schedule revision，而不是让 cursor 猜测。

### 274.5 terminal 检查

<code>finish_success()</code> 前新增：

~~~text
schedule_cursor_consumed == schedule_bindings.size()
last binding has TERMINAL
current_epoch == terminal epoch
automatic_epoch_transition_count == 0
completion mailbox empty
pending release bytes == 0
all required instance observations satisfied
~~~

任一不满足都不能返回成功。未消费事件表示模型提前退出或 hook 漏接；多余事件表示 runtime 分支逃逸。

## 275. explicit allocation instance 与 required-site registry 的代码级方案

### 275.1 为什么只有 site string 还不够

当前 <code>try_reserve_site(site_id, class, upper)</code> 可以在同一个 site 的多个合法 instance 中根据当前 epoch选择；
这能够做上界保护，但不能证明 adapter 知道自己正在分配哪个 block、tile 或 slot。required-site closure 最终需要
同时证明：

~~~text
site identity
instance identity
semantic epoch
actual backing identity
completion owner
~~~

因此建议引入 explicit instance gate，方式与 explicit epoch gate 对称。

### 275.2 runtime 类型

~~~cpp
struct MemoryAllocationSiteRef {
    uint32_t site_code = 0;
    uint32_t instance_id = 0;
};

struct CompiledAllocationInstance {
    uint32_t site_code = 0;
    std::string site_id;
    uint32_t instance_id = 0;
    // existing fields...
};

std::optional<MemoryReservation> try_reserve_instance(
    MemoryAllocationSiteRef ref,
    MemoryClass memory_class,
    uint64_t upper_bytes);
~~~

<code>CompiledMemoryPlan</code> 和 <code>MemoryCapabilityRecord</code> 增加
<code>require_explicit_instance</code>。executable production record 必须同时要求 explicit epoch 和 explicit
instance。旧 <code>try_reserve_site()</code> 只保留 compatibility fixture；在 strict plan 下调用它直接失败并增加
<code>implicit_instance_attempt_count</code>。

### 275.3 registry 文件格式

每行显式包含稳定数值 code，防止插入新行导致所有 enum 值漂移：

~~~c
TC_MEMORY_SITE(
    0x01010001u,
    H3_TEXT_TOKEN_IDS,
    "h3.text.token_ids",
    CONDITIONING,
    STAGE,
    SYNC,
    REQUIRED,
    TEXT,
    "h3.text.token_ids.v1")
~~~

建议 code 空间：

| 范围 | 所有者 |
|---|---|
| <code>0x01000000–0x01ffffff</code> | H3 |
| <code>0x02000000–0x02ffffff</code> | LTX |
| <code>0x03000000–0x03ffffff</code> | Flux |
| <code>0x04000000–0x04ffffff</code> | Z-Image |
| <code>0x7f000000–0x7fffffff</code> | test fixture，production 禁止 |

删除 site 后 code 永久保留为 tombstone，不能复用。rename 必须创建新 code 和新 source inventory digest。

### 275.4 allocation hook ABI 演进

当前 H3/LTX hook 使用 string tag。为了兼容，不修改既有 callback 的含义，而是在 struct 尾部追加 v3 callback：

~~~c
int (*reserve_site)(void *user,
                    uint32_t site_code,
                    uint32_t instance_id,
                    uint32_t memory_class,
                    uint64_t upper_bytes,
                    void **token,
                    char *error,
                    size_t error_size);
~~~

规则：

- default path 继续允许旧 classified/string API；
- constrained compatibility plan 可以使用旧 reserve，但只得到 plan-only evidence；
- strict executable plan 要求 hook version >= 3 且 <code>reserve_site</code> 非空；
- v3 token 保存 site code、instance、epoch、allocator domain 和 generation；
- commit/retire/complete 必须重复校验 token identity，不能只在 reserve 时校验；
- tag string只用于人类 trace，可由 registry生成，不再作为 authority。

### 275.5 instance 编码

instance id 必须由 metadata-only compiler 和 adapter 共用同一公式。建议：

~~~text
request/stage singleton              0
block-scoped                         block
step-block-scoped                    checked(step * block_count + block)
tile-scoped                          tile
slot-scoped                          slot
stage-block-scoped                   checked(stage * block_count + block)
stage-step-block-scoped              checked((stage * steps + step) * blocks + block)
~~~

公式 id 进入 site registry 的 <code>UPPER_FORMULA_ID</code> 或独立
<code>INSTANCE_FORMULA_ID</code>。所有乘加使用 checked arithmetic；超过 <code>uint32_t</code> 时 manifest compile
失败，不能截断。

### 275.6 closure 生成器

新增工具建议分成两步：

~~~text
tools/memory/generate_site_inventory.py
  输入 .def
  输出 canonical source-inventory.json + C/C++ generated headers

tools/memory/verify_required_site_closure.py
  输入 inventory/manifest/plan/trace/candidate
  输出 closure.json
~~~

生成文件必须带：

~~~json
{
  "schema": "turbocider.memory_source_inventory.v1",
  "generator_revision": "...",
  "model": "h3",
  "source_revision": "...",
  "sites": [],
  "digest": "..."
}
~~~

CI 先重新生成到临时目录再 byte-compare；仓库中的 generated header 与 .def 不一致时失败。禁止工具直接扫描
allocator 函数名来猜 required site，因为宏、wrapper、Objective-C helper 和 MLX 路径会产生漏报。

## 276. refill slot 的完整事务状态机与 overlap 合同

### 276.1 slot 状态

每个 refill slot 使用显式状态：

~~~text
Empty
  -> Reserved
  -> Filling
  -> Ready
  -> Submitted
  -> Retired
  -> CompletionPosted
  -> Reusable
  -> Reserved ...

任何状态
  -> Poisoned
~~~

| 转移 | 执行者 | 必须满足 |
|---|---|---|
| Empty -> Reserved | owner | reserve-before-allocate 成功 |
| Reserved -> Filling | owner 启动 I/O | backing 已 commit，尚未 GPU submit |
| Filling -> Ready | owner join loader | read/dequant 成功、实际 bytes 不超 upper |
| Ready -> Submitted | owner | event=COMPUTE，command encode/commit 成功 |
| Submitted -> Retired | owner | last-use 已编码，pending token 已注册 |
| Retired -> CompletionPosted | Metal callback | generation/domain/queue/status 校验后只入 mailbox |
| CompletionPosted -> Reusable | owner | mailbox drain 成功，ledger pending 已归零 |

<code>slot_id</code> 只在 Reusable 后可以再次 reserve。将指针覆盖成下一 block 不等于 slot reusable。

### 276.2 背景 I/O 线程边界

背景线程只能访问：

- 已经 commit 的 slot backing；
- immutable checkpoint mapping/index；
- 线程私有 error/result；
- cancellation 的只读原子标记。

它不能：

- 调用 <code>MemoryExecutionContext</code>；
- reserve 新内存；
- 改 schedule cursor；
- release/retire ledger lease；
- 创建无上界的临时 vector；
- 在失败后自行启动重试。

需要 dequant/conversion scratch 时，owner 必须先为 scratch reserve/commit，再把指针交给 worker。scratch 的 last-use
若早于 GPU completion，可以在 upload 已复制完成的明确 event 释放；否则与 slot 一起 retire。

### 276.3 一次 streamed block 的参考伪代码

~~~cpp
emit(PREFETCH, step, next_block, next_slot);
auto slot = reserve_refill_instance(next_slot_instance, block_upper);
allocate_slot_after_reserve(slot);
start_loader(slot, next_block);

emit(COMPUTE, step, block, current_slot);
encode_block(current_slot);
commit_command_buffer();
retire_after_last_use(current_slot);

join_loader(next_slot);
validate_loader_result_and_actual_bytes(next_slot);
emit(UPLOAD, step, next_block, next_slot);

drain_completion_mailbox_at_safe_point();
mark_completed_slots_reusable();
~~~

真实实现可以先 join loader 再 submit current block，或让两者 overlap；但事件顺序、live interval 和计划峰值必须覆盖
实际最坏交叠。若 Metal shared buffer 直接被 <code>pread</code> 填充，不能另外把文件 cache 当作“免费”；campaign 的
framework/process envelope 必须覆盖可能的 clean page 增量。

### 276.4 统一内存上的 offload 定义

在 Apple Silicon 上，以下操作不应被称为已降低 live set：

- 把 private/shared Metal buffer 复制到新的 malloc buffer；
- 把 MLX array 转成另一个 host tensor但保持原 tensor/cache；
- 只调用 <code>madvise</code> 而保留 materialized backing；
- 等待系统压缩或 swap。

本方案中的 offload 必须至少满足一种可验证语义：

1. 释放 GPU/MLX/host backing，权重 authority 回到 checkpoint/SSD；
2. 移交给独立 helper process，并按 combined system envelope 计费；
3. 转为经过 manifest 计费且确实更小的压缩 backing，同时旧 backing 已释放；
4. 进入明确的 cache state，cache bytes 仍计入 ledger 和 plan。

### 276.5 压力下降时的正确行为

首发版本不做 mid-request plan replacement。对于已准入的 2-slot plan：

- Tight 可以停止启动第二个可选 lookahead；
- 已存在的第二 slot 仍按 2-slot plan 计费直到释放；
- 不能因此把 request 的 admitted upper 改写成 1-slot upper；
- 真正的 1-slot 低预算执行必须在 preflight 选择独立 candidate/plan；
- Critical 只允许完成已提交 work、drain、失败或 quarantine，不能临时改 pinned prefix。

这样 runtime pressure action 不会让 plan digest 与实际执行策略分叉。

## 277. H3 的逐函数 semantic schedule 与 site 接线表

### 277.1 首发 H3 candidate 再收窄

第一条 H3 executable candidate 建议固定：

~~~text
adapter=h3_c_metal_streamed_v2
audio=false 或固定独立 audio key
dynamic_block_skip=false
first_block_cache=false
token_reduction=false
ANE/CoreML=false
LoRA=false
quant-cache mode=fixed-off 或独立认证值
TAE/full-VAE 二选一并进入 key
refill_slots=1 或 2，独立 record
~~~

任何环境变量或私有 configuration 使上述值改变，都必须在 preflight 前归一化或拒绝；不能在 H3 runtime 内部静默
改变 schedule。

### 277.2 顶层 session 接线

<code>native/platform/apple/h3_session.mm</code>：

1. <code>bind_memory_context()</code> 创建/填充 schedule hooks、allocator hooks 和同一 generation；
2. 将 schedule hooks 同时传给 top-level params、text options、DiT options、VAE/TAE options；
3. generate 前发 ADMISSION/END 或 TEXT/BEGIN；
4. text、DiT、VAE 每次 stage handoff 都在 owner thread 完成 drain/checkpoint；
5. generate 返回前执行 TERMINAL_DRAIN event；
6. <code>drain_memory_completions()</code> 成功、terminal event 消费完成后才允许
   <code>finish_success()</code>；
7. unbind 先把 native options 中 hook pointer 置空，再清 bridge context，最后增加 generation。

### 277.3 H3 事件插入点

| 文件/函数 | 事件 | 位置 |
|---|---|---|
| <code>h3_text_encode_*_with_options</code> | TEXT BEGIN/END | 任何 text tensor/weight allocation 前；deferred tensor 全部 retire 后 |
| <code>layer_prefetch_main</code> 的启动者 | TEXT PREFETCH | owner 启动线程前，不能在线程内发 |
| <code>retire_deferred</code> 的 owner 调用点 | TEXT DRAIN | GPU drain/mailbox drain 后 |
| <code>h3_dit_reprepare</code> | DENOISER_LOAD BEGIN/END | activation、stream slot 分配边界 |
| <code>read_stream_layer</code> 的启动者 | DENOISER PREFETCH | reserve/commit slot 后、pthread_create 前 |
| <code>h3_dit_forward</code> block loop | COMPUTE/LAST_USE/RETIRE | block encode 前、最后 use 后、submit 前后 |
| <code>h3_dit_drain_gpu</code> | DENOISER DRAIN | <code>h3_gpu_drain</code> 成功且 mailbox drain 后 |
| <code>h3_dit_release_request</code> | DENOISER END | request-scoped activation 已释放 |
| <code>h3_video_vae_decoder_decode</code> | VIDEO_VAE tile BEGIN/COMPUTE/RETIRE | tile backing acquire、dispatch、last-use |
| <code>h3_video_vae_decoder_drain_gpu</code> | VIDEO_VAE DRAIN | GPU 和 mailbox 均归零 |
| <code>h3_taeh3_decoder_decode</code> | VIDEO_VAE branch=TAE | 与 full VAE 不共享 schedule revision |

由于 <code>read_stream_layer_thread</code> 是背景线程，实际 event 必须由创建它的 block loop 发；worker 只更新
<code>h3_dit_stream_job</code> 的成功、bytes 和 timing。

### 277.4 H3 instance 映射

建议：

| site 族 | instance id |
|---|---|
| text layer weight/refill | layer index |
| DiT pinned block weight | block |
| DiT streamed refill backing | slot |
| DiT streamed block logical use | step * 50 + block |
| DiT step activation | step |
| VAE tile scratch/output | tile ordinal |
| request latent/conditioning/output | 0 |

一个物理 refill slot 跨多个 block 复用时，需要同时记录：

- physical storage identity：allocator domain、handle、generation；
- slot instance：0 或 1；
- logical block use：step/block；
- reuse count；
- 上一次 completion 已被 owner drain 的证据。

manifest 中 physical slot 可以是长期 interval；logical block use 可以是单独零 backing observation，或 trace binding。
二者不能混成一个动态 tag。

### 277.5 H3 必须先处理的 generic site

closure 前至少清理：

~~~text
h3_gpu_tensor
h3_host_memory
h3.stage
未带 component 的 activation/refill/output tag
~~~

处理方式只能是：

1. 替换为 registry site；
2. 证明首发 candidate compile-time/runtime unreachable，并在 closure exemption 中绑定 source revision；
3. 将整个 branch 标为 unsupported。

禁止把 generic tag 改名为另一个 generic tag 来通过字符串检查。

## 278. LTX 的逐函数 semantic schedule、双 stage 与 helper 边界

### 278.1 首发 LTX candidate 固定值

~~~text
adapter=ltx_c_metal_video_streamed_v2
audio=false
parallel_av=false
Sol/sparse=false
ANE=false
LoRA=false
video_attention_batch=固定认证值
batch_audio_commands=false
graph cache=disabled_per_request
helper_mode=in_process 或 disposable_child_v1，二选一
refill_slots=1 或 2，独立 record
~~~

现有 <code>ltx_native_options</code> 中任何影响 command batch、ANE attach/detach、stage mask 或 sparse route 的字段都必须
进入 candidate identity，或在首发 validator 中固定拒绝。

### 278.2 LTX 事件插入点

| 文件/函数 | 事件 | 位置 |
|---|---|---|
| <code>ltx_gemma_encoder_prepare_prompt</code> | TEXT BEGIN | packed input/graph allocation 前 |
| <code>ltx_gemma_encoder_encode</code> | TEXT COMPUTE/END | encode 前；text queue drain 后 |
| <code>ltx_native_create</code> | DENOISER_LOAD BEGIN/END | pinned block/refill slot 创建边界 |
| <code>ltx_native_run</code> | stage BEGIN/END | stage geometry 激活后；cleanup/drain 后 |
| <code>run_denoise_schedule</code> | step BEGIN/END | workspace 完成后；Euler 更新完成后 |
| <code>run_streamed_block_stack</code> | block PREFETCH/COMPUTE/RETIRE | start load 前、run block 前、last-use 后 |
| <code>start_streamed_block_load</code> 的 caller | PREFETCH | pthread_create 前 |
| <code>finish_streamed_block_load</code> 的 caller | UPLOAD/READY | join 和 actual 校验后 |
| <code>ltx_native_drain</code> | stage DRAIN | video/audio queue 均 drain，mailbox 归零后 |
| <code>ltx_native_upsample_stage2</code> | LATENT_HANDOFF BEGIN/END | stage1 release 与 stage2 acquire 之间 |
| <code>ltx_video_vae_decode_*</code> | VIDEO_VAE tile/chunk | graph/input/output lifetime 边界 |

<code>run_audio_stream_thread</code> 不能推进 shared cursor。audio candidate 若未来开启并行，应为 video/audio 各设
substream mailbox，由 owner 在 join barrier 发一个合并 event。

### 278.3 stage1/stage2 的 resource rule

stage1 END 必须证明：

~~~text
stage1 video/audio command queues drained
stage1 refill logical uses 全部 retire/complete
stage1 workspace 和临时 conditioning 已释放
只有 handoff latent、必要 conditioning 和计划允许的 physical slot 仍 live
~~~

进入 stage2 BEGIN 前：

- <code>ltx_native_upsample_stage2</code> 的 host <code>denormalized</code>、MLX upsampler 和 Metal stage1/stage2 buffer
  都必须有 registry site；
- <code>ltx_mlx_clear_cache()</code> 的效果要由 process checkpoint 验证，不能假定调用即释放；
- 如果 physical refill slot 跨 stage 复用，必须先 drain stage1，再由 stage2 显式 reserve/reactivate；
- stage1 和 stage2 的 logical instance id 绝不相同。

### 278.4 LTX queue/completion identity

LTX token 已有 queue id。下一轮再把以下字段写入 trace：

~~~text
queue_id
stage_id
step
block
slot_id
site_code
instance_id
allocator_domain
generation
completion_status
~~~

<code>complete()</code> callback 只能发布固定大小 message。若 mailbox 满、queue id 不一致、generation stale 或 status
非零，bridge 设置 sticky failure；owner 下一 safe point 必须失败并进入 cleanup。callback 不得 delete token 后继续让
owner 使用其地址。

### 278.5 helper process

若 VAE/helper 使用 child：

1. parent 在 spawn 前 reserve IPC/handoff envelope；
2. child 启动后返回自己的 baseline 和 identity；
3. parent/child 各自采样 process peak 和 swap counter source；
4. child 写 terminal child report，parent 只接收 bounded summary；
5. parent waitpid 确认退出后才能 release IPC/handoff；
6. child report 缺失、digest 不匹配或 timeout 都不能成功；
7. combined evidence 保存 parent peak、child peak 和时间重叠，但不把两者 peak 简单相加作为同一时刻事实；
8. admission 仍要有保守 combined upper，保证并发窗口不超系统预算。

## 279. Flux 与 Z-Image：MLX opaque framework envelope 的可执行设计

### 279.1 为什么不能套用逐 MTLBuffer hook

Flux2 和 Z-Image 当前主要通过 MLX tensor、Weights 容器和 framework graph 执行。底层临时 allocation 不完全经过
TurboCider 自有 allocator，因此首发不能宣称逐 allocation closure。要支持低内存模式，需要把可精确管理的权重/
输出与不可逐个拦截的 framework temporary 分开。

### 279.2 新增 opaque envelope

在 manifest v2 中增加：

~~~cpp
struct FrameworkEnvelopeSpec {
    std::string envelope_id;
    std::string component;
    MemoryEpochKey acquire;
    MemoryEpochKey release;
    uint64_t upper_bytes = 0;
    std::string profile_digest;
    std::string reset_action;
    bool process_isolated = false;
};
~~~

planner 把 envelope 当作普通 live resource 计入 peak；runtime 通过 process footprint 减去 ledger committed/pending/cache
得到 unattributed/framework bytes，并要求：

~~~text
unattributed_bytes <= active framework envelope upper
~~~

envelope upper 只能来自真实 campaign 的 validated envelope，不能来自当前 request 的首次观察。未知 framework/version/
shape bucket 只允许 plan-only。

### 279.3 Flux 首发 component-staged route

建议先实现组件级，不直接做 block streaming：

~~~text
TEXT_LOAD -> TEXT_COMPUTE -> TEXT_CLEAR/DRAIN
TRANSFORMER_LOAD -> DENOISE -> TRANSFORMER_CLEAR/DRAIN
VAE_LOAD -> VAE_DECODE -> VAE_CLEAR/DRAIN
EXPORT -> TERMINAL
~~~

逐文件建议：

| 文件 | 修改 |
|---|---|
| <code>native/models/flux2/pipeline.cpp</code> | stage event、明确 text/transformer/VAE handoff |
| <code>flux_text.cpp</code> | text Weights load/materialize/clear 前后 hook |
| <code>flux_transformer.cpp</code> | block event 仅用于 trace；无 allocator closure 前不认证 block streaming |
| <code>flux_vae.cpp</code> | decode envelope、decoded output 和 export staging 分开 |
| <code>native/backends/mlx.*</code> | 只读 active/cache/peak sample 和受控 clear helper，不伪装逐 array admission |
| 新 <code>flux_memory_sites.def</code> | exact weights/output/handoff site；opaque temporary 用 envelope |

首发 candidate 要求每个 stage clear 后 process footprint 在 bounded grace checkpoints 内回落。仅调用
<code>Weights::clear()</code> 或 MLX cache clear 但 footprint 不回落，candidate 认证失败。

### 279.4 Z-Image/GGUF 首发 route

Z-Image 至少拆成：

~~~text
z_image_safetensors_component_staged_v1
z_image_gguf_resident_plan_only_v1
z_image_gguf_component_staged_v1
~~~

当前 GGUF executor 只支持 resident 时，不能通过 scheduler 强行把 residency 字符串改成 streamed。component-staged
candidate 要先实现：

- text encoder load/encode/clear；
- transformer materialize 与 VAE materialize 不同时驻留，或证明必须同时驻留并计入 peak；
- GGUF mapped logical bytes、resident clean pages、dequant/materialize bytes 分别观测；
- LoRA merge scratch 与 fused output 单独 manifest；
- <code>mx::get_active_memory()</code> 与 process footprint 双重 checkpoint；
- graph/cache mode 进入 capability key。

<code>ZImage::load()</code> 当前会 materialize transformer 和 VAE；若要 component staging，需要拆为独立
<code>load_transformer()</code>、<code>load_vae()</code> 和 release barrier。这个重构必须在 disabled path 保留原来的
<code>load()</code> 顺序和缓存行为，受限模式通过独立方法进入。

### 279.5 MLX candidate 的额外验收

每个 stage 至少采样：

~~~text
before_load
after_materialize
before_compute
after_eval
after_clear_requested
after_clear_grace_1
after_clear_grace_2
~~~

若 after_clear 最终未回落到 manifest 允许的 next-stage baseline，不能用更长 sleep 掩盖；应提高 envelope、隔离进程或
保持 plan-only。

## 280. 预算、admission 与压力控制器的进一步收敛

### 280.1 不使用静态百分比分仓

不建议把 B 固定切成“权重 60%、activation 25%、output 15%”。正确做法是由 live interval 算每个 epoch 的总峰值：

~~~text
epoch_upper =
    process_baseline
  + framework_envelopes_live
  + concrete_storage_alias_groups_live
  + host_staging_live
  + helper/IPC_live
  + output/export_live
~~~

分仓百分比只能作为 candidate 枚举的启发式，不能作为 admission 证明。

### 280.2 reserve 时需要同时考虑 ledger 与观测 headroom

建议 reserve gate：

~~~text
ledger_future =
    committed + reserved + pending_release + requested_upper

observed_future =
    latest_process_footprint
  + max(0, requested_upper - already_backed_alias_credit)
  + framework_guard

accept iff:
    ledger_future <= plan_increment_upper
    process_baseline + ledger_future + framework_upper <= B
    observed_future <= B
    system_available_after_guard >= S
~~~

alias credit 只能来自同一 plan 声明、旧 lifetime 已结束且 storage identity 可证明复用的 backing。不能因为两个 tensor shape
相同就给 credit。

### 280.3 pressure 水位建议

不用固定“80%/90%”覆盖所有模型。阈值应与下一笔最大合法 allocation 关联：

~~~text
next_required_upper = plan.max_required_allocation_after(current_epoch)
normal_headroom = next_required_upper + framework_guard + jitter_guard
tight_headroom = next_required_upper + framework_guard
critical_headroom = next_required_upper
~~~

状态：

| 条件 | 动作 |
|---|---|
| headroom >= normal_headroom | 允许 plan 内 optional lookahead |
| tight_headroom <= headroom < normal | 停止 optional lookahead，尽快 drain |
| critical_headroom <= headroom < tight | 不启动下一 block，完成当前提交并 checkpoint |
| headroom < critical 或 observer over budget | fail-closed；按 cleanup 证明 clean/quarantine |

阈值和 <code>jitter_guard</code> 必须进入 framework profile/candidate identity，不能由运行中自由学习后反向修改本次预算。

### 280.4 Y 小于基线的行为

若 clean baseline 已经大于 B：

- constrained request 在任何模型 allocation 前拒绝；
- 结果显示 baseline、B、超出 bytes；
- service 可以选择新 worker 或 disposable child 重新 probe，但不能在同一 engine 上边 unload 边尝试；
- disabled request 不受此 gate 影响；
- 不自动把 Y 解释为“额外可用内存”。

### 280.5 预算变更

request 开始后 Y/X/S 均不可变。UI 或 service 修改配置只影响下一 request。为了支持未来 broker reclaim，可以增加独立
cancel/retry 协议，但首版不允许在同一 execution context 内缩小预算后继续。

## 281. capability identity v2 的 canonical 编码与验证

### 281.1 不使用分隔符拼接裸字符串

当前 canonical helper 已经支持 length-prefixed field。v2 继续采用：

~~~text
|field_name=value_length:value
~~~

所有字段按固定 schema 顺序编码，不依赖 struct declaration、map iteration、locale 或 JSON key order。空字符串只有在
schema 明确允许时有效；unknown 不能用空字符串伪装。

### 281.2 v2 字段全集

建议 key 的顺序固定为：

~~~text
schema
adapter
model_id
checkpoint_digest
backend
dtype
model_variant
operation
shape_bucket
sampler_mode
refill_slots
tiling_mode
runtime_revision
device_family
schedule_revision
source_inventory_digest
framework_profile_digest
helper_mode
process_topology
platform_build_class
allocator_revision
graph_cache_mode
~~~

record 再绑定：

~~~text
candidate_key_digest
manifest_digest
plan_schema
schedule_digest
closure_digest
campaign_digest
evidence_digest
maximum_validated_process_peak_bytes
maximum_validated_unattributed_bytes
minimum_validated_system_available_bytes
swapout_zero_required
require_explicit_epoch
require_explicit_instance
~~~

### 281.3 registry add 的完整校验

<code>MemoryCapabilityRegistry::add()</code> 应拒绝：

- v1 key 或空 schema；
- 任一 digest 不是 64 位小写 hex；
- record 中的 candidate digest 无法由 key 重算；
- schedule/source/framework digest 与 key 字段不一致；
- executable record 未要求 explicit epoch/instance；
- certified record 缺 closure/campaign/evidence；
- <code>release_enabled=true</code> 但 state 不是 certified；
- refill slots 或 tiling 不在 verified set；
- <code>maximum_validated_upper_bytes</code> 超过对应设备/shape campaign 上界；
- <code>swapout_zero_required=false</code> 的 production record。

测试要通过 table-driven field mutation 逐一证明 equality、ordering、canonical 和 digest 都变化。

### 281.4 platform build class

建议由专用 probe 输出有限字段：

~~~text
os_major
metal_language/runtime major
device family
unified memory=true
page size class
framework build revision
~~~

不要绑定机器序列号、用户名或精确可识别硬件 id。未知组合返回
<code>platform_build_class=unknown</code>，production lookup miss；experimental 工具可以记录但不能自动找最近邻。

## 282. service evidence writer 的实现级规范

### 282.1 分层实现

建议分为三个对象：

~~~text
MemoryEvidenceSnapshot
  纯内存、不可变、已经去敏

MemoryEvidenceEncoder
  生成 canonical JSON/JSONL bytes，不访问模型或系统 probe

MemoryEvidenceStore
  service 负责目录、原子写、retention、错误报告
~~~

runtime 只负责冻结事实，service 负责落盘。这样 library 调用者不被强制写文件，disabled/default 路径仍然零 writer。

### 282.2 snapshot 字段

~~~cpp
struct MemoryEvidenceSnapshot {
    std::string schema;
    std::string request_id;
    std::string runtime_revision;
    std::string sanitized_request_json;
    std::string device_json;
    std::string candidate_json;
    std::string manifest_json;
    std::string plan_json;
    std::string closure_json;
    std::string terminal_report_json;
    std::string cleanup_json;
    std::string primary_error_class;
    std::string primary_error_message;
    std::vector<MemoryTraceEvent> trace;
};
~~~

snapshot 构造失败不能覆盖 primary runtime 结果；它生成独立
<code>evidence_status=incomplete</code> 和 <code>evidence_failure</code>。

### 282.3 文件与大小上限

建议：

| 文件 | 上限 |
|---|---:|
| request/normalized-request JSON | 各 256 KiB |
| device/candidate/manifest/plan JSON | 各 4 MiB |
| report/cleanup JSON | 各 1 MiB |
| trace JSONL | 32 MiB 或固定 event count |
| errors JSON | 256 KiB |
| digest JSON | 1 MiB |

超过上限时不能静默截成合法 JSON。应写一个小型 replacement object，包含
<code>truncated=true</code>、原长度、保留 event count 和 failure class，并让认证 verifier 失败。

### 282.4 原子落盘

service 流程：

~~~text
resolve configured evidence root once at startup
validate request id
mkdir private temp directory under same root
open files with no-follow/exclusive semantics
write encoded bytes
fsync files（认证 runner 必须；普通 diagnostic 可配置）
compute sorted per-file SHA-256
write digest.json last
fsync temp directory
rename temp -> final request directory
fsync root directory
~~~

final 目录已存在时使用 server 生成的 monotonic attempt id，不覆盖旧 evidence。任何 partial temp 目录在启动清理时只移动到
quarantine/partial，不直接当成功 evidence。

### 282.5 root digest

root digest 输入固定为：

~~~text
schema
candidate_key_digest
manifest_digest
plan_digest
closure_digest
sorted(file_name, file_size, file_sha256)
terminal execution state
failure disposition
~~~

mtime、绝对目录、inode、PID 不进入 digest。trace 中的 pointer/handle 必须先转为 request-local opaque id。

### 282.6 service 接线

<code>services/turbociderd/service.mm</code>：

1. 只对 enabled constrained request 创建 evidence intent；
2. engine 返回后立即 take terminal report；
3. 读取 API 层冻结的 candidate/manifest/plan snapshot，而不是从日志重建；
4. 无论 success/failure/quarantine 都尝试写 artifact；
5. writer 失败保留原 HTTP/IPC error code，另加 evidence diagnostic；
6. quarantined worker 先保存 snapshot，再销毁；若销毁前无法保存，记录 service-level incident；
7. request response 只返回 evidence id/digest 和相对标识，不返回本机绝对路径。

## 283. 测试、故障注入和性能验收的新增细节

### 283.1 schedule L0 测试

新增：

~~~text
memory_schedule_test.cpp
  exact sequence success
  missing event
  duplicate event
  backward event
  unknown coordinates
  slot mismatch
  forged GPU_DRAINED flag
  callback after unbind/generation change
  terminal event missing
~~~

每个失败都检查：

~~~text
cursor 未错误推进
automatic epoch 仍为 0
primary error class 稳定
terminal report 可读取
cleanup disposition 符合预期
~~~

### 283.2 explicit instance L0 测试

覆盖：

- strict plan 调用 string-only reserve 失败；
- site code 正确但 instance 错误失败；
- step/block formula overflow 失败；
- 同 instance 超 maximum live count 失败；
- 同 physical storage 在未 completion 前被下一 logical use 复用失败；
- alias group 合法复用只计一次 backing；
- tombstone site code 不能出现在 manifest；
- test code 范围不能加入 production registry。

### 283.3 adapter L1 synthetic 测试

H3/LTX 使用 fake allocator、fake queue、fake loader 跑完整 schedule：

| fault | 注入点 | 期望 |
|---|---|---|
| reserve denied | block/tile 前 | 无真实 allocation，clean failure |
| host allocation fail | reserve 后、commit 前 | cancel token |
| short read | Filling | slot poison，停止新 submit |
| actual > upper | commit | fail-closed，必要时 quarantine |
| command encode fail | Ready | 未 submit backing 同步 release |
| command commit fail | Submitted 边界 | 按平台可证明性 clean/quarantine |
| completion status fail | callback | sticky，owner drain 失败 |
| mailbox overflow | callback | quarantine |
| cancel | 每个 safe point | 无新 prefetch，完成 cleanup |
| swap delta | 任一 checkpoint | request 失败，预算不变化 |

### 283.4 L2 Metal 事件证据

真实 Metal run 必须证明每个 async backing 有以下 trace 子序列：

~~~text
reserve
commit
schedule compute
submit
retire
completion posted
mailbox consumed
release/reusable
~~~

时间戳只能用于性能，正确性依赖 token sequence/domain/generation。即使 callback 时间早于 owner 下一 checkpoint，也必须
等 owner drain 后才能 reuse。

### 283.5 L3 低内存矩阵

每个 H3/LTX candidate 至少：

| 维度 | 点 |
|---|---|
| 物理内存 | 16/24/32 GiB 或审核通过的等价机器 |
| buffer X | 10%、15%、20% |
| slot | 1、2（分别有 record） |
| shape | 每个认证 bucket 的最小/典型/最大 |
| 路径 | cold、warm、cancel、fault |
| 重复 | 每点至少 3 次，任一超预算即失败 |

通过条件除既有 peak/swap 外，再加：

~~~text
automatic_epoch_transition_count == 0
implicit_instance_attempt_count == 0
schedule_cursor_complete == true
generic_site_seen == false
closure_status == closed
evidence_root_digest_recomputes == true
post-request zero-state == true
~~~

### 283.6 disabled ABBA

每个 adapter 执行：

~~~text
A disabled baseline
B compiled with feature but disabled
B constrained enabled（只作功能）
A disabled baseline
~~~

对 A/B-disabled 比较：

- wall time、GPU time、allocation count、peak footprint；
- model load/unload 次数；
- MLX/Metal cache clear 次数；
- process/swap probe 调用数必须为 0；
- schedule/allocator hook callback 数必须为 0；
- 输出 bitwise/tolerance parity。

性能门限用历史噪声分布确定，但任何新增 probe/hook/unload 行为都是硬失败，不靠统计豁免。

## 284. 逐文件代码修改建议：下一轮可直接拆成 PR

### 284.1 PR-S1：共享 schedule ABI

修改：

~~~text
native/core/memory_schedule_c.h                 新增 POD ABI
native/runtime/memory_schedule.hpp/.cpp         key、binding、cursor
native/runtime/memory_plan.hpp/.cpp             schedule table/digest
native/runtime/memory_execution.hpp/.cpp        emit、terminal completeness
tests/native/memory_schedule_test.cpp
tests/native/test_memory_schedule.py
tools/native/build.sh
Makefile
~~~

完成门：

- 无模型接线也能通过 pure C++ fixture；
- disabled contract 无新增调用；
- plan digest 绑定 schedule；
- strict fixture 没有 allocation auto-advance。

### 284.2 PR-S2：H3 schedule bridge

修改：

~~~text
native/models/h3_runtime/h3.h
native/models/h3_runtime/h3_memory.h
native/models/h3_runtime/h3_gpu.h
native/models/h3_runtime/h3_text_encoder.h/.c
native/models/h3_runtime/h3_dit.h/.c
native/models/h3_runtime/h3_video_vae.h/.c
native/models/h3_runtime/h3_taeh3.h/.c
native/platform/apple/h3_session.mm
tests/native/h3_schedule_memory_test.c
tests/native/h3_gpu_memory_hooks_test.mm
~~~

完成门：

- synthetic full request cursor 完整；
- 背景 loader 不调用 schedule hook；
- H3 strict fixture automatic epoch 为 0；
- disabled H3 callback count 为 0。

### 284.3 PR-S3：LTX schedule bridge

修改：

~~~text
native/models/ltx_runtime/ltx_native.h
native/models/ltx_runtime/ltx_blocks.c
native/models/ltx_runtime/ltx_gemma_encoder.h/.m
native/models/ltx_runtime/ltx_video_vae.h/.m
native/platform/apple/ltx_session.mm
tests/native/ltx_gpu_memory_hooks_test.mm
新增 ltx schedule fixture
~~~

完成门：

- stage1、handoff、stage2、VAE 顺序固定；
- video-only 没有 audio event；
- queue drain flags 不能伪造；
- helper mode 变化导致 identity 变化。

### 284.4 PR-I1：site registry 与 explicit instance

修改：

~~~text
native/models/h3_runtime/h3_memory_sites.def
native/models/ltx_runtime/ltx_memory_sites.def
native/runtime/memory_site_registry.hpp/.cpp
native/runtime/memory_manifest.hpp/.cpp
native/runtime/memory_plan.hpp/.cpp
native/runtime/memory_execution.hpp/.cpp
H3/LTX hook struct 尾部追加 v3 reserve_site
tools/memory/generate_site_inventory.py
tools/memory/verify_required_site_closure.py
~~~

完成门：

- 所有首发 reachable allocation 使用 code+instance；
- generic site 在 strict path 硬失败；
- closure fixture status=closed；
- production registry 仍为空。

### 284.5 PR-K1：capability identity v2

修改：

~~~text
native/runtime/memory_manifest.hpp/.cpp
native/platform/apple/memory_probe.hpp/.mm
native/platform/apple/h3_session.mm
native/platform/apple/ltx_session.mm
native/platform/apple/profile.mm
native/platform/apple/results.mm
tests/native/memory_manifest_test.cpp
tests/native/memory_probe_test.mm
~~~

完成门：

- 所有新增字段 equality/order/canonical/digest 一致；
- v1 sidecar 只能 plan-only；
- unknown platform/framework fail-closed；
- 不存在 wildcard/nearest-match lookup。

### 284.6 PR-E1：evidence snapshot 与 writer

修改：

~~~text
native/runtime/memory_evidence.hpp/.cpp
native/api/c_api.mm
bindings/c/include/turbocider/turbocider.h
services/turbociderd/memory_evidence.mm
services/turbociderd/service.mm
native/platform/apple/results.mm
tools/memory/verify_evidence.py
tests/native/memory_evidence_test.cpp
tests/native/memory_report_c_api_test.mm
~~~

完成门：

- success/failure/quarantine 均有 terminal artifact；
- writer fault 不覆盖 primary error；
- digest tamper 可检测；
- path traversal/symlink/overwrite 被拒绝；
- disabled request 不创建目录。

### 284.7 PR-M1：Flux/Z-Image plan-only probe

此 PR 只增加：

- metadata-only component inventory；
- stage template；
- opaque framework envelope schema；
- deterministic plan/golden；
- 明确的 <code>execution_supported=false</code> 原因。

不增加 executable record。component-staged allocator/context 接线和真实 framework campaign 应拆到后续独立 PR。

## 285. 实现和验收的 reviewer checklist

### 285.1 代码 review

~~~text
[ ] disabled/default 请求未创建 MemoryExecutionContext
[ ] disabled/default 请求未安装 schedule/allocation hooks
[ ] schedule callback 只由 owner thread 调用
[ ] adapter 不传 epoch id
[ ] strict plan 不允许 automatic epoch 或 implicit instance
[ ] reserve 发生在真实 allocation 之前
[ ] commit 使用真实 backing bytes
[ ] async last-use 经过 retire/complete/mailbox drain
[ ] slot reuse 只发生在 Reusable 状态
[ ] generic site 在 strict path 不可达
[ ] all digest fields 在 equality/order/canonical 中一致
[ ] evidence writer 不重新 probe 模型/allocator
[ ] failure 不会被 writer 错误覆盖
[ ] production registry 仍为空，除非真实 L2/L3 签核另行完成
~~~

### 285.2 单 candidate release 签核

每条 record 必须附：

~~~text
candidate-key.json
source-inventory.json
manifest.json
schedule.json
plan.json
closure.json
framework-profile.json
L2 campaign index
L3 campaign index
quality report
performance report
disabled ABBA report
release approval
~~~

签核者需要独立重算所有 digest，并从 trace 随机抽查至少：

- 一个 request-persistent backing；
- 一个 block/tile instance；
- 一个 asynchronous completion；
- 一个 slot reuse；
- 一个 stage handoff；
- 一个失败 cleanup。

### 285.3 发布前硬门

任何一项失败都保持 plan-only：

~~~text
schedule cursor incomplete
automatic epoch > 0
implicit instance attempt > 0
unknown/generic site > 0
actual > instance upper
unattributed > framework upper
process peak > B
system available < S 且未按规则失败
swapouts delta > 0
completion stale/overflow/status failure
terminal evidence 缺失或 digest 不一致
disabled ABBA 新增行为或显著回归
quality 不满足 candidate contract
~~~

## 286. 本轮设计增补结论

本轮没有把任何模型标记为可发布，也没有增加 production capability record。新增设计把下一步工作从“补一些 hooks”
进一步收敛为以下可验证链路：

~~~text
semantic schedule event
  -> plan resolver/cursor
  -> explicit epoch
  -> explicit site + instance
  -> reserve/commit/retire/complete
  -> owner safe-point checkpoint
  -> closure/evidence
  -> L2/L3 certification
  -> production registry
~~~

最重要的工程决策是：

1. schedule event 由 adapter 表达语义，但 epoch authority 属于 compiled plan；
2. 背景 I/O 和 Metal callback 不能直接推进 execution context；
3. 首版压力控制不做 mid-request plan replacement；
4. 统一内存上的“复制到 host”不自动等于 offload；
5. MLX 模型在没有逐 allocation hook 时必须使用认证过的 opaque framework envelope，并优先从 component staging 开始；
6. H3/LTX 先冻结最小 candidate，关闭会改变 allocation graph 的动态分支；
7. evidence writer 只是记录事实，不能改变 runtime 结果；
8. 在真实 Metal completion、required-site closure 和低内存 swapout=0 campaign 完成前，production registry 继续为空。

后续实现者建议优先阅读第 272–285 节，并按 PR-S1、S2、S3、I1、K1、E1 的依赖顺序施工。每完成一个 PR，都应把
源码事实、测试命令、skip 原因和新增 evidence 样例回写到本文，而不是只更新任务状态。

## 287. 2026-09-16 当前实现快照与事实边界

本节记录第 284.1 节 PR-S1 在当前工作树中的实际状态。它是实施日志和后续 patch 的输入，不是 release 声明。

### 287.1 已进入当前工作树的 schedule scaffolding

当前已有：

~~~text
native/core/memory_schedule_c.h
  - tc_memory_schedule_event_v1
  - tc_memory_schedule_hooks_v1
  - stage/action/branch enum
  - SAFE_POINT/GPU_DRAINED/PREFETCH_ALLOWED/TERMINAL flags

native/runtime/memory_schedule.hpp/.cpp
  - MemoryScheduleEventKey
  - MemoryScheduleBindingSpec
  - CompiledScheduleBinding
  - CompiledMemorySchedule
  - enum/flag/terminal/epoch/duplicate 校验
  - deterministic SHA-256 schedule revision

native/runtime/memory_plan.hpp/.cpp
  - MemoryPlanCompileOptions
  - require_explicit_schedule
  - schedule_schema/schedule_revision/schedule_bindings
  - plan digest 绑定 schedule requirement 和 revision

native/runtime/memory_execution.hpp/.cpp
  - make_schedule_hooks()
  - owner-thread emit_schedule_event()
  - semantic key/slot/flags strict cursor
  - safe-point completion mailbox drain
  - GPU_DRAINED pending-state 校验
  - schedule-driven explicit epoch
  - finish_success() cursor completeness gate

native/runtime/memory_accounting.hpp
native/platform/apple/results.mm
  - schedule accepted/expected/mismatch/complete 指标及 JSON
~~~

兼容 plan 仍可不带 schedule；strict schedule 目前由
<code>MemoryPlanCompileOptions.require_explicit_schedule</code> 显式开启。capability record/probe 尚未强制提供 schedule，
因此当前实现不能形成 executable production capability。

### 287.2 2026-09-16 定向测试结果

本轮实际运行：

~~~text
python3 -B tests/native/test_memory_schedule.py
  PASS

python3 -B tests/native/test_memory_plan_compiler.py
  PASS

python3 -B tests/native/test_memory_execution.py
  FAIL: strict schedule terminal event assertion

python3 -B tests/native/test_contract.py
  81 PASS / 1 fixture SKIP
~~~

execution 用例失败发生在 terminal event 的 <code>checkpoint_after=true</code>。fixture 构造 context 时只给了 admission
baseline，<code>MemoryAdmission</code> 构造函数消费第一次 observer sample；terminal checkpoint 再取样时回退到真实
<code>observe_process_memory()</code>。测试使用的预算是几十字节量级，因此真实进程 footprint 必然超过预算，callback
返回失败。该失败不是 Metal 缺失，也不能记录为 strict cursor 已通过。

后续状态：该 fixture 已按下述方案注入 deterministic observer，并增加精确 observation count 断言；当前测试已 PASS。
本段保留作为测试设计缺陷的根因记录，最新实施结果见第 305 节。

建议测试修正不是放宽 budget，而是传入 deterministic observer sequence：

~~~cpp
ProcessMemoryObserver stable_observer(uint64_t footprint,
                                      uint64_t available) {
    return [footprint, available] {
        return ProcessMemoryObservation{
            true, footprint, 0, available, "synthetic"};
    };
}

auto context = std::make_unique<MemoryExecutionContext>(
    policy, std::move(plan), baseline,
    stable_observer(/*footprint=*/10, /*available=*/512));
~~~

测试还必须断言 observer 调用次数，防止未来多一次隐式 probe 仍因返回相同数值而被掩盖。

### 287.3 当前仍未完成

~~~text
H3 schedule hook ABI 接线                         未完成
LTX schedule hook ABI 接线                        未完成
request top-level terminal event ownership         未完成
explicit schedule 与 progress checkpoint 去重      未完成
sticky schedule poison/reentrancy guard             未完成
schedule coverage 到 allocation instance 的闭包    未完成
required-site explicit instance registry            未完成
capability identity v2                              未完成
service evidence writer                             未完成
真实 Metal L2                                      未完成
16/24/32 GiB 或等价低内存 L3                       未完成
Flux/Z-Image executable constrained route           未完成
~~~

production capability registry 必须继续为空。任何只依赖 pure C++ PASS 的结论最多是 L0 scaffolding evidence。

## 288. strict schedule cursor 的单向事务、中毒状态与错误协议

现有 cursor 能发现乱序，但还需要明确“发生副作用后失败”如何处理。这里不能尝试通用 rollback：completion mailbox 已被
消费、epoch 可能已推进、watchdog 可能已记录 failure、swap sample 也不能撤销。因此 schedule event 是单向事务；成功
commit 或进入 poisoned，两者之外没有 retry 状态。

### 288.1 建议新增状态

在 <code>native/runtime/memory_execution.hpp</code> 增加：

~~~cpp
enum class MemoryScheduleCursorState : uint8_t {
    Disabled,
    Healthy,
    Processing,
    Poisoned,
    Complete,
};

struct MemoryScheduleCursorStatus {
    MemoryScheduleCursorState state =
        MemoryScheduleCursorState::Disabled;
    uint64_t attempted = 0;
    uint64_t accepted = 0;
    uint64_t rejected = 0;
    uint64_t next_sequence = 0;
    uint64_t last_accepted_sequence = UINT64_MAX;
    std::string first_failure;
};
~~~

<code>MemoryExecutionContext</code> 持有：

~~~cpp
MemoryScheduleCursorStatus schedule_cursor_status_;
bool schedule_emit_active_ = false;
~~~

初始化规则：

- 没有 schedule binding：<code>Disabled</code>；
- 有 binding：<code>Healthy</code>；
- 接受最后一个 binding 后：<code>Complete</code>；
- 任一校验、drain、epoch、checkpoint 或 trace hard failure：<code>Poisoned</code>；
- <code>Processing</code> 只在 owner thread 的同步 callback 栈内存在。

### 288.2 emit 的精确算法

建议把实现拆成 prepare/apply/commit 三段，避免一个长函数难以 review：

~~~cpp
PreparedScheduleEvent prepare_schedule_event(
    const tc_memory_schedule_event_v1 &event);

void apply_schedule_event(
    const PreparedScheduleEvent &prepared);

void commit_schedule_event(
    const PreparedScheduleEvent &prepared);
~~~

主流程：

~~~text
1. attempted++
2. require owner thread
3. CAS-like local guard: Healthy -> Processing
4. prepare:
   - ABI/version/size
   - cursor bounds and contiguous sequence
   - exact key/slot/flags
   - target epoch exists and does not move backwards
   - terminal/non-terminal relation
5. apply:
   - SAFE_POINT: drain completion mailbox
   - GPU_DRAINED: verify zero pending/outstanding/mailbox/overflow
   - validate old epoch resources can cross boundary
   - enter target epoch
   - optional checkpoint/process/swap/watchdog
6. commit:
   - record accepted schedule trace
   - accepted++ / next_sequence++
   - if last event: Complete; else Healthy
7. return success
8. catch any failure:
   - rejected++
   - store first_failure once
   - state = Poisoned
   - mark execution tainted
   - record best-effort rejection trace
   - rethrow to C callback
~~~

prepare 阶段不得修改 ledger、epoch、mailbox 或 metrics 中的 accepted count。apply 阶段允许不可逆副作用，因此任何失败
都必须 poison。commit 阶段只做不应失败的有界操作；若 trace buffer 满，记录 dropped/overflow 指标，但不能因为诊断
写入失败而把已成功的运行时事务重新解释为未执行。

### 288.3 为什么不允许 retry

以下事件使 retry 无法保持 exactly-once：

- completion mailbox 已消费；
- pending token 已从 Pending 转为 Reusable/Released；
- epoch 已推进并释放旧 lifetime；
- checkpoint 已采样 swap/process，sample count 已增加；
- watchdog 已发生 pressure transition；
- adapter 可能已执行真实 allocation 或 command submission。

因此 callback 返回 0 后，adapter 只能停止新工作并向顶层返回失败。即使错误是 error buffer 太小、trace overflow 或用户
progress callback 抛错，也不能再次发送同一个 schedule event。

### 288.4 C callback 错误字符串

当前 callback 把 exception message 写入 caller buffer。建议同时保存完整内部错误，并保证外部 buffer 的稳定前缀：

~~~text
memory_schedule_invalid:
memory_lifetime_violation:
memory_observation_unreliable:
memory_swap_activity_detected:
memory_pressure_abort:
~~~

当 <code>error_size</code> 不足时，至少保留完整错误类和 NUL；terminal report 使用内部完整错误，不能从截断 buffer 反向
重建。adapter 不得用自己的字符串覆盖 callback error。

### 288.5 指标语义

建议替换模糊的单一 count：

| 字段 | 语义 |
|---|---|
| <code>schedule_event_attempted_count</code> | 进入 callback 的次数 |
| <code>schedule_event_accepted_count</code> | 完整 commit 的次数 |
| <code>schedule_event_rejected_count</code> | 进入 poisoned 的失败次数；首版应为 0 或 1 |
| <code>schedule_expected_event_count</code> | compiled binding 数 |
| <code>schedule_next_sequence</code> | 下一期望 sequence |
| <code>schedule_cursor_state</code> | disabled/healthy/processing/poisoned/complete |
| <code>schedule_cursor_complete</code> | state==Complete 且 accepted==expected |
| <code>schedule_first_failure</code> | sticky 内部完整错误 |

保留旧 <code>schedule_event_count</code> 时，应定义为 accepted count，并在 schema 升级前保持兼容。

### 288.6 必须新增的 L0 用例

~~~text
reentrant callback                         -> poisoned
callback from non-owner thread             -> poisoned
failure before apply                       -> epoch/cursor 不变
completion drain failure                   -> poisoned，cursor 不前进
checkpoint over-budget after epoch advance -> poisoned，禁止 retry
event after poisoned                       -> 返回 first failure 派生错误
event after complete                       -> unexpected extra event
finish_success while healthy/incomplete    -> failure
finish_success while poisoned              -> failure
trace overflow after successful apply      -> accepted，但 trace_overflowed=true，认证失败
~~~

## 289. schedule compiler 的闭包校验与 canonical identity

当前 compiler 已校验枚举、flags、终止事件、epoch 单调和 duplicate key。要让它成为 certification compiler，还需要把
schedule 与 request shape、manifest instance、slot plan 和 adapter template 做完整闭包。

### 289.1 建议 compiler 输入

~~~cpp
struct MemoryScheduleCompileContext {
    std::string adapter;
    std::string operation;
    std::string template_revision;
    std::string manifest_digest;
    uint32_t denoise_passes = 0;
    uint32_t steps_per_pass[2] = {};
    uint32_t blocks = 0;
    uint32_t video_tiles = 0;
    uint32_t audio_tiles = 0;
    uint32_t refill_slots = 0;
    bool has_text = false;
    bool has_video = false;
    bool has_audio = false;
    bool dynamic_branching = false;
};

struct MemoryScheduleCompileResult {
    CompiledMemorySchedule schedule;
    std::vector<std::string> warnings;
    std::string coverage_digest;
};
~~~

production candidate 必须 <code>dynamic_branching=false</code>；warnings 非空时只能生成 plan-only candidate。

### 289.2 结构校验

compiler 必须额外要求：

1. <code>require_explicit_schedule=true</code> 时同时要求 <code>require_explicit_epoch=true</code>；
2. 第一个 event 为 request-owner admission/end 或首个显式 stage begin，且 epoch 为 0；
3. 恰好一个 TERMINAL，且为最后一项；
4. TERMINAL 同时具有 SAFE_POINT 与 GPU_DRAINED；
5. <code>checkpoint_after=true</code> 必须有 SAFE_POINT；
6. GPU_DRAINED 必须有 SAFE_POINT；
7. PREFETCH_ALLOWED 只能用于 plan 明确允许启动 I/O 的边界；
8. slot 为 NONE 或小于 <code>refill_slots</code>；
9. step/block/tile 均在 request bucket 范围内；
10. branch 与 stage 合法，例如 AUDIO branch 不得出现在 video-only candidate；
11. schedule 中所有 epoch 均属于 plan epoch；
12. 所有含 required live instance 的 epoch 均可由至少一个 event 到达；
13. 任何 release epoch 前都有 last-use/retire/drain 证明；
14. terminal epoch 等于 plan 最终 epoch。

### 289.3 allocation coverage

对每个 <code>CompiledAllocationInstance</code> 生成 coverage row：

~~~text
site_id
instance_id
acquire_binding_sequence
last_use_binding_sequence
retire_binding_sequence or NONE
release_binding_sequence
slot or NONE
asynchronous
~~~

校验：

- acquire binding 的 epoch 等于或早于 <code>live_begin</code>，且 allocation 实际只能在进入 begin 后发生；
- synchronous instance 的 release binding 不早于 last-use；
- asynchronous instance 必须有 retire 和后续 GPU_DRAINED/complete 证明；
- alias group 的新 logical instance 不早于旧 instance 可复用；
- refill slot 在相同 sequence 区间内最多绑定一个 Submitted instance；
- required instance 缺任一 row 时 compile fail，而不是 runtime 再猜。

coverage rows 按 <code>(site_id, instance_id)</code> 排序后计算 digest，并同时绑定到 plan 和 capability record。

### 289.4 canonical schedule revision

schedule revision 输入至少包括：

~~~text
schema
adapter
operation
template_revision
manifest_digest
shape bucket
steps/pass boundaries
block/tile counts
refill slot count
ordered binding fields
ordered coverage rows
checkpoint placement
all required flags
~~~

不能只 hash binding vector；否则同一事件序列用于不同 shape/manifest 时可能错误复用 identity。canonical integer 使用十进制
无前导零或固定宽度 binary，两者择一并固定；字符串使用 length-prefix，不使用裸分隔符拼接。

### 289.5 plan compiler 失败协议

所有 validation error 使用确定性错误类和 path：

~~~text
memory_schedule_invalid: bindings[17].slot exceeds refill_slots
memory_schedule_invalid: instances[h3.dit.block_weight,42] has no retire binding
memory_schedule_invalid: terminal epoch 19 differs from final plan epoch 20
memory_plan_invalid: schedule coverage digest does not bind manifest digest
~~~

不要把第一个失败改成 unordered container 遍历结果。compiler 对 site/instance/event 都应排序后报告，以便 golden 和
CI 在不同编译器上稳定。

## 290. request 顶层编排：checkpoint 去重与 terminal event 所有权

当前 <code>native/api/c_api.mm</code> 在 progress 的 <code>current==0 || current==total</code> 时自动 checkpoint，并在
session 返回后再执行 <code>checkpoint("complete")</code>/<code>checkpoint("prepare_complete")</code>。一旦 adapter 发
explicit schedule，这些通用 checkpoint 会与 schedule checkpoint 重复，且 terminal GPU drain 的时序不清晰。

### 290.1 progress callback 修改

generate 和 prepare 两条路径都改成：

~~~cpp
if (memory_execution &&
    !memory_execution->uses_explicit_schedule() &&
    (current == 0 || current == total)) {
    memory_execution->checkpoint(phase);
}
~~~

disabled request 的 <code>memory_execution</code> 仍为 null，因此不会新增 method call、probe 或 branch 内工作。compatibility
constrained plan 保留旧 checkpoint；strict schedule 只由 binding 的 <code>checkpoint_after</code> 决定采样点。

### 290.2 terminal event 由 request wrapper 发出

模型 adapter 不能可靠知道所有跨 session 的 GPU/MLX drain 是否已经结束。推荐所有模型只发到自己的最后一个
EXPORT/decoder event；统一的 <code>TERMINAL_DRAIN.END</code> 由 <code>c_api.mm</code> request owner 在完成以下动作后发出：

~~~text
session generate/prepare returns
drain_memory_execution(session, context) succeeds
MLX synchronize（仅实际使用 parent MLX 时）
completion mailbox owner drain succeeds
session reports no model-specific outstanding work
~~~

可增加：

~~~cpp
void MemoryExecutionContext::emit_terminal_schedule_event();
~~~

该 helper 从 compiled plan 读取下一 binding，要求它恰好是
<code>TERMINAL_DRAIN.END + SAFE_POINT + GPU_DRAINED + TERMINAL</code>，然后调用同一内部 emit，不由顶层手工拼 epoch 或
flags。

### 290.3 generate 成功路径参考代码

~~~cpp
result = session->generate(request, event, cancelled);

if (memory_execution) {
    drain_memory_execution(*session, *memory_execution);

    if (memory_execution->uses_explicit_schedule()) {
        memory_execution->emit_terminal_schedule_event();
    } else {
        memory_execution->checkpoint("complete");
    }

    memory_execution->begin_draining();
    memory_execution->finish_success();
    auto report = memory_execution->take_report();
    attach_report(result, report);
}
~~~

<code>emit_terminal_schedule_event()</code> 在 Running 状态执行；<code>begin_draining()</code> 之后不再允许普通 schedule
event。这样 state machine 保持单向：Running 接收最后 event，Draining 只做 terminal zero-state validation 和 report。

### 290.4 prepare 路径

prepare 可能只加载/编译而不执行完整生成。它不能借用 generate 的 schedule revision。需要独立 operation：

~~~text
operation=prepare
schedule_template=adapter.prepare.v1
~~~

prepare schedule 至少包括 component load、optional warmup、cache publish 和 terminal drain。若当前 capability 只认证
generate，则 constrained prepare 应明确 plan-only 或 unsupported，不能用 generate schedule 伪装成功。

### 290.5 failure 路径

任何 callback、model、drain 或 terminal event 失败：

1. 保存第一 primary error；
2. 禁止发送后续 schedule event；
3. 调用 <code>finalize_memory_failure()</code>；
4. 若 disposition 为 NeedsGpuDrain，执行一次受控 drain/cleanup；
5. cleanup 失败追加 <code>cleanup_failure</code>，不覆盖 primary error；
6. take terminal report；
7. 需要 quarantine 时 session/worker 不再接新请求。

### 290.6 disabled/default 硬隔离

必须保留最外层 fast path：

~~~text
if memory_constrained.enabled == false:
    no memory probe
    no capability lookup
    no MemoryExecutionContext
    no hook struct initialization
    no schedule generation
    no schedule callback
    no swap sampling
    no evidence writer intent
    existing model/cache/progress behavior unchanged
~~~

不能通过创建一个 state=Disabled 的 context 再让每个 hot path 判断来实现；那会污染默认性能和崩溃面。

## 291. 采样 observer 的测试合同与 probe 计数

strict schedule 的 terminal checkpoint 失败说明 observer 本身也需要可审计协议。

### 291.1 observer 消费顺序

当前 <code>MemoryAdmission</code> 构造时立即消费一次 observation。之后每个 checkpoint 再消费一次。因此 synthetic fixture
必须声明：

~~~text
sample[0] admission
sample[1] event checkpoint A
sample[2] event checkpoint B
...
sample[N] terminal checkpoint
~~~

测试不应依赖“传给 context 的 baseline 会自动作为 admission observer 的唯一 sample”这一假设；baseline 参数和
admission 内 observer sample 目前是两个相关但独立的输入。

### 291.2 建议新增测试 helper

~~~cpp
class ScriptedProcessObserver {
  public:
    explicit ScriptedProcessObserver(
        std::vector<ProcessMemoryObservation> samples);

    ProcessMemoryObservation operator()();
    size_t calls() const;
    bool exhausted() const;
};
~~~

行为：

- 每次严格返回下一项；
- 超出 vector 时返回 unavailable 或抛固定错误；
- 不循环最后一个值；
- 可配置在指定 index 抛异常；
- 测试结束断言 <code>calls()==expected</code>。

swap observer 使用同一模式，但 process 和 swap 调用次数分别断言。

### 291.3 probe 次数验收

| 模式 | process probe | swap probe | 期望 |
|---|---:|---:|---|
| disabled/default | 0 | 0 | 硬门 |
| constrained preflight rejected before context | 允许 metadata probe；process/swap 按 admission 设计 | 不进入模型 |
| strict success | 1 admission + schedule checkpoints | 1 baseline + schedule checkpoints + terminal policy | 精确匹配 plan |
| compatibility success | 1 admission + progress/final checkpoints | 对应 checkpoint | 只用于非认证 |
| failure before first checkpoint | 1 admission | baseline + failure sample（若已安装） | terminal report 可读 |

observer 调用数变化必须导致 test review；不能只验证最终 peak 数值。

## 292. H3/LTX 共享 schedule hook ABI 接线方式

### 292.1 不复制 ABI 定义

H3/LTX 均直接 include <code>native/core/memory_schedule_c.h</code>，不得各自复制 enum 或定义
<code>h3_schedule_event</code>/<code>ltx_schedule_event</code>。模型 options 只保存
<code>const tc_memory_schedule_hooks_v1 *</code>。

### 292.2 H3 options

在 <code>h3_options</code> 尾部增加：

~~~c
const tc_memory_schedule_hooks_v1 *schedule_hooks;
~~~

默认宏设置为 <code>NULL</code>。<code>h3_ctx</code> 在 <code>h3_generate()</code> 调用期间复制 hook POD 值，不长期保存
caller 指针；复制前检查 struct_size/version/emit。所有子组件通过内部 options 得到同一复制值和 generation，而不是
回指 C++ context。

建议内部 helper：

~~~c
int h3_memory_schedule_emit(
    const tc_memory_schedule_hooks_v1 *hooks,
    uint32_t stage, uint32_t action,
    uint32_t step, uint32_t block, uint32_t tile,
    uint32_t branch, uint32_t slot, uint32_t flags,
    char *error, size_t error_size);
~~~

若 <code>hooks==NULL</code>，helper 直接返回 1 且不构造 event；这样 disabled path 的循环中不发生 callback，也不写
trace。为保证真正零开销，热点 block loop 应在进入 constrained 专用函数前分流，而不是每个 block 都检查 NULL。

### 292.3 LTX options

在 <code>ltx_native_options</code> 尾部增加：

~~~c
const tc_memory_schedule_hooks_v1 *schedule_hooks;
~~~

<code>ltx_native_create()</code> 校验后把 POD copy 到 <code>ltx_native_denoiser</code>，并保存
<code>schedule_enabled</code>。stage1/stage2 run 由调用它们的 owner thread 发 event；任何内部 prefetch worker、Metal
completion handler 和 helper process reader 都不能调用 emit。

### 292.4 session bridge 生命周期

<code>H3Session</code>/<code>LtxSession</code> 在 <code>bind_memory_context()</code> 时：

~~~cpp
schedule_hooks_ = context.make_schedule_hooks();
schedule_hooks_bound_ = true;
~~~

在 <code>unbind_memory_context()</code> 前必须：

~~~text
model call returned
all loader threads joined
all Metal completions posted
owner drained completion mailbox
no schedule callback is active
~~~

unbind 后把整个 POD 清零并增加 session generation。stale worker 若仍试图通过保存的 hook 调用，属于 UAF 风险；实现上
应让 worker 永远不持有 hook，测试上仍需 sanitizer/fault coverage。

### 292.5 adapter callback helper 的失败规则

模型 helper 返回 0 时，caller 立即沿已有 error cleanup 路径退出。禁止：

- 记录 warning 后继续；
- 只禁用 streaming 并切回 resident；
- 重发同一个 event；
- 更换 slot 后重试；
- 清空 error buffer；
- 在 C 层自行推进 epoch。

session 顶层将 callback error 原样提升为 primary memory failure。

## 293. H3 首发 candidate 的逐事件 schedule 与代码插入点

H3 的首发目标不是覆盖所有现有优化组合，而是冻结一个 allocation graph 可预测、数值行为不变、能够完整证明
stream/load/offload 生命周期的 candidate。

### 293.1 首发约束

建议 candidate 固定：

~~~text
backend                native Metal H3
operation              generate
dynamic gate skip      off
first-block cache      off
TeaCache               off
token reduction        off
Sol/sparse             off
ANE/Core ML            off
LoRA                   off，或独立 identity
refill slots           1 首发；2 为独立 candidate
step count             固定 bucket
resolution/frames      固定 bucket
audio branch           固定 on/off；分别认证
VAE mode               固定 tile/chunk 配置
~~~

任何环境变量、profile 或 request 使上述分支变化时，capability lookup 必须 miss，不能“尽量沿用”同一 schedule。

### 293.2 顶层阶段事件

推荐顺序：

| sequence group | stage/action | owner | flags | 说明 |
|---|---|---|---|---|
| H0 | ADMISSION.END | request wrapper | SAFE_POINT | context 已构造，未进入模型 allocation |
| H1 | TEXT.BEGIN | <code>h3.c</code> owner | 无 | 开始 text encoder |
| H2 | TEXT.END | <code>h3.c</code> owner | SAFE_POINT | text 输出已固定，临时可释放 |
| H3 | CONDITIONING_HANDOFF.END | owner | SAFE_POINT | conditioning 已转入 DiT 所需布局 |
| H4 | DENOISER_LOAD.BEGIN | owner | PREFETCH_ALLOWED | resident prefix/refill slot 准备 |
| H5 | DENOISER_LOAD.END | owner | SAFE_POINT | 初始可运行集合 ready |
| H6.. | DENOISER.* | <code>h3_dit.c</code> owner | 见下表 | step/block schedule |
| HN | LATENT_HANDOFF.END | owner | SAFE_POINT + GPU_DRAINED | DiT completion 已 drain |
| HN+1 | VIDEO_VAE.BEGIN/END | owner | END 为 SAFE_POINT | video decode |
| optional | AUDIO_VAE.BEGIN/END | owner | END 为 SAFE_POINT | audio decode |
| HN+2 | EXPORT.BEGIN/END | owner | END 为 SAFE_POINT | host/output encode |
| final | TERMINAL_DRAIN.END | request wrapper | SAFE_POINT+GPU_DRAINED+TERMINAL | 统一顶层发出 |

如果 H3 的 text/conditioning 已在 session 外缓存，不能静默省略 H1–H3。应使用独立
<code>conditioning_source=cache</code> schedule template，事件明确表示 cache acquire/validation；candidate identity 绑定
cache key schema。

### 293.3 DiT block 事件

对每个 global denoise step <code>s</code> 和 block <code>b</code>：

~~~text
DENOISER.PREFETCH(step=s, block=b, slot=k)
  owner 在启动异步 I/O 前发
  flags=PREFETCH_ALLOWED

DENOISER.UPLOAD(step=s, block=b, slot=k)
  owner join/观察 refill 完成后、首次 GPU 使用前发

DENOISER.COMPUTE(step=s, block=b, slot=k)
  command encode/submit 边界前发

DENOISER.LAST_USE(step=s, block=b, slot=k)
  最后一个引用该 logical instance 的 command 已提交后发

DENOISER.RETIRE(step=s, block=b, slot=k)
  token 已绑定 command completion；backing 仍不可复用

DENOISER.DRAIN(step=s, block=b, slot=k)
  仅在 plan 指定的 owner safe point 发；先 drain mailbox，再进入 release epoch
~~~

不是每个 block 都必须有 DRAIN；为了 overlap，通常在复用同一 slot 前或 command batch barrier 发。schedule compiler
根据 slot 数生成确定的 drain placement。例如单 slot：block b+1 的 UPLOAD 前必须完成 b；双 slot：b+2 复用 slot 前
完成 b。

### 293.4 global step 与重复权重

若权重在多个 denoise step 间保持 resident，不能为每一步重复声明 weight acquire/release。事件仍可每步 COMPUTE，但
instance coverage 表只把 resident prefix 的 weight lifetime 绑定到 DENOISER_LOAD 和 LATENT_HANDOFF。streamed block
若每步从存储重新 refill，则 instance id 必须包含 step：

~~~text
instance_id = checked(step * block_count + block)
slot = instance_id % refill_slots
~~~

若实现实际在 step 间保留某些 streamed block，应成为另一个 residency plan/candidate；不能让 runtime 临时决定而 plan
仍按每步释放计算。

### 293.5 H3 代码插入点

| 文件/函数区域 | 修改 |
|---|---|
| <code>native/models/h3_runtime/h3.h</code> | options 尾部增加 schedule hooks |
| <code>native/models/h3_runtime/h3_memory.h</code> | 内部 schedule options/helper declaration |
| <code>native/models/h3_runtime/h3.c</code> | text、conditioning、DiT、VAE、export 顶层 event；错误向上传播 |
| <code>native/models/h3_runtime/h3_text_encoder.c</code> | text BEGIN/END 内部边界或由顶层包围，二选一且不能重复 |
| <code>native/models/h3_runtime/h3_dit.c</code> | constrained 专用 step/block loop 事件；不改 resident/default loop |
| <code>native/models/h3_runtime/h3_dit_schedule.c</code> | metadata-only event/template expansion，不调用 runtime hook |
| <code>native/models/h3_runtime/h3_video_vae.c</code> | decode tile event 与 host/GPU instance coverage |
| <code>native/models/h3_runtime/h3_taeh3.c</code> | 若首发使用 TAEH3，则独立 schedule revision |
| <code>native/platform/apple/h3_session.mm</code> | bind hook、candidate key、drain、telemetry |

### 293.6 H3 schedule cardinality

compiler 必须能预先计算 event 数。若每个 streamed block使用五个事件、每 <code>D</code> 个 block 一个 DRAIN：

~~~text
dit_events = steps * streamed_blocks * 5
           + steps * ceil(streamed_blocks / D)
           + resident_stage_events
~~~

实际公式应由 template generator 输出并写入 schedule artifact。runtime accepted count 必须精确相等，不能允许
“至少这些事件”。

### 293.7 H3 synthetic 验收

使用 fake block reader、fake Metal submit/completion：

~~~text
[ ] 全 schedule 完成，automatic epoch=0
[ ] 单 slot 不在 completion 前 reuse
[ ] 双 slot 能出现 read(n+1) 与 compute(n) overlap
[ ] short read poison slot 和 cursor
[ ] block index 错误在 allocation 前失败
[ ] callback 非 owner thread 失败
[ ] disabled H3 不调用 schedule helper
[ ] resident H3 default loop 源码/trace 行为不变
[ ] final event 由 request wrapper 发，不由 h3.c 伪造 GPU_DRAINED
~~~

## 294. LTX 首发 candidate 的双阶段 schedule 与 helper 边界

LTX 比 H3 多出 stage1/stage2 latent handoff、可选 audio、dynamic Gemma、MLX VAE/helper process 和多组 ANE/MLX
配置。首版必须进一步收窄。

### 294.1 首发约束

~~~text
denoiser              native Metal streaming
stage count           exactly 2
dynamic Gemma         固定 on 或 off；独立 candidate
audio                 video-only 首发建议；A/V 为后续独立 candidate
ANE MLP/QKV/V2A       off
Sol/sparse            off
video_attention_batch 固定值
batch_audio_commands  固定值
refill slots          1 首发；2 独立 candidate
stage2 upsampler      固定实现和 checkpoint
video VAE             isolated helper 或 in-process 固定一种
graph/cache policy    固定
~~~

### 294.2 stage1/stage2 坐标编码

ABI v1 没有单独的 <code>pass</code> 字段。首版桥接方案使用 global step：

~~~text
stage1 global_step = local_step
stage2 global_step = stage1_step_count + local_step
~~~

schedule artifact 必须同时保存：

~~~json
{
  "step_coordinate": "global_across_denoiser_passes",
  "stage1_begin": 0,
  "stage1_count": 20,
  "stage2_begin": 20,
  "stage2_count": 20
}
~~~

不允许使用 tile、branch 或 slot 暗中编码 stage。未来 ABI v2 可以增加 <code>pass</code>，但 v1 identity 必须完整绑定
global-step convention。

### 294.3 LTX 顶层事件

~~~text
ADMISSION.END
optional VIDEO_VAE.BEGIN/END for image-to-video first-frame encode
TEXT.BEGIN/END
CONDITIONING_HANDOFF.END
DENOISER_LOAD.BEGIN/END
DENOISER stage1 global steps/blocks
LATENT_HANDOFF.BEGIN
  native denormalize
  stage2 upsample load/materialize/compute
  renormalize
LATENT_HANDOFF.END SAFE_POINT+GPU_DRAINED
DENOISER stage2 global steps/blocks
LATENT_HANDOFF.END SAFE_POINT+GPU_DRAINED for decoder handoff
VIDEO_VAE.BEGIN/END
optional AUDIO_VAE.BEGIN/END
EXPORT.BEGIN/END
TERMINAL_DRAIN.END by request wrapper
~~~

两个 latent handoff 不能用完全相同 key。建议第一个使用 <code>tile=0</code> 表示 stage1→stage2，第二个
<code>tile=1</code> 表示 denoiser→decoder，并在 schedule schema 明确其语义；或者在 ABI v2 增加 segment。若不接受
coordinate bridge，应在 production 认证前升级 ABI，而不是让 duplicate check 放宽。

### 294.4 video/audio 并行

若 <code>parallel_av</code> 启用，video/audio worker 不直接发 event。owner 使用 barrier event：

~~~text
DENOISER.COMPUTE(branch=VIDEO, step=s, block=b)
DENOISER.COMPUTE(branch=AUDIO, step=s, block=b)
DENOISER.DRAIN(branch=COMMON, step=s, block=b, SAFE_POINT)
~~~

实际 worker 完成消息写入有界 mailbox，owner 按固定 video→audio 或 schedule 声明的顺序消费。不能让 OS 调度顺序决定
cursor。video-only candidate 的 schedule 中不得出现任何 AUDIO branch；runtime 若产生即 fail-closed。

### 294.5 Gemma encoder

<code>ltx_gemma_encoder</code> 的 memory hooks 已有接线基础，schedule 增加：

~~~text
TEXT.BEGIN
TEXT.PREFETCH per weight group（若真正 streaming）
TEXT.COMPUTE per layer/group
TEXT.LAST_USE/RETIRE
TEXT.END SAFE_POINT+GPU_DRAINED
~~~

若当前 Gemma 路径只是权重流式读取但没有稳定逐层 schedule identity，则首版把它视为独立 opaque component envelope：
TEXT.BEGIN/END 两个 event，中间用 certified framework upper 包住；不能伪造逐 layer event。

### 294.6 VAE helper process

isolated helper 时主进程 ledger 需要显式 IPC envelope：

~~~text
helper process baseline upper
input shared/file-backed payload upper
helper peak upper
output payload upper
simultaneous parent+helper overlap upper
~~~

主进程 schedule：

~~~text
VIDEO_VAE.BEGIN
helper spawn/admission
input publish
helper execution
output receive
helper exit verified
VIDEO_VAE.END SAFE_POINT
~~~

helper 的 swapout/peak evidence 单独采样并并入 root artifact。若无法获得 helper process footprint 或退出后仍有 orphan，
candidate 保持 plan-only。

### 294.7 LTX 文件级修改

| 文件 | 修改 |
|---|---|
| <code>native/models/ltx_runtime/ltx_native.h</code> | options 尾部 schedule hooks；文档化 global step |
| <code>native/models/ltx_runtime/ltx_blocks.c</code> | constrained block loop event 与 slot 状态 |
| <code>native/models/ltx_runtime/ltx_gemma_encoder.h/.m</code> | TEXT envelope/事件；callback 失败传播 |
| <code>native/models/ltx_runtime/ltx_gpu.h/.m</code> | 继续只负责 allocation/completion，不推进 schedule |
| <code>native/platform/apple/ltx_session.mm</code> | 两 stage、handoff、VAE/helper 顶层 event；hook 生命周期 |
| helper executable/service | helper evidence 和 bounded IPC，不接 parent context pointer |

### 294.8 LTX synthetic 验收

~~~text
[ ] stage1 与 stage2 global step 无重叠/空洞
[ ] handoff 两个 semantic key 可区分
[ ] stage2 allocation 不会在 handoff END 前发生
[ ] video-only 不发 audio event
[ ] parallel A/V 的 worker 完成顺序不改变 cursor 顺序
[ ] helper failure 保留 primary/cleanup 双错误
[ ] helper exit 后 parent 才允许 decoder release epoch
[ ] automatic epoch=0
[ ] disabled LTX hook/callback/probe count=0
~~~

## 295. Flux 与 Z-Image 的 MLX component-staged 路线

Flux/Z-Image 当前更适合先做 component staging，而不是声称逐 block Metal allocation 可控。MLX allocator、graph cache、lazy
materialization 和 unified-memory residency 使单个 C++ tensor 的逻辑析构不能证明物理内存立即回落。

### 295.1 首发仅 plan-only 的代码范围

Flux：

~~~text
native/models/flux2/flux_text.cpp
native/models/flux2/flux_encode.cpp
native/models/flux2/flux_transformer.cpp
native/models/flux2/flux_vae.cpp
native/models/flux_module.cpp
~~~

Z-Image：

~~~text
native/models/z_image/z_image.cpp
native/models/z_image/z_image.hpp
native/models/z_image_module.cpp
native/models/z_image_gguf_module.cpp
~~~

先新增 metadata-only component inventory、stage schedule 和 measured opaque envelope，不改变默认 load/cache 路径。

### 295.2 component 生命周期

推荐首版：

~~~text
text encoder load -> encode -> materialize conditioning -> unload/clear -> grace checkpoints
transformer load -> denoise -> materialize latent -> unload/clear -> grace checkpoints
VAE load -> decode -> materialize host output -> unload/clear -> grace checkpoints
export -> terminal
~~~

每个 component 必须有显式 <code>load()</code>、<code>execute()</code>、<code>release()</code> 边界。若当前类把所有组件在
constructor 一次性加载，应新增 constrained 专用 orchestrator，而不是修改默认 constructor。

### 295.3 opaque envelope

每个 component profile 保存：

~~~text
component_id
checkpoint_digest
shape bucket
dtype/quantization
MLX/framework revision
graph cache mode
before_load baseline upper
peak_delta upper
after_release steady upper
release_grace_checkpoint_count
unattributed upper
sample count/device/platform class
~~~

运行时 ledger 只 reserve 一个 framework envelope token；process observer 证明总 footprint 没有超过 envelope。不能把
MLX 内部不可见分配分类为 0 bytes。

### 295.4 release 证明

固定采样：

~~~text
before_load
after_weights_materialized
peak_compute
after_output_materialized
after_component_destroy
after_cache_clear_requested
after_grace_checkpoint_1
after_grace_checkpoint_2
~~~

cache clear 不是 release 证明；只有 process footprint 在允许的 next-stage baseline 内稳定回落才算。若不回落：

1. 提高同进程 envelope，并重新判断是否仍 fit；
2. 把 component 移到 disposable helper process；
3. 若仍不能证明，保持 plan-only。

禁止通过 sleep 到系统恰好压缩/换出后宣称 offload 成功。

### 295.5 Flux/Z-Image 可执行升级条件

~~~text
component inventory closed
all checkpoint/model identities pinned
framework envelope L2/L3 certified
release/grace behavior deterministic
swapouts_delta == 0
process peak <= B
helper lifecycle closed（若使用）
output parity pass
disabled ABBA pass
~~~

在达到上述条件前，UI 可以展示估算计划，但 <code>execution_supported=false</code>，错误明确指出缺少哪个 envelope/campaign，
不能自动退回 resident constrained execution。

## 296. 预算计算、候选枚举与 overlap 控制的可执行算法

### 296.1 Y、X、B、S 的固定含义

用户输入：

~~~text
Y = 用户声明的进程内存上限 bytes
X = buffer 百分比，0 < X < 100
B = floor(Y * (100 - X) / 100)
S = 系统可用内存保留 bytes
~~~

实现必须使用 checked arithmetic，建议避免先乘溢出：

~~~cpp
uint64_t effective_budget(uint64_t y, uint32_t x) {
    require(y > 0, "memory_policy_invalid: limit is zero");
    require(x > 0 && x < 100,
            "memory_policy_invalid: buffer percent must be in (0,100)");
    const uint64_t whole = y / 100;
    const uint64_t rem = y % 100;
    return whole * (100 - x) + rem * (100 - x) / 100;
}
~~~

B 是 admission/observed hard ceiling，不是“期望值”。Y-B 的 buffer 不允许 planner 再分配；它用于 framework jitter、测量
误差和 OS 风险。swap 或 memory compression 不计入 B，也不能扩大 B。

### 296.2 候选枚举

planner 从最少 streaming 到最保守候选枚举：

~~~text
C0 resident/default-like（仅当完整峰值 <= B）
C1 resident prefix + 2 refill slots
C2 resident prefix + 1 refill slot
C3 zero/minimum prefix + 1 refill slot
C4 component staged / helper isolated
unsupported
~~~

每个 candidate 的 allocation graph、schedule、manifest 和 identity 预编译。runtime 只能选择已认证 candidate，不能在中途把
2 slots 改成 1 slot 或缩小 prefix；那是另一个 plan。

候选排序目标：

~~~text
minimize predicted wall time
subject to planned_peak <= B
           required headroom >= next allocation
           certified on exact identity
           quality contract unchanged
~~~

若多个候选预测差异低于噪声阈值，选择更少 slots/更低峰值的候选。

### 296.3 resident prefix 搜索

对于 block 大小不完全相同的模型，不用平均 block bytes。预计算 prefix sum：

~~~text
resident_bytes(p) = sum(block_upper[0..p-1])
stream_peak(p,k) = max over epochs(
    baseline + framework + activation + resident_bytes(p)
    + live refill slots(k) + staging + output)
~~~

对每个认证 slot 数 k，从最大 p 向下寻找第一个 fit。若 block 顺序允许非前缀 pinning，必须用独立 candidate 和 source
inventory；首版只允许前缀，减少组合空间和随机 I/O。

### 296.4 overlap window

runtime 不通过百分比猜测 lookahead，而由 plan 给出下一允许 prefetch event。开始 prefetch 前同时满足：

~~~text
cursor at PREFETCH_ALLOWED binding
slot state == Empty or Reusable
ledger reserve succeeds
observed headroom >= requested_upper + framework_guard
pressure state == Normal
no cancellation/failure/poison
I/O queue depth < plan.max_io_inflight
~~~

Tight 状态停止启动 optional lookahead，但允许已经 admitted 的 refill 完成；Critical 状态停止新 block，并在最近 safe point
fail-closed。不能取消已经提交给 GPU 的 backing 后立即复用。

### 296.5 双缓冲时间模型

对 block n：

~~~text
T_block = max(T_read(n+1), T_upload(n+1), T_compute(n))
          + T_barrier_if_slot_reuse
~~~

只有满足以下证据才开启双 slot：

- trace 显示 read/upload 与 compute 真正重叠；
- 峰值增加在 B 内；
- wait time 显著下降；
- 没有增加 swap/compression 依赖；
- completion/mailbox 不溢出；
- 真实 wall time 改善超过 campaign 噪声阈值。

否则单 slot 更适合低内存机器。

### 296.6 不支持 mid-request replan

压力上升时只允许：

~~~text
停止 optional prefetch
缩短已经在同一 plan 中允许的 lookahead
等待/drain
在 safe point 失败
~~~

不允许更换 schedule revision、释放 plan 声明为 resident 的权重、切换 helper mode 或降低 shape/quality。需要降级时，终止
当前 request，由上层以新 request/candidate 重试，并明确告知用户。

## 297. 内存估算、校准与 manifest 生成流水线

可靠的低内存模式不是运行时看见 OOM 再卸载，而是在请求开始前有一个可审计的上界。估算应分成静态可证明部分和框架
测量部分，二者不能混成一个“经验系数”。

### 297.1 静态 inventory

工具从 checkpoint、模型配置和 adapter source 生成：

~~~text
weight tensors: name, dtype, shape, packed bytes, alignment
resident prefix candidates
per-block streamed payload
quantization scales/metadata
constant buffers
activation formulas by shape bucket
conditioning/latent/output formulas
refill slot/staging formulas
helper IPC payload formulas
allocation site and instance mapping
~~~

每一项保存 source provenance：checkpoint file digest、tensor offset/length、生成工具 revision、adapter source inventory digest。
遇到 unknown dtype、dynamic shape、duplicate tensor 或 overflow 立即失败。

### 297.2 物理 backing 与逻辑 tensor 分离

manifest 需要区分：

~~~text
logical tensor bytes
allocator requested bytes
allocator allocated/backing bytes
file mapping residency envelope
alias group
framework unattributed envelope
~~~

预算使用真实 backing upper，不使用 tensor 元素数直接相加。Metal buffer alignment、heap granularity、shared storage、mmap
page residency和 allocator cache 都需要独立处理。

### 297.3 activation 公式

每个 site 使用 checked formula，例如：

~~~text
tokens = checked(frames * latent_height * latent_width)
qkv = checked(batch * tokens * heads * head_dim * dtype_bytes * 3)
attention_workspace = implementation_specific_formula(...)
vae_tile = checked(tile_frames * tile_height * tile_width * channels * bytes)
~~~

公式 version 进入 manifest identity。不能只保存某次 shape 的数字而失去重算依据。bucket 内若允许多个 shape，上界由 bucket
最大点或证明过的单调公式产生；非单调 kernel workspace 必须逐点校准或拆 bucket。

### 297.4 framework envelope 校准

每个 candidate/device/platform/build 运行 cold/warm 重复 campaign，记录：

~~~text
observed process peak
ledger known peak
unattributed peak = max(0, process - known)
minimum system available
allocator/cache steady state before and after
compression delta
swapins/swapouts delta
~~~

建议 upper：

~~~text
framework_upper = max_observed_unattributed
                + deterministic_alignment_guard
                + reviewed_jitter_guard
~~~

不是用均值、P95 或“多数不超”作为 hard upper。认证 campaign 中任何超过既有 upper 的点都使 record 失效并需要新
revision。jitter guard 的来源和数值进入 framework profile digest。

### 297.5 估算误差闭环

每次 L2/L3 run 生成：

~~~text
planned peak vs observed peak
per-site upper vs actual backing
per-instance lifetime vs trace
framework upper vs unattributed
post-stage expected baseline vs observed
~~~

分类误差：

| 类型 | 处理 |
|---|---|
| actual site bytes > upper | code/manifest bug，fail-closed |
| lifetime 超出 | schedule/adapter bug，fail-closed |
| unknown site | closure bug，fail-closed |
| unattributed > envelope | framework profile 失效 |
| process > B 但各项未超 | alias/double-count/probe 模型需审计 |
| planned 远高于 observed | 可后续优化，但不能在线自动缩上界 |

校准工具只能产生候选 artifact，不能自动修改 production registry。reviewer 审核 source、quality、L2/L3 后显式加入 record。

### 297.6 建议工具

~~~text
tools/memory/scan_checkpoint.py
tools/memory/generate_site_inventory.py
tools/memory/compile_schedule.py
tools/memory/compile_manifest.py
tools/memory/verify_required_site_closure.py
tools/memory/run_campaign.py
tools/memory/verify_evidence.py
tools/memory/diff_candidate.py
~~~

所有工具支持 <code>--check</code> 模式，CI 只验证生成文件未漂移；本地显式 <code>--write</code> 才更新 artifact。

## 298. 并发、session cache 与服务级 admission

本设计首版按单 worker 单 constrained request 认证。即使单 request 自身不超 B，多个 request 或 resident session cache
叠加仍可能触发系统 swap，因此 service 需要外层 broker。

### 298.1 首版并发规则

~~~text
每个 worker 同时最多 1 个 constrained execution context
constrained 与 default request 不在同一 worker 并发
prepare 与 generate 不并发
helper process 计入同一 request budget/campaign
quarantined worker 不再 admission
~~~

如果服务已有队列，constrained request 在 worker 获取前只做轻量 schema validation；真实 process baseline 和 capability
preflight 在目标 worker 内执行，避免用调度进程的 footprint 代替执行进程。

### 298.2 resident cache

constrained admission 前列出所有已有 cache：

~~~text
model/session resident weights
MLX graph/tensor cache
Metal pipeline cache
Core ML/ANE objects（首版应禁用）
conditioning cache
decoded/output cache
service-owned buffers
~~~

若这些 cache 是 candidate 的合法 baseline，必须进入 capability identity 和 baseline upper。若不是，request 在创建 context
前选择干净 worker；不能临时清 cache 后继续复用一个未认证 baseline。

默认模式不得因为 constrained feature 存在而主动清 cache。

### 298.3 未来多请求 broker

后续若支持并发，broker 需要 reservation：

~~~text
worker_budget = min(configured worker cap, system policy cap)
request_claim = certified planned process increment
shared_baseline = measured/pinned worker baseline
sum(active claims) + shared_baseline <= worker_budget
system_available - sum(new claims) >= global reserve
~~~

这只能作为 admission 上层约束；每个 request 内仍有自己的 ledger/schedule/watchdog。不能把一个 request 未使用的临时
headroom在运行中借给另一个 request，因为这会破坏各自的认证峰值与失败隔离。

### 298.4 cancellation

取消只在 owner safe point 转为 runtime action：

1. user callback 设置 atomic cancel flag；
2. owner 在 schedule event 前检查；
3. 停止新 prefetch/allocation；
4. join I/O；
5. 已 submit GPU work 走 retire/complete/drain；
6. finalize failure，错误类保持 cancelled，但 cleanup 事实写入 memory report；
7. 不发送 TERMINAL success event。

progress callback 线程不得直接释放 backing 或 unbind context。

## 299. slot/pager 的状态机、锁边界与 I/O 实现

### 299.1 slot 状态

~~~text
Empty
  -> Reserved
  -> Filling
  -> Ready
  -> Uploading（若需要独立 upload）
  -> Submitted
  -> Retired
  -> Reusable
  -> Filling for next instance

任意非 Submitted 状态可因本地失败进入 Poisoned
Submitted/Retired 的失败必须等待 completion 后才能决定 Released/Quarantine
~~~

slot 保存：

~~~text
slot_id
logical site/instance id
generation/domain
reservation token
host backing identity/bytes
GPU backing identity/bytes
file offset/expected length
I/O result and checksum status
completion token/sequence
first failure
~~~

### 299.2 锁边界

ledger mutex 内只做计数和 token state transition。以下操作不得持 ledger/slot global mutex：

~~~text
pread/read/mmap page touch
Metal newBuffer/heap allocation
memcpy/upload
command encode/commit/wait
user progress callback
schedule callback
process/swap probe
trace/evidence file write
thread join
~~~

slot 自身可用短 mutex/condition variable保护 I/O result，但 schedule owner 等待时不得同时持 ledger lock。

### 299.3 direct I/O 与 mmap

低内存模式优先使用明确范围的 <code>pread</code> 到已 reserve staging/slot。若使用 mmap：

- mapping virtual size 不等于 resident bytes，仍需 page residency envelope；
- 不允许整个大 checkpoint 被无意 sequential fault-in；
- <code>madvise</code>/<code>fcntl</code> 只是 hint，不是 release 证明；
- file descriptor 和 mapping lifetime 进入 plan；
- short read、offset overflow、file identity change 均 fail-closed。

### 299.4 refill 参考伪代码

~~~cpp
bool refill_block(Execution &exec, Slot &slot, BlockId block) {
    exec.emit(prefetch_event(block, slot.id));
    auto reservation = exec.reserve_instance(block.site, block.instance,
                                             block.upper_bytes);
    if (!reservation) return false;

    slot.begin_fill(std::move(reservation), block);
    io_queue.submit([&slot] { slot.read_exact_and_verify(); });

    slot.join_fill();
    if (!slot.ok()) return exec.fail(slot.failure());

    exec.emit(upload_event(block, slot.id));
    slot.allocate_and_commit_gpu();

    exec.emit(compute_event(block, slot.id));
    auto command = encode_block(slot);
    auto completion = exec.retire_on_submit(slot.token(), command);
    command.commit();

    exec.emit(last_use_event(block, slot.id));
    slot.mark_retired(completion);
    return true;
}
~~~

实际 <code>retire</code> 与 command commit 的先后要与 Metal API 的失败可观察性严格匹配：必须避免 command commit 失败但
ledger 认为一定会收到 completion。该细节需要 L2 fault injection；无法证明 clean 时 disposition 为 quarantine。

### 299.5 slot reuse

复用前：

~~~text
owner reaches plan drain binding
completion mailbox consumed matching token
completion status success
ledger pending release becomes zero for instance
slot generation still current
slot state becomes Reusable
new reserve succeeds
~~~

仅 command buffer 已 commit、scheduled 或 CPU encode 返回都不够。禁止用固定 sleep 或 fence timeout 后假设完成。

## 300. 失败分类、cleanup 和 quarantine 决策表

### 300.1 失败分类

| 失败点 | primary class | 可继续新工作 | cleanup 目标 | 默认 disposition |
|---|---|---:|---|---|
| preflight/capability miss | unsupported/estimate | 否 | 无 runtime state | Clean |
| reserve denied | budget/pressure | 否 | cancel reservations | Clean |
| allocation 返回 null | allocation | 否 | cancel token | Clean |
| actual > upper | lifetime/estimate | 否 | release；审计 allocator | Quarantine unless proven clean |
| schedule key/flag mismatch | lifetime | 否 | stop/join/drain | NeedsGpuDrain 或 Clean |
| process probe unavailable | observation | 否 | stop/join/drain | Quarantine for certified path |
| swapout delta > 0 | swap activity | 否 | stop/join/drain | Quarantine/worker recycle |
| I/O short read/checksum | checkpoint I/O | 否 | poison slot/release unsubmitted | Clean |
| command encode fail | GPU encode | 否 | release unsubmitted | Clean if no submission |
| command commit uncertain | GPU submit | 否 | drain if possible | Quarantine |
| completion error/stale | GPU completion | 否 | drain/report | Quarantine |
| mailbox overflow | runtime | 否 | full drain | Quarantine |
| helper crash/orphan | helper | 否 | terminate/reap | Quarantine worker/request topology |
| evidence writer fail | evidence | runtime 已结束 | 保留 primary result | 不改变 runtime disposition |

### 300.2 primary error 不可覆盖

保存：

~~~text
primary_failure
cleanup_failure
evidence_failure
~~~

响应的主错误码来自 primary_failure。cleanup/evidence 只增加诊断字段。若 cleanup 发现更严重的 stale completion，需要提升
failure disposition，但不改写最初事件和消息。

### 300.3 cleanup 终态

Clean 必须证明：

~~~text
reservation_count == 0
pending_release_count == 0
outstanding_completion_count == 0
mailbox_count == 0
storage/site allocation count == 0
all loader/helper threads joined
hooks detached
no stale generation callback
~~~

NeedsGpuDrain 表示还有可识别 submitted/pending state，允许顶层做一次 drain。Quarantine 表示无法证明 backing/completion/helper
状态，worker 不得再次 admission。禁止通过清空计数器把 Quarantine 变成 Clean。

### 300.4 destructor

<code>MemoryExecutionContext</code> destructor 只做 best-effort 标记，不应在异常展开期间执行长时间 GPU wait 或抛异常。
正确 cleanup 必须由显式 failure path 完成。destructor 发现未 finished 时记录 programing error 并使 session/worker
quarantine；terminal report 应在对象销毁前冻结。

## 301. 结果协议、trace 与 evidence 的新增字段

### 301.1 terminal report

在现有 admission metrics 基础上补：

~~~text
policy_schema/revision
candidate_key_digest
manifest_digest
schedule_schema/revision/coverage_digest
plan_digest
schedule_cursor_state
schedule attempted/accepted/rejected/expected
schedule next/last sequence
schedule first failure
explicit/automatic epoch transitions
current/final/peak epoch
slot count/refill/reuse/wait metrics
I/O bytes/time/short-read count
GPU submitted/completed/failed count
mailbox high-water/overflow/stale count
process observation count
swap observation count/delta/source
planned/observed peak and margin
framework upper/observed unattributed
closure status/unknown site count
failure disposition/cleanup/evidence status
~~~

所有 digest/identity 字段在 terminal 时从 immutable plan/capability snapshot 复制，不从日志重建。

### 301.2 schedule trace event

建议字段：

~~~text
monotonic event_index
relative timestamp_ns
sequence
stage/action/step/block/tile/branch/slot/flags
target epoch
cursor state before/after
accepted/rejected
ledger committed/reserved/pending/cached
process footprint/system available（仅采样时）
completion mailbox consumed count
pressure state
error class
~~~

timestamp 仅用于性能分析，不参与正确性排序；排序以 event_index、sequence、token sequence 为准。

### 301.3 JSON 兼容

新增字段只追加，旧字段保持语义。schema version 升级时：

- consumer 遇到 unknown field 忽略；
- verifier 对认证 artifact 要求 exact supported schema；
- bool 不用 0/1 string；
- bytes 全部为 unsigned integer；
- enum 输出稳定 lowercase name，同时可保留 numeric code；
- 缺字段与 0 值不同，不能用默认 0 伪装未采集。

### 301.4 evidence 最小文件集

~~~text
normalized-request.json
device.json
candidate-key.json
source-inventory.json
manifest.json
schedule.json
schedule-coverage.json
plan.json
framework-profile.json
closure.json
terminal-report.json
cleanup.json
trace.jsonl
quality.json
performance.json
digest.json
~~~

L2/L3 campaign 再有 index 将多次 request artifact 关联，但不复制大文件。所有路径使用 request-local opaque id，不记录
用户名、绝对 checkpoint path 或 pointer。

## 302. 测试矩阵与本机/Metal/低内存验收命令

### 302.1 L0：无 Metal 逻辑测试

~~~text
python3 -B tests/native/test_memory_accounting.py
python3 -B tests/native/test_memory_manifest.py
python3 -B tests/native/test_memory_schedule.py
python3 -B tests/native/test_memory_plan_compiler.py
python3 -B tests/native/test_memory_scheduler.py
python3 -B tests/native/test_memory_watchdog.py
python3 -B tests/native/test_memory_trace.py
python3 -B tests/native/test_memory_execution.py
python3 -B tests/native/test_contract.py
~~~

必须新增/调整：

~~~text
tests/native/memory_execution_test.cpp
  - deterministic process observer for every checkpoint
  - observer exact call count
  - poisoned/reentrant/non-owner tests
  - terminal helper success/failure
  - explicit schedule skips compatibility checkpoints

tests/native/memory_plan_compiler_test.cpp
  - schedule requires explicit epoch
  - shape/slot/coverage closure
  - digest mutation for every bound field

tests/native/test_contract.py
  - memory_schedule_c.h exposed to model adapters
  - request progress checkpoint guarded by !uses_explicit_schedule()
  - terminal event helper used in generate/prepare
  - result JSON has new cursor fields
  - production registry remains empty
~~~

L0 success 只能证明 deterministic logic 和源码合同，不证明 Metal resource lifetime。

### 302.2 L1：adapter synthetic

H3/LTX fake runtime 跑完整事件表，覆盖：

~~~text
exact success
every event omission one at a time
duplicate event
wrong step/block/tile/branch/slot
wrong flags
loader completion reorder
short read
allocation fail before/after commit
cancel at every safe point
mailbox overflow/stale generation
terminal missing/extra
~~~

每个 failure 检查 accepted cursor、epoch、ledger zero-state、disposition 和 first error。

### 302.3 L2：真实 Metal

每个 candidate 至少证明：

~~~text
real allocatedLength <= instance upper
real command completion drives release
slot not reused before completion
multi-queue drain covers every queue
completion status failure path
command encode/commit injected failure
cancel during submitted work
process/swap samples available
trace has reserve->commit->submit->retire->complete->release
quality parity
~~~

当前机器无 Metal device 时测试必须显示 SKIP(no Metal)，不能记 PASS。编译通过单独记录 compile PASS。

### 302.4 L3：16/24/32 GiB 或等价机器

矩阵：

| 维度 | 点 |
|---|---|
| memory class | 16/24/32 GiB 或审核的等价设备 |
| X | 10/15/20% |
| candidate | slot1/slot2、每个 prefix |
| shape | bucket min/typical/max |
| cache | cold/warm/dirty-baseline rejection |
| path | success/cancel/fault |
| repetition | 每点至少 3；任何一次违反 hard limit 即失败 |

每个 request 硬条件：

~~~text
peak_process_footprint_bytes <= B
swapouts_delta == 0
schedule_cursor_complete == true
schedule rejected == 0
automatic_epoch_transition_count == 0
unknown/generic site == 0
actual instance bytes <= upper
framework unattributed <= upper
completion stale/overflow/error == 0
post-request zero-state == true
quality pass
evidence digest recomputes
~~~

系统在 request 前已发生 swap 不自动失败；判定使用稳定来源的 request baseline 到 terminal delta。但若 baseline 本身处于高
压力、available < S，admission 应拒绝，不能把现有 swap 当成预算。

### 302.5 disabled ABBA

~~~text
A1: feature code present, request disabled
B : constrained enabled candidate
A2: same request disabled again
~~~

A1/A2 与合并前 baseline 比较：

~~~text
probe count = 0
hook callback count = 0
schedule generation count = 0
cache clear/unload count unchanged
allocation count unchanged within exact contract
output parity
wall/GPU time within predeclared noise threshold
peak footprint无系统性上升
~~~

任何新增 probe/hook/unload 是硬失败，不用性能统计豁免。

### 302.6 当前测试阻塞清单

截至 2026-09-16 后续实施，strict execution fixture 的 deterministic observer、精确 probe count、schedule
header/hook/result/request orchestration contract 均已补齐并通过。PR-S1a/PR-R1 的共享 runtime L0 已 green；模型
semantic event 接线、instance coverage、Metal L2/L3 仍未完成，详见第 305 节。

## 303. 逐 PR 代码修改建议、依赖与完成门

### 303.1 PR-S1a：cursor hardening 与测试修复

修改：

~~~text
native/runtime/memory_execution.hpp/.cpp
native/runtime/memory_accounting.hpp
native/platform/apple/results.mm
tests/native/memory_execution_test.cpp
tests/native/test_memory_execution.py
tests/native/test_contract.py
~~~

内容：sticky poison、reentrancy guard、attempted/accepted/rejected metrics、deterministic observer、terminal helper 的 pure C++
fixture。完成门：全部 L0 通过，失败 event 不可 retry，production registry 为空。

### 303.2 PR-S1b：compiler coverage

修改：

~~~text
native/runtime/memory_schedule.hpp/.cpp
native/runtime/memory_plan.hpp/.cpp
native/runtime/memory_manifest.hpp/.cpp
tests/native/memory_schedule_test.cpp
tests/native/memory_plan_compiler_test.cpp
~~~

内容：shape/template context、slot bound、epoch/instance coverage、coverage digest。完成门：逐字段 mutation 改 digest，required
instance 缺绑定 compile fail。

### 303.3 PR-R1：request orchestration

修改：

~~~text
native/api/c_api.mm
native/platform/apple/request.mm
native/runtime/memory_execution.hpp/.cpp
tests/native/test_contract.py
新增 request orchestration fixture
~~~

内容：explicit schedule 跳过 progress/final compatibility checkpoint；top-level terminal event；generate/prepare 独立 operation；
failure 保留 primary error。完成门：disabled probe/hook 仍为 0，strict terminal drain 顺序可测试。

### 303.4 PR-H3S：H3 schedule bridge

修改第 293.5 节文件。先只支持单一冻结 candidate和 slot1；slot2 后续独立 patch。完成门：L1 full cursor、所有 fault、
automatic epoch=0、default H3 ABBA。

### 303.5 PR-LTXS：LTX schedule bridge

修改第 294.7 节文件。先 video-only、native Metal stage1/2、固定 helper mode。完成门：global step 和两个 handoff 明确、L1
full cursor、automatic epoch=0、default LTX ABBA。

### 303.6 PR-I2：explicit instance/site closure

修改：

~~~text
native/runtime/memory_site_registry.hpp/.cpp
native/runtime/memory_manifest.hpp/.cpp
native/runtime/memory_plan.hpp/.cpp
native/runtime/memory_execution.hpp/.cpp
H3/LTX allocator hook ABI v3
generated h3/ltx memory_sites.def
closure tools/tests
~~~

完成门：strict path 不接受 string-only ambiguous reserve；所有 reachable site/instance closed；generic/unknown 为 0。

### 303.7 PR-P1：pager overlap

在 schedule 和 instance closure 已 green 后再实现 slot1，然后 slot2。不得先用旧 string site 做高性能 pager，否则性能 patch 会
掩盖生命周期错误。完成门：L2 completion、fault injection、single/double slot trace 和峰值证据。

### 303.8 PR-FZ1：Flux/Z-Image plan-only inventory

只生成 component inventory、opaque envelope schema、stage plan/golden 和明确 unsupported reason。默认路径不重构。完成门：
metadata-only 无模型 allocation；无 executable record。

### 303.9 PR-E2：identity/evidence/service

在 H3/LTX L1 closure 后升级 capability v2 和 evidence writer。完成门：artifact 原子写、digest tamper、failure/quarantine artifact、
disabled 不写目录。

### 303.10 PR-CERT：每 candidate 单独认证

PR-CERT 不应混入 runtime 代码；只加入经审核的 immutable record/artifact digest。每个模型/shape/slot/device/platform 一个或一组
明确 record。真实 L2/L3、quality、performance、ABBA、rollback 全部签核后才允许 production registry 非空。

## 304. 最终实现与验收清单（可直接用于 issue/PR 模板）

### 304.1 配置与默认路径

~~~text
[ ] memory-constrained 是显式 add-on，默认关闭
[ ] Y/X/S 校验与 checked B 计算
[ ] request 开始后预算不可变
[ ] disabled 不 probe、不建 context、不装 hook、不写 evidence
[ ] disabled 不改变 cache/load/unload/progress 行为
~~~

### 304.2 preflight/plan

~~~text
[ ] exact capability identity，无 wildcard/nearest match
[ ] source/checkpoint/device/platform/framework identity pinned
[ ] manifest、schedule、coverage、plan digest 互相绑定
[ ] required instance lifetime 闭包
[ ] planned peak <= B
[ ] baseline + increment/system reserve admission 通过
[ ] unsupported 明确在 allocation 前失败
~~~

### 304.3 runtime schedule

~~~text
[ ] owner-thread only
[ ] non-reentrant
[ ] adapter 不传 epoch/sequence
[ ] exact key/slot/flags
[ ] SAFE_POINT 先 drain 再推进 epoch
[ ] checkpoint 由 schedule 唯一驱动
[ ] failure sticky poison，禁止 retry
[ ] terminal 由 request wrapper 在全 drain 后发
[ ] cursor complete 且 automatic epoch=0
~~~

### 304.4 allocation/pager

~~~text
[ ] reserve-before-allocate
[ ] commit 使用真实 backing bytes
[ ] site+instance identity explicit
[ ] slot 状态合法
[ ] I/O short read/checksum fail-closed
[ ] submit 后 release 只由 completion/mailbox 驱动
[ ] alias reuse 有 lifetime 证明
[ ] unknown/generic/implicit allocation 为 0
~~~

### 304.5 observation/pressure

~~~text
[ ] deterministic process/swap observer contract
[ ] probe source稳定、counter 单调
[ ] process peak <= B
[ ] system available reserve 满足
[ ] framework unattributed <= upper
[ ] swapouts delta == 0
[ ] Tight 只抑制 optional prefetch
[ ] Critical 在 safe point fail-closed，不 replan
~~~

### 304.6 failure/cleanup

~~~text
[ ] primary/cleanup/evidence error 分离
[ ] loader/helper thread join
[ ] completion/mailbox zero-state
[ ] hooks detach 与 generation invalidate
[ ] clean/needs-drain/quarantine 判定有证据
[ ] quarantined worker 不再接请求
[ ] destructor 不承担主要 cleanup
~~~

### 304.7 模型

~~~text
[ ] H3 首发分支冻结，schedule cardinality 精确
[ ] H3 slot1 L1/L2/L3；slot2 单独认证
[ ] LTX stage1/stage2 global step 与 handoff 明确
[ ] LTX video-only 先认证，A/V 独立
[ ] Flux/Z-Image 未取得 envelope/release 证据前保持 plan-only
[ ] 每个模型质量 parity 通过
~~~

### 304.8 测试与发布

~~~text
[ ] L0 logical 全绿
[ ] L1 synthetic adapter/fault 全绿
[ ] L2 real Metal 全绿；无设备只能 SKIP
[ ] L3 低内存矩阵全绿
[ ] disabled ABBA 全绿
[ ] evidence root digest 可重算
[ ] rollback 开关只影响 constrained admission
[ ] production registry 加 record 前独立 reviewer 签核
~~~

### 304.9 当前结论

基于 2026-09-16 当前工作树，正确方向仍然是主动的逐层 streaming/loading/offloading + admission/schedule/ledger，而不是
依赖系统 swap。swap 只能作为 request 失败信号和 L3 验收指标。共享 schedule ABI 和 compiler 已有部分实现，但 strict
execution 新用例尚未全部通过，H3/LTX 真实 semantic bridge、required-site closure、Metal completion 和低内存 campaign
仍是硬阻塞。因此现阶段可继续开发和做 L0/L1，不应加入 production capability record，也不应对外宣称任一模型已经在
用户给定 Y/X 上限下获得可发布保证。

## 305. 2026-09-16 实施记录：PR-S1a、PR-R1 与模型 hook plumbing

本节记录第 287–304 节之后实际完成的代码，不把尚未完成的 H3/LTX semantic event 或 Metal 证据算作通过。

### 305.1 strict fixture 与 observer 合同已修复

<code>tests/native/memory_execution_test.cpp</code> 的 strict context 现在显式传入稳定 synthetic
<code>ProcessMemoryObserver</code>。admission baseline 仍由 context wrapper 首次消费，terminal schedule checkpoint 消费
测试 observer；用例额外断言 checkpoint observer 只调用一次。因此测试不再回退到真实进程 footprint，也能发现未来的
隐式重复 probe。

### 305.2 cursor 单向事务已经落地

新增 <code>MemoryScheduleCursorState</code>：

~~~text
Disabled
Healthy
Processing
Poisoned
Complete
~~~

<code>emit_schedule_event()</code> 当前行为：

1. 记录 attempted；
2. 要求 Running、owner thread、Healthy；
3. 进入 Processing，拒绝递归 callback；
4. 严格校验 ABI、sequence、key、slot 和 flags；
5. SAFE_POINT 先 drain completion mailbox，GPU_DRAINED 再验证 zero pending state；
6. 由 binding 推进 explicit epoch；
7. 可选 checkpoint；
8. 只有全部成功才记录 accepted trace、推进 cursor 并回到 Healthy/Complete；
9. 任一异常调用 <code>poison_schedule()</code>，保存 first failure、taint context，并禁止 retry。

已覆盖：

~~~text
out-of-order
flag mismatch
event after poisoned
non-owner thread
reentrant callback through checkpoint observer
incomplete finish_success
unexpected terminal ordering
~~~

### 305.3 新增 runtime/result 指标

~~~text
schedule_event_attempted_count
schedule_event_count                  accepted count，保留旧字段
schedule_event_rejected_count
schedule_expected_event_count
schedule_mismatch_count               兼容字段
schedule_next_sequence
schedule_cursor_state
schedule_first_failure
schedule_cursor_complete
~~~

<code>native/platform/apple/results.mm</code> 已输出对应 JSON。旧 consumer 可忽略新增字段；认证 verifier 后续应要求 exact
schema。

### 305.4 terminal event 与 request orchestration

新增 <code>MemoryExecutionContext::emit_terminal_schedule_event()</code>。它只从 compiled plan 的下一 binding 构造 event，
要求该 binding 为 <code>TERMINAL_DRAIN.END</code> 且具有 SAFE_POINT/GPU_DRAINED/TERMINAL；request wrapper 不传 epoch
或自行选择 flags。

<code>native/api/c_api.mm</code> 的 generate/prepare 现在分流：

~~~text
compatibility constrained plan:
  progress boundary checkpoint
  begin_draining
  backend drain/unload/mailbox drain
  complete/prepare_complete checkpoint
  finish_success

explicit schedule plan:
  progress callback 不做通用 memory checkpoint
  backend drain/unload/mailbox drain（context 仍为 Running）
  emit_terminal_schedule_event
  begin_draining
  finish_success
~~~

disabled/default request 的 <code>memory_execution</code> 仍为 null，不新增 probe、hook、unload 或 trace。

### 305.5 plan/capability schedule 传递

<code>MemoryCapabilityRecord</code> 新增：

~~~cpp
bool require_explicit_schedule = false;
std::vector<MemoryScheduleBindingSpec> schedule;
~~~

registry bridge validation：

- require schedule 必须同时 require explicit epoch；
- require schedule 但 binding 为空拒绝；
- 有 binding 但 require flag 为 false 拒绝。

<code>authorize_memory_capability()</code> 使用完整 <code>MemoryPlanCompileOptions</code> 把 record schedule 交给 plan compiler。
synthetic certified fixture 已通过该真实 authorization 路径生成 strict plan，不再直接绕过 capability 手工 compile。

这是 bridge schema，不是 capability identity v2。schedule revision 目前由 compiled plan digest 绑定，但 candidate key 尚未
直接包含 schedule/framework/helper/platform identity，因此仍禁止 production record。

### 305.6 compatibility hook 保持 inert

没有 compiled schedule 的 context 调用 <code>make_schedule_hooks()</code> 时只返回 struct_size/version，
<code>user==NULL</code>、<code>emit==NULL</code>。这样旧 compatibility plan 不会意外安装可调用 hook；测试同时检查 cursor
state 为 Disabled。

### 305.7 H3 hook plumbing 与 LTX 粗粒度事件（本轮校正）

H3：

~~~text
h3_params 尾部新增 const tc_memory_schedule_hooks_v1 *schedule_hooks
H3_PARAMS_DEFAULT 初始化为 NULL
H3Session bind 时复制 context hook POD
constrained generate 仅在 emit 非空时传入 h3_params
unbind 时清零 POD
h3_valid_params 校验 ABI，且 schedule hook 必须配合 constrained GPU/host hooks
~~~

LTX：

~~~text
ltx_native_options 尾部新增 schedule_hooks
LtxSession bind/unbind 复制/清零 POD
native denoiser create 校验 ABI，并复制 POD 到 denoiser-owned storage
schedule hook 必须同时有 streaming 和 memory hooks
worker/completion callback 仍未持有或调用 schedule hook
~~~

当前 H3 尚无 semantic emit；LTX 已在 `ltx_native_create()` 发出 DENOISER_LOAD.BEGIN/END，在
`ltx_native_run()` 发出 DENOISER.BEGIN/END，并使用连续 global step。共享
`native/core/memory_schedule_adapter.h` 已实现 ABI 校验、事件构造和 callback error 传播。
这仍不是全流程 schedule：TEXT、step/block/slot、VAE、handoff 等事件尚缺。
完整认证 schedule 不能通过 terminal gate；下一轮必须完成第 293、294 节事件表和 L1 adapter 测试，不能放宽 finish gate。

### 305.8 本轮实际验证

~~~text
memory accounting                 PASS
memory manifest                   PASS
memory schedule                   PASS
memory plan compiler              PASS
memory scheduler                  PASS
memory watchdog                   PASS
memory trace                      PASS
memory execution                  PASS
memory probe                      PASS
H3 schedule memory               PASS
contract                          81 tests run: 80 PASS / 1 fixture SKIP
native-only build                 PASS
H3 GPU memory hooks               SKIP(no Metal device)
LTX GPU memory hooks              SKIP(no Metal device)
~~~

native build 命令：

~~~text
env MLX_ROOT="$PWD/Python/lib/python3.13/site-packages/mlx" \
  TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
~~~

### 305.9 下一实现顺序

1. shared adapter emit helper 已完成；下一步补模型调用点和 error/cleanup 测试；
2. H3 owner-thread 顶层 stage event，再接 streamed DiT step/block/slot event；
3. LTX stage1/stage2 global-step event、handoff barrier 和 streamed block slot event；
4. fake loader/fake completion L1，证明 cursor complete、automatic epoch=0；
5. explicit site+instance registry 和 schedule coverage digest；
6. capability identity v2/evidence writer；
7. 真实 Metal L2 与低内存 L3；
8. 最后才允许审核 production record。

production registry 在本轮后仍为空；本节验证不能解释为 H3/LTX constrained execution 已 release-ready。

## 306. 2026-09-16 当前状态复盘：完成度、性能边界与下一步原则

### 306.1 结论先行

当前实现已经完成了内存受限模式的“控制面”，尚未完成所有模型的“数据面闭环”。控制面包括配置解析、Y/X/B 预算归一化、manifest/plan 编译、site/instance lifetime、reserve/commit/retire、completion mailbox、压力状态、swap 计数、strict schedule cursor、terminal drain 和 fail-closed admission。数据面仍缺少 H3 的完整 semantic event、LTX 的逐 step/block/slot event、所有 required allocation-site 的完整闭包、真实 Metal completion 证据和 16/24/32 GiB L3 campaign。

因此当前不能把“已经有 streaming 代码”解释为“已经保证进程不会 swap”。准确的发布状态是：

~~~text
L0 通用逻辑：通过
L1 synthetic/runtime adapter：部分通过
L2 真实 Metal：未通过（当前环境无 Metal，只能 SKIP）
L3 低内存实机：未开始认证
production capability registry：空
~~~

### 306.2 当前代码完成度矩阵

| 子系统 | 当前实现 | 完成度判断 | 下一道硬门 |
|---|---|---|---|
| request/profile 配置 | `MemoryConstrainedConfig`、profile overlay、`limit/buffer/min_free/slots/tiling` | 已完成基础合同 | 增加多模型 profile schema 校验 |
| 预算与 admission | `B=floor(Y*(100-X)/100)`、baseline/framework reserve、plan fit | 已完成框架 | manifest-backed exact upper |
| MemoryLedger | site/instance、active/pending/cache、alias、generation/domain | 已完成逻辑层 | 逐模型 required-site closure |
| schedule compiler/cursor | ABI、sequence、epoch、flags、poison、terminal ownership | 已完成逻辑层 | 模型真实事件覆盖 |
| watchdog/swap | owner polling、swapout delta、pressure state | 已完成框架 | L3 实机可靠性和噪声分类 |
| LTX slot pager | 1/2/3 slot 代码路径、异步 loader、slot refill 统计 | 已有可运行基础 | schedule token 与 ledger lease 绑定 |
| H3 pager | SSD streaming、pinned prefix、既有双 slot 基础 | 已有基础 | H3 slot 状态机和事件接线 |
| H3 schedule | hooks plumbing 和 ABI 校验 | 未闭环 | TEXT→DiT→VAE→EXPORT 全事件 |
| LTX schedule | load/stage BEGIN/END、global step | 粗粒度已完成 | step/block/refill/handoff 事件 |
| Flux/Z-Image | plan-only 设计 | 未执行 | opaque framework envelope + component staging |
| production registry | 空 registry | 有意保持为空 | 独立 reviewer 审核 L2/L3 evidence |

### 306.3 为什么不能简单依赖 swap

系统 swap 的主要问题不是单次 page-in/page-out 的固定耗时，而是它由系统全局压力、压缩器、其他进程和 allocator 行为共同决定，无法作为 request 级硬预算。对模型推理而言，swap 还会造成：

1. GPU/CPU unified-memory page fault 把原本可重叠的 I/O 变成不可预测 stall；
2. 热页被反复换入换出，出现抖动，长尾远大于平均值；
3. 进程无法知道某个模型 tensor 何时会被系统驱逐；
4. swap 使用量不能映射回具体 layer、slot 或 live interval；
5. 失败发生后通常已经污染了后续请求的缓存和温度状态。

主动 multi-slot streaming 的设计收益是利用模型的已知访问顺序管理 working set；它不关闭或替代 OS 虚拟内存。
每个 slot 有 upper、owner、generation、last-use 和 completion，因而更容易做 admission 和复现。
它是否改善吞吐/P95，必须实测；resident 全部放得下、OS 文件缓存命中或 I/O/解包受限时，主动 streaming 也可能更慢。
只有覆盖所有实际分配和 framework envelope 后，才能对受控 allocation 给出预算保证；周期 footprint 采样本身不能证明连续时间的硬上限。

### 306.4 当前性能数据应该如何解释

已有历史数据只能证明 streaming/pinned-prefix 的局部性能趋势，不能证明新的 hard-cap 端到端性能：

| 历史实验 | 观察到的结果 | 可以得出的结论 | 不能得出的结论 |
|---|---:|---|---|
| LTX 12 GiB，2→3 slots | warm denoise 11.030s→8.233s，约 1.34× | 计算窗口足以隐藏大部分第三槽 I/O | 不能证明完整进程峰值≤12 GiB |
| LTX 16 GiB，2→3 slots | warm denoise 10.063s→8.191s，约 1.23× | 三槽在该 workload 上有 overlap 收益 | 不能证明所有 shape 都适合三槽 |
| LTX resident→2-slot streamed | 首次 request 41.134s→29.471s | 冷启动时 streamed 可减少一次性加载成本 | warm resident 仍可能更快 |
| H3 pure stream→16 GiB pinned | denoise 17.306s→15.180s，约 1.14×；wait 降约 35.6% | pinned prefix 能降低 I/O 暴露等待 | 总 wall 几乎不变，且内存占用明显增加 |

这些结果支持“planner 根据预算选择 slot/pinned prefix”，不支持固定宣称“slot 越多越快”。新的 hard-cap 性能报告必须同时记录 `wall_p50/p95`、`time_to_first_step`、`compute_ms`、`exposed_io_wait_ms`、`slot_refill_count`、`drain_tail_ms`、`actual_peak` 和 `swapouts_delta`。

原始证据及其限制：

- `docs/design/validation/ltx-streamed-lookahead-2026-09-09.json`：Apple M4 Max、64 GiB；上表 1.34×/1.23× 来自
  64×64×9、11 steps、同一 retained engine 的第二次请求。12 GiB 改动同时把 pinned blocks 从 20 改成 19，
  streamed suffix 从 28 改成 29；不是只改 slot 数的纯微基准。逻辑 read bytes 从约 119.59 GB 增到 123.87 GB，
  并不等于物理 SSD bytes。生产 shape 704×448×97 记录样本少且存在温度差异，不能估计可靠 P95。
- `docs/design/validation/ltx-streamed-residency-2026-09-09.json`：12 GiB 是 denoiser working-set target，
  worker peak RSS 仍约 24.8 GiB，且未包含独立 decoder 子进程；不是 12 GiB 机器的成功证据。
- `docs/design/validation/h3-ssd-pinned-prefix-dit-2026-09-08.json`：64 GiB 机器、256×256×22、4 steps、synthetic text embedding，
  仅 DiT，每路线两次、未清 OS 文件缓存。`total_seconds` 只是 DiT load+denoise，不是完整生成端到端时间。

因此不使用“工程完成 70%”“L0 100%”等没有冻结需求分母的完成率。本轮能确认的是哪些具体测试通过，不能由此推断所有测试已齐备。

## 307. Multi-slot streaming 的通用架构：从 LTX 推广到 H3、Flux、Z-Image

### 307.1 把 slot 从“模型实现细节”提升为通用资源抽象

LTX 当前的 `streamed_block_slot` 只在 C 文件中存在，下一步应抽象成不依赖模型名称的 pager contract：

~~~cpp
struct StreamSlotDescriptor {
    uint32_t slot_id;
    uint32_t logical_item;
    uint64_t upper_bytes;
    uint64_t allocator_domain;
    uint64_t generation;
    SlotState state;              // Empty, Loading, Ready, InUse, Retiring, Failed
    uint64_t load_sequence;
    uint64_t last_use_sequence;
    StorageLease backing_lease;   // 池分配后持有；不是每次 refill 都重建 reservation
};

struct StreamWindow {
    uint32_t first_logical_item;
    uint32_t item_count;
    uint32_t lookahead;
    uint32_t slot_count;
};
~~~

模型只需提供 `load_item(item, slot)`、`bind_item(slot)`、`compute_item(slot)`、`last_use(item)` 四个 adapter 回调；通用 pager 负责：

- 按 window 计算 look-ahead；
- 在池初始化时 reserve/commit 真实 backing；refill 只推进独立 content/use generation；
- 复用 slot 前等待 GPU completion；
- 在 owner thread 发 schedule event；
- 把 loader 线程的结果写入 mailbox，而不直接推进 cursor；
- 失败时 join 所有 loader、取消未提交 reservation、把 slot 标成 Failed；
- 维护 `read/upload/compute/wait/retire` 时间。

建议新增：

~~~text
native/runtime/stream_slot.hpp/.cpp       通用 C++ pager/slot 状态机
native/core/stream_slot_c.h               C ABI adapter contract
native/runtime/stream_window.cpp          lookahead 与预算枚举
tools/native/stream_slot_simulator.py     无 Metal 的离散事件仿真器
tests/native/stream_slot_test.cpp         状态、并发、故障注入
tests/native/test_stream_slot.py          Python contract + simulator tests
~~~

### 307.2 Multi-slot 的统一事务顺序

每个逻辑 item（block、layer、tile、chunk 或 shard）采用以下顺序：

~~~text
POOL: reserve → allocate → commit backing → [many content cycles] → drain → destroy → release
CONTENT: Vacant → Loading → Ready → InUse → AwaitingFence → Vacant(next generation)
ERROR: any content state → Failed → join/drain → destroy, or quarantine if drain is unproven
~~~

约束：

- `slot_count=1` 是拟增 pager 的正确性基线；H3 当前只有双槽，LTX 当前单槽仍需要至少一个 pinned block；
- `slot_count=2` 是首发 overlap 候选；
- `slot_count=3` 只有在预算和 evidence 同时满足时启用；
- 本请求不得临时增槽或切换 plan；只能延后不改变 semantic event 顺序的 I/O 启动。serial candidate 只在下一个请求重新 admission 时选择；
- slot 的 `upper_bytes` 必须进入 manifest instance，不能只统计 logical tensor 大小；
- slot reuse 前必须有 completion token 或明确 GPU drain；
- loader/completion 线程永远不直接 emit strict schedule event；
- 同一个 slot 不能同时具有两个有效 generation；
- alias 复用必须证明旧 item 已经过 last-use 和 completion。

上一段的“EMPTY”只能表示 content 可重写，不能表示 backing 已从 ledger 删除；完整协议见第 314 节。

### 307.3 各模型的映射

| 模型 | logical item | resident prefix | slot 内容 | 首发 slot 建议 |
|---|---|---|---|---|
| H3 DiT | transformer block | norms、AdaLN、固定 pinned prefix | 首发 BF16 matrix backing；quantized cache 是独立候选 | 当前 2；另实现 1 后认证 |
| H3 Video VAE | temporal/spatial tile | 必要 decoder weights/metadata | tile activation/output staging；weights 不随 tile 无故重载 | 1，后续 2 |
| LTX DiT | 48 个 block | `pinned_blocks` | `block_weights` | 1→2→3 |
| Flux transformer | component/shard 或 block group | text/rope metadata | MLX/Metal component staging | 1 |
| Z-Image GGUF | shard/block group | tokenizer/embedding metadata | GGUF shard decode/upload | 1→2 |
| VAE/export | frame tile/chunk | codec metadata | decoded tile + RGB conversion staging | 1 |

Flux/Z-Image 在 MLX opaque allocator 尚未有 exact framework envelope 前，只能使用 component-staged plan-only 路径，不得把 MLX cache 当成可证明的 slot backing。

## 308. Planner、配置文件和不同内存机器的策略

### 308.1 配置文件分层

请求字段只表达用户意图；机器 profile 表达资源事实；candidate profile 表达已经验证的模型策略。三者不能互相越权。

以下是策略目录的概念草案，不是当前 parser 可读取的 profile；正式格式采用第 318 节的 versioned schema，
不要将下例直接传给现有 API。目录可置于 `profiles/memory/*.json`：

~~~json
{
  "schema": "turbocider.memory_profile.v1",
  "profile_id": "apple-unified-24g-conservative",
  "device_match": {"family": "AppleGPU", "physical_memory_gib": 24},
  "system_reserve_bytes": 2147483648,
  "default_buffer_percent": 15,
  "max_process_concurrency": 1,
  "candidates": {
    "h3_c_metal_streamed_v1": {
      "slot_policy": {"allowed": [1, 2], "preferred": 2},
      "pinned_prefix_policy": {"min": 0, "max": 8},
      "lookahead": 1,
      "allow_full_resident": false,
      "allow_tiling": true
    },
    "ltx_c_metal_streamed_video_v1": {
      "slot_policy": {"allowed": [1, 2, 3], "preferred": 3},
      "pinned_prefix_policy": {"min": 0, "max": 12},
      "lookahead": 2,
      "allow_full_resident": false,
      "allow_tiling": true
    }
  }
}
~~~

以下只列离线搜索档位，不自动启用、不给出这些机器能运行所有模型的保证，也不覆盖显式 X/Y：

| 机器档位 | 默认 buffer | 并发 | slot 策略 | 说明 |
|---|---:|---:|---|---|
| 16 GiB | 20% | 1 | 1，预算允许时 2 | 以不 swap 为第一目标，谨慎 pinned |
| 24 GiB | 15% | 1 | H3 2，LTX 2/3 | 首发平衡档 |
| 32 GiB | 12% | 1 | 2/3，允许更大 prefix | 追求较低 exposed wait |
| 48 GiB+ | 10% | constrained 仍为 1 | 优先评估 certified resident | 不得因为内存充足而改变 disabled 路径 |

这些只是默认建议，不是认证值。最终 slot/prefix 仍由 `budget - baseline - framework_upper - activation_reserve` 计算，并受 candidate record 限制。

### 308.2 与当前 profile schema 的兼容迁移

当前 `native/platform/apple/profile.mm` 使用 `schema_version=1`、`match.gpu_name`、`match.memory_bytes` 和 `models.<model>`。因此不应直接替换现有 profile 格式。建议采用向后兼容的增量方式：

1. 保留现有顶层字段和未知字段拒绝逻辑；
2. 现有 v1 不增加未知键；新字段只在拟增 `schema_version=2` 分支读取，旧文件继续走原解析器；
3. 用户预算保留在 `models.<model>.memory_constrained`，策略建议置于并列的 `memory_tuning`，不混入执行授权；
4. 解析后分别归一化到预算 config 和候选偏好；禁止 profile 写入 `execution_supported`、`framework_upper`、`verified_refill_slots` 或 `release_enabled`；
5. profile provenance digest 进入结果；只有实际执行布局/shape/slot/backend 改变才需匹配新的认证 identity。注释或排序变化不应生成不同执行计划。

这样可以继续读取现有 `profiles/*.json`，又为不同内存档位增加 slot/prefix 策略，不会让配置文件绕过 capability registry。

### 308.3 “内存充足但开启 constrained” 的性能定义

`memory_constrained.enabled=true` 即使在 48 GiB/64 GiB 机器上，也不能承诺与 disabled 完全相同：受限路径需要 admission、ledger、watchdog 和 terminal drain。设计上分三种情况：

- **disabled**：绝对零额外对象/调用，作为默认性能基线；
- **constrained + 已认证 fully-resident candidate**：允许 resident weights，但仍执行硬预算和 completion 约束，性能目标由该 candidate 的 evidence 声明；
- **constrained + streamed candidate**：接受可测量的 I/O/drain 开销，以换取硬上限。

因此“内存充足机器不影响性能”应验收为：默认模式零回归；受限模式在预算足够时自动选择已认证 resident candidate，且相对该 candidate 的 unconstrained 基线只允许预先声明的运行时开销。不能通过在高内存机器上偷偷关闭 enforcement 来制造零开销结果。

当前并没有这种完整 resident constrained candidate：`prepare_memory_execution()` 会 unload 已有 session，terminal 又要求资源 zero-state。
LTX 旧局部 planner 的 `fully_resident=true` 只代表 denoiser weights 能常驻，不代表 text/VAE/output/整个请求已认证。
不能为消除 warm regression 直接删去 unload 或放宽 zero-state；retained cache 需要独立 session-root ledger 和认证，见第 316.5 节。

### 308.4 Candidate 选择算法

对每个候选 `k` 计算：

~~~text
available = B - process_baseline - framework_upper - non_denoiser_reserve
working_set(k) = activation_reserve + pinned(k)*block_upper
                  + slots(k)*slot_upper + staging(k) + output_overlap(k)
~~~

只保留 `working_set(k) <= available` 的候选，然后按以下顺序排序：

1. 安全硬过滤：manifest complete、unknown upper=0、required sites complete；
2. 质量硬过滤：sampler/dtype/shape 与 reference parity；
3. 稳定性硬过滤：completion/cleanup/release gate 均已通过；
4. 性能：预测 exposed wait、predicted wall、SSD bytes；
5. headroom：剩余预算越大越优先。

不能用瞬时 RSS、随机数或 wall-time 单次样本改变安全排序。预测只影响性能排序，不能放宽硬 budget。

### 308.5 压力状态下的动作

~~~text
Normal   : 固定 lookahead，允许已认证 prefetch
Tight    : 停止 optional prefetch，保持当前 slot 数，不增加 resident prefix
Critical : 立即禁止新派发，在最近合法 safe point drain 并失败；不通过暂时回落继续成功
~~~

压力控制器不负责“救活”一个已经越界的计划，也不能把 swap 当作新预算。发生 `actual_peak > B`、mailbox overflow、未知 allocation 或 swapout 增长时，必须进入失败/隔离流程。

## 309. 如何模拟“multi-slot 比 swap 更快”

### 309.1 无 Metal 的离散事件模拟器

新增 `tools/native/stream_slot_simulator.py`，输入：

~~~text
block_count
block_upper_bytes
activation_reserve
slot_count
read_ms_distribution
upload_ms_distribution
compute_ms_distribution
completion_ms_distribution
process_baseline
budget
swap_threshold
swap_penalty_ms
~~~

模拟两个策略：

```text
controlled-slot: 固定 slot/window，超预算立即拒绝
vm-sensitivity:  访问轨迹 + file/anonymous 页类型 + 显式假设的回收/压缩策略；不声称复现 macOS pager
```

输出至少包含：

- P50/P95/P99 wall time；
- first-step latency；
- exposed I/O wait；
- GPU idle ratio；
- peak accounted bytes；
- synthetic swap events；
- slot refill count；
- drain tail；
- request failure rate。

模拟器只能用于选择 slot、lookahead 和 profile 的候选参数，不能替代真实 Metal/L3 证据。

### 309.2 真实机器 A/B 试验

每个 workload 固定 checkpoint、shape、seed、steps、线程数和温度窗口，采用：

~~~text
A = disabled/default resident 或现有默认路径
B = constrained multi-slot
C = swap-enabled stress baseline（仅诊断，不作为产品路径）
顺序 = A/B/C/B/A 或 A/C/B/A/C
~~~

C 只用于诊断，不作为 release candidate。5 次 warm、3 次 process-cold 仅能作为 smoke；不能据此宣布稳定 P95/P99。
正式配对样本数、文件缓存状态、隔离压力 campaign 和停止条件见第 317 节。
production 遇到全局 swapout 仍 fail-closed；离线报告若能确认外部噪声，可标记 inconclusive，但不能改写成 PASS。

### 309.3 判断标准

multi-slot 相对 swap 的结论必须按三层给出：

1. 安全：`actual_peak<=B`、`unknown_peak=0`、`swapouts_delta=0`；
2. 稳定：P95/P99 没有 swap-induced 长尾，失败可定位；
3. 性能：在同一预算和 workload 下，`wall_p50`、`wall_p95`、GPU idle、exposed wait 均不劣于 swap baseline，或以可接受的固定 slowdown 换取硬上限。

不能只比较一次平均 wall time。swap 可能偶尔更快，但只要出现不可预测的长尾或越界，就不能作为内存受限模式的实现。

### 309.4 仿真参数的校准要求

模拟器中的 `read_ms`、`upload_ms`、`compute_ms` 和 `swap_penalty_ms` 必须来自同一机器、同一 checkpoint、同一 shape 的实测分布。没有实测数据时只能使用区间分析：例如对 swap penalty 使用 `[p50,p95]` 两套参数，并把结果标记 `model_only`。禁止通过人为设置极大的 swap penalty 来预先“证明” multi-slot 更快。

仿真与实机的差异必须单独报告：

~~~text
simulation_prediction_error = |predicted_wall - measured_wall| / measured_wall
simulation_peak_error        = |predicted_peak - measured_peak| / measured_peak
~~~

误差只影响 planner 参数选择；硬安全门始终来自真实 ledger、真实 process observation 和真实 swap counters。

## 310. 默认路径零回归的代码架构约束

### 310.1 入口短路

`memory_constrained.enabled == false` 时，memory preparation 必须在创建 memory context 前返回；普通 session 的创建和缓存行为保持原样。
禁止“创建一个 Disabled context，再让每个 hot path 判断”的实现。此处的零开销指无新增 memory-policy 工作，
不否认历史默认路径自身已有的设备查询、eval 或 cache 操作；本轮也不能删除它们。

### 310.2 双路径编译与运行时开销

默认路径要求：

- `make_schedule_hooks()` 不被调用；
- process/swap observer 调用次数为 0；
- memory ledger/watchdog/trace 对象不存在；
- 不增加 unload/cache-clear；
- 不改变线程拓扑、cache key、resident session 生命周期；
- 不在 block/step hot loop 增加不可预测锁。

受限路径的 callback 只在 semantic boundary 调用；每个 block 的高频 telemetry 应使用无分配、无锁的固定结构，最终由 owner thread 批量归并到 trace。

### 310.3 性能回归门

默认路径采用交错 ABBA：同一 checkpoint、shape、seed、warmup、测量次数、并发度和 cache 状态。首轮门槛：

~~~text
steady-state wall median regression <= 2%
P95 regression <= 5%
额外 probe/hook/unload/cache-clear = 0
输出 digest/quality unchanged
~~~

任何新增 probe 或 hook 都是硬失败，不能用“平均只慢 1%”抵消架构回归。

## 311. 下一阶段实施拆分与完成门

### 311.1 推荐 PR 顺序

~~~text
PR-S2  通用 StreamSlotDescriptor / state machine / simulator
PR-H2  H3 owner-thread coarse stage events
PR-H3  H3 streamed DiT block/slot events + required site closure
PR-L2  LTX per-step/block/refill/handoff events
PR-C1  synthetic L1 exact event sequence + fault injection
PR-M1  manifest coverage digest + capability identity v2
PR-P1  pager overlap instrumentation and candidate enumerator
PR-F1  Flux component-staged plan-only adapter
PR-Z1  Z-Image shard/component plan-only adapter
PR-M2  real Metal L2 campaign
PR-L3  16/24/32 GiB low-memory campaign
PR-R1  disabled ABBA + reviewer evidence + registry decision
~~~

每个 PR 只推进一个 capability level；在 L2/L3 证据完成前，production registry 不得加入 record。

### 311.2 立即下一步

1. 新增通用 `stream_slot` 状态机和无 Metal 离散事件 simulator；
2. 为 constrained 新增 LTX slot adapter，复用数值 kernel；默认 `run_streamed_block_stack()` 暂不替换，先做同配置配对验证；
3. 给 LTX 补齐 `DENOISER.step.*`、`block.*`、`slot.*` 事件，并增加 exact sequence synthetic test；
4. 给 H3 增加 TEXT、conditioning、denoiser load、DiT block、VAE、export 粗粒度事件，再接 block/slot 细粒度事件；
5. 把每个 slot 的 reserve/commit/retire 与 compiled instance 绑定，生成 coverage digest；
6. 完成后再做真实 Metal L2；没有 Metal 时只运行 simulator、L0、L1，结果标记 SKIP/NOT CERTIFIED；
7. 最后执行低内存 L3 和 disabled ABBA。

### 311.3 复盘后的总判断

主动的 request-local multi-slot streaming/loading/offloading 是更适合 TurboCider 的长期方向：它能提供可证明的上限、可定位的失败和可优化的 overlap；系统 swap 只能作为压力测试基线和失败信号。要获得比 swap 更快的结论，必须比较 P95/尾延迟、GPU idle、exposed wait 和 swap-induced stall，而不是只比较平均耗时。

推广到 H3、Flux、Z-Image 的关键不是复制 LTX 的 C 结构，而是抽象出统一的 slot/pager contract、manifest instance、completion token 和 owner-thread schedule。这样可以让内存充足机器继续走完全不带 ledger/hook 的默认路径，同时让低内存机器按 profile 选择 1/2/3 slot、pinned prefix、tile 和 component staging。

## 312. 当前工作树的下一步验收顺序

为避免“代码已经接线”被误认为“性能已经认证”，建议按以下顺序执行：

~~~text
T0  grep/静态合同：disabled 路径无 probe/hook/watchdog/cache-clear
T1  L0：memory_* 单元测试、adapter、schedule cursor、plan compiler
T2  L1：fake pager + fake completion + exact event sequence + fault injection
T3  simulator：1/2/3 slot 与 swap baseline 的区间比较
T4  Metal L2：真实 buffer allocation、completion、peak、quality parity
T5  L3：16/24/32 GiB，cold/warm，swapout=0，P50/P95，重复运行
T6  disabled ABBA：wall/peak/output/thread/cache 全部回归
T7  reviewer：capability identity、manifest digest、evidence digest、rollback
T8  registry：仅在 T0–T7 全部通过后加入 release record
~~~

当前已达到 T0/T1 的大部分要求；T2 尚未完成，T3 需要新增工具，T4/T5 因当前环境无 Metal 和低内存实机尚不能执行。production registry 继续保持为空是正确状态。

## 313. 基于当前源码的逐模块审计与改造边界

### 313.1 LTX：已有 slot 代码与需要补齐的部分

当前 `native/models/ltx_runtime/ltx_blocks.c` 的 `run_streamed_block_stack()` 已具备：

- `LTX_MAX_REFILL_SLOTS=3` 的固定容量；
- `streamed_block_slot` 保存一份可复用 `block_weights`；
- depth>1 时异步启动 look-ahead loader；
- owner thread `pthread_join()` 后才消费 ready block；
- slot refill/allocation/load/wait 统计；
- pinned prefix 与 streamed suffix 分离。

但它还不是完整的 MemoryLedger slot：slot reuse 的逻辑身份与物理 lease 还没有闭合，
`load_streamed_block_worker()` 首次会 `load_block_weights()` 分配资源，失败时本地 free/reset；
这与 owner-only epoch/cursor 不是天然兼容的。不是说后台 allocator 一定有 bug，而是当前缺少冻结 epoch 和并发身份证明。
建议在 constrained 新分支由 owner 预先 reserve/allocate 所有 slot backing，worker 仅向独占的现成 backing 写入，
不访问 `MemoryExecutionContext`，不自行选择 allocation instance。需要 staging 时也由 owner 预分配独立 lease。
因此下一步不重写数值 kernel，而是新增 adapter 层：

~~~text
owner reserve/commit pool → async fill → owner consumes ready message
→ semantic bind/compute → per-slot last-use fence → mark content reusable
~~~

### 313.2 H3：当前已有固定双槽，不能在文档中写成任意 N 槽

当前 `native/models/h3_runtime/h3_dit.c` 使用两个 `stream_slots`，并通过 `stream_ready_layer/stream_ready_slot` 连接前后台加载。
这证明了双槽 overlap 的实现基础，但不等于已经支持动态 multi-slot。H3 下一步应先完成：

1. 为现有双槽定义显式 `slot_id`、`logical_block`、`generation` 和 `last_use`；
2. 在加载线程结束后由 owner thread 发 READY/UPLOAD 事件；
3. 在复用另一槽前等待 GPU completion；
4. 把 H3 block upper、pinned prefix、upload staging 纳入 manifest instance；
5. 只有在双槽 L2/L3 证据后，才评估第三槽。

H3 的 double-slot 是当前 candidate 约束，不应通过把 `stream_slots[2]` 改成可变数组就宣称已经完成通用 multi-slot。

### 313.3 Flux/Z-Image：MLX lazy graph 和 cache 是主要边界

当前 Flux/Z-Image 路径大量使用 `mx::eval()`、`mx::async_eval()`、`mx::clear_cache()` 和 MLX `Weights` 容器；它们的 tensor backing、lazy dependency 和 allocator cache 不能仅凭 C++ 对象析构推断生命周期。
因此：

- 第一阶段只做 component-staged（text→transformer→VAE）plan-only；
- 每个 component 必须有 framework upper 和显式 drain/clear 证据；
- 不把 MLX 的 memory limit 当作 TurboCider 进程硬上限；
- 不在默认 Flux/Z-Image hot path 插入 per-block ledger callback；
- 只有取得 opaque framework envelope 后，才可实现 transformer block/shard slot。

### 313.4 Apple/Metal completion 的线程边界

当前 H3/LTX GPU wrapper 使用 command buffer completion 和 owner-side drain。严格 schedule 的正确性依赖 owner thread；completion callback 只能投递轻量 token 到 mailbox，不能：

- 直接推进 schedule cursor；
- 直接进入下一个 epoch；
- 访问已解绑的 session/context；
- 在 callback 中分配新的 slot；
- 调用可能触发 `mx::eval()` 或全局 cache clear 的操作。

这条边界必须由 synthetic test 和真实 Metal fault test 同时覆盖。

### 313.5 本轮发现的必须先解决的接口不一致

| 源码位置/行为 | 风险 | 下一修改建议 |
|---|---|---|
| `ltx_native_create()` 先真实加载 block0 才 `block_resident_bytes()` 估算 | 首次大分配早于精确 whole-request plan；默认路径有价值但不满足严格预估 | constrained metadata probe 从 header 计算各 block 完整 layout，首次 materialization 只能在 admission 后 |
| `block_residency.c` 用单一 `block_bytes`，`slot_bytes/minimum_bytes` 保留双 block 语义 | 不能直接解释成任意 N 槽总预算；异构 block 大小也未覆盖 | 新 internal layout plan，保留旧 C helper；为 K=1/2/3/resident 分别验证总 capacity |
| LTX stream loop 要求 `pinned_count>0`，创建 workspace 使用 `weights[0]` | 文档里的零 pinned 单槽不是当前可用功能 | 首轮仍保留 block0；另 PR 抽离 immutable metadata 后再支持 P=0 |
| policy 取 `refill_slots=max_refill_slots`，底层 planner 可减槽或转 fully resident | record 的槽数可能与实际不一致 | constrained 使用唯一 ResolvedLayout；底层只验证，不再次自由规划 |
| H3 用 `slot ^ 1`、数组 `[2]` | 三槽不能只增加常量 | 独立显式 slot assignment；先保留现有双槽精确映射 |
| LTX `run_block()` 返回后立即开始 refill | 是否可复写依赖 command batch 实际完成 | `ltx_finish_command()` 非 batch 会等待，但 batch 会延期；adapter 必须证明所有 reader 完成，不能通用假设 return=GPU done |
| LTX 当前 schedule helper 和 global-step 计算在默认 run 中也可执行 | “NULL hook 无 callback”不等于零新增 CPU 工作 | schedule-only sigma 校验/坐标/helper 调用置于 constrained 入口；默认数值循环不加 probe/clock/lock |
| `memory_execution.cpp` cursor 先修改 attempted，再校验 owner | 错线程拒绝不自动等于并发安全 | 业务线程永不直接调用；诊断错线程测试需 TSan 验证，若支持并发拒绝须独立原子 ingress fault，不让非 owner 写 cursor/string |
| Flux 4B compiled single blocks 特意不每块 `eval()` | 通用 pager 强插 per-block wait 会丢失现有收益 | 保留默认 compiled-step 路径；仅在单独 streamed candidate 设 group materialization boundary |
| Z-Image `load()` 整组件加载，`Weights::erase_prefix()` 不能证明 graph 已释放引用 | “删 map=offload”会错误回收预算 | component 路线先做评估+同步+release 证据，再实现按 group 读文件的 reader |

这里列的是待修复/待证明项，不是本轮已实施补丁；尤其不能将最后一列当成当前行为。

## 314. 通用 StreamSlot API 的建议接口

### 314.1 两种生命周期，不能混用 completion token

新增接口分为三层：纯计划/状态机、固定容量 executor、模型 adapter。状态机本身不创建线程、不做 I/O；executor
只在 constrained request 中按 plan 创建固定 worker，adapter 保留 layout、格式转换及数值 kernel。
以下是内部接口草图，名称均待实现，不是已支持的 public API。

~~~cpp
struct ItemKey {
    uint32_t component, pass, global_step, block_group;
};
struct SlotTicket {
    uint32_t slot_id;
    uint64_t request_generation, content_generation;
    ItemKey item;
};
struct SlotBacking {                      // owner-only
    uint64_t capacity_bytes;
    BackendStorageHandle storage;         // opaque，示意类型
    StorageLease lease;                   // 从池创建一直到实际销毁
};
struct SlotUseFence {                     // 新增，不等于 MemoryCompletionToken
    SlotTicket ticket;
    uint64_t queue_id, submission_sequence;
};
~~~

`MemoryCompletionToken` 当前表示 backing 进入 pending-release 后的释放完成。它不能直接拿来表示每次 refill 的 GPU 完成：
否则池 backing 明明仍存在，却会被账本当成释放。新 `SlotUseFence` 仅证明内容可重写，不能减少 ledger bytes。
池最终销毁才走现有 retire/complete 路径。同一 slot 可包含多块 Metal buffer，manifest 按唯一 backing ID 去重；
`SlotBacking` 可以是 bounded bundle，不能伪装成一个未对账的大对象。

### 314.2 Adapter 的操作合同

| 接口（拟增） | 线程 | 输入/输出 | 必须满足 |
|---|---|---|---|
| `describe_items(metadata, shape)` | preflight owner | item 数、文件 ranges、格式、layout class、容量上界 | 不实例化 tensor；checked offset/size |
| `allocate_pool(resolved_plan, reservations)` | owner | 已 commit 的 backing bundle | 分配前逐 site reservation，失败反向回滚 |
| `fill(ticket, const_read_plan, writable_spans)` | I/O worker | bounded ReadyMessage | 不分配 model GPU backing、不访问 cursor、不 free slot |
| `accept_ready(message)` | owner | Loading→Ready | 校验 item、generation、完整 bytes/status；worker release/acquire 可见性 |
| `bind_and_encode(item, slot)` | owner | submission + last-use fences | 不允许隐式物化下一整模型；保留原 dtype/kernels |
| `post_use_complete(fence,status)` | GPU callback | 有界 mailbox | 不分配、不碰模型、不 emit schedule |
| `consume_use_complete()` | owner | AwaitingFence→Vacant | 所有 reader queues 完成，检查 duplicate/stale/error |
| `destroy_pool()` | owner | retired backing/zero pool leases | 先 join I/O，再证明 GPU drain，最后释放 backing |

C 模型用 `native/core/stream_slot_c.h` 的 opaque handle + size/version POD 接入；C++ root 管理资源，C adapter
不跨 ABI 传 `std::vector`、exception 或 `MemorySiteToken`。共享 C helper 仍用于 semantic schedule emit。
runtime capacity、worker stack、read descriptor、trace/mailbox 大小均属于 control-memory 上界；不得把“线程池”视为免费。

### 314.3 确定性 owner 调度与三槽示例

首发布局静态绑定 `slot=(suffix_index % K)`；K 不等于 read workers。设三槽、单 I/O worker、连续 block B0…B4：

~~~text
owner:  plan PREFETCH B0/s0,B1/s1,B2/s2 -> accept B0 -> COMPUTE B0
I/O:    read B0 -> read B1 -> read B2            (与 GPU B0 可重叠)
GPU:                       compute B0 --fence-->
owner:  consume fence s0 -> Vacant s0 -> PREFETCH B3/s0
        accept B1 -> COMPUTE B1 -> ... -> B4/s1
~~~

该图是依赖关系示意，不是已测 overlap。允许 GPU B1 与 I/O B3 同时进行，但绝不允许 read B3 写入尚被 GPU B0 读取的 s0。
多 I/O worker 的完成顺序可以随机，owner 仍按确定的 semantic sequence 接受逻辑 READY/UPLOAD/COMPUTE；
mailbox 到达顺序不进入 plan identity。实时 timestamp 只做 telemetry，不影响 epoch。

区分两类事件：

- strict schedule event：逻辑操作的 admission/ownership 边界，顺序固定；READY 可映射为既有 `UPLOAD`，不凭空使用不存在的枚举；
- telemetry event：read start/end、fence posted/consumed，允许异步乱序，通过 ticket 关联，不驱动 cursor。

Pressure=Tight 时不删除必需 PREFETCH binding；保持同一逻辑 intent，可以把其物理派发延后到 just-in-time。
若延后会超出已编译 lifetime 或破坏操作顺序，则该候选不支持 runtime suppression：失败或下个请求改选计划。
首发不实现“请求中途换到 serial plan”，避免 poisoned cursor 后继续执行。

### 314.4 现有 schedule ABI 的 key 限制

当前 `tc_memory_schedule_event_v1` 只有 stage/action/step/block/tile/branch/slot/flags，没有 content_generation、pass 或 load-item ID。
因此必须：

1. 确定每类事件 key 的规范；stage boundary 与 step boundary 不得都编码成同一个 `DENOISER.BEGIN(step=s,block=NONE)`；
2. 在不歧义的子集中采用 `DENOISER.COMPUTE(step=s,block=b,slot=k)` 等既有动作；
3. 需要重复 pass/retry/不同 group 的 candidate，无法在 v1 唯一表示时提出 v2，不能把 pass 硬塞进 branch/tile；
4. schedule revision 必须包含 key 约定和 exact count，模型 adapter 测试从实际 emit 收集事件与独立期望序列比较；
5. slot ticket generation 先存在内部 pager 中，不能误说当前 ABI 已支持它。

### 314.5 Offloading 的含义与统一内存

Apple Silicon 的 CPU/GPU 使用同一物理池；“GPU tensor 搬到 CPU tensor”不自动减少机器物理内存，还可能形成副本。
slot 策略应优先选：

- 不可变 weights：完成使用后丢弃内容/复用 backing，下次从原 checkpoint range 重读；不写回相同 weights；
- shared buffer：可由 worker 向已分配且独占的 writable span 读入，upload 阶段可以只是 bind/可见性边界，时间为 0；
- private buffer：显式 host staging + copy destination + in-flight copy，两个 backing 均记账；
- activation：先复用 last-use workspace；有状态 latent/KV/conditioning 不可像不可变 weight 一样丢弃；
- VAE：权重常驻该阶段，tile 控制 activation/output，不默认每 tile 重新加载全部 decoder；
- dirty cache/LoRA merge：首发不支持隐式磁盘写回，需要单独的 artifact identity 和容量模型。

`clear_cache()`、map erase 或逻辑 slot Vacant 不是物理释放证明。保留 backing 继续计费，实际释放才减账。
文件页的系统缓存即使未完全计入某一进程 footprint，仍会竞争系统 RAM；因此同时采集系统可用内存，不能以
“managed buffers≤B”宣称整个系统不会 swap。

### 314.6 取消、错误与有界性

- cancel：owner 停止派发；worker 按 bounded read chunk 检查取消；I/O 返回后 join，GPU completion 确认后销毁；
- short read：循环处理 EINTR/短读，真正 EOF 直接失败；offset+length、dtype unpack 输出容量做溢出检查；
- checksum：使用预先校验的 checkpoint/pack identity；逐次读校验若开启需计时，不把整权重 hash 放到每次 refill；
- partial fill：不标 READY、不覆盖仍可运行的上一内容身份；失败 request 不继续运行；
- callback 延迟/重复/旧 generation/mailbox 满：sticky failure，不以超时到达直接视作释放完成；
- thread-create 失败：首发 fail-closed；若要同步 fallback，必须证明 layout/semantic sequence 相同且控制资源预算不增加；
- drain 无法证明：worker quarantine，保持对象可达直到可安全清理；不能为了归零而 free in-flight storage。

worker 只需向预分配 mailbox 写 fixed-size status。正常路径热循环不生成长字符串、不扩容 trace vector；详细错误在 owner 失败路径格式化。
首次 read workers 建议只认证 1，再评估 2；三槽不需要三读线程。独立 perf PR 才能替换现有 LTX 每 refill create/join 的策略。


## 315. 工具链设计：从估算到 evidence

建议形成以下离线工具链，而不是把所有逻辑塞进运行时：

~~~text
1. model_inspect.py
   读取 checkpoint metadata、shape、dtype、shard/block 大小
2. manifest_builder
   生成 site/instance/live interval，不创建 GPU tensor
3. stream_slot_simulator.py
   比较 1/2/3 slot、lookahead、pinned prefix 和 swap 假设
4. native campaign runner
   运行 cold/warm、记录 trace、peak、swap、quality、wall
5. evidence_packager.py
   固化 candidate key、manifest digest、plan digest、环境和原始日志
6. verifier
   检查 exact event count、zero unknown、swapout=0、quality、性能门
~~~

工具输出必须区分：

```text
model_only       仅由 metadata/仿真推导
synthetic        fake allocator/completion 的逻辑证据
metal_l2         真实设备但非低内存 campaign
l3_low_memory    指定物理内存档位的实机证据
release_reviewed reviewer 已签核，可考虑 registry
```

禁止把 `model_only` 或 `synthetic` JSON 复制到 production evidence 目录。

## 316. Multi-slot 的预算与参数关系

slot 数、lookahead、I/O 队列深度和 pinned prefix 必须分开建模：

~~~text
resident = pinned_prefix * item_upper
slot_backing = slot_count * slot_upper
staging = decode_buffer + upload_buffer * in_flight_uploads
activation = model_activation_upper
framework = graph/cache/helper_upper
peak_upper = baseline + resident + slot_backing + staging
             + activation + framework + output_overlap
~~~

`lookahead` 影响何时启动 load；`slot_count` 影响同时可保留多少 backing；`in_flight_uploads` 影响 GPU/CPU 交叠；三者不能共用一个 `max_refill_slots` 字段。
建议在内部 plan 中拆为：

~~~text
max_slots
preferred_slots
lookahead_items
max_inflight_uploads
pinned_prefix_items
~~

对当前 LTX，兼容映射暂时为 `max_slots = max_refill_slots`、`lookahead_items = max_slots`；完成通用 pager 后再升级 schema。

### 316.5 Retained cache 的单独认证边界

retained cache 不能只把 request ledger 留在 session 中。要认证它，必须有独立的 session-root record，至少包含：

~~~text
checkpoint digest + backend/runtime/device identity
resident site/instance list + capacity
cache generation + eviction generation
last request owner + active lease count
warm/cold cleanup evidence
~~~

admission 时先把 retained backing 作为 process baseline 或 session claim 纳入预算；不能在 request 开始后才发现已有 20 GiB resident model。
释放 retained cache 也必须经过 owner-side drain、generation invalidate 和可重算的 zero-state 证据。首发 constrained 模式继续采用 clean session boundary，直到该 record 独立通过 L2/L3。

## 317. 仿真与实测 campaign 的最小可复现规范

### 317.1 仿真输入

仿真器至少接受 deterministic seed、每个 item 的 read/upload/compute 分布、slot upper、activation/framework upper、预算和 swap penalty 区间。
每个结果必须包含输入文件 digest，避免同名配置被覆盖后无法复算。

### 317.2 实测矩阵

每个 candidate 至少建立：

~~~text
memory_budget: 12/16/24/32 GiB（按 candidate 可行性裁剪）
slots:         1/2/3（仅测试支持的值）
cache_state:   cold-process / warm-process / retained-session
shape:         small-smoke / production-shape
repeat:        >=5 warm + >=3 cold smoke
~~~

`retained-session` 只对 disabled 或有独立 cache lease 证明的 constrained candidate 开放。constrained 首发默认使用 clean process boundary。

### 317.3 性能报告公式

~~~text
speedup = baseline_wall / candidate_wall
overlap_ratio = hidden_io_ms / max(compute_window_ms, 1)
exposed_wait_ratio = exposed_io_wait_ms / max(wall_ms, 1)
peak_headroom = (budget_bytes - actual_peak_bytes) / budget_bytes
~~

speedup 只用于性能排序；`peak_headroom < 0`、`unknown_peak > 0` 或 `swapouts_delta > 0` 直接失败。

## 318. 配置 schema v2 的最终建议

在保持现有 v1 parser 的前提下，拟增 v2 结构如下：

~~~json
{
  "schema_version": 2,
  "enabled": true,
  "match": {"gpu_name": "Apple M4 Max", "memory_bytes": 68719476736},
  "memory_defaults": {
    "buffer_percent": 15,
    "min_free_bytes": 2147483648,
    "max_refill_slots": 3,
    "allow_quality_preserving_tiling": true
  },
  "models": {
    "ltx-2.5-distilled": {
      "memory_constrained": {"limit_bytes": 25769803776},
      "memory_tuning": {
        "slot_candidates": [1, 2, 3],
        "lookahead_candidates": [0, 1, 2],
        "pinned_prefix_max": 42,
        "allow_full_resident": true
      }
    }
  }
}
~~~

解析顺序固定为：机器 defaults → model tuning → request override → candidate/manifest authorization。`memory_tuning` 只能缩小或排序候选集合，不能扩大用户预算，也不能开启未认证 slot。

## 319. 当前阶段的最终判断

当前代码已经足以支撑 L0、L1 的继续开发和旧 streaming 性能回归，但还不足以证明：

- constrained request 的完整进程峰值始终不超过 B；
- H3/LTX 所有 allocation site 都被 manifest 覆盖；
- multi-slot 在所有机器上都比 swap 更快；
- Flux/Z-Image 可以安全地做 block-level streaming；
- constrained 在高内存机器上与 disabled 零开销。

下一步的正确工程策略是保持默认路径冻结，新增通用 slot/pager 只服务 constrained 分支，先用 simulator 和 synthetic completion 把状态机跑通，再用真实 Metal 和低内存 campaign 认证。只有得到完整 evidence 后，才把具体 candidate 写入 capability registry；在此之前保持 plan-only/fail-closed。
