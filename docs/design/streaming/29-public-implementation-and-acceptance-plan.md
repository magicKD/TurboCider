# 29 · Public Streaming 实施、实验与验收计划

[目录](README.md) · [Runtime 代码设计](28-public-runtime-code-design.md) · [详细集成规格](30-public-streaming-detailed-integration.md) · [Runtime/App 工程规格](33-public-runtime-app-engineering-spec.md) · [配置与校准手册](31-public-streaming-config-calibration-runbook.md) · [档位产品语义](23-public-memory-tier-presets.md) · [候选探索](24-memory-tier-exploration-and-acceptance.md) · [发布验收](26-public-preset-acceptance-and-release.md)

日期：2026-09-17。分支：`feat/stream`。当前控制面提交：`63b73d9`。

状态：**实施计划和验收规格，尚未完成 public 发布**。本文给出可以直接分配给开发、性能、App 和发布负责人的任务单。
它特别补充三项内容：

1. 将代码工作拆成可回滚的 PR 和明确的完成门；
2. 将 8/10/12/16/20 GiB 的探索变成有界、可重复、可审计的实验；
3. 将“比 swap 快”“不影响内存充足机器”拆成可测量的假设，而不是口头结论。

本文不改变文档 12 的 P0/P1/P2/P3/P4 统计定义，也不把当前 private anchor 或 metadata skeleton 写成 public 资格。

## 1. 当前真实状态（必须先对齐）

已提交 public 控制面基线仍是 `63b73d9`；当前工作树另有 R1/R2 与 exact C ABI 增量，尚未形成阶段提交：

| 项目 | 状态 | 证据/限制 |
|---|---|---|
| selector schema v2 | 已提交 | parser、profile merge、五档 target、disabled/preset 语义 |
| production catalog | 结构扩展中且 production 仍为空 | 工作树已加入完整 source/workload/runtime/device/plan/calibration/performance/release 字段；仍故意 fail-closed |
| canonical identity | 工作树已实现基础 | typed length-prefix encoder、record/source/workload/runtime/device/resolution digest；跨语言 golden 尚缺 |
| host resolver/authority | 工作树已接 engine | select/authorize、internal-only authority、stale/source/layout/device 检查；execution-time source lease revalidation 尚缺 |
| session public hooks | 工作树实现中 | probe/compile/generate_resolved 默认拒绝；四模型尚未 override |
| options C ABI | 已提交 | metadata-only、tentative、无 artifact exact identity |
| Swift v2 类型 | 工作树扩展中 | resolution/error 类型和 resolve API 已接；semantic parity/job envelope 尚缺 |
| unresolved selector gate | 已扩展为 exact fail-closed | resolve/generate 空 catalog 在 GPU/session 前拒绝；prepare 明确 unsupported |
| exact engine resolve | 工作树已实现 | `tc_engine_resolve_streaming_json`、engine model/root/container identity；null/busy/ownership/cancel/error envelope 测试尚缺 |
| public generate route | 工作树已接通框架分支 | resolve→global GPU lock→revalidate→`generate_resolved`；四模型默认 hook 仍拒绝 |
| App 高级开关 | 未实现 | 无 `StreamingChoice`/OptionsStore/picker |
| calibration 工具链 | 未实现 | 无 fresh-process tree sampler/catalog builder |
| reviewed records | 未实现 | 没有任何模型/target 可公开 |
| default P0 | 仅有基础回归 | 必须在每个 runtime PR 后重跑；控制面提交不等于性能签核 |

下一步 exact runtime 收口的最新错误优先级、source revalidation、RunResult、App 和模型接入合同见
[32](32-public-streaming-completion-spec.md)。下文 PR-R1/R2/R3 仍作为可回滚任务分解使用，但“尚未实现”状态以本节和32为准。

### 1.1 已有性能锚点的正确用法

已有 LTX、Z-Image、H3 Turbo 和 Flux 9B 结果只用于：

- 保护当前 private adapter 的性能和质量；
- 估算实验耗时、读取量和候选顺序；
- 找出需要重新测量的边界。

它们不能直接证明：

- 8/10/12/16/20 GiB 任一 public target；
- App embedded 或 LTX worker 的 full-request peak；
- 低物理内存下比 swap 快；
- 合并 `dev` 后 ANE/default route 不回退。

## 2. 发布前的整体门控图

```text
R1 types/canonical digest
  ↓ host contract pass
R2 resolver/snapshot/authority
  ↓ fake execution + fault pass
R3 C ABI + Swift semantic parity
  ↓ default/off P0 pass
R4 App opt-in + job migration
  ↓ catalog empty staging pass
R5 model public adapter
  ↓ quality/lifecycle pass
R6 full-request calibration
  ↓ memory coverage + M + H(T) ≤ T
R7 performance P0/P1/P4 (P3 only if claimed)
  ↓ independent review
R8 compile-reviewed record
  ↓ staging/revoke/replay drill
Public release for exactly one model/workload/target
```

任一门失败都停止后续门。关闭 catalog 不能掩盖 default 路径已经产生的性能或生命周期回归。

## 3. 可回滚 PR 计划

### PR-R1：Identity 和 canonical encoding

修改：

- `native/core/streaming_public_types.hpp`；
- `native/runtime/streaming/canonical_encoding.*`；
- `native/runtime/streaming/preset_catalog.*`；
- resolver fixture tests、Python golden fixtures。

实现：

1. 增加 `PresetKey`、`SourceIdentity`、`WorkloadIdentity`、`RuntimeIdentity`、`DeviceIdentity`；
2. 将旧扁平 record 映射到 `Applicability/CanonicalPlan/Calibration/Performance/Release`；
3. 使用 length-prefixed、显式 optional tag、UTF-8 byte-order map 编码；
4. 生成 `record_digest`、`layout_digest`、`source_snapshot_digest` 的纯 host helper；
5. 在 catalog build 时拒绝重复 ID/revision、未知字段、未排序 map、缺失 release/review 字段。

完成门：

- C++/Python golden digest 一致；
- JSON key order 不影响 digest；absent/false/0 互不混淆；
- P/K/retention/source/runtime 变化能分别改变预期 digest；
- `make test-streaming-host` 和新增 `test-streaming-public-host` 通过；
- production catalog 仍为空；无 GPU、App、默认 runtime 改动。

回滚点：只回滚新增文件和 include；不改现有 manual v1 parser。

### PR-R2：Resolver、snapshot、authority

修改：

- `native/runtime/streaming/preset_resolver.*`；
- `native/runtime/streaming/resolved_request.*`；
- `native/runtime/session.hpp`；
- fake probe/snapshot/session test。

实现：

1. `probe_public_streaming` 只读取 metadata/index/token geometry；
2. resolver 按 model→source→workload→device/container→release→calibration→target 过滤；
3. 对 memory-tier 按 reviewed rank、calibrated bytes、logical reads、ID/revision排序；
4. 对 preset selector 做 exact replay，不允许近似匹配；
5. 只对最终一个候选调用 `compile_public_streaming`；
6. 对 compiled layout、component policy、runtime/reader/kernel identity 逐项比较；
7. 私有构造 `StreamingAuthority`，返回 immutable `ResolvedRequestExecution`；
8. 实现 source quick identity 与 catalog revoke 的 revalidation。

完成门：

- fake session 可完成 intent→resolve→snapshot；
- default `generate_resolved` 明确抛 `streaming_public_adapter_unsupported`；
- stale/revoked/source mutation/device mismatch 全部 fail-closed；
- resolver 零 GPU lock、零 pool、零 worker、零 ordinary generate；
- authority 不可 copy、不可 JSON deserialize、无公开 mint API；
- resolver host tests 全通过。

回滚点：保留现有 active selector gate；如果 resolver 不稳定，generate 继续返回 `streaming_preset_resolution_required`。

### PR-R3：Engine identity、C ABI 和 result

修改：

- `native/api/c_api.mm`；
- `bindings/c/include/turbocider/turbocider.h`；
- `native/platform/apple/results.mm`；
- `tests/native/test_streaming_public_api.py`。

实现顺序：

```text
tc_engine_create_model
  → set model_id/model_root

tc_engine_resolve_streaming_json
  → engine try-lock
  → parse/merge/validate
  → resolver (metadata only)
  → return exact selector + digest + summary

tc_engine_generate
  → off/private: old helper
  → public: parse → resolve again → global GPU lock → revalidate → generate_resolved
```

完成门：

- C ABI 结果/错误所有权、null pointer、busy、cancel 不变量通过；
- active selector 不再静默回退到 resident/manual；
- selector disabled 的 generate 字节/语义路径与历史相同；
- public test catalog 只能在 test target 注入；release binary 无 test setter；
- result 的 exact/actual digest、authority source、memory scope 字段存在且 off 时省略。

回滚点：保留旧 API；新 `tc_engine_resolve_streaming_json` 可独立关闭，active selector 回到明确拒绝，不回 resident。

### PR-R4：Swift/App 控制面

修改：

- `bindings/swift/TurboCiderNative.swift`；
- `apps/macos/StreamingOptions.swift`；
- `apps/macos/StudioState.swift`；
- `apps/macos/JobStore.swift`；
- `apps/macos/RunInsights*.swift`；
- Swift/integration tests。

实现顺序：

1. 先完成 v1/v2 semantic snapshot 和 explicit `CodingKeys`；
2. 增加 `StreamingChoice`，旧 draft decode 为 off；
3. 增加 options state/debounce/cache；
4. 将高级设置接到 target picker，不展示 P/G/K/D/Q 编辑框；
5. 增加 `NativeJob` 自定义 Codable 和 common accessors；
6. 提交时执行 resolve→persist→generate 事务；
7. RunInsights 展示 target/preset/scope/actual layout 和 unavailable reason。

完成门：

- streaming off 不发 options query，不增加 native calls；
- 旧 jobs 无损读取/写回；unknown future schema 不覆盖 jobs 文件；
- resolve/persist 失败零 GPU；
- stale/revoked 不自动换档；
- catalog 为空时 App 可运行且准确显示“未发布方案”；
- App release build 不引用 candidate API。

回滚点：隐藏 UI 开关并保留后端 selector fail-closed；旧 v1 jobs/生成路径必须继续可用。

### PR-R5：Calibration 和 catalog builder

修改：

- `tools/native/streaming_memory_schema.py`；
- `tools/native/calibrate_streaming_memory.py`；
- `tools/native/verify_streaming_calibration.py`；
- `tools/native/inspect_streaming_candidates.py`；
- `tools/native/explore_streaming_presets.py`；
- `tools/native/build_streaming_catalog.py`；
- synthetic tests。

实施要求：

- coordinator 不加载 native、不读取 MLX 统计；
- worker 自报自身 footprint、MLX/Metal counters 和 process identity；
- raw samples append-only JSONL，bounded batch，overflow 令校准失效；
- PID 使用 `(pid,start_identity,role)`；短命 helper 通过 launch/exit hooks补 coverage；
- fresh、warm、mode transition、release timing 分开运行；
- calibration verifier 独立重算同一时刻 tree peak，不能信任 summary 的 PASS；
- builder 默认只输出 `proposed`，`compile-reviewed` 要求 review digest。

完成门：

- 缺样本、时间 skew、PID 漏采、reset epoch 错误、post-hoc 删除全被拒绝；
- `M + H(T) <= T` 边界±1 byte 测试通过；
- 请求/读取/磁盘/停止预算超限会安全停止并 drain；
- 默认 native runtime 没有 sampler/thread/audit 新开销。

### PR-R6：模型 public adapter

按以下顺序，每个模型单独 PR、单独 candidate record：

1. Z-Image 或 LTX：验证 MLX/native descriptor 形态；
2. 另一模型；
3. H3 Turbo；
4. Flux 9B P0；
5. Flux 9B prefix P>0。

每个 PR 必须包含 probe、snapshot、generate_resolved、actual-plan check、full request lifecycle、模型专项 host tests 和
最小 real GPU smoke。不要在一个 PR 中同时修改 catalog release、kernel 优化和 App 默认值。

### PR-R7：Reviewed records

只提交：

- generated immutable catalog table；
- source/evidence digest；
- release channel/revocation metadata；
- App target label/reason mapping；
- record-specific acceptance report。

record PR 不修改 executor/kernel。回滚单个档位只需撤回 record，避免影响其他模型和 off/default path。

### PR-R8：合并 `dev` 和 ANE 回归

当前 `dev` 为 `02148b7`，包含 M5/ANE 优化。合并顺序：

1. R1–R4 至少完成并通过 default P0；
2. 记录 merge-base、两侧 build/source identity；
3. 解决 request/Swift/App/Device profile 冲突时保留 dev 的 validated ANE scope；
4. public v1 继续 GPU-only；
5. 重跑 default GPU/ANE、Z-Image partition routing、Core ML cache 和 selector-off App tests；
6. `gpu_ane + active public selector` 在无独立 record 时必须返回 `streaming_route_unsupported`。

## 4. 统一实验协议

### 4.1 Workload card 命名

每个 card 使用稳定 ID：

```text
<model>.<variant>.<operation>.<shape>.<frames>.<steps>.<feature-set>.<container>
```

首批建议：

| 模型 | Card | 首轮功能范围 |
|---|---|---|
| LTX | `ltx25.distilled.t2v.512x320x33.s11.dense.gpu.worker` | C/Metal、无 audio/I2V/LoRA/Sol |
| Z-Image | `zimage.original.t2i.512sq.s9.bf16.gpu.app` | original BF16、无 LoRA/ANE/GGUF |
| H3 Turbo | `h3turbo.original.t2v.512sqx22.s4.bf16.gpu.app` | original BF16、无 audio |
| Flux | `flux2.klein9b.t2i.512sq.s4.bf16.eager.gpu.app` | 9B BF16 eager、无 LoRA/ANE |

1024²、更多帧、不同 prompt token 上限、I2V/audio、LoRA、ANE 都建立新 card；不能从首卡外推。

### 4.2 每个 candidate 的固定生命周期

```text
environment preflight
  → source/build identity capture
  → clean baseline sample
  → engine construct
  → metadata resolve
  → model load/prepare (若该模式需要)
  → text/conditioning
  → denoise/transformer
  → VAE/upsample/audio/export
  → cancellation/drain checkpoint
  → unload/cleanup
  → final sample + result/quality verification
```

校准窗口从 construct 开始到 safe cleanup 结束。resident、streaming、swap/pressure 三者都必须使用同一 card、同一输出和同一
container 规则，不能只比较 denoiser 片段后宣称 full-request 优势。

### 4.3 运行次数和预算

首轮每 candidate：

- smoke：1 fresh；
- memory exploration：3 fresh + 10 warm；
- mode transition：2 resident→streaming→off；
- confirmation：至少 3 fresh + 10 warm；
- timing：按文档12的 ABBA policy；
- quality：至少 3 matched outputs；
- audit/fault：单独构建和请求数。

工具在启动前计算：总请求数、逻辑读取字节、预计墙钟时间、输出容量、最小磁盘空间。超出 policy 限额拒绝启动，不在运行中
无限扩展 candidate。`max_coarse_candidates=8` 不是整次 campaign 只运行8个请求。

## 5. 8/10/12/16/20 GiB 档位探索方法

### 5.1 目标和余量

对于目标 `T`：

```text
H(T) = max(512 MiB, ceil(10% × T))
eligible iff calibrated_request_bytes + H(T) <= T
```

因此 target 是“请求级校准目标”，不是物理显存大小，也不是 OS hard cap。catalog 只有在 exact card/device/container 上满足上述
条件后才能标记 available。

### 5.2 候选生成顺序

候选生成器先根据模型 descriptor 做合法性剪枝，再探索 Pareto 前沿：

```text
for each legal prefix P:
  for each legal group G:
    for each legal slot K:
      for each legal prefetch D:
        for each legal reader Q:
          compile layout
          estimate resident/prefix/suffix/activation floor
          estimate logical reads and pass transitions
          reject if component floor > target
          retain non-dominated candidates
```

不允许的做法：

- target 不足时运行期偷偷减 K；
- 通过改变 dtype/quantization 伪造更低档；
- 运行中动态扩大 slot；
- I/O 慢时回退 resident；
- 把系统 swap、page cache 或一次偶然低峰值写进 record。

### 5.3 各模型的首轮候选族

#### LTX

- 优先探索 prefix `P=1,4,8,12,16`；
- `G=1` 固定；
- `K=2,3` 分开；
- `D=1,2`，`Q=1,2,3` 只在合法 K 范围内；
- P8/G1/K3/D2/Q3 仅为现有同布局 anchor；
- 低 target 先测 P1/K2，再测 P4/K2、P8/K2；
- 48 block 与 suffix `48-P >= K` 约束必须在 compiler 阶段拒绝非法布局；
- audio/I2V/97 frames 作为新 component/workload card。

#### Z-Image

- 固定 `G=1,K=2,D=0,Q=1`，先仅探索 P；
- 候选 P：`0,4,8,12,14,18,22,26,28`；
- P29 对当前 30 blocks/K2 拒绝；
- 512² 和 1024² 分开；
- old 6/8/10/12 GiB denoiser budget 不可映射为 public target；
- VAE release/pool policy 变化时 bump component revision。

#### H3 Turbo

- 固定 `G=1,K=2,D=1,Q=1`；
- P：`0,2,4,8,12,16,24`，每个 P 都检查 suffix carry；
- 保持 `carry_first_group` 和大矩阵单次大 `pread`；
- 22/39/73 frame 分开 card；
- 只做 MiniMax H3 Turbo original BF16，不做普通 H3/量化/FastH3。

#### Flux 9B

- 第一阶段只做 P0/G1/K2/D1/Q2，确认较高 target full request record；
- 第二阶段实现 prefix P2/P4/P6/P8/P12；
- P7 因 dual suffix 不足 K2 拒绝；
- P8 消除空 dual suffix pool，P12 验证 block8 concatenate 在 prefix 内只发生一次；
- P30 只作边界测试；P31/32 不作为 streamed record；
- 9B eager 和 4B compiled graph 完全分开。

### 5.4 从数据到 target record

每个 candidate 计算：

```text
M_confirmed = max(complete full-request observed peaks)
M_catalog   = max(M_confirmed, complete stage/prediction upper bound)
for T in [8,10,12,16,20] GiB:
    if M_catalog + H(T) <= T:
        candidate is memory-eligible for T
```

若多个 candidates 对同一 target eligible：

1. 先按 P0/P1/P4 性能结果筛掉回归或质量失败；
2. 再按 performance rank；
3. 再按 calibrated bytes、logical reads；
4. 最后按 ID/revision；
5. 将选中的 canonical layout 写入 record。

memory-eligible 不等于性能最优；若最低档可行但 wall tail 极差，可以保持 unavailable 或标记 experimental，而不是无条件发布。

## 6. 模拟、实测和 swap 对照

### 6.1 离散事件模拟

模拟器输入：

- 每个 group 的 source ranges/bytes；
- P/G/K/D/Q、pool class、retention、pass transition；
- read service time 分布（冷 SSD/热 page cache 分开）；
- kernel/activation 时间分布；
- reader completion 和 fence；
- component live interval。

输出：

- 预计 wall/denoise；
- slot occupancy 和 I/O worker utilization；
- refill wait、prefetch hit/miss；
- predicted peak backing bytes；
- 对 target 的可行性和敏感性。

模拟只用于 candidate 剪枝，不能生成 public record。held-out candidate 的 wall 中位误差目标≤10%、P95≤20%；不达标仅作 sensitivity 图。

### 6.2 真实 swap 对照的三层顺序

1. **无压力**：resident vs exact streaming，建立质量、wall、I/O 和 peak；
2. **自然低内存**：选择真实可获得的较低统一内存设备（例如 8/16/24 GiB 等实际配置），在其上只运行 native 判定可用的
   8/10/12/16/20 GiB 请求目标，记录 footprint、pressure、swap delta、fault、tail；物理内存不要求等于 target；
3. **受控 pressure（单独授权）**：冻结 helper 上限、timeout、停止和 cleanup，不能关闭系统 swap，也不能使用危险无限分配。

每次比较必须包含：

```text
success_rate
wall median/P95
denoise median/P95
observed process-tree peak
swapin/swapout delta
major/minor fault delta
GPU/CPU/SSD utilization
thermal/pressure state
output quality
```

如果 streaming 没有 swap 但 wall 更慢，结论是“减少 paging、速度代价为 X”，不能写成“更快”。只有 wall speedup 的 95%区间下界
大于 1、质量和成功率不差，才可宣称在该 card/设备上更快。

### 6.3 “比单纯 swap 更快”的可证伪假设

预注册以下假设：

| 假设 | 观测 | 可能结论 |
|---|---|---|
| H1：streaming 减少随机 page fault | major fault、swap delta、I/O pattern | 支持/否定 paging 机制优势 |
| H2：prefetch overlap 降低 GPU wait | refill wait、GPU idle、Q/D sweep | 支持/否定 overlap 设计 |
| H3：显式 component release 降低全请求 peak | stage live interval、tree peak | 支持/否定生命周期收益 |
| H4：系统 swap 在压力下 tail 更差 | wall P95、timeout、失败率 | 只能在同设备/压力协议下成立 |
| H5：streaming 的额外读取抵消了收益 | logical reads、SSD bandwidth、wall | 可能得到“更稳但不更快” |

实验失败时保留 raw evidence，不调高 target 或改变统计口径来得到预期结论。

## 7. 性能验收与零回归

### 7.1 P0：默认路径

每次 R1–R8 合并后都要比较：

- default resident；
- legacy streamed（若该模型已有）；
- default GPU+ANE（dev merge 后）；
- engine create/load/generate/unload；
- cold/warm request。

阻断线沿用文档12：median ≤1.02、P95 ≤1.05；bootstrap interval 上界超过门槛即 FAIL，点估计在范围内但区间跨界为 INCONCLUSIVE。
控制面 off 的新增 audit：framework hook、probe、worker、pool、cache clear/unload 必须为零或明确解释为旧路径已有计数。

### 7.2 P1：同布局框架成本

baseline 与 candidate 使用：

- 完全相同 P/G/K/D/Q；
- 相同 source、dtype、kernel、reader、component policy；
- 相同 lifecycle、cold/warm、输出和 seed；
- 只改变 generic executor/direct adapter。

P1 通过才说明框架成本可接受；不能用改变 prefix/K/retention 后的收益补偿 P1 回归。

### 7.3 P4：策略取舍

比较不同布局时必须单独标 `comparison_kind=strategy_tradeoff`，同时展示：

- peak 降幅；
- logical reads 增幅；
- wall/denoise 变化；
- tail/timeout；
- slot/pool/worker 利用率；
- quality/失败率。

P4 不使用统一“≤2%”承诺；如果要对外宣称加速，wall speedup 95%区间下界必须 >1。

### 7.4 日常路径控制面预算

工程目标（需实测冻结，不是当前成绩）：

| 操作 | 目标 |
|---|---:|
| streaming off 新增控制面 | 无可测 wall 回归；audit 新增为0 |
| cached options | P95 < 10 ms，主线程无同步I/O |
| uncached metadata options | P95 < 100 ms，不加载权重 |
| exact resolve | 有界 metadata/tokenizer，单列耗时 |
| block hot path | 0 catalog lookup、0 JSON、0 allocation、0 thread create |
| sampler build | 不进入 release timing binary |

## 8. 生命周期和故障验收

### 8.1 Fault matrix

每个 public adapter 至少注入：

```text
probe missing metadata
compile identity mismatch
source changed after resolve
catalog revoked before execute
pool allocation failure
first fill failure
mid-pass fill/read failure
GPU reader/fence timeout
cancel before first dispatch
cancel during prefetch
cancel during GPU encode
VAE/export failure
mailbox overflow
worker join timeout
unsafe destructor/drain
```

每项都验证：

- stable error code；
- no success result；
- no active worker/reader/ticket；
- source/snapshot lifetime正确；
- unsafe 状态进入 quarantine；
- 后续请求被拒绝或要求重建 engine；
- 原因和 raw evidence 被保留。

### 8.2 模式切换矩阵

```text
resident → public streaming → resident
legacy streamed → public streaming → legacy streamed
public target A → public target B → A
public cancel/fail → off/default
App embedded → LTX worker → App embedded
```

切换过程中记录 cache clear/unload、worker/thread、source lease、GPU synchronization 和 wall 成本。切换成本必须进入 full-request
报告，不能藏在 UI resolve 或 benchmark warmup 之外。

## 9. 四模型 public record 验收卡

### 9.1 LTX 2.5 distilled

必须：

- 仅 C/Metal dense T2V；
- 48-block descriptor、Gemma token shape、connector/stage1/stage2/upsampler/VAE/export identity完整；
- worker container 独立 calibration；
- 33 frames 和 97 frames 分开；
- audio/I2V/LoRA/Sol/ANE明确 unavailable；
- current P8/G1/K3/D2/Q3 anchor重新做 full-request memory/quality确认。

最低 release evidence：

```text
LTX-FUNC: 20 matched successful requests
LTX-MEM: 3 fresh + 10 warm + complete tree coverage
LTX-P0/P1: documented interval status
LTX-FAULT: first-fill/cancel/worker-kill/drain
LTX-APP: ordinary worker route + replay
```

### 9.2 Z-Image Turbo

必须：

- original BF16、GPU、T2I；
- no GGUF/LoRA/ANE；
- tokenizer valid/padded/compute rows进入 workload key；
- 512²、1024²分别建 record；
- P29 等非法边界被拒绝；
- dev 合并后的 encoder manifest/image weight release 逻辑重新验证；
- old 6/8/10/12 GiB denoiser budget 与 public target 分开报告。

最低 release evidence：

```text
ZIMG-FUNC: byte/quality matched outputs
ZIMG-MEM: full request through VAE/cleanup
ZIMG-P0/P1: default and same-layout intervals
ZIMG-APP: options/replay/unsupported reason
```

### 9.3 H3 Turbo

必须：

- 只 MiniMax H3 Turbo original BF16；
- K2/G1/D1/Q1、carry_first_group、四 pass 和大矩阵 pread identity固定；
- 22/39/73 frames 分别验证；
- full Qwen3 text→DiT→video VAE→MP4；
- ordinary public constructor 使用 authority，不永久开启 private bool；
- existing bootstrap INCONCLUSIVE 如实写入 evidence。

H3 的“工程上合理波动”只能写成 release note 或 experimental exception，不能转换为统计 PASS，也不能扩到普通 H3。

### 9.4 Flux 9B

必须：

- 9B BF16 eager、two-shard、GPU；
- P0/G1/K2/D1/Q2 先做较高 target full request；
- P>0 前先完成 prefix pager、block8 concatenate、pool elimination 和 P7 rejection tests；
- 4B compiled graph、LoRA、ANE 不共享 record；
- 两步输出与 resident/reference 一致；
- 记录 prefix source bytes 和每 pass prefix compute 次数。

已有 resident→streaming MLX peak 下降和 P1 同布局结果只能作为 candidate anchor，不能直接编入 catalog。

## 10. Evidence bundle 和 review

每条 candidate/record 必须提交：

```text
policy.json
source-identity.json
build-identity.json
artifact-manifest.json
environment.json
workload-card.json
candidates.json
rejections.jsonl
raw-native-results.jsonl
memory-samples.jsonl
memory-summary.json
quality.json
lifecycle.json
audit.json
performance.json
pareto.json
catalog-proposal.json
verifier-report.json
review.json
```

reviewer 必须能从 bundle 复算：

1. candidate 的 exact layout 和 source identity；
2. 同时刻 process-tree peak；
3. `M + H(T) <= T`；
4. P0/P1/P4 统计区间；
5. 失败率、质量、swap/pressure 状态；
6. release channel、review commit、record digest；
7. 不存在 post-hoc 删除失败样本或切换 estimator。

缺少任一 required field → `incomplete`，builder 不得输出 public-stable。

## 11. Staging、发布、撤回和回滚

### 11.1 Staging

先在 App staging build 中加入 `public-experimental` record：

- UI 只显示 native options 返回的 target；
- 一台设备、一种 container、一张 workload card；
- 结果中输出 exact/actual digest；
- telemetry 记录 code/status/target，不记录 prompt/path；
- 支持一键关闭该 record 的 catalog feature flag；
- 不改变 off/default 和 legacy streamed。

### 11.2 最小 public release

第一次 release 只放：

- 一个模型；
- 一个 variant；
- 一个 execution container；
- 一个 workload card；
- 一个或少量经过独立确认的 target；
- `public-experimental` 或 `public-stable` 明确标签。

没有必要等五档都齐全，也不能为了“8–20 GiB 全覆盖”填入未测记录。不同物理内存机器只会看到本机可用且 exact-compatible 的 target。

### 11.3 撤回

触发条件：

- 新 build/runtime/reader/kernel identity 不匹配；
- artifact/source 变化；
- memory verifier 发现 coverage 或 peak 错误；
- success/quality/performance regression；
- unsafe cleanup/quarantine；
- user-visible crash 或大量 unavailable mismatch。

撤回操作：

1. 将 record 标记 revoked；
2. 递增 catalog revision；
3. 新请求返回 `preset_revoked` 或重新 query 显示 unavailable；
4. 历史 exact replay 失败，不自动换 preset；
5. 保留 evidence 和失败样本；
6. 只回滚 record，必要时再回滚 adapter PR。

## 12. 运行命令与验收映射

当前已存在并应在每个相关 PR 后运行：

```sh
make test-streaming-host
make test-streaming-contract
make test-streaming-campaign
make test-streaming-source-identity
make test-streaming-audit
make test-streaming-pager
```

建议新增：

```sh
make test-streaming-public-host       # types/catalog/digest/resolver
make test-streaming-public-api        # C ABI/authority/gate
make test-streaming-public-app        # Swift Codable/job/options mocks
make test-streaming-calibration      # fake PID/time/tree/verifier
make test-streaming-public-lifecycle  # fault/quarantine/mode transition
make test-streaming-public-model MODEL=... WORKLOAD=... TARGET=...
```

真实模型命令必须显式传 MODEL、WORKLOAD、TARGET、OUTPUT、POLICY 和 evidence 目录。默认 `make test` 不下载权重、不启动 pressure、
不隐式运行 public GPU campaign。

## 13. Definition of Done

### 框架层

- [ ] off/default 路径通过 P0，新增 audit 为零；
- [ ] manual v1 行为和 ABI 保持兼容；
- [ ] selector→resolve→authority→snapshot→generate_resolved 闭环；
- [ ] authority 不可伪造/反序列化；
- [ ] stale/revoke/source mutation/device mismatch fail-closed；
- [ ] snapshot、worker、reader、slot、mailbox 生命周期通过 fault tests；
- [ ] catalog builder/reviewer/revoke/rollback 可追溯；
- [ ] Swift/App job migration 无损；
- [ ] GPU-only/public 与 ANE/default 路径隔离。

### 单模型/单 target

- [ ] exact source/workload/container/device/runtime identity；
- [ ] full-request quality 和成功率；
- [ ] memory tree coverage complete；
- [ ] `M + H(T) <= T`；
- [ ] P0/P1/P4（需要 swap 宣称时还要 P3）；
- [ ] cancellation/read failure/drain/quarantine；
- [ ] ordinary App/worker replay；
- [ ] evidence/review/catalog digest；
- [ ] revoke drill。

### 产品层

- [ ] 高级开关默认 off；
- [ ] 只显示 native 返回的 target，不显示 P/G/K/D/Q 编辑器；
- [ ] 推荐与资格分离；
- [ ] unavailable reason 可理解且 stable code 驱动；
- [ ] 结果显示 target/preset/scope/actual layout；
- [ ] 不承诺 hard cap、零 swap、无条件提速；
- [ ] 无记录时 App 和 engine 都安全 fail-closed。

## 14. 给实现团队的最终判断

这套方案可行的关键不是把更多参数暴露给用户，而是把“用户目标”和“内部 exact layout”彻底分开：

```text
用户选择内存目标
  → native 根据已审核 record 选择 exact P/G/K/D/Q
  → immutable authority 绑定 source/workload/device/runtime
  → 复用现有 slot/block executor 做有界 streaming
  → 用 full-request 证据决定是否在该设备/档位公开
```

在内存充足机器上，默认路径不访问这条链，因此性能不会因 public 功能本身下降；用户主动选择 streaming 时，性能和峰值由具体
record 的 P0/P1/P4 数据决定。在低内存机器上，显式 slot/block streaming 很可能比不可预测的系统 swap 更稳定，并可能更快，
但只有在同一 workload、同一设备和冻结 pressure protocol 下测出统计证据，才能对外写“更快”。

当前距离 public 的实际剩余工作是 R1–R8 和至少一条完整 model record；控制面提交和文档完善本身不改变这个判断。
