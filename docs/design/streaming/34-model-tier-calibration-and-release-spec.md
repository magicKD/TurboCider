# 34 · 四模型档位校准、性能对照与 Public Release 规格

[目录](README.md) · [Runtime/App 工程规格](33-public-runtime-app-engineering-spec.md) · [配置与校准手册](31-public-streaming-config-calibration-runbook.md) · [收口规格](32-public-streaming-completion-spec.md) · [当前进度](13-implementation-progress.md)

日期：2026-09-17。状态：**校准与发布规格；尚未生成 reviewed production record。**

本文回答“用户只选择 8/10/12/16/20 GiB，后台如何为不同模型选择 layout/preset”这一问题，并把候选探索、完整请求内存测量、swap 对照、review、catalog 发布和撤回细化为可执行流程。

## 1. 产品抽象：用户选目标，后台选计划

### 1.1 用户看到的模型无关选择

~~~text
Streaming：关闭
Streaming：自动推荐
Streaming：8 GiB
Streaming：10 GiB
Streaming：12 GiB
Streaming：16 GiB
Streaming：20 GiB
~~~

首版 App 可只显示“关闭”与 native 返回的 available targets；“自动推荐”是后续 opt-in，不得在没有用户明确同意时改变默认路径。

用户永远不直接填写：

- `resident_prefix_blocks`（P）；
- `block_group_size`（G）；
- `slot_count`（K）；
- `prefetch_distance`（D）；
- `io_workers`（Q）；
- pass transition；
- multi-pool policy；
- component release policy；
- artifact chunk size。

### 1.2 后台的确定性选择

native 根据完整 workload 和设备 identity 查找 reviewed record：

~~~text
target + model + operation + shape + frames + steps + audio + inputs
      + source variant + dtype + runtime + device + execution container
  → filter exact eligible records
  → rank by release/performance/calibrated bytes/read bytes/revision
  → select one record
  → issue exact selector
~~~

没有 exact record 时返回 unavailable；不得根据 target 反推出一个未经校准的 K/G/D/Q，也不得自动切换到 legacy memory budget、resident 或系统 swap。

## 2. Target、margin 和 eligibility 语义

### 2.1 五个 target

固定支持：

~~~text
8 GiB  =  8 × 2^30 bytes
10 GiB = 10 × 2^30 bytes
12 GiB = 12 × 2^30 bytes
16 GiB = 16 × 2^30 bytes
20 GiB = 20 × 2^30 bytes
~~~

这些是“完整请求进程树的校准目标”，不是 GPU VRAM hard cap、系统物理内存上限或 swap 上限。

### 2.2 Guard margin

首版统一：

~~~text
H(T) = max(512 MiB, ceil(10% × T))
eligible(T) = complete_calibrated_peak + H(T) <= T
~~~

其中 `complete_calibrated_peak` 必须来自完整请求观测；模型估算只能用于候选剪枝，不能直接填入 production record。

### 2.3 物理内存推荐

推荐只在设备物理内存足够且 native 返回 available 时生效：

~~~text
device_headroom = max(2 GiB, ceil(15% × physical_memory))
candidate_targets = available targets where T <= physical_memory - device_headroom
recommended = max(candidate_targets)
~~~

如果物理内存低于最小 target，UI 仍显示 Off；不要把“8 GiB target”解释为 8 GiB 机器一定可运行。

## 3. Workload card：每个 record 的闭包

一个 record 只能覆盖一个明确的 workload card。至少包含：

| 类别 | 字段 |
|---|---|
| 模型 | model ID、variant、checkpoint manifest digest、dtype、quantization |
| 操作 | operation、输入类型、batch、audio、LoRA、approximation |
| 形状 | width、height、frames、fps、steps、token shape、padding |
| 运行时 | TurboCider build、adapter、reader、kernel、allocator revision |
| 设备 | GPU name、family、OS build family、physical memory class |
| 容器 | embedded app / ltx worker / CLI worker |
| 布局 | P/G/K/D/Q、component policy、pass transition、multi-pool |
| 证据 | calibration ID、sample count、evidence digest、review commit |

Workload card 不允许使用“任意尺寸”“所有 H3”“Flux 全系列”这种模糊范围。扩大 shape、steps、audio 或 input 后必须新建 card 和 record。

## 4. 候选布局生成

### 4.1 候选维度

对每个模型生成有限候选族：

~~~text
P ∈ model-specific prefix set
G ∈ descriptor block grouping set
K ∈ {1, 2, 3}
D ∈ [0, K-1]
Q ∈ [1, K]
multi_pool ∈ descriptor-approved set
component_policy ∈ descriptor-approved set
~~~

候选必须先通过 layout compiler：

- block/group 覆盖完整；
- source range 不重叠且在 artifact 内；
- K/D/Q 约束合法；
- pool class 与 component policy 一致；
- retention 和 pass transition 明确；
- logical read bytes、slot bytes、resident bytes 可计算。

### 4.2 候选剪枝顺序

1. 解析/descriptor 不合法：丢弃；
2. source range、shape、dtype 不匹配：丢弃；
3. 明显 `resident_bytes + workspace_lower_bound + H(T) > T`：丢弃；
4. simulator 预测 refill wait 超过当前最佳 2 倍：标记低优先级，不直接丢弃；
5. 不能通过 no-overlap、cancel、short-read、stale-source fake tests：丢弃；
6. 仅保留每个 target 的 Pareto 前沿：peak、wall estimate、logical reads、worker count。

### 4.3 不允许的自动“修正”

候选或 manual layout 被拒绝时，不自动：

- 缩小 K；
- 改小 G；
- 减少 prefix；
- 改 dtype；
- 跳过 block；
- 改变 pass 顺序；
- 回退 resident；
- 启用系统 swap 作为隐式 fallback。

这些变化都会改变 layout identity，必须成为新的 candidate 或显式错误。

## 5. 四模型候选族

以下是第一轮探索边界，不是已经公开的 record。

### 5.1 LTX 2.5 Distilled

首批限制：原始 dense BF16、GPU-only、明确 video workload card、`ltx_worker` 或 embedded container 分开认证。

建议候选族：

| 组件 | 设计 |
|---|---|
| Denoiser | 先重放已冻结的 P8/G1/K3/D2/Q3 private 锚点；再探索 K2/K1 低槽位、低并发候选，G 以 transformer block group |
| VAE/upsampler/vocoder | 组件级 staged release；不得在 denoise 与 finalizer 之间保留无证的整个模型 |
| Worker | public authority 在 worker 内 resolve；App 仅传 exact selector |
| Audio | audio on/off 为不同 workload card，不能共享同一 record |
| I2V | first-frame encode 单独 card，不能复用 T2V record |

必须重点测量：worker 进程峰值、VAE/音频最终化峰值、reader drain、进程退出是否释放所有 file descriptor 和 Metal resource。

### 5.2 MiniMax H3 Turbo

首批限制：只支持 **H3 Turbo original BF16**、GPU-only、固定 50 streamed blocks、4-pass DiT；普通 H3、VDN、K2/G1 之外布局、量化和 ANE 不在本轮。

建议候选族：

- P0/G1/K1/D0/Q1：最低槽位和并发，作为内存/无 overlap 对照；
- P0/G1/K2/D0/Q1：当前已具备 private exact 工程锚点；
- P0/G1/K2/D1/Q1：当前冻结 private 工程锚点，并用于测试 prefetch window；
- P0/G1/K2/D1/Q2：只在真实设备上证明 Q2 不增加 allocator/thread churn 后保留；
- G>1：仅在 block fusion 不改变数学顺序且 descriptor 能提供完整 range table 时探索。

H3 的 carry-first-group 是跨 pass 的布局语义，不能被 generic resolver 当作普通 K2 重载；record 必须显式写入 `pass_transition`。

### 5.3 Z-Image Turbo

首批限制：original BF16、GPU-only、image.generate、固定 token shape/steps card；GGUF、量化、GPU+ANE 和任意文本长度不共享 record。

建议候选族：

- 单 component denoiser block/group streaming；
- P14/G1/K2/D0/Q1 作为已验证 private 锚点的 replay 候选；
- K1 作为低槽位/低并发对照；logical read bytes 仍按完整 block replay 计算；
- K2/D1 只在 refill wait 明显下降且完整请求 peak 不超目标时保留；
- text encoder、VAE 输出阶段的 resident policy 必须记录在 component policy 中。

Z-Image 的 token count 影响 workload identity。提示词长度变化超出 card 必须返回 unavailable，不得截断或复用错误 record。

### 5.4 Flux.2 Klein 9B

首批限制：BF16 eager GPU、无 LoRA、Diffusers 两分片、Flux 4B compiled graph 不纳入。

建议候选族：

- dual block pool：K2，G1，D0/1，Q1/2；
- single block pool：K2，G1，D0/1，Q1/2；
- `multi_pool_policy=retain_all`，dual→single→dual 跨 pass 不重建 pool；
- direct same-layout replay 只用于 P1 框架开销，不是 production execution implementation；
- image input card 与 text-only card 分开，因 image encode 会改变 component peak。

record 必须同时覆盖 dual/single 两类 pool 的 `layout_digest`、pool count、slot bundle count 和实际 pass transition。

## 6. 完整请求校准协议

### 6.1 采样窗口

每个 candidate 的一个样本从独立干净进程开始，覆盖：

~~~text
process launch
→ model construction
→ source/metadata resolve
→ tokenizer/conditioning
→ denoise all passes
→ VAE/upsampler/audio/export
→ GPU fence
→ reader drain
→ cleanup
→ worker exit（如适用）
~~~

不得只采 denoise 阶段或只看 MLX allocator peak。

### 6.2 需要同时记录的内存域

| 域 | 采样 |
|---|---|
| process tree | parent + worker RSS/phys_footprint peak |
| MLX | active/peak allocator bytes、cache bytes |
| Metal | device allocation/heap peak、command buffer completion |
| Core ML | 若 route 包含 ANE，记录 model/runtime allocation；public v1 暂拒绝 |
| I/O | logical bytes、physical bytes、read calls、short reads |
| OS pressure | compressed memory、page faults、swap in/out |
| runtime | pools、slot bundles、workers、allocations、cache clear/unload |

### 6.3 采样质量

每个 candidate 至少：

- warmup 2 次（不计入统计）；
- ABBA 或 BAAB 成对执行；
- 20 对成功样本作为工程最低门；
- 40/40 请求成功；
- 20/20 输出 byte-exact 或通过模型专属 quality equivalence；
- 失败、取消、short-read、source mismatch 各至少一次 fault drill；
- 确认最大样本 gap 和 sampler coverage。

若设备处于 thermal throttling、memory pressure、后台编译或用户交互状态，样本必须标记并按 policy 处理，不得静默混入。

### 6.4 Target 判定

对每个样本计算：

~~~text
peak_i = max(process_tree_peak, mlx_peak, metal_peak, coreml_peak)
calibrated_peak = percentile_or_upper_bound(peak_i, policy)
eligible(T) iff calibrated_peak + H(T) <= T
~~~

`scope` 必须明确是 `full_process_tree_request`、`embedded_process_tree` 或 `ltx_worker_tree`。不同 scope 的数字不能直接比较。

## 7. Resident、Streaming、Swap 三路实验

### 7.1 四路矩阵

| 路线 | 目的 |
|---|---|
| resident / no pressure | 默认性能基线 |
| public streaming / no pressure | 显式搬运代价和 peak |
| resident / controlled pressure | 被动压缩/系统 swap 参考 |
| public streaming / controlled pressure | 低内存收益 |

若 legacy streamed 存在，可作为第五路诊断，但不能代替 public exact route。

### 7.2 必须报告

- wall median/P95；
- denoise median/P95；
- full process-tree peak；
- MLX/Metal peak；
- swap in/out bytes；
- compressed memory/page faults；
- logical/physical I/O；
- GPU busy/idle；
- refill wait 和 reader completion；
- success/cancel/failure rate；
- output equality/quality；
- thermal/pressure state。

### 7.3 “比系统 swap 更快”的严格结论

只能在 controlled pressure 下比较 `public_streaming` 与 `resident+swap`：

~~~text
speedup = wall_swap / wall_streaming
~~~

只有 speedup 的 95% CI 下界 > 1，且两路成功率、质量、scope、pressure state 可比时，才可以声称“更快”。

如果结果是：streaming peak 更低但 wall 更慢，应准确写成“更可控/更少 swap/成功率更高”，不能写成“更快”。

### 7.4 Swap 实验的安全边界

- 只在专用测试机器或明确可恢复的虚拟内存压力 harness 中执行；
- 不修改用户持久数据；
- 不使用不可逆的系统级禁用 swap 作为默认测试步骤；
- 每次实验前后记录 OS pressure、swap counters 和可用磁盘；
- pressure 注入失败时标记 invalid，不把无压力样本当 swap 对照。

## 8. 仿真器与工具链

### 8.1 工具分层

~~~text
descriptor inspector
  → layout compiler
  → discrete-event schedule simulator
  → fake pager/slot fault harness
  → real-model calibration runner
  → independent verifier
  → catalog builder/reviewer
~~~

### 8.2 Simulator 输入

- canonical layout；
- 每 block/group source bytes；
- K/D/Q；
- GPU compute duration distribution；
- read latency/throughput distribution；
- pool class/pass transition；
- workspace and resident bytes；
- cancellation/fault injection。

### 8.3 Simulator 输出

- predicted peak；
- refill overlap ratio；
- executor wait；
- logical/physical reads；
- slot reuse violations；
- queue starvation；
- cleanup pending count；
- candidate ranking。

Simulator 只能剪枝和解释，不可替代实体 GPU 证据。

### 8.4 建议 CLI

~~~sh
tools/streaming/inspect_descriptor --model ... --json
tools/streaming/compile_layout --model ... --target 12GiB --out candidate.json
tools/streaming/simulate_schedule --candidate candidate.json --trace trace.json
tools/streaming/run_campaign --policy policy.json --out results/
tools/streaming/verify_campaign --policy policy.json --results results/
tools/streaming/build_catalog --proposal proposal.json --out catalog.json
tools/streaming/review_catalog --catalog catalog.json --evidence evidence/
~~~

所有工具输出必须包含 tool version、git commit、input digest、runtime/device identity 和 schema version。

## 9. Evidence bundle

每个候选/record 一个目录：

~~~text
evidence/<model>/<workload>/<target>/<candidate>/
  request.json
  exact_selector.json
  descriptor.json
  layout.json
  source_identity.json
  runtime_identity.json
  device_identity.json
  samples.jsonl
  memory_trace.jsonl
  pressure_trace.jsonl
  output_hashes.json
  faults.json
  audit.json
  environment.json
  verifier.json
  policy.json
  manifest.sha256
~~~

Evidence 必须脱敏：不上传 prompt 原文、用户路径、完整模型路径或个人文件名；使用 digest 和相对 artifact path。

## 10. Catalog builder 与 review 状态机

~~~text
candidate
  → simulated
  → real_calibrated
  → independently_verified
  → proposed
  → reviewed
  → staged
  → production
  → revoked
~~~

状态转换规则：

- `simulated → real_calibrated`：完整请求采样齐全；
- `real_calibrated → independently_verified`：独立 verifier 重跑通过；
- `independently_verified → proposed`：record digest 与 evidence digest 固定；
- `proposed → reviewed`：runtime/model/performance/release owner 签字；
- `reviewed → staged`：catalog diff、rollback、App options 验证；
- `staged → production`：灰度期间无回归；
- 任意 production record 可 `revoked`，但历史 job 仍保留原 record identity。

## 11. Release gate

### 11.1 框架 gate

- host/compiler/executor/canonical/resolver tests 全绿；
- C ABI null/busy/ownership/cancel 全绿；
- App transaction、migration、stale options 全绿；
- streaming off audit 零新增工作；
- latest dev merge 后 default GPU/GPU+ANE/ANE cache 回归通过。

### 11.2 模型 gate

- probe/snapshot/source lease/generate_resolved 已 override；
- actual plan hard verification；
- cancel、short read、stale source、worker exit；
- 质量和 output hash；
- 不依赖 private `allow_experimental_streaming`；
- 不改变默认 resident/compiled/ANE 路线。

### 11.3 Target gate

- target eligibility 通过；
- 20 对性能样本和 40/40 成功；
- P0 默认路径 ≤2% median、≤5% P95 工程门；
- P1 同布局框架成本通过；
- P3 swap 对照有完整 pressure evidence；
- review/revoke/replay drill 通过。

## 12. 撤回与回滚

触发条件：

- 任一 shape 发生 actual-plan mismatch；
- source/runtime identity 漂移；
- 完整请求 peak 超 target；
- 失败率或取消 cleanup 回归；
- 默认路径性能超过门槛；
- output quality/hash 回归；
- OS/runtime 更新导致 device identity 不再匹配。

撤回动作：

1. 将 record 标记 `revoked=true`；
2. catalog revision 递增；
3. options 立即返回 unavailable；
4. 正在运行的 request 不强杀，由 authority/source lease 决定是否完成；
5. 新提交不得使用 revoked record；
6. 历史 job 保留原 resolution/result；
7. 记录回滚 commit 和 review digest。

## 13. 当前四模型状态（事实边界）

截至本文日期，已知事实只能写成：

| 模型 | Private exact/candidate | Public hook | Reviewed target | Production |
|---|---:|---:|---:|---:|
| LTX 2.5 | 有冻结 candidate tuple | 未完成 | 无 | 无 |
| H3 Turbo | 有 K2/G1 工程验收 tuple | 未完成 | 无 | 无 |
| Z-Image Turbo | 有冻结 candidate tuple | 未完成 | 无 | 无 |
| Flux.2 Klein 9B | 有 private BF16 exact/P1 tuple | 未完成 | 无 | 无 |

因此 App 不能把任何 8/10/12/16/20 GiB 显示为已支持；native options 当前应为 unavailable 或 tentative。

## 14. 实施顺序

1. 完成 [33](33-public-runtime-app-engineering-spec.md) 的 runtime validator、C ABI、source lease、result 和 App transaction；
2. Z-Image 接 public hooks，完成一个小 workload card 的 end-to-end 骨架；
3. Flux 9B 接 dual/single multi-pool 和 actual-plan verifier；
4. H3 Turbo 接 carry-first-group、worker/reader audit；
5. LTX 接 worker-local resolve、组件级 peak 和 finalizer lifecycle；
6. 实现 inspector/compiler/simulator/campaign/verifier/catalog builder；
7. 逐 target 做完整请求 calibration 和 resident/streaming/swap 对照；
8. 每次只将一个 reviewed record staged/production；
9. 最后合并最新 dev，做 default/ANE/M5 回归；
10. 完成 public release checklist 后才在 App 开启对应 target。

## 15. 验收签字单

### Runtime owner

- [ ] off fast path 没有 public helper/counter；
- [ ] authority/source lease/actual plan 全部硬约束；
- [ ] cleanup/quarantine 逻辑可证明；
- [ ] C ABI/Swift schema 稳定。

### Model owner

- [ ] descriptor/source range/token shape 固定；
- [ ] probe/snapshot/generate_resolved 已接通；
- [ ] private candidate 与 public record 身份隔离；
- [ ] quality/lifecycle/fault evidence 齐全。

### Performance owner

- [ ] P0/P1/P3 统计报告和置信区间；
- [ ] peak scope 明确；
- [ ] swap pressure 真正生效；
- [ ] 没有用 denoiser-only 或 simulator 数字替代 full request。

### App owner

- [ ] 默认 Off；
- [ ] 只显示 target，不显示 P/G/K/D/Q；
- [ ] options stale response 不覆盖新 query；
- [ ] resolve→persist→generate 事务原子；
- [ ] revoked/stale/mismatch 不静默 fallback。

### Release owner

- [ ] catalog diff 可审阅、可撤回；
- [ ] record/evidence/review digest 一致；
- [ ] history/replay 行为明确；
- [ ] production 前保留空 catalog 或仅加入已签字 record。
