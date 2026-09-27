# 45 · Streaming 验收追踪、性能实验与 Evidence Bundle 规格

修订日期：2026-09-17。状态：实施验收规格；所有 `PASS` 需要真实命令和可复核 artifact，文档中的候选值不是已发布支持声明。

本文把 [39 验证工作簿](39-validation-benchmark-and-release-workbook.md) 和 [43 工具链/发布门禁](43-toolchain-simulation-and-release-gates.md) 进一步细化为逐项可执行的测试追踪表；相应的代码结构、C2 receipt 和四模型施工顺序见 [44 实现收口规格](44-next-implementation-code-and-integration-spec.md)。目标是让实现、测试、模型和 release 负责人能够回答同一组问题：

1. 这次运行到底执行了哪个 source/layout/runtime？
2. 每个 slot、group、pass、reader fence 是否按授权计划发生？
3. 低内存目标是否覆盖完整请求的 process-tree 峰值，而不是只测 denoiser backing？
4. resident/default 是否保持原性能？generic executor 是否引入可接受开销？
5. 发生 source 变化、worker 崩溃、fence 丢失或 catalog revoke 时，结果是否 fail-closed？

## 1. 证据状态与禁止混淆

### 1.1 状态定义

```text
NOT_STARTED       尚未运行
IMPLEMENTED       代码存在，但没有完整证据
PASS              当前冻结 tuple 和环境通过指定门槛
FAIL              明确不满足门槛
INCONCLUSIVE      采样、设备、身份或统计条件不完整
QUARANTINED       运行发生安全不确定性，engine/session 不可复用
REVOKED           已发布记录因回归/身份变化撤回
```

`IMPLEMENTED` 不能写成 `PASS`；`PASS` 只对精确的 model/source/workload/layout/device/runtime tuple 有效，不能泛化到同模型的其他 shape、dtype、LoRA、ANE 或 compiled route。

### 1.2 三类证据必须分开

| 证据 | 能说明 | 不能说明 |
|---|---|---|
| Host/compiler | schema、layout、receipt 算法正确 | 真实 GPU 性能、全请求峰值 |
| Synthetic Metal/fake adapter | slot/fence/取消/故障状态机正确 | 模型质量、真实磁盘 I/O、swap 速度 |
| Real model campaign | 完整请求在指定环境的峰值、速度和质量 | 其他机器/其他 shape 自动支持 |

## 2. 测试分层和文件映射

### 2.1 Host-only 层

| ID | 目标 | 代码/测试 |
|---|---|---|
| HOST-CFG-001 | selector 五档、类型、重复键、禁用语义 | `tests/native/test_streaming_contract.py` |
| HOST-CFG-002 | manual layout exact，不因 budget 重新选择 | `tests/native/test_streaming_contract.py` |
| HOST-LAYOUT-001 | 3535+ 合法布局和边界拒绝 | `test_streaming_layout.py` |
| HOST-LAYOUT-002 | canonical digest 对字段顺序稳定 | `streaming_layout_test.cpp` |
| HOST-RESOLVE-001 | empty catalog fail-closed | `streaming_preset_resolver_test.cpp` |
| HOST-RESOLVE-002 | record/device/workload/runtime exact match | 同上 |
| HOST-SOURCE-001 | fd lease capture/duplicate/revalidate | `streaming_source_lease_test.cpp` |
| HOST-SOURCE-002 | alias、replace、same-size mutation、truncate、short read | `test_streaming_source_lease.py` |
| HOST-RECEIPT-001 | expected/actual matrix、digest、错误优先级 | `streaming_actual_receipt_test.cpp` |
| HOST-RECEIPT-002 | C ABI `struct_size`/version/finish 时序 | `streaming_c_bridge_failure_test.cpp` |

### 2.2 Synthetic executor/Metal 层

| ID | 目标 | 必须覆盖 |
|---|---|---|
| SYN-EXEC-001 | K=1/2/3 owner pump | G=1/2，D=0..K−1，Q=1..K |
| SYN-EXEC-002 | pass transition | reload、carry-first-group |
| SYN-EXEC-003 | heterogenous pools | serial、retain-all、class barrier |
| SYN-EXEC-004 | fence ordering | immediate、delayed、out-of-order、already-complete |
| SYN-EXEC-005 | cancellation | enqueue 前、fill 中、reader 中、pass boundary |
| SYN-EXEC-006 | safety failure | zero fill、short read、mailbox overflow、worker error |
| SYN-EXEC-007 | sanitizer | address/undefined/thread |
| SYN-EXEC-008 | audit | Off/default new allocation/thread/hook 全为零 |

### 2.3 Real model 层

每个模型都必须独立提交一张 campaign card，不能用另一个模型的 generic executor PASS 替代：

```text
REAL-<MODEL>-PROBE     source lease、metadata、workload identity
REAL-<MODEL>-SNAPSHOT  descriptor/layout 与 record digest
REAL-<MODEL>-RUN       完整输出/receipt/drain
REAL-<MODEL>-CANCEL    协作取消和 caller output 不变
REAL-<MODEL>-FAULT     source/fill/fence/worker 故障
REAL-<MODEL>-MEMORY    process-tree peak 与 swap delta
REAL-<MODEL>-QUALITY   输出 hash/图像或视频质量
REAL-<MODEL>-P0        resident/default 非回归
REAL-<MODEL>-P1        same-layout generic overhead
```

## 3. Receipt v2 逐项追踪

### 3.1 正常路径

| ID | 场景 | 断言 |
|---|---|---|
| RCP-001 | single pool reload | 每个 pass/group 恰好一个 fill；fence issued=completed |
| RCP-002 | K2 normal | slot rotation 与 expected pool/slot 一致 |
| RCP-003 | K3 + D2/Q3 | prefetch window 不超过未完成 slot 数 |
| RCP-004 | carry-first-group | carry event 坐标正确；逻辑 fill 数不减少 |
| RCP-005 | serial pools | switch 前旧 pool 全部 drain |
| RCP-006 | retain-all pools | 两个 pool capacity 和实际使用均被记录 |
| RCP-007 | already-complete reader | issued/completed 仍成对出现，不能漏记 |
| RCP-008 | final drain | `drain_completed=true` 且 source post-drain revalidate 成功 |

### 3.2 故障路径

| ID | 注入 | 结果 |
|---|---|---|
| RCP-F01 | zero fill | `receipt_logical_bytes_mismatch`，无成功 result |
| RCP-F02 | duplicate completion | `receipt_group_duplicate`，必要时 quarantine |
| RCP-F03 | wrong pool/slot | `receipt_pool_slot_mismatch` |
| RCP-F04 | reader issued > completed | `receipt_fence_incomplete` + quarantine |
| RCP-F05 | source generation mismatch | `receipt_source_generation_mismatch` |
| RCP-F06 | carry rotation error | `receipt_carry_mismatch` |
| RCP-F07 | event digest mutation | `receipt_digest_mismatch` |
| RCP-F08 | mailbox overflow | owner 停止新 dispatch，join/drain；无法证明安全则 quarantine |
| RCP-F09 | adapter `drain=false` | 不允许 destroy backing，quarantine |
| RCP-F10 | receipt read before finish | 稳定 API error，不暴露半成品摘要 |

每一项都记录：`error_code`、`output_exists`、`drain_completed`、`quarantine`、`retry_allowed`、`engine_recreated`。

## 4. Source Lease 与身份验收

### 4.1 单一 fd lineage

生产 adapter 必须遵循：

```text
SourceLease::capture(files)
  -> open + fstat each file once
  -> metadata/header parser reads duplicate fd
  -> probe and snapshot share immutable lease
  -> worker reader uses duplicate_fd only
  -> pre-GPU revalidate paths + open fds
  -> post-drain revalidate paths + open fds
```

若测试发现 adapter 在 fill callback 内再次按 path `open()`，即使输出正确也判 `FAIL`，因为无法证明 TOCTOU 安全。

### 4.2 身份矩阵

| ID | 操作 | 预期 |
|---|---|---|
| SRC-001 | 无变化 | generation/digest 稳定 |
| SRC-002 | alias symlink 指向新 inode | pre-GPU reject |
| SRC-003 | 原文件 rename | held fd 可读，但 named path revalidate reject |
| SRC-004 | 原地同 size 改写 | open-fd metadata/content policy reject |
| SRC-005 | truncate | descriptor/read range reject |
| SRC-006 | short read | 不得标记 Ready；后续 dispatch 停止 |
| SRC-007 | probe 与 snapshot 使用不同 lease | authority reject |
| SRC-008 | generation=0 | public authorize reject |
| SRC-009 | post-drain mutation | `source_lease_verified=false`，无 public success |
| SRC-010 | Off/default | capture/open/fstat/hash 计数全为零 |

当前 stat identity 不是 content SHA-256 的替代品。production record 需要明确 `identity_kind` 和 artifact import policy；如果只具有 snapshot identity，catalog builder 不得把它误标为 immutable content certification。

## 5. 完整内存校准

### 5.1 采样边界

单次运行必须覆盖：

```text
fresh process
clean baseline
model/session construction
tokenizer/text conditioning
prefix and every denoise pass
upsampler (LTX)
VAE/audio/Vocoder/export
worker join + reader drain
terminal sample before process exit
```

采样器每条记录至少包含：

```json
{
  "mono_ns": 0,
  "pid": 0,
  "ppid": 0,
  "role": "denoiser-worker",
  "phys_footprint_bytes": 0,
  "resident_bytes": 0,
  "mlx_active_bytes": 0,
  "mlx_peak_bytes": 0,
  "gpu_bytes": null,
  "swap_in_bytes": 0,
  "swap_out_bytes": 0,
  "pressure": "normal",
  "pool": 0,
  "slot": 0,
  "fills": 0,
  "fences_issued": 0,
  "fences_completed": 0,
  "correlation_id": "..."
}
```

`tree_peak` 定义为同一时间点进程树 footprint 的总和：

```text
tree_peak = max_t(sum(phys_footprint(pid, t) for pid in request_process_tree(t)))
```

不允许把每个进程的独立峰值相加。最大采样间隔超过 campaign policy、发现未知子进程或 worker 提前退出而无 terminal sample 时，状态为 `INCONCLUSIVE`。

### 5.2 target 判定

对目标 `T`：

```text
H(T) = max(512 MiB, ceil(0.10 × T))
PASS(T) iff every measured sample is complete
          && tree_peak + H(T) <= T
          && no unplanned swapout/failure
```

一个 sample 超限即可 fail，不用均值掩盖 peak。不同 target 的 record 不能互相推导；8 GiB PASS 不代表 10/12/16/20 GiB 有同一布局或同一性能。

### 5.3 校准样本数量

每个冻结 candidate/target 最低：

```text
3 warmup（不进入统计）
20 measured matched requests
10 cancel/fault requests
2 source mutation requests
1 fresh-process rerun
1 independent verifier rerun
```

若 20 个 measured 中有任何一次 output/receipt/peak 不完整，candidate 不能发布。bootstrap 仅用于置信区间，不替代逐样本 hard cap。

## 6. Resident、Streaming、Bounded、Swap 四臂实验

### 6.1 固定变量

四臂必须固定：

```text
model/source digest
workload shape/steps/seed/prompt
dtype/precision/kernel/backend
text/conditioning/VAE/export route
OS build/GPU/build identity
cache warmup policy
```

只允许 residency/swap policy 改变。若为了让 resident 能运行而修改 shape、steps 或 dtype，实验作废。

### 6.2 四臂定义

| Arm | 定义 | 主要回答 |
|---|---|---|
| A resident | 当前 dev/default 原路径 | 新框架是否影响默认 |
| B exact streaming | reviewed layout + generic executor | 低峰值和框架成本 |
| C bounded legacy | 既有 memory guard/legacy streamed | 新旧低内存路径比较 |
| D controlled pressure | resident + 独立 pressure helper | 与系统 reclaim/swap 的真实对照 |

pressure helper 必须独立进程、有限字节上限、超时自动退出、无 root、不得修改系统 swap 配置，也不得计入 TurboCider request tree peak。不能通过关闭 swap 或写系统级 sysctl 制造结果。

### 6.3 swap 结论规则

只有同时满足以下条件，release note 才能写“相对受控 swap pressure 更快”：

```text
streaming wall median < pressure-resident wall median
streaming P95 < pressure-resident P95
bootstrap interval 不跨越预设非劣/优势阈值
swap_out/compression 显著降低
quality/output/成功率不劣
两臂 peak 和 workload identity 完整
```

否则只能写“在该 tuple 下减少峰值/系统压力”，不能泛化为所有机器或所有模型更快。

## 7. P0–P4 门禁与统计方法

### 7.1 P0 默认非回归

至少 20 个 ABBA 或随机交错 matched pairs；报告 median、P95、MAD、bootstrap 95% CI：

```text
wall median ratio <= 1.02
wall P95 ratio <= 1.05
output/quality parity PASS
audit new hooks/probes/workers/pools/allocations = 0
```

若 dev 与 candidate build identity 不同，必须同时保存两者 source/build manifest；不能只比较一次本机 wall。

### 7.2 P1 Generic executor overhead

direct exact 和 generic executor 必须使用同一：

```text
source lease/layout/P/G/K/D/Q/pool/component policy
```

门槛：wall median ≤ 1.02、wall P95 ≤ 1.05、denoise median ≤ 1.02、output parity PASS。改变 layout 的结果只能放到 P4。

### 7.3 P2 Target fit

每个 target 都要求：

```text
20/20 measured complete
tree_peak + H(T) <= T for every sample
receipt/source generation verified
no unexpected swapout
independent verifier PASS
```

### 7.4 P3 Swap 对照

P3 不设“必须更快”的先验；若 streaming 只降低 peak 但更慢，应如实记录为 trade-off。任何压力 helper 异常退出、系统 pressure 状态不一致或 swap counter 不可用，都将 P3 标为 `INCONCLUSIVE`。

### 7.5 P4 策略搜索

比较 P/G/K/D/Q、serial/retain-all、reload/carry 时，输出 Pareto 表：

```text
target fit, tree_peak, wall median/P95, fill wait, reader wait,
logical read bytes, pool memory, swap delta, quality, failure rate
```

不得用 P4 的更快结果掩盖 P0/P1 回归，也不得把未 review 的 Pareto 点写入 production catalog。

## 8. Evidence Bundle 规范

### 8.1 目录

```text
results/streaming/<model>/<date>/<campaign-id>/
  manifest.json
  environment.json
  build-identity.json
  source-identity.json
  workload-card.json
  selector.json
  resolved-record.json
  candidate-plan.json
  actual-receipt.json
  sampler.jsonl
  raw-samples.jsonl
  timings.json
  quality.json
  faults.json
  audit.json
  swap.json
  verification-rerun.json
  review.json
  summary.json
```

### 8.2 `manifest.json` 最小字段

```json
{
  "schema_version": 1,
  "campaign_id": "...",
  "model_id": "...",
  "target_request_memory_bytes": 0,
  "comparison_kind": "P0|P1|P2|P3|P4",
  "status": "PASS|FAIL|INCONCLUSIVE",
  "source_digest": "sha256:...",
  "layout_digest": "sha256:...",
  "runtime_digest": "sha256:...",
  "device_digest": "sha256:...",
  "receipt_digest": "sha256:...",
  "sampler_max_gap_ns": 0,
  "independent_verifier": "PASS"
}
```

禁止进入 bundle 或 App result：fd、GPU pointer、authority pointer、worker PID、绝对模型路径、用户原始 prompt（除非用户明确允许并已脱敏）。

### 8.3 Builder 入门条件

catalog builder 只有在以下条件全部满足时才生成 record：

```text
manifest schema valid
source/layout/runtime/device identities exact
receipt verifier PASS
all measured samples complete
P0/P1/P2/P3 required gates PASS
model quality review PASS
independent verifier PASS
runtime/model/performance/release review signed
record not revoked
```

builder 输出的 canonical record 必须再次计算 digest，并与 evidence 中的 digest 比较；任何手工修改字段都会导致 digest mismatch。

## 9. 四模型验收卡片

### Z-Image Turbo

首张 card 固定 512²、原始 BF16 GPU、single pool、无 LoRA/ANE/compiled/GGUF。必须包括：shard identity、完整 denoise/VAE/export、receipt v2、20 matched requests、source mutation、cancel、P0/P1/P2。

### Flux.2 Klein 9B

固定 8 dual + 24 single blocks、BF16 eager GPU。必须分别验证 serial 和 retain-all 的 memory accounting；VAE/export 不能省略。Flux 4B compiled 不属于该 card。

### MiniMax H3 Turbo

只做 original BF16 native Metal。必须验证 carry-first-group 坐标、跨 pass reader fence、C bridge receipt、ordinary H3/VDN/FastH3 默认回归；普通 H3 变体不自动继承资格。

### LTX 2.5

必须在 worker-local session 中完成 probe/snapshot/resolve/generate；覆盖 denoiser、upsampler、video/audio VAE、vocoder、export、worker crash/quarantine、视频帧数和音频存在性。

## 10. 本机执行顺序与命令模板

### 10.1 Host 回归

```sh
env TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
python3 -B tests/native/test_streaming_contract.py
python3 -B tests/native/test_streaming_layout.py
python3 -B tests/native/test_streaming_source_lease.py
python3 -B tests/native/test_streaming_preset_resolver.py
python3 -B tests/native/test_streaming_actual_receipt.py
env TC_STREAMING_SANITIZER=address,undefined python3 -B tests/native/test_streaming_layout.py
env TC_STREAMING_SANITIZER=thread python3 -B tests/native/test_streaming_layout.py
make test-streaming-host
make test-streaming-contract
make test-streaming-audit
```

### 10.2 Real-device opt-in

```sh
python3 -B tests/native/test_streaming_metal.py
python3 -B tests/native/test_<model>_streaming_descriptor.py
python3 -B tests/native/test_<model>_candidate_streaming_lifecycle.py
python3 -B tools/native/run_streaming_campaign.py \
  --policy docs/design/streaming/examples/<model>-<policy>.json \
  --output results/streaming/<model>/<date>/<campaign-id>
python3 -B tools/native/verify_streaming_campaign.py \
  results/streaming/<model>/<date>/<campaign-id>
```

真实 Metal/模型命令必须显式 opt-in，不进入默认 CI，也不能在未经授权时主动施加系统压力。

### 10.3 证据审阅

```sh
python3 -B tools/native/inspect_streaming_evidence.py <bundle>
python3 -B tools/native/verify_streaming_campaign.py <bundle>
python3 -B tools/native/build_streaming_catalog.py \
  --evidence <bundle> --out <record.json> --dry-run
```

`dry-run` 通过后仍需 model/performance/release owner 签字；builder 不自动把 record 写入 production catalog。

## 11. Release、撤回与回滚验收

### 11.1 发布前 checklist

- [ ] production catalog 只包含 reviewed records；
- [ ] App options 与 catalog revision 一致；
- [ ] resolve 与 generate 使用同一 record/resolution digest；
- [ ] source lease、receipt、device/runtime identity 均可回溯；
- [ ] P0/P1/P2/P3 状态和样本数写入 summary；
- [ ] revoke/rollback 操作在 staging 验证；
- [ ] Off/default 审计无新增热路径工作；
- [ ] 用户文案不承诺“一定低于显存”或“一定比 swap 快”。

### 11.2 撤回触发条件

```text
source artifact digest 变化
runtime/adapter/kernel/reader revision 变化
同 tuple P0/P1 回归超门槛
任意 target sample 超 hard cap
receipt mismatch 可绕过
quarantine/backing 泄漏
质量/输出不一致
catalog record digest 不一致
```

撤回后旧 Job 标记 `stale`，不能静默切换到另一个布局；若有安全不确定性，engine 必须重建。

### 11.3 回滚方式

1. 将 record `release.revoked=true`，保留原 evidence 供审计。
2. 发布只读 catalog revision，App options 立即显示 unavailable。
3. 默认 Off/resident 路径继续可用。
4. 不删除用户 Job 历史；旧 resolved job 只能重新 resolve。
5. 重新测量并生成新 revision 后再走完整 review。

## 12. 完成定义

本规格的完成是“每个实现任务都有测试 ID、数据字段、命令、统计门和回滚动作”。整个 public streaming 目标只有在以下状态同时满足时才算完成：

```text
C2 receipt v2 PASS
至少一个模型 public exact PASS
四模型 card 均有明确 PASS/FAIL/INCONCLUSIVE
目标档位由完整 process-tree evidence 生成
P0/P1 无回归
P2 target fit 通过
P3 swap 结论可复核
App/JobStore 事务通过
catalog builder/revoke/rollback 通过
```

在此之前，任何文档、UI 或 API 都必须保留 private/candidate/unavailable 标识。
