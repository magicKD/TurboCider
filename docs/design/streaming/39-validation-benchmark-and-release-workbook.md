# 39 · Streaming 验证、性能对照与发布工作簿

[目录](README.md) · [代码合同与施工设计](38-framework-code-contracts-and-implementation-workbench.md) · [校准与发布验收](37-public-streaming-calibration-performance-acceptance.md) · [当前进度](13-implementation-progress.md)

修订日期：2026-09-17。状态：**可执行的测试与证据工作簿；所有阈值是验收门，不是当前成绩。**

本文将“能编译”“能生成”“满足目标内存”“没有默认回归”“比 swap 更快”拆成互不替代的证据层，给出测试 ID、输入、输出、统计方法和失败处理。没有真实模型或 Metal 权限时必须标记 `SKIP`/`INCONCLUSIVE`，不能记为 PASS。

## 1. 证据分层

| 层级 | 证明内容 | 典型工具 | 不能替代 |
|---|---|---|---|
| H0 静态/Host | schema、digest、compiler、slot safety | host tests | 真实 GPU/内存 |
| H1 ABI/App | ownership、Swift parity、JobStore 事务 | contract/App tests | 模型正确性 |
| H2 Runtime | source lease、fence、drain、quarantine | synthetic adapter | target peak |
| H3 Model | 真实 exact output/质量/生命周期 | model campaign | 档位资格 |
| H4 Memory | 完整 process-tree peak + margin | sampler/calibration | 速度优势 |
| H5 Performance | resident/streaming/bounded/swap | ABBA/P3 campaign | 安全门 |

release record 必须逐层列出状态。缺少必需层时只能是 `candidate`、`unavailable` 或 `INCONCLUSIVE`。

```text
可执行：请求成功结束且 drain 安全
可合格：满足 workload/source/device/target 的全部门
可发布：evidence 独立复核、review、catalog builder 通过
```

private candidate 的“可执行”不等于 public target 的“可合格”。

## 2. 环境和身份冻结

每个 campaign 在第一条请求前写入：

```text
TurboCider commit / dirty flag
compiler + SDK + deployment target
OS build
device model / GPU family / physical memory
MLX/Metal/Core ML revision
model root canonical identity
artifact stat tuples + manifest digest
workload card digest
candidate/layout/record digest
execution container (embedded_app / worker)
sampler interval/max gap
pressure helper version（若有）
```

任一身份变化都产生新的 campaign/candidate ID。不能把新 build 结果追加到旧 campaign，也不能用同名目录代替 source identity。

## 3. Host、Coordinator 与 API 矩阵

### 3.1 Compiler/resolver

| ID | 输入 | 期望 |
|---|---|---|
| PUB-HOST-001 | 3535 layout golden | canonical/layout digest 稳定 |
| PUB-HOST-002 | 14 组 K/D/Q | 合法约束和拒绝原因稳定 |
| PUB-HOST-003 | duplicate block/group/pool | fail-closed，无部分 backing |
| PUB-HOST-004 | exact selector replay | record/catalog/resolution 全匹配 |
| PUB-HOST-005 | 非五档 target | `unsupported_memory_target` |
| PUB-HOST-006 | incomplete calibration | 不进入 selected |
| PUB-HOST-007 | physical range mismatch | `unvalidated_device` |
| PUB-HOST-008 | revoked record | `preset_not_public` |

### 3.2 Coordinator/catalog

| ID | 场景 | 断言 |
|---|---|---|
| PUB-COORD-001 | 一次 resolve | provider snapshot 恰好一次 |
| PUB-COORD-002 | preflight 后 request 修改 | probe/compile 为零，返回 mismatch |
| PUB-COORD-003 | empty catalog | model probe/GPU calls 为零 |
| PUB-COORD-004 | resolve 期间替换 catalog | 本次使用冻结 snapshot |
| PUB-COORD-005 | revalidate 新 revision | stale，不能换 layout |
| PUB-COORD-006 | selector Off/absent | coordinator/catalog/probe 不创建 |

### 3.3 C ABI/Swift/App

| ID | 场景 | 期望 |
|---|---|---|
| PUB-ABI-001 | null 参数 | 稳定错误且无泄漏 |
| PUB-ABI-002 | engine busy | 不阻塞，ownership 正确 |
| PUB-ABI-003 | resolve-only | 不持有 global GPU lock |
| PUB-ABI-004 | cancel before resolve | 不产生 authority/job |
| PUB-ABI-005 | stale exact selector | 不执行模型、不产生 output |
| PUB-ABI-006 | actual mismatch | result 为空，返回 mismatch |
| PUB-SWIFT-001 | Codable round-trip | selector/resolution/error 语义一致 |
| PUB-SWIFT-002 | stale options | 旧 model/shape 不覆盖新状态 |
| PUB-APP-001 | resolve 失败 | JobStore 不写 resolved job |
| PUB-APP-002 | resolve 成功 | persist/generate nonce 与 digest 一致 |

## 4. Source Lease 与故障矩阵

Synthetic fixture 至少包含：

```text
single artifact
multi-shard artifact
same-size mutation
path replace / rename / unlink
truncate / short read
duplicate logical_id
descriptor 未租赁 artifact
missing index/header
fd duplication failure
```

| ID | 故障 | 期望 |
|---|---|---|
| PUB-SOURCE-001 | 内容不变 | digest 稳定，generation 非零 |
| PUB-SOURCE-002 | path replace | source changed，不进入新 GPU |
| PUB-SOURCE-003 | rename 原文件 | fd 可读但 path revalidate fail |
| PUB-SOURCE-004 | size/mtime/inode 变化 | fail-closed |
| PUB-SOURCE-005 | short read | 停止新 refill、join、drain |
| PUB-SOURCE-006 | shard 枚举顺序变化 | canonical digest 不变 |
| PUB-SOURCE-007 | duplicate logical_id | compile 前拒绝 |
| PUB-SOURCE-008 | descriptor 漏 artifact | snapshot compile 拒绝 |
| PUB-SOURCE-009 | fd dup 失败 | 不创建 reader |
| PUB-SOURCE-010 | Off | open/fstat/hash 计数为零 |

以下情况必须 quarantine：reader completion 丢失、mailbox overflow、GPU 已提交但 drain 未知、worker 异常退出且 backing 可能仍被引用、adapter 报告 `drained=false`。preflight 前取消、catalog empty、GPU 工作前 source 变化通常是普通失败。

每个 fault 记录 `expected_error`、`actual_error`、`output_exists`、`drain_completed`、`quarantine`、`retry_allowed`。

## 5. Actual Receipt v2 验收

正常路径覆盖：single-pool reload、K2 carry、multi-pool serial/retain-all、already-complete reader、Q=1/2/3、D=0/1。

| ID | 注入 | 必须拒绝 |
|---|---|---|
| PUB-RECEIPT-001 | 某 pass/group fill=0 | 是 |
| PUB-RECEIPT-002 | duplicate fill | 是 |
| PUB-RECEIPT-003 | logical bytes 少一段 | 是 |
| PUB-RECEIPT-004 | wrong pool/slot | 是 |
| PUB-RECEIPT-005 | reader issued > completed | 是 |
| PUB-RECEIPT-006 | source generation mismatch | 是 |
| PUB-RECEIPT-007 | carry rotation 错误 | 是 |
| PUB-RECEIPT-008 | event digest mutation | 是 |
| PUB-RECEIPT-009 | drain=false | 是，并 quarantine |
| PUB-RECEIPT-010 | mismatch 后 serialize | 永远无 public success |

低开销要求：begin 一次性分配；worker 不写 vector、不加 receipt mutex；digest 只在 finish 计算；Off/default receipt allocation 为零。

## 6. 四模型真实验收卡

### 6.1 Z-Image Turbo

首个 candidate 固定为 512²、原始 BF16 GPU、single pool、无 LoRA/ANE/compiled/GGUF/ConvRot。必须有 source/layout/runtime/component identity、三 public hook、fd lease、receipt v2、20 matched requests、output/质量、完整 peak、fault 和 independent verifier。

### 6.2 Flux.2 Klein 9B

固定 8 dual + 24 single blocks、BF16 eager GPU。额外验证 dual/single class barrier、`retain_all` capacity、setup/steady allocation、concatenate boundary、完整 VAE/export；不得把 Flux 4B compiled 加入资格。

### 6.3 MiniMax H3 Turbo

只接受 `minimax-h3-turbo` original BF16 native Metal。额外验证 carry 坐标、C receipt/source generation、authority 与 experimental bool 分离，以及 ordinary H3/VDN/FastH3 默认回归。

### 6.4 LTX 2.5

额外验证 worker-local authority/lease、parent/worker/finalizer correlation、denoiser/VAE/upsampler/audio/export 完整闭包、SIGTERM/SIGKILL/quarantine，以及视频帧数、时长和音频存在性。

## 7. 内存校准工作流

### 7.1 Candidate 状态机

```text
generated -> plan_validated -> simulator_screened -> measured
-> independently_verified -> model_reviewed -> performance_reviewed
-> release_reviewed -> staging -> production
```

每一步产生状态和 digest；不能跳过 independent verifier。

### 7.2 单次运行

```text
fresh process -> baseline sample -> probe/lease -> text/conditioning
-> all denoise passes -> VAE/upsample/audio/export
-> drain readers/workers -> terminal sample -> release -> verify exit
```

建议每 candidate：3 warmup、20 matched measured、10 fault/cancel、2 source mutation、1 fresh-process rerun、1 independent verifier rerun。

### 7.3 Process-tree sampler

每点记录 monotonic timestamp、pid/ppid/role、resident/phys_footprint、MLX active/peak、GPU accounting（可得时）、swap/compression、pressure、I/O、pool/fill/fence counters 和 correlation id。

```text
tree_peak = max_t(sum(process_bytes(pid,t) for pid in request_process_tree))
```

不能将各进程独立 peak 相加。最大采样 gap 超标时 campaign 为 `INCONCLUSIVE`。

### 7.4 Target 判定

```text
H(T) = max(512 MiB, ceil(0.10 × T))
eligible(T) = complete
              && tree_peak + H(T) <= T
              && source/device/workload/runtime exact match
```

`tree_peak` 覆盖 parent、worker、encoder、denoiser、VAE/upsampler/audio、export 和 drain/exit。任一样本超限即 fail，不用平均值掩盖 peak。

## 8. Resident/Streaming/Bounded/Swap 四臂

| Arm | 路径 | 目的 |
|---|---|---|
| A | resident/default | 速度和质量基线 |
| B | public exact streaming | 显式 slot/pool 路径 |
| C | legacy bounded-memory | 旧 guard/streamed 对照 |
| D | resident + controlled pressure | reclaim/swap 对照 |

若 A 在目标设备自然无法运行，标为 unavailable，不换 workload。pressure helper 必须独立进程、有最大字节数/超时、逐步加压；不修改系统 swap 参数、不使用 root、不计入 TurboCider target peak。

四臂都记录 wall/text/denoise/VAE/export、tree/MLX peak、slot backing、logical/physical reads、fill/wait/fence、swap/compression、pressure、failure/drain/quarantine 和 output quality。

## 9. 性能门

### 9.1 P0 默认路径

至少 20 个 warm paired requests，ABBA 或随机交错：

```text
wall median ratio <= 1.02
wall P95 ratio <= 1.05
quality/output parity PASS
public probe/catalog/source/receipt/worker allocation = 0
```

### 9.2 P1 同布局框架成本

direct exact 与 generic executor 固定相同 source/layout/P/G/K/D/Q/pool/component policy：

```text
wall median ratio <= 1.02
wall P95 ratio <= 1.05
denoise median ratio <= 1.02
quality/output PASS
receipt/layout identity PASS
```

现有 private P1 只作锚点，不能直接转 public record。

### 9.3 P2 target

20 measured requests 无 memory failure；fresh-process rerun；independent verifier PASS；`tree_peak + H(T) <= T`；无未预期 swap failure。

### 9.4 P3 swap 结论

只有 streaming median/P95 优于 pressure-resident、bootstrap interval 不跨越“更慢”阈值、swap delta 显著更低、failure/quality 不劣，release note 才能写“相对 swap pressure 更快”。否则只能写“降低峰值/系统压力”。

### 9.5 P4 策略

比较 P/G/K/D/Q、pool、prefetch、worker 时，将 layout、I/O、kernel、source-format 变化分开。策略收益不能补偿 P0 回归。

## 10. Evidence Bundle

```text
results/streaming/<model>/<date>/<campaign>/
  manifest.json
  environment.json
  build-identity.json
  source-identity.json
  workload-card.json
  candidate-plan.json
  sampler.jsonl
  raw-samples.jsonl
  receipt.json
  metrics.json
  quality.json
  faults.json
  audit.json
  summary.json
  verification-rerun.json
  review.json
```

Builder 只接受 schema-valid、independent verifier PASS、model/performance/release review 完整且未 revoke 的 evidence。production catalog 为内嵌编译资源，App/环境变量不能覆盖。

## 11. CI 与本机命令

每个代码 PR 至少运行：

```sh
env TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
make test-streaming-host
make test-streaming-contract
make test-streaming-audit
tools/native/build_app.sh
git diff --check
```

拟议 test-only catalog smoke：

```sh
TURBOCIDER_STREAMING_CATALOG=test-only \
tools/native/run_streaming_calibration.py \
  --workload profiles/streaming/workloads/z-image-turbo-image-512.json \
  --candidate profiles/streaming/candidates/z-image-turbo/seed.json
```

拟议 independent verifier：

```sh
python3 -B tools/native/verify_streaming_campaign.py \
  --campaign results/streaming/<model>/<date>/<campaign> \
  --independent
```

以上两个命令是工具实现合同，未实现前不能当作现成能力；真实模型缺失时报告 SKIP 原因。

## 12. Release 签字单

### Runtime owner

- [ ] preflight once、snapshot stable；
- [ ] fd lease/source generation；
- [ ] receipt v2；
- [ ] cancel/drain/quarantine；
- [ ] Off audit zero。

### Model owner

- [ ] 三 public hook；
- [ ] public/private 共用 exact core；
- [ ] unsupported route fail-closed；
- [ ] output/quality/fault matrix。

### Performance owner

- [ ] P0/P1/P2；
- [ ] P3（如需 swap 速度声明）；
- [ ] P4 strategy report；
- [ ] 样本数和 bootstrap interval。

### App/Release owner

- [ ] 默认 Off，只显示五档；
- [ ] unavailable/stale/revoke 可解释；
- [ ] JobStore v2/LTX handshake；
- [ ] evidence/builder/linter/reviewer；
- [ ] staging/rollback/revoke；
- [ ] release note 不夸大。

## 13. Public 开放顺序与完成定义

```text
framework H0/H1/H2 green
-> Z-Image 一个 workload + target staging/production
-> Flux 一个 workload + target
-> H3 Turbo 一个 workload + target
-> LTX 一个 workload + target
-> 再逐步扩展档位、shape 和 device range
```

一次 release 不同时加入四模型五档。只有 evidence bundle、独立 review 和 production catalog record 三者同时存在，App 才能把相应 model/workload/target 显示为 available；当前 private candidate、tiny P1 或 fake receipt 都不能描述为 public 支持。
