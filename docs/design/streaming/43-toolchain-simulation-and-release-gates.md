# 43 · Streaming 工具链、仿真、性能对照与发布门禁

[目录](README.md) · [产品化合同](40-public-productization-and-app-contract.md) · [调度器](41-scheduler-multi-slot-and-multi-pool-implementation.md) · [模型施工单](42-model-adapter-playbooks.md) · [验证工作簿](39-validation-benchmark-and-release-workbook.md)

修订日期：2026-09-17。状态：**工具和发布实施方案；命令与阈值在工具落地前属于合同，不代表当前已经具备全部命令**。

本文把“怎么证明这个方案可用、性能没有退化、低内存比 swap 更合理”落成一套可重复的工具链。原则是：先用 host/simulator 缩小候选，再用完整请求实测，最后由独立 verifier 生成不可变 catalog record。没有完整证据，状态只能是 `candidate` 或 `INCONCLUSIVE`。

## 1. 工具链分层

```text
inspect        读取模型/设备/workload/source 身份
compile        生成 canonical layout + digest
simulate       离散事件估算峰值、I/O、wait、overlap
campaign       在真实设备运行候选并采样完整请求
verify         独立检查样本、峰值、质量、receipt、统计门
catalog-build  从 verified evidence 生成只读 production record
app-options    将 catalog record 转成用户可见档位状态
```

工具之间只能通过 JSON artifact 传递，不共享可变内存或临时全局状态。每个 artifact 带 `schema_version`、`producer_revision` 和输入 digest。

## 2. Inspect 工具

输入：

```text
--model-id --model-root --workload-card
--device-json（可选，默认发现本机）
--execution-container
```

输出至少包含：

```json
{
  "schema_version": 1,
  "model_id": "z-image-turbo",
  "source_files": [{
    "logical_id": "transformer.shard.0",
    "path": "...",
    "bytes": 123,
    "device": 1,
    "inode": 2,
    "mtime_ns": 3,
    "ctime_ns": 4,
    "header_digest": "..."
  }],
  "artifact_manifest_digest": "...",
  "workload_digest": "...",
  "runtime_identity": {
    "adapter_revision": "...",
    "reader_revision": "...",
    "kernel_revision": "..."
  }
}
```

inspect 不创建 GPU 资源，不修改模型目录，不把 `stat` snapshot 误写成 content hash。对大权重的 full SHA-256 只允许在导入/受控校准阶段执行，不能每次 Off/default 请求重复扫描。

## 3. Compile 工具

`compile` 将 adapter Descriptor 与显式 `StreamingConfig` 编译成 canonical layout、layout digest、stage/group/pool/slot capacities、source bytes、memory lower bound 和拒绝原因。

拟议命令：

```sh
python3 -B tools/native/streaming_compile_plan.py \
  --descriptor descriptor.json \
  --config candidate.json \
  --out plan.json
```

必须拒绝：duplicate block/group/pool、suffix groups<K、D≥K、Q>K、unsafe boundary、source range 越界、materialization 缺失、capacity overflow、非法 carry/multi-pool，以及从 target memory 反推并改写 P/K/G。

## 4. 离散事件 Simulator

Simulator 不运行 MLX/Metal，仅模拟：

```text
FillEnqueue -> FillReady -> PrefixEncode -> GroupClaim
-> ReaderSubmit -> ReaderFenceComplete -> PoolSwitch -> Drain
```

输入为 layout、fill/compute 延迟分布、activation/workspace estimates 和 component lifetime policy；输出为 predicted peak、fill/reader wait、queue depth、overlap ratio 与候选风险标记。

Simulator 不能证明 MLX lazy materialization、Metal driver 峰值、OS page cache/swap、VAE/export 真实内存、输出质量或 worker crash/drain，所以只能筛选，不能生成 public evidence。候选按 `(predicted_peak, predicted_wall, logical_read_bytes, risk_flags)` 做 Pareto 过滤。

## 5. Campaign Runner

### 5.1 身份冻结

第一条请求前保存：

```text
TurboCider commit + dirty flag
compiler/SDK/deployment target + OS build
GPU model/family + physical memory
MLX/Metal/Core ML revision
model root/source lease identity
workload card + candidate/layout/record digest
execution container + process roles
sampler interval/max gap
pressure helper revision（如适用）
```

任何 identity 改变都生成新 campaign ID；不允许把两个 build 的样本追加到同一目录。

### 5.2 单次运行生命周期

```text
fresh process -> baseline sample -> probe/lease/snapshot
-> text/tokenizer/conditioning -> all denoise passes
-> VAE/upsampler/audio/export -> GPU fence + reader drain
-> terminal sample -> result/receipt/quality -> process exit check
```

Measured run 固定 seed、输入、步骤、frames/fps、LoRA/ANE/compiled 状态；warmup、冷启动和 fresh-process run 分开统计。

### 5.3 样本数量

```text
3 warmup
20 matched measured
10 cancel/fault
2 source mutation
1 fresh-process rerun
1 independent verifier rerun
```

设备噪声过大时增加样本，不能删除 outlier 让门槛通过。温度、后台负载、文件 cache 状态写入 environment artifact。

## 6. Process-tree Sampler

每点记录 monotonic time、pid/ppid/role/correlation id、resident/phys_footprint、MLX/GPU accounting、swap/compression/pressure、pool/slot/fill/fence、I/O queue 与 phase。

```text
tree_bytes(t) = Σ process_bytes(pid, t)
tree_peak = max_t tree_bytes(t)
```

不能将各进程独立 peak 相加。parent/worker 由 correlation id 绑定；最大 sample gap 超过 record 门槛时标记 `INCONCLUSIVE`。

## 7. 四臂对照实验

| Arm | 说明 | 目的 |
|---|---|---|
| A resident | 当前 dev/default resident 或 compiled/ANE 路径 | 速度/质量基线 |
| B exact streaming | public exact selector + reviewed candidate | 目标内存路径 |
| C legacy bounded | 旧 memory guard/streamed/offload | 兼容对照 |
| D pressure-resident | resident + 独立受控压力 helper | swap/reclaim 对照 |

如果 A 在目标设备本来就无法运行，记为 unavailable，不换 workload。Arm D 不修改系统 swap 参数、不使用 root、不把压力 helper 算入 TurboCider target peak。

四臂记录 total/text/denoise/VAE/export、tree/MLX peak、swap delta、logical/physical I/O、fill/reader wait、pool churn、cancel/drain、failure/quarantine 和质量。

## 8. 验收门

### 8.1 P0 默认路径非回归

```text
>=20 warm paired requests
wall median ratio <= 1.02
wall P95 ratio <= 1.05
quality/output parity PASS
public probe/catalog/lease/receipt/worker allocations == 0
```

### 8.2 P1 Generic framework overhead

固定 source/layout/P/G/K/D/Q/pool/component policy：

```text
wall median <= 1.02
wall P95 <= 1.05
denoise median <= 1.02
output/quality parity PASS
receipt/layout identity PASS
```

### 8.3 P2 Target fit

```text
20 measured runs success
fresh-process rerun PASS
independent verifier PASS
tree_peak + H(T) <= T for every measured run
no unexpected swap failure
```

任一 measured run 超限即 fail，不能用平均峰值掩盖。

### 8.4 P3 Streaming vs swap

只有 B 的 median/P95 优于 D、置信区间不跨越失败门、swap/compression 显著更低且质量/失败/drain 不劣，才可写“相对 swap pressure 更快”；否则只能写“降低峰值/系统压力”。

### 8.5 P4 策略探索

P/G/K/D/Q/pool 的变化单独统计；策略收益不能补偿 P0 默认回归。排序报告包含速度、峰值、logical read、I/O wait、failure rate 和 confidence interval。

## 9. 故障注入矩阵

| 故障 | 注入点 | 预期 | 是否 quarantine |
|---|---|---|---:|
| source path replace | pre-GPU/post-fill | source changed，拒绝结果 | pre-GPU 否，in-flight 是 |
| same-size rewrite | source file | ctime/content digest mismatch | 视 reader 状态 |
| short read | FillJob | 停止新 dispatch，drain | 是 |
| mailbox overflow | completion | sticky failure | 是 |
| reader fence lost | GPU callback | drain unknown | 是 |
| cancel before first fill | coordinator | cancelled，无 authority | 否 |
| cancel during denoise | owner pump | stop + drain | drain unknown 时是 |
| worker SIGKILL | LTX worker | parent 标记 failed/quarantine | 是 |
| catalog revoke | revalidate | stale，不提交 GPU | 否 |
| actual receipt mismatch | finish | 无 public success | 是 |

每个故障 artifact 至少记录 `expected_error`、`actual_error`、`output_exists`、`drain_completed`、`quarantine`、`retry_allowed`。

## 10. Evidence Bundle 与 Catalog Builder

### 10.1 目录

```text
results/streaming/<model>/<date>/<campaign>/
  manifest.json
  environment.json
  build-identity.json
  source-identity.json
  workload-card.json
  candidate-plan.json
  simulator.json
  sampler.jsonl
  raw-samples.jsonl
  receipt.json
  metrics.json
  quality.json
  faults.json
  audit.json
  verification-rerun.json
  review.json
  summary.json
```

### 10.2 Builder 输入门

Builder 只接受：

```text
schema valid
source/runtime/workload/device exact match
independent verifier PASS
P0/P1/P2 required gates PASS
model/performance/release review complete
record digest reproducible
not revoked
```

Builder 输出的 `StreamingPresetRecord` 只读编译进 production catalog。App 和环境变量不能覆盖 catalog 内容。

### 10.3 Record 撤回

发现质量、峰值、source 或性能问题时：

```text
mark release.revoked = true
increment catalog revision
ship new app/native catalog
existing resolved jobs -> stale at revalidate
new options -> unavailable/revoked
```

不得原地修改旧 record 的 layout、calibration 或 runtime identity。

## 11. CI 与本机命令

每个 common-runtime PR：

```sh
env TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
make test-streaming-host
make test-streaming-contract
make test-streaming-audit
tools/native/build_app.sh
git diff --check
```

每个 adapter PR：

```sh
python3 -B tests/native/test_<model>_streaming_descriptor.py
python3 -B tests/native/test_<model>_public_streaming.py
```

真实 campaign（拟议）：

```sh
python3 -B tools/native/run_streaming_campaign.py \
  --model z-image-turbo \
  --workload profiles/streaming/workloads/z-image-turbo-image-512.json \
  --candidate profiles/streaming/candidates/z-image-turbo/p14-k2-d0-q1.json \
  --arm exact-streaming --repetitions 20

python3 -B tools/native/verify_streaming_campaign.py \
  --campaign results/streaming/z-image-turbo/2026-09-17/<campaign> \
  --independent
```

上述命令若尚未实现，必须在文档/CI 中标为 TODO；不能把命令存在于设计稿误报成工具已经可用。

## 12. 阶段发布顺序

```text
R0 host/contracts/source lease/receipt
R1 simulator + campaign schema
R2 Z-Image 一个 workload/target staging
R3 Flux 9B 一个 workload/target staging
R4 H3 Turbo 一个 workload/target staging
R5 LTX worker/full closure staging
R6 App options + JobStore transaction
R7 production catalog + release/revoke
R8 扩展其他 shape/device/target
```

每个阶段都有停止条件：

| 阶段 | 停止条件 |
|---|---|
| R0 | lease、receipt、Off audit、fake fault 全绿 |
| R1 | simulator 与 golden trace 稳定，campaign artifact 可重放 |
| R2–R5 | 模型 full request、P0/P1/P2、fault、independent verifier 全绿 |
| R6 | UI 只显示五档，resolve→persist→generate 原子事务全绿 |
| R7 | reviewed record、catalog builder、staging smoke、rollback 全绿 |
| R8 | 每个新增 workload/device 单独有 evidence，不复用不匹配 record |

任一阶段失败都保留当前可工作的旧路径；不通过删除资格门或开启 experimental bool 来继续发布。

## 13. 本方案对“swap 还是 streaming”的可验证结论

设计上不能先验断言 streaming 永远更快。可以验证的分解是：

```text
streaming 的成本 = 受控 source I/O + fill/reader overlap + pool 管理
swap 的成本      = OS reclaim + page fault + 压缩/解压 + 不可预测调度
```

在模型 working set 远超物理内存时，streaming 可能以更低峰值和更稳定 tail 获胜；在磁盘慢、模型小或 resident 能完整放入内存时，resident 可能最快；在压力不高时，swap 对照可能与 streaming 接近。最终结论必须来自同 workload、同设备、四臂 ABBA/P3 证据，而不是凭实现直觉。

## 14. 完成定义

工具链与发布设计完成意味着：

1. 候选可由 inspect→compile→simulate→campaign→verify→builder 重放；
2. 每个 public record 有完整身份、峰值、性能和质量证据；
3. 默认 Off 路径经 audit 证明零新增热路径工作；
4. fault/cancel/source mutation 的结果和 quarantine 可解释；
5. swap 结论有独立对照，不能把“可运行”写成“更快”；
6. revoke/rollback 不需要改用户模型文件或清空整个 JobStore；
7. 没有 evidence 的档位在 App 中显示 unavailable，而不是猜测可用。
