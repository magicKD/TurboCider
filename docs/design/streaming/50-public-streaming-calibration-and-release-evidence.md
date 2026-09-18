# 50 · Public Streaming 档位校准、性能实验与发布证据

修订日期：2026-09-18。状态：**校准与发布运行手册；当前没有 production record**。

本文定义如何把用户选择的 `Off / 8 / 10 / 12 / 16 / 20 GiB` 转换成经过验证的内部 preset。它回答三个问题：

1. 如何为 LTX、H3 Turbo、Z-Image Turbo、Flux 9B 搜索合适的 `P/G/K/D/Q` 和 pool policy；
2. 如何测量完整 process tree、GPU/Metal/MLX、swap 和输出质量，而不是只看一个瞬时 counter；
3. 如何把候选实验结果转换为可撤回的 staging/production catalog record，并保证 default/resident 路径不受影响。

本文的所有数字都是验收规则或实验协议，不是当前已经测出的模型成绩。没有 clean evidence 和独立 verifier 的数据不能写成 public 支持。

## 1. 校准对象和命名

### 1.1 一个 record 的完整键

record 不是“模型 + 8 GiB”这么简单，最少由下面的 tuple 唯一决定：

```text
model_id
source_digest / logical artifact closure
workload_digest
runtime_revision
device_class / execution_container
component_policy_revision
target_tier
layout_digest
```

其中 workload 至少包括：

- width/height 或 latent shape；
- frames、audio、batch；
- denoise steps、scheduler revision；
- prompt/token shape/conditioning revision；
- VAE/upsampler/export branch；
- dtype/quantization/compiled graph route。

如果 shape 或 component branch 不在 identity 中，就不能把一次校准泛化到另一种请求。

### 1.2 target tier 的含义

```text
8 GiB  = 8 * 2^30 bytes
10 GiB = 10 * 2^30 bytes
12 GiB = 12 * 2^30 bytes
16 GiB = 16 * 2^30 bytes
20 GiB = 20 * 2^30 bytes
```

它是完整请求 process tree 的预算目标，不等于：

- GPU 物理显存容量；
- `mx::get_memory()` 的单次峰值；
- 主进程 RSS；
- 操作系统的绝对 hard cap。

验收上限：

```text
allowed_peak = target_bytes - max(512 MiB, ceil(target_bytes * 0.10))
```

只有 `tree_peak_p95 <= allowed_peak` 且没有未知 child、采样 gap 或非计划 swap，record 才可以进入 fit 候选。

### 1.3 candidate、staging、production

| 状态 | 可以做什么 | 不可以做什么 |
|---|---|---|
| `candidate` | 本机实验、开发者 explicit request | App 默认推荐、自动 fallback |
| `tentative` | 记录数据不完整或统计不充分 | 作为 fit record |
| `staging` | 内部 App/QA 选择，带 feature gate | 对所有用户可见 |
| `production` | App 正式显示并可执行 | 绕过 revoke/identity 检查 |
| `revoked` | 保留审计和回滚信息 | 新请求选择 |

catalog builder 只能从 `verified` evidence 生成 staging/production record；不接受手工复制 JSON 作为发布输入。

## 2. 校准流水线

### 2.1 六个阶段

```text
inspect
  -> compile candidate layouts
  -> simulate static/resource bounds
  -> campaign real requests
  -> independent verify evidence
  -> build/review/release catalog
```

每一阶段都有明确的输入和失败出口：

| 阶段 | 输入 | 输出 | 失败时 |
|---|---|---|---|
| inspect | model root、route、workload | source closure、metadata identity | 不完整 closure，停止 |
| compile | descriptor + candidate profile | exact layout candidates | checked overflow/unsupported |
| simulate | layout + ledger model | static peak/read/overlap | 公式不闭合，不能实测 |
| campaign | candidate + real device | raw run/evidence bundle | crash/quality mismatch |
| verify | raw evidence | verified record candidate | digest/统计/采样不一致 |
| release | verified candidates | staging/production catalog | reviewer 拒绝或 revoke |

### 2.2 inspect 输出

`inspect` 必须输出机器可读 JSON，不直接生成可执行 record：

```json
{
  "schema": "tc-streaming-inspect-v1",
  "model_id": "flux2-klein-9b",
  "route": "bf16-eager-gpu",
  "source": {
    "root": "redacted",
    "logical_ids": [],
    "file_count": 9,
    "source_digest": "sha256:...",
    "closure_complete": true
  },
  "workload": {
    "shape": [512, 512],
    "steps": 20,
    "batch": 1,
    "feature_digest": "sha256:..."
  },
  "unsupported_reasons": []
}
```

必须拒绝：

- index 引用不存在 shard；
- parser 运行中需要 path reopen 的 artifact；
- route 同时声明 ANE/compiled/LoRA 但 policy 未覆盖；
- source logical id 重复；
- workload identity 缺少影响 memory/quality 的字段。

### 2.3 compile 候选生成

候选不是全排列暴力搜索。每个模型提供 profile：

```json
{
  "model_id": "h3-turbo",
  "prefix": [0, 1, 2, 4, 8, 16],
  "group": [1, 2, 4],
  "slots": [1, 2, 3, 4],
  "prefetch": [0, 1, 2],
  "io_workers": [1, 2, 3, 4],
  "pool_policy": ["serial", "retain_all"],
  "pass_transition": ["reload", "carry_first_group"]
}
```

生成器先做静态剪枝：

```text
G <= blocks_in_class
K <= available_group_count
D < K
Q <= K
prefix + K * slot_capacity + scratch <= target - safety_margin
pass_transition 在 adapter capabilities 中
pool_policy 与 component policy 匹配
```

静态剪枝只减少实验数量，不能直接宣布 fit。GPU driver、MPSGraph、VAE、worker/IPC 和 allocator fragmentation 必须由完整 request 实测。

### 2.4 simulate 的静态 ledger

每个候选产生以下 projection：

```text
prefix_bytes
per_slot_backing_bytes
per_slot_scratch_bytes
pool_count
pool_live_intervals
max_live_slot_bytes
io_queue_bytes
known_component_overlap_bytes
estimated_process_overhead
```

模拟器必须使用 checked arithmetic：

- 加法前检查 `UINT64_MAX - lhs >= rhs`；
- 乘法前检查 `lhs == 0 || rhs <= UINT64_MAX / lhs`；
- 所有 Metal/MLX 对齐都在同一函数中完成；
- 不接受负数或隐式截断；
- 输出 `unknown` 而不是把缺失数据当零。

### 2.5 campaign 真实运行

每个 candidate 至少包含：

```text
warmup: 3
paired measured requests: 20
quality replay: 3
fault/cancel probes: 3
```

正式 P0/P1 的 20 个 paired request 必须随机化 ABBA 顺序：

```text
pair i: baseline -> candidate 或 candidate -> baseline
```

同一 pair 使用相同 seed、prompt digest、shape、steps 和 output policy；baseline/candidate 之间不共享可变 session、slot 或 receipt。

冷 cache、热 cache、首次模型加载必须分层统计，不能把初始化优势误当作 executor 性能。

## 3. 四个模型的候选搜索策略

### 3.1 Z-Image Turbo

先冻结 single denoiser、BF16 eager GPU、512×512、batch 1、固定 steps 的 workload。候选顺序：

```text
低档位：P0/P1 + G1 + K1 + D0 + Q1 + reload
中档位：提高 K 或 D1，保持 serial pool
高档位：增加 prefix，尝试 K2/D1/Q2
```

VAE 边界必须单独测量。若 transformer pool 在 VAE 开始时仍有 pending reader，候选必须标记 `boundary_overlap=true` 并拒绝进入 public，直到 adapter 证明 drain/release 顺序。

### 3.2 Flux 9B

Flux 至少有 dual block 和 single block 两类 pool。候选必须分别记录：

```text
dual_pool:   P/G/K/D/Q/policy
single_pool: P/G/K/D/Q/policy
```

低档位优先 serial/reload，避免 retain-all 同时持有两个 pool。高档位才比较：

- dual/single 两池 retain-all；
- denoiser 结束时 transformer backing drain/release；
- text encoder/VAE 是否与 denoiser overlap；
- `/dev/fd` lease loader 是否产生 lazy read。

Flux 的 9B public record 不能套用 4B compiled graph record；runtime identity、source closure 和 route 都必须不同。

### 3.3 H3 Turbo

H3 只校准 MiniMax H3 Turbo。先固定 reload/K1 作为正确性基线，再比较 K2/G1/carry：

```text
baseline: K1, D0, Q1, reload
candidate: K2, D1, Q1/2, carry_first_group
```

carry 的收益必须同时报告：

- carry 的额外 live bytes；
- pass boundary 等待时间；
- group receipt 中的 `content_generation`；
- 与 reload 输出的逐元素差异。

如果 carry 只在某个 block count/shape 成立，layout digest 必须包含该约束。

### 3.4 LTX 2.5

LTX 必须按 stage 独立搜索，不把 stage1/stage2 的 K 简化为一个全局 K：

```text
stage1:      P1/G1/K1..3/D/Q
upsampler:   component boundary policy
stage2:      P2/G1/K1..3/D/Q
VAE/export:  release-before-enter 或经证据允许的 overlap
```

低档位先禁止跨 stage overlap，确保 peak 可控；高档位再验证 upsampler 与 stage transition 的 overlap。每个 LTX record 必须含 `streaming_stages[]` 和 `streaming_boundaries[]`，不能把两个 stage 压成一个 legacy summary。

## 4. 进程树和 swap 四臂实验

### 4.1 四个 arm

| Arm | 执行方式 | 目的 |
|---|---|---|
| A | resident/default | 当前最佳性能和 P0 基线 |
| B | public streaming exact layout | 验证框架 overhead 和峰值 |
| C | streaming + bounded guard | 验证目标档位 admission/失败行为 |
| D | resident/default + 自然系统 pressure/swap | 现实低内存对照 |

Arm D 只在专门实验机器运行，不能在开发者日常机器上隐式创建 pressure。压力脚本必须：

- 明确记录 physical memory、OS build、background processes；
- 在实验结束后释放压力并检查 swap 状态；
- 使用与 A/B/C 相同的 model/source/workload；
- 记录成功率、swap-in/out、wall、P95、输出质量和系统响应。

结论只能是“在某设备/某 workload/某 target 下 B 比 D 的某些指标更好”，不能泛化成 streaming 永远比 swap 快。

### 4.2 采样器要求

process-tree sampler 必须递归发现 child，记录 PID 生命周期和未知 child。每个 sample：

```text
wall_time_ns
pid/ppid
rss
compressed
swap_out
gpu_or_metal_bytes
phase/stage/pass/group
```

采样器终止时必须有 terminal sample；任意 gap >100 ms、未知 child 或采样器异常都将 evidence 标为 `inconclusive`。

### 4.3 峰值与尾延迟

每次 run 输出：

```text
tree_peak_bytes
tree_peak_phase
driver_peak_bytes
swap_out_total
wall_seconds
denoise_seconds
first_output_seconds
terminal_status
```

不能只报告平均值。正式报告至少包含 median、P95、MAD/置信区间、成功率和失败分类。

## 5. 性能统计和质量验收

### 5.1 P0 默认路径

执行顺序建议：

```text
build baseline binary
build candidate binary
ABBA resident/default requests
audit allocation/thread/hook counters
compare output digest/quality
```

发布门：

```text
wall median ratio <= 1.02
wall P95 ratio <= 1.05
denoise median ratio <= 1.02
new default allocation = 0
new default thread creation = 0
new default streaming hook = 0
```

如果 baseline/candidate runtime identity不同，必须在 evidence 中说明原因，并重新建立配对；不能把不同 commit 的非配对数据当作 P0。

### 5.2 P1 同布局

比较 `generic_stage_executor_v2` 与模型 direct/private executor 时，必须使用完全相同：

- source closure；
- layout digest；
- P/G/K/D/Q；
- pool policy；
- output route；
- seed/prompt/shape；
- cache 温度。

P1 只测 framework overhead，不能通过改变 layout 来换取更快结果。输出若不是 byte-exact，必须提供明确的 numeric tolerance 和独立质量分析；否则失败。

### 5.3 P2 bounded fit

每个 target 至少需要：

```text
20 successful full requests
3 cancel/fault requests
3 source mutation/replace requests
0 unknown child
0 unplanned swap out
terminal sample present in 100% runs
```

峰值必须使用 P95 或更严格的 policy；单次最低峰值不能生成 fit record。cancel/fault 不应被计为 successful fit，但必须证明 cleanup 不泄漏/不 unsafe free。

### 5.4 质量和输出

image：保存 PNG/latent digest、尺寸、通道和 finite 检查。video/audio：分别保存 frame/audio digest、时长、采样率、编码成功状态。模型允许非确定性时，记录 seed、tolerance 和 perceptual metric。

任何 output mismatch 都先标记 `output_verification_failed`，不能仅因 peak 达标而发布。

## 6. evidence bundle

每次 run 使用不可变目录：

```text
evidence/<run_id>/
  request.json
  normalized_request.json
  source_manifest.json
  device.json
  runtime.json
  resolved_record.json
  layout.json
  actual_receipt.json
  process_tree.jsonl
  counters.json
  output_manifest.json
  stdout.log
  stderr.log
  command.txt
  environment.json
  checksums.sha256
  verifier.json
```

`environment.json` 至少包括：

- git commit、dirty status、binary SHA256；
- OS build、GPU name、physical memory；
- MLX/Metal/runtime revision；
- model source digest；
- sampler revision；
- catalog snapshot revision。

dirty worktree 默认不能生成 production record，除非 reviewer 显式批准并将 dirty diff digest 加入 identity。更安全的默认是要求 clean tree 重新运行。

## 7. 独立 verifier

verifier 不加载模型、不调用 GPU、不信任运行时的“成功”布尔值。它只读取 evidence bundle，检查：

```text
request digest == resolved request digest
record identity == source/workload/runtime/device identity
authority digest == layout/record digest
receipt stages/groups/pools/fences complete
receipt source generation == lease generation
post-drain source revalidation == pass
process tree samples complete and gap-safe
peak + margin <= target
output digest/quality policy == pass
```

verifier 输出：

```json
{
  "status": "verified",
  "run_id": "...",
  "checks": {
    "identity": "pass",
    "receipt": "pass",
    "drain": "pass",
    "memory": "pass",
    "quality": "pass",
    "performance": "pass"
  },
  "record_candidate": "sha256:..."
}
```

任何 `inconclusive` 都不能被 builder 自动升级成 `verified`。

## 8. catalog builder、review 和 revoke

### 8.1 builder 规则

builder 接受一组经过 verifier 的 evidence，并生成一个 deterministic record：

```text
record_id = sha256(canonical(record_without_release_fields))
```

它必须检查：

- 同一 identity 不存在不同 layout 的两个 production record；
- target tier 是允许的五档之一；
- P/G/K/D/Q 与 adapter capabilities 匹配；
- source/workload/runtime/device identity 非空；
- P0/P1/P2/P3 所需证据全部存在；
- calibration peak 使用同一 sampler/headroom policy；
- verifier revision 与 catalog schema 兼容。

### 8.2 review checklist

reviewer 至少签核：

1. model route 是否确实是 GPU-only public 范围；
2. source closure 是否覆盖 text/VAE/tokenizer/upsampler/audio 等 component；
3. Off/default P0 是否无回归；
4. target peak 是否来自完整 process tree；
5. candidate 是否按同一 layout 与 direct baseline 比较；
6. cancel/fault/source mutation 是否安全；
7. App 是否只显示 target，不泄漏内部布局；
8. revoke/rollback 是否能只撤回一个 record。

### 8.3 revoke/rollback

revoke 不删除历史 evidence，只把 record 状态改成 revoked，并增加：

```text
revoked_at
revoked_by
reason
replacement_record_id (optional)
```

resolver 读取新 catalog snapshot 后立即拒绝 revoked record。正在运行的 request 不被中途切 layout；若 runtime/source 发生变更，运行完成后仍需 verifier 决定结果是否可见。

## 9. 推荐的实机实验顺序

每台机器先执行：

```text
1. default/resident warmup
2. P0 ABBA
3. 每模型 inspect/source closure
4. candidate compile/simulate
5. 最低 target（8 GiB）serial/K1
6. 10/12 GiB 增加 K/D/Q
7. 16/20 GiB retain/carry/multi-stage overlap
8. P2 process-tree fit
9. P3 四臂对照
10. cancel/fault/source mutation
11. independent verify
12. staging catalog + App QA
```

如果最低档位 serial/K1 都不满足 quality 或 target，应记录 `unavailable`，而不是继续盲目增大 Q 或缩小 buffer。高档位也不能跳过低档位的 source/receipt/cleanup 证据。

## 10. 当前仓库的下一批实施任务

### T1 · Flux lease loader

- 完成 `Weights::load_lease()` 编译和 host fixture；
- 验证 `/dev/fd` 对 safetensors loader 的真实行为；
- 明确 lazy mmap 的 fd 生命周期；
- 把 text/VAE logical IDs 写入 Flux public test；
- default `Weights::load()` 路径计数器保持零 streaming work。

### T2 · common multi-stage receipt

- additive `streaming_stages[]` 和 `streaming_boundaries[]`；
- 单 stage v2 replay 兼容；
- stage order/count mismatch；
- boundary drain/release verifier。

### T3 · H3 Turbo public hook

- source lease + probe/snapshot/authority；
- reload/K1 full request；
- carry/K2 receipt；
- default P0。

### T4 · LTX worker

- worker-local authority/lease；
- SIGKILL/EOF/quarantine；
- stage1→upsampler→stage2 transaction；
- multi-stage evidence。

### T5 · App and catalog

- options query + physical-memory recommendation；
- 五档 UI 与 unavailable state；
- JobStore stale/revoke/atomic output；
- staging catalog 的单 record rollback。

## 11. 发布前最终签字表

```text
[ ] model/source/workload/runtime/device identity 已固定
[ ] source closure 与 fd lineage 已验证
[ ] exact layout 与 receipt verifier 通过
[ ] Off/default P0 通过
[ ] same-layout P1 通过
[ ] target P2 通过，完整 process tree 无未知 child/gap
[ ] resident/streaming/bounded/swap 四臂数据已归档
[ ] quality/output digest 已签核
[ ] cancel/fault/source mutation/worker kill 已验证
[ ] App 只暴露 Off/五档，未暴露 P/G/K/D/Q
[ ] unavailable 不会静默 fallback
[ ] catalog builder 从 verified evidence 生成 record
[ ] dev merge 后 identity/evidence 已重跑
[ ] revoke/rollback 演练成功
[ ] production catalog 变更由 reviewer 签名
```

任何一项未完成，模型/档位保持 `unavailable` 或 `staging`。这套规则的目的不是阻止实验，而是防止把“能跑一次”误报成“低内存机器可稳定 public 使用”。
