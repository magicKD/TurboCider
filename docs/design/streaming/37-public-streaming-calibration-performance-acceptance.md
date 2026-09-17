# 37 · Public Streaming 校准、性能保护与发布验收实施规格

[目录](README.md) · [代码实施规格](36-public-adapter-code-implementation-spec.md) · [代码合同工作台](38-framework-code-contracts-and-implementation-workbench.md) · [验收工作簿](39-validation-benchmark-and-release-workbook.md) · [档位基础规格](34-model-tier-calibration-and-release-spec.md) · [当前进度](13-implementation-progress.md)

修订日期：2026-09-17。状态：**校准与发布操作规格；production catalog 为空，所有档位均需真实证据后才能开放。**

本文把“怎么测、怎么判定、怎么生成 catalog、怎样证明不影响性能”写成可以交给测试、性能、模型和 release owner 执行的操作手册。所有数字均为验收规则或候选搜索边界，不是当前已支持的 public 档位。

## 1. 发布边界与术语

### 1.1 用户语义

App 高级设置只显示：

~~~text
Streaming Off
8 GiB
10 GiB
12 GiB
16 GiB
20 GiB
~~~

用户不看到：

~~~text
resident_prefix_blocks
block_group_size
slot_count
prefetch_distance
io_workers
pass_transition
multi_pool_policy
worker/reader 参数
~~~

target 是完整请求 process tree 的校准目标，不是 GPU VRAM、denoiser-only peak 或单个模型文件大小。

### 1.2 当前不可声称

- production catalog 为空；
- 四模型只有 private exact/candidate 基础；
- 没有任何 8/10/12/16/20 GiB target 被完整校准；
- 没有结果可以直接证明 streaming 一定比系统 swap 快；
- GPU+ANE streaming、LoRA、量化缓存、compiled graph 和其他 checkpoint 不因通用框架存在而自动获得资格。

### 1.3 一个 record 的最小资格

一个可发布 record 必须同时绑定：

~~~text
model + operation + execution + shape + frames/fps/steps/audio
source artifact/manifest identity
TurboCider build/runtime/adapter/reader/kernel identity
device class + physical memory range
execution container
component policy
sealed layout digest
calibrated full-request peak
performance evidence
quality evidence
review/release evidence
~~~

任何字段缺失都只能是 candidate 或 unavailable，不能是 public-stable。

## 2. 配置文件分层

### 2.1 仓库目录

建议目录：

~~~text
profiles/streaming/
  schema-v2/
    selector.schema.json
    catalog-record.schema.json
    workload-card.schema.json
    device-profile.schema.json
  devices/
    apple-m4max-64g.json
    apple-m4pro-24g.json
  workloads/
    z-image-turbo-image-512.json
    flux2-klein-9b-image-512.json
    minimax-h3-turbo-video-512.json
    ltx-2.5-distilled-video-768.json
  candidates/
    z-image-turbo/
    flux2-klein-9b/
    minimax-h3-turbo/
    ltx-2.5-distilled/
  catalog/
    production.json
    staging.json
  evidence/
    <model>/<device>/<workload>/<campaign>/
~~~

production catalog 当前应是空 records 的编译资源。candidate JSON 只能被 calibration runner 读取，不能被运行时自动发现。

### 2.2 User selector

用户 JSON 仅允许：

~~~json
{
  "schema_version": 2,
  "enabled": true,
  "selection": "memory_tier",
  "retention": "request",
  "target_request_memory_bytes": 10737418240
}
~~~

禁止把 P/G/K/D/Q 或布局 alias 放进 user schema。exact preset replay 只用于 JobEnvelope、debug 和 catalog replay，不显示给普通 App 用户。

### 2.3 Workload card

每个 workload card 必须冻结：

- model/operation/execution；
- width/height/frames/fps/steps/batch/audio；
- prompt token shape（valid/padded/compute）；
- input/reference 数量和尺寸；
- dynamic text；
- conditioning、VAE、upsampler、audio policy；
- LoRA 必须明确为 none，或单独 record；
- output/export format；
- warmup、重复次数和 seed 集合。

一个 record 不得用 512² 的证据覆盖 1024²、image.edit、不同 frames 或不同 audio policy。

## 3. Target 与预算判定

### 3.1 固定 target

~~~text
8 GiB  =  8  × 2^30
10 GiB = 10  × 2^30
12 GiB = 12  × 2^30
16 GiB = 16  × 2^30
20 GiB = 20  × 2^30
~~~

### 3.2 Guard margin

统一计算：

~~~text
H(T) = max(512 MiB, ceil(0.10 × T))
eligible(T) =
  calibration.complete
  && calibrated_request_bytes + H(T) <= T
  && device physical range matches
  && record not revoked
~~~

calibrated_request_bytes 必须包含：

- App/parent；
- LTX worker（如有）；
- text encoder/conditioning；
- denoiser；
- VAE/upsampler/audio；
- export/finalizer；
- worker exit/drain；
- request startup 到 terminal cleanup 的全窗口。

不能把瞬时 MLX peak 直接填入 calibrated_request_bytes。MLX peak 是诊断字段，完整 process-tree peak 才是 eligibility 输入。

### 3.3 选择规则

按以下顺序筛选：

~~~text
target 支持
-> record release channel
-> model/workload/source/runtime/device exact identity
-> calibration complete
-> calibrated + margin <= target
-> component/route compatibility
-> rank
-> calibrated bytes（越小越优）
-> logical read bytes（越小越优）
-> preset ID/revision（稳定 tie-break）
~~~

未找到 record 时返回 unavailable，不自动降低 target、不偷偷改 K、不切换 resident、不启用系统 swap。

## 4. 四模型候选探索

### 4.1 通用搜索顺序

候选生成不得直接从 target 推导唯一布局，而要在 adapter capability 内进行离散搜索：

~~~text
descriptor capability
-> 固定正确性/格式约束
-> 生成合法 P/G/K/D/Q/pool candidates
-> simulator 估算峰值和 I/O
-> 先跑最有希望的少量 candidates
-> 实测 full-request memory
-> 质量/生命周期筛选
-> 性能排序
~~~

固定禁止：

- 运行时动态改变已授权布局；
- 为了 fit target 自动改变 dtype；
- 为了 fit target 自动启用 LoRA fusion/quantized cache；
- 把不支持的 multi-pool policy 强行降级为 serial；
- 在 record 中混入未测的 shape/component。

### 4.2 Z-Image Turbo

当前 seed：

~~~text
Comfy BF16 GPU
P14/G1/K2/D0/Q1
single pool
reload
~~~

候选首轮只改变 descriptor 已支持的 P/K/D/Q；保持 block order、dtype、kernel、component policy 不变。候选矩阵示例：

~~~text
P ∈ {0, 4, 8, 12, 14}
G = 1
K ∈ {2}（K>2 需 adapter 单独认证）
D ∈ {0, 1} 且 D<K
Q ∈ {1, 2} 且 Q<=K
pool = serial
~~~

P14/G1/K2/D0/Q1 的 tiny P1 证据只说明同布局框架成本，不代表 8/10/12/16/20 GiB 已有可用 record。

### 4.3 Flux.2 Klein 9B

当前 seed：

~~~text
BF16 eager GPU
8 dual + 24 single blocks
dual/single retained multi-pool
K2/G1/D1/Q2
pool=retain_all
reload
~~~

候选必须保留 dual/single class boundary：

~~~text
P = 0（当前 descriptor）
G = 1
K = 2（K3/K4 需新 evidence）
D ∈ {0, 1}
Q ∈ {1, 2}
pool = retain_all
~~~

如果降低 target 需要 serial pool 或 K1，必须重新认证 class barrier、peak、quality 和性能；不能把 current retained record 的 calibrated bytes 改写为新策略。

### 4.4 MiniMax H3 Turbo

当前 seed：

~~~text
original BF16 native Metal GPU
P0/G1/K2/D1/Q1
carry_first_group
single pool
~~~

候选优先顺序：

~~~text
P ∈ descriptor-supported prefix values
G = 1
K = 2
D = 1
Q = 1
pass_transition = carry_first_group
~~~

H3 carry 的 fill 计数、slot rotation、最后一个 pass 的无残留 carry 必须通过 receipt v2。普通 H3、VDN-H3、FastH3 和其他变体不得复用 H3 Turbo record。

### 4.5 LTX 2.5 distilled

当前 seed：

~~~text
C/Metal exact
P8/G1/K3/D2/Q3
request-scoped worker
component policy separately measured
~~~

LTX 候选必须将 denoiser、upsampler、video VAE、audio path 和 finalizer 作为完整 resource closure。不能只按 denoiser blocks 估算 target。

首轮搜索：

~~~text
P ∈ descriptor-supported values around 8
G = 1
K ∈ {2, 3}
D < K
Q ∈ {1, 2, 3} 且 Q<=K
component policy 固定
~~~

LTX 的 parent/worker/finalizer 进程树必须以同一 correlation id 采样，不能拿单独 CLI 的 peak 代替 App 真实闭包。

## 5. Calibration Runner 实施

### 5.1 工具分工

建议实现：

~~~text
tools/native/streaming_inspect.py
  检查 source/index/descriptor/workload card

tools/native/streaming_compile_plan.py
  生成并校验 candidate layout，不执行 GPU

tools/native/streaming_simulate.py
  离散事件模拟 peak、I/O、wait、overlap

tools/native/streaming_process_sampler.py
  采集 parent/child process-tree 统一时间线

tools/native/run_streaming_calibration.py
  运行 warmup/candidate/repeat，写 evidence bundle

tools/native/run_streaming_campaign.py
  ABBA/P1/P2 campaign

tools/native/verify_streaming_campaign.py
  独立解析/统计/质量/identity verifier

tools/native/build_streaming_catalog.py
  只从 reviewed evidence 生成 catalog

tools/native/lint_streaming_catalog.py
  digest、schema、release、目标和 record 唯一性检查
~~~

runner、verifier、builder 必须是三套相互独立的逻辑；不能让生成 record 的同一函数同时充当 verifier。

### 5.2 Candidate ID

candidate id 必须由 canonical 字段生成：

~~~text
<model>/<source-short>/<workload-short>/<device-class>/<target>/<layout-digest>/<runtime-digest>
~~~

人类可读 alias 不能代替 digest。layout 改变、runtime/kernel 改变、source 改变必须产生新 candidate id。

### 5.3 单次运行生命周期

~~~text
create fresh engine/session
-> record baseline host memory
-> source probe/lease
-> load/conditioning
-> denoise passes
-> VAE/upsample/audio/export
-> drain all readers/workers
-> terminal sample
-> release engine
-> verify process exited
~~~

每个 candidate 至少包含：

- 3 次 warmup（不计入统计）；
- 20 个 matched measured requests；
- 10 个 fault/cancel requests（若 adapter 支持）；
- 2 个 source mutation checks；
- 1 个 fresh-process rerun；
- 1 个 independent verifier rerun。

P1 same-layout 可使用 20 pair；P2 target eligibility 还必须有 full-request memory 样本；P3 swap 对照必须单独 campaign。

### 5.4 macOS process-tree sampler

采样器必须同时记录：

~~~text
monotonic timestamp
pid/ppid/process role
resident/phys_footprint（可用 API）
MLX active/peak（若有）
GPU allocation/accounting（若有）
vm.swapusage baseline/current
memory pressure state
I/O bytes/read time
worker/pool/fill counters
correlation id
~~~

峰值计算使用“每个采样时刻的进程树总和”的最大值：

~~~text
tree_peak = max_t Σ process_bytes(pid, t)
~~~

不得把每个进程独立 peak 相加，否则会重复计算不同时刻的内存。记录采样间隔和最大 gap；gap 超过 workload 要求时该 campaign 只能为 INCONCLUSIVE。

### 5.5 采样安全

- 不禁用系统 swap；
- 不修改系统内核参数；
- 不使用 root；
- pressure helper 有硬上限和超时；
- memory pressure 进入 critical 或系统 UI 卡顿时立即终止；
- 所有 helper 退出并回收；
- 记录 baseline 与 delta，不把其他用户进程的变化归因给 TurboCider；
- 任何被中止的 campaign 不能写入 reviewed record。

## 6. 离散事件 Simulator

### 6.1 输入

~~~text
sealed Layout
group bytes/source ranges
pass transition
pool policy
worker count
reader count
measured read latency distribution
kernel group duration distribution
component boundary
target + margin
~~~

### 6.2 事件

~~~text
fill_submit
fill_complete
claim
reader_submit
reader_complete
group_retire
pass_drain
pool_switch
component_release
~~~

### 6.3 输出

- theoretical live slot bytes；
- resident prefix bytes；
- peak backing bytes；
- I/O overlap ratio；
- executor wait ratio；
- estimated wall；
- carry/pool barrier timeline；
- target fit/not-fit；
- candidate pruning reason。

Simulator 只能剪枝和解释，不能替代实体 GPU、真实 process-tree peak、输出质量或 swap evidence。

## 7. Resident、Streaming、Bounded、Swap 四路对照

### 7.1 对照矩阵

同一 workload card、seed、source、build、device 和 output path 使用四臂：

| Arm | 路径 | 目的 |
|---|---|---|
| A | resident/default | 速度和质量基线 |
| B | public/exact streaming | 显式 slot/pool 的收益 |
| C | legacy bounded-memory | 现有 memory guard/旧 streamed 行为对照 |
| D | resident + controlled pressure | 观察系统 reclaim/swap 成本 |

如果 resident 在目标机器自然无法运行，A 只能标为 unavailable，不能用不同 workload 代替。

### 7.2 Pressure harness

高内存机器模拟低内存时，使用独立受限 helper 逐步保留匿名内存，使可用 headroom 接近目标。helper 必须：

- 与 TurboCider 分离进程；
- 有明确最大字节数；
- 每步等待系统状态；
- 达到 swap/pressure 观测阈值后停止；
- 不参与 TurboCider process-tree eligibility，只用于 D 臂的外部压力；
- 记录压力 helper bytes、swap delta 和终止原因。

D 臂的外部压力必须在 A/B/C 之间保持同一策略和同一停止条件；不能为了让 streaming 胜出而只给 resident 加压力。

### 7.3 指标

每臂都记录：

~~~text
request wall / denoise / text / VAE / export
MLX active / MLX peak
process-tree peak
resident prefix / slot backing peak
logical read bytes / physical read bytes
fill count / wait seconds / fence counts
swap used delta / compressed memory delta
page-in/page-out（可用时）
memory pressure transitions
failure/cancel/drain/quarantine
output hash / quality metrics
~~~

### 7.4 “比 swap 更快”的严格结论

不要凭单次 wall 声称更快。只有同时满足才可写入 release note：

~~~text
streaming median wall < pressure-resident median wall
streaming P95 wall <= pressure-resident P95 wall
bootstrap confidence interval 不跨越“streaming 更慢”的 release threshold
streaming swap delta 显著更低或为 0
failure rate 不高于 resident pressure arm
quality/output gate PASS
~~~

如果 streaming 只降低 peak 但没有统计上更快，结论写成：

~~~text
在该 workload/device 上降低了内存峰值并减少系统压力；相对 swap 的速度优势未证实。
~~~

如果 streaming 更慢，但仍是唯一能在 target 下完成的路径，record 可以作为“可完成性” candidate，不能在 UI 或 release note 声称更快。

### 7.5 Break-even 诊断模型

用于解释而非代替实测：

~~~text
T_stream ≈ T_compute + max(T_prefetch, T_compute_overlap)
           + T_unhidden_read + T_barrier

T_swap ≈ T_compute + T_major_fault
         + T_reclaim + T_compress/decompress
         + T_pressure_stall
~~~

关键诊断：

- streaming 的逻辑读取字节高但 overlap 好，可能仍比 swap 快；
- K/D/Q 增大可能降低 wait 但提高 peak；
- pool retain_all 可能减少重建但提高常驻；
- swap delta 为 0 不能证明没有压缩/回收停顿；
- 任何 break-even 结论必须注明 workload、设备、OS、source 和 build identity。

## 8. 性能与内存发布门

### 8.1 P0 默认路径

selector absent/disabled 的 resident、compiled、GPU+ANE、legacy streaming 必须满足：

~~~text
public probe/catalog/hash/authority/receipt allocations = 0
new slot/pool/worker = 0
new synchronize/cache-clear/unload = 0
wall median ratio <= 1.02
wall P95 ratio <= 1.05
quality/output parity PASS
~~~

统计使用 ABBA 或随机交错配对，至少 20 个 warm matched requests；报告 median、P95、样本数、bootstrap interval 和 build/source digests。

### 8.2 P1 同布局框架成本

将 direct exact baseline 与 generic StageExecutor candidate 固定在完全相同：

~~~text
source/layout/P/G/K/D/Q/pool/component policy
~~~

门槛：

~~~text
wall median ratio <= 1.02
wall P95 ratio <= 1.05
denoise median ratio <= 1.02
output/quality PASS
receipt/layout identity PASS
~~~

Flux 9B 当前已有正式 same-layout P1 证据；它不自动转化为 public record。

### 8.3 P2 目标内存资格

每个 target record 必须：

- 完整 process-tree peak + H(T) <= target；
- 最小采样间隔和最大 gap 合格；
- 不发生未预期的 swap pressure；
- reader/pool/drain 全部 PASS；
- 20 个 measured requests 中无 memory failure；
- fresh-process rerun 至少 1 次；
- independent verifier PASS。

如果一次样本超过 target，record 立即 fail；不能用平均值掩盖 peak。

### 8.4 P3 压力与 swap

P3 不是 public 可执行性的前置条件，但决定 release note 是否能声称“优于 swap”。必须独立记录：

- pressure helper 曲线；
- swap/compression；
- resident/streaming/bounded 四臂；
- wall/P95/failure；
- 结果与质量。

P3 INCONCLUSIVE 时，release note 只能写“显式受控 streaming”，不能写“比 swap 快”。

### 8.5 P4 策略优化

比较 P/G/K/D/Q、pool、prefetch 和 worker 的收益时，必须将：

- layout change；
- I/O change；
- kernel change；
- source format change；

拆开报告。策略更快不等于 framework 更快；不能用策略收益补偿 P0 回归。

## 9. Quality、生命周期与故障验收

### 9.1 输出质量

按模型既有质量门：

- image：PNG byte hash 或允许误差的 pixel/L2/cosine；
- video：frame count、duration、pixel metrics、audio presence；
- deterministic seed：同 seed matched pair；
- approximate route：必须明确 approximation flag，不能与 exact parity 混写。

### 9.2 Fault matrix

~~~text
catalog empty/revision mismatch
record revoked
wrong source digest
path replace/rename
short read
worker completion lost
reader fence lost
slot double claim
fill before retire
GPU command failure
cancel before first fill
cancel mid-fill
cancel mid-pass
cancel before VAE/export
drain timeout
worker SIGTERM/SIGKILL
~~~

每一项都要记录：

~~~text
expected error code
result_json 是否为空
output 是否存在/是否标记成功
drain 是否完成
engine/session 是否 quarantine
App 是否允许 retry/recreate
~~~

### 9.3 Quarantine 规则

仅当 backing/GPU reader 生命周期无法证明安全时 quarantine。source 在 GPU 工作前变化属于普通失败；source 在 reader 运行中变化或 drain 不确定则 quarantine。

quarantine engine 不得通过 unload 强行复用。App/worker 必须销毁并重建，历史 job 保留 failure identity。

## 10. Evidence Bundle

### 10.1 目录

~~~text
results/streaming/<model>/<date>/<campaign>/
  manifest.json
  environment.json
  build-identity.json
  source-identity.json
  workload-card.json
  candidate-plan.json
  sampler.jsonl
  raw-samples.jsonl
  metrics.json
  quality.json
  faults.json
  audit.json
  summary.json
  verification-rerun.json
  review.json
~~~

### 10.2 manifest 必填

- campaign id；
- runner/verifier revision；
- source/workload/runtime/device digests；
- candidate/layout/record digest；
- sample count and warmup count；
- sampler interval/max gap；
- pressure strategy；
- output hash set；
- status PASS/FAIL/INCONCLUSIVE；
- failure reason；
- catalog target（若只是 candidate，明确 no-production）。

### 10.3 Review 状态机

~~~text
candidate
  -> measured
  -> independently_verified
  -> model_reviewed
  -> performance_reviewed
  -> release_reviewed
  -> staging
  -> production
~~~

任何阶段失败都回到 candidate/rejected，不允许 builder 直接将 measured 写进 production catalog。

## 11. Catalog Builder 与撤回

### 11.1 Builder 输入

Builder 只接受：

- schema-valid evidence；
- independent verifier PASS；
- model/performance/release review；
- complete full-request calibration；
- exact source/runtime/device/workload identity；
- non-revoked record。

### 11.2 Builder 输出

~~~text
production catalog JSON
catalog revision
record canonical digest
record review digest
catalog linter report
~~~

production runtime 内嵌编译后的 catalog，App/环境变量不能覆盖。

### 11.3 撤回

撤回步骤：

1. record revoked=true；
2. catalog revision +1；
3. options 返回 unavailable；
4. 新 resolve 不得 select；
5. 已经运行的 request 不换 layout，按 authority/source lease 完成或失败；
6. 历史 job 保留 record/resolution/result digest；
7. 写 rollback/reason/reviewer。

撤回触发：

- source artifact 不再可复现；
- kernel/runtime identity 变化；
- target peak 超限；
- output quality regression；
- drain/lifetime bug；
- P0 default regression；
- P1/P3 release claim 被新 evidence 推翻。

## 12. App 验收

### 12.1 UI

- 默认 Off；
- 只显示五个 target；
- unavailable 显示原因，不隐藏为“自动”；
- 不显示 P/G/K/D/Q；
-切换 target 有 debounce；
-旧 options response 不能覆盖新 model/shape；
- resolve 失败不提交 job；
- catalog empty 时 UI明确“当前版本未开放”。

### 12.2 Options 两阶段

未打开 model 时可调用全局 tentative options，但 query_status 必须显示 tentative_without_artifact_identity。打开 model 后调用 engine-scoped options，才可显示 artifact-specific availability。

### 12.3 Resolve → persist → generate

非 LTX：

~~~text
draft selector
-> engine resolve
-> persist exact selector + resolution summary
-> generate（native重新resolve/revalidate）
~~~

LTX worker：

~~~text
spawn worker
-> worker resolve
-> streaming_resolved event
-> App persist JobEnvelope
-> ACK
-> worker generate exact selector
~~~

App 永远不持有 authority、GPU handle、slot backing 或 source fd。

### 12.4 JobEnvelope v2 最小字段

- schema/version；
- request nonce；
- original user selector；
- exact selector；
- resolution/record/layout/source/runtime/device digests；
- model root identity；
- worker pid/role（如有）；
- status/state timestamps；
- result/error/quarantine；
- catalog revision；
- app/build identity。

## 13. 本机执行命令与验收映射

### 13.1 每个代码 PR

~~~sh
env TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
make test-streaming-host
make test-streaming-contract
make test-streaming-campaign
make test-streaming-source-identity
make test-streaming-audit
make test-streaming-pager
tools/native/build_app.sh
git diff --check
~~~

### 13.2 Adapter test catalog smoke

~~~sh
TURBOCIDER_STREAMING_CATALOG=test-only \
TURBOCIDER_STREAMING_MODEL=z-image-turbo \
tools/native/run_streaming_calibration.py \
  --workload profiles/streaming/workloads/z-image-turbo-image-512.json \
  --candidate profiles/streaming/candidates/z-image-turbo/seed.json
~~~

命令是设计约定；具体 CLI 参数在工具实现 PR 中冻结。没有实体模型/Metal 权限时必须输出 SKIP，而不是 PASS。

### 13.3 Campaign verifier

~~~sh
python3 -B tools/native/verify_streaming_campaign.py \
  --campaign results/streaming/<model>/<date>/<campaign> \
  --independent
~~~

verifier 必须能在不启动模型的情况下重算：

- canonical identity；
- expected fill/group/pass；
- performance summary；
- bootstrap interval；
- quality hash；
- target eligibility；
- review/release status。

### 13.4 Default regression

每次 public adapter 或 catalog runtime 改动，都重跑：

- resident GPU；
- resident GPU+ANE；
- compiled GPU；
- legacy residency=streamed；
- memory-constrained legacy；
- Flux 4B compiled；
- H3 ordinary/VDN routes；
- LTX default worker；
- Z-Image GGUF/ConvRot/ANE routes。

## 14. 四模型验收卡

### 14.1 Z-Image Turbo

进入 staging 前必须具备：

- source/layout/runtime identity；
- public probe/compile/generate_resolved；
- fd/lease reader；
- receipt v2；
- 512² GPU workload card；
- 20 matched measured requests；
- process-tree peak；
- output/quality；
- P0/P1/P2；
- failure/cancel；
- independent verifier。

### 14.2 Flux.2 Klein 9B

额外要求：

- dual/single class receipt；
- retain_all pool；
- pool create/reuse evidence；
- concatenate boundary；
- 8/24 block source ranges；
- current same-layout P1 引用；
- full request VAE boundary；
- no Flux 4B promotion。

### 14.3 MiniMax H3 Turbo

额外要求：

- authority 分离 experimental bool；
- carry-first-group 坐标；
- C receipt v2；
- exact generation/pass count；
- cache disabled on public exact；
- only minimax-h3-turbo；
- H3 ordinary/VDN non-regression。

### 14.4 LTX 2.5

额外要求：

- worker-local resolve；
- App ACK handshake；
- parent/worker/finalizer correlation；
- component policy；
- quarantine；
- video/audio/upsampler/VAE closure；
- SIGTERM/SIGKILL evidence。

## 15. Staging 与最小发布顺序

严格顺序：

~~~text
framework host green
-> C ABI/Swift green
-> one model test catalog
-> one model target calibration
-> one model staging record
-> one model production record
-> next target
-> next model
~~~

推荐模型顺序：

~~~text
Z-Image Turbo
-> Flux.2 Klein 9B
-> MiniMax H3 Turbo
-> LTX 2.5
~~~

LTX 放后是因为 worker/process-tree/component closure 更复杂，不是因为 streaming executor 不适合 LTX。

一次 release 只加入一个 model + workload + device range + target record。不要一次把五档和四模型全部放开。

## 16. 合并 dev 的验收

当前 dev 基线为 02148b7；实际合并时以 fetch 后最新已审阅 commit 为准。

合并前：

- C0–C7 对应代码和 evidence 已完成；
- working tree clean；
- production catalog revision 已冻结；
- release rollback commit 已准备。

合并步骤：

~~~sh
git fetch --all --prune
git merge dev
~~~

冲突优先级：

1. 保留 dev 的默认 GPU/ANE/M5 优化；
2. Off/default 不调用 public framework；
3. 未认证 GPU+ANE streaming 继续拒绝；
4. 未 reviewed record 不进入 production catalog；
5. 保留 actual receipt/source lease hard verification；
6. 合并后重跑全套 default/public/test-catalog 回归。

## 17. Release Checklist

### Runtime owner

- [ ] preflight once；
- [ ] catalog snapshot stable；
- [ ] source lease fd-based；
- [ ] receipt v2；
- [ ] drain/quarantine；
- [ ] C ABI ownership/cancel；
- [ ] default audit zero。

### Model owner

- [ ] probe/compile/generate_resolved；
- [ ] private/public shared exact core；
- [ ] source/component/layout identity；
- [ ] output/quality；
- [ ] fault matrix；
- [ ] unsupported routes fail-closed。

### Performance owner

- [ ] P0；
- [ ] P1；
- [ ] P2 target；
- [ ] P3 swap/pressure；
- [ ] P4 strategy report；
- [ ] bootstrap/CI evidence；
- [ ] no unsupported speed claim。

### App owner

- [ ] Off default；
- [ ] five target UI；
- [ ] engine-scoped availability；
- [ ] stale options guard；
- [ ] JobStore v2；
- [ ] LTX handshake；
- [ ] replay/revoke UI。

### Release owner

- [ ] evidence bundle；
- [ ] independent verifier；
- [ ] catalog builder/linter；
- [ ] reviewer digests；
- [ ] staging smoke；
- [ ] rollback/revoke；
- [ ] dev merge regression；
- [ ] release notes准确描述“可控内存”与“swap速度结论”。

## 18. 最终判断

这个框架只有在“默认路径零新增工作、public exact 路径有 source/receipt/lifecycle 证据、每个 target 有完整进程树校准、release note 不夸大 swap 结论”四件事同时成立时，才适合在 App 中 public。

如果某个模型已经能在 private candidate 下跑通，但还缺 source lease、receipt v2、full-request peak 或 review，它的正确状态是：

~~~text
private candidate: executable
public target: unavailable / not reviewed
~~~

这不是失败，而是为了保护低内存机器和内存充足机器的共同稳定性。
