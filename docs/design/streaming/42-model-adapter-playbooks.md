# 42 · 四模型 Adapter 施工手册与档位候选设计

[目录](README.md) · [公共产品化](40-public-productization-and-app-contract.md) · [调度器](41-scheduler-multi-slot-and-multi-pool-implementation.md) · [验证工作簿](39-validation-benchmark-and-release-workbook.md)

修订日期：2026-09-17。状态：**模型接入与校准 playbook；表内候选是实验范围，不是 public 支持声明**。

TurboCider 需要统一管理 LTX 2.5、MiniMax H3 Turbo、Z-Image Turbo 和 Flux.2 Klein 9B，但不能把四个模型压成同一个“block_count + slot_count”模板。统一的是 source/descriptor/layout/authority/receipt/scheduler 合同；模型特有的 block 几何、component 生命周期、pass 语义、source artifact 和 backend 仍由 adapter 声明。

## 1. Adapter 统一合同

每个 public adapter 必须实现 `ModelSession` 的三个 hook：

```cpp
std::shared_ptr<const ModelStreamingProbe>
probe_public_streaming(const PublicResolveInput &) const override;

std::shared_ptr<const ModelStreamingSnapshot>
compile_public_streaming(
    std::shared_ptr<const ModelStreamingProbe>,
    const StreamingPresetRecord &) const override;

RunResult generate_resolved(
    std::shared_ptr<const ResolvedRequestExecution>,
    const Event &, std::atomic<bool> &) override;
```

职责严格分层：

| Hook | 允许做什么 | 禁止做什么 |
|---|---|---|
| `probe` | 读取 manifest/index/header、构造 SourceLease、构造 workload/runtime identity | GPU buffer、MLX array、worker、compiled graph、修改 request |
| `compile` | 用 exact record 生成 Descriptor/Layout/Snapshot，复用同一 lease | 改 P/G/K/D/Q、fallback 到另一个 preset、分配完整模型 |
| `generate_resolved` | revalidate 后创建 pool/worker、执行 exact core、生成 receipt/result | 调普通 `generate`、按 path 重开 source、绕过 authority |

adapter 不可自报 `source_lease_verified=true`。该字段只有在 common coordinator 完成 pre-GPU revalidate，且 adapter 在所有 reader drain 后完成 post-drain revalidate 后才可写入。

## 2. 统一源与组件闭包

### 2.1 两个 source closure

```text
execution lease closure
  transformer/denoiser shard、index、header、reader 实际会读的全部文件

qualification closure
  tokenizer/encoder、connector、VAE、upsampler、vocoder、audio、LoRA/quant cache、export helper
```

第一类必须被 `SourceLease` fd-backed；第二类必须进入 source/workload/component policy identity，并由完整请求采样覆盖。只给 denoiser shard 做 lease 不能签发 public record。

### 2.2 组件策略

每个 adapter 的 `component_policy_revision` 要明确：

```text
resident components
streamed components
release-before-next-stage boundaries
pool retention across VAE/upsample/audio/export
allocator/cache policy
```

改变组件释放顺序、是否保留 VAE、是否启用音频，都会改变 policy revision 和 calibration record；不能在同一 record 上隐式变化。

## 3. 档位候选生成原则

用户档位是 8/10/12/16/20 GiB，adapter 不需要为每个档位写一套代码。候选生成由离线 compiler/campaign 完成：

```text
model descriptor + workload card
    -> candidate family (P/G/K/D/Q/pool)
    -> layout compiler
    -> simulator prefilter
    -> full-request measurement
    -> reviewed record
```

首轮候选只在 adapter 已支持的范围内枚举。所有表格中的 `P/G/K/D/Q` 都是研究 ID；没有 evidence 之前不能显示为 `available`。

## 4. Z-Image Turbo

### 4.1 适用范围（第一张 public card）

建议首张 card 固定：

```text
model_id            = z-image-turbo
operation           = image.generate
source              = Comfy BF16 单文件
execution           = GPU eager Metal
compiled graph      = false
ANE                 = false
LoRA                = false
GGUF/quant cache    = false
image edit          = false
```

本机 descriptor 已知 30 个 dense blocks；仍必须以 source metadata 重新确认 block count、field geometry 和 artifact digest。

### 4.2 建议候选族

保持 `G=1`、single pool、`pass_transition=reload`，先只改变 P/K/D/Q：

```text
P ∈ {0, 4, 8, 12, 14, 18, 22, 26, 28}
G = 1
K ∈ {1, 2}
D ∈ {0, 1} 且 D < K
Q ∈ {1, 2} 且 Q <= K
pool = serial
```

先 coarse 测 `{P=0,8,14,22}`，每个 target 附近补 `P_est-1/P_est/P_est+1`。P28 只能在 suffix 仍至少有 K 个 group 时成立；P29/K2 不应被当作“准 resident”。

### 4.3 组件与 token identity

text encoder token 数直接影响 activation/workspace；必须进入 `PresetWorkload.token_shapes` 和 `feature_digest`。超过 card 的提示词长度：

```text
options -> unavailable
resolve -> unvalidated_workload
```

不能自动截断或复用短文本 record。VAE 释放、LoRA、ConvRot、ANE manifest 和 compiled graph 都必须单独拒绝或建立新 card。

### 4.4 代码落点

```text
native/models/z_image/z_image.hpp/.cpp
  - public probe/snapshot/generate_resolved
  - exact denoise core 与普通 GPU 路径共用

native/models/z_image/streaming_descriptor.hpp
  - 30-block descriptor/model revision

native/platform/apple/z_image_streaming_descriptor.mm
  - metadata -> common Descriptor/SourceLease

native/platform/apple/z_image_weight_stream.mm
  - fd-backed reader，禁止按 path 重开

tests/native/z_image_public_streaming_test.cpp
tests/native/test_z_image_public_streaming.py
  - source mutation、token mismatch、layout/receipt、cancel/drain
```

### 4.5 必测错误

- 30 block 中任一 field 缺失或 source range 越界；
- prompt token shape 超出 reviewed card；
- LoRA/ANE/compiled/GGUF 混入；
- VAE 前 pool 未 drain；
- output 不完整或 `actual_layout_digest` 不匹配；
- source path replacement/same-size mutation。

## 5. Flux.2 Klein 9B

### 5.1 适用范围

第一张 card 固定：

```text
model_id          = flux2-klein-9b
weight_format     = BF16 eager
execution         = GPU Metal
compiled_graph    = false
LoRA              = false
image input       = text-only card first; image-input later
```

Flux 4B compiled graph 不是 9B public adapter 的别名，也不能从 4B 的速度/内存数据推导 9B record。

### 5.2 dual/single 两类 pool

当前 descriptor 识别 `dual_count + single_count`（本机 private 记录为 8 dual + 24 single）；public record 必须把实际计数写入 source/layout identity，不写死到 resolver。

第一阶段保持 P0：

```text
G=1
K=2
D∈{0,1}
Q∈{1,2}
multi_pool_policy=retain_all
pass_transition=reload
```

dual/single 是两个 class barrier：

```text
pool dual:   dual[0..N)
pool single: single[0..M)
```

每个 pool 需独立 source range、slot capacity、fill/fence matrix；class 切换不可把 dual ticket 交给 single pool。

### 5.3 高档 prefix 研究

只有完成 P0/retain_all 的完整请求基线后，才研究 prefix：

```text
P2/P4/P6: 常驻前若干 dual blocks
P8:       dual 全常驻，只 stream single
P12:      dual + 前若干 single 常驻
```

跨 class prefix 会改变 pool 拓扑，必须是新的 layout digest。若某 class 剩余 group 数小于 K，拒绝该候选，而不是自动降 K。

### 5.4 代码落点

```text
native/models/flux2/flux_transformer.cpp
  - dual/single exact bind 与 pass 顺序

native/models/flux2/pipeline.cpp
  - public resolve/generate、VAE boundary、metrics

native/platform/apple/flux_streaming_descriptor.mm
  - source ranges、class identity、pool policy

tests/native/flux_public_streaming_test.cpp
  - dual/single barrier、retain_all peak、receipt、VAE/output parity
```

### 5.5 必测错误

- dual/single field signature 不一致；
- retained pool 总容量超过 target；
- class barrier 前 reader 未 drain；
- dual/single concatenate boundary 改变计算顺序；
- image input/LoRA 误复用 text-only record；
- 4B compiled route 被 9B selector 匹配。

## 6. MiniMax H3 Turbo

### 6.1 冻结范围

本轮只做：

```text
model_id       = minimax-h3-turbo
checkpoint     = original BF16
execution      = native Metal GPU
steps/passes   = Turbo 固定 pass contract
approximation  = false
ANE/VDN/VSA    = false
ordinary H3    = false
```

普通 H3、VDN-H3、FastH3、量化 checkpoint、ANE route 都是不同 adapter/revision，不得因为同一 block kernel 而共享 public record。

### 6.2 carry-first-group

H3 使用 generic executor 的 `carry_first_group` 语义时，layout 必须满足：

```text
single pool
G=1
K=2
pass_count>1
suffix groups >= 2
```

每个 pass 的 carry 坐标由：

```text
slot = (pass * group_count + ordinal) % K
```

计算，receipt 要核对 incoming/outgoing carry 的 pass/step/group/slot。最后一个 pass 必须 `carry_out = none`。

### 6.3 候选族

先测 `P ∈ {0,2,4,8,12,16,24}`，保持 G1/K2/D1/Q1；之后才考虑 K1 或 Q2。P 值必须小于 active block count 且 suffix 至少 K 个 group。由于 H3 的 active block 数由 descriptor/workload 决定，不能在 resolver 内假设一个固定数字。

### 6.4 代码落点

```text
native/platform/apple/h3_session.mm
  - public probe/snapshot/generate_resolved、source closure

native/models/h3_runtime/h3_streaming_descriptor.*
  - active blocks、carry/pass metadata

native/models/h3_runtime/h3_dit.c
  - exact block bind、reader fence、C receipt bridge

tests/native/h3_public_streaming_test.cpp
  - carry rotation、last-pass、ordinary H3 non-regression
```

### 6.5 必测错误

- active block mask 与 descriptor 不一致；
- carry ticket 坐标错位；
- pass boundary 存在两个 Ready/一个 InUse 以上；
- cancel 后仍 enqueue 下一 pass；
- ordinary H3 路径出现 public probe/catalog/receipt 工作；
- VAE/export 不在完整 request receipt/peak 中。

## 7. LTX 2.5

### 7.1 适用范围与 worker

LTX 的 public route 必须单独冻结 source model、weight format、backend 和 workload。当前本机已有 48-block descriptor、两 stage/upsample boundary、request-scoped worker 原型；这些是实现事实，不是 production record。

推荐 public container：

```text
execution_container = ltx_worker
parent App           = serializable JobEnvelope only
worker               = probe + lease + snapshot + exact generate
finalizer            = drain + source revalidate + result handoff
```

父进程不得持有 worker 的 fd、GPU pointer 或 authority。worker SIGTERM/SIGKILL、IPC 断开或 drain 未知必须 quarantine。

### 7.2 候选族

保留 G1，按现有 descriptor 支持范围研究：

```text
P ∈ {1,4,8,12,16,20,24,32,40}
K ∈ {1,2,3}
D < K
Q <= K
pool = serial（首轮）
```

`P8/G1/K3/D2/Q3` 可作为 private anchor replay；`P1/G1/K1/D0/Q1` 作为低内存安全基线。两者都不能直接成为 public target record。

### 7.3 完整组件闭包

单次 campaign 必须覆盖：

```text
Gemma/tokenizer -> connector -> Stage1 denoiser
-> latent upsampler -> Stage2 denoiser
-> video VAE -> audio/vocoder（若 enabled）-> export/finalizer
```

audio、I2V、不同 frame/fps、不同 text rows 和 Sol/attention batch 都要有独立 workload card。不能用 64×64 tiny latent 证据替代 App 的真实视频输出。

### 7.4 代码落点

```text
native/platform/apple/ltx_session.mm
  - worker-local public authority、parent/worker correlation

native/models/ltx_runtime/ltx_streaming_descriptor.*
  - 48 blocks、source closure、stage/pass metadata

native/models/ltx_runtime/ltx_streaming_plan.*
  - record -> exact LTX plan projection

native/models/ltx_runtime/ltx_streaming_adapter.inc
  - generic executor -> ltx_blocks.c exact core

tests/native/ltx_public_streaming_test.cpp
  - worker lifecycle、cancel/quarantine、VAE/audio/export parity
```

### 7.5 必测错误

- parent/worker request digest 不一致；
- worker 使用旧 catalog revision；
- worker source lease 与 snapshot generation 不一致；
- Stage1/upsampler/Stage2 之间 pool 未 drain；
- video/audio/export 失败却返回 succeeded；
- SIGKILL 后 parent 错误地复用 session。

## 8. 统一代码模板

### 8.1 Probe

```cpp
auto lease = SourceLease::capture(source_files); // proposed single fd lineage
PresetSourceIdentity source = make_source_identity(*lease, manifest);
PresetWorkload workload = make_workload_identity(request, component_policy);
PresetRuntimeIdentity runtime = make_runtime_identity(build, adapter, reader, kernel);

return std::make_shared<ValueModelStreamingProbe>(
    ValueModelStreamingProbe::Values{
        model_id, source, workload, runtime, component_policy, lease});
```

### 8.2 Snapshot

```cpp
auto layout = compile_layout(record.plan.canonical_config, descriptor);
require(layout.digest == record.plan.layout_digest,
        "streaming_layout_digest_mismatch");
return std::make_shared<ValueModelStreamingSnapshot>(
    ValueModelStreamingSnapshot::Values{
        model_id, source, runtime, descriptor, layout,
        component_policy, probe->lease_ptr()});
```

### 8.3 Generate

```cpp
auto binding = bind_public_execution<ModelSnapshot>(execution,
                                                    model_id,
                                                    container);
binding.lease->revalidate_paths();
binding.lease->revalidate_open_files();

StageExecutor executor(stage, request_id, adapter);
executor.begin(binding.snapshot->layout().stages.front());
run_exact_core(request, executor, event, cancel);
executor.finish();

binding.lease->revalidate_after_drain();
attach_receipt_v2(result, executor.receipt(), binding.lease->generation());
```

### 8.4 不允许的快捷方式

```cpp
// 错误：普通路径看不到 sealed snapshot/authority
return generate(request, event, cancel);

// 错误：在 adapter 内自选更小 K
if (!fits()) config.slot_count = 1;

// 错误：fill worker 重新按 path 打开
std::ifstream file(group.path);
```

## 9. Adapter 交付阶段

| 阶段 | 模型 | 代码出口 | 必须先通过 |
|---|---|---|---|
| C3 | Z-Image | probe/snapshot/exact generate | source lease、receipt v2、20 matched image runs |
| C4 | Flux 9B | dual/single retained pools | class barrier、retain_all peak、VAE parity |
| C5 | H3 Turbo | carry + C receipt | carry matrix、last-pass、ordinary H3 P0 |
| C6 | LTX | worker-local authority | parent/worker/quarantine、full video closure |

任何 adapter 若只能完成 private candidate，不得把 `release.channel` 写为 public。

## 10. 统一验收矩阵

每个模型至少要有：

```text
host descriptor golden
source mutation/path replacement
probe metadata-only audit
snapshot layout digest replay
K/G/D/Q/pool safety
cancel before/during/after fill
reader fence delay and drain
full output/quality parity
process-tree peak
P0 default non-regression
P1 same-layout generic overhead
P2 target fit
P3 swap comparison（若要宣称更快）
independent verifier
```

只有以上矩阵和 reviewed catalog record 都完成，App options 才可把对应模型/档位标记为 `available`。
