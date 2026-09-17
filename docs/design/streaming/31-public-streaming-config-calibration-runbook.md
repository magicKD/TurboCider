# 31 · Public Streaming 配置、校准与验收操作手册

[目录](README.md) · [详细集成规格](30-public-streaming-detailed-integration.md) · [Runtime/App 工程规格](33-public-runtime-app-engineering-spec.md) · [模型档位与发布](34-model-tier-calibration-and-release-spec.md) · [档位探索计划](24-memory-tier-exploration-and-acceptance.md) · [发布验收](26-public-preset-acceptance-and-release.md)

日期：2026-09-17。状态：**实施前操作规格，尚无 public record**。

本文把 public streaming 的配置、候选探索、完整内存校准、swap 对照、性能统计和发布证据写成可执行 runbook。实现人员可以按
本文件建立工具和 CI；测试人员可以按测试 ID 复核；发布人员可以按 evidence bundle 决定某一个模型/工作负载/目标是否可见。

本文不改变以下既有约束：

- streaming off/absent 继续走当前默认路径；
- 用户只选择 memory tier，不直接编辑 P/G/K/D/Q；
- public v1 只支持 GPU；
- H3 只支持 MiniMax H3 Turbo original BF16；
- production catalog 在所有证据完成前保持为空；
- target 是完整请求目标，不是显存 hard cap、denoiser budget 或零 swap 保证。

## 1. 配置分层和职责

### 1.1 四层配置

配置必须分成四层，不能把设备事实、模型事实和用户偏好混在一个 JSON：

| 层 | 文件/来源 | 谁能修改 | 进入 digest |
|---|---|---|---:|
| Runtime policy | native 编译常量/版本 | runtime maintainer | 是 |
| Device profile | 本机 probe + reviewed profile | 安装/设备适配 | 是 |
| Workload card | 仓库配置、实验提交 | model/perf maintainer | 是 |
| User intent | App v2 request | 用户 | 只作为 requested digest，不授予 authority |

Runtime policy 决定 reader、allocator、slot safety 和最大值；device profile 决定 GPU/OS/container qualification；workload card
决定 shape/token/component 范围；user intent 只包含开关和 target/preset 选择。

### 1.2 推荐目录

~~~text
configs/streaming/
  schema/
    selector-v2.schema.json
    workload-card-v1.schema.json
    device-profile-v1.schema.json
    catalog-record-v1.schema.json
  devices/
    apple-m4-max-gpu-64g.json
    apple-m3-pro-gpu-18g.json
  workloads/
    ltx25-distilled-t2v-512x320x33-s11.json
    zimage-original-t2i-512-s9.json
    h3turbo-original-t2v-512x22-s4.json
    flux9b-klein-t2i-512-s4.json
  candidates/
    proposed/
    rejected/
  catalog/
    proposed/
    reviewed/
    revoked/
~~~

工具输出放在 evidence/ 或 results/，不与可读配置共用目录；模型权重、生成视频、临时采样和本机绝对路径不得提交。

## 2. User intent schema

### 2.1 最小 v2 selector

~~~json
{
  "schema_version": 2,
  "enabled": true,
  "selection": "memory_tier",
  "retention": "request",
  "target_request_memory_bytes": 12884901888
}
~~~

字段语义：

- enabled=false 或 streaming 对象 absent：旧路径；
- selection=memory_tier：只允许五个 target；
- selection=preset：必须同时有 preset_id、preset_revision、catalog_revision 和 target；
- retention=request：请求结束后释放 request-scoped backing；
- expected_resolution_digest：仅用于 stale 检查；
- 不允许 authorized、public、authority_id、manual_layout 等外部权限字段。

### 2.2 请求与 profile 合并

profile 和 request 的 selector 必须采用完整对象覆盖：

~~~text
request.streaming_selector 存在 → 使用 request 对象
否则 profile.streaming_selector 存在 → 使用 profile 对象
否则 → selector absent，走旧路径
~~~

不能逐字段拼接。例如 profile 提供 target=10 GiB、request 只提供 selection=preset 时，必须作为不完整 selector 拒绝，而不是继承
profile 的 target 后尝试选择。

### 2.3 与旧字段的冲突

public selector active 时，下列字段组合必须早拒绝：

- legacy residency=resident/streamed/component_staged；
- memory_budget_bytes；
- memory_constrained.enabled；
- streaming manual schema v1 stages；
- quantized_cache、LoRA、approximation（除非该 record 明确支持，但 v1 默认全拒绝）；
- execution=gpu_ane 或 profile（除非未来有独立 public record）。

冲突在 parse/validate 阶段返回 streaming_config_conflict，不取得 GPU lock，不创建 session execution。

## 3. Device profile

### 3.1 最小结构

~~~json
{
  "schema_version": 1,
  "gpu_name": "Apple M4 Max",
  "device_class": "Apple M4 Max/gpu/68719476736",
  "os_build_family": "macOS-26",
  "physical_memory_bytes": 68719476736,
  "execution_containers": ["embedded_app", "ltx_cli_worker"],
  "runtime": {
    "turbocider_build_id": "from-native-build",
    "runtime_revision": "streaming-runtime-v1",
    "reader_revision": "pread-bounded-v1",
    "kernel_revision": "from-runtime",
    "allocator_policy_revision": "mlx-shared-v1"
  },
  "verified": false,
  "verification_digest": null
}
~~~

physical_memory_bytes 是设备事实，不等于 target。device_class 需稳定到足以区分 backend/OS/allocator 行为；但不要把用户路径、序列号
和 prompt 写进 key。

### 3.2 设备验证

verified device profile 必须同时有：

1. GPU 名称和 family；
2. physical memory probe；
3. OS/build family；
4. runtime/reader/kernel/allocator revision；
5. 对应 execution container 的最小 smoke；
6. profile 内容 digest。

缺 verified profile 时，options 可返回 tentative_without_device_identity；exact resolve 返回 artifact_verification_required 或
unvalidated_device，不猜测兼容性。

## 4. Workload card

### 4.1 字段

~~~json
{
  "schema_version": 1,
  "id": "zimage.original.t2i.512sq.s9.bf16.gpu.app",
  "model": "z-image-turbo",
  "model_variant": "original-bf16",
  "operation": "image.generate",
  "execution": "gpu",
  "execution_container": "embedded_app",
  "width": 512,
  "height": 512,
  "frames": 1,
  "fps": 24,
  "steps": 9,
  "batch": 1,
  "audio": false,
  "dynamic_text": true,
  "approximation": false,
  "token_shapes": [
    {
      "encoder": "qwen3",
      "tokenizer_revision": "installed",
      "template_revision": "zimage-v1",
      "valid_rows": 192,
      "padded_rows": 256,
      "compute_rows": 256
    }
  ],
  "conditioning_revision": "zimage-conditioning-v1",
  "vae_policy_revision": "zimage-vae-release-v1",
  "feature_digest": "sha256-of-empty-lora-and-input-policy",
  "quality_policy": {
    "seed_set": [17, 42, 99],
    "image_metric": "byte_exact_or_declared_pixel_tolerance",
    "max_failed_requests": 0
  }
}
~~~

raw prompt 不进入 card。token shape、template、conditioning、VAE release 和 feature digest 必须进入 card，因为它们会改变 activation、
component lifetime 或 kernel route。

### 4.2 首批四模型 cards

| 模型 | card | 首轮排除项 |
|---|---|---|
| LTX 2.5 | ltx25.distilled.t2v.512x320x33.s11.dense.gpu.worker | audio/I2V/LoRA/Sol/ANE |
| Z-Image Turbo | zimage.original.t2i.512sq.s9.bf16.gpu.app | GGUF/LoRA/ANE |
| H3 Turbo | h3turbo.original.t2v.512sqx22.s4.bf16.gpu.app | 普通 H3/MLX/quant/audio 变体 |
| Flux 9B | flux2.klein9b.t2i.512sq.s4.bf16.eager.gpu.app | 4B compiled/LoRA/ANE |

1024²、更多 frame、不同 token bucket、I2V/audio 和 worker/app container 都必须建立新 card。

## 5. Model candidate 配置

### 5.1 通用结构

~~~json
{
  "candidate_id": "zimage.original.t2i.512sq.s9.bf16.gpu.app.p14-g1-k2-d0-q1",
  "workload_card": "zimage.original.t2i.512sq.s9.bf16.gpu.app",
  "stages": {
    "transformer": {
      "resident_prefix_blocks": 14,
      "block_group_size": 1,
      "slot_count": 2,
      "prefetch_distance": 0,
      "io_workers": 1,
      "pass_transition": "reload",
      "multi_pool_policy": "serial"
    }
  },
  "limits": {
    "max_target_bytes": 21474836480,
    "max_logical_read_bytes": 0,
    "max_wall_seconds": 0
  },
  "status": "proposed"
}
~~~

candidate 配置只描述 layout 和 policy；实际 calibrated bytes、performance rank 和 review digest 由工具生成，不能手写填充。

### 5.2 合法性约束

编译器必须在运行前拒绝：

- block/group/pool 顺序不稳定；
- suffix blocks < slot_count；
- group bytes > slot capacity；
- prefetch_distance >= slot_count；
- io_workers=0 或大于 slot_count；
- adapter 不支持的 pass transition/multi-pool policy；
- materialization、source range、dtype 或 artifact identity 不完整；
- prefix 等于或超过 block count；
- P/G/K/D/Q 改变但 candidate_id 未改变。

## 6. 四模型候选探索矩阵

下表是探索顺序，不是已支持清单；每个 target 的 available 必须由实测 evidence 产生。

### 6.1 LTX

| target | 首选 P | K | G | D | Q | 备注 |
|---:|---:|---:|---:|---:|---:|---|
| 8 GiB | 1,4 | 2 | 1 | 1 | 1 | 先确认 component floor |
| 10 GiB | 4,8 | 2 | 1 | 1,2 | 1,2 | 低读取优先 |
| 12 GiB | 8,12 | 2,3 | 1 | 1,2 | 1,2 | 现有 P8/K3 仅为 anchor |
| 16 GiB | 12,16 | 3 | 1 | 2 | 2,3 | 检查 pool peak |
| 20 GiB | 16 | 3 | 1 | 2 | 3 | 以 wall/P95 选优 |

48 blocks，suffix>=K；audio/I2V/97 frames 不复用首卡 record。

### 6.2 Z-Image

| target | P 候选 | K/G/D/Q | 备注 |
|---:|---|---|---|
| 8 GiB | 0,4,8 | 2/1/0/1 | 低 target 先测 VAE/activation floor |
| 10 GiB | 8,12 | 2/1/0/1 | 与 legacy 10 GiB budget 不等价 |
| 12 GiB | 12,14,18 | 2/1/0/1 | P14 仅为 private anchor |
| 16 GiB | 18,22,26 | 2/1/0/1 | 检查 prefix release |
| 20 GiB | 26,28 | 2/1/0/1 | P29 对 30 blocks/K2 非法 |

512² 与 1024²独立 card；GGUF、LoRA、ANE 另行拒绝。

### 6.3 H3 Turbo

| target | P 候选 | K/G/D/Q | 备注 |
|---:|---|---|---|
| 8 GiB | 0,2 | 2/1/1/1 | 检查四 pass activation floor |
| 10 GiB | 2,4,8 | 2/1/1/1 | carry-first-group |
| 12 GiB | 8,12 | 2/1/1/1 | original BF16 only |
| 16 GiB | 12,16 | 2/1/1/1 | 大矩阵单次 pread |
| 20 GiB | 16,24 | 2/1/1/1 | 22/39/73 frame 独立 |

H3 不能因 private 工程波动合理就扩大到普通 H3 或量化变体。

### 6.4 Flux 9B

| target | P 候选 | K/G/D/Q | pool | 备注 |
|---:|---|---|---|---|
| 8 GiB | 0 | 2/1/1/2 | retain_all 或拒绝 | 先测 fixed/activation floor |
| 10 GiB | 0 | 2/1/1/2 | retain_all | P0 首卡 |
| 12 GiB | 0,2,4 | 2/1/1/2 | retain_all | prefix 第二阶段 |
| 16 GiB | 4,6,8 | 2/1/1/2 | retain_all | P8 消除空 dual suffix |
| 20 GiB | 8,12 | 2/1/1/2 | retain_all | P7 非法，P12 独立验证 |

dual block 0–7、single block 8–31；block 8 concatenate 每 pass 恰一次。

## 7. 完整内存校准协议

### 7.1 采样窗口

一个 candidate 的校准窗口从 worker/engine construct 前开始，覆盖：

~~~text
process launch
  → clean baseline
  → engine/model construct
  → metadata resolve
  → load/prepare
  → tokenize/conditioning
  → denoise/transformer all passes
  → VAE/upsample/audio/export
  → output fsync/close
  → GPU fence and reader drain
  → unload/worker exit
~~~

不能只在 denoiser 阶段采样，也不能在 output 写完后立即结束窗口而遗漏异步 GPU/reader release。

### 7.2 采样记录格式

~~~json
{
  "schema_version": 1,
  "timestamp_monotonic_ns": 123456789,
  "wall_timestamp": "2026-09-17T00:00:00Z",
  "pid": 1234,
  "start_identity": "pid-start-token",
  "role": "embedded_engine",
  "parent_pid": 1000,
  "phase": "denoise",
  "event_sequence": 42,
  "physical_footprint_bytes": 9876543210,
  "resident_bytes": 8765432100,
  "compressed_bytes": 0,
  "mlx_active_bytes": 7654321000,
  "mlx_peak_bytes": 8000000000,
  "metal_accounted_bytes": 0,
  "swap_in_bytes": 0,
  "swap_out_bytes": 0,
  "pressure_state": "normal",
  "thermal_state": "nominal",
  "sample_gap_ns": 50000000,
  "sampler_epoch": 1
}
~~~

PID 不能单独作为身份；必须与 start_identity 配对。短命 child 由 launch/exit hook 记录，不能依赖固定周期恰好采到。

### 7.3 覆盖要求

校准 verifier 必须确认：

- construct/load/prepare/denoise/VAE/export/cleanup 每阶段至少有一条样本；
- 每个发现的 process role 都有 launch 和 exit；
- 最大 sample gap 小于 policy 上限；
- sampler epoch 没有重置丢样；
- 正常结束时 active reader、pending fill、worker、ticket 和 output file handle 为零；
- source artifact 在请求期间未改变；
- cold/warm/transition 样本没有混写同一统计分布。

任一项失败，calibration.complete=false，builder 不得生成 reviewed record。

### 7.4 计算 target eligibility

~~~text
M_confirmed = max(complete fresh/warm/transition tree peaks)
M_catalog = max(M_confirmed, independent upper-bound estimate)
H(T) = max(512 MiB, ceil(0.10 × T))
eligible(T) = calibration.complete && M_catalog + H(T) <= T
~~~

边界测试必须覆盖 T-1、T、T+1 byte 的等价整数运算；不能用浮点比较或四舍五入。

## 8. 模拟器和候选剪枝

### 8.1 输入

模拟器读取：

- compiled layout groups/pools/slot capacities；
- 每组 source bytes 和 measured read-time distribution；
- pass transition、component live interval；
- kernel/activation time distribution；
- reader fence latency；
- device SSD/page-cache class。

### 8.2 输出

- predicted process backing peak；
- predicted logical read bytes；
- predicted wall/denoise median/P95；
- slot occupancy、I/O utilization、GPU idle；
- refill wait、prefetch hit/miss、barrier count；
- target eligibility sensitivity。

模拟结果只能用于剪枝和排序。held-out candidate 的 wall 中位误差超过 10% 或 P95 误差超过 20% 时，模拟器只能作为定性参考。

### 8.3 CLI 约定

~~~sh
python3 tools/native/inspect_streaming_candidates.py \
  --model z-image-turbo \
  --workload configs/streaming/workloads/zimage-original-t2i-512-s9.json \
  --targets 8,10,12,16,20 \
  --output evidence/zimage/candidates.json

python3 tools/native/simulate_streaming_schedule.py \
  --candidate evidence/zimage/candidates.json \
  --read-distribution evidence/zimage/read-times.json \
  --output evidence/zimage/simulation.json
~~~

命令默认只做 host/metadata 工作；未显式传 --run-real 不得启动 GPU。

## 9. Resident、streaming 和 swap 对照

### 9.1 对照矩阵

每个 workload card 至少运行：

| 路线 | 目的 |
|---|---|
| resident | quality/wall/peak 基线 |
| legacy streamed（若存在） | 兼容性基线 |
| public exact streaming | slot/pool 策略收益 |
| natural low-memory | 真实压力下稳定性 |
| controlled pressure | 与系统 paging 机制的可证伪比较 |

五条路线必须固定 model artifact、prompt set、seed、shape、steps、output、container 和 runtime build。

### 9.2 需要报告的指标

~~~text
success_rate
wall_median / wall_p95
denoise_median / denoise_p95
process_tree_peak
mlx_peak / metal_accounted_peak
swap_in / swap_out delta
major_fault / minor_fault delta
logical_read_bytes / physical_io_bytes（若可得）
gpu_busy / cpu_busy / SSD throughput
thermal / pressure state
quality metric and output equality
cancel/failure/quarantine counts
~~~

streaming 无 swap 但变慢时，应报告“峰值下降、速度代价”；只有 speedup 95% 区间下界大于 1 才报告“更快”。

### 9.3 统计设计

采用 ABBA 或随机区组顺序，避免先跑路线固定造成温度/page cache 偏差。cold/warm 分层，不把 warm cache 当作 fresh。

建议首轮：

- smoke：每路线 1 次；
- memory exploration：fresh 3 次、warm 10 次；
- performance confirmation：ABBA 每臂至少 10 个有效样本；
- quality：至少 3 个固定 seed；
- fault：每个故障点至少 1 次成功注入和 1 次恢复/重建。

如果失败导致有效样本不足，结果为 INCONCLUSIVE，不得删除失败样本再计算。

## 10. 验收矩阵

### 10.1 配置和 resolver

| ID | 验收 | 通过标准 |
|---|---|---|
| CFG-001 | off/absent | 解析成功，public audit=0 |
| CFG-002 | target 五档 | 仅 8/10/12/16/20 GiB 通过 |
| CFG-003 | decimal lexical | exponent/decimal/negative/overflow 拒绝 |
| CFG-004 | profile/request | 完整对象覆盖，不逐字段拼接 |
| CFG-005 | conflict | legacy residency/budget/manual 与 public 冲突 |
| CFG-006 | preset exact | id/revision/catalog/target 缺一拒绝 |
| CFG-007 | unknown fields | stable schema error |
| RES-001 | empty catalog | catalog_has_no_public_records |
| RES-002 | deterministic rank | 乱序输入选择相同 record |
| RES-003 | source mismatch | artifact_verification_required/artifact_changed |
| RES-004 | device mismatch | unvalidated_device |
| RES-005 | stale digest | streaming_resolution_stale |
| RES-006 | revoked | preset_revoked 或 preset_not_public |
| RES-007 | layout mismatch | streaming_actual_plan_mismatch |

### 10.2 C ABI/App

| ID | 验收 | 通过标准 |
|---|---|---|
| API-001 | resolve null/ownership | 无崩溃，out/error 可独立释放 |
| API-002 | resolve lock | 不持 global GPU lock/DeviceLease |
| API-003 | active prepare | streaming_prepare_unsupported |
| API-004 | active generate empty catalog | GPU/session 未调用 |
| API-005 | exact generate | 只调用 generate_resolved |
| APP-001 | options debounce | 旧 revision 不覆盖新结果 |
| APP-002 | off | 不发 options/resolve |
| APP-003 | persist failure | generate 次数为 0 |
| APP-004 | old job | 无损 decode/replay |
| APP-005 | unknown job schema | 不覆盖原文件 |
| APP-006 | LTX worker | resolve 在 worker container 内完成 |

### 10.3 Runtime/lifecycle

| ID | 注入 | 通过标准 |
|---|---|---|
| RT-001 | pool allocation | 失败安全清理或 quarantine |
| RT-002 | first fill | reader/ticket/source lease 无泄漏 |
| RT-003 | mid-fill read | 无 slot overwrite |
| RT-004 | fence timeout | 不提前释放 backing |
| RT-005 | mailbox overflow | stable error + quarantine |
| RT-006 | cancel before dispatch | 无 GPU group submit |
| RT-007 | cancel during prefetch | worker join/pending=0 |
| RT-008 | VAE/export failure | stage policy 正确释放 |
| RT-009 | unsafe drain | engine/process quarantine |
| RT-010 | mode transition | resident/public/off 无残留 |

### 10.4 性能和内存

| ID | 验收 | 通过标准 |
|---|---|---|
| PERF-001 | default P0 | median≤1.02，P95≤1.05，区间不越线 |
| PERF-002 | same-layout P1 | generic/direct 在固定布局下通过 |
| PERF-003 | target eligibility | M_catalog+H(T)≤T |
| PERF-004 | coverage | 全阶段/全进程 role 覆盖 |
| PERF-005 | no swap claim | swap delta 和 pressure evidence 完整 |
| PERF-006 | speed claim | wall speedup 95% CI 下界>1 |
| PERF-007 | quality | fixed seeds 成功率和质量不劣 |
| PERF-008 | audit | off/default 无新 pool/thread/sampler/cache-clear |

## 11. Evidence bundle

每个 candidate 必须生成不可变 bundle：

~~~text
policy.json
device-profile.json
workload-card.json
source-identity.json
build-identity.json
candidate.json
compiled-layout.json
rejections.jsonl
simulation.json
raw-events.jsonl
memory-samples.jsonl
memory-summary.json
quality.json
performance.json
swap.json
lifecycle.json
audit.json
verifier-report.json
catalog-proposal.json
review.json
~~~

bundle manifest 记录每个文件的 SHA-256、生成工具版本、git commit、环境摘要和时间窗口。reviewer 应能从 bundle 重算：

1. candidate canonical layout/digest；
2. M_confirmed、M_catalog 和 H(T)；
3. P0/P1/P4 统计；
4. swap/fault/pressure 变化；
5. quality/failure/lifecycle；
6. record release/revoke 状态。

缺文件、字段缺失、失败样本被删除、digest 不一致或工具版本未知，均标记 incomplete。

## 12. Builder、review 和撤回

### 12.1 Builder 状态机

~~~text
proposed
  → calibrated
  → independently_verified
  → reviewed
  → public-experimental
  → public-stable
  ↘ revoked
~~~

builder 默认只生成 proposed。public-experimental 需要 reviewer digest；public-stable 还需要至少一个 staged replay 和 revoke drill。

### 12.2 Record-only 发布

record PR 只能修改 catalog/release/evidence 索引，不修改 executor/kernel/default App。这样可以独立撤回单个模型或 target。

### 12.3 撤回触发

- source/runtime/reader/kernel identity 变化；
- 新样本发现 peak/coverage 不满足 target；
- success/quality/performance 回归；
- unsafe drain、crash、quarantine；
- App/worker container 不一致；
- catalog 或 authority digest 无法复算。

撤回只需递增 catalog revision、标记 revoked、刷新 options；历史 exact replay 必须失败，不自动换 record。

## 13. 运行命令模板

### 13.1 Host/contract

~~~sh
python3 -B tests/native/test_streaming_contract.py
python3 -B tests/native/test_streaming_preset_resolver.py
make test-streaming-host
make test-streaming-public-host
~~~

### 13.2 Native build

~~~sh
TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
tools/native/build.sh
~~~

### 13.3 Calibration

~~~sh
python3 tools/native/explore_streaming_presets.py \
  --model z-image-turbo \
  --workload configs/streaming/workloads/zimage-original-t2i-512-s9.json \
  --targets 8,10,12,16,20 \
  --policy configs/streaming/policy/exploration.json \
  --evidence evidence/zimage

python3 tools/native/verify_streaming_calibration.py \
  --bundle evidence/zimage/<candidate-id> \
  --require-full-request \
  --output evidence/zimage/<candidate-id>/verifier-report.json
~~~

### 13.4 Catalog proposal

~~~sh
python3 tools/native/build_streaming_catalog.py \
  --bundle evidence/zimage/<candidate-id> \
  --channel proposed \
  --output evidence/zimage/catalog-proposal.json
~~~

命令不应默认访问生产 catalog，不应修改 native binary；所有真实 GPU/pressure 操作必须由显式 policy 允许。

## 14. 当前实施状态和停止条件

当前可继续施工的顺序：

1. 收口 R1/R2 canonical/resolver/authority worktree；
2. 完成 C ABI exact resolve 和 active generate 分支；
3. 完成 Swift/App off-by-default transaction；
4. 完成 calibration schema/sampler/verifier；
5. 先对 Z-Image 或 LTX 建立一个 full-request candidate；
6. 再接 H3 Turbo、Flux 9B；
7. 合并 dev，重跑 GPU/ANE default；
8. 逐 card/target reviewed release。

必须立即停止并回滚当前 PR 的情况：

- off/default 出现可测热路径回归；
- public selector 能静默回 resident/manual；
- authority 可由 JSON 或 test symbol 伪造；
- source/record/revoke mismatch 没有稳定错误；
- calibration 未覆盖 full request 却要生成 record；
- 低 target 通过改变 dtype/shape/steps 伪造；
- swap 结论没有同设备、同 card、同压力协议；
- failure/cancel 后 reader、worker、ticket 或 process 未归零且未 quarantine。

## 15. 最终验收签字单

### Runtime owner

- [ ] off/default 路径静态和动态审计通过；
- [ ] exact resolve/generate_resolved/authority 完成；
- [ ] lock order、TOCTOU、drain/quarantine 完成；
- [ ] generic executor 无第二套实现。

### Model owner

- [ ] probe/compile/generate_resolved 三个 hook；
- [ ] full pipeline、actual layout、component policy；
- [ ] 模型限定和 unsupported reason；
- [ ] model-specific lifecycle/fault。

### Performance owner

- [ ] candidate sweep 和 simulator；
- [ ] fresh/warm/transition full-request memory；
- [ ] resident/streaming/swap 对照；
- [ ] P0/P1/P4 和 quality 统计；
- [ ] evidence 可复算。

### App owner

- [ ] advanced setting 默认 off；
- [ ] 只显示 native target；
- [ ] options revision/debounce；
- [ ] resolve→persist→generate transaction；
- [ ] old job 无损、错误码稳定、LTX worker container 正确。

### Release owner

- [ ] production record 仅来自 reviewed bundle；
- [ ] catalog revision/revoke/replay drill；
- [ ] dev merge 后 GPU/ANE 回归；
- [ ] release note 不夸大 hard cap、零 swap 或速度收益。

只有所有签字单完成，且某一条具体 model/workload/target record 具有独立 digest 和复算证据，才允许将该条目从 proposed 提升到
public-experimental 或 public-stable。其余模型和档位继续显示 unavailable，不影响默认模式。
