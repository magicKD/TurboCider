# 23 · App 内存档位与 Public Streaming Preset 设计

[目录](README.md) · [当前框架](22-current-framework-guide.md) · [分模型探索](24-memory-tier-exploration-and-acceptance.md) · [代码规格](25-public-preset-implementation-spec.md) · [验收与发布](26-public-preset-acceptance-and-release.md)

日期：2026-09-17。源码基线：`7308db3`，其 runtime 来自 `052265f`。
状态：**待实施的产品/API/架构规格，不是现有 public 功能，也不是新性能成绩。**
本轮只编写方案，没有放开 public gate、改变默认路径、进行 GPU sweep 或启动系统 memory pressure。

## 1. 决策摘要

建议实现。App 不暴露 P/G/K/D/Q，而是在高级设置增加一个默认关闭的“Streaming 流式加载”开关。
用户开启后选择“生成内存目标”：8、10、12、16、20 GiB 等。后台根据模型、权重格式、工作负载和设备，
从已审核 catalog 中选择符合目标的最快已验证 preset，再交给现有 layout compiler 和 StageExecutor 执行。

这不是运行时从预算任意生成一个未知布局，而是：

```text
用户选择内存目标
  → 过滤已验证 preset
  → 按适用工作负载的实测速度排序
  → 冻结一个 exact layout
  → 原有 compiler / executor / adapter 执行
```

必须同时成立：

1. 默认关闭时，现有 resident、component-staged、legacy-streamed 行为与配置原样保留。
2. 用户不需要理解槽位；调试报告仍能追溯实际 P/G/K/D/Q、pool、prefix 和 preset revision。
3. 档位是**整请求实测校准后的内存目标**，首版不是硬上限，也不是设备总内存规格。
4. 所有模型都可以使用同一档位 UI，但不强迫每个模型都有 8/10/12/16/20 五个可用档。
5. 相同目标可以在不同模型、shape 下映射到不同 preset；多个档位也可以映射到同一 preset。
6. 不存在符合条件的 preset 时明确提示“不支持此档位/工作负载”，不偷偷改变精度、尺寸、步数或 slot 数。
7. Public layout-only 和 bounded-memory certification 分开推进；无须等 hard cap 或“快于 swap”全部实现后才提供前者。

## 2. 阅读地图与规范边界

| 问题 | 本文位置 / 其他文档 |
|---|---|
| 用户看见什么、如何与现有 App 共存 | 第 4–5 节 |
| 8/10/12/16/20 如何定义和匹配 | 第 3、6–7 节 |
| catalog、request、resolver、资格对象 | 第 8–11 节 |
| App / Swift / native / CLI 改哪些文件 | 第 12 节 |
| 取消、缓存、模型切换、默认性能保护 | 第 13–15 节 |
| 每个模型具体探索哪些 P/K/D/Q | [24 第 3–7 节](24-memory-tier-exploration-and-acceptance.md) |
| 采集 MLX/Metal/进程峰值、如何测档位 | [24 第 8–11 节](24-memory-tier-exploration-and-acceptance.md) |
| 实施顺序、测试矩阵、发布 checklist | [24 第 12–15 节](24-memory-tier-exploration-and-acceptance.md) |
| 类型/API/所有权、profile 合并、public gate、Swift job 迁移 | [25 第 2–6 节](25-public-preset-implementation-spec.md) |
| 四模型逐接缝施工、Flux prefix/pool 边界算法 | [25 第 7–8 节](25-public-preset-implementation-spec.md) |
| 采样数据合同、资源限额、故障反例与逐 PR 完成门 | [26](26-public-preset-acceptance-and-release.md) |

本方案新增的是 opt-in preset selection 控制面；不创建另一套 slot 调度器。Manual v1 的精确布局合同仍以
[02](02-configuration.md) 为准，内存 guard 以 [05](05-memory-contract.md) 为准，P0–P4 定义与阈值以
[12](12-acceptance-playbook.md) 为准。下文新字段和 API 名称均为**拟议接口**。
本文负责产品语义，24负责候选/实验路线，25负责实施接口，26负责可执行验收；实现中的类型/调用顺序以25的细化为准，
不能只摘本文概念伪代码就把 engine/artifact resolver 放进没有模型 root 的 `make_plan`。

## 3. “内存档位”到底表示什么

### 3.1 不叫“显存格式”

Streaming 是执行/驻留策略，不是模型格式。首版是 Apple GPU；App 应显示“生成内存目标（统一内存）”，
不把 16 GiB 机器解释成“可以把 16 GiB 全交给 GPU”。系统、App、CPU tensor、GPU-visible backing 共享物理资源。

统一用二进制 GiB：1 GiB = 1,073,741,824 bytes。用户口语中的 8GB/10GB 在 UI 明示为 8/10 GiB；协议传整数 bytes。

| UI 档位 | `target_request_memory_bytes` |
|---|---:|
| 8 GiB | 8589934592 |
| 10 GiB | 10737418240 |
| 12 GiB | 12884901888 |
| 16 GiB | 17179869184 |
| 20 GiB | 21474836480 |

后续可按证据增加 24/32 GiB，不能为了整齐填满无证据的档位。

### 3.2 三个数不能混成一个

| 数量 | 用途 | 不是 |
|---|---|---|
| physical memory / recommended GPU working set | 硬件筛选、提示 | 可独占的内存配额、瞬时空闲内存 |
| 用户目标 T | 在 catalog 中筛选 preset 的目标 | OS 强制内存分区、硬 cap |
| preset 的 measured/calibrated request envelope | 判断预期是否适合 T | 完整资源形式化上界、所有 prompt 都保证的峰值 |

`MLX peak` 是重要诊断数据，但不能单独定义档位。LTX/H3 的原生 Metal backing 不应因 MLX 指标缺失而视作零。
MLX active/peak、Metal allocated、process footprint/RSS 存在包含或重叠关系，不能求和冒充“总显存”。

### 3.3 首版内存目标的观测域

定义 `scope=execution_process_tree_v1`：执行请求的进程及其生成相关 helper，从构造/prepare 到安全 cleanup，
按同一时间轴观测。内嵌 App 执行时包括该 App 进程基线；独立 worker 时包括 worker/helper，App shell 单列为系统余量。
记录 `execution_container=embedded_app|cli_worker|service_worker`，这三种部署不能未经校准共用一个基线数值。

跨进程保守合计要标注 `conservative_no_shared_dedup`；不是唯一物理字节的精确和。同进程内 MLX/Metal 字节不再重复加。
无法采到 helper 或关键阶段时，标 `coverage=incomplete`，不可据此发布一个“预计 8 GiB”的档位。

### 3.4 留余量而不冒充 hard cap

对已通过质量/lifetime 的 preset，离线生成工作负载专属 `calibrated_request_bytes=M`。首轮候选筛选规则：

```text
H(T) = max(512 MiB, ceil(0.10 * T))
fits_target = M + H(T) <= T
```

10%/512 MiB 是**拟议离线分档初值**，需在 exploration policy 冻结；不是现有 `memory_constrained.buffer_percent`，
不自动启用 guard，不在 runtime 中改变用户 layout。M 取完整样本集合与阶段估计的保守值，来源见文档 24。
如果存在未知的大额 allocation，仅增加 H 不能使它自动合格。

系统空闲余量另做一次显式开启后的 preflight。必须比较 `expected_additional_bytes` 与可靠的 available-memory 观察，
避免把已在系统占用中的 App baseline 再扣一次。系统余量、H(T)、worker 基线各有 scope，不能重复扣减。
仅“当前没空闲”时返回可重试错误，不趁机改变 preset；默认模式不新增此类探测。

## 4. App 交互

### 4.1 高级设置的最小呈现

```text
高级设置
  [ ] Streaming 流式加载                    默认关闭

  开启后：
    生成内存目标    [ 12 GiB ▼ ]
    当前模型：Z-Image Turbo · Comfy BF16
    当前任务：512 × 512 · 9 steps · GPU
    预计峰值：<经验证区间>    预计速度：<本设备等级实测范围>
    状态：实验 / 已验证；权重会按需读取，速度受磁盘影响
    说明：这是内存目标，不是硬上限，不保证零 swap

  [详情]  preset ID、证据日期、适用 shape、实际布局（只读）
```

这里的区间是占位说明，不是新增实测数据。UI 不展示可编辑 slot 数或 block 数，也不列几十个内部 candidate。
支持的目标由 native query 返回；未认证档位隐藏或置灰并说明原因。勾选本身不触发权重加载、benchmark 或压力测试。

### 4.2 默认值和持久化

- 老 draft 缺失新字段 → `streamingChoice.mode=off`，旧请求编码结果不变。
- 按 model ID 保存用户档位偏好；checkpoint、backend、workload 变化后重新验证适用性。
- 用户首次开启时可预选后端建议的可用档，但显示目标与成本；不得单凭总 RAM 直接运行。
- 用户关闭时恢复其之前的 legacy residency/budget，而不是一律改成 resident。
- 不跨模型携带 resolved layout；可以记住同一个目标 T，但必须重新 resolve。
- 排队任务保存提交时的意图和冻结结果，不随着 UI 当前选项变化。

### 4.3 分辨率、帧数和 prompt 变化

同一模型不能只有一张永久“12 GiB → P14”表。选择键包括宽高、帧数、steps、token 数、operation、精度、组件策略。
UI 编辑时做便宜的 bucket 查询；实际 tokenization 后再做执行前精确匹配。两个查询结果不一致时显示更新，
不得先加载大模型再发现没有支持的档位。

UI 的异步 query 带 `draft_revision`；旧 query 返回不得覆盖新模型/新尺寸状态。运行中/提交中不可改当前任务档位。

### 4.4 不支持时的文案

区分以下原因，不能都显示“显存不足”：

- `unsupported_model_or_format`：例如 Flux 4B 或 Z-Image GGUF 尚未接此 preset。
- `unvalidated_workload`：此分辨率/长 prompt 暂无记录。
- `no_preset_fits_target`：可用 preset 的校准峰值超过所选目标。
- `component_floor_exceeds_target`：最低占用来自 text/VAE，减少 denoiser prefix 也无效。
- `device_or_runtime_mismatch`：此设备/backend 版本尚未覆盖。
- `temporary_memory_pressure`：当前其他任务占用导致暂不适合运行。
- `preset_revoked` / `artifact_changed`：已撤回或模型文件发生变化。

可以建议用户选择更高目标、恢复默认模式，或显式改变尺寸；建议不是自动执行。

## 5. 与当前 App 的兼容

### 5.1 已有 Z-Image 流式选项

当前 `apps/macos/App.swift` 已有 Z-Image “常驻/流式加载（实验）”和 `[6,8,10,12] GiB` 采样预算；
`StudioState.swift` 写入 legacy `residency=streamed` 与 `memory_budget_bytes`。
这控制的是旧版采样规划，不等价于本方案的整请求目标，**不得直接迁移数值含义**。

推荐 UI 共存策略：

1. 新开关关闭：完整保留原控件与旧配置行为。
2. 新开关开启：旧 residency/budget 控件只读/暂时收起，值保存在 draft；新请求不序列化旧字段。
3. 老保存任务照旧回放，不自动转换成 public preset。
4. 如果用户明确选择“迁移到新版”，重新展示峰值口径和目标，并重新 resolve。

### 5.2 当前 Swift 请求还是 v1

`bindings/swift/TurboCiderNative.swift::NativeRequest` 当前是 schema v1，`execution` 为字符串。
不能直接把嵌套 streaming 对象塞进这个字符串，也不能全局改编码而影响默认 App 路径。

拟新增 type-safe `NativeRequestV2` 和请求封装枚举：

```text
streaming off  → 现有 NativeRequest/v1 serializer
streaming on   → NativeRequestV2/嵌套 execution serializer
```

`NativeEngine` 继续使用 public constructor；新增 v2 payload overload 进入同一个 public generate C API。
App 不链接 candidate constructor，不设置 `TURBOCIDER_H3_EXACT_STREAM`，也不把测试用 authority 带到用户请求。

### 5.3 组件释放与 worker 路由不能被 UI 暗改

现有 `LTXWorker.accepts()` 依据 legacy `component_staged` 决定是否使用 CLI。新 streaming 请求不能靠伪造这个字段
触发 worker。若需要独立 worker，这是 preset 的显式 `execution_container`/component plan，须测量进程切换、
latent IPC、VAE/export 与取消成本。App 进程不可执行 CLI-only finalizer 的 `exec` 逻辑。

## 6. 档位与 preset 是多对多

用户看到目标 T；catalog 维护有限个 execution preset，不为每个整数 GiB生成一份模型配置。

```text
T = 8 / 10 / 12 / 16 / 20 GiB
       |
filter(model, artifact, operation, workload, device, channel)
       |
filter(measured envelope + margin <= T)
       |
choose best validated performance rank
       |
exact preset + component policy + request retention
```

没有必要让 16 GiB 档一定比 12 GiB 档占更多内存。若增加 prefix 没有收益，两个档位共用一个 preset 并在详情中说明。
显式开启 streaming 后不静默改成 resident；若 resident 在高内存下更快，只建议关闭 streaming。

### 6.1 推荐与资格分离

- `ExecutionCapability`：语义、安全、reader/lifecycle、格式、workload 的支持资格。
- `MemoryCalibration`：特定设备/部署/bucket 的实测内存目标适配。
- `PerformanceProfile`：同设备和磁盘等级的排序参考。

换 SSD 可能使排序变化，不应把原本正确的 layout 变成不安全，也不应沿用旧 SSD 的速度承诺。
未覆盖的硬件可以显示“缺少档位校准”；首版 public 不自动泛化到所有 Apple Silicon。

### 6.2 Deterministic resolver

相同 catalog revision、artifact、workload、hardware class、目标 T 应选相同 preset。
不根据瞬时 free RAM 在每个 block、pass 或 request 中来回改变 P/K；内存压力只决定准入/拒绝。
后台没有运行时 benchmark，不做 bandit 探索，不用本次用户运行自签新资格。

相邻档位的可行集合应单调：T 增大不能丢掉原先可行候选。排序用离线验证 rank；性能接近时按
“较低内存 → 较少读取字节 → 较小稳定 preset ID”确定 tie-break，避免噪声推动升级。

## 7. 硬件建议与 workload 选择

不硬编码“16 GiB 机器必选 12 GiB”，因为还受操作系统/App基线、helper、其他任务和 GPU working set 影响。
仅当用户开启 streaming 后，执行以下建议步骤：

1. 读取已有 system info 的 physical memory、recommended working set、GPU/OS/runtime identity。
2. 根据部署方式预留系统/App shell 余量；需要实际校准，不沿用单一固定比例适配所有机器。
3. 过滤 GPU 预估工作集不适合该设备、或整请求目标明显超过可用物理范围的选项。
4. 从剩余已验证档位中给出一个“建议”，用户可手选。
5. 点击生成后再次核验瞬时条件；不改变已选择的 layout。

只允许向已审核的 workload 记录匹配。首版记录 exact shapes；未来 bucket/range 覆盖需要单独验证，
不能仅因“最大 shape 跑过”就把所有小 shape/长短 prompt 都认为安全。

## 8. 拟议 API 与配置

### 8.1 新 selector 使用 streaming schema v2

现有 streaming schema v1/manual 保持不变。拟新增 v2 selector；外层 request 仍为 v2，两者版本独立。
下例是设计稿，**当前 parser 会拒绝这些新字段**：

```json
{
  "schema_version": 2,
  "model": "z-image-turbo",
  "operation": "image.generate",
  "inputs": [{"kind": "text", "role": "prompt", "text": "A red fox in snow"}],
  "outputs": [{"kind": "image", "path": "output.png", "width": 512, "height": 512}],
  "sampling": {"seed": 42, "steps": 9},
  "execution": {
    "policy": "gpu",
    "streaming": {
      "schema_version": 2,
      "enabled": true,
      "selection": "memory_tier",
      "retention": "request",
      "target_request_memory_bytes": 12884901888
    }
  }
}
```

目标只影响 preset selection，不转换成 legacy `memory_budget_bytes`，也不自动生成 `memory_constrained`。
实验通道由 App 发布构建/用户显式实验功能设置与 native policy 控制，不由普通请求一个 bool 授予。

### 8.2 Replay 使用同一 schema 的 exact preset selector

拟议执行片段：

```json
{
  "execution": {
    "policy": "gpu",
    "streaming": {
      "schema_version": 2,
      "enabled": true,
      "selection": "preset",
      "retention": "request",
      "preset_id": "illustrative.zimage-bf16-k2-p14",
      "preset_revision": 1,
      "catalog_revision": "illustrative-catalog-r1",
      "target_request_memory_bytes": 12884901888
    }
  }
}
```

`illustrative.*` 不是已发布 ID。请求不能附带 capability record、性能数据、模型 hash 或 `release_enabled=true` 自授权。
Native 重新核验真实 source、workload 和 record；撤回的 preset 不允许 replay 绕过门禁。
同一 preset ID 不改语义，布局/component/reader/format 变化都递增 revision。
25进一步定义可选 `expected_resolution_digest`：用于绑定提交时的解析结果，不是授权票据；digest不一致需重新确认，
native仍依据自身catalog/真实artifact签发内部authority。

### 8.3 一次解析、一个执行权威

保留 `requested_selector`，resolver 输出 `ResolvedStreamingSelection`：

```text
requested target + selection source
selected preset ID/revision + catalog revision
actual artifact/workload/backend identity
canonical manual StreamingConfig (v1 semantics)
component plan + execution container
compiled layout digest
calibration and quality/performance evidence digests
capability/authority object (internal only)
memory scope + enforcement=none
```

Public preflight 后 session 必须消费同一个 snapshot/compiled plan，不再次用 heuristic 选 P/K。
不能把 unresolved v2 selector 临时填进旧 `stages`，让下游分支各自猜测含义。

### 8.4 Native query / resolve API（新增）

拟在 public C ABI 增加两个只读接口并通过 Swift/CLI/service 复用：

- `tc_streaming_options_json`：输入 model/artifact/workload/device，返回可用档、不可用原因、实验标签。
- `tc_engine_resolve_streaming_json`：在真实 engine/snapshot 上冻结 preset，返回 plan ID/digest 与展示信息。

这里只读 metadata/index；不创建 GPU pool、不启动 fill worker、不 materialize 权重。Trusted content identity
优先复用模型安装时的校验记录；确需完整 hash 时异步显示进度并缓存，不能在 UI 每次改尺寸时扫一遍大权重。
缓存按 source snapshot/verified digest 与 descriptor revision 失效。

最终 generate 重新校验 catalog/source/revision，plan token 只是绑定数据，不是外部提供的权限票据。
UI preview 可以是 `tentative`（token 行数未知），真正运行前必须 `resolved`；无适用记录就早拒绝。

### 8.5 配置优先级与冲突

| 场景 | 规则 |
|---|---|
| 无新 selector，或 explicitly off | 保持现有默认/legacy，不走 catalog resolver |
| v1/manual | 保留精确 P/G/K/D/Q；不做推荐覆盖 |
| v2/memory_tier | 不允许同时传 `stages` 或 exact preset 字段 |
| v2/preset | 不允许 `stages`；必须冻结 ID/revision 并核验目标 |
| request selector 与 profile selector 不同 | request 整体替换 selector，不按字段拼成混合 authority |
| request explicitly off 覆盖 profile on | 新 streaming 关闭；独立 memory guard 不被暗关 |
| new selector + explicit legacy residency/budget/offload | 拒绝冲突；App 主动不输出这些旧字段 |
| guard enabled + preset | 首版无 bounded record 则明确拒绝，不把目标误当 guard |
| GPU+ANE / LoRA / approximation 与 preset 不匹配 | 明确拒绝，不悄悄禁用用户配置 |

需要新的 uint64 bytes 校验（当前 slot 数解析为 uint32）；首版wire限定正十进制整数≤`2^53−1`，内部仍为uint64。
拒绝负数、bool、小数、指数写法、溢出、未知字段和重复 key；重复key需要raw JSON检查，不能依赖NSDictionary事后发现。
不接受 arbitrary executable path 或任意 JSON 替换运行时 callback。

## 9. Catalog 数据模型

推荐在版本库维护数据源，构建时嵌入只读 registry。不是把用户 profile 当 production catalog，也不让 benchmark 自行签发。
每条 record 的字段至少包括：

| 部分 | 必需内容 |
|---|---|
| Identity | stable ID、revision、schema、catalog revision、记录摘要 |
| Release | proposed/explored/confirmed/public-experimental/public-stable/revoked、review owner/evidence |
| Model | model ID、variant、weight format/dtype、可信 artifact manifest |
| Execution | adapter/backend/kernel/reader revision、GPU-only、operation、禁止的输入分支 |
| Workload | exact shape/frames/steps、prompt token rows、batch、audio、LoRA、tile 语义 |
| Layout | P/G/K/D/Q、pool classes、pass transition、multi-pool policy、retention |
| Components | text/denoise/upsample/VAE/export live order、释放边界、worker/container |
| Memory | raw metric scopes、coverage、样本范围、calibrated envelope、适配目标、余量 policy |
| Performance | baseline 类型、wall/denoise/P95、I/O bytes、设备和 SSD class、排序 rank |
| Evidence | quality/lifecycle/P0/P1/策略对照报告和 raw bundle digest |
| Invalidation | runtime/source/workload/device mismatch、撤回原因、替代建议 |

空值表示未知，不写成零；缺必要字段的 record 只能停在 proposed/explored。
Calibration 与 performance 可由多条记录引用同一个 execution preset，避免复制大量等价 layout。

### 9.1 记录状态机

```text
proposed
  → metadata-valid
  → GPU-smoke-valid
  → measured/explored
  → independent-confirmed
  → reviewed public-experimental / public-stable
  → revoked（任意已发布状态可撤回）
```

阶段升级不靠一个 `enabled=true`。每个证据只能授权覆盖的 workload/部署；tiny denoiser-only 不升级成 full-video 档位。
H3 Turbo 当前统计 INCONCLUSIVE 的记录可保留工程决策和实验标签，不能变成 statistics PASS。

## 10. Public 准入与原 private gate 改造

当前 `c_api.mm` 根据 `allow_experimental_streaming` 拒绝普通 generate；prepare 仍拒绝 active streaming。
新增 resolver/registry 后，替换为：

```text
off                 → old path（早返回，不查 catalog）
candidate/manual    → 现有开发测试权限 + adapter 检查
public/selector     → exact registry preflight → internal authority → exact route
bounded request     → 再匹配独立 memory capability；不因 layout 通过而自动授权
```

内部 authority 至少绑定 artifact、workload、layout、component plan、catalog revision、release channel。
Public H3 session 也要从内部资格对象获得权限，不能只改 API 层 bool 而遗留 session 级拒绝。
结果从 `private_candidate_constructor` 改为真实 `public_preset_registry` 时，必须确实经过新路径；旧测试路径仍如实报告。

Public prepare 首版可以明确返回 `streaming_prepare_unsupported`，并让 App 对该模式跳过 prepare，直接生成。
如果实现了 prepare，则必须与 generate 共用资格和生命周期；不能 prepare 先全驻留加载后再转 streaming。

## 11. 峰值与速度如何显示

用户主界面只显示目标、预计范围、证据适用范围和状态。详情区显示：

```text
requested target
selected preset + revision
expected full-request footprint range
observed execution-process-tree footprint peak（有采样则注明 sampled）
MLX peak / cache（可用时）
Metal live backing / allocated observation（可用时）
peak phase / memory scope / missing coverage
wall / denoise / load / wait / drain
actual P/G/K/D/Q + pool count + component policy
comparison: default-resident | legacy-streamed | same-layout | strategy
```

不能把 H3 的缺失 MLX 指标绘成“0 GiB”；不能把理论减少的权重 bytes 当成全请求实测节省量。
“慢 30%”必须说明相对哪个 baseline、哪个 shape/设备、冷还是热；没有本设备性能证据就只展示功能/内存状态。
额外每 block trace、密集采样仅在 audit/benchmark 开启，日常仅保留已存在或低频 boundary 指标。

## 12. 代码修改落点

表中带“新增”的文件名为建议；现有文件路径已按当前树核对。

| 层 | 文件 | 变更与边界 |
|---|---|---|
| UI | `apps/macos/App.swift` | 高级开关、档位 picker、不可用原因、详情；旧 Z-Image 控件保留 |
| State | `apps/macos/StudioState.swift` | `StreamingChoice` 可选 Codable、模型级偏好、旧 draft 默认 off、冲突验证 |
| Request | `bindings/swift/TurboCiderNative.swift` | v2 typed envelope、options/resolve binding；旧 NativeRequest serializer 不变 |
| Jobs | `apps/macos/JobStore.swift` | 排队冻结 selector/resolution、回放/撤回错误、运行中不可变 |
| UI query | 新增 `apps/macos/StreamingOptions.swift` | 异步 debounce、draft revision、只读缓存，不枚举 native layout 算法 |
| Insights | `apps/macos/RunInsights.swift`、`RunInsightsView.swift` | 分 scope 指标、null/partial、显示 preset 与实际布局 |
| Monitor | `apps/macos/ResourceMonitor.swift` | App RSS 保持其名称；新 worker 指标不混入旧读数 |
| Worker | `apps/macos/LTXWorker.swift` | 仅在已验证 container policy 要求时扩展 v2，保留原 component-staged 路径 |
| Contracts | `native/core/streaming_contracts.hpp` | selector intent 与 resolved manual 类型分开，新增 uint64 target |
| Parser | `native/platform/apple/streaming_config.mm`、`request.mm`、`profile.mm` | schema v2 selector、presence、整体替换和冲突；兼容 manual v1 |
| Registry | 新增 `native/runtime/streaming/preset_catalog.hpp/.cpp` | 只读 record、分级资格、revocation、artifact/workload key |
| Resolver | 新增 `native/runtime/streaming/preset_resolver.hpp/.cpp` | 纯过滤排序，输出 exact config/authority；不读取瞬时 RAM 重调布局 |
| Capability query | `native/api/c_api.mm` 与 `bindings/c/include/turbocider/turbocider.h` | additive API；public generate 的精确准入；off 快路径 |
| Device/metrics | `native/platform/apple/device.mm`、`native/runtime/memory_accounting.*` | 复用观察器，补 scope、worker/baseline、unknown，不另起默认 watchdog |
| Plan/report | `native/runtime/plan.cpp`、`native/platform/apple/results.mm` | requested/resolved/actual、record evidence、enforcement=none；避免重复 compile |
| Model routes | `native/platform/apple/ltx_session.mm`、`h3_session.mm`；Z/Flux pipeline | 消费 authority 和同一 plan；不借 legacy budget 反推 prefix |
| Entrypoints | `apps/cli/main.mm`、`services/turbociderd/service.mm` | 同一 native resolver、错误与报告；不实现各自映射表 |
| Research | `tools/native/run_streaming_campaign.py`、`verify_streaming_campaign.py` 等 | 复用 runner；内存观测、候选 export 与签核见文档 24 |
| Build | `tools/native/build.sh`、`Makefile`、Swift build/test 接线 | 注册新增文件/测试，release 不导出调试权限 |

Public ABI 扩展统一进入上表已有 C header，不新增另一份重复定义。功能拆分以文档 24 的 PR 为准。

## 13. 默认性能与缓存保护

最重要的是 off 路径不变：

- 不因展示 App 页面自动 hash 权重、tokenize prompt、编译 layout 或检测 swap。
- 未开启时不查新 catalog、不创建 executor、worker、pool、monitor，不额外 clear cache/unload。
- 不更换默认 kernel、dtype、compileGPU、ANE 或 LTX acceleration 行为。
- 老 draft 和历史 jobs 编码语义保持一致；service/CLI 默认请求同样不经过新 resolver。

从已缓存 resident session 切到 streaming 可能先释放旧缓存；这属于显式模式切换成本，计入首次 wall/峰值。
默认 cache 不因为新增一个 picker 永久失效。切回默认后的 reload 时间与下一次 warm 时间分别报告。
Preset 必须声明是否在 request boundary 清理自身权重，不用 request-scoped 与旧 retained session 的差异冒充框架开销。

## 14. 失败、更新与回退

1. 没有匹配项就拒绝；不自动转 resident、降分辨率、降精度或扩目标。
2. 可以给出“重新选择”的建议，须用户确认并提交新任务；后台不循环重试不同计划。
3. I/O/alloc/cancel/fence 失败沿用 slot fail-closed/drain/quarantine，失败不得自动签成较低档有效。
4. Catalog 更新只影响新请求；执行中使用冻结 record。排队任务执行前重新检查撤回状态。
5. 用户提供的 preset/catalog JSON 只允许开发工具解析，不授予 public authority。
6. Source 更换/文件修改使 snapshot 和资格失效；受信 hash 不能只有路径名称。
7. 回滚移除/撤回对应 record，新请求恢复“不可用”；默认路径和其他记录不受影响。
8. App 不承诺异常 drain 永不导致隔离/终止；布局发布前需证明对应 session 的取消与隔离路径。

## 15. 首次发布范围

建议发布名称：“Streaming（实验）— 已验证内存档位”。首版仅 GPU、精确 artifact 与 workload，不自动开启。
可以先上线 LTX/Z-Image/Flux 9B 中证据完整的若干档；H3 仅 Turbo，待完整 video pipeline 通过后再显示 App 档位。
这不是要求支持普通 H3、Flux 4B 或所有量化格式后才能公开。

档位发布需同时满足：

- public exact capability 与 default P0、layout correctness/lifecycle、相关 P1；
- 整请求内存覆盖完整，校准结果含余量且符合目标；
- 对选择策略做独立确认（P4），不能复用挑选赢家的数据给自己证明“最佳”；
- UI、public API、历史任务、CLI/service 的一致性验收；
- 声明 `memory_enforcement=none`，且不把目标宣传成 hard cap/zero swap；
- 低物理内存设备的适用性有专门验证；未测设备不显示“已验证可运行”。

完整 bounded guard 是后续独立发布，仍受 [05](05-memory-contract.md) 的资源覆盖与 registry 要求约束。
P3 只有在宣传“比系统 swap 更快”时必须有相应速度证据；不能用未测 P3 的目标档位做这种宣传。

## 16. 本方案结论

用户的选择可以很简单，但后台数据必须完整。推荐顺序是：**先测真实候选 → 校准整请求峰值 → 筛选稳定档位 →
审阅注册 → App 暴露目标选项**。不要先写“8 GiB 对应 K2”的映射，再用测试去补一个预先假定的结论。
四模型具体候选、Flux 所需扩展、测量工具、样本与资源限额、验收测试见下一篇。
进一步施工请使用[25 代码规格](25-public-preset-implementation-spec.md)和[26 实施验收](26-public-preset-acceptance-and-release.md)：
它们补充了当前v1任务迁移、一次规划、source lease、Flux per-pass prefix算法和独立校准verifier；这些仍是待实施设计。
