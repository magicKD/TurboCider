# TurboCider 模块职责、完整产品与高性能运行时设计

> 当前项目已经实施重构；以下为此前设计及待完成目标。现行目录/旧代码退役见 [项目重构](project-restructure.md)，真实能力见 [实现状态](rewrite-implementation-status.md)。
日期：2026-09-05。状态：详细设计，未实现。本文细化 [统一多模态框架主设计](unified-multimodal-framework.md)，覆盖代码边界、框架与 App 功能、GPU/ANE 生命周期、缓存、预热、调度和 offloading。本轮只更新文档，不修改代码或配置，不执行模型下载/编译/预热。

## 1. 总体目标与默认行为

默认使用原生 GPU 路线。用户启用配置后，才允许为匹配的模型/operation/shape/设备选择 GPU+Core ML（目标 ANE）并行；GPU 始终是完整、可验证的基础路线。`auto` 是显式启用后的策略选择能力，不表示安装后就自动采用量化或实验 ANE。

性能目标是在正确性、质量权限、内存和系统响应约束下，尽量减少端到端时间。需要同时优化首次可用时间、暖生成延迟、批量吞吐、峰值内存和功耗；不同模式使用不同目标函数，不能承诺同一配置同时达到全部最优。

首个可维护实现采用模块化单一原生运行时，默认在用户级服务中执行。不要把每个模块变成独立微服务；模型加载、GPU/ANE 执行、内存与取消必须共享明确的资源所有权。

## 2. 现有代码中的可复用资产与缺口

本轮继续读取 `src/turbocider/model_management.py`、`runner.py`，FLUX hybrid session/bindings，H3 `h3.c` / `h3_dit.c` / `h3_gpu.h`，以及 LTX `ltx_ane_mlp.m`。

| 来源 | 已有机制 | 新框架中应提取的职责 | 不应直接继承的假设 |
|---|---|---|---|
| TurboCider model_management | 下载/merge/export/compile 的准备动作、磁盘估计、receipt | Preparation DAG、产物存储、资源预算、进度与失败恢复 | 若干 cached 判断依赖文件存在；正式缓存还需身份/完整性/兼容性检查 |
| TurboCider runner | 常驻子进程、回收、进度传递 | Session 生命周期、作业状态和取消契约 | 常驻进程不等于模型仍驻留；日志不应成为内部事件协议 |
| FLUX hybrid backend | 按 manifest/bucket/variant/blocks 复用 sessions、输入输出 bindings、GPU 子图编译 | SessionPool、BindingPool、ExecutableCache | 当前路径字符串 key 不能代替内容身份；monkey patch 不进入最终模型接口 |
| H3 h3.c | resident key、request state 重建、缓存失效 | ResidentModel 与 JobWorkspace 分离 | key 中的诸多模型参数需迁入 typed identity；不能简化为 model path |
| H3 h3_dit.c | SSD streaming、下一 block 预取、final-pass eviction | ResidencyPlanner、WeightPager、PrefetchTicket、last-use release | 已有 streaming 与部分量化/ANE/复用路线不兼容；组合能力须显式声明 |
| H3 h3_gpu.h | 直接读入 shared buffer、映射 BF16、顺序流式读取 | WeightSource 与受控 I/O provider | mmap 不代表零物理内存；避免文件缓存第二份副本的系统提示不构成硬保证 |
| H3 Core ML | 异步加载、共享 I/O、manifest/hash/complement/fallback 身份 | ArtifactVerifier、异步 SessionLoader、BufferLease | 不能仅凭成功创建 MLMultiArray 就声称跨框架零拷贝 |
| LTX MLP | prefix/suffix width、full width、rows 校验；pack/overlap/GPU/ANE/join 计时 | PartitionSpec、ScheduleTrace、资源等待/计算分开统计 | 目前 hidden=4096/full=16384、256 对齐等属于该模块，不放入公共 runtime |
| 新 native FLUX | exact fixtures、C ABI、Swift cancel/recovery、Metal Euler | 数学基线、Backend 验收套件、接口生命周期回归 | Flux-only Engine、MLX Tensor 全局别名、硬编码 request 不是通用架构终点 |

这些是可迁移的设计与实现资产，均需按新接口和真实数据验证；不是宣称已经成为统一运行时。

## 3. 代码模块地图：职责、输入输出和边界

### 3.1 产品与服务层

| 建议目录/模块 | 主要功能 | 输入 → 输出 | 不负责 |
|---|---|---|---|
| `apps/macos/Workspace` | 项目/创作页面、任务关联 | 用户草稿 → SDK 请求 | 模型数学与设备切分 |
| `apps/macos/Composer` | 文本、图片、视频、音频、角色/顺序、参数控件 | OperationSchema + 素材 → InputBundle | 用模型名猜测能力 |
| `apps/macos/Library` | 素材/结果预览、来源关系、导出 | Asset/Artifact metadata → UI | 直接管理推理 buffers |
| `apps/macos/Models` | 安装/导入/完整性/加速准备状态 | Model catalog + preparation events → UI | 启动随意 shell 脚本 |
| `apps/macos/Performance` | 配置启用、设备匹配、内存/缓存与基准展示 | EffectiveConfig + diagnostics → UI | 把 unknown ANE activity 显示为利用率 |
| `services/turbociderd/ApplicationService` | Prepare、Submit、Cancel、管理与查询入口 | 版本化 DTO → plan/job/artifact handles | 模型专属分支逻辑 |
| `services/.../JobService` | 幂等、排队、事件、重试、终态、恢复 | PreparedPlan → JobRecord/EventStream | 内核调度和内存直接分配 |
| `services/.../PreparationService` | 导入、转换、分区导出、编译、校验、预热任务编排 | PreparationPlan → verified receipts | 在生成中无提示下载大模型 |
| `services/.../Transport` | XPC/本地 socket/可选 HTTP、连接和访问控制 | wire message ↔ application call | 第二套请求规则 |
| `bindings/c,swift,python` | C ABI/typed async SDK/后续薄客户端 | SDK request ↔ core/service | 独立复制模型实现 |
| `apps/cli` | doctor/models/prepare/plan/generate/jobs/cache/profile/benchmark | 参数/JSON → 相同 SDK 调用 | 自行选择 shell engine |

App 草稿状态可保存在客户端，权威作业状态在服务中；App 关闭后 CLI 或 App 发起的后台任务按用户策略继续。服务重新连接用 event cursor 补齐状态，而不是重复提交。

### 3.2 共享领域与规划层

| 模块 | 主要职责 | 关键产物 |
|---|---|---|
| `contracts` | 版本、类型、错误、事件、输入/结果 schema；统一边界 | Request/OutputBundle/ReasonCode |
| `assets/AssetStore` | 内容身份、原始/派生素材、权限 lease、导入/垃圾回收 | AssetRef/InputLease |
| `models/ModelStore` | checkpoint/component/adapter identity、本地安装锁、完整性 | LocalModelLock/ComponentCatalog |
| `models/Registry` | ModelModule factory、operation schema、版本兼容 | ModelCapabilities/ModuleHandle |
| `planning/InputResolver` | 调用模型角色校验、媒体探测、实际 token/shape 推导 | ResolvedRequest/ShapeSignature |
| `planning/RecipeBuilder` | 编码/循环/上采样/解码/媒体阶段依赖 | RecipeGraph |
| `planning/PartitionPlanner` | 枚举合法分区、筛除无产物/不兼容/质量不许可路线 | CandidatePlan[] |
| `planning/CostModel` | 硬件、shape、冷热状态、I/O 与预算下预测成本 | EstimatedCost/Confidence |
| `planning/PlanCompiler` | 固定分区、内存、事件和 weight use schedule；非通用神经编译器 | PreparedPlan/ExecutionGraph |
| `planning/PlanExplainer` | 为 UI/CLI 解释选择/拒绝/shape 转换/近似 | Explanation/PlanDiff |
| `policy/ConfigResolver` | profile 加载、schema、覆盖规则、能力上限、快照 | EffectiveConfig/ConfigDigest |
| `policy/DeviceInventory` | 硬件、OS/runtime、Metal/Core ML 可用信息 | HardwareKey/EnvironmentKey |

选择策略与物理资源分配分开：Planner 产生候选与估计，ResourceCoordinator 在运行前发放真实 admission lease。排队期间内存压力变化时重新准入，必要时产生新 plan revision；不得修改已经运行的数学图。

### 3.3 执行与性能层

| 模块 | 负责什么 | 重要接口/状态 |
|---|---|---|
| `runtime/ResourceCoordinator` | 统一 CPU/I/O/GPU/Core ML/内存/磁盘准入；前台与后台优先级 | reserve/release、ResourceLease |
| `runtime/JobScheduler` | 作业级顺序、公平性、session affinity、取消与超时 | admit/queue/age_priority |
| `runtime/GraphExecutor` | 执行既定 stage/partition 依赖、循环、join、安全点 | submit(Executable, bindings) → CompletionToken |
| `runtime/SessionPool` | 只读模型/已加载 Core ML/GPU executables 的引用、驻留和逐出 | acquire/release/evict，generation ID |
| `runtime/WarmupManager` | 低优先级、有预算的 load/first-run warmup；记录热状态 | WarmupPlan/WarmState |
| `memory/Storage` | 实际 backing/所有权/量化存储 | StorageHandle/TensorView |
| `memory/WorkspacePlanner` | 静态生命周期复用、activation arena、alias/in-place 合法性 | WorkspaceLayout/PeakEstimate |
| `memory/ResidencyPlanner` | resident/component/block streaming 选择，组件最后使用与保留成本 | ResidencyPlan/WeightUseSchedule |
| `memory/WeightPager` | 读取、转换、预取、双缓冲、in-flight pin、释放 | PrefetchTicket/WeightLease |
| `memory/PressureController` | 压力响应，逐出优先级，停止预热，边界处重新规划 | PressureEvent/RecoveryAction |
| `artifacts/BuildCoordinator` | 构建去重、依赖 DAG、可重试步骤、原子发布 | BuildKey/BuildReceipt |
| `artifacts/ArtifactVerifier` | checksum/ABI/shape/partition complement/编译兼容 | VerifiedArtifact |
| `artifacts/CacheManager` | 磁盘预算、pin/reference、版本隔离、GC | CacheLease/CacheInventory |
| `telemetry` | trace、模型阶段、资源等待、copy/I/O、命中、质量/收益证据 | Trace/EvidenceRecord |

这些是逻辑模块，不要求每项一个动态库或复杂继承层次。初期可合成 `TurboCiderCore`、`TurboCiderAppleBackends`、`TurboCiderModels`、`TurboCiderService` 与 Swift SDK。ModelModule/backend 边界使用抽象接口，热循环内部采用直接调用、预绑定参数和预建 executable，避免每个小算子都解析 JSON/动态分发。

### 3.4 Backend 与模型层

Backend 提供 capability query、compile/load、bind/encode、completion、memory accounting hook。模型不能通过访问全局 Metal queue 绕过 Coordinator。

- MetalBackend：buffer/queue/event、GEMM/attention/norm/RoPE/conv/量化融合等 kernel registry、pipeline cache、I/O provider。
- MLXBackend：阶段或块级可验证子图、内部拥有 MLX arrays、显式跨后端 tensor materialization；作为生产可用计算后端与迁移基线。
- CoreMLBackend：已验证 MLModel artifact、加载、固定/受支持 shape、IO binding、预测完成、观测与回退支持。公共框架无法承诺纯 ANE 执行。
- CPU/AppleMedia：媒体探测/解码、tokenizer、小规模预处理/参考算子、ImageIO/AVFoundation 导出。独立服务的数据对象不直接暴露 UI 对象。

每个模型模块内部按 `config/operations/preprocess/conditioning/transformer/sampling/encode_decode/partitions/validation` 分层。Samplers 可共享算法实现，但 timestep、噪声状态、shift、mask、舍入与多阶段 schedule 由模型 recipe 固定。

## 4. GPU/ANE 是一条完整生命周期

```mermaid
flowchart LR
    A[模型与 shape 身份] --> B[合法分区候选]
    B --> C[权重切分与转换]
    C --> D[编译与产物验证]
    D --> E[磁盘缓存注册]
    E --> F[会话加载与 IO 绑定]
    F --> G[受预算约束的预热]
    G --> H[GPU / Core ML 并行执行]
    H --> I[质量 · 重叠 · 总内存 · 端到端收益]
    I --> J[设备策略证据与后续选择]
```

默认 GPU 只需要它自己的 pipeline/weights/shape 准备。启用 hybrid 后，Planner 才考虑 Core ML 候选。需要新 artifact 时生成独立 PreparationPlan；用户选择“先用 GPU”或“准备后生成”，不在第一张图的进度中偷偷编译几十个大子模型。

### 4.1 PartitionSpec 必須是模型证明的合法拆分

包含：operation/recipe revision、block family/range、输入输出 ports、通道切片、完整覆盖关系、activation/gate 配对、GPU complement、shape bucket、padding/mask、dtype/量化、layout/strides、join 顺序、bias/scale 归属、fallback 算法身份。

例如 FLUX 可利用合法的 attention/MLP 分支 overlap；H3/LTX 可采用 MLP prefix/suffix。各 partition 必须明确数学等价性或近似类型，禁止跨顺序依赖错误提前执行。不能用一项“ANE 比例 50%”替代它。

shape 组合可能爆炸，禁止为所有宽高/文本长度/参考图数量预编译。先选用户常用、模型允许的少量 buckets；不匹配时走 GPU 或显式构建新的 bucket。mask/padding 不正确的模型子图不能仅补零适配。

### 4.2 构建服务与发布规则

构建阶段：source lock → cut/convert → export → compile → manifest/ABI 检查 → 数值 fixture → benchmark/evidence（按是否请求）→ atomic publish。编译成功不等于质量或性能验证完成，三个状态独立。

同 BuildKey 的并发请求采用 single-flight：一个构建任务，多个订阅者；取消一个消费者不必取消仍被使用的构建。失败保留错误与可复用中间产物，不把半成品标 ready；进程崩溃后的临时目录可识别并清理。全局 disk budget 包括原始、切片、临时、编译与校准文件，不只看最终大小。

模型切片和 compiled artifacts 默认不可变。GC 只清理未 pin、未被作业/会话引用的对象；metadata/文件提交有事务或恢复协议，避免任务引用已被清理的 artifact。单个大 Core ML compile 调用可能无法即时取消，调度只能在受支持边界停止，不承诺抢占。

Metal 的 metallib 与设备 pipeline/binary archive 也需要对应生命周期，不能把 GPU 视为完全无编译成本。Apple 提供 [Metal binary archives](https://developer.apple.com/documentation/metal/metal-binary-archives) 以减少运行时编译。Core ML compiled asset 与 loaded model 同样分离，见 [MLModelAsset](https://developer.apple.com/documentation/coreml/mlmodelasset)。实际可用 API 随 SDK/OS 检查，不写入未经验证的兼容承诺。

### 4.3 成本模型需包含准备摊销

对相同质量/输出契约的候选，计算：

`T_session(N) = T_prepare_missing + T_load + T_warmup + Σ T_request(i)`

单次请求仍须拆开 queue/asset decode/text/image encode/weight fetch/denoise/VAE/media export。对 GPU/ANE block：

`T_block = T_pack + max(T_gpu_branch, T_coreml_branch) + T_join + T_unoverlapped`

它是待测估计；共享内存带宽、队列争用或框架内部复制可能使 overlap 变差。另以实际完成事件计算设备忙时重叠与 critical path，不能把两个异步调用的 host 返回时间当作执行并行证据。

若 hybrid 每次节省 ΔT，额外准备成本为 C，理论摊销点 N≈C/ΔT（ΔT>0），实际再计内存/能耗和命中率。冷启动用户可能 GPU 更快，连续生成可能 hybrid 更优。对 INT8 hybrid 与 BF16 GPU，质量许可先通过再做比较；不得把近似带来的收益宣传成等精度提升。

## 5. 缓存必须分层，不能只有一个 clear_cache

| 层 | 缓存内容 | 主要 key | 失效/释放条件 |
|---|---|---|---|
| 原始权重/源资产 | immutable checkpoint、tokenizer、输入素材 | 内容 hash/格式 revision | 无引用且用户清理；不是每次生成清理 |
| 转换/切片产物 | GPU pack、ANE prefix、LoRA merge、量化结果 | 源 hash + converter/quantizer/calibration + partition | 转换语义改变；通常可跨兼容设备共享 |
| 编译产物 | Core ML mlmodelc、Metal archive/executable metadata | 上层产物 + compiler/target ABI/兼容性 | 编译环境或实际验证失效；不任意拷贝私有系统缓存 |
| Executable/Session | loaded MLModel、Metal pipeline、MLX compiled callable、绑定 | artifact + runtime/device/config/shape | 驻留预算、模型切换、错误 quarantine、进程退出 |
| Conditioning | text/vision/VAE encoder 输出、静态 K/V | 模型/adapter + 有序 inputs + preprocessing + precision | 输入/顺序/角色/预处理/模型变化 |
| Workspace/Binding | scratch arenas、Core ML IO bindings、staging | shape/layout/dtype/backend/session generation | 未完成读写前不可回收；shape 或 session 变化重建 |
| Tuning/Evidence | 候选成本、质量与硬件验证 | workload + model/recipe/partition + environment | 版本、OS、硬件或质量语义变化；证据不伪升级 |

不要对所有层使用完整机器 fingerprint：原始权重与可移植切片不应因 OS 升级重下载；设备编译缓存则需细粒度兼容域。缓存身份是多层派生，而不是唯一字符串路径。

应区分 `cache_hit`（磁盘存在且验证）、`session_hit`（已加载可用）、`binding_hit`（布局/地址/生命期有效）、`warm_hit`（当前 session/shape 已完成预热）。App 可将这些状态翻译为“已准备/已加载/已预热”，诊断报告保留细节。

warm state 不能跨进程、会话逐出或系统重启持久化为事实；磁盘上可保存历史 warmup 时间作为成本参考。session 重新创建后，即使路径不变也要重新绑定 buffers，不复用悬空指针。

GC 采用全局预算 + 每类上限 + 成本感知逐出。避免单纯全局 LRU：一个小型文本缓存重建成本和大 compiled artifact 不同。清理顺序通常为无引用临时文件→低价值派生物→闲置 workspace→闲置 loaded sessions→用户允许清理的编译产物；源模型由用户控制，不因自动加速缓存清理而删除。

## 6. 预热和调度

### 6.1 热状态要可解释

`catalogued → verified_on_disk → executable_ready → session_loaded → bindings_ready → warm_for_shape`

这些状态可组合，不表示质量已通过；`numerically_validated` 与 `performance_validated` 另有 evidence。warmup 使用模型声明的合法 fixture/shape，完成 GPU 与 Core ML event，同步确认成功。不能运行越界 dummy shape；预热结果不混入用户任务或 conditioning cache。

触发策略：明确用户“准备加速”、当前队首计划、预计即将切换的 stage；选中模型时只轻量探测，是否加载由偏好与预算控制。不要每次修改 prompt 就加载多个大模型，也不要启动时预热全部 H3/LTX/FLUX。

预热遵守独立 warmup memory/time/I/O budget，优先级低于用户请求；真实任务到来停止提交新 warmup 工作，在途工作完成后释放。编译与 warmup 也经过 ResourceCoordinator，不能由后台线程绕过内存准入。

### 6.2 三种调度时间尺度，共用一个资源账本

1. **作业级**：任务优先级/公平性、队首预备、session affinity、批量任务。可在有限窗口内合并相同模型/shape 以减少切换，但设置等待上限与 aging，避免视频任务饿死。
2. **阶段级**：text encoder→DiT→VAE/AV decode 的 weights 驻留与释放；可在预算允许时提前加载下一阶段，避免两个大组件同时峰值。
3. **block/partition 级**：IO prefetch、GPU/Core ML 分支、join、临时 buffer 生命周期；按真实依赖重叠。

多个 command queues 不等于更多吞吐。先限制一项大模型 generation，验证单任务内并发，再测试多任务；小型素材处理也要限制 CPU/I/O 与内存，以免推高首图延迟。

coreml load/compile 有不透明且可能较大的峰值，不能与 text encode/DiT 一起无条件并发。warming 保留的 session 可能迫使 weights streaming，因此选择 hybrid 时要联合评估 ResidencyPlan；PartitionPlanner 与 ResidencyPlanner 以有限候选迭代到可行计划，不各自局部最优。

### 6.3 性能优先级

优先减少不必要工作和转换：命中有效 conditioning、避免重复读权重/编译、stage release、预建 shape-specific bindings、减少提交/同步、GPU/ANE 合法 overlap。之后优化 fusion、量化 GEMM/attention/conv、tile shapes、布局和 kernel specialization。只优化一个 kernel 而新增两次全 tensor copy，端到端可能更慢。

不进行每层 GPU→CPU 标量读取来更新进度或计时；事件按 stage/block 边界采样，正常模式少量 trace。精细 tensor dump、频繁 finite check 和 Instruments 属诊断路线，报告其开销，不能与生产性能混比。

## 7. 大模型 offloading：统一内存下的驻留规划

### 7.1 先区分四种行为

Apple Silicon 的 GPU 与 CPU 使用统一内存；shared/private 描述访问/存储策略，不意味着两套可互相腾挪的独立容量池。将一个 GPU buffer 改成 host array 通常不会凭空降低总物理内存，还可能制造副本。参见 [Apple GPU storage mode](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus)。

| 策略 | 释放什么/付出什么 | 适用情况 |
|---|---|---|
| Component offload | 结束 text/vision 阶段后释放 weights，仅保留 conditioning；下次需重载 | 多组件 diffusion，首先实现 |
| Weight streaming | SSD 保存源/packed weights，RAM 只保留有限 block/window；付出重复 I/O/转换 | H3 等无法全驻留的大模型 |
| Activation eviction/recompute | 只对合法中间值释放、重算或受控 spill；付出算力/I/O | 激活成为峰值，且存活分析证明可恢复 |
| CPU execution fallback | CPU 执行合适小子图；可能省特定 workspace，常有延迟损失 | 单独算子/兼容路线；不是默认大模型解法 |

OS swap 是系统行为，不是引擎应依赖的性能设计。mmap 只建立映射，不保证所需页已在物理内存；不能把映射成功算作准入成功。

### 7.2 分级 ResidencyPlan

建议 `resident`、`component_staged`、`block_streamed` 三个主要模式，`auto` 先选择满足预算的低成本路线，强制 resident 不可行则报错。用户可以要求保留模型供后续任务复用，也可选择完成后释放。

`WeightUseSchedule` 固定每份 tensor/shard 的使用顺序、首次/最后使用、每个 step 重用、预计 bytes、格式转换和可逐出边界。ResidencyPlanner 由它产生：常驻集合、streaming window、staging buffers、Core ML resident set、prefetch depth 和 peak envelope。

模型模块声明兼容约束，例如源 H3 streaming 当前不兼容部分 ANE/INT8/复用选项。首轮迁移宁可准确拒绝这些组合，不能在通用 planner 中默认可以拼装。未来若要支持 streaming+ANE，必须增加覆盖这组组合的执行路径与验收。

### 7.3 双缓冲与完成事件

```text
I/O:        读取 block k+1 → 转换/pack → Ready(k+1)
GPU:        使用 slot A 执行 block k → Completed(k)
Core ML:    使用同一 block 的合法分区 → CompletedANE(k)
Join:       等待两分支完成 → 产生下一 block 输入
Recycling:  slot A 的所有消费者完成 → 才能读取 block k+2
```

WeightPager 的 ticket 保存 offset/length/dtype/checksum identity/cancellation generation；读入时不能覆盖在途 GPU/Core ML buffer。维护有界 I/O 队列，优先避免当前 block stall，再考虑更远预取；SSD 顺序性、页缓存压力与 packing CPU 开销都要测量。

数据来源可以是直接 pread、受控 mmap 或经验证的 Metal I/O provider。选择由源格式/SDK/对齐/设备支持决定，不把某个新 API 当作所有机器的硬依赖。目标尽量一次读入最终可用存储；若需要解量化或 layout pack，单独计 staging 峰值和时间。

对完全流式 DiT，粗略总耗时下界受 `max(total_compute_time, total_stream_bytes / measured_effective_bandwidth)` 限制，另加依赖/pack/join 等不可重叠成本；每步重新读的 bytes 都必须计入。此处 bandwidth 必须是在目标负载下测得的有效值，不用 SSD 标称速度。

重排计算以减少权重读取必须证明保持同一模型语义。常规 diffusion 步骤之间 latent 存在依赖，不能简单把“某层跑完所有 steps”来摊销加载。

### 7.4 最后一次使用与缓存取舍

H3 已有最后一轮按 block 组释放权重，可为正式 VAE decode 腾内存；它会使下个请求重新加载，不能同时声称完整暖 session 仍在。新计划记录 final-use eviction 与 next-request reload cost，根据当前/预计队列目标选择。

LTX stage1→upsample→stage2 边界有不同 shape/workspace；可复用公共常量与权重，但 stage1 scratch 必须按完成事件释放。VAE tiling、视频 chunk decode、流式 mux 可降低峰值，但 tile overlap/temporal context/拼接正确性必须由模型 decoder 定义，不能在 media 层任意切块。

Core ML 对内部权重与临时内存不提供与自有 Metal allocator 等价的控制。可以控制 MLModel session 生命周期或以预先编译的较小子模型拆分；不能声称任意把正在执行 MLModel 的部分权重 offload 到 SSD。

### 7.5 压力响应

实时 memory pressure 到来：停止新预热/预取扩张→释放闲置 caches/workspaces/sessions→在安全边界改变驻留计划（数学不变）→若无法满足则明确失败或要求重配。不能靠 GC 去释放在途资源，也不能静默减 frames/steps/context。

账本记录逻辑所有权与估计峰值，观测记录系统/进程 footprint、MLX active/cache、Metal 与 Core ML 估计等。它们可能重叠，不能直接相加形成“总内存”。统一内存/共享文件页和不透明 allocator 使精确物理占用不可完全控制，admission 采用实测余量而非假硬保证。

## 8. 可复用能力：共享到什么粒度

| 应共享 | 可提供的功能 | 保留模型专属的部分 |
|---|---|---|
| 资产与媒体 | 图片方向/色彩元信息、视频/audio 探测、时间轴、受管文件、原子输出 | 最终 resize/normalize、CRF/reference recipe、latent patch 与 mask |
| Tokenizer/Encoder infrastructure | tokenizer 实现注册、模板身份、batch/mask 描述、组件加载 | 特定模板、hidden layer extraction、RoPE、mask、精度和词表 |
| Weights/Quantization | safetensors 分片、范围读取、hash、共享只读 views、pack cache | 名称映射、ConvRot/量化语义、LoRA 合并与对应格式 |
| Tensor kernels | GEMM、norm、RoPE、attention、conv、激活、元素融合 | 数学参数、布局、舍入、允许融合和算法选择 |
| Sampling primitives | RNG stream、Euler 等公式、step checkpoint/复现 | flow shift、ancestral 噪声、双阶段、per-token timestep |
| Execution | typed ports、循环、依赖、fork/join、completion、cancel | 图结构与合法 partition |
| Acceleration infrastructure | artifact DAG、manifest、验证、Core ML loading/bindings、warmup | 导出子图的数学定义、切片轴、join semantics |
| Residency | resource lease、last use、weight windows、I/O、workspace reuse | 组件依赖、合法重算/tiling 和不兼容组合 |
| Diagnostics | shape/finite/误差比较、trace、延迟/内存统计、证据索引 | 各模型的质量指标、阈值与媒体判断 |
| Product contracts | schemas、jobs/assets/plans API、事件、UI 通用控件 | operation 参数与角色组合 |

抽象至少由两个实际使用者验证后再提取成复杂通用组件。先共享底层机制和数据契约，避免为了“未来所有模型”创建大量无实际用途的基类。H3/LTX 能以受控 CompatibilitySubgraph 暂时接入，但 opaque 内存/同步必须声明，最终替换子进程执行。

### 最小接口边界（设计签名，不是实现代码）

```text
ModelModule.resolve(request, inputMetadata, modelLock) -> ResolvedRecipe
ModelModule.partitionCandidates(recipe, shape) -> CandidateSet
Planner.prepare(recipe, policySnapshot, inventory, evidence) -> PreparedPlan
ResourceCoordinator.admit(plan) -> ResourceLease | Wait | Reject
SessionPool.acquire(executableKey, lease) -> SessionLease
WeightPager.prefetch(weightRange, slot, lastUseToken) -> PrefetchTicket
Backend.submit(executable, bindings, dependencies) -> CompletionToken
GraphExecutor.run(plan, sessionLeases, jobWorkspace, cancelToken) -> ResultBundle
```

PreparedPlan 包含数学/模型/输入身份、执行与驻留计划、产物集合、workspace、事件依赖、回退许可、成本与解释。运行中可变状态集中在 JobWorkspace；模型不可将上次 request 的 hidden state 留在公共 session。

Lease 的析构/释放不等于立即回收 backing：由 completion 驱动 deferred free。取消后必须 drain 在途工作。编译/预热/生成同用 resource lease，防止三套模块都认为机器仍有全部空闲内存。

## 9. 完整 TurboCider 框架功能清单

### 核心推理与扩展

- 多模型、多个 checkpoint/adapter 版本、多个 operation；按实际组件、后端与验收范围报告可用性。
- text/image/video/audio InputBundle，有序参考、首尾帧、角色与时间轴；未来 mask/control 通过对应模型 operation 加入。
- 单/多阶段 sampling、条件编码、上采样、图像/视频/audio decode、关联输出与 mux。
- 默认原生 GPU，显式 profile 开启 hybrid，质量/精度/算法近似三类权限分开。
- 常驻和 staged/streamed 模式、预算化缓存、shape-specific executables、流式媒体导出。

### 加速工程能力

- 检查本机可用分区，解释为什么某个模型/输入不能启用 ANE。
- 明确的 prepare/compile/warmup 操作，有磁盘/时间/内存估计，独立进度与取消。
- 分区及 GPU complement 的一致性验证、编译缓存复用、并发构建去重、版本失效与清理。
- 设备 profile 导入/选择/启用，候选路线试验与本机 evidence；未知机器仍可用 GPU，实验 hybrid 不跳过硬约束。
- 冷/暖/持续负载基准、质量对照、重叠 trace、自动策略仅使用合格证据。

### 完整服务与开发者面

- 用户级服务，App 退出后任务策略、CLI/SDK 同资源准入，取消与 structured errors。
- prepare/explain/submit/status/events/results；job idempotency、断连重放、持久化与 interrupted 恢复。
- Model/Asset/Artifact/Cache 管理 API；本地路径经受管资产或权限 lease 导入。
- 可选 HTTP，本地 XPC/socket；嵌入式 C/Swift SDK；后续 Python client 无需成为推理依赖。
- 无 GPU 的 contract/shape/plan 测试与真实 GPU/ANE/model 验收分离；性能测试默认关闭高开销诊断。

## 10. App 功能设计

App 建议采用统一 workspace 与以下页面；初期可合并页面，但职责明确。

| 页面 | 用户可完成的工作 | 对应框架模块 |
|---|---|---|
| 创作 | 选择可用模型/操作，文本与素材拖放，参考排序，首尾帧/crop/strength，尺寸/seed/模型允许参数 | Registry、OperationSchema、AssetStore、InputResolver |
| 计划与生成 | 查看实际输入处理/输出规格、路线、内存、准备状态；提交/排队/取消 | Planner/Explainer、JobService |
| 任务中心 | 阶段进度/ETA、重试、复制参数、批量 seed、失败原因、断连重连 | Events/JobStore/Scheduler |
| 素材与结果库 | 原生图片/视频/audio 预览、元数据、结果比较、素材来源、导出 | Asset/ArtifactStore、AppleMedia |
| 模型中心 | 本地导入、组件缺失、版本/LoRA、支持 operation、可选主动下载、删除与占用 | ModelStore、PreparationService |
| 加速中心 | 查看/选择机器 profile，开启 GPU/ANE，准备分区、编译、加载、预热，查看收益/质量状态 | Config/Build/Cache/Session/Warmup/Evidence |
| 内存与缓存 | 显示 resident/staged/streamed 模式、模型保留策略、缓存分类清理、磁盘余量 | Residency/Pressure/CacheManager |
| 诊断与基准 | 一键导出诊断、选定工作负载测冷/暖、GPU/hybrid 对比、设备信息 | Telemetry/Benchmark/Evidence |
| 设置 | 服务后台行为、离线/下载许可、输出目录、电源偏好、API 和资源预算 | ApplicationService/Config |

普通用户只需理解“GPU”“已验证混合加速”“实验加速”“低内存模式”，高级界面才显示 bucket/block split/compile identity。不能把正常错误原样变成一大段环境变量说明。

### 几个必须完整设计的交互

**首次使用模型**：导入→检查组件→可运行 operation→默认 GPU 生成。缺 ANE artifact 不应阻止 GPU；缺必要模型资产则明确不可运行，不隐式下载。

**启用 GPU/ANE**：选择匹配 profile→查看分区与质量权限→准备任务（空间/时间估计）→验证/编译→可选预热→hybrid 就绪。编译期间仍允许有预算的 GPU 任务；若不足则排队。用户可以取消准备并继续 GPU。

**加入图片后**：立即重新校验 operation/角色/shape；实际 token 数改变时更新计划，可能从 hybrid 变 GPU。保留素材和用户参数，解释原因，不悄悄缩图或裁文本。

**大模型**：计划显示常驻不可行、将采用阶段释放/流式权重和预计代价；可选择完成后释放或保留。模型根本无法在合法 streaming budget 下运行时明确拒绝，不让用户等到 OOM。

**预览与结果**：快速 VAE/低质量预览显式标 preview，不覆盖最终 output。多个候选图比较、视频首帧/时长/音轨预览使用已输出资产，不阻塞 compute critical path。

**恢复与清理**：重启只恢复历史/事件/队列状态；是否断点续推由模型 checkpoint 能力决定。缓存清理不删除用户原图、已导出结果或正在使用的分区。

动态表单使用有限的原生控件集合，由 schema 描述参数和约束；复杂模型有受控 editor。不能为了完全通用把 App 变成要求用户手动连每个 tensor 节点的开发工具。

## 11. 端到端示例

### A. FLUX 连续出图

初次 GPU：resolve token/shape→加载 text encoder→缓存 conditioning→释放 text weights→DiT/VAE→export。下一张相同 prompt/输入/recipe 且不同 seed：复用 conditioning/weights/pipelines/workspace，不复用 latent/RNG。切换 prompt 后按预算决定是否需暂时卸载 DiT 再编码。

用户开启已准备 hybrid：加载选中 shape 的 Core ML sessions/绑定→低预算预热→GPU attention/Core ML MLP 合法并行→join。短会话时 Planner 可建议 GPU，因为 Core ML load/warmup 成本可能高于单张节省；强制 hybrid 则按用户选择执行可行路线。

### B. LTX 首帧视频

AssetStore 导入首帧→模型定义的图像预处理/VAE encode→text/connector→stage1 8-step→释放 stage1 scratch→latent upscale→stage2 3-step→video/audio decode→原子 mux。两个 stage 的 rows/首帧 mask 不同，各自选择 partition/artifact。缺权重时只能验证 recipe/shape/准备计划，不能给出本机性能承诺。

### C. H3 大模型

检查目标 FL2VA/Ref2VA 与 adapter→估算常驻是否可行→选择合法 component/block streaming→建立逐块 weights 使用表与窗口→I/O 预取下一块、GPU 运行当前块→等待所有消费者后复用 slot。最后一轮可释放不再使用的 DiT blocks，为 AV decode 腾空间；是否保留 warm session由下一任务成本决定。streaming 与 hybrid 若未实现兼容组合就拒绝组合，不把两个开关都开即视为可用。

## 12. 性能验收与实现顺序

### 12.1 性能报告必须给出的维度

- 新进程/磁盘编译命中/loaded session/warm shape 分别标记，系统页缓存状态不明就写明。
- request queue、preparation、load、warmup、text/image encode、DiT、VAE/AV/media、完整墙钟。
- GPU/Core ML 分支实际完成、pack/copy/join、重叠比例与 critical path、buffer 复用命中。
- 实际读取 bytes、有效 I/O 带宽、prefetch 命中/等待、每步重新读与 final-use eviction。
- 进程/系统内存观测与各 allocator 逻辑计数分别报告；不重复加总，不只展示 MLX peak。
- 质量比较、非有限值、实际分辨率/帧数/steps/context/精度/近似，与性能一起固定。
- 足够样本后的 p50/p95/波动、持续负载和电源状态；不以单次最好结果选 auto。

测试还应覆盖：两个客户端同时准备同 artifact、编译中断、错误 manifest、缓存被清理后计划提交、session eviction 后 binding 失效、预热遇前台任务、memory pressure、cancel 时 I/O/GPU/Core ML 均在途、SSD 读取失败、warm session 的引用泄漏、模型/shape 交替后的有界内存。

### 12.2 从真实基线逐步构建

1. 固定 contracts/ModelModule/asset 与 profile；把 FLUX 已测数学迁入，保留 exact fixtures。
2. 原生 JobService/SDK/App 补齐计划与素材能力；实现 component staging、可复用 session/workspace、分层缓存身份。
3. BuildCoordinator/ArtifactVerifier 与正式 Core ML partition 接口，移除 monkey patch；固定少量本机已验证 buckets。
4. ResourceCoordinator/WarmupManager + 真实 overlap trace，联合选择 partition/residency，只有证据合格才允许 auto。
5. 抽取 H3 WeightPager/prefetch/final-use eviction，建立无权重 I/O/lease tests；H3/LTX 模块静态迁移与真实模型验收按各自清单推进。
6. 根据 profile 测量替换 Metal 热点、引入更多 shape/量化/图像编辑/视频路径；只优化已定位瓶颈。
7. 完整 App 加速/模型/内存中心、服务后台生命周期、版本升级与发行验收，再移除旧正式运行链路。

关键交付判断：不仅有更多模块名字，而是新增模型不再改 JobService/App 分支；新增分区不再复制加载/缓存/预热/回退机制；新增机器 profile 不改数学代码；offloading 不破坏 buffer 生命周期；每项优化能在同一质量契约下解释实际端到端收益。
