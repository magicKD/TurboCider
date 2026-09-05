# TurboCider App 实现、契约与验收方案

状态：实现提案，2026-09-05。配套：[产品设计](app-product-design.md) · [视觉规范](app-visual-system.md)。

第二轮优先约束见[多模态输入、种子与高级参数](app-inputs-and-parameters.md)：首版一次一个主输出、App 一次一个活动生成；图片插入/粘贴/拖放、排序与 seed 策略列入首版。当前 native 已加入 FLUX 编辑及 H3 入口，但静态实现不等于真机验收。

## 1. 实施边界与迁移策略

本轮交付是设计文档和界面原型，不修改推理代码、不启动模型，也不宣称 UI 已完成实现。

保留 SwiftUI + AppKit + AVKit，复用既有请求/能力过滤/素材暂存行为。新 App 界面只依赖产品领域对象和 `GenerationService` 协议；当前 Python/HTTP 控制平面与新原生引擎由 adapter 接入。App 不通过引擎名字分支去猜功能。

默认产品目标为原生运行时。过渡适配器是明确标识的部署选择，不在原生操作不可用时偷偷启动旧后端；backend identity 写进任务快照、日志和性能记录。现有两套模型文件、历史目录不直接混写。

建议先在 `native/swift/` 实现新外壳和 FLUX 完整纵切，提取可复用 UI/领域组件到同一模块；需要保留旧 App 时让其复用同一组件。具体拆包与最低系统版本沿用实际构建目标，不仅为了 UI 重构提高系统门槛。下列文件布局是拟新增结构，不是已有文件：

```text
App/
  AppShell.swift                 窗口、导航、菜单、服务注入
  DesignSystem/                  tokens、按钮/字段/状态组件
  Features/Studio/               Composer、Inspector、查看器、结果条
  Features/Library/              索引、浏览、详情、比较
  Features/Tasks/                队列、任务详情、性能详情
  Features/Models/               登记、准备、会话与组件状态
  Features/Settings/             存储、执行策略、服务、诊断
  State/                         DraftStore、JobStore、SelectionStore
Domain/
  Draft.swift                    可变编辑状态
  RequestSnapshot.swift          不可变提交请求
  CapabilityCatalog.swift        operation 与字段约束
  JobSnapshot.swift              任务状态与遥测
  AssetRecord.swift              资产身份、关系与所有权
Services/
  GenerationService.swift        模型/计划/提交/观察/取消
  NativeGenerationService.swift  原生 C ABI/未来服务适配
  CompatibilityService.swift     可选 HTTP 适配
  AssetService.swift             受管素材、缩略图、导出
  ModelService.swift             登记/准备/会话控制
```

视图不拥有 Engine、daemon 生命周期或生成 Task。窗口关闭只解除观察；Application/Service 层决定真正取消还是保持运行。共享 `@MainActor` stores 发布小型不可变快照；解析事件、文件 I/O、缩略图与 AV 元数据探测在后台执行。

## 2. 能力、草稿与计划契约

### 2.1 CapabilityCatalog

当前 `TCModelCapabilities` 的 modes、roles、recommended 参数足以启动基本表单，但不足以描述全部离散约束。拟补充版本化 `OperationDescriptor`：

- `id / modelID / backendID / schemaVersion`；操作运行状态与 unavailable reason。
- 输入角色、类型、数量、顺序意义、互斥/依赖组合、strength 语义与范围。
- 输出尺寸约束、合法帧数、fps、音频 allowed/required、固定或可编辑 schedule。
- 有限字段类型：text、integer、decimal、enum、toggle、asset、orderedAssets；默认值与显示层级。
- 当前设备/资产/shape 的执行候选由 planner 判断；模型描述不能直接给“本机已验证”结论。

两级校验：UI 提供即时字段校验；planner 是最终可执行性权威。原生只提供 FLUX 时只开放 FLUX 文生图；兼容 adapter 单独翻译自己的 catalog。未知 schema 展示可读的“不支持的模型描述”，不能忽略未知必需字段继续生成。

### 2.2 Draft 与提交

草稿保存 prompt、model/operation、输入资产有序 ID、输出/采样/执行字段、schemaVersion、updatedAt。建议每 500 ms 去抖保存、失焦和提交前 flush，原子写入；保存失败必须可见。

每次修改产生 `draftRevision`。计划请求去抖约 300 ms；响应只用于同一 revision，旧响应丢弃。提交绑定 validated revision，若模型资产/资源状态改变，服务重新校验。当前 `/v1/plans` 是候选列表，不是资源预留；UI 不承诺看到 plan 就一定能立即运行。

生成创建 `RequestSnapshot`，记录原始及解析后的参数、模型 revision/权重身份、backend、operation、输入资产引用、seed、真实输出规格、执行计划与近似许可。不可用粗略 seed+prompt 宣称完全可复现。

将 `idempotencyKey` 作为未来 submit 契约：同一次点击重试不会生成两个任务。旧服务无此能力时，提交超时视作“提交结果未知”，先刷新任务并人工定位，不能盲目自动重试。

### 2.3 GenerationService 拟议接口

```swift
// 设计签名，需按项目既有编码风格与最低系统版本实现。
protocol GenerationService: Sendable {
    func capabilities() async throws -> CapabilityCatalog
    func plan(_ request: RequestSnapshot) async throws -> PlanExplanation
    func submit(_ request: RequestSnapshot, idempotencyKey: UUID) async throws -> JobSnapshot
    func jobs() async throws -> [JobSnapshot]
    func observe(jobID: String, afterSequence: Int64?) -> AsyncThrowingStream<JobSnapshot, Error>
    func cancel(jobID: String) async throws
}
```

`supportsQueue / supportsReplay / supportsBackgroundExecution / supportsPreload` 是服务能力。当前 native 单任务 busy adapter 对第二任务返回明确不可排队；首发之后经产品与服务能力验收再考虑“加入队列”。两条 adapter 都不能编造缺失的 API 返回值。

## 3. 遥测与真实生成速度

### 3.1 现有可复用与缺口

| 来源 | 当前有 | 缺口/处理 |
|---|---|---|
| Swift `TCJobRecord` | state、phase、overall progress、elapsed、ETA、输出 | 无 step scope、采样时间、内存、标准 metrics，需可选扩展 |
| Python JobManager | 阶段进度与末尾日志、最终 metrics、持久化 | 不从任意 log 文本在视图中提取速度；结构化适配器上报 |
| 当前 SSE | 约 250 ms 检查并推送变更快照 | 无 seq/replay；先 snapshot reconcile，后续新增恢复契约 |
| NativeEvent | phase、completed、total、elapsed_seconds | 计数单位依阶段变化；只能对已知 sampler phase 推导阶段步速 |
| 系统报告 | 芯片、总内存、设备信息 | 静态容量不是实时占用；资源 sampler 单独实现 |

### 3.2 事件契约（拟新增）

采用版本化 envelope：`schema_version, job_id, sequence, kind, observed_at, monotonic_ns, payload`。

`sequence` 在同一 job 内递增，重复去重，倒序丢弃；流重启需新 stream identity 或保持持久 sequence。`observed_at` 用于显示采样时间，速率和计时用单调时钟；不同进程的单调时钟不能直接相减。

payload 支持：

| 字段 | 语义 |
|---|---|
| state | queued / preparing / running / cancelling / succeeded / failed / cancelled / interrupted |
| phase_id, phase_label, phase_instance | 当前阶段及其唯一实例；多阶段/重复阶段不混合 |
| completed, total, unit | 当前阶段计数；unit 是 step / frame / byte / item，不默认都为步 |
| elapsed_run_ms, elapsed_phase_ms, queue_ms | 分离运行、阶段和排队时间 |
| overall_fraction, progress_kind | 有可靠计划才给总体；kind 为 measured / estimated / indeterminate |
| step_index, step_total, step_duration_ms | 可选、真正已完成采样步，异步提交时间不能冒充执行完成 |
| rate_value, rate_unit, rate_window_ms, sample_count | 有定义的速率及统计窗口 |
| eta_low_ms, eta_high_ms, eta_basis | 可选估计范围；来源为历史/阶段测量等 |
| preview_asset_id, preview_sequence | 可选可解码预览，区别最终产物 |
| process_footprint_bytes, memory_pressure | 可选进程与系统观测，带 scope 与采样时间 |
| requested_execution, resolved_plan, session_reused | 用户请求与实际计划、复用情况 |
| error_code, recovery_actions | 稳定错误分类与恢复动作 |

缺失值使用 null/缺省而非 0。终态附 artifacts、真实媒体元数据、分阶段时间及总墙钟；保留旧 Codable 字段的解码兼容。旧 `failed + phase interrupted` 在 UI 适配为“中断”；不修改历史原始事实。

### 3.3 指标定义

1. 当前阶段最近最多 5 个有效完整采样步，`秒/步 = sum(step_duration) / sample_count`，`步/秒 = sample_count / sum(duration_seconds)`。至少 2 个有效样本再显示；样本不足显示“测速中”。LTX 8 步 stage1 与 3 步 stage2 独立计算，load/encode/decode 不混入。长单步过程中保留最后样本并注明采样时间，不将当前未完成步算成更快。
2. 完成后图像端到端“秒/张” = 请求从实际执行开始至最终输出提交的时间 / 成功图像数。单张优先显示总用时，冷加载单列；排队与下载另列。
3. 视频端到端吞吐 = 实际输出总帧数 / 执行到最终提交秒数，标“生成吞吐”。仅最终帧数已知且统计窗口有效时计算；当前 denoise 阶段不能根据进度百分比推造已生成帧数。若有流式 decode，单独标“解码帧/秒”。
4. 媒体播放 fps 来自输出元数据，与上述吞吐完全分离。
5. ETA = 当前阶段剩余量 × 当前阶段稳健步时 + 已知后续关键路径估计。历史匹配维度至少 model revision、operation、shape、steps、精度/plan、芯片、冷/暖、预览配置。样本不足或没有后续阶段估计时只显示当前阶段估计，不能冒充任务总 ETA；未知展示“估算中”。
6. 总进度仅在 planner 给有效权重时汇总；视频音频并行解码按依赖关键路径估计，不能把并行耗时相加。阶段 100% 不等于全程完成；最终提交前不能显示任务成功。
7. 进程 footprint、MLX active/cache、Metal/Core ML 计数分别标来源，不相加为一个虚假的“总内存”。系统总容量来自 system report；系统压力独立显示。普通进程未拿到可靠 GPU/ANE 占用时显示“未提供”，不引入特权采集作为首发依赖。

### 3.4 刷新、断连与性能预算

- 引擎只在安全完成边界发轻量事件；App 每秒最多合并刷新状态 4 次，终态立即应用。elapsed 数字 1 Hz 即可。
- 内存采样初始 1 Hz，后台或非详情页降至 0.2 Hz；阶段事件不会触发媒体重建。
- 最近 120 个速率点内存环形缓冲足够，详情按需绘图；不持续写高频全量曲线。
- 预览默认不额外开启；只有已有低成本输出时更新，初始上限每 2 秒一次。生成/解码预览的成本单独测量并让用户选择，禁止为 UI 强制同步 GPU。
- 有心跳时超过 3 个心跳间隔标“数据已过时”；无心跳的旧接口不能因长步无事件判定断连，以连接状态及查询为准。
- 断流后立即查询任务快照，再有限退避重连；polling fallback 默认 1 秒、连续失败后 2–5 秒，不重复提交任务。有事件重放时带 cursor；快照修复 sequence 缺口，终态不能被迟到 running 覆盖。
- 性能验收预算：遥测+UI 开启对匹配工作负载的端到端中位耗时额外影响目标 ≤2%；样本至少 5 次且注明噪声，失败先降低采样/预览，不修改质量或步数。该值是工程门槛，不是当前测量结论。

## 4. 任务/模型/资产服务补齐

### 4.1 队列与持久化

JobService 持久化接受后返回 jobID。状态机：queued → preparing → running → succeeded/failed；queued 可直接 cancelled，active 经 cancelling 后进入 cancelled 或已提交成功，服务中断为 interrupted。任务取消与输出原子提交竞态以服务终态为准。

首版 App 一次一个活动生成，每次一个主输出；运行中允许编辑但不再提交。底层服务可保留资源等待队列，App 不开放批量或重排。准备/预热/生成共享资源准入。首发之后的队列重排仅针对未执行任务；正在使用的模型不能立即释放。暂不实现暂停、断点续推或多模型并行。服务独立运行、App 重连和崩溃恢复是后台继续生成的前置任务。

首个 UI 增量可读取既有历史；原生 jobs.json 扩展需 version、原子替换与备份。后续数据量增大时迁移 SQLite，模型元信息/资产/任务分别索引；不能让每个遥测 tick 重写整个素材库。

### 4.2 模型管理接口（均为新增要求）

`inventory / importLocal / validate / prepare / observePreparation / cancelPreparation / load / unload / removeRegistration`。下载与删除文件为不同明确动作，不能用“Load”同时做下载、编译和内存驻留。

模型 identity 包含 family、revision、组件/manifest hash、格式与路径登记。导入先只读校验，默认登记外部位置不复制数十 GB 权重；可选择迁入受管目录并预估空间。准备过程显示下载+临时转换+最终产物空间，H3 按 preparation 真实估算，不固定写某个“最低内存”数字。

加载/预热单独 task 与进度；无底层进度时用未知进度状态。load completion 必须来自可复用 session，不凭文件存在推断。卸载等待 lease 释放；删除模型需检查引用与任务占用。外部模型默认只能移除登记，明确选择文件删除时才进入独立确认。

当前 HTTP SDK 未暴露完整 assets/preparation/load/unload 生命周期；不能仅在 Models 页面加按钮就视为完成。native 的 engine 创建也不等于所有权重已预热；需真正 session 状态事件后显示“预热完成”。

### 4.3 资产与结果

AssetRecord 包含 UUID、类型、来源/父资产、受管路径或 bookmark、内容 hash、媒体元数据、所有权、引用关系、预览位置。输入仍通过 staging/受管副本传给服务；临时导入文件在无草稿/任务引用后回收。

任务输出先写临时产物，完成原子提交再入库；预览与最终结果有不同 asset kind。导出副本保留用户原文件；缺文件/外部卷离线允许重新定位。沙盒发行时实现 security-scoped bookmarks，当前非沙盒运行不凭空假定已满足权限。

图片使用降采样缩略图，按可见区域加载；视频按需提取封面。AVPlayer 由 viewer model 持有，遥测更新不能重建播放器导致跳回开头。列表/网格虚拟化、缩略图有缓存上限，原图不放全局 Published 数组。

## 5. 分阶段实施与退出条件

按小纵切提交；时间依赖引擎迁移，不用 UI 工期掩盖 H3/LTX executor 的工作量。

| 阶段 | 实施内容 | 依赖 | 验收退出条件 |
|---|---|---|---|
| P0 契约与基线 | 确认运行时、能力矩阵、tokens、JobSnapshot、mock fixtures | 现有代码审计 | 两 adapter 能力/缺失状态明确，视觉评审通过 |
| P1 Studio 重构 | AppShell、Composer、Inspector、大预览、状态条；图片插入/粘贴/拖放、角色槽/排序、seed 编辑/随机策略、单输出；原生 FLUX 纵切 | P0 | 真实 FLUX 生成/取消/保存/历史；1280×800 核心无需滚动；无假遥测 |
| P2 可观察创作 | 结构化阶段步时、可选 metrics、断连修复、任务列表、草稿与结果索引 | P1 + 事件契约 | 速度与原始事件计算一致；切页/重启不丢草稿；兼容字段解码通过 |
| P3 三模型首发 | FLUX 编辑、LTX 视频/首帧、H3 支持的参考组合；播放器与模式专属参数 | 各原生 executor 与模型验收 | 三模型按 operation 真机生成、输入约束/取消/媒体完整性通过；不靠 mock 放行 |
| P4 模型生命周期与后台 | 导入校验/准备/load/unload、资源准入、独立服务、App 退出策略 | ModelService + SessionPool + JobService | 模型切换、空间不足、离线卷、取消准备、释放占用、退出后任务完成可恢复 |
| P5 首发后进阶 | A/B、批量 seed（需另行定义产品范围）、加速详情、缓存、诊断导出 | P2/P4 + 能力支持 | 数据可追溯、性能不退化、清理不破坏引用 |
| 后续 | mask、局部重绘、LoRA、延长/重拍/多关键帧 | 新 operation executor/schema | 单独功能验收后再出工具入口 |

P1/P2 可以发布明确的 FLUX 预览；“完整三模型首发”必须等待 P3，若首发要求模型中心完整 load/释放与后台则也必须完成 P4。P4 的部分导入校验可前移到 P1，避免首次使用仍依赖手填路径。优先顺序是依赖关系，不表示推迟用户提出的首发模型承诺。

### 首个可评审实现切片

1. 固定测试 fixture：无模型、GPU 就绪、缺组件、文生图运行、无 telemetry、失败、长提示词。
2. 提取现有参数逻辑与文件选择行为，替换视觉外壳；保留旧构建入口用于回归。
3. 接 native FLUX 文生图、job 观察与能力描述；对新加入的 FLUX transform/edit、H3 executor 逐项做真机验收，再开放生产入口；LTX executor=false 时不启用。
4. 检查数据更新不重建播放器/缩略图，任务、草稿与选中历史互不污染。
5. 真实生成与取消通过后，再接入并验收编辑/视频；UI 重构不同时改模型数学。

## 6. 验收用例与质量门槛

| 范畴 | 用例 | 通过要求 |
|---|---|---|
| 布局 | 1440×900、1280×800、980×700；中英文；浅/深主题 | 无重叠/截断主操作，检查器可访问，结果主体清晰 |
| 可访问性 | 全键盘、VoiceOver、减少动态效果、高对比 | 控件有名称、顺序合理；错误/阶段可读，不高频播报速度 |
| 能力 | 文件/粘贴/拖放、FLUX 参考顺序；LTX 首帧/8+3；H3 互斥组合；seed 随机只解析一次；单输出 | 请求与 UI 相符；能力不足明确阻止，未用素材不丢失 |
| 参数 | 模型切换、计划响应乱序、非法 shape、固定 schedule | 旧计划不覆盖新草稿，不静默改变 frames/steps/输入 |
| 任务 | 双击提交、超时、取消/完成竞态、事件重复乱序、断连 | 单逻辑提交；终态稳定；不把断连等同失败 |
| 遥测 | 0/1/多步样本、阶段重置、停顿、缺失值、并行 decode | 速率公式正确，阶段不混算，无 NaN/虚假 0/假 100% |
| 模型 | 缺组件、外盘断开、load 失败、使用中 unload、空间不足 | 保留草稿；准确状态；不删除源文件或在途资源 |
| 媒体 | PNG、长视频、有/无音轨、缺输出、方向/色彩元信息 | 播放不因进度更新重置；导出与原输出一致 |
| 性能 | 基线 vs UI/遥测；大库至少 1000 条 | 主线程不做权重/原图读取；有界缓存；测量满足预算或记录待修 |
| 迁移 | 旧任务 records、新字段缺省、服务重启 | 历史可读；不能恢复的任务显示中断；不假装断点续推 |

测试分层：纯契约/状态 reducer/速率计算单测；SwiftUI 可访问性与关键旅程 UI 测试；真实模型端到端测试单列并需要权重/设备。只对会破坏正确性的行为写有意义测试，tokens 和简单留白修改以截图审阅即可。

发布检查必须附实际测试环境、model revision、backend、执行计划、运行记录、媒体可播放性与性能口径。现有 `docs/design/validation/` 是历史证据，不作为这次新 UI 或尚未迁移操作的验收结果。

## 7. 风险与已作出的取舍

- 后端能力不齐：统一产品语义，运行时各自报告，原生缺能力不静默回退。
- UI 陷入参数堆砌：常用/专属/高级三层，工具只随 operation 出现。
- 实时数字看起来专业但不真实：规范单位与采样来源，允许未知值，默认不展示 GPU/ANE 百分比。
- 大模型加载与内存压力：资源准入/会话 lease 必须在服务层完成，视图按钮不自己释放 engine。
- 提前建设完整工程编辑器：先草稿/资产/任务关系，延后 timeline 与图层；保留模型编辑扩展点。
- 性能被 UI 拖慢：状态合并、缩略图/播放器稳定持有、默认不新增中间 decode，测量后再扩展图表。
