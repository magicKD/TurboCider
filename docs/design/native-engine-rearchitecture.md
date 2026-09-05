# TurboCider 原生多模态推理引擎重构提案

日期：2026-09-05。状态：架构建议，尚未实施。

[查看架构图](/Users/kd/Documents/project/TurboCider/docs/design/native-engine-architecture.png) · [可编辑 Excalidraw](/Users/kd/Documents/project/TurboCider/docs/design/native-engine-architecture.excalidraw) · [SVG](/Users/kd/Documents/project/TurboCider/docs/design/native-engine-architecture.svg)。图中 Native Library 在服务模式由 Engine Host 加载，也可直接嵌入 SDK 调用方；PNG/SVG 在本地从图元导出。

## 1. 建议与边界

将 TurboCider 定义为 Apple Silicon 上的原生多模态生成运行时：它拥有模型执行计划、张量与内存生命周期、GPU/ANE 调度和媒体输出，向外提供可嵌入库、常驻本地服务、SwiftUI App、CLI 与 SDK。

核心技术组合建议为 **C++20 + Objective-C++ + Metal + Core ML；Swift 负责产品、服务和 SDK**。现有 C11/Objective-C 优化可逐步纳入，不要求为了语言一致性重写正确的算子。Python 留在模型转换、校准与验证工具链；发布后的标准生成链路不依赖 Python、ComfyUI、mflux 源码目录或开发机虚拟环境。

彻底重构应改变模块边界和资源所有权，但按可验证的垂直切片替换实现。现有代码作为参考实现、性能基线和优化素材，不继续在旧适配器上叠加产品功能。

首发范围明确为三个已知模型包：H3 的指定 checkpoint/LoRA recipe、LTX-2.5 22B distilled、FLUX.2 Klein 4B。模型家族名称不代表全部变体受支持；9B、其他 H3 recipe、未验证的 LTX 模式分别验收。现有 FastMetal 暂不纳入本次核心重写，后续可验证新模型接入能力。

建议主发行版以 macOS 26+、Apple Silicon 为目标，让 Metal 4 成为主路径；若必须支持 macOS 15，则另设经过测试的经典 Metal/MPS 后端。API/硬件可用性与模型内存门槛分别探测，不把“支持 M1”写成“所有 M1 都能运行所有模型”。这是产品选择建议，实施前需确定支持的设备矩阵。

## 2. 现状：已有资产与真正缺口

| 部分 | 阅读到的事实 | 重构处理 |
|---|---|---|
| TurboCider | Swift App/SDK → HTTP → Python 任务与策略 → 各引擎；FLUX 有常驻 worker，H3/LTX 在这里主要按原生子进程接入 | 保留请求、任务、模型清单与显式近似策略的设计经验；重新实现控制面，计算统一下沉 |
| h3.c-fork | 已有 C 库 API、Metal 张量、权重映射/流式加载、会话与 request state 分离、预览和多模态输入；包含公开 Core ML 与私有 ANE 实验 | 提炼张量/设备/权重/算子资产，保留 H3 特定语义，重新审查全局状态和所有权；私有 ANE 不进入标准发行路径 |
| ltx-mac | 主计算 C/Objective-C/Metal/MPSGraph；MLX C++ upsampler/VAE；文本 conditioning 和部分媒体流程仍依赖外部工具；完整流程集中于部分 bench 工具 | 将正式 pipeline 从 bench 中移入库，原生化 conditioning 与媒体环节，保留双流和 8+3 recipe |
| flux2-engine | Python+mflux/MLX 模型层、进程级 attention patch、Objective-C++ Core ML buffer bridge、常驻模型与 shape bucket | 用原生模型模块替换 patch；吸收独立 attention/MLP 分支并行和复用 backing 的经验 |

这些事实来自代码和仓库记录，不是本轮新跑的推理结果。独立仓库与 TurboCider 的内置副本必须先逐一固定 commit；不能默认它们相同，也不能默认某份 README 的支持声明已经覆盖最新源码。

现有证据尤其说明：

- LTX 报告的 GPU dense 59.483 s → GPU+ANE dense 49.467 s，是 M4 Max 64 GB、固定形状的 **decoded-pixel** 口径，排除了 conditioning、fresh setup 和 MP4 封装；不能当作点击生成到成片时间。其 704×480 请求实际解码为 704×448。
- FLUX 的 1024 示例报告 22.65 s → 17.19 s，但排除了 session 初始化；首次对应 Core ML session 额外约 8.68 s。常驻和单次运行需要不同选择。
- H3 的 2026-09-01 报告：同 BF16 基线比较约 1.0407×；相对当时更快的默认 GPU 约 0.9960×。这不是所有设备的永久结论，但足以否定“开启 ANE 必然更快”的默认假设。

## 3. 目标分层与依赖方向

```text
SwiftUI App             CLI                  Swift/Python 客户端 SDK
       └──────── 本地任务协议 / 版本化 HTTP+事件流 ────────┘
                            ↓
TurboCiderService：任务、模型资产、持久化、访问控制、全局准入
                            ↓
Engine Host：常驻原生进程；内嵌同一套 libTurboCider
                            ↓
H3 recipe            LTX recipe             FLUX recipe
             └── 构建有语义的 Pipeline Graph ──┘
                            ↓
Planner：形状推导、布局、精度、融合、设备放置、内存与依赖规划
                            ↓
Runtime：Session / Tensor / WeightStore / Executor / Cache / Trace
                 ↓                 ↓                 ↓
        Metal GPU backend    Core ML backend    CPU / Media backend
       Custom / MPS / MPP     经验证的 ANE 子图    Accelerate / AVFoundation
```

模型资产清单和设备能力共同约束 recipe/Planner；诊断贯穿全部层。上图的 Core ML 后端与 Metal 后端并列，绝不通过 Metal 去提交独立 ANE 指令。

依赖始终向下：App 不知道 kernel；模型模块不创建自己的 MTLDevice、私有总内存池或 HTTP server；后端不知道 prompt、模型名、任务数据库；核心库不依赖 App、网络、服务进程或 Python。

### 3.1 两种交付方式，共享一个核心

**服务方式（App/常规 CLI 默认）**：用户级常驻 Service 持有任务和模型目录，通常只启用一个重型 Engine Host。App 断开不结束已经接受的任务。CLI 关闭、窗口关闭和主动停止服务具有不同语义，状态需持久化。

**嵌入方式（原生 SDK）**：直接加载 libTurboCider，自行创建 Engine/Session，不要求后台服务。默认不在嵌入进程中隐式连接或启动 daemon。多进程嵌入无法共享一个精确的全机分配器，明确由调用者协调，或选择服务方式获得集中准入。

Service 和 Host 之间用 XPC 做生命周期、控制与故障隔离；产品客户端首版可统一复用 HTTP/事件协议，避免首版同时维护两套客户端协议。需要 App 私有媒体预览优化时，再增加轻量 XPC 句柄传递。XPC 是控制与受控资源通道，不是逐算子 RPC。

正常推理阶段在同一个 Host 内执行，不按模型模块或 Transformer block 拆成进程。只有无法可靠回收的旧框架阶段才临时使用隔离 Host；跨进程仅传媒体或明确的阶段产物，传输成本计入性能。

## 4. 核心运行时应拥有哪些抽象

| 对象 | 职责与不变量 |
|---|---|
| Engine / DeviceContext | 探测能力、后端、队列、编译缓存和预算；生命周期长于请求 |
| ModelPackage / WeightStore | checkpoint 身份、组件索引、量化格式、分片读取、不可变权重、按需装入/卸载 |
| ModelSession | 持有模型/组件和兼容编译计划；不得把“常驻”解释成所有组件始终留在 RAM |
| GenerationRequest | 输入角色、采样和输出意图，不含路径透传式底层开关 |
| PreparedPlan | 请求验证后的实际尺寸、节点依赖、设备放置、精度、工作区需求、近似声明和失效条件 |
| ExecutionContext | 单次请求的随机状态、latent、临时张量、取消令牌和事件；不得泄漏到下次请求 |
| Tensor / Storage | dtype、shape、stride、layout、byte offset、alignment、owner、别名关系和完成依赖 |
| MediaAsset / Result | 图像、帧序列、音频、编码文件及实际元数据；张量结果和文件导出可分离 |

公开 C ABI 使用 opaque handle、显式 retain/release、错误码、结构体 size/version 和 callback context。C++ 异常不穿过 ABI，回调线程与缓冲区有效期必须写清。Swift 封装为 async/await、AsyncSequence 与取消；Python 服务 SDK 不需要加载模型框架，后续可通过 C ABI 提供进程内绑定。

### 4.1 两层图，不先造一个通用编译器

第一层 **Pipeline Graph** 描述文本/图像编码、DiT、采样循环、latent upsample、VAE、音频和封装，可含多输出和分支。模型模块保留完整数学语义。

第二层 **Execution Plan** 描述具体算子与融合子图、buffer 生存期和 GPU/Core ML/CPU 依赖。循环采用可复用的 block/step 模板与运行时参数，避免将每个 diffusion step 展开成巨型图。

首版采用手写模型图、有限算子集、确定的 rewrite passes 和离线已验证的设备切分模板。无需立即引入 MLIR、通用 ONNX importer 或自动搜索任意图切分。把 compiler 接口预留为“从语义图生成执行计划”即可，后续有实际收益再扩展。

一次准备流程：请求/模型清单验证 → 推导实际尺寸与 token → 选择 recipe → 约束精度与近似 → 选择候选后端/子图 → 布局与融合 → 生存期/预算检查 → 形成依赖 DAG → 编译/缓存 → 执行。

## 5. Metal GPU 后端

第一版优先统一 Linear/GEMM、量化解码、RMSNorm/LayerNorm/AdaLN、QKV+RoPE、SDPA、激活/残差、layout 转换、patchify，以及 VAE 所需卷积和上下采样。

采用三类实现并允许逐形状择优：

1. 自有 Metal kernel：需要融合、减少带宽或处理专用量化格式的热点。
2. MPS/MPSGraph：参考与通用覆盖，特别是尚未证明自研更优的 attention/conv。
3. Metal 4 TensorOps/MPP：针对适合的矩阵运算和新 GPU 能力进行专化；保留能力检查和旧设备路径。

MPS 与 MLX C++ 都可作为过渡后端，但必须声明不透明内部资源和同步边界。MLX 具有自己的执行与分配机制，不能仅靠包一层 Tensor 就宣称已经统一零拷贝/显存池。按完整子图接入、预算留余量、测量交接成本，逐步替换最关键热点。最终标准路径不保留对 Python mflux 内部类的运行时 patch。

重要优化目标是减少激活物化、融合 norm/quantize/projection epilogue、复用 workspace、批量编码小算子、避免逐算子 waitUntilCompleted，以及选用不会显式构造巨大 attention score 矩阵的实现。保留 CPU/参考实现作为数值 oracle，不要求 CPU 能实用地完成每个巨大模型。

不要把 M5 GPU 中的 Neural Accelerators 与独立 ANE 混为一谈。Apple 的 TensorOps/MPP 用于 GPU 内矩阵加速；Core ML 是本方案调用独立 ANE 的公开路径。[Apple Metal 4](https://developer.apple.com/videos/play/wwdc2025/205/)、[M5 GPU TensorOps](https://developer.apple.com/videos/play/tech-talks/111432/)。

## 6. ANE 后端与异构调度

### 6.1 接入粒度

Core ML 后端执行固定或受约束 shape 的较大子图，例如合适的 FFN、独立 projection 或 encoder 阶段，不为每个 elementwise op 单独调用 Core ML。模型包提供候选子图与兼容清单，Planner 根据实测数据选择。

FLUX 已有适合的结构：同一归一化输入分出 GPU attention 分支和 ANE MLP 分支，在 residual 前汇合。LTX 根据真实双流依赖安排 MLP/文本 K/V；不能机械套用 FLUX 的 fork/join。H3 默认使用 GPU，现有 ANE 方案作为候选而非上线前提。

规划收益使用端到端成本，而非 TOPS：

```text
顺序子图：T_hybrid = T_pack + T_CoreML + T_unpack + T_sync
可并行分支：T_hybrid ≈ max(T_GPU_branch, T_CoreML_branch)
                         + T_boundary + T_join + T_contention
```

成本还要区分首次编译、冷模型装入、热预测、计划复用次数与共用内存带宽。这里的 max 公式只适用于真实独立分支，不能跨依赖强行重叠。

### 6.2 内存与同步正确性

在 CPU 可访问的共享 Metal 存储上构造支持的 MLMultiArray 输入/输出 backing，复用包装对象和 session，减少调用端拷贝。但是共享物理内存与 caller-backed 数组 **不保证 Core ML 内部零拷贝**；布局转换、内部 staging 与隐式内存须通过诊断和测量评估。

生产者 GPU 完成事件 → host 调度 Core ML prediction → prediction 完成 → 消费者 GPU 提交/解锁，是正确依赖链。Core ML 不暴露与 Metal 完全同构的通用 command queue；不能把它写成可以直接等待任意 Metal event 的设备接口。

Buffer 带最后写入 token、读者计数和 owner。CPU/Core ML 写入未完成前不能给 GPU 使用；GPU 尚在读时不能复用 ANE backing。首版每个 session 串行请求；跨请求重叠要先做 workspace 分离和正确性验证。

### 6.3 设备放置的诚实语义

`.cpuAndNeuralEngine` 排除了 GPU，但仍允许 CPU，并不保证每一层都在 ANE。`MLComputePlan` 给出预计设备和成本，不是每次执行的硬件使用记录。应保存配置、预计放置与可获取的观测证据，无法确认时报告未知。[Apple compute units](https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine)、[Apple compute plan](https://developer.apple.com/documentation/coreml/mlcomputeplan-85vdw)。

API 可暴露 auto / gpu / validatedHybrid。显式 validatedHybrid 缺少已验证 artifact 或形状不匹配时，在执行前失败；auto 可退回 GPU并报告理由。不要提供实际上无法保证的“100% ANE”承诺。

运行中 fallback 必须具备恢复点：保留该子图只读输入，等所有在途写入结束，再在 GPU 重算并覆盖完整输出；对已经更新 latent/RNG 的步骤不能简单重试。首版不支持的恢复场景明确失败，避免静默生成错误结果。

私有 `_ANE*` 路径只放到单独的 research target，默认不编译、不链接到发行 App。这个隔离源于已有代码使用私有符号及版本耦合，不是为了牺牲已验证的公开 Core ML 支持。

## 7. 内存与模型生命周期是中心能力

内存计划按执行阶段计算：

```text
峰值需求 ≈ max_t(常驻权重(t) + 活跃激活(t) + workspace(t)
                  + 重叠分支/预取(t) + Core ML/框架隐式资源(t))
           + 操作系统与应用余量
```

不能把 GPU 分配器统计视为全进程或整机峰值，Core ML 模型和映射文件也不能重复或漏计。记录已知分配、进程 footprint、系统压力，并为未知资源预留经验余量。

内存管理分为不可变权重区、请求临时 arena、跨步有效缓存、预览/媒体缓冲区。规划 buffer 的最后使用点，复用不重叠区间；GPU 未完成的内存绝不提前释放/别名复用。shared/private/heap 由实际访问决定，并非所有 buffer 都强制 shared。

大模型按组件调度：text encoder → conditioning 保留 → 释放或降驻 encoder → DiT → 保留 latent → 卸载 DiT → VAE/audio。必要时保留 H3 的权重流式预取，但真实测试 SSD 吞吐、页缓存和读写竞争。mmap 不等于无限 RAM。

“常驻会话”意味着可复用身份与资源，有预算的权重和编译缓存；不能要求 H3、LTX、FLUX 三个完整模型同时常驻。Service 先按预测峰值准入，默认一个重型生成任务；轻量预处理仅在预算允许时重叠。

内存压力时先驱逐无活动引用的缓存、减少预取，再在合法边界换计划。禁止在用户不知情时减少帧数、上下文、步数或精度。热状态变化可降并发/暂停准入，不承诺精确功耗测量在所有系统都可用。

## 8. 三个模型模块的职责

| 模块 | 必须保留的模型语义 | 最关键的迁移风险 |
|---|---|---|
| H3 | 文本/视觉/音频条件、first/last 与有序 references 规则、FL2VA/Ref2VA、video/audio timestep、Turbo LoRA 身份和 flow shift、refiner 与 decoder | 把不同 Turbo 权重/schedule 混成一个“4步”；以跳层/复用收益冒充异构收益；破坏长权重/会话生命周期 |
| LTX | Gemma+connector、音视频双流及跨模态 attention、原始 8+3 schedule、中间 latent ×2 upsample、两阶段 first-frame 条件、VAE/vocoder/同步 | 把单流抽象强加到 AV block；遗漏文本和封装耗时；静默改变 704×480 为 704×448 |
| FLUX | tokenizer/text encoder、图像 latent packing、dual/single block、RoPE、flow schedule、img2img strength、有序多参考编辑 | 只移植 single-block MLP 而漏掉完整生成；动态 token 被错误 padding/mask；依赖 mflux patch |

只共享算子、张量、内存、设备、缓存与媒体基础设施；不把三个模型强行改成一个万能 TransformerBlock。不同模型 latent 不可直接互换；跨模型链路默认使用有明确色彩/时间信息的媒体资产，latent 接口必须携带模型/VAE/schema 身份。

分辨率在 `prepare` 返回 requestedShape、modelShape、decodedShape、exportShape。尺寸不能满足时明确 reject 或采用用户选择的 crop/pad/resize；后处理规则进入 result provenance。

## 9. 模型包、转换与缓存

模型包至少声明：architecture/schema version、源 revision 与 hash、组件/Tokenizer/VAE、量化方案及 group/scale/packing、输入角色和互斥规则、形状约束、采样 recipe、支持的后端、许可证信息、转换工具版本及质量证据。

保留 safetensors 作为可追溯源；设备专用 packed weights 与 Core ML artifact 是可再生产物。Core ML 不在每次生成时转换，模型准备阶段完成 export/compile/warmup，展示独立进度。

缓存 key 至少包含权重/adapter 身份、模型结构、算子/编译版本、精度/量化、layout、shape bucket、block 覆盖和设备/OS 兼容信息。`M=1088` 相同不代表两个 artifact ABI 相同。prompt cache 还包含 tokenizer/encoder 和预处理身份；不能只按 prompt 字符串缓存。

下载/转换需支持断点、hash 验证、磁盘预算、原子完成、读写锁和可回收失败产物；多进程不能同时写同一缓存。编译目录尽量保持稳定路径，但 OS 更新后的失效/重编译必须可恢复，不承诺 `.mlmodelc` 永久跨系统复用。

Python 转换工具可作为开发者工具单独发行；普通用户优先下载与模型包匹配的转换产物。若某模型仍要求运行时 Python，应标记为迁移中的兼容路线，不能宣称其原生迁移已完成。

## 10. App、CLI、SDK 与完整后端

### 10.1 统一生成契约

请求分成有类型的多模态输入、输出规格、采样 recipe、质量策略、执行策略和资源预算。图像/视频/音频都有 role、顺序和必要的时间信息。共通字段以外提供版本化 modelOptions schema，代替任意环境变量或原生 flag 透传。

建议逻辑调用为：`capabilities → prepare → submit → events → result/cancel`。prepare 返回计划解释、实际形状、设备路径、资源估算、编译需求和近似说明；估算允许未知值及置信范围。

服务 API 保留版本化的 models/assets/plans/jobs/events/results；提交支持 idempotency key，事件具有递增序号和断线续订，日志与结构化事件分离。预览事件只携带小媒体 asset 引用，禁止把大张量反复序列化到 JSON/SSE。

### 10.2 任务可靠性

SQLite 持久化任务/状态/结果索引，媒体放独立文件。状态至少含 queued、preparing、running、finalizing、succeeded、failed、cancelled/interrupted。客户端断线不改变任务状态。

取消在已提交 GPU/Core ML 工作的安全边界生效，不声称任意 kernel 都可中途抢占；进度流需能表达“取消请求已接受，正在收尾”。Host 崩溃后 Service 回收租约、标记 interrupted 并清理临时资源。首版先保证状态准确，恢复执行只针对显式持久化 latent+RNG+recipe 的 checkpoint 实现。

输出先写临时文件、完成后原子提交；任务成功需验证媒体尺寸/帧数/音轨/可解码性。原始 latent、tensor 和最终媒体可选保留，不能因 App 崩溃丢失已完成输出。

### 10.3 原生产品与发行

SwiftUI 提供创作、模型库、队列、结果库和诊断，控件来自模型能力；默认显示实际输出尺寸、准备/生成/解码阶段、预览及有依据的 ETA。未知 ETA 不伪装成精确倒计时。高级选项可查看执行计划和量化/近似，不让普通生成流程依赖环境变量。

媒体层使用 CoreVideo/Metal texture 处理帧、AVFoundation/VideoToolbox 编码和封装、原生音频处理接口处理 PCM。逐步移除 FFmpeg 子进程依赖，但先用参考样例固定色彩空间、范围、resize、FPS/时间戳和音量规则；原生替换不等于输出自动一致。

发布包包含原生二进制、metallib 和资源，模型与可写缓存位于用户数据目录。签名、公证、库依赖、CLI 安装、服务注册/升级及旧任务兼容纳入发行测试。App Store sandbox 与直接发行的 helper 生命周期不同，建议先把直接发行路径做完整，再单独验证 App Store 版本。

HTTP 默认 loopback 且带客户端令牌；仅 loopback 不是身份验证。文件通过受控 asset ID/授权目录访问，远程监听显式启用。这里是完整本地服务的实现边界，不增加云依赖。

## 11. 性能、数值与质量验收

新引擎的价值通过与参考实现的同条件验证证明，重写本身不保证更快。

| 层次 | 必测内容 |
|---|---|
| Tensor/算子 | dtype/layout/stride/别名、shape 边界、溢出、量化 scale、绝对/相对误差、NaN/Inf、同步正确性 |
| Transformer block | 固定真实权重/输入；每个关键中间张量；GPU reference、optimized GPU、ANE 子图配对 |
| 完整生成 | 固定 tokenizer、prompt、初始 latent、RNG 算法、schedule、checkpoint；图像细节、视频时序/闪烁、音频幅值/同步、人工检查 |
| 运行时 | 连续请求内存是否稳定、切模型/shape、取消、失败回收、事件重连、缓存失效、服务重启 |
| 发行 | 干净机器不依赖源码目录或 Python；App/CLI/SDK 使用同一结果/任务语义 |

质量策略必须拆开：设备放置、weight/activation 精度、算法近似、输出 recipe。INT8、稀疏 attention、text pruning、跳层和减步都不是同一个维度。即便无算法近似，跨后端浮点舍入也可能导致扩散轨迹差异；seed 相同不等于逐像素相同。

性能报告拆成首次安装/编译、冷进程、热 session、conditioning、DiT、upsample、VAE、audio、encode、排队和完整请求墙钟。冷热缓存、设备/OS、内存、热状态、预览开关、sample 数及中位数/尾延迟一起保存。额外报告全进程 footprint 与系统压力，不能只报 MLX 或 Metal 已知分配。

每个模型建立多 prompt/seed/shape 测试集；像素 PSNR/SSIM 只作辅助，不能替代运动、人物、文字和音频的质量验收。定量阈值在基线阶段确定，不事后为了通过而调整。

ANE 进入 auto 候选必须同时满足：支持矩阵、公开 API、正确性、质量、完整链路收益或明确的节能目标、内存预算和重复运行稳定性。在新 GPU 上重测，与该设备当前最优 GPU 比较。

## 12. 推荐源码结构

```text
apps/macos/                 SwiftUI 产品
service/                    Swift 任务/资产/事件/XPC host 管理
cli/                        命令行客户端与显式 local 模式
include/turbocider/          版本化 C ABI
sdk/swift/                  嵌入 Engine + 服务 Client
sdk/python/                 服务 Client；可选原生绑定
core/tensor/                存储、视图、同步依赖
core/runtime/               session、executor、资源生命周期
core/graph/                 语义图与执行计划
core/planner/               shape、layout、设备与内存规划
core/weights/               safetensors、packed weights、流式加载
backends/metal/             MPS/MPSGraph、MPP、自有 kernels
backends/coreml/            model/session/backing/placement
backends/cpu/               参考算子、Accelerate
backends/mlx/               受限的过渡子图后端
models/h3/                  模型图、conditioning、recipe
models/ltx/                 AV 双流和两阶段 recipe
models/flux/                图像生成/编辑 recipe
media/                      image、video、audio、native export
packages/                   模型 manifest 与支持矩阵
tools/modelprep/            转换/量化/ANE export，允许 Python
tests/                      算子、block、pipeline、生命周期、发行
benchmarks/                 固定 workload、执行结果与对照工具
research/                   私有 ANE/未验收算法；独立构建目标
```

核心构建使用 CMake，Swift 使用 SwiftPM/Xcode，CI 产生库、Host、CLI 和 App。采用一个主仓库维护统一 runtime，参考仓库以固定 revision/外部路径纳入验证；避免长期同时维护三份共享内核副本。

## 13. 迁移顺序与阶段退出条件

| 阶段 | 工作 | 完成条件 |
|---|---|---|
| P0：固定契约与基线 | 冻结三个参考 revision、目标 checkpoint、硬件/OS 矩阵；保存 tensor fixtures 和完整媒体基线；定义 ABI/模型包/质量策略 | 同一 workload 能复跑；所有计时口径、实际尺寸和近似均明确；原始仓库只作参考 |
| P1：原生最小纵切 | Engine/Storage/Session/命令执行/权重读取；先迁入 H3、LTX 可复用基础算子并做 block 验证；完成 FLUX Klein 4B text-to-image 全链路与最薄 CLI | 已准备模型上无 Python 生成图片；数值、真实图像与稳定内存通过；不是只完成 mock 或 DiT |
| P2：统一异构机制 | FLUX 已验证 MLP 子图、Core ML backing/session、bucket/成本/失败处理；模型资产准备流程 | 冷热对照、质量、取消与 buffer 复用通过；收益不足时 auto 选 GPU |
| P3：LTX 原生全链路 | conditioning、connector、AV DiT、8+3、upsample、VAE/vocoder、媒体同步；必要的 MLX C++ 过渡子图 | 原生生成可播放有声视频；尺寸策略明确；计时包含文本和封装；GPU 基线先通过再迁 ANE |
| P4：H3 完整迁入 | 模型图与多模态输入、Turbo provenance、常驻/流式加载、预览、完整解码 | 指定 checkpoint 的输入模式与完整音视频通过；GPU 优化保留；ANE 不阻塞完成 |
| P5：产品化与发行 | 常驻服务、App、完整 CLI/SDK、缓存/模型管理、故障恢复、干净机器安装验证 | 三个模型由同一 runtime 执行；接口一致、进程生命周期可靠，运行时无开发目录依赖 |

最薄 App/Service 联调应在 P1/P2 就开始，以及时暴露取消、进度、媒体和进程边界；P5 是完善与发行，不是到最后才首次集成 UI。img2img/多参考编辑等模型附加模式在对应阶段逐项验收，首个 FLUX text-to-image 切片不代表整个 FLUX 模块完成。

选择 FLUX 作为首个完整纵切是因为它能较快覆盖 tokenizer→encoder→DiT→VAE，并已有清晰的异构分支；并不意味着它现有代码最原生。先移入 H3/LTX 已验证的基础组件，避免从 Python 重新发明全部 kernel。如果实测 tokenizer/encoder 迁移成本阻塞，应保持核心契约不变，调整切片规模，不能以保留 Python 热路径代替原生验收。

每一阶段同时比较旧实现和新实现，替换一个层次后才推进下一个。固定金标准来自参考实现，最终模型支持声明来自新引擎自身通过的测试。工程量要根据 P0 的组件依赖和 P1 的端到端迁移结果评估；在未完成 profiling 和依赖盘点前不承诺统一倍速或精确交付周数。

## 14. 第一批应写下的架构决策

1. 核心 C++/ObjC++/Metal，公开 C ABI；Swift 做服务与产品。
2. 发行推理不依赖 Python；转换和校准工具允许 Python。
3. GPU 是完整功能基线；Core ML ANE 是按模型/shape/设备验收的子图后端。
4. 有限模型语义图与显式执行计划先行，暂不建设通用模型编译器。
5. 核心统一张量与资源所有权；不透明框架以有预算的子图边界接入。
6. 同一核心同时支持嵌入和服务；逐算子执行不跨进程。
7. 设备选择、精度、近似和输出 recipe 独立；所有结果保存实际计划。
8. 主产品对三个具体模型包负责，不以家族识别代替支持验收。

## 15. 阅读依据

以下本地材料用于本提案；性能数字均引用已有记录，未在本轮重新测量。

- [现有 TurboCider 架构](/Users/kd/Documents/project/TurboCider/docs/ARCHITECTURE.md)
- [H3 公共 API](/Users/kd/Documents/project/h3.c-fork/h3.h)、[GPU 张量 API](/Users/kd/Documents/project/h3.c-fork/h3_gpu.h)
- [H3 ANE 实验结论](/Users/kd/Documents/project/h3.c-fork/notes/gpu-ane-h3-turbo-final-2026-09-01.md)
- [LTX 技术报告](/Users/kd/Documents/project/ltx-mac/notes/technical-report-2026-09-04.md)
- [FLUX 引擎说明](/Users/kd/Documents/project/mac_image_generation/flux2-engine/README.md)、[架构](/Users/kd/Documents/project/mac_image_generation/flux2-engine/docs/ARCHITECTURE.md)
- [FLUX 原生 Core ML bridge](/Users/kd/Documents/project/mac_image_generation/flux2-engine/native/flux2_ane_bridge.mm)、[当前 attention patch](/Users/kd/Documents/project/mac_image_generation/flux2-engine/src/flux2_engine/backends/hybrid_ops.py)
- [Apple Metal 4](https://developer.apple.com/videos/play/wwdc2025/205/)、[GPU TensorOps/MPP](https://developer.apple.com/videos/play/tech-talks/111432/)
- [Core ML compute units](https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandneuralengine)、[Core ML compute plan](https://developer.apple.com/documentation/coreml/mlcomputeplan-85vdw)
