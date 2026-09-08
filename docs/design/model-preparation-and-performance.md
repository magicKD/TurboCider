# 模型准备、GPU/ANE 管理与性能实现

> 后续修订：自动策略已从兼容性优先改为按任务匹配，见 [任务感知加速](task-aware-acceleration.md)。下文自动适配章节保留为历史实现说明，不再代表最新自动选择规则。
2026-09-06；以本次真实 FLUX 权重测试为准。已更新本机 App；默认策略后续升级为自动适配，详见下节。H3/LTX 不具备可运行的正式执行器，不因此开放按钮。没有下载模型或修改外部参考引擎。

## 自动适配补充实现

上一版 `auto` 只等同纯 GPU，App 不扫描已有分区，示例 profile 默认禁用，所以本机即使已有可用分区也必须手动选择。本次补齐自动发现与执行期适配。

- App 默认“自动适配本机 · 优先 GPU + ANE”。上一版无自动选项的默认 GPU 草稿会迁移一次；已有明确混合/设备 profile 保留，新版手动 GPU 配置在重启后保留。
- 仅检查已配置 manifest、`TURBOCIDER_ANE_MANIFEST`、App 自有缓存、当前模型下 `coreml/`，以及旧项目布局的已知相邻分区目录。不递归扫描用户主目录，不下载，不在 UI 线程加载 Core ML 模型。
- `AccelerationDiscovery.swift` 检查 schema、3072 隐藏维度、单固定桶、当前 checkpoint 路径与大小，以及全部 20 个编译目录；拒绝越界路径、缺块及不匹配权重。这是预检，实际 Core ML ABI 仍在加载时核对。
- 引擎在编码后按真实 token 数选择路线，覆盖生成图、文本和所有参考图。自动模式需要 `allow_approximation=true`；缺少许可或候选时使用 GPU。当前自动混合硬件策略只开放实测过的 Apple M4 Pro 48 GiB，其他机器回退 GPU；显式设备配置/手动混合入口保持可用。不能把所有 Apple Silicon 都视为同一性能配置。
- 超出桶、内存预算不足或分区加载失败时回退 GPU，返回 `acceleration_selection` 原因；显式 `gpu_ane` 仍严格报错。取消独立传播，不被误吞为回退。预测过程中失败不会自动重跑整次任务。
- 静态 plan 返回 `selection_pending=true`，实际 prepare/generate 报告返回最终路线；模型页就绪状态显示实际选中的 GPU 或 GPU + ANE。

自动发现发生在模型页检测以及生成/准备请求构造时；实际权重按加载、预热或生成操作载入，不在启动 App 时强占全部内存。自动模式会采用 INT8 近似计算；要保留原始 BF16 路线，可明确选择 GPU。

这版是兼容性策略，不是逐输入实测的自动调优器。固定桶 padding、冷加载以及小尺寸任务可能抵消混合收益；不能宣称自动模式在所有输入上都最快。后续性能策略应按设备、bucket 与操作类型记录实测阈值。没有兼容桶时绝不通过静默缩小图片来适配。

验证：真实权重 8 项测试通过，包含硬件/权重/桶匹配、混合会话复用、自动与显式混合 PNG 完全一致、超容量回退、缺失/损坏 manifest 回退、近似许可、显式模式严格失败。Swift 测试另覆盖本地发现、权重大小不匹配和缺少分区。App 真机已自动识别本机 1088 桶并加载 20 分区，显示“GPU + ANE”就绪。

复现：`python3 tests/native/test_auto_acceleration.py --model "$MODEL" --manifest "$COMPILED_MANIFEST" --output "$OUTPUT"`。这组测试用于选择与正确性，不是冷热交错性能基准。[归一化证据](validation/auto-acceleration-validation.json)

## App 操作与准确语义

缓存计数与 safetensors 转换已扩展为 [完整资源管理](coreml-artifacts-and-storage.md)。下表描述前一阶段接口；最新 App 统计会包含外部选中分区，且已新增离线导出。

模型页的“推理加速与准备”提供以下能力：

| 操作 | 实际工作与边界 |
|---|---|
| 加载当前配置 | 编码并缓存当前文本，materialize DiT/VAE 权重；混合模式还加载指定 Core ML 分区并检查 token 桶。不会生成图片。Z-Image/LLaDA 现在也走真正的 load-only 分支，不执行 latent denoise 或 VAE decode。文本编码器本身仍按阶段释放。 |
| 预热当前任务 | 按当前提示词、输入图、尺寸、步数、种子和计算模式完整执行一次，包含 DiT 与 VAE；跳过文件输出和任务历史。改变形状/条件后可能重新编译或准备。Core ML 分区可额外设置 `warmup_iterations` 做零输入预测预热。 |
| 卸载模型 | 安全边界同步，释放权重、conditioning、Core ML 会话及 MLX allocator cache。不会删除源文件，也不宣称 OS 页缓存归零。 |
| GPU | 原始 BF16 路线，可显式启用单流块编译融合，默认关闭。 |
| GPU + ANE | GPU attention 与 Core ML INT8 MLP 分支并发；用户必须提供适配的分区 manifest。量化属于近似，不承诺与 BF16 位级一致。 |
| 配置文件 | 保留硬件匹配配置文件入口；选择配置文件会切换到配置策略，恢复默认会回到 GPU。 |
| 预编译全部分区 | 从已有源 manifest 编译 20 个 `.mlpackage` 分区，生成 App 自有编译 manifest。不是任意模型自动切分、导出或量化。 |
| 查看/清除缓存 | 统计、清理本 App 管理的 Core ML 编译副本；清理前卸载会话，保留源分区、外部分区和系统缓存。 |

操作与生成互斥；耗时任务提供进度与取消。Core ML 单次同步编译不能中途打断，取消在分区之间生效。预编译成功后仍由用户选择混合模式，避免自动改变数值路线。界面自动保存加速配置，旧草稿/历史兼容缺省字段。

当前混合分区固定为 1088 token；检查的是文本、生成图与所有参考图的总 token，不能将文生图性能直接套到多图编辑。`cpuAndNeuralEngine` 是允许设备集合，不能据此证明每个算子实际在 ANE 运行；需另行使用系统分析工具核实。[Apple compute units](https://developer.apple.com/documentation/coreml/mlcomputeunits)

## 模块与接口

| 文件 | 职责 |
|---|---|
| `apps/macos/AccelerationView.swift` | 模式、分区文件选择、加载、预热、卸载与缓存操作 |
| `apps/macos/StudioState.swift` | 可持久化加速配置、请求转换与历史兼容 |
| `apps/macos/JobStore.swift` | 异步操作与互斥、进度、取消；准备操作身份过滤过期回调 |
| `bindings/swift/TurboCiderNative.swift` | async prepare/cache，C 字符串和回调所有权 |
| `bindings/c/include/turbocider/turbocider.h` | `tc_engine_prepare` 与 `tc_engine_cache` 公共 C ABI |
| `native/core/api.mm` | 同一会话、进程和跨进程设备准入；退出时同步 |
| `native/models/flux.mm` | conditioning 复用、实际权重载入、完整无输出预热 |
| `native/models/flux_transformer.mm` | 可选单流块 MLX compile，保留块间取消边界 |
| `native/backends/artifact_cache.mm` | 内容寻址编译缓存与 manifest 生命周期 |

`tc_engine_load` 保留其原有“仅图像权重载入”契约；App 使用更完整的 `tc_engine_prepare(request, warmup)`。`warmup=0` 是 load-only 准备，`warmup=1` 执行无输出完整预热，后者返回 `output:null`。请求支持 `compile_gpu`，v2 对应 `parameters.compile_gpu`；Core ML 的 `warmup_iterations` 位于 schema 1 顶层或 schema 2 的 `execution` 对象，范围为 0–8。Swift 暴露相同能力，其他语言通过 C ABI 接入即可。

Core ML 结果中的 `hybrid` 遥测会拆分 `manifest_validation_seconds`、`output_backing_setup_seconds`、`model_load_seconds`、`model_interface_setup_seconds`、`zero_input_warmup_seconds`、`first_runtime_prediction_seconds_session_total` 和 `subsequent_runtime_prediction_seconds_session_total`。`load_seconds` 仍表示整个 `HybridSession` 构造/预热阶段，`prediction_seconds_session_total` 包含所有预测；因此不能把总 load 时间误认为 Core ML 编译时间。`.mlmodelc` 编译发生在独立的 `coreml compile`/`compile_manifest` 管理操作中，命中内容寻址缓存时不会再次编译。

缓存请求形如 `{"action":"compile_manifest","cache":"/absolute/app-cache","source":"/absolute/source/manifest.json"}`，另有 `inspect`、`clear`。路径是调用方提供的本地配置，不写死本机目录。

## 缓存与内存策略

App 缓存位于 Application Support 下 `TurboCiderNative/cache/coreml`。编译键包含源内容 SHA、系统 build、GPU 架构与编译器身份；每项使用文件锁、原子提交并在编译后复核源内容。批量 manifest 使用相对编译路径。管理入口检查所有权标记、路径和符号链接，只删除识别到的编译项，保留用户未知文件。空缓存可以直接清理；非空且无所有权标记的目录拒绝清除。

这不是系统 ANE 全局缓存管理器；Core ML 内部可能还进行设备相关准备，因此“已编译”“已加载”和“已预热”是三层状态。已编译只表示 `.mlmodelc` 产物存在或被缓存；`prepare(load-only)` 会加载模型和 backing，但不执行预测；`warmup_iterations>0` 才会提前触发零输入预测。旧源 manifest 的权重 provenance 只有路径/大小，不能把新增的分区内容 SHA 误称为完整源权重 SHA 证明。

修复了预加载后首次编码文本无条件释放图像权重的问题：已有驻留权重且物理内存至少 32 GiB、请求预算未设置或至少 24 GiB 时，编码新提示词保留图像权重及混合会话；较低内存仍采用阶段释放。这是保守启发式，不是完整的内存压力预测器。`component_staged` 仍在阶段结束释放权重，不能同时宣称其具有完整驻留预热收益。

## 为什么 App 需要约 6 秒

读取本机最近任务的数值遥测，未保存用户提示词或图片内容：512×512 文生图 App wall 5.399 秒，引擎 5.386 秒，其中去噪 4.515 秒、文本编码 0.383 秒、VAE 0.473 秒。App 与引擎差约 13 ms。另一个单参考图暖请求 App 6.588 秒、引擎 6.571 秒，其中去噪 5.994 秒；参考图增加序列长度与注意力计算。

这些是生成请求耗时，不是 App 进程冷启动测试。它们支持“主要耗时在 DiT”，不支持“所有情况下 UI 开销都是 13 ms”。加载和预热前移准备成本，无法消除每次采样的计算。

## MLX 编译实验与语言选择

现有推理已使用 MLX C++ 与 Metal，没有 Python 推理进程。MLX 构建 lazy 计算图，在 eval/synchronize 边界执行；默认块路径可称为未整块 compile，不能理解为每个 Python 算子立即启动 GPU。原先已有 SiLU/Euler 等编译以及混合 attention 编译。本次增加纯 GPU 的 20 个单流块编译；5 个双流块仍沿用原路径，权重作为编译函数参数传入以避免捕获旧会话。

M4 Pro 48 GiB、MLX 0.32.0、512×512、4 步、seed 42、动态文本、纯 GPU BF16：

| 路线 | 暖请求中位数，含导出 |
|---|---:|
| 原块路径 | 4.6845 秒 |
| 单流块 compile | 4.5967 秒 |

约 1.87% 改善。每组 5 次，排除首个请求，先原路径后 compile；这是探索性单轮测试，未随机交错，不能视为所有设备和输入均有稳定 2% 收益。文生图 22、多图编辑 26、图生图 21 组张量均完全一致。保留显式开关，暂不默认开启。完整样本见 [性能证据](validation/gpu-block-compile-performance.json)。形状/dtype 变化可能重新编译；避免把首次编译成本混入稳定运行比较。[MLX compile](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.compile.html)

建议继续采用 SwiftUI 产品层、C ABI 边界、C++20 模型与调度、Metal 热点内核、薄 Objective-C++ Apple API 桥接。`.mm` 可以直接编译 C++，扩展名本身不表示热路径使用 Objective-C 消息。把同一计算从 Objective-C++ 改写成 C，不会自动加速 GPU 矩阵运算。MLX 本身提供 C++ API。[MLX](https://mlx-framework.org/)

不能宣称当前已达最优。后续按收益优先级推进：

1. 对 DiT attention/MLP、GPU/ANE 重叠、同步和内存带宽做设备 trace；以稳定同条件样本确定真实关键路径。
2. 为实际输入分布增加混合分区桶，减少 padding；在峰值内存允许时复用已加载会话及工作区。量化路线单独做质量验收。
3. 比较双流块融合、自有 Metal 热点、提交批次与 eval 边界，兼顾峰值内存和取消延迟；不能仅凭更大的 compile 图决定优劣。
4. 基于真实内存压力完善准备/驻留/阶段释放选择，保持低内存机可用；不把 offloading 当成免费加速。

此前同分工原引擎比较见 [FLUX 性能对比](flux-performance-comparison.md)，不能用本轮纯 GPU compile 数字替代同条件混合路线回归。

## 验证与复现

`tests/native/test_preparation.py --model "$MODEL" --source "$PARTITION_MANIFEST" --output "$OUTPUT"` 使用真实权重与已有源分区：加载、完整无输出预热、文本缓存复用、20 分区编译/命中、真实混合预测与会话复用、仅清理自有缓存、释放内存和取消，共 10 项通过。[证据](validation/model-preparation-validation.json)

`make test-model MODEL="$MODEL" OUTPUT="$OUTPUT"` 已通过 Studio 三操作/直接引擎 PNG 一致性、生命周期、服务、取消恢复与设备互斥回归。App 真机界面验证了配置加载、完整预热、卸载、空缓存清理和 GPU/ANE 选择；非空编译缓存的清理及 20 分区预测由上述 C ABI 集成测试覆盖。未把 GUI 菜单检查冒充完整 GUI 端到端混合生成测试。

数值证据：[文生图](validation/gpu-block-compile-parity.json)、[多图编辑](validation/gpu-block-compile-multi-parity.json)、[图生图](validation/gpu-block-compile-transform-parity.json)。原始模型、图像、张量及过程日志仅保留本机 `outputs/preparation-validation/`。发布摘要已归一化本机路径，数值不改写。
