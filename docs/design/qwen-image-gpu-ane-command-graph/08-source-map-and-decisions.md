# 08 · 来源映射、设计决策与审查清单

本章是可追溯入口。文档中的架构/代码草案由本项目设计，不应当作上游原样实现。引用固定版本，避免 PR 页面或 main 后续变更导致结论漂移。

## 1. Splash 来源

主版本：`80827e1c3593f7d10c266394e4d4381321b73ad0`，见 [PR #260](https://github.com/incoai/splash/pull/260)。本设计以该版本实现为依据，PR 中的 LLM 性能不转写为本机 Qwen Image 性能。

| 源文件 | 本设计参考内容 | 保留的差异 |
|---|---|---|
| [CommandGraph.hpp](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/metal/CommandGraph.hpp) | dispatch、参数生命周期、event step | TurboCider 还需模型语义/资源/事务计划 |
| [MetalBackend.mm](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/metal/MetalBackend.mm) | signal 分 CB、异步提交、完成回调 | N 接入统一内存/事务；A 的 MLX 适配合同另证 |
| [MetalBackend.hpp](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/metal/MetalBackend.hpp) | 原生 device/buffer/dispatch/ticket 边界 | N 热路径不以 MLX Tensor 为执行接口 |
| [Linear.cpp](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/ops/Linear.cpp) | 格式、shape、epilogue 的 kernel 选择 | 逐项核对 dense BF16 覆盖，不继承 Q4/GGUF 跑分 |
| [PagedAttention.cpp](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/ops/PagedAttention.cpp) | attention workspace、K/V 与 dispatch 组织 | 必须适配 Qwen Image segments / prefix / target 数学 |
| [ane_ffn.metal](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/metal/kernels/prefill/ane_ffn.metal) | 原生量化/准备/合并 kernel 组织 | BF16 来源、basis 和舍入仍绑定本地 recipe |
| [AneFfn.cpp](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/ops/AneFfn.cpp) | 多行数 function、binding、双 bank、编码/提交 | Qwen LoRA 修正与首步语义不在该 FFN API 中 |
| [QwenTarget.cpp](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/model/QwenTarget.cpp) | 模型层 mixer/FFN 全部编码进共同 graph；split 的实际 caller | LLM attention/RMSNorm 不等于 Image DiT 的 mask/LayerNorm/modulation |
| [Runtime.cpp](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/model/Runtime.cpp) | 整段提交、暂存结果、失败回放与计时 | 换为 DiT latent/sigma/prefix 事务，不复制 LLM KV/采样回滚 |
| [Program.mm](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/ane/Program.mm) | ABI 检查、load、binding 与 request 分离 | 保留本地 source/cache 身份和能力隔离 |
| [Handoff.mm](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/ane/Handoff.mm) | timeline、ready 后 deadline、一次结算 | 扩展前要证明晚到访问及资源隔离 |
| [AneFfnCalibration.cpp](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/ops/AneFfnCalibration.cpp) | 两份额拟合、邻居稳健选择、逐 rows 收益 | 本地真实输入、LoRA、attention 重新测量 |
| [AneFfnMeasurement.cpp](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/ops/AneFfnMeasurement.cpp) | 提交固定开销与完整窗口 | 不把 synthetic/mixer stand-in 当 DiT E2E |
| [AneFfnStartup.cpp](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/engine/AneFfnStartup.cpp) | 准备、内存约束和校准生命周期 | 不照搬 LLM KV 内存扣除方式 |
| [RuntimeResources.mm](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/engine/RuntimeResources.mm) | calibration key、idle release/reload | 质量/LoRA 资格仍须绑定本地实际来源 |

注意 Startup 与 RuntimeResources 位于 `runtime/engine/`。引用“同名文件”时必须使用此处已核对路径。

## 2. vpipe 来源

固定版本：`3e2682a0847c9e76dd37978f2be01f53195ae5e3`。

2026-10-08 本次只读核验：该上游提交日期为 2026-10-07，GitHub commit／完整 tree 与下表六个源码路径均确认存在，并重新阅读 runtime LoRA、ANE FFN 和 Qwen21 caller。**本机 vpipe checkout 是较旧的 `40706e01583b8087cfbb07c148974b35aa4fa9c1`（2026-09-06），不是此设计 pin**，不能拿本地旧路径／实现代替这些固定链接；核验未 fetch／修改外部仓库或下载模型。

| 源文件 | 参考内容 | 设计边界 |
|---|---|---|
| [command-stream.cc](https://github.com/tgo-app-dev/vpipe/blob/3e2682a0847c9e76dd37978f2be01f53195ae5e3/apple-silicon/metal-compute/command-stream.cc) | 原生命令批处理和提交 | 不直接接管 MLX 私有队列 |
| [runtime-lora.h](https://github.com/tgo-app-dev/vpipe/blob/3e2682a0847c9e76dd37978f2be01f53195ae5e3/generative-models/shared/runtime-lora.h) | rank scratch、factor 布局、应用接口 | 低秩中间精度与本地不同 |
| [runtime-lora.cc](https://github.com/tgo-app-dev/vpipe/blob/3e2682a0847c9e76dd37978f2be01f53195ae5e3/generative-models/shared/runtime-lora.cc) | factor 准备、按形状调度、有效 rank | 不填零扩张 LoRA，也不强制所有投影融合 |
| [ane-ffn.cc](https://github.com/tgo-app-dev/vpipe/blob/3e2682a0847c9e76dd37978f2be01f53195ae5e3/generative-models/shared/ane-ffn.cc) | Core ML worker、row split、staging LoRA merge | 独立的 FP16 merged-weight recipe |
| [metal-qwen-image21-transformer.cc](https://github.com/tgo-app-dev/vpipe/blob/3e2682a0847c9e76dd37978f2be01f53195ae5e3/generative-models/qwen-image/metal-qwen-image21-transformer.cc) | Qwen 对 shared LoRA/ANE 的实际接线 | 不仅依据通用库推断模型已用某项 fusion |
| [block-residency.h](https://github.com/tgo-app-dev/vpipe/blob/3e2682a0847c9e76dd37978f2be01f53195ae5e3/generative-models/shared/block-residency.h) | 固定 pinned 子集、bounded refill、压力策略 | 进入本地 memory ledger 后再复用，不另建私账 |

Qwen 源码目录是 `generative-models/qwen-image/`，不是 `qwen-image21/`。

## 3. MLX 来源与适配假设

- 本地 [device.h](../../../.venv/lib/python3.11/site-packages/mlx/include/mlx/backend/metal/device.h)：当前环境中存在 `CommandEncoder::signal_event/wait_event/commit`。该链接依赖本地 .venv，其他 checkout 应以同版本依赖头文件为准。
- 上游 [v0.32.0 device.cpp](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/device.cpp)：event 编码、commit、错误传播、临时对象管理。
- 上游 [v0.32.0 event.h](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/event.h)：EventImpl 自己持有 SharedEvent；不能随意强转另一对象。
- 本次另只读核对本地 MLX source `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`：[compile.h](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/compile.h)、[metal.h](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/metal.h)、[device.h](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/device.h)。分别是函数编译、GPU trace capture，以及 stream encoder 的 wait／signal／commit／command-buffer 接口；不代表可接管全部 MLX 图。source checkout 与实际链接库的 build identity 分开记录，不能用源码版本替代当前库身份。
- 本地 [GPU 时间 probe](../../../tools/native/z_image_gpu_time_probe.cpp)：CaptureBuffer 自定义 Primitive 是诊断示例，不是生产外部执行认证。

MLX 内部 ABI 的稳定性不由这些接口存在来保证。升级时应重新运行适配合同测试；能力不满足时保留原生 FFN 或旧 GPU 路线。

## 4. TurboCider 来源映射

| 文件 / 符号 | 需要复用或改造的部分 |
|---|---|
| [ane_ffn.cpp](../../../native/backends/ane_ffn.cpp) `run_channels` | 输入 eval、异步 head、host finish、down-LoRA/join、fallback |
| [ane_gpu.mm](../../../native/backends/ane_gpu.mm) | GPU staging、共享事件、scale/PSO cache、Surface |
| [ane_w8_executor.mm](../../../native/backends/private/ane_w8_executor.mm) | 双 bank、A8 finish、driver request、输出还原与 overflow |
| [ane_program.mm](../../../native/backends/private/ane_program.mm) | ABI 检查、缓存、prepare/enqueue、异步持有 |
| [ane_mil.cpp](../../../native/backends/private/ane_mil.cpp) | W8A8、LoRA correction ABI、输出布局 |
| [ane_cost_model.hpp](../../../native/backends/ane_cost_model.hpp) | 保守耦合模型、measured-range 选择 |
| [ane_channel_calibration.cpp](../../../native/backends/ane_channel_calibration.cpp) | 多深度、两份额和独立 candidate trial |
| [ane_channel_selection.hpp](../../../native/backends/ane_channel_selection.hpp) | bounded 进程内校准 cache、source owner 失效 |
| [qwen21/transformer.cpp](../../../native/models/qwen21/transformer.cpp) | attention segments、prefix、末层、compiled block、MLP 回调 |
| [qwen21/pipeline.cpp](../../../native/models/qwen21/pipeline.cpp) | 生命周期、step、source、条件缓存、输出提交 |
| [runtime_ffn_graphs.hpp](../../../native/models/qwen21/runtime_ffn_graphs.hpp) | 普通 FFN、通道、correction、down 图 |
| [runtime_activation.hpp](../../../native/models/qwen21/runtime_activation.hpp) | split/图持有历史防护，需生命周期回归 |
| [mlx.cpp](../../../native/backends/mlx.cpp) | LoRAWorkspace、project、ordered LoRA、FP32 rank、SDPA |
| [memory_execution.hpp](../../../native/runtime/memory_execution.hpp) | 请求 admission、drain/quarantine、状态 |
| [memory_scheduler.hpp](../../../native/runtime/memory_scheduler.hpp) | storage/completion/访问阶段 |
| [memory_plan.hpp](../../../native/runtime/memory_plan.hpp) | 图临时、阶段预算、异步区间 |
| [source_lease.hpp](../../../native/runtime/streaming/source_lease.hpp) | 文件身份、generation、fd 所有权 |
| [H3 GPU](../../../native/models/h3_runtime/h3_gpu.m) | 自有 Metal continue/completion 经验 |
| [LTX GPU](../../../native/models/ltx_runtime/ltx_gpu.m) | batch/drain、pending memory token |
| [RuntimeImageWorker.swift](../../../apps/macos/RuntimeImageWorker.swift) | 当前 profile、env、request identity |
| [JobStore.swift](../../../apps/macos/JobStore.swift) | 单任务 worker、退出检查、App 状态 |
| [ModelLibraryController.swift](../../../apps/macos/ModelLibraryController.swift) | 模型/LoRA/ANE 登记与链接 |
| [TensorCacheController.swift](../../../apps/macos/TensorCacheController.swift) | cache inventory、保留和清理入口 |

## 5. 阅读快照

Git HEAD：`8d12b44cc764d079ee31c6092725e22fcc8fb9e0`，分支 `dev-verify`。上述文件包含阅读时工作树改动，不能用 HEAD 单独复现。下列是关键文件在设计阅读时的 SHA-256，不表示它们已经提交：

```text
native/backends/ane_ffn.cpp
77ffa7e1f939e313c07b5d3e9cc9084b6dbf42cf3cbde37013abafdc68cc9f98
native/backends/ane_gpu.mm
dec2326f829b604006c389058c341b3d71533703edc2559c0c82a4e4ee91a1cb
native/backends/private/ane_program.mm
38897afa731046c67f155e2216b2e7187da5b77795b9d48fb846c3696275ceb4
native/backends/private/ane_w8_executor.mm
c5aa55a5249898025b9f74e9ec3f54a42516edd425c272d4ae4d602b3b651316
native/models/qwen21/runtime_ffn_graphs.hpp
9305a3800dc9c9829442e0e3b46ae507ab44a8c18f6ad7f9b9b4f84320523627
native/models/qwen21/transformer.cpp
84af5bde8b3bc74ad0a087d1de65136901092eb3a8b6e690089d126b01034a1c
native/models/qwen21/pipeline.cpp
cdd0b8f9b7e9c8963cbf6e0d4337a2d401209e07864760e5d2b5e061bc0cc3a6
apps/macos/RuntimeImageWorker.swift
c212c5136ca1413142fc0c88e57d40f39d85943217c6e3f0c72198b3dd5225cc
```

真实 FFN 报告的 E2E runtime build：
`tc-runtime-build-v1-ec25a1634a7abf42f4e29524c8873bef4c603468f0160060a546c10eac96246f`。
frozen 对照属于另一个 runtime build，分别读取其回执，不跨构建拼接速度结论。

用户提供的《TurboCider_ANE_Runtime_v2_Splash260_Design.md》《TurboCider ANE Runtime Backend 技术设计稿.md》作为背景输入；本目录对 command ownership、shared events、统一 memory/cache 等方向进一步落实到当前源码，并补充失败访问、LoRA 舍入和证据边界。背景文档中的建议不是已实现能力或额外执行授权。

## 6. 设计决策记录

| ID | 决定 | 理由 / 重新评估条件 |
|---|---|---|
| D01 | 采用应用拥有 plan/graph 的主线 | 能统一调度、内存和恢复，不再扩散阻塞回调 |
| D02 | A 路线验证 MLX adapter；N 后端独立推进 | A 保留现有算子；N 不依赖 A 成败，直接拥有 DiT command graph |
| D03 | 首版 GPU attention + channel FFN | 与本机已有有效路线接续；不同时扩大变量 |
| D04 | phase-specific row 策略 | 首步和稳态实际工作量不同 |
| D05 | BaseOnly 独立交付后才接入单 adapter；多 adapter 再单列 | 避免基础调度同时引入 LoRA ABI/舍入变量；不继承 base 资格 |
| D06 | 资源完成与结果成功分离 | timeout/人工 signal 不能证明资源空闲 |
| D07 | W1 可以是 B1 有效原型；B2 设备驱动目标需已认证的 W2/W3 或等价协议 | 不把逐层 host admission 声称为 Splash 式最终目标已完成；安全与净收益优先 |
| D08 | 采用现有 memory/source 基础 | 避免重复账本和身份验证 |
| D09 | program / weights / bindings / qualification 分 key | 防止错误复用，也避免无谓重新编译 |
| D10 | GPU 与 hybrid 共用 worker 复用能力 | 避免生命周期差异制造虚假 ANE 优势 |
| D11 | 新量化与调度分别验收 | 能归因收益和误差；不掩盖回归 |
| D12 | 不默认全模型导出/复制 W8 | 本机 frozen 负结果、磁盘限制与已隐藏 staging |
| D13 | 条件缓存与近似 DiT cache 分离 | 精确复用与跳计算不是同一种质量合同 |
| D14 | App/CLI 共用 profile registry 与 receipt | 最佳 CLI 跑分不等于 App 实际路线 |
| D15 | BaseOnly 使用精确 ABI，不带零 LoRA 修正和 hidden 导出 | 能力超集兼容不等于最快 base；支持双向切换回归 |
| D16 | 先预算、查质量/性能正负记录，再准备选定程序 | 采用 Splash 的启动闭环；NoGain 不每次付编译/校准成本 |
| D17 | 有界 session 组合既有资源，持久 worker 独立推进 | 一程序、多 function/binding、双 bank、小 scales，不新增内存私账 |
| D18 | Breaker 分设备安全与完整路径性能两层 | ANE 局部较快不足以证明完整请求快；性能退出在有效边界完成 |
| D19 | 新增独立 Native Metal Graph 后端 | 参考 Splash 完整 runtime，拥有 Qwen DiT kernels、queues 和 ANE 编排 |
| D20 | N 同时与自身 GPU 和当前最佳路线比较 | 不用退化的重写 GPU 基线夸大 ANE 收益 |
| D21 | encoder/VAE 阶段桥接，DiT 热路径原生 | 不重复整份模型，不把逐层 MLX island 声称为完整新后端 |
| D22 | W1 分为同步准入 W1S 与待实现 reactor W1R | 消除设计与当前 API 的命名冲突，两者都不自动成为 B2 |
| D23 | N-05a Base B1 后，N-05b B2 与 N-06 LoRA 分别推进 | 不以深窗口协议阻塞首个 LoRA 扩展，保持各自资格 |
| D24 | 完整参考 Splash 模型 caller/事务与启动管理 | 不把完整运行时复用收缩成 ANE FFN 类；BF16/H4096/舍入需适配 |
| D25 | 显式 native 请求超域明确拒绝；自动策略按资格降级 | 不静默替换执行器或忽略配置；当前新路线仍是实验域 |
| D26 | 深图优先设计小窗口独立 generation，另认证真实完成或 abort 访问协议 | 独立地址不解决 output 晚写竞态，force-signal 不授予读取／复用许可 |

## 7. 需要通过实现解决的未决问题

这些是明确的研发闸门，不应通过文字假设关闭：

1. MLX 自定义 Primitive 在 compile、多 stream、提前 commit 下能否满足副作用与资源寿命合同？最小无模型/小 tensor probe 先回答。
2. Private driver 的 callback/event 对物理访问结束能提供哪些可验证保证？无法证明的状态按 quarantine 处理。
3. W2 预排队的 abort/poison/ResultReadPermit 协议是否能阻止 restore 读取以及 bank 重写？03 给出条件，未认证前停留 W1。
4. Qwen 通道 gate/up 联合 kernel 在本机 M4 Pro 是否真正比现有 MLX 补集快？需同 M/N/K 完整 FFN 而非孤立 GEMM。
5. 分片 down-LoRA 在当前 FP32 rank 与 BF16 边界下的误差和净收益如何？先逐张量，再完整 block。
6. 首步 GPU、较大 row function、多 function 三者哪个完整请求最优？优先测试少量候选。
7. 可以安全保留哪些 program/binding 资源跨 VAE 与请求？由 memory admission 和实际 footprint 决定。
8. N 的 native SDPA 能否保留 Qwen Image segment/prefix 语义且达到现有性能？这是 N 完整 DiT 的必要覆盖；A 是否扩大原生范围仍由其完整 block 证据决定。
9. 新 SDK 的 public runtime INT8 是否适用？需要可编译、可加载、完整路径与质量证据，现有回退不能回答。

## 8. 设计评审与后续 PR 检查表

- [ ] A 的 Base 验证不被 LoRA/worker/N 阻塞；N 独立实施 FFN→block→完整 DiT。
- [ ] N 的 native 覆盖、桥接拷贝、PSO 和后端身份进入回执；无未声明 MLX 热路径。
- [ ] N 同时对照自身 GPU-only 和当前最佳路线，不继承 A 的性能资格。
- [ ] BaseOnly 无 LoRA IO；base→LoRA→base 不保留旧 ABI 或旧内容。
- [ ] NoGain 命中不重复 prepare；压力/取消不污染长期性能资格。
- [ ] B1 与 B2 清楚区分，实际 host gap 和窗口协议写入回执。
- [ ] 新类型在共同词汇中有唯一含义；执行侧 view 不是另一套 plan。
- [ ] 所有伪代码明确为草案，路径/命令不伪装成现行接口。
- [ ] 实线依赖与人为排程选择区分，GPU head 不被不必要的 ANE 准备依赖卡住。
- [ ] 每个共享 buffer 有 source/storage/generation/物理完成合同。
- [ ] partial enqueue/commit 异常走统一 cleanup；GPU replay 也走验证和提交。
- [ ] done、callback、结果有效、读取许可、可释放不混为一个状态。
- [ ] 新旧 adapter、strength、prefix、quant basis 与 rounding 不误命中。
- [ ] 普通 GPU oracle、兼容 hybrid oracle、局部 GPU replay 正确区分。
- [ ] benchmark 的 worker/cache/source verification 冷热状态匹配。
- [ ] 所有性能数字来自既有回执或明确标注条件计算。
- [ ] 没有把计划偏好 ANE、INT8 张量或 fallback 当物理加速证明。
- [ ] 新实现有回退和移除旧路径的明确门槛，App 不误报。
- [ ] 不下载模型、不默认大规模导出，owned 衍生缓存有预算和清理清单。

本次 Base 优先复审的具体缺口与合同见 [09](09-base-first-splash-review.md)。
新增完整独立后端复审、准确上游调用与不可直接移植边界，以及 Base→LoRA／W2 工作包见 [16](16-splash-native-backend-spec.md)。

本设计阶段已进行章节交叉审查：修正了部分提交异常清理、replay 提交、restore 读取许可、binding slot 身份、prepared worker 状态、计划 key 及性能归因等接口草案。实现时仍必须通过代码与故障测试，文字审查不能替代设备验证。

## 9. 许可证与来源维护

Splash 与本次固定版本 vpipe 根许可证均为 Apache-2.0。已有本地 [Splash attribution](../../../THIRD_PARTY_NOTICES/splash-ane.md) 说明早期借鉴范围。后续若移植具体源代码，按实际文件与固定版本更新 attribution、保留所需许可证和 notice；本轮参考代码以设计伪代码为主，没有把上游 runtime 加入构建依赖。
