# Qwen Image GPU / ANE Command Graph 改造设计

更新：2026-10-08。状态：**独立 Native Metal Graph 已执行完整 Qwen base DiT／Euler／Session 和 private W1S。上一整图验证构建 `4fd1e980…`（未开启请求内尺度缓存）的一组 512²、20 步同 seed 配对中，register W1S 相对本后端 GPU 去噪省时 13.60%、请求省时 9.97%，608 次 FFN／19 个稳态步完成，两路 native phase 均释放成功。当前结论是“此组有收益、未获性能资格”；App、LoRA、编辑、Z-image 尚未通过此路线。W1S 仍有逐层主机准入，register 仅为 CLI 显式实验选项，generic 仍为 W staging 默认值，产品默认路线不变。** 详见[Base E2E 状态](../../status/2026-10-08-native-register-w1-base-e2e.md)。历史 `16fecb41…` 构建的默认关闭尺度缓存已通过生成数据及真实 FFN 的 MLX 身份／命中／逐位一致验证；两种执行顺序的耗时反转，正式均值没有净收益，因此未追加整图，见[真实缓存筛选](../../status/2026-10-08-native-scale-cache-real-screen.md)。旧整图收益不转授缓存。

本设计以 Splash #260 的应用自有 command graph、通道切分、设备事件和有界资源为主线，参考 vpipe 的原生 Metal 执行、LoRA scratch、权重 staging 与 residency，再接入 TurboCider 的模型语义、现有 GPU 算子、模型库和内存执行合同。目标是形成可测量、可恢复、可复用的 Qwen Image 2.1 异构后端，而不是增加另一套独立实验开关。

01–10 中的目标架构接口、JSON 和代码片段主要是**建议设计或伪代码**；不能直接当作已有 API 或性能证据。11–13 对照实际源码明确组件、完整图、Session 与内存阶段；14 保留原始 W1 草案，15 记录当前 W1 实现／接线及验证边界；16 补充完整 Splash 风格新后端规格和复审决策；17 对照当前 Native GPU LoRA 全目标接线及 generated 验证，并给出后续独立 hybrid ABI 方案；18记录MLX Edit encoder的1–3图GPU融合实现；19记录最新Splash参考、ANE runtime encoder桥和完整负结果。组件/历史实验另有状态回执，不转授当前代码。开发继续只使用本地资产，没有下载新模型或复制完整 checkpoint。

较早库 `61fe967c…` 的 Native GPU LoRA 算子与两层 tiny DiT 已通过 generated MLX 对照，当时请求门为 BaseOnly，见[组件状态](../../status/2026-10-08-native-lora-gpu-components.md)。随后新库 `6d914980…` 已接通显式 GPU Viggle r128／strength1／六步窄域，一组真实配对完成全部 227 target、物理释放和目视接近；native request 25.437 s 对 MLX 20.939 s，慢 21.48%，因此没有性能／自动资格，hybrid LoRA 仍拒绝。详见[真实 LoRA 状态](../../status/2026-10-08-native-lora-gpu-e2e.md)。独立新后端完整规格、上游复用和当前缺口集中在 [16](16-splash-native-backend-spec.md)。

**本次新增后端路线的实施入口：** [10](10-native-metal-graph-backend.md) 明确 A（MLX 基础改造）与 N（独立 Native Metal Graph）并列；[16 §3](16-splash-native-backend-spec.md#3-参考-splash-完整实现复用到哪一层) 对照 Splash 完整 runtime 的复用，§4.2–4.3 把可信算子、访问范围、buffer liveness、CB 切分和双 block 窗口落实为开发合同。N 自己编码并提交完整 DiT／Euler，不通过捕获 MLX trace 获得提交权。当前 W1S 与未来设备窗口分别验收，不能预先保证 Splash 的 LLM 收益也出现在图像模型中。

`77863b3f…` 新构建的 opt-in SGMMA 已完成[独立组件筛选](../../status/2026-10-08-native-lora-sgmma-screen.md)与[一次六步整图配对](../../status/2026-10-08-native-lora-sgmma-e2e.md)。真实单层 FFN 两种顺序省时 7.26%／3.59%，但完整请求 Native 23.987 s、MLX 20.960 s，仍慢 14.44%；目视和阶段释放通过，保持实验，不授性能／ANE／W2 资格。

后续 `e6ea1909…` 的独立 paired base 原语完成完整构建、CPU 合同及一组真实 FFN 筛选；两顺序暖机后慢 0.0970%／0.5691%，输出逐位相同，属于 NoGain，未接模型或追加 E2E。详见 [17 §10–11](17-native-lora-integration.md#10-paired-base-原语实现通过当前输入没有净收益)与[状态](../../status/2026-10-08-native-lora-paired-base-screen.md)。[17 §5.3](17-native-lora-integration.md#53-后续最小实施包hiddenreference-与-rank128candidate)的独立 ABI Spec、MIL emitter、NativeLoRA low／apply 拆分已在 `47d3ed80…` 完整构建中落地：CPU 合同与微型共享 low 检查通过，stages／controller／hybrid 请求尚未接通。暖机 MLX 对照因框架预算保护提前结束，没有完整性能结论；见[本轮状态](../../status/2026-10-08-native-lora-foundation.md)。

最新[18 · Edit encoder](18-edit-encoder-acceleration.md)使用 `769123f8…` 构建：真实1024 reference的1–3图视觉＋语言计算省时18.70%／20.54%／21.77%，5%张量筛选通过、2%严格线未通过；512 reference三图六步整图质量通过，但encoder无明显提速，不转授稳定／整图性能资格。App编译通过，实际执行GPU；Native encoder／ANE／VAE尚未迁移。详见[状态](../../status/2026-10-08-qwen-edit-encoder.md)。

接续[19 · ANE runtime encoder](19-edit-encoder-ane-runtime.md)参考Splash HEAD `0d5a035e…`，接入显式MLX／PrivateW8语言FFN桥。三图layer0两顺序省时约16%，完整三图encoder **11.998→13.202s（慢10.03%）**，retained conditioning RRMS5.754%未过预声明5%线；保留诊断候选、默认GPU，没有Native encoder完整command graph或发布资格。详见[本轮状态](../../status/2026-10-08-qwen-ane-encoder.md)。

新增 [22：完整Native Metal编辑encoder](22-native-edit-encoder.md) 已完成视觉27层＋语言36层的原生GPU图，支持GQA/mRoPE/DeepStack和两层提交/来源窗口。
单图独立FP32 encoder省时24.93%，实际512²六步编辑encoder省时7.49%、请求少1.59%，图像pixel RRMS0.88%；保持显式实验。
本轮不是Native encoder ANE/W2，也未改变DiT或默认MLX路线，见[状态](../../status/2026-10-08-qwen-native-encoder.md)。

## 1. 决策摘要

最新 [21 · Encoder 调度与校准](21-encoder-scheduling-and-calibration.md) 已落地有界分区、单Session准备缓存、GPU先提交后预取及独立36层S1校准。三图完整encoder省时8.85%，准备复用后11.01%；完整conditioning仍5.754%，符合预声明8%筛选。仍为MLX/private桥接，完整Native encoder/W2与整图质量另行验收；见[状态](../../status/2026-10-08-qwen-encoder-scheduling.md)。

1. **TurboCider 拥有执行计划、依赖与资源生命周期。** MLX 和原生 Metal 是执行计划的不同执行器，不让模型层依赖各执行器的内部队列。
2. **同时设计两条可独立推进的后端路线。** A 路线保留 MLX 并验证设备事件适配；N 路线新增 Splash 式 Native Metal Graph 后端，自主执行完整 Qwen DiT。N 不以 A 失败为前提；两者共用模型、内存和事务合同，分别证明收益。
3. **基础分工是 GPU attention + GPU/ANE 通道切分 FFN。** 先保留当前 GPU oracle、BF16/FP32 数值边界和模型语义，再逐项引入融合。
4. **首步与后续步分别计划。** 512² 对应 1024 个目标图像 token，但首步实际行数还包括文本/参考图，不能统一套用 1024 桶，不能截断 prompt。
5. **先独立完成 BaseOnly，再接入 LoRA。** Base 阶段不带修正输入、rank scratch 或 hidden 导出；LoRA 后续单独校准、扩展 ABI 与资格。其 gate/up 修正在非线性前，down 修正和多 adapter 累加遵守独立数值合同。
6. **程序、绑定、权重、校准、条件缓存分开管理。** 动态程序不能无谓绑定所有 checkpoint/adapter；包含权重的衍生物也不能只按形状命中。
7. **逻辑失败不等于物理完成。** 解除 GPU event wait 后，晚到的 ANE 仍可能访问 surface。资源隔离、回放和提交规则先成立，才扩大预排队窗口。
8. **生产策略由完整路径证据决定。** 组件快、计划偏好 ANE、或者 INT8 张量存在，都不足以证明完整请求快或硬件执行方式。

## 2. 阅读与实施顺序

| 文档 | 内容 | 主要读者 |
|---|---|---|
| [01 证据与差距](01-evidence-and-gap-analysis.md) | 固定版本来源、当前实现与实验、不应照搬的假设 | 全体 |
| [02 总体架构](02-architecture-and-contracts.md) | 模块边界、计划编译、公共合同、目标执行图、兼容路线 | Runtime / 模型 |
| [03 Command graph 与交接](03-command-graph-and-handoff.md) | MLX/Metal 编码、事件、双 bank、错误恢复、参考代码 | Metal / ANE |
| [04 Qwen、LoRA 与量化](04-qwen-lora-and-quantization.md) | 模型数学、行数、精度、kernel fusion、量化 recipe | 模型 / kernel |
| [05 资源、缓存与 worker](05-resources-cache-and-worker.md) | 统一内存、分层 key、持续 worker、失效与清理 | Runtime / App |
| [06 校准与验证](06-calibration-performance-and-validation.md) | 指标定义、轻量实验、E2E、正确性及性能门槛 | 验证 / 性能 |
| [07 实施工作包](07-implementation-roadmap.md) | 分阶段改造、依赖、文件归属、完成定义和退出条件 | 实施负责人 |
| [08 来源与设计决策](08-source-map-and-decisions.md) | 可追溯引用、关键决策、未决问题、审查清单 | 审查 / 维护 |
| [09 Base 优先复审](09-base-first-splash-review.md) | Splash 管理细节、BaseOnly ABI、启动/负缓存、分阶段验收和 LoRA 复用边界 | 建议先读 |
| [10 独立 Native Metal Graph 后端](10-native-metal-graph-backend.md) | Splash 完整 runtime 复用、原生 DiT、GPU/ANE graph、后端接入和独立性能验收 | 原生后端实施 |
| [11 原生 block 与 ANE 桥接线](11-native-block-and-handoff-integration.md) | 已实现组件、准确数学、32 层算子/scratch 复用、prefix 事务、W0→W1 与验收顺序 | 当前实施入口 |
| [12 独立原生 DiT runtime](12-native-denoiser-runtime.md) | 实际 NativeDenoiser API、完整 base 图、PSO/arena 复用、缓存事务和生产接线边界 | 完整原生执行实施 |
| [13 原生 Session 与内存阶段](13-native-session-and-memory-phase.md) | 真实请求选择、held-fd 来源、阶段就绪、统一账本、失败隔离与报告 | 当前请求接线 |
| [14 Native W1 完整并行 FFN](14-native-w1-parallel-ffn.md) | 原始接线设计；未由 15 确认的 API／双 bank／预取继续是提案 | 设计参考 |
| [15 W1 实际实现与整图接线](15-native-w1-implementation.md) | 完整 FFN 控制器、register W staging、完成域、账本与当前整图证据边界 | 当前实施入口 |
| [16 Splash 风格完整后端规格](16-splash-native-backend-spec.md) | 独立 N 后端完整 runtime、上游复用边界、W1S/W1R/W2、Base→LoRA、缓存／故障和下一步工作包 | 本次复审入口 |
| [17 Native LoRA 全目标接入](17-native-lora-integration.md) | 本地 adapter target、held-fd／重绑／materialize、GPU 数学、hybrid correction/hidden ABI 与验证阶段 | 后续 LoRA 实施 |
| [18 Edit encoder GPU 融合](18-edit-encoder-acceleration.md) | 1–3参考图、视觉D72→D80、语言GQA/causal、Splash复用、真实组件与整图边界 | Edit encoder 实施 |
| [19 Edit encoder ANE runtime](19-edit-encoder-ane-runtime.md) | 最新Splash、语言FFN通道桥、固定图变长分块、真实组件收益与完整负结果 | ANE encoder 实施 |
| [20 Encoder runtime 继续优化](20-encoder-runtime-optimization.md) | Splash旋转W8A8、动态BF16 partial、完整encoder小幅净收益、校准与设备图缺口 | encoder 最新实测 |
| [21 Encoder 调度与校准](21-encoder-scheduling-and-calibration.md) | 有界分区、准备缓存、下一层预取、36层S1与完整encoder净收益 | encoder 调度实测 |
| [22 完整Native编辑encoder](22-native-edit-encoder.md) | 原生视觉/语言模型塔、GQA/mRoPE/DeepStack、两层窗口、实际单图编辑净收益 | 单图Native encoder最新实测 |

先读 16 了解独立新后端完整规格和复审结果，再读 09–10 的 Base→LoRA 顺序与两路线边界。11–13 对照实际组件、去噪图与请求；14–15 对照原始并行设计与当前实现。开始写代码前，至少读 01、02、03 的失败语义及 04 的舍入合同。03–05 的接口由 02 的共同词汇约束；实现阶段若改变共同词汇，应同步所有章节。

## 3. 首个纵向交付与最终目标

首个可比较纵向交付限定本地 Qwen Image 2.1 BF16、512²、无参考图、batch=1、base 20 步，使用独立 BaseOnly ABI。精确条件缓存与近似 DiT cache 的开关保持与对照一致，性能验收初始关闭近似 cache。

Base 阶段独立通过后，再接入现有 Viggle r128 / strength=1 六步 adapter，单独验证 LoRA 数学、准备开销和完整请求收益。持久 worker、LoRA、公共 runtime INT8 都不是首个 Base 交付的前置条件。A 路线不依赖完整原生 DiT；N 路线将完整原生 DiT 作为自身目标，按 10 的 FFN → block → DiT 顺序独立实施。

区分 **B1 有收益的安全原型**（可以是 W0/W1）与 **B2 设备驱动目标**（成功路径无逐层 host admission，需要已认证的 W2/W3 或等价协议）。保留 B1 合理，但不能据此声称 B2 已完成；具体管理合同见 09。

该边界是**首个验收域**，不是永久删减产品能力。长文本、编辑、多 adapter、其他 strength/分辨率和量化来源继续由现有兼容路线处理。自动策略未满足新路线资格时选择既有 GPU；显式 native 实验请求不满足域时明确失败，不静默回退。

最终目标是：

- Qwen 模型语义只维护一份，生成带副作用和资源依赖的执行计划；
- GPU-only 与 GPU+ANE 使用同一计划、相同条件和数值合同；
- 稳态不靠 CPU 逐层等待驱动设备，CPU 在有界事务边界验收；
- source、compiled program、bindings、scratch、calibration 可复用且可失效；
- public/fixed/private executor 按能力接入，不掩盖同步及数值能力差异；
- App、CLI 使用同一个经过 qualification 的配置与实际执行回执；
- 失败可安全恢复，无法确认设备结束访问时隔离 worker/资源；
- 在相同质量与内存约束下，按完整请求耗时选最快路线。

“彻底完备”指这些合同、回退、测量和扩展边界完整，不表示所有设备/模型组合都应强制打开 ANE。

## 4. 已知证据与收益口径

2026-10-07 的历史本机记录：M4 Pro / 48 GiB，Qwen base 纯 GPU 48.400 s，当时旋转 W8A8 候选 43.044 s；LoRA 六步纯 GPU 20.664 s，候选 21.292 s。后者去噪变快但初始化抵消收益。它们是特定原生 CLI、单提示词结果，尚非 App 或多场景结论。

历史 native GPU-only 后端的一组完整配对：[Session/E2E 状态](../../status/2026-10-08-native-session-and-e2e.md) 记录 native 49.885 s、MLX 49.062 s，图片 RGB relative L2 约 1.10%。去噪减少约一秒，但 native prepare 约 1.65 秒抵消收益；不据此发布或预测 ANE 收益。它与前一段的不同构建／prompt 不作直接排名。

上一整图验证构建的配对使用本地 BF16、512²、20 步、seed=42、blue teapot 提示词与关闭的近似 DiT cache：native GPU 去噪 **41.616892042 s**、request wall **48.602237417 s**；register W1S 分别为 **35.956603708 s**、**43.758467125 s**。两张最终图经目视接近，4 个原始来源文件的 stat 未变，外部前后采样未观察到新增 swap。每路线只有 1 个样本，固定 native→hybrid，各自新进程但系统 source／driver cache 未清；不外推稳定收益、所有提示词或其他设备，也不与历史构建混算。完整回执与限制见[本轮状态](../../status/2026-10-08-native-register-w1-base-e2e.md)。没有同构建 generic W1S 的完整请求对照，因此这里的收益不能单独归因于 register 内核。

register 仅替换 dense BF16 Sylvester H128／H512 的 **W staging**，使用 32-thread 寄存器 kernel，prepare 时额外准备 4 个 PSO；A8／restore、Surface 与统一账本不变。它不是 W1R／W2，也不证明硬件 INT8 算术或实际 ANE placement。接口和计时范围见 13／15。

历史证据继续独立保留：[较早 W1 组件状态](../../status/2026-10-08-native-w1-session-components.md) 的 tiny 完整去噪与真实 FFN 中，step 1 约省时 38.85%，held-out step 19 约慢 1.07%，当时未推进整图；不能改写为本轮测量。更早的 layer 0 首步／后续步约 6.4%／3.9% 下降见[完整图与 block 状态](../../status/2026-10-08-native-denoiser-and-real-block.md)，tiny 并行 FFN RRMS 约 2.48% 见[原始 W1 状态](../../status/2026-10-08-native-w1-parallel-ffn.md)。这些记录不授予当前构建资格。实际 API 见 11–13，原始 W1 设计见 14，当前实现见 15；16 明确距 Splash 多层设备图的缺口，06 给出验证门槛。

## 5. 工作约束

- 使用已有本地模型；不下载、不复制完整 checkpoint。
- 首轮只保留小型程序/编译缓存和已有必要样本，新增空间先有预算与所有权记录。
- 笔记本按轻量组件筛选后最多两组 E2E 推进；失败即停止该候选，避免笛卡尔积扫描。
- 不把量化质量、步数、分辨率、prompt 截断、近似 cache 的变化混入调度收益。
- 保留已有分发构建和 private API 能力隔离；自动策略缺少资格时选择兼容路线，显式 native 实验缺少 capability 或超出域时明确拒绝。
- 设计提案与实际实现状态分开记录；不能把目录/接口已经创建写成完整后端已交付。

本轮新增 [23：Native encoder 与无 KV cache 编辑首步 ANE](23-native-encoder-prefill-ane.md)，补上有界行适配、原生 FFN 分块及提交后下一层源准备。实际完整 phase 收益以本轮状态为准；不是 W2 发布资格。
