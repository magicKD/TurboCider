# GPU/Core ML 与 runtime LoRA：实验总结和维护入口（2026-09-28）

当前简明决策统一维护在 [GPU/ANE 加速入口](acceleration.md)。本页保留
Qwen-Image-2.1 和 Z-Image Turbo 的详细操作说明与历史实验结论；
各次测试的提示词、精度、调用数及限制分别见
[Qwen 记录](qwen21-runtime-lora-fused-2026-09-28.md)和
[Z-Image 记录](z-image-runtime-lora-fused-2026-09-28.md)。以下速度均为
M4 Max、**已加载的 resident 热请求**，除标注 1024² 外均为 512×512，包含 VAE/PNG；
不能当作冷启动、别的 LoRA、设备或画质的保证。

这里区分原有冻结图/LoRA 成绩与后续 runtime-weight 的优化复测。
runtime-weight 已从单层原型接入两模型 base 路径，见
[整模型实验](runtime-ane-integration-2026-09-28.md)；
[组件实验](runtime-ane-component-2026-09-28.md)保留为集成前的历史记录，
单层计时不可与整请求数字混比。
旧的逐项报告保留用于复盘；1024² 与 512² 是不同工作负载，不能混比。

本次收尾的选择：保留此前最快的冻结 base 图；完整 runtime LoRA 以
`lora_fused` 为已有研究入口；新 `runtime` graph-v2 保留为显式 optional。
**不把 LoRA 合入 checkpoint、Core ML artifact 或 runtime base 权重槽。**
Qwen 六步 LoRA 尚无可靠 GPU/ANE 优势，继续以普通 GPU 为默认对照。

## 历史性能快照（bc1及更早构建）

本节保留实验演进，不是当前版本排行榜；最新匹配对照统一见
[当前性能表](acceleration.md#性能快照与口径)。不要将下列旧构建的分母
与后来Q/K融合、长chunk或LoRA实验拼接。

接续 `bc1…` 增加稳定混合阶段的周期采样，保留完整 GPU probe，减少额外
同步；同构建 Qwen base GPU/runtime/frozen 为 **42.773 / 39.340 / 30.313 s**，
Z 为 **7.010 / 6.562 / 5.363 s**。Qwen 六步 LoRA runtime/GPU 为
**8.821 / 8.227 s**，仍未胜出。两模型真实共图状态切换通过；见
[该轮优化与证据](runtime-ane-periodic-sampling-2026-09-29.md)。

此前进展见 [2026-09-29 完整 GPU block 探测](runtime-ane-full-block-probe-2026-09-29.md)：
`f287…` 的 Qwen 六步 runtime LoRA 为 **8.777 s**，GPU **8.159 s**，约
**0.930×**。最后两次请求没有 ANE prediction，不是混合加速胜出；默认不改。
以下各表保留历史构建口径。最新简明选择表见 [状态入口](README.md)。

后续新增的 [runtime-weight LoRA graph-v2](runtime-ane-lora-2026-09-28.md)
已能让 BF16 Z/Qwen 在不合并 base 权重的情况下执行完整 LoRA，支持同图
base/adapter 切换；当前 Z 仅小幅收益、Qwen 变慢，不推广为最快路线。
下列 base 和冻结 LoRA 数字保留各自原始构建/工作负载口径。

### graph-v2 首轮 LoRA（ed7 构建）

| 工作负载 | 普通 GPU | runtime v2 | GPU/runtime | 决策 |
| --- | ---: | ---: | ---: | --- |
| Z，distill LoRA，8 步，c352/auto | 8.598 s | 8.293 s | 1.037× | 收益小，保留 optional；不替换更快的冻结图 |
| Qwen，Viggle v0.2.1 r256，6 步，c288/auto | 8.125 s | 10.811 s | 0.751× | 负结果，不推荐为加速配置 |
| Qwen，同 LoRA，`chunks=0` | 8.158 s | 10.762 s | 0.758× | 仅拆图消融，热请求没有 ANE prediction |

同一 `ed7…` 原生库，每路排除冷请求后两个热样本，非 ABBA 验收。
chunks=0 的 denoise 仍慢约 2.10 s，说明关掉 ANE 计算并未消除拆图路径成本；
下一轮优先恢复无收益层的 unsplit GPU，而非继续盲目增大 chunk。
Z/Qwen 都通过真实 base → A → 合成 B → base 共图状态检查；狐狸样本肉眼接近，
并非广泛画质通过。细节及完整 SHA 见 [v2 记录](runtime-ane-lora-2026-09-28.md)。

接续的 `fb79…` 构建已补上 Qwen 真正的 unsplit GPU 回退及独立编译缓存：
六步 LoRA 一冷三热中位 **9.530 s**，同构建 GPU **8.168 s**，仍仅 0.857×。
同构建 BF16 base 三路回归：Qwen 40 步 GPU/runtime/frozen 为
**42.735 / 38.988 / 30.221 s**；Z 8 步为 **7.009 / 6.587 / 5.357 s**。
两模型 runtime 分别约 1.096×/1.064×，冻结图仍更快。配置、证据和新测试见
[unsplit GPU 优化与回归](runtime-ane-unsplit-2026-09-28.md)。下列旧数字仍保留原始构建口径。

以下先列最近完整的两模型 runtime-weight 对照，再列冻结图和 LoRA 的独立实验。
各行只比较本行的配对实验；base 与六步 LoRA 不能直接比秒数。
“冻结”指 Core ML 固定 base 权重，**并不等于融合 LoRA**。

### Runtime-weight 两模型候选（c288 / auto，量化组件接入前）

| 工作负载 | GPU 热请求 | GPU/Core ML 热请求 | 加速比与决策 |
| --- | ---: | ---: | --- |
| Z base，512²，8 步 | 7.009 s | 6.608 s | 1.061×；显式 optional |
| Qwen base，512²，40 步 | 42.771 s | 39.183 s | 1.092×；显式 optional |
| Z base，1024²，8 步 | 31.444 s | 30.012 s | 1.048×；多 chunk 循环已测，样本仍少 |

以上三行来自同一最终候选二进制，512² 各每路两次热请求，1024² 各一次；
冷请求排除，不是广泛质量或性能资格验收。完整证据与 library SHA 见
[最终候选](runtime-ane-integration-2026-09-28.md#本轮最终候选两模型自动策略)。

后续 Q4/Q8 staging 组件接入后，Z BF16 路线另做交错回归：GPU 7.005 s、
runtime 6.561 s，约 **1.068×**（各四次热请求），未见明显退化。
这不是 Q4/Q8 整模型成绩，也未在该新构建上重测 Qwen；详见
[组件接入后回归](runtime-ane-integration-2026-09-28.md#量化组件接入后的-bf16-回归)。

再后续已接入 Z 原生 GGUF runtime：Q8_0 的 512²/8 步 c352 交错复测，
约 **9.199 → 8.097 s，1.136×**（四次热请求/路线）；c416 筛选近似持平。
1024²/8 步初测 **41.041 → 35.607 s，1.153×**（每路仅一次热请求），
已验证同图多 chunk 循环；不是编辑或 LoRA 的成绩。
这是 Q8 对自身 GPU 基线的提升，不比上述 BF16/冻结图更快；Q4 真实模型
在该轮仍未验证；现已补充 [Q4_0 初测与 padding 修复](runtime-ane-q4.md)。
配置、混合 dtype 修正和后续复测见
[Q8 整模型进展](runtime-ane-integration-2026-09-28.md#z-原生-gguf-q8_0整模型接入与-chunk-筛选)。

Q8 接入后的同构建 Z BF16 回归（各两次热请求）：GPU **7.008 s**、
runtime c288 **6.540 s / 1.072×**、冻结 W8A8 **5.361 s / 1.307×**。
原 GPU PNG 与上一构建相同；冻结图仍最快，默认策略不变。
见[同构建回归](runtime-ane-integration-2026-09-28.md#同构建-bf16--冻结图回归)。

### 冻结 base 图与完整 runtime LoRA（独立历史对照）

| 工作负载 | GPU 热请求 | GPU/Core ML 热请求 | 加速比与决策 |
| --- | ---: | ---: | --- |
| Z base，8 步，冻结 W8A8 | 7.009 s | 5.360 s | 1.308×；保留已测更快路线 |
| Qwen base，40 步，冻结 W8A8 | 42.780 s | 30.252 s | 1.414×；该轮筛选较快，保留 |
| Qwen Viggle v0.2.1 r256，6 步，完整 `lora_fused` | 7.811 s | 7.715 s | 1.012×；落在波动内，GPU 仍为默认 |
| 同一 Qwen LoRA，三张缩到 512px 的参考图编辑 | 12.34 s | 12.50 s | 无可靠优势，不推广 |

冻结 base 两行来自同一初版集成二进制，**早于 runtime 输出恢复 SIMD 优化**；
Z 每路两次热请求，Qwen 每路一次，不是 ABBA 资格验收。
LoRA 行来自独立历史实验（生成四次、编辑两次热请求中位）。
Z 完整 runtime LoRA 的已测收益约 1.20×，可选直接 FP16 修正约 1.217×；
对应配置与质量边界见下文。不能将这些结果解释为任意 adapter 都能同样加速。

runtime 的输出恢复 SIMD、c288 chunk 和自动调度修正均予以保留；
改进过程及负结果见[优化复测](runtime-ane-integration-2026-09-28.md#simd-输出恢复与-chunk-复测)。
它们没有改变冻结图或完整 LoRA 的历史成绩。

## 保留什么、如何选择

| 请求与路线 | 入口 | 当前结论 |
| --- | --- | --- |
| 无 LoRA，原生最快的已测 base 路线 | 保留现有 `execution=auto` 及模型专用显式 manifest；不要为了共图强制使用 `lora_fused` | 不改变此前 base 的加速选择；Z-Image 另有显式 1024-image-row/5120-channel W8A8 路线，已测约 1.31× GPU。 |
| runtime LoRA、严格 GPU 对照 | `execution=gpu`、`lora_strategy=inference_time` | 两模型的正确性/观感基准；Qwen 仍以此为默认。 |
| runtime LoRA、完整可复用 base Core ML FFN | `execution=gpu_ane`、`hybrid_mlp_mode=lora_fused`、匹配的 **base-only** `ane_manifest`、`allow_approximation=true` | GPU 运行 LoRA，Core ML 只保存冻结 base 权重。同一个图可供 base 和不同 LoRA 复用；Z-Image 4096-channel 完整路线实测约 1.20×，可选的直接 FP16 修正路线约 1.217× GPU；Qwen 6144-channel 的整图优势尚不稳固。 |
| runtime-weight、跨层共用无权重常量图 | `hybrid_mlp_mode=runtime`；base 可用 v1，BF16 LoRA 必须 v2 `--lora-inputs` | token-row 切分，权重槽始终 base-only；保留 FP16/SIMD、Q4/Q8 staging、自适应及回退，不取代冻结图。Qwen LoRA 当前反而变慢。 |
| Qwen 仅导出 base gate/up | `hybrid_mlp_mode=lora_gate_up`、匹配的专用 manifest | 完整计算 LoRA，但 GPU 负责 SiLU/down；比原有融合 base 图慢，只保留显式研究选项。 |
| 仅 GPU 后缀计算 LoRA | `hybrid_mlp_mode=lora_suffix`、匹配的 base manifest | **故意不完整**：漏掉 Core ML 前缀的 LoRA gate/up/down 效果；不得作为正确的 LoRA 加速成绩或自动默认。 |
| Z-Image 将 LoRA 合入模型 | `hybrid_mlp_mode=lora_merged`、匹配的 adapter-bound manifest | 只用于独立对照；换 LoRA 要换图，不满足 runtime 共图要求。 |

`lora_fused` 的名称沿用已有 CLI 契约：它融合的是 **base FFN 算子**，
不是 LoRA 权重。GPU 先计算该次请求的 gate/up LoRA 差值，作为
Core ML 的第二个 FP16 **激活输入**，在 SiLU 前相加；图输出 base down
和 pre-down hidden，再由 GPU 计算该 hidden 上的 LoRA down。没有 LoRA
时差值是零，GPU 不计算 LoRA down。注意力及非 FFN LoRA 仍由 GPU 执行。
若只让 Core ML 输出已完成的 base FFN，之后无法准确补回 SiLU 前的
LoRA；旧 `lora_suffix` 因而不能替代此路线。manifest 校验 base checkpoint
的 SHA 与几何，并要求它不绑定适配器身份；显式选错图应报错而非
悄悄使用不相容的近似。

对 Z-Image，选择固定 1056-row、4096-channel、W8/FP16 激活的
`--runtime-lora-fused` 图，512px、8 步，resident；额外的
`TURBOCIDER_Z_RUNTIME_LORA_DIRECT_FP16=1` **仅为可选实验**，令 GPU
LoRA gate/up 差值从 FP32 直接舍入至 FP16，不先经过 BF16。它不跳过
投影，但有小幅数值差异（单狐狸样本对原 hybrid 的 RGB RMSE 0.0110），
同二进制配对约快 0.6%，两次最终版热请求与 GPU 的中位比约
**1.217×**；未满足自动推广到其他 LoRA 的质量与跨提示词证据。

对 Qwen，当前已验证的是 Viggle v0.2.1 r256、6 步、512px：
正确的共图 GPU/Core ML 与 GPU 四次热态中位分别为 **7.715 s** 和
**7.811 s**，名义 1.012×，小于运行波动，不作为默认加速结论；
三张诊断性缩到 512px 的参考图编辑分别约 **12.50 s** 与 **12.34 s**
（两次热请求中位），无可靠整图优势。原尺寸参考编辑尚不能按这
组结果推广。GPU 上可选 FP16 低秩计算和 Metal Q/K norm-RoPE 优化
仍是显式选项；不能把基础模型的提升或仅 GPU 后缀的更快数据
记为完整 LoRA 的收益。

## 保留的代码与已撤回的实验

- `native/backends/coreml.mm` 负责冻结图验证、共享输出和动态激活
  接口；`native/backends/mlx.cpp` 负责带适配器的任意矩阵行/列切片。
- `native/models/qwen21/hybrid.cpp` 和 `native/models/z_image/z_image.cpp`
  仅在匹配的显式图上组装 LoRA gate/up、GPU 补集和 down-LoRA；
  `native/models/*_module.cpp` 在 plan 阶段把未经验证的组合挡住。
- `apps/cli/main.mm` 的 `--hybrid-mode MODE --ane-manifest MANIFEST`
  是一次请求的选择，不改写 JSON，也不按文件名猜适配器或图。
- 真实模型的 base → LoRA A → 临时派生 B → base 共图切换由
  `tools/validation/runtime_lora_shared_graph_switch.py` 检查；
  B **只是合成文件**，不能代表另一个训练 LoRA 的画质。
- 不保留 Qwen 的延后 GPU 后缀、双 GPU stream、直接 FP16 LoRA
  差值、只将 3/5/7 层回退全 GPU 等慢速实验开关。延后 GPU 后缀
  的同图交错测试从 **8.018 s** 退化到 **9.880 s** 热态中位；
  负结果在 Qwen 记录中，隔离的对照工具
  `tools/validation/hybrid_runtime_schedule_screen.py` 仍可复用。

切换 manifest、适配器或模式后，先运行 `plan` 和项目测试，再核对
实际绑定投影、Core ML 每请求调用数、PNG 与肉眼任务完成情况；
对不同 adapter 的训练步数、质量和速度重新验证。Core ML 的
CPU/Neural Engine **调度策略不证明**每个算子物理运行在 ANE 上。

## Base、序列切分和 1024² 的历史进度

默认/冻结图混合 FFN 的主体仍是 **intermediate-channel 切分**：GPU 与 Core ML
分别计算通道补集，再相加。Qwen 的 `tiled_sequence()` 则在此基础上把
长序列按固定行数分块，循环使用同层图；它不是独立的 GPU-token / ANE-token
全宽 FFN 行切分后端，也不是所有层共用一个 runtime-weight 图。
新显式 `runtime` 路线才是后者，两者同时保留，不相互替换。
每个 tile 消费完共享输出才能进入下一次预测，避免覆盖尚未求值的输出。

- Qwen 无 LoRA 的 6144-channel W8A8 路线已有收益：512² 输出、40 步、
  1/2/3 张 **256px 缩图**参考、条件缓存命中的历史整请求比约
  **1.410× / 1.400× / 1.387×**。这不是六步 LoRA、原尺寸参考或冷启动收益。
  见 [公开说明](../public/QWEN_IMAGE_21.md)和[路线索引](qwen21-acceleration-routes-2026-09-27.md)。
- 1024² **base 文生图**也做过：5/40 步 resident 暖请求相对 GPU 约
  **1.114× / 1.159×**，含两边一致的验证转储 I/O；仅显式诊断、无参考图。
  未证明 1024² 编辑或 runtime LoRA 的提升；图像面部和局部细节有变化。
  见 [1024² 实验](qwen21-w8a8-ane-1024-diagnostics.md)。
- runtime-weight、按形状共图的 Core ML 路线现已接入两模型 base BF16 的
  显式 `--hybrid-mode runtime`。最终 c288/auto 的 512px 热请求：Qwen 40 步
  约 1.092×、Z 8 步约 1.061× GPU，仍慢于已有冻结图；该轮未测 LoRA
  或完整质量资格。Z 1024² 已有循环 chunk 及视觉对照，不外推编辑场景。
  见[两模型集成与整图筛选](runtime-ane-integration-2026-09-28.md)。
  native 构建包含显式路由，自动选择不变；不增加生成时 Python 依赖。

Qwen 的单层提升为什么没成为 runtime LoRA 的整图提升？完整路线必须先在
GPU 求出 pre-SiLU LoRA 差值、等待后再启动 Core ML，并把 hidden 返回 GPU
计算 down-LoRA。6144-channel/1024-row 每次约有 24 MiB 修正输入和
20 MiB 输出，六步请求有 160 次预测。记录中的单层约 1.48× 不能直接外推：
整图还含 attention、首步、VAE、同步和共享资源竞争。桥接测量中 feature
绑定加输出处理不到 0.02 ms/次，主要时间在 prediction 内及依赖链，不应
把瓶颈简单归于 Python、输出拷贝或 INT8 算力不足。具体硬件执行位置仍未知。

## 保留的 optional 与边界

| 选项 | 维护决策 |
| --- | --- |
| `lora_fused`、匹配 base-only manifest | 完整计算 LoRA，共图主研究路线；不改变最快 base 路线 |
| `TURBOCIDER_Z_RUNTIME_LORA_DIRECT_FP16=1` | Z 独立的小幅有损优化，默认关闭；不外推 Qwen |
| `TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE=1` + `TURBOCIDER_QWEN21_VIGGLE_LORA_FP16=1` | Qwen 已测 GPU 优化组合，显式开启并允许近似；不是跳过 LoRA |
| DBCache、FFN 复用、参考缩图、局部注意力、序列 tile | 保留现有请求门禁，仅在各自已支持的 base/尺寸/步数组合启用；不得一并套到六步 LoRA |
| `lora_gate_up` / `lora_suffix` / `lora_merged` | 保留兼容与研究入口；分别是较慢完整图、故意不完整计算、adapter-bound 图，均不自动推荐 |
| per-layer / Core ML phase profiling、共图切换及 ABBA 工具 | 保留排错与复现实验工具，不在默认性能测试中开启额外 profiling |
| `hybrid_mlp_mode=runtime`；`make test-runtime-ane` | token-row 路由；Qwen/Z BF16，v2 可选完整 LoRA；原生 GGUF 仍仅 base（Q8 已测，Q4_0 已有初测、Q4_1 待验证）；不自动选择、不替代冻结图 |
| `TURBOCIDER_RUNTIME_ANE_CHUNKS=auto` 或 `0...128` | 仅对显式 runtime 生效；未设置为 auto，0 是保留切分边界的 GPU 消融，其余为固定 chunk 数，不改变普通 GPU 基准 |
| `TURBOCIDER_RUNTIME_ANE_PROFILE=1` | runtime 分段诊断，默认关闭；正式测速不额外开启 profiling |

负结果保留文档，不重新引入已撤回的调度分支。性能比较必须固定二进制、
输入、参考尺寸、LoRA、步数及缓存状态；测试前检查其他推理负载，繁忙时等待，
不要终止他人进程。用交错 GPU/hybrid 重复和整请求墙钟判断收益。
画质以肉眼主体、构图、参考身份/细节及编辑指令为准；不以像素相等为目标。
共图切换时要求首尾 base 相同，则是在检查状态泄漏，不是限制近似路径的画质。

## 代码维护边界

| 层次 | 保留的核心文件 | 维护原则 |
| --- | --- | --- |
| 冻结 base 图及完整 LoRA | `native/backends/coreml.mm`、`native/models/qwen21/hybrid.cpp`、`native/models/z_image/z_image.cpp` | 保留最快 base 路线与 `lora_fused`，不把低秩权重合入 artifact |
| runtime-weight 执行器 | `native/backends/ane_runtime.{hpp,mm}`、`ane_runtime_convert.hpp`、`ane_runtime_quant.hpp` | 图/槽/dense 与 affine 转换/worker；不依赖模型、LoRA 或 MLX 调度 |
| token-row 调度与模型桥接 | `native/backends/ane_scheduler.hpp`、`ane_ffn.{hpp,cpp}` | 调度独立；桥接持有 tensor 生命周期并负责失败后 GPU 重算 |
| 模型与公开契约 | 两模型 pipeline、`*_module.cpp`、`apps/cli/main.mm`、`native/platform/apple/results.mm` | 显式选择、拒绝不支持的组合、报告真实模式和失败状态 |
| 离线工具 | `tools/coreml/export_runtime_ane.py`、`tools/native/ane_runtime_probe.cpp` | 导出/单层探针独立构建，不成为生成时 Python 依赖 |
| 可复用整图验证 | `tools/validation/runtime_ane_model_screen.py`、`runtime_lora_shared_graph_switch.py`、`hybrid_runtime_schedule_screen.py` | 分别负责路线对照、LoRA 共图状态、调度 A/B；原始证据留本地 |

不复制一套模型专用 runtime executor，不将试验开关升级为全局默认。
`execution=auto` 是原有路由策略；`TURBOCIDER_RUNTIME_ANE_CHUNKS=auto`
只在已经显式选择 `runtime` 后调节 chunk，两种 auto 含义不同。
新增输出 SIMD 已通过穷举转换测试及整图复测，c320/c288 已进入真实
模型筛选；BF16 的 c288 和 Z Q8 的 c352 是本机值得继续验证的候选，不自动替代
冻结图，也不代表其他 SoC 上最优。未证明有收益的原型保留证据，
不推荐给正常请求。

## 便携 CLI 示例

在仓库根目录运行，先准备本地 checkpoint、adapter 和已经编译的 base-only
manifest。示例不下载或编译模型，不把本机路径、缓存哈希写入源码。

```sh
# 完整 runtime LoRA GPU 基准（两模型各自的步数/adapter）
build/native/turbocider plan examples/requests/z-image-runtime-lora-512.json
build/native/turbocider plan examples/requests/qwen21-viggle-runtime-lora-512.json

# Z 共图：1056 rows / 4096 channels，W8 权重、FP16 激活
build/native/turbocider generate models/Comfy-Org-z_image_turbo \
  examples/requests/z-image-runtime-lora-512.json \
  --hybrid-mode lora_fused --ane-manifest path/to/z-shared-base/manifest.json

# Qwen 共图研究：1024 rows / 6144 channels / 32 blocks，W8A8
build/native/turbocider generate models/Comfy-Org-Qwen-Image-2.1 \
  examples/requests/qwen21-viggle-runtime-lora-hybrid-512.json \
  --ane-manifest path/to/qwen-shared-base/manifest.json
```

把 `path/to/...` 替换成真实的匹配 manifest；同一命令将 `generate MODEL`
换成 `plan` 可先检查请求。`plan` 不加载权重，不能代替真实生成验证。
Z 示例也可原样用于 GPU `generate`；Qwen GPU 使用非 hybrid 示例。
若启用 Qwen GPU 优化组合，对照两路必须设置相同环境变量。
不要把 fastest base-only manifest 当成接受 pre-SiLU 修正输入的共图 manifest。
每个示例有固定输出名，重复或并行测试前请复制请求并指定独立输出。

## 代码与产物整理

- 保留 GPU/ANE 主路径、LoRA 完整计算和既有 optional 路由，不改变默认策略。
- Qwen profiling 开关复用 `diagnostic_options.hpp` 的严格 `1` 解析，移除重复判断；
  增加开关解析和便携 CLI 示例的测试。
- 将误入版本管理的 249 个 `results/z-image-*` 请求、PNG、导出/编译元数据及
  lock 文件移出 Git 索引；**本地文件全部保留**，不删模型、不清空正在使用的缓存。
  本轮不是磁盘回收，也不改写 Git 历史；历史对象仍存在。
- 新增忽略规则与回归检查，便携请求放 `examples/requests/`，结论放 `docs/`；
  保留其他不相关的 streaming 证据，不批量移除所有 results。
- 刷新状态索引并删除其中失效的历史链接，修正公开 Qwen 文档仍只介绍
  suffix-only LoRA 的过时描述。
- 无推理回归统一为 `make test-acceleration-contract`：调度、dense/affine
  转换、几何、benchmark 结果门禁及 CLI optional 契约。默认 `make test`
  复用该入口；真实 Core ML 微图仍显式运行，不成为普通测试的依赖。
- benchmark 门禁拒绝缺失、布尔值、浮点值、负数和批次内倒退的累计计数，
  防止遥测缺失/会话重置被误报为成功加速；合法的自适应 GPU 探测和
  `chunks=0` 消融保留。这是验证工具改进，不是推理性能优化。
- 两套 runtime 验证工具共用负载等待及实验环境清理，识别 Python 启动的
  常见推理任务；检查参数但不保存参数，避免把 token/提示词写入进程快照。
  共图切换在合成 adapter 前也等待；繁忙超时即退出，不终止其他任务。
  这仍是 CPU 活跃度启发式检查，不保证 GPU/ANE 独占。
- 整理阶段不再扩展内核或新实验；保留前述 Q8 接入和 dtype 正确性修正，
  不改变默认路由或已验证最快路径。不新增磁盘删除、不提交 commit，
  原有暂存状态保持不变。

## 验证入口与历史边界（2026-09-28）

以下保留前几轮构建的测试记录；当时的整理构建与 Q4 兼容修复见
[Q4 与回归记录](runtime-ane-q4.md)，当前维护决策见 [加速入口](acceleration.md)。

| 命令 | 覆盖范围 | 依赖与边界 |
| --- | --- | --- |
| `make test-acceleration-contract` | 4 项 host、18 项 benchmark/preflight、11 项 CLI 契约 | 不做模型推理；CLI 未构建则该组跳过 |
| `make test-runtime-ane-host` | 调度、SIMD、Q4/Q8 affine、图几何 | 无 Core ML Python SDK、无模型 |
| `make test-runtime-ane` | 4 项 host + 7 项图/集成 | 显式 Core ML/MLX 小图计算，需离线 SDK 与原生构建；含 v2 LoRA、Qwen full/split 缓存切换 |
| `make test-qwen21` | Qwen 模型/CLI/App 契约 | 需对应原生测试构建，不是画质或速度验收 |
| `make test` | 仓库边界、请求及各模块回归 | 部分缺失夹具/专用构建用例按既有门禁跳过 |

上一 `fb79…` 构建通过上述入口（包括全量 `make test`）、两模型
真实共图切换和 base 三路对照；详见上方 unsplit 报告。`f287…` 的验证范围
见 [完整块记录](runtime-ane-full-block-probe-2026-09-29.md#验证及边界)，
后续 `bc1…` 见 [周期采样记录](runtime-ane-periodic-sampling-2026-09-29.md#构建与验证)。以下保留上次整理
阶段的验证记录，不把旧构建测试次数与新构建混用。

上次整理阶段的原生库 SHA：
`ed7d8c8381e85b1b8be8f4d20fe0000e8bb984f638a8f7a586706838f33fdc9a`。
该次整理阶段未修改原生推理库；`ed7…` 构建重新通过 `make test-runtime-ane`
（4 host + 6 图/集成）、`make test-acceleration-contract`（4 + 16 + 11）
和 `make test-qwen21`。微图覆盖 v1/v2、完整 LoRA、headroom、失败 tail
重算、取消/drain、内存不足及 packed GPU FFN 对照；LoRA 合成用例最差
relative L2 为 0.00533939，Q4/Q8 用例为 0.00658136。
真实 Z/Qwen 共图切换已在同一构建通过，见 v2 记录；不等于编辑或任意
LoRA 的画质验收。旧 `a81…` 构建的 base/Q8 性能与测试记录保留在
[集成报告](runtime-ane-integration-2026-09-28.md)，不能冒充当前构建的整图回归。
随后 `make test` 全量回归成功退出；专用 audit/test-hook 构建、部分 GPU
opt-in 与缺失夹具的用例仍按既有门禁跳过，不计为已覆盖。链接阶段仍有
macOS deployment-target 警告，微图 SDK 有临时目录清理警告，均未导致失败。
85 处本地文档链接/锚点、路径独立性与 staged/unstaged 空白检查通过；
249 个此前移出 Git 索引的产物全部仍在本地，暂存状态未改动。
该次未重跑旧冻结图/base 的整模型性能验收，不外推默认路径在该构建中的
具体秒数；新证据仅包括 v2 既有 screen、接续完成的 chunks0 消融与上述测试。

该历史阶段的 runtime-weight 后端已新增可选 v2 LoRA，当时仍缺其性能与广泛质量资格、Q4 更广泛整模型验证、
Qwen 1024²/编辑、广泛视觉质量与完整内存/换页验收、不可变 artifact lease。
Z Q8_0 已有真实生成和性能筛选，但不能从单样本外推所有量化 checkpoint
或广泛质量。已有冻结 base 图 `lora_fused` 的可用性与这些
未完成项分开描述。后续1024² base、三图编辑及匹配性能已取得的证据，
以[当前维护入口](acceleration.md)为准，不将本段当作最新缺口清单。

### 复现实验

以下默认验证显式 runtime base 后端；v2 LoRA 对照可额外传 `--lora PATH`，
使用兼容的 v2 manifest 和对应步数。脚本拒绝未绑定/已合并/不完整图的 LoRA 成绩：

```sh
.venv/bin/python3 tools/validation/runtime_ane_model_screen.py \
  --model models/Comfy-Org-z_image_turbo --model-id z-image-turbo \
  --runtime-manifest path/to/z-runtime-c288/manifest.json \
  --output outputs/runtime-ane/new-abba --routes runtime,gpu,gpu,runtime \
  --steps 8 --warm-repeats 2 --chunks auto
```

每 trial 首次冷请求排除，四次热样本/路线；必须选新输出目录。
此脚本只做启发式进程检查、等待和计时，不能证明设备独占或自动批准画质。
观察 PNG 的主体、构图、细节与编辑指令是否符合预期，再决定允许的近似。

此前共图切换脚本已跑通 Z-Image 与 Qwen，Core ML 累计
调用数分别是 256/512/768/1024 与 160/320/480/640；适配器请求
分别绑定 238 与 227 个投影，首尾 base PNG 均逐字节相同。
这是旧冻结图的状态隔离回归，不是另一训练 LoRA 的质量或性能结论。
新 runtime v2 的真实共图切换调用数与视觉证据另见 [v2 记录](runtime-ane-lora-2026-09-28.md)。
