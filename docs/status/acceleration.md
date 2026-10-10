# GPU/ANE 加速：当前选择与维护入口

本文维护当前决策；完整实验与负结果见[状态索引](README.md)。性能必须绑定
构建、checkpoint、adapter、尺寸、步数、参考图与缓存状态，不能跨记录拼接。
当前实现层次、整理范围和已验收/未验收边界见
[2026-09-30 代码与进度](runtime-ane-current-2026-09-30.md)。

使用入口：[请求模板与选路](../../examples/requests/README.md)；
维护入口：[原生后端](../../native/backends/README.md)、
[验证工具](../../tools/validation/README.md)。最快base与可复用LoRA图是两种
选择，不将两者混称为一个“最快”方案。

## 当前进度摘要

2026-10-10 的512²接续以[状态索引](README.md)和各专项证据为准。
最新[完整性能进度摘要](performance-progress-2026-10-10.md)区分已测整请求、
kernel组件和默认选择；[首步share正反序](local512-qwen-prefill-shares-2026-10-10.md)
显示单ref4096/5120排名翻转、双ref4096有约4.4% GPU-relative整请求信号。
保持显式候选、cache off、KV-hit完整GPU；所有load checks失败，不默认推广。
[当前库同任务正反序base三路对照](local512-qwen-frozen-base-2026-10-10.md)
中，GPU/Private/frozen四热中位43.353/34.542/33.121s；两个CPU load
checks失败，frozen有系统compression/swap-out，保持显式诊断/默认不变。
六步LoRA生成仍优先GPU；1–2ref编辑可独立筛选首步FFN通道并行、
KV-hit完整GPU，不把base冻结图收益套给LoRA或encoder。

2026-09-30 接续更新；本页是决策入口，逐轮经过留在专项报告。

- **Qwen 1024² base-only v1 在最新同库对照中更快。** 875b 库的狐狸
  40步正反向三路各一冷一热、统一Q/K融合：GPU/v1/冻结图两热中位
  183.086/144.564/160.285 s，GPU/v1 为1.266×，冻结/v1为1.109×。
  狐狸和成人肖像 v2→v1→v1→v2 两组四热中位显示 v1 少约1.6–1.7%
  耗时；仅 base 可用。Z 的同几何组件含 staging 仅快约1.2%，未升为整
  请求候选。[完整记录](runtime-ane-v1-base-2026-09-30.md)保留两个冻结
  窗口 compression、两热样本与视觉边界。默认路线与 LoRA v2 不变。
- **Z 1024² BF16 base 已完成 v1/GPU 双提示词交错对照。** 同一 875b 库
  的灯塔/seed7 GPU/runtime 四热中位 31.429/28.400 s（1.107×），
  肖像/seed43 31.401/28.388 s（1.106×）；八个 trial 采样完整、无
  swap-in/out。图是 c352/K1024/N512 v1，仅 base，
  [完整记录](runtime-ane-z-1024-v1-2026-09-30.md)保留冷请求恢复、
  进程树内存、单样本视觉和缺少匹配 1024² 冻结图的边界。
  同库 c704 在组件快约4.4%，整请求却仅比 c352 快0.38%，不替换
  c352。512² 已测最快冻结路线与默认 GPU 不变。
- **Z Q8 GGUF 1024² base 的 runtime v1 也有同库收益。** 同一 875b 库，
  灯塔/肖像 GPU/runtime 四热中位分别为 40.891/35.198 s（1.162×）
  和 40.894/35.227 s（1.161×）；仅显式 optional。肖像的一次 GPU
  batch 有 56.7 MB compression，两次 runtime 输出并非逐字节相同；
  硬件放置仍未知。[Q8 对照](runtime-ane-z-gguf-q8-1024-2026-09-30.md)
  记录完整采样、重试和视觉限制；默认 GPU 与 512² 选择不变。

- **runtime 图加载使用已校验私有快照。** 消除逐文件哈希验证到 Core ML
  延迟加载原图之间的间隙；旧/新库两模型 512² ABBA 热态接近。
  新库 `28ff6aaf…` 的同组三路四热中位：Z GPU/runtime/冻结图
  6.997/6.399/5.347 s，Qwen 同样开启 Q/K 融合为
  41.744/35.592/29.062 s；runtime 相对 GPU 1.093×/1.173×，冻结仍最快。
  同库 Qwen 1024² 正反向三路四热中位为
  183.165/147.840/154.664 s，runtime 相对 GPU 1.239×、相对冻结
  1.046×；一个冻结窗口有系统压缩，不把它解释成物理 ANE 驻留证明。
  [安全快照与完整对照](runtime-ane-artifact-lease.md)区分旧库 A/B、
  新库三路分母、调度波动和系统 compression，默认选择不变。
- **默认不变，最快已测 base 路线保留。** 普通 GPU/既有受限 auto 不
  自动启用实验；已完成对照的3986库512²匹配冻结图的Qwen（双方开启Q/K融合）/Z
  加速为1.436×/1.310×。
- **不合并 LoRA 的共图功能已打通。** `lora_fused` 与 `runtime` v2
  支持 base/不同 runtime LoRA 复用图，保留完整激活、失败重算及状态隔离。
- **Qwen 六步生成/编辑仍优先 GPU。** 3986库三图 c1792 四配置对照完成，
  同样开启Q/K融合时GPU/runtime全部热请求为0.998×，较晚窗口为1.016×；
  此前8e库无融合为1.005×/1.018×，尚无显著普遍收益。
  旧库短 batch 为 0.977×，Z 已测 distill patch LoRA 为 1.105×。不能把 base
  或单层收益套给任意 LoRA、编辑或分辨率。
- **1024² base当前实测runtime最快。** 55e5库六配置正反向12 trial/36请求
  完成并独立复核；同样开启Q/K融合时GPU/runtime/冻结图为
  182.961/147.349/160.718 s，runtime相对GPU为1.242×、相对冻结图为1.091×。
  [完整记录](runtime-ane-qwen-qk-1024.md)保留内存、波动和单prompt视觉边界，
  不外推到编辑/LoRA；512²仍保留最快冻结图。
- **保留有证据的内部优化，实验显式 optional。** 稳定层异步 GPU head、
  就绪屏障合并、base hidden 免复制及 SIMD staging 保留；GPU-first
  未获足够收益已撤回，不给每个失败候选增加开关。
- **Z 短序列 QKV ANE 不进入产品路径。** 真实 BF16 fused QKV 权重、
  合成 1024-row 输入的 MatMul 组件八 trial 筛选中，最好的 512-row
  并行投影为 7.867 ms，同组 GPU-only 6.281 ms；含暴露 staging 为
  8.690 ms。四种分区均未越过 GPU，对 Qwen 长序列不作推断。
  [QKV 组件筛选](runtime-ane-qkv-component.md)保存真实权重来源、数值和
  完整计时边界；Z 的产品 QKV/默认路由不变。
- **Qwen QKV 已接入显式研究路由，不作为当前优选。** 1024² BF16 base
  的 `runtime_qkv` 直填 Q/K/V 权重、GPU FFN 保持原路径。最新 875b 库
  一冷一热正反向四路线，统一 Q/K 融合：GPU/QKV auto/FFN runtime/
  冻结图两热均值 183.077/182.961/147.001/151.766 s；QKV auto 相对
  GPU 约 1.001×，远非稳定收益。auto 仅在固定一 chunk 和完整 GPU block
  间开关，不调 chunk 数；一个 prompt、两热和物理放置不足以升级默认。
  [当前对照](runtime-ane-qwen-qkv-auto.md)与
  [早期固定路线初筛](runtime-ane-qwen-qkv-product.md)分开保留。
- **可选 runtime 的请求级内存准入已接入。** 图加载前、resident 请求和
  host scratch 扩容前重查，低余量安全释放后 GPU 重算；两模型 512²
  和 Qwen 1024² 的正常内存烟测通过。最终 `f3f6b6a0…` 库又完成
  Qwen 1024² 六 trial 正反向对照：同组 GPU/runtime/冻结图四热中位
  183.146/147.429/151.966 s，runtime 相对 GPU 1.242×；冻结图窗口
  有系统压缩。它不是完整内存资格，详情见
  [准入接续](runtime-ane-memory-admission-2026-09-29.md)。
- **第二个 base 提示词已完成两模型正反向验证。** 相同 f3f6 库的成人
  肖像/seed43，Z 512² GPU/runtime/冻结图四热中位 7.320/6.501/5.506 s，
  Qwen 512²同样统一开启 Q/K 融合为 41.762/35.610/29.076 s；冻结仍最快。
  [肖像记录](runtime-ane-portrait-and-z-chunk-2026-09-29.md)也保存 Z c416/N512
  单层变慢的负结果；不把本次组件候选升级为整模型优化。
- **Qwen 1024² 第二提示词也完成三路反向对照。** 同一 f3f6 库、肖像/seed43
  且统一开启 Q/K 融合，GPU/runtime c1792/冻结图四热中位为
  183.156/148.011/152.866 s；runtime 相对 GPU 为 1.237×、相对冻结图
  为 1.033×。[完整记录](runtime-ane-qwen-1024-portrait-2026-09-29.md)
  保留两个冻结窗口冷请求期间的系统压缩和视觉样本限制；不升级默认。
- **1024² 大 chunk 的整请求收益已验证。** [c1792/c320 交错对照](runtime-ane-qwen-chunks.md)
  四 trial、12 请求完成，热请求 165.000 → 153.050 s（1.078×，耗时少
  7.24%）。保留为显式 optional；没有同组 GPU/冻结图结果，不替换三路表，
  不外推到 LoRA/编辑或 512² 短序列。
- **c1664 不替换 c1792。** 同一 f3f6 库、Qwen 1024² BF16 base、Q/K 融合开的
  [正反向四 trial](runtime-ane-qwen-c1664-2026-09-29.md)：c1664 组件并行
  FFN 快约 2.9%，整请求四热中位却为 c1792/c1664 的 148.282/148.804 s。
  小差异不支持升级候选；不拼接其他三路 campaign 的 GPU/冻结分母。
- **真实三图共图切换已补齐。** 修复 ref512 batch 对 runtime base 的
  校验拒绝；新库实测 base → A → 合成 B → base 复用 c1792 图，实际
  每请求 32 次 prediction，返回 base 字节一致；见[编辑记录](runtime-ane-qwen-edit-chunks.md)。

已完成512²资格的库为 `3986e035d805b559c0c3581afb46d772805428393df1d9a3d715d1532eab3fb4`，
支持既有 Q/K norm-RoPE kernel 与显式 runtime 路线组合；不改权重、完整
SiLU、LoRA精度或默认设置。构建、专项与完整回归已通过。
[同库三图72请求及两模型base对照](runtime-ane-qwen-qk.md)已完成并独立复核；
Qwen base共12 trial/36请求，开启融合后runtime为1.170×、冻结图1.436×。
此前1080库独立保存，8d90/e55/8e历史成绩仍归属于原构建。
接续构建 `55e5e736…` 仅将同kernel的门禁扩展到1024² resident base
文生图，构建/专项/完整回归已通过；[1024²六配置对照](runtime-ane-qwen-qk-1024.md)
已完整结束并独立复核，六配置代表图肉眼接近；3986库另存，512²成绩
不重标为55e5。两个尺寸的最快路线不同，不自动切换默认。
代码边界见[后端维护说明](../../native/backends/README.md)，整理与验证见
[收尾记录](acceleration-cleanup.md)。

## 结论与保留级别

| 级别 | 保留路线 | 使用原则 |
| --- | --- | --- |
| 默认与512²已测最快 base | 普通 GPU、既有受限 `auto`、匹配冻结 base 图 | 默认不变；冻结图显式匹配 checkpoint/shape |
| 完整 runtime LoRA，共用 base 图 | `lora_fused` + base-only manifest | optional；Z 有历史收益，Qwen 尚无可靠整请求优势 |
| 跨层复用 runtime-weight 图 | `runtime` v1/v2、token-row 切分、多 chunk、自动调度 | optional；v1 仅 base，v2 支持 BF16 base/LoRA 共图 |
| 1024² base 长序列 | base-only runtime v1 c1792/K1024/N512，auto chunks、可选Q/K融合；LoRA仍用v2 | 875b同库同融合相对GPU 1.266×、冻结图1.109×；仅狐狸两热三路，不外推base收益到三图LoRA |
| Z 1024² BF16 base | runtime v1 c352/K1024/N512、auto chunks，仍显式选择 | 875b同库灯塔/肖像相对GPU 1.107×/1.106×，无匹配冻结图分母；512²另有更快冻结图 |
| Z Q8 GGUF 1024² base | runtime v1 c352/K1024/N512、auto chunks，仍显式选择 | 875b同库灯塔/肖像相对GPU 1.162×/1.161×；肖像 runtime 正反序 PNG 不同，硬件放置和更广质量待验 |
| Qwen 1024² base QKV 研究 | `runtime_qkv`，三份独立 Q/K/V 直填 MatMul 图，GPU FFN；auto 仅一 chunk 开/关 | 875b库 Q/K 融合开、40步正反序两热相对GPU约1.001×；更广画质与物理放置待验 |
| 特定工作负载或近似优化 | Z direct-FP16、Qwen GPU Metal/LoRA FP16、DBCache | 分别启用、分别验证，不组成未经测试的“最快”预设 |
| 兼容/诊断 | `lora_gate_up`、`lora_suffix`、`lora_merged`、固定 chunks/profile | 不自动推荐；suffix 不完整，merged 不符合本任务共图要求 |

`lora_fused` 的 fused 指 **base FFN 算子融合**，不是 LoRA 权重融合。
不得把 LoRA 合入 checkpoint、Core ML artifact 或 runtime base slots。
GPU gate/up 修正必须进入 SiLU 之前，down-LoRA 必须使用修正后的 hidden；
base 请求传零修正。同一图 base → A → 合成 B → base 已做真实状态检查，
无需重载图，返回 base 一致；合成 B 不代表第二个训练 LoRA 的质量资格。

## 性能快照与口径

M4 Max 64 GB、resident，时间为整请求墙钟，含 VAE/PNG、排除冷请求。
下面每行只在同一构建内比较；不同步数/模型/LoRA 的绝对耗时不可横比。

| 工作负载 | 构建 | GPU | runtime 图 | 冻结 base 图 | 加速比 GPU/runtime；GPU/冻结图 |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen base，1024²，40步，Q/K融合开，v1 base-only两热正反向 | `875b0d3d…` | 183.086 s | 144.564 s (v1) | 160.285 s | 1.266×；1.142× |
| Z BF16 base，1024²，8步，灯塔/seed7，v1四热交错 | `875b0d3d…` | 31.429 s | 28.400 s (v1) | 不适用 | 1.107×；不适用 |
| Z BF16 base，1024²，8步，肖像/seed43，v1四热交错 | `875b0d3d…` | 31.401 s | 28.388 s (v1) | 不适用 | 1.106×；不适用 |
| Z Q8 GGUF base，1024²，8步，灯塔/seed7，v1四热交错 | `875b0d3d…` | 40.891 s | 35.198 s (v1) | 未测 | 1.162×；未测 |
| Z Q8 GGUF base，1024²，8步，肖像/seed43，v1四热交错 | `875b0d3d…` | 40.894 s | 35.227 s (v1) | 未测 | 1.161×；未测 |
| Z base，512²，8步，私有图快照后同库复测 | `28ff6aaf…` | 6.997 s | 6.399 s | 5.347 s | 1.093×；1.309× |
| Qwen base，512²，40步，Q/K融合开，私有图快照后同库复测 | `28ff6aaf…` | 41.744 s | 35.592 s | 29.062 s | 1.173×；1.436× |
| Qwen base，1024²，40步，Q/K融合开，私有图快照后同库复测 | `28ff6aaf…` | 183.165 s | 147.840 s | 154.664 s | 1.239×；1.184× |
| Qwen base，512²，40步，Q/K融合关 | `3986e035…` | 42.678 s | 36.685 s | 30.224 s | 1.163×；1.412× |
| 同组Q/K融合开（GPU与两条混合路线统一设置） | `3986e035…` | 41.742 s | 35.675 s | 29.068 s | 1.170×；1.436× |
| Z base，同构建复测，512²，8步 | `3986e035…` | 6.994 s | 6.343 s | 5.340 s | 1.103×；1.310× |
| Qwen base，1024²，40步，c1792，Q/K融合关 | `55e5e736…` | 187.816 s | 153.789 s | 163.514 s | 1.221×；1.149× |
| 同组Q/K融合开（全部路线统一设置） | `55e5e736…` | 182.961 s | 147.349 s | 160.718 s | 1.242×；1.138× |
| Z distill patch LoRA，512²，8 步，收尾原版复测 | `e55b1a0…` | 8.584 s | 7.766 s | 不适用 | 1.105×；不适用 |
| Qwen Viggle v0.2.1 r256，512²，6步，三张ref512，FP32，c1792，Q/K融合关，热1–8各16样本 | `3986e035…` | 12.906 s | 12.855 s | 不适用 | 1.004×；不适用 |
| 同组Q/K融合开，热1–8各16样本 | `3986e035…` | 12.666 s | 12.690 s | 不适用 | 0.998×；不适用 |

28ff 的 512² 两行各六 trial/18 请求、1024² 一行六 trial/18 请求，
均为 GPU→runtime→冻结→反向匹配对照，每路线四热池化中位，独立内存采样。
512² Qwen 两个冻结窗口、1024² 一个冻结窗口有系统压缩。
对应旧库 A/B 与新库原始证据见[图快照记录](runtime-ane-artifact-lease.md)，
不要将旧库 `3986`、`f3f6` 的 GPU/冻结结果作为 28ff 的分母。

3986组预声明热6–8中，融合GPU/runtime为12.672843/12.474506 s（1.016×）；
融合对两条路线各有小幅收益，不等于提高混合相对优化GPU的加速比。
全72请求、9,553独立内存样本与四配置图片复核；两trial各64KiB系统swap-in，
蓝壶透明材质仍共同失败，详见[完整组合报告](runtime-ane-qwen-qk.md)。

表中 base 为三路线正反向对照，Z LoRA 为 ABBA，每配置全部四热样本。
Qwen base另将Q/K开/关组成六配置正反向，完整36请求和13,498内存样本
已重验；一trial有64KiB系统swap-in，六配置末次热图片主体构图接近。
三图为四配置正反向对照，每配置十六热样本，保留自动调度的 GPU 探测请求。
正式计时均关闭 profile；上表各 campaign 使用100 ms独立内存采样。
此前短 batch、旧构建长 resident 与 FP16 低秩的详细结果分别保留在
[编辑 chunk 报告](runtime-ane-qwen-edit-chunks.md)、
[精度实验](runtime-ane-qwen-lora-rank.md)和[历史整理](acceleration-cleanup.md)，
不重复列作当前成绩，也不跨 campaign 拼接加速比。

55e5的1024²共36请求、60,420独立内存样本，最大间隔120.309 ms；runtime
无失败/重试/错误回退。部分trial有系统swap-in，冻结trial2约522.8 MB
compression、trial8约229 KB；不是全组零压力，也不是ANE内存归因。
六张反向末次热图的狐狸、松枝、雪景与色调接近，有毛发/耳形等局部变化。
完整口径见[1024²报告](runtime-ane-qwen-qk-1024.md)。e55的旧三路
187.800/168.465/163.665 s移回[历史对照](runtime-ane-qwen-1024-v2.md)，
不再作为当前1024²性能行。

1024² base 的独立 c1792/c320 ABBA 为 **153.050/165.000 s（1.078×）**，
各四热样本；该组没有 GPU/冻结图分母，不替换上表三路成绩。图像、少量
系统 swap-in 与采样资格详见[完整记录](runtime-ane-qwen-chunks.md#整模型对照已完成)。

内存报告包含加载/冷/热/退出，不等于 ANE 独占内存或完整 driver/wired
资格。[512² 报告](runtime-ane-matched-memory.md)记录末个 GPU trial 的
少量系统 swap-in；[1024² v2 报告](runtime-ane-qwen-1024-v2.md)重验
31,690 样本，记录少量 swap-in 和正向冻结图的约 306.5 MB compression。
不能称全组零换页，也不将系统计数全部归因于 ANE。旧 v1 单热样本初测
单独保留在[历史记录](runtime-ane-qwen-1024.md)，不混入当前统计。

已查看 512² 和上述 1024² base 对照代表图片，主体与构图接近；
收尾 LoRA 复测没有新增视觉验收。Qwen 两路都未很好保留蓝壶
透明材质，不能称所有编辑指令完成。质量按肉眼主体、参考身份、构图和
编辑意图判断，不要求逐像素接近；有限性/异常检查与返回 base 的状态
隔离测试仍保留。单一样本相近不代表所有提示词和适配器通过。

## 为什么共图与单层收益不等于整请求收益

冻结图主要沿 FFN intermediate channels 切分，权重已在图内；长序列可
循环固定 token 行数的同层图。`runtime` 沿 token rows 分配 GPU/ANE，
同形状图跨层复用，但增加逐层 base 权重 staging。

完整 LoRA 还带来 GPU 修正 → Core ML 激活/矩阵计算 → GPU down-LoRA 的
依赖、同步与 hidden 传输。Qwen 编辑首步长 prefill 可以循环多 chunks，
后续短 decode 不一定划算；attention/VAE 也不会随 FFN 同倍提速。
adapter SHA 校验在六步短请求中不可忽略，但不通过删校验换取性能。
细分证据见[编辑诊断](runtime-ane-qwen-edit.md)和
[完整 block/校验成本](runtime-ane-full-block-probe-2026-09-29.md)。

Q4/Q8 runtime staging 转为 FP16，**不是 INT8 ANE compute**；
`cpuAndNeuralEngine` 也不能证明实际 ANE residency。没有硬件执行证据时，
不把测得收益归因于 INT8 峰值算力。

## Optional 的明确入口

| 入口 | 默认与兼容边界 |
| --- | --- |
| `--hybrid-mode lora_fused --ane-manifest MANIFEST` | 匹配的 base-only 冻结图；base/不同 LoRA 共用，不与普通 base 图混用 |
| `--hybrid-mode runtime --ane-manifest MANIFEST` | resident、显式近似；BF16 Qwen/Z LoRA 需 v2 `--lora-inputs`；GGUF 仅 base |
| 导出 `--rows 1792 --tile-k 1024 --tile-n 512` 的 Qwen v1 图（不传 `--lora-inputs`） | 875b同库1024² base实测最快，仍显式optional；仅无 adapter，请求的chunk行数不是chunks数量 |
| 导出 `--rows 352 --hidden 3840 --width 10240 --tile-k 1024 --tile-n 512` 的 Z v1 图（不传 `--lora-inputs`） | 1024² BF16 base相对同库GPU两个提示词均过5%初筛，显式optional；512²冻结图不用于1024² |
| 为 Qwen LoRA 导出相同几何、额外传 `--lora-inputs` 的 v2 图 | 保留完整 gate/up 修正和 hidden 输出；base可复用但本轮base略慢于v1，512²短序列仍可能回到GPU |
| `TURBOCIDER_RUNTIME_ANE_CHUNKS=auto` | 仅 runtime 内默认自适应；不会把普通 GPU 请求改为 runtime |
| `TURBOCIDER_RUNTIME_ANE_CHUNKS=0...128` | 固定分区诊断；0 保留 split GPU 边界，不等于普通 GPU 基线 |
| `TURBOCIDER_RUNTIME_ANE_PROFILE=1` | 默认关闭；正式计时不设置 |
| `TURBOCIDER_Z_RUNTIME_LORA_DIRECT_FP16=1` | 默认关闭；仅 Z `lora_fused` 的近似选项，Qwen 不使用 |
| `TURBOCIDER_QWEN21_VIGGLE_LORA_FP16=1` | 低秩乘法近似，默认关闭；三图 GPU 小幅受益，runtime auto 未受益，base 图不变 |
| `TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE=1` | 512² BF16 GPU/受限冻结图/runtime已测，base runtime自身1.028×；1024² resident base文生图已测，自身1.044×；1024²编辑/LoRA不支持，不自动启用 |
| DBCache、参考缩图、FFN 复用 | 沿用各自已验证范围，不套用到任意六步 LoRA/编辑 |

[CLI 使用](../public/USAGE.md)、[便携请求与图准备](runtime-lora-acceleration-2026-09-28.md#便携-cli-示例)
提供具体命令。manifest 从命令行传入；示例不包含机器绝对路径。
`plan` 只检查契约，不代表权重/图兼容、实际生成、速度或质量已验证。
base 可选 tile 为 K1024/N512，Qwen c320、Z c352；它们没有证明能让
Qwen LoRA 编辑超过 GPU，详见 [tile/chunk 筛选](runtime-ane-tiles.md)。

## 代码维护与验证

| 代码 | 保留职责 |
| --- | --- |
| `native/backends/coreml.mm`、模型 `hybrid.cpp`/`z_image.cpp` | 已测最快冻结图与完整 LoRA 激活接口 |
| `native/backends/ane_runtime.{hpp,mm}`、`ane_runtime_{convert,quant}.hpp` | runtime 图/worker/slots、SIMD、dense/affine staging、有限性检查 |
| `native/backends/ane_scheduler.hpp`、`ane_ffn.{hpp,cpp}` | token-row 调度、就绪屏障、张量生命周期、失败后完整 GPU 重算 |
| `native/backends/ane_memory.{hpp,cpp}` | Mach观测、可选内存准入和host扩容预算；不是完整driver/wired认证 |
| 模型 `*_module.cpp`、CLI、`results.mm` | 显式路由、不兼容组合拒绝、真实调用/回退指标 |
| `native/models/z_image/padding.hpp` | GGUF 浮点 padding 形状规范化；拒绝 packed 常量 |
| `tools/coreml/export_runtime_ane.py`、`tools/native/ane_runtime_probe.cpp` | optional 离线导出/单层探针；生成时无新增 Python 依赖 |
| `tools/validation/runtime_ane_model_screen.py`、`runtime_lora_shared_graph_switch.py` | 可复用整请求对照、共图切换、竞争推理预检 |
| `tools/validation/runtime_ane_common.py` | 标准库遥测/编辑 receipt 校验、预检、流式文件哈希；不导入 runner、不加载模型 |
| `tools/validation/runtime_ane_memory.py` | optional 独立进程采样/验证、证据绑定及仅限自有进程组的超时清理 |
| `tools/validation/qwen21_ane_placement.py` | 冻结/runtime离线设备计划、artifact身份与嵌套算子校验；不是物理ANE trace |

保留稳定层异步 head、readiness 合并、base 未使用 hidden 只校验不复制、gate/up
分段简化和 alpha 类型修复。输出直接写入、更大起始分区、更高调度门槛、
GCD 四组转换等未获收益的候选均已撤回，回归测试和负结果保留在
[实验索引](README.md)；不为每个失败候选维护一个开关。

整理阶段撤回尚无端到端证据的整数输出转换分支，保留 65,536 位模式 ×
13 个 headroom 的穷举测试。当前概览与 CLI 文档不再重复堆叠历次跑分；
原始请求/PNG/日志仍放在被忽略的 `outputs/`、`results/`，不删除模型、
缓存、用户产物或历史报告。既有 249 项暂存移除及本地文件不动，不改暂存区。

```sh
TURBOCIDER_NATIVE_ONLY=1 make build
make test-acceleration-contract  # host / 报告 / CLI，无真实模型推理
make test-runtime-ane            # 显式 Core ML/MLX 微图与集成
make test-qwen21
make test
```

当前55e5候选在启动1024²对照前，构建、Core ML微图、91项加速契约、
Qwen专项、扩展Q/K kernel及完整 `make test` 已通过，日志见
[1024²验证记录](runtime-ane-qwen-qk-1024.md)。3986的性能和测试保留在
[512²组合验证](runtime-ane-qwen-qk.md)，不重标为当前候选的测量。
本轮整理未修改benchmark绑定源码、工具、库或图，也未重建运行库；
完整对照结束后重跑91项加速契约、8项布局、7项独立性检查。新增便携1024²
GPU模板及对应路径回归，不修改执行数学或默认选路；历史完整回归不写成本轮
重跑。验证工具职责与测试入口见
[工具维护说明](../../tools/validation/README.md)。缺夹具、专用
构建或 GPU opt-in 的 skip 不算覆盖。运行 benchmark 时不并行构建/测试，
预检繁忙则等待，不终止他人进程；不以微图回归代替真实编辑/LoRA 质量验收。

## 未完成，不作为已验收能力

1024² 编辑/LoRA 与更多 prompt 验收、一/两图编辑与三图加速优化、多训练
LoRA/多提示词视觉检查、Q4_1 整模型与 Q4_0 更广覆盖、完整 driver/wired
内存与压力验收、物理 ANE residency。runtime 图加载已有已校验的私有
快照（见[图快照回归](runtime-ane-artifact-lease.md)），但其它工件/整个
外部驱动的不可变归因未覆盖。双缓冲、QKV 分区优化及与 FFN 联合调度、
runtime INT8 仍属后续研究；QKV 单独 on/off 不等于这些门槛完成。
当前实测只覆盖上述工作负载，不宣称这些研究目标已完成。
内存方面已接入机会性请求准入和host scratch扩容保护，见
[实施与两模型复测](runtime-ane-memory-admission-2026-09-29.md)；原
[准入代码审查](runtime-ane-memory-admission-audit.md)是接入前快照。
不将这项保护或低预算回退测试视为完整内存验收。
