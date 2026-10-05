# TurboCider 状态文档入口

## 当前加速结论

以下 Private ANE 接续记录按实验时间归档；构建哈希和“未提交”等说明描述
当时快照。组件测试或单向初筛通过，不等于产品默认路线或正式性能/质量验收。

2026-10-03 双后端接续见
[Private ANE 基础执行器](private-ane-foundation-2026-10-03.md)。这是原生
私有同步/共用 Executor 的阶段实现，尚非 W8A8 或四格性能目标验收。
当前接续实现与完整请求负结果见
[GPU IOSurface I/O](private-ane-gpu-io-2026-10-03.md)。
GPU 权重转换与实机 W8A8 MatMul 组件见
[W8 stager](private-ane-w8-stager-2026-10-03.md)。完整 SwiGLU、统一 W8A8
Executor、两套银行和模型初筛接续见
[W8A8 Executor](private-ane-w8-executor-2026-10-03.md)；层间预取、性能和
完整质量验收尚未完成。
已集成的 intermediate-channel 分区与完整 down-LoRA join 接续见
[Private channel split](private-ane-channel-split-2026-10-03.md)。原任务四格
≥1.2×、层间 staging 重叠和画质资格仍未验收。
下一层 source-matched bank、独立 worker/queue、lease 与双 reuse fence 的
实现和 matched prefetch 负结果见
[Private weight prefetch](private-ane-prefetch-2026-10-03.md)。默认 Public 不变。
紧凑 FP16 scale cache 与 generation/stride/recipe 失效、同库性能接续见
[Private scale cache](private-ane-scale-cache-2026-10-04.md)，Z512 初筛约1.080×，
未达到原任务四格1.2×。
有界 A8 双槽、不可变符号元数据、Z512 channel share 搜索与 Z1024 对照见
[A8 lookahead 接续](private-ane-a8-lookahead-2026-10-04.md)。Z512 新库初筛
约1.155×、Qwen512/Fa5120 初筛1.212×，Z1024 lookahead 后仍慢于 GPU；
单格单向初筛不等于四格性能与完整画质验收，原目标仍 active。
1024² 的较大固定 bucket、私有行数边界和 fallback 标签修正见
[大 bucket 接续](private-ane-large-bucket-2026-10-04.md)：Z1024约1.120×、
Qwen1024约1.187×，仍未达到要求的四格正式1.2×。
W8 Metal format/dtype/block 专用化、逐位回归和连续负载观察见
[staging 专用化](private-ane-stage-specialization-2026-10-04.md)。组件有正信号，
三次完整请求比较被 competing CPU load 拒绝，未填新的有效端到端倍率。
对齐读取、逐位回归、GPU dual-head 的近乎持平结果和新库512²/1024²
兼容检查见 [dense-load 接续](private-ane-dense-load-2026-10-04.md)。新的
整请求 arm 仍被 CPU load gate 拒绝，不更新四格正式倍率或默认路线。
共用 channel LoRA 子范围 callback、实际执行 counters 和 Z/Qwen512
新旧 LoRA 逐位兼容见 [LoRA range 接续](private-ane-lora-range-2026-10-04.md)。
四个 arm 的 timing 仍被 load verifier 拒绝，不以输出缩小代替端到端收益。
固定分区去测量fences的同库开关、row/channel异常所有权回归与Z512/1024
数值兼容见 [fixed async](private-ane-fixed-async.md)。默认关闭；host async
counters不证明device overlap，两个runtime arm仍无有效速度比较。
负载触发PID的只读 argv 复核见 [load origin](private-ane-load-origin.md)：
最近捕获到 `download_ltx25.py`，不能表述为已证明ComfyUI GPU推理争抢，
也没有放宽原 gate 或追认历史速度。
用户授权的临时安静窗口及同库对照见
[quiet window](private-ane-quiet-window-2026-10-05.md)：Qwen512 base v1 两方向
约1.263–1.265×；Z512 中位约1.20×但裕量窄，两个1024²格仍未达标。正式四格、
LoRA/质量/内存/device trace资格未完成，不升级默认路线。
1024²显式4224-row bucket、逐元素self-test与长行数physical MPP接续见
[full bucket](private-ane-full-bucket-2026-10-05.md)：新库Z1024狐/灯塔约
1.214×/1.209×，Qwen1024约1.258×；保守边界亦超过1.2，但同提示词
反序、最新库512/LoRA、完整质量/内存/并发与带宽calibration仍待验收。

先读 [GPU/ANE 加速：当前选择与维护入口](acceleration.md)。
它统一维护默认/最快 base、完整 runtime LoRA、optional 和诊断边界。
512² base 保留已测最快冻结图；最新Qwen三图对照中，同样开启Q/K融合的
GPU/runtime全热为0.998×、预声明较晚窗口1.016×，不改变默认选择。
1024² base六配置历史对照已完成；最新同库 base-only v1 正反向复测中，
同融合设置下 runtime 为1.266× GPU、1.109×冻结图，此工作负载实测最快，
仍显式 optional；LoRA 继续用 v2。
同一 875b 库的 Z-Image 1024² BF16 base 两提示词正反向对照，runtime v1
相对 GPU 分别为 1.107×/1.106×；该尺寸没有匹配的冻结图分母，不外推到
512² 或 LoRA。
同库 Q8 GGUF 1024² base 两提示词的 runtime v1/GPU 为 1.162×/1.161×；
仅显式 optional，硬件 ANE 驻留与更广泛质量仍未确认。
默认路由不变，LoRA 不合入 base checkpoint、Core ML artifact 或 runtime 权重槽。

性能必须按构建、checkpoint、步数、尺寸、参考图和缓存状态对照；
历史报告中的“更快”不自动覆盖当前决策。详细命令与共图原理保留在
[GPU/Core ML 与 runtime LoRA 操作指南](runtime-lora-acceleration-2026-09-28.md)。

## 关键实验与证据

| 内容 | 入口 |
| --- | --- |
| 当前保留级别、性能表、optional、代码职责和未完成项 | [维护入口](acceleration.md) |
| 2026-09-30 runtime ANE 代码层次、BF16/Q8/QKV 实测进度、整理范围与提交边界 | [当前代码与进度](runtime-ane-current-2026-09-30.md) |
| 875b库 Qwen 1024² base-only v1：两提示词 v1/v2 ABBA 与 GPU/v1/冻结图正反向三路；Z v1/v2 组件收益不足 | [v1 base 筛选](runtime-ane-v1-base-2026-09-30.md) |
| 875b库 Z 1024² BF16 base：灯塔/肖像双提示词 GPU/runtime v1 正反向；c704 组件正信号未转成明显整请求收益 | [Z 1024² v1 对照](runtime-ane-z-1024-v1-2026-09-30.md) |
| 875b库 Z Q8 GGUF 1024² base：灯塔/肖像双提示词 GPU/runtime v1 ABBA、内存与视觉检查 | [Z Q8 1024² 对照](runtime-ane-z-gguf-q8-1024-2026-09-30.md) |
| runtime compiled graph 私有快照：加载后源图移动的真实预测回归、旧/新构建 ABBA 与新库512²/1024²三路匹配对照 | [图快照与回归](runtime-ane-artifact-lease.md) |
| Z 短序列真实 BF16 fused QKV：四种 ANE row 分区均慢于 GPU-only，不接入产品 QKV tier | [QKV 组件负结果](runtime-ane-qkv-component.md) |
| Qwen 1024² base 显式 `runtime_qkv` 产品接入：早期固定一 chunk 单热 1.013×，非当前完整对照 | [QKV 产品初筛](runtime-ane-qwen-qkv-product.md) |
| 875b 库 Qwen 1024²/40步 QKV auto 正反序四路线：相对 GPU 约 1.001×；auto 只决定一 chunk 开/关，仍不默认启用 | [QKV auto 完整请求](runtime-ane-qwen-qkv-auto.md) |
| Qwen 长序列真实 BF16 Q 权重：单投影组件在 1024–1536 ANE rows 呈正信号；不能代替组合 QKV 或产品 block | [Q 投影组件筛选](runtime-ane-qwen-q-projection.md) |
| Qwen 三份真实 Q/K/V 权重在计时前打包，完整组合投影组件相对两种 GPU-only 对照均有正信号；仍缺产品 block/内存资格 | [QKV 组合投影筛选](runtime-ane-qwen-qkv-packed.md) |
| Qwen Q/K/V 三份独立 checkpoint 权重直接填入单个 runtime MatMul 槽，免常驻打包副本；组件对照 1.411×，产品 block/联合内存仍待验 | [QKV 多源直填](runtime-ane-qwen-qkv-parts.md) |
| Qwen 1024² base：55e5六配置36请求完成，Q/K融合runtime相对GPU 1.242×、冻结图1.091×；内存和单样本视觉已复核 | [1024² GPU融合组合](runtime-ane-qwen-qk-1024.md) |
| GPU/冻结base/共图LoRA/runtime模板与命令，含1024²实测最快可选组合 | [请求示例索引](../../examples/requests/README.md) |
| Qwen Q/K融合：三图仍无明显混合优势；base36请求完成，同样优化GPU后runtime1.170×/冻结图1.436×；Z复测与离线设备计划 | [GPU融合组合](runtime-ane-qwen-qk.md) |
| 输出恢复局部3.69×，但Qwen/Z整请求无增益；撤回并行分支，保留布局/有限性回归 | [输出恢复候选](runtime-ane-output-restore-parallel.md) |
| 后端分层、LoRA/异步生命周期约束、保留代码与离线工具边界 | [代码维护说明](../../native/backends/README.md) |
| 可复用验证工具、共用校验、一次性实验及运行中代码冻结边界 | [验证工具维护说明](../../tools/validation/README.md) |
| 请求级内存准入/host scratch扩容与两模型同构建复测；完整driver/wired/低内存资格仍缺 | [内存准入接续](runtime-ane-memory-admission-2026-09-29.md) |
| f3f6库第二提示词肖像：Z/Qwen 512²三路正反向对照；Z c416/N512组件负结果 | [肖像与Z chunk筛选](runtime-ane-portrait-and-z-chunk-2026-09-29.md) |
| Qwen 1024² c1664/c1792：单层 c1664 快约2.9%，同库整请求反而慢约0.35%；保留 c1792 | [c1664 整请求筛选](runtime-ane-qwen-c1664-2026-09-29.md) |
| f3f6库 Qwen 1024² 第二提示词肖像：六 trial 三路同构建对照，runtime 1.237× GPU、1.033× 冻结图；冷图压缩边界保留 | [1024² 肖像对照](runtime-ane-qwen-1024-portrait-2026-09-29.md) |
| 内存准入接入前只读审查、原始缺口与隔离草稿 | [内存准入旧审查](runtime-ane-memory-admission-audit.md) |
| e55 库 Qwen 1024² v2 正反向对照完成：runtime 1.115×、冻结图 1.147×，含波动/换页边界 | [1024² v2 对照](runtime-ane-qwen-1024-v2.md) |
| Qwen 1024² base c1792/c320 交错对照完成：165.000 → 153.050 s，1.078×，保留为 optional | [Chunk 与分区](runtime-ane-qwen-chunks.md) |
| 三图短 batch 0.977×；新库长 resident 全热1.005×/较晚1.018×；ref512 base/A/B/base 共图通过 | [编辑 chunk 与共图](runtime-ane-qwen-edit-chunks.md) |
| 本次收尾范围、代码/文档整理与实际验证结果 | [整理记录](acceleration-cleanup.md) |
| LoRA GPU-first 初始采样负结果；Qwen 成本后移、Z 小幅变化不足以推广，恢复原顺序 | [初始采样消融](runtime-ane-gpu-first.md) |
| e55 库原版 LoRA 收尾复测：Qwen 三图 0.922×、Z 1.105×；后续候选的原版基线 | [本次收尾](acceleration-cleanup.md#本次整理) |
| e55 库 Qwen 三图 FP32/FP16 低秩交错对照：固定分区改善，auto 仍不如 GPU；精度标记修复 | [LoRA 低秩精度](runtime-ane-qwen-lora-rank.md) |
| 零修正输入复用只完成微图与原版 before；撤出候选，保留状态隔离回归 | [零输入候选](runtime-ane-zero-input.md) |
| 历史 8d90 库 base 三路双向对照、独立进程采样与换页边界 | [匹配性能与内存](runtime-ane-matched-memory.md) |
| 整数输出转换仅完成微基准，撤出产品路径，保留穷举回归 | [输出转换候选](runtime-ane-output-convert.md) |
| 最新稳定层异步 head、base 前后筛选、LoRA ABBA、计时语义与共图回归 | [异步 head](runtime-ane-async-join.md) |
| 此前 LoRA 屏障合并、固定分区前后对照、两模型 auto ABBA 与共图切换 | [LoRA readiness](runtime-ane-lora-ready.md) |
| 此前 b554 同库 GPU/v2/冻结图双向对照：共图 base 1.155× / 1.086×，冻结图仍最快 | [v2 匹配对照](runtime-ane-v2-matched.md) |
| 转换任务归并为最多四组的负结果；原版恢复，未改产品库 | [Staging 任务粒度](runtime-ane-staging-partition.md) |
| v2 base 不复制未使用 hidden，等价有限性检查与两模型 LoRA 切换 | [v2 base 与 hidden 处理](runtime-ane-v2-base.md) |
| 图内 tile + token chunk 联合调优，两模型 base 同构建 GPU/runtime/frozen 对照 | [Tile 筛选](runtime-ane-tiles.md) |
| 统一文生图/1–3 参考图测试、ABBA、结果完成标记与复现边界 | [实验复现指南](runtime-ane-validation.md) |
| 此前保留修正：gate/up 局部简化、alpha 类型修复、三图 ABBA 与共图切换 | [LoRA 分段修正与回归](runtime-ane-lora-slices.md) |
| Qwen 1024² base 三路采样初测、视觉检查与内存边界 | [1024² runtime/GPU/frozen](runtime-ane-qwen-1024.md) |
| Qwen 三参考图 runtime LoRA ABBA、长/短序列调度与串行成本 | [编辑长序列诊断](runtime-ane-qwen-edit.md) |
| Qwen LoRA GPU 编译边界、固定分区消融与 BF16 测试标量读取修复 | [LoRA 编译边界](runtime-ane-lora-boundary.md) |
| 整块 on/off 门槛 5%/8% 负结果；恢复 2%/5%，不混淆 FFN/整块测量窗口 | [门槛消融](runtime-ane-block-margin.md) |
| 输出直接写 MLX 的负结果；撤回候选，保留输出/回调生命周期测试 | [输出复制消融](runtime-ane-output-copy.md) |
| 长 prefill 四分之一 rows 起步的负结果；恢复一 chunk 起步并实测调节 | [初始分区消融](runtime-ane-initial-share.md) |
| Z 完整 block on/off、免计时混合、Q4 c352 与 BF16 回归 | [Z 调度接续](runtime-ane-z-block.md) |
| 真实 Q4_0、padding 兼容修复与整理构建回归 | [Q4 记录](runtime-ane-q4.md) |
| Qwen 周期采样、两模型同构建 base 对照、LoRA 负结果 | [周期采样](runtime-ane-periodic-sampling-2026-09-29.md) |
| 完整 GPU block probe、LoRA SHA 校验成本 | [完整块探测](runtime-ane-full-block-probe-2026-09-29.md) |
| runtime-weight base 集成、token-row、staging、Q8 和多 chunk | [整模型集成](runtime-ane-integration-2026-09-28.md) |
| runtime graph-v2：完整 LoRA 激活修正、base/A/B/base 共图 | [LoRA 接入](runtime-ane-lora-2026-09-28.md) |
| Qwen 自适应关闭后恢复完整 GPU block，保留 full/split 缓存 | [unsplit 实验](runtime-ane-unsplit-2026-09-28.md) |
| Qwen 完整 LoRA，共用冻结 base 图 | [Qwen 实验](qwen21-runtime-lora-fused-2026-09-28.md) |
| Z 完整 LoRA、可选 direct-FP16 | [Z 实验](z-image-runtime-lora-fused-2026-09-28.md) |
| Qwen GPU、编辑、序列切片与已撤回实验 | [路线索引](qwen21-acceleration-routes-2026-09-27.md) |
| Qwen 1024² base 文生图，不含编辑/LoRA | [1024² 诊断](qwen21-w8a8-ane-1024-diagnostics.md) |
| Qwen 20–40 步可选残差缓存 | [DBCache](qwen21-dbcache-2026-09-27.md) |
| Qwen 矩形画布复用固定行数图 | [矩形画布](qwen21-rectangular-w8a8-2026-09-27.md) |
| 集成前单层探针，不作为整请求成绩 | [早期组件实验](runtime-ane-component-2026-09-28.md) |

## 本次代码整理与验证

最近的整理范围与实际测试见[收尾记录](acceleration-cleanup.md)，代码职责见
[后端维护说明](../../native/backends/README.md)。本页只做索引，性能、默认与
optional 选择统一维护在[决策入口](acceleration.md)，不复制逐轮进度。

保留 GPU、冻结 base、完整 runtime LoRA 和有收益的异步/数据转换优化。
撤回候选不重新加入产品路径，数值边界、失败恢复和 adapter 隔离回归继续
保留。离线 probe、导出与微图测试显式运行，不进入产品构建；验证工具共用
标准库校验/预检/哈希模块。图/权重/PNG/缓存与源码分离，不删除原始证据，
不改既有暂存区。

测试通过不等于所有分辨率、编辑、LoRA 或物理 ANE placement 已验收。
运行测试与 benchmark 不重叠；历史通过、skips 和新执行分别记录。

## 运行入口

- [CLI 与请求契约](../public/USAGE.md)
- [Z-Image GPU/ANE](../public/Z_IMAGE_ANE.md)
- [Qwen-Image-2.1](../public/QWEN_IMAGE_21.md)
- [冻结 base 图复用与便携示例](runtime-lora-acceleration-2026-09-28.md#便携-cli-示例)

## 其他专项与历史记录

- [Streaming/App 集成](streaming-app-integration-2026-09-27.md)
- [Z-Image Metal 优化](z-image-metal-2026-09-23.md)
- [Z-Image INT8 streaming](z-image-int8-streaming-2026-09-20.md)
- [Encoder prefill 可选路径](encoder-prefill-optional-2026-09-15.md)
- [ANE 加速分阶段决策](../design/ane-acceleration-status-2026-09-15.md)
- [2026-09-10 状态快照](current-status-2026-09-10.md)
- [原始目标审计](objective-audit-2026-09-09.md)

逐项 screen 记录保留原文件名，供复盘负结果，不作为默认配置清单。
`outputs/`、`results/qwen21/` 和 `results/z-image-*` 是本地实验产物，
不随源码分发；稳定结论在本目录，请求模板在 `examples/requests/`。
