# Apple Silicon 源码研究与实测：工作台

[总目录](../README.md) · [既有自包含验收](self-contained-validation.md) · [阶段地图](../models/stage-map.md)

状态：进行中。基线为TurboCider `7e96f51`；已完整读取用户提供的`TurboCider_Apple_Silicon_Wiki_Learning_Roadmap.md`（1350行）。该roadmap是研究计划，不是性能证据。此工作台保留全目标与尚未完成的工作，不以文档数量或微基准通过冒充研究完成。

## 要完成的工作流

1. 知识工程：稳定ID、来源/综合机制/实验分离；受控词表与别名；只读查询、来源展开；candidate/version/revisit账本；人和Agent共用入口。
2. 源码研究：先贯通KernelWiki/KDA、MLX/GGML/MFA/ccv、CoreML/ANE图表达、vpipe/现有hybrid与profiling的纵向链；再扩展图搜索、私有能力/PMU边界及内存媒体。
3. 实机实验：相同shape/source/precision与受保护oracle；GPU、ANE、串行、并发完整consumer边界；搜索和独立确认分开，负结果保留。
4. 模型应用：以可用的Qwen Image 2.1真实shape/activation/request为纵切；候选与默认隔离，数值/质量、时间、内存与实际fallback一起判断。
5. 交付：将可复用know-how回写既有techniques分类、生成查询索引与实验任务；保留来源许可、版本、完整样本与适用范围。

## 当前事实与执行边界

- 工作树开始时干净，未改产品默认或已安装App；现有模型、工具环境与原始报告均保留。
- 本机是M4 Max / 64GB；当前选择的developer directory是CLT，尚未发现完整Xcode。因此Xcode/Instruments能力须实际探测，不伪造trace/counter结果。
- 实验按一个SoC实验槽串行执行；下载/源码阅读不当作设备实测。查询工具不自动下载、安装、运行benchmark或修改安全设置。
- 新参考checkout仅在用户指定的references目录新建；已有参考项目只读，HEAD和dirty文件SHA分别记录。
- 编译/性能/数值资格key分开；API策略、计划估计、实际wall、派生指标与物理设备归因分开。

## 进度

| 项目 | 当前状态 |
| --- | --- |
| roadmap与当前仓库核查 | 已读，基线已确认 |
| 来源项目固定版本与关键源码阅读 | 7张新卡固定ccv/vpipe/coremltools/ANE Transformers/anemll-profile/MLX/GGML的SHA与已读范围；其他P0/P1继续 |
| 来源卡/词表/查询/候选/版本索引 | 单一catalog/来源登记，source/evidence区分与有界展开可用；来源索引自动生成。细分过滤、别名、版本claims与薄SKILL仍待完成 |
| GPU量化FFN与attention候选 | affine Q4/Q8 cache/decode/SG32/64/连续加载/skew与原始BF16控制已做完整FFN对照及step19确认；attention、真正GGML全backend、MFA兼容性继续 |
| 公共ANE图变体/tiling/handoff | 真实1024×4096×12288、frozen/runtime、monolithic/tiling、serial/hybrid已测一组；没有晋升生产候选，native shared IO等继续 |
| profiling工具代码审核与实际探针 | 公共plan/host trace和同compiled bundle policy数值诊断已实测；CPU-only重试完成。原profiler安全hold、真GPU trace/counter/PMU/gputrace等仍未完成 |
| Qwen Image 2.1实际应用与对比 | 原始BF16 checkpoint/capture/消费者控制已补齐；layer0两类实验均未获原始GPU之上的收益，未改默认。LoRA/更多层/请求资格未完成 |
| 媒体/内存/算法组合扩展与维护检查 | 待分层推进 |

后续记录必须将upstream/source-reported与本机measured分开。没有得到通过证据的项目保持未完成/待确认，不自动晋升默认或硬件资格。

## 2026-10-15 代码整理验收

本次仅整理已有研究源码、补齐可用的catalog检索与忽略规则，不继续性能研究、不修改产品默认，也不把未完成的多维知识库或MFA实验标为通过。

- `python3 -B techniques/scripts/check_standalone.py --isolate`：128项工具/CPU单元测试、40个CPU示例、80页链接检查、45个章节映射与自包含依赖检查全部通过；隔离副本不包含产品或兄弟仓库。
- `python3 -B -m unittest discover -s tests/repository`：20项仓库测试通过，包含新增的忽略规则正/反例。
- `swiftc -frontend -parse techniques/integrations/mfa_gemm_bench.swift`：语法解析通过，不代表上游package构建、GPU数值或性能验证。
- `git diff --check`通过；新增规则没有屏蔽已跟踪文件。模型、第三方源码、原始测量与机器本地状态仍不纳入提交。

## 续接：真实FFN纵向链

上一goal轮是progress：`56b5892`整理了源码、忽略规则与基础检索，并验证测试；不是全研究目标完成。本轮从该提交继续，未再次自动commit。

当前证据：[真实Qwen FFN实验](qwen-ffn-tiling-20261015.md)与[全部样本/身份](qwen-ffn-tiling-evidence-20261015.json)。新增[来源层](../sources/README.md)与同一登记的source/evidence展开，让human/AI能从技术页走到固定源码、失败对照和本机数据。

初批实测区分两件事：exp形式通过固定数值门槛，但完整Core ML消费者明显慢于compiled GPU；相同6144-channel切分的同进程serial/hybrid配对约1.16×，仍不能战胜完整GPU。两个CPU_AND_NE sigmoid图被独立抽样门槛拒绝，未给speedup。初次CPU-only控制遇到NoSpace，当时保持未测；后续有admission后重试及同bundle policy诊断，见2026-10-16续接，不能用初次资源失败编造数值结论。

磁盘不足只清理本轮两个已终止实验的4个可重建编译目录，事先保留逐文件hash；原模型、capture、fixture、原始数据和旧实验均保留。后续增加admission与源码快照/结束校验；早期探索仅有runner hash、缺少完整源码快照，明确保留这一局限，关键runtime确认在带快照的新进程重跑。编译产物重建不承诺字节相同。

下一批仍沿roadmap推进：MLX/GGML packed GEMM与attention、native handoff/precision、ANE图搜索与真实profiling能力，再扩展VAE/媒体/内存/CPU/cache组合；完善受控多维metadata、版本失效与AI skill。当前只有layer0组件这条链取得新实测，不能将整个目标或所有参考项目标成完成。

本轮续接验收：148项techniques标准库测试、20项repository测试通过；只复制techniques的隔离检查再次运行全部工具测试、40个CPU cases、87页/809个本地链接、45项catalog与自包含依赖检查，均无错误。来源索引`--check`通过，13份证据保留；两项带快照确认的源码hash与当前发布代码一致，证据JSON在300000-byte处的有界续读也通过。以上不是额外的GPU/ANE性能资格或其他SKU验收。

## 2026-10-16：GPU源码、格式与强原始控制

上一goal轮为progress：新增真实Core ML/MLX FFN证据、来源层和有界检索，不是整体目标完成。本轮新增独立MLX v0.32.0 sparse checkout，保留既有checkout，逐文件比较release/另一固定源码；并读取GGML块布局、decoder、Metal matmul与device policy。只有两段带MIT许可的GGML原函数体被摘录执行，没有复制/运行整个engine。

当前[GPU affine报告](gpu-affine-20261016.md)及[全部样本](gpu-affine-evidence-20261016.json)包含16个比较（1 synthetic smoke、10真实discovery、5独立step19确认）、2组诊断及1次frontend编程错误。BM32/64、coalescing/skew通过数值却慢于MLX；Q8 cache对packed约1.047×，但对原始BF16约0.998×，额外288 MiB逻辑驻留。Q4配方内cache确认区间覆盖1。未把压缩、近似budget或源码直觉当产品加速。

强控制直接读原始BF16 tensor与原始capture并检查SHA，不从FP16派生物倒推“原始”权重；保持`[1,M,H]` BF16消费者，计入候选输入cast/输出恢复。原始baseline对BF16 replica的输出误差为0。精度探针还记录同库shape-dependent Q4行为；四个strided operands和合法上游excerpt均有实际GPU输出，不把GPU storage Q8标为整数MAC或ANE。

磁盘余量后来恢复至约2.8 GiB，满足现有admission后重做之前NoSpace的CPU控制，原失败未删。现在Core ML corpus为14个report加一个同bundle policy诊断：同compiled assets/同输入、ABBA与前后hash检查，CPU_ONLY样本误差0.4769%，CPU_AND_NE 2.8337%，重复可见。只收窄到本图/环境/policy相关行为，不定位单op或硬件错误；不将诊断host spans当正式timing。

来源、候选与evidence登记沿用单一catalog/来源清单，来源索引自动生成；默认验证区分本地asset hash与外部checkout验证，避免新excerpt让旧字段误报。当前产品默认、模型和已安装App未改，本轮未自动commit。

剩余完整目标不变：真实GPU attention与更强kernel/profile链、native GPU/ANE handoff、ANEForge/私有能力和PMU安全探针、完整请求/LoRA/其他深度，再扩展媒体/VAE/CPU/streaming/cache组合，以及受控多维metadata、aliases、version claims与薄AI skill。不是因为当前几个候选没赢就停止探索，也不因测试通过宣称研究完成。

本批最终验证：161项techniques标准库测试、20项repository测试通过；techniques-only隔离副本再次完成测试、40个CPU cases、90页/859个本地链接、45项catalog与依赖审计。默认source检查实际核对1个本地licensed asset hash，不冒充外部checkout验证；7卡来源索引与两份curator SHA检查通过。GPU corpus为16个比较/2组诊断/1个frontend失败，Core ML corpus为14个比较/1个同bundle诊断，全部保留。当前仍无产品资格、其他SKU或完整画质宣称。
