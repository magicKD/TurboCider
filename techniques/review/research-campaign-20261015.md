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
| 来源项目固定版本与关键源码阅读 | 9张仓库主题卡覆盖ccv/vpipe/coremltools/ANE Transformers/anemll-profile/MLX/GGML/gputrace（MLX affine/SDPA分卡）；另有1张官方Metal SDK卡，SHA与已读范围登记；其他P0/P1继续 |
| 来源卡/词表/查询/候选/版本索引 | 单一catalog/来源登记，source/evidence区分与有界展开可用；来源索引自动生成。细分过滤、别名、版本claims与薄SKILL仍待完成 |
| GPU量化FFN与attention候选 | affine完整FFN与原始BF16确认之外，两seed真实attention replay/三候选/语义/noop共21比较已做，无稳定晋升；更多层/prompt/长KV/precision、真正GGML全backend、MFA继续 |
| 公共ANE图变体/tiling/handoff | 真实1024×4096×12288、frozen/runtime、monolithic/tiling、serial/hybrid已测一组；没有晋升生产候选，native shared IO等继续 |
| profiling工具代码审核与实际探针 | native三模式/10份capture解析之外，12份clock控制和真实BF16三点host profile已做；同配置单位/窗口一致性有证据，不外推MLX kernel。profiler streamData、GUI/PMU与完整模型关联继续 |
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

## 2026-10-16续接：公共GPU profiling与来源反例

上一goal轮为progress：用户要求的整理与本地提交完成于`5ed9424`，不是研究目标完成。此轮回到完整roadmap，新增[GPU profiling报告](gpu-profiling-20261016.md)和[逐样本证据](gpu-profiling-evidence-20261016.json)，未自动再次commit/push。

在只读参考checkout之外隔离构建固定gputrace，选定两个parser package测试通过；运行10份owned capture的stats/timing，不启动Xcode/replay、默认shader source扫描或改权限。CLT下xctrace退出72，但真实native capture和公共stage timestamp均可用。9个三模式主实验、1个stage smoke、1个dispatch采样不支持记录全部保留；正式计时与诊断分开。M4 Max本机只暴露timestamp集合，stage支持而dispatch不支持；counter原值/配对时钟保留，seconds/occupancy未校准。

native捕获中1/1/16个dispatch有受控程序ground truth。MLX则必须在停止capture前显式同步，才从8-byte header变成可解析结构；eager、换输入、仅开始前同步均未改善。parser在这些MLX结构中仍识别不到dispatch，所以不称为0次执行或完整coverage。源码的output-event等待和encoder end/commit/wait分开讨论，不推翻已记录的host-output-ready边界。

更重要的反例是函数名录并非执行记录：MLX读出16,446个名字，synthetic累计25.731秒；空capture的timing也会成功退出并输出0。新的只读normalizer按固定commit保留predicted/unknown，不拿boolean、Go字段名或猜测端点晋升实测。大名录只导出有界前缀与完整hash；native全部样本、失败、源码快照与capture内容均复核，raw UUID、buffer与capture原件不发布。

知识入口增加两篇profiling技术卡、gputrace源码卡与独立官方SDK卡，沿用同一catalog/sources登记和生成索引，官方来源不伪造Git commit。主9个native实验使用当时同一份C++源码；较早smoke及历史wrapper/helper的不同源码快照分别保留，不强行合并身份。未删模型、旧实验、原始数据或系统缓存；本轮Go cache和capture占用本地空间，后续设备实验仍须检查余量。

下一步优先核对原始timestamp单位、配对校准误差和尺寸/重复缩放，关联真实Qwen组件；仍需GPU attention、native GPU/ANE handoff、ANEForge/PMU、streamData/GUI、LoRA/更多层/完整请求及媒体/内存/CPU组合。受控多维metadata、aliases、version claims与薄AI skill也未完成。此轮不是整体完成，也不因当前parser限制或候选负结果停止研究。

本批最终检查：184项techniques测试、21项repository测试通过；techniques-only隔离副本运行40个CPU cases、95页/913个本地链接、47项catalog和依赖检查，无错误。来源索引`--check`通过；官方source metadata、source/evidence展开与大JSON分页核查通过。32KiB共同预算会明确列出被source/code挤出的evidence，128KiB能展开到证据，不把省略当丢失或修改检索预算合同。检查不授予新的全模型、时钟单位、物理ANE或其他SKU资格。

## 2026-10-16续接：时钟辨识、采集器负控与真实BF16 host关联

上一goal轮为progress：公共profiling采集/归一化与知识入口实际改变了权威状态，不是整个目标完成。本轮继续同一roadmap（1350行，SHA256 `247f7e92e49ae6ba886c6a0dca6fc53b01aa82028a40f258d4f33473121aa242`），重核P01、G03等完整范围，不以新增probe替代attention/模型/ANE等剩余工作。

新增[校准与Qwen报告](gpu-clock-and-qwen-20261016.md)。12份clock记录包含一个明确采集器bug：新增的command-window可变数组被复用，单位虽正确、7/8关联却错误。保留原记录和源码，改为不可变copy后，新smoke与GEMM/向量规模及1/4/16/32段graph共11份通过、716个stage窗口；实际CPU ns候选、125/3 Mach timebase与1 ns/GPU raw tick的同配置映射有实测见证。不声称875ns bracket或约41.5ns残差就是总精度/独立物理传感器/其他SKU保证。旧raw数据无bracket仍不回填秒数；native代码因新采集字段改变，历史快照继续独立标识。

真实Qwen layer0的step1/19先做，step10新进程随后确认。原始checkpoint/activation SHA与完整`[1,1024,4096]` BF16消费者不变，3点compiled/staged对BF16 replica全输出误差均0，独立FP64抽样最大relative L2约0.00399，过预设3%局部门槛。compiled host-output-ready中位约21.66–22.08ms；分阶段drain会改变程序，单列host诊断，不输出speedup、不假装MLX GPU kernel counter或完整请求质量。

新[时钟合同](../profiling/timestamp-calibration.md)与[MLX边界](../profiling/mlx-boundaries.md)通过同一catalog/sources进入检索；gputrace候选状态已与实际源码/CLI执行对齐。公开Qwen capsule可显式选run/point/trial离线导出host trace，不依赖模型、原始输出、产品/兄弟仓库或MLX安装；输入hash实际复核，原raw报告hash仅标为declared lineage。没有启动viewer或自动下载/安装。

剩余完整目标不变：优先读Qwen attention调用与MLX SDPA dispatch，设计有界真实QKV采集，再做head/mask/shape候选及强GPU基线；同时继续native GPU/ANE handoff、ANEForge/PMU与生命周期、LoRA/更多层/请求、媒体/VAE/内存/CPU组合，以及多维metadata、aliases、version claims和薄AI skill。当前磁盘约1.2GiB，未新增大capture/权重副本、未删旧资料或改系统设置；资源限制不等于没有后续安全研究动作。未改产品默认，未自动commit/push，目标继续active。

本批验收：199项techniques测试、21项repository测试通过；隔离副本只含techniques，重跑40个CPU cases、98页/950个本地链接、49个章节catalog与依赖审计，无错误。clock/Qwen curator内容hash、官方索引、typed数值/ABI/gates反例、公开capsule的`-I -S`离线trace导出及450000-byte处证据续读均通过。GPU probe本身需要本地helper目录，help核查使用`-E -S`，不把Python `-I`排除脚本目录的导入错误当设备问题。验证不等于剩余roadmap或产品性能/画质/ANE资格完成。

## 2026-10-16续接：真实attention与empty-row语义

上一goal轮为progress：时钟校准、采集器负控及真实BF16 host profile提供了新证据，整个目标仍未完成。本轮沿G03推进，先读原Transformer norm/RoPE/segment/cache与observer，再读固定MLX SDPA guard/layout/softmax；没有以已有CPU online-attention toy替代真实工作负载。

[本机attention报告](qwen-attention-20261016.md)和[全部证据](qwen-attention-evidence-20261016.json)保留2个独立seed source、21个ABBA比较与1个四case诊断。只编译可选小客户端，复用648个native source/386个MLX input逐项匹配的public库；未改生产源码或App。actual原block arguments上按原配方重放，额外暴露QKV/O，全block及实际prefix过exact控制。它不是原未改变compiled kernel内部的直接抓取；序列化副本与replay物理stride分别保存并在GPU见证。

实际为32 heads/D128、1048-row prefill与1024Q/1048KV steady，24-token text prefix；seed42 discovery、seed43相同prompt确认。3候选全部heads/keys及原消费者恢复都保留，所有真实O对replay reference误差0，独立CPU完整K/输出列抽样relative L2最大约0.001391。single-mask与contiguous-V无收益，head-chunks steady约1.019×的确认区间跨1；noop也有约1%表面偏移和离群，未晋升默认/性能profile。准备/residency与prepared operator边界分开，不能推广完整block/request/LoRA质量。

tail、非方阵causal、D80和GQA controls通过；两个全false首行比较拒绝本轮明确的zero合同，不给ratio。常量V=1诊断中，D64/K29为29/32，K32为1；D96也为1，非空行均为1。固定源码的finite_min/tail/归一化提供解释方向，但没有物理dispatch定位，不能宣布通用API/硬件故障。legacy control名fused/fallback只是源码预测；实现fallback也不等于CPU设备回退。诊断return0仅代表采集完成，zero gate仍失败。

来源、候选和查询沿用同一登记；MLX两个主题由topic区分生成索引，不创建平行手工目录。人工检查发现代码链接误挂encoders卡，已移回attention并加selector→implementation回归测试。两个seed tensor合计约130.25MiB留本地ignored目录，旧资料/模型不删，余量约1.1GiB；公开证据只存hash/数值/范围，不导出原始tensor、机器路径或用户会话。未自动commit/push，目标active。

完整剩余范围继续：带完整cast成本与source预算的precision island、projection/attention融合、更多层/prompt/编辑/LoRA/长KV与全请求；native GPU/ANE actual-backing/handoff、ANEForge/private能力与PMU安全审查/生命周期；媒体/VAE/内存/CPU组合，以及多维metadata/aliases/version claims/薄AI skill。现有negative结果只收窄本shape/配方/边界，不关闭这些方向，不以绿色测试或文档数宣布目标完成。

本批验收：212项techniques标准库测试、21项repository测试通过；techniques-only隔离副本重跑40个CPU cases、100页/984个本地链接、49项章节catalog、14个显式设备selector和依赖检查，无错误。来源索引`--check`、source/evidence展开、attention canonical code routing与300000-byte处JSON续读通过；模型/产品/兄弟源码不在隔离副本。新case有实际GPU执行记录，但默认测试和查询不运行GPU；source hash/数据验证也不冒充原kernel物理定位或新全请求资格。
