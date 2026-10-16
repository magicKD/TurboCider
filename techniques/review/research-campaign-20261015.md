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
| 来源项目固定版本与关键源码阅读 | 12张仓库主题卡增加ane_pmu_profiler与measure_ane_capacity，覆盖ccv/vpipe/coremltools/ANE Transformers/anemll-profile/MLX/GGML/gputrace/ANEForge（MLX affine/SDPA分卡）；另有1张官方Metal SDK卡，SHA与已读范围登记；其他P0/P1继续 |
| 来源卡/词表/查询/候选/版本索引 | 单一catalog/来源登记，source/evidence区分与有界展开可用；来源索引自动生成。细分过滤、别名、版本claims与薄SKILL仍待完成 |
| GPU量化FFN与attention候选 | affine完整FFN与原始BF16确认之外，两seed真实attention replay/三候选/语义/noop共21比较已做，无稳定晋升；更多层/prompt/长KV/precision、真正GGML全backend、MFA继续 |
| 公共ANE图变体/tiling/handoff | 真实1024×4096×12288、frozen/runtime、monolithic/tiling、serial/hybrid已测一组；没有晋升生产候选，native shared IO等继续 |
| 私有ANE图搜索与生命周期 | 固定ANEForge小图/缓存负控之外，真实layer0完整FFN的512/1024/2048分担已测，冻结2048后step19/10及24-group观察约1.08–1.12×warm组件收益；actual backing/零拷贝、LoRA/请求、失败/取消和PMU资格未完成 |
| profiling工具代码审核与实际探针 | native三模式/10份capture解析、12份clock控制和真实BF16三点host profile之外，私有PMU9组/81次输出与CLI假阳性/skip控制已做；所测路径无payload，单位/累计语义未取得。profiler streamData/GUI、其他capture路径与完整模型关联继续 |
| Qwen Image 2.1实际应用与对比 | 原BF16 source/capture/消费者控制之外，private copy-bridge在指定layer0 warm完整FFN已有held-out timestep局部收益；既有public CoreML/GPU affine/attention负结果不改写，未改默认。LoRA/更多层/完整请求资格未完成 |
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

## 2026-10-16续接：ANEForge图搜索、失败缩减与私有同步执行

上一turn为progress：用户要求的整理/忽略规则回归与提交已完成于`6b3964b`，不是全研究目标完成。本轮重读1350行roadmap并核对同一SHA，推进A03/N06；原参考checkout保持只读，不继续自动commit/push。

新增[ANEForge固定源码卡](../sources/repos/aneforge-rewrite-search.md)、[图搜索技术页](../ane/graph-rewrite-search.md)与[本机报告](aneforge-20261016.md)。隔离副本构建小e5rt shim，明确dylib/cache、禁止autobuild、保留breaker；未开profiling、fork例外、async或多op共享。完整副本58文件后验比对，关键每run源码前后SHA、shim与MIL/blob均复核；不把上游buildInfo字符串当现场工具链身份。

[全部15份receipt](aneforge-evidence-20261016.json)含13组实际设备比较（351条first/warm/measured完整输出检查）、成功CPU合同和一个保留的frontend关键字错误。真实图key同结构/不同W相同，编译内容key不同；mock cache-hit build1/measure0，mock precision预算全部未达仍返回model。两类mock明确不是设备测量。新增核心CPU身份/gate示例不依赖上游、NumPy或MLX。

完整合成FFN的原生SiLU/INT8 arm未过1% full FP64门槛，保留但不给ratio。逐层缩减至projection+SiLU、单算子和常量点后，当前e5rt单SiLU程序仍约4.2%误差，零输入观察到约−0.001704；不定位内部硬件/编译器故障。exp形式在本轮域内将FFN误差降至约0.43%，两个seed通过局部数值；对增强后的FP16/FP32 compiled GPU，独立确认的完整host-fed小图约3.07/3.26×。所有raw样本、早期GPU cast版本、不同源码hash、resident GPU诊断、identity噪声与reduction无优势均保留，不把合成CPU生产者边界借成真实Qwen/GPU→ANE提速。

活owner的view更新、Model副本稳定、poison完整覆盖控制通过，没有真正做release后访问。原始数据/编译文件仍ignored；没有删模型、旧实验或系统cache，磁盘约0.9GiB。没有修改产品runtime、默认图或已安装App；约3×不是原真实Qwen公共CoreML负结果的替代品。

完整剩余范围保持：exp大负值/真实激活范围、原BF16/FP16 source与cast成本、Qwen更多层/LoRA/长KV/完整请求；GPU融合/更强kernel、native GPU/ANE actual-backing/handoff与并发；private错误/取消生命周期、PMU/streamData/GUI安全和语义审查；媒体/VAE/内存/CPU/streaming/cache组合；受控多维metadata、aliases、version claims和薄AI skill。ANEForge整套upstream corpus、async与PMU未运行，不能用本轮小图或绿色测试宣布这些完成。

本批验证：223项techniques标准库测试、21项repository测试通过；仅techniques的隔离副本运行41个CPU cases、103页/1027个本地链接、50个章节catalog与依赖审计，无错误。完整仓库可选来源检查核对132文件/31内容hash，无警告；来源索引`--check`通过。独立FP64 oracle的5种公式另用真实NumPy对手写math.fsum小矩阵校验，最大差0。默认suite/query不触发上游导入或设备执行，测试不授予产品/物理ANE或全目标资格。

## 2026-10-16续接：PMU源码、采集缺失与严格负控

上一goal轮为progress：ANEForge源码/实机实验、失败缩减、核心合同与索引已改变权威状态，整个研究目标仍未完成。本轮核对roadmap同一SHA，继续P03/P04私有PMU路径；保留所有上一轮未提交改动，不再次自动commit/push。

完整阅读固定`ane_pmu_profiler@20080a945...`的1162行采集器、496行loader、headers及836行报告，核对小型compiler/ODIE/Swift路径。发现并在原CLI复现missing-model仍ACTIVE+exit0，valid自有MIL与普通debug-signature控制也无stats但exit0。直接ANECIR分支缺client赋值、host wall被命名为silicon时间、未经验证的累计/下降夹0与CoreAI覆写/临时bundle行为均单列；没有执行安全降级或加载附带模型。

沿一手链接新建428KiB sparse参考`measure_ane_capacity-research-20261016@2d3809ba...`，不改已有checkout。完整读723行CLI及选定App/说明，隔离编译原CLI，owned TMPDIR/package下实际运行有界FP16；package导出但无ANE bundle，PMU分支跳过，汇总0不是0活动。两个来源分别登记许可：前者固定tree未找到授权文件，不复制实现；后者BSD-3-Clause，不将其许可扩大到前者/Apple框架。

新增[PMU合同/来源](../profiling/ane-pmu-lab.md)、[本机报告](ane-pmu-20261016.md)与[全部证据](ane-pmu-evidence-20261016.json)。原创自包含MIL/native客户端运行9组81次：affine、dense1×1 conv、小/较大形状、mask0/2/15、compile-profiling请求、新进程确认。全部物理pitch/输出poison/数值/unload通过；独立标准库math.fsum/FP16逐元素生成完整input/output SHA全部一致，不仅相信native布尔值。17列周期明确仅是合成输入，不当真实激活。

所测63次profile请求均无stats object，后期54次还观察到自有4KiB stats surface全零；较早9次没有surface字段，不回填。当前framework只读mask翻译0→0/2→4/15→15，并不授予counter资格。strict入口实测native0而wrapper3，明确“计算通过、PMU未取得”。没有产生物理cycle/byte-rate/GHz/Joule/TOPS或Qwen/hybrid性能资格，不能宣称M4普遍无PMU。

核心CPU decoder区分missing/empty/zero-filled/unknown/raw，保留下标而不合并同名未知寄存器；reset/wrap算术要求显式位宽/见证，不夹0。默认prepare/CPU all/查询不导入私有框架、下载或执行设备。真实selector为显式`private:pmu`；raw地址、boot信息和opaque buffer未发布。产品runtime/默认/App未改，模型/旧实验/系统cache未删。

完整剩余目标保持：P03实际payload及schema/累计/per-call/overflow/聚合/单位/缩放仍未完成；其他ANE/ANECIR/公开profiling、streamData/GUI与真实模型时钟关联继续。GPU更强kernel/融合/precision，真实Qwen更多层/LoRA/长KV/完整请求，native actual-backing/handoff/并发与失败生命周期，媒体/VAE/内存/CPU/streaming/cache组合，以及多维metadata/aliases/version claims/薄AI skill，都不能由本轮否决结果或绿色测试标为完成。

本批验收：234项techniques标准库测试、21项repository测试通过；techniques-only隔离副本重跑42个CPU cases、107页/1077个本地链接、51项章节catalog、15个显式device selector与依赖检查。完整仓库可选来源检查核对136文件/31内容hash，无警告；额外逐项核对两新卡的19个固定上游文件SHA与范围，原树均clean。来源索引`--check`、64KiB内展开技术页/两source/四code且无省略、证据90000-byte处续读通过；native当前源码syntax检查无警告。curator重建证据与发布JSON字节一致，raw binary/模型/临时产物仍ignored。以上不授予PMU/Qwen/完整roadmap资格，目标继续active。

## 2026-10-16续接：真实Qwen完整FFN private分担

上一goal轮为progress：PMU源码/两原CLI实际控制、自有81次输出与null/lifecycle合同已改变权威状态，没有取得physical counter。本轮核对同一roadmap SHA，回到真实模型完整消费者，不以约3×的合成ANE小图替代Qwen实验；保留全部未提交研究，不自动commit/push。

新增[完整边界技术页](../gpu-ane/private-ffn-boundary.md)、[本机报告](qwen-private-ffn-20261016.md)与[完整证据](qwen-private-ffn-evidence-20261016.json)。直接核对原BF16 checkpoint/tensor与forced-GPU capture，保持M1024/H4096/F12288和BF16 GPU输出；private只接配对gate/up/down前缀，完整GPU余下channels保留。使用exp(-abs)稳定SiLU，不改变固定3%/0.999门槛；完整full/prefix与独立FP64 sample在所有first/warm/timed检查。GPU完整分母全输出对保存的BF16 replica误差0。

step1预设512/1024/2048三个share：完整GPU→hybrid配对中位约0.9855/1.0335/1.1262×。固定2048后，step19/step10确认约1.1216/1.0818×；新进程step19的24-group确认约1.1196×、探索区间[1.0977,1.1256]，GPU约21.747ms vs hybrid19.421ms。再增加pre-arm validation drain的24-group控制，仍约1.1241×、区间[1.1167,1.1284]，GPU21.700ms vs hybrid19.309ms。serial→hybrid约1.614/1.629×另列，不借作完整GPU收益。当前只有layer0同轨迹三个timestep，不是新prompt/seed/LoRA或全请求资格。

共7组实测、21个配对比较、297次完整调用，另保留最初BF16→NumPy PEP3118 frontend失败；转换修复位于计时外验证读回，没有改GPU计算精度或降低预算。固定stable witness/domain源码在所有数值成功run之间未改。24-group确认对所有2M个原BF16 GPU prefix gate另有域见证，约[-2.484375,2.09375]，不当作private内部或其他层上界。

这个实现明确是copy bridge：ready BF16 GPU input→FP16 host staging→private feed/execute/read FP32→GPU remainder/FP32 join/BF16 drain，交接全部计时；只记录同process host重叠，不授予物理overlap/placement。GPU逻辑权重共存576MiB、native prefix48MiB，未测独立arm峰值。单次emit/compile/load约0.68s，不能把warm每FFN约2.33ms收益直接当冷请求收益。raw模型/激活/机器时间戳不公开，原件不删；当前磁盘约0.6GiB，后续大图/构建先做admission。

只读curator用标准库独立复核原始tensor/capture、NPY真实row pitch的native前缀内容、源码/MIL/blob/shim、全部顺序/gate/host依赖与ratio；不导入候选/设备，不重跑输出tensor或物理硬件。它不是硬件attestation。原public CoreML/GPU affine/attention结果不因这条新backend/图/边界而改写，产品默认和App仍未修改。

完整剩余目标保持：优先actual GPU/ANE backing、stride/alias/owner/完成与更少copy，然后更多层/独立prompt/seed/LoRA、完整block/request质量与时间、cache/load与内存/长时/失败回收；GPU更强kernel/precision/fusion、profiling/PMU/GUI、媒体/VAE/CPU/streaming/cache组合及多维metadata/aliases/version claims/薄AI skill继续。warm组件正结果不是整体完成，也不自动授权产品采用。

本批最终验证：245项techniques标准库测试、21项repository测试通过；techniques-only隔离副本重跑43个CPU cases、109页/1115个本地链接、52个章节catalog与15个显式device selector，无错误。完整仓库可选来源检查核对138文件/31内容hash；来源索引`--check`、48KiB内展开技术页/两source/三code且无省略、约632KiB证据在300000-byte处续读通过。真实NumPy下独立witness对手写math.fsum小矩阵最大差约4.34e−19；CLI的`-I -S --help`不需要可选框架。curator发布JSON与显式原件重建字节一致；源码/原checkpoint/旧记录保留。测试不授予完整请求、物理overlap或剩余roadmap资格。

## 2026-10-16续接：实际Metal/request backing与padding负控

上一turn为progress：用户要求的整理、忽略规则和提交已完成于`c5a5750`，包含隔离验证的缓存/capture排除及回归；不代表研究goal完成。本轮复核roadmap同一SHA `247f7e92e49ae6ba886c6a0dca6fc53b01aa82028a40f258d4f33473121aa242`，重点推进H02/H03及第13–15节完整交付/维护合同，不以toy替代既有真实FFN消费者。

新增[actual backing技术页](../gpu-ane/shared-surface.md)、[官方Metal/IOSurface来源卡](../sources/official/shared-memory.md)、[本机报告](private-shared-20261016.md)和[完整脱敏证据](private-shared-evidence-20261016.json)。官方Markdown与本机SDK26.5相关header范围/内容hash登记在同一sources清单，不镜像Apple正文；private selector/ABI独立逐项检查，不把公共API承诺扩张到`_ANE*`。

原创Metal producer→private同步MIL `2*x+0.25`→Metal consumer `3*y-0.5`在四个新进程执行：C32/W33和C64/W65、seed17/29、8组与24组确认，共120次完整正调用。每次实际request inputArray/outputArray返回的ioSurface与目标ID相同，Metal contents与实际surface base相同；初始/逐call前后/最终均检查，只发布bool、不发布raw ID/地址。完整input/ANE output/FP32 consumer三个逻辑hash/call由独立标准库重建，全部精确一致；poison、尾guard、owned GPU完成、同步返回/unload与释放通过。

W33实际row/plane128bytes，W65实际192bytes，不是紧密66/130，也不推成统一256-byte对齐。每run两个真实GPU负控，共8次：packed-stride读取仍全finite，却有1011/1056、3990–3991/4160错误；上一代实际输出按下一代验证全部元素拒绝、最大误差12。负控不加入正样本或ratio，owner活到所有消费者结束，不做真正UAF。

shared在记录执行边界的external feature copy数为0；copy控制用独立Metal staging和两次CPU copy，8KiB/24KiB及lock/unlock计入相同GPU producer encode→GPU consumer完成边界。四组paired copy/shared诊断约1.0633/1.0105/0.9894/1.0109，后者paired中位与两组中位之比符号不同；未取得稳定性能晋升，更没有optimized GPU-only分母。runtime内部copy仍unknown，串行host窗口不是物理ANE放置/并行；当前小图成功不回填旧Qwen copy-bridge的zero-copy或完整模型资格。

明确记录缺口：失败/取消/quarantine只有保守实现，四进程没有触发；没有OS build/framework binary hash、逐arm峰值、长时热或冷请求资格，不为旧run回填字段。四个owned目录合计约1.1MiB；磁盘观察从约0.6GiB降至约0.3GiB，不据此归因全部变化，也不擅自清旧数据。没有新增权重/大capture或修改参考checkout、产品runtime/default/App；本轮研究不自动commit/push。

下一步仍接真实Qwen而不是扩大toy数字：已只读定位固定MLX0.32.0的`array.cpp` raw-pointer constructor与Metal allocator `make_buffer/release`。构造先尝试no-copy，失败会copy并立即调用caller deleter；`array.data()`注释也不能当GPU alias保证。需实际验证导入buffer/base/lifetime，再按真实二维MIL descriptor GPU pack/cast、private prefix、GPU remainder和restore/join，重新过完整BF16/full-prefix/FP64及强GPU分母，不复制或假装共享已有host桥接。

完整剩余范围不变：真实更多层/独立prompt/seed/LoRA与完整block/request质量/时间，冷启动/常量cache/load/内存/长时/失败取消；更强GPU kernel/precision/fusion、profiling/PMU/streamData/GUI；媒体/VAE/CPU/streaming/cache组合；受控多维metadata、aliases、version claims与薄AI skill。当前绿灯只证明本轮限定合同，目标继续active。

本批验收：257项techniques标准库测试、21项repository测试通过；techniques-only隔离副本运行44个CPU cases、112页/1157个本地链接、53个章节catalog和16个显式device selector，无错误。完整可选来源检查核对141文件/31内容hash，来源索引`--check`通过；新技术页在96000-byte共同预算下展开两source/四code，无省略，341567-byte公开证据在300000-byte处可继续读取。独立curator从四个明确原件重建与发布JSON字节一致；native当前源码syntax检查无警告。原件/源码/模型/旧记录均保留，这些验证不授予Qwen共享路径、稳定性能或剩余roadmap资格。

## 2026-10-16续接：MLX实际adoption、Copy alias与过期结果

上一goal轮为progress：四进程实际request/Metal backing、120次完整消费者和stride/stale负控已改变权威状态，但没有稳定性能或Qwen共享路径资格。本轮复核相同roadmap SHA，继续H02/H03真实接缝前提；保留上一轮所有改动，不自动commit/push。

新[MLX来源卡](../sources/repos/mlx-external-memory.md)固定v0.32.0/`7a1d4f5c...`，选读constructor、Metal allocator、AsStrided/Copy与ops，七个文件含LICENSE的内容hash/范围进入同一sources登记。raw-pointer构造会fallback复制并立即调用deleter，strict make_buffer须release不能普通free；Metal包装为shared/untracked。Copy::eval在此版本直接copy_shared_buffer，source tag只作阅读身份，不声称已安装wheel由它可复现构建。

[实际adoption报告](mlx-adoption-20261016.md)与[完整证据](mlx-adoption-evidence-20261016.json)含一个保留客户端失败、四个完整新进程：padded `[32,33]`/pitch64和完整`[1024,4096]`逻辑FP16 8MiB，seed17/29，两种raw-pointer/strict-buffer模式。导入整块page region后再建view；实际buffer.contents/raw_ptr/base、长度、设备、offset和storage/hazard身份只发布bool。flat parent离开scope后owner仍活着，全部consumer/controls完成、最后view释放后deleter恰好一次。没有真正UAF或未获授权的系统变更。

首轮错误地把same-dtype mx::copy当独立快照，alias/generation0全输出通过后被strict snapshot gate拒绝；原source/binary/报告、退出1和mapping quarantine保留，不称硬件unsupported。更正控制后，same-dtype Copy作为alias阳性，新consumer读取generation1；独立backing由已实际求值的FP16→FP32 materialization建立。

四份完整协议有24个主代际input/consumer全量hash，另8个same-dtype Copy新consumer，共32个完整正向consumer；失败记录另有1个partial完整hash，不混入成功数。8个旧已求值结果与8个materialized snapshot按新generation验证均全元素拒绝、最大误差4；4个padded packed-stride负控有1023/1056错误与496 nonfinite，contiguous case不伪造stride错误。20项负控观察中cached-result re-eval不是新GPU dispatch，计数和性能不混淆。

新版另外采OS26.6.2/build25G83、SDK26.5、clang21.0.0，installed383 headers/三library SHA前后不变，实际MLX0.32.0/architecture applegpu_g16s。独立stdlib verifier利用明确合成周期流式重建完整hash，native仍逐元素写读完整8MiB；不是少行重复来假装真实激活或模型加速。curator只读明确原件、fixed clean源码与installed assets，不导入MLX/候选或跑设备；raw地址/registry ID不公开。完整manifest用Path组件顺序而非字符串顺序核对，不省略headers以让验证通过。

本轮只验证新建未显式compile consumer的正常完成协议，没有ANE调用、optimized GPU-only分母或性能score，没有取消/失败回收、dyld loaded-image attestation、峰值/冷/热/长时资格。不能把上一轮private surface toy、本轮8MiB adoption与旧copy-bridge约1.12×相乘或拼成未跑过的Qwen共享图。五个owned目录约1.2MiB，磁盘约0.3GiB；没有复制权重、删旧数据、改参考checkout或产品runtime/default/App。

下一步实连真实FFN：先在新owned图目录核对原MIL/blob来源与真实2D private port descriptor，再从ready原BF16 GPU input做GPU pack/cast，private paired prefix、完整GPU remainder和GPU restore/join全部计时；full/prefix/独立FP64固定预算及完整BF16 GPU分母不变。需交替真实不同timestep输入抓住compiled graph/旧output复用，不只重复同一输入；常量clone/编译存储先admission，不为低磁盘擅自清数据或降低原实验门槛。

完整目标仍未完成：更多层/独立prompt/seed/LoRA、完整block/request质量/时间、cache/load与内存/长时/失败取消；更强GPU kernel/precision/fusion、profiling/PMU/streamData/GUI；媒体/VAE/CPU/streaming/cache组合；受控多维metadata/aliases/version claims与薄AI skill继续。当前正常adoption证据不替代这些资格，goal保持active。

本批验收：269项techniques标准库测试、21项repository测试通过；techniques-only隔离副本重跑45个CPU cases、114页/1186个本地链接、53个章节catalog与17个显式device selector，无错误。完整可选来源检查核对144文件/31内容hash，来源索引`--check`通过；Copy查询先返回新来源卡，共同160000-byte预算下展开技术页/三source/八code无裁剪或省略，实际正文110580 bytes。curator复核fixed source七文件、installed383 headers/三library及五个原件，重建公开456793-byte JSON字节相同；首轮失败保留原source而不要求与更正后源码相同。当前native syntax检查无警告，默认prepare/help不需要MLX安装或设备。以上不授予compiled callable/真实FFN/全请求或整体roadmap完成资格。

## 2026-10-16续接：真实shared完整FFN、actual join与load边界

上一goal轮为progress：MLX两个导入模式、8MiB合成实际alias/owner与Copy/过期结果已经实测，仍不是完整Qwen共享路径。本轮复核roadmap同一SHA `247f7e92e49ae6ba886c6a0dca6fc53b01aa82028a40f258d4f33473121aa242`，推进真实完整消费者，不用两个toy拼接资格；全部旧改动和原件保留，不自动commit/push。

新增[真实shared FFN报告](qwen-shared-ffn-20261016.md)和[完整证据](qwen-shared-ffn-evidence-20261016.json)，实际读出原二维MIL input/output为`[1,1,1024,4096]` FP16、row8192/plane-batch8388608 bytes，而非channel/sequence小图。原BF16 checkpoint/capture、完整M1024/H4096/F12288、base/no LoRA和冻结paired prefix2048不变。原图48MiB constant blob经严格文件clone到新owned目录、source/MIL/blob hash核对，独立client不链接产品库/ANEForge实现；[Apple文件clone合同](../sources/official/clonefile.md)与设备alias分开。

实际五臂：compiled完整BF16 GPU、相同`_ANEClient`的copy serial/hybrid、shared serial/hybrid。GPU pack从原BF16 buffer写FP16 physical input；shared request实际output surface严格导入MLX、作为compiled join的运行时参数；完整GPU remainder、FP32恢复/相加/BF16 drain都计时。copy控制同入口/图/pack，额外两次FP16 CPU feature copy共16MiB，不和旧e5rt/NumPy桥接的8MiB输入+16MiB host FP32 partial混为相同copy税。

四个成功run为65/165/165/165次，共560个完整调用（500 measured、60 first/warm）。每group相同point，五臂十行位置平衡顺序；三point循环1/19/10、两point19/10和反向10/19，都是同forced-GPU轨迹下重复，不宣称新prompt/seed/layer hold-out。所有112个GPU输出对保存BF16 replica全元素误差0；所有candidate full/prefix和独立full/prefix FP64 sample过固定3%/0.999预算。全输出最大L2约0.002667，full FP64 sample0.006079，prefix full0.004585，prefix FP64 sample0.011051，未降低预算或裁小/重复真实rows。

后两run额外保留264个完整actual-part join见证：独立CPU按实际GPU BF16 tail与实际private FP16 partial做FP32加法和BF16 RNE，全部4194304元素exact mismatch0，hash相等；64个上一实际point输出对下一actual join的CPU负控每次4189812元素拒绝，不冒充新device dispatch或模型无损。早期184个candidate没有该较新字段，不回填。所有run正常完成/实际request-Metal-MLX alias/guard/unload和两份restore deleter后释放通过，没有实际取消/失败注入或物理ANE counter资格。

warm完整GPU约21.6ms。首个三point/10组shared19.345ms、paired GPU/shared1.11718×，copy/shared0.99038×；两point/30组shared18.810ms、1.14900×，copy/shared1.01154×。后两exact-join run shared18.858/18.853ms、GPU/shared1.14575/1.14715×、copy/shared1.01142/1.01097×；后者GPU/shared探索区间[1.14396,1.14766]/[1.14452,1.14958]。长确认约12.7%–13.0%只属于本warm FFN边界，不是生成wall下降15%；同client减少copy约1.1%很小且首轮方向不同，不能把全部GPU收益归因于zero external copy，也不晋升通用profile。serial约26.5–27.0ms仍慢于GPU，host API交叠不是物理placement/overlap。

一次clone/compile admission在约254MiB被原256MiB门槛拒绝，native/device未启动；另一只读prepared/load流程不创建新constant clone、不调用compile selector、不fallback编译，128MiB门槛只用于这个不同操作，并未放宽原门槛。两次load-only selector duration约650.99/32.11ms、compile=null；显式compile的两run约510.83/534.42ms另加load20.51/21.35ms。没有清系统cache，不能把API名字当内部无编译/无setup或请求break-even证明。存储波动保留观察，不把所有空间变化归因本轮，不删旧数据。

独立stdlib curator重核原始BF16 spans/capture/fixture、真实NPY row-pitch native paired-prefix、MIL/blob、per-run source/binary、383 headers/三library与owned终态receipt，重算完整顺序/gates/时钟依赖/ratio，公开不带raw路径/张量/F64向量/地址。它不重跑private输出/FP64数学，不是attestation；oracle和actual join的计算由原采集源码绑定。当前CLI后来补了prepared replay须显式benchmark及prior lineage guard，不为旧run换源码；数学witness/reader在全部数值run保持固定。每run另采NumPy/clang版本、dyld actual-loaded-image、逐arm峰值/冷/长时仍缺，不回填。

下一步优先把收益带到更广真实消费者：只读核对已有capture/client和兼容runtime是否可复用，取得独立prompt/seed、更多层或完整block输入，再做完整request/图像质量与baseline；不为实验自动重建大产品或复制全权重。常量bank/LoRA失效、load/setup和逐arm内存/低内存/长时、失败取消也会否决warm收益。更强GPU kernel/precision/fusion、profiling/PMU/streamData/GUI、媒体/VAE/CPU/streaming/cache组合以及多维metadata/aliases/version claims/薄AI skill的完整范围继续，goal保持active。

本批验收：281项techniques标准库测试、21项repository测试通过；techniques-only隔离副本重跑46个CPU cases、116页/1232个本地链接、53个章节catalog和18个显式device selector，无错误。完整可选来源检查核对147文件/31内容hash，来源索引`--check`通过；real-shared-join查询指向已有完整FFN页，128000-byte共同预算下展开技术页/五source/五core code无裁剪或省略，正文75256 bytes。2225564-byte公开证据在2100000-byte处可续读，完整curator从六份明确原件重建字节相同；native当前源码syntax检查无警告，prepared replay未显式benchmark的拒绝在无MLX/设备的子进程单测。公开capsule保留per-run源码差异/早期缺字段/inspection与storage非执行状态，不授予完整请求、物理ANE或整体roadmap完成资格。

## 2026-10-16归档整理：独立输入完整block与交付拒绝

补齐已有[完整block上下文报告](qwen-block-context-20261016.md)与[公开capsule](qwen-block-context-evidence-20261016.json)的单一登记、章节入口、可选integration说明和离线测试。本次仅整理与验证，未重跑GPU/ANE或继续压缩变体；原始采集源码、数学witness、verifier和capsule内容hash保留，没有为格式化重写已绑定的实验源码。

真实新seed43/不同prompt来自ordinary GPU request的step1/layer0 observer。原始compiled block callable为强分母，staged GPU exact ablation、实际runtime-argument零控、attention/norm与modulation/residual完整边界均保留。一份bulk capture在267MiB admission前置拒绝，一份owned TMPDIR load返回false/NSError nil，另两份完成330次完整block调用与198个actual-part exact join后，分别被raw-pair/实际大小lossless archive reserve拒绝。三份block记录实际9参数/FFN input/reference与timed-loop SHA一致；执行/数值/交付/promotion轴分开，comparisons仍null，没有新的合格倍率。TMPDIR对照不证明因果，候选未回灌20steps请求，也没有VAE decode/图片/物理ANE或完整请求资格。

新增`block-evidence-axes`进入同一catalog，纯CPU合同与只读curator不引入产品/模型依赖。15项新回归包含完整顺序/gate/alias/copy/host依赖/owner与失败晋升反例、便携C++小矩阵对math.fsum、合成archive字节roundtrip/坏header/trailing/解压cap；合成codec通过不回填真实模型archive交付。`.tcbf16z`原始激活归档加入Git及隔离副本排除，公开JSON、源码和小型NPY fixture仍可跟踪，无已跟踪文件被新规则隐藏。

本次整理验证：298项techniques测试中297通过、1项因实际不足256MiB而明确skip；原PMU admission门槛未降低，另加低磁盘必须在source/model/device工作前拒绝的回归。22项repository测试通过；techniques-only隔离副本重跑同一套测试、47个CPU cases、117页/1255本地链接、53张章节卡与18个device selector，无错误。完整可选来源检查核对149文件/31内容hash，来源索引`--check`通过；四个新增Objective-C++客户端syntax检查无警告，两个holdout/block CLI的标准库help通过。磁盘仍约176MiB，没有删除模型、旧实验或系统cache。

完整研究范围仍未完成：先处理受约束的raw持久化与复核，再谈完整请求/画质、更多层/LoRA、冷/内存/长时/失败取消；其余GPU/profiling/媒体与metadata/aliases/version claims/薄AI skill等roadmap范围不由本次整理取得资格。
