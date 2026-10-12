# 原目标与最新约束逐项核验

2026-10-13。权威输入为59组sample catalog、保留的best-local-configs、
source/实际请求/完整媒体/独立内存证据，不以目录、标签或测试绿色代替实测。
本轮新增此前未测的FLUX4派生GPU/混合候选：4进程/16PNG/4complete memory，
没有重跑原packed GPU、参考或其它TC模型。所有owned实验均已terminal。

原目标为本地base模型、GPU/最佳GPU-ANE、BF16/量化、常见框架与完整
耗时；随后限定FLUX4B/Z/Qwen2.1/LTX、9B暂缓、不兼容外部略过、报告
收敛、TC取已有最优。新候选补原扩大目标的实际缺口，不重新开已测网格。

| 要求 | 权威证据 | 核验结论 |
| --- | --- | --- |
| 四家族/base-only，无额外LoRA | [主报告](PERFORMANCE_REPORT.md)、catalog47active/12历史9B、实际requests与source profiles | 四家族有完整媒体；不混LoRA/PE overlay、9B不代4B |
| 简洁且详细的docs/benchmark | [Comfy重点表](COMFYUI_MODEL_COMPARISON.md)、逐家族/表示页、逐请求JSON | 耗时、绝对差、倍率、来源与样本数可查；当前catalog重导出相等、214本地链接已核验 |
| 原BF16与本地量化表示 | Z五表示16配置、Qwen原Q4_K_M/BF16/mflux4/8、4B原savedQ4及single40派生、LTX ConvRot | 完整任务证据；不同recipe不假定bit-exact或统一成原BF16 |
| 纯GPU与已有最佳GPU/ANE | Z frozen/Private/Public、Qwen frozen/Private GGUF、LTX GPU/INT8/FP16/Sol/text256；[4B新派生完整组](../design/validation/flux4-derived-hybrid-current-20261013.json) | 四家族均有内部GPU/混合测量；LTX更快GPU保留；4B派生3.483532/2.574438s，W8/F16不是W8A8激活 |
| 本机参数探索与保存最优配置 | Z14-share/join、Qwencache/residency/多精度、LTX8完整配置、4B单候选正反序；best-local-configs | 保存实际赢家，不宣称穷尽shape/global最优；4B配置源/manifest/库绑定 |
| Unsloth | Qwen GGUF Metal、Z原BF16 CPU/GPU text完整对照 | 使用更快Z GPU-text分母；不推算未运行的Desktop路径 |
| mflux | Qwen/Z0.22完整cohort、4B原savedQ4两个fresh完整参考 | 已测；Qwen pureGPU参考胜出保留；4B与新派生视图不同窗口/表示 |
| mlx-serve | 4B原savedQ4 4-step SSE/8PNG参考组之一；LTX原组合memory preflight拒绝 | 4B成功；LTX拒绝审计，不强适配、不用失败时延作分母 |
| Comfy Mac实现与性能 | v0.39/PyTorch2.12.1 MPS，Z32PNG、Qwen12PNG、LTX两97帧AV MP4 | 三家族成功；4B原saved-MLX whole index不是stock UNET choice，按兼容政策N/A，不声称所有loader失败 |
| 完整encoder/DiT/VAE/输出耗时 | PNG/AV MP4 wall、actual steps/events；cold/fresh/repeat分列 | LTX单Comfy warm不是中位数；回用TC最佳记录不是新paired；4B新组有实际80calls/100block events |
| SHA只初次验证 | source-generation注册/metadata guard、Qwen验证bytes0回归、4B已有注册复用 | 不每次inference重扫；新来源/ANE session初次绑定成本计入cold |
| 宽松delta但披露质量/内存 | LTX opt-in MAE25.03/255、Z/Qwen漂移、4B32原尺寸/原通道PNG对照MAEmax2.573/SSIMmin.98059 | 不当无损、hardRAM、物理ANE/INT8 MAC或任意prompt/shape资格 |
| 已测TC不用充分重测 | Comfy reference-only、4B参考native_retested=false、新派生原packed/external_retested=false | 复用旧记录，仅补新来源recipe自身的GPU control与混合，不重跑旧网格 |

## 当前收尾状态与不适用项

此前缺的当前4B来源绑定混合时延已补。它是当前Q4/g64系数的BF16
预解码派生视图，不是直接UInt32 packed hybrid；原可运行配置仍保留
gpu_ane_current_q4_result: null，并新增gpu_ane_derived_q4_seconds及
源绑定的完整最优配置。不能把派生结果或旧BF16 1.628s填成直接packed
时延。原packed与新GPU的本组输出exact也只证明有限prompt/seed。

Comfy4B当前格式缺少完整分母、LTX原组合mlx-serve拒绝按用户要求N/A，
不为本次bench下载/强制转换外部格式。更多设备、直接packed hybrid、
Public发布、hardRAM、物理trace、全局画质/形状资格不由本报告证明。

**原benchmark目标已完成，按照用户后续明确的四家族/兼容/复用范围验收。**
不是只完成Comfy子任务：四家族GPU与混合、可用表示、已运行兼容框架、
完整耗时/媒体、实测配置探索与最优保存均有上表对应证据。未实现的
直接packed hybrid不冒充派生结果；它也不否定实际完成的当前来源混合测量。

收尾核验：59个原始evidence文件SHA与catalog全部一致，189条route
的样本数/中位数逐一重算；catalog与源码重导出逐字段相等。四家族
5个当前保存配置的模型/库/manifest/CLI/派生receipt全部存在；5项同
cohort优势、4项Comfy复用、2项FLUX4复用分别保留，不相互升级。
214个benchmark本地链接通过；各实际媒体/步数/runtime/memory/source
的协议与SHA链见上述验收记录和定向回归。make test-base-benchmark
241项通过，另有1组历史fixture跳过；没有运行全量make test，
没有把一组历史fixture跳过写成通过；没有因benchmark完成修改各实验
原有qualification_passed=false或goal_complete=false。

本报告完成不是Public发布、hardRAM、物理ANE trace、任意形状/画质
或所有设备的资格。历史审计/计数见[完成记录](COMPLETION_AUDIT.md)，
不是新执行队列。所有owned模型/导出/编译进程已终态，未改用户Comfy服务。

[逐请求catalog](results-20261011.json) · [已测配置](best-local-configs-20261011.json)。
collector-only阶段名恢复保留原failed summary；inconclusive memory仍拒绝。
