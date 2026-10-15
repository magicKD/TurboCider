# 知识库覆盖与清理审计

[总目录](../README.md) · [本轮验证](validation.md)

核查对象是用户要求的可复用优化百科，而非“所有新硬件/模型都已提速”。每行分别检查内容、证据和新项目可执行性；自动链接/测试不能代替这里的语义审稿。

后续实现扩展已增加 [实验室](../labs/README.md)、[28项逐章代码/case/controls映射](../labs/catalog.json)、[SOP](../guides/sop.md)与 [技术卡](../templates/technique-card.md)。本次“每个技术子项有实现参考”的覆盖由独立catalogcheck与人工匹配共同检查，不只看Markdown数量；真实运行与限制见 [扩展验证](labs-validation.md)。

| 原始要求 / 关键面 | 本地交付证据 | 审查结论 |
| --- | --- | --- |
| techniques分类百科 | [总目录](../README.md)，按guides/hardware/metal/ane/gpu-ane/memory/algorithms/media/profiling/models/references组织 | 有问题入口与主题入口，不按日期堆日志 |
| Metalkerneloptimization | [kernel](../metal/kernel-tuning.md)、[attention](../metal/attention.md)、[fusion/layout](../metal/fusion-and-layout.md)、[graphs](../metal/command-graphs.md) | 涵盖tile/SIMD/MPP、tails、finite、storage、graph/owner与负结果 |
| ANEoptimization | [CoreML](../ane/coreml-graphs.md)、[lifecycle](../ane/lifecycle.md)、[quant](../ane/quantization.md)、[limits](../ane/performance-limits.md)、[private](../ane/private-ane.md) | 明确frozen/runtime/W8A8、layout/bucket、SRAM边界、来源/发布 |
| GPU/ANE并行重点 | [partition](../gpu-ane/partitioning.md)、[sharedIO](../gpu-ane/shared-memory.md)、[schedule](../gpu-ane/scheduling.md)、[LoRA](../gpu-ane/lora.md)、[calibration](../gpu-ane/calibration.md) | 从代数到实际完成、双bank、row/phase调度与完整fallback，未只列开关 |
| 不同芯片/ANE特点 | [hardware](../hardware/apple-silicon.md)、M5Z/Qwen专项来源 | profile精确到RAM/OS/workload；M1–M3未测不造数，不把M5GPU加速器叫ANE |
| 图像视频cache/fusion等 | [caches](../algorithms/caches.md)、[approximation](../algorithms/approximations.md)、[VAE](../media/vae.md)、[media](../media/video-and-audio.md)、[upscale](../media/upscaling.md) | exact与lossy分开；globalstats、causalhistory、完整帧/FPS/alpha包含 |
| streaming/内存/IO | [residency](../memory/residency.md)、[streaming](../memory/streaming.md)、[storage](../memory/storage.md) | trunk/activations/slots/external/VAE/磁盘、alignment/errorlatch均有 |
| 读TurboCider文档/代码/历史 | [evidence map](../references/evidence-map.md)、[sources.json](../references/sources.json)、[history](../references/history-lessons.md) | 当前代码、datedreport、machineJSON、Gitdiff和聊天线索交叉；未冒称全量审读 |
| Splash/vpipe等参考 | [Splash](../references/splash.md)、[vpipe](../references/vpipe.md)、[others](../references/related-projects.md) | 直接读本地来源，精确head/关键文件hash、许可证、迁移限制与可选依赖 |
| scripts/profiling/参考代码 | [scripts](../scripts/README.md)、[工具箱](../profiling/tools.md)、[CPUFFN](../examples/split_ffn_reference.py) | 4个无依赖工具+代数demo+测试，优先链接已有真实设备工具 |
| 问题/可能解决方法/比较分析 | [troubleshooting](../guides/troubleshooting.md)、[20cases](../gpu-ane/experiment-ledger.md) | 症状/先查证据/下一步；正负结果/范围/分母分开 |
| 新模型能否按wiki实施 | [newmodel](../guides/new-model.md)、[workedexample](../guides/worked-example.md)、[checklist](../templates/new-model-checklist.md)、[recipes](../models/recipes.md) | 已从真实接口做桌面走查，CPU代数/预算可跑；未虚称新模型GPU资格 |
| 干净且可持续完善 | 入口、来源manifest、bounded辅助工具、覆盖/验证记录、根README导航 | 不复制大型产物，不改runtime/default，不引入第二套模型/内存系统 |

## 审稿发现后修正的关键问题

1. 旧“private只在experimental/”与后来nativeprivate实现不一致：按历史/当前/可分发三层写清。
2. 旧“FLUX9没有ANE”不再作为永久能力结论：路由到后续专项，不复制陈旧总表。
3. 早期calibrationtoy与后来integratednative模型callbacks不同：对照当前源码补正，合成activation/FFN试验仍不叫完整trajectory。
4. M5不能只写Z：补Qwen24GiBlayerstream/suffix/临时MLXwired恢复与held-out质量限制。
5. VAE实验和codec长循环不能授予fullVAE/低RAM/长模型视频资格：单列phase与121帧限制。
6. 辅助比较工具不应自动输出“资格通过”：即使数据标签全pass也始终`qualification_passed=false`。
7. 链接审计中一个猜测的`stage_executor.cpp`路径不存在：已改为实际`context.cpp/StageExecutor`。

## 刻意不冒充已完成的范围

未重新采集GPU/ANEhardwaretrace、全模型ABBA、多设备/低RAM媒体矩阵或新App部署；这些属于以后实际优化的资格任务，不是本次wiki已经获得的新成绩。上游最新代码、所有Codex历史、每个论文方向也未穷尽。对应缺口在相关章节保留明确实验入口，不用猜测数值填空。

本手册已覆盖本仓库主要图像生成、编辑、视频/音频生成和超分路线；后续新模型若有MoE、不同activation/动态shape或新的时域架构，应扩展事实表与实际验收，而不是未经验证复用已有数字。
