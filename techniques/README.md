# Apple Silicon 模型优化知识库

<!-- knowledge
id: technique-readme
type: navigation
-->

为工程师和 AI 保存可迁移的优化认识：**解决什么问题、为什么有效、怎样实现、付出什么代价、如何验证**。重点是 Metal GPU、ANE、统一内存与图像/视频模型，也包含可迁移的 LLM 和语音实现参考。

这里不是产品开关清单，也不是逐轮实验工作目录。阅读笔记可以在尚未本机验证时收录，但必须区分来源事实、推导、假设和实测；收录不代表产品采用。

## 按问题开始

| 当前问题 | 先读的方法 | 再看什么 |
| --- | --- | --- |
| 不知道模型慢在哪里 | [优化工作流](guides/optimization-workflow.md) · [测量](profiling/measurement.md) | [症状排查](guides/troubleshooting.md) · [工具](profiling/tools.md) |
| GEMM / FFN / 单 token 慢 | [GEMM/GEMV 与量化数据流](metal/gemm-and-quantization.md) | [GPU 方法导航](metal/README.md) · [实现案例](metal/case-studies/README.md) |
| attention 或 KV 成本高 | [Attention 实现](metal/attention-implementations.md) | [Prefill](models/llm-prefill.md) · [Decode](models/llm-decode.md) |
| 小算子、复制或提交过多 | [融合与布局](metal/fusion-and-layout.md) · [Command graph](metal/command-graphs.md) | [编译与状态](metal/mlx-compile-cache-and-state.md) |
| 想利用 ANE 或 GPU+ANE | [ANE 方法导航](ane/README.md) · [协同优化方法](gpu_ane/README.md) | [切分代数](gpu-ane/partitioning.md) · [共享交接](gpu_ane/memory-and-handoff.md) |
| 多步生成计算重复 | [算法方法地图](algorithms/methods/README.md) | [实现技巧](algorithms/implementation-patterns.md) · [质量与评测](algorithms/evaluation.md) |
| 内存不足或 I/O 太慢 | [驻留](memory/residency.md) · [Streaming](memory/streaming.md) | [存储格式](memory/storage.md) · [VAE](media/vae.md) |
| 接入新模型或迁移实现 | [阶段地图](models/stage-map.md) · [新模型接入](guides/new-model.md) | [设计样本](designs/README.md) · [完整走查](guides/worked-example.md) |

## 怎样阅读和检索

日常顺序是 **方法 → 实现思路 → 案例/反例 → 固定来源**。这里不分发实验 JSON、验证回执或复算环境；案例数字只保留为有范围的历史观察。完整约定见[使用指南](guides/knowledge-base.md)和 [AGENTS.md](AGENTS.md)。

```sh
python3 -B techniques/scripts/query.py DeepCache --intent learn --limit 3 --compact
python3 -B techniques/scripts/query.py small-M --intent implement --limit 3 --compact
python3 -B techniques/scripts/query.py MLX --intent sources --limit 3 --compact
python3 -B techniques/scripts/get_page.py gpu_ane/memory-and-handoff.md --follow-sources
```

检索直接读取 Markdown 标题、链接和短主题注释，不依赖 JSON catalog，不运行模型或设备。[主题索引](queries/README.md)提供阅读导航，[小型代码](examples/README.md)帮助理解数学与数据流。

## 内容地图

- 方法与设计：[Metal](metal/README.md)、[ANE](ane/README.md)、[GPU+ANE](gpu_ane/README.md)、[算法](algorithms/README.md)、[设计样本](designs/README.md)、[硬件](hardware/apple-silicon.md)。
- 模型与媒体：[阶段地图](models/stage-map.md)、[迁移配方](models/recipes.md)、[GPU/ANE 选型](models/hybrid-selection.md)、[VAE](media/vae.md)、[视频音频](media/video-and-audio.md)、[超分](media/upscaling.md)、[纹理交接](media/tensor-texture-interop.md)。
- 案例与反例：[GPU 案例](metal/case-studies/README.md)、[ANE 案例](ane/case-studies/README.md)、[协同案例](gpu_ane/case-studies/README.md)、[算法案例](algorithms/case-studies/README.md)、[媒体案例](media/case-studies/README.md)、[早期协同案例摘要](gpu-ane/experiment-ledger.md)。
- 使用与贡献：[工程指南](guides/README.md)、[技术卡](templates/technique-card.md)、[实验设计模板](templates/experiment.md)、[工具](scripts/README.md)。
- 来源与边界：[固定版本读本](sources/README.md)、[参考项目](references/related-projects.md)、[怎样读来源](references/evidence-map.md)、[历史经验](references/history-lessons.md)、[KernelWiki 导读](references/kernelwiki-sana.md)、[署名与许可](NOTICE.md)、[维护规则](MAINTENANCE.md)。

`gpu-ane/`保留基础代数、ABI 和早期设计；`gpu_ane/`按问题组织跨引擎方法与案例，引用前者而不重复维护通用原理。实验资产已移到本地备份，不属于知识库或普通分支推送。
