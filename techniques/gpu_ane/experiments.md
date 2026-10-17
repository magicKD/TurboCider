# 实验与反例：按验证问题选择案例

<!-- knowledge
id: technique-gpu-ane-experiments
type: case-study
-->

[方法导航](README.md) · [离线方法实验](examples/README.md) · [来源与证据](sources-and-coverage.md)

本页是学习索引，不是新benchmark，也不按研究批次排名。数字保留各案例的输入、环境和计时分母；不同范围的收益不能相乘或组成同一引擎排行榜。

| 验证问题 | 具体案例 | 可以学习什么，不能推出什么 |
| --- | --- | --- |
| 提交减少是否真的更快 | [tinygrad完整FFN](case-studies/tinygrad-strong-baselines-and-paired-ffn.md) | 强GPU分母、4/3/2-command、分支误差与配对统计；更少command没有通用排名 |
| 编译消除了哪层开销 | [MLX compile与FFN](case-studies/mlx-compile-and-cross-runtime-ffn.md) | scalar/cache、同步完成、跨进程窗口；不把source route当实际trace |
| FFN分担是否抵过交接 | [真实Qwen shared FFN](case-studies/qwen-shared-ffn-consumer-cost.md) | 固定layer0 warm组件约1.15×，同client去copy约1.1%；不是全模型或全收益来自零拷贝 |
| 实际backing/stride是否被正确消费 | [Metal4消费者](case-studies/metal4-consumer-layout-and-handoff.md) · [Whisper完整交接](case-studies/whisper-generation-and-complete-encoder-handoff.md) | 读真实返回布局并检查完整下游数组；不证明物理ANE/内部零拷贝 |
| 量化finite是否足够 | [ccv scale-floor](case-studies/ccv-dense-precision-and-scale-floor.md) | 稠密与小幅值行负例；不将独立诊断当所有模型故障 |
| 融合是否保持原精度语义 | [MetalPetal](case-studies/metalpetal-render-graphs-and-pixelbuffer-handoff.md) | 中间UNorm裁剪被移除可改变结果；不是所有fusion都错误 |
| bank/I/O是否输出正确专家 | [MoE I/O](case-studies/anemll-flash-mlx-banked-streaming.md) · [完整GPU消费者](case-studies/anemll-original-bank-gpu-consumer.md) | short-read、slot有效性与三arm数值；不是SSD/整模型性能 |
| 单阶段换后端是否提高完整请求 | [Whisper真实ASR](case-studies/whisper-coreml-metal-stages-and-real-asr.md) | first/warm、资产来源、同转录与encoder差异分层；不证明ANE通用胜出 |
| 导出/状态元数据是否可靠 | [FastVLM](case-studies/fastvlm-coreml-vision-and-mlx-handoff.md) · [ANE-LM](case-studies/ane-lm-programs-state-and-numerics.md) | 原JSON片段与独立多步数学分别标注，不能冒充模型运行 |

## 怎样读一个实验

先找真实消费者、固定输入/权重、基线与唯一变化；再读完整输出/反例、全部计时和未验证项；最后才判断是否值得迁移。来源JSON核固定代码，结果JSON用于复算，交付测试数量不证明优化收益。

原始GPU小链、SDPA、CoreML压缩数据及逐批补充保留在[历史实验账本](../MAINTENANCE.md)，批次报告见[归档](../MAINTENANCE.md)。原历史计数和PR状态不刷新为本次结果。当前文档结构和离线检查见[维护说明](sources-and-coverage.md)。
