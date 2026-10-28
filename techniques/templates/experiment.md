# 可复用实验记录模板

<!-- knowledge
id: technique-templates-experiment
type: guide
-->

[总目录](../README.md) · [测量规则](../profiling/measurement.md)

复制到新的任务记录目录，填实际事实；`unknown`不是0，未测不是pass。原始日志/媒体/模型放ignored产物目录，仅提交便携摘要和工具。

## 问题与假设

- 症状 / 关键路径：
- 要减少的工作 / bytes / exposed wait：
- 数学是否不变；若近似，新的precision/recipe与许可：
- 为什么当前GPU基线已经足够强：
- 预设成功/否决门槛与未覆盖范围：

## 身份与输入

| 项目 | 实际值 |
| --- | --- |
| date/timezone / campaign | 待填写 |
| SoC/GPU cores/RAM/OS build/power/thermal | 待填写 |
| source commit/dirty source manifest | 待填写 |
| actual loaded CLI/library/build/source hashes | 待填写 |
| SDK/Metal/MLX/CoreML工具版本 | 待填写 |
| checkpoint/components/tokenizer/adapter/strength | 待填写 |
| prompt/seed或noise/reference顺序与SHA/resize | 待填写 |
| geometry/实际rows/steps/sigma/frame/FPS/audio | 待填写 |
| route/residency/share/bucket/chunks/precision | 待填写 |
| artifact compile/model/conditioning/prefix/cache policy | 待填写 |

## 对照与命令

写baseline、candidate、必要的same-boundaryGPU ablation；一次只改明确变量。填完整可复现命令，所有输出为新目录。说明prepare/constructor是否计时、warmup与sample数量、ABBA/BAAB顺序、采样间隔/最大gap与load规则。计时期间不运行build/tests/其他自有GPU任务。

## 原始结果

每次样本保留：status、cold/warm、requestwall、phase/block时间、actualbackend/recipe/calls、fallback/retry、quality、load/memory证据与产物SHA。失败样本不可删；只提取成功样本时必须显式取消正式资格。

## 质量与资源

- source/shape/dtype/finite/tail/完整输出集合：
- component/每step/最终latent/decodedRGB/alpha：
- whole/detail/temporal/AV人工检查范围与未检查项：
- MLX active/cache/peak、native/external、RSS/footprint/process树：
- load/cold/warm/exit覆盖、samplegap、compression/swap、未归因daemon/driver：
- 磁盘source/derived/scratch/compiled/日志与free reserve：
- cancel/retry、latechunk失败完整重算、A/B/base、timeout/latecompletion、unload：

## 分析与决定

先写完整请求结论，再解释组件；倍率与耗时下降分别算。预测/alone/concurrent/device trace各自标scope，nestedspans不相加。若component好但E2E没收益，明确保留负结果。

最终选择：默认 / 显式候选 / 低内存但慢 / research / 否决；说明精确硬件/workload范围、还有哪些门缺失、何时应重测。已有组合需另做匹配实验，不能相乘历史倍率。

## 复核

链接实际代码符号、小型机器摘要、原始记录相对ID和SHA、所有执行命令与终态；测试数附skip原因。源码/构建/实际App部署分别验，不借旧结果重标新库。
