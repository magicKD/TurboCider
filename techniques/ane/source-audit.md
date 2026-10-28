# ANE 来源阅读指南：如何找到值得借鉴的实现

<!-- knowledge
id: technique-ane-source-audit
type: navigation
operator: scheduling
backend: coreml, private-ane, cpu-reference
device: cpu, ane
symptom: stale-cache, measurement
semantics: contract-only
boundary: producer-consumer
api: reference-only
-->

[ANE 学习目录](README.md) · [固定源码](sources.md) · [案例](case-studies/README.md)

这里回答“怎样从来源学到可迁移的方法”。只保留固定版本链接和阅读重点，不附逐批抓取日志、文件 hash 或来源 JSON。

## 按优化目标找原件

| 要学习的能力 | 固定来源导航 | 建议先看本库的什么 |
| --- | --- | --- |
| 布局、Conv 投影、attention 与 norm 参数迁移 | [Apple ANE Transformers](https://github.com/apple-aiml-research/ml-ane-transformers/tree/590eb35913622cdef217ebc5329047aab389ac98) | [布局配方](layout-and-lowering.md)、[完整参数与消费者](apple-transformer-norm-and-position.md) |
| attention 分块、mixed-bit、阶段资源管理 | [Apple Core ML Stable Diffusion](https://github.com/apple-aiml-research/ml-stable-diffusion/tree/ea2805dc1945be20561c77e5f6d1d9a5a637cda2) | [SD 方法导读](stable-diffusion.md) |
| prefill/decode、state、模型分块 | [ANEMLL](https://github.com/Anemll/anemll/tree/f4ad26d061dc2e426faa19f81c367343bdeb9f0d) | [公共部署](public-coreml-deployment.md)、[状态与重试](stateful-multifunction-and-retry.md) |
| runtime weights、图内 tiling、worker 与输出交接 | [vpipe emitter](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/apple-silicon/coreml/ane-emitter.cc)、[ccv bridge](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_coreml.mm) | [调用链](vpipe-callchain-and-failure.md)、[异构比较](hybrid-runtime-comparison.md) |
| 图搜索、动态参数、VJP 与 optimizer 驻留 | [ANEForge](https://github.com/sbryngelson/ANEForge/tree/896c4943c5ebf811c0e815f8d1381fd1e157fa7d) | [搜索门槛](graph-rewrite-search.md)、[训练总览](training.md) |
| delta reload 与 LoRA 输入 | [Orion](https://github.com/mechramc/Orion/tree/4dcc1c2a626c9ee1f35576bd0b290d489db17d08) | [参数更新路线](training-delta-and-lora.md)、[LoRA ABI](orion-lora-inputs-and-scalar-serialization.md) |
| host profiling、统计可见性与背压 | [Apple Core AI Models](https://github.com/apple/coreai-models/tree/3b25d934ccaee2d05edc6e281c555a5a9a13a55c) | [统计/reset/背压](coreai-profiling-reset-and-backpressure.md)、[平台门槛](coreai-blocked-attention-and-runtime-gates.md) |

这些是已固定的读本，不声称最新版本。完整项目与官方文档导航见 [总目录](README.md#直接读哪些项目)；各专题页的来源链接对应实际阅读范围，而非“已审完整仓库”。

## 读代码时沿着五个边界走

1. **输入合同**：shape、layout、dtype、mask、权重方向和参数含义。先写出函数，再读优化表达。
2. **构图/转换**：哪些算子被替换、哪些参数变成常量或运行时输入；检查实际导出图，不只看配置名。
3. **运行时**：compile/load、cache key、state、返回对象、stride 与最后读者；源码里的 fallback 要追到真正采用的分支。
4. **测量**：单位、时钟域、scope、缺失值与分母。作者报告、计划估计和本机实测应分列。
5. **消费者与失败**：完整输出、下游、尾块、重试和状态提交。小图成功不自动覆盖完整模型或训练轨迹。

## 证据标签与版本使用规则

- 官方资料、选定源码、作者报告和本库实测分别阅读；获取文件不等于读完，读过源码不等于运行过。
- 固定 commit 说明读本的版本；运行环境与当前行为仍需在自己的任务中确认，不能从阅读链接推导。
- 历史案例按原日期和原范围阅读，既不伪装为当前验证，也不要求读者拥有原始机器回执。
- 许可证根提示不替代逐文件授权；私有 Apple 接口的发行权也不能从开源 wrapper 的许可证继承。参见 [NOTICE](../NOTICE.md)。

## 从阅读到迁移

沿实际调用链追输入、图表达、运行时、消费者和失败路径。将可复用机制写成方法，将特定版本行为写成案例；新 SoC/SDK、精度或消费者变化时，重新判断前提。实现自己的实验属于新的工程任务，不是知识库阅读的先决条件。
