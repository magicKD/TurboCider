# 阶段划分与流水：从完整请求寻找稳定交接点

<!-- knowledge
id: technique-gpu-ane-stage-pipelines
type: method
stage: encoder, dit, vae, io, pipeline
operator: scheduling, codec
backend: cpu-reference
device: cpu, gpu, ane, media-engine
symptom: memory-bound, compile-bound, measurement
semantics: contract-only
boundary: request
api: reference-only
-->

[方法导航](README.md) · [切分代数](partition-and-scheduling.md) · [共享交接](memory-and-handoff.md)

## 什么时候用

音频encoder/decoder、视觉encoder/projector/语言层、扩散denoiser/VAE可以选择不同执行入口。阶段足够大、边界稳定时，往往比零散offload小算子更值得先验证；但依赖链本身不会因为用了两设备而变成并行。

```text
单请求：准备 → encoder → 真实张量交接 → decoder → 完整输出
跨请求流水：仅当输入/状态独立、设备资源允许时，重叠不同请求的阶段
```

流水主要影响吞吐、排队和尾延迟，不自动降低单请求latency。相机丢旧帧/只保留最新帧是背压策略，不是硬件流水证明。

## 从实现提炼方法

| 实现 | 可借鉴方法 | 应保留的边界 |
| --- | --- | --- |
| [whisper.cpp固定CoreML bridge](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/src/coreml/whisper-encoder.mm) · [真实ASR](case-studies/whisper-coreml-metal-stages-and-real-asr.md) | 替换封闭encoder，保留既有decoder与显式IO | 同文本不证明encoder逐值一致或ANEplacement |
| [Whisper完整encoder交接](case-studies/whisper-generation-and-complete-encoder-handoff.md) | 按实际stride读取逻辑元素，比较下游实际收到的完整数组 | 观察器/回读会扰动计时；观察实验与速度实验分开 |
| [官方FastVLM固定实现](https://github.com/apple-aiml-research/ml-fastvlm/blob/6f7b13179bbc81e6aa7d921bc7c6335c43904778/app/FastVLM/FastVLM.swift) · [接缝分析](case-studies/fastvlm-coreml-vision-and-mlx-handoff.md) | 封闭vision阶段；token、shape和内存布局显式匹配语言消费者 | 当前本地只运行原JSON片段，没有模型/App或异构速度实测 |
| [Apple扩散pipeline](case-studies/apple-coreml-pipeline-lifecycle.md) | 阶段驻留、chunk、preview与取消路径一起设计 | 预览可能改变denoiser期间的峰值工作集 |

## 怎样选择交接点

优先选择输出较小、dtype/layout稳定、无隐藏共享状态且后续消费者明确的边界。比较转换、copy、同步、加载与模型驻留成本；避免为了局部更快把大量activation来回搬运。

Vision到语言还要匹配图像token ID、patch/token数量、projector维度和特殊token规则；音频还要匹配mel、有效长度、encoder shape与decoder状态。不能只比较元素数量或最终文本。

## 最小验证与计时

1. 固定checkpoint、转换recipe、输入前处理与原GPU完整请求。
2. 全量比对实际producer逻辑输出与consumer实际接收值；覆盖变化输入和A/B/A，排除旧结果。
3. 分开load、first、warm、TTFT、完整response、排队和回读；TTFT不是端到端响应时间。
4. 引入并发前明确每请求state、owner、generation和有界队列。取消先停止后续提交，已提交工作完成后再复用资源。

对未观测的物理placement、overlap或内部copy保持unknown。通用阶段选择见[模型阶段地图](../models/stage-map.md)，CPU协议学习见[方法实验](examples/README.md)。
