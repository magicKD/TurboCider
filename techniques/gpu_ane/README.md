# Apple Silicon GPU + ANE 协同推理研究

核查日期：2026-10-16。目标是解释**实现机制、适用条件、失败原因和复现办法**，不是给不同模型/精度/机器做混杂的速度排行榜。本专题按用户指定使用下划线目录；原有 [`gpu-ane/`](../gpu-ane/partitioning.md)、[`metal/`](../metal/kernel-tuning.md)、[`ane/`](../ane/coreml-graphs.md) 保留为基础手册，不搬迁、不复制外部引擎。

## 先读结论

1. **先优化 GPU，再决定是否借 ANE。** packed GEMV、Flash-style attention、融合和提交复用改变分母后，原先合算的 ANE share 可能不再合算。
2. **GPU 内的 NAX/Neural Accelerators 不是独立 ANE。** M5 TensorOps 的收益与 GPU+ANE 协同是两条不同路线；本机 M4 Max 无法验证 M5 硬件快路径。
3. **最值得探索的是大而封闭的子图。** FFN 内保留 gate/up→激活→down，避免大 intermediate 穿过设备边界；小型 decode 要特别警惕 dispatch 和权重带宽。
4. **统一内存不是无成本共享。** 真实 stride、格式转换、backing 是否被采用、完成事件、owner 和 slot generation 都要验证。
5. **没有通用 50/50 或通用“INT8 更快”。** 选型依据是 concurrent 下的完整消费者耗时、误差、驻留和冷启动；不是 isolated TOPS 或模型文件体积。

以上是跨来源与本库实验的综合判断；各项证据、版本和反例分别在下表中展开。

## 专题报告与学习顺序

| 顺序 | 报告 | 回答的问题 |
| --- | --- | --- |
| 1 | [硬件、API 与 M5 TensorOps](hardware-and-tensorops.md) | SIMD-group、cooperative tensor、NAX、ANE、版本门槛有什么区别？ |
| 2 | [推理引擎地图](inference-engines.md) | MLX、GGML、MFA、MPS、MNN、Candle、MLC、Uzu、vLLM-Metal 分别优化哪一层？ |
| 3 | [GPU GEMM、量化与布局](gpu-gemm-quantization.md) | 小 M 与大 M 为何换核？低比特如何解码、复用和融合？ |
| 4 | [GPU attention、融合与提交](gpu-attention-runtime.md) | 怎样避免 score 矩阵？lazy graph 与 command overhead 怎么测？ |
| 5 | [ANE 图、尺寸与量化](ane-graphs-and-quantization.md) | BC1S、Conv/Linear、图内 tile、W8/W8A8/LUT 到底改变什么？ |
| 6 | [私有 ANE 与图搜索](ane-private-and-search.md) | ANEForge、ANE-LM、Field Guide 和直接训练研究有哪些可靠边界？ |
| 7 | [异构切分与关键路径](partition-and-scheduling.md) | row/F/channel/branch/stage/pipeline 有何不同，什么时候不合法？ |
| 8 | [vpipe 源码案例](vpipe-runtime-weights.md) | 一份 runtime-weight 图如何服务多层，如何隐藏 staging 并允许退回 GPU？ |
| 9 | [ccv / Draw Things 源码案例](ccv-rowwise-iosurface.md) | rowwise INT8、K-split、IOSurface、fast fence 与 output backing 如何衔接？ |
| 10 | [其他协同项目与模型部署](hybrid-projects-and-stages.md) | Splash、FLUX.2、whisper.cpp、ANEMLL、Core AI、ggml-ane 分别属于哪种协同？ |
| 11 | [共享内存与消费者 ABI](memory-and-handoff.md) | 如何防止“零拷贝”却错 stride、读到旧结果或提前释放？ |
| 12 | [优化 PR 与演进记录](optimization-prs.md) | 哪些改动已合并、哪些仅是实验提案，值得迁移的机制是什么？ |
| 13 | [实验与已有证据](experiments.md) | 本机测到了什么、哪些只是上游报告，怎样安全复测？ |
| 14 | [来源、阅读范围与覆盖审计](sources-and-coverage.md) | 怎样追溯固定源码、官方文章和路线图要求？ |

## 按工作负载选路线

| 场景 | 优先基线 | ANE 候选 | 主要否决条件 |
| --- | --- | --- | --- |
| LLM 单 token decode | packed GEMV、KV cache、减少 launch | fused projection/FFN、低功耗背景任务 | ANE dispatch/权重流式读取压倒计算；不拿 CPU 分母宣传 |
| LLM prefill / encoder | tiled GEMM、fused SDPA、chunked prefill | 固定 bucket、封闭 Transformer 子图 | mask/KV/state/shape 语义不匹配，padding 太多 |
| 大 token DiT FFN | 优化 full GPU FFN | row split、paired intermediate split | staging 暴露、共享带宽争抢、FP16 溢出、LoRA 不一致 |
| 真正 parallel block | 同输入独立 branches 的 GPU 图 | attention GPU + MLP ANE | 把 sequential block 错当独立分支；漏掉输出投影 |
| VAE / 视频卷积 | 成熟卷积实现、精确 tiling | 可封闭且高复用的 conv 子图 | halo/norm/causal history 或完整交接开销抵消收益 |
| 多请求/音频流水 | cache、batching、有界队列 | encoder ANE + 其他请求 decoder GPU | 只改善吞吐却宣称单请求延迟；状态串请求 |

不默认引入生产依赖，不修改 TurboCider 的默认后端、系统安全设置、wired limit 或模型。研究实现存在、编译成功、数值通过、组件提速、整请求提速是五种不同状态。

## 如何使用

先按上表挑一个真实消费者，读机制页和对应源码案例；再沿 [实验页](experiments.md) 选择 CPU 数学演示或显式设备实验。每次变更 checkpoint/LoRA/精度、布局、SoC、OS/SDK 或 GPU 基线后重新验证。来源与测试通过不会自动授予产品启用资格。

交付核查：[来源内容与实验/离线检查摘要](validation.json)；[本专题8项标准库测试](../tests/test_gpu_ane_survey.py)。原始源码/编译物仅留ignored outputs，核心报告和整理数据可随本目录分发。
