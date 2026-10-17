# 推理引擎地图：按可借鉴机制选读本

<!-- knowledge
id: technique-gpu-ane-inference-engines
type: navigation
-->

[方法导航](README.md) · [主题索引](report-index.md) · [全部案例](case-studies/README.md) · [固定来源](../MAINTENANCE.md)

不是引擎速度排名，也不是所有项目均已在本机运行。先选瓶颈，再读host选路、实际consumer和失败边界；固定版本的源码机制、作者报告与本机实验分开阅读。

## GPU数据流、编译与提交

| 实现读本 | 最值得学习 | 迁移时的主要边界 |
| --- | --- | --- |
| [MLX GEMM](gpu-gemm-quantization.md) · [NAX选路](case-studies/mlx-nax-mid-m-and-prefetch.md) | small/mid/large-M分核、packed解码、split与预取 | 软件gate、实际dispatch与硬件支持分开；无本机M5实测 |
| [GGML few-row](case-studies/ggml-fewrow-mma-and-reuse.md) · [融合回归](case-studies/ggml-residual-identity-regression.md) | 解码复用、完整epilogue与强负控制 | 量化格式、tail、bias/residual身份 |
| [tinygrad Metal/JIT](case-studies/tinygrad-metal-hcq-and-replay.md) · [完整FFN](case-studies/tinygrad-strong-baselines-and-paired-ffn.md) | ICB重放、owner、paired投影与强GPU分母 | command数不等于速度；只覆盖固定组件 |
| [MLX编译](case-studies/mlx-compile-and-cross-runtime-ffn.md) | specialization、scalar/cache、同步完成 | 闭包常量/状态变化与跨进程窗口 |
| [MLC/TVM](case-studies/mlc-compiler-and-quantization.md) | 高层图到TIR/Metal、量化融合、权重与模型库分开 | 固定依赖、compiler/runtime分别验证 |
| [Uzu](case-studies/uzu-metal4-and-routing.md) · [vLLM-Metal](case-studies/vllm-paged-nax.md) | 提交生命周期、mixed-batch、分页KV与route | serving吞吐不等于单请求收益；NAX不是ANE |
| [ML Drift](case-studies/mldrift-metal-runtime.md) · [完整INT4 FC](case-studies/mldrift-complete-int4-fc.md) | packed格式、argument binding、shape路由与完整op消费 | 选定FC运行不代表整引擎/模型或MPP已验证 |
| [MNN布局/卷积](case-studies/mnn-metal-copy-and-convolution-abi.md) · [ncnn](case-studies/ncnn-vulkan-int8-and-apple-policy.md) | C4/batch、staging、packing与平台策略 | Vulkan/MoltenVK不是Metal原生API，不能继承性能结论 |
| [MetalPetal](case-studies/metalpetal-render-graphs-and-pixelbuffer-handoff.md) | 图融合、精度语义、PixelBuffer与最后读者 | 中间材质裁剪不能随意删除；不是AI模型速度证据 |

更广泛的Candle、MPS、MFA等单GPU实现由[Metal引擎手册](../metal/inference-engines.md)维护；本目录不重复做一套通用引擎百科。

## GPU/ANE切分与公共委托

| 实现读本 | 最值得学习 | 迁移时的主要边界 |
| --- | --- | --- |
| [vpipe](case-studies/vpipe-runtime-weights.md) · [控制器](case-studies/vpipe-controller-and-failure.md) | runtime weights、封闭FFN、启停/份额两层控制 | 数值缩放、bias/LoRA、接缝和完整失败恢复 |
| [Draw Things→ccv](case-studies/drawthings-to-ccv-hybrid-route.md) · [成本](case-studies/ccv-hybrid-cost-and-failure-boundaries.md) | 上层资格到rowwise bridge、在线chooser | 真正触发路线、concurrent成本和scale-floor |
| [Splash](case-studies/splash-w8a8-prefill-and-layout.md) · [完成门](case-studies/splash-command-capacity-and-watchdog.md) | paired-channel/W8A8、原pitch、owner与退役 | 私有ABI、量化质量、强制signal不等于drain |
| [ORT CoreML](case-studies/onnxruntime-coreml-partition-cache.md) · [ExecuTorch CoreML](case-studies/executorch-coreml-delegation.md) | 保留高层op、减少trivial小岛、资产共享 | 具体node support、fallback、实际backing与请求state |
| [MNN CoreML](case-studies/mnn-coreml-graph-boundaries.md) · [LiteRT](case-studies/litert-coreml-partition-and-fp16.md) | 封闭预测、张量交换、分区与生命周期 | partial/full分区、低精度、错误返回和owner |
| [ggml-ane](case-studies/ggml-ane-backend-cache-and-failure.md) · [ANE-LM](case-studies/ane-lm-programs-state-and-numerics.md) | 常量cache、完成合同、多步state与公式 | 原插件/原生执行范围不同；不默认启用私有后端 |
| [公共Metal4 ML pass](case-studies/metal4-machine-learning-passes.md) · [完整消费者](case-studies/metal4-consumer-layout-and-handoff.md) | 图专门化、实际layout和GPU→ML→GPU链 | API成功不证明物理ANE、overlap或内部零拷贝 |

## 模型阶段与权重工作集

[Whisper encoder/decoder](case-studies/whisper-coreml-metal-stages-and-real-asr.md)、[FastVLM vision/语言](case-studies/fastvlm-coreml-vision-and-mlx-handoff.md)、[SD.cpp分段](case-studies/sdcpp-metal-residency-and-segments.md)、[MoE bank](case-studies/anemll-original-bank-gpu-consumer.md)分别提供阶段边界、驻留与预取的不同设计。按[阶段方法](stage-pipelines.md)和[驻留方法](residency-and-prefetch.md)比较完整消费者，不从项目名称推断最佳后端。

## 怎样比较与采用

固定checkpoint/adapter、格式、shape/layout、输出需求和完成边界；同时记录cold/warm、prepare/交接/归约、完整请求与峰值。先建立成熟GPU分母，再比较一项变化，保留失败和回退。具体方法见[验收](validation-and-fallback.md)。

原逐批深度标签与演进记录移至[历史引擎账本](../MAINTENANCE.md)。历史PR状态不刷新为当前状态；本次整理没有新增引擎、模型或设备benchmark。
