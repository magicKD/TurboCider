# Apple Silicon GPU / Metal 优化技巧与实现参考

<!-- knowledge
id: technique-metal-readme
type: navigation
-->

[百科总目录](../README.md) · [实现案例](case-studies/README.md) · [离线示例](examples/README.md) · [源码阅读](sources.md)

本目录是按瓶颈组织的参考手册：每种技巧回答适用场景、作用原理、实现/源码、代价和验证方法，不要求先了解研究批次。整理日期：2026-10-17。顶层只放方法与选型；历史观察与反例进入案例；不分发实验数据或验证包。

先按下表选择一个方法，再读对应源码和案例；动手从[离线学习示例](examples/README.md)开始。本机案例来自 M4 Max，没有 M5/NAX 实测；GPU 内 NAX 与独立 [ANE](../ane/coreml-graphs.md)分开，跨设备见 [GPU/ANE 交接](../gpu-ane/shared-memory.md)。

## 可复用的六条原则

1. **优先改变数据流与计算需求，再考虑重写 kernel。** 少做无用输出、保留 producer-consumer 布局、融合短算子、避免显式 score/权重展开，通常比单纯增大 tile 更值得先验证。[llama.cpp 的 last-only](case-studies/llama-pretrained-output-demand.md)已观察到末层 FFN/head 从 39 行减至 1 行；这是需求传播的实例，不能照搬到需要全部 image tokens 的 DiT，也不保证浮点逐位相同。
2. **prefill、大矩阵与单路 decode 要分开优化。** 小 M 重点是权重流量、寄存器复用与提交成本，大 M 重点是 tile 复用与占用；split-K 用额外 scratch/reduction 换并行。MLX 的 [wide GEMV](case-studies/mlx-small-m-wide-gemv-and-head-attribution.md)与[实际路由观测](case-studies/mlx-attention-routes-and-wide-gemv-observed.md)是学习 host selector→shader 的好入口，不能只凭函数名或 shape 推断安装版本走了哪条核。
3. **量化是压缩格式与消费方式的组合，不是“位宽越低越快”。** packed 直接消费、tile 内解码、常驻展开、逐次展开的成本不同。[真实 Qwen FFN 对照](case-studies/affine-ffn-cost-and-resident-tradeoff.md)展示了常驻展开的小收益与额外完整权重内存之间的取舍。应比较完整消费者的延迟、内存和质量，而非只报 decoder/kernel 倍率。
4. **融合、编译与 replay 解决不同层的开销。** [算子融合](fusion-and-layout.md)减少中间读写，[MLX compile](mlx-compile-cache-and-state.md)处理图与 specialization，[command graph](command-graphs.md)处理编码/提交与资源存活。三者可以组合，但先分别归因；首编译、重新分配、LoRA/权重更新、取消和回收都要入账。
5. **KV 与服务层优化必须从 writer 一直读到 consumer。** 分页、压缩、分块 prefill、连续 batching 都可能改变 GPU 的实际输入和工作量；page/stride、有效长度、lookahead、cache 身份与完成语义是算法的一部分。参考 [paged runtime](vllm-metal-paged-runtime.md)、[压缩 attention 对照](case-studies/compressed-attention-crossover.md)和[连续 batching](case-studies/mlx-lm-continuous-batching-and-cache-handoff.md)，不要把单个 gather 的收益当整个服务吞吐。
6. **用证据限定结论，不建立脱离工作负载的全局排名。** 完整预训练模型、全词表检查、dispatch/PSO 关联、CB 与 encoder-stage 观测回答不同问题；严格 tensor 预算、任务分布、观察保值、性能收益是独立结论。[held-out NLL/KL](case-studies/qwen-heldout-distribution-scoring.md)与[权重来源配对](case-studies/qwen-gguf-baseline-lineage.md)说明“小分布差异”与“逐元素门槛拒绝”可以同时成立，不能直接归因为硬件错误或任务质量失败。

## 按瓶颈选方法：先读这一张表

下表是迁移建议，不是全部方法均已在 TurboCider 验收。每个入口继续给出实现、实验与失败条件。

| 遇到的瓶颈 | 方法与作用机制 | 学习 / 参考入口 | 主要代价与否决条件 |
| --- | --- | --- | --- |
| 算了大量未消费的输出 | 将输出需求向上游传播，裁掉确实无后续依赖的行 | [需求与融合方法](fusion-and-layout.md) · [末层 FFN/head 案例](case-studies/llama-pretrained-output-demand.md) | 不能裁仍用于后续 K/V 的 hidden；实际需求与数值路径必须保持可解释 |
| 小 M、宽词表或单路 decode 慢 | GEMV、多向量共用 W；并行不足时再试 split-K | [GEMM/GEMV](gemm-and-quantization.md) · [wide 源码与控制](case-studies/mlx-small-m-wide-gemv-and-head-attribution.md) | 寄存器、partial buffer、reduction、tail 与额外 launch |
| 低比特权重占用小但执行慢 | packed 直接消费 / tile 内 decode，对照常驻展开 | [五种量化数据流](gemm-and-quantization.md) · [MLC 量化 FFN](case-studies/mlc-quantized-ffn-and-lowbatch-boundaries.md) | repack、metadata、额外 dense W；量化格式和源权重不可偷换 |
| attention score 内存大 | tiled online softmax，多 Q 复用 KV，避免提前 repeat GQA | [Attention 实现](attention-implementations.md) · [实际 GQA/精度边界](case-studies/mlx-sdpa-gqa-fallback-and-score-precision.md) | mask 对齐、全空行、typed QK/softmax、长 KV 与输出布局 |
| norm/RoPE/activation 间多次读写 | producer-consumer 融合，保留可消费的物理 layout | [融合与布局](fusion-and-layout.md) · [packed view/copy](case-studies/candle-packed-views-and-copy.md) | rounding、epsilon、LoRA 顺序、stride/offset；融合过大也可能退化 |
| CPU 编码或 GPU 间隙明显 | 稳定 block 编译、批量提交、受控 replay | [MLX cache/状态](mlx-compile-cache-and-state.md) · [command graph](command-graphs.md) | 首编译、shape cache、旧闭包常量、owner、取消和物理完成 |
| 自动调优后仅热态快 | 将搜索结果缓存，按使用次数核算回本 | [BEAM 搜索实测](case-studies/tinygrad-beam-search-and-amortization.md) | 搜索/编译耗时、失效 key；调优不能缓存旧 dispatch grid |
| 长 context/KV 占用高 | 分页、压缩读取，比较直接消费与 materialize+SDPA | [paged runtime](vllm-metal-paged-runtime.md) · [压缩交叉点](case-studies/compressed-attention-crossover.md) | codec 精度、临时 dense KV、split 基线、fresh handle 与完整 attention |
| 多请求或 prefill 峰值高 | chunked prefill、ragged KV、continuous batching | [分块策略](case-studies/mlx-prefill-chunking-and-numerical-routes.md) · [ragged KV](case-studies/mlx-lm-ragged-cache-and-padding.md) | 物理 padding、调度公平性、lookahead；块越小并非越好 |
| VAE/视觉卷积慢、resize 后异常 | 按 shape 选择 direct/Winograd 等路线，分开准备与编码 | [MNN 卷积/resize](case-studies/mnn-engine-convolution-and-resize.md) · [调优 cache 覆盖](case-studies/mnn-persisted-tuning-and-grid-coverage.md) | Winograd 小 shape 成本、grid 覆盖与 stale cache；少算导致快不是优化 |
| 想采用新矩阵指令/API | 将 TensorOps 运算层与 Metal 4 提交层分开实验 | [TensorOps / NAX](tensorops-and-m5.md) · [Tensor ABI](tensor-abi-and-feature-tables.md) | 实际 SDK/运行时/设备 guard、packing、精度和 fallback；无 M5 实测 |
| 想进一步减少模型计算量 | 稀疏 attention、缓存、token 减少或少步数另开质量候选 | [算法优化总览](../algorithms/README.md) · [媒体引擎](diffusion-and-media.md) | 数学/分布可能改变，需真实图像/视频质量与完整请求验证 |

## 从技巧到落地的采用顺序

1. **固定完整 GPU 基线。** 明确 checkpoint/LoRA、dtype、分辨率/序列、steps、KV 和输出需求；分开 denoiser、attention、FFN、VAE 与准备/交接。用[测量流程](profiling-playbook.md)判断关键路径，不先按项目名字选“最快后端”。
2. **先试低集成成本的数据流优化。** 稳定 block 的编译、去掉不必要同步/复制、复用合法布局、成熟 SDPA 与 epilogue 融合；每次只改一个机制，重新检查完整输出和请求时间。详见[基础 kernel 调优](kernel-tuning.md)与[attention 合同](attention.md)。
3. **对确认的热区做 shape 专用候选。** GEMV/GEMM、split-K、量化消费、VAE 卷积分别选型，保留成熟 MLX/GGML/MFA 或公共 MPS 对照。不要把 LLM 单 query 的结论直接迁移到 DiT 大 M。
4. **再评估高维护成本方案。** 独立 native graph、压缩 KV、新硬件路径、稀疏/缓存与 GPU/ANE 并行，都应先证明完整消费者受益，再扩大到请求。量化/近似候选额外检查质量，资源方案额外检查峰值、取消与 drain。

建议使用同一条晋级规则：组件正确 → 完整消费者正确 → 未采集性能/内存受益 → 完整请求与生命周期验证。历史组件收益不自动变成产品默认；本次整理没有改变生产后端或启动新 GPU 实验。

## 推荐学习顺序与源码入口

- **第一遍：理解数据流。** [引擎地图](inference-engines.md) → [GEMM/量化](gemm-and-quantization.md) → [Attention](attention-implementations.md) → [融合/布局](fusion-and-layout.md)。先回答输入在哪里、重用什么、消除了哪次读写。
- **第二遍：理解 host 如何选核与执行。** [MLX guard / cache / 状态](mlx-compile-cache-and-state.md)、[MLC/TVM](mlc-tvm-metal-compiler.md)、[tinygrad JIT](tinygrad-metal-jit.md)、[command graph](command-graphs.md)。同时读 selector、shader、runtime 和 owner，不只抄 kernel。
- **第三遍：用真实案例校验理解。** [MPS 三臂对照](case-studies/mps-vs-mlx-20261016.md)、[量化 FFN](case-studies/mlc-quantized-ffn-and-lowbatch-boundaries.md)、[KV writer→consumer](case-studies/native-kv-writer-and-consumer.md)、[源码/PR 账本](optimization-prs.md)。先写可被否决的假设，再看数据。
- **动手理解：** 阅读[小型教学代码](../examples/README.md)的数据流与状态机，按自己的任务设计验证；本库不提供设备实验入口。

可直接追源码的固定读本：

| 阅读目标 | 链接 | 建议关注 |
| --- | --- | --- |
| MLX 分 shape 选矩阵核 | [matmul.cpp @64ea011](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/matmul.cpp) · [gemv.h](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/kernels/gemv.h) | 外层 guard、wide 配置、split-K 与实际布局 |
| MLX 量化消费 | [quantized.cpp @36969c0](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/quantized.cpp) | QMV/QMM/NAX、metadata、contiguous 准备；不是当前 wheel 身份声明 |
| MLX attention 路由 | [scaled_dot_product_attention.cpp @64ea011](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/scaled_dot_product_attention.cpp) | 完整 GQA/shape/dtype 条件、fallback 与精度 |
| GGML 需求传播 | [Qwen2 graph @64e9bce](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/src/models/qwen2.cpp) | output IDs 在哪里裁行、哪些计算仍必须保留 |
| MFA attention 设计 | [固定版本项目 @8671cdd](https://github.com/philipturner/metal-flash-attention/tree/8671cddc38f19a6eadb804dee6a3ca2954b8bf32) | online softmax、tile/fragment 与生成器；非完整模型 runtime |
| 框架编译与 GPU 诊断 | [MLX compile 官方文档](https://ml-explore.github.io/mlx/build/html/usage/compile.html) · [Metal capture 指南](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html) | 活动文档须与已安装版本核对，区分 graph 编译与命令重放 |

固定 commit、关键文件与阅读重点从[主题来源索引](sources.md)与[来源解读](../sources/README.md)按需展开。固定读本不宣称是 2026-10-17 的最新发布；PR 状态按各账本的核查日期阅读。

## 完整方法地图

- 矩阵与布局：[kernel 调优](kernel-tuning.md)、[GEMM/GEMV 与量化](gemm-and-quantization.md)、[融合与布局](fusion-and-layout.md)。
- Attention 与缓存：[实现对照](attention-implementations.md)、[mask/数值合同](attention.md)、[paged serving](vllm-metal-paged-runtime.md)、[cache/量化 runtime](cache-and-quantized-runtime-designs.md)。
- 编译与提交：[MLX cache/状态](mlx-compile-cache-and-state.md)、[MLC/TVM](mlc-tvm-metal-compiler.md)、[tinygrad JIT](tinygrad-metal-jit.md)、[command graph](command-graphs.md)。
- 视觉与硬件：[扩散/媒体阶段](diffusion-and-media.md)、[MNN 卷积/调优](mnn-metal-deepdive.md)、[TensorOps/NAX](tensorops-and-m5.md)、[Tensor ABI](tensor-abi-and-feature-tables.md)、[微基准读法](empirical-hardware.md)。
- 选型、观测与上游实现：[引擎地图](inference-engines.md)、[profiling](profiling-playbook.md)、[优化 PR](optimization-prs.md)、[编译 PR](compiler-optimization-prs.md)、[服务 PR](serving-optimization-prs.md)。

## 内容与维护

方法与选型在本层，[实现案例](case-studies/README.md)保留有范围的历史观察和反例，[源码阅读索引](sources.md)给出固定版本入口，[学习练习](examples/README.md)指向小型参考代码。这里没有实验数组、采集驱动、hash 账本或归档验证包。

新增方法写“场景 → 原理 → 实现思路 → 成本与失效条件 → 迁移”；具体规则见[维护说明](../MAINTENANCE.md)。源码已读、作者报告、历史结果与产品采用分别判断，笔记不代表当前验证。
