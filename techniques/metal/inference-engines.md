# Apple Silicon 推理引擎地图：按执行层而非产品名字比较

<!-- knowledge
id: technique-metal-inference-engines
type: method
stage: pipeline, prefill, decode
operator: gemm, attention, scheduling
backend: mlx, metal, mpsgraph
device: cpu, gpu
symptom: measurement, precision-loss, launch-bound
semantics: contract-only
boundary: producer-consumer
api: public, reference-only
-->

[GPU 导航](README.md) · [源码与哈希](sources.md#research-sources) · [GEMM/量化](gemm-and-quantization.md)

核查：2026-10-16。下列版本是本轮读本，不承诺是所有仓库同一时刻的最新发布；未运行的框架不参加本机速度排名。

## 如何把引擎调研变成可借鉴的方法

2026-10-17 整理：下表保留各轮读本身份，不把早期 MLX 0.32.0 与后续 [0.32.3 模型/dispatch 实验](case-studies/mlx-dispatch-pso-and-function-association.md)混为同一构建。选引擎时先定义部署约束；学优化时按下面的链路读，而非比较品牌。

| 想学习什么 | 优先读什么 | 迁移时先做的检查 |
| --- | --- | --- |
| 消除无用计算 | GGML [输出需求传播](case-studies/llama-pretrained-output-demand.md) | 下游到底需要哪些 rows；提前裁剪是否影响后续 K/V |
| shape 专用矩阵核 | MLX [matmul selector](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/matmul.cpp) → [GEMV 实现](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/kernels/gemv.h) | guard、alignment、tail、fallback，而非只复制 tile |
| 融合和低开销执行 | [MLX compile/cache](mlx-compile-cache-and-state.md)、[tinygrad JIT](tinygrad-metal-jit.md) | 区分图优化、编译缓存和命令重放；测 cold/warm 与失效 |
| 原生/AOT 集成 | [MFA](attention-implementations.md)、[TVM AOT](case-studies/tvm-metal-aot-and-runtime-only.md)、[Relax VM](case-studies/mlc-runtime-only-relax-vm.md) | ABI、权重/模型库身份、PSO 首用、完整消费者与生命周期 |
| KV 与服务调度 | [vLLM Metal](vllm-metal-paged-runtime.md)、[MLX-LM batching](case-studies/mlx-lm-continuous-batching-and-cache-handoff.md) | page/stride、有效长度、lookahead、cache 交接和真正完成 |

一次迁移只提取一个机制：固定同一权重/输入/输出需求，保留原后端作对照，先验完整输出，再量准备、热态、内存与请求。读本的价值是可追踪的设计，不是要求引入整个框架；具体路线选择见 [GPU 总结](README.md)。

## 引擎、库与包装层不能混为一类

| 项目/入口 | Apple GPU 路线与已读范围 | 值得学习 | 成本/边界 |
| --- | --- | --- | --- |
| MLX `36969c0`；本机 wheel 0.32.0 | 自有 Metal backend；Steel GEMM/SDPA、quantized dispatch、NAX guard（S/R） | lazy graph、分 shape dispatch、量化 GEMV/GEMM、layout 和融合 | 主线源码不能证明旧 wheel 含新核；CPU/GPU 不等于 ANE |
| MLX-LM `93e6d4a` | `mlx_lm/generate.py`（S）在 MLX 之上管理生成 | chunked prefill、prompt cache、KV quantization、生成边界 | cache 复用需要 prompt/source 身份；不是另一套 GPU 指令实现 |
| llama.cpp / GGML `64e9bce` | Metal host/kernel/fusion与[完整预训练Qwen引擎](case-studies/llama-pretrained-output-demand.md)（S/R） | GGUF、last-output需求传播、KV/Flash、[metadata observer](case-studies/llama-metal-flash-and-eval-observer.md) | 616调用/3676完整词表rows保留3613严格预算拒绝；IR/同词不授予所有op物理GPU、任务质量或跨引擎排名 |
| MFA 独立库 `8671cdd` | Swift 源码生成 attention/GEMM（S） | tile、寄存器驻留、online softmax、JIT pipeline | 读本 README 为 single-head 教学核心；不直接当完整多模型 runtime |
| ccv / MFA `66dc16a` | `lib/nnc/mfa` descriptor→生成 shader→pipeline（S） | application-oriented GEMM/attention、NA 核、精度、cache key、scratch | 不把同目录 ANE bridge 与 GPU NA 核混淆；与独立 MFA 版本不同 |
| PyTorch2.14.1 / `5c48869` 的 MPS | `Attention.mm`、MPP、graph/PSO profiler（S/R） | 原框架SDPA分流、GQA/view、typed graph cache、profiling覆盖 | [第十五轮实测](case-studies/pytorch-mps-sdpa-dispatch-and-layout.md)：4642正式全输出含240真实Qwen激活；两次mask-cache native abort。组件/CB日志不授予[完整模型或物理kernel资格](case-studies/pytorch-mps-profiler-and-real-attention.md) |
| tinygrad `3d057f0` | `runtime/ops_metal.py`（S）编译/提交，ICB、residency、fence | 可搜索的算子变换、生成 kernel 与执行复用 | 该读本直接调用未文档化 MTLCompiler 服务；不能归类为纯稳定公共 API |
| MNN `dbf9ba6` | `MetalMatMul.mm` 与 shader（S） | onResize 准备 shape/常量、onEncode 绑定 buffer、融合 bias、线程组选择 | 所读普通 matmul 是逐输出 K-loop；这不代表整个 Metal backend 的性能上限 |
| Candle `12bb5d1` | Rust Metal matmul/quantized与原始Qwen2（S/R） | [view/copy](case-studies/candle-packed-views-and-copy.md)、[完整预训练Qwen/tokenizer](case-studies/candle-pretrained-qwen-and-tokenizer.md) | [第十六轮](case-studies/candle-q8-fragments-and-request-cost.md)448完整logits含152正常精度预算拒绝；同词/half次级参考不替代原目标。旧合成/queue/capture范围另留历史记录 |
| MLC-LLM `8978ea9` | 编译与 Metal 部署文档（D），KV dispatch pass（S） | TVM 目标编译、模型库/权重分开、TIR KV cache 路线 | CUDA FlashInfer 不能直接迁到 Metal；编译目标/设备形状影响产物复用 |
| whisper.cpp `0c7be69` | README Metal 与 Core ML 章节（D） | 音频 encoder/decoder 的阶段分配、GPU/ANE 路线分别管理 | Core ML encoder 的 CPU 对照收益不是对优化 Metal 的收益；本轮未跑音频 |
| mflux `8c00dab` | Qwen21 transformer/attention（S） | 编译 denoise step、分段 attention、共享几何缓存 | 模型层优化不等于新 GPU 后端，须追到 MLX dispatch |
| stable-diffusion.cpp `2f88688` | backend-fit 与项目后端组织（S/D） | GGML 执行和模型阶段驻留、设备/权重 fit | GPU 算子覆盖、VAE、offload 与量化格式分别检查 |
| SolAttn-MPS `4507112` | tiled Metal 源和 routing（S） | sparse summaries、exact/approx tile、PyTorch 集成 | 近似 attention；Pytorch 内部头文件/版本是兼容边界 |

表中短 hash 的完整值、文件、SHA 与阅读范围在来源清单。官方 [MLX 项目](https://github.com/ml-explore/mlx)、[GGML Metal 读本](https://github.com/ggml-org/llama.cpp/tree/64e9bceb2c3a856efed96feda784a50947049feb/ggml/src/ggml-metal)、[MNN 读本](https://github.com/alibaba/MNN/tree/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/metal)、[Candle 读本](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/candle-core/src/metal_backend/mod.rs)可沿源文件继续读。

### 两类容易误标的项目

**ncnn 不是原生 Metal backend 案例。** 固定 `c15904b` 的 [README](https://github.com/Tencent/ncnn/blob/c15904bb8918187ec100f1ebe708942af83b84be/README.md)明确是 CPU/Vulkan；macOS GPU 使用 Vulkan/MoltenVK 链路时，还需算翻译层、shader、packing 与平台覆盖。它适合研究移动视觉/超分，不能因支持 Apple 平台就写成“Metal 引擎”。

**Core ML delegate/EP 不是独立 Metal kernel 库。** [ExecuTorch Core ML backend](https://docs.pytorch.org/executorch/stable/backends-coreml.html)与 [ONNX Runtime CoreML EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)是图分区、模型编译与框架执行入口；CPU/GPU/ANE 放置需计划与 trace。文档级补充，未编译测试；`CPUAndGPU`、`ALL` 或成功创建 session 都不证明某个 op 用了 GPU/NAX。

Ollama、LM Studio、MLX server、oMLX、exo 等产品/服务层还要识别实际 backend 及版本：模型管理、prefix cache、batching、并行和 API 延迟可能决定吞吐，但不凭品牌重复计为新的 GPU kernel。分布式吞吐尤其不能作为单机 DiT kernel 优势的证据。

## 四个有用的选择场景

**Python 扩散模型验证**：先选 MLX/mflux 或保留 PyTorch MPS 生态，确保权重、dtype、RoPE 和 mask 语义一致，再沿 trace 找热区。跨框架移植的工作量与质量回归成本应进入决策。

**C/C++ 本地 LLM**：GGML/llama.cpp 的 packed 权重和 backend 调度是高价值参考；分别量 prefill、单路 decode、batch decode、KV 增长与首 token。`-ngl`/offload 配置要核对实际日志，不能只看参数接受成功。

**原生应用的热 kernel**：MFA/ccv、原生 Metal/TensorOps 更直接；代价是自己负责图缓存、shape、owner、错误回收和模型覆盖。小 kernel 优势必须恢复到真实消费者 ABI。

**移动视觉 / Rust 服务 / AOT 部署**：分别考察 MNN/ncnn、Candle、MLC/ExecuTorch。它们的价值常是可集成性、二进制/运行时约束，不应被一个 LLM tok/s 排名否决。

## 统一比较合同

至少固定 SoC/GPU cores、OS/SDK、binary/metallib、模型 revision/权重哈希、量化配方、batch/序列/分辨率、KV、生成步数与输出精度。分别报告 setup/load/compile、first-call、warm latency、steady throughput、allocator/全进程内存与质量。

数学、source 与消费者 ABI 相同才比较实现；改变量化、稀疏或 cache 策略另开质量类别。本机合成比较仅验证机制，见 [实验](../MAINTENANCE.md)；既有 [真实 Qwen BF16 FFN](case-studies/affine-ffn-cost-and-resident-tradeoff.md)提供更强消费者边界。当前证据不足以选出“Apple Silicon 全局最快引擎”。
