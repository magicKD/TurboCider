# Apple Silicon 推理引擎地图：按执行层而非产品名字比较

[GPU 导航](README.md) · [源码与哈希](research-sources.json) · [GEMM/量化](gemm-and-quantization.md)

核查：2026-10-16。下列版本是本轮读本，不承诺是所有仓库同一时刻的最新发布；未运行的框架不参加本机速度排名。

## 引擎、库与包装层不能混为一类

| 项目/入口 | Apple GPU 路线与已读范围 | 值得学习 | 成本/边界 |
| --- | --- | --- | --- |
| MLX `36969c0`；本机 wheel 0.32.0 | 自有 Metal backend；Steel GEMM/SDPA、quantized dispatch、NAX guard（S/R） | lazy graph、分 shape dispatch、量化 GEMV/GEMM、layout 和融合 | 主线源码不能证明旧 wheel 含新核；CPU/GPU 不等于 ANE |
| MLX-LM `93e6d4a` | `mlx_lm/generate.py`（S）在 MLX 之上管理生成 | chunked prefill、prompt cache、KV quantization、生成边界 | cache 复用需要 prompt/source 身份；不是另一套 GPU 指令实现 |
| llama.cpp / GGML `64e9bce` | `ggml/src/ggml-metal` 的 host、kernel、fusion 分发（S） | GGUF packed 权重、small-M 与大 M、backend graph、Flash Attention、tensor library fallback | 支持某模型架构和某 GGUF dtype 不等于每个算子都在 GPU |
| MFA 独立库 `8671cdd` | Swift 源码生成 attention/GEMM（S） | tile、寄存器驻留、online softmax、JIT pipeline | 读本 README 为 single-head 教学核心；不直接当完整多模型 runtime |
| ccv / MFA `66dc16a` | `lib/nnc/mfa` descriptor→生成 shader→pipeline（S） | application-oriented GEMM/attention、NA 核、精度、cache key、scratch | 不把同目录 ANE bridge 与 GPU NA 核混淆；与独立 MFA 版本不同 |
| PyTorch `eb429a0` 的 MPS | `Attention.mm`（S）同时有 MPSGraph 与自定义 Metal SDPA | 保留 PyTorch 模型生态，定位 contiguous、graph cache、fallback | installed torch 缺席，本轮未跑；ATen/backend 名称不保证 fused kernel |
| tinygrad `3d057f0` | `runtime/ops_metal.py`（S）编译/提交，ICB、residency、fence | 可搜索的算子变换、生成 kernel 与执行复用 | 该读本直接调用未文档化 MTLCompiler 服务；不能归类为纯稳定公共 API |
| MNN `dbf9ba6` | `MetalMatMul.mm` 与 shader（S） | onResize 准备 shape/常量、onEncode 绑定 buffer、融合 bias、线程组选择 | 所读普通 matmul 是逐输出 K-loop；这不代表整个 Metal backend 的性能上限 |
| Candle `12bb5d1` | Rust `metal_backend/mod.rs::matmul`（S） | Rust ABI/owner、strided storage、dtype 检查、调用 `call_mlx_gemm` | 复用 MLX 风格 GEMM 不等于整个 MLX lazy runtime；未做 Rust 对照实测 |
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

数学、source 与消费者 ABI 相同才比较实现；改变量化、稀疏或 cache 策略另开质量类别。本机合成比较仅验证机制，见 [实验](experiments-20261016.md)；既有 [真实 Qwen BF16 FFN](../review/gpu-affine-20261016.md)提供更强消费者边界。当前证据不足以选出“Apple Silicon 全局最快引擎”。

## 可执行学习控制（不替代上游验证）

本页实现参考：[labs/model_stages.py](../labs/model_stages.py)、[labs/metal/engine_probe.py](../labs/metal/engine_probe.py)；case：`stage-llm`、`stage-decode`、`mlx:engine-mechanisms`。CPU/旧 Metal 控制只验证对应数学与合同；`mlx:engine-mechanisms` 对应[显式组件 CLI](experiments-20261016.md)，不是自动启动 GPU 的普通单元测试，也不证明全部上游引擎或 PR 已运行。设备、source、shape、精度或消费者改变后重新验证。
