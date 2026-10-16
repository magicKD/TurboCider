# Apple Silicon 推理引擎地图：比较执行层，不比较产品名字

[专题导航](README.md) · [GPU 机制](gpu-gemm-quantization.md) · [来源范围](sources-and-coverage.md)

S=本轮读了指定源码片段，D=项目/官方文档核查，L=本库已有固定来源导读；不是完整仓库审计。版本是读本，不承诺所有项目同一时刻的最新发布；未执行的引擎不参加本机速度排名。

## 核心 GPU 引擎和算子库

| 项目 | Apple 路线与阅读深度 | 可迁移机制 | 主要边界 |
| --- | --- | --- | --- |
| [MLX](https://github.com/ml-explore/mlx/tree/36969c094354086e15d2a327a9c50feb20b9cf11) | 自有 Metal backend，quantized dispatch/SDPA/NAX guard，S | lazy graph、compile fusion、shape dispatch、packed QMM、临时对象寿命 | 主线源码与本机 wheel 0.32.0 分开；不自动使用 ANE |
| [llama.cpp/GGML](https://github.com/ggml-org/llama.cpp/tree/64e9bceb2c3a856efed96feda784a50947049feb) | `ggml-metal-ops.cpp` small-M/MM 路由，S | GGUF 数学与布局、GEMV/GEMM、backend graph/stride | 单模型支持不证明所有算子 GPU；量化格式不是同一种 Q4 |
| [独立 MFA](https://github.com/philipturner/metal-flash-attention/tree/8671cddc38f19a6eadb804dee6a3ca2954b8bf32) | README 算法/数据流，D | online softmax、D 分块、register pressure、JIT | 教学 single-head 核心，不当完整 serving engine |
| [ccv/MFA](https://github.com/liuliu/ccv/tree/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701) | ANE bridge 关键函数，S；GPU descriptor 既有导读，L | 应用图→kernel/ANE、cache、surface 与交接 | GPU `NA*` 不等于 ANE；与独立 MFA 版本不同 |
| [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html) | 官方后端/环境变量文档，D | PyTorch 生态、MPSGraph/Metal kernels、allocator、profiler | MPS 是 GPU；fallback、contiguous 和 host reads 可吞收益；未跑本机对照 |
| [MNN](https://github.com/alibaba/MNN/tree/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/metal) | `MetalMatMul.mm` 全文件，S | resize 阶段准备 shape/stride/线程组，encode 阶段绑定，融合 bias | 一个普通 matmul 实现不是整个 backend 上限；不是为借核而引入整引擎 |
| [Candle](https://github.com/huggingface/candle/blob/12bb5d1f6350f4ea9827d9086f213feec1f0af6a/candle-core/src/metal_backend/mod.rs) | `matmul` 1729–1780，S | Rust owner、start_offset/stride、dtype、`call_mlx_gemm` | 借用 MLX-style GEMM 不等于完整 MLX graph/runtime |
| [tinygrad](https://github.com/tinygrad/tinygrad) | 既有 search/runtime 导读，L | 有限 tile/轴/矩阵原语搜索，编译 cache | 搜索启发式计时不能替代真实完整 workload；接口稳定性另查 |
| [MLC-LLM](https://llm.mlc.ai/docs/deploy/ide_integration.html) | 官方 Metal 部署文档，D | TVM 编译目标、模型库/权重分开、目标相关调优 | 编译产物绑定平台/模型；CUDA FlashInfer 不直接成为 Metal 能力 |

### 从 MNN 与 Candle 学到的不同层次

MNN `onResize` 把 transpose 选项转换成显式 stride、缓存常量与线程组；`onEncode` 复用 pipeline 并绑定可选 bias。它说明**把 shape 工作移出热路径**，不是只优化数学 kernel。

Candle 为输出创建 buffer，检查 F32/F16/BF16，再把两个 operand 的 stride/byte offset 传给 GEMM。它说明共享 kernel 思路仍需要本框架的 storage/owner 合同；不能借到 shader 就假定 view 全部零成本。

## 模型库、serving 层与新项目

| 项目 | 实际优化层 | 不应误标 |
| --- | --- | --- |
| [MLX-LM](https://github.com/ml-explore/mlx-lm) / [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) | MLX 之上的模型、prefill/decode、KV/prompt cache、生成管理（入口级 D） | 不是另一套硬件指令实现，也不是 ANE backend |
| [vLLM-Metal](https://github.com/vllm-project/vllm-metal/tree/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e) | 固定 README，D：vLLM scheduler/block manager、mlx_lm layers、项目自有 request-aware paged/varlen attention/NAX prefill | serving 吞吐改善不等于单请求延迟同比改善；NAX 不是 ANE |
| [Uzu](https://github.com/trymirai/uzu/tree/ee5ec5708b2b71991ae9c5221c5107f5aa87a8f6) | 固定 README 的 API/UMA/模型 session 与 benchmark 入口，D | 未读 shader，不复述未核实的 kernel 细节；文案“zero latency”不当物理结论 |
| [mflux](https://github.com/mflux-community/mflux) | MLX 模型层、编译 denoise、量化、模型图（L） | 不统一标成现成 GPU+ANE 并行引擎 |
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | GGML 模型/阶段 offload、权重和 VAE 驻留（L） | 模型可加载不等于 VAE/全部 op 使用最快 GPU 核 |
| [SolAttn-MPS](https://github.com/yshenaw/ComfyUI-SolAttn-MPS) | 稀疏 attention、模型 patch、Metal 核（L） | 近似算法与 exact attention 实现提速分开 |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp/tree/0c7be696371069e0baa2515ed7aa08d5a5473aa6) | Core ML encoder + 其余 runtime，S/D | 按 stage 分工不是单请求同时执行；CPU 分母倍率不等于 Metal 分母倍率 |

vLLM 团队 [2026-09-22 的架构文章](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0)说明共享 KV pool、paged/varlen kernel、prefill/decode dispatch 和 Torch/MLX lowering 的目的。学习价值是请求级数据结构如何避免 host roundtrip 与 cache materialization；其中不同版本/精度模型的上游 TTFT/throughput 数字本页不用于横向排名。

## 其他平台路线

[ncnn](https://github.com/Tencent/ncnn)重点是 CPU/Vulkan；macOS GPU 若经 MoltenVK，不是原生 Metal backend 案例。ONNX Runtime、TFLite/LiteRT、ExecuTorch 的 Core ML delegate 是子图导出/委托路线：需读分区、unsupported-op fallback、张量交换，而不是仅看设备支持表。它们在本轮为分类/后续入口，未做源码性能审计。

Core ML/Core AI、ANEMLL、ANEForge、ggml-ane 的 ANE 路线单列在 [项目部署](hybrid-projects-and-stages.md) 和 [私有图](ane-private-and-search.md)，不能与 GPU backend 混成一个表的“性能名次”。

## 怎样做有意义的引擎比较

固定 checkpoint、量化数学、tokenization、prompt/output 长度、batch、KV 精度、上下文、图像尺寸/steps、LoRA、resident budget 和冷/热边界。报告 compile/load/TTFT、decode、完整 request、峰值内存与质量；先比较共同支持的一个 stage，再扩到全模型。

本轮只对已安装 MLX/Core ML 做机制实验，没有安装/运行全部引擎。项目成熟度、移植成本、公开/私有 API、模型覆盖与性能应作为不同决策维度，不宣称“Apple Silicon 全局最快”。
