# 01 · 现状、证据与外部参考审计

[目录](README.md) · 2026-09-30

## 1. TurboCider 源码事实

| 位置 | 已有能力 / 实際边界 |
| --- | --- |
| `native/core/gguf.hpp` | GGUF v2/v3 目录预检，仅接受 F32/F16/BF16/Q4_0/Q4_1/Q8_0；不是完整安全 parser 或 K/IQ decoder |
| `native/backends/mlx.cpp: Weights::load_gguf_file` | 调用 MLX loader 后接收 tensor，未建立统一的 Z 执行精度合同 |
| 同文件 `pack_convrot_q8` | signed I8 和每输出行 scale 映射成 MLX unsigned affine Q8；Z 默认 scales 为 BF16，源 checkpoint scales 为 FP32 |
| 同文件 `convrot_rotate_*` | Comfy H4 Kronecker 得到 H256；有 dense 和 Metal butterfly，两者均为浮点旋转 |
| 同文件 `project_many` | 多个 ConvRot 投影显式共享一次旋转；存在 runtime LoRA 时回到独立 project |
| 同文件 `Weights::dequantize` | ConvRot 可 FP32 解码+逆旋转后存 BF16；普通 affine 分支目前输出 FP16，不能无审查复用为 Z BF16 执行合同 |
| `native/models/z_image/z_image.cpp` | dense/BF16 图与量化兼容图不一致；GGUF/ConvRot 未获得全部 compile/fusion；runtime-weight 路由排除 ConvRot |
| `native/components/text/qwen3.cpp` | 有 Qwen3 conditioning 与 prefill 策略，checkpoint 解析针对 safetensors；不是通用 GGUF encoder loader |
| `native/models/qwen21/text_encoder.cpp` | 有模型专属 mRoPE、mask、deepstack 与 hidden 输出约定；GGUF 名字映射不能替代这些语义 |
| `native/runtime/streaming/mlx_weight_pager.hpp` | BF16 safetensors→共享 backing；明确限制无转换/派生，需扩展才能解码 GGUF |
| `native/backends/ane_runtime.*` | 固定形状图、动态 FP16 slots、持久 worker、自检与失败恢复；并非 INT8 算术 |
| `native/backends/ane_runtime_quant.hpp` | Q4/Q8 affine→FP16 SIMD staging，不是原始 GGUF blocks 解码 |
| `tools/coreml/export_z_image.py` | 冻结 W8A8 需要校准；ConvRot native+A8 明确拒绝，必须先 derotate |
| `native/runtime/streaming/public_request_validation.cpp`、Z `probe_public_streaming` | 旧公开入口排除 ANE/compiled 等组合，Z 还拒绝 GGUF/ConvRot；新增 decoder 不会自动接通产品入口 |
| `native/components/text/qwen3.cpp: qwen3_conditioning` | 短 prompt 默认最多四层 lazy eval interval；新 1–3 槽模式必须有独立的逐层提交/reader 合同 |

本轮新增项另见 [08](08-experiments-and-delivery.md)。仓库历史的
[量化与 streaming 对照](../quantized-streaming-vpipe-comparison.md) 涉及旧 sd.cpp
路线，不能据此声称**当前 native MLX**已支持全部 K/IQ。历史
[M5 记录](../m5-ane-adaptation.md) 是其他构建/设备上的证据，本轮没有 M5 实机。

## 2. Q8 慢因：事实与假设分开

上一轮核对的 MLX 0.32.0 loader 将 Q8_0 表示为 g32 affine codes + FP16
scales/biases；Z 激活多处转为 BF16。该版本小探针已观察到：

```text
BF16 activation + FP16 Q8 scales -> FP32 quantized_matmul output
BF16 activation + BF16 Q8 scales -> BF16 output
BF16 activation + FP16 dense weights -> FP32 matmul output
```

不能因此断言整个网络所有操作都是 FP32。FP32 norm 的存储也不等于激活被提升：
`rms` 自身会转回输入 dtype。应记录每个主要 tensor 的实际执行 dtype。

上一轮合成矩阵证据：M4 Max，MLX 0.32.0，K=3840/N=10240，3 warmups、
8 个交错样本/模式，单位 ms。不是本轮重新测量、不是整图成绩：

| 路径 | M=1056 | M=4128 |
| --- | ---: | ---: |
| dense BF16 | 5.9372 | 21.9137 |
| Q8，BF16 X/FP16 scales→FP32 | 7.2462 | 27.7440 |
| Q8，BF16 X/BF16 scales | 6.5882 | 25.0833 |
| Q8，FP16 X/FP16 scales | 6.1233 | 23.2982 |
| 按次展开 BF16 + GEMM，含展开 | 6.5340 | 22.4915 |
| 已展开的 Q8 BF16 GEMM | 5.9380 | 21.9049 |

支持的推论：大 M 值得测试按层展开；不支持的推论：每个 GGUF 格式都只慢
2.6%、全模型预展开低内存、FP16 可无条件替代 BF16。此表作为研究线索，不作为
本轮发布 gate 的可审计原始回执；正式验收须保存原始样本和二进制身份。

当前整请求历史证据见
[runtime 当前状态](../../status/runtime-ane-current-2026-09-30.md)：Q8 runtime
试验是 resident FP16 路线，进程树峰值较大，不能认证本设计的受限内存模式。

## 3. Unsloth：参考的是格式与优化思路，不是 CUDA 成绩

本地只读参考：`../references/unsloth`，提交
`b66d2a4c8912828337407218d6afd655bf0b93ca`。

### 3.1 已读文件和可借鉴内容

- `unsloth/save.py` 的 `ALLOWED_QUANTS`、`IMATRIX_QUANTS`：Q4_0/Q4_1、
  Q5_0/Q5_1、Q4_K_S/M、Q5_K_S/M、Q6_K、Q8_0 及 IQ 选择。
  `_M/_S` 是导出配方，文件内部仍逐 tensor 指定 GGML type。
- `studio/backend/core/inference/diffusion_gguf_compile.py`：其 PyTorch
  GGUF eager 解码有许多小算子，单独编译解码链可减少 launch；整块 compile
  路线又不应该套独立解码 graph break。启示是**分别测解码融合与整块融合**。
- `diffusion_transformer_quant.py`：storage quantization 与 dynamic A8
  compute quantization 分开；小 M 调制层/文本流需要 shape 门禁，量化前
  先完整 dense load 会抬高峰值。不能照搬 CUDA 的最小 M 或性能阈值。
- `diffusion_te_prequant.py`：encoder 独立于 DiT；embedding/norm/特定
  模块不一定压缩；必须按加载全过程峰值预算。该文件 v1 是 FP8 storage，
  **不是**通用 GGUF encoder 的现成实现。
- `studio/backend/utils/models/gguf_metadata.py` 及相关 tests：可作为
  split/metadata/组件身份测试用例的需求来源，文件内容仍以 GGML 规范为准。

不能照抄 `save.py` 的说明文字作为 block 定义。例如此快照 `q6_k` 描述写成
“Uses Q8_K”；实际 Q6_K 解码必须以 `ggml-common.h` / `ggml-quants.c` 为准。
用户提到“Q4–Q8”不是要求创造不存在的统一 Q7 tensor 格式。

### 3.2 代码来源边界

TurboCider 根目录为 MIT；Unsloth 根许可为 Apache-2.0，但上述 Studio 文件
含 `AGPL-3.0-only` 标头。**本轮只读分析，不复制其实现到本项目。** 后续
decoder 优先使用独立 GGML 规范/参考实现，固定来源版本并保留相应许可及声明；
引入第三方代码前另做许可审查，不将本段视为法律意见。

## 4. vpipe：动态 ANE、矩阵切块与设备工作队列

本地只读参考：`../references/vpipe`，提交
`c2989a76841888e4d8d208fad4f476c487710962`。

已读 `generative-models/shared/ane-module.{h,cc}`、`ane-ffn.{h,cc}`、
`ane-tier.{h,cc}`、`i8-gemm.cc`、`apple-silicon/coreml/ane-emitter.{h,cc}`、
`ane-worker.h` 及 `gpu-kernels/metal/gemm/dense_gemm_mma.metal` 的相关路径。

| 参考思路 | TurboCider 决策 |
| --- | --- |
| 固定 shape 的 runtime-weight 模块、IOSurface 原地填权重 | 复用本项目现有 public Core ML runtime；扩展量化描述而非另做一套模型后端 |
| `[out,in]` 输入减少大矩阵转置 | 保留；比较 staging+predict 总时间，不只比 predict |
| K/N 切块放在一张图内 | 采用为待筛选设计；host 不逐 tile 发 prediction |
| 一个持久 ANE dispatch worker，独立转换池 | 已有类似结构；未来 encoder/DiT 阶段共用有界服务，禁止两个 pool 相互等待 |
| token-row 分配、chunk 数动态平衡 | 复用现有 scheduler；按完整 block、含量化/解码成本重调 |
| 大 M 输入分 chunk，host scratch 只保留一个 chunk | 采用；避免按完整视频序列准备巨大 staging |
| weight stamping / 直接构造 `.mlmodelc` | 不作为产品前提；依赖未公开格式，单独实验也须 OS pin、自检、fallback 和安全目录管理 |
| compiled bundle 中 `main_ane` 等名称判断设备 | 仅调试线索，不替代运行期 placement/INT8 算术证据 |
| M5 grouped I8 GEMM、split-K、scale 分段应用 | 算法参考；不能把其每次重新量化浮点权重的流程用于已有 ConvRot codes |

参考中的约 8–10 MB buffer、runtime 比静态慢约 37%、特定 tile 的 TOPS
均是该项目注释中绑定 shape/设备的测量，不是 Apple 公开硬件规格，也不是本机
验收数据。尤其其“GPU↔ANE crossing 免费”的局部结论不推广到我们完整系统。

## 5. 官方技术来源与版本边界

核对日期 2026-09-30；本机 MLX 0.32.0、coremltools 8.3.0 的实际接口优先于
对“最新版本”的想象。官方资料用于硬件/算子方向，不充当 TurboCider 跑分：

```text
https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html
https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html
https://developer.apple.com/metal/Metal-Performance-Primitives-Programming-Guide.pdf
https://developer.apple.com/documentation/metalperformanceshaders/mpsgemmdatatype/int8
https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html
```

Apple 将 M4 级 ANE 的 INT8×INT8、per-channel 权重量化列为优化方向；MPP
文档区分矩阵操作 API 与 GPU 代际实现。支持 API、编译成功、放置到目标设备、
真实 INT8 算术、整请求加速是五个不同证据层级，后续文档保持该区分。

## 6. 本次审阅新增的本地证据

只读解析了本地 Z Q8/Q4 的 GGUF 目录，未读取权重 payload 或重新跑模型：Q8 样本
包含 F32/BF16/Q8_0，架构标签为 lumina2；Q4 样本为 BF16/Q4_0 且没有 metadata。
精确数量、大小、shape 差异和未完成的 content binding 见 [09](09-release-scope-and-component-contracts.md)。
不能将它们当作仅 bits 不同的严格性能对照，也不能按架构字符串单独决定 adapter。

Qwen3 adapter 命名进一步核对 02 固定 GGML 快照的 `conversion/qwen.py` 与
`gguf-py/gguf/tensor_mapping.py`，不是从文件名猜 Q/K permutation。产品接入和
验收不足分别收敛到 [10](10-execution-and-product-integration.md)、[11](11-acceptance-profiles-and-feasibility.md)。
