# 用本手册加速一个新模型

<!-- knowledge
id: technique-guides-new-model
type: guide
-->

[总目录](../README.md) · [逐项清单](../templates/new-model-checklist.md)

这是一条实施路线，不是声明任意模型可自动使用 ANE。推荐先复制接口合同与测试形状，不复制 Qwen/Z 的专用开关。

## 第一步：写模型事实表

记录 `H/F/heads/head_dim`、层数、各阶段 rows、门控 FFN / GELU、并行或串行 block 拓扑、position encoding、参考 prefix 是否只读、全局 norm / attention、VAE temporal phase、输入最大帧数、实际权重 dtype/编码。

特别回答：

- attention 与 FFN 是否读取同一份输入？只有并联拓扑才可跨两支重叠。
- gate/up 的通道是否逻辑成对？packed 物理顺序与逻辑顺序是否一致？
- 模型是否含 CFG 双 batch、音视频互相依赖、shifted windows、padding / mask？
- LoRA 覆盖哪些 projection？是否必须支持不合并 adapter？
- 准备时、denoise 时和 decode 时各自的存储下界是什么？

## 第二步：接普通 GPU 与框架入口

按 [模型 registry](../../native/models/registry.cpp)、[Session](../../native/runtime/session.hpp) 和现有 module 接入。模型语义放 `native/models/<model>/`；通用执行 / provenance / memory 合同放现有 runtime/backend，不新建第二套模型库或账本。

先保留现有高质量 GPU 实现：MLX compiled block 或原生 Metal/MPS，使用 fused SDPA、模型自己的 Q/K norm/RoPE、正确的量化读取。独立 native command graph 是另一条显式后端，不是为了调用一个 kernel 就重写完整模型。

用同一 noise / conditioning 做 layer → step → latent → media parity。记录默认不支持的 operation，拒绝未实现 I2V/音频，而不是静默丢输入。

## 第三步：选一个收益最大的纵切

| 新模型特征 | 首个候选 | 最小对照 |
| --- | --- | --- |
| FLUX 式 parallel attention/MLP | FFN intermediate-channel split | 完整 block GPU vs hybrid；attention overlap 真实依赖 |
| sequential pre-norm SwiGLU | attention 后的 FFN channel split | 含 stage、corrected hidden、join 的完整 FFN |
| 很长 prefill、后续短 rows | per-phase row/channel policy | 首步 + KV-hit 各自最优 GPU 对照 |
| 同 shape 多层但内存紧 | runtime-weight 模板 + bounded banks | 计入每层 weight stage；与 frozen 和 GPU 分别比 |
| 小内存大 checkpoint | 原始 dtype streaming / packed 原生 QMM | 完整 tensor table、双槽、端到端峰值 |
| VAE 比 DiT 更重 | convolution / normalization / IO boundary | 原始 FP32 witness、complete decoder、全请求峰值 |

首个切分可以用 [CPU 演示](../examples/split_ffn_reference.py)验证代数；这不能验证设备精度或速度。

## 第四步：复用 GPU/ANE 合同

以 [HybridFfn](../../native/backends/ane_ffn.hpp) 为参考：模型提供自己的完整 GPU callback、逻辑 weight view、可选 gate/up/down adapter 回调和未来层 source。不要把模型 attention 搬进通用后端。

为每一个异步输入登记：实际 buffer、byte offset、physical row pitch、dtype/encoding、logical slice、不可变 generation、allocation owner。权重绑定 lifetime 到实际完成，不到 `submit()` 返回。

先串行正确再并行：固定 share、固定 bucket、一 current bank；再加入 GPU async branch、独立 future bank、actual ready/done events。每一步做消融，不默认所有预取都有收益。

## 第五步：建立资格矩阵

最少覆盖下面的轴，明确未测项而非笛卡尔积全宣称通过：

- GPU / Public Core ML / Private research；base / 真实 LoRA；dense / packed source。
- 最小/目标/尾部 shape、长短 prompt、一/多参考图、prefill / hit、cold / warm。
- baseline / hybrid outputs、分支失败/晚 chunk 失败、取消/重试、A→B→base。
- encode / denoise / VAE / export / full process，MLX / external surfaces / physical footprint / swap。
- 至少目标设备 profile；小内存模拟明确标模拟，不能代表实机。

固定收益与质量门槛后运行，不沿用其他模型历史的 1.02 / 1.2 / 5% 作为通用真理。工具 receipt 必须证明候选实际被执行，而不只是 plan 允许。

## 第六步：接产品并交付

给可用候选一个准确的 request/plan/actual label；回退时改为真实 GPU label。不要写“ANE加速”却实际全 GPU。公开/私有构建隔离、manifest 来源与 shape、软件/芯片 allowlist 一并接入。

交付物：便携请求、构建身份、命令、原始报告路径和小型摘要、质量样本清单、已知失败、默认选择理由、下次排障入口。按 [覆盖审计](../MAINTENANCE.md) 反查一次：陌生工程师能否不读历史聊天就重做这条纵切？
