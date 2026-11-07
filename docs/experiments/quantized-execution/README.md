# 量化执行实验归档

本目录统一保存 2026-09-30 起的 GGUF、ConvRot、有界 GPU 执行与 ANE staging
历史实施和测量记录。编号沿用原专题顺序，08 与 12–26 是实验/实施账本；
01–07、09–11 的架构、组件与验收合同仍在[设计目录](../../design/quantized-execution/README.md)。
这些记录解释当时做过什么，不表示当前代码已取得全部后端、完整请求或发布资格。

## 记录顺序

| 记录 | 内容与原有边界 |
| --- | --- |
| [08 本轮交付与实验](08-experiments-and-delivery.md) | 实际改动、命令、已跑/未跑证据，不与规划混用 |
| [12 实施进度](12-implementation-progress.md) | 已实现的安全目录/CPU Q4–Q8 decoder、真实 Z 权重验证、SIMD 原始回执与剩余工作 |
| [13 有界 GPU 实验实现](13-gguf-bounded-runtime-progress.md) | SourceLease/ledger/slots接入真实Z生成、MLX浮点加载差异、兼容profiles与通过/失败记录 |
| [14 ANE直填与W8A8筛选](14-packed-ane-staging-and-w8a8-screen.md) | GGUF/ConvRot FP16 staging API、有界转换并行、CPU SIMD实测、动态QDQ完整投影与负结果 |
| [15 Qwen3 GGUF conditioning](15-qwen3-gguf-conditioning.md) | verified config/tokenizer与gather、逐层固定槽、Q8/mixed K真实组件及Z组合、质量差异与剩余资格 |
| [16 CPU量化SIMD加速](16-cpu-quant-simd-acceleration.md) | Q4/Q5/K直填SIMD、同源encoder控制、独占投影/encoder实测与完整Z exact回归 |
| [17 packed-streamed/三槽](17-packed-streamed-and-three-slots.md) | 单有界read buffer、source/task读取复用、p2真实12格与Z/取消exact，剩余floor/envelope |
| [18 refiner流式/银行交接](18-streamed-refiners-and-bank-boundaries.md) | interleaved refiners单槽、drained银行释放重建、浮点SIMD转换、真实Q8/Q4 exact与取消；6/8/10/16 GB速度优先方向 |
| [19 预算速度筛选](19-speed-and-memory-budget-screen.md) | 七候选完整Q8请求的独立测速/内存采样、6/8/10/16 GB及GiB经验选择，native额外驻留负结果与后续优化 |
| [20 direct packed导入/阶段释放](20-direct-packed-import-and-stage-release.md) | compute-ready压缩bank直填、cache0对照、refiner eval/VAE前verified释放、Q8/Q4 exact与分预算较快候选 |
| [21 最快BF16与compiled packed](21-fastest-bf16-and-compiled-packed-screen.md) | 实际更快BF16基线、Q8/Q4原算术参数图/压缩bank复用、独立速度内存screen；BF16密集槽和ConvRot runtime FP16质量负结果 |
| [22 FP16 packed compute/逐block N1](22-fp16-packed-compute-and-block-validation.md) | explicit FP16 QMM/FP32 glue、cache分账与独立source轨迹；Q8/Q4单cell全部block N1及≤20%/lower-memory screen通过，dynamic refiner中间层负结果 |
| [23 raw-GPU affine streaming](23-raw-gpu-affine-streaming.md) | raw Ready/CPU只读与当前GPU bank、有界批量packing/claims/cache容量拒绝；K1/K2/K3和真实Q8/Q4 exact，6/8GB组合observed fit但BF16速度门失败 |
| [24 fixed bank/dependency-ready](24-fixed-gpu-bank-and-dependency-ready.md) | 固定GPU输出bank/ticket、常驻浮点refiners、显式packing依赖；Q8/Q4全部block N1与取消exact、逐tensor压缩预测；全请求内存下降但BF16速度门仍失败 |
| [25 verified encoder metadata](25-verified-encoder-metadata-cache.md) | 仅CPU metadata/tokenizer复用、文件/配置/取消失效与publish门；同库cache-hit省约.34s、4.93GB observed fit；非保留Q4约2s CPU repack负结果，速度/整体资格未完成 |
| [26 lossless fused affine](26-lossless-fused-affine-import.md) | Q4/Q8一次遍历NEON packing、half bias穷举、direct BF16 alias；非保留Q4同库6.716→5.058s、全部block/端点exact；BF16速度门仍失败，10/16GB仅observed fit |

## 如何使用这些证据

- 各阶段的 library/source/checkpoint/profile、输入、设备和测量范围分别保留在原文及回执中。
  不同二进制、精度、缓存状态和输入的测量不能拼成同一次对照。
- CPU 解码、单投影、encoder、DiT、完整生成及独立内存采样是不同范围；
  数值一致、观测内存可容纳或局部提速，不单独构成全请求速度、媒体质量或硬预算保证。
- BF16 密集槽、ConvRot runtime FP16、raw-GPU/fixed-bank 等阶段的质量或速度失败
  按原结论保留。后续单个 cell 的通过不能覆盖前一阶段的失败，也不能替代完整验收矩阵。
- M5、静态/runtime W8A8、whole-request envelope 与产品接入等未完成项仍属于
  当时记录的边界；归档不把它们改写成当前版本已交付能力。

当前运行方式从[用户指南](../../public/README.md)进入；复验政策参见
[固定验收与可行性合同](../../design/quantized-execution/11-acceptance-profiles-and-feasibility.md)。

## 存储与复现

每篇正文在这里保留唯一完整副本。原 `docs/design/quantized-execution/` 页面保留
转发和原章节锚点。已被脚本、测试或其他文档引用的
[validation 摘要](../../design/quantized-execution/validation/)保持原路径和原字节，
本次不复制或改写 JSON、源码、构建配置及模型文件。

复现命令仍从仓库根目录运行。原文中的 `models/`、`build/`、`outputs/` 和本机回执
只是历史产物位置，不保证随 checkout 提供；先核对原记录要求的版本和身份，再向
新的输出目录复验。本次迁移没有重新执行这些历史模型实验。
新测量按[实验归档约定](../README.md)建立新记录，保留失败样本，不覆盖原观察。
