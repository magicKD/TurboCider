# 受限内存量化执行：GGUF、ConvRot、M5 GPU 与 ANE

日期：2026-09-30。本文集是**设计、实施分解与验收合同**，不是所有后端已经实现的声明。

审阅补充：实施先读 **09 → 10 → 11**，再按 07 选工作包。09–11 将原有候选收敛为
首发范围、实际接口接缝和冻结验收政策；仍是待实现规范，不代表新配置/runner 已可用。
研究分支的负结果可以完成可行性调查，但不能标记加速功能交付。

## 目标与不可改变的前提

量化首先服务于内存受限设备。默认只解码当前层和未来 **1 层**，可配置预取
**2 层**；也允许零预取的按需模式。禁止以全模型 BF16/FP16 常驻作为此方案的
性能前提。提前层数不等于槽总数：当前层 + 1/2 层前瞻通常需要 2/3 个 dense 槽。
如果预算连一层都容不下，使用投影/输出通道分片或直接 packed kernel，而不是
强行分配、依赖 swap 或静默降低分辨率。

本设计覆盖：

- Q4–Q8 GGUF 的 tensor 解码、混合格式、分片文件与 encoder→DiT 组件适配。
- 压缩权重读取、有限提前解码、GPU dense/packed 选路及完整内存分账。
- ConvRot 的旋转复用、局部展开、W8A8 数学与质量约束。
- M5 GPU INT8 矩阵路径的可移植接口和无实机时的 fail-closed 兼容设计。
- 静态 Core ML W8A8、动态权重 runtime ANE W8A8，以及图内 tile/图复用。
- 可执行工作包、负结果退出标准、组件与完整请求验收。

## 阅读顺序

| 文档 | 内容 |
| --- | --- |
| [01 现状和参考审计](01-current-state-and-reference-audit.md) | 代码事实、Unsloth/vpipe 借鉴边界、历史证据与许可边界 |
| [02 GGUF 与组件支持](02-gguf-format-and-component-support.md) | Q4–Q8 registry、encoder/DiT 映射、文件安全、接口与质量 |
| [03 有限解码与内存](03-bounded-dequant-and-memory.md) | 1–2 层前瞻、预算、状态机、过大层、预取/计算重叠 |
| [04 GPU 与 ConvRot](04-gpu-performance-and-convrot.md) | 慢因、多候选、dtype、旋转、W8A8 数学 |
| [05 M5 W8A8](05-m5-w8a8-backend.md) | MPP、INT32 累加、动态 A8、能力门禁、无 M5 验证合同 |
| [06 静态与 runtime ANE](06-static-and-runtime-ane.md) | 两种 artifact、动态 scale 难点、tiling、模块复用、并行与内存 |
| [07 实施和验收](07-implementation-and-acceptance.md) | 逐文件工作包、测试矩阵、性能/质量/内存/发布门禁 |
| [08 本轮交付与实验](08-experiments-and-delivery.md) | 实际改动、命令、已跑/未跑证据，不与规划混用 |
| [09 首发与组件合同](09-release-scope-and-component-contracts.md) | R0–R5、真实 GGUF 目录差异、Z/Qwen3 映射、逐类 dtype profile |
| [10 执行与产品接入](10-execution-and-product-integration.md) | 配置冲突、同步/异步 decode、slot/lazy graph、分片恢复、P4 产品接入 |
| [11 固定验收与可行性](11-acceptance-profiles-and-feasibility.md) | 数值/媒体/内存/性能硬判定、runner 合同、INT8 证据与退出规则 |
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

## 当前状态速查

| 能力 | 本轮状态 |
| --- | --- |
| 原生 MLX Q4_0/Q4_1/Q8_0 及浮点 GGUF | 已有，仍不是全部 Q4–Q8 |
| CPU Q4/Q5/Q6/K/Q8 有界目标解码 | 已实现，含 Q8 ARM SIMD；见12；不自动授予整图/后端资格 |
| IQ 解码和 encoder GGUF | Qwen3-4B Q8/mixed K实验接入与Z组合已跑，未发布质量/产品资格，见15；IQ仍待实施 |
| GGUF 0/1/2层前瞻 GPU 执行 | Z/Qwen3 resident/streamed及Z浮点refiner流式实验已跑，见13/15/17/18；24新增固定GPU bank/依赖Ready与同源全部block N1单cell；tiles、量化refiner真实fixture与整体资格仍未完成 |
| Z ConvRot gate/up 共用旋转 | 本轮新增显式实验路径；证据见 08 |
| Z 默认启用 butterfly / ConvRot W8A8 | 未启用；现有 butterfly 不是 INT8 GEMM |
| M5 GPU ConvRot W8A8 | 设计/未实机验证；本轮无 M5，不发布默认配置 |
| 静态 Core ML W8A8 | 已有独立研究路线；原生旋转 ConvRot+A8 仍未实现 |
| runtime ANE | GGUF/ConvRot FP16直填API已验证；ConvRot模型consumer实际执行但N1失败，见21；动态QDQ数值筛选通过但INT8算术unknown，见14 |
| 低内存 GPU+ANE 全请求认证 | 本方案尚未取得；resident 历史成绩不能替代 |

## 架构决策

1. 文件格式、数学变换、执行 dtype、设备 placement 分开描述。GGUF、INT8 文件
   或 Q/DQ 节点都不是硬件 INT8 运算的证明。
2. 保留原始量化 codes/scales；反量化不额外重做 W4/W8 量化。A8、反旋转后的
   重新 W8 量化和 SmoothQuant 均是独立近似，需要 provenance 与授权。
3. `storage=quantized` 与 `compute=dense_bf16` 可以同时成立，dense 只占有限槽。
4. 复用 [streaming 协议](../streaming/03-runtime-protocol.md) 和
   [内存合同](../streaming/05-memory-contract.md)。既有默认/认证布局不被此草案重写。
5. GPU-only 是必须持续优化的分母；ANE 必须超过**同预算、同量化、已优化 GPU**，
   不能靠旧 Q8 dtype/调度问题制造漂亮加速比。
6. 先实现 GPU 有界解码和小范围 ConvRot 优化，再独立证明动态 ANE W8A8。
   后者失败不阻塞 GGUF 支持。

文中配置和类型若标注“拟议”，不可直接复制进当前产品请求；只有 08 明确列出的
命令/开关是本轮新增可执行入口。没有新增生产 manifest、自动选路或 M5 资格。

[验收 profile 模板](validation/z-q8-qualification-template.json) 明确为 unbound/not_run，
不能用作资格凭证。源码工作从 R0 + 单一 Q8 纵切开始，不等待全部 Q4–Q8 decoder。
R1 暂不支持 tile/p=2/encoder GGUF，最终目标不变；这些能力分别由 R2/R3 验收。
