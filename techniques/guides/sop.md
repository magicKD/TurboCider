# Apple Silicon模型优化SOP：工程师与AI共用

[总目录](../README.md) · [实验室](../labs/README.md) · [技术卡](../templates/technique-card.md) · [新模型清单](../templates/new-model-checklist.md)

目标不是找到最多开关，而是对一个真实模型形成可复用、能解释、可迭代的优化路线。探索空间保持开放；严格的是输入/来源/有限性/所有权/证据，不是某种优化永远不得尝试。

## P0 · 建立最小上下文包

收集而不改写：checkpoint/index/tensor table、原始loader/tokenizer/resize/scheduler、operation、H/F/heads/actualrows/frames、baseline命令与媒体、实际binary/library/source身份、SoC/RAM/OS/SDK。

```sh
python3 -B techniques/labs/inspect_model.py "$TC_SHARD" --block-prefix blocks. --output outputs/new-model-header.json
python3 -B techniques/labs/run.py technique-composition
```

header-only inspector没有验证payload SHA；实际source/weights仍走已有SourceLease。列出未知项，不按文件名猜dtype、按canvas猜rows或默认认为支持编辑/音频。

输出：一张事实表、一个完整GPU可运行请求、明确的精度与质量基线。

## P1 · 先保留数学，再测全链路

先逐tensor/step比原始backend；跨框架尽量注入同noise/conditioning。原始source的packing/rounding、mask、position、temporal phase、alpha和AV边界先正确，避免把实现bug当“近似优化”。

完整request拆prepare/encode/denoise/VAE/export，分别观察MLX/native/external/process-tree与磁盘。得到criticalpath与actual高水位，才选首个纵切。

## P2 · 按瓶颈选第一组实验

| 现象 | 起点实验 | 可探索的实现轴 |
| --- | --- | --- |
| 小kernel/host submit主导 | Metal `graph/fusion/norm-rope` | blockcompile、epilogue、layout传播、pipeline cache |
| GEMM/权重流量主导 | Metal `gemm/vector`、`quant-formats` | tile、SIMD/MPP、packedconsumer、typeddecode |
| FFN大、存在独立分工 | `reference:split-ffn`、CoreML `--compare-gpu` | channels/rows、backend、phase、LoRA ABI |
| ANE overhead或长cold | CoreML lifecycle、`startup-amortization/bucket-selection` | frozen/runtime/multifunction/load-only/bank |
| attention长 | `online-attention`、Metal `attention`、`token-merge` | exact tiled、prefix KV、structured sparse/近似 |
| pressure/swap/IO | `bounded-io`、phaseplanner、`slot-ownership` | phase释放、typedslots、fixedsubset、mmap/pread |
| VAE/素材内存大 | `causal-halo/global-stat-tiles/upscale-tiles/media-backpressure` | exact上下文或明确近似crop、history/sink |

这些是选择入口而非固定优先级；新架构可能改变主导项。一次改变一个可推翻假设，记录失败条件与下一次重开条件。

## P3 · 用技术卡定义合同

写明source/geometry/layout、输入输出dtype、weight encoding、LoRA顺序、callback phase、读写seams、owner/lastconsumer、cold/hot成本和内存。标记CPU参考、模拟、实际设备、真实模型四种证据等级。

用 [catalog](../labs/catalog.json) 找case/代码，以技术卡补实际新模型参数；教学H/F或某机型share只是起点。先用CPUoracle与negativecontrol抓pitch、尾部、bias次数、rotation/scale等错误。

## P4 · GPU/ANE实现纵切

GPU完整FFN → 合法cut → source/graph → shared inputready → 两branch → correctedhidden/partial restore → 一次join/downLoRA → 下一层。复用现有HybridFfn/Executor/SourceLease/memory账本，不把模型数学搬到通用后端。

先稳定一个share/bucket并完成错误/取消，再探索更多banks/lookahead/deferred/deepgraph。Public与Private接口能力分别验证；Private研究须显式授权与独立构建，不能混成稳定App配置。

## P5 · 从alone到concurrent，再到E2E

固定最强GPU基线；测actual GPU-alone、ANE-alone、concurrent完整traffic与独立候选，不能用exposedwait拟合ANE算力。校准share/cache记完整identity、memoryadmission、sourceweakowner与TTL。

先组件 → 完整FFN/block → 全steps/phase → 全request。source/LoRA/ref/cache/workload变化重建匹配对照。保留load/cold/warm与所有失败/重试，不拼接历史最快分母。

## P6 · 质量、资源与生命周期一起审

fixednumericgate、finite、source/shape、原参数witness、holdoutprompt/seed/adapter/step；图像whole/detail/alpha，视频完整帧/FPS/PTS/运动/身份与音频同步。近似允许改变数值，不允许漏输入或把旧scratch发布为结果。

同时覆盖cancel/retry、latechunk/IO/worker/GPU异常、timeout/latecompletion、A→B→base、跨request/phase、所有者释放。内存必须覆盖未知external/driver项的声明范围，不以allocatorpeak冒充物理RAM硬上限。

## P7 · 组合优化做消融

先A/B单项，再A×B组合；Sana式capability/phase/effectchecker帮助找结构冲突，不能证明近似可数值叠加。缓存会改变后续rows，量化会改变GEMMconsumer，fusion会改变overhead，因而最优share/route可能重排。

输出：单项/组合的完整质量—延迟—内存Pareto表，而非倍率连乘。

## P8 · 沉淀与交付

每张技术卡补具体代码入口、便携命令、小型report身份、测试/skip、成功范围、失败原因与`revisit_when`。适合产品的接CLI/API/App actualreceipt与设备门控；仅实验的保留清晰入口，不改默认。

工程师/AI在离开前核对：陌生人能否按手册复现机制、看懂限制、知道如何替换模型和设计下一次实验？如果还要找某条历史聊天才知道关键guard，就把通用经验补入对应卡片。

## AI执行约定

先读相关技术卡和真实source；区分“代码存在”“本机执行”“模型通过”。不因一次负结果关闭整个方向，也不靠改阈值/删校验追绿色。遇到设备/SDK/内存/权限限制，报告准确边界并给下一步实验；不擅自改系统设置、停止用户服务或删除权重腾空间。
