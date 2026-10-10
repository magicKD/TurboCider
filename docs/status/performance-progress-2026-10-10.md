# 512² 本地性能优化进度：2026-10-10

设备为M4 Max 64GB / macOS26.6.2。只使用已有本地 Qwen Image2.1、
Z-Image Turbo 和原 inference-time LoRA。以下是各自匹配GPU对照的 warm
request中位数，不跨模型/source/步数/构建借分母；耗时下降不是吞吐提升。
当前实验路线仍显式 opt-in，产品默认 GPU 不变，完整优化目标未完成。

## 已有整请求加速信号

| 512² workload | 匹配GPU s | 候选 s | 名义耗时下降 | 当前判断 |
| --- | ---: | ---: | ---: | --- |
| Qwen base生成 /40步，当前库正反序汇总 | 43.353 | Private34.542 / frozen33.121 | 20.3% /23.6% | 两路均有信号；frozen热快、冷加载更重 |
| Qwen原LoRA单ref编辑 /6步，最新反序 | 10.587 | 首步ANE5120 10.257 | 3.1% | 后五步完整GPU；4096/5120排名翻转 |
| Qwen原LoRA双ref编辑 /6步，最新反序 | 12.740 | 首步ANE4096 12.175 | 4.4% | 首步3.745→3.251s；4096为显式候选 |
| Z原BF16 base灯塔 /9步 | 8.211 | Private matched6.674 | 18.7% | 长caption按真实rows匹配bucket |
| Z原BF16 LoRA灯塔 /8步 | 8.669 | Private matched7.320 | 15.6% | 完整原adapter，冷启动尚无收益保证 |
| Z ConvRot base狐狸 /9步 | 9.614 | Private matched7.745 | 19.4% | 绑定自身ConvRot GPU配方，不跨BF16排名 |
| Z ConvRot LoRA灯塔 /8步 | 9.795 | Private matched8.387 | 14.4% | 保留可见屋顶颜色变化，需要更多画质确认 |
| Z GGUF Q4原LoRA灯塔 /8步 | 10.756 | Private matched9.458 | 12.1% | resident packed bank；不是ahead解码收益 |

Qwen最新四个share窗口及base两个对照窗口的continuous load check均失败，
Z上述数据也是诊断而非正式完整性能资格。不同窗口不要平均成一个全局
倍率；没有device trace证明物理GPU/ANE占用/重叠，也未做全部scene/seed
的画质验收。所有原始冷/热数据、图像与失败证据保留。

依据：[Qwen base当前库三路](local512-qwen-frozen-base-2026-10-10.md)、
[编辑正反序share](local512-qwen-prefill-shares-2026-10-10.md)、
[Z匹配bucket](local512-z-matched-bucket-2026-10-09.md)。

## GPU kernel、GGUF 和提前解码

已有typed packed-word共享解码、BF16 partial、LoRA joint A/B/F32 ranks、
source复用和有限buffer所有权机制。新shared-word核把opt-in F32 partial
推近BF16路线，但没有超过已较快BF16选择，不修改普通packed QMM默认。
参见[实际kernel与ConvRot完整请求](local512-affine-shared-word-2026-10-10.md)。

融合typed affine解码和finite status后，75MiB准备约3.9→1.7–1.8ms；
实际R1 decode+GEMM约10.3–10.5→8.0ms，仍慢于packed约6.1–6.6ms。
所以“提前转dense”不能仅看更快解码核就接入模型。
参见[融合finite](local512-affine-fused-finite-2026-10-10.md)。

真正的有界两job/独立producer-stream ahead-decode已实现并运行四层真实
不同权重FFN依赖链。最新Q4 packed73.533/ahead76.141ms，Q8
74.011/76.818ms，ConvRot81.079/80.043ms。GGUF仍选packed；ConvRot
仅约1.3%组件信号，还没有attention/LoRA/全图收益证明。内存ledger覆盖
实际current/next消费者和escaped readers，不把两job说成两完整FFN槽。
参见[ahead实际结果](local512-affine-ahead-2026-10-10.md)。

本轮新增两路LoRA rank候选，216实Metal numeric/192非法contract通过；
真实adapter layer0的大行数在线A→B多数持平或更慢，独立保留、不接模型。
参见[paired-rank负结果](local512-qwen-paired-ranks-2026-10-10.md)。

## 当前选路与尚未完成项

Qwen编辑的路线是首步FFN通道分工，不是整步交ANE；attention仍依其GPU
依赖执行。后续KV-hit五步完整GPU，避免低工作量时staging/同步税。
单ref保留5120候选，双ref4096有正反序同方向信号，但新比例不升为自动
默认。W-code cache虽然真实命中，却多保留1.0–1.3GB且收益反序翻转，
继续关闭，见[cache对照](local512-qwen-prefill-weight-cache-2026-10-10.md)。

Qwen40步base不能套用“后续不值得并行”：后39步是主要计算，两种ANE
候选已有约20–24%热请求信号。原六步LoRA纯生成仍约8.5–8.6s，没有
稳定整体优势，继续GPU，见[生成phase screen](local512-qwen-generation-phases-2026-10-10.md)。
Encoder已经实现compiled GPU/GQA和Private/Public混合入口，但完整单/双
图编辑没有稳定显著收益，优先GPU，见[encoder完整对照](local512-encoder-compiled-gqa-2026-10-09.md)。

Private支持可复用runtime-weight W8A8通道分工；Public有checkpoint-bound
frozen/Core ML及共用执行器路径。Public compute-unit selection不能当作
所有算子物理落ANE或原生INT8 MAC证明。当前没有把Private收益借给Public，
也没有完成所有base/LoRA、编辑/encoder、冷/热和Public/Private的统一矩阵。

下一阶段重点仍是减少首步handoff和LoRA成本、确认双ref4096候选、更多
场景/seed与负载对照、GGUF更长实际模型链收益。允许适度数值近似，但
不放宽source/shape/finite/memory与完整GPU fallback安全边界。

## 本轮代码整理与提交边界

整理share screen参数/实际channel receipts及新测试，保留rank候选在独立
research工具，补齐两份机器记录和状态入口。44项host回归、1项实Metal
回归通过且无skip；48个native请求与16份内存记录独立核对通过。
没有新生产native库build或默认推广，没有下载/改写原权重或LoRA。
原先六个tracked草稿及ConvRot未跟踪草稿未改写、未夹带提交；日志/PNG/
模型/build binary不加入Git。本轮owned jobs已结束，临时测试build已回收，
无需删除用户缓存或保留的已验证库。

后续用户要求收好剩余工作区代码，原ConvRot实验组已另行核对并单独
提交，见[工作区收尾与当前库回归](convrot-compiled-commit-2026-10-10.md)。
