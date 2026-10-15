# 其他参考项目与可迁移技巧

外部项目的可复用机制已落为[本地设计样本](../designs/README.md)、[GPU/ANE 方案](../designs/hybrid-ffn.md)与[分阶段地图](../models/stage-map.md)。以下版本/文件是可选溯源，不需要为了理解方案打开其他仓库。

[总目录](../README.md) · [证据地图](evidence-map.md)

本页是已读本地源码/项目文档的路由索引，不进行跨项目速度排名。外部目录可缺失，不进入TurboCider构建；具体head见`sources.json`。不复制第三方代码时无需搬整套依赖，若以后摘录实现须保留实际许可证/NOTICE并重新核查。

| 本地参考（相对TurboCider根目录） | 本轮读到 / 对照的内容 | 迁移与边界 |
| --- | --- | --- |
| `../gpu_ane/mac_transformer` | `docs/EXPLORATION_REPORT_2026-09-01.md`、异构topology审计 | pairedgate/up channels、publicbacking、CPU first-consumer deadline；generic MLP收益不覆盖专用H3 |
| `../gpu_ane/flux2-engine` | `docs/ARCHITECTURE.md` | residentbackend、smallestbucket、bridge wrapper复用、原attention dispatcher；有选择的内部mflux依赖需升级回归 |
| `../gpu_ane/ANE` | README与动态runtime组织 | channel-first surfaces、MIL/dynamicweights、SRAM/dispatch研究；privateAPI非生产框架，单convINT8观察不是所有模型MAC证据 |
| `../references/splash-ffn` | AneFfn/Measurement/Calibration、早期本地历史 | FFN-onlySmoothQuant、rowthreshold、breaker、双bank；LLM retrieval质量不能替代扩散trajectory |
| `../references/smoothquant` | README、校准/scale思想 | `X/S`与`W*S`，校准outlier；CUDA/LLM速度不能移为ANE速度 |
| `../ltx-mac` | 本仓库migration/验收合同与共享runtime源码 | dualGPUqueue、audio batching、ConvRot、cleanexecVAE、stage释放；相关数学与媒体parity另验 |
| `../references/FastVideo` | 本仓库FastH3INT6/VSA parity与runtime对照 | geometry/MRoPE/noise/INT6、dense/VSA provenance；referencegather不标SIMD实现 |
| `../references/ComfyUI-SolAttn-MPS` | README与本仓库LTXsparse研究对照 | tiledsimdgroup、online softmax、exact sinks、首步dense、finite诊断；H3M3Ultra数字不套LTXM4Max |

## 参考实现给出的“失败也有用”的知识

mac_transformer证实：一次micro最快值和随机反序确认中位数可明显不同；H3CPU tail在genericMPSGraph有效，却赶不上真实专用runtime deadline。保留机制不等于默认启用。

SolAttn MPS说明H3强制FP16可产生非有限，AAC错误可能只是上游NaN的下游症状；不应首先调codec。其fallback仍须有稳定denseSDPA，日志里“kernel编译失败后有效输出”不能冒称稀疏被执行。

SmoothQuant原理是outlier平滑，不是一个“对任何backend都快”的magicflag；fake quant测试、Q/DQ图、actual integer execution分别报告。

## 从外部思路到仓库功能的检查顺序

读取目标版本真实代码 → 写数学/格式/owner合同 → 用本项目优化GPU作分母 → 组件/真实block → 完整媒体和内存 → public/private发布边界。不要为了引用一个技巧把兄弟项目设成隐式运行时依赖。

## 外部第一手补充

- [Apple ANE Transformers](https://machinelearning.apple.com/research/neural-engine-transformers)：layout/graph表达的背景，不是本仓库性能证据。
- [Apple Metal性能讲解](https://developer.apple.com/videos/play/wwdc2020/10632/)：GPU性能诊断背景。
- [MLX extensions](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)：执行图/stream与custom primitive边界。
- [MLX v0.32 Load](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/common/load.cpp)：异步IO wait语义；对应本仓库错误锁存案例。

在线API/工具版本会变化；本手册的测量结论来自固定本地记录，不能因上游README更新就给历史binary换标签。
