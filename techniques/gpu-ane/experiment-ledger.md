# GPU/ANE与系统优化案例账本

[总目录](../README.md) · [证据地图](../references/evidence-map.md)

这是“遇到类似问题先读哪份证据”的比较索引，不是可横向排名的benchmark表。每行使用其来源自己的构建、输入和分母；本轮没有重跑这些设备实验。

| 案例与范围 | 观察 / 结论 | 可迁移经验 |
| --- | --- | --- |
| [FLUX4B历史M4Max512/4步](../../docs/design/parallel-acceleration.md) | 匹配GPU2.2663s vs hybrid1.6279s；特定历史来源/质量范围 | parallelattention/MLP是真窗口；不继承到不同Q4来源 |
| [H3拓扑/CPUtail](../../docs/design/transformer-heterogeneous-report.md) | genericmicro快，不赶得上专用runtimefirst-consumerdeadline | 用真实kernel与消费者deadline，而非总FLOPs分工 |
| [H3QKV / row split](../../docs/design/transformer-heterogeneous-report.md) | 去concat后仍有straggler/all-layer收益很小 | 数据搬运不是唯一问题；完整DAG/生命周期更重要 |
| [M5Z最小bucket](../../docs/design/m5-ane-adaptation.md) | 512误用大bucket12.387s，匹配桶7.119s | rows/profile选择错误可盖过全部kernel优化 |
| [M5Qwen24GiB](../../docs/status/qwen21-m5pro-24gb-streaming-ane.md) | 每步重建suffix被否；compactcache/受控临时wired有该机观察 | 改善重复IO要算冷构建/恢复，质量仍近似 |
| [PrivateW预取](../../docs/status/private-ane-prefetch-2026-10-03.md) | 744hits仍使该窗口Z512hybrid更慢 | hit不是关键路径收益，prefetch必须消融 |
| [deferredjoin](../../docs/status/private-ane-deferred-join-2026-10-05.md) | 同库ON/OFF约0.084%差异，不稳定收益证明 | 安全owner是前提，减少hostfence不保证E2E |
| [Qwen首步share](../../docs/status/local512-qwen-prefill-shares-2026-10-10.md) | 双ref4096有约4.4%信号；全部loadchecks失败 | 保持显式候选；单ref排名翻转不升默认 |
| [QwenW-codecache](../../docs/status/local512-qwen-prefill-weight-cache-2026-10-10.md) | 多保留约1.0–1.3GB，收益反序翻转 | 缓存需同时测内存与cold/warm/顺序 |
| [完整PublicW8](../../docs/status/public-w8-shared-ffn-2026-10-07.md) | 实际256calls执行，旧latentN1fail；诊断时间不算正式倍率 | 功能、质量、速度、物理placement独立 |
| [ConvRotinput/hiddengroup](../../docs/status/convrot-group-scope-2026-10-05.md) | 四步真实512/1024两scope仍fail既定N1 | 组件通过与“更低误差”不能自动升级配方 |
| [nativeSGMMA](../../docs/status/2026-10-08-native-lora-sgmma-e2e.md) | 完整FFN局部改善；整请求native仍比同构建MLX慢14.44% | component→block→E2E，不能借旧scalar时间因果归因 |
| [wideSwiGLUtile](../../docs/design/validation/z-image-swiglu-wide-screen-2026-09-22.md) | 更大tile无稳定赢家，部分NaN曾退出0 | numericfail不计timing，比较真实production控制 |
| [typeddecode/ahead](../../docs/status/performance-progress-2026-10-10.md) | 转换kernel更快，Q4/Q8dependentFFN链ahead仍不如packed | 融合转换不证明全局densebank值得 |
| [LTXStage2sparse](../../docs/design/ltx-stage2-sparse-results.md) | core约4.05×，Stage2约1.238×，质量未过 | regularity与全pipeline税、媒体门禁一起看 |
| [VAEfullsharedtile](../../docs/design/seedvr2-vae-ane-feasibility.md) | smallCoreMLtile快，完整512pipeline仍比GPU慢约8.6% | pack/assembly/pool必须计入 |
| [VAEdualcausalconv](../../docs/design/seedvr2-vae-upscale-causal-conv.md) | decoderABBA约2.68%更低延迟，非balancedfull-request速度资格 | graph边界/virtualhalo比只调share更关键 |
| [SeedVR2serialblocks](../../docs/design/seedvr2-block-weight-residency.md) | DiTactive约11.73→5.23GB；单warm约慢4.76% | 组件memorytradeoff不等于整请求/小RAM资格 |
| [QwenDiTcache新合并](../../docs/status/2026-10-15-main-into-dev.md) | 两promptfast约1.84×但有可见细节变化；defaultoff | 近似与无损、conditionhit与residualreuse分开 |
| [SeedVR2Comfy完整请求](../../docs/benchmark/SEEDVR2_COMFYUI.md) | 4任务8组ABBA速度证据；TC为GPU，overall资格仍false | actualdtype/补丁/完整帧/边界准确才可比较 |

## 如何使用账本

先匹配问题机制，再匹配本机workload；不要找最大的倍率。新实验从其最强已优化基线与正确负控制出发，固定一个可推翻的假设，完整保留失败记录。若旧记录与当前源码冲突，先核对commit/actualrecipe，而不是把某一列改成“当前成绩”。

## 实现参考与实验

可读实现：[参考代码](../labs/media.py)；[实验步骤与边界](../labs/scheduling.md)；[本项全部 case / controls](../labs/catalog.json)。

```sh
python3 -B techniques/labs/run.py profiling-scopes
```

本项控制：operator vs request scope；paired inputs and failed evidence retained。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：historical failure trigger has materially changed。历史负结果限定于其来源/shape/边界，不关闭该方向。
