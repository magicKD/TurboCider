# Apple Silicon 图像 / 视频生成优化百科

面向新模型接入、已有模型提速和性能回退诊断的工程手册。重点是 **Metal GPU、ANE，以及二者在统一内存上的协同**，不是开关大全或硬件峰值排名。核查基线：TurboCider `ad8100f`，2026-10-15；新增内容不改变产品默认、请求协议、权重或发行资格。

## 从这里开始

- 新项目：[完整SOP](guides/sop.md) → [新模型接入](guides/new-model.md) → [完整走查示例](guides/worked-example.md) → [验收清单](templates/new-model-checklist.md)。
- 已有瓶颈：[故障 / 症状索引](guides/troubleshooting.md) → [测量口径](profiling/measurement.md) → [现有工具](profiling/tools.md)。
- GPU/ANE 专题：[切分代数](gpu-ane/partitioning.md) → [交接和所有权](gpu-ane/shared-memory.md) → [调度](gpu-ane/scheduling.md) → [校准](gpu-ane/calibration.md)。
- 内存不足：[驻留与阶段峰值](memory/residency.md) → [权重 streaming](memory/streaming.md) → [VAE](media/vae.md)。
- 实验复用：[模型配方](models/recipes.md)、[实验模板](templates/experiment.md)、[辅助脚本](scripts/README.md)。
- 动手实现：[实验室](labs/README.md) → [28项技术章节的代码/case映射](labs/catalog.json) → [技术实现卡](templates/technique-card.md)。含23个CPU/模拟case、7个真实Metal对照入口、公共CoreML/MLX混合与原生W8A8共享IO。

## 目录

| 分类 | 手册 |
| --- | --- |
| 硬件与适配 | [芯片、能力与设备 profile](hardware/apple-silicon.md) |
| Metal kernel | [性能调优](metal/kernel-tuning.md) · [Attention](metal/attention.md) · [融合与布局](metal/fusion-and-layout.md) · [Command graph](metal/command-graphs.md) |
| ANE | [Core ML 图与布局](ane/coreml-graphs.md) · [加载 / 缓存 / 失败回收](ane/lifecycle.md) · [量化与旋转](ane/quantization.md) · [SRAM / 性能边界](ane/performance-limits.md) · [Private ANE 边界](ane/private-ane.md) |
| GPU + ANE | [切分选择](gpu-ane/partitioning.md) · [Sequence / hidden轴与带宽争抢](gpu-ane/axes-and-bandwidth.md) · [并行调度](gpu-ane/scheduling.md) · [共享 IO](gpu-ane/shared-memory.md) · [完整 LoRA](gpu-ane/lora.md) · [带宽与校准](gpu-ane/calibration.md) · [20 组案例索引](gpu-ane/experiment-ledger.md) |
| 内存与存储 | [驻留](memory/residency.md) · [Streaming](memory/streaming.md) · [Packed 格式与 IO](memory/storage.md) |
| 算法级优化 | [缓存](algorithms/caches.md) · [稀疏 / 重用 / 近似](algorithms/approximations.md) |
| 媒体链路 | [VAE / 卷积 / tiling](media/vae.md) · [视频音频 IO](media/video-and-audio.md) · [辅助图像超分](media/upscaling.md) |
| Profiling | [计时、内存、质量](profiling/measurement.md) · [工具与命令](profiling/tools.md) |
| 实战与来源 | [模型迁移配方](models/recipes.md) · [Splash](references/splash.md) · [vpipe](references/vpipe.md) · [其他项目](references/related-projects.md) · [证据地图](references/evidence-map.md) · [历史对话经验](references/history-lessons.md) |
| 维护 | [覆盖审计](review/coverage.md) · [当前验证](review/labs-validation.md) · [历史验证](review/validation.md) |
| 实现与复现 | [实验室](labs/README.md) · [GPU kernels](labs/gpu.md) · [ANE/W8A8](labs/ane.md) · [量化推导](labs/quantization.md) · [SOP](guides/sop.md) · [KernelWiki/Sana](references/kernelwiki-sana.md) · [Splash W8A8导读](references/splash-w8a8-walkthrough.md) · [扩展验证](review/labs-validation.md) |

## 阅读约定

每个技巧回答：什么时候用、为什么成立、怎样实现、如何测、哪些情况不适用。标签含义：

- **实现**：当前代码中存在，不代表当前设备 / 工作负载已经验收。
- **实测**：绑定具体构建、机器、输入和测量边界的观察；跨机器不继承资格。
- **负结果**：保留失败条件与`revisit_when`；布局、SDK、芯片、source、bucket、qualitybudget或消费者变化后可重新探索，不证明该方向普遍不可能。
- **设计 / 建议**：后续实验或迁移方法，不是可直接调用的生产 API。

收益只和同一实验内的匹配基线比较。历史数字是解释机制的案例，当前产品能力以 [acceleration](../docs/status/acceleration.md)、[使用参考](../docs/public/USAGE.md)、实际 `plan` 与执行 receipt 为准。旧文档中“private 只在 experimental”“9B 无 ANE”等结论不能覆盖后来的代码；本手册按来源日期与适用范围区分。

## 不可破坏的底线

1. 先建立优化后的 GPU 基线，再评价 ANE；少步数、裁剪 token、换参考尺寸不属于同语义加速。
2. 统一内存 ≠ 零交接成本；compute units ≠ 物理 ANE 驻留证明；host wait ≠ device time。
3. checkpoint、LoRA、布局、shape、精度、source generation、owner、completion 都是计算合同的一部分。
4. 近似许可不替代质量验证；组件快不替代完整请求快；预算 hint 不替代实际低内存机器验收。
5. 失败 / 取消先 drain，再释放或重用；半个输出绝不发布。工具不删除原始模型、实验记录或系统缓存。

## 维护与快速检查

`examples/`保留可移植的小型输入，`review/`保留核查摘要；原始报告、编译产物和依赖 overlay 放在根目录已忽略的`outputs/`。若在实验目录内工作，可使用同样已忽略的`build/`、`dist/`、`outputs/`、`results/`、`artifacts/`、`scratch/`或`private/`子目录；对话摘录只保留在这些本地目录中，不纳入提交。

从仓库根目录执行（只需 Python 标准库）：

```sh
python3 techniques/scripts/check_wiki.py
python3 -m unittest discover -s techniques/tests -v
python3 techniques/scripts/compare_runs.py techniques/examples/paired-runs.json
python3 techniques/scripts/interval_peak.py techniques/examples/memory-plan.json
python3 techniques/examples/split_ffn_reference.py
python3 -B techniques/labs/run.py all
python3 -B techniques/labs/check_catalog.py
```

`examples/`是合成CPU演示；`labs/`明确区分参考数学/模拟与真实Metal/CoreML执行，小型设备结果不等于新模型加速。真实模型工具、依赖和风险见 [工具手册](profiling/tools.md)。不复制大型模型代码或产物；独立教学实现直接链接生产参考与第三方出处。
