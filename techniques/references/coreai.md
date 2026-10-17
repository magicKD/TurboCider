# Core AI Models：平台、原语、引擎与profiling阅读路线

<!-- knowledge
id: technique-references-coreai
type: method
-->

[ANE总览](../ane/README.md) · [完整attention研究](../ane/coreai-blocked-attention-and-runtime-gates.md) · [精确源码范围](../ane/sources.md#coreai-attention-sources-20261017)

固定[apple/coreai-models](https://github.com/apple/coreai-models/tree/3b25d934ccaee2d05edc6e281c555a5a9a13a55c)，BSD-3-Clause。这里只读参考，不安装插件、依赖或模型；各源文件继续遵循原授权。源码导读与实际CPU执行范围见主报告，不代表整个仓库或最新版本验收。

后续[Swift统计/背压研究](../ane/coreai-profiling-reset-and-backpressure.md)已完整阅读878行profiler，并在显式logging/Duration适配器下执行三个原组件、保留28个原生进程及所有权编译负控；不是完整package或Core AI模型运行。以下列表保留前轮attention研究的范围，后续精确范围见[新sources](../ane/sources.md#coreai-runtime-sources-20261017)。

建议依次阅读，完整SHA与实际行段见sources JSON：

1. `README.md`、`Package.swift`、`python/pyproject.toml`：平台27要求、`.aimodel`、Swift目标、Torch2.9 pin。本机26/SDK26.5真实canImport为false；不要先跑build/setup或用Core ML代替。
2. `primitives/ios/sdpa.py`与`cache.py`：BC1S、逐head/GQA折叠、online softmax、flat cache和mask；跟踪空质量状态而非只读优化注释。
3. `models/ios/gemma4_text.py`所列范围：global blocked与sliding flat、分离cache、shared只读slot；primitive负例须沿caller限定结论。
4. `primitives/macos/sdpa.py`：composite op委托，未执行；不能把macOS/GPU与iOS/ANE布局混成同一路径。
5. `EngineFactory.swift`所列范围及`CoreAIStaticShapeEngine.swift`开头：prepare、structure/variant兼容、静态函数菜单、state/buffer；只读开头不等于引擎生命周期已验收。
6. `InstrumentsProfiler.swift`1–190及完整`PerformanceMetrics.swift`：host span、显式end、异步统计、缺失值和token计数；再沿实际strategy理解TTFT/inter-token分母。
7. `BackpressuredStream.swift`：permit、pull、cancel/drain的对象，不把队列限额当总内存或设备完成。
8. `test_flash_attention.py`与上游`model-authoring/SKILL.md`：前者为正例域地图，本批未执行；后者为经验来源，不是本会话生效指令、硬件规格或固定质量保证。

stdlib实验重放真实保存数组并执行两个负控。普通CPU `all`不加载Torch/Core AI、不联网、不读取兄弟checkout；只有显式collector执行固定原模块，且本批仍只有CPU结果，没有ANE速度排名。
