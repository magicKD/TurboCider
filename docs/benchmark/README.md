# TurboCider benchmark

主报告：[图像与视频性能评测](PERFORMANCE_REPORT.md)。

2026-10-19更新：[当前最快已测配置](CURRENT_FASTEST_CONFIGS.md)。集中记录各模型
的Public/Private/GPU、尺寸/步数、热命中与fresh边界、LoRA/近似档，以及已清理
须重建的产物；Qwen Base完整40步文生图研究基线约28.7–28.8s为Private runtime。
本次仅整理已有证据，不重跑、不将受负载污染的观察晋升为正式速度资格。

2026-10-15补入[SeedVR2与ComfyUI完整超分对照](SEEDVR2_COMFYUI.md)：
图像256→512／512→1024为**4.81×／4.43×**，完整25帧视频为
**3.46×／3.48×**（原生GPU，非ANE收益）。每路6个warm，采用各任务
更快的Comfy预设及其同组TC记录；包含导入、VAE、采样和导出。
速度门不等于整体质量资格；内存未形成同口径跨框架排名。

原2026-10-12生成模型评测收敛已有实测，不再持续扩大实验网格，范围为
FLUX4B、Z-Image Turbo、Qwen Image2.1、LTX2.5，无额外LoRA；只把
兼容且完整成功的框架放进对比，不强制适配不兼容项目。

已验证优势：固定Unsloth/Metal路径上Qwen GGUF **3.40×**、Z-Image
**3.15×**；对mflux0.22，Qwen/Z混合约 **1.32–1.34×**。倍率各用
同组分母，混合的数值近似、cold、内存和参考胜出结果明确保留。

详细结果：

- [SeedVR2图像／视频超分与ComfyUI对照](SEEDVR2_COMFYUI.md)、[全部样本与身份](../design/validation/seedvr2-comfy-benchmark-current-20261015.json)
- [Z完整五路线Unsloth/mflux对照](Z_UNSLOTH_FRAMEWORKS.md)
- [Qwen BF16/量化/缓存对照](QWEN_BF16_FRAMEWORKS.md)
- [FLUX4B历史与当前运行状态](FLUX4_CURRENT_STATUS.md)
- [LTX完整音视频与近似权衡](LTX_VIDEO.md)
- [原始样本catalog](results-20261011.json)、[保留配置](best-local-configs-20261011.json)
- [框架身份与公开评测](OPEN_SOURCE_COMPARISONS.md)、[证据审计](COMPLETION_AUDIT.md)

FLUX9B只留[历史附录](HISTORICAL_RESULTS.md)，不进入当前排名。下载、
单cold预检和未配对历史速度不作为新的外部加速比；原模型、媒体、
失败日志与只读reference均保留，模型/媒体/build不提交Git。
