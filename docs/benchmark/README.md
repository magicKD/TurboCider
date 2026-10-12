# TurboCider benchmark

主报告：[图像与视频性能评测](PERFORMANCE_REPORT.md)。

按2026-10-12最新要求收敛已有实测，不再持续扩大实验网格。只考虑
FLUX4B、Z-Image Turbo、Qwen Image2.1、LTX2.5，无额外LoRA；只把
兼容且完整成功的框架放进对比，不强制适配不兼容项目。

已验证优势：固定Unsloth/Metal路径上Qwen GGUF **3.40×**、Z-Image
**3.15×**；对mflux0.22，Qwen/Z混合约 **1.32–1.34×**。倍率各用
同组分母，混合的数值近似、cold、内存和参考胜出结果明确保留。

详细结果：

- [Z完整五路线Unsloth/mflux对照](Z_UNSLOTH_FRAMEWORKS.md)
- [Qwen BF16/量化/缓存对照](QWEN_BF16_FRAMEWORKS.md)
- [FLUX4B历史与当前运行状态](FLUX4_CURRENT_STATUS.md)
- [LTX完整音视频与近似权衡](LTX_VIDEO.md)
- [原始样本catalog](results-20261011.json)、[保留配置](best-local-configs-20261011.json)
- [框架身份与公开评测](OPEN_SOURCE_COMPARISONS.md)、[证据审计](COMPLETION_AUDIT.md)

FLUX9B只留[历史附录](HISTORICAL_RESULTS.md)，不进入当前排名。下载、
单cold预检和未配对历史速度不作为新的外部加速比；原模型、媒体、
失败日志与只读reference均保留，模型/媒体/build不提交Git。
