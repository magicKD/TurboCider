# 独立原生后端 Session 接线与首组完整请求（2026-10-08）

**完整 native GPU-only Qwen21 base 请求已跑通；本组去噪快约 2.41%，首次完整请求反而慢约 1.68%。图片非常接近，但当前没有整图加速、ANE 并行、LoRA 或 App 发布资格。** 只运行一组两张 512² 图片，没有下载模型、复制 checkpoint 或导出全模型。

这次实现的是用户要求的另一条独立后端：应用自己编码 Metal command graph，完整 32 层 DiT／Euler 不调用 MLX 算子。既有 loader、文本 encoder 和 VAE 作为阶段边界保留。默认路线仍是 MLX。

## 已完成的工程合同

- `qwen21_denoiser_backend= native-metal-graph` 显式选择实验 executor；v1／v2 参数、域限制和 plan／result backend 一致。当前只接受本地 Qwen21 dense BF16、base 512²、20 步、GPU、cache off；不忽略 LoRA／ANE／quantized／streaming 等混入配置。
- Session 使用 held-fd DiT、文本、Tokenizer、VAE 来源；实际 Data owner、source lease、shape／stride／device／readiness 跨阶段保留；最终只复制 128 KiB latent 回 MLX，不存第二份 checkpoint。
- 所有 native 分配接入现有 MemoryExecutionContext／账本，22 类 site、89 个 backing，actual capacity 提交。MLX sources 明确包含于产生后的 observed baseline，不给匿名数组虚构账本身份。
- 内存资格仅属于 native DiT phase，1 GiB framework guard／4 GiB system reserve 是准入估计；完整请求 constrained-memory 认证尚无。报告独立于 MLX allocator peak 和完整请求 admission。
- 失败时区分资源释放、phase 终结和永久隔离。修复了释放 native owner 后 phase 收尾失败仍可能返回清理成功的问题；物理完成或账本清理不明时保留执行 owner 并禁止复用。
- 准备、denoise、VAE、request wall 分开；当前每次新请求重新准备 native PSO／arena，不把它描述成跨请求复用。

具体 API／来源／scope／错误语义见 [设计 13](../design/qwen-image-gpu-ane-command-graph/13-native-session-and-memory-phase.md)。完整 native W1 的两分支、提交次序、slots、集合物理完成、Base→LoRA 方案见 [设计 14](../design/qwen-image-gpu-ane-command-graph/14-native-w1-parallel-ffn.md)，**14 尚为设计，未实现**。

## 当前构建与轻量验证

runtime：`tc-runtime-build-v1-c45b3ba125afd3d90aa0b5cb4f239e179b37e52f90f22ef123adc75affd130c6`。全库使用 CommandLineTools、private-ANE 支持构建；本次请求只使用 GPU。因为内部 Request 增加字段，已全库重建，没有将旧 library 与新 C++ ABI 混链。

- 原生 MemoryBridge／NativePhase CPU 合同通过：容量与 view 去重、thread／opaque-token 身份、close 后 release、未物理 drain 不复用、全部分配计费、finish ledger=0、父 Context 拒绝。
- 新 executor 请求合同 5 个测试通过；常规合同 129 个测试、3 个既有条件 skip，通过。常规 suite 首次受限环境有一项读取 GPU 名称不匹配，获得 Metal 访问后完整复核通过；没有修改测试来放宽设备匹配。
- 新构建 tiny 生图／单参考图／双参考图的实际 MLX Transformer＋BF16 Euler 对照全部 0 ULP／RRMS=0；原严格 1%／0.01 门槛保持。它只证明小几何完整图，不授予真实编辑 Session 资格。
- Session CPP 严格 C++20／Wall／Wextra／Werror host syntax 与独立生命周期审查通过。

## 真实配对条件与结果

M4 Pro／48 GiB，已有本地 Qwen Image 2.1 BF16；512²、base、seed 42、20 步、24 个实际文本 token、相同初始 noise 算法／sigma schedule，近似 DiT cache 和额外诊断关闭。

prompt：`A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.`

顺序 native → MLX，每个 route 仅一张，分别新 CLI 进程；没有真实模型 warmup／重复采样。两次共享 OS 文件／驱动缓存，未重置，之前 tiny 测试也可能预热通用 shader。因此这是一组本机工程筛选，不是统计稳定跑分或完全冷缓存排名。MLX 对照是普通 base 路线，不是对所有已有优化候选选出的最快路线。

| Host phase / seconds | Native GPU | 现有 MLX GPU |
|---|---:|---:|
| Text encoder | 2.8014 | 2.8729 |
| Native prepare（单独报告） | 1.6483 | — |
| Denoise | 41.6823 | 42.7120 |
| VAE decode | 1.0701 | 0.9391 |
| Session request wall | 49.8845 | 49.0618 |
| 监督进程 wall（含 CLI 初始化／退出） | 50.4120 | 49.1466 |

Native denoise 下降约 **2.41%**，request wall 增加约 **1.68%**。两边 encoder／noise／VAE 数学仍相同；native prepare 放在 denoise 计时外，但已包含于完整请求 wall，不能用 denoise 快来掩盖首次成本。尚未测第二次 warm request，不能声称缓存后一定更快。

最终 8-bit RGB PNG（不含 alpha）relative L2=**1.1041%**，MAE=0.7134／255，RMSE=1.6245／255，PSNR=43.9164 dB；最大单通道差 76／255，不能据整体误差声称所有像素都相等。已分别查看原生和 MLX 图，壶体、壶嘴／把手、反光、桌面和背景构图非常接近，未见明显质量退化。这不是所有 prompt／seed／LoRA 的质量认证。

图像：[Native](../../outputs/qwen-native-session-20261008/e2e-pair/native.png)、[MLX](../../outputs/qwen-native-session-20261008/e2e-pair/mlx.png)。请求、raw stdout／stderr、来源 stat 身份、runtime／binary SHA 和机器快照见 [配对回执](../../outputs/qwen-native-session-20261008/e2e-pair/receipt.json)，像素指标见 [image-metrics.json](../../outputs/qwen-native-session-20261008/e2e-pair/image-metrics.json)。这些不复制模型内容或计算额外全 checkpoint SHA。

## 内存与清理

Native phase 最终 `succeeded`，89 次 reserve；storage／reservation／pending-release／cache／unknown 全部归零，未隔离、无 taint、trace 无丢失。已有 source 在 baseline=14,414,736,160 B；实际 native allocation 增量峰值 **225,764,352 B（215.31 MiB）**，native phase sampled process peak=14,651,239,200 B。这些是 phase 指标，不能叫完整请求峰值。

`/usr/bin/time -l` 全进程 peak footprint：native=15,416,798,584 B、MLX=16,913,099,960 B；它和 MLX allocator peak／阶段 sampled peak 各有不同口径。两次前后系统 swap used 均为 3305.75 MiB，没有增长；无全请求 swap-free 认证。来源文件 size／inode／mtime／ctime 前后相同。

本轮清理 237 个 object／archive／module-cache 与临时 CPU fixture binary，共 **20,990,080 B（20.02 MiB）**，逐项 SHA／移除确认见 [owned-cleanup.json](../../outputs/qwen-native-session-20261008/owned-cleanup.json)。保留测量 library／CLI、tiny probe、日志、source/build manifest、两张 PNG 与请求回执。只删除有明确所有权且可重建的本轮产物，不改系统缓存、模型、旧 App 或历史测量 library。

## 下一步

1. 准备开销主要抵消当前约一秒的去噪收益。可复用与 source 无关的 PSO／kernel registry，但需保留 device／math／shape／build 身份、预算和失效合同；现有 Backend 的 allocation callbacks 绑定 phase，不能直接跨请求复用已关闭的 Context。
2. 依据完整 FFN 历史结果，接原生 W1：GPU leading 6144／ANE suffix 6144，producer 完成及量化 status 成功后先提交 ANE，再提交 GPU complement，FP32 restore／join 后 gated residual。所有 staging／同步／restore／join 必须纳入整个 block 计时。
3. 新代码先 CPU 合同／轻量 block，确有完整路径收益再一组 E2E。首步仍 native GPU，预取／S1／LoRA 暂不与首个 base 并行候选混入；一个 ANE 请求槽的物理使用期必须覆盖最后 restore／join consumer。
4. 之后才扩大 LoRA、编辑 encoder、Z-image 和 App；不能把历史 private／MLX 组件倍率授予新 native 后端。

本轮两张图后停止真实模型测试，不追加扫描或连续高负载生成。
