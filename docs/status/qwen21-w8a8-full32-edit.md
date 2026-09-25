# Qwen Image 2.1：6144-channel W8A8 全 32 层编辑候选

在既有 512² 显式 W8A8 编辑的 29/32 层路线之外，本机 Apple M4 Max
验证了同一 1024-row、6144-channel、完整 32-block manifest 的
**32/32 层**路线：GPU 保留 BF16 FFN hidden 后缀，但不再让层 3/5/7
完整回退 GPU。两条路线均只在明确选择 `gpu_ane`、
`allow_approximation=true`、256px 参考缩放时可用；默认仍为 BF16 GPU。
为防止误用，正式 Session 对全 32 层编辑还要求经过此处试验的
6144-channel W8A8 manifest，4096-channel manifest 继续要求 3/5/7 回退。
显式准入不等于普遍画质合格或物理 ANE 驻留证明。

每条路线在独立进程中 prepare、2 步 warmup，然后对同一有序参考、
prompt、seed 连续两次生成 40 步、**编辑条件缓存命中**的请求。
每轮混合实际执行 `39×32=1248` 次 Core ML FFN，output copy 0；
同一路线两轮 PNG 逐字节相同。请求墙钟包含 VAE decode/PNG，
不含 prepare/模型加载。

| 参考数 | GPU 两轮墙钟 | 6144 W8A8 29/32 两轮 | 6144 W8A8 32/32 两轮 | 全覆盖相对 GPU |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 43.894/43.925 s | 32.352/32.370 s | 31.131/31.135 s | 1.410× |
| 2 | 44.733/44.777 s | 33.236/33.187 s | 31.962/31.985 s | 1.400× |
| 3 | 45.581/45.612 s | 34.111/34.054 s | 32.874/32.862 s | 1.387× |

双参考 **5 步**同条件缓存命中另测 GPU `6.706/6.707 s`，
29/32 `5.560/5.543 s`，32/32 `5.459/5.414 s`；全覆盖请求墙钟
约 `1.234×` GPU。5 步不作为最终画质验收。

| 40 步参考数 | 29/32 对 GPU RGB correlation / RMSE | 32/32 对 GPU RGB correlation / RMSE | 目视重点 |
| ---: | ---: | ---: | --- |
| 1 | 0.99833 / 0.01903 | 0.99830 / 0.01922 | 茶壶结构接近；GPU 和混合均未达到哑光要求 |
| 2 | 0.92777 / 0.11320 | 0.94449 / 0.09906 | 两把茶壶仍在且左右正确，位置和局部形状与 GPU 不同 |
| 3 | 0.98257 / 0.05755 | 0.97218 / 0.07231 | 两把壶与橙色龙贴纸保留；左壶壶嘴/把手细节比回退路线偏得更多 |

按用户允许的生成图像视觉近似标准，三组均能辨认编辑主体，
但三参考若需要壶嘴／把手等参考细节严格保留，应选择 29/32
或纯 GPU；不能把双参考数值的改善外推到其它题材。
独立诊断生成器还对相同双参考请求逐项核验了 text、initial、
sigmas、image_slots、reference0/1 相同；正式 Session 比较器
仅核对元数据和缓存命中，**未转储比较** 1–3 参考的完整条件张量。
各参考数量只有一组 prompt/seed，仍需更多样本和跨机器测试。

显式 CLI 请求可在现有 W8A8 编辑示例上将
`qwen21_gpu_full_ffn_blocks` 设为 `[]`（或省略），并指定
checkpoint-matched 的 6144-channel 32-block compiled manifest。
`plan` 只给出预计覆盖率，不读取大模型；正式推理会核查
manifest 的 checkpoint、精度、形状与分区。实测将 4096-channel
manifest 用于全 32 层编辑会以明确错误拒绝。原有
`[3,5,7]` 路线和默认 GPU 均保持可用。

原始 PNG、报告和配对 JSON 位于 Git 忽略的 `results/qwen21/`：
`session-w8a8-edit{1,2,3}-full32-512-40-cache/`、
`session-w8a8-edit2-full32-512-5-cache/`、
`session-w8a8-edit*-full32-512-*-comparison.json`；配对旧路线和
GPU 见[既有诊断](qwen21-w8a8-ane-diagnostics-2026-09-25.md)。
