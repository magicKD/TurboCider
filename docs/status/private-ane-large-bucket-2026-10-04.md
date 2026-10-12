# Private ANE 1024²：大 bucket 接续

接续 [A8 lookahead](private-ane-a8-lookahead-2026-10-04.md)。原目标保持 active；
四格正式 ≥1.2×、完整 LoRA/媒体/latent、带宽校准与实际 device trace 未完成。

## 本轮实现

- Private W8 emitter 的软件行数边界从 2048 扩至 4096。4097 仍拒绝；
  IOSurface extents、两套 W bank、内存准入、A8 producer/consumer fences、
  非有限性与完整 GPU 重算边界不变。新增 2112-row 实际硬件自测，逐行
  检查三次换权，包含最后一行；不据此声称所有 4096-row 几何已资格通过。
- 2112-row bucket 覆盖 Z/Qwen 1024² 的所有 rows，仅需两次 ANE 请求。
  Source/weight/scheduler 接口不变，不创建全模型 W8 或 dense 副本。
- 未选中 executor（例如私有 capability / memory admission 失败）时，
  backend/precision 明确报告普通 GPU；不再默认称作 Public Core ML。
  新 fixture 同时检查 malformed configuration 必须失败，真实 memory
  rejection 必须有 failure receipt，不能伪装成成功执行。
- `projection_range` 提供显式 BM16/BM32 参数，默认仍 BM32。独立 GPU
  sweep 对比 native / fused gate / MPP32 / MPP16，未找到 BM16 正收益。
  模型默认未改为 BM16，也未根据组件约3%的收益提升整模型资格。

Public 默认与 private feature gate 保留。新 Public library 通过稳定发行
检查（flags、actual class strings、direct links）；未执行 package/分发。

## 构建与验证

Private：`cf87e99c679532ec0ba5915689380df6d2af4aa7d62e463d890913457e900870`。
Public：`63e8b046b859f40e8b9a17e9c7fdcbdef059a8e91ccc6cbdc3519f6c23224d2a`。
隔离目录分别为 `build/private-ane-large-bucket`、`build/public-ane-large-bucket`。

- 13 Private host/hardware/MLX（包括 2112-row、A8 off/on、换权、alias、
  source failures、late-chunk 全 GPU、base/A/B/base 与 full-hidden down-LoRA）通过。
- 13 Public Core ML/MLX/receipt、7 runtime host、50 screen host 通过。
- Range fixture 检查 FP16/BF16、BM16/BM32、非零 row/column 起点、实际
  physical pitch vs compact control 逐位一致；非法 tile 拒绝。
- `git diff --check` 通过。完整 make test 的 HEAD 既有 Qwen3 源码字符串
  断言失败未在本轮修改；见前一份记录，不宣称全套通过。

Standalone tool：`tools/native/ane_channel_gpu_benchmark.cpp`，显式调用
`bash tools/native/build_ane_channel_gpu_benchmark.sh` 后运行 1056/4128。
不进入产品构建，不使用 checkpoint 或 ANE，附有 compiled-mode tag 检查。
FG6144 synthetic 的 native / MPP32 / down16 热中位数：1056 rows 为
10.7211 / 10.2505 / 10.2507 ms；4128 rows 为39.6411 / 38.5039 / 38.6242 ms。
该输入的 source L2=0，不是全模型画质/权重质量批准或 end-to-end speedup。

## 完整请求初筛

Apple M4 Max 64 GB，fox/seed42，resident，一冷两热。含 VAE/PNG 的 native
request wall、排除冷请求、关闭 profile。cache=1/fence=1/A8-lookahead=1/
W-prefetch=0/chunks=1。Qwen GPU/Private 都开启相同 Q/K norm-RoPE。

| 模型 / share / bucket | 顺序 | GPU hot median | Private hot median | 倍率 |
| --- | --- | ---: | ---: | ---: |
| Z1024 / Fa4096 / c2112 / 8步 | GPU→Private | 31.227637 s | 27.881100 s | 1.12003× |
| Qwen1024 / Fa5120 / c2112 / 40步，base v1 | Private→GPU | 183.088496 s | 154.228248 s | 1.18713× |

两份 summary 均 complete，全部 channel blocks，0 error fallback。Z 每请求
512 ANE calls、256 future A8 submissions；Qwen 每请求2560 calls、1280 future
A8 submissions。不是 GPU decline、组件倍率或借不同库分母。
最终 receipt 复核时观察到一个短暂 ComfyUI Python CPU 负载，随后该 PID
已不存在；缺少其启动时间，不能据此证明 benchmark 中发生了 GPU 干扰，
也不能证明完全独占。本轮只有 route-start 负载快照，正式重复须加入连续
负载观测；这些数值仍只列为未资格初筛。

Z 实际 slot/估算 bytes=128401408/416303552；Qwen=162316288/474220992。
这是 runtime slots/估计，不是完整 driver/wired/RSS 或低内存资格。
Qwen 热请求比旧 Public/Private 构建的 GPU 数字不同，不能跨库拼接分母。

Z 累计三请求的 GPU FFN / post-GPU join=30.979411 / 2.695270 s；Qwen 为
189.179433 / 22.831417 s。各自整请求均低于要求的1.2×；继续优化 GPU
head、逐 chunk epilogue/reuse ordering 与带宽竞争，不能据 request/ready
提交次数宣称连续 device overlap 已资格通过。

Z 单图 GPU/Private RMSE=7.707359（uint8）、PSNR=30.392692 dB、max error=207。
只有一幅 fox，不是 latent/多提示词/媒体质量批准。

原始结果：

- `outputs/private-ane-large-z1024-a4096-c2112-20261004/`
- `outputs/private-ane-large-qwen1024-a5120-c2112-20261004/`

逐 case summary byte hash、构建身份和未资格声明见
[机器记录](../design/validation/private-ane-large-bucket-pilot-20261004.json)。

边界扩展前 `outputs/private-ane-bucket-z1024-a4096-c2112-20261004/` 被旧
2048 上限拒绝，GPU fallback 回执导致 screen 失败，summary 仍 incomplete；
保留该证据，不把其 GPU wall 当 Private 结果。

## 参考与后续

2026-10-04 复核 Splash #260：2026-10-03 commit
`6010c61ddb613fc79e68def24ee90f78bec5d385` 增加 per-128-row programs over shared
surfaces。本地 reference 仍为先前 pin；未修改 reference/设计稿。本轮只有
较大固定 bucket，不声称已经移植动态行程序。

目前四格最好的已记录初筛（不同隔离构建，各自 matched GPU）：Z512约
1.155×，Qwen512约1.212×，Z1024约1.120×，Qwen1024约1.187×；不是正式
qualification。仍需 profitable share/prefetch calibration、更紧的
GPU-head/epilogue 并发、Z512 staging/FFN、Qwen LoRA（上轮仍负收益）、
完整质量/内存和多提示词正反序验收。Planning JSON 的 FP16/token-row
历史标签也仍待和实际 W8/channel receipt 对齐。

未改发行 build/native、用户模型或 adapters；未 stage/commit。
