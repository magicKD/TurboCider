# Private ANE：有界 A8 lookahead 与 channel share 搜索

接续 [scale cache / launch fence](private-ane-scale-cache-2026-10-04.md)。
日期采用 Asia/Singapore 2026-10-04；原任务保持 active。不是四格 ≥1.2×、
完整 LoRA/latent/media、实际 device overlap 或低内存资格通过报告。

## 实现与正确性

- W8 仍恰好两套 bank，未创建全模型 W8 或 dense checkpoint 副本。
- `TURBOCIDER_PRIVATE_ANE_A8_LOOKAHEAD=1` 增加恰好两组 chunk-sized
  A8 code/token-scale surfaces。当前请求与 leading ready producer 提交后，
  staging queue 准备下一块的 alternate slot；复用只发生在旧 ANE consumer
  完成之后。A8 与 layer-ahead W prefetch 是两个独立机制。
- 首块仍先验证 A8 producer 再 launch。下一块 producer 的非有限性、
  GPU 失败与超时仍必须被检查；失败不发布 partial output，HybridFfn
  完整 GPU 重算。所有出口 drain 已提交的 A8 producers；producer 超时
  禁止 executor 继续复用，ticket 本身保留异步资源。
- Headroom retry 保留当前 activation，未来块只提交一次；gate/up LoRA
  修正与 hidden ABI 不变，channel down-LoRA 仍消费 full hidden 恰好一次。
- 开关进入 executor identity、JSON receipt 与 screen；新增
  `a8_lookahead_enabled`、`a8_prefetches_session_total`、
  `a8_wait_seconds_session_total`。Host readiness span 不是 GPU kernel duration，
  prefetch count 也不是物理 overlap 证明。校验完整性、有界、单调及 session
  policy 一致，纯 GPU 分母清理所有 private 环境。
- 独立 follow-up：每个 Device 仅保留当前 seed 的 2 KiB 只读 +/-1 符号表，
  active ticket 保留旧 seed 表直至 producer 完成。CPU 使用同一个 unsigned64
  `rotation_sign` oracle；H128 取 H512 表的前 128 项。GPU 不再逐 block
  计算固定符号的 64 位 hash；Hadamard 顺序、scale、RNE、recipe v2 均不变。

Public 默认、public-only 构建与 private 动态加载隔离不变。A8 lookahead
默认关闭，继续显式消融；本轮不根据组件收益提升模型默认资格。

## 验证

- 13 Private host/hardware/MLX 测试通过，无 skip；其中 executor 同时测
  lookahead off/on 的三块循环复用、base/LoRA、逐位输出/hidden 对照、
  换权、headroom、alias 与第二/第三块失败后的恢复。
- Channel MLX fixture 显式开启 A8 lookahead 与 W prefetch，覆盖短行/tail
  padding、base/A/B/base、一次 full-hidden down-LoRA 与 late-chunk 全 GPU
  重算。原 raw source L2 0.0668762 / partial-energy-normalized L2 0.029908
  未改变；不以此宣称模型画质通过。
- 9 source/dtype、H128/H512 scale/code oracle、弱 generation、LRU 与 cache-hit
  nonfinite 检查通过；新增 high unsigned64 seed 和 in-flight sign-table
  替换后的独立 CPU oracle 检查通过。
- 13 Public Core ML/MLX/receipt、7 runtime host、50 screen host 测试通过。
  Public library 无 private class strings / PrivateFrameworks link。
- `git diff --check` 通过。8 repository layout / 3 release guard tests 通过。
- 完整 `make test` 已执行，未通过：先遇到旧「任何 native 文件均不得含
  私有 class」断言。本轮更新为仅允许 `private/ane_program.mm`，要求 dynamic
  loader、默认未启用及私有 source gate；稳定发行再检查 build flags、实际
  class strings 和 direct PrivateFrameworks links。真实 Public 库通过，
  Private 库被拒绝，未执行 App/CLI 替换或分发。
- 重跑 `make test` 在 `test_qwen3_hybrid_quality_gate_is_explicit_and_telemetry_is_recorded`
  失败：旧源码字符串 `if (used_hybrid_output ||` 已被既有 refactor 改为
  `if (used_hybrid ||`。两文件本轮无 diff，`git show HEAD` 也包含同样的
  源码/断言不一致，确认是 HEAD 已有失败；不改无关 Qwen3 数学，也不把
  全套测试说成通过。原始日志：
  `outputs/private-ane-a8-make-test-public-guard-20261004.log`。

Public regression library：
`2dc7049e391e5a2567df15d3293a01a66787515f324d88ead0aa6b834947937f`。
A8/share pilot private library：
`1b2104e59cf898d64f3933ba09f055d7e03b7149f72af8c37c3413cc544a2026`。
Sign-metadata private library：
`4ddb33d329c6d633acfb9035cff49d97410d101e27bc74ea5354a1910f9891e1`。

## 整请求实测

Apple M4 Max 64 GB，fox/seed42、Z 8 步，request wall 含 VAE/PNG，排除
每条 route 的冷请求，关闭 profile。各格使用同库 GPU 分母，固定 channel
split 实际执行全部 blocks，无 error fallback；不能把 GPU decline 当加速。
均为初筛，不是正反序、多提示词、内存/质量全资格。

| 库 / case | GPU hot median | Private hot median | 对 GPU 倍率 |
| --- | ---: | ---: | ---: |
| A8 库，Z1024/Fa3072，lookahead off，GPU→Private/两热 | 31.249647 s | 33.293670 s | 0.93861× |
| A8 库，Z1024/Fa3072，lookahead on，Private→GPU/两热 | 31.256500 s | 32.318452 s | 0.96714× |
| A8 库，Z512/Fa4096，GPU→Private/三热 | 6.991136 s | 6.069138 s | 1.15192× |
| A8 库，Z512/Fa4608，GPU→Private/三热 | 6.990099 s | 6.095510 s | 1.14676× |
| A8 库，Z512/Fa5120，Private→GPU/三热 | 6.997014 s | 6.388988 s | 1.09517× |
| Sign 库，Z512/Fa4096，Private→GPU/三热 | 6.988846 s | 6.049238 s | 1.15533× |
| Sign 库，Qwen512/Fa4096，GPU→Private/两热 | 41.728168 s | 36.132286 s | 1.15487× |
| Sign 库，Qwen512/Fa5120，Private→GPU/两热 | 41.719460 s | 34.414720 s | 1.21226× |
| Sign 库，Z512/distill-patch LoRA/Fa4096，Private→GPU/两热 | 8.557910 s | 7.650955 s | 1.11854× |
| Sign 库，Qwen512/Viggle r256 LoRA/Fa5120，GPU→Private/两热 | 7.946363 s | 9.263337 s | 0.85783× |

共同 c1056/chunks=1/cache=1/fence=1/W-prefetch=0。Z512 单块，不提交 A8
lookahead；不能把 share 收益归到 A8。更大的 ANE share 不必更快：Fa5120
暴露的 ANE join 成本抵消更短 GPU head，继续需要带宽校准。
以上 Fa5120 负变化仅针对 Z。Qwen512/40 步保持所有 route 同样开启 Q/K
norm-RoPE，Fa5120 比 Fa4096 更快，单向初筛首次在本轮 W8A8 cell 超过
1.2× 门槛。每请求 1280 calls、全部 channel blocks、0 fallback；并非完整
四格或该 cell 的正反序/多提示词/内存/质量资格。
LoRA 使用实际 inference-time adapter，base/adapter 未合并或修改。Z 为
8 步、1056-row v2 template，cold 出现 2 次正常 headroom retries，hot
每请求 256 calls、0 retries；Qwen 为 6 步、FP32 low-rank，所有 route 同样
开启 Q/K kernel，每请求 192 calls、0 retries。两者无错误回退，完整
correction/hidden/down ABI 回执通过；Qwen LoRA 仍负收益，不能沿用 base
的 1.212×，也不能把这些回执当整体媒体/latent 或 base/A/B/base 模型批准。

Z1024 A8 on 的累计未来 A8 submissions=2304（每请求 768），实际 ANE
calls=3072（每请求 1024）；off/on A8 wait session totals=1.452606/0.305872 s，
GPU FFN totals=35.859650/36.054494 s，post-GPU joins=14.069869/11.076174 s。
两个 policy 不同顺序的初筛约改善 3%，但仍慢于 GPU，不能当成正式 ABBA。
实际 slot bytes=84377600→88571904，多 4 MiB A8，W bank 数量未变。

单 fox 样例 A8 off/on 像素逐位相同；Z512 sign metadata 前后也逐位相同。
Z1024 Private vs GPU 单图 RMSE=5.732755（uint8）、PSNR=32.963536 dB，
max error=193；目测构图接近。这不是 latent、多提示词或整体画质批准。
Qwen512/Fa5120 单 fox 样例 vs GPU：RMSE=4.706274、PSNR=34.677259 dB、
max error=142，构图/主体目测接近，同样没有完成正式质量检查。

原始数据保留：

- `outputs/private-ane-a8-z1024-{off,on}-20261004/`
- `outputs/private-ane-share-z512-a{4096,4608,5120}-20261004/`
- `outputs/private-ane-sign-z512-a4096-20261004/`
- `outputs/private-ane-sign-qwen512-a{4096,5120}-20261004/`
- `outputs/private-ane-sign-{z512-lora-a4096,qwen512-lora-a5120}-20261004/`

逐 case 数值、summary byte hash、库身份、calls/retries 与未资格声明见
[机器可复核记录](../design/validation/private-ane-a8-lookahead-pilot-20261004.json)。

符号元数据 standalone BF16 full FFN staging 的八热样本初筛：Z 4.18679→
4.23483 ms（初始样本波动，未显示收益），Qwen 5.32665→5.05198 ms。
是组件测量，不是模型速度；Z 整请求变化也很小，不夸大成独立显著提升。

## 下一步与未完成项

本轮十份初筛均已完成，所有 bench/export/test 句柄已终止，无本轮仍在
运行的推理。仍需 Qwen1024 新库、Z1024 更大 bucket / share、Z512 GPU
channel tile / 编译边界、Qwen LoRA correction/down 成本优化、
bandwidth-aware share/prefetch calibration、连续 device trace、latent/media
与正式四格 matched 正反序 ≥1.2×。Planning JSON 的旧 FP16/token-row
标签也需与实际 W8/channel executor receipt 一致，不把历史标签当实现证明。

未修改发行 `build/native`、用户权重、参考仓库或设计稿，未 stage/commit。
