# Qwen 首步 FFN 并行：GPU 提前提交没有盈利

2026-10-09，Asia/Singapore，M4 Max64GB/macOS26.6.2。接续
[首步 bucket 与原生 LoRA 校验复用](local512-qwen-lora-lease-bucket-2026-10-09.md)。
完整 Z/Qwen base/LoRA、生成、1–2参考图编辑、encoder、GGUF/ConvRot
加速目标仍 active。本轮不是整步交给 ANE，也没有模型下载或改写。

## 判断与实现

先检查现有双图 c3168/deferred 实模型记录：三请求累计
`stage_wait_seconds_session_total=.000087046`。单纯把 GPU 提交提前到
`wait_stage` 前没有足够收益空间，不将总 staging 时间误当暴露等待。

只读参考 `../splash` 的 `feat/ane`，commit
`be83e8f895bc8d9036ace8348c7ddc2098dc5ebc`，`runtime/ane/ForkJoin.hpp`
和 `Prefill.hpp`：可变 branch launch order、waitable ownership、首个
异常保留且两支路都 drain。没有 checkout/fetch 或修改参考仓库。
同时重读本仓 `docs/design/parallel-acceleration.md` 与近期首步记录。

新增默认关闭的 `TURBOCIDER_RUNTIME_ANE_CHANNEL_GPU_FIRST=1`。
共同 FFN input、gate/up LoRA correction、weight stage 都就绪后，先
`mx::async_eval` GPU channel complement，再 launch ANE worker，让
A8/第一条 ready producer 有机会与 GPU complement 重叠。并非更早
计算尚无输入的 FFN，也没有独立流/优先级、A8 kernel 或权重改写。
原来的 ANE-first 顺序仍是默认。

只允许明确授权的 Private 固定正通道、fixed async、profile关闭及
post-GPU prefetch policy；Public/auto/row、不合法 flag 与同步模式拒绝。
配置在 executor 构造时 snapshot；unset/0 不改旧 identity，启用时新增
namespace，防止 retained executor/calibration/prefix KV 跨策略复用。
成功计数只在两支路完成且 join/down-LoRA 构造成功后增加；失败不计。
JSON 单列 enabled 与实际累计 successful blocks，不据 counters 声称
物理 overlap 或原生 INT8 MAC。异常时两支路都 drain，保留最初异常。

gate/up correction 仍在 SiLU 前；GPU 与 ANE 各计算不相交通道的
gate/up/hidden/base-down partial，最后相加并应用一次完整 down-LoRA。
与原路线算术/源精度相同，不是新增数值近似或漏掉 ANE-side adapter。
这里的 decode 是后续 diffusion KV-hit 步骤，不是自回归 token decode。

## 完整请求对照

原本地 Qwen BF16、原 Viggle v0.2.1 r256/strength1、512²六步/seed29，
GPU encoder、同一原 encoder source retention、joint BF16 A/B/F32 ranks，
shared gate/up ranks，Fa5120/Fg7168。仅真实 prefill 走 hybrid，后五步
完整 compiled GPU。无 time reuse/down-rank split/W-code cache/prefetch/
A8 lookahead。Private launch fence=1；两种 hybrid 仅改变提交顺序。

每臂独立进程、一冷两热、三个不同 prompt 全部 conditioning miss。
单图 c2112、prefill2096rows/eager join：GPU→ANE-first→GPU-first；
双图 c3168、prefill3144rows/deferred join：GPU-first→ANE-first→GPU。
不同 workload 的反向顺序不是同 workload ABBA。每个 trial 都有100ms
完整 process-tree memory observation，额外有 enclosing CPU-load observation。
两个 load checks 均因竞争 CPU load 失败，保留原始证据，全部
`qualification_passed=false`；没有向外部进程发信号。

| refs / route | fresh warm request s | 首步 s | 后五步 s |
| --- | ---: | ---: | ---: |
| 1 / complete GPU | 10.542797 | 2.427685 | 6.126892 |
| 1 / ANE-first | 10.334239 | 2.264393 | 6.112969 |
| 1 / GPU-first | 10.711538 | 2.702845 | 6.083640 |
| 2 / complete GPU | 12.743863 | 3.747744 | 6.413626 |
| 2 / ANE-first | 12.236150 | 3.345239 | 6.325007 |
| 2 / GPU-first | 12.580750 | 3.721710 | 6.349953 |

同窗口，GPU-first 相对 ANE-first 首步慢19.36%/11.25%，整请求
慢3.65%/2.82%。原 ANE-first 相对 GPU 名义首步少6.73%/10.74%，
整请求少1.98%/3.98%；不与旧库更好的数字混用，也不称稳定倍率。

两个 hot 请求的 A8 worker wait 中位数（累计 counters 作差）：

| refs | ANE-first s/request | GPU-first s/request |
| --- | ---: | ---: |
| 1 | .012039 | .527579 |
| 2 | .017055 | .796183 |

GPU-first 的 async ANE wait 反而缩短，不能挑这个 host span 宣称提速。
A8 等待增加与 GPU complement 延迟 ANE ready 的解释一致；但没有
device trace/queue priority/spill证据，不能精确归因或把嵌套跨度相加。

四个 hybrid process 每请求227实际 adapter bindings、32成功 channel
blocks/32driver calls，后续 calls0；累计32/64/96，GPU-first 的实际
successful order blocks 同步增加，control 为0。failure/fallback/overflow
retry0、headroom1。四个源证明 validator 都确认 cold full SHA 读取
1,359,147,904 bytes，warm native hit1/read0/rebindfalse；没有删掉校验。

## 图片、决策与未完成项

每种参考图下，ANE-first 与 GPU-first 三个 PNG 逐字节相同；本轮
启动顺序不改变这六张 hybrid 图片。已看单/双图 case1 的 GPU 与
GPU-first 全图：壶形、把手、布局、颜色、暖光和阴影非常接近，纹理/
釉面/高光略有变化，未见明显新增棋盘格、断裂或色块。不是所有
case/detail/seed 或用户接受的证明，不以严格 latent 等价作图像门槛。

保留有完整安全检查的 default-off launch-order 研究开关和测试，不
推广 GPU-first、不改变 auto GPU 默认或已有最快 ANE-first 配方。
这轮排除一个具体启动顺序候选，不代表否定 FFN GPU/ANE 并行。
接续仍应优化 correction readiness、GPU partial-down/融合、实际 row/
layer/share 选择及 encoder；Public model-operation 性能、base/generation、
更广 scene/seed 和真正可复用 GGUF ahead-decode 仍未完成。

构建、回归、memory 峰值、原始/失败证据、库身份与清理的最终记录见
[机器证据](../design/validation/local512-channel-launch-order-20261009.json)。

Private/Public 两份 native-only build exit0，各499 source inputs重新核对
无 mismatch；构建包含原用户 ConvRot working-tree 草稿，不冒称 clean
staged-only 或 App发行。Private10项、Public9项 selected native regression、
25项 host contracts通过、无skip；包括新顺序的actual channel eager/
deferred、晚期完整GPU恢复、两顺序异常清理与typed lifetime，以及原
compiled Qwen phase24、BF16 ConvRot partial18、leased source/原joint
A/B数学。Public actual release binary guard通过，不是全仓suite或
Public实模型加速资格。

六份 memory报告complete，swap-in/out0，process-tree phys-footprint
峰值约39.06–40.65GB（十进制）；不归因 external service/driver/wired。
所有本轮owned jobs终止后，清理两份isolated build的429个可重建`.o`，
33,318,960 logical bytes（约31.8MiB）与两个空module-cache。保留库、
CLI、probes、全部logs/PNGs/raw observations/manifests，无模型/adapter/
reference或用户cache删除；可按原build命令重新生成objects。
