# Qwen 编辑：首步 FFN GPU/ANE 并行与独立 decode 选路

2026-10-09，Asia/Singapore，M4 Max64GB / macOS26.6.2。本轮落实用户
明确的“首步 GPU/ANE 分工并行，后续按最快路线”方案，不是把首步
全部交给 ANE。只读本地 Qwen、Viggle adapter 和参考图，无下载、模型
改写或 dense sidecar。完整 Z/Qwen base/LoRA、1–2参考图、encoder、
GGUF/ConvRot 和整请求加速目标仍未完成。

## 实现：两阶段不再强制共用一套 FFN 回调

`TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE=all|prefill|decode`，默认 `all`
保持既有路线。`prefill` 只在尚无 prefix KV 时启用原中间通道分工；
`decode` 只在实际 prefix 复用时启用。未选中的阶段直接使用原完整
lazy/compiled GPU block，不是 zero-channel split，不 staging、不作
split GPU probe、不增加该路径的 whole-block measurement fence。

选择在请求 owner 开始时 snapshot，分别绑定既有 prefill/decode hooks；
不在每步修改 process environment。实际 prefix state 而非 step index
决定执行阶段。非默认策略进入 early/late executor identity，已有
prefix key 也包含 runtime identity；默认 identity 不变。不同策略的
callbacks 不跨请求留存，退出仍 drain。真实晚期失败仍完整 GPU 重算。

非默认策略目前要求 approximate resident512²、1–2 ref512、明确
Private 固定非零通道、GPU encoder；base 和原 inference-time LoRA
均可用。Public/auto/constrained/streaming/组合 encoder 与时序复用
不在本次资格范围；有效策略在普通 GPU control 忽略，非法值仍拒绝。
没有把 Private/Public capability fallback 视为自动盈利调度。

回执 `qwen_ffn_phases` 按每步 finite/eval 成功后汇总：实际阶段的
steps/rows、host step seconds、request-local driver calls 与成功
channel blocks。累计计数逐请求独立对照，非物理 kernel/INT8 MAC/
overlap 证明。step profiler 也修正为实际 prefix state，不能把
跨请求 KV hit 的 step0 误标成 full prefill。

## 真实六步 LoRA：四路线、单/双参考图

原 BF16 DiT、Viggle v0.2.1 r256、strength1、seed29、512²六步。
同库、每臂独立进程、一冷两热，三个不同 prompt 全部 conditioning
miss。所有臂同 encoder source retention，encoder 都是 GPU；原
F32 A ranks、相同新 B epilogue，混合臂共享 gate/up ranks。关闭
down split、W-code cache、prefetch/lookahead 和时序 FFN reuse。
100ms 进程树 memory sampling；未申请 strict competing-load 资格。

Fa5120/Fg7168，bucket1056。单图首步2096行、双图3144行，后续均1024。

| route | 单图 fresh warm request s | 单图首步 / 后五步 s | 双图 fresh warm request s | 双图首步 / 后五步 s |
| --- | ---: | --- | ---: | --- |
| complete GPU | 10.809493 | 2.523486 / 6.272520 | 12.939887 | 3.834177 / 6.499943 |
| prefill parallel / decode GPU | 11.578446 | 2.823679 / 6.239586 | 14.209047 | 4.481483 / 6.549570 |
| prefill GPU / decode parallel | 11.293939 | 2.521055 / 6.168204 | 13.795490 | 3.835628 / 6.797590 |
| both phases parallel | 11.471918 | 2.844375 / 6.134867 | 14.285416 | 4.426206 / 6.692508 |

单图顺序 GPU→prefill→decode→all；双图 all→decode→prefill→GPU。
这些是不同 workload 的相反顺序，不冒称同 workload ABBA/reverse
重复。保留每个 cold/warm 样本，不将不同臂的 phase 时间拼成一条
不存在的最快路线。切换 precision 后后续输入也不同，不能从这些
host spans 推断唯一 slowdown 原因。

当前 Fa5120 的首步并行确实执行，但比 GPU 慢约0.30/0.65s。
单图每请求 phase calls：prefill-only64/0、decode-only0/160、all64/160；
双图96/0、0/160、96/160。成功 channel blocks 分别32/0、0/160、32/160。
每请求227真实 adapter bindings；共享 ranks 只计选中阶段成功工作。
failure/fallback/overflow retry0、headroom1，绝非能力失败后纯 GPU。

八个 memory reports complete、system swap-in/out0；phys-footprint
peak约38.0–40.7GB，scope为进程树 load+cold+warm+exit，不归因外部
服务/driver。零swap不等于无compression/bandwidth压力。

同输入三张 GPU PNG 全部与前轮 B-on GPU exact；all-phase 三张也与
前轮 B-on hybrid exact，两种参考图 workload 都成立。新phase控制/
telemetry没有改变这些默认计算结果，不外推为全仓/全精度逐位等价。

目录 `outputs/local512-qwen-edit{1,2}-ffn-phase-v1-diagnostic-20261009/`。

## 较大 bucket：减少调用不等于盈利

另起同库单图诊断，已有 bucket2112，Fa7168/Fg5120，顺序 prefill→GPU；
其余同真实六步 LoRA。首步 calls 从64减到32，后五步仍0；成功
blocks32，zero failure/fallback/retry、headroom1。

| matched route | fresh warm request s | 首步 / 后五步 s |
| --- | ---: | --- |
| GPU | 10.893506 | 2.549141 / 6.281857 |
| prefill parallel | 11.523818 | 2.774279 / 6.234098 |

仍未超过 GPU。此处 bucket 和 share 一起变更，不能将相对上一窗口
的差值全归因于减少调用；不能跨窗口借较快 GPU 分母。两个 memory
报告complete、swap0。没有新增/改写模板，使用已有 checkpoint-free
runtime template，实际 Private W8 executor 仍按 admitted geometry运行。

目录 `outputs/local512-qwen-edit1-prefill-c2112-a7168-v1-diagnostic-20261009/`。

## Base：首步有收益，40步整请求收益很小

同库、原 BF16 base、单ref512、40步、seed29、三个fresh prompts，
GPU→prefill。无 adapter/B实验，bucket2112、Fa7168，后39步完整 GPU。

| route | fresh warm request s | 首步 / 后39步 s |
| --- | ---: | --- |
| GPU | 46.869651 | 2.195966 / 42.774322 |
| prefill parallel | 46.715864 | 2.006951 / 42.773365 |

首步约1.094×，耗时下降8.61%；整请求仅减少约0.154s/0.328%，不足
以称稳定显著加速。每请求32实际calls/32成功channel blocks，后续0；
failure/fallback/retry0、headroom1。两个memory报告complete、swap0，
peak约37.52/37.96GB。同时间窗比较，不用 LoRA GPU 作 base 分母。
base/LoRA配方及schedule不同，此结果不证明某个LoRA步骤是唯一慢因。

目录 `outputs/local512-qwen-edit1-base-prefill-c2112-a7168-v1-diagnostic-20261009/`。

## 视觉与决策

已检查 case1 的单/双图 c1056、单图 LoRA c2112、base c2112 原像素
whole和center；双图还检查top-left/bottom-right。主体壶形、盖/钮、
把手、双壶布局与阴影很接近，釉面、高光/纹理有小差异，未见新增
明显块纹、断裂或色块。其他case/detail未冒称逐一验收；这些是有限
agent观察，不是多scene/seed或用户批准，automatic visual manifests
保持pending，`qualification_passed=false`。允许近似，不以旧严格
latent N1作为本轮新图自动验收，也不改写旧失败记录。

本轮证明方案可行且base首步能局部盈利；没有证明LoRA首步/整图
盈利，因此保持显式实验、不改变默认。当前六步 GPU 首步仅占请求
约23–30%，不能借历史三张1024px参考图的74%去噪占比预测本轮收益。
40步base首步占比更小，后续再快的KV命中仍要计算目标图attention/FFN。

接续优先实际首步 LoRA correction readiness、GPU partial-down 和
staging/restore/join 成本，测试更匹配的share/bucket或按层选择；
同时独立保留/校准最快 GPU decode。不盲加大offload、不重新开启
已有明显块纹的时间复用，也不把单次首步改善代替全请求目标。

构建、回归、库身份和清理的最终记录见
[机器证据](../design/validation/local512-qwen-prefill-parallel-20261009.json)。
本轮构建包含原ConvRot working-tree草稿，提交只选择本轮owned
hunks，不冒称clean staged-only/App发行或完整模型矩阵验收。

最终选定Private41项、Public50项回归通过，无skip；含24个compiled
base/真实LoRA phase cases、432个新B Metal cases、108个B runtime
cases、原projection/shared/down、ConvRot LoRA/MPP、actual Private
channel晚期失败/完整GPU恢复、实际Public encoder与CoreML wrapper。
另host screen18项与独立parser1项通过，重复运行不重复计数。
两份native-only build exit0、相邻thin CLI完成，Public实际发行guard
通过；各495 source input hashes独立匹配。

```text
Private cb4a5dcd236fdb825bafc52915c4a2ac29f0fec5577fb1df714127a047424520
Public  db1870ed0d8721ccb3afc4075a6f5a4e0de21ca2fcd701a99218664b6f2d7a51
```

所有owned jobs确认terminal后，清理B v1 Private、phase v1
Private/Public三个build的639个可重建`.o`，50,215,064 logical bytes
（约47.9MiB），以及三个空module-cache。库/CLI/probes、日志、PNG、
manifest与所有原模型/adapter保留；没有清理用户缓存或外部进程。
