# Qwen 首步逐层选路：固定前/后八层 GPU 没有超过全部并行

2026-10-09，Asia/Singapore，M4 Max64GB/macOS26.6.2。接续
[首步 bucket/LoRA 源证明复用](local512-qwen-lora-lease-bucket-2026-10-09.md)及
[启动顺序负结果](local512-channel-launch-order-2026-10-09.md)。
上一轮 [Z512 row matching](local512-z-matched-bucket-2026-10-09.md)已提交；
本轮回到 Qwen 编辑首步。GPU/ANE 仍并行分担同层 FFN，不是整步交给ANE。
只用本地原模型/adapter/ref，无下载、模型改写或dense sidecar；完整目标
保持active。新增按层机制已经接入/实测，但没有把负结果升级成默认优化。

## 为什么测按层，而不重复小融合核

现有base+B、corrected gate/up→hidden、cooperative GPU down和GPU-first
启动顺序均已有实测，尚无足够整请求盈利证据。当前更值得验证的是
哪些层需要付出 split 的输入/LoRA就绪、ANE staging/restore/join成本。
因此比较全GPU、全部32层首步FFN并行、前8层完整GPU、后8层完整GPU；
两种局部策略的其余24层仍是 GPU/ANE 通道分工。后五步一律完整GPU。
本轮没有新增GPU kernel或重用组件快数字替代完整模型结果。

## 实现与边界

新 `TURBOCIDER_QWEN21_PREFILL_GPU_FFN_BLOCKS=0,1,...` 默认unset/off，
不是0/1开关：值0表示第0层完整GPU。仅允许0..31、无重复、部分层；
canonical排序进入请求snapshot。要求explicit authorized固定正Private
W8通道、chunks1/fixed async1、`TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE=prefill`、
approximate/unconstrained resident512编辑、1–2 ref512、GPU encoder。
Public/auto、其他phase、streaming/预算、PE和all32假混合不允许；合法列表
在普通GPU控制中无效，非法语法仍拒绝。原phase gate仍排除其他时间复用。

共享 `HybridFfn::set_gpu_layers`/`plan_block` 已有安全owner/drain机制。
选中层回到原完整lazy/compiled GPU block，不是zero-channel split，不
做ANE stage、correction、probe或whole-block measurement fence。非选中
层保持原dual branch/完整LoRA与一次joined-hidden down-LoRA。decode
callback仍清空，因此KV命中步骤不继承这张表、不新增padding或桥接税。

同一canonical策略写入early/late executor identity：
`prefill-gpu-ffn-layers-v1=<list>`，已有prefix namespace也绑定该identity。
不在每步修改env，不跨策略借旧executor；unset保持旧identity。没有改
GraphGeometry/class data layout、原source/finite/shape/memory准入或晚期
完整GPU失败恢复。selection和既有requested/forced/unsplit GPU counters
分别报告策略与实际工作，不拿label当执行证明。

只读参考 `../splash` 的 `feat/ane:runtime/ane/Prefill.hpp`（be83e8f）：
显式layers与request-bound配置边界。不fetch/checkout或修改参考仓库。

`qwen_ffn_phase_screen.py --prefill-layer-screen --joint-ab` 同库比较四臂；
可选 `--defer-prefill-join` 对全部hybrid共用原owned deferred join。
validator核对实际phase rows/steps/calls、24/32成功blocks、8个完整GPU
blocks的逐请求/累积计数、Private channel axis、真实LoRA/source proof及
后续0calls。`qwen_prefill_busy_diagnostic.py --phase-layers` 记录enclosing
CPU load，始终qualification=false，原strict `--observe-load`行为不变。

## 原始六步 LoRA：单/双参考图完整请求

原BF16 Qwen Image2.1、原Viggle v0.2.1 r256、strength1、227实际bindings，
512²/6steps/seed29，三个fresh prompts，conditioning全部miss；每臂独立
process一冷两热。所有臂GPU encoder且保留同原source，joint BF16 A/B/
F32 ranks；hybrid shared gate/up ranks，Fa5120/Fg7168。关闭time reuse、
down-rank split、W-code cache、prefetch/lookahead，保持原ANE-first/fence。

单图首步2096行/c2112/eager；双图3144行/c3168/deferred。这里decode指
后续diffusion KV-hit步骤，仍计算目标图attention/FFN，不是自回归token。

| route | 单图 warm request s | 单图首步 / 后五步 s | 双图 warm request s | 双图首步 / 后五步 s |
| --- | ---: | --- | ---: | --- |
| complete GPU | 10.453406 | 2.439016 / 6.105758 | 12.669453 | 3.745619 / 6.341807 |
| 全32层首步并行 | 10.248336 | 2.250261 / 6.086365 | 12.198844 | 3.296813 / 6.348731 |
| 前8层GPU、其余并行 | 10.315923 | 2.325794 / 6.075677 | 12.312249 | 3.424172 / 6.337870 |
| 后8层GPU、其余并行 | 10.329618 | 2.320912 / 6.112638 | 12.361575 | 3.486164 / 6.336350 |

单图order GPU→all32→first8→last8；双图反序last8→first8→all32→GPU。
这是不同workload的相反顺序，不是同workload ABBA。两个enclosing CPU
load checks均失败；全部是有竞争负载的诊断，不称稳定正式倍率或设备
独占。只使用各自同窗口GPU，不借旧库分母，不拼各臂最快phase成虚构请求。

全32并行相对GPU名义首步少7.74%/11.98%，整请求少1.96%/3.71%。
first8相对all32整请求慢0.66%/0.93%，last8慢0.79%/1.33%；首步也分别
更慢约3.36%/3.86%和3.14%/5.74%。因此不能用少25%ANE calls宣称提速。
未证明稳定层级赢家，不改更快的已测all32配方或auto GPU默认。
不能将差值唯一归因为layer position、ANE wait或某个GPU kernel，轨迹/
输入、bandwidth和host spans也会改变，缺少physical device trace。

每请求all32实际32成功channel blocks/32driver calls；first8/last8为
24成功blocks/24calls加8个完整GPUblocks。三请求累计calls分别32/64/96
和24/48/72，forcedGPU8/16/24；后五步全部0ANE calls。6个hybrid process
各自cold完整SHA读1,359,147,904bytes，warm native hit1/read0/rebindfalse。
headroom1，failure/fallback/overflow retry均0，没有退化为全部GPU。

完整cold样本见机器记录；单/双图GPU cold16.044138/17.357838s，all32
15.592469/17.333362s。冷请求包含源校验/图编译/首次执行，不从单一cold
差值推断稳定冷启动加速。8个memory reports complete、swap-in/out0，
process-tree最大phys-footprint40,746,068,416bytes；scope为load+cold+
warm+exit，不归因外部service/driver或宣称无compression/带宽压力。

## 图片、回归与构建证明

单/双图各三张GPU PNG与以前对应joint GPU exact，各三张all32 PNG
亦与以前同share/bucket/join hybrid exact。新功能unset没有改变这些
cases；first8/last8不是与GPU或all32逐字节等价的声明。
已看case1每workload四张whole512：单壶/双壶布局、壶形/盖/钮/把手、
米色/蓝色、暖光和阴影很接近，釉面/高光/纹理有小变化，未见新增明显
棋盘格、断裂或色块。仅有限agent观察，不扩展为所有scene/seed/detail
或用户批准；不用严格latent等价作为本轮图像接受条件。

host31项、最新Private12项、Public11项selected regressions通过、无skip。
包含新18个compiled base/real-LoRA first/last/mixed layer cases，原24个
phase cases、changed-condition重新prefill、forced layers无stage/probe及
decode完全GPU；另覆盖joint math、leased source、Private实际late full-
GPU恢复、typed ownership/source/memory与原Z/ConvRot gates。
不是全仓或完整Public/Qwen/Z模型矩阵资格。

初始独立header测试暴露 `require`依赖声明/include顺序：需要在
diagnostic_options之前include common。三份失败host logs保留，修正后
最终通过；v1 Private是修正前build，保留但不用于表内实模型数字。
最终v2 Private/Public native-only build exit0，各501 source hashes与
当前tree独立匹配、manifest seal一致；Public actual release class/
flags/links guard exit0。构建含原用户ConvRot草稿，不冒称clean staged-
only/App发行；新提交不夹带原草稿。库SHA：

```text
Private 83c9484f9e4a6ff9a99d20e0e427c2ea3e89745dde9eb2bc66378f59f7c81768
Public  c7008bbab5d64508406dc149081311dfe9587856780ed9c79c3cb776bf81eeef
```

确认所有owned jobs terminal后，清理v1 Private/v2 Private/v2 Public三个
build的647个可重建`.o`、50,670,696 logical bytes（约48.3MiB）与三个空
module-cache；回查objects0。保留library/CLI/probe、日志/失败记录、PNG/
manifest和全部原模型/adapter/ref，没有清理用户cache或向外部进程发信号。

## 接续：不能用固定八层启发式替代盈利校准

保留安全的显式per-layer机制，后续按实际shape/layer/share/adapter/
phase测完整成本；本轮两个固定八层策略不作为更快预设。仍需要GPU
partial-down/LoRA/ANE handoff、Qwen generation/base/更广LoRA与encoder
矩阵、Public/Private operation选择、真正有界GGUF ahead-decode/fused
packed consumer以及更多scene/seed和安静窗口复测。整体目标不缩成
这个default-off机制或单次首步改善，保持active。

完整cold/warm、实际phase/GPU/ANE counters、source/load/memory、PNG
哈希、构建/失败日志与清理记录见
[机器证据](../design/validation/local512-qwen-prefill-layers-20261009.json)。
