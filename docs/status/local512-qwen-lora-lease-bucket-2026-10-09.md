# Qwen 首步分工：单 bucket 与原生 LoRA 校验复用

2026-10-09，Asia/Singapore，M4 Max64GB/macOS26.6.2。接续
[延后合并与 cooperative 筛选](local512-qwen-defer-cooperative-2026-10-09.md)。
完整 Z/Qwen base/LoRA、1–2refs、encoder、GGUF/ConvRot 和整请求目标仍
active。本轮不是首步全交 ANE：仍为 FFN 中间通道 GPU/ANE 两支路，
后五步完整 compiled GPU。无下载或模型/adapter 改写，原用户草稿保留。

## 决策

双参考图改用3168行单 bucket，并复用原生 generation-bound LoRA 内容
校验证据后，名义 warm request 为12.189s，对同库 joint GPU12.686s
少约3.9%；首步3.346s，对GPU3.741s少约10.5%。所有对照 PNG 与原
对应 recipe 三个case逐字节相同。**这还是有竞争负载的诊断，不是
正式稳定倍率或默认 promotion。** 数学和 source checks 不为提速放宽。

## 先排除弱的 rank→correction surface 候选

`ane_lora_upload_kernels.hpp` 和 standalone `qwen_lora_upload.hpp`
在 Metal 中将 F32 ranks 窄化到原 B dtype，MPP B projection/F32 scale，
先保留原 delta dtype boundary，再用整数RNE转FP16，直接写
channel-major correction IOSurface。没有完整 global gate/up delta
数组、B compaction/widening、weight merge 或 ANE MIL改变。两种
矩阵 orientation、BM16/32/BN128独立测，不由理论带宽选择最快tile。

研究 harness只支持每call最多两个独立single-contributor projections，
不是任意 stacked adapter 的通用 executor。源/目标 physical backing
interval也检查，不只比较ObjC handles。source owners、surfaces和bounded
scratch由completion handler保留，timeout后禁止reuse；本轮没有
注入实GPU timeout或接入model，因此不声称完整model fallback验证。

144 actual Metal cases通过：原BF16/FP16 B、F32 ranks、rank64/256、
M1/33/145、两orientation、正负scale、三delta boundaries、physical
pitch/offset、padding。对原B consumer再转FP16的输出最大relL2为0。
另8 malformed/alias contracts拒绝；4非有限/FP16 overflow producer
返回failure，之后finite producer恢复。v1 strict build因deprecated
fastMathEnabled失败，改用SDK Safe/Precise math options；失败log保留。

原layer0 Viggle BF16 gate/up A/B、synthetic input、scale.75，每个timed
invocation真正重算两个joint A-ranks、B/scale/delta boundaries/padding/
joined surface uploads；不是prepared-rank计时。六geometries×五recipes，
3warmup/15cyclic-order hot，native库不含研究核。

| M / ANE channels / chunk | joint B→upload ms | direct candidate ms | candidate BM/orientation |
| --- | ---: | ---: | --- |
| 2096 / 7168 / 2112 | 2.718875 | 2.638166 | 32 / original |
| 3144 / 5120 / 1056 | 3.529209 | 3.699000 | 32 / original |

单图geometry只有约3%局部差异，双图三chunk还更慢，不wire模型。
每candidate scratch capacity2,162,688 bytes，对照global correction
logical outputs约21.6–90.8MB；这不是whole-process内存下降证明。
所有表值为component诊断，无continuous load资格、真实FFN/ANE/
整图或physical overlap证据。最终source/probe见v2 receipt；v1数据保留。

## 单 bucket：相同通道、相同 padded rows，少两次交接

用原 `export_runtime_ane.py` 离线导出 checkpoint-independent v2
geometry3168/4096/12288/K1024/N512、LoRA activation inputs，约612KiB。
无checkpoint payload或新dense sidecar。Public FP16 package存在且有
export hashes，**本轮不加载/性能资格化它**；Private只读template几何，
发射自己的W8 native micrograph，不能称其验证了Public compiled artifact。

双图首步3144 rows，旧1056×3和新3168×1都pad到3168；Fa5120/Fg7168
不变，chunk calls96→32/请求，32successful channel blocks仍完整。
后五步1024 rows直接GPU，不能让大bucket给decode增加padding税。
新host regression拒绝把旧96calls只改label冒充3168/32calls。

```sh
.venv/bin/python tools/coreml/export_runtime_ane.py --kind swiglu \
  --rows 3168 --hidden 4096 --width 12288 --tile-k 1024 --tile-n 512 \
  --silu-lowering exp --lora-inputs --output outputs/new-qwen-c3168-template
```

最初三个严格 `--observe-load` screen都在首arm完成后因外部CPU load
失败，summary保持incomplete，无matched GPU/hybrid倍率。后续只读
45s检查观察到同ComfyUI解释器的 `download_ltx25.py` 子进程反复启动；
不能将随后PID与每个早期busy PID强行等同，也不声称GPU正被占用。
当时ComfyUI queue/pending为0；没有向外部进程发信号或启动下载。

新增显式 `qwen_prefill_busy_diagnostic.py` 包住既有完整screen，保存
enclosing continuous CPU observation，允许先完成各臂诊断数据。
它无论load check通过与否都 `qualification_passed=false`，严格
`--observe-load` 行为不改；拒绝混用严格flag，没有把busy窗口变绿。
v4旧库完整screen：GPU→deferred→eager，一冷两热/三fresh prompts，
conditioning全miss；每臂独立process，100ms memory samples。

| old joint library / c3168 route | warm request s | 首步 / 后五步 s |
| --- | ---: | --- |
| GPU | 12.637435 | 3.738056 / 6.342185 |
| prefill deferred | 12.800201 | 3.374115 / 6.334439 |
| prefill eager | 12.995705 | 3.510277 / 6.329329 |

首步有名义下降但整请求还慢。outer load check确实失败，raw保留；
不与旧c1056不同窗口拼成一次causal bucket实验。

## 找到并移除重复整文件 hash，不移除校验

`prepare_transformer(experimental_adapter=true)` 原来每个请求都读
整个1,359,147,904-byte LoRA来append SHA256；普通GPU warm却没有
同样的全文件hash。这不在denoise/encoder/VAE计时中。v4 hybrid
warm request减这些stage的residual约.624/.633s，GPU约.10s；这是
定位线索，不把整个residual都归因给SHA或kernel。

新路线复用**已有** `SourceLease::capture_verified`：首次同held fd
读完整内容；之后只消费原生process-local、dev/ino/size/mtime/ctime
绑定的crypto proof。caller digest不能建立信任，metadata-only capture
不能作为已验内容；cache bounded256，变化/eviction重新读内容。
每次都重新capture并核对named path、canonical target与open fd，
包括mtime被恢复但ctime已改变；source content digest仍进入原identity。

新增 `Weights::apply_loras_leased` 保留旧public C++ signature/算术，
新bind用lease duplicate fd和已有 `MlxLeaseFdReader`，不重新打开
payload pathname；成功前materialize原A/B并revalidate，失败仍clear
component，不留半绑定。没有merging/requantizing adapter或改source dtype。
named alias也保留在lease中。v1 fixture发现dup共享header cursor，
第二次同fd解析失败；v2逐个reader rewind，lazy reads用pread。失败log
和当时缺thin CLI的test invocation错误都保留，不冒称通过。

新的actual MLX fixture覆盖4 stacked gate/up bindings与原路径exact，
正负strength、FP16/BF16 alpha、immutable master、materialized-source
mutation后旧输出稳定；same-size/restored-mtime新capture必须rehash，
旧lease、metadata-only proof、wrong logical ids、alias replacement
race、cancellation均拒绝，后四类in-bind failure检查component清空。
既有完整SourceLease CPU fixture也重新通过，不改变其缓存信任规则。

Profile stderr `qwen_lora_source_verification` 冷请求为完整bytes/0hit/
bound=true，两个warm为0bytes/1hit/bound=false。独立validator要求
真实三个records和上述严格状态，不接受“没输出校验”当cache hit。
这是native diagnostics加库/源码身份，不是OS read syscall trace。

## 新库真实双图：GPU分母也重测

原Qwen BF16、原Viggle v0.2.1 r256/strength1、512²六步/seed29，三
fresh prompts，joint A/B/F32 ranks、GPU encoder及同retained sources。
Fa5120/Fg7168/c3168，shared gate/up ranks，prefill-only，decode完整GPU；
无time reuse/down split/W cache/prefetch/lookahead。一冷两热，顺序
deferred→eager→GPU；不是同workload双向ABBA。新同库控制不借旧库GPU。

| leased source route | warm request s | 首步 / 后五步 s |
| --- | ---: | --- |
| matched joint GPU | 12.685788 | 3.740695 / 6.352057 |
| prefill deferred | 12.188703 | 3.346149 / 6.332084 |
| prefill eager | 12.405017 | 3.618312 / 6.329232 |

deferred名义整请求下降约3.9%、首步约10.5%；outer CPU-load check
仍失败，未建立稳定正式性能资格，也不能用before/after不同order的
差额全部归因给SHA。实际warm source diagnostics各1 native hit/0byte，
所有227bindings、32successful blocks和32calls/请求，decode calls0；
failures/fallback/retries0、headroom1，累计32/64/96，没有GPU decline。
三个memory reportscomplete、swap0，phys-footprint peak约39.7–40.4GB。

GPU、eager、deferred各三PNG与旧c3168同recipe exact，亦与以前c1056
对应recipe exact。bucket与source loader改变没有改变这些cases的图。
visual资格范围、单图后续及最终verification/hashes/cleanup见机器证据。

## 新库单图：独立选 share 与 join，不照搬双图

同库同原始模型/adapter/prompt配方、GPU encoder、单ref512，
Fa5120/Fg7168/c2112，首步2096 rows，后五步仍1024完整GPU；
顺序GPU→eager→deferred，一冷两热，outer CPU load check仍失败。
这不是对旧Fa7168单图的同窗口独立share消融。

| single-reference route | warm request s | 首步 / 后五步 s |
| --- | ---: | --- |
| matched joint GPU | 10.593985 | 2.431419 / 6.133992 |
| prefill eager | 10.249378 | 2.227858 / 6.087752 |
| prefill deferred | 10.260392 | 2.249462 / 6.092967 |

eager名义整请求下降3.25%、首步8.37%；deferred下降3.15%/7.48%。
两种join约0.11%的整图差异不足以断言稳定赢家，不能默认总选deferred。
source diagnostics同样冷full-hash/两warm 1hit且0byte；227bindings、
32calls/32successful blocks、decode0、failures/fallback/retries0。
三个memory报告complete、swap0、peak约39.06–40.25GB。
GPU三PNG与旧joint GPU exact；本轮hybrid share不同，不声称其与旧
Fa7168逐位一致。已看单/双图case1 GPU/hybrid whole和center：壶形、
把手、左右布局、颜色和暖光很接近，纹理/高光略有变化，没有明显
新棋盘格、断裂或色块。其他case/detail/seed未声称逐一验收；
automatic visual manifests仍pending，有限agent观察不是用户接受。

## 范围与接续

保留原生source proof reuse/held-fd loader；研究GPU上传核不wire模型。
不改auto GPU默认，不把大bucket推给decode，也不取消source/finite/
memory admission/late full-GPU recovery。Public actual发行guard通过。
Z/GGUF/encoder和更广model/scene/seed仍是原目标，不以本轮组件或busy
诊断替代整矩阵。下一步需要无竞争窗口的同workload正反序和更多
场景，再判断适合哪些ref/layer/share；source成本已经不再应反复支付。

完整机器记录：
[local512-qwen-lora-lease-bucket-20261009.json](../design/validation/local512-qwen-lora-lease-bucket-20261009.json)。
新Private/Public native-only库与thin CLI都独立构建，各499 source inputs；
构建仍含用户ConvRot草稿，不是clean staged-only/App分发。研究核虽
进入输入inventory，并不等于wired模型消费者。
Private8+额外actual fallback3+Z/ConvRot2，Public8+Z/ConvRot2 selected
regressions通过、无skip；SourceLease CPU fixture与26 host contracts通过。
不是全仓suite/完整Public实模型或Z模型的新性能验收。新库SHA：

```text
Private 4ee4300dbb36c5426c4d506160e09302e70f2de1bb276850234314e2064e7628
Public  8f99a7d55df8990400e5900da02bbcda6897179a1145f6bec1ebcd87e7342ce3
```

所有owned jobs终止后清理三个isolated build的637个可重建`.o`，
50,247,792 logical bytes（约47.9MiB）和三个空module-cache。保留库、
CLI、probes、全部原始/失败日志、PNG和manifest；没有清理用户模型/
缓存或终止外部进程。test TemporaryDirectory自动清理，编译对象可
按原build命令重新生成。
