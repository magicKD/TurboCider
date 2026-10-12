# Qwen channel 融合核与 FP16 rank 对照：不提升默认

2026-10-09，Asia/Singapore，M4 Max64GB/macOS26.6.2。接续
[联合A/B](local512-qwen-joint-ab-2026-10-09.md)。本轮实际实现并测试
base+B、corrected gate/up+hidden两种Metal原型，再对当前首步并行/
后续GPU路线比较已有FP16 ranks与joint BF16/F32 ranks。没有下载、
模型/adapter改写或dense sidecar；完整Z/Qwen/base/LoRA/1–2refs/
encoder/GGUF/ConvRot加速目标仍未完成。

## 两matmul base+B 原型

`dense_gpu_base_lora.hpp` 一个dispatch执行原physical dense base
projection与低秩B projection。base的F32 accumulation先写成原dtype
小TG tile，再与F32 scaled B相加，最后一次输出cast；rank narrowing
沿用已测B recipe。没有global base/delta buffer、W compaction或merge。
它不是默认Weights consumer，也没有接入模型。

72个实际Metal case对原MPP base+B epilogue按bytes一致，最大差0：
FP16/BF16、M1/33/67、physical base row/column与B row offset、tails、
六tiles与正/负scale。另6个非法shape/dtype/range/finite/tile contract
拒绝。不是对未量化原始F32 LoRA的严格等价证明。

实layer0 BF16 gate/up base及原Viggle A/B，合成输入，准备好的joint
F32 ranks：M1024/2096/3144、GPU channels5120/7168、gate/up两物理
区间，共12×8 recipes，各3warmup/15循环换序hot。计时包括base/B/
cast/eval，排除A投影、down、ANE、读盘和完整FFN。

| M/N，gate区间 | compiled MLX base+B ms | compiled MPP base+B ms | fused32×128 ms |
| --- | ---: | ---: | ---: |
| 1024/5120 | 3.304250 | 3.238334 | 3.270125 |
| 2096/5120 | 6.598291 | 6.446375 | 6.454750 |
| 3144/7168 | 13.492084 | 13.284250 | 13.252125 |

对MLX control局部差异约1–2%，对更快MPP control大多无明确盈利；
BM64/BN128更慢。输出relL2最大仅约1.24e-10，不以接近逐位一致为
目标换取速度。**没有足够收益依据接入模型或改变默认。**源码的TG
tile不证明无spill/occupancy或唯一慢因。

receipt `outputs/local512-qwen-base-lora-v1-component-20261009.json`。

## 四matmul corrected gate/up→SwiGLU hidden 原型

`dense_gpu_lora_hidden.hpp` 保持原base与corrected gate/up dtype
boundaries，用两小TG tiles执行gate-base/gate-B/up-base/up-B与激活，
只输出完整corrected hidden，不发布两份global gate/up。支持独立
ranks/scales/physical B offsets，两个sigmoid lowering单独screen。

48实际Metal cases：FP16/BF16、M1/33/67、四tiles、两激活recipe、
不同gate/up ranks及正负scale。对compiled原SiLU/multiply组件最大
relL2=.00348641（约.349%），低于5%component预算；另5个非法contract
拒绝。没有删除source finite/range或原alpha，更不是整图自动验收。

实layer0相同original sources、prepared joint ranks，六M/N×10 recipes，
各3warmup/15循环换序hot。对照包含完整corrected gate/up与compiled
SiLU/multiply，排除A/down/ANE/整模型。取32×128、rounded sigmoid：

| M/N | MLX base+B+hidden ms | MPP base+B+hidden ms | fused hidden ms |
| --- | ---: | ---: | ---: |
| 1024/5120 | 6.358625 | 6.265584 | 6.291709 |
| 2096/7168 | 17.789750 | 17.756084 | 17.740125 |
| 3144/7168 | 26.810875 | 26.529459 | 26.346208 |

实际hidden relL2约.00347（rounded）/.00375（不round sigmoid）。仍
只有约0–2%局部差异，尚无whole-model/LoRA/visual盈利，不wire入默认
FFN。不能从省global buffers推断全请求更快或memory peak下降。

receipt `outputs/local512-qwen-lora-hidden-v1-component-20261009.json`。
两个probes均binary unchanged/observer errors0，最大gap约.273/.266s。
standalone absolute rpath链接保留joint-v1 Private库，无adjacent dylib；
不是loaded-image、physical kernel/INT8 MAC/GPU–ANE overlap证明。

## 当前真实模型精度选择：joint 仍快于已有 FP16 ranks

新增 `qwen_encoder_residency_screen.py --lora-precision-screen`，使用
**已有**两种完整native profiles：joint BF16 A/B/F32 ranks 与
`TURBOCIDER_QWEN21_VIGGLE_LORA_FP16=1`。无新native flag或adapter
重写，FP16仍保留base BF16/scale及全部LoRA，不漏ANE correction。
validator要求真实227 bindings、不同selection、实际phase与累积calls，
不能仅根据env把joint/F32 relabel为FP16。

原BF16 Qwen、原Viggle v0.2.1 r256、strength1、512²六步/seed29、三
fresh prompts，conditioning全miss。同retained encoder sources、GPU
encoder，hybrid only prefill、后五步完整GPU，shared gate/up ranks；
关闭time reuse/down split/cache/prefetch/lookahead。各独立process一冷
两热、100ms memory sampling，无strict competing-load资格。

| route | 单图 request s | 单图首步/后五步 s | 双图 request s | 双图首步/后五步 s |
| --- | ---: | --- | ---: | --- |
| GPU joint | 10.608320 | 2.428076/6.163826 | 12.779422 | 3.746173/6.394768 |
| GPU FP16 ranks | 10.818703 | 2.535059/6.269935 | 12.902929 | 3.862273/6.447616 |
| prefill-parallel joint | 11.266884 | 2.698380/6.094274 | 13.753474 | 4.244166/6.373998 |
| prefill-parallel FP16 ranks | 11.497676 | 2.735872/6.227483 | 14.166269 | 4.504185/6.501823 |

单图FA7168/FG5120/bucket2112、双图FA5120/FG7168/bucket1056；不同
workload/shape不合并分母。单图GPU joint→FP16→hybrid joint→FP16；
双图相反order，但不是同workload reverse/ABBA或统计稳定保证。
这些结果偏向joint，不将FP16列为新的最快预设；hybrid仍慢于匹配GPU。
此前三参考图/旧Public row实现的FP16收益不能套给本次Private prefill。

单/双图各hybrid每请求32成功channel blocks、32 shared rank sets/
64 arrays；32/96 actual driver calls，decode calls0、headroom1、
failure/fallback/retry0。全部227 adapter bindings，无GPU-only decline。
8个memory reports complete、system swap-in/out0，peak约38–41GB，
scope为load+cold+warm+exit进程树，不归因外部services/driver。

两workload的joint GPU三张PNG与上一轮joint GPU均exact；hybrid也
与上一轮相同recipe三张exact。FP16与joint不是bytes equivalent。
已看case1的单/双图GPU和hybrid精度对照whole：主体、壶体/把手、
左右布局和光照很接近，纹理/高光略变，未见明显新块纹/断裂/色块。
其余case/crops未冒称验收；automatic manifests保持pending，agent
有限观察不是多seed或用户批准，`qualification_passed=false`。
工具固定GPU reference/ANE candidate标题不是physical backend事实，
GPU组两边都是GPU，hybrid组两边都是Private，以source manifest为准。

目录 `outputs/local512-qwen-edit{1,2}-fp16-vs-joint-v1-diagnostic-20261009/`。

## 验证、磁盘与接续

本轮没有重建native library或改其默认pipeline；实模型使用保留
joint-v1 Private SHA `1862a3662020839ed6fad6161231b6e750e3427f5024017d402b44c1cc35b86a`。
新增两个research headers不是该库的输入；不能冒称native manifest
涵盖当前所有源文件。Public测试用保留joint-v1 Public库，其actual
release guard再次通过，不冒称这些standalone核已经随App发布。

研究核实际Metal+selected native regression、host contracts、库身份与
清理范围见 [机器证据](../design/validation/local512-qwen-fusion-precision-20261009.json)。
Private/Public retained库各9项选定回归通过、无skip；final Private又
独立复跑两研究核的6/5非法contract拒绝，重复run不重复算unique cases。
20项host contracts通过。没有新native build对象或owned probe `.o`，
测试temporary directories均自动清理，不声称清理用户/外部缓存。
没有新全模型dense副本或大型activation dump；临时测试目录自动释放，
保留probes/旧库/CLI、全部raw samples/logs/PNG/manifest和原模型。
原ConvRot drafts仍保留，提交只包括owned research/tool/test/docs。

下一步不再仅减少小epilogue dispatch：更值得独立测dense partial-down/
gate/up的cooperative tensor布局与K-blocked kernel、真实LoRA readiness/
ANE handoff以及按layer/phase的完整成本选择。只有实际frame/整请求
收益才wire新核，不用本轮组件或精度选择证明完整目标完成。Z/Qwen
base/LoRA/encoder、Public/Private杂糅、GGUF真实复用、多scene/seed与
有效strict窗口仍在原目标范围内。
