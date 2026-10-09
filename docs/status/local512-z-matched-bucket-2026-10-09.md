# Z512：请求匹配 Private FFN bucket，消除长 caption 调用台阶

2026-10-09，Asia/Singapore，M4 Max64GB / macOS26.6.2。接续
[BF16 ConvRot partial](local512-convrot-bf16-partial-2026-10-09.md)与
[Qwen 首步分工](local512-qwen-lora-lease-bucket-2026-10-09.md)。
本轮实际改进 Z-Image BF16、ConvRot 与 GGUF 的 GPU/ANE FFN 通道分工，
没有把整个步骤交给 ANE，也不改变 Qwen 的 prefill/decode 或 encoder
选择。只使用原本地模型/adapter；无下载、原始文件改写或 dense sidecar。
完整目标仍 active，下面均为诊断而非正式性能/产品 promotion。

## 实现：足够的桶保持，越界才扩展

新 `TURBOCIDER_Z_RUNTIME_MATCH_ROWS=1` 默认关闭。真实 conditioning
完成后 snapshot image/padded-caption rows，再选择 Private native 行数：

- 原模板 aligned32 且已装得下实际总行数：保持原桶。
- 否则将实际总行数向上取整到128行网格；最多8192。
- 512² image 为1024行；caption 正数/aligned32，最多7168行。

例如原模板1056：31个text tokens对应caption32，总1056，仍用1056；
37个text tokens对应caption64，总1088，改用1152。没有裁剪caption、
减少steps、改变通道share或偷偷使用zero-channel GPU路线。

共享 factory/HybridFfn 新增独立 overload，保留旧 signature、GraphGeometry
及 class data layout。覆盖原模板的 rows 之前，先按完整 logical FFN
geometry/LoRA ABI 检查 checkpoint-independent template；仅 Private
native micrograph消费 override，不重写/重标 Public compiled artifact。
仅允许 explicit authorized Private SwiGLU、固定正通道、无 calibration，
override必须32..8192/aligned32；原self-test、memory admission、source/
finite/shape检查与晚期完整GPU重算不变。

Z planning/run 使用同一 scope gate：明确 approximate、unconstrained
resident512 generation、W8A8或ConvRot W8A8、chunks1/fixed async1；
排除streaming、显式预算、encoder ANE和auto渠道。合法flag在普通GPU
control上无效，非法值仍拒绝。executor identity 增加
`matched-native-rows-v2-keep-fit=<rows>`，不同几何不能共用旧程序。
prepare-only和实际结果均在最终selection替换之后追加实际rows marker。
这是有界shape政策，不是自动盈利校准，也不保证所有长caption都获益。

只读参考 `../splash` 的 `feat/ane:runtime/ane/Prefill.hpp`：借鉴按实际
工作负载冻结配置/identity的边界，不移植其语言模型假设，不fetch/checkout。

## 同最新 Private 库：完整请求，而非单个 GEMM

每臂独立进程、一冷两热、同prompt；100ms process-tree memory采样。
没有strict continuous-load或physical overlap资格，不跨库借GPU分母。
ConvRot hybrid均Fa4096/Fg6144、BF16-rounded packed GPU partial/F32 join；
LoRA为原distill patch、strength1、238实际projections，敏感ordinal2
完整GPU。ConvRot base的GPU control使用working-tree原有compiled recipe；
LoRA GPU control为完整packed GPU。不同source/recipe不能跨行排名。

| workload | GPU warm s | fixed hybrid s | matched hybrid s | matched耗时下降 / 自己GPU |
| --- | ---: | ---: | ---: | ---: |
| ConvRot base fox / 9steps / seed42 | 9.613737 | 7.756663 | 7.745051 | 19.44% |
| ConvRot LoRA fox / 8steps / seed42 | 9.515012 | 8.095458 | 8.110186 | 14.76% |
| ConvRot LoRA lighthouse / 8steps / seed17 | 9.795147 | 11.008220 | 8.387358 | 14.37% |
| 原BF16 base lighthouse / 9steps / seed17 | 8.211045 | — | 6.674280 | 18.72% |
| 原BF16 LoRA lighthouse / 8steps / seed17 | 8.668830 | — | 7.320284 | 15.56% |
| GGUF Q4 LoRA lighthouse / 8steps / seed17 | 10.756004 | — | 9.457878 | 12.07% |

短fox保持1056，没有padding到1152；ConvRot LoRA fixed/matched对应
三个PNG逐字节相同，约0.18%wall差异不视为稳定退化或收益。
长lighthouse改1152：ConvRot实际calls从480降至248/request，matched
比同窗口fixed耗时少23.81%。两个noise-refiners仍仅1024行，只有主层
的caption容量越界会增加chunk，不能简单声称所有FFN调用都减半。

base9的两种source各有2次cold overflow retry，warm0；累计calls
290/578/866，对应288blocks/request。明确允许 `--cold-retry-cap 2` 的
诊断，不列为zero-retry正式资格。原BF16 base cold GPU10.890064s、
runtime11.070530s，仍更慢。LoRA cases retry/failure/fallback均0，
calls248/request；所有近似结果仍有限、完整生成。

GGUF使用原Q4_0、CPU-direct affine packed import、retained packed bank、
1GiB allocator hint、原adapter238、GPU ordinals0,1,2、Private W8A8/
Fa4096/FP16 join；没有在FP16 GGUF开启BF16 F32-join实验。
matched实际232calls/request，累计232/464/696。cold源读4,509,395,072
bytes，两个warm源读0、reused packed=true，dense_scope=none。
cold GPU13.266749s/runtime13.543412s，冷启动依然未受益。
另一独立fixed窗口runtime12.118983s、自己的GPU10.753628s，464calls/
request；不将两个窗口拼成一次因果实验或正式倍率。

## 保留失败、状态切换与视觉差异

v1实际成功执行1152，但后面的旧selection replacement擦掉新marker；
validator正确中止，`lighthouse-v1` summary保持incomplete。v2修复marker，
最初所有caption都上取整128，短fox fixed8.121184s/matched8.180462s，
候选慢约0.73%；因此最终v3采用keep-fit、只在cliff时扩展。
旧窗口与错误全部保留，不混入上表最新v3结果。

真实同一process/原LoRA跑 fox→lighthouse→lighthouse→fox：实际bucket
1056/1152/1152/1056，executor累计calls248/248/496/248。重复几何复用，
变化几何重建；两次fox PNG exact、两次lighthouse PNG exact。新增host
validator检查实际model/dimensions/token/seed、Private channel axis、
LoRA、counter/reset/retries/fallback和四张SHA；不能只改bucket label。
最终validator也在保留的实际四请求stdout上replay通过。

已看case1 whole512：ConvRot base fox姿态/脸/尾巴/雪很接近，毛发纹理
稍有变化。ConvRot LoRA灯塔结构/光照/石材接近，**GPU红屋顶变成候选
深色/黑屋顶**，这是需要保留的可见近似差异，不能说所有细节一致。
不能把该差异独归因于bucket；GPU与候选本身也使用不同W8/A8配方。
原BF16 LoRA与GGUF Q4的各自GPU/runtime灯塔图，主体几何、屋顶颜色、
暖光和前景岩石都很接近，只有少量纹理/细节变化。
此前原BF16 base检查也接近；以上仅有限agent观察，不是所有scene/
seed/detail或用户批准，不改写旧严格latent失败/automatic visual资格。

7个完整比较窗口加状态切换，共18个memory reports complete、
system swap-in/out均0；最大process-tree phys-footprint25,971,693,784
bytes出现在状态切换。scope为load+cold+warm+exit，不冒称整机RAM上限、
没有compression压力、device driver归因或物理GPU/ANE重叠。

## GGUF提前解码与完整目标仍待推进

本轮实际优化的是原packed GPU/ANE消费者的shape/call工作量，**没有
实现新的异步提前dense解码**。已有 `AffineDenseWindow` 是同步研究、
两个矩阵槽，不是两个完整FFN槽；旧512-ish GGUF组件需要约23–46次
真实同矩阵复用才能摊销decode，而32层轮转的两槽只有一次消费。
因此不把没有复用/重叠证据的dense cache或磁盘sidecar接入默认模型。
原始依据见 [寄存器核与窗口](convrot-register-dense-window-2026-10-06.md)。

接续仍包括真正有界ahead-decode/fused packed GPU consumer、按shape/
layer/adapter/phase选择盈利share/operation，Private/Public区别与实际
成本，Qwen generation/base/LoRA/1–2ref编辑和encoder，以及更多场景/
seed和安静窗口对照。Qwen首步并行后续GPU的已测结果不被本轮Z替代。
新flag保持default-off，完整目标不缩成一个组件/几张图通过。

## 验证、构建身份与清理

最新host80项通过、无skip；包含四项新switch evidence contracts。
最新Private选定11项、Public10项回归通过、无skip。覆盖实际64/96-row
Private程序/late完整GPU恢复、原BF16 partial18、Qwen compiled phase24、
joint LoRA24、leased adapter4bindings/6rejections及runtime/source/memory/
typed ownership/fallback。首次额外Private invocation漏开runtime测试flag，
跳过2项；日志保留，正确flag后全9项重跑通过，重复不重复计数。
早期host invocation误写不存在的GGUF测试模块，日志也保留；不计通过，
最终正确模块组合的80项结果才计入本轮验证。
Public实际release class/flags/link guard exit0。

两份v3 native-only库各500 source hashes独立与当前tree匹配、manifest
seal一致，thin CLI保留。构建包含原用户ConvRot草稿，不是clean
staged-only/App发行或完整全仓/全模型矩阵验收。只选择本轮owned hunks
提交，原草稿不夹带。库SHA256：

```text
Private 7ac38bb2af3fe88f36b20974c22c656a3b0e01e7b23c1cca4526f18a2176a296
Public  e67f74866200132245ef3a93c829319262a43b5e5b2699ce55c432db2824ea52
```

确认所有owned jobs terminal后，只清理v1/v2/v3 Private和v3 Public
四个build的865个可重建`.o`、67,782,072 logical bytes（约64.6MiB）及
四个空module-cache；回查objects0。保留全部library/CLI/probe、日志、
PNG、manifest与失败记录。对象可按原build命令重建，没有删除原模型、
adapter、参考图、用户缓存或终止外部process。

命令与身份以各runner/raw stdout和build日志为准；完整逐臂cold/warm、
calls/retries、source/retention、memory、validator与清理记录见
[机器证据](../design/validation/local512-z-matched-bucket-20261009.json)。
