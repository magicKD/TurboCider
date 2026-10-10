# ConvRot/GGUF：共享 packed-word 解码，F32 partial 接近快速 BF16 路线

组件测量于2026-10-09深夜，实模型/最终构建于2026-10-10，Asia/Singapore，
M4 Max64GB/macOS26.6.2。接续 [row matching](local512-z-matched-bucket-2026-10-09.md)
与 [首步按层选路](local512-qwen-prefill-layers-2026-10-09.md)。只使用本地原
模型/adapter/ref，无下载、GGUF/header改写或dense sidecar。完整目标仍active；
本轮有真实GPU kernel与模型消费者进展，但没有新的默认最快路线声明。

## Kernel：一个 packed word 解一次，供整个 TG 复用

新 `affine_gpu_shared.hpp` 读取原MLX affine Q4/Q8。每个word提取4/8个
code，共用原dtype scale/bias，写入有界threadgroup weight tile，再由
四个SIMD group合作matmul；每个K块前后barrier保护这张tile。F32累加，
可输出F32或原metadata dtype，不支持另一种窄dtype的静默转换。
最大显式W scratch32KiB，无global candidate dense bank，不宽化stored
metadata、不inverse ConvRot、不重打包或改变原source/basis。源码的
SRAM/cooperative描述不证明occupancy、无spill或physical INT8 MAC。

支持BM32/64/128、BN64/128、BK32/64/128，直接使用原physical W pitch与
row/column offsets。若K不能整除请求BK64/128，明确采用完整K32迭代；
`shared_effective_k_tile`可查询实际值，receipt分别报告requested/effective。
不是把tail标成仍运行大tile。实际模型选64/64/64，当前6144等K整除64。

参考Apple Metal Performance Primitives Programming Guide的tensor/
matmul接口，并以当前SDK实测确认，不借其他SDK/device研究结论做资格。
原register/cooperative decode实现仍保留作为显式component控制。

## 数值验证与保留失败

最终576个实际Metal numeric cases通过：Q4/Q8、FP16/BF16、g32/64/128、
M33/67/128、物理pitch与row/column offset、M/N/K480尾块、strided输入、
六tile、F32/原dtype输出；另108个非法contract拒绝。

F32对独立原dtype decode→F32 GEMM保持原relL2≤3e-6、maxabs≤2e-5门槛。
窄输出要求与同一kernel的F32结果cast逐字节一致，并满足原dtype epsilon
逐元素bound加原2e-5 F32 envelope、relL2≤.003；观测最大rel约.000193838
（.0194%）。这些是component边界，不外推全trajectory5–8%保证。

初版FP16 Q4/M128/K480/BK64出现NaN，未接入模型；明确K32安全策略后
修复。第二次出现BF16 narrow abs=.00390625、rel≈.000194：F32检查已
通过，旧统一abs=.003不能容纳一个BF16表示步长。补充output类型诊断
确认后，改为上述显式cast/epsilon契约，未放松F32门槛或finite检查。
期间standalone test误用未定义Tensor别名，改mx::array；四份失败log
及576/84的初步通过log全部保留，最终以576/108计unique cases。

## 真权重组件：胜过旧 F32，不等于全面胜过 packed QMM

同standalone v1 binary，原Z layer0权重、prepared synthetic finite input，
各recipe3warmup/15cyclic-order hot。含最终output cast/eval，不含文件read/
初始packing/旋转、ANE或完整FFN。oracle单独有一次dense decode/setup，
约4.56ms等，不把它摊成零；结束dense ledger claim0。

| source / projection / M / K | packed QMM ms | 原register MPP F32 ms | shared F32→原dtype ms | shared窄输出 ms |
| --- | ---: | ---: | ---: | ---: |
| GGUF Q4 gate /1056/3840 | 6.086542 | 12.969125 | 6.190417（64/64/128） | 6.128875（64/64/128） |
| GGUF Q4 down /1088/6144，原pitch10240 | 3.940375 | 7.645750 | 3.887708（64/64/128） | 3.863750（64/64/128） |
| ConvRot down /1088/6144，原pitch10240 | 4.332500 | 8.237292 | 4.509583（64/64/64） | 4.477292（64/64/64） |

shared对旧F32 consumer约1.83–2.10×；GGUF窄down仅约2%局部差异，gate
仍稍慢，ConvRot也慢于原packed QMM约3–4%。因此不替换普通GPU packed
consumer，不从一个小差异或省global buffer推断整请求/RAM收益。
所有三组F32 real-source error对typed oracle为0；窄输出与packed按bytes
相同。仅限这些prepared inputs，不泛化所有source/trajectory。

三个wrapper exit0、observer errors0、binary unchanged，保存全部samples。
probe通过absolute rpath用保留Qwen-layers-v2 Private501-input库，新header
inline编入probe；没有adjacent library，不冒称502-input新库或loaded-image
trace。源码与binary identities见机器记录，不混用旧v3 register组件分母。

## 接入既有 opt-in F32 API，保持快 BF16 路线不变

`TURBOCIDER_Z_CONVROT_FP32_MPP=1`仍default-off，保留原scope/授权门禁，
在既有 `Weights::project_range_fp32` 与base-only
`project_base_slice_fp32`使用新64/64/64 F32 consumer。原signature、
class data layout、BF16 partial及普通dense/packed projection不变。
请求owner原有drain/synchronize/snapshot和晚期完整GPU重算保留，ANE
W8/A8、headroom、hidden ABI、source/finite/shape/memory准入不变。

runtime identity版本化为
`convrot-gpu-f32-mpp-shared-word-m64-n64-k64-tail32-v2`，selection也报告
实际shared recipe，不把新kernel标成旧register tile。旧历史marker/数据
未改写，当前runner使用新marker，不能用旧库假称运行新consumer。

新 `--bf16-partial-screen --runtime-match-rows` 在mpp/bf16两臂使用相同
实际Private bucket，校验geometry与exact calls，避免长caption cliff
让其中一臂多一chunk。不是固定/匹配bucket消融，不改变GPU control。

## 最新同库完整请求：不以慢原 F32 为唯一分母

原ConvRot checkpoint，Fa4096/Fg6144、Private/F32 join、fixed async1、
stage/fence1、prefetch/lookahead/cache0。LoRA为原distill patch/strength1、
238实际projections，ordinal2完整GPU，gate/up同原shared rotation。
每臂独立process一冷两同prompt热，100ms memory sampling；未取得strict
continuous-load或physical overlap资格。base GPU使用working-tree原有
compiled-dense/retained3GiB hint，LoRA GPU为完整原packed/shared consumer。
这些构建含原用户ConvRot草稿，不能冒称clean HEAD最佳GPU基线。

| workload | GPU warm s | shared F32 warm s | BF16 partial warm s | shared耗时下降 / 自己GPU |
| --- | ---: | ---: | ---: | ---: |
| base fox /9steps/seed42 | 9.601735 | 7.778232 | 7.753037 | 18.99% |
| LoRA fox /8steps/seed42 | 9.494945 | 8.116792 | 8.087222 | 14.51% |
| LoRA lighthouse /8steps/seed17 | 9.783108 | 8.423119 | 8.385401 | 13.90% |

shared比同窗口更快的BF16分别慢约.325%/.366%/.450%，差距很小但不足
以称新增最快预设；新kernel让F32路线接近BF16速度，不升级auto默认。
fox两窗口order GPU→shared→BF16，灯塔BF16→shared→GPU；不同workload
反向order不是同workload ABBA。不跨窗口/库借分母，不拼各臂最好stage。

short fox保留1056；lighthouse选1152。base9两hybrid各2次cold overflow
retry、headroom16，warm0；累计calls290/578/866，真实288blocks/request，
明确 `--cold-retry-cap 2` diagnostic，未追认zero-retry资格。LoRA三请求
calls248/496/744、238bindings、retry/failure/fallback0、headroom1。
共享核改变GPU分支后，ANE配方/完整LoRA和call工作量没有偷偷减少。
全部cold样本保留；例如base GPU12.391566/shared9.460363/BF169.055645s，
不从一个包含compiler/storage caches的cold请求概括冷启动优势。

9份memory reports complete、swap-in/out0，最大process-tree phys-footprint
19,462,719,384bytes；scope为load+cold+warm+exit，不是整机RAM上限、
driver归因或无compression/带宽压力证明。

## 视觉、构建、清理与仍待推进

三workload各GPU与BF16 control对应三张PNG均与此前matched-v3 recipe
exact，新的shared F32不冒称与两control逐字节一致。已看base fox与LoRA
lighthouse case1 whole512：主体/姿态/面部/耳尾/雪或灯塔结构/暖光/
石材很接近，纹理有小变化。灯塔GPU红屋顶、两candidate深色/黑屋顶，
可见差异仍存在，新F32核未消除此差异；不将它唯一归因于GPUpartial。
仅有限agent观察，不是所有scene/seed/detail或用户批准，不用严格latent
等价当本轮图像接受条件，也不改写旧失败记录。

最终Private9、Public8项selected native tests与host75项通过、无skip；
包括576/108新核、8个default/original/candidate/original/两F32 API、
BF16 partial18、actual Private typed lifetime/late full-GPU恢复及Qwen
phase24+per-layer18。不是全仓或完整全模型矩阵。两native-only build
exit0、各502source hashes与当前tree独立匹配/seal一致，Public实际
release class/flags/links guard exit0。库SHA：

```text
Private 21c3f40677e91ece8792590532b4821ac894ff44e5d3139f8aa8bb1c37f86b84
Public  88ac0f369ce50dd5abcb45eca94c7e94650d9b5971ef8ea127fe264d1a6d5e84
```

所有owned jobs terminal后清理两v2 build的429个可重建`.o`、33,488,120
logical bytes（约31.9MiB）与两个空module-cache，回查objects0。standalone
probe direct compile无额外`.o`；保留全部library/CLI/probe、日志/失败/
PNGs/manifests和原模型/adapter/ref，不清用户cache，不向外部进程发信号。

本轮是consumer内tile-local decode，不是已经实现跨层异步ahead-decode。
512²仍保留packed sources；先前同步two-matrix window的真实reuse/预算/
ownership与整请求回本问题依然存在。接续仍需要真正有界ahead-decode/
fused raw consumer、Public/Private operation盈利选择、Qwen generation/
base/LoRA/1–2ref/encoder、更多scene/seed与安静窗口。完整目标保持active，
不缩成这个组件或一条F32路线完成。

完整samples、原始/失败日志、source/binary identities、cold/warm/calls/
memory/PNG与清理记录见
[机器证据](../design/validation/local512-affine-shared-word-20261010.json)。
