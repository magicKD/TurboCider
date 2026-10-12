# 扩展 benchmark：完整目标仍在推进

更新2026-10-12。本页对应新增要求：充分覆盖本机 FLUX、Z-Image Turbo、Qwen
Image、LTX 的 BF16/可用量化表示与参数，继续探索 GPU/ANE 最佳路线，
保留最快配置，并补常见框架的成功端到端对照。此前18个campaign是已
完成的第一阶段证据，不是这份扩大目标的完成证明。

## 当前覆盖与必须继续做的工作

| 家族 | 已有权威实测 | 扩展缺口 / 下一项 |
| --- | --- | --- |
| Qwen Image 2.1 | 原 GGUF GPU/Private/Unsloth；BF16 resident/streaming；旧五路线/cache0/1024/4096、新0.22 BF16/partialFP16完整四路线 | 新版quant/cache、512/1024和share/phase最佳配置再核验；stock有明显内存压力 |
| Z-Image Turbo | 五表示16配置、旧七路线框架对照、14个Private网格、新mflux0.22 BF16/partial FP16四路线完整完成 | 额外尺寸/checkpoint与新版本量化候选，不能把有限网格当全局最优 |
| FLUX.2 Klein 9B | stock/cache0、compiled、9B混合、两种Q4与提前解码完成；新predecoded7.211/packed8.001/referenceQ4 8.028s、peak24.4GB | 其它bits与mlx-serve更兼容格式/成功路径、其它shares/shapes、进一步prefill/加载；低RAM与性能候选分开 |
| LTX 2.5 distilled | 最新8配置/32MP4/20memory，quality/Sol/text256、GPU/INT8、batch/clean-exec分开；原组合mlx-serve preflight失败 | GPU Sol+text256只省3.1%且漂移明显，INT8更慢；继续驻留/加载瓶颈与成功参考布局/其它尺寸，不只重用旧0.86% |

除了主表热态延迟，还保留新进程constructor、cold、fresh prompt与
立即重复prompt，512/1024图像与480p/720p-class视频的独立口径，完整
PNG/MP4、memory pressure、执行分区/实际fallback/retry及表示差异。
额外LoRA/PE/VDN附件仍不进入base排名；不同checkpoint不互当分母。

## 本轮实际新增证据

最新Qwen0.22完整40步四路线新增后catalog为52组：8process/32PNG/
8memory、每路4fresh完整验收。GPU43.309/frozen31.810/referenceBF16
41.952/partialFP16 42.000s，新版参考胜pureGPU约3.2%，mixed仍1.319×；
FP16仅差0.114%没有稳定收益，warm-light同engine SSIM0.997113/
MAE0.563。actual288FP16/9BF16参数、cache1GiB与1248frozen calls
均保留，不拿旧更慢分母宣传。[当前完整Qwen](QWEN_BF16_FRAMEWORKS.md)。
FLUX新版七路线工具准备中；native改动/转换/测试不与计时实验并行。

最新Z/mflux0.22四路线新增后catalog为51组：8process/32PNG/8memory
完整，独立满足声明依赖的新环境，GPU7.354/frozen5.959/reference
BF16 7.976/partial FP16 8.355s；mixed优势缩至1.339×，FP16比自己
BF16慢4.746%，reference peak22.51GB优于native约32GB。完整同engine
FP16质量误差与实际306个FP16参数/215BF16、每图generation metadata
均核验，旧参考9.520s不混池。[最新Z结果](Z_IMAGE_FRAMEWORKS.md)。
FLUX新版本、Qwen/Z更多shape/quant/hybrid与成功外部LTX仍pending。
Qwen0.22 BF16/partialFP16短4步单cold预检已完成，288FP16/9BF16 DiT
参数与全部required weights、两个memory窗口核验；仅预检，不增加
完整campaign计数或宣传40步倍率，见[Qwen当前边界](QWEN_BF16_FRAMEWORKS.md)。

10-12参考1024/4096MiB两组Qwen完整对照新增后catalog为50组。各自六
进程、24PNG/6memory、四fresh/route完整验收；实际缓存bytes/全部
required weights/1248 frozen runtime calls、cold/repeat与pressure均保留。
GPU/frozen/reference为43.377/32.393/45.451s及43.331/32.407/45.498s，
约1.05×/1.40×；reference仅差0.104%，不作缓存速度优劣的因果证明。
1024的参考peak34.48–34.88GB低于4096的37.71–37.76GB，保存较低
开销reference候选，不跨cohort拼最小native分母。[完整记录](QWEN_BF16_FRAMEWORKS.md)。
扩大目标仍active；原固定参考是2026-09-20commit/版本字段0.19.2，
FLUX当前上游版本同机重测与更多形状/分区/成功LTX对照继续pending。

10-12新增9B完整hybrid cohort使catalog达到45组。8进程/32PNG/8memory
完整，实际4096-prefix/8192-GPU补集、96calls/request、无fallback、原
index与2shards核验通过；hybrid7.841/compiled7.844s只差0.031%，参考
7.355s胜。CoreML是INT8权重/FP16 I/O，不冒充W8A8；cold/压力/repeat
与误差单列。[完整新结果](FLUX9_HYBRID.md)。下方保留10-11历史过程，
当时的pending/N/A不覆盖这个精确512/4已测候选，也不变成更广资格。

最新终态优先于下方历史过程：Qwen v11完整有效五路线GPU/frozen/mflux
BF16/Q4/Q8四fresh中位43.238745/31.058807/48.511753/56.335344/55.100963s，
40PNG/10memory完整；[完整记录与内存压力](QWEN_BF16_FRAMEWORKS.md)。
Z v5完整七路线GPU/frozen/Private/Public/mflux BF16/Q4/Q8为7.363311/
5.780123/6.139409/6.537484/9.519604/9.971474/10.080610s，56PNG/14memory
完整；[七路线来源/执行/冷启动](Z_IMAGE_FRAMEWORKS.md)。
FLUX最新stock参考BF16/Q4/Q8为8.962395/8.303592/9.466238s，24PNG/
6memory完整；没有同窗口native，不跨轮算倍率，[精度与缺口](FLUX_PRECISIONS.md)。
后续Qwen cache0六进程完整对照GPU43.347453/frozen31.250897/mflux
45.968219s；参考peak32.92–33.84GB、swap-out0，原生对该参考为1.060×/
1.471×。不能只用较慢stock数字宣传优势，也不跨轮拼倍率。
FLUX v5最新原生/参考同窗口8.042414/8.828487s，为1.098×，16PNG/
4memory完整；旧1.370×不当成当前优势。进一步cache0完整ABBA为
native7.979642/mflux7.351531s，参考胜出、native多用8.544%时间。
9B compiled GPU/混合/匹配量化缺口不因此标完成。
LTX v2完整8配置/16arm/32MP4/20memory已完成，batch GPU quality/
Sol/text256为63.847972/62.848950/61.868682s，INT8相应71.333099/
67.165562/66.036188s。clean-exec GPU/INT8估计62.638723/65.740014s
另列；97帧差异与E5重编译、加载波动保留，不因此完成扩大目标。
扩大目标仍active/partial，不因新增成功就把参数/混合/框架空缺标完成。

1. 为Qwen原BF16创建只读派生布局：encoder仅键名调整，VAE仅键名与
   已确认singleton空间轴reshape；整个payload字节逐一校验一致。
   不改原权重，不修改只读mflux参考代码，不将格式转换时间扣进或
   加进单次inference。准备encoder约24.256s、VAE约0.917s，单独记录。
2. Qwen v1完整512²/4步预检完成计时：mflux BF16首个request+PNG为
   7.519705s，native GPU为8.922268s，两个完整进程树、原token IDs/
   template校对通过。后续图片检查发现v1遗漏64个真实FFN权重、输出
   为噪声；因此连“正确模型可执行”都未证明，不作排名或质量通过。
   原sourceSHA注册后复用，不在inference再扫模型。
3. 首次正式40步四进程native→mflux→mflux→native窗口在第二进程被
   memory verifier拒绝：实际四张完整图均完成，exit0，但一次采样间隔
   522.920ms超过原500ms上限。native已完成fresh43.226/43.179s，
   mflux已完成58.628/60.036s；这两个进程不能升级为完整ABBA排名。
   保留failed窗口，恢复混合配置后用原门槛在fresh目录重新实测。
4. 以前Qwen frozen最快历史配置的compiled manifest当前不存在。
   原32个W6144 source mlpackage与source manifest仍在；已全部重新编译到
   独立持久cache。新40步单cold完整PNG/memory预检成功，46.877607s，
   实际1248runtime+32warmup calls、failure0、checkpoint SHA通过。
   尚需当前warm/反序实测，不能把旧33.121s冒充当前最佳测量。
   本轮启动前失败保留在
   `outputs/qwen21-mflux-original-v2-40step-palindrome-20261011/startup-error.json`。
5. v1 mflux affine Q4/Q8完成512²/4步PNG预检计时，首request为
   9.059559/8.905662s，两完整进程树。此框架按原定义保留BF16 text
   encoder、不量化该组件；不能与encoder也为Q4_K_M的GGUF组合宣称
   同精度/同内存。v1同样遗漏FFN权重，输出噪声；不作有效模型执行。
6. v6五路线40步回文十进程/40PNG/memory全部完成。有效native GPU/
   frozen四fresh中位43.249697 /31.007695s，省时28.305%。MF三route
   是invalid-output diagnostics：Comfy gate_up未拆成64个required
   gate_layer/proj，stock loader默默保留random initialization，不能
   用56.879/60.699/61.443s宣传跨框架领先。
7. v2完整布局已拆分全部32个gate_up，完整payload仍相同；required
   coverage297 DiT/398 encoder/226 VAE均满。三参考4步输出狐狸；另
   BF16完整40步单cold50.961601s、图片正常、memory complete。新的
   v9五路线正反序后来因采样gap失败；v11换同门槛async observer后
   完整成功，上方最新结果才是有效框架排名。

原始实验路径：

- `outputs/benchmark-model-views/qwen21-bf16-mflux-v1-20261011/layout-receipt.json`
- `outputs/qwen21-mflux-original-v1-precheck-20261011/summary.json`
- `outputs/qwen21-mflux-original-v3-40step-abba-20261011/summary.json`
- `outputs/qwen21-mflux-original-v4-frozen-restored-precheck-20261011/summary.json`
- `outputs/qwen21-mflux-original-v5-affine-precheck-20261011/summary.json`
- `outputs/qwen21-mflux-original-v6-40step-five-route-palindrome-20261011/summary.json`：
  已完成，但MF参考噪声仅作失败诊断，native两route有效。
- `outputs/qwen21-mflux-original-v9-complete-weights-five-route-20261011/summary.json`：
  修复后该窗口因memory gap失败，保留，不追认为成功。
- `outputs/z-mflux-original-v4-full-seven-route-20261011/summary.json`：
  原排队七路线；v9失败所以当时未启动，后续v5已完整完成。
- [原生完整结果及invalid参考标记](QWEN_BF16_FRAMEWORKS.md)
- [可携带预检/无损证明/alpha差异记录](../design/validation/qwen21-mflux-layout-precheck-20261011.json)

原生Qwen PNG保存模型生成的RGBA，当前mflux输出RGB，后者只有隐含
opaque alpha。比较明确分别报告raw RGB、完整alpha、白背景composited
RGB；不会悄悄丢掉透明度或宣称两边输出功能完全等价。

## 后续入口与实现核对

- `z_mflux_framework_screen.py`已加入原生GPU/frozen/Public/Private与
  mflux BF16/affine4/8；Private候选支持2048–8192的七个已声明partial
  shares、F16/F32 join，核对actual channels与join receipt，不仅记录env。
  BF16/Q4/Q8短预检图片正常。七路线v5完整对照已完成，新增shares/
  join候选尚未硬件实测，不进入最佳配置或性能表。
- Qwen派生encoder的大副本占用影响下一批模型布局预算。新增held-fd
  APFS clone + compact-header工具，仅缩短reference不消费的352个tensor
  名字，保留全部payload和所有language targets；实际fixture三项通过。
  大模型clone已经完整校验，v1的17.53GB自建重复副本在旧窗口terminal
  后移除，原source和clone保留，换出空间用于正确DiT派生文件。当前
  计时中不改文件或做大批I/O；`st_blocks`不等于独占物理磁盘字节。
- FLUX9B实际为H4096/F12288、8 dual+24 single。现有HybridSession已
  支持index+完整shard SHA，可直接保持原sharded模型，不必创建18GB
  dense单文件。待扩展的是4B-only GPU图/padding/selected checkpoint与
  export geometry，之后用真实block/全图验证再解除9B显式拒绝。
- LTX当前batch走spawned VAE helper，而single-shot CLI/disposable worker
  可走clean exec finalizer。后续把两种完整MP4边界分开测，并筛选现有
  Sol/text256/INT8 profiles；不能把历史clean decode micro当新whole-request
  加速，也不能把换进程cold算成同session warm。

## 采样写盘阻塞修正（不放宽验收）

v9修复权重后的mflux BF16四张40步图正常、actual source coverage完整，
但一次579.013ms的采样间隔超过原500ms，window保留failed，不推广完整
排名。原采样器每sample在采样线程同步fsync，写盘可能阻塞后续采样。
新增显式bounded async writer，把同样的每sample fsync移到独立writer；
64-record queue上限、不丢record，写入/overflow/drain失败均拒绝成功，
结束时完全drain/fsync再独立重放。原gap/role/exit/hash-chain判定不改。

原同步writer保持默认；Qwen/Z新screen显式选择async，对所有路线一致。
blocked-fsync/overflow/write-error三项、真实CPU进程独立replay均通过。
新mflux BF16四请求窗口`qwen21-mflux-original-v10-async-memory-precheck-20261011`
完整1953samples，actual最大gap110.011ms，allowed仍500ms，memory complete。
新匹配库`build/benchmark-async-v1-private`已经重建，源/工具身份不借旧库。
LTX完整配置入口已完成：persistent与clean-exec、GPU/Sol/text256/INT8
独立口径；不把finalizer estimate伪称同session request_wall。

当前owned串行任务：

- `outputs/qwen21-mflux-original-v11-async-five-route-20261011`：修复后的
  完整五路线40步正反序，使用新库与相同async observer，terminal complete。
- `outputs/z-mflux-original-v5-async-seven-route-20261011`：原生四路线与
  三参考精度8步正反序，terminal complete。
- `outputs/qwen21-mflux-cache0-v1-20261011`：完整同窗口GPU/frozen/
  mflux BF16 process-local cache=0候选，terminal complete；仍保留原
  40步/4request/500ms memory门槛，不改OS策略，不与stock样本混池。
- `flux9_mflux_quant_screen.py`已完成stock BF16/Q4/Q8参考精度入口，
  独立标精度与required parameter coverage，不与native BF16拼等精度
  倍率；最新六进程完整实测已完成。原生group64与quantized embedding尚未验证，后续
  同源压缩表示/匹配layout仍需补齐，不因此把FLUX量化目标标完成。
- `outputs/flux9-bf16-framework-v5-async-matched-20261011`：新同窗口BF16
  native→mflux→mflux→native配对，使用匹配async build、完整required
  source coverage；terminal complete，实际1.098×而非旧1.370×。
- `outputs/flux9-bf16-framework-v6-cache0-matched-20261011`：独立同窗口
  BF16 reference cache0候选，完整ABBA；不改模型/数学/OS策略、不混
  stock样本，terminal complete，参考胜出，负结果保留。
- `outputs/ltx-full-configs-v1-palindrome-20261011`：terminal failed；GPU/
  INT8各两完整视频与memory完成，Core ML E5警告粘连JSON导致collector
  解析失败，不推广完整排名。raw/媒体保留，显式诊断解析器已修正。
- `outputs/ltx-full-configs-v2-explicit-diagnostics-20261011`：terminal complete，
  实际32MP4/20memory完整。8配置/16arm正反序，quality/Sol/text256的
  GPU与INT8 batch，另GPU/INT8 clean-exec；每arm两完整704×448×97/
  24fps/11步/stereo MP4，不预填尚未测的其它形状或更优配置。
  batch同进程warm与clean-exec新进程conditioning-hit/estimate边界
  分开统计；8个候选的完整97帧RGB/motion均已比较并保留。
  Core ML E5 resource/recompile消息与raw stdout SHA显式保留，未知
  文本仍拒绝；不隐去cold setup或伪造物理ANE/无warning资格。

最新FLUX compiled v2/v3/v4三cohort均terminal complete，每组24PNG/
6memory、四fresh/route、每compiled请求96实际分支，默认/deferred/
fused分别7.483/7.382/7.374s；各自参考7.315/7.326/7.343s，不能混池。
compiled/eager当前RGBA中的RGB exact，9B ANE guard仍未解除。
Z网格已完整结束：outputs/z-private-partition-grid-v1-20261011，
七个partial channel shares×F32/F16 join共14候选，每项GPU/frozen/
Private完整六进程正反序、24PNG，84窗口/336PNG与独立audit全部完成。
最快Private4096/F32为6.147741s，同窗口frozen5.793671s更快；不把更大
ANE份额或F16 join假定更优。[完整参数网格](Z_PRIVATE_PARTITIONS.md)。匹配量化、
9B混合和参考布局缺口也继续保留。

FLUX9B新增原始index/shard导出已实际24/24完成，H4096/F12288、
ANE4096/GPU8192、固定rows1536，无18GB平铺副本。受控engine-cache
以真正9B id编译24分区成功；低层bridge在blocks0/12/23两输入幅度
共6次成功，源index+全部shards SHA通过，rel-L2 1.52–3.62%、cosine
最低0.99934。它是INT8 per-channel权重/FP16 I/O前缀对照，不冒充
activation8/native integer MAC、完整生成加速或已通过9B混合资格。
导出、编译和probe分别保留在outputs/flux9-ane-prefix4096-r1536-*
及outputs/flux9-coreml-prefix4096-r1536-probe-v1-20261011.json。
随后10-12已实现/核验9B GPU complement与完整四路线请求，见上方新
结果；首个source-bound候选跑通，不等于其它分区/尺寸或扩大目标完成。

随后DiT-only Q4/g32 matched cohort已完整完成，catalog达到46组。
native BF16/Q4为7.921748/7.986974s，mflux BF16/Q4为7.379937/
8.030596s；32PNG/8memory、同payload与实际539参数/153linears/g32
核验通过。Q4不加速，native peak18.16GB而不是30.53–30.65GB，保存
memory candidate；全组件/其它bits仍待补。[新匹配结果](FLUX9_MATCHED_Q4.md)。

再后encoder252+DiT153线性Q4/g32也已完整匹配，catalog达到47组：
native BF16/Q4 7.741839/8.005275s、mflux BF16/Q4 7.346223/8.042573s，
32PNG/8memory，Q4不省时但native peak11.99GB、比自己BF16约少60%。
encoder36层/902参数与两component整payload通过；旧28层fallback的
转换失败保留，复用只读DiT CoW使准备满足2GiB reserve。原embedding/
VAE仍BF16，不能称每个tensor4bit。pinned mlx-serve新派生布局因缺
embed_tokens.scales实际startup拒绝，不作分母。[完整新结果](FLUX9_FULL_Q4.md)。

提前解码source-Q4 DiT→BF16 operands后保留compiled singles的新五路线
已完成，catalog为48组：predecoded7.210639/packed8.001289s，省9.88%，
peak24.40–24.45GB而不是packed11.95–11.97GB。40PNG/10memory、每
session153decode与后续96compiled branches/request，完整输出误差与
同session packed/predecoded/packed/predecoded恢复均通过。额外BF16
rounding和原Q4误差不混称同精度；[高内存performance候选](FLUX9_PREDECODE.md)。

尚未运行的量化/参数格子继续标pending，格式/memory失败保留原原因，
仍尝试安全的同源派生格式，不用失败启动时间、不同精度或其它机器
数字凑胜率。赢/输/无收益都发布。最快配置需留下参数、来源、artifact
和复现入口，且在当前文件状态下能实际加载；不只保存一个旧最小数字。
