# Qwen：新同库 GGUF / BF16 streaming 与 GPU 路径筛选

2026-10-10，M4 Max64GB / macOS26.6.2。接续
[首次 SHA 注册](qwen21-source-registration-2026-10-10.md)。完整目标仍 active。
此轮补齐最新库/优化 profile 的实测同内存对比，再筛选真实 FFN 与 encoder
导入候选，不把微基准盈利冒称整模型盈利，也不启用已测更慢的路线。

## 同库四路与边界

原 Q4_K_M encoder/DiT，原 BF16 VAE；BF16 两路也使用原始 BF16，不
转 Q8、不跳层。512² / 40steps / seed29 / 三个 fresh fox-light prompts，
每臂独立新进程、一 cold 两 warm，全部 conditioning miss。先4步做首次
注册和全 tensor 预检，40步跨进程复用，不在 inference 重复原文件 SHA。
resident BF16 本身没有该 base 路径的 full-SHA lease，不人为添加成本。

库为 `build/qwen21-source-registration-v1-private`，native-only / private
enabled / experimental-probes0，512个 source inputs匹配。GGUF GPU prefill
开、原 encoder retention开、8 CPU import workers、1MiB共享读取buffer、
CPU verified import schedule开。Private混合固定 ANE5120/GPU7168、原
Hadamard W8A8、GPU IOSurface、两阶段均混合，不使用 temporal FFN 复用。

BF16 stream 仅每阶段 prefix28 + 两个 suffix slots，managed weight budget
13GiB；resident原 encoder保留。该 budget不是整请求硬RAM cap，采用
实际 process-tree physical peak 验证与 GGUF 内存相近。stream会在每请求
重载 prefix，而 GGUF/原resident encoder保留；这是内存策略的真实代价。

| route | warm encoder s | warm denoise s | warm whole request s | tree peak physical bytes |
| --- | ---: | ---: | ---: | ---: |
| 原 BF16 stream prefix28 | 1.477781 | 43.427133 | 45.477626 | 14,764,599,472 |
| GGUF GPU | .093528 | 43.735217 | 44.432830 | 15,213,059,632 |
| GGUF GPU/ANE5120 | .092301 | 36.189039 | 36.879544 | 15,592,955,464 |
| 原 BF16 resident | .572073 | 42.053734 | 43.321393 | 36,963,900,960 |

stream/GGUF混合峰值比 .9468763，差约5.31%，在本窗口±10%内。
与同内存 BF16 stream 比，GGUF GPU整请求少约2.2974%，仅1.02351×；
混合少18.9062%，约1.23314×。量化本身不是大幅 kernel 加速：GGUF GPU
的 denoise还略慢，主要 encoder/驻留策略抵消了它；真正显著的整体收益
来自 mixed GPU/ANE FFN。原BF16 resident更快于 GGUF GPU，但峰值约
2.43倍；不能用不限内存的 resident替代相近内存条件。

stream的 denoise包含prefix/source/slot setup，其实际完整step spans的
warm中位数约42.503485s；不把43.427133直接与source-excluded GGUF计时
计算“kernel倍率”。warm prefill/decode完整step中位数分别约：stream
1.153047/41.350438s，GGUF GPU1.138007/42.594719s，mixed .955110/
35.230658s。混合每请求1个prefill与39个decode，32/1248成功calls，
failure/fallback/retry0。stream36 encoder层、1280 DiT层、8/160实际fills
与160 last-reader fences，完整pass/drain证据验真。

4步 BF16 stream/resident 的12对完整text/initial/latents/pixels safetensors
和3对PNG逐字节相同；40步3对完整PNG同样精确相同。没有将不同精度的
GGUF与原BF16或另一engine的图片称为bit parity。三个相似prompt/单seed
不代替泛化编辑/LoRA/多场景质量资格。

四臂所有进程树complete、swap-in/out0。前三臂system compression0；
resident system compression7,260,487,680、decompression7,053,508,608bytes。
stream/GGUF GPU/mixed decompression为0/2,277,376/655,360bytes。它们是
系统计数，不能归因某矩阵或称整个窗口没有内存压力。

## 与原 Unsloth 参考及 cold 问题

之前完成的同 prompt / 40steps / 原 pinned sd-server、明确GPU conditioner
记录继续有效：warm condition .13s、sampling107.27s、generation125.325s。
本库 warm encoder约.092–.094s，GPU/mixed DiT约43.74/36.19s，完整请求
约44.43/36.88s，确实领先；但不是相同RNG/noise或跨engine质量等价。

不能把前轮已注册独立新进程 cold encoder约1.010–1.014s推广为所有 cold。
本四路窗口 GGUF在原BF16 stream的高I/O阶段之后 cold encoder2.018405/
2.176693s，仍高于参考cold condition1.18s；前后文件缓存状态并未严格
匹配，所以既不宣称普遍cold赢家，也不把这当同状态因果倍率。源SHA
实际仍0bytes，轻量检查 .301/.958ms，但完整encoder import达到1.6845/
1.83922s，成为下一优化目标。实际whole cold为46.976507/39.867699s。
BF16 stream/resident whole cold45.354152/45.572687s。

## 真实 FFN：提前 decode 与 Splash-style 直接 U4

只读本地 Splash commit `134807b80bd6f64b1533cb1306358c042bb750d5` 的
`runtime/ops/Linear.cpp` 与 `runtime/metal/kernels/prefill/linear_q4.metal`：
形状专化、M32/N128或N256、4/8 simdgroups、预计算activation group sums，
原group64/tiled UInt4 MPP dot后应用affine scale/bias。本模型group32和
row-major原affine planes不同，不能直接粘贴原package/kernel解释权重。

新增验证专用完整layer0 FFN probe，原 source注册命中、仅2个DiT或3个
encoder原tensor，全部原FP16 coefficient/codes不变，synthetic FP16输入，
完整输出而非抽样rows。packed、每调用即时dequant+dense、保留一个block
dense三路交错，各3warmups/9hot samples，fresh packed参数明确动态，
不能被compiler-folded常量cache冒充“每次解码”。一个dense FFN288MiB，
两份最多576MiB，不扩成完整checkpoint。

| actual layer0 / M | packed FFN ms | fresh decode+dense ms | one-block cached dense ms |
| --- | ---: | ---: | ---: |
| DiT 32 | 1.898 | 7.164 | 3.309 |
| DiT 1024 | 23.271 | 24.805 | 22.172 |
| DiT 2112 | 47.040 | 47.773 | 44.919 |
| DiT 4096 | 91.021 | 89.198 | 86.309 |
| encoder 32 | 1.251 | 5.531 | 1.900 |
| encoder 1024 | 23.242 | 25.029 | 22.197 |
| encoder 4096 | 91.224 | 89.472 | 86.368 |

完整6种M两组件的relative L2<=.000410；M>=512此输入下误差0。
两个树complete，physical peak约1.64/1.67GB。512²重要M1024的fresh路线
更慢；cached只少约4.5–4.7%，但全32 FFN dense约9GiB会破坏同内存条件，
不能直接启用all-model cache。有界ahead decode是否能隐藏GPU解码成本
还需要真实跨block调度/依赖与全模型证据；此轮没有声称已证明重叠。

验证专用新group32 row-major direct UInt4 MPP kernel：硬件uint4 dot、
每group32 F32 partial、应用原FP16 scale/bias和预计算输入sum，M32/N128/
4simdgroups；支持38/1048不整tile行数，padding后crop。完整DiT FFN八个M
通过，relative L2约.00042–.00053，但M1024约25.987ms vs原23.191ms，
M1048约26.701ms vs23.946ms，M4096约98.307ms vs91.035ms，均更慢。
所以不改production/default。tile/sums/逐group partial仍是下一微优化点，
不是已经获得Splash性能数字。

第一direct窗口的DiT证据complete，但encoder layer0是Q6→affineQ8，
必须拒绝U4解释，worker返回1，整个窗口inconclusive，原失败保留；没有
将Q8伪装成Q4或放宽dtype门禁。随后相同候选只对真实Q4 DiT重新独立测，
完整窗口通过。两个组件的原dense路线窗口则都完整。
最终匹配当前probe的非U4窗口再测两组件八种M（含38/1048），全部16个
case与进程树complete。encoder实际M38：packed3.025ms、fresh7.816ms、
cached3.221ms；DiT M1048：23.932/26.316/23.658ms。更明确地支持保留
现有512²路径，没有通过只选择M4096的约2%fresh优势推广全model。

## 完整 encoder 导入：缓冲区筛选与实施

同原encoder、原packed embeddings、8 workers、已注册源，1/4/8/8/4/1MiB
正反顺序六个fresh process。每个读取真正消费的4,511,322,112原bytes，
完整源SHA覆盖未消费head仍保持原digest；没有读取source做重复SHA。
新记录0、native persistent hit1，全部process tree完整。

| shared buffer | bank load s | actual read s | decode s | whole import + 13 token gather s |
| --- | ---: | ---: | ---: | ---: |
| 1MiB | .648519 | .218225 | .408032 | .664754 |
| 4MiB | .538983 | .192700 | .323900 | .553422 |
| 8MiB | .523930 | .185467 | .315939 | .539346 |

8MiB whole span少约18.87%，4MiB少约16.75%；更大的row batch减少read/
worker唤醒成本。scratch只增加7MiB（不按decode worker复制buffer），
bank/metadata/SHA/proof/finite/publication合同不变，不把组件收益直接推广
为整model19%。read buffer仍有界，ledger与物理准入按真实8MiB计入，
若CPU SHA/import overlap启用，再单独计入原1MiB hash buffer。

显式 `TURBOCIDER_QWEN21_GGUF_READ_BUFFER_MIB=1|4|8` 接入Qwen原两component
loader，默认1。新CLI runner `--read-buffer-mib`核对实际native容量receipt，
不是只设置环境变量声称使用新buffer。

新版 `build/qwen21-import-buffer-v2-private` 同配置native-only/private/
experimental0，512源inputs全匹配。真实 GGUF bank mixed-K/ownership、
64 typed GPU prefill、原BF16 streaming数值/reader回归三个methods通过，
14个host屏幕合同/负例methods通过，独立parser拒绝0/2/16/08/空值等。
同库1MiB与8MiB，GPU/mixed两路三个fresh prompts的4步预检，24对全
text/initial/latents/pixels safetensors与6对PNG逐字节相同。

该4步GPU encoder import .657443→.532414s，whole text1.054538→.857369s，
whole request6.721400→6.374021s；mixed encoder import .653973→.703210s，
此次cold略波动，而DiT import .587914→.476112s。不只挑GPU成功项或
把组件18.9%直接冒称所有cold改善。全部实际source SHA bytes0、persistent
hit1、不重写记录，read scratch确为1,048,576 / 8,388,608bytes。
40步无dump的同库两profile真实计时另行记录。

## 新buffer同库40步最终窗口

同新v2库、上述GGUF GPU/mixed配置、三fresh prompts、已注册新进程；
先1MiB两路，再8MiB两路，全部完整native/file/phase/进程树验证。

| route / scratch | cold encoder s | cold request s | warm encoder s | warm request s | physical peak bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU 1MiB | .972480 | 45.903988 | .089796 | 44.431477 | 15,474,925,080 |
| GPU 8MiB | .886502 | 45.732274 | .090449 | 44.443898 | 15,455,493,704 |
| mixed 1MiB | .986509 | 38.614690 | .094499 | 36.855134 | 15,512,362,640 |
| mixed 8MiB | 1.018848 | 38.569104 | .090051 | 36.835791 | 15,387,975,240 |

GPU cold encoder少约8.84%，全请求只少 .172s / .374%；mixed cold encoder
此次多3.28%，全请求少 .046s / .118%，不声称稳定cold收益。warm仍为原
计算路径，12–19ms差异是小幅波动，不制造新denoise kernel倍率。
encoder/DiT import GPU8MiB约.553/.490s，mixed约.682/.492s。
40步6对完整PNG与同库1MiB control逐字节相同，SHA实际读取全0，未重写
源注册，read buffer实际8,388,608bytes。没有增加额外per-worker buffer、
all-model dense缓存或改变任何浮点/量化表示。

8MiB GPU窗口system compression475,136、decompression50,675,712、swap-in
131,072、swap-out589,824bytes，mixed compression/swap0、decompression
3,112,960bytes。不能称quiet-load，也不能将系统活动归因7MiB scratch。
因此保留默认1MiB，8MiB为已实现的显式优化实验，不通过放宽memory guard
自动推广。选8MiB时仍按实际scratch做原物理准入。

[机器记录](../design/validation/qwen21-matched-gguf-bf16-20261010.json)绑定
两版本库/源 SHA、完整四路首注册/40步、原与最终FFN probe、直接U4负例、
CPU buffer正反筛选、新库buffer4/40步、物理内存、完整精确性与回归日志。
所有raw失败和source注册key保留本地，key/weights/generated media不进Git。

此窗口已注册cold encoder .887/1.019s低于历史参考1.18s，warm encoder
约.090s、DiT约43.734/36.144s也更快，但前述高I/O四路cold2.02–2.18s
仍保留；没有完成所有冷加载状态的保证。下一步优先检查encoder权重
读取/CPU解码的有界重叠和更严格匹配文件缓存状态的参考测试。
