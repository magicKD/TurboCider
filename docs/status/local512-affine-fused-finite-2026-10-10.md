# GGUF / ConvRot 融合解码与 finite status：准备快一倍，R1仍不如packed

2026-10-10，Asia/Singapore，M4 Max64GB/macOS26.6.2。接续
[准备成本归因](local512-gguf-prepare-parts-2026-10-10.md)，本轮真正实现
了GPU融合核，不是删除finite检查或只修改标签。完整Z/Qwen base/LoRA、
编辑/encoder、GPU/ANE与真正ahead-decode目标仍active。

## 实现与边界

研究header `tools/native/affine_dense_finite_candidate.hpp`，由已有
`gpu_weight_consumer_probe` 消费，不改变production native源码/默认模型
路由。每线程解码1/4个packed word，保留原FP16/BF16 metadata、MLX
`T scale * uchar code + T bias`表达式及最终T存储；在写出系数时检查
有限性，SIMD内汇总bad，输出小UInt32 status，再归约max。

尾部所有lane参加SIMD归约，in-range flag恰写一次；inactive SIMD不写。
Safe math，没有提前宽化metadata、inverse ConvRot或raw Q4_K/Q6_K支持。
原UInt32 words和metadata必须row-contiguous，合法非零physical row
offset保留；strided source拒绝，不藏一次contiguous copy充当快核。
仅Q4/Q8、group32/64/128，fresh dense目标上限256MiB。

dense与status分别先ledger reserve；fresh-bin cache scope约束两backing
实际capacity。完成kernel、status reduction及synchronize，finite与两个
capacity都通过后才发布；Data持有dense/status storage claim，离开owner
或escaped/lazy reader不假装释放。nonfinite/overflow/admission失败回滚，
原source不可变，没有persistent sidecar、weight merge或新模型导出。
这里只是同步guarded prepare，**还没有异步跨层pipeline或model fallback
新验证**；framework reduction/compiler scratch不冒称整个进程RAM上限。

## 三个真实权重，同binary、循环换序完整消费

同本地layer0 Q4 gate、Q8 down、ConvRot gate。M1056为合成prepared
sine输入的component几何，不是完整512请求。每目标75MiB，Q4/Q8最大
status分别614,400/1,228,800bytes；batch4降到153,600/307,200bytes。
两dense加最大status的ledger上界约150.6/151.2MiB，不能称两完整FFN槽。

五recipe：原window及batch1/4 × TG128/256；每臂3warmup、15hot，保留
所有预声明样本。prepare独立测，另外真正测fresh decode+一次GEMM，并
与直接packed QMM循环换序。不把两段median相加代替whole window。

每recipe实际38次成功prepare（系数/输出preflight、warmup和hot），原
window真实miss38/hit0、actual R1；每轮reader消亡后再开新目标，没有
跨层LRU命中或把步数当同矩阵复用。相同内容ticket/原source一直保留。

下表统一展示batch4/TG256，不事后为每case拼不同最佳recipe：

| 原source | original prepare ms | fused prepare ms | 原decode+GEMM ms | fused decode+GEMM ms | packed QMM ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| GGUF Q4 gate / FP16 | 3.948542 | 1.705750 | 10.307083 | 8.012208 | 6.059375 |
| GGUF Q8 down / FP16 | 3.942000 | 1.742625 | 10.456375 | 8.021625 | 6.246125 |
| ConvRot gate / legacy BF16 scale | 3.913417 | 1.762958 | 10.332667 | 7.913041 | 6.564625 |

prepare下降约55–57%，完整decode+一次GEMM下降约22–23%；但R1仍比
packed慢约20.5–32.2%。q4 whole-window最小样本中位偏向batch1/TG256，
q8偏向batch4/TG128，差额很小，不用组件winner声明稳定全局selector。
完整四candidate结果、每样本和source/binary哈希见机器记录/receipts。

三个real-source所有candidate的typed coefficients与原prepare逐字节
一致，GEMM输出也与该合成输入的packed control逐字节一致/relL2=0。
允许5% component输出误差，未把该预算当完整latent/图像资格。没有
模型生图、真实LoRA或视觉新验收，不因此声称新整图提速。

三continuous host observers无error、thread joined、binary unchanged，
max gap约.166/.132/.132s；不是quiet-window、设备独占、GPU timestamps
或物理overlap证明，qualification始终false。

## 回归与后续决定

1项opt-in实Metal unit通过无skip：原12 window及12 attribution cases，
新144 numeric/lifetime、48 boundary coefficient cases、252非法contract，
240个last-group NaN/Inf/finite-metadata typed overflow拒绝。新numeric
cases还各检验finite reject/refill、status独立admission、escaped reader
后的next-output拒绝。小矩阵/SIMD尾部、offset、Q4/Q8、FP16/BF16、
group32/64/128、batch1/4、TG128/256、signed zero/取消/tiny metadata均覆盖。
另12个非法CLI组合在source access前拒绝。计时用的v3 binary直接重跑
新cases也通过；重复测试不重复计数，不是全仓suite/完整模型资格。

v1基本核、v2补充boundary/consumer、v3完整实际计数三个standalone build
均terminal exit0。v1/v2 logs与binary保留，但本表只绑定v3；没将初版144
cases冒称最终192的全部资格。v3链接保留Private库absolute rpath，无
adjacent dylib或loaded-image证明。production两个502-input snapshots
仍独立匹配，含原用户草稿，不称clean staged-only或新App发行。

决策：保留融合prepare作为更低成本的研究基础，**不将每层R1 dense
重解码接入默认**。下一步必须真正有界current/next层准备与ready消费，
验证同GPU带宽竞争、in-flight source/目标claim、escaped reader/bank
generation、取消/失败drain及实际decode+GEMM/完整模型收益。两矩阵
window不能冒称两完整FFN，MLX async提交不能冒称物理overlap。

所有owned jobs已terminal；temporary unit目录自动清理，三probe build无
`.o`或module-cache。仅保留小binary/log/receipt和原source，没有下载/
改写模型、删除compiled artifacts、清driver/用户cache或signal外部进程。

[机器记录](../design/validation/local512-affine-fused-finite-20261010.json)
绑定源码/header、library/probe、source payload、全部样本及未完成边界。
