# Qwen base512：合并QKV W8 producer，局部收益未保留到完整投影

2026-10-19，M4 Max/64GiB。只关注原BF16 base checkpoint，接续
[提交前沿的等待移位实验](qwen-base512-qkv-frontier-2026-10-19.md)。本轮减少了
W producer本身的提交/metadata写回工作，但**完整QKV消费者没有稳定净收益**，
所以没有接入主native、改变App/default，或声称新的512²/40步整图纪录。
总体优化目标仍ACTIVE。

## 机制与四arm

原QKV每次分别对三份`[4096,4096]` BF16权重调用`Device::stage_w8`。hot scale
cache命中时，每份有scale复制和H128旋转/量化两个dispatch、独立status/event/
command buffer。仍需读实际W并完成量化；命中compact scale不等于已经缓存W8矩阵。

研究helper保留原native stager作为强控制，用一个codes/scales bank比较：

| arm | hot W command buffers | hot dispatches | 改动 |
| --- | ---: | ---: | --- |
| native-three-cb | 3 | 6 | 原Device三次stage与原完成/校验 |
| batch-three-codes | 1 | 6 | 同一CB内编码原copy与code kernel，保留算术 |
| fused-three-dispatch | 1 | 3 | code kernel直接读compact缓存scale，并由tile0/lane0写回实际scale surface |
| fused-one-dispatch | 1 | 1 | 再把Q/K/V行域合到一个grid，按实际part选择三个原buffer |

没有创建`[12288,4096]` BF16 packed副本。三个source buffer/offset仍独立，
目标bank只保留当前W；复用原H128 register蝶形、F32除法、FP16 row-scale、RNE/
clamp边界与非有限flag。codes读独立缓存scale而非同kernel正在写的surface，
避免跨threadgroup读写竞态。shader fastMath关闭，固定specialization一次建立。

metadata是两个真实层Q/K/V的compact scale，共49152 bytes；通过原native完成
后单独prime，不放入hot性能分母。这个fixture cache只接受两组已prime的实际
Tensor identity，不是完成了32层生产cache/失效策略。冷prime/setup和新增激活/
metadata workspace另计；输入原W无新增copy。

单bank有pending时拒绝覆盖。CPU仅在真实GPU完成后读状态与surface；GPU callback
持有source allocation、cache和surface直到实际完成。timeout sticky停用，事件/
逻辑返回不授予复用权限；本轮未做真实driver hang/reset资格。

## W-only局部结果

直接从原checkpoint lazy加载两层六份Q/K/V，只物化这些矩阵。每arm2 warm、
32或64 measured，按rep轮转，层0/1交替；完整producer wall包括提交、实际GPU
完成和状态校验。单位ms：

| arm | stock32 | stock64独立窗口 |
| --- | ---: | ---: |
| native-three-cb | 0.898584 | 0.911813 |
| batch-three-codes | 0.871813 | 0.895417 |
| fused-three-dispatch | 0.853229 | 0.881230 |
| fused-one-dispatch | 0.831438 | 0.861501 |

最强候选分别减少约7.47%/5.52%，绝对只约67/50µs。独立timeline仅作观察，
不能用其秒数替代stock分母，也不能将微秒差乘1280充当图片收益。

## 完整QKV消费者：收益消失

同一owner/同一固定512-row Private W8 MatMul图，GPU处理1024目标行中的前512，
ANE处理后512，包含真实W staging、A8、GPU分支、预测、恢复、join和完整BF16输出
完成。原权重仍为两层真实Q/K/V，输入是预先构造的1024×4096 dyadic周期fixture；
不是自然完整DiT block、FFN并行或QKV lookahead命中场景，不导PNG。

| arm | stock32完整projection | stock64完整projection |
| --- | ---: | ---: |
| native-three-cb | 5.118146ms | 5.070688ms |
| batch-three-codes | 5.182750ms | 5.131625ms |
| fused-three-dispatch | 5.151209ms | 5.114417ms |
| fused-one-dispatch | 5.114542ms | 5.084146ms |

最强W-only候选在完整projection首窗只快约0.070%，加倍样本窗反而慢约0.265%。
不把三个候选各自的最小值拼成新的路线。timeline的完整消费者也不是速度资格。

首stock32中，native W准备约0.902ms，fused-one约0.881ms，差额仅约21µs；
A8准备从约0.248ms变为0.262ms。predict/restore join两侧都约3.63ms，仍占大头。
这是消费者上下文下观察到的取舍，不能唯一归因硬件带宽或某一个kernel。host
边界可能重叠，不相加为wall，也不等于纯ANE硬件时间。即使projection本身赢，
现有one-hop lookahead还会隐藏部分W准备，仍须完整40步验证；本轮连这个更前的
消费者门槛都没有稳定通过，因此不扩成runtime开关。

## 全字节与独立CPU复算

每W-only窗口保存两层×四arm的完整I8 codes与FP16 logical row-scales，共8对；
三成功窗共24对，全部与原stager逐字节一致，跨repeat/timeline也exact。
原权重A/B/A恢复、改变层权重、非法arm/dtype/未prime source identity、pending
覆盖拒绝5个负控通过；没有故意改写原checkpoint或借陈旧bank作为成功输出。

独立CPU auditor完整重读原两层六份BF16 QKV payload（192MiB）并核对SHA，按
splitmix64 signs、H128 FP32蝶形、原norm/FP16 row-scale与RNE独立计算所有
100663296个code、24576个scale。当前实际数据全部codes和scale bits exact，
不是采样或只看relative-L2。最终gate拒绝任何CPU code差异；没有借近半整数
容差隐藏不同结果，也不授予其它输入/硬件或物理INT8 MAC资格。

完整projection每窗17份BF16数组：输入A/B、三组GPU控制、三种输入/权重fixture
×四arm，三窗共51份。候选对native完整raw QKV全部bits exact，A/B/换W/A恢复和
相同case重复/timeline也exact。原W8 raw输出相对GPU的relative-L2约0.00822–
0.00864；调度增量exact不是原W8全局无损或语义图像质量声明。

## 真实执行、时间线失败与修正

W-only成功三窗共512 measured producers、581次完整stage组调用，含prime/
preflight/负控；host账本879个W CB、2345个dispatch，cold scale misses另计，
不能把每次cold scale计算/回填误记成hot的6个dispatch。无ANE模型加载/预测。
独立timeline实际逐scope确认native3与候选1个CB，保留Metal完成/时钟证据。

消费者成功三窗共512 measured projections、584次实际预测；每窗只有一个
evaluate model身份，换W/输入/producer策略不新建图。timeline实际152次workload
evaluate与native预测数一致，hot compile/load0；不声称GPU/ANE物理并行或物理
INT8。每窗source/native/库身份、原始memory JSONL hash chain/出生identity/
最大gap/成功结束均独立重读。

首次W-only timeline `outputs/qwen-qkv-stage-stagetimeline-v1-20261019`因观测scope
失败：计时结束后核验六份权重触发6个MLX CB，仍被标进最后一个sample；原计时
sample都符合3/1/1/1，最后scope却是7而非1。原失败不改写、不算成功资格。
修正只在正式循环结束后切scope(-3)，重建stage v3/consumer v2 helper并重采，
未改shader/数值/完成门或放宽原scope计数门槛。已有stock不用重跑，源码由
`outputs/qwen-qkv-stage-tools-pilot-v1-20261019.tar.gz`绑定历史版本。

W-only峰值physical footprint约0.609GiB，消费者约0.995GiB，系统swap-in/out0；
只加载组件，不证明完整Qwen能放进1GiB。本轮无DiT采样、encoder/VAE或完整图像
请求。小样本无独占/统计/thermal资格，统一`qualification_passed=false`。

## 磁盘、工程和后续

本轮新结果中18份48MiB I8文件经逐字节确认后硬链接到本窗相同layer的native
原件，所有路径/完整字节/审计hash均保留，去掉约864MiB重复存储。它们是不可变
实验输出；不是runtime W cache或模型零拷贝收益。没有删除权重/旧结果/日志或
停止ComfyUI，磁盘约3.1GiB可用。继续实验需先预算，不能盲目复制矩阵/下载。

- 新工程：`qkv_stage_batch.hpp`、`qkv_stage_{probe.mm,collect.py,audit.py}`和
  `qkv_stage_consumer_{probe.mm,collect.py,audit.py}`，对应两个`build_qwen_qkv_stage*` builder。
- stage helper：`build/qwen-qkv-stage-probe-v3-20261019/libqwen_qkv_stage_probe.dylib`；
  consumer helper：`build/qwen-qkv-stage-consumer-v2-20261019/libqwen_qkv_stage_consumer_probe.dylib`。
  旧成功stock绑定其各自旧helper，不混用C++ ABI；审计`--cohort-library ROOT LIB`
  和`--tool-source-archive`用于明确绑定，不用新库冒充旧采集。
- 原native仍为661 inputs，`build/qwen-base-gpu-dual-v1-20261019/libturbocider.dylib`，
  SHA256 `7da7a7125bd7cb401344bf885c2425ec7eacc14d81b0563fd45317a94b4b260a`，主源码未改。
- 独立审计：`outputs/qwen-qkv-stage-full-audit-v1-20261019/audit.json`及
  `outputs/qwen-qkv-stage-consumer-full-audit-v1-20261019/audit.json`；设备窗终止后运行。
- 新增7项host负控，最终153项profiler host测试见
  `outputs/qwen-qkv-stage-host-tests-v2-20261019.log`，不是153项硬件/图片资格。
- 工具/测试/报告封存`outputs/qwen-qkv-stage-tools-v1-20261019.tar.gz`；摘要见
  `outputs/qwen-qkv-stage-artifacts-v1-20261019.json`。

复跑分别用`qkv_stage_collect.py`或`qkv_stage_consumer_collect.py --library MATCHED
--native-library CURRENT --repeats 32|64 --mode stock|timeline --output FRESH`；
先用对应builder生成绑定当前native的helper。所有设备窗串行，外部CPU数组审计/
测试/构建在性能窗外。正式完整40步研究基线仍保留原fixed FFN/QKV＋one-hop
lookahead配置，本轮没有默认采用新的stager或新的整图速度数字。
