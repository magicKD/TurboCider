# Qwen base512：restore encoder实测与GPU complement消融

后续已完成[F32 down独立计时、12种tile与FP16 gate/up筛选](qwen-base512-ffn-down-tiles-2026-10-18.md)：
gate/up本身更长，两个方向均未取得新增速度收益，native仍未改。本页原始
counter/packed实验数字与范围保持不变。

2026-10-18，M4 Max/64GiB。继续[互补激活分工/交接分析](qwen-base512-complement-share-handoff-2026-10-18.md)。
只关注原BF16 base的512²生成/编辑优化，不涉及LoRA/student/Z Image。当前native
仍是activation-v3，659-input源码、二进制和原source archive均未改。

本轮取得了更细的GPU完成证据，但**没有新的整图加速纪录**。保留FFN6144＋
QKV512速度候选；GPU侧gate/up合并没有净收益，未接入native或产品默认。

## 把event等待与encoder执行区间分开

上轮5120每张图的worker output wait约4.5s，而6144约0.5s；这只是host等待，
不能决定该改CPU callback、恢复kernel，还是GPU分支。新增独立诊断observer：

1. 复用未改动的`observer.mm`，继续记录真实Private compile/load/evaluate及
   Metal CB完成；另外记录实际`encodeSignalEvent`/`encodeWaitForEvent`的event
   identity和value，不从模型开关猜测消费者。
2. 仅对已编码event wait之后的compute encoder，通过compute-pass descriptor
   附加两个Metal GPU timestamp样本。保留shader、参数、dispatch与依赖；这是
   诊断encoder创建路径，不是stock零开销或原调用完全不变的承诺。
3. 两端`sampleTimestamps`与host时钟包络标定counter ticks，拒绝缺失、错误值、
   drop、未drain、错时钟包络、错误event/consumer及scope数量；不假定跨机器
   counter单位。每次真实Private block必须对应一个实际restore encoder。
4. 先运行真实64-byte blit、shared-event/increment数据自检。V2追加host延迟
   放行event的负控：实际CB存活约24–31ms，encoder采样约4.9–9.0µs，数据确实
   增加。该样本表明当前采样没有把早先的event等待直接算进encoder区间。

这里观察到的是**GPU encoder边界经过的时间**，仍可能包含竞争、抢占或执行
内部停顿；不是shader busy time、独占restore算术成本、物理ANE时间或UMA带宽。
不能把host wait、CB window和encoder elapsed相加成完整wall。自检只约束当前
设备/API路径，不证明所有调度/故障下均有相同性质。

## 真实完整block重放，不冒充40步图片

沿用真实base40请求：先完成第一DiT step，捕获首decode layer0，再停止原请求。
每窗普通compiled GPU、分阶段GPU、Private每次重备真实W三arm旋转，各2warm+
8measured；包括attention前段、GPU complement、ANE、F32 join、残差与完成。
QKV仍GPU；未导出PNG、未完成40步，不把本表乘1280当整图收益。

Private固定FFN1024/H4096、C6144或5120，互补激活、descriptor投影、fixed-program、
cache/prefetch off。每窗15次成功workload预测（含A/B/A及换层权重控制），能力
检查另列；同一owner一个Program，零fallback/retry/热compile/load。

原探针新增编译时固定通道参数，默认仍6144；两个份额使用独立二进制/进程，
不是热改图。额外`GPU complement alone`只测ready捕获FFN输入与相同GPU分片，
无ANE在飞，属于独立分支对照，不是完整block的速度分母。

## 结果：5120的设备侧restore区间也变长

V2同配方stock/timeline都完成；以下同一次timeline内取8个measured样本中位数：

| FFN份额、原GPU策略 | worker prediction wait | worker output wait | restore encoder GPU elapsed |
| --- | ---: | ---: | ---: |
| 6144 | 约10.1ms | 0.293ms | 0.202ms |
| 5120 | 约8.4ms | 3.632ms | 3.530ms |

V1独立复现约0.197/3.574ms，但V1没有延迟event负控，不回填成新自检已执行。
V2补上负控后仍复现。相同1024×4096输出的encoder elapsed也增加，故不能仅
归因host完成通知变慢；但这仍不能唯一定位为restore shader本身的算术退化。
较早就绪的restore与更长GPU complement并存，竞争/调度仍需更细消融。

这次Private evaluate、event value与sample scope逐个核对。6个timeline均是
一个稳定FFN模型，workload15次，hot compile/load0，所有记录无drop/pending；
不是将两个Program当两颗ANE或用host重叠替代硬件重叠证明。

## GPU gate/up打包合并：没有收益，不增加运行时负担

假设：当前GPU complement的gate/up为两次BF16 GEMM，request-local拼成一份
连续权重后用一次GEMM，可能减少输入重复读取/提交。保留原BF16中间结果、
原F32 down partial和最终join；禁止adapter，不改原checkpoint。

| 固定份额/策略 | 普通完整GPU block | Private完整block | GPU complement alone |
| --- | ---: | ---: | ---: |
| 6144 separate | 32.850896ms | 24.016063ms | 10.591438ms |
| 6144 packed | 32.866854ms | 24.073791ms | 10.595521ms |
| 5120 separate | 32.930563ms | 25.554833ms | 12.359125ms |
| 5120 packed | 32.867021ms | 25.587312ms | 12.331438ms |

来自无observer的stock窗口，每arm8个measured block。packed完整block分别慢
约0.24%/0.13%，不是有意义的加速；timeline也未显示其消除了restore长区间。
不能把split/staged GPU的额外边界当作普通完整GPU分母。

代价是真实新增packing：捕获两层共192MiB（6144）或224MiB（5120），两层首次
准备约7–10ms，单独记录，不藏入“免费缓存”。若32层全部保留，逻辑权重副本
理论为3/3.5GiB，但**没有实现或测量这种整模型bank**，不能称实际RSS增加量。
本轮两层candidate数组与separate逐位相同，没有新的数值收益或精度损失；W8
路线本身相对普通BF16 GPU仍近似。没有速度收益就不将大packed bank带进产品。

下一步更值得隔离GPU complement的F32 down partial：它走当前
`dense_gpu::projection_range`的MPP kernel（BM32/BN128、动态slice），而普通
gate/up是MLX BF16 matmul。可以保持物理权重pitch和F32 join，筛选合法BM/BN、
静态extent或更好的consumer；这只是有源码依据的候选，不宣称已证明down独占
了全部10–12ms。每项仍需真实权重、强GPU基线、全数组、完整请求和固定图合同。
不通过删除有限性检查或提前复用ANE output来缩短等待。

## 独立审计与工程入口

新增`handoff_audit.py`重读全部12窗的source/native/probe/observer身份、原checkpoint
两层四份完整FFN payload及120份完整BF16数组，独立重算误差；不是dense FFN
全CPU oracle。相同真实输入与全部普通GPU controls exact，同配方跨版本/stock/
timeline的十份数组恢复exact。packed/separate的十份数组也实际核对为exact。
首decode完整blockrelative-L2约6144为0.000862332、5120为0.000784511；不把此
误差排序当作完整图片、文字、编辑身份/材质的语义资格。

最终审计：`outputs/qwen-base512-ffn-handoff-audit-v1-20261018/audit.json`。
原件：

- V1四窗：`outputs/qwen-base512-ffn-handoff-c{6144,5120}-{stock,timeline}-v1-20261018`。
- V2八窗：`outputs/qwen-base512-ffn-handoff-{separate,packed}-c{6144,5120}-{stock,timeline}-v2-20261018`。
- native仍`build/qwen-base-ffn-activation-v3-20261018/libturbocider.dylib`，SHA256
  `e978d1fec176366c402e20fff296c0839c3a7960875f6452a6d64020b2f8605e`；659-input
  source archive仍`outputs/qwen-base-ffn-activation-native-source-v3-20261018.tar.gz`。
- 最终helper为`build/qwen-ffn-handoff-probe-v3-20261018`，含observer及四份诊断
  probe；它们不是产品为每层创建四份ANE模型。V1 helper/原件保留。V2 helper
  在macOS Bash空数组nounset处构建失败，未运行模型；修正builder后新建V3，
  不覆盖/伪装失败版本。
- V1绑定工具：`outputs/qwen-ffn-handoff-tools-v1-20261018.tar.gz`，SHA256
  `b500823bd55d697b25e3559987b1c7c7701f617589d7dfdb5d5b568e6c331acf`。
- 最终工具/tests/docs：`outputs/qwen-ffn-handoff-tools-v2-20261018.tar.gz`，摘要
  独立保存于同名`.sha256.json`。审计旧窗口必须携带相应历史工具archive。
- 最终85项profiler host测试通过，日志`outputs/qwen-base-handoff-host-tests-v3-20261018.log`。
  不是85项硬件资格测试；真实counter/data自检另在timeline中有原始记录。

12个模型窗口全部exit0，共288个measured完整block重放、96个measured独立GPU
complement，以及180次Private workload预测；捕获只完成各自第一DiT step，
不是12个完整40步生成。进程树峰值physical footprint约15.59–15.90GiB，系统
swap-in/out均0。部分窗口记录竞争CPU负载；其余load检查通过也不证明GPU独占。
无thermal/统计、物理INT8/重叠或整图速度新资格，全部qualification=false。

全部性能窗口串行；自有构建/host测试/全数组审计均在窗口外。没有下载、删除、
改App/default/reference/原权重或停止ComfyUI。目标保持ACTIVE，继续优化真实
关键路径，不因这轮工具更细就声称全部瓶颈已经解决。
