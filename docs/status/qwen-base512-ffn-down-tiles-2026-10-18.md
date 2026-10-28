# Qwen base512：F32 down分解、tile筛选与FP16 gate/up负结果

后续已实现并测试[显式runtime text-prefix复用](qwen-base512-text-prefix-2026-10-19.md)：
完整请求有小幅速度信号，但发现命中时首步算术变化的画质反例，仍不采用默认。
本页原tile/FP16实验数据与旧native范围保持不变。

2026-10-18，M4 Max/64GiB，继续[restore encoder与GPU complement分析](qwen-base512-ffn-handoff-encoder-2026-10-18.md)。
只使用原BF16 base、512²，无LoRA/student/Z Image；当前native仍是activation-v3，
659-input源码与原二进制未改。这轮没有新的整图加速纪录，也不将未胜出的候选
接入native/App/default。

结论：原GPU complement的F32 down不是整段10–12ms的唯一来源；独立gate/up
反而更长。12种down tile未取得可靠完整block收益，计入每次转换的FP16 gate/up
也较慢。保持现有6144/原tile/BF16 GPU分支，不为“更小tile/更低精度”增加负担。

## 保持什么、改变什么

使用真实base40请求的第一DiT step之后、首decode layer0的原始捕获。每窗只
完成第一step，然后重放完整block；未导出PNG，也不是完整40步生成/编辑速度。
本轮捕获源均为文生图，没有新的图生图样本或对应速度资格。
普通compiled GPU、分阶段GPU、Private真实重备W三arm旋转，通常各2warm/8measured；
追加一窗16measured。QKV仍GPU，强基线不换成拆分的较弱控制。

ANE仍是固定1024-row/H4096的FFN图，动态权重跨block复用；本轮先选C5120，
GPU7168通道，最终补C6144/GPU6144控制。Private fixed-program、互补激活、
descriptor投影、F32 join、cache/prefetch off均不变，无热调参或逐层ANE模型。

仅GPU down的`dense_gpu::projection_range`改变BM16/32/64、BN64/128与dynamic/
static extent，共12档。保留原BF16操作数、完整12288物理W行宽、逻辑选片和F32
输出；没有compact/widened W bank。static只适用完整tile，不能把本轮1024-row
结果扩大到所有不对齐prefill；本轮未把这些策略集成到运行时。

首次GPU kernel编译发生在未计时的A/B/A和warmup阶段，不算热样本。ANE的
Program构造/错误合同与普通GPU完整控制仍独立检查。GPU JIT与Private ANE
compile不是同一个计数；不据“fixed”标签承诺不可见driver内部行为。

## 独立分解：gate/up约8.28ms，down约4.27ms

另外编译同算术的gate/up、candidate down、原F32 down控制，使用已完成的FFN
输入/hidden，无ANE在飞，三arm各2warm/8measured。down数组保持F32，不先窄化
再比较。转换/准备与完整block分开列；这些独立host完成边界不能相加冒充关键路径。

| 份额/原策略 | gate/up alone | F32 down alone | 完整GPU complement alone |
| --- | ---: | ---: | ---: |
| C5120，BM32/BN128 dynamic | 8.276396ms | 4.265646ms | 12.328416ms |
| C6144，BM32/BN128 dynamic | 7.127501ms | 3.690354ms | 10.612979ms |

这纠正了“将整段10–12ms归给F32 down”的假设；仍不是按shader hardware busy
time分账，也不证明gate/up的每个kernel都使用同一种硬件路径。

## 12档tile：初始微弱差异未通过重复验证

下表为C5120的无observer完整Private block中位数，包含实际重备W、attention
前段、GPU/ANE、join与残差，各arm8个measured样本。原完整GPU控制约32.84–32.87ms。

| BM / BN | dynamic | static extent |
| --- | ---: | ---: |
| 16 /64 | 25.749833ms | 25.801729ms |
| 16 /128 | 25.793709ms | 25.713438ms |
| 32 /64 | 25.770834ms | 25.789522ms |
| 32 /128（原tile） | 25.595875ms | 25.609729ms |
| 64 /64 | 25.538688ms | 25.654104ms |
| 64 /128 | 26.103917ms | 26.066437ms |

64/64 dynamic初始只比原tile少约0.22%，不是可靠收益。其16样本重放回到
25.658396ms；独立反向原tile为25.616959ms，原完整GPU控制保持约32.87ms。
这不支持采用新tile。64/128的独立down约4.75ms，比原4.27ms更慢；static
extent也未自动带来收益。

新C6144原tile完整block24.035042ms，仍明显快于所有这些C5120变体；不得把
“本分区里微弱最好”误当成优于当前强候选。C6144没有执行其余11档tile，不
给它们授予跨分区最优或失败结论。

64/64 dynamic的独立timeline也完整结束：一个稳定FFN模型、workload15次
evaluate，hot compile/load0，无drop/pending，延迟event负控通过。restore
encoder仍约3.577792ms，没有消除上轮5120的长区间；诊断完整block25.706209ms
不是stock分母。其它新tile只完成stock，不借此授予它们实际selector零热编译证明。

## FP16 gate/up：每次转换也计入，结果更慢

另一候选只将gate/up操作数转FP16，两个matmul各自输出回BF16，再使用原BF16
SiLU×up、F32 down与join。原BF16 checkpoint不改变。权重作为显式compiled
运行时输入，转换在调用体内，未在计时外准备常驻half bank或只测ready FP16岛。

| 配方 | gate/up alone | GPU complement alone | Private完整block |
| --- | ---: | ---: | ---: |
| C5120原BF16 | 8.276396ms | 12.328416ms | 25.595875ms |
| C5120 FP16 gate/up | 8.777750ms | 12.851083ms | 26.179375ms |
| C6144原BF16 | 7.127501ms | 10.612979ms | 24.035042ms |
| C6144 FP16 gate/up | 7.721271ms | 11.223625ms | 24.813250ms |

两个FP16控制的普通完整GPU block约32.85/32.84ms，没有换弱分母。半精度
连同转换没有提速，不能用理论吞吐或较窄dtype给它加速标签。

FP16 hidden相对原BF16的完整数组relative-L2约0.00227–0.00230，数十万个元素
改变，是额外有损方案；不是同精度kernel替换。完整block残差后的误差较小不
表示原信息无损，更不能把偶然与W8误差抵消说成画质提高。没有整图速度优势，
不进入完整图片资格或产品实现；FP16溢出/极值生产恢复也未在此取得资格。

## 更严格的独立数值审计

新增`down_audit.py`：

- 重读所有source/native/probe/observer身份、原checkpoint两层四份完整FFN payload，
  以及每窗十份完整BF16 block数组。匹配输入和普通GPU controls exact；同配方
  反向/加倍/timeline的全部数组恢复exact。
- 新reader逐字段要求hidden BF16、partial/control F32、精确shape、完整无重叠
  payload。首次CPU审计沿用旧BF16-only reader，正确拒绝了F32文件；没有放宽旧
  reader，而是新增mixed-type严格入口。原件未改、模型未因此重跑，v2/v3审计通过。
- 逐元素重算candidate与普通F32 down的完整数组误差；FP16还核对原BF16 hidden
  控制。再从原始4096×12288物理W的32个固定输出坐标、每点完整K做独立F64 dot，
  覆盖首尾行/列。总2304个见证点，最大abs约1.57325e-6、最大relative-L2约
  9.27884e-7；这是采样oracle，不冒称完整CPU dense FFN或图像语义资格。
- CPU负控覆盖物理pitch/少K/改值、BF16/F32混淆、重叠/越界payload、错误policy/
  sample顺序、遗漏转换成本标记与缺失数值控制。首次host负控fixture把原False
  又写False，没有实际扰动；改成True后检测成功，原失败日志保留，不是放宽检查。

最新18窗审计：`outputs/qwen-base512-ffn-down-audit-v3-20261018/audit.json`；
旧14窗v2保留。原生FFN数值自检、固定失败策略和有限性门槛均未削减。

## 工程、原件与后续

- `ffn_projection_probe.mm`新增仅诊断的down tile/FP16开关；普通默认仍不变。
  `build_qwen_ffn_down_probe.sh`按显式份额/policy构建helper，不为每层生成模型。
- `handoff_collect.py --down-policy m32n128d`等选择对应诊断库；`--gate-up fp16`
  必须显式指定，拒绝与已否决packed策略混用。`down_audit.py`在模型结束后运行。
- 首12窗：`outputs/qwen-base512-ffn-down-m{16,32,64}n{64,128}{d,s}-c5120-stock-v1-20261018`。
- FP16两窗：`outputs/qwen-base512-ffn-down-fp16-m32n128d-c{5120,6144}-stock-v1-20261018`。
- 后续4窗：同前缀的`m32n128d-c6144-baseline-v2`、`m32n128d-c5120-reverse-v2`、
  `m64n64d-c5120-repeat16-v2`、`m64n64d-c5120-timeline-v2`，日期后缀均20261018。
- helper为`build/qwen-ffn-down-probe-v{1,2,3}-20261018`；所有旧库/原件保留。
- native仍`build/qwen-base-ffn-activation-v3-20261018/libturbocider.dylib`，SHA256
  `e978d1fec176366c402e20fff296c0839c3a7960875f6452a6d64020b2f8605e`；659-input
  archive仍`outputs/qwen-base-ffn-activation-native-source-v3-20261018.tar.gz`。
- 早期down工具封存`outputs/qwen-ffn-down-tools-v1-20261018.tar.gz`，SHA256
  `08568ce915841f02fa2e89225b55d443658fa2232da3e8a96610ad8f270e1584`；最终
  `outputs/qwen-ffn-down-tools-v2-20261018.tar.gz`摘要独立保存于同名`.sha256.json`。
- 最终91项profiler host测试通过，日志`outputs/qwen-base-down-host-tests-v5-20261018.log`。
  不是91项硬件测试；实际counter/Private调用/完成另有原始timeline证据。

十八窗全部exit0，共456个measured完整block重放、278次Private workload预测，
180份block数组＋112份组件数组；每窗只捕获第一DiT step，不是18个完整40步
图片。进程树峰值physical footprint约15.64–15.71GiB，系统swap-in/out均0。
部分窗口有竞争CPU负载；无GPU独占、thermal/统计或整图新资格，qualification=false。

全部性能窗口串行，自有构建/测试/CPU数组审计均在窗口外。无下载、删除、停止
ComfyUI，未改App/default/reference/权重。磁盘约17GiB free，暂不清理原件。

下一步停止在此tile矩阵上盲搜，转向减少实际重复工作或ANE固定图内部成本。
重复文生图的text-prefix snapshot已有只读源码依据：prefix零timestep调制、文本
causal边界、空reference snapshot结构；目前准入仍明确edit-only，未新增flag，
未测off/on收益。若实施，必须显式default-off、完整recipe/conditioning身份、
callback drain/叶值所有权及同库完整40步/多seed/变长恢复检查，不能静默扩大旧
编辑开关。当前最佳完整请求仍来自前轮6144配方，本轮不借旧秒数宣称新加速。

目标保持ACTIVE；这轮排除两类无收益改动并补齐独立归因，不声称所有瓶颈已解决。
