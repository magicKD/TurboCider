# Qwen base512：定位SiLU近似，固定图内的等价改写

后续已完成[互补激活下的5120/6144分工与输出交接复核](qwen-base512-complement-share-handoff-2026-10-18.md)：
当前native未改，5120较慢，保留6144；新增按请求区分prediction/output/caller
等待的审计。本页原始激活实验的数字与证据保持不变。

2026-10-18，M4 Max/64GiB，原BF16 base、512²、无LoRA/student/Z Image、无temporal
DiT cache。继续[投影独立消融](qwen-base512-ffn-mixed-projection-2026-10-18.md)。
知识库已有[SiLU失败缩减与图重写](../../techniques/ane/graph-rewrite-search.md)的
局部反例，本轮在runtime W8/实际FFN宽shape复核并尝试更快表达，不声称首次
发现通用ANE/Core ML缺陷。
组件与完整40步文生图/图生图均保留约5%的进一步收益，反向/timeline/变长及
五组质量配对已审计，所有模型进程已结束。不是全部瓶颈已解决，也不改产品默认。

## 为什么不能直接改成原生SiLU

原FFN使用`neg → exp → add1 → real_div`计算SiLU。新增独立typed
`W8Activation`，默认`exp`保持旧字节；`sigmoid`和`silu`只替换激活表达，MatMul
K2048、Hadamard常量、权重/IO ABI、GPU complement、F32 join均不变。

V1真实block筛选中，默认Exp完整block25.319042ms、普通GPU32.875625ms；另外
两种表达均被原数值/动态权重切换自检拒绝，**没有timed candidate样本**。
不是“不支持该MIL算子”，不能把失败后的GPU路径当成ANE速度，也不放宽5%门槛。

V2仅增强失败日志，不改阈值/正常算术。后续V3复现两种失败都是scale0.125、
relative-L2 **0.112092 > 0.05**。顶层memory工具因子进程exit1而报不完整，不是
内存采样间隙故障；独立失败审计确认唯一reason为`command_failed`。失败原件、
日志、非零退出和源码身份保留，不改写成成功。

## 原始FP16探针：融合后的结果与独立sigmoid不同

另建只用自有合成FP16输入的Private Program诊断，不加载任何checkpoint，不
绕过生产W8自检来取得性能资格。测试C128/M32和实际FFN的C6144/M1024；输入为
17点[-0.125,0.125]、负scale0.25、[-16,16)以及恢复首输入。保存完整FP16输入/
输出，identity逐位检查，实际完成后才读取，原输入及A/B/wide/A状态恢复均exact。

独立F64 oracle对C6144/M1024全部6291456个元素重算，小值区间结果：

| 表达 | relative-L2 | x=0输出 |
| --- | ---: | ---: |
| 原exp/divide SiLU | 0.000307 | 0 |
| 独立sigmoid（参照sigmoid oracle） | 0.000753 | 0.5 |
| 原生SiLU | 0.105195 | -0.001704 |
| x×sigmoid(x) | 0.105195 | -0.001704 |
| x×(1−sigmoid(−x)) | 0.000896 | 0 |

SiLU与sigmoid-mul在每个输入模式的完整FP16输出逐位相同，两种shape都复现。
独立sigmoid却没有同等级误差；与融合/优化后使用不同近似的解释一致，不能仅
据API名称断言唯一硬件根因或具体查表实现。

互补表达在负scale0.25时relative-L2约0.001420，wide时0.000258；仍不是逐位
等价于原Exp。FP16减法对负大值存在取消误差，wide最大abs约0.006119，不称
通用无损修复，更不能用operator-only误差替代完整模型画质。

第一次探针空GPU transfer被拒绝，发生于首次evaluate前；保留失败v1。修正为
真实只读output-validation binding后，v2成功10个固定程序/40次evaluate；追加
互补表达的v3为12个固定程序/48次evaluate，均实际校准/完成、hot compile/load0。
这些是诊断程序，不是产品给每层编译12个模型。初始compile/load分别记录。

原件`outputs/qwen-activation-raw-operators-v{2,3}-20261018`；独立全数组审计及
trace为`outputs/qwen-activation-raw-operators-audit-v{1,2}-20261018/`。各版本按
对应source快照/封存工具校验，不用修改后的6-arm工具重标旧5-arm结果。

## 固定FFN互补激活：完整block约5.1%信号

新开关`TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION=sigmoid-complement`，由owner构造时
捕获，进入实际source recipe、executor/prefix身份。只准入显式fixed-program、
授权Private W8、原base Sylvester/ungrouped、unit headroom、descriptor投影；
拒绝LoRA/Comfy/grouped/BF16 boundary及混合投影策略。默认Exp身份和六类emitter
fingerprint保持原字节。旧export签名保留，新表达通过单独typed入口生成。

FFN仍为固定1024-row/H4096/C6144图，QKV固定512-row图，动态权重跨block复用，
不是per-layer checkpoint模型。激活改写没有新增host prediction或热shape切换。

同一V3库的三arm旋转，各2warm/8measured，真实首decode layer0：

| FFN激活 | 原完整compiled GPU | staged GPU | Private每次实际重备W |
| --- | ---: | ---: | ---: |
| Exp | 32.851042ms | 32.867771ms | 25.401688ms |
| sigmoid-complement | 32.853542ms | 32.847208ms | 24.105167ms |

相对原Private候选约快5.10%，普通GPU对照稳定。Private计入真实权重staging、
attention前段、GPU complement、ANE、F32 join、残差及实际完成；**QKV仍GPU**，
此数字不能乘1280外推整图。完整源输入/GPU controls exact，两层四个原BF16
FFN payload与十份数组/窗已独立复核，原输出在输入/权重A/B/A之后不变。

完整block相对原GPU的relative-L2为Exp0.000597648、互补0.000862332；后者
仍高于原Exp，不能称精度优于原配方或逐位无损。
组件审计：`outputs/qwen-base512-ffn-activation-v3-block-audit-v1-20261018/audit.json`。
两种原生表达的失败复核：`outputs/qwen-base512-ffn-activation-v3-rejection-audit-v1-20261018/audit.json`。

## 溢出：不以数值回退换掉固定Program合同

Exp与互补表达都运行真实有限BF16输入导致FP16 hidden溢出的合成负控，独立
oracle结果为32768。每个直接/Hybrid owner只1个Program，blocked rebuild1、
disabled=true；当前及下一调用完整GPU输出与oracle逐位相同，同配置下一调用
ANE提交0，对等QKV不创建Device/Program。改变ambient激活/固定策略后，原owner
仍保持捕获的recipe；旧输出及源数组未改变。

每窗实际evaluate按scope为3/1/1/0/3/1/0/0=9，hot compile/load0；setup/capability
另列。未削减有限性检查、producer/ANE/restore完成门或失败drain。代价仍是
失败owner后续暂时走完整GPU，不声称溢出恢复免费。

独立原数组/selector/CPU oracle审计：
`outputs/qwen-ffn-activation-v3-overflow-audit-v1-20261018/audit.json`。

## 完整文生图与编辑：matched stock与反向/diagnostic复核完成

同一V3库、joint FFN6144＋QKV512、fixed-program，Exp与互补表达之间只改激活
env。每进程seed73→74→73、实际40步，完整wall含VAE/PNG。编辑双方都保留
标准1024参考和runtime-prefix snapshot；不借旧库秒数作分母。

| 任务/激活 | 首次完整请求 | 后两次完整请求 | hot中位数 |
| --- | ---: | ---: | ---: |
| 生成Exp | 33.879856s | 31.648518 /31.556809s | 31.602663s |
| 生成互补 | 32.311516s | 29.890452 /30.085127s | 29.987789s |
| 编辑Exp | 48.680569s | 38.728927 /40.327410s | 39.528169s |
| 编辑互补 | 47.003977s | 36.612635 /37.105989s | 36.859312s |

初始耗时下降约5.11%/6.75%，但编辑baseline后一次明显偏慢。生成FFN暴露ANE
等待约15.64→14.02s，preFFN约8s不变；stage lifetime不下降，不能把收益称作
权重缓存提速。host spans有重叠，以上不是硬件kernel分解或可相加的CPU秒数。

初始两份独立整请求审计通过，当前Exp的全部PNG与冻结旧Exp control exact，
旧秒数未借作分母。两个contact sheet均实际查看：生成互补相对Exp的SSIM
0.961363/0.993593，seed73把手/壶嘴几何可见变化；相对冻结完整GPU则约
0.902176/0.977323，Exp为0.921952/0.977659。它是有损候选，不把同prompt的
另一种有效几何说成逐位一致或普遍更高质量。增量alpha最大2/1/2（255尺度）。

编辑增量SSIM0.997534–0.997780，alpha最大1；大结构基本一致，局部纹理有变，
双方强高光/饱和红色/环纹不能归因于单一候选，更不能据高SSIM授予材质保持
或硬mask inpainting资格。

反向Exp初步为生成31.673952s、编辑38.649003s；相比反向控制，互补约快5.3%/
4.6%。这支持保留速度候选，但也说明不能宣称初始6.75%是精确永久编辑收益。
生成互补timeline已完成：两稳定模型、evaluate2563/2560/2560，first compile1/
load2，hot均0/0；首compile约1.083849s、load合计0.004991s单列。诊断29.943740s
不替代stock29.987789s。编辑timeline也完成：first0/2、evaluate2691/2560/2560，
两次hot compile/load0；诊断37.098662s不作为主分母。

最新两份整请求独立审计v2通过：timeline全部PNG与同配方stock exact，反向
Exp全部恢复原baseline PNG；实际selector、模型计数、source/native/archive/
request/memory均重读校验。原件与审计见
`outputs/qwen-base512-ffn-activation-v3-request-sigmoid-complement-{gen,edit}-audit-v2-20261018/`。

同进程变长矩阵也完成且审计通过：

| 请求 | 实际prefill rows | FFN/QKV预测 | hot compile/load | 完整wall |
| --- | ---: | --- | --- | ---: |
| 生成 | 1048 | 1312 /1248 | 首次0 /2 | 31.655154s |
| 单参考 | 5155 | 1440 /1248 | 0 /0 | 45.378534s |
| 双重复参考 | 9257 | 1568 /1248 | 0 /0 | 64.584898s |
| 恢复生成 | 1048 | 1312 /1248 | 0 /0 | 29.836974s |

两个稳定model身份；首轮能力3次evaluate另列。首尾RGBA exact，也与Comp stock
seed73相同；不同任务不pool成hot median。参考处理仍1024，单图4096 reference
tokens，未裁剪参考来缩短序列。双参考是同一文件重复，只证明length/state恢复，
不授予两个不同主体的语义资格。峰值whole-process physical footprint约30.44GiB，
系统compression/decompression约2.07/1.14GiB、swap0；不能归因于单一模型或
称小ANE图消除了KV/encoder/attention内存。

这9个窗口共28次完整请求/1120实际steps；不包括组件捕获或raw operator诊断。
变长审计`outputs/qwen-base512-ffn-activation-v3-sequence-audit-v1-20261018.json`。

## 同库五组质量配对

完整GPU与候选均为同一V3库/原checkpoint/seed73/实际40步/512²，不是只相对
已经W8量化的baseline。每进程单次首次请求，只作质量，不授予hot速度。

| 提示/任务 | GPU vs候选RGB SSIM | alpha最大差 | 实际查看 |
| --- | ---: | ---: | --- |
| 文字海报 | 0.998492 | 4 | 两行literal文字和蓝方块正确 |
| 成人肖像 | 0.994437 | 1 | 短黑发、圆眼镜、绿色外套、灰背景保持；非身份参考测试 |
| 三种材质 | 0.993556 | 1 | 红方块/金属球/玻璃瓶的数量和顺序正确，形状/反射有小差异 |
| 原茶壶matte编辑 | 0.995950 | 1 | 两侧绿色、较少光泽，但共享原参考的环纹/强光照 |
| 干净参考局部改色 | 0.998157 | 1 | 红方块变深蓝，球/瓶及大结构保留；两侧背景纹理都有改变 |

5张contact sheet均实际查看。干净编辑使用本轮完整GPU materials图作为新参考，
不替换旧茶壶；prompt只改红方块为matte deep blue并保留其它内容。它说明此
样本上能保留主要物体/布局，不是像素不变、hard-mask inpainting、通用材质
保持或人脸身份资格。RGB/alpha分数也不能替代字符/语义检查。

四任务8请求原件：`outputs/qwen-base512-ffn-activation-v3-quality-matrix-v1-20261018`。
干净编辑2请求：`outputs/qwen-base512-ffn-activation-v3-quality-clean-edit-{gpu,joint}-v1-20261018`。
最新5-pair审计及sheets：`outputs/qwen-base512-ffn-activation-v3-quality-audit-v2-20261018/`；
旧4-pair v1保留。全部实际40步、无fallback/retry/headroom变化，标准参考处理1024。

总计19个完整请求窗口、38次请求/1520实际steps，均进程树证据完整/exit0。
只有最初Exp-edit stock窗口记录128KiB**系统**swap-in，swap-out0；其它窗口均0。
这与该窗较大波动并存，但不把系统计数或全部时差归因于本模型/单一进程，
也不删除其后一次hot样本。最终采用更谨慎的反向编辑对照约4.6%作为解释。

## vpipe与运行时灵活性边界

本地只读vpipe仍为`18ddb3b5bd9c74ab7f2b4408ee78efc6cecd48e7`、checkout干净。
本轮没有重跑其DiT forward；此前同版本对照见[profiling/vpipe报告](qwen-base512-runtime-ane-profile-2026-10-18.md)。
默认chunk2048在1024-row decode实际预测0，不能因ANE开关开启就认为执行了
加速，也不能把独立forward秒数同本轮含encoder/VAE/PNG的40步请求算倍率。

这里保留FFN1024与QKV512两份固定图，长prefill重复FFN chunks，QKV长prefill
仍GPU；跨层/步/参考长度的灵活性来自host计划与动态权重，不是热编译。
本轮只优化固定FFN图内的非线性表达，没有改分工、code cache或预取来制造收益。
ANE模型源图小仍不等于总内存小：原GPU权重、weight slots、prefix/KV、encoder、
scratch与driver服务另有生命周期/预算。长参考attention/encoder和约14s的
FFN暴露等待仍有改进空间，实际硬件kernel归因也没有被host spans补齐。

下一步优先在新激活下重新筛选固定5120/6144份额，不能直接继承旧Exp的最佳
分工；再考虑GPU激活＋共享投影模板等更大结构变化。每个新方案仍须实际
权重准备、同库强基线、完整consumer/requests、质量和固定失败合同验证。

## 构建与范围

- 当前native V3：`build/qwen-base-ffn-activation-v3-20261018/libturbocider.dylib`，
  SHA256 `e978d1fec176366c402e20fff296c0839c3a7960875f6452a6d64020b2f8605e`。
- Runtime ID：`tc-runtime-build-v1-ab01cef2a455d9aa65e19dc3ee05f1992227877a769ff8299f67794f3c943b3e`。
- 659-input archive：`outputs/qwen-base-ffn-activation-native-source-v3-20261018.tar.gz`，
  SHA256 `afc3606df103b1035f2948a6300f4da18ace6e93b4208911b9aa24ddd8379cf2`。
- 最终tools/tests/fingerprint封存：`outputs/qwen-base-ffn-activation-tools-v4-20261018.tar.gz`，
  SHA256 `a470233a1e3fb976a31d8eb6c420a2f1e35f7143ad7f82b501bac45982b31442`。
- V1/V2库、source、失败/成功原件和各阶段工具封存全部保留，不重标版本。
- 原生策略/兼容性3项通过；最终77项profiler host测试、V3真实MLX生命周期
  （24cycles/8variants＋joint/idle）通过，均在性能模型窗口之外运行。日志
  `outputs/qwen-base-ffn-activation-host-tests-v7-20261018.log`、
  `outputs/qwen-base-ffn-activation-native-policy-tests-v4-20261018.log`。
- 知识库368页链接/metadata/no-data检查通过（不是模型测试）；源代码diff无
  whitespace错误。自有CPU数组审计/构建/测试未与性能模型窗口并行。
- 实际client selectors不证明物理INT8 MAC/ANE kernel时长或GPU/ANE硬件重叠；
  当前无设备独占/thermal/统计或广泛图像语义资格，统一qualification=false。

没有下载/删除、改写master/references/安装或产品默认，没有停止用户ComfyUI。
优化目标仍ACTIVE；互补表达是有真实收益、仍有损且范围有限的显式研究候选，
默认Exp与正常GPU/App路径保留。
