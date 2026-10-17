# 独立除法、方差与观察接口：公开/私有入口的完整边界对照

<!-- knowledge
id: technique-ane-division-variance-and-observer-boundaries
type: method
backend: coreml, private-ane, cpu-reference
device: cpu, ane
symptom: precision-loss, measurement, lifetime
semantics: contract-only
representation: fp16, fp32
boundary: producer-consumer
api: public, private, reference-only
-->

[ANE目录](README.md) · [前批optimizer-only缩减](adam-optimizer-cuts-and-finite-range.md) · [11文件来源](sources.md#numeric-boundary-sources-20261017) · 全部数组/发射MIL/逐项域与数值审计 · [验证记录（过程原件已移出）](../MAINTENANCE.md)

2026-10-17，M4 Max / macOS26.6.2，固定ANEForge `896c4943`、既有shim，以及已安装coremltools8.3.0、匹配源码`07ffd860`、NumPy2.0.2。八个新进程、两种子置换×正反序，**420次实际求值、12次明确拒绝的host-feedback**；没有Trainer、VJP、Adam状态反馈、任务模型或速度测量。

本批把上轮的有限除法现象拆成独立算子和消费者。结果既不是“FP16都必须逐节点舍入”，也不是“公开API永远精确”：公开CPU_ONLY的独立除法通过已定义域合同，换成乘inverse后却不同；新增输出接口又改变部分inverse端口。私有图中，导出的平方可以是inf，而消费它的方差仍有限。**声明dtype、独立producer输出、诊断图端口和完整消费者是四个不同证据。**

## 可复用的方法：数值重排用范围与消费者共同验收

1. 同一输入域比较除法、inverse 乘法和范围重写；epsilon 位移不能当通用绝对值 floor。
2. 保留正负零、微小值、大值及 NaN/Inf，按完整消费者判断，不能过滤失败后只比较 finite 子集。
3. 新增观察输出当作新编译图，核对主输出是否变化；分别描述名义参考与端口观测，不猜物理指令。

适用于低精度算术诊断；数学上等价的重排仍须重新验证，不能直接作为优化默认值。

## 1. 同一输入合同，不假装三个入口是同一物理设备

| 入口 | 实际软件协议 | 此批能确认 / 不能确认 |
| --- | --- | --- |
| ANEForge/e5rt | 原graph/compile/runtime，显式opt=0；多输出用原compile_multi，mask4 | 真实调用、完整返回、release；不是物理trace或指令精度证明 |
| Core ML CPU_ONLY | 原8.3 MIL builder、保存/compile公开模型、CompiledMLModel | 明确CPU许可；没有硬件kernel或阶段时间采集 |
| Core ML CPU_AND_NE | 复用同一公开compiled资产，另一owner | 所有已知plan首选CPU，unknown保留；许可不等于ANE放置 |

公开两许可共252对完整输出数组逐字节相同，不能据此称测到了ANE。plan来自公开MLComputePlan，只是predicted；源码/发射MIL也不能证明内部每个操作的物理位宽。私有和公开是相同数学表达及32×32 FP16特征合同，不是逐字节相同的compiled程序：公开默认转换pass、私有opt0、节点共享和输出命名等差别均保留，不把差异单因果归给ANE。

每进程四套完整自有有限输入：division-grid、square-range、positive-normal、zero。前两套覆盖±0、2^-24至65504、epsilon邻域及平方边界255.875/256/256.25/512；positive-normal提供非零、正常范围控制。seed17/29只做确定性轴/符号置换，**不是两份独立随机任务数据**。

九种图：`x/y`、`inverse(y,0)`、`inverse(y,1e-4)`、`x*inverse(y,0)`、四输出division-cuts、`g*g`、`B*v+C*g*g`、五输出variance-cuts，以及把实际独立平方读回后再喂入affine的variance-from-square。B/C取**实际FP16 scalar**：B=half(.999)、C=half(.001)，避免把常量舍入误当硬件误差。

72个显式graph artifact、432个预定求值位置：私有140次、公开280次完成；12个square-range因独立平方含非有限值而拒绝后续feedback。拒绝是一整份输入的预先安全策略，不把其中其余finite元素挑出来计通过，也不调用nan_to_num/clamp使它获准。所有上游平方输出、拒绝和未执行后续仍完整保留。

## 2. 除法和inverse乘法：代数相等不代表消费者等价

最小的已定义、目标也可用FP16表示的坐标：

```text
x = 2^-20, y = 2^-24
理想 x/y = 16
私有独立 real_div = .0625
公开 CPU_ONLY / CPU_AND_NE 独立 real_div = 16
```

这不是0/0或目标溢出的歧义。固定本文门槛，在全部四个置换/顺序组的私有独立divide中，已定义且目标finite的1,664个scalar输出被拒绝，另96个输出不匹配指定非有限类别；公开独立divide两许可在相同已定义域均无拒绝。positive-normal的12份完整divide控制全过，不能把边界失败泛化成所有输入都错。

但公开`x*inverse(y)`也不能继承直接除法的资格。48对完整divide/mul-inverse数组，仅24对数值类别和内容全同；共有18,224个数值差异、4,144个类别或zero-sign差异。两者均含未定义点，后者不据哪一个返回finite就判“正确”；已定义点上的完整消费者比较也必须单独做。

差异个数不是合同拒绝个数：例如positive-normal的乘inverse可只差少量ULP，仍满足2%数值门槛；域、非有限类别、数值误差和byte相等分别报告，不用一个大差异数代替验收。

独立inverse端口本身可出现inf，而同图的乘法消费者返回有限值。不能把“将x/y改写为x*inverse(y)在实数中等价”直接用作FP16图优化规则，也不能从一个已导出的r端口推断所有内部消费者都拿到了同样的已物化值。本批未发现或修改上游必然执行此替换的pass；这是显式构造的合法数值反例，不称上游已经自动做错该优化。

## 3. epsilon是位移，不是通用绝对值floor

固定[CoreML inverse定义](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/converters/mil/mil/ops/defs/iOS15/elementwise_unary.py#L406)明确表达`1/(y+epsilon)`，value_inference也用该式；默认epsilon是1e-4。本批显式0或1e-4，两入口的发射MIL保留相同half常量。ANEForge的[graph docstring](https://github.com/sbryngelson/ANEForge/blob/896c4943c5ebf811c0e815f8d1381fd1e157fa7d/aneforge/graph.py#L70)写“floors”，不能把这个词扩展为`1/max(abs(y),eps)`等新函数。

| ε=half(1e-4)，本批坐标 | 私有独立inverse | 公开独立inverse | 公开division-cuts的re端口 |
| --- | ---: | ---: | ---: |
| y=0 | 9360 | 10000 | 9984 |
| y=ε | 4680 | 5000 | 完整数据另存 |
| y=−ε | inf | inf | 未定义域，不授予安全资格 |

y=−1仍返回负数，而y=−ε仍有奇点；加入epsilon并不使任意正负输入安全。私有近零结果对本文加法位移参考仍有超过2%的偏差，不能由“参数非零”推论数值通过。公开独立与新增输出后的差异也存在于CPU_ONLY，不可自动归因为切换ANE。

## 4. 输出观察本身会改变被观察的图

比较独立单输出与新增多输出图：288对完整数组中232对逐字节相同、56对不同，合计47,104个数值变化、1,024个类别/zero-sign变化。本批差异来自公开inverse/re端口；private对应对照保持一致。正反序新进程的378对完整数组则全部位复现，不能把接口差异解释成仅一次随机噪声。

公开division-cuts的d/p消费者与各自单输出结果保持相同，变化发生在额外观察的r/re上。因此这里不声称所有消费者都因加tap而改变，也不把新tap值倒填为原单输出模型内部trace。**每次诊断既要校验新增输出，也要核查原消费者是否保持语义。**

记录完整FeatureDescription、实际发射MIL、标量与graph输出名，排除只按dict顺序猜测端口。公开接口均声明32×32 FP16；Python bridge会把FP16输入dict容器转成FP32，collector保存调用前完整FP16字节并检查调用后数值仍同。它没有另采native预测后的signed-zero字节或物理DMA见证，不把host容器dtype升级为内部算术类型。

## 5. 平方、方差与非有限类别必须一起看

g=512、v=0，C=half(.001)：

| 输出 / 参考 | 私有 | 公开两许可 |
| --- | ---: | ---: |
| 独立q=g*g | inf | inf |
| variance-cuts的q端口 | inf | inf |
| variance / cuts中的a | 65.5625 | inf |
| cuts的sqrt(a) | 8.1015625 | inf |
| 理想实数C*g²再round输出 | 262.25 | 同一数学目标 |

若先按软件IEEE-half提交g²，得到inf再乘C仍为inf；若只在最终端口舍入，则目标a为262.25。公开该方差图符合前一种软件模型的这组点，私有则两者都不完全符合。**这不证明CPU物理ALU逐op为16位、私有某条指令精度或具体clamp阈值**；源程序的FP16标签也不足以选定某种模型。

与[上批Adam切口](adam-optimizer-cuts-and-finite-range.md)的有限大偏差一致，本批进一步观察到：即使同一新诊断图将q导出为inf，也不能把该端口值直接代入a的消费者去重建实际内部执行。不得由这组结果推荐“放大中间值即可修复”或某个统一的safe-divide替换。

actual-square feedback的12次拒绝仍不改。其余36次实际stage，同时保存**按实际物化q的条件参考**和**按原g²的完整表达参考**；符合实际caller输入不自动等于原请求算法通过。这些都是独立stateless调用，不是训练轨迹或非有限状态恢复。

## 6. 两套参考、完整类别审计与证据边界

发布756份完整输出、774,144个scalar元素，含316份有非有限值的数组、136,448个非有限输出元素，全部编码保留；99个唯一f16/f32 payload仅lossless去重。输入有限不意味着输出必须finite；其中一些inf是目标输出范围溢出，一些NaN/inf落在未定义域，不能统称错误或全部视为通过。

两套独立标注的参考是：

- `real_fused_port_rounding`：实际half输入与scalar、Python实数表达，只在输出round到half；分母为零不取得数学资格。
- `software_per_edge_IEEE_half`：每条软件边round到IEEE half并保留inf/NaN。这是反事实诊断，不是硬件仿真；其`undefined_real`描述软件舍入后的域，不等于舍入前原函数的域。

finite目标逐元素门槛固定为`abs_error ≤ 2^-20 + .02*abs(target)`；指定非有限类别须匹配；±0的差异另列，不把数值接近当位相同。无定义域的输出仍保留实际类别，但不据finite、inf或NaN授予whole-domain资格。门槛在采集math模块SHA中预先冻结，不修改前批Adam/梯度门槛，也不拿更贴近某实现的参考替换原目标去消除拒绝。这里的“拒绝”指本文合同，不自动证明框架违反其所有公开保证或芯片损坏。

四套private副本各46个Python文件与2个native来源匹配固定Git；公开六份installed Python来源逐字节匹配8.3 Git。shim沿用旧hash，另记录导入扩展文件hash；均不构成native loaded-image/firmware或原wheel可复现build attestation。

首次curator误把运行cache当不可变模型源，因退出后文件缺失而拒绝。独立只读核查发现72条captured private cache已不可重新核对，252份不可变MIL/weights/公开模型文件仍全部匹配。**保留cache采集hash和原失败，不重建/重新编译来冒充原缓存身份**；curator只重新核查明确不可变资产，缓存当前内容及消失原因未逐文件取证。本轮没有主动清缓存/删除模型；所有设备进程已经终止，无需为这个文件假设错误重跑设备。

本地实现：显式双入口collector、stdlib数学/域/完整比较、、portable tests。默认不导入框架、调用设备或联网：

下一步：在明确当前域与目标后评估数值重排，再回到完整VJP/optimizer或norm消费者；若需要compiled-native字节取证，必须在新协议中release前归档，不回填本批缺失cache。真实模型、GPU基线、matched completion边界、速度/能耗、物理placement与跨设备并行仍需独立实验。源码/OS/SDK/库/shim、表达式/常量、shape/输入域或输出接口变化时重开；整体研究继续。
