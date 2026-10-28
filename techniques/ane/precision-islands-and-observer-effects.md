# 精度岛与观测图：定位SwiGLU误差，修好数值却未获得加速

<!-- knowledge
id: technique-ane-precision-islands-and-observer-effects
type: method
stage: ffn
operator: gemm, swiglu, layout, scheduling
backend: coreml, metal, mlx, cpu-reference
device: cpu, gpu, ane
symptom: precision-loss, measurement, compile-bound
semantics: numerically-equivalent, contract-only
representation: fp16, fp32, iosurface
boundary: producer-consumer, cpu-gpu
api: public, reference-only
-->

[ANE目录](README.md) · [固定来源](sources.md#precision-sources-20261017) · [验证记录（过程原件已移出）](../MAINTENANCE.md) · [原resident FFN批次](resident-ffn-layout-and-gpu-baselines.md)

2026-10-17，沿用M4 Max / macOS26.6.2、SDK26.5、coremltools8.3.0、NumPy2.0.2、MLX0.32.0。本批回答上轮large合成SwiGLU约2.18% relative L2误差从何处扩大，以及恢复精度后是否还有完整边界收益。

结论：观测图中，GU投影误差约0.022%，到SiLU输出约2.18%；保持activation为FP32的单图将最终误差降至约0.108%，两投影在compute plan中仍首选NE。但真实ready GPU X→ready FP32 GPU Z边界比固定MLX基线约慢42%。这是**可定位、可恢复精度，但当前组件未提速**的例子，不是ANE累加器位宽或完整模型质量证明。

## 可复用的方法：只在需要的边界尝试提高精度

1. 固定完整失败输入与参考，先核查 Python/native dtype 和实际导出图，区分端口精度与图内转换策略。
2. 单因素尝试精度岛，诊断 tap/isolated 图单独编译；比较其主输出，不能默认观察无扰动。
3. 对数值合格的图再测完整消费者成本；FLOAT32 策略、计划或输出一致性不证明物理 accumulator 精度。

本页提供定位流程与受限案例，不推荐把全图高精度当作免费的加速修复。

## 单算子合格，不代表组合表达合格

精度是实际lowered图及其输入域的属性，不能仅按源算子名分配误差预算。
独立`sigmoid(x)`与`x*sigmoid(x)`可能走不同优化路径；后者和原生SiLU输出
相同、前者却更准确，是融合/近似假设的线索，不是对具体硬件kernel的证明。
应同时保存identity、单算子、组合表达和实际生产shape的完整输出，包含零点、
小值、宽范围及A/B/A恢复。小幅绝对误差在小值域也可能成为很大的relative L2。

数学等价改写可作为反证控制，例如`x*(1-sigmoid(-x))`避开某些表达模式；
但FP16减法会有取消误差，不能据局部通过称为全域无损。保持原数值门槛、完成/
所有权合同和失败回退，再测真实完整block与整请求；不要通过放宽gate或拆掉
finite检查发布更快结果。激活配方也应进入固定program/prefix身份，而非热路径
悄悄换图。后续实际runtime W8案例见[固定FFN激活研究](../../docs/status/qwen-base512-ffn-activation-2026-10-18.md)；
下文仍保留原2026-10-17案例的版本和结果，不把后续收益回填旧图。

## 1. 固定旧失败，新增两类证据

沿用上轮源FP16权重和GPU输入snapshot，M,K,F,N=512,512,1024,512，seed17/29：

```text
[G,U] = X · Wgu
A = SiLU(G)
H = A ⊙ U
Y = H · Wd
Z = 3·FP32(Y) − 0.5
```

无bias/residual/LoRA/量化/真实checkpoint。原定最终Y门槛仍为relativeL2≤.015且maxabs≤.02，完整GPU Z必须精确等于实际Y的消费者公式。不放宽旧门槛，也不为中间stage生造同一1.5%门槛。

| 记录 | 范围 | 能证明什么 |
| --- | --- | --- |
| 主host诊断，4进程 | 两seed×full/IO，generation0/11/22；132次prediction、186输出tensor、83,361,792个canonical元素 | 逐输入/输出数值、serialized dtype、计划及有限的观测一致性；不计速度 |
| 原生resident重测，2进程 | 各first1、warmup2、20同输入五臂timed trial；230份完整Y/Z，200份正式timing output | 固定GPU基线、数值资格与完整producer-consumer组件边界 |
| 保留早期诊断 | baseline17-v1与full17-v1，共53次prediction | 旧source revision/pilot；不补写其缺失控制、不与主诊断合并 |

诊断每份资产记录实际shape/dtype/bytes/SHA；host与canonical表示分开。原生每份Y和Z均完整raw复核；两者各60,293,120个元素。230份中46份旧Conv CPU+NE控制仍数值失败；**不是230份全部通过**。新candidate本身46份全部通过。

## 2. 先验Python桥接，再看数值路径

固定8.3的[`CompiledMLModel.predict`](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/models/_compiled_model.py#L219)调用[`_update_float16_multiarray_input_to_float32`](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/models/model.py#L995)：FP16 NumPy输入被换成FP32数组。原生bridge另将FP16 MLMultiArray[转成FP32 Python输出](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremlpython/CoreMLPythonUtils.mm#L454)。这不是MLProgram声明自动变成FP32。

本批完整记录feed before/after，验证每个值不变；对FP16声明输出，host FP32必须可精确round-trip为FP16才canonicalize。真正声明FP32的输出保留FP32，不能一律降成half后比较。

两seed×三个输入：host original CPU+NE的canonical Y与旧native `matmul_ne`完整字节相同；tap图完整Y也相同。这支持当前bridge和观测最终结果的一致性。

CPU诊断是row-major matmul，旧native只采集了CPU Conv控制；二者Y不相同、gen0 relative L2约.00496对.00593。**不同图布局的CPU结果不相等不是Python桥接失败**，不要求这个不匹配控制bitwise一致。

另一个容易误判的地方：activation32的FP32 IO版本输出canonical为FP32，FP16 IO版本canonical为FP16，SHA自然不同。按各自dtype解码后六组完整Y数值完全一致；本批不把跨表示hash不同当成精度差异。

## 3. FLOAT32是“不做转换”，不是把已有half图升级

固定[`convert`源码](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/converters/_converters_entry.py#L330)中，`compute_precision=FLOAT32`不插FP16转换；[`_need_fp16_cast_pass`](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/converters/_converters_entry.py#L675)返回False。它不会把原本FP16-authored MIL、weights和inputs自动上转换。

因此要检验真正FP32控制，必须显式以FP32 author图/常量，再检查serialized model；不能只改一个enum然后宣布“FP32仍有2.18%误差”。

| variant | authored图与转换 | serialized两matmul输出dtype | 非const计划首选概况 | gen0最终Y relative L2，seed17 |
| --- | --- | --- | --- | ---: |
| original / tap | FP16 | FP16 / FP16 | NE | .02180794 |
| half_no_promote | FP16 + FLOAT32不转换 | FP16 / FP16 | NE | .02180794 |
| true32 | 显式FP32 + FLOAT32 | FP32 / FP32 | CPU | .000000404 |
| auto16 | FP32 + 全FP16转换 | FP16 / FP16 | NE | .02180794 |
| keep_mats32 | 不转换matmul，其余可转half | FP32 / FP32 | CPU | .00054656 |
| keep_down32 | 只保留down matmul FP32 | FP16 / FP32 | CPU+NE | .02178491 |
| keep_activation32 | 只转换matmul | FP16 / FP16 | CPU+NE | .00107969 |

seed29及另两输入方向一致；全部原始metrics保留。true32请求仍是CPU_AND_NE，但计划选CPU，不写成“FP32在ANE运行”。keep_mats32改善结果也不能仅归功于矩阵dtype：其activation选路随图一起改变。

[`FP16ComputePrecision`](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/converters/mil/mil/passes/defs/quantization.py#L385)的selector=True表示**把该op的合适FP32输入/输出转为FP16**，不是“保留此op为FP32”。本候选从FP32图开始，用`op.op_type == 'matmul'`选中两个投影；其间split/activation/mul保持FP32，再由down投影输入cast回half。

固定8.3[typed execution文档](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/docs-guides/source/typed-execution.md#L83)将显式dtype描述为最低精度；half类型可以采用更高精度实现。不得把旧`neuralnetwork`未typed中间量的CPU规则套到本MLProgram，也不得把serialized FP16输出等价于物理FP16 accumulator。这里没有ISA/物理op timing。

## 4. tap与isolated图：误差出现在何处，不等于物理追踪

下表为seed17、gen0的完整CPU FP64比较。global reference始终从原始X/weights计算理想GU/A/H/Y；conditional reference则从**当前stage真正收到的feed**计算，不把上游误差算作本stage新引入的误差。

| 观察 | global relative L2 | conditional relative L2 |
| --- | ---: | ---: |
| tap NE GU | .00022075 | .00022075 |
| tap NE A=SiLU(G) | .02181171 | .02181171 |
| tap NE H | .02185958 | .02185958 |
| tap NE Y | .02180794 | .02180794 |
| isolated activation，actual up_ne GU→A | .00045460 | .00037587 |
| isolated activation，actual up_ne GU→H | .00057047 | .00043014 |
| isolated down，actual isolated H→Y | .00114118 | .00095955 |
| isolated activation，ideal GU先round-half→H | .00054590 | .00043234 |
| isolated down，ideal H先round-half→Y | .00098134 | .00095898 |

tap的conditional仍是整个图的输入X参考，**不是拿tap输出GU构造出的activation局部参考**。真正的stage-conditional只在独立activation/down调用中定义。不可把表中不同参考分母/不同图的relative errors相减相加，得到所谓每层误差预算。

主诊断六组均有tap GU=isolated up GU、tap Y=original Y的完整值一致性。独立activation只支持/首选CPU，其actual GU feed与up_ne输出逐值相同，局部误差显著较低；独立up/down在CPU+NE请求下首选NE，投影误差相对小。这将误差扩大关联到观测activation/图路线，而非笼统归因为“所有NE matmul都精度不足”。

但增加tap、切图、host materialization和精度cast可能改变融合/优化/放置。最终Y一致性不能证明原图内部与tap图每个隐藏硬件操作完全相同，也没有单独改变一个物理因素的消融。本证据不命名具体NE sigmoid近似、内部量化格式或accumulator宽度。

## 5. 恢复FP16公开IO，让候选回到原生resident边界

FP32 IO诊断会引入新的buffer/type接缝；不能拿它直接替代原有half PB路径。新增`keep_activation32_io16`：公开输入FP16→内部显式cast FP32→仅matmul转half→FP32 activation→half down→公开FP16输出。serialized输入/输出确为65552/FLOAT16。

计划逐op验证：两个`ios18.matmul`首选NE；GU cast、split、`ios18.silu`、mul和hidden cast首选CPU。不是只看到一个“CPU+NE”集合就默认两个投影都在NE。所有计划仍是predicted、physical_operation_timing=null。

该candidate复用原生client/shader、旧helper、相同frozen weights和未修改Conv控制。只把新compiled model复制到旧native约定的`matmul.mlmodelc`位置，legacy arm `matmul_ne`在wrapper里明确alias为**activation32 IO16混合计划候选**，不再代表旧all-half图；旧结果/模型没有覆盖。

边界、counterbalance及prebuilt MLX input view保持[原批次协议](resident-ffn-layout-and-gpu-baselines.md#2-边界是ready-gpu-xready-gpu-z不是predict时间)：ready GPU X→完整ready FP32 GPU Z；必要pack/等待/返回copy/消费者均计时，snapshot/文件IO/CPU参考在timer外。两seed串行，每个native退出后才计算CPU FP64。138次CoreML输出全部采用backing；保留实际返回检查和必要copy fallback，不能推广为系统内部零拷贝。

## 6. 数值通过后，完整路径仍输给GPU

单位ms，为各20正式trial的wall中位。†旧Conv控制数值失败，只保存诊断wall，合格速度比为null。

| arm | seed17 | seed29 | 完整数值资格 |
| --- | ---: | ---: | --- |
| native compiled MLX | .49694 | .49090 | 全部通过 |
| native MPS | .54146 | .53708 | 全部通过 |
| activation32 IO16 candidate | .71050 | .69852 | 46/46通过，最大rel .00108162、maxabs .001096以内 |
| unchanged Conv CPU+NE | .53783† | .54965† | 46/46拒绝 |
| unchanged Conv CPU_ONLY | .68873 | .67479 | 全部通过 |

按同trial固定基线/candidate比值取中位，candidate相对MLX为**.69990× / .70400×**，相对MPS为.77273× / .77272×。这不是median之商，也不是每trial挑最快GPU构造oracle。对固定MLX即约1.42–1.43倍耗时；修复精度没有转化为当前resident组件收益。

六个选定输入的原生candidate Y与Python IO16 canonical Y完整字节相同。新native的所有23×两seed candidate输出都过原门槛；旧Conv控制保留约2.2%误差与null ratio。没有用更高精度candidate的成绩“修正”旧half图失败，也没有从跨进程旧wall推断某段CPU/NE交接的精确成本。

此处端到端较慢与图中的CPU/NE边界并存，但未做physical分段time消融，不能将总差值全部归因于某个cast或一次transfer。未独占主机、锁频或测能耗；结论限本机、两seed、此shape和合成组件，不是公共CoreML/ANE的一般排名。

## 7. 独立验证、可执行入口与后续条件

诊断driver、原生重测wrapper、和标准库合同为原创。15个portable tests验证类型/协议/资格、finite错误、缺失控制、cross-dtype比较、逐op计划标签、旧pilot与helper身份；**它们不重建opaque设备dense输出**。

独立curator不导入CoreML/MLX，明确使用既有NumPy2.0.2与ignored raw资产。它分别计算gate/up的CPU FP64完整参考、重放每stage actual-feed conditional reference，所有pre/post桥接及host/canonical值逐一比较；主诊断metrics最大差约4.45e-16。原生另复用独立完整Y/Z审计；原collector的完整K/F scalar fsum佐证仍保留。

第一次curator因理想H转half的合法subnormal/zero舍入触发严格NumPy underflow而停止，receipt/stderr保留；没有改设备结果或放宽Y门槛。后续改为标准库IEEE binary16 pack/unpack独立建立该feed舍入值，不全局ignore浮点warning。各早期diagnostic runner按自己的归档SHA验证，不要求它们等于最后新增IO16控制后的源码。重测oracle的runner SHA则绑定**复制的旧helper**，不是新wrapper。

不下载真实模型、安装依赖、build上游、清全局cache或改产品默认/电源/安全策略。新输出保留原始数组、models、binary及监督receipt；只将curated JSON/原创代码和报告纳入techniques。

重开条件：source/OS/SDK/compiler/CoreML/MLX版本，dtype/图融合/selector、shape/输入范围/outlier、weights/bias/LoRA/quantization、IO/owner/stream/完成边界或消费者变更。仍需真实checkpoint/任务质量、完整block/request、冷启动、能耗/峰值与物理trace；不继承本次速度、误差或物理精度结论。整体研究继续。
