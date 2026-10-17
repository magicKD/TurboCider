# W8A8校准的实际调用链：样本顺序、观测分组与未插入Q/DQ的matmul

<!-- knowledge
id: technique-ane-calibration-order-and-qdq-patterns
type: method
stage: ffn
operator: gemm, conv, layout
backend: coreml, cpu-reference
device: cpu, ane
symptom: precision-loss, measurement, lifetime
semantics: approximate, contract-only
representation: fp16, q8, w8a8
boundary: producer-consumer
api: public, private, reference-only
-->

[ANE目录](README.md) · [九文件固定来源](sources.md#calibration-sources-20261017) · 原函数/完整数据 · [验证记录（过程原件已移出）](../MAINTENANCE.md) · [W8A8能力边界](w8a8-int8.md)

2026-10-17，M4 Max / macOS26.6.2 build25G83，固定coremltools8.3.0、NumPy2.0.2。这里研究**量化前端是否按预期构造模型**，不是整数吞吐benchmark，也不声称这是更新版本仍存在的问题。

关键发现：该8.3实现的两个校准range helper都在合并前覆盖历史极值；同一组宽/窄样本只换顺序，实际Conv Q/DQ scale变16倍。公共入口没有转发分组参数，且校准临时模型使用CPU_ONLY。等价matmul图虽然API返回成功，却没有新增activation Q/DQ；后续压权重的结果仍只是W8，而非本实验已证明的W8A8整数执行。

后续[三版本/所有权审计](coreml-versioned-calibration-and-ownership.md)已单独核查固定9.0和main：9.0修正group转发但range仍覆盖，main累计range；这是原函数/source对照，不是把本篇8.3设备输出回填为新版运行。release、PR状态及模型包/GIL修复的不同范围也另存。

## 可复用的方法：校准要追到最终导出图

1. 用可辨识的多批输入测试 range 聚合和覆盖顺序，核实配置参数是否真正传到内部 helper。
2. 检查最终模型哪些节点插入 Q/DQ，哪些算子仍未覆盖；配置名不等于整数执行路径。
3. 分别保留校准观测图和最终图的完整输出，按相同质量预算比较；CPU 观测不作为 ANE 整数性能证据。

适合建立自己的量化导出回归，而不是直接复制本批校准样本或量化速度结论。

## 1. 从公共API进入内部校准器

固定[`linear_quantize_activations`](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/optimize/coreml/_post_training_quantization.py#L519)的流程是：

```text
样本名推断/校验
  → _get_activation_calibration_stats
      → ModelDebugger追加中间输出，临时模型CPU_ONLY预测
      → 输入/中间tensor min/max；concat相邻range调整
  → prefix Q/DQ pass
  → suffix Q/DQ pass
  → 冗余DQ→Q消除
  → 新MLModel
```

8.3已提供非experimental公共名字，但该文件仍[导入experimental目录中的校准helper](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/optimize/coreml/_post_training_quantization.py#L21)。不能因API名称稳定，就推断内部路径、校准设备或所有算子覆盖已经改变。

固定9个源码/文档身份；实际执行前验证其中6个installed Python文件与固定Git对象逐字节相同。它只绑定这些Python代码，不是native CoreML image/source attestation。reference只读，未安装/升级/修改上游。

## 2. 原range helper的覆盖顺序，与实际导出互相验证

[`_update_tensor_range`](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/optimize/coreml/experimental/_post_training_quantization.py#L97)先计算本样本min/max，随即写入字典；后面的min/max又与刚写入的值比较。[`record_intermediate_output`](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/optimize/coreml/experimental/_model_debugger.py#L267)有相同顺序。因此本次调用链没有保留跨样本的历史极值。

原创helper probe从固定Git对象只提取5个已审函数体；不执行上游模块初始化、不导入CoreML。8个真实range函数CPU fixture的结果均等于最后样本range；独立标准库union参考与之对照：

| 校准顺序 | 应有跨样本range | 两个原helper实际结果 |
| --- | --- | --- |
| wide[-16,8] → narrow[-1,1] | [-16,8] | [-1,1] |
| narrow → wide | [-16,8] | [-16,8] |
| wide → middle[-4,4] → narrow | [-16,8] | [-1,1] |

这不是只凭源码提出猜测。两seed实际公共导出，校准L/H为**同一个输入pattern的1/16倍与原倍数**，只交换顺序：

| Conv，seed17 | LH（wide最后） | HL（narrow最后） |
| --- | ---: | ---: |
| returned input range | [-1,1] | [-.0625,.0625] |
| first Q scale，FP16 | .00787353515625 | .000492095947265625 |
| h0 Q scale，FP16 | .005268096923828125 | .0003292560577392578 |
| p1 Q scale，FP16 | .0016231536865234375 | .00010144710540771484 |

三处均相差16倍。seed29相同方向；每个输入及中间量的返回统计均与**最后一次实际CPU观测输出**的极值完全一致，和独立跨样本union分别保存。

此处只证明固定8.3路径的行为。没有检查所有后续版本/PR或发布上游修复，不写“CoreML所有版本校准都有此问题”。

## 3. 分组参数没有穿过公共入口，校准也不是NE观测

公共函数文档描述`calibration_op_group_size`，但[line604](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/optimize/coreml/_post_training_quantization.py#L604)仅向helper传model和sample_data。原public函数在CPU call doubles上接受-1/0/1/2，实际helper参数均为默认-1。0只用于call-double，不曾作为实际临时模型分组执行。

实际导出另请求group_size=1，记录到每次仍一次追加`p0,h0,p1`三个中间输出，两个样本各一次；返回统计和全输出都存raw。`ModelDebugger.step`及`predict_intermediate_outputs`的[默认compute_units为CPU_ONLY](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/optimize/coreml/experimental/_model_debugger.py#L225)，本批临时预测实际传入也全是CPU_ONLY，不继承base模型的CPU_AND_NE许可。

这有两个工程含义：

- 大模型不能仅因传了group_size=1，就假定这个版本已经限制了每个临时模型的输出数/内存；应观察实际调用或单独使用可审计路径。
- 校准采用CPU临时观测图，部署可采用另一计划/融合/最低精度；校准激活不是原部署图内部的物理NE trace。见[精度岛/observer批次](precision-islands-and-observer-effects.md)。

wrapper只记录原方法的输入/返回值，不改统计与结果，并在finally恢复。另用未插桩公共API重新导出LH控制；其最终MIL Q/DQ参数和全部确认Y与插桩版完全一致。不拿wrapper一致性证明隐藏硬件路径无变化。

## 4. 同样的Linear函数，A8 pass不覆盖同样的图

测试函数为两个32→32 dense linear，各后接ReLU，M=16、无bias；两seed，固定源FP16混合权重和输入。Conv采用BC1S/1×1、matmul采用row-major；独立数学参考相同：

```text
H = ReLU(X · W0)
Y = ReLU(H · W1)
```

固定[prefix支持列表](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/optimize/coreml/_quantization_passes.py#L1643)只有conv、avg_pool、max_pool、add；[suffix pattern](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/converters/mil/mil/passes/defs/optimize_activation_quantization.py#L145)同样未列matmul。suffix还依赖后继与activation匹配；本例最终在x、h0、p1附近保留3对Q/DQ，不是把所有ReLU/output都各量化一次。

| 实际转换 | Conv Q/DQ对数 | matmul Q/DQ对数 | weight constexpr个数 |
| --- | ---: | ---: | ---: |
| FP16 base | 0 | 0 | 0 |
| 只linear_quantize_weights | 0 | 0 | 2 |
| public A8 LH/HL、未插桩LH、局部union | 3 | 0 | 0 |
| 上述A8再W8 | 3 | 0 | 2 |
| decompress_weights(W8+A8 union) | 3 | 0 | 0 |

matmul A8几种输出全与原FP16逐值相同；其“W8A8”标签arm实际全与W8控制相同，且没有activation Q/DQ。**数值好不代表已经执行目标变换**。这不是matmul永远不能做整数计算，而是本固定自动pass的pattern范围；不可把Conv的计数移植给等价matmul或[上轮SwiGLU图](resident-ffn-layout-and-gpu-baselines.md)。

Q/DQ计数/参数来自转换对象最终MIL，包体逐文件SHA绑定；本批没有第二套protobuf parser去独立复核这些操作，也没有编译后IR/ISA解码。所有小图有已知首选设备的非const操作都首选CPU，压缩constexpr部分usage/首选缺失仍保留unknown，不擅补CPU或NE。

## 5. 修正统计是控制，不是自动获得代表性校准集

局部union控制不修改安装库：从两份实际CPU观测tensor独立汇总全部min/max，再显式调用同一组三个graph pass。此例H覆盖L，所以其Q/DQ与public LH相同；它不是一般模型的数据集选择算法或已集成修复。

确认输入独立于校准pattern：narrow、wide、带两个±8 outlier的wide和全zero。新函数/数据在预测前固定Y门槛为**relativeL2≤.03且maxabs≤.02**，不与旧SwiGLU门槛混用，也不是模型质量标准。

下表只列CPU_AND_NE请求的Conv输出，数值为relative L2原始比例（非百分数），seed17 / seed29；CPU_ONLY同样全量保留，没有只挑通过样本。

| 路线 | narrow | wide | outlier |
| --- | ---: | ---: | ---: |
| FP16 | .000639 / .000599 | .000639 / .000599 | .000690 / .000788 |
| W8 only | .005187 / .006268 | .005187 / .006265 | .005773 / .005909 |
| A8 LH = 本例union | .151780 / .228259 | .060915 / .013300 | .434494 / .514492 |
| A8 HL | .060915 / .013300 | .927488 / .907161 | .943166 / .941277 |
| W8+A8 LH | .151794 / .224501 | .060761 / .014542 | .434377 / .514481 |
| W8+A8 HL | .060761 / .014542 | .927542 / .907378 | .943206 / .941415 |

wide-last修复了这组校准样本的历史极值，却仍有narrow量化分辨率不足、确认域变化和outlier风险；seed17的wide也未过门槛。不能凭seed29普通wide通过就称修复后全部可用。所有zero都精确通过，但零例不代表质量；失败都保留。

只按声明的first Q scale和int8 round→clip公式做CPU模拟，HL使wide确认输入512个值中的467/469个被clip；LH的wide输入第一处clip为0、outlier为2。它不是物理指令或内部所有stage的饱和counter；即便first Q无clip，后续范围/量化误差仍可能使最终Y失败。

两进程共320份最终Y、163,840个元素，240通过、80拒绝；另保存16份**已插桩观测的**临时校准预测及其全部输入/输出。还有未插桩重复校准，不把16冒充进程全部临时调用数。不发布延迟/速度比，不用未过门槛输出做加速证据。

## 6. decompress_weights不等于移除A8或恢复原checkpoint

固定[`WeightDecompressor`](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/optimize/coreml/_quantization_passes.py#L1495)只处理`constexpr_`前缀，materialize解码值，再替为const。它没有删除activation的quantize/dequantize，也没有恢复量化前丢掉的权重信息。

本例解压W8+A8后，weight constexpr由2变0，Conv的3对activation Q/DQ仍在，全部确认Y与解压前相同。matmul的解压结果仍表现为解码后W8权重，不是原FP16 checkpoint的无损还原。LUT/线性低比特表示的解码与整数ALU继续分开看；本批没有运行LUT组合模型，相关已有证据见[权重压缩](weight-compression.md)。

Apple固定[性能文档](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/docs-guides/source/opt-quantization-perf.md#L8)还提醒A8在CPU/部分GPU可能因runtime解码变慢，建议用于主要在NE的网络，并描述A17 Pro/M4的int8×int8吞吐收益。本次小图首选CPU、未计时、没有物理整数证据，既不验证也不否定那条硬件能力声明。

## 7. concat与错误处理：额外两个不能从exit0推导的性质

固定concat分组helper在第一处相交就break。原函数fixture `[[a,b],[c,d],[b,c]]`留下两个仍有交集的group，后续range未全部一致；换成`[[a,b],[b,c],[c,d]]`则合并为一个。独立标准库连通分量两种顺序均得到abcd。**这里只执行原helper和range fixture，没有跑完整concat模型，不宣称所有concat网络已复现错误。**

两个设备进程虽exit0且全部数值可回放，stderr各有20条`TemporaryDirectory`隐式清理的unraisable ResourceWarning：`-W error`在finalizer中并未让主进程失败。证据明确`strict_warning_free=false`，没有全局ignore警告，也不从这些日志直接断言泄漏/设备故障。长期内存/显式资源清理仍未验收。

初版helper call-double缺少`__proxy__`，在原public函数中被拒绝；后来补齐double后14个fixture完成，旧receipt/stderr保留，不将该构造错误说成CoreML失败。独立curator首版语法错误也保留receipt，修正后未重跑设备或改变数值预算。

## 8. 本地入口与复查条件

实际转换/确认客户端、原helper probe、标准库合同、和17个portable tests为原创。

curator不用NumPy/CoreML/MLX，按独立32-bit hash配方重建全部源权重/输入，以math.fsum重算完整两层函数，逐值验证桥接、observer extrema、union及输出，原metrics最大差为0。portable tests只验证tiny数学、协议/metadata/资格，**没有raw文件时不冒充重算opaque设备输出**。

原始models/raw/日志约5.3MiB留ignored output；未删除旧实验或全局cache、改产品默认或reference。重开条件：版本/source/OS/SDK/硬件，pass pattern/opset/融合，校准顺序/域/分组/设备，dtype/scale/布局或真实checkpoint/bias/LoRA/消费者变化。下一步仍需更新版本状态单独核查、真实任务校准与质量、主要NE图的完整性能/物理trace，以及长期资源生命周期。整体研究继续。
