# NNC packing 签名：反驳角色交替必然重编译的假设

<!-- knowledge
id: technique-algorithms-case-studies-nnc-packing-signatures
type: case-study
stage: dit
operator: sampling, layout
backend: metal, cpu-reference
device: gpu, cpu
symptom: launch-bound, stale-cache, wrong-layout, measurement
semantics: contract-only
representation: fp32
boundary: producer-consumer
api: public, private, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [gate成本](nnc-gate-controller-cost.md) · [实验入口](../../examples/README.md) · [S58](../sources.md#s58)/[S59](../sources.md#s59)

2026-10-17，第十九轮。在固定NNC CPU和MPS GPU上，分母norm与分子difference交替进入同一个packing Model，完整输入参数保持相等；**没有证据支持把上一轮波动归因为这种角色切换触发参数重编译**。改变几何的正对照确实检测到mismatch，所以不是观察器永远返回match。这里不新增性能、模型或媒体结论。

## 为什么要补这一项

[第十八轮](nnc-gate-controller-cost.md)发现缓存分母比原串行gate快，却不稳定胜过批量读回强对照，并留下一个未测混杂因素：同一packer用于commit norm与query difference，会不会因隐藏metadata变化而反复走compile/absorb？不能仅凭源码存在`memcmp`就把假设写成实际发生的性能原因。

固定ccv `66dc16a` 的`ccv_nnc_dynamic_graph_evaluate`先检查`model->graph`；cold时调用compile，warm时比较所有输入的完整`ccv_nnc_tensor_param_t`。mismatch会进入compile；已有graph的compile实现再copy/absorb。这里观察的是这条**参数分支的前置条件与调用后的保存签名**，不是所有动态abs/reduce操作、MPSGraph计划、shader/pipeline缓存或真实compile函数调用次数。

## 只读观察边界

C适配器包含该ref的私有`_ccv_cnnp_model.h`，读取`graph/inputs/input_size`，再用公开symbol getter取得已保存输入参数；incoming用公开variable getter。两个getter的定义均读取参数，不触发执行。每份描述导出16个有名整数：`type,format,datatype,reserved,dim[0..11]`，而非任意内存dump。编译期检查int32大小、12维及完整struct为64字节；不导出指针、opaque padding或模型数据。

Swift客户端只为取variable句柄反射固定`underlying._tensor`，Model句柄来自公开`cModel`。使用真实NNC Tensor/Model与原第十八轮的abs/mean、fixed-arity packing/readback helper；未修改上游、未安装compiler hook或关闭断言。反射与私有布局**不是稳定API**，换ref/编译器必须重新核查，不能作为生产实现建议。

每次packing前读incoming和保存的before，真实执行并读回全部scalar后再读after。标准库整理器不信任native的状态标签：独立重建所有16-word描述、每个model的完整时间顺序、期望cold/match/mismatch及after；另外用独立FP64配方验证全部统计量。

## 对照与结果

同一最终binary，M4 Max/64GiB、OS25G83、Swift6.3.3、SDK26.5；三组`(audioRows,videoRows,channels)`为`(2,256,64)`、`(8,1024,128)`、`(16,4096,256)`。输入沿用原创signed pattern、current为anchor的`65/64`倍，非真实H3 activation。每几何六周期；每周期一次commit和三次query，four-stats仅三次query。

下表是**每设备**的实际packing观察行数，CPU与GPU逐项相同。列名描述调用前状态，不能读作直接编译计数。

| 策略 | packing调用 | cold：无保存graph | warm：参数匹配 | warm：参数不匹配 |
| --- | ---: | ---: | ---: | ---: |
| shared-packer：norm/difference共用 | 72 | 3 | 69 | 0 |
| split-packer：两个角色分开 | 72 | 6 | 66 | 0 |
| four-stats：一次打包四统计量 | 54 | 3 | 51 | 0 |
| cross-geometry：一个packer跨三几何 | 3 | 1 | 0 | 2 |

合计CPU201+GPU201次packing、1020份完整incoming scalar描述和1020个输出scalar参考，观察到的最大统计绝对误差为0；每次after均等于incoming。这里没有逐字节重建输入bank，也没有生成轨迹或画质测量，不借用上一轮输入SHA当作新数据的证明。

以最小几何为例，两个角色的CPU描述完全相同：

```text
audio: [1,2,16384,0, 1,1,1,0,512,0,0,0,0,0,0,0]
video: [1,2,16384,0, 1,1,1,0,65536,0,0,0,0,0,0,0]
```

GPU仅此例的memory-type首word变为2。所有reduced scalar可见维度都是`[1,1,1]`，但终止0后的`dim[4]`仍带producer row-step：三几何的audio/video分别为`512/65536`、`4096/524288`、`16384/4194304`。同几何角色切换不改它，跨几何则改变；正控两次mismatch后，保存签名各自更新到incoming。**不能因此把完整参数比较降为可见shape比较。**

split多出三个cold状态，是三个几何各有第二个model的结果；没有测量这些cold的耗时，不能宣称split更慢或更快。本轮没有计时臂、重复跑分或选择新赢家。第十八轮的强对照负结果和配对估计原样有效；本轮只排除一个特定的签名交替解释，不证明所有隐式规划都不存在。

## 准入、失败与重放

GPU launcher在提交前重新采三次Device≤5%，native在每个几何setup前各采三次，共四个通过窗口/12样本，实际Device均0；记录无argv的进程摘要。没有杀竞争进程，也不把准入当连续独占或CPU空闲证明。CPU和GPU均由180秒bounded supervisor确认自有进程正常退出。

初次编译因自有pattern的一行长表达式超过Swift type-check预算而失败；仅把表达式拆为有类型子表达式，incremental repair成功。加入跨几何正控后用fresh v2 package及独立默认scratch重建；旧CPU prototype不进入最终402行。失败receipt、诊断、log/client SHA及排除理由均在，不是ccv/NNC后端错误的证据。

准备器和启动器均须显式`--run`/新输出目录，后者校验own/generated source及binary SHA；GPU另需`--gpu`。源码来源固定S58/S59，与原S53—S57共同限定依赖；本轮未声称审计完整编译闭包。默认重放只读分发JSON和本地代码：

这两条不编译Swift、不导入可选框架、不读取模型/activation或提交GPU。教学case是原创整数合同，不是假装重跑native。后续若需真正编译次数/成本归因，应另做有调用事件计数的非计时诊断、证明覆盖范围，再用未被观察器扰动的匹配计时；不能把当前402行改名为402次compile。
