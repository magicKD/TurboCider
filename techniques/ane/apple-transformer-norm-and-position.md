# Apple ANE Transformer：LayerNorm、参数迁移与位置编码边界

<!-- knowledge
id: technique-ane-apple-transformer-norm-and-position
type: method
stage: encoder, decode, pipeline
operator: conv, attention, norm, layout
backend: coreml, cpu-reference
device: cpu, ane
symptom: precision-loss, wrong-layout, lifetime, measurement
semantics: numerically-equivalent, contract-only
representation: fp16, fp32
boundary: weight-runtime, producer-consumer
api: public, reference-only
-->

[ANE目录](README.md) · [18文件固定来源](sources.md#transformer-sources-20261017) · 全量合成数组与独立审计 · [验证及后续队列（过程原件已移出）](../MAINTENANCE.md) · [布局概览](layout-and-lowering.md)

2026-10-17批次，固定`apple-aiml-research/ml-ane-transformers@590eb35913622cdef217ebc5329047aab389ac98`，package自报0.1.3；不是新版本状态声明。原参考代码未改动，实际CPU组件与完整小型encoder–decoder、公开CoreML转换/预测均执行；不运行上游setup、单元测试或下载教程的DistilBERT/checkpoint/tokenizer。

结论：BC1S/Conv表达值得借鉴，但**同名参数、finite输出、可选参数存在与“推荐ANE”都不是语义或硬件资格**。36次保存了完整输出的forward（12次原Torch CPU、24次CoreML显式预测）产生的72份encoder/decoder输出全部过独立FP64检查；组件级同时保留12份norm目标失败、4份finite位置编码目标失败，以及明确的拒绝/非有限结果。不把它们混成全图失败或全图都正确的笼统结论。

## 可复用的方法：布局优化连同参数迁移一起做

1. 从标准 Linear/LayerNorm 的数学式推导目标 Conv/norm 的权重和 bias，不能照搬同名参数。
2. 单独覆盖 zero gamma、clipping、position 和调用方输入所有权；同时检查函数语义与副作用。
3. 组件通过后，再检查完整 attention→投影→残差和完整 Transformer；布局正确不等于已证明 ANE 提速。

配合 [布局改写步骤](layout-and-lowering.md)阅读；下文提供参数迁移、完整消费者和有限值负例。

## 1. 真正运行的原图，保留了什么

[原reference实现](https://github.com/apple-aiml-research/ml-ane-transformers/tree/590eb35913622cdef217ebc5329047aab389ac98/ane_transformers/reference)以BC1S组织Conv1×1投影，按head做QK/softmax/AV，再经输出投影、残差、channel LayerNorm及ReLU FFN。这里的FFN不是SwiGLU，不能把其参数或非线性直接替换现代DiT/LLM。

本例B=2、C=8、heads=2、FFN隐藏16、source长度7、target长度5、encoder/decoder各2层，dropout=0且eval。参数按完整确定性hash配方逐元素填充，输入三代更新，padding及decoder causal mask显式非零；不是identity、预训练权重或真实语言质量测试。

优化机制在于**表达与数据流，不是减少模型语义**：Conv1×1保留每token线性投影但把channel交给卷积通道；每head的score维持`[B,K,1,Q]`，softmax固定key轴，避免把所有head一起升成更高rank的中间图；Q/V split及K transpose尽量沿用同一BC1S接缝。所有head及完整K/V仍须计算，没有剪掉token或attention工作，也不是FlashAttention式不落完整score的算法。原Python先构造各head score列表，编译器的liveness/融合/实际copy和峰值必须另测；不能仅把单head张量尺寸缩小解释为同倍物理内存节省。CoreML SD的query chunk是另一维切分，参见[原Split-Einsum边界](../gpu_ane/case-studies/apple-split-einsum-tail-and-mask.md)。

| fixture | encoder/self/FFN | decoder cross | 最终decoder输出 |
| --- | --- | --- | --- |
| post | 原ResidualSelfAttention/ResidualFFN，残差后norm | 原ResidualMultiHeadAttention，残差后norm | `[2,8,1,5]` |
| pre_self_ffn | 原PreNormResidualSelfAttention/PreNormResidualFFN | **仍是post-norm cross**，原树没有本fixture可直接替换的pre-cross类 | 原`return_intermediate`堆叠两层，`[2,2,8,1,5]` |

两种都经过原decoder最终norm，不将`pre_self_ffn`叫作“整个模型全pre-norm”。每模型实际86份参数tensor覆盖32个Conv投影和11个affine norm；独立oracle也重做encoder、decoder self/cross、FFN、残差和中间输出，而不使用Torch输出充当CoreML的唯一参考。

## 2. 同名bias不等价：迁移hook的数学域

[原LayerNormANE](https://github.com/apple-aiml-research/ml-ane-transformers/blob/590eb35913622cdef217ebc5329047aab389ac98/ane_transformers/reference/layer_norm.py#L46)沿channel计算population variance：

```text
z = (x − mean_channel(x)) / sqrt(mean_channel((x−mean)²) + eps)
source affine = (z + b_internal) * gamma
standard affine = z * gamma + beta
```

[DistilBERT原pre-load hook](https://github.com/apple-aiml-research/ml-ane-transformers/blob/590eb35913622cdef217ebc5329047aab389ac98/ane_transformers/huggingface/distilbert.py#L26)用`b_internal=beta/gamma`换序。本批只从固定Git blob提取这个完整函数，使用真实Torch tensor调用；**没有import HuggingFace模块、实例化DistilBERT或执行其完整load_state_dict链**。Linear→Conv的另一hook只读源码，不借这批实验称已执行迁移。

两seed的nonzero gamma控制中，正确换序输出与标准LayerNorm目标相符；直接按名字copy bias仍能执行、输出全部finite，但最大abs误差约.75。它与**原内部affine公式**的误差又小于1e-6：这是参数迁移目标错了，不是后端执行该公式错了。

零gamma控制更直接：一个通道beta非零、一个beta为0，原除法产生inf和nan，load没有拒绝，最后每seed有28个非有限输出。gamma=0、beta≠0时，不存在有限`b_internal`能用该乘法形式表示目标常量；beta=0时本可选有限bias，但该hook的0/0仍会产生nan。采用前需要显式处理参数域或保留不同affine表达，不是简单把nan清零。本例不声称实际checkpoint含零gamma。

rank3 BSC输入会被原层transpose+unsqueeze到BC1S，**输出仍是BC1S**，并不自动恢复BSC。无affine、无clip控制通过；这类返回布局也应写入consumer合同。

## 3. clipping可以防溢出，也会改变输入owner和模型函数

原代码使用`inputs.clamp_`。BSC分支的transpose/unsqueeze是view，真实CPU实验确认原调用方输入随之被修改。clip=.5后完整输出符合“先clip再norm”的参考，但相对原未clip目标误差达.74–1.10；若另一个残差/消费者还要使用原输入，这个副作用尤其重要。这里没有泛化为所有训练/autograd行为。

半精度诊断全部是**原PyTorch层的CPU F16软件计算**，不是ANE累加器/硬件指令实验。输入每channel为±256/512/768/1024：

| F16控制 | 原层实际结果 | 相对原FP64 LayerNorm目标 |
| --- | --- | --- |
| 不clip，eps=1e-5 | 中间平方溢出，最终112个值全为有限的零 | relativeL2=1，不能仅靠finite gate通过 |
| clip=16，eps=1e-5 | source FP16逐操作重放吻合，输出finite | 原目标已改变，maxabs约.635，仍拒绝 |
| 常量输入1，eps=1e-12 | epsilon在这条F16表达中下溢，输出112个nan | 拒绝 |
| 同常量输入，eps=1e-5 / 1e-7 | 输出0，完整通过 | 有限正控制，不证明所有近常量输入无误差 |

本库另写binary16每操作round/overflow/rsqrt的标量模型，完整重现这些software-half输出类别和有限值，不将CPU Torch internals包装为ANE ISA。coremltools可融合/改变中间精度，所以不能以这些CPU负例断言转换后的所有ANE norm也会如此。

参考LayerNorm默认eps=1e-5，DistilBERT适配源码选择1e-7并注明其原模型用1e-12。**改eps与clip都改变函数**；是否可接受需完整checkpoint/任务预算。本批公开全图使用原default eps、无clip、非零gamma以及显式FLOAT32图，未测试这些F16负例的CoreML迁移。

## 4. 可选vpos参数存在，不代表实现按它的值计算

[原MultiHeadAttention](https://github.com/apple-aiml-research/ml-ane-transformers/blob/590eb35913622cdef217ebc5329047aab389ac98/ane_transformers/reference/multihead_attention.py#L163)在检查`vpos is not None`后，实际相加的是**kpos**，不是vpos。

两seed均用B2/C8、Q5/K7及非零q/k/v、不同位置编码、原始Conv参数执行：

- qpos/kpos正常、vpos=None：完整输出与softmax weights正确。
- 同时提供不同kpos/vpos：完整结果符合错误的`v+kpos`参考，不符合参数语义的`v+vpos`；四份输出仍finite，maxabs误差约.00206–.00401，大于预先固定2e-5门槛。
- 只改变vpos内容：原输出完整字节不变，独立确认它没有消费vpos的值。
- 仅提供vpos、kpos=None：实际TypeError，没产生输出。

这是固定ref的原函数级可复现缺口。**原encoder/decoder调用点只传qpos/kpos、不传vpos**，所以本批完整小模型通过与这个组件负例不矛盾；不称默认DistilBERT、原文章结果或所有转换模型都受影响。没有改上游、提交fix/PR或检查其他commit的修复状态。

## 5. mask的dtype、方向、finite sentinel与空行策略

原mask ABI为qk=`[B,K,1,Q]`、k=`[B,K,1,1]`，强制float32；softmax沿key轴dim1。两seed均实际拒绝bool qk mask及交换Q/K后的错误shape，各没有输出。不能把PyTorch其他attention API的bool语义或BHQK布局直接搬入。

用零Q/K投影分离数值归约影响，所有key都加−1e4时原softmax是完整uniform `1/7`，输出不是零；它**正确执行了finite additive-mask公式，却没有实现零empty-row policy**。不能把所有finite低值当作真正移除key，也不能简单把这个对照叫作原函数计算bug。

改为所有key加−inf时，原CPU代码输出80个nan，softmax weights另有140个nan/seed；没有静默替换为0。本例保留两种行为，提醒接入方先决定空行应拒绝、跳过、返回零还是保留NaN，再验证消费者。有限mask与−inf的空行结果不同，不互作无损替代。

## 6. 完整公开CoreML验证：成功转换不等于真的用ANE

四份原图trace（两模式×两seed）按相同shape、FLOAT32 precision、macOS15 minimum target导出、compile一次，再分别用`CompiledMLModel`加载CPU_ONLY和CPU_AND_NE。每route三代完整输入；每次encoder/decoder以及两层intermediate（相应模式）全部收集，不只拿末token或一个norm数值代表全图。

独立stdlib oracle使用完整输入/参数、scalar FP64点积/稳定softmax/population norm，逐tensor重算所有输出；另从hash配方重建全部参数与输入，核对八个输入的shape/dtype、精确padding/causal mask、代际与86份参数身份。它不调用Torch/CoreML作为数学oracle。

36次已捕获forward→72份完整全图输出全部满足maxabs≤2e-5，最大abs约`5.3311e-7`；其中24次为真实公开CoreML显式预测。再加20份norm、10份attention输出及10份完整softmax weights，共检查12,792个输出/attention权重元素。这里仅计明确持久化的顶层fixture结果：Torch trace/check-trace还会额外运行原模型，未单独保存/计数；全图内部的norm/attention调用也不计成独立fixture。API调用数更不是物理dispatch数。故意错误和非有限组件结果单独分类，不被“实验进程exit0”洗成数值通过。

两种完整图的public MLComputePlan分别有516/523条operation记录；ALL和CPU_AND_NE下，已知preferred的162/164条均为CPU，其余多为constant/unknown。公开capsule保留每个完整plan的规范化JSON hash及计数摘要；逐op记录在ignored raw。**计划仍是预测；这些小图既没有物理ANE证明，也没有性能比较资格。** 没有把CPU_AND_NE这个名称当作实际ANE执行，亦未跑同预算GPU baseline。

## 7. 上游benchmark、依赖版本与trace警告的边界

[原test_transformer](https://github.com/apple-aiml-research/ml-ane-transformers/blob/590eb35913622cdef217ebc5329047aab389ac98/ane_transformers/reference/test_transformer.py)用ALL与CPU_AND_GPU比较、`rough_timeit`做13次host预测wall；调用内还包含输入字典/NumPy转换。10×是源码的sanity常量，低于时是logger.error，不能当本机通过的硬性速度断言或物理ANE时间。测试中PSNR目标、作者文章历史设备数据和本批2e-5全输出标准各有不同作用；这里**没有运行上游benchmark或复现其历史倍数**。

实际重用了已有CPU-packages的Torch2.5.1和项目既有NumPy2.0.2/CoreML8.3.0，未安装新依赖。它不满足这个旧setup.py的Torch<=1.11约束；8.3还明确warning：2.5.1未在它的tested列表里。每主进程另有8条TracerWarning，bool/shape分支被固定；因此只验证当前静态shape与分支，不授予完整package支持、动态shape/generalization或更新版本资格。warning和stderr哈希保留，不过滤掉后宣称无警告运行。

## 8. 复算、失败历史与继续问题

显式原代码运行参数为`--run-original --reference ../references/ml-ane-transformers --torch-packages EXISTING_CPU_PACKAGES --seed 17 --output-dir NEW`；另加`--device`才转换/运行公开CoreML。必须用supervisor和新owned output、已有固定依赖及2GiB磁盘reserve，不执行README的from_pretrained、setup/测试或全局清理。仅原reference最小模块和一个selected hook被执行；上游代码、trace、模型包、compiled资产与库仍留ignored outputs。

raw审计入口：

保留三项本库观察器/测试错误：首轮将mlmodelc交给MLModel而非CompiledMLModel，compile后、prediction前失败，原CPU部分仍独立核验但不并入主计数；修正API的一次缩进错误在执行前中止，确切失败source已存档；首个portable test错误要求F16 clipping结果等于FP64含eps公式的误差恰为0，实际约1.95e-8，改为1e-7数值一致性断言，**没有更改原目标2e-5 gate、半精度逐操作重放或失败分类**。不是上游/设备故障，也未通过重跑隐藏。

后续：真实checkpoint及任务质量、完整HF迁移/异常gamma域、F16/混合精度导出和clip/eps消融；更大图/长度/batch与实际placement、优化GPU基线及完整成本；dropout/train/autograd/动态形状和并发/资源/取消。SoC/OS/SDK/compiler、Torch/CoreML、源码/参数/shape/mask/pos/eps/clip/consumer变化均重开。原参考层只是推理图表达教材，不是本批已实现ANE训练、真实语言模型或性能突破。
