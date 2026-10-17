# Headroom缩放的有限错误：原发射器图、down bias与Python桥接

<!-- knowledge
id: technique-ane-headroom-bias-and-bridge
type: method
stage: ffn
operator: gemm, swiglu, layout
backend: coreml, cpu-reference
device: cpu, ane
symptom: precision-loss, measurement
semantics: contract-only
representation: fp16, fp32, bf16
boundary: weight-runtime, producer-consumer
api: public, reference-only
-->

[ANE目录](README.md) · [此前源码/实数反例](vpipe-callchain-and-failure.md) · [固定来源](sources.md#headroom-sources-20261016)

2026-10-16，M4 Max / macOS26.6.2，coremltools8.3.0、NumPy2.0.2。本批把此前的实数反例推进到**原始vpipe C++ emitter发出的真实Core ML图**：受控缩放输入中，非零down bias会产生全部有限、却不等价于原FFN的输出。补偿bd、零bd和GELU对照通过。

这不是“发射器matmul算错”，也不是ANE硬件bug复现：图按缩放后的输入计算正确，问题在host侧缩放/恢复是否保持原函数。本轮**未执行完整AneFeedForward、自动overflow retry、GPU并发、LoRA或真实模型消费者**。所有小图的compute plan首选均为CPU/unknown，CPU+NE许可不改写成物理ANE证据。

## 1. 哪部分真用了原实现，哪部分没有

固定`tgo-app-dev/vpipe@c2989a76841888e4d8d208fad4f476c487710962`，原tree未改动。新小型C++ API client链接原`ane-emitter.cc`，调用`AneEmitter::emit`，生成原格式`model.mil`和`coremldata.bin`；runtime weights不需要常量weight blob。没有重写MIL、调用上游self_test或构建完整应用。

| 层次 | 本批执行情况 |
| --- | --- |
| 原AneEmitter图构造/手工bundle格式 | 原源码直接编译/调用，所生成bundle由Core ML实际加载 |
| runtime weight/bias输入与预测 | 原图，原创Python客户端经public CompiledMLModel提交 |
| 上游stage/retry的缩放规则 | 读原源码，在客户端明确重建source-like变换；不是执行原stage函数 |
| 补偿方案 | 本库实验变体，仅对SwiGLU额外将bd除以s；不修改上游 |
| host输出乘s及BF16截断 | FP32恢复实际用于数值检查；BF16消费规则另由标准库建模，不称原consumer已运行 |
| overflow触发、worker/chunk、output backing、完整模型 | 未执行，不获得资格 |

上游`write_bundle`会先`remove_all(dir)`。本client拒绝所有既存target和symlink，只允许新任务目录内不存在的`model.mlmodelc`叶子；Python也拒绝覆盖output。此次没有删除已有模型、旧结果或系统cache。不可把这个上游函数直接指向任意用户模型目录。

## 2. 被验证的数学合同

SwiGLU的完整函数是：

```text
G = XWg + bg
U = XWu + bu
Y = [SiLU(G) ⊙ U] Wd + bd
```

源码stage对projection index1的权重及bias除以`_up_scale`，retry也在该slot原地乘.25；最终把整个输出乘回s。对SwiGLU，index1是up，所以G不变、U→U/s，**但bd没跟着缩小**：

`s · ([SiLU(G)⊙U/s]Wd + bd) = Y + (s−1)bd`。

本库对照将`bd→bd/s`后再恢复，才得到原函数。`s=1`或`bd=0`时source-like方案本来就应通过，不能把这些控制漏掉。

GELU的source index1则是**down**：`Y=GELU(XWu+bu)Wd+bd`；同时缩小Wd和bd后，整个输出本来就是Y/s。这正是本批GELU对照通过、不能把两类FFN的slot1都叫“up branch”的原因。

关键源码：[bias在最终down之后相加](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/apple-silicon/coreml/ane-emitter.cc#L425)、[stage权重因子](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/generative-models/shared/ane-ffn.cc#L337)、[bias staging](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/generative-models/shared/ane-ffn.cc#L548)、[retry与恢复](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/generative-models/shared/ane-ffn.cc#L740)。完整文件SHA和选段见manifest，不是整个引擎审计。

## 3. 不是只跑一个标量：布局、尾tile和策略控制

主试完整交叉：SwiGLU/GELU × `[in,out]`/`[out,in]` × 两套shape × CPU_ONLY/CPU_AND_NE，共16个进程；另外两个seed23新进程作明确配置的held-out确认。

| family | M,K,N | bk,bn | 覆盖点 |
| --- | --- | --- | --- |
| aligned | 4,32,64 | 16,16 | 两次/多次K归约与N拼接，三投影bias |
| ragged | 3,5,7 | 3,4 | 非整除的最后K/N tile；非方矩阵转置不被方阵掩盖 |

每个run有56个完整预测：三个变化的dense dyadic输入generation，各测biased/zero-bd × s1/4/16/64 × source-like/compensated；另一个generation将X、bg、bu设0，精确隔离down bias。基准及candidate都用相同FP16-origin数据，未下载真实权重。

补偿policy仅改变SwiGLU的bd处理；GELU的source-like本已正确，两个policy的输入一致。这是正对照，不被算成额外优化。没有人为制造overflow，也没有看到原retry被自动触发；这些scale是显式实验参数。

### 精确bias-only样本

在seed17/generation3、column3，bd=.25且hidden contribution为0。实际原图预测及host恢复给出：

| s | source-like原图输出 | 乘s后 | 原FFN目标 | 误差 |
| --- | --- | --- | --- | --- |
| 1 | .25 | .25 | .25 | 0 |
| 4 | .25 | 1 | .25 | .75 |
| 16 | .25 | 4 | .25 | 3.75 |
| 64 | .25 | 16 | .25 | 15.75 |

全部finite。补偿bd后四个scale均精确回到.25。这比“只是数值噪声变大”更明确：偏差与每列bd的符号/大小对应，全部输出的`observed−reference−(s−1)bd`也被逐元素检查。

## 4. 结果如何计数，失败如何保留

18个主试/held-out完整协议有1,008份输出，其中108份是预设的finite semantic negatives；再加一个56份输出的有效pilot，合计**1,064份完整输出、79,240个输出元素、120个finite语义负例**。所有negative都属于SwiGLU、非零bd、source-like、s>1；其他控制均通过。没有把预期语义负例算作崩溃，也没有把“finite”升级成等价。

对每个实际scaled recipe，预先固定relativeL2≤.04、maxabs≤.02；与原函数比较的maxabs门槛同样为.02，未在看见错误后放宽。全部候选对其**实际staged函数**的最大maxabs约.006313、最大relativeL2约.004870；对预测bias差分的最大残差约.006313。120个negative对原函数不通过，正是需要暴露的设计风险。

其中152次bias-only检查没有浮点近似残差。模型化BF16恢复之后，120个语义负例仍被拒绝；这里检查的是source的`float32乘s→截取高16位`规则，不是RNE，也不是运行真实GPU残差消费者。

22个模型任务目录还包括1次emit-only和2次早期输出ABI门槛拒绝。两次拒绝都发生在预测返回之后、完整样本记录之前，不算完整预测协议或模型bias失败；首个只留合并的dtype/shape/finite错误，未保存单独ABI，第二次才记录有限FP32输出供独立检查。没有把后次观测反填为前次已记录的事实；所有旧目录/source SHA保留。

## 5. Coremltools桥接：FP32 Python数组不等于图被改成FP32

首版client要求host输出dtype必须是float16；增加诊断后的拒绝记录显示实际返回finite float32。核对固定coremltools **8.3**源码后确认两项不同的桥接行为：

- [CompiledMLModel.predict](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/models/_compiled_model.py#L219)经过[输入更新函数](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/models/model.py#L995)，会把传入dict中的FP16 ndarray替换为FP32数组。预测后再对同一个feed dict做hash，得到的是**post-bridge**而非原始FP16 bytes。
- [native→Python转换](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremlpython/CoreMLPythonUtils.mm#L449)对Float16 MLMultiArray先转Float32再暴露给Python。不能靠Python dtype推断MIL算子或硬件累加精度。

新协议分别记录predict前FP16 feed SHA、predict后FP32 SHA/dtype/值不变检查，以及host FP32输出SHA和canonical FP16 SHA。只有**逐元素FP16往返完全无损**才接受canonical值；不是悄悄round一次来遮住误差，原数值门槛未变。旧pilot的feed hash只代表post-predict FP32，单独标记，不回填成原先没有观测的pre-bridge hash。

源码tag解析为commit`07ffd860...`，三个文件经只读HTTP取回并保留内容SHA；安装的两个Python文件与此tag内容相同。没有对native wheel或Apple framework做loaded-image/source attestation。源解释与本机观察一致，不把它升级成所有coremltools版本的接口承诺。

## 6. 独立审计与可复用实现

标准库数学/NPY审计不导入NumPy、Core ML或上游包：从固定整数配方独立重建全部base矩阵，核对每个输入的FP16/FP32桥接hash，再用`math.fsum`、exp/tanh重算完整unscaled与staged FFN，检查所有输出和bias差分，不只接受runner的布尔值或误差数字。

GELU参考使用源图的tanh approximation，不偷换成erf版本。BF16 codec另外以标准库bit操作建模并测试，明确区分truncation与round-to-nearest。数据整理入口为；它还核对固定原源码、owned runner/client/binary、两个bundle源文件和监督进程退出记录。

公开JSON可复算inventory/metric摘要和来源身份；要重新审计原始数组，仍需本地ignored NPY目录。原始bundle、模型数组、binary和大段上游源码不随Wiki分发。本批诊断时间没有counterbalancing、完整staging/restore消费者或独占SoC控制，**不计算CPU/NE或IO/OI速度比**。

可选实际图实验，需固定只读reference、macOS/CLT以及已有NumPy/coremltools；不自动clone/install、不运行上游build/demo脚本。默认只emit，只有`--device`才加载/预测：

## 7. 迁移时不要只补一处除法

原emitter的`self_test`是小型常量matmul与bundle格式检查，不覆盖runtime bias、非线性FFN和缩放恢复。自测通过不自动授予这些组合变换的资格，应让测试范围与采用的变换范围对应。

如果决定采用bd/s补偿，需要同时覆盖initial stage、in-place rescale、跨layer restaging和持久scale状态；LoRA/quantized权重与bias来源也要按同一语义恢复。本批只验证这套明确输入域的补偿，未为任何产品打补丁。

仍待验证：真实非零bd模型的自动overflow路径、BF16→FP16输入overflow、极端LoRA/scale域、晚chunk失败、实际output backing/consumer，以及物理ANE与完整请求性能。原方案在bd=0等条件下可能完全适用，不能用此条件反例宣布所有vpipe/所有模型错误。
