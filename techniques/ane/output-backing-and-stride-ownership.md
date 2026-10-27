# 公共Core ML输出交接：backing建议、stride、返回对象与代际所有权

<!-- knowledge
id: technique-ane-output-backing-and-stride-ownership
type: method
operator: layout, scheduling
backend: coreml, cpu-reference
device: cpu, ane
symptom: wrong-layout, stale-cache, lifetime, precision-loss, measurement
semantics: numerically-equivalent, contract-only
representation: fp32
boundary: producer-consumer
api: public, reference-only
-->

[ANE目录](README.md) · [固定源码/SDK](sources.md#output-backing-sources-20261016) · [ORT分区实验](onnxruntime-coreml-partitions-and-cache.md)

2026-10-16，M4 Max、macOS26.6.2/build25G83、SDK26.5、clang21、既有coremltools8.3.0/NumPy2.0.2。四个独立进程，seed17/29×CPU_ONLY/CPU_AND_NE，运行原创公共API客户端及自有小模型。**108份完整输出中12份是接受错误dtype后产生的数值负例，不是108份原模型语义全部通过。** 未运行原vpipe/ccv/ExecuTorch wrapper，没有GPU消费者、物理ANE、能耗或速度资格。

## 可复用的方法：安全复用输出缓冲区

1. 把 output backing 当建议；预测后检查真正返回的对象、dtype、shape 和作用域回调提供的 stride。
2. 用 padded/permuted 布局、poison 和完整下游读取验收；API 成功及同 pointer 不代表语义正确。
3. 为每个复用 slot 记录 generation 和最后读者；旧 result 若共享存储就不是快照，需要快照时显式复制。

可用于减少重复分配和交接准备，但不据此宣称 driver 内部零拷贝；实际节省需按完整边界计时。

## 1. 请求backing不是交付证明

本机SDK的`MLPredictionOptions.h`将outputBackings称为提议：某些模型不支持，框架可以分配别的输出，应检查返回对象身份。`MLMultiArray.h`还要求scoped pointer只在回调内有效；mutable回调可能更换backing/stride，应使用回调给出的strides，而非缓存旧属性。来源JSON记录两份header的完整SHA和实际选段，不镜像Apple正文。

公开导航：[outputBackings](https://developer.apple.com/documentation/coreml/mlpredictionoptions/outputbackings)、[MLMultiArray](https://developer.apple.com/documentation/coreml/mlmultiarray)。本批网页正文未完整取得，以本机固定SDK注释和实际调用为证据，不称全文网页审核。

要分别回答：

1. 是否接受这个参数、返回对象是否等于所提议对象？
2. scoped pointer是否指向自有region，shape/dtype/stride是否仍匹配？
3. 所有逻辑元素是否属于当前generation，padding/tail是否未被破坏？
4. 下游能否正确读取，旧result引用是否只是同一可变storage的另一个owner？

即便以上都通过，也只证明所测外部接口采用了该region；不证明driver内部没有staging，也不证明ANE执行或整个系统零拷贝。

## 2. 同一个小图的八种控制

自有MLProgram：FP32输入/输出描述`[1,4,3,5]`，`Y=2X+[0,.25,-.25,.5]`逐channel相加。所有值是明确dyadic，可独立完整重算。每case三代输入；backing每次先poison，两个page的自有allocation采用page对齐。预测不与另一个设备/线程竞用buffer。

| 模式 | 本机实际结果（两seed/两策略一致） |
| --- | --- |
| 不提议backing | 返回框架分配的FP32，完整正确 |
| 连续`[60,15,5,1]` | 返回同一个MLMultiArray与自有pointer，完整正确 |
| padded `[128,32,8,1]` | 接受并保留非紧密stride，完整正确；padding/tail不变 |
| permuted `[60,15,1,3]` | 接受，返回非C-order布局；按实际stride读完整正确 |
| 未知feature名`not_y` | 初次预测返回error；新options去掉backing后同一输入正确 |
| batch，两个相同输入provider | 带backing初次报“不支持batch”；无backing重试仍为两个输出，均正确 |
| 错shape `[1,4,3,4]` | description检查为false，初次预测拒绝；无backing重试正确 |
| INT32 backing，模型描述仍FP32 | description检查为false，但初次预测成功、返回同对象/同pointer的INT32，实际值符合逐元素向零截断 |

未知名和batch两项不能只照抄SDK注释中的“忽略”：**本机这个模型/OS实际返回错误**。这不是所有模型/所有系统的永久规则；保留初始NSError和完整重试输出。stateless小图的重试不证明stateful模型可以无条件重复执行，后者需确认首次失败是否已修改state。

INT32案例更重要：返回值仍不满足输出description的`isAllowedValue`；所有值finite，最大相对原FP32函数的绝对误差为.75。不能在成功后把这些整数转换回FP32，就称原模型输出正确。这里没有使用异常的raw字节reinterpret来制造错误：按返回的真实INT32读取，再独立对照原函数与截断函数。API成功、identity成立与语义正确是三个门槛。

本批合计36次初始error、36次显式重试；batch重试保留原来两个请求的cardinality。108份实际输出共6,480元素，另有84份提议backing的完整payload核查；未完成输出的初始error不伪造数值或设备执行计数。

## 3. 旧result不是快照；owner释放也要看阶段

客户端强持有第一代返回的MLMultiArray，然后复用同一自有backing执行后两代。采用backing的case中，**第一代对象后来读到的是新一代数据**。这是明确的caller复用，不是异步race或Core ML擅自回写独立快照；保留对象只保住allocation生命，不冻结内容。

另在第一代调用公共`transferToMultiArray:`复制到独立storage，其完整hash一直保持第一代。每次实际返回值也通过该API传到FP64 padded消费者，再按实际stride读完整结果；108个消费者hash全部与对应实际输出一致。INT32负例在消费者中仍是错误的原模型结果，copy不修复语义。

96个held-first检查与96个独立snapshot检查都可离线重算。每次预测后padding/tail均不变；所有case和model的autorelease pool释放后，每进程七个自有backing都触发deallocator，共28个。case内观察到的全局deallocator计数可能包含前一个case延后释放，**不按case计数推断model内部owner**，也不在未知完成时强行free。

本批不涉及CVPixelBuffer/IOSurface-backed MLMultiArray。SDK对这些对象另有预测前不要通过data方法锁定pixel buffer的限制，不能把本次heap region的poison/读取流程不加区分地复制过去。

## 4. 四个开源实现怎样交付输出

所有项目均是固定源码选段，未构建/运行这些原wrapper。本表与原创API实验分级记录：

| 项目 | 所读路径的合同与局限 |
| --- | --- |
| [vpipe manager](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/apple-silicon/coreml/coreml-model-manager.cc#L676) | 先检查固定shape/dtype/capacity，设置bound；预测成功后bound分支跳过实际输出回读。若框架接受但未采用，便需要额外identity检查/复制。这是条件风险，本批没有复现该条件在原manager内发生 |
| [ccv rowwise bridge](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_coreml.mm#L1131) | 提议pixel-buffer-backed output；读取返回值，核对可读及总字节数，但所读路径不比较预定surface身份，也未在不同时copy回scratch。原bridge未运行，不能把heap控制称作其IOSurface bug复现 |
| [ExecuTorch manager](https://github.com/pytorch/executorch/blob/3f15285f3b668b0ecdac6fa985bc207edeb50efc/backends/apple/coreml/runtime/delegate/ETCoreMLModelManager.mm#L85) | 比较scoped buffer pointer；不同则按实际layout/dtype复制。初次带backing失败会设置executor的ignoreOutputBackings并重试，后续调用可持续走copy；不等于每次都会重新尝试backing |
| [ORT1.31 model](https://github.com/microsoft/onnxruntime/blob/402979ac1bef89dcd55add41eaa7277c407599eb/onnxruntime/core/providers/coreml/model/model.mm#L620) | 所读Predict不提议backing，取真实返回值再复制到ORT tensor；支持限定的单gap块式stride，多个非连续维度明确拒绝，不是任意view gather |

ExecuTorch的same-dtype prepareArgs可包装caller原storage，dtype不同则创建匹配constraint的临时MLMultiArray。copy helper使用mutable回调传入的目的stride。这比仅凭相同指针更接近完整交付，但其pointer相等快速路径仍依赖caller的相同布局/类型前置条件；本批没有测试恶意同pointer不同view、动态resize或stateful失败重试。

独立CPU重建ORT的copy-info公式后，本次padded/permuted布局均越过其单gap限制。**这不表示实际ORT图返回了这些布局后出错**：没有执行原函数或强迫ORT返回它们，只说明不能把本次通用stride消费者的能力直接赋给该wrapper。

## 5. 原始model字节与语义身份也要分开

最初离线测试误要求四个`.mlmodel`的raw SHA相同，实际有三个hash，测试失败。用只解析protobuf的独立客户端逐一检查原始model的IO、const/mul/const/add、常量和连接后，四个完整parsed message做deterministic serialization得到同一个SHA。差异来自非规范化的序列化顺序，而非本批图/权重改变。

保留四份raw hash，没有用规范化hash覆盖原文件身份。这个有界message比较不是通用模型语义等价算法，也不把hash相同当作native framework/image attestation。初始测试错误及修正原因保留在验证记录。

## 6. 复现、独立审计与未完成项

重建所有输入、返回输出、提议backing、held result和独立snapshot的hash，明确保留12个INT32数值负例。再读raw完整数组，检查source/binary/model/监督receipt及原始源码身份。9个离线测试不需要Core ML、NumPy、产品库或兄弟源码。

可选native客户端与显式driver不安装依赖、不下载模型、不清全局cache。仅编译原创公共API客户端及自有模型到全新owned目录，留256MiB reserve：

两个pilot保留：首次是本client错误的types import，未进入模型/native执行；第二次是客户端reader不接受实际INT32而停止，且旧batch重试错误地降为single，只作失败历史、不纳入108份主试。随后分别修正reader和重试cardinality，四个主试新目录另跑，没有覆盖旧结果或上游源码。

尚未完成：真实CVPixelBuffer/IOSurface和GPU/MLX消费者、支持但不采用合法同名backing的原wrapper现场、动态输出、多method/stateful重试、实际完整模型质量与prepare→predict→consume成本。CPU_AND_NE只是许可策略，本批没有compute-plan/device trace，不计算CPU/NE速度比。

重开条件：OS/SDK/source/ABI、返回dtype/shape/stride、input域、同region复用/取消/owner、实际消费者或编译入口改变；独立快照与代际数值门槛不能由链接/目录检查代替。
