# MNN Metal：卷积消费者的C4布局、staging与跨后端成本

<!-- knowledge
id: technique-gpu-ane-case-studies-mnn-metal-copy-and-convolution-abi
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [CoreML图边界](mnn-coreml-graph-boundaries.md) · [源码身份](../sources.md#mnn-boundaries)

2026-10-16，固定 `alibaba/MNN@dbf9ba61b08106a3b43426e295d1a29cb41cc8fe`。本页补充输入输出/分配合同，不重复卷积核、Winograd、量化1×1和replay的已有研究，也不认领其设备实验。没有运行MNN、Metal转换shader或VAE；这里的数值均是独立CPU地址/分配公式见证，不是性能或峰值内存测量。

## 1. C4不是把NCHW的C简单补齐

[MetalBackend.mm][metal]的 `getTensorShape`以逻辑batch N、channel C、plane S生成转换参数；选定 `gNC4HW4Convert`把每四个channel当一个向量。令 `p=h×W+w`，其**标量lane**偏移为：

```text
q = floor(c/4)
lane = c mod 4
offset = 4 × ((q×N+n)×S+p) + lane
```

因此这段Metal转换核的物理顺序是 `[ceil(C/4), N, S, 4]`，不是 `[N, ceil(C/4), S, 4]`。两者在N=1时相同，只测单batch会漏掉差异。此公式描述本读本的转换核，不推广成所有框架、所有MNN后端的“NC4HW4标准”。

CPU反例：shape `[2,5,1,2]`、坐标 `[1,0,0,0]`，正确标量偏移8；假设N在最外层会得到16。即使buffer够大、没有越界，也可能读到别的batch/channel。

学习实现为正4D NCHW写入/读回上述映射，检查多batch、C=1/3/4/5/7/8/9、不同plane和尾通道。logical→C4时未使用的lane清零；C4→logical时不把它们当真实channel。它不执行FP16舍入、卷积、底层GPU访存或任意rank的MNN转换。

## 2. 三种size：逻辑值、packed主体、实际分配

[TensorUtils::getRawSize][utils]只乘逻辑维度；C4主体有 `N×ceil(C/4)×4×H×W` 个lane；[MetalBackend::getTensorSizeInBytes][metal]还为一些kernel的尾部读取添加extraPadding，最后16字节对齐。float32元数据在 `mUseFloatAsFp16` 时按2字节计算物理存储，不能只看逻辑dtype。

对本页限定的正4D NC4HW4 float形状，选定代码的公式为：

```text
Cpad = round_up(C,4)
extra = max(round_up(NHW,16)-NHW,
            (round_up(H,4)-H)×W + round_up(W+3,4)-W)
allocated_bytes = round_up((N×Cpad×H×W + 4×extra)×physical_item_bytes,16)
```

这是源码分配公式，不是理论最小容量。CPU复算示例：

| N,C,H,W | logical元素 | C4主体lane | FP32分配字节 | FP16分配字节 |
| --- | --- | --- | --- | --- |
| 1,3,2,3 | 18 | 24 | 272 | 144 |
| 2,5,2,3 | 60 | 96 | 560 | 288 |
| 2,4,1,2 | 16 | 16 | 256 | 128 |
| 1,1,1,1 | 1 | 4 | 256 | 128 |

表中不是实际MTLBuffer/heap/resident bytes；allocator还可能共享、取整或保留其它资源。不能把主体lane×itemsize当全部子分配范围，也不能把额外padding当业务输出传给CoreML。实际别名/重叠检查需要allocator的真实范围及offset；完整消费者还需验证尾读/尾写。

## 3. 从拷贝API看真正发生的工作

[普通 `onCopyBuffer`][metal]先flush encoder、提交net，再走具体copy分支。下表是该实现的分支，不保证任意外部Tensor都满足它的前提。

| 源/目标 | 选定分支 | 成本与合同 |
| --- | --- | --- |
| 两边都没有host指针 | device→device转换/拷贝kernel | 仍是dispatch，不是指针别名 |
| device→host，不用格式/dtype转换 | wait后从device contents memcpy | shared storage也不能省略完成边界或host copy |
| device→host，需要转换 | GPU转换到host buffer，commit、wait、memcpy | 转换+提交+等待+copy都在消费者链上 |
| host→device，默认queued路径 | CPU memcpy到staging；GPU转换或blit到目标；commit | 免去旧的前置drain不等于免copy |
| host→device，旧/外部encoder等路径 | 按userSync决定wait，再转换或直接memcpy | 用户队列的同步责任不能忽略 |

`needConvert`同时考虑format差异与float的FP16物理表示，`_makeCopyInfo`按输入/输出scalar/vector类型选择copy、C4转换或NCHW↔NHWC。不能只因shape相同就走raw字节拷贝，也不能给一个FP32 host指针贴half标签而跳过舍入/转换。

这解释了为何逐层交替CoreML与Metal可能比封闭子图更差：节约的计算时间要支付多次转换、CPU可见性和提交边界。这里只是成本分解；没有测得交替方案必慢，也没有量化哪种shape的转换足以抵消收益。

## 4. queued H2D的优化点：把“读旧目标”交回队列顺序

在该ref，[MetalEnv][env]默认开启queued H2D，除非环境显式设成0；本轮只读，没有修改开关。源码解释的目的，是把host字节快照放进staging，再在同一command queue里排入staging→dst，利用顺序保证旧GPU reader先完成，免去每次upload前CPU阻塞等待。

**成立的关键不止是queue order，还有staging的owner和复用。** `acquireUploadStaging`寻找容量够、没有lastUse或lastUse已terminal的slot；`markUploadStagingUse`把本次command buffer记为lastUse。在GPU读完之前，即使CPU原始输入已改变，也不能覆写那份staging快照。

两点容易误标：

- “ring”不是这里可证明的固定容量承诺：选定acquire函数在没有可复用slot时增加一个slot，没有固定slot数门槛。真实峰值还需结合in-flight队列与整个allocator测量。
- 复用判据 `status >= Completed`也包括Error。terminal足以表示不再有reader，却不表示上一轮数学结果成功。独立CPU模型明确给Error返回“可复用、prior_command_success=false”；不能用slot释放作为输出正确/消费者可发布的依据。

这些是[选定staging函数][metal]的机制，不是CPU模拟证明Metal内存模型、跨queue hazard或失败后的完整恢复已经正确。没有做设备错误注入、长队列压力或取消/drain测试。

## 5. CoreML↔Metal不能从generic wrapper自动取得资格

[WrapCopyExecution][wrap]在两端都不是普通CPU时准备CPU中间Tensor，再依次调用源和目标backend的copy；但 `needWrap`对**目标NN**直接返回false。它是一套泛用抽象与特殊分支的组合，不是已经验证的CoreML↔Metal零拷贝桥。

CoreML端还有[独立合同](mnn-coreml-graph-boundaries.md)：其普通I/O是host FP32，多数组输出借用MLMultiArray的owner；NC4HW4元数据与实际host dense视图之间由CoreML backend自己的copy适配。跳过该适配，把借来的host字节直接交给Metal并沿用C4元数据，不能凭元数据判断正确。实际返回strides和输出generation也必须先核对。

若设计一个明确的卷积/VAE阶段边界，至少要逐项取得以下证据：

1. producer确实完成，错误状态已传回，而不只是MNN逐op回调已返回；
2. 实际shape/strides、dense/C4、physical dtype与byte offset一致；
3. 借用输出的owner在CPU复制/异步GPU upload取快照之前持续存活；
4. staging直到最后GPU reader完成才可重用；
5. 对端真实使用了这些数据，包含多batch、尾通道、padding和完整输出检查。

这是待验收边界清单，不是一份已经跑通的MNN CoreML→Metal示例。公共 [Metal4 GPU→ML→GPU小链路](metal4-consumer-layout-and-handoff.md)验证的是另一组API、专门化shape和buffer合同，不能替本页MNN路径背书。

## 6. 对VAE实验的实际启示

优先从可封闭且重复利用的卷积块/完整stage研究，而不是让每个Conv在GPU/ANE之间跳转。先给成熟Metal卷积、相同权重/精度、相同tile与halo建立完整消费者基线，再比较CoreML整块预测加所有边界成本。CoreML的all policy可能仍使用GPU，必须另取placement证据，不能把两条API路径当成两套物理设备。

至少分账：构图/编译/load、热预测、format/dtype转换、上传/下载、owner snapshot、同步与完整request。再检查bias、归一化、causal history、tile seam、LoRA/动态权重更新和图像质量。本轮只推进源码/CPU合同，没有结论说CoreML、Metal、Winograd或异构方案谁更快。

本地复算无需MNN、NumPy、网络或设备：

[metal]: https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/metal/MetalBackend.mm
[env]: https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/metal/MetalEnv.hpp
[utils]: https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/core/TensorUtils.cpp
[wrap]: https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/core/WrapExecution.cpp
