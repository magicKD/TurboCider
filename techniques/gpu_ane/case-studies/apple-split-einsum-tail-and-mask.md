# Apple Split-Einsum V2：原始CPU执行发现的query尾部与mask边界

<!-- knowledge
id: technique-gpu-ane-case-studies-apple-split-einsum-tail-and-mask
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [公共pipeline生命周期](apple-coreml-pipeline-lifecycle.md) · [来源](../sources.md#apple-coreml-pipeline) · 全部摘要/失败

2026-10-17，固定`apple-aiml-research/ml-stable-diffusion@ea2805dc1945be20561c77e5f6d1d9a5a637cda2`。本轮用真实PyTorch2.5.1、显式CPU/F32 tensor、单CPU线程执行**未改动的attention.py原函数**，不是独立数学模型冒充上游运行。没有模型权重、CoreML转换、Metal/ANE执行、图像质量或性能benchmark。

## 为什么这种图表达值得学习

`SPLIT_EINSUM`把Q/K/V保持为BC1S式4D表示，按head拆开，用两个einsum、沿key维softmax再合并通道；上游注释/README将它作为ANE候选、ORIGINAL作为部分Mac GPU候选。它改变的是导出图表达，不是运行时切换一个已有compiled模型的按钮。

V2进一步按512个query切小块，每块仍看全K/V；数学上query独立，合法分块可减小单块score形状。但源代码把每head每chunk的score先构造成Python列表，后续才softmax/AV，本轮没有编译器活跃区间或设备peak证据，不能把单块大小直接等同实际峰值缩小比例。SDK、compiler、shape及模型均要实测。

## 固定版本有两个可复现的函数级缺口

`attention.py:79–143`取`num_chunks = query_seq_length // 512`，仅在num_chunks==0时fallback V1；否则只切固定完整块，没有尾块。这个fast分支也没有把mask加进attn_weights；短query fallback则仍使用V1 mask。

| Q / mask | 原始V2实际行为 | 本次判定 |
| --- | --- | --- |
| 127 / 无或有 | fallback V1，保留127个query并处理mask | 正向对照 |
| 512、1024 / 无 | 完整块，数值与独立FP64 oracle一致 | 正向对照 |
| 513 / 无 | 返回shape `[1,8,1,512]`，少1个query | 形状错误 |
| 768 / 无 | 返回shape `[1,8,1,512]`，少256个query | 形状错误 |
| 512、1024 / 有 | 输出shape完整，但fast分支忽略mask | 数值/语义错误 |

保留下来的query前缀仍可数值正确；这并不使丢掉尾部的完整输出正确。下游可能在reshape/residual处报错，本轮没有复现“生成图片被裁掉”或默认模型生成失败。

`unet.py:45–59`确实会按枚举调用这些函数；`torch2coreml.py:1526–1530`设置该全局选择。但所读`BasicTransformerBlock:586–590`正常调用没有传mask，常见静态resolution还可能恰好全层query都满足整除条件。因此mask反例说明的是函数不能任意迁入masked模型，**不代表默认SD图当前一定用了mask或会失败**；tail反例也未完成完整UNet/导出验证。

## 真实执行、独立oracle与正负结果

固定源文件先核完整Git tree blob才import；probe不导入UNet/diffusers/CoreML模型，只加载原始attention模块。两个确认进程分别seed319和337，第二个反序执行8种fixture。B=1、heads=2、dim_head=4，非mask用K=8的dyadic随机输入；mask用K=Q的self-attention形状，全部CPU float32。

非mask oracle由Python math独立计算点积、稳定softmax及AV，不用torch算参考。mask fixture令Q/K为0，只有被屏蔽的最后一个V key非零、各channel为`(channel+1)*K`，mask对它加−1000；正确答案精确为0，而忽略mask得到各channel1到8，最大误差8。这使语义错误与普通浮点归约误差明显分开。

同一逻辑key遮罩按各函数的einsum布局传入：ORIGINAL为`[1,1,1,K]`，V1/V2为`[1,K,1,1]`；V1和V2使用同一mask布局。这里不声称三函数可以不经适配共用所有mask ABI，也未运行CrossAttention wrapper的全部masked模式。

两进程合计16个用例观察（8种fixture重复两轮，不称16种独立shape），48次顶层函数调用；V2内部fallback不另外算成设备调用：

- ORIGINAL和V1各16/16正确，最大正向绝对误差约`1.37e-7`，门槛`2e-5`。
- V2为8次正向通过、4次tail形状错误、4次mask语义错误；不将负结果合并成“全部通过”。
- 两份raw JSON保存所有Q/K/V、全部输出、oracle及分类，共4,764,772 bytes，hash绑定于分发摘要。没有仅抽样几个值代表全量输出。

probe代码验证所有output都在CPU、F32且finite，不调用MPS API；两个实验终态receipt均exit0，是因为观察符合预先声明的正/负控制合同，**不是因为V2所有用例正确**。没有提交本应用GPU工作；也不把import PyTorch等同于证明整个进程从未加载任何Metal/MPS动态库。

## 保留的首轮失败：不要把oracle门槛问题藏掉

最初mask正对照让所有未屏蔽V都为非零常数。512-key下ORIGINAL/V1的FP32归约相对理想常数出现约`2.670288e-5`绝对误差，超过本库`2e-5`门槛；首轮和诊断进程因此exit1。

这不是原函数throw/crash，也不能仅据该任意门槛认定整个原attention不可靠。旧probe两个版本、stderr、terminal receipts全部保留。后续**没有放宽门槛**，改为只有blocked key非零、正确答案精确为零的分离式mask对照；换seed/反序确认仍得到V2误差8。记录更换fixture的原因及结果，不把探索失败抹成最终顺利故事。

## 采用和修复前的验证合同

对本ref的V2，函数级数学前置条件至少是：短于512时走V1；长query必须整除512，且不能假定fast分支处理mask。接入新shape/模型时应先拒绝不满足条件的配置或选择已验证路线，再验证全部输出；不要把“推荐ANE”当广泛能力声明。本轮没有给上游打补丁或提交PR。

后续若补tail/mask，需覆盖Q=511/512/513/768/1024等边界、self/cross attention、mask广播、dtype/heads、完整UNet残差与CoreML导出/设备数值/peak，再比较同输入GPU/ANE优化基线。PyTorch CPU通过不证明ANE compiler会保持同一图语义，更不证明更快。

上游README的移动设备10–30%等描述是其特定历史模型/OS/压缩环境报告；本轮没有重现，也不迁成M4/M5的倍率。LUT palettization、FP16图表达与ANE整数MAC仍是不同证据轴，见[pipeline报告](apple-coreml-pipeline-lifecycle.md)。

## 复算与依赖边界

实际原函数重放需按[来源](../sources.md#apple-coreml-pipeline)取得同一attention.py/tree元数据，并准备本轮固定依赖，再用probe的新raw-output/summary路径在有界supervisor中执行；完整argv在执行数据。PyTorch2.5.1及8个固定小依赖从PyPI wheel装到本轮ignored `cpu-packages`（约437MiB），没有修改项目/系统Python或下载checkpoint。pip报告的Torch wheel SHA与官方PyPI元数据一致；没有保留完整wheel做独立二次归档哈希，边界明确记录。

原始代码、二进制、依赖和raw数组只留`outputs/gpu-ane-coreml-pipeline-round19-20261017`，不进轻量Wiki。涉及CoreML/ANE/GPU的后续设备实验仍需先过GPU空闲guard，不能直接重用本CPU结果授予设备资格。

固定源码：[V2函数](https://github.com/apple-aiml-research/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/python_coreml_stable_diffusion/attention.py#L79)、[真实dispatch](https://github.com/apple-aiml-research/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/python_coreml_stable_diffusion/unet.py#L45)、[默认无mask调用](https://github.com/apple-aiml-research/ml-stable-diffusion/blob/ea2805dc1945be20561c77e5f6d1d9a5a637cda2/python_coreml_stable_diffusion/unet.py#L586)。隔离验证见[第十九轮](../../MAINTENANCE.md)。
