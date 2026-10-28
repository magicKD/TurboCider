# 同等四步展开再对照：GPU 分组、ANE 请求路径与计算图身份

<!-- knowledge
id: technique-ane-grouped-training-and-graph-identity
type: method
operator: gemm, scheduling
backend: private-ane, mlx, cpu-reference
device: cpu, gpu, ane
symptom: measurement, launch-bound, compile-bound
semantics: contract-only
representation: fp16
boundary: producer-consumer
api: private, public, reference-only
-->

[ANE目录](README.md) · [前一批四arm窗口](matched-training-windows.md) · [固定源码](sources.md#grouped-training-sources-20261016)

2026-10-16，M4 Max / macOS26.6.2 / MLX0.32.0。本批回答上一批留下的具体问题：**GPU也四步展开以后，私有SGD四步图是否仍然更短？** 结果不是通用胜出：短K4窗口私有路径较短，K16接近或GPU较短，所有K64主试窗口均是GPU较短。GPU自身“单步编译调用→四步编译调用”只出现几个百分点差异。

这里的“私有路径”是请求mask0x4的固定ANEForge/e5rt程序；没有物理ANE trace。所有时间是完整host checkpoint窗口，不是硬件kernel、完整模型训练或能效资格。旧批次原始结果保留，不用新数据覆盖旧数字，也不跨批次拼分子/分母。

## 1. 这次究竟匹配了什么

沿用已完整审计的n32/n128、seed97/101四套FP16原始W/X/T，逐文件与旧证据SHA匹配。64个minibatch各不相同，SGD、lr=.5、loss scale32、FP16端口与前批一致；参考仍是 `W_next=W-.5*Xᵀ(XW-T)/n` 的独立FP64理想轨迹。

三arm均从同一初始W重置，使用前K批，K4/16/64：

| arm | 分组 | 参数、输入与最终checkpoint |
| --- | --- | --- |
| gpu_step | 一步函数编译，调用K次 | ready host X/T在窗口内转MLX；依赖链末尾eval，再owned host FP16 copy |
| gpu_group4 | 相同一步公式静态展开四次，编译后调用K/4次 | 相同host数据边界和最终eval/copy，无中途checkpoint |
| private_group4 | 原创四步SGD/VJP图，K/4次execute | 同一W输出alias初始W输入；逐组feed不同X/T，最后owned host W |

GPU两arm使用同一个`formula`定义，不更换优化器或把CPU/FP32基线混进来。私有图不是上游Adam-only `UnrolledTrainer`。输入转换、必要提交/等待、最终host输出都在timer里；compile/load、reset、oracle、数值检查和NPY/JSON写入在外。实际代码为三arm客户端。

每K先3组warmup，再12组计时；三arm全部6种排列重复两遍，各位置出现4次、组内每个有向相邻pair出现4次。不宣称平衡跨group边界。主试在调用编译函数前显式`mx.enable_compile()`，并记录GPU/default stream；首批默认compile-mode pilot也完整保留，不反填为当时已显式启用。

## 2. 主试结果：不把跨campaign漂移当作优化

下表为显式编译主试的每arm 12次完整host窗口中位数，单位ms。配对商的中位数、IQR与全样本保存在证据JSON；不是拿两个中位数的商冒充配对统计。

| n | seed | K | GPU step | GPU group4 | private group4 |
| --- | --- | --- | --- | --- | --- |
| 32 | 97 | 4 | .2740 | .2789 | .1606 |
| 32 | 97 | 16 | .6458 | .6389 | .6374 |
| 32 | 97 | 64 | 1.8780 | 1.8562 | 2.4207 |
| 32 | 101 | 4 | .2634 | .2620 | .1459 |
| 32 | 101 | 16 | .6567 | .6505 | .6561 |
| 32 | 101 | 64 | 1.3874 | 1.3512 | 2.4221 |
| 128 | 97 | 4 | .2832 | .2729 | .1806 |
| 128 | 97 | 16 | .6237 | .6209 | .6785 |
| 128 | 97 | 64 | 1.9304 | 1.8981 | 2.6279 |
| 128 | 101 | 4 | .2840 | .2807 | .1865 |
| 128 | 101 | 16 | .6710 | .6625 | .7345 |
| 128 | 101 | 64 | 1.9652 | 1.9321 | 2.7706 |

- `GPU step / GPU group4`的同trial配对中位数约.994–1.031，**不是四倍收益**，也未做统计显著性资格。
- K4的`GPU group4 / private group4`约1.48–1.81；K64约.555–.763，方向相反。K16的n32近似持平，n128的GPU较短。
- 本批普通GPU step本身就比前一批某些窗口短，因此“差异全来自新增GPU展开”也不成立。两批调用端的Python遍历/容器实现并非逐字节相同，也没有独占SoC、连续温度/频率或后台负载trace；未确定漂移原因。
- 新进程之间也存在明显时间波动，例如n32两个seed的K64 GPU窗口不同。没有把“seed导致GPU变快”当作因果解释，更不能用最佳单次结果自动选部署策略。

原来的数值结果没有被推翻；改变的是速度结论的适用范围。先前“n32某K64窗口私有图略短”的观察不应晋升为稳定策略，本批更直接的同进程对照没有复现那个优势。

## 3. 实际图检查：一次函数调用不等于一次Matmul或kernel

图检查客户端只从hash匹配的本库timing runner提取`formula`、`gpu_group4`两个函数定义；不运行其main、导入ANEForge或加载私有shim。请求GPU和compile mode，构造图后导出DOT；没有显式eval或读取MLX输出tensor，也没有硬件counter/dispatch观测。它独立于正式计时，不给计时arm加observer。

在两种shape，实际安装库导出的图都显示：

| 图 | Python编译函数调用 | 不同primitive对象 | 运算声明实例 | Matmul实例 |
| --- | --- | --- | --- | --- |
| 单步 | 1 | 5 | 5 | 2 |
| 连续调用同一单步函数四次 | 4 | 5 | 20 | 8 |
| 一次四步展开函数 | 1 | 20 | 20 | 8 |

两个四步图的实例类型计数相同：8 Matmul、4 Transpose、4 Subtract、4 fused elementwise primitive。**这不是20次或8次GPU dispatch的证明。** primitive、reshape/transpose后端行为、Metal encoder/command buffer和物理kernel没有一一映射保证。

### 3.1 保留的错计数负例

首个parser按`primitive_id`去重，得到chain4只有5个对象。这项对象数本身是真的，却不能解释成“只有5次运算”。逐行检查原DOT发现同一个对象ID有多次声明，各自连接不同array代际；cached compiled tape复用primitive对象。

修正版逐声明计运算实例，为每次出现分配独立P编号，并另外记录不含地址的O对象键。否则DOT也可能把多次执行的算子折叠到同一个节点，画出不真实的操作依赖。原始pointer-ID图只留本地ignored目录，公开的是规范化图、计数及hash；首个对象计数diagnostic另以pilot身份保留，没有静默改成成功的operation计数。

## 4. 从固定MLX源码解释，哪些仍只是推断

固定`ml-explore/mlx@7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`，源码版本v0.32.0；不把它当作安装wheel的已证明loaded-image身份。[manifest](sources.md#grouped-training-sources-20261016)给出四个文件的SHA/实际范围。

- [Python binding](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/python/src/transforms.cpp#L466)将array与pytree/常量签名送入compile。减少外层Python调用可能减少wrapper工作，**可能**是分组收益来源之一，但本轮没有拆时归因。
- [compile与replace](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/compile.cpp#L1026)使用缓存tape并重建array，重用`primitive_ptr`；解释为何对象数少于操作实例数。记录的实际图与此机制一致。
- [fusion predicate/停止条件](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/compile.cpp#L77)不是“把整张训练图变成一个kernel”；所读predicate列出逐元素等操作，遇到不可融合primitive、stream/深度等边界停止。本实验图中Matmul仍单独出现，不能把四步编译称作融合四次SGD的单个GPU kernel。
- [DOT exporter](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/graph_utils.cpp#L105)按array遍历、用primitive身份标记算子；它是图调试输出，不是硬件活动日志。需要区分array代际与primitive对象复用。

### 可迁移技巧

1. 比较相同优化层级：resident vs host往返、私有group4 vs GPU group4，各自单独回答问题。
2. 把数量分层：逻辑更新、Python调用、图实例、对象、native dispatch、command buffer、硬件执行，不借一个字段推断其他层。
3. 动态W应是显式函数参数/alias state；不要把变动state错误捕获成编译常量。对不同X/T和代际做完整验证。
4. 展开降低调用数，也可能增加图大小与编译/临时内存；小图结果不能外推大layer、长时训练或冷启动摊销。

## 5. 独立复核与失败保留

8个训练进程：4 pilot、4显式编译主试。主试540份checkpoint（432计时、108 warmup）；合计1,080份checkpoint、30,240次实际逻辑更新。所有checkpoint均过预设relativeL2≤.02、maxabs≤.002。

标准库审计不导入NumPy/MLX/ANEForge、不执行设备；完整重建四套byte-identical数据各64步理想轨迹，共256个独立理想step。第二个版本复用同一curator进程已重建的、按workload SHA键控的reference，但重新读/hash所有原始文件，并逐元素复核所有新checkpoint：**合计9,400,320元素，不是抽样**。没有把reference重算缓存称为再次观测内部训练轨迹。

最大relativeL2约5.68e-4、maxabs约5.96e-4。所有arm各自重复checkpoint位稳定；本批GPU step/group4观察到位相同，私有group4与原批对应配方也位相同。仍只验证窗口末W，不宣称观察了内部所有逐步梯度、临时tensor或物理累加精度。

完整gate/order/数组/身份/图计数反例见单元测试。中间一次curator缩进错误曾只导出2个process记录，虽然遍历复核了全部raw数据；该不完整输出只留ignored目录，不作为本批证据。修正后加入8进程覆盖不变量及缺失/重复campaign反例，最终证据必须完整导出才通过。

## 6. 复跑与下一步

便携CPU合同不自动触发设备：

实际训练实验仍需已准备的固定隔离ANEForge源码、相同shim、上批合成fixture和NumPy/MLX；不会自动clone/build/download，输出必须是新目录：

另检查保存runner、supervisor receipt、规范化图与全部数组。公开JSON含逐样本和规范化synthetic图，不含原始数组、opaque地址、shim或模型。下一步优先研究真实layer/更大shape、GPU-resident上游数据边界、Adam/master/moment精度与checkpoint恢复；profiling继续以有效request关联/单位标定为门槛，不从本轮图计数补造ANE/GPU硬件时间。
