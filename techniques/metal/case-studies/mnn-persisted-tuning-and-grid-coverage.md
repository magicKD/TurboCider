# MNN 持久调优缓存：近似命中不等于合法 dispatch grid

<!-- knowledge
id: technique-metal-case-studies-mnn-persisted-tuning-and-grid-coverage
type: case-study
stage: vae, pipeline
operator: conv, layout, scheduling
backend: metal, cpu-reference
device: cpu, gpu
symptom: stale-cache, wrong-layout, measurement, compile-bound, lifetime
semantics: numerically-equivalent, contract-only
representation: fp32, fp16, packed
boundary: producer-consumer, cpu-gpu
api: public, private, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [第九轮真实引擎](mnn-engine-convolution-and-resize.md) · [固定来源](../sources.md#mnn-cache-sources-20261016) · [验证快照（过程原件已移出）](../../MAINTENANCE.md)

2026-10-16，第十轮。本轮将上一轮的close-shape疑点推进到原始MNN Session执行：**加载小shape缓存后，近似查询直接复用旧groupNum，确实可使较大shape少发线程并输出错误**。没有修改上游实现，也没有把错误路径工作量变少计为加速。

结论固定于 `alibaba/MNN@dbf9ba61b08106a3b43426e295d1a29cb41cc8fe` 与记录的M4 Max环境，不代表所有MNN发布版本/封装器；本轮未确认对应PR及最新修复状态。

## 1. 先分清缓存保存的是什么

沿公共API读取 [`Interpreter.cpp`](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/core/Interpreter.cpp)：`setCacheFile`读文件，createSession将数据交给runtime，resize产生/复用调参信息，`updateCacheFile`或析构按size增长条件写回。`keySize`在该实现中没有提供模型内容校验。

Metal的 [`MetalCache.fbs`](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/metal/schema/MetalCache.fbs) 保存kernel key、请求threadSize、groupNum、groupSize、timeCost，以及可选op/shape信息。**这条flatbuffer不是MTLBinaryArchive或序列化shader库**；驱动编译cache、进程内shader/pipeline cache和本轮调优cache必须分开。

| 层 | 本轮读到的复用对象 | 不应误认为 |
| --- | --- | --- |
| Execution cache | 按Op身份复用Execution，之后onResize | 每次shape变化都会重新跑creator选型 |
| tuning exact map | `(kernelName, GWS)`→`groupNum/local/score` | 已包含所有dtype/source/SKU的显式资格证 |
| 从文件加载的nearest向量 | 同kernel的若干GWS及完整返回tuple | 只借用local、再自动重算新groupNum |

普通调参向exact map插入记录；[`MetalBackend.mm::setCache`](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/metal/MetalBackend.mm)加载文件时才同时填充nearest向量。因此，“全新runtime内resize正确”不能反证“跨进程加载调优cache后resize正确”。这也不推翻上一轮没有持久cache的成功输出。

## 2. 关键实现闭环

[`MNNMetalContext.mm`](https://github.com/alibaba/MNN/blob/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/metal/MNNMetalContext.mm) 的真实顺序是：

```text
先查(kernelName, 新GWS) exact
  → 没有则按同kernel的GWS L1距离找nearest
  → 直接返回旧groupNum、旧local、旧score
  → 有命中就提前返回；不走后续新grid计算/测量/插入
```

`getCloseThreadgroup`中的`minScale=0.1`没有参与筛选；不是已执行的10%距离保护。`getGridAndThreadgroup`在close命中后也没有按新GWS重算groupNum。

`MetalConvolution1x1.mm`把返回group/local存进实际 `mThreads`，onEncode使用它。该路径把C4NHW4的N×H×W折叠到W，内部batch设为1；所以N=1→2会增加GWS.x，不能只盯GWS.z是否变化。

`TuneLevel::Never`判断发生在这些cache查询之后。None模式禁止新搜索，**不等于禁用已有调优cache**。其新生成未计时条目可以携带`UINT_MAX` score；那不是实际时延。Fast模式则可能拿旧shape的score选择新shape的kernel，不能将命中分数当新shape的性能。

## 3. 实际引擎实验，不是只调用孤立lookup

复用上一轮未修改的Release dylib，library SHA为`a4f37842aff2168db62808ad2e0e0b1988bca246d3c44e517e9dcb1db50130c2`。每个子进程核对固定源码、加载镜像路径/hash及实际五个Metal Execution节点。

完整消费者为：NCHW→Raster/C4→1×1投影(CI7、CO13或17)→ReLU→1×1投影(CO5)→Raster/NCHW。独立FP64直接计算全部输出；High门禁仍为L2≤`5e-5`、max abs≤`1e-4`，Low为L2≤`1e-2`、max abs≤`3e-2`，没有因错误放宽。

seed17/29 × High/Low × None/Fast，共8个上下文。每个上下文22个**独立原生进程**，每进程4次A/B输入交替、每次重新上传；正式共176进程、704个完整输出：528通过、176失败。失败是有意覆盖边界的矩阵结果，**不是线上失效率估计**。所有数值失败均发生于加载cache的臂，fresh/producer全部通过。Supervisor的0仅表示完整收集结束，不把176个失败改成通过。

每个reader使用producer cache的独立副本；原始producer文件保持不变。所有reader的输入cache hash均与对应producer终态hash相同。模型输入初始序列化固定为17×19，在第一次resize前设置实际运行shape；grow等同CO13对照的**模型及权重字节相同**，并非换模型后忘了匹配参数。

| 对照 | 运行shape/变化 | None结果 | Fast结果 |
| --- | --- | --- | --- |
| exact |17×19→17×19|通过|通过|
| neighbor |17×19→19×21|通过|失败|
| grow |17×19→37×41|失败|失败|
| batch |N1→N2，空间仍17×19|失败|失败|
| channels |主投影CO13→17|失败|失败|
| kernel-switch |17×19→5×7|通过|通过|
| shrink |37×41 cache→17×19|通过|通过|
| large-exact |37×41 cache→37×41|通过|通过|
| new weights |同shape，新seed/参数|通过|通过|

每格覆盖两个seed、两种精度、每进程4个全输出。新权重对照也很重要：该调优cache不是权重cache；本轮同shape换参数的通过反证了“只要没绑定模型权重就必然取到旧权重”的错误解释。

## 4. 线程覆盖与错误位置可以对上

None的源shape面积323，`conv1x1_g1z8`主投影请求GWS=`[41,4,1]`，消费者=`[41,2,1]`。缓存返回两组x=2、local.x=32；即每核最多覆盖64×8=512个flattened空间位置。

grow的面积是37×41=1517，实际新GWS.x为190。无cache会返回x=6组；加载源cache却仍为2组。完整输出中，**前512个空间位置符合FP64阈值，后1005个位置错误**。两个seed、High/Low均保留同样分界；不是“多查几个像素恰好没查到错误”。

Fast的neighbor也解释了为何“变化不大”不够：消费者实际选中的 `conv1x1_g1z4` 使用旧x=3组、local.x=32，最多写384个位置；19×21=399，恰好漏15个位置。该例的完整输出错误空间位置数正好为15。主投影在一个Low上下文选了`w4c2`，不能把两条投影统称同一kernel。不是所有候选缺覆盖都被叫作实际dispatch：observer同时记录pipeline请求与原grid返回，归档器对应两条最终选中kernel的记录。

None的neighbor虽也是close命中，但新请求50仍落在旧64线程覆盖包络内，因此通过。shrink的额外线程被本轮选中shader的边界判断挡住，也通过。它们是必要正控：**close并非一律数值错；coverage足够也不是任意shader/layout的通用安全证明**。

## 5. 排除观察器与预填干扰

观察进程只包装原始Objective-C方法以记录请求/返回，不改变kernel参数或返回grid。首次尝试普通category因MNNMetalContext类符号隐藏而链接失败；后改为运行时查找class/安装wrapper，原库未重编或打补丁。另有无观察器进程用于否决ABI包装制造错误的解释。

- 每个上下文都有不安装任何方法observer的grow fresh/cached对照；fresh通过，cached四个输出均失败。
- 进一步完全不预填逻辑输出，仅正常上传输入和运行引擎，共32个自然cached输出仍全部失败；对应natural fresh全部通过。
- 对有预填的臂，公共Tensor copy只在运行前初始化**逻辑输出**；arena复用和较早合法op可能覆盖这些值。因此不声称物理scratch guard已守住，也不把尾部错误值必然等于某个sentinel当作证据。

早期自然smoke、编译失败、第一次扩展矩阵均保留但不混入704最终输出。所有失败值在本次记录中仍是有限数，说明仅检查NaN或runSession返回码会漏掉问题。

## 6. 如何吸收这个优化设计，而不是只记一个bug

可迁移的收益仍然存在：昂贵搜索可被合法缓存摊销。但必须把**候选提示**与**当前合法launch**分开：借用local时重新按当前GWS计算groups，并重新检查shader边界、packing、dtype/source/设备条件与数值。原始nearest距离只能用于缩小搜索空间，不能替代资格验证。

复盘中的可复用测试判据：shape扫描必须跨越线程组覆盖包络，并包含“写盘→新进程加载”，不能只测同一进程中的小幅resize。一个仍落在旧grid内的邻近shape通过，并没有验证更大shape的合法性。参考仓库/技能只读，本轮将该方法留在本地知识库，而不编辑上游技能。

这里的“重算groupNum”是本地CPU算术示例，不是已经提交/集成的MNN修复。仅重算覆盖也没有解决所有不同kernel、tail、资源和性能条件，不能宣称通用修复完成。本文没有向上游发issue/PR，也没有改变TurboCider后端默认。

调优schema缺少显式dtype、source、SDK或SKU字段，是应用层需要审查的边界，**不是本轮所有失败的已证根因**；本轮闭合证据是旧groups→少线程→完整尾部错误。新权重、exact、reverse、large-exact和无observer控制将其他解释分开。

原始 `forward_wait_ms_diagnostic_only` 保留在ignored运行目录；带observer/预填的时间不参加性能排名。尤其错误缓存少做工作时，不能以更短时间宣传加速。所有cache写回都只作用于任务文件，无系统缓存清理、全局依赖变更或GPU fault注入。

## 7. 复现与本地学习代码

CPU：labs/mnn_cache_contracts.py；测试：test_mnn_cache_research.py。只校验identity/grid算术与保存的证据，不隐式导入MNN或启动设备。实际客户端：mnn_cache_probe.py、mnn_cache_probe.mm，复用上一轮自己的harness helper；归档器：curate_mnn_cache.py。

输出目录必须新建。最终原始前缀为 `outputs/metal-mnn-cache-20261016-final`，owned receipt在同前缀`-launch`。portable capsule保存每个完整输出的指标/hash与输入/参数身份；重做全量数值校验仍需要原始数组或重新运行，不把离线测试通过冒充再次执行704个GPU输出。

本轮依照Metal/测试/缓存排查技能保留失败，并以exact/fresh/reverse/无observer控制核对因果；没有改参考技能文件。后续更换MNN源码、wrapper缓存命名/校验、编译器/SDK/SKU、dtype/shape、tile/layout或完整消费者都要复验。MLC/TVM实际AOT、更多完整模型与独立GPU profiling仍在持续队列；这一项bug复现不关闭整个研究目标。
