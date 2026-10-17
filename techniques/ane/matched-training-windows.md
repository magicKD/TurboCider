# 匹配K步训练窗口：驻留、四步展开与编译MLX GPU

<!-- knowledge
id: technique-ane-matched-training-windows
type: method
operator: gemm, scheduling
backend: private-ane, mlx, cpu-reference
device: cpu, gpu, ane
symptom: measurement, precision-loss, lifetime
semantics: contract-only
representation: fp16, fp32
boundary: producer-consumer
api: reference-only
-->

[ANE目录](README.md) · [先前逐代数值实验](resident-training-lab.md)

2026-10-16，M4 Max / 64 GiB / macOS26.6.2，NumPy2.0.2、MLX0.32环境。先前host/device/resident实验的计时包含不对称probe和文件写入，不能计算速度比。本轮另起四个进程、采用共同checkpoint边界；旧证据和旧结论不改写。

**本机小型SGD组件中，驻留比有host往返的device optimizer短；普通resident仍比编译MLX GPU慢；四步展开在短窗口更短，在128×128/K64却略慢于GPU。** 这是API/host窗口观测，不是物理ANE核时间、完整模型提速、Adam或收敛结论。

## 1. 真正匹配的工作与边界

任务为动态参数 `W[n,n]` 的平方误差，64份按seed固定、互不相同的FP16 host `X,T`：

```text
L = 0.5/n * sum((XW - T)^2)
g = Xᵀ(XW - T)/n
W_next = W - 0.5*g
```

两个shape为n32、n128；每shape有seed97、101；每次trial从同一初始W重置，只使用前K份minibatch，K=4/16/64。loss scale32在更新前抵消；存储/端口FP16，内部累加与融合舍入没有物理观测。以同一FP64理想轨迹为质量基准，不强求不同fusion之间位一致。

共同计时窗口是**准备好的host minibatches + 已重置state → K次真实更新 → 拥有独立host FP16 checkpoint**。包括必要的输入转换、提交、等待与最终host复制；排除compile/load、state reset、生成oracle、校验和所有文件写入。不调用逐步loss，不装gradient observer。候选输出保存在RAM，所有NPY在计时轮次全部结束后才导出。

| arm | 每步/每组实际工作 | state与checkpoint |
| --- | --- | --- |
| device | 上游Trainer反向图+独立SGD更新图；2K次program execute | 每步固有host梯度与W往返，最终host FP16 copy |
| resident | 上游Trainer将反向+SGD融合；K次execute | 更新输出alias输入，K步后统一读W |
| unroll4 | 本库原创四步SGD图；K/4次execute | 四组不同X/T，最终W alias初始W端口；K步后读W |
| gpu | `mx.compile`的FP16 matmul/SGD，K次逻辑step调用 | 显式GPU及其default stream；host X/T在窗口中转MLX，lazy链末尾eval并复制checkpoint |

前三种私有程序检查请求mask均0x4；不是物理placement证明。GPU逻辑step次数也不是Metal command buffer数。

固定[ANEForge Trainer源码](https://github.com/sbryngelson/ANEForge/blob/896c4943c5ebf811c0e815f8d1381fd1e157fa7d/aneforge/autograd.py#L725)。上游`UnrolledTrainer`在此revision是**Adam-only**，没有拿它冒充同SGD对照。本库unroll4用所读primitive/VJP实现，每一步只对“本步current W”求梯度，停止在该参数frontier，不对前序优化器更新做高阶微分。实现见。

device/resident另外编译了checkpoint forward图，但计时执行不调用它。驻留消除了中间往返，同时融合了图；本轮没有分别隔离“alias、融合、少dispatch、少Python转换”各自贡献，不能把所有差异解释成DMA带宽。

## 2. 排序、覆盖与独立完整oracle

每K先3组warmup，再12组计时，每组四arm。Williams四序列重复三次，使每个arm在每个位置出现3次，每个组内有向相邻pair出现3次；不是声称枚举24种排列，也没有平衡所有跨group边界。各shape/seed进程串行，无独占SoC lease，未取得温度/能耗/后台负载trace。GPU首次lazy编译发生于被丢弃warmup，不把first-call当warm latency。

共720份checkpoint，其中576计时、144 warmup；包含20,160次逻辑优化器更新。每份样本保留K、trial、完整order、position、host ns、payload SHA和误差，不能用一条success布尔值代替。

不导入NumPy/MLX/ANEForge，也不运行设备；从原始FP16 NPY重建**四个run全部64步的理想SGD**，用`math.fsum`计算每个完整matmul，再校验K4/16/64的FP64参考、720份完整checkpoint的6,266,880个元素。不是抽样验证，也不是只检查runner自己报出的loss/误差。完整重算保存参考的maxabs差≤1.12e-16。

预先固定relativeL2≤.02、maxabs≤.002。实测所有checkpoint：最大relativeL2约5.68e-4、maxabs约5.96e-4。所有K均可拒绝“直接返回初始W”负对照。每arm重复checkpoint位稳定；本轮device/resident/unroll4各K观察到位相同，但这不是对未来图/精度的保证；GPU只按相同预设数值门槛通过。**未读取resident/unroll图内部逐步梯度或所有中间W，完整重建的是独立理想轨迹，实际检查的是各窗口末checkpoint。**

## 3. 本机warm窗口结果

下表为每arm 12次完整host窗口的中位数，单位ms。不是单个核或一次提交时间。IQR、min/max及同trial配对比值保留在证据JSON的 `verification.summary`；不要将表中中位数的商冒充配对比值中位数。

| n | seed | K | device | resident | unroll4 | compiled GPU |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 97 | 4 | .8864 | .4308 | .1489 | .2538 |
| 32 | 97 | 16 | 3.4626 | 1.7004 | .6112 | 1.0750 |
| 32 | 97 | 64 | 12.4762 | 5.9800 | 2.0575 | 2.3336 |
| 32 | 101 | 4 | .7919 | .3824 | .1291 | .2664 |
| 32 | 101 | 16 | 3.1118 | 1.4877 | .5201 | 1.0494 |
| 32 | 101 | 64 | 12.4333 | 5.8738 | 2.0459 | 2.3241 |
| 128 | 97 | 4 | 1.3390 | .4812 | .1627 | .2623 |
| 128 | 97 | 16 | 4.9739 | 1.5973 | .6834 | 1.0954 |
| 128 | 97 | 64 | 19.9131 | 6.0547 | 2.5639 | 2.5015 |
| 128 | 101 | 4 | 1.3171 | .4590 | .1645 | .2854 |
| 128 | 101 | 16 | 5.0007 | 1.6456 | .6660 | 1.0959 |
| 128 | 101 | 64 | 19.7567 | 6.0026 | 2.5514 | 2.4797 |

同trial `device/resident`配对比值中位数约2.00–3.29；`resident/unroll4`约2.32–2.99。**相对一个低效私有路径更快，不代表优于GPU**：这里所有普通resident窗口都慢于GPU。四步展开在n32/K64相对GPU的配对比值仅约1.11–1.13；n128/K64则为0.96–0.98，GPU略短，不能宣布通用胜者。未做统计显著性、长期热稳定或冷启动摊销资格。

## 4. 机制解释：减少往返与dispatch，但不声称测到带宽

在源码层，单矩阵端口payload记为B=2n²字节，忽略padding/driver开销：

- device约`7KB+2K`：反向读X/T/W、写G，更新读W/G/lr、写W；共2K次execute。
- resident约`(2K+1)B+2K`：仅逐步X/T/lr和最终W；共K次execute。
- unroll4约`(2K+1)B+2(K/4)`：数据量近似相同，但仅K/4次execute。

这是源码payload模型，不是实测DRAM流量、PCIe式传输、ANE SRAM占用或性能计数器。统一内存仍有转换、绑定、缓存和同步成本。unroll的图大小、临时tensor、编译时间与实际融合策略也可能上升；编译/load另记在各run setup，GPU首次编译没有对称独立setup数字，不能据此算完整冷启动break-even。

## 5. 复跑与研究边界

先运行便携CPU检查，不触发设备：

可选私有实验需要隔离且hash相同的ANEForge源码、预先编译的shim及NumPy/MLX。runner不自动下载/构建，不改变生产配置。`--source`必须在本仓库outputs下，已存在output拒绝覆盖：

默认四arm；`--without-unroll`是明确不同的三arm协议，不是失败后的静默退路。本轮四个四armrun全部通过，未使用该退路。验证保存的source/runner/binary/receipt身份与所有数组；本地raw NPY、shim、build cache不随知识库发布，便携JSON提供逐样本和hash，不假装可脱离raw重新审计原始数组。

剩余问题：更大shape/真实layer、Adam/master state、持续数据管线、共享GPU上游activation、与优化GPU graph/replay的对称比较、冷启动/内存/能耗，以及有来源和单位校准的物理trace。本批编译GPU没有跨K步的手工静态展开或Metal capture replay，因此没有证明胜过最优化GPU实现。后续独立完成了[同SGD GPU group4消融与实际图检查](grouped-training-and-graph-identity.md)，新批长窗口私有路径未胜出；不回写本批数据或跨batch配对。不能直接把这里的比值用于LLM训练预算或整模型部署。
