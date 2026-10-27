# T-GATE MPS 成本：同配方数值通过，少投影仍只带来有限净收益

<!-- knowledge
id: technique-algorithms-case-studies-tgate-mps-resident-cost
type: case-study
stage: dit
operator: attention, conv, sampling, layout
backend: metal, cpu-reference
device: gpu, cpu
symptom: launch-bound, measurement
semantics: approximate, contract-only
representation: fp32
boundary: producer-consumer, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [阶段/分支与CPU消费者](../methods/tgate-phases-and-branches.md) · [S66](../sources.md#s66) · [实验入口](../../examples/README.md)

2026-10-17，第二十三轮。在真实PyTorch MPS API上完成四个随机U-Net cohort、40个平衡trial/200个正式完整latent-pipeline窗口。所有cold/warm/formal输出均通过**同配方CPU reference**的预定门槛；相对CFG-only强对照，fi2配对中位倍率约1.047–1.069，但两个cohort的IQR跨1。cross-cache单项在一个cohort配对中位数低于1。

这里取得的是指定resident随机网络的device-ready成本与backend数值对照，**不是预训练画质、完整图像/视频、CPU对GPU速度排名或默认启用资格**。CPU reference没有计性能；CPU↔MPS小误差也不等于近似方法相对full的损失很小。

## 固定身份、配方与强对照

Torch2.14.0声明git `08187d9`，安装`torch/mps/__init__.py`与该Git blob逐字节相等；区别于旁路[2.14.1 SDPA研究](../../metal/case-studies/pytorch-mps-sdpa-dispatch-and-layout.md)，未借用其运行身份。Diffusers0.39.0的17个选定安装源码继续与`a3608b5`匹配，完整未改T-GATE `ab2a88f`包真实导入。均非完整native binary/所有依赖闭包证明。

本机post-run只读补录为M4 Max/64GiB、macOS26.6.2/build25G83、host SDK26.5；post-run SDK不是Torch构建SDK或连续设备状态证明。没有预训练资产、CLIP/VAE执行、源码/安装包修改、系统cache清理、memory fraction或全局wired-limit调整。

预登记计划在提交GPU前固定：两组channels `16/32/64/64`与`128/128/128/128`，分别2143028与16692100个随机参数；latent grid为`16×24`和`32×48`，共四cohort。heads参数为2、F32数据；第二组名为head64-eligible，仅表示源码选择条件可能满足，不是执行分支/ISA观测。

每组10-step DDIM、32个`beta=1/64`、offset1、eta0，提供resident合成latent/positive/negative embedding。五臂为full CFG、`cfg-null`强分母（step3后只减CFG batch，继续重算attention）、cross-only、gate4/fi1及gate4/fi2；完整配方同上一轮。每臂都有相同形式的step-end callback，null臂只在边界改变guidance/条件；其余为no-op，避免只给强分母额外记录工作。

缓存臂的每个计时请求都包括本请求的warm-up/边界写入、late phase和controller，不单独计cache hit。每臂用独立同参数pipeline，正常重复请求沿用已验证warm/reset策略；不使用gate0或上一轮不安全bypass作为快臂。

## 门槛、准入与测量边界

提交前固定`abs_error≤1e-4+1e-4*abs(CPU)`，并要求全向量relative-L2≤.001；未根据结果调整。CPU reference逐配方重建，不把近似臂错比成CPU full。260个GPU输出的独立最坏max abs为`3.17e-6`、max scaled约.03031、relative-L2约`1.95e-6`；全部有限/通过。该门槛验证的是backend实现一致性，**不是图像或有损质量预算**。

新鲜三样本Device≤5%在launcher、framework前、每cohort GPU setup和每trial前检查；只记录无argv的PID/CPU/程序名，不停止竞争进程。native共86个三样本窗口，其中45个通过、41个暂缓/冷却后重查，加outer一次通过。暂缓可能包含本任务前一trial的recent work，未归因成“41次外部竞争”。每个通过trial后依次执行五臂，不证明其间全系统连续独占、CPU空闲或每PID GPU监控。

进程内固定可选unsupported-op fallback=0、fast-math=0、prefer-metal=0、disable-prefill=0，profile/signpost环境为0。当前源码还显式把`embedding_renorm_`注册到CPU，因此环境0不能证明所有op都GPU；metadata/scheduler本来也有CPU工作。本轮没有逐op placement或ISA/counter证明，所有已测controller/CPU开销保留在wall内。

计时为`perf_counter_ns`包围完整latent请求，结束显式`torch.mps.synchronize()`。当前源码串到Torch stream的coalescing结束与commit/wait；这是Torch API同步观察，不是其它进程/全设备物理drain资格。排除model setup/上传、CPU reference、准入、完整输出CPU拷贝/验证/JSON；**是device-ready，不是host-ready image E2E**。

每cohort五臂first-call各一次、再各两次warmup排除，共20 cold+40 warm；cold在同一进程按固定次序运行并共享backend cache，不能当独立cold boot排名。正式10trial用五种rotation再各自reverse，每臂每位置恰两次，无截尾/删慢样本。200正式窗口之外的60个也完整做同配方数值核查。

## 结果：配对倍率，不用时间中位数相除替代

下表时间为完整窗口中位ms；倍率为每trial `cfg-null/candidate`的比值再取中位数，>1才较快。IQR=P25–P75，不是置信区间。参数量/conv/FFN也随channel组变化，不能把两组时间差只归因给head或某条kernel。

| channels / latent | full ms | cfg-null ms | cross-only倍率 | fi1倍率 | fi2倍率 / IQR |
| --- | ---: | ---: | ---: | ---: | --- |
| 16/32/64/64，16×24 | 112.79 | 112.04 | 1.025 | 1.024 | 1.047 / 1.036–1.070 |
| 16/32/64/64，32×48 | 134.09 | 122.15 | 1.012 | 1.032 | 1.054 / .977–1.109 |
| 128/128/128/128，16×24 | 119.82 | 126.76 | 1.020 | 1.030 | 1.069 / .884–1.244 |
| 128/128/128/128，32×48 | 137.26 | 128.18 | .997 | 1.012 | 1.050 / 1.007–1.081 |

fi2在四组配对中位数都高于1，但所有组min都低于1，两个IQR跨1；cross-only在最后一组中位倍率低于1，未取得稳定通用赢家。不能拿逻辑project次数减少直接推导同倍wall收益，也不能把作者1080Ti的10%–50%数据搬到M4。减少batch的`cfg-null`对full本身也有噪声和一次反收益中位数。

第三行尤其不能用`126.76/119.82`代替实际同trial的估计量；各臂不同trial的中位数除法不等于配对比值中位数。完整min/max/IQR及所有原始start/end、order、admission和输出都保留于。host/driver负载、温度/频率和连续独占未完全隔离，未用噪声挑冠军。

相比GPU full，近似配方的合成latent相对差约5%–9%，不是媒体质量百分比，`cfg-null`本身也是不同配方。这里的paired强分母只隔离cache净成本，**并未证明各方法在同一预训练质量预算下可比**。不能因此推荐fi2默认打开。

## backend形态与可迁移解释

固定`Attention.mm`的D32/64等prefill条件、Q≤8 vector条件、mask/dtype/stride与MPSGraph fallback分别核对。small-head组并非全部Graph：其深层D32也可能满足prefill；head64组的短Q cross可能仍走Graph。F32不满足该版MPP prefill的half/bfloat dtype条件。以上是源码资格分析，未instrument实际每层分支，不能叫MPP/NAX/ANE收益或矩阵指令证明。

实际效果同时含Python递归forward注册、processor接口、cache分配/复用、norm/FFN/conv、batch切换、GPU提交和等待。若继续优化，应以当前full/CFG-only强分母做局部消融并重新验证取消/reset与质量；不是只优化cache命中的某一个SDPA后把倍率乘到整模型。

## 重放与失败范围

启动器需显式`--run --gpu`、Python和固定checkouts/plan/新目录，fresh准入后只监督自己的有界进程；GPU worker另要求显式admitted与固定MPS controls。标准库整理器逐条恢复606行journal、86个native窗口、260次GPU输出引用及20个CPU reference。

完整readback按内容/shape去重为44个payload（包括input bank），每次输出仍分别查同配方门槛，未以去重数量替代260次观察。每个cohort/arm在本次cold/warm/hot重复内容一致是观测，不是所有版本确定性保证。完整输入bytes前后相等、每臂GPU参数fingerprint与CPU一致；没有发布权重、GPU backing或真实activation资产。

前置自有语法错误在import前停止，一次审批服务503也明确未执行；修正只在自有worker，并通过同一审批入口重试，均不算GPU算法失败。正式研究只有上述完整run，未把未执行或排除cold/warm放入正式比值。

默认两条仅标准库数据/合同重放，不导入Torch/T-GATE、执行MPS/IORegistry、读取模型或提交GPU。主CPU/GPU/model/media、同配方backend误差与近似失真资格保持分开。
