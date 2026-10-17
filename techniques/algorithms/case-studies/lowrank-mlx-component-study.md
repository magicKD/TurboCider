# MLX packed QMM＋低秩补偿：实际设备成本、权重结构与解码精度边界

<!-- knowledge
id: technique-algorithms-case-studies-lowrank-mlx-component-study
type: case-study
stage: dit, ffn
operator: gemm, quantization, layout
backend: mlx, metal, cpu-reference
device: gpu, cpu
symptom: launch-bound, memory-bound, precision-loss, measurement
semantics: approximate, numerically-equivalent
representation: fp16, q4, q8, mlx-affine, packed
boundary: weight-runtime, producer-consumer
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [SVDQuant/Nunchaku机制](../methods/low-rank-quantization.md) · [精度组合](attention-precision-study.md) · [实验](../../examples/README.md)

2026-10-16第六轮。M4 Max上实际运行MLX0.32.3的packed affine QMM与独立FP16低秩分支。**不是原Nunchaku W4A4/NVFP4、不是完整FFN/请求，也没有生成媒体质量资格。**这里分解的是权重，不是[SVD-Cache](../methods/research-frontier.md)的feature时间预测。

主要观察：构造的channel-outlier权重中，Q4误差约0.13–0.15可被rank32降到约0.03–0.04；高斯权重改善很小。所有32点里，compiled rank32都慢于同组compiled Q4 rank0，配对速度比约0.82–0.97。更少权重bytes、更低向量误差与更低延迟是三个不同结论。

## 本轮实际运行的数学与数据

按MLX存储约定，`X[M,K]`、`W[N,K]`，分解`W=R+B Aᵀ`，其中A为K×r、B为N×r。候选输出为`QMM(X,Q4(R))+(X A)Bᵀ`；rank0直接使用原W的Q4 bank，没有空分支GEMM。

两种权重尺寸：K/N=1024/2048和2048/4096；M=1/64/256/1024；seed17/29；两个合成regime，共32点。没有真实模型权重或新activation fixture：

- Gaussian：W按输入宽度归一化的独立高斯值，转FP16后作为共同原始目标。
- column-outliers：相同生成配方的每64列第一列乘16；两尺寸分别有16/32个outlier列。这是有意构造的结构压力对照，不是对实际模型outlier比例的估计。
- X为独立生成、可表示的FP16值；分解没有用X拟合输出。不能将这些合成向量误差当成图像保真。

CPU先做range80的随机化range finder、一轮power iteration和小矩阵SVD，取rank16/32/64；不是精确full truncated SVD。将奇异值平方根均分给A/B并转FP16，随后对**同一FP16目标W**减去已舍入因子的乘积，再把residual转FP16。没有执行SmoothQuant校准、activation量化、LoRA加载或原项目checkpoint转换。

## 设备路径和比较分母

lowrank_mlx_probe.py调用`mx.quantize/quantized_matmul`，`mode="affine"`、group64、bits4或8，weight carrier确为uint32。12个计算臂为dense/Q8/Q4-r0/Q4-r16/Q4-r32/Q4-r64各自的eager和compiled版本；另有已eval输入上的空eval+sync计时臂。

compiled函数显式接收X、words、scale、bias和因子，不依赖隐藏Python状态。但图编译不等于Nunchaku的手写融合；本轮未采集实际shader次数、ISA、timeline或物理并行。**使用packed API不证明INT4整数MAC，也不证明实现了W4A4。**

计时是每次完整候选linear的host-until-ready：包括建图/已缓存编译入口、QMM、分离的低秩计算、join和同步。权重bank、CPU分解和GPU pack在正式计时外，单独记录；不是cold load、完整FFN、含激活/投影的block或请求时间。全部候选同时驻留，所以也不是模型换层streaming或offload测试。

每臂4次warmup、12轮旋转且奇数轮反向的计时，共4608次计算候选求值及384次host control，总4992段。空eval+sync中位数约0.017–0.080ms，未从候选中相减；它不是等价图，也不能据此恢复纯device时间。

## 先修正参考语义，而非放宽数值门槛

首个pilot在packed decode控制处拒绝，**没有进入计时**：CPU按实际uint32 words、scale和bias计算实数解码；`mx.dequantize(..., dtype=float32)`却与之有相对L2约6.69e−5、max abs约6.10e−5的差异，超过预设1e−6。

查[v0.32.3固定API源码S39](../sources.md#s39)可见：affine路径先以scale dtype产生结果，再在外层`astype(out_type)`。因此指定float32返回值没有把内部FP16解码变成FP32算术。这是本探针原参考前提的问题，**不是据此发现MLX算错**。

只在非计时解码控制中，把scale/bias operands先显式升为FP32；定时QMM仍使用原FP16 bank。五个packed bank的独立解码均精确匹配，原1e−6门槛未改变；原FP16后cast的诊断误差也继续保留。pilot v1完整拒绝和准确旧脚本SHA已在本地核验归档；Wiki不附旧GPU脚本，不授予当前harness重放资格。成功的单点pilot v2也不混入最终32点性能矩阵。

全部计算输出另与CPU FP64的同一decoded-weight/因子公式对照，最大global/row相对L2约0.00273，低于事先0.02 gate；每臂还有两个独立math.fsum cell。全高精度residual+因子重构控制通过0.01 gate。它们验证数值/布局接缝，**不是与原W近似误差的合格阈值，更不是媒体门槛**。

## 精度曲线：结构决定低秩补偿的价值

以下为compiled臂在两个seed、四个M共8点的相对L2范围，分母是原FP16 W/X的CPU FP64 linear。范围不是置信区间。

| K/N / regime | Q8 | Q4-r0 | Q4-r16 | Q4-r32 | Q4-r64 |
| --- | --- | --- | --- | --- | --- |
| 1024/2048 Gaussian | 0.00540–0.00549 | 0.08966–0.09507 | 0.08662–0.09159 | 0.08739–0.09082 | 0.08455–0.08615 |
| 1024/2048 outliers16 | 0.00771–0.00926 | 0.12807–0.15306 | 0.03463–0.04072 | 0.03307–0.03992 | 0.03328–0.03869 |
| 2048/4096 Gaussian | 0.00539–0.00541 | 0.09076–0.09275 | 0.08961–0.09051 | 0.08897–0.09272 | 0.08785–0.09102 |
| 2048/4096 outliers32 | 0.00794–0.00930 | 0.13103–0.15379 | 0.09615–0.11205 | 0.03541–0.04091 | 0.03430–0.04022 |

较大尺寸有32个被放大的列，rank16只补偿一部分，rank32的变化更明显；这符合此构造的结构，不证明真实模型都需要rank32。更多rank也非逐输入保证更准：1024/2048 Gaussian、seed17、M=1中，r0误差0.08966，r16为0.08662，r32反而0.09054。量化网格、因子舍入与输入加权误差不是单纯的SVD剩余能量。

Q8在本矩阵更准，但占用更多权重bytes。没有用户质量预算，因此只保留Pareto信息，不宣布一个rank/bit preset为模型默认。

## 延迟：补偿分支可能吃掉packed收益

下表为两个seed各自中位数的范围，单位ms；都为compiled入口。G=Gaussian，O=column-outliers。

| K/N / regime | M | dense | Q8 | Q4-r0 | Q4-r32 |
| --- | --- | --- | --- | --- | --- |
| 1024/2048 G | 1 | 0.160–0.160 | 0.154–0.156 | 0.147–0.159 | 0.169–0.183 |
| 1024/2048 G | 64 | 0.256–0.259 | 0.221–0.223 | 0.220–0.223 | 0.251–0.256 |
| 1024/2048 G | 256 | 0.325–0.374 | 0.311–0.337 | 0.262–0.337 | 0.321–0.375 |
| 1024/2048 G | 1024 | 0.480–0.855 | 0.496–0.910 | 0.496–0.881 | 0.545–1.001 |
| 1024/2048 O | 1 | 0.171–0.192 | 0.171–0.193 | 0.171–0.191 | 0.194–0.216 |
| 1024/2048 O | 64 | 0.315–0.342 | 0.285–0.319 | 0.316–0.319 | 0.352–0.377 |
| 1024/2048 O | 256 | 0.429–0.470 | 0.416–0.436 | 0.401–0.433 | 0.447–0.463 |
| 1024/2048 O | 1024 | 0.817–0.870 | 0.883–0.923 | 0.868–0.932 | 0.966–1.051 |
| 2048/4096 G | 1 | 0.194–0.319 | 0.183–0.247 | 0.183–0.253 | 0.208–0.279 |
| 2048/4096 G | 64 | 0.472–0.492 | 0.423–0.443 | 0.410–0.436 | 0.465–0.486 |
| 2048/4096 G | 256 | 0.941–0.944 | 0.901–0.912 | 0.905–0.908 | 0.997–1.012 |
| 2048/4096 G | 1024 | 1.493–1.495 | 1.570–1.593 | 1.563–1.586 | 1.684–1.715 |
| 2048/4096 O | 1 | 0.289–0.320 | 0.221–0.244 | 0.222–0.245 | 0.249–0.310 |
| 2048/4096 O | 64 | 0.480–0.494 | 0.433–0.433 | 0.415–0.427 | 0.459–0.477 |
| 2048/4096 O | 256 | 0.951–0.967 | 0.916–0.917 | 0.904–0.932 | 1.027–1.031 |
| 2048/4096 O | 1024 | 1.481–1.507 | 1.575–1.586 | 1.563–1.586 | 1.661–1.700 |

跨进程绝对时间有明显漂移，例如1024/2048 Gaussian的M1024 dense中位数0.480与0.855ms；没有据此归因于seed、weight值、时钟或温度，也未删掉较快/较慢组。JSON保留全部trial和本组配对比值；**median(A/B)不等于median(A)/median(B)**，不拿表中范围跨regime排名。

按逐trial配对统计，32点的compiled r32相对compiled Q4-r0都更慢（速度比0.821–0.969）；M1024时，r32对compiled dense的配对速度比约0.83–0.89。少数小点相对dense显示优势，最大配对速度比约1.11；结合host control与跨进程波动，不能称为稳定通用加速。

## bytes与FLOPs不能替代完整边界成本

每方法逻辑权重bytes包含uint32 words、FP16 scale/bias以及A/B因子：

| K/N | FP16 dense MiB | Q8 MiB | Q4-r0 MiB | r16 MiB | r32 MiB | r64 MiB |
| --- | --- | --- | --- | --- | --- | --- |
| 1024/2048 | 4.000 | 2.125 | 1.125 | 1.219 | 1.313 | 1.500 |
| 2048/4096 | 16.000 | 8.500 | 4.500 | 4.688 | 4.875 | 5.250 |

MiB=2²⁰ bytes；不是checkpoint总大小、运行峰值或allocator实际驻留。原始FP16 target、三个residual/control和全部候选都在harness中共存；MLX active/peak只反映此进程的受跟踪分配，还不包含全部NumPy分解内存，更不能当每候选峰值。

低秩额外算术相对原linear为`r(K+N)/(KN)`，但W4A16本身没有删掉dot products；rank还增加输入读取、中间M×r与M×N流量、算子提交及join。CPU构造/随机分解本次约25–73ms、GPU bank/pack约8–11ms，均单列，不能视为每step固定开销或完整cold-start。

[Nunchaku原融合](../methods/low-rank-quantization.md)在activation quantization和GEMM epilogue边界复用IO，并使用其特定packed因子/FP32 hidden。这里用常规MLX算子和FP16分离分支，是可检验的迁移起点，不是原融合的等价系统实现。原作者大型DiT的W4A4/offload收益也不能转写为本表预期。

## 空闲检查、证据与复现

每个bank setup、每个shape和正式timing前都有三次新IORegistry样本，Device≤5%；最终通过窗口为0或1。另保留四个未通过的早期quiescence窗口，等待后重新采样，没有把busy样本解释为通过。记录进程名/CPU占用但不读取argv，不杀其它任务、不清系统或模型缓存；没有连续per-PID GPU、温度或频率证明。

保留8份完整report、实际输入/packed/factor运行前后hash、native wheel/stub SHA、4992段时间、所有误差与旧拒绝。curate_lowrank_mlx.py只用标准库独立检查byte/shape、SSE/norm、scalar见证、admission、全样本中位数和配对比；不执行设备或原项目。

源码：[S17 Nunchaku](../sources.md#s17)、[S39 MLX v0.32.3](../sources.md#s39)。tag和相同wheel版本字符串不构成reproducible-build证明；二者边界保持明确。

默认tests不运行第一条、不加载MLX/原模型或读取旧输出目录。第二条验证残差重构、double-count负例、factor gauge、转置拒绝、uint32低位解码和postcast精度，不是设备性能证明。

后续优先补真实activation/weight校准、完整SwiGLU FFN/消费者和按holdout质量预算选rank；再研究目标设备的activation低精度与真正融合。不能以这32个合成点授予生产默认、完整画质或跨SoC最佳方案资格。

## 第六轮交付复核

算法专项82项通过，其中16项为第六轮新增；覆盖源/binary/数据身份、uint32高位codes与有符号scale、全部4992段样本/中位数/配对比、输入与bank不变、SSE/scalar、空同步不冒充候选、旧失败与数值gate不改标，以及busy/缺臂/篡改拒绝。测试不重新运行GPU或读旧输出目录。
