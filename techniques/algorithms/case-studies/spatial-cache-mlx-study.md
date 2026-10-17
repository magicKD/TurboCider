# 将频谱差搬到空间：MLX GPU 的缓存周期成本与精度边界

<!-- knowledge
id: technique-algorithms-case-studies-spatial-cache-mlx-study
type: case-study
stage: dit
operator: sampling, layout
backend: mlx, metal, cpu-reference
device: gpu, cpu
symptom: launch-bound, stale-cache, precision-loss, measurement
semantics: approximate, numerically-equivalent, contract-only
representation: fp32
boundary: producer-consumer, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [FasterCache consumer](fastercache-consumer-study.md) · [缓存机制](../methods/fastercache.md) · [实验](../../examples/README.md)

2026-10-17，第十三轮。Apple M4 Max/64GiB、macOS26.6.2/build25G83、已安装MLX0.32.0，实际GPU执行四个ready-tensor cache-adapter cohort、六个计算臂、每臂14轮正式对照；另保留大偏置精度诊断及两轮pilot。**不是CogVideoX模型、完整CFG/solver、真实视频或产品性能验收**。保留所有逐plane误差、timing、hash、busy窗口和终态。

观察：将缓存差的逆变换提前到full refresh后，在本轮强`source-unshifted-compiled`分母下，两种空间缓存的完整50-call **adapter周期**配对中位比约1.16–1.78×。但FP32不逐位等价；大偏置/小差值压力测试中新旧路径都违反冻结的绝对误差门槛，不允许据“更接近FP64”自动授予替换资格。

## 可以搬动的是线性变换，不是近似预算

沿[第十二轮确认的uc-first身份](fastercache-consumer-study.md)，记U为真实执行的unconditional输出、C为full时算出的conditional，F为H/W二维FFT，P为固定低频mask。在完整anchor r保存：

```text
D_L = P · (F(C_r) - F(U_r))
D_H = (1-P) · (F(C_r) - F(U_r))
hit：C_hat = real(F^-1(F(U_current) + D_L + D_H))
```

本配方的跨步更新是对每个band施加统一实scalar（对应窗口每hit乘1.1）。因此可在refresh保存`R_L=real(F^-1(D_L))`、`R_H=real(F^-1(D_H))`，之后用相同scalar更新R并执行`C_hat=U_current+R_L+R_H`，省去每hit的FFT/IFFT。

这对实输入、Hermitian对称mask、固定layout和各band统一scalar在精确数学上成立；不能据此断言有限精度逐位相等。若每步变mask、用逐frequency gain、变geometry/branch身份或引入非线性谱操作，不能直接套这个两份residual的更新。尤其不是把SeaCache随sigma变化的频率gain无条件替换成两scalar。

本轮两种原创变体：

- `spatial-matched-compiled`：refresh仍分别FFT(C)、FFT(U)，先作同配方频谱差，再分别IFFT low/high并取实部；多付refresh，hit只更新/相加实residual。
- `spatial-delta-compiled`：refresh先FP32算`d=C-U`，一次FFT(d)和一次low-band IFFT，`R_H=d-R_L`；进一步改变subtraction/FFT及high-band重建的舍入顺序。

```text
spatial-delta refresh：d=C-U → R_L=real(IFFT(P·FFT(d))) → R_H=d-R_L
hit：R_L←a·R_L，R_H←b·R_H → C_hat=(U+R_L)+R_H
```

这是减少同一缓存公式的实现成本，**没有新增损失控制器，也没有消除FasterCache旧anchor/1.1补偿相对full分支的近似误差**。

## 分母、数据与计时合同

六个计算臂为原数学的shifted eager/compiled、unshifted eager/compiled，以及两空间compiled变体。主分母在pilot前固定为**去掉配对shift后的compiled谱缓存**，不从每个点事后选择较慢baseline。原source端全轴shift与逆shift是可消去置换，unshifted版配对应的未中心化mask，不只改一端。

输入全部ready FP32，无模型权重：

| cohort | T,C,H,W | 数据性质 |
| --- | --- | --- |
| source17 / source23 | 3,2,5,5 | 第十二轮seed17/23的明确affine模型替身full两路输出，FP64→FP32；不是实际CogVideoX激活 |
| video9 | 9,16,60,90 | seed17有界harmonics，source-shaped latent几何；不是生成视频 |
| video13 | 13,16,60,90 | seed23有界harmonics，同上 |
| offset（不正式计时） | 3,2,5,5 | source17两路先加65536再转FP32，精度压力诊断 |

每周期沿原counter1…50：15次pass、9次refresh、26次hit。pass/refresh返回当前完整C，hit重构C。100个branch inputs、两个mask、grow/keep两个scalar共104个GPU数组在前后hash一致；所有input生成、上传、CPU参考、编译和数值warmup在正式计时外。提前准备未来C只是组件对照/参考条件，不是声称真实cache命中仍应计算C。

每个cycle所消费的cache state从None重新开始；进程仍保留bank、数值观察和先前返回对象等引用，未给各方法隔离allocator，也没有测跨请求残留state、取消或外部buffer重用。

**每步都执行`mx.eval(output,state)`和`mx.synchronize()`**，报告从ready输入到ready输出的host wall；包含Python控制、相关GPU算子、state更新和同步。整个cycle wall另包含循环/计时bookkeeping，不能以逐phase中位数之和冒充cycle中位数。上游SAT/PyTorch可能异步排队，本轮没有复现其异步调度，也没有测多步graph capture；50次ready边界不能包装成原作者整请求速度。

每轮七个计时臂含一个已ready输入的`host-eval-sync-only`控制，14次轮转/交替反序使每臂每个位置出现两次。host控制不参与正确性候选、不从任何计算时间中扣除。其cycle中位数约1.43–1.90ms，原始14个样本均保留。

正式392个cycle（336计算臂、56 host控制）共19600段step interval。其中16800段属于计算臂，**含5040个passthrough、3024个refresh、8736个hit**；另2800段host控制。这不是19600次FFT或物理GPU dispatch。

## 完整refresh/hit周期的收益，不只挑hit

表内时间为14个cycle的中位数，×为逐trial `primary/candidate`比的中位数，括号为快于primary的trial数。倍率不是两个时间中位数简单相除，也不是置信区间。

| cohort | 强谱缓存ms | spatial-matched ms / 配对×（胜次） | spatial-delta ms / 配对×（胜次） |
| --- | ---: | ---: | ---: |
| source17 | 11.636 | 9.870 / 1.177（12/14） | 9.795 / 1.199（13/14） |
| source23 | 12.299 | 10.508 / 1.156（12/14） | 10.076 / 1.213（11/14） |
| video9 | 24.781 | 18.854 / 1.361（13/14） | 17.016 / 1.479（14/14） |
| video13 | 30.657 | 21.068 / 1.456（14/14） | 17.107 / 1.779（14/14） |

小shape并非每trial都更快，不能把中位收益写成稳定逐次胜出。source-like shifted eager在video13约48.203ms，但用它对17.107ms作分母会混入shift/编译差异；上述核心结论仍使用预固定30.657ms强谱缓存。

refresh确实更贵：video13的9次refresh总时间中位数，primary约7.085ms、matched约13.153ms、delta约9.075ms；26次hit对应约22.839/7.282/7.035ms。这正是必须测整周期的原因；减少FFT数量不保证refresh本身更快，新的缓存布局/Real/cast与提交也有成本。

每50-call周期的**逻辑API变换数**为谱缓存44 FFT2+26 IFFT2；matched为18+18；delta为9+9。只说明配方，不能将compile graph中的逻辑节点数升级成实际shader数或按70/18预测速度。没有比较rfft2/irfft2、捕获整个多步graph或手写融合Metal；本轮强分母不是宣称全平台最优FFT方案。

若cache adapter只占完整请求1%（仅算术假设、不是本轮测量），组件1.78×也只给约1.0044×整请求；attention/transformer、CFG merge、noise update、solver、VAE和IO都不在本计时范围。不与第十一轮Qwen完整请求cache收益相乘。

## 精度：实现gate不等于近似质量gate

四个正式cohort对六臂各50个完整输出做数值检查，共1200个output观察：

1. 全shape独立NumPy FP64空间FFT参考，输入为同一FP32值，按同一refresh/scalar日程重建原谱缓存公式。
2. 同源GPU强unshifted compiled原公式，比较新布局/舍入后的全部输出。
3. source17/23另对第十二轮实际原Torch CPU方法的对应conditional输出；不是只比较两个MLX写法。

预固定实现门槛同时要求max-abs≤5e-5、global relative-L2≤1e-5。四cohort全部通过；两个空间变体相对强GPU参考的最大绝对差≤1.43052e-6。逐T×C plane保留完整H×W的SSE/energy，video13每output覆盖208个plane、1,123,200个元素，不只抽几像素。

`versus_full_branch`另保留缓存预测相对当前完整C的非零误差，它不属于上述实现gate。移动线性算子后数值很接近原cache，不代表这个cache已接近full模型，更不是视频质量通过。正式timing只eval/sync输出，不逐次CPU回读每一个timed output；数值矩阵与计时矩阵分开，不声称19600段都做了独立完整输出复核。

### 大偏置/小差值拒绝被保留

在source17输入上先加65536再转FP32，FP32这个幅值附近的ULP已约0.0078125。300个数值观察中，各方法的26个hit都未通过同一绝对门槛；24个pass/refresh返回现成C。没有调门槛、没有正式计时这个cohort，pilot也不混入正式统计。

| 方法 | 对FP64参考最大差 | 对强GPU原公式最大差 | gate失败hit |
| --- | ---: | ---: | ---: |
| 强谱缓存 | 0.0391031 | 0 | 26 |
| spatial-matched | 0.0274696 | 0.0234375 | 26 |
| spatial-delta | 0.00771806 | 0.0390625 | 26 |

delta在这个诊断中更接近FP64，但对原FP32路径不同；先减后FFT与两个大谱相减的cancellation、逆变换/相加位置及每步实/复state舍入都变了。相对范数很小也不能覆盖冻结的绝对门槛。该压力测试不是实际生成视频失败，更不能用“更接近数学参考”省掉真实模型的质量与稳定性验收。

## Apple源码与内存单位

[S47](../sources.md#s47)固定MLX v0.32.0的七个文件。前端FFT2接口默认H/W axes、complex64路径及shift/roll；Metal实现的ND FFT按多个1D FFT组织。对60/90等长度，所读source planner可由Stockham支持的radices分解；这是**条件source plan，不是已安装binary的dispatch/ISA实测**。wheel core extension/libmlx完整hash已记录，版本同名不等于可复现构建或loaded-image attestation。

新缓存两份FP32与原两份complex64的逻辑payload：video13为8,985,600对17,971,200 bytes（约8.57对17.14MiB）。没有测每方法物理峰值、RSS或完整模型内存。

关于`mx.real`的ownership：所读tag中它构造Real primitive，未融合Metal unary路径按output itemsize分配，donation要求in/out itemsize相同，complex64→float32不满足该条件。编译融合、临时FFT buffer、分配器复用/驻留和多method生命周期仍会影响实际峰值，不能从两份carrier的nbytes推导“整个请求峰值减半”。记录的MLX allocator active/peak包含bank、warmup、引用和所有候选，不是per-arm物理比较。

在线[FFT导航](https://ml-explore.github.io/mlx/build/html/python/fft.html)与[Real API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.real.html)仅作补充；读取时文档显示v0.32.4，本轮实际wheel/tag为0.32.0，不混作同版本证明。

## 准入、复核与后续

每child导入MLX前、每formal trial前均取3个driver样本，Device≤5%才进入；最后样本到formal开始≤2秒。保留22个先前busy窗口，部分紧随自身warmup/上一trial/child，不能归因为22次外部任务。附pid/pcpu/comm脱敏快照，无argv、不终止别人的进程；没有连续per-PID独占、频率/温度控制或物理GPU timestamp。

四正式child、offset child和pilot均正常退出。所有104数组前后hash相同；不加载checkpoint，不安装依赖、不改上游或产品默认。pilot两trial只作可运行性观察，14个pilot cycles明确排除；正式trial冻结后没有挑选有利试次或删掉offset拒绝。

运行探针需显式`--run`；标准库整理器重算所有phase/ratio/SSE/energy/gate、plane-wise距离三角界、原small-source输入/参考bytes、mask/scalar、hash与终态。它**不重跑完整大shape NumPy/GPU数学**，不能把离线统计一致性当独立重算全部原始GPU值。默认tests无NumPy/MLX/上游、模型或设备执行。

接入真实模型前仍需：固定branch/geometry/mask身份与异常/取消事务；完成独立residual的物化再提交，不能保留会被外部输出buffer覆盖的lazy结果；比较真实rFFT/异步或融合强基线、完整model budget与视频语义/运动/negative-prompt质量。当前只支持这组ready FP32 adapter成本和数值观察，不授予FasterCache、SeaCache、ANE或任一视频生成路径的新资格。

## 交付复核：2026-10-17

标准库CLI独立重新整理与公开16,110,625-byte JSON逐字节一致；S47七个Git对象重新收集与ledger一致，累计48条/190个文件hash条目。版本登记绑定实际harness、GPU evidence及父CPU evidence；匹配M4 Max/64GiB/MLX0.32.0/build25G83仍只是历史context匹配，不授予采用资格，变化时需按完整scope重开实验。
