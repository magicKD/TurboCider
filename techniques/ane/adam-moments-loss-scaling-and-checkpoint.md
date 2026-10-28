# ANEForge Adam：非线性VJP、loss scaling、完整moments与状态恢复

<!-- knowledge
id: technique-ane-adam-moments-loss-scaling-and-checkpoint
type: method
operator: gemm, scheduling
backend: private-ane, cpu-reference
device: cpu, ane
symptom: precision-loss, lifetime, measurement
semantics: contract-only
representation: fp16, fp32
boundary: weight-runtime, producer-consumer
api: private, reference-only
-->

[ANE目录](README.md) · [autograd总览](training-autograd.md) · [线性SGD历史实验](resident-training-lab.md) · [8文件来源](sources.md#adam-sources-20261017) · 全部数组与逐次审计 · [验证记录（过程原件已移出）](../MAINTENANCE.md)

2026-10-17，固定ANEForge `896c4943c5ebf811c0e815f8d1381fd1e157fa7d`，M4 Max / macOS26.6.2 build25G83、SDK26.5、Python3.11.16、NumPy2.0.2。原Trainer的host/device/resident三条Adam路径均实际执行；48个独立监督进程正常退出并返回release。沿用既有精确hash的e5rt shim，不重新build、安装包、下载数据或修改上游。

结论不是“finite就正确”或“scale越大越好”：默认epsilon的小型非线性模型中，device/resident共128份参数更新有32份未过预先固定的source-reference gate；另有64份直接观测scaled梯度未过absolute gate。同批全部W/m/v与标量loss仍finite。较大epsilon控制另揭示**source算法和标准Adam的epsilon位置不同**，不能把符合源码配方当成符合标准Adam。完整恢复W/m/v/t的24次续步逐字节复现baseline，48次错误恢复续步均不复现。

## 可复用的方法：驻留更新必须连状态一起验收

1. 将动态参数、device optimizer 与 resident optimizer 分开比较，固定相同 K 步和 checkpoint 完成点。
2. 使用变化梯度，逐代核对 W/m/v/t；loss scaling 同时检查 epsilon 与 bias correction 的配方。
3. 先验证单步，再验证同 owner 恢复和连续状态；finite loss、零梯度 noop 不能代表真实更新正确。

这是一套驻留训练的验收方法；下文失败案例不提供默认 epsilon 或 scale 的通用推荐。

## 1. 从动态参数到三种Adam执行边界

原[autograd.py](https://github.com/sbryngelson/ANEForge/blob/896c4943c5ebf811c0e815f8d1381fd1e157fa7d/aneforge/autograd.py#L727)把parameter作为runtime input；每代feed新数据、当前参数和lr。VJP反向遍历图，fan-out处累加。host/device逐参数编backward，resident把backward与更新接进multi-output图，以输出alias回W/m/v输入。这是buffer反馈协议，不意味着物理片上常驻或任意in-place算术。

| 模式 | W与moments的实际可见边界 | 本批直接观察 |
| --- | --- | --- |
| host | FP32 master W/m/v；前向/梯度端口仍FP16；host先unscale梯度 | 原optimizer调用前完整scaled梯度，全部W/m/v/t，独立scalar loss |
| device | 图端口FP16，host持有FP16 m/v shadows；每参数另执行更新图 | 原device Adam调用前完整scaled梯度，全部W/m/v/t，独立scalar loss |
| resident | fused更新，W/m/v经原share_buffer回到输入 | owner仍存活时copy完整moments view及W/t，独立scalar loss；内部梯度未观察 |

W checkpoint统一存为`parameter.attrs['value']`的FP32容器。非host第一次更新的实际wire起点是`half(W)`，不是未经舍入的master；验证delta时明确使用这个起点。host/device observer仅包装当前实例的原更新方法，记录后调用原实现，不修改原文件。resident字段保持null，不能拿另跑的backward当该fused step的观测梯度。

原[device Adam构图](https://github.com/sbryngelson/ANEForge/blob/896c4943c5ebf811c0e815f8d1381fd1e157fa7d/aneforge/autograd.py#L575)用natural-width拼接状态再slice，resident则保留multi-output。编译成功、输出形状正确、mask=4和有限性都不能单独证明更新数值或物理ANE归属；本批没有placement trace、PMU、能耗或训练计时资格。

## 2. 两矩阵tanh与变化梯度，而不是恒定梯度单步

所有矩阵32×32，lr=.003、betas=(.9,.999)，seed17/29：

```text
H = tanh(X W0)
Y = H W1
L = 0.5/32 · sum((Y − T)^2)
dW1 = Hᵀ (Y−T)/32
dW0 = Xᵀ [((Y−T)/32 · W1ᵀ) ⊙ (1−H²)]
```

两个可训练矩阵，每run四个不同batch。teacher由独立FP64 tanh模型确定生成，target存FP16；每一步在**实际当前wire权重**重新计算完整analytic梯度。没有bias、norm、dropout、Conv、真实数据或预训练checkpoint，不将四步称作收敛。

| profile | epsilon / scale | 进程数 | 每进程显式step |
| --- | --- | ---: | ---: |
| zero | 默认1e-8 / 1 | 6 | 1 |
| scaled | .1 / 1、32 | 12 | 10，含6个恢复分支step |
| mlp | .01 / 32 | 6 | 4 |
| mlp-default | 默认1e-8 / 1、32、256、1024 | 24 | 4 |

`scaled`采用线性目标`sum(W*G_input)`，四代预先固定的dyadic梯度bank含zero、sign变化，scale1/32使用同一未放大梯度。两个seed的bank索引因mod4相同，但初始化W不同；不宣称两份独立随机gradient序列。MLP的数据/参数则随seed改变。

总共246次保存的`Trainer.step`和246次单独`trainer.loss()`，包含72个恢复分支step，**不是246个连续训练batch**，也不是compiler或硬件dispatch数。366份参数更新、244份直接观察的完整梯度（249,856元素）、1,098份完整after W/m/v数组（1,124,352元素）均保留。没有保存实际H/Y内部全量，只保存其完整scalar loss；不称全forward activation验证。

## 3. epsilon还有bias-correction因子，不能只除以scale

设loss scale为S，source device/resident的moments存放放大梯度后的状态：

```text
m2 = b1 m + (1−b1) g_scaled
v2 = b2 v + (1−b2) g_scaled²
bc1 = 1−b1^t; bc2 = 1−b2^t
lr_t = lr sqrt(bc2)/bc1
W2 = W − lr_t m2/(sqrt(v2)+eps)
```

令未放大moments为m̄=m2/S、v̄=v2/S²，实数代数得到：

```text
W2 = W − lr (m̄/bc1) / (sqrt(v̄/bc2) + eps_eff(t))
eps_eff(t) = eps / (S sqrt(1−b2^t))
```

因此不仅有eps/S，还有把bias correction折进lr所带来的时间因子。若希望这个raw-moment公式等效于固定标准Adam epsilon，denominator中的epsilon要动态取`eps_target*S*sqrt(bc2)`；不能再把Adam lr除以S。这里只解释实数等价，**不是已经修复上游、验证新训练配方或保证FP16实现等价**。

本批每个更新同时报告两个条件参考：

- `source_update`：实际before W/m/v/t和本次真实观察梯度（resident仅analytic）代入名义source实数系数；考虑FP16 wire W/lr及输出checkpoint舍入。
- `canonical_epsilon_at_same_state`：在同一实际状态归一化moments、同一梯度下，只改成标准Adam的epsilon/bias-correction位置。不是从t0另跑的完整标准Adam训练轨迹。

两个reference均**不是逐op FP16仿真或序列化/hardware算术的精确模拟**。发射出的beta/epsilon常量、运算融合及内部精度可能另外引入差异；审计不把某一差异武断归因于单个物理算子。

显式梯度eps=.1控制：host在S1/32的四代W逐字节相同；device/resident两scale不同，最大W差.0068511962890625。source-reference通过，但标准epsilon counterpart常失败。eps=.01的MLP也通过source门槛，而每个device/resident run有6/8份parameter updates未过canonical counterpart；这不能代替默认epsilon测试，所以另做完整default sweep。

## 4. 默认epsilon：保留非单调失败及finite零梯度正控

default MLP在device+resident合计，每个scale32份parameter updates：

| S | source-reference拒绝 | 最大update-delta relative L2 |
| --- | ---: | ---: |
| 1 | 22/32 | 2.1020375 |
| 32 | 8/32 | .4588393 |
| 256 | 0/32 | .0096368 |
| 1024 | 2/32 | .2133998 |

S1024的两个拒绝来自seed29，不能据S256通过推荐“增大scale即可解决”。host default各scale均过当前source gate；host FP32 state和非hostFP16 state并非同精度benchmark。所有失败仍finite，scalar forward与实际权重的参考也仍通过；这恰好说明loss/finite检查不足以覆盖optimizer。

上表只统计**给定实际梯度/状态的局部optimizer更新**，不是整步gate。直接观察的244份scaled梯度中有64份未过固定maxabs .02，均在default S256/1024、host/device：各scale32份，最大abs .435963740659588。它们的relativeL2仍小于.02；absolute误差是在放大后的梯度单位上衡量，会随S放大，不能据此直接说未放大梯度误差也有.436，更不能丢掉absolute gate后宣称全部通过。S256即使source局部更新0拒绝，**完整梯度+更新合同仍未全过**。resident内部梯度未观察，不纳入这64份也不默认通过。

所有profile/恢复分支的最大直接观测gradient relativeL2为.00376673；source moment relativeL2最大.01115363、无moment gate拒绝；scalar loss relative error最大.000985142。source更新累计拒绝32，canonical local counterpart累计拒绝128。这些拒绝不是运行崩溃，不用“源码配方有解释”把它们改成通过。

zero控制必须另看：[_compile._e_adds](https://github.com/sbryngelson/ANEForge/blob/896c4943c5ebf811c0e815f8d1381fd1e157fa7d/aneforge/_compile.py#L335)先把1e-8舍入成FP16零常量，实际发射MIL确有zero epsilon和real_div。但六个真实zero-gradient run仍全部finite；host W不变，非host只发生首次wire舍入，m/v保持零。**不能仅从dtype/常量推论实际一定0/0、NaN，亦不能据这些positive例推论任意小梯度安全**。本批没有做division-zero硬件因果归属或动态scale算法。

固定gate：gradient relativeL2≤.02且maxabs≤.02；更新delta relativeL2≤.02且参数maxabs≤.002；moment relativeL2≤.02；scalar loss relative error≤.02。参考分母为零时必须误差精确为零；不能用参数总相对误差小、丢掉小更新或放宽gate来掩盖失败。

## 5. 恢复的是完整状态，不只是权重

仅`scaled` profile：baseline做四次更新，在generation1/t2保存完整W/m/v/t；之后用相同预计算generation2/3分别续步：

| restore分支 | W | m/v | t | 完整after字节复现 |
| --- | --- | --- | --- | ---: |
| full-restore | checkpoint | checkpoint | 2 | 24/24 |
| weights-only-fresh-optimizer | checkpoint | 清零 | 0 | 0/24 |
| weights-and-t-stale-moments | checkpoint | baseline结束t4 | 2 | 0/24 |

第三个控制证明即使W/t正确，旧moments仍改变续步；第二个也不是“训练断点恢复”，只是从旧W重新开始一个optimizer。验证器同时检查分支起点、连续before/after linkage和W/m/v/t全部内容，不只比较终点loss。

这是原创**同进程、同owner、同shape、固定配置、预计算数据**的state adapter，直接恢复现有正确大小的输入port；没有finalize后重绑alias、错误size/UAF探测、异步交错或释放后读取view。没有框架checkpoint API、跨进程序列化恢复、dataset位置/RNG/loss-scale调度恢复、overflow原子skip/回滚；不把24次复现扩展成这些保证。device/resident缺少host同级finite guard的源码风险仍存在，本批观察器只是事后检查，不是替它们补上原子保护。

## 6. 可移植复核、身份与下一步

发布全部完整自有tensor，采用有界lossless zlib+hex与raw/compressed双SHA去重；global pool只共享相同dtype/完整字节，不删去失败元素或用少数统计代替全量。portable测试先后暴露本库测试误取不存在的metric元素数、以及遗漏scaled梯度absolute拒绝汇总；已修正测试/汇总/说明，保留原失败receipts，不改设备数据、gate或逐项判定。原程序源/MIL/权重hash、48份terminal receipts及历史collector v1/v2/v3身份保留；v2只增加默认epsilon的同MLP，v3扩scale并安全记录nonfinite loss（本批未触发），旧结果不重新标为新runner。

来源账本选择8个原代码/文档/政策/license文件作语义阅读，另独立核对**每个复制包全部46个Python文件**与固定Git一致；46个identity不等于46文件都语义审计或全部执行。沿用shim SHA `1822a6eda222cd48efd6c4fcf9cf127514393c819e300d58626ab9261e6d33b1`，不是新build或loaded-image attestation。

本地实现：显式collector、、、portable tests。CPU入口不触发框架/private/network：

两seed的四个连续数学坐标各做central difference，共8点，最大abs约2.06e-10，只验证自有FP64非线性reference的导数一致性；扰动不重新round权重，**不是硬件/FP16 master量化函数的有限差分**。设备实际梯度另按完整1024元素独立检查。

继续研究：先缩减default S1/S1024的失败坐标，拆开scalar常量舍入、moments量化、更新delta与fused执行边界，再讨论改配方；保持zero正控与全量失败，不提前认定硬件原因。之后才是相同精度/相同checkpoint边界的K步性能、overflow原子恢复、跨进程完整checkpoint和真实任务holdout。源码/SoC/OS/SDK/shim、S/eps/betas/lr、shape/layout/VJP、状态owner或实际消费者任一改变均重验。研究保持active，本批不发布训练提速或长期收敛结论。
