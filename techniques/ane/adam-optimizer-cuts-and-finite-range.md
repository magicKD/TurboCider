# Adam optimizer-only缩减：有限错误、输出切口与范围改写

<!-- knowledge
id: technique-ane-adam-optimizer-cuts-and-finite-range
type: method
operator: gemm, scheduling
backend: private-ane, cpu-reference
device: cpu, ane
symptom: precision-loss, measurement
semantics: contract-only
representation: fp16, fp32
boundary: weight-runtime, producer-consumer
api: private, reference-only
-->

[ANE目录](README.md) · [前批完整Trainer/Adam](adam-moments-loss-scaling-and-checkpoint.md) · [7文件来源](sources.md#adam-cuts-sources-20261017) · 全量输入/输出/逐项审计 · [验证记录（过程原件已移出）](../MAINTENANCE.md)

2026-10-17，沿用M4 Max / macOS26.6.2、固定ANEForge `896c4943c5ebf811c0e815f8d1381fd1e157fa7d`与既有精确hash的e5rt shim。四个新监督进程完成1,088次显式模型求值，全部输出finite；原optimizer-only完整W/m/v **128/128复现前批Trainer**，包含原来被拒绝的更新。失败不需要重新执行VJP就能复现；不据此宣称已定位物理指令或修复上游。

## 可复用的方法：把错误缩减到最小更新图

1. 捕获失败时完整的 W/m/v、实际梯度和学习率；独立重放 optimizer，不让重算梯度或新轨迹替换原问题。
2. 逐次改变路由、输出打包或算术表达；新增诊断输出视为新图，先验证其主输出是否仍复现旧错误。
3. 同时保留 zero、小值、大值和正负号控制；finite、单坐标改善或新 epsilon 都不能代替完整更新门槛。

适用于训练更新偏差的定位，不是通用 Adam 修复或已证明的加速。下文是这套方法的具体案例与拒绝条件。

## 1. 缩减边界：用真实已观察梯度，不重新求一次梯度代替它

前批的两seed、device模式、default epsilon、S=1/32/256/1024，每run四代、两个参数，共64组真实before W/m/v、实际scaled梯度和after W/m/v。本批分别在两个独立进程、正反输入/arm顺序复查，得到128个原stack状态三元组比较。

所有输入都来自这些**完整捕获数组**，按原调用协议转换到FP16端口，lr也取原bias-correction后的FP16值。每case独立feed，不让新输出变成下一个case输入；不是新的训练轨迹、checkpoint恢复或长期复现。不使用resident内部梯度，原字段仍为unobserved。

另加每进程两个自有完整32×32控制：zero-gradient，以及含正负0、微小梯度到±512的signed-range bank。16种幅度×正负号按seed置换，W各通道不同；这不是模型自然梯度分布或独立随机总体。总共每进程34个case×8种图，四进程；32个显式program对象不等于内部compiler调用数或物理dispatch计数。

## 2. 原helper、编译入口、输出打包与新增切口分开

原[_adam_update](https://github.com/sbryngelson/ANEForge/blob/896c4943c5ebf811c0e815f8d1381fd1e157fa7d/aneforge/autograd.py#L610)不改动；原[_stack3/_split3](https://github.com/sbryngelson/ANEForge/blob/896c4943c5ebf811c0e815f8d1381fd1e157fa7d/aneforge/autograd.py#L634)保持自然宽度`[96,32]`。本库分别调用：

| arm | 图与执行入口 | 136份W对各自名义实数配方拒绝 | 128份旧W/m/v完整byte复现 |
| --- | --- | ---: | ---: |
| source-stack | 原helper+原stack，默认routes | 36 | 128 |
| source-stack-opt0 | 同图，显式opt=0 | 36 | 128 |
| source-multi | 原helper，compile_multi三个输出 | 36 | 128 |
| source-cuts | 原helper，另增加四个诊断输出 | 36 | 128 |
| divide-first | `W−(m2/denom)*lr_t`，opt=0 | 36 | 0 |
| variance-range32 | 放大方差表达，再缩回sqrt，opt=0 | 92 | 18 |
| range32-divide-first | 上一表达+先除后乘 | 88 | 0 |
| epsilon-floor | 原helper但eps改为1e-4，opt=0 | 32 | 16 |

原stack与opt0的发射MIL在每进程都**逐字节相同**。multi/cuts的图和输出接口不同，但34个case全部W/m/v与原stack相同，共408组额外完整状态匹配；正反序新进程共544组完整输出（含cut）逐字节匹配。这削弱了“仅由本批routes、concat或状态别名导致”的解释；本批根本没有share_buffer状态反馈，不据此排除所有更大图、其他编译器或并发问题。

128不是128个独立seed/训练batch：64个原参数状态各复查两次。四种原表达各有32个旧case更新拒绝和4个signed-range拒绝；zero控制通过。所有arm共392个W对各自配方拒绝，32个moment数组拒绝（signed-range的v，每arm每进程一份），**全部仍finite**。

## 3. 新诊断图的零denominator与有限delta：观察和解释不能越界

`source-cuts`新图输出W/m/v之外，还输出numerator、sqrt(v2)、denominator、delta。它使用相同原helper节点，但新增输出可能影响编译器；这些值属于**新编译诊断图的端口**，不是偷读前批Trainer内部执行。W/m/v全量一致是此批的实测控制，不是任意tap永远无影响的定理。

seed17/S1/generation0/parameter0，完整array中的coordinate263：

| 量 | 完整捕获/新图端口值 |
| --- | ---: |
| 原真实scaled梯度 | −0.0052642822265625 |
| W_before / W_after | .11328125 / .14453125 |
| m_after / v_after | −.00052642822265625 / 0 |
| numerator | −4.76837158203125e−7 |
| sqrt_v / denominator | 0 / 0 |
| delta | −.03125 |
| 前批名义source参考W_after | .11627197265625 |

实际发射epsilon因FP16 scalar先舍入为0。新诊断图全部139,264个坐标中，7,536个denominator端口为零：2,342个numerator非零、5,194个也为零；这些delta均finite，**7,536个数值全部符合`half(numerator*65536)`**，该经验式不另宣称±0的bit相同。它是这个图/输入域的全量经验描述，不是IEEE除法定义、ANE硬件规格、底层reciprocal实现证明或可以依赖的安全除零合同。

zero case的0/0端口对应delta=0，并不意味着真实小梯度就可靠；小梯度的有限大偏差更需要gate。也不能把FP16输出为零倒推“所有原内部算术每一步必定FP16”。本批没有独立loaded-image、指令、物理placement或counter取证。

## 4. 两个看起来合理的重排为什么仍不够

在实数中`lr*m/denom`和`(m/denom)*lr`相等；实际divide-first没有复现任何一组旧完整W/m/v，仍36/136份W被拒绝。它会改变微小更新/舍入，却未获得完整数值资格，不能仅凭某个坐标更好就作为fix。

范围候选为：

```text
v2_for_output = .999*v + .001*g²
q = (.999*1024)*v + .001*(32*g)²
denom = sqrt(q)/32 + eps
W2 = W − lr_t*m2/denom
```

实数中`q=1024*v2`，但编译图的中间幅度、scalar舍入与运算路径变化；m/v输出仍沿用原helper，不更改历史状态配方。候选在部分小值上接近参考，却在大值/旧高scale输入上大量失败，92/136或88/136拒绝，比原表达更多。固定放大不是自适应稳定算法，也未证明到底是哪一个硬件操作造成这些差异。

signed-range的g=512时，名义实数v2=.001×512²=262.144（checkpoint half为262.25），新原cut图实际v为65.5625、delta约.00600052而非约.003。它仍finite，所有arm的v都保留该拒绝。可提出“中间范围受限”的后续假设，但没有直接捕获原执行的g²，不应把某个推测的clamp阈值写成芯片规格。

把epsilon改成1e-4也不是等价变换：对**自己的新配方**仍32/136份W拒绝；对原默认epsilon目标则46/136拒绝，分开存储。不能用新配方的改善掩盖旧目标不再等价，也不能据少数finite输出推荐全局调参。前批完整scale sweep和64份scaled梯度absolute拒绝原样保留，本批没有重跑或晋升其训练合同。

## 5. 验证、身份、适用域与下一步

本批保存3,264份W/m/v和544份额外cut矩阵，3,899,392个完整输出元素；708个唯一FP16数组编码只做lossless去重，每次调用/失败仍保留。输入与旧输出均从前批完整tensor逐字节重建，连±0都不能用Python数值相等冒充byte相同。

W gate仍为delta relativeL2≤.02且maxabs≤.002，moments relativeL2≤.02；新cut诊断单独采用relativeL2≤.02且maxabs≤.02。零参考必须精确零误差。reference只模拟**名义实数系数、实际FP16输入/lr和最终checkpoint舍入**，不是逐op half、实际序列化算术或硬件仿真；epsilon-floor的目标明确另列。

四种额外cut各136份完整数组，对该名义实数诊断门槛的拒绝分别为numerator 0、sqrt_v 8、denominator 12、delta 36；这些也不称全部通过。它们与W/m/v拒绝有重叠，不相加成独立坏case数；例如zero控制的W/m/v完全noop，但发射后zero epsilon与名义1e-8不同，denominator相对门槛仍拒绝。

原代码只在ignored输出复制/调用，每进程全部46个Python文件及两个native来源文件匹配固定Git，沿用shim SHA `1822a6eda222cd48efd6c4fcf9cf127514393c819e300d58626ab9261e6d33b1`。七个selected语义/政策/许可文件不是46文件全部语义审计。collector初版fixture/reference core保持原SHA，新增portable验证器只追加在明确marker之后，curator精确重建旧prefix核对，不把新代码回填成采集版本。首次curator多加一个换行导致prefix身份拒绝，修正后独立3.9/3.11raw审计一致；没有重跑设备或更改数据/gate。

本地实现：显式collector、、、portable tests。CPU入口不启动framework/private/network：

没有新Trainer轨迹、RNG/数据恢复、UnrolledTrainer、真实任务、优化GPU基线、速度或能耗测量。下一步应使用独立除法/方差边界图和不同输出接口检验经验范围，严格分开诊断与原程序；任何候选都要在完整VJP+optimizer轨迹、原质量gate和相同checkpoint完成边界重新验收。source/OS/SDK/shim、表达式/eps/scale、shape、owner或消费者变化均重开。整体研究继续，不给本批候选通用修复/训练加速资格。
