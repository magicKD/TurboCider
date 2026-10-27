# MLX TeaCache：真实CPU gate、CFG配对与生成事务

<!-- knowledge
id: technique-algorithms-case-studies-mlx-teacache-lifecycle
type: case-study
stage: dit
operator: sampling, attention, layout
backend: mlx, cpu-reference
device: cpu, gpu
symptom: stale-cache, wrong-layout, precision-loss, measurement
semantics: approximate, contract-only
representation: fp16, fp32
boundary: request, producer-consumer
api: reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [TeaCache机制](../methods/teacache.md) · [缓存身份](../methods/caches.md) · [实现技巧](../implementation-patterns.md) · [实验](../../examples/README.md)

2026-10-16第八轮，固定[IonDen/mlx-teacache S42](../sources.md#s42)，补充此前S03的只读研究。执行的是原gate/cache/stats/window/lifecycle/pairer，以及Qwen的原prelude/signal/body-loop/tail/forward函数；tensor运算为实际MLX0.32.0 **CPU**，不是NumPy仿真或no-op backend。模型层则是两个明确的affine替身，不是Qwen权重；没有导入mflux、执行apply/proxy安装或上游package initializer。

保留14个gate控制和两seed、30个generation包络：188次forward尝试、184次完成、264次替身block调用。独立标准库整理器重建全部输出元素、state、分支计数和提交/清理；**没有GPU计时、完整模型或媒体资格**。

## 真实被跳过的边界

Qwen adapter每步两次调用：positive计算一次gate，negative复用决策，但两者分别读写`cached_residual`与`cached_residual_neg`。prelude与当前tail始终执行；encoder prep、RoPE及block loop放在compute-only body，hit时也被跳过。step_counter只在negative完成后加一，pairer在每个branch返回时翻转。

signal由image prelude、timestep modulation和block0 image norm得到；在正负分支共享当前latent/time的前提下，它不依赖正负encoder内容。共享decision不是共享residual，更不是省去CFG的negative branch。

本轮替身：`img_in=2h`；两个block分别加`e`、`2e`，其中e是encoder均值；当前tail为`2*(body+t/8)+1`。因此独立full oracle为`4h+6e+t/4+1`。positive取e=1、negative取e=−2，residual分别3与−6；h随step改变，tail也随t改变，能抓出“直接返回旧总输出”和串用分支cache。signal刻意固定为1、系数刻意全0，只为可靠制造hit，**不是作者默认校准或模型稀疏性样本**。

| 5步fixture | prelude | signal | encoder/RoPE各 | block调用 | tail | 记录的逻辑决策 |
| --- | --- | --- | --- | --- | --- | --- |
| threshold=0 | 10 | 0 | 10 | 20 | 10 | 全部computed |
| live，首尾各1 | 10 | 5 | 6 | 12 | 10 | forced / computed / skipped / skipped / forced |

两条路径的全部toy输出相同，且命中期间当前h/t仍进入输出。这是实际省调用的CPU接缝控制，不是普通模型会同样零误差或更快的证据。

## dtype、forced窗口与安全控制的真实顺序

原`mean_abs_rel_l1`先在输入dtype算`current−previous`，再把abs结果转FP32做reduction。FP32 reduction不等于FP32 subtraction。实际MLX CPU上，有限FP16的+65504与−65504先减出Inf；gate识别数值失败、清掉两分支residual和累积量，不用`max(0,NaN)`把它伪装成零变化。相同数值的FP32对照得到rel-L1=2。多项式由有限大系数溢出的另一个控制也保留numerical-miss。

其它原gate观察：

- threshold≤0总compute且不更新cache；领先/尾随forced也不更新residual。forced判断在live nonfinite处理之前，不能把“数值失败总会清状态”写成无条件命题。
- 本轮直接给forced gate预置旧residual与NaN signal时，旧状态保留。这是低层优先级控制；正常generation先由lifecycle reset，尾窗之后也不再进入gated步骤，不能据此断言正常生成正在复用坏cache。
- below-range的实测delta=0，polynomial输入仍clamp到下界.125；above-range即使predicted=0、threshold=100也强制compute。自定义系数未给range时则不继承该保护。
- accumulator恰等threshold时compute；负poly被clamp到0；达到8次连续skip后compute。失配range在threshold=0短路之前拒绝。

14例均由独立`struct`输入dtype减法、Horner和state oracle复核。FP16观察不等于已经跑过模型BF16 gate或GPU路径；任意自定义系数也没有模型质量保证。

## active窗口：接受、零命中与零调用不是一回事

img2img nominal=8、init=4时，lifecycle记录active=4；实际t为4/5/6/7，gate step_idx仍为0/1/2/3，首尾保护相对本次生成而不是nominal schedule。观察forced/computed/skipped/forced，共8个branch输出。

active=3、首尾各1可被接受，但中间只够seed，不能消费cache；实际零skip并发出no-benefit warning。active=2则首个forward在prelude前拒绝。active=0的generation根本不调用transformer，可以正常提交空generation；不要拿单独window checker的拒绝代替完整lifecycle的零步语义。

## after-loop不是事务提交点

`call_before_loop`递增generation token、按active count清两分支cache；`call_after_loop`只设置PendingFinalize，并在VAE之前release数组引用。只有包裹的generate自然返回后才提交staging统计；中断或晚到异常均清context、release数组并discard统计。

本轮真实原方法验证：

- positive刚写入、negative还没执行时触发KeyboardInterrupt：当时cache不对称，但退出后数组和staging清空，public generations仍0。复用同一handle的新generation token从1到2，pairer从pending-negative恢复positive，重新seed。
- 完整loop与after-loop回调结束后再抛异常：5条staging decision不进入public counters。
- 只有2条decision却声明active=5：原finalizer拒绝提交；在本轮默认warning处理下，wrapper返回已完成的toy结果并发RuntimeWarning，不把统计错误变为模型结果。
- 生命周期callback被移除：包装器在原generate调用之前拒绝，而不是悄悄运行无reset的缓存路径。

所有30个退出点的三个数组字段、pending/context和staging均被检查为空。它证明的是Python引用/状态合同；不证明allocator已缩小、物理内存已回收或Metal异步消费者已完成。probe每次snapshot还显式eval/readback数组，**不用于lazy内存、host同步次数或性能测量**。

## 三个刻意违反前提的诊断，不能写成正常pipeline缺陷

本轮包络能够人为漏调用、换分支shape或改变prompt；真实mflux模型没有在这里执行。以下限定为检查集成前提的反例：

1. 最后一步只完成positive就调用after-loop并自然返回：5条decision仍可提交，但只有9个branch输出，step_counter=4、pairer仍在negative。原统计只检查decision长度，不能单独证明最后一对CFG完成；正常orchestrator的“两次调用”合同仍是必要证据。
2. step2只改变negative encoder均值−2→−3，而image signal不变：gate继续hit，该toy negative输出相对当前full oracle差6。它说明每请求条件稳定的前提；不是作者在默认固定prompt上失效的证据。
3. step2 negative image长度2→1，旧residual长度2可以广播，原包络返回长度2而非full oracle的长度1；相同变化在positive signal路径会被shape guard拒绝。正常CFG两分支共享latent，不能因此宣称常规请求支持不对称shape或实际发生了回归。

因此查询“每步一条记录”“生成自然返回”“输出finite”都不能独立授予有效generation资格。新原创case `cache-lifecycle-contracts`验证更强的显式branch ledger：同request身份、每logical step正负顺序、相同image shape、两条tail都完成才可提交。这是本库建议，不修改上游，也不冒充其完整控制器。

## Apple迁移与评测建议

从该实现最值得迁移的是：threshold-zero独立fast path、compute-only preparation、每branch residual、active窗口和generation事务。`store_residuals`显式eval真实cache，避免仅eval输出却留下它的lazy sibling；但本轮没有测其buffer donation或峰值收益。

`TeaCacheStats.speedup_estimate`在5步/3个full的fixture给5/3，它是少算步数估计，不是测量。原创成本witness把每步prelude/tail、gate和剩余body加回：同样skip数既可以有收益，也可以被gate成本反超。不能把此属性写进Mac测速表。

真正的下一层验证需要固定mflux/MLX/权重/量化、未参与拟合的prompt与真实完整请求，比较vanilla、wrapped threshold0和live三路，并计入materialization、allocator冷暖、编译、CFG、取消/异常和VAE。这里的Qwen是所读adapter的模型合同；**不授予Qwen-Image-2.1支持或沿用旧checkpoint校准**。

上游源码只从显式固定Git对象读取，未分发；LICENSE为Apache-2.0。默认tests与整理器均不导入MLX/mflux、不读取真实activation、不执行上游或设备。源码probe在MLX初始化被沙箱拦截后经授权在沙箱外运行，default device与工作stream始终选CPU；这不是Metal不支持或GPU测速失败结论。

## 本轮验证快照

2026-10-16：第八轮21项测试通过，全部算法专题125项通过；原始CPU记录经独立CLI重新整理，与公开JSON逐字节一致。43条/149个文件hash的来源结构及source Markdown生成一致性、检索索引和`git diff --check`通过。
