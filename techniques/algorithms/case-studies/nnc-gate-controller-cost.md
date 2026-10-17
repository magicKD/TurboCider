# NNC gate 成本：缓存分母、合并读回与刷新开销

<!-- knowledge
id: technique-algorithms-case-studies-nnc-gate-controller-cost
type: case-study
stage: dit
operator: sampling, layout
backend: metal, cpu-reference
device: gpu, cpu
symptom: launch-bound, stale-cache, measurement
semantics: numerically-equivalent, approximate, contract-only
representation: fp32
boundary: producer-consumer, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [Draw Things原生控制器](../methods/draw-things-caching.md) · [实验入口](../../examples/README.md) · [S56](../sources.md#s56)/[S57](../sources.md#s57)

2026-10-17，第十八轮。结论不是“缓存分母总能更快”：相比原串行gate有收益，并不证明比合并读回的对照更快。一次query窗口里，候选的总统计量没有减少，却多一次读回；三次query时结果也有明显波动。以下只测 **ready signal 的gate窗口**，不是原H3首块/tail/head、实际cache replay、完整模型/图像/视频速度或质量。

## 实现与固定比较

继续使用固定Draw Things `8d9c62a` 的完整未改H3 `TeaCache.swift`、s4nnc `f64e4a6` / ccv `66dc16a`。新增Swift客户端使用真实NNC FP32 tensor/abs/mean/Concat与host readback；新准备器复用未改第十七轮准备器，只替换自有客户端，不改上游。启动器与Native的setup/trial均做三样本Device≤5%准入，只保留无argv进程快照；正式数据131个通过窗口（外层1、setup10、trial120）且无忙窗口，不构成连续独占/CPU空闲证明。

四臂是两个因素的对照，另有host timer/bookkeeping floor：

| 臂 | 分母 | 统计读回 | 一次commit+3query的mean链 / readback调用 |
|---|---|---|---:|
| source-sequential-four | 每query重算 | 原source依次4次 | 12 / 12 |
| batched-four（预固定主分母） | 每query重算 | 4标量打包一次 | 12 / 3 |
| cached-sequential-two | commit时保存2个host norm | query依次2次 | 8 / 8 |
| cached-batched-two | commit时保存2个host norm | commit/query各打包一次 | 8 / 4 |

合并读回采用hoisted、fixed-arity的packing Model；stat操作仍是公共动态abs/mean，没有声称完整统计图融合。Concat底层创建output aliases与FORMAT_TRANSFORM节点，不是零成本，更不是“一个Concat等于一个shader”。表是请求的逻辑API链与读回调用，不是physical dispatch/barrier计数。

一次query时，batched-four与cached-batched-two均要4个mean，后者却有2次读回/packing，前者1次。只计候选命中query而漏掉commit norm会制造收益。source commit用step0强制anchor更新，随后1/3个query在窗口内；本实验没有调用reduced head来消费skip budget，不能把gate query数说成实际跳过的denoiser/block次数。

## 测试矩阵、数值与观测边界

M4 Max / 64GiB、OS build25G83、SDK26.5、Swift6.3.3。三组 `(audioRows,videoRows,channels)` 为`(2,256,64)`、`(8,1024,128)`、`(16,4096,256)`，两种query数，共六cohort。tensor为`[1,rows,channels]`的原创signed/magnitude pattern，不是实际H3 activation。每cohort有8份完整ready tensor；CPU重建全部little-endian FP32 bytes的SHA，前后不变。

每组20个预固定旋转/反转配对trial、四计算臂加host floor；600个窗口、1800个commit/query子区间，其中480个计算窗口/960个gate query槽，120个floor窗口/240个floor query槽。完整stat参考为144个scalar，独立FP64/FP32配方核查；7个signed阈值点、修改anchor/代际与构造器shape控制分开。另保留CPU实现控制及60个排除的GPU pilot窗口，不拼入正式计时。

计时为host-ready wall，含本臂controller、commit/query和统计readback，排除显式setup、warmup、bank hash、ioreg与JSON。没有GPU event/ISA/能耗/峰值测量。host floor有0.77–3.52µs窗口中位数；53个无GPU工作的子区间落在计时tick内，原样保留0，不填成1ns。真正GPU统计/读回区间均为正。

## 结果：强对照与配对估计改变结论

下表为全窗口中位µs和预固定的**每trial baseline/candidate比值的中位数**，>1才表示候选较快。IQR是P25–P75，不是置信区间。

| rows / channels | query数 | batched-four µs | cached-batched-two µs | 配对倍率 | 配对IQR |
|---|---:|---:|---:|---:|---:|
| 2 / 256 / 64 | 1 | 367.73 | 480.90 | 0.751 | 0.393–0.904 |
| 2 / 256 / 64 | 3 | 1028.90 | 895.62 | 1.096 | 0.647–1.260 |
| 8 / 1024 / 128 | 1 | 1256.33 | 1718.79 | 0.781 | 0.506–0.960 |
| 8 / 1024 / 128 | 3 | 2809.60 | 2596.27 | 1.120 | 0.517–1.688 |
| 16 / 4096 / 256 | 1 | 1117.67 | 1625.98 | 0.762 | 0.412–1.058 |
| 16 / 4096 / 256 | 3 | 2632.77 | 2506.50 | 0.933 | 0.638–1.580 |

相比原source-sequential-four，cached-batched-two的配对中位倍率约1.30–2.27；换成强对照，一次query全部变慢，三次query的两个中位数略高于1、一个低于1，且三个IQR都跨1。**不授予稳定赢家/默认启用资格，也不把不同shape的非单调时间解释成硬件规律**。

最后一行尤其不能用`2632.77/2506.50≈1.05`替换实际配对中位数0.933；它们是不同统计量。所有原始pair/min/max/IQR与phase均在中。观察到波动，不删除慢trial或挑选噪声冠军；CPU/driver/runtime负载与连续独占均未完全隔离。

当时还有一个未定量的成本边界：ccv对完整input params做memcmp，变化可走compile/absorb；reduce自动shape仅修改指定轴而保留其它metadata。本客户端cached packing Model同时用于commit norm和query difference，本轮未instrument timed重建次数。后续[第十九轮只读签名诊断](nnc-packing-signatures.md)在CPU/GPU各201次真实packing中发现：同几何共享角色69次warm参数匹配、0次mismatch，跨几何正控各两次mismatch。**不支持把角色交替引起的参数重编译作为本轮波动的原因**；它不是直接compile函数计数，也没有证明所有隐式重规划均不存在或改写这里的性能结论。不能只按mean数比较GPU内核效率。

## 修改anchor会改变判定：代际是正确性合同

在同构造协议下，先存anchor norm为`[.546875,1.125]`，原位把anchor改成`[.5,1]`，current设为`[.53125,1.0625]`。source与batched-four的实际相对变化均为.0625，严格`<`所以拒绝；两个缓存分母臂若仍用旧norm则错误返回true。显式推进调用者提供的anchor generation后，候选拒绝并重新commit，四臂均false。

这证明unsafe版本不等价，不证明它在正常不可变anchor下必然错误。代际检查不自动发现任意内存写入，仍要求生产者正确提供token；它也可保守地多full，不是与原策略任意输入完全相等。真实模型/音画质量尚未评估。

## 相同可见shape不一定是相同原生描述

开发期反例最初全false，并没有证明norm-cache安全：两个构造路径生成的`[1,2,16]`/`[1,3,16]`数组相同，原生TensorShape比较却不同。ccv CPU三维NHWC dense-matrix桥接在终止0之后的`dim[4]`保存row step；raw路径保留`[128,192]`，direct graph.variable路径为`[0,0]`。s4nnc原生相等比较全部12槽，数组显示只投影有效维度，source guard先拒绝。

CPU及GPU控制均保留该差异，未将guard改成数组比较。统一producer构造协议后才测到上述真实unsafe-hit；正式所有bank同构造、完整hash核查，不能把拒绝快路径冒充正常gate加速。这是固定ref的表示边界观察，不是所有版本/实际H3模型错误或未知padding内容泄露。

## 保留的失败与继续方向

可迁移设计是“immutable/versioned anchor、合并小统计读回、把刷新成本纳入分母”，不是盲目再加一层cache。第一块、tail refresh、cached head和真实sampler仍需完整测量；不能把本组件倍率乘到H3全请求。ANE/MLX/MPS之间也不能仅因数学相似就转授资格。

复现入口：`nnc_gate_cost_prepare.py --help`生成新目录，再用独立scratch构建；`nnc_gate_cost_run.py --help`显式CPU/--gpu/--pilot及binary SHA绑定；独立整理器只读JSON/来源，不运行Swift/GPU或读权重。默认学习/回归：
