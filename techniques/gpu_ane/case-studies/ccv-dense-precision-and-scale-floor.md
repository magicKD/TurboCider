# ccv 稠密输入精度：有限输出、行 scale 下溢与相关性负例

<!-- knowledge
id: technique-gpu-ane-case-studies-ccv-dense-precision-and-scale-floor
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [前轮格点正控](ccv-original-rowwise-component-runtime.md) · [本轮来源](../sources.md#ccv-precision) · 独立复核/重跑 · [验证范围（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17；固定 `liuliu/ccv@2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b`，不是最新版本声明。本机 M4 Max、macOS26.6.2/25G83、SDK26.5。实际运行原 MFA GPU、CoreML rowwise 和显式256/256 row-split；不改原库，不调用真实 calibration chooser。本页讨论组件精度，不是模型质量或速度排行榜。

## 结果先行

- 两个非格点稠密 seed 上，GPU 相对 L2 约0.0206–0.0207%，CoreML 约0.455–0.458%，split 约0.321–0.325%。全部通过预声明的**随机稠密诊断**限值，但不再像第二十四轮的友好格点数据那样逐位相同。
- 最大码值正相关输入：外部精确答案127，GPU返回127，CoreML返回128.75，split前半127、后半128.75。全部有限、原 CoreML/split status均1；数值偏差不能因 API 成功而消失。
- 行 activation 为 `2^-19 / 2^-20 / 2^-22` 时，存储的 FP16 row scale 舍入为零，CoreML把本应非零的35328个输出变成零；split仅后半行发生，丢17664个值，GPU路径无此丢失。真零输入行三条路径都正确返回零。
- 新进程重复完整12观察，所有 actual FP16 字节与主实验相同。pilot3观察、主12观察和确认12观察分账；不将重复样本冒充更多独立分布。

## 先纠正 headroom 假设

原 emitter 给 **两个** int8 输入各做 `dequantize(scale=dq)`，其中 `dq=1/sqrt(K)`；FP16 matmul的数学输出是 `sum(qA*qW)/K`，不是 `sum(qA*qW)/sqrt(K)`。恢复 kernel 再乘 `activation_scale * weight_scale * K`。

本轮 K=256，dq=1/16恰好可表示。对 `|qA|,|qW|<=127`，归一化 dot 的绝对上界16129，低于 FP16 最大有限65504。先前拟议的 Wcode32/33 溢出边界在设备运行前撤回；code33对应4191，不是67056。K-split的局部matmul及FP16 add也不能被误读成“只缩放一边”。这是源码推导，不是内部 compiled artifact 或硬件累加追踪。

本轮最大码值没有 Inf/NaN，验证了所选数据的实际有限性；**不等于输出数学正确**，也不证明任意 dtype、bias、H256、不同K或完整FFN都已解决headroom。

## 保持原实现的受控矩阵

M512/N256/K256，batch1，contiguous/offset0，FP16输入输出；weight packet为65536字节 signed-I8 codes，再接256个FP16 per-output-channel scale。无bias/H256/LoRA。三条路径与前轮相同：

| arm | 实际函数/精度边界 | status来源 |
| --- | --- | --- |
| gpu | `encode_scaled_gemm`；自然Apple9 Int8MatMul，FP16 activation直接消费 | void API；1仅为client标签 |
| coreml | `run_ane_rowwise_gemm`；原GPU量化/上传→CoreML预测→GPU恢复 | 原函数返回1 |
| split | `run_ane_na_rowwise_split_gemm(...,256)`；前256行GPU、后256行CoreML | 原函数返回1 |

新 package/build/cache 与第二十四轮隔离，原完整源码只读复用；构建前后重核全部2468个Git blob。原Package及CPU/Metal/CoreML源不改；3文件新读本明确行范围，不称重读整个仓库。Release、4 jobs、`CCV_METAL_LOGGING_ENABLE=1`和client log_level1，仍不是原默认发行构建的性能复现。

输入由标准库fixture在设备运行前生成，绑定预声明计划：

| case | activation / W | 检查目的 |
| --- | --- | --- |
| dense41 / dense53 | deterministic integer mix→FP16非格点A；稠密signed-I8 W；交替1/4096、1/8192 scale | 一般非格点数据的额外A量化与剩余路径误差 |
| maxcode | A=.5；所有Wcode127；scale1/128 | 合法、有限且精确答案127的强正相关负例 |
| scale_sweep | 每行恒 `2^e`，e=-4,-6,-8,-12,-14,-16,-18,-19,-20,-22及真零；Wcode127、scale1 | 存储scale精度、零舍入和真零行对照 |

sweep的 `(row%256)%11` 决定组，后256行完整复制前256行输入，避免把两半的数据差异当作split差异。最大外部答案2032，仍在FP16有限范围内。

每次native调用先把**最终GPU目的buffer**填NaN，显式commit/stream wait后保存完整FP16数组；不是poison CoreML pixel buffer，也没有读回其身份。数值失败不截断后续arm；进程exit0只表示收集完成，original status与数值指标分别记录。

## 稠密随机数据：误差分两层看

独立CPU先解码实际FP16 A、signed-I8 codes和FP16 W scales，FP64重算完整dot。另一份reference模拟原kernel的stored-FP16行scale、FP32 reciprocal/product、nearest-even `rint`及[-127,127] clamp，再以FP64重算量化A的dot；它**不是**CoreML中间值、编译结果或每步内部舍入的读回。

| seed | GPU 对原始A相对L2 | CoreML 对原始A相对L2 | split 对原始A相对L2 | CoreML 对量化A参考相对L2 |
| --- | --- | --- | --- | --- |
| 41 | 0.02064% | 0.45523% | 0.32095% | 0.23612% |
| 53 | 0.02071% | 0.45798% | 0.32497% | 0.23872% |

单纯A量化的reference相对误差约0.39066%/0.39064%；不能与最后一列线性相加，也不能据此把全部误差归给量化。GPU最大绝对误差约0.000242/0.000244，CoreML约0.003968/0.004907。

预声明随机诊断限值是GPU相对L2<=0.002、CoreML/split<=0.015，六观察均通过；只是此toy分布的检查。它不授予checkpoint、图像、LoRA、特定低幅值分组或应用端质量资格；尤其不把这个aggregate阈值推广到下面的sweep。

## 有限正相关负例：127 → 128.75

maxcode外部数学为 `.5 * 127 * 256 /128 =127`。GPU全部131072个值正确；CoreML全部偏到128.75，最大绝对误差1.75、相对L2约1.37795%。split只在后半行出现此误差。

stored scale舍入的量化A参考与原数学仅差0.0061035%，不足以解释上述差异。源码图是FP16 matmul，但本轮没有内部accumulator或artifact追踪，**不直接断言是ANE硬件、某条指令或某种编译器策略的故障**。

一个独立host-only诊断可作为调查线索：将 `(127/16)^2` 先转FP16，再依次作256次FP16加法，可得到16352，而精确归一化dot为16129；按原restore因子恢复并转FP16后恰为128.75。此例与观测相容，不是原实现用了该顺序的证明；也不据此提出已经验证的fix。

## 行 scale floor：global误差掩盖100%局部丢失

原fused准备kernel把 `max_abs/127` 存为与输入相同的FP16；当stored scale为零时，inverse退回127，而最终恢复仍乘stored scale零。这里是组件边界的可复现精度损失，不是“原输入本来全零”。

| 每行A | 模拟stored FP16 scale | 外部正确值（每列） | GPU actual | CoreML actual |
| --- | --- | --- | --- | --- |
| 2^-8 | 3.07559967e-5 | 127 | 127 | 128.75 |
| 2^-14 | 4.76837158e-7 | 1.984375 | 1.984375 | 1.99609375 |
| 2^-18 | 5.96046448e-8 | 0.1240234375 | 0.1240234375 | 0.12493896484375 |
| 2^-19 | 0 | 0.06201171875 | 0.06201171875 | 0 |
| 2^-20 | 0 | 0.031005859375 | 0.031005859375 | 0 |
| 2^-22 | 0 | 0.00775146484375 | 0.00775146484375 | 0 |
| 真零行 | 约1/127，而非0 | 0 | 0 | 0 |

这些stored scale来自独立CPU源码语义模拟，不是内部GPU scale buffer读回。后面三种非零A的最终非零答案也完全可由FP16表示；GPU路径在本机均正确，不应把CoreML桥的损失解释为所有FP16路径的必然行为。

sweep总相对L2仅约1.37795%，但小幅值三个组各自是100%相对误差。必须看per-group/per-half、expected-nonzero→actual-zero计数，不能让大幅值行稀释掉负例。split的GPU前缀逐值正确、CoreML后缀重复同样错误，是明确的执行/质量边界；不是物理并发或独立queue的证明。

可能的后续研究包括scale存储/范围保护、归一化及累加精度、质量感知fallback；本轮没有改原kernel或验证任何修复。不能简单把scale夹到最小正值便宣称修复：其量化分辨率、舍入、恢复和下游质量仍须匹配。

## CPU oracle警告与完整复核

既有NumPy2.0.2报告使用Accelerate BLAS，CPU FP64 oracle请求单线程，无安装依赖。初次matmul出现divide-by-zero/overflow/invalid浮点状态警告；保存值实际均有限，仍没有直接忽略警告。

后续audit另行捕获24条BLAS警告，并以 `einsum(..., optimize=False, dtype=float64)` 的直接contraction复算四case×两reference的全部1048576个值：无警告、全部有限，且与BLAS重算和原先保存的FP64 reference逐值精确相同。另有24个Python scalar dot核对方向与signed packet。警告根因未证明，不把它归咎于原GPU/CoreML或断言系统BLAS有某个已知bug。

## 证据、运行边界与复现

- 原客户端、CPU合同、主capture、额外audit；设备原算法没有用CPU stand-in替换，NumPy只作离线oracle。
- 原始根为 `outputs/gpu-ane-ccv-precision-round25-20261017`：fixture、12份主actual、3份pilot、12份确认、完整FP64 references、binary、package、源preparation及stdout/stderr/receipt保留；核心JSON含字节数/SHA，ignored raw不是独立分发的必要运行依赖。
- 三次guard分别连续五次0%采样，均owned terminal0；准入是粗粒度总占用检查，不能证明没有瞬时外部工作或精确device归属。后两次并非计时测试。
- 重测必须使用新输出目录，先 `guarded_run.py --output-dir NEW-ADMISSION --timeout 120 -- ...`，不可绕过缺计数拒绝；guard内已使用有界supervisor。固定源/完整command可从JSON与raw receipt追溯，不复用或修改已绑定的旧build做新实验。
- 原桥仍请求CPUAndNeuralEngine，CPU允许；没有placement/ANE指令/overlap、returned backing/zero-copy、内部手工model artifact或物理drain资格。未执行NNC graph dispatcher、s4nnc/Draw Things、真实chooser或checkpoint，未测latency/energy，不拿supervisor elapsed算倍率。

## 固定来源

本轮重读范围见[三文件来源清单](../sources.md#ccv-precision)，GPU/split接口和stream合同沿用[第二十四轮读本](../sources.md#ccv-runtime)。原始实现可追溯到：

- [MIL生成、K-split和CoreML配置](https://github.com/liuliu/ccv/blob/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b/lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_coreml.mm)：37–72、642–682、849–935。
- [FP16行scale、量化及恢复](https://github.com/liuliu/ccv/blob/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b/lib/nnc/mfa/kernels/ANERowwiseTransformKernel.cpp)：1–250、316–355。

结论：格点正控、随机稠密诊断、最大码值相关性和小幅值scale-floor必须分层验证；“status1、全finite、总L2小、split成功”四项都不能单独证明可用质量，更不能作为更快路径的合法分母。
