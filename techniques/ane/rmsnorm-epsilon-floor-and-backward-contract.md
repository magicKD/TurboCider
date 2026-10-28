# RMSNorm的epsilon配方与VJP：完整norm→dense→residual消费者对照

<!-- knowledge
id: technique-ane-rmsnorm-epsilon-floor-and-backward-contract
type: method
operator: norm, gemm
backend: private-ane, cpu-reference
device: cpu, ane
symptom: precision-loss, measurement, lifetime
semantics: contract-only
representation: fp16, fp32
boundary: producer-consumer, weight-runtime
api: private, reference-only
-->

[ANE目录](README.md) · [10文件来源](sources.md#norm-vjp-sources-20261017) · 全部输入/输出、原MIL与逐项审计 · [验证记录（过程原件已移出）](../MAINTENANCE.md) · [前批独立数值边界](division-variance-and-observer-boundaries.md)

2026-10-17，M4 Max / macOS26.6.2、NumPy2.0.2，固定ANEForge `896c4943c5ebf811c0e815f8d1381fd1e157fa7d`与既有精确hash shim。四个新进程完成704次显式求值、880份完整输出，**全部finite**；但48份RMS dX不符合所发射前向的连续数学导数，却符合原VJP采用的另一套加法epsilon公式。

这不是从独立除法反例推断整个模型。实际执行的是原norm frontend、原VJP registry、原baked matmul反向及完整非对称投影/残差消费者；没有修改上游，也没有optimizer、真实任务或训练收敛/速度结论。LayerNorm作为公式一致的对照，同时保留其近常量高偏置输入的数值失败。

## 可复用的方法：norm lowering 同时验证前向与导数

1. 先写清 epsilon 位于加法、sqrt 内外还是 max floor；相同名字不构成数学等价。
2. 比较完整 norm→dense→residual，而不只看方向 cosine；用可控 adjoint 独立验收 dX 与 affine 参数。
3. 原 VJP、替代表达的导数和实际执行结果分列；前向成功不能为另一配方的反向自动授予资格。

适用于 norm 融合、动态训练图与数值排错；下文 floor 反例不是通用 epsilon 推荐。

## 1. 优化表达减少了什么，又改变了什么

[原RMS lowering](https://github.com/sbryngelson/ANEForge/blob/896c4943c5ebf811c0e815f8d1381fd1e157fa7d/aneforge/_compile.py#L714)使用native `reduce_l2_norm`，把√D折进gamma，再取分母下限：

```text
GW = half(gamma * sqrt(D))
floor = half(sqrt(D * eps))
Y = X / max(reduce_l2_norm(X), floor) * GW
```

忽略常量和算术舍入，在每行D=32、q=mean(X²)的实数模型里，它是：

```text
floor-RMS:     Y = gamma * X / max(sqrt(q), sqrt(eps))
additive-RMS:  Y = gamma * X / sqrt(q + eps)
```

复用native L2、常量fold和安全分母是可学习的表达技巧，但**max下限不等于往mean-square加epsilon**。只有q远大于epsilon等条件下它们才近似；若q随输入变化，也不能用一份固定gamma普遍补偿差别。本批没有测这条lowering比其他实现快多少。

加法RMS对照参考了固定[PyTorch2.5.1定义](https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/torch/nn/modules/normalization.py#L321)及[对应native实现](https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/aten/src/ATen/native/layer_norm.cpp#L266)。GitHub tag已解析成确切commit，三份原文件/许可只读留在ignored来源包，**没有import或执行Torch**。本批显式用eps=1e-5或1/64，不把PyTorch的dtype-dependent默认epsilon当成同默认/同运行时对照。

## 2. 原VJP对应的是另一套前向

设norm上游梯度为g，h=gamma⊙g。对所发射floor函数，在远离q=eps的点：

```text
q < eps:  dX = h / sqrt(eps)
q > eps:  dX = (h − X*mean(h*X)/q) / sqrt(q)
```

而原[_vjp_rms_norm](https://github.com/sbryngelson/ANEForge/blob/896c4943c5ebf811c0e815f8d1381fd1e157fa7d/aneforge/autograd.py#L238)构建`rsqrt(mean(X²)+eps)`，对应：

```text
dX_additive = (h − X*mean(h*X)/(q+eps)) / sqrt(q+eps)
```

这两者在源码的连续数学层面就不同，不需要猜测内部FP16指令才能指出。若要求“所发射前向的精确数学反向”，原规则在本批输入上不满足；若有意采用surrogate/近似梯度，需要单独说明合同和任务证据。不能据此宣布原MNIST/CIFAR或所有训练一定失败，本批也没有运行这些上游训练任务或检查其他commit是否改变了实现。

原LayerNorm则前向采用population variance加epsilon、后向采用同一公式；它的有限数值失败和上述**RMS配方不一致**是两类问题，不能混为一个硬件故障。

## 3. 完整消费者、可控adjoint与真实执行边界

```text
Y = original_norm(X; gamma, beta, eps)
Z = Y P + R
L = sum(U * Z)
original dX = backward_from(U, Z, [X])
```

X/Y/Z/U/R都是32×32。P是全稠密±1/8的Hadamard列置换，**非对称、非identity**，满足PPᵀ=.5I；因此可构造U使`U Pᵀ = h/gamma`，把norm的径向/切向方向精确传到完整消费者。gamma为非零±.5/1/2，LayerNorm的beta和残差R均非零；P/gamma/beta都是自有固定合成参数，不是训练权重分布或真实checkpoint。

每seed有两种epsilon、五类输入：zero、below-radial、above-radial、normal-tangent、offset。径向和normal控制各在相对scale .875/1/1.125喂入同一owner，保持U/R不变；zero/offset仅scale1，每配置共11代。offset为8附近带小差异的可表示FP16输入；normal-tangent对RMS径向正交，LN仍用其自身公式判断。两seed改变符号/参数/列置换，正反配置/角色/输入顺序另起进程复查，不是独立随机总体样本。

每配置分别创建原norm单输出、原Z单输出、额外导出Y/Z的multi-output及原dX图，共64个显式program artifact。**只验dX**：baked投影的activation VJP和norm的原规则实际执行，gamma在原VJP中作为value-input重注入；没有验证trainable-affine分支、dGamma/dBeta/dP、Conv、GroupNorm、channel norm、UnrolledTrainer或optimizer步骤。

## 4. 前向正确执行source，不代表后向是它的导数

下表每项44份完整数组，含两seed×正反序×11输入代际；拒绝指本文固定数值合同，不是进程崩溃：

| 配置 | Z对原前向源式拒绝 | Z对加法对照拒绝 | dX对原前向导数拒绝 | dX对原VJP公式拒绝 |
| --- | ---: | ---: | ---: | ---: |
| RMS，eps=1/64 | 0 | 32 | 24 | 0 |
| RMS，默认1e-5 | 0 | 24 | 24 | 0 |
| LayerNorm，eps=1/64 | 4 | 4 | 0 | 0 |
| LayerNorm，默认1e-5 | 4 | 4 | 4 | 4 |

352份RMS前向输出（包括单Y、单Z及观察图Y/Z）全部通过原源式和实际常量的实数参考；88份原RMS dX全部通过加法VJP参考，但48份未通过floor前向导数。失败不只出现在刻意较大的epsilon，默认1e-5也保留24份。

LayerNorm的forward拒绝来自offset控制，默认epsilon还有4份对应dX拒绝；数值敏感性并没有因“前后公式一致”而消失。两种norm的zero与正常范围控制另存，不拿offset失败宣称所有输入或所有norm都坏。部分非径向RMS输出对加法目标仅有小相对差异，也可能因maxabs预算拒绝，不能把表内所有失败都重新称作同一个径向反例。

源floor与加法RMS的某些输出仍完全共线，cosine可以为1却有超过10%的幅度差。因此单独报cos≈1或loss下降不能覆盖此配方问题。本批不放宽旧Adam/独立除法门槛，也不拿更贴近某实现的参考替换原目标去消除拒绝。

## 5. 大步径向诊断：实际forward不变，原VJP却预测非零

取h=X_base，eps=1/64、X元素为±.25，整个scale区间都在floor以上。floor归一化对正的整体缩放不变，故真实连续数学径向导数为0；加法epsilon版本仍有非零径向导数。seed17首个进程的实际完整Z结果：

| 输入/配置 | 实际Z损失有界secant | 原实际dX·X_base | floor连续参考方向导数 |
| --- | ---: | ---: | ---: |
| dyadic，below-radial | 32 | 22.890625 | 32 |
| dyadic，above-radial | 0 | 45.71875 | 0 |
| default，below-radial | .3085419014 | .26904296875 | .3088161778 |
| default，above-radial | 0 | 1.2294921875 | 0 |

所有8个RMS above-radial组的实际有界secant都为0，原VJP方向预测都大于1；它们分别保留完整三代Z和dX，不是只采一个norm值。公式参考还对64条coordinate/reference记录做continuous central difference，最大绝对误差约5.15e-9，并核对完整投影损失与独立闭式contraction一致。

这些损失由保存的完整Z与固定U在host用`math.fsum`归约，没有另执行compiled loss graph，也不计为额外ANE调用。原反向入口是显式seed的`backward_from`，不是靠一个可能不同归一化系数的scalar loss猜测上游梯度。

**两类差分不能混用**：实际secant使用±12.5%有限缩放；reference FD在固定wire点的连续数学函数上扰动，不重新round端口。都不冒充量化设备函数的无穷小导数、物理梯度trace或任务训练质量。q=eps的不可微kink没有纳入FD点；LN source/standard有相同公式，重复的reference记录不算独立模型。

## 6. 观察接口、打包与缓存身份的独立控制

本批352对单输出/额外观察接口数组逐字节相同，440对正反序新进程输出也逐字节相同；不能把RMS差别归给这次新增tap或仅一次噪声。这个正控制不推翻[前批公开inverse接口变化](division-variance-and-observer-boundaries.md)，也不是任意图/shape的tap无扰动保证。

标准库独立解析64份BLOBFILE容器、120份完整FP16常量payload，核对gamma fold、beta、Pᵀ及原反向使用的P；不是仅比shape或信任某个typed banner。每run46个Python加2个native来源文件与固定Git逐一匹配，原shim hash仍是`1822a6eda222cd48efd6c4fcf9cf127514393c819e300d58626ab9261e6d33b1`，没有新建/安装库或修改上游。

针对前批cache退出后不可重核的限制，本批采用**新协议**：owner尚未调用release时，对明确owned、大小受限的cache文件读两次相同字节，再复制到独立immutable archive。128条文件副本（含H16C.e5及model.anehash）合计181,888个每run去重字节，在进程结束后仍全部可核对。它们不是128个硬件kernel或OS-loaded-image attestation；当前源cache是否继续存在不作为归档身份前提，也没有重建旧72条cache来补造历史。

## 7. 复现与未完成范围

发布880份实际输出（901,120个元素），255个唯一matrix/vector payload只做lossless去重；所有fail与finite字段保留。forward gate为maxabs≤.02且relativeL2≤.02，dX逐元素门槛为`.002+.02*abs(reference)`，不以大X/大参数的相对误差掩盖小梯度，也不把应为零的源导数放过明显非零值。

四套参考分别是请求参数的source函数及其导数、实际打包常量的实数函数及其导数、请求epsilon的加法对照、原VJP使用的typed-epsilon方程。`vjp`标签下的Y/Z只是该方程对应的假想前向目标，不是额外执行/观察到的原前向；全部都不是逐op FP16/硬件仿真。

本地实现：显式原项目collector、完整数学/输入/codec/导数核查、、portable tests。默认不导入框架或调用设备：

此批没有speedup、优化GPU基线、能耗、真实模型/数据/收敛或物理ANE放置证据；mask4仍只是请求意图。若接受floor作为推理近似，应给出输入域/任务误差预算；若训练要用其精确导数，须另行设计、验证并回到完整训练轨迹。源码/OS/SDK/shim、epsilon/shape/布局、affine trainability、输入分布、consumer或输出接口变化均重开。整体研究继续。
