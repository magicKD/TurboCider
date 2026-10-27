# FoCa / Dynamic Mode Decomposition：替换 residual 的预测值，不替换 gate

<!-- knowledge
id: technique-algorithms-methods-forecast-calibrators
type: method
stage: dit
operator: sampling, gemm
backend: cpu-reference, mlx, metal
device: cpu, gpu
symptom: stale-cache, precision-loss
semantics: approximate
representation: bf16, fp32
boundary: producer-consumer, request
api: reference-only
-->

[算法入口](../README.md) · [DBCache边界](block-caches.md) · [状态反例](../case-studies/forecast-state-audit.md) · [配套实验](../../examples/README.md)

来源：Cache-DiT固定校正器 [S33](../sources.md#s33)，FoCa作者论文。主要阅读对象是**第三方集成版本**，不是原FoCa代码；DMD此处指Dynamic Mode Decomposition，不能与Distribution Matching Distillation混同。

## 校正器处在什么位置

DBCache命中时，本步front依然真实计算，middle残差由calibrator预测，然后加到当前front输出并接后段。gate负责是否省算，calibrator负责被省段的近似值，两者独立。一个forecaster支持更长外推，不自动授权gate提高阈值或无限连跳。

## FoCa：递归局部状态与full anchor

Cache-DiT引用的FoCa工作把feature沿步演化看作ODE，组合prediction与calibration以抑制长距离预测漂移；本轮核心证据来自该集成代码，未逐式审计原论文，任务指标接近也不等于数值无损。

Cache-DiT的`FoCaState`保存最近两份rolling值及两份full anchor。所读实现用anchor差分与rolling差分的平均，从最近full anchor外推；每次`approximate()`又将预测写入rolling窗口。超过3步时将有效horizon限制并向anchor混合。这一硬编码阻尼也是策略参数，即使类注释称“无超参数”。

注释提到BDF2/Heun，但不能据此宣称代码调用了通用隐式ODE求解器。要核对实际差分、Δt、更新时序和何时读state。所读实现未用保存的`prev_full_step`归一化full差分；在full间隔变化时，需要额外合同。[状态探针](../case-studies/forecast-state-audit.md)保留确定性反例，而不改写上游。

## DMD：学习短窗线性传播算子的模态

将同间隔full snapshots按列组织成X和X'，用瘦SVD截断/正则化构造低维传播算子，再求特征值λ、模态Φ和幅值b，预测 `Y_hat(k)=Re(Φ(λ^k⊙b))`。这里的低秩是**feature演化算子**，不是SVDQuant的权重outlier分支。

该实现按batch item独立拟合；新full snapshot使fit失效，skip期间只改变k，不重复SVD/eig。`_uniform_tail`只取等间隔历史后缀，并用horizon/spacing缩放；少于4份snapshots用Taylor回退，退化fit或输出cast后非finite则reuse。动态gate使间隔变化，可能长期只有短tail，因此“启用DMD”不保证实际做过DMD预测。

## Apple 最大障碍是实际数值/设备路径

固定快照中DMD堆叠H为FP64，eig/lstsq用complex128。不能把PyTorch MPS对普通BF16/FP32算子的支持推导成这条路径可用。内部catch异常后静默reuse，可能令“DMD开关开启”看似成功却没有DMD工作。

可研究GPU保留full history并先构造小Gram/投影，CPU做小型double/complex求解，最后GPU线性组合；但这属于新的实现路线，不能把省略fit/传输的数字当完整收益。大feature的多份history与Φ也会争统一内存带宽。

## 比较与必要反证

同一full-step表：replay、Taylor1、FoCa-adapter、DMD分别运行，再统计 actual fit成功、fallback原因、fit/evaluate时间、历史与modal bytes。先用恒定、线性、指数、振荡和非均匀refresh建立oracle，再进入真实trajectory。

CPU已验证该DMD小型指数路径工作，但短历史linear fallback、FoCa gapped anchors等未满足对应小型合同，见[精确范围与复现](../case-studies/forecast-state-audit.md)。它们不是“所有预测cache都不可靠”的结论；说明forecast核心和fallback都需要审计。

第三轮[真实forward/manager接缝](../case-studies/cache-consumer-callchain.md)确认普通pattern在hit只读一次Bn；额外step5读取是显式诊断臂，不是默认行为。实际gate的gapped full anchors仍可产生FoCa小型线性偏差；命名state的lazy创建使clock比logical step少1，DMD在该链中始终Taylor fallback。保留与孤立state探针不同的调用前提，不宣称真实模型已发生同样回归。

论文：[FoCa 2508.16211](https://arxiv.org/abs/2508.16211)。第三方adapter参数/状态与原论文方法不是自动等价。
