# 动态参数训练：把权重变成输入，编译成本与权重流量交换

<!-- knowledge
id: technique-ane-training-dynamic-weights
type: method
operator: gemm
backend: coreml, private-ane, cpu-reference
device: cpu, ane
symptom: compile-bound, precision-loss
semantics: contract-only
boundary: producer-consumer
api: reference-only
-->

[ANE 目录](README.md) · [训练总览](training.md) · [生命周期](lifecycle.md)

固定源码：[maderix/ANE@d91c9845](https://github.com/maderix/ANE/tree/d91c9845c0784dec7753048954fc6d0e8411fe29)，重点 [mil_dynamic.h](https://github.com/maderix/ANE/blob/d91c9845c0784dec7753048954fc6d0e8411fe29/training/training_dynamic/mil_dynamic.h)、[io.h](https://github.com/maderix/ANE/blob/d91c9845c0784dec7753048954fc6d0e8411fe29/training/training_dynamic/io.h)、[train.m](https://github.com/maderix/ANE/blob/d91c9845c0784dec7753048954fc6d0e8411fe29/training/training_dynamic/train.m)。本轮阅读选定范围，未运行其训练脚本/下载数据。

## 1. Packed-input matmul 的真实数据布局

`io_write_dyn` 将一个输入 surface 组织为 `[1,IC,1,SEQ+OC]`。每个 input channel 的前 SEQ 项放 activation、后 OC 项放该输入通道对应的权重，物理行宽是 `SEQ+OC`，不是 `SEQ`。

`gen_dyn_matmul` 在图内分别 slice activation 与 weight，将前者重排为 `[1,1,SEQ,IC]`，后者变为 `[1,1,IC,OC]`，matmul 后恢复 `[1,OC,1,SEQ]`。参数值改变不改变图结构，因此同一个 program 可处理新的 layer/optimizer 版本。

这里的代价是每次 activation 和权重准备、FP32→FP16、surface 写入、图内 slice/transpose。权重“dynamic”不是免费常驻；若每个训练阶段重复 stage 大矩阵，免编译收益也可能被带宽吃掉。单 surface 是此实现适应特定 request 约束的设计，不是所有 public/private 入口永远只能有一个 input。

## 2. 共享 kernel 与逐层状态分离

所读 `train.m` 的 `compile_dynamic_kernels` 生成一组10个 GQA kernel，主路径明确跨全部 layer 共用；每层另配 IOSurfaces/request。包括 attention/FFN forward、FFN backward、Wo transpose 与 Q/KV backward，不能从“10 kernels/layer”误算成28层编译280个不同权重程序。

前向导出 Q/K/V、attention/hidden 等供反向使用；GQA 的 Q_DIM、KV_DIM、head group 不相等，反向还要对广播 head 的梯度归约，不能直接套 MHA 的 shape。

CPU 处理 loss、若干 norm/residual、dW、Adam；dW 的 `cblas_sgemm` 通过 serial GCD queue 与部分 ANE eval 重叠。更新前及部分前向复用点等待 group，避免 dW 未完成就改参数。减少 visible wait 不等于 BLAS 工作消失，也不证明所有权重梯度已放 ANE。

## 3. 为什么值得与常量 Conv 路线比较

若一步编译成本为 C、runtime 传参成本为 P，训练 K 步的简化成本为：常量重编译约 `K*(C+compute)`；动态图约 `C+K*(P+compute_dynamic)`。这是结构比较，不是性能预测；常量 Conv 和 runtime matmul 的 compute 也可能不同。

用相同 shape、权重与更新序列验证三种候选：常量图重编译、动态参数、冻结 base+LoRA。分别测总训练时间、每个 optimizer update、host copy、编译/加载、峰值与收敛。不能只报告不含 C 的热态算子，或把梯度积累窗口内“未更新权重”当动态参数胜利。

## 4. ANEgpt 的扩展与不能继承的结论

[ANEgpt](https://github.com/vipuldivyanshu92/ANEgpt/tree/0f60ed113306221944e9091b1dc1e160c8698e7d)在同族训练结构上增加 classifier、softmax、RMSNorm backward 的 offload，并保留 CPU dW/Adam。README 有部分继承自上游的 benchmark，自己也注明未独立验证；不将这些数字作为本轮实测。

“Adam 必须原地改权重，所以不可能在 ANE”是某个常量图设计的限制，不是数学上的不可能。ANEForge 已用更新后的参数作为输出并回馈输入，甚至 alias resident state；见 [autograd](training-autograd.md)。同理，某个大输入通道 Conv 被拒绝，也不能推出所有 ANE matmul 或分块都不行。

## 5. 最小可复现实验

training_probe.py不依赖上述 checkout，用公共 Core ML 同 shape 的 matmul输入复用 forward/dX/dW，保留 host SGD 与 FP64 oracle。它没有采用 packed-single-input 的 private ABI，且计划 CPU，因此仅验证动态参数训练的代数/更新机制，不能替此源码的私有图取得设备资格。

采用前还需尾部 pitch、极值、underflow/overflow、异步 dW 的 owner、checkpoint/resume 和 generation 故障对照。约119次 compile 的历史观察不应写进产品定律；动态图的主要价值是减少必要工作，而非规避安全保护。
