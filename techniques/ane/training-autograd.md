# ANEForge Autograd：VJP、动态参数、设备优化器与驻留状态

[ANE 目录](README.md) · [训练路线](training.md) · [运行时](private-runtime-stack.md)

固定读本：[autograd.py@896c4943](https://github.com/sbryngelson/ANEForge/blob/896c4943c5ebf811c0e815f8d1381fd1e157fa7d/aneforge/autograd.py)、[开发文档](https://github.com/sbryngelson/ANEForge/blob/896c4943c5ebf811c0e815f8d1381fd1e157fa7d/docs/developer/autograd.md)。阅读参数、VJP、Conv、Trainer/optimizer/resident 路径，范围见 [source-audit](source-audit.md)。未在本轮复跑其训练程序、全部 UnrolledTrainer 或长期收敛。

## 1. 参数是 graph input，而不是 baked constant

`parameter` 存 FP32 master 元数据并创建 trainable input；前向/梯度图读取当前参数。`backward`/`backward_from` 反向遍历图、调用 VJP registry，并在 fan-out 汇合处累加梯度。

`bmm` 的 VJP 形成 `G Bᵀ` 与 `AᵀG`，对 broadcast batch 还要 unbroadcast 归约。普通 baked-weight `matmul` 只返回 activation gradient，不能把两者混成“所有 Linear 权重都可训练”。norm 的 gamma 若是 attrs 值，也不自动变成可求导参数。

## 2. Trainable Conv 如何绕开 native const-weight Conv

固定 native Conv 的 VJP 使用 transposed-conv 求输入梯度；要训练 W，则 `conv_param` 将权重展成 patch matrix，`conv2d` 以 slice/concat 构建 im2col，再与 runtime parameter 做 broadcast bmm。由于这些 primitive 有 VJP，可同时求 dX/dW。

代价是 patch 物化、图规模、slice/concat 和更大临时量。所读实现明确限制 stride=1；不能把“有 trainable Conv”扩展成任意 Conv2D/3D、stride、dilation、padding 都已验证。图尺度和 batch 变化后，编译时间也可能成为主成本。

## 3. 三种 optimizer/状态层级

| 配置 | 运算/状态位置 | 主要成本和风险 |
| --- | --- | --- |
| 默认 host optimizer | 各参数梯度图执行；FP32 host SGD/Adam/moments | 每参数 dispatch、梯度 readback；易独立验梯度，保留 master 精度 |
| `device_optimizer=True` | 每参数另编更新图；host 传/收参数与 FP16 moments | 更新数学在图中不等于免 IO；FP16 moments 与 host FP32 不是同配方 |
| `resident_state=True` | fused multi-output step，output 通过 `share_buffer` alias 回 input | 减状态往返；所有权、FIFO、读旧写新、checkpoint 与 overflow 更难 |

常驻状态依赖真实运行时 alias 与调度语义，不意味着硬件支持任意 in-place update。必须防止 step 尚未完成就启动下一次、checkpoint 读到半更新，以及 session 共享同一参数 backing。

## 4. 精度合同比“loss 下降”更严格

默认 host optimizer 使用 `_check_finite_grads`，失败时不更新 master/moments。**所读 device optimizer 与 resident 路径没有同等的逐步 host finite gate**；resident 同步参数处有 warning，但警告发生时状态可能已污染。因此不能把默认 guard 的资格借给驻留版本。采用前要明确原子 skip、checkpoint 恢复或设备内 finite 机制。

Adam 里放大梯度 s 后，理想实数 `m/sqrt(v)` 的 s 可以抵消，但 epsilon 并不自动抵消：`s*m/(s*sqrt(v)+eps) = m/(sqrt(v)+eps/s)`。不缩放 eps 就不是与原始 Adam 完全相同的更新，尤其小梯度时差异可能明显；二阶矩还放大 s²，FP16 overflow 风险更高。bias correction、epsilon 放置和 FP16 舍入都要与独立 reference 对照。

训练图主动跳过通用 precision heuristic（如结构性减法），不能被理解为它们不需要正确性检查。建议用有限差分/FP64 或 PyTorch autograd 交叉核对每个参数、broadcast、norm、slice/concat，再跑真实 mini-batch 与 holdout。

## 5. 不同于 maderix 的地方

maderix 的经典路径将 fwd/dX 留 ANE、dW/Adam 留 CPU，并重叠执行；ANEForge 能构建 dW 和 optimizer 更新图，进一步探索驻留状态。它扩大的是研究表达能力，不自动更快或更准确。按参数单独编梯度图也可能重算共享前向子图，必须报告真实 step 总工作，而不是把某次 backward dispatch 当总训练时间。

应比较 host-FP32 optimizer、device optimizer、resident 三臂：相同数据/初始化/loss/更新次数，记录每步参数与梯度误差、收敛、编译/加载、峰值和完整 update wall。checkpoint/resume 还应涵盖 m/v/t/loss scale/RNG，而非只保存 W。

本库 [手写反向实验](experiments-20261016.md)没有使用 ANEForge，不能替以上私有训练路径取得实测资格；现有 ANEForge 推理实验也不能代替训练证据。

## 本地学习入口与复查

- [ane/training_math.py](../labs/ane/training_math.py)：原创CPU合同/显式实验客户端，不是上游完整实现。
- [quantization.py](../labs/quantization.py)：原创CPU合同/显式实验客户端，不是上游完整实现。

CPU入口：`python3 -B techniques/labs/run.py ane-training-math`。只运行参考数学/合同，不自动触发Core ML、私有API或网络；实际设备命令与证据边界见[实验报告](experiments-20261016.md)。

SoC/OS/SDK、源码/编译入口、权重/LoRA、输入域/shape/layout/精度或实际消费者改变时重开；不继承旧数值、性能或物理设备资格。
