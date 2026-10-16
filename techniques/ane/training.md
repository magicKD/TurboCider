# ANE 训练路线：反向传播是计算图，参数更新才是关键接缝

[ANE 目录](README.md) · [动态权重](training-dynamic-weights.md) · [delta/LoRA](training-delta-and-lora.md) · [autograd](training-autograd.md)

## 先回答“能不能训练”

矩阵乘、逐元素运算和归约可以组成前向与反向；“ANE 对外主要是推理加速器”并不意味着硬件不能求导。研究实现将梯度公式编成普通推理计算图，host 控制训练循环。这与公共训练 API 自动选择 ANE 是不同命题。

Apple SDK26.5 明确限制 MLCompute `.ane` 仅推理；Core ML 的 on-device update 也不等于通用 LLM ANE trainer。不能依据 `.all`、一个 training 类或模型能下降 loss 声称物理 ANE 训练。[公共接口边界](public-coreml-deployment.md)与 [MLCompute 官方设备文档](https://developer.apple.com/documentation/mlcompute/mlcdevicetype/ane)提供依据。

## 主要技术方案对照

| 路线 | 参数更新后怎么办 | 前向/反向/优化器 | 优势 | 成本与风险 |
| --- | --- | --- | --- | --- |
| 常量图 + 重编译 | 每次或每积累窗口重新 compile/load | ANE 算子 + CPU dW/Adam 常见 | 复用高效常量 Conv | compile、程序资源、磁盘/cache；不是实用全参数训练默认 |
| maderix / ANEgpt 动态参数 | 参数随 activation 输入，graph 不变 | ANE fwd/dX，CPU dW/Adam；部分扩展继续 offload | 摊销编译、共享同形 kernel | 每步 staging/传参、dW/optimizer 状态、梯度精度 |
| Orion delta reload | 固定编译结构，unload→改权重→reload | ANE 子图 + CPU dW/Adam | 保留常量图结构、降低重编译税 | 私有产物格式假设、重载与文件 IO、stale weight |
| 冻结 base + runtime LoRA | 更新/切换低秩 A/B，无需重编译 base | base Conv + adapter matmul；训练分工另定 | 更新数据少、适合微调 | 反向/adapter rank、scale、量化语义与生命周期 |
| ANEForge graph autograd | trainable inputs；VJP 编成图 | 可 ANE fwd/dX/dW；host 或 device optimizer | 自动求导、trainable Conv 分解、resident state | 图膨胀、FP16 moments、alias、overflow、长期收敛 |

这些是方案对照，不是同一模型/机器的速度排名。maderix 当前动态训练代码共用一组 kernel；Orion 的常量每层程序与 ANEgpt 的具体版本有不同数量，不能把“每层 dispatch 数”当“总 compile 数”。

## 数学最小核心

对行向量约定 `Y=XW`，给定上游梯度 `G=dL/dY`：

`dX=GWᵀ`，`dW=XᵀG`。前向、dX、dW 都能用同类矩阵算子，但 shape、转置和归约轴不同；native 常量 Conv 可只支持输入梯度，而动态 bmm 才让 W 成为可求导输入。

反向至少还要保存或重算 activation、norm 统计、attention probabilities、gate/up 等。全量 forward taps 省重算却增加输出/内存；gradient checkpointing 反过来用重算换状态。CPU dW 与 ANE dX 可以重叠，但 optimizer 必须等待相应梯度完成，不能用前一步参数或提前复用 buffer。

## 为什么 FP16 是训练中的实质限制

小梯度可能 underflow，loss scaling 可扩大反向值，最终再 unscale；过大 scale 又可能让激活、dW 或 Adam 二阶矩 overflow。FP32 master、FP32 moments、FP16 device state 是不同算法数值配方。只看 loss 下降不证明梯度正确、模型质量好或长时稳定。

训练报告应同时给 gradient check（解析/有限差分）、finite/zero-gradient 比例、loss normalization、scale、clip、optimizer epsilon、master/state dtype、更新次数、checkpoint/resume 等价性与独立验证集。step/s 还必须固定 sequence、microbatch、accumulation 和“step 是否真有 optimizer update”。

## 上游性能应怎样读

[maderix README](https://github.com/maderix/ANE/blob/d91c9845c0784dec7753048954fc6d0e8411fe29/README.md)报告 Stories110M/Qwen3-0.6B 的动态训练；[ANEgpt README](https://github.com/vipuldivyanshu92/ANEgpt/blob/0f60ed113306221944e9091b1dc1e160c8698e7d/README.md)列 classifier/softmax offload 与若干继承的上游数字；[Orion README](https://github.com/mechramc/Orion/blob/4dcc1c2a626c9ee1f35576bd0b290d489db17d08/README.md)将 delta 的重载收益与总训练收益分列。均是作者报告，本轮未复跑整模型，不相互相除，也不复制“纯 ANE”标题覆盖正文 CPU 分工。

另外 [iPhone 训练项目](https://github.com/slavko-at-klincov-it/ANE-Training-iPhone/blob/2564a23d5e8f0f37d55b72c0367e9b626db57846/README.md)提供后台/温控/恢复的研究入口；只核对 README，不接受其“MLCompute 曾有公开 ANE 训练”“72 kernel 同时运行”等陈述为事实。kernel 数量、loaded、提交与物理并行是四回事。

## 本轮实际验证了什么

[原创 training_probe](../labs/ane/training_probe.py)将 forward/dX/dW 复用一个 runtime-input matmul，host FP32 SGD，12次更新及37次完整输出均以独立 FP64 数学检查；另有纯 CPU 对照。**计算计划首选 CPU**，所以它验证不重编译的手写反向、参数 generation 和 loss scaling 学习机制，不宣称取得 ANE 训练速度或通用 Core ML trainer。见 [全部结果](experiments-20261016.md)。

下一步若评估私有训练，应先小 MLP/LoRA，建立相同 MLX GPU trainer 和相同 dtype/optimizer 基线，再逐项考察 dynamic weights、device optimizer、resident state。整模型训练、低功耗或长期收敛不属于本轮已实测范围。

## 本地学习入口与复查

- [ane/training_math.py](../labs/ane/training_math.py)：原创CPU合同/显式实验客户端，不是上游完整实现。
- [ane/training_probe.py](../labs/ane/training_probe.py)：原创CPU合同/显式实验客户端，不是上游完整实现。

CPU入口：`python3 -B techniques/labs/run.py ane-training-math`。只运行参考数学/合同，不自动触发Core ML、私有API或网络；实际设备命令与证据边界见[实验报告](experiments-20261016.md)。

SoC/OS/SDK、源码/编译入口、权重/LoRA、输入域/shape/layout/精度或实际消费者改变时重开；不继承旧数值、性能或物理设备资格。
