# 布局与 lowering：从 Linear 到 ANE 友好的 Transformer 图

[ANE 目录](README.md) · [图与 shape 合同](coreml-graphs.md) · [Stable Diffusion](stable-diffusion.md)

## Apple ANE Transformers 的可迁移机制

[官方研究文章](https://machinelearning.apple.com/research/neural-engine-transformers)与固定 [FFN](https://github.com/apple-aiml-research/ml-ane-transformers/blob/590eb35913622cdef217ebc5329047aab389ac98/ane_transformers/reference/ffn.py)、[attention](https://github.com/apple-aiml-research/ml-ane-transformers/blob/590eb35913622cdef217ebc5329047aab389ac98/ane_transformers/reference/multihead_attention.py)、[LayerNorm](https://github.com/apple-aiml-research/ml-ane-transformers/blob/590eb35913622cdef217ebc5329047aab389ac98/ane_transformers/reference/layer_norm.py)是一条完整阅读链。

原始 Transformer 常用 BSC，即 batch、sequence、channel；样本改为 BC1S，把 channel 放到卷积通道维。线性投影 `Y[b,s,o]=Σc X[b,s,c]W[o,c]+bias[o]` 等价地表达为 kernel 1×1 的 Conv2D。attention 按 head 构造较小 einsum，LayerNorm 显式沿 channel 求均值与方差，保持后续算子能消费的布局。

收益假设不是“Conv 数学更少”，而是让编译器选择更合适的映射，减少布局往返/分割，提高片上复用。原项目的性能收益有自己的模型/设备/长度范围，不迁移为 DiT 倍率。

## 数学等价仍需逐项核对

| 变换 | 必须保留的语义 | 主要风险 |
| --- | --- | --- |
| Linear → Conv1×1 | `[out,in]` 权重、bias、token 独立性 | 转置方向、实际 stride、输出恢复 |
| BSC → BC1S | 索引映射、mask、norm axis | 只改 shape、不改物理数据；重复 pack |
| 多头拆分 | 每 head 缩放、softmax 的 key 维、head 拼接顺序 | 把 softmax 移到错误维度 |
| query 分块 | 每块仍看完整合法 K/V | 丢 tail、漏 mask、把 key 分块结果直接相加 |
| K 维分块 | 线性归约结束后才激活/加完整 bias | `SiLU(Σp)` 与 `ΣSiLU(p)` 不同 |
| 图融合 | 残差、adapter、非线性位置不变 | 将历史 ReLU FFN 替代实际 SwiGLU |

LayerNorm 样本的 affine 表达是先加其内部 bias 再乘 weight；迁移标准 `gamma*x+beta` 的参数时须核对重参数化，而不是看到同名 bias 就逐字节搬运。可选 clipping 也会改变函数，不能无条件打开。

## 物理布局不止一种

本库 frozen Conv 常用 `[1,C,1,R]`；二维 matmul 的私有运行时实验实际端口是 `[1,1,R,C]`；另一些 4D matmul 样本是 `[1,1,C,R]`。它们都可能合法，**不共享同一个 pitch/通道解释**。真实共享 surface 读到的 shape/rowBytes/planeBytes 优先于文档里的逻辑标签。

`reshape` 在数学层只重解释索引；`transpose` 是否被编译器消除、是否物化以及 producer/consumer 是否能沿用，必须分别测。不要继承某个项目注释中的“所有 transpose 免费”。

## 实验配方

固定权重与输入，构造 2D matmul、BC1S Conv 和仅布局变换三臂；先以 CPU FP64 检查完整输出，再测 compile、load、first、warm、pack/restore 和首个真实 consumer。至少覆盖一组不整除 token 数和 mask。

本轮 [压缩样本](../labs/ane/compression_survey.py)只用了 Conv/BC1S，并不是 Conv 对 matmul 的速度消融。已有 [真实 FFN tiling 对照](../review/qwen-ffn-tiling-20261015.md)保留了原图/数值/消费者的负结果；不能把本轮 Conv 小图更快解释成“Conv 总胜出”。

迁移顺序建议是 encoder → 有限 shape 的 FFN → 大 attention；长序列 attention 的 O(S²) 中间结果和跨设备 mask 不会因为布局改变而自动消失。

## 本地学习入口与复查

- [heterogeneous.py](../labs/heterogeneous.py)：原创CPU合同/显式实验客户端，不是上游完整实现。
- [quantization.py](../labs/quantization.py)：原创CPU合同/显式实验客户端，不是上游完整实现。

CPU入口：`python3 -B techniques/labs/run.py consumer-strides`。只运行参考数学/合同，不自动触发Core ML、私有API或网络；实际设备命令与证据边界见[实验报告](experiments-20261016.md)。

SoC/OS/SDK、源码/编译入口、权重/LoRA、输入域/shape/layout/精度或实际消费者改变时重开；不继承旧数值、性能或物理设备资格。
