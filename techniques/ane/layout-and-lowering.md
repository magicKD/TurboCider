# 布局与 lowering：从 Linear 到 ANE 友好的 Transformer 图

<!-- knowledge
id: technique-ane-layout-and-lowering
type: method
operator: layout
backend: coreml, private-ane, cpu-reference
device: cpu, ane
symptom: wrong-layout, precision-loss
semantics: numerically-equivalent
boundary: producer-consumer
api: reference-only
-->

[ANE 目录](README.md) · [图与 shape 合同](coreml-graphs.md) · [Stable Diffusion](stable-diffusion.md)

## 可以直接借鉴的改写步骤

以行向量线性层`Y=XWᵀ+b`为起点，先固定一组完整参数和输入，再实施下面的单因素改写。这里是迁移配方，不是“改完必快”的承诺。

1. 将`X[B,S,C]`映射为`Xc[B,C,1,S]`，将`W[O,C]`映射为Conv权重`[O,C,1,1]`，bias仍逐输出通道；先证明索引和参数等价。
2. 尽量让相邻投影、逐元素激活和后续投影沿用同一布局；只在真实外部消费者处恢复。不要每层重复BSC↔BC1S，抵消图内收益。
3. attention保留head顺序、scale、softmax的key轴和完整mask。query分块可独立输出；若沿key分块，必须合并softmax归一化统计，不能直接相加各块的已归一化结果。
4. 分别保留“原表达”“仅改布局”“改Conv表达”对照，完整输出通过后，再计入pack→predict→restore→consumer。测试非整除长度、不同norm参数和异常范围。

对应原件：[FFN](https://github.com/apple-aiml-research/ml-ane-transformers/blob/590eb35913622cdef217ebc5329047aab389ac98/ane_transformers/reference/ffn.py)、[多头attention](https://github.com/apple-aiml-research/ml-ane-transformers/blob/590eb35913622cdef217ebc5329047aab389ac98/ane_transformers/reference/multihead_attention.py)。参数迁移和完整消费者对照见[原Transformer研究](apple-transformer-norm-and-position.md)；长上下文online softmax的边界见[Core AI原语](coreai-blocked-attention-and-runtime-gates.md)。

## Apple ANE Transformers 的可迁移机制

[官方研究文章](https://machinelearning.apple.com/research/neural-engine-transformers)与固定 [FFN](https://github.com/apple-aiml-research/ml-ane-transformers/blob/590eb35913622cdef217ebc5329047aab389ac98/ane_transformers/reference/ffn.py)、[attention](https://github.com/apple-aiml-research/ml-ane-transformers/blob/590eb35913622cdef217ebc5329047aab389ac98/ane_transformers/reference/multihead_attention.py)、[LayerNorm](https://github.com/apple-aiml-research/ml-ane-transformers/blob/590eb35913622cdef217ebc5329047aab389ac98/ane_transformers/reference/layer_norm.py)是一条完整阅读链。

原始 Transformer 常用 BSC，即 batch、sequence、channel；样本改为 BC1S，把 channel 放到卷积通道维。线性投影 `Y[b,s,o]=Σc X[b,s,c]W[o,c]+bias[o]` 等价地表达为 kernel 1×1 的 Conv2D。attention 按 head 构造较小 einsum，LayerNorm 显式沿 channel 求均值与方差，保持后续算子能消费的布局。

收益假设不是“Conv 数学更少”，而是让编译器选择更合适的映射，减少布局往返/分割，提高片上复用。原项目的性能收益有自己的模型/设备/长度范围，不迁移为 DiT 倍率。

## 数学等价仍需逐项核对

[RMSNorm完整消费者/VJP对照](rmsnorm-epsilon-floor-and-backward-contract.md)给出另一个lowering边界：native L2+max floor不是加法epsilon RMS的恒等改写。原前向与原VJP还使用不同配方；必须分别验证完整norm→dense→residual和dX，不能仅因方向cos相同、算子更少或输出finite就授予数学等价/训练资格。

| 变换 | 必须保留的语义 | 主要风险 |
| --- | --- | --- |
| Linear → Conv1×1 | `[out,in]` 权重、bias、token 独立性 | 转置方向、实际 stride、输出恢复 |
| BSC → BC1S | 索引映射、mask、norm axis | 只改 shape、不改物理数据；重复 pack |
| 多头拆分 | 每 head 缩放、softmax 的 key 维、head 拼接顺序 | 把 softmax 移到错误维度 |
| query 分块 | 每块仍看完整合法 K/V | 丢 tail、漏 mask、把 key 分块结果直接相加 |
| K 维分块 | 线性归约结束后才激活/加完整 bias | `SiLU(Σp)` 与 `ΣSiLU(p)` 不同 |
| 图融合 | 残差、adapter、非线性位置不变 | 将历史 ReLU FFN 替代实际 SwiGLU |

LayerNorm 样本的 affine 表达是先加其内部 bias 再乘 weight；迁移标准 `gamma*x+beta` 的参数时须核对重参数化，而不是看到同名 bias 就逐字节搬运。可选 clipping 也会改变函数，不能无条件打开。

后续[原实现完整合成Transformer与组件实验](apple-transformer-norm-and-position.md)已执行该hook及LayerNorm/attention原代码：naive bias copy产生finite目标错误、zero gamma产生nan，BSC clip会修改调用方输入；另复现vpos存在却使用kpos的固定函数分支。两种完整2+2层图的72份输出通过独立FP64，但其无clip/非零gamma/default-eps分支不为组件负例授予资格；公开CoreML小图计划首选CPU，不宣称ANE性能。

## 物理布局不止一种

本库 frozen Conv 常用 `[1,C,1,R]`；二维 matmul 的私有运行时实验实际端口是 `[1,1,R,C]`；另一些 4D matmul 样本是 `[1,1,C,R]`。它们都可能合法，**不共享同一个 pitch/通道解释**。真实共享 surface 读到的 shape/rowBytes/planeBytes 优先于文档里的逻辑标签。

`reshape` 在数学层只重解释索引；`transpose` 是否被编译器消除、是否物化以及 producer/consumer 是否能沿用，必须分别测。不要继承某个项目注释中的“所有 transpose 免费”。

## 实验配方

固定权重与输入，构造 2D matmul、BC1S Conv 和仅布局变换三臂；先以 CPU FP64 检查完整输出，再测 compile、load、first、warm、pack/restore 和首个真实 consumer。至少覆盖一组不整除 token 数和 mask。

本轮 压缩样本只用了 Conv/BC1S，并不是 Conv 对 matmul 的速度消融。已有 [真实 FFN tiling 对照](../gpu_ane/case-studies/qwen-ffn-tiling-and-concurrent-cost.md)保留了原图/数值/消费者的负结果；不能把本轮 Conv 小图更快解释成“Conv 总胜出”。

迁移顺序建议是 encoder → 有限 shape 的 FFN → 大 attention；长序列 attention 的 O(S²) 中间结果和跨设备 mask 不会因为布局改变而自动消失。
