# 小型实现参考

<!-- knowledge
id: technique-examples-readme
type: navigation
-->

[总目录](../README.md) · [设计样本](../designs/README.md) · [来源阅读](../sources/README.md)

这些代码帮助理解数学、数据流和状态机，不是 benchmark、实验复算包或生产运行时。无模型权重、原始样本或设备驱动，不会由检索自动执行。

| 想理解什么 | 参考入口 |
| --- | --- |
| FFN 按 channel/row 切分，LoRA 与 bias/residual 顺序 | [split_ffn_reference.py](split_ffn_reference.py) |
| owned graph、双 bank 与不可变 packed store | [contracts.py](reference/contracts.py) |
| capability/effect 组合与调度 | [policies.py](reference/policies.py) |
| cache 身份、residual 更新和 token merge | [caches.py](reference/caches.py) |
| online softmax、norm/RoPE、VAE tile 数学 | [operators.py](reference/operators.py) |
| packed 量化格式与 metadata 解释 | [quantization.py](reference/quantization.py) |
| 调度、驻留与有界 IO 的简化模型 | [systems.py](reference/systems.py) |
| 并行拓扑、stride/segments 和 planner | [heterogeneous.py](reference/heterogeneous.py) |
| 小型 MSL kernel 与 tail/融合思路 | [kernels.metal](metal/kernels.metal) |
| packed affine 的 SG 矩阵计算片段 | [affine_gemm.metal](metal/affine_gemm.metal) |

[共同数学辅助](reference/common.py)只服务这些独立教学样本。Metal 片段不是完整编译/运行工程，affine 片段需要外层提供模板参数和数组 ABI；不要把它作为可直接启动的生产 kernel。

可选择运行一个无设备的代数演示：

```sh
python3 -B techniques/examples/split_ffn_reference.py
```

其输出只说明这个小例子的代数关系，不能推出芯片执行、模型质量或速度。
