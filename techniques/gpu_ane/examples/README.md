# 离线方法实验：先验证代数、布局和状态，再上设备

<!-- knowledge
id: technique-gpu-ane-examples-readme
type: navigation
-->

[方法导航](../README.md) · [案例](../case-studies/README.md) · [小型代码](../../examples/README.md)

只保留能帮助理解机制的短代码，不附 GPU/ANE 实验启动器、原始样本或验证包。

- 切分代数与 LoRA 顺序：[split_ffn_reference.py](../../examples/split_ffn_reference.py)。
- owner、bank、失败提交与驻留：[contracts.py](../../examples/reference/contracts.py)。
- topology、stride、segments 和 planner：[heterogeneous.py](../../examples/reference/heterogeneous.py)。
- capability/effect 与逐步组合：[policies.py](../../examples/reference/policies.py)。
- 调度与有界 IO：[systems.py](../../examples/reference/systems.py)。

阅读时画出 producer → backing → ready → consumer → last reader 的链，并说明哪些是数学关系、哪些需要具体运行时支持。CPU 示例不证明跨设备同步、真实 backing、物理 overlap 或模型收益。
