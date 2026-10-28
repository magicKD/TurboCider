# GPU 技巧的离线学习入口

<!-- knowledge
id: technique-metal-examples-readme
type: navigation
-->

[方法导航](../README.md) · [实现案例](../case-studies/README.md) · [小型代码总览](../../examples/README.md)

这些是阅读练习，不是 selector/runner、设备实验或复算入口。先理解机制，再将它迁移到自己的实现。

| 目标 | 阅读代码 | 思考的问题 |
| --- | --- | --- |
| packed 格式与 typed metadata | [quantization.py](../../examples/reference/quantization.py) | code order、signedness、group/scale 怎样影响消费者？ |
| online softmax 与融合 | [operators.py](../../examples/reference/operators.py) | 分块合并如何保留归一化，尾部和空行怎样处理？ |
| cache 身份与失效 | [caches.py](../../examples/reference/caches.py) | 哪些输入变化必须失效，失败何时提交状态？ |
| graph owner、双 bank 和完成 | [contracts.py](../../examples/reference/contracts.py) | 逻辑取消与最后读者完成为什么不同？ |
| tiling 和 epilogue | [kernels.metal](../../examples/metal/kernels.metal) | 少了哪次全局读写，增加多少局部状态？ |

框架里的实际选核、精度和成本从[固定源码](../sources.md)和案例阅读。教学代码不是成熟框架或生产 kernel 的性能分母。
