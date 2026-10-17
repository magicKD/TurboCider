# 子图融合与提交复用：分别减少数据流和host开销

<!-- knowledge
id: technique-gpu-ane-fusion-and-replay
type: method
stage: pipeline
operator: scheduling
backend: metal, mlx
device: gpu, ane
symptom: launch-bound, lifetime
semantics: contract-only
boundary: producer-consumer, gpu-ane
api: public, private
-->

[方法导航](README.md) · [GPU attention](gpu-attention-runtime.md) · [测量与回退](validation-and-fallback.md)

## 解决什么问题

短算子链可能同时受中间读写、前端构图和命令编码限制。融合减少中间量与dispatch；编译减少重复图处理；replay复用已编码的提交结构。三者应分别归因，不能把某个弱eager分母上的倍率称为模型或异构收益。

## 实现顺序

1. 先保留完整数学消费者，只在确实不再使用时移除中间materialize。
2. 对SwiGLU，将gate/up沿输出轴配对，保留正确row stride；SiLU、down bias和residual各在正确位置执行。
3. 将稳定纯表达式交给编译器；记录shape、Python scalar、闭包常量与可变状态的specialization规则。
4. 最后研究JIT/ICB重放，明确输入地址更新、资源owner、同步完成和旧输出复用规则。

融合不是单纯“少一个kernel”：较大kernel可能增加寄存器和临时空间；更少command也可能被host开销或执行效率抵消。

## 可借鉴实现与反例

| 入口 | 学习重点 | 已有证据边界 |
| --- | --- | --- |
| [tinygrad固定源码](https://github.com/tinygrad/tinygrad/tree/3d057f05df532deca7661bfb2400c6564f96a04a) · [HCQ/JIT](case-studies/tinygrad-metal-hcq-and-replay.md) | 调度、参数patch、owner与重放 | 固定原Metal小组件，不是完整模型 |
| [完整paired SwiGLU](case-studies/tinygrad-strong-baselines-and-paired-ffn.md) | 强GPU分母、gate/up布局、分支质量 | 4/3/2-command未形成通用速度排序 |
| [MLX编译与FFN对照](case-studies/mlx-compile-and-cross-runtime-ffn.md) | scalar/cache、normal completion、跨进程计时 | 跨runtime窗口不是同窗口配对试验 |
| [MetalPetal融合](case-studies/metalpetal-render-graphs-and-pixelbuffer-handoff.md) | 保留中间裁剪、UNorm量化语义 | 固定图中融合开关最多差64码；不代表所有融合错误 |

## 哪些边界必须保留

实数代数等价不保证有限精度等价。中间FP16/UNorm写回、clamp、scale、bias、alpha、transfer和LoRA顺序都属于原函数。不能用F16正控的一致性证明任意BGRA8图可合并。

编译cache命中也不意味着新权重/新状态生效。闭包里捕获已eval数组与传入运行时参数不同；旧返回对象可能复用存储。重放前后应覆盖A/B/A、scalar改变、重新分配、state恢复和取消后的owner。

## 最小验收

比较同一数学与输入的成熟GPU实现，计同步完成后的完整调用；分开cold compile、warm前端、编码、设备完成和回读。除完整输出外检查去residual分支，避免大残差掩盖坏FFN。

动手从[离线实验](examples/README.md)的`segmented-ffn`开始；通用实现继续阅读[Metal融合](../metal/fusion-and-layout.md)与[command graph](../metal/command-graphs.md)。只在目标消费者实测有收益后采用。
