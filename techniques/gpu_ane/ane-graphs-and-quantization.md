# ANE：图表达、尺寸敏感性与量化的真实含义

<!-- knowledge
id: technique-gpu-ane-ane-graphs-and-quantization
type: method
stage: ffn
operator: swiglu, conv, layout
backend: coreml
device: ane, gpu
symptom: graph-interruption, wrong-layout, compile-bound
semantics: numerically-equivalent, approximate
representation: fp16, w8a8
boundary: gpu-ane, weight-runtime
api: public
-->

[专题导航](README.md) · [私有路径](ane-private-and-search.md) · [ccv](case-studies/ccv-rowwise-iosurface.md) · [实验](../MAINTENANCE.md)

一手入口：[Apple ANE Transformers 文章](https://machinelearning.apple.com/research/neural-engine-transformers)、[其固定 FFN 实现](https://github.com/apple-aiml-research/ml-ane-transformers/blob/590eb35913622cdef217ebc5329047aab389ac98/ane_transformers/reference/ffn.py)、[Core ML 量化性能指南](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html)。图执行行为随硬件、OS、deployment target 和 converter 变化，不把历史经验当永久硬件限制。

## 先让图适合后端，再讨论峰值

Apple 的 ANE Transformers 将 BSC 表达改为 BC1S `[batch,channels,1,sequence]`，FFN 使用 1×1 Conv2D，尽量沿一条布局执行。其价值在于避免昂贵 transpose/lowering 和非 ANE 小岛，不只是“Linear 改名 Conv”。

读 `LayerNormANE` 时还需核对 affine 参数语义：其实现先加 bias 再乘 weight，不能直接把普通 LayerNorm 的 post-scale bias 参数照搬。模型权重迁移与数值测试是图改写的一部分。BC1S 不是所有 ANE 程序的唯一合法布局；本库真实二维 private matmul port 就不同，见 [共享边界](memory-and-handoff.md)。

Core ML graph 可以混合设备。`CPU_AND_NE` 是允许集合，不是强制每个算子 ANE-only；计划 preferred device 是预估放置，不是实际 counter。转换成功、MIL op 存在、模型能预测、ANE 物理执行是不同结论。

## 为什么有些 size 更快

外部 I/O padding、channel/vector 对齐、编译器 tile/lowering、片上工作集、dispatch 固定成本与 fusion 可以产生不连续行为。一个维度多一点，可能是多做 padding，也可能触发更好的核或不同设备；不能只从耗时拐点宣布某个 SRAM 容量。

| 参数 | 应扫的候选 | 不能忽略的代价 |
| --- | --- | --- |
| rows / bucket | 实际 rows、向上 bucket、多个固定 chunk | padded rows、预测调用次数、tail |
| K/N graph tile | monolithic、图内分块、host 多图 | partial scratch、归约、compile/load |
| frozen/runtime weights | bake 常量、IOSurface 输入 | 每层程序驻留 vs 每层 staging |
| layout/operator | Linear/matmul/Conv、transpose 位置 | 完整 producer→consumer conversion |
| 融合边界 | 单 projection、完整 FFN、更多 block | 编译合法性、内部流量、dispatch 节省 |

例如 ccv 的 `select_matmul_k_split` 是尺寸 cliff 的工程应对；vpipe 的 host chunk 和 graph tile 又是不同层次，见各自案例。阈值不跨芯片继承。

## W8、W8A8、LUT4 不是同一件事

| 表示 | 减少什么 | 可能的计算路线 | 必须验证 |
| --- | --- | --- | --- |
| weight-only W8 / W4 | 序列化与权重流量 | load/图内 decode 后浮点，也可能融合 | 文件、resident、时延分别记录 |
| LUT / palettization | 索引+码本的权重字节 | 查表/融合解码再计算 | per-channel/per-block、quality 与解码成本 |
| W8A8 | 激活+权重表达/流量 | 合法融合后的低精度执行 | calibration、Q/DQ 开销、实际放置 |
| Q/DQ 或 fake quant | 限制数值格点 | 不一定整数 MAC | op 存在不等于硬件指令证明 |

Core ML 官方页明确区分 CPU/GPU/ANE 的建议：ANE 的 activation quantization 对 A17 Pro/M4 或更新硬件更值得测试，并指出卷积输入通道/激活更大时收益可能更明显；weight-only 的 memory footprint/latency 又随芯片变化。不能把“INT8 算得更快”缩成没有条件的规格。

### 关于“ANE 的 INT8 是否真的整数算术”

第一方逆向作者 [maderix 的 M4 文章](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)在其私有 probe/反汇编路径中报告低比特权重被展开为 FP16，推测收益来自带宽；官方压缩指南则报告特定新硬件的 W8A8 性能收益。二者测量入口、图、编译器、运算类别和“收益/物理指令”的问题并不相同。**本专题不据任一来源推出“所有 ANE 一律 FP16”或“所有 INT8 图必用整数 MAC”。** 本机实验只测 wall time/误差/plan，不声称已观察内部 MAC 类型。

## 校准与精度岛

激活量化必须覆盖真实分布：层、timestep、prompt/seed、参考图、LoRA、视频长度。独立 hold-out 检查 clipping、outlier、zero/near-zero 与 FP16 range。SmoothQuant/旋转可重新分配 outlier，但后续舍入仍是近似，增加的 GPU/CPU 路径和 scale 恢复也要计时。

封闭 FFN 内部 FP16→返回 BF16 residual 是一种精度岛设计，不代表自动无损。down/LoRA/bias 的高精度恢复、非有限拒绝和完整质量应保留；关键 state 长期累积的误差更需独立预算。

## 本机观察与可运行入口

[本轮压缩实验](../MAINTENANCE.md)比较相同固定小图的 FP16/W8/LUT4/W8A8，并保留 compiled MLX 的 host→GPU→host 控制。小图 preferred CPU，较大图需要另测；不能从 CPU+NE 选项给结果贴上“ANE 速度”。LUT4 即使实验门槛允许，也可能误差很大，必须展示数值，不能只报 `component-pass`。

既有可选入口：

以上为 CPU 演示。实际公共图脚本与本轮命令见实验页，显式 opt-in，保留编译失败、数值拒绝和 GPU-only fallback。更换 OS/SDK/图表达后可以重开负结果，但不可删除旧记录。
