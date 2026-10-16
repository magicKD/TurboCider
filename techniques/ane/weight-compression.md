# 权重压缩：LUT4、线性 INT4/INT8 与按需解压

[ANE 目录](README.md) · [W8A8](w8a8-int8.md) · [旋转与校准](quantization.md)

## 4-bit 不只一种编码

| 表示 | 恢复单个权重的概念公式 | 额外信息 | 不等于 |
| --- | --- | --- | --- |
| FP16 | 直接浮点值 | 无量化 scale | 原 BF16 无损 |
| 线性 W8/W4 | `scale*(code-zero_point)` | 每张量/通道/块的 scale、zero point | LUT4、硬件整数乘加 |
| LUT4 / palettization | `palette[index4]` | 每个码本的浮点或量化 centroids | 16 个整数权重值、INT4 ALU |
| vector LUT | 一个索引恢复多个权重 | 多维 centroid、轴/cluster_dim | 和 scalar LUT 相同的物理 packing |
| GGUF / MLX Q4 | 由具体格式定义 | group、scale/bias、块结构、排列 | 可直接交给 Core ML LUT decoder |

[Apple LUT 概述](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-overview.html)给出多种 index bit width、分组、per-channel scale 和 vector clustering。[压缩组合](https://apple.github.io/coremltools/docs-guides/source/opt-joint-compression.html)进一步区分 LUT 自身量化和权重索引压缩。它们应写进 cache/recipe 身份，而非一个 `bits=4`。

## 为什么可能快，为什么也可能不快

压缩减少权重传输，只有在执行端保留压缩到临近计算时解码，才更可能降低运行带宽/驻留。本轮所用 W8 MIL 有 `constexpr_blockwise_shift_scale`，LUT4 有 `constexpr_lut_to_dense`；这些是语义表达，不证明在哪里、何时物化。官方 [LUT 性能说明](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-perf.html)指出 iOS17/macOS14 起 ANE 更可能按需解压，且 grouped-channel LUT 太多会损害速度。

粗略体积模型（原创推导，不是运行内存预测）：N 个标量、b-bit 索引、G 个码本、每个码本 2^b 个 q-bit centroid，约需 `N*b + G*2^b*q` bits，另加 scale、对齐和容器开销。若每组只有很少权重，码本开销会明显侵蚀理想 4× 压缩；vector LUT 还要乘 centroid 的维数并改变索引数量。

执行是否受权重带宽限制同样关键。小 graph 常由调用/转换成本支配；更多 LUT、Q/DQ 或重新 staging 可抵消收益。模型 package 体积也不是 ANE SRAM、进程峰值或实际 DRAM 字节。

## 三种实现路线

1. **转换后 PTQ**：`ct.optimize.coreml.palettize_weights` / `linear_quantize_weights`。不训练，易做逐层 mixed-bit；对同源 FP 模型校验质量。适合先建立部署基线。
2. **训练时压缩**：Apple `optimize.torch` 的 DKM 等方法在训练/微调时学习更适配码本的权重，然后导出。**训练时压缩不表示训练在 ANE 上发生**；通常训练后端另选 CPU/GPU。参见 [官方算法](https://apple.github.io/coremltools/docs-guides/source/opt-palettization-algos.html)。
3. **私有流式压缩**：ANEForge emitter/runtime 提供 INT8、LUT 等表达；参数仍必须经过其具体 blob/MIL 配方。不能把 GGUF 的原始 packed bytes 当成同一种 LUT 输入；私有兼容性及 checkpoint 精度另验。

ANEMLL 的 [转换器](https://github.com/Anemll/anemll/blob/f4ad26d061dc2e426faa19f81c367343bdeb9f0d/anemll/ane_converter/llama_converter.py)在 `_make_palettizer_config` 按 per-tensor 或 grouped-channel 构建 kmeans config，并支持 embeddings/lm-head 独立 bit/组宽。Apple SD 则采用 [mixed-bit 配方](stable-diffusion.md)。两者都说明“所有层统一 4 bit”不是唯一选择。

## 本机观察与采用规则

[完整实验](experiments-20261016.md)中，LUT4 package 约是 FP16 的 1/4，但小图执行计划首选 CPU、延迟基本不变；大图仅出现小幅热态改善。自写的简单标量 Lloyd 码本在随机多层图产生约 20%–27% relative L2，**只通过预先声明的宽松合成压缩观察门槛，不是模型可用精度**。不能拿它宣称无损加速，更不能给扩散质量资格。

实际模型应先比较同一 checkpoint 的 FP、fake/decompressed quant 和后端输出，再评估完整生成/识别质量；敏感层保留高精度或更高 bit。冻结 graph、group axis、scale dtype、calibration、LoRA 和输入域，才能比较精度/内存/延迟的 Pareto 取舍。

## 本地学习入口与复查

- [quantization.py](../labs/quantization.py)：原创CPU合同/显式实验客户端，不是上游完整实现。
- [ane/compression_survey.py](../labs/ane/compression_survey.py)：原创CPU合同/显式实验客户端，不是上游完整实现。

CPU入口：`python3 -B techniques/labs/run.py quant-formats`。只运行参考数学/合同，不自动触发Core ML、私有API或网络；实际设备命令与证据边界见[实验报告](experiments-20261016.md)。

SoC/OS/SDK、源码/编译入口、权重/LoRA、输入域/shape/layout/精度或实际消费者改变时重开；不继承旧数值、性能或物理设备资格。
