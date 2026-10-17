# GGML Metal：块布局、实际摘录执行与tensor政策

<!-- knowledge
id: source-upstream-ggml-metal
type: source
-->

[来源索引](../README.md) · [存储合同](../../memory/storage.md) · [MLX对照](mlx-affine-dispatch.md)

来源：`ggml-org/llama.cpp@64e9bceb2c3a856efed96feda784a50947049feb`，工作树只读。读取`ggml-common.h`中Q4_0/Q8_0定义，Metal dequantizer、mul_mm的所选路径和device capability/policy段；逐文件SHA与准确范围见[来源登记](../README.md)。未运行整个llama.cpp或GGML backend benchmark。

## 大小相等不是格式相等

| 编码 | 每块数学与布局 | 不能省略的区别 |
| --- | --- | --- |
| GGML Q4_0 | 32元素；half d + 16 bytes；低nibble对应前16个元素，高nibble对应后16个；值为(q-8)*d | 不是相邻元素低/高nibble交错 |
| GGML Q8_0 | 32元素；half d + 32个signed int8 | 负数不能按unsigned affine code解释 |
| MLX affine Q8 group64/FP16 meta | uint32 codes，共64 bytes；另有half scale/bias，共4 bytes | 虽与两个Q8_0块同为68 bytes，但group、metadata位置与数学不同 |

CPU[独立decoder与错误对照](../../examples/reference/quantization.py)覆盖错误nibble交错、signed extremes、非有限scale、错误块大小；不因为普通Q4_0通过，就说Q4_K/IQ/FP8等格式兼容。

## 实际执行了什么上游实现

保留MIT许可与版权后，摘取了`dequantize_q4_0`与`dequantize_q8_0`两个**未改函数体**：dequantizers.metal。PROVENANCE记录原文件SHA、line ranges与包装差异；原许可随资产分发。

小型GPU客户端提供相同18/34-byte block structs，以float4x4实例化原函数，safe math，经公共MLX JIT在本机实际执行。正常Q4 nibble与Q8 signed控制和CPU参考一致；tiny subnormal scale的Q4存在差异，完整向量和绝对误差保留。

这是所选函数/模板实例/编译选项的实测，不是完整GGML kernel、半精度输出实例或默认编译选项的全覆盖，也不是模型速度结论。不能把诊断客户端return0当作“所有数值比较通过”：它表示采集完成，case.equal显式区分匹配与反例。

## 数据流与尾部安全

所读`mul_mm.metal`tensor分支把量化A解码到threadgroup，B通过device tensor直接消费；K末tile同时限制两操作数的有效extent。只给A补零不足以保证B不越界；consumer的物理stride和valid extent仍需正确。原型从这个数据流家族学习，不复制整套engine。

`GGML_METAL_HAS_TENSOR`由device capability与policy共同决定。该版本即便看到Metal4 capability，也默认禁用部分旧设备的tensor路线，除非显式override，随后还做编译探针。源码中的旧设备速度注释只是source-reported，不是本轮数字；API可见不等于默认启用，不等于GPU NAX，更不等于ANE。

当前学习交付是格式oracle、带许可摘录执行和原创affine GPU变体；真正GGUF模型、graph scheduler与完整backend timing仍待匹配工作负载验证。新OS/SDK、量化格式、scale dtype、tail/stride或compile policy是重开条件。

[固定类型定义](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/ggml/src/ggml-common.h) · [固定Metal解码](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/ggml/src/ggml-metal/kernels/dequantize.h) · [固定matmul](https://github.com/ggml-org/llama.cpp/blob/64e9bceb2c3a856efed96feda784a50947049feb/ggml/src/ggml-metal/kernels/mul_mm.metal)。
