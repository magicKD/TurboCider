# MLX affine：dispatch、metadata精度与原始模型基线

[来源索引](../README.md) · [调优机制](../../metal/kernel-tuning.md) · [独立实验驱动](../../labs/metal/quantized_bench.py)

固定读取两份源码：`ml-explore/mlx@7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`（tag v0.32.0）和`36969c094354086e15d2a327a9c50feb20b9cf11`。所读`.cpp/.h`范围、完整SHA见[同一来源登记](../../references/sources.json)。旧checkout只读；为tag另外建立小型sparse checkout，不fetch/reset旧树。许可证MIT，Apple版权保留；原创SG实验不是MLX内核的复制。

实际执行的是已安装MLX0.32.0，报告记录core/dylib/metallib SHA。**版本号相同不是wheel与tag的可复现构建证明**；不把较新源码里的heuristic反填给旧二进制，也不将网页版本冒充运行版本。

## 小M与大M走的不是同一条路

`QuantizedMatmul::eval_gpu`先处理X、packed W及scale/bias的row-contiguous需求，再按实际architecture、D/O和transpose选择QMV/QMM。`get_qmv_batch_limit`在两个固定源码版本中并不相同；不应把一个batch阈值当硬件常数。

本机API返回的architecture是`applegpu_g16s`，虽然营销名是M4 Max。不要凭Max/Ultra名称猜测源码中architecture末尾的`'d'`分支。此查询不是逐kernel trace。

大M的`qmm_splitk`目标是增加threadgroups；本轮真实M1024、大N已有足够tiles，按该源码公式会回到split1。小M/窄N则可能分K，不能跨shape搬性能和误差结论。v0.32.0的partial暂存是输出类型；另一个固定版本改为FP32 partial，并增加BK/group对齐约束。这里只证明源码差异，不宣称所有已发布wheel都等同其中一个版本。

## Packed不等于整数MAC，也不等于唯一解码精度

affine编码是uint32从低位到高位的codes，加每group的浮点scale与bias；它与[GGML的块布局](ggml-metal-layouts.md)不同。`QuantizedBlockLoader`把数据解码为浮点tile，再送BlockMMA；仅凭Q4/Q8标签不能证明原生整数MAC或独立ANE执行。NAX属于GPU路径，不是ANE调度入口。

v0.32.0的Q4 decoder用类型U保存`scale/16`；另一个固定版本先转float计算，再cast回U。半精度division先舍入与最后统一舍入在subnormal/抵消输入上可能不同。

本轮小探针在同一安装库观察到：公开dequantize与独立canonical FP16 storage一致；一组Q4 tiny-scale数据的M1 packed matmul与该参考一致，M33却不同。原创float-decode SG核与canonical一致。该现象与已读half-division表达的模拟相符，但没有物理dispatch trace，**不能将一个例子说成整个版本/硬件都错误**。正常值控制和Q8控制也保留。

## 真正可迁移的是数据流与验证合同

[原创核](../../labs/metal/affine_gemm.metal)比较BM32/64、连续K加载、16-byte row skew；X/Q/S/B都使用实际stride。完整FFN保留fused gate/up、SiLU、down和输出guard，未把scalar教学核当强基线。

连续加载和padding是可测试假设，不是自动收益。本轮改变这两项没有让原创核战胜MLX，不能从源码直觉虚构bank-stall或DRAM counter。更大的M tile改善了此原型，却仍不如现有强路径；需另用实际counter/指令与编译器证据解释。

缓存展开矩阵能绕过每次decode，但增加resident bytes和setup；与packed基线快4%不等于快过原始BF16图。实验另读取原始BF16 checkpoint和activation，保留`[1,M,H]` BF16输入/输出，计入候选输入转换和输出恢复；这比用量化图自比更接近产品消费者，但仍不是整请求质量门禁。

重开条件：库/OS/编译器、architecture、M/N/K、metadata dtype、布局、权重/LoRA source、驻留/IO政策变化。只有新的数值、消费者与请求证据才能晋升，不靠源码存在或版本号。

[固定release源码](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/backend/metal/quantized.cpp) · [固定kernel源码](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/backend/metal/kernels/quantized.h) · [官方quantized_matmul API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html)。
