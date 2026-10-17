# GPU、NAX、ANE 与 Metal TensorOps：能力和版本分开看

<!-- knowledge
id: technique-gpu-ane-hardware-and-tensorops
type: method
operator: scheduling
backend: metal, coreml, private-ane
device: cpu, gpu, ane
symptom: straggler, memory-bound
semantics: contract-only
boundary: request
api: public, private
-->

[专题导航](README.md) · [GPU GEMM](gpu-gemm-quantization.md) · [ANE 图](ane-graphs-and-quantization.md)

核查：2026-10-16。主要一手资料：[Apple M5/MLX 研究文章](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)、[M5/A19 Tech Talk](https://developer.apple.com/videos/play/tech-talks/111432/)、[WWDC25 Metal 4](https://developer.apple.com/videos/play/wwdc2025/205/)、[WWDC26 Metal tensors](https://developer.apple.com/videos/play/wwdc2026/330/)、[MPP 编程指南](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)。官方的特定 workload 倍率不继承为本库性能。

## 物理设备与软件接口

| 名称 | 是什么 | 不能推导什么 |
| --- | --- | --- |
| CPU / Accelerate / BNNS | CPU 上的系统向量、矩阵和图执行入口 | CPU 矩阵指令不是独立 ANE |
| Metal shader / MPS / MPSGraph | GPU 编程与高层 GPU 算子/图 | 使用普通 shader/MPS 不表示使用 ANE |
| Metal 4 machine-learning pass | 预转换模型在Metal timeline中的公共入口；官方说明系统可选GPU或ANE | 不表示普通MSL自动上ANE，也不保证双设备并行或实际ANE放置 |
| MLX Metal backend | CPU/GPU 数组图，GPU 侧自有 kernels | 统一内存不表示自动将图调度至 ANE |
| M5 GPU Neural Accelerators / NAX | GPU core 内矩阵加速单元，经 Metal/MPP 等入口使用 | 不是独立 Neural Engine；GPU+NAX 不是 GPU+ANE |
| ANE | 独立加速器，经 Core ML/Core AI、Metal4模型pass等公共入口或私有路径执行图 | 不能直接提交任意 MSL shader；公共入口也不保证特定图实际放置 |
| Core ML / Core AI | 导出、编译、放置与执行框架 | compute-units 配置不是物理执行轨迹 |

## GPU 的三个矩阵层次

传统 shader 的标量/向量 FMA 是重要 fallback；SIMD-group matrix 原语让一组线程协作矩阵乘累加；TensorOps/MPP 再提供 tensor、matmul2d、cooperative tensor 等更高层表达。三者是可比较的实现候选，不是“越新一定越快”的排序。

M5/A19 会话说明 TensorOps 可在 M1–M5 的 Apple GPU 上工作，但旧 GPU 走 shader 路线；只有对应硬件才走新的 Neural Accelerator。**可编译/可运行/使用硬件快路径/端到端更快**必须分别取证。芯片营销名称之外还要检查 OS、SDK、GPU family、dtype/shape 和框架 dispatch。

### 精确版本门槛

M5/A19 会话 15:58 起给出：26.1 增加 BF16 tensor；26.3 让 cooperative tensor 可作矩阵输入；26.4 增加 4/8-bit integer tensor 输入。WWDC26 又介绍 multi-plane quantization、operation-derived cooperative helpers 等新能力。不要把不同 overload、helper 和 runtime 支持归为单一“Metal 4 支持”，也不要从会话的“now”倒推任意旧 SDK 已有所有接口。

本地 MLX 读本 `36969c0` 的 `device.cpp::is_nax_available` 同时检查 `MLX_METAL_NO_NAX`、OS 26.2 和 architecture generation（非 `p` 末尾 ≥17、`p` ≥18）。这只是该版本 MLX 的 guard，不是建议应用复制私有 architecture 字符串。BF16 API 的 26.1 与框架 NAX 的 26.2 也不是同一个门槛。本机 `applegpu_g16s` 未达到此 guard。

后续实现读本又展示两层独立性：[PMetal](case-studies/pmetal-speculative-and-dispatch.md)的MPP helper仍可用传统commandBuffer，而[Uzu](case-studies/uzu-metal4-and-routing.md)已真正使用Metal4 allocator/argument table/feedback。反过来，Uzu部分能力查询依赖私有selector，不能把“Metal4实现”整体标成纯公共合同。这些都是指定commit的源码事实，不是本机M5性能。

再续研已在本机M4运行[公共Metal4生命周期probe](case-studies/metal4-public-lifecycle-probe.md)，并核对/执行[独立machine-learning pass入口](case-studies/metal4-machine-learning-passes.md)。后者的官方文档允许系统自动选GPU/ANE，不能因标题写“GPU timeline”而一概排除ANE；但本机两个小包专门化的物理推理单元仍未知。两条实验都没有M5 MXU或物理overlap证据。

## Cooperative tensor 为什么重要

它把一个逻辑 tile 分布到线程协作域，各线程持有 fragment；不是每个线程持有完整矩阵。可将前序运算、解码或 norm 结果接入矩阵原语，减少为了 API 输入形式而额外写回/重读。

WWDC26 的 attention 例子把 Q/K/V、score、softmax 与第二次 matmul 联系起来：如果中间 cooperative operand 合法且寄存器装得下，就可减少 shared memory 与 barrier。维度、scope、布局由 operation helper/SDK 合同决定，不能任意强转 thread array；register pressure/occupancy 也可能使大融合失利。

MPP 指南同时提醒，普通 dense GEMM 的 direct device 访问是有价值基线，不要机械照搬 CUDA 的 threadgroup staging 与多级软件流水。自定义 packed 权重可能仍需 decode staging。应根据具体数据流决定是否增加中间层，而不是一律“先搬 threadgroup”。

其2026-03-16版§2.3还讨论tile可整除性、locality-preserving walk order、长K下simdgroup进度失配和周期barrier、full tile用static extents而tail保留bounds checks。这里的barrier也可能只用于收敛进度，不是要求每步做数据搬运；频率过高会限制可调度指令。M5示例参数只能作为候选起点，不复制成跨芯片最优值。

## 原生量化 tensor 与现有模型格式

官方 integer/multi-plane tensor 支持不表示任意 GGUF Q4、MLX affine Q4、LUT4、FP4 都可零成本直接输入。需要匹配 packed layout、scale/zero-point、group/plane、对齐和运算类型。重新编码、动态量化、重排和 staging 要单列计时；若改变权重所代表的值，要进入模型质量验证。

operand storage、乘法精度、accumulator、输出舍入分别记录。FP32 输入的 TF32 快路径不等于 full FP32 语义；MLX guard 为此显式检查 `enable_tf32` 或非 FP32 dtype。FP16/BF16 则注意范围、subnormal、极值与下游误差存活。

第十七轮的[MLX NAX最终实现案例](case-studies/mlx-nax-mid-m-and-prefetch.md)展示这一差别：affine低比特先在shader解码为T，再接MPP cooperative operand，split partial是float；16-row双weight buffer显式占18KiB（FP16/BF16），FP32对应34KiB并保留旧schedule。新API、量化storage、decoded dtype、register/threadgroup成本和真正硬件执行仍是分开的证据轴，本轮只有原始CPU策略函数执行，没有M5测量。

## ANE 特性采用什么证据

官方 [Core ML 压缩性能页](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html)建议 ANE 的 W8A8 路线优先在 A17 Pro/M4 或更新硬件评估，且强调 shape、融合和 context 影响。它不是“所有 INT8 图都快两倍”。私有反汇编/尺寸探针关于 tile、SRAM 和 FP16 lowering 的观察则绑定其特定芯片/图/编译路径，见 [ANE 专题](ane-graphs-and-quantization.md)。

未来型号没有本轮测量，不填推测峰值或通用最佳 tile。新硬件先做 capability probe，再比较优化后的旧核、SIMD matrix、TensorOps direct、cooperative decode 四臂；最后重新搜索 GPU/ANE share。
