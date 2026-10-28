# TensorOps、M5/NAX 与 Metal 4：能力分层与迁移路径

<!-- knowledge
id: technique-metal-tensorops-and-m5
type: method
stage: ffn, prefill
operator: gemm, quantization
backend: metal, mlx
device: cpu, gpu
symptom: measurement, precision-loss, launch-bound
semantics: contract-only
boundary: producer-consumer
api: public, reference-only
-->

[GPU 导航](README.md) · [量化 GEMM](gemm-and-quantization.md) · [PR 账本](optimization-prs.md)

核查：2026-10-16；本机 M4 Max，**没有 M5/NAX 实测**。本页按官方 API、固定上游源码和待运行实验分层，不用芯片营销名推断 kernel 路径。

## 当前结论与学习重点（2026-10-17 整理）

最值得借鉴的是**能力保护与数据流设计**：先用成熟矩阵核作基线，再判断 cooperative fragment 能否消除某次暂存；TensorOps 与 Metal 4 提交层分开改，保留旧核 fallback。API 可编译、pipeline 可创建、GPU 输出正确、完整消费者更快，是四个不同阶段，不能用其中一个代替其余三个。

学习接口时，应核对实际 runtime MSL 的 view/const 类型与 element-validity 方法，不能机械照抄不同 SDK 的示例。当前本库可移植性探针仅验证到编译/创建，未完成正式 GPU 数值与性能对照，因此不能据此推荐加速路径；过程细节保留在[归档入口](../MAINTENANCE.md)，不作为本页教学步骤。

独立完成的 [Metal 4 counter heap 实验](case-studies/metal4-counter-heaps-and-time-domains.md)只提供执行/计时与生命周期的学习材料，不证明 TensorOps/NAX 已运行，也不代表已迁移 MLX。

## 五个名字各指什么

| 名字 | 所在层 | 不应推导出的结论 |
| --- | --- | --- |
| Metal | GPU 编程/资源/执行 API 与着色语言生态 | 使用 Metal 就用了矩阵加速 |
| MPS / MPSGraph | GPU primitive / 图 API | 使用 MPS 就使用 ANE；图必然只有一个 kernel |
| TensorOps | MSL 中的高层 tensor 运算接口 | API 在旧 GPU 可用就有 M5 的硬件吞吐 |
| Metal Performance Primitives（MPP） | shader 内可组合的库；tensor_ops 是其中一部分 | 等同于 host 的 MPSGraph |
| Neural Accelerators / NAX | M5 一代 GPU 内的矩阵加速路径；NAX 是上游代码常用名 | 等同于独立 Neural Engine/ANE；GPU+NAX 是跨设备并行 |

依据：[Apple MPP 编程指南](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)、[Apple 的 M5/MLX 研究文章](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)、[WWDC26 TensorOps](https://developer.apple.com/videos/play/wwdc2026/330/)。官方文章中的跨代吞吐/首 token 数字有特定模型和系统条件，本页不将其当 TurboCider 的速度保证。

## 版本与硬件：必须同时检查

[Apple M5/A19 Tech Talk](https://developer.apple.com/videos/play/tech-talks/111432/)说明 TensorOps 可移植于 M1–M5，旧 GPU 走 shader 实现；会话 15:58 起将 BF16 tensor、cooperative matmul 输入、4/8-bit integer tensor 分别对应 26.1、26.3、26.4 的 API 增量。它们不是同一次“Metal 4 支持”。具体重载仍需核对实际 SDK、运行时与 deployment target。

WWDC26 进一步介绍 multi-plane 量化和 cooperative-input attention。不要将新文档接口倒推为任意 26.x 已安装 wheel 支持。**框架部署门槛可严于单项 API 门槛。** 下述 MLX 的 26.2 门槛与 BF16 API 的 26.1 不是同一件事。

MLX 固定 `36969c0` 的 `device.cpp::is_nax_available` 还有独立条件：构建不能定义 `MLX_METAL_NO_NAX`；OS 检查为 26.2；architecture generation 检查按末尾代号区分（普通门槛 17、`p` 门槛 18）。这是 MLX 的实现判定，不是推荐应用复制的 Apple 公共 capability API。本机 `applegpu_g16s` 不满足该代码路径。编译通过、guard 通过、trace 看见目标 kernel、性能受益是四级证据。

GGML 固定 `64e9bce` 的 `ggml-metal-device.m` 又是另一套策略：检查 family/environment，再对旧芯片默认关闭 tensor 路线，并编译一个小型 FP16/BF16 TensorOps kernel 作能力探测；独立 tensor metallib 缺失时也会关闭。源码内有“所测旧芯片收益不足”的注释，本报告只将它作为上游选择器的理由，不继承为所有旧 GPU 的性能定论。源码为可能设备预留名字也不是该型号已经发布的证据。

## 为什么 cooperative tensor 能改变数据流

传统可移植实现通常是：device packed W → 解码/写 threadgroup → barrier → SIMD-group matrix/累加 → store。若矩阵操作只接受内存 tensor，norm、旋转、解码或前一 GEMM 的寄存器结果还要先落内存。

cooperative tensor 将一个逻辑 tensor 分布到执行域内的线程，每线程只持有 fragment；它不是每线程一份完整矩阵。类型由目标 operation 的 operand/destination helper 得到，不能猜测 layout 或把任意 thread array 强转成合法 operand。支持 cooperative 输入后，可探索在 fragment 中进行转换并直接衔接矩阵运算，减少暂存；但 register pressure、fragment 分布、scope、合法维度和同步必须满足实际 SDK 合同。

这段是数据流综合解释，不声称某个编译器必定消除所有 load/store。最终仍看 shader 编译产物、counter 与完整 consumer。`threadgroup_barrier` 与 `simdgroup_barrier` 不能随便互换；不同 execution scope 的操作须分别验证。

MPP 指南 §2.2 特别提醒：普通 dense GEMM 不必机械地先 staging 到 threadgroup，也不必照搬其他平台的显式软件流水；直接 device 访问、缓存与足够 occupancy 是重要基线。§2.3 再讨论 tile、walk order、K-loop 同步与静态 extents。因此“直接 device TensorOps”与“自定义 packed decode 后暂存”应是两个候选，而非把后者当强制模板。指南的出发点不否决特定 packed PR 的双 buffer，但要求测量其额外开销。

### Precision 是接口的一部分

operand/storage、compute 与 accumulator 三者分开。FP32 输入启用 TF32 路线会改变有效乘法精度，不能当成 bitwise 不变优化；MLX matmul 的 NAX guard 明确检查 `enable_tf32()` 或非 FP32 dtype。FP16/BF16 也要检查输出舍入、极值、尾部、非有限值和 softmax 稳定性。

“量化 tensor”要求匹配官方 packing、scale/zero-point、分组和对齐；GGUF Q4、MLX affine Q4 和某个硬件支持的 int4 格式不自动兼容。动态 activation quantize、重排、repack 或重定标的成本都要计入。若重新量化会改变所代表的权重，另开模型质量候选，不能称为只换 kernel。

## 四个可迁移读本

1. **MLX Steel NAX GEMM**：`matmul.cpp` → `kernels/steel`。读 host 的小矩阵/split-K/NAX 条件，再追 operand、tile、function constants，别只复制 shader。
2. **MLX quantized NAX**：`quantized.cpp` → `quantized_nax.h/.metal`。关注 packed load、metadata 复用、不同 M、gather 和 tail；[PR #4682](optimization-prs.md)另加预取、双 buffer 和 split-K，尚未合并。
3. **ccv NA kernel**：`NAInt8MatMulSmallMDescriptor.cpp` → kernel；该 descriptor 按 shape 决定 split-K 和 scratch，并管理 binary archive。`NA*` 命名本身不等于 ANE。
4. **SDK 演进适配**：MLX [#4594](optimization-prs.md)将 gated-delta 的 cooperative 类型推导改用 operation helper。它是兼容性修复，不应宣传为已测加速倍率。

固定读本和文件 SHA 见 [来源清单](sources.md#research-sources)。本轮未重建这些 M5 kernel，未移植上游 shader。

## Metal 4 执行 API：与 TensorOps 分开做实验

command allocator、argument table、residency 与新 command buffer 是提交/资源管理层；TensorOps 是 shader 运算层。可以先保持数学 kernel，单独测 host 编码/提交、复用和内存；再保持提交层，单独比较矩阵核。两项一起改变会失去归因。官方入口：[Discover Metal 4](https://developer.apple.com/videos/play/wwdc2025/205/)，参数/可用性以实际 SDK 声明复核，不将这里的学习入口作为最低系统版本变更授权。

## 建议的迁移实验（矩阵未完成，以下四臂未运行）

同一份 packed W 做四臂：现有成熟 kernel、tile 内 decode+SIMD matrix、cooperative decode+TensorOps、合法原生量化 TensorOps。先验证同一 source 语义，再扫 M1/16/64/256/1024、N/K tail、物理 stride、不同 group/dtype。另测准备/repack、首调用、scratch、resident 峰值与完整 FFN/block；有 NAX 的 GPU 上用匹配旧核作控制。

否决条件：只在预准备好的岛里快、使用不同权重、提升有效精度预算后才通过、非有限/尾部错误、完整请求慢、无有界 cache/lifetime。保留旧核 fallback，不为了“跑通新 API”删除 guard。
