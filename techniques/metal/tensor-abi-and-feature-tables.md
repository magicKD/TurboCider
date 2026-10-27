# Tensor ABI 深读：低比特格式、逻辑尺寸、物理 pitch 和使用方式

<!-- knowledge
id: technique-metal-tensor-abi-and-feature-tables
type: implementation
stage: ffn, prefill
operator: layout, quantization
backend: metal, cpu-reference
device: cpu, gpu
symptom: measurement, memory-bound, lifetime
semantics: contract-only
boundary: producer-consumer
api: public, reference-only
-->

[GPU 导航](README.md) · [TensorOps/M5](tensorops-and-m5.md) · [第二轮来源](sources.md#deepdives-sources-20261016)

2026-10-16 核查 Apple [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)，文件页脚日期 2026-05-21；重点视觉核对第 17 页的表格及六条脚注。PDF 和 PNG 只保存在本地 ignored 实验目录，不将整页官方图片复制进知识库；登记内容 hash 和页码。

## 四种“支持”不能合并

`MTLTensor` 对象可表达某 dtype/layout、MPP 的某 operation 接受 operand、特定 GPU 走硬件矩阵路线，以及完整模型更快，是四件事。该功能表的 tensor limits 也不等于 `MPSMatrixDescriptor` 的 rowBytes 规则。原生 MPS 的实验必须用 MPS 自己的 descriptor/helper，而非照搬本表。

## 先分 ordinary 与 tensor-format，再看位宽

第 17 页显示 sub-byte/格式化 FP8 与 ordinary dtype 的 rank、行尺寸、行 stride、slice 和 base offset 条件不同；同 8-bit 的普通 Int8 与 MetalFloat8 不能共享一套 packing 假设。16/32-bit 普通类型与低比特 tensor-format 的 block-scale 支持也不同。

以下只摘取工程最容易错的必要事实，非完整规格镜像：

| 格式类别 | 逻辑行尺寸的元素倍数 | 行 stride 的元素倍数 | buffer 起点的字节倍数 |
| --- | ---: | ---: | ---: |
| Int2 / UInt2 | 32 | 512 | 128 |
| Int4 / UInt4 / Float4E2M1 | 32 | 256 | 128 |
| 格式化 Float8 E4M3/E5M2 | 32 | 128 | 128 |
| ordinary Int8 | 1 | 1 | 1 |
| ordinary Float16 / Float32 | 1 | 1 | 2 / 4 |

脚注还规定 `strides[0]=1`，而 machine-learning usage 对行 stride 和 buffer offset 再加 64-byte 条件；它不会降低原先更严的 128-byte 条件。slice 对齐是另一列，不能把 slice origin 合法误当 buffer offset 合法。

## 用一个 Int4 例子区分三个尺寸

设原始 K=257，打算创建 Int4 tensor，并将 K 映射为最快维 `extents[0]`：逻辑 extent 需按行尺寸规则补为 288；物理 row stride 仍需按 256-element 倍数补为 512，即 256 bytes。这里的 tensor 维度约定不能直接当成 MLX/PyTorch 的 shape 索引顺序。不是“288 个 int4 =144 bytes，所以 rowBytes=144 即可”。多出的逻辑元素需定义为对数学无贡献的 padding，并保留原始 K/scale/mask 合同；特别是 affine 格式的 code=0 可能仍代表非零 bias，不能只填零字节就假定数学为零。

同样，Int2 的 512 **elements** 对应 128 bytes，不是 512 bytes。字节偏移、元素 stride 与 code 数量在接口里需要不同类型/名字，最好不要都叫 `stride`。

这个例子是从表中做的算术推导，未创建/执行 Int4 `MTLTensor`，没有 M5 性能资格。本地 tensor_layout CPU 模型验证计算和负控；它不是通用 driver validator，也未覆盖 rank/slice/plane/设备功能等全部约束。

## Scale plane、低比特重排与真实模型

支持 block scaling 不意味着任意 affine group64 或 GGUF block 可直接绑定。scale plane 的 shape/stride 来自 data-plane 配置；FP16 scale+bias、共享 exponent/codebook、signedness 与 zero point 等仍需按具体 operation 比较。若新增 repack/requant，必须记录 source 权重是否改变、准备开销、临时分配和 cache 身份。

较稳妥的迁移分层：先证明 decode 后的数学值 → 再证明 driver 接受 layout → 再捕获实际 kernel/硬件路线 → 最后量完整 FFN/block/request。源 dtype 不同的“相同 bit 数”不能跳过第一层，SDK 新 enum 也不能跳过第二、三层。

与 [vLLM 的页表/stride](vllm-metal-paged-runtime.md)有共同点：逻辑形状、物理起点和 owner/完成状态都要一起传递。与 [MPS Matrix 对照](case-studies/mps-vs-mlx-20261016.md)的区别是 API 不同；MPS tail 通过不证明上面的 Int4 tensor 合法。

可执行控制：CPU 算术与页地址与负控，case `paged-runtime-contracts`。使用 SDK/OS、dtype、plane layout 或消费者变化后，按新的官方表和实际 header 重核。
