# MLC/TVM Metal 深读：从压缩权重到 schedule、MSL 与执行

<!-- knowledge
id: technique-metal-mlc-tvm-metal-compiler
type: implementation
stage: prefill, decode, ffn
operator: gemm, gemv, quantization, layout
backend: metal, cpu-reference
device: cpu, gpu
symptom: compile-bound, launch-bound, wrong-layout, lifetime
semantics: contract-only
boundary: producer-consumer
api: public, reference-only
-->

[GPU 导航](README.md) · [本轮来源](sources.md#compiler-sources-20261016) · [实现与实验核查](../MAINTENANCE.md)

2026-10-16。固定 MLC-LLM `8978ea9aec626d64096aaa92fbc89bda2e3ea802`，并读取它的 `.gitmodules`/gitlink：实际依赖为 **mlc-ai/relax `201fa0001c81431ccce05419b442c47779ea917c`**，不是任意 Apache TVM HEAD。新建该版本的只读 sparse checkout，没有更改已有 MLC 工作树或安装/构建 TVM。

本页记录第三轮选定源码纵向链的阅读深度，当时**没有运行 MLC/TVM 编译、生成模型库或完整模型 benchmark**。后续[第十一轮真实TVM/Metal AOT链](case-studies/tvm-metal-aot-and-runtime-only.md)已实际构建固定fork/FFI，运行原始DLight及336个完整小型消费者输出，含纯runtime原生reload；仍未运行完整MLC模型/Relax pipeline。当前路径为`python/tvm/s_tir/dlight/`等，本文短名不能按旧TVM import路径机械拼接。

## 1. 图、kernel 与 runtime 是三个优化层

```text
GroupQuantizeLinear + model graph
  → Relax passes：transpose / norm / dequant 等模式
  → LegalizeOps / FuseOps / FuseTIR
  → FuseDequantizeMatmulEwise / LowBatchGemvSpecialize
  → DLight Matmul / GEMV / reduction / fallback
  → Metal tensor intrinsics + CodeGenMetal
  → MSL 或 metallib payload + host library/VM
  → pipeline cache / pending encoder / submit & sync
```

MLC 的 `compiler_pass/pipeline.py` 将高层、TIR、低层 schedule 与内存/VM lowering 分阶段，支持 debug-phase dump。图里出现 `dequantize` 不表示完整 dense W 一定先落显存；反过来，写了 fuse pass 也不表示每个模型/shape 必然匹配。应查看 pass 后 IR 和最终 kernel，再量消费者。

`FuseDequantizeMatmulEwise` 检查调用模式、dequant/matmul 名称族和静态 ShapeExpr 等条件，再 FuseTIR。它不是识别任意语义等价 decoder 的超级优化器。多 consumer、额外 aux tensor、transpose/layout 和动态形状都会影响能否形成期望的融合。

## 2. Group quantization 的数学与布局

`group_quantization.py` 的普通 group 路线用 uint32 packing、group scale 和 `(code - max_int) * scale`；4-bit 的 `max_int=7`，常见 registry 的 group=32。它不同于 MLX 常见 affine group64 的 scale+bias，也不是 GGUF Q4_0。

`linear_weight_layout` 明确区分 KN/NK，quant axis、q_weight 与 q_scale shape 随之改变；NK 前向还插入 transpose。int3 每 uint32 放十个 code、留下两 bit，而不是按别的项目格式猜测 packing。源码 quantizer 限制的 code 范围与完整 nibble 所有可能值也要分开。

可迁移机制：将 unpack、偏移、scale 作为 producer，尽可能 inline 到消费权重的 kernel，使一份解码 tile 被多行复用；但必须保持 metadata 精度与布局。全零 group、尾 group、KN/NK、scale underflow 和非法 code 应进入独立数值控制，不能只测温和随机权重。本轮未据源码表达式推断实际量化 bug。

## 3. Small batch 专用化不只是 M=1

`LowBatchGemvSpecialize` 尝试 bucket=2/4 的 LowBatchGEMV；只有 schedule 实际改变原函数、且仅存在一个 symbolic variable 等条件满足时，才组合运行时分支：M≤2、M≤8、其余 GEMM。**bucket 名与分支上界不是同一个数**。静态 shape、不匹配 reduction/access 模式的函数不能直接套这个分支表。

依赖中的 `gemv.py` 对 Metal 有专门 tag/vector/load 策略，还保留 M2 Ultra 校准注释；这属于该实现的经验，不是通用 Apple GPU 最优值。Dynamic M、量化 decode、scale 复用与输出维的线程映射一起决定结果。不能把 CUDA warp 或 shared-memory 配置原样搬到 Metal。

## 4. MetalMatmul：真正落到 SIMD-group matrix

固定依赖的 `dlight/gpu/matmul.py::MetalMatmul` 先把乘法归一化，再 pad/reindex、将 A/B 缓存到 shared，加载到 `metal.simdgroup` fragment，C 分解 init/compute/store，再 inline consumer epilogue。所读配置包含 16×64 的输出覆盖、BK32、8×8 micro tile、32-lane group 与向量化加载；不是对所有芯片/shape 的最优证明。

`tensor_intrin/metal.py` 用 description/implementation 对描述标量循环与 `simdgroup_load/store/multiply_accumulate`；codegen 把 IR op 打印成 MSL。**tensorize 这个编译术语不证明使用 M5 NAX**：所读路径是 SIMD-group 8×8 原语。Metal 分支会尝试专用 schedule，并在异常时回到通用 schedule；编译成功不能证明 tensorization 成功。

这里还有精度边界：所选 f16 multiply-accumulate intrinsic 的 A/B/C 均声明为 float16，不能沿用其他项目的习惯把它写成 FP32 accumulator。helper 接受 dtype 名、实际注册完整 load/compute/store intrinsic、最终 schedule 匹配成功，也是三个不同条件；缺少某个 lowering 组合不等于 Apple GPU 硬件没有这种算术能力。

dtype/shape、padding 的无贡献语义、fragment stride/transpose、TG memory、epilogue 和尾部都应验证。源码中的某些识别/调参条件是编译器实现边界，不应上升为 GPU 硬件不支持。

## 5. AOT 与首次调用：dylib 不等于没有 JIT

`BuildMetal` 的 payload 可以是 MSL 字符串，也可以由 `tvm_callback_metal_compile` 生成 metallib。callback 可显式声明格式，但同一模块的 kernel 格式必须一致。host `.dylib`/VM 已导出，不等于全部 device pipeline 都已创建。

`metal_module.mm::GetPipelineState` 按 device/function 缓存；源代码 payload 在首用时调用 `newLibraryWithSource`，metallib payload 则走 `newLibraryWithData`，然后创建 compute pipeline。MSL 版本受 SDK/runtime family guard 控制，所读 source 分支还启用 fast math；这与离线生成 IR、打包模型权重、生成 device binary、创建 pipeline 是四个阶段。

因此必须单列 conversion/build、load、首调用/首 shape、cache-hit hot run。不要把“只量 warm forward”称为 AOT 消除了所有编译，更不要拿未生成 pipeline 的库文件证明目标 GPU 快路径。

## 6. Runtime：少提交不是省略完成边界

`metal_common.h::Stream` 复用 pending command buffer/compute encoder；blit 插在同一 buffer，遇到 sync/readback/free 等边界 flush。默认 GPU storage 为 private，CPU upload 通过共享 staging pool 再 blit，GPU readback 要 flush/wait；不能因物理统一内存就默认零 copy。

buffer free 前的 drain 是 correctness 成本：尚未提交/完成的 kernel 仍引用 buffer 时不能释放或置 purgeable-empty。staging pool 的复用也有 in-flight 生命周期，不能只保留 CPU 指针。源码的 dispatch/flush/sync/blit 等 profile counters 是软件事件计数，**不是硬件 PMU 或 DRAM bytes**。

codegen 还处理每个 trailing POD scalar 的 eight-byte slot、buffer 参数 binding 和 Metal `float3` 的物理 padding。逻辑 IR dtype/维度正确，不代表 native ABI 自动正确；与 [Tensor ABI](tensor-abi-and-feature-tables.md)同理，但具体接口不能混用。
