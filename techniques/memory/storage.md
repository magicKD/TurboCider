# Packed 格式、来源身份、mmap 与 pread

<!-- knowledge
id: technique-memory-storage
type: method
stage: pipeline
operator: quantization, layout
backend: metal, mlx, cpu-reference
device: cpu, gpu, ane
symptom: wrong-layout, stale-cache, storage-pressure
semantics: bitwise, numerically-equivalent
representation: q4, q8, gguf, mlx-affine, packed, bf16, fp16
boundary: weight-runtime
api: public, reference-only
-->

[总目录](../README.md) · [ANE 量化](../ane/quantization.md)

## 格式不是只有 bit 数

MLX affine `[out,in]` U32 words、scales/biases、GGML Q4_K/Q6_K/Q8_0 blocks、signed ConvRot、NVFP4 各有 code order、group、metadata 与变换。都叫“Q4/Q8”也不能 raw reinterpret。

传给 stage/kernel 的合同应包含 encoding enum、完整 physical columns/pitch、offset、group、metadata dtype、logical region 和 transform。参考 [WeightView](../../native/backends/ane_runtime.hpp)、[GGUF decoder](../../native/core/gguf_decode.hpp)、[affine weights](../../native/components/weights/affine.cpp)。

## 三种读法的选择

| 方式 | 更可能适合 | 必须测 |
| --- | --- | --- |
| mmap + no-copy Metal view | 已按 consumer packed/aligned，几乎无转换 | 实际是否no-copy、page fault、file-backed/dirty、owner lifetime |
| pread 到已分配 typed buffer | repeated streaming、明确slot控制 | alignment、short read/EINTR、uncached语义、decode成本 |
| bounded decode/derived suffix | GPU只需一部分、原packed不直接支持 | 初次转换、临时磁盘、原来源/recipe、后续读取量 |

Splash 的 WeightStore 使用只读 MAP_SHARED、page-aligned packed sections，适合其离线自有格式；vpipe 的当前指导对 conversion-heavy loaders 倾向 Copied/pread。两者不矛盾：工作集和格式不同，不能用一个项目的偏好定全局 allocator。

## Alignment 是三个独立检查

- tensor/file section 是否满足 Metal offset/自定义 packed ABI。
- CPU/VM no-copy backing 是否按实际 page extent 可映射。
- `F_NOCACHE` 读区间在本机/文件系统是否达到预期，非对齐是否退回缓存路径。

vpipe 文档与 WeightSet 实现特别处理非对齐文件offset的读窗口；这是项目实测策略，不保证所有 OS/文件系统语义。通用 safetensors 的 `8+header_len` 和小 tensor 大小可能破坏后续对齐；不要简单在 blob 中插空洞破坏原格式连续范围约束。需要新 pack 或兼容 writer/读取路径。

## 不保留双份无用数据

拼接 fused matrix 后释放消费掉的源 tensor；read-once 不加入长期 cache；stream_into 不额外先建全 tensor 再 memcpy。mmap后又保存converted copy会两份都占住。只保留后缀时 w1/w3 可连续行读，w2需按列打包，时间与磁盘都算入 setup。

[本轮真实FFN数据](../metal/case-studies/affine-ffn-cost-and-resident-tradeoff.md)提供了一个具体反例：288 MiB原始16-bit权重，affine Q8/Q4及metadata分别153/81 MiB；若保留packed再缓存FP16展开，就变成441/369 MiB。cache相对packed快一点，并不表示相对原始dense更快或更省RAM。这里都是逻辑权重量，不是实测request峰值；原生bank/租约/eviction与IO仍需完整请求确认。

[GGML源码与GPU摘录](../sources/repos/ggml-metal-layouts.md)和[CPU格式oracle](../examples/reference/quantization.py)还展示：两个Q8_0 blocks与MLX affine Q8 group64/FP16 metadata恰好都是68 bytes，却不能raw reinterpret；Q4_0前16/后16的nibble顺序也不是相邻交错。字节数、逻辑shape和路径相同都不能代替encoding/source证明。

## 来源验证与复用

首次内容验证 → 持有source FD/generation → metadata/stat guards → request lease → actual receipt。派生 artifact 记录 parent digest、每项变换、覆盖范围、dtype/rounding 和输出内容身份。不要把派生BF16 view叫原始BF16权重，也不把路径相同当来源没变。

[文件clone官方合同](../sources/official/clonefile.md)用于隔离已有常量实验：同卷共享未改blocks不是hard link，也不是GPU feature alias；高层copy可能fallback普通复制。本轮[真实shared FFN](../gpu_ane/case-studies/qwen-shared-ffn-consumer-cost.md)用严格clone并核对新inode/hash，仍为编译临时文件/cache留admission。只读prepared/load另设流程，不因存储不足降低新clone门槛，也不把未调用compile selector等同于内部没有编译。

验证缓存命中只允许复用已证实的generation；跨进程/路径替换后要重建证明。source相同但refill地址变化时，可用受验证logical content identity复用小metadata，不可因此持有所有权重或忽略mutable内容变化。

## 安全与可复现

先只读估算 checkpoint、ANE源/compiled、derived、scratch、日志和reserve，磁盘不足就不启动大转换。产物写新目录，完整验证后原子发布；失败清理只处理本次owned staging。原模型、第三方仓库和旧实验不因知识整理被改写或删除。

## 实现参考与实验

可读实现：[参考代码](../examples/reference/quantization.py)；[实验步骤与边界](../examples/README.md)；本项全部 case / controls。

本项控制：full pitch vs local pitch；metadata bounds/offsets and positional FD cursor。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：aligned pack, mmap/pread/typed refill or new format。历史负结果限定于其来源/shape/边界，不关闭该方向。
