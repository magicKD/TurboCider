# Metal kernel 调优：先边界，再 tile

[总目录](../README.md) · [布局与融合](fusion-and-layout.md) · [Attention](attention.md)

## 适用场景

Profiler 已确认某个 GEMM、norm、activation、pack 或转换算子在关键路径上，而不是被 upstream lazy work / IO / Core ML load 误算。起点应是实际最快 GPU 路线，不是一个方便击败的未融合实现。

## 一张 kernel 规格卡

固定 `M/N/K`、输入输出 dtype、accumulator dtype、physical strides、alignment、batch、mask/padding、bias 加法顺序、rounding boundary、alias 允许范围、owner lifetime。quantized GEMM 还要固定 code order、group size、scale/bias dtype 和 source transform。

操作数不要只检查逻辑 shape：切下来的 down projection 仍可能跨完整原行 pitch。`K_local` 与 `K_physical` 必须是两个参数。

## 调优顺序

1. 先消除不必要 materialize / contiguous / transpose，测真实字节流量。
2. memory-bound elementwise 先测试连续 vector load/store（如对齐 float4）、coalescing、一次写回和 tail；不要强行矢量化不对齐视图。
3. GEMM 分清小 M / 大 M：前者权重读取和 launch 比例高，后者更可能获益于 tiling / 矩阵原语。
4. 在 runtime 允许的 threadgroup limits 内，测 SIMD groups、BM/BN/BK 和每线程寄存器占用。更大 tile 可能降低并发、spill 或触发精度问题。
5. 最后做 function constants / shape 专用化：限量 pipeline cache，记录编译成本与 cache key，不按每个任意请求无限编译。
6. 正式结果测完整 consumer，包括 padding、转换和同步；同时保留纯 GPU timestamp 用于归因。

Apple 的 Metal 指南强调资源使用、threadgroup 和调度约束；可参考 [Optimize Metal Performance](https://developer.apple.com/videos/play/wwdc2020/10632/)。这些原则不是某个固定 tile 的通用最优证明。

## SIMD-group / MPP 不是盲目替换

本仓库的 [affine_gpu_mpp.hpp](../../native/backends/affine_gpu_mpp.hpp) 演示 register tile 中按原 metadata 精度 decode，FP32 累加，不建立全局 dense W。它是 research kernel：明确拒绝未验收 tile、M 太小、非法 group 或范围。不要删 guard 来让新模型“能跑”。

[cooperative 负结果](../../docs/status/local512-qwen-defer-cooperative-2026-10-09.md)记录了 K 尾、小 M、SDK cooperative operand 限制和早期 FP16 NaN；修复后数值通过，但 register-right 路线仍明显慢于原核。需要分别验证：

- 完整 tile、M/N/K 任意尾部、偏移、非连续 X、physical W pitch。
- FP16/BF16 源、FP32 / 窄输出、极值、零、非有限拒绝。
- 一致的原始系数 decode；先把 scales 提到 FP32 可能改变所代表的权重。
- candidate selector 实际选择的 fallback，不以 bundle 名称证明每个 projection 使用同一原语。

## 更大tile的一次负结果与后续探索

[wide SwiGLU screen](../../docs/design/validation/z-image-swiglu-wide-screen-2026-09-22.md)与实际 production 32×256 / 4 SIMD-group 核比较，96/128 行、更宽输出 tile 没有稳定赢家，部分组合非有限。最初 probe 退出码为 0 但输出 NaN，后来改成 `valid:false`、无有效 timing、非零退出。

可复用结论不是“32×256 永远最好”，而是：跟真实 production 控制比较、数值失败不纳入最快排名、重复实验排序翻转就不推广。

## Shared / private storage 的边界

`private` 存储需要单独计入 upload/readback；统一内存设备上也不应预设必胜。[独立 parent/storage 实验](../../docs/design/validation/z-image-parent-storage-diagnostic.md)没有建立通用 private-storage 优势，因而没有理由全局更换 allocator。避免用清缓存、重启系统或改变用户电源策略掩盖尚未定位的 kernel 回退。

## 验收最小集

CPU oracle + 独立索引 witness；tail/stride/极值；非有限强失败；实际 kernel label；同 binary 交错控制；组件 → 完整 FFN/block → E2E；新增 scratch 和编译/cache 峰值；取消后 owner 可释放。工具入口见 [profiling/tools](../profiling/tools.md)。

## 实现参考与实验

### Packed affine的真实FFN对照

[MLX固定版本源码](../sources/repos/mlx-affine-dispatch.md)与[GGML布局/摘录执行](../sources/repos/ggml-metal-layouts.md)提供dispatch、scale dtype、物理pitch与矩阵数据流的阅读链。位宽相同不是格式相同；uint32 codes也不是整数MAC资格。

[原创BM32/64核](../labs/metal/affine_gemm.metal)使用float decode→FP16 tile→FP32 accumulator，并显式读取X/Q/S/B的实际strides。连续K加载与16-byte skew是可枚举假设，不是性能保证。小tail/stride控制后，测了[真实Qwen完整FFN](../review/gpu-affine-20261016.md)：BM64比本原型BM32好，但这些配置仍慢于MLX，不能以击败scalar教学核作为晋升标准。

同encoded Q8内驻留展开约快4.7%，但增加288 MiB逻辑cache；与原始BF16消费者比较基本持平。必须区分格式内实现收益和原始模型收益。后者显式使用原始checkpoint/capture，包含输入cast与输出restore，不拿ready FP16岛替代原始ABI。

实际wheel的binary hashes、源码snapshots、所有样本和独立step19确认均在证据中；没有bank/DRAM counter，不用推断假装实测。更多GPU kernel、compile policy、shape与真trace仍可带原控制重试。

可读实现：[参考代码](../labs/metal/kernels.metal)；[实验步骤与边界](../labs/gpu.md)；[本项全部 case / controls](../labs/catalog.json)。

```sh
python3 -B techniques/labs/metal/run.py vector
```

本项控制：scalar/vector4 including tail；scalar/SG8 CPU double witness。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：new BM/BN/BK/SIMD groups, dtype, SDK or physical layout。历史负结果限定于其来源/shape/边界，不关闭该方向。
