# Apple Silicon：按能力与证据适配，不按型号猜测

<!-- knowledge
id: technique-hardware-apple-silicon
type: method
operator: scheduling
backend: metal, coreml, private-ane
device: cpu, gpu, ane
symptom: straggler, memory-bound
semantics: contract-only
boundary: request
api: public, private
-->

[总目录](../README.md) · [校准](../gpu-ane/calibration.md)

## 三个容易混淆的计算资源

- Metal GPU：attention、norm/RoPE、量化读取、图调度和通用 kernel 所在。
- Neural Engine / ANE：通过 Core ML，或显式私有研究 client 执行固定图；不是可任意 dispatch Metal kernel 的另一个 GPU。
- M5 系列 GPU 的 Neural Accelerators：属于 GPU，和独立 Neural Engine 是不同资源。不能把 Metal 4 tensor/matmul 的提速写成 GPU+ANE 并行。

Apple 的 M5 Pro/Max 公告分别列出了 GPU Neural Accelerators 与 Neural Engine；这里只用来区分资源，不用公告峰值预测 TurboCider 吞吐。公开来源核查日期 2026-10-15：[Apple 公告](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/)。

## 应记录的 profile key

精确 SoC 名称、GPU cores、物理 RAM、macOS build、SDK / Metal language、MLX / Core ML 工具版本、实际 library/runtime build、供电/热状态、workload geometry、精度和 source encoding。基础 / Pro / Max / Ultra，乃至同型号不同内存都不是可互换的性能 profile。

| 范围 | 可借鉴的起点 | 不可继承的结论 |
| --- | --- | --- |
| M1/M2/M3 或本仓库无实测机器 | 普通 GPU、已支持 Metal API；实际探测可用 features | 不继承 M4/M5 的 ANE share、MPP tile、INT8 能力或内存资格 |
| M4 Max 64 GiB | 仓库多数 channel split / shared IO 实验的测量平台 | 不把这里的速度或无 swap 外推到 24/48 GiB |
| M4 Pro 48 GiB | 保留自身 Core ML copy / cache 历史验证 | 与 Max 同 ANE 命名不代表最优比例相同 |
| M5 Pro 24 GiB | 只读 GPU suffix、最小 row bucket、阶段释放的本机经验 | 不泛化到 M5 Max、其他容量或任意 LoRA |
| 未来 / 未登记设备 | 查询能力、编译小 probe、numeric gate，再建 profile | SDK 编译成功不等于指令快，也不授予自动 ANE |

当前内置设备优化见 [device_optimizations.hpp](../../native/runtime/device_optimizations.hpp)。其中 M5 Pro 24 GiB 精确匹配才开启一组 streaming/生命周期策略；`legacy` 只是这张表的策略回退，不代表芯片不支持 Metal。其他自动选择还存在模型级策略，不能只读此表宣称全产品能力。

## 为什么 GPU 越强，ANE 越不一定划算

更快的 GPU 会缩短可隐藏 ANE 的窗口；ANE 固定启动、pack、restore 却不会按 GPU core 数同比缩短。反之，小 GPU 也不自动获益：小内存可能放不下 Core ML weights、双 surface 和完整 GPU suffix。实际判断必须比较 `full_gpu` 与 `stage + max(gpu_shard,ane_shard) + exposed_join`。

统一内存还让 GPU、CPU staging、ANE、SSD/page-cache 和媒体工作争用资源。独立运行的吞吐不能相加；并行双支都变慢时应测耦合，而不是继续增加 offload。

## M5 小内存的可复用教训

[M5 实验](../../docs/design/m5-ane-adaptation.md)中，从 1024 切回 512 后误用大 row bucket，使同一 512 hybrid 请求由 7.119 s 退化到 12.387 s；修复是按真实图像 rows + 实际文本长度重新选最小可用桶，不是换芯片或加大 ANE 比例。

同文的 GPU-only suffix loading 使一组 streaming 权重读取由 55.36 降到 26.48 GB；要同时考虑 w1/w3 的连续行切片和 w2 的列切片整理成本。首次整理、临时磁盘和跨请求 reuse 都要记账。不要把 10 GiB denoise hint 写成整进程 10 GiB 上限。

## 采集与验证

只读环境检查可用 `sw_vers`、`uname -m`、`sysctl -n hw.memsize`，GPU 能力应由 Metal device/runtime 报告而非 CPU 名称推断。记录 power/thermal 是观测，不擅自改变电源模式、wired memory、swap、关闭用户进程或清系统缓存。

新 profile 必须同时通过精度、尾形状、物理完成/所有权、完整请求延迟和内存观察。默认策略宁可保持 GPU，也不能用用户填写的设备名绕过真实硬件检查。

## 实现参考与实验

可读实现：[参考代码](../examples/reference/systems.py)；[实验步骤与边界](../examples/README.md)；本项全部 case / controls。

本项控制：actual rows vs canvas-only bucket；actual Metal device vs assumed profile。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：new SoC/RAM/OS or bucket/geometry; establish a new profile。历史负结果限定于其来源/shape/边界，不关闭该方向。
