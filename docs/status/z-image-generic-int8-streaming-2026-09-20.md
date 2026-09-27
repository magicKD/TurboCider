# Z-Image Turbo INT8：通用 streaming 接入与对照

通用实现基于 `feat/stream@8e2e97e`，旧实现基线为 `25b6d08`。性能对照使用各自的冻结 library。

## 通用性与能力边界

- `StageExecutor`、`SlotSafetyTracker`、`IoExecutor` 和 layout compiler 的执行逻辑不变。框架只增加可选的 `ModelModule::validate_manual_streaming_execution` 回调；未提供回调的模型仍要求 GPU-only，公开 selector 继续独立校验。
- Z-Image 适配层负责 ConvRot INT8 转换、ANE 分区、设备限制和 K1–K3 / D<K / Q≤K 能力。descriptor 与 reader 共用 materialization plan，明确源 dtype/range、目标 dtype/shape、内容字节和对齐容量。
- reader 在预分配缓冲区中完成 I8 → packed U32、行 scale → group-32 scales/biases、F32 → BF16；保持原有舍入顺序，每槽 64 KiB scratch 纳入工作集估算。I/O worker 不调用 MLX，也不嵌套旧预取线程。
- 原始权重通过 SourceLease fd 读取，ANE 的 GPU suffix 使用本次来源/分区绑定的匿名临时文件。复用原计算 kernel，在 GPU/ANE 计算完成后才释放槽位。
- 填充回调按既有 ABI 返回 **materialized content bytes**；源读取量单独记录在 `request_bytes_loaded`，并与 descriptor 对照。INT8 不假设源字节数等于目标内容字节数。

仅启用私有 `tc_engine_create_model_candidate` + manual layout。没有新增公开 preset/catalog；公开 BF16 路径按 tensor metadata 拒绝改名的 INT8 checkpoint。现有 receipt v2 的 `logical_read_bytes` 实际累计内容字节，公开量化支持仍需完善源/目标计量和独立资格验证。

手动布局对齐旧 6 GiB 方案选出的 prefix/slot，不继承其预算参数，也不代表完整请求的 6 GiB 内存上限。request retention 保持原语义；若继续优化连续生成延迟，应另行设计通用、显式的会话保留策略。

## 对照结果

Apple M5 Pro / 24 GiB、macOS 26.4.1；相同 ConvRot checkpoint、提示词、seed 42、512×512、8 步、90 text tokens、无 LoRA。ANE 为同一 a4096 分区。ANE 使用 P10/K3/D2，GPU 使用 P7/K2/D0/Q1。独立进程按 ABBA 顺序运行，每会话 3 张，首张不计入热态中位数，未按耗时剔除样本。

正式对照 **60 张全部成功，执行模式内新旧 PNG 字节一致**。下表为热态中位数；ANE Q1 每侧 8 个样本，其余每侧 4 个样本。

| 场景 | 指标 | 旧实现 | 通用框架 | 差异 |
|---|---|---:|---:|---:|
| INT8 GPU，复用引擎 | 整张生成 / 去噪 | 9.217 / 8.522 s | 8.319 / 7.131 s | −9.7% / −16.3% |
| INT8+ANE，复用引擎，Q1 | 整张生成 / 去噪 | 7.045 / 5.945 s | 8.228 / 6.091 s | +16.8% / +2.5% |
| INT8+ANE，复用引擎，Q2 | 整张生成 / 去噪 | 6.922 / 5.970 s | 8.328 / 6.179 s | +20.3% / +3.5% |
| INT8+ANE，每张重建引擎，Q1 | 创建→生成→释放 | 12.313 s | 12.288 s | −0.2% |

本轮数值对齐；相同引擎生命周期下耗时接近，连续复用引擎的 ANE 端到端延迟尚未对齐。Q2 没有改善总耗时。ANE Q1 四对会话端到端比值为 1.141、1.206、1.243、1.036；每张重建引擎的两对比值为 1.006、0.995，不能据此宣称稳定提速。

ANE Q1 热态 MLX 峰值两边约 5.968 GiB，生成后的 active 为 2.820 → 0.324 GiB；采样 process footprint 峰值中位数为 7.821 → 7.774 GiB，整体峰值未明显下降。每张重建引擎的 MLX 峰值两边约 8.396 GiB，包含文本编码。GPU 热态 MLX 峰值两边约 5.943 GiB。

ANE 旧实现热态源读取 21.471 GB/张且不再打包；通用框架为 23.483 GB/张，另有 suffix 打包读取 1.258 GB、写入 0.755 GB，约 0.38–0.39 秒。每张重建引擎时两边读取量与打包量相同，说明生命周期差异是连续生成开销的重要来源。

单设备、单工作负载的探索性结果，不能推广为发布资格。新旧 library 来自不同分支，不能把所有耗时差异都归因于调度器。未清 OS 缓存、未关闭其他桌面应用；100 ms process footprint 为采样峰值，MLX 不包含 Core ML/OS，系统 VM 增量不能归因到单一进程，逻辑读取量不等于物理 SSD 流量。

## 验证与复现

- INT8 reader + 通用执行器：K1/K2/K3、完整/裁剪权重、Q2、FP32 scales，逐张量与 resident MLX packing 精确比较；覆盖源/目标字节、对齐、工作集、复用、取消和来源变化。
- BF16 reader 7 项、请求 contract 13 项、公开/私有入口隔离、改名 INT8 拒绝、metadata-only descriptor 回归。
- 通用核心：3535 个有效布局、14 种 K/D/Q、descriptor 身份与字节、故障清理、C bridge、receipt 和跨 pass carry。
- BF16 真实生成 PNG 一致；INT8+ANE 在第 12 个 main block 取消后，同一引擎重试成功且 PNG 一致。

```sh
TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
make test
TURBOCIDER_TEST_GPU=1 .venv/bin/python3 tests/native/test_z_image_int8_exact.py
```

上述 `make test` 使用默认发布构建。`test_memory_constrained_routes_remain_plan_only_without_manifest` 固定请求 48 GiB，在 24 GiB 机器上会因超出物理内存失败。性能样本使用带 test hooks 的冻结 library；发布构建另行验证，不混用两种构建的性能结果。

对比工具：`tools/native/benchmark_z_image_generic_streaming.py`。指定旧/新 library、同一模型目录、legacy streamed 请求与新输出目录。ANE Q1 参数为 `--prefix 10 --slots 3 --distance 2 --workers 1`；用 `--lifecycle per_request` 比较每张重建引擎。
