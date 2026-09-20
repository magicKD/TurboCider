# Z-Image Turbo INT8：通用 streaming 接入与对照

实现基于 `feat/stream@8e2e97e`，位于 `feature/z-image-generic-int8-streaming`。旧方案基线为 `25b6d08` 的冻结 native library。原始工作区的用户改动未修改。

## 架构边界

- 通用 `StageExecutor`、`SlotSafetyTracker`、`IoExecutor`、layout compiler 的调度和状态机实现不变，没有添加模型名称、INT8 转换或 ANE 分区分支。
- 通用请求规划增加可选的 `ModelModule::validate_manual_streaming_execution` 回调。未提供回调的模型仍要求 GPU-only；Z-Image 自己验证手动 GPU / 已测设备 GPU+ANE 能力。公开 selector 的校验独立保留。
- Z-Image 的 `StreamingMetadata` 生成同一份 materialization plan，供 descriptor 和 reader 使用。它区分源 dtype/shape/range、转换后的 dtype/shape/bytes、256-byte 槽位字段容量、固定权重和派生 suffix 文件。
- INT8 reader 保留已有 ConvRot 数值语义：signed I8 → XOR 128 的 packed U32；行 scale 展开为 group-32 affine scales/biases；默认 BF16 scale 先舍入再求 bias；F32 辅助参数转 BF16。转换在预分配缓冲区内进行，worker 不调用 MLX。每槽 64 KiB scratch 纳入规划。
- 精确路径读取 metadata 捕获的 SourceLease fd；ANE down-projection 后半段写入与该来源/分区绑定的匿名临时文件。原始 checkpoint 不修改。
- Z-Image 槽位和 job 数组按编译后的布局分配，INT8 支持 K1–K3、D<K、Q≤K。并行 worker 的统计更新加锁。数据读取由通用执行器调度，没有嵌套旧 `std::async` 预取。
- 复用既有 ConvRot kernel 和 GPU+ANE 分工；同步完成 GPU 输出与 ANE predict 后，再向通用执行器确认 reader 完成。

填充完成回调返回的是 **materialized content bytes**，不是源读取字节；这是原 C ABI 已有的协议。转换后的数据量可能变大或变小。源读取量单独记录在 `request_bytes_loaded`，并与 descriptor 的 source ranges 对照。

## 验证入口与限制

本次使用私有 `tc_engine_create_model_candidate` + manual layout，未添加公开 preset/catalog，未让 INT8 走 BF16 的公开身份或执行凭证。公开路径额外按 tensor metadata 拒绝改名为 BF16 文件名的 INT8 checkpoint。

现有公开 receipt v2 的 `logical_read_bytes` 由 materialized content completion 累计；发布具有转换的 reader 前仍需为通用 receipt 增加源读取与目标内容的明确区分。本次 INT8 candidate 不签发该公开凭证，实测读取量来自 reader 统计。

手动布局不接受旧 `memory_budget_bytes`。因此本次对齐旧 6 GiB 方案选出的常驻层数和槽位数，并检查实际峰值与 estimated working set；这不等于认证公开 6 GiB 档位，也不是进程内存硬上限。

## 测量条件

Apple M5 Pro / 24 GiB，macOS 26.4.1。相同 INT8 ConvRot checkpoint、提示词、seed 42、512×512、8 步、90 text tokens、无 LoRA。ANE 使用同一 a4096 分区与 manifest。

- INT8+ANE：P10 / K3 / 20 个流式层，每张 160 次 refill。通用框架测 Q1 和 Q2，D2；旧实现双层 lookahead 会并行预取。
- INT8 GPU：P7 / K2 / 23 个流式层，每张 184 次 refill，通用框架 D0/Q1。
- 每个独立进程只加载一份 library。顺序 ABBA；每个会话 3 张，第一张单独保留，不计热态中位数。未按耗时剔除样本。
- 同时记录 native generate（含 PNG 导出）、去噪、完整 create/generate/free、MLX allocator、100 ms 采样的 process phys_footprint 以及系统级 VM 增量。
- 正常 OS 缓存，不清缓存、不关闭其他桌面应用。系统 VM 增量不能归因到单一进程；进程 footprint 是采样峰值。MLX 指标不含 Core ML、OS 与文件缓存。

## 结果

正式对照共 **60 张成功生成图片**，每个执行模式内，新旧输出 PNG 字节完全一致。另有数值、BF16 与取消恢复检查。以下是热态中位数；Q1 主组每侧 8 个样本，其他组每侧 4 个样本。

| 场景 | 指标 | 旧实现 | 通用框架 | 本轮差异 |
|---|---|---:|---:|---:|
| INT8+ANE，复用引擎，Q1 | 整张生成 | 7.045 s | 8.228 s | +16.8% |
| INT8+ANE，复用引擎，Q1 | 去噪 | 5.945 s | 6.091 s | +2.5% |
| INT8+ANE，复用引擎，Q2 | 整张生成 | 6.922 s | 8.328 s | +20.3% |
| INT8+ANE，复用引擎，Q2 | 去噪 | 5.970 s | 6.179 s | +3.5% |
| INT8+ANE，每张重建引擎，Q1 | 整张生成 | 12.147 s | 12.125 s | -0.2% |
| INT8+ANE，每张重建引擎，Q1 | 去噪 | 6.208 s | 6.355 s | +2.4% |
| 同上 | 创建→生成→释放 | 12.313 s | 12.288 s | -0.2% |
| INT8 GPU，复用引擎，Q1 | 整张生成 | 9.217 s | 8.319 s | -9.7% |
| INT8 GPU，复用引擎，Q1 | 去噪 | 8.522 s | 7.131 s | -16.3% |

**对齐结论：** 本轮输入上数值已对齐；INT8+ANE 的去噪接近旧方案。两边都每张重建引擎时，完整生命周期差异约 −0.2%，在本轮波动内。复用引擎时，通用框架的 ANE 端到端耗时仍高约 17%–20%。纯 GPU 本轮整张生成低约 9.7%，两组配对方向一致，但样本不足以推广到其他尺寸、提示词或设备。

**内存：** INT8+ANE 热态 MLX 峰值都是约 5.968 GiB；generate 返回后的 MLX active 从 2.820 GiB 降到 0.324 GiB。Q1 组采样 process footprint 峰值中位数约 7.821 / 7.774 GiB，未显示明显整体峰值下降。纯 GPU 热态 MLX 峰值约 5.943 GiB，两路也基本相同。每张重建引擎的 ANE 对照中，两边 MLX 峰值均约 8.396 GiB，包含重新进行文本编码的阶段；旧 6 GiB 预算并非整次请求上限。

**读取与生命周期：** ANE 旧实现热态每张 logical source reads 为 21.471 GB，不再打包。通用框架每张为 23.483 GB，另有 suffix 打包读取 1.258 GB、写入 0.755 GB，打包中位数约 0.38–0.39 秒。每张重建引擎的对照中，两边读取量与打包量相同。I/O 统计包含文件缓存命中，不能当作物理 SSD 流量；并行读取耗时之和也不能直接当成端到端耗时。

Q2 已验证正确，但本轮未观察到总耗时改善；不能仅凭双 worker 假定更快。Q1 主组四对会话的端到端比值分别为 1.141、1.206、1.243、1.036，均高于旧方案；去噪差异存在方向变化。相同生命周期两对会话的完整耗时比值为 1.006、0.995，支持“基本接近”，不支持宣称稳定提速。

这些是单设备、单工作负载的探索性测量。新旧 library 来自不同分支；已核对 ConvRot 计算与 Core ML 实现沿用同一数学路径，并验证输出一致，但不将所有差异都归因于调度器。未关闭其他应用，桌面环境和缓存会引入波动。

若要进一步对齐连续生成的 ANE 延迟，下一步应设计通用、显式的会话保留策略，区分固定权重、resident prefix、slot pool、派生文件的所有权与释放边界；不能在声明 request retention 时偷偷保留这些权重。


## 测试

- INT8 exact reader + 通用执行器：K1/K2/K3、完整/裁剪权重、Q2 并行填充、FP32 scale 选择；所有张量与 resident MLX packing 精确一致。覆盖实际源读取字节、内容字节、字段对齐、固定权重、槽位复用、转换 scratch、取消与来源变化。
- 既有 BF16 reader 7 项回归通过。修正了分支原有 single-slot 测试漏乘 13 个张量数的求和预期。
- 请求 contract 13 项通过；私有 candidate / 公开入口隔离测试通过。
- 通用核心：3535 个有效布局、14 种 K/D/Q 组合、descriptor 字节与身份、故障清理、C bridge、receipt、跨 pass carry 测试通过。
- BF16 metadata-only descriptor 回归通过；新旧 BF16 真实生成 PNG 字节一致。
- 新增公开适配器测试：改名为 BF16 的 INT8 checkpoint 在编译公开计划时被明确拒绝。
- 真实 INT8+ANE 在第 12 个 main block 取消，随后同一引擎重试成功，输出与旧方案 PNG 字节一致。

## 复现

构建需 MLX native 依赖，candidate benchmark 使用带 test hooks 的构建：

```sh
TURBOCIDER_BUILD_TEST_HOOKS=1 TURBOCIDER_NATIVE_ONLY=1 tools/native/build.sh
TURBOCIDER_TEST_GPU=1 .venv/bin/python3 tests/native/test_z_image_int8_exact.py
```

对比工具：`tools/native/benchmark_z_image_generic_streaming.py`。传入旧/新 library、同一模型目录、原有 legacy streamed 请求和新输出目录；例如 ANE 使用 `--prefix 10 --slots 3 --distance 2 --workers 2`。`--lifecycle per_request` 比较相同的引擎创建/释放策略。公开 catalog 不会被修改。

本机完整原始结果与冻结 library：`local-experiments/z-image-generic-int8-20260920/`（原始工作区内，未加入版本控制）。包含每次请求 JSON、PNG、进程采样、`summary.json`、`experiment.json`、取消恢复记录和测试日志。
