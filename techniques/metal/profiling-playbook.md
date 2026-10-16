# GPU / Metal profiling：从请求时间到矩阵指令的证据链

[GPU 导航](README.md) · [工具箱](../profiling/tools.md) · [测量合同](../profiling/measurement.md)

核查：2026-10-16。Profiling 的首要产物是**解释关键路径的证据**，不是一张 GPU 利用率截图。本机只有 CLT：可以运行 Metal、counter probe 和 capture API，不代表可运行完整 Instruments/Xcode GPU debugger；既有 [本机能力报告](../review/gpu-profiling-20261016.md)保留成功/失败与缺失能力。

## 工具选择与观测边界

| 工具/接口 | 能回答什么 | 不能单独证明什么 |
| --- | --- | --- |
| host monotonic clock + 同步 | 请求/组件从提交到完成的 wall latency | 每个 kernel 的持续时间；不同队列物理重叠 |
| `MTLCommandBuffer` GPU start/end | 完成后 command buffer 的 GPU 时间跨度 | 内部 kernel 耗时之和；DRAM bytes 或 compute utilization |
| `MTLCounterSampleBuffer` | 设备支持的 timestamp/counter 采样 | 所有 Mac 都支持相同 counter；两个数相减天然为 ns |
| Instruments Metal System Trace | CPU 编码、queue、GPU 调度与等待关联 | 一个采集开关对应用完全无扰动 |
| Xcode GPU capture / shader profiler | dispatch、资源、pipeline、shader 热点和可用硬件 counter | capture 下的时间等于未 capture 正式 latency |
| `os_signpost` / Points of Interest | 模型层、stage 与系统事件关联 | signpost span 就是物理 GPU busy span |
| MLX capture / PyTorch MPS profiler | 框架操作映射到 Metal 采集入口 | 框架名或 `fast` 名称就是具体指令路径 |
| gputrace 解析 / Perfetto | 离线关联、时间线、检索 | 任意解析出的字段都有公开稳定 schema/可靠单位 |
| powermetrics / IOReport / 监控软件 | 低频功耗、频率或系统利用率趋势 | 单个 GEMM 占用、哪一层在 NAX/ANE、有效 MAC/s |

API 合同见 [Metal counter 来源](../sources/official/metal-counters.md)与[官方 GPU/CPU 时钟换算指南](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time)。离线解析的私有字段、支持范围和 provenance 见 [gputrace 源码导读](../sources/repos/gputrace-timing-provenance.md)。不要把模型推算的工作量、schedule estimate 与设备实测放在同一列。

## 推荐的六步工作流

1. **固定请求**：source/binary/精度/shape/LoRA/KV/steps；分别记录 cold load、compile、首调用、warm component、完整 request。确认同一台机器没有另一份自有 GPU benchmark 并跑。
2. **未采集 A/B**：预热后 ABBA 或 BAAB、保存全部样本与输出。`mx.eval` / `torch.mps.synchronize` 等完成边界必须一致；读回用于验证要在声明的计时边界内或外固定。
3. **低扰动 host 标记**：标 denoiser、FFN、SDPA、VAE、staging、wait；先找 CPU gap、重复 compile、每层同步、IO 与 CPU fallback。不要每个 layer 强同步后将结果当原请求表现。
4. **小窗口 capture**：同形状同设置只抓少数迭代；排除模型加载和巨型 trace。检查真实 kernel、拷贝、cast、contiguous、encoder、队列与依赖。
5. **有目标的 counter**：先枚举支持集合，校准时间域；根据访存/占用假设选指标。没有 spill、DRAM 或 bank counter 时，明确写“未测”，不用 tile 大小代替物理证据。
6. **回到未采集完整消费者**：保持正确性、scratch/驻留、取消与回收；确认采集器 off 时收益仍成立。归因成立与收益成立分开验收。

本地可复用：`python3 -B techniques/labs/metal/run.py fusion`，以及 `graph`、`overlap`、`--diagnostic counter-stage` / `counter-dispatch`。先看 [runner](../labs/metal/run.py) help；capture 必须给新建输出路径。所有模式是学习探针，不替换产品 library。

## MLX：lazy evaluation 与 capture 的三个坑

[官方 Metal debugger 指南](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html)提供 capture 工作流。先确认已安装版本存在 `mx.metal.start_capture/stop_capture`，启动前设置 `MTL_CAPTURE_ENABLED=1`；开始 capture 前将输入/权重准备与 lazy work drain，窗口内实际执行目标并 `mx.eval`，最后停止。具体流程已封装在 [本地 capture probe](../labs/metal/mlx_capture_probe.py)，不要复制不匹配 wheel 的 API。

`mx.compile` 也不是记录一次 GPU command buffer 后永久 replay：它会处理 graph、融合与 specialization。[官方编译文档](https://ml-explore.github.io/mlx/build/html/usage/compile.html)说明限制和 shape 行为；应单列第一次 trace/compile 与重复 shape 命中，检查闭包常量、可变状态、shape bucket 和 cache 数量。tensor 第一次 `eval` 可以连带执行此前 lazy 工作；仅量函数返回极易得到虚假“极快”。

本地 [MLX 边界报告](../profiling/mlx-boundaries.md)特意区分带 drain 的诊断和原程序延迟。新 [合成探针](experiments-20261016.md)测的是 graph 构造到 `mx.eval` 的 host wall，不是 shader-only。

## PyTorch MPS：开关不是一组通用加速秘方

以 [官方 MPS 环境变量表](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)和固定读本 `MPSProfiler.h` 为准：

- `PYTORCH_MPS_TRACE_SIGNPOSTS` / `PYTORCH_MPS_LOG_PROFILE_INFO` 是 bitmask，参考枚举，不随意把所有数字写成 `1` 后声称获得全部 kernel 时间。
- `PYTORCH_MPS_PREFER_METAL=1` 调整相关 matmul 路径偏好，不会强迫所有 MPSGraph op 变成自定义 Metal。
- `PYTORCH_MPS_FAST_MATH=1` 改变数学编译语义；需要另作数值门禁，不能默认与 baseline 完全等价。
- `PYTORCH_ENABLE_MPS_FALLBACK=1` 允许不支持的 op 回 CPU，可能引入转换与同步；没有报错不代表全部在 GPU。
- allocator high/low watermark 是内存/提交策略，不是免费提速；不要为跑分禁用限制、挤占系统内存或把 swap 当省内存成功。

capture API、profiler context 与 `torch.mps.synchronize()` 以实际 torch 版本文档为准。本机没有安装 torch；本轮只核对源码/文档，没有伪造 MPS 实测。[PR #189019](optimization-prs.md)的 MetalGraph 是未合并候选，不作为稳定公开功能的使用说明。

## 从症状到可否决假设

| 症状 | 第一组检查 | 有用的反证 |
| --- | --- | --- |
| GPU 有周期性大空隙 | host tokenizer/循环、`.item()`、逐层 eval、编译和 IO | 同 kernel 保持、只改变提交 batching；看空隙及 E2E 是否一起下降 |
| 小 M 的低比特核慢 | launch、metadata、packed decode、低 occupancy | 同编码权重比较 QMV / QMM / resident decode，并报告额外 cache |
| 大 M attention 内存暴涨 | score materialization、mask 广播、GQA repeat | online SDPA；同 mask/shape/输出布局、全空行负控 |
| 加大 tile 反而慢 | occupancy、register/TG memory、尾部浪费 | 保持数学/频率策略，扫 shape；没有 counter 不直接命名 spill 根因 |
| 并行两个设备没提速 | 关键路径、依赖、共享带宽、staging | 串行/并行/各自空载与同 consumer 配对；host span 重叠不是设备重叠 |
| capture 快、正式慢 | collector 改变同步、分配或 cache | 同 binary collector-off 配对，检查前后 source/环境完全一致 |

时间域必须保留单位、clock domain、校准来源、误差与不支持状态；见 [timestamp calibration](../profiling/timestamp-calibration.md)与[归一化规则](../profiling/trace-normalization.md)。缺 counter 应输出 null/unsupported，不回填上游峰值。

## 可执行学习控制（不替代上游验证）

本页实现参考：[labs/media.py](../labs/media.py)、[labs/metal/run.py](../labs/metal/run.py)；case：`profiling-scopes`、`metal:counter-stage`。CPU/旧 Metal 控制只验证对应数学与合同；`mlx:engine-mechanisms` 对应[显式组件 CLI](experiments-20261016.md)，不是自动启动 GPU 的普通单元测试，也不证明全部上游引擎或 PR 已运行。设备、source、shape、精度或消费者改变后重新验证。
