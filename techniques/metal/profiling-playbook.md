# GPU / Metal profiling：从请求时间到矩阵指令的证据链

<!-- knowledge
id: technique-metal-profiling-playbook
type: method
stage: pipeline
operator: scheduling
backend: metal, mlx, mpsgraph
device: cpu, gpu
symptom: measurement, precision-loss, launch-bound
semantics: contract-only
boundary: producer-consumer
api: public, reference-only
-->

[GPU 导航](README.md) · [工具箱](../profiling/tools.md) · [测量合同](../profiling/measurement.md)

核查：2026-10-16。Profiling 的首要产物是**解释关键路径的证据**，不是一张 GPU 利用率截图。本机只有 CLT：可以运行 Metal、counter probe 和 capture API，不代表可运行完整 Instruments/Xcode GPU debugger；既有 [本机能力报告](case-studies/capture-counters-and-failure-boundaries.md)保留成功/失败与缺失能力。

## 工具选择与观测边界

2026-10-17 方法归纳：先选能回答当前问题的最低必要观测层，再核对覆盖、单位和扰动。当前证据链可按以下顺序学习：

- **请求 / CB：** [MLX 真实窗口](case-studies/mlx-command-buffer-windows-and-queue-scope.md)、[llama.cpp CB 计时](case-studies/llama-command-buffer-timing.md)与[foreign stream 完成反证](case-studies/mlx-benchmark-evaluation-and-stream-completion.md)。先判断谁提交、等到了谁；sum/union/span 分开，不当 shader busy。
- **实际选核：** [dispatch→PSO→函数](case-studies/mlx-dispatch-pso-and-function-association.md)接[attention/wide/split-K 路由](case-studies/mlx-attention-routes-and-wide-gemv-observed.md)。函数创建、绑定、编码与完成分别记录；库中有符号不等于执行。
- **encoder-stage：** [MLX 真实 stage 时间](case-studies/mlx-concurrent-encoder-stage-counters.md)与[采样能力/primitive 边界](case-studies/mlx-stage-sampling-capability-and-primitive-boundaries.md)。一个 stage 可含多次 dispatch；时间不能按数量均分，unsupported 不回填零。
- **新 API 时间域：** [Metal 4 heap/粒度](case-studies/metal4-counter-heaps-and-time-domains.md)与[生命周期/范围失效](case-studies/metal4-feedback-queue-lifetime-and-range-invalidation.md)。实际查询并验证 frequency/epoch；precise 有扰动，relaxed 可合并 marker，整 heap 正控不代表非零 range 已修复。
- **采集有效性：** [Candle capture 反证](case-studies/candle-qwen2-queues-and-capture.md)。API 成功、文件存在、解析器 exit0，都不能替代目标命令覆盖和真实时间来源。

| 工具/接口 | 能回答什么 | 不能单独证明什么 |
| --- | --- | --- |
| host monotonic clock + 同步 | 请求/组件从提交到完成的 wall latency | 每个 kernel 的持续时间；不同队列物理重叠 |
| `MTLCommandBuffer` GPU start/end | 完成后 command buffer 的 GPU 时间跨度 | 内部 kernel 耗时之和；DRAM bytes 或 compute utilization |
| `MTLCounterSampleBuffer` | 设备支持的 timestamp/counter 采样 | 所有 Mac 都支持相同 counter；两个数相减天然为 ns |
| `MTL4CounterHeap` | 新 API 下可用的 marker 与粒度对照 | 继承旧 counter 单位；marker 差等于无扰动 shader 时间 |
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

本地可复用：`python3 -B techniques/labs/metal/run.py fusion`，以及 `graph`、`overlap`、`--diagnostic counter-stage` / `counter-dispatch`。先看  help；capture 必须给新建输出路径。所有模式是学习探针，不替换产品 library。

## 原图buffer观察也会改变执行边界

2026-10-17的[MLC实际stage观察](case-studies/mlc-ffn-stage-observation-and-precision.md)没有新增原图输出或跳过kernel；36个case的before/observed/after完整Y逐位一致。但每stage同步读回让软件flush从1变4、sync从1变9、D2H从1变5，dispatch仍为4。**输出相同和dispatch相同都不能证明计时无扰动**；不能将这种诊断模式的elapsed当原路径性能。

该报告另将实际GPU上游值送入原组件作条件检查，保留完整参考失败，再用typed-FP32投影控制检验数值预算。实际feed条件通过不是从原X出发的质量证明，生成float数组也不是物理accumulator trace。CPU学习见labs/mlc_vm_stage_contracts.py；设备客户端只在显式CLI启动。

## MLX：lazy evaluation 与 capture 的三个坑

[官方 Metal debugger 指南](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html)提供 capture 工作流。先确认已安装版本存在 `mx.metal.start_capture/stop_capture`，启动前设置 `MTL_CAPTURE_ENABLED=1`；开始 capture 前将输入/权重准备与 lazy work drain，窗口内实际执行目标并 `mx.eval`，最后停止。具体流程已封装在 本地 capture probe，不要复制不匹配 wheel 的 API。

`mx.compile` 也不是记录一次 GPU command buffer 后永久 replay：它会处理 graph、融合与 specialization。[官方编译文档](https://ml-explore.github.io/mlx/build/html/usage/compile.html)说明限制和 shape 行为；应单列第一次 trace/compile 与重复 shape 命中，检查闭包常量、可变状态、shape bucket 和 cache 数量。tensor 第一次 `eval` 可以连带执行此前 lazy 工作；仅量函数返回极易得到虚假“极快”。

本地 [MLX 边界报告](../profiling/mlx-boundaries.md)特意区分带 drain 的诊断和原程序延迟。新 [合成探针](../MAINTENANCE.md)测的是 graph 构造到 `mx.eval` 的 host wall，不是 shader-only。

## PyTorch MPS：开关不是一组通用加速秘方

另一种observer陷阱见[原始llama.cpp实测](case-studies/llama-metal-flash-and-eval-observer.md)：eval callback只取metadata并始终queryfalse，source scheduler仍在split后synchronize。56个完整payload逐位保值只能证明这组结果没变，不是零同步/零计时扰动。原始模型IR的last-FFN39→1和FA输入F16已取证，但IR op/分配buffer标签不当物理dispatch/ISA。

完整请求边界可再看[第十六轮预训练Candle](case-studies/candle-q8-fragments-and-request-cost.md)：原chat encode、IPC、token上传、全模型、全词表D2H/greedy与文本decode都计入ready-model host wall，256份实际timed logits在计时结束后flush验算。packed虽然更快但152个正常输出超精度预算，不给合格倍率；CPU权重核查、model load、first-use、round1、逻辑bytes和实际峰值均分账。此处没有新增物理timeline/PMU证据。

以 [官方 MPS 环境变量表](https://docs.pytorch.org/docs/stable/mps_environment_variables.html)和固定读本 `MPSProfiler.h` 为准：

- `PYTORCH_MPS_TRACE_SIGNPOSTS` / `PYTORCH_MPS_LOG_PROFILE_INFO` 是 bitmask，参考枚举，不随意把所有数字写成 `1` 后声称获得全部 kernel 时间。
- `PYTORCH_MPS_PREFER_METAL=1` 调整相关 matmul 路径偏好，不会强迫所有 MPSGraph op 变成自定义 Metal。
- `PYTORCH_MPS_FAST_MATH=1` 改变数学编译语义；需要另作数值门禁，不能默认与 baseline 完全等价。
- `PYTORCH_ENABLE_MPS_FALLBACK=1` 允许不支持的 op 回 CPU，可能引入转换与同步；没有报错不代表全部在 GPU。
- allocator high/low watermark 是内存/提交策略，不是免费提速；不要为跑分禁用限制、挤占系统内存或把 swap 当省内存成功。

capture API、profiler context 与 `torch.mps.synchronize()` 以实际 torch 版本文档为准。首轮只核对源码/文档；后续[第十五轮原始Torch2.14.1实测](case-studies/pytorch-mps-profiler-and-real-attention.md)复用已有私有环境，补真实BF16 attention、48个observer输出和10份capture：不同op日志可重复整CB时间，printed `cpu`字段并非独立host wall，direct attention无hook而整库符号又可能多达上万。禁止把日志相加/缺项补零或将synthetic parser时间当实测。[PR #189019](optimization-prs.md)的MetalGraph仍按其历史未合并候选记录，不据此提供稳定API用法。

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

## 可复用的最小测量记录

每个性能结论至少能回答：**什么输入、哪份构建、从哪里量到哪里、如何确认完成、怎样检查输出、代价有没有移到窗外。** 建议复用[实验模板](../templates/experiment.md)，记录下列字段：

- 身份：SoC/OS/SDK、运行库与 binary/metallib 哈希、权重/adapter、shape/dtype/layout、cache 冷热与准备状态。
- 边界：host 开始/结束、目标 stream/queue、同步对象、读回/校验在窗内还是窗外；first-use 与 warm 分列。
- 时间来源：host / CB / encoder-stage / marker / shader 分列，附单位、校准和覆盖；缺失写 unknown/null。
- 验证：完整输出门槛、observer off/on 对照、全部样本/配对顺序、scratch/驻留/进程峰值与失败记录。

例如，“一个 CB 内有 52 次 dispatch、stage 为 t”只能说明该聚合边界的时间；不能报告每个 kernel 为 t/52。采用与否最终回到未采集、同完成语义的完整消费者，而不是挑一张更好看的 timeline。
