# ANE Profiling：工具分层、指标来源与可重放 SOP

<!-- knowledge
id: technique-ane-profiling
type: method
operator: scheduling
backend: coreml, private-ane, cpu-reference
device: cpu, ane
symptom: measurement
semantics: contract-only
boundary: producer-consumer
api: reference-only
-->

[Core AI原Swift组件实测](coreai-profiling-reset-and-backpressure.md)在明确依赖适配器下验证end后异步统计、reset后旧写入、缺测/零值与noncopyable所有权；816事件/216完整consumer数组，不是Core AI模型、物理ANE时间或Instruments trace。

[Core AI Swift profiler源码纵切](coreai-blocked-attention-and-runtime-gates.md#5-swift引擎与profiling对象和时间分别取证)补充host Mach span、显式end/析构、异步StatsStorage与缺失category默认0的边界；仅固定源码阅读，不是本机Core AI/ANE计时，需沿实际strategy确认token分母。

[ANE 目录](README.md) · [通用测量](../profiling/measurement.md) · [本轮实验](experiments-20261016.md)

## 每个工具实际回答什么

| 工具 | 能回答的问题 | 证据性质/盲区 | 本轮状态 |
| --- | --- | --- | --- |
| Xcode Core ML performance report / Instruments Core ML | load/prediction、模型瓶颈、运行时活动；配合 Neural Engine instrument | 官方诊断；具体可见轨道与 OS/Xcode 有关，不是完整公开 ANE 指令级 profiler | 本机只有 CLT，`xcrun --find xctrace` 失败，未采 trace |
| MLComputePlan / MLModelStructure | 哪些 op 可/首选哪个设备、相对成本、结构 | **计划/估计**，不等于真实逐 op latency 或持续 occupancy | 原创客户端已跑；小/大图的计划分配不同 |
| ORT session profile / ExecuTorch Core ML SDK profiler | 前者记录软件EP kernel边界；后者按模型窗口×plan cost生成op区间 | 软件派发、计划估计、物理时间三者分开；preferred设备与ns格式不是硬件trace | [ORT实际分区/回退](onnxruntime-coreml-partitions-and-cache.md)；[ET估计公式源码/CPU控制](executorch-coreml-aot-and-estimates.md)，未跑完整ET |
| anemll-profile | 合并 plan、编译日志、非 ANE 岛、预测计时，导 JSON | measured/derived/predicted 必须拆开；默认边界惩罚为启发式 | 固定源码已审，原工具有清 cache/全局日志副作用，未原样执行 |
| anemll-bench | 模型 warmup/预测时间、单/双模型对照 | weight bytes/time 是**有效权重吞吐**，不是硬件 DRAM counter | 本轮深读计时/带宽/helper，未下载模型运行 |
| Orion core profiler / CLI bench | caller eval/total wall、decode样本、调用方FLOPs和RSS；CLI阶段统计 | utilization为软件span比、TFLOPS派生；cap4096忽略后续decode，CLI forward/backward是固定40/60估计 | [原native profiler12个注入控制与CLI源码](orion-rewrites-and-profiler-metric-provenance.md)，非实测模型/ANE；zero/inf、未喂分母、当前RSS与peak分开 |
| Apple ANE Transformers原test/rough_timeit | ALL与CPU_AND_GPU许可下的完整predict host wall | 13次call含字典/NumPy准备，10×sanity常量不等于测得物理ANE/现代设备保证；低于时只log | [源码及原图数值验证](apple-transformer-norm-and-position.md)，未运行其benchmark；本批小图均CPU首选，无速度资格 |
| whisper.cpp阶段timer / WhisperKit timings与signpost | encoder/cross、mel/decoder/KV与循环/流水分母 | `t_encode_us`含mel准备和cross投影；Kit loops/fullPipeline不是去重输出token吞吐，host异步span不是物理ANE时间 | [固定源码与选定原声明验证](speech-stages-profiling-and-failure-boundaries.md)，未跑ASR性能；零/unknown保留inf/nan |
| ane_pmu_profiler / measure_ane_capacity | 私有 stats、DMA/stall/cycle 等研究线索 | 私有 ABI、缺失/单位/聚合/生命周期须验证 | 已有81次完整数值调用；当前所测路径未取得 PMU |
| powermetrics / asitop | 系统功耗、频率/热状态等采样上下文 | 粗粒度、系统级；权限/字段随版本变；不是逐 op 归因 | 本轮未提权运行 |
| mactop | IOReport/SMC 等系统上下文、ANE/DRAM 历史 | 有些指标是计数、有些是估算，须逐字段辨别 | 旧入口已迁移；本轮 README 核对，不运行控制功能 |
| 自有 signpost / Perfetto | request、prepare、submit、wait、restore/join、取消的关联 | 只能展示已采事件，不能凭空生成 ANE 设备执行轨道 | 已有 host 合同；不把 host overlap 写为设备并行 |
| Metal System Trace / gputrace | GPU stager/suffix/join 是否暴露、GPU 队列瓶颈 | **GPU** 工具，不是 ANE counter | 复用 [GPU profiling](../profiling/gpu-timeline-and-counters.md)、[trace 归一化](../profiling/trace-normalization.md) |

官方一手资料：[WWDC24 模型性能分析](https://developer.apple.com/videos/play/wwdc2024/10161/)、[MLComputePlan](https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n)、[WWDC22 Core ML Instruments/Performance 教程](https://developer.apple.com/videos/play/wwdc2022/10027/)。计划 API 与工具可见性要在实际 SDK/设备确认，不能只因文档存在就声称本机 trace 成功。

## anemll-profile：保留结构诊断，修正证据标签

固定 [anemll_profile.m@bf852810](https://github.com/Anemll/anemll-profile/blob/bf8528105a0e61fda2c80d2c9b19615ecab9b7fa/anemll_profile.m)的 [本库审计](../sources/repos/anemll-profile-safety.md)记录：采整个 Espresso subsystem 日志并开启私有日志显示，删除工具 e5bundlecache 以触发日志重发；后续 timed calls 未逐次处理 NSError，所读 dummy 输入构造未明确初始化确定值。因此本轮不用原工具直接给分。

可迁移的是机器可读诊断：model/graph 身份、plan 中的非 ANE 岛、相对 cost、实际 prediction wall。但缺失条目应 unknown；固定“切换开销”仅为候选排序模型；用 wall 与估计时长倍数判断 fallback 也不是物理测量。应把全局日志/删缓存改为显式独立诊断模式，并对每次成功与输入做 gate，再执行上游工具。用户允许清理任务缓存不等于需要清应用/系统缓存来研究性能。

## anemll-bench：公式与硬件映射要审代码

固定 [benchmark.py@865e78bf](https://github.com/Anemll/anemll-bench/blob/865e78bf8199188be612809356dc90f78b925957/anemll_bench/benchmark.py)的 `_calculate_throughput` 是 `model_size_GB / prediction_seconds`。压缩字节/解压字节、冷 DRAM/片上复用、输入输出与中间 traffic 都可能不同；不能直接命名成“ANE 总线带宽”。双模型实验还需实际并发完成边界，不能简单把两个 isolated 比率相加。

所读 benchmark 会捕获预测异常并可能只对成功的前若干次算平均，因此收集完成不代表所要求 N 次全部成功或数值通过；“低于5ms很可能 ANE”也只是启发式。其 [ane_verification.py](https://github.com/Anemll/anemll-bench/blob/865e78bf8199188be612809356dc90f78b925957/anemll_bench/utils/ane_verification.py)用型号字符串推 ANE 核数，把若干 Max 写成32，不能用作真实硬件规格；M4 Pro/Max 官方是16-core ANE。报告上传是可选外部写操作，本轮未触发。

## 私有 PMU：缺失不是零，CLI 成功不是采集成功

[ane_pmu_profiler](https://github.com/freedomtan/ane_pmu_profiler/tree/20080a945096eb5932490b024120b513ae5427e2)的确提供 stats/loader/编译路径研究入口；具体 ref 以 [原始来源卡](../sources/repos/ane-pmu-provenance.md)为准。不能仅按 README 的寄存器名字相信每项单位。

本库 [81次 PMU 实验](case-studies/pmu-payload-and-missing-counters.md)中推理输出全部通过，但 mask2/15 路径没有 stats object，后期自有 stats buffer 为零；capacity CLI 某路径根本未进入 PMU。故保持 unavailable/null，而非“0 stall”“达到峰值”或“M4 不支持 PMU”。原 CLI 缺模型仍 exit0 的假阳性也保留。

只有 scope、per-request/累计、有效位宽、reset/overflow、采样单位、core 聚合和 elapsed 都匹配后，才可计算 bytes/s、GHz、occupancy 或能量。完整合同见 [PMU lab](../profiling/ane-pmu-lab.md)。没有权限/数据时不关闭 SIP/AMFI，不为完成图表编造 counter。

## 系统采样和能耗

固定 [asitop utils](https://github.com/tlkh/asitop/blob/74ebe2cbc23d5b1eec874aebb1b9bacfe0e670cd/asitop/utils.py)启动 sudo powermetrics，并清 `/tmp/asitop_powermetrics*`；不要在共享机器上不经审查直接运行。其默认 sampler 和字段也不能自动覆盖新 OS 的 ANE occupancy。

路线图 `context-labs/mactop` 的 [旧 README](https://github.com/context-labs/mactop/blob/eaa2fab31481d10bc7544466434eb74a2e2318c3/README.md)已指向 [metaspartan/mactop](https://github.com/metaspartan/mactop/tree/8dbfaaed7426cff3bcfbcb7951a2e15dfb6d1667)。所读新 README 区分 counter/histogram 与 power-derived bandwidth，且有 SMC fan-control/kill 等状态改变功能；本报告只将其当监测入口，不采用未审的型号推断、不启用控制。界面显示 `GB/s` 本身不能说明采到了硬件字节。

能耗需说明 rail/package/system、采样周期、idle/background 和积分区间。用1000ms采样归因一个100µs kernel不成立；可重复长循环以积分平均，并与无 profiler 的正式计时分开。whole-package energy 不能直接叫 ANE 芯片能量。

## 值得复用的profiling输出格式

先确定字段的含义，再做漂亮的时间线；建议每条记录至少带上下面的信息。字段名为本页建议，不是要求上游工具已经采用同一JSON schema。

| 信息 | 最小内容 | 用途 |
| --- | --- | --- |
| 身份 | source/model/weight/input generation、SoC/OS/SDK、shape/dtype | 防止不同图、参数或任务混成配对样本 |
| scope与完成 | request或prediction或op；begin/end及ready边界 | 区分host提交、实际完成和下游可读 |
| 时间与来源 | 时钟域、单位、wall/plan-estimate/counter/derived | 防止把相对cost或parent窗口当逐kernel时间 |
| availability与覆盖 | valid/missing/unsupported、成功/失败次数、未知项 | 缺失保留null/unknown，不补0或只报成功前缀 |
| 正确性与成本 | 完整consumer数值gate、prepare/restore、baseline、峰值内存 | 避免“图快、整条请求慢”或错误结果更快 |

先用[MLComputePlan与官方Instruments方法](https://developer.apple.com/videos/play/wwdc2024/10161/)定位候选图岛，再做独立无profiler对照。现有项目中可借鉴[anemll-profile的机器可读诊断](../sources/repos/anemll-profile-safety.md)，但需保留其启发式与副作用边界；[Core AI原Swift实验](coreai-profiling-reset-and-backpressure.md)则说明统计写入可见/reset协议也需要独立验收。

## 可重放的诊断顺序

1. 无 profiler 建立同语义强 GPU 和候选，检查每次输出；记录 compile/load/first/warm、实际准备与输出完成。
2. 读取 MLComputePlan，保留 CPU/ANE/unknown 和相对 cost；先判定是不是根本没选 ANE。
3. 用 Xcode/Instruments 对**自有进程**诊断 load、prediction、GPU stager/join 和设备活动；记录 OS/SDK/tool 与采集扰动。
4. 对照 unsupported islands、shape、layout、Q/DQ、权重驻留，提出单因素图变体；用独立无 profiler 配对计时决定是否采用。
5. 需要微架构解释才探索 PMU；先已知工作量/流量缩放与生命周期控制，数据缺失时仍可依赖严格 wall-time 优化。

所有时序标时钟域；Python perf_counter、GPU timestamp、外部采样的 ns 不自动可叠加。host call/window 相交可以说明依赖机会，物理并发需另外证据。默认只操作自有合成模型，不订阅/发布全局私有日志或用户输入。
