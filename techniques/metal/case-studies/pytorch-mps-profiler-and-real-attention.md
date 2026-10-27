# PyTorch MPS真实Qwen attention与profiler：完整输出、重复CB时间和kernel清单陷阱

<!-- knowledge
id: technique-metal-case-studies-pytorch-mps-profiler-and-real-attention
type: case-study
stage: prefill, decode, dit, pipeline
operator: attention, layout, scheduling
backend: metal, mpsgraph, cpu-reference
device: cpu, gpu
symptom: measurement, launch-bound, memory-bound, graph-interruption, precision-loss
semantics: numerically-equivalent, bitwise, contract-only
representation: bf16, fp16, fp32
boundary: cpu-gpu, producer-consumer, request
api: public, private, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [SDPA分流/布局/cache失败](pytorch-mps-sdpa-dispatch-and-layout.md) · [来源](../sources.md#pytorch-mps-sources-20261017) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第十五轮消费者/profiling部分。用已有两个真实Qwen请求的原BF16 Q/K/V重放原始Torch2.14.1 MPS SDPA，240个完整输出全部对FP64与原reference检查。诊断又显示：**MPS profiler每个操作的`gpu`字段可能重复整段command buffer时间；gputrace的kernel清单可能来自整库符号，不是实际dispatch清单。** 两者不能相互替代，也不能相加得到物理kernel耗时。

## 1. 是真实激活重放，不是新跑了完整模型

来源为[已有Qwen attention捕获](qwen-attention-replay-and-empty-row.md)中两个seed42/43、不同prompt的source-stable ordinary GPU block replay。客户端先核对历史receipt允许集、runtime/build身份、完整tensor SHA与原始BF16，再消费Q/K/V；不重新运行产品、下载checkpoint或改LoRA/dtype。原fixture是暴露输出的同配方replay，不是对未改变kernel内部的直接抓取。

prefill为32 heads、D128、1048 Q/KV，其中24行prefix单独causal，后1024行读完整KV。steady为1024 Q/1048 KV，使用实际prefix cache语义，全量attention。没有用整张`is_causal=True`替换原来的分段结构。

两个seed×default/禁prefill四进程；每phase三种同值bank：owning BHTD、带非零offset的token-major view、末维stride2 view。每臂first/warmup/8个hot，共240个完整`[1,L,4096]`输出，每份约419万或429万元素，全部读回检查，不只抽几行。

独立CPU按完整heads/rows执行FP64 QK、softmax、PV及相同输出layout；它用128行Q tile限制临时CPU存储，**不是抽样**。所有输出通过前页预置BF16门槛；相对FP64的L2约0.00140–0.00146。另对原BF16 reference的完整L2约`7.65e-6–5.61e-5`，但均非bitwise相同，不能写成跨框架逐位等价。

### 完成边界与有限的性能观察

计时为ready resident Q/K/V→原SDPA/分段concat→连续`[B,L,H*D]`→`torch.mps.synchronize()`；不含bank构造/初次上传、CPU验证/读回、checkpoint加载或输出projection。stride2所需的内部contiguous准备仍在计时内。

下表为hot完成host wall中位数ms，单格seed42/43；原始样本全保留，没有删除较慢seed。配置除分流开关外固定fallback0、fast-math0、prefer-metal0；“default”仅指prefill selector未禁用。

| 模式 / bank | prefill1048（含分段） | steady Q1024/K1048 |
| --- | --- | --- |
|default / head|3.578 /3.356|2.434 /3.244|
|graph / head|6.067 /6.106|5.889 /5.928|
|default / token-view|3.479 /3.335|2.280 /3.265|
|graph / token-view|6.065 /6.145|5.904 /5.937|
|default / stride2|4.131 /4.017|2.527 /3.798|
|graph / stride2|6.225 /6.306|6.021 /6.078|

本组default head相对graph head的**跨进程中位数之比**约prefill1.70–1.82×、steady1.83–2.42×；不是同trial切换分流开关的paired ratio。布局臂在每进程中轮换正反顺序，但selector开关在进程初始化前固定。启动时三次低利用率快照不证明连续独占，较慢seed的原因未归因给未测的温度/频率或其他PID。

源码general路径显式repeat GQA KV、建立完整attention输出；custom路径的融合/online softmax是值得借鉴的机制。实际本组没有独立DRAM/PMU证据，不能将上述host倍率全部归因为省带宽、MPP/NAX物理指令或每个kernel变快。三bank均从ready输入开始，也不意味着修改真实producer布局免费。

更重要的缺项：QKV投影、RoPE、output projection、residual、FFN、完整denoise/文本请求没有在新Torch消费者中运行。它不是模型迁移、图片质量、tok/s或Torch/MLX引擎排名。

## 2. MPS profiler的字段从哪里来

按固定[MPSProfiler.h](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/mps/MPSProfiler.h)、[MPSProfiler.mm](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/mps/MPSProfiler.mm)和[公开Python包装](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/mps/profiler.py)读到：

- `PYTORCH_MPS_LOG_PROFILE_INFO=399`是1/2/4/8/128/256组合：operation/copy/fallback info、all stats、GPU time及单列scheduling字段；没有打开buffer地址标识位。
- operation完成handler从**所在CB**取`GPUEndTime-GPUStartTime`。另一字段取`kernelEndTime-kernelStartTime`，虽然日志文字打印成`cpu=...`，它不是另一次CPU host计时。
- 多个instrumented op的handler可以挂在同一个CB，分别累计同一段CB耗时；这些“每op GPU时间”不能直接相加成无重复物理时间。
- stats按graph/PSO handle聚合，`strKey`每次begin都会重写。因此最后显示的一组shape并不代表该handle的全部历史运行都用了那组shape。
- `wait_until_completed=True`在**有hook的边界**提交并等待；不能据文档短句推断每个未instrumented dispatch都强制拆开。

实际各一个default/graph新进程，调用两次原SDPA加square/add/连续输出；before、signpost-no-wait、signpost-wait、after各3次，合计48份完整结果对FP64全部通过，同case跨context bytes相同。记录的是context/API+stderr观察，**没有录制/打开Instruments signpost trace**。

default日志中连续的copy、square、copy、add各打印相同`gpu=0.432ms,cpu=0.052ms`，与整CB时间重复归属一致；不能相加称为1.728ms GPU kernel时间。signpost-wait后square/add可显示约0.004ms，表示执行/归属边界已改变，不是原先未观察程序中测得该kernel只有这个耗时。

该读本custom Attention.mm的直接encode没有调用`beginProfileKernel/endProfileKernel`；general MPSGraph路径通过graph cache/run获得hook。本轮default长Q的stderr看不到prefill名称，但能看到下游copy/square/add；切到D33则出现`sdpa_general_mps`。**没有日志条目不是没有GPU执行**，`There are no CPU Fallbacks logged`也只覆盖已挂hook的fallback记录，不授权整个用户程序绝无CPU工作。

日志开启本身已改变环境，profile实验的before/after也不是未profile正式latency基线。default长Q完成wall的诊断中位数约0.595ms（no-wait）对1.019ms（wait）；D33约0.356ms对1.433ms。小型单进程诊断不提供全局observer开销百分比。

## 3. 有dispatch记录，仍可能没有可信kernel时间

使用公开`torch.mps.profiler.metal_capture`，MTL_CAPTURE_ENABLED只在自己的子进程设置。API会给文件名加序号前缀，客户端只在新输出目录切cwd，未改全局工作目录。固定C++ stop路径先drain stream再停止capture，异常也保留stop；没有故意制造teardown race。

default/graph各五个独立窗口：Q1/K1023、Q1/K1024、Q9/K1031、D33以及全空行mask。30个before/captured/after完整输出全部通过且各case逐位稳定。capture body包含**SDPA、token-major输出准备、验证float cast和D2H**，因此下面不是裸SDPA的kernel计数：

| 窗口 | default parser dispatch记录 | 禁prefill记录 |
| --- | --- | --- |
|Q1/K1023|2|2|
|Q1/K1024|3|3|
|Q9/K1031|3|11|
|D33|11|11|
|empty-inf|3|11|

这与源码vector阈值/禁prefill范围及graph展开方向相符，但parser未独立关联每次dispatch到真实PSO，**分支名仍是source-supported inference，不是物理指令确认**。

复用固定`tmc/gputrace@3d7d32e...`的只读stats/timing，不运行GUI、replay或系统权限变更。10份stats的unique-kernels约16653–21521，远大于2–11条dispatch；timing同样为上万行且标注`synthetic/approximate`。Torch整库metallib的名称提取不是执行清单，不能说“这个小attention运行了两万个kernel”，也不能把合成累计秒数与毫秒host wall比较。

严格normalizer保持`physical_gpu_duration_seconds=null`、`formal_performance_score=false`。buffer-usage=0也没有变成零allocation结论。MPS profiler的CB来源时间与parser的synthetic时间分别保留，不互相洗成更强证据。

## 4. 固定源码与wheel的身份不能省略

18个源码/许可文件逐一固定commit/blob SHA，参考checkout保持干净。安装包报告同一Git commit，但Prefill/Mpp头文件**完整bytes并不相同**：包增加稳定API宏guard。独立拆除准确的已知首尾包络后，两份完整body均与commit blob逐字节相同；公开Python profiler文件直接相同。保留raw SHA差异，不随便忽略diff，也不把这当作所有native对象可复现构建证明。二进制dylib SHA另登记。

2026-10-17重取的PR API中，[#198564](https://github.com/pytorch/pytorch/pull/198564)为open/unmerged；[#182256](https://github.com/pytorch/pytorch/pull/182256)和[#191362](https://github.com/pytorch/pytorch/pull/191362)为closed但`merged=false`。与此同时固定源码树中确有MPP和capture-drain实现。两层证据并存：**不以closed或非空merge_commit_sha推断GitHub已merged，也不因PR状态就否认独立读到的代码存在。** #182256仅选读MPP/host/stride/scale patch，不是完整PR审计或新提交的上游贡献。

本轮只复用已有私有Torch环境，无包安装、上游构建、生产默认或系统策略修改。新reference约23MiB；raw capture因bundled库较大留ignored目录，结束时约28GiB余量，无需清用户cache/模型。

## 5. 可重放入口和未完成队列

CPU学习见labs/pytorch_mps_contracts.py，case `pytorch-mps-profiler-contracts`；归档器只用标准库/显式可选NumPy读取数据，不加载Torch或Metal。设备客户端显式`--mode real --seed 42/43 --capture-root /verified/capsule`；`--mode capture/profile`必须另开诊断进程，不混入正式latency。普通单元测试不启动GPU。

总账为12个成功正式进程/4642全输出全部通过（合成4324、真实240、capture30、profile48），外加六个mask-cache缩减进程/8个已完成输出和两次native abort。更早的一次原生abort、两次busy拒绝、ImportError和两个成功pilot也保留，但不重复计数。

source/wheel/binary、OS/SDK/SoC、dtype/布局/mask、cache身份、PSO聚合、profiler/capture/parser版本或真实consumer改变均重开。后续继续完整projection/block/request、更多原始checkpoint/LoRA、真实峰值内存/并发/失败恢复以及可信硬件timeline/counter；本轮结果不关闭整体GPU/Metal调研。
