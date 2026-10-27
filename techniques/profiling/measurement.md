# 计时、内存与质量：让比较可解释

<!-- knowledge
id: technique-profiling-measurement
type: guide
stage: pipeline
operator: scheduling
backend: cpu-reference
device: cpu, gpu, ane
symptom: measurement
semantics: contract-only
boundary: request, producer-consumer
api: reference-only
-->

[总目录](../README.md) · [实验模板](../templates/experiment.md)

## 同时记录的时间轴

| 时间 | 边界 | 不等于 |
| --- | --- | --- |
| constructor / prepare | 验证、加载、binding、预热 | 纯推理或compile |
| process wall | executable启动到退出 | 用户点击/整服务排队，除非确实计入 |
| request wall | 规定的encode→denoise→VAE→export | 单个denoise kernel |
| phase/block/operator host | 主机提交到指定完成 | GPU/ANE独占kernel时间 |
| device timestamps | 某command-buffer/device事件 | 全请求；多个queue durations简单相加 |
| exposed wait | caller到达join后等待 | branch alone耗时或所有device work |

异步spans可能重叠/嵌套，不能相加成请求。`pre_ffn`可能包含上游lazy工作；deferred join把GPU消费留到后面，不是工作消失。profiling强同步会改变生产图，因此“定位模式”与“正式性能模式”分开。

当worker的prediction等待减少而output完成等待增加时，先区分event、CB和
encoder边界，不把等待换位置解释成设备计算提速。GPU encoder counter可补充
证据，但应以实际shared-event延迟放行负控，确认采样是否包含此前等待，并
校验counter时钟、数据、scope与完整consumer数量。encoder elapsed仍可含竞争/
抢占/内部停顿，不是独占shader busy time；改变encoder创建方式的诊断也必须
与无observer强基线分开。案例见[Qwen交接消融](../../docs/status/qwen-base512-ffn-handoff-encoder-2026-10-18.md)。

## Cold / warm / cache 状态

至少标出 process-cold、compiled artifact hit/miss、OS file cache未清、model/session状态、conditioning/prefix hit、GPU compile、首prediction和warmup次数。独立进程共享driver/文件cache，不称严格系统冷启动。

Warmup要完整且不混入样本，shape/share切换后的首次编译也要按预定规则处理。fresh prompt与repeat prompt应两组，不能只让候选命中缓存。

## 对照设计

先固定基线/候选、样本数和排除规则；ABBA/BAAB或交错顺序、相同机器/输入/库identity。所有失败、重试、fallback和竞争负载样本保留，不挑最快控制。不要在计时同时build、pytest、比较图像或跑其他自有GPU任务；不擅自暂停用户服务。

速度比为 `baseline/candidate`，耗时下降为 `1-candidate/baseline`；例如1.25×是耗时少20%，不是少25%。报告中位数、样本数、p90/最差值与顺序；小样本p90只是描述性顺序统计，不是置信区间。

同一campaign的pool中位比与matched-pair比中位数是两种统计，不混用；phase的分支中位数也不能拼成不存在的最快请求。跨窗口数字可以作历史参照，不宣称严格因果。

计时区间外也不一定没有观察者效应：在每对短sample之间CPU扫描大输出做checksum，
会改变下一sample的cache、内存与调度状态。先在preflight做完整输入/权重变化
检查，计时循环内保留真实设备完成与成功计数，再在循环后核对各arm独立owned
末输出；若只能逐sample检查，应单列为诊断模式，并另跑无扫描的性能对照。
不要靠移走完成等待或丢弃输出所有权来消除这项干扰。

## 实际执行证据

必须读 actual result/receipt：backend/precision/recipe、实际rows/share/bucket、完成steps/frame count、calls/counter增量、source/adapter身份、fallback/retry/失败数。Session累计计数做差，不把第N个请求读到的总calls当本请求calls。

`plan`只证明可计划；`CPU+NE`只证明策略；两个worker只证明host并发机制；native标记不证明所有projection都走同一个kernel。要声称物理overlap/INT8 MAC，另需合适的device trace。

## 全过程内存观察

采样覆盖load/cold/warm/退出，跟踪root与descendants，记录PID出生identity、防止PID复用；保存间隔/最大gap/能力缺失/结束状态，gap超过策略不伪装complete。

MLX active/cache/peak、external surfaces、native ledger、RSS、physical footprint、system compression/swap分别列；不相加重叠口径。采样max可能漏短峰，结束swap used不变也不证明中途无swap traffic。CoreML外部服务/driver未覆盖就明示未知。

预算hint、opportunity admission、完整认证容量和实际某RAM设备验收是不同层次。不能用64GiB上的8GiBhint声称8GiB设备可用。

## 质量观测与可追溯性

先shape/finite/source，再component/trajectory，最后decoded媒体与人工whole/detail/temporal审阅。PNG bytes不等于decodedRGB；RGB忽略alpha的工具不能冒称RGBA完整。lossy codec后差异不直接归因模型。

原始报告append-only或保留原件，摘要绑定report/binary/input hashes；源码后来变化不重标旧结果。失败gate保留，`allow_approximation`或肉眼近似不能覆写它。更多见 [近似验收](../algorithms/methods/approximations.md)。

## 功率、能耗与持续负载

延迟优化和能耗优化分别衡量：单请求能量约为匹配边界上的 `∫ power(t) dt`，平均功率更低但运行更久不一定省能。先声明是否扣idle、采样覆盖哪些引擎、是否包含CPU/内存/媒体；系统总功率不是ANE独占功率。持续视频还需观察长窗口thermal/throttling，短micro不代表分钟级稳定吞吐。需要额外采集权限时先授权，不自动改频率/供电/用户进程来改善数字。

## 仓库好范例

- [2026-10-15合并对照](../../docs/status/2026-10-15-main-into-dev.md)：相同显式steps、fresh条件、ABBA、实际calls、before/after RGBA与全过程内存；区分旧外部框架倍率和新分支收益。
- [SeedVR2 Comfy对照](../../docs/benchmark/SEEDVR2_COMFYUI.md)：actual dtype日志、MPS补丁、完整帧/FPS、speed pass但overall资格仍false。
- [LTX sparse](../../docs/design/ltx-stage2-sparse-results.md)：core/stage/request以及质量负结果不混称。

## 实现参考与实验

可读实现：参考代码；[实验步骤与边界](../examples/README.md)；本项全部 case / controls。

本项控制：interval union vs sum of nested spans；power vs total energy and matched input pairs。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：new timestamp source, boundary, sample plan or energy instrumentation。历史负结果限定于其来源/shape/边界，不关闭该方向。
