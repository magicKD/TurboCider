# Splash：单调事件退役、校准模型与失败整块重跑

<!-- knowledge
id: technique-gpu-ane-case-studies-splash-handoff-calibration-and-replay
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [W8A8/布局](splash-w8a8-prefill-and-layout.md) · [来源](../sources.md#splash-ffn)

2026-10-17，同一固定 ref `80827e1…` 的完整 selected Handoff、split、startup/measurement 与实际模型完成闭包已读。本轮原始 CPU test 通过，不代表这里的设备 handoff、private client 或故障恢复已在本机运行。

## 等待被放行不等于工作成功，更不等于 drain

Apple 官方 [signaledValue](https://developer.apple.com/documentation/metal/mtlsharedevent/signaledvalue)说明 shared event 只接受更大值；[encodeWaitForEvent](https://developer.apple.com/documentation/metal/mtlcommandbuffer/encodewaitforevent(_:value:))在事件已达到目标时不再等待。这是以下退役设计的公共基础，不是“ANE 一定完成了”的信号。

[`Handoff.mm`](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/ane/Handoff.mm)为所有层分配递增 ready/done，值不按 command 清零。GPU pack 后 signal ready，ANE request 等待 ready、写 partial、signal done，GPU complement/next-weight-stage 后 wait done 再 join。所有 jobs 在 command submit 前接受到 private client，代码依赖其带 shared events 的异步接受语义。

job 有 queued/started/succeeded/failed；只有首次 report 生效。ready 通知开始其 bound，失败、start 抛错或超时使 handoff 永久 retired，立即把事件升到已分配最大值；以后的值也立即放行，且不再接受新 agent job。已接受请求仍可能因事件放行而继续计算无用数据。这是为了不让 GPU 等到自身 timeout，不是返回成功的捷径。

| 状态 | GPU wait | 当前输出 | outstanding/idle |
| --- | --- | --- | --- |
| 正常 done + 成功 report | 可通过 | 等整体 judgment 后使用 | 对应 report 已收到 |
| retire 强行提高事件值 | 可通过 | 失败 command 必须丢弃 | 仍可能有 agent 工作 |
| finish 清除本 command 的 jobs | 不等价于新同步 | 取决于成功/失败 | outstanding 独立保留 |
| cancel/析构等待 bound 到期 | 已放行 | 不能当完成数据 | 未必 idle/drained |

迟到 report 只归还 outstanding；不复活已失败结果。通知晚于 report 时观测 duration 可为零，所以 `ane_ms` 是 host 观察 ready 通知到 report 的累计，不是 ANE core 的精确硬件 busy time。

本轮的四个逻辑 witness显示：超时后 wait 已满足但 outputs_usable=false/idle=false；finish 后仍未 drain；late report 才清空 outstanding；duplicate 不改变首次失败。这是 CPU 模型演示，不是原始 Objective-C++ Handoff、真实 Metal 信号或线程 liveness 实测。原始 `handoff_test.mm` 的 fake-agent Metal fault suite 仅阅读，未执行。

## 三种 timeout 与私有接口假设

该实现用 2 秒 evaluation bound、120 秒 compile/load 限额、10 秒 idle unload/reload 限额。它们属于不同阶段，不能说“所有失败都在两秒内完全回收”。clock 选择旨在按 awake 时间计限；本轮未做 sleep/wake 验证。注释中的具体设备 watchdog/编译时长是作者观察，非通用系统承诺。

[`Program.mm`](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/ane/Program.mm#L209)的 service queue 工作可在 caller timeout/interrupted 后继续，state 由排队工作持有，末次释放也回到该 queue。单串行 queue 的前一项挂住，后项等待也会用掉限额；timeout 不是取消服务端运算。

completion wrapper 先记录早到 callback，只有 enqueue 确认接受后才调用上层 done；enqueue 若抛错便不交付那个 report。binding 保持 surfaces，private request 接手资源；本轮没验证不同 OS 下服务端的实际 retain/drain。事件退役后的安全还依赖失败输出被丢弃、资源不被当成可重用 scratch。

另一个边界：evaluation 的私有接受调用是在 caller 上发起，源码假定它会返回/抛错；它不是取消任意阻塞私有消息的通用 watchdog。ABI 检查类、selector 与 arm64 type encoding，能拒绝一类不兼容，不证明未来 OS 的异步/ownership 语义没变。这是源码边界，不是本机 hang 复现。

共享事件还必须有可分享的 Mach port；该 ref 显式拒绝 validation-layer wrapper。不要自动关闭用户全局验证来“修好”研究样例。本轮没有加载 private framework，也未改系统/验证/缓存设置。

它不是[公共 Metal4 ML pass](metal4-machine-learning-passes.md)或普通 Core ML delegate：公共 event 是同步基础，执行入口仍是 private `_ANEClient`。二者的可移植性、placement 与失败合同分别验证。

## 失败在真实消费者中怎样结束

[`Runtime::prefillAsync` 完成闭包](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/model/Runtime.cpp#L2071)在 `AneFfn::finish()` 判失败后：恢复各 lane 的 RNG draws，用同一 items 重编码整个 chunk、GPU-only 再执行，把 rerun 的 GPU/wall 时间计入 chunk，增加 rerun counter。state parity、length、selected token、image rows 的提交在 judgment 后。

因此不是只补算一个坏 partial，不能先把第一次采样 token 或 KV/state length 对外提交再随便重试。源代码同时说明 image rows/state/KV revision 如何沿用，未据此宣称本机已验证所有取消/缓存/多请求交错。

性能 breaker 不同：它在一条数值可用 command 后 retire，`finish()` 仍返回 true，所以保留当前结果，仅让后续 chunk GPU-only、忘记校准。不能把所有 retire 都当需要重跑，也不能吞掉 backend unhealthy、用户取消和 service Interrupted 等应向上传播的错误。

## 校准不是取 isolated ANE/GPU 时间的最大值

原始数学模型为 `T(s)=max(A(s),G(s)+u*A(s))`：G 由两个 share 的 GPU 部分计时拟合；A 通过 high-share split 拟成正比；u 根据 low-share 的 concurrent slowdown 估计并 clamp 到 [0,1]。high share 为 ANE-dominated 是经验假设，不是设备定理。

[`Measurement`](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/ops/AneFfnMeasurement.cpp)走真实 stage/handoff/join，以 GPU FFN 作为 mixer stand-in，再减去同法计时的 stand-in；差分层数抵消提交/首层 stage 固定成本。使用跨深度样本、median-of-three，不是完整请求或完整真实 attention 链。derived A/G/u 也不能标成各设备独立硬件计时。

选择 unit 时最小化“该 unit 及左右一 unit 的预测最坏值”，而非直接四舍五入连续最优点；预测收益不足 5% 则不用。内存 plan 再限制最多 ANE units；先做可行性算术，再进入重编译/计时。GPU 优化、带宽竞争或 share 量化都会改变最优解，不存在通用 50/50。

每个函数 bucket 单独测，再找所有更大 chunk 都允许 split 的连续 suffix。minimumRows 可在 bucket 内，例如合成 fixture 得 553；允许该函数耗时至多比该逻辑 chunk 的 GPU 预测慢 2%，不是“必须快 2%”。这两个百分比是本实现策略，不是置信区间或硬件规律。

## live breaker、cache 与内存是不同合同

breaker 每 8 commands 比累计 **ANE observed durations** 与校准的 GPU-only layer 成本；到达/超过才停。它是保守的一侧警报，不是持续成对重测整图：GPU 部分后来变慢但 ANE 未超过旧基线，也可能不触发。手动给 share 不做自动性能选择/缓存/breaker，但仍要过格式、内存、接口和数值 verify。

calibration key 包含层 shapes/formats、device、macOS、build 和 feasible unit cap，不是 weight-content fingerprint；prepare 仍验证当前 split。Program 的编译 key 则来自 MIL/constant blob 的 FNV-1a，和本库追溯 SHA-256 不是一回事。shape-cost cache、程序资产 cache、模型内容身份不要混为一谈。

自动 context 为了跨启动稳定，可能按已校准/假定 split 的 capacity 宣告，即使该次 setup 失败退回 GPU；显式禁用 ANE 或 remembered no-gain 是另一路。因此公平比较应固定 max-context、KV precision、完整内存/冷启动范围，不能拿不同自动容量的默认值直接排名。

idle release 卸载 ANE program 的服务端工作区，但保留该类 surfaces/scratch；stopped 且 outstanding 尚未全 report 时不会 unload。reload 不重新 compile，并核对 procedure 描述；失败保持 GPU。Metal allocations/RSS/服务内存、编译磁盘缓存、wired 与仍在运行的请求不是同一个数。

## 本轮实际执行了什么

未改的 `AneFfnCalibration.cpp` + 上游 calibration test 编译后通过（11 个 source named groups，不是 11 次硬件 benchmark）。另一个原创 observer链接同一原始数学代码，输出 5 组合成模型、183 个 cost 点和 183 个 cap 选择，独立 Python 复算；minimumRows 为 512/553/none/2008，breaker 的 8-command 边界亦核对。

两程序的声明依赖仅 libc++/libSystem，源码无 Metal/ANE 执行；四个 build/run 进程均有 terminal receipt。原 tests 中的 M5/M6 等设备标签只是作者数值 fixture，本轮不继承为硬件测量或产品存在/支持证明。

离线复核（不编译、不联网、不触碰设备）：

完整原始 CPU 测试的复编译命令/输入 hashes 和两份 Apple 文档字节保存在[来源清单](../sources.md#splash-ffn)。本轮没有 GPU/ANE/model、private-client availability、真实故障/取消/drain、跨请求所有权或 physical overlap 资格。
