# Splash：command-buffer 容量、完成门与超时所有权

<!-- knowledge
id: technique-gpu-ane-case-studies-splash-command-capacity-and-watchdog
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [FFN布局](splash-w8a8-prefill-and-layout.md) · [handoff/重跑](splash-handoff-calibration-and-replay.md) · [来源](../sources.md#splash-command-lifecycle)

核查日：2026-10-17。固定 `incoai/splash` commit `80827e1c3593f7d10c266394e4d4381321b73ad0`、tree `af79421149362f80fc140a2342c9f0cae61da200`；现有 checkout 是 fork，官方对象身份沿用[第十五轮核对](../sources.md#splash-ffn)，不声称最新 main/release。本页新增的是原始 host 构建、原始 watchdog header 的 CPU 执行及运行时合同深读，**不是原始 Metal handoff 测试通过**。

## 核心机制：事件切分也有提交前容量成本

`MetalBackend.mm:140–167,792–898,1070–1077` 将一个 runtime command 切成多份 Metal command buffer：起初创建一个，每个 `Signal` 编码结束后保存当前 buffer 并新建下一个；`Wait` 不新增 buffer。所有 buffer 创建、编码结束后，才由 `commitSubmission:334–343` 顺序 commit。

因此，对这一特定策略：

```text
command_buffers = signal_steps + 1
首次 commit 前必须容纳全部 command_buffers
```

这不是 Metal 的普遍“每 signal 都必须另建 buffer”规则，而是此 ref 为故障放行所选的编码策略。Apple 的 [默认队列文档](https://developer.apple.com/documentation/metal/mtldevice/makecommandqueue())与 [commandBuffer 文档](https://developer.apple.com/documentation/metal/mtlcommandqueue/makecommandbuffer())分别说明默认上限64、无空闲 buffer 时创建调用阻塞 CPU，直到 GPU 完成一个。

| signal 数 | 所需 CB | 假设队列64 | 本 ref 队列512 |
| --- | --- | --- | --- |
| 0 | 1 | 能容纳 | 能容纳 |
| 8 | 9 | 能容纳 | 能容纳 |
| 63 | 64 | 刚好 | 能容纳 |
| 64 | 65 | 首次 commit 前容量不足 | 能容纳 |
| 511 | 512 | 容量不足 | 刚好 |
| 512 | 513 | 不适用 | `checkEvents` 在编码前拒绝 |

`newCommandQueueWithMaxCommandBufferCount:512` 配合 `signals >= 512` 拒绝，使被接纳 command 的需求不超过队列容量。若只把队列恢复默认64、保留其余 pre-create-all 策略，64层/每层一个 GPU signal 就可能在第65次创建时卡住，而此时一个都未提交，GPU 无法靠完成它们腾位。这是**官方合同加源码的条件性推导**，本轮未故意制造设备挂起。8层原始测试即使未来通过，也不等于覆盖64层边界；更大 queue 也不消除内存不足、驱动失败或其他依赖问题。

学习迁移：增加 event/fence 不只是增加 wait 成本，还会改变未提交对象数。预算应在提交前包含真实的拆分数；不能只按逻辑 layer 数或“一个 runtime ticket”估算。另一个设计方向是有界增量提交，但需重新证明故障收尾、owner 与跨 buffer 依赖，不能直接当作等价改动。本轮未修改上游实现。

## 为什么前序 buffer 出错仍要补 signal

`MetalBackend.mm:822–838` 为每个带 signal 的前序 buffer 注册 completion handler；若其状态为 Error，由 CPU 补发该 signal。否则后续 GPU 等待 agent 的 done，而 agent 又等待未发出的 ready，容易形成闭环。

补 signal 只是让等待链能够结束，不会把错误变成功。最终 `finishCommand:434–466` 检查全部 leading buffers，采用第一个失败 buffer 的错误；不能只看最后一个 buffer 为 Completed。`finish:469–491` 保存错误并标 backend unhealthy。与此不同，CPU fake-agent 失败的 handoff 可永久退役并转 GPU-only，同时 Metal backend 保持健康——两类故障不可混为一谈。

`timing.gpuSeconds` 是首个 leading buffer 的 GPU start 到最后一个的 GPU end；可能包含中途 event 等待，不等于各 kernel 的纯计算时间之和。source wall clock 使用 `AwakeClock`/`CLOCK_UPTIME_RAW`；本轮没有 sleep/wake 或时钟准确性实验。

## GPU完成、结果发布、ticket消费是三个边界

源头 API 的 `MetalBackend.hpp:272–275,345–351,369–377` 规定一个 submitted-but-not-applied command。实现中：

| 阶段 | watchdog | active gate | ticket 的 allocation 保留义务 | 下一次 submit |
| --- | --- | --- | --- | --- |
| commit 后、GPU未完成 | 计时 | 占用 | 保留 | 拒绝 |
| completion 发布，ticket未消费 | 停止 | 仍占用 | 仍保留 | 拒绝 |
| 正常 wait 消费并 release | 停止 | 清除 | 清除该列表 | 健康时允许 |
| GPU未完成即超时 | backend unhealthy | 不提前清除 | completion handler 继续保活 | 拒绝 |

这是源码合同，不是本轮设备状态观测；“清除列表”也不表示所有 view/Metal 对象立刻释放物理内存。

`completeSubmission:356–359` 只停 watchdog；`releaseSubmission:346–354` 才清 active sequence、command 和 completion。`State::finish` 以 first-result-wins 发布并 notify；notify 在 ticket 锁外执行，职责是唤醒 host control flow，不是从 callback 直接递归提交下一块。即便 `ready()` 为真，仍需消费 ticket。

正常 `wait:1028–1045` 先取回结果，再 `release()`，之后才返回或抛 stored GPU error。析构/覆盖赋值走 abandon 等待路径。上轮模型 fallback 要先处理旧 ticket、丢弃失败结果再 GPU 整块重跑，不能看到设备 callback 就认为当前运行时 command 已应用完毕。

## watchdog 不是 GPU cancel，也不是 free 权限

`commandAbandoned:361–400` 在 deadline 后区分两种情况：

- 最后一个 command 已为 Completed/Error，但 callback 延迟：取出 completion，在 gate 锁外补 finish；终态 Error 仍会被记录，终态不是无条件成功。
- command 仍未终态：标 unhealthy，等待者可以放弃，禁止新工作；**没有取消 GPU，也不把被访问 allocation 释放或复用**。

先在 gate 锁内拷贝 recovery callback、再退出锁执行，可避免与 `finish` 持 ticket 锁后进入 `completeSubmission` 的锁顺序冲突。first-result-wins 还防止 recovery 与迟到驱动 callback 重复发布。这里只分析所读分支，不宣称全部并发竞态已验证。

`awaitCompletion:514–537` 每一秒检查 watchdog/shutdown。未完成而放弃时返回 false；`wait` 先 reset 自己的 state 引用再抛异常，避免异常展开再次等待。`MetalBackend.mm:883–897` 的 command completion block 强持 ticketState，使 retained allocations 在 GPU 结束前仍有 owner；`abandon:551–556` 的注释也明确这一点。底层如果永不回调，保留资源可能持续到进程退出；不能用“timeout已返回”冒充 drain 或可安全回收。host timeout 与进程 supervisor 的杀进程边界同样不能证明设备 drain。

`releaseMemory:1213–1223` 还会拒绝 wrapped-owner 内存和仍有 in-flight command 的释放；共享 no-copy buffer 的 Metal deallocator 自身也捕获外部 owner。统一内存、取消、析构和 owner 是不同维度。

## 实际运行了什么

原始 header observer直接 include 未改 `CommandWatchdog.hpp`，仅链接 libc++/libSystem。输入是假定的 CPU 时间，不等待真实 deadline，也不加载 Metal：

1. 未启动：不超时；seq1 在10启动、10.5未超时、11已超时。
2. `complete(99)` 不清除 seq1；`complete(1)` 清除。
3. seq2 在20启动；迟到的 `complete(1)` 不清除 seq2，21超时；`complete(2)` 清除。
4. timeout 为0、负数、NaN、Inf 的4种输入均抛 invalid_argument。

保存状态为 `[false,false,true,true,false,true,false]`。这些结果实际来自 native executable，只有该纯CPU header获得执行资格；不能外推到 driver callback、取消、内存释放或真实 handoff。

独立CPU见证另复算7个容量点与简化的发布/消费/超时状态，分发数据可离线重算。它不是原始 backend，也没有线程、事件、锁或驱动；未模拟 abandoned 后迟到 callback 的完整清理，不能用“模型通过”替代原实现测试。

## 原始测试为何停在 host build

实际完成：原始 `MetalBackend.mm`、`DeviceCapabilities.cpp`、`Handoff.mm`、`handoff_test.mm` 四个 TU 通过 C++/Objective-C++20、ARC、O2、Wall/Wextra/Werror、最低macOS26.4编译并链接。使用本库最小 CMake wrapper，不是原仓库根 Makefile 的整引擎构建；编译器 `.d` 记录的14个参考文件逐一核 committed blob/hash，字节核验不算全文人工阅读。没有改原始源码，也没有启动生成的 handoff executable。

该原始 executable 需要一个真实 metallib。原 loader 从文件调用 `newLibraryWithData`，且构造时就需要 `residency_kick`；测试库应由原始 `metal_backend_test.metal` 与 `shared/residency.metal` 编译、链接。本机只读 preflight 中 `xcrun --find metal` 和 `metallib` 都失败，SDK/Clang存在不代表离线 Metal 组件存在。

Apple [组件文档](https://developer.apple.com/documentation/xcode/downloading-and-installing-additional-xcode-components)提供 Metal Toolchain 的安装/export/import方式；本轮只阅读，没有执行安装、切换 xcode-select 或系统设置。SDK26.5 的 `MTLDynamicLibrary` 序列化描述动态依赖代码；`MTLLibrary.h:329–335` 明确与含 kernel entry 的 executable library 不同，不能将其文件冒充所需测试库。改 loader 为 source compilation 可形成另一项实验，但不能仍称“未改原始测试”。

原 suite 的源码范围：8层、bound100ms/slack250ms、command watchdog1s；7种行为×故障位置0/3/7，共21配置，每配置两条8层command，另测单调event、计时、cancel/析构/late-report。agent 是CPU线程，不是私有ANE服务。原 `dev/native.mk` 以进程级 `MTL_SHADER_VALIDATION=1` 运行；本轮未运行此环境或启用任何全局验证层。

后续有匹配工具链时，应先构建来源匹配的测试库，再用本库 GPU guard 的连续5次一秒占用采样（均低于5%，缺计数拒绝），在有界 supervisor 下串行运行并保存完整输出。需要另加64层/容量测试才能覆盖本页边界；即使原 suite 通过，也不证明真实ANE服务的接受/生命周期、模型质量、请求恢复或物理GPU+ANE overlap。本轮不自动安装组件，不将这个局部实验阻碍记成整个研究目标 blocked。

## 重放与追溯

原始构建/运行/预检日志、binary、Apple正文留 ignored `outputs/gpu-ane-splash-handoff-round16-20261017`；保留五项terminal receipts、log/binary/source hashes、动态依赖与未运行字段。[来源JSON](../sources.md#splash-command-lifecycle)分开17个选定读本、14个compiler依赖、三份Apple正文、三个SDK header范围。核心报告不分发上游源码/编译库。

固定源码入口：[容量/构造](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/metal/MetalBackend.mm#L1070)、[watchdog/状态](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/metal/MetalBackend.mm#L320)、[原始CPU header](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/metal/CommandWatchdog.hpp)、[原始handoff test](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/dev/tests/engine/handoff_test.mm)、[原始测试构建](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/dev/native.mk#L593)。范围及离线验证见[第十六轮](../../MAINTENANCE.md)。
