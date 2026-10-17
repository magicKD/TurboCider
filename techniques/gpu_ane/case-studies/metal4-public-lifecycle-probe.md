# 公共 Metal 4：从源码生命周期合同到 M4 小型执行验证

<!-- knowledge
id: technique-gpu-ane-case-studies-metal4-public-lifecycle-probe
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [Uzu 实现](uzu-metal4-and-routing.md) · 数值与计时数据 · [失败/拒绝历史（过程原件已移出）](../../MAINTENANCE.md)

2026-10-16，M4 Max、macOS 26.6.2/25G83、SDK 26.5、Apple clang 21.0.0。这里运行的是本库原创小客户端，不是 Uzu、PMetal 或模型请求。结论是**公共 Metal 4 command API 在本机可执行且这组显式生命周期安排通过检查；没有稳定性能优势或 M5 MXU 证明**。

## 1. 公共合同与上游设计怎样对应

对照 Apple [核心 API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api)、[资源同步](https://developer.apple.com/documentation/metal/resource-synchronization) 和本机 SDK 声明，来源身份/阅读范围保存在 [公共来源清单](../sources.md#metal4-public)。

| 对象/操作 | 必须保持的合同 | 这次怎样处理 |
| --- | --- | --- |
| command buffer | Metal 4 buffer 不默认强持有资源；可在 commit 后重新编码 | probe 更保守：等 terminal feedback 后才复用同一 slot |
| command allocator | `endCommandBuffer` 后可服务后续编码；`reset` 则要求所有相关执行完成 | 每次反馈和 CPU 验证之后才进入下一代 reset |
| argument table | 提供地址/ID 绑定；compute dispatch 时快照绑定 | 四个固定 binding，数据 buffer 地址带 64-byte 偏移 |
| residency | 资源驻留不是生产完成或引用保活 | 四个 buffer 入 set 并 commit；每次 begin 后重新 `useResidencySet` |
| dispatch 依赖 | Metal 4 不因资源的 tracked 配置自动处理 hazard | producer→consumer 中插 dispatch→dispatch、device visibility barrier |
| feedback | 读取错误并保留 owner，不能只收到通知便发布成功 | 强引用整个 slot；错误不计成功；成功后独立遍历验证 |

因此不能把“allocator 可再次用于编码”和“allocator 的旧命令存储可以 reset”合成一个许可，也不能把 binding 快照理解成 backing 自动保活。probe 选择串行等待，是为了验证最小合同，不是声称 Metal 4 必须一提交一等待。

这与 Uzu 的 callback 保活/terminal 后回收有共同设计动机，但不是执行了它的 Rust runtime。真实 pool 还需追踪所有 command buffers、alias、generation、取消与错误 drain。

## 2. 同一 shader/pipeline 的两条提交路径

shader 与 Objective-C++ 客户端 使用同一份运行时编译的 library 和两个 pipeline：

```text
temporary[i] = input[i] * 2 + phase
output[i] = temporary[(i + offset) % N] * 0.5 - 3
```

跨位置消费使大 N 的依赖不限于同一线程组。输入为可精确表示的小型二进制有理数，每轮改变 phase；独立 CPU oracle 检查全部 input 未改、temporary、output，以及三个 buffer 的前后各 16 个哨兵。**不是只抽查 checksum**；checksum 另由 Python 重算。

N 为 1、257、65539，包含不足一个 threadgroup 和非整齐尾部。传统 encoder 显式 `memoryBarrierWithScope`，Metal 4 使用 encoder barrier；不运行“删掉 barrier 看是否碰巧正确”的未定义并发负例。

两进程 seed1/2；每个尺寸每个 API 2 次预热、7 次正式，顺序按轮反转。总计 108 个计时记录，84 个正式记录，另有 4 个保活用例。计时只包含已准备资源上的 host 编码→commit→完成通知，初始化、分配、编译和结果验证在外。

## 3. 数值与保活实测

两个成功进程合计完整验证 **2,500,284 个输出元素**，同等数量的 input/temporary 元素，以及 10,752 个哨兵元素，全部 mismatch=0、max abs=0。这只是该有限精确数值域，不是任意 FP32/模型误差资格。

保活用例使用默认反馈队列，在 feedback handler 内暂缓处理。调用方释放 slot 并排空局部 autorelease pool；弱引用观察到 owner 仍在，析构数为 0。开放 gate、完成输出检查并释放 callback owner 后，弱引用变空，析构数为 1；257/65539 两种尺寸、两个进程都通过。

该用例证明**我们的显式 callback owner**在受控时点持有并释放资源；不证明 runtime 自动保活，也不证明调用方丢 owner 时 GPU 一定还在计算。feedback 可能已经表示设备完成，只是 CPU 处理被延后。错误 GPU command、取消、多 slot 并发、提前 reset 和物理 drain 没有注入验证。

另有两次安全的 shader `#error` 编译拒绝：得到 `MTLLibraryErrorDomain/code3`，没有提交该错误网络。它不等于 GPU 执行失败恢复测试。

## 4. 性能：这个负载上接近，不能授予通用加速结论

下表是每组 7 个正式样本的 host-total 中位数，单位 µs；包含通知/唤醒，不是纯 kernel time。

| seed / N | 传统提交 | Metal 4 |
| --- | ---: | ---: |
| 1 / 1 | 146.666 | 144.583 |
| 1 / 257 | 147.291 | 145.917 |
| 1 / 65539 | 160.833 | 161.667 |
| 2 / 1 | 129.542 | 129.250 |
| 2 / 257 | 129.875 | 128.917 |
| 2 / 65539 | 145.625 | 145.875 |

host encode 的组中位约 2.3–4.8 µs；feedback GPU duration 约 6–13 µs。可见这个小客户端的大头是提交至通知边界，不应把 total 当 GPU 算力。JSON 另保留逐对差值统计；**中位数之差不是差值的中位数**。

各尺寸的逐对差值范围都跨零，部分正向尾部超过 100 µs。7 次/组和两进程不足以判断细小收益；新 API 的结构性潜力需要更多 dispatch、批量 commit、CPU 多线程编码与完整消费者对照。

两进程 pipeline build 约 37.05/4.55 ms，不计入 warm 表。没有清系统缓存，所以“新进程”也不是严格 cold-cache。没有测能耗、内存峰值或 request。

## 5. 不隐藏失败与准入限制

最初 sandbox 无法读取 IORegistry，guard 在启动任何设备 child 前拒绝；获准后重试。首次自定义 serial feedback-queue 版本则在 `lifetimeCase` 的释放阶段 SIGTRAP；系统诊断是 libdispatch 的 `API MISUSE: Over-release of an object`。native 没有输出最终 JSON，该 child 不计为通过。

其根因尚未缩减，不能归咎于 Apple runtime、Uzu 或某个模型，也不能说“只改队列便修好了所有生命周期问题”。后续使用**默认队列 + handler gate 的不同变体**通过；失败源码 hash、构建/receipt hash 与筛选后的诊断保留在历史文件，原始代码/日志留 ignored outputs。

两次成功运行启动前各有五次占用采样 `[0,0,0,0,0]`。during 也读到 0，说明采样可能漏掉短工作；它不能证明无外部 burst、频率稳定或任何物理 overlap。失败 child 已退出，但没有观察它的物理设备 drain；后续不用其资源，并重新准入。

## 6. 复现与离线检查

驱动/整理器 默认只编译 host，不获取 Metal device。macOS 26+ 的相应公共 SDK 为依赖，无上游 checkout 或第三方 Python 包。

输出目录必须未存在。若 guard 忙，先继续技术工作，之后新目录重试；不自动终止其他程序。离线测试中的 synthetic timings 只验证 receipt 拒绝规则，不混入硬件测量。

本轮另一条公共入口见 [Metal 4 machine-learning pass](metal4-machine-learning-passes.md)：它有不同的模型编译/放置合同，不能由本页两个普通 compute kernel 自动推导 ANE 使用。
