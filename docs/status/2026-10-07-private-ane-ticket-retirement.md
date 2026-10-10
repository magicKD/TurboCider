# Private ANE Ticket 退役状态修复

日期：2026-10-07，W0 adapter 接线续于 2026-10-08。范围：已有 private 请求的 CPU 状态协议和原生 handoff 基础 API；没有运行 GPU／ANE、没有下载模型，没有更改 W8 算术或默认路由。

## 已修复

原来的 `Ticket::finish` 超时只返回局部失败，未把请求永久退役。之后成功的 driver callback 可以写入成功结果，第二次 `finish` 可能返回 `ok=true`。另一个独立问题是：`evaluateWithModel` 返回 NO 被记作 `done`，但代码本身已承认驱动可能收到请求，不能据此回收或读取其 surface。

本次将逻辑退役与完成证据分开：

- `Completion` 尾部追加 `physical_complete`、`retired`，保留原前三个 aggregate 字段的顺序。此变更要求库与调用方统一重新编译，不承诺旧二进制 ABI 兼容。
- `Ticket::status()` 只取状态，不等待、不退役请求；不能再通过 `finish(0)` 轮询。
- timeout、拒绝提交、提交异常、失败 callback 都永久关闭该结果的成功资格。重复错误不覆盖首次原因；迟到 callback 只能补充完成证据。
- 即使 callback 在提交函数返回前已成功，随后提交函数返回 NO 仍会使该结果退役。
- 未提交的 `PreparedRequest` 析构不会产生 driver 工作；可直接解除其资源引用循环。该路径没有可发布的输出。

这里的 `physical_complete` 精确定义为：在当前 private driver 协议假设下，观察到了该请求的 completion callback；或已知请求根本未提交。它不证明实际 ANE placement，也不是所有 OS／driver 故障模式的认证。对已提交请求，强制提高 event、`!timed_out`、提交被拒绝、CPU worker 返回均不构成该证据。

正常结果还须满足 `ok && physical_complete && !retired`；新 handoff 仍须验证 GPU producer、输出 generation 和 epoch，才能发放设计中的 `ResultReadPermit`。其他已有 `gpu::Transfer`／`QuantStage` 还未迁移此新增状态，追加字段保持保守的 `false`，不能把它们当作已完成新协议。

## 验证

CPU contract 直接使用生产代码调用的 `TicketCompletionState`，覆盖 pending 状态观察、正常完成、timeout 后迟到成功／失败、拒绝后迟到成功／失败、callback 先于提交拒绝、首次诊断保持、重复 callback、未提交销毁、旧 aggregate 调用兼容。

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools \
  bash tools/native/build_private_ane_ticket_state_test.sh
```

结果：exit 0；不链接 Metal 或 private ANE，不创建设备。测试没有注入真实 driver 故障，不构成硬件路径验收。

```sh
DEVELOPER_DIR=/Library/Developer/CommandLineTools xcrun clang++ \
  -std=c++20 -Wall -Wextra -Werror -Wno-deprecated-declarations \
  -fobjc-arc -mmacosx-version-min=26.2 -fsyntax-only \
  native/backends/private/ane_program.mm
```

结果：exit 0。弃用诊断沿用 private 构建现有的 CommonCrypto 兼容选项；完整 native 构建由主任务统一进行。

## 仍未完成的 handoff 工作

本次保留旧 force-signal waiter-release 策略。它不解决 legacy `prepare_transfer` 已预排输出 restore 的完整失败协议，也不证明 CPU failure flag 对任意更深 command graph 都安全。不能据此宣称独立 native GPU＋ANE handoff 已完成。

新路径应先采用 W0：输入／权重 producer 成功完成后再运行请求；成功 callback 和完整语义检查后，才提交输出 restore。失败或超时不得通过提高 event 授权读取 output；请求物理状态未知时仍隔离其 program、event、surface 与 bank，不复用其存储。

同日后续已补 W0 基础桥接口；下表区分已完成的接口与仍需接线的资格检查：

| 接口 | 已有依据 | 尚需补充 |
| --- | --- | --- |
| `SurfaceBufferLease` | `Surface::metal_buffer_lease()` 已导出现有 buffer/device/offset/pitch/extent 和原 Surface owner，含稳定 allocation identity | 上层按原 backing 导入 Native Backend；不得新增 buffer 包装、第二次物理内存计费或把 lease 当作数据就绪证明 |
| `AneEventBinding` | 新 `Program::prepare` overload 已接收 retained 外部 ready/done event、role 与 provenance，保留 native event；新路径失败不 force-signal | W0 adapter 已绑定它自己提交的 producer ticket；仍需独立 native GPU→ANE→GPU 的小张量硬件验收 |
| `PreparedBinding` | `Program::prepare` 已验证命名输入和 Surface | 复用绑定 wrapper；每 job 保持独立请求，绑定 program/function/surface generation |
| W0 `NativeAneHandoff`／`NativeAneReadPermit` | 已串联 producer 的成功物理完成、实际 ANE callback、全部 provenance 检查，成功才授予输出读取许可 | 只实现单控制线程 W0；真实桥接、restore 正确性及缺 callback 故障注入未认证 |
| W1 有限窗口 | `PrivateW8Graph` 已有双权重 bank | 当前只有单个输出和 correction surface；必须扩展独立输出 slot，不能仅因有双 bank 就允许并发请求覆盖 |

新外部绑定同时检查 program cache key、`Program::binding_generation()`、procedure 0、非零 domain／epoch／request／slot generation／producer generation；submit 前要求 ready 精确达到本次值，done 尚未达到本次值。独立 ready/done events 是推荐方式，兼容同一个 event 时还要求 done > ready。元数据和 event 值仍不能证明 GPU producer 成功，须由上层 `ProducerCompletionPermit` 证明 GPU `ok && physically_complete && !retired` 并绑定这组 provenance。

基础桥不复用 one-shot driver request：每次 `prepare` 仍构造独立 request、每个 `PreparedRequest` 只提交一次。参考了本地 Splash `0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7` 的 `Program::bind/enqueue`（`runtime/ane/Program.mm` 673–740）：可复用的是 binding 元数据，事件和 request 属于单次执行；本次尚未引入长期 `PreparedBinding` 缓存。外部事件按相同来源的机制检查 `eventPort`，拒绝 ANE 无法共享的 Metal validation wrapper。

追加验证：`tools/native/build_native_ane_handoff_types_test.sh` 的 CPU contract exit 0，覆盖 retained owner／range／overflow、设备和角色错误、provenance 错误、W0 事件值、独立／共享事件、move-only binding、结果 receipt 全字段匹配和 read permit 的构造边界。`ane_program.mm`、`ane_gpu.mm` 和 `native_ane_handoff.mm` 均以同一 macOS 26.2 严格语法命令通过。没有创建设备或执行 ANE／GPU；W1 窗口、故障注入与硬件共享事件验收仍未认证。

## 2026-10-08 W0 adapter 的实际调用边界

`native/backends/metal_graph/native_ane_handoff.hpp/.mm` 已实现以下路径，没有 MLX 调用：

1. 接收已准备的 Program、完整输入／输出 Surface 与 job identity；导入现有 Metal buffer 并保持 Surface owner／allocation identity。
2. 创建独立且全新的 ready／done events。`submit_producer(graph)` 在该 producer 图末追加自己的 ready signal，并保存准确的 Backend Ticket；不接受任意外来成功 ticket 作为证明。
3. `finish(producer_timeout, ane_timeout)` 主机等待 producer `ok && physically_complete && !timed_out`，再检查 ready/done 和绑定本次 provenance，创建并提交一次 private request。ANE 时间预算从成功输入就绪开始，包含请求准备／提交。
4. ANE callback 必须 `ok && physical_complete && !retired && !timed_out`，其 receipt、program generation、domain／epoch／request／slot／producer generation 和事件值必须全部匹配，才能返回 `NativeAneReadPermit`。
5. 调用方使用 `permit.output(name).view` 单独构造 restore 图。Graph 持有其 BufferView 后，相关 backing 会保活至 restore 实际完成。adapter 不预排读取输出，不 force-signal。

接口使用示意（省略模型、surface 和 PSO 准备）：

```cpp
NativeAneHandoff handoff(backend, program, inputs, outputs,
                        {.domain=1, .epoch=1, .request=1,
                         .slot_generation=1, .producer_generation=1});
CommandGraph producer;
// 用 handoff.input("x").view 等编码全部输入／权重写入。
handoff.submit_producer(std::move(producer));
auto permit = handoff.finish();
CommandGraph restore;
// 将 permit.output("y").view 作为只读输入编码到 restore 中。
auto restored = backend.submit(std::move(restore));
// 必须检查 restored 的成功物理完成，随后才能向模型发布恢复后的结果。
```

GraphStep 尚无完整写集合元数据，因此调用方仍负责 producer 图确实覆盖所有输入／权重，或显式保证未写入部分是已完成的不可变快照。当前 adapter 证明自己的 ticket／event／epoch 关系和生命周期，不证明这一写集合。所有控制调用必须在同一主机线程串行进行；单次对象、fresh events 和单次许可不可解释为 W1 或更深窗口资格。超时结果永久退役，晚到 callback 可以更新实际完成状态，但不得恢复许可。

原有 `PrivateW8Graph::run` 内仍自行 stage、等待、预排 restore 和 overflow retry；整体直接包进 Native Backend 不会自动变成由 native command graph 掌控的执行。overflow retry 在更深窗口应退役当前事务并 GPU 回放，到安全空闲边界再切换 headroom／program，不能改写仍在排队的 binding。

代码入口：`native/backends/private/ane_ticket_state.hpp`、`native/backends/private/ane_program.mm`、`tests/native/private_ane_ticket_state_test.cpp`。窗口资格与输出读取规则继续以 [设计 03](../design/qwen-image-gpu-ane-command-graph/03-command-graph-and-handoff.md) 为准。
