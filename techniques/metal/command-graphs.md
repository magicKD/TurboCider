# Command graph、资源存活与完成证据

[总目录](../README.md) · [Splash 对照](../references/splash.md)

## 什么时候值得用独立 native graph

需要精确控制实际 buffers、dispatch batching、跨 GPU/ANE ready/done events、可取消的 bounded segments 或完整 storage liveness 时。若只是减少 MLX 小算子调度，先试 compiled block；独立 graph 同时意味着维护完整算子、dtype、scheduler 和模型 parity。

TurboCider 的 [backend.hpp](../../native/backends/metal_graph/backend.hpp) 与 [backend.mm](../../native/backends/metal_graph/backend.mm) 是真实实现；设计说明中的伪代码不是现成 API。已有 [Qwen native 后端](../../native/models/qwen21/native_denoiser.cpp) 保持显式实验。

## 必须一起设计的五件事

1. Dispatch 记录拥有参数 bytes 和 buffer views，不借用栈上的临时参数。组装完整 operator 后原子 append，验证失败不污染已有 graph。
2. Command buffer 足够大以摊销编码/提交，但要有 segment、最大 in-flight、watchdog 和取消检查。不能为了少一次 submit 让整模型不可响应。
3. 每个 storage 有 actual allocated bytes、owner、last consumer、external readers 和 reservation。bridge 要接现有 memory ledger，不新建一套“看起来更低”的私有计数。
4. Ticket/Completion 区分提交、逻辑失败、timeout 和物理完成。timeout 不自动证明 GPU 已停，也不是释放 buffer 的许可。
5. Public/Private ANE handoff 明确支持什么 event。不能因接口名字相同就给 Public 同步 prediction 添上不存在的设备 event。

## 完成与回收状态机

```text
prepared → submitted → in_flight → physically_complete → validated → publish
                         └→ timeout/error → quarantine/drain → proven-safe release
```

timeout 后即使为了退出设备等待而 signal external event，也不能把未完成 ANE 输出当有效、复用 surface 或退还预算。异常栈退出同样必须保住 owner；late completion 只能回收它实际拥有的 generation，不能回收新请求的同地址 allocation。

## 不要过度依赖指针

Buffer 的 allocation identity 是 ownership control block，而不是 `void*`。缓存 key 还需 source generation / immutable 内容、offset、pitch、slice、recipe。allocator 可以重用地址，raw pointer hit 不能证明还是同一权重。

MLX bridge 要保住 allocator allocation owner，光 retain 一个 Metal object 不一定阻止 MLX cache 重用对应存储。外部 queue 的 writes 必须在 MLX consumer 前有被支持的完成边界。

## Profiling 口径

GPU first-start 到 last-end 包含同 graph 内的空隙，不是所有 kernels durations 的简单和；host 编码、queue wait、device execute 和 full request 要分开。Graph 优化后的 end-to-end 对照还要计 encoder/VAE，以及 graph preparation 和 cache miss。

[native SGMMA 完整请求](../../docs/status/2026-10-08-native-lora-sgmma-e2e.md)报告 tracked native allocation 归零，同时保留约 15 GB retained baseline；“账本归零”不是整个进程内存归零。这类清晰边界应复制到新后端。

## 迁移检查

从小型已知算子开始，测试 append 失败原子性、非法 view、非有限值、tail、丢弃 ticket、错误 callback、取消、timeout、late completion、跨请求 generation 和真实 teardown。数值/生命周期通过后，才扩大到一个完整 block、两个 dependent blocks、完整 DiT/Euler。

## 实现参考与实验

可读实现：[参考代码](../labs/metal/bench.mm)；[实验步骤与边界](../labs/gpu.md)；[本项全部 case / controls](../labs/catalog.json)。

```sh
python3 -B techniques/labs/run.py slot-ownership
```

本项控制：per-dispatch vs one command buffer；late completion and overwrite refusal。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：deeper segment or different completion/cancellation contract。历史负结果限定于其来源/shape/边界，不关闭该方向。
