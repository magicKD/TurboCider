# Command graph、资源存活与完成证据

<!-- knowledge
id: technique-metal-command-graphs
type: method
stage: pipeline
operator: scheduling
backend: metal, mlx
device: gpu, ane
symptom: launch-bound, lifetime
semantics: contract-only
boundary: producer-consumer, gpu-ane
api: public, private
-->

[总目录](../README.md) · [Splash 对照](../references/splash.md)

## 怎样借鉴：先证明瓶颈在提交层

2026-10-17 方法整理。若短算子之间有明显 host 编码/提交间隙，先保持原数学 kernel，对比逐次提交与受控批量提交；若 GPU 本来就持续执行长 GEMM，graph/replay 不能直接提高该 GEMM 的算术吞吐。第一步通常是[稳定 block 编译](mlx-compile-cache-and-state.md)，独立 native graph 则适合确实需要更细资源/事件控制的场景。

建议按成本逐层推进：固定 block 的 graph 编译 → 合并不必要的 eval/同步与提交 → 为稳定拓扑研究 replay/ICB → 有明确需求后再迁移 [Metal 4 执行 API](tensorops-and-m5.md)。这是采用顺序建议，不要求框架内部恰好按这个顺序实现。源码可从 [tinygrad Metal runtime](https://github.com/tinygrad/tinygrad/blob/3d057f05df532deca7661bfb2400c6564f96a04a/tinygrad/runtime/ops_metal.py)追到 ICB/资源绑定；它的未文档化 compiler service 与公共执行 API 应分开评估。

每一步固定同一输入、输出和 kernel，分别量准备/首用、host 编码、提交到完成、完整请求与峰值，并测试新输入/新权重、地址复用、取消和 timeout。不要给不支持拓扑变化的 replay 喂新 shape；不要用 GPU 时间下降掩盖 host 准备增长。

两个已经取证的边界可直接借鉴：[默认 stream 同步不等待 foreign stream](case-studies/mlx-benchmark-evaluation-and-stream-completion.md)；[Metal 4 end recording 不等于完成/allocator 可 reset](case-studies/metal4-counter-heaps-and-time-domains.md)。因此“少同步”应建立在正确依赖和 completion 证据上，不是删除所有等待。

## 有界异步：把每步验证也纳入依赖

逐步同步改成有界异步时，保留原求值与精度边界，让每一步的状态和独立 finite mask 一起进入 request-owned pending 集合；不能只检查最后一步，因为较早的 Inf 可能在后续变为有限值。drain 必须等待所有实际依赖、检查全部 mask，之后才发布完成进度，奇数尾步也不能遗漏。取消/异常退出同样保住这些 owner，直到已提交工作得到完成证据；没有设备故障实验就不承诺任意 hang/error 可恢复。

验证至少包括普通/融合算子对照、先非有限后恢复、取消时仍有 pending、无提前进度/导出、同 Session 重试和恢复原策略。队列容量限定的是 retained step 数，latent＋mask 的逻辑字节不是整个图或 driver 的内存预算。策略改变若影响编译 recipe 或 prefix 语义，应使对应 bank 重新 prime。

“下一步构图时先前结果尚未 host-ready”只是 host-ahead 观察，不证明物理重叠，更不自动证明加速。[本地 Qwen512 双步实验](../../docs/status/qwen512-async-step-pair-2026-10-18.md)在保持原始数组和图像一致后仍只有微弱整请求变化，未采用为默认。可迁移的教训是先测同一完成边界的净收益；长 GEMM 主导时，减少少量 host 等待未必值得更复杂的生命周期。

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

可读实现：参考代码；[实验步骤与边界](../examples/README.md)；本项全部 case / controls。

本项控制：per-dispatch vs one command buffer；late completion and overwrite refusal。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：deeper segment or different completion/cancellation contract。历史负结果限定于其来源/shape/边界，不关闭该方向。
