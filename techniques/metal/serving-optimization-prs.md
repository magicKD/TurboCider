# Serving 优化 PR 深读：混合批次、压缩 KV 和采样边界

<!-- knowledge
id: technique-metal-serving-optimization-prs
type: method
stage: prefill, decode
operator: attention, sampling, scheduling
backend: metal, mlx, cpu-reference
device: cpu, gpu
symptom: measurement, memory-bound, lifetime
semantics: contract-only
boundary: producer-consumer
api: public, reference-only
-->

[GPU 导航](README.md) · [vLLM runtime](vllm-metal-paged-runtime.md) · [第二轮固定 diff](sources.md#deepdives-sources-20261016)

2026-10-16 核对下列四个 `vllm-project/vllm-metal` PR 的 GitHub pull 详情（含 `merged_at`）、head/base/merge SHA 与完整 diff hash；选读关联实现与关键 patch 区段。**未在本机运行这些 PR 的服务性能测试**，原始 PR 与当前固定读本的后续演进分开。

| PR | 本轮状态 | 机制 | 阅读时最重要的边界 |
| --- | --- | --- | --- |
| [#851](https://github.com/vllm-project/vllm-metal/pull/851) | 2026-09-28 merged | mixed prefill batch 分离长 decode prefix | 原 patch 的功能范围不能替代当前版本更宽的 NAX/GQA 路由 |
| [#853](https://github.com/vllm-project/vllm-metal/pull/853) | 2026-09-29 merged | bounded TurboQuant gather/decode→FP16/BF16 prefill | 临时 workspace 与每 layer 强同步也是成本；M1–M4 rollout 不是默认全开 |
| [#649](https://github.com/vllm-project/vllm-metal/pull/649) | 2026-08-31 merged | MLX-native temperature/top-k/top-p 采样留在 lazy graph | RNG stream/seed/penalty/logprob/constraints 合同不同，不只比较吞吐 |
| [#611](https://github.com/vllm-project/vllm-metal/pull/611) | 2026-08-15 merged | one-step-ahead pipeline 默认启用 | 主要是 rollout；状态/实现还要回读 #588 后的源码与 executor gate |

## 从 #851 学习：优化调度形状，而不是重写所有 kernel

长 decode row 落在 prefill tile 会浪费 Q 行并失去 KV 分区并行；patch 把 decode prefix 与 prefill suffix 分开、维护 Q-block offset。作者同时给局部 kernel 与 serving 结果，两种幅度并不相同。应学习其分层测量，不继承 M2 Max 上的倍率为 M4/M5 资格。短 context、spec、压缩 KV 与 NAX 的初始排除条件需要按版本看，不能拿旧 PR body 覆盖当前 dispatcher。

## 从 #853 学习：省内存的策略也会限制并发

压缩驻留不排斥临时解码：足够多新 query 可摊薄按页 gather/解码的成本。patch 联合写最终 FP16/BF16 K/V，去重共享物理页，并把 workspace 在 KV 池 sizing 前预留。每层 `eval+synchronize` 保证临时对真正释放后再复用预算；只看物化 kernel 时间会漏掉这条边界。

短 suffix/prefix-hit、滑窗、verification、dtype/head/格式或 workspace 不满足时保留 compressed fallback。当前读本后来还增加了长历史 D128 的门槛校准；原 PR 的 `max(128,D/2,ceil(256*KVheads/Qheads))` 不是所有后续版本唯一规则。

## 从 #649 学习：采样可以成为 GPU 推理的主瓶颈

作者把 top-k/top-p mask 与 categorical sampling 留在 MLX graph，让合格 non-greedy decode 继续采用延迟同步；避免 MLX→Torch 的采样 round-trip。这个改变不是单纯换一个等价 deterministic kernel：RNG stream 改变，带独立 seed、penalty、logprob 或 token 约束的请求另有回退/拒绝合同。相同随机分布、固定 seed 可重现、相同逐 token 输出是三种不同要求。

## 从 #611 学习：默认启用需要比微基准更强的证据

该 PR 主要翻转 env 默认值；它依赖更早实现中的 pending-row 覆盖、uniproc executor 检查和真实模型里程。读到默认为 1 不代表任意 batch 都 pipelined；正确方法是先查 gate/实际 route，再用相同生成设置 kill-switch A/B。模型名和 README 倍率都不能替代路由和消费者证据。

可执行学习控制：CPU paged/预算反例、[owned graph](../examples/reference/contracts.py)；case `paged-runtime-contracts` / `owned-command-graph`。这些代码不运行上游 patch，不授予服务或模型资格。
