# LLM decode：权重 / KV 带宽、小 M 与精确验证

<!-- knowledge
id: technique-models-llm-decode
type: guide
stage: decode
operator: gemv, attention, sampling, quantization
backend: cpu-reference, metal
device: cpu, gpu, ane
symptom: memory-bound, launch-bound, precision-loss
semantics: numerically-equivalent, approximate
representation: q4, q8, packed
boundary: producer-consumer, request
api: public, reference-only
-->

[阶段地图](stage-map.md) · 本地概率与 KV 样本 · [GPU 数据流](../designs/gpu-dataflow.md)

## 与 prefill 分开优化

单请求每步一个新 token，projection M 常很小。大 GEMM tile 可能空转，权重与历史 KV 的读取、launch、logits/sampling 往返更显著。Continuous batching 可让多请求共同提高 M，但增加排队延迟、不同长度 mask 和 cache 管理；吞吐与单请求 latency 要分别报告。

首先尝试 format-specific packed GEMV/QMM、融合 norm/activation/residual、少 command submissions、KV 的顺序读与 head layout。避免每步把完整 logits/KV 同步回 host；sampler 与 stop/EOS 规则必须保持实际语义、精度和 RNG 合同。

## KV 和权重的取舍

GQA/MQA 的 KV sharing 由原模型定义，不能直接修改训练架构；paged allocation 减少碎片但有页表和间接访问；prefix sharing 需只读 owner 与 copy-on-write；sliding-window 只对原图规定窗口或明确近似预算成立，丢旧 KV 不能冒充同语义优化。

W4/W8 或 KV quant 降流量，也增加 scales/decode/误差成本。必须在 target kernel 中测实际 packed consumer；先全量解码再运算可能增大 scratch 和流量。上下文很长时，对 logits/task 和长程引用分别门禁，不能仅看一层相对 L2。

小 M 的 ANE 固定 handoff/row padding 税可能不合算；只有完整 layer/request 的匹配测量证明盈利时才启用。不要把 prefill 测到的 share 和 rows threshold 原封不动套到 decode。

## Speculative decoding：完整思想，不等于少验几个 token

Draft 给出候选分布 q，target 给出 p。对从 q 采样的 token i，用 `min(1,p_i/q_i)`接受；拒绝时从归一化 `max(p-q,0)`重新采样。单 token 的最终概率是：

```text
accepted_i = min(p_i,q_i)
rejection_mass = 1 - sum(accepted)
final_i = accepted_i + rejection_mass * normalized(max(p_i-q_i,0)) = p_i
```

speculative_distribution直接积分全部离散分支验证这个恒等式，并覆盖 p=q、draft 零支持和不合法概率。它给出可学习的核心机制，**不是已经实现的完整 speculative engine**。

多 token 实现还需要 target 在正确前缀条件下并行验证、首拒绝处截断、丢弃被拒绝 token 的 KV、处理 bonus token，并保持 sampler/penalty/temperature 对应分布与独立随机流。Greedy verify 是另一种确定性合同。无条件接受 draft 会把输出分布换成 q；只实现 proposal 没有 rejection 修正不是同分布加速。

## 验证与适用范围

分别测 time-to-first-token、steady-state inter-token latency、tokens/s、长上下文峰值、排队 latency、采样质量。Draft acceptance 率必须配上 draft/verify/KV rollback 成本；不是接受率越高就一定更快。扩散模型没有同一个 causal token 分布合同，不能直接拿 speculative token 方法替代 denoising steps。

重新探索：batch、KV 长度/布局、bitwidth、draft/target、sampler、SDK 或设备变化。本文所有核心公式和样本在本目录内，不需要其他模型仓库。
