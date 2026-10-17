# LLM prefill：大矩阵、全局因果边界和 FFN 并行

<!-- knowledge
id: technique-models-llm-prefill
type: guide
stage: prefill, ffn
operator: attention, gemm, rope, swiglu
backend: cpu-reference, metal
device: cpu, gpu, ane
symptom: memory-bound, wrong-layout
semantics: numerically-equivalent
boundary: producer-consumer, gpu-ane
api: public, reference-only
-->

[阶段地图](stage-map.md) · 本地 GQA/KV 实现 · [Hybrid FFN](../designs/hybrid-ffn.md)

## 工作量与主要接缝

Prefill 一次处理多个新 tokens：projection 的 M 是有效 tokens×batch，通常比 decode 大，权重可跨 rows 复用。attention 需要按真实 causal mask 访问历史与本 chunk keys。长 prompt 的 score 中间数组、KV 容量与 FFN activation 峰值都可能成为约束；先测实际 shape，不用“LLM”一词推断瓶颈。

投影采用 tiled/packed GEMM；QK norm/RoPE 融合只改变数据流，不改变位置与 dtype；online softmax 保留完整 allowed key set。GQA 用多个 Q heads 共享较少 KV heads，不需要在物理内存把 KV 复制成 Q head 数；这是利用已有架构，不是擅自把 MHA 模型改成 GQA。

## Chunked prefill 的正确算法

设之前缓存长度 P，本 chunk 有 C 个 query，第 i 个 query 的全局位置为 `P+i`，允许的 key 满足 `key_position<=P+i`。本 chunk 的 K/V 可以先追加到 cache，再依此 mask；RoPE 使用全局 position，不在每个 chunk 重置为 0。

`CausalKV.append`先验证 source/adapter/position-recipe identity、start==当前长度、head 数、维度和 finite，再建立新 KV、计算 attention，最后一次性发布完整 append。它保留 GQA 的 KV head 数，没有做 repeat。`llm_prefill`把 full prefill、3/4/2 分块、5-token prefill+四次 decode 逐项对比，并提供“第二块仍用 query offset=0”的错误样本。

Chunking 降低单次 activation/score 临时峰值，但保留全部历史时不会消除最终 KV；过小 chunk 增加 launch、weight reread 和 padding 税。对每个候选同时看 prefill latency、峰值、后续 decode latency，而不是只报告一个 chunk。

## Causal prefix cache 的成立条件

标准因果堆叠中，prefix 的状态不依赖未来 suffix，因此可以跨请求复用所有层的 prefix KV；要求 prefix tokens、position base/rope、attention policy、checkpoint/adapters、cache dtype/scale、layer count 等一致。编辑 prefix 中一个 token 会影响其后所有层/位置，必须回退到最后一个被证明一致的 prefix。

多模态 embedding、双向窗口、prefix-LM mask、动态 rope scaling 或会随全序列长度变化的规则可能破坏简单证明。缓存所有权和请求隔离也要成立，不能通过共享可写 KV 让一个请求污染另一个。

## GPU/ANE 并行方案

本地 [SwiGLU 切分实现](../examples/split_ffn_reference.py)展示 F-channel prefix/suffix 各过完整非线性、down partial 相加、完整 LoRA 修正。层内 FFN 的 X 必须来自已经完成的 attention；ANE FFN 不能凭“存在 GPU attention”就提前跑同一层。可重叠的是独立 suffix/prefix 和合法的下一层 W staging。

选择最小可容纳 actual rows 的 function/bucket，对完整 pack→fork→restore/join 建 rows threshold 与 runtime breaker。短 prompt/最后小 chunk 可直接 GPU。双 W bank、A8 slot、source owner 和失败整块回退详见 [吸收设计](../designs/hybrid-ffn.md)，不需要查 Splash 源码。

## 内存与验收

连续 KV 的逻辑字节近似 `2×layers×batch×T×KV_heads×head_dim×bytes_per_element`，再加 scale/页表/对齐。Paged KV 主要减少分配碎片和支持共享，不自动让相同 token 的 KV 字节消失。KV quant 是近似，独立做长上下文/任务误差门禁。

必测 chunk=1、非整除尾、长短混 batch、非零全局位置、source/adapter 切换、head 映射、finite、失败 cache 不变和 decode 连续性。重新探索条件：新 attention mask、rope、KV format、chunk/bucket、LoRA、device 或 graph。
