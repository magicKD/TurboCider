# Attention：强 dense 基线，再谈规则稀疏

<!-- knowledge
id: technique-metal-attention
type: method
stage: encoder, prefill, decode, dit
operator: attention, norm, rope
backend: metal, mlx
device: gpu
symptom: memory-bound, precision-loss, wrong-layout
semantics: numerically-equivalent, approximate
representation: bf16, fp16, fp32
boundary: producer-consumer
api: public
-->

[总目录](../README.md) · [近似门禁](../algorithms/methods/approximations.md)

## 优先级

先保留支持目标 dtype/shape/mask 的 fused SDPA / tiled exact attention，融合 Q/K norm/RoPE、保持 head layout 连续，避免显式 `N×N` score。小序列、长视频、GQA、参考 prefix 的 kernel 策略应分别测，不强制同一个 fused heuristic。

一个 tile 的在线 softmax 可用下面的稳定递推（每个 query 独立）：

```text
m_new = max(m_old, max(scores_tile))
l_new = exp(m_old-m_new)*l_old + sum(exp(scores_tile-m_new))
o_new = exp(m_old-m_new)*o_old + exp(scores_tile-m_new) @ V_tile
output = o / l
```

边界必须覆盖 mask 全空、causal/非causal、尾 K/V、极大 logits、head_dim、输出 dtype；`l=0` 的语义要和参考一致。少 FLOPs 不意味着低延迟，尤其 route/top-k/gather/repack 又增加全张量流量时。

## 参考 prefix 与 GQA

Qwen 编辑的首步包含 text/reference/target；后续 prefix KV 命中后只更新目标部分。每层必须有自己的 K/V，不能错误共享一份 bank。Prefix 是否可重用还受 timestep/position/adapter 语义影响，参见 [缓存](../algorithms/methods/caches.md)。

Encoder GQA 应避免不必要复制 KV heads；但必须确认 backend 支持的 layout 和 mask，不能只从理论 GQA heads 比例推速度。实现与整请求对照见 [compiled GQA](../../docs/status/local512-encoder-compiled-gqa-2026-10-09.md)。

## Metal-friendly sparse execution

建议实验路线：规则连续 K/V tile traversal，tile-level routing bitmask；Q tile 由 threadgroup 持有，遍历选中的原位 K/V，不 materialize gather tensor。局部/同帧/conditioning exact、远端 pooled correction 是不同近似策略，必须分别标记。

路由本身也有成本：pooled QK、top-k、bitmask、summary、shared memory、barrier、余量。不要只报告 sparse core；应依次测 core → QKV/gate/output 分支 → 完整 stage → 整视频。

## 本仓库最有价值的反例

[LTX Stage-2 结果](../../docs/design/ltx-stage2-sparse-results.md)中，720p direct top-k32 在捕获 QKV core 上约 4.05×，完整 Stage-2 仅 1.238×，完整请求约 1.106×，且 RGB/LPIPS 未达到所需质量。all-exact sparse control 还比 dense 慢，说明路由框架本身有税。

同一研究把 route 改为 packed u32 bitmask，降低 scratch，但不能把这点字节节省写成整个模型等量降内存。固定某层最优 head budget 用到其他 step/block 会退化；离线 oracle 不是在线策略。

## 长序列attention迁移ANE：已有负结果与重开条件

[异构报告](../../docs/design/transformer-heterogeneous-report.md)中的 public Core ML H3 单 head 大 shape 已有很高 hot 成本，还没包括 QKV/RoPE/output；QKV split 去掉 concat 后仍会由 ANE straggler 限制。

可表达一个 SDPA 图 ≠ 高效执行长序列 SDPA。新模型如果要做 head/sequence partition，必须计入全局 softmax、sequence-to-head 重排和所有跨设备归约。FFN 的简单线性 partial-add 不能直接套到 attention。

可探索新的fused projection/attention边界、固定tile图、不同mask/head形状、KV布局与新SDK/芯片；先构建真实smallcase并比较完整分支，不把旧H3结果扩成“所有ANE attention都不值得做”。

## 2026-10-16：真实Qwen纵切与源码走查

[MLX固定读本](../sources/repos/mlx-sdpa-dispatch.md)说明full/vector的head_dim、GQA、mask与copy predicates不同。full路径接受末维stride=1的interleaved V，输出又故意按token-major存储；不能把所有`row_contiguous=false`都当错误。新SDK或营销芯片名也不证明实际NAX路径，源码guard与运行binary身份分开保存。

本机使用原GPU block observer的actual arguments，按同一MLX公式重放并额外导出Q/K/V/O；全block/prefix控制通过，但不是未改变compiled kernel内部的直接抓取。数据包含两seed、32 heads、D128、真实1048-row prefill/1024-row steady，没有缩小head数或重复toy行。

`single-mask`、调用内`contiguous-v`、8-head chunks恢复完整原消费者后都保留数值；目前未见稳定胜过原分段/单次SDPA的收益。原样graph noop、显式准备/驻留和所有确认pair说明为什么不能晋升1–2%的表面差异。全O对BF16 replay reference误差0；FP64是选head/query、完整K/输出列的独立row见证，不是全输出FP64或请求画质。

特别的反例：当前binary的全false mask行未满足本轮零输出约定；常量V=1时，D64/K29返回29/32，K32返回1。有限sentinel、tail与归一化是解释方向，不是已定位的硬件故障。需要zero语义的调用方应明确实现/验证safe-key与置零等机制，不能假设所有路径自然返回0；本轮未修改产品默认。

完整[实验/失败/不确定性](case-studies/qwen-attention-replay-and-empty-row.md)与。复现入口是attention_bench.py、CPU独立数学、只读原始BF16 reader与empty-row探针。

## 最小验收

同 source 的 dense control、all-exact sparse control、真实中晚层 QKV、多 prompt/seed、完整 stage 和视频；视频 motion/temporal/身份门禁独立于 latent cosine。默认 dense，除非候选真正通过既定范围。

## 实现参考与实验

可读实现：[参考代码](../examples/reference/operators.py)；[实验步骤与边界](../examples/README.md)；本项全部 case / controls。

本项控制：dense/online exact masks；structured sparse error and empty-mask CPU control。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：different sequence/head shape, block traversal, routing or pooled correction。历史负结果限定于其来源/shape/边界，不关闭该方向。
