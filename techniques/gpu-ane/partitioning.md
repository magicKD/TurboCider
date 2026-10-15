# GPU / ANE 切分：从合法代数找到并行窗口

[总目录](../README.md) · [CPU 参考演示](../examples/split_ffn_reference.py)

**M / F / H输入 / H输出 / Q / K-V sequence轴的区别、合法merge与UMA争抢**见 [维度与带宽专章](axes-and-bandwidth.md)，并有可运行negativecontrols；不把所有“hidden切分”混称为同一种FFN分工。

## 1. FFN intermediate-channel tensor parallel

采用 checkpoint `[out,in]` 权重记法：`X[M,H]`，gate/up `Wg/Wu[F,H]`，down `Wd[H,F]`。

```text
h = SiLU(X Wgᵀ + dg) * (X Wuᵀ + du)
Y = h Wdᵀ + down_LoRA(h) + bias + residual
I_gpu ∪ I_ane = [0,F), I_gpu ∩ I_ane = ∅
P_d = h[:,I_d] Wd[:,I_d]ᵀ
Y = P_gpu + P_ane + complete_down_LoRA + bias + residual
```

每台设备都执行同一逻辑通道范围内完整 gate/up→SiLU×up→down，返回 `[M,H]` partial。不是 GPU 算 gate、ANE 算 up，也不是各算完整 FFN。GELU MLP 同理但没有 gate/up 配对。

好处：避免 `[M,F]` concat 和跨设备 FC1→FC2 大交接。代价：两份 `[M,H]` partial、共同输入和不同设备 dtype、join、可能需要 corrected hidden 给 down-LoRA。带 bias 时 gate/up 分片 bias 按通道分，down bias 只加一次。

这是实数代数等价；不同 reduction/FP16/W8A8 会近似，不保证 bitwise。

## 2. Token-row split

对独立逐 row FFN，把 `X` 的行分 GPU/ANE，每支返回自己行的完整结果，按原行位置写回，无 channel partial sum。runtime-weight 模板可沿固定 chunk 重复使用；须计入逐层 W stage、每 chunk 调用和 padding。

row split 的优点不会自动延伸到 attention：后者跨 row 依赖 K/V 与全局 softmax。不能把整个 Transformer row 切开而省略 redistribution。

## 3. 拓扑决定能否与 attention 重叠

```text
并联 block：norm(X) ┬→ GPU attention ───────────┐
                   └→ GPU/ANE FFN shards ─────┴→ residual

串联 block：attention → attention residual → FFN input
                                     ├→ GPU shard ─┐
                                     └→ ANE shard ─┴→ join
```

FLUX-style 并联 MLP 可隐藏在 attention 后；Z/H3 类串联模型只能在 FFN 内分工，当前层 attention 不能和依赖它的 FFN 虚假重叠。LTX 音视频还存在 self/text/cross/FFN 支路依赖，必须按真实 DAG。

## 4. 其他切分轴

| 轴 | 合法条件 | 风险 / 本仓库结论 |
| --- | --- | --- |
| QKV/head | 完整逻辑 head/output channel、正确 norm/RoPE/layout | ANE straggler；Qwen/Z 历史组件与整请求不支持普遍默认 |
| sequence | 线性层局部可行；attention 需全局重排/归约 | H3 局部好但 all-layer 收益很小 |
| CPU tail | CPU 计算正确 shard 并在 first-consumer 前完成 | 历史 CPU tail 成为 straggler；默认不分 CPU |
| convolution output channels | 同输入/halo，disjoint outputs，按 channel assemble | 不做 input-channel partial reduction；VAE 全局状态仍保持 |

负结果依据：[异构技术报告](../../docs/design/transformer-heterogeneous-report.md)、[Qwen QKV auto](../../docs/status/runtime-ane-qwen-qkv-auto.md)、[VAE 可行性](../../docs/design/seedvr2-vae-ane-feasibility.md)。历史 micro 的 1.5–2× 不能移成新模型端到端承诺。

## 5. 怎么选第一刀

先测 `full_gpu`、GPU shard alone、ANE shard alone、concurrent 完整 FFN。选形状合法、stage 可摊销且接近负载平衡的几档 share，内存准入后用真实 complete-block 比较。不要预设 50/50，也不要只增加 ANE channels。

frozen、runtime row、Private/Public channel 有不同合同。当前 Public W8 row ABI 不能因为 Private 支持 channel-auto 就自动获得相同能力。调用入口以 actual executor/manifest 为准。

## 实现参考与实验

可读实现：[参考代码](../examples/split_ffn_reference.py)；[实验步骤与边界](../labs/scheduling.md)；[本项全部 case / controls](../labs/catalog.json)。

```sh
python3 -B techniques/labs/run.py overlap-dag
```

本项控制：full/channel/row algebra；full GPU vs real GPU suffix/CoreML prefix。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：different topology, channel unit, rows or precision。历史负结果限定于其来源/shape/边界，不关闭该方向。
