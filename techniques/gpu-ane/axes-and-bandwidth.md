# Sequence / hidden / FFN维度怎么切，切完怎么合

<!-- knowledge
id: technique-gpu-ane-axes-and-bandwidth
type: method
stage: ffn, prefill, dit
operator: gemm, swiglu, attention, norm
backend: cpu-reference
device: cpu, gpu, ane
symptom: memory-bound, wrong-layout, precision-loss
semantics: numerically-equivalent
boundary: gpu-ane, producer-consumer
api: reference-only
-->

[总目录](../README.md) · [基本切分代数](partitioning.md) · 可运行实现 · [带宽校准](calibration.md)

## 先给轴明确命名

设`X[M,H]`，gate/up权重`Wg/Wu[F,H]`、down权重`Wd[H,F]`，全部用checkpoint `[out,in]`记法。M是token/sequence rows，H是模型hidden，F是FFN intermediate。不同资料的row/column parallel命名可能反过来，因此最好写实际矩阵维而非只写“横切/纵切”。

| 切分对象 | 每branch做什么 | 合并位置 | 主要通信/依赖 |
| --- | --- | --- | --- |
| M / token rows | 每支完整逐row FFN | 原row位置concat/directwrite | 两引擎逻辑读取完整weights；attention/norm若跨M不再独立 |
| F / intermediate channels | 同组gate+up通道→SiLU×up→对应down列 | 两个`[M,H]`partial相加 | 通常只在hidden大小join；downLoRA还需完整correctedhidden或rank归约 |
| H输入 / gate-up reduction K | 各算`X[:,I]W[:,I]ᵀ`的gate/up partial | **SiLU之前分别sum gate与up** | 交换较大的`[M,F]`两组partial；不能先各自非线性再相加 |
| H输出 / down output rows | 共享完整hidden，各算`Wd[J,:]` | concat到`[M,H]` | 下层RMS/LayerNorm需全H统计；后续denseprojection也有依赖 |
| Q rows（attention） | 分Q，各自看完整K/V | concat输出 | 不需要跨Q的softmax归约，但K/V逻辑读取/存储可能重复 |
| K/V sequence（attention） | 同Q，各支只看部分K/V | 合并全局softmax统计 | 独立localsoftmax结果不能直接sum/平均 |

当前`HybridFfn/PartitionAxis`显式支持rows和intermediatechannels；H输入/输出与KV sequence在这里是**数学/设计参考**，不是声称当前Executor已有这些backend模式。可以继续探索，但需扩展graph、通信、precision/owner和真实consumer验收。

## 运行一个全轴对照

实现同时验证H输入先归约、H输出concat、M-row FFN与KV全局softmax；并给三个错误control：sum局部非线性、各outputshard独立RMS、平均localattention。这样不只是背一个“按hidden切分”的名词。

## H输入切分为何不能把激活留在branch里

```text
g = Σ_s X_s Wg_sᵀ + bias_g
u = Σ_s X_s Wu_sᵀ + bias_u
h = SiLU(g) * u
y = h Wdᵀ + bias_d
```

一般`Σ SiLU(g_s)*u_s ≠ SiLU(Σg_s)*Σu_s`。bias/residual只在逻辑位置加一次。gate/up LoRA若也沿输入H拆，A-rank partial同样要在对应B/非线性前正确归约；不能漏掉另一个device的input贡献。

H输出切分只对一个linear的output是concat；接RMSNorm需合并全hidden的sum-of-squares，LayerNorm还需count/mean/variance。不能因为本次down可独立，就假设整个下一block都独立。

若模型实际用per-head/group-wise norm，只需覆盖其真实统计域；是否跨shard归约取决于norm axis，而非“hidden”这个名称。

## K/V sequence split如何恢复一个softmax

每个query在shard s上保存`m_s=max(logits_s)`、`l_s=Σexp(logits_s-m_s)`、`z_s=Σexp(logits_s-m_s)V_s`：

```text
m = max_s m_s
l = Σ_s exp(m_s-m) * l_s
z = Σ_s exp(m_s-m) * z_s
out = z / l
```

这与online softmax同源。全mask/emptyshard、causal的globalposition、dropout/scale/precision、KVheadlayout、跨device归约与尾部都要另验。可以研究head/sequence联合分区，不能只减少KV长度而保留错误local归一化。

## 统一内存仍有带宽争抢

粗略研究下界可以写为：

```text
T >= max(compute_gpu, compute_ane,
         (bytes_gpu + bytes_ane + bytes_W/A_stage + bytes_restore_join) / available_UMA_bandwidth)
```

这是逻辑traffic模型，不是Apple官方带宽或DRAM真实读次数：cache/SRAMreuse、压缩/页fault、encoder/媒体/其他程序都会改变可用带宽。F切分避免大activation跨设备，但共同X、双Wbank、A8stage和partialjoin仍消耗流量；M切分也可能让两引擎各读全weights。

脚本只用合成字节预算演示compute-only 4ms可被UMA floor推高到6ms，数值不是本机测量。实机应独立采GPUalone/ANEalone/concurrent完整traffic，观察两支是否同时变慢，再用真实时序和counter归因；exposedwait不足以替代alone。

## 确认争抢后可以探索什么

缩小或改变share、选更小bucket、按prefill/decode分phase；对W/A预取独立off/on、推迟非critical producer；减少重复decode/copy、compact GPU suffix、typedpackedconsumer、共享小metadata；限制inflight/双槽生命周期，或改用更粗的graph边界。

这些不是固定优先级：更深overlap也可能有效，只需证明少等的时间没有被更多带宽/驻留税抵消。新shape/source/SDK/SoC、不同cache或LoRA策略都值得重测，不能从一轮负收益推出某种轴永久不可行。
