# GPU/ANE 并行拓扑：同为 Transformer，合法边界却不同

<!-- knowledge
id: technique-designs-parallel-topologies
type: method
stage: prefill, dit, ffn
operator: gemm, swiglu, attention, scheduling
backend: cpu-reference
device: cpu, gpu, ane
symptom: straggler, wrong-layout
semantics: numerically-equivalent
boundary: gpu-ane, producer-consumer
api: reference-only
-->

[设计目录](README.md) · [按模型选择](../models/hybrid-selection.md) · [本地实现](../examples/reference/heterogeneous.py)

本篇吸收mac_transformer的topology/workload建模、mac_local_ai与flux2-engine的实际分工，并对照mflux的FLUX.2 single block源码。来源与文件SHA见[专项阅读记录](../references/gpu-ane-projects.md)。不要求安装或打开任何原仓库。

## 六种不同的并行，不混用公式

| 拓扑 | 分支输入 / 合并 | 适用边界 | 易错点 |
| --- | --- | --- | --- |
| 单projection输出N切分 | 共享X；按输出channels concat/segmented view | 独立输出行的linear/conv projection | 只省算子、不计下一consumer的concat/transpose |
| 完整FFN intermediate F切分 | gate/up配对channels；down hidden partial求和 | 非线性逐channel的SwiGLU等 | gate/up错配；只切gate不切up；down bias/residual重复 |
| attention与MLP平行分支 | 两者消费同一norm/modulated X；hidden partial相加 | 具有该图结构的FLUX.2 single block | 把顺序block强行改成平行，改变模型数学 |
| 完整attention heads | 每head保留完整允许KV；output projection列分片后求和 | heads独立且output projection线性 | 截断keys、切开head norm/RoPE、漏head、GQA映射错误 |
| token/query rows | 独立输出rows；attention仍见完整合法KV | 逐tokenFFN、正确chunked attention | 把local softmax平均或丢跨chunk KV |
| pipeline / weight lookahead | 独立请求/阶段或下一层已知W | 有合法ready/last-reader边的IO/算子 | 同层attention输出未ready就启动依赖它的FFN |

## A. 顺序 pre-norm block

```text
U = X + Attention(Norm(X))
Y = U + FFN(Norm(U))
```

FFN的输入依赖attention residual，因此同层attention与整个FFN不独立。合法候选是在attention完成后切FFN内部F，成本近似：

`T = T_attention + T_pack + max(T_gpu_FFN_shard,T_ane_FFN_shard,T_cpu_FFN_shard) + T_join`。

这是mac_transformer对普通顺序Qwen/H3路径所做的区分。模型具体norm/残差/激活仍按实际图确认，不按名字猜。详见[FFN设计](hybrid-ffn.md)与[维度代数](../gpu-ane/axes-and-bandwidth.md)。

## B. FLUX.2式 parallel single block

读到的mflux图：同一个调制后的Z送给QKV与MLP gate/up，attention输出A、SwiGLU输出B先concat，再经一个linear输出投影。将输出权重按输入轴拆为`Wo=[Wa | Wb]`，有：

```text
Concat(A,B) Woᵀ = A Waᵀ + B Wbᵀ
Y = residual + modulation_gate * (attention_partial + mlp_partial)
```

因此GPU可做QKV/norm/RoPE/attention/attention输出partial，ANE做完整MLP投影→SwiGLU→MLP输出partial。只返回`M×H`，不将`M×F`或`M×2F`中间值来回交接。调制gate、输出bias和residual必须加一次；如果output projection后还有非线性，要先求和再过该非线性。

成本是`input_pack + max(GPU_attention_branch, ANE_MLP_branch) + hidden_join`，不是把两分支时间相加，也不是从普通顺序block删除依赖。本地`parallel-branches`直接比较concat-linear与两partial，并给出错误顺序改写反例。

ANE不能容纳全部F时，可把MLP再按F切成ANE prefix和GPU complement。这是两层分工：GPU attention与GPU MLP complement仍在共享GPU资源上，不能把它们都假设为完全并行；两者与ANE的UMA争抢需完整计时。

## C. 完整 head 分片

每head的softmax必须覆盖该head原本允许的全部keys。分片经过QKV、head norm/RoPE、SDPA，再将Wo对应输入列带入分片；各device输出`M×H`partial。切分只改变工作放置，不改head数或KV head映射。GQA可能让多个Q shards借用同一KV backing，增加跨分支读流量；不是“head切分就线性省内存”。

本地`attention-head-shards`含三个head及不均匀hidden输出，完整对照与漏一个head但shape仍正确的反例。它是MHA数学样本，不是新ANE SDPA后端；ANE全attention仍需[完整放置门禁](heterogeneous-planner.md)。

## 运行与重新探索

新模型先从事实图标记边，再枚举拓扑；phase、bias、adapter、nonlinearity、mask、position、source或consumer变化后重审代数与lifetime。上述公式只是完整边界的分解；实际pack、precision、队列与争抢必须测量，不把原项目某台机器的倍率作为本库新结果。
