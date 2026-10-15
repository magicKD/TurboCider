# 按模型阶段选择加速：从 encoder 到输出媒体

[总目录](../README.md) · [设计样本](../designs/README.md) · [既有模型配方](recipes.md) · [按图选择GPU/ANE方案](hybrid-selection.md)

先画完整请求：`输入预处理 -> encoder/LLM conditioning -> N步DiT -> VAE -> 媒体/后处理`。自回归任务则是 `prefill -> 逐token decode`。并非每个模型都有全部阶段，也不能因为都叫 transformer 就共用 mask、cache 或 ANE bucket。

| 阶段 | 常见主成本 | 优先保持语义的方案 | GPU/ANE 候选 | 近似方案须额外许可 | 本地样本 |
| --- | --- | --- | --- | --- | --- |
| [Text/vision/audio encoder](encoders.md) | 一次性 dense projections、attention、预处理 | 精确 conditioning cache、masked bucketing、fused norm/attention | 长 rows 的 FFN split；固定 shape 的 conv/投影 | 权重量化、token pruning、降输入分辨率 | [stage-encoder](../labs/model_stages.py) |
| [LLM prefill](llm-prefill.md) | 大 M GEMM、长序列 attention、KV 写入 | tiled GEMM、online causal attention、GQA、正确 chunking、因果 prefix cache | F-channel FFN split、row threshold、双 W bank | W8/W4、KV quant、attention sparsity | [stage-llm](../labs/model_stages.py)、[hybrid FFN](../designs/hybrid-ffn.md) |
| [LLM decode](llm-decode.md) | M 很小，权重/KV 读、launch、采样 | packed GEMV、GQA、KV layout、融合、batch 调度 | 只有完整边界证实有利才 offload；通常先排除固定交接税 | 低 bit 权重/KV、精确 speculative 算法中的 draft | [stage-decode](../labs/model_stages.py) |
| [DiT / diffusion transformer](dit.md) | N步×多层 attention/FFN，latent 与临时流量 | 融合 AdaLN/QK/RoPE、online attention、合法 cross-KV hoist、驻留与 streaming | 逐层 FFN split、有限 lookahead、shape bucket | residual reuse、token merge、sparse、低 bit、减步 | [stage-dit](../labs/model_stages.py)、[cache](../labs/caches.py) |
| [VAE encode/decode](vae-and-media.md) | conv/upsample、巨大激活、全局 norm、时间历史 | 精确 halo、global-stat pass、output-phase tiling、bounded temporal history | 固定完整子图的 public Core ML，计 pack/restore 全成本 | 独立 tile/feather、局部 norm、较低精度 | [stage-vae](../labs/model_stages.py)、[causal-halo](../labs/operators.py) |
| [完整 pipeline](pipeline.md) | cold load、phase peak、IO、后处理与输出队列 | source/graph cache 分层、释放 encoder、低内存协作、bounded frame queue | 仅独立 stage/request 的合法 overlap | 任何改变输入/步数/质量的方案都单列 | [phase plan](../scripts/interval_peak.py)、[media queue](../labs/media.py) |

[拓扑分类](../designs/parallel-topologies.md)特别区分：同一调制输入的FLUX.2式parallel attention/MLP分支，与attention residual之后才运行FFN的顺序图。对不同模型的候选、拒绝条件和可运行demo见[选择指引](hybrid-selection.md)。

## 一个统一但不混淆的选择过程

1. 记录实际 shape：projection 的 M/K/N、attention Q/K/head 数、VAE 的 C/T/H/W，以及 mask、dtype、物理 strides、LoRA、step 数。
2. 建立优化 GPU 控制；分别测 cold compile/load/first-call、hot operator、完整 stage、完整请求。不要先人为放慢 GPU 再证明 ANE。
3. 用 [GPU 数据流](../designs/gpu-dataflow.md)判断算力/内存/launch/IO 限制，优先消除无意义中间数组与重复初始化。
4. 满足代数和 ownership 合同时才试 [GPU/ANE FFN](../designs/hybrid-ffn.md)。允许拒绝不盈利的短 rows，而不是一定要让两个设备都忙。
5. 保持语义的工程方案与近似方案分开记录；量化也要数值和任务质量门禁，不因张量 shape 相同就归入 exact。
6. 到完整请求重测；组件更快可能把峰值移到 VAE 或把墙钟瓶颈移到 encoder/媒体。

## 本目录内的最小学习路径

读对应阶段页 -> 读 [设计说明](../designs/README.md) -> 运行 [labs/run.py](../labs/run.py)指定 case -> 读反例测试 -> 运行需要本机 SDK 的 Metal/Core ML 样本。全部 CPU 样本只用标准库；硬件依赖、失败和未验证边界均明确报告。

```sh
python3 -B techniques/labs/run.py stage-encoder
python3 -B techniques/labs/run.py stage-llm
python3 -B techniques/labs/run.py stage-decode
python3 -B techniques/labs/run.py stage-dit
python3 -B techniques/labs/run.py stage-vae
```

这些是机制样本而非整模型后端。catalog 将章节、代码、case、controls 和 revisit 条件连起来；[自包含检查](../scripts/check_standalone.py)会把本目录复制到没有 TurboCider 其他源码或兄弟仓库的临时目录，实际运行 CPU 套件。
