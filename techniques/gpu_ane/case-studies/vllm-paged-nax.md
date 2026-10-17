# vLLM-Metal 深读：分页 KV、混合 batch 路由与 NAX fragment

<!-- knowledge
id: technique-gpu-ane-case-studies-vllm-paged-nax
type: case-study
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

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [GPU attention基础](../gpu-attention-runtime.md) · [续研证据](../../MAINTENANCE.md)

固定 [`vllm-project/vllm-metal@468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e`](https://github.com/vllm-project/vllm-metal/tree/468b0c8d6a3f2afe617311d76ed25ee5e3b0d05e)。本轮已从 README 升级到 `_try_init_nax_library`、`paged_ops.cpp` 的实际选择与提交、454行 `pagedattention_nax.metal` 全文及loader测试；未执行上游、未建立M5性能资格。

## 1. 硬件、库、shape、请求形态是四层门禁

Python loader先允许显式禁用，再检查硬件，最后加载独立metallib或选择JIT；缺少可选库或加载失败会警告并保留非NAX路径。`nax_eligible` 又限制half/BF16、headD∈{64,96,128,256,512}、KV block∈{8,16,32}及库ready/启用状态。

host真实路由还排除纯decode、speculative verify window、TurboQuant、query/cache dtype不匹配和某些多模态prefix mask。**NAX supported、library ready、该batch实际走NAX是三个不同结论**。不能因开机识别到M5，就把每次请求都算作TensorOps加速。

所读hardware gate是macOS26.2加Apple10 family能力；loader测试是mock政策测试，不会执行M5 shader。host的`g_last_dispatch`又是process-wide测试诊断，并非per-request trace：并发请求下不能据最后一个标签给每个请求归因。

## 2. 一批请求不必使用一种核

单token decode若塞进BQ=64的prefill tile，绝大部分行空转，且没有沿长KV的并行度。该读本在满足条件的mixed batch中拆出前缀普通decode rows，以独立context上界规划split-KV；prefill suffix继续NAX或tiled kernel。

不能用整个cache allocation上界替代decode真正长度：prefill可能很长，使decode分区数、scratch和reduction不必要地膨胀。spec verification window又有多query复用与自身split-KV路线，不能因它“有多个token”就当普通prefill。

注意两个“hybrid”：此处是prefill/decode任务混合，模型里的GDN/SDPA也是另一种hybrid；**均不等于GPU+ANE**。该页全部NAX计算仍位于GPU。

## 3. 不把分页 KV 先拼成连续大张量

kernel通过block table将逻辑KV token映射到物理page。每个32-token tile分为4个8-token半fragment，8/16/32的page size均能保持每次宽加载不跨page。K/V指针在tile开头计算，后续按固定fragment offset取vec4。

这把cache的动态寻址直接带入Flash-style数据流，避免每次attention先gather完整cache。但page lookup、尾部mask、地址计算不是免费；cache容量、block table与actual序列长度必须属于同一请求generation。

variable-length Q以 `floor(cu_seqlens_q[s]/64)+s` 作为每序列的虚拟block起点；因此grid允许少量padding block并在kernel中早退。mixed前缀的offset也必须用同一几何，而不是简单除以64。CPU几何见证验证不漏/不重真实query行，不能代替GPU读写验证。

## 4. Cooperative tensor：显式寄存器布局与数值代价

一个threadgroup为4个simdgroup，每组负责16 query行；QK与PV以16×32×16的MPP operation完成。`nax_coord` 手动镜像descriptor决定的lane分布，然后填充operation helper创建的cooperative fragments。

这一技巧绑定descriptor、transpose与precision模式，不能把任意lane map套到其他API重载。所读descriptor显式启用`relaxed_precision`；源码说明PV的FP32 probability会被截断，因此不同于保留FP32 P的旧tiled路径。新增硬件路径需独立logits/token parity与质量预算，不称数学数据流相同就必然bitwise中性。

Q按每个KV tile重读，V按D pair重读，是降低寄存器活跃集的取舍。没有threadgroup staging不等于没有同步：shader仍有simdgroup/threadgroup barrier来约束进度，且所有scope线程须共同参与。

## 5. Mask、sink和完全masked tile

在线softmax使用有限最小值作mask sentinel，并对masked logit显式选择exp=0。若只令sentinel参与max再算exp，整tile masked时会出现`exp(sentinel-sentinel)=1`，凭空增加denominator。

attention sink是分母中的额外logit，不带value行；初始化running max/sum，而不是向输出加一个value。最终还有`sum+1e-6`的零分母保护。mask、sliding-window、softcap、sink、GQA和tail是完整数学合同的一部分，不能只用无mask随机SDPA验证。

## 6. 什么值得迁移

- 把host路由、可选库可用性和kernel资格拆开；记录实际dispatch path。
- 同batch按请求几何选择kernel，decode分区使用自己的有效context界。
- 在kernel里直接消费paged layout，避免framework边界反复materialize。
- 用scope内fragment融合减少搬运，同时明确precision模式和layout绑定。
- 保留完全masked row/sink/sliding-window以及错误virtual-offset的负控。

本轮新增的是静态实现分析与CPU数学/索引见证。原先896个MLX计时输出不是该vLLM kernel的实测；本机M4也不能验证此NAX路径。后续先做matching parent/head或NAX on/off的同模型、同cache/精度A/B，再测TTFT、TPOT、吞吐、scratch与request p95。
