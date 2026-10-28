# 新模型 GPU/ANE 方案选择：按图结构而不是模型名套配方

<!-- knowledge
id: technique-models-hybrid-selection
type: guide
stage: encoder, prefill, decode, dit, ffn, vae, pipeline
operator: attention, swiglu, conv, scheduling
backend: cpu-reference, metal, coreml
device: cpu, gpu, ane
symptom: straggler, wrong-layout, precision-loss
semantics: contract-only
boundary: gpu-ane, request
api: public, reference-only
-->

[阶段地图](stage-map.md) · [拓扑设计](../designs/parallel-topologies.md) · [本地可选设备样本](../examples/reference/heterogeneous.py)

## 先填写这张事实表

block是sequential还是parallel branches？FFN是SwiGLU/GELU/其他？attention mask是causal/bidirectional/局部/跨modality？H与head_inner是否相同？F、actual rows、完整tokens与buckets是多少？GPU权重packed格式、ANE图是否frozen/runtime、LoRA、dtype/rounding和state是否跨step存活？不能仅因类名叫attention就认为整个类只做attention：读到的FLUX.2 attention模块也包含MLP和联合output projection。

## 第一轮可实施候选

| 实际图结构 / 阶段 | 首选GPU控制 | 可探索并行 | 必须拒绝的替代 |
| --- | --- | --- | --- |
| 普通causal LLM prefill / 顺序pre-norm | online attention、packed/tiled投影、chunk全局KV | attention完成后的完整FFN F切分；下一W预读 | 把同层FFN提前与attention重叠 |
| FLUX.2式parallel single block | 同一调制Z的QKV/MLP融合基线 | GPU attention output partial + ANE完整MLP partial；必要时再切MLP F | 把concat后的非线性拆到各分支，重复residual/gate/bias |
| H3式顺序DiT或其他attention→FFN | 强GPU attention和layer链 | FFN内部F或合法projection/heads切分 | 从parallel single block复制调度图 |
| 小M decode | packed GEMV、KV/GQA、少提交、sampler | 仅实际完整边界获益后小shard/跨请求batch | padding最大ANE bucket、强制CPU参与 |
| 固定encoder / vision projection | 预处理+encoder完整GPU、正确conditioning cache | N-split projection或F-split，consumer直接接受布局 | 双向prefix套用因果KV缓存 |
| VAE conv / upsample | 全局norm/halo/phase正确的GPU图 | 固定完整子图CoreML及shared IO | 局部norm或独立tile错误被当工程等价 |
| 长视频 / 低RAM | bounded IO、固定resident subset、phase释放 | 有界预读、独立请求/已证明独立帧阶段pipeline | 无限slots、未drain提前复用、LRU循环thrash |

CPU只参与完整成本确实改善的projection/FFN分片；控制逻辑/采样适合CPU并不证明把大GEMM分给CPU也合算。先准入memory，再评deadline和性能。

## 第二轮优化图边和持久化

若trace显示bridge/join/layout占主导，先选[消费者ABI](../designs/consumer-abi.md)：producer stride、segmented pointwise、down partial、caller backing和绑定复用。若成本在compile/load/first，选持久backend、合适bucket/function数和idle策略。若来自真实带宽争抢，多offload可能更慢，应调整share或直接关闭该设备。

真实FP16/BF16/INT8语义需要单独对照：存储相同bit数不等于相同basis/scales，改变cast点也会改变结果。精度岛按状态误差生命周期选择，不能用组件oracle替代可运行的shadow路径或真实图像/语言holdout质量。

## 本目录内的小实验路线

这些按技术点独立运行，含数学正确控制与错误反例。它们不加载mac_local_ai/mac_transformer/flux2-engine/mflux，不执行Private ANE或下载真实权重。实际GPU/CoreML练习接[设备实验室](../examples/README.md)；全模型接入还需source/adapter、真实activation、完整stage/request和质量/低RAM验证。

## 从结论回到证据

每个新候选写一张技术卡：输入图/axis、source/shape/recipe、ABI/owner、memory、matched GPU、candidate scope、数值/画质、冷/热/尾、拒绝原因、revisit。当某项目报告速度很高，先问其baseline是否eager、是否换bitwidth/steps、是否只测热service、是否包含model construction和VAE。来源介绍见[项目对照](../references/gpu-ane-projects.md)，不把其历史数字改写成本轮TurboCider收益。
