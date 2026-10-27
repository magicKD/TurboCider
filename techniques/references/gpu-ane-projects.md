# gpu_ane 专项：mac_local_ai、mac_transformer 与图像生成路径

<!-- knowledge
id: technique-references-gpu-ane-projects
type: method
-->

[总目录](../README.md) · [拓扑设计](../designs/parallel-topologies.md) · [消费者ABI](../designs/consumer-abi.md) · [模型选择指引](../models/hybrid-selection.md)

本轮对本地`../gpu_ane`进行只读源码核查，不运行其模型、编译Private API、不下载权重，也不修改参考仓库。下列HEAD仅标识本地checkout，不声称上游最新；精确文件SHA和dirty状态见机器阅读账本。核心设计与独立demo已经移入本目录，不要求原checkout存在。

## 项目分工与阅读到的实现

| 项目 / 本地HEAD | 实际检查的核心 | 已吸收的机制 / 本地代码 |
| --- | --- | --- |
| mac_local_ai `a2113ff` | HeteroMem报告、FLUX2_ANE、Flux2ANEBridge、MetalResourceHazard、GDN精度岛脚本 | producer stride、paired segments、resource identity、持久binding/outputs、precision lifetime；[heterogeneous.py](../examples/reference/heterogeneous.py) |
| mac_transformer `9f322e1` | planner/workloads、architecture、native CoreML bindings/worker/join接缝 | exact channel MLP、可选GPU/ANE/CPU、p95 critical path、parallel vs sequential拓扑；[planner与拓扑样本](../examples/reference/heterogeneous.py) |
| flux2-engine `98aa6f1` | architecture、hybrid_ops的packed slices与async调用、backend/artifact组织 | GPU attention+ANE fused MLP、GPU MLP complement、persistent service、bucket/backing gate；[选择指引](../models/hybrid-selection.md) |
| mflux `051ba9f` | FLUX.2 single_transformer_block / parallel_self_attention | joint QKV+MLP projection、分支concat再Wo、共同modulation与residual；[parallel-branches](../examples/reference/heterogeneous.py) |

mac_local_ai与flux2-engine存在用户原有tracked修改，已保留；账本以**实际读取的工作树文件SHA**为准，不能把HEAD当成所有文件内容相同的证明。mac_transformer和mflux本轮检查时tracked worktree干净。

## 1. mac_local_ai：把物理边界纳入优化

其重点不仅是矩阵N切分，也包括consumer-driven tensor ABI：Core ML的4D `[1,N,1,M]`通过非默认strides直接写GPU row-major；配对gate/up segments穿过SwiGLU；不同buffer/heap身份影响tracked hazard粒度；UMA contention可能吃掉并发收益。

Flux2ANEBridge通过当前Py_buffer保住FP16输入/输出，noop deallocator只用于借用，不接管内存；缓存wrapper/provider/options减少设置开销。它检查返回backing；本地更通用的native样本也演示了未采用时按真实strides回拷并报告。两者是不同失败/性能合同，不该混成“必然零拷贝”。完整解释与正反例见[consumer ABI](../designs/consumer-abi.md)。

GDN precision研究按组构造FP32 shadow projection/conv/参数/state/consumer路径，启发按误差存活时间分配精度；本地只实现标量递推见证，不冒充完整Qwen3.5/GDN。高精度岛和量化残差只有真实完整路径成本/质量通过才有工程意义。

## 2. mac_transformer：先问图能不能并行

核心MLP按intermediate F配对gate/up、每设备完成down并返回H宽partial，residual join一次；baseline用相同source shards重组full GPU weights。持久MPSGraph/CoreML/BNNS、shared start gate、各分支wait之后join，以及caller output pointer检查构成完整计时边界。

Planner允许ANE/CPU为零，使用有效吞吐、launch、tail与带宽/争抢模型；这些是估计，最终要用真实sweep与candidate试验确认。Workloads明确普通顺序块与FLUX.2 parallel枝干不能共用`max(attention,MLP)`。这比复制一个固定50/50 share更可复用，已在[拓扑](../designs/parallel-topologies.md)和[planner](../designs/heterogeneous-planner.md)重述并独立实现。

## 3. 图像生成：mflux图 -> flux2-engine放置

读到的FLUX.2 single block先LN和shift/scale得到Z，attention模块内部同时投影QKV及gate/up；attention A与SwiGLU B concat后乘Wo，外层再乘mod_gate加residual。`Wo`按输入列拆成两partial后相加，因此两枝可以合法放在不同设备。

hybrid_ops只取GPU需要的QKV输出行，保留packed Q4 weight/scales/group格式；`mx.async_eval`启动GPU路径后同步调用公共Core ML bridge，最后合并已ready的输出。某些F仍留GPU时，必须按完整quant groups切input columns、对应packed words/scales，不能把FP16列索引直接用于packed weight。

持久backend保留session/binding/output，每request复用减少allocation；相同输出slot的请求串行化。实际rows包含图像与动态text，选最小bucket；missing bucket、one-shot、cold与tail敏感任务不必强行ANE。历史文档中的INT8/q4、三步/四步、eager/compiled不同分母单列，不把它们作为“同语义默认加速”继承。

## 技术点与本地演示

tests检验错误拓扑、漏head、默认布局误读、gate/up错配、CPU/ANE straggler、别名写、更多overlap无wall收益和参数已舍入的精度岛下界。CPU样本的零误差只证明小型代数/索引，不是SDK接受任意stride、实际物理ANE overlap或整模型性能资格。
