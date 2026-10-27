# plow：persistent packet、两种 ANE 分担与事务式 fallback

<!-- knowledge
id: technique-gpu-ane-case-studies-plow-packets-and-offload
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [第三轮证据](../../MAINTENANCE.md) · [来源清单](../sources.md#compiler-packets)

已核实的公开上游是 [`infervisor/plow@35dfa22ed319ece06255e10e4d76996f7a48977b`](https://github.com/infervisor/plow/tree/35dfa22ed319ece06255e10e4d76996f7a48977b)。此前候选中的`maderix/plow`不可访问，未算已读来源；本页纠正owner，旧账本保留其unverified身份。root README的主要serving支持表是CUDA/AMD，Apple另有Metal/ASR路径，不能直接继承其他GPU的性能。

本页为第三轮源码范围。第七轮已用相同shader SHA执行[原始kernel小packet](plow-packet-protocol-probe.md)，含预读、两组counter和定义好的逻辑fault；不是运行完整plow或ANE路径，也不修改本页当轮未执行的历史身份。

## 1. Static packet 减少的是哪一种开销

Apple shader定义64-byte `Inst`、24-byte stream entry、wait threshold和successor表。`plow_interp`每个threadgroup执行一条virtual-CU stream，在同一dispatch内按依赖计数器执行多个op/slice，减少逐算子CPU提交。

这不是Metal对任意动态计算图的自动融合：编译器必须正确安排slice、counter、segment、指针和GPU资源。shader的1024线程、32KiB threadgroup scratch和virtual-CU数量是该实现选择，不是可移植最优值；驻留/进展条件不能从CUDA cooperative-launch保证机械继承。

GPU内的等待使用relaxed atomic counter，但数据发布还显式有device-scope fence与threadgroup barrier。消费者看见counter不等于普通payload load已可见。超出spin上限或遇到未知op写fault word；command buffer Completed只表示设备提交结束，不证明packet成功，caller还须检查fault。

其`plow_single`诊断核每次只执行一条instruction，用kernel边界隔开producer/consumer。这是定位“op数学错误”与“persistent内跨threadgroup可见性错误”的好对照；本轮没有执行这两种kernel。

## 2. 两条异构路径不能混为一个加速方案

| 方案 | 切分/边界 | ANE执行单元 | 实际交接 |
| --- | --- | --- | --- |
| `apple/hetero.rs` row lanes | 编译时segment；运行时按真实rows重新分GPU/ANE/CPU | pre、mid、post等封闭CoreML图，64-row调用并pad尾部 | GPU先commit，CPU持久pool和host ANE运行，segment处join |
| `apple/channel.rs` channel MLP | SwiGLU intermediate通道分片；独立GPU/ANE partial | 对特定validated bucket/层构建frozen MLP图 | producer先ready，GPU complementary MLP与ANE预测，再GPU reduction/residual |

row路径中，注意力需要全序列的Q/K/V，不能各row独立算局部attention再拼接。ANE的`mid`把本层post/MLP与下一层norm/QKV合进一个图，减少跨设备回合；全局依赖仍在segment join上。原始persistent walk也因此在host/ANE边界分段，不是整次推理始终只有一个GPU launch。

## 3. Row policy：真实row、对齐与不可用设备

`split_rows`按真实chunk长度而非整个bucket分配ANE/CPU百分比，向下对齐8 rows，并给GPU保留尾部/至少一小块。`prepare_chunk`从pristine instructions重建GPU行数；非活跃lane会恢复GPU覆盖，避免把未计算的row留空。

CPU lane将特定row-wise instruction重基到自己的row区间，同一tensor handle出现多次只offset一次，防止in-place alias被偏移两遍。不是所有packet op都可这样重基，加载时有逐op验证。

**错误路径另看。** `run_lanes`可在CPU任务启动后因ANE错误返回；所读`run_prog_hetero`先`r?`、后wait CPU/GPU。因此仅靠正常路径末尾的join，不能证明早退也会drain。是否由外层所有权保证仍需故障验证；不把channel路径的RAII保护自动归到row路径。

## 4. Channel MLP：原始输入不被partial覆盖

GPU gate/up与down写独立intermediate/partial，ANE使用自己的prepared输入/输出。`run_layer`在预测成功或失败后都先join已提交GPU；`Submitted::drop`也会wait。输入和residual在两支计算时不被覆写，所以失败时可以执行完整原始MLP。

成功前还检查GPU/ANE partial finite。只有reduction完成后才修改residual、发布原packet span的successor计数；失败则不发布partial/counter，恢复原MLP及suffix，并禁用该优化。它与[vpipe in-place失败恢复](vpipe-controller-and-failure.md)的条件明显不同。

skip一个packet span时，successor可能在多个slice/entry中重复。`completion_bumps`累加**实际出现次数**，不是把counter ID去重后各加一次；否则依赖threshold可能永远不到。这是异构替换图节点时很容易漏掉的调度合同。

以上是静态实现/控制流观察，不是本轮故障注入结果；GPU失败、reduction失败和private/runtime内部状态仍需分别确认。

## 5. 公共 Core ML、prepared I/O 与实际backing

所读ANE入口构造公开NeuralNetwork格式`.mlmodel`，通过CoreML compile/load并显式设置compute units；不是把任意MSL交给ANE，也不能据module注释“only through CoreML”否定别的私有研究路径。

`PreparedNet`保有arrays/provider/options，初始化时检查contiguous分配。predict返回后检查实际shape、dtype、stride与dataPointer：采用目标backing则计hit；没有采用则按实际strides复制；同pointer但stride变化则拒绝。比只传`outputBackings`或比count更强。

不过，同文件的旧`run_timed`还有按声明F32/连续输出直接memcpy的路径。不能因为PreparedNet较严谨就宣布整个项目所有CoreML I/O都具备同样检查。prepared backing也只证明外部alias，不证明runtime内部无staging。

channel编译cache绑定完整weight/scales、shape、encoding、OS信息和graph生成源码hash，区别于仅采样权重前1KiB的key。编译锁与rename保护并发资产发布，但存在cache目录不等于内容已经验收。

placement预检要求inspector输出preferred ANE；它仍是plan证据，不是物理逐op计数或真实GPU/ANE overlap测量。

## 6. 数值配方要跟着数据流走

Matrix bridge把原BF16/FP8/MXFP4源解码后转FP16常量，GPU支可能继续消费packed源；这改变有限精度路径，不能称原模型bitwise不变。RNE BF16恢复、输入headroom、累加顺序和LoRA需独立质量验证。

row lane把residual缩放为原来的1/16，o_proj/down也配套缩放，并将RMSNorm epsilon乘以该scale的平方。若只缩放输入却不改epsilon，近零激活并不满足所谓norm尺度不变。这个正确的epsilon补偿是可迁移技巧，不应仅记“除16不会溢出”。

## 7. 迁移与测试建议

优先迁移完整span替换合同、保留输入的partial、join后fallback、counter multiplicity、prepared backing验证与内容绑定cache。persistent interpreter另做资源/进展/可见性探针，不为了减少launch一次塞入无限工作。

设备对比应同时包含优化GPU packet、逐op诊断、row lanes、channel MLP；分别记录compile/load、host转换、ANE预测、GPU等待、join、完整请求和quality。某个span收益不能覆盖新的segment开销，更不能继承CUDA/AMD或其他Apple型号的数字。
