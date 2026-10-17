# Espresso源码纵切：Swift所有权、surface重绑定与大词表输出头

<!-- knowledge
id: technique-ane-espresso-runtime-walkthrough
type: implementation
operator: gemm, layout, scheduling
backend: private-ane, metal, cpu-reference
device: cpu, gpu, ane
symptom: precision-loss, lifetime, measurement
semantics: contract-only
representation: fp16, fp32, iosurface
boundary: producer-consumer
api: reference-only
-->

[ANE目录](README.md) · [本轮来源](sources.md#implementation-sources-20261016) · 辅助路径实测

新增研究项目：[christopherkarani/Espresso@05ea6a3da91be642abd3d4741c0341a941ce1064](https://github.com/christopherkarani/Espresso/tree/05ea6a3da91be642abd3d4741c0341a941ce1064)，MIT，Copyright2026 Christopher Karani。已在独立references子目录做稀疏clone；上游工作树未改。阅读重点不是README速度表，而是生成器→kernel→C bridge→surface→consumer的调用链。

这里的Espresso是开源Swift项目，不是macOS私有Espresso.framework；后者是[ANEForge e5rt路线](resident-training-lab.md)的系统依赖，不能因同名就混用API/版本或实验结果。

## 1. 一个Swift kernel究竟拥有什么

`MILGenerator / ANEBuilder → ANEKernel.init → ane_interop_compile → _ANEInMemoryModel compile/load → _ANERequest`。主要源码：[ANEKernel.swift](https://github.com/christopherkarani/Espresso/blob/05ea6a3da91be642abd3d4741c0341a941ce1064/Sources/ANERuntime/ANEKernel.swift)、[ane_interop.m](https://github.com/christopherkarani/Espresso/blob/05ea6a3da91be642abd3d4741c0341a941ce1064/Sources/ANEInterop/ane_interop.m)。

Swift `~Copyable` 禁止隐式复制 kernel 句柄，`deinit`调用native free；typed throws把编译、surface和执行失败分开。这个类型技巧降低重复释放/跨任务误复制风险，但不自动证明裸指针、所有Sendable标记或设备异步生命周期安全。

`inputSurface/outputSurface` 使用 C 的 copy getter，主动CFRetain surface，因此它们可独立于kernel持有；**surface存活不等于其中数据仍属某一次prediction**，generation仍要跟踪。

`HybridDecodeSurfaceHandles`在kernel set附近保存“哪个端口是Q/K/V/residual/FFN”的知识：projection residual绑定到原QKV输入，FFN输入绑定到projection输出。将端口布局放在其producer旁，比让模型每处重复硬编码索引更可维护；GQA的kvDim也不假定等于dim。

## 2. rebind不是一个无条件零拷贝开关

native `ane_interop_rebind_input`先检查index和allocation容量，构建包含新surface的request，成功后再retain新surface、替换旧request并释放旧owner。原绑定失败时保留，避免先释放旧资源再尝试新请求。

Swift接口明确要求不能与eval或其他mutations并发。allocation足够只证明容量，不证明shape、channel/spatial pitch和dtype吻合。绑定多个kernel可省中间host copy，但每个kernel eval、锁、格式转换和同步仍可能存在。

[SurfaceIO](https://github.com/christopherkarani/Espresso/blob/05ea6a3da91be642abd3d4741c0341a941ce1064/Sources/ANETypes/SurfaceIO.swift)对batch读/写/拷贝合并锁；底层仍有FP16↔FP32转换与memmove。不要把“NEON-vectorized read”统称成没有数据移动。所读部分按逻辑 `[C,S]` 连续元素索引，不能替代真实ANE端口pitch检测。

## 3. 设备入口、fallback与profiling

`ANE_EVAL_PATH`可选inmem/client/clientDirect/realtime；所读C实现会在某些路径不可用或失败时退到同步in-memory evaluate。因此指定realtime不等于本次真实使用realtime，计时必须保存actual route而非只有请求选项。

`ANEKernel.eval`记录host wall；另读取 `_ANEPerformanceStats.hwExecutionTime`。disabled/unsupported返回0，在研究归一化层应标 unavailable，而不是0ns硬件执行。bridge会尝试request附带的stats、statsArray或原stats对象，但仍需验证单位/生命周期和payload真实性。**本轮未运行这些私有profiling/virtual-client探针**。

load选项还有keep-wired、power saving、skip prepare、IO fences等开关；源码存在不等于应启用。本轮不触发这些选项，不用关闭栅栏或改变电源策略制造成绩。公共工具优先级仍见[profiling指南](profiling.md)。

## 4. 训练的关键不是Swift重写，而是捕获资源的方式

[GradientAccumulator](https://github.com/christopherkarani/Espresso/blob/05ea6a3da91be642abd3d4741c0341a941ce1064/Sources/Espresso/GradientAccumulator.swift)使用serial DispatchQueue+group，让CPU dW与ANE dX重叠。BackwardPass在enqueue前将会被复用的scratch复制到move-only SendableBuffer，再由异步BLAS闭包独占捕获。

可迁移技巧是：**读者比producer活得久，就持有独立快照或明确owner**；不能仅把裸指针包装为Sendable。CPU梯度目标buffer、optimizer更新和后续重用仍须barrier。复制这些activation也有可见成本，不能在“纯ANE训练”标题下省略。

本轮没有运行Espresso的完整trainer；新训练实测来自另一个固定ANEForge实现，见[驻留训练实验](resident-training-lab.md)，两者不混挂性能资格。

## 5. 大LM head为何可能离开ANE

[ClassifierStrategy](https://github.com/christopherkarani/Espresso/blob/05ea6a3da91be642abd3d4741c0341a941ce1064/Sources/RealModelInference/ClassifierStrategy.swift)按vocab×dim、是否有FP32 sidecar及架构选择ANE、CPU分区/FP16 tiled或Metal FP16 GEMV。其16M-element阈值是软件政策且有Stories allowlist，不是Apple公开SRAM规格。

其`usesCPUExactClassifier`在所读实现里等价于“非ANE”，也会包含Metal策略，因此不是CPU设备放置证据；应继续看actual backend label与真正调用点，不能只解析一个布尔字段。

[MetalLMHeadArgmax](https://github.com/christopherkarani/Espresso/blob/05ea6a3da91be642abd3d4741c0341a941ce1064/Sources/Espresso/MetalLMHeadArgmax.swift)把GEMV和greedy argmax合并：第一阶段按词表row算dot并输出每组(max,index)，第二阶段归约为一个UInt32 token，避免物化/回读整个logit向量。权重只在初始化复制到shared Metal buffer，hidden每次仍memcpy并commit/wait。

它约束dim为256的倍数、使用FP32累加，tie取最小index。**这是greedy-only边界**；temperature/top-p、repeat penalty、logit bias等需要额外语义，不能任意沿用只返回argmax的接口。

### 原实现的有界本机对照

仅编译原 `neon_convert.c`、原Metal head Swift文件及原创client，未构建私有ANEInterop.m、未执行SPM/demo准备脚本。三个shape、每臂first/warmup3+timed10，共78次token；输入/系数为精确dyadic、包含零输入和重复row的tie、最后不满16-row group。全部与独立Swift Double dot/first-max一致。

| vocab×dim | CPU NEON中位ms | Metal完整API中位ms | 解释 |
| --- | ---: | ---: | --- |
| 17×256 | 0.000271 | 0.143834 | 小头由提交成本主导 |
| 513×256 | 0.007500 | 0.136771 | 此范围CPU更合算 |
| 4097×512 | 0.132480 | 0.153980 | 已接近，仍不能预测467MB真实head |

是warm resident合成组件，不是全模型、DRAM带宽或最优CPU/GPU排行榜。未测大词表、非greedy、冷加载、峰值或能耗；不从小矩阵给新策略永久判负。

随后用同一个已hash的binary在新进程再次运行，同样78次token全部正确、五个索引控制的观察相同；两批合计156次。它是相同确定性工作负载的进程重复，不冒充新checkpoint/独立seed或更大词表验证；上表保留首批计时，不挑两批最小值。

独立标准库复核还重建了全部系数bytes/输入hash与156个token reference。它利用这组**明确披露的17类重复row**计算reference，同时检查完整权重hash；这不是抽样验证真实LLM系数，也不会替新权重或近似累加策略取得资格。

### 实际复现的helper索引边界

连续 `ane_interop_neon_argmax_f16` 用uint16 SIMD索引；count65544时，峰值index65536/65543分别返回0/7，value仍正确为1；count65536的末元素、以及65545的scalar tail控制正确。见JSON中的五个完整控制。

这不等于serving head出错：所读spatial调用用的是保留int索引的strided helper，Metal head用UInt32，本轮CPU streaming GEMV也用int。应把该连续helper限于已验证范围，或改用更宽/分块加全局offset的索引策略；**本任务只记录，没有修改上游或产品实现**。

## 6. 不继承的README结论

该固定README平台表将M4写为38-core、M3写为18-core，与官方M4 Pro/Max的16-core资料不符；不纳入硬件矩阵。其最快表格又涉及recurrent fused的local/demo artifact。`LegacyGraphSupport.recurrentLayer`实际是input/state投影、sigmoid carry与状态反馈，并非自动保持任意Transformer attention/KV语义。

要迁移“多层融合”的系统技巧，应保留原模型数学再测dispatch；若改成recurrent/student，必须重新训练并验任务质量，不能叫原Transformer的无损kernel加速。README本身也区分demo与预训练serving范围，应沿具体artifact验证。

## 本地学习与复查

CPU宽度/状态合同可独立阅读；原实现的显式可选构建入口是 espresso_head_probe.py。新SoC/SDK、source、dtype、维度/词表、greedy策略或调用路由变化时重验，不能将本轮辅助路径结果授予完整Espresso或ANE性能资格。
