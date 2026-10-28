# ccv原始组件运行：GPU、CoreML rowwise与固定行split的同输入验证

<!-- knowledge
id: technique-gpu-ane-case-studies-ccv-original-rowwise-component-runtime
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [前轮调用链](drawthings-to-ccv-hybrid-route.md) · [成本/失败合同](ccv-hybrid-cost-and-failure-boundaries.md) · [来源](../sources.md#ccv-runtime)

2026-10-17，M4 Max、macOS26.6.2/25G83、SDK26.5。固定ccv `2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b`。本轮已经从假设成本的CPU selector，推进到**未改上游源码的实际MFA GPU/CoreML/fixed-split组件**；不是运行Draw Things、s4nnc、NNC graph dispatcher或真实checkpoint。

## 1. 先核整套来源，不能只对上一个cpp

发现另一项研究留下的构建，其中rowwise cpp与固定ref相同；但只读核查可用1,640个源码blob后，有49个文件与目标ref不同（包含公开头文件），另外缺828份。差异不能无证据归咎于用户修改或上游错误；也不能凭一个相同文件把514个旧C对象当作精确ref的原始执行。

因此本轮**没有复用那批对象**，没有改它的源码/构建。另取官方固定归档，107,805,644bytes、SHA256见[来源](../sources.md#ccv-runtime)，全部2,468个原文件核canonical Git tree；原始[Package.swift](https://github.com/liuliu/ccv/blob/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b/Package.swift)与数值/Metal/CoreML实现均不改。全树字节核验不是全树人工阅读；14个选定文件另记范围。

自有SwiftPM C++ executable只依赖原始ccv的C/Objective-C++产品。构建、module cache、配置和显式CoreML临时目录均放本次ignored输出；没有安装系统依赖、改参考checkout、生产默认后端、Metal验证层或服务。最初manifest执行遭nested sandbox拒绝，保留失败receipt，获准后正常构建，没有通过禁用内部检查或补写上游函数解决。

构建使用Release加`CCV_METAL_LOGGING_ENABLE=1`，client设置本进程context log_level1。这是便于诊断的构建条件，**不是默认发行配置的性能复现**；两条原始exclude指向未生成文件的warning也保留。

## 2. 三条路径都是真的原组件，但层次要说清

本库client在实际context报告M4 Max/MFA支持、ANE项目policy支持、GPU NAX不可用后，固定`use_neural_accelerators=0`，不强开M5路径：

| arm | 实际调用 | status含义 |
| --- | --- | --- |
| gpu | 原`encode_scaled_gemm`，Apple9 dense Int8MatMul/FP16 direct activation路径 | API本身void；日志的1是client完成标签，不伪装成原返回值 |
| coreml | 原`run_ane_rowwise_gemm`，含GPU准备、CoreML预测、GPU恢复 | 1是原组件返回成功 |
| split | 原`run_ane_na_rowwise_split_gemm`的非NAX分支，显式256 ANE行/256 GPU行 | 1是原组件成功；原日志确认单tile固定行切分 |

没有调用真实cost chooser，所以不会隐式启动前轮的4096×4096校准。固定split用于验证接缝，不声称当前成本模型会自动选择这个比例。也没有经过GEMM backend的最终tensor subtype/flags资格判定；rowwise packet由client按已读格式明确构造。

每条路径结束显式commit并调用原stream wait，再读自有Shared Metal output。destination先poison为FP16 NaN，**不是**预锁定/poison CoreML pixel buffer。输入和输出buffer由client持有到GPU完成；原cache/框架生命周期仍是另一个层次，不能从wait、free或进程结束推导物理ANE drain。

## 3. 为什么这个数值fixture能精确反证错误交接

固定M512/N256/K256、batch1、contiguous、offset0、无bias/H256/LoRA，输入/输出和row scales均FP16。

- 每row activation是整数code除128，且首元素强制127/128，令max_abs固定；原[融合准备kernel](https://github.com/liuliu/ccv/blob/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b/lib/nnc/mfa/kernels/ANERowwiseTransformKernel.cpp#L225)保存max/127的IO精度scale后rint/clamp，本例可以精确量化。
- W是明确的signed-INT8 rowwise packet，每output channel只有8个±1非零code；scale交替1/64、1/128，位于128-byte对齐的code区之后。不是从Q8P/GGUF或真实checkpoint借来另一套量化数学。
- 受限整数和中间结果使FP16可精确表示；最终正确答案不是全0。两seed同时改变输入和稀疏位置，而同seed的三arm使用**完全相同**原始字节。

这个设计适合检查权重/scale偏移、转置、row位置、旧slot和未写输出；同时它刻意避开一般FP16 activation的量化误差。**本例误差0不证明ANE额外activation量化对真实模型无损。**

## 4. 实际结果与全量独立复算

两个串行进程分别跑pilot及两seed主确认，均先经连续5次一秒间隔GPU占用采样，全部0%，threshold5%，owned child均terminal/exit0。粗采样不证明系统级独占或所有短时外部活动。

主6份输出，每份131,072个FP16值，共**786,432个actual值、1,572,864bytes**。curator不信任native oracle：先逐字节验证保存的activation/weight packet，重新解出所有code/scale，再用独立Python FP64 sparse-dot复算完整M×N；三条路径所有值finite、最大误差均0，且每seed跨arm bitwise相同。预声明诊断门槛`2e-5`未改。

pilot另3份/393,216actual值/786,432bytes，不能混入主计数。完整input、原始packet、native FP32 oracle、actual FP16数组和独立数学记录都留ignored outputs；发布数据保存每个完整数组的SHA，离线测试还能从确定输入重建**完整expected FP16 payload hash**，不是只验证几个样本。

没有计时、测功耗或比较GPU/ANE速度；source/build/supervisor elapsed包含框架、编译、load和IO，不能当推理延迟或倍率。

## 5. 真正跨过的边界与仍未证明的部分

这次运行确实跨过原始activation量化/转置、权重上传、CoreML load/predict、原split提交顺序和GPU output restore，而非CPU stand-in。coreml与split的返回1及完整数值，使这些接口在该fixture/版本上获得了实际功能证据。

但原bridge请求的是`CPUAndNeuralEngine`，允许CPU；Apple9 policy或函数名字都不是物理放置。**没有compute-plan/instruction/ANE trace、GPU+ANE overlap或zero-copy identity证明**，不把成功API路径写成“ANE-only实测”或“异构提速”。

原[compile_program](https://github.com/liuliu/ccv/blob/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b/lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_coreml.mm#L849)手工生成`.mlmodelc`，加载后即移除自己的临时目录、保留MLModel对象。因此运行后目录为空不表示没加载，也不表示model已释放。本轮没有拦截/归档这些生成字节或系统内部compiled artifact，不能拿源码hash当其字节attestation。

原输出backing仍只检查可读与count/bytes；本次数值匹配支持**这组完整接缝结果正确**，不替代实际returned pointer/stride/IOSurface身份检查，更不证明driver没有staging。当前pipeline的已加载model cache、surface复用和stream wait也不等于取消、错误恢复、跨线程或长跑内存资格。

### stream对象不是独立queue

[MPS实现](https://github.com/liuliu/ccv/blob/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b/lib/nnc/mps/ccv_nnc_mps.m#L47)默认使用共享execution state；只有显式fork才分离本线程的pending command/cache。本client没有fork、没有跨线程，创建stream不代表创建第二GPU queue。GPU前缀与CoreML后缀的行切分设计需device trace才能证实物理重叠，不能仅从两个API调用区间相交得出结论。

## 重放与后续

21项离线测试和[源/执行验证（过程原件已移出）](../../MAINTENANCE.md)不需要SwiftPM、CoreML、GPU或raw outputs。设备重放须恢复来源匹配的原始构建，用全新输出/TMPDIR并先过`guarded_run.py`；client只接受`pilot/full`，不直接调用真实校准chooser。完整构建/准入argv与receipt在。

后续重点是任意activation分布与尺度、bias/H256/stride/batch/更多M/N/K、实际GEMM dispatcher/权重loader、LoRA和完整consumer质量；再加真实chooser的cold/warm校准、物理ANE/带宽trace以及同条件端到端timing。不能把本例稀疏dyadic绿灯推广到一般模型或全局引擎排名。
