# whisper.cpp：Core ML阶段交接与真实tiny.en转录

<!-- knowledge
id: technique-gpu-ane-case-studies-whisper-coreml-metal-stages-and-real-asr
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [本轮摘要](../../MAINTENANCE.md) · [来源/构建/参数](../sources.md) · 转录与计时 · 计划 · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，路线图X03。固定whisper.cpp `0c7be696371069e0baa2515ed7aa08d5a5473aa6`、tree `c4fcfb1888781372e7e5aa3d71f58a2e2769dd27`。本轮完整官方归档2039个Git blob实验前后核对，16个文本文件选段＋1个实际音频资产，不声称全仓库全文阅读或永久最新release。[S1]

本机M4 Max / macOS26.6.2 build25G83 / SDK26.5，执行原始Whisper encoder→decoder完整API：官方tiny.en权重、配套Core ML encoder、11秒JFK音频，**24主请求＋2 pilot**。在本轮固定资产和参数下，Metal-enabled GGML的warm请求中位数范围为33.92–35.21ms，Core ML encoder路线为41.44–43.92ms；两路文本和token ID相同。**不能据此认定物理ANE更慢/更快、已有GPU/ANE重叠，或所有Whisper模型如此。**实际数据

## 1. 一个稳定encoder替换点，不是设备并行开关

原调用依赖如下。`whisper_full`包含host log-mel、模型和sampling；Core ML分支同步返回后，才进入GGML cross和decoder。[S2][S3][S4]

```text
PCM FP32 → CPU log-mel / mel staging
         → 原GGML conv+encoder，或同步Core ML encoder
         → embedding → GGML cross-attention K/V投影
         → GGML token decode + host sampling → text/token results
```

Core ML wrapper借用FP32 mel指针构造`MLMultiArray`，配置为`MLComputeUnitsAll`，调用同步prediction后`memcpy`返回output。All允许CPU/GPU/ANE选择，不强制ANE；借用输入也不证明Core ML内部zero-copy。compiled资产MIL输出声明`[1,1500,384]` FP32，本轮没有截获真实返回stride或验证整个encoder中间tensor；已知adapter的dense输出假设不能仅因本次转录通过就升级为通用正确性。[S2][资产](../sources.md)

GGML的同步graph helper调用async后再`synchronize`所有backend；encoder返回前还完成cross投影。因此这条正常单请求链不是“encoder ANE与同一请求decoder GPU同时跑”。`whisper_full_parallel`另外拆音频、创建额外state并起host线程，具有分段边界和质量风险；它不自动成为有界多请求异构流水。本轮没有执行parallel API，也没有做overlap trace。[S3][S4]

原source将GGML权重整体加载，再创建可能含Core ML模型的state；启用Core ML并不自动删除GGML中的67份encoder参数。外部encoder跳过native encoder allocator，日志workspace会变，但Core ML另有模型/缓存，不能据此宣称总峰值内存或驻留下降。[S3]

## 2. 构建与资产身份

两个新输出构建链接同一未修改上游库：CoreML OFF与ON，均Metal ON、embedded Metal source ON、Accelerate ON、原context默认flash attention ON。为可控构建设`GGML_NATIVE/OPENMP/CCACHE=OFF`；这不是所有机器最优配置或默认CPU基线。ON版本将`WHISPER_COREML_ALLOW_FALLBACK=OFF`，实际日志确认Core ML模型加载成功，未静默退回native encoder。[构建证据](../sources.md)

GGML记录MTL0/Apple M4 Max，并保留BLAS/CPU辅助backend；**Metal-enabled不是每个算子都在GPU的trace证明**。未设置私有ANEForge环境，客户端发现相关环境就拒绝；未调用私有ANEForge、改设备策略、装依赖/SDK或修改参考/产品。上游AGENTS允许学习探索，本轮无上游改动、PR或提交。[S1][S3]

官方模型仓库固定revision `5359861c739e955e79d9a303bcbc70fb988958b1`，而非可变main。[S5]

| 资产 | 大小 / SHA-256 |
| --- | --- |
| `ggml-tiny.en.bin` | 77,704,715 bytes；`921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f` |
| `ggml-tiny.en-encoder.mlmodelc.zip` | 15,034,655 bytes；`82b32eef73c94bb0c432a776a047b757d9525c26d84038a15d8798d7c8d1ee58` |
| 原`samples/jfk.wav` | 176000帧、mono PCM16/16kHz、11秒；SHA见[资产清单](../sources.md) |

音频逐sample按`int16/32768`转FP32，所有样本在计时前加载。模型下载、WAV解码/转换、JSON/文本写出、进程启动、GPU准入均不在`full_api_wall_ms`中，**该数值不是完整应用end-to-end或TTFT**。客户端

zip内部时间戳为2023-06-16；metadata为Float16 storage、Mixed(Float16/Float32/Int32) compute。未运行当前`--optimize-ane True` converter，不能把固定发布资产的结果当当前导出流程/ANE优化上限。现行download-coreml脚本直接声明不维护并exit1，未执行；本轮使用固定公开资产URL、大小和LFS SHA核验。[S6][S5]

## 3. 不只凭同名：67份参数的CPU字节比对

原创CPU parser遍历完整GGML header/filters/vocab/tensor记录，取得67份encoder参数；将其按FP16转换后的payload hash和元素数做multiset。另一边读取MIL `BLOBFILE`声明，按blob metadata定位真实数据；71个FP16常量中完整消耗这67份参数，剩余4个均是全零key bias。CPU解析器[逐参数证据](../sources.md)

这比同名资产强，但只证明**learned constant multiset在FP16转换后相同**，不证明每个role连线、converter commit、算子/LayerNorm/GELU精度、图融合或encoder输出逐位一致。原GGML部分向量参数仍FP32，Core ML有FP16常量和不同图配方；不能把两路定义成完全相同的有限精度算术。[参数边界](../sources.md)

## 4. 真实请求协议与结果

每进程一个真实context；`greedy/best_of=1/T=0/temperature_inc=0`，4线程、English、no_context=true、VAD off、timestamps off，其他解码设置沿原default params。`whisper_full`每次清结果、重做mel、清prompt history并重置decoder RNG；本轮没有文本缓存或温度重试混入对照。[S3]客户端

两次pilot各1请求；primary进程外层为Metal-a→CoreML-a→CoreML-b→Metal-b，各6请求，iteration0为本context首请求，其余5次warm。每个新进程重新做五次GPU利用率准入，6进程共30次采样均0%，实际owned子进程全部terminal0。粗粒度准入不能排除之后的短暂外部活动或CPU/热状态变化。数据

| 独立进程 | init API wall | 本context首请求 | 5次warm中位数 |
| --- | --- | --- | --- |
| Metal-a | 61.40ms | 41.07ms | 33.92ms |
| CoreML-a | 90.12ms | 61.63ms | 41.44ms |
| CoreML-b | 90.14ms | 61.10ms | 43.92ms |
| Metal-b | 60.84ms | 50.24ms | 35.21ms |

这是**分进程的两个窗口范围，不是同窗口paired样本或有置信区间的全局排名**。pilot中Metal首请求316ms、Core ML首请求57ms包含不同首次shader/cache历史，不能挑这对数算5.5倍。pilot的init还分别约2.53/2.85秒；主试开始前系统/driver cache已经可能warm，不能把表中的init叫机器冷启动，也没有清全局缓存。完整未筛选样本

26次实际转录的原文本和25个token ID序列完全一致（22个词＋标点/EOT），与固定README示例做word-normalized比较为0 edits；不是独立人工标注数据集或通用ASR质量资格。匹配token位置的probability最大差异约0.00735，所以**文本一致不等于数值逐位相同，也不能用probability较高宣称更准确**。[S7]数据

本轮反复使用同一音频，不能凭重复一致证明任意新音频/本代encoder输出freshness；没有故障注入或变化输入的旧代检测。timestamps明确关闭，API返回的负`t0`保留但不解释为有效音频对齐，也不称默认timestamp功能已复现错误。数据

## 5. 原timer的分母为什么会误导

`t_encode_us`从mel padding/tensor set前开始，包含Core ML预测或native encoder及**GGML cross K/V投影**；不是纯ANE、纯Core ML或GPU kernel time。源码的`wctx.t_load_us`在权重加载结束记录，早于部分state/backend/Core ML准备；不能用日志约30ms load替代外部init wall。[S3]

本次每请求原日志为encode1、decode24、batchd2、sample1、prompt1。含义不同：[S3]原计数

- decode：单token graph调用次数；实际结果含25个token，不是与结果token数恒等。
- batchd：源码对小batch按`n_tokens`累计，2不是“两批推理”；prompt大batch同样按token数计。
- sample：源码在非空beam候选处递增；本次greedy日志的1来自`max(1,n_sample)`口径，不能将约5ms解释成每个输出token采样成本。
- `whisper_get_timings`给各自分母下的average；本库同时解析原total/denominator，不能把不同average简单堆成完整请求。host chrono独立记录API完成，写结果和`print_timings`在区外。

`reset_timings`没有重置所有fallback统计；本批原日志p/h均0，但这不等于所有模型/错误场景永远无重试。源码默认Core ML允许策略All也不能直接当物理设备counter。[S2][S3]

## 6. M4的Tensor API日志是软件选路，不是硬件缺失证明

实际日志同时见`MTLGPUFamilyMetal4`与`has tensor=false`。固定GGML实现先按family探测，再基于设备名/环境选择默认关闭旧芯片tensor route，注释称其当前实现在M4无显著收益，并在启用时再验证编译。**M4的false不是Metal4/Tensor API不可用，更不是M5 NAX实测**；本轮未改这些环境开关、强开tensor或使用M5硬件。[S8]

## 7. 演进、失败和迁移边界

[#566](https://github.com/ggml-org/whisper.cpp/pull/566)官方API核为2023-04-15T10:21:28Z merged，merge `5e47e223bd74af8a16cd4f2900cfbcfb92cd1f18`；15个changed paths，完整阅读返回的encoder wrapper、whisper.cpp和generate脚本三份hunk，其余不作全文读本。初始wrapper没有今天显式All配置，固定80×3000与encoder替换接缝值得学习；本轮未做PR parent/head或作者CPU倍率复现。[来源](../sources.md)

上游init失败回退只受编译macro控制，不是运行中prediction失败的完整GPU重算协议。当前wrapper仍忽略prediction error并平copy输出，本批真实资产成功不覆盖其它shape/stride/dtype、nil、短copy、取消或device drain。额外state可能涉及额外Core ML model load；不能只见parallel/async方法就声称吞吐或单请求延迟获益。[S2][S3]

前置准备也保留负结果：系统Python3.9不支持safe tar filter，切既有3.11后才解包/核2039blob；两次提前下载因目标目录缺失exit56，后续完整资产hash通过。初版capture错误绑定自己的running receipt/stdout，已保存初版JSON并只重新curate terminal日志，未重跑或改写24个计时值。数据

磁盘约余12GB时设置本轮512MiB预算、至少10GiB余量；当前原始输出约250MB量级，完整源码、模型、编译物只在ignored outputs，没有删用户文件、装依赖或清系统缓存。总峰值内存、功耗、物理ANE/overlap和一般产品质量均未测。计划

向视频模型迁移，应借稳定封闭stage、完整handoff/状态/失败协议，而非照搬tiny.en速度。下一步需变化输入freshness、多音频/语言/噪声、当前导出配方、冷/热资源与物理trace；真正多请求流水还需有界队列、独立state和backpressure。整体研究保持active。

## 固定源码与资产入口

- [S1：固定官方树/AGENTS/许可](https://github.com/ggml-org/whisper.cpp/tree/0c7be696371069e0baa2515ed7aa08d5a5473aa6)
- [S2：Core ML bridge](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/src/coreml/whisper-encoder.mm)；[实际同步generated wrapper](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/src/coreml/whisper-encoder-impl.m#L173)
- [S3：完整引擎与stage/timer](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/src/whisper.cpp#L2402)
- [S4：GGML同步完成](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/ggml/src/ggml-backend.cpp#L2167)；[公开API线程/state边界](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/include/whisper.h#L600)
- [S5：固定模型revision](https://huggingface.co/ggerganov/whisper.cpp/tree/5359861c739e955e79d9a303bcbc70fb988958b1)
- [S6：当前生成脚本](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/models/generate-coreml-model.sh)；[未维护下载脚本](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/models/download-coreml-model.sh)
- [S7：固定音频示例文本](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/README.md#L692)
- [S8：GGML tensor-route软件策略](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/ggml/src/ggml-metal/ggml-metal-device.m#L1146)

原创标准库合同/专项测试不导入框架、模型或GPU。重测完整模型必须经空闲准入，原始artifact root为`outputs/gpu-ane-whisper-stages-round32-20261017`。整理JSON不分发权重或第三方源文件；[验证范围（过程原件已移出）](../../MAINTENANCE.md)区分实测、源码推断与未取得资格。
