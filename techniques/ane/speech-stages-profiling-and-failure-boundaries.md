# 语音阶段不是一个ANE开关：Whisper调用链、输出布局、失败与计时

<!-- knowledge
id: technique-ane-speech-stages-profiling-and-failure-boundaries
type: method
stage: encoder, decode, pipeline
operator: attention, layout, scheduling
backend: coreml, cpu-reference
device: cpu, gpu, ane
symptom: wrong-layout, stale-cache, lifetime, measurement, compile-bound
semantics: contract-only
representation: fp16, fp32
boundary: producer-consumer, request, weight-runtime
api: public, private, reference-only
-->

[ANE目录](README.md) · [33文件固定来源](sources.md#speech-sources-20261017) · [验证与后续队列（过程原件已移出）](../MAINTENANCE.md) · [公共部署概览](public-coreml-deployment.md)

2026-10-17批次，M4 Max / macOS26.6.2 build25G83、SDK26.5、clang21。固定whisper.cpp `0c7be696`、WhisperKit/Argmax `9f99b5cb`、ANEForge ABI `896c4943`；不是最新版声明。Kit在该commit的package名已为`argmax-oss-swift`，但`WhisperKit` target和旧组件协议仍存在，不能把同树另一套新ModelManager行为自动套上去。

核心结果：**实际小型CoreML输出正确，不代表原bridge平拷贝正确；dispatch返回码没有传回调用者，也不能因encode返回就认为本代输出有效。** 本批16次真实小模型prediction有8次窄布局copy错误；22个CPU mock场景中保留12份finite但错误的目的数组。没有执行Whisper checkpoint、转录音频、真实私有dispatch或性能排名。

## 可复用的方法：先 offload 稳定阶段，再接完整语音请求

1. 选择输入输出合同稳定的 encoder 边界，保留既有 decoder，并显式处理输出 stride 与 owner。
2. 区分组件策略、prewarm、实际 prediction 和外层状态；私有 adapter 的 ready、返回码与 freshness 分别验收。
3. 计时包含边界转换及真实消费者，失败重试保留状态语义；load-only 或 mock 不能替代完整 ASR。

可用于设计分阶段加速；下文组件与故障实验不提供端到端语音提速或识别质量结论。

## 1. 三条路径，阶段与owner不同

| 路径 | encoder之前/之后 | IO与模型生命周期 | 能确认与不能确认 |
| --- | --- | --- | --- |
| whisper.cpp公共CoreML encoder | 原引擎mel准备→外部encoder→GGML cross投影→decoder | 借用FP32 mel指针；init load一次；同步prediction；输出copy | 固定原wrapper源码及自有小模型可验；ALL允许CPU/GPU/ANE，不保证ANE |
| whisper.cpp ANEForge encoder | 替换同一external encoder接缝，其他引擎步骤仍在 | `ports.txt`/MIL/pos资产；动态库5个C符号；FP32↔FP16 staging；pos绑定一次 | 原wrapper+原创CPU动态库可验错误传播；mask4是请求意图，不是本批私有ANE执行 |
| WhisperKit/Argmax | 每窗口await mel→encoder→token loop；decoder带显式KV特征 | 每组件独立computeUnits；公开CoreML异步预测/加载；组件和外层状态分开 | 原声明/自有模型load实测，加所选源码调用链；没有运行完整包、tokenizer或ASR模型 |

固定[外部encoder调用链](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/src/whisper.cpp#L2402)跳过原conv/encoder计算，但并非整个语音请求都在ANE。`ctx_aneforge`存在时优先于公共CoreML。启用外部路径还会跳过原encoder调度器的分配，因此运行中“失败后直接改跑GPU”不是现有代码保证，需资源准备、状态恢复和错误协议设计。

公共CoreML init失败的GPU回退取决于`WHISPER_COREML_ALLOW_FALLBACK`编译宏；显式ANEForge encoder init失败在外层state创建中返回失败。两者都不能泛化成自动Metal回退。它们和WhisperKit低质量/温度重试的**解码策略fallback**不是同一种事。

## 2. 公共bridge：借用输入没有解决返回输出的stride

[原`whisper-encoder.mm`](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/src/coreml/whisper-encoder.mm)约定：FP32输入`[1,n_mel,n_ctx]`、dense strides、deallocator=nil；config为`MLComputeUnitsAll`。同步调用generated wrapper，`error:nil`；最后按`count*sizeof(float)`从输出`dataPointer`做`memcpy`，没有按输出stride收集，也没有显式检查dtype、目标capacity或错误。

这不是“MLMultiArray一定dense”的接口。输入确实alias传入指针，只能证明这层包装借用成功，不能证明CoreML内部零拷贝、返回dense或整个路径没有CPU工作。头文件的80×3000注释也不是运行时shape验证。

原创fixture生成两个没有真实权重的FP32模型：`output=2*logmel_data+.25`，shape分别`[1,4,8]`和`[1,4,16]`。每seed17/29、每width前三代真正调用原prediction；观察器只记录输入和**真实返回tensor**的shape/stride及完整logical值，方法IMP随后恢复。没有把观测输出重新送入原图。

| 实际模型 | 实际返回stride | captured logical输出 | 原平copy，主试与无observer确认 |
| --- | --- | --- | --- |
| `[1,4,8]` | `[64,16,1]` | 6/6完整等于独立FP32公式 | 6次observed+2次unobserved均错误；每次24/32位置不符 |
| `[1,4,16]` | `[64,16,1]` | 6/6完整等于独立FP32公式 | 6次observed+2次unobserved完整正确 |

窄模型前32个physical float包含row0、padding、row1、padding，省掉后两逻辑row。其中physical16–23实际是logical8–15，却被放到目的16–23；**仅这8个已知logical错位就足以证伪**，不需要解释padding具体值。无observer进程不读取返回stride，独立确认的只是最终目的数组错误，不能声称其stride也被直接测量。

原始padding没有初始化语义，可能包含任意旧内存。ignored raw只留本机；公开证据删去这些错误目的数组的hex，保留SHA、元素数、raw mismatch/nonfinite计数与已知坐标错位。portable tests可重放logical tensor和坐标映射，**不能从hash重建redacted padding或独立重算其24个raw mismatch**。本批raw审计先核对了完整原数组，再做redaction；不直接分发整个raw parent report。

我们另编译不含method interception/swizzling的新client，仍链接原封未动的bridge；它在同两个seed/两width上各跑一代，再现窄失败、dense成功。因此结果不只依赖observer。另有4次owned padded-view控制：pitch=`width+2`、padding显式42.25，原copy错而stride-aware控制完整正确；4次注入exception只由外层harness捕获，目的数组仍为−91.5。后者不是实际CoreML错误复现。

边界：**这发现的是固定adapter依赖dense输出的风险及合法自有资产反例，不是已测到发布版Whisper模型转录错误。** Whisper资产的encoder维度可能对齐、布局不同；本批未跑它。没有试验nil prediction、错误dtype、超大输出或越界写入。4个缺失资产init检查均NULL，只覆盖factory拒绝，不覆盖运行中恢复。

## 3. 私有adapter：ready、返回码与freshness必须分开

[原ANEForge adapter](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/src/aneforge/whisper-aneforge.cpp)从`ANEFORGE_DYLIB`加载compile/set/get/execute/release五个符号，读三行port并按`nelems*2`声明byte大小，compile时mask=`0x4`。pos.f16读入后绑定一次；每次encode先按元素乘积检查mel大小，再CPU转换FP16、set、execute、get和转换FP32。返回类型是void，pos绑定以及这三个热调用的返回码均未检查。

原创fake动态库只做确定性CPU计算：mel先半精度round，pos为8个`[-4..3]/16`，结果`half(2*half(mel)+pos)`再由原adapter还原FP32。`ANEFORGE_DYLIB`仅在child env设为精确owned路径；没有加载真实ANEForge/Espresso。mock model.mil只是明确标记的文本placeholder，从不交给私有compiler。

每seed先prime上一代，再注入当前代；所有32元素完整按独立IEEE FP16/FP32公式重算，外侧8个guard不变：

| 场景 | fake返回/行为 | 原adapter目的数组 |
| --- | --- | --- |
| success | set/execute/get成功 | 全部当前代正确 |
| set_failure | set=−7，旧mel仍保留 | 全部等于上一代prime |
| execute_failure | execute=−9，旧output保留 | 全部上一代 |
| get_failure | get=−11，不写ctx out16 | 全部上一代 |
| short_success | get=0，只写前16个 | 前半当前代、后半上一代，仍全部finite |
| wrong_count | 原wrapper在dispatch前返回 | 目的全−91.5；before/after计数完全不变 |
| pos_bind_failure | init pos set=−13 | 仍ready；本mock零初始化pos，缺常量结果错误 |

两seed共14个ready context、26次原encode调用，12个当前代目的数组为有意finite数值负例。另8个factory场景：compile失败、pos缺失、port malformed、缺符号分别返回NULL；pos缺失发生在compile成功后，release=1，其余release=0。只验证这些明确拥有的对象计数，不推广为所有文件/dlopen/并发资源无泄漏。

[真实ANEForge ABI选段](https://github.com/sbryngelson/ANEForge/blob/896c4943c5ebf811c0e815f8d1381fd1e157fa7d/aneforge/_lib/ane_e5rt_dispatch.mm#L617)中set/get/execute本来有状态返回；get还会把copy量clamp到port大小并返回成功。这说明“状态成功”和“调用方要求的全部bytes已更新”应分别验证。但**固定正常caller是否能自然触达本次short-write条件未证实**；fake允许的故障和零初始化策略不能冒充真实runtime错误/未初始化内容。

建议的接口合同是status+本代身份+完整expected shape/dtype/bytes及destination-valid状态；init绑定任何常量失败就拒绝ready，encode失败令结果无效，再由上层决定是否有资源和状态进行fallback。本研究没有修改上游adapter、实现产品fix或执行故障后GPU恢复。

## 4. WhisperKit：组件策略、prewarm和外层状态并非一回事

[原ModelComputeOptions](https://github.com/argmaxinc/WhisperKit/blob/9f99b5cb6a907730543823ad271b82532782a899/Sources/WhisperKit/Core/Models.swift#L92)在本机现代OS分支默认mel=`cpuAndGPU`，encoder/decoder=`cpuAndNeuralEngine`；显式CPU_ONLY/ALL覆盖可用，simulator策略flag会统一CPU_ONLY。两种都含CPU许可，不能将它们写作“mel实际GPU、encoder实际ANE”。

我们只编译三个**原文声明选段**：WhisperMLModel及extension、ModelComputeOptions、TranscriptionTimings，加原ModelState整文件。唯一上下文stub是可控的`WhisperKit.isRunningOnSimulator` flag；没有模拟器runtime、legacyOS分支、完整SwiftPM package或包级默认初始化。两seed共12个policy case验证默认/覆盖/flag及恢复。

原WhisperMLModel `prewarmMode`通过MLModel.load，然后把组件`model`赋nil；**不调用dummy prediction**。原创组件类在本机自有dense小资产上实际执行以下load-only序列：

```text
initial nil → prewarm load后nil/Unloaded → load后nonNil/Loaded
            → component-only失败reload保留旧model → unload后nil/Unloaded
```

两次共10个事件，Swift段没有model prediction。harness还显式保留previous强引用，所以赋nil绝不是实测物理释放或内存归零。外层[WhisperKit.loadModels](https://github.com/argmaxinc/WhisperKit/blob/9f99b5cb6a907730543823ad271b82532782a899/Sources/WhisperKit/Core/WhisperKit.swift#L366)另设Prewarmed状态，并在catch时unloadModels；不能把组件-only失败后旧model保留叫作已复现的app bug。

实际`loadModelComponents`依次await **feature→decoder→encoder**，不是并行load。共享ArgmaxCore `MLModelLoading`/ModelManager虽有另一套actor/coalescing设计，但该WhisperKit类仍用自己的legacy协议；不能因同package有新manager就假定已迁移。本批也未运行新manager。

`download:false`约束模型setup，不能独立证明全应用离线：`loadTokenizerIfNeeded`会调用有Hub fallback的tokenizer helper。实验没有调用这一路，完全没有模型或tokenizer下载。

## 5. 计时名、计数分母、异步和fallback的四个陷阱

第一，whisper.cpp `t_encode_us`从mel padding和tensor set之前开始，经过external encoder、**cross-attention K/V投影GGML graph**才累加。它是引擎encoder阶段host wall，不是纯CoreML、纯ANE或硬件kernel时间；必须同时看原输入/输出交接、cross与decoder。

第二，Kit [TranscriptionTimings](https://github.com/argmaxinc/WhisperKit/blob/9f99b5cb6a907730543823ad271b82532782a899/Sources/WhisperKit/Core/Models.swift#L730)公式是：

```text
tokensPerSecond = totalDecodingLoops / fullPipeline
realTimeFactor  = fullPipeline / inputAudioSeconds
speedFactor     = inputAudioSeconds / fullPipeline
```

loops包括prompt/fallback相关循环，并非最终去重输出token数；不要与只计decoder预测的tok/s互比。两次共6个原公式case：12 loops/4s=3，12/2s=6；零pipeline、零audio保留`+inf/nan/nan`，不清洗成零成本。默认firstTokenTime和pipelineStart都是greatestFiniteMagnitude，二者相减为0不代表测得TTFT=0。

第三，Kit的`EncodeAudio`/`ExtractAudioFeatures`/`Decode` signpost围住host async调用，不是物理counter。每窗口mel→encoder→decode有await依赖；decoder还通过FP16 key/value feature切片、host cache更新串接下一个token，该路径不是CoreML MLState。workers、多窗口、detached progress callback或异步预测API都不单独证明GPU/ANE硬件重叠，也不证明cancel已drain所有设备工作。

第四，decode质量/温度fallback会改变循环次数和质量，不是ANE backend失败后无损重算。比较策略时需固定音频、window/chunk/worker、语言/prompt、温度/阈值/重试及结果文本；本批没有这种质量/完整路径对照，所以**不发布两引擎speedup、tok/s或能耗数字**。

建议先用每请求/每window/每attempt关联ID拆出prepare/load/prewarm/first/warm、feature、encoder、cross、prediction、KV更新、sampling/filter、fallback、最终wall；保持失败attempt及零/unknown区间。再用CoreML plan（预测）、Instruments/signpost（host span）、系统采样（上下文）和有效硬件trace（若取得）分别回答不同问题，不把某一层图直接当另一层证据。见[profiling分层](profiling.md)。

## 6. 复现、证据身份与继续问题

默认CPU合同和22项portable tests仅依赖stdlib，不导入CoreML/Swift，也不重跑设备：

本地实现为、及22项专项测试。raw副本审计需要调用方明确提供旧实验目录和固定reference；默认CPU入口不访问这些外部资源。

已有ignored raw可做完整原数组审计；curator只读Git blobs、资产、binary和receipt，重算自有tensor/控制并redact，不编译/load/predict：

显式新设备实验入口是原始collector的`--device --references ../references --seed 17 --output-dir NEW`，另用无observer确认的`--device --base-run BASE --output-dir NEW`。需要macOS、已有coremltools8.3.0/NumPy2.0.2和本机SDK；不安装/升级依赖、不运行converter的`load_model`、上游setup/demo或完整包。先用supervisor，新输出目录，保留失败记录。源码编译身份、Swift精确选段边界与许可证hash见来源/证据；binary/模型和第三方源码副本只在ignored outputs。

早期两次width8 discovery observer没有保存logical输出，6次prediction仅作为发现pilot另计；不补造当时缺失的数据，也不并入16次主试。随后原curator-v1已通过raw审计，但ANEForge仓库URL写错；v2校正为`sbryngelson/ANEForge`并补入已读AGENTS，不重跑设备或改数值。whisper.cpp AGENTS限制以AI为主的上游贡献，但明确允许学习探索；本批没有上游修改、PR或提交。

尚未完成：真实Whisper/CoreML资产与合法音频/任务质量；更大/其他dtype/stride与容量检查；decoder实际KV/quality retry和完整边界；实际私有status/短copy可达性；并发/取消/drain、资源/冷启动/能耗；有效物理ANE profiling。任一commit、package/模型hash、SoC/OS/SDK/compiler、shape/stride/port/dtype、owner/observer/队列或质量策略变化均需重验。当前批次完成不等于ANE研究目标完成。
