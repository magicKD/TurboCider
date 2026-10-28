# Whisper：变化输入、完整encoder交接与实际Core ML布局

<!-- knowledge
id: technique-gpu-ane-case-studies-whisper-generation-and-complete-encoder-handoff
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [前轮真实ASR](whisper-coreml-metal-stages-and-real-asr.md) · [本轮摘要](../../MAINTENANCE.md) · [来源](../sources.md) · 完整数据 · 计划 · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17。本轮补前轮“重复同一音频不能发现旧代输出”的边界。继续固定whisper.cpp `0c7be696371069e0baa2515ed7aa08d5a5473aa6`及官方tiny.en资产，原2039个Git blob与模型只读复核，没有下载新权重、改参考/上游/系统/产品。

核心观察：**A→B→A在两路都恢复完整encoder值，B在复用state与fresh context中一致；真实Core ML返回为dense，完整逻辑输出与GGML接收数组逐字节一致。但两路encoder间仍有约1.91%–2.07%的相对L2差异，即使转录相同，也不能叫数值无损。**这些是所选正常输入的值/交接证据，不是通用ASR、物理ANE或并发/故障恢复资格。数据

## 1. 源码原样，观察方式与前轮不同

自有Objective-C++ TU通过`#include`编译**原字节不变的`whisper.cpp`**，连接原GGML/Core ML库和原ANEForge helper；完整模型、DSP、encoder、cross、decoder与sampling实际执行。它增加了读取内部state的观察代码，**不是前轮packaged-library同一个binary**，也没有用CPU模拟代替GPU/Core ML。构建与客户端观察器

保持Metal/embedded source/Accelerate ON、原GPU/flash-attention默认、Core ML All、init fallback OFF；NATIVE/OPENMP/CCACHE OFF，与前轮对应配置一致。私有ANEForge环境存在则拒绝，未调用私有dispatch。四个进程全部先经五次低于5%的准入，20个preflight均0%，owned/native全部terminal0。

原AGENTS的学习探索范围遵守，无上游修改、PR或提交。SDK中`MLMultiArray` scoped buffer声明及3个源文件新选段另绑定，整仓库仍不宣称人工全文阅读。[来源](../sources.md)

## 2. 固定同长度B，观察A/A/B/A/A与fresh B

A是前轮176000个FP32样本、11秒JFK音频。B在GPU运行前就固定为`A[88000:]+A[:88000]`，保持长度、样本multiset和取值范围，只循环平移半段；不是根据结果挑出的更容易样本。计划

| step | context | 输入 | 正常运行实际观察 |
| --- | --- | --- | --- |
| a0 | 新context1 | A | 保存完整基准值/转录 |
| a1 | context1 | A | 与a0 bitwise一致，先刻画重复稳定性 |
| b-reused | context1 | B | 当前mel、encoder及转录均不同于A |
| a2、a3 | context1 | A | 完整mel/encoder恢复a0，转录/token恢复 |
| b-fresh | 新context2 | B | 与b-reused完整mel/encoder及转录/token一致 |

两种backend各有观察版和plain版，共**24完整请求、8个context**。每个请求要求原`n_encode==1`，关闭历史prompt与温度重试，greedy/best_of1/T0、English、4线程、VAD/timestamps off。fresh context并不意味着OS/compiler cache冷或物理缓存完全隔离。数据

B是编辑后的响应控制，没有独立人工转录标注；其返回末尾等内容不能当正确识别或一般质量评分。A的所有token序列仍与前轮原library的25-token序列一致。当前序列值正确不证明每次都物理重算，内部正确缓存也可能保留相同函数值。

## 3. 实际Core ML布局：在原平copy之前检查

观察版仅在own进程包住原`predictionFromLogmel_data:error:` IMP：先取当前input，再调用原IMP**一次**，读取真实output，最后返回**同一个输出对象**。没有改computeUnits、tensor数值或替换prediction；退出/异常路径恢复原IMP。观察器

使用公开`getBytesWithHandler`，指针仅在block内消费；在访问前、block内、返回后检查shape/dtype和活跃维度stride，验证scoped容量后只写逻辑元素。unsupported布局在返回原`memcpy`前拒绝，不导出padding，也没有为上游实现stride-aware修复。[SDK声明](https://developer.apple.com/documentation/coreml/mlmultiarray/getbyteswithhandler(_:))

六次真实Core ML预测都观察到：数据

| tensor | shape / dtype | 实际before/after stride | scoped容量 |
| --- | --- | --- | --- |
| 输入mel | `[1,80,3000]` FP32 | `[240000,3000,1]` | 960000 bytes |
| 输出encoder | `[1,1500,384]` FP32 | `[576000,384,1]` | 2304000 bytes |

前后descriptor相同，活跃维度dense且容量足够；没有读取未定义padding。本轮只给这个真实资产/shape正控，不覆盖nil、其它dtype/stride、短buffer或运行时失败。教学合同拒绝负例不等于在Core ML中执行过那些故障。

## 4. 从逻辑返回一直核到GGML实际接收

原API完成后、下一请求/free之前，验证`state->embd_enc`是已分配、contiguous的GGML F32 `[384,1500]`，使用原`ggml_backend_tensor_get`读取**全部576000值**；同步完成点与tensor-get大小检查沿原runtime，不在ctx释放后保留裸view。[原source](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/src/whisper.cpp#L2295)

native结果元数据位于各自sched的持久meta，buffer不会因普通scheduler reset就立即free；encoder、cross、decoder有各自sched。本次读取严格在下一次构图/复用之前，不给任意旧tensor跨代访问或并发owner资格。[reset/getter源码](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/ggml/src/ggml-backend.cpp#L2105)

保存12份encoder快照，**6,912,000个FP32值完整finite检查**；另存12份完整staged mel。六份Core ML输出与对应GGML接收576000值逐字节相同；六份预测前实际input也与该请求staged mel的240000值逐字节相同。完整payload绑定

因此，本次实际dense资产的平copy没有额外改值，借用input的当前逻辑值也正确。输入借用与值相同仍不证明Core ML内部zero-copy、backing地址稳定或没有内部分配；返回/接收双重观察不累计成额外模型预测。

## 5. 值一致、响应当前输入与数值等价是三件事

两路各自的四份A encoder/mel全量bitwise相同，B与A不同，复用B与fresh B全量bitwise相同。B相对于A的encoder变化相对L2约0.7691（Metal）/0.7667（Core ML）。这是指定序列正常路径的响应/恢复证据，**不是对所有音频、错误状态或取消/重试的freshness保证**。数据

两路的实际staged mel逐字节相同后，比较完整encoder：数值诊断

| 输入 | Core ML相对Metal的L2差异 | 最大绝对差 | 不同FP32位置 | frame L2差异P95 |
| --- | --- | --- | --- | --- |
| A | 0.02073376 | 0.47167770 | 575999 / 576000 | 0.03073889 |
| B | 0.01911851 | 0.22022533 | 576000 / 576000 | 0.02707445 |

所有1500帧都保留独立摘要，没有只报全局平均。统计用FP64 CPU累加，**不等于构造了独立FP64 Whisper encoder oracle**；Metal只是另一实现的比较基线，不能凭这些差异决定哪路更准确，也没有事后套一个通用tolerance来授予等价资格。捕获器

由相同mel、Core ML逻辑输出↔GGML目的值相同，可把这组跨backend分歧定位到encoder生成路径（导出/表示/算子数值实现等），而非本次平copy；尚未定位具体算子，更不能归因物理ANE。前轮67参数FP16-cast multiset对应也没有证明graph/role/LayerNorm/GELU的有限精度配方一致。[前轮资产边界](whisper-coreml-metal-stages-and-real-asr.md)

## 6. plain对照能排除什么观察器影响

另两个plain进程运行同一序列，但不安装IMP观察器、不取GGML encoder/mel快照，也不导出未经布局检查的Core ML bulk目的数组。每一路的plain与observed文本/token序列均一致。数据

这只确认这些高层可见响应没有改变，**没有证明unobserved完整encoder逐位相同、backing/同步/内存开销不变**。getBytes、完整readback和磁盘写出都会改变执行，不能拿本轮elapsed比较两路速度；前轮无此观察器的timers保持原样。

## 7. 可迁移合同与未完成项

1. 把请求身份、真实输入、逻辑返回、交接目的与消费者分别关联；同样shape/相同文本不代表当前代数值正确。
2. 用A/A/B/A与fresh B把正常重复噪声、输入变化和复用污染分开；本次同长度避免单靠音频长度/loop边界推断响应。
3. 先确认真实layout、dtype、容量及生命周期，再决定能否flat copy；只检查`count`不够。特定dense正控不能消除其它shape的padding风险。
4. 高层任务质量、完整数值差异、接口copy一致性、性能、物理placement分别授予资格，不能互相代替。

没有注入真实nil/short/fault、并发/取消、跨请求异步复用或私有dispatch；没有修改产品后端/设备策略/系统缓存，未获得物理ANE/M5/overlap/能耗/峰值内存/drain证据。下一步仍需更广真实音频与质量、导出配方定位、故障/回退和设备trace；持续目标保持active。
