# ccv异构row split：在线拟合、256行量化、4096行分段与失败边界

<!-- knowledge
id: technique-gpu-ane-case-studies-ccv-hybrid-cost-and-failure-boundaries
type: case-study
stage: pipeline
operator: layout, scheduling
backend: coreml, metal
device: gpu, ane
symptom: wrong-layout, lifetime
semantics: contract-only
representation: iosurface, pixelbuffer, fp16
boundary: gpu-ane, producer-consumer
api: public
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [上层调用链](drawthings-to-ccv-hybrid-route.md) · [来源](../sources.md#drawthings-ccv)

2026-10-17，固定ccv `2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b`。本轮读到实际chooser、两种GPU分支、CoreML bridge及消费者，并用显式stand-in执行未改的CPU策略函数。**所有成本系数都是假设输入，没有本机GPU/ANE测速、CoreML执行、真实cache/calibration或模型质量结果。**

## 1. 名为choose，首次调用却可能跑设备probe

[select_cached_hybrid_cost_model](https://github.com/liuliu/ccv/blob/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b/lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_gemm.cpp#L935)不是纯表查询：首次分别用1024/4096行、固定N=K=4096拟合ANE和GPU的fixed+slope，还测bandwidth/fence；较大行数先跑，兼作预热。源注释中的约300ms是作者设备经验，本页不当本机数字。

因此调用这个原始chooser本身就属于可能的设备实验，必须先过GPU空闲guard；不能因只想打印比例就绕过准入。本轮stand-in校准函数直接返回预设系数，没有执行这些probe。

源码以GPU kernel类别区分两个全局cache项：Int8MatMul与NAInt8MatMul；不是每个N/K、dtype、model或device fingerprint一项。所读select中，initialized但invalid不会重新probe。具体单GPU进程可以使用这种折中，但迁移到多device/context、热状态改变或不同shape时，必须重新确认适用范围；本轮未运行真实cache失败/失效流程。

## 2. 成本不是简单的max(ANE,GPU)

[chooser](https://github.com/liuliu/ccv/blob/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b/lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_gemm.cpp#L995)将工作量近似为rows×N×K，分别加固定成本；activation准备、权重复制、恢复输出和fence计入串行bytes/bandwidth项。只有两侧kernel阶段取max，非NAX时再乘tile数。

关键机制：

- 先求两侧预计同时完成的balanced rows，再按256行取最近边界；这是项目经验粒度，不是官方ANE唯一shape限制。
- 非NAX GPU路径要求Apple9、K%8=0、单batch、无bias、FP16或FP32、32-bit索引范围；NAX路径有不同资格和batch表达。
- 非NAX分担按全GEMM分tile，每个ANE program最多4096行；还比较“少一个tile、填满4096”候选，避免固定/串行成本抵消平衡收益。
- chooser同时比较GPU-only和ANE-only，返回的ANE rows可为0、M或内部边界。ANE-only时间公式没有在该行显式pad M，不能把其连续模型当逐shape实测。

算法仅评估若干候选，不是穷举全部比例的全局最优证明；线性斜率、共享带宽与两侧并发互扰也需实际验证。上层源码还明确GPU-only可能重建浮点权重再GEMM，而split使用Int8MatMul；拟合的GPU分母不能不经核对就当最终fallback的实测延迟。原始串行/拟合假设不会因预测时间写成ms而变成硬件counter。

## 3. 两种实际执行顺序

| 分支 | 准备与提交顺序 | 重要数值差别 |
| --- | --- | --- |
| GPU有NAX | GPU全量quantize（可H256）→复制ANE子集/权重→prep fence；提交GPU剩余行；CPU等prep后CoreML；GPU恢复ANE输出 | 两侧复用已准备的activation codes/scales，ANE transform不重复旋转 |
| GPU无NAX | 每tile先准备ANE行/权重并提交，CPU等prep；再提交GPU前缀行，随后CoreML和输出restore；按tile顺序继续 | 源策略把FP16 GPU行视为直接消费原行，ANE行另量化；不能假定两侧与全GPU数值配方相同 |

两条路都把GPU行和ANE行写回各自原始row位置，不是K-split部分和归约。非NAX的start/end按`floor(M*tile/tile_count)`划分，tile末尾留固定ANE行数，前面由GPU完成。程序数量、行对齐与合法余数仍要满足run函数的额外条件。

这说明源代码具备尝试重叠的提交设计，不证明当前请求的物理overlap，更不是完整FFN/attention分支的性能结论。row代数切分可成立，activation量化/累加/恢复却可能改变数值；质量验证必须覆盖两侧与完整consumer，不能只测一边的matmul。

## 4. bool、ANE rows与运行status是三个值

调用方[MPS GEMM](https://github.com/liuliu/ccv/blob/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b/lib/nnc/cmd/blas/mps/ccv_nnc_gemm_mps.m#L538)按次序处理：

| chooser / run结果 | 后续 |
| --- | --- |
| chooser拒绝 | 仍可能尝试ANE单独或GPU；不是hybrid成功 |
| chooser接受，ANE rows=0 | 刻意跳过ANE-only，走GPU路线 |
| chooser接受，ANE rows=M | 不调用内部split，尝试ANE-only |
| 0<ANE rows<M，run>0 | 当前GEMM完成 |
| split run=0 | 可继续fallback尝试 |
| split run<0 | 传播INVALID，不无条件重算 |

尤其不能把bool false与“选GPU-only的rows0”混淆。资格门、初始化失败与执行后失败对应不同恢复行为。

header把0描述为未编码、负值描述为编码后失败，但所读NAX函数在提交partial GPU并commit后，prep wait失败仍可return0；非NAX也按tile进度返回不同值。故不能仅凭0就推导**完全没有副作用或在途命令**。这是一条需结合stream ordering/失败现场继续核验的合同边界，本轮没有复现race、错输出或证明fallback不安全。

## 5. 公共runtime API与手工artifact、backing意图要分层

当前[CoreML bridge](https://github.com/liuliu/ccv/blob/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b/lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_coreml.mm#L849)调用MLModel并请求CPU+NE，但自行生成`.mlmodelc`目录、MIL、coremldata header/metadata等。使用公共prediction API不等于生成的文件格式都是公开稳定合同；metadata中的coremltools版本字符串也不是实际运行过该版本工具的证明。

PixelBuffer创建后明确拒绝不紧密的物理bytesPerRow，避免后续GPU转换按packed rows误解释；不能从统一内存推导任意stride都能零拷贝。预测后仍只核返回数组可读与总字节数，所读路径未核对返回pointer/预定surface身份或不一致时copy回scratch。本专题[输出backing合同](../memory-and-handoff.md)解释为什么identity/stride另需验证，本轮没有将独立heap实验升级为此bridge故障复现。

`fast_fence_cpu_wait`的原始循环只有atomic acquire比较，**所读循环没有deadline/backoff**；async finish helper只结束batch，不等于host/其它device已可读。未来设备测试需要有界外层监督、明确失败排空与资源代际；不能从方法名或cache owner推出取消/drain安全。

## 6. 本轮实际执行了什么

CPU probe从固定原文件提取未改的batch helper、非NAX资格函数和chooser，编译原始参数header；calibration、Apple9属性、precision byte width是显式stand-in，不链接Metal/CoreML。UBSan native stderr为空。

完整11520点为6个假设profile（含一个拒绝profile）×3dtype×10M×2N×2K×2batch×2NAX×2bias×2Apple9。原始CPU结果逐点与独立公式一致：6320拒绝、2846 GPU-only、1287 ANE-only、1067 hybrid。**这是给定假设系数的策略结果，不是硬件选型推荐、11520次GPU/ANE执行或速度数据。**

其中59个非NAX hybrid点另检查tile row区间：无重叠/遗漏、恰覆盖M，ANE行数之和正确；不冒充原始GPU/ANE调度或量化数学执行。初次checker误算总数为23040，在完整native输出后失败；确认以完整笛卡尔积核11520，前后stdout字节相同，没有缩小输入集或放宽结果比较。

来源curator还拒绝过commit别名tree响应与canonical tree ID不一致的顶层字段；重新按commit metadata的真实tree ID取canonical walk，逐项entries完全相同，原文件/CPU结果未改变。完整原始来源、初次失败、二次监督和所有输出仅留ignored artifacts；20项标准库测试可离线复核发布数据，不调用设备。

下一步应在合法上层weight/shape条件下跑原ccv组件，先guard再允许真实chooser的隐藏校准；记录cold fit、运行选路、全量输出、实际backing、device trace及完整consumer计时。不能用本轮假设系数代替这条设备验证链。
