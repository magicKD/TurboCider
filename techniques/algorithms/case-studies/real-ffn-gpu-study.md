# 真实BF16 FFN GPU：强dense基线、packed权重与低秩分支成本

<!-- knowledge
id: technique-algorithms-case-studies-real-ffn-gpu-study
type: case-study
stage: dit, ffn
operator: gemm, quantization, layout
backend: mlx, metal, cpu-reference
device: gpu, cpu
symptom: precision-loss, memory-bound, launch-bound, measurement
semantics: approximate, numerically-equivalent, contract-only
representation: bf16, fp32, q4, q8, packed, mlx-affine
boundary: weight-runtime, producer-consumer
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [真实CPU校准/误差](real-qwen-lowrank.md) · [SVDQuant机制](../methods/low-rank-quantization.md) · [合成GPU组件](lowrank-mlx-component-study.md) · [实验](../../examples/README.md)

2026-10-16第十轮。M4 Max/64GiB、macOS26.6.2、已安装MLX0.32.0，实际执行原始Qwen-Image-2.1 image FFN的完整SwiGLU GPU组件：GU[24576,4096]、SiLU×up、Down[4096,12288]，输入/输出/scale/bias/低秩因子均BF16，packed carrier为uint32。

核心观察：在本轮全部36个**resident warm组件**点，compiled Q8/Q4/r32/r64均慢于强compiled BF16 dense；低秩可降低部分近似误差，却进一步增加延迟。**不是原Nunchaku W4A4、完整量化请求、低内存/offload或媒体结论**，不改产品默认。完整数据见。

## 配对数据与计时边界

沿用[第九轮核验的原BF16权重与两条真实轨迹](real-qwen-lowrank.md)：layer0/15/31、baseline seed42茶壶/heldout seed43自行车、step1/10/19，均无LoRA、cache off。每份输入分别运行33个固定均匀rows与完整1024 rows；K/F/N不裁剪，共36点。33行是数值/shape诊断，不当作512×512请求默认token数；独立prompt不用于低秩分解或重新选rank。

每层新进程准备五组参数：原BF16 dense、group64 affine Q8、Q4-r0、Q4-r32、Q4-r64。range80、一次power iteration/reduced SVD沿用本库原创CPU函数；将sqrt奇异值分配的因子舍入BF16，再从原weight减去**已舍入因子**的乘积，residual也舍入BF16后pack。不是第九轮FP32 scale/factor配方，不能直接沿用其误差。

每组均有eager/compiled，共10计算臂；另有已eval输入的host eval+sync计时control。compiled函数显式传入全部tensor参数，无隐藏weight状态；图编译不等于Nunchaku手写fusion。每点4次warmup/臂，12轮轮转、奇数轮反序，共4320次计算臂调用＋432次host control＝4752段formal host-ready时间。

计时从**ready原BF16 GPU input到完整BF16 output ready**，包含Python建图/compiled入口、两个QMM、SiLU×up、分离低秩及join和同步。模型加载、输入gather/upload、CPU分解、GPU pack/参数核查、首次图构造/数值观察不在formal timing内。它不是cold start、整个block、encoder/VAE、权重streaming或生成请求wall time。

组件止于Down投影输出；模型外层的norm/modulation、tanh gate和residual addition不在这个计时边界，不能用FFN误差或耗时替代完整block的误差传播/延迟。

raw字段`first_use_host_ready_ns`仅是数值观察调用耗时：强dense已先用于reference，故并非所有臂的首次调用，更不是cold latency。`gpu_pack_and_bank_setup_ns`是bank准备/核查的host范围，包含CPU decode检查，不是纯GPU pack/device时间；不拿这些字段混作formal速度分母。

## 强基线下没有净加速

下表为三层/两prompt/三step共18个点各自中位数的范围，单位ms；括号为每点12个trial的**median(dense-compiled/candidate)**范围。不是置信区间，也不能用范围两端互除。ratio<1表示候选更慢。

| compiled臂 | M33 ms（配对速度比） | M1024 ms（配对速度比） |
| --- | --- | --- |
| 原BF16 dense | 1.739–1.948（1） | 21.120–21.388（1） |
| Q8 | 1.835–2.094（0.919–0.969） | 23.810–24.076（0.886–0.889） |
| Q4-r0 | 1.812–2.115（0.936–0.974） | 23.778–24.074（0.887–0.890） |
| Q4-r32 | 1.936–2.208（0.866–0.916） | 24.388–24.695（0.865–0.868） |
| Q4-r64 | 1.941–2.235（0.858–0.916） | 24.576–24.892（0.859–0.860） |

两种低秩相对同点compiled Q4-r0也全部更慢。本批四种compiled候选相对eager dense同样没有胜点；“弱分母可反转赢家”仅由原创算术case说明，**不捏造本批发生了这种反转**。compiled dense在34/36点快于eager，另两点的配对中位比显示最多约0.39%更慢；主分母仍用预先固定的optimized compiled dense，不逐点选较快分母或因候选结果不利而切换。

host control的点中位数约0.0215–0.1045ms，保留为host尺度诊断，既不是同等FFN，也不从候选耗时相减。只有两轮的早期pilot不混入formal cohort或当成独立确认；formal三进程各对应不同层，也不是同一point的跨进程复核或长期热稳态。

## 完整1024行误差，不以平均遮住尾行

以下是M1024、每层6个真实输入点，compiled候选对同点**原BF16 GPU compiled dense输出**的global relative L2范围，单位%。第九轮分母为FP64数学函数、参数精度也不同，两轮数字不相减作因果归因。

| 方法 | layer0 | layer15 | layer31 |
| --- | --- | --- | --- |
| Q8 | 0.153–0.269 | 0.552–0.683 | 0.201–0.262 |
| Q4-r0 | 1.312–2.479 | 6.040–7.730 | 1.809–2.174 |
| Q4-r32 | 0.947–2.189 | 5.846–7.461 | 1.073–1.557 |
| Q4-r64 | 0.952–2.157 | 5.805–7.408 | 0.936–1.279 |

例如中层Q4-r0最差row约13.879%，Q8约1.200%。不能把33行抽样的最大值当完整1024行上界。JSON保存每个输出row的SSE/energy、全部p95/max、输出hash和所有timing，不删坏row或不利的layer/step。

实际已执行的两矩阵、非线性和join，还与按真实packed codes、BF16 scale/bias/因子计算的独立FP64公式比较：每点固定33行、全部K/F/N通道，两格完整F维math.fsum见证。最大global误差约0.006417、最大row约0.012885，均通过**预定同配方数值gate**0.03；weight未量化重构≤0.01、promoted decode≤1e−6、BF16因子RNE与设备cast完全一致。

这些gate只核查实现与同配方参考；没有把“对原dense误差≤3%”当作合格条件。完整1024行对原GPU dense误差全部计算，但独立FP64仅覆盖33行，**不是全M全输出FP64复现，也不是媒体质量预算**。输入仍来自原始base轨迹，量化输出未反馈进后续模型/采样。

## shape与compile也不能默认逐位一致

所读[MLX0.32.0 S44](../sources.md#s44)包含contiguous准备、qmv阈值及qmm_splitk：在“nonbatched/transpose/contiguous、BM=BN=32”的选中源码条件下，M33的GU预测split1、Down预测split2；M1024两者预测split1。源码还包含NAX等守卫，**此表只是源公式推导，不是安装wheel执行了哪个shader的观测**。

同输入33行在M33和完整M1024中的输出确有差异，最大global relative L2约0.002969。eager与compiled的输出hash也在各方法1–3/36点不等；强dense compiled自身重复控制则全输出相同。没有shader/counter/实际loaded-image取证，不能把差异唯一归因于split-K、fusion、累加顺序或某条ISA。

原创`gpu-ffn-comparison-contracts`保留三类学习witness：配对中位比不等于两个中位数之比；换强分母可反转假设赢家；先将BF16 partial分别舍入再求和不等于最后一次舍入。它们不是源kernel重放或本机物理dispatch证据。

## 逻辑权重更小，不等于该边界更快

每层同时驻留的候选各自逻辑bytes，含BF16 scales/biases与因子：

| 参数表示 | MiB |
| --- | --- |
| 原BF16 dense | 288 |
| Q8 | 153 |
| Q4-r0 | 81 |
| Q4-r32 | 83.75 |
| Q4-r64 | 86.5 |

这里没有对每方法独立量峰值，全部34份timed参数数组及CPU FP64 decode/reference共存；逻辑bytes不是进程/系统峰值。GPU BF16 metadata也不同于第九轮CPU FP32 metadata，不能混写文件/运行容量。

W4A16没有把activation降到4bit；减少weight表示仍要支付目标kernel的decode/矩阵数据流、额外分支与提交/join。未测ISA或DRAM流量，不能凭结果断言唯一瓶颈；packed API也不证明INT4 MAC。Nunchaku W4A4和真正融合、模型不驻留时的offload收益、其它SoC/MLX版本都需要重新实验，本轮负结果不否定它们。

若当前需求是本记录的warm resident FFN延迟，不能仅按“4bit”“rank额外FLOPs很少”启用此配方。若瓶颈是容量，先做完整模型驻留/低内存和media预算；不能借本表逻辑MiB自动授予全请求收益。

## 空闲、来源与复核

每个bank setup、每个case、formal timing前均重新取3个GPU利用率样本，Device≤5%才进入；计时门控末样本与开始差≤2秒。保留52个先前未通过窗口后重新采样；一些出现在本探针warmup之后，不能将driver数字归因为52次外部任务。另有脱敏process name/CPU占用，无argv、不杀其它进程、不清系统/model cache，也无连续per-PID、频率或温度证明。

三个owned numerical/timing进程都正常退出；每份原input和全部34个weight/packed/factor参数前后hash不变。当前wheel的extension/lib/header身份已记录，缺失`.pyi`明确保留为缺失；同version源码tag不是binary可复现构建或loaded-image attestation。

执行入口需显式`--run`、原checkpoint与两份capture；标准库整理器独立复核全部4752段原始样本、轮转反序、full/sample SSE/分位数、paired ratios、shape-overlap三角界和source/terminal/hash。它不执行GPU、不重新跑全部FFN数学或读checkpoint；默认tests也不导入NumPy/MLX或执行上游。

结论仅属于上述来源、shape、精度、warm/resident边界；后续需固定媒体/误差预算、目标GPU低精度/融合、同点独立重跑、真正模型消费者及完整量化trajectory。没有新增生成画质或产品默认资格。

## 离线交付与版本复查

来源45条/163个文件hash，生成检索索引及`git diff --check`通过。版本登记绑定主要GPU evidence、实际执行入口和父数据；精确匹配M4 Max/64GiB/MLX0.32.0 context也只是recorded-context-only，变更会提示复查，不授予采用资格。它不自动覆盖所有OS/依赖/工作负载变化，仍要运行完整hash/verifier与相关新实验。

日期跨日不回填旧run metadata，也不把复核日期当新GPU测量日期；上述离线复核没有重新提交GPU工作。
