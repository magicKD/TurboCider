# GPU/ANE优化技巧总览：学什么、怎样借鉴、如何验收

<!-- knowledge
id: technique-gpu-ane-optimization-playbook
type: guide
-->

[快速入口](README.md) · [主题索引](report-index.md) · [实验与反例](experiments.md) · [来源与证据范围](sources-and-coverage.md)

整理日期：2026-10-17，综合固定源码和已保存实验，不新增benchmark或刷新所有PR状态。下面的“可借鉴”指设计方法，不等于代码可直接搬进产品、原参数跨机通用或已验证完整模型提速。上游链接尽量固定commit；每项另给本地机制页与证据入口。

## 一、当前最有用的结论

1. 优化顺序应围绕**完整消费者**：先建立强GPU基线，再减少中间量、提交和交接，最后研究ANE分担。tinygrad的[强基线对照](case-studies/tinygrad-strong-baselines-and-paired-ffn.md)说明，弱分阶段GPU分母会放大“调度加速”。
2. 异构收益取决于合法独立性和暴露成本。FFN按row、paired intermediate、parallel branch、stage流水是不同方案；详见[切分推导](partition-and-scheduling.md)。顺序Transformer的attention与FFN不能直接并行。
3. 统一内存下，**布局、完成、所有权、代际**仍必须对齐；参见[真实GPU→ML pass→GPU交接](case-studies/metal4-consumer-layout-and-handoff.md)及[Whisper完整输出交接](case-studies/whisper-generation-and-complete-encoder-handoff.md)。
4. “更少kernel”“INT4/INT8文件”“全finite”“同转录”均不是充分质量/速度证明。[MetalPetal融合反例](case-studies/metalpetal-render-graphs-and-pixelbuffer-handoff.md)、[ccv scale-floor](case-studies/ccv-dense-precision-and-scale-floor.md)和[ANE-LM递归公式](case-studies/ane-lm-programs-state-and-numerics.md)给出不同层次的否决条件。
5. M5 GPU内NAX与独立ANE分开；Core ML/Core AI/Metal模型pass是执行入口，不是物理设备标签。[硬件页](hardware-and-tensorops.md)与[Apple M5/MLX文章](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)应结合阅读。本机M4结果不为M5快路径背书。

### 几项实测怎样转化为可借鉴结论

| 固定实验中的观察 | 可以借鉴的方向 | 不能外推的结论 |
| --- | --- | --- |
| MLX compile RMS-chain局部约1.13–1.32× | 将完整纯表达式交给编译/融合，减少前端和提交成本 | 不是任意模型或纯GPU kernel倍率；[第一轮范围](../MAINTENANCE.md) |
| 三组no-mask fast SDPA对物化基线约1.20–1.39× | 优先减少score/softmax中间写回 | 不覆盖所有mask、GQA、长视频或最佳attention分母；[实际case](../MAINTENANCE.md) |
| 真实Qwen shared FFN确认中GPU/shared约1.146–1.149× | 封闭FFN分担值得用真实消费者验证，并拆分计算与交接收益 | 仅固定layer0 warm组件；同client copy→shared约1.1%，不能把全部收益归零拷贝或全模型；[原记录](case-studies/qwen-shared-ffn-consumer-cost.md) |
| tinygrad完整FFN减少到2command未稳定胜过3command | 在强GPU分母下测完整调用和配对窗口 | command数不是速度排名；[主实验/确认](case-studies/tinygrad-strong-baselines-and-paired-ffn.md) |

这些是不同实验边界，不能相乘、相除或拼成同一引擎排行榜。优先借鉴验证方法，再重新测目标工作负载。

## 二、按瓶颈选择学习路线

| 观察到的问题 | 先学习的方法 | 完整验证至少包含 |
| --- | --- | --- |
| decode小batch慢 | GEMV/GEMM分流、packed解码复用（方法1） | M分布、quant group、offset/stride、冷/热 |
| attention显存/带宽高 | 分块在线softmax与paged KV（方法2） | mask、GQA、tail、空row、完整输出 |
| 小op多、CPU提交占比高 | block融合与重放（方法3） | 强GPU分母、同步完成、scalar/state变化 |
| ANE图慢或容易回退 | 封闭子图、layout/shape/量化（方法4） | plan与实测分开，含转换/交接 |
| GPU/ANE某支总在等 | 合法切分＋在线成本控制（方法5–6） | concurrent成本、join、share=0与原GPU |
| 输出偶发旧值或取消后异常 | owner/完成门与状态事务（方法7、11） | A/B/A、代际、失败、取消/drain |
| 模型能放下但I/O卡顿 | 稳定bank与驻留预算（方法9） | packed字节、slot有效性、完整consumer |
| 单stage快、整请求不快 | 分区/阶段组合和计时分母（方法10、12） | load、first/warm、TTFT/TPOT、排队/回读 |

## 三、14种可以迁移的实现方法

### 方法1：按M/K/N和格式选核，避免“一条GEMM走天下”

**机制：**小M优先考虑权重读取/解码和launch摊销，大M再提高tile内复用；split-K只在输出tile不足且归约成本合算时增加并行度。

**可借鉴实现：**[MLX quantized dispatch固定源码](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/quantized.cpp)、[GGML few-row MMA](case-studies/ggml-fewrow-mma-and-reuse.md)。先读[GEMM布局总解](gpu-gemm-quantization.md)，再看[MLX mid-M/预取](case-studies/mlx-nax-mid-m-and-prefetch.md)的route→split→kernel三层gate。

**迁移要点：**把dtype、quant group、物理row pitch、byte offset和tail加入资格检查；保留成熟fallback。不要只复制M阈值或把原K改成shard K。验收比较同一packed source代表的值、完整FFN和resident预算；M5源码读本不等于本机M5执行。

### 方法2：融合attention数据流，不物化完整score矩阵

**机制：**分块维护running max、normalizer和输出累加，减少score/probability的全量写回；paged/varlen进一步减少不必要padding和KV搬移。

**可借鉴实现：**[MLX SDPA固定dispatch](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/mlx/backend/metal/scaled_dot_product_attention.cpp)、[MFA教学实现](https://github.com/philipturner/metal-flash-attention/tree/8671cddc38f19a6eadb804dee6a3ca2954b8bf32)、[vLLM paged/NAX读本](case-studies/vllm-paged-nax.md)。原理见[attention总解](gpu-attention-runtime.md)。

**迁移要点：**一起移植mask、all-masked、GQA、head-D tail、page table和实际KV offset，而不是只换matmul。Flash-style重排与稀疏/缓存近似不是同一语义类别；后者另验长序列/图像质量。

### 方法3：先融合完整block，再研究JIT/ICB提交复用

**机制：**融合减少中间张量和dispatch；重放减少重复前端/编码。两者解决不同成本，收益不能混算。

**可借鉴实现：**[tinygrad固定源码](https://github.com/tinygrad/tinygrad/tree/3d057f05df532deca7661bfb2400c6564f96a04a)、[完整paired SwiGLU实测](case-studies/tinygrad-strong-baselines-and-paired-ffn.md)、[MLX函数编译/跨runtime对照](case-studies/mlx-compile-and-cross-runtime-ffn.md)。

**迁移要点：**把gate/up沿正确轴合并，保持SiLU、down bias与residual只执行一次。输入、shape、Python scalar、可变状态和返回对象复用都进入cache合同。比较同步完成后的完整调用；本机4/3/2-command实验没有给出通用速度排序。

### 方法4：把ANE做成大而封闭的子图，而非零散小算子

**机制：**将projection/激活/down留在图内，摊薄编译、dispatch、转换与回读；按目标编译器友好的layout表达算子。

**可借鉴实现：**[Apple ANE FFN固定实现](https://github.com/apple-aiml-research/ml-ane-transformers/blob/590eb35913622cdef217ebc5329047aab389ac98/ane_transformers/reference/ffn.py)、[vpipe图与runtime weights](case-studies/vpipe-runtime-weights.md)、[公共Split-Einsum实际导出](case-studies/coreml-split-einsum-export-and-runtime.md)。

**迁移要点：**把BC1S、Conv/Linear、bucket和W8A8视作候选表达，不写成永久硬件定律。按[Core ML官方量化指南](https://apple.github.io/coremltools/docs-guides/source/opt-quantization-perf.html)比较目标graph/硬件条件；存储压缩、图Q/DQ、整数MAC和端到端加速分别取证。

### 方法5：选对异构切分轴，并保证完整消费者代数

**机制：**row split拼接独立行；paired-F split将gate/up和down对应分片并求和；独立branches才能用两支并发关键路径。

**可借鉴实现：**[Splash paired FFN固定源码](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/ops/AneFfn.cpp#L402)、[ccv rowwise链](case-studies/drawthings-to-ccv-hybrid-route.md)、[切分公式与反例](partition-and-scheduling.md)。

**迁移要点：**gate/up必须同片，down保原pitch；量化group与packed word边界不能切坏。bias、residual、modulation和LoRA都纳入公式。先验serial split的完整值，再上并发；图内K-split不是GPU/ANE分工。

### 方法6：把“是否分担”和“分多少”分成两个控制层

**机制：**以原GPU、split和保留接缝的probe分辨收益/固定损失；离散chunk搜索配合平滑、迟滞，减少振荡。

**可借鉴实现：**[vpipe plan/finish固定代码](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/generative-models/shared/ane-ffn.cc#L581)、[双层控制器读本](case-studies/vpipe-controller-and-failure.md)、[ccv实际chooser](https://github.com/liuliu/ccv/blob/2de9f60fb93cda950a8484a59ad14ba4f2a6bb4b/lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_gemm.cpp#L995)。

**迁移要点：**使用并发时的成本，计入staging、fence、restore/join及probe自身代价；允许share=0。3%/5%、256/4096等均是源版本参数，不直接作为产品默认。首次chooser可能启动设备probe，不能因调用名是“choose”就跳过GPU准入。

### 方法7：将布局、完成事件、owner和generation作为同一ABI

**机制：**共享backing只省部分复制，不能自动解决stride、dtype、写后读、复用或旧代输出问题。

**可借鉴实现：**[Splash buffer owner](https://github.com/incoai/splash/blob/80827e1c3593f7d10c266394e4d4381321b73ad0/runtime/metal/MetalBackend.mm#L1176)、[Metal4完整消费者probe](case-studies/metal4-consumer-layout-and-handoff.md)、[Whisper固定CoreML bridge](https://github.com/ggml-org/whisper.cpp/blob/0c7be696371069e0baa2515ed7aa08d5a5473aa6/src/coreml/whisper-encoder.mm)。

**迁移要点：**以实际返回layout读取逻辑元素，不导出padding；buffer持有至真实完成，再交给下游或帧池。scheduled/BOOL成功不是completed；强制放行event不是drain。[共享合同](memory-and-handoff.md)和[MetalPetal实际task](case-studies/metalpetal-render-graphs-and-pixelbuffer-handoff.md)可作对照。

### 方法8：融合必须保留中间精度、裁剪与量化语义

**机制：**实数域可合并的两个节点，隔着UNorm、半精度写回或非线性后不一定等价。

**可借鉴实现：**[MetalPetal固定optimizer](https://github.com/MetalPetal/MetalPetal/blob/f9b78897bd4214bb097f352a1bde0a4f4a1e2ddb/Frameworks/MetalPetal/Kernels/MTIRenderPipelineKernel.m#L582)、[实际UNorm反例](case-studies/metalpetal-render-graphs-and-pixelbuffer-handoff.md)、[GGML原始融合回归](case-studies/ggml-residual-identity-regression.md)。

**迁移要点：**把中间format、alpha、transfer、scale/bias和rounding写进recipe。BGRA8亮度两步的融合开关在本机最多差64码；这不意味着全部fusion有错，也不允许用F16控制的一致性证明任意图等价。

### 方法9：稳定专家bank，把I/O暂存与GPU驻留分开

**机制：**按专家ID映射稳定slot，复用驻留packed权重；CPU staging只负责装载，不等于GPU bank的生命周期。

**可借鉴实现：**[原MoE完整block](https://github.com/Anemll/anemll-flash-mlx/blob/f86c0682a5dcd44eee1f1c090c588e4f209b62c9/flash_moe_mlx/model.py#L647)、[原I/O wrapper](https://github.com/Anemll/anemll-flash-mlx/blob/f86c0682a5dcd44eee1f1c090c588e4f209b62c9/flash_moe_mlx/expert_io.py)、[完整GPU消费者核验](case-studies/anemll-original-bank-gpu-consumer.md)。

**迁移要点：**绑定packed格式、scale、expert ID/slot/generation、short-read和有效性；无效slot不得送GPU。预取命中统计、文件缓存热度、实际SSD吞吐和模型速度分开；该组件输出验证没有授予整模型/I/O性能资格。

### 方法10：先保护高层op，再按边界成本分区

**机制：**过早分解SDPA/Linear会丢掉后端融合，过碎分区则增加跨框架copy/同步；小op支持应服务于连通大分区。

**可借鉴实现：**[ORT固定CoreML EP](https://github.com/microsoft/onnxruntime/tree/f33c1744e1e1064d3618d1d0ce533f0b946aed2d)、[ExecuTorch CoreML](case-studies/executorch-coreml-delegation.md)、[ExecuTorch MLX bytecode](case-studies/executorch-mlx-bytecode-slots-and-materialization.md)。

**迁移要点：**对具体node参数查support，不只查op名；记录实际fallback。资产共享不等于请求state共享；outputBacking是请求，不是采纳证明；逻辑slot数、lazy eval阈值和全局cache也不是同一个内存上限。

### 方法11：给编译、性能与质量建三套身份；失败后恢复完整状态

**机制：**编译cache复用资产，性能cache预测路线，质量cache绑定checkpoint/adapter/recipe；三个key不能互相替代。

**可借鉴读本：**[FusionML抽样权重hash边界](case-studies/fusionml-branches-and-observer-effect.md)、[ggml-ane常量/失败合同](case-studies/ggml-ane-backend-cache-and-failure.md)、[ANE-LM递归/状态分析](case-studies/ane-lm-programs-state-and-numerics.md)、[PMetal推测验证回退](case-studies/pmetal-speculative-and-dispatch.md)。

**迁移要点：**用完整不可变内容或显式generation，包含shape/layout/dtype、SoC/OS/工具链和并发条件。失败后别把部分覆盖的in-place输入用于GPU重算；KV位置回退不能恢复conv/SSM等递归state。中止/完整重算/rollback应在真实consumer决定，不能只打印warning。

### 方法12：按stage组合音频/视觉模型，保留端到端计时边界

**机制：**vision/encoder和language/decoder可采用不同执行入口；只有依赖允许时才可做跨请求流水。

**可借鉴实现：**[Whisper真实模型与阶段计时](case-studies/whisper-coreml-metal-stages-and-real-asr.md)、[完整encoder交接](case-studies/whisper-generation-and-complete-encoder-handoff.md)、[官方FastVLM固定实现](https://github.com/apple-aiml-research/ml-fastvlm/blob/6f7b13179bbc81e6aa7d921bc7c6335c43904778/app/FastVLM/FastVLM.swift)、[FastVLM接缝读本](case-studies/fastvlm-coreml-vision-and-mlx-handoff.md)。

**迁移要点：**分别计load、first、warm、TTFT、完整response、排队和回读；验证token/image ID及输出shape。Newest(1)是背压/丢帧策略，不是硬件pipeline；同转录不能推出完整encoder无损。当前FastVLM实际只跑原JSON语句，没有模型/App执行。

### 方法13：用强分母、独立参考和反序确认衡量收益

**机制：**局部内核、host同步调用和完整request回答不同问题；探测器本身也可能改lowering/工作节奏。

**可借鉴记录：**[GGML同输入融合计时](case-studies/ggml-matched-fusion-timing.md)、[tinygrad paired统计](case-studies/tinygrad-strong-baselines-and-paired-ffn.md)、[MLX跨runtime窗口](case-studies/mlx-compile-and-cross-runtime-ffn.md)、[trace归一化](../profiling/trace-normalization.md)。

**迁移要点：**独立FP64/格式oracle、完整输出、branch误差与极端输入同时查；先冻结候选再新进程/新输入确认。总体中位数之比不是配对比值中位数；CoreML plan、软件command数、推测timeline和物理counter分标签，不发布混合排行榜。

### 方法14：M5 TensorOps同时学习host选路和协作数据流

**机制：**新矩阵单元要配套operand布局、协作scope、解码、tail与同步，不是把旧MMA函数名替换掉。

**可借鉴入口：**[Apple M5/MLX](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)、[WWDC26 tensors](https://developer.apple.com/videos/play/wwdc2026/330/)、[固定MLX mid-M head读本](case-studies/mlx-nax-mid-m-and-prefetch.md)、[Uzu Metal4路由](case-studies/uzu-metal4-and-routing.md)。

**迁移要点：**分清API可编译、GPU family支持、实际dispatcher选中与完整消费者更快。预取增加寄存器/双buffer成本；低比特storage先解码成浮点后MMA，不应称原生INT4乘。本机M4无法为M5/NAX性能或其最优GPU/ANE比例作结论。

## 四、优先复查的反例

| 容易误判的“通过” | 当前记录提示的缺口 | 学习材料 |
| --- | --- | --- |
| status成功、所有值finite | scale-floor可把应非零值变零 | [ccv稠密/scale边界](case-studies/ccv-dense-precision-and-scale-floor.md) |
| 第一token/零state正确 | 递归顺序第二步已可不同 | [ANE-LM公式与执行身份](case-studies/ane-lm-programs-state-and-numerics.md) |
| API/异步wait返回 | scheduled/强制signal不等于完成/drain | [MetalPetal](case-studies/metalpetal-render-graphs-and-pixelbuffer-handoff.md)、[Splash退役](case-studies/splash-handoff-calibration-and-replay.md) |
| 更少command、总体误差小 | 排序受窗口影响，residual可掩盖坏branch | [完整SwiGLU对照](case-studies/tinygrad-strong-baselines-and-paired-ffn.md) |
| 导出函数正常返回 | tokenizer实际ID与模型config可能不一致 | [FastVLM原JSON片段](case-studies/fastvlm-coreml-vision-and-mlx-handoff.md) |

这些证据类型不同：真实GPU/模型、原CPU片段和独立数学并列展示，不合成一个“上游故障复现数”。

## 五、把方法变成一个可验收的小任务

每次迁移至少写清：

1. **对象与基线：**真实消费者、checkpoint/LoRA、M/K/N、dtype/quant recipe、原强GPU实现。
2. **唯一变化：**改tile、融合、分片、提交还是驻留；是否改变近似/量化语义。
3. **代码来源：**固定commit/PR parent-head、具体函数、许可和依赖；源码与binary身份分别保存。
4. **正确性：**完整输出＋独立参考，tail/stride/offset、bias/残差、dense/极值、A/B/A及状态恢复。
5. **成本：**prepare、dispatch、device/wait、join、first/warm、完整request、resident/峰值；测不到的保持unknown。
6. **否决与回退：**不盈利允许关闭分担；失败结果不发布，确认owner和drain后再复用。

学习顺序建议：**完整FFN强GPU基线 → 一种合法切分 → 一条真实交接 → 完整请求**。确认这条链有效后再加入新量化、M5快路径、更多设备或自动搜索。现阶段不默认启用私有ANE后端，不安装依赖或更改产品最低系统要求。
