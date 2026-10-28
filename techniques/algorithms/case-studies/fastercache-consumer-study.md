# FasterCache 调用链：CFG 身份、空间 FFT 与 attention 刷新耦合

<!-- knowledge
id: technique-algorithms-case-studies-fastercache-consumer-study
type: case-study
stage: dit
operator: attention, sampling, layout
backend: cpu-reference, mlx, metal
device: cpu, gpu
symptom: stale-cache, wrong-layout, precision-loss, measurement
semantics: approximate, contract-only
representation: fp16, fp32
boundary: request, producer-consumer
api: reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [FasterCache机制](../methods/fastercache.md) · [SeaCache/MagCache](../methods/spectral-magnitude-cache.md) · [CFG生命周期](mlx-teacache-lifecycle.md) · [实验](../../examples/README.md)

2026-10-17，第十二轮。固定作者仓库`02d05ef7edb48bdec01d2b4df6edfe7f472a41b9`，扩展[S46](../sources.md#s46)十个文件的consumer审计，实际Torch2.14.0 CPU执行选定原函数/方法。这里不是最新版调查或CogVideoX模型复现：没有SAT/einops包初始化、checkpoint、sampler求解、VAE、GPU、视频、速度或媒体资格。

本轮纠正一个重要阅读误区：**选定CogVideoX配方实际保留unconditional路，重构conditional路**。脚本局部变量叫`cond/uncond`，但configured guider输入顺序是`[uc,c]`；忽略consumer会写反分支身份和CFG误差系数。另观察到两个省算策略因batch守卫耦合，不能将比例直接相乘。完整数据保留两seed四臂400行、reset20行、换条件1行、六个诊断及FFT控制。

## 先沿消费者确定分支，不沿变量名猜

本轮实际执行的选定原方法链为：

```text
VideoDDIMSampler.denoise
  → VanillaCFG.prepare_inputs：context = concat(uc,c)
  → Denoiser.forward：按batch进行scale，不交换两路
  → OpenAIWrapper.forward：context透传，concat只沿channel
  → fastercache_dit_forward：x[:1]、context[:1]
  → 微型Transformer / 原fastercache_layer_forward
  → DynamicCFG.__call__ / NoDynamicThresholding
```

原`SATVideoDiffusionEngine.sample`传`cond, uc=uc`，固定yaml选DynamicCFG/50 steps；这些入口也已阅读，但没有执行完整SAT/session。实际标签由原guider把positive=1、negative=0的fixture拼起来验证：full transformer收到`[0,1]`，single收到`[0]`。Denoiser/Wrapper没有把它再翻转。

| 脚本局部字段 | 实际来自 | 在该configured链中的身份 |
| --- | --- | --- |
| `single_output` / `cond` | 第0路/`context[:1]` | unconditional，真实执行 |
| full分支中`uncond` | 第1路`output[1:2]` | conditional |
| `delta_lf/hf` | 第1路谱－第0路谱 | conditional－unconditional |
| `recovered_uncond` | 当前第0路谱＋缓存差 | 重构conditional，放回第1路 |

因此不是“模型把CFG顺序弄错了”的结论：其输出仍回到guider期望的`[u,c]`。这里纠正的是变量名造成的解读错误，不能推广为作者所有model adapters都采用同一身份。

对`y=u+g(c-u)`，只近似conditional时`e_cfg=g*e_c`；只近似unconditional才是`(1-g)*e_u`，两路都近似则相加。本轮cfg-only替身中第0路与full逐元素一致，保留第1路误差及guide后误差，确认前一种系数；组合臂不能再套单路公式。真实denoiser的scale、后续solver和非线性会继续影响传播，不能从这个代数式直接保证RGB/视频误差。

## 三种时钟不能混成一个step

DiT入口每次调用将`self.counter+=1`；attention cache和CFG cache使用这个**从1起始的调用计数**，不是上一轮native日志的0起始step，也不是训练timestep。

选定config的`num_steps=50 / scale=6 / exp=5`中，sampler用`guidance_step_index = 50 - training_timestep`，不是调用序号。原离散helper给出的50个training indices为999、979、…、19，传给guider的坐标为−949、−929、…、31。实际读到的公式为`1 + 6*(1-cos(pi*(index/50)^5))/2`：本轮scale范围约1.00000053–6.99994102，最后为1.12339794，并非简单按50次调用单调从1升7。

这是固定旧源码/config的可观察行为，不替它改成更合理的日程，也不据此宣判当前上游/论文的质量。移植时应分别记录call counter、训练timestep、solver坐标及guidance schedule；“同样50步”不足以证明配方相同。

## FFT是空间分解，不是视频时域滤波

原helper接收`[(B*T),C,H,W]`，`fft2`默认仅沿最后H/W两轴；圆形低频mask半径`min(H,W)//5`，高频为补集。不会沿frame/channel做FFT。

但`fftshift`和重构的`ifftshift`都未指定dims，因而**移动所有轴**，包括flattened frame/batch和channel。配对使用时这些轴上的移动会抵消；只把inverse shift改成H/W、却保留原forward shift，会错配frame/channel。本轮两组odd/even小shape独立DFT检查中，正确roundtrip最大误差≤4.77e-7，错误partial inverse shift的最大差分别3.90625/7.5。移动time轴不等于变换time频率，更不能把本方法等同SeaCache的THW spectral indicator。

DiT wrapper对供FFT的数据显式`.float()`，本轮实际看到float32→complex64→float32→模型dtype的精度岛。直接对CPU Half调用原helper得到`NotImplementedError: Unsupported dtype Half`，正常wrapper走的是FP32；这条CPU观察不证明PyTorch MPS、MLX或Metal对Half/FFT的支持情况。

full refresh时分别算两路空间谱、存`delta_lf/hf`。single时：

- counter≤40：每次复用都把存储中的low差乘1.1。
- counter≥30：每次复用都把存储中的high差乘1.1。
- 期间二者都启用时一起放大；这是对缓存的原地逻辑更新，不是始终从full anchor乘一次固定系数。

例如counter17的full差在18、19经过1.1、1.21倍，再由20 full重新seed；31–34两带连续放大，41–44只有high继续放大。这里的1.1不是统一误差上界；调模型/日程/geometry之后不能直接继承。

每次helper还重建arange/meshgrid/mask并搬到tensor.device。固定50-call配方的每个CFG臂实际有44次fft2、26次ifft2，以及配对shift；这些工作不能漏计。按原config的13×16×60×90 latent估算，仅两份complex64差谱就有17,971,200 bytes（约17.14MiB）逻辑payload，尚不含临时谱、mask、attention历史、allocator/模型/OS；不是峰值测量。

## Attention外推与CFG省算相互影响

attention在counter15保存两个相同输出，后续full时滚动保留两个实际attention输出；从18起，`counter%3!=0`且最新缓存batch≥当前batch时复用`latest + .3*(latest-previous)`。hit不发布新anchor。固定.3没有除以实际anchor间隔，也没有乘目标时间距离，不能当作任意非均匀时钟上的严格一阶Taylor外推。

CFG从counter18起在`counter%5!=0`时只计算第0路，其余full。它没有独立“最后一步必须full”的guard；counter50 full来自整除5。已观察的counter48仍是single，若同一controller简单截断到48，不能假定会自动变成末步full。

两种seed的原源码计数完全一致；下表单位是**单层fixture的branch-evaluation次数/50次外层调用**，不是GPU时间、MAC数或速度比。

| 臂 | transformer branch | attention branch | 实际attention调用 | MLP branch | fft2 / ifft2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full替身 | 100 | 100 | 50 | 100 | 0 / 0 |
| attention-only | 100 | 56 | 28 | 100 | 0 / 0 |
| CFG-only | 74 | 74 | 50 | 74 | 44 / 26 |
| 组合 | 74 | 57 | 33 | 74 | 44 / 26 |

组合臂在counter20/25/35/40/50切回双路时，最新attention缓存只有单路，batch守卫强制重算；因此比attention-only多5次attention调用。组合attention branch=57，甚至略多于attention-only的56，同时MLP降到74。独立比例相乘会错误预测`100*.56*.74=41.44`，并非实测57；这些branch计数本身仍不等于真实GPU batch效率、FLOPs或完整请求净收益。

## CPU执行与独立参考的边界

显式probe先核对十个Git对象hash，再受限提取原函数/方法，不导入上游初始化、SAT、distributed、model loader或CUDA内核。三个类方法只改绑定名，body/参数不改；FFT proxy仅记录调用并交给真实Torch CPU，不改全局Torch模块。

两seed17/23、四臂、每臂50 calls。fixture为`[B,3,2,5,5]`输出、单个affine attention/MLP、identity norm、zero time embedding、identity denoiser scaling；支持两个固定rearrange模式的显式reshape adapter，不是einops。dense/attention-only的full-CFG controller和各臂启用哪些source接缝由原创host fixture控制，不宣称上游有同名开关。

标准库verifier用独立分离DFT、FP32舍入、完整fixture公式和状态时钟重放全部421行的branch与guided向量，另核对两组FFT完整实虚部。预先固定实现gate为max-abs≤5e-5，实际421行最大3.33786e-6，FFT控制最大1.84774e-6。模型主体用FP64仅为参考清晰，source FFT/guider仍显式FP32；不是作者FP16大模型的数值资格。

raw数据包含原创替身完整向量、state聚合和计数，没有实际model activation、权重或视频。实现gate只核验“是否执行了这个公式”，不是cache与full之间的质量通过线；近似臂相对full的非零错误全部保留，不据toy误差评选真实视频赢家。子进程正常退出，默认tests不执行上游、Torch、NumPy、GPU或checkpoint读取。

## 六个诊断与正常restart要分开

| 诊断 | 观察 | 不能扩大成什么结论 |
| --- | --- | --- |
| 未seed直接跳到counter18 | 缺delta的AttributeError，counter已前进 | 不是正常counter0启动失败 |
| direct batch=1/4，从15进入16 | 硬切片与B/T声明冲突，reshape adapter拒绝 | 不是实际einops或所有上游batch API验收 |
| counter18 attention后MLP注入失败 | 新单路attention已发布、counter前进，频谱未更新 | 没有完整事务rollback或可安全任意retry证明 |
| counter19改变token geometry | 仅batch guard不足，旧token shape导致Torch运算失败 | 正常demo每请求geometry固定，属前提破坏诊断 |
| `scale=1`走sampler单路bypass | 第16次full频谱refresh遇到单路而拒绝 | 不能将双路配方直接用于guidance-distilled模型 |

另外，按原sampling_main只把counter重设0，保留旧字段，换seed/positive后正常warmup20 calls可通过独立参考：前14次旧attention仍驻留但不被读，第15次重置attention anchor，16/17刷新谱，然后才允许18起近似。这不等于立即释放旧cache、按identity验证任意resume或具备异常恢复。中途换positive另留一行完整输出，明确违反demo的固定conditioning前提，不冒称常规pipeline自动换条件。

## Apple迁移的可用设计与尚未验证部分

可以迁移的是控制器/状态边界、GPU空间FFT与固定full/single图选择的数学结构，不是CUDA/SAT runtime或实机速度。建议依次做：

1. 显式绑定CFG order/branch count、训练/solver/call坐标、negative/positive、geometry与checkpoint/LoRA；单路CFG直接走它自己的完整路线，不让频谱cache强行要求双路。
2. 将attention history与频谱history作为两个不同clock的状态；完成完整step后提交，失败/取消重新seed或清理，不能只依赖counter自增。
3. 若有跨步batch/shape变化，验证每个缓存anchor的身份、token layout与完整shape，不只看最新batch大小；保持text/vision切分一致。
4. 在真正Apple后端测FP32/complex64的H/W FFT、mask缓存、临时buffer、cast和host同步，并纳入优化dense/full分母；源CPU Half拒绝不替代这项验证。Core ML/ANE若分图，还要计入GPU FFT与固定图的交接。
5. 分别测attention-only、CFG-only、组合和优化full；用真实长视频的语义/negative prompt、运动与时间一致性预算验收。不能因单项skip或理论branch数更少便相乘速度，也不能借本轮toy错误/频谱roundtrip授予媒体资格。

没有新增GPU或视频实验；完整请求/alpha预算链仍见[第十一轮](native-cache-request-study.md)。本轮补的是先前仅按局部变量阅读遗漏的consumer身份、时钟与组合成本，后续真实Apple FFT/模型消费者验证仍待继续。

后续[第十三轮MLX GPU组件](spatial-cache-mlx-study.md)单独比较谱差搬到空间residual的实现成本，包含refresh/hit与offset精度拒绝；没有重跑本轮Torch CPU source cohort或给CogVideoX视频生成授予新资格。它不是将这里的branch计数直接兑换成GPU速度。

## 交付复核：2026-10-17

标准库CLI重新整理与公开JSON逐字节一致；从固定Git对象重新收集的S46全部10个file hashes与ledger一致，累计来源47条/183个文件hash。来源采集器、数值/状态verifier和默认tests各自的边界保持分离：metadata/hash不自动证明语义，完整fixture误差不自动证明真实视频质量。本轮没有修改上游或产品默认，也不据通过测试宣布全网探索完成。
