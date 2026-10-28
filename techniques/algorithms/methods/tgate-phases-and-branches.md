# T-GATE：attention 时域缓存、CFG 减批与两套分支合同

<!-- knowledge
id: technique-algorithms-methods-tgate-phases-and-branches
type: method
stage: dit
operator: attention, sampling, layout
backend: cpu-reference, metal
device: cpu, gpu
symptom: stale-cache, wrong-layout, measurement
semantics: approximate, contract-only
representation: fp32
boundary: producer-consumer, request
api: public, reference-only
-->

[算法入口](../README.md) · [模型冗余地图](model-sparsity.md) · [FasterCache分支身份](../case-studies/fastercache-consumer-study.md) · [实验入口](../../examples/README.md) · [S62](../sources.md#s62)/[S63](../sources.md#s63)/[S64](../sources.md#s64)/[S65](../sources.md#s65)

2026-10-17，第二十二轮。T-GATE把采样分成语义规划/细节改善阶段，定期复用self-attention输出、在边界保存cross-attention，并在后段减少CFG batch。它是**approximate-train-free**，不是稀疏kernel或减少denoiser NFE，也不因项目标题中的“for free”而数值无损。

固定作者`HaozheLiu-ST/T-GATE@ab2a88f`（MIT）、移植`JettHu/ComfyUI_TGate@cc1437b`（GPL3），并追到Diffusers0.39.0与ComfyUI `b0b7435`消费者。论文同一身份有v1的Cross-Attention标题和v2/v3的[Faster Diffusion Through Temporal Attention Decomposition](https://arxiv.org/abs/2404.02747v3)；本轮论文只核摘要/身份，机制来自所列固定代码。作者README的1080Ti/T4速度和FID不是本机结果；未测GPU速度或真实图像/视频质量。

## 三种省算必须分开

1. 早期按`sp_interval`复用部分self-attention输出，cross-attention仍重算。
2. `gate_step-1`处计算一次新attention，保留原B2输出供本步CFG，同时把两路**投影后的attention输出**取均值存为B1 cache。
3. 后段cross-attention直接取cache，self按`fi_interval`更新；U-Net从B2降为B1，不再执行两路CFG差分。

`tgate_scheduler`使用传入的`cur_step`，原pipeline传的是`i-num_warmup_steps`，并非训练t/sigma；DDIM/order1与多阶/自定义步表的坐标不能混用。warm-up、边界强制写入、早晚两周期各有独立分支。gate是经验日程，代码没有在线残差误差界或迁移到任意prompt/模型的质量保证。

原`register_forward`按精确类名`Attention`递归替换forward，而不是改processor对象。`tgate_processor`先判断cache hit；命中时绕过Q/K/V、SDPA和output projection，但BasicTransformerBlock外层norm、residual、FFN等仍执行。存的是attention分支投影输出，不是最终hidden或guided denoiser输出；residual/rescale在缓存接缝之外。

平均两路不是CFG恒等式：`(u+c)/2`一般不同于`u+g*(c-u)`。后段复用还忽略当前query变化；作者观察到的条件收敛只能作为适用假设，不能变成全输入无损证明。

## 分支身份：SD、SDXL、SVD和Comfy并不相同

| 路径 | gate后实际传入条件 | 注意事项 |
| --- | --- | --- |
| 作者SD | negative/null prompt embeddings | 不是继续positive分支；cache miss会真正用该条件重新算 |
| 作者SDXL | negative prompt、negative pooled text/time IDs | 不只cross-attention缓存；其它条件路径也改变 |
| 作者SVD | `_encode_image(...,False)`与`_encode_vae_image(...,False)`返回的conditional图像数据 | 变量虽名`negative_image_*`，False实际是不拼零分支，不能按名字解释为null image |
| Comfy Advanced proxy | 意图保留语义label0的conditional、给label1补零 | 必须按实际拼批位置找payload；源码存在下述次序/等号边界 |

SVD身份由原图像encoder/VAE helper的True拼零、False返回条件选段核对，未运行SVD模型。SD/SDXL/SVD分别是不同adapter，不给其它joint-attention/多模态模型自动授予这些分支语义。

## 实际CPU消费者：全原包、真实随机网络和强消融

探针固定完整原T-GATE包导入，使用真实Diffusers `StableDiffusionPipeline`/`UNet2DConditionModel`与原`SD.tgate`函数。17个安装源码与`v0.39.0@a3608b5`匹配；原文件未改。DeepCache只在其可选loader内导入，不是普通SD路径的必需依赖；PixArt/DeepCache/SVD虽有模块定义导入，未执行其模型。

U-Net为四stage `8/16/16/16`、207404个随机参数/684个state项，latent `[1,4,8,12]`。提供两份不同的原创prompt embedding，不构造或执行CLIP/VAE，使用支持的latent-only返回路径；不是预训练SD1.5、文本到图像或完整媒体测试。CPU单线程/eval/no-grad，Torch2.14.0、Python3.13.12，所有request参数manifest前后相同。

实际DDIM有32个`beta=1/64`、10步leading `28,25,…,1`、offset1、eta0、不clip。16请求中15条完成10步，一条guidance=1在边界3步后失败。153个完整U-Net返回/154次尝试、84864个prediction scalar，以及成功窗口内15872次**投影module API调用**和失败前另外4次调用均保留；不是physical dispatch计数。

强对照不仅是full：还运行原pipeline的CFG-only null cutoff、Diffusers官方`SDCFGCutoffCallback`（保留positive）、以及仅有cross-cache数值影响的配方。前三类切换在step3结束后发生；与cross-only的前4次预测均和full逐字节相等。示例10步成本如下，表中attention次数按每个同角色层计，batch-row不含token/通道权重：

| 配方 | 成功U-Net调用 | U-Net batch-row | self重算/row | cross重算/row |
| --- | ---: | ---: | ---: | ---: |
| full / ungated no-cache | 10 | 20 | 10 / 20 | 10 / 20 |
| CFG-only cutoff（无cache） | 10 | 14 | 10 / 14 | 10 / 14 |
| cross-only T-GATE | 10 | 14 | 10 / 14 | 4 / 8 |
| gate4 / sp3 / fi1 / warm1 | 10 | 14 | 9 / 12 | 4 / 8 |
| 同上fi2 | 10 | 14 | 6 / 9 | 4 / 8 |
| fresh gate0 | 10 | 10 | 10 / 10 | 10 / 10 |

因此不能把全部节省归给cross-cache，也不能把20→14 row称为NFE减少。gate0没有经过`gate-1`写入，fresh cross-cache一直缺失，原处理器每次走真正计算；“越早gate必然少算更多attention”在此不成立。

标准库整理器恢复完整FP32 bytes/SHA，独立重放全部观察预测条件下的CFG/DDIM、15个signed schedule控制、所有32层cache manifest与投影调用顺序，以及两个选定层的完整cache均值/复用接缝，含73728个cache写入scalar核查。其它30层保存shape/hash和调用路径，不冒充其完整cache数值或U-Net/attention数学独立验证。保留全部原值、partial和对应范围。

随机fixture中cross-only相对同phase/null-CFG-only端点约0.73%向量差，而官方positive cutoff与null cutoff约1.43%；这些是**合成latent函数差异，不是图片画质百分比**，也不足以选出质量/速度赢家。

## 请求/接口负控：正常repeat与stress分开

- 原SD `guidance_scale=1`是可执行的基础pipeline路径（full控制完成10步），但此T-GATE SD在边界仍`chunk(2)`做平均，B1无法解包，3步后ValueError保留；不当作3步加速。SDXL的LCM特殊开关不能自动转授给SD函数。
- 相同正常gate4/warm1重新请求并改变条件时，reused与fresh输出逐字节相等；已有cache在warm-up/边界被更新后正常使用。因此不是“所有复用请求都会错误”。
- 改为gate0并换条件则不同：reused对象晚期cross零重算、fresh对象cross每步重算，端点不同。`cache=False`只表示不写，wrapper不会把旧`self.cache`清成None；没有条件/权重/shape/request generation身份校验。
- 在已patched对象上改回调用原pipeline `__call__`，不会自动恢复Attention.forward；同条件fresh full控制不同，前者cross仍零重算。绕过`.tgate`不等于unpatch；这里没有修改库来实现恢复。
- 不缓存的原默认Attention与自定义processor逐字节相等；启用实际`qk_norm=layer_norm`的独立module控制却有最大约0.158差，因为旧copy未执行新processor中的norm_q/norm_k。不能把这种兼容性差异全归给缓存误差，也不能对任意processor用“no-cache=all-exact”假设。
- 原processor的4D uncached调用完成，但4D cache hit在恢复layout时引用未赋值的`batch_size`而拒绝。实际本轮U-Net attention为3D；这不是默认完整请求必现错误，也不授予任意Attention布局支持。

两次开发期失败来自本库observer未把scalar Tensor clear flag编码成bool，分别保留receipt/log SHA与末行诊断；只改自有记录器，不能把它们写成T-GATE后端bug。早期proto不进入正式统计。

## Comfy端口：语义label不是拼批位置

原Advanced proxy在`σ>gate`时传完整batch，否则给unconditional补零并只算一支；它用`chunk_idx=min(label,batch_chunks-1)`。但当前固定Comfy核心会反转待拼列表，再按列表**位置**cat，并用`cond_or_uncond[position]`回填。因此正常合法的`[1,0]`次序不意味着chunk1仍是unconditional。

本轮实际执行原proxy、原custom CFG和当前core `cfg_function`，配明确affine `apply_model`与identity sigma→timestep替身；共16个正反/单路及sigma上下/等号/零点控制，不是完整Comfy节点图或真实U-Net：

| 两路次序 / sigma | 原proxy+实际CFG控制结果 | 条件payload为正确位置时 |
| --- | ---: | ---: |
| `[0,1]`, σ>1 | 21.625 | 21.625（完整CFG） |
| `[0,1]`, σ=1 | 16.625 | 既非full的21.625，也非单路4.75 |
| `[0,1]`, σ<1 | 4.75 | 4.75 |
| `[1,0]`, σ<1 | -2 | conditional应为4.75 |

反序时label0取到位置0的negative context和y，却标记为conditional。等号则proxy已进入减支（`<=`），而attention/CFG的判断仍为严格`<`，造成不同phase。不能仅靠Comfy源码变量名或README“修复”声明否定这两个选定版本的控制结果；也不把affine差异放大为所有生成图像必错。

源码中deprecated pre-CFG rescaler和Advanced的proxy/custom-CFG是不同路线；本轮没有执行旧全局monkey-patch、真实transformer cache forward、ModelPatcher/节点图或GPU CPU-cache传输。清理控制只核对原proxy传出的flag，identity sigma→timestep不是实际模型schedule，不能据此推断正常应用何时释放cache。

## Apple迁移要点

算法不依赖CUDA稀疏ISA，适合研究静态B2/B1分阶段图和连续attention-output缓存；但这是迁移推导，不是MPS/MLX/ANE实测。应分别计phase切换/编译、cache写入和读回、当前norm/residual/FFN、原强dense/fused处理器与真实完整denoiser。不能乘单项加速比或用投影调用数替代wall time。

保留明确branch position→semantic ID映射和真实H/W/token布局；缓存key包含请求/条件/权重/shape/phase，提供实际restore/清理事务。复制旧processor会绕过既有backend/fused/QK-norm语义，应先做不缓存等价控制。对Comfy源码每层`.cpu().item()`读sigma，只有sigma实际在设备上才会形成设备→host等待，不能未测就断言耗时；`use_cpu_cache`的统一内存也不自动等于零拷贝。

README所链[作者2024-04-25评论](https://github.com/JettHu/ComfyUI_TGate/issues/4#issuecomment-2077823182)现已核对：其环境为M1 Pro/32GB、Sonoma14.4.1（23E224）、Torch2.2.2，报告OS升级后出现异常，作者只是怀疑MPS后端而未确证根因。评论中的“latest mac”是当时的具体环境，不是当前macOS/Torch兼容性结论。真实GPU实验仍须新鲜空闲准入，并以质量预算约束实际预训练模型/媒体；当前不授予默认启用资格。

默认只做标准库参考/分发JSON审计，不导入Torch/TGate、执行Comfy核心初始化、下载或读模型/设备。真实probe必须显式`--run`、固定源码和新目录；source/runtime fingerprints不等于完整native binary闭包。

后续[第二十三轮MPS resident成本](../case-studies/tgate-mps-resident-cost.md)已在Torch2.14.0/M4 Max完成四cohort/200正式窗口与同配方CPU parity；fi2对CFG-only强分母仅约1.047–1.069配对中位倍率且两组IQR跨1，cross单项有一组反收益。它不是把本轮CPU语义控制直接换成GPU资格，也没有取得预训练媒体/default资格。
