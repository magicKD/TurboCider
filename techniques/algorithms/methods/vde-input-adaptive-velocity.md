# VDE：用当前输入估计速度，而不是直接重放旧输出

<!-- knowledge
id: technique-algorithms-methods-vde-input-adaptive-velocity
type: method
stage: dit
operator: attention, sampling, layout
backend: cpu-reference, metal
device: cpu, gpu
symptom: stale-cache, wrong-layout, measurement
semantics: approximate, contract-only
representation: fp32, fp16
boundary: producer-consumer, request
api: public, reference-only
-->

[算法入口](../README.md) · [预测器比较](forecast-calibrators.md) · [DiCache](dicache.md) · [来源 S67/S68](../sources.md#s67) · [实验](../../examples/README.md)

2026-10-17，第24轮。结论：VDE把完整denoiser调用换成“当前latent＋两次完整求值的系数＋最近法向方向”，数学上不依赖CUDA专用kernel，值得研究Apple迁移；但所读实现不是在线质量门控器，也没有证明batch独立或退化输入稳定。本轮真实随机FLUX CPU请求验证了跳算/反馈接缝，同时保留原Qwen版本拒绝。没有新增GPU测速、预训练图像/视频或产品默认资格。

## 先核对论文、源码和实际发布范围

正确论文为Junwen Tan、Jinglin Liang、Hongyuan Chen、Shuangping Huang的[arXiv 2605.23381v1](https://arxiv.org/html/2605.23381v1)，题名含“via Velocity Decomposition and Estimation”，[CVPR目录](https://openaccess.thecvf.com/CVPR2026?day=2026-06-07)与作者仓库交叉确认。选读方法、稳定阶段和算法伪码；CVF单篇页本轮抓取失败，不宣称已经通过该页完整复核全文。

作者[固定源码快照](https://github.com/Tan-Junwen/VDE/tree/3d08e45de4378ba0e4848c0dfa6c7216b4311f2c)，Apache-2.0；[源码计划（过程原件已移出）](../../MAINTENANCE.md)记录14个采用文件的精确身份。该tree只有FLUX、Qwen两份Python脚本；Wan2.1/Z-image README仍是待发布占位，ComfyUI目录有图片/占位文本，没有节点Python。根README的支持勾选与To-Do并不能代替具体实现。这个结论只限该commit，不证明作者其它分支/仓库或以后也没有代码。

论文把RF速度与当前输入分成平行分量和正交分量；系数随时间较平滑、方向较稳定时，用少数完整锚点外推。这是近似成立的经验条件，不是每个prompt/阶段/模型的误差上界。下面的具体epsilon、batch、CLI与版本问题来自实际源码及本库实验，不从论文摘要推断。

## 完整锚点到底保存了什么

把整个packed latent tensor记作向量`z`，网络输出为`v`，`ε=1e-6`。FLUX helper的实际计算为：

```text
r = ||z||₂                 u = z / (r + ε)
p = sum(v * u)             q = v - p * u
s = ||q||₂
α = p / (r + ε)            β = s / (r + ε)
n = q / (s + ε)

估计步：v_hat = α_hat(t) * z_current + β_hat(t) * ||z_current||₂ * n_last_anchor
```

`α/β`是全tensor标量，`n`是一个完整latent大小的方向。源码变量`tangential`在这里实际指平行于`z`的分量，不按变量名误认为球面的切向。epsilon使`u`不严格单位长，重建也不严格等于原`v`；旧`n`更不保证与当前`z`正交。零latent且非零速度时，估计式直接归零。

`predict_alpha_beta`只用最近两个完整求值的`t`做线性外推，不用循环索引；单锚点保持常数，重复时间返回最新系数，空history拒绝。`torch.lerp`的weight可大于1，`β_hat`可为负，没有clip/trust-region/在线误差检测。history列表保存所有后期锚点，但查询只读最后两个，方向只保留最新一个；因此不是TaylorSeer的多阶全输出导数，也不是TeaCache的输入差阈值，更不是DiCache的block残差比例。

### 日程：少网络调用，不是少采样更新

令循环索引从0开始，`s=stable_step`、`k=interval`。原脚本在`i<=s`、末步或`(i-s)%k==0`时完整执行，且从`i>=s-1`的完整步开始存锚点。第一次估计已有两个锚点；`stable_step`不是“完整执行s次”。所谓stable由用户选定，不是运行时测量达到稳定。

| 配方 | 完整网络索引/次数 | latent更新 |
| --- | --- | --- |
| 本轮10步，s2/k3 | 0,1,2,5,8,9：6次 | 10 |
| 本轮10步，s2/k5 | 0,1,2,7,9：5次 | 10 |
| FLUX README 50步，s6/k3、4、5 | 分别22、18、16次 | 50 |
| Qwen README 50步，s10/k3、5 | 按代码分别24、19次 | 50 |

Qwen README fast表中的18步与其给出的s10/k5命令按源码算出的19次不同，不能拿表格数字冒充该命令复现。论文伪码里“跳过n步”与CLI的调用周期也不要直接混用。完整调用次数比只是计算上界线索，不是本机速度倍率。

## CPU实验证据与边界

显式客户端核对Git blob后完整导入两份原脚本，无AST替身、源码改写或全局class patch；用现有Python3.13.12/Torch2.14.0/Diffusers0.39.0 CPU执行。六个采用的已安装Diffusers文件与固定release commit相等，**不是整个wheel/native二进制的身份闭包**。

FLUX是实际`FluxTransformer2DModel`：1个双流block＋1个单流block、2头×8维、24,176随机参数，guidance embedding开启；实际`FluxPipeline`和`FlowMatchEulerDiscreteScheduler`，6个packed image token、4个text token，预先给定全部embedding和latent。VAE/text encoder/tokenizer为支持的None路径，没有下载/读取预训练模型或真实activation。完整网络预测是独立重放的条件输入，不声称重新实现了transformer数学。

及标准库整理器保留9条完成轨迹：90次latent更新、69次真实完整网络返回、21个估计步。完整10560个Euler输出scalar逐个精确一致；2400个估计scalar连同direct controls共2442值，用独立归约/FP32参考核对，预定`2e-5 + 2e-5*abs(ref)`，最大绝对差`3.8147e-6`。这验证实现合同，不是近似相对full的质量通过。

正控：interval1每一步预测及端点与stock pipeline逐字节相同；同一pipeline先跑A再跑B，与fresh B逐步/端点逐字节相同，局部anchor正常重置。输入和66项参数manifest完整绑定，所有臂参数不变。

### 反例一：batch组成会改变近似轨迹

独立A/B与合并B2用相同网络/输入/条件，B的latent幅度为A的8倍，embedding也不同。full的batched/single最大差均约`2.38e-7`；s2/k3近似路径的对应差明显放大：

| 样本 | VDE相对自身full的端点relative-L2 | VDE合并batch相对单独的relative-L2 |
| --- | ---: | ---: |
| A | 16.61% | 10.51% |
| B | 0.769% | 0.415% |

根因候选由源码全tensor归约支持：共用scalar系数和方向归一化分母，样本不再数学独立。这是随机网络反例，不是预训练图像损失比例，也不证明所有batch都产生同等误差。改成per-sample FP32归约应另立算法/质量对照，不把修改后的版本叫原实现。

### 反例二：epsilon和归约dtype是算法合同

原helper在普通FP32向量上的重建relative-L2约`8.07e-7`；零latent为100%，tiny latent约99.76%。有限FP16输入`[49984,49984]`的范数溢出，最终重建两项均NaN。这里没有声称真实模型轨迹必经这些输入；它说明原实现没有全输入数值保证，且只有防除零epsilon不够。应分别校验norm/dot/系数的accumulation dtype、finite和近零策略。

### 三个原实现拒绝，不静默修补

1. FLUX Python调用的默认`28`步与`stable_step=49`冲突，直接ValueError；CLI的50步/6则不同，不能称CLI默认也坏了。
2. FLUX传入自定义`timesteps`时脚本仍同时生成`sigmas`，在原retrieve helper拒绝，两者不可同时给。本轮保留0步失败，而不是缩短成功请求。
3. 原Qwen传`txt_seq_lens`，实际Diffusers0.39.0 forward签名不接受，TypeError发生在网络计算前。真实Qwen类构造/pre-hook到达，但没有Qwen forward、CFG或采样轨迹资格；没有删除参数或用shim伪造兼容成功。不能外推其它版本也失败。

## Qwen的CFG顺序与组合边界

源码先分别分解/外推conditional和unconditional速度，再做CFG，最后逐token末维范数rescale；不是只估计guided向量，也不是FasterCache保留一个分支重建另一个。norm/CFG不线性，不能交换次序；combined norm没有epsilon，零分母是另一个须验证的边界。本轮此节是源码阅读＋原创scalar合同，**不是原Qwen消费者的数值复现**。

两份脚本的anchor在调用内初始化，普通warm重用有正控；但callback能改latent/condition而不主动失效当前锚点，概念编辑需要额外generation身份/强制refresh。FLUX替代call没有保留当前stock pipeline的全部true-CFG、IP-Adapter、cache-context、current-timestep及begin-index接缝；这些功能不能只因基础调用成功就获组合资格。Qwen的MPS dtype回转是防御代码，不是Apple性能或全op GPU证据。

## Apple Silicon迁移：值得测什么，先拒绝什么

估计步是norm＋pointwise，刷新有多次norm/dot；可研究MLX编译融合或MPS驻留buffer，不需要把CUDA稀疏kernel搬来。但每步仍有Euler和大向量读写，FP32归约/方向state、同步与anchor成本必须进完整窗口。原predictor检查重复时间会调用tensor `.item()`；若时间张量在MPS上会有host取值边界，不能按纯设备端闭图估价。本轮没有提交GPU工作，也没有为这个假设做计时。

建议未来强对照：full、interval1、直接旧速度replay、全输出两点外推、VDE；固定sigma/seed/模型，包含warmup、末步refresh、per-sample/global归约消融及真实画质门槛。最先检查小幅度/溢出、batch组成、非均匀时间与guidance变化；先在相同质量预算下比较完整请求，再谈谁更快。所有GPU实验仍须新鲜三样本空闲准入，不能把本轮CPU正常执行补写成Apple设备结果。

## 离线复核与重新运行

前两条只依赖公开JSON与标准库；测试会逐元素重新整理并检查现有源码hash。真正原模块执行另需明确`--run`、固定Git对象、已安装的指定依赖和全新ignored输出目录，probe不联网、不安装包。整理器是本次归档cohort的严格审计器，launch路径/参数也固定；改变recipe或另起cohort需新记录，不能放宽旧证据的门槛来接纳。

重开条件：作者新增可执行Wan/Comfy实现、Diffusers接口变化、模型/权重/dtype/shape/batch、sigma/CFG/callback策略或真实媒体质量/性能预算改变。Delta-DiT的论文/代码可得性另见[前沿候选](research-frontier.md)，不把未取得作者实现的路线当本轮源码复现。
