# HiDiffusion 完整 Tiny U-Net：skip 消费者、组合拒绝与请求时钟

<!-- knowledge
id: technique-algorithms-case-studies-hidiffusion-unet-consumer
type: case-study
stage: dit
operator: attention, conv, sampling, layout
backend: cpu-reference, metal
device: cpu, gpu
symptom: wrong-layout, stale-cache, measurement
semantics: approximate, contract-only
representation: fp32
boundary: producer-consumer, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [方法与上一轮接缝](../methods/hidiffusion-spatial-compute.md) · [S60](../sources.md#s60)/[S61](../sources.md#s61) · [实验入口](../../examples/README.md)

2026-10-17，第二十一轮。完整未改HiDiffusion包已在真实Diffusers0.39.0的小型U-Net上导入并执行，不再用identity/affine resnet或均匀attention替身。20条合成latent轨迹中19条完成10步，一条odd组合在第二次U-Net调用拒绝；不能把部分轨迹当省NFE的加速。另验证中断后重用、reapply和fresh对象的不同状态。

**这仍不是预训练SD/SDXL、完整生成pipeline或画质/性能验收。** 模型是随机初始化的小网络，官方名称只用于选择原adapter；没有text encoder、VAE、真实prompt语义、ControlNet、GPU、速度或媒体测量，也没有独立重算整张U-Net的数值。

## 固定实际消费者，而不借用同名版本

已有Diffusers main快照`d42ebf9`与安装包并非所有选定文件相等，因此未拿它当运行身份。另固定`v0.39.0`标签的commit `a3608b5`，13个安装的Python源码文件逐字节与Git blob匹配，涵盖U-Net/block、BasicTransformerBlock/processor、resnet/Transformer2D、down/up、配置/owner及DDIM；17个来源文件还含许可和维护规则。不是完整包/依赖/compiled native image的哈希证明。

固定HiDiffusion `e9ec39a`的9个文件仅在ignored工作目录原样准备，实际导入原`hidiffusion`包，直接调用原apply/remove/factory/forward；**没有AST替代、compatibility shim或改上游**。探针需显式`--run`/新目录，框架只在该入口内导入。Python3.13.12、Torch2.14.0、Diffusers0.39.0，显式CPU/单线程/no-grad/eval，不安装包或读取预训练资产。

两种真实`UNet2DConditionModel`构造：

| 声明拓扑 | stage channels | transformer层配置 | 参数 / state项 |
| --- | --- | --- | ---: |
| 四stage SD15-style | 8/16/16/16 | 每组1层 | 207404 / 684 |
| 三stage XL-like | 8/16/32 | 1/2/2 | 525348 / 716 |

每down block两对resnet/attention，每up block三对；原GroupNorm/LayerNorm、非线性、QKV、SDPA、GEGLU、time embedding、skip和output conv均实际运行。XL-like不是原SDXL的宽度、深层transformer或added-conditioning配方。`DiffusionPipeline`只作为真实module注册/信息owner，实验自有采样循环，没有调用原StableDiffusionPipeline生成入口。

## 20条同参数轨迹，只有19条完整

两种拓扑×`8×12`/`9×13` latent grid，共四cohort。每cohort full、两个开关都关闭的disabled-patch、RAU-only、window-only及组合五臂，均从同一完整随机parameter bank复制；每臂前后检查全部state manifest的hash一致，不公开权重。

每条使用实际DDIMScheduler：32个`beta=1/64`，10个leading timestep为`27,24,…,0`，epsilon预测、eta0、不clip。两个distinct合成条件batched成B2，CFG为`u+3.5*(c-u)`；每次成功U-Net输出反馈到下一步latent。NFE仍为每步一次batched U-Net，不把两路说成两次独立forward，也不把提前失败的轨迹当少算赢家。

保存191个完整网络预测/161976个prediction scalar，以及所有guided/x0/输入/输出latent和31812个形状事件。另有20次remove后同首步输入读回、9次请求时钟控制：总计221次逻辑U-Net forward尝试，220次正常返回、1次部分前缀后异常。不是kernel/dispatch计数。

整理器用标准库解码全部FP32 little-endian bytes/shape/SHA；独立重建alpha表、CFG和每个DDIM反馈，所核条件标量最大误差0。它以**观测到的网络预测为条件**，不等于独立验证了该预测的U-Net数学正确性。独立shape程序还逐事件重放down/up、skip stack、目标尺寸、原transformer/attn1/attn2/FFN输入输出；记录的是module边界，不是Q/K/V内部数值或物理backing。

示例：偶数XL-like的10步累计`ΣB*N²` self输入槽数，full为123840，RAU为101328，window为37440，组合为29760。这是按每head归一的逻辑token-pair槽，不计实际head倍数、投影、cross、FFN、roll/resize或后端调度。不能把`123840/29760`叫实测速度，也不能乘两个单项缩减比例；这些信息只能指导后续profile。

随机网络的latent端点差异只描述函数/反馈改变，不作为图像质量预算或方法排名。所有完整/部分记录均保留，失败的端点比较为null，不补成0。

## 组合拒绝：面积推断不能恢复任意真实网格

XL-like的`9×13`组合在step0成功；step1进入aggressive pooling窗口。down stage1最初实际grid为`5×7`，第一对后ceil pool得到`3×4`，即12 tokens。随后原window forward用root面积和N推断：

```text
round(sqrt(9*13/12)) = 3
ceil(9/3) × ceil(13/3) = 3×5  ≠  3×4
```

真实原transformer收到`[2,12,16]`，却尝试view为`[2,3,5,16]`，384个元素无法reshape，RuntimeError保留。失败前的完整回调前缀、停在clock1的四个controller和独立推断原因都核查；不是手造错误metadata。本cohort的RAU-only/window-only各自完成10步，组合不能继承单项资格。

该反例来自明确的小型随机网络/尺寸，不证明作者默认高分辨率SDXL必然失败，也没有修改H/W推断以强行通过。迁移时应携带真实spatial axes/phase，而不是仅靠area ratio恢复网格；对未审计shape应显式拒绝或回退已验证full路径。

## 中断重用：reapply不是请求状态reset

在同一个真实四stage U-Net上完成旧10步请求的前6次调用，然后把owner声明长度改为4。下面是同输入、同`t=27`的隔离控制，**不是完整四步sampler请求**：

| 对象状态 | module clock | owner长度 | num_upsamplers | 实际首步输出 |
| --- | --- | ---: | ---: | --- |
| 中断对象直接重用 | 6→7 | 4 | 15 | 与full首步逐字节相等 |
| 再次apply同对象 | 7→8 | 4 | 27 | 仍与full首步逐字节相等 |
| 同参数fresh对象 | 0→1 | 4 | 15 | 与正常RAU首步逐字节相等、不同于full |

原归零条件是增加后`==max_timestep`，不是`>=`；旧clock已超过新长度时不会正常归零。reapply先恢复class再换新class，但旧调用创建的instance字段仍在，优先于新class上的clock0默认值。这不是weight更新或prompt变化造成的差异；两个对象完整parameter digest相等。

这说明调用者不能把apply/remove当任意取消后的完整事务恢复。后续工程应明确request generation/clock所有权，验证reset或使用fresh对象；这里未实现生产包装修复，也未证明原完整pipeline在所有中断出口的行为。

## 元数据未恢复，不等于每次输出都错

上一轮发现num_upsamplers在remove后仍增12。本轮真实消费者给出了重要限制：disabled-patch在四个cohort的全部10步预测/端点均与full逐字节相等；全部20次remove后的原U-Net同首步输入也与full一致，尽管计数仍不同。

原因接缝是原U-Net用`2**num_upsamplers`决定是否向父upsampler传入skip的目标size；在这些网格上，显式目标与原nearest路径得到相同实际形状/值。**不能从计数残留直接断言已经产生错误图像**，也不能由这些等值控制推断任意shape/版本都无影响。数值、结构恢复和请求状态分别验证。

## Apple迁移与继续范围

本轮把空间变换从替身推进到真实完整CPU消费者，但没有扩大成MPS/MLX/ANE性能资格。迁移仍需成对skip/resize合同、真实H/W传递、fresh request时钟、不同CFG执行方式和额外CPU RNG来源。若要测速，先确认GPU竞争负载，再测同参数full强基线、单项和组合的完整窗口；失败/未完成的组合不能进入速度排行榜。

下一步仍是预训练模型、真实条件/媒体及Apple设备的独立质量/成本预算；当前随机latent差异和逻辑slot数不替代这些证据。默认离线入口：

默认仅标准库读取分发的合成JSON/纯shape和scalar参考，不导入Torch/Diffusers、执行上游包、读预训练权重或提交GPU。context tag只描述所报版本、选定源码与client hash，不是完整runtime二进制身份。
