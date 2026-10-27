# HiDiffusion：空间降算、四窗口注意力与分辨率日程

<!-- knowledge
id: technique-algorithms-methods-hidiffusion-spatial-compute
type: method
stage: dit
operator: attention, conv, sampling, layout
backend: cpu-reference, metal
device: cpu, gpu
symptom: wrong-layout, stale-cache, launch-bound, measurement
semantics: approximate, contract-only
representation: fp32
boundary: producer-consumer, request
api: public, reference-only
-->

[算法入口](../README.md) · [来源S60](../sources.md#s60) · [实验入口](../../examples/README.md) · [结构化稀疏](structured-sparsity.md) · [ToMe](token-reduction.md)

2026-10-17，第二十轮。新增一条不依赖跨步cache的路线：在选定U-Net层/早期调用缩小feature map，在选定高分辨率self-attention中真正分成四个窗口。它不减少采样器NFE，不是把完整attention矩阵乘mask，也不是全网token pruning。属于**approximate-train-free**，不因作者称lossless就升级为逐元素无损。

固定作者实现`megvii-research/HiDiffusion@e9ec39a`，Apache-2.0；[论文v2](https://arxiv.org/html/2311.17528v2)为2024-04-29版本，另核对[作者项目页](https://hidiffusion.github.io/)。论文的高分辨率目标及原作者质量/速度结论，不是本机验收。本文没有跑原SD/SDXL、完整Diffusers pipeline、ControlNet、MPS GPU或生成媒体。

## 从实现看真正省掉什么

RAU-Net改变部分深层网络接收的空间网格。原downsampler卷积通常`stride=2,padding=1,dilation=1`；选中窗口内，原方法临时改为`4,2,2`，保持原weight/bias而减少输出位置，不新增训练参数。其它分支在第一对resnet/attention**执行完后**做`avg_pool2d(2,ceil_mode=True)`，再继续该block；对应up block在第二对计算后按共享`upsample_size`做bicubic恢复，第三对才回到完整网格。

因此并非“整个选中block都只算四分之一”：第一对down计算仍full，up后段也full。省下的主要是两者之间的空间计算，另有pool、插值、skip连接和布局成本。`make_diffusers_upsampler_block`本身只计算阈值/推进时钟并做conv，不能把它误读成额外4×插值；完整U-Net/父upsampler通过skip形状与显式size配合。该父调用链本轮仅源码查看，未执行完整模型。

MSW-MSA在原transformer forward的norm/可选pos_embed之后，将`[B,H*W,C]`变成`[4B,ceil(H/2)*ceil(W/2),C]`调用原`attn1`，再逆序恢复后接residual。对偶数H/W，QK pair数为`4*(N/4)^2=N^2/4`。这是逻辑pair数，不是实测4×速度或物理内存减为四分之一；QKV/输出投影仍覆盖所有token。`attn2`的cross-attention和FFN在恢复后执行，仍处理完整N，不能乘上同样的节省比例。

窗口shift来自每次forward的`torch.rand(1)`，四档为0、window轴长`//4`的1/2/3倍；使用cyclic roll，未额外构造Swin式边界mask。故跨边界连接是该实现函数的一部分，不能无说明地换成非循环局部窗口。

## 两种日程：论文符号不等于代码变量名

代码的`timestep`是**每个patched module自己的调用计数**，不是传入的训练t/sigma；总长度来自`info['pipeline']._num_timesteps`，成功调用后加一，在等于总长度时归零。要求模块每次denoiser调用按预期执行；solver内部重复、条件分支绕过模块、中断重用或缺少pipeline私有计数，都需要重新核查。

普通路径将ratio乘总调用数后取整。当前快照还默认开启SDXL的aggressive路径：早段conv阈值改为`int(8/50*n)`；cross-down/up的pool区间则为`int(8/50*n),int(ratio*n))`。它不能只用README或论文中的一个T解释。inpainting/Playground默认关aggressive，ControlNet另有ratio；本轮只执行这些flag分支，**没有运行ControlNet网络**。

| 独立微型控制 | 总调用n | conv降采样调用（0起） | paired down/up pool调用 |
| --- | ---: | --- | --- |
| SD1.5普通小档T1 | 10 | 0–3 | 本组未执行 |
| SD1.5大档额外T2 | 10 | 0–2 | 0–2 |
| SDXL默认小档 | 50 | 0–7 | 8–19 |
| SDXL inpaint flag | 10 | 0–3 | 0–3 |
| Turbo | 4 | 0–1 | 0–1 |
| SDXL aggressive短日程 | 4 | 空，取整为0 | 本组未执行 |

源T1/T2 ratio还分别绑定不同module key；论文极高分辨率附录的两阈值符号顺序与代码key命名不能直接对号入座，应追到消费者和具体上下界。它也不是同一网络完全不变的solver加速。

## 原方法CPU执行和独立全数组重放

[探针通过Git固定hash读取原factory/forward/hook/patch函数，只执行选定AST声明，不导入上游包或调用原构造器。Python3.13.12/Torch2.14.0，显式CPU/单线程/no-grad；无权重、activation capsule或GPU提交。

窗口内使用真实CPU SDPA的`Q=K=0,V=input`，norm为identity，cross/FFN为明确affine替身；full-sequence对照是本库自己的同一fixture，不是原Diffusers transformer。RAU链由两对down、三对up的affine替身组成，无真实learned resnet。小tensor与合成root-size路由metadata明确分开；这些接缝不能叫完整SDXL请求。

标准库整理器独立实现坐标映射、half-pixel/edge-clamp cubic、ceil pooling、逐元素卷积、全序列均匀参考和全部阈值/时钟顺序。7个window、178个conv调用、74对down/up chain的输出/skip/hidden全数组重放；分别核对4076、13632、31878个输出scalar槽，另核原始回调输入/输出、legacy及状态。conv参考数值精确相等，不另授予signed-zero逐位资格；全部所核数组最大绝对偏差约`4.18e-7`，不是只抽查首尾或只比shape。保留全部值、环境、hash及180秒bounded child正常退出。

四个偶数窗口seed均实际进入不同shift档，self输入为`[8,24,2]`，full参考为`[2,96,2]`；cross/FFN均保留`[2,96,2]`。均匀attention微型fixture仍有约10.65%–18.72%的最终向量相对差，说明窗口删除确实改变函数；这**不是原图像的质量下降百分比**，也不是判断实际方法不可用的画质证据。

## 必须保留的反例和状态边界

- 奇数`7×9`为分窗先bicubic到`8×10`，逆变换再回`7×9`；即使attention为identity、FFN为零，最终与无resize的`2*x`对照仍有最大约0.184差。多出来的是重采样损失，不是attention量化误差。原警告保留为类别；不能把odd路径当可逆排列。
- `4×16`时随机选择请求shift`(0,6)`，但原条件只判断第一个分量是否大于0，实际两轴都不roll。独立参考保留这一行为，不偷偷修成理想窗口规则；这只是小尺寸控制，不是默认高分辨率退化证明。
- 实际改变全局CPU RNG state，独立创建但未传入该method的generator不变。模型/pipeline generator相同不能自动证明窗口随机流相同；复现实验须固定并记录这一额外随机源，不能只对latent generator设seed。
- 不一致size与未分窗的全局mask显式控制均在完成回调前拒绝；源码把原mask传给窗口attention，没有做通用mask重排。这些非默认输入不授予任意mask/only-cross/joint-attention适配资格。
- 合法modern conv成功后恢复stride/padding/dilation。但在临时改参后注入错误dtype导致conv异常，会留下`4/2/2`与clock0；同对象重试又把变更后的值保存为“原值”，一次成功后仍为`4/2/2`。这是明确无效输入的异常安全控制，不是正常推理必现故障。
- 选择legacy分支、使用形状匹配的零LoRA替身时，连续两次return均发生在恢复/clock之前，留下clock0和stride4；本轮未安装或运行真实旧版Diffusers/LoRA模型，不能泛化到现代PEFT。
- 官方名称的微型结构scaffold上，apply/reapply使`num_upsamplers`从3到15再27；remove恢复类、移除hook，却保留27和info。两个开关都false仍增加12并装hook；unsupported名称在抛错前已改计数/装hook。本轮只证明原patch方法在该scaffold的状态变化，不声称重现了完整Diffusers卸载/请求输出；部署包装应事先验证、保存原值并提供显式恢复事务。

## Apple迁移：可以学什么，不能直接抄什么

以下是从源码/成本模型得到的迁移建议，而非新的MPS/MLX/ANE实测：

1. 先分开测RAU-only、window-only、组合和未改full强基线；相同步数/条件/最终分辨率，分别计pack/roll/restore、pool/resize、所有projection、cross/FFN及完整denoiser。不要乘两项局部倍率，也不要把减少pair数当减少NFE。
2. MLX/Metal研究可把四窗变batch，保留fused dense SDPA，避免为了画出sparse mask构造完整N×N。真实收益仍取决于布局copy、短矩阵利用率和GPU后端；本轮CPU可运行不证明MPS端到端更快。
3. U-Net层级/skip size是合同。原key表绑定SD1.5/2.1/SDXL/Turbo，不能套到FLUX/Qwen/Wan的joint tokens、文字/图像前缀、RoPE或视频轴。原方形/矩形推断也不应代替明确H/W/token映射。
4. ANE方案应先研究有限静态shape buckets及成对skip/resize边界，不在每步重新编译任意尺寸。新增图/转换/驻留成本必须进预算，不能从公共primitive名字推导实际ANE放置或加速。
5. 与cache组合会改变history的shape与空间含义。应把分辨率phase写入cache身份，在切换点刷新；本轮没有执行组合cache，不能假设旧residual插值后等价或与VSA/ToMe误差互相抵消。

后续[第二十一轮完整Tiny U-Net消费者](../case-studies/hidiffusion-unet-consumer.md)已实际导入原包并使用真实Diffusers随机网络贯通19条轨迹，保留一条odd组合拒绝、取消/reapply的真实状态控制，以及metadata不同但remove后同输入数值相等的限制；没有把本轮替身资格直接转授给完整模型。仍需预训练模型/多prompt/长宽比/步数和媒体质量预算；GPU工作先检查竞争负载，没有默认启用建议。

默认两条仅做标准库重放/原创数学合同，不下载、导入Torch、读取模型或提交GPU。实际原方法probe需显式`--run`、固定checkout及全新输出目录；AST裁选不是安全沙箱，也不冒充项目原生测试套件。
