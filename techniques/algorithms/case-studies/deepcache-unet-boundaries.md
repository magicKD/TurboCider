# DeepCache：U-Net分支边界、真实body跳算与MLX移植差异

<!-- knowledge
id: technique-algorithms-case-studies-deepcache-unet-boundaries
type: case-study
stage: dit
operator: attention, conv, sampling, layout
backend: cpu-reference, metal
device: cpu, gpu
symptom: stale-cache, wrong-layout, measurement
semantics: approximate, contract-only
representation: fp32
boundary: producer-consumer, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [真实SD1.5 MPS质量](deepcache-sd15-mps-quality.md) · [DBCache](../methods/block-caches.md) · [来源 S70/S71](../sources.md#s70)

2026-10-17，第26轮。DeepCache复用U-Net的高层特征，同时重算浅层路径；它不减少outer denoiser调用，也不是DBCache的middle residual相加，更不是VDE的整速度估计。作者[论文v2方法](https://arxiv.org/html/2312.00858v2)与[固定helper源码](https://github.com/horseee/DeepCache/blob/1691a3ab22308fe06685e4ff151a9ab175be2dc4/DeepCache/extension/deepcache.py)分开追溯。“almost lossless”是作者特定评估结果，不是所有函数/日程的数值保证。

## helper与论文实验入口不是同一套API

`DeepCacheSDHelper`直接替换已有Diffusers实例的forward，缓存各wrapped module的原始返回值（Tensor/tuple），包源码未改；root、resnet、attention、down/up sampler及mid有各自槽位。本轮标准四stage U-Net为51个保存的原方法、50个cache key，不是只存一个decoder feature，也不能把所有槽位numel相加称物理峰值（可能共享对象/存储）。

根目录`stable_diffusion.py`则使用作者定制`DeepCache.sd` pipeline和非均匀日程，README要求旧Diffusers0.24。它的`uniform/pow/center`参数不能移植成helper的任意`skip_mode`字符串。此次执行的是原extension，不执行定制SD/SDXL/SVD/DDPM代码；[官方文档](https://huggingface.co/docs/diffusers/main/en/optimization/deepcache)列出MPS路径只是导航，不替代本机版本验证。

## branch ID是结构坐标，不是质量阈值

`cache_block_id = branch_id // 3`，`cache_layer_id = branch_id % 3`。这里硬编码3，依赖常见SD结构的down两resnet、up三resnet等布局；换网络不能只照搬整数。

完整步：相对起始index满足`% interval == 0`，所有wrapped原方法运行并更新cache。其余步：更深block及mid复用；同深度的down在`layer>=boundary`复用，up在`layer>boundary`复用。up block和子层都按反方向重编号，父block可能先整块返回，子模块根本未进入。

因此branch0仍重算input conv/time embedding和最浅decoder尾部，能使用当前latent的浅skip，而不是整噪声重放；更大branch保留更多当前浅层计算。没有在线误差indicator、阈值或末步强制刷新，period不是输出误差上界。

## 真实随机U-Net消费者，不以root hook冒充计算次数

客户端完整导入原包，实际Diffusers0.39.0 U-Net/StableDiffusionPipeline/DDIM，207,404随机参数、batch2 CFG、8×12 latent，预计算合成embedding，未执行CLIP/VAE。17个安装Python文件匹配固定release；不是预训练模型或Apple性能。

13条10步轨迹完成，2条首步拒绝；标准库整理器检查130份完整预测payload的99,840个scalar，独立逐元素重放49,920个条件CFG-DDIM反馈值及36,530个leaf执行决策。网络预测是参考输入，未独立重算完整U-Net数学。

root/module `__call__` hook在返回缓存时仍可能触发，故统计未被helper包装的实际Conv2d/Linear子body。每个full步281次；10步完整轨迹中实际总调用25,376。以下是同一个Tiny模型的非刷新步，**不是GPU kernel数/FLOPs或预训练模型的相同计数**：

| branch | 保留的Conv2d/Linear body数 | 与full比较 |
| --- | ---: | --- |
| 0 | 20 | 最浅尾部 |
| 1 | 51 | 多保留一个micro-boundary |
| 2 | 82 | 重算整个浅stage的主要路径 |
| 3 | 100 | 进入下一stage |
| 6 | 181 | 保留更多深度 |
| 11 | 263 | 非刷新时mid仍复用，并非全full |

interval1全部10个预测与stock逐字节相同。正常从step0开始的warm新输入/新条件与fresh匹配；disable后与未patch pipeline匹配。这些正控不能转授给任意partial schedule、LoRA或异常恢复。

原始边界也保留：interval0→ZeroDivisionError；不支持的skip_mode首次就取不存在cache→KeyError；直接keyword-only UNet调用→IndexError，因为root wrapper读取`args[1]`；double-enable→RecursionError，运行中`reset_states`清掉原方法表→KeyError。后两者是刻意的生命周期压力，不把正常enable/disable说成默认失败。所有对象私有、原包/安装库未改。

## list.index时钟不是调用ordinal

原root wrapper先对`timestep.item()`取值，再用`list(scheduler.timesteps).index(value)`取第一次相等的位置。手工列表`[9,6,6,3]`配实际U-Net直接调用得到`[0,1,1,3]`，不是`[0,1,2,3]`；interval2的刷新布尔序列因此不同。这是明确列表fixture，不是执行PNDM solver。

所读Diffusers PNDM `skip_prk_steps`会产生重复时间值；不能泛称所有重复值都让interval3/5刷新错误，也不能把value-first位置和调用ordinal混为一谈。GPU时间张量上的`.item`及逐项比较还可能带来host依赖，实际成本另在[真实MPS对照](deepcache-sd15-mps-quality.md)测量，不从API次数推出物理同步数。

## 作者链接的MLX公开port：存在代码，不代表相同边界

[固定port快照](https://github.com/Barry0121/mlx_deepcache/tree/6b3564b95b41050a543597d477efad0da020e948)由Yiming Zheng、Xinran Zhang、Barry Xue署名，README引用Apple MLX示例并说明SDXL Turbo仍在开发。固定tree未发现LICENSE，文件有Apple copyright头；修改部分复用授权未确认。本轮只静态阅读、不运行该port/模型/发布基准，也不向Wiki复制其函数body。

| 维度 | 原Torch helper | 所读MLX port |
| --- | --- | --- |
| 边界坐标 | block与micro-layer，branch÷3/%3 | whole down/up stage的`cache_layer_id` |
| 保存内容 | 多个wrapped返回值/tuple | 最近完整步的一份`cache_features` |
| hit路径 | 当前conv-in及选定浅decoder尾部 | 重算到指定down stage，丢掉最后downsample residual，再替换decoder输入 |
| 布局/CFG | NCHW，Diffusers负→正 | NHWC，正→负，之后CFG |
| sampler | 跟已有pipeline | 此port的Euler/ancestral实现，默认SD2.1-base |

在四stage、2down/3up结构下，MLX layer0更接近Torch branch2的结构边界，而不是Torch branch0；这不是二进制cache兼容或逐位等价。MLX多算后丢弃的downsample图是否真的执行，取决于lazy图的消费者/eval，不能只数Python调用。CLI确实每步`mx.eval(x_t)`，本轮未观察实际kernel/leaf驻留。

静态接口还有重开项：SD I2I对`_denoising_loop`的positional实参从`text_time`槽开始错位；`type=bool`会把字符串`"False"`当True；非均匀函数的负底数分数幂在Python进入complex域，不能直接按论文实数公式理解。这里不给整个port或所有模式盖“运行失败”章：没有执行它，需在许可、版本和模型身份明确后逐项验证。`i in mx.array`的host循环控制也需单独测，不推定零同步成本。

默认练习：`python3 -B techniques/labs/run.py deepcache-contracts`只做原创stdlib gate/祖先/参数域合同，不导入框架或读取模型。重开条件包括U-Net层数/拓扑、processor/CFG顺序、scheduler重复/partial-start、callback/权重/LoRA、缓存失效、MLX许可/API和真实质量预算变化。
