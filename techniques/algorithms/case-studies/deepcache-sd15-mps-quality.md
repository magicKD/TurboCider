# DeepCache真实SD1.5 MPS生成：约3倍不等于通过质量预算

<!-- knowledge
id: technique-algorithms-case-studies-deepcache-sd15-mps-quality
type: case-study
stage: dit
operator: attention, conv, sampling, layout
backend: metal, cpu-reference
device: gpu, cpu
symptom: stale-cache, launch-bound, measurement
semantics: approximate, contract-only
representation: fp32
boundary: producer-consumer, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [U-Net源码/MLX边界](deepcache-unet-boundaries.md) · 冻结计划 · [来源 S70/S72/S73](../sources.md#s70)

2026-10-17，第26轮。本轮不是随机模型或latent-only：实际执行学习过的CLIP、859.5M参数U-Net及83.7M参数VAE，512²完整PNG生成。原DeepCache i5b0约2.91–3.00倍完整请求收益，但所有有损臂均未通过预设RGB保真预算；SSIM最低约0.179。没有据速度或preset名称授予采用资格，也没有调阈值让坏样本通过。

## 模型身份、精度与窗口

本地没有兼容U-Net资产，因此取得[公共SD1.5镜像](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/451f4fe16113bff5a5d2269ed5ad43b0592e9a14)的13个固定文件，约2.0GiB。asset plan含revision、长度、Git blob/LFS SHA；三个safetensors的完整payload SHA以及配置/tokenizer/card逐项校验，前后不变，不只是bounded header。镜像card明确自称非官方镜像，许可标签CreativeML OpenRAIL-M；保留来源/许可，不向Wiki分发权重、实际激活或图片。不是新的官方发布/原始模型签名证明。

权重文件是fp16 variant，**计算统一FP32**；不是原始全FP32 checkpoint等价声明。所有学习组件完整加载，info无missing/unexpected/mismatch/error；旧文件的77项position buffer与新CLIP的非persistent buffer相等。UNet859,520,964、VAE83,653,863、CLIP123,060,480参数，1130项状态fingerprint前后一致。

M4 Max/64GiB、macOS26.6.2/build25G83、Python3.13.12、Torch2.14.0声明08187d9/MPS wrapper匹配、Diffusers0.39.0、Transformers4.57.6。20个安装Diffusers Python文件和3个CLIP文件匹配固定源码，不是完整native/wheel闭包。Host SDK26.5不是Torch构建SDK。

采用50步DDIM、CFG7.5、eta0，继承模型beta与`set_alpha_to_one=false/steps_offset=1`，唯一t表为981到1、间隔20。**不是模型默认PNDM/论文PLMS或作者0.24定制pipeline复现**。两提示词复用此前茶壶/自行车研究模板，每词两个预定seed（42/73、43/74），不是独立语义覆盖集。

worker/launcher每cohort用一个新owned进程、相同resident模型。每次从CPU显式seed生成同一初始latent并copy到MPS，均在计时外。计时内包含helper enable/cleanup、原公共pipeline的真实正/负CLIP各一次、50次UNet入口调用和DDIM更新、VAE/PIL、共同软件trace/末latent与VAE finite检查、PNG保存及末端Torch同步；pipeline子窗口另存。加载/准入/图片指标在外，不是冷启动、无仪器化API或纯kernel时间。VAE未tile/slice，未量化/LoRA/缩分辨率；只用预定良性物品提示词，未下载/运行safety checker，非安全或生产评估。

## 48个计分请求，所有正反方向与拒绝保留

四cohort各先一次full warmup排除，再按六臂正序/反序各一次。正序配第一个full，反序配最后一个full，不挑较快oracle。共48计分、4warmup排除，另有2pilot排除：54次完整生成，计分2400次outer UNet调用。所有模式/重复的模型、输入、时间、geometry、CFG相同。

`full / i1b0 / i3b0 / i3b2 / i5b0 / ordinal-i3b0`中，interval1不跳算；i3深层刷新17次、缓存33次，i5刷新10次、缓存40次。所有臂outer UNet仍50次；原helper没有强制末步full，i3/i5最后step49确实hit。软件刷新数不是GPU kernel计数。

各模式的两次PNG/RGB重复一致；i1b0对matched full、ordinal-i3b0对原i3b0的RGB与末latent bytes均相同。Warm/pilot不进入下表。每个数是两方向`paired_full_time / candidate_time`的中位数，不是中位时间相除；n=2不是置信区间。

| cohort | full秒（中位） | i3b0加速 | i3b2加速 | i5b0加速 | ordinal-i3b0加速 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 茶壶42 | 12.433 | 2.213× | 1.272× | 2.969× | 2.291× |
| 自行车43 | 12.405 | 2.160× | 1.256× | 2.910× | 2.269× |
| 茶壶73 | 12.565 | 2.231× | 1.282× | 2.998× | 2.305× |
| 自行车74 | 12.426 | 2.214× | 1.278× | 2.993× | 2.296× |

i1b0对full约0.984–0.987×，显示即使输出等值，wrapper/cache存储也有净成本。不能拿较慢的i1当主分母放大缓存收益。完整请求和pipeline子窗口都在JSON保留；不乘以前几轮其它模型的组件倍率。

## 所有有损臂都被预定参考图预算拒绝

正式出分前冻结两个本地预算：RGB SSIM≥0.95且PSNR≥30dB；更严一档为SSIM≥0.98且PSNR≥30dB。它们只是本实验相对同seed full图的保真预算，不是感知/语义或生产标准。

| cohort | i3b0 SSIM / PSNR | i3b2 SSIM / PSNR | i5b0 SSIM / PSNR |
| --- | --- | --- | --- |
| 茶壶42 | .7776 / 17.44dB | .7751 / 17.30dB | .7246 / 15.34dB |
| 自行车43 | .6564 / 19.45dB | .6665 / 19.36dB | .5136 / 17.96dB |
| 茶壶73 | .8539 / 19.44dB | .8451 / 19.67dB | .7459 / 16.31dB |
| 自行车74 | .2523 / 12.09dB | .2631 / 12.09dB | .1791 / 10.91dB |

ordinal臂与i3b0完全相同。32个计分有损输出对两个预算共64个决策全部拒绝；通过的32个决策仅来自full/i1 controls。没有删掉自行车74、改prompt/seed、增大可接受误差或把RGB指标换成较有利的分数。更深branch2成本大得多，却并非每个指标/seed都更接近full。

人工查看茶壶42和自行车74的正反接触图：缓存能改变主体数量/轮廓、墙面颜色与车轮/构图，不只是纹理。full茶壶自身也有不自然形状、颜色不完全符合文字；某些缓存图反而更蓝。**full不是语义正确groundtruth**，SSIM/PSNR不代表提示词符合度排名；本轮拒绝的是“按预定预算保留full图”，不是证明缓存图在所有审美/语义上更差。无盲评、CLIP/LPIPS新模型或视频时序评分。

## ordinal-clock是受限实现优化，不是质量修复

原helper每次对t取host值并在scheduler时间Tensor列表中找位置。变体只给私有实例替换root时钟：预先在CPU确认固定50个时间唯一，保持每步一个B2 UNet调用及所有原block缓存策略，按调用ordinal写`helper.cur_timestep`；实际50个clock/refresh条目及t表再次核对。原DeepCache包/安装库未改。

该变体对原i3b0配对中位约1.033–1.050×，不是上一轮小随机VDE循环的1.4倍，也不是通用scalar复制成本。它删掉了列表构建/值比较/host依赖等整条控制路径，未做物理同步计数或profile。输出逐位等值只说明本合同下实现保持同一近似轨迹；它同样未过相对full的RGB预算。

不得直接推广到重复时间的PNDM、partial Img2Img、额外UNet调用、CFG unbatched或会改变condition/schedule的callback。把phase坐标与数据validity变成显式合同，才是可迁移设计技巧；不是默认启用ordinal或缓存的理由。

## 验证范围与资源记录

PNG CPU阶段重新读取全部52个正式进程PNG/RGB原像素，保留整数直方图/moments、PSNR及Gaussian SSIM11/sigma1.5/population/valid502×502×3；28个有序内容对去重仍检查每次引用。两个CPU实现逐元素核对全部distinct SSIM maps，门槛1e-9；不合成alpha、不resize度量图，不是两种感知模型。

stdlib整理器重放158行完整journal、48计分/4warmup、所有输入/时钟/参数/图片身份和pixel aggregates；默认不读取PNG/权重或导入框架。**aggregate复核不等于stdlib重新计算全部原图像素，也未独立重写整个CLIP/U-Net/VAE数学**。

GPU每次请求前连续三样本Device≤5%，native 60个通过/46个暂缓窗口完整保留，可含自身冷却。另有两个outer启动在模型加载前被拒，原`idle_window`只保存异常消息，没有留下那6个失败原始样本；明确保留此证据缺口，不补造0值、不计成外部作业。确认空闲后只对未执行cohort用新目录重试，已完成cohort不重跑。RAM16GiB粗reclaimable-page准入不是预留或峰值证明；无持续独占、物理dispatch/all-op GPU/峰值内存资格。

前两条只读公开JSON/stdlib，不导入模型或执行设备。真实复跑需明确flags、新目录、指定本地文件及新鲜准入。资产收集器只下载固定数据/safetensors，不安装包/执行远程代码；初次partial文件名的header摘要检查补正后按同asset plan重验现有文件，未另下/替换权重。Pilot客户端/时序检查扩展前快照单列，所有pilot计时排除。

当前结论：Apple MPS确能从U-Net深特征跳算取得完整请求收益，但本数据的有损配方均不满足所选保真预算。采用前需在实际任务/更多prompt-seed、正确语义指标与相同质量预算下重新校准；不能因为一条缓存代码可执行或约3倍更快就称损失已受控。产品默认未改。
