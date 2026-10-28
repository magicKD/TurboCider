# Qwen512 VAE FP32融合：局部改善、整请求小幅收益及透明回退

2026-10-18，Asia/Singapore，M4 Max/64GiB、macOS26.6.2/build25G83。
仅 Qwen Image 2.1 文生图/图生图；不改 Z Image、已安装 App 或默认配置。
完整优化目标仍 active。

## 结果与保留决策

新 `TURBOCIDER_QWEN21_VAE_FUSED_NORM=1` 默认关闭。它保留FP32状态与原系数，
将channel L2 norm、可选SiLU及卷积输入NHWC布局合在一个Metal kernel。
encoder不变。FP32求和顺序不同，不冒称任意输入逐位等价。

真实潜变量的独立VAE约0.331→0.284秒，但完整六步r128请求只少约0.9–1.0%。
与原有FP16卷积岛组合的完整请求少约2.4–2.7%；相对已经较快的FP16控制，
新增融合的增量只有约1.0–1.1%。都未达到预设3%整请求门槛，保留研究入口，
不晋升App默认或宣传可推广倍率。

本轮还发现可选FP16 decoder在透明数值输入上会产生非有限输出，现已加入
**完整原精度卷积重算**。回退不是加速，其失败尝试与重算时间全部计入；
实际precision/selection不能继续冒称用了成功的FP16路径。

## 先定位成本，再改数据流

沿用VAE原本五个`eval`完成边界，未增加逐算子同步。相同真实r128生成latent：

| 边界 | 原FP32约/ms | 只有FP16卷积约/ms |
| --- | ---: | ---: |
| post-quant/middle/block0 | 34.8 | 29.0 |
| block1 | 58.1 | 45–47 |
| block2 | 50.1 | 43.5 |
| block3 | 66.6 | 59.2 |
| block4，512² | 109.7 | 108.6 |
| 最终head及finite | 5.2 | 5.4 |

FP16对末段几乎无帮助，因此测试norm/activation/layout融合，而非继续只改
卷积dtype。新核一组SIMD处理一个spatial pixel，保留完整channel归一化、
`max(sqrt(sum(x*x)),1e-12)`、原gamma和FP32状态，输出紧凑NHWC，再以NCHW
view交回原API。gamma只做很小的owned FP32 view，BF16数值扩大不损失精度；
不compact大型输入，不改帧/像素/空间上下文，也不把它称物理零拷贝。

## 被真实模型否决的初版：lazy stride不可先猜

初版在host构建kernel时把`x.strides()`烘焙为模板参数。小型已求值输入可以
通过，真实lazy decoder输入却可能在求值时改变物理布局，导致严重错误；
真实PNG SSIM约0.493、alpha max差254，未授予该候选任何性能收益。
该研究库`05e194d4…`与完整失败质量原件仍保留，不覆盖为成功。

修正为MLX执行时传入的`x_strides`，不使用host的预求值快照。扩展实际GPU
回归到214项，含全部真实gamma宽度96/144/192/288/384/576/768/1152，
NCHW与NHWC物理布局、非零origin、BF16/F32 gamma、with/without SiLU、
zero/tiny及未预先求值的producer；其中104项确实发生过求值后stride变化。
对独立FP64全输出/原MLX数学的最差relative L2约1.15206e-7，门槛2e-6不变，
另5项非法契约拒绝。初版还曾错把标量作数组访问、偏移gamma用例失败；
分别修正标量接口及显式owned系数view，未放宽数值门槛。

## 独立VAE及完整请求，不混淆分母

使用原BF16 VAE和真实r128生成/编辑latent。每组件原/候选各一warm、四计时；
构造、系数准备、decode和finite均计入，模型load/readback/export在外。
纯FP32融合约0.331→0.283–0.285秒，组合约0.331→0.266秒。纯融合末段
约109.6→85.6ms。组件数字不外推为整图，亦不拼接不同窗口的最快阶段。

三组完整cohort各16请求，共48张实际请求图：同原BF16组件、Viggle v0.2.1
r128/strength1/inference-time/227 terms、joint BF16 A/B/F32 ranks，512²，
实际六步，seed73/74。无ANE、residual cache、FFN时间复用或权重merge。
每任务每臂3个计分样本，其他为warm/prime；conditioning-hot，编辑两臂
均实际immutable prefix hit。三个窗口独立，不能视为交错ABBA或长期稳定性。

| 候选 / 匹配控制 | 文生图控制→候选 / s | 名义减少 | 编辑控制→候选 / s | 名义减少 |
| --- | ---: | ---: | ---: | ---: |
| 纯FP32融合 / 原精度 | 7.490154→7.417182 | 0.974% | 7.628975→7.558714 | 0.921% |
| FP16卷积+融合 / 原精度 | 7.535558→7.334825 | 2.664% | 7.637191→7.457541 | 2.352% |
| 同组合 / 较强FP16控制 | 7.381975→7.298717 | 1.128% | 7.531580→7.454666 | 1.021% |

后一行才是给原FP16路线加融合的增量，不能把全部2.7%归给新kernel或把
历史倍率相乘。全部48个照片请求未触发precision fallback；每臂返回态PNG
各自exact。纯融合四个seed/task的RGB SSIM≥0.99999938、RGB/alpha max差≤1；
组合相对原精度RGB SSIM≥0.999641、MAE≤0.064/255、alpha max差1。
这是少量照片观察，不是所有文字、身份、材质或自然透明图像资格。

## 全alpha数值输入与必要的完整回退

另建确定性RGBA数值fixture，输入alpha0–255，含145,933透明、108,111半透明、
8,100不透明pixel。它是合成测试数据，不冒充真实prompt生成或画质验证集。
由**原VAE encoder**得到实际latent后，用相同权重/latent比较decoder，计时
不含这一步encode。原解码alpha也覆盖0–255。

纯FP32融合通过，RGB SSIM0.99999963、RGB/alpha max差1。FP16组合首个尝试
却产生非有限输出，原失败窗口保留；没有用黑图、clamp或只看RGB掩盖它。

`decode_checked`只对可选FP16结果的非有限信号重跑整个decoder，局部拷贝
owner/metadata并清除该拷贝的FP16 compute view，原系数不动，保留独立选中的
FP32 norm。其他输入/设备异常及取消照常传播；原精度若仍非有限则拒绝。
回执记录完整回退和实际输出精度，失败尝试/重算都在request wall内。

同真实alpha fixture的5个候选调用全部触发完整回退，每次5+5阶段；最终PNG
与纯FP32融合逐位一致。含重算中位约0.545s，慢于原约0.331s，
`candidate_speed_claim_allowed=false`。因此组合路线不能保证透明输入加速。

另有小型合成权重GPU契约，明确不是性能模型：有/无融合两次受控FP16溢出均
完整回到对应FP32输出；覆盖提前取消、失败后重试前取消、非法latent、原精度
非有限拒绝，以及随后原输出恢复exact。不吞掉异常或用fallback marker替代执行。

## 原始数据、构建和回归

6个额外parity-only请求覆盖三种decoder recipe×两任务，不用于测速分母。
实际BF16 denoised latent、text及edit reference数据全部逐位相同，并复现48请求
对应PNG；关闭研究开关的5张代表图与旧稳定r128库逐像素一致。
`outputs/qwen512-vae-norm-audit-20261018.{py,json}`独立CPU核对649项native
source hash、memory原件、raw及所有窗口摘要，并确认真实alpha回退结果一致。
CPU核对未重跑设备，不授予physical placement/overlap。

```text
当前研究库 build/qwen512-vae-fused-norm-v3-20261018/libturbocider.dylib
SHA256 8260a6ec032c193b1727d46b93ce185130293cd8f5bf11f173479d717f7363f3
```

保留初版错误库及修正stride但尚无回退的v2库作原件，不用于正常推理。
当前Private-enabled库构建成功，实际全部GPU；contract133项：132通过/1跳过，
真实GPU lifecycle1项、norm/fallback两项、options host1项、screen host5项、
510项无设备key契约通过。不是全仓或全部checkpoint/故障/长时资格。

三完整窗口memory sampling均complete，peak31,750,429,448–31,935,175,600 bytes；
纯融合窗系统swap-in65,536 bytes，其他两窗0，三窗swap-out均0。全部持续
CPU-load检查因竞争推理失败，qualification=false，未停止用户ComfyUI进程。
同进程交错臂不提供每arm独立peak，也不把logical tensor字节当DRAM流量。

原件位于 `outputs/qwen512-vae-norm-*20261018/`，保留错误质量、非有限失败、
修正版、完整/增量/RAW窗口、源码快照与全部日志/图片/数组/memory。
没有下载新模型、改写原模型/adapter、删除用户缓存或部署App；磁盘约17GiB可用。

## 后续方向

保留默认关闭的FP32融合和显式组合研究路径，以及必要的完整precision fallback。
没有达到3%采用门槛，不称新的通用最快配置。VAE局部改善对六步整图受限，
下一轮优先检查主要DiT成本或可验证的author-supported sampler profile。

已只读核对 `Viggle/Qwen-Image-2.1-viggle-turbo` 的固定README revision
`b667ee50696e8f4ccd161e02f8af0e43bb5665ed`：作者给出可选5步raw节点
`[1,.875,.75,.5,.25]`，不是任意裁步或换旧v0.1 adapter。该规则尚未在本机
实现/验证；若试验需单列新质量tier、真实5步回执、相同source/control及生成/
编辑质量，不把少一步当GPU kernel加速，更不将此候选视为已完成目标。
