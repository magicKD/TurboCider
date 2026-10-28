# Qwen512 VAE：无损名称规则复用与 FP16 卷积岛小幅收益

2026-10-18，Asia/Singapore，M4 Max/64GiB、macOS26.6.2/build25G83。
只涉及 Qwen Image 2.1 文生图/图生图，不修改 Z Image、已安装 App 或 App 默认设置。
完整优化目标保持 active。

后续 [FP32融合与完整回退验证](qwen512-vae-fused-norm-2026-10-18.md)补充了真实
透明范围数值用例：FP16组合可产生非有限输出，当前原生执行已加入完整FP32
卷积重算并如实记录耗时/实际精度。这里的近不透明照片观察不能外推为透明图资格，
也不为旧构建回填新回退能力。

## 结论

新增默认关闭的研究开关 `TURBOCIDER_QWEN21_VAE_FP16_CONV=1`，只把 decoder
卷积的 operands/output 改为 FP16，立即恢复 FP32 后加 bias；状态、残差、
L2 normalization、attention算子仍 FP32，参考图 VAE encoder 完全不改。
原始 BF16 checkpoint不被改写，FP16系数 view单独拥有，并检查其有限性。
该路线是近似模型，不能叫 BF16 原模型的逐位等价实现。

本轮完整请求名义减少约1.18%/1.35%，质量门槛通过，但**未达到事前3%整请求
收益门槛**，不晋升常规路线或 App 默认。研究入口保留用于后续对照；默认关闭。
另保留纯 CPU、无损的权重名称规则复用，实际约省1.4–1.6ms/次VAE映射；
不能把这个阶段约20%的改善说成整图快20%。

## 实现范围

- `VAE::DecodeConvolution::Original`是默认；FP16 compute view只覆盖
  `decoder.*.weight`及`post_quant_conv.weight`。encoder/quant_conv/gamma/bias
  保留原数组，没有对整个VAE或输入参考图统一改 dtype。
- Session在请求开始 snapshot flag，planner和执行使用同一0/1解析/资格检查。
  要求显式approximation、resident512普通GPU、无PE/streaming/预算/ANE；
  当前研究仅原BF16组件，排除另一路BF16 streaming executor。非法值拒绝。
- 实际运行后回执同时记录专用selection和`+vae_fp16_conv_fp32_state`；
  prepare-only不冒称执行过decoder。不增加请求schema字段、不保存App新preset。
- VAE名称转换原本每个key都重新构造15条regex，现在只复用不可变CPU规则。
  pattern、顺序、replacement完全不变，不缓存权重或GPU graph。

## 先测组件，不能替代完整请求

使用原644MiB BF16 VAE、既有r256狐狸真实latent，以及新r128茶壶生成/编辑
各自保存的真实latent，三次独立进程。每次原/FP16各一项warm，再各四项计时；
每次计入VAE构造、FP16系数转换/finite检查、完整RGBA decode及输出finite，
模型加载、readback验证与PNG导出在外。三个component共30次完整decode。

原VAE约0.330秒，FP16岛约0.309–0.312秒，只少约19–22ms。单独组件与完整
pipeline的内存/调度上下文不同，不能用它替代下表的request分母或推导唯一慢因。
三个输入的参考图encoder对照均逐位相同。238个真实checkpoint key及510个
有界名称案例均与旧规则一致；四次换序CPU mapping中位约6.92–7.12ms→
5.50–5.52ms，省约1.4–1.6ms，不是显著整图突破。

## 同库、同student、同强prefix控制的16请求

原 BF16 encoder/DiT/VAE，已pin的 Viggle v0.2.1 r128/strength1/inference-time，
joint BF16 A/B operands/F32 ranks，实际六步及227 adapter terms，512²，seed73/74。
没有ANE、DiT residual cache、FFN时间复用或adapter合并。

一个resident Session内按原→FP16→交错控制→原恢复运行。文生图、单ref512
编辑各8请求，前两项为warm/prime，不计分；每任务每臂3个完整计分请求。
计分conditioning-hot，编辑两臂全部实际prefix snapshot hit。首项原精度保存
raw latent/pixels，因此dump不在计分窗；首个不dump的编辑单独填充prefix。

| 完整请求，含导出 | 原精度中位 / s | FP16卷积岛中位 / s | 名义减少 |
| --- | ---: | ---: | ---: |
| 文生图 | 7.469350 | 7.381009 | 1.183% |
| 编辑，两臂prefix hit | 7.630051 | 7.526937 | 1.351% |

对应decode阶段原约0.592–0.621s、FP16约0.463–0.501s。阶段差不等于完整
request差，不能从主机时间相减推断GPU kernel或唯一资源瓶颈。小样本、一次
有序窗口，不是长期稳定性或独占设备资格；不与rank/prefix/缓存收益相乘。

## 质量、decoder-only归因与默认回归

事前像素门槛：RGB SSIM≥0.995、MAE≤1/255、alpha max abs≤2。四组匹配：

| 请求 | RGB SSIM | MAE /255 | RGB max abs | alpha max abs |
| --- | ---: | ---: | ---: | ---: |
| 生成seed73 | 0.999638 | 0.054170 | 6 | 1 |
| 生成seed74 | 0.999646 | 0.050696 | 7 | 1 |
| 编辑seed73 | 0.999772 | 0.063283 | 3 | 1 |
| 编辑seed74 | 0.999735 | 0.053981 | 3 | 1 |

这是当前少量近不透明照片，不是任意图像/透明边缘验收；baseline alpha范围
247–255，未覆盖完整透明抠图分布。人工查看狐狸、蓝茶壶及红茶壶原/候选，
未见新增明显差异；原编辑材质/构图偏差仍存在，不称编辑指令或全部场景合格。
所有返回原/FP16状态的匹配PNG各自逐位一致；关闭flag的5个代表请求PNG
与上一稳定r128 library逐像素一致。

另开2个**仅用于parity、不得当测速分母**的FP16请求保存raw：生成、编辑的
实际denoised BF16 latent与原精度逐位一致；单独decoder重放分别复现原/候选
完整请求PNG。由此确认本批差异仅在VAE decoder，不是改了DiT、步数、LoRA
或reference encoder，也没有借较弱编辑prefix控制。

`outputs/qwen512-vae-audit-20261018.{py,json}`独立CPU核对648项native source
hash、实际loaded runtime ID、raw latent/像素及PNG，复算request质量/时间摘要。
CPU复核不重跑设备，也不授予物理placement或长期图像质量资格。

## 构建、验证与资源

```text
build/qwen512-vae-conv-20261018/libturbocider.dylib
SHA256 153d49f6d84dd2e9ede1c5bf152a959f11623355fee2e2958aa4c6665c0025ca
build/qwen21-vae-precision-20261018/qwen21-vae-precision-probe
SHA256 2a28dff667d51edd2b026a7455e3246e25d2312804f8ddc72bc8ada736c9adea
```

Private-enabled native库构建成功，实际本轮全GPU，没有ANE执行。
contract132项：131通过、1跳过；真实GPU lifecycle1项、options host1项、
precision-screen host4项及无设备510项key contract通过。实际3份组件各自再
验证238个真实key、原decoder重放exact、reference encoder exact。
不是全仓或整个VAE/故障/长时/所有adapter资格。

第一次full runner用了错误的`dump`字段，在模型推理前被API拒绝；修正为正式
`dump_tensors`，增加host断言并新开v2目录。原失败原件保留，没有重写为成功。
新contract最初过度限定错误字符串，被先行的ANE通用guard拒绝；修正测试只在
decoder guard确实先执行时断言该信息，不放宽任何请求接受条件。

五个成功窗口（3component、1full、1parity）memory evidence complete，系统
swap-in/out均0；full峰值31,936,109,416 bytes，parity峰值32,040,655,792 bytes。
同进程交错臂不能从峰值相减给各dtype独立归因。全部持续CPU-load检查失败，
qualification保持false；未停止用户ComfyUI或其他进程。

没有下载新模型、重写checkpoint/adapter、清理用户缓存或替换已安装App。
原始目录为 `outputs/qwen512-vae-fp16-*20261018/`，含失败/源码、16+2张真实
请求图、30次decoder观察及每个case的原/末次候选raw/reference、source、
stdout/stderr和load/memory。
磁盘约17GiB可用，owned构建/推理/采样均已结束。

## 保留决策与下一步

保留无损CPU规则复用及默认关闭的FP16研究路线和复现工具；因未达到3%整请求
门槛，不对外称已取得可推广的提速，不改变App默认精度。下一轮应调查更大
VAE边界（真实layout、norm/activation/state流量）或主要GPU算子，而不是从
当前约1%的窗口选择单个最快样本。若尝试FP16 state，必须另立配方并覆盖真正
透明图像，不能复用本次FP32 state门槛宣称新精度已经通过。
