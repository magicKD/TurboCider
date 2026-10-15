# Qwen encoder runtime 继续优化：小幅完整收益

2026-10-08，M4 Pro / 48GiB，`dev-graph`。本轮延续上一轮 encoder runtime；只使用本地模型，两个有限设备实验组，没有DiT/VAE/编辑图测试。详细改造及下一步见[设计20](../design/qwen-image-gpu-ane-command-graph/20-encoder-runtime-optimization.md)，结构化回执见[JSON](2026-10-08-qwen-encoder-followup.json)。

当前结论：旋转W8A8候选在完整三图encoder出现 **3.93%** 净省时，语言部分省时 **10.09%**；没有达到预声明完整路径5%性能筛选，只有单样本，不授自动路由或发布资格，默认GPU。

## 实现和验证

- [encoder_runtime.hpp](../../native/models/qwen21/encoder_runtime.hpp)：BF16源operands＋FP32 MPP partial、动态权重compiled FFN复用、取消导入前额外全stream synchronize，保留eval与finish/drain；完整fallback和source所有权不变。
- [encoder_runtime_policy.hpp](../../native/models/qwen21/encoder_runtime_policy.hpp)：进程固定的`TURBOCIDER_QWEN21_ENCODER_RUNTIME_GPU=mpp|legacy`；只影响显式encoder诊断入口。
- [probe](../../tools/native/qwen21_ane_encoder_probe.mm)：物理range／动态权重切换检查、graph构造／调用和input readiness计数；同构建GPU／legacy／mpp-register对照。完整候选使用已有寄存器Hadamard staging，未增加SmoothQuant profile或next-layer provider。
- 原生库/CLI和Swift App/helper编译通过，51项CPU policy、CPU MIL geometry、encoder实际后端／fallback结果检查通过；4个小型BF16／FP16 partial与动态权重rebind检查通过，RRMS≤0.01%。

测量库与最终代码相同：
`tc-runtime-build-v1-fb2118ce83b7466902c755b8b70831549f3532b3b86e8faa76db65f5d3ba242a`。
574份native身份source前后匹配，probe／库／脚本／plan哈希前后匹配。App仅编译，无UI验收／安装。

Splash当前HEAD固定为`0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7`。其[AneFfn实现](https://github.com/incoai/splash/blob/0d5a035ea26d1cc566e3c7ee4e08e5e06fa68cf7/runtime/ops/AneFfn.cpp)是随机符号Hadamard旋转W8A8，不是SmoothQuant校准；动态INT8权重／activation和FP16 scale／carrier不构成硬件INT8指令证据。最新直接调用接口要求hidden为2560的整数倍，本地H4096不符合；我们的shape与dense BF16 staging是本地适配。

## 真实 FFN

1071／2101／3131行来自完整视觉塔和语言layer0 attention。每个shape一次warm和AB／BA，计时含weight stage、input ready、GPU+ANE、join与设备完成；构造另列。

| 参考图数 | 新6144通道省时 AB / BA | FFN RRMS |
|---|---:|---:|
| 1 | 20.87% / 19.48% | 0.748% |
| 2 | 27.30% / 24.17% | 0.745% |
| 3 | 22.37% / 22.38% | 0.740% |

3072候选三图约省9.74%／10.77%，没有推进完整encoder；6144经预声明≥5%组件门槛推进。

[组件supervisor](../../outputs/qwen21-ane-encoder-followup-20261008/screen-supervisor.json)包含两个候选的全部正／负配对。legacy一图6144的AB省17.56%、BA慢1.26%，保留波动。程序缓存先由legacy建立，再被新arm复用；组件构造0.95–1.44s与新arm约0.21s不能归因于新kernel，也不作为首次运行性能排名。

## 完整 encoder

完整27层视觉×3＋36层语言，mRoPE／DeepStack／retained pre-final-norm输出，逐层释放权重。GPU→legacy→mpp-register，各一个fresh process；使用组件准备的private program cache，OS/source/driver cache未重置。

| 路径 | 视觉 | 语言 | ANE准备 | encoder总耗时 |
|---|---:|---:|---:|---:|
| GPU | 4.546s | 7.327s | 0 | 12.144s |
| legacy | 4.555s | 6.938s | 0.277s | 12.048s |
| mpp-register | 4.550s | 6.587s | 0.249s | 11.666s |

新候选相比GPU总省3.93%、语言省10.09%；相比本轮legacy总省约3.17%。每个hybrid都是36个成功FFN、108次ANE evaluation，无fallback。compiled callable构造36→1，调用仍36次，不表示已获得整个MLX Metal command graph的控制权。

新旧hybrid retained conditioning的184,320个F32记录逐字节一致；相对GPU RRMS均 **5.75399%**。新计划在运行之前按此前用户允许的5–8%范围声明8% encoder数值筛选，已通过。[上一轮5%筛选失败](2026-10-08-qwen-ane-encoder.md)保留，未追溯改门槛。没有最终图片质量或多prompt泛化结论。

新arm的input readiness2.599s包含上游attention/norm，async ANE host wait3.359s与GPU重叠，stage1.127s受queue/带宽争用影响，post join0.059s；不能相加成独占设备时间。legacy stage0.622s，组件stage变快并未证明完整encoder staging也更快。legacy graph host encode计数未单独计时，0不是实测零成本。

视觉仍约4.55s，程序准备约0.25s，且1152 bucket使3131行补到3456、逐层三次ANE调用。这解释了单FFN约22%收益为何只对应完整encoder约4%。本轮不与上一构建13.202s直接配对；本轮legacy已接近GPU，表明缓存／顺序等影响不能忽略。

[完整supervisor](../../outputs/qwen21-ane-encoder-followup-20261008/full-supervisor.json)的`status=pass`表示执行及源复核通过，`encoder_candidate_pass=false`表示仍未达到完整性能筛选。`qualified=false`保持不变。

## 资源、清理和后续

两组设备工作总wall约 **57.98s**（组件21.85s、完整36.14s），无重试／长期扫描。前后swap都是4581.69MiB，系统未记录thermal/performance warning；没有逐毫秒采样，不能排除中间短暂变化。MLX peak约2.16GiB，未包含private IOSurface／driver；与legacy相同，不声称进程peak降低。

清理本轮owned中间对象／static archive／module cache及未选3072程序source，共 **215.58MiB**。只保留6144 geometry的小程序source cache **6.01MiB**、运行库／CLI／App／probe、日志及约2.11MiB conditioning。缓存里的Hadamard constants不包含模型权重副本；未清理系统／模型／其他轮次缓存。[清理回执](../../outputs/qwen21-ane-encoder-followup-20261008/owned-cleanup.json)与[最终校验](../../outputs/qwen21-ane-encoder-followup-20261008/final-verification.json)可复核。

下一步按[设计20](../design/qwen-image-gpu-ane-command-graph/20-encoder-runtime-optimization.md)：拆分可复用program和request source绑定，筛选1664行bucket减少108→72次evaluation，建立唯一input/import fence，把producer-ready next-layer staging放在当前GPU提交之后，再建设完整Native encoder graph。现有prefetch会在当前GPU提交之前eval懒加载next weights，不盲目启用。质量方面单独做36层encoder SmoothQuant／rotation校准，避免使用32层DiT profile。本轮没有执行这些新增候选的设备测试。
