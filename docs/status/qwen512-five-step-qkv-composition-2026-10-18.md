# Qwen512：五步采样与原缓冲区 corrected-QKV 的四组组合验证

2026-10-18，Asia/Singapore，M4 Max /64GiB。仅 Qwen Image 2.1 的512²文生图和
单ref512图生图；不涉及 Z Image，不替换已安装 App、不改默认值。持续优化目标
保持 active，本页只记录本轮完成的组合实验。

## 结论与比较边界

作者五节点采样器可以与 corrected-QKV 原缓冲区直读组合。七任务×两seed的
四组矩阵中，**28/28 同采样档 kernel 对的最终潜变量和完整 RGBA 逐位一致**。
六→五→六、opt-out 和换seed恢复验证通过；原有六步矩阵的28个输出也没有回归。

同档 kernel 优化的热请求用时减少约1.85%–2.53%。五步融合档相对六步原始档
的减少为文生图17.13%、图生图14.73%，但这是包含一次 NFE 减少的**不同画质档**，
不授予六步等价画质。没有相乘任何历史加速比例。

所有窗口仍检测到外部推理 CPU 负载，`qualification_passed=false`。下面是新库
串行完整请求的观察值，不是独占设备统计资格、冷启动倍率或通用性能保证。

## 固定配方与实现

四组分别为 `six-original`、`six-direct`、`five-original`、`five-direct`。相同原始
BF16组件、固定 Viggle v0.2.1 r128 strength1/inference-time、joint BF16 A/B及
F32 ranks、原encoder/VAE、seed73/74。无ANE、prefix snapshot、时间复用、
额外packed bank或其他encoder/VAE实验。不下载、裁剪或merge权重。

作者节点保持不变：六步`[1,.9375,.875,.75,.5,.25]`；五步
`[1,.875,.75,.5,.25]`。使用既有分辨率相关Qwen shift，追加0，不做base terminal
.02伸展。来源和画质取舍见[五步采样说明](qwen512-five-step-2026-10-18.md)；
原缓冲区直读、typed rounding和RoPE修复见[direct说明](qwen512-corrected-qkv-direct-2026-10-18.md)。

原有adapter固定为：

```text
Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors
679604800 bytes
SHA256 bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b
```

本轮只扩大独立 corrected-QKV 准入：实际steps5且显式开启作者五步、corrected-QKV、
direct和joint A/B时，允许相同有界512²生成/单ref512编辑。五步packed bank仍拒绝。
复用六步shape/source检查时仅修改局部准入副本，实际请求和sampler不变；原
`pinned_student_gpu_request`及block observer仍六步限定，其他pilot冲突保持拒绝。

成功覆盖改为按实际请求核对`32*(steps-1)`：五步128、六步160个completed decode
block，prefill不改。receipt新增`completed_sampler_steps`，五步direct须实际为5；
plan单独标记`qwen21_five_step_direct_qkv_composition_experimental`。这些是逻辑
执行覆盖，不是硬件kernel次数或GPU/ANE物理重叠证据。

所有入口默认关闭；五步direct的显式开关为：

```text
TURBOCIDER_QWEN21_LORA_BF16_AB=1
TURBOCIDER_QWEN21_STUDENT_FIVE_STEP=1
TURBOCIDER_QWEN21_CORRECTED_QKV=1
TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT=1
TURBOCIDER_QWEN21_PREFIX_SNAPSHOT=0
```

请求须为`steps:5`并满足其余准入条件；返回六步须把五步开关恢复为0或未设置。
研究runner在每个请求前显式设置，不依赖上一个请求残留的环境状态。

## 原始数组、恢复与质量矩阵

raw窗口18次完整请求，每任务按六原始73→六direct73→五原始73→五direct73→
五direct74→五direct73→六原始73→六direct73→六原始73运行。
同档noise/text/reference/final-latent/RGBA逐位一致；跨档原始输入相同，但最终
输出不同。换seed确实重新计算，恢复原seed和六步policy后输出exact。

两任务全部九次请求的MLX active bytes分别恒为14,914,718,962和14,931,922,162，
direct没有额外持久weight owner。bank bytes/pack time均0、bank hit为false。
这不等于所有activation零复制或同幅度的进程树/OS物理内存下降。

独立质量矩阵56次完整请求：照片、英文文字、中文文字、空间布局、编辑改色/
玻璃材质/添加物体，七任务×两seed×四组。每个采样档14对，共28对kernel比较
的最终BF16潜变量和完整RGBA全部exact。跨档14对的RGB/alpha差异另列，不作
等价门槛，也不把SSIM当作指令质量分数或透明素材资格。

contact sheet在质量窗口`trial/sampler-quality-tier-comparison.png`。六步输出与
前一封存v4矩阵的28张图及最终潜变量全部逐位一致。五步仍保留独立画质档标签，
没有全任务、多参考、身份/人脸或生产默认资格。

## 八个无dump完整请求窗口

每个固定task/arm使用独立进程，六次完整请求；首请求含load/JIT单列，后五次
conditioning-hot取中位数，seed73/74交替，同seed PNG恢复exact。文生图按
六原始→六direct→五原始→五direct；编辑反序。未并行自己的模型、构建、测试
或CPU数组审计。全部计入实际五/六步、VAE和PNG导出，不使用dump时间作分母。

| 任务 | 六原始 s | 六direct s | 五原始 s | 五direct s |
| --- | ---: | ---: | ---: | ---: |
| 文生图 | 7.5411113 | 7.3499637 | 6.3744995 | 6.2495169 |
| 单ref512编辑 | 8.8850249 | 8.7147370 | 7.7185558 | 7.5760626 |

| 任务 | 六步同档kernel减少 | 五步同档kernel减少 | 仅NFE减少，五原始/六原始 | 组合，五direct/六原始 |
| --- | ---: | ---: | ---: | ---: |
| 文生图 | 2.53474% | 1.96067% | 15.47003% | 17.12738% |
| 单ref512编辑 | 1.91657% | 1.84611% | 13.12848% | 14.73223% |

右两列不是等价工作/画质比较；所有比例由本轮完整请求直接重算，不引用旧窗口
时间。编辑两档均prefix-off，不能与以前开启immutable prefix snapshot的编辑
时间直接拼接。各臂只有一个进程窗口、五个计分项，不是置信区间。

八个首请求依次为gen六原始10.6901、六direct9.1391、五原始8.1656、五direct8.0967，
edit五direct10.1917、五原始10.2568、六direct11.2818、六原始11.3790秒。执行顺序、
系统cache/JIT不同，不从这些数字声称冷启动倍率。所有窗口负载检查失败，未停止
用户ComfyUI；CPU负载采样不是GPU利用率或每进程GPU归因。

## 测试、独立审计与来源

native probe保留40个数值case/90个非法拒绝，packed/direct各20个typed-reference
exact case及50/65个非法合同拒绝，实际RMS typed rounding witness为true。
本库C API contract138项：137通过、1个既有Wan fixture缺失跳过；真实GPU scheduler
和普通Transformer lifecycle回归通过。组合host4、既有QKV/storage host8及options
host1通过。文档检查不构成模型性能或语义质量资格。

`tools/validation/qwen_sampler_qkv_composition_audit.py`独立CPU复核complete：
650项native/build输入与链接manifest、当前tree及完整source archive一致；重算
raw/矩阵的数组和RGBA、恢复、八组无dump PNG、全部时间分母与memory digest，
并核对历史六步控制。结果为
`outputs/qwen512-five-step-qkv-composition-audit-20261018.json`。

本轮实际122次完整请求：raw18、质量56、无dump48；其中五步60、六步62，
actual denoise steps合计672。direct62次，实际completed decode block合计8896。
不是122个独立prompt，也不是硬件kernel计数。

```text
build/qwen512-five-step-qkv-direct-v5-20261018/libturbocider.dylib
SHA256 a508156f7a73d6eb9afc7e921af4db42c7b8865115058b6cab74dd9731b621cf
runtime tc-runtime-build-v1-14de1a907982f0d46f15b37cfd337b606da5c4b31e7b131e797186e0f1945deb

outputs/qwen512-five-step-qkv-composition-native-source-20261018.tar.gz
SHA256 7b691b760e7c7fdc0ebbfa2a1a164e3045ae53996cf168eea05d87595d50f0f6
```

以后native tree变化时按归档恢复到新目录，不用新源码冒充该二进制的来源。

## 内存、原件与后续

十个请求窗口的进程树memory采样全部complete，system swap-in/out均0。
raw/质量峰值physical footprint分别31,781,903,136/31,707,142,920 bytes，系统
compression分别约2.90/3.20GB，不归因到单进程。八个无dump峰值约18.84–19.15GB，
compression均0；不是DRAM带宽或GPU独占证据。

原件均在`outputs/qwen512-five-step-qkv-composition-*-20261018/`：`raw`、`quality`、
`gen-{six-original,six-direct,five-original,five-direct}`及对应四个`edit-*`目录。
native数值日志为`outputs/qwen512-five-step-qkv-direct-kernel-20261018.log`。
所有历史库、原图、权重和用户改动保留；未下载模型或清理用户缓存，结束可用磁盘
约12GiB。owned模型/构建/测试/审计任务均已退出。

后续继续重复完整请求窗口、寻找主要DiT成本的净收益。若组合immutable prefix或
其他GPU/ANE方案，须另立匹配的四组控制和lifetime/质量验证，不据本轮结果直接
扩大准入或改变默认值。
