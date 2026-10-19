# Qwen512 corrected-QKV：局部 RoPE 收缩修复，七类整图逐位一致

2026-10-18，Asia/Singapore，M4 Max / 64GiB。仅 Qwen Image 2.1 的 512² 文生图和
单 ref512 图生图；不处理 Z Image，不部署 App，不改默认值，完整优化目标仍 active。

## 已确认的变化

[上一版整请求](qwen512-corrected-qkv-request-2026-10-18.md)虽观察到约 1.8–2.2%
热态收益，但生成 final-latent relative L2≈.0177、alpha max difference1，未通过
事前门槛。本轮没有放宽门槛，而是做实际输入的分阶段数值反事实。

在 layer0 的同一个真实 decode 输入上，base、加 B 后和 weighted RMSNorm 后均
逐位一致，差异在 RoPE 后出现。只在 RoPE 算式的局部 scope 加
`#pragma clang fp contract(off)`，同输入 Q/K/V 和完整 block 恢复逐位一致。
base/B 的计算、RMSNorm、dtype、LoRA factors 和 sampler 都没改。

迁入默认关闭的新库后，两个 canonical 任务的完整六步/VAE/PNG 全部恢复一致；
进一步七类任务 × 两个 seed 的 14 对 final latents 与 RGBA 也全部逐位相同。
这是相对原 joint BF16 A/B 路径的样本等价证据，不是对任意 prompt 或原 F32
LoRA recipe 的全称证明，也不是 instruction-adherence 质量评分。

## 实际输入的分阶段反事实

固定原 BF16 组件、Viggle v0.2.1 r128、strength1/inference-time、joint BF16 A/B /
F32 A-ranks，512² / 六步 / seed73。observer 借用首个完整 DiT step 后 step1/layer0
的原 GPU callable、9 个实际参数及已完成参考输出。这里只做一个真实 block，受控
停止在 VAE/PNG 之前，不能与后面的整请求混为一谈。

各阶段完整比较三个 `[1,32,1024,128]` BF16 输出，共 12,582,912 元素。保留原
base 和 supplied-native-base 两类 producer，以及 B 的 BM16 / BM32 两种 tile：

| 同输入阶段 | 生成 changed（Q/K/V） | 编辑 changed（Q/K/V） |
| --- | --- | --- |
| base | 0 / 0 / 0 | 0 / 0 / 0 |
| base + 在线 B | 0 / 0 / 0 | 0 / 0 / 0 |
| weighted RMSNorm | 0 / 0 / 0 | 0 / 0 / 0 |
| 原融合 RoPE | 41 / 53 / 0 | 45 / 45 / 0 |
| 局部禁止 RoPE contraction | 0 / 0 / 0 | 0 / 0 / 0 |

四类 producer/tile 都复现同一结果。matched-input norm 的差异为 0；matched-input
RoPE 仍复现 94 / 90 个差异，替换为原生 base 并不能消除它们。原差异的最大绝对值
分别 .0078125 / .015625。probe 的原完整表达式还以实际 v2 kernel 的输出 hash 作
见证，确认分解未先改掉被调查的计算。不得把该 layer0 事实宣称所有层都只可能有
这一个误差源；随后完整图片实验单独约束整条链。

`outputs/qwen512-qkv-numeric-audit-20261018.json` 已独立 CPU 复核四个窗口的所有
完整 stage/consumer 数组、22 项实际 layer0 base/factor 来源、一致的 normalized
input、source snapshots、memory digest 与完整 block 分母。记录 bitwise changed，
不靠只看 cosine 或隐式规范化负零判 exact。修正臂完整 block 的 4,194,304 元素也
全部 exact。原始数组与版本各自保留，没有重写上一版失败样本。

这些隔离窗口的 GPU-replay / 原融合臂出现额外耗时和较大波动；不把其约 44–47ms
当弱分母宣布巨额提速。后续性能只用独立无 dump 的原普通 GPU 整请求对照。

## Native 修复与边界

修复位于 `native/models/qwen21/metal/lora_qkv_prepare.hpp`，pragma 只包围 RoPE
两项 F32 乘积与 Add/Subtract，不全局修改 GEMM/B/norm 的编译数学策略。weighted
BF16 RMSNorm 的 normalized-value→BF16→gamma→BF16 边界仍保持此前修复。
新 kernel/receipt recipe 为 `decode-only-corrected-qkv-v2-rope-rounding`。

开关仍为 `TURBOCIDER_QWEN21_CORRECTED_QKV=1`，默认关闭。仍只运行后五步全部
32 层，首步及 prefix K/V 保持原普通 GPU；原 227 个 adapter terms 完整绑定，
各投影独立在线 A/B，不 merge、不少步、不跳层、不裁 token、不引入 ANE。
成功请求在全部六步 latents 完成后必须实际覆盖 160 个 decode block。

原 3GiB base bank 与约 .15s 首次 pack 成本仍存在，不称免费。opt-out、adapter
变化、异常和 unload 清理保持。planner 与 runtime 现在共享同一准入 predicate，
显式 plan label 标为 experimental；不因此扩大 public 默认能力。

## 整请求质量

canonical raw：每任务 off73→on73→on74→on73→off73，十张完整图片。初始噪声、
text 和 reference latents 的同 seed 对照 exact，换种子确实重新计算，回到同
policy/seed 后恢复 exact。生成和编辑的 final-latent relative L2 都为 0，RGB
SSIM1、RGB/alpha max difference0。仍使用原门槛 L2≤.01、cosine≥.999、SSIM≥.99、
alpha exact，没有后验调宽。

独立 broader matrix 执行 28 次完整请求，14 对对照：照片、英文文字、中文文字、
空间布局、编辑颜色、编辑材质、编辑添加物体，每项 seed73/74。两种 policy 同 seed
的 final latents 与完整 RGBA **14/14 全部逐位一致**；noise/text/reference 对照
也全部 exact。数据位于 `outputs/qwen512-corrected-qkv-rope-matrix-20261018/`。
不把相对 baseline 一致解释为原模型本身的文字/编辑遵循质量已获认证。

## 性能与审计状态

新的无 dump 四窗口全部完成。每个 task/arm 独立进程执行六次完整请求，首请求
含 load/setup 单列，其后五次 conditioning-hot 取中位数；seed73/74 交替，检查
回到同 seed 的完整 PNG exact。生成 off/on、编辑 on/off 顺序，进程不并行。
时间包括六步、VAE 与 PNG 导出，不以 raw/matrix 带 dump 请求作性能分母：

| 任务 | 原普通 GPU 中位数 s | 修正融合中位数 s | 减少 |
| --- | ---: | ---: | ---: |
| 文生图 | 7.5277926 | 7.3511184 | .1766742s / 2.34696% |
| 图生图 | 8.8983785 | 8.6938580 | .2045205s / 2.29840% |

各 arm 首请求分别 gen off/on=10.6552/9.2367s、edit on/off=11.7431/11.8157s。
OS/JIT/file cache 已受前序工作影响，不从这些 cold-process 数字宣称冷启动收益。
两个 on 窗口首次 pack+原 BF16 位比较约 .1525/.1522s，随后 pack 实际为 0。

两个 task 各只有一对固定-arm 窗口，所有窗口仍检测到外部推理 CPU 负载，
`qualification_passed=false`。上述是新的整请求观察，不沿用上一版速度，也不宣称
独占设备的统计性能资格或全 prompt 加速保证。默认与安装 App 均未改变。

已完成：20 个 native kernel case 对原 typed arithmetic 逐位一致，50 个 native
非法合同拒绝；既有40数值/90拒绝保留；136 项 C API contract 中135通过、1项缺
fixture跳过；普通 Transformer lifecycle 回归通过；full-request host gate 六项、
stage counterfactual host gate 四项通过。

本轮独立库为 `build/qwen512-corrected-qkv-rope-v3-20261018/libturbocider.dylib`。
没有覆盖历史 stable/v2 library、历史探针、模型权重或用户缓存。

```text
SHA256 fda5b1ce5d3c2846da67f97b6aeda9674264bf3489645dbd9bcf43788dbfc8e8
runtime tc-runtime-build-v1-effbd6c11b3ec20a6a5289fed1cf58ad627e185775ab731e8c6e692202f126e2
```

`outputs/qwen512-corrected-qkv-rope-request-audit-20261018.json` 已独立复核 complete：
650 项 native/build inputs 与 manifest、当前 tree、完整 source archive 一致；
重算 canonical 与 broader matrix 的全部原始数组、L2/cosine/SSIM/alpha；重读所有
无 dump PNG，确认同 task/policy/seed73 与 canonical raw 图片逐位一致，且矩阵
中的同任务也一致；复算完整计时分母、memory digest 和 3GiB bank owner 生命周期。
旧版质量失败数据未重写，新版全部原门槛通过，但不由此授予 semantic/physical
或独占性能 qualification。

本轮共62次完整请求（raw10、matrix28、无dump24），其中32次候选，累计5120个
completed decode block；不是62个独立 prompt。MLX active owner 的 on/off 精确
差值仍为3,221,225,472 bytes，返回 off 后回到原值，不能当 OS 物理退回证明。

source archive 为 `outputs/qwen512-corrected-qkv-rope-native-source-20261018.tar.gz`，
SHA256 `dcfdf71b54d76be80e208567f6c2230de458e63f52daede1707b92416fc0c4b3`。
未来 native tree 变化后使用该归档恢复到新目录，不把新源码冒充此库源码。

## 内存、原件与后续边界

六个整请求窗口的进程树采样全部 complete，system swap-in/out 均0。raw/matrix
峰值分别31,769,090,848 / 31,704,783,840 bytes，系统 compression 增量约3.39 /4.03GB，
不归因于某一个进程。无dump gen off/on、edit on/off 的峰值分别18,840,071,424 /
22,182,554,832 /22,163,844,400 /18,900,823,272 bytes，compression均0。这不是
每张图、GPU专属或驱动全归属统计；原3GiB bank的驻留代价没有消失。

主要原件：

- `outputs/qwen512-qkv-numeric-{gen,edit}[-rope]-20261018/`：四组实际 stage 与 block 数组。
- `outputs/qwen512-qkv-numeric-audit-20261018.json`：隔离阶段 CPU 审计。
- `outputs/qwen512-corrected-qkv-rope-raw-20261018/`：十张完整原始对照。
- `outputs/qwen512-corrected-qkv-rope-matrix-20261018/`：七任务/两seed、28张完整图及数组。
- `outputs/qwen512-corrected-qkv-rope-{gen,edit}-{off,on}-20261018/`：四个无dump窗口。
- `outputs/qwen512-corrected-qkv-rope-kernel-20261018.log`：native参考逐位测试。

结束磁盘约12GiB可用，无下载、无用户文件清理，owned build/test/model/audit均已退出。
下一步应重复完整性能窗口，评估减少3GiB bank的方案，并把保留 typed rounding /
RoPE contraction 的方法验证到原普通LoRA的更轻量融合路径；不凭这里的样本把所有
精度、分辨率、参考数量或硬件全局改为同一默认。

后续已验证 [原Q/K/V buffer直读路径](qwen512-corrected-qkv-direct-2026-10-18.md)：
不再建立该3GiB权重副本，canonical和七任务/两seed对照仍逐位一致；其速度由新的
无dump整请求窗口单独判断，不从本页packed版倍率推导。
