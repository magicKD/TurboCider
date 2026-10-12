# ConvRot 数值 BF16 舍入边界：诊断实现与负验收

本地日期2026-10-06（Asia/Singapore）。接续寄存器核/有界解码实验，
完整目标仍是双后端、Z/Qwen两分辨率base≥1.2×、LoRA、正式质量/
内存/device trace、拟合与自动调度集成；下面不替代这些条件。

## 已定位的两个算术边界

原文本MIL `bf16` / `bfloat16` cast仍被当前编译器拒绝，FP16 cast正控制
256个样本零错误。原divide-based模拟有14,977个错误。显式二进制幂
分段乘法移除小量倒数，bounded-coefficient版本可编译，但直接MIL
`round`版本仍有4,097个错误，分布符合halfway tie问题，含一个零符号
差异。不是在未验证的log2估计上增加grouping开关。

最终用floor、精确fraction与整数奇偶判定实现ties-to-even；
`2*floor(lower/2)`的mul形式又产生8,191个错误，等价的
`floor(lower/2)+floor(lower/2)`形式通过所有非零数值对照。这里仅
确认具体整图/系统tuple的lowering结果，不推断所有mul或round都有错。

共享native emitter的实际16次driver调用覆盖63,480个finite-half
encoding；8个BF16-rounded值超出FP16 carrier范围的encoding明确排除。
独立CPU BF16-RNE→FP16 oracle：numeric mismatches=0，raw-bit
mismatches=1，唯一差异为`-0`→`+0`。原严格bit gate仍返回失败，未把
它修改为通过；本实验只使用另有明确标识的数值/canonical-zero政策。
不称原生BF16 cast、FP32 ANE arithmetic或原生INT8 MAC。

证据：`outputs/ane-bf16-*-20261006.log`。v2的compile rejection、v3/v4
负结果以及v5/v6的零符号差异均保留。

## 显式实验配方

```sh
TURBOCIDER_ANE_BACKEND=private
TURBOCIDER_ALLOW_PRIVATE_ANE=1
TURBOCIDER_PRIVATE_ANE_DATA_PATH=convrot_w8a8
TURBOCIDER_PRIVATE_ANE_CONVROT_BF16_BOUNDARIES=1
```

当前实现为gate、up、SiLU、hidden、rotated-hidden五个数值舍入边界；
四边界binary保留作hidden-rotation独立对照。实模型表使用原v1诊断
recipe；随后增加carrier安全检查，v2为
`comfy-h256-direct-q8-source-round-row-a8-rne-bf16-value-guarded-zeros-canonical-v2`。
实际MIL/constants hash及library hash另行隔离图与版本。

默认关闭。只接base-only、Comfy、row A8、原BF16 activation、headroom=1；
禁止group256/LoRA/Sylvester混用和非法配置静默忽略。新的executor
configuration identity包含该开关。旧默认MIL/原LoRA hidden ABI、master
codes/scales、两套W banks、两套可选A8 slots、producer drain与shared
events不改变。新工作区在memory estimate额外计入16个FP16 slice及
64MiB allowance；这不是物理ANE scratch观测，正式process-footprint
资格仍未完成。

FP16 carrier overflow不裁剪，也不换headroom来改变此精度配方；失败
返回原model wrapper，整个operation由GPU重算，不发布部分scratch。
这是精度定位工具，不是推荐的默认加速配置。
增加overflow fixture时发现v1会把部分非有限中间值量化成有限codes，
最终输出可能被误报成功。v2在五个边界量化前以`abs(value)<65408`
检测carrier是否安全，将每个token的失败汇总到原hscale行的负哨兵；
GPU transfer对非正token scale的原validator拒绝它，headroom=4重试
被此recipe明确阻止。raw/packed×serial/lookahead四格overflow/failure/
healthy refill通过。没有增加第二个output signal或发布部分结果。

## 实模型对照

原ConvRot checkpoint、legacy packed BF16 scales、同prompt/seed42，
四步、Fa4096/Fg6144、1056-row bucket、FP32 partial join。每个GPU与
candidate pair使用相同binary，独立CPU FP64比较trajectory。512的
conditioning/initial latent精确相同，各有128个channel blocks、248次
driver calls，零fallback/overflow retry/runtime failure。

| 尺寸 / boundary | final rel L2 | cosine | 原N1 |
| --- | ---: | ---: | --- |
| 512 / gate-up-SiLU-hidden | 0.0383927793 | 0.9992645647 | fail |
| 512 / 再增加rotated-hidden | 0.0672675977 | 0.9977396648 | fail |
| 1024 / 五边界 | 0.0564142207 | 0.9984097244 | fail |

原门槛rel L2≤0.03、cosine≥0.999不变。四边界比旧row+FP32的历史
0.04336734有所改善，但没通过；五边界反而变差。不能从63,480个
单点通过推断FP16 reduction/A8/Comfy rotation或完整trajectory通过，
不能选对自己有利的某个scope提升默认。native quality validation
disabled/calls=0时的passed字段仍不作为验收。

诊断时间也明显不合加速目标：四/五边界512 cold wall约271.9/336.8s，
实际load分别约157.4/196.0s；Private等待累计约107.6/133.6s。
这些包含cold编译、dump和观察器，部分窗口同时有build/组件检查，
不计算或宣称正式倍率。它们已足以否定把此大型软件舍入图当默认
性能优化的推荐；还未证明任何物理GPU/ANE overlap。
1024复用已编译五边界program，load约0.0556s，diagnostic wall约291.6s，
512次driver calls、128个channel blocks、零fallback/retry/runtime failure。
conditioning/initial latent仍精确一致，Private等待累计约276.7s；即便
消去cold compile，也未展现可推荐的加速。不是正式hot E2E测速。

目录：`outputs/convrot-bf16-value{4,5}-512-{gpu4,private4096-4}-diagnostic`，
对应`*-quality-20261006.json`与raw events/receipts保留。
1024目录为`outputs/convrot-bf16-value5-1024-{gpu4,private4096-4}-diagnostic`。

## 构建/组件

Private suite19项：17 pass、2项显式MLX/channel/calibration opt-in skip。
新配方实driver raw/packed×serial/lookahead四格、three-row-chunk、
physical channel stride、负/零down scale、F32 restore、future identity、
late activation/W producer failure与refill通过，base-only不冒称LoRA测试。
旧LoRA路径仍由原suite验证。Public Core ML/MLX/receipt13＋host8通过；
actual public release-binary guard通过。不是完整`make test`绿色。

四边界Private SHA256：
`e22754ea1bea3d5facfd7785f6145ca7cb155e9faac2108cadb7af6b95ffd211`。
五边界Private SHA256：
`3297f21c272ac6442a61f408e8f5ff419195577cc0d76532e924ca66164ff5bc`。
Public SHA256：
`d9c430cd3e4088b7368807c4d1abd19331dd73e1e77f8d444bae189a18dfdd7d`。
上列v1两库在当时输入source hash无不匹配，随后carrier guard修改
不由这些旧binary验证。它们仍是保留既有
ConvRot drafts的working-tree snapshot，不是clean staged-tree rebuild；
本轮selective commit不混入那些旧改动。

下一步应比较GPU接手敏感中间激活/尺度恢复与ANE projection的完整
handoff成本、按operation/layer选择Public/Private/GPU，而不是继续
盲加ANE舍入节点。shared-event、原hidden/down-LoRA一次性合并与
whole-operation fallback必须保持。实际bandwidth fit/cache/share/
prefetch、GGUF有界consumer调度、Z/Qwen四格与实LoRA完整验收仍未完成。

## Carrier guard接续

v2新Private library SHA256：
`b6944e101b64887be16324d81cdf104e0462b77373283aec14c6be1bc9e04ca2`。
v2新Public library SHA256：
`8dba5633e7f0c4a652d83d6c7faf23ac381509b7a53ee9bfe82ec3a9245a4f9c`。
这两库的source manifest对当前tree无不匹配，Public actual release
guard通过。Private suite19项仍为17 pass / 2显式MLX集成skip，新增
65504 carrier overflow fixture在四格组合中验证failure、不改headroom、
健康refill。未删除旧的`ok=1`错误日志，也不把v1模型表填成v2实测。
v2仍默认关闭，正式模型质量/性能/内存资格尚未完成。
