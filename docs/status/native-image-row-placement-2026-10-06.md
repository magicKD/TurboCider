# 图像行切分与 caption GPU 保护：精度改善仍未通过最终门槛

2026-10-06，Asia/Singapore，M4 Max 64GB。接续
[真实FFN回放和compact值舍入](native-ffn-capture-compact-values-2026-10-06.md)。
完整目标仍active：双后端、GGUF/Q4/Hadamard/双缓冲/shared events、
Z/Qwen四格base≥1.2×、真实LoRA加速、正式质量/内存/device trace没有
全部完成。本轮不提升默认、不缩小目标、不改变原N1阈值。

## 共享实现，不靠输入/输出整体轮转

新增portable `RowPolicy`/`RowWindow`，由family明确指定placement和
保留GPU的尾部行数；共享FFN层负责plan、合法chunk上限、精确ANEx
view offset、GPU complement、LoRA correction行范围、原行序join和
失败时整个ANE span GPU重算。两种backend共用，不嵌入模型weights。

| policy | 1056行 / ANE352 / caption32 的实际span | GPU complement |
| --- | --- | --- |
| suffix，旧默认 | [704,1056)，包含caption | [0,704) |
| image_prefix | [0,352)，仅图像 | [352,1056) |
| image_tail | [672,1024)，仅图像 | [0,672)+[1024,1056) |

prefix仅改view offset，没有新增GPU input pack。interior的两片GPU
输入只concat一次，调用原完整GPU FFN callback一次，之后按原行序
拼回。无第二份W、无精度cast、无CPU ndarray/整体row rotation。
新的complement pack在提交ANE前做owner-thread memory admission，
计入local budget/retained host scratch及system headroom；未获admission
完整GPU回算，不publish scratch。receipt给出logical peak pack bytes，
不把它当physical RAM上限。已有producer/consumer ownership保持。

模型入口默认`suffix`，experiment build才接受：

```sh
TURBOCIDER_Z_RUNTIME_ANE_ROWS=image_prefix  # 或 image_tail
TURBOCIDER_RUNTIME_ANE_CHUNKS=1
```

当前family gate只接explicit dense base resident FP16 runtime、合法
Public或明确授权Private、无LoRA/streaming/channel split；模板bucket
须放得进image-only rows。noise refiners没有caption，保持旧suffix；
30个main blocks保护actual padded caption rows。默认Public、旧W8
channels、GGUF/ConvRot、frozen与模型LoRA入口不静默选此实验。
request identity绑定placement。generic row bridge的synthetic LoRA
prefix/interior已实际测试，但不冒充整模型LoRA加速资格。

actual telemetry包括policy和成功suffix/prefix/interior blocks、protected
rows及pack逻辑bytes；quality工具显式`--row-placement`检查实际
counters/axis/data path/model/base和calls，不能仅根据env或label认定
caption保护执行。legacy未带字段的suffix receipts保持兼容。

## 回归发现与修复

首次chunk上限约束把原合法零-chunk `SplitProbe`误转成普通GPU，
原`ane_ffn_test`断言失败；v2 integration的失败保留，没有弱化测试。
修复为仅在原positive chunks被eligible约束降成0时才转GPU，保留
原零-chunk timing/bridge诊断语义。新增prefix zero-chunk probe与全
protected无合法chunk检查，最终v3 integration通过。

实际CoreML/MLX测试使用非恒定row IDs，验证三种placement、GPU/
ANE/LoRA callback精确输入、原顺序输出、GPU保护尾行逐位一致、
base/LoRA、typed lazy lifetime、非有限correction后的整个ANE span
重算，以及成功counter不含失败launch。prefix/image-tail不会静默
套到channel split，mixed nondefault policies不能在同runtime session
中切换；noise suffix可与main保护共存。

## 真实生成：固定同offload量，仅交换placement

fox/seed42、8步、512/1024，resident，原dense BF16 checkpoint；runtime
Private FP16、GPU IO、352-row/K1024/N512原activation-input模板、
chunks1/fixed-async1。compact value-rounding关闭，不改变private/public
FP16 graph算术。GPU/candidate独立进程串行，未与owned build/test重叠。

初始v1同库八次GPU/suffix/prefix/image-tail生成，最终v3又独立生成
GPU/suffix/image-tail、以及两个gpu-refiners消融，共十次。v3 suffix
和image-tail与v1对应完整输出一致；ordinary GPU与前轮同形状GPU
输出一致。所有conditioning/initial latent、全部8步、final/PNG、
observed execution receipt和quality报告保留；没有复用不同库GPU
分母。v1未计入complement pack预算，已由v3补全。

| 尺寸 / recipe | final relL2 | cosine | 原N1 |
| --- | ---: | ---: | --- |
| 512 suffix | 0.0934619628 | 0.9956320612 | fail |
| 512 image_prefix（v1） | 0.0882875706 | 0.9961028597 | fail |
| 512 image_tail | 0.0707613430 | 0.9974984977 | fail |
| 1024 suffix | 0.0779761958 | 0.9969570683 | fail |
| 1024 image_prefix（v1） | 0.0687401282 | 0.9976346234 | fail |
| 1024 image_tail | 0.0560220825 | 0.9984301150 | fail |
| 512 image_tail + GPU refiners0,1 | 0.0975859794 | 0.9952433681 | fail |
| 512 image_prefix + GPU refiners0,1 | 0.0996122656 | 0.9950395785 | fail |
| 1024 image_tail + GPU refiners0,1 | 0.0675481116 | 0.9977177998 | fail |
| 1024 image_prefix + GPU refiners0,1 | 0.0544088062 | 0.9985189429 | fail |

原门槛仍relL2≤0.03、cosine≥0.999，全部`qualification_passed=false`。
caption保护能降低该prompt上的误差，但不充分；保留两个refiners为
GPU也未稳定改善，不能据此认定唯一根因或选择有利cell作为验收。

不强制GPU refiners时，每arm actual calls259、retries3、headroom64、
zero runtime failure/fallback、ANE rows90112。suffix成功blocks256；
保护方案是noise suffix16+main prefix/interior240，protected rows7680。
image_tail pack logical peak：512=5406720、1024=28999680 bytes；prefix
和suffix为0新增pack。GPU refiners0,1消融actual calls243、retries3、
ANE rows84480、main保护blocks240、forced GPU blocks16，零failure/
fallback；不能把其较少offload当相同share的placement对照。

证据目录：`outputs/native-z-row-placement-v1-z{512,1024}-base-20261006/`
和`outputs/native-z-row-placement-v3-z{512,1024}-base-20261006/`。
带dump/观察器、无cold+hot/reverse/multiprompt/strict load/memory资格，
不计算正式倍率，也不宣称达到≥1.2×。

## 构建、发行隔离与回归

最终Private v3 native-only、Public v3 ordinary library-only成功，各480
个source inputs与current native tree独立匹配，包含原用户dirty草稿，
不是staged-only制品；提交仍只选本轮owned hunks。
Private SHA256 `7238ffdb2235cda7857046526e51e371767d2e9710ed4c85b921e2d778f6aa03`，
ID `tc-runtime-build-v1-1f2ad634e54add60aa05480801bca49d7c24ca28630ec094330d8305b643f57b`。
Public SHA256 `9badfb08e00b38990b6f95148c1752909f6fbb2b6bfb532282686d167da3ca65`，
ID `tc-runtime-build-v1-6929013fbdfaa025720236e29611044d42137416357140d08e17191d669ac862`。
ordinary release在load weights前拒绝image-only placement，suffix可
正常创建；四种Private client/request/event/surface strings未见于Public
library。未改变发行默认或允许私有API偷偷进入Public。

124项host/tool/capture/source guards、16项Public graph/CoreML/MLX/
receipts/ordinary gates（含新actual row placement coverage）、2项实际
Private channel/MLX calibration/LoRA/ONE down/F32 partial/late failure
回归通过，共142项；不是全仓/Swift/所有设备或最终四格资格。
完整目标保持active，接下来仍需定位剩余迭代误差并优化可通过的
W8A8/FP16混合recipe，再完成同预算、matched optimized GPU的正式
加速/LoRA/媒体/物理trace审计，不能以当前改善替代要求。
