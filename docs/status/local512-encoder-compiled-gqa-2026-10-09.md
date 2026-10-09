# Qwen encoder：有界图复用与原 dtype GQA

2026-10-09，Asia/Singapore，M4 Max64GB/macOS26.6.2。接续
[FFN launch-order 筛选](local512-channel-launch-order-2026-10-09.md)。
原目标仍 active：Z/Qwen base/原 LoRA、生成、1–2参考图编辑、DiT/
encoder、GGUF/ConvRot 与真正整请求加速。无模型下载、原 source/adapter/
reference改写。使用本地原Qwen Image2.1/Qwen3-VL/Viggle与已有小模板。

## 先排除 Qwen BF16 partial-down

原layer0 BF16 down physical pitch12288、合成BF16 hidden，compiled
projection/widen/eval；每recipe3warmup/15cyclic hot，无额外dense W bank。
对照MPP F32、MPP BF16+widen、原MLX BF16 matmul+widen，18geometries。

| M/K7168 | 原MPP F32 ms | MPP BF16+widen ms | MLX BF16+widen ms |
| --- | ---: | ---: | ---: |
| 302 | 1.696291 | 1.677958 | 1.592291 |
| 2096 | 8.400875 | 8.503333 | 8.637416 |
| 3144 | 12.534666 | 12.665875 | 12.852667 |

额外BF16舍入relL2约.167%，但首步geometries不盈利，不wire入模型。
原始：`outputs/local512-qwen-bf16-partial-v1-component-20261009.json`。
独立probe链接当时HEAD92466cb的已保留库及其匹配C++ header；随后
encoder实现改变后不重新用旧库编译当前TextConfig。不是实FFN/模型验收。

## Encoder 实现与源/依赖边界

新增默认关闭的 `TURBOCIDER_QWEN21_ENCODER_COMPILED_GPU=1`，要求
approximate unconstrained resident512、无prompt enhancement；支持生成
或1–2ref512编辑，DiT与其原inference-time LoRA独立，不改其默认执行。
输入/causal/padded-key mask、FP32 norm、mRoPE、GQA与DeepStack早期
层注入顺序不裁剪。four-layer/final eval、取消与完整GPU恢复边界保留。

GPU route编译完整语言block；hybrid route独立编译GPU attention段与
GPU FFN complement/full fallback。所有原norm/projection/bias/metadata
arrays是动态参数，不是首次trace捕获的常量W。source不重排/widen，
不将adapter漏算或merge进encoder；显式compiled encoder拒绝encoder
本身的runtime LoRA（本目标的adapter仍在DiT）。

第一版每层新建函数；第二版把layer名规范为`block`，同一个encode内
用结构keys复用body。缓存严格request-local，layers上限128，不留全局
graph/source owner，不制造新权重cache。uniform原模型实际block body1；
channel模式full-fallback/partial两个FFN body，只有调用的body算invocations。
declared body计数不当作GPU kernel/physical overlap或内部JIT trace证明。

第三版opt-in在attention直接使用原Q/K/V dtype与原32/8 heads，免除
pre-repeat和F32 Q/K/V临时扩展；0/-Inf mask转原dtype仍exact。
MLX SDPA保持F32 softmax，不改FP32 norm或mRoPE。原默认仍为repeat
K/V加F32 attention。此approximation必须明确披露，不因一个synthetic
case exact就称所有encoder/图片逐位一致。

参考官方[MLX SDPA说明](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)
中的native GQA与softmax dtype；只读Splash `feat/ane`
`be83e8f895bc8d9036ace8348c7ddc2098dc5ebc` 的
`runtime/metal/CommandGraph.hpp`，复核提交参数/源owner生命周期。
不是移植其LLM收益，也没有checkout/fetch或修改参考工程。

选择在request snapshot，编译/numerical recipe进入encoder identity；
切换时generation/edit conditioning两cache一起失效，防止旧条件跨
recipe或retained executor复用。code profile检查evaluated finite hidden，
stderr记录真实whole/attention/channel/full-FFN graph invocations；
cache hit不冒充新的encoder执行。不是device trace。

## Attention 独立筛选

BF16/FP16、heads32/kv8/d128、rows33/318/574/1024，合成post-norm Q/K/V，
因果及padding mask，完整compiled repeat/cast/attention/eval，各
3warmup/15cyclic hot。数据不含原模型/权重/完整encoder。

| BF16 rows | F32+repeat ms | F32 GQA ms | 原dtype+repeat ms | 原dtype GQA ms |
| --- | ---: | ---: | ---: | ---: |
| 318 | .771250 | .704834 | .540084 | .455500 |
| 574 | 1.036792 | .957458 | .720417 | .673834 |

这些synthetic BF16 cases relL2=0；不外推成真实conditioning/生成图片
exact。原始：`outputs/local512-encoder-attention-v1-component-20261009.json`。

## v1/v2：完整请求否定单纯编译推广

原encoder/vision source与prepare输入，单ref318语言rows、Fa3072/c320，
source/executor retained，四recipes5cyclic hot，仅language component。
v1：GPU eager .353293s、compiled .347263s；hybrid eager .336444s、
compiled .326357s。conditioning relL2分别约2.635%/3.330%。
v2同窗口各自control：.351192/.342088s、.336443/.325415s。
compiled首个timed call .343502s；v1首个timed call .617800s。
跨版本cold差不是同窗口causal实验；GPU oracle已先跑一次，first不是
whole-process cold start。两版各432实ANE calls，不能代替整请求。

原Viggle v0.2.1/r256/strength1、512²六步、seed29，DiT完整GPU/joint
BF16 A/B/F32 ranks；same source retention，encoder两hybrid retained。
每臂独立process、一冷两不同prompt热请求，三fresh conditioning miss。

| version / refs | GPU eager request s | GPU compiled s | encoder eager s | encoder compiled s |
| --- | ---: | ---: | ---: | ---: |
| v1 / 1 | 10.479168 | 10.623917 | 10.720926 | 10.654455 |
| v1 / 2 | 12.668630 | 12.698956 | 12.906614 | 12.909382 |
| v2 / 1 | 10.523652 | 10.517639 | 10.700978 | 10.693347 |
| v2 / 2 | 12.734747 | 12.693246 | 12.934217 | 12.949405 |

单图正序/双图反序不是同workload ABBA。全部完整屏均保留cold/两warm、
100ms process-tree memory及enclosing CPU-load观察；busy诊断不qualified。
v1有局部compiler热收益，whole request仍无可靠赢家；v2cold overhead
改善也不能变成稳定显著整图收益。两版数据/库身份不混用，不默认启用。
v3真实单图component：同窗口GPU eager/compiled .351161/.343814s，
hybrid .335671/.329562s；conditioning relL2仍为2.635%/3.330%。
单个attention kernel较快，没有相应放大为整段language收益，不将
synthetic局部倍率当原模型encoder保证。

v3完整request，GPU eager/compiled与encoder hybrid eager/compiled
分别匹配；DiT始终完整GPU。base无LoRA/参考图、40步；编辑原LoRA/
六步，1–2参考图；seed29、三fresh prompts、一冷两热，各臂独立process。

| workload | GPU eager s | GPU compiled s | encoder eager s | encoder compiled s |
| --- | ---: | ---: | ---: | ---: |
| LoRA / 1ref | 10.490213 | 10.513979 | 10.644622 | 10.649441 |
| LoRA / 2refs | 12.678365 | 12.669016 | 12.937661 | 12.880790 |
| base / generation40steps | 43.369143 | 43.373726 | 43.611365 | 43.551624 |

v3仍没有稳定显著整请求盈利。组件/attention收益和warm中的微小差额
不能推广为默认，保留default-off研究配置及完整GPU fallback，最快
encoder route仍优先GPU。所有cold/两warm都保留，不能只挑最后一个
text样本，也不能跨版本借较快GPU分母。三份enclosing load checks均
不qualified；continuous CPU观察只是heuristic，不证明GPU独占。

已查看单/双图LoRA case1 GPU eager、GPU compiled及hybrid compiled
whole PNG；base generation case1 GPU/hybrid compiled全图。壶形、
把手/壶嘴、盖/钮、双壶布局、颜色、暖光/阴影非常接近，釉面/纹理/
高光略有变化，未见明显新增块纹或断裂。只覆盖明确这些cases，不
冒称所有细节/seed或用户批准。GPU eager三张单图PNG与92466cb同
workload GPU controls逐字节相同，default helper重构未改这些默认图。

每个新版compiled请求的language blocks/attention segments均36，
canonical declared block body1；GPU route full blocks36、FFN bodies0，
Private channel route attention/channel36、FFN bodies2、full-FFN invocations0。
后者两个body之一是备用full-GPU callback，declared不等于已执行。
native-GQA bool、finite evaluated hidden与sources/retention/calls共同
验证，不根据env/label凭空算执行。对照eager不发布compiled records。

Private15项/Public14项selected native回归通过，无skip；额外两库各一次
更新后的actual Public bridge恢复用例通过：原BF16 post-norm仍finite，
FP16 ANE转换overflow，完整compiled GPU FFN重算对原GPU在5%预算内
且finite；不是常量sentinel。原common Private late-chunk恢复也重新通过。
48个compiled语言fixture覆盖BF16/FP16/F32、causal/padding、GQA/mRoPE、
bias/早期DeepStack/final norm、原source重绑定、取消/nonfinite与immutable
master。Public CoreML shared encoder16cases亦通过。

额外1项actual Private C API prepare-only session test通过：same executor/
source，eager→compiled→eager各自miss后hit，generation/edit两namespace
同步失效；manifest切换/invalid/recovery、源/执行器保留off/unload仍通过。
这不是denoise/媒体/性能资格。27项host contracts通过；最初新增host
test漏import json的v1失败log与之后修复v2/v3/v4结果分别保留，不覆盖。

最新两份native-only build exit0，各499source inputs重新核对无mismatch；
仍含原用户ConvRot working-tree草稿，不是clean staged-only/App发行。
Public实际release二进制private/test/audit隔离guard通过，非Public实模型
性能资格。12份v3memory报告complete、swap-in/out0，process-tree
phys-footprint peak约36.81–41.15GB（十进制），不归因全系统/driver。

所有本轮owned jobs终止后，清理v1/v2/v3Private与v3Public四个isolated
build的865个可重建`.o`，67,710,792 logical bytes（约64.6MiB）及四个
空module-cache。保留所有库、CLI、probes、logs、PNGs、raw observations/
manifests与原始模型/adapter/reference；无用户cache删除或外部process
signals。旧库只与其当时的header/source snapshot配对，可重建objects。

完整路径/库/摘要hash、旧失败、实际counter/质量观察和未完成范围见
[机器记录](../design/validation/local512-encoder-compiled-gqa-20261009.json)。
完整Z/Qwen全矩阵、encoder盈利/Private-Public按operation选择、更多
scene/seed及GGUF可复用ahead-decode仍未完成；不以本轮通过兼容测试
或局部kernel速度替代原目标。
