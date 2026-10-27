# Qwen base512：固定function family的完整40步接入

2026-10-19，M4 Max/64GiB。目标仍为更快的真实Qwen生成/编辑。本轮将已过真实
完整FFN检查的family接入显式生成路径，**没有默认采用，也没有commit**。

## 接入与边界

默认不设置`TURBOCIDER_PRIVATE_ANE_FUNCTION_ROWS`，原单bucket路径保持。
显式设`512,1024`或`512,1024,1536,2048`，manifest的物理rows必须等于最大
function。初始Qwen域：原BF16 transformer/encoder叶文件、base40、resident512、
MLX、approximation显式开启、无LoRA/PE/encoder ANE/DiT cache/streaming/S1，
生成或1–2 full-ref1024编辑。保持原强Private QKV、one-hop lookahead及prefix
快照；生成prefix hit的首步QKV仍按既有策略留GPU。

Executor新增逻辑row alignment、实际function列表及capability计数。HybridFfn
按最小512行补齐，PrivateW8Graph以同一最大workspace按function执行；每个
function都做三种带符号/换W的原self-test并逐点验证。两function实际6次，四
function实际12次初始化预测，而不是把所有function只验证成一次flag。

family进入executor身份、调度/快照算术namespace，prefix健康门同时核实际
function列表、物理bucket、最小alignment、完整capability次数和原fixed健康
条件。不同family和旧fixed的recipe SHA实际不同，匹配重复请求内则稳定；
不允许把旧prefix算术冒充新family。QKV自己的固定512行图不因此改成family。

Native V9为666输入exact source archive；新增虚函数/metrics改变C++ ABI，
旧V7/V8 helper不可混链V9，Python C API路径按实际runtime ID绑定。

## 完整请求结果：灵活性接通，512没有稳健净增益

同V9、同原权重、全部40步、seed73→74→73，分别独立owner/process，保留
原QKV/lookahead及生成/编辑prefix策略。单位秒；first只叫首请求，非OS冷缓存。

| 工作 | 路线 | first | 热中位数 |
| --- | --- | ---: | ---: |
| 生成 | 原R1024 fixed | 30.3291 | 28.7052 |
| 生成 | family512/1024 | 31.9732 | 28.7101 |
| 生成 | family512/1024/1536/2048 | 30.4886 | 28.8883 |
| full-ref编辑 | 原R1024 fixed | 44.5590 | 36.1278 |
| full-ref编辑 | family512/1024 | 44.5709 | 36.0636 |
| full-ref编辑 | 四function | 44.8432 | 36.0650 |

主decode是1024行，原图已经一个chunk；两function基本持平，四function生成
反而略慢。更多灵活性没有自动变成512热请求提速。首请求还包含不同编译cache
状态、验证、加载和conditioner成本，不能只截掉多function初始化后宣称冷更快。

实际每请求workload FFN预测如下；各请求仍1280个成功混合FFN block：

| 路线 | 首次生成 | 首次编辑 | prefix-hit热请求 | 初始化capability |
| --- | ---: | ---: | ---: | ---: |
| fixed1024 | 1312 | 1440 | 1280 | 3 |
| 两function | 1312 | 1440 | 1280 | 6 |
| 四function | 1280 | 1344 | 1280 | 12 |

四function减少的是prefill padding/分块，不能把这些少调用误算为少采样步数。
两个/四个function都仍只用一个FFN model，另一个model是原QKV图；每条三请求
timeline均两个稳定model、hot compile/load0，所有function已在请求初始化验证。
不据此承诺opaque驱动内部永远零准备成本或物理INT8 MAC。

所有11个成功窗口均记录外部ComfyUI Python CPU负载，性能gate未过，正式
speedup字段为null；上述为有限原始观察，不更改默认或据此执行条件commit。

## 独立完整媒体/来源审计

`outputs/qwen-function-session-audit-v2-20261019.json`通过：11成功窗、31张完整
RGBA PNG、1240个实际采样步、39680个成功混合FFN block；666输入native归档、
原model/encoder/VAE全SHA、实际manifest/reference/observer、工具原字节、原始
process-tree memory hash-chain和真实Private selectors全部绑定。

四组fixed/family、每组3请求的**全部RGBA逐像素一致**，RGB SSIM1、alpha差0。
这只证明本轮family相对原W8路线没有新增图片差异，不把原W8对纯GPU的近似
误差或单提示词语义质量自动视为合格。生成和编辑样本目视检查未发现新增
family伪影；不能凭这两类茶壶样本承诺文字/身份等任意场景质量。

编辑本来就有miss/hit的算术路由差异：miss长prefill QKV留GPU，hit首步进入
Private QKV。相同seed首/第三请求在原fixed和两种family中均RGB SSIM约
0.998314，而不是全RGBA exact；family对对应fixed请求仍exact。审计记录此
差异，不把不同cache/placement的请求强行当逐位恢复试验。后续可单独检查
编辑hit首步保留GPU QKV是否同时改善这一差异与耗时，尚未实现/测量。

Stock原callback按设计不解析/保存progress JSON，所以其40步资格使用native
actual_steps和每请求全部1280成功FFN block；timeline另核完整0...40进度。
不为stock虚构一份progress日志，也不拿requested steps单独当完成证据。

73项相关CPU测试通过，覆盖旧默认golden、请求域、prefix健康、逻辑padding/
预测次数、完整媒体比较和受污染/timeline/单请求不得产生倍率的负控。

V8初pilot在denoise前因遗留joint-QKV硬1024物理guard拒绝；该失败保留。修正
为共享descriptor/family最大值检查，仍要求actual loaded family相符；没有删
健康门、降低精度门或停止共享服务。最终V9完整请求和审计独立完成。

## 工程与下一步

`runtime_function_options.hpp`负责Qwen请求域；`ane_w8_functions.hpp`负责有界
列表/identity；backend创建、最小pad、全部function self-test和JSON metadata
沿原统一Executor接缝接入。`collect.py --function-rows ...`按实际geometry核
成功预测与capability，`function_request_audit.py`单独审计完整PNG/来源/内存，
受污染或timeline/单请求不产生速度比。所有中间产物在已忽略的outputs/build。

保持原fixed为512默认研究基线。更大输出/其它参考数、LoRA/GGUF/ConvRot/
encoder及Z/LTX没有随本接入自动扩域。后续优先用已定位的阶段/交接瓶颈做
单变量完整40步对照，而不是把已无padding的1024主decode再机械拆小。
