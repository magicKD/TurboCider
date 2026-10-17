# 异构规划：可选设备、first-consumer deadline、尾延迟与精度岛

<!-- knowledge
id: technique-designs-heterogeneous-planner
type: method
stage: ffn, pipeline
operator: scheduling, quantization
backend: cpu-reference
device: cpu, gpu, ane
symptom: straggler, precision-loss, memory-bound
semantics: contract-only, approximate
representation: fp16, fp32
boundary: gpu-ane, producer-consumer
api: reference-only
-->

[设计目录](README.md) · [本地planner / 精度岛](../examples/reference/heterogeneous.py) · [按模型选择](../models/hybrid-selection.md)

## 设备必须允许缺席

mac_local_ai与mac_transformer都把GPU、ANE、CPU分片视作可选策略，而不是追求三条曲线都繁忙。CPU的utility QoS可降低对主路径干扰，也会受系统负载影响；ANE固定row与launch也会造成straggler。Planner目标是完整first consumer所见的wall/p95，不是把三台设备的标称TFLOPS相加。

本地`DeviceCost`使用明确的合成`launch + width*slope`及p95 factor。`choose_partition`枚举F的离散quantum，保持GPU非零（该join adapter的限制，不是硬件定律），允许CPU/ANE为零，加入join与同时执行的contention。示例阈值拒绝ANE错过GPU分支deadline、CPU p95超过较慢accelerator和无最低收益的方案；GPU-only一直是候选。

`optional-devices`展示三组假设：CPU较慢时退出；CPU确实有足够低的完整成本时三设备参与；ANE/CPU都交接昂贵时选完整GPU。这些是可修改的策略示例，不是某个芯片的标定profile。

## 从估算到实际candidate的准入

1. source/shape/basis/adapter合法；memory先容纳weights、banks、partial、join、program与其他阶段reserve。
2. 用匹配完整边界采样GPU、各分支单独、同时执行；记录实际input producer、restore、join、tail和热状态。
3. profile可按有效FLOPs和权重/activation流量构建roofline，但UMA并发争抢需额外共同采样，不能把离散设备带宽相加。
4. 仅在测过的范围提出share，不将large M的slope外推decode；quant group/channel quantum与实际graph一致。
5. 最优估算需要新的实际candidate trial、finite/parity与独立warm windows。低噪声小收益也不自动成为默认。
6. 运行中以有界窗口做[profit breaker](../examples/reference/contracts.py)，关闭后完整GPU回退，变更identity后显式重新资格。

## 小块、artifact数量和持久服务

row bucket按实际image+text/reference总rows选最小可容纳者；最后小chunk可能GPU更好。多function共享weight可节约磁盘，却可能有specialization与dispatch税；单function更快也会增加按shape保存artifact的成本。把prepare/compile/load/first/hot和长期idle/reload分别入账。

flux2-engine把模型、session、binding、output buffer放backend而非request中，明确序列化复用同一slot的请求。one-shot、portable missing bucket和稳定tail可能适合pure GPU；明确auto与explicit offload的失败合同，不用“auto”隐藏数值错误或半个输出。

## 为什么完整attention放置要单独门禁

不只比较ANE SDPA内核：QKV投影、head/position/norm布局、mask、backing、context、output projection和返回tensor一起定义候选。长rows时attention约随sequence平方增长，MLP约线性；某个图像尺寸的MLP offload受益不意味着更高分辨率仍同倍率。固定桶、tail spikes和device placement还需独立trace/测量。

mac_transformer与flux2-engine保留了reject-full-attention的准入思路：完整bridge不赢强GPU、数值不合格或tail不稳就不启用。这不是证明ANE永远不能做attention；改变graph/布局/芯片/consumer后可重开。

## 高精度岛：把误差生命周期也加入规划

mac_local_ai的GDN探索按组保留高精度投影、norm、状态参数与递推路径，而非笼统把整个模型升精度。持久状态会累积误差，普通feed-forward临时值则可能在边界消亡；分配精度预算要按这个生命周期。

本地`precision-islands`是标量递推`state=a*state+b`：比较全低精度、只升state但参数已被舍入、参数+state完整FP32岛。它说明FP32累加不能找回已舍入的a，及state每次低精度写回的累积代价；**不是GDN算子实现或真实质量结论**。

真实精度岛要算额外shadow weights、状态、cast、selected group packing、consumer projection和完整holdout质量；若只替换某个中间capture而不计重算成本，只是oracle，不是可交付加速。将已观察的失败保留为当前recipe边界，而非禁止其他group/layer/bitwidth探索。

重开条件：rows/shape、source、quant、QoS、热/内存状态、artifact打包、consumer、group选择或device改变。原项目版本与源码指纹见[阅读记录](../references/gpu-ane-projects.md)，本篇已包含理解planner所需的主要方法。
