# VAE、因果卷积、全局统计与空间 tiling

<!-- knowledge
id: technique-media-vae
type: method
stage: vae
operator: conv, norm
backend: metal, mlx
device: gpu
symptom: memory-bound, precision-loss, wrong-layout
semantics: numerically-equivalent, approximate
representation: fp16, fp32
boundary: producer-consumer
api: public
-->

[总目录](../README.md) · [视频 IO](video-and-audio.md)

## 先保住原始精度与因果语义

VAE常常决定最终峰值和可见质量，尤其视频。记录原checkpoint系数、FP32/FP16计算边界、norm范围、attention上下文、temporal kernel/history、upsample phase、padding/stride、输出cast。不能用DiT的量化资格覆盖VAE。

SeedVR2稳定路线保留原FP16 projection系数、FP32 residual/state/norm等配方；更换为BF16或把初始重复帧fold改成FP16卷积不是无损重写。可复用入口：[VAE优化全景](../../docs/design/seedvr2-vae-optimization.md)、[真实实现](../../native/models/seedvr2/vae.cpp)。

## 高价值的 exact-context 优化

| 技巧 | 机制 | 验收重点 |
| --- | --- | --- |
| temporal-kernel-one用2D | 移除无必要3D维 | 原权重、layout、cast、bias |
| 初始重复帧fold | 对确实相同的temporal taps合并系数 | FP32系数/累加与原始witness |
| owned causal history | 不让小view pin全temporal concat | owner、下一帧复用、峰值是否移动 |
| norm/SiLU/halo pack融合 | 不materialize大normalized tensor | whole-frame统计、tap/halo顺序 |
| streamed frame sink | 及时转换/消费而非concat全clip | backpressure、舍入、source帧完整 |
| row-file stage scheduler | 中间层存储外置，bounded tiles | 全局norm/attention、磁盘配额与IO |

这些是不同边界，不能因一个卷积workspace有界就称整个VAE有界。

## 精确 stage tiling 与独立 VAE crops 不同

精确上下文：whole-frame统计单独pass；query tiles依然读完整K/V；卷积保留halo、padding、stride和所有temporal history；时域upsample的首帧/后续相位保持。

独立crop：各tile自己norm/attention，语义已经变；overlap/blending只能缓解接缝，需显式近似模式和视频闪烁门禁。不能把独立crop结果借用exact-context资格。

## Conv3D → stacked Conv2D

固定causal temporal taps可按tap/channel顺序堆为Conv2D输入，权重按同一顺序重排，保留空间halo和bias位置。初始帧重复是特殊case，不可假装非初始帧无history。

GPU/ANE合作可切output channels：双方读相同输入/taps，各算完整对应输出，再按原channels组装；不在temporal/input通道上引入额外partial sum。小CoreML tile必须把pack、ready、predict、restore、assembly全部计入。

[可行性screen](../../docs/design/seedvr2-vae-ane-feasibility.md)里某些small-tile CPU+NE core很快，但完整512卷积的两槽pipeline仍慢于GPU Conv3D，且外部surface使footprint增加。随后 [direct-frame virtual halo](../../docs/design/seedvr2-vae-virtual-causal-conv.md)改变边界、避免全halo存储，才在特定完整decoder上降低active maximum；不能把早期core数字重标成后来的E2E。

## Tile变大/对齐不一定更快

[halo-aligned tiles](../../docs/design/seedvr2-vae-halo-aligned-tiles.md)比较64×62/64×94：较宽tile减少prediction次数，却未建立显著完整decoder或全请求收益，active峰值不变。tile需同时考虑halo比例、IO pool、图限制、predict开销和新峰值，而非只追calls少。

## GPU/ANE VAE 的采用边界

当前特定projection/causal conv实验不等于whole-VAE on ANE。全局norm、attention、upsample、causal state及其他层仍要实现/验收；compute plan preference不证明全部物理在ANE。

完整验证：原始FP32witness、每层finite、initial/subsequent phase、image/complete25等实际上下文、全部帧RGB/alpha、late cancel/recovery、complete decoder ABBA、full request峰值。更长codec测试不能抬高模型121源帧限制，64GiB机器的实验也不证明16GiB支持。

## 推荐迁移顺序

先phase trace定位真实高水位 → 修history owner/冗余输入 → fuse norm-pack/减少halo → bounded output → exact-context stage tiling → 测单个CoreML边界 → 合作卷积 → 联合DiT residency。每步先证精度，再重新定位全请求最大值。

## 实现参考与实验

可读实现：[参考代码](../examples/reference/operators.py)；[实验步骤与边界](../examples/README.md)；本项全部 case / controls。

本项控制：full context vs lost halo/local norm；initial fold vs subsequent temporal taps。CPU参考/模拟与实际设备入口分开标记；设备环境和更多变体见实验步骤。

可以重开探索的条件：different causal/spatial phases, graph boundaries or tile schedule。历史负结果限定于其来源/shape/边界，不关闭该方向。
