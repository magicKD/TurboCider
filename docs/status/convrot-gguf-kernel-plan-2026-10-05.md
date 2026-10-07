# ConvRot / GGUF：GPU kernel、提前解码与混合 ANE 接续

后续 direct-code ANE 实现与新验证见
[ConvRot W8A8 接续](convrot-direct-ane-w8a8-2026-10-05.md)。下文保留本次
接续之前的 GPU／GGUF 分析与历史缺口，不将旧记录改写成已通过资格。

用户于2026-10-05明确将现有ConvRot与GGUF优化纳入范围。本轮审查已有
working-tree实现/记录并新增GPU旋转候选；不将旧的未提交研究改动
悄然视为发布默认。原双后端四格base≥1.2×、LoRA与质量目标仍未完成。
校准GPU traffic实现提交为`5294c07`；Quad GPU实现为`5f62b84`。

## 本轮已经实现的 GPU 候选

`native/backends/convrot_rotation.hpp` 保留原shared-memory H256为control。
单组SIMD减少barrier后，在K10240仍略慢，不选为runtime优化。四组
H256共用一个TG，以float4独立计算四个Comfy radix-4变换：前两级
SIMD shuffle、后两级共享scratch；保留原加减顺序、FP32中间值与一次
最终dtype舍入。它不是Sylvester排序，也不是A8。

72个FP32/FP16/BF16、contiguous/strided、1/33/1056/4128-row与多K
case对shared核逐位一致；256个独立basis row验证Comfy H4^4 ordering。
实际库的原packed BF16-scale Q8完整FFN在1056/4224行也对shared核
逐位一致；原96项raw/packed、mixed/bias、多dtype helper回归通过。

同一独立binary、BF16、3次warmup、9个hot全部保留、交替顺序，host
operator span（不是GPU timestamp或E2E资格）：

| rows / K | shared median (ms) | quad median (ms) | 局部倍率 |
| --- | ---: | ---: | ---: |
| 1056 / 3840 | 0.265542 | 0.256416 | 1.03559× |
| 1056 / 10240 | 0.453542 | 0.413166 | 1.09772× |
| 4224 / 3840 | 0.648708 | 0.572041 | 1.13402× |
| 4224 / 10240 | 1.438917 | 1.262708 | 1.13955× |

`convrot_rotate_metal`仅在既有显式Metal ConvRot路径、M4 Max、BF16、
1024–4224行与K3840/10240使用quad。默认dense-H、其它device/dtype与
小modulation向量不变。没有把旋转组件收益当成整请求收益。

新实验库 `build/convrot-quad-private` SHA-256：
`e0040c566be7fa09d6bd8b5291f7b169fff6e13e95e2a57a528c6a4b22ce081d`。
Private=1且experimental-probes=1，不用于稳定App分发。独立Public
默认库 `5de78f66948769601e52749ed8bd8826ed34da947c2ae225525ce1ffbf96aeca`
的发行flags/private-class/direct-link guard通过。

实模型512²/一步diagnostic、原ConvRot checkpoint/legacy BF16尺度，
compiled-butterfly与legacy dense-H独立trajectory共35项comparison：
7个后段block N1失败，final latent N1通过。**整体验收未通过，不能
扩大默认或放宽门槛。** Quad是对旧Metal butterfly逐位保持的优化；
这不消除Metal butterfly与dense-H不同归约顺序造成的误差。本轮尚无
同binary shared-vs-quad完整trajectory A/B、1024²或正式媒体资格。
目录 `outputs/convrot-quad-model512-source-check/`。

## GPU 慢因与提前解码

现有retained3GiB同源匹配warm screen中，ConvRot compiled-dense对BF16
wall ratio=1.234759，process memory ratio=0.562420；它仍没达到其
20% latency门槛，不是1.2×加速。见既有
`docs/design/quantized-execution/validation/convrot-compiled-retained3g-bf16-screen-20261002.json`。
因此不能假设compressed权重必然比当前优化BF16 GPU快。

GGUF的容器header/index初始化和每层读/解码是不同成本；当前pager
已经有packed source、fixed-output GPU affine bank、pool/ticket/retire
生命周期与bounded prefetch。重点应是消费格式与数据流，不先改GGUF
磁盘文件或制造永久dense sidecar。

推荐对每种M/K/N/encoding分别测三条同源GPU路线：packed QMM、按槽
提前decode到dense BF16/FP16+优化GEMM、fused decode/GEMM。选择条件：
`decode_cost / 实际复用次数 + dense_GEMM + 额外memory traffic < packed_QMM`。
resident跨step复用可以摊薄decode；bounded eviction后每step重解码
则不能按“只付一次”分析。M=1 modulation/text GEMV和M≈1056/4224的
DiT大GEMM应分开，不能用整模型名作统一选择。

以Z FFN H3840/F10240为例，三张FP16/BF16 dense matrix合计225MiB，
两槽仅FFN就是450MiB，尚未包括attention/modulation、旧槽、decoder
temporary、allocator slack和ANE slots。完整dense常驻会侵蚀ConvRot/
GGUF内存优势；先比较当前+一层前瞻，受ledger/backing fences约束。
提升prefetch深度只能覆盖抖动，不能修复长期供给吞吐不足，并可能
与GPU GEMM/ANE争统一内存带宽。原master codes/scales不能改。

ConvRot提前decode有两种独立precision recipe：

1. 保留rotated basis，`signed C × stored row scale → dense`，在线XH仍在。
2. 解码时inverse Comfy H256，生成effective W后普通GEMM；增加W的
   N×K变换及新舍入边界，必须独立source/latent资格。

第一种优先试有界dense槽；第二种只在实测复用/预算允许时试，不
恢复原始未量化BF16，也不将rounded BF16尺度冒称原始FP32尺度。

## ANE / Private–Public 混合路线

当前Private W8 stager已支持dense、MLX affine Q4/Q8、raw GGUF
Q4_0/Q4_K/Q8_0/Q6_K，无完整dense中间库，且有两套W banks与共享
事件。新增的独立校准GPU arm覆盖W/A staging、冻住的restore/join、
future weight与memory admission，已用Z/Qwen完整BF16 transformer
常驻、多个深度、1056/4224 buckets验证组件正确性。它仍未与
bandwidth fit/cache及runtime自动share/prefetch选择接通。

ConvRot仍有明确接入缺口：`device_weight_view()`拒绝
`ComfyH256Inverse`，当前Private W8A8不能直接消费该transform。
Public FP16 runtime可以通过现有integer-H256 CPU SIMD converter
消费legacy packed源；不能把Private退到GPU说成Private已加速。

优先方案：给Private增加独立`Comfy H256 + 原signed Q8` recipe，
保留Qw的-128、原row scale，X经Comfy旋转后A8，正常scaled epilogue；
FFN down也用正确Comfy basis。现有Sylvester H128/H512 graph不能直接
套用。此方案避免再次W8量化，但仍新增A8/中间舍入，必须 gate/up/
hidden/down/source oracle和整模型验证。另一方案是GPU有界inverse
Comfy→现有Sylvester W8，改动较小但增加转换带宽与重量化误差，
不默认认为更快或更保真。

混合建议按operation/layer选一种ANE backend，再与其GPU complement
并发：Private适合已资格的长行数W8 FFN；Public适合不支持的transform/
格式、粗粒度FP16稳定回退；短/敏感层及M=1留GPU。不要为同一片权重
同时准备Private W8和Public FP16三路常驻副本。Private/Public的
身份/计时/校准cache必须隔离；真实失败仍整operation GPU重算，不
发布partial scratch。Public CPU+NE policy不是物理ANE驻留证明。

## 接下来的实现和验收

- GPU：先完成shared/quad同binary完整trajectory和组件/整请求分开
  计时，再试rotate+GEMM epilogue融合，不能去掉BF16边界换取伪收益。
- GGUF：格式与dtype专用decode、当前/前瞻slot ready wait、decode
  bytes/time、actual GEMM/host dispatch分别测；不改model/reference文件。
- ANE：实现ConvRot独立direct-code recipe或有界compound变换，再按
  model/shape/encoding/recipe/backend/ABI进行实测share/prefetch选择。
- 所有候选保留完整GPU fallback、原source/adapter与负结果；依次验
  512/1024、LoRA、latent/媒体/内存/物理trace和matched optimized GPU。

目前14 Private、13 Public Core ML/MLX、60 screen、8 memory/host
回归通过；不声称完整`make test`绿色或native INT8 MAC被观察到。
现有ConvRot working-tree研究改动仍保留；本轮GPU候选不替代其未完成
的formal performance/quality/cold/whole-request资格。

关键log/库哈希与失败门槛见
[机器记录](../design/validation/convrot-quad-and-calibration-20261005.json)。
