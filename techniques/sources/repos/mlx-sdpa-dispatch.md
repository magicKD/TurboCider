# MLX SDPA：guard、stride、mask与输出布局

<!-- knowledge
id: source-upstream-mlx-sdpa
type: source
-->

[来源索引](../README.md) · [attention技术卡](../../metal/attention.md) · [本机完整对照](../../metal/case-studies/qwen-attention-replay-and-empty-row.md)

固定源码[ml-explore/mlx v0.32.0](https://github.com/ml-explore/mlx/tree/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247)，commit `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`；MIT、Apple版权。SHA与选读范围保留在[同一登记](../README.md)。执行的是已安装0.32.0，其binary/dylib/metallib hashes在证据中；没有从源码重建MLX，也不把相同版本号当构建等价证明。

## 先读入口guard，不假设fast名字一定fused

`fast.cpp`的`scaled_dot_product_attention`检查rank4、batch、Q/K维度、Q/KV head整除、mask/sinks类型与广播。GQA fallback按组解释query heads，不是随意重复或删KV heads。causal非方阵使用`offset = kL - qL`，即右下对齐，不是简单左上三角。

`scaled_dot_product_attention.cpp`的`ScaledDotProductAttention::use_fallback`对training、CPU、head_dim和query长度分别判断。该固定版本的full路径是Q>8、匹配Q/V head_dim且D为64/80/128；vector路径支持的D集合不同，还限制`Q*gqa_factor<=32`等。长KV的小Q另有两pass条件。当前小GQA controls不等于验证了所有长KV vector路径。

这里的fallback首先是实现选择；它可在GPU stream上调用unfused matmul/softmax，并不自动等于CPU设备回退。安装版本的docstring也确认causal右下对齐、softmax FP32和GQA不应预先复制KV heads；本轮未把当前网页文字当旧binary的额外语义保证。

`is_nax_available`还受编译条件、OS和architecture generation限制；源码中桌面generation门槛为17，本机API为`applegpu_g16s`。不能仅凭“M4 Max支持新SDK”断定NAX。以上是固定源码的路径预测，不是本轮每个dispatch的物理追踪证明。

## 非contiguous不是一个统一坏标签

full attention的`copy_unless`主要检查末维stride=1；它接收其他head/token stride作为参数。实际prefill V是token-major interleaved head view，满足该要求。强制`contiguous(V)`并非修复非法输入，而是另一种有复制成本的数据流。vector路径的Q/K/V copy predicates又不同，不能跨路径搬规则。

full输出主动以`strH=D, strL=H*D`布局分配，即逻辑BHLD但物理token-major；这方便下游transpose/reshape到`[B,L,H*D]`。额外强制head-major可能增加复制，不能只看`row_contiguous=false`就宣布性能缺陷。

## Online softmax仍需要明确empty-row合同

所读Steel attention使用FP32 accumulator、逐KV tile更新max/sum/output；bool mask与tail/causal会写`finite_min`。最终按sum归一化，不存在“因为mask全false就必定返回零”的通用推论。fallback也使用有限最小值替换bool禁用项。对当前binary的常量V探针，D64/K29的全false行返回29/32，K32返回1；这与有限sentinel、tail加载和归一化方向相符，但本轮没有物理kernel定位，不能升级为已证明的单行根因或硬件故障。

API的[公开说明](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html)用于接口导航，活动网页版本不冒充固定读本。若调用方要求全空行输出零，应把该约定、输入有效性或安全key/最终置零机制明确实现并验证；本轮未把这种新wrapper加到产品默认。

## 已运行的边界

7个synthetic tail/mask/head_dim/GQA controls，两个真实seed的prefill/steady三候选，共21个比较含2个noop；另有常量V诊断。真实QKV来自actual GPU block arguments的原配方重放，完整block/prefix过控制，但不是未改变compiled kernel内部的直接抓取。详见。

没有移植上游shader、运行全部上游测试或授予全请求资格。新版本/architecture、不同D/Q/K/GQA、mask空行/类型、stride、LoRA或完整consumer改变时重新审读与测量。
