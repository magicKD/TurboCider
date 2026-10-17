# GPU/ANE 并行 FFN：代数、双 bank、校准和失败回退

<!-- knowledge
id: technique-designs-hybrid-ffn
type: method
stage: ffn
operator: gemm, swiglu, scheduling
backend: coreml, mlx, cpu-reference
device: gpu, ane, cpu
symptom: straggler, lifetime, wrong-layout
semantics: numerically-equivalent
representation: fp16, fp32
boundary: gpu-ane, producer-consumer
api: public, reference-only
-->

[设计目录](README.md) · [本地完整切分 / LoRA 实现](../examples/split_ffn_reference.py) · [状态机](../examples/reference/contracts.py)

## 先给出可以移植的计算图

令 `X[M,H]`、gate/up 权重 `[F,H]`、down `[H,F]`，完整 SwiGLU：

```text
G = X Wgᵀ; U = X Wuᵀ; Z = SiLU(G) ⊙ U; Y = Z Wdᵀ
```

沿 intermediate F 分成互斥全集 A/B：各自计算 gate/up、各自做完整 SiLU×up、各自乘对应 down 列，最后 `Y=YA+YB`。这是合法的，因为非线性在 F 元素上独立、down 对 F 是归约。把 H-input 分支分别过 SiLU 再求和则错误，必须先合并 gate/up 再过非线性。Sequence 切分对逐 token FFN 合法，对全局 attention 要额外处理全局 KV/softmax。partition_axes.py给出全部正反例。

LoRA 若 `ΔW=B A`，gate/up 的修正必须先进入 G/U；down 的 `Z A_downᵀ`可按 F 分支求 partial rank 后相加，再乘一次 B_down。不能把未经 adapter 修正的 Z 给 down LoRA，也不能多乘两次 B。可运行 [split_ffn_reference.py](../examples/split_ffn_reference.py)验证不均匀 cuts、strength=0/负值和完整 down 修正。

## 从 Splash 提炼的执行方案

不是永久持有所有层的 ANE 权重，而是图/程序按 shape/recipe 复用，runtime weights 放入两套固定 W banks。每套含 gate/up/down codes 与 scales，bank identity 包括 checkpoint 内容、layer、slice、basis、scale/rounding 配方和 source generation。下一层的 staging 可以与当前层独立工作重叠；不得跨越真实 source 就绪边。

```text
W bank0(stage L) ------ ready0 ----- ANE FFN L ------ done0 ---- 可复用
attention L -> input pack -> fork: GPU suffix L + ANE prefix L -> join L
W bank1(stage L+1) ---- ready1 --------------------- ANE FFN L+1
```

W bank 数量与 activation slot 数量是两个容量约束，不要把“双 W bank”理解成无限 activation 预取。内存下界包含两套 W、A8 codes/scales、partial outputs、恢复 scratch、program 和仍借用的 source，不能只算最终量化权重。

`DualBank`保留 staging/ready/executing/quarantined/free 和每次 generation；第三个 claim 触发 backpressure；timeout 不释放 bank；物理完成后的 late output 仍丢弃。设备版应把 token 接到实际 ticket/event，而不是仅调用一个 CPU 状态方法。

## 精度与 ANE 图

W8/A8 的“8”只定义存储，不证明整数 MAC 或硬件驻留。一个 normalized 配方可用 `q/128×s`表达；activation 与 weight 必须匹配相同 rotation basis，down 也要匹配 hidden basis。F16 rounding、scale shape、零行、group size、padding、source pitch 都必须固定。[quantization.py](../examples/reference/quantization.py)和 [common.rotate](../examples/reference/common.py)给出可读实现。

Splash 被检查版本选用特定 segment 粒度及 mt19937 signs；本地教学样本采用显式 unsigned64 recipe，**不是该项目 bit-exact 重现**。分段/最小 row function 是实现与 shape 的选择，不是所有 ANE 的硬件定律。Public Core ML 的本地 FP16 graph、并行 FFN bench和 native INT8 shared ABI分别隔离图语义、真实 fork/join 与共享特征边界；最后一个只是单 matmul，不等于整套 FFN。

## 校准、最小 rows 与运行中 breaker

优化后的 full GPU 是控制。对每个 rows/bucket、precision、share 和 source 测完整 pack→fork→restore/join；不能用 ANE exposed wait 代替独立分支跨度。理想下界为 `pack + max(G,A) + join`，实际同时执行时 G/A 会因统一内存争抢变慢。Splash 的 `max(A,G+uA)`给出可用的耦合思路，但其 u 和高 share 假设不可跨机器继承。

本地 [select_share](../examples/reference/policies.py)只在两档已测 share 间插值，拒绝外推和不盈利候选；选择后仍需独立实际 candidate trial。Padding 到固定 rows 有固定税，最小盈利 rows 必须逐 bucket 测，不能用一个长 prefill 斜率推短 decode。运行中 breaker 应观察相同 geometry 的完整层窗口，扣除 warmup，持续不盈利则退回 GPU；改变 graph/芯片/形状/配方后再显式重开，而不是无条件永久关闭研究方向。

本地[ProfitBreaker](../examples/reference/contracts.py)把这一策略实现为有界窗口：先丢弃显式warmup，再比较匹配完整层的split/GPU比值中位数；持续超过噪声容差才关闭；单个慢样本不触发；关闭后不隐式重开，变更identity必须显式重新资格验证。`hybrid-policy`是合成时间对照，真实adapter应低频采样控制，不能每个请求额外跑一次完整GPU而不计成本。

## 失败与验收

当前 partial 失败、finite/parity 不合格、ready 超时：该 chunk 不发布；停止新 split；保活/隔离在途资源；整边界 full GPU 重算。不将旧层 output、另一 source 的 bank 或只完成一半的 LoRA join 当 fallback。

真实设备命令见 [ANE lab](../examples/README.md)。重新探索条件：不同 H/F/M、bucket、source、rank、basis、program 生命周期、带宽、SDK 或消费者；验收仍须完整请求数值/画质与内存边界。出处只作 [溯源](README.md)，本篇与本地代码已包含核心设计。
