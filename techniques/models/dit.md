# DiT：重复步的计算、布局、缓存和混合 FFN

<!-- knowledge
id: technique-models-dit
type: guide
stage: dit, ffn
operator: attention, gemm, norm, rope, swiglu
backend: cpu-reference, metal, mlx
device: cpu, gpu, ane
symptom: stale-cache, launch-bound, precision-loss
semantics: numerically-equivalent, approximate
representation: bf16, fp16
boundary: producer-consumer, gpu-ane
api: public, reference-only
-->

[阶段地图](stage-map.md) · 本地 AdaLN / KV / CFG 样本 · [GPU/ANE 样本](../designs/hybrid-ffn.md)

## 先确认架构和重复边界

记录 latent token 数、text/reference token 数、head geometry、modality/position axes、joint 或 cross attention、AdaLN/调制方式、CFG、step schedule、LoRA。DiT 的输入和调制通常逐 denoising step 变化；“prompt 不变”不证明 text 相关的全部中间状态都不变。双流模型与 joint self-attention 中，text 分支可能通过 image 交互逐层改变。

热循环约为 steps×blocks；减少每层流量、保持 weights/bindings/program 热、消除每步重复编译，收益可重复积累。但冷请求还受 encoder、load、VAE 和媒体 floor 限制，必须完整测量。

## 优先保持语义的 GPU 方案

QKV projection 与 packed GEMM；QK norm/RoPE 融合；AdaLN 的 norm 后 scale/shift 融合；activation/gate/residual epilogue；online attention；固定 shape 编译和 command graph。每个融合保留 norm 归约维度/epsilon、position ordering、dtype/cast 点与真实 bias。不能把全局 norm 擅自变成每个 tile 局部 norm。

`dit_conditioning`独立比较分开 LN→scale→shift 与 `adaln`。这是 Python float 代数等价，不是 FP16 cast 顺序相等；设备实现仍要与原 recipe 对照。

## 允许缓存什么

独立 cross-attention 且 context K/V 只由固定 context 和固定权重投影得到时，可在 step loop 外计算一次 K/V；Q 随 latent/timestep 改变但不影响该证明。本地样本对不同 step 的 Q 比较重算 KV 与 hoist，变化 context 则触发明确反例。

若 context 先经过 timestep-dependent 调制、adapter 改变投影、joint attention 更新 context，或 K/V 使用当前 image，便不能使用上述 cache。跨 step 的 self-attention KV 一般随 latent 改变。跨 seed conditioning cache、reference prefix cache、cross-KV cache 和 residual reuse 是不同对象，key 与质量门禁不可混用。

## GPU/ANE 与 LoRA

对逐 token FFN 用 F-channel split；GPU 保留 attention 与 suffix，ANE 计算 prefix 完整非线性，最后 down partial 相加。运行时 staging/graph/source、W banks 和 A8 slots 有界；非法把同层 attention 与依赖其输出的 FFN 重叠。完整设计与可执行 LoRA algebra 均在 [hybrid-ffn](../designs/hybrid-ffn.md)和 [split_ffn_reference](../examples/split_ffn_reference.py)。

CFG 两分支最终 prediction 可 batch 或分别算；这会改变实际 M 和内存峰值。不能在任意非线性前先合并 conditional/unconditional latent，再套相同 guidance；本地反例展示错误。Branch parallelism 也需测共享带宽与实际 device 边界。

若实际图像生成block本来是parallel attention/MLP、共享调制输入，可将joint output projection按输入列拆开，在GPU跑attention partial、ANE跑完整MLP partial；不需要等待attention再启动MLP。这与上文顺序block不同，完整代数、真实源码提炼和demo见[拓扑设计](../designs/parallel-topologies.md)与[模型选择](hybrid-selection.md)。

## 近似方案的单独验收层

量化按 layer/timestep/granularity 搜索，保留敏感层 high precision；calibration 覆盖多个 prompt、step、latent幅值、参考图和 adapter，holdout 不参与阈值搜索。Residual replay/多项式预测、省某些 block、token merge、sparse attention 都需要严格的 off/full control 和误差预算。少 step 或换 sampler/分辨率是工作负载变化，不列为同语义倍率。

[caches.py](../examples/reference/caches.py)提供真实可读 residual 状态、失败 reset、token mass merge；[policies.py](../examples/reference/policies.py)检查组合冲突。它们不声称完整复现所有缓存论文或授予真实模型质量资格。

## 内存路径和实测命令

若 weights 放不下，保留固定 resident subset，余下 bounded pread/typed refill，不用循环 LRU；scratch、conditioning、跨步 latent 和 VAE reserve 都入账。热每步省时若被 SSD 读、driver waits 或过度编译吃掉，应如实保留负结果。

验收包括完整 seed/prompt/reference/LoRA 纵切、首中末 step、full request、temporal consistency、cache reset 与失败回退；不是最终 tensor finite 即合格。新 topology、step schedule、precision、shape、source 或模型版本改变后重开实验。
