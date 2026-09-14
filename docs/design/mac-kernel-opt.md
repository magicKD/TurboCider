# TurboCider：Apple Silicon 视频生成 Metal Kernel 优化路线

> 目标：在 Apple Silicon（优先以 M4 Max / 64 GB Unified Memory 为开发与验证平台）上，为 H3、Wan、HunyuanVideo 等视频 Diffusion Transformer 构建**真正 Metal-friendly** 的高性能推理 kernel。  
> 核心目标不是单纯追求最高 sparsity，而是同时优化：
>
> 1. **真实 wall-clock latency**
> 2. **视频生成质量与时间一致性**
> 3. **统一内存带宽与中间 tensor 流量**
> 4. **Metal GPU 的规则并行度与 tile 利用率**
> 5. **跨模型可复用性**

---

## 1. Executive Summary

在 CUDA 生态中，“高 sparsity”经常能够直接转换成较大的 kernel speedup，因为 CUDA/Hopper/Blackwell 已经拥有成熟的：

- FlashAttention / cuDNN attention
- Triton / CuTe
- block-sparse primitives
- 高效 top-k / compaction / gather / scatter
- TMA / Tensor Core
- FlashInfer 等动态 sparse runtime

但是在 Apple Silicon 上，情况明显不同。

当前 Apple Silicon benchmark 已经显示：

- **gather-based sparse attention 可能比 dense/masked attention 慢很多**；
- 真正有优势的是 **block-level、连续访存、规则 tile 的 sparse execution**；
- 对视频超长序列而言，block sparse 的优势会随着 sequence length 增长而放大；
- 动态 sparsity 如果需要 `top-k → indices → gather K/V → repack → attention`，省下来的 FLOPs 很容易被 memory movement 和 kernel dispatch 吃掉；
- 一个 kernel 在 isolation 中快很多，并不意味着最终视频生成会同比例加速；
- attention 数值误差很小，也不意味着经过几十个 DiT block 和多个 diffusion step 后不会出现 temporal flicker。

因此 TurboCider 的核心原则应该是：

> **Regularity before sparsity. Fusion before FLOPs. Layout before theoretical complexity.**

也就是优先做到：

```text
少搬数据
  >
少 materialize 中间 tensor
  >
少 dispatch
  >
规则 tile / 连续访存
  >
真正跳过计算
  >
最后才追求更激进的 sparsity
```

### 推荐总体路线

第一阶段：

```text
Dense Metal FlashAttention
+ QKNorm/RoPE fusion
+ Layout optimization
+ MLP / modulation fusion
```

建立一个足够强的 dense baseline。

第二阶段：

```text
Sol-Attn Metal
        +
Structured exact regions
```

发展为：

```text
CiderSol
=
Local Exact
+ Conditioning Exact
+ Global Anchors Exact
+ Remote Dynamic Approximation
```

第三阶段：

```text
CiderBlockSparse
=
STA / LVSA / GRAT inspired
structured 3D sparse attention
```

第四阶段才考虑：

```text
VSA / semantic clustering / SVOO / SVG2
```

这类更复杂的动态 sparse pattern。

---

# 2. 为什么很多 CUDA 加速 Kernel 搬到 Metal 后效果不好

## 2.1 CUDA 的“高 sparsity”不等于 Metal 的“高速度”

典型 CUDA sparse attention：

```text
Q / K
  ↓
pooled score
  ↓
top-k
  ↓
indices
  ↓
gather K/V
  ↓
repack
  ↓
block sparse attention
```

对于 NVIDIA，这条链路拥有高度优化的生态。

Apple GPU 上，问题往往集中在：

```text
top-k
gather
scatter
permutation
temporary tensor
irregular memory access
many small kernel launches
```

这些步骤会明显降低有效收益。

---

## 2.2 Apple Silicon 的真实 benchmark 给出的信号

AttnBench 在 Apple M4 Mac mini 上报告：

### Gather-based sliding window

N=128：

```text
Gather SWA      ≈ 7.08 ms
Masked SWA      ≈ 1.24 ms
```

N=256：

```text
Gather SWA      ≈ 12.35 ms
Masked SWA      ≈ 1.69 ms
```

也就是说，用 gather 真正“少算”反而出现约 **5.7–7.3× 的额外开销**。

### 真正 block sparse

在 N=1024：

```text
Dense MHA       ≈ 4.38 ms
BlockSparse     ≈ 2.31 ms
```

约：

```text
1.90× speedup
```

这还是 1024 token，而视频模型经常是：

```text
10K
30K
100K+
```

因此我们非常看好**直接 block traversal 的 Metal sparse kernel**。

---

# 3. Metal-Friendly Kernel 的定义

一个 kernel 是否适合 Apple GPU，可以从以下几个维度判断。

## 3.1 优先固定 tile

推荐：

```text
BQ = 32 / 64
BK = 32 / 64
D  = 64 / 128
```

对于视频 token，可以进一步使用：

```text
4 × 4 × 4 = 64 tokens
```

作为 3D video tile。

例如：

```text
Q: [64, 128]
K: [64, 128]

QK^T → [64, 64]
```

内部可以进一步分解成 SIMD-group matrix operations。

---

## 3.2 一个 threadgroup 尽量拥有完整 Q tile

理想模型：

```text
Threadgroup
   │
   ├── Load Q tile once
   │
   ├── Iterate selected K/V tiles
   │
   ├── QK
   │
   ├── online max
   │
   ├── online exp/sum
   │
   ├── PV accumulate
   │
   └── write O once
```

而不是：

```text
kernel 1: route
kernel 2: top-k
kernel 3: gather
kernel 4: attention
kernel 5: scatter
```

---

## 3.3 不 materialize attention matrix

永远避免：

```text
[N, N]
```

attention score 落到 unified memory。

使用 FlashAttention 风格：

\[
m_i,\quad l_i,\quad O_i
\]

流式更新：

\[
S_j = QK_j^T
\]

并做 online softmax。

---

## 3.4 不 materialize sparse K/V

不推荐：

```python
K_sparse = gather(K, index)
V_sparse = gather(V, index)
```

推荐：

```text
block_id
   ↓
直接计算 K/V address
   ↓
直接 load 原始 K/V tile
```

也就是说 sparse metadata 应该只是一个很小的 block map，而不是产生新的 K/V tensor。

---

## 3.5 Sparse pattern 越规则越好

优先：

```text
local window
temporal stripe
spatial stripe
global anchor
fixed prefix
```

其次：

```text
query-block-level threshold
```

再其次：

```text
small pattern dictionary
```

最不推荐：

```text
per-token arbitrary top-k
```

---

# 4. Tier S：首先应该实现的“低风险” Dense Metal Kernel

Sparse attention 之前必须有一个足够强的 dense baseline。

否则：

```text
Sparse vs PyTorch MPS
```

得到的加速比没有意义。

真正应该比较：

```text
Sparse Metal
vs
Optimized Dense Metal
```

---

## 4.1 Dense Tiled FlashAttention

### 目标

实现：

```text
QK^T
→ softmax
→ PV
```

单 kernel streaming。

### 推荐结构

```text
Q64 × K64

head_dim = 128
```

使用：

- FP16 / BF16 input
- FP32 softmax accumulation
- threadgroup / cooperative tile
- SIMD-group matrix multiply
- online softmax

### 质量

基本等价于 dense attention。

只存在 floating-point operation ordering 差异。

### 优先级

**P0 / 必做。**

原因：

1. 它是所有 sparse kernel 的基准；
2. sparse exact block 可以复用完全相同的内部 primitive；
3. 未来 M5+ 可以切换到 Metal TensorOps；
4. M4 仍可走现有 SIMD-group matrix path。

---

# 5. Tier S：QKNorm + RoPE Fusion

视频 DiT 经常有：

```text
Q/K projection
↓
QK RMSNorm
↓
RoPE / MM-RoPE
↓
attention
```

普通实现会反复写回 unified memory：

```text
Q
↓ write
Norm
↓ write
RoPE
↓ write
Attention
```

Metal-friendly 实现：

```text
Q/K projection
      ↓
[Norm + RoPE]
      ↓
attention-ready layout
```

甚至未来可以将部分逻辑合并到 projection epilogue。

### 优点

- 不改变 attention topology；
- 质量风险非常低；
- 显著减少 wide tensor memory traffic。

### 已知工程经验

H3 Apple Silicon 实现中，局部 fused Q/K RMSNorm + RoPE kernel 可以有数倍 kernel-level speedup，但因为该算子占总运行时间有限，所以 E2E 收益可能只有几个百分点。

因此：

> 必须做，但不能把它当作主要的 2× 加速来源。

---

# 6. Tier S：Layout-Aware QKV Kernel

Apple unified memory 并不意味着 transpose 是免费的。

避免：

```text
[B,T,3,H,D]
  ↓ transpose
[B,H,T,D]
  ↓ attention
  ↓ transpose
[B,T,H,D]
```

推荐直接生成：

```text
[H,T,D]
```

或者 kernel 最适合的 head-major / tile-major layout。

### 设计原则

最好让：

```text
QKV projection output layout
=
Attention input layout
=
Attention output projection preferred layout
```

避免：

```text
transpose
repack
gather
scatter
```

---

# 7. Tier A：Sol-Attn Metal

Sol-Attn 是目前最值得优先尝试的**有损、training-free、dynamic sparse attention**。

---

## 7.1 Sol-Attn 的关键优势

它不是传统：

```text
router
→ top-k
→ index tensor
→ sparse K/V
```

而是：

```text
Q block
  ↓
lightweight block score
  ↓
threshold = mean + τ·std
  ↓
┌─────────────┐
│             │
exact       approximate
│             │
└──────┬──────┘
       ↓
online softmax
```

NVIDIA 原始设计中：

- 每个 query block 动态确定 threshold；
- 重要 block 精确计算；
- 被跳过的 block 使用 pooled K/V correction；
- routing + sparse attention + correction 可以共享在线 softmax 流程；
- 不要求重新训练模型。

---

## 7.2 为什么 Sol-Attn 特别适合 Metal

一个原生 Apple Silicon 实现已经采用：

```text
BQ64
BK64
8×8 simdgroup_matrix
FP32 online softmax
fused Q/K/V block summary
on-the-fly routing
```

并且：

```text
No global route matrix
No Triton
No CUDA
```

这正是我们希望的 execution model。

---

## 7.3 已公开 Apple Silicon 实测

M3 Ultra / BF16 / 56 heads / D=128：

### 约 15K tokens

```text
Native MPS SDPA    308.36 ms
Sol-Attn Metal      59.53 ms

≈ 5.18× attention speedup
```

### 约 100K tokens

```text
Native MPS SDPA    14.898 s
Sol-Attn Metal      2.023 s

≈ 7.36× attention speedup
```

MiniMax-H3 Turbo E2E：

```text
480p / 5s      ≈ 1.435×
480p / 15s     ≈ 1.390×
720p / 5s      ≈ 1.305×
720p / 15s     ≈ 1.678×
```

这个结果非常重要：

> 对超长 video sequence，Metal native dynamic block sparsity 完全有可能有效。

---

# 8. Sol-Attn 的问题：质量风险不是单次 attention error

不同 Apple H3 工程对 Sol-Attn 的质量观察并不完全一致。

一部分测试认为：

- 构图保留；
- 视频正常；
- 没有明显视觉问题。

但另一个独立 H3 Apple Silicon 工程发现：

```text
temporal pulsing
flicker
```

即：

> 单个 attention call 的 rel-L2 看起来并不大，但经过几十个 transformer block 和多个 diffusion steps 后，误差会改变生成 trajectory。

因此不能使用：

```text
Attention output rel-L2
```

作为唯一 acceptance criterion。

---

# 9. 推荐自研：CiderSol

相比直接照搬 Sol-Attn，我们建议设计：

# CiderSol

核心思想：

```text
Dynamic approximation
+
Structured exact safety regions
```

---

## 9.1 Attention region

```text
                    Q tile
                      │
      ┌───────────────┴────────────────┐
      │                                │
 STRUCTURED EXACT                 REMOTE BLOCKS
      │                                │
      ├ Local 3D                       ↓
      ├ text / prompt             Sol threshold
      ├ audio / condition          /        \
      ├ first/last frame         exact     pooled
      └ rotating anchors           \        /
      │                              \      /
      └───────────────────────┬────────────┘
                              ↓
                     online softmax
```

---

## 9.2 为什么比 pure Sol 更适合视频

视频最怕：

- 人物身份漂移；
- 长距离 motion consistency 丢失；
- camera trajectory 出现跳变；
- audio/video 对齐被破坏；
- intermittent flicker。

因此强制：

```text
local temporal neighbors  = exact
conditioning prefix       = exact
selected global anchors   = exact
```

只把：

```text
remote redundant blocks
```

交给动态近似。

### 推荐优先验证

```text
tau = 1.0 / 1.2 / 1.3 / 1.5
```

以及不同 exact band：

```text
local ±1 temporal tile
local ±2 temporal tile
```

---

# 10. Tier A：Structured 3D Block Sparse

如果希望找到**最 Metal-friendly 的 approximate attention**，structured block sparse 很可能比动态 top-k 更有潜力。

---

## 10.1 Video tiling

将：

\[
T\times H\times W
\]

划分成：

```text
4 × 4 × 4
=
64 token
```

每个 Q tile 只访问规则 KV tiles。

---

## 10.2 推荐 pattern

### Local 3D

```text
3 × 3 × 3 neighborhood
```

### Temporal stripe

同一空间区域跨时间：

```text
(t-Δ ... t ... t+Δ)
```

### Spatial stripe

当前时间片内更大的 spatial region。

### Global anchors

例如：

```text
frame 0
frame 8
frame 16
...
```

或者 rotating：

```text
step 0: 0, 8, 16, 24
step 1: 2,10,18,26
step 2: 4,12,20,28
```

---

# 11. STA：Sliding Tile Attention

STA 是非常值得复现到 Metal 的 training-free 方法。

它最大的优势不是“window attention”本身，而是：

> 从一开始就是 tile-by-tile、hardware-aware sliding window。

公开 CUDA 数据中：

- attention 比 FA2 快约 2.8–17×；
- 比 FA3 快约 1.6–10×；
- 在 HunyuanVideo 上 training-free，从 945 s 降到 685 s；
- 论文报告不需要训练即可保持质量。

但注意：

> 这些数字不是 Mac benchmark。

### 为什么仍值得做 Metal

因为 STA 不需要：

```text
top-k
sorting
clustering
permutation
```

核心可以变成：

```text
Q tile
↓
constant neighbor offset table
↓
direct KV traversal
```

这非常适合 Apple GPU。

### 优先级

**P1。**

---

# 12. LVSA：Local Window + Rotating Global Anchors

LVSA 也是非常适合 Metal 的 training-free sparse attention。

核心：

```text
local structured window
+
rotating global anchors
```

相比 pure local window，它可以减少：

- fixed-grid bias；
- 长视频 static/frozen pattern；
- 长距离信息丢失。

公开结果显示：

- Wan 2.1 / HunyuanVideo 上 compute reduction 接近 3×；
- 在 NPU 上也能得到 2.7–3.2× 级别加速；
- 在训练 horizon 内报告 quality-neutral。

虽然没有 Mac 直接数据，但：

> 能在 NVIDIA 之外的 NPU 上保持明显收益，是其硬件可移植性较强的信号。

### Metal 实现建议

不要实现成 mask tensor。

直接：

```text
for local_block:
    exact()

for global_anchor:
    exact()
```

### 优先级

**P1。**

---

# 13. GRAT：Group-Level Shared Sparse Pattern

GRAT 的核心思想非常适合 Metal：

> 一组 contiguous Q tokens 共享同一组 KV region。

例如：

```text
Q Q Q Q
Q Q Q Q   → same KV region
Q Q Q Q
Q Q Q Q
```

而不是：

```text
Q0 → different top-k
Q1 → different top-k
Q2 → different top-k
```

### Metal-friendly 原因

它提高：

- SIMD/workgroup uniformity；
- branch coherence；
- block reuse；
- memory locality。

### 适合 TurboCider 的改造

不要完整复刻论文。

可以直接吸收：

```text
group shared routing
```

作为 CiderSparse 的基础 primitive。

例如：

```text
1 query tile = 1 routing decision
```

而不是：

```text
1 query token = 1 routing decision
```

### 优先级

**P1–P2。**

---

# 14. SVG1：Spatial Head / Temporal Head

Sparse VideoGen 1 观察到 attention head 经常表现成：

```text
Spatial Head
```

或者：

```text
Temporal Head
```

然后通过 online profiling 选择对应 kernel。

公开 CUDA 系统结果中：

```text
CogVideoX     up to 2.28× E2E
HunyuanVideo  up to 2.33× E2E
```

并报告保持生成质量。

### 为什么比 SVG2 更适合 Metal

因为只有少量 pattern：

```text
Spatial
Temporal
```

可以做：

```text
kernel_spatial()
kernel_temporal()
```

甚至通过 layout 让两者都尽量连续。

### 风险

H3 这类模型未必天然适合把 full 3D attention 简单拆成 spatial / temporal。

独立 H3 Apple 工程测试过 axial/factorized replacement，质量误差过大。

所以：

> SVG 思路值得借鉴，但必须先 profile H3 attention map，再决定是否适合该模型。

### 优先级

**P2。**

---

# 15. VSA / FastH3

FastVideo 已经提供原生 MLX 的 H3 VSA runtime：

- tile 64 `(4,4,4)`
- tile 256 `(4,8,8)`
- per-head pooled Q/K score
- top-k routing
- prefix dense
- learned compression branch
- experimental SIMD Metal backend

但是它的 reference MLX backend仍然使用：

```text
grouped gather
+
mx.fast.scaled_dot_product_attention
```

而 Apple benchmark 已经显示 gather 很可能是一个危险点。

FastVideo 同时提供 experimental：

```text
tile=64
head_dim=128
SIMD-group 8×8
```

直接 sparse tile kernel。

### 结论

VSA 值得 benchmark，但：

> 不建议把 gather reference backend 当成 TurboCider 的最终设计。

### 更大的问题

FastH3 VSA 不是简单 inference-time sparse。

推荐 checkpoint 自身包含：

```text
trained compression gate
```

因此它已经是：

```text
model + sparse kernel co-design
```

如果目标是支持 arbitrary H3 checkpoint，Sol/STA/LVSA 更方便。

### 优先级

**P2。**

---

# 16. 不建议优先：SVG2 / SVOO / Clustering Sparse

这一类方法通常：

```text
feature
↓
cluster
↓
sort / permutation
↓
block construction
↓
sparse attention
↓
inverse permutation
```

CUDA 上可能非常有效。

但是 Metal 下可能出现：

```text
FLOPs ↓↓↓
memory movement ↑↑
dispatch ↑
cache locality ↓
```

### 除非满足下面条件，否则不建议第一阶段做

1. clustering 可以高度融合；
2. permutation 可以隐藏在 layout transform 中；
3. sparse pattern 可以跨多个 step 重用；
4. 最终 E2E 至少提升 >10–15%。

否则工程复杂度不划算。

---

# 17. Tier B：Linear / Hybrid Attention

当 sequence 极长时：

\[
O(Nd^2)
\]

linear attention 比：

\[
O(N^2d)
\]

更有吸引力。

Apple benchmark 中普通 Linear Attention 在 N=1024 已经能看到约 1.6× 的相对优势。

但对于已有 dense video model：

> 直接替换 dense attention 通常会明显改变模型。

因此更适合：

```text
local exact softmax
+
remote linear memory
```

---

# 18. VDN / Video DeltaNet 路线

OpenVDN/VDN-H3 走的是：

```text
Local Softmax
+
Long-range linear state
```

这是非常值得研究的 Mac-native architecture。

优势：

```text
Local branch
→ Metal sparse attention

Global branch
→ regular dense GEMM / state update
```

相比 arbitrary sparse：

- 更规则；
- remote context 不完全丢弃；
- 有潜力映射到 GPU + ANE / GPU Neural Accelerator。

但是它不属于纯 kernel replacement：

> 需要模型适配 / post-training。

因此应该作为 TurboCider 第二阶段甚至独立论文路线。

---

# 19. Tier A：Cross-Step Feature Cache

严格来说它不是 attention kernel，但对于 Mac 视频推理，它可能比很多 kernel optimization 更重要。

MiniMax-H3-Swift 的 Apple Silicon 测试中：

```text
Conservative cache
≈ 1.79× – 1.93×
```

更激进可以更高，但会损失高频细节。

### 为什么特别适合 Mac

它不是：

```text
把一个 operation 算快 30%
```

而是：

```text
直接少跑整个 transformer stack
```

因此与硬件无关。

### 与 CiderSol 正交

可以组合：

```text
Step
 ↓
cache admission
 ↓
reuse ? ----------------→ cached residual
 ↓ no
CiderSol
 ↓
Transformer
```

因此：

```text
Cross-step redundancy
+
Token-level sparsity
```

可以同时利用。

---

# 20. 不要忽略 VAE / Temporal / Spatial Approximation

如果目标是“用户感知到的生成速度”，不能只盯 attention。

---

## 20.1 Temporal Fast + RIFE

FastVideo 在 Apple Silicon 上已经提供：

```text
较少 video frames 参与 diffusion
↓
RIFE 插值恢复目标 FPS / frame count
```

优点：

- 极其硬件友好；
- 减少整个 DiT workload；
- 对 preview 很适合。

---

## 20.2 Spatial Fast

```text
低分辨率 diffusion
↓
upsample
```

也是最稳定的系统级加速手段之一。

---

## 20.3 Tiny / Approximate VAE

如果 full VAE decode 占总时间明显：

```text
tiny VAE / approximate VAE
```

可能比继续优化小 kernel 更划算。

但这类方法应主要用于：

```text
Preview Mode
```

不应该默认用于最高质量 final render。

---

# 21. 推荐三个 TurboCider Runtime 档位

## 21.1 Faithful

```text
Optimized Dense FlashAttention
+
Fused QKNorm / RoPE
+
Layout optimization
+
Full frames
+
Full resolution
+
Full VAE
```

目标：

> 尽量接近原始模型数学行为。

---

## 21.2 Balanced

推荐作为默认：

```text
Cross-step cache
+
CiderSol
    exact local
    exact conditioning
    exact global anchors
    approximate remote
+
Full output resolution
+
Full VAE
```

目标：

```text
明显降低 latency
但避免肉眼可见 flicker / identity drift
```

---

## 21.3 Preview

```text
Aggressive cache
+
CiderSol aggressive
+
Temporal Fast + RIFE
+
Spatial Fast
+
Tiny / approximate VAE
```

目标：

> 快速交互式 prompt iteration。

---

# 22. 推荐实现的 Kernel 清单

## P0：必须做

### K0 — DenseMetalAttention

```text
BQ32/BQ64
BK32/BK64
D64/D128
FP32 online softmax
```

---

### K1 — FusedQKNormRoPE

```text
Q/K
↓
RMSNorm
↓
3D/MM-RoPE
↓
attention layout
```

---

### K2 — FusedResidualAdaLN

```text
residual
+
norm
+
modulation
```

减少 unified-memory round trip。

---

### K3 — FusedSwiGLU

尽可能减少：

```text
FC intermediate
```

落到 global memory。

---

# 23. P1：最值得研究

## K4 — SolMetal

复现：

```text
64×64
block summary
mean + tau × std
exact / pooled
online softmax
```

先获得可信 baseline。

---

## K5 — CiderSol

在 SolMetal 上增加：

```text
exact local window
exact prefix
exact anchors
dynamic approximate remote
```

这是最推荐的核心 research kernel。

---

## K6 — CiderBlockSparse3D

支持：

```text
Local3D
TemporalStripe
SpatialStripe
GlobalAnchor
```

不允许 arbitrary token sparse。

---

## K7 — PatternDictionaryAttention

预定义：

```text
P0 local
P1 local + temporal
P2 local + spatial
P3 local + anchor
P4 temporal-heavy
P5 spatial-heavy
P6 hybrid
P7 dense
```

runtime 只选择：

```text
pattern_id
```

而不是生成整个 mask。

---

# 24. P2：后续 benchmark

## K8 — VSA64 Metal

直接 block traversal。

禁止：

```text
gather K/V
```

重点比较：

```text
Sol
vs
VSA
vs
CiderSol
```

---

## K9 — SVG Spatial/Temporal

先分析 attention map。

只有确认某模型确实存在稳定 head-level spatial/temporal specialization 后再做。

---

## K10 — Hybrid Local + Linear

为下一代 Mac-specific sparse model / SFT 准备。

---

# 25. 推荐统一 Kernel Interface

建议所有 attention backend 共享接口：

```cpp
struct AttentionConfig {
    int block_q;
    int block_k;
    int head_dim;

    AttentionMode mode;

    float tau;

    int local_temporal_radius;
    int local_spatial_radius;

    bool exact_prefix;
    bool exact_condition;
    bool exact_global_anchor;

    int anchor_stride;
};
```

runtime 可以动态 dispatch：

```text
short sequence
    → Dense

medium
    → Dense / structured sparse

long
    → CiderSol / block sparse
```

---

# 26. Adaptive Dispatch 非常重要

不要对所有 N 使用同一个 kernel。

建议：

```text
if N < threshold_1:
    dense

elif N < threshold_2:
    structured sparse

else:
    CiderSol
```

因为 sparse routing 本身有 overhead。

Apple M4 benchmark 已经表明：

```text
BlockSparse 在短 N 可能比 Dense 慢
```

但在长 N 开始快速占优。

而视频恰好属于最有利于 sparse 的超长 context。

---

# 27. M4 Max 推荐的第一轮 Tile Sweep

目标机器建议首先 sweep：

```text
BQ ∈ {32,64}
BK ∈ {32,64}
D  ∈ {64,128}
```

对于 3D video：

```text
tile:
2×4×8 = 64
4×4×4 = 64
4×8×8 = 256
```

测试：

```text
memory bandwidth
GPU occupancy
threadgroup memory
SIMD utilization
latency
```

不要假设 CUDA 的最佳 tile 就是 Metal 的最佳 tile。

---

# 28. Sparse Degree 不应该任意变化

不要：

```text
Q0 → 7 blocks
Q1 → 29 blocks
Q2 → 13 blocks
Q3 → 46 blocks
```

这会造成严重 workload imbalance。

建议 degree bucket：

```text
K ∈ {8,16,32,64}
```

或者更简单：

```text
low
medium
high
dense
```

每批 query tile 运行相同 specialization。

---

# 29. 质量验证：绝对不能只测 Rel-L2

视频 diffusion 的误差会累积。

必须同时评估：

## 29.1 Attention-level

```text
relative L2
cosine similarity
max absolute error
```

仅用于 debugging。

---

## 29.2 DiT block-level

比较：

```text
block residual
```

和 dense teacher 的：

```text
cosine
relative L2
```

---

## 29.3 Diffusion-step level

比较 latent trajectory：

```text
x_t
```

是否快速偏离 dense baseline。

---

## 29.4 Final video

必须包括：

### Spatial quality

- VBench / appropriate perceptual metrics
- LPIPS / DISTS 类感知差异
- sharpness / high-frequency energy

### Temporal quality

- flicker
- temporal LPIPS
- optical-flow warped frame residual
- motion continuity
- camera trajectory stability

### Identity / semantics

- subject consistency
- prompt adherence
- text rendering
- object count
- face consistency

### Audio-video 模型

还必须测试：

```text
lip synchronization
audio continuity
audio/video semantic alignment
```

---

# 30. 建议建立 Stress Set，而不是只跑普通 Prompt

至少包含：

## Dialogue

```text
close-up face
visible mouth
continuous speech
```

测试 lip-sync 和 facial stability。

## Fast Motion

```text
fast camera pan
running subject
object crossing frame
```

测试 temporal attention。

## Fine Texture

```text
hair
leaves
fabric
water
fine text
```

测试近似导致的 smoothing。

## Long-range Identity

人物短暂离开视野再回来。

## Multi-subject

3–5 人同时存在并运动。

## Camera Motion

orbit / dolly / tracking / zoom。

---

# 31. Benchmark 必须分四个层级

## Level 1 — Microkernel

```text
Q/K/V random tensor
fixed N/H/D
```

测：

```text
µs / ms
effective TFLOPS
memory bandwidth
occupancy
```

---

## Level 2 — Attention Layer

真实 H3/Wan tensor shape。

测：

```text
routing
attention
projection
```

完整时间。

---

## Level 3 — DiT Step

真实 50 block / full model step。

---

## Level 4 — End-to-End

必须包含：

```text
encoder
denoise
VAE decode
audio
mux
```

否则容易把 kernel-level 5× 写成“视频生成 5×”。

---

# 32. 第一批应该做的对照实验

建议固定：

```text
same prompt
same seed
same checkpoint
same steps
same precision
same resolution
same frame count
```

比较：

| Backend | Approximation |
|---|---|
| Dense Metal | none |
| PyTorch / MLX Dense | none |
| SolMetal | dynamic |
| CiderSol | dynamic + exact structure |
| STA-like | structured |
| LVSA-like | structured + anchors |
| VSA Metal | top-k |
| Cache + Dense | cross-step |
| Cache + CiderSol | combined |

---

# 33. 我们真正关心的 Pareto Frontier

不要单独排名 latency。

横轴：

```text
E2E generation time
```

纵轴：

```text
quality score
```

每一个 configuration：

```text
tau
local radius
anchor stride
sparsity
cache threshold
```

都是一个点。

最终找：

```text
Pareto frontier
```

例如：

```text
Faithful
Balanced
Fast
Preview
```

四个可用档位。

---

# 34. 推荐的开发顺序

## Phase 0 — Profiling

首先确定：

```text
Attention %
MLP %
VAE %
encoder %
memory movement %
```

否则不要盲目写 kernel。

---

## Phase 1 — Dense Foundation

实现：

```text
Dense FlashAttention
QKNorm+RoPE
layout
MLP fusion
modulation fusion
```

建立 strong baseline。

---

## Phase 2 — Sol Reproduction

实现：

```text
SolMetal
```

目标：

- 重现 Metal Sol-Attn speedup；
- 在自己的 M4 Max 上得到真实数据；
- 建立 temporal-quality stress set。

---

## Phase 3 — CiderSol

加入：

```text
local exact
conditioning exact
global anchors exact
remote approximation
```

测试能否减少 Sol 的 temporal artifacts。

---

## Phase 4 — Pure Structured Sparse

实现：

```text
STA-like
LVSA-like
GRAT group pattern
```

目标：

> 测试更低 theoretical sparsity 是否因为更规则，最终比高动态 sparsity 更快。

这是非常有研究价值的 hypothesis：

> **Less but structured sparsity can outperform higher dynamic sparsity on Apple Silicon.**

---

## Phase 5 — Hybrid Model-Kernel Co-design

如果 training-free 到达质量瓶颈：

```text
Dense teacher
↓
Mac-friendly sparse student
```

只训练：

```text
router
gate
QKVO LoRA
compression branch
```

约束模型只使用 Metal-friendly primitive：

```text
tile ∈ {32,64}
degree ∈ {8,16,32,64}
pattern ∈ small dictionary
```

这时就从：

```text
kernel adaptation
```

进入：

```text
hardware-aware model adaptation
```

---

# 35. 未来 GPU + ANE 路线

不建议把 arbitrary sparse attention 直接塞到 ANE。

更合理的是：

```text
GPU:
local / sparse / softmax branch

ANE or regular accelerator:
dense regular projection
global linear-memory branch
```

例如未来：

```text
Hybrid Attention
   │
   ├ GPU: exact sparse local
   │
   └ ANE: linear global state
```

这需要：

- zero-copy/shared tensor path；
- 足够大的并行 branch；
- 避免每层 GPU↔ANE synchronization overhead。

因此不应该作为第一版 kernel。

---

# 36. 最终推荐优先级

## S Tier — 立即做

1. Dense tiled Metal FlashAttention
2. QKNorm + RoPE fusion
3. QKV/layout optimization
4. MLP / SwiGLU fusion
5. Residual + AdaLN/modulation fusion

这些是所有后续方案的基础。

---

## A Tier — 最值得做

1. **Sol-Attn Metal**
2. **CiderSol**
3. **Structured 3D Block Sparse**
4. **STA-like**
5. **LVSA-like**
6. **Cross-step cache + sparse composition**

其中最核心：

```text
CiderSol
```

---

## B Tier — 选择性做

1. GRAT-style group routing
2. SVG1 spatial/temporal routing
3. FastH3 VSA direct sparse Metal backend
4. local-softmax + linear global memory

---

## C Tier — 暂不优先

1. SVG2 semantic clustering
2. SVOO co-clustering
3. arbitrary token top-k
4. gather-based sparse
5. runtime heavy permutation/reorder

---

# 37. 推荐最终架构

```text
                       TurboCider Runtime
                              │
                              ▼
                      Sequence Profiler
                              │
              ┌───────────────┼────────────────┐
              │               │                │
           Short N         Medium N         Long N
              │               │                │
              ▼               ▼                ▼
       Dense Flash      Structured      CiderSol
        Attention        Sparse          Attention
              │               │                │
              └───────────────┴────────────────┘
                              │
                 QKNorm / RoPE / Layout Fusion
                              │
                         DiT Blocks
                              │
                  Cross-Step Cache Layer
                              │
                     Full / Fast Decoder
```

对于用户：

```text
Faithful
Balanced
Fast
Preview
```

只是不同策略 preset。

---

# 38. 最关键的研究问题

整个项目最值得回答的不是：

> “我们能不能把 CUDA sparse attention port 到 Metal？”

而是：

> **什么样的 attention sparsity 才是 Apple Silicon 原生高效的 sparsity？**

可以进一步形成几个明确的研究问题：

### RQ1

相同 FLOP reduction 下：

```text
Structured sparsity
vs
Dynamic top-k sparsity
```

谁在 Metal 上 wall-clock 更快？

### RQ2

是否存在一个：

```text
70–80% structured sparse
```

比：

```text
90–95% arbitrary sparse
```

更快的区域？

### RQ3

什么样的 exact safety region 可以最有效地避免：

```text
flicker
identity drift
motion instability
```

？

### RQ4

Sparse pattern 是否可以跨 diffusion steps 重用？

### RQ5

是否可以用极小规模 LoRA / SFT，让模型主动适应：

```text
Metal-friendly pattern dictionary
```

？

如果 RQ2–RQ5 能得到正结果，就不只是一个工程优化，而会成为一个比较完整的：

> **Apple-Silicon-aware video model–kernel co-design**

研究方向。

---

# 39. 当前最推荐的第一版 Prototype

第一版不需要做得复杂。

直接实现：

```text
CiderSol-v0
```

### 输入

```text
Q/K/V
[B,H,N,D]
```

### tile

```text
BQ = 64
BK = 64
D  = 128
```

### Exact

```text
prefix
local ±1 temporal tile
first frame
last frame
periodic anchor
```

### Remote

```text
pooled Q/K score
↓
mean + tau * std
↓
exact / pooled correction
```

### 内部

```text
FP32 online softmax
```

### 明确禁止

```text
top-k tensor
full route matrix
gathered K tensor
gathered V tensor
sorting
permutation
scatter
```

---

# 40. 成功标准

第一阶段不要追求论文式夸张数字。

建议实际 acceptance gate：

### Kernel

```text
≥ 2×
```

相对 optimized dense attention 才值得继续。

### DiT step

```text
≥ 15–20%
```

才值得进入 full quality campaign。

### End-to-end

Balanced 模式：

```text
目标 1.5×+
```

且：

- 不出现持续 flicker；
- 不出现明显 identity drift；
- audio sync 不显著下降；
- fine detail degradation 可控。

如果只能：

```text
1–3%
```

E2E gain，却增加新的质量 failure mode：

> 不值得上线。

---

# 41. References / 当前值得持续跟踪的实现

## [R1] Sol-H3 / Sol-Attn — NVIDIA Research

Sol-Attn 的核心机制、64-token block、query-dependent threshold、pooled correction 和 fused online-softmax 设计。

https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/

---

## [R2] ComfyUI-SolAttn-MPS

Apple Silicon 原生 Metal Sol-Attn 实现及 M3 Ultra benchmark。

https://github.com/yshenaw/ComfyUI-SolAttn-MPS

---

## [R3] AttnBench

Apple M4 上 sparse / block sparse / linear attention benchmark，特别值得参考 gather overhead。

https://github.com/dirmacs/attn-bench

---

## [R4] Apple WWDC26 — Custom ML operations with Metal tensors

Apple 官方 TensorOps、quantized tensors、custom FlashAttention kernel 实现方向。

https://developer.apple.com/videos/play/wwdc2026/330/

---

## [R5] Sliding Tile Attention

Training-free、hardware-aware 2D/3D tiled sparse attention。

https://arxiv.org/abs/2502.04507

---

## [R6] LVSA

Structured local window + rotating global anchor。

https://arxiv.org/abs/2605.31057

---

## [R7] Sparse VideoGen

Spatial / Temporal head sparsity 与硬件协同。

https://arxiv.org/abs/2502.01776

---

## [R8] GRAT

Group-shared structured sparse region。

https://arxiv.org/abs/2505.14687

---

## [R9] FastVideo MiniMax-H3 VSA MLX

当前 MLX VSA runtime：tile 64/256、pooled Q/K top-k、reference gather backend 与 experimental SIMD backend。

https://haoailab.com/FastVideo/api/fastvideo/mlx_runtime/minimax_h3_vsa/

---

## [R10] MiniMax-H3-Swift Performance Guide

非常重要的反例集合：Sol-Attn temporal flicker、axial attention failure、cache 的实际收益，以及“kernel 快不等于 E2E 快”的工程经验。

https://github.com/loading-awesome/MiniMax-H3-Swift/blob/main/docs/PERFORMANCE_GUIDE.md

---

## [R11] FastVideo FastH3 Apple Silicon

FastH3 / FastMetal 在 Apple Silicon 的最新 runtime 和系统优化路线。

https://haoailab.com/blogs/fasth3-local/

---

# 42. 一句话结论

TurboCider 最值得做的不是：

```text
CUDA sparse kernel → Metal translation
```

而应该是：

```text
Apple GPU execution model
          ↓
设计规则的 block/tile primitive
          ↓
设计 sparsity
          ↓
必要时再让模型适应这种 sparsity
```

最终目标应该是：

> **让模型的近似方式天然符合 Metal 的执行方式，而不是让 Metal 被迫执行一个为 CUDA 设计的 sparsity pattern。**

当前最值得优先推进的主线：

```text
Optimized Dense Metal
        ↓
Sol-Attn Metal
        ↓
CiderSol
        ↓
Structured STA/LVSA-style sparse
        ↓
Sparse-aware lightweight SFT
```

这条路线同时兼顾：

```text
速度
+
质量
+
工程可实现性
+
研究 novelty
```

