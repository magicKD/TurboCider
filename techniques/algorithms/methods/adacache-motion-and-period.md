# AdaCache：从 residual 变化选重算间隔，MoReg 如何改变实际计算

<!-- knowledge
id: technique-algorithms-methods-adacache-motion-and-period
type: method
stage: dit
operator: attention, sampling, layout
backend: cpu-reference
device: cpu
symptom: stale-cache, precision-loss, wrong-layout
semantics: approximate, contract-only
representation: fp16, fp32
boundary: request, producer-consumer
api: reference-only
-->

[算法入口](../README.md) · [PAB](pab.md) · [MagCache](magcache-calibration-and-consumer.md) · [实验入口](../../examples/README.md) · [固定来源 S52](../sources.md#s52)

研究日期：2026-10-17。AdaCache 的关键不是降低 denoiser 调用次数，而是用上一次完整计算的 residual 变化决定后续多少次调用重用 block 分支。固定 Open-Sora adapter 还用 feature motion 及其变化收紧间隔。它适合研究 Apple 上的粗粒度 full/reuse 调度；**本轮只有实际源码的 tiny CPU 调用链，没有真实 STDiT 权重、视频、GPU 时间或质量资格**。

## 固定来源和运行深度

作者仓库：[AdaCache-DiT/AdaCache，commit `2c6d3ca`](https://github.com/AdaCache-DiT/AdaCache/tree/2c6d3ca0320afd1f3ac41ac52673c79c8f6e00a4)。已读 README、Apache-2.0 许可、baseline/AdaCache/MoReg 三份配方、video shell 与 inference consumer、STDiT3 root/block、RFlow sampler、timestep transform 和 modulation 选段，11 个文件哈希。[论文入口](https://arxiv.org/abs/2411.02397)只作导航，本轮不宣称论文逐式审计。README 的 A100 运行说明不是 Apple 基准。

CPU probe 核验本地 Git blob 后选择原方法，不导入 Open-Sora/ColossalAI/einops，不执行作者 main/模型构造器、训练、checkpoint、SP 或 CUDA。明确替身包括：14 对 spatial/temporal blocks（保留全局 layer13 选择点）、C2/unit patches、3×1×2 tiny latent、identity norm/zero position、简单 affine attention/cross/MLP、Torch reshape facade、direct checkpoint 调用和微型 text fixture。不是生产 STDiT 的 depth28/hidden1152，也不说明真实 attention 在这些 shape 上的能力。

运行的是原 `RFLOW.sample → STDiT3.forward → STDiT3Block.forward/compute_next_step → CFG/Euler`，不是只抄一个 gate 公式。原 RFlow 保留 cond-first packed CFG、sigma-channel split、非线性 timestep grid 和更新顺序。51-frame/512×512 metadata 仅用于固定 timestep transform；tiny latent 不是实际 VAE 编码或原模型几何复现。

 含 8 个合成请求 / 800 次 denoiser 调用 / 38400 个完整预测 scalar、142 次选中层 metric 调用、19 个独立 metric 控制、1 个 current-mask/gate 对照及 1 个注入 prefix 失败。标准库整理器 逐元素独立 FP32 重放每次预测、输入反馈、CFG/Euler endpoint，全部相等；另用 FP64 复核 metric/MoReg、codebook 与各层时钟。它不导入 Torch、不执行上游、不读取权重或 GPU。

这里的“相等”是全部有限 FP32 数值逐项相等，不另行授予 signed-zero bit-pattern 资格；文件/来源身份则分别有 SHA-256 绑定。

## 真正缓存什么，`cache_res` 不表示只省哪一层

每个 block 保存两份完整 residual：

- `st_attn_cache` 是 attention 输出经过当前 gate/drop_path 的结果。
- `ca_mlp_cache` 是 cross-attention 与 gated MLP residual 的和。

hit 用当前 hidden 加这两份旧 residual，跳过该 block 的 attention、cross-attention 和 MLP。它不是缓存 softmax map，也不是只复用 temporal attention；gate 之前还是之后缓存，决定了当前 modulation 是否还能重算。与 [ToCa](token-feature-caching.md) 的 gate 前 feature cache 不同。

`cache_res='t-attn' / 's-attn' / 'ca-mlp'` 选择**用哪个 residual 测量**，不限制只缓存哪种模块。source root 将同一 do_cache/config 装到所有 spatial/temporal blocks，所有 blocks 消费同一个当前 `attn_cache_rate`；MLP 也用这个 rate，不由另一个独立 MLP 周期控制。

## 调度先完整测量，再把新间隔用于下一次调用

在选中 global layer、对应 module 类型且已有旧 cache 时，计算

`d = ||r_old-r_new||₁ / ||r_old||₁ / previous_rate`。

原 norm 覆盖 batch、tokens、channels。多个选中 residual 更新一个按计数平均的 raw d；MoReg 在该平均之后作用，最后一个选中点发布 `new_attn_cache_rate`。root 在全部 blocks 结束后才把新值改为下一次调用的 period；本次各层继续使用旧 rate，避免半个模型在新旧周期上运行。

默认 `cache_loc=[13]`、`t-attn` 只有 temporal13 提议；改成 `ca-mlp` 时 spatial13 和 temporal13 都可提议，第二个使用累计的 raw 平均。多层配置下不是“各层最后独立选择自己的周期”，也不是平均所有层 MoReg 后再统一选档。

原 100 步 codebook：

| raw/MoReg 调整后指标 | 下一重算间隔 |
|---|---:|
| `< .03` | 12 |
| `< .05` | 10 |
| `< .07` | 8 |
| `< .09` | 6 |
| `< .11` | 4 |
| 其它 | 3 |

这是顺序比较，不是字典查近邻或概率。第六个 key `1.0` 不参与比较，最终取最后一个 **value**；改成 `1e9` 不改变结果。大误差甚至 NaN/Inf 落到最后一档 3，**不是 full/fail-open**。边界 `.03125` 的二进制精确控制在相同首阈值处进第二档，证明严格 `<`；乱序字典、短 codebook 均有显式诊断，不假定它们受校验。

每个 block 内部 `fwd_id` 按调用递增；full 条件是 `fwd_id == prev_compute_step + rate` 或强制步 `[1,N−1,N]`。首步不计算 d；初始 rate1 使第二步也 full。最后两步可能提前 full，但 d 和 motion gradient 仍除以传入的 configured rate，而非实际离上次 full 的步距。实验末尾 99/100 的实际间隔都是 1，传入 rate 却是 10/12。迁移时不要悄悄换公式，再声称复现了原 gate。

## MoReg 测的是特征运动，不是光流模型

将 flattened token 视为 T 个连续 frame、每帧 S 个位置。对 stride k，计算 `||r[t+k]-r[t]||₁ / (||r[t+k]||₁+||r[t]||₁)`，跨配置 strides 平均；默认只 stride1。没有 optical-flow 网络，也不能随意把 channel 或 text tokens 当成时间轴。

窗口内先做 `(motion/.385)^8 / 1`，窗口外设 base=1；再加 `abs(10*(base-prev_base)/previous_rate)`，乘到 d 后选档。stored `prev_moreg` 保存的是 base，不是加 gradient 后的总 factor。第四个 hyperparameter `moreg_hyp[3]` 在选中实现中未使用；改成 999 的原函数对照不变。

静态非零 feature 得到 base0，但首次从 prev1 进入窗口仍有 gradient 罚项；退出窗口设 base1 时也可能出现罚项，不是窗口外永远 factor1。直接运动反例中 base≈2071.64，多 stride 后≈8.09；这些是明确 tiny vectors，不是视频 motion score 或模型校准范围。空 frame-pair / 零范数没有 epsilon，可产生 NaN。

## 实际合成消费链：省 block，不省 NFE

相同初值、packed CFG7、同一 100-step timestep grid，原 sampler 的反馈轨迹如下。相对误差对完整合成请求的 full endpoint 计算，**不是 RGB/视频质量**。

| 路径 | denoiser 调用 | full-body 调用 | 每种模块调用数 | endpoint 相对 L2 |
|---|---:|---:|---:|---:|
| full | 100 | 100 | 2800 | 0 |
| AdaCache，temporal metric | 100 | 12 | 336 | 8.586% |
| AdaCache + 默认 MoReg 窗口 | 100 | 25 | 700 | 2.352% |
| spatial metric | 100 | 12 | 336 | 8.586% |
| cross+MLP metric | 100 | 12 | 336 | 7.536% |
| 自定义 MoReg 末步仍 active | 100 | 26 | 728 | 2.206% |

“每种模块”分别指 attention、cross、MLP，不能把 2800 当 CUDA kernel 数。AdaCache full 步为 `[1,2,14,26,38,50,62,74,86,98,99,100]`；MoReg 改成 25 次 full，确实多分配计算并降低这个 affine 反馈端点误差。没有计时，不能把 `100/12` 写成 8.33× 加速；hit 仍有 root embedding、modulation准备、residual adds、最终投影、CFG/Euler 等工作，metric 另含 norms、motion、scalar/host 分支。

最后两步 full 并没有把 AdaCache 的轨迹误差归零，因为此前 Euler 状态已不同；最后一次精算只对当前近似输入精算，不是对原 full 轨迹 rewind。相同 full 次数也不足以选赢家：cross+MLP 与 temporal 的 full 位置/误差可不同。

## CFG pooling 的负控：100% 分支变化被压到约 0.1%

原 RFlow 明确 `y=[cond,null]`、`z=[z,z]`，预测按 batch 分成 cond/uncond，再做 `uncond + cfg*(cond-uncond)`。只有一次 model forward，不应把 block 内部 clock 乘 2。cache 保存两路独立行，但 interval metric 是合并 norm，所以控制策略仍共享。

构造 cond cache/res 都是1000、uncond 从1变2：整体 d=1/1001≈0.000999，选 period12；单独看 uncond，d=1，选 period3。这个 source 函数控制证明 norm-weighted pooling 可以掩盖小分支的相对漂移，不证明真实模型有该 norm 比例。CFG 还可能放大 cond/uncond 差分误差；统一内存不能修复这个数学前提。

多 video batch 也会共享同一指标/周期；默认 batch1 仍因 packed CFG 有两路。Apple 移植可以研究 max-of-branches/per-sample budget，但那是新策略，应以原 pooling、各路和 guided output 三条误差线对照，不宣称自动更好或与原版等价。

## 状态、当前 mask 与失败的边界

正常完成 N 步时，source 清 clocks、两 cache、recomputed lists、previous_compute 和 root period；`prev_moreg` 本身没有在 reset 中归一。默认窗口(10,90)在末尾强制 full 时将 base 恢复1，所以两个相同完整默认请求输出、周期和末态一致，**不构成正常完成状态泄漏反例**。

自定义窗口(1,100)末态 base≈3.25939，第二请求第一次测量看到这个保留值，而非1。但这个固定 coarse codebook 仍选择同一批 full 步，两个输出也相同；不能把“状态不同”夸大成“本次输出已污染”。请求身份、CFG布局、shape、条件切换与中断仍应在移植控制器里显式定义，不能只依赖成功走满 N 步。

注入第二个 spatial block 的 attention 失败，实际 root clock 仍0，但前3个 blocks 已到1，其余保持0；它说明异常边界没有全模型 rollback，不说明真实设备产生这种故障或已经有可达的正常服务事故。

current-mask/gate 控制在已有 cache 后改变当前 gate 和 temporal mask，hit 的 attention/cross/MLP 调用都是0；旧 **gated** residual 没有套新 mask/gate，输出与同路 full 不同。这是近似边界诊断，不把源码写成“原算法只缓存 gate 前值”。source sampler 另有 reference-frame noise/update 保护；本轮完整 sampler cohort未启用该 mask-noise 路径，因此不授予 I2V/reference-frame 端到端正确性。

19 个 direct controls保留 zero cache→Inf、zero/zero→NaN、FP16 L1 norm overflow→NaN、空 frame-pair→NaN，均回到 rate3而非拒绝/full；公开 null 用 kind/path ledger 区分。FP16 operands 在完整 FP64 norm 下 ratio 为0.5，但 source half norm 已溢出；不把“标量控制便宜”当作稳定性证明。

## 对 Apple Silicon 的可迁移价值

与 MagCache 的离线 magnitude curve 不同，AdaCache 在选中 full 步用当前残差与缓存测在线变化；与 PAB 固定周期不同，它改变下一段间隔；与 ToCa token cache 不同，它保留全尺寸 residual，粗粒度跳过整个分支。代价是 blind interval 内不观察当前 residual，且缓存/运动统计/host scalar 仍有成本。

MLX/MPS 原型可以保留固定 full/hit 图，host 只管理 period；无需先移植 CUDA sparse kernel。ANE/Core ML 可考虑有限 period 桶和固定图外部控制，但不能因 scalar gate 小就认定 full/cache 交接没有成本。建议按 request/branch/stage/shape/calibration 身份持有 cache；明确非有限 full/refuse、实际步距与 sampler 坐标、末步保护和失败 cleanup，并记录这些相对原版的改动。

不要直接搬 layer13、100-step codebook 或 `.385^8` 参数到 Qwen/FLUX/H3；flatten layout、调制边界和模型分布不同。优先相同 quality budget 对照 PAB/TeaCache/MagCache/AdaCache，计入完整 controller、所有模型阶段和 cache驻留压力；本轮只为源码/设计学习，不提供项目速度排行榜。

## 重放与可追溯性

原始正式包为 `adacache-cpu-v2`，owned child 已确认 exit0；v1 是本地不同 affine recipe/冗余 trace 的开发观察，不混入本轮计数。公开只含源码身份、原创 tiny tensors、完整调度与数值/终态，不含真实模型 activation、权重或媒体。没有 GPU 工作；后续性能实验仍需[重新确认 GPU 空闲](../evaluation.md)。
