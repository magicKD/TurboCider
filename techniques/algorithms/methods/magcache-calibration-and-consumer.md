# MagCache：校准曲线、真实 hook 时钟与分支边界

<!-- knowledge
id: technique-algorithms-methods-magcache-calibration-and-consumer
type: method
stage: dit
operator: sampling, layout
backend: cpu-reference
device: cpu
symptom: stale-cache, precision-loss, wrong-layout
semantics: approximate, contract-only
representation: fp16, fp32
boundary: request, producer-consumer
api: reference-only
-->

[算法入口](../README.md) · [SeaCache / MagCache](spectral-magnitude-cache.md) · [MLX TeaCache lifecycle](../case-studies/mlx-teacache-lifecycle.md) · [来源 S50](../sources.md#s50) / [S51](../sources.md#s51)

研究日期：2026-10-17。结论是 **scalar gate 容易迁移，校准和 consumer 合同不能省略**。本轮运行作者 FLUX/Qwen/Wan 的原 forward/calibration 声明，以及 Diffusers 的真实 HookRegistry、MagCache、CacheMixin 与 cleanup 方法；模型部分是明确的 CPU affine 替身。没有新 GPU 时间、真实模型、图像或视频质量结论，也没有改产品缓存实现。

## 固定实现与实验边界

作者 [MagCache](https://github.com/Zehong-Ma/MagCache/tree/df81cb181776c2c61477c08e1d21f87fda1cd938) 固定 `df81cb1`，Diffusers 固定 `d42ebf9`。本轮账本新增 14 个文件哈希，合计 52 条来源 / 214 个文件条目；不是 214 个互不重复文件，也不是全仓库审计。三个作者 README 的速度是作者 A800 测量，不能当成本机 Apple Silicon 结果。

CPU probe 先校验本地 Git blob 的 SHA，再 AST 选择声明。不执行 imports、顶层 pipeline loader、权重下载或作者文件写入。Qwen 原 initializer 只修改每次新建的自有 Toy class；embedding 显式 clone，避免 identity 替身把源 `+=` 误变成调用者原始输入 mutation。Wan 使用 no-op AMP facade、零 sinusoidal/time fixture 和内存 `save_json` sink；不是 CUDA autocast、Wan patch embedding 或视频模型复现。Diffusers 使用明确的 toy metadata registry，不执行生产架构注册器；保留真实 registry 的 wrapper 嵌套、上下文、移除和重置行为，仅移除 compiler-disable 装饰器和 CacheMixin 内的相对 imports、注入依赖。

 含 28 个作者情景 / 219 次 forward 调用、21 个 Diffusers 情景 / 228 次调用、10 个刻意 shape/失败诊断。调用数不是 GPU kernel/NFE 或作者实际模型 block 数。标准库整理器 不导入 Torch/NumPy，不运行上游；独立重放 100 行正常作者推理、全部 228 行 Diffusers 的输出/gate/内部时钟、3969 组插值索引，另核对校准与明确失败观察。219 行不全是同一种 oracle 验证，不能把日志断言说成完整 diffusion 数学重放。

## 校准并不免费，也不是每步在线精算

在完整 block 链边界取 `r_t = body(x_t)-x_t`，按 token 计算 `||r_t|| / ||r_prev||` 再求均值。推理时查该先验曲线，累计 `R ← R·ratio[t]`、`E ← E+|1-R|`，在 warmup、E、K 等条件下复用旧 residual。**输出是 `x_t+r_cached`，不是 `x_t+R·r_cached`**；实验的 `ratio-not-output-scale` 保留这个区别。

所有校准调用都执行两个 fixture block，外层 norm/projection/head 仍运行，另有 norm/std/cosine、host scalar 和打印/导出成本。作者 FLUX 输出 N−1 个 ratios，示例前补 1；Qwen/Wan 隔两次调用比较同 parity，输出 2N−2 个 ratios，Qwen 初始化前补 `[1,1]`。实验将实际旋转向量校准得到的曲线补齐，传给新的推理 fixture，并非直接假定曲线。

复用一次校准的粗略摊销条件是 `Q > T_cal/(T_full-T_cached)`，前提是完整请求速度确实改善，而且该曲线对 Q 个请求仍有效。本轮没有测这些时间，不能据 full-count 计算加速比。checkpoint、scheduler/timestep grid、steps、CFG 模式、LoRA、prompt/domain 和 residual 边界都属于校准身份；nearest 插值只改变索引，不证明跨身份有效。

## 原版不同 adapter 不是同一个状态机

| 接缝 | 作者 FLUX | 作者 Qwen | 作者 Wan2.1 | 固定 Diffusers MagCache |
|---|---|---|---|---|
| warmup 计数 | `int(frac*N+.5)` | `int(frac*2N)` | `int(frac*2N)` | `int(frac*N+.5)`，每 context |
| E 阈值 | `<=` | `<` | `<` | `<=`，并检查已有 residual |
| 分支历史 | 单路 | `cnt%2` 两槽 | `cnt%2` 两槽 | context.name 独立 state |
| 特殊刷新 | 映射到 28 步 index11 强制 full | 无此守卫 | 无此守卫 | 无此守卫 |
| 完成时推理 reset | cnt + accumulators；保留 residual | 只 cnt；保留 accumulators/cache | cnt + accumulators；保留 cache | 每 context 达 N 清 residual/accumulators；正常 pipeline 再清 manager |
| 校准除法 | 原 dtype，无 epsilon | 原 dtype，无 epsilon | 原 dtype，无 epsilon | residual 减法后 `.float()`；分母 `+1e-8` |

二进制精确 ratio1.125 / threshold0.125 的第二次可缓存调用，FLUX/Diffusers 命中，Qwen/Wan full；这不是一般浮点比较噪声。FLUX 28 步、K100、只保留首步时 full 在 `[0,11]`，末步仍可 hit；双/三块 Diffusers 对照只有 `[0]` full，没有 index11 或 final-full 规则。不能把原版 FLUX 的额外 guard 当论文里所有模型的规则。

第二请求 stress 把 K 设大，明确观察 Qwen 最后留下的非零 skip 步数；下一请求 warmup full 不主动清除这些 accumulators。FLUX/Wan 清零。第二次校准则 FLUX 诊断列表重新开始，Qwen/Wan 列表累积；Wan 内存 sink 记录两组三次导出。作者单脚本 demo 不是多租户服务，这些 stress 不宣称日常单请求失败。

## 负结果一：幅值预算 0，方向误差仍可达 2

校准 residual `[1,0] → [0,1] → [-1,0] → [0,-1]`，每个 token 的范数都为 1，因此校准 ratios 全为 1，在线 E 始终为 0。warmup 后复用 `[1,0]`，误差分别为 `√2` 和 `2`（参考范数为 1，所以也是相对 L2）。K 超界 full 后恢复。两 parity adapter 把同样序列各重复两路，现象相同。

作者额外记录的 cosine distance 为 1、std 为 0，但这两个指标没有进入本次 inference gate。这个原创向量反例只否定“幅值预算是向量/RGB 误差上界”的推断，不否定作者在实际模型上观察到的经验相关性，也不声称真实 Qwen 误差必然等于 2。

## 负结果二：实际 registry 的单块缓存时钟

固定 `apply_mag_cache` 明确支持单块，顺序是先 Tail 再 Head；真实 HookRegistry 将后注册的 Head 包在外层。full 会调用内层 Tail，hit 则 Head 直接返回，因此 Tail 的 `_advance_step` 不运行。

K2、8 步、ratios 全 1 时，单块实际内部 step 序列为 `[1,1,1,2,2,2,3,3]`；full 调用是 `[0,3,6]`。双/三块的 head 与 tail 分开，即使中间 skip，tail 仍运行，内部序列 `[1,2,3,4,5,6,7,0]`。单块校准始终 full，时钟反而正常；这说明“校准成功”不能覆盖 hit wrapper 的正确性。

这是指定 commit、toy block 但真实安装/执行链上的功能反例，不宣称所有版本、所有模型都有单块布局，也没有就地修复上游。request 完成后原 pipeline cleanup 能清掉残留 state，但不能倒推修复此前重复读取的 ratio 索引。

## CFG 身份和正常 cleanup 的正控

StateManager 按 `CacheContext.name` 分开，MagCache 的预算/计步只读自身 `state.step_index`，不消费 context 的 step/sigma/num_steps。命名 cond/uncond 的测试保留两份 residual；刻意共享同一个 name 会混用。重复传 step0/num_steps999 的诊断不会改变内部 clock；这是无效调用者 metadata stress，不是 Qwen 正常 consumer 失败。

固定 Qwen pipeline 真实调用点显式使用 `cache_context("cond")` / `("uncond")`，正常两路隔离；**不要因为作者代码变量叫 con/ucon 就反推作者未固定依赖的 runtime 顺序**。作者两槽只是调用 parity，改变 CFG 开关、额外调用或分支顺序前要重新绑定。Diffusers 残差历史分开，但同一个 config 的 mag_ratios 仍共享，不等于自动得到每路独立校准曲线。

CacheMixin 的第二次 `enable_cache` 会拒绝，需先 disable；本轮执行该公开 guard，以及 disable→enable。不能把内部 apply 函数的直接重装风险说成公开 enable 默认行为。`cache_context` finally 只清上下文；正常 Qwen 成功结束调用 `maybe_free_model_hooks`，该方法在 offload 早退之前重置 component cache。本轮执行原 cleanup 方法，即使没有 offload，也把全部 context state evict。因此 `.reset()` 没清 head input **不构成正常完成 pipeline 泄漏证据**；没有测物理内存释放。

## 插值与数值/失败边界

原 NumPy FLUX/Wan 用 float64 grid，Qwen 用 linspace；Diffusers 强制 float32 grid。长度 2…64 的全部 3969 组原函数调用中，65 组索引不一致，完整差异图在 JSON 中；独立 struct-float32 oracle 重现。首个例子 8→47，在 target index23，NumPy 取 source4，Torch 取 source3。这个有限集合中两个作者函数一致，不证明任意长度一致。空 source/target1 时作者 IndexError，Diffusers 返回空 slice；无效输入差异不是合法模型结果。

零 residual 后变为 unit residual，作者 ratio 为 Inf（并有 NaN std），Diffusers epsilon 对照为 1e8；有限不等于先验有效。Diffusers 的 `.float()` 在 `out-in` 之后，FP16 `60000-(-60000)` 已 overflow；不能将后续 FP32 norm 当作完整 FP32 校准。

10 个刻意诊断分别保留 equal、长/短/channel 不配的 head、长 tail、singleton 短 tail 广播、不可广播短 tail、channel fallback、FP16 减法溢出、block 注入失败。head 不匹配时可能只返输入而不 full；singleton 短 tail 可广播出大于实际输出的 residual，非 singleton 不匹配可抛异常。**这些是破坏 toy metadata 合同的诊断，不是已证明某个正常 adapter 输入可到达的 bug**。

late projection 失败时，Qwen 已先发布 residual 并 advance cnt；FLUX/Wan 还没 advance，但也非完整 rollback。Diffusers pre-tail block 失败保留旧 residual/clock、head/预算可已改变；显式 cleanup 能清除。source config 接受 NaN/负 ratios，list 默认 FP32、显式 FP64 tensor 保留 FP64；作者 retention0 的 cold-hit 没 residual guard，实验保留 TypeError。不要把这些非默认/注入输入当作正常使用必然失败。

Qwen 另用同一个自有 Toy class 的两实例证明 initializer 的 Tensor/list 类属性共享：一个实例 `cnt += 1` 后另一个见到 1，修改 skip list 后另一实例见到 `[7,0]`。这是明确的多实例 stress，不执行或改动实际 Qwen 类，也不声称作者单 pipeline demo 发生跨用户污染。

## Apple Silicon 可迁移设计

优先借鉴的是固定 full/hit 双路径与少量 scalar controller，不需要 FFT，也不依赖 CUDA 稀疏 kernel。MLX/MPS 可用 GPU resident residual + add 路径；Core ML/ANE 可考虑固定 shape full 图、外部 request-local controller。它们是设计方向，不是本轮性能结果。

实现时让每次成功的 denoiser 调用统一推进一次 clock，hit/full 共用 completion 边界；state key 至少包含 request、branch、stage、shape/dtype 和校准身份。校准统计先 FP32 再减法，拒绝非有限/不合理曲线，保存分支标记、曲线 generation 与校准成本。shape/metadata 失配应明确 full fallback 或拒绝，不静默丢 residual。提交 cache/counter 的事务边界与异常 cleanup 要分开设计。

与 TeaCache 比，MagCache 降低在线信号开销，但把成本和风险转到离线曲线、漂移与映射；与 SeaCache 比，不必付 FFT 但不观察当前 latent 的方向变化。评测应在相同 residual 边界、完整校准成本、holdout prompts/steps/CFG 和质量预算下对照，不按 full-count 或作者 A800 表格给 Apple 上的赢家。

## 重放

公开证据来自 `magcache-cpu-v2`，有原始 bytes/launcher/代码/oracle/来源绑定及确认退出终态。先前 v1 为本地开发试跑，未混入本轮计数。所有非有限值以 null + kind/path ledger 保留，不伪造 finite pass。本轮没有 GPU 工作，无需占用 GPU 空闲窗口；以后设备评测仍须按[评测合同](../evaluation.md)重新准入。
