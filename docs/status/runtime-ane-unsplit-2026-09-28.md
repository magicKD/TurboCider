# Runtime ANE：Qwen 无收益层恢复完整 GPU 块

2026-09-28，接续 [graph-v2 与 chunks0 消融](runtime-ane-lora-2026-09-28.md)。
此改动只影响显式 runtime-weight 路由，不改变自动默认、冻结图选择或 LoRA 权重。

本页保留 `fb79…` 的历史行为和成绩。后续 `f287…` 已把自动 GPU probe 改为
[完整块探测](runtime-ane-full-block-probe-2026-09-29.md)，只有显式 chunks=0
仍走拆图消融；不要把下表旧的 GpuProbe 行为当成当前实现。

## 已实现

参照 VPIPE 的 `plan_block()` / `needs_barrier()` 思路，把三种决策分开：

| 决策 | 执行行为 |
| --- | --- |
| Hybrid | staging 与 attention 重叠，GPU 处理 head，Core ML 处理 tail |
| GpuProbe | 保留拆图边界，测全行 GPU FFN，更新调度样本 |
| Gpu | Qwen 直接运行完整编译 GPU block，不进入 FFN 桥接/计时同步 |

`RowScheduler::plan()` 每次只推进一次 visit，`HybridFfn::plan_block()` 把
决定传给后续 stage。旧 `select()` / stage-only 调用仍兼容，Z 当前仍使用
原来的 stage-only 路径；不能把 Qwen 的 unsplit 改进记到 Z 上。

Qwen 的每个 block、prefill/decode/cache 模式各自保存 full/split 两个编译函数。
`split_this` 是编译时捕获值，不能在同一缓存里换标志后继续使用旧函数。
从 unsplit 层重新进入测量层时，在 staging 计时开始前 `eval(args)`，避免
将此前排队的 GPU 工作计入当前层 attention/staging。

`TURBOCIDER_RUNTIME_ANE_CHUNKS=0` 仍是原有**拆图 GPU 消融**，不偷偷改成
普通 GPU 基线。固定正数仍强制混合，auto 的 disabled 层周期性重新探测。
新遥测 `unsplit_gpu_blocks_session_total` 是 `gpu_blocks_session_total` 的子集；
普通 GPU 层不会作为新的性能样本影响调度 EMA。

LoRA 在 full GPU 块使用原 `Weights::project`；混合块仍通过 pre-SiLU
gate/up 修正及 hidden 上的 down-LoRA 完整计算，不合并、跳过或近似省略投影。

## 当前构建与验证

原生库 SHA256：
`fb79ad015ebc2cd6b1fdc6fe85785e4168e5f0853b2ff8835ce7766c6b47f435`。

- `TURBOCIDER_NATIVE_ONLY=1 make build` 成功。
- `make test-runtime-ane`：4 host + 7 图/集成通过。
- 新 Qwen 合成模型反复切换 full/split，覆盖 prefill、decode、prefix K/V
  返回索引和移除 callback；对普通 GPU 的 relative L2 为 0。
- 新调度测试覆盖 Hybrid/Probe/Gpu、周期性 reprobe、层/形状隔离；桥接
  测试检查 plan/stage 不重复推进、错误层号拒绝及短序列 bypass。
- 完整 LoRA 微图最差 relative L2 0.00533939，Q4/Q8 0.00658136，
  headroom、全 tail 回退、取消/drain 用例仍通过。
- `make test-acceleration-contract`：4 host + 16 报告/preflight + 11 CLI 通过。
- 所有整模型测试结束后，同一构建的 `make test-qwen21` 和 `make test` 均
  成功退出；专用 audit/test-hook、显式 GPU opt-in 和缺失夹具用例仍按既有
  门禁跳过，不计为已覆盖。deployment-target/临时目录警告不影响成功退出。
- 90 处本地文档链接/锚点、路径独立性及 staged/unstaged 空白检查通过；
  249 个此前移出索引的本地产物仍全部保留，未改暂存状态、未提交 commit。

这些微图不代表真实 Q4 checkpoint、编辑质量或性能资格。测试中一次提前于
原生构建完成的链接失败不计为验证结果；最终测试是在构建终止后重跑通过。

同一新构建还重新通过两模型真实共图切换：固定 chunks=1，先暖 base，再
base → A → 合成 B → base，首尾 base PNG 相同、A/B 不同、Core ML 未重载。
Qwen 累计调用 192/384/576/768/960，绑定投影 0/0/227/227/0；Z 调用
259/515/771/1027/1283（含初始三次 headroom 重试），投影 0/0/238/238/0。
证据在 `outputs/runtime-ane/{qwen,z}-runtime-lora-v2-unsplit-switch/`。
固定 chunk 是状态隔离检查，不是 auto 性能成绩；B 只是合成 LoRA，不是另一
个训练 adapter 的质量资格。Qwen 六步 base 仅用于状态检查。

## 六步 Qwen LoRA 初测

M4 Max 64 GB，512²，Viggle v0.2.1 r256、六步，fox/seed42，c288/tile1024，
runtime auto；每路一冷三热、runtime→GPU 顺序，热整请求含 VAE/PNG。
证据：`outputs/runtime-ane/qwen-runtime-lora-v2-unsplit-screen/`。

| 路线 | 三次热请求 | 热中位 |
| --- | --- | ---: |
| 普通 GPU | 8.168 / 8.099 / 8.227 s | 8.168 s |
| runtime v2，允许 unsplit | 9.688 / 9.530 / 9.351 s | 9.530 s |

GPU/runtime 约 **0.857×**，仍未超过普通 GPU，不作为推荐加速路径。
此前 `ed7…` 构建约 10.811 s；新结果提示减少了开销，但这是跨构建小样本筛选，
不是严格 ABBA 因果量化。三个热请求仍随调度变化，不能忽略较慢请求只报最小值。

四个请求累计 hybrid block 为 102/159/175/187，unsplit GPU block 为
26/161/305/453；最后一个请求为 12 hybrid、148 unsplit GPU、32 split GPU probe。
无错误回退或 headroom 重试。最后 denoise 为 8.192 s，普通 GPU 对应
7.557 s；请求外的 adapter 身份检查也仍有成本，不能把全部差距记为 FFN。
已肉眼检查最后一对 PNG：主体、姿态、色调接近，局部枝叶/毛发细节略变；
不是其他提示词、第二个训练 LoRA 或参考编辑的质量验收。

## 同构建 base 三路回归

同一 `fb79…` 构建，BF16 base、512²、fox/seed42，每路排除一冷请求后取两个
热请求中位；Qwen 顺序 GPU→runtime→frozen，Z 顺序相反。不是正式 ABBA 验收。

| 工作负载 | 普通 GPU | runtime c288/auto | 原冻结图 | GPU/runtime | GPU/冻结图 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen base，40 步 | 42.735 s | 38.988 s | 30.221 s | 1.096× | 1.414× |
| Z base，8 步 | 7.009 s | 6.587 s | 5.357 s | 1.064× | 1.308× |

证据：`outputs/runtime-ane/qwen-base-unsplit-regression/` 和
`outputs/runtime-ane/z-base-unsplit-regression/`。Qwen 冻结图沿用 6144-channel
W8A8，Z 沿用 1024-image-row/5120-channel W8A8；没有换配置重新定义基线。
新 runtime 仍没超过已有冻结图，故保留更快的旧路线，不改自动默认。

Qwen runtime 三请求累计 3648 hybrid / 192 GPU probe，unsplit 为 0，说明
无 LoRA 时该图仍持续有收益；不能把六步 LoRA 的退化外推到 base。
Z 累计 640 hybrid / 128 GPU，643 predictions 包含冷请求的三次 headroom
恢复至 scale64；两模型都无错误回退。

已肉眼检查两模型最后的 GPU/runtime PNG，主体、构图、色调接近，局部细节不同。
另外，两模型普通 GPU 与冻结图各自最后的 PNG，分别与旧 `qwen-model-screen-a/`
和 `z-gguf-integration-bf16-regression/` 的对应 PNG SHA256 完全相同。
这是一个固定样本的原路径输出回归证据，不要求不同近似路线像素相同。

六个 base trial 的前后 swap usage 都为 958.38 MiB，swapins 都为 2726；
没有观察到这些 trial 内新增 swap-in，但系统 wired 数值会变化，且 MLX peak
不包含 Core ML/OS，不能据此宣称已通过 §72 完整内存压力资格。
CPU+NeuralEngine 仍只是调度策略，实际物理 ANE residency 未独立确认。

## 尚未满足的验收

本轮不是整个设计目标完成。§82 的 Q4 真实模型、广泛分辨率/LoRA 质量、
无换页回归、完整 overlap/内存证据仍不足；新 runtime 也尚未取代更快的冻结图。
§42 之后的双缓冲、QKV、runtime W8A8/W4 compute 等后续路线未在本轮实现。
下一步处理剩余 probe/依赖链和内存测量：目前 GPU probe 仍使用拆图 FFN，
可能高估相对于真正完整 GPU 块的混合收益，需要进一步对照完整块计时；
短请求还会跨多个请求经历 prefill 的调度采样。不能只选最好一次热请求，
也不以组件通过代替完整模型验收。
