# Runtime ANE：稳定混合阶段减少计时同步

本页保留 `bc1…` 构建证据；后续整理构建已补充
[真实 Q4 初测及 Z BF16/冻结图回归](runtime-ane-q4.md)。当前决策见
[加速维护入口](acceleration.md)，不覆盖本页历史数据。

接续 [完整 GPU block 探测](runtime-ane-full-block-probe-2026-09-29.md)。
本轮继续落实设计中的低干扰调度：保留完整 GPU 对照，不在每个稳定混合块
都强制同步输入和残差。只影响显式 runtime 的 Qwen 模型级计划路径，
不改变自动产品路由、冻结 base 图、LoRA 权重或完整计算公式。

## 实现与约束

- `RowScheduler::HybridUntimed`：已经取得 GPU/混合样本且当前 chunk 稳定后，
  正常执行混合 FFN，不采集完整 block 样本。初始化前六次仍测量；之后每个
  layer/row-count 的 32-visit 周期内，在 GPU probe 前后重新采集混合样本。
  未完成新 chunk 的 warmup/measurement 时继续计时，不提前停止学习。
- Qwen 的 `SplitUntimed` 复用已有 split 编译函数。FFN 执行器返回已经求值、
  自有存储的输出，因此 residual 可保持 lazy。冻结图 callback 可能借用共享
  backing，仍保留原同步，不能照搬这个优化。
- 只有完整受测 block 更新 on/off 控制器。未计时路径不把较短的 FFN 时窗
  偷换成完整 block 时间；stage-only 的 Z 调用方式和既有样本策略不变。
- 未计时计划在 `run()` 完成或抛出时释放，不等待不存在的 observe callback。
  重复/错误 observe 拒绝；staging 失败仍全 GPU 重算，后续 block 不被挂住。
- 新增 `untimed_hybrid_blocks_session_total`，是累计 hybrid blocks 的子集；
  报告工具检查类型、单调性和子集关系，兼容未报告此字段的旧版本。

这不是跳过层、SiLU、LoRA 或残差，也没有改动量化或数值容差。
固定 chunks 和 chunks=0 消融保持原行为。周期学习只调节已选择的 runtime，
不会把普通 GPU 请求改成 GPU/ANE。

## 构建与验证

原生库 SHA256：
`bc1c665d17942720c34b02fa260cc4f69fe2cef9d2be6e56525c96463618978e`。

- 原生构建成功；host 4 项、报告/preflight 18 项、CLI 11 项通过。
- Core ML 7 项微图/集成通过；Qwen full/split/untimed/probe 切换 relative L2 0。
  LoRA 微图最差 relative L2 0.00533939；Q4/Q8 微图 0.00658136。
- 新增测试覆盖周期样本、稳定块不 observe、untimed staging 失败后回退，
  以及返回输出不会被下一次 prediction 的 backing 覆盖。

最新整模型记录见下文；微图不是完整模型性能或广泛画质验收。

同构建两模型真实共图状态切换均通过：固定 chunks=1，暖 base → base →
训练 adapter A → 合成 B → base，首尾 base PNG 相同、A/B 不同，图未重载。
Qwen 调用 192/384/576/768/960、绑定投影 0/0/227/227/0；Z 调用
259/515/771/1027/1283、投影 0/0/238/238/0。依据为
`outputs/runtime-ane/{qwen,z}-runtime-lora-v2-sparse-probe-switch/`。
Qwen 回归开始前检测到 ComfyUI 推理并等待后才运行，没有终止其他进程。
B 是临时合成文件，不代表第二个训练 LoRA 的质量；Qwen 六步 base 仅用于
检查状态，不是 base 画质验收。固定 chunk 切换也不冒充 auto 性能成绩。

随后重新通过完整 7 项微图（含新增输出所有权检查）及 `make test-qwen21`。
完整 `make test` 也成功退出；专用 audit/test-hook、GPU opt-in 与缺失夹具
用例仍按既有门禁跳过，不当成已覆盖。layout 7 项、103 处本地文档链接/锚点、
staged/unstaged 空白检查通过。249 个原有暂存移除对应文件仍全部在本地，
暂存内容摘要未变；没有提交 commit 或删除模型/缓存。微图 SDK 的临时目录
清理警告不影响成功退出。

## 两模型 base 三路筛选

M4 Max 64 GB；BF16 base，无 LoRA；512²、40 步、狐狸提示词、seed42。
runtime c288/tile1024，auto；resident，每路一冷两热，顺序 runtime → GPU → frozen。
原始依据：`outputs/runtime-ane/qwen-base-sparse-probe-regression/`。

同构建 Z base 为 512²、8 步、同提示词/seed，c288/auto；每路一冷两热，
顺序 frozen → runtime → GPU。两模型串行，运行器在每个 trial 前检查竞争负载。
它是启发式检查，不证明整个测试期间设备独占。

| 模型 | GPU 热中位 | runtime 热中位 | frozen 热中位 | GPU/runtime | GPU/frozen |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen，40 步 | 42.773487 s | 39.340022 s | 30.313312 s | 1.087× | 1.411× |
| Z，8 步 | 7.010356 s | 6.561668 s | 5.363362 s | 1.068× | 1.307× |

热样本全部保留（不从中挑最低值）：

- Qwen runtime：39.320215 / 39.359829 s；GPU：42.757098 / 42.789876 s；
  frozen：30.216268 / 30.410356 s。
- Z runtime：6.563393 / 6.559943 s；GPU：7.010658 / 7.010053 s；
  frozen：5.359732 / 5.366991 s。

Qwen runtime 累计 3648 hybrid、192 完整
GPU probe，其中 3264 个混合块省去完整块计时同步；没有错误回退或溢出重试。
相比上一 `f287…` 的 39.692170 s，名义快约 **0.9%**，但这是跨构建、反向
顺序的小样本筛选，不是严格 A/B 因果或正式 ABBA 验收。

Z 的证据为 `outputs/runtime-ane/z-base-sparse-probe-regression/`，累计
674 hybrid / 94 GPU、677 predictions，其中冷请求 3 次 headroom 重试至
scale64；无错误回退。Z 仍走 legacy stage-only 路径，untimed/full-GPU-probe
两个计数均为零，不把本轮 Qwen 优化宣传成 Z 的新提速。

Qwen 三路最后 PNG 分别与 `f287…` 的对应输出 SHA 完全相同；已查看那组
GPU/runtime 对比，主体和构图接近。Z GPU/frozen 最后 PNG 也各自与此前
`fb79…` 的输出相同；本轮 Z GPU/runtime 已肉眼比较，主体、构图和色调接近，
局部细节有变化。固定狐狸样本不代表广泛生成/编辑质量验收。
六个 trial 前后 swap usage 均 958.38 MiB、swapins 2742；wired 会变化，
MLX peak 不含 Core ML/OS，不能用这些快照宣称通过完整内存/换页资格。

## 六步 Qwen LoRA 复测

同构建、Viggle v0.2.1 r256、512²、6 步，runtime v2 c288/auto → GPU，
各一冷三热。证据：`outputs/runtime-ane/qwen-runtime-lora-v2-sparse-probe-screen/`。

| 路线 | 三次热请求 | 热中位 |
| --- | --- | ---: |
| runtime | 8.988773 / 8.821256 / 8.701361 s | 8.821256 s |
| GPU | 8.242489 / 8.226921 / 8.174660 s | 8.226921 s |

GPU/runtime **0.933×**，仍慢约 **7.2%**，没有可靠提升。四请求累计 hybrid
97 / 129 / 129 / 129，最后两个请求完全走 GPU；untimed hybrid 始终为 0，
说明这次稳定混合优化未作用于已被关闭的 LoRA 混合路径。每请求绑定 227
投影，无错误回退；最后 GPU/runtime PNG 相同且肉眼正常。不能将这对图片
作为持续 ANE 混合生成的质量证据。

## LoRA 非计算开销

单独测量确认，对 Viggle r256 文件执行原 `tc::sha256_file` 的空闲热中位约
**0.561 s**。它能解释此前 runtime LoRA 相对 GPU 的大部分非 denoise 差距；
完整计时和限制见 [SHA 诊断](runtime-ane-full-block-probe-2026-09-29.md#接续诊断lora-文件校验成本)。
本轮没有通过取消校验、只看 size/mtime 或合并 adapter 来取得性能数字。
Qwen runtime LoRA 尚未证明稳定快于普通 GPU。

## 仍需继续的工作

新 runtime-weight 仍未超过原冻结 base 图。Q4 真实 checkpoint、Qwen
runtime-weight 1024²/编辑、多训练 LoRA 视觉验收、完整峰值内存/换页与物理
ANE residency、不可变 artifact lease 仍缺证据。双缓冲、QKV、runtime INT8
属于后续优化，不能把当前 FP16 staging 当成已完成这些路线。
不因本次测试通过而宣告整个设计目标完成。
