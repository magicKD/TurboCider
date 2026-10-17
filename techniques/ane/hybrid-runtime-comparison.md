# GPU + ANE：vpipe、ccv、Splash 与真实消费者的方案对照

<!-- knowledge
id: technique-ane-hybrid-runtime-comparison
type: method
operator: swiglu
backend: coreml, private-ane, cpu-reference
device: cpu, ane
symptom: straggler, memory-bound
semantics: numerically-equivalent
boundary: producer-consumer
api: reference-only
-->

[ANE 目录](README.md) · [切分代数](../gpu-ane/partitioning.md) · [共享 IO](../gpu-ane/shared-surface.md)

## 先区分四种“用了两个设备”

encoder→denoiser 是阶段串行；GPU prefill→ANE decode 是按阶段切换；同一 FFN 按 row/intermediate 分担才可能同时完成同一次计算；多请求 batching 又是另一种吞吐优化。代码中开两个 worker、两个 host span 相交、两设备真的同时计算，证据强度不同。

对 SwiGLU 的 intermediate 分担，gate/up 必须成对切，down 取对应列，最终 partial 相加；bias/residual 只加一次。普通 attention→FFN 依赖链不能因为有两个设备就改成并行。更完整的本地数学/反例在 [split_ffn_reference](../examples/split_ffn_reference.py)。

## 可借鉴的最小设计：合法分担一个完整SwiGLU

约定`X[M,H]`、gate/up权重`Wg/Wu[H,F]`、down权重`Wd[F,H]`。按intermediate索引集合分为两部分时，每个设备计算：

`Pi = (SiLU(X·Wg_i + bg_i) ⊙ (X·Wu_i + bu_i)) · Wd_i`

最后`Y=P0+P1+bd+residual`。gate/up必须配对、down选择对应行，down bias和residual只加一次。它不同于将Linear的输入K拆开后分别做SiLU；后者通常破坏原函数。可运行数学与负控在[split_ffn_reference](../examples/split_ffn_reference.py)。

迁移时先实现同图serial控制，再实现并发与join，并同时保留完整强GPU基线。候选的完整时间模型是：

`T = exposed_prepare + max(T_gpu_concurrent, T_ane_concurrent) + exposed_restore_join`

这里两支时间必须来自并发条件下的实际完成边界，不是isolated benchmark。数据格式、buffer owner、失败回退和带宽竞争都进入方案；晚chunk失败后用原输入重做完整结果，不能拼入未验证partial。实现阅读顺序可沿[vpipe参数/worker/consumer](vpipe-callchain-and-failure.md)或[公共backing与真实GPU交接](public-pixel-buffer-gpu-handoff.md)，再回到[五臂完整FFN](resident-ffn-layout-and-gpu-baselines.md)判断是否值得采用。

## 实现方案比较

| 项目/入口 | 权重与图 | 交接/调度 | 最值得学习 | 主要失效条件 |
| --- | --- | --- | --- | --- |
| vpipe | frozen 或 runtime；图内 K/N tile；SwiGLU/GELU 模板 | worker、GPU remainder、同一 prediction 内归约 | 从 checkpoint 布局到 graph，再到模型调用 | 手写 `.mlmodelc` 格式、频繁 host chunk、权重准备税 |
| ccv/MFA rowwise | runtime INT8 codes/scales，图内 K 分块/DQ/matmul/sum | pixelbuffer/IOSurface、output backing、分类型 cache | 把资源复用与 kernel 边界一起设计 | 实际 output backing 未采用、stride/owner、特定 shape 阈值 |
| 历史 Splash FFN 分支 | normalized W8A8、Hadamard/sign basis、paired segments | GPU stager、双 weight bank、events、校准 share | 完整 prepare→execute→restore→join 与 breaker | rotation 配方错、prefetch 暴露、ANE straggler、精度失败 |
| Draw Things/ccv 应用链 | 模型/精度/LoRA 选择决定底层路径 | 应用 memory budget、kernel/bridge、阶段驻留 | 上层模型选择如何形成实际成本 | 不能从 App 名称推某个请求默认跑 ANE/NAX |
| 本库 public 路线 | frozen/runtime FP16/W8，公共 Core ML worker | 公共 IO、staging、完整 GPU fallback | 生命周期、artifact/recipe、参数身份 | 允许 CPU fallback；plan 非物理 trace |
| 本库 private research | MIL + `_ANEClient`/e5rt；copy 或实际共享 surface | 同步 private evaluate + GPU suffix/join | 匹配强 GPU 分母与真实 backing 见证 | 私有兼容性、冷启动/驻留、完整请求尚未资格 |

固定来源与代码位置：[vpipe emitter](https://github.com/tgo-app-dev/vpipe/blob/c2989a76841888e4d8d208fad4f476c487710962/apple-silicon/coreml/ane-emitter.cc)、[ccv rowwise](https://github.com/liuliu/ccv/blob/66dc16a3a3b688adfa6dc1d19eeeb3071beeb701/lib/nnc/mfa/ccv_nnc_mfa_ane_rowwise_coreml.mm)；历史 Splash 为本地 `liang2kl/splash@80827e1c` 分支，与路线图当前 `incoai/splash` 入口不能不经版本核对就合并。其 [AneFfn/Measurement/Calibration/Handoff 导读](../references/splash-w8a8-walkthrough.md)与 [来源登记](../sources/README.md)保留逐文件身份。Draw Things 社区入口：[draw-things-community](https://github.com/drawthingsai/draw-things-community)，本轮只作为上层学习导航，未审核整条应用调用链。

## 三个不应直接照抄的实现细节

vpipe 调用公共 Core ML，却自行构造未文档化 compiled bundle；**公共函数调用不覆盖自写文件格式的稳定性风险**。本库实验用 coremltools 正规 compile 路径，不能替它的 bundle writer 取得兼容资格。

ccv 请求 output backing 后仍需验证实际返回 pointer/shape/stride。已读路径存在“若 backing 被忽略，预定 scratch 是否仍被消费者读取”的条件风险；本轮没有运行该上游 bridge，不能称已复现 bug，见 [源码卡](../sources/repos/ccv-coreml-rowwise.md)。cache 的 512MiB/128MiB/1GiB 是软件额度，不是 ANE 硬件缓存。

Splash 的 sign RNG、rotation、scale/offset 与本库配方不完全相同；同 seed、同 W8A8 名字不代表 bitwise 同编码。预取/双 bank 成功也不保证关键路径缩短，反而可能争抢统一内存带宽。

## 完整成本模型

候选必须覆盖 exposed preparation + concurrent GPU/ANE 较慢分支 + exposed restore/join；并发时两支速度要实际测，不能拿 isolated 时间直接取 max。GPU-only 强基线优化后，ANE share 的最优值也会变；可选 CPU tail 同样可能成为 straggler。

共享物理内存只省去某些外部复制，不自动消除 dtype/packing、page residency、cache coherence、hazard、锁和同步。shared surface、Metal buffer、MLX view 的 owner 必须保活到最后读者完成。

## 本库已有实测怎样帮助选择

[公共真实 Qwen FFN tiling](../gpu_ane/case-studies/qwen-ffn-tiling-and-concurrent-cost.md)保留“某些表达数值失败、另一些能运行但完整消费者更慢”的负结果。[private shared FFN 五臂](../gpu_ane/case-studies/qwen-shared-ffn-consumer-cost.md)在 M4 Max、layer0、M1024/H4096/F12288、固定 share2048 的 warm 完整组件里，长确认约1.146–1.149×完整原 BF16 GPU；同 client 的 copy→shared 仅约1.1%，首轮甚至方向相反。

这说明主要收益不能全部归给“零拷贝”。serial 方案也不应当作提速分母；它在该实验慢于完整 GPU。上述是历史已记录组件证据，本轮未重新执行；没有完整图像/视频请求、LoRA、冷启动/峰值内存或物理重叠资格。

实践优先级：先合法切分 → 匹配数值 → serial/candidate/GPU 对照 → concurrent 完整 consumer → 真实 request；最后才基于 profiler 调 share/tiling。不要以“让 ANE 满载”为独立优化目标。
