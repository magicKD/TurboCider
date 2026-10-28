# 可控损失的评测合同：速度、质量、内存与证据分层

<!-- knowledge
id: technique-algorithms-evaluation
type: method
operator: scheduling
backend: cpu-reference
device: cpu
symptom: measurement
semantics: contract-only
boundary: request, producer-consumer
api: reference-only
-->

[算法入口](README.md) · [实现技巧](implementation-patterns.md) · [原有验收原则](methods/approximations.md) · [配套实验](../examples/README.md)

“可控”在本目录表示可配置、可观测、可回退，并在冻结测试集上满足预先声明的预算，**不表示 heuristic threshold 给出了全输入的数学误差上界**。

本页是复用研究结论时的验收合同，不表示目录内所有近似方案已经满足它。尤其要分开“完整生成已执行”和“质量预算已通过”：[DeepCache SD1.5](case-studies/deepcache-sd15-mps-quality.md)前者成立，32个有损计分输出对两个预定RGB预算全部拒绝；[原生Qwen cache](case-studies/native-cache-request-study.md)也保留显著RGB/alpha差异，不能仅凭完整请求加速授予采用资格。

## 实验前先保证比较对象一致

冻结模型/weights hash、LoRA/strength、tokenizer、参考 bytes/顺序/resize、prompt rewriting、seed/noise、sigma/step 表、CFG、精度/route、输出尺寸/帧数/fps、VAE 与导出。对照分成：同权重同采样配方的近似；更少 NFE 的 solver；不同 student/decoder/latent 架构。不得把后两类包装成第一类。

使用优化后的 MLX/MPS/Metal dense 分母；all-exact sparse 控制与 gate-disabled wrapper 控制不是同一个分母。cold load/compile、warm resident request、重复编辑 prefix-hit 分开；保存每次原始耗时，不取最快一次给 candidate、最慢一次给 baseline。

## GPU 占用门槛

GPU 实验前连续采样 `ioreg -r -c AGXAccelerator -d 1` 的 Device/Renderer/Tiler utilization，并只读检查推理/benchmark 进程。任一明显竞争 workload 或利用率超预设门槛，等待后重采，期间继续整理技术；不杀其他任务、不清系统/模型缓存。

`ps` 的 CPU 百分比不是 GPU 利用率；一份 idle 文件不是 live 状态。IORegistry 是驱动采样快照，可能陈旧/缺字段，也不能逐 PID 归因。若无法可靠读取则设备探针 fail-closed；记录 sample 时间、窗口和最大值。测量后再采样，且不把自己的 GPU 利用率误判成竞争。持续 PMU attribution 未采集时明确说“没有发现明显竞争”，不能说“证明 GPU 完全独占”。

本轮可选探针采用3个相隔约0.2秒的 pre samples、Device≤5%，记录 before/after；这是短小微基准的 admission，不足以排除长请求中途有竞争。真实模型的长实验应额外持续记录并在污染后废弃/重跑该 pair。

## 分母与消融

| 方案 | 必备控制 | 要计入的隐藏开销 |
| --- | --- | --- |
| cache | 原 full、wrapper/no-hit、replay/forecast | signal、host scalar sync、缓存读写/graph 驻留 |
| sparse | 优化 dense、sparse all-exact、真正 sparse | summary、selector、sort/gather、layout、restore |
| quant | 原精度、fake-quant、packed runtime | scales、pack、解码、conversion、LoRA与fallback |
| token reduction | ratio=0、仅attn、attn+FFN | matching、indices、bucket/padding、scatter与RoPE |
| distill/solver | 原配方、改solver、student各自默认 | actual NFE/CFG、encoder、VAE、导出 |

按预登记配方做warmup与平衡顺序，例如ABBA/BAAB或轮转/反序；两次warmup只是微基准起点，不是所有报告必须一致的已执行事实。至少多轮保存p50/均值/离散度；微秒级kernel应批量执行并同步，另测空批次开销。MLX必须在输出实际求值并达到已声明的完成边界后停止计时，不能只测建图。静态mask构建若排除，两路都排除并说明；动态routing不能排除。

每个trial先计算`matched_baseline_time / candidate_time`，再汇总中位数/IQR；不要用两个时间中位数相除代替配对统计。IQR不是置信区间，也不意味着每次都更快。[NNC gate](case-studies/nnc-gate-controller-cost.md)存在中位时间之商约1.05、实际配对中位数却约0.933的例子；[T-GATE](case-studies/tgate-mps-resident-cost.md)小幅中位收益伴IQR跨1，均应保留。

包含完整refresh/hit/cleanup窗口，不能只测hit或从时间中扣除host floor。host-ready、device-ready、latent-loop、公共pipeline、媒体落盘和冷启动分别命名；相同NFE/branch调用数不等于相同墙钟或质量。[VDE共同loop](case-studies/vde-mps-controller-cost.md)不是公共pipeline性能，[空间cache](case-studies/spatial-cache-mlx-study.md)是完整adapter周期而非完整视频。

## 质量矩阵与失败定义

数值：全部 finite、shape/layout/valid spans、max abs、相对 L2/cosine；constant/near-zero 明确处理，不把 NaN 当 pass。检查实际 full/reuse/sparse calls、fallback 与分支配对。

trajectory：逐步 velocity/x0/latent 的误差、最后状态、CFG residual；不要只有一层 QKV。图像：固定 decoded color/alpha，PSNR/SSIM/LPIPS 和文字、人脸、手、小物体、边缘；感知模型版本与权重固定，不能静默下载。分布指标需要足够样本，不把单图 SSIM 替成 FID。

视频：全帧均值+最差帧、identity/运动/闪烁/遮挡/参考保真、场景切换与长时漂移；保持分辨率、帧数、fps 和解码成功。含音频时检查样本时长、裁剪、AV同步；LPIPS/光流不能替代 lip-sync。生成文件也需真正解码，不是只看 mux 成功。

阈值在调参前冻结；探索集与 holdout prompt/seed 分开。没有用户质量门槛时先输出 Pareto 曲线，不宣布最佳 preset。通过一组点不等于所有层、步数、参考组合与 LoRA 获得资格。

### 两道不能互相替代的门槛

| 门槛 | 比较对象 | 能支持什么 / 不能支持什么 |
| --- | --- | --- |
| 实现/后端一致性 | 新backend或实现变体 vs 同输入、同近似公式参考 | 排除移植、dtype、分支/时钟错误；不能证明该近似接近full或有好画质 |
| 近似/任务质量 | 近似 vs 匹配full轨迹/媒体，并检查实际任务 | 在冻结数据与预算内给出取舍；相对full的SSIM/PSNR不能替代语义正确性或视频时序 |

例如VDE去guard对原版byte等值、T-GATE GPU对同配方CPU通过，属于第一道；DeepCache i1对full等值也是no-op正控，不给i3/i5授予质量资格。其RGB预算的SSIM与PSNR是**同时满足**的条件，不是二选一；full自身不一定符合prompt，仍需独立的语义/感知评估。

必须保留不通过、非有限、未完成和被排除样本及理由；不能改prompt/seed、调宽阈值或更换指标后仍叫原冻结确认。计划/客户端不符时整条受影响run排除，修正后新run复测，不混合挑选有利臂；[VDE首次run排除](case-studies/vde-mps-controller-cost.md)给出实例。

## 组合成本与性能上限

若 attention-core 占请求比例 f、core 提速 s，其余不变，则理想请求加速 `1/((1-f)+f/s)`；cache/quant/sparse改变 f 后不能沿用旧比例。Amdahl 公式只作上限分析，不能代替测量。

cache 使 sparse 跑更少次；量化噪声可能改 proxy 排序；token merge 改变 cache shape；ANE share 与 staging 隐藏窗口也变化。用 A、B、A+B 的真实组合验证，不乘速度、不将误差预算简单相加。

## 研究结果的四种状态

保留原有标签：`source-reviewed`表示看过指定代码/文档；`math-witness`表示小型数学/反例；`device-component`表示真实设备局部工作；`request-qualified`必须同时具有匹配完整请求与通过的任务质量预算。它们不是自动升级的四级阶梯；**不要把“request executed”写成“request-qualified”**，也不要把CPU接口正控写成设备支持。

一项研究应另外逐项写清以下证据，而非只给一个“已验证”总标签：

| 证据维度 | 必须写明 | 常见越界 |
| --- | --- | --- |
| 来源与实现 | 固定commit/文件、原作者/第三方port/本库变体、许可与实际阅读范围 | 有论文/仓库即视为已发布可执行实现；port沿用原版资格 |
| 输入与消费者 | 标量/affine替身、随机真实网络、真实权重组件、真实完整生成 | 用真实权重的一层FFN代替整模型；用随机pipeline代替预训练画质 |
| 数值核查 | 独立数学、同配方参考、条件反馈重放或aggregate审计各自范围 | 网络输出作重放输入却声称独立重写全NN；hash/finite即正确 |
| 设备与性能 | 实际backend、精度、shape、ready边界、强分母、完整计时样本 | 逻辑API/body数当物理kernel；组件倍率当媒体E2E |
| 质量与资格 | 冻结预算/数据、通过/拒绝/未测、语义/感知/时序范围 | 同配方CPU parity或真实PNG存在即质量通过 |

历史数字应与其模型、输入、环境和计时范围一起阅读；文档或检索检查成功不等于重新执行上游、核查媒体、验证整个网络数学或证明物理 GPU 行为。本库不分发来源账本或 JSON 整理器。

负结果应记录来源/设备/shape/预算、被否决原因和 `revisit_when`；本库 LTX 结果不是否决所有 sparse，MLX 短日程零 hit 也不是否决所有 cache。GPU 被占用时等待/延后，不用污染数字填表。

## 新增记录的最小模板

下面只是分析思路模板，不要求生成机器回执或把实验原件放进知识库：

```text
问题 / 实际省掉的对象：
来源：论文/固定源码链接、commit、许可；原实现/port/本库变体
模型与输入：权重和配方身份；替身/随机网络/真实权重/媒体
合同：布局、dtype、CFG语义身份、step/call/substep、状态owner
正反控：no-op等值、实际hit、fallback、异常/取消/新请求
计时：窗口、强分母、刷新/恢复成本、原始pair与排除理由
数值：同配方实现门槛、独立参考范围
质量：对full与实际任务的冻结预算；通过/拒绝/未测
结论：适用范围、反收益/失败；不支持的推论
复查：何种版本/模型/日程/布局变化需要重测
```

按这个模板复用结论：先匹配本机瓶颈，再移植最小设计，最后在实际任务的相同质量预算下测完整请求。不要先从文献挑最高倍率，再为它寻找有利的计时窗口。
