# 可控损失的评测合同：速度、质量、内存与证据分层

[算法入口](README.md) · [原有验收原则](approximations.md) · [配套实验](../labs/algorithm-research.md)

“可控”在本目录表示可配置、可观测、可回退，并在冻结测试集上满足预先声明的预算，**不表示 heuristic threshold 给出了全输入的数学误差上界**。

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

先2次 warmup，再交替 ABBA/BAAB 或轮换多个候选，至少多轮保存 p50/均值/离散度；微秒级 kernel 应批量执行并同步，另测空批次开销。MLX 必须 `mx.eval(output)` 后停止计时，不能只测建图。静态 mask 构建若排除，两路都排除并说明；动态 routing 不能排除。

## 质量矩阵与失败定义

数值：全部 finite、shape/layout/valid spans、max abs、相对 L2/cosine；constant/near-zero 明确处理，不把 NaN 当 pass。检查实际 full/reuse/sparse calls、fallback 与分支配对。

trajectory：逐步 velocity/x0/latent 的误差、最后状态、CFG residual；不要只有一层 QKV。图像：固定 decoded color/alpha，PSNR/SSIM/LPIPS 和文字、人脸、手、小物体、边缘；感知模型版本与权重固定，不能静默下载。分布指标需要足够样本，不把单图 SSIM 替成 FID。

视频：全帧均值+最差帧、identity/运动/闪烁/遮挡/参考保真、场景切换与长时漂移；保持分辨率、帧数、fps 和解码成功。含音频时检查样本时长、裁剪、AV同步；LPIPS/光流不能替代 lip-sync。生成文件也需真正解码，不是只看 mux 成功。

阈值在调参前冻结；探索集与 holdout prompt/seed 分开。没有用户质量门槛时先输出 Pareto 曲线，不宣布最佳 preset。通过一组点不等于所有层、步数、参考组合与 LoRA 获得资格。

## 组合成本与性能上限

若 attention-core 占请求比例 f、core 提速 s，其余不变，则理想请求加速 `1/((1-f)+f/s)`；cache/quant/sparse改变 f 后不能沿用旧比例。Amdahl 公式只作上限分析，不能代替测量。

cache 使 sparse 跑更少次；量化噪声可能改 proxy 排序；token merge 改变 cache shape；ANE share 与 staging 隐藏窗口也变化。用 A、B、A+B 的真实组合验证，不乘速度、不将误差预算简单相加。

## 研究结果的四种状态

`source-reviewed`：看过指定代码/文档；`math-witness`：小型数学/反例；`device-component`：真实设备局部工作；`request-qualified`：匹配完整请求与质量矩阵。四个轴不互相继承。

负结果应记录来源/设备/shape/预算、被否决原因和 `revisit_when`；本库 LTX 结果不是否决所有 sparse，MLX 短日程零 hit 也不是否决所有 cache。GPU 被占用时等待/延后，不用污染数字填表。
