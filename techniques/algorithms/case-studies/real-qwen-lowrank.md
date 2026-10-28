# 真实Qwen全宽FFN：低秩目标、校准迁移与量化组合误差

<!-- knowledge
id: technique-algorithms-case-studies-real-qwen-lowrank
type: case-study
stage: dit, ffn
operator: gemm, quantization
backend: mlx, cpu-reference
device: cpu, gpu
symptom: precision-loss, memory-bound, measurement
semantics: approximate, contract-only
representation: bf16, fp32, q4, q8, packed, mlx-affine
boundary: weight-runtime, producer-consumer
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [SVDQuant/转换端](../methods/low-rank-quantization.md) · [合成GPU组件](lowrank-mlx-component-study.md) · [实验](../../examples/README.md)

2026-10-16第九轮。从合成权重推进到原始Qwen-Image-2.1 BF16 checkpoint、image-stream FFN和两条真实轨迹。[DeepCompressor S43](../sources.md#s43)用于理解校准/转换语义，**没有执行其代码、W4A4、GPTQ或作者模型转换**。本轮原创近似使用真实MLX0.32.0 CPU affine packed W4/W8、FP32 activation/scale/bias/factor；不是第六轮的FP16合成GPU配方。

第十轮另见[真实BF16 FFN GPU成本](real-ffn-gpu-study.md)：同源权重/输入，重新冻结BF16配方、强GPU dense和4752段formal计时；不是将本页FP32误差直接配给GPU速度。

## 数据与独立prompt边界

原base轨迹为seed42的公开茶壶示例，缓存关闭、无LoRA、20步；新轨迹为seed43自行车示例，不同prompt和seed、同一source-matched native运行时。原运行时648个源码文件匹配；新采集前3个GPU利用率样本均0，另有不读argv的脱敏进程检查。driver快照不证明连续per-PID独占。新采集保留owned child正常退出和原模型/二进制未改变，**仪器化生成耗时不作性能分数**。

此前同prompt采集曾因267MiB磁盘门槛拒绝，不能当作成功holdout；本次使用新的成功采集。两轨迹各取layer0/15/31、step1/10/19，共18份原始BF16输入。每份原形状1×1024×4096，取33个固定均匀image rows（含0/1023），不抽掉K/F维。这里的holdout表示未用于scale/factor拟合的独立prompt/seed，不表示覆盖全提示词分布。

每层完整weight是GU[24576,4096]、Down[4096,12288]，SwiGLU后仍为12288维，输出4096维。reference用原BF16值无损展开到FP64后计算数学FFN，**不是原BF16 runtime的逐位输出oracle**。候选也仅在原轨迹输入上teacher-forced局部比较，没有将量化输出反馈进后续blocks或采样轨迹。

## 七配方与拟合隔离

矩阵固定为3层×2轨迹×3步×7配方。先冻结方法，再保留所有校准/非校准点，不按heldout误差重新挑rank或scale。

正式重跑完成全部126点、252个scalar见证，见。旧pilot与最终封装失败的数据不混入正式cohort；全部6份weight和各candidate参数前后hash不变，独立读取器也核对三个MLP只有已计入的两份weight、没有被遗漏的bias/额外参数。

| 配方 | 主分支 | 高精度支路 / scale |
| --- | --- | --- |
| Q8 / Q4-r0 | 直接affine Q(W)，group64 | 无低秩 |
| Q4-r32 / r64 | Q(W−BAᵀ) | 对W作rank32/64近似 |
| Q4-act-r32 / r64 | Q(Ws−BAᵀ) | 对Ws分解，输入X/s |
| Q4-error-r32 | **不变的原Q(W)** | 对W−decoded Q(W)分解，另加误差branch |

随机分解固定seed17、range80、1次power iteration、reduced SVD，sqrt奇异值分给A/B；它不是作者double full SVD、100轮OutputsError/early-stop或smooth grid search。diagonal scale只从本层baseline step1的33行RMS/中位RMS得到，clip到[.01,100]；Down的校准hidden来自同点原始GU的FP64计算。其它步与独立prompt不更新scale、因子或packed bank。

diagonal-RMS是本库原创输出加权近似，不称为已完成SmoothQuant/AWQ/SVDQuant。它忽略XᵀX非对角项，也改变group量化的range；本轮没有逐项辨识相关性、range、随机分解和跨步漂移的贡献。

## 完整函数与独立消融

每点保存GU linear、SwiGLU hidden、GU-only（原Down）、Down-only（原hidden）、两矩阵组合、真实MLX CPU相对decoded公式，以及真实CPU相对原FP64函数。norm覆盖全部33行/全输出通道，同时保留每行SSE/energy、p95和最大值。

真实CPU的两个packed QMM、FP32 SiLU×up、分离低秩与join都执行；独立参考按实际uint32 codes、FP32 scale/bias解码，在FP64按同一公式计算。每点另有两格完整F维math.fsum见证。原数值gate固定为native-vs-formula global≤1e−4、最大row≤5e−4、未量化weight重构≤1e−6、decode≤1e−6；**这些不约束对原模型的量化误差，更不是媒体质量门槛**。

还要保留GU/Down组合cross term：`ΔY=Δz Dᵀ+z ΔDᵀ+Δz ΔDᵀ`。独立两项误差不能直接加为组合预算；native backend与其同配方reference的条件三角界则另外检查，不混用分母。

## 真实误差：不能用早层代表整个模型

下表是每层两轨迹/三个step共6点的combined global relative L2范围，单位%，不是置信区间、画质分数或允许上线的预算：

| 配方 | layer0 | layer15 | layer31 |
| --- | --- | --- | --- |
| Q8 | 0.0586–0.1437 | 0.3517–0.5446 | 0.1050–0.1333 |
| Q4-r0 | 1.2421–2.5708 | 5.7835–9.2411 | 2.0118–2.3850 |
| Q4-r32 | 0.8599–2.2190 | 5.5484–8.8765 | 1.1326–1.5206 |
| Q4-r64 | 0.8548–2.2490 | 5.5340–8.7394 | 0.9820–1.2744 |
| Q4-act-r32 | 1.4725–4.7799 | 10.3306–12.7553 | 0.9237–3.6013 |
| Q4-act-r64 | 1.4208–5.0732 | 10.0970–12.5538 | 0.8133–3.1508 |
| Q4-error-r32 | 0.9823–2.3486 | 5.6298–9.1092 | 1.8670–2.2034 |

普通r32/r64与fixed-main error-r32在这18个点均比本组Q4-r0的global误差小，但改善幅度不同：中层的低秩收益较小，不能借早层结果给所有层相同预算。Q8在本矩阵更准且逻辑bytes更多；这不等于在原BF16 runtime/完整生成里胜出。

更高rank不保证逐点更好：普通r64有3/18点比r32的global误差更大。两种RMS加权rank各有13/18点比Q4-r0更差；layer0的baseline step19，act-r32约4.398%对Q4-r0约1.495%，独立prompt对应约4.780%对1.558%。该负结果属于本库单点RMS/随机分解配方，不是作者完整平滑/输出目标算法的反证。

全局L2也不能遮住尾行：layer15的Q4-r0最差row约12.924%，act-r32约19.242%；layer31普通r64最差row约1.593%。JSON保留全部33行SSE/energy、每点p95/max和原生输出hash，不仅有这张范围表。

实际MLX CPU与相同decoded公式的最大global相对L2约3.23e−6、最大row约8.41e−6，均通过原数值gate；这与表中对原函数的百分比近似误差是两个分母/目的。本次真实消融没有观察到combined row范数超过GU-only与Down-only范数之和（全部4158个method-row均未违反）；**cross term的数学风险仍由独立标量witness说明，不捏造真实数据中的违例**。

## 逻辑bytes与Apple采用边界

以下为每层GU＋Down的实际格式算术，MiB=2²⁰ bytes；不是峰值或全模型压缩比：

| 表示 | MiB |
| --- | --- |
| 原BF16 weights | 288 |
| Q8，含FP32 scale/bias | 162 |
| Q4-r0，含FP32 scale/bias | 90 |
| Q4-r32 / error-r32 | 95.5 |
| Q4-r64 | 101 |
| Q4-act-r32 | 95.5625 |
| Q4-act-r64 | 101.0625 |

rank32/64支路额外算术约为原两矩阵GEMM的0.955%/1.910%，不包含非线性/scale/launch/join，**不是延迟overhead**。harness还同时持有FP32/FP64原权重、参考和分解临时量，不能用表中bytes宣称本进程峰值下降。

第六轮独立FP16分支的GPU负结果仍成立于其合成配方；不能把这里真实权重的CPU误差与那里的GPU时间拼成Mac模型Pareto曲线。部署FP16 activation/scale/factor或不同group、实际权重streaming、融合、完整SwiGLU/请求时都要重新验收。packed API和weight bit数不证明INT4 MAC，也不授予W4A4后端支持。

## 两个参考/封装拒绝均保留

初次FP64参考在NumPy BLAS matmul抛`divide by zero`，尚未进入量化比较。独立诊断入口核对该真实输入abs max约.664、weight约.789，全部输出finite；warning模式仅作诊断，BLAS与strict einsum全部811008个元素最大差约1.11e−16，两个独立fsum cell相同。该观察不证明所有BLAS或OS都有相同问题。正式参考改为`einsum(optimize=False)`，仍对invalid/overflow/divide抛错，QR/SVD保留LAPACK；没有在正式候选里忽略异常或降低数值gate。

## 可追溯交付与后续实验

原始执行入口仅在显式`--run`后读取权重/激活/MLX。独立标准库整理器另从原文件重新核对6份BF16 weight spans、18份完整capsule、33行BF16→FP32展开和capture终态；默认tests只读公开统计/身份，不读取模型或原始activation，也不执行上游/NumPy/MLX/GPU。

数据verifier重算SSE、分位数、bytes、scalar和资格字段，但不重新执行全部SVD/FFN输出，不是硬件attestation。失败、holdout、尾部行与没有改善的rank不能删除。后续仍需独立prompt集合、更多step/层/LoRA、目标GPU packed/fused实现、完整量化trajectory和RGB/视频；本轮只支持条件化研究结论，不改产品默认。

## 本轮离线交付验证

原始CPU报告两次经标准库curator独立重读原件整理，公开JSON与ignored整理件逐字节一致；来源44条/161个文件hash、生成检索索引与`git diff --check`通过。参考BLAS拒绝与报告封装失败仍保留，数值gate没有放宽，也未用早先丢失输出回填正式结果。

实际权重/设备实验入口需另行显式提供原checkpoint、baseline/heldout capture及终态来源路径；`--help`给出参数，不自动寻找其它用户的模型、下载资产或执行新请求。
