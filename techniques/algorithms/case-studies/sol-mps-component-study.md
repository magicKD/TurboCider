# 原始 Sol-Attn Metal 实测：长度门槛、误差曲线与双 dense 基线

<!-- knowledge
id: technique-algorithms-case-studies-sol-mps-component-study
type: case-study
stage: prefill, dit
operator: attention, layout
backend: metal, mlx, cpu-reference
device: gpu, cpu
symptom: precision-loss, launch-bound, measurement
semantics: approximate, numerically-equivalent
representation: bf16, fp32
boundary: producer-consumer
api: reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [Sol机制](../methods/sol-attention.md) · [调试器边界](sol-debug-bounds.md) · [真实QKV分析](real-qwen-sparsity.md) · [实验合同](../../examples/README.md)

2026-10-16第三轮。执行固定Sana源码的tiled MPS/Metal forward，不再用自行写的keep/drop冒充Sol实现。本机观察：8192-token合成输入上tau1/BQ32相对MPS约2.93–2.96×，但attention向量相对L2约0.217–0.222；约1K-token的真实Qwen输入上没有稳定的低误差加速。**这些不是图片/视频画质百分比，也不是完整请求提速。**

## 实际执行范围

sol_mps_probe.py从Sana `d0c0a4685ab5dc2336d18b7213d85f13def92418`读取并校验两个完整文件，然后仅在隔离import包络执行`preprocess.py`与`metal.py`，不导入模型/包初始化，不改上游。文件和运行范围见[S01](../sources.md#s01)。运行环境为M4 Max、macOS26.6.2、Torch2.14.1、MLX0.32.3、NumPy2.5.4；Torch自带`PrefillAttention.h`的哈希也保留。

八臂：MPS SDPA、MLX fast SDPA、强制全部KV blocks exact的Sol，以及tau−2/−1/0.5/1（BQ64）和tau1（BQ32）。`thresh_type="exact"`只表示阈值统计；tau0不是all-exact。所有Sol臂均禁止dense fallback。

七个点：Qwen公开样例seed42/43的prefill与steady共四点；独立随机seed17/29的8192-token、H8/D128两点；137-token、H2/D64的尾块控制一点。Qwen来自：layer0、step0/1的exposed-output replay，并非未改动的内部kernel捕获，也不覆盖中晚层、多prompt、LoRA或视频。

### 输入与计时边界

- 两框架使用相同可表示BF16值；MPS预备bank为BTHD contiguous，MLX为BHTD contiguous。bank构造/layout转换不计时；这不是两框架从相同物理layout开始的端到端比较。
- 输出统一BHTD contiguous。计时包括host调用、summary/阈值/forward、同步至可读，以及必要的padding、输出恢复与text因果替换；不是纯device timestamp、投影/FFN、完整block或请求。
- prefill：Q=KV=1048，前24个text query只对text做causal SDPA，视觉query看全部KV。Sol结果的text部分被精确替换，额外计算仍计时。
- steady：Q=1024/KV=1048。Sol入口要求等长，因此补24个dummy query并丢弃其输出，使视觉query对齐原来的位置；不将dummy输出当真实token。
- KV sink block0精确保留24个text和40个visual keys；首64-query路由块强制dense，再保留邻居。并非“只保护文字”。
- 两个GPU输入bank在运行前后转FP32后的值hash全部一致。tensor/模型/媒体不随Wiki分发。

每臂5次warmup、12轮旋转且奇数轮反向的配对计时，共672个正式timed calls。每次启动与timing前均取得三个≤5%的device-utilization样本；最终各窗口均为0，warmup后的早期非空闲窗口原样保留并等待重新采样。它们是driver快照，不证明逐PID全程独占。没有重叠运行这七个实验或终止别人的任务。

## Qwen：降低误差通常回到更慢的一侧

下表是四个真实输入点各自统计量的范围，不是置信区间。速度列为每轮`dense_ns / candidate_ns`的中位数；大于1才是更快。

| 臂 | 延迟中位数范围 ms | 相对MPS配对速度 | 相对MLX配对速度 | 全输出相对L2 vs MPS | debug block density |
| --- | --- | --- | --- | --- | --- |
| dense MPS | 1.609–1.738 | 1 | 1.035–1.046 | 0 | — |
| dense MLX | 1.703–1.800 | 0.956–0.967 | 1 | 0.000008–0.000014 | — |
| Sol all-exact | 2.626–2.825 | 0.615–0.637 | 0.640–0.647 | 0.000018–0.000023 | 1 |
| tau−2 / BQ64 | 2.608–2.783 | 0.613–0.644 | 0.643–0.664 | 0.031–0.045 | 0.993–0.995 |
| tau−1 / BQ64 | 2.544–2.692 | 0.643–0.664 | 0.659–0.688 | 0.102–0.120 | 0.925–0.934 |
| tau0.5 / BQ64 | 1.767–1.966 | 0.899–0.924 | 0.923–0.963 | 0.319–0.353 | 0.433–0.446 |
| tau1 / BQ64 | 1.604–1.757 | 0.996–1.041 | 1.044–1.068 | 0.376–0.399 | 0.332–0.339 |
| tau1 / BQ32 | 1.574–1.836 | 0.955–1.116 | 1.030–1.135 | 0.376–0.399 | 0.332–0.339 |

没有给出用户媒体质量预算，因此不把某个tau自动标为合格。尤其不能因为“99.4%的block仍exact”就推出向量误差近零；被近似块的影响并不均匀。

短计时存在轮间波动和首轮长尾，例如seed43/prefill的MPS首轮5.654ms，随后多数约1.65–1.85ms；没有删除这些样本，也未把原因归给某个未单独测量的host/driver环节。该点BQ32的配对ratio中位数1.044，但延迟中位数1.836ms高于MPS的1.738ms：**中位数的比不等于配对比的中位数**。这更不能支持稳定小幅收益或通用BQ32获胜；两个统计量和全部原始数组都保留。

## 长序列：省算可见，但误差仍需模型验收

以下范围来自两个8192-token合成seed，不能外推成真实长视频结果。

| 臂 | 延迟中位数范围 ms | 相对MPS配对速度 | 相对MLX配对速度 | 全输出相对L2 | debug block density |
| --- | --- | --- | --- | --- | --- |
| dense MPS | 20.652–20.659 | 1 | 1.037–1.038 | 0 | — |
| dense MLX | 21.414–21.424 | 0.963–0.964 | 1 | 0.000020–0.000024 | — |
| Sol all-exact | 26.148–26.274 | 0.787–0.790 | 0.816–0.819 | 0.000040–0.000044 | 1 |
| tau−2 / BQ64 | 25.618–25.701 | 0.803–0.806 | 0.833–0.837 | 0.034–0.035 | 0.980 |
| tau−1 / BQ64 | 22.604–22.623 | 0.912–0.914 | 0.946–0.949 | 0.093–0.095 | 0.849–0.851 |
| tau0.5 / BQ64 | 10.539–10.689 | 1.931–1.960 | 2.003–2.032 | 0.197–0.201 | 0.332–0.334 |
| tau1 / BQ64 | 7.222–7.357 | 2.809–2.867 | 2.916–2.973 | 0.217–0.222 | 0.188 |
| tau1 / BQ32 | 7.023–7.061 | 2.926–2.958 | 3.040–3.068 | 0.217–0.222 | 0.188 |

137-token/D64控制中所有路由恰好都exact，各Sol臂中位数0.608–0.813ms，MPS0.256ms、MLX0.281ms，没有收益。它覆盖非整尾块和精确退化，不覆盖“未选尾块的近似补偿”；后者仅有[标准库数学witness](../../examples/README.md)，需要另建实际GPU路由fixture。

density来自独立、非计时的debug block connectivity，包含sink/邻居与steady dummy-query区域；不是token权重比例、实测FLOPs、硬件计数或真实forward分支trace。不能简单用其倒数预测速度。

## 数值资格分成两层

第一层：MLX dense和Sol all-exact与MPS dense全输出相对L2均≤0.01；Qwen还与原BF16 reference对齐。第二层：每head抽8行，用实际forward消费的舍入K centroid/V sum，独立CPU FP64计算相同近似公式；最大行误差跨点约0.00183–0.00258，低于预设0.02。独立CPU selector与修正debug routes无分歧，记录wrapper输出与无wrapper输出逐值相同。

第二层验证“实现符合该近似语义”，**不保证与dense相同**。抽样也不是所有行的证明。D128原debug库存在越界布局，修正仅用于非计时参考库，详见[诊断记录](sol-debug-bounds.md)；生产forward与计时库未改。

## 证据、复现与下一步

保留七份完整report、672个timing样本、配对ratio、输入/源码/环境hash及两份早期拒绝；独立整理器不导入Torch/MLX、不访问GPU、不执行上游。Qwen的seed只取验证过的`source_identity.source_seed`，不能用CLI的synthetic RNG默认值17代替。

后续优先取得真实长序列与中晚层/step fixture，冻结质量预算、保留模态spans，再测试tau/阈值统计、完整block和请求。若需要转换MPS↔MLX或接入产品，应另测bank构造与消费者layout；本轮host-ready数字不授权任何默认后端改变。
