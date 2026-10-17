# 真实 Qwen QKV 的稀疏性：4224行CPU抽样，不是生成加速基准

<!-- knowledge
id: technique-algorithms-case-studies-real-qwen-sparsity
type: case-study
stage: dit
operator: attention
backend: cpu-reference
device: cpu
symptom: precision-loss, measurement
semantics: approximate, contract-only
representation: bf16, fp32
boundary: producer-consumer
api: reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · [模型sparsity地图](../methods/model-sparsity.md) · [Sol/PISA](../methods/piecewise-attention.md) · [实验室](../../examples/README.md)

2026-10-16第二轮。使用已有公开固定样例的Qwen-Image-2.1胶囊，**不加载权重、不生成媒体、不提交GPU工作、不报告速度**。结果在，执行代码为qwen_sparsity_probe.py，只读curator为curate_qwen_sparsity.py。

## 来源、抽样与oracle

来自[原attention验证](../../metal/case-studies/qwen-attention-replay-and-empty-row.md)的seed42/43，layer0、step0 prefill和step1 steady。先用已有receipt/文件/逻辑tensor hash验证，再读取BF16逻辑副本；是actual block arguments上的exposed-output replay，不是未改变内部kernel的物理捕获。

每个场景取全部32heads、33个均匀target queries、全部1048有效K/V，共4场景4224行；prefill的24个text queries不在抽样中。head_dim128，scale=1/sqrt(128)，target查询都看全KV。没有中晚层、其他prompt、编辑reference或LoRA。

CPU FP64 dense结果对胶囊BF16 reference的最差row relativeL2为0.002346以内；另8行用独立Python `math.fsum`完整K/V验证，最大abs约1.45e-15。它们验证局部数学/来源，不证明所有图像和视频质量。

第一版NumPy matmul虽输出finite且reference匹配，却产生浮点状态warnings，未用作发布结果；没有归因成硬件/输入错误。发布版改为非BLAS `einsum(optimize=False)`，divide/overflow/invalid严格拒绝，并做独立fsum复核；不隐藏warnings后沿用原版本。

## token稀疏和block稀疏的尾部差异

以下是每个query/head理想排序后，覆盖99% dense attention mass所需数量，未加额外sink约束；是离线诊断，不是可部署选择器。

| seed / phase | token数 median / mean / p95 / max | block数 median / mean / p95 / max |
| --- | --- | --- |
| 42 prefill | 1 / 177.72 / 929 / 1012 | 1 / 5.05 / 17 / 17 |
| 42 steady | 1 / 177.30 / 927 / 1012 | 1 / 5.03 / 17 / 17 |
| 43 prefill | 2 / 165.51 / 921 / 1010 | 1 / 4.91 / 17 / 17 |
| 43 steady | 2 / 164.85 / 916.5 / 1007 | 1 / 4.85 / 17 / 17 |

block为连续64tokens，最后块只有24tokens，共17块。许多行非常集中，不代表所有head都能极稀疏；p95行仍要绝大多数tokens和全部块。不能用median=1得出“模型可固定丢掉99%”。JSON保留逐head的紧凑统计，避免平均值隐藏敏感head。

## 同预算比较路由与补偿

两种路线：`mean-key-row-query`用当前单query与每块mean-K打分；`offline-dense-mass-oracle`直接用已算出的dense块质量排序，仅作oracle。两者都强制block0 exact以保护24个文字keys，同时多保留40个视觉keys。

单query路由比真正query-block共用selector更细，并非完整Sol/PISA复现；oracle的dense cost更不可能拿来当免费runtime。固定k=4时，实际exact token density随尾块选择约23.3–24.4%，不能简单用4/17代表真实token密度。

| seed / phase | mean-K keep/drop平均row L2 | mean-K 0阶平均 / 最差row L2 | dense-mass oracle 0阶平均 / 最差row L2 |
| --- | ---: | --- | --- |
| 42 prefill | 0.1730 | 0.1489 / 1.7802 | 0.1060 / 0.8816 |
| 42 steady | 0.1693 | 0.1438 / 1.7623 | 0.0993 / 0.8028 |
| 43 prefill | 0.1523 | 0.1298 / 1.1000 | 0.0954 / 0.8250 |
| 43 steady | 0.1480 | 0.1255 / 1.0205 | 0.0920 / 0.8056 |

0阶分支对非exact块使用mean-K、V-sum与valid-length进入共同softmax分子/分母，exact/remainder互斥。此组平均误差改善，但仍有很大的最差行误差；甚至oracle路由也不能消除0阶模型误差。**保留质量较高不等于输出向量误差必然最小**，dense-mass oracle不是输出误差的全局最优选择。

报告还包含k=1/2/8/17、keep/drop和0阶两种执行数学；all-exact k17控制均在1e-12以内。relativeL2可能受小输出范数影响，不能把上表直接解释成RGB损失百分比；也未设定/通过新的整模型质量门槛。

## 可以据此继续研究什么

优先headwise预算或保守dense回退，区分误差来自proxy漏选还是块内高方差补偿；不要直接固化本样本的逐head最优值。需要补中晚层/step、更多prompt/reference/seed，之后才冻结route policy。

真正Metal比较还要计算summary、选择、QK/PV、投影/restore及完整消费者，确认GPU空闲再测。本轮CPU“oracle route更好”不证明更快；本轮所需全量logits在分析中已算出，所以所有策略都**不是稀疏推理耗时测量**。

raw tensor/capsule不进入Wiki；机器可读统计仅包含来源hash、几何、抽样indices与聚合误差，不含激活或私有prompt。该报告只约束所测layer/steps，不能变成通用Qwen、视频模型或产品preset结论。
