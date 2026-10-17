# 2026-10-16 算法研究实验：数学反例与 MLX attention 对照

<!-- knowledge
id: technique-algorithms-case-studies-experiments-20261016
type: case-study
operator: scheduling
backend: cpu-reference
device: cpu
symptom: measurement
semantics: contract-only
boundary: request, producer-consumer
api: reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

[算法入口](../README.md) · 完整整理后数据 · [协议](../evaluation.md) · [配套实验](../../examples/README.md)

结论：**稀疏语义不等于省算；实际少算也不等于质量可接受。** 两轮合成实验支持把算法选择、kernel 开销和输入分布分开检查，没有产出新的图像/视频模型资格或通用最快方案。

## 范围与环境

本机 Apple M4 Max、64 GiB、macOS26.6.2、MLX0.32.0，BF16 Q/K/V、B=1/H=4/D=64。没有加载模型权重；不是对 TeaCache、PISA、Sol-Attn 或 SVG 的完整复现，也没有真实视频质量结果。

两次新 Python 进程：seed17，每点4轮×5次；独立seed29，每点4轮×7次。每轮轮换/反向顺序，候选与 dense 交替，保留全部1152次计时；另有 warmup、数值控制，不混入计时数。此处是 warm component host wall，含建图、切片/拼接、提交和 `mx.eval`，不是纯 GPU timestamp，也不是冷启动/E2E。

开始及每个 workload 前分别采3个 IORegistry utilization 样本，Device/Renderer/Tiler均0%；只读进程快照看到已存在的低CPU Python服务，无明显活动推理进程。读数只能说明未发现明显竞争，不证明逐PID GPU独占。8192点结束即刻读到78–83% Device，保留原值，不能当作新的竞争证据或热降频归因。没有持续功耗/频率/逐进程GPU trace。

## 四个执行臂

- dense：一次 optimized `mx.fast.scaled_dot_product_attention`。
- full-local-mask：完整 N 的 SDPA，仅传入 `abs(q-k)≤128 OR key<16` mask；不声称跳过全部被 mask 的矩阵计算。
- sliced-local-sink：query chunk128，只传需要的局部 K/V 切片和不重复的16-token sink，再施加同一局部 mask。
- all-exact-chunk：query 分块但每块仍看全部 K/V；用于验证执行拆分与新输出确实正确，不是近似算法。

静态 mask/index 构建与materialize不计时，两个local臂一致；切片/concat在timed调用内。动态 selector/cluster 不存在，不能用该结果推导动态稀疏的收益。峰值内存未测。

## 速度结果

倍率均为同进程、同输入 dense 中位数 / candidate 中位数；小于1表示更慢。两个seed不合并成一份置信区间。

| N / 输入 | dense ms，seed17 / 29 | full-mask倍率17 / 29 | sliced倍率17 / 29 | all-exact chunk倍率17 / 29 |
| --- | --- | --- | --- | --- |
| 512 independent random | 0.220 / 0.213 | 0.924 / 0.924 | 0.769 / 0.743 | 0.876 / 0.803 |
| 512 position clustered | 0.210 / 0.219 | 0.916 / 0.913 | 0.776 / 0.723 | 0.841 / 0.796 |
| 2048 independent random | 0.960 / 0.569 | 0.918 / 0.894 | 1.090 / 1.132 | 0.839 / 0.912 |
| 2048 position clustered | 0.947 / 0.532 | 0.915 / 0.919 | 1.178 / 1.169 | 0.867 / 0.864 |
| 8192 independent random | 5.457 / 5.424 | 0.922 / 0.923 | 1.101 / 1.519 | 0.774 / 0.796 |
| 8192 position clustered | 5.447 / 5.434 | 0.923 / 0.923 | 1.002 / 1.065 | 0.761 / 0.792 |

full-mask 在全部点比 dense 慢；512的实际切片也慢，局部执行的粒度/提交成本超过节省。2048的clustered点两轮约1.17×，但8192的方向与大小有明显分布/轮次差异；尤其 random 的1.10→1.52×不能选较大值发布“稳定1.5×”。2048绝对时间两轮差异也很大，未采集频率/counter，不能断言原因。

## 相同局部语义与对 dense 的近似误差是两条轴

all-exact chunk 相对 dense 的误差在本次记录为0。sliced 对 full-local-mask 的相对L2均低于1e-4，说明这两个执行方式实现了同一个局部策略；不代表局部策略与dense等价。

| N | local相对dense L2，random17 / 29 | local相对dense L2，clustered17 / 29 |
| --- | --- | --- |
| 512 | 1.10386 / 1.10336 | 0.00417 / 0.00414 |
| 2048 | 2.59115 / 2.58946 | 0.00537 / 0.00544 |
| 8192 | 5.41920 / 5.43130 | 0.00772 / 0.00774 |

random 的 dense 输出平均抵消、能量较低，相对L2容易很大；完整数据同时保留 max-abs，不能仅用相对值排序。clustered 是人为按位置分组的高相关QK/V，不是某个真实DiT分布，不能据此宣布生成损失小。这里只证明同一个窗口假设对不同输入可能很不同。

## CPU 数学控制与反例

原创 algorithm_research.py 使用Python float，不模拟BF16，也不测速度。

| 控制 | 本轮观察 | 支持的结论 |
| --- | --- | --- |
| constant-key、块长4+尾块2的0阶补偿 | maxabs约1.11e-16 | 正确计入有效token mass时的精确退化 |
| all-exact | maxabs0 | exact/approx集合边界可核对 |
| 故意去掉块mass | maxabs约0.2000 | centroid不计块长度会改变softmax |
| 相反key -6/+6被合成均值 | relativeL2=1 | 补偿并不保证高方差块正确 |
| 非均匀坐标的线性预测 | 17，等于oracle | 导数的时间间隔与anchor正确 |
| sin转折处的一阶预测 | 误差0.7736，replay为0.06783 | 高阶/趋势预测不总优于复用 |
| 全部K共享均值平移 | maxabs0 | 实数softmax row-shift不变；不证明量化无损 |

新增离线回归覆盖三个seed、非法shape/nonfinite、bad history、sink不重复、显式GPU opt-in、来源commit/hash/URL突变拒绝、缺失计时样本拒绝。测试绿色只证明这些范围，不证明上游模型实现正确。

## 如何复现

复核seed29时改seed/repetitions并使用新输出。GPU busy时脚本拒绝该轮而非继续测速；等待空闲再运行，不删除其他进程/缓存。`.venv`仅是本库已有环境的示例，可替换为已安装MLX的Python。

curate_experiments.py 校验完整矩阵、原始样本、数值与script hash并去除PID；随Wiki保留，没有权重/私有prompt。source收集与数学实验不产生GPU负载。

重开条件：真实模型QKV、不同chunk/tile、fused sparse Metal、dynamic routing、long-run空闲/热状态监控、更多独立进程与完整媒体质量。不能拿本次合成结果自动选择产品preset。

## 交付检查

另实际重新读取固定checkouts，77个源文件SHA-256与30条来源身份全部匹配；这强于只校验JSON格式，但仍不等同于审计全部源码、在线URL可用性或复现论文性能。整理后JSON保留两进程全部计时；同局部语义最差relativeL2为3.514e-5，真实图像/视频质量仍未测试。
