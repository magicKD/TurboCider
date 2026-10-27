# MLX benchmark的完成合同：105次构图不等于105次GPU工作，默认同步不等于全stream同步

<!-- knowledge
id: technique-metal-case-studies-mlx-benchmark-evaluation-and-stream-completion
type: case-study
stage: prefill, decode, pipeline
operator: scheduling, layout
backend: metal, cpu-reference
device: cpu, gpu
symptom: measurement, launch-bound
semantics: contract-only
representation: fp32
boundary: cpu-gpu, producer-consumer, request
api: public, reference-only
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 固定版本实现案例，供理解机制与复核边界；学习先读[方法导航](../README.md)，数据保留规则见[证据说明](../../MAINTENANCE.md)。

[GPU导航](../README.md) · [完整模型CB窗口](mlx-command-buffer-windows-and-queue-scope.md) · [既有host边界](../../profiling/mlx-boundaries.md) · stream原记录 · [来源](../sources.md#mlx-cb-profile-sources-20261017) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，第二十三轮。执行固定MLX0.32.3读本中两个**原始benchmark helper函数**，并用真实公开CB窗口验证它们的完成边界；另用foreign stream的异步矩阵链证明，默认stream的同步返回并不意味着另一个stream已完成。不是复写一个timer后声称运行过上游benchmark，也不是报告“上游计时器有bug”。

## 1. 两个相似helper有不同的调用方合同

[`benchmarks/python/time_utils.py`](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/benchmarks/python/time_utils.py)：

| 原函数 | warmup / timed | 完成由谁保证 | 返回/打印单位 |
| --- | --- | --- | --- |
| `time_fn(fn, ...)` | 5 / 100调用 | 每次`mx.eval(fn(...))` | 打印每调用ms |
| `measure_runtime(fn, ...)` | 5 / 100调用 | caller的fn自行完成 | 返回每调用ms |

所以同名“runtime”不能替代阅读fn。纯lazy函数只构建图却未求值，传入measure_runtime可得到很小数字，但不是GPU算术时间；fn内部已经等待则可以。原[comparative读本](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/benchmarks/python/comparative/bench_mlx.py)的matmul wrapper本身有eval和10个逻辑matmul，外层bench又按100次函数调用计时间；函数调用数、逻辑矩阵节点、CB提交和物理kernel不同。

本轮只执行time_utils的两个固定函数，没有运行完整comparative/compile CLI或跨PyTorch benchmark，也不继承其速度表。原function body只从显式固定Git blob受限提取到本地执行，Wiki仅发布原创客户端与hash，不分发上游body。

## 2. 真正执行原helper的四个控制

相同已准备好的256×256 FP32输入：每行常数`(row mod17)/32`，权重各元素1/256，独立数学结果仍为原行常数。每case只将最后一个完整输出保存并全元素核查；没有把105个未保存输出都称作独立oracle通过。

每个原函数都实际调用fn **105次**。两个独立反序进程，四种组合的原生CB观察一致：

| 原helper / caller | source调用数 | timer scope里的真实CB数 |
| --- | --- | --- |
| measure_runtime / 仅返回新lazy matmul | 105 | **0** |
| time_fn / 仅返回新lazy matmul | 105 | 105 |
| measure_runtime / fn内eval结果 | 105 | 105 |
| time_fn / fn内eval结果 | 105 | 105，而非210 |

lazy+measure_runtime报告约0.000398/0.000417ms每调用；该窗口没有捕获到这个已绑定类/queue的CB提交，最后保留的图是在timer之后另行eval/验证的。这是“量到了构图”的直接反证，不是GPU快了数百倍。记录集合0也不证明整机GPU空闲，未观测的framework/queue仍未知。

其他组合约0.128–0.169ms每调用，只作带observer的诊断值，不能拿其相除为库/设备性能。640条完整CB行包含准备、warmup、计时和后验验证。benchmark scope捕获**5warmup+100timed**的CB，不是只捕获100次计时子集；不能将其sum直接除100冒充原函数的GPU均值。

为什么最后一行不是210？[`eval`实现](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/transforms.cpp)对已scheduled/evaluated结果走wait/early-return，不再重新构图执行。双eval字面出现不等于双CB。反过来，GPU stream的`mx.synchronize`调用会走encoder同步/commit路径；不要把二者混为同一种barrier。

## 3. 不同stream的真实反证

[Python stream binding](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/python/src/stream.cpp)说明：`mx.synchronize()`默认只同步默认device的默认stream。指定stream则走其encoder。[`CommandEncoder::synchronize`](https://github.com/ml-explore/mlx/blob/64ea011cb65f14d9ce2737e60db9a4ae91ed7441/mlx/backend/metal/device.cpp)提交当前CB、等待该CB，不会自动遍历所有其他stream。

本轮每个完整模型进程另执行同一**合成FP32**反例，不把它当Qwen延迟：2048×2048的8个依赖matmul放在foreign stream，结果按行常数数学应保持不变。先`mx.async_eval(result)`，再同步默认stream，最后`mx.eval(result)`及同步foreign。12份完整矩阵都与独立全元素参考精确相同。

on四进程用前置校准把默认/foreign分别关联到不同进程内queue。真实记录一致显示：

1. 默认stream同步已经返回。
2. foreign queue的GPU窗口仍在其后结束。
3. 与结果依赖关联的eval及foreign同步返回后，全部观测到的foreign窗口已结束。

错误等待的host段约0.110–0.282ms，后面的正确结果等待段约25.82–30.63ms；前后不是两套同工作量benchmark，**不能相除称加速**。后者还含异步提交/编码的剩余成本，不是纯GPU kernel时长。该反证证明默认等待不覆盖该foreign工作，不证明两个硬件引擎物理并行或整设备busy百分比。

## 4. 输出ready、scope与应用语义

`mx.eval(result)`等待其结果依赖，不是全进程所有不相关stream的全局drain。要覆盖side effects、KV、通信或其他输出，必须让实际需要的状态进入完成合同；只返回一个与foreign工作无依赖的常数不能替它建立barrier。反之，不必给每层强加全局同步来“修复计时”，那会测另一个提交程序。

[完整Qwen实验](mlx-command-buffer-windows-and-queue-scope.md)明确eval完整logits和所有KV状态，其144个payload在stock/off/on一致；采样、tokenizer、D2H/cast、日志drain和文件另分scope。这个host-ready边界可以辅助定位CPU gap/提交开销，仍不是整个生成请求或shader-only速度。

CB记录在commit时赋scope而不是在图构建时赋scope。异步工作若跨scope提交，不能只凭函数调用时的标签归属；本反例把整个async→两种等待放在同一scope，再按已绑定queue和真实时间分析。队列ID也不跨进程拼接。更复杂服务需要显式请求/依赖token关联，而不是给时间线补一个看起来合理的名称。

## 5. CPU合同与复查

CPU合同有lazy计数、错误queue、缺失/重复/不可用窗口与disabled未知负控；普通tests不导入MLX，不执行原上游body或GPU。真正原函数由显式helper客户端在新目录运行，SDK窗口由本轮进程内诊断helper采集，完整身份/许可和局限见[窗口页](mlx-command-buffer-windows-and-queue-scope.md)。

复查条件：helper/caller实现、warmup/iters/单位、source/wheel/native文件、eval返回状态、stream/输出依赖/副作用、scope与consumer改变。需要性能结论时应重新定义同一完成边界和原始工作量，保存未观察配对样本及完整质量；不能用“同名timer”“有synchronize”“数字更小”替代这些证据。
