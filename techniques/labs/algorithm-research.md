# 算法研究配套实验：数学 witness 与设备探针分离

[算法报告目录](../algorithms/README.md) · [研究来源](../algorithms/sources.md) · [本轮结果](../algorithms/experiments-20261016.md)

[algorithm_research.py](algorithm_research.py) 是原创 Python 标准库实验，不复刻整套 TeaCache/PISA/Sol/SVG/SVDQuant 引擎，也不下载模型。对应报告在 catalog 中登记的是**支持性机制/证据合同练习，不是该算法的完整本地实现**。

```sh
python3 -B techniques/labs/run.py algorithm-evidence
python3 -B techniques/labs/run.py residual-reuse
python3 -B techniques/labs/run.py token-merge
python3 -B techniques/labs/run.py smoothquant
```

新 case 检查：constant-key block 与部分尾块的零阶补偿精确退化、all-exact 退化、遗漏块 mass 的错误负控、高方差 centroid 近似失效、非均匀时间线性外推、转折处预测劣于 replay、共享 K 均值平移的 softmax 不变性。Python float 不是 GPU dtype 模拟，数值小于1e-12也不是生成质量门槛。

既有 residual-reuse 演示真实调用计数与条件/step gap 重置；token-merge 演示 mass 与位置保护；SmoothQuant 演示重参数化与量化的区别。它们都不是媒体基准。

[MLX 探针](../algorithms/mlx_probe.py) 是可选设备实验，需显式 `--run`、已有 MLX 和 GPU admission 检查；不被 `labs/run.py all`、Wiki 检索或默认 unittest 自动调用。它比较 dense、全长局部 mask、实际局部 KV 切片和 all-exact chunk。不同 approximation 的输出误差与相同局部语义的数值等价分开记录。

来源采集使用 [source_probe.py](../algorithms/source_probe.py)；可读 source/commit/hash 不代表执行过项目。更新算法、后端、shape、校准数据或消费者时重开实验，不从来源名称继承性能资格。
