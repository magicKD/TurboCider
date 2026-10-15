# 公共Metal profiling：正式计时、capture与counter三份记录

[工具地图](tools.md) · [trace归一化](trace-normalization.md) · [官方合同](../sources/official/metal-counters.md) · [本机实验](../review/gpu-profiling-20261016.md)

## 一次只改一个观察方式

复用已有[run.py](../labs/metal/run.py)、[bench.mm](../labs/metal/bench.mm)与[kernels.metal](../labs/metal/kernels.metal)，不另造推理引擎。相同seed、shape与算法分别运行`none`、`capture`、`counter-stage`。数值与tail guards每次都验证；source快照、binary hash与结束时源码核对随报告保存。

```sh
python3 -B techniques/labs/metal/run.py gemm --m 256 --n 256 --k 256 \
  --output outputs/my-new-gemm-formal/report.json
python3 -B techniques/labs/metal/run.py gemm --m 256 --n 256 --k 256 \
  --diagnostic counter-stage --output outputs/my-new-gemm-counter/report.json
python3 -B techniques/labs/metal/run.py gemm --m 256 --n 256 --k 256 \
  --diagnostic capture --capture-output outputs/my-new-gemm-capture/owned.gputrace \
  --output outputs/my-new-gemm-capture/report.json
```

设备实验必须串行；可外包给[owned实验监督器](../scripts/run_experiment.py)保留超时/错误receipt，不安装Xcode、不改系统设置。captured trial固定为正式编号0的baseline，其他样本也保留但整份capture run都不是正式性能分数。不能把其余未capture样本挑出来混入正式分母；也不能把跨窗口median差异当纯采样overhead。

## 实际能力与失败语义

- 程序先枚举counter sets与stage/dispatch支持位，再选择API。`counter-dispatch`不支持时明确失败，不以stage代替；采不到的counter不是0。
- stage模式在每个compute encoder附加开始/结束sample index；本demo每encoder一个dispatch，边界仍叫encoder/stage，不能普遍称为逐kernel执行时间。
- 所有相关command完成后才resolve shared sample buffer；原值为sentinel、0或非递增时记录无效。原始counter单位未校准，seconds保持null。
- 原有command-buffer GPU时间也必须为完成后的正值；未完成的0时间不再作为有效零耗时通过。
- capture只针对本进程设备对象，程序不启动viewer、不读其他App。raw bundle含buffer与机器信息，只保存在ignored目录。

本轮用GEMM、向量读写、16层elementwise chain建立结构ground truth。graph baseline每层submit/wait，candidate把16个encoder放入一个command buffer，CPU/设备边界不同；这验证host wait/batching诊断流程，不是实现了更快数学kernel或证明了Qwen端到端收益。

## MLX捕获的边界控制

[mlx_capture_probe.py](../labs/metal/mlx_capture_probe.py)提供compiled/eager、新输入、捕获前/停止前同步的独立控制。当前环境中，停止前显式同步才产生可解析结构；仅开始前同步不够。建议显式使用`--synchronize-boundary`并独立解析，不能只检查文件存在或exit0。

这不是否定`mx.eval`等待输出的语义：固定MLX源码的`eval`等待output event，而`CommandEncoder::synchronize`还执行end/commit/wait；实际wheel与源码构建等价并未证明。本轮是capture边界观察，不推翻此前host-output-ready计时，更不把同步扰动塞进正式样本。

后续已经补上独立[时钟校准与规模控制](timestamp-calibration.md)和[真实Qwen host边界](mlx-boundaries.md)；旧raw记录仍不事后补秒数，native资格也不移植到MLX内部。目前没有occupancy/带宽counter、GUI timeline验收、ANE PMU或跨设备overlap证明。
