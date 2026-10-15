# 实验报告：区分真实测量、模拟和资格

[实验室](README.md) · [summarize.py](summarize.py) · [原始测量规则](../profiling/measurement.md)

## 直接读一份设备报告

```sh
python3 -B techniques/labs/summarize.py outputs/my-gemm.json
python3 -B techniques/labs/summarize.py outputs/my-hybrid.json
```

每次只汇总一份同source/geometry/campaign的报告，不跨窗口拼分母；检查实际Metal输出finite/tailguard、明确timing边界、同pair的inputgeneration。输出各variant样本数/中位/最差毫秒，仍`production_qualified=false`。

native W8 integration的整个fixture进程时间不是算子latency，summarizer会拒绝把它纳入operator timing。CPU/SIM模型同样不当硬件report输入。

## 读四个不同的时间

1. CoreML convert/compile/load/firstprediction。
2. 完整host operator boundary（含submit/wait，hybrid还含restore/join）。
3. command-buffer GPU timestamps或CoreML prediction主机span。
4. 完整模型request/process/userclick边界。

它们不相等、可能嵌套；量化staging/共同producer是否计入要明确。CoreML/GPU重叠时，host predictionspan不代表ANEalone；两个GPUqueue时可用同设备时间戳交集，但仍不提供ANE证据。

```sh
python3 -B techniques/labs/run.py profiling-scopes
```

该例子用合成timeline展示spans错误相加；更低功率但更长时间也可能耗更多Joule。系统功率/driver/privatecounter的来源与权限另有合同，工具不自动改电源/进程/swap。

## 下一层实验

把实际source/inputs/精度/缓存、load/cold/warm、actualcalls、失败/重试、样本顺序与独立内存采样绑定。再用 [compare_runs](../scripts/compare_runs.py)作明确normalizedcampaign对比，或现有模型screen的严格receipt verifier。小型lab正常不等于全模型画质/冷启动/小RAM/并发成立。

所有负结果保留rawreport；若改变了source、SDK、归因模型或targetboundary，另建record而非重写旧实验。辅助工具的结构check只证明其承诺范围，最后仍按 [SOP](../guides/sop.md)做人工语义审查。
