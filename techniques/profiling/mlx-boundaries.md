# MLX真实FFN profiling：output-ready与分阶段drain不是一个程序

<!-- knowledge
id: technique-profiling-mlx-boundaries
type: guide
stage: dit, ffn
operator: swiglu, gemm
backend: mlx, metal
device: gpu
symptom: measurement, launch-bound
semantics: contract-only
representation: bf16, fp16
boundary: producer-consumer
api: public
-->

[原始BF16控制](../metal/case-studies/affine-ffn-cost-and-resident-tradeoff.md) · [clock校准](timestamp-calibration.md) · [本机结果](../metal/case-studies/clock-calibration-and-qwen-profiling.md) · 全部样本

## 从真实source进入，不拿教学shape替代

qwen_ffn_profile.py读取明确的原始checkpoint与已有强制GPU capture，复用BF16有界reader检查选定tensor SHA、capture receipt/activation SHA和FP16 fixture的转换关系。实际是完整Qwen layer0，M1024/H4096/F12288，step1/10/19原始activation；没有重复小样本行、量化替代原权重或下载模型。

保持`[1,M,H]` BF16消费者与base无LoRA身份；只在ignored目录保存源码快照和JSON，不保存另一份权重/activation、不改产品默认。独立CPU FP64抽样和完整BF16 replica比较放在计时之外。Replica不是完整请求oracle，3%局部门槛不授予图像/LoRA质量。

## 两种结果分别保存

| 记录 | 真实计时边界 | 不能推出 |
| --- | --- | --- |
| compiled formal samples | 原始完整FFN函数→`mx.eval` output-ready | GPU逐kernel时间、request时长 |
| staged diagnostic | 每个projection/epilogue之后`mx.eval + mx.synchronize` | 原compiled程序各kernel成本或正式speedup |

每point有2个compiled warmup、4个正式编号样本及2个staged诊断。分阶段强制drain改变了提交、融合、临时张量与等待边界；不能把phase之和与另一程序相除当优化倍率。新native counter的单位校准也没有自动移植到MLX内部queue，因此报告明确`physical_device_timestamps=null`与`native_counter_calibration_applied=false`。

分阶段graph仍执行同一完整BF16 SwiGLU，独立数值通过后可帮助定位下一批实验的优先级，例如在这个staged程序里gate/up投影主导。它不是“已经从真实GPU counter证实GEMM/带宽瓶颈”，也不保证把其中某段孤立优化会降低完整请求时间。

## 人与Agent共用入口

后续第二十三轮补[完整MLX Qwen的公开CB SDK窗口](../metal/case-studies/mlx-command-buffer-windows-and-queue-scope.md)及[原benchmark helper/stream完成控制](../metal/case-studies/mlx-benchmark-evaluation-and-stream-completion.md)。这些是新scope下的真实CB时间，不能回填本页历史staged程序或旧gputrace的null；当前类级覆盖、callback扰动和全设备/逐kernel未知继续分开。

curate_qwen_profile.py只读明确报告，独立检查完整trial集合、来源/数值、host clock、阶段顺序及边界，再分别输出描述性median；失败case和staged时间不获得speedup。

```sh
python3 -B techniques/scripts/host_trace.py "$TC_PROFILE_REPORT" \
  --point layer-0-step-10 --trial 0
```

trace exporter要求明确point与diagnostic trial，输出CPU-host-only lane和µs单位，不补GPU/ANE时间。未启动GUI viewer。源码、数值、profile布局或精度改变时需要新记录；后续应关联真实attention、更多层、LoRA和完整请求，而不是用这个局部profile覆盖所有模型。

只拿到本wiki也能从公开capsule离线重建一个诊断，不需要模型、原始输出或MLX安装：

capsule必须显式选择run，不跨进程合并clock。输入capsule的SHA实际重算；原raw报告SHA只是curator登记的lineage，另标`declared_source_hash_not_recomputed=true`，不冒充已持有完整原文件。
