# Metal实验：能看懂的kernel与完整对照边界

[实验室](README.md) · [MSL](metal/kernels.metal) · [原生驱动](metal/bench.mm) · [构建入口](metal/run.py)

## 共同合同

前五项的独立Objective-C++驱动只链接Foundation/Metal，不依赖生产库/MLX/privateAPI；第六项另有明确MLX/NumPy依赖。前五项FP32输入为确定性合成值，每个AB/BA pair相同输入、不同pair更新输入；4个warmup排除，8个hot默认ABBAABBA。所有输出逐元素比CPU double witness，末尾16个guard防越界；非有限/设备错误返回非零。

pipeline编译/setup与正式host wall分开；host wall含encode/submit/wait，GPU字段是command-buffer timestamps之和。比较单CB与多CB时，总和不包含同样的空隙，不能把它当requestwall。PSO/cache、validation和CPUoracle均不夹在timed span。

## 1. Vector load与尾部

```sh
python3 -B techniques/labs/metal/run.py vector --count 65539 --threads 128 --output outputs/lab-vector.json
```

baseline每thread一个float，candidate以对齐float4读写，最后不足4项走scalar。改变count为1/3/4/5和不同threadgroup，验证不是只对齐尺寸才正确。此例使用独立对齐allocation；处理真实非零offset/stridedview时需扩展guard，不可强reinterpret。

## 2. 真正的8×8 SIMD-group GEMM

```sh
python3 -B techniques/labs/metal/run.py gemm --m 33 --n 41 --k 29 --output outputs/lab-gemm.json
```

baseline一thread一个输出、完整K循环。candidate一个32-lane SIMDgroup负责8×8输出：两次per-lane cooperative loads→threadgroup A/B tiles→barrier→`simdgroup_load/multiply_accumulate`→store；M/N/K尾部零补，不在barrier前早退。

这不是最优GEMM：单SIMDgroup的小tile主要用于理解数据布局。可探索多SIMDgroup、BM/BN/BK、寄存器reuse、doublebuffer、MPP/tensor_ops、packed W的register decode；以当前最快MLX/生产consumer为下一层基线，而不仅击败scalar教学核。

## 3. Fusion与Q/K预处理

```sh
python3 -B techniques/labs/metal/run.py fusion --output outputs/lab-fusion.json
python3 -B techniques/labs/metal/run.py norm-rope --m 33 --k 32 --output outputs/lab-qk.json
```

fusion比较scale+bias→SiLU→residual三个dispatch与一个epilogue；norm-RoPE比较RMSNorm中间tensor+RoPE与单kernel。保留epsilon/interleaved pairs，实际QK gamma、多轴position与BF16边界请按模型扩展。

FP32 compiler contraction/exp/reduction可影响末位，驱动报真实误差，不宣称bitwise。更大tile、不同精度或fusion失败时，先定位触发的shape/semantic boundary，再设计新control，不关掉finite测试。

## 4. Online attention的可读设备实现

```sh
python3 -B techniques/labs/run.py online-attention
python3 -B techniques/labs/metal/run.py attention --m 33 --n 47 --k 16 --output outputs/lab-attention.json
```

设备代码一query/thread，baseline保存N个局部scores，candidate每次处理8个keys并更新online max/denominator/weighted accumulator；causal mask和尾K分块都实际执行。CPUcase另覆盖完全空mask。它是可读oracle，不是声称高性能FlashAttention；真正tiledQK/PV可参考仓库Steel/LTX/Sol kernels。

## 5. Command batching与两队列

```sh
python3 -B techniques/labs/metal/run.py graph --output outputs/lab-graph.json
python3 -B techniques/labs/metal/run.py overlap --output outputs/lab-two-queues.json
```

graph比较16个依赖dispatch分别submit+wait与同一个CB里的16个dispatch。输入/中间owners保持到完成，最后结果验证；不是通过删掉依赖“加速”。

overlap比较独立A/B branches在同queue或两queue，B signal共享event、join wait；同时读实际GPU timestamps的交集。两queue不保证硬件同时执行，交集可能0；这是GPU/GPU实验，不提供物理ANE证据。把同层dependentattention/FFN替换成独立分支会改数学，应先用 [DAG实验](scheduling.md)审图。

## 结果如何迁移

先总结错误/波动、host与device边界，再用真实weights/rows/strides测完整consumer。小shape的单窗口倍率不等于全模型；参考 [本轮真实设备记录](../review/labs-validation.md)。所有historical negative都可在硬件、SDK、布局或分工变化后重新建立候选，保留原失败控制以便判断到底改进了什么。

## 6. MLX affine、原创SG变体与原始BF16控制

[quantized_bench.py](metal/quantized_bench.py)比较packed、resident decoded cache、动态decode表达式、BM32/64、连续加载/row skew；默认是合成FFN，`--fixture`可使用明确的完整真实数组。macOS、NumPy、MLX需已安装；不安装包、不下载模型、不读取兄弟源码。

```sh
python3 -B techniques/scripts/run_experiment.py --timeout 120 \
  --output-dir outputs/my-new-affine-launch -- .venv/bin/python -B \
  techniques/labs/metal/quantized_bench.py --scope gemm --rows 33 --hidden 96 \
  --outputs 41 --group 32 --bits 4 --candidate sg64-skew --input-layout stride2 \
  --output-dir outputs/my-new-affine-smoke
```

小样本是tail/stride/guard检查，不是Qwen速度证据。真实实验固定原始数据与source hashes，保留fused gate/up→SiLU×up→down。数学参考使用canonical FP16 decoded storage，先以CPU FP64见证检查FP32参考；与原BF16组件budget分开，不把近似量化当原模型无损。

原始BF16控制另显式读取checkpoint/capture，只取有界的选定tensor并验证SHA，不写另一份大权重文件：

```sh
.venv/bin/python -B techniques/labs/metal/quantized_bench.py \
  --fixture "$TC_FFN_FIXTURE" --point layer-0-step-19 --bits 8 \
  --baseline bf16 --candidate packed --source-checkpoint "$TC_QWEN_CHECKPOINT" \
  --capture-root "$TC_CAPTURE_ROOT" --output-dir outputs/my-new-original-control
```

这里保留原始`[1,M,H]` BF16消费者；候选输入cast/输出restore计入wall。记录的是host-until-output-ready，不是GPU逐kernel时间；本轮没有物理counter。缓存的setup与resident bytes单列，源码每run快照并结束校验，超时只停止owned进程组。

[affine_decode_probe.py](metal/affine_decode_probe.py)实际检查四个strided operands及decode precision；[ggml_dequant_probe.py](metal/ggml_dequant_probe.py)执行有许可的选定上游模板体。它们是显式诊断，不自动生成模型资格。完整[本机结果与全部失败/确认记录](../review/gpu-affine-20261016.md)说明为什么原型更大tile、cache约4%和格式压缩都不能直接替换原始GPU。
