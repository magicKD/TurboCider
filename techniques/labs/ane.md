# ANE实验：冷启动、runtime图与真实fork-join

[实验室](README.md) · [Python生命周期/混合驱动](ane/bench.py) · [独立原生INT8驱动](ane/shared_w8.py) · [本地图定义](ane/graphs.py)

## 1. 三种实现边界

| 模式 | 实际实现 | 学习重点 |
| --- | --- | --- |
| `frozen-fp16` | 独立MIL SwiGLU，base weights为图常量 | 省runtime W stage vs 多图/多来源绑定成本 |
| `runtime-fp16` | 直接调用本目录graphs.py，X/Wg/Wu/Wd是输入 | 固定shape跨层复用，但prediction/weightfeature有成本 |
| `runtime-w8a8` | 历史CLI别名，转入本地单INT8 matmul shared IO样本 | INT8 feature、physical pitch、actual output与owner；不是完整FFN |

都是公共CoreML；没有为方便而自动加载PrivateAPI。`CPU+NE`是策略，actualANE placement/overlap/INT8MAC另需独立证据。

独立复制本目录时，把命令中的`.venv/bin/python`替换为已准备NumPy/CoreML等依赖的Python；虚拟环境/Apple SDK不是随本目录打包的源码。FP16和native样本的必要实现均在本目录，不需要TurboCider其他源码或任何兄弟checkout。

## 2. 冷启动要拆到哪里

```sh
.venv/bin/python -B techniques/labs/ane/bench.py --kind frozen-fp16 --compute-units cpu-ne --output outputs/lab-coreml-frozen.json
.venv/bin/python -B techniques/labs/ane/bench.py --kind runtime-fp16 --compute-units cpu-ne --output outputs/lab-coreml-runtime.json
python3 -B techniques/labs/run.py startup-amortization
```

驱动分别记录convert、compile、CompiledMLModel load、first prediction和排除warmup后的samples；CPU staging与prediction/restore/join分别可见。CoreML/OS可能命中缓存，未清缓存，不称严格系统冷启动。NumPy FP64 einsum在计时外提供独立数值oracle，避免仅用同一个backend自比。

两种FP16 CoreML模式均显式exp-SiLU/full-K三projection，避免把sigmoid/exp lowering或K-reduction分段差异错误归给frozen/runtime存储ABI。对真实大shape可另探索segmentation，但需新的控制与recipe。

先同机比较`cpu/cpu-ne/cpu-gpu`，但不同策略load并不证明每个op落在对应芯片。真实prompt/模型source与coldworkload要另外接入；合成FFN没有tokenizer、VAE或媒体。

## 3. 真正的channel fork-join

```sh
.venv/bin/python -B techniques/labs/ane/bench.py --compare-gpu --ane-channels 256 --output outputs/lab-ffn-split.json
```

同一个X和baseweights：CoreML常量图计算gate/up前256channels及对应down列，实际MLX GPU异步算剩余256channels；持久worker同步prediction，主线程join返回partial与GPU suffix，最后materialize。基线是实际完整MLX FFN，ABBA每pair同输入，其他pair刷新输入。

`wall_seconds`包括目标branch/restore/join；`request_boundary_seconds`另加host staging。共同GPU输入producer在边界前已完成。此教学桥接使用NumPy features/restore，有复制税，**不是IOSurface零拷贝方案**；代码明确不把它称最快hybrid。小shape可能更慢，这给出优化overhead的起点，不否定大shape/新边界。

可探索：更大rows、不同share、preparedinputs、真正sharedbacking、nativegraph、FP32join、LoRAcorrection、阶段选择。改变一轴后重测相同优化GPU分母，保留来源/finite/ownership。

## 4. INT8 Python桥接的真实限制与原生解法

上一轮在CoreML9中成功构图后，Python `CompiledMLModel.predict()` 对NumPy int8 features报`value type not convertible`。这是Python adapter限制，不是W8A8图或ANE不能执行的证明。把输入cast为float后继续挂W8A8标签是不正确的对照。

`shared_w8.py`现在只编译本目录的[native_shared_io.mm](ane/native_shared_io.mm)，图来自[graphs.py](ane/graphs.py)：Metal写真实INT8 IOSurface，CoreML消费同一backing，实际输出再由Metal读取。固定M33/K128/N64、两次不同输入generation、FP64 integer-dot oracle与tail guard。它特意隔离单normalized matmul的feature ABI，不声称完整W8A8 SwiGLU/LoRA、rotation或GPU/ANE overlap。

`outputBackings`只是一项请求；原生代码检查实际返回pixelBuffer，若没有采用目标backing则按实际strides复制并报告copy次数，不能把所有执行称为零拷贝。

公共INT8 type gate由macOS26/SDK26明确检查；本地编译使用当前SDK，不依赖产品build脚本。旧的完整生产fixture曾设置deployment26.2，其限制只属于可选integration，不是ANE普遍硬件限制。

```sh
.venv/bin/python -B techniques/labs/ane/shared_w8.py --output outputs/lab-public-w8.json
```

若当前工具环境仍为8.3，可显式使用一个已安装的隔离CoreML9目录：

```sh
.venv/bin/python -B techniques/labs/ane/shared_w8.py --coreml-overlay "$TC_COREML9_OVERLAY" --output outputs/lab-public-w8-ct9.json
```

overlay只改该子进程import路径，不替换项目环境。安装/网络访问是另一步明确操作，脚本不自动下载。macOS/SDK/工具不满足就报告unsupported，不能把跳过计为真实预测通过。

本轮已验证的基础环境是Python3.11.16/NumPy2.0.2；可在具备这些依赖的`.venv`上显式装一个全新overlay（需要网络权限，不是脚本自动执行）：

```sh
.venv/bin/python -m pip install --no-deps --target outputs/my-new-coreml9-overlay coremltools==9.0
.venv/bin/python -B techniques/labs/ane/shared_w8.py --coreml-overlay outputs/my-new-coreml9-overlay --output outputs/my-native-w8.json
```

新机器宜先按环境指南准备依赖或建立独立venv；`--no-deps`仅适用于已确认兼容的基础环境。目标目录应新建，不把overlay混入现有site-packages。

未来Python bridge可能改进；[probe_python_int8.py](ane/probe_python_int8.py)可重新测试一个最小INT8→FP16 cast，不预设一定失败：

```sh
.venv/bin/python -B techniques/labs/ane/probe_python_int8.py --coreml-overlay "$TC_COREML9_OVERLAY" --output outputs/lab-python-int8.json
```

返回0代表该最小bridge成功，3是明确toolchain/bridge限制，1是其他prediction失败；它不证明整个W8A8 FFN或ANE支持。

## 5. 不离开本目录的阅读路线

[混合FFN完整设计](../designs/hybrid-ffn.md) → [CPU normalized W8A8](quantization.py) / [完整LoRA切分](../examples/split_ffn_reference.py) → [本地MIL图](ane/graphs.py) → [本地原生共享IO](ane/native_shared_io.mm)。双bank、graph/source/scale/rotation、失败整块回退均在这些本地说明和代码中；不需要先打开其他仓库。

## 6. 可选产品接入，不是核心依赖

原先更完整的Public W8A8/LoRA/late-failure两个生产fixture保留在[public_w8_product.py](../integrations/public_w8_product.py)，没有为自包含而删除该能力；它需要完整TurboCider，独立复制techniques时不要调用。详见[integrations](../integrations/README.md)。以下链接只用于生产实现对接：

[PublicW8](../../native/backends/ane_public_w8.mm) → [shared Surface/Transfer](../../native/backends/ane_gpu.hpp) → [MSL stager](../../native/backends/ane_w8_kernels.hpp) → [graph emitter](../../tools/coreml/export_runtime_w8a8_ffn.py)。

重点追踪：dynamic code/scales/headroom、hidden rotation/QDQ、down scales恢复、caller backing是否真被采用、late failure完整GPU重算、future banks、input/output alias、取消后reuse。可选两个fixtures的总进程wall是整套正确性检查时间，不可除以calls冒充ANE算子延迟。

这些入口可扩成真实模型实验；当前toy验证只证明指定ABI/所有权/数值边界可运行，不授予端到端速度、普通adapter画质或分发资格。
