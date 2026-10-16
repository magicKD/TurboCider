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

## 7. 完整图内tiling：合成smoke与真实fixture分开

[tiling_graphs.py](ane/tiling_graphs.py)是原创公共MIL图，不复制上游未文档化bundle writer；[tiling_bench.py](ane/tiling_bench.py)比较compiled fused MLX与实际Core ML输出恢复/可选suffix/join。默认只在显式执行时运行小型合成数据，依赖已安装NumPy、MLX、CoreML和macOS；不安装包。

```sh
python3 -B techniques/scripts/run_experiment.py --timeout 120 \
  --output-dir outputs/my-new-tiling-launch -- .venv/bin/python -B \
  techniques/labs/ane/tiling_bench.py --runtime-weights --layout channels \
  --tile-k 41 --tile-n 63 --output-dir outputs/my-new-tiling-smoke
```

这个smoke含不整除K/N tail，但不是大模型性能结果。`--fixture`读取明确的本地通用fixture.json及校验过的完整数组，`--point`选择独立discovery/confirmation输入；不能tile/repeat几行模拟1024个真实row。产品捕获和原checkpoint导出在[integrations](../integrations/README.md)，核心实验不导入产品代码。

`--route coreml`算完整FFN；`serial/hybrid --ane-channels ...`计算配对gate/up prefix与对应down partial，GPU算剩余channels，最后join。`--baseline serial --route hybrid`是同图serial/hybrid配对诊断，不能把这个分母的收益称全GPU加速。

固定全输出与独立FP64抽样门槛失败时保留样本、停止该候选、不输出speedup。四次warmup与全部ABBA样本留下；不通过筛掉失败、降低容差或混合权重source“救回”候选。硬件放置始终unverified，plan不能替代物理trace。

只使用新目录。监督器有独立进程组与显式timeout，保留失败日志；停止进程不自动证明设备drain，后续不复用该进程的buffer。runner检查磁盘余量与图大小，frozen可能占用多份权重副本，余量不足不自动删任何旧数据。权重常驻、准备与每层staging的边界须分别报告。

2026-10-15已完成[真实Qwen layer0 FFN的这组实验](../review/qwen-ffn-tiling-20261015.md)，不是仅toy通过；但尚未达到产品/所有层/LoRA/端到端资格。

## 8. 图搜索与私有 e5rt 的显式研究入口

[search-qualification](search_contracts.py)是纯标准库的身份/接受逻辑教学case；[机制页](../ane/graph-rewrite-search.md)把结构提示、编译资产、数值/性能资格拆开。实际[ANEForge执行](../integrations/README.md)依赖调用方的固定上游副本，明确属于可选integration，不是本目录核心依赖或默认device selector。

[本机记录](../review/aneforge-20261016.md)包含13组真实同步实验、SiLU失败缩减、exp重写、INT8/reduction负结果和活owner的view/copy/poison控制。全部first/warm/timed检查由[独立FP64 oracle](ane/precision_oracle.py)与固定门槛约束；约3×的host-fed小图局部观察不能借给真实Qwen GPU生产者、GPU/ANE并发或完整请求。

## 9. 私有PMU采集不等于数值执行

[原创native客户端](ane/pmu_probe.mm)和[显式CLI](ane/pmu_probe.py)只依赖本地代码/Apple SDK；默认仅准备小MIL，`--device`才实际执行。完整physical pitch、输出poison、代际输入、完整FP16输出SHA与unload控制见[81次本机记录](../review/ane-pmu-20261016.md)。mask0/2/15的所测路径没有取得counter，不称“0活动”。

检索入口和CPU负控见[PMU技术卡](../profiling/ane-pmu-lab.md) / [pmu.py](pmu.py)。真实counter还需要独立生命周期、单位、聚合和workload缩放证据；不能用其他项目的性能数字填空。

## 10. 真实BF16 GPU分母与私有FFN分担

[完整消费者机制](../gpu-ane/private-ffn-boundary.md)说明为什么serial与hybrid都必须和完整GPU比较。[七个真实source运行](../review/qwen-private-ffn-20261016.md)保留全M/H/F及GPU余下channels，交接/转换/join全部计入；固定2048channels后在step19/10和24-group及pre-arm排空控制中确认局部warm收益，不是只测小matmul。

核心[稳定公式/切片](silu_domain.py)与[独立FP64 witness](ane/qwen_witness.py)在本目录内；[ANEForge真实数据入口](../integrations/README.md)是明确可选upstream integration。它尚不是实际shared IOSurface、全请求或新产品后端默认。

## 11. 实际共享surface：先证明接缝

[shared_surface_probe.py](ane/shared_surface_probe.py)默认只准备原创FP16 affine MIL，`--device`才编译并运行[本地native客户端](ane/shared_surface_probe.mm)。只需标准库/Apple SDK，无产品、upstream checkout、NumPy或MLX依赖；使用新ignored目录和owned进程监督器，见[完整复现](../gpu-ane/shared-surface.md)。

[四个新进程120次完整消费者](../review/private-shared-20261016.md)实际比较request input/output surface身份与Metal contents，全部input/ANE/consumer hash由[独立标准库witness](shared_backing.py)重建。128/192-byte实际pitch不外推；故意按紧密stride读取仍全finite却错误，上一代真实输出也被新generation完整拒绝。shared arm只取得外部feature边界copy数0的资格，runtime内部未知，不是物理ANE/并行、MLX adoption或真实Qwen提速。

后续[MLX adoption实验](../review/mlx-adoption-20261016.md)另取得两个导入模式的实际alias与view/deleter证据，覆盖padded和逻辑8MiB合成范围；same-dtype Copy alias、已求值结果过期和materialized snapshot分开。它没有ANE调用，仍不能与前面的独立小图拼成真实FFN共享路径。

## 12. 真实完整shared FFN五臂，不继承toy资格

[新完整记录](../review/qwen-shared-ffn-20261016.md)实际跑通原始BF16 layer0的GPU→private→MLX完整消费者，port为`[1,1,1024,4096]`而非channel/sequence toy布局。GPU pack、同client copy控制、完整suffix/restore/join均计时；560次full/prefix/FP64门槛与后两轮264个actual-part BF16位级join见证分开标识。

[CPU RNE/合同](real_shared_ffn.py)与[独立curator](../scripts/curate_real_shared_ffn.py)不运行设备，真实[入口](../integrations/README.md)需明确原始资产与依赖。长确认约1.15×是warm FFN replica消费者，不是完整block/request、LoRA或物理ANE资格；文件clone与load selector也不能当内部无copy/无编译证明。
