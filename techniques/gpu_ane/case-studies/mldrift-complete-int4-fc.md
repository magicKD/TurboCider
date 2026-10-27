# ML Drift：原始 INT4 FC、完整 UCL 与 Metal task 实测

<!-- knowledge
id: technique-gpu-ane-case-studies-mldrift-complete-int4-fc
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [packing/解码前置](mldrift-quant-packing-and-decode.md) · [隔离构建](mldrift-isolated-build-and-dependencies.md) · 完整输出 · [源码范围](../sources.md#mldrift-int4-fc)

2026-10-16，在 Apple M4 Max 上实际运行固定 ML Drift `1b18771f44c0be3d972847b31a80f6dc3d6964b5` 的完整 external-INT4 FC operation。两个独立进程、12 个用例、438 个逻辑输出，离线用独立整数公式逐值重算，全部 exact。这里的“完整”指选定 FC 的 packing→UCL assembly→原始 Metal task→readback，不是完整模型、图编译器覆盖率或 LiteRT 引擎运行；没有推理计时、MPP、ANE 或速度排名。

## 与第十一轮的实质区别

第十一轮只执行两个源选定 raw-string，六个位置参数和三种表达式 lowering 由本库 wrapper 提供。那份历史记录的“完整 FC/UCL 未运行”仍然正确，不改旧 manifest。本轮没有抽取 decoder 再手工拼 FC；自有 host直接链接未改动的原始库：

```text
合成 signed INT4 值（用 int8 容器承载）
  → GetFullyConnectedInt4WeightsDesc + GetTensorDescriptorForWeightsLayout
  → CreateFullyConnectedExternalWeights（含 scalar scale/zero-point 和 bias）
  → operation.AssembleCode（完整选定 FC 的 UCL/selector 解析）
  → 原始 MetalSpatialTensor + ComputeTask.Compile/SetSrc/SetDst/UpdateParams
  → ComputeTask.Encode → command Completed 检查
  → 原始 ToDescriptor/DownloadData → 独立逻辑输出比较
```

没有运行 `gpu_model_builder` 的整图分区/选路、自动调优或上游完整测试套件。直接调用 FC factory 不证明任意真实模型都会选择该核。自有 wrapper 负责输入、顺序、错误检查和保存输出；没有替换 packing、FC kernel、UCL resolver 或 Metal compiler 接口。

## 为什么这些边界有学习价值

原始 factory 的 [`fully_connected.cc` 941–1030](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/common/kernels/fully_connected.cc#L941)根据真实权重 descriptor 创建 operation，设置有无 zero-point/bias、block size，并按原始参数顺序注册资源。不能拿一个自己定义的 row-major byte buffer 直接冒充该 ABI。

原始 [`testing_util.cc` 87–119](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/common/task/testing_util.cc#L87)先做 batch/layout 检查，再调用 `AssembleCode`；[`metal/testing/test_util.cc` 58–131](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/metal/testing/test_util.cc#L58)展示 Compile→bind→UpdateParams→Encode→readback。自有 host 按实际 API 组织相同核心顺序，但没有复用其测试封装、也不宣称上游 tests 通过。

[`ComputeTask` 120–220](https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/metal/compute_task.mm#L120)还要完成 Metal arguments、保留向量类型替换、宏定义、pipeline 和 grid 更新。只保存先前 UCL raw string 无法覆盖这些集成边。本轮使用默认 direct binding，不测试 argument buffer/ICB。每进程明确提交六个 FC command buffers；上传/读回可能有额外辅助操作，不能说整个进程只有六次 Metal 操作。

## 用例与数学合同

H=1，输入/输出均 buffer storage；B=1 使用 HWC，B=2 使用 BHWC。每种形状分别运行 F32、F16、F32F16：

| B / W / I / O | 每精度逻辑输出 | 检查意义 |
| --- | --- | --- |
| 1 / 1 / 7 / 5 | 5 | 输入/输出通道都不是 4 的倍数 |
| 2 / 2 / 33 / 17 | 68 | 多 batch、多 spatial 位置及两侧尾通道 |

每进程为 `(5+68)×3=219` 个输出。fixture 定义：

```text
x[row,i] = ((row*3+i*5)%17 - 8) / 32
q[o,i]   = (o*7+i*3)%16 - 8
scale = 0.125; zero_point = -1
bias[o]  = (o%5 - 2) / 4
y[row,o] = bias[o] + Σ x[row,i] * (q[o,i]+1) * 0.125
```

`q` 是有符号 4-bit 语义，原始 packing 决定 biased nibble/物理布局。这里是低比特**权重表示与融合浮点解码**，不是 INT4 activation 整数矩阵乘。scale/zero-point 的符号不能丢；bias 必须在最终输出上保留。

host 使用 double 参考和逐值原始数组；另外用整数分子/256 计算 oracle，不接受只回报 max error 或 pass。F32 的 max_abs/relative_l2 门槛均为 `1e-5`；F16/F32F16 分别为 `0.05/0.005`，两项都必须通过。所有 actual、native oracle、指标、shape 顺序和 finite 均重算。

实际六种用例的 max_abs 与 relative_l2 都为 0，两进程输出与对应 shader hash 一致。这些数是 dyadic、范围小、计算短，不能推广为任意 FP16 累加 exact、任意分组量化准确或真实模型质量无损。

离线 negative control 中，漏掉 zero-point 的 max_abs 为 `0.02734375`，低于宽松的半精度绝对门槛，但 relative_l2 超过 `0.005`；只检查一个绝对误差门槛会漏检。漏 bias 的 max_abs 为 `0.5`。这是独立 CPU 反例，不是故意破坏上游后取得的 GPU 故障复现。

## GPU 准入、原始输出与 shader 身份

两次运行均由 guarded_run 在五次一秒总 GPU 占用采样均为 0% 后准入；child 正常退出 0，supervisor 确认进程结束。粗粒度总占用不是物理独占、全程无外部干扰或 GPU+ANE overlap 证明。

原始 logs、五次构建记录、六个/进程的生成 `.metal` 文件仅留 ignored `outputs/gpu-ane-mldrift-fullop-round12-20261016/`。保存全部 438 个 actual/oracle、每个 native shader SHA、admission/stdout/child receipt、原始库构建和当前 curator 的绑定。它是可审阅的本地证据，不是硬件 attestation。

shader dump 来自 Compile 后的 `GetCode()`，没有 `ucl::`，但仍含 `MAIN_FUNCTION`/`__local` 宏；原始 task 在编译时传 `kernel void ComputeFunction`/`threadgroup` 定义。因此不能以 dump 缺少字面 `kernel void` 判定未完成原生编译。它也不是包含所有编译选项、驱动状态和宏展开结果的完整可独立重建 archive。

## 离线复查与设备复测

已有 artifacts 的核验不启动 GPU：

重建见[隔离构建页](mldrift-isolated-build-and-dependencies.md)。显式复测必须使用新目录、串行运行且保留准入失败，不直接绕过 guard：

```sh
python3 techniques/gpu_ane/guarded_run.py \
  --output-dir outputs/mldrift-fc-replay-admission --timeout 180 -- \
  outputs/gpu-ane-mldrift-fullop-round12-20261016/workspace/bazel-bin/local_probe/full_fc \
  outputs/mldrift-fc-replay-shaders
```

本轮没有外部 OOB guard buffer、fault injection、cancel/drain、multi-slot reuse、冷/热性能、强 MPS/MLX 分母或真实模型。`waitUntilCompleted` 和正常退出不补齐这些资格；[第十二轮范围（过程原件已移出）](../../MAINTENANCE.md)分别记账。
