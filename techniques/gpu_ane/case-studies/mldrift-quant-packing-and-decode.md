# ML Drift低比特：biased packing、UCL解码与受控GPU检查

<!-- knowledge
id: technique-gpu-ane-case-studies-mldrift-quant-packing-and-decode
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [Apple MPP与选路](mldrift-apple-mpp-and-routing.md) · [16文件读本](../sources.md#mldrift-quantization)

2026-10-16，继续固定ML Drift `1b18771f44c0be3d972847b31a80f6dc3d6964b5`，即上一轮LiteRT WORKSPACE声明的依赖。本页进入权重表示和实际解码；没有运行完整ML Drift op、UCL C++编译器、GEMM或模型。GPU部分只执行源选定的Apple half U4解码文本，范围在第5节明确。

## 1. 四个bit数并不是一个“INT4”

[FullyConnected][fc]和[辅助函数][fcutil]区分权重存储、解码后的数值、activation类型以及累加路线。小空间的低比特FC可以读取压缩权重，解成half/float，应用scale/zero-point后浮点累加；另一条路线才量化activation并走整数dot/conv。

所以，模型文件小、权重上传小、运行时仍压缩、实际执行整数乘加，是四个不同结论。U4 decoder输出0到15并不意味着业务权重非负，也不意味着硬件执行原生INT4 MMA。

## 2. 同样四bit，two's-complement与biased编码不同

[权重转换][packing]有不同入口。选定的signed→unsigned路径对原signed值加8；已有two's-complement packed输入的路径则在重排后对每个nibble XOR 8。对合法 `q∈[-8,7]`，两者在所读biased-U4合同下等价：

```text
u = q + 8 = (q & 15) XOR 8
byte = u_low | (u_high << 4)
```

因此signed零对的目标字节是0x88，不是0x00。另一函数 `RearrangeWeightsInt4`保留two's-complement nibble，不能仅按函数名把所有四bit路径的buffer互换。

本库CPU见证穷举256个signed pair，检查上述两种表达；另对25种正OI尾形状构造1×1、单输出group覆盖全部O-slices的I4O4子集，逐值打包/读回。它不是完整OHWI、swap_dims、纹理、所有output groups的上游C++执行。

### Padding也有数学含义

所读默认biased路径用8填充，zero-point=0时解码后代表零。若误填0，会变成-8；若zero-point不为0，8也不再无条件代表业务零。CPU尾形状fixture特意限定zero-point=0，不给其它校准策略取得资格。

## 3. 解码之后仍有scale/zero-point和group索引

[fc辅助函数][fcutil]把unsigned存储与signed解释连接起来：INT8/4/2的range shift分别是128/8/2；所读仿射式为：

```text
w = u × scale + bias
bias = -scale × (range_shift + zero_point)
```

scalar、per-output linear与block/grouped会从不同坐标读metadata。I4O4/O4I4方向又决定一个scale向量如何应用到四个权重向量；不能把一个投影或group的scale挪给另一个。是否显式用FMA也由backend条件决定，不能只凭实数等式推导FP16逐位一致。

小M的selected FC代码先读取压缩16-weight片段，再解码、缩放并累加；grouped时还按src_group循环刷新metadata。Metal/M-series上另有一个窄条件下的workaround同步点，不应把它变成所有shape都要增加barrier的通用优化。上述完整FC算术、group/尾部与同步均未由本轮小decoder执行。

## 4. UCL不是运行在GPU上的解释器

[GPUOperation::AssembleCode][operation]先处理tensor selectors/linkables，再把 `ucl::`系统函数替换成目标语言表达式，并记录所需builtin IDs；[MetalArguments][metalargs]补实际参数与Metal header；[ComputeTask][task]处理类型/宏并创建Metal pipeline。

选定[U4 helper][ucl]在Apple+half时用浮点算术拆nibble：先把uint中的字节重解释成uchar4，转half4并乘1/16；floor得到高nibble，fraction×16恢复低nibble。字节范围仅0..255，这些二进制分数在half中可精确表示。CPU穷举全部256字节检查了这条数学性质；不是证明后续dequant/长dot同样无误。

本轮小client仅按源码已读的三条Metal lowering规则处理该分支：

| UCL表达式 | native Metal含义 |
| --- | --- |
| Reinterpret<uint,uchar4> | as_type<uchar4>，位模式重解释 |
| Convert<half4> | half4数值转换 |
| Init<half> | half标量构造 |

三条规则与同一[util源码][ucl]的Metal分支核对；没有执行完整parser、selector展开、argument分配或所有优化pass。因此“源选定解码分支执行”与“完整UCL编译器执行”分别记账。

## 5. 实测：两进程、完整字节覆盖，但不是GEMM benchmark

从固定util.cc识别两个唯一raw-string payload，保存原始片段SHA与完整源文件/许可证，只对六个位置参数和上述三种表达式做明确替换。自写Metal host client负责编译、dispatch和CPU对照；原始片段、生成shader及编译物仅留ignored outputs，不随Wiki分发。

两进程均在五次GPU占用读数0%后启动。每进程两个variant，每variant256 rows×16输出half；每个输入byte位置均覆盖全部256种字节值。逐元素与整数mask/shift oracle比较，并检查输出两侧guard。

| 实际检查 | 两进程合计 |
| --- | --- |
| GPU dispatch / completed command buffer | 4 / 4 |
| 解码元素 | 16,384，全exact/finite |
| guard元素 | 128，全未改写 |
| mismatch / nonfinite / guard error | 0 / 0 / 0 |
| 运行模式 | MSL3.2，fast math=false，M4 Max |

这验证特定解码表达式及向量lane映射，不验证权重重排C++、metadata读取、bias/LoRA、完整FC、矩阵tile、模型质量、ANE或M5 MPP。没有推理计时或性能A/B，不能据此说浮点拆nibble比bit-shift更快；guard粗计数也不是物理独占证明。

离线新测试还检查receipt类型/计数、精确source/driver/shader绑定与CPU负对照；它们不会重新启动GPU。

## 6. 复算与受控复测

只用本目录即可复算CPU见证与已保存receipt：

显式构建需要固定参考checkout和系统SDK；新运行必须用fresh目录并先准入，不与其它GPU实验并行：

```sh
python3 techniques/gpu_ane/mldrift_u4_probe.py \
  --reference ../references/ml-drift-research-20261016 \
  --build-dir outputs/mldrift-u4-new-build
python3 techniques/gpu_ane/guarded_run.py \
  --output-dir outputs/mldrift-u4-new-admission --timeout 45 -- \
  outputs/mldrift-u4-new-build/mldrift-u4-probe \
  outputs/mldrift-u4-new-build/decode.metal
```

本轮raw artifacts在 `outputs/gpu-ane-mldrift-kernels-round11-20261016`，保存全部准入、child日志、host构建与shader身份。真实采用仍要回到完整消费者、同checkpoint/quant语义和强GPU分母；[MPP与图级选路](mldrift-apple-mpp-and-routing.md)另列。

[fc]: https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/common/kernels/fully_connected.cc
[fcutil]: https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/common/kernels/fully_connected_util.cc
[packing]: https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/common/task/weights_conversion.cc
[ucl]: https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/common/task/util.cc
[operation]: https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/common/task/gpu_operation.cc
[metalargs]: https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/metal/metal_arguments.mm
[task]: https://github.com/google-ai-edge/ml-drift/blob/1b18771f44c0be3d972847b31a80f6dc3d6964b5/ml_drift/metal/compute_task.mm
