# LUT不只是“几bit”：真实打包、联合码本压缩与完整resident代价

<!-- knowledge
id: technique-ane-lut-packing-joint-compression-and-resident-cost
type: method
stage: ffn
operator: gemm, swiglu, layout, scheduling
backend: coreml, metal, mlx, cpu-reference
device: cpu, gpu, ane
symptom: precision-loss, storage-pressure, measurement
semantics: approximate, contract-only
representation: fp16, fp32, q4, q8, packed, iosurface
boundary: producer-consumer, cpu-gpu, weight-runtime
api: public, private, reference-only
-->

[ANE目录](README.md) · [12文件固定来源](sources.md#lut-sources-20261017) · 全部模型/样本/独立审计 · [验证记录（过程原件已移出）](../MAINTENANCE.md) · [权重压缩概览](weight-compression.md)

2026-10-17，M4 Max / macOS26.6.2 build25G83、SDK26.5、coremltools8.3.0、MLX0.32.0、NumPy2.0.2。这里从公开压缩API一直检查到**实际weights.bin、解压后权重、完整FFN和GPU消费者**，不把模型变小解释成INT4/INT6乘加。

结果：LUT4 package约为dense的25.1%，但本例最终Y约9.85%–9.88% relative L2，未过原1.5%门槛；相对解压后权重理想结果的执行误差仅约0.11%。自动LUT6能无损表示这组33值权重，最终数值通过，但完整GPU-origin边界仍慢于两个固定GPU基线。联合INT8码本不改4-bit索引，package还略变大。

## 可复用的方法：把低比特表示、误差与驻留成本分开

1. 同时记录请求配置、实际 MIL dtype、码本/索引和文件 payload；按序列化元数据解码，不猜 offset。
2. 源权重参考与 decoded-weight 参考分别验收，避免把压缩误差和运行误差混为一项。
3. 比较资产大小、加载、驻留和完整消费者，保留强 GPU 基线；joint 码本量化不等于矩阵 INT4 MAC。

适用于选择压缩方案和制作导出检查器；下文的特定 bit/group 结果不外推到所有模型。

## 1. 同一个源目标、七种模型、两个GPU基线

沿用[resident FFN](resident-ffn-layout-and-gpu-baselines.md)原始FP16 X/weights，M,K,F,N=512,512,1024,512、seed17/29。借用[精度岛](precision-islands-and-observer-effects.md)已验证的图：公开IO和两个投影为FP16，activation保持FP32。

```text
[G,U] = X · Wgu
H = SiLU(G) ⊙ U
Y = H · Wd
Z = 3·FP32(Y) − 0.5       # 实际GPU消费者
```

本例无bias/residual/LoRA/真实checkpoint。源权重来自完整dense hash配方、每矩阵恰有33个不同FP16值，不是identity/低秩解析捷径；**离散33值是codec正例，不是任意真实权重可无损LUT6的保证**。

七种frozen CoreML模型：dense、LUT4、decoded4、自动LUT6、decoded6、joint4、decoded_joint4。`decoded*`由公共`decompress_weights`生成，保留同样的FP32 activation/FP16投影与IO；不是恢复原权重的神奇逆变换。

LUT4使用原创、确定性的33值histogram加权Lloyd更新24次，最终16个FP16 centroid，避免构造巨大N×16距离临时量或引入sklearn。它不是全局最优KMeans/训练码本，也不能代表所有LUT4算法。LUT6使用`mode="unique"`自动选bit，没有传`nbits`。

## 2. 请求名、MIL dtype、NumPy数组与文件类型分别看

固定[`OpPalettizerConfig`](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/optimize/coreml/_config.py#L729)拒绝unique/custom显式nbits，也拒绝5-bit；支持的集合为1/2/3/4/6/8。实际unique pass按各组唯一值数选择可容纳的位宽，本例两矩阵均为**uint6**，不是按变量名叫LUT4就变成4 bit。

| 层次 | 本例4-bit索引 | 本例6-bit索引 | 不能混同 |
| --- | --- | --- | --- |
| Python materialized数组 | uint8，每index一字节 | uint8，每index一字节 | 这是logical code容器，不是文件真实字节数 |
| 序列化MIL tensor enum | UINT4=35 | UINT6=38 | 不等于下面BlobDataType的编号 |
| MILBlob数据类型 | UInt4=11 | UInt6=13 | 不是INT4/INT6矩阵算术指令 |
| 实际payload | 每byte低nibble先放前一index | 连续6-bit低位先行、跨byte | 仍需码本、header、alignment等 |

原[serializer helper](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/converters/mil/backend/mil/helper.py#L252)根据MIL类型调用`write_uint4_data`/`write_uint6_data`，不是按NumPy uint8直接写整字节。两个原创decoder逐元素核对了所有实际index与恢复出的完整矩阵。

## 3. 真实MILBlob：offset指向metadata，不是payload

[StorageFormat](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/mlmodel/src/MILBlob/Blob/StorageFormat.hpp)定义64-byte file header和64-byte metadata；本机捕获为little-endian、version2。protobuf `blobFileValue.offset`指metadata；metadata中的offset才指raw data，当前writer紧接其后，并都按64-byte对齐。

检查项包括file version/count、sentinel `0xDEADBEEF`、BlobDataType、实际payload位置/长度、unused padding bits及保留字段，不凭扩展名猜dtype。不能拿`metadata_offset`直接当首个压缩index地址。

```text
uint4 logical [1,2,15,0,7]  → 21 0f 07       # 高4个尾padding bit为0
uint6 logical [1,2,3,63,17] → 81 30 fc 11    # 跨byte，尾padding 2 bit
```

新标准库codec与[C++原packer选段](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/mlmodel/src/MILBlob/SubByteTypes.cpp#L20)互相对照；独立curator另用vectorized bit-window decoder，不复用collector标量解包或上游LUT工具。CPU tests含nibble顺序错但仍finite/in-range的负例，不能只靠范围检查验codec。

本批只处理**scalar、per-tensor、cluster_dim=1**。grouped/vector的LUT维度、vector_axis和索引数量是另一种映射，两个decoder明确拒绝而非按本例flatten误读；没有获得其设备性能/精度资格。其他格式/版本/端序也须重开。

## 4. joint=True压的是码本，不是把全部weights换成INT4 MAC

两条链在最终包中真实存在：

```text
LUT4:   uint4 indices + FP16 LUT → constexpr_lut_to_dense → FP16 matrix
joint4: INT8 LUT codes → constexpr_blockwise_shift_scale → FP16 LUT
        uint4 indices + FP16 LUT → constexpr_lut_to_dense → FP16 matrix
```

固定[联合量化分支](https://github.com/apple/coremltools/blob/07ffd86005e385d8eddfd22072836cfb457f7350/coremltools/optimize/coreml/_quantization_passes.py#L1220)只进一步量化LUT输入。本例选择per-tensor INT8；per-channel设置被真实API的NotImplementedError拒绝，因为码本轴不等于原权重轴。`joint=False`控制的op counts与LUT4相同；这项控制只记录计数，不冒充完整payload等价证明。

下表单位byte，为实际整个package；最后一列是实际weight.bin，17 / 29两seed：

| 模型 | package，seed17 | package，seed29 | weight.bin，17 / 29 |
| --- | ---: | ---: | ---: |
| dense | 3,149,444 | 3,149,444 | 3,145,920 / 3,145,920 |
| LUT4 | 790,515 | 790,515 | 786,848 / 786,848 |
| 自动LUT6 | 1,183,891 | 1,183,891 | 1,180,224 / 1,180,224 |
| joint4 | 791,216 | 790,691 | 786,880 / 786,752 |
| 三种decoded控制 | 3,149,348 | 3,149,348 | 3,145,920 / 3,145,920 |

INT8 centroid数据较FP16小，但这里绝大部分已是index流，小码本的节省会被新表达式/protobuf/header/alignment覆盖。seed17有两个shift-scale producer；seed29量化后的共享码本只剩一个producer，被两个LUT引用，weight.bin还能更小，**整个package仍比LUT4大176 byte**。不将不同DAG/共享情况强行假设为“一矩阵一quantizer”，也不把差值全部归因于某个未消融的pass。

这些是磁盘包/compiled资产的尺寸，不是已测的resident DRAM流量、峰值内存、SRAM占用或执行加速。全部模型同时load的warm组件比较也不能给单模型产品内存资格。

## 5. 两个参考分开，才能区分压缩损失和执行损失

原定最终门槛仍是Y的relativeL2≤.015且maxabs≤.02；完整Z精确满足实际Y的消费者公式。没有为LUT4放宽，也不把decoded-reference的好结果拿来过原模型门槛。

| 参考/比较 | 问的问题 |
| --- | --- |
| original FP64 ideal vs actual Y | 压缩加执行之后，是否仍满足原源权重目标？这是采用gate |
| decoded-weight FP64 ideal vs actual Y | 相对该压缩方案实际恢复出的权重，后端执行误差多大？仅诊断 |
| original ideal vs decoded-weight ideal | 即使精确执行，这个权重压缩本身已经改变多少输出？仅诊断 |

三种relative error分母/目标不同，不能简单相加相减。本批七种serialized常量图由递归解析器恢复全部权重，再与公共decompress控制逐值相等；随后独立FP64引用才使用这些权重，不以GPU/CoreML输出为oracle。

所有first/warm/timed中，LUT4相对原目标最大rel为.098529 / .098770，joint4为.098721 / .098933；maxabs也超原.02。其相对decoded理想值的最大执行rel仅约.00109–.00117，压缩-only rel约.09852–.09893。**主要问题在这组码本的近似，不是发现了9.8%的LUT后端计算错误。**

自动LUT6及decoded6权重与原FP16源字节相同，压缩-only误差为0；它们与dense_ne的完整Y也相同，最终rel最大约.001082。这个结果只证明33唯一值正例和当前图路径，不推广为真实checkpoint“6 bit无损”。

## 6. 真实ready GPU X→ready FP32 GPU Z的十臂

原创十臂native client从本库旧client派生，保留原生compiled MLX、MPS两条库级GPU基线及原support shader/严格input alias。其余8臂为dense_ne/dense_cpu和6个compressed/decoded arm；所有CoreML模型row-major FP16 IO，无Conv pack税。

`dense_cpu`只指模型CPU_ONLY，producer/consumer仍真实在GPU。其余CoreML模型请求CPU_AND_NE；本例计划的两投影首选NE、FP32 activation首选CPU，constexpr usage部分未知。计划是预测，不是硬件放置/指令/内部解码时间trace。

timer内包含预测、必要返回copy fallback、GPU消费者提交及完成；timer外才snapshot/文件IO/diagnostic gather。共享backing未采用时会从**实际返回tensor**完整恢复；本批368次CoreML返回均采用backing，没有走copy fallback。不等于CoreML内部零拷贝。

每seed1 first、2 warmup、20 timed trial；每trial同一份fresh ready GPU输入，十臂旋转并反向，每臂在每个位置恰好2次。MLX输入view/各frontend对象均预建；首次前已export、compile、查询plan和load，所以first不是真正cold。

两个native进程串行，均退出后才做本进程CPU oracle；独立curator又在设备进程全部结束后重放。没有机器独占/锁频，外部负载与cache状态未知；诊断在timer外仍可能扰动相邻样本，不包装成纯kernel时间。

## 7. 体积下降与完整边界速度并不成比例

单位ms，20个正式trial的wall中位。†仅保留诊断wall，原目标数值失败，所有合格speed ratio均null。

| arm | seed17 | seed29 | 原目标gate |
| --- | ---: | ---: | --- |
| native compiled MLX | .525521 | .484354 | 通过 |
| native MPS | .573813 | .533167 | 通过 |
| dense，CPU+NE许可 | .764688 | .702250 | 通过 |
| dense，CPU_ONLY控制 | .703292 | .666917 | 通过 |
| LUT4 | .753854† | .672687† | 全部拒绝 |
| decoded4 | .735438† | .676459† | 全部拒绝 |
| 自动LUT6 | .732771 | .689479 | 通过 |
| decoded6 | .746479 | .671521 | 通过 |
| joint4 | .754417† | .680105† | 全部拒绝 |
| decoded_joint4 | .798438† | .696229† | 全部拒绝 |

LUT6相对同trial固定MLX的配对速度比中位为**.71167× / .69116×**，相对MPS为.78912× / .75446×；仍慢于两个GPU基线。相对本轮dense_ne为1.06048× / 1.02596×，但相对同权重decoded6则1.03496× / **.95405×**，方向随seed反转，不能泛称编码LUT一定更快。

以上是每trial固定baseline/candidate比值的median，不是两独立median之商，也不以每trial最小GPU组成不可部署oracle。不跨seed拼配对、不挑更快seed；较快但数值失败的LUT4也不晋升为采用收益。没有归因到特定ANE kernel、整数MAC、频率或内部decompression成本。

460份完整Y/Z共400份timed output；276份过原门槛、184份拒绝（其中160 timed）。分别完整检查120,586,240个Y及同数Z元素。另验证184对完整Y字节一致，包括LUT4/decoded4、joint4/decoded_joint4和dense_ne/LUT6/decoded6关系。

## 8. 独立审计与保留失败

model probe以public-decompress权重为一个控制，独立递归解析own protobuf和MILBlob为另一个控制。native wrapper只复用原helper的配方/MLX assets识别，自己的完整oracle在native结束后执行。

再解析own protobuf、用不同vectorized解包恢复全矩阵；分别计算gate/up的CPU FP64参考，而非复用collector concat路径。两代/两row/三column对全部三种decoded权重组另做完整K/F的math.fsum佐证。全部metrics与原oracle最大差≤2.23e-16。

逐样本`source_reference_sha256`保留原collector concat参考的身份；curator核对独立参考算出的metric，不把该字段重标为独立参考数组的hash。不同FP64求和/结合顺序的末位可以不同，不能用旧hash反证新参考错误或声称两套数组bitwise一致。

完整raw replay明确需要已保留数组、NumPy2.0.2和coremltools8.3的generated protobuf classes；curator不convert/compile/load/plan/predict。19项portable tests只依赖标准库合同，验证codec、协议、资格和已发布统计，**不在raw缺失时冒充重算240M个opaque设备元素**。

保留两项修正：初次observer直接调用LUT op materialization，未递归解析joint的上游palette producer而遇到None，未运行任何native timing；改为公共全图decompress控制及独立递归serialized解析，不修改安装库。首个curator又因严格dict equality拒绝FP64末位差；后续仅对质量metric采用既定1e-10一致性容差，时间/资格/配对ratio仍要求完全一致，**没有放宽最终质量gate或重跑设备**。

prepare阶段另保留共16条临时目录清理的unraisable ResourceWarning，不授予长期生命周期资格；包、weights临时目录和native释放边界参见[版本/所有权审计](coreml-versioned-calibration-and-ownership.md)。

本批raw约913MiB，仍有约33GiB空闲，未清全局cache/旧模型或改产品默认。后续需真实checkpoint与任务质量、grouped/vector/per-channel-scale/QAT码本、其他shape和cold/内存/能耗/完整请求，以及物理设备/低比特执行证据。source/OS/SDK/MLX/CoreML、位宽/码本/轴/布局、weights/输入/bias/LoRA或owner/完成边界改变时重开；不继承这组33值合成实验的质量、倍率或物理算术资格，整体研究继续。
