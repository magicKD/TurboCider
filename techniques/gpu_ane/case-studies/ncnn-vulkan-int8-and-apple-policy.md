# ncnn的Apple GPU路线：INT8布局、选路与local-memory偏好

<!-- knowledge
id: technique-gpu-ane-case-studies-ncnn-vulkan-int8-and-apple-policy
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [交接与cache](ncnn-vulkan-handoff-and-cache.md) · [来源范围](../sources.md#ncnn-vulkan-policy)

2026-10-17，固定`Tencent/ncnn@c15904bb8918187ec100f1ebe708942af83b84be`，tree `efee7ddb02831417c9b02b0a2e79fbd0ef1f4353`。本轮从原先的Vulkan入口说明深入到InnerProduct的INT8 host/shader、Apple选项策略和跨CPU边界；没有运行ncnn引擎、GLSL、MoltenVK、模型或GPU/ANE benchmark。

## 1. 经Metal执行，不等于原生Metal后端或ANE

固定[README](https://github.com/Tencent/ncnn/blob/c15904bb8918187ec100f1ebe708942af83b84be/README.md)描述CPU/Vulkan后端。顶层CMake的`NCNN_VULKAN`默认OFF；`Option`的`use_vulkan_compute`也默认false。构建能力、请求选项、有效device、具体layer支持是不同门槛，安装在Mac并不会自动跑GPU。

[Apple链接配置](https://github.com/Tencent/ncnn/blob/c15904bb8918187ec100f1ebe708942af83b84be/src/CMakeLists.txt#L282)包含simplevk静态Vulkan链接回退和Metal/IOSurface等框架；[shader宏生成](https://github.com/Tencent/ncnn/blob/c15904bb8918187ec100f1ebe708942af83b84be/src/gpu.cpp#L5658)在Apple下定义`NCNN_moltenvk`。MoltenVK一手[运行指南](https://github.com/KhronosGroup/MoltenVK/blob/main/Docs/MoltenVK_Runtime_UserGuide.md)说明Vulkan叠在Metal之上，并转换SPIR-V为MSL。该指南是本轮保存hash的可变文档，不冒充固定MoltenVK checkout或本机已安装的driver。

因此，这条读到的链是`ncnn算子/GLSL → Vulkan/SPIR-V → MoltenVK/MSL → Metal GPU`，不是CoreML/ANE delegate。不能因为底层最终是Metal，就把ncnn的dispatch、缓存和内存行为等同于直接写Metal的GGML/MLX；更不能从Mac标签推导GPU+ANE协同。

## 2. 低比特的四层合同

[GpuInfo接口实现](https://github.com/Tencent/ncnn/blob/c15904bb8918187ec100f1ebe708942af83b84be/src/gpu.cpp#L2112)将这些能力分开：

| 条件 | 固定源中的判断 | 不证明什么 |
| --- | --- | --- |
| INT8 packed | `support_int8_packed()`直接true | 不等于shader可直接使用8-bit storage |
| INT8 storage/uniform | 查询8-bit storage feature | 不等于8-bit算术或整数dot硬件加速 |
| INT8 arithmetic | `shaderInt8` feature | 不等于packed四元素dot的accelerated属性 |
| packed integer dot快路 | shader同时要求extension、`shaderIntegerDotProduct`和`integerDotProduct4x8BitPackedSignedAccelerated` | 不等于所有shape都更快，也不等于ANE INT8 |

[direct shader](https://github.com/Tencent/ncnn/blob/c15904bb8918187ec100f1ebe708942af83b84be/src/layer/vulkan/shader/innerproduct_int8.comp)的快路使用`dotPacked4x8EXT`；条件不满足时，显式解包有符号4×8-bit并做整数乘加，结果累到`ivec4`，再乘float descale、加bias、activation和存储。这不是“没有该扩展就无法推理”的同义词。

[host](https://github.com/Tencent/ncnn/blob/c15904bb8918187ec100f1ebe708942af83b84be/src/layer/vulkan/innerproduct_vulkan.cpp#L517)还显式禁用该INT8计算配方的FP16 arithmetic及INT16中间布局选项；输出storage另受FP16/BF16等选项影响。模型字节数、输入量化、累加、descale与输出存储不可合并成一个“INT8精度”。本轮没有测量这几个硬件feature的本机实际值，也未执行量化舍入/饱和kernel或验证真实模型校准质量。

## 3. K8/N4 padding不是row-major矩阵

令原权重为`W[N,K]`。原始INT8建图路径：

- `Kp=ceil(K/8)*8`、`Np=ceil(N/4)*4`，分配并清零`Kp*Np`字节。
- 按4个output、4个input组成16-byte块，块内先output lane再input lane；超出N/K的系数写0。
- 每output的descale为`1/(activation_scale*weight_scale[q])`；源代码分别处理scale为0的情况。bias也有N4 padding。

一个逻辑元素的byte index为`(q//4)*(Kp*4)+(k//4)*16+(q%4)*4+k%4`。不能把这块内存直接当`[Np,Kp]`行优先矩阵；也不能只比较有效系数而漏掉padding。

本库CPU probe从原始`create_pipeline_int8`方法内提取**字节不改的packing block**，以自写的有界byte-Mat适配器提供`create/row/reshape`。不是ncnn原始Mat allocator、上传路径或完整operation执行。输入两seed，K覆盖1/3/4/7/8/9/120/121/127/128/129，N覆盖1/3/4/5/7/8/9，共154case。

实际native输出的66,816字节（其中18,198个padding字节）全部复核；输入/输出适配器共19,712个canary字节检查通过，ASan/UBSan运行stderr为空。再从保存的packed bytes独立重建系数，做2,442个小整数dot的Python精确数学复算，误差0；这些**不是shader输出**。故意按row-major错读时148/154个fixture有差异，余6个退化小shape确实不区分，不能抹成154个都检出。

初次提取使用了重复的reshape锚点，选到浮点packing，编译失败，未运行native binary。修正是先限制到INT8方法再取唯一block，未改上游代码、输入、容差或负例；失败脚本/生成TU/编译stderr与terminal receipt均保留。

## 4. 向量与GEMM走不同粒度

`dims==2 && width==K`走GEMM，含batch行及可能的pack4输入；其它形状先flatten，需要时调用Quantize。没有shape hint时会为多条潜在路径建立pipeline，不表示任意运行时shape都合法。

向量路径的分界不是简单的“K>=128”：判断是`Kp/4>=32`，因此K120仍direct，K121因pad到128已走`sum8 → reduce_sum8`。前一pass把八组四字节的点积留为`ivec4`partial，第二pass先整数相加，再descale/bias/activation。增加一个pass与工作buffer是否值得，需要真实shape/consumer计时；源码阈值不能变成跨芯片定律。

GEMM若启用local memory，host固定local size `8×8×1`；[shader选段](https://github.com/Tencent/ncnn/blob/c15904bb8918187ec100f1ebe708942af83b84be/src/layer/vulkan/shader/innerproduct_gemm_int8.comp#L44)用两个`int[8][9]`shared数组、K方向每次8元素、两次barrier复用数据。每invocation积累4×4输出；`+1`padding用于避免bank conflict。这个数据流旨在增加组内复用，不是Metal TensorOps/cooperative matrix或独立ANE执行证据。

## 5. Apple优化PR改变的是偏好，不是实测硬件定律

[PR #6980](https://github.com/Tencent/ncnn/pull/6980)在2026-09-09合入。312行两段patch均读：初稿名为“fast workgroup memory”且覆盖更多vendor；第二段收窄并改名为`prefer_shader_local_memory`。固定实现先排除CPU device，再允许discrete GPU或Apple vendor `0x106b`。

Net加载参数时只会在“不偏好”时关闭`use_shader_local_memory`，不会把用户主动关掉的选项重新开启。它是白名单启发式，不是query得出的带宽/延迟测量，不等于“Apple所有算子都应使用shared memory”。

本轮用35组**假device type/vendor字段**实际执行未改原策略函数，包含Apple-vendor但CPU-type仍拒绝、Apple integrated保留、非Apple integrated不保留的控制；没有调用真实Vulkan枚举。实际`Option`头文件/constructor也参与编译，但两个CPU拓扑helper明确stub为count1和A53/A55=false，不能拿其值报告机器CPU拓扑。

### 尚未验证的tail读地址风险

启用local memory的shader必须让组内线程走到barrier，所以逻辑x/y的早退在K循环后；activation加载有边界条件，但选段里的weight加载未见同样的`gx`逻辑界限。[command dispatch](https://github.com/Tencent/ncnn/blob/c15904bb8918187ec100f1ebe708942af83b84be/src/command.cpp#L1601)会把group count向上取整。

独立地址模型中K9/N9：逻辑packed payload为192bytes，而8个x invocation的索引包络为512bytes；N32则无额外x invocation。这是值得用真实buffer capacity、上传转换和robustness合同进一步核验的线索，**不是已经复现的分配越界、崩溃、错误输出或Apple特有故障**。本轮没有审计完整allocator/MoltenVK lowering，也没有执行该shader；给所有tail线程提前return又会破坏barrier参与，不能据此草率“修复”。

## 怎样迁移到TurboCider研究

优先迁移分层能力表、packed原格式oracle、padding全值检查、明确的向量/GEMM候选与缓存身份；不要整体引入ncnn来借一段kernel。性能实验应比较local-memory开/关的同权重/输入/输出ABI，覆盖121附近和N4/N32尾部，计入quantize、partial、上传/下载及完整consumer，再和优化native Metal基线比较。任何未来设备执行必须先过本库idle guard；本轮只CPU片段，不存在GPU占用准入或硬件性能结论。
