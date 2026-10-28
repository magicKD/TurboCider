# ncnn实际Vulkan Net执行：空输入回归、INT8输出与descriptor边界

<!-- knowledge
id: technique-gpu-ane-case-studies-ncnn-original-vulkan-runtime-and-descriptor
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [前轮源码/CPU片段](ncnn-vulkan-int8-and-apple-policy.md) · [来源](../sources.md) · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17，M4 Max、macOS26.6.2/25G83、SDK26.5。固定ncnn `c15904bb8918187ec100f1ebe708942af83b84be`，本轮从上一轮host stand-in推进到**原始Net/Extractor、原始Vulkan实现和真实MoltenVK运行**。没有改上游CPU/GLSL源码，也没有下载checkpoint、运行真实生成模型、ANE或性能benchmark。

## 1. 隔离依赖，不能把“能编译”当设备可运行

初查本机无可复用Vulkan/MoltenVK/GLSL工具。仅在ignored输出下载ncnn固定源码、其精确gitlink `glslang@780b6a209132dc74e99458f62ebe5fa7e57c41cf`和官方[MoltenVK v1.4.2](https://github.com/KhronosGroup/MoltenVK/releases/tag/v1.4.2)的`MoltenVK-macos.tar`，未用名字含privateapi的资产。后者为2026-07-24发布的59,563,008bytes包，完整SHA256与官方release元数据digest一致。没有安装到Homebrew、系统、项目Python环境或参考checkout。

完整ncnn的4,361个blob与glslang的3,538个blob核官方complete tree；这是字节身份检查，不是7,899份文件人工审计。新增15文件读本范围和两份release包内文档另列。构建仅启用Input/Cast/Packing/Flatten/Quantize/InnerProduct/ReLU/Split/Clip九层，以及原始Net测试工具；不是完整算子覆盖的默认发行包。上游source/内核不改，自己的client/CMake另放独立目录。

### 首个失败来自本库ISA配置，保留而不归咎GPU

本库最初为了缩小CPU构建分支，设置`NCNN_RUNTIME_CPU=OFF`；原CMake按compiler能力将SVE系列选项组合为**全局**`-march=armv8.6-a+fp16+dotprod+sve+sve2+bf16+i8mm+f32mm`。进程已枚举到M4 Max，但在第一个回归结果前SIGILL。只摘取本进程最小OS诊断符号：`try_initialize_global_cpu_info → get_physical_big_cpu_count → optional-input test`，不收录地址/寄存器/完整系统报告。

保留失败binary、flags、cache、owned terminal receipt与日志后，恢复上游默认`NCNN_RUNTIME_CPU=ON`。全局flags不再含SVE，专用CPU对象保留各自flags并由runtime选择；相同原始回归随后通过。证据支持ISA不合适的本地构建解释，不是#6870空输入问题重现、Vulkan shader崩溃或“ncnn不支持M4”。compiler接受某ISA与当前CPU可执行该ISA是不同条件。

构建另保留新target尚未重新configure、client漏加`src/layer` include的失败，以及curator两项range末端超过EOF的拒绝；均为本库构建/记录问题，没有通过修改上游、放宽数值门槛或删除失败日志处理。

## 2. 四次真实空闲准入，三次成功设备进程

首次sandbox内guard无法执行ioreg，空采样即拒绝，未启动client；获准后的每次设备运行仍从guard开始，不绕过占用检查。包括SIGILL初次运行、确认Net回归、INT8 pilot和主矩阵四次，各连续五次一秒间隔采样均0%，threshold5%。三个后续进程terminal/exit0；SIGILL单列，不算成功。粗采样不证明系统级独占或所有瞬时活动。

实际ncnn GpuInfo与MoltenVK日志报告Apple M4 Max、vendor `0x106b`、integrated；INT8 storage/arithmetic=true，cooperative matrix=false。client显式要求该GPU，不接受无设备/软件ICD下的skip。`otool -L`及显式driver路径指向本轮包，含Metal等框架，与上一轮纯CPU观察器不同。

这里已实际经过原始Vulkan队列和Metal driver的Net预测，不再只是CoreML anticipated plan或一个CPU数学模型。但**没有instruction/PMU trace**，不能从extension/storage/arithmetic字段再推导具体packed-dot硬件指令、M5 NAX、ANE或GPU+ANE overlap。

## 3. 原始空optional-input回归不是无设备跳过

本库小观察器include字节不改的原始`tests/test_net.cpp`，先确认GPU count1，再显式调用两方向×lightmode开关×optional是否提供的8组原始函数，每组返回0；随后原始整个main也返回0。main内会再运行同样8组，不能把它说成另一套独立case。

这次实际执行原始Net/Extractor、CPU/Vulkan交界和原始VkCompute clone/传输，没有前轮的byte-Mat或device-property stand-in。整个main的stderr包含故意无效图/参数测试的错误消息，全部保存；不能靠“stderr非空”判失败，也不靠exit0隐去这些负控制。

本轮没有把旧的两条条件换回去制造设备fault；只确认固定修复ref在本机执行这些原始回归。作者PR中Windows/其它GPU测试仍属于作者证据，不混成本机结果。

## 4. 实际INT8消费者：24配置不等于24预测通过

本库client仅调用原始Net加载、原始ModelBin/InnerProduct建图上传、Extractor输入与host输出；没有重写计算shader。生成模型是带tag的signed-INT8权重、全1量化scale、四分之一bias和无activation；外部输入/输出均FP32，输入选小整数，避免把未测的舍入/饱和当已验证。所有参数/model bytes/input/actual数组保存。

6种shape×seed41/53×local-memory开关=24次load：GEMM `(K,N,M)` 为`(8,32,8)`、`(9,9,9)`、`(127,33,5)`、`(129,32,8)`，向量为K120/121、N9。没有shape hint，原实现可能建立多条pipeline；“GEMM pipeline对象存在”不证明向量执行了GEMM。

主结果：**20个实际预测，4个预测前拒绝**。2,612个实际输出值、10,448bytes逐值独立复算，最大误差0，预声明诊断门槛`2e-5`未改。8组完整local-memory开/关配对输出bitwise相同；缺失对照的4份不能编成配对通过。pilot另2预测/512值，不混入主计数。

实际Net选项读回保留了请求的local-memory开关，layer支持Vulkan；相应GEMM pipeline local size在开时8×8、关时由原实现选择。正确性覆盖原始input量化、权重packing/上传、INT8计算、bias/descale及host提取的这组玩具输入，不授予真实校准分布、模型质量或速度资格。

## 5. 重要接缝：view范围不等于allocator容量

[VkMat::buffer_capacity](https://github.com/Tencent/ncnn/blob/c15904bb8918187ec100f1ebe708942af83b84be/src/mat.h#L2379)在该构建按`nstep*n*elemsize`返回view/batch范围；原command用它设置descriptor range。它**不是**`VkBufferMemory::capacity`，更不是底层大VkBuffer整块size。weight allocator会对suballocation对齐，甚至共用大块；不能因此把descriptor外的字节算成shader可安全读取。

本轮在原始Net load/upload后、predict前读到：

| GEMM | 实际weight descriptor bytes | SLM静态读索引span | 处理 |
| --- | --- | --- | --- |
| K9/N9 | 192 | 512 | 两seed的SLM-on均拒绝预测；off实际正确 |
| K127/N33 | 4608 | 8192 | 两seed的SLM-on均拒绝预测；off实际正确 |
| K8/N32、K129/N32 | 与span相同 | 256、4352 | 开/关均实际预测正确 |

这是把前轮“逻辑packed大小风险”推进到**实际上传后的descriptor范围**，仍不是shader/driver fault复现。guard拒绝规则属于本库client，不是ncnn原有fallback；这4份已load/upload但没有predict，不能写成“ncnn自动安全回退”或“24/24图正确”。向量走其它原始路径，不套用未使用的GEMM索引规则。

还需核MoltenVK生成MSL、robustness与actual access trace，才能判定超descriptor的源索引在本环境是否产生真实访问/错误值；本轮没有故意提交这些可疑tail load，没有改验证层、驱动开关或上游barrier。提前return所有tail线程会破坏组内barrier参与，不是可直接采用的修复。

## 复算与后续

23项离线测试只读发布数据，不需要Vulkan、ncnn、driver或outputs。实际重放必须恢复来源匹配的私有构建、显式driver路径和全新输出目录，再由`guarded_run.py`包装client的`pilot/full`模式；完整argv与所有成功/失败receipt在。不要直接运行设备client。

本轮未计时，不使用build/supervisor elapsed做推理倍率。未运行checkpoint/图像、原始整套InnerProduct测试、M5、ANE、真实GPU+ANE并行、cache持久化或故障取消/drain。MoltenVK销毁日志的0MB只是其记账，不能升级为全系统物理内存释放证明。下一步应取得合法tail策略及生成MSL证据，再做匹配完整consumer的local-memory on/off计时与真实模型质量验证。
