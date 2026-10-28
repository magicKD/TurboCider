# Split-Einsum导出与Core ML运行：结构合法不等于语义正确

<!-- knowledge
id: technique-gpu-ane-case-studies-coreml-split-einsum-export-and-runtime
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [原始CPU反例](apple-split-einsum-tail-and-mask.md) · [公共pipeline](apple-coreml-pipeline-lifecycle.md) · [来源](../sources.md#coreml-split-einsum)

2026-10-17，Apple M4 Max、macOS26.6.2/25G83、SDK26.5。接续固定`ml-stable-diffusion@ea2805dc1945be20561c77e5f6d1d9a5a637cda2`，本轮已从原始PyTorch CPU函数推进到**实际MLProgram导出及Core ML预测**。没有改上游源码、运行完整UNet/Swift pipeline或下载checkpoint。

## 本轮新增了哪一层证据

本库小型wrapper调用未改的V1/V2函数，再用TorchScript静态trace和coremltools8.3.0导出；不是执行上游整个转换脚本。固定环境复用上一轮隔离PyTorch2.5.1，NumPy2.0.2；没有安装新依赖或改变项目/系统环境。

主矩阵为4种fixture×2个原始实现×2种precision，共16个toy模型：

| fixture | Q / K | 两组输入 | 用途 |
| --- | --- | --- | --- |
| aligned | 512 / 8 | seed509、521的dyadic随机Q/K/V | 动态数值正控，避免把trace样本当常量 |
| tail1 | 513 / 8 | 两seed | V2是否保留最后一个query |
| tail256 | 768 / 8 | 两seed | V2是否保留后256个query |
| mask | 512 / 512 | Q/K/V完全相同，仅mask从屏蔽改为全0 | 检查输入mask是否实际影响结果 |

B=1、heads=2、dim_head=4。canonical mask输入为`[1,1,1,K]`，wrapper同样reshape为两函数需要的`[1,K,1,1]`；函数体不改。masked输入只有最后key的V非零，blocked答案精确0，unblocked每channel为1到8。

FP32/F16指**请求的图计算配方**：两者外部I/O都显式F32，FP16导出的MIL确实有fp16变量，FP32没有。它不证明设备的每个乘法/accumulator精度，更不是INT8/LUT压缩实验。macOS13固定deployment target和每模型静态shape也不授予动态Q支持。

## 1. 转换与运行分开，警告不能隐藏

导出使用`skip_model_load=True`，并检查返回模型的CoreML proxy为None；这一阶段只保存MLProgram、输入/输出schema、MIL op/dtype统计、trace文本及两组fixture，不编译加载运行时模型。随后才在独立进程通过`MLModel(..., compute_units=...)`执行预测。

安装的8.3源码明确：skip_model_load阻止调用Core ML framework编译/load；compute_precision控制FP16图pass。导出成功只证明该转换流程接受结构，**不会替作者补query尾块或mask语义**。

全部converter stdout/stderr保留两类重要警告：coremltools8.3称其最新测试的Torch为2.5.0，本轮2.5.1未列入其测试版本；源函数的Python boolean判断在trace中固定，不能推广到其它shape。本轮没有升级包、静默patch frontend或更改这些警告。

## 2. 四次实际空闲准入，主实验96个预测

先做2个FP32 aligned pilot，4次预测；再做完整16模型，每模型2组输入，分别以CPU_ONLY、CPU_AND_GPU、CPU_AND_NE在三个串行子进程运行。pilot不混入主矩阵。

四次guard各取得连续5次一秒间隔GPU占用采样，均0%，threshold5%；child均terminal/exit0，期间/结束的粗粒度采样也保存。不是进程级独占或所有瞬时kernel活动的证明；没有关闭其它进程、改验证层或重启服务。

| 允许配置 | 主预测数 | 语义正确 | tail形状错误 | mask语义错误 |
| --- | --- | --- | --- | --- |
| CPU_ONLY | 32 | 22 | 8 | 2 |
| CPU_AND_GPU | 32 | 22 | 8 | 2 |
| CPU_AND_NE | 32 | 22 | 8 | 2 |

共96次主预测：66语义正确、24tail负结果、6mask负结果，417,888个**actual输出值**全部重算。30个源语义错误不能写成“96个正确通过”；exit0表示正/负控制符合预期，不是所有图都数学正确。

raw保存actual、完整FP64 oracle和traced-source值，共5,024,004 bytes；输入fixtures另保存Q/K/V/mask。curator逐文件hash、全值finite、shape、oracle/trace误差复核，同模型/同输入跨三允许配置输出bitwise相同。没有只抽查checksum或几个元素。

## 3. 两个源问题真实穿过了Core ML边界

**tail：**V2的Q513和Q768在导出spec中就成为`[1,8,1,512]`，实际预测也如此；V1保留完整Q。保留下来的前缀与原始traced输出接近，不表示被丢掉的query正确。Core ML可以成功load/predict一个结构合法、但不满足外部完整语义的图。

**mask：**四个masked导出模型的schema都仍列出mask输入；V2并没有因为它无用而必然从接口删除。只切mask、不切Q/K/V时，V1输出L∞变化8，V2变化0；V2 blocked预测相对正确0答案误差8。该配对在两precision、三允许配置均复核。

这不是ANE/GPU硬件算错，也不是compiler改变了原本正确的源语义；原始V2已经缺tail/mask处理，导出/运行忠实保留了它。默认BasicTransformerBlock并不传mask，本轮没有证明默认SD生成失败、完整UNet转换失败或生成图片被裁剪。源函数风险、toy导出、实际模型影响是三层证据。

正向图的最大绝对误差：FP32约`1.58e-7`，FP16约`9.50e-4`。预先声明的诊断门槛分别为`2e-5`与`0.02`，不是生产质量标准。每个result还与同输入traced-source逐值比较，用于区分源语义错误与导出/runtime偏差；没有以FP16更宽门槛掩盖tail或误差8的mask问题。

## 4. 支持、偏好、物理执行：三层必须分开

公开[MLComputePlan](https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n)用于预测前估计cost/resources；[DeviceUsage](https://developer.apple.com/documentation/coreml/mlcomputeplandeviceusage)描述anticipated compute devices。本轮分别用相同allowed policy查询每个loaded模型的plan，取得完整preferred/supported列表：

| 允许配置 | 非空preferred | supported GPU | supported NE | 其它 |
| --- | --- | --- | --- | --- |
| CPU_ONLY | CPU319 | 0 | 0 | 2个identity无usage |
| CPU_AND_GPU | CPU319 | 319 | 0 | 同上 |
| CPU_AND_NE | CPU319 | 0 | 176 | 同上；NE支持全来自FP16图 |

这些是跨16个模型累计的非const op记录，不是物理事件计数。FP32图有143个CPU preferred，FP16图有176个；176个FP16记录在CPU_AND_NE的supported列表中含NE，**preferred仍是CPU**。不能把“CPU+NE预测成功”写成“ANE实际执行通过”，也不能从GPU采样为0排除所有瞬时GPU活动。

Python模型的compute_unit属性只是保存的请求值；它与传入policy一致不是driver placement读回。三种配置输出相同也不证明三种物理设备都跑了同一图。该小shape可能未触及有利的accelerator分派范围，但本轮没扫dispatch门槛/采device trace，不判定系统选择CPU的具体原因。

因此本轮有实际CoreML runtime数值资格，但**没有物理GPU/ANE、M5 TensorOps、GPU+ANE overlap或加速倍率资格**。没有计时Torch对CoreML、FP16对FP32或三设备排名；supervisor elapsed包含load/plan/预测，不是可比较的推理延迟。

## 5. provenance与生命周期边界

[来源清单](../sources.md#coreml-split-einsum)复核前轮两个上游文件hash及固定commit；新增5个安装版coremltools8.3 API源文件指定范围、license/native binding指纹及两份Apple正文。安装源是observed wheel内容，不伪造Git checkout身份或完整环境attestation。

16份MLProgram package成员、trace、fixture、全部runtime数组及stdout/stderr都留ignored outputs并绑定hash。`get_compiled_model_path`只在Python对象存活期有效；本轮没有拷出或attest系统内部compiled artifact，也不把`del model`称物理内存drain。未调用私有ANE接口、清CoreML缓存或修改全局后端。历史第十九轮“未运行CoreML”仍是当轮事实，不回写成这次结果。

## 复算/重放

导出的完整argv、四次admission、owned terminal receipts、所有shape/error/plan见。guard缺计数拒绝、忙时延后；不要直接绕过guard调用predict。后续应扩大合法dtype/head/shape、做真实UNet导出和图像质量，再取得device trace及匹配完整消费者计时；不能用更大的toy绿灯替代这些要求。范围及隔离验证见[第二十轮](../../MAINTENANCE.md)。
