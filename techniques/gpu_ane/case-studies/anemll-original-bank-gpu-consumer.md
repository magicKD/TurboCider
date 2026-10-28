# anemll原始MoE block：packed bank的真实GPU消费者

<!-- knowledge
id: technique-gpu-ane-case-studies-anemll-original-bank-gpu-consumer
type: case-study
-->

> 历史案例摘要：用于学习机制、取舍与失败边界；不随库分发实验数据或验证环境，不代表当前实测。

> 实现案例：按机制学习，不按研究批次采用。文中的次数、环境、命令和结果是固定历史实验；旧采集脚本依赖当时的输出布局，不是当前一键复现入口。可运行的离线学习见[方法实验](../examples/README.md)，原始来源和数值证据保留在文内链接。

[专题导航](../README.md) · [前轮I/O合同](anemll-flash-mlx-banked-streaming.md) · [本轮摘要](../../MAINTENANCE.md) · [来源](../sources.md) · 实际数据 · 计划 · [验证（过程原件已移出）](../../MAINTENANCE.md)

2026-10-17。本轮把前轮CPU字节I/O推进到**原始完整MoE block组件的MLX GPU消费**：router、top-k、packed专家up/gate/down、score combine、shared expert和gate tail均实际执行。不是完整语言模型、包导入、checkpoint、文本生成、性能或物理ANE验证。

## 1. 源码身份和不能隐藏的依赖边界

继续使用官方 `Anemll/anemll-flash-mlx` commit `f86c0682a5dcd44eee1f1c090c588e4f209b62c9`，原30个Git blob保持不变。现有环境mlx/mlx-metal均0.32.0、NumPy2.0.2，缺少`tokenizers`，直接导入完整model模块不可行；**未安装依赖，也未提供假Tokenizer或替代GPU kernel**。[S1][本轮来源](../sources.md)

客户端从原`model.py` AST原样选取6个完整节点：`ModelArgs`、两种stage stats、`_apply_sparse_moe_tail`、`Qwen3TextMLP`和`Qwen3TextSparseMoeBlock`。保留完整class/method body、装饰器和future annotations语义，逐节点AST hash与原始文本范围绑定；globals只绑定真实MLX、原expert_io和原upstream_switch_layers，以及标准库/NumPy。**这是原始选定组件执行，不是未修改完整包已成功运行。**[S1][S2]客户端

现有两distribution的10个关键组件重新核指纹，MLX官方tag读本另固定 `7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247`；新增3个源文件读段。版本和组件hash不证明安装wheel完整build来自该commit。没有改上游、参考树、系统、依赖或产品后端。[本轮来源](../sources.md)

## 2. 比较的是同一函数的三种权重供给

同一配置、层和step的三arm使用**相同packed专家、router、shared权重、输入、K和数学参考**，不是不同模型或不同K互比。计划数据

| arm | 原始实现路径 | 驻留/换入 |
| --- | --- | --- |
| `packed_resident` | 原`_bank_index_switch_mlp`和完整8专家executor | 启动从原loader预加载，每层`mx.eval`后才复用CPU staging；按expert ID计算 |
| `python_bank` | 原`_slot_bank_switch_mlp`的Python owner/LRU路径 | 每层3slot，原普通C loader读miss，原9-array executor安装，再按slot ID计算 |
| `native_bank` | 同一方法的原NativeSlotBankLoader路径 | C逐层owner/LRU、跨层共享CPU staging；miss_views安装到每层独立3slot executor |

两层在每step交错执行，三arm观察顺序轮转。bank路径没有开启oracle/direct/device-hit-lookup、compiled-bank或compiled-tail旁路；正常top-k→host lookup与miss处理都在原路径上。原routing callback捕获实际ID/score，引入额外host同步；**这是一组正确性观察，不是自然调度下的速度benchmark**。[S1][S2]

每次调用完整`block(x)`后`mx.eval(output)`，读回并保存全部FP32结果；全部消费完成后才继续/关闭loader。旧输出保留原MLX对象，不另作GPU snapshot clone；只授予选定逻辑值保留，不授予物理backing身份、zero-copy、并发owner或取消/drain资格。数据

## 3. 有效低比特fixture，而非opaque字节或checkpoint

矩阵为q2/q4 × D/F=128/128、256/1024，group64、8专家、K2、bank3、2层，activation/output逻辑FP32，scale/bias保存BF16位模式。q2和q4分别生成各自合法函数，**不是同一原始浮点checkpoint的两种量化质量比较**。生成器计划

| 配置 | 每专家原始packed大小 | source允许request split4的有效split |
| --- | --- | --- |
| q2，D128/F128 | 15,360 bytes | 1 |
| q2，D256/F1024 | 245,760 bytes，15页 | 4 |
| q4，D128/F128 | 27,648 bytes | 1 |
| q4，D256/F1024 | 442,368 bytes，27页 | 4 |

expert文件低位到高位打包uint32，非零scale/bias使用可精确表示的BF16位模式。独立CPU decoder按lane解码 `w = code*scale+bias`，以保存的F32 router/shared权重计算FP64 top-k归一化、三投影SwiGLU和shared tail。capture又从**磁盘packed bytes**独立逐字段解码，264个FP64参考向量（88个step-layer配置×routed/shared/full）与保存参考完全相同，不调用MLX quantize/dequantize来充当oracle。[S3]生成器捕获器

MLX固定源码会将affine输出类型与activation提升；本实验FP32 activation使scale/bias在GatherQMM输入中转成FP32。**BF16 metadata存储不是已验证BF16 MAC或低精度累加。**[S4]

## 4. GPU实际观察与独立重跑

M4 Max、macOS26.6.2/build25G83、SDK26.5、Python3.11.16。pilot、primary、confirm共3个进程均经guard，15个preflight样本全0%，缺计数仍拒绝。某次完整矩阵的自动权限审查在CreateProcess前超时，确认目标路径不存在后按工具允许重试一次；**没有把该失败当已启动GPU或额外准入**。全部实际owned子进程terminal0。数据

- primary：`4配置×2层×11step×3arm = 264`个完整输出，50,688个FP32值/202,752字节全查。最大绝对误差约`1.19e-8`；全输出relative L2及扣除shared后的routed branch预算均通过，所有值finite。
- confirm：新进程同264输出/50,688值全查，264份主输出逐字节复现；不把它和pilot重复累计成新的主样本数。
- 三arm在88个相同配置/层/step分组中全部bitwise相同，不只是都低于宽松误差阈值。
- 每个完整进程24个旧输出逻辑值保留控制；pilot另66输出/8,448值及6个owner控制，单列不加进primary。实际数据

上述是有限生成权重组件，不覆盖checkpoint outlier、长上下文、完整attention/KV、tokenizer、真实生成质量、prefill batch、多请求或所有tail/shape。`mx.eval`是正常完成边界，不是GPU故障/drain、物理ANE placement或设备独占证明。数据

## 5. 命中/换入历史不是只看最后输出

每层路由依次为`[0,1] / [0,1] / [0,2] / [2,3] / [3,2] / [4,0] / [0,4] / [5,6] / [7,5] / [1,7] / [1,7]`。原router实际ID和归一化score被callback核对；本库解释用LRU reference不参与控制原cache。[S1]合同

每条bank层流的miss数实际匹配`[2,0,1,1,0,1,0,2,1,1,0]`，11调用/22专家请求中9miss、4次full-hit。primary两bank×两层×四配置共176层调用，**64 full-hit、80单miss、32双miss**；352专家请求、144miss。这些是逻辑cache计数，不是SSD读次数/物理带宽或完整token全层命中率。数据

这个跨层结果给前轮补上了消费者证据：CPU staging确实共享，但按原顺序完成安装/计算时，每层GPU bank仍计算对应专家。它不推翻前轮borrowed view被覆盖的观察，也不允许把后续安装延迟到staging被下一层覆盖以后；此种异步协议本轮未运行。[前轮合同](anemll-flash-mlx-banked-streaming.md)

## 6. API名字、实际数学与负索引边界

`gather_qmm`是矩阵级gather API，名字不保证每次用matrix-matrix核。所读tag的`GatherQMM::eval_gpu`根据M、B、sorted、vector_limit等选择；本例单token、K2、未排序hint，源码预测走gather_qmv分支。**这是固定源码预测，不是安装wheel的driver/kernel trace，更不是M5 NAX或ANE观察。**[S3][S4][S5]

同一源码会检查index dtype为integer，然后转uint32；所读构图函数未见逐元素范围验证。因而前轮的`-1`不能想当然理解成安全的Python“最后一个专家”：转换后是`4294967295`。本轮始终只提交有效0..S-1/0..E-1索引，**不执行该无效GPU路径，也不报告越界/安全结果**。迁移时应保留显式failed/miss/hit和consumer前完整slot admission。[S4][前轮失败合同](anemll-flash-mlx-banked-streaming.md)

## 7. 得到了什么，仍缺什么

已从“CPU原loader返回字节正确”推进到“原始router＋完整MoE block消费同packed数据，三种供给方式在hit/miss/层切换下结果一致”。但没有由此获得native比Python快、cache比resident快、真实SSD上限、prefetch overlap、最优bank容量或完整模型资格。数据

向dense视频模型借用时，下一步仍应固定实际checkpoint/LoRA、访问序列与内存上限，比较single/double bank和lookahead的完整请求、峰值内存/带宽与质量；本轮没有把dense的全权重访问偷换成MoE稀疏命中。没有更改默认产品后端。[前轮迁移问题](anemll-flash-mlx-banked-streaming.md)

## 固定入口与复核

- [S1：原model与完整MoE class](https://github.com/Anemll/anemll-flash-mlx/blob/f86c0682a5dcd44eee1f1c090c588e4f209b62c9/flash_moe_mlx/model.py#L647)
- [S2：原packed executor](https://github.com/Anemll/anemll-flash-mlx/blob/f86c0682a5dcd44eee1f1c090c588e4f209b62c9/flash_moe_mlx/upstream_switch_layers.py#L297)；[原I/O wrapper](https://github.com/Anemll/anemll-flash-mlx/blob/f86c0682a5dcd44eee1f1c090c588e4f209b62c9/flash_moe_mlx/expert_io.py)
- [S3：MLX固定affine packing/gather API](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/python/src/ops.cpp#L4660)
- [S4：GatherQMM类型、shape和index转换](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/ops.cpp#L5417)
- [S5：Metal GatherQMM dispatch](https://github.com/ml-explore/mlx/blob/7a1d4f5c12ac82f4b4d0a6e71538d89ca0605247/mlx/backend/metal/quantized.cpp#L1555)

新标准库合同测试不导入MLX/NumPy，不构建C或使用raw outputs；显式GPU客户端只能通过空闲准入运行。全部fixture、原始AST文本、C构建物、actual、准入和owned receipt只保留ignored `outputs/gpu-ane-flash-bank-round31-20261017`；整理数据和原创客户端随专题分发。隔离验证不授予未测引擎/硬件资格。[验证范围（过程原件已移出）](../../MAINTENANCE.md)
