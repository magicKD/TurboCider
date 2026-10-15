# 真实 Qwen FFN：私有 prefix 分担与完整消费者确认

[总目录](../README.md) · [机制与边界](../gpu-ane/private-ffn-boundary.md) · [ANEForge来源](../sources/repos/aneforge-rewrite-search.md) · [全部样本/失败](qwen-private-ffn-evidence-20261016.json)

2026-10-16，M4 Max/64GiB、macOS26.6.2、Python3.11.16、NumPy2.0.2、MLX0.32.0/实际architecture applegpu_g16s；ANEForge固定`896c4943...`，只使用之前隔离构建的e5rt shim。原checkout、checkpoint、捕获、产品runtime和App均未改。

## 源、配方和计时

真实layer0、M1024/H4096/F12288，原BF16 checkpoint和forced-GPU 20-step轨迹中的step1/10/19；全部row，不缩小或重复。直接复核原始gate_up/down tensor SHA与capture/input SHA，不从FP16派生物倒推原始权重。GPU完整分母是相同公式的compiled BF16 FFN replica，所有输出对保存reference误差0；不是宣称测到了未改变的原Transformer大kernel内部时间。

只把配对prefix转FP16给private图，使用stable exp(-abs) SiLU、完整H输出partial；GPU保持原始BF16余下channels，FP32 join再还原`[1,1024,4096]` BF16。输入在共同ready边界前准备，转换/NumPy桥接、同步、上传和join都计入；权重prepare/emit/compile/load、验证单列。常量常驻，不是runtime-weight streaming或零拷贝。

每个短测为3arm×(first1+warmup2+measured6)=27个完整调用；每个24-group确认为81个调用。共7组实测/21个配对比较/297次调用，另保留1个frontend错误。三arm用六种平衡排列；six-pair/24-pair bootstrap只作探索性不确定性，不声称长期独立样本或正式显著性。

## 实际结果

| 选择/确认 | full GPU中位ms | serial中位ms | hybrid中位ms | GPU/hybrid配对中位比 |
| --- | --- | --- | --- | --- |
| step1，512channels discovery | 21.717 | 29.415 | 22.046 | 0.9855×；未赢 |
| step1，1024channels discovery | 21.673 | 29.741 | 20.927 | 1.0335×；探索性 |
| step1，2048channels discovery | 21.733 | 31.465 | 19.283 | 1.1262×；固定后确认 |
| step19，2048channels确认 | 21.715 | 31.610 | 19.355 | 1.1216×，区间[1.0609,1.1264] |
| step10，2048channels确认 | 21.712 | 31.428 | 20.080 | 1.0818×，区间[1.0644,1.1190] |
| 新进程step19，24-group长确认 | 21.747 | 31.365 | 19.421 | 1.1196×，区间[1.0977,1.1256] |
| 新进程step19，24-group，pre-arm验证排空 | 21.700 | 31.435 | 19.309 | 1.1241×，区间[1.1167,1.1284] |

2048是预设512/1024/2048 grid中观察最好的值，不是全局最优。后两timestep与确认没有再改share/质量门槛。约1.12×是本FFN边界的倍率；两次24-group确认的中位wall下降约10.7%/11.0%，不是整次生成下降12%。serial→hybrid分别约1.614/1.629×，保留但不借作GPU加速比。

最后一轮另在每个arm计时前显式排空上一轮计时外的FP32数值验证，避免把diagnostic转换残留混入下一个arm。candidate的生产转换/同步/join仍在timer内，GPU计算和数值门槛不变；该更严格边界的收益仍可见。此前记录保持原source hash与协议，不回填这个marker。

## 数值与activation域

所有first/warm/timed均过固定relative L2≤3%、cosine≥0.999：完整4M元素对BF16 reference、完整private prefix对BF16 component reference，以及两个独立FP64抽样。全输出候选最大L2约0.002887；独立full sample最大约0.006079；完整prefix最大约0.007687；独立prefix sample最大约0.011327。不能只用最终误差小来遮住坏prefix。

FP64 witness是4个实际row/4个输出column，每个row用完整K/F；不是全输出FP64或图像质量。该witness与domain代码在全部数值成功run之间未修改。长确认额外检查原BF16 GPU projection的全部2,097,152个prefix gate值：约[-2.484375,2.09375]，abs>11为0。它是本输入/表达的域见证，不是其他层、LoRA、所有输入或private FP16内部值的上界。

## 成本、来源和负控

private逻辑权重512/1024/2048分别12/24/48MiB；各arm共存GPU逻辑权重576MiB。长确认native emit/compile/load约0.682s，warm每call只节省约2.33ms；没有清系统cache，不能将这笔setup当所有机器的冷启动常数。完整请求可能因按层编译/加载、额外驻留、LoRA或真实producer/consumer而失去收益。

host事件记录显示private API与GPU suffix调用窗口相交，但不与别的process或native GPU时钟相叠。没有物理ANE counter/placement/overlap资格，也没有把逻辑8MiB输入/16MiB partial接口量标为DRAM流量。

最初frontend直接把BF16对象交给NumPy，报PEP3118 buffer-format错误；已有private图编译但没有计入推理样本。原件/hash/source snapshot保留，修复为计时外FP32验证读回，不改GPU计算精度或预算，也不标为ANE/GPU不支持。较早512成功记录未采MLX binary assets，后续run有前后hash；这项差别不回填。stat时间戳和机器路径不发布，只保留raw hash与前后检查结论。

只读curator重新核对原BF16 checkpoint的两tensor、捕获及fixture内容，标准库按真实NPY row pitch重组native FP16 gate/up/down prefix SHA，检查MIL/blob/shim与所有source snapshots、顺序、gate、时钟边界和ratio。不导入候选/MLX/ANE，不重新执行输出tensor或硬件；公开的数值metrics仍是受源码约束的采集记录，不是硬件attestation。

## 采用状态

将其保留为**值得继续验证的layer0 warm FFN候选**，不修改默认、不自动提交/推送。不能把这里的private结果覆盖之前[公共CoreML真实FFN负结果](qwen-ffn-tiling-20261015.md)，二者backend/图/边界不同；也不能拿[小图约3×](aneforge-20261016.md)解释本次完整消费者收益。

尚未完成：实际shared backing/更少copy、更多层/独立prompt/seed/LoRA、完整block/request和图像质量、冷启动/常量cache失效、低内存与长时热状态、失败/取消回收及受控产品集成。当前磁盘约0.6GiB，未删模型、原始数据或系统cache；后续大构建先做admission。
