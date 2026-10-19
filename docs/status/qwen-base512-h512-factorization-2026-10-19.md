# Qwen base512：H512旋转因式分解未带来收益

2026-10-19，M4 Max/64GiB。继续
[QKV单组前看后的FFN瓶颈分析](qwen-base512-qkv-lookahead-2026-10-19.md)。
本轮没有新增整图速度纪录，没有改变native、App或默认配置；当前最快已验证
完整40步配置仍是上一轮的固定FFN/QKV＋显式QKV前看，生成约28.84s、重复编辑
约36.22s。这些是上一轮完整请求结果，不是本轮重新测得的图片时间。

本轮实测的是原FFN内H512 hidden旋转的独立固定FP16算子：原分组Conv、四种
两级Conv、两种batched MatMul，以及两种无显式transpose的MatMul，共八个候选。
**全部候选在实际shape的原始诊断wall都更慢，且wide输入未通过预先固定的逐点
误差筛选，因此不接入runtime。** 不把这些数值不合格样本登记为可用速度收益。

## 假设与保持的语义

原hidden旋转在ANE图内使用`[1,C,1,M]`的512-channel grouped 1×1 Conv，模型
无关常量为`H512·D / sqrt(512)`。实际目标为C6144/M1024。因式分解利用
`H512 = H8 ⊗ H64 = H16 ⊗ H32`，也比较相反factor大小和H4/H128。

第一阶段包含原始D符号，第二阶段不重复D。序列/token轴不归约，不改变原始
通道顺序。Conv和普通batched MatMul通过两次transpose恢复索引；flat版本
第一阶段后reshape为`[1,G,A,B*M]`，第二阶段在A上归约，直接恢复原顺序，避免
显式transpose，同时减少重复第二阶段常量。所有图shape固定、无checkpoint常量。

CPU C++测试检查两件不同的事：原H512完整serialized constants与现有native
emitter逐字节一致；八种factor的实际FP16系数乘积精确重构原系数、符号及索引。
因此减少乘加/常量不是通过换一个目标矩阵实现。但中间FP16值和compiler lowering
可以改变误差；实数矩阵相同不代表执行逐位一致或更快。

## 强控制与真实设备边界

另测C512/M32小shape与identity，先以signed impulse检查各组/轴/符号，再用
确定性random-A、wide及A恢复。每个shape/策略一个独立诊断Program，不是给主
runtime或32层增加模型。源input每次检查未改；A/B/wide/A输出逐位恢复。

source与helper嵌入C++/header SHA及native ID；进入C++对象前拒绝错误库身份。
每个实际预测通过GPU ready/ANE driver physical completion/输出finite验证后
才读surface。计时含完整operator handoff与finite-transfer，输入准备、快照、
保存和CPU oracle在timer外；不是ANE硬件kernel busy time，也不是FFN完整block。

原小shape的两级Conv在compile阶段拒绝，实际大shape则都能编译并运行；不能
据此断言“小group通道一律不支持”。C/M同时变化，未唯一缩减compiler失败根因。
optional失败逐项记录为`compile-rejected`，不转GPU、不生成0ms样本，不跳过
应有的原图/identity控制。编译拒绝不是推理失败或设备性能。

## 重复测量：更少乘加，原始wall仍更长

以下为C6144/M1024，每arm两个warmup；stock各12/24个测量，timeline独立。
带“候选”的时间仅保留原始诊断包络，**不具备本轮数值资格，不作speedup分子**。

| 策略 | stock12原始ms | stock24重复原始ms |
| --- | ---: | ---: |
| identity控制 | 0.803479 | 约0.8 |
| 原Conv H512 | 0.986083 | 1.016604 |
| 候选Conv H64/H8 | 1.666188 | 1.693896 |
| 候选Conv H32/H16 | 1.625292 | 1.684959 |
| 候选Conv H16/H32 | 1.616208 | 1.675583 |
| 候选Conv H128/H4 | 1.713354 | 1.722437 |
| 候选MatMul H64/H8＋transpose | 1.718938 | 1.781667 |
| 候选MatMul H32/H16＋transpose | 1.675083 | 1.722000 |
| 候选flat H64/H8 | 1.296084 | 1.352041 |
| 候选flat H32/H16 | 1.261813 | 1.292125 |

较早六候选stock窗口的原图为0.988229ms，排序没有相反信号；独立timeline
原图0.986709ms、flat32为1.265313ms，但timeline不参与stock排名。
去掉显式transpose并改变batched shape/常量布局后确有改善，仍没有胜过原图。
没有硬件分段消融，不能把差额全部归给transpose、group大小或物理带宽。
identity也有真实handoff/validation成本，不能简单相减当纯旋转硬件时间。

这些operator结果不乘1280推导整图秒数，不外推为“因式分解在所有融合FFN里
无效”。本轮没有改变完整FFN融合后的图，因此那种变化仍未实测；当前证据
不支持为其扩展主runtime。优先停止无收益接入，而非继续扩大所有factor组合。

## 数值：整体L2和逐点预算必须分开

独立CPU审计读取每份完整FP16输入/输出，用F64 signed Walsh–Hadamard butterfly
重算所有元素，系数取原serialized FP16的精确值181/4096。它不调用ANE/MLX，
不使用新factor矩阵作循环oracle。CPU测试另用独立整数dense H512 dot证明
butterfly全值；固定输入网格为integer×2^-18，不依赖平台BLAS浮点状态。

预先选择的operator筛选是relative-L2<0.002，且每点满足
`abs(error) <= .002 + .005*abs(reference)`，两个条件都要成立。它是本轮局部
domain预算，不是完整图片的质量门槛或“低于它就无损”的证明。原图所有模式
通过；所有candidate在actual-shape wide模式存在逐点超限，未放宽门槛。

以flat H32/H16为例：

| 实际输入 | 原Conv relative-L2 | flat relative-L2 | flat逐点超限数 |
| --- | ---: | ---: | ---: |
| random-A，幅值≤0.125 | 0.00141728 | 0.00054261 | 0 |
| signed impulse | 0 | 0 | 0 |
| wide，幅值≤16 | 0.00020819 | 0.00029409 | 29586 /6291456 |

wide最大abs error原图0.01581848、flat0.02265143。不能因small模式误差更小
就称全domain精度更好，也不能仅凭小global L2掩盖稀疏局部误差。signed impulse
exact、serialized系数乘积exact与额外中间值/lowering误差并存；没有物理
accumulator观测，不唯一归因于某个FP16舍入操作。

更慢是停止接入的主要原因；本轮没有运行checkpoint、FFN down消费者或新图片，
这些operator超限不冒充真实Qwen图像画质失败，也不修改现有W8数值门槛。

## Profiling/审计工程与证据

- 四个成功诊断窗：`outputs/qwen-h512-rotation-stock-v{2,3,4}-20261019`与
  `outputs/qwen-h512-rotation-timeline-v4-20261019`。共912个measured operator
  边界、1272次实际预测，不是1272个FFN或图片请求。
- 240份完整输出、957743104个元素、所有源模式及同recipe128次完整payload
  重放一致。最终独立审计：`outputs/qwen-h512-rotation-audit-v2-20261019/audit.json`。
- timeline实际16个成功诊断模型、288次evaluate；4个小shape compile拒绝单列。
  所有改变输入/测量阶段compile/load0，校准/完成、drop/pending门通过。
  原始失败compile保留在trace的`capability-setup-rejection`，不放松共享
  observer的成功workload合同。trace不是ANE硬件计时或物理INT8证明。
- 入口：`rotation_mil.hpp`、`rotation_probe.mm`、`rotation_collect.py`、
  `rotation_audit.py`，均在`tools/profiling/qwen_base/`。builder为
  `tools/native/build_qwen_rotation_probe.sh`。最新helper为
  `build/qwen-rotation-probe-v4-20261019/libqwen_rotation_probe.dylib`。
- 旧八arm helper为v3；旧工具source由
  `outputs/qwen-h512-rotation-tools-v{1,2,3}-20261019.tar.gz`按版本复核，
  没用新十arm代码重标旧设备记录。最终工具/测试/文档为v4 archive，摘要见
  `outputs/qwen-h512-rotation-artifacts-v1-20261019.json`。
- 最终112项profiler host测试通过：`outputs/qwen-h512-rotation-host-tests-v2-20261019.log`。
  新增7项含原emitter/serialized矩阵、独立oracle、完整payload、pointwise负控、
  精确timing顺序及只能在声明的setup中出现的compiler失败/trace导出。

最早stock-v1在首个factor compile处终止，日志保留，不算成功性能窗。其父级
环境元数据已仅保留backend/private授权，不再保存无关继承环境；device/source/
memory失败证据未改。初版CPU审计在trace exporter遇到声明的compiler拒绝后
退出，v1结果不作为最终完成回执；v2将失败setup明确分类，并在全部trace导出
成功后才发布complete audit。没有改设备数组或放宽任何误差/成功预测检查。

四窗进程树physical footprint峰值约327–352MiB，system swap-in/out均0。
存在竞争推理CPU负载，非独占/统计/thermal资格，统一`qualification_passed=false`。
所有性能窗串行，自有build/test/CPU数组审计在窗外。磁盘仍约14GiB free，未删除
模型、旧原件、cache或停止ComfyUI。

## 下一步

native659-input源码、原library SHA及上一轮full40 audit均再次确认未变，因此
没有给失败候选增加产品开关，也不改变当前最佳路径。下一项更值得独立筛选
的是hidden归一化/量化链中同token divisor的复用：`hr/floor`与显式
`inv_floor=1/floor; hr*inv_floor`。这只是待证伪候选，可能已被compiler合并，
也可能改变近零/范围数值；尚未实现或取得速度资格。仍需保留原finite、quantize、
完成和fallback合同，从算子/真实FFN到完整40步逐级验证。总体目标保持ACTIVE。
