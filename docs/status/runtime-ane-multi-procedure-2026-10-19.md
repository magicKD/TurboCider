# Runtime ANE 多固定行数 procedure：基础能力验证

2026-10-19，M4 Max/64GiB。目标仍是Qwen Image 2.1 base512生成/编辑的完整性能，
不是用小矩阵能力检查替代图片速度。App/default及PrivateW8Graph选路未变。

## 为什么先改这个边界

已有同一固定图处理不同尺寸的结果，但固定bucket可能产生额外padding和预测。
用户允许按size使用不同runtime图；同一model内的多个固定function是另一个可
检验方案：启动时编译/加载一次，换长度只选procedure，层/LoRA变化只更新输入。
尚不能据此声称它必然胜过独立最优size图或GPU。

参考Splash `be3b083c5f413673d44e44881042e875e674e333` 的多procedure设计。
新增独立研究构造器`Program::MultipleProcedures`与`prepare_function`，从真实
model attributes检查function→ID及每个function的全局输入/输出symbol编号。
默认构造器保持原单procedure路径、原MIL/缓存身份和现有W0外部event接口。
多procedure尚不开放W0或多输出；不能忽略全部输出完成和owner/drain合同。

实际driver把声明为`x,w`的输入列为`w,x`，两个function的全局input indices
分别为0/1和2/3，output indices为0和1。不能用声明顺序、本地ordinal或统一
procedureIndex0代替真实映射。全局名字允许跨function重复，每个function内则
拒绝重复/未知/越界symbol、不完整entry及非单输出。

## 已完成的真实设备能力与独立复算

`multi_function.hpp`生成两个FP16 MatMul function，固定32/64行、H128。
同一真实input、weight、output IOSurface服务32→64→32的A/B/A；输入和输出
共用最大64行的128-byte pitch。W是动态输入，两个function不嵌入模型权重。
回到32行时检查输出后32列canary不被覆盖，防止逻辑shape与物理pitch混淆。

V2完成独立finite/oracle修复，V3进一步绑定实际编译的MIL头文件；V3完整通过：

- 实际一个ANE model、三个evaluate，function调用中compile/load均0。
- 两个真实procedure都执行，均经过成功driver completion及done-event检查。
- 独立读取完整input/W/output，用F64全K128 `einsum(optimize=False)`复算全部
  16384个有效输出，half参考误差0；A/B不同，回A所有输出bits exact。
- 共享surface的padding canary保持，源/库/工具、664输入exact native archive、
  原始process-tree memory hash-chain和selector事件均复核。
- CPU负控覆盖乱序/非零procedure、跨function重名、重复/越界/缺失symbol，
  以及NaN/Inf输入/输出、错误输出、错误tail和错误shape。

首V1审计中平台BLAS报浮点状态警告，且NaN误差比较可能漏拒。因此该审计不作
最终证据；V2不用BLAS并显式检查所有数组、参考结果和误差finite，重新采集及
独立复算。原V1记录保留，没有压掉警告或放宽预算。

V3的probe同时报告编译时CPP/MIL-header SHA，build前后检查两份源不变，采集与
独立审计核对实际库/两份compiled pin与captured source，避免“CPP没变但头文件
改变”仍把旧helper当新图。V2工具版本独立封存，旧结果不被覆盖或改标签。

原legacy固定Program的溢出/禁止rebuild、完整GPU fallback、sticky失效与资源
退役探针亦在V6重新通过。29项相关CPU测试通过。base512生成与full-ref1024
编辑分别完成完整40步、各1280个健康混合FFN block、1312/1440次预测；两张
完整RGBA、全部原模型完整SHA、native/archive、memory与selectors独立审计
通过，见`outputs/qwen-base40-procedures-regression-audit-v6-20261019/audit.json`。
两窗均记录外部ComfyUI负载，不作为新性能对照；各只有首请求，也没有热倍率。
不能把合成或legacy回归数字当多procedure W8A8 FFN吞吐、物理INT8 MAC、硬件
重叠或新增图片质量资格。

## 工程入口与下一道gate

- `native/backends/private/ane_procedure.hpp`：CPU metadata合同。
- `native/backends/private/ane_program.{hpp,mm}`：显式研究接口；生产调用不换图。
- `tools/profiling/qwen_base/multi_function{.hpp,_probe.mm,_collect.py}`：能力源、
  实际driver采集及terminal后独立数组复算。
- `tools/native/build_ane_multi_function_probe.sh`：必须链接匹配的新native。
- `outputs/ane-multi-function-capability-v3-20261019`、
  `outputs/ane-multi-function-audit-v3-20261019.json`：被忽略的原件。

下一步不是直接打开模型开关：先让现有W8 SwiGLU emitter表达共享最大物理pitch、
各固定逻辑rows，保持旋转/A8/归一化输出/FP32 restore配方和动态W ABI；再测
真实FFN所有输入/消费者与stale/tail负控，最后才接PrivateW8Graph的chunk选择。
需要证明与单bucket相同数值配方、双bank/预取身份、失败与有限性门均保持。
尺寸和参考数的完整40步/8步对照、首/热请求与质量必须分开补齐。

FP16基础之后的W8后端组件接入见下节；仍没有据此产生新图片倍率或满足用户
的条件commit。已有尺寸特定收益见Qwen持续报告，不能与本能力相乘。

## 后续：W8 SwiGLU共享pitch组件已在V7跑通

新增`w8_swiglu_function_family_program`与显式`PrivateW8Graph(..., fixed_rows)`：
一个MIL program、一个共用的Hadamard常量blob，运行时权重仍是inputs。函数行数
严格递增、是最小bucket的整数倍，最大值等于物理workspace；worker按剩余行数
选可容纳的最大chunk。没有全局环境开关或生成Session采用，默认构造器不变。

ANE input保持最大物理行数，读取后取固定逻辑列；output用逻辑shape和最大
物理pitch。GPU `Surface::leading_columns`只形成保留同owner/MTLBuffer/pitch的
view，不重新分配；staging/restore使用实际逻辑列，driver绑定仍必须是原完整
Surface。禁止越界、零列、从小view扩回大view；行/列view不能成为driver端口。

首个域为fixed-program、base Sylvester、无LoRA/Comfy/group/BF16边界/A8
lookahead，保留两W bank、有限性检查、normalized输出与GPU FP32 restore。
额外function的compiler/workspace余量进入保守memory admission，实际slot仍
只有最大集合。它不为GGUF/ConvRot/encoder或任何模型质量自动授予资格。

V7/665-input exact native archive，V4采集与独立审计已经验证：

- H128/F512的完整合成SwiGLU，32/64/96/32行，含正负输入和两套动态W；96行
  实际64+32，legacy控制按64补齐，候选只stage/restore有效逻辑行。
- Exp与sigmoid-complement分别一个独立进程，每窗一个legacy model加一个
  family model；每个family只有一个program，所有function调用无新compile/load。
- 每窗6次self-test capability、10次workload预测；两窗共32个实际evaluate。
- 两侧各622592 bytes实际工作槽；family没有按function复制X/tx/Y或W bank。
- 114688个有效FP32输出独立全K维F64 gate/up→SiLU×up→down复算。相对未量化
  F64的最大relative L2分别约1.735%/1.679%；候选相对相同W8控制relative L2为0。
  这是原W8近似误差，不是新function特性的新增误差。所有输出finite，目标尾部
  canary保持，回到首个输入/权重/shape的原始字节完整一致。
- auditor以uint32/uint16视图检查bits恢复，新增signed-zero负控，避免把数值
  相等当逐位相等。CPP pin和编译时native ID一起验证，ABI不匹配时在构造任何
  Surface前拒绝；独立审计绑定原始memory hash-chain与actual selectors。
- 旧V6/V7实际native库导出的五组legacy MIL与常量共10份文件逐字节一致，覆盖
  real Qwen1024/2048-row、tiny base/LoRA/Comfy；不是只比较requested flags。
- 原固定Program overflow/禁止rebuild、完整GPU fallback、sticky失效及资源
  退役回归在V7重跑通过；相关34项CPU测试覆盖default golden、几何、NaN/Inf、
  wrong tail、错误policy与仅2%的新增delta也必须被拒绝。

当前原件为`outputs/ane-w8-functions-{exp,complement}-signed-v4-20261019`，
独立审计为`outputs/ane-w8-functions-{exp,complement}-audit-v4-20261019.json`。
工具为`w8_functions_probe.mm`、`build_ane_w8_functions_probe.sh`与
`multi_function_collect.py --kind w8 --activation exp|sigmoid-complement`。旧工具
版本和golden均独立封存，未覆盖旧证据。所有原件在已忽略的outputs/build。

**这仍不是真实Qwen H4096/F12288的FFN或图片跑分。** 下一道gate必须读取原
BF16 gate_up/down和已冻结的实际FFN输入，比较完整GPU head＋ANE partial＋join，
测1024/1536/2048及尾块并区分真实输入与拼接/扰动输入。通过后再让HybridFfn
按最小bucket pad并接入Session；目前其外层仍按原bucket pad，不能仅创建
family就宣称生产路径已经省掉padding。LoRA等其它域需要独立准入/验证。

后续真实H4096/F12288的完整GPU＋ANE FFN、正反向与全数组审计已经完成，见
[真实FFN报告](qwen-base512-w8-function-family-real-2026-10-19.md)。1536/2560行
有局部信号，1024主路径没有新稳健收益；外部负载使性能门槛未过，Session仍
未采用。上述“下一道gate”不再表示真实FFN检查尚未执行，但完整图片gate仍在。

Surface的C++布局发生变化，所有C++ helper必须重新链接V7或更新库；旧V6库及
其归档只用于历史审计，不可与新header混用。cold较小function的首次成本也要
记录；最大行数self-test通过不等于所有function都已完成性能warmup。
