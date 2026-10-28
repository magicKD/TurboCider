# Qwen base512：成对dense gate/up＋typed SwiGLU的微弱完整block信号

2026-10-19，M4 Max/64GiB。接续
[归一化与相邻分工筛选](qwen-base512-hidden-quant-neighbor-share-2026-10-19.md)。
本轮实现了不复制checkpoint的GPU complement融合原型，完成合成全输出oracle、
真实BF16权重完整block及生成/编辑来源重放。**寄存器版生成block有约0.35–0.56%
的小样本信号，编辑仅约0.08%；还不是整图提速或采用结论。**主native、App/
默认值未改，当前已验证完整40步配置仍是原固定FFN/QKV＋QKV前看。

## 改变的机制，不是重跑packed权重方案

原GPU complement分别计算gate与up，再进行typed SiLU×up和原F32 down partial。
新kernel在一次MLX custom-kernel调用内执行两次MPP `matmul2d`，直接读原
`[24576,4096]` BF16 gate_up矩阵，gate范围I与up范围12288+I严格对应。
first/physical pitch/offset不压成新的紧凑W。新增packed checkpoint bytes0。

比较两种实现：register同时保留两份F32 accumulator；shared先把BF16 gate写入
threadgroup tile，释放第一accumulator后执行up。二者保留BF16 gate、up、typed
sigmoid、SiLU及最终乘法边界，只发布hidden，不发布全局gate/up。共享X的view
不等于已证明物理读取复用；MPP/编译器可改变实际调度。

与此前“把gate/up权重拼接成一份再做一次GEMM”的负方案不同：本轮没有额外W
bank，直接做kernel级epilogue融合。也不假设基线的每个逻辑中间量一定物化；
实际字节流量/占用率没有counter资格，收益由匹配完整消费者决定。

## 合成资格：完整数值与索引先过关

实际安装MLX `unary_ops.h::Sigmoid`使用T-valued exp表达，不能换成全F32 logistic
最后才cast。原生`tc::silu`的BF16表达也逐阶段保留。

两个固定合成fixture：M33/K128/逻辑F320，first17/count257，输入/权重都有非零
view offset与tile tail；实际M1024/K4096/F12288，count6144。输入和权重采用
可独立重算的dyadic周期网格，而不是checkpoint或性能分母。

两个kernel×5阶段（gate/up/sigmoid/SiLU/hidden）×2fixture，共20个输出，分别与
普通MLX和未融合MPP控制比较，40对完整原BF16字节全部exact。20个非法dtype/
layout/range/tile/stage拒绝；输入和权重分别A/B/A恢复、旧输出owned不被覆盖。
原source前后未改，没有通过捕获常量或返回已完成旧tensor实现“快”。

CPU审计重读全部payload与MLX/native/helper身份。利用输入周期127、权重周期257，
计算**所有unique完整K整数dot**再展开每个输出，并非采样；全部gate/up值精确
符合独立投影oracle，完整W hash也按生成配方独立重算。非线性另与F64数学目标
做有范围的误差检查，typed bitwise主控制仍是实际MLX/MPP，不冒充硬件ISA仿真。

原件`outputs/qwen-dual-gup-kernel-v1-20261019`，独立复核
`outputs/qwen-dual-gup-kernel-audit-v1-20261019/audit.json`。helper为
`build/qwen-dual-gup-kernel-v2-20261019/libqwen_dual_gup_kernel_probe.dylib`，绑定
C++/header SHA及native ID。该独立GPU资格没有ANE预测或整图速度结论。

## 真实完整FFN：同窗四arm、一个固定ANE Program

只使用原BF16 base512、无LoRA/student/Z Image。来源请求声明40步，但先完成
第一sampler step，再捕获首decode layer0并停止，不导PNG；不是完成40步。
生成和编辑都用相同原checkpoint；编辑保留标准1024参考/4096参考tokens。

每窗ordinary full compiled GPU、staged GPU、原GPU complement的Private FFN、
融合GPU complement的Private FFN四arm轮转，各2warm/8measured，register另
加倍16。两个Private arm共用**同一个**FFN owner/Program及固定C6144分工，
原gate/up/down逐次真实stage；K2048、互补ANE激活、F32 join不变，无W cache/
prefetch。QKV仍GPU，不能直接挪成joint QKV前看路线的图片时间。

首A/扰动B/恢复A及layer0/1/0检查额外保留原hybrid三个完整控制；每窗实际
workload为`2*repeats+12`，即28或44，setup capability3次另列。输入/权重/输出
真实完成及drain合同不变，Program1、headroom1，无retry/fallback/rebuild。

### 实测包络

单位ms，为完整block wall中位数，包括attention前段、真实重备W、GPU/ANE、F32
join、残差和完成；强增量分母固定同窗private-original，不换成较慢完整GPU。

| 窗口 | 原完整GPU | 原hybrid | 融合hybrid | 增量耗时下降 |
| --- | ---: | ---: | ---: | ---: |
| register生成，8 samples | 32.861562 | 24.066209 | 23.930834 | 0.563% |
| shared生成，8 samples | 32.868312 | 24.030667 | 23.985355 | 0.189% |
| register生成，16 samples | 32.855000 | 23.969063 | 23.885959 | 0.347% |
| register编辑，8 samples | 见原始审计 | 29.732375 | 29.708312 | 0.081% |

独立timeline生成为23.977604/23.956480ms，仅作诊断，不进入stock速度排序。
准备就绪FFN输入下，register GPU complement alone约10.61→10.57ms，编辑
约10.64→10.58ms。总差额小，不能以逻辑少了几次写回或把0.1ms乘1280，宣称
完整请求省了某个秒数；没有kernel/带宽/独占设备归因。

shared没有更强信号，因此后续优先固定register、BM32/BN64；本轮没有再盲扩
所有tile或重试已否定的份额。显式register候选仍须完整请求确认，编辑尤其
接近噪声，不能说已稳定加速。没有置信区间或统计/thermal资格。

## 全数组、原payload与独立时间线

两层四份原FFN完整BF16 payload独立hash核对；每窗14份完整block BF16数组、
10份GPU component数组（BF16 input/hidden及F32 partial）重读复算。生成和
编辑的原/融合hidden、F32 partial和完整hybrid增量输出全部按完整字节exact；
同case普通GPU/原hybrid控制与源输入相同，同recipe重复/timeline也全量恢复。
这只覆盖本轮真实两层及输入，不能授予32层/全部prompt/图片全域无损资格。

F32 down保留原physical pitch12288。额外对原/融合、A/B四套partial各取32个
固定输出坐标并使用完整K6144做独立F64 dot，共640个见证点。是采样down oracle，
不是完整checkpoint FFN的全CPU GEMM；上面的完整投影oracle只适用于合成fixture。

timeline确认original/fused改变GPU分支期间只有一个稳定FFN model、28次实际
workload evaluate，hot compile/load0；Private selector和Metal完成/时钟校准
无drop/pending。GPU JIT在preflight，未证明所有未来shape都不编译，也未证明
物理INT8或GPU/ANE硬件重叠。编辑只完成stock，不借生成trace授予其独立API资格。

五个真实来源窗口共192个measured完整block、96个measured独立GPU complement，
156次workload＋15次setup capability预测；另有5次首step来源捕获。没有新的
完整40步图像请求。审计
`outputs/qwen-dual-gup-full-block-audit-v1-20261019/audit.json`及同目录trace。

## 实现和复跑

- `tools/profiling/qwen_base/dual_gup.hpp`：原buffer双MPP＋typed epilogue；未接入
  native，含range/dtype/layout/tile检查。不是低精度cast或W repack候选。
- `dual_gup_kernel_{probe.mm,collect.py,audit.py}`：合成完整GPU资格/独立oracle。
- `ffn_projection_probe.mm`的`TC_FFN_DUAL_SCREEN`显式诊断分支；默认三arm保留。
  `dual_gup_collect.py`/`dual_gup_audit.py`严格四arm、实际计数、paired source/
  F32/BF16类型、权重/输入A/B/A及one-program selector合同。
- builder：`tools/native/build_qwen_dual_gup_{kernel,ffn}_probe.sh`。真实helper为
  `build/qwen-dual-gup-ffn-v1-20261019/libqwen_dual_gup_{register,shared}_c6144.dylib`。
  进入Session前验证native ID，不能把旧C++ helper和不同RunResult ABI的库混用。
- 最终130项host测试通过：`outputs/qwen-dual-gup-host-tests-v3-20261019.log`，
  不是130项硬件/图片资格。新增6项覆盖独立周期oracle、typed位控制、四arm
  强分母、精确预测scope、双private完成、完整source/原hybrid/重放合同。
- native仍`build/qwen-base-qkv-lookahead-v1-20261019/libturbocider.dylib`，完整659
  input/source archive和上轮full40审计身份再次核对未变。没有App/default变化。
- 工具/测试/本报告封存`outputs/qwen-dual-gup-tools-v1-20261019.tar.gz`，摘要见
  `outputs/qwen-dual-gup-artifacts-v1-20261019.json`。所有旧库、pilot和失败检查
  日志保留，未修改设备数组或放宽原数值/完成门槛。

所有设备窗串行，自有build/test/CPU数组审计在性能窗外。原生kernel资格峰值
physical footprint约1.94GiB（含生成/快照/JIT）；真实gen/edit窗口约15.83–18.00GiB，
system swap-in/out0。记录竞争推理CPU负载，无GPU独占或物理counter资格，统一
`qualification_passed=false`。未下载/删除/清缓存或停止ComfyUI；磁盘仍保留空间。

## 尚未完成的关键验证

下一阶段需用默认OFF、base512/fixed6144专用的研究入口接入真实pipeline，并给
出actual GPU分支计数。保持现有correct text-prefix GPU-step0、标准编辑prefix
和QKV lookahead强基线，只改变该开关；长prefill先保留旧GPU分支，不为新length
热建ANE图。GPU策略进入prefix算术namespace，先验证实际模型源与生命周期。

之后同库完整40步生成/编辑off/on、反向、不同seed/prompt及1/2参考长度恢复，
重读final RGBA、实际预测、compile/load和内存。只有端到端收益保留才采用；
若GPU/ANE竞争或额外验证成本抹去信号，继续保留旧配置。当前总体目标ACTIVE，
不能把本轮0.35–0.56%的局部信号写成整图加速或全部瓶颈已解决。
