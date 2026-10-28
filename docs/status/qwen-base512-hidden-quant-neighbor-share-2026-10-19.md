# Qwen base512：归一化倒数复用与相邻FFN分工筛选

2026-10-19，M4 Max/64GiB。继续
[H512因式分解负结果](qwen-base512-h512-factorization-2026-10-19.md)和
[QKV权重前看后的主FFN等待](qwen-base512-qkv-lookahead-2026-10-19.md)。

本轮完成两类新的独立实验，**没有新增完整图片速度纪录，不改变主native、App
或默认配置**：hidden归一化的倒数复用没有稳定完整算子收益；真实FFN block的
5632/6656相邻分工均慢于6144。上轮完整40步生成约28.84s、重复编辑约36.22s
仍是当前已验证配置的历史结果，没有借本轮组件时间重标或外推。

## 1. hidden归一化：有界倒数不等于更快

生产子图为：`H512 → abs/reduce_max → floor≥2^-12 → hr/floor → ×127 → int8`。
比较三种固定图表达：原`divide`；`real_div(1,floor)`后乘hr；以及明确
`inverse(floor, epsilon=0)`后乘hr。epsilon不能沿用API默认1e-4。
在该floor合同下倒数至多4096，避免了subnormal denominator倒数直接溢出的
一般反例，但并不保证数值等价或优化有效。

分别测raw hidden和原H512之后的完整链，C512/M32和实际C6144/M1024。每个策略
独立固定Program，不在热路径换策略/编译；不加载checkpoint。输出只有一份
packed `[q/128,hscale]`，dequantize scale与生产down消费者相同，保留原scale
表达，不额外添加内部tap。这个诊断消费者不是完整down MatMul，融合/放置变化
仍不能外推原FFN硬件内部行为。

CPU原生测试确认原归一化子图、H512完整常量与当前native逐字节对应；诊断边界
SSA命名不同只在测试中明确映射。helper编译绑定C++、直接header、传递使用的
rotation header SHA及native ID，进入C++对象前检查身份，避免混用旧ABI。

### 匹配stock与加倍样本

每策略2warm，stock12/24 measured轮转；单位ms，为完整operator＋finite-transfer
边界，不是ANE kernel busy time。包含旋转的时间只作原始诊断包络，不能称该
全域数值资格已通过（tiny反例见后文）。

| 实际shape/策略 | stock12 | stock24重复 |
| --- | ---: | ---: |
| raw：divide | 1.466625 | 1.523395 |
| raw：倒数divide后mul | 1.454312 | 1.518709 |
| raw：inverse后mul | 1.450625 | 1.507834 |
| H512＋链：divide | 1.915729 | 1.879417 |
| H512＋链：倒数divide后mul | 2.008375 | 1.887750 |
| H512＋链：inverse后mul | 1.885688 | 1.884625 |

raw窗口约1%的细微信号没有在更完整旋转消费者上稳定保留：inverse首轮略快，
重复转为略慢；更早pilot的两种倒数也没有胜出。timeline独立，不用于挑更快
分母。没有依据把该改写接入主FFN；也未证明编译器是否已复用某个物理reciprocal。

### 全数组、量化codes与异常域

保留random-A、zero、tiny、wide、有限源旋转溢出及恢复A。所有物理请求/transfer
实际完成后才读surface；预期数值拒绝不冒充driver失败或成功性能。有限overflow
输入为原D符号×4096，经H512的数学peak为92672，超过FP16；实际packed scale
非有限、GPU validator拒绝，再恢复A逐位一致。raw同输入不溢出，应正常通过。

独立CPU F64 H512/raw→RNE int8→原FP16 scale重构全部codes/scale，并检查整数
范围[-127,127]、scale正有限、逐点预算及relative-L2。zero和raw tiny的zero
codes/scale精确检查；没有把改变后的原实现输出当独立oracle。

发现一个**原图和两候选共有**的tiny反例：源为约2^-24量级，H512后数学参考
仍有少数非零codes；C6144/M1024有7226个非零目标codes，三图实际全为0且scale
一致。相对独立requantized目标的L2为1，未放宽0.02筛选门槛；因此rotate-family
的whole-domain数值资格均为false。该域输入很小、输出能量也很小，不能把此
比例直接解释成真实模型100%误差或图像失败。与subnormal处理/低精度lowering
一致，但没有内部tap/ISA资格，不唯一断言具体flush指令。

初版auditor误把“tiny输入”都当“旋转后zero目标”，修正的是参考域分类，保留
实际L2拒绝，不通过改目标或放宽预算取得绿色。正常随机/wide的codes也并非
全与原divide逐位相同，不能以代数等价宣称全图无损。

最终3窗共576个measured operator边界、864次实际预测、216份完整packed输出，
含18次预期逻辑overflow拒绝。144次跨窗完整payload重放exact。独立timeline
12个固定诊断模型、240次evaluate，改变输入/测量期间compile/load0，校准/
drop/pending/physical completion检查通过。不是产品新增12个逐层模型。

## 2. 相邻固定分工：当前6144仍更平衡

为避免只在小算子上打转，另筛选C5632/6144/6656，每进程一份固定FFN Program。
原BF16 base40来源先完成第一sampler step，再捕获首decode layer0，随后完整
block重放；**不是完成40步或导出图片**。QKV在GPU，不能将结果直接移植成joint
QKV前看路线的图片倍率。

保持互补SiLU、descriptor K2048、F32 partial join、fixed1 async、launch fence1，
无W cache/prefetch、LoRA/student。权重来自原layer0/1/0，未新增checkpoint副本。
普通完整compiled GPU、staged GPU、Private真实重备W三arm轮转；各2warm/8measured，
5632追加16。另测ready输入下GPU complement alone，不把拆分GPU当强基线。

| 固定ANE通道 | GPU通道 | 普通完整GPU ms | Private完整block ms | 相对6144 |
| --- | ---: | ---: | ---: | ---: |
| 6144 | 6144 | 32.853229 | 24.137334 | — |
| 5632 | 6656 | 32.843375 | 24.908333 | 慢3.19% |
| 6656 | 5632 | 32.863854 | 25.039042 | 慢3.74% |

反向6144恢复24.080729ms；5632加倍样本为24.992250ms，没有恢复收益。
独立timeline的5632/6656约24.922833/24.990771ms，不参与stock分母。
没有基于这些组件数字声明新的生成/编辑完整请求提速。

### 为什么“多交ANE”或“少交ANE”都未胜出

以下独立/重叠host边界不能相加成完整wall或当物理ANE计算时间：

| C | prediction等待 ms | output完成等待 ms | caller等待 ms | GPU complement alone ms |
| --- | ---: | ---: | ---: | ---: |
| 6144 | 10.067 | 0.369 | 10.410 | 10.618 |
| 5632 | 9.255 | 2.097 | 11.323 | 11.522 |
| 6656 | 11.034 | 0.234 | 11.234 | 9.784 |

5632虽缩短prediction等待，却增加输出等待/GPU分支；6656缩短GPU分支却增加
prediction等待。与6144当前更平衡一致，不唯一归因ANE compute或restore shader。
这不同于上轮C5120的实验，不重复其tile/FP16/packed负方案；也不是穷尽所有
分工或换GPU实现后仍然最优的证明。

三分工普通GPU controls与源输入全部exact；同分工重复/timeline全部十份BF16
数组exact。两层四份原FFN完整payload独立核对。完整block relative-L2约
0.000823/0.000862/0.000939（5632/6144/6656），不授予新分工的整图语义资格。
每owner Program1、headroom1，无retry/fallback/新adapter工作。两新分工timeline
各一个稳定模型、15次workload evaluate、3次setup capability另列，hot compile/
load0。新helper构建绑定native ID，构造Session前拒绝不匹配的C++ ABI。

## 工程与证据位置

- hidden链：`hidden_quant_mil.hpp`/`hidden_quant_probe.mm`/`hidden_quant_collect.py`/
  `hidden_quant_audit.py`，位于`tools/profiling/qwen_base/`；builder为
  `tools/native/build_qwen_hidden_quant_probe.sh`。最终helper
  `build/qwen-hidden-quant-probe-v2-20261019/libqwen_hidden_quant_probe.dylib`。
  最终审计`outputs/qwen-hidden-quant-full-audit-v1-20261019/audit.json`及trace。
- 相邻分工：`ffn_projection_collect.py --channels 5632|6144|6656`只选择显式诊断库，
  主完整请求collector/default未扩大；`build_qwen_ffn_share_probe.sh`分别构建
  `build/qwen-ffn-share-probe-v1-20261019/libqwen_ffn_share_c*.dylib`。
  `ffn_share_audit.py`复核完整输入/原payload/arrays/memory/selectors。
  审计`outputs/qwen-ffn-neighbor-full-audit-v1-20261019/audit.json`，含7窗、192个
  measured完整block，113次workload＋21次setup capability预测、70份完整BF16数组。
- 最终124项host测试通过，日志
  `outputs/qwen-hidden-quant-neighbor-host-tests-v1-20261019.log`。新增8项hidden链、
  4项邻近分工合同，不把host测试合并成硬件或整图资格。
- native仍`build/qwen-base-qkv-lookahead-v1-20261019/libturbocider.dylib`，659-input
  source archive仍`outputs/qwen-base-qkv-lookahead-native-source-v1-20261019.tar.gz`。
  库SHA与上轮full40 audit摘要再次核对未变。所有旧库、pilot、错误日志保留；
  本轮不新增产品开关或热图切换。
- 工具/测试/报告封存`outputs/qwen-hidden-quant-neighbor-tools-v1-20261019.tar.gz`；
  摘要见`outputs/qwen-hidden-quant-neighbor-artifacts-v1-20261019.json`。

hidden三正式窗源为`outputs/qwen-hidden-quant-{stock,repeat,timeline}-v2-20261019`；
早期v1只作pilot，其helper缺少传递header独立stamp，不纳入正式三窗计数。
最初CPU文本检查因诊断SSA名不同而失败；只修正测试边界名，发射图未改。
首次tiny参考分类失败也保留，后续没有重写设备数组。

所有模型/算子窗串行，自有build/test/CPU数组审计在窗外；无需下载、删除模型/
缓存或停止ComfyUI。hidden窗physical footprint峰值约259MiB；完整FFN窗约
15.68–15.75GiB，system swap-in/out0。非独占小样本，无统计/thermal/物理INT8
或hardware overlap资格，统一`qualification_passed=false`。磁盘仍约11GiB free。

## 后续方向

当前不再盲扩归一化表达或固定份额。更值得先检查GPU gate/up是否能在不新增
per-layer packed W副本、保留BF16值边界的条件下复用X载入并融合SiLU×up；
此前“先拼接权重再做一次GEMM”的负结果不等同于这种kernel级融合，但也提醒
新增寄存器压力/低occupancy可能抹去收益。尚未实现或测得加速；若GPU complement
确有改进，再重新验收GPU/ANE分工、真实block与完整40步生成/编辑。目标保持
ACTIVE，本轮两个方向的负结果不被包装成全部瓶颈已经解决。
