# Qwen base512：分别隔离gate/up和down的固定ANE投影

后续[固定FFN激活表达研究](qwen-base512-ffn-activation-2026-10-18.md)找到新的
完整请求收益。本文保留原混合K的构建/负结果，不据后续数据重标本轮速度。

2026-10-18，M4 Max/64GiB，原BF16 base、512²、无LoRA/student/Z Image。
本轮接续[FFN整体K分片实验](qwen-base512-ffn-projection-2026-10-18.md)，完成
两因素完整block筛选、加倍样本/反向控制、实际selector和独立数组审计。
**没有新的加速收益，不采用两种混合策略。没有运行新的完整40步图片请求。**

## 假设与实现

上一轮K4096与single同时改变gate/up和down，不能据此排除某个局部组合。本轮
新增显式`up-single`和`down-single`，在executor构造时捕获，逐operand映射为
已有Descriptor/Single策略。只有被选择的投影才省略完整operand的identity
slice；小K时未改变的投影也必须保留原字节，不能只保持MatMul数量。

| 策略 | gate/up K | down K | canonical图内投影MatMul数 |
| --- | --- | --- | ---: |
| descriptor | 2048 | 2048 | 7 |
| up-single | 4096 | 2048 | 5 |
| down-single | 2048 | 6144 | 5 |
| single | 4096 | 6144 | 3 |

shape仍1024 rows/H4096/ANE6144；权重/IO ABI、Hadamard常量、F32 join及GPU
complement不变。每owner只一份固定FFN图，实际切换原layer0/1/0权重；不是
每层常量模型。非默认policy继续要求fixed-program、Private授权、ungrouped
Sylvester原base路径，拒绝LoRA/Comfy/grouped/其它BF16边界配方。source recipe/
executor/prefix身份自动区分策略，原失败停用/完整GPU重算合同不放松。

CPU测试确认canonical及小H128/C512的被保留投影子图exact，常量/packed ABI
不变；新旧默认emitter六类fingerprint全部相同。输入文件仍658个，但内容和
runtime ID已改变，不能把上一轮658库重标为本轮库。

## 实测：混合K也更慢

真实source先完成首个sampler step，再捕获首decode的layer0；退出不导PNG。
三arm按轮旋转，各2warm＋8measured，Private每次实际准备生产FFN权重，包含
GPU attention前段、GPU complement、ANE、F32 join、残差及完成。**QKV在GPU**，
不把结果乘进已有QKV＋FFN完整图片倍率。

| 策略 | 原完整compiled GPU | staged GPU | Private实际重备W |
| --- | ---: | ---: | ---: |
| descriptor | 32.861875ms | 32.882209ms | 25.428813ms |
| up-single | 32.831000ms | 32.858875ms | 25.926813ms |
| down-single | 32.832646ms | 32.839750ms | 27.070312ms |
| single | 32.858104ms | 32.861500ms | 27.760021ms |

相对现有Private基线，up-single慢约2.0%，down-single约6.5%，single约9.2%。
两因素观察差：仅up＋0.498000ms，仅down＋1.641499ms，两者＋2.331208ms，
交互项＋0.191709ms。这是非独占窗口的完整block差，不是各ANE硬件kernel
时间，更不代表有精确统计显著性或可分解相加的物理成本。

追加各16measured的up和反向descriptor：普通GPU32.833167/32.855333ms，
Private26.004646/25.445667ms。没有恢复隐藏的up收益，原策略回到原来速度。
没有把不同样本/诊断模式pool成更有利分母，也未将stagedGPU作为弱化主分母。

前四原件：`outputs/qwen-base512-ffn-mixed-block-{descriptor,up-single,down-single,single}-gen-stock-v1-20261018`。
追加原件：`...up-single-gen-stock-v2-20261018`、`...descriptor-gen-reverse-stock-v1-20261018`。

## 真实固定模型、动态权重和完整数组

up/down各独立timeline，均观察到一个稳定evaluate模型身份，15次workload
预测，输入A/B/A及原layer0/1/0权重切换期间compile/load均0。两进程的ordinal1
不是跨进程同一对象。setup各有3次能力预测、1次load；down还观察到1次compile，
up为0。初始成本不隐藏到“零冷启动”表述里，诊断耗时不作stock分母。

八窗累计240measured＋48warm完整block，native workload预测136，能力检查
24次单列；只有两timeline的实际selectors可观察，不把stock host计数冒充
独立API证据。每owner Program实例1、headroom1，无retry/fallback/blocked rebuild。

独立CPU审计已重读每窗十份完整BF16数组、原checkpoint两层四份完整FFN
payload及全部source/archive/library/memory。2×2的真实FFN输入、普通GPU
reference、alternate reference和其它权重reference全部exact；同策略加倍/
反向/timeline的十份数组也全部exact，不只比较某个汇总误差。

完整block relative-L2分别为descriptor0.000597648、up0.000598617、
down0.000591846、single0.000590473；stagedGPU exact。不同reduction仍近似，
不能从约0.06%的block误差授予整图质量；上一轮已出现明显茶壶几何变化。
本轮没有新PNG或完整40步请求，明确不作新的端到端速度/语义质量声明。

审计与两个派生trace：`outputs/qwen-base512-ffn-mixed-block-audit-v1-20261018/`。
新`ffn_projection_audit.py --factorial`要求前四cohort按表顺序、等repeat的stock；
额外cohort须同source且同策略数组exact。改变源输入、乱序、诊断混入分母、
不等repeat或重放输出不同均有host负控。它不放松普通审计或伪造硬件归因。

全部八窗进程树证据完整、退出0，system swap-in/out0，峰值physical footprint
约15.64–15.74GiB（含source/capture/probe/dump/exit，不是单独ANE驻留）。部分
窗口检测到其它ComfyUI Python CPU活动，无设备独占/thermal/统计资格，统一
`qualification_passed=false`；CPU安静也不能证明GPU独占。

## 结论与下一步

更少节点和更大K在本机这条固定FFN路径上没有收益，包括两种局部组合。
继续保留原K2048。source inspection表明base路径没有LoRA/BF16-boundary的
多重finite guard，不能把那些未执行的检查当成当前成本，也不会删除错误检查
来制造速度。现有base链是恢复scale→手写exp/add/div SiLU→hidden乘积→H512
grouped conv旋转→abs/reduce/div/mul/quantize→down。

下一项可证伪假设：比较MIL原exp/div链与独立sigmoid或silu表达，观察是否降低
pointwise中间值/执行开销。已查官方`apple/coremltools@9.0`的
`coremltools/converters/mil/mil/ops/defs/iOS15/activation.py::{sigmoid,silu}`，
但其注册不证明本Private ANE parser支持或更快。尚未实现/运行，需单独固定
recipe、数值/溢出负控和完整block；有真实收益才进入同库完整文生图/编辑，
编辑必须保留标准1024参考和现有prefix强基线。不继续无依据地增加所有K。

## 版本与回归

- 当前native：`build/qwen-base-ffn-mixed-projection-v2-20261018/libturbocider.dylib`，
  SHA256 `bf1d459f120be4c2b1e4e081f7d76455a7d1e09c2edd9733dd40ed94998fe237`。
- Runtime：`tc-runtime-build-v1-bc1e3ec2435fb4bdaa6787884c73509af2c6a028cd3468b3785553f4cc958f76`。
- 当前native archive：`outputs/qwen-base-ffn-mixed-projection-native-source-v2-20261018.tar.gz`，
  SHA256 `05c7304a3fecffde687e4a6007ca854d966be29eb37e4a1be4ed7a8c2508635b`。
- 实际probe：`build/qwen-ffn-mixed-projection-probe-v1-20261018/libqwen_ffn_projection_probe.dylib`，
  SHA256 `799a0f4c0920774215f264ec73fcf6e20cf91c7a76e33a5c0449a2e774194a66`。
- 当前tools/tests/fingerprint封存：`outputs/qwen-base-ffn-mixed-projection-tools-v1-20261018.tar.gz`，
  SHA256 `e42730306641e4ee43d228b549e73f5a435ad44da4f3770a393a16c703124e7d`。
- 真实MLX lifecycle通过：24cycles/8variants＋joint组合/idle；全部模型窗口
  结束后66项profiler host＋2项原生CPU策略测试通过。日志
  `outputs/qwen-base-ffn-mixed-projection-{host-tests,native-policy-tests}-v2-20261018.log`。
- 首次build v1误用仅库参数名，主动停止自有build进程于App构建前（exit143），
  partial目录/日志保留、无设备样本；v2用正确`TURBOCIDER_BUILD_LIB_ONLY=1`
  在新目录构建成功。没有App/default/安装修改，没有覆盖旧二进制或证据。

没有下载/删除或改写模型master、references，没有停止用户ComfyUI。当前所有
build/test/model/audit进程已结束。目标保持ACTIVE：这一轮排除了混合K，不是
完成全部性能瓶颈，也不是证明W8A8物理INT8或物理GPU/ANE重叠。
