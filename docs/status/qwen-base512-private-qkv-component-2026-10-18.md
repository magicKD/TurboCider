# Qwen base512：固定 Private W8 QKV 组件筛选

后续已完成显式原生联合接入与整请求实验，见[固定QKV＋FFN报告](qwen-base512-joint-qkv-ffn-2026-10-18.md)。
本文冻结组件阶段的输入、二进制与数据，不将后续整请求结果回填为组件结果。

2026-10-18，M4 Max/64GiB。继续优化原base文生图、图生图，不使用LoRA/student，
不涉及Z Image。结果是**局部完整block的可行性信号**，尚未接入完整图片请求，
也没有与当前FFN-ANE/prefix路线组合；不得更新整请求速度纪录或相乘倍率。

## 实现与真实输入

`tools/profiling/qwen_base/qkv_ane.hpp`封装一个固定runtime-weight Private MatMul：
H4096→Q/K/V共12288输出，固定256或512行；图没有checkpoint常量。Q、K、V
从原三份BF16矩阵分别GPU H128量化到同一W surface的三个row区间，不制作或
永久保留32层的BF16 packed副本。激活H128/A8转置，ANE返回归一化FP16输出，
GPU按token/row scales在FP32中恢复并写回独立BF16 allocation。

GPU计算1024目标tokens的头部，ANE计算尾部256/512行；完整QKV重新组合后，
Q/K RMSNorm、RoPE、完整attention、out projection、残差及完整FFN仍在GPU。
不是切开attention或忽略另一半K/V。ANE/GPU、各staging ticket与输出消费者
全部完成后才结束计时；异常会drain已提交producer，不拿旧surface当成功输出。

真实源使用原BF16 DiT/encoder、40步调度、512²和seed73：先完成首步，再捕获
第一decode的真实layer0参数/输入及普通compiled block。编辑保留单张1024参考，
不降参考尺寸。**诊断在step1 layer0之后退出，没有完成40步或输出PNG。**
另用小幅扰动的真实hidden输入作A/B/A，而不是把它叫第二个独立seed/prompt。

旧捕获入口只准入staged20 base或student，新增窄的base40 resident GPU捕获
predicate；保持原有executor/cache/streaming/quantized-source排斥条件，source
request禁止approximation和LoRA。新库只改变诊断准入，不改变普通生成/编辑
算术；QKV候选的近似计算发生在诊断consumer里，不默认启用产品路线。

## 对照边界

每个进程四条路线，2个warmup/arm，10个measured/arm，按轮旋转顺序；另有一窗
20个measured/arm复测。A/B输入交替，所有GPU输出都必须新建/实际完成。

| 路线 | 实际计入 |
| --- | --- |
| gpu-original | 捕获到的普通完整compiled GPU block，主分母 |
| gpu-staged | norm→三GPU投影→完整GPU consumer，单独揭示拆分边界成本 |
| private-prestaged | ANE/GPU分投影及完整consumer，W已准备；仅作上界 |
| private-restaged | 上述加每次三份W的实际GPU量化/提交/等待，主要候选 |

四条路线的FFN都在GPU，用于隔离QKV。当前产品最佳路线已有FFN-ANE，本研究
不能替代与它的完整请求对照。stage/submit/join记录是host spans，不是硬件
kernel时间，不能把重叠span相加归因。

## 本机结果

以下为各独立stock窗口完整block中位数，单位ms；不是QKV kernel独占时长，
也不是完整出图秒数。

| 场景/固定图 | 普通GPU | 已备W | 含重备W | 含W耗时下降 |
| --- | ---: | ---: | ---: | ---: |
| 文生图，256行/tiled | 32.8422 | 31.7610 | 32.7634 | 0.24% |
| 文生图，512行/tiled | 32.8858 | 30.6366 | 31.5852 | 3.95% |
| 文生图，512行/single | 32.8614 | 30.3427 | 31.2140 | 5.01% |
| 文生图，512行/single，20轮复测 | 32.8290 | 见raw | 31.1835 | 5.01% |
| 标准参考编辑，512行/tiled | 38.4896 | 36.3567 | 37.2569 | 3.20% |
| 标准参考编辑，512行/single | 38.4821 | 见raw | 36.7708 | 4.45% |

single为native已有的单MatMul发射器，tiled为按N512切成24个MatMul后concat的
诊断variant。两者结果数组一致，但single更快；不能只凭历史on-chip权重大小
经验推断手工tiling必胜。没有硬件counter，未确定差异来自哪一项lowering成本。

gpu-staged通常比原GPU多约0.2–0.3ms，主要候选仍对原完整compiled GPU分母，
没有把这个弱一点的拆分控制拿来放大收益。256行的净信号很小，不据此采用。

## 数值、动态权重和所有权

预检切换激活A→B→A及真实layer0 QKV→layer1 QKV→layer0 QKV，使用同一个
program，检查原输出在后续调用后不变、恢复输出hash一致。三份权重逐层stage，
没有在weight变化时重新创建program。原借用的输入、普通输出和六份QKV权重
测后hash均保持，原权重未改写。

独立CPU审计直接读取原checkpoint两层、六个QKV payload（不是仅查文件名），
完整payload SHA与native捕获一致；还重读全部保存的BF16控制数组：

| 512行候选 | QKV raw relative-L2 | 完整block relative-L2 |
| --- | ---: | ---: |
| 文生图 | 0.003532 | 0.001403 |
| 编辑 | 0.003535 | 0.001195 |

GPU-staged相对原完整GPU输出exact；Private候选不是bitwise等价。最大abs差的
单位是hidden tensor数值，不能当成PNG像素差。重复/最后一轮consumer也按完整
数组复核，不以某次预检代替后面的输出。没有完成多prompt或整图质量资格。

另对两个实际normalized输入的全部1024行，以原layer0权重做独立CPU F32 dot、
BF16 RNE。Q/K/V结果在此数据上都与GPU raw reference逐元素相同。NumPy/BLAS
同时报告过divide-by-zero/overflow/invalid浮点状态警告；完整CPU输出有限且
比较通过，v2 audit保留每个warning、有限性与max值，未断言warning根因或把它
悄悄抹去。它不是Private ANE的数值错误证据，也不代替完整模型轨迹验证。

## 固定图、实际观测与资源

tiled与single各有一个正确的timeline窗口：每进程都观察到一个evaluate model
identity，29次成功预测；切换权重/输入与timing循环期间compile/load均0。两个
进程的ordinal1不是跨进程同一个对象。两窗58次真实selector与native计数吻合，
无drop/pending，真实64-byte blit校准通过；没有把API名称当成物理INT8或重叠证明。

512行surface槽位账本65814528 bytes，约62.77MiB；256行58474496 bytes。
这是固定槽位，不含GPU/master权重、scale cache、compiled driver服务、allocator
cache等。`setup_seconds`只覆盖Program发射/创建，不含之前Device/Surface构造，
不是完整冷启动时间。整进程树另采样load/capture/probe/dump/exit，不用槽位字节
冒充进程峰值。外部服务和driver未独立归因。

8个成功窗口（6 stock、2 timeline）包含360个measured与64个warmup完整block
重放，252个成功Private预测；并非8个完整图片请求。所有窗口系统swap-out均0，
source/array后处理不与本机其他自有模型benchmark并行。部分窗口发现竞争推理
CPU负载；没有GPU独占或长时间thermal资格，整体`qualification_passed=false`。

三次失败保留且不计样本：最初请求未关audio；随后被旧observer准入拒绝；第一次
timeline把source encoder/prefill误归到-2 blit校准scope，被验证器拒绝。最后一项
数值重放完成，但其观测不授予资格；修正scope(-1)后在新目录实际重跑，没有
回写或重标旧原件。

## 实现、构建和审计

- 当前native：`build/qwen-base-qkv-capture-v1-20261018/libturbocider.dylib`，
  SHA256 `75f8eb7e0904d9eca65dcd9e2b34533ad2c25e3d302f0f3eeb587c08433c0781`。
  Runtime ID `tc-runtime-build-v1-24cb39345cd4924c76c2da5481bc170a713bed63fd8183ac7f0d4585e95a8391`。
- 653个native/build inputs已核对current/archive：
  `outputs/qwen-base-qkv-capture-native-source-v1-20261018.tar.gz`，SHA256
  `e0b446c824d4ec079e3ef0ba871990fb749212f16bd70190e475ad87e4366806`。
- 实际probe：`build/qwen-base-private-qkv-probe-v3-20261018/libqwen_base_qkv_probe.dylib`，
  SHA256 `c495195ef3e43e8cdd0da8006791f623ddf6ff556bb38d4653dd0a6c03d4dda1`。
  旧v1/v2 probe与旧prefix/v8库保留，不混用其source身份。
- 工程：[说明](../../tools/profiling/qwen_base/README.md)，`qkv_ane.hpp`、
  `qkv_probe.mm`、`qkv_collect.py`、`qkv_audit.py`；构建脚本
  `tools/native/build_qwen_base_qkv_probe.sh`。
- 原始窗口：`outputs/qwen-base512-private-qkv-*`；独立审计与派生trace在
  `outputs/qwen-base512-private-qkv-audit-v3-20261018/`。v1/v2审计保留。
  trace将实际sample的arm/rep/warmup映射回API/CB scopes，不伪造全请求时长。
  新工具在启动模型前要求256MiB输出余量外再保留2GiB free；不自动删除文件。
- 40项相关host回归通过：`outputs/qwen-base-private-qkv-host-tests-v5-20261018.log`。
  新库的真实MLX生命周期回归通过：
  `outputs/qwen-base-qkv-capture-lifecycle-v1-20261018.log`（原24 cycles/8 variants）。
  它们不是40项设备性能或图片语义资格测试。

本轮尚未改变产品QKV路由或默认，也没有把它与现有FFN模型的worker/预算合并。
`Transformer`当前仍拒绝普通QKV callback与split-MLP并存；下一步需要显式、
受限的组合合同，联合可选内存准入、真实完成后的bank复用、异常/取消drain，
再对当前最快FFN（编辑还包括prefix）做完整请求on/off验证。QKV第一消费者前
必须join；FFN stage窗口改变、两program的同ANE竞争或数值误差都可能吃掉本次
局部收益。不能把本文5%直接套到整次出图。
