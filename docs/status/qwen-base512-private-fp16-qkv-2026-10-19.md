# Qwen base512：Private FP16 QKV与输入就绪调度

2026-10-19，Apple M4 Max/64GiB，macOS26.6.2/25G83。只用原BF16 base checkpoint，
不涉及LoRA/student/Z Image。接续
[W producer合并未保留到完整投影的结果](qwen-base512-qkv-w-stage-batch-2026-10-19.md)。

本轮验证了一个不同精度/输入表示路径，并修复候选中约3ms的输入准备调度成本。
但最好的受测FP16完整投影仍约11.21ms，原生W8约5.17ms，**没有替换主runtime、
改变App/default或取得新的完整40步图片纪录**。总体优化目标仍ACTIVE。

## 假设与范围

假设：固定512-row FP16 MatMul可省掉W8的H128旋转/量化与图内dequantize；
代价是W输入从48MiB变成96MiB，且typed端口/图lowering可能改变实际执行。
不能因为少了几个源算子就预设ANE更快，也不从MIL dtype推断物理MAC位宽。

已有公共Core ML FP16实验是4096 source rows、1536-row ANE分担，并且后来的
1024² AUTO会退回GPU；不是本轮Private512-row/base512的证据。没有拿它的
旧倍率或其它SoC数字作分母，相关历史见
[三源直填](runtime-ane-qwen-qkv-parts.md)和[AUTO完整block](runtime-ane-qwen-qkv-auto.md)。

原W8强控制直接调用当前native `RuntimePrivateQkv`，GPU头512行/ANE尾512行，
保持原checkpoint、数据、完整输出完成和原W8实际预测/校验回执。不是改用更慢
的拆分GPU或旧库时间。完整GPU作为独立参考，三个原Q/K/V GEMM和concat都完成。

FP16只保存当前层的`[12288,4096]` FP16 surface，GPU一次直接从三份原BF16 W
做integer-RNE cast；不构造逐层packed副本。输入用原GPU upload transpose到
FP16 surface，输出经原恢复核写入独立owned BF16 tail并与GPU head重组。
source和GPU allocation owner保留到真实完成，非有限/溢出不发布输出。

FP16显式slot为117440512 bytes，W8为65814528 bytes。两个FP16图在对照进程中
共享同一bank、串行完成后复用，不是两份权重bank。小图源码不等于进程小，也
不是32层checkpoint嵌入ANE模型。新FP16只实际测试了两层W；主W8的32层/变长/
完整40步证据仍属于之前的独立研究，不借给本候选。

## 同时控制图分解

原生`fp16_program`按N512切成24个MatMul；当前W8 emitter是一个整MatMul。
因此本轮保留两种FP16：`fp16-tiled`用原emitter；`fp16-single`手写相同整体
端口/尺寸、无Q/DQ的一个MatMul。每图只在owner初始化构建一次，没有按层换图。
另有`--single-only`完全不构造tiled Program，排除额外候选驻留的混淆。

所有新测量使用已冻结、已审计的实际base40 GPU请求首decode normalized输入：
文生图和单标准1024参考编辑，seed73、512²。原来源完成首sampler step后捕获
layer0，不是这轮又执行40步。本轮只重放投影，没有新的DiT步骤、encoder/VAE/
PNG。来源请求/原数组hash/实际六份W完整payload均重新核对；历史时间不参与排名。
B是normalized输入的小扰动，不是第二个seed/prompt；换W使用真实layer1 Q/K/V，
并非自然完整layer1 block。

## 正式stock完整投影

每arm2 warm，按rep轮转，输入A/B交替；完整wall包含动态W准备、输入准备、GPU
head、Private请求、恢复/重组和最终输出完成。单位ms：

| arm | 生成16，三Program对照 | 编辑16，三Program对照 | 生成32，仅W8＋single |
| --- | ---: | ---: | ---: |
| GPU完整QKV | 7.107959 | 7.112959 | 7.105146 |
| 当前native W8 | 5.172875 | 5.102063 | 5.191813 |
| FP16 tiled | 14.671958 | 13.753688 | 未构造 |
| FP16 single | 14.447958 | 13.551355 | 14.181500 |

改成整MatMul没有恢复竞争力。tiled/single的完整输出在实际数据上逐位一致，
但不能据此断言opaque compiler内部融合完全相同。省去tiled Program后，single
仍约为同窗W8的2.73倍耗时；不能将慢点只归给同时保留三个候选模型。
初版三arm tiled-only pilot约14.49ms，单列保留，不混入上述正式统计。

## 从时间线找到候选内部的输入调度成本

最初FP16把A16上传放在leading transfer里，在提交它后立即提交大GPU head。
W8则先完成A8 producer，再提交Private请求和GPU head。独立timeline发现，
FP16权重cast所在GPU CB约0.48ms，但leading A16 CB的GPU时间窗口约2.7–3.25ms。
这可能含资源竞争/抢占，不能当成独占upload kernel成本或ANE硬件时间。

新增`--single-only --compare-prepack`，两条FP16策略共用**同一个single Program、
同一bank与同一数值路径**，只在Private/GPU-head提交前完成独立A16 upload；
仍有真实ready/done、源/输出owner、状态校验与全部完成等待，不是删除fence。

| 同窗stock16 | 完整QKV wall |
| --- | ---: |
| native W8 | 5.167188ms |
| FP16，A16在leading transfer | 14.278646ms |
| FP16，先完成A16 | 11.205354ms |

候选内部约少21.52%；独立timeline为13.557604→11.231938ms，仅作诊断。
同窗实际GPU CB/host lifetime进一步分开：

| FP16 single边界 | leading A16策略 | prepack策略 |
| --- | ---: | ---: |
| W cast CB GPU elapsed | 0.477563ms | 0.482563ms |
| A16 CB GPU elapsed | 2.599250ms | 0.033958ms |
| output-restore CB lifetime | 12.570271ms | 9.962709ms |
| output-restore CB GPU elapsed | 0.136251ms | 0.143105ms |

因此GPU输入准备的调度窗口确实缩短；剩余大头是Private执行/完成链路的暴露
等待，而非W cast或restore算术独占时间。async evaluate API span本身仅约几十
微秒，不等于整个预测。CB elapsed/lifetime、输入等待、driver/ANE链路可重叠，
不能相加或唯一定位成物理ANE compute。无硬件counters/物理INT8 MAC证明。

即使修复这项成本，FP16仍约为同窗W8的2.17倍。较低误差不自动满足速度目标，
不把候选内部21.5%的改进包装成现有产品或整图提速。没有据此扩展主runtime。

## 完整数值、来源与复算

正式六窗共81份完整BF16数组，全部shape/extent/finite校验通过；GPU head保留，
输入/权重A/B/A及每策略末输出恢复通过，同case重复/timeline共同字段逐位一致。
同一single Program的prepack/leading策略所有完整输出严格bits exact。

生成来源W0：W8 full relative-L2约0.003532、ANE tail约0.004995；FP16 full约
0.001017、tail约0.001438。两层W及编辑来源也过原组件门槛，FP16较接近GPU。
这是局部raw投影，不是完整图片、全部32层或人物/文字/多主体语义质量资格。

auditor重读原六份BF16 W、共192MiB完整payload，并核对新helper和冻结来源的
全部SHA；新input_a和原normalized完整bytes相同。原GPU A/换W控制与冻结数据
完整逐位相同。B不能拿历史`alternate_raw_reference`直接对照：旧实验先扰动
hidden再做layer_norm，本轮先normalized再扰动。首次auditor误配这两个B，被拒
日志保留；修正后完整重算typed BF16 B配方，并对生成/编辑各三份1024×4096
投影做独立CPU F32 GEMM＋BF16 RNE，共25165824个输出，全部与新GPU B一致。
另有576个固定坐标的完整K4096 F64见证；不是全CPU FFN/全部投影F64 oracle。

CPU/BLAS每case保留9条RuntimeWarning（divide/overflow/invalid等），完整结果
有限且上述比对exact，没有全局屏蔽警告、修改设备结果或放宽数值门槛；原因未
建立。旧B配对错误的失败不是FP16数值错误，也没有被重标为成功原件。

## 实际图、调用数与测量边界

正式六窗416个measured投影、406次实际Private预测（W8 148，FP16 258）；其中
258含tiled66、single192，single内含两种输入准备策略。两个独立timeline分别
观测3个候选模型/66次预测，以及共用single的2个模型/66次预测；hot compile/
load均0，实际per-scope/每model调用与native计数一致，Metal数据/时钟校准、
drain、无drop/pending通过。不是三份生产模型或32个逐层模型。

process-tree峰值physical footprint约1.13–1.23GiB，系统swap-in/out0；只加载
QKV组件，不能当作完整Qwen内存。存在竞争推理CPU负载，无GPU独占/统计/thermal
资格，统一`qualification_passed=false`。全部设备窗串行，自有build/test/CPU
数组复算在模型窗外。没有停止ComfyUI、改App/default、下载或删除权重。

## 工程、磁盘和交付

- `fp16_qkv.hpp`：原BF16→FP16直接GPU cast、一个current-layer bank、tiled/
  single动态权重图、leading/prepack两种就绪策略；实际有限域检查和真实完成。
- `fp16_qkv_{probe.mm,collect.py,audit.py}`：绑定原native661 ABI、冻结真实输入、
  原W8强控制、full-array/source/CPU/selector/memory独立复核。
- 当前helper `build/qwen-fp16-qkv-probe-v4-20261019/libqwen_fp16_qkv_probe.dylib`。
  CAPI_v3/v4分名，不能把旧helper当成新调用签名；旧成功窗分别绑定v2/v3原库。
  对应源码封存`outputs/qwen-fp16-qkv-tools-pilot-v2-20261019.tar.gz`与v3归档。
- 主native仍是`build/qwen-base-gpu-dual-v1-20261019/libturbocider.dylib`，SHA256
  `7da7a7125bd7cb401344bf885c2425ec7eacc14d81b0563fd45317a94b4b260a`，源码/ABI未改。
- 最终独立审计`outputs/qwen-fp16-qkv-full-audit-v5-20261019/audit.json`；初版误配B
  的失败日志、后续v2/v3/v4审计和所有设备原件保留。
- 新增8项host负控，最终161项profiler host测试见
  `outputs/qwen-fp16-qkv-host-tests-v3-20261019.log`，不等于161项图片/硬件资格。
- 本轮进一步无损硬链接去重24个旧I8副本路径与两对相同raw投影文件，完整路径/
  bytes/hash保留，消除约1.16GiB重复存储；不当成runtime内存或推理速度收益。
  一次透明文件压缩尝试未减少blocks，未宣称释放空间。磁盘仍约2.3GiB，后续须
  先预算或无损整理新中间结果，维持2GiB reserve，不碰模型/旧原件的内容。
- 工具/测试/本报告封存`outputs/qwen-fp16-qkv-tools-final-v1-20261019.tar.gz`；摘要
  索引`outputs/qwen-fp16-qkv-artifacts-v1-20261019.json`。

```sh
TURBOCIDER_NATIVE_OUT=build/FRESH-FP16-QKV \
TURBOCIDER_NATIVE_LIBRARY_DIR=build/qwen-base-gpu-dual-v1-20261019 \
  bash tools/native/build_qwen_fp16_qkv_probe.sh
.venv/bin/python -B tools/profiling/qwen_base/fp16_qkv_collect.py \
  --library build/FRESH-FP16-QKV/libqwen_fp16_qkv_probe.dylib \
  --native-library build/qwen-base-gpu-dual-v1-20261019/libturbocider.dylib \
  --single-only --compare-prepack --case generate --mode stock \
  --output outputs/FRESH-FP16-QKV
```

编辑改`--case edit`；timeline用独立进程。设备全部退出后运行auditor，逐个
`--cohort ROOT MATCHED_HELPER`配合native source archive、历史工具archive与fresh
输出。完整40步推荐仍保留原fixed FFN/QKV＋one-hop lookahead，GPU-dual默认OFF。
