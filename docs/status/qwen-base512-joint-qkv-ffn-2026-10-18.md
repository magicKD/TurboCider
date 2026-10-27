# Qwen Image 2.1 base512：固定Private QKV＋FFN联合运行

后续的[固定份额/K分片筛选与五组质量配对](qwen-base512-joint-share-quality-2026-10-18.md)
使用同一库完成，无新增参数速度收益；也明确记录FFN overflow热重建分支尚需
改为显式fixed-program合同。本文的零热compile/load结论限于实际已观察窗口。

2026-10-18，M4 Max/64GiB。只测试原BF16 base、512²、完整40步文生图和图生图，
无LoRA/student/Z Image、无temporal DiT cache。新功能显式开启，App及产品默认
不变。当前为小样本pilot，发现竞争推理CPU负载，无设备独占/thermal资格，
`qualification_passed=false`。不是“全部瓶颈已解决”的结论。

## 已落地的运行方式

不是为32层分别编译模型，也不是把所有层权重烘焙进一个巨大ANE模型：

| 固定图 | GPU/ANE分工 | 权重与序列变化 |
| --- | --- | --- |
| FFN：1024 rows、H4096、ANE intermediate6144 | ANE/GPU各6144通道，完整SwiGLU＋down，F32 partial join | 原32层gate/up/down逐层stage；长prefill重复固定chunk |
| QKV：512 rows、H4096、N12288 | GPU头512行、ANE尾512行；完整Q/K/V交给原attention | 原三矩阵直接stage进一份W槽位；只在目标1024 rows选择ANE |

两个operator各一份固定program，跨层/step/request复用，不是两颗ANE。QKV其它
rows先保留普通GPU，FFN仍可处理标准1/2参考的长序列；不用热路径临时编译来
改变shape。QKV图无checkpoint常量，也不创建32层的常驻BF16 packed副本。

Private W8/A8、Hadamard与FP16 program output是近似表示，GPU做scale恢复。
输入/图的INT8类型不证明物理ANE INT8 arithmetic；没有ANE硬件counter或物理
GPU/ANE重叠资格。相应限制也在原生回执中保留。

QKV surface槽位65,814,528 bytes（62.77MiB），保守准入估计288MiB，与FFN
共享原2GiB optional allowance；不是整模型/进程/driver峰值。每次stage结束、
ANE实际完成和GPU restore验证后才复用槽位；输出独立owned/evaluated。
首次真实projection对完整GPU QKV做一次relative-L2检查（本机约0.00353），
不额外提交ANE自测。失败后sticky关闭、完整GPU重算并计数，不热重建或调headroom。

本机FFN估计384MiB＋256 bytes，两图合计约672MiB。实际QKV source cache仅有
1067-byte `model.mil`及0-byte `weights.bin`；这证明源图没有烘焙checkpoint，
不是driver编译产物或驻留内存只有1KiB的声称。

`Transformer`普通QKV callback＋split-MLP仍默认拒绝。只有明确的joint consent
才准入；Session另外检查base/shape/数值配方、预算、回调所有权与drain。回调和
consent清除后才导出紧凑prefix，recipe含QKV program身份；失败不发布新bank。

## 同库完整请求：第一轮结果

`stock`不加载observer。每进程73→74→73，首请求单列，后两个完整请求取中位数，
包括encoder、sampler、VAE和PNG。编辑两侧都开启runtime-prefix snapshot，
保留当前强基线；开关之间只差`TURBOCIDER_QWEN21_PRIVATE_QKV`。

| 任务 | FFN-only热中位 | FFN＋QKV热中位 | 耗时下降 |
| --- | ---: | ---: | ---: |
| 文生图 | 33.967238s | 31.783645s | 6.43% |
| 标准单参考重复编辑，prefix hit | 40.361814s | 39.348438s | 2.51% |

首请求：生成off/on为35.919275/34.868943s，编辑off/on为49.393399/47.914136s。
这不是系统严格冷启动，缓存program仍可能需要load；不声称冷启动没有成本。
不能把历史GPU42.67/49.91s或QKV组件5%乘进本轮开关收益。

反向off复测：生成33.774870s、编辑40.484709s，原PNG全部恢复；以这些独立控制
计算，联合路线耗时仍少约5.90%/2.81%。不将两组off请求混合成更有利的分母，
也不以timeline更快的38.84s编辑作为主要速度纪录。尚无置信区间或设备独占资格。

生成每请求实际FFN1312/QKV1248次预测；编辑miss为1440/1248，prefix hit为
1280/1280。全部执行1280个FFN blocks；原有FFN首轮3次能力evaluate另列。
新QKV成功block、预测、W stage严格一致，无GPU fallback，validation calls为1。

## Profiling、变长与质量复核

生成独立timeline已完成：两个evaluate model身份，预测次数2563/2560/2560，
首请求load2、compile0（命中已有program缓存），后两次compile/load均0，
无drop/pending、真实64-byte Metal校准和完成检查通过。selector是可观察边界，
不覆盖不可见driver内部编译。

编辑独立timeline也完成：实际evaluate2691/2560/2560，两个稳定身份，首load2、
compile0，hot compile/load均0。生成和编辑timeline全部PNG都与同配置stock
逐像素一致；两组反向off也与原off全部一致。不是把两个不同program当一个模型。

新observer保持有界，但容量从10万提高至20万，以容纳四请求实际127,223条
selector/CB记录；旧二进制/工具已封存，首六窗仍按旧source身份复核，没有
重标旧记录。三个timeline全部校准通过、无drop/pending/失败调用。

同一进程、同一checkpoint generation的完整变长矩阵：

| 请求 | prefill rows | FFN/QKV预测 | compile/load | 完整wall |
| --- | ---: | ---: | ---: | ---: |
| 文生图 | 1048 | 1312/1248 | 0/2 | 33.571150s |
| 标准单参考 | 5155 | 1440/1248 | 0/0 | 47.377075s |
| 两份标准参考 | 9257 | 1568/1248 | 0/0 | 61.869482s |
| 回到原文生图 | 1048 | 1312/1248 | 0/0 | 31.945403s |

首轮能力3次evaluate另列。四请求都完成40步、1280个FFN block；首尾RGBA exact，
单/双参考输出不同。参考是同一文件重复两份，只验证length/状态恢复，不授予
两个不同主体的多参考语义资格；异构任务的hot median刻意为null。

独立CPU审计重读PNG、请求、source/archive、实际selector计数和完整memory
证据。只有quality复用了冻结GPU PNG，没有复用它的时间作QKV开关分母：

| 比较 | 文生图RGB SSIM | 编辑RGB SSIM |
| --- | ---: | ---: |
| 新增QKV，相对同库FFN-only | 0.98416 /0.99419 | 0.99803–0.99837 |
| 联合路线，相对冻结完整GPU | 0.92195 /0.97766 | 0.99690–0.99750 |

不是bitwise：新增QKV的alpha最大差1/255，relative-L2首检也不代表整图无损。
已实际查看两个对照contact sheet、完整GPU生成/编辑及双参考编辑。生成相对
FFN-only有小的壶嘴/纹理差异；相对原GPU，seed73把手、壶嘴和旋钮几何仍有
可见变化。编辑两侧与GPU控制均有很强的高光/饱和风格，不能把共同缺陷归成
新增QKV，也不能称这些图已通过“保持材质/写实风格”的完整语义验收。
多prompt、文字、人脸身份、不同参考/材质质量矩阵仍未完成，不改为默认。

新9窗共28次完整请求/1120步。每个进程树memory记录完整、command exit0，
系统swap-in/out均0；系统计数不归因外部服务。生成joint stock峰值physical
footprint约17.73GiB，编辑约19.86GiB，变长矩阵约30.36GiB；后者系统压缩/
解压计数约3.37/3.06GiB，不能用“图只有62.77MiB”掩盖KV/encoder等内存成本。

原生FFN `pre_ffn_seconds`在联合路线中不再包含外置QKV部分，约16s→8s的边界
变化不是省了8秒。生成整请求实际只少约2.18秒。FFN累计暴露ANE wait仍约
15–16秒，是后续平衡候选，不是纯ANE硬件kernel时间；stage lifetime包含前序
GPU/norm就绪等待，QKV与FFN这些span不可相加冒充wall。

原step诊断JSON的`hybrid`布尔来自旧CoreML hybrid分支，runtime路径可能为false；
不能据此判为GPU。以最终runtime/QKV成功回执和真实evaluate选择器为准。

已有FFN实验显示：大code cache虽减少staging工作，却没有减少关键路径；
4096→6144通道才有稳定信号。新组合没有启用未经验证的prefetch，也不把stage
提交提前等同于实际重叠。编辑长K/V attention仍保留，不能靠投影优化消除。

## 与本地vpipe的关系

只读核对`../references/vpipe@18ddb3b5bd9c74ab7f2b4408ee78efc6cecd48e7`，checkout
仍干净；没有改参考源码。其Qwen consumer用runtime-weight固定图，QKV先stage，
join后再stage FFN、隐藏在attention下；同一worker串行服务两模块。这个实际
依赖顺序值得保留，不能把两program当两设备。

其默认chunk2048，而这里decode只有1024 rows。`begin`按`(seq-1)/chunk`裁到0，
已armed也不表示执行了ANE；当前本机参考窗口中decode实际预测均0。长编辑
prefill确有调用，但AUTO后来退GPU/周期probe；没有完整图片请求稳定加速资格。
完整数据与误差边界见[原profiling/vpipe报告](qwen-base512-runtime-ane-profile-2026-10-18.md)。
它的合成DiT forward秒数不能同本轮包含encoder/VAE/PNG的40步请求直接作倍率。

本轮的改进是更小固定chunk覆盖目标decode、FFN按intermediate通道切分、GPU
直接stage W8、完整block与完整请求两级验证。灵活性来自host有界计划、固定
chunk和明确fallback，而非每种参考长度各编译一份图。后续调度必须按operator/
phase/rows/precision隔离状态，短序列无合法share应在stage/barrier前回GPU。

## 实现与复跑

- [工程入口](../../tools/profiling/qwen_base/README.md)：`collect.py --private-qkv`、
  `analyze.py`、`joint_audit.py`、`sequence_audit.py`及原始selector/CB trace。
- Native owner：`native/models/qwen21/runtime_private_qkv.hpp/.mm`；准入：
  `runtime_private_qkv_options.hpp`；默认关闭的环境开关：
  `TURBOCIDER_QWEN21_PRIVATE_QKV=1`。
- 新库：`build/qwen-base-joint-qkv-ffn-v1-20261018/libturbocider.dylib`，SHA256
  `1e88bcd9208d677c196148371e97cf8c977c57ffde0ae015ef829532706194cd`。
- Runtime ID：`tc-runtime-build-v1-9eb25fc567dfc4f39d125cdaab146f547395677b3c9bcbb7d5b3a6877ead2b05`。
- 656 inputs原件：`outputs/qwen-base-joint-qkv-ffn-native-source-v1-20261018.tar.gz`，
  SHA256 `ca9f5d136fa82d2a3030c2bc204eff093078f51e34aa57d26014880a729afdb8`。
- 原始窗口：`outputs/qwen-base512-joint-v1-*`。旧库、旧原件与模型权重全部保留；
  不把新source身份回填旧测量。
- 生成/编辑独立审计：`outputs/qwen-base512-joint-v1-{gen,edit}-audit-20261018/audit.json`，
  同目录`comparison.png`；变长审计：`outputs/qwen-base512-joint-v1-sequence-audit-20261018.json`。
- 旧10万条observer配套工具：`outputs/qwen-base-joint-profile-tools-v1-20261018.tar.gz`，
  SHA256 `e7e3045a77781f87bc97453e5c9469fb63c787de7f3ce083ab02902cb921d7e7`；
  当前`collect.py`默认使用新`build/qwen-base-profile-observer-v2-20261018`。
- 当前工具/新增测试封存：`outputs/qwen-base-joint-profile-tools-v2-20261018.tar.gz`，
  SHA256 `655e6313da70ed8e329331f431b99822a64971d8cec697608e04d404838aa37b`。

新库真实MLX生命周期回归通过（24 cycles/8 variants，增加0/1/2参考的显式
joint consent/完整GPU组合数值控制），窗口后又重跑，含第二轮joint后idle内存检查。
设备全部结束后48项profiler host测试＋3项CPU options测试通过，原始日志为
`outputs/qwen-base512-joint-v1-host-tests-20261018.log`；真实MLX回归日志为
`outputs/qwen-base512-joint-v1-lifecycle-20261018.log`。public-disabled QKV TU单独编译通过，
有4条未用private helper警告；这不是完整public库/设备回归。

本轮所有设备、构建、测试和审计进程均已退出。磁盘仍约6GiB free；没有下载、
删除、停止ComfyUI或改动系统设置。后续优先在联合路线重新筛选5120/6144份额，
分析FFN等待与长参考attention，并补语义质量矩阵；不能从旧FFN-only分区最优
直接推导新的联合最优。
