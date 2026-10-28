# Qwen base512：单组QKV权重前看，固定图零热编译

2026-10-19，M4 Max/64GiB。仅原BF16 base、512×512、实际40步文生图/图生图，
无LoRA/student/Z Image、无temporal DiT cache。继续
[保留首步GPU QKV的text-prefix基线](qwen-base512-text-prefix-qkv-step0-2026-10-19.md)。

本轮新增默认关闭的单组QKV权重预取：**当前FFN提交后，提前准备下一次QKV权重**。
不增加QKV权重bank、不复制checkpoint、不增加Program。匹配完整请求中生成约
快2.2%、编辑约快2.1%；反向控制仍有约2.7%/1.8%的信号。增量输出全部RGBA
exact；这不是原W8路线相对完整GPU逐位无损的声明，也不是所有瓶颈已解决。

## 固定图与灵活性

| operator | 固定ANE shape与分工 | 跨层/序列变化 |
| --- | --- | --- |
| FFN | rows1024、H4096、intermediate6144；GPU/ANE各6144通道，F32 partial join | 同一图接受32层原gate/up/down，长prefill按1024分块、尾块padding并裁切 |
| QKV | rows512、H4096、N12288；目标1024行中GPU头512/ANE尾512 | 同一图接受各层Q/K/V；长prefill直接GPU，后39步复用固定图 |

这是两个operator各一个固定Program，不是32层各一图，也不是两颗ANE。
W8/A8是runtime输入/量化数据流；没有硬件counter证明物理INT8 MAC或物理引擎
重叠。图内INT8类型、Private evaluate和GPU CB区间均不能替代此证明。

小源图也不意味着整个进程小：QKV explicit slots为65,814,528 bytes，准入
估计288MiB；FFN估计约384MiB，两者共享原2GiB optional allowance。没有增加
这些bank/估计，也没有把原BF16权重、encoder/KV、driver或输出内存排除出进程测量。

已有fixed-program策略保持不变：正常热路径不换Program，数值失败sticky停用并
完整GPU重算，不通过调整headroom重新编译来“修复”性能窗。首进程仍有setup/
load成本，不称严格冷启动零成本；GPU首遇新shape的编译也不在ANE零热编译结论内。

## 前看如何保持安全

新增`TURBOCIDER_QWEN21_PRIVATE_QKV_PREFETCH=1`，必须显式joint base40 resident512、
fixed-program、untimed fixed1 channel FFN及实际submission fence。App/default未改。

`HybridFfn::AfterSubmit`只在实际ANE launch和GPU async_eval提交后、finish前调用；
提交不等于物理执行。此时本层QKV早已完成，其W bank可用，而FFN使用自己的bank。
预取只做下一次QKV的三个W8 producer，不提前运行下一次ANE projection。

future最多一组，持有原Tensor引用；消费时核对source generation、layer、rows及
三个实际Tensor ID，不仅核对shape。不同权重/请求先drain再discard/restage。
自身projection运行时不允许改写bank；成功输出独立owned/evaluated，不能借旧scratch。
取消/错误走已有drain；失败不伪装成成功命中，未做真实driver timeout/reset资格测试。

预测下一层/step时先排除GPU路线、长prefill及最后step39/layer31，避免无用W搬运。
layer31可为下一step的layer0准备权重。text-prefix命中仍保留首步GPU QKV，不能
为增加ANE命中率重引入上一轮的画质反例；policy进入prefix recipe命名空间。

## 同库stock完整请求

每进程seed73→74→73，首请求单列，后两次完整request wall取中位，包含encoder、
40步、VAE及PNG。仅预取env0/1不同。生成两侧均保留正确text-prefix/GPU-step0，
编辑两侧均保留runtime-prefix、标准1024参考图和4096参考tokens。

| 对照 | 预取off热中位 | 预取on热中位 | 耗时下降 |
| --- | ---: | ---: | ---: |
| 生成首配对 | 29.486912s | 28.835203s | 2.21% |
| 生成反向off/独立on重复 | 29.634552s | 28.824001s | 2.74% |
| 单参考重复编辑 | 36.980384s | 36.217944s | 2.06% |
| 编辑反向off/原on | 36.878182s | 36.217944s | 1.79% |

不混合不同off分母来放大收益。生成首配对first为31.318091/30.820900s，编辑为
45.695659/44.969914s；编辑反向first47.474508s说明首请求存在波动，未混入热样本。
timeline的生成28.896849s、编辑36.083205s只是诊断，不参与stock速度排序。

四组完整配对全部RGBA exact；同recipe反向、独立重复和timeline共有36次完整
图像重放检查通过。另与上一轮冻结的生成/编辑请求及6张完整RGBA比较全部exact，
历史像素只作回归，不借旧库时间当新分母。没有新增人工语义资格；原W8路径已知
与完整GPU有近似差异，调度exact不能消除那些差异。

## 瓶颈归因：等待被隐藏，但有竞争成本

新回执将QKV的weight join等待单列，明确排除norm就绪等待。每请求累计值：

- 生成off约0.831–0.835s，on约0.011–0.013s；
- 编辑hot off约0.935–0.985s，on约0.025–0.027s。

因此QKV权重生产到消费的暴露等待确实显著减少，而非只改“命中率”标签。
但生成FFN caller约14.33–14.45s升至14.62–14.79s，FFN async wait也略增；
与新增GPU工作/带宽竞争一致，不能据host spans唯一定位物理原因。
完整请求净少约0.65–0.76s，而不是把约0.8–0.9s等待全部当成提速。

`producer_lifetime`约14s/请求**包含整个lookahead驻留区间**，不是14s GPU kernel
或权重量化成本。QKV stage lifetime边界也因claim改变，不拿其差值充当纯硬件节省。
这些重叠累计spans不能与FFN/Metal CB存活区间相加成wall。

下一主线仍是FFN：当前生成hot async wait约13.8–13.9s，但这是caller等待边界，
不是已分离的ANE硬件计算时间。长参考还保留attention/KV和padding成本。此前
down tile/FP16 gate-up/大W cache/更大异步化的负结果保持有效，不因这次预取有效
就重新默认打开它们。后续应独立测调度前沿、GPU complement与恢复交接，再做
完整请求验收；继续放大预取距离或bank数没有本轮证据支持。

## 实际预测、变长与零热编译

生成每请求FFN为1312/1280/1280，QKV均1248、GPU-QKV均32；预取每次1248，
producer为3744。编辑QKV为1248/1280/1280，预取1248/1279/1279；命中请求
第一层没有更早FFN，因此其QKV按需stage，不伪造第1280次预取。
所有成功窗attempt/submission/claim/completion一致，无discard/retired/failure/
pending、无fallback/retry/headroom变化；FFN每owner仍一个Program。

生成、编辑命中、变长各有独立timeline：每进程两个evaluate model身份，hot
compile/load均0；实际per-model evaluate与成功native计数一致，FFN首轮3次
capability调用另列。scope结束真实drain、无drop/pending、64-byte Metal完成/
时钟校准通过。结论限可观察Private client selectors，不覆盖opaque driver内部。

同一进程同一checkpoint的四次完整请求：

| 请求 | prefill rows | FFN/QKV预测 | 完整wall |
| --- | ---: | ---: | ---: |
| 生成 | 1048 | 1312/1248 | 30.901309s |
| 单参考编辑 | 5155 | 1440/1248 | 44.223794s |
| 双参考编辑 | 9257 | 1568/1248 | 61.671957s |
| 恢复原生成 | 1048 | 1312/1248 | 29.343450s |

首后无新ANE compile/load，首尾PNG exact，单/双参考输出不同。双参考是同一文件
重复两份，仅验证length/状态恢复，不授予多主体语义资格；异构hot median为null。
本矩阵没有匹配off窗口，不从这些时间推导“变长预取提速百分比”。

## vpipe对照应如何理解

再次只读核对`../references/vpipe@18ddb3b5bd9c74ab7f2b4408ee78efc6cecd48e7`，
checkout仍干净。`ane-ffn.h`默认chunk2048，`begin`按`(seq-1)/chunk`限制合法share；
本任务decode1024因此可以实际ANE预测为0，即便开关已开启。Qwen consumer也是
先stage QKV、join后stage FFN并隐藏在attention下，同一ANE worker串行服务。

原本机完整对照见[profiling/vpipe报告](qwen-base512-runtime-ane-profile-2026-10-18.md)：
vpipe生成decode GPU/FFN/FFN+QKV约1.0617/1.0619/1.0622s，实际ANE预测均0；编辑
长prefill有调用，但AUTO后来退GPU并周期probe，未取得稳定完整图片加速资格。
这些是合成DiT forward，不与本轮含encoder/VAE/PNG的40步wall直接作倍率。

可迁移的是动态权重固定图、有界bank/完成与fallback合同；改进重点是合法小chunk、
通道分工、stage前zero-share检查，以及本轮的跨operator单组前看。调度状态应按
operator/phase/rows/precision隔离。不能仅打开ANE或借作者其他硬件数字证明加速。

## 工程、原件和复跑

- native：`build/qwen-base-qkv-lookahead-v1-20261019/libturbocider.dylib`，SHA256
  `eb4631e0bfa75888606dae799ff5a2f04aa67a0b227bb9245b46ebe70a0cf734`。
- Runtime ID：`tc-runtime-build-v1-61ee908be83f296f634e82a782e30008d9c565c27d2d74476645df88474e187c`。
- 659项native源码：`outputs/qwen-base-qkv-lookahead-native-source-v1-20261019.tar.gz`，
  SHA256 `b80009a84c61ed09d3757c1c3249a93036bc68834ecff2a99eecc30cdb8589bd`。
- 最终独立审计：`outputs/qwen-base512-qkv-lookahead-full-audit-v4-20261019/audit.json`。
  含10个完整窗口/31请求/1240步、合成探针全数组oracle、所有实际输入/PNG/
  source/memory/selectors以及冻结回归，不仅汇总collector自报结果。
- 合成探针：`qkv_lookahead_probe.mm`/`qkv_lookahead_collect.py`，真实固定QKV图，
  对角BF16 A/B/A、同layer/rows但不同Tensor IDs、换generation、claimed-drain、
  disable负控；5次evaluate，完整oracle relative-L2约0.00680/0.00690，旧输出与
  源权重未改。helper构建绑定native ID，进入C++前拒绝旧ABI。
- 新collector参数`--private-qkv-prefetch`默认off，并明确发env0；旧Python
  `environment()`字典默认不新增key以保持兼容。新回执严格要求完整计数与drain。
- 最终105项profiler host测试通过：`outputs/qwen-qkv-lookahead-host-tests-v6-20261019.log`。
  native policy1项及真实MLX lifecycle1项分别见`outputs/qwen-qkv-lookahead-native-options-v1-20261019.log`
  与`outputs/qwen-qkv-lookahead-lifecycle-v1-20261019.log`；不合并成硬件资格数。
- 工具/测试/本报告封存：`outputs/qwen-qkv-lookahead-tools-v2-20261019.tar.gz`，
  binary/source/archive摘要见`outputs/qwen-qkv-lookahead-artifacts-v1-20261019.json`。
  v1采集器误把准备工作放进校准scope，以及初版auditor误读reference字典的失败
  日志均保留；修正的是采集/审计边界，没有放宽数值或成功计数门槛。

```sh
.venv/bin/python -B tools/profiling/qwen_base/collect.py \
  --library build/qwen-base-qkv-lookahead-v1-20261019/libturbocider.dylib \
  --route runtime --channels 6144 --private-qkv --fixed-program \
  --ffn-activation sigmoid-complement --private-qkv-prefetch \
  --case generate --mode stock --runtime-text-prefix-snapshot \
  --text-prefix-qkv-step0 gpu --output outputs/FRESH-QKV-LOOKAHEAD
```

off对照只去掉`--private-qkv-prefetch`；编辑改`--case edit`，去掉两项text参数，
改用`--runtime-prefix-snapshot`。timeline独立进程；变长用`--scenario sequence-matrix
--requests 4 --mode timeline`，不带prefix复用。所有模型窗串行，不并行build/test/
CPU数组审计/看图。完成后用`qkv_lookahead_audit.py --help`指定`--pair`、`--window`、
`--probe`、`--frozen`及上述完整native source archive独立重读。

进程树physical footprint峰值约17.05–30.24GiB，系统swap-in/out均0；变长窗有约
614MiB compression，不归因单独ANE。全部窗口存在竞争推理CPU负载，无设备独占/
统计/thermal资格，统一`qualification_passed=false`。没有停止ComfyUI、下载/删除
权重缓存或更改App/default。整体优化目标仍ACTIVE，不把约2%增量称为全部瓶颈解决。
