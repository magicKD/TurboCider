# Qwen base512：固定runtime FFN投影粒度与完整请求

后续已完成[gate/up与down分别隔离](qwen-base512-ffn-mixed-projection-2026-10-18.md)，
两种混合策略同样未提速。本文保留原构建、完整请求和数据，不重标后续版本。

2026-10-18，M4 Max/64GiB。只用原BF16 base、512²、实际40步、文生图/图生图，
无LoRA/student/Z Image、无temporal DiT cache。继续
[固定Program合同](qwen-base512-fixed-program-2026-10-18.md)和
[联合QKV＋FFN](qwen-base512-joint-qkv-ffn-2026-10-18.md)。
本轮只检验更大/整K的FFN MatMul是否有利；目前是负结果，没有新的速度收益，
不修改App或产品默认。全部10个完整请求窗口及独立审计已结束，30次请求/
1200个实际采样步；另有3个完整block窗口，不混称完整图片。

## 实现与假设

已有QKV single比手工N分片更快，不能由此推断FFN也应合并。新的可证伪假设是：
在权重、IO、GPU complement、Hadamard和F32 join均不变时，减少FFN图内K分片
能否降低真实block或完整请求时间。会改变内部lowering与reduction，也可能增加
activation/中间值压力；原因不能仅凭节点数确定。

`TURBOCIDER_PRIVATE_ANE_W8_PROJECTION`在owner初始化时捕获：

| policy | gate/up的K分片 | down的K分片 | H4096/C6144图内MatMul数 |
| --- | ---: | ---: | ---: |
| descriptor（原行为） | 2048 | 2048 | 7 |
| k4096 | 4096 | 4096＋2048 | 4 |
| single | 4096 | 6144 | 3 |

descriptor继续采用原发射器路径和原clamp；非默认的完整operand直接
dequantize→MatMul，不额外加identity slice。上述是MIL节点，不是硬件tile或
每block的predict次数；每个FFN chunk仍只一次predict。

实验只允许显式fixed-program、授权Private W8、ungrouped Sylvester原base路径；
拒绝LoRA、grouped、Comfy、BF16-value/native-LoRA等配方。输入/输出和权重slots
ABI不变，非默认policy进入source recipe/executor/prefix身份。正常1024-row
FFN图和512-row QKV图各一份，跨32层、全部步骤和请求共享动态权重，不做每层
常量模型，也不热切策略。原来的溢出禁止重建/完整GPU fallback合同保持。

默认FFN6144的CPU emitter实测MIL 8,895 bytes、model-independent Hadamard常量
6,291,584 bytes（约6MiB），不是32层checkpoint；三个策略常量字节不变。QKV
源图仍没有checkpoint常量。这不代表driver编译产物/ANE驻留只占这些字节：
FFN slots账本165,707,776 bytes，QKV slots65,814,528 bytes，FFN估计384MiB＋
256 bytes、QKV估计288MiB；总准入仍用原2GiB allowance。GPU原权重、encoder、
KV、scratch和框架副本另计，不能用“图小”隐去它们。

默认兼容性：657旧库与658新库的六类CPU emitter fingerprint完全一致，包含
base6144/5120、小shape及兼容配方；未运行LoRA模型。默认之外的reduction可能
改变PNG，不要求候选与descriptor逐位相同，但同配方timeline/反向重放必须相同。

## 真实完整block：更大的K没有获益

原BF16模型先完成首个sampler step，再捕获首decode的layer0真实输入/权重；
随后三arm每轮旋转，各2warm＋8measured。另做激活A/B/A和原layer0/1/0权重
切换，并确认原输入、原权重及旧输出不被后续调用改写。

| FFN policy | 原完整compiled GPU | staged GPU | Private每次重备W |
| --- | ---: | ---: | ---: |
| descriptor | 32.867126ms | 32.899875ms | 25.549229ms |
| k4096 | 32.866063ms | 32.851271ms | 26.191083ms |
| single | 32.874751ms | 32.866626ms | 27.753625ms |

普通GPU主分母跨窗稳定，stagedGPU输出exact。Private含真实生产6144通道FFN
权重staging、GPU attention前段、GPU complement、ANE、F32join、残差和完成。
K4096/single相对原FFN候选慢约2.5%/8.6%，不是新加速。三个arm的QKV都在GPU，
因此表格不能替代或乘进已有QKV＋FFN整请求结果。

每窗24 measured＋6warm完整block、另5次预检，总15次FFN workload预测，首轮
能力3次另列。捕获退出没有40步完成、VAE或PNG，不算完整图片请求。
十份完整BF16数组/窗、原checkpoint两层四份FFN payload独立重读/哈希通过；
single完整block relative-L2约0.00059047，不代表整图语义资格，也不是完整CPU
denseFFN oracle。

原件：`outputs/qwen-base512-ffn-projection-block-{descriptor,k4096,single}-gen-stock-v1-20261018`。
独立审计：`outputs/qwen-base512-ffn-projection-block-audit-v1-20261018/audit.json`。
这些组件窗只有stock，不能用Program计数冒充实际selector零热compile证明。

## 同库联合路线：完整文生图

同一658库、FFN6144＋QKV512、fixed-program、相同manifest，只有projection
环境变量不同。每进程seed73→74→73，各实际40步，完整wall含VAE/PNG；第一
请求单列，后两次取hot中位数。不是系统严格冷启动或统计资格。

| FFN policy | 首次完整请求 | 热请求中位数 | 相对descriptor耗时 |
| --- | ---: | ---: | ---: |
| descriptor | 34.150730s | 31.788099s | — |
| k4096 | 35.481701s | 32.692878s | ＋2.85% |
| single | 37.296410s | 35.139069s | ＋10.54% |

FFN累计暴露ANE等待约15.7→16.7→18.9s；preFFN约8.0s基本不变，postjoin约0.3s。
这支持退化发生在FFN路径，而非省下QKV时间；它们仍是host包络，不能据此
认定某个ANE硬件kernel变慢或将全部重叠span相加。
每请求FFN1312/QKV1248，1280个FFN blocks，未fallback/retry/headroom变化，
同一FFN owner Program实例数保持1。实际selector与独立整请求审计见下节。

原件：`outputs/qwen-base512-ffn-projection-request-{descriptor,k4096,single}-gen-stock-v1-20261018`。
前三窗进程树峰值physical footprint约17.99/17.57/17.01GiB，system swap-in/out0。
不是compiled program或单独ANE占用；系统/driver和其它应用没有独立归因。

## 编辑、selector、反向与质量

编辑三个stock窗口也都完成，各3次真实40步请求；全部对照保留标准1024参考和
runtime-prefix snapshot，首miss后hit，不能关闭强基线制造收益。

| FFN policy | 首次完整编辑 | 后两次完整编辑 | 热中位数 |
| --- | ---: | ---: | ---: |
| descriptor | 49.176367s | 38.579131 /39.794142s | 39.186637s |
| k4096 | 50.316537s | 42.009154 /39.543443s | 40.776298s |
| single | 54.481843s | 41.314564 /41.460771s | 41.387667s |

仍没有收益，且k4096单次波动较大。首请求FFN1440/QKV1248，prefix hit时两者
各1280；不把不同prefix状态的wall混成局部算子倍率。
原件`outputs/qwen-base512-ffn-projection-request-{descriptor,k4096,single}-edit-stock-v1-20261018`。

独立`projection_request_audit.py`已实际运行通过：只准许projection
环境变化，同库/manifest/reference、实际请求、40步和native计数一并校验；读取
完整PNG和source/memory证据。timeline与stock不混作性能分母；改变reduction
只查增量图像差，同配方重放要求RGBA exact。

两个独立timeline窗口：

| 场景/策略 | 实际evaluate（首/热/热） | 首compile/load | 两次热compile/load |
| --- | --- | --- | --- |
| 文生图single | 2563 /2560 /2560 | 0 /2 | 均0 /0 |
| 标准参考编辑k4096 | 2691 /2560 /2560 | 1 /2 | 均0 /0 |

每窗两个稳定workload model身份，FFN一个、QKV一个，含首轮3次FFN能力检查；
无drop/pending，Metal校准与真实完成检查通过。编辑首编译host span约0.803841s，
两load合计约0.024397s，不隐去初次成本。这是client选择器观察，不是不可见
driver内部编译或物理INT8证明，也不能从这两窗借给另两个stock-only场景。

文生图single、编辑k4096的timeline全部PNG与各自stock逐像素一致。反向恢复
descriptor后，文生图hot31.671184s、编辑38.550766s；全部PNG与原同库baseline
exact。新descriptor两任务的全部PNG还与冻结657 fixed-program控制exact，
旧时间没有进入本轮分母。编辑反向控制比最初baseline快约0.64s，说明小样本/
竞争负载波动不小，不能宣称测得精确永久回归幅度。

| 候选相对同库descriptor | 文生图RGB SSIM（seed73/74） | 编辑RGB SSIM（3请求范围） |
| --- | --- | --- |
| k4096 | 0.872886 /0.986550 | 0.997469–0.997970 |
| single | 0.951318 /0.989050 | 0.997120–0.997714 |

4张contact sheet均实际查看：k4096生成seed73把手高度/位置和壶嘴形状明显
变化；single也有把手/细节变化，不能因完整block误差很小就称整图无损。编辑
大结构基本一致，有局部纹理差异；强高光/环纹/饱和红色是双方共有的视觉表现，
不归因于某一个新策略，更不授予通用材质保持或语义质量资格。
增量alpha最大差分别为生成2/3、编辑均1（255尺度）。

相对冻结完整GPU（只作质量、不借旧秒数），生成descriptor SSIM0.921952/
0.977659，k40960.916376/0.981195，single0.940495/0.981600。编辑分别约
0.996895–0.997502、0.997285–0.997602、0.996617–0.996802；更接近某张GPU
并不说明语义普遍更好。候选既没有速度收益，也没有取得广泛质量资格，不采用。

4份最新独立审计：
`outputs/qwen-base512-ffn-projection-request-{k4096,single}-{gen,edit}-audit-v2-20261018/audit.json`。
它们重读全部原请求/PNG/selector/native/source/archive/memory，含反向及657
default exact控制；frozen GPU也先核对实际40步和相同prompt/seed/geometry。
初次gen v1审计仅因旧GPU报告缺少新status envelope被拒绝，日志保留；新增
严格识别旧`engine/constructor_seconds/runs`格式及负控后，v2全部通过。没有
修改任何模型窗口原件，两个成功edit v1也保留，不用其旧script hash冒充v2。

全部30请求进程树memory证据完整、退出0，system swap-in/out0；部分系统压缩/
解压计数增长，不归因于单一进程。竞争ComfyUI Python CPU活动仍有记录，没有
设备独占资格，selector零热compile并不使性能小样本自动具备统计资格。

## 与vpipe、变长和后续瓶颈的关系

只读本地vpipe checkout仍为`18ddb3b5bd9c74ab7f2b4408ee78efc6cecd48e7`且干净。
其Qwen consumer固定默认chunk2048，`begin`保留至少一个GPU row，1024 decode
实际ANE rows为0。原本机对照已经测得decode预测0，长参考prefill会预测，但
AUTO随后退回GPU/周期probe；见[完整profiling/vpipe报告](qwen-base512-runtime-ane-profile-2026-10-18.md)。
那是独立DiT forward，不是含encoder/VAE/PNG的40步图片，不跨边界计算倍率。

这里FFN采用1024-row图＋intermediate通道切分，QKV用512-row图＋token行切分，
保持完整attention在GPU。长参考仍用同一FFN图分块，长prefill QKV先走GPU；
QKV在1024目标decode使用ANE。此前fixed-program版本已实测
generation→单/双重复参考→generation，无热compile/load且恢复原PNG；本轮
不把657的观察重标为658或新的projection变长证据。

当前可迁移结论：灵活性来自固定计算ABI＋有界host计划，不需要按参考长度
重新编译。分工必须在stage/barrier前判断合法geometry，controller状态按
operator/phase/rows/数值配方隔离。两个program不是两颗ANE，也不能同时独占
一颗ANE；QKV join在attention前，FFN stage窗口在后，要测完整DAG。

更大FFN K没有解决暴露等待，暂不采用。后续应优先直接改善FFN的真实吞吐/
交接，以及长参考attention/KV；大code cache此前减少staging工作但未改善
关键路径，不再次仅凭命中率启用。runtime scalar headroom可作为固定ABI
恢复候选，但尚未实现/验证；当前仍是明确停用＋完整GPU重算，不能声称已经
在溢出时无代价恢复。

一个更具体的下一步是把gate/up和down分别消融。当前k4096/single的gate/up
图完全相同，只有down从4096＋2048变成整6144，而single明显更慢；这只定位到
down图策略，不是某个硬件kernel的归因。尚未测试“gate/up整4096＋down保留
2048”的混合策略，也不能从两处同时改变的descriptor/k4096对照排除它。
新策略仍须独立recipe、固定owner、当前强基线、全block/全请求和质量检查。

## 构建、工程与资格范围

- Native：`build/qwen-base-ffn-projection-v1-20261018/libturbocider.dylib`，SHA256
  `66e2c6c2b1c56fd549ee139e0e154557fa3a6843d6472f1fe56675d8f6f50488`。
- Runtime ID：`tc-runtime-build-v1-6f98a417df26868e0b3eaca1127458030a65f8620da4ed08756ed3e05ccf910e`。
- 658-input source archive：`outputs/qwen-base-ffn-projection-native-source-v1-20261018.tar.gz`，
  SHA256 `78d08024b3e770b02684b9c2b30f9dcde096857f8ad053c5b04a30815feabdf3`。
- Probe：`build/qwen-ffn-projection-probe-v1-20261018/libqwen_ffn_projection_probe.dylib`，
  SHA256 `eb9e71c4c9069f431b9d8f813747909d36d8c8c6934297d60cc76264a09891f1`。
- [工程入口](../../tools/profiling/qwen_base/README.md)：完整request stock/timeline、
  CPU独立auditor、block probe、default fingerprint及host负控分开。
- 当前tools/tests/fingerprint封存：`outputs/qwen-base-ffn-projection-tools-v1-20261018.tar.gz`，
  SHA256 `1af06a5fbf972b658824d1054d2a0aa1fb6fd8b14c0c04d08e7aa70ffe86a872`。
- 新库真实MLX lifecycle通过，24cycles/8variants＋joint组合/idle检查；原生
  projection/fixed CPU测试通过。全部设备结束后，最新64项profiler host测试及
  2项原生CPU policy/emitter测试通过，日志
  `outputs/qwen-base-ffn-projection-host-tests-v3-20261018.log`及
  `outputs/qwen-base-ffn-projection-native-policy-tests-v2-20261018.log`。
  旧61/63测试日志保留。知识库链接/metadata/no-data检查通过，368页、零错误/
  警告；这是文档检查，不是模型测试。未将这些检查/CPU审计与性能窗口并行。

本轮存在竞争推理CPU负载，没有设备独占、长期thermal、统计或广泛图像语义
资格，统一`qualification_passed=false`。W8/A8/Hadamard/FP16是近似表示；MIL
dequantize→MatMul不能证明物理INT8 MAC。工程有真实client selectors及Metal
CB时间窗口，但没有每kernel counter、ANE硬件时间或物理GPU/ANE重叠证明。
未下载/删除、未改模型master、未改references/产品默认或停止用户ComfyUI。
