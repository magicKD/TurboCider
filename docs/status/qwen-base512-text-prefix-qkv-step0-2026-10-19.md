# Qwen base512：保留首步GPU QKV，修复text-prefix反例并保留收益

2026-10-19，M4 Max/64GiB；原BF16 base、512²、实际40步，无LoRA/student/Z Image，
无temporal DiT cache。继续[text-prefix速度信号与画质反例](qwen-base512-text-prefix-2026-10-19.md)。

本轮在同库三arm对照中定位并修复了此前的缓存命中反例：**首步QKV保持GPU，
后39步保持原Private QKV分工**。茶壶/材质两提示词的6张同seed完整RGBA与
prefix-off全部exact，热请求约少0.28–0.37s（约0.9–1.2%）。这是显式研究选项，
不改App/默认，也不把有限样本称作普遍无损或正式性能资格。

## 为什么cache本身独立，结果却曾退化

原序列含text24＋target1024 rows，首步1048-row QKV不满足Private固定1024
目标rows准入，实际走GPU。缓存命中后首步只剩1024 target rows，旧计划因此
新增了一整步W8 QKV，改变了数值路径；不是只省掉独立text计算。

同库O/A/G三arm，实际请求、checkpoint、步数、FFN6144/QKV512、互补激活、
F32 join、cache/prefetch off均固定：

| arm | text snapshot | 命中后的首步QKV | 其余39步 |
| --- | --- | --- | --- |
| O | off | 原prefill GPU | 原Private分工 |
| A | on | 旧Private分工 | 原Private分工 |
| G | on | GPU | 原Private分工 |

A→G只改变明确的首步policy env；冷miss三arm完整RGBA exact。A在新库复现
旧库的开口/钩状把手，G恢复为O的闭环把手，且全部RGBA exact。这组受控干预
将该recipe中的反例定位到首步新增W8 QKV路径，而不是以猜测给cache数学归因。
不据此宣称所有shape、模型或硬件的图像漂移都只有这一原因。

## 实现：在stage之前路由，不新建ANE图

新env：`TURBOCIDER_QWEN21_RUNTIME_TEXT_PREFIX_QKV_STEP0=ane|gpu`，默认`ane`
保留旧研究行为。正确复用方案需同时显式开启text snapshot并选`gpu`；旧ANE
首步模式只保留为负控，不当作推荐速度预设。

`gpu`只准入generation text-prefix＋joint Private QKV，编辑和普通默认路径
不变。配置在请求开始捕获，request-owned sampler step在每次forward前更新；
只在text-prefix hit且step0时禁止Private，其余步骤不改。独立recipe namespace
防止把不同策略的bank当作同一算术。

`RuntimePrivateQkv::plan(rows, permit_private)`在stage/enqueue之前返回GPU并
记GPU credit；旧`plan(rows)`签名仍保留。不会先stage再以零prediction冒充加速，
也不卸载/重建后续39步要用的固定模型。未改变class布局、W8/F32计算、surface
lease、有限性、取消/failure drain、sticky fixed-program或完整GPU重算合同。

实际Transformer已有不同compiled-block缓存槽区分split-FFN/GPU-QKV与
split-FFN/external-QKV，同shape切换不会借用错误route。新增真实MLX fixture
验证GPU首步→外部QKV后续的callback数量、完整输出与最后所有权清理；它是
GPU fixture，不能替代实际Private/full40结果。

## 完整请求性能：同库、stock强对照

每进程seed73→74→73，首请求单列，后两次完整wall取中位数，含encoder、
40步、VAE、PNG；不清OS/driver cache，不称严格系统cold。原完整GPU不是
本次分母；O本来就是已优化的joint runtime强对照。

| 配方 | 首次完整请求 | 后两次中位数 |
| --- | ---: | ---: |
| O，text-prefix off | 31.000608s | 29.829341s |
| A，旧ANE首步（有画质反例） | 31.213930s | 29.543443s |
| G，GPU首步 | 31.254395s | 29.541948s |
| O，反向控制 | 31.273709s | 29.809634s |
| G，独立重复 | 31.320322s | 29.533957s |
| 材质O，46-token prompt | 31.511006s | 30.025859s |
| 材质G，46-token prompt | 31.385919s | 29.653640s |

茶壶首对少0.287393s（0.963%），反向/重复仍少约0.275677s（0.925%）；材质
配对少0.372219s（1.240%）。它没有靠保留旧数值退化获得额外速度。冷miss没有
缓存工作可省，首次时间不推导加速；fresh prompt、一次性请求不是本优化的
收益场景。只有少量非独占样本，无thermal/统计资格。

## 工作量、热编译与prompt长度

| 状态 | FFN预测/请求 | QKV预测/请求 | GPU QKV blocks/请求 |
| --- | ---: | ---: | ---: |
| O或cold miss | 1312 | 1248 | 32 |
| A hit | 1280 | 1280 | 0 |
| G hit | 1280 | 1248 | 32 |

G保留40个真实target步骤与1280 FFN blocks，只去掉首步多余的32次FFN padded
chunk；不是降步数、复用target/残差或关闭必要精度检查。不同模型调用不等成本，
host阶段span有重叠，不能把预测/等待总和当硬件关键路径。

两份独立timeline均只有两稳定model identity（FFN1024、QKV512），cold
evaluate2563含3次能力检查；G hot2528，按每模型/每请求与native差分核对。
first compile0/load2，hot compile/load0；这是观察到的Private client API，
不承诺不可见driver内部活动、真实INT8 MAC或物理GPU/ANE重叠。

同进程A73 miss→A74 hit→B73 miss→A73 miss，text24→46→24，bank12/23MiB。
真实prompt、K/V预算/recipe/conditioning identity、完整输出与失效恢复通过；
首尾A RGBA exact，B不是stale茶壶。矩阵不pool hot median。材质独立repeat
另外实际命中46-token bank，覆盖非8对齐text长度，不能从A24单例外推。

## 图像与原行为回归

`--require-gpu-exact`要求两个提示词的所有6个GPU/off对照完整RGBA exact，
其中4个是命中后的真实对照，另外2个是cold miss。两提示都含seed73/74/73，
不能把重复seed73计成第三个独立seed。独立重复、timeline及prompt恢复也
恢复相同的全部PNG，不只比较采样几个点或RGB SSIM。

三列茶壶sheet与材质sheet均已实际查看。G没有新增把手/壶嘴缺口；材质的红
方块/钢球/透明瓶、反射与背景在同seed下完整一致，换seed后按预期变化。
此exact是相对现有joint W8基线，不等于相对未量化GPU无损，更不是文字、人物
身份、材质编辑或所有提示词的普遍资格。

旧edit路径保持标准1024参考与原runtime edit snapshot，完整三请求hot
37.092043s；全部RGBA与旧库edit trial exact。O及旧A也分别恢复旧库的全部
PNG，保留旧反例而不重标历史性能。旧trial只作request/像素control，秒数未借用。

完整审计：`outputs/qwen-base512-text-qkv-step0-full-audit-v1-20261019/audit.json`。
三列图为同目录`comparison.png`，材质为`additional-pair-0.png`；人工检查
及绑定摘要为`visual-review.json`。pilot3窗审计原件保留。

## vpipe与灵活runtime的边界

只读复核本地vpipe仍为`18ddb3b5bd9c74ab7f2b4408ee78efc6cecd48e7`，worktree干净。
本轮未重跑其forward，已有本机开/关对照见[固定runtime/vpipe报告](qwen-base512-runtime-ane-profile-2026-10-18.md)。
不把vpipe默认2048 chunk在1024 decode零prediction说成ANE加速，也不将forward
与本轮encoder/VAE/PNG完整请求算倍率。

当前灵活性来自host的合法phase计划、固定chunk和动态权重。图不随prompt/
block变；某phase不适合ANE时，先选普通GPU，避免无用stage/fence。还必须
保持该phase的数值路线，不能因cache减少rows就无意引入新的近似。

## 构建、测试与范围

- native：`build/qwen-base-text-prefix-qkv-step0-v1-20261019/libturbocider.dylib`，SHA256
  `307455f6dc329bdd526f724720e30b7f653d9caff0628e99b152e4fa7b2211e9`。
- runtime ID：`tc-runtime-build-v1-e075484ab238362b5cc4fe5f02b94752c806026165613c6c8cd94fa4f7c18e39`。
- 659-input source：`outputs/qwen-base-text-prefix-qkv-step0-native-source-v1-20261019.tar.gz`，
  SHA256 `06b7f9430421308616238d1000c30b6b8b454ad582664def4ab8c4be7b96524c`。
- pilot tools：`outputs/qwen-base-text-qkv-step0-tools-pilot-v1-20261019.tar.gz`，SHA256
  `dd7fd5b386247bd9c80fbfbedbe2336b00473a45f4880b74fedacca54a791290`。
- 最终tools/tests/docs：`outputs/qwen-base-text-qkv-step0-tools-v1-20261019.tar.gz`，摘要
  独立保存于同名`.sha256.json`；所有旧库/source/失败与成功原件保留。
- 最终97项profiler host测试通过，日志`outputs/qwen-base-text-prefix-qkv-step0-host-tests-v3-20261019.log`；
  native request/policy1项及真实MLX lifecycle1项分别见
  `outputs/qwen-base-text-prefix-qkv-step0-native-policy-tests-v2-20261019.log`和
  `outputs/qwen-base-text-prefix-qkv-step0-lifecycle-v1-20261019.log`。
  不是99项硬件资格测试，真实full40/selectors/PNG证据另列。

十窗共31次完整请求/1240个实际steps全部exit0；无fallback/retry/headroom变化，
所有性能模型窗口串行，自有构建/测试/CPU数组审计/看图均在窗口外。
whole-process physical footprint峰值约17.06–28.84GiB，系统swap/compression
增量均0；不是小ANE图的独占内存，也不能将矩阵峰值全归给12/23MiB K/V。

全部窗口记录竞争CPU负载，无GPU独占或统计/thermal资格；统一qualification=false。
没有停止ComfyUI、下载/删除、改App/default/权重/参考。新路径只保留为显式研究
选项；`ane`首步负控仍有已知退化，使用正确方案必须显式选择`gpu`。

整体优化目标仍ACTIVE。后续继续处理主FFN/交接关键路径及更广的质量/长度
检查，不把这约1%的重复生成收益宣称为全部瓶颈已解决或fresh/edit普遍提速。
