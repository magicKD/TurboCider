# Qwen base512：text-prefix复用的速度信号与首步画质反例

后续已完成[同库三arm首步QKV修正](qwen-base512-text-prefix-qkv-step0-2026-10-19.md)：
命中时首步保留GPU、后39步仍Private，在已测茶壶/材质上恢复完整RGBA exact并
保留小幅收益。本页旧版反例与数字保留，不把旧`ane`模式改写成已通过。

2026-10-19更新，沿用`20261018`实验目录标签。M4 Max/64GiB，仅原BF16 base、
512²、实际40步生成/编辑，无LoRA/student/Z Image/temporal DiT cache。
继续[GPU complement筛选](qwen-base512-ffn-down-tiles-2026-10-18.md)。

本轮已实现独立default-off的runtime text-prefix开关，并完成完整请求、反向、
重复、变prompt和编辑回归。热生成有约0.2–0.4s的速度信号，**但命中后的seed73
有明显把手几何退化，不授予采用资格，不是无损缓存优化**。App和默认设置未改。
先保留可复核反例，下一步隔离命中时的首步QKV算术，不能只展示更快的秒数。

## 实现与固定图合同

新env：`TURBOCIDER_QWEN21_RUNTIME_TEXT_PREFIX_SNAPSHOT=1`。与现有编辑开关
`TURBOCIDER_QWEN21_RUNTIME_PREFIX_SNAPSHOT`分开：旧开关仍edit-only，新开关
只作用于generation，不静默扩张旧语义。公共module先检查请求，pipeline再检查
实际Private/fixed-program/健康W8状态及冲突路线。

限resident512 base40、显式runtime GPU/ANE、允许近似、无参考/LoRA/PE、GPU
encoder、无额外编码/phase/attention/temporal实验。默认不启用；新开关不绕过
Private授权、manifest或fixed-program，失败后不得发布半成品。

沿用一个紧凑K/V bank，而不是保留整个Transformer：

- import检查实际conditioning Tensor identity、geometry、完整叶值与独立recipe
  namespace；新text命名空间与编辑、普通GPU及其它算术分开。
- runtime source/configuration与原checkpoint generation进入身份；同shape不等于
  同输入或同权重。提示词更换时不能用旧text K/V。
- 先完成所有目标consumer并drain/移除runtime回调，才导出独立连续K/V叶值；
  只在PNG成功后发布。原有取消/异常清理、有限性与完整GPU失败重算不削减。
- 一份FFN1024图、一份QKV512图仍跨32层/40步使用；没有增加逐层模型、hot
  shape重编译、code cache或weight预取。snapshot不是ANE Program缓存。

本地源码的prefix modulation取零timestep行，文本segment causal attention只
读自身/此前tokens，FFN逐row独立。真实MLX小型fixture对不同noise和sigma重算
后，全部text K/V逐位一致；另查conditioning、recipe、callback、预算、源对象
销毁与compact leaf所有权。它支持复用代数/生命周期，不替代真实W8整图画质。

## 完整40步请求结果

同一新库、同prompt/seed73→74→73、FFN6144＋QKV512、互补激活/F32 join、
cache/prefetch off，stock无observer。完整wall包括encoder、40步、VAE和PNG。
首请求单列，不称严格系统cold；不清OS/driver缓存，不借旧GPU秒数作分母。

| stock配方 | 首次完整请求 | 后两次完整请求中位数 |
| --- | ---: | ---: |
| text-prefix off | 31.125904s | 29.896522s |
| text-prefix on | 31.181660s | 29.602003s |
| 反向off | 31.333891s | 29.824953s |
| 独立重复on | 32.540537s | 29.469041s |

首对约少0.294519s（0.985%）；与反向/重复一起看是小幅、非独占设备pilot，
不是精确永久收益。冷请求没有提速，不能把重复prompt收益写成所有fresh生成
收益。两次on的四个hot样本都保留，不挑最快单张；无thermal/统计资格。

真实工作量解释比总evaluate数更重要：

| 请求状态 | FFN预测 | QKV预测 | 实际采样步骤 |
| --- | ---: | ---: | --- |
| off或cold miss | 1312 | 1248 | prefill1＋target39 |
| text-prefix hit | 1280 | 1280 | target40 |

命中省去32次FFN padded chunk，却增加32次QKV预测；总evaluate同为2560，
不能因总数未变而认定没省工作，也不能把两种模型的调用视为等成本。所有
实际1280个FFN blocks、40步都完成，无fallback/retry/headroom变化。

## 重要反例：前缀独立不代表当前组合无损

新开关命中后，首步走target-only decode：原本1048-row首步QKV在GPU，现在
1024 target rows可以选择QKV W8。首步FFN/GPU shape也改变。因此“prefix本身
独立”不证明整个目标轨迹与off逐位相同，尤其已有多个近似组件组合时。

独立审计与实际看图：

| 样本 | on相对同库off RGB SSIM | alpha最大差 | 实际观察 |
| --- | ---: | ---: | --- |
| 首次seed73、miss | 1.000000 | 0 | 完整RGBA exact |
| seed74、hit | 0.992966 | 1 | 大结构相似，纹理/高光有变化 |
| 恢复seed73、hit | 0.892467 | 2 | 闭环把手变成明显开口/钩状，壶嘴也改变 |

comparison sheet已实际查看；重复on/timeline逐像素恢复相同结果，反向off恢复
原图，故不是只凭一次随机展示。此处将seed73记录为几何退化反例，不当作
另一种风格就授予通过。也不能唯一归因W8 QKV：需要首步GPU counterfactual
再与形状/reduction因素隔离。目前不叫无损缓存、不推广为产品速度预设。

人工记录及绑定PNG摘要：
`outputs/qwen-base512-text-prefix-audit-v1-20261018/visual-review.json`；
整张sheet为同目录`comparison.png`。

## 提示词切换、实际模型身份与编辑回归

同进程timeline矩阵为A73→A74→B73→A73，真实prompt、PNG和状态均检查：

| 请求 | 文本tokens | bank逻辑大小 | cache状态 | FFN/QKV预测 |
| --- | ---: | ---: | --- | --- |
| A73 | 24 | 12MiB | miss、发布 | 1312/1248 |
| A74 | 24 | 12MiB | hit | 1280/1280 |
| B73 | 46 | 23MiB | miss、替换 | 1312/1248 |
| 恢复A73 | 24 | 12MiB | miss、替换 | 1312/1248 |

B为红木方块/钢球/玻璃瓶，已实际看图，三个物体按左中右出现，不是stale茶壶。
首尾A的实际request和完整RGBA exact；B不等于任何A输出。算法recipe SHA保持
一致是预期，conditioning identity另检；不以recipe相同替代输入校验。

两份timeline（repeat和prompt矩阵）各只有两个稳定model identity，actual
evaluate首请求2563（含3次能力检查）、后续每请求2560，按每个模型的native
差分核对。首次compile0/load2，hot compile/load均0；缓存了已有MIL artifact
不等于冷启动无装载成本，也不覆盖不可见driver内部编译或物理INT8/重叠。
矩阵任务/prefix状态不同，不pool hot median；repeat诊断29.684728s不用作stock分母。

新库edit回归保留标准1024参考及旧runtime edit snapshot，首45.282411s、hot
36.933099s。三张RGBA与冻结旧库编辑control exact；生成off三张也与冻结旧库
exact。旧control只查request/像素，不读取其时间当新库分母。不能将这些检查
扩大成所有未执行实验组合、脸/文字/材质或双参考的语义资格。

## 审计、构建与范围

- 最新独立审计：`outputs/qwen-base512-text-prefix-audit-v1-20261018/audit.json`。
  重读实际请求、原件/source/native/PNG/memory、chunk/step和每模型selector；
  off/on只准许新env变化，miss exact、reverse/timeline/repeat exact及prompt恢复
  分别检查。`qualification_passed=false`不因数值/来源审计成功而变成画质通过。
- 新库：`build/qwen-base-text-prefix-v1-20261018/libturbocider.dylib`，SHA256
  `3e8f94de9fa733a1f1cf558c51e7e24a16d8f557079062629e4cff85b577d4ed`。
- Runtime ID：`tc-runtime-build-v1-ab4a12cda02167d5e8544c36a937f2f9ab243a9555b5d32c5341971f4c49b48d`。
- 659-input source：`outputs/qwen-base-text-prefix-native-source-v1-20261018.tar.gz`，
  SHA256 `8a5a4795a87ba84d1195ac3f02081d6adbdb919de095fb12823ace728dce50ae`。
- 最终tools/tests/docs：`outputs/qwen-base-text-prefix-tools-v1-20261019.tar.gz`，
  摘要独立保存于同名`.sha256.json`。所有旧库/source/archive/原件保留。
- 95项profiler host测试通过：`outputs/qwen-base-text-prefix-host-tests-v2-20261018.log`。
  CPU native request/health options1项通过、真实新库MLX lifecycle1项通过，分别
  为`outputs/qwen-base-text-prefix-native-policy-tests-v1-20261018.log`与
  `outputs/qwen-base-text-prefix-lifecycle-v1-20261018.log`；不是97项硬件资格测试。

七个窗口共22次完整请求/880个实际steps全部exit0；自有构建/测试/CPU图像审计/
看图均在性能窗口外，所有性能窗口串行。whole-process physical footprint峰值
约17.69–28.84GiB；不能把长prompt矩阵的全部增长归给12/23MiB prefix bank。
系统swap-in/out均0，编辑窗口有系统compression；不是单模型独占归因。

检测到其它ComfyUI CPU活动，全部窗口保留竞争负载标记；无GPU独占、thermal/
统计或普遍画质资格。没有停止用户服务、下载、删除、改App/默认/原权重/参考。

下一步先保持命中时首步QKV的GPU算术做独立对照，注意plan改变必须与实际
compiled-block路径、GPU credit及recipe namespace一致；不能简单改标签或
减少成功调用要求。需要新native/source archive、同库off/on完整40步、多seed
及失效/取消检查。当前目标ACTIVE，本版text-prefix只保留为显式研究路径。
