# Runtime ANE + Qwen GPU Q/K norm-RoPE

2026-09-29，继续 runtime-weight GPU/ANE 目标。重读设计稿及 VPIPE 的
FFN row split、worker、staging 代码：优化对象应是整块/整请求关键路径，
不是单独 Core ML 或输出转换的最短时间。设计稿第19节的LoRA权重合并
不采用，仍服从本任务的base-only权重与runtime激活修正要求。

## 动机与范围

[输出恢复并行](runtime-ane-output-restore-parallel.md)在Qwen三图局部约
3.69×，整请求没有收益；原有GPU工作掩盖了多数输出恢复。下一候选组合
现有GPU Q/K RMSNorm、RoPE及输出布局融合，缩短GPU侧工作，不改ANE图、
token-row划分、调度门槛、SiLU、LoRA精度或修正顺序。

该Metal kernel已有[GPU历史实验](qwen21-gpu-fused-qk-norm-rope-2026-09-26.md)
和[冻结图历史实验](qwen21-hybrid-qk-fusion-2026-09-26.md)，但不能把那些
成绩当作runtime组合收益。规约与BF16舍入顺序不同，仍是显式近似；本轮
不宣称逐位等价，不改变默认路线，也不增加另一套kernel。

## 实现

- 原 `TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE=1` 门禁扩展至显式resident
  runtime路线，仍仅512²、BF16 GPU、允许近似且不能同时启用paired RoPE。
  不放开1024²、冻结W8A16 GPU suffix或streaming。
- pipeline将该kernel的实际选择标记统一放在各路线描述替换之后，避免
  runtime的route描述覆盖它；计划标签和实际结果都必须说明所选kernel。
- 整请求screen和共图switch新增 `--qwen-qk-norm-rope`，共用标准库配置/
  receipt校验。screen在所有所选路线使用同一设置，不能只加速混合一侧；
  默认清除继承的实验变量。启用但缺计划或实际标记时不能完成报告。
- 原kernel测试补齐2178/3202/4226长prefill行数，与原17/1024/1048一起
  检查有限性及relative L2，不放宽既有0.01阈值。契约覆盖base/LoRA、
  文生图/1–3图编辑及不支持组合；这些不替代真实整模型质量检查。

## 预声明验收

先完成native-only构建、Core ML微图、加速契约、Qwen专项、扩展Metal
kernel测试及完整回归，期间不benchmark。再做同c1792 v2图上的
base → 原始A → 合成半幅B → base状态隔离测试，要求每请求实际prediction、
adapter完整绑定、图不重载、返回base一致；合成B不是第二个训练LoRA。

性能为同一构建的四配置正反向对照：GPU关 → runtime关 → runtime开 →
GPU开 → 反向。512²、原始Viggle v0.2.1 r256、FP32低秩、6步、seed29、
相同三张有序ref512、auto chunks、profile关闭；每trial一冷八热。
统计全部热1–8及预声明较晚热6–8；每路线两trial，不删GPU探测请求，
不跨旧构建拼接分母，分别报告GPU优化、runtime优化及相同kernel下混合/GPU比。
所有trial独立100ms进程树采样，最大间隙门槛500ms；保留负结果与原始PNG。

候选native-only构建通过，库SHA为
`3986e035d805b559c0c3581afb46d772805428393df1d9a3d715d1532eab3fb4`。
首轮契约发现测试将ref512编辑专用标记误用于普通文生图；已修正测试的
环境设置，未放宽产品门禁。用系统旧Python直接跑工具测试曾因已有
`hashlib.file_digest`依赖失败；改用项目Python3.11后40项screen、17项
switch host测试通过，未为该环境问题修改哈希实现。
修正后 `make test-runtime-ane test-acceleration-contract test-qwen21` 与
完整 `make test` 均exit 0；4项host、9项Core ML/MLX微图、79项加速契约
（含同一4项host）、Qwen 38+7+30项及三个native/工作流入口通过。
扩展Metal kernel测试也exit 0：六种行数Q relative L2均0，K最坏
0.00325761，低于原0.01阈值。完整回归的缺夹具/专用构建/GPU opt-in
skips不算新增覆盖，已有macOS目标版本链接warning未导致失败。

真实三图共图切换已正常结束（exit 0），原始结果独立重验：累计prediction
**32/64/96/128/160**，adapter绑定 **0/0/227/227/0**，reference tokens=3072，
图加载计时不变，headroom=1，无失败/fallback/重试；返回base的PNG字节一致。
库/manifest/原始adapter/有序参考SHA重验通过。合成B改变了227个LoRA B
张量，只用于状态隔离；六步base也不是正常四十步base的质量资格。

已查看原始A的candidate PNG与此前关闭融合的8e共图PNG：两只壶、贴纸、
构图与暖光肉眼接近，蓝壶仍不透明，不能称所有编辑指令成功。PNG哈希
不同，不宣称逐像素等价；旧库图像只作视觉参考，不作本轮性能分母。
该切换没有独立内存采样，不能作内存资格。

四配置正反向对照现已完成，8trial/72请求，父子报告均complete，driver
正常exit 0。设备确认为Apple M4 Max、64GiB。测试、真实切换和正式
对照串行运行；每次模型工作前检查竞争推理，繁忙则等待，不终止他人进程。
正式对照期间不重建、不运行测试、不修改绑定的源码或工具；没有声明独占设备。

## 三图六步结果：小幅GPU优化，未扩大混合优势

另用 `outputs/runtime-ane/verify-qk-edit-factorial.py` 独立复核72条原始结果、
请求几何/顺序/精度、计划及实际融合标记、源文件/库/图/adapter/参考SHA，
以及8份独立内存流、报告和工具身份。校验exit 0，完整数值保留在
`outputs/runtime-ane/qk-edit-factorial-verified.json`。

下表均为同3986库，整请求墙钟包括VAE/PNG，排除冷请求，不排除GPU探测。
全部窗口每配置16热样本；较晚窗口每配置6样本，均在实验前声明。

| 配置 | 全部热1–8中位 s | 较晚热6–8中位 s | 全部热denoise中位 s |
| --- | ---: | ---: | ---: |
| GPU，融合关 | 12.906005 | 12.901263 | 12.219663 |
| runtime，融合关 | 12.855090 | 12.695722 | 11.661145 |
| GPU，融合开 | 12.666126 | 12.672843 | 11.971273 |
| runtime，融合开 | 12.689879 | 12.474506 | 11.471049 |

- GPU关/开：全部 **1.01894×**，较晚 **1.01802×**。
- runtime关/开：全部 **1.01302×**，较晚 **1.01773×**。
- 同样关闭融合，GPU/runtime：全部 **1.00396×**，较晚 **1.01619×**。
- 同样开启融合，GPU/runtime：全部 **0.99813×**，较晚 **1.01590×**。

融合对两条路线都带来小幅收益，但没有让runtime相对同样优化过的GPU
获得更大的优势，更不接近1.3×。全部热请求中开启融合的runtime约慢0.19%，
较晚窗口约1.6%的混合加速仍小；不能只选择后一个窗口推荐混合默认。

runtime四trial的prediction增量（含冷请求）依次为：

```text
关1: 32/32/0/0/31/31/31/31/31
开2: 32/32/0/0/31/31/31/31/31
开5: 32/32/0/0/32/32/32/32/32
关6: 32/32/0/0/31/31/31/31/31
```

热2/3各有32个完整GPU probes；热6–8每请求31/32个异步hybrid block。
全组无失败/fallback/重试，headroom=1。自动调度计数不完全相同，不将
整请求变化全部解读成Q/K kernel自身耗时减少，也不将较晚窗口称作全面收敛。

开启融合后，GPU/runtime的每请求 `wall-denoise` 中位为 **0.695741 /
1.233518 s**；runtime虽有更短denoise，额外请求级成本仍会抵消它。
源码保留完整runtime adapter SHA校验，而固定原始GPU LoRA的热请求沿用
既有缓存身份约定；没有为本候选删校验。这支持继续关注请求级开销，但
不是将上述差额精确归因于某个单独组件，亦不能直接相加不同窗口中位数。

8份独立采样重验：**9,553样本**，最大间隙 **110.019083 ms**。正向
runtime关/开各有 **65,536 bytes** 系统swap-in，其余为0；全部无新增
swap-out/compression。不能称零换页或把系统swap-in归因于ANE。
进程树峰值physical footprint（十进制GB）GPU关 **23.920–24.045**、
GPU开 **23.743–24.018**、runtime关 **22.654–22.655**、runtime开约
**22.627**；采样含加载/冷/热/退出，不等于ANE独占或完整driver/wired资格。

已查看反向四配置末次热PNG：主体、壶嘴/把手、贴纸与暖光构图肉眼接近，
局部纹理/轮廓有轻微差异，无新增明显崩坏；四者的蓝壶均不透明。只覆盖
该prompt/seed/adapter，不扩展为全部编辑意图、训练LoRA或base质量验收。

**决策：保留为显式可选组合，不改默认、不新增“最快混合”预设。** 这是
小幅GPU侧优化，不是更快的ANE计算；base与冻结图结果继续单独验收。

## 证据入口

```text
outputs/runtime-ane/qk-norm-before-bin/
outputs/runtime-ane/qk-norm-build.log
outputs/runtime-ane/qk-norm-tests.log
outputs/runtime-ane/qk-norm-tests-rerun.log
outputs/runtime-ane/qk-norm-kernel-test.log
outputs/runtime-ane/qk-norm-full-test.log
outputs/runtime-ane/qwen-edit3-qk-norm-factorial.py
outputs/runtime-ane/qwen-edit3-qk-norm-shared-switch/
outputs/runtime-ane/qwen-edit3-qk-norm-factorial/
outputs/runtime-ane/qk-norm-factorial-driver.log
```

before库保留1080构建；以上四配置均使用3986构建，不将旧成绩重标。
Z计算路径未改；本轮Qwen候选不外推为Z、1024²或其他LoRA的收益。

## Base同构建资格（按预声明完成）

编辑campaign结束并复核后，继续同3986库的base对照，不更改权重、图、
精度或调度。驱动 `outputs/runtime-ane/qk-norm-base-matched.py` 已准备：

- Z 512²/8步，c352/K1024/N512 v2，无Q/K融合；GPU → runtime → frozen
  → 反向，共6trial/18请求，验证未修改的Z路径并更新同构建三路分母。
- Qwen 512²/40步，c320/K1024/N512 v2；GPU关 → runtime关 → frozen关
  → frozen开 → runtime开 → GPU开 → 反向，共12trial/36请求。
  c1792不用于512² base，因为该chunk大于普通decode行数。
- 原狐狸雪景prompt/seed42、BF16、无LoRA、resident、auto/profile-off；
  每trial一冷两热，池化全部四热样本，不挑最低值。全部开启100ms独立
  采样/max-gap500ms；先Z后Qwen串行，不与编辑campaign、编译或测试重叠。

编辑campaign完全退出、独立重验及四配置PNG查看之后，已启动上述串行
队列，先Z后Qwen。Z的6trial/18请求和Qwen的12trial/36请求均已完成，
队列正常exit 0；独立验证原始请求、计时、调用、源文件/运行库/manifest
身份及全部内存流通过。
证据目录为 `outputs/runtime-ane/z-qk-build-base-matched/` 与
`outputs/runtime-ane/qwen-qk-base-matched/`，driver日志为同前缀对应的
`z-qk-build-base-driver.log`、`qwen-qk-base-driver.log`。
下面分别报告两模型的实测；编辑收益不替代base、冻结图或Z的资格。

### Z base：完成，原有runtime收益保留

同3986库，每路线全部四热请求中位：GPU **6.993967 s**、runtime
**6.343233 s**、冻结图 **5.339539 s**。GPU/runtime **1.10259×**，
GPU/冻结图 **1.30984×**；冻结图仍最快。未修改Z计算路径，这不是Q/K
候选对Z带来的增益，也不将与旧库的小差异全部归因于代码。

runtime两个trial累计调用均 **192/440/688**，冷请求3次headroom重试，
热请求无重试、错误或fallback。独立采样重验共 **1,324样本**，最大间隙
**110.017541 ms**；无新增系统swap-in/out/compression。进程树峰值
footprint（GB）GPU **22.946–22.947**、runtime **23.041–23.042**、
冻结图约 **23.070–23.071**；不是完整driver/wired或物理ANE驻留资格。

独立复核脚本 `outputs/runtime-ane/verify-qk-base-matched.py`，结果
`outputs/runtime-ane/z-qk-base-verified.json`。当时Qwen正在正式计时，
没有同时渲染Z图片或另跑模型；Z的图像身份及视觉边界另行记录。
只读SHA比较确认反向GPU/runtime/frozen三张末次热PNG分别与8d90
[匹配实验](runtime-ane-matched-memory.md)对应图片字节一致。可沿用此前
该prompt/seed的视觉参考，但不称为本轮新目检或更广提示词质量验收。

### Qwen base：完成，同样优化GPU后runtime仍有1.170×收益

同3986库、512²/40步、BF16 base、狐狸prompt/seed42、c320/K1024/N512 v2，
六配置正反向对照，每trial一冷两热，每配置池化全部四热样本。整请求时间
包含VAE/PNG，不含模型加载/冷请求，不剔除自动调度的GPU探测。

| 路线 | Q/K融合关，s | Q/K融合开，s | 关/开 |
| --- | ---: | ---: | ---: |
| 普通GPU | 42.677987 | 41.742390 | 1.02241× |
| runtime v2 | 36.685170 | 35.675016 | 1.02832× |
| 冻结base图 | 30.224421 | 29.067995 | 1.03978× |

同样关闭融合，GPU/runtime为 **1.16336×**，GPU/冻结图为 **1.41204×**；
同样开启融合，分别为 **1.17007× / 1.43603×**。runtime自身整请求耗时
减少约2.75%，但同样优化GPU后混合优势只小幅增加；不能拿关融合GPU除以
开融合runtime并归因于ANE。本轮没有新ANE kernel、权重或图，冻结图仍最快。
保持Q/K显式optional，不把单prompt/seed的结果自动提升为全部请求默认。

四个runtime trial累计prediction全部为 **1184/2432/3648**，每请求增量
1184/1248/1216；无headroom重试、失败或fallback。原始结果及六配置选择
标记通过独立复核，父子报告complete、源码/库/图身份不变。

十二份内存证据重验共 **13,498样本**，最大间隙 **110.976292 ms**。
反向runtime开融合trial7有 **65,536 bytes** 系统swap-in，其余trial为0；
全组无新增swap-out/compression，不能称零换页或将系统计数归因到ANE。
加载/冷/热/退出范围的进程树峰值physical footprint（十进制GB）：

| 路线 | Q/K关 | Q/K开 |
| --- | ---: | ---: |
| GPU | 19.349–19.722 | 19.346–19.419 |
| runtime | 19.506–19.737 | 18.970–19.997 |
| 冻结图 | 24.646–25.030 | 24.839–25.084 |

不是完整driver/wired或物理ANE驻留资格；不能根据这些波动断言融合必然
节省内存。benchmark结束后再查看反向六配置末次热PNG：狐狸、松枝雪景、
构图与色调接近，尾巴/前爪/毛发细节变化，无新增明显崩坏。不宣称逐像素
一致，也不外推到其他prompt/seed或LoRA编辑。

独立复核结果：`outputs/runtime-ane/qwen-qk-base-verified.json`；复核脚本
同上。原始PNG位于 `qwen-qk-base-matched/{6…11}-*-qk*/0-*-2.png`。
计划检查和新增host测试均在队列退出后执行，不污染本次正式计时。

## 接续：runtime设备计划观测

参照设计稿的capability/observability要求，扩展现有
`tools/validation/qwen21_ane_placement.py`，不新增产品依赖或推理开关。
保留冻结图 `--blocks` 接口，新增runtime单个共图manifest；延迟SDK导入，
遍历函数/嵌套block，分别统计计算、常量和未分配节点。前后绑定manifest、
编译树及工具SHA；runtime核对完整files receipt，拒绝未覆盖文件、变化、
symlink、路径越界和覆盖已有输出。`ios18.constexpr_*`归入常量，避免把
冻结权重解码节点当成普通计算节点。运行命令见[验证指南](runtime-ane-validation.md#离线设备计划检查不是物理执行证明)。

benchmark完全退出、预检未发现竞争推理后，串行检查当前使用的三个runtime
图以及Qwen/Z冻结图的block0，全部正常完成、前后身份不变：

| 图 | NE偏好的非constant节点 | 其中matmul/conv | 其他/未分配计算节点 |
| --- | ---: | ---: | ---: |
| Qwen v2 c320 | 836 | matmul 288 | 0/0 |
| Qwen v2 c1792 | 836 | matmul 288 | 0/0 |
| Z v2 c352 | 698 | matmul 240 | 0/0 |
| Qwen冻结图block0 | 12 | conv 2 | 0/0 |
| Z冻结图block0 | 23 | conv 2 | 0/0 |

这些是 **MLComputePlan在CPU_AND_NE策略下的偏好**，不是实际时间线、
物理ANE驻留或INT8执行证明。算子数量不能换算为耗时/FLOPs/利用率；冻结
与runtime的图范围和切分不同，不能把288/2当作性能比。只检查冻结block0，
不冒称全32层计划都已检查。当前没有发现计划层面的CPU回退，但真实执行
仍为unknown；短LoRA的请求级校验、激活修正、同步以及编译GPU block边界
仍是后续优化对象，不能凭计划将它们精确分摊为性能差额。

证据在 `outputs/runtime-ane/` 下：
`qwen-c320-v2-planned-placement.json`、`qwen-c1792-v2-planned-placement.json`、
`z-c352-v2-planned-placement.json`、`qwen-frozen-block0-planned-placement.json`、
`z-frozen-block0-planned-placement.json`。更早无工具SHA的
`qwen-c320-v2-device-plan.json`保留为初查，不作为最终报告。

随后`make test-acceleration-contract`正常exit 0：89项通过，含新增10项
计划检查host测试；另行8项布局、7项独立性检查通过，无skip。日志
`outputs/runtime-ane/placement-contracts.log`。未重建产品库、未更改绑定的
原生源码/图，不重跑Core ML微图或完整`make test`，此前完整回归仍归原轮。

本机`xcrun --find xctrace`返回不可用，开发目录为CommandLineTools，
`/Applications`下未找到Xcode.app。因此本轮没有取得Instruments硬件执行
时间线，不以计划检查或host等待计数替代真实GPU/ANE重叠资格。没有安装
大型开发工具、更改系统developer目录或要求终止其他工作；物理trace与
完整driver/wired内存仍是未完成项，不代表整个runtime优化目标被阻塞。

五份最终计划报告另行复核工具/manifest/图SHA及分类计数一致；六份维护
文档125个本地链接存在，无机器用户目录路径，`git diff --check`通过。
库SHA保持3986，暂存diff保持694224eb…，249项既有暂存移除及本地文件
全部保留。所有本轮对照、计划检查与测试均已结束；未stage/commit，未删除
模型/缓存/图片或原始依据。上述进展不代表研究目标已全部完成。
