# Qwen Base512：固定runtime ANE与保守残差缓存组合

2026-10-19，M4 Max/64GiB、macOS26.6.2。原BF16 Base、512²、40次实际Euler更新，
无LoRA/student/Turbo。整体优化目标active，App/default未改，未commit。

## 结果：编辑出现明确速度信号，生成累计质量预算未过

这是**独立近似画质档**：复用若干步的中间24层残差，减少实际block计算；
不是把原1280块全部算完的kernel提速，不替代原完整计算基线。

同一新库、原encoder/DiT/VAE、相同输入/seed73→74→73，各进程首请求单列，
后两次完整请求含PNG导出取中位数。编辑为标准1024参考，热请求真正prefix hit。

| 场景 | GPU无缓存 | GPU conservative | 原runtime ANE | runtime ANE＋conservative |
| --- | ---: | ---: | ---: | ---: |
| 生成 | 42.594152s | 30.991988s | 28.643842s | 20.942690s |
| 单参考编辑 | 49.946543s | 36.183146s | 35.862993s | 26.509963s |

编辑原runtime首组on→off，另起进程反序off→on得到**35.796589→26.158729s**，
方向保留；原始用时少约9.35–9.64s，不挑最快一次、不乘历史倍率。
同cache策略GPU36.183146s与runtime26.509963s另列，避免把少算层的全部收益都
归给ANE。GPU与W8的proxy输入不同，即使cache策略相同也不保证跳层mask相同。

十个stock首请求分别记录：生成runtime off/on约30.255/22.429s，编辑约
44.308/34.890s；反序编辑约44.093/35.259s。这不是清OS缓存的严格冷启动，
不把首请求/encoder/初始化成本悄悄扣掉。timeline和四请求恢复窗不作速度分母。

十四窗均检出外部推理CPU负载，正式speed为null，qualification仍false。
以上为有限本机原始观察，不是GPU独占、统计/thermal或全场景性能认证。

## 质量：增量通过不等于累计通过

调试前在工具中固定暂定预算：RGB SSIM≥0.98、MAE≤3/255、alpha最大差≤4，
三项同时满足。它不是用户确认的感知预算，也不保证文字、身份、材质和编辑指令。

| 完整三张配对 | RGB SSIM | RGB MAE /255 | alpha最大差 | 暂定数值预算 |
| --- | --- | --- | ---: | --- |
| 编辑：runtime缓存对原runtime | 0.997422–0.997580 | 0.580–0.669 | 1 | 通过 |
| 编辑：runtime缓存对完整GPU | 0.996141–0.996437 | 0.889–0.899 | 1 | 通过 |
| 生成：runtime缓存对原runtime | 0.988907–0.992082 | 0.960–1.459 | 1 | 通过 |
| 生成：runtime缓存对完整GPU | seed73 0.903902、seed74 0.974196 | 8.733 /2.242 | 2 /1 | **未过，不推荐为质量合格档** |

原runtime生成对完整GPU已为0.902176/0.977323，说明累计问题不能用缓存增量的
较高SSIM掩盖；也不能从个别指标略改善推导缓存普遍改善画质。没有调宽预算。
编辑虽平均误差小，仍存在稀疏较大RGB差异，不称逐像素无损。

实际查看同seed完整GPU编辑与新缓存编辑，当前红茶壶样例的主体/构图相近；
原GPU本身已有的高对比、材质与指令保真问题并未因此解决。仅一张参考、一个
编辑指令、两个seed，不授予普通prompt、文字、身份、透明度或多参考语义资格。
推荐保留**显式可选编辑候选**继续holdout，而不是更改产品默认。

## 机制与安全交接

新增默认OFF的`TURBOCIDER_QWEN21_RUNTIME_CONSERVATIVE_CACHE=1`。
只允许原Base40/resident512、无LoRA/其他编码、Private fixed C6144/F32 join、
sigmoid-complement、QKV one-hop；拒绝GPU-dual/shared-compiled/family组合。
JSON仍需`qwen21_dit_cache: conservative`；显式Off保持原路由和namespace。
原GPU preset、原frozen准入和原完整计算profiler均不放宽。

F8/W8、threshold0.15、最多连续复用一步、最后一步全算。本批每请求在
`8,10,...,36`这15个实际step复用中间24层，少算360块；每个runtime请求仍
成功执行920个混合FFN block，并完成40次实际采样更新、VAE及完整PNG。
数学最多16个合法skip，工具按真实mask/计数验证，不为任意prompt硬编码15。

关键不只是打开cache开关，还包括有界动态W消费者：

```text
当前FFN layer7提交 → one-hop已准备QKV layer8
  proxy不命中 → 正常消费8...31
  proxy命中   → 加回前一完整步middle residual
              → drain并弃用未消费的layer8 W producer
              → 在同一bank预取下一步layer0 → 原Euler更新
```

旧QKV stage遇到不匹配future也会安全退役，不是会直接消费错误权重。本实现在
实际跳层边界显式退役/重新预取，记录开销和消费者身份；未新增W bank、不提前
计算下一步QKV或复用旧target。discard是这里的合法、可计数行为，普通full-work
路径的“零discard”检查不放松。

每个实际skip必须有一次callback、一次真实future discard和一次成功step-head
restage。prefetch `submissions=claims+discards`、producer数为3×submissions，
成功claim/weight-completion/hit严格相等，失败/retired-claim/pending0。
`qwen_runtime_conservative_cache`记录真实step索引、saved blocks及三项边界计数。

静态conditioning K/V与动态残差分离：每请求重新configure/清空DB状态；结束后
先复制实际cache计数、清回调并drain，再disable/clear DB后调用原严格snapshot
export。没有放宽“runtime recipe不得带DB/live callback”的export guard，快照
不带target、middle residual、旧回调或图。新cache recipe与原full-work key隔离。

## 固定图、实际工作与恢复

FFN仍R1024/H4096/C6144，GPU/ANE各6144 intermediate channels；QKV仍R512、
GPU/ANE各512目标行。每operator一份动态W图跨32层/步骤复用，不烘焙32层W。

| 本批conservative每owner三请求 | 生成 | 单参考编辑 |
| --- | --- | --- |
| 成功混合FFN blocks | 920 /920 /920 | 920 /920 /920 |
| FFN workload预测 | 952 /920 /920 | 1080 /920 /920 |
| QKV workload预测 | 888 /888 /888 | 888 /920 /920 |
| 合法future discard /head restage | 每请求15 /15 | 每请求15 /15 |

初始化3次FFN能力检查单列，不混入workload。四条独立runtime timeline（含两条
恢复）各两个稳定ANE model，hot compile/load0，实际evaluate与成功计数一致。
W8A8是数据流/类型，不是物理INT8 MAC证明；selectors不覆盖opaque driver内部。

额外同owner四请求：prime73→hit73→hit74→hit73。生成和编辑均在**相同warm
route**下恢复完整RGBA exact，换seed没有返回陈旧图。编辑prime/hit原有首步QKV
placement差异仍记录：缓存档SSIM约0.998195，不强行称miss/hit逐位相同。

四组三请求stock/timeline/反序重放共12张比较均RGBA exact；new Off控制的六张
PNG与V11原控制逐字节相同，未拿历史时间作新分母。当前没有取消/真实驱动超时、
任意其他尺寸/LoRA/更多参考数或全部场景资格。

## Profiling与独立审计

新`runtime_cache_collect.py`是独立近似tier工程，不把原`collect.py`/full-work
auditor的1280-block合同改松。它保留完整原请求、progress、首/热时间、PNG、
固定program健康、真实执行/跳过/retire计数、原始process-tree内存及Private
selectors；GPU on/off也实际执行，encoder保持GPU。GPU stock只有skip数量，
未记录的逐步mask标为null，不能虚构“与runtime相同mask”。

`runtime_cache_audit.py`重新读取实际请求、全RGBA、原权重/encoder/VAE全SHA、
669输入exact native archive、捕获工具字节、memory原始hash链、load原始采样、
实际sampler进度、每model evaluate和完整时间分母。speed只允许干净stock三请求；
timeline、四请求恢复或受污染窗口均为null。质量与source/功能资格分别记录。

最终审计：`outputs/qwen-runtime-cache-independent-audit-v2-20261019.json`，
十四成功窗、44张完整PNG、1760实际采样步、32680个成功混合FFN block；
11组完整配置/质量对照，两个warm-route恢复通过。所有失败/被拒的预算保留。

十个stock窗口process-tree峰值约17.88–21.62GB，swap-out/压缩增量均0；
系统swap-in最大约4.26MB。只是本机整窗口观察，不是hard cap、单独ANE/driver
归因或跨设备内存排名。build/test/CPU图片审计均在模型窗口外，未停止共享服务。

## 复跑、版本与后续

```sh
.venv/bin/python -B tools/profiling/qwen_base/runtime_cache_collect.py \
  --library build/qwen-runtime-conservative-cache-v12-20261019/libturbocider.dylib \
  --route runtime --cache conservative --case edit --mode stock --requests 3 \
  --output outputs/FRESH-runtime-cache-edit
```

控制改`--cache off`；同cache的GPU控制改`--route gpu`。`--mode timeline`独立进程，
`--requests 4`只作上述prime/hot恢复。`--prompt`支持另一个固定指令/提示词；
更多holdout须先固定输入/预算，不用通过的小样例替代广泛编辑质量。

- Native：`build/qwen-runtime-conservative-cache-v12-20261019/libturbocider.dylib`，
  SHA256 `7d2f73686190a01aa69665c6e7b4e07d40bedc93a3b0302507aeea53243474bb`。
- Runtime ID：`tc-runtime-build-v1-dae7d111441612e9e763f1b099cd1b4f7d5bd5a38e846c7e413e599fde17e4aa`。
- 源码：`outputs/qwen-runtime-conservative-cache-native-source-v12-20261019.tar.gz`，
  SHA256 `840156f5a68d62507d8888599cc6621c3922d70b22466a6d4b9c677f7d7a3850`。
- Pilot旧采集器：`outputs/qwen-runtime-cache-pilot-tools-v12-20261019.tar.gz`；
  添加四请求恢复后源字节不同，审计显式带此archive，不用新字节冒充旧窗口。
- 40项相关host测试通过，日志`outputs/qwen-runtime-cache-host-v15-20261019.log`。
  含域/默认、全40×32边界模拟、actual/skip计数、暖恢复、预算、原始load负控。
- 新库C API＋真实MLX生命周期142项：141通过、1既有fixture跳过，日志
  `outputs/qwen-runtime-cache-regression-v12-20261019.log`；不是142项图片质量资格。
  首个host验证器漏写tuple中的actual40而失败，修复前未运行新tier模型窗口，失败日志保留。
  新C++布局不可混链接旧helper，Python C API仍按实际runtime ID绑定。

生成与编辑分开选路：保留原完整计算研究基线，编辑缓存候选继续更多指令/参考
holdout与异常生命周期；生成先处理原W8/QKV累计误差，不把20.94s当已通过质量
的生成档。所有新中间产物仍在已忽略outputs/build，约31GiB可用，没有下载、
删除模型、清用户缓存、改App/default或commit。完整优化目标未完成。
