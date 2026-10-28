# Qwen base512：固定Program合同与真实失败分支

2026-10-18，M4 Max/64GiB。只优化原BF16 base、512²、完整40步，不涉及LoRA/
student/Z Image。继续[联合QKV＋FFN](qwen-base512-joint-qkv-ffn-2026-10-18.md)
和[份额/K分片/质量检查](qwen-base512-joint-share-quality-2026-10-18.md)。
本轮解决的是热重建和失败后自动重试，不是新的矩阵吞吐加速；产品默认仍不变。

## 实现：由观测推进到显式策略

新增默认关闭的`TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM=1`，在executor构造时捕获，
不在溢出时重新读取可变环境。只准入显式授权的Private W8 SwiGLU；拒绝public
preparation、自动通道校准、request-row重配置。rows/通道/权重ABI初始化后固定，
长序列仍分块消费同一图，不烘焙checkpoint，也不按层建模型。

| 状态 | ANE行为 | GPU行为 | 缓存/owner |
| --- | --- | --- | --- |
| 正常成功 | 原program、动态权重、固定shape | 原GPU complement与join | 同一program跨层/步/请求复用 |
| 溢出/无效结果 | 禁止build、join后停止失败路径 | 完整FFN重算，不发布部分scratch | 释放失败owner资源；拒绝新prefix |
| 同配置下一请求 | 不自动创建/装载/预测失败路径 | 普通GPU blocks | 保留失败状态和累计计数 |
| 显式卸载/关闭路由后重启 | 允许新的冷初始化 | 按新请求重新选择 | 新owner/配方命名空间 |

`PrivateW8Graph::run()`在overflow分支、`build()`自身都加了禁止替换program的
检查；worker错误在该策略下停用executor。`HybridFfn`在释放graph前保存计数，
失败对象仍被Session视为已处理的同一配置，避免下次请求重新尝试。
Qwen的对等QKV也会停用；FFN初始失败时直接创建无Device/Program的失败占位。
所有旧生产者/ANE/restore/GPU head完成与取消drain合同保留，不放松安全检查。

该策略不改变正常数值、W8/Hadamard、F32 join或采样步骤。启用策略进入单独的
executor/prefix身份；unset/0保留旧身份和自适应headroom行为。实验仍要求显式
`allow_approximation`；不是无损BF16或物理INT8执行证明。

新增原生receipt：`fixed_program_requested/enabled`、`program_disabled`、
`program_instances_session_total`、`blocked_rebuilds_session_total`。
Program数是host owner计数（初始化/自检也包含），不等于编译API次数；早期未
选中factory的失败需结合原因与实际selector判断，不能只用默认0推断没有尝试。
实际compile/load仍由独立observer检查，不承诺不可见driver内部绝不编译。

## 真实溢出负控：旧分支确有热编译，新分支无

用自有合成BF16数组，不加载或改写checkpoint。M32/H128、完整FFN1024通道，
ANE/GPU各512。X8、对角gate/up32产生256，SiLU×up约65536，超过FP16 carrier
上界但仍是合法BF16；两项down .25令完整GPU输出恰为32768。
因此是有限输入、有限正确结果下的真实中间值溢出，不是伪造错误flag。

每个进程直接executor和HybridFfn各一个owner；先验证能力/安全输出，再触发
实际overflow。原始Private选择器、完成记录和全部BF16输出数组单独保存。

| 观测项 | 旧策略 | fixed-program |
| --- | ---: | ---: |
| 每个owner创建的Program | 3 | 1 |
| 两owner热compile/load | 4 /4 | 0 /0 |
| 包含能力自检的实际evaluate总数 | 15 | 9 |
| headroom | 1→16 | 保持1 |
| Hybrid下一请求实际ANE预测 | 1 | 0 |
| 当前与下一调用完整GPU fallback | 不需要，自适应恢复 | 两次均逐位一致 |

新策略阻止重建次数1、disabled=true、旧输出和输入不变；改变环境flag后，原
owner仍坚持构造时策略。Hybrid释放所有optional资源，失败状态/计数保留；无
Device的QKV占位在独立scope内没有compile/load/evaluate。
独立CPU复读完整数组，检查input8、GPU oracle32768和两次fallback的每个元素，
并复算实际API计数。不是拿两份相同GPU输出自证，也不是完整图片速度benchmark。

原件：`outputs/qwen-fixed-program-negative-{fixed,legacy}-v1-20261018`；
CPU审计：`outputs/qwen-fixed-program-negative-audit-v2-20261018/audit.json`。
第一版CPU审计的JSON phase int/string键比较失败日志保留；只修了派生审计，
没有回写/重跑或重标设备原件。编译次数是该实际窗口值，缓存命中后的旧策略
重跑也可能少编译，但仍可有replacement load；validator不要求每次都compile4。

## 完整Session失败：三次请求不重试

另用原base完整40步、seed73→74→73，保留原2GiB optional cap，刻意请求2GiB
code cache，使graph/scratch加cache的估计无法准入。没有实际分配2GiB缓存，
也不是伪装成ANE成功。此处验证初始准入失败，不冒充整图中途的数值溢出。

三次完整请求的FFN GPU blocks与QKV GPU plans都累计1280→2560→3840，失败计数
保持1；Program实例0、实际compile/load/evaluate全部0。说明同一失败owner保留，
QKV占位也没有被重新构造/加载。独立审计已将三张PNG及实际请求与冻结完整GPU
控制逐一比较，RGBA全部逐位一致，且seed74不同、seed73返回原图。
原件：`outputs/qwen-base512-fixed-program-session-fallback-v1-20261018`。
44.36/42.66/42.67s是故意all-GPU负控，不进入成功ANE性能分母。

中途溢出时是完整**当前FFN**重算与后续GPU路径，不回滚已经成功执行的近似
W8步骤；不宣称这样的整张图片会逐位等于从头纯GPU。初始准入失败则尚未执行
任何ANE，用完整GPU原件可独立检查整图一致性。

## 同库正常路径回归

同一新库，只差fixed-program开关，FFN6144＋QKV512，40步。stock不加载observer；
生成无参考，编辑两侧都保留标准1024参考和runtime-prefix hit强基线。

| 场景 | legacy热中位 | fixed热中位 | 反向legacy |
| --- | ---: | ---: | ---: |
| 文生图 | 31.653988s | 31.946884s | 31.843321s |
| 重复编辑 | 39.701146s | 38.728926s | 38.644302s |

存在明显窗口波动，如编辑控制热请求40.47/38.93s；没有设备独占/thermal/置信
区间资格。不把编辑表面差异说成代码提速，也不宣称已证明成功路径零开销。
该策略目标是避免失败路径的热program重建，正常算术没有新增queue等待。
相对各自反向控制，fixed热中位约+0.33%/+0.22%；不将不同窗口拼成更有利
分母，也不把timeline更快的数值当速度纪录。成功路径的精确开销界限尚未建立。

独立fixed timeline的生成和编辑均已完成，实际两个model身份，热compile/load
均0，FFN Program数全程1、blocked0；正常请求没有fallback/retry/headroom变化。
包含初始能力3次的实际evaluate：生成2563/2560/2560，编辑2691/2560/2560。
变长矩阵也完成：生成→1标准参考→2份相同标准参考→原生成；两个固定图、
FFN Program始终1，后续任务/shape切换没有client compile/load。任务不同，
不pool热中位数；双份同参考不授予不同主体的多图语义质量资格。
所有设备窗口结束后，独立审计检查原PNG、source/archive、memory及实际API：
生成/编辑off-on、timeline-stock、反向控制全部RGBA exact；新库legacy输出也
与旧同算术joint6144输出逐位相同。fixed仅改变失败策略，没有新增成功路径近似。

| 变长请求 | prefill rows | FFN/QKV实际预测 | client compile/load | wall |
| --- | ---: | ---: | ---: | ---: |
| 生成 | 1048 | 1312/1248 | 2/2（初始化） | 34.415730s |
| 单参考 | 5155 | 1440/1248 | 0/0 | 47.364667s |
| 双份参考 | 9257 | 1568/1248 | 0/0 | 62.555074s |
| 返回原生成 | 1048 | 1312/1248 | 0/0 | 31.686128s |

首请求另有3次能力evaluate。首尾RGBA exact；两个evaluate model身份稳定，
FFN Program实例始终1。初始2次compile不能隐藏成“冷启动零成本”；后续任务/
shape变化均没有client compile/load。每请求都完成40步/1280 FFN blocks。
该矩阵process-tree峰值physical footprint约30.27GiB，系统压缩/解压累计约
4.51/3.96GiB，swap-in/out均0；系统计数不归因单一进程。不是ANE槽位大小的
内存宣传，也不证明driver/GPU独占。全部窗口仍`qualification_passed=false`。

本轮31次完整请求/1240步，其中28次成功ANE请求、3次all-GPU失败负控；另有
两组小模型真实溢出组件，不混为图片请求。

最终审计：

- `outputs/qwen-base512-fixed-program-{gen,edit}-audit-v1-20261018/audit.json`。
- `outputs/qwen-base512-fixed-program-sequence-audit-v1-20261018.json`。
- `outputs/qwen-fixed-program-negative-session-audit-v3-20261018/audit.json`，含
  合成CPU oracle、实际selectors与完整Session GPU PNG比较，旧派生审计保留。

## 构建、工程与范围

新库：`build/qwen-base-fixed-program-v1-20261018/libturbocider.dylib`，SHA256
`9afaea9f10bcd375f4c5c93f013055261a43ad95f608a664de8dfa9d49d58a6a`；runtime ID
`tc-runtime-build-v1-01079399483fca84fce2b16ec418cb9de38393065069599c5ff82d97ab1d1044`。
657输入封存：`outputs/qwen-base-fixed-program-native-source-v1-20261018.tar.gz`，
SHA256 `2594dc3ff17bbf12dfc9a2291a1655fe51ce12fc945acb642ffeadf4572dec32`。
本轮工具/测试/合成probe/descriptor封存：
`outputs/qwen-base-fixed-program-tools-v1-20261018.tar.gz`，SHA256
`05f715db32ae30bdf3da083ea719e3cbea3dd98111d66337d1ba00220ba8c6ef`。
旧656库/原件保留，不用新source身份回填旧测量。

工程：`collect.py --fixed-program`、`fixed_program_collect.py`与合成C++probe、
`fallback_collect.py`、`fixed_program_audit.py`、更新后的sequence auditor。
原生真实MLX生命周期24cycles/8variants及joint数值/idle内存回归通过，日志
`outputs/qwen-base-fixed-program-lifecycle-v1-20261018.log`。最终58项profiler
host＋1项fixed-policy CPU＋3项Qwen options（62项host）通过，日志
`outputs/qwen-base-fixed-program-host-tests-v2-20261018.log`。public-disabled QKV
TU单独编译通过，4条既有未用helper警告，不冒充完整public库资格。
设备期间不并行自有构建/测试/CPU数组审计；所有模型/构建/测试/审计进程已退出。

此轮仍不改App/产品默认，不下载、不删权重/旧数据、不停止ComfyUI。新的矩阵
吞吐瓶颈（FFN、长参考attention、stage/restore边界）与更广泛语义质量仍需
继续优化，目标保持active。

下一步值得验证的替代方案是把变化的headroom做成固定ABI的运行时标量输入，
在同一program内处理可恢复数值范围，避免热编译，也避免数值溢出后长期退GPU。
这尚未实现/采用：必须比较正常路径的新增算子成本、异常重试次数、数值/前缀
配方和整请求吞吐，不能以“更灵活”自动判为更快。
