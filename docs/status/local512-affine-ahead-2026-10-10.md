# 实际有界GPU ahead-decode：四层真实FFN链已跑，GGUF仍优先packed

2026-10-10，Asia/Singapore，M4 Max64GB/macOS26.6.2。接续
[融合typed解码/finite](local512-affine-fused-finite-2026-10-10.md)。本轮实现
真正独立producer stream的current/next准备与ready消费，不把已有同步
window或async计数重新命名。完整Z/Qwen base/LoRA、编辑/encoder、GPU/
ANE和模型级GGUF/ConvRot优化目标仍active。

## 实现：两个逻辑job，不覆盖escaped物理目标

研究 `AffineAheadWindow` 位于 `tools/native/affine_ahead_candidate.*`，
不改production native源码、library或默认路由。owner thread管理1/2
logical pending jobs；prepared immutable source arrays、content ticket/
source generation和两个预分配目标一起留存。第三个pending提交拒绝，
ticket不能重复take/乱借；不同线程不能改owner状态。

先验证source availability/typed geometry与物理span，owner预编译Safe
Metal shader并检查TG256，再经既有gguf_storage分别ledger admission/
fresh-bin allocation为dense和status创建独立已计费backing。自定义
MLX多输出Primitive在独立stream写入这些目标，outputs明确alias自己的
target，不用修改input或用未申明in-place fast kernel。primitive不CSE
不同提交；source/target与target/target alias拒绝。

融合核仍保留FP16/BF16 metadata、typed expression、存储与finite status，
batch4/TG256。`take`等该job的dense/status/max完成且available才发布；
finite失败先退休producer command-buffer owners，丢弃整个新目标。
drain先完成全部job再退休stream和owner；析构不能证实drain时停止研究
进程，不冒称实现了production全GPUfallback/真实GPU timeout恢复。

每目标的allocation epoch是process-local atomic新epoch，不复用source
generation：同一immutable source可一直generation1，allocator handle
回收后的不同大小新backing不能误认为旧alias。Data/StorageLease随
escaped/lazy reader保留；清logical slot不假装归还物理budget。

这是真实软件层异步producer/consumer，不是物理GPU overlap证明；同GPU
带宽/调度、CPU allocation和command-buffer retirement仍会影响收益。
两logical jobs不等于两张物理dense总量，更不是两个完整FFN槽。

## 不同层真实source、依赖链、完整窗口

只读本地Z的layer0..3 gate/up/down，12个不同payload/codes SHA，拒绝
重复源伪装多层复用；原prepared packed bank保持不变。M1056合成sine
输入，每层实际SwiGLU→down→tanh稳定输出，再输入下一层；ConvRot对
当前input/hidden用相同Shared H256旋转，保留legacy BF16 stored scale。
没有attention/DiT/LoRA/encoder/图像推理，不能当完整请求。

三臂：完整compiled packed FFN；同一producer的immediate submit/take；
两slot提前提交下一层gate/up、与当前down计算并发调度。两dense臂用同
一primitive，pipeline当前层consumer保留3张dense，未来pending2张；
因此ledger覆盖5×75MiB加2status，Q4/Q8/ConvRot上界分别393,543,680/
393,838,592/393,838,592bytes（约375.3/375.6MiB）。不是150MiB双矩阵
retention或整个进程RAM上限，resident packed source/其他tensor另计。

每层三个consumer完成后，所有臂都退休default compute stream；链尾退休
producer stream后检查claim0。array ready不等于command-buffer captured
readers已释放，不能把省掉这些retirement成本当真实内存优化。

每臂3warmup、9hot，循环换序；每dense trial实际submit/publish12、fail0/
drain0、pending最终0，immediate峰值1、ahead峰值2，全部claim最终0。
读/packing与编译不计时，所有预声明样本/观察保留。v8和v9为两个顺序
窗口/两个probe版本，v9只加stress/CLI回归，不能混用binary或跨窗口分母。

| source / version | compiled packed ms | immediate take ms | layer ahead ms |
| --- | ---: | ---: | ---: |
| GGUF Q4 / v8 | 73.900250 | 90.520083 | 77.479750 |
| GGUF Q4 / v9 | 73.532667 | 95.219291 | 76.140500 |
| GGUF Q8 / v8 | 73.908583 | 92.602667 | 78.050375 |
| GGUF Q8 / v9 | 74.010791 | 92.788875 | 76.817583 |
| ConvRot / v8 | 81.128583 | 93.528333 | 80.379417 |
| ConvRot / v9 | 81.078917 | 94.836125 | 80.042542 |

ahead对immediate名义减少14.4–20.0%（Q4）、15.7–17.2%（Q8）、14.1–15.6%
（ConvRot）；对packed，Q4/Q8仍慢3.55–5.60%，ConvRot仅少.92–1.28%。
六组输出在当前合成四层链对packed逐字节一致/relL2=0，不是所有shape/
scene/seed或完整模型画质资格。5%component budget不替代用户图像接受。

六个host observers无errors、joined、binary unchanged；v9最大gap约
.151/.151/.152s。qualification均false，不称quiet-window/设备独占/
GPU timestamp/物理overlap；host job wait有依赖/嵌套，不与compute span
相加或全部唯一归因给解码kernel。

## 保留的失败与最终回归

- v1缺Metal-cpp include路径，compile失败，改用现有SDK headers。
- v2设备编译拒绝program-scope普通constexpr；改为macro，并将shader
  compile/能力检查前移到owner submit之前，不让它在async worker才失败。
- v4严格编译拒绝misleading indentation；补显式loop braces。
- v5取消全producer sync后暴露finite失败回滚仍有command-buffer owners；
  失败/drain加完整retirement，普通take保留job-specific完成检查。
- v6触发aliased-storage capacity保护；不复用source generation作为fresh
  target epoch，新增独立atomic分配epoch与不同大小288次重复source压力回归。
- v7真实FFN链结束尚有旧command-buffer reader claims，保留failed receipt，
  在计算层/链尾加入完整retirement；不将incomplete结果追认倍率。

v8/v9真实三source链均完成；最终v9 1项opt-in实Metal unit通过无skip。
包含原window12/attribution12、融合numeric144/boundary48/非法252/typed
overflow240，以及新ahead12个dtype/group场景：A/B乱序、source隔离、
owner guard、escaped admission、nonfinite reject/refill、cancel drain/
destructor；每场景另24次相同source generation、不同大小fresh target
共288次通过。20个非法CLI组合在source access前拒绝；实际计时binary
的ahead tests也通过。重复不重复计数，不是全仓suite/完整模型资格。

生产Private/Public保留库的502 inputs/seal仍独立匹配，原用户草稿未
夹带；standalone使用absolute rpath，没有adjacent dylib/loaded-image
证明或新App发行。v9 probe SHA为
`00c61c894a14a8554896e86a3250bf0d83437a6818c20b5aafa12c9fea5f327f`。

全部owned builds/tests/链jobs已terminal；temporary test目录自动清理，
9个owned probe build没有`.o`/module-cache。保留小binary、失败/有效
logs与receipts；不下载/改写模型、清driver/用户cache、删除旧compiled
artifacts或signal其他进程。

## 决策与完整目标

有界ahead从“未实现”变为可执行研究路径，但当前GGUF仍选更快packed。
ConvRot约1%组件差异不足以默认接入，尤其没有真实LoRA/attention/全图
或memory完整资格。下一步以实际模型生命周期/更长层序/有利shape确认
buffer/handoff、部分权重consumer和Public/Private盈利；继续Z/Qwen
base/LoRA、Qwen1–2ref编辑首步GPU/ANE与encoder，不缩成这个四层probe。

[机器记录](../design/validation/local512-affine-ahead-20261010.json)绑定原始
各样本、实际工作/ledger、source与源码/binary身份及失败路径。
