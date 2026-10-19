# Qwen Image 2.1 base：固定 runtime ANE、profiling 与 vpipe 对照

后续已完成[固定Private QKV＋FFN联合实验](qwen-base512-joint-qkv-ffn-2026-10-18.md)。
下文仍保留原v8的构建/测量范围，不据后续结果改写历史数据。

后续已有独立的[runtime-prefix实验](qwen-base512-runtime-prefix-2026-10-18.md)，
重复编辑获得新增收益。本文保留之前冻结v8及其测量范围，不将新库结果回填成旧数据。

2026-10-18；M4 Max /64GiB。仅512²文生图、图生图，原BF16 base，完整40步，
无LoRA/student、无temporal DiT cache，不包含Z Image。所有数字均为本机小样本
观察；存在竞争推理负载、无GPU独占证明，`qualification_passed=false`。
没有修改App或产品默认，也没有将未通过画质资格的路线默认启用。

## 已取得的完整请求结果

无observer的stock进程，首请求单列，后两次完整请求取中位数，包括VAE/PNG。
每进程seed73→74→73，不是系统严格冷启动，也没有置信区间或完整反序资格。

| 任务/路线 | 热请求中位数 | 相对同任务GPU耗时下降 |
| --- | ---: | ---: |
| 文生图GPU | 42.668064s | — |
| 固定ANE4096，cache off | 35.828683s | 16.03% |
| ANE4096，1600MiB surface code cache | 35.961744s | 15.72% |
| 固定ANE6144，cache off | 33.732293s | 20.94% |
| 单参考编辑GPU | 49.911089s | — |
| 单参考编辑ANE6144 | 45.393535s | 9.05% |

编辑GPU保留已有immutable prefix snapshot：首次miss、随后hit。runtime现有
规则不准入snapshot，因此这是完整路线对比，不是同prefix状态的局部消融。
不能关闭GPU已有加速来制造更大的hybrid倍率。参考处理仍为标准1024，4096
reference latent tokens，不是把参考降到512后声称收益。

原始成功窗口见`outputs/qwen-base512-runtime-{gpu,off}-gen-20261018`、
`qwen-base512-stock-{cache-gen-v1,c6144-gen-v2,gpu-edit-v2,c6144-edit-v3}-20261018`。
独立CPU复核：`outputs/qwen-base512-fixed-runtime-profile-audit-v3-20261018.json`。
失败的2GiB cache准入窗口保留：它实际生成3张GPU fallback图，**不算ANE样本**。

## 一个跨层共享的固定计算图

descriptor是1024 rows /H4096 /原始F12288；进程初始化选择固定ANE intermediate
通道分区。已测4096、5120、6144；不在block或请求间更改分区。
每层gate/up/down作为runtime输入，同一图服务32层及所有步骤。首步长文本/
参考序列重复调用同一1024-row图，后39步使用1024目标rows。不存在逐层常量模型。

当前6144配方是Private W8/H128–H512、GPU/ANE各6144个FFN通道、F32 partial
join、fixed async、launch fence1、无future-bank prefetch、无code cache。
同一图只存算子/固定变换，不嵌入32层checkpoint。**图小不等于全部内存小**：

| 6144路线内存口径 | 大小/含义 |
| --- | --- |
| 单层ANE逻辑W8矩阵 | 72MiB理论值，3×4096×6144 bytes，不含scales/双bank |
| native slot账本 | 165707776 bytes，包含活跃slots，不能等同于纯权重 |
| native runtime估计 | 402653440 bytes，准入估计，非独立RSS/driver实测 |
| 原GPU权重/encoder/KV/attention/VAE | 另有各自生命周期和预算，不包含在小计算图里 |

W8A8是当前输入表示和量化数据流。MIL是INT8 slice→dequantize→MatMul模式，
不能据此声称已证明物理INT8 MAC、真实ANE硬件时长或GPU/ANE物理重叠。
源码入口：`native/backends/private/ane_mil.cpp`、`ane_w8_executor.mm`。

## 同进程跨任务/长度证据

`outputs/qwen-base512-profile-sequence-matrix-v5-20261018`运行四个完整请求，
每个seed73、40步，固定6144分区和同一个ANE model identity。实际native phase
rows与API调用一致：

| 请求 | 实际prefill rows | 模型预测次数 | compile/load | 完整请求wall |
| --- | ---: | ---: | ---: | ---: |
| 文生图 | 1048 | 1312 | 1/1 | 36.202594s |
| 一份1024参考编辑 | 5155 | 1440 | 0/0 | 49.250057s |
| 两份1024参考编辑 | 9257 | 1568 | 0/0 | 67.102627s |
| 回到原文生图 | 1048 | 1312 | 0/0 | 34.097373s |

首请求还有3次能力检查evaluate，单列，不混进模型预测。每请求实际1280个
成功hybrid FFN，零overflow retry、headroom保持1、无GPU fallback。首尾PNG
RGBA逐像素一致；中间两张编辑不是同一张stale输出。已实际查看生成及两张编辑。
两份参考是同一文件重复输入：验证sequence length，不授予多主体编辑语义资格。

这些wall来自timeline且任务不同，不计算pooled hot median或跨任务倍率。
该窗口进程树峰值physical footprint约30.3GiB，系统compression/decompression
计数有明显增长，swap-in/out均0；系统计数不是本进程独占归因。没有因为图
固定就忽略长prefix/KV/encoder内存。

独立审计：`outputs/qwen-base512-sequence-matrix-audit-v2-20261018.json`。
派生trace：`outputs/qwen-base512-sequence-trace-export-v6-20261018/trace.json`。
v5原分析中的sampler-step组曾跨请求汇总，保留原件但不要使用这些混合中位数；
v6按request拆分全部160个step，不改原始设备记录。编译/模型身份/计数证据未变。

这里证明的是被观察到的Private client selectors：首后请求没有compile/load。
不承诺不可见driver内部永远不会有编译。首编译约0.991s、加载约0.024s，未隐藏
进“冷启动零成本”的说法中。

## Profiling工程与归因边界

入口：[工程说明](../../tools/profiling/qwen_base/README.md)。主工程包含CMake
observer、完整请求collector、CPU analyzer、artifact audit、sequence audit、
vpipe对照consumer/collector/auditor及host负控。

| 模式 | 实际运行状态 | 正确用途 |
| --- | --- | --- |
| stock | GPU/runtime/cache、生成/编辑均已运行 | 完整请求性能分母，不加载observer |
| timeline | 4096/6144、编辑、变长矩阵已运行 | 真实compile/load/evaluate、Metal CB窗口、进度/采样步 |
| gpu-ops | GPU block0已运行 | 强同步算子诊断，不能当普通GPU分母 |
| runtime-layers | 全32层1280 FFN诊断已运行 | 同步FFN归因，不能与stock秒数混用 |

observer用真实64-byte Metal blit检验数据、完成与时钟包络，Private方法只转发
一次，保存bounded记录/drop/pending与退出恢复证据。Metal用公开commit/completion
和GPUStartTime/GPUEndTime。CB区间取并集；commit到completion总存活时间不是CPU
执行时间，也不能与request wall相加。`graph_build_seconds`包含runtime同步/
设备等待，不应误判成纯CPU构图瓶颈。当前没有shader counters、ANE硬件计数或
外部服务归因，不能用本工程声称这些覆盖已经齐全。

## 当前实际瓶颈与已验证取舍

1. Denoise占主要wall，VAE约0.6s。GPU block0 decode诊断中，FFN gate/up约14.03ms、
   down/residual约7.37ms，QKV约7.09ms。它们是同步host诊断包络，不乘32×40外推。
2. 4096→6144减少GPU complement暴露。native `post_join`约6.18s降至约0.31s，
   其中包含GPU head就绪，不是一个6秒纯join kernel被消灭；6144的暴露ANE等待
   相应增加。应继续平衡两引擎，不是无限加ANE通道。
   追加固定5120的stock三请求，首38.191166s、热33.804418/33.936853s，中位
   33.870635s；没有可靠优于6144的信号，不据此新增默认。该档只有stock窗口，
   不从其它分区借用其实际selector零热编译证明。
3. Cache把staging work约6.8s降到约0.04s，但原暴露stage wait本来很小；完整
   stock请求没有受益，故不采用1.6GiB缓存。命中率/工作量下降不能替代关键路径。
4. Runtime编辑仍有约6s单参考prefill，而GPU强基线有跨请求prefix snapshot。
   runtime-prefix复用是后续候选，尚未安全接入，不能复用GPU prefix却隐瞒数值
   配方变化。双参考还涉及更长attention/KV和内存，不能仅靠FFN解决。

## 当前vpipe源码与本机ANE对照

参考`../references/vpipe@18ddb3b5bd9c74ab7f2b4408ee78efc6cecd48e7`，checkout干净，
没有修改。读取的是当前Qwen2.1 consumer及shared ANE module/worker/emitter，
不是知识库旧c2989a版本的行为假设，也不把作者其他SoC/H3倍率当本机结果。

用独立fresh build链接原源码，权重来自同一Comfy BF16文件。它的gate/up为融合
布局，vpipe仅接受分离名称，故创建APFS CoW header-only views；不改原文件、
不复制或转换payload。view有约26.5GiB逻辑大小，但extent共享；创建时free-space
差约160KiB。view仅适合此reader的非连续索引，不是通用safetensors导出。
FFN first12288 rows与up second12288 rows来自原native切片语义，并单独复核
每层metadata及边界字节；不是全payload hash证明。

每个进程一条路线，原32层、resident BF16，无adapters、无i8/sparse附加tier。
固定合成latent/text，64文本rows，512目标，编辑增加1024参考。首轮单列、随后
3对prefill/decode。**这不是完整图片请求，没有encoder/sampler/VAE/PNG**。

| vpipe路线 | 文生图prefill/decode中位数 | 编辑prefill/decode中位数 |
| --- | ---: | ---: |
| GPU | 1.121440 /1.061707s | 5.897813 /1.298947s |
| ANE FFN AUTO | 1.121308 /1.061882s | 5.878605 /1.339173s |
| ANE FFN+QKV AUTO | 1.121642 /1.062236s | 5.891397 /1.336809s |

文生图实际ANE模型预测为0，三路线输出exact：默认2048-row chunk不能分给只有
1088/1024 rows的任务。编辑prefill实际预测计数（首轮加3次measured）FFN为
30/32/1/1，FFN+QKV为60/64/2/2；decode全部为0。控制器后来退回GPU，只周期
probe。不能把AUTO退回GPU后的中位数说成“全32层ANE稳定加速”。

额外timeline进程中，FFN、QKV各一个稳定model身份，hot compile/load均0，实际
evaluate与native credits一致。首次decode即使零ANE预测仍观察到130个CB；
源码先plan/stage/drain后`begin`才返回零，存在无有效ANE工作的边界成本。
其section历史又按模块实例共用，需隔离不同phase/rows。这里只给出源码与窗口
共同支持的归因，不将功耗/带宽竞争或控制器当成唯一硬件原因。

最大组件relative-L2：FFN约3.982%，FFN+QKV约5.543%。后者超过上游另一个测试
使用的5%界限；本次是不同编辑几何，不能挪用它的测试通过声明。没有端到端
图像质量资格，也没有观察到稳定完整路线收益，因此未移植为默认。

独立复核：`outputs/qwen-base512-vpipe-audit-v2-20261018/audit.json`，7个进程、
56个完整DiT forward，不是56个完整图像生成。stock与timeline分开；timeline
有约1.5MiB系统swap-in，stock各窗口swap-in/out均0，外部服务不归因。
原始cohort为`outputs/qwen-base512-vpipe-*`，失败的首次直读融合checkpoint
尝试保留，不计入成功forward。

## FFN / QKV的下一步实现选择

保留一份固定FFN图，必要时加一份固定MatMul/QKV图，而不是每层/每个sequence
编译一份。共享session-level ANE调度与总内存预算，但不将两个模块误当两颗ANE。
权重bank必须在真实完成后重用；独立输出、generation、取消/失败drain合同不放松。

FFN优先按intermediate通道分担，适合当前1024-row decode；可在独立进程初始化
时筛选固定合法分区。长prefix按同图分块，tail可padding或交给GPU，但必须计入
预测次数、浪费和同步。形状不支持时回GPU，不在热路径即席编译。

QKV是后续候选，不是已经获得512²整请求收益的功能。它在attention前必须join，
权重stage可隐藏的窗口比FFN短，增加边界容易吃掉约7ms诊断级机会。可测试较小
固定row chunk、Q/K/V分段直接stage到一个slot、避免永久3GiB packed副本，并融合
输出restore/scatter到norm/RoPE消费者。需完整block及FFN+QKV组合实测，再比较
整请求；W8误差通过attention传播，不能只过一次MatMul数值检查。

调度状态按operator/phase/实际rows/数值配方分开、有界保存；无合法share时在
staging/barrier前直接选普通GPU。固定图与灵活调度是两件事；调度不需要重编译。
目前没有完成这套新Private QKV组合或runtime-prefix snapshot，不能宣称全部
瓶颈已经解决。

## 质量与交付状态

W8与GPU BF16不全局逐位相同。ANE6144生成SSIM约0.9175/0.9748，编辑约
0.9971/0.9979；生成茶壶把手、壶嘴/旋钮几何可见变化，编辑两条路线都带较强
高光风格。未完成多prompt、文字、人物身份/材质或双不同参考的语义质量矩阵。
5120生成SSIM约0.9682/0.9744，在seed73上更接近GPU，而速度差只有约0.41%；
因此保留为速度/画质的另一个候选，不因没有速度纪录就丢弃，也不把单张改善
外推成普遍质量提升。两档都未取得产品采用资格。

26项本工程host测试通过，覆盖observer/fixed graph/route拒绝、混合任务不pool、
header view、actual prediction与状态恢复负控；不是26项设备资格测试。
原native v8库未改，652输入与封存源码吻合。当前所有已列设备进程均终止；不
停止用户ComfyUI，不做系统设置修改、下载、删除或产品默认替换。
