# Qwen512 base：同库正反序 frozen / Private / GPU 对照

2026-10-10，Asia/Singapore，M4 Max64GB/macOS26.6.2。接续
[生图分阶段](local512-qwen-generation-phases-2026-10-10.md)：Private 已比
GPU 快，但不能因此称其超过已有 checkpoint-bound frozen 图。本轮只用
现有本地模型、两类 manifest 和已编译 artifact；没有下载、导出、重编译
或模型/adapter/ref 改写。完整 Z/Qwen、LoRA、编辑/encoder、GPU kernel、
GGUF/ConvRot 与真正 ahead-decode 目标仍 active。

## 同一任务、同一库、相反顺序

原 BF16 Qwen Image2.1、512²、40步、seed29；相同三个狐狸提示词，只有
natural/soft/warm winter light 不同。每臂独立进程，一冷两 fresh warm，
conditioning 全 miss；共18请求。forward 为 GPU→Private all→frozen，
reverse 为 frozen→Private all→GPU，是真正同 workload 的相反顺序，
不是拿单图和双图、base与LoRA的不同任务互相充当反序。

全部使用同一 Private build 的 CLI/库。GPU encoder eager，**每请求加载
原 source、不保留 encoder 权重**：所有臂真实 enabled/retained/reused
均 false，retained_bytes0、loads_session_total1/2/3、source_bytes
17,534,247,392、decline_reason为空。没有伪造 retention 记录，也没有
把取消 retention admission 当内存 guard 绕过。这个生命周期只用于本次
公平对照，不把结果乘到此前 retained-encoder 窗口或推荐默认关闭复用。

Private 为 Fa5120/Fg7168、c1056 v2、W8A8/F32 partial join、fixed async1；
prefetch/lookahead/W-code cache/time reuse关闭。frozen 使用已有32个
int8_pc / sq_v1_both compiled图，rows1024、hidden4096、width12288、
ANE prefix[0,6144]；首步完整GPU，后39步 GPU suffix/Core ML prefix并行。
不是首步全部ANE，也不是 Private runtime-weight 图。Core ML
CPUAndNeuralEngine 不证明所有算子物理落在ANE或原生INT8 MAC。

| route | forward warm request s | reverse warm request s | forward 首步 / 后39步 s | reverse 首步 / 后39步 s |
| --- | ---: | ---: | --- | --- |
| GPU | 43.457422 | 43.349057 | 1.131452 / 40.893068 | 1.136840 / 40.895779 |
| Private all | 34.601921 | 34.523965 | .940994 / 32.245310 | .968871 / 32.267512 |
| frozen base | 33.313633 | 32.995109 | 1.144776 / 29.163854 | 1.146411 / 29.006661 |

各自窗口的 Private request 相对GPU下降20.38%/20.36%；frozen下降
23.34%/23.89%，对Private下降3.72%/4.43%。四个warm原始样本的合并
中位为 GPU43.352718 / Private34.541752 / frozen33.121251s；只作同
workload汇总，不替代窗口各自分母或广泛统计资格。没有拼接各臂最快
阶段构造一条从未跑过的混合请求。

两个 continuous enclosing CPU load check 均失败，因此以上始终为
busy diagnostic，qualification=false；不升级默认或宣称设备独占。
frozen两个窗口的warm request范围分别32.394–34.234 /32.141–33.849s，
text_encode分别1.455–3.256 /1.527–2.884s；Private对应约.648–.947 /
.681–.688s。source生命周期相同不代表stage耗时相同；没有将text差值
唯一归因给ANE、compression或某个kernel。

## 冷启动、实际执行和内存

冷请求 forward GPU/Private/frozen为45.469439/36.897084/44.791788s；
reverse分别44.893228/37.344383/42.076648s。frozen冷 hybrid_setup
11.600196/8.381534s，包括原checkpoint完整校验、加载和warmup；Private
冷请求在本轮较低，热态赢家不能无条件充当冷请求赢家。

每 frozen request 实际1248次 runtime Core ML prediction；累计
1248/2496/3744，warmup只在session首次32次，总calls1280/2528/3776。
checkpoint SHA verified、32block、int8_pc、CPUAndNeuralEngine、failure0。
它的 `qwen_ffn_phases.policy=gpu`、runtime-weight callback calls0，是
该telemetry的正确scope，不把Core ML预测重新贴成Private callbacks。
Private每请求1280实际calls/成功channel blocks，累计1280/2560/3840；
phase分别32+1248，failure/fallback/retry0、headroom1。GPU全程0calls。

六份100ms进程树memory evidence独立重放，均complete、原SHA/report一致。
GPU/Private峰值约30.55–30.96GB，frozen约36.12–36.14GB。frozen forward/
reverse系统swap-out为2,555,904/2,097,152bytes，compression累计计数为
16,318,382,080/15,434,219,520bytes。forward GPU和reverse Private分别
有65,536bytes swap-in，其余swap-in0；其余swap-out0。不能说整个窗口
无swap/compression，也不能把系统计数归因于单个进程/ANE/外部App；
scope是process-tree load+cold+warm+exit，排除external service/driver归因。

## 两个保留的失败，不追认倍率

v1的GPU/Private保留encoder，但frozen retention因growth_limit decline，
实际变为request-local；native请求完成，validator仍拒绝不等生命周期，
summary保持incomplete。不得拿其中frozen热时间与retained controls算比。

v2统一关闭retention，但旧validator错误期待无record；native本来就会
输出disabled record及真实source loads。GPU三请求完成，screen在首臂
validation失败，summary仍incomplete。最终新增显式typed disabled-source
contract，直接验证raw record，不改写enabled/reuse或拿缺少record当成功。

额外raw重放发现native base省略可选的零LoRA字段：保留合法absence，
拒绝present bool/非零/非法counter，并新增对应回归；不要求base伪造
LoRA-only telemetry。两份v3原始数据最终完整通过新validator。

## 图片、工具和保留决策

每route三张forward/reverse PNG分别逐字节一致，不代表GPU/Private/
frozen互相逐字节一致。已检查两个窗口的case1 whole512：狐姿态、头脸、
尾巴、雪地和枝条构图很接近；Private纹理差别很小，frozen毛发、眼部/
雪粒细节变化稍大，未见新增明显棋盘格、断裂或色块。仅有限agent观察，
不是多scene/seed、用户接受或完整质量资格，不要求严格latent等价。

`qwen_ffn_phase_screen.py --generation --frozen-manifest <existing compiled
manifest> --request-local-encoder --modes gpu,all,frozen` 支持本轮对照；
frozen限定base generation，不接受LoRA或layer screen。反序另起fresh
output并用 `--order frozen,all,gpu`。可用已有busy wrapper保存CPU/load/
memory，原strict `--observe-load`没有放宽。

最终36项host回归通过无skip，六route raw receipts和memory独立重放通过。
没有新native源码/build；保留的Private/Public各502 source inputs重新与
当前tree匹配、seal正确；仍含原用户ConvRot草稿，不称clean staged-only
或App发行。Public库身份核验不等于本轮Public runtime-weight性能实测。

本轮没有生成native `.o`/module-cache，也没有新compiled模型；回查两个
retained build的`.o`均0，所有owned模型jobs已terminal。保留库/CLI、
全部失败/有效日志、PNG/manifest和原模型/adapter；未清用户或driver缓存。

决策：匹配checkpoint/geometry且内存准入允许的base热请求，保留frozen
候选；Private不是所有route的全局最快，但有更低的本轮cold时间和可
复用权重路线。六步LoRA不套用base图；1–2ref编辑仍独立按首步FFN
GPU/ANE通道分工、KV-hit完整GPU筛选，encoder仍用已测较快GPU。
下一步继续实际handoff/LoRA收益、encoder与更多scene/seed，以及真正
有界GGUF ahead-decode，不重复已有负结果，也不因此宣告整体目标完成。

完整原始路径、哈希、样本、memory与资格边界见
[机器记录](../design/validation/local512-qwen-frozen-base-20261010.json)。
