# Runtime ANE compiled graph 的私有快照与整模型复测

`ane_runtime.mm` 现在在加载 Core ML 前逐文件验证 runtime-weight compiled
graph 的 manifest 摘要，将验证过的字节复制到独立私有临时目录，并仅从该目录
加载模型。私有目录保留到异步 worker 已结束且 Core ML 模型引用释放之后，
再清理；源 artifact 后续移动/替换不能改变已加载模型所见的文件。
manifest/图缺失、未列出的文件、符号链接或摘要不匹配仍拒绝加载，
不把不可信图的失败悄悄算作成功 runtime 加速。

离线 probe 用临时小图验证：加载后移动原 compiled graph，既有实例继续
实际 prediction；再次从同一 manifest 加载必须拒绝；进程退出后私有目录
为空。`make test-runtime-ane` 复测退出 0（5 host、10 Core ML/MLX
集成；另有图结构检查）。这修复逐文件校验后、Core ML 延迟读取原路径
之间的间隙；不证明物理 ANE residency 或系统内存资格。

## 构建 A/B：安全快照的成本

M4 Max 64 GB；BF16 base，狐狸雪景/seed42，512²，Z 8 步、Qwen 40 步；
resident，每 trial 独立进程，一冷两热，runtime v2、auto chunks。
Qwen 旧/新两库都开启同一可选 GPU Q/K norm-RoPE。
顺序为旧→新→新→旧，每个模型四 trial、共 12 请求。
旧库 SHA256 `f3f6b6a05ce6eecb476f314f1fa1fccbdb1d2ef24fc09fdc5b72049b0390ca22`；
新库 `28ff6aafe0001d916077c0216d4ac8ae1f742dee1d18fa56a84153ed711719e1`。
驱动在每 trial 前后核对 CLI/库、runtime manifest 及逐个图文件的摘要；
screen 拒绝 failed/fallback 并核验真正的 prediction 与采样完成标志。

| BF16 base | 旧库四热中位 | 新库四热中位 | 旧/新 |
| --- | ---: | ---: | ---: |
| Z 512²/8 | 6.356 s | 6.403 s | 0.993× |
| Qwen 512²/40，Q/K 融合开 | 35.610 s | 35.554 s | 1.002× |

整请求 `request_wall` 含 VAE/PNG、不含冷请求或模型加载。
这组数据未显示可可靠归因于快照的热态退化，也不能证明冷加载无额外成本。
8 个 runtime trial 的实际 prediction 调用均大于零、失败和错误回退为零；
独立进程树采样合计 5,250 个样本，最大间隔 110.188 ms，窗口内系统
swap-in/out 和 compression 增量均为零。首个 Z 旧库 trial 的 auto 调度
与其余三个不同：三请求累计 667 次 runtime prediction，其余 Z trial
均为 688 次；因此第一旧库 warm PNG 与后续不同。第二个旧库 trial 和
两个新库 trial 的末次热图字节相同；Qwen 四 trial 的末次热图也字节相同。
这些变化不能归为安全快照的数值差异或广泛画质验收。

本地、被忽略的原始证据：
`outputs/runtime-ane/artifact-lease-abba-2026-09-30/`，含 driver、summary、
各请求/JSONL/PNG、独立采样流及 verifier 报告。

## 新库同组 GPU/runtime/冻结图对照

同一新库 `28ff6aaf…`，相同模型、prompt、seed、步数、512²、Qwen 三路
统一开启 Q/K 融合；顺序 GPU→runtime→冻结→冻结→runtime→GPU，
每 trial 一冷两热，三路线各池化四个热请求。runtime 为 Z c352 与
Qwen c320/K1024/N512 v2；冻结图沿用各模型的 base W8A8 manifest。
driver 对前后 CLI、库、两类 manifest 哈希和 receipt 完成状态做门禁。

| BF16 base | GPU 四热中位 | runtime 四热中位 | 冻结四热中位 | GPU/runtime | GPU/冻结 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Z 512²/8 | 6.997 s | 6.399 s | 5.347 s | 1.093× | 1.309× |
| Qwen 512²/40，Q/K 融合开 | 41.744 s | 35.592 s | 29.062 s | 1.173× | 1.436× |

两个 runtime trial/模型均有实际 Core ML prediction、无 runtime failure
或错误回退；冻结图亦有实际 prediction、无失败。Z 自动调度在正向
runtime trial 累计 627 次 prediction，反向 688 次；Qwen 两次均为
3,648 次，不能把不同 chunk 调度假装成固定分区基准。
12 个进程树采样窗口共 8,003 个样本，最大间隔 110.271 ms，均无
swap-in/out；两次 Qwen 冻结窗口的系统 compression 增量分别约
36.6/11.9 MB，其他窗口为零。它们是系统计数，不归因于 Core ML，
且进程树 footprint 不含所有外部服务/driver-wired 内存。

已查看新库 Z runtime、Qwen GPU/runtime/冻结的代表 PNG：狐狸主体与雪景
可辨，Qwen 冻结图局部姿态/毛发不同；仅单 prompt/seed 的人工观察。
本地证据：`outputs/runtime-ane/artifact-lease-matched-2026-09-30/`。
旧库的 GPU/冻结结果**未**用作新库的分母；512²的冻结 base 图仍最快，
runtime 保持显式 optional，默认选路不变。先前 1024²、Q4/Q8、LoRA
和编辑结果仍归属各自构建/工作负载，不能借本轮 512²数字重新标记。

## 同库 Qwen 1024² 长序列、多 chunk 回归

另在新库 `28ff6aaf…` 做 BF16 base Qwen 1024²/40步、狐狸雪景/seed42，
所有路线统一开启 GPU Q/K norm-RoPE。runtime 使用 c1792/K1024/N512 v2
图与 auto chunks，冻结路线使用已有 4096-row W8A8 图；
GPU→runtime→冻结→冻结→runtime→GPU，六个独立 resident trial，
每 trial 一冷两热。driver 在每 trial 前后核对 CLI/库、两类 manifest、
runtime compiled graph 的逐文件摘要；screen 检查实际 backend、计时、
prediction、失败/回退、独立采样完成状态。

| GPU 四热中位 | runtime 四热中位 | 冻结四热中位 | GPU/runtime | GPU/冻结 | 冻结/runtime |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 183.165 s | 147.840 s | 154.664 s | 1.239× | 1.184× | 1.046× |

runtime 正反向 trial 的两热中位分别 148.311/147.372 s，每 trial
累计 3,648 次真实 prediction，均无失败或 fallback；冻结两 trial
分别 156.029/153.255 s，每 trial 3,744 次 prediction、无失败。
18 份 JSONL 请求结果与 18 张 1024² PNG 已复核；正反向各路线末次热图
在路线内部字节相同。人工查看代表图：GPU/runtime 的狐狸姿态、雪景和
细节非常接近，冻结图有局部毛发、枝叶差别；这仅是一个 prompt/seed，
不能充当更广画质或编辑/LoRA 的验收。

独立进程树采样合计 29,616 个样本，最大间隔 117.668 ms，六窗口均无
swap-in/out；正向冻结窗口的系统 compression 增量约 871 MB，另外
五窗口为零。该增量不能归因于冻结图或定量推断热请求受到的影响；
进程树也不覆盖所有 Core ML 外部服务和 driver/wired 内存。
原始命令、各请求/PNG/JSONL、采样流与 verifier 报告保留在被忽略目录
`outputs/runtime-ane/artifact-lease-qwen1024-matched/`，summary 为 complete。
此前 `f3f6…` 1024²匹配组的旧库三路结果仍归属旧库，不作为本表分母。
此新库 1024² base 中 runtime 最快，但收益幅度与内存资格不足以改变
默认选路或推广到 1024² 编辑和 LoRA。

尚未完成：物理 ANE 驻留证明、真实低内存与完整 driver/wired 资格、
Q4_1 全量模型及更广提示词/编辑/LoRA 验收。当前磁盘余量约 17 GiB，
本轮没有生成约 5 GiB 的 Q4_1 全量 GGUF。总体目标仍在进行中。
本机 `powermetrics` 列出 `ane_power` sampler，但要求 superuser；
非交互 `sudo -n` 返回“a password is required”，故没有采到 ANE rail
数据，也没有把 Core ML 的 compute-unit 策略冒充为物理执行证明。
