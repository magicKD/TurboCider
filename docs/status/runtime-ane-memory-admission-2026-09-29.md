# Runtime ANE：请求级内存准入与 host scratch 扩容

2026-09-29，接续[内存准入只读审查](runtime-ane-memory-admission-audit.md)。
这是显式 runtime-weight 路线的机会性保护，不是物理 ANE residency、driver
内存上界、系统压力无回退或低内存机型的资格认证。普通 GPU、冻结 base 图、
LoRA 数学及默认选路未改；参考 vpipe 的 worker/host memory 生命周期，
但没有照搬它按一个 chunk 计算 host 的公式：本实现的 output/hidden
scratch 随实际 ANE rows 增长。

## 代码与生命周期

`native/backends/ane_memory.{hpp,cpp}` 在 owner thread 采 Mach
`phys_footprint`、`hw.memsize`、原始 `free_count` 和 `inactive_count`。
macOS SDK 指出 free 已包含 speculative，故不重复加 speculative；最多
折算一半 inactive、上限 8 GiB，不加 purgeable。保留 4 GiB 系统空间、
原有最多 2 GiB 的可选 tier 额度，并比较进程 footprint、MLX active、
图估算、已留 host scratch 与新 payload。未知观测/溢出保守拒绝。
这是启发式余量，不意味着 inactive 可以无代价全量回收。

`RuntimeGraph` 验证 artifact/几何和预算之后、分配 IOSurface/加载 Core ML
之前准入；非法 manifest 仍报错而非误作 GPU 回退。`HybridFfn` 在每个
resident 请求的安全点重查；扩容前按实际 chunks 计算两个 host buffer
需要的新 payload，旧 buffer 在新 buffer 分配时仍计入已有额度。
使用精确容量的替换 vector 避免 `resize` 的几何扩容超出预检；内存分配
失败时 drain、释放可选图与 scratch，GPU 完整重算，不发布部分 ANE 行。
模型下个请求会尝试重新构造 optional tier，不在 worker callback 释放自身。
拒绝原因记录 free/inactive/process MiB 供诊断。

失败后 `runtime_failed` 标记为真，真实性能 screen 会拒绝把 GPU 回退伪装
成持续 runtime 收益；输出正确性仍由完整 GPU FFN 保证。已有导入/实际
Core ML、独立 MLX 临时/缓存、模型 correction 和 GPU 输出副本未获得精确
预算；没有定时压力线程或虚构的 driver/wired 计量。

纯 host 测试检查未知观测、预算边界、页数与矩阵尺寸溢出、inactive 折算、
三 chunk 扩容时新/旧 payload 共存，以及 adapter→base 的容量保留。
Core ML/MLX 集成注入 resident 低余量，不占满共享机器 RAM：第一请求
成功用图，下一请求 drain/释放，GPU 结果与完整 GPU 参考相等。

## 先前过严门槛与修正

最初只使用 raw free，Z-Image 冷请求成功后，两次 resident 热请求
得到 `system_reserve` 回退；`runtime_ane_model_screen.py` 正确将这次
筛选标为 incomplete。保留证据在忽略目录
`outputs/runtime-ane/z-base-memory-admission-2026-09-29/`，不把那两次
GPU 结果当成 ANE 性能。host 实测 free ~23.4 GiB、inactive ~27.4 GiB
是另一个时点，不是失败瞬间的系统压力证明；因此增加有限的 inactive
折算，并保留 on-denial 数值，修正后再实际复测。

## 本机整模型验证

M4 Max 64 GB、BF16 base，狐狸雪景/seed42、resident；512²同构建三路顺序
runtime→GPU→冻结图，每路一冷一热。1024²先单测 runtime 一冷一热，
之后再做同构建三路匹配筛选（见下）。
Qwen 三路都开启相同的可选 GPU Q/K norm-RoPE；runtime 图 512² c320、
1024² c1792/K1024/N512；Z 图 c352/K1024/N512。所有数据为整请求
`request_wall`（含 VAE/PNG，不含加载；冷请求不计热值）。原生库 SHA256：
`f3f6b6a05ce6eecb476f314f1fa1fccbdb1d2ef24fc09fdc5b72049b0390ca22`。

| 工作负载 | GPU 热请求 | runtime 热请求 | 冻结热请求 | GPU/runtime |
| --- | ---: | ---: | ---: | ---: |
| Z 512²、8 步 | 6.991 s | 6.328 s | 5.354 s | 1.105× |
| Qwen 512²、40 步 | 41.753 s | 35.609 s | 29.072 s | 1.173× |
| Qwen 1024²、40 步 | 183.073 s | 148.085 s | 158.207 s | 1.236× |

512² runtime 累计 Z 451、Qwen 2432 次 prediction，1024² Qwen 单测 2432
次；这些 trial 均无 runtime 失败或 GPU 错误回退。原组 trial 的进程树
采样 gap < 112 ms，系统 swap-in/out 均为零；Qwen 512² 冻结图在采样
窗口内有约 224 MB compression，其余本轮 trial 为零。采样不是 ANE
独占内存归因，也不证明长期无换页或低内存设备可用。

原始报告、请求、stdout、PNG 和独立内存流留在被忽略目录：
`outputs/runtime-ane/{z-base-memory-admission-final-2026-09-29,`
`qwen-base-memory-admission-final-512-2026-09-29,`
`qwen-base-memory-admission-final-1024-2026-09-29}/`。
上述 1024²单测热值是 148.582 s；表内 1024²的三路热值均来自之后
的匹配组，不混作原单测样本。
已看两模型 512² GPU/runtime 代表图：主体、姿态、雪景构图与色调接近，
局部纹理有差异；没有更多 prompt/训练 LoRA 的视觉资格。单热样本只作
新构建无明显回归的筛选，不替换原有多 trial 正反向性能表。

此前仅折算 inactive 的候选库 `74ee41ce…` 有两热样本三路 512²筛选和
Qwen 1024²一热样本，保留在同目录带 `-inactive-`/`-512-`/`-1024-`
的报告；它们不重标成最终库的结果。最初 free-only 失败、二次候选与
最终库分别留档，不拼接加速比。

### 同构建 Qwen 1024² 匹配筛选

另以最终库顺序执行 GPU→runtime c1792→冻结 W8A8；每路独立 resident
进程，一冷一热，狐狸雪景/seed42、BF16 base/40 步、同样的 GPU Q/K
norm-RoPE。三份原始 JSONL 和请求核对了 1024²、模型、步数、无 LoRA、
route 与图路径；库前后 SHA 保持 `f3f6b6a0…`。每路两次请求均成功，
整组 summary complete，三个进程返回 0，六张 PNG 均存在。

| 路线 | 冷请求 | 热请求 | 热 denoise | 峰值进程树 footprint |
| --- | ---: | ---: | ---: | ---: |
| GPU | 186.171 s | 183.073 s | 181.012 s | 31.120 GB |
| runtime c1792 | 152.354 s | 148.085 s | 146.020 s | 30.915 GB |
| 冻结 W8A8 | 173.456 s | 158.207 s | 156.147 s | 39.305 GB |

runtime 相对同组 GPU 为 1.236×，相对冻结为 1.068×；冻结相对 GPU
为 1.157×。runtime 共 2,432 次 prediction，hybrid blocks 为 2,432，
无 runtime failure、fallback 或 overflow retry，热请求复用图。
冻结图累计 2,528 次 calls，失败数为零。这一次顺序筛选与先前
`e55b1a…` 的六 trial c320/冻结对照、另一次 c1792/c320 交错对照不是
同一构建或采样方案；不可拼接样本或据一次反超宣布长期稳定优势。

独立重验三个 100 ms 进程树采样流，共 10,023 个样本，最大间隙
119.797 ms；三个窗口均无 swap-in/out。GPU/runtime 窗口没有新增压缩，
冻结窗口系统 compression/decompression 约 3.511/3.335 GB；计数不能
归因于 Core ML 或冻结图。采样在冷 PNG 写出时已达到最终压缩计数，
热请求未新增压缩；仍不能证明该路热计时不受当时内存状态影响。
峰值是加载+冷/热+退出窗口的进程树数值，不含外部服务或 driver/wired。
runtime/冻结 `observed_ane_residency` 仍未由硬件测量证实。

完整本地证据在忽略目录
`outputs/runtime-ane/qwen-base-memory-admission-matched-1024-v2-2026-09-29/`；
这组仍只有每路一个热样本，不替代多次交错和更广质量验证。
已实际打开三张末次热 PNG：均为雪中坐姿红狐，位置、姿态、松枝背景及
整体色调接近；runtime/GPU 局部毛发和雪面纹理有微差，冻结图毛发及
枝叶细节差别更明显。三张 PNG 的 SHA 不同；肉眼比较不是逐像素相等
或多提示词质量认证。

### 同构建 1024² 正反向重复对照

单热筛选结束后，不修改库/图/模型/工具，另以 GPU→runtime→冻结→
冻结→runtime→GPU 做六个独立 resident trial，每 trial 一冷两热，共
18 个请求。三路均开启 Q/K norm-RoPE、同一 prompt/seed42、BF16 base
和 40 步；runtime 使用 c1792/K1024/N512 v2 图、auto chunks，冻结
使用相同 4096-row W8A8 图。运行库前后 SHA 仍是 `f3f6b6a0…`；
`summary.json` complete。18 份原始结果和请求重新检查模型/尺寸/步数/
seed、无 LoRA、执行路线及图身份，18 张 PNG 均存在。

| trial / 路线 | 冷请求（s） | 热 1（s） | 热 2（s） |
| --- | ---: | ---: | ---: |
| 0 / GPU | 187.594 | 183.157 | 183.156 |
| 1 / runtime | 152.115 | 147.071 | 147.927 |
| 2 / 冻结 | 167.058 | 152.033 | 151.899 |
| 3 / 冻结 | 162.619 | 151.743 | 160.199 |
| 4 / runtime | 153.037 | 146.891 | 147.787 |
| 5 / GPU | 185.368 | 183.122 | 183.136 |

各路池化四个热请求的中位数：GPU **183.146 s**、runtime **147.429 s**、
冻结 **151.966 s**。同组 GPU/runtime 为 **1.242×**，GPU/冻结为
**1.205×**，冻结/runtime 为 **1.031×**。runtime 四个热请求全部短于
冻结的四个；反向冻结的 160.199 s 不剔除，样本仍少，不声称统计
显著性或全模型/设备普适。两个 runtime trial 各累计 3,648 次
prediction、3,648 个 hybrid block 和 192 个正常完整 GPU probe；
都无 runtime failure、错误回退或 overflow retry。冻结两路各累计
3,744 次 runtime prediction，失败数为零；普通 GPU 无 Core ML 调用。

六份 100 ms 独立采样流重新运行 verifier，合计 **29,478** 个样本，
最大间隙 **110.968 ms**，六路均无新增 swap-in/out。两个 runtime 和
两个 GPU 窗口无新增压缩；冻结正反向窗口系统 compression 分别为
1.478 GB 和 0.204 GB。这两个增量在各自冷请求 PNG 写出时已经达到
窗口终值，热请求期间没有新压缩计数；PNG 时间只能粗略定位请求边界，
不能将系统压缩归因于图或证明热请求完全不受其他内存状态影响。
进程树 footprint 峰值 GPU 31.207–31.625 GB、runtime
30.915–32.046 GB、冻结 39.722–40.182 GB；窗口包含加载/冷/热/退出，
不覆盖外部服务或 driver/wired。

已打开反向三路末次热图：红狐、坐姿、松枝与雪景构图接近，runtime 与
GPU 细节相近，冻结图毛发/枝叶有局部差异，无明显缺主体或黑图。
两次 trial 对应路线的末次热 PNG 各自字节相同，不同路线之间不是逐像素
相同；仅此一个 prompt/seed。完整被忽略证据目录：
`outputs/runtime-ane/qwen-base-memory-admission-1024-reverse-v2-2026-09-29/`。
首次沙箱运行因 `ps` 无权限，仅留下另一目录中的 incomplete summary；
没有把它算成 trial 或删除。完整多轮对照加强了 1024² base 这条显式
optional 路线的本机证据，不改变 512² 最快冻结图或默认选路。
对照完成后重跑 `make test-acceleration-contract`：沙箱内仅因 `ps`
权限使超时子进程用例失败，获权限后原命令退出 0（5 host、10 placement、
42 screen、6 memory screen、17 switch、12 repository 检查）。
本轮只改状态文档，未重建或改写计时绑定的原生库/图。

最终构建通过 `TURBOCIDER_NATIVE_ONLY=1 make build`、5项 host、9项
Core ML/MLX 集成、Qwen 专项 38+7+30 项与完整 `make test`。首次测试
在沙箱内因 Metal/Core ML 临时目录或 `ps` 权限失败，获得权限后重跑
exit 0；缺夹具、专用 audit/test-hook 和 GPU opt-in 跳过项不算覆盖。

## 尚未证明

- 真实低内存设备/压力事件下停止并恢复、压缩/换页和完整 driver/wired
  生命周期；瞬时 `free+inactive/2` 是采样启发式，不能替代压力状态。
- MLX activation correction、拥有所有权输出、Core ML/驱动内部的精确
  峰值，以及系统其他进程并发变化。使用一个图的估算不能认证整个请求。
- Qwen 1024² 同最终构建已有单热及六 trial 正反向对照，但仍仅一个
  prompt/seed、本机正常余量，冻结窗口有压缩；尚缺编辑/LoRA、Q4_1
  广覆盖、物理 ANE residency 与多提示词质量。

完整目标继续保持未完成；不因这次保护和正常内存筛选升级默认路线。
