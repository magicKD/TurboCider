# Qwen Q4_K_M：完整校验与 CPU 导入重叠

2026-10-10，M4 Max 64GB / macOS26.6.2。接续
[GPU encoder prefill](qwen21-gpu-encoder-prefill-2026-10-10.md)。
本文保留当时每个 fresh process 全 SHA 的实测合同；用户之后要求仅首次
校验，现有默认行为见[首次注册与跨进程复用](qwen21-source-registration-2026-10-10.md)。
完整目标仍 active；此轮减少冷准备开销，不删除原完整 SHA、不引入
序列化 proof/dense checkpoint cache，也不将 warm 或导入组件冒称整个
cold encoder 已胜过 Unsloth。

## 排除无效方向的实际证据

原 5,027,785,568byte encoder，原 held-fd whole SHA，native / 1MiB / 8MiB
buffer、default / user-initiated QoS、cached / F_NOCACHE，正反顺序14次。
每次完整读取原文件，SHA相同，独立进程树全部complete。
读取约 .177–.185s、CommonCrypto纯hash约1.452s，native整体1.63–1.66s；
thread CPU几乎等于wall。更大buffer、QoS和cache flag都没有实质收益。
原机已安装OpenSSL3.6.3的SHA吞吐约3.43GB/s，也没有更快；没有为此
给生产运行时增加OpenSSL依赖或新crypto实现。

## 实现与边界

新增 `PendingSourceVerification`：先捕获同一组held fds，独立native
worker完整SHA；metadata()返回不可变、**未verified**的lease，只有
finish()在worker drain、SHA/expectation、named-path/held-fd复验成功后
返回verified lease。未完成/停止状态不能当proof，重复native进程proof
仍只按原generation-bound cache复用，普通metadata/serialized descriptor
仍被旧bank constructor拒绝。

新private候选bank入口只允许这项native任务；owner做原CPU mixed-K
导入，线程仅读fd/hash，不操作MLX allocator。原codes/scales/biases、
dtype、shape、worker schedule与最终plan digest保持。候选数组只在本地，
join与全部source/finite/cancel检查前不发布、不交GPU/ANE消费；失败/
取消/源变更/准入拒绝会drain worker，释放所有候选claims。
hash1MiB + importer1MiB，额外scratch有界，物理准入计入两者；没有
增加每decode worker的payload buffer，没有inactive-cache信用。
这是CPU校验与CPU导入的并发安排，不是GPU/ANE或物理I/O overlap证明。
当前仍两次逻辑读取；真正single-pass verify/import未实现，不能冒称一次读。

GGUF目录reader增加固定4KiB窗口，减少token/merge数组短字段的pread；
不保留词表内容/完整directory，原directory/rank/retained/string/array/
file-range/overlap检查不放宽。旧模型、其他GGUF调用也继续这些检查。

显式 `TURBOCIDER_QWEN21_GGUF_VERIFY_IMPORT_OVERLAP=1` 启用，默认关闭。
推荐实验保留前轮 GPU encoder prefill / original encoder retention /
8 import workers，DiT可保持GPU或原Private W8A8 ANE5120/GPU7168。
新scope不提升Public、hard-RAM、物理并行或GGUF editing/LoRA资格。

## 完整原 encoder 导入：新同库六次交错

每次fresh native process，8workers、packed embedding、原文件full SHA，
仅过滤未消费LM head，但hash覆盖head在内的全部原文件。三次serial/
三次parallel，包含13个真实embedding rows，不包含36层encoder forward。
directory4KiB改动在两个新同库臂都启用，另与旧v7数据分开说明。

| route | native verification s | bank decode s | component total s |
| --- | ---: | ---: | ---: |
| serial SHA→import | 1.630851 | .408395 | 2.291907 |
| parallel SHA/import | 1.745247 | .433717 | 1.756280 |

component耗时少约23.37%，约1.305×。并行存在CPU/内存竞争，hash/decode
各自略慢，但完整span更短；不能把两个overlapping spans相加造新倍率。
完整SHA仍读取5,027,785,568bytes，每次cache-hit0；完整process trees验真。
bitwise field、final plan、source guard和lazy GPU reader生命周期单元通过。

## 完整请求与质量

最终同库4步先行预检，三个fresh fox-light prompts，GPU与Private混合
两路；双方encoder-prefill1、retention1、workers8，只有verify/import
schedule不同。cold text GPU 2.608809→2.091539s，混合2.599533→2.073378s。
24对完整text/initial/latents/pixels tensor、六对PNG全部SHA相同，数值
误差0；不是以宽松delta替代原导入精确合同。
该4步窗口不代替40步性能或泛化语义质量结论。

最终v1库/CLI同模型、512²/40steps/seed29，一cold两fresh warm，全部
conditioning miss；serial GPU→serial mixed，然后parallel mixed→parallel
GPU。四个完整进程树和每个cold的原encoder/DiT完整SHA receipt均验真。

| route | cold text s | cold request s | warm text s | warm request s | tree peak physical footprint bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| serial GPU | 2.613939 | 48.937583 | .090557 | 44.435205 | 14,918,262,392 |
| parallel GPU | 2.155164 | 48.066836 | .093086 | 44.446411 | 15,098,895,944 |
| serial ANE5120/GPU7168 | 2.664106 | 41.695132 | .098149 | 36.855662 | 15,387,647,584 |
| parallel ANE5120/GPU7168 | 2.105721 | 40.643064 | .092439 | 36.865824 | 15,537,020,512 |

cold text分别少17.55% / 20.96%，cold总请求少1.78% / 2.52%，约
.871 / 1.052s。warm基本不变，不制造新的DiT kernel倍率。目录改动
在两臂相同；旧v7 GPU prefill cold2.841643s与当前2.155164s方向改善约
24.16%，但不是同窗口单项因果收益，不替代上面同库的17.55%。

parallel cold实际完整encoder verification5,027,785,568bytes，DiT
4,199,565,024bytes，cache-hit均0。GPU臂实际两段source准备wall约
1.81381/1.54124s，混合约1.77291/1.49112s；各自包含完整hash+候选import，
不从request计时扣除。新两路warm DiT约43.744362/36.174074s，与原路径
基本不变；混合每request1280成功Private calls，phase32/1248，failure/
fallback/retry0。相比Unsloth上一有效GPUcold1.17s，native仍未胜出。

40步六对完整PNG全部SHA相同；4步24对全tensor同样相同。即与同库
serial control完全相同，不需要额外放大delta容忍度，也不将它推广为
与原BF16/Unsloth图片等价。510个native source inputs与最终库逐一匹配。

新GPU/混合过程peak比serial约增加1.21%/.97%，仍不能当硬RAM cap。
system compression/swap-out均0，serial mixed swap-in65,536bytes；其余
swap-in0。serial GPU/serial mixed/parallel mixed/parallel GPU system
decompression为0/65,536/16,384/3,358,720bytes；这些是系统计数，不能
归因单矩阵，也不追认quiet-load资格。

6项实Metal/native regression（含原prefill64病例、compiled48病例、
mixed-K bank新并发proof/publication/取消/源变更/非有限值/准入/escaped
reader负例与BF16双阶段streaming）通过。SourceLease原fixture及新增
async proof、错误expected digest、metadata不升级、cache复用、停止与
源变更测试通过；独立ASan/UBSan和TSan两轮同样通过。最终18项CPU
host/parser/receipt/质量/timeout tests、real prepare session10生命周期
requests和独立PyTorch conditioning10场景通过，无skip。

[机器记录](../design/validation/qwen21-gguf-verified-import-overlap-20261010.json)
绑定完整源/库、14次hash诊断、六次导入、四个完整screen、tensor/PNG
精确性、memory与sanitizer/回归/原失败日志。

## 保留失败与后续

SourceLease新增源变更负例初次只接受`changed`字样；实际合法拒绝也会
报告source size/read等原因，fixture修正为必须有source相关拒绝，不改
runtime门禁。原failure log保留。QoS/大buffer/替代crypto无收益同样保留。
剩余关键点是更快的完整cold路径、GPU tiled group32/direct GGUF消费、
有界ahead-decode、更多scene/seed与编辑/LoRA，以及真实匹配的Unsloth
完整cold/warm生命周期。源校验成本不能从cold数字偷偷扣除。
