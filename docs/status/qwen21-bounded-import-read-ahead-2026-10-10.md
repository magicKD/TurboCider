# Qwen GGUF：有界读取 / CPU 解码流水线

2026-10-10，M4 Max64GB / macOS26.6.2。接续
[同内存四路与buffer筛选](qwen21-matched-gguf-bf16-2026-10-10.md)。
完整目标仍 active；此轮针对原encoder/DiT导入，不改GPU kernel、原量化
表示、step/层数或精度，不重复源SHA，不创建all-model raw/dense缓存。

## 实现与正确性

显式 `TURBOCIDER_QWEN21_GGUF_READ_AHEAD=1`，默认关闭。
`TURBOCIDER_QWEN21_GGUF_READ_BUFFER_MIB=1|4|8` 是**每个**buffer容量；
预读模式两份共享buffer、一个持久CPU pread worker、一项待处理job，
owner解码当前buffer时读取下一批到另一个buffer。数据不按8个decoder
线程复制。最多两个buffer，4MiB ×2总8MiB，与前轮单8MiB scratch相同。

小于一chunk的tensor直接沿用同步读取。BF16同宽原位转换直接写最终
已准入数组，next job写入不重叠的下一个row range；仍无第二份浮点
checkpoint。原Q4/Q5/Q6 K解析、typed coefficient、finite、shape/offset、
held-fd/named-path generation复验与分组重编码完全不变。

`ReadAheadWorker`只借fd、写owner提供的已准入span，不解析、hash、操作
MLX allocator或发布字典。job必须满足file range和固定buffer容量，限制
单项pending，owner/thread错误拒绝，failed worker不可继续提交。取消/
事件错误/decoder非有限值等路径都先stop/join读线程，再销毁candidate
数组、两份输入buffer或fd；成功也在发布前验证idle并drain线程。
原RowDecodePool每项解码在切换buffer前全部drain。nativeSHA/import任务
未完成时仍只有局部CPU candidates，不会交给GPU/ANE或返回verified bank。

ledger floor与物理准入计算**两份实际buffer**；若原SHA/import overlap
启用，再保守计入独立1MiB hash scratch。没有减少4GiB system reserve、
增加inactive信用或推广public/hard-RAM资格。

native metrics增加read_buffer_slots、read_ahead_enabled/drained/jobs/bytes、
read_wait_seconds。read_seconds是读worker实际span之和，decode_seconds
是owner真实decode spans，可能重叠，不能相加当完整wall或物理I/O证明。
同步policy plan digest保持原值；预读另绑定bounded-two-slot-pread-v1。

## 测试

CPU fixture130个chunk（两buffer切换和17byte尾部）、完全输出比较、
不重叠final-storage写入、pread offset不变、range/owner/backpressure、
read失败后poison、取消、128次queued/inflight销毁drain等全部通过。
相同fixture独立ASan/UBSan与TSan通过。

真实Metal bank fixture增加跨chunk的Q4/Q5/Q6与BF16矩阵，FP16/BF16
两种输出、1/3/8decoder并发，全部plane bits与serial相同；包含两buffer
budget floor、cancel、非有限值、source mutation、lazy escaped GPU reader、
并发完整SHA的候选publication/error/drain负例。原完整bank、ASan/UBSan、
TSan三轮均通过。64 typed encoder GPU prefill与原BF16 streamer数字/
last-reader回归同库通过，CPU receipt validator拒绝假slots/bytes/jobs/drain。
匹配库native-only/private-enabled/experimental0，513源码inputs全匹配。

## 完整原encoder：12次正反导入

已注册原5,027,785,568byte source，每次新native process，8workers，
packed embeddings、13个原token rows，无36层forward。顺序
1serial/1ahead/4serial/4ahead/8serial/8ahead与其倒序。每次真正消费原
4,511,322,112bytes（只排除未消费head），hash covers完整原file但本次
验证SHA读取0、native persistent hit1。12个独立process trees全部complete。

| MiB / schedule | bank wall s | actual read span s | decode span s | import+gather total s |
| --- | ---: | ---: | ---: | ---: |
| 1 serial | .649371 | .218091 | .409112 | .666087 |
| 1 ahead | .490464 | .221409 | .447886 | .505140 |
| 4 serial | .541408 | .192379 | .326958 | .556079 |
| 4 ahead | .412745 | .194987 | .346697 | .427639 |
| 8 serial | .521688 | .186510 | .312770 | .536356 |
| 8 ahead | .425619 | .194713 | .329256 | .440622 |

4ahead比相同4MiB serial的total少约23.10%，对默认1serial少35.80%；
比相同8MiB**总scratch**的8serial少20.27%。4ahead优于8ahead，较大chunk
会增加pipeline开头/尾部不能隐藏的等待；不默认认为buffer越大越快。
4ahead两个重复窗口实际1110jobs、4,320,755,712ahead bytes，wait约
.0337/.0339s，两buffer总8,388,608bytes，发布前drained；其余少量tiny
fields同步。decode span略变长，是CPU/带宽竞争，完整wall仍更短。

此处是CPU读/CPU解码任务安排的实测span，不是GPU/ANE或SSD物理overlap
证书。组件tree peak约4.72–5.22GB，preload和bin影响不可全归因scratch。
12次system compression/decompression/swap-in/out全0，尚不等价public
内存上限或全系统quiet-load长期保证。

## 同库4步完整预检

512² / seed29 / 三fresh fox-light prompts，GPU与原Private W8A8
ANE5120/GPU7168。prefill GPU1、原encoder retention1、8workers、source
注册跨进程复用。分别1serial、4serial、4ahead，保留完整36层encoder/
32层DiT，以4step做功能/精确性预检；没有用它代替40step性能。

相对1serial，4serial/4ahead两路48对完整text/initial/latents/pixels
safetensors与12对PNG逐字节相同。4ahead冷text GPU .755059s，mixed
.750300s；完整request6.160726/5.646201s。真实encoder import约.4275/
.4286s、DiT约.3736/.3700s，源SHA读取0、native persistent hit1、不重写
记录。两component各1110实际jobs，drain true、total read scratch8MiB，
CPU仍从原held fd读取packed payload，不把零SHA误说成零模型I/O。

## 同库40step独立进程窗口

同原source/512²/40steps/seed29/三fresh prompts，一cold两warm，source
注册已完成。比较默认1serial与4ahead（包含buffer增长和CPU读/解码安排，
不单独将全差异归因线程），另由上述同4MiB正反组件window隔离流水线项。

| route / import | cold encoder s | cold request s | warm encoder s | warm request s | tree physical peak bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU 1serial | .976725 | 45.910255 | .090494 | 44.425377 | 15,243,779,632 |
| GPU 4ahead | .771868 | 45.517030 | .092568 | 44.450152 | 14,909,546,128 |
| ANE5120/GPU7168 1serial | 1.200819 | 38.826064 | .092038 | 36.847581 | 15,426,674,440 |
| ANE5120/GPU7168 4ahead | .777683 | 38.183984 | .091418 | 36.843327 | 15,512,215,136 |

cold encoder分别少20.97%/35.24%，完整cold request少.393/.642s，即
.857%/1.654%。warm保持原kernel，其小幅变化不构成新增decode加速。
4ahead实际encoder import .432495/.439652s、DiT .374289/.375627s，两个
component各1110 jobs，待read约.036/.040s，native全源SHA bytes0、hit1、
write0、双buffer实际总8MiB、drain true。原codes/scales/biases保持。

四个进程树完整，4ahead两路system compression/swap-in/out0；GPU
decompression0，mixed16,384bytes。它们是系统计数，不赋予public/hard
RAM或物理overlap资格。40step六对完整PNG与同库control逐字节相同。
此独立窗口cold encoder均低于历史原GPU参考1.18s，但不能忽略此前
BF16 streaming高I/O后encoder2.02–2.18s的反例；故新库随后另测完整
四路same-memory/high-I/O顺序窗口，严格保留各自边界。

## 新同库高I/O与相近内存四路

相同注册/原模型/40step/三个fresh prompt，一cold两warm；顺序原BF16
stream prefix28/managed13GiB → GGUF GPU4ahead → mixed4ahead → 原BF16
resident。read/decoder新flags在两BF16 arms显式关闭，不改其原算术。

| route | cold encoder s | cold request s | warm encoder s | warm request s | tree peak physical bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原BF16 stream | 1.390849 | 45.442938 | 1.504719 | 45.532613 | 14,752,016,560 |
| GGUF GPU4ahead | 1.206573 | 46.015204 | .090683 | 44.442035 | 14,614,879,888 |
| GGUF mixed4ahead | .980560 | 38.484149 | .090318 | 36.879159 | 15,387,287,208 |
| 原BF16 resident | 2.275633 | 45.477146 | .584123 | 43.346015 | 36,965,244,424 |

stream/mixed峰值约.958714，实际约4.13%差，满足本窗口±10%条件。GPU
整请求比同内存stream少约2.40%，mixed少约19.01%、约1.235×。核心GPU
denoise仍约43.74s，mixed约36.19s，不把source导入改善冒称kernel提速。
此时GPU cold encoder较前窗口2.018s方向改善到1.207s，读等待约.381s、
import约.866s；它仍略高于历史参考1.18s，且参考文件缓存状态不完全匹配，
不能宣布普遍cold领先。mixed cold .981s也是当前实际结果，不隐藏较慢项。

四个进程树与全source/40step/层数/reader/drain/calls receipts完整，native
encoder/DiT source SHA读取0。原BF16 stream/resident三个完整PNG逐字节
相同。所有arm系统swap-in/out为0；stream compression6,455,296、
decompression81,920bytes，两GGUF arms这两项为0；resident compression
13,797,785,600、decompression13,528,891,392bytes。全部保留机器记录，
不归因单个矩阵或追认硬RAM/public/quiet-load资格。
完整目标继续active：下一步仍需更严格匹配文件缓存状态的原GPU参考
cold/encoder验证，及更广编辑/LoRA/场景质量，不以此window代替全部范围。

[机器记录](../design/validation/qwen21-bounded-import-read-ahead-20261010.json)
绑定原模型/库、12次导入、三路4step、两路40step、最新完整四路、CPU/
Metal/sanitizer/负例回归、source/drain/实际scratch与完整输出精确性。
