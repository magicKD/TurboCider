# Qwen BF16：有效完整框架对照

更新2026-10-12，M4 Max64GiB/macOS26.6.2。原BF16 encoder/DiT/VAE、512²、
40步、seed29、三个fox-light prompts，无LoRA。最新有效v11为十进程
五路线回文顺序；每新进程cold、两fresh、一次repeat，constructor另列。
全部40张PNG和十条独立process-tree memory证据完整；不删除慢样本。

## 同窗口完整生成

| 路线 | cold 正/反 s | 四fresh中位 s | repeat 正/反 s | tree peak GB（十进制） |
| --- | --- | ---: | --- | --- |
| TurboCider GPU | 47.444 /45.289 | 43.238745 | 42.628 /42.620 | 30.65–30.78 |
| TurboCider GPU/Core ML frozen W6144 | 38.798 /48.480 | 31.058807 | 30.064 /30.375 | 35.73–36.01 |
| mflux BF16 | 51.721 /50.509 | 48.511753 | 46.267 /46.183 | 50.91–51.01 |
| mflux affine Q4 | 57.595 /57.313 | 56.335344 | 53.802 /51.945 | 51.24–51.32 |
| mflux affine Q8 | 57.556 /57.655 | 55.100963 | 55.138 /52.124 | 51.42–51.43 |

相对mflux BF16，原生GPU为1.122×、少10.870%；frozen为1.562×、少
35.977%。frozen相对自身GPU少28.169%。这是完整框架路径比较，不是
等组件生命周期的纯kernel倍率：原生encoder为GPU/request-local，
mflux保留组件；混合是部分W8A8近似，而非严格BF16同精度。
mflux Q4/Q8按stock定义保留BF16 text encoder，不等价于原GGUF
encoder与DiT同时Q4_K_M。量化在此窗口没有带来更低延迟。

## 参考cache=0：更低压力的独立完整对照

process-local MLX free-cache limit=0，原模型/数学/40步不变、不改OS策略。
六进程GPU→frozen→mflux→mflux→frozen→GPU，24PNG/6memory完整，
四fresh/route。它是单独调优候选，不与上方stock默认样本混池。

| 路线 | 四fresh中位 s | cold 正/反 s | tree peak GB（十进制） |
| --- | ---: | --- | --- |
| TurboCider GPU | 43.347453 | 47.661 /45.526 | 29.75–30.78 |
| TurboCider frozen | 31.250897 | 39.152 /42.116 | 35.58–36.17 |
| mflux BF16/cache0 | 45.968219 | 49.371 /47.984 | 32.92–33.84 |

对该窗口参考，GPU为1.060×、少5.701%；frozen为1.471×、少32.016%。
frozen对自己的GPU少27.906%。参考pressure大幅下降：compression累计
2.65–3.36GB，swap-out0；不是完全无pressure。相比stock默认约51GB
peak、48.512s，候选更省内存且更快，但两独立cohort不构成严格单因素
同窗口A/B因果证明，也不把43.239/45.968跨轮拼倍率。

[cache0独立便携记录](../design/validation/qwen21-cache0-framework-current-20261011.json)
保留实际cache bytes/previous limit、全部样本、raw/PNG/内存链和输出
差异。这使“纯GPU大幅领先参考”的结论更保守：对已调低缓存参考，
GPU只有约6%倍率优势，GPU/ANE约47%；两者cold/setup另列。

## 新增1GiB/4GiB缓存：参考约45.45s，混合约32.4s

2026-10-12两组独立六进程回文完成，任务/模型/40步不变，每组24PNG/
6个完整memory窗口、四fresh/route。实际process-local cache bytes分别
核验为1073741824/4294967296；没有修改OS缓存、wired/swap设置或参考源码。

| 参考cache / 路线 | 四fresh中位s | cold正/反s | repeat正/反s | tree peak GB（十进制） |
| --- | ---: | --- | --- | --- |
| 1024MiB / GPU | 43.377048 | 47.485 /45.900 | 42.677 /42.739 | 30.62–30.78 |
| 1024MiB / frozen | 32.393132 | 47.220 /48.784 | 30.346 /30.280 | 36.16–36.16 |
| 1024MiB / mflux BF16 | 45.450516 | 48.682 /47.992 | 45.309 /45.233 | 34.48–34.88 |
| 4096MiB / GPU | 43.330965 | 44.774 /45.881 | 42.638 /42.671 | 29.84–30.21 |
| 4096MiB / frozen | 32.406713 | 45.864 /47.580 | 30.951 /30.246 | 35.80–36.16 |
| 4096MiB / mflux BF16 | 45.497862 | 48.665 /48.276 | 45.245 /45.229 | 37.71–37.76 |

1GiB同窗口GPU/混合对参考为1.048×/1.403×，混合对自身GPU省25.321%；
4GiB为1.050×/1.404×。两个参考中位只差0.047346s，约0.104%，不足以
证明缓存容量的速度因果优劣。两组各自与controls比较，不混池，不把
前一cache0的31.251s与本组45.451s拼出更高倍率。新mixed约32.4s也
不能据独立cohort归因为缓存造成回退；native没有改变参考cache设置。

当前保存1GiB为较低观测内存的reference候选：相对4GiB约少3GB，
不是全局最佳证明；cache0更低peak33–34GB，但其独立中位45.968s。
所有新窗口swap-out0；native GPU compression0，frozen累计5.05–5.78GB、
参考4.27–5.31GB，不能称全窗口无压力。frozen冷请求45.86–48.78s，
远慢于自己的fresh，单次启动需要把setup算上。

frozen仍是32个W6144图、每请求1248runtime+首次32warmup calls、failure0，
首步GPU而不是全部交ANE，后续image-only FFN channel split6144/6144。
两组完整原尺寸RGBA/白背景RGB差异均保留，warm-light相对GPU仍为
SSIM0.958431、MAE4.609/255；跨框架同seed不是同noise或BF16逐像素parity。
[1GiB完整验收](../design/validation/qwen21-cache1024-framework-current-20261012.json)、
[4GiB完整验收](../design/validation/qwen21-cache4096-framework-current-20261012.json)
绑定actual limits、所有cold/fresh/repeat、PNG/内存SHA与514个native输入。

参考固定commit `8c00dab2505a96019df9d30bc9c223bf20d733c4`（2026-09-20），
包版本字段0.19.2；这是含Qwen2.1支持的固定checkout，不是官方当前所有
版本的排名。更新框架版本的同机完整对照仍pending。当前任务为
image.generate；不把其倍率冒充Qwen Image Edit首步无KV-cache收益。

## 内存压力是结果的一部分

mflux各窗口system compression累计增量约81–121GB，一个BF16窗口
swap-out4.263GB；累计活动不是当时占用，也不是单进程独占。
native GPU/frozen的compression/swap-out增量均0，少量swap-in/decompression
仍保留。mflux BF16 fresh57.285204s慢样本保留在中位数中，不能因压力
删掉它。process-tree physical peak含load+cold+warm+exit，不归因所有
外部Core ML服务与driver，也不是硬RAM容量资格。

因此目前可以说本机stock完整路径原生更快、更省观测内存，不能把全部
收益归因Metal数学。另起同窗口cache-limit候选以核验参考内存策略，
不改OS/wired limit、不关闭用户app，不与stock样本混池。

## 实际GPU/ANE执行与输出差异

32个checkpoint-bound W6144/r1024 source graphs与compiled manifest
已经恢复到独立benchmark cache，当前实际可加载。每40步请求1248次
runtime，首次32次warmup，failure0，checkpoint SHA通过；首步GPU，
后续image-only FFN为ANE6144/GPU6144分工。Public Core ML配置不等于
Private/native INT8 MAC或硬件并行occupancy trace。

warm-light GPU/frozen的狐狸姿态、雪景/构图接近，局部毛发有差异：
RGB correlation0.993641、Gaussian11 SSIM0.958431、MAE4.609/255；
alpha MAE0.05455/255。同一prompt的native/mflux BF16 correlation0.863236、
SSIM0.799399、MAE22.614/255，构图有差异，不能宣称跨engine画质parity。
原生RGBA、参考RGB的opaque alpha与白背景composited RGB分别报告，
不悄悄丢掉透明度。单seed/有限prompt不证明广泛感知资格。

## 之前的无效参考仍保留，不作为速度分母

原Comfy DiT有32个img_mlp.gate_up.weight，mflux0.19.2要求分别为
gate_layer.weight与proj.weight。v1派生布局遗漏拆分，64个required
linears留在random initialization，图片为彩色噪声。即使步数、完整PNG
和memory通过，也不是原模型正确执行。

| 原错误参考路线 | 诊断fresh中位 s | 有效排名 |
| --- | ---: | --- |
| mflux BF16，v1未拆分FFN | 56.879010 | 否，彩色噪声 |
| mflux affine Q4，v1 | 60.698671 | 否 |
| mflux affine Q8，v1 | 61.443489 | 否 |

修复的是本仓库无损加载布局：连续gate/up byte ranges拆成required键，
原BF16 payload不变；参考源码、权重值与步数未改。v2 required coverage
297 DiT/398 encoder/226 VAE完整。旧v6原生有效43.249697/31.007695s
单独保留，不与v11混池；v9采样gap579.013ms失败也没有追认为成功。
v11所有路线统一bounded async证据writer，仍每sample fsync、64record
上限、结束drain，原500ms最大gap和独立验收门槛未放宽。

## 证据和复现

[最新有效完整便携记录](../design/validation/qwen21-valid-framework-current-20261011.json)
保留514个native source输入身份、模型/库/只读mflux源码身份、四fresh
原始数组、cold/repeat、每请求执行/内存/PNG与完整RGBA差异。raw40张
PNG、模型与compiled artifacts不进Git。使用mflux0.19.2、commit
8c00dab2505a96019df9d30bc9c223bf20d733c4，不是未来版本排名。

入口为tools/validation/qwen21_mflux_framework_screen.py，使用v2完整
无损布局、build/benchmark-async-v1-private与512²/40步/4request：
native-gpu,native-frozen,mflux-bf16,mflux-q4,mflux-q8,mflux-q8,mflux-q4,mflux-bf16,native-frozen,native-gpu。
初次源SHA验真后复用generation-bound注册，不在每次inference扫描模型。

[旧原生结果/invalid参考标记](../design/validation/qwen21-native-cohort-current-20261011.json)
和失败raw仍保留。[本机最佳已测配置](best-local-configs-20261011.json)保留
当前request/manifest与恢复入口，不自动改变生产默认。扩大benchmark
仍active，额外参数/尺寸/编辑与LTX/FLUX缺口不因此标完成。
