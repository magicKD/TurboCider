# Qwen首步并行＋W缓存：真命中，但反序未复现额外收益

2026-10-10，Asia/Singapore，M4 Max64GB/macOS26.6.2。接续
[首步逐层选路](local512-qwen-prefill-layers-2026-10-09.md)与
[native surface](local512-native-weight-surfaces-2026-10-08.md)。此前cache
测的是六步全混合，本轮改在已较快的“首步GPU/ANE FFN通道分工、后五步
完整GPU”上公平筛选。不下载、改写原模型/adapter/ref，不导出新图；
完整Z/Qwen base/LoRA、编辑/encoder、GPU kernels/GGUF目标仍active。

## 相同整请求，分别比较cache off / copy / surface

同一保留generation-phases-v1 Private CLI/库，原BF16 Qwen Image2.1、
Viggle v0.2.1 r256/strength1、512²/6steps/seed29，227真实bindings。
三个fresh prompts只改natural/gentle/golden sunlight，conditioning全miss。
所有臂joint BF16 A/B、F32 ranks，GPU encoder且相同原source retention。
Private Fa5120/Fg7168、shared gate/up ranks、F32 partial join、fixed async1，
prefetch/lookahead/down-rank split/time reuse关闭。全部32层首步并行，不
把整个首步交ANE；GPU control仍完整lazy/compiled block。

单图实际首步2096 rows、已有c2112/eager；双图3144、已有c3168/deferred。
原第二张blue RGBA PNG为1024²，native显式reference_size512，两图有效
tokens各1024；没有重写原参考图。后五步都是1024 rows完整GPU，无cache/
staging/padding税。每臂独立process、一冷两warm，保留全部样本/PNG。

工具 `qwen_ffn_phase_screen.py --prefill-code-cache-bytes <budget> --joint-ab`
新增显式GPU/prefill/copy/surface四臂，限定原LoRA编辑，不与generation/
frozen/layer policy混用。Env/identity继续用原native机制，没有新native
math或source签名绕过。可选subset仍要求GPU；deferred对所有hybrid统一。
shared validator默认仍要求cold actual reuse，新prefill contract明确允许
cold fills/hits0，并要求warm实际进度，不伪造cached-call labels。

## 相反顺序、同workload复测

v2四臂surface→copy→off→GPU；v3三个共用臂GPU→off→surface，是同一
workload三个共有route的相反顺序。copy仅v2，不称其有同任务反序资格。
单/双图分别有自己的GPU/off分母，不互作ABBA；全部42有效native请求。

| workload/version | GPU warm request s | prefill off s | copy s | surface s |
| --- | ---: | ---: | ---: | ---: |
| 单ref / v2 | 10.542005 | 10.295422 | 10.367275 | 10.229890 |
| 单ref / v3 | 10.540892 | 10.238559 | — | 10.306506 |
| 双ref / v2 | 12.712212 | 12.203540 | 12.153714 | 12.100803 |
| 双ref / v3 | 12.700983 | 12.197060 | — | 12.224431 |

surface对off的request名义下降：单图v2+.637% / v3−.664%，双图+.842% /
−.224%，符号均翻转。缓存实际多保留1.0–1.3GB，却没有可重复整请求
收益，因此推荐仍cache off，不加新“最快”预设或强行平均出正结果。

off的首步/后五步：单图v2 2.270469/6.093229、v3 2.254429/6.092869s；
GPU首步2.428093/2.426038s。双图off为3.305759/6.368033、
3.299554/6.350924s，GPU首步3.744024/3.748533s。off整请求对同窗口GPU
单图名义少2.34–2.87%、双图3.97–4.00%，不是本轮新cache带来的收益。
surface首步单图2.202999/2.270778，双图3.286216/3.309378s；后五步
无显著新赢家。不能拼各臂最快phase成为未实际执行的组合request。

四个enclosing CPU load checks都失败；这些始终busy diagnostic，
qualification=false，不称设备独占、稳定倍率或物理GPU/ANE overlap。
两warm单图不同prompt request差约.68–.85s，远大于cache几十ms差值，
不把stage差额全部唯一归因给copy、binding或某个kernel。

## 准入失败与真实cache工作

单图预算1280MiB；双图最初同预算在首层被原scratch准入拒绝，完整GPU
重算（prefill/decode actual calls0、successful blocks0、runtime_failed）。
3个请求正常生成PNG，但validator拒绝当混合加速，summary保持incomplete。
未放宽guard/追认倍率。该graph estimate1,902,379,264，保守scratch
394,592,256，合计2,296,971,520>原optional2GiB，单此项就足以拒绝。
双图另起v2/v3减budget到1024MiB后实际执行，不提高system reserve/
optional upper；不能把不同budget的失败/有效列并成独立causal cache试验。

单图copy/surface均63ready entries，cold fill63/hit0，两个warm分别
新增63hits，累计0/63/126；miss96/129/162，decline33/66/99，eviction/
failed-fill0。copy retained1,321,807,872，surface1,340,473,344bytes。
双图copy51/surface50ready，hits累计0/51/102、0/50/100，retained
1,070,034,944/1,063,911,424bytes；不同physical pitch容量，不是相同
budget就应接纳相同矩阵数。copy/bind counters独立覆盖实际hit，未选
representation的hits0；原source checks/finite/shape/headroom全部保留。

每cache request仍96个eligible W producers（三矩阵×32层），hit+miss
实际delta96，cold fill+decline=miss，warm不新填旧源；每request32成功
channel/shared-rank blocks/32driver calls，后五步0。fail/fallback/retry0、
headroom1、adapter bindings227，LoRA crypto source cold读完整内容，warm
native proof hit1/read0/rebindfalse，没有合并adapter或藏source成本。

单图初始v1又因validator误要求`ineligible=0`失败；raw请求完成，screen
保持incomplete。Shared Device的code_binding也记录transpose A8：Private
self-test三次各3mutable W+1A8=12，实际每prediction再1A8，所以这里
ineligible累计44/76/108，既不计W miss也不缓存activation。最终contract
严格按12 bootstrap＋actual phase calls验证，warm extra32；多chunk
fixture也核对64，不删该计数或放宽旧all-step默认reuse要求。v1日志/
failed summaries保留，不将raw replay通过改写成完成的四臂实验。

## 视觉、内存与验证

单/双图每组三张copy/surface与off PNG均exact，v2/v3对应GPU/off/surface
三PNG也分别exact；GPU与hybrid不互称逐字节相同。已检查两个workload
case1的four-arm whole512：壶身/盖钮/把手、米/蓝色、左右布局、桌面和
暖光阴影很接近，仅釉面/高光等少量细节变化，未见新增明显棋盘格/
断裂/色块。缓存增加没有改变这些case，不能当更多seed/scene/用户验收。
不要求严格latent等价；完整图像/更广质量目标仍未完成。

14份100ms进程树memory evidence独立重放，complete、hash/report一致。
有效窗口swap-in/out均0，peak最高41,016,894,592bytes；system compression
累计约11.93–15.72GB，不说“无内存压力”或归因外部App/某个cache。
scope是process-tree load+cold+warm+exit，不包含external service/driver。

最终41项host tests通过无skip（初版40项不重复计数），native源码/库未
新改/编译，Private/Public retained库各502 inputs/seal独立匹配。实模型
执行证明不等于新Public cache实测/全仓回归/App发行，含原用户草稿的
working-tree build不称clean staged-only。所有owned jobs terminal，无
新`.o`/module-cache；保留库/CLI、失败/有效logs、PNG/manifest和原模型，
没有清driver/用户缓存或signal外部进程。

结论：保留显式prefill缓存screen/严格实际counter校验，继续更省内存的
cache-off首步并行＋GPU KV-hit路线。下一步优先真正LoRA/FFN handoff
和GPU计算开销、Qwen generation/encoder及Z base/LoRA矩阵，不盲加大
缓存或把这次小幅名义改善当完整目标完成。

[机器记录](../design/validation/local512-qwen-prefill-weight-cache-20261010.json)
绑定四完整窗口、全部冷/热/phase/source/memory与失败路径。
