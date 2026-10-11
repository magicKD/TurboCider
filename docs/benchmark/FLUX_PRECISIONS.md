# FLUX.2 Klein 9B：stock参考精度筛选

2026-10-11，M4 Max64GiB/macOS26.6.2，原本地9B BF16完整源、512²、
4步、seed42，无LoRA。只读mflux0.19.2的stock quantize参数，BF16→
Q4→Q8→Q8→Q4→BF16，六进程24张图，全部独立进程树内存验收完整。
每进程cold、两fresh、一次repeat；constructor、量化准备不混入request。

| stock路线 | 四fresh中位 s | tree peak GB（十进制） | 结果 |
| --- | ---: | ---: | --- |
| mflux BF16 | 8.962395 | 40.48 | 本轮参考基线 |
| mflux affine Q4 | 8.303592 | 33.62 | 对本轮BF16省约7.35% |
| mflux affine Q8 | 9.466238 | 37.01 | 对本轮BF16慢约5.62% |

上表不是原生排名。先前[同窗口原生/参考BF16配对](../status/flux9-bf16-framework-2026-10-11.md)
为8.220596/11.263455s；最新参考BF16已降至8.962395s，不能继续把旧
1.370×当未来/current普遍优势，也不能跨campaign用8.220596/8.303592
宣称等精度领先。随后v5已完成新同窗口配对，见下方；区分cache/组件
生命周期、内存压缩与kernel贡献，不因更快参考缩小优势而忽略它。

## 最新同窗口BF16配对：优势约10%，不是37%

native→mflux→mflux→native，新匹配async build、16PNG/4memory完整；
两边各四fresh warm，纯GPU，BF16同源，constructor/cold/repeat另列。

| 路线 | 四fresh中位 s | cold 正/反 s | repeat 正/反 s | tree peak GB |
| --- | ---: | --- | --- | --- |
| TurboCider GPU | 8.042414 | 8.111 /9.990 | 6.898 /6.933 | 30.13–30.48 |
| mflux BF16/stock cache | 8.828487 | 8.231 /9.469 | 9.062 /9.154 | 40.48 |

原生为1.097741×、少8.904%，仍领先但明显小于旧1.370×，也不是所有
cold都更快。参考required weights完整；warm-light correlation0.998518、
SSIM0.990202、MAE1.286/255，仍非byte-exact/广泛感知parity。
native compression累计2.96–3.13GB、reference42.46–42.68GB，swap-out
均0；不称无压力，不把完整路径差异全部归因kernel。后续独立cache0
已完成，确实逆转差距，见下方。

[最新配对便携证据](../design/validation/flux9-bf16-current-v5-20261011.json)
保留514输入/库/source/实际coverage/每请求/质量/内存与原mlx-serve
拒绝。首次source generation SHA后持久复用，代码独立绑定；没有
在每个inference重扫模型，未用repeat条件cache命中冒充fresh。

## cache0完整配对：参考胜出，必须保留

仅改变reference process-local MLX free-cache limit，数学/原BF16/步数
不变，四进程完整ABBA、16PNG/4memory、每路四fresh；不混stock。

| 路线 | 四fresh中位 s | cold 正/反 s | repeat 正/反 s | tree peak GB |
| --- | ---: | --- | --- | --- |
| TurboCider GPU | 7.979642 | 8.737 /9.120 | 7.039 /6.930 | 30.01–30.45 |
| mflux BF16/cache0 | 7.351531 | 8.443 /8.771 | 7.374 /7.347 | 32.22–32.40 |

mflux对native为1.085439×；native多用8.544%时间（参考省7.871%时间）。
reference compression累计3.24–3.44GB、native2.81–3.12GB，swap-out均0；
比较更低压力的参考后不能宣称“FLUX也始终更快”。repeat原生略快是
不同conditioning cache边界，不能覆盖fresh的负结果。

[cache0完整便携证据](../design/validation/flux9-bf16-cache0-current-20261011.json)
记录实际cache bytes/previous limit、完整coverage/PNG/内存/样本与图像
差异。原生fresh阶段encoder约0.60–1.36s、denoise约6.62–6.77s；下一
优化主项仍是DiT，不能只优化constructor或忽略完整request。
这个v6窗口的9B仍eager；后续已真正实现参数化compiled single graph，
见下方新实验。9B混合几何继续独立验证，不直接解除guard。

## 9B compiled GPU与encoder参数：完整三路线对照

新增H4096/heads32/F12288参数化single graph，weights显式输入，按
session保留编译函数；quantized/bias/runtime-LoRA路径仍走正确eager
投影和同步。9B默认未自动改为compiled，ANE拒绝保护也未解除。
每组eager→compiled→mflux→mflux→compiled→eager，24PNG/6memory，
四fresh/route。每compiled请求实际96个成功host分支；不是只看flag。

| 独立cohort / encoder配置 | eager s | compiled s | mflux/cache0 s | compiled对自身eager省时 |
| --- | ---: | ---: | ---: | ---: |
| 默认encoder求值间隔 | 7.718005 | 7.482987 | 7.314891 | 3.045% |
| encoder interval0 | 7.606133 | 7.381664 | 7.325627 | 2.951% |
| interval0 + fused attention | 7.612181 | 7.374041 | 7.343156 | 3.128% |

三组分别解读，不把最小samples拼成新倍率。compiled/eager当前八对
PNG每组RGB exact，完整输出/所有内存验收通过；不是任意形状/seed/
操作或硬件occupancy资格。fused相对deferred-only约0.1%的跨cohort
名义差异在噪声内，不当稳定收益。最后组reference仍略快；目前可以
说native接近这组调优参考，不能说native全局最优。

[默认间隔完整证据](../design/validation/flux9-compiled-current-20261011.json)、
[interval0证据](../design/validation/flux9-compiled-deferred-current-20261011.json)、
[fused证据](../design/validation/flux9-compiled-fused-current-20261011.json)保留
新匹配库514输入、96真实分支、全部cold/fresh/repeat/quality/压力。
新库build/flux9-compiled-v1-private，不借旧库身份；最高名义native
配置在best-local-configs中保留，未自动改变生产默认。

## 表示与正确性

mflux stock quantize覆盖eligible text/DiT modules，保留其它原源组件。
required source target为233 DiT/398 encoder/250 VAE，完整覆盖，包含
mapped buffers；不是只检查24个PNG存在。完整原尺寸图像均保留，
Q4 warm-light已查看为正常狐狸。跨精度RGBA差异、alpha、白背景合成与
所有cold/fresh/repeat/peak/compression/swap在[便携记录](../design/validation/flux9-stock-precisions-current-20261011.json)。
近似数值不强求pixel-exact，但也不把不同recipe标为同精度。
warm-light Q4相对同engine BF16：RGB correlation0.958838、SSIM0.867560、
MAE10.771/255，近似图像漂移不当成native/reference parity。

原生affine当前支持g32，stock reference默认g64、embedding行为不同。
同源匹配g32 linear-only/dense embedding pack生成器已准备，还未完成
真实大模型转换/匹配运行；Q4磁盘预算与Q8更大预算分别预检，不删除
原模型腾空间。实际只读stock loader已支持由压缩planes形状恢复g32；
匹配runner须验实际bits/group/embedding与完整source coverage，不只
相信metadata，不需要为此修改参考源码。

9B实际H4096/F12288、8 dual+24 single；现有4B-only hybrid graph/
padding/export几何不能直接套用。原生9B混合仍明确拒绝未验证分区，
不把历史4B的2.266/1.628s套到9B，也不以此N/A冒充探索已完成。

## 复现

tools/validation/flux9_mflux_quant_screen.py，model=models/FLUX.2-klein-9B，
order=bf16,q4,q8,q8,q4,bf16；完整PNG墙钟，无额外逐stage GPU fence。
父进程初始验源，每arm前后generation guard，不在inference逐次SHA。
tools/validation/flux9_mflux_quant_evidence.py独立绑定raw/内存/PNG、
重算全fresh中位数；不得把其native_latency_ranking_valid=false改为
原生胜出。原mlx-serve格式拒绝仍保留，转换成功路径继续pending。
