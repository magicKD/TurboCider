# ConvRot GPU partial：允许额外 BF16 舍入后的整请求收益

2026-10-09，Asia/Singapore，M4 Max64GB/macOS26.6.2。接续
[MPP partial](local512-convrot-mpp-partial-2026-10-08.md)与
[完整 LoRA](local512-convrot-lora-2026-10-08.md)。只读本地原模型和
distill-patch adapter，无下载、模型/adapter改写或dense sidecar。
完整Z/Qwen/base/LoRA/edit/encoder/GGUF目标仍active，不以本轮替代。

## 实现与近似范围

新增默认关闭的 `TURBOCIDER_Z_CONVROT_BF16_PARTIAL=1`。GPU complement
保持原packed Q8/BF16 scales与Comfy H256，base-down使用原typed
QMM，输出先舍入BF16，再widen到F32与ANE F32 partial合并。它明确
比原MPP F32 partial增加一次舍入；不是更高精度F32 GEMM，不改变
权重、metadata dtype、ANE graph或hidden ABI。down-LoRA仍只在完整
joined corrected hidden上施加一次，不能对每个shard单独舍入adapter。

要求显式authorized Private fixed-channel ConvRot W8A8、resident512²、
approximation、F32 join，LoRA另需原完整runtime-LoRA开关。FP32 metadata
override、Public/auto/non512/streaming/constrained拒绝；普通GPU的有效
flag不生效，非法值拒绝。与MPP recipe互斥。owner先drain/synchronize
再snapshot Weights；request identity包含新recipe，失败仍完整GPU重算。
auto/default GPU路由不变。原用户ConvRot compiled/cache草稿未提交。

## 组件与数值

原layer0 ConvRot、legacy g32/BF16 metadata、synthetic BF16 hidden，
计时包含H256、consumer与F32 widen/eval；3warmup/15交错hot。
M33/1024/1056、K5120/6144/8192，原physical K10240不compact。
M1056/K6144：MPP8.497041ms → QMM+widen4.452875ms，relL2=.00165250
（约.165%）。M33最大约.254%。是component诊断，不直接外推整图。

18 actual MLX cases覆盖g32/64/128、dense/Metal rotation、M33/67/128、
physical row/column offsets、原QMM boundary exact、F32 output、原/新/原
切换与immutable master；正负stacked adapter、BF16 alpha在场时base
partial仍不含LoRA。组件预算1%，未为失败放宽。不同recipe与FP16输入
拒绝。旧MPP8-case、ConvRot72-case、leased LoRA、Qwen phase回归也通过。

## 完整请求：同库优化GPU与MPP控制

每臂独立进程、一冷两同prompt热请求，512²/原source；100ms进程树
memory sampling，无continuous-load/物理overlap或正式性能资格。
base用原compiled-dense GPU/retained3GiB allocator hint，不借慢legacy
作分母；LoRA用完整shared-rotation GPU、原151MiB rank32 adapter/strength1，
238实际投影。两hybrid都Fa4096/Fg6144、fixed async/F32 join、相同
GPU-sensitive ordinal2（LoRA）、无prefetch/lookahead/W-code cache。

| workload / bucket | optimized GPU s | MPP F32 hybrid s | BF16 partial hybrid s | GPU/new |
| --- | ---: | ---: | ---: | ---: |
| base fox/seed42/4步/c1056，reverse | 4.511733 | 4.090700 | 3.579456 | 1.26045× |
| real LoRA fox/seed42/8步/c1056，forward | 9.503763 | 9.113520 | 8.101684 | 1.17306× |
| real LoRA lighthouse/seed17/8步/c1056，reverse | 9.809163 | 11.739707 | 11.016405 | 0.8904× |
| 同lighthouse/c1152，forward | 9.795130 | 9.387681 | 8.405630 | 1.16531× |

base名义wall下降20.66%，LoRA fox/lighthouse适配bucket后下降14.75%/
14.19%。仅所列诊断；方向与scene同时变化，不冒称同workload ABBA。
不能把不同窗口share/bucket或component差值全部归因一个kernel。

base沿用显式cold-retry-cap2诊断：两hybrid冷各2次overflow，headroom
1→4→16，warm无新增retry且recipe相同；不是zero-retry正式资格。
每请求128 successful blocks，calls130/258/386含两次retry。
LoRA fox每请求248 blocks/calls，0fail/fallback/retry、headroom1；
ordinal2每请求8次完整GPU是policy，不是失败回退。

## 提示词长度的bucket cliff：保留退化，不挑有利scene

fox31tokens，caption pad32，main rows1056；lighthouse37tokens，pad64，
main rows1088。noise-refiner两层仍1024。旧c1056在lighthouse主干需两
次调用：每请求480calls而非248，成功blocks仍248，因而比GPU慢。
用原exporter离线生成checkpoint-free c1152/v2小template（无模型
payload），同caption/seed/share恢复248calls。Private只读取几何再
发射native graph，不称Public artifact已加载或性能资格化。
validator现检查actual bucket/tokens/32-row pad/noise/main/forced policy/
retry的精确call公式，不能用successful block数掩盖额外calls。

## 视觉、状态与限制

已看fox base/LoRA GPU/new whole及center，LoRA fox另看top-left/
bottom-right；主体姿态、眼耳、毛色、尾巴、雪/松枝接近，细纹略变，
未见明显新块纹/断裂。lighthouse c1152 whole/center布局、灯光、塔身
石纹接近，但GPU屋顶红色、candidate较暗，必须保留该颜色差异。
非byte-equivalent精度配方，未用旧严格N1作新图自动验收；旧失败记录
不改写。limited agent观察不是用户接受、多seed/所有detail资格，visual
manifests保持pending。新/旧GPU和MPP fox PNG与历史各自control exact，
说明未改这些默认数学；新BF16 partial PNG与MPP不同。

同一进程base→原LoRA1→同LoRA.5→base，same admitted executor、累计
248/496/744/992calls、binding0/238/238/0、corrections0/248/496/496，
零失败/retry。首尾basePNG exact，两strength不同；不是第二训练adapter。

最初v1误用未启用experimental capability的native库：GPU arm完成，
runtime被原guard拒绝，summary incomplete，不能借其GPU分母；保留
错误log。v2 Private明确experimental-probes=1，Public普通=0，未降低
guard。两新native-only库/thin CLI成功，499inputs各匹配；仍含用户
草稿，非clean staged-only/App发行。Public实际release guard通过。
Private/Public selected各9项通过，额外Private actual late-failure/full
GPU recovery3项、host7项通过，无skip，不是全仓或完整模型矩阵。

继续：按实际rows选择bucket/盈利route、更多scene/seed和无竞争窗口，
base正常步数与更广LoRA，再回到Qwen/encoder/GGUF真实消费优化。
本轮没有自动Private/Public盈利调度或native INT8 MAC/物理并发证明。
默认仍关闭新近似，不将一格1.26×宣称整个目标完成。

机器记录：[local512-convrot-bf16-partial-20261009.json](../design/validation/local512-convrot-bf16-partial-20261009.json)。

全部owned jobs确认terminal后，仅清理三个isolated native build的639个
可重建`.o`，50,299,200 logical bytes（约48MiB）及三个空module-cache。
库/CLI/probes、原始/失败logs、PNGs、manifests和原模型全部保留；
未清用户缓存或external process，编译对象可按原命令重建。
