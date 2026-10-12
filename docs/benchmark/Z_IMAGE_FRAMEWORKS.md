# Z-Image Turbo：完整框架与GPU/ANE对照

更新2026-10-12。最新当前版本结果优先；下方10-11七路线保留为独立
历史cohort，不把较慢旧参考继续当作当前优势的分母。

## mflux0.22.0完整四路线：混合1.34×，参考更省内存

原BF16 encoder/DiT/VAE、512²/8步/seed42、无LoRA。新版本clone保持
只读，原reference checkout不改；独立环境满足其声明依赖，pip check通过。
实际MLX0.32.3/Torch2.14.1/Transformers5.19.0，原生库仍是已绑定实验
库；这比较完整框架环境，不是相同MLX版本的纯kernel单因素实验。

8进程四路线完整回文，每进程cold/两fresh/repeat；32PNG/8memory全部
完整，四fresh/route。参考实际process-local free-cache为1024MiB，
不修改OS/wired/swap策略；constructor另列。

| 路线 | 四fresh中位s | cold正/反s | repeat正/反s | tree peak GB（十进制） |
| --- | ---: | --- | --- | --- |
| TurboCider GPU | 7.354224 | 9.266 /7.803 | 6.989 /6.989 | 31.81–31.97 |
| TurboCider frozen W8A8 | 5.959064 | 18.351 /16.793 | 5.329 /5.326 | 32.03–32.16 |
| mflux0.22 BF16 | 7.976479 | 8.994 /8.030 | 7.972 /7.977 | 22.51–22.52 |
| mflux0.22 partial FP16 compute | 8.355058 | 8.415 /8.420 | 8.351 /8.371 | 22.61–22.63 |

对本窗口BF16参考，GPU为1.085×、frozen为1.339×；frozen对自身GPU
省18.971%。仍是热态领先，但不是旧参考的1.647×。参考peak比native
约低9–10GB，不能宣称我们同时在时延与内存上全面最好。frozen冷请求
16.79–18.35s仍贵，参考constructor约1.09–1.40s、native约0.14–0.16s
也保留，不扣成本或用repeat当fresh。

FP16候选在此任务反而比自己的BF16慢4.746%，负结果保留；不把
precision flag当作成功加速。实际306个transformer floating参数为FP16、
215个为BF16，text embedding仍BF16，float32_stream=false；每张图
实际generation_parameters=compute_precision:float16。它只改变上游
attention/FFN内部数学，residual/modulation/text/VAE仍按模型精度，
不是全模型FP16或W8A8。BF16 control显式拒绝偷偷启用该选项。

warm-light同engine FP16对BF16 SSIM0.989234、MAE1.116/255；相对native
GPU的SSIM：frozen0.932267、mflux BF16 0.908011、FP16 0.910548。
16对正反序RGBA均exact；8组同engine FP16误差与全部跨engine/alpha/
白背景RGB另列，有限prompt仍不是广泛感知资格或同noise证明。

全部新窗口swap-out0，native frozen一个窗口compression累计0.341GB，
其它为0；不是所有外部CoreML服务/driver的归因或硬RAM cap。frozen
每请求256成功calls、无fallback；仍没有physical NE occupancy trace。
current required coverage521 DiT/398encoder/244VAE完整，模型payload不改。

新版源码将time embedding cast回模型精度，旧固定源码未做该cast；
旧表的“BF16”是source/global配置标签，不代表旧隐流所有算子均BF16。
版本/依赖/cache/cohort一起变化，不能把旧9.520s到新7.976s差异全部
归因一个kernel或单因素release。旧样本保留，不与新样本混池。

[完整当前验收](../design/validation/z-mflux022-bf16-fp16-current-20261012.json)
绑定commit `31c9640399572ea394d07092763be9a36007a754`、0.22.0源码
snapshot、原payload/layout、514个native输入及所有32PNG/8memory。
当前保存BF16/1024MiB为此新版本已测参考候选，FP16未选为最快；
原生frozen配置保留，其旧5.780s不能与新7.976s拼倍率。

```sh
.venv/bin/python3 -B tools/validation/z_mflux_framework_screen.py \
  --reference-model outputs/benchmark-model-views/z-bf16-mflux-v1-20261011 \
  --library build/flux9-predecode-v2-private/libturbocider.dylib \
  --mflux .deps/mflux-v0.22.0-20261012 --python .deps/mflux-0.22-env/bin/python3 \
  --extra .deps/mflux-0.22-env/lib/python3.13/site-packages --reference-cache-mib 1024 \
  --order native-gpu,native-frozen,mflux-bf16,mflux-fp16,mflux-fp16,mflux-bf16,native-frozen,native-gpu \
  --output outputs/new-z-current-reference
```

更多shape/quantized reference与其它家族当前版本仍pending，扩大目标未完成。

## 保留10-11旧参考七路线

2026-10-11，M4 Max64GiB/macOS26.6.2。原BF16 encoder/DiT/VAE经无损
布局适配给只读mflux0.19.2；512²、8步、seed42，无LoRA。十四进程
七路线完整正反序，每进程cold、两fresh、一次repeat，constructor另列。
56张PNG、14条独立进程树内存证据完整；fresh不混入repeat cache hit。

## 完整PNG墙钟

| 路线 | 四fresh中位 s | cold 正/反 s | repeat 正/反 s | tree peak GB（十进制） |
| --- | ---: | --- | --- | --- |
| TurboCider GPU | 7.363311 | 10.231 /7.590 | 6.991 /6.990 | 30.76–31.91 |
| TurboCider frozen Core ML | 5.780123 | 21.587 /11.608 | 5.334 /5.332 | 30.90–32.08 |
| TurboCider Private runtime | 6.139409 | 11.867 /6.646 | 5.753 /5.763 | 30.57–31.64 |
| TurboCider Public runtime | 6.537484 | 8.751 /9.816 | 6.276 /6.165 | 30.93–31.57 |
| mflux BF16 | 9.519604 | 10.337 /9.545 | 9.527 /9.536 | 31.57–31.58 |
| mflux affine Q4 | 9.971474 | 10.054 /10.055 | 10.022 /9.961 | 26.89 |
| mflux affine Q8 | 10.080610 | 10.192 /10.189 | 10.162 /10.076 | 28.67–28.68 |

对mflux BF16，原生GPU为1.293×，frozen为1.647×。frozen对自身GPU
省21.501%，当前七路线热态最快；冷启动明显更贵，不应将它推广为
一次性请求最优。Private/Public不同路径均保留，不互换标签。

参考Q4/Q8覆盖其eligible text/DiT/VAE modules，是独立affine recipe，
不是原GGUF Q4_0/Q8_0或ConvRot。这轮降低观测peak但未降低延迟。
模型组件/allocator生命周期不同，完整框架倍率不等于纯kernel倍率。

## 实际执行和来源

- GPU是原生compiled fused blocks，encoder GPU，不是CPU慢分母。
- frozen使用checkpoint-bound image-only W8A8，ANE5120/GPU5120、
  rows1024；每request256成功runtime calls。配置cpuAndNeuralEngine，
  physical NE residency未观测，不把Core ML成功当硬件占用trace。
- Private为Hadamard W8A8、intermediate channels ANE4096/GPU6144、
  F32 partial join、gpu_iosurface。cold出现已记录的overflow retry；
  四warm均不继续改变headroom，fallback0，不能抹掉初始化retry。
- Public为row partition、352-row模板、gpu_iosurface；实际256个hybrid
  block/request，fallback0。它与Private通道分割不是同一算法边界。
- mflux required coverage：519 DiT、398 encoder、244 VAE全部覆盖，
  不只检查文件存在。mapped VAE buffers也纳入required target。
  stock Z每次重新编码prompt，repeat不能伪称reference cache hit。

原生和参考warm-light输出已查看，为相近构图的正常狐狸图，而非Qwen
旧失败的彩色噪声。完整原尺寸RGBA/RGB差异、alpha/白背景合成和正反
序确定性保留在机器记录；有限prompt/单seed不代表广泛画质资格。

warm-light相对原生GPU，frozen/Private/Public的RGB SSIM分别0.932267/
0.981101/0.989876，MAE4.973/2.027/1.315每255；Private/Public虽稍慢，
此样本漂移更小。mflux BF16 SSIM0.904900、MAE7.394；affine Q4
SSIM0.792111、MAE21.958，不能只报省内存而抹去图像差异。28对正反
序RGBA均exact；这只证明当前固定任务确定性，不是广泛质量parity。

## 复现与范围

[完整便携证据](../design/validation/z-valid-framework-current-20261011.json)
绑定源/库/514个native输入、无损layout receipt、参考源码snapshot、
全部请求与内存/PNG SHA、channel/join/calls/retry/fallback和图像差异。
内存是load+cold+warm+exit的process-tree physical peak，不是硬RAM cap。
system compression/swap-in/out都保留，不只报告MLX allocator。

入口为tools/validation/z_mflux_framework_screen.py，使用原BF16派生
布局与build/benchmark-async-v1-private，8步/4request、private4096/f32：
native-gpu,native-frozen,native-private,native-public,mflux-bf16,mflux-q4,mflux-q8,mflux-q8,mflux-q4,mflux-bf16,native-public,native-private,native-frozen,native-gpu。
初次完整source SHA后generation未变时复用，不在每个inference重扫。

[五种本地表示16配置](Z_IMAGE_FORMATS.md)与本页是独立campaign，不
混用最小单样本。更高/更低Private share、F16 join、1024²仍需当前
完整筛选；[已测最佳配置](best-local-configs-20261011.json)不是全局最优
证明，不自动改变生产默认。raw PNG/模型/compiled graphs不进Git。
