# FLUX9B：mflux0.22与原生七路线完整对照

2026-10-12，M4 Max64GiB/macOS26.6.2，512²/4步/seed42，base-only。
原BF16源及已保留encoder252+DiT153 eligible linears Q4/g32 pack，
embedding/VAE保持BF16。新版本只读clone与独立声明依赖环境；原生
实验库仍保留，不升级/改写原模型或原reference checkout。

14进程七路线正反序，每进程cold/两fresh/repeat；56PNG/14个独立
process-tree memory窗口全部验收，四fresh/route。每行只用本cohort
分母，constructor另列；不拼旧7.211s、旧参考或repeat cache hit。

## 完整生成墙钟与内存

| 路线 | 四fresh中位s | cold正/反s | repeat正/反s | tree peak GB（十进制） |
| --- | ---: | --- | --- | --- |
| native BF16 compiled GPU | 7.661436 | 8.599 /8.261 | 6.712 /6.912 | 30.48–30.62 |
| native BF16源/GPU+CoreML prefix | 7.433898 | 21.864 /21.687 | 7.071 /7.068 | 30.96–31.25 |
| native packed encoder+DiT Q4 | 8.015237 | 8.815 /8.702 | 7.492 /7.480 | 11.92–11.93 |
| native Q4源/预解码BF16 DiT+compiled | 7.253946 | 8.031 /8.939 | 6.693 /6.690 | 24.36–24.43 |
| mflux0.22 BF16/cache0 | 7.318347 | 8.875 /8.969 | 7.327 /7.354 | 32.12–32.13 |
| mflux0.22 partial FP16/cache0 | 7.649527 | 9.303 /10.430 | 7.636 /7.634 | 32.72–32.78 |
| mflux0.22 matched encoder+DiT Q4/cache0 | 7.875917 | 8.177 /7.934 | 7.884 /7.877 | 12.44–12.45 |

可以确认的收益是预解码对自己的packed Q4省9.498%，但peak多约
12.44GB，仍是高内存approximation opt-in。对同stored-Q4参考名义
约1.086×；对原BF16参考仅约0.88%名义差异，而且权重数学不同，
不能称严格BF16同精度或全局最优。native packed Q4反而比自己的
BF16慢，参考packed Q4也比自身BF16慢；量化的主要价值是低内存。

BF16源GPU参考胜：native compiled比reference BF16多约4.69%
时间；native packed Q4比matched reference Q4多约1.77%时间。
partialFP16比同engine BF16慢4.525%，不是成功加速。所有负结果保留。
原生runtime为主环境MLX0.32.0，参考为0.32.3/Torch2.14.1/
Transformers5.19.0；比较的是完整框架环境，不把差异全部归因kernel。

## 混合收益约3%，依赖prompt/生命周期

本cohort hybrid对compiled中位省2.970%，不沿用旧独立cohort0.031%
说这组“完全没有收益”，也不把约3%当任意请求的稳定承诺。
全部四fresh样本原样保留：

| route | 正序soft /light s | 反序soft /light s |
| --- | --- | --- |
| compiled BF16 | 7.826 /7.300 | 8.298 /7.496 |
| hybrid | 8.147 /6.655 | 8.142 /6.726 |

light两方向改善、soft一方向变慢；每route四样本中位包含这两种
prompt，不删除慢值。hybrid repeat7.07s又慢于compiled6.71–6.91s，
cold21.69–21.86s明显昂贵；旧repeat省11.5%不能跨轮套用。
因此保存已有experimental hybrid及本组复测，不自动替换BF16 GPU
质量配置或更快的source-Q4预解码性能候选。

实际24个single-block CoreML partitions、ANE prefix4096/GPU8192、
rows1536，每4步请求96successful calls、warmup0、failure/fallback0。
这是INT8 per-channel weights/FP16 I/O，不是W8A8/native integer MAC，
没有physical ANE occupancy/硬件重叠trace。源index/两shards与所有
compiled artifacts身份绑定；首步不整体交给ANE。

预解码每新session153次真实decode、后续三个请求0次；每请求96
compiled singles，全部128个dual+single block events。它仍只允许
既有9B/GPU/resident/512²/4步/无LoRA条件，physical memory≥48GiB、
显式request budget若给出≥28GiB，不改变生产默认或历史FP16 dequant。

## 质量与压力

全部原尺寸RGB/opaque native alpha、64组同/跨engine误差、28对
正反序输出均验证；各route固定任务4个request正反序RGB exact。
warm-light的代表性差异：

| 同engine或明确跨engine比较 | SSIM | MAE /255 |
| --- | ---: | ---: |
| native hybrid vs native BF16 | 0.989281 | 1.334 |
| native预解码 vs native packed Q4 | 0.994113 | 1.109 |
| native预解码 vs native原BF16 | 0.838604 | 11.562 |
| reference partial FP16 vs reference BF16 | 0.987527 | 1.656 |
| native BF16 vs reference BF16 | 0.986036 | 1.953 |

允许较宽delta不等于无损；Q4源漂移远大于额外预解码舍入。有限
prompt/seed不证明广泛感知资格，跨engine同seed也不保证同noise。
已查看reference FP16/Q4预检和完整图，均为正常雪林狐狸。

全部窗口swap-out0；native/reference packed Q4与predecoded compression
累计0，BF16 native约3.25–5.45GB、hybrid6.00–6.44GB、reference
BF16 3.98–6.18GB/FP16 4.21–5.08GB。累计活动不是当时占用，不称
全窗口无压力。peak包含load+cold+warm+exit，不归因全部外部服务/
driver，不是硬RAM cap。

## 证据、配置与复现

[完整便携验收](../design/validation/flux9-mflux022-current-20261012.json)
绑定0.22.0 commit `31c9640399572ea394d07092763be9a36007a754`、源码/
原生514输入/模型/相同g32压缩planes、全部请求/PNG/内存、实际
runtime事件与64个媒体比较。[配置记录](best-local-configs-20261011.json)
新增本cohort current reference/复测绑定；原best native旧中位仍按
自己的controls解释，不重新拼当前reference分母。

```sh
.venv/bin/python3 -B tools/validation/flux9_current_framework_screen.py \
  --model models/FLUX.2-klein-9B \
  --pack outputs/flux9-matched-full-q4-g32-pack-v2-20261012 \
  --library build/flux9-predecode-v2-private/libturbocider.dylib \
  --mflux .deps/mflux-v0.22.0-20261012 --python .deps/mflux-0.22-env/bin/python3 \
  --extra .deps/mflux-0.22-env/lib/python3.13/site-packages \
  --ane-manifest outputs/flux9-ane-prefix4096-r1536-compiled-v1-20261011/manifest-75dd56c7090d32e0dc36978b24f1853b950d28086357df9bb90c82ee9d612a4c.json \
  --output outputs/new-flux-current-frameworks
```

initial source-generation SHA后跨campaign持久复用，arm边界stat guard，
无每次inference模型SHA、OS cache purge或系统RAM策略更改。更多
native库版本/precision/shares/shapes、mlx-serve兼容布局及LTX成功
外部完整视频仍pending，扩大目标保持active。
