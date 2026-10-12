# FLUX9B：提前解码Q4 DiT并用compiled GPU执行

2026-10-12，M4 Max64GiB/macOS26.6.2，base-only、512²/4步/seed42。
源为已有encoder+DiT affine Q4/g32 pack，embedding/VAE保持BF16；只把
153个DiT投影在内存解码成BF16 operands，encoder仍Q4。额外BF16舍入
使它不是packed Q4的bit-exact数学，更不是原BF16权重，不混淆精度。

## 完整五路线：省时9.88%，用更多内存

10进程正反序，每进程cold/两个fresh/一次repeat；40PNG和10个独立
memory窗口全部完整，四fresh/route。全部用本cohort分母。

| 路线 | 四fresh中位s | cold正/反s | repeat正/反s | tree peak GB（十进制） |
| --- | ---: | --- | --- | --- |
| native原BF16 compiled GPU | 7.750545 | 8.542 /8.963 | 6.667 /6.722 | 30.26–30.63 |
| native packed encoder+DiT Q4 GPU | 8.001289 | 8.805 /8.172 | 7.491 /7.480 | 11.95–11.97 |
| native Q4源/predecoded BF16 DiT+compiled GPU | 7.210639 | 7.957 /9.026 | 6.822 /6.714 | 24.40–24.45 |
| mflux原BF16/cache0 | 7.342802 | 8.967 /10.264 | 7.445 /7.344 | 32.28–32.33 |
| mflux packed encoder+DiT Q4/cache0 | 8.027872 | 8.405 /8.098 | 8.032 /8.078 | 12.56–12.57 |

predecoded对自己的packed Q4为1.10965×、省9.8815%；对同源stored-Q4
参考为1.11334×。本窗口它也名义比原BF16参考少约1.80%时间，但只有
四fresh且权重数学不同，不能称严格BF16同精度或全局胜利。旧独立
cohort的7.374s/8.005s不拼作分母。

代价是peak比packed多约12.45GB；它是高内存performance opt-in，不
替换约12GB的memory配置。BF16窗口compression约5.19–6.28GB，packed
Q4为0，predecoded一个窗口0、另一个114688bytes；swap-out全0，不称
所有窗口无压力。peak是进程树load+cold+warm+exit，不是硬RAM cap。
cold的解码/编译/加载都保留，反序cold9.026s没有删掉或扣成本。

## 实际执行、安全与默认

- 独立experimental library及显式环境opt-in；只允许9B、GPU、resident、
  allow_approximation/compile_gpu、512²/4步/image.generate、固定512文本
  tokens、无LoRA/streaming/ANE。physical memory至少48GiB，显式request
  budget若提供至少28GiB；不推广任意硬件、尺寸或低RAM。
- 每新model session实际153次projection解码，后续三个请求0次。
  每请求96个actual compiled single-block events，全部128个dual+single
  block events；不是只看request flag。所有dense operands检查BF16。
- 在复制的Weights集合中逐一eval解码，完整成功后才commit，取消/
  失败不留下半解码集合；临时同时持有packed/dense数组，所以保守
  admission不能省略。weights文件、模型数学定义与参考源码不改写。
- decode后删除scales/biases，不让dense Weight走packed分支。卸载、
  LoRA/source clear、streaming切换和关闭开关都会重置状态/恢复packed
  来源；single graph仍显式传weights，不捕获旧model数组。
- 原Weights::dequantize默认FP16不变，新overload明确BF16；小型真实
  MLX Q4/Q8 fixture验证两种dtype、旧行为、copied source不变、无效
  dtype先拒绝。未把这个fixture的速度当成模型倍率。

源/API没有新增自动推荐资格。当前保留BF16质量配置、packed Q4低RAM
配置，另保存此performance候选；允许近似且满足严格shape/内存条件
时可显式使用。它不是ANE加速，不套用BF16 CoreML manifest。

## 输出与session恢复

warm-light相对packed Q4：SSIM0.994113、MAE1.109/255、correlation
0.999547；毛发/雪地局部有小差异。对原BF16：SSIM0.838604、MAE
11.562/255，保留源Q4量化漂移，不能因extra rounding小就说BF16 parity。
原尺寸RGB/opaque alpha全部验证；五路线20对正反序RGB均exact。

额外同一session packed→predecoded→packed→predecoded真实生成4张图：
compiled events0/96/0/96、decode events0/153/0/153；两次packed pixels
exact，两次predecoded pixels也exact，证明关闭开关确实重载源，而非
悄悄继续用dense缓存。[未计时恢复诊断](../design/validation/flux9-predecode-transition-20261012.json)
不作额外performance或memory分母。

## 证据与复现

[完整便携记录](../design/validation/flux9-predecode-current-20261012.json)
绑定514个native输入/新库、原/派生source、mflux0.19.2源码、40PNG/
10memory、真实runtime事件及48个完整RGB比较。模型/raw/媒体不进Git。
[保存配置](best-local-configs-20261011.json)的performance_configurations
保留library/环境/request及自己的controls，未自动改生产默认。

```sh
.venv/bin/python3 -B tools/validation/flux9_matched_affine_screen.py \
  --model models/FLUX.2-klein-9B \
  --pack outputs/flux9-matched-full-q4-g32-pack-v2-20261012 \
  --library build/flux9-predecode-v2-private/libturbocider.dylib \
  --mflux ../references/mflux \
  --python .deps/qwen21-oracle-env/bin/python3 --extra .deps/mflux-reference-extra \
  --order native-bf16,native-q4,native-predecoded,mflux-bf16,mflux-q4,mflux-q4,mflux-bf16,native-predecoded,native-q4,native-bf16 \
  --output outputs/new-flux9-predecode
```

初次source-generation SHA后持久复用，arm边界stat guard，无逐次
inference模型SHA或OS cache purge。扩大目标仍active，更多source/
shape/precision、GPU/ANE分工和成功外部LTX/兼容布局继续pending。
