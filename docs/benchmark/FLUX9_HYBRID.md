# FLUX9B：完整GPU/ANE实验，而不是前缀微基准

完成/发布：2026-10-12，Asia/Singapore。raw目录的20261011是启动标签，
不将跨午夜完成的结果冒充10月11日已完成。M4 Max64GiB/macOS26.6.2；
原FLUX.2 Klein9B BF16 encoder/DiT/VAE，512²、4步、seed42、无LoRA。
8进程四路线正反序，每进程cold、两个fresh prompts、一次立即repeat；
32张完整PNG和8个独立进程树memory窗口全部完整，四fresh/route。

## 同窗口结果：fresh没有稳定混合收益

| 路线 | 四fresh中位s | cold正/反s | repeat正/反s | tree peak GB（十进制） |
| --- | ---: | --- | --- | --- |
| native eager GPU | 8.153859 | 9.054 /8.944 | 6.935 /6.953 | 30.65 |
| native compiled GPU | 7.843808 | 8.182 /8.241 | 6.953 /6.735 | 30.65 |
| native experimental GPU/CoreML prefix | 7.841375 | 15.728 /16.031 | 6.050 /6.065 | 31.27 |
| mflux BF16/cache0 | 7.355288 | 9.044 /9.164 | 7.349 /7.367 | 32.27–32.33 |

compiled相对自身eager省3.803%；hybrid相对compiled只少0.031%，是噪声
量级，不能据此选作稳定更快的默认。hybrid比调优参考多用约6.61%时间。
旧独立cohort的7.374/7.343s仍保留，不拼接旧GPU与新hybrid计算倍率。

同prompt立即重复的两native路线都实际conditioning cache hit：compiled
中位6.844s、hybrid6.058s，hybrid少约11.5%。这是这组cache-hit请求的
单独诊断，不混入fresh排名，也不将mflux的其它组件/cache策略当成
等价命中。有限prompt/单seed尚不足以推荐一般repeat任务自动切换。

hybrid cold明显更贵：setup约7.70–8.01s，其中manifest validation
约7.51–7.74s；后续request不重新创建该session。这里报告的是host
span，不把它说成ANE kernel耗时，也不扣掉初次校验来美化cold数字。
首个fresh native encoder约1.50–1.76s，第二个约0.55–0.64s；完整
原始样本都参与统计，没有删去较慢的第一个fresh请求。全部窗口有
系统compression活动约5.47–8.02GB、swap-out0；不宣称无内存压力。

## 真实分工与安全边界

- 真实9B几何：H4096/F12288、32heads、8dual+24single。dual与encoder
  保持GPU；每个single的FFN前4096通道交给CoreML，其余8192通道与
  attention由compiled GPU图执行。GPU使用async提交，再进行blocking
  CoreML prediction；这只是host并行调度证据，不是物理重叠trace。
- 原checkpoint保持index+两个shards，没有平铺18GB副本。24个前缀
  source graphs已编译，每请求4×24=96次实际runtime calls，session
  累计96/192/288/384；warmup0、runtime failures0、fallback0，
  checkpoint/index/全部shards身份核验通过。
- 新前缀是CoreML INT8 per-channel **权重**压缩、FP16 I/O。它不是
  activation8，不等同于Qwen/Z Private W8A8，不证明原生整数MAC。
  配置为cpuAndNeuralEngine，observed ANE residency仍unknown。
- 只在experimental build、显式环境opt-in、allow_approximation、
  非空manifest、512²/4步/image.generate、dynamic_text=false、无LoRA
  下开放。正式build仍拒绝；9B自动路由与descriptor没有新增普遍混合
  资格。完整API fixture已核验错误环境值/尺寸/步数/dynamic-text拒绝。

## 完整输出误差

所有原尺寸PNG/RGB与opaque alpha均验证，未缩图。hybrid相对同窗口
eager GPU的三个prompt，Gaussian11 RGB SSIM为0.992573/0.995018/
0.989281，MAE为1.397/0.992/1.334每255；不是pixel-exact。狐狸主体、
构图接近，毛发/雪地局部有差异；有限prompt不证明广泛感知资格。
compiled/eager记录中的八对RGB exact；四路线各四个请求的正反序RGB
均exact，共16对。repeat输出也在自身session中exact。

低层blocks0/12/23、两个输入幅度的六次prefix probe另保留，rel-L2
1.52–3.62%、最低cosine0.99934；它不是上述完整生成的图像误差。

## 证据与复现

[完整便携记录](../design/validation/flux9-hybrid-current-20261012.json)
绑定514个native构建输入、32PNG、8memory、完整raw report、原source
和所有compiled partition身份、actual calls/geometry以及全部RGB比较。
参考为只读mflux0.19.2，cache limit0，required targets全覆盖；不改
参考数学。原mlx-serve布局拒绝仍另列，不作为成功耗时分母。
[保留配置](best-local-configs-20261011.json)的experimental_configurations
保留可加载候选，但不替换既有pure-GPU推荐。

```sh
.venv/bin/python3 -B tools/validation/flux9_bf16_framework_screen.py \
  --model models/FLUX.2-klein-9B \
  --library build/flux9-hybrid-v1-private/libturbocider.dylib \
  --mflux ../references/mflux \
  --python .deps/qwen21-oracle-env/bin/python3 --extra .deps/mflux-reference-extra \
  --async-evidence --reference-cache-mib 0 \
  --native-encoder-eval-interval 0 --native-encoder-fused-attention \
  --ane-manifest outputs/flux9-ane-prefix4096-r1536-compiled-v1-20261011/manifest-75dd56c7090d32e0dc36978b24f1853b950d28086357df9bb90c82ee9d612a4c.json \
  --order native,native-compiled,native-hybrid,mflux,mflux,native-hybrid,native-compiled,native \
  --output outputs/new-flux9-hybrid-palindrome
```

原model/compiled artifacts/raw媒体不进Git。父进程初次source SHA后
复用generation注册，arm前后只检查文件身份，不在每次inference重扫
模型；CoreML session自己的初次manifest/checkpoint校验耗时保留。
匹配量化、更多份额/尺寸、进一步prefill/加载优化与其它家族缺口仍在，
扩大目标继续active，不将这个单一实验当作全目标完成。
