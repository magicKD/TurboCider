# FLUX 9B BF16：同源 mflux 对照与 mlx-serve 格式边界

2026-10-11，M4 Max64 GiB/macOS26.6.2。没有改模型、native 计算或参考
实现；补齐当前本地 FLUX.2 Klein **9B** 的 base benchmark，不拿历史4B
替代。当前 native 9B DiT 明确拒绝未验证的 GPU/ANE 分区，因此本轮
最好的已测可用配置是 GPU，不虚构混合成绩。

## 同文件、同任务、正反序

原 `models/FLUX.2-klein-9B` 两个 DiT、四个 encoder shards 和原 VAE，
BF16，不量化、不加 LoRA；512²、4步、seed42、guidance1。native/mflux
统一512-token padding；参考实际token IDs逐一对照native源码中的字面
chat template，valid counts另记录。native taps8/17/26 post-layer与
reference hidden states9/18/27对应；没有为了比较改参考层数或tokenizer。

顺序 native→mflux→mflux→native，四独立进程，每进程四请求：
natural-light cold、soft-light fresh warm、warm-light fresh warm、立即
重复warm-light。完整request wall含PNG export，不含Python startup/import
或constructor。cold是该新进程首个生成请求，不是从exec到首图，也没有
清系统文件/driver cache；源在整个窗口开始前只做一次完整SHA，之后
arm boundaries仅canonical/stat generation复验，不每次inference重扫。
主窗口的tokenizer校对放在全部timed generations之后，不预热首个
生成调用。reference只有host端四步计数，无额外GPU sync、没有encoder
wrapper或额外retained conditioning。fresh wall为本轮主要结论。

native使用匹配`build/qwen21-fast-defaults-v1-private`，513inputs与当前
源码相符；实际GPU、512 text tokens，前三请求cache miss、第四hit。
mflux只读commit`8c00dab2505a96019df9d30bc9c223bf20d733c4`、v0.19.2，
`Flux2Klein(...quantize=None,ModelConfig.flux2_klein_9b())`，未改默认
`mx.compile`、decoder或cache策略。两边MLX均0.32.0；Python依赖另存
忽略目录，不安装/改写参考仓库或覆盖原oracle环境。

| 路线 / 顺序位置 | constructor s | cold request s | fresh soft s | fresh warm s | repeated warm s |
| --- | ---: | ---: | ---: | ---: | ---: |
| native 0 | 0.126970 | 8.162579 | 8.904480 | 7.603170 | 6.969681 |
| mflux 1 | 0.369737 | 8.792367 | 10.865965 | 11.668388 | 11.742383 |
| mflux 2 | 0.361009 | 9.946851 | 10.556266 | 11.660946 | 11.899637 |
| native 3 | 0.137537 | 10.132145 | 8.838022 | 7.593342 | 7.076567 |

每路四个fresh warm原始样本池化中位：native **8.220596 s**，mflux
**11.263455 s**，1.370×，完整耗时少27.02%。不是拿第四次cache hit
当fresh分母，也不是最快单张。第二个native cold反而比第二个reference
cold慢，不宣称所有cold都领先。重复prompt中位native7.023124 /
mflux11.821010 s另列：native真正condition cache hit，而原mflux每次
仍执行prompt encoder；此差额不能全部归因于GPU kernel。

主记录保留native自然暴露的`text_encode/denoise/vae`，reference不为
拆分阶段额外插入eval。v3辅助observer窗口以callback和before-loop eval
观察`encoder_and_noise/denoise/vae_and_export`，其fresh8.212/11.041s
只用于理解阶段，不充当主表stock分母。v3/v4同engine各完整RGB全部
一致；数值不变不代表观测开销为零，两个窗口差值也不唯一归因observer。

## 内存与输出，不省略压力

四独立100ms进程树采样/重放均complete，所有实际程序exit0。native
tree peak physical30.26/30.08GB，mflux40.48/40.48GB（十进制，覆盖
load+cold+fresh+repeat+exit）。不是相同内存预算比较；native按现有策略
在conditioning后释放临时encoder权重，reference持有其模型组件，
allocator/compile缓存生命周期也不同，收益不是纯算子等精度microbenchmark。

系统compression累计增量native约5.341/4.531GB，mflux约103.873/
99.115GB；这是活动累计计数，不是当前占用。mflux两个窗口系统
swap-out13,369,344/9,502,720bytes，native为0，swap-in另列记录。
不可宣称无内存压力、设备独占或将这些系统值完全归因某个进程/框架。
它们也是不把27.02%解释成纯kernel收益的重要原因。

16张完整PNG成功。同一engine的重复请求RGB exact，正反序同一prompt
也可直接按保留图核对；跨engine不是byte-exact。已打开warm-light两图：
狐狸姿态、脸部、尾巴、树木/雪地和暖光构图近乎一致，细小纹理不同，
未观察到新增明显色块/破损。该case RGB correlation0.998518、
Gaussian11 RGB SSIM0.990202、MAE1.2859/255，全部8对cross-engine RGB
指标保存在记录。这是有限agent观察/数值检查，不是广泛感知资格或
相同noise的独立证明；不以严格latent门阻断这次性能报告。

## mlx-serve：原BF16布局不能作速度分母

官方二进制26.10.1（MLX0.32.3）实际loopback probe，未转权重或键名。
它识别为image engine并读入四个text encoder shards、399 weights，随后：

```text
MISSING WEIGHT: embed_tokens.weight
model load failed: MissingFluxWeight
error: LoadFailed
```

退出1，没有生成请求或PNG，owned server已terminal。原HF文件使用
`model.embed_tokens.weight` / `model.layers.*`；pinned release的FLUX
loader要求mflux-converted键名，linear/embedding还显式要求affine
`scales/biases`。因此本地原BF16组合的mlx-serve列标N/A（格式不匹配），
不是“比0.26s启动失败慢”，也不是整个mlx-serve不支持FLUX9B。
若后续转成Q4/Q8，必须单列精度/权重表示并做同源转换/输出验证，不能
把本轮BF16 native时间直接和量化参考组成等精度框架排名。

来源：官方release commit`02bee553f48cd3bc7d82aba0f8073820bd924738`
的`src/flux.zig`，只下载到本仓库忽略目录；它与本地参考仓库的另一
commit分开记录。binary/source SHA和实际启动错误绑定在机器证据中。

## 保留失败与复现

初次v1 guard把合法空Python `__init__.py`当成坏模型文件；没有进入
模型计时。v2两边实际图片完成，但Torch/Python启动同解释器helper，
memory verifier因unknown_child返回inconclusive，整个窗口failed。
两个原窗口保留，不追认其倍率。v3为reference绑定精确解析后的Python
executable角色，仍拒绝其它unknown children/gaps，无数值或质量门放宽。

```sh
.venv/bin/python3 tools/validation/flux9_bf16_framework_screen.py \
  --model models/FLUX.2-klein-9B \
  --library build/qwen21-fast-defaults-v1-private/libturbocider.dylib \
  --mflux ../references/mflux --python path/to/reference-python \
  --extra path/to/isolated-reference-dependencies --output outputs/new-flux9-abba
```

新output必须不存在，完整任务串行。source/config/steps/math不变；只修复
通用native benchmark helper已有的意外缩进，使它可解析/显示help。
host5项、probe host3项、report7项通过，独立portable audit逐一检查actual rows、四memory proof、
源码/库身份、完整RGB与真实mlx-serve拒绝；未跑全量make test。

[机器记录](../design/validation/flux9-bf16-framework-20261011.json)含全部
cold/fresh/repeat样本、stage、PNG身份、8对完整RGB、压力计数、未改参考
来源与失败/拒绝。模型/图片/依赖/binaries/raw logs不入Git。
多模型整体目标仍active，继续补Z的其它本地格式与适用视频边界。
