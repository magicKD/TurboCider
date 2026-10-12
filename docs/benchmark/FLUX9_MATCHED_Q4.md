# FLUX9B：同压缩payload的DiT Q4/g32完整对照

2026-10-12，M4 Max64GiB/macOS26.6.2；原9B base，512²/4步/seed42，
无LoRA。native BF16 compiled、native DiT Q4、mflux BF16/cache0、mflux
DiT Q4/cache0共8进程正反序；32PNG/8memory全部完整，四fresh/route。
这是DiT-only量化，encoder/VAE仍原BF16，不冒充全组件Q4或GGUF。

## 量化省内存，不是本组最快

| 路线 | 四fresh中位s | cold正/反s | repeat正/反s | tree peak GB（十进制） |
| --- | ---: | --- | --- | --- |
| native BF16 compiled GPU | 7.921748 | 7.565 /7.643 | 6.921 /6.732 | 30.53–30.65 |
| native DiT Q4/g32 GPU | 7.986974 | 8.777 /9.890 | 7.471 /7.494 | 18.16 |
| mflux BF16/cache0 | 7.379937 | 8.780 /8.809 | 7.313 /7.355 | 32.27–32.28 |
| mflux DiT Q4/g32/cache0 | 8.030596 | 9.281 /8.136 | 8.042 /8.016 | 19.73 |

native Q4比自己的BF16 compiled多用0.823%时间；mflux Q4比自己的
BF16多用8.817%。匹配Q4下native名义1.0055×，仅约0.5%差异，不宣传
稳定大幅优势。BF16参考仍胜出。不把旧独立cohort的7.374s或stock
g64/Q4的8.304s拼作本组分母。

Q4的实用价值是较低观测内存：native peak18.16GB而不是30.53–30.65GB，
约少40.7%；参考也从32.27降到19.73GB。四个Q4窗口compression/swap-out
均0；BF16窗口compression约5.01–6.45GB、swap-out0。不是所有窗口
无压力，也不是18GB硬内存或16GiB机器的运行资格；peak为整个进程树
load+cold+warm+exit，不含所有外部driver归因。

## 真正匹配了什么

- 153个DiT线性层按同一MLX affine Q4/g32 recipe量化，BF16 scales与
  affine biases均保留；80个其它原BF16张量不变，共539个stored tensors。
  原encoder、VAE、tokenizer通过只读目录链接共享，未量化embedding。
- mflux格式与native HF alias格式只改header，APFS CoW共用约5.674GB
  payload；整个压缩payload逐字节比较通过。原checkpoint/两个shards
  保留，没有重写原模型或参考仓库。
- stock mflux loader直接从stored planes恢复实际bits/group；153个
  loaded QuantizedLinear都实际为4/32，全部539个参数dtype/shape匹配，
  原encoder/VAE required targets398/250完整；没有monkeypatch数学。
- native使用真实U32/scales/biases与Weights::project的affine GPU分支，
  所有8dual+24single跨4步实际128个block events；Q4不错误进入dense
  compiled single graph，actual compiled count0。BF16 control实际96
  compiled singles/request。该量化candidate没有已验证ANE路线。
- 初次source-generation SHA后复用持久注册，arm边界只检查身份；
  无每次inference模型SHA、无OS cache purge。转换/布局准备单列，
  不计入warm推理，也没有扣除cold请求的实际开销。

## 输出误差与边界

warm-light native Q4相对同engine BF16：SSIM0.868303、MAE9.555/255、
correlation0.959764；量化输出狐狸正常，但局部毛发、背景细节有漂移。
匹配native/mflux Q4之间SSIM0.994068、MAE1.021/255；BF16跨engine
SSIM0.990202、MAE1.286/255。量化漂移和跨engine差异分开统计，不能
据同seed宣称noise或像素parity。原尺寸RGB/opaque alpha全部验证，
四路线的16对正反序RGB均exact；有限prompt不证明广泛感知资格。

## 证据、空间与复现

[完整便携记录](../design/validation/flux9-matched-dit-q4-current-20261012.json)
绑定514个native输入、原source/库/只读mflux全部源码、完整转换receipt、
所有raw report/32PNG/8memory与32个完整图像比较。
[保留配置](best-local-configs-20261011.json)将它列为memory candidate，
不取代BF16 GPU性能推荐。

header-derived preflight：DiT Q4 payload5,674,127,360bytes、最大单张量
临时序列化94,371,840bytes，加CoW header/slack与2GiB reserve需
7,932,760,064bytes可用；转换时检查通过。全组件从头转换的对应预算
14,686,205,952bytes当时未满足，不删除原模型/历史输出腾空间；全组件
与其它精度仍需独立补齐，不能拿这组DiT-only当全部完成。

```sh
.venv/bin/python3 -B tools/validation/flux9_matched_affine_screen.py \
  --model models/FLUX.2-klein-9B \
  --pack outputs/flux9-matched-dit-q4-g32-pack-v1-20261012 \
  --library build/flux9-hybrid-v1-private/libturbocider.dylib \
  --mflux ../references/mflux \
  --python .deps/qwen21-oracle-env/bin/python3 --extra .deps/mflux-reference-extra \
  --output outputs/new-flux9-matched-dit-q4
```

量化weights/媒体/raw不进Git。扩大benchmark仍active；更多精度、全组件
匹配、进一步GPU/ANE分工及成功外部布局不因这个完整campaign标完成。
