# FLUX9B：encoder+DiT线性层Q4/g32完整匹配

2026-10-12，M4 Max64GiB/macOS26.6.2，原9B base、512²/4步/seed42，
无LoRA。encoder252+DiT153个线性层都为同一affine Q4/g32；embedding、
norms和VAE仍原BF16，不是每个tensor都4bit，也不是stock g64 recipe。
8进程四路线正反序、32PNG/8memory完整，四fresh/route；constructor、
cold、立即repeat单列，不跨其它cohort拼最小值。

| 路线 | 四fresh中位s | cold正/反s | repeat正/反s | tree peak GB（十进制） |
| --- | ---: | --- | --- | --- |
| native BF16 compiled GPU | 7.741839 | 9.094 /8.829 | 6.721 /6.990 | 30.33–30.65 |
| native encoder+DiT Q4/g32 GPU | 8.005275 | 8.963 /8.949 | 7.483 /7.492 | 11.99 |
| mflux BF16/cache0 | 7.346223 | 8.974 /10.234 | 7.352 /7.440 | 32.25–32.40 |
| mflux encoder+DiT Q4/g32/cache0 | 8.042573 | 8.356 /8.115 | 8.081 /8.049 | 12.57–12.58 |

native Q4比自身BF16 compiled多用3.403%时间；参考Q4比自身BF16多用
9.479%。匹配Q4下native只是名义1.00466×，约0.5%差异，不宣传稳定
时延优势。纯性能仍保留BF16 GPU选择，低内存候选保留此Q4配置。

native peak11.99GB对同组BF16约少60.5–60.9%。四个Q4窗口compression/
swap-out均0；BF16窗口compression约4.96–6.29GB，swap-out0。peak覆盖
进程树load+cold+warm+exit，不是硬RAM cap或所有driver归因，不据此
宣称16GiB设备已qualified。旧[DiT-only候选](FLUX9_MATCHED_Q4.md)的
18.16GB/7.987s仍在自己cohort中，不跨轮算encoder单因素加速。

## 完整层数、同payload与失败保护

原encoder的36层全部保存，902个参数覆盖398个原learned targets加
252组scales/biases；DiT539个参数、153组planes。两个component的
native/mflux布局均逐字节验证整个压缩payload一致；实际loaded
QuantizedLinear分别153/252、全部4bit/group32，没有修改参考源码。
生成自原config的rotary_emb.inv_freq不是learned tensor，明确不保存；
其它未映射参数一律拒绝，embedding仍dense BF16。

首个转换v1在layers.28.input_layernorm.weight处失败，保留failed
receipt/已克隆DiT，不进入排名：参考的HF-key层数辅助函数看不到保存
格式layers.*，回退28，而实际为36。修复的是我们转换器的contiguous
actual-layer推导；v2重新独立完成，没有删去28–35层或绕过coverage。

复用原DiT-only的两个只读CoW files，只新增encoder权重，不删除原
模型/历史输出。全组件从头准备预算14,686,205,952bytes不满足时，
复用预检需9,012,078,592bytes（含最大1,244,659,712bytes临时embedding
与2GiB reserve）；实际available9,303,056,384bytes，通过后才转换。
新encoder5,586,511,880bytes，原DiT5,674,258,440bytes保持原hash；
logical file sizes不等于CoW独占空间或RAM。

## 输出差异

warm-light native Q4相对同engine BF16：SSIM0.838039、MAE11.543/255、
correlation0.940211；狐狸/雪林正常，但毛发、背景细节有明显量化漂移。
匹配native/mflux Q4为SSIM0.993360、MAE1.146/255；BF16跨engine仍
SSIM0.990202、MAE1.286/255。原尺寸RGB/opaque alpha核验，没有缩图；
16对正反序RGB全exact，不等于广泛感知parity或不同engine同noise。
允许较宽delta也不把这个结果称无损。该量化候选仍GPU-only，没有
已验证encoder/DiT ANE分区，不套用原BF16 manifest。

## mlx-serve的实际边界

额外用未修改的26.10.1 binary探测该派生mflux目录：memory preflight
通过、902个encoder tensors加载、H4096/layers36/heads32/8/F12288
识别正确，随后因embed_tokens.scales缺失拒绝。此匹配recipe刻意保留
dense BF16 embedding，而该pinned loader要求quantized embedding。
没有成功generation，不能拿startup耗时作分母或说整个框架不支持9B。
[兼容性与失败绑定](../design/validation/flux9-full-q4-compatibility-20261012.json)
保留raw SHA、terminal exit1、requests空数组；更兼容格式仍待补。

## 证据与复现

[完整便携记录](../design/validation/flux9-matched-full-q4-current-20261012.json)
绑定514个native source输入、原模型、两种布局/两component hash、
只读mflux0.19.2、全部raw reports/PNG/memory与32个完整RGB比较。
[保留配置](best-local-configs-20261011.json)区分性能与memory candidates。

```sh
.venv/bin/python3 -B tools/validation/flux9_matched_affine_screen.py \
  --model models/FLUX.2-klein-9B \
  --pack outputs/flux9-matched-full-q4-g32-pack-v2-20261012 \
  --library build/flux9-hybrid-v1-private/libturbocider.dylib \
  --mflux ../references/mflux \
  --python .deps/qwen21-oracle-env/bin/python3 --extra .deps/mflux-reference-extra \
  --output outputs/new-flux9-matched-full-q4
```

源初次SHA后复用generation注册；不逐次inference扫描模型。weights/
media/raw不进Git。扩大目标仍active；其它bits、进一步GPU/ANE分工、
更多尺寸与LTX成功外部框架对照不因这一完整campaign标完成。
